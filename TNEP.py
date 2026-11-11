from __future__ import annotations

import copy
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Callable, NamedTuple

from DescriptorBuilder import make_descriptor_builder
from SNES import SNES
from TNEPconfig import (TNEPconfig, resolved_dipole_cutoff, resolved_dipole_rc,
                        neighbor_readout_on, bond_rank, kernel_channels)
from nep_descriptor import (validate_nep_cfg, nep_dim, nep_raw_width, nep_slices,
                            find_q_tf, fn_and_fnp_tf, real_harmonics_tf,
                            real_harmonics_grad_tf, nep_pair_kwargs, NUM_ABC,
                            EXPERT_WEIGHTS, nep_gram_on, nep_gram_n_pairs,
                            gram_pairs, q3b_from_gram_tf)


class AnnBlock(NamedTuple):
    """One typed single-hidden-layer ANN in the SNES typed slab: weights
    W0<suffix> [T, dim_in, hidden], b0<suffix> [T, hidden], W1<suffix>
    [T, hidden, head_slots]; scale_slice = its channels of cfg.q_scale."""
    suffix: str
    dim_in: int
    hidden: int
    scale_slice: slice


@tf.function(jit_compile=True)
def _nep_radial_xla(alpha, slot, fnp, rhat, K):
    """F^r [Cc, P, 3] = (Σ_k α[c, slot_p, k] fnp[p, k]) r̂_p; k-loop so the largest
    live tensor is [Cc, P] (alpha [Cc, N·T, K] gathered by slot = i·T + t_j)."""
    acc = tf.zeros(tf.shape(slot)[:1], alpha.dtype)[tf.newaxis]          # [1, P]
    for k in range(K):                                                   # static K
        acc = acc + tf.gather(alpha[:, :, k], slot, axis=1) * fnp[tf.newaxis, :, k]
    return acc[:, :, tf.newaxis] * rhat[tf.newaxis]


@tf.function(jit_compile=True)
def _nep_angular_xla(Gs, c_a_flat, i_lin, slot, fn, fnp, rhat, Pc, dPc, n_a, K):
    """F^a [Cc, P_a, 3] = Σ_n [ gnp_n (Σ_c Gs[i,n,c] P_c) r̂ + g_n Σ_c Gs[i,n,c] ∇P_c ],
    g_n = Σ_k c_a[t_i,t_j,n,k] fn_k (gnp_n with fnp_k). Gs [Cc, N, n_a, 24] gathered
    by i_lin; c_a_flat [Cc, T·T, n_a, K] by slot = t_i·T + t_j; Pc [P_a, 24];
    dPc [P_a, 3, 24]. n-loop keeps the largest live tensor at [Cc, P_a, 24]."""
    F = tf.zeros([1, 1, 3], Gs.dtype)
    for n in range(n_a):                                                 # static n_a
        Gs_n = tf.gather(Gs[:, :, n, :], i_lin, axis=1)                  # [Cc, P_a, 24]
        GP = tf.reduce_sum(Gs_n * Pc[tf.newaxis], axis=-1)               # [Cc, P_a]
        GdP = tf.stack([tf.reduce_sum(Gs_n * dPc[tf.newaxis, :, x, :], axis=-1)
                        for x in range(3)], axis=-1)                     # [Cc, P_a, 3]
        c_n = tf.gather(c_a_flat[:, :, n, :], slot, axis=1)              # [Cc, P_a, K]
        g = tf.reduce_sum(c_n * fn[tf.newaxis], axis=-1)                 # [Cc, P_a]
        gnp = tf.reduce_sum(c_n * fnp[tf.newaxis], axis=-1)
        F = F + (gnp * GP)[:, :, tf.newaxis] * rhat[tf.newaxis] + g[:, :, tf.newaxis] * GdP
    return F


class TNEP(layers.Layer):
    """Per-type single-hidden-layer ANN predicting energy, dipole, or polarizability.

    Per atom i (type t): U_i = tanh(q_i @ W0[t] + b0[t]) · W1[t] + b1.
        W0 [num_types, dim_q, num_neurons]  in→hidden
        b0 [num_types, num_neurons]         hidden bias
        W1 [num_types, num_neurons]         hidden→scalar
        b1 ()                               global scalar bias (PES / tnep readout)
        b1 [T, S]                           neighbor readout: per-type site bias

    Modes (cfg.target_mode):
        0 PES     : E = -Σ_i U_i                                   → [1]
        1 Dipole  : tnep     μ = -Σ_ij |r_ij|² · (dU_i/dr_ij_vec)  → [3]
                    neighbor μ = Σ_i Σ_n s_i^n · c · V_i[:, n, :]  → [3]
        2 Polar.  : neighbor, rank 2 — two even bond kernels          → [6]

    Mode 2 is always the neighbour readout (bond_rank 2) on two kernel groups
    with separate site heads (channels group-major, Ck = 2C):
        α = Σ_{i<j} R_n(r_ij)·[ c⁰ (s_i^n + s_j^n) |d|²·I
                               + c¹ (u_i^n + u_j^n) (d⊗d − |d|²I/3) ]
    Isotropic and anisotropic parts differ ~100× in scale, so they must not share
    a coefficient (GPUMD's scalar + anisotropic pair). Output order
    [xx, yy, zz, xy, yz, zx].
    """

    def __init__(self,
                 cfg: TNEPconfig,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

        # NEP block: raw basis from data, contracted in-model by c_r/c_a.
        self._nep_mode = validate_nep_cfg(cfg)

        # A saved dim_q that disagrees with this build's descriptor predates a
        # descriptor change; the model must be retrained.
        from descriptor_layout import compute_dim_q, soap_dim
        _dim_q_now = int(compute_dim_q(cfg))
        if int(cfg.dim_q) != _dim_q_now:
            raise ValueError(
                f"cfg.dim_q={int(cfg.dim_q)} but this build's descriptor gives "
                f"{_dim_q_now}. The saved config predates a descriptor change "
                f"(removed learned-basis NEP extras / preprocess contraction); "
                f"the model must be retrained.")
        self.dim_q = cfg.dim_q
        # SOAP-only width; under "concat" dim_q is wider by the NEP block.
        self._soap_dim = int(soap_dim(cfg))
        self.num_types = cfg.num_types
        # Rank-2 readout: G hidden blocks of num_neurons, W1 block-diagonal
        # (_head_mask) so the isotropic and traceless sites share no hidden unit.
        self._head_groups = bond_rank(cfg) if neighbor_readout_on(cfg) else 1
        self.num_neurons = cfg.num_neurons * self._head_groups
        self._H_final = self.num_neurons
        # Any Keras activation name is accepted; _activation_grad raises at the
        # first dipole/pol pass if its derivative is not plumbed.
        self._activation_name = str(cfg.activation).lower().strip()
        self._glorot_gain = 1.0
        self.activation = tf.keras.activations.get(cfg.activation)

        # Learned radial basis (K = cfg.radial_basis_size): the builder emits raw
        # centre coefficients [.., A, S·K·k_max·2]; soap_from_raw finishes them.
        _K = getattr(cfg, "radial_basis_size", None)
        self._learn_radial = _K is not None
        if self._learn_radial:
            from descriptor_layout import (radial_basis_init,
                                              radial_basis_channel_map)
            K = int(_K)
            alpha = int(cfg.alpha_max)
            if int(cfg.descriptor_mode) != 1 or str(cfg.compress_mode) != "trivial":
                raise ValueError(
                    "cfg.radial_basis_size requires descriptor_mode=1 (the "
                    "TF/GPU builder; quippy cannot emit raw centre "
                    "coefficients) and compress_mode='trivial'.")
            if not alpha <= K <= 10:
                raise ValueError(
                    f"cfg.radial_basis_size={K} must satisfy alpha_max="
                    f"{alpha} <= K <= 10 (the poly3 overlap matrix is SPD "
                    f"only up to ~10).")
            from data import descriptor_gradients_needed
            if descriptor_gradients_needed(cfg):
                raise NotImplementedError(
                    "cfg.radial_basis_size: descriptor gradients through the "
                    "learned radial basis are not implemented — only the "
                    "neighbor readout (dipole or polarisability) "
                    "is supported.")
            self._rb_species_sum = bool(cfg.radial_basis_species_sum)
            self._rb_per_l = bool(cfg.radial_basis_per_l)
            self._rb_norm = str(cfg.radial_basis_norm)
            if self._rb_norm not in ("l2", "none", "sqrt"):
                raise ValueError(
                    f"cfg.radial_basis_norm={cfg.radial_basis_norm!r}; expected "
                    f"'l2', 'none' or 'sqrt'.")
            # O(1) scale for the raw coefficients: undo SOAP-turbo's 4π·(rcut/σ_t)².
            self._rb_coef_scale = float(
                1.0 / (4.0 * np.pi * (float(cfg.rcut_hard) / float(cfg.atom_sigma_t)) ** 2))
            cm = radial_basis_channel_map(alpha, int(cfg.l_max), cfg.num_types,
                                          species_sum=self._rb_species_sum)
            if int(cm["gather_idx"].size) != self._soap_dim:
                raise RuntimeError(
                    f"radial_basis_channel_map gives {cm['gather_idx'].size} "
                    f"channels, SOAP dim={self._soap_dim}.")
            L = int(cfg.l_max) + 1
            self._rb_alpha, self._rb_K = alpha, K
            self._rb_L, self._rb_kmax = L, L * (L + 1) // 2
            self._rb_rows = tf.constant(cm["rows"])
            self._rb_n_rows = int(cm["rows"].size)
            self._rb_n_max = int(cm["n_max"])
            self._rb_gather = tf.constant(cm["gather_idx"])
            self._rb_scale = tf.constant(cm["scale"])
            # m-multiplicity per k within each l block (2 for m > 0, else 1).
            self._rb_mw = [tf.constant(cm["Wm"][l * (l + 1) // 2:
                                              l * (l + 1) // 2 + l + 1, l]
                                       [:, tf.newaxis])
                           for l in range(L)]
            M0 = radial_basis_init(alpha, K)
            M0_shape = (cfg.num_types, cfg.num_types, alpha, K)
            if self._rb_per_l:
                M0 = M0[..., np.newaxis]
                M0_shape = M0_shape + (L,)
            self.M_radial = tf.Variable(
                np.broadcast_to(M0, M0_shape).astype(np.float32),
                trainable=False, name="M_radial")
        else:
            for f in ("radial_basis_species_sum", "radial_basis_per_l"):
                if bool(getattr(cfg, f, False)):
                    raise ValueError(
                        f"cfg.{f}=True requires cfg.radial_basis_size (the "
                        f"learned radial basis).")
            _norm = str(getattr(cfg, "radial_basis_norm", "l2"))
            if _norm not in ("l2", "none"):
                raise ValueError(
                    f"cfg.radial_basis_norm={_norm!r}: the fixed builder supports "
                    f"'l2' or 'none' ('sqrt' requires cfg.radial_basis_size).")
            if _norm == "none" and int(cfg.descriptor_mode) != 1:
                raise ValueError(
                    "cfg.radial_basis_norm='none' requires descriptor_mode=1 "
                    "(the quippy builder always L2-normalises).")
            self.M_radial = None
        # NEP contraction c[t_centre, t_neighbour, n, k]: uniform(−1, 1) like
        # GPUMD's snes.cu; non-trainable like M_radial (SNES assigns, Adam
        # tape-watches). "off": None.
        self.c_r = self.c_a = None
        self._nep_dim = 0
        self._nep_gram = False
        # "sum": ANN_2b + ANN_3b experts on the NEP contraction, added to the SOAP ANN.
        self._experts_on = self._nep_mode == "sum"
        # Models in order (main ANN first); each has its OWN species-pair table.
        self._models = ("soap", "2b", "3b") if self._experts_on else ("soap",)
        self._expert_suf = ()     # weight-name suffix per extra model
        # Per-expert amplitudes δ = (δ_soap, δ_2b, δ_3b): redundant with W1
        # rescaling but excluded from the L1/L2 penalty. Non-trainable like
        # M_radial; None (no-op) when experts off.
        self.sum_delta = None
        if self._experts_on:
            _d = np.asarray(getattr(cfg, "sum_delta", [1.0, 1.0, 1.0]),
                            dtype=np.float32).reshape(-1)
            if _d.size != 3:
                raise ValueError(
                    f"cfg.sum_delta has {_d.size} entries, expected 3 "
                    f"(δ_soap, δ_2b, δ_3b).")
            self.sum_delta = tf.Variable(_d, trainable=False, name="sum_delta")
        # tnep dipole under NEP: per-candidate NEP pair forces (_dipole_nep_pairs).
        self._nep_pair_forces_on = (self._nep_mode in ("concat", "replace")
                                    and cfg.target_mode == 1
                                    and str(cfg.dipole_readout) == "tnep")
        if self._nep_mode != "off":
            self._nep_dim = int(nep_dim(cfg))
            self._nep_raw_width = int(nep_raw_width(cfg))
            # Data-static 3-body Gram at the tail of nep_raw: one GEMM per centre
            # type instead of a [C, N, n+1, 24] s tensor (nep_gram_on).
            self._nep_gram = bool(nep_gram_on(cfg))
            if self._nep_gram:
                _p, _q, _ = gram_pairs(cfg)
                self._gram_p = tf.constant(_p.astype(np.int32))
                self._gram_q = tf.constant(_q.astype(np.int32))
                self._gram_n_pairs = int(nep_gram_n_pairs(cfg))
            T = int(cfg.num_types)
            rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
            self.c_r = tf.Variable(rng.uniform(-1.0, 1.0, (
                T, T, cfg.nep_n_max_radial + 1, cfg.nep_basis_size_radial + 1)
            ).astype(np.float32), trainable=False, name="nep_c_r")
            self.c_a = tf.Variable(rng.uniform(-1.0, 1.0, (
                T, T, cfg.nep_n_max_angular + 1, cfg.nep_basis_size_angular + 1)
            ).astype(np.float32), trainable=False, name="nep_c_a")
        # q_scale covers the main input (dim_q) then, under "sum", the expert
        # input ([q_soap ‖ q_nep], gen0_ann_inputs): one slice per typed ANN.
        self._sl_main = slice(0, int(self.dim_q))
        self._scale_width = int(self.dim_q)
        self._expert_in = ()      # (input slice of q_nep, slice of q_scale) per expert
        if self._experts_on:
            nsl, key = nep_slices(cfg), {"2b": "radial", "3b": "angular"}
            ds = self._scale_width
            extra = self._models[1:]
            self._expert_suf = tuple({"2b": "_r", "3b": "_a"}[m] for m in extra)
            self._expert_in = tuple(
                (nsl[key[m]], slice(ds + nsl[key[m]].start, ds + nsl[key[m]].stop)) for m in extra)
            self._scale_width += self._nep_dim
        # δ slot (δ_soap, δ_2b, δ_3b) of each model, main first.
        self._delta_idx = tuple(("soap", "2b", "3b").index(m) for m in self._models)
        # descriptor_scale: m folded into W0 (_scale_fold); stored weights keep
        # their plain init and act on m∘(q − μ), GPUMD's convention. Do NOT
        # divide the init by m: that saturates tanh and inflates L1/L2 by 1/m.
        self._q_scale = None
        if str(getattr(cfg, "descriptor_scale", "off")) != "off":
            if str(cfg.descriptor_scale) not in ("block", "channel", "minmax"):
                raise ValueError(
                    f"cfg.descriptor_scale={cfg.descriptor_scale!r}; expected "
                    f"'off', 'block', 'channel' or 'minmax'.")
            if cfg.q_scale is None:
                raise ValueError(
                    "cfg.descriptor_scale is on but cfg.q_scale is None — the "
                    "training driver computes it from the training set "
                    "(compute_q_scale) before building the model.")
            m = np.asarray(cfg.q_scale, dtype=np.float32)
            if m.shape != (cfg.num_types, self._scale_width):
                raise ValueError(
                    f"cfg.q_scale has shape {m.shape}, expected "
                    f"({cfg.num_types}, {self._scale_width}) — one row per "
                    f"centre species; recompute for this descriptor.")
            self._q_scale = tf.constant(m)
            if cfg.q_shift is None:
                raise ValueError("cfg.q_shift is None — computed with q_scale.")
            mu = np.asarray(cfg.q_shift, dtype=np.float32)
            if mu.shape != m.shape:
                raise ValueError(f"cfg.q_shift has shape {mu.shape}, expected {m.shape}.")
            # b0 fold uses (m∘μ); stored once.
            self._q_shift_m = tf.constant(m * mu)
        self.builder = make_descriptor_builder(cfg)

        # W0 : [num_types, dim_q, num_neurons] — input-to-hidden weights per type
        self.W0 = self.add_weight(
            name="W0",
            shape=(cfg.num_types, cfg.dim_q, self.num_neurons),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b0 : [num_types, num_neurons] — hidden bias per type
        self.b0 = self.add_weight(
            name="b0",
            shape=(cfg.num_types, self.num_neurons),
            initializer="zeros",
            trainable=True,
        )

        # Neighbor readout is decided before W1 so the head can carry its channel axis.
        _readout = str(cfg.dipole_readout)
        if cfg.target_mode == 2 and _readout != "neighbor":
            raise ValueError(
                f"target_mode=2 is the rank-2 neighbour readout; the "
                f"descriptor-gradient (dual ANN) polarisability form was "
                f"removed. Set dipole_readout='neighbor' (got {_readout!r}).")
        self._neighbor_readout = neighbor_readout_on(cfg)
        # Kernel groups: 1 → d [3]; 2 → Voigt [6], isotropic + traceless (_bond_geometry).
        self._bond_rank = bond_rank(cfg)
        self._dipole_channels = int(cfg.dipole_channels)      # radial channels C
        # Long-range field channel (mode 1): one extra staged channel after the
        # radial groups, with its own slot under the scalar head.
        self._field_channel = (bool(getattr(cfg, "dipole_field", False))
                               and cfg.target_mode == 1 and self._neighbor_readout)
        # Channel axis of V / c_pair / vector head: C per group, group-major, then field.
        self._kernel_channels = kernel_channels(cfg)
        self._dipole_radial = str(cfg.dipole_radial)
        self._dipole_zeta = float(cfg.dipole_zeta)
        self._dipole_cutoff = resolved_dipole_cutoff(cfg)
        _sp = cfg.dipole_species_pairs
        cfg.dipole_species_pairs = {False: "off", True: "symmetric"}.get(_sp, _sp)
        self._species_pairs = str(cfg.dipole_species_pairs)
        # THE learnability flag: every save/load/SNES-tail site keys on this.
        self._c_learnable = self._species_pairs != "off"
        # dipole_head: "vector" one slot per channel; "scalar" one per group (_split_head).
        self._dipole_head = str(getattr(cfg, "dipole_head", "vector"))
        if self._dipole_head not in ("vector", "scalar"):
            raise ValueError(
                f"cfg.dipole_head must be 'vector' or 'scalar', "
                f"got {cfg.dipole_head!r}.")
        # THE source for W1's last axis (SNES/model_io read it back): Ck under
        # the vector head, one slot per kernel group under "scalar", else 1.
        if not self._neighbor_readout:
            self._head_slots = 1
        elif self._dipole_head == "vector":
            self._head_slots = self._kernel_channels
        else:
            self._head_slots = self._bond_rank + int(self._field_channel)
        if (self._neighbor_readout and self._dipole_head == "scalar"
                and self._species_pairs == "off"):
            warnings.warn(
                "dipole_head='scalar' with dipole_species_pairs='off': every "
                "channel carries the same site value with c ≡ 1, so the C "
                "radial channels collapse to Σ_n R_n — set "
                "dipole_species_pairs='symmetric' to learn a radial profile.")
        if self._neighbor_readout:
            if self._dipole_channels < 1:
                raise ValueError(
                    f"cfg.dipole_channels must be ≥ 1, got "
                    f"{self._dipole_channels}.")
            if self._dipole_radial not in ("chebyshev", "chebyshev_warped",
                                           "gaussian", "bessel", "screened"):
                raise ValueError(
                    f"cfg.dipole_radial must be 'chebyshev', "
                    f"'chebyshev_warped', 'gaussian', 'bessel' or 'screened', "
                    f"got {self._dipole_radial!r}.")
            if self._dipole_cutoff not in ("cosine2", "cosine"):
                raise ValueError(
                    f"cfg.dipole_cutoff must be 'cosine2', 'cosine' or None "
                    f"(None = 'cosine2'), got {cfg.dipole_cutoff!r}.")
            if self._dipole_radial == "screened" and self._dipole_channels != 1:
                # Raise, don't force C=1: data.py builds V from cfg before the model exists.
                raise ValueError(
                    f"cfg.dipole_radial='screened' is single-channel — set "
                    f"dipole_channels=1 (got {self._dipole_channels}).")
            if self._species_pairs not in ("off", "symmetric", "ordered"):
                raise ValueError(
                    f"cfg.dipole_species_pairs must be 'off', 'symmetric' or "
                    f"'ordered' (legacy bool accepted), got {_sp!r}.")

        # W1 : [T, H_final] hidden→scalar; [T, H_final, S] under the neighbor readout.
        self.W1 = self.add_weight(
            name="W1",
            shape=((cfg.num_types, self._H_final, self._head_slots)
                   if self._neighbor_readout
                   else (cfg.num_types, self._H_final)),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Block-diagonal W1 mask [H_final, S]: slot k reads hidden block k // (S/G).
        # ponytail: masked genes stay in the SNES genome (T·H·G·(G−1)); store
        # W1 per block if that ever matters.
        self._head_mask = None
        if self._head_groups > 1:
            G, S, Hb = self._head_groups, self._head_slots, cfg.num_neurons
            grp_h = np.arange(self._H_final) // Hb
            grp_s = np.arange(S) // (S // G)
            self._head_mask = tf.constant((grp_h[:, None] == grp_s[None, :]).astype(np.float32))

        # b1 : () global scalar bias; neighbor readout: [T, S] per-type site bias
        # (a global one cancels in s_i − s_j; per-type is the fixed-charge part).
        self.b1 = self.add_weight(
            name="b1",
            shape=((cfg.num_types, self._head_slots)
                   if self._neighbor_readout else ()),
            initializer="zeros",
            trainable=True,
        )

        # c_pair : packed per-(channel, species pair) coefficients, [C, T(T+1)/2]
        # unordered or [C, T²] ordered; learnable (init 1) when _c_learnable,
        # else frozen c ≡ 1. _pair_idx maps (t, t') → slot; _c_full unpacks.
        if self._neighbor_readout:
            T_ = cfg.num_types
            if self._species_pairs == "ordered":
                # t·T + t': an asymmetric Ω breaks the (s_i − s_j) antisymmetry.
                pair_idx = np.arange(T_ * T_, dtype=np.int32).reshape(T_, T_)
            else:
                pair_idx = np.zeros((T_, T_), dtype=np.int32)
                p = 0
                for t in range(T_):
                    for t2 in range(t, T_):
                        pair_idx[t, t2] = pair_idx[t2, t] = p
                        p += 1
            self._pair_idx = tf.constant(pair_idx)
            # One table per model on the pair axis: model k owns columns
            # [k·n_pairs, (k+1)·n_pairs) (_c_full).
            self._n_pairs = int(pair_idx.max()) + 1
            c_shape = (self._kernel_channels, self._n_pairs * len(self._models))
            if self._c_learnable:
                self.c_pair = self.add_weight(
                    name="c_pair", shape=c_shape,
                    initializer="ones", trainable=True)
            else:
                self.c_pair = tf.ones(c_shape, dtype=tf.float32)

        if cfg.target_mode == 1 or self._neighbor_readout:
            _N = int(cfg.dipole_rij_power)
            if _N < 0:
                raise ValueError(
                    f"cfg.dipole_rij_power must be an integer ≥ 0, got {_N}. "
                    f"0 = self-pair only (μ = -Σ_i de_dq[i] · ∂q_i/∂R_i), "
                    f"1 = |r|·F (first radial moment), "
                    f"2 = |r|²·F (Xu et al. JCTC 2024 default), "
                    f"≥3 = higher radial moments."
                )
            if _readout not in ("tnep", "neighbor"):
                raise ValueError(
                    f"cfg.dipole_readout must be 'tnep' or 'neighbor', "
                    f"got {_readout!r}.")
            if self._neighbor_readout:
                self._rc = resolved_dipole_rc(cfg)      # readout cutoff (staged V is image-exact)
                if self._dipole_radial == "gaussian":
                    # μ_n = C interior points of linspace(0, r_c, C+2); σ = scale·r_c/(C+1).
                    _C = self._dipole_channels
                    self._gauss_mu = np.linspace(0.0, self._rc, _C + 2)[1:-1]
                    self._gauss_sigma = (float(cfg.dipole_gauss_sigma_scale)
                                         * self._rc / (_C + 1))
                if float(getattr(cfg, "charge_aux_weight", 0) or 0) > 0:
                    raise ValueError(
                        "charge_aux_weight > 0 is incompatible with "
                        "dipole_readout='neighbor': the site scalar s is an "
                        "electronegativity-like potential, not a charge.")

        # Typed ANN blocks in SNES order: SOAP ANN, then under "sum" the 2-body
        # (radial block) and 3-body (angular block) experts whose heads add to
        # the SOAP head. Expert W1 starts at zero so a fresh model equals the
        # SOAP-only one; b1 stays the one global bias.
        self._ann_blocks = [AnnBlock("", int(cfg.dim_q), int(self.num_neurons), self._sl_main)]
        for suf, (sl_in, sl_m) in zip(self._expert_suf, self._expert_in):
            d_in, He = sl_in.stop - sl_in.start, int(cfg.nep_num_neurons)
            self._ann_blocks.append(AnnBlock(suf, d_in, He, sl_m))
            for name, shape, init in (("W0", (cfg.num_types, d_in, He), "glorot_uniform"),
                                      ("b0", (cfg.num_types, He), "zeros"),
                                      ("W1", (cfg.num_types, He, self._head_slots), "zeros")):
                setattr(self, name + suf, self.add_weight(
                    name=name + suf, shape=shape, initializer=init, trainable=True))

        # Optional descriptor mixing (GPUMD c_nk analog): one padded square block
        # per (unordered species pair, l); placement matrices P ∈ R^{α×Q} map
        # active features to their non-contiguous q-indices (_U_full).
        self.descriptor_mixing = bool(getattr(cfg, "descriptor_mixing", False))
        self.descriptor_mixing_per_type = bool(
            getattr(cfg, "descriptor_mixing_per_type", False))
        if self.descriptor_mixing:
            from descriptor_layout import descriptor_block_layout
            self._mix_layout = descriptor_block_layout(cfg)
            self._mix_pair_keys = self._mix_layout["pair_keys"]
            self._mix_num_pairs = len(self._mix_pair_keys)
            self._mix_Q = int(self._mix_layout["dim_q"])
            T = cfg.num_types
            self._mix_alpha_per_pair = [
                int(self._mix_layout["alpha_eff_per_pair"][k])
                for k in self._mix_pair_keys]
            self._mix_max_alpha = int(self._mix_layout["max_alpha_eff"])
            self._mix_L = int(cfg.l_max) + 1
            # Per-(pair, l) placement matrices P_{p,l} ∈ R^{α_p × Q}.
            mix_P_ln: list[list[tf.Tensor]] = []
            for k in self._mix_pair_keys:
                per_pair: list[tf.Tensor] = []
                for l in range(self._mix_L):
                    qidx = self._mix_layout["pair_ln_index"][k][l]
                    a = qidx.size
                    P = np.zeros((a, self._mix_Q), dtype=np.float32)
                    P[np.arange(a), qidx] = 1.0
                    per_pair.append(tf.constant(P))
                mix_P_ln.append(per_pair)
            self._mix_P_ln = mix_P_ln
            # Stacked projector [num_pairs·L, α, Q] for the uniform-α fast path
            # of _U_full (one einsum instead of num_pairs×L launches).
            if len(set(self._mix_alpha_per_pair)) == 1:
                alpha = self._mix_alpha_per_pair[0]
                PL = self._mix_num_pairs * self._mix_L
                P_stack = np.zeros((PL, alpha, self._mix_Q),
                                   dtype=np.float32)
                for p_idx, k in enumerate(self._mix_pair_keys):
                    for l in range(self._mix_L):
                        qidx = self._mix_layout["pair_ln_index"][k][l]
                        row = p_idx * self._mix_L + l
                        P_stack[row, np.arange(alpha), qidx] = 1.0
                self._mix_P_ln_stack = tf.constant(P_stack)
                self._mix_P_ln_uniform_alpha = alpha
            else:
                self._mix_P_ln_stack = None
                self._mix_P_ln_uniform_alpha = None
            # U_pair [(T,) num_pairs, L, max_α, max_α], padding stays zero.
            if self.descriptor_mixing_per_type:
                shape = (T, self._mix_num_pairs, self._mix_L,
                         self._mix_max_alpha, self._mix_max_alpha)
            else:
                shape = (self._mix_num_pairs, self._mix_L,
                         self._mix_max_alpha, self._mix_max_alpha)
            self.U_pair = self.add_weight(
                name="U_pair",
                shape=shape,
                initializer="zeros",
                trainable=True,
            )
        else:
            self.U_pair = None

        self.optimizer = SNES(self)

    # ── descriptor mixing ──

    def _U_full(self, U_pair: tf.Tensor | None = None) -> tf.Tensor:
        """U_full = I_Q + Σ_p Σ_l P_{p,l}ᵀ · V_{p,l} · P_{p,l} — block-diagonal
        mixing matrix [..., Q, Q] from V [..., (T,) num_pairs, L, max_α, max_α];
        leading dims (candidate / per-type) broadcast. V = 0 ⇒ identity.
        """
        V = self.U_pair if U_pair is None else U_pair
        # Fast path, uniform α: flatten (num_pairs, L) → PL, one batched einsum.
        if self._mix_P_ln_stack is not None:
            alpha = self._mix_P_ln_uniform_alpha
            PL = self._mix_num_pairs * self._mix_L
            V_active = V[..., :alpha, :alpha]                # drop padding rows/cols
            new_shape = tf.concat(
                [tf.shape(V_active)[:-4], [PL, alpha, alpha]], axis=0)
            V_flat = tf.reshape(V_active, new_shape)
            V_full = tf.einsum(
                'pji,...pjk,pkm->...im',
                self._mix_P_ln_stack, V_flat, self._mix_P_ln_stack)
            return tf.eye(self._mix_Q, dtype=V_full.dtype) + V_full
        parts = []
        for p_idx, alpha_p in enumerate(self._mix_alpha_per_pair):
            for l in range(self._mix_L):
                V_block = V[..., p_idx, l, :alpha_p, :alpha_p]            # [..., α_p, α_p]
                P_pl = self._mix_P_ln[p_idx][l]                            # [α_p, Q]
                placed = tf.einsum('ji,...jk,kl->...il', P_pl, V_block, P_pl)
                parts.append(placed)
        V_full = tf.add_n(parts)
        return tf.eye(self._mix_Q, dtype=V_full.dtype) + V_full

    def _W0_eff(self, W0: tf.Tensor,
                U_pair: tf.Tensor | None = None) -> tf.Tensor:
        """W0_eff = U_fullᵀ·diag(m)·W0 [(C,) T, Q, H]: descriptor mixing and the
        descriptor_scale m absorbed into the weights, so forward/backward/dipole
        use raw descriptors and grad_values unchanged. U_pair [(C,) (T,)
        num_pairs, L, α, α]. Every entry point must go through this fold.
        """
        W0, _ = self._scale_fold(W0, None)                       # [..., Q, H]
        if self.descriptor_mixing:
            U_full = self._U_full(U_pair)
            if self.descriptor_mixing_per_type:
                W0 = tf.einsum('...tqp,...tqh->...tph', U_full, W0)
            else:
                W0 = tf.einsum('...qp,...tqh->...tph', U_full, W0)
        return W0

    def _b0_eff(self, b0: tf.Tensor, W0_stored: tf.Tensor) -> tf.Tensor:
        """b0_eff = b0 − (m∘μ)·W0_stored (W0 BEFORE _W0_eff). Identity when
        scaling is off; call it wherever _W0_eff is called."""
        return self._scale_fold(W0_stored, b0)[1]

    def _scale_fold(self, W0: tf.Tensor, b0: tf.Tensor | None,
                    sl: slice | None = None) -> tuple:
        """descriptor_scale fold for the ANN reading scaler slice `sl` (None →
        main): W0_eff = m∘W0, b0_eff = b0 − (m∘μ)·W0_stored. Identity when off.
        The ONE fold site, shared by the SOAP ANN and both experts."""
        if self._q_scale is None:
            return W0, b0
        sl = self._sl_main if sl is None else sl
        W0_eff = W0 * self._q_scale[:, sl, tf.newaxis]
        b0_eff = (None if b0 is None
                  else b0 - tf.einsum('tq,...tqh->...th', self._q_shift_m[:, sl], W0))
        return W0_eff, b0_eff

    # ── learned radial basis ──

    @property
    def descriptor_input_width(self) -> int:
        """Width of the `descriptors` slot (descriptor_layout); not cfg.dim_q."""
        from descriptor_layout import descriptor_input_width
        return descriptor_input_width(self.cfg)

    def soap_from_raw(self, raw: tf.Tensor, Z: tf.Tensor,
                      M: tf.Tensor | None = None) -> tf.Tensor:
        """SOAP-turbo descriptor from raw centre coefficients with the learned
        radial basis:
            c[a, (s,n), k] = Σ_κ M[t_a, s, n, κ] · c_raw[a, s, κ, k]
            q[a, (n,n',l)] = Σ_m mult · Re(c_{n l m} conj c_{n' l m})   (trivial
            compression), then per-atom norm per cfg.radial_basis_norm ("l2" with
            SOAP-turbo's ‖q‖ < 1e-5 guard; "none" × scale²; "sqrt" sign·√|q| × scale).
        species_sum sums s into α shared channels (dim_q = α(α+1)/2·L); per_l
        gives each l block its own M[..., l].
            raw : [..., A, S·K·k_max·2]  (species, κ, k, re/im)
            M   : [T, S, α, K(, L)] (None → self.M_radial) or per-candidate [C, ...]
        Returns [..., A, Q], or [C, ..., A, Q] for per-candidate M.
        """
        M = self.M_radial if M is None else M
        S, K, kmax, alpha, L = (self.num_types, self._rb_K, self._rb_kmax,
                                self._rb_alpha, self._rb_L)
        per_cand = len(M.shape) == (6 if self._rb_per_l else 5)
        lead = tf.shape(raw)[:-1]
        N = tf.reduce_prod(lead)
        r = tf.reshape(raw, [N, S, K, kmax * 2])                 # [N, S, K, 2k]
        Zf = tf.reshape(Z, [N])
        # Partition atoms by centre type and contract each with its own M[t].
        # Never gather M per atom: that [N, S, α, K(, L)] copy is hundreds of MB.
        T = self.num_types
        order = tf.argsort(Zf)
        inv = tf.math.invert_permutation(order)
        counts = tf.math.bincount(tf.gather(Zf, order), minlength=T, maxlength=T)
        parts = tf.split(tf.gather(r, order), counts, num=T, axis=0)
        atom_axis = 1 if per_cand else 0

        def _contract(M_of_t, k_slice):
            """c[(C,) N, n_max, k_len, 2] = Σ_κ M[t_a, s, n, κ] · r[a, s, κ, k_slice],
            atoms back in input order."""
            outs = []
            for t in range(T):
                M_t = M_of_t(t)                                   # [(C,) S, α, K]
                r_t = parts[t][..., k_slice]                      # [N_t, S, K, kk]
                if self._rb_species_sum:
                    outs.append(tf.einsum('csak,nskj->cnaj', M_t, r_t) if per_cand
                                else tf.einsum('sak,nskj->naj', M_t, r_t))
                else:
                    outs.append(tf.einsum('csak,nskj->cnsaj', M_t, r_t) if per_cand
                                else tf.einsum('sak,nskj->nsaj', M_t, r_t))
            c = tf.gather(tf.concat(outs, axis=atom_axis), inv, axis=atom_axis)
            return tf.reshape(c, [-1, N, self._rb_n_max, tf.shape(r_t)[-1] // 2, 2])

        if not self._rb_per_l:
            cc = _contract(lambda t: M[:, t] if per_cand else M[t],
                           slice(None))                           # [C, N, n_max, k, 2]
            piv = tf.gather(cc, self._rb_rows, axis=2)            # [C, N, R, k, 2]
        G_l = []
        for l in range(L):
            k0 = l * (l + 1) // 2
            if self._rb_per_l:
                c_l = _contract(lambda t, l=l: (M[:, t, ..., l] if per_cand
                                                else M[t, ..., l]),
                                slice(2 * k0, 2 * (k0 + l + 1)))
                p_l = tf.gather(c_l, self._rb_rows, axis=2) * self._rb_mw[l]
            else:
                p_l = piv[:, :, :, k0:k0 + l + 1, :] * self._rb_mw[l]
                c_l = cc[:, :, :, k0:k0 + l + 1, :]
            G_l.append(tf.einsum('cnpkr,cnmkr->cnpm', p_l, c_l))   # [C, N, R, n_max]
        G = tf.stack(G_l, axis=-1)                                # [C, N, R, n_max, L]
        G = tf.reshape(G, [-1, N, self._rb_n_rows * self._rb_n_max * L])
        q = tf.gather(G, self._rb_gather, axis=-1) * self._rb_scale
        sc = self._rb_coef_scale
        if self._rb_norm == "l2":
            norm = tf.norm(q, axis=-1, keepdims=True)
            q = q / tf.where(norm < 1e-5, tf.ones_like(norm), norm)
        elif self._rb_norm == "none":
            q = q * (sc * sc)          # spectrum is quadratic in the coefficients
        else:
            q = tf.math.sign(q) * tf.sqrt(tf.abs(q) + 1e-12) * sc
        out_shape = tf.concat([lead, [self._soap_dim]], axis=0)
        if per_cand:
            return tf.reshape(q, tf.concat([[-1], out_shape], axis=0))
        return tf.reshape(q[0], out_shape)

    def nep_from_raw(self, raw: tf.Tensor, Z: tf.Tensor,
                     c_r: tf.Tensor | None = None,
                     c_a: tf.Tensor | None = None,
                     return_s: bool = False) -> tf.Tensor:
        """GPUMD NEP descriptor from the raw per-atom basis (nep_descriptor.py):
            q_r[a, n]  = Σ_{t',k} c_r[t_a, t', n, k] · raw_r[a, t', k]
            s[a, n, p] = Σ_{t',k} c_a[t_a, t', n, k] · raw_a[a, t', k, p]
            q_a = find_q(s)  → [q_r ‖ angular l-major (ln = l·(n_max_a+1)+n) ‖ 4b ‖ 5b]
            raw : [..., A, W_raw]  flatten([raw_r [T, K_r+1] ‖ raw_a [T, K_a+1, 24]])
            c_r, c_a : [T, T, n+1, K+1] (None → self.c_r / self.c_a) or per-candidate [C, ...]
        Returns [..., A, dim_nep] (leading C for per-candidate c); with return_s
        also s [(C,) ..., A, n_a+1, 24] (for the tnep pair forces).
        """
        c_r = self.c_r if c_r is None else c_r
        c_a = self.c_a if c_a is None else c_a
        cfg = self.cfg
        T, Kr, Ka = self.num_types, cfg.nep_basis_size_radial + 1, cfg.nep_basis_size_angular + 1
        per_cand = len(c_r.shape) == 5
        lead = tf.shape(raw)[:-1]
        N = tf.reduce_prod(lead)
        raw = tf.reshape(raw, [N, self._nep_raw_width])
        W_r = T * Kr
        Zf = tf.reshape(Z, [N])
        axis = 1 if per_cand else 0
        raw_r = tf.reshape(raw[:, :W_r], [N, T, Kr])
        c_r_g = tf.gather(c_r, Zf, axis=axis)                            # [(C,) N, T, n, k]
        if per_cand:
            q_r = tf.einsum("atk,catnk->can", raw_r, c_r_g)              # [C, N, n]
        else:
            q_r = tf.einsum("atk,atnk->an", raw_r, c_r_g)
        if self._nep_gram:
            # Angular block of nep_raw IS the 3-body Gram: one GEMM per centre type.
            if return_s:
                raise ValueError(
                    "return_s with the NEP Gram layout: s is not recoverable "
                    "from the Gram (nep_gram_on excludes the tnep pair-force "
                    "path, which is the only consumer of s).")
            s = None
            q_a = q3b_from_gram_tf(raw[:, W_r:], c_a, Zf, T, self._gram_p,
                                   self._gram_q, self._gram_n_pairs)     # [(C,) N, L, n]
        else:
            raw_a = tf.reshape(raw[:, W_r:], [N, T, Ka, NUM_ABC])
            c_a_g = tf.gather(c_a, Zf, axis=axis)
            if per_cand:
                s = tf.einsum("atkp,catnk->canp", raw_a, c_a_g)          # [C, N, n, 24]
            else:
                s = tf.einsum("atkp,atnk->anp", raw_a, c_a_g)
            q_a = find_q_tf(s, cfg.nep_four_body, cfg.nep_five_body)     # [(C,) N, n, L]
            q_a = tf.linalg.matrix_transpose(q_a)                        # → l-major
        q_a = tf.reshape(q_a, tf.concat([tf.shape(q_a)[:-2], [-1]], 0))
        q = tf.concat([q_r, q_a], -1)
        out_shape = tf.concat([lead, [self._nep_dim]], 0)
        if per_cand:
            out_shape = tf.concat([tf.shape(c_r)[:1], out_shape], 0)
        q = tf.reshape(q, out_shape)
        if not return_s:
            return q
        return q, tf.reshape(s, tf.concat([out_shape[:-1], tf.shape(s)[-2:]], 0))

    def _ann_inputs(self, descriptors: tf.Tensor, nep_raw: tf.Tensor | None,
                    Z: tf.Tensor, c_r: tf.Tensor | None = None,
                    c_a: tf.Tensor | None = None):
        """(inputs [..., A, dim_q], s): the one place that knows cfg.nep_descriptor —
        "off" → descriptors; "concat" → [descriptors ‖ nep]; "replace" → nep.
        s (tnep pair forces) is None unless the NEP block is formed for them."""
        if self._nep_mode == "off":
            return descriptors, None
        if nep_raw is None:
            raise ValueError(
                f"nep_raw missing: cfg.nep_descriptor={self._nep_mode!r} needs "
                f"the raw expert input [.., W_raw] (the NEP basis) at every forward site.")
        if self._experts_on:              # experts read nep_raw in _expert_outputs
            return descriptors, None
        # s only for the tnep pair forces: elsewhere it forces the [C, N, n, 24] slow path.
        if self._nep_pair_forces_on:
            q_nep, s = self.nep_from_raw(nep_raw, Z, c_r, c_a, return_s=True)
        else:
            q_nep, s = self.nep_from_raw(nep_raw, Z, c_r, c_a), None
        if self._nep_mode == "replace":
            return q_nep, s
        # Either side may carry a candidate axis: broadcast the other to it.
        if len(q_nep.shape) > len(descriptors.shape):
            descriptors = tf.broadcast_to(
                descriptors[tf.newaxis],
                tf.concat([tf.shape(q_nep)[:1], tf.shape(descriptors)], 0))
        elif len(q_nep.shape) < len(descriptors.shape):
            q_nep = tf.broadcast_to(
                q_nep[tf.newaxis],
                tf.concat([tf.shape(descriptors)[:1], tf.shape(q_nep)], 0))
        return tf.concat([descriptors, q_nep], -1), s

    @property
    def expert_keys(self) -> tuple:
        """Weight names of the extra models' ANNs, in SNES / Adam / save order
        (a subset of EXPERT_WEIGHTS: only the selected models exist)."""
        return tuple(n + suf for suf in self._expert_suf for n in ("W0", "b0", "W1"))

    @property
    def expert_weights(self) -> tuple:
        """The extra models' tensors in expert_keys order."""
        return tuple(getattr(self, k) for k in self.expert_keys)

    def _expert_ann(self, q: tf.Tensor, Z: tf.Tensor, atom_mask: tf.Tensor,
                    W0: tf.Tensor, b0: tf.Tensor, W1: tf.Tensor,
                    sl_in: slice, sl_m: slice) -> tf.Tensor:
        """One summed expert: activation(q[.., sl_in]·W0[t] + b0[t])·W1[t] →
        [(C,) ..., A, head_slots]. STORED weights [(C,) T, ...]: the scale fold
        on slice sl_m happens here (no mixing reaches the experts). Per-candidate
        q needs per-candidate experts with the same C (einsum, no broadcast)."""
        W0, b0 = self._scale_fold(W0, b0, sl_m)
        x = q[..., sl_in]
        cand = len(W0.shape) == 4
        if not cand:
            W0, b0, W1 = W0[tf.newaxis], b0[tf.newaxis], W1[tf.newaxis]
        per_cand_q = len(x.shape) == len(Z.shape) + 2   # x [C, ...] when c_r is per-candidate
        if per_cand_q:
            tf.debugging.assert_equal(
                tf.shape(x)[0], tf.shape(W0)[0],
                message="per-candidate c_r/c_a need per-candidate expert weights (same C)")
        eq = 'c...ad,cdh->c...ah' if per_cand_q else '...ad,cdh->c...ah'
        z = tf.add_n([
            tf.einsum(eq, x, W0[:, t])
            * tf.cast(tf.equal(Z, t), x.dtype)[..., tf.newaxis]
            for t in range(self.num_types)])                    # [C, ..., A, H]
        h = self.activation(z + tf.gather(b0, Z, axis=1)) * atom_mask[..., tf.newaxis]
        out = tf.einsum('c...ah,c...ahk->c...ak', h, tf.gather(W1, Z, axis=1))
        return out if cand else out[0]

    def _delta_scale(self, d: tf.Tensor | None, i: int, x: tf.Tensor) -> tf.Tensor:
        """x · δ_i. d: [3] (shared) or [C, 3] per-candidate (x leads with C)."""
        if d is None:
            return x
        if len(d.shape) == 1:
            return x * d[i]
        return x * tf.reshape(d[:, i], [-1] + [1] * (len(x.shape) - 1))

    def _expert_outputs(self, nep_raw: tf.Tensor, Z: tf.Tensor, atom_mask: tf.Tensor,
                        c_r: tf.Tensor | None = None, c_a: tf.Tensor | None = None,
                        experts: tuple | None = None,
                        sum_delta: tf.Tensor | None = None) -> list:
        """Extra models' heads δ_k·ANN_k(q_k), one [(C,) ..., A, head_slots] each.
        experts: STORED tensors in expert_keys order (optional leading C);
        sum_delta [3] or [C, 3]; None → the model's variables."""
        q = self.nep_from_raw(nep_raw, Z, c_r, c_a)
        w = self.expert_weights if experts is None else tuple(experts)
        d = self.sum_delta if sum_delta is None else sum_delta
        return [self._delta_scale(d, self._delta_idx[i + 1],
                                  self._expert_ann(q, Z, atom_mask, *w[3 * i:3 * i + 3], *sl))
                for i, sl in enumerate(self._expert_in)]

    def _model_outs(self, out: tf.Tensor, nep_raw: tf.Tensor, Z: tf.Tensor,
                    atom_mask: tf.Tensor, c_r: tf.Tensor | None = None,
                    c_a: tf.Tensor | None = None, experts: tuple | None = None,
                    sum_delta: tf.Tensor | None = None) -> list:
        """Per-model head outputs [main, *extras] for _site_coeff."""
        if not self._experts_on:
            return [out]
        d = self.sum_delta if sum_delta is None else sum_delta
        return ([self._delta_scale(d, self._delta_idx[0], out)]
                + self._expert_outputs(nep_raw, Z, atom_mask, c_r, c_a, experts, d))

    def _site_bias(self, out: tf.Tensor, b1: tf.Tensor, Z: tf.Tensor,
                   atom_mask: tf.Tensor) -> tf.Tensor:
        """out[.., a, :] += b1[t_a] (b1 [(C,) T, S]; padded atoms stay 0)."""
        b = (tf.gather(b1, Z, axis=1) if len(b1.shape) == 3   # [C, B, A, S]
             else tf.gather(b1, Z))                            # [..., A, S]
        return out + b * atom_mask[..., tf.newaxis]

    def _site_coeff(self, outs: list, c_pair: tf.Tensor, Z: tf.Tensor) -> tf.Tensor:
        """G[.., a, n, t'] = Σ_k s_k[.., a, n]·c_k[n, t_a, t'] (each model through
        its OWN pair table); μ = Σ_{a,n,t'} G·V. c_pair packed [(C,) Ch, M·n_pairs]."""
        G = None
        for k, out in enumerate(outs):
            s = self._split_head(out)                                   # [(C,) ..., A, Ch]
            c = self._c_full(c_pair, k)                                 # [(C,) Ch, T, T]
            if len(c.shape) == 4:                                       # per-candidate table
                c_at = tf.gather(tf.transpose(c, [0, 2, 1, 3]), Z, axis=1)   # [C, ..., A, Ch, T]
            else:
                c_at = tf.gather(tf.transpose(c, [1, 0, 2]), Z)              # [..., A, Ch, T]
            g = s[..., tf.newaxis] * c_at
            G = g if G is None else G + g
        return G

    def predict(self, descriptors: tf.Tensor, gradients: tf.Tensor, grad_index: tf.Tensor,
                positions: tf.Tensor, Z: tf.Tensor, box: tf.Tensor,
                atom_mask: tf.Tensor, neighbor_mask: tf.Tensor,
                M_radial: tf.Tensor | None = None,
                nep_raw: tf.Tensor | None = None,
                c_r: tf.Tensor | None = None,
                c_a: tf.Tensor | None = None) -> tf.Tensor:
        """Single-structure forward on padded tensors → [1] energy, [3] dipole
        or [6] polarisability. Not for the NEP tnep readout: the padded [A, M]
        table is the rcut_hard MIC SOAP list, not the image-exact NEP stream.
            descriptors [A, dim_q] (raw [A, W_in] under radial_basis_size)
            gradients [A, M, 3, dim_q]; grad_index [A, M]; positions [A, 3]
            Z [A]; box [3, 3]; atom_mask [A]; neighbor_mask [A, M]
        """
        if self._nep_pair_forces_on:
            raise NotImplementedError(
                "predict() (padded single-structure path) has no NEP pair "
                "stream; use predict_batch for the NEP tnep readout.")
        if self._learn_radial:
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)
        descriptors = self._ann_inputs(descriptors, nep_raw, Z, c_r, c_a)[0]
        W0_eff = self._W0_eff(self.W0)
        b0_eff = self._b0_eff(self.b0, self.W0)

        W0_t = tf.gather(W0_eff, Z)    # [A, dim_q, H]
        b0_t = tf.gather(b0_eff, Z)    # [A, H]
        W1_t = tf.gather(self._masked_W1(self.W1), Z)   # [A, H]

        # z kept for the swish backward chain.
        z = tf.einsum('nd,ndh->nh', descriptors, W0_t) + b0_t   # [A, H]
        h = self.activation(z)                                   # [A, H]
        h = h * atom_mask[:, tf.newaxis]                   # [A, H]

        if self.cfg.target_mode == 0:
            E_per_atom = tf.reduce_sum(h * W1_t, axis=1) + self.b1  # [A]
            E_per_atom = E_per_atom * atom_mask                       # zero padding
            E = tf.reduce_sum(E_per_atom)
            out = tf.expand_dims(-E, axis=0)  # [1]
            return out

        if self._neighbor_readout:
            # μ = Σ_i Σ_n s_i^n Σ_t' c[t_i,t',n]·V_i[:,n,t'],
            # V_i[:,k,t'] = Σ_{j∈t'} R_n(|d|)·g_group(d), d = r_i − r_j (MIC),
            # k = group·C + n (_bond_geometry). No de_dq chain: the readout reads
            # no descriptor gradients.
            out = self._site_bias(tf.einsum('ah,ahk->ak', h, W1_t),
                                  self.b1, Z, atom_mask)              # [A, S]
            G = self._site_coeff(
                self._model_outs(out, nep_raw, Z, atom_mask, c_r, c_a),
                self.c_pair, Z)                                       # [A, C, T]
            if self._field_channel:
                raise ValueError("predict(): cfg.dipole_field needs the staged V "
                                 "(Ewald field channel); use predict_batch with W_atom.")
            dr, rij = self._neighbor_displacements_single(
                positions, box, grad_index)          # dr = neighbor − center
            rc = self._rc
            center = tf.range(tf.shape(grad_index)[0],
                              dtype=grad_index.dtype)[:, tf.newaxis]
            keep = tf.logical_and(rij < rc, tf.not_equal(grad_index, center))
            w = tf.cast(keep, rij.dtype) * neighbor_mask              # [A, M]
            Rw = self._radial(rij) * w[:, :, tf.newaxis]           # [A, M, C]
            tp = tf.one_hot(tf.gather(Z, grad_index), self.num_types,
                            dtype=tf.float64)                      # [A, M, T]
            g = self._bond_geometry(tf.cast(-dr, tf.float64))      # [A, M, D, G]
            V = tf.einsum('amn,amdg,amt->adgnt',              # float64 accum.
                          tf.cast(Rw, tf.float64), g, tp)    # [A, D, G, C, T]
            V = tf.reshape(V, [tf.shape(V)[0], tf.shape(V)[1],
                               self._kernel_channels, self.num_types])  # [A, D, Ck, T]
            mu = tf.einsum('ant,adnt->d', G, tf.cast(V, G.dtype))     # [D]
            return mu

        forces = self.calc_forces(h, gradients, W1_t, W0_t, neighbor_mask,
                                  z=z)  # [A, M, 3]

        if self.cfg.target_mode == 1:
            # μ = -Σ_pair |r_ij|^N·F_ij (N ≥ 1) or -Σ_i F_ii (N == 0, self-only).
            _, rij = self._neighbor_displacements_single(
                positions, box, grad_index)
            rij_n = (self._dipole_pair_weight_padded(tf.square(rij), grad_index)
                     * neighbor_mask)                                     # [A, M]
            dipole_contribs = rij_n[:, :, tf.newaxis] * forces            # [A, M, 3]
            dipole = -tf.reduce_sum(dipole_contribs, axis=[0, 1])         # [3]
            return dipole

    def _activation_grad(self, h: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """dh/dz for the backward chain, shaped like h.
            tanh : 1 − h²                        (needs only h)
            swish: σ(z) · (1 + z · (1 − σ(z)))   (needs z; avoids the z≫0
                   cancellation of the equivalent σ(z) + h·(1−σ(z)))
        """
        if self._activation_name == "tanh":
            return 1.0 - tf.square(h)
        if self._activation_name == "swish":
            if z is None:
                raise ValueError(
                    "_activation_grad: 'swish' requires the pre-activation z; "
                    "got z=None. Plumb z through the forward → backward "
                    "chain (do not overwrite the pre-activation tensor in "
                    "place when applying self.activation).")
            sig = tf.sigmoid(z)
            return sig * (1.0 + z * (1.0 - sig))
        raise NotImplementedError(
            f"_activation_grad: backward not implemented for activation "
            f"{self._activation_name!r}. Add a branch here, or use 'tanh' "
            f"or 'swish' / 'silu'.")

    def calc_forces(self, h: tf.Tensor, gradients: tf.Tensor, W1_t: tf.Tensor,
                    W0_t: tf.Tensor, neighbor_mask: tf.Tensor,
                    z: tf.Tensor | None = None) -> tf.Tensor:
        """dU_i/dR_j [N, M, 3] by chain rule (padded). h [N, H], gradients
        [N, M, 3, dim_q], W1_t [N, H], W0_t [N, dim_q, H], neighbor_mask [N, M],
        z [N, H] pre-activation (required for swish)."""
        dact = self._activation_grad(h, z)                       # [N, H]
        de_da = dact * W1_t                                       # [N, H]
        de_dq = tf.einsum('nh,nqh->nq', de_da, W0_t)             # [N, dim_q]
        forces = tf.einsum('nq,nmcq->nmc', de_dq, gradients)     # [N, M, 3]
        forces = forces * neighbor_mask[:, :, tf.newaxis]
        return forces

    def fit(self, train_data: dict[str, tf.Tensor], val_data: dict[str, tf.Tensor],
            plot_callback: Callable | None = None,
            resume_state: dict | None = None) -> dict:
        """Train with SNES (or Adam per cfg.optimizer).

        resume_state: dict from model_io.load_checkpoint; training continues
        from last_gen + 1.

        Returns:
            history (dict of lists), final_model, best_val_model
        """
        if str(self.cfg.optimizer) == "adam":
            if resume_state is not None:
                raise NotImplementedError(
                    "optimizer='adam' has no checkpoint/resume support.")
            from adam import fit_adam
            return fit_adam(self, train_data, val_data,
                            plot_callback=plot_callback)
        if str(self.cfg.optimizer) != "snes":
            raise ValueError(
                f"cfg.optimizer={self.cfg.optimizer!r}; expected 'snes' or 'adam'.")
        history, final_model, best_val_model = self.optimizer.fit(
            train_data, val_data, plot_callback=plot_callback,
            resume_state=resume_state)
        return history, final_model, best_val_model

    def score(self, test_data: dict[str, tf.Tensor]) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        """Evaluate on a pad_and_stack() dict, chunked so peak memory is one
        chunk's grads.

        Returns:
            metrics : rmse, r2, r2_components [T], cos_sim_mean / cos_sim_all [S]
                      (modes 1,2), and total_* twins when scale_targets is on
            preds : [S, T]
        """
        from data import prefetched_chunks
        S_test = test_data["num_atoms"].shape[0]
        chunk_sz = (self.cfg.batch_chunk_size
                    if self.cfg.batch_chunk_size is not None else S_test)
        W0_eff = self._W0_eff(self.W0)
        b0_eff = self._b0_eff(self.b0, self.W0)
        ranges = [(s, min(s + chunk_sz, S_test)) for s in range(0, S_test, chunk_sz)]
        pred_parts: list = []
        for _, _, chunk in prefetched_chunks(
                test_data, ranges,
                pin_to_cpu=self.cfg.pin_data_to_cpu):
            pred_parts.append(self.predict_batch(
                chunk["descriptors"], chunk["grad_values"],
                chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
                chunk["positions"], chunk["Z_int"], chunk["boxes"],
                chunk["atom_mask"],
                W0_eff, b0_eff, self.W1, self.b1,
                W_atom=chunk.get("V_atom"),  # neighbor readout: [B,A,D,C,T] kernel
                nep_raw=chunk.get("nep_raw"), **nep_pair_kwargs(chunk),
            ))
            del chunk
        raw_preds = tf.concat(pred_parts, axis=0)
        del pred_parts
        targets = test_data["targets"]

        if self.cfg.scale_targets and self.cfg.target_mode == 1 and "num_atoms" in test_data:
            num_atoms = tf.cast(test_data["num_atoms"], tf.float32)  # [S]
            num_atoms_col = tf.maximum(num_atoms, 1.0)[:, tf.newaxis]  # [S, 1]
            preds = raw_preds / num_atoms_col
        else:
            preds = raw_preds

        diff = preds - targets
        mse = tf.reduce_mean(tf.square(diff))
        rmse = tf.sqrt(tf.maximum(mse, 0.0))

        ss_res = tf.reduce_sum(tf.square(diff))
        ss_tot = tf.reduce_sum(tf.square(targets - tf.reduce_mean(targets, axis=0)))
        r2 = 1.0 - ss_res / tf.maximum(ss_tot, 1e-12)   # guard degenerate (e.g. 1-structure) sets

        ss_res_comp = tf.reduce_sum(tf.square(diff), axis=0)       # [T]
        ss_tot_comp = tf.reduce_sum(
            tf.square(targets - tf.reduce_mean(targets, axis=0)), axis=0)  # [T]
        r2_components = 1.0 - ss_res_comp / tf.maximum(ss_tot_comp, 1e-12)

        metrics = {
            "rmse": rmse,
            "r2": r2,
            "r2_components": r2_components,
        }

        # Total (un-scaled) twins when target scaling is active.
        if self.cfg.scale_targets and self.cfg.target_mode == 1 and "num_atoms" in test_data:
            total_targets = targets * num_atoms_col
            total_preds = raw_preds
            total_diff = total_preds - total_targets
            total_rmse = tf.sqrt(tf.reduce_mean(tf.square(total_diff)))
            total_ss_res = tf.reduce_sum(tf.square(total_diff))
            total_ss_tot = tf.reduce_sum(tf.square(
                total_targets - tf.reduce_mean(total_targets, axis=0)))
            total_r2 = 1.0 - total_ss_res / tf.maximum(total_ss_tot, 1e-12)
            total_ss_res_comp = tf.reduce_sum(tf.square(total_diff), axis=0)
            total_ss_tot_comp = tf.reduce_sum(tf.square(
                total_targets - tf.reduce_mean(total_targets, axis=0)), axis=0)
            total_r2_comp = 1.0 - total_ss_res_comp / tf.maximum(total_ss_tot_comp, 1e-12)
            metrics["total_rmse"] = total_rmse
            metrics["total_r2"] = total_r2
            metrics["total_r2_components"] = total_r2_comp

        if self.cfg.target_mode >= 1:
            dot = tf.reduce_sum(preds * targets, axis=1)          # [S]
            norm_p = tf.linalg.norm(preds, axis=1)                # [S]
            norm_t = tf.linalg.norm(targets, axis=1)              # [S]
            cos_sim = dot / tf.maximum(norm_p * norm_t, 1e-12)    # [S]
            metrics["cos_sim_mean"] = tf.reduce_mean(cos_sim)
            metrics["cos_sim_all"] = cos_sim

        return metrics, preds

    def score_from_file(self, path: str, *,
                         allowed_species: list | None = None,
                         max_structures: int | None = None,
                         filter_mode: str | None = None,
                         pin_to_cpu: bool = True,
                         plot: bool = False,
                         save_plots: str | None = None,
                         show_plots: bool = True,
                         suffix: str | None = None,
                         presentation: bool = False,
                         shared_axis_scale: bool = False) -> tuple[dict, tf.Tensor]:
        """Score an .xyz file or directory (all .xyz concatenated, sorted) via
        collect → descriptors → assemble → pad_and_stack → score.

        Training-time num_types/types/dim_q are kept so the layout matches; test
        types are re-indexed into the training ordering (unseen species raise).
        allowed_species / filter_mode / max_structures override cfg for the load
        only. plot writes parity + error figures (save_plots dir, suffix);
        presentation is a no-op; shared_axis_scale shares one range across panels.

        Returns:
            (metrics, preds) as score().
        """
        import os
        import copy
        from ase.io import read as _ase_read
        from data import collect, assemble_data_dict, pad_and_stack
        from DescriptorBuilder import make_descriptor_builder

        cfg = self.cfg

        TRAIN_NUM_TYPES = int(cfg.num_types)
        TRAIN_TYPES     = list(cfg.types)
        TRAIN_DIM_Q     = int(cfg.dim_q)

        # Directory → concat all *.xyz into a tempfile so collect sees one set.
        cfg_for_load = copy.copy(cfg)
        _tmp_xyz = None
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path)
                            if f.lower().endswith(".xyz"))
            if not files:
                raise FileNotFoundError(
                    f"score_from_file: no .xyz files found in directory {path}")
            if len(files) == 1:
                cfg_for_load.data_path = os.path.join(path, files[0])
            else:
                import tempfile
                from ase.io import write as _ase_write
                atoms_list = []
                for f in files:
                    atoms_list.extend(_ase_read(os.path.join(path, f),
                                                 index=":"))
                tmpf = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".xyz", delete=False)
                tmpf.close()
                _ase_write(tmpf.name, atoms_list)
                _tmp_xyz = tmpf.name
                cfg_for_load.data_path = _tmp_xyz
                print(f"  concatenated {len(files)} xyz files "
                      f"({len(atoms_list)} structures) → {_tmp_xyz}")
        elif os.path.isfile(path):
            cfg_for_load.data_path = path
        else:
            raise FileNotFoundError(
                f"score_from_file: path does not exist: {path}")

        if allowed_species is not None:
            cfg_for_load.allowed_species = list(allowed_species)
        if filter_mode is not None:
            cfg_for_load.filter_mode = filter_mode
        if max_structures is not None:
            cfg_for_load.total_N = int(max_structures)

        print(f"score_from_file: loading {path} ...")
        dataset, ti_loaded = collect(cfg_for_load)
        if max_structures is not None and len(dataset) > max_structures:
            dataset = dataset[:max_structures]

        # collect overrode num_types from the data; restore the trained layout.
        cfg_for_load.num_types = TRAIN_NUM_TYPES
        cfg_for_load.types     = TRAIN_TYPES
        cfg_for_load.dim_q     = TRAIN_DIM_Q

        # Re-index types into the training ordering so per-type routing is right.
        Z_to_train_idx = {int(z): i for i, z in enumerate(TRAIN_TYPES)}
        ti = []
        for s in dataset:
            Z = s.get_atomic_numbers()
            try:
                ti.append(np.array(
                    [Z_to_train_idx[int(z)] for z in Z], dtype=np.int32))
            except KeyError as exc:
                raise KeyError(
                    f"score_from_file: structure contains species Z={exc} "
                    f"not in the model's training-time species "
                    f"{TRAIN_TYPES}. Use a model trained on the broader "
                    f"species set, or filter the test set.") from exc

        print(f"  building descriptors for {len(dataset)} structures ...")
        # Same gradient gating as data.materialize_test_data.
        from data import soap_gradients_needed, dipole_self_pairs_only
        builder = make_descriptor_builder(cfg_for_load)
        descs, grads, gidx = builder.build_descriptors(
            dataset, calc_gradients=soap_gradients_needed(cfg_for_load))

        data_dict = assemble_data_dict(
            dataset, ti, descs, grads, gidx, cfg_for_load)
        test_data = pad_and_stack(
            data_dict, num_types=TRAIN_NUM_TYPES, pin_to_cpu=pin_to_cpu,
            self_pairs_only=dipole_self_pairs_only(cfg_for_load))

        print(f"  scoring on {test_data['descriptors'].shape[0]} structures ...")
        try:
            metrics, preds = self.score(test_data)
            if plot:
                from plotting import (plot_correlation, plot_error_vs_magnitude,
                                      plot_error_decomposition)
                if save_plots is not None:
                    os.makedirs(save_plots, exist_ok=True)
                targets_np = test_data["targets"].numpy()
                preds_np   = preds.numpy()
                # plot_correlation needs an RRMSE entry score() doesn't produce.
                diff = targets_np - preds_np
                std_overall = max(float(targets_np.std()), 1e-12)
                std_comp = np.maximum(targets_np.std(axis=0), 1e-12)
                rmse_comp = np.sqrt(np.mean(diff ** 2, axis=0))
                metrics_plot = dict(metrics)
                metrics_plot["rrmse"] = float(metrics["rmse"]) / std_overall
                metrics_plot["rrmse_components"] = rmse_comp / std_comp
                # presentation is stashed on cfg (no-op today).
                _prev_pres = getattr(self.cfg, "_presentation_mode", None)
                self.cfg._presentation_mode = bool(presentation)
                print(f"  plotting (save_plots={save_plots}, "
                      f"show={show_plots}, suffix={suffix!r}, "
                      f"presentation={presentation}) ...")
                try:
                    plot_correlation(targets_np, preds_np, metrics_plot,
                                      self.cfg,
                                      save_plots=save_plots,
                                      show_plots=show_plots,
                                      suffix=suffix,
                                      shared_axis_scale=shared_axis_scale)
                    plot_error_vs_magnitude(targets_np, preds_np, self.cfg,
                                             save_plots=save_plots,
                                             show_plots=show_plots,
                                             suffix=suffix)
                    plot_error_decomposition(targets_np, preds_np, self.cfg,
                                             save_plots=save_plots,
                                             show_plots=show_plots,
                                             suffix=suffix)
                finally:
                    if _prev_pres is None:
                        try:
                            delattr(self.cfg, "_presentation_mode")
                        except AttributeError:
                            pass
                    else:
                        self.cfg._presentation_mode = _prev_pres
            return metrics, preds
        finally:
            if _tmp_xyz is not None:
                try:
                    os.unlink(_tmp_xyz)
                except OSError:
                    pass

    def score_summary(self, test_data: dict[str, tf.Tensor]) -> dict:
        """Print per-atom (GPUMD loss.out) and total-dipole (NEP-paper) metrics.
        RRMSE = √(1 − R²) is centred, unlike the SNES training RRMSE (Σy²
        denominator); they agree only when target means are small.
        """
        metrics, preds = self.score(test_data)
        m = {k: (float(v.numpy()) if hasattr(v, "numpy") else float(v))
             for k, v in metrics.items()
             if v is not None and getattr(v, "shape", ())  == ()}
        rrmse_pa = (1.0 - m["r2"]) ** 0.5
        print(f"  PER-ATOM space (matches GPUMD loss.out 'rmse_virial'):")
        print(f"    RMSE  = {m['rmse']:.6f}  e·bohr / atom / component")
        print(f"    R²    = {m['r2']:.6f}")
        print(f"    RRMSE = √(1−R²) = {rrmse_pa:.4%}")
        if "total_rmse" in m and "total_r2" in m:
            rrmse_tot = (1.0 - m["total_r2"]) ** 0.5
            print(f"  TOTAL-DIPOLE space (matches NEP paper headline RMSE / R²):")
            print(f"    RMSE  = {m['total_rmse']:.6f}  e·bohr / structure / component")
            print(f"    R²    = {m['total_r2']:.6f}")
            print(f"    RRMSE = √(1−R²) = {rrmse_tot:.4%}")
        if "cos_sim_mean" in m:
            print(f"  Vector quality:")
            print(f"    cos_sim_mean = {m['cos_sim_mean']:.6f}")
        return metrics

    @tf.function(reduce_retracing=True)
    def predict_batch(self, descriptors: tf.Tensor, grad_values: tf.Tensor,
                      pair_atom: tf.Tensor, pair_gidx: tf.Tensor, pair_struct: tf.Tensor,
                      positions: tf.Tensor, Z: tf.Tensor,
                      boxes: tf.Tensor, atom_mask: tf.Tensor,
                      W0: tf.Tensor, b0: tf.Tensor, W1: tf.Tensor, b1: tf.Tensor,
                      W_atom: tf.Tensor | None = None,
                      c_pair: tf.Tensor | None = None,
                      M_radial: tf.Tensor | None = None,
                      nep_raw: tf.Tensor | None = None,
                      c_r: tf.Tensor | None = None,
                      c_a: tf.Tensor | None = None,
                      experts: tuple | None = None,
                      sum_delta: tf.Tensor | None = None,
                      nep_pair_atom: tf.Tensor | None = None,
                      nep_pair_gidx: tf.Tensor | None = None,
                      nep_pair_struct: tf.Tensor | None = None,
                      nep_pair_D: tf.Tensor | None = None,
                      nep_pair_ang: tf.Tensor | None = None) -> tf.Tensor:
        """Batched forward for B structures with COO gradients → [B, T_dim].
        Weights are explicit (already U-folded by the caller; no mixing here).
            descriptors [B, A, Q] (raw [B, A, W_in] under radial_basis_size)
            grad_values [P, 3, Q]; pair_atom / pair_gidx / pair_struct [P]
            positions [B, A, 3]; Z [B, A]; boxes [B, 3, 3]; atom_mask [B, A]
            W0 [T, Q, H]; b0 [T, H]; W1 [T, H] ([T, H, S] neighbor); b1 ()
            W_atom      : dipole kernel [B, A, 3, Q]; neighbor readout: V
                          [B, A, D, Ck, T] (built in-graph when None)
            c_pair, M_radial, c_r, c_a, experts (STORED, EXPERT_WEIGHTS order),
                          sum_delta [3]: None → the model's variables
            nep_pair_*  : image-exact NEP pair stream (data.py nep_pairs):
                          atom/gidx/struct [P], D [P, 3] = r_j − r_i, ang [P] bool
        """
        if self._learn_radial:
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)
        descriptors, s_nep = self._ann_inputs(descriptors, nep_raw, Z, c_r, c_a)
        box_inv = tf.linalg.inv(boxes)  # [B, 3, 3]

        W0_use = W0

        b0_t = tf.gather(b0, Z)   # [B, A, H]
        W1_t = tf.gather(self._masked_W1(W1), Z)   # [B, A, H_final]

        # Per-type loop avoids materialising a [B, A, Q, H] W0 gather.
        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(self.num_types)
        ]
        # z1 kept for the swish backward chain.
        z1 = tf.add_n([
            tf.einsum('baq,qh->bah', descriptors, W0_use[t]) * type_masks[t]
            for t in range(self.num_types)
        ]) + b0_t
        h1 = self.activation(z1)
        h1 = h1 * atom_mask[:, :, tf.newaxis]

        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h1 * W1_t, axis=2) + b1  # [B, A]
            E = E * atom_mask
            E = tf.reduce_sum(E, axis=1, keepdims=True)  # [B, 1]
            return -E

        if self._neighbor_readout:
            # μ[b] = Σ_{a,n,t'} s[b,a,n]·c[t_a,t',n]·V[b,a,:,n,t']; no de_dq chain.
            if c_pair is None:
                c_pair = self.c_pair
            out = self._site_bias(tf.einsum('bah,bahk->bak', h1, W1_t),
                                  b1, Z, atom_mask)                   # [B, A, S]
            G = self._site_coeff(
                self._model_outs(out, nep_raw, Z, atom_mask, c_r, c_a,
                                 experts, sum_delta), c_pair, Z)     # [B, A, C, T]
            V = W_atom  # [B, A, D, C, T] geometry kernel via the W_atom slot
            if V is None:
                # Full-pair callers (spectroscopy): in-graph V, r_c ≤ rcut_hard.
                V = self._neighbor_V_coo(
                    positions, boxes, box_inv, pair_struct, pair_atom,
                    pair_gidx, Z, tf.shape(descriptors)[0],
                    tf.shape(descriptors)[1])
            mu = tf.einsum('bant,badnt->bd', G, V)                # [B, D]
            return mu

        # Backward chain: ∂U/∂a1 = activation'(h1, z1)·W1, ∂U/∂q = ∂U/∂a1·W0ᵀ.
        de_da = self._activation_grad(h1, z1) * W1_t
        de_dq = tf.add_n([
            tf.einsum('bah,qh->baq', de_da, W0_use[t]) * type_masks[t]
            for t in range(self.num_types)
        ])

        B = tf.shape(descriptors)[0]

        if self._nep_pair_forces_on:
            # NEP columns → image-exact pair forces; SOAP columns → W_atom / COO path.
            off = self._soap_dim if self._nep_mode == "concat" else 0
            mu = self._dipole_nep_pairs(
                de_dq[tf.newaxis, :, :, off:], s_nep[tf.newaxis], c_r, c_a, Z, B,
                nep_pair_atom, nep_pair_gidx, nep_pair_struct, nep_pair_D, nep_pair_ang)[0]
            if self._nep_mode == "replace":
                return mu
            de_dq = de_dq[:, :, :off]
            if W_atom is not None:
                return mu - tf.einsum('baq,basq->bs', de_dq, W_atom)
            return mu + self._dipole_coo(
                self._calc_forces_coo(de_dq, grad_values, pair_struct, pair_atom),
                pair_struct, pair_atom, pair_gidx, positions, boxes, box_inv, B)

        if self.cfg.target_mode == 1 and W_atom is not None:
            # Precomputed kernel: dipole[b,s] = -Σ_{a,q} de_dq·W_atom (no [C, P, Q]).
            return -tf.einsum('baq,basq->bs', de_dq, W_atom)  # [B, 3]

        forces_per_pair = self._calc_forces_coo(de_dq, grad_values, pair_struct, pair_atom)

        if self.cfg.target_mode == 1:
            return self._dipole_coo(forces_per_pair, pair_struct, pair_atom, pair_gidx,
                                    positions, boxes, box_inv, B)

        tf.debugging.assert_equal(True, False, message="Unsupported target_mode")

    @tf.function(reduce_retracing=True)
    def predict_batch_candidates(self,
                                  descriptors: tf.Tensor,
                                  W_atom: tf.Tensor | None,
                                  Z: tf.Tensor,
                                  atom_mask: tf.Tensor,
                                  W0: tf.Tensor, b0: tf.Tensor,
                                  W1: tf.Tensor, b1: tf.Tensor,
                                  U_pair: tf.Tensor | None = None,
                                  c_pair: tf.Tensor | None = None,
                                  M_radial: tf.Tensor | None = None,
                                  nep_raw: tf.Tensor | None = None,
                                  c_r: tf.Tensor | None = None,
                                  c_a: tf.Tensor | None = None,
                                  experts: tuple | None = None,
                                  sum_delta: tf.Tensor | None = None,
                                  nep_pair_atom: tf.Tensor | None = None,
                                  nep_pair_gidx: tf.Tensor | None = None,
                                  nep_pair_struct: tf.Tensor | None = None,
                                  nep_pair_D: tf.Tensor | None = None,
                                  nep_pair_ang: tf.Tensor | None = None) -> tf.Tensor:
        """Forward for C candidates × B structures → [C, B, T_dim], as batched GEMMs:
            forward  [B*A, Q] @ [Q, C*H]     → [C, B, A, H]
            backward [C, B*A, H] @ [C, H, Q] → [C, B*A, Q]
        Per-candidate twin of predict_batch; STORED weights (U/scale folded here):
            W0 [C, T, Q, H]; b0 [C, T, H]; W1 [C, T, H] ([C, T, H, S] neighbor); b1 [C]
            W_atom [B, A, 3, Q] (neighbor: V [B, A, D, Ck, T]); U_pair / c_pair /
            M_radial / c_r, c_a / experts / sum_delta with a leading C (shared
            [3] sum_delta allowed; None → the model's)
        """
        Q = self.dim_q
        H = self.num_neurons
        H_final = self._H_final
        T = self.num_types

        B = tf.shape(descriptors)[0]
        A = tf.shape(descriptors)[1]
        C = tf.shape(W0)[0]

        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(T)
        ]

        b0 = self._b0_eff(b0, W0)          # centring fold needs the STORED W0
        W0 = self._W0_eff(W0, U_pair)

        # ── forward: input→hidden ──
        pre_h_terms = []
        if self._learn_radial:
            if M_radial is None:
                raise ValueError(
                    "predict_batch_candidates needs per-candidate M_radial "
                    "[C, T, S, α, K] under cfg.radial_basis_size.")
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)         # [C, B, A, Q]
        descriptors, s_nep = self._ann_inputs(descriptors, nep_raw, Z, c_r, c_a)
        if len(descriptors.shape) == 4:
            # Per-candidate descriptors [C, B, A, Q]: batched GEMM over C per type.
            desc_c = tf.reshape(descriptors, [C, B * A, Q])                    # [C, B*A, Q]
            for t in range(T):
                ph = tf.reshape(tf.matmul(desc_c, W0[:, t, :, :]), [C, B, A, H])  # [C,B,A,H]
                pre_h_terms.append(ph * type_masks[t][tf.newaxis])
        else:
            # One GEMM per type: [B*A, Q] @ [Q, C*H] → [C, B, A, H].
            desc_flat = tf.reshape(descriptors, [B * A, Q])
            for t in range(T):
                W0_t     = W0[:, t, :, :]                                              # [C, Q, H]
                W0_t_mat = tf.reshape(tf.transpose(W0_t, [1, 0, 2]), [Q, C * H])      # [Q, C*H]
                ph_flat  = tf.matmul(desc_flat, W0_t_mat)                             # [B*A, C*H]
                ph       = tf.transpose(tf.reshape(ph_flat, [B, A, C, H]), [2, 0, 1, 3])  # [C,B,A,H]
                pre_h_terms.append(ph * type_masks[t][tf.newaxis])
        pre_h = tf.add_n(pre_h_terms)  # [C, B, A, H]

        # ── bias / output-weight gathers ──
        Z_flat   = tf.reshape(Z, [B * A])
        b0_t_all = tf.reshape(tf.gather(b0, Z_flat, axis=1), [C, B, A, H])
        _w1_shape = ([C, B, A, H_final, self._head_slots]
                     if self._neighbor_readout else [C, B, A, H_final])
        W1_t_all = tf.reshape(tf.gather(self._masked_W1(W1), Z_flat, axis=1), _w1_shape)

        # z1 kept for the swish backward chain.
        z1 = pre_h + b0_t_all
        h1 = self.activation(z1)
        h1 = h1 * atom_mask[tf.newaxis, :, :, tf.newaxis]

        # ── PES ──
        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h1 * W1_t_all, axis=3) + b1[:, tf.newaxis, tf.newaxis]
            E = E * atom_mask[tf.newaxis]
            return -tf.reduce_sum(E, axis=2, keepdims=True)  # [C, B, 1]

        # ── neighbor readout ──
        if self._neighbor_readout:
            # W_atom slot carries V [B, A, D, Ck, T]; no backward matmul.
            out = self._site_bias(tf.einsum('cbah,cbahk->cbak', h1, W1_t_all),
                                  b1, Z, atom_mask)                 # [C, B, A, S]
            # c_pair None: candidate-independent table, broadcasts over C.
            G = self._site_coeff(
                self._model_outs(out, nep_raw, Z, atom_mask, c_r, c_a,
                                 experts, sum_delta),
                self.c_pair if c_pair is None else c_pair, Z)      # [C, B, A, Ch, T]
            mu = tf.einsum('cbant,badnt->cbd', G, W_atom)          # [C, B, D]
            return mu

        # ── dipole: backward matmul ──
        de_da    = self._activation_grad(h1, z1) * W1_t_all   # [C, B, A, H]
        de_da_flat = tf.reshape(de_da, [C, B * A, H])         # [C, B*A, H]

        de_dq_terms = []
        for t in range(T):
            W0_t_T  = tf.transpose(W0[:, t, :, :], [0, 2, 1])    # [C, H, Q]
            dq_flat = tf.matmul(de_da_flat, W0_t_T)               # [C, B*A, Q]
            dq      = tf.reshape(dq_flat, [C, B, A, Q])
            de_dq_terms.append(dq * type_masks[t][tf.newaxis])
        de_dq = tf.add_n(de_dq_terms)  # [C, B, A, Q]

        if self._nep_pair_forces_on:
            off = self._soap_dim if self._nep_mode == "concat" else 0
            mu = self._dipole_nep_pairs(
                de_dq[:, :, :, off:], s_nep, c_r, c_a, Z, B,
                nep_pair_atom, nep_pair_gidx, nep_pair_struct, nep_pair_D, nep_pair_ang)
            if self._nep_mode == "replace":
                return mu
            return mu - tf.einsum('cbaq,basq->cbs', de_dq[:, :, :, :off], W_atom)

        return -tf.einsum('cbaq,basq->cbs', de_dq, W_atom)  # [C, B, 3]

    def _dipole_nep_pairs(self, de_dq_nep: tf.Tensor, s: tf.Tensor,
                          c_r: tf.Tensor | None, c_a: tf.Tensor | None,
                          Z: tf.Tensor, B: tf.Tensor,
                          nep_pair_atom, nep_pair_gidx, nep_pair_struct,
                          nep_pair_D, nep_pair_ang) -> tf.Tensor:
        """μ [Cc, B, 3] = −Σ_p |D_p|^N F_p from the NEP pair forces (N ≥ 1).
        |D| from the stored image-exact D, never MIC: rc_r = 8 Å exceeds L/2."""
        if nep_pair_D is None:
            raise ValueError(
                "nep_pair_* stream missing: the tnep readout under "
                f"cfg.nep_descriptor={self._nep_mode!r} needs data['nep_pair_*'] "
                "(built by prepare_eval_data when descriptor gradients are needed).")
        F = self._nep_pair_forces(de_dq_nep, s, c_r, c_a, Z, nep_pair_atom, nep_pair_gidx,
                                  nep_pair_struct, nep_pair_D, nep_pair_ang)      # [Cc, P, 3]
        w = self._scalar_rij_pow(tf.reduce_sum(tf.square(nep_pair_D), axis=-1))   # [P]
        contrib = tf.transpose(w[tf.newaxis, :, tf.newaxis] * F, [1, 0, 2])        # [P, Cc, 3]
        mu = tf.math.unsorted_segment_sum(contrib, nep_pair_struct, num_segments=B)
        return -tf.transpose(mu, [1, 0, 2])                                       # [Cc, B, 3]

    def _nep_pair_forces(self, de_dq_nep: tf.Tensor, s: tf.Tensor,
                         c_r: tf.Tensor | None, c_a: tf.Tensor | None, Z: tf.Tensor,
                         nep_pair_atom, nep_pair_gidx, nep_pair_struct,
                         nep_pair_D, nep_pair_ang) -> tf.Tensor:
        """Per-pair NEP forces F_p = ∂U_{i_p}/∂D_p [Cc, P, 3], D = r_j − r_i (the
        neighbour side, GPUMD's x12; centre side is −F_p), U_i = Σ_d Fp[i,d] q_d:
            radial   F^r_p = (Σ_k α[i,t_j,k] fnp_k(r)) r̂,  α[i,t',k] = Σ_n Fp[i,n] c_r[t_i,t',n,k]
            angular  Gs[i,n,c] = ∂(Σ_d Fp[i,d] q_d(s_i))/∂s[i,n,c]  (tape over find_q_tf)
                     F^a_p = Σ_n [ gnp_n (Σ_c Gs P_c) r̂ + g_n Σ_c Gs ∇P_c ]  (XLA n-loop)
        de_dq_nep [Cc, B, A, dim_nep] (Cc = 1 for predict_batch); s [(Cc,) B, A,
        n_a+1, 24]; c_r, c_a [(Cc,) T, T, n+1, K+1]; nep_pair_D [P, 3]; nep_pair_ang
        [P] bool is the authoritative rc_a mask (never re-derived from float32 D).
        """
        cfg = self.cfg
        c_r = self.c_r if c_r is None else c_r
        c_a = self.c_a if c_a is None else c_a
        T = self.num_types
        n_r, n_a = cfg.nep_n_max_radial + 1, cfg.nep_n_max_angular + 1
        K_r, K_a = cfg.nep_basis_size_radial + 1, cfg.nep_basis_size_angular + 1
        Cc = tf.shape(de_dq_nep)[0]
        A = tf.shape(Z)[1]
        N = tf.shape(Z)[0] * A
        if len(c_r.shape) == 4:
            c_r = tf.broadcast_to(c_r[tf.newaxis], tf.concat([[Cc], tf.shape(c_r)], 0))
        if len(c_a.shape) == 4:
            c_a = tf.broadcast_to(c_a[tf.newaxis], tf.concat([[Cc], tf.shape(c_a)], 0))
        if len(s.shape) == 4:
            s = tf.broadcast_to(s[tf.newaxis], tf.concat([[Cc], tf.shape(s)], 0))
        Fp = tf.reshape(de_dq_nep, [Cc, N, self._nep_dim])
        s = tf.reshape(s, [Cc, N, n_a, NUM_ABC])
        Zf = tf.reshape(Z, [N])
        i_lin = nep_pair_struct * A + nep_pair_atom
        j_lin = nep_pair_struct * A + nep_pair_gidx
        t_i, t_j = tf.gather(Zf, i_lin), tf.gather(Zf, j_lin)
        slot = t_i * T + t_j                                                 # (t_i, t_j) → T·T

        r = tf.norm(nep_pair_D, axis=1)                                      # [P]
        rhat = nep_pair_D / r[:, tf.newaxis]
        _, fnp_r = fn_and_fnp_tf(r, cfg.nep_rc_radial, K_r - 1)              # [P, K_r]

        # Radial: α [Cc, N, T, K_r], gathered per pair at (i, t_j).
        c_r_i = tf.gather(c_r, Zf, axis=1)                                   # [Cc, N, T, n_r, K_r]
        alpha = tf.einsum('cin,citnk->citk', Fp[:, :, :n_r], c_r_i)
        F = _nep_radial_xla(tf.reshape(alpha, [Cc, N * T, K_r]),
                            i_lin * T + t_j, fnp_r, rhat, K_r)                # [Cc, P, 3]

        # Angular: Gs by tape at the atom level (l-major Fp columns → [n, L]).
        Fp_a = tf.transpose(tf.reshape(Fp[:, :, n_r:], [Cc, N, -1, n_a]), [0, 1, 3, 2])
        with tf.GradientTape() as tape:
            tape.watch(s)
            G = tf.reduce_sum(find_q_tf(s, cfg.nep_four_body, cfg.nep_five_body) * Fp_a)
        Gs = tape.gradient(G, s)                                             # [Cc, N, n_a, 24]
        ang = tf.where(nep_pair_ang)[:, 0]                                   # [P_a]
        r_a, u = tf.gather(r, ang), tf.gather(rhat, ang)
        fn_a, fnp_a = fn_and_fnp_tf(r_a, cfg.nep_rc_angular, K_a - 1)        # [P_a, K_a]
        Pc = real_harmonics_tf(u[:, 0], u[:, 1], u[:, 2])                    # [P_a, 24]
        dP = real_harmonics_grad_tf(u[:, 0], u[:, 1], u[:, 2])               # [P_a, 3, 24] free r̂
        dPc = (dP - u[:, :, tf.newaxis] * tf.reduce_sum(u[:, :, tf.newaxis] * dP, 1, keepdims=True)
               ) / r_a[:, tf.newaxis, tf.newaxis]                            # (I − r̂r̂ᵀ)/r · dP
        F_a = _nep_angular_xla(Gs, tf.reshape(c_a, [Cc, T * T, n_a, K_a]),
                               tf.gather(i_lin, ang), tf.gather(slot, ang),
                               fn_a, fnp_a, u, Pc, dPc, n_a, K_a)             # [Cc, P_a, 3]
        F_a = tf.math.unsorted_segment_sum(tf.transpose(F_a, [1, 0, 2]), ang,
                                           num_segments=tf.shape(r)[0])      # [P, Cc, 3]
        return F + tf.transpose(F_a, [1, 0, 2])

    def _scalar_rij_pow(self, rij2: tf.Tensor) -> tf.Tensor:
        """|r_ij|^N from rij² (N ≥ 1; no sqrt for even N). N=0 is handled by
        _dipole_pair_weight_*."""
        N = int(getattr(self.cfg, "dipole_rij_power", 2))
        if N == 1:
            return tf.sqrt(rij2)
        if N == 2:
            return rij2
        if N % 2 == 0:
            return tf.pow(rij2, N // 2)
        return tf.pow(rij2, (N - 1) // 2) * tf.sqrt(rij2)

    def _dipole_pair_weight_coo(self, rij2: tf.Tensor,
                                 pair_atom: tf.Tensor,
                                 pair_gidx: tf.Tensor) -> tf.Tensor:
        """Per-pair dipole weight (COO). N == 0: 1 on true self pairs
        (pair_atom==pair_gidx AND rij² < 1e-20 — the guard rejects periodic
        self-images); N ≥ 1: |r_ij|^N."""
        N = int(getattr(self.cfg, "dipole_rij_power", 2))
        if N == 0:
            is_self = tf.logical_and(tf.equal(pair_atom, pair_gidx),
                                     rij2 < 1e-20)
            return tf.cast(is_self, rij2.dtype)
        return self._scalar_rij_pow(rij2)

    def _dipole_pair_weight_padded(self, rij2: tf.Tensor,
                                    grad_index: tf.Tensor) -> tf.Tensor:
        """Padded [A, M] twin of _dipole_pair_weight_coo. Caller MUST multiply
        by neighbor_mask: padding rows look like self pairs (grad_index==0, rij²==0)."""
        N = int(getattr(self.cfg, "dipole_rij_power", 2))
        if N == 0:
            A = tf.shape(grad_index)[0]
            center = tf.range(A, dtype=grad_index.dtype)[:, tf.newaxis]
            is_self = tf.logical_and(tf.equal(grad_index, center),
                                     rij2 < 1e-20)
            return tf.cast(is_self, rij2.dtype)
        return self._scalar_rij_pow(rij2)

    def _calc_forces_coo(self, de_dq: tf.Tensor, grad_values: tf.Tensor,
                         pair_struct: tf.Tensor, pair_atom: tf.Tensor) -> tf.Tensor:
        """Per-pair forces [P, 3] = de_dq[B, A, Q] gathered per pair · grad_values [P, 3, Q]."""
        ba = tf.stack([pair_struct, pair_atom], axis=1)          # [P, 2]
        de_dq_per_pair = tf.gather_nd(de_dq, ba)                 # [P, Q]
        return tf.einsum('kq,kcq->kc', de_dq_per_pair, grad_values)  # [P, 3]

    def _neighbor_displacements_single(self, positions: tf.Tensor,
                                       box: tf.Tensor,
                                       grad_index: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """MIC displacements for one padded structure: positions [A, 3], box [3, 3],
        grad_index [A, M] → dr [A, M, 3] (neighbor − center), rij [A, M]."""
        box_inv = tf.linalg.inv(box)
        # ASE row-vector cells: s = r @ inv(cell), so contract the FIRST index of
        # box_inv ('ji'); 'ij' gives wrong MIC wrapping for triclinic cells.
        s = tf.einsum('ji,nj->ni', box_inv, positions)       # [A, 3]
        s_j = tf.gather(s, grad_index)                        # [A, M, 3]
        s_i = s[:, tf.newaxis, :]                             # [A, 1, 3]
        ds = s_j - s_i
        ds = ds - tf.round(ds)
        dr = tf.einsum('ji,nmj->nmi', box, ds)                # [A, M, 3]
        rij = tf.linalg.norm(dr, axis=-1)                     # [A, M]
        return dr, rij

    def _precompute_dipole_kernel(self, grad_values: tf.Tensor,
                                  pair_struct: tf.Tensor, pair_atom: tf.Tensor,
                                  pair_gidx: tf.Tensor, positions: tf.Tensor,
                                  boxes: tf.Tensor, B: tf.Tensor,
                                  A: tf.Tensor) -> tf.Tensor:
        """Candidate-independent dipole kernel W_atom [B, A, 3, Q] =
        Σ_{p: struct=b, atom=a} |r|^N[p]·grad_values[p, :, :], so the dipole is
        -einsum('baq,basq->bs', de_dq, W_atom) with no [C, P, Q] intermediate.
        grad_values [P, 3, Q] (already scaled); positions [B, A, 3]; boxes [B, 3, 3]."""
        P = tf.shape(grad_values)[0]
        Q = tf.shape(grad_values)[2]
        # N == 0: the COO list is already self-pairs only (data.py), weight 1.
        _N = int(getattr(self.cfg, "dipole_rij_power", 2))
        if _N == 0:
            weight = tf.ones([P], dtype=grad_values.dtype)
        else:
            box_inv = tf.linalg.inv(boxes)
            _, rij2 = self._neighbor_displacements_coo(
                positions, boxes, box_inv,
                pair_struct, pair_atom, pair_gidx)
            weight = self._dipole_pair_weight_coo(rij2, pair_atom, pair_gidx)  # [P]
        W = weight[:, tf.newaxis, tf.newaxis] * grad_values   # [P, 3, Q]
        W_flat = tf.reshape(W, [P, 3 * Q])                  # [P, 3*Q]
        ba_linear = pair_struct * A + pair_atom              # [P] linear index into [B*A]
        W_atom_flat = tf.math.unsorted_segment_sum(
            W_flat, ba_linear, num_segments=B * A)           # [B*A, 3*Q]
        return tf.reshape(W_atom_flat, [B, A, 3, Q])         # [B, A, 3, Q]

    def _neighbor_displacements_coo(self, positions: tf.Tensor, boxes: tf.Tensor,
                                    box_inv: tf.Tensor, pair_struct: tf.Tensor,
                                    pair_atom: tf.Tensor,
                                    pair_gidx: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """COO twin of _neighbor_displacements_single: positions [B, A, 3], boxes /
        box_inv [B, 3, 3], pair_* [P] → dr [P, 3] (neighbor − center), rij2 [P]."""
        ba_c = tf.stack([pair_struct, pair_atom], axis=1)   # [P, 2]
        ba_n = tf.stack([pair_struct, pair_gidx], axis=1)   # [P, 2]
        pos_c   = tf.gather_nd(positions, ba_c)              # [P, 3]
        pos_n   = tf.gather_nd(positions, ba_n)              # [P, 3]
        box_k   = tf.gather(boxes,   pair_struct)            # [P, 3, 3]
        binv_k  = tf.gather(box_inv, pair_struct)            # [P, 3, 3]
        # 'kji' contracts the first index of box_inv (ASE row-vector cells), see _single.
        s_c = tf.einsum('kji,kj->ki', binv_k, pos_c)        # [P, 3] fractional
        s_n = tf.einsum('kji,kj->ki', binv_k, pos_n)
        ds  = s_n - s_c
        ds  = ds - tf.round(ds)                              # MIC wrap
        dr  = tf.einsum('kji,kj->ki', box_k, ds)            # [P, 3] Cartesian
        rij2 = tf.reduce_sum(tf.square(dr), axis=-1)         # [P]
        return dr, rij2

    def _fc(self, r: tf.Tensor) -> tf.Tensor:
        """f_c(r): "cosine2" ((1+cos(πr/r_c))/2)², C³ at r_c; "cosine" (1+cos(πr/r_c))/2,
        C¹. Callers mask r ≥ r_c."""
        rc = self._rc
        fc = self._dipole_cutoff
        if fc == "cosine2":
            return tf.square(0.5 * (1.0 + tf.cos(np.pi / rc * r)))
        if fc == "cosine":
            return 0.5 * (1.0 + tf.cos(np.pi / rc * r))
        raise ValueError(f"unknown dipole_cutoff {fc!r}")

    def _cheb_R(self, r: tf.Tensor) -> tf.Tensor:
        """R_n(r) = T_n(x)·f_c(r), n = 0..C−1 → [..., C]; x = 2r/r_c − 1, or the
        NEP warp 2(r/r_c − 1)² − 1 for "chebyshev_warped". Recurrence, not
        cos∘arccos, so out-of-cutoff r stays finite until masked."""
        C = self._dipole_channels
        if self._dipole_radial == "chebyshev_warped":
            x = 2.0 * tf.square(r / self._rc - 1.0) - 1.0
        else:
            x = 2.0 * r / self._rc - 1.0
        Ts = [tf.ones_like(x), x]
        for _ in range(2, C):
            Ts.append(2.0 * x * Ts[-1] - Ts[-2])
        return tf.stack(Ts[:C], axis=-1) * self._fc(r)[..., tf.newaxis]

    def _radial(self, r: tf.Tensor) -> tf.Tensor:
        """Radial channels [..., C] per cfg.dipole_radial; "screened" is
        r^(ζ−1)·f_c(r), one channel. Callers mask AFTER the radial and r=0
        self-images exist, so guard the power's INPUT (0**(ζ−1) = inf → nan)."""
        rad = self._dipole_radial
        if rad == "screened":
            r_safe = tf.where(r > 0, r, tf.ones_like(r))
            R = r_safe ** (self._dipole_zeta - 1.0) * self._fc(r)
            return R[..., tf.newaxis]
        if rad in ("chebyshev", "chebyshev_warped"):
            return self._cheb_R(r)
        if rad == "gaussian":
            mu = tf.constant(self._gauss_mu, dtype=r.dtype)
            g = tf.exp(-tf.square(r[..., tf.newaxis] - mu)
                       / (2.0 * self._gauss_sigma ** 2))
            return g * self._fc(r)[..., tf.newaxis]
        if rad == "bessel":
            # j₀ family sin(nπr/rc)/(r/rc); guard the r=0 division only.
            rc = self._rc
            n1 = tf.constant(np.arange(1, self._dipole_channels + 1), dtype=r.dtype)
            x = tf.where(r > 0, r, tf.ones_like(r)) / rc
            return (tf.sin(np.pi * n1 * x[..., tf.newaxis]) / x[..., tf.newaxis]
                    * self._fc(r)[..., tf.newaxis])
        raise ValueError(f"unknown dipole_radial {rad!r}")

    def _bond_geometry(self, d: tf.Tensor) -> tf.Tensor:
        """Bond tensors [..., 3] → [..., D, G]: rank 1 d itself (D=3, G=1); rank 2
        Voigt (D=6, G=2) isotropic |d|²·I and traceless d⊗d − |d|²I/3, both even
        in d so the per-site sum equals the pair form. TF twin of data.bond_geometry."""
        if self._bond_rank == 1:
            return d[..., tf.newaxis]
        x, y, z = tf.unstack(d, axis=-1)
        r2 = x * x + y * y + z * z
        zero = tf.zeros_like(r2)
        iso = tf.stack([r2, r2, r2, zero, zero, zero], axis=-1)
        tr = tf.stack([x * x, y * y, z * z, x * y, y * z, z * x], axis=-1) - iso / 3.0
        return tf.stack([iso, tr], axis=-1)

    def _masked_W1(self, W1: tf.Tensor) -> tf.Tensor:
        """W1 [.., T, H, S] → block-diagonal over the kernel groups (_head_mask)."""
        return W1 if self._head_mask is None else W1 * tf.cast(self._head_mask, W1.dtype)

    def _split_head(self, out: tf.Tensor):
        """Head [.., S] → site values [.., Ck]: "scalar" repeats each group's
        slot over its C channels (group-major like V)."""
        if self._head_slots == self._kernel_channels:
            return out
        if self._field_channel:                      # groups repeated, field slot as is
            return tf.concat([tf.repeat(out[..., :-1], self._dipole_channels, axis=-1),
                              out[..., -1:]], axis=-1)
        return tf.repeat(out, self._dipole_channels, axis=-1)

    def _c_full(self, c_pair: tf.Tensor, k: int = 0) -> tf.Tensor:
        """Model k's table from packed c_pair [..., C, M·n_pairs] → [..., C, T, T]."""
        return tf.gather(c_pair, self._pair_idx + k * self._n_pairs, axis=-1)

    def _neighbor_V_coo(self, positions: tf.Tensor, boxes: tf.Tensor,
                        box_inv: tf.Tensor, pair_struct: tf.Tensor,
                        pair_atom: tf.Tensor, pair_gidx: tf.Tensor,
                        Z: tf.Tensor, B: tf.Tensor, A: tf.Tensor) -> tf.Tensor:
        """In-graph geometry kernel V [B, A, D, Ck, T]:
            V[b,a,:,k,t'] = Σ_{pairs (a,j), |d|<r_c, t_j=t'} R_n(|d|)·g(d), d = r_a − r_j (MIC)
        from the SOAP pair list (needs r_c ≤ rcut_hard); self-images excluded;
        float64 accumulation. Twin of the staged V from data.py."""
        if self._field_channel:
            raise ValueError(
                "in-graph V: cfg.dipole_field needs the staged V (Ewald field "
                "channel built at data assembly); this path cannot build it.")
        if self._rc > float(self.cfg.rcut_hard):
            raise ValueError(
                f"in-graph V: readout r_c={self._rc} > "
                f"rcut_hard={self.cfg.rcut_hard}, but this path builds V from "
                f"the SOAP descriptor pair stream, which only reaches "
                f"rcut_hard — it would silently truncate the readout. Pass the "
                f"staged V (data['V_atom']) instead; training, validation and "
                f"score already do.")
        dr, rij2 = self._neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)
        rc = self._rc
        C = self._dipole_channels
        T = self.num_types
        r = tf.sqrt(rij2)
        keep = tf.logical_and(r < rc, tf.not_equal(pair_atom, pair_gidx))
        Rw = self._radial(r) * tf.cast(keep, r.dtype)[:, tf.newaxis]  # [P, C]
        # contrib[p, d, k] = R_n·g_group(d), k = group·C + n.
        g = self._bond_geometry(-dr)                                # [P, D, G]
        D, Ck = int(g.shape[1]), self._kernel_channels
        contrib = tf.reshape(
            tf.einsum('pdg,pn->pdgn', tf.cast(g, tf.float64),
                      tf.cast(Rw, tf.float64)), [-1, D, Ck])       # [P, D, Ck]
        tprime = tf.gather_nd(Z, tf.stack([pair_struct, pair_gidx], axis=1))
        seg = (pair_struct * A + pair_atom) * T + tf.cast(tprime, pair_atom.dtype)
        V = tf.math.unsorted_segment_sum(
            tf.reshape(contrib, [-1, D * Ck]), seg, num_segments=B * A * T)
        V = tf.reshape(tf.cast(V, dr.dtype), [B, A, T, D, Ck])
        return tf.transpose(V, [0, 1, 3, 4, 2])                  # [B,A,D,Ck,T]

    def _dipole_coo(self, forces_per_pair: tf.Tensor, pair_struct: tf.Tensor,
                    pair_atom: tf.Tensor, pair_gidx: tf.Tensor,
                    positions: tf.Tensor, boxes: tf.Tensor,
                    box_inv: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
        """μ [B, 3] = -Σ_pair |r_ij|^N·F_ij (N==0: -Σ_i F_ii) from COO forces [P, 3]."""
        _, rij2 = self._neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)
        weight = self._dipole_pair_weight_coo(rij2, pair_atom, pair_gidx)  # [P]
        dipole_contrib = weight[:, tf.newaxis] * forces_per_pair         # [P, 3]
        dipole = -tf.math.unsorted_segment_sum(
            dipole_contrib, pair_struct, num_segments=B)                  # [B, 3]
        return dipole


def gen0_ann_inputs(cfg, data) -> np.ndarray:
    """[S, A, dim_q] gen-0 ANN input of a padded data dict (what compute_q_scale
    sees): a throwaway unscaled model finishes the `descriptors` slot chunk by
    chunk; under "sum" the expert input follows ([q_soap ‖ q_nep]). "minmax"
    uses c ≡ 1 like GPUMD's q_scaler (fitness.cu:165-175); block/channel the real c."""
    q0, Z = data["descriptors"], data["Z_int"]
    if cfg.radial_basis_size is None and str(getattr(cfg, "nep_descriptor", "off")) == "off":
        return q0.numpy() if hasattr(q0, "numpy") else np.asarray(q0)
    cfg0 = copy.copy(cfg)
    cfg0.descriptor_scale = "off"
    m0 = TNEP(cfg0)
    nep_raw = data.get("nep_raw")
    c = {}
    if m0.c_r is not None and str(cfg.descriptor_scale) == "minmax":
        c = dict(c_r=tf.ones_like(m0.c_r), c_a=tf.ones_like(m0.c_a))
    S = int(q0.shape[0])
    step = cfg.batch_chunk_size or S
    out = []
    for s in range(0, S, step):
        d, z = q0[s:s + step], Z[s:s + step]
        if cfg.radial_basis_size is not None:
            d = m0.soap_from_raw(d, z)
        raw = None if nep_raw is None else nep_raw[s:s + step]
        x = m0._ann_inputs(d, raw, z, **c)[0]
        if m0._nep_mode == "sum":
            x = tf.concat([x, m0.nep_from_raw(raw, z, **c)], axis=-1)
        out.append(x)
    return tf.concat(out, axis=0).numpy()
