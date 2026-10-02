from __future__ import annotations

import copy
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Callable, NamedTuple

from DescriptorBuilder import make_descriptor_builder
from SNES import SNES
from TNEPconfig import TNEPconfig, resolved_dipole_cutoff, resolved_dipole_rc
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
    """F^a [Cc, P_a, 3] = Σ_n [ gnp_n (Σ_c Gs[i,n,c] P_c) r̂ + g_n Σ_c Gs[i,n,c] ∇P_c ]
    with g_n = Σ_k c_a[t_i,t_j,n,k] fn_k, gnp_n likewise with fnp_k (plan §B.2).
    Gs [Cc, N, n_a, 24] gathered by i_lin; c_a_flat [Cc, T·T, n_a, K] by slot =
    t_i·T + t_j; Pc [P_a, 24]; dPc [P_a, 3, 24] Cartesian ∇P_c. n-loop (and
    Cartesian loop) so the largest live tensor is [Cc, P_a, 24] (K ≤ 24)."""
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
        b1 ()                               global scalar bias

    Modes (cfg.target_mode):
        0 PES     : E = -Σ_i U_i                            → [1]
        1 Dipole  : μ = -Σ_ij |r_ij|² · (dU_i/dr_ij_vec)    → [3]
        2 Polar.  : α[6] via dual ANN (scalar + tensor)     → [6]

    Mode 2 adds a scalar ANN (W0_pol/b0_pol/W1_pol/b1_pol) giving the isotropic
    diagonal; the primary ANN's forces give the anisotropic virial.
    Output order: [xx, yy, zz, xy, yz, zx].
    """

    def __init__(self,
                 cfg: TNEPconfig,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

        # GPUMD NEP descriptor block (multi_descriptor_plan.md §0.3): raw
        # per-atom basis from data, contracted in-model by c_r/c_a (below).
        # validate_nep_cfg owns every NEP cfg guard.
        self._nep_mode = validate_nep_cfg(cfg)

        # cfg.dim_q is the descriptor width the ANN is built for; nothing
        # overrides it any more, so a saved config whose dim_q disagrees with
        # this build's descriptor predates a descriptor change.
        from DescriptorBuilderGPU import compute_dim_q, soap_dim
        _dim_q_now = int(compute_dim_q(cfg))
        if int(cfg.dim_q) != _dim_q_now:
            raise ValueError(
                f"cfg.dim_q={int(cfg.dim_q)} but this build's descriptor gives "
                f"{_dim_q_now}. The saved config predates a descriptor change "
                f"(removed learned-basis NEP extras / preprocess contraction); "
                f"the model must be retrained.")
        self.dim_q = cfg.dim_q
        # SOAP-only width (the builder's output after soap_from_raw); under
        # nep_descriptor="concat" dim_q is wider by the NEP block.
        self._soap_dim = int(soap_dim(cfg))
        self.num_types = cfg.num_types
        self.num_neurons = cfg.num_neurons
        self._H_final = cfg.num_neurons
        # Any tf.keras.activations.get name is accepted here; _activation_grad
        # raises NotImplementedError at first dipole/pol pass if not plumbed.
        self._activation_name = str(cfg.activation).lower().strip()
        self._glorot_gain = 1.0
        self.activation = tf.keras.activations.get(cfg.activation)

        # Learned radial basis (cfg.radial_basis_size = K): the builder emits
        # raw centre coefficients [.., A, S·K·k_max·2] in the descriptors slot
        # and soap_from_raw finishes the SOAP-turbo descriptor with M_radial.
        _K = getattr(cfg, "radial_basis_size", None)
        self._learn_radial = _K is not None
        if self._learn_radial:
            from DescriptorBuilderGPU import (radial_basis_init,
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
                    "neighbor dipole readout (target_mode=1) "
                    "is supported.")
            self._rb_species_sum = bool(cfg.radial_basis_species_sum)
            self._rb_per_l = bool(cfg.radial_basis_per_l)
            self._rb_norm = str(cfg.radial_basis_norm)
            if self._rb_norm not in ("l2", "none", "sqrt"):
                raise ValueError(
                    f"cfg.radial_basis_norm={cfg.radial_basis_norm!r}; expected "
                    f"'l2', 'none' or 'sqrt'.")
            # Fixed O(1) scale for the raw coefficients: undo SOAP-turbo's 4π
            # and angular amplitude (rcut/σ_t)² they carry.
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
        # NEP contraction tensors c[t_centre, t_neighbour, n, k]: uniform(−1, 1)
        # like GPUMD's snes.cu init; non-trainable like M_radial (SNES assigns,
        # Adam tape-watches). "off": None, and dim_q is the SOAP width.
        self.c_r = self.c_a = None
        self._nep_dim = 0
        self._nep_gram = False
        # "sum" / "gap_sum": ANN_2b + ANN_3b experts added to the SOAP ANN;
        # their inputs are the NEP contraction or the GAP q2 ‖ q3 raw slot.
        self._experts_on = self._nep_mode in ("sum", "gap_sum")
        # Per-expert amplitudes δ = (δ_soap, δ_2b, δ_3b) on the summed head.
        # Exactly redundant with rescaling each expert's W1 — meaningful only
        # because δ is excluded from the L1/L2 penalty on W1 (cfg.sum_delta).
        # Non-trainable tf.Variable like M_radial / c_r: SNES assigns it,
        # Adam tape-watches it. None (and a no-op everywhere) when experts off.
        self.sum_delta = None
        if self._experts_on:
            _d = np.asarray(getattr(cfg, "sum_delta", [1.0, 1.0, 1.0]),
                            dtype=np.float32).reshape(-1)
            if _d.size != 3:
                raise ValueError(
                    f"cfg.sum_delta has {_d.size} entries, expected 3 "
                    f"(δ_soap, δ_2b, δ_3b).")
            self.sum_delta = tf.Variable(_d, trainable=False, name="sum_delta")
        # Phase B: dipole via per-candidate NEP pair forces (plan §B.2).
        self._nep_pair_forces_on = (self._nep_mode in ("concat", "replace")
                                    and cfg.target_mode == 1
                                    and str(cfg.dipole_readout) == "tnep")
        if self._nep_mode == "gap_sum":
            from gap_descriptor import gap_dim
            self._nep_dim = self._nep_raw_width = int(gap_dim(cfg))
        elif self._nep_mode != "off":
            self._nep_dim = int(nep_dim(cfg))
            self._nep_raw_width = int(nep_raw_width(cfg))
            # Data-static 3-body Gram block at the tail of nep_raw: it makes
            # the per-candidate angular contraction one GEMM per centre type
            # instead of a [C, N, n+1, 24] s tensor (nep_gram_on).
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
        # descriptor_scale sees the main ANN's input (dim_q) and, under
        # "sum" / "gap_sum", the expert input after it ([q_soap ‖ q_nep], see
        # gen0_ann_inputs): one slice of cfg.q_scale per typed ANN.
        self._sl_main = slice(0, int(self.dim_q))
        self._scale_width = int(self.dim_q)
        self._expert_in = ()      # (input slice of q_nep, slice of q_scale) per expert
        if self._experts_on:
            if self._nep_mode == "gap_sum":
                from gap_descriptor import gap_slices
                nsl, keys = gap_slices(cfg), ("two_body", "three_body")
            else:
                nsl, keys = nep_slices(cfg), ("radial", "angular")
            ds = self._scale_width
            self._expert_in = tuple(
                (nsl[k], slice(ds + nsl[k].start, ds + nsl[k].stop)) for k in keys)
            self._scale_width += self._nep_dim
        # Descriptor scale preconditioning: m folded into W0 (see _W0_eff);
        # the Keras W0 init is divided by m so the initial function is the
        # unscaled one. SNES's μ init does the same in _build_mu_init.
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
            if m.shape != (self._scale_width,):
                raise ValueError(
                    f"cfg.q_scale has shape {m.shape}, expected "
                    f"({self._scale_width},) — recompute for this descriptor.")
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
            shape=(cfg.num_types, cfg.dim_q, cfg.num_neurons),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b0 : [num_types, num_neurons] — hidden bias per type
        self.b0 = self.add_weight(
            name="b0",
            shape=(cfg.num_types, cfg.num_neurons),
            initializer="zeros",
            trainable=True,
        )

        # Neighbor-basis dipole readout: decided before W1 so the site head
        # can carry its channel axis.
        _readout = str(cfg.dipole_readout)
        self._neighbor_readout = (cfg.target_mode == 1
                                  and _readout == "neighbor")
        self._dipole_channels = int(cfg.dipole_channels)
        self._dipole_radial = str(cfg.dipole_radial)
        self._dipole_zeta = float(cfg.dipole_zeta)
        self._dipole_cutoff = resolved_dipole_cutoff(cfg)
        # Legacy bool → 3-value field: False→"off", True→"symmetric".
        _sp = cfg.dipole_species_pairs
        cfg.dipole_species_pairs = {False: "off", True: "symmetric"}.get(_sp, _sp)
        self._species_pairs = str(cfg.dipole_species_pairs)
        # THE learnability flag: every save/load/SNES-tail site keys on this.
        self._c_learnable = self._species_pairs != "off"
        # dipole_head: "vector" = one head slot per radial channel, "scalar" =
        # one slot broadcast over the channels (_split_head does the tiling).
        self._dipole_head = str(getattr(cfg, "dipole_head", "vector"))
        if self._dipole_head not in ("vector", "scalar"):
            raise ValueError(
                f"cfg.dipole_head must be 'vector' or 'scalar', "
                f"got {cfg.dipole_head!r}.")
        # THE single source for W1's last axis; SNES/model_io read it back.
        # C slots under the vector head, 1 (broadcast in _split_head) under
        # "scalar"; 1 for every non-neighbour readout (the head is a scalar).
        self._head_slots = (self._dipole_channels
                            if self._neighbor_readout
                            and self._dipole_head == "vector" else 1)
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
                                           "gaussian", "screened"):
                raise ValueError(
                    f"cfg.dipole_radial must be 'chebyshev', "
                    f"'chebyshev_warped', 'gaussian' or 'screened', "
                    f"got {self._dipole_radial!r}.")
            if self._dipole_cutoff not in ("cosine2", "cosine", "cubic",
                                           "tanh3"):
                raise ValueError(
                    f"cfg.dipole_cutoff must be 'cosine2', 'cosine', 'cubic', "
                    f"'tanh3' or None (None = the radial family's traditional "
                    f"form), got {cfg.dipole_cutoff!r}.")
            if self._dipole_radial == "screened" and self._dipole_channels != 1:
                # RAISE, don't force: data assembly reads cfg.dipole_channels
                # before the model exists (data.py V build), so silently
                # forcing C=1 here would ship a shape-mismatched V.
                raise ValueError(
                    f"cfg.dipole_radial='screened' is single-channel — set "
                    f"dipole_channels=1 (got {self._dipole_channels}).")
            if self._species_pairs not in ("off", "symmetric", "ordered"):
                raise ValueError(
                    f"cfg.dipole_species_pairs must be 'off', 'symmetric' or "
                    f"'ordered' (legacy bool accepted), got {_sp!r}.")

        # W1 : hidden-to-scalar weights per type — [T, H_final], with an extra
        # radial-channel axis [T, H_final, C] under the neighbor readout.
        self.W1 = self.add_weight(
            name="W1",
            shape=((cfg.num_types, self._H_final, self._head_slots)
                   if self._neighbor_readout
                   else (cfg.num_types, self._H_final)),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b1 : () — global scalar bias shared across all types
        self.b1 = self.add_weight(
            name="b1",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

        # c_pair : packed per-(channel, species pair) coefficients of the
        # neighbor readout — [C, T(T+1)/2] unordered ("off"/"symmetric") or
        # [C, T²] ordered. Learnable (init 1.0, in the SNES tail) when
        # _c_learnable; frozen c ≡ 1 for "off" — the forward always contracts
        # through it either way. _pair_idx maps (t_center, t_neighbor) to the
        # packed slot; _c_full gathers it back to [C, T, T].
        if self._neighbor_readout:
            T_ = cfg.num_types
            if self._species_pairs == "ordered":
                # Ordered index t·T + t' — an asymmetric Ω breaks the
                # (s_i − s_j) difference-form antisymmetry across species.
                pair_idx = np.arange(T_ * T_, dtype=np.int32).reshape(T_, T_)
            else:
                pair_idx = np.zeros((T_, T_), dtype=np.int32)
                p = 0
                for t in range(T_):
                    for t2 in range(t, T_):
                        pair_idx[t, t2] = pair_idx[t2, t] = p
                        p += 1
            self._pair_idx = tf.constant(pair_idx)
            c_shape = (self._dipole_channels, int(pair_idx.max()) + 1)
            if self._c_learnable:
                self.c_pair = self.add_weight(
                    name="c_pair", shape=c_shape,
                    initializer="ones", trainable=True)
            else:
                self.c_pair = tf.ones(c_shape, dtype=tf.float32)

        # Validate dipole contraction power (mode 1 only) at construction time.
        if cfg.target_mode == 1:
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
                # Any positive radius is allowed. The staged V is built from
                # an image-exact neighbour list, so it is correct past L/2 and
                # past rcut_hard; the box-size consequences are WARNED about at
                # data assembly (_neighbor_dipole_guards), and the one path that
                # genuinely cannot serve rc > rcut_hard — the in-graph V built
                # from the SOAP pair stream — refuses in _neighbor_V_coo.
                cfg.dipole_rc = resolved_dipole_rc(cfg)
                _rc = float(cfg.dipole_rc)
                if _rc <= 0.0:
                    raise ValueError(f"cfg.dipole_rc={_rc} must be > 0.")
                if self._dipole_radial == "gaussian":
                    # Gaussian window constants, fixed at init: μ_n = the C
                    # interior points of linspace(0, r_c, C+2); σ = σ-scale ×
                    # spacing r_c/(C+1) (C=1 ⇒ σ-scale·r_c/2, same formula).
                    _C = self._dipole_channels
                    self._gauss_mu = np.linspace(0.0, _rc, _C + 2)[1:-1]
                    self._gauss_sigma = (float(cfg.dipole_gauss_sigma_scale)
                                         * _rc / (_C + 1))
                if float(getattr(cfg, "charge_aux_weight", 0) or 0) > 0:
                    raise ValueError(
                        "charge_aux_weight > 0 is incompatible with "
                        "dipole_readout='neighbor': the site scalar s is an "
                        "electronegativity-like potential, not a charge.")

        # Scalar ANN for polarizability mode (target_mode == 2)
        if cfg.target_mode == 2:
            self.W0_pol = self.add_weight(
                name="W0_pol",
                shape=(cfg.num_types, cfg.dim_q, cfg.num_neurons),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b0_pol = self.add_weight(
                name="b0_pol",
                shape=(cfg.num_types, cfg.num_neurons),
                initializer="zeros",
                trainable=True,
            )
            self.W1_pol = self.add_weight(
                name="W1_pol",
                shape=(cfg.num_types, self._H_final),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b1_pol = self.add_weight(
                name="b1_pol",
                shape=(),
                initializer="zeros",
                trainable=True,
            )

        # Typed single-hidden-layer ANN blocks in SNES order — (suffix, input
        # width, hidden width, q_scale slice): the SOAP ANN, then under
        # nep_descriptor="sum" the 2-body expert on the radial NEP block and
        # the 3-body expert on the angular block, whose head outputs add to
        # the SOAP ANN's before _split_head. W1_r / W1_a start at zero so a
        # fresh model equals the SOAP-only one; b1 stays the one global bias.
        self._ann_blocks = [AnnBlock("", int(cfg.dim_q), int(cfg.num_neurons), self._sl_main)]
        for suf, (sl_in, sl_m) in zip(("_r", "_a"), self._expert_in):
            d_in, He = sl_in.stop - sl_in.start, int(cfg.nep_num_neurons)
            self._ann_blocks.append(AnnBlock(suf, d_in, He, sl_m))
            for name, shape, init in (("W0", (cfg.num_types, d_in, He), "glorot_uniform"),
                                      ("b0", (cfg.num_types, He), "zeros"),
                                      ("W1", (cfg.num_types, He, self._head_slots), "zeros")):
                setattr(self, name + suf, self.add_weight(
                    name=name + suf, shape=shape, initializer=init, trainable=True))

        # Optional descriptor-mixing layer (GPUMD c_nk analog). One padded
        # square block per unordered neighbour-species pair (padded to
        # max_block_size, active size tracked per pair). Per-pair placement
        # matrices P_p ∈ R^{bs×Q} map active features to their non-contiguous
        # q-indices so U_full = Σ_p P_pᵀ · U_pair[...,p,:bs,:bs] · P_p is one
        # einsum broadcasting over any leading batch dims.
        self.descriptor_mixing = bool(getattr(cfg, "descriptor_mixing", False))
        self.descriptor_mixing_per_type = bool(
            getattr(cfg, "descriptor_mixing_per_type", False))
        if self.descriptor_mixing:
            from DescriptorBuilderGPU import descriptor_block_layout
            self._mix_layout = descriptor_block_layout(cfg)
            self._mix_pair_keys = self._mix_layout["pair_keys"]
            self._mix_num_pairs = len(self._mix_pair_keys)
            self._mix_Q = int(self._mix_layout["dim_q"])
            T = cfg.num_types
            # Per-(pair, l) residual blocks: only radial channels at the same l
            # mix, cross-l is forbidden. alpha_eff_per_pair = radial dim; L=l_max+1.
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
            # Stacked projector [num_pairs*L, α, Q] for the batched _U_full fast
            # path when α is uniform (common case): one batched einsum instead
            # of num_pairs×L launches.
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
            # U_pair shape (padded to max_α, padding stays zero):
            #   shared   [num_pairs, L, max_α, max_α]
            #   per-type [T, num_pairs, L, max_α, max_α]
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

        if self._q_scale is not None:
            # Stored W0 = Glorot / m and b0 = (m∘μ)·W0_stored, so W0_eff / b0_eff
            # at init equal the unscaled Glorot / zero init exactly — for every
            # typed ANN on its own slice of q_scale.
            anns = [(getattr(self, "W0" + b.suffix), getattr(self, "b0" + b.suffix), b.scale_slice)
                    for b in self._ann_blocks]
            if cfg.target_mode == 2:
                anns.append((self.W0_pol, self.b0_pol, self._sl_main))
            for W0, b0, sl in anns:
                W0.assign(W0 * (1.0 / self._q_scale[sl, tf.newaxis]))
                b0.assign(tf.einsum('q,tqh->th', self._q_shift_m[sl], W0))

        self.optimizer = SNES(self)

    # ---- descriptor mixing helpers ----------------------------------------

    def _U_full(self, U_pair: tf.Tensor | None = None) -> tf.Tensor:
        """Assemble the full block-diagonal mixing matrix.

        l-aware: (l_max+1) [α_p × α_p] sub-blocks per pair, cross-l forbidden.
            U_full = I_Q + Σ_p Σ_l P_{p,l}ᵀ · V_{p,l} · P_{p,l}
        tf.eye(Q) broadcasts over leading batch dims (single/per-candidate ×
        shared/per-type). V init = 0 ⇒ U_full == I_Q at gen 0.
        """
        V = self.U_pair if U_pair is None else U_pair
        # V: [..., (T?), num_pairs, L, max_α, max_α].
        # Fast path: uniform α. Flatten (num_pairs, L)→PL, one batched einsum
        # over stacked projector P:[PL, α, Q], V:[..., PL, α, α].
        if self._mix_P_ln_stack is not None:
            alpha = self._mix_P_ln_uniform_alpha
            PL = self._mix_num_pairs * self._mix_L
            V_active = V[..., :alpha, :alpha]                # drop padding rows/cols
            # Reshape (num_pairs, L)→PL, preserving leading batch dims (C and/or T).
            new_shape = tf.concat(
                [tf.shape(V_active)[:-4], [PL, alpha, alpha]], axis=0)
            V_flat = tf.reshape(V_active, new_shape)
            # p=PL (pair,l), j/k=α projections, i/m=output q-axes.
            V_full = tf.einsum(
                'pji,...pjk,pkm->...im',
                self._mix_P_ln_stack, V_flat, self._mix_P_ln_stack)
            return tf.eye(self._mix_Q, dtype=V_full.dtype) + V_full
        # Fallback: non-uniform α.
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
        """Pre-multiply W0 by U_fullᵀ along Q — equivalent to mixing the
        descriptor (desc' = U_full · desc) but absorbed into the weights, so the
        forward/backprop/dipole sum use raw descriptors and grad_values unchanged.

        Shapes:
            shared   U_pair [(C,) num_pairs, bs, bs]    + W0 [(C,) T, Q, H] → W0_eff [(C,) T, Q, H]
            per-type U_pair [(C,) T, num_pairs, bs, bs] + W0 [(C,) T, Q, H] → W0_eff [(C,) T, Q, H]

        V init = 0 ⇒ U_full = I ⇒ W0_eff == W0 at gen 0.
        With cfg.descriptor_scale the per-channel m is folded here too, on
        W0's own input (the mixed descriptor): W0_eff = U_fullᵀ·diag(m)·W0,
        so a scaled model is exactly an unscaled one with W0' = m∘W0 for
        any U. This is the ONLY fold site — every entry point must go
        through _W0_eff unconditionally.
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
        """Centring fold of descriptor_scale: the ANN sees m∘(q − μ), so
        b0_eff = b0 − (m∘μ)·W0_stored (W0 BEFORE _W0_eff; any leading
        candidate dim broadcasts). Identity when scaling is off. Call it
        wherever _W0_eff is called."""
        return self._scale_fold(W0_stored, b0)[1]

    def _scale_fold(self, W0: tf.Tensor, b0: tf.Tensor | None,
                    sl: slice | None = None) -> tuple:
        """descriptor_scale fold of one typed ANN whose input is channel
        slice `sl` of the scaler vector (None → the main ANN's): the ANN sees
        m∘(q − μ), so W0_eff = m∘W0 and b0_eff = b0 − (m∘μ)·W0_stored (any
        leading candidate axis broadcasts). Identity when scaling is off.
        The ONE fold site — the SOAP ANN and both "sum" experts use it."""
        if self._q_scale is None:
            return W0, b0
        sl = self._sl_main if sl is None else sl
        W0_eff = W0 * self._q_scale[sl, tf.newaxis]
        b0_eff = (None if b0 is None
                  else b0 - tf.einsum('q,...tqh->...th', self._q_shift_m[sl], W0))
        return W0_eff, b0_eff

    # ---- learned radial basis ---------------------------------------------

    @property
    def descriptor_input_width(self) -> int:
        """Width of the per-atom vector in the `descriptors` slot (see
        DescriptorBuilderGPU.descriptor_input_width). Distinct from
        cfg.dim_q, the width of the W0 matmul."""
        from DescriptorBuilderGPU import descriptor_input_width
        return descriptor_input_width(self.cfg)

    def soap_from_raw(self, raw: tf.Tensor, Z: tf.Tensor,
                      M: tf.Tensor | None = None) -> tf.Tensor:
        """SOAP-turbo descriptor from raw centre coefficients with the learned
        radial basis (cfg.radial_basis_size):
            c[a, (s,n), k] = Σ_κ M[t_a, s, n, κ] · c_raw[a, s, κ, k]
            q[a, (n,n',l)] = Σ_m mult · Re(c_{n l m} conj c_{n' l m})   (trivial
            compression: n or n' a species pivot), then L2-normalised per atom
            with SOAP-turbo's ‖q‖ < 1e-5 guard (padded atoms stay 0).
        cfg.radial_basis_species_sum sums the species axis into α shared
        channels, c[a, n, k] = Σ_s Σ_κ M[t_a, s, n, κ] · c_raw[a, s, κ, k],
        and keeps every (n ≤ n', l) channel (dim_q = α(α+1)/2·L).
        cfg.radial_basis_per_l gives each l block its own M[..., l].
        cfg.radial_basis_norm: "l2" per-atom projection; "none" raw × scale²;
        "sqrt" sign·√|q| × scale (degree 1). Pair the latter two with descriptor_scale.
        Args:
            raw : [..., A, S·K·k_max·2]  builder output (species, κ, k, re/im)
            Z   : [..., A]               type indices
            M   : [T, S, α, K(, L)] (None → self.M_radial) or per-candidate
                  [C, T, S, α, K(, L)]
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
        # Partition atoms by centre type once and contract each type's atoms
        # with its own M[t] (a small GEMM). Never gather M per atom: that
        # [N, S, α, K(, L)] copy is hundreds of MB per candidate at 6 species
        # and its backward makes one copy per l block.
        T = self.num_types
        order = tf.argsort(Zf)
        inv = tf.math.invert_permutation(order)
        counts = tf.math.bincount(tf.gather(Zf, order), minlength=T, maxlength=T)
        parts = tf.split(tf.gather(r, order), counts, num=T, axis=0)
        atom_axis = 1 if per_cand else 0

        def _contract(M_of_t, k_slice):
            """c[(C,) N, n_max, k_len, 2] = Σ_κ M[t_a, s, n, κ] · r[a, s, κ, k_slice]
            (Σ_s too under species_sum), atoms back in input order."""
            outs = []
            for t in range(T):
                M_t = M_of_t(t)                                   # [(C,) S, α, K]
                r_t = parts[t][..., k_slice]                      # [N_t, S, K, kk]
                if self._rb_species_sum:
                    # NEP-style: sum over neighbour species into α shared channels.
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
                # This l block's own radial functions M[..., l] on its k slice.
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
            # NEP-like: no per-atom projection; the spectrum is quadratic in
            # the coefficients, so scale² puts it on an O(1) footing.
            q = q * (sc * sc)
        else:
            # "sqrt": degree 1 in the coefficients (like NEP's radial block):
            # halves every log-spread of "none", keeps magnitude.
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
        Args:
            raw : [..., A, W_raw]  flatten([raw_r [T, K_r+1] ‖ raw_a [T, K_a+1, 24]])
            Z   : [..., A]         type indices
            c_r, c_a : [T, T, n+1, K+1] (None → self.c_r / self.c_a) or
                       per-candidate [C, T, T, n+1, K+1]
        Returns [..., A, dim_nep], or [C, ..., A, dim_nep] for per-candidate c;
        with return_s also s [(C,) ..., A, n_a+1, 24] (the tnep pair forces).
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
            # The angular block of nep_raw IS the 3-body Gram (nep_gram_on):
            # one GEMM per centre type, no [C, N, n+1, 24] s tensor at all.
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
        """(inputs, s): the [..., A, dim_q] vector the ANN consumes — the one
        place that knows cfg.nep_descriptor: "off" → descriptors (same
        object); "concat" → [descriptors ‖ nep_from_raw]; "replace" →
        nep_from_raw — and s from nep_from_raw (the tnep pair forces; None
        when no NEP block is formed). `descriptors` is the builder output
        after soap_from_raw."""
        if self._nep_mode == "off":
            return descriptors, None
        if nep_raw is None:
            raise ValueError(
                f"nep_raw missing: cfg.nep_descriptor={self._nep_mode!r} needs "
                f"the raw expert input [.., W_raw] (the NEP basis, or GAP q2‖q3 "
                f"under gap_sum) at every forward site.")
        if self._experts_on:              # experts read nep_raw in _expert_outputs
            return descriptors, None
        # s is consumed ONLY by the tnep pair-force contraction; asking for it
        # elsewhere would force the slow path and build a [C, N, n, 24] tensor.
        if self._nep_pair_forces_on:
            q_nep, s = self.nep_from_raw(nep_raw, Z, c_r, c_a, return_s=True)
        else:
            q_nep, s = self.nep_from_raw(nep_raw, Z, c_r, c_a), None
        if self._nep_mode == "replace":
            return q_nep, s
        # One side may carry a candidate axis (learned SOAP basis with shared
        # c, or per-candidate c with fixed SOAP): broadcast the other to it.
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
    def expert_weights(self) -> tuple:
        """The six "sum"-mode expert tensors in EXPERT_WEIGHTS order."""
        return tuple(getattr(self, k) for k in EXPERT_WEIGHTS)

    def _expert_ann(self, q: tf.Tensor, Z: tf.Tensor, atom_mask: tf.Tensor,
                    W0: tf.Tensor, b0: tf.Tensor, W1: tf.Tensor,
                    sl_in: slice, sl_m: slice) -> tf.Tensor:
        """One summed expert: activation(q[.., sl_in]·W0[t] + b0[t])·W1[t] →
        [(C,) ..., A, head_slots]. q [(C,) ..., A, dim_nep]; Z / atom_mask
        [..., A]; STORED weights [(C,) T, ...] — the descriptor_scale fold on
        slice sl_m happens here (no mixing reaches the experts,
        so no caller pre-folds). Type-mask sum, no [.., A, d, H] gather.
        Per-candidate q (c_r / c_a with a candidate axis) requires
        per-candidate experts with the same C: the einsum names both axes
        'c' (no broadcast); shared q with per-candidate experts is fine."""
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
                        sum_delta: tf.Tensor | None = None) -> tf.Tensor:
        """δ_2b·ANN_2b(q_r) + δ_3b·ANN_3b(q_ang) head outputs
        [(C,) ..., A, head_slots] under nep_descriptor="sum" / "gap_sum"
        (q2 / q3), added to δ_soap·(the SOAP ANN's) before _split_head.
        experts: the six STORED tensors in EXPERT_WEIGHTS order (optional
        leading candidate axis); None → the model's variables. sum_delta:
        [3] or per-candidate [C, 3]; None → self.sum_delta."""
        q = nep_raw if self._nep_mode == "gap_sum" else self.nep_from_raw(nep_raw, Z, c_r, c_a)
        w = self.expert_weights if experts is None else tuple(experts)
        d = self.sum_delta if sum_delta is None else sum_delta
        return tf.add_n([
            self._delta_scale(d, i + 1,
                              self._expert_ann(q, Z, atom_mask, *w[3 * i:3 * i + 3], *sl))
            for i, sl in enumerate(self._expert_in)])

    def _head_with_experts(self, out: tf.Tensor, nep_raw: tf.Tensor, Z: tf.Tensor,
                           atom_mask: tf.Tensor, c_r: tf.Tensor | None = None,
                           c_a: tf.Tensor | None = None, experts: tuple | None = None,
                           sum_delta: tf.Tensor | None = None) -> tf.Tensor:
        """The summed head: δ_soap·ANN_soap + δ_2b·ANN_2b + δ_3b·ANN_3b.
        Called at every forward site when self._experts_on."""
        d = self.sum_delta if sum_delta is None else sum_delta
        return (self._delta_scale(d, 0, out)
                + self._expert_outputs(nep_raw, Z, atom_mask, c_r, c_a, experts, d))

    def predict(self, descriptors: tf.Tensor, gradients: tf.Tensor, grad_index: tf.Tensor,
                positions: tf.Tensor, Z: tf.Tensor, box: tf.Tensor,
                atom_mask: tf.Tensor, neighbor_mask: tf.Tensor,
                M_radial: tf.Tensor | None = None,
                nep_raw: tf.Tensor | None = None,
                c_r: tf.Tensor | None = None,
                c_a: tf.Tensor | None = None) -> tf.Tensor:
        """Run the forward pass for a single structure using padded tensors.

        Not available for the NEP tnep readout (cfg.nep_descriptor != "off"
        with dipole_readout="tnep"): the padded [A, M] table is the
        rcut_hard-limited MIC SOAP list and cannot serve the 8 Å image-exact
        NEP pair stream — use predict_batch (score / validate / fit do).

        Args:
            descriptors    : [A, dim_q]     padded per-atom SOAP descriptors
                             ([A, W_in] raw centre coefficients under
                             cfg.radial_basis_size; see soap_from_raw)
            gradients      : [A, M, 3, dim_q]  padded descriptor gradients
            grad_index     : [A, M]         padded neighbor indices
            positions      : [A, 3]         padded atom positions
            Z              : [A]            padded integer type indices
            box            : [3, 3]         lattice vectors
            atom_mask      : [A]            1.0 for real atoms, 0.0 for padding
            neighbor_mask  : [A, M]         1.0 for real neighbors, 0.0 for padding

        Returns:
            target_mode 0: [1]  total energy
            target_mode 1: [3]  dipole vector
            target_mode 2: [6]  polarizability tensor
        """
        if self._nep_pair_forces_on:
            raise NotImplementedError(
                "predict() (padded single-structure path) has no NEP pair "
                "stream; use predict_batch for the NEP tnep readout.")
        if self._learn_radial:
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)
        descriptors = self._ann_inputs(descriptors, nep_raw, Z, c_r, c_a)[0]
        # Absorb U_pairᵀ into W0 once so forward and calc_forces share the
        # folded weights (no-op when mixing is off).
        W0_eff = self._W0_eff(self.W0)
        b0_eff = self._b0_eff(self.b0, self.W0)

        # Gather per-type weights for each atom
        W0_t = tf.gather(W0_eff, Z)    # [A, dim_q, H]
        b0_t = tf.gather(b0_eff, Z)    # [A, H]
        W1_t = tf.gather(self.W1, Z)   # [A, H]

        # Hidden layer: h = activation(z), z = q @ W0[t] + b0[t]. Keep z for the
        # swish backward chain (tanh needs only 1 − h²).
        z = tf.einsum('nd,ndh->nh', descriptors, W0_t) + b0_t   # [A, H]
        h = self.activation(z)                                   # [A, H]
        # Mask out padded atoms
        h = h * atom_mask[:, tf.newaxis]                   # [A, H]

        if self.cfg.target_mode == 0:
            # PES: E = -sum_i (h_i . W1[t_i] + b1)
            E_per_atom = tf.reduce_sum(h * W1_t, axis=1) + self.b1  # [A]
            E_per_atom = E_per_atom * atom_mask                       # zero padding
            E = tf.reduce_sum(E_per_atom)
            out = tf.expand_dims(-E, axis=0)  # [1]
            return out

        if self._neighbor_readout:
            # Neighbor-basis readout, C Chebyshev channels + species-pair c:
            #   μ = Σ_i Σ_n s_i^n Σ_t' c[t_i,t',n]·V_i[:,n,t'],
            #   V_i[:,n,t'] = Σ_{j∈t'} R_n(|d|)·d,  d = r_i − r_j (MIC).
            # No de_dq backward chain here — the readout reads no descriptor
            # gradients; don't re-add it. b1 stays excluded from s: with c
            # symmetric a per-channel uniform shift of s^n is exact gauge;
            # under dipole_species_pairs="ordered" that identity breaks and
            # the exclusion is a modeling choice — keep it, don't "fix" it.
            out = tf.einsum('ah,ahk->ak', h, W1_t)                    # [A, S]
            if self._experts_on:
                out = self._head_with_experts(out, nep_raw, Z, atom_mask, c_r, c_a)
            s = self._split_head(out)                                 # [A, C]
            dr, rij = self._neighbor_displacements_single(
                positions, box, grad_index)          # dr = neighbor − center
            rc = float(self.cfg.dipole_rc)
            center = tf.range(tf.shape(grad_index)[0],
                              dtype=grad_index.dtype)[:, tf.newaxis]
            keep = tf.logical_and(rij < rc, tf.not_equal(grad_index, center))
            w = tf.cast(keep, rij.dtype) * neighbor_mask              # [A, M]
            Rw = self._radial(rij) * w[:, :, tf.newaxis]           # [A, M, C]
            tp = tf.one_hot(tf.gather(Z, grad_index), self.num_types,
                            dtype=tf.float64)                      # [A, M, T]
            V = tf.einsum('amn,amd,amt->adnt',                # float64 accum.
                          tf.cast(Rw, tf.float64),
                          tf.cast(-dr, tf.float64), tp)      # [A, 3, C, T]
            c_full = self._c_full(self.c_pair)                        # [C, T, T]
            c_at = tf.gather(c_full, Z, axis=1)                       # [C, A, T]
            mu = tf.einsum('an,nat,adnt->d',
                           s, c_at, tf.cast(V, s.dtype))                 # [3]
            return mu

        # Modes 1 and 2 need forces
        forces = self.calc_forces(h, gradients, W1_t, W0_t, neighbor_mask,
                                  z=z)  # [A, M, 3]

        if self.cfg.target_mode == 1:
            # Dipole. cfg.dipole_rij_power selects the per-pair weight:
            #   N >= 1 : μ = -Σ_pair |r_ij|^N · F_ij
            #   N == 0 : μ = -Σ_i de_dq[i] · grad_values[i, i]  (self-only)
            _, rij = self._neighbor_displacements_single(
                positions, box, grad_index)
            rij_n = (self._dipole_pair_weight_padded(tf.square(rij), grad_index)
                     * neighbor_mask)                                     # [A, M]
            dipole_contribs = rij_n[:, :, tf.newaxis] * forces            # [A, M, 3]
            dipole = -tf.reduce_sum(dipole_contribs, axis=[0, 1])         # [3]
            return dipole

        elif self.cfg.target_mode == 2:
            # Polarizability via dual ANN (GPUMD approach)
            dr_gathered, _ = self._neighbor_displacements_single(positions, box, grad_index)

            # --- Scalar ANN (isotropic) ---
            W0_pol_eff = self._W0_eff(self.W0_pol)
            W0p_t = tf.gather(W0_pol_eff, Z)   # [A, dim_q, H]
            b0p_t = tf.gather(self.b0_pol, Z)  # [A, H]
            W1p_t = tf.gather(self.W1_pol, Z)  # [A, H]

            h_pol = tf.einsum('nd,ndh->nh', descriptors, W0p_t)  # [A, H]
            h_pol = h_pol + b0p_t
            h_pol = self.activation(h_pol)
            h_pol = h_pol * atom_mask[:, tf.newaxis]
            F_pol = tf.reduce_sum(h_pol * W1p_t, axis=1) + self.b1_pol  # [A]
            F_pol = F_pol * atom_mask
            scalar_sum = tf.reduce_sum(F_pol)

            # --- Tensor ANN (anisotropic virial) ---
            pol_outer = -tf.einsum('nma,nmb->nmab', dr_gathered, forces)  # [A, M, 3, 3]
            pol_outer = pol_outer * neighbor_mask[:, :, tf.newaxis, tf.newaxis]
            pol_matrix = tf.reduce_sum(pol_outer, axis=[0, 1])  # [3, 3]

            # Extract 6 unique components: [xx, yy, zz, xy, yz, zx]
            pol = tf.stack([
                pol_matrix[0, 0],
                pol_matrix[1, 1],
                pol_matrix[2, 2],
                pol_matrix[0, 1],
                pol_matrix[1, 2],
                pol_matrix[2, 0],
            ])

            # Add scalar ANN to diagonal
            pol = pol + tf.stack([scalar_sum, scalar_sum, scalar_sum,
                                  0.0, 0.0, 0.0])
            return pol

    def _activation_grad(self, h: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """dh/dz for the activation, applied as a Hadamard factor in the dipole/pol
        backward chain.
            tanh : 1 − h²                        (needs only h)
            swish: σ(z) · (1 + z · (1 − σ(z)))   (needs z; this form avoids the
                   z≫0 cancellation of the equivalent σ(z) + h·(1−σ(z)))

        Args:
            h: activation output, broadcastable to z.
            z: pre-activation (W0·q + b0). Required for swish; None only for tanh.
        Returns:
            tensor shaped like h.
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
        """Compute dU_i/dR_j for every atom i and neighbour j via chain rule (padded).

        Args:
            h              : [N, H]              hidden activations f(z)
            gradients      : [N, M, 3, dim_q]   padded descriptor gradients
            W1_t           : [N, H]              per-atom output weights
            W0_t           : [N, dim_q, H]       per-atom input weights
            neighbor_mask  : [N, M]              1.0 for real neighbors, 0.0 for padding
            z              : [N, H]              pre-activation; required for swish.

        Returns:
            forces : [N, M, 3]  dU_i/dR_j per atom per neighbor
        """
        # dU/dh * dh/da = W1 * f'(z)
        dact = self._activation_grad(h, z)                       # [N, H]
        de_da = dact * W1_t                                       # [N, H]
        # dU/dq = dU/da @ W0^T  ->  [N, dim_q]
        de_dq = tf.einsum('nh,nqh->nq', de_da, W0_t)             # [N, dim_q]
        # Contract dU/dq with dq/dR_j: sum over dim_q
        forces = tf.einsum('nq,nmcq->nmc', de_dq, gradients)     # [N, M, 3]
        # Zero out padded neighbors
        forces = forces * neighbor_mask[:, :, tf.newaxis]
        return forces

    def fit(self, train_data: dict[str, tf.Tensor], val_data: dict[str, tf.Tensor],
            plot_callback: Callable | None = None,
            resume_state: dict | None = None) -> dict:
        """Train the model using the SNES evolutionary optimizer.

        Args:
            train_data    : dict with keys descriptors, gradients, grad_index,
                            positions, Z_int, targets, boxes (lists over structures)
            val_data      : same structure, used for validation each generation
            plot_callback : optional callable(history, gen) for periodic plotting
            resume_state  : optional dict from `model_io.load_checkpoint`
                            (SNES distribution + best-val + history + RNG). When
                            given, training continues from last_gen + 1.

        Returns:
            history        : dict of generation, train_loss, val_loss (lists)
            final_model    : model at the last generation
            best_val_model : model at the best-validation generation
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
        """Evaluate RMSE, R², per-component R², and cosine similarity.

        Args:
            test_data : dict with COO tensors from pad_and_stack()

        Returns:
            metrics : dict with keys:
                rmse          : scalar float — overall RMSE
                r2            : scalar float — overall R²
                r2_components : [T] tensor — per-component R²
                cos_sim_mean  : scalar float — mean cosine similarity (modes 1,2 only)
                cos_sim_all   : [S] tensor — per-structure cosine similarity (modes 1,2)
            preds : [S, T] tensor of predictions
        """
        # Streaming chunked scoring; peak memory bounded to one chunk's grads.
        from data import prefetched_chunks
        S_test = test_data["num_atoms"].shape[0]
        chunk_sz = (self.cfg.batch_chunk_size
                    if self.cfg.batch_chunk_size is not None else S_test)
        # Pre-fold U_pairᵀ into W0 (and W0_pol) once so every chunk shares the
        # absorbed weights (no-op when mixing is off).
        W0_eff = self._W0_eff(self.W0)
        b0_eff = self._b0_eff(self.b0, self.W0)
        W0_pol_eff = (self._W0_eff(self.W0_pol)
                      if (self.cfg.target_mode == 2
                          and getattr(self, "W0_pol", None) is not None)
                      else getattr(self, "W0_pol", None))
        b0_pol_eff = (self._b0_eff(self.b0_pol, self.W0_pol)
                      if self.cfg.target_mode == 2 else getattr(self, 'b0_pol', None))
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
                W0_pol_eff,
                b0_pol_eff,
                getattr(self, 'W1_pol', None),
                getattr(self, 'b1_pol', None),
                W_atom=chunk.get("V_atom"),  # neighbor readout: [B,A,3] kernel
                nep_raw=chunk.get("nep_raw"), **nep_pair_kwargs(chunk),
            ))
            del chunk
        raw_preds = tf.concat(pred_parts, axis=0)
        del pred_parts
        targets = test_data["targets"]

        # Normalize predictions to per-atom space when target scaling is active
        if self.cfg.scale_targets and self.cfg.target_mode == 1 and "num_atoms" in test_data:
            num_atoms = tf.cast(test_data["num_atoms"], tf.float32)  # [S]
            num_atoms_col = tf.maximum(num_atoms, 1.0)[:, tf.newaxis]  # [S, 1]
            preds = raw_preds / num_atoms_col
        else:
            preds = raw_preds

        diff = preds - targets
        mse = tf.reduce_mean(tf.square(diff))
        rmse = tf.sqrt(tf.maximum(mse, 0.0))

        # Overall R² = 1 - SS_res / SS_tot
        ss_res = tf.reduce_sum(tf.square(diff))
        ss_tot = tf.reduce_sum(tf.square(targets - tf.reduce_mean(targets, axis=0)))
        r2 = 1.0 - ss_res / tf.maximum(ss_tot, 1e-12)   # guard degenerate (e.g. 1-structure) sets

        # Per-component R²
        ss_res_comp = tf.reduce_sum(tf.square(diff), axis=0)       # [T]
        ss_tot_comp = tf.reduce_sum(
            tf.square(targets - tf.reduce_mean(targets, axis=0)), axis=0)  # [T]
        r2_components = 1.0 - ss_res_comp / tf.maximum(ss_tot_comp, 1e-12)

        metrics = {
            "rmse": rmse,
            "r2": r2,
            "r2_components": r2_components,
        }

        # Total (un-scaled) metrics when target scaling is active.
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

        # Cosine similarity for vector targets (modes 1 and 2)
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
        """Score directly on an XYZ file or directory, via the standard pipeline
        (collect → descriptors → assemble → pad_and_stack) then :py:meth:`score`.

        Training-time cfg.num_types/types/dim_q are preserved so the descriptor
        layout matches; test types are re-indexed into the training species
        ordering (unseen species raise KeyError).

        Args:
            path: single ``.xyz`` or a directory (all ``.xyz`` concatenated, sorted).
            allowed_species: override cfg.allowed_species; None keeps the filter.
            max_structures: cap on structures; None = no cap.
            filter_mode: override cfg.filter_mode ("subset"/"exact"); None keeps it.
            pin_to_cpu: forwarded to pad_and_stack; True keeps test data on host RAM.
            plot: generate parity + error-vs-magnitude figures after scoring.
            save_plots: dir to write figures (auto-created); None skips saving.
            show_plots: show via matplotlib backend; False for headless.
            suffix: string appended to figure filenames.
            presentation: poster-style flag, currently a no-op (plumbed via cfg).
            shared_axis_scale: True → all parity panels share one x/y range from
                the joint min/max (useful when one component dominates).

        Returns:
            (metrics, preds) as :py:meth:`score` — preds is [S, T_dim].
        """
        import os
        import copy
        from ase.io import read as _ase_read
        from data import collect, assemble_data_dict, pad_and_stack
        from DescriptorBuilder import make_descriptor_builder

        cfg = self.cfg

        # Snapshot training-time species + dim so the layout stays as expected.
        TRAIN_NUM_TYPES = int(cfg.num_types)
        TRAIN_TYPES     = list(cfg.types)
        TRAIN_DIM_Q     = int(cfg.dim_q)

        # Resolve the path. Directory → concatenate all *.xyz children
        # via a tempfile so the existing `collect` reader sees one set.
        cfg_for_load = copy.copy(cfg)
        _tmp_xyz = None     # only set when we materialise a concat tempfile
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

        # Apply optional overrides on the temp cfg only — the model's
        # own cfg.attributes are restored before exit.
        if allowed_species is not None:
            cfg_for_load.allowed_species = list(allowed_species)
        if filter_mode is not None:
            cfg_for_load.filter_mode = filter_mode
        if max_structures is not None:
            cfg_for_load.total_N = int(max_structures)

        # collect overrides num_types from the data; restore below so the builder
        # produces the trained dim_q, not the data-implied one.
        print(f"score_from_file: loading {path} ...")
        dataset, ti_loaded = collect(cfg_for_load)
        if max_structures is not None and len(dataset) > max_structures:
            dataset = dataset[:max_structures]

        # Restore training-time species so the descriptor builder produces
        # the model's expected layout.
        cfg_for_load.num_types = TRAIN_NUM_TYPES
        cfg_for_load.types     = TRAIN_TYPES
        cfg_for_load.dim_q     = TRAIN_DIM_Q

        # Re-index per-atom type arrays using the training-time species
        # ordering so per-type ANN routing puts each atom on the correct head.
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

        # Build descriptors using the model's training-time SOAP layout.
        print(f"  building descriptors for {len(dataset)} structures ...")
        # Same gradient gating as data.materialize_test_data (score() is the
        # same consumer): no SOAP gradients for the neighbour readout
        # or tnep under nep_descriptor="replace".
        from data import soap_gradients_needed, dipole_self_pairs_only
        builder = make_descriptor_builder(cfg_for_load)
        descs, grads, gidx = builder.build_descriptors(
            dataset, calc_gradients=soap_gradients_needed(cfg_for_load))

        # Assemble + pad. Descriptors are now [N_i, TRAIN_DIM_Q] per structure.
        data_dict = assemble_data_dict(
            dataset, ti, descs, grads, gidx, cfg_for_load)
        test_data = pad_and_stack(
            data_dict, num_types=TRAIN_NUM_TYPES, pin_to_cpu=pin_to_cpu,
            self_pairs_only=dipole_self_pairs_only(cfg_for_load))

        print(f"  scoring on {test_data['descriptors'].shape[0]} structures ...")
        try:
            metrics, preds = self.score(test_data)
            if plot:
                # Lazy import: keeps matplotlib out of module import.
                from plotting import plot_correlation, plot_error_vs_magnitude
                if save_plots is not None:
                    os.makedirs(save_plots, exist_ok=True)
                targets_np = test_data["targets"].numpy()
                preds_np   = preds.numpy()
                # plot_correlation needs an RRMSE entry score() doesn't produce;
                # inject RMSE / target-std on the reported (per-atom) space.
                diff = targets_np - preds_np
                std_overall = max(float(targets_np.std()), 1e-12)
                std_comp = np.maximum(targets_np.std(axis=0), 1e-12)
                rmse_comp = np.sqrt(np.mean(diff ** 2, axis=0))
                metrics_plot = dict(metrics)
                metrics_plot["rrmse"] = float(metrics["rmse"]) / std_overall
                metrics_plot["rrmse_components"] = rmse_comp / std_comp
                # cfg carries plot_units; stash presentation on cfg as
                # _presentation_mode for future plot paths (no-op today).
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
                finally:
                    # Restore/remove the presentation flag.
                    if _prev_pres is None:
                        try:
                            delattr(self.cfg, "_presentation_mode")
                        except AttributeError:
                            pass
                    else:
                        self.cfg._presentation_mode = _prev_pres
            return metrics, preds
        finally:
            # Clean up the concat tempfile if we made one.
            if _tmp_xyz is not None:
                try:
                    os.unlink(_tmp_xyz)
                except OSError:
                    pass

    def score_summary(self, test_data: dict[str, tf.Tensor]) -> dict:
        """Print and return a comparison-ready scoring summary.

        Reports metrics in BOTH per-atom space (GPUMD loss.out rmse_virial) and
        total-dipole space (NEP-paper headline RMSE/R²). RRMSE = √(1 − R²) is the
        centered definition, differing from the un-centered SNES training RRMSE
        (Σy² denominator) — nearly equal only when target means are small.
        """
        metrics, preds = self.score(test_data)
        m = {k: (float(v.numpy()) if hasattr(v, "numpy") else float(v))
             for k, v in metrics.items()
             if v is not None and getattr(v, "shape", ())  == ()}
        # Per-atom (always populated):
        rrmse_pa = (1.0 - m["r2"]) ** 0.5
        print(f"  PER-ATOM space (matches GPUMD loss.out 'rmse_virial'):")
        print(f"    RMSE  = {m['rmse']:.6f}  e·bohr / atom / component")
        print(f"    R²    = {m['r2']:.6f}")
        print(f"    RRMSE = √(1−R²) = {rrmse_pa:.4%}")
        # Total (only when scale_targets active):
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
                      W0_pol: tf.Tensor | None = None, b0_pol: tf.Tensor | None = None,
                      W1_pol: tf.Tensor | None = None, b1_pol: tf.Tensor | None = None,
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
        """Batched forward pass for B structures using COO gradient storage.

        Weights are passed explicitly (not read from self) so this method can
        be used for SNES population evaluation with different candidate weights.

        Args:
            descriptors : [B, A, Q]    padded descriptors
            grad_values : [P, 3, Q]    COO gradient blocks (P = total pairs in batch)
            pair_atom   : [P]          center atom index for each pair
            pair_gidx   : [P]         neighbor atom index for each pair
            pair_struct : [P]          batch-relative structure index for each pair
            positions   : [B, A, 3]   padded positions
            Z           : [B, A]      padded type indices
            boxes       : [B, 3, 3]   lattice vectors
            atom_mask   : [B, A]      atom mask (1.0 real, 0.0 padding)
            W0          : [T, Q, H]   input weights
            b0          : [T, H]      hidden bias
            W1          : [T, H]      output weights ([T, H, C] neighbor readout)
            b1          : ()          scalar bias
            W0_pol..b1_pol : same shapes, for mode 2 only (None otherwise)
            W_atom      : dipole kernel [B, A, 3, Q]; neighbor readout: V
                          [B, A, 3, C, T] instead (built in-graph when None)
            c_pair      : [C, T(T+1)/2] species-pair coefficients (neighbor
                          readout; None → self.c_pair)
            M_radial    : [T, S, α, K] learned radial basis (None → self.M_radial)
            nep_raw     : [B, A, W_raw] raw expert input (cfg.nep_descriptor
                          != "off"): the NEP basis, or GAP q2‖q3 under gap_sum
            c_r, c_a    : NEP contraction tensors (None → self.c_r / self.c_a)
            experts     : "sum" mode: the six STORED expert tensors
                          (EXPERT_WEIGHTS order; None → model variables)
            sum_delta   : [3] per-expert amplitudes (δ_soap, δ_2b, δ_3b);
                          None → self.sum_delta
            nep_pair_*  : image-exact NEP pair stream (data.py nep_pairs; the
                          tnep readout under NEP): atom/gidx/struct [P],
                          D [P, 3] = r_j − r_i incl. image, ang [P] bool

        Returns:
            predictions : [B, T_dim]  T_dim = 1 (PES), 3 (dipole), 6 (pol)
        """
        if self._learn_radial:
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)
        descriptors, s_nep = self._ann_inputs(descriptors, nep_raw, Z, c_r, c_a)
        box_inv = tf.linalg.inv(boxes)  # [B, 3, 3]

        # Pure forward primitive: caller decides whether W0/W0_pol are raw or
        # already U-absorbed, so no double-mixing here.
        W0_use = W0
        W0_pol_use = W0_pol

        b0_t = tf.gather(b0, Z)   # [B, A, H]
        W1_t = tf.gather(W1, Z)   # [B, A, H_final]

        # Per-type loop for W0 avoids materialising [B, A, Q, H] (memory-dominant);
        # b0/W1 are only [B, A, H] so their gathers are fine.
        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(self.num_types)
        ]
        # z1 preserved for the swish backward chain.
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
            # Neighbor-basis readout (C channels + species-pair c):
            #   μ[b] = Σ_{a,n,t'} s[b,a,n]·c[t_a,t',n]·V[b,a,:,n,t'].
            # The de_dq backward chain is skipped entirely — the readout
            # reads no descriptor gradients; don't re-add it. b1 stays
            # excluded from s: exact gauge with c symmetric; a modeling
            # choice (kept deliberately) under "ordered" Ω.
            if c_pair is None:
                c_pair = self.c_pair  # model-weight callers (score, spectro)
            out = tf.einsum('bah,bahk->bak', h1, W1_t)               # [B, A, S]
            if self._experts_on:
                out = self._head_with_experts(out, nep_raw, Z, atom_mask, c_r, c_a,
                                              experts, sum_delta)
            s = self._split_head(out)                                # [B, A, C]
            V = W_atom  # [B, A, 3, C, T] geometry kernel via the W_atom slot
            if V is None:
                # Full-pair callers (spectroscopy fused / legacy pack):
                # build V in-graph from the r_d pair stream (r_c ≤ rcut_hard).
                V = self._neighbor_V_coo(
                    positions, boxes, box_inv, pair_struct, pair_atom,
                    pair_gidx, Z, tf.shape(descriptors)[0],
                    tf.shape(descriptors)[1])
            c_full = self._c_full(c_pair)                         # [C, T, T]
            c_at = tf.gather(c_full, Z, axis=1)                   # [C, B, A, T]
            mu = tf.einsum('ban,nbat,badnt->bd', s, c_at, V)      # [B, 3]
            return mu

        # Single-hidden backward chain (∂U/∂q):
        #   ∂U/∂h1 = W1, ∂U/∂a1 = activation'(h1, z1)·W1, ∂U/∂q = ∂U/∂a1·W0^T
        de_da = self._activation_grad(h1, z1) * W1_t
        de_dq = tf.add_n([
            tf.einsum('bah,qh->baq', de_da, W0_use[t]) * type_masks[t]
            for t in range(self.num_types)
        ])

        B = tf.shape(descriptors)[0]

        if self._nep_pair_forces_on:
            # NEP columns → per-pair forces on the image-exact stream; SOAP
            # columns (concat) → the unchanged W_atom / COO path below.
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
            # Precomputed-kernel path (avoids [C, P, Q]):
            # dipole[b,s] = -Σ_{a,q} de_dq[b,a,q]·W_atom[b,a,s,q], identical to
            # the COO forces path by W_atom's definition.
            return -tf.einsum('baq,basq->bs', de_dq, W_atom)  # [B, 3]

        # Standard COO path (W_atom not precomputed: score(), predict()).
        forces_per_pair = self._calc_forces_coo(de_dq, grad_values, pair_struct, pair_atom)

        if self.cfg.target_mode == 1:
            return self._dipole_coo(forces_per_pair, pair_struct, pair_atom, pair_gidx,
                                    positions, boxes, box_inv, B)

        elif self.cfg.target_mode == 2:
            # Scalar ANN uses the U-absorbed W0_pol_use (same feature space as
            # the main ANN); raw descriptors ≡ feeding desc_mixed into raw W0_pol.
            return self._polarizability_coo(
                descriptors, forces_per_pair, pair_struct, pair_atom, pair_gidx,
                positions, boxes, box_inv, Z, atom_mask,
                W0_pol_use, b0_pol, W1_pol, b1_pol, B)

        else:
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
        """Forward pass for C candidates × B structures using explicit batched GEMMs.

        Replaces vectorized_map for target_mode 0 (PES) and 1 (dipole).
        Both the input→hidden and hidden→descriptor matmuls are executed as
        single large GEMMs over all C candidates simultaneously:

            Forward:  [B*A, Q] @ [Q, C*H]    → [B*A, C*H] → [C, B, A, H]
            Backward: [C, B*A, H] @ [C, H, Q] → [C, B*A, Q]  (batched GEMM)

        Descriptors are assumed pre-scaled by the caller.

        Args:
            descriptors : [B, A, Q]
            W_atom      : [B, A, 3, Q]  precomputed dipole kernel (mode 1 only);
                          neighbor readout: V [B, A, 3, Ch, T] geometry kernel
            Z           : [B, A]        type indices
            atom_mask   : [B, A]        1.0 real, 0.0 pad
            W0          : [C, T, Q, H]
            b0          : [C, T, H]
            W1          : [C, T, H]     ([C, T, H, Ch] neighbor readout)
            b1          : [C]
            c_pair      : [C, Ch, T(T+1)/2] per-candidate species-pair
                          coefficients (neighbor readout only)
            M_radial    : [C, T, S, α, K] per-candidate learned radial basis
                          (required under cfg.radial_basis_size; descriptors
                          are then raw centre coefficients [B, A, W_in])
            nep_raw     : [B, A, W_raw] raw expert input (cfg.nep_descriptor
                          != "off"): the NEP basis, or GAP q2‖q3 under gap_sum
            c_r, c_a    : per-candidate [C, T, T, n+1, K+1] NEP contraction
                          tensors (None → the model's shared c_r / c_a)
            experts     : "sum" mode: per-candidate [C, ...] STORED expert
                          tensors (EXPERT_WEIGHTS order; None → the model's)
            sum_delta   : per-candidate [C, 3] amplitudes (δ_soap, δ_2b, δ_3b)
                          or shared [3]; None → the model's self.sum_delta
            nep_pair_*  : image-exact NEP pair stream (see predict_batch); the
                          tnep readout under NEP contracts it per candidate

        Returns:
            predictions : [C, B, T_dim]  T_dim = 1 (PES) or 3 (dipole)
        """
        Q = self.dim_q
        H = self.num_neurons
        H_final = self._H_final
        T = self.num_types

        B = tf.shape(descriptors)[0]
        A = tf.shape(descriptors)[1]
        C = tf.shape(W0)[0]

        # Type masks [B, A, 1] — C-independent, reused for both matmul directions.
        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(T)
        ]

        # Per-candidate mixing: absorb U_pairᵀ into W0 so forward/backward stay
        # in raw-desc space (de_dq combines with raw grad_values). See _W0_eff.
        b0 = self._b0_eff(b0, W0)          # centring fold needs the STORED W0
        W0 = self._W0_eff(W0, U_pair)

        # ── Forward: input→hidden ─────────────────────────────────────────────
        pre_h_terms = []
        if self._learn_radial:
            if M_radial is None:
                raise ValueError(
                    "predict_batch_candidates needs per-candidate M_radial "
                    "[C, T, S, α, K] under cfg.radial_basis_size.")
            descriptors = self.soap_from_raw(descriptors, Z, M_radial)         # [C, B, A, Q]
        descriptors, s_nep = self._ann_inputs(descriptors, nep_raw, Z, c_r, c_a)
        if len(descriptors.shape) == 4:
            # Per-candidate descriptors [C, B, A, Q] (learned radial basis or
            # per-candidate c): the input→hidden product is a batched GEMM
            # over C per type.
            desc_c = tf.reshape(descriptors, [C, B * A, Q])                    # [C, B*A, Q]
            for t in range(T):
                ph = tf.reshape(tf.matmul(desc_c, W0[:, t, :, :]), [C, B, A, H])  # [C,B,A,H]
                pre_h_terms.append(ph * type_masks[t][tf.newaxis])
        else:
            # Per type: [B*A, Q] @ [Q, C*H] → [C, B, A, H]. One GEMM/type, not C.
            desc_flat = tf.reshape(descriptors, [B * A, Q])
            for t in range(T):
                W0_t     = W0[:, t, :, :]                                              # [C, Q, H]
                W0_t_mat = tf.reshape(tf.transpose(W0_t, [1, 0, 2]), [Q, C * H])      # [Q, C*H]
                ph_flat  = tf.matmul(desc_flat, W0_t_mat)                             # [B*A, C*H]
                ph       = tf.transpose(tf.reshape(ph_flat, [B, A, C, H]), [2, 0, 1, 3])  # [C,B,A,H]
                pre_h_terms.append(ph * type_masks[t][tf.newaxis])
        pre_h = tf.add_n(pre_h_terms)  # [C, B, A, H]

        # ── Bias / output-weight gathers ──────────────────────────────────────
        # tf.gather along T axis: b0[C,T,H] gathered by Z_flat[B*A] → [C,B*A,H]
        Z_flat   = tf.reshape(Z, [B * A])
        b0_t_all = tf.reshape(tf.gather(b0, Z_flat, axis=1), [C, B, A, H])
        _w1_shape = ([C, B, A, H_final, self._head_slots]
                     if self._neighbor_readout else [C, B, A, H_final])
        W1_t_all = tf.reshape(tf.gather(W1, Z_flat, axis=1), _w1_shape)

        # ── Activation (single hidden layer) ──────────────────────────────────
        # z1 preserved for the swish backward chain below.
        z1 = pre_h + b0_t_all
        h1 = self.activation(z1)
        h1 = h1 * atom_mask[tf.newaxis, :, :, tf.newaxis]

        # ── PES ───────────────────────────────────────────────────────────────
        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h1 * W1_t_all, axis=3) + b1[:, tf.newaxis, tf.newaxis]
            E = E * atom_mask[tf.newaxis]
            return -tf.reduce_sum(E, axis=2, keepdims=True)  # [C, B, 1]

        # ── Dipole, neighbor-basis readout ────────────────────────────────────
        if self._neighbor_readout:
            # W_atom slot carries V [B, A, 3, Ch, T] (SNES threads it); no
            # backward matmul — the readout reads no descriptor gradients.
            # b1 stays excluded: exact gauge with c symmetric; a modeling
            # choice (kept deliberately) under "ordered" Ω.
            out = tf.einsum('cbah,cbahk->cbak', h1, W1_t_all)      # [C, B, A, S]
            if self._experts_on:
                out = self._head_with_experts(out, nep_raw, Z, atom_mask, c_r, c_a,
                                              experts, sum_delta)
            s = self._split_head(out)                           # [C, B, A, Ch]
            if c_pair is None:
                # Candidate-independent c (frozen c ≡ 1, or model weights):
                # tile once so ONE einsum serves both cases. (Symbolic C —
                # never `list + TensorShape`, it does not trace.)
                c_full = tf.tile(self._c_full(self.c_pair)[tf.newaxis],
                                 [C, 1, 1, 1])                     # [C, Ch, T, T]
            else:
                c_full = self._c_full(c_pair)                      # [C, Ch, T, T]
            c_at = tf.gather(c_full, Z, axis=2)                    # [C, Ch, B, A, T]
            mu = tf.einsum('cban,cnbat,badnt->cbd', s, c_at, W_atom)   # [C, B, 3]
            return mu

        # ── Dipole: backward matmul ───────────────────────────────────────────
        # ∂U/∂a1 = activation'(h1, z1)·W1.
        de_da    = self._activation_grad(h1, z1) * W1_t_all   # [C, B, A, H]
        de_da_flat = tf.reshape(de_da, [C, B * A, H])         # [C, B*A, H]

        # Per type: [C, B*A, H] @ [C, H, Q] → [C, B*A, Q]  (batched GEMM over C)
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

        # W_atom [B, A, 3, Q]: dipole[c,b,s] = -Σ_{a,q} de_dq[c,b,a,q]*W_atom[b,a,s,q]
        return -tf.einsum('cbaq,basq->cbs', de_dq, W_atom)  # [C, B, 3]

    def _dipole_nep_pairs(self, de_dq_nep: tf.Tensor, s: tf.Tensor,
                          c_r: tf.Tensor | None, c_a: tf.Tensor | None,
                          Z: tf.Tensor, B: tf.Tensor,
                          nep_pair_atom, nep_pair_gidx, nep_pair_struct,
                          nep_pair_D, nep_pair_ang) -> tf.Tensor:
        """μ [Cc, B, 3] = −Σ_p |D_p|^N F_p from the NEP pair forces, N =
        cfg.dipole_rij_power ≥ 1, |D| from the stored image-exact D (never MIC:
        rc_r = 8 Å exceeds L/2 for the water box — plan §B.2)."""
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
        """Per-pair NEP forces F_p = ∂U_{i_p}/∂D_p [Cc, P, 3] (D = r_j − r_i, the
        neighbour-side derivative as GPUMD's x12 = x[j] − x[i]; the centre side is
        −F_p), for U_i = Σ_d Fp[i,d] q_d with Fp = de_dq over the NEP block.
        Plan §B.2 factorisation:
            radial   F^r_p = (Σ_k α[i,t_j,k] fnp_k(r)) r̂,  α[i,t',k] = Σ_n Fp[i,n] c_r[t_i,t',n,k]
            angular  Gs[i,n,c] = ∂(Σ_d Fp[i,d] q_d(s_i))/∂s[i,n,c]   (GradientTape over
                     find_q_tf — covers L=1..4, 4-body, 5-body)
                     F^a_p = Σ_n [ gnp_n (Σ_c Gs P_c) r̂ + g_n Σ_c Gs ∇P_c ]  (XLA n-loop)
        Args:
            de_dq_nep : [Cc, B, A, dim_nep]     (Cc = 1 for predict_batch)
            s         : [(Cc,) B, A, n_a+1, 24] from nep_from_raw(return_s=True)
            c_r, c_a  : [(Cc,) T, T, n+1, K+1] (None → self.c_r / self.c_a); shared
                        c is broadcast over the candidate axis
            Z         : [B, A]
            nep_pair_atom/gidx/struct [P], nep_pair_D [P, 3], nep_pair_ang [P] bool —
            ang is the authoritative rc_a mask (never re-derived from float32 D).
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

        # Candidate-independent per-pair basis (in-graph from D).
        r = tf.norm(nep_pair_D, axis=1)                                      # [P]
        rhat = nep_pair_D / r[:, tf.newaxis]
        _, fnp_r = fn_and_fnp_tf(r, cfg.nep_rc_radial, K_r - 1)              # [P, K_r]

        # Radial: α [Cc, N, T, K_r] → gathered per pair at (i, t_j).
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
        """|r_ij|^N per-pair weight from the rij² primitive (N ≥ 1; avoids sqrt
        for even N). N=0 (self-pairs only) is handled by _dipole_pair_weight_*.
            N=1 → √rij²;  N=2 → rij²;  even≥4 → rij²^(N/2);  odd≥3 → rij²^((N-1)/2)·√rij²
        """
        N = int(getattr(self.cfg, "dipole_rij_power", 2))
        if N == 1:
            return tf.sqrt(rij2)
        if N == 2:
            return rij2
        if N % 2 == 0:
            return tf.pow(rij2, N // 2)
        # Odd N ≥ 3
        return tf.pow(rij2, (N - 1) // 2) * tf.sqrt(rij2)

    def _dipole_pair_weight_coo(self, rij2: tf.Tensor,
                                 pair_atom: tf.Tensor,
                                 pair_gidx: tf.Tensor) -> tf.Tensor:
        """Per-pair dipole weight, COO-pair interface.
            N == 0 : 1 where pair_atom==pair_gidx AND rij²<1e-20 (true self pair),
                     else 0. The rij² guard rejects periodic self-images (same
                     atom index, nonzero displacement) that would double-count.
            N >= 1 : |r_ij|^N (self pairs → 0, images weighted correctly).
        """
        N = int(getattr(self.cfg, "dipole_rij_power", 2))
        if N == 0:
            is_self = tf.logical_and(tf.equal(pair_atom, pair_gidx),
                                     rij2 < 1e-20)
            return tf.cast(is_self, rij2.dtype)
        return self._scalar_rij_pow(rij2)

    def _dipole_pair_weight_padded(self, rij2: tf.Tensor,
                                    grad_index: tf.Tensor) -> tf.Tensor:
        """Per-pair dipole weight, padded [A, M] interface (same dispatch as COO).
        Centre of row i is i, so self pairs are grad_index[i,m]==i AND rij²<1e-20.

        Caller MUST multiply by neighbor_mask afterwards: padding rows can look
        like self pairs (grad_index==0, rij2==0). predict()'s dipole branch does.
        """
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
        """Compute per-pair forces via COO gather + einsum.

        Args:
            de_dq       : [B, A, Q]   energy derivative w.r.t. descriptor
            grad_values : [P, 3, Q]   COO gradient blocks
            pair_struct : [P]         batch-relative structure index
            pair_atom   : [P]         center atom index

        Returns:
            forces_per_pair : [P, 3]
        """
        ba = tf.stack([pair_struct, pair_atom], axis=1)          # [P, 2]
        de_dq_per_pair = tf.gather_nd(de_dq, ba)                 # [P, Q]
        return tf.einsum('kq,kcq->kc', de_dq_per_pair, grad_values)  # [P, 3]

    def _neighbor_displacements_single(self, positions: tf.Tensor,
                                       box: tf.Tensor,
                                       grad_index: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Compute neighbor displacements for a single structure (padded interface).

        Used by the single-structure predict() path (e.g. spectroscopy).

        Args:
            positions  : [A, 3]
            box        : [3, 3]
            grad_index : [A, M]

        Returns:
            dr  : [A, M, 3]  displacement vectors to neighbors
            rij : [A, M]     scalar distances to neighbors
        """
        box_inv = tf.linalg.inv(box)
        # ASE row-vector cell convention: fractional s = r @ inv(cell), i.e.
        # contract the FIRST index of box_inv (='ji'), so the minimum-image
        # round() below happens in the true fractional basis. Using 'ij' here
        # transposes the basis and gives wrong MIC wrapping for non-orthogonal
        # (triclinic/sheared) cells (no-op for orthorhombic/diagonal boxes).
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
        """Aggregate rij²-weighted gradients per (structure, atom); candidate-independent.
            W_atom[b,a,s,q] = Σ_{p: struct=b, atom=a} rij²[p]·grad_values[p,s,q]
        Dipole is then -einsum('baq,basq->bs', de_dq, W_atom), no P axis inside
        the candidate loop (eliminates the [C, P, Q] intermediate).

        Args:
            grad_values : [P, 3, Q]  (already scaled if descriptor scaling is active)
            pair_struct : [P]
            pair_atom   : [P]
            pair_gidx   : [P]
            positions   : [B, A, 3]
            boxes       : [B, 3, 3]
            B           : number of structures
            A           : max atoms (padded)

        Returns:
            W_atom : [B, A, 3, Q]
        """
        P = tf.shape(grad_values)[0]
        Q = tf.shape(grad_values)[2]
        # N == 0 : weight = 1 everywhere — the COO list is already filtered to
        #          self-pairs (data.py self_pairs_only), so skip displacements
        #          + tf.linalg.inv(boxes) entirely.
        # N >= 1 : per-pair |r|^N weight; needs displacements.
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
        """Compute per-pair displacements in COO format with MIC wrapping.

        Args:
            positions  : [B, A, 3]
            boxes      : [B, 3, 3]
            box_inv    : [B, 3, 3]
            pair_struct: [P]  batch-relative structure index
            pair_atom  : [P]  center atom index
            pair_gidx  : [P]  neighbor atom index

        Returns:
            dr   : [P, 3]  displacement vectors (neighbor - center)
            rij2 : [P]     squared distances
        """
        ba_c = tf.stack([pair_struct, pair_atom], axis=1)   # [P, 2]
        ba_n = tf.stack([pair_struct, pair_gidx], axis=1)   # [P, 2]
        pos_c   = tf.gather_nd(positions, ba_c)              # [P, 3]
        pos_n   = tf.gather_nd(positions, ba_n)              # [P, 3]
        box_k   = tf.gather(boxes,   pair_struct)            # [P, 3, 3]
        binv_k  = tf.gather(box_inv, pair_struct)            # [P, 3, 3]
        # ASE row-vector convention: s = r @ inv(cell) → contract the first
        # (row) index of box_inv (='kji'), matching _neighbor_displacements_single.
        # 'kij' transposes the fractional basis → wrong MIC for triclinic cells.
        s_c = tf.einsum('kji,kj->ki', binv_k, pos_c)        # [P, 3] fractional
        s_n = tf.einsum('kji,kj->ki', binv_k, pos_n)
        ds  = s_n - s_c
        ds  = ds - tf.round(ds)                              # MIC wrap
        dr  = tf.einsum('kji,kj->ki', box_k, ds)            # [P, 3] Cartesian
        rij2 = tf.reduce_sum(tf.square(dr), axis=-1)         # [P]
        return dr, rij2

    def _fc(self, r: tf.Tensor) -> tf.Tensor:
        """Dipole cutoff f_c(r) per the resolved cfg.dipole_cutoff:
        "cosine2" ((1+cos(πr/r_c))/2)² — C³ contact at r_c; "cosine"
        (1+cos(πr/r_c))/2 — C¹; "cubic" 1 − 3x² + 2x³, x = r/r_c — C¹;
        "tanh3" tanh³(1 − r/r_c)/tanh³(1) — C² (normalized so f(0) = 1).
        All finite on raw r (tanh3 goes NEGATIVE beyond r_c before
        masking); callers mask r ≥ dipole_rc to zero."""
        rc = float(self.cfg.dipole_rc)
        fc = self._dipole_cutoff
        if fc == "cosine2":
            return tf.square(0.5 * (1.0 + tf.cos(np.pi / rc * r)))
        if fc == "cosine":
            return 0.5 * (1.0 + tf.cos(np.pi / rc * r))
        if fc == "cubic":
            x = r / rc
            return 1.0 - 3.0 * x * x + 2.0 * x * x * x
        if fc == "tanh3":
            return tf.tanh(1.0 - r / rc) ** 3 / np.tanh(1.0) ** 3
        raise ValueError(f"unknown dipole_cutoff {fc!r}")

    def _cheb_R(self, r: tf.Tensor) -> tf.Tensor:
        """Chebyshev radial channels R_n(r) = T_n(x)·f_c(r), n = 0..C−1,
        stacked on a trailing channel axis → [..., C]; f_c per
        cfg.dipole_cutoff (_fc). x = 2r/r_c − 1 ("chebyshev") or the NEP
        warp x_w = 2(r/r_c − 1)² − 1 ("chebyshev_warped"). Recurrence
        (not cos∘arccos) so out-of-cutoff r stays finite until the caller
        masks it."""
        C = self._dipole_channels
        if self._dipole_radial == "chebyshev_warped":
            x = 2.0 * tf.square(r / float(self.cfg.dipole_rc) - 1.0) - 1.0
        else:
            x = 2.0 * r / float(self.cfg.dipole_rc) - 1.0
        Ts = [tf.ones_like(x), x]
        for _ in range(2, C):
            Ts.append(2.0 * x * Ts[-1] - Ts[-2])
        return tf.stack(Ts[:C], axis=-1) * self._fc(r)[..., tf.newaxis]

    def _radial(self, r: tf.Tensor) -> tf.Tensor:
        """Radial channel matrix [..., C] per cfg.dipole_radial. Screened:
        R(r) = r^(ζ−1)·f_c(r), single channel; f_c per cfg.dipole_cutoff
        (_fc). Both TF call sites mask AFTER the radial and real r=0
        self-image entries exist in the pair streams, so guard the power's
        INPUT (at ζ < 1, 0**(ζ−1) = inf and inf·0 = nan downstream)."""
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
        raise ValueError(f"unknown dipole_radial {rad!r}")

    def _split_head(self, out: tf.Tensor):
        """Head output [.., S] → site values s [.., C], S = _head_slots (C for
        dipole_head="vector"; 1 for "scalar", broadcast over the C channels)."""
        if self._head_slots == self._dipole_channels:
            return out
        return tf.broadcast_to(
            out, tf.concat([tf.shape(out)[:-1], [self._dipole_channels]], 0))

    def _c_full(self, c_pair: tf.Tensor) -> tf.Tensor:
        """Packed c_pair [..., C, n_pairs] → [..., C, T, T] via _pair_idx
        (symmetric table for "off"/"symmetric", ordered for "ordered")."""
        return tf.gather(c_pair, self._pair_idx, axis=-1)

    def _neighbor_V_coo(self, positions: tf.Tensor, boxes: tf.Tensor,
                        box_inv: tf.Tensor, pair_struct: tf.Tensor,
                        pair_atom: tf.Tensor, pair_gidx: tf.Tensor,
                        Z: tf.Tensor, B: tf.Tensor, A: tf.Tensor) -> tf.Tensor:
        """Geometry kernel of the neighbor-basis dipole readout:
            V[b,a,:,n,t'] = Σ_{pairs (a,j), |d|<r_c, t_j=t'} R_n(|d|)·d,
            d = r_a − r_j (MIC), R_n per cfg.dipole_radial (_radial).
        Pairs come from the r_d descriptor list (r_c ≤ rcut_hard guarantees
        coverage); j==a self-image entries are excluded. Accumulated in
        float64, cast back to the pipeline dtype.

        Returns:
            V : [B, A, 3, C, T]
        """
        if float(self.cfg.dipole_rc) > float(self.cfg.rcut_hard):
            raise ValueError(
                f"in-graph V: dipole_rc={self.cfg.dipole_rc} > "
                f"rcut_hard={self.cfg.rcut_hard}, but this path builds V from "
                f"the SOAP descriptor pair stream, which only reaches "
                f"rcut_hard — it would silently truncate the readout. Pass the "
                f"staged V (data['V_atom']) instead; training, validation and "
                f"score already do.")
        dr, rij2 = self._neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)
        rc = float(self.cfg.dipole_rc)
        C = self._dipole_channels
        T = self.num_types
        r = tf.sqrt(rij2)
        keep = tf.logical_and(r < rc, tf.not_equal(pair_atom, pair_gidx))
        Rw = self._radial(r) * tf.cast(keep, r.dtype)[:, tf.newaxis]  # [P, C]
        # contrib[p, d, n] = R_n·d, d = −(neighbor − center).
        contrib = tf.cast((-dr)[:, :, tf.newaxis]
                          * Rw[:, tf.newaxis, :], tf.float64)      # [P, 3, C]
        tprime = tf.gather_nd(Z, tf.stack([pair_struct, pair_gidx], axis=1))
        seg = (pair_struct * A + pair_atom) * T + tf.cast(tprime, pair_atom.dtype)
        V = tf.math.unsorted_segment_sum(
            tf.reshape(contrib, [-1, 3 * C]), seg, num_segments=B * A * T)
        V = tf.reshape(tf.cast(V, dr.dtype), [B, A, T, 3, C])
        return tf.transpose(V, [0, 1, 3, 4, 2])                  # [B,A,3,C,T]

    def _dipole_coo(self, forces_per_pair: tf.Tensor, pair_struct: tf.Tensor,
                    pair_atom: tf.Tensor, pair_gidx: tf.Tensor,
                    positions: tf.Tensor, boxes: tf.Tensor,
                    box_inv: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
        """Batched dipole prediction using COO forces.

        Args:
            forces_per_pair : [P, 3]
            pair_struct     : [P]
            pair_atom       : [P]
            pair_gidx       : [P]
            positions       : [B, A, 3]
            boxes           : [B, 3, 3]
            box_inv         : [B, 3, 3]
            B               : int scalar — number of structures

        Returns:
            dipole : [B, 3]
        """
        # μ: N>=1 → -Σ_pair |r_ij|^N·F_ij; N==0 → -Σ_i F_ii (self-only).
        _, rij2 = self._neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)
        weight = self._dipole_pair_weight_coo(rij2, pair_atom, pair_gidx)  # [P]
        dipole_contrib = weight[:, tf.newaxis] * forces_per_pair         # [P, 3]
        dipole = -tf.math.unsorted_segment_sum(
            dipole_contrib, pair_struct, num_segments=B)                  # [B, 3]
        return dipole

    def _polarizability_coo(self, descriptors: tf.Tensor, forces_per_pair: tf.Tensor,
                            pair_struct: tf.Tensor, pair_atom: tf.Tensor,
                            pair_gidx: tf.Tensor, positions: tf.Tensor,
                            boxes: tf.Tensor, box_inv: tf.Tensor,
                            Z: tf.Tensor, atom_mask: tf.Tensor,
                            W0_pol: tf.Tensor, b0_pol: tf.Tensor,
                            W1_pol: tf.Tensor, b1_pol: tf.Tensor,
                            B: tf.Tensor) -> tf.Tensor:
        """Batched polarizability via dual ANN using COO forces.

        Args:
            descriptors     : [B, A, Q]
            forces_per_pair : [P, 3]
            pair_struct     : [P]
            pair_atom       : [P]
            pair_gidx       : [P]
            positions       : [B, A, 3]
            boxes           : [B, 3, 3]
            box_inv         : [B, 3, 3]
            Z               : [B, A]
            atom_mask       : [B, A]
            W0_pol..b1_pol  : scalar ANN weights

        Returns:
            pol : [B, 6]  — [xx, yy, zz, xy, yz, zx]
        """
        dr, _ = self._neighbor_displacements_coo(positions, boxes, box_inv,
                                                  pair_struct, pair_atom, pair_gidx)

        # Scalar ANN (isotropic contribution) — same type-loop pattern as main ANN
        b0p_t = tf.gather(b0_pol, Z)   # [B, A, H]
        W1p_t = tf.gather(W1_pol, Z)   # [B, A, H]
        type_masks_p = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(self.num_types)
        ]
        h_pol = tf.add_n([
            tf.einsum('baq,qh->bah', descriptors, W0_pol[t]) * type_masks_p[t]
            for t in range(self.num_types)
        ]) + b0p_t
        h_pol = self.activation(h_pol)
        h_pol = h_pol * atom_mask[:, :, tf.newaxis]
        F_pol = tf.reduce_sum(h_pol * W1p_t, axis=2) + b1_pol  # [B, A]
        F_pol = F_pol * atom_mask
        scalar_sum = tf.reduce_sum(F_pol, axis=1)               # [B]

        # Tensor part: per-pair outer product, then segment-sum per structure
        pol_outer = -tf.einsum('ki,kj->kij', dr, forces_per_pair)  # [P, 3, 3]
        pol_flat  = tf.reshape(pol_outer, [-1, 9])                  # [P, 9]
        pol_mat_flat = tf.math.unsorted_segment_sum(
            pol_flat, pair_struct, num_segments=B)                  # [B, 9]
        pol_matrix = tf.reshape(pol_mat_flat, [B, 3, 3])

        pol = tf.stack([
            pol_matrix[:, 0, 0], pol_matrix[:, 1, 1], pol_matrix[:, 2, 2],
            pol_matrix[:, 0, 1], pol_matrix[:, 1, 2], pol_matrix[:, 2, 0],
        ], axis=1)  # [B, 6]

        diag_add = tf.stack([scalar_sum, scalar_sum, scalar_sum,
                             tf.zeros_like(scalar_sum),
                             tf.zeros_like(scalar_sum),
                             tf.zeros_like(scalar_sum)], axis=1)
        return pol + diag_add


def gen0_ann_inputs(cfg, data) -> np.ndarray:
    """[S, A, dim_q] generation-0 ANN input of a padded data dict — what
    compute_q_scale must see. The builder's `descriptors` slot is returned as
    is unless it needs finishing: a throwaway model (scaling off) applies
    soap_from_raw (learned basis) and _ann_inputs (NEP concat/replace) chunk by
    chunk; under "sum" / "gap_sum" the expert input follows the SOAP block
    ([q_soap ‖ q_nep] or [q_soap ‖ q2‖q3]). "minmax" assembles the NEP block with c ≡ 1 like GPUMD's q_scaler
    (fitness.cu:165-175, initial_para = 1); block/channel use the real gen-0 c."""
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
        if m0._nep_mode == "gap_sum":
            x = tf.concat([x, raw], axis=-1)
        elif m0._nep_mode == "sum":
            x = tf.concat([x, m0.nep_from_raw(raw, z, **c)], axis=-1)
        out.append(x)
    return tf.concat(out, axis=0).numpy()
