from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Callable

from DescriptorBuilder import make_descriptor_builder
from SNES import SNES
from TNEPconfig import TNEPconfig
import kernels


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

        # Preprocess contraction phase 1/2: run guards, then override cfg.dim_q
        # to the contracted dim BEFORE allocating W0. Phase 2 allocates W_pre_*.
        self.descriptor_preprocess_contract = str(getattr(
            cfg, "descriptor_preprocess_contract", "off"))
        if self.descriptor_preprocess_contract not in (
                "off", "angular", "species_pair", "both", "nep4_radial"):
            raise ValueError(
                f"descriptor_preprocess_contract={self.descriptor_preprocess_contract!r} "
                f"not recognised in this build; expected one of: "
                f"'off', 'angular', 'species_pair', 'both', 'nep4_radial'.")
        if self.descriptor_preprocess_contract == "nep4_radial":
            _cm = str(getattr(cfg, "compress_mode", "trivial"))
            if _cm != "trivial":
                raise NotImplementedError(
                    f"descriptor_preprocess_contract='nep4_radial' requires "
                    f"compress_mode='trivial' (got {_cm!r}); the bilinear "
                    f"fold needs the (n, n', l) decomposition that only "
                    f"trivial compression preserves.")
            # nep4_radial collapses (n, n', species_pair) into a flat (n'', l)
            # axis, so the block layout descriptor_mixing assumes is undefined.
            if bool(getattr(cfg, "descriptor_mixing", False)):
                raise NotImplementedError(
                    "descriptor_preprocess_contract='nep4_radial' is "
                    "mutually exclusive with descriptor_mixing in this "
                    "build. Disable mixing or run nep4_radial standalone.")
        if self.descriptor_preprocess_contract != "off":
            from DescriptorBuilderGPU import (
                descriptor_block_layout, descriptor_preprocess_layout)
            _layout_pre = descriptor_block_layout(cfg)
            self._preprocess_layout = descriptor_preprocess_layout(
                cfg, _layout_pre, self.descriptor_preprocess_contract)
            # Override cfg.dim_q so W0 is allocated at the contracted dim.
            if not hasattr(cfg, "dim_q_raw") or cfg.dim_q_raw is None:
                cfg.dim_q_raw = int(cfg.dim_q)
            cfg.dim_q = int(self._preprocess_layout["dim_q_new"])
        else:
            self._preprocess_layout = None

        self.dim_q = cfg.dim_q
        # With preprocess on, W0 stores at Q_new but the forward matmul folds
        # the contraction in and operates on RAW descriptors at Q_raw; track it.
        if self.descriptor_preprocess_contract != "off":
            self.dim_q_forward = int(cfg.dim_q_raw)
        else:
            self.dim_q_forward = int(cfg.dim_q)
        self.num_types = cfg.num_types
        self.num_neurons = cfg.num_neurons
        self._H_final = cfg.num_neurons
        # Any tf.keras.activations.get name is accepted here; _activation_grad
        # raises NotImplementedError at first dipole/pol pass if not plumbed.
        self._activation_name = str(cfg.activation).lower().strip()
        self._glorot_gain = 1.0
        self.activation = tf.keras.activations.get(cfg.activation)
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

        # W1 : [num_types, H_final] — hidden-to-scalar weights per type.
        self.W1 = self.add_weight(
            name="W1",
            shape=(cfg.num_types, self._H_final),
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

        # Validate dipole contraction power (mode 1 only) at construction time.
        if cfg.target_mode == 1:
            _N = int(getattr(cfg, "dipole_rij_power", 2))
            if _N < 0:
                raise ValueError(
                    f"cfg.dipole_rij_power must be an integer ≥ 0, got {_N}. "
                    f"0 = self-pair only (μ = -Σ_i de_dq[i] · ∂q_i/∂R_i), "
                    f"1 = |r|·F (first radial moment), "
                    f"2 = |r|²·F (Xu et al. JCTC 2024 default), "
                    f"≥3 = higher radial moments."
                )

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
            from DescriptorBuilderGPU import (
                descriptor_block_layout,
                descriptor_post_preprocess_block_layout)
            # With preprocess on, mixing operates at Q_new so it needs the
            # post-preprocess layout; else the raw SOAP-turbo layout at Q_raw.
            if self.descriptor_preprocess_contract != "off":
                _raw_layout = descriptor_block_layout(cfg)
                self._mix_layout = descriptor_post_preprocess_block_layout(
                    cfg, _raw_layout, self.descriptor_preprocess_contract)
            else:
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
            # L_eff when present (post-preprocess collapses l>l_keep to one slot);
            # else cfg.l_max+1.
            self._mix_L = int(self._mix_layout.get("L_eff",
                                                   int(cfg.l_max) + 1))
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

        # Preprocess contraction phase 2/2: W_pre_* allocation. cfg.dim_q was
        # already overridden in phase 1 so W0 above is sized at Q_new.
        if self.descriptor_preprocess_contract != "off":
            init_scheme = str(getattr(cfg, "descriptor_preprocess_init", "mean"))
            T_pre = int(cfg.num_types)
            self.preprocess_per_type = bool(getattr(
                cfg, "descriptor_preprocess_per_type", True))
            if self.descriptor_preprocess_contract == "nep4_radial":
                # c tensor rank-4 [T_centre, T_neighbour, n_max_out, α]. The
                # linear-fold tail machinery below doesn't apply; allocate c and
                # precompute the bilinear-fold gather indices for _W0_preprocess_eff.
                _lay = self._preprocess_layout
                n_max_out = int(_lay["nep4_n_max_out"])
                alpha_max = int(_lay["nep4_alpha_max"])
                coef_shape = tuple(_lay["coef_shape"])
                full_coef_size = int(_lay["nep4_full_coef_size"])
                # Init: Glorot with fan_in=α (each c contracts α primitives);
                # per-element σ gives a roughly unit-scale bilinear product.
                init_norm = float(_lay["coef_init_norm"])
                if init_scheme == "glorot":
                    fan_in = max(1, alpha_max)
                    fan_out = max(1, n_max_out)
                    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
                    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
                    init_np = rng.uniform(
                        -limit, limit, size=coef_shape).astype(np.float32)
                elif init_scheme in ("mean", "sum"):
                    # Uniform fan-in normalised init; SNES searches from here.
                    init_np = np.full(coef_shape, init_norm, dtype=np.float32)
                else:
                    raise ValueError(
                        f"descriptor_preprocess_init={init_scheme!r} not "
                        f"recognised; expected 'mean', 'sum', or 'glorot'.")
                self.W_pre_angular = tf.Variable(
                    init_np, trainable=False, name="W_pre_angular_nep4",
                    dtype=tf.float32)
                # All c entries are SNES μ-slots (no passthrough); layout mirrors
                # the linear-fold tail so SNES wiring keeps working over full c.
                smask_full = np.ones(coef_shape, dtype=bool)
                self._preprocess_summed_mask = tf.constant(smask_full, dtype=tf.bool)
                flat_idx = np.arange(full_coef_size, dtype=np.int32)
                self._preprocess_summed_flat_idx = tf.constant(
                    flat_idx, dtype=tf.int32)
                self._preprocess_summed_count = int(full_coef_size)
                # Kept-template zero (no passthrough); SNES scatters μ everywhere.
                base_template = np.zeros(coef_shape, dtype=np.float32)
                self._preprocess_kept_template = tf.constant(
                    base_template, dtype=tf.float32)
                _M_np = np.eye(full_coef_size, dtype=np.float32)
                self._preprocess_summed_scatter_M = tf.constant(
                    _M_np, dtype=tf.float32)
                self._preprocess_kept_template_size = full_coef_size
                # Precomputed gather indices for the NEP4 bilinear fold.
                self._nep4_n_max_out = n_max_out
                self._nep4_alpha_max = alpha_max
                self._nep4_L = int(_lay["L"])
                # [n_max_out, L] mid-shape constant for reshaping W0 in
                # _W0_preprocess_eff_nep4; precomputed to avoid a fresh
                # tf.constant per call in the traced predict_batch path.
                self._nep4_mid_shape = tf.constant(
                    [self._nep4_n_max_out, self._nep4_L], dtype=tf.int32)
                self._nep4_l_of_q = tf.constant(
                    _lay["nep4_l_of_q"], dtype=tf.int32)
                self._nep4_n_global = tf.constant(
                    _lay["nep4_n_global"], dtype=tf.int32)
                self._nep4_np_global = tf.constant(
                    _lay["nep4_np_global"], dtype=tf.int32)
                self._nep4_n_to_species = tf.constant(
                    _lay["nep4_n_to_species"], dtype=tf.int32)
                self._nep4_n_to_local = tf.constant(
                    _lay["nep4_n_to_local"], dtype=tf.int32)
                # Linear-fold-only attrs left None; branch-checked in _W0_preprocess_eff.
                self._preprocess_q_to_q_new = None
                self._preprocess_scatter = None
                self.optimizer = SNES(self)
                return
            Q_raw_pre = int(self._preprocess_layout["coef_shape"][0])

            # Per-q_raw classification from the layout. summed_mask: True where
            # q_raw feeds a SUMMED channel (learnable W_pre); False for KEPT
            # passthrough (W_pre fixed 1.0). Shape 1-D [Q_raw] (angular) or
            # 2-D [T, Q_raw] (species_pair/both).
            _summed = self._preprocess_layout.get("summed_q_raw_mask")
            if _summed is None:
                _summed = np.zeros((Q_raw_pre,), dtype=bool)
            _per_q = self._preprocess_layout.get("coef_init_per_q_raw")
            if _per_q is None:
                _scalar = float(self._preprocess_layout["coef_init_norm"])
                _per_q = np.full((Q_raw_pre,), _scalar, dtype=np.float32)

            # W_pre init: kept entries start at 1.0 (passthrough); summed at the
            # per-q_raw mean/sum/glorot value.
            def _build_init_2d() -> np.ndarray:
                """Return [T, Q_raw] init (source before squeezing T for per_type=False)."""
                if _per_q.ndim == 1:
                    init_2d = np.broadcast_to(
                        _per_q[None, :], (T_pre, Q_raw_pre)).astype(np.float32)
                else:
                    init_2d = _per_q.astype(np.float32)
                if _summed.ndim == 1:
                    smask_2d = np.broadcast_to(
                        _summed[None, :], (T_pre, Q_raw_pre))
                else:
                    smask_2d = _summed
                if init_scheme == "mean":
                    out = init_2d.copy()
                elif init_scheme == "sum":
                    out = np.where(smask_2d, 1.0, 1.0).astype(np.float32)
                elif init_scheme == "glorot":
                    fan_in = int(self._preprocess_layout["L"])
                    fan_out = 1
                    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
                    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
                    rand = rng.uniform(
                        -limit, limit, size=(T_pre, Q_raw_pre)).astype(np.float32)
                    out = np.where(smask_2d, rand, 1.0).astype(np.float32)
                else:
                    raise ValueError(
                        f"descriptor_preprocess_init={init_scheme!r} not recognised; "
                        f"expected 'mean', 'sum', or 'glorot'.")
                # Always force kept positions to 1.0 (identity passthrough).
                out = np.where(smask_2d, out, 1.0).astype(np.float32)
                return out

            init_2d_np = _build_init_2d()
            if self.preprocess_per_type:
                # W_pre shape [T, Q_raw]
                init_np = init_2d_np.astype(np.float32).copy()
            else:
                # Global W_pre [Q_raw]: collapse T. 1-D layouts identical across
                # t; 2-D take max (ghost entries 0, active carry the init value).
                init_np = np.max(init_2d_np, axis=0).astype(np.float32).copy()
            self.W_pre_angular = tf.Variable(
                init_np, trainable=False, name="W_pre_angular",
                dtype=tf.float32)

            # Summed mask for SNES (exposes only summed entries to μ); shape
            # mirrors W_pre.
            if self.preprocess_per_type:
                if _summed.ndim == 1:
                    smask_full = np.broadcast_to(
                        _summed[None, :], (T_pre, Q_raw_pre)).copy()
                else:
                    smask_full = _summed.copy()
            else:
                if _summed.ndim == 1:
                    smask_full = _summed.copy()
                else:
                    smask_full = np.any(_summed, axis=0)
            self._preprocess_summed_mask = tf.constant(smask_full, dtype=tf.bool)
            # Flat indices of summed entries; SNES scatters μ into W_pre by these.
            flat_idx = np.flatnonzero(smask_full.reshape(-1)).astype(np.int32)
            self._preprocess_summed_flat_idx = tf.constant(flat_idx, dtype=tf.int32)
            self._preprocess_summed_count = int(flat_idx.size)
            # Base template: kept positions keep init (e.g. 1.0), summed zeroed.
            # tf.constant so candidate-path reconstruction needs only a
            # tensor_scatter_nd_update, never a Variable read.
            base_template = np.where(smask_full, 0.0, init_np).astype(np.float32)
            self._preprocess_kept_template = tf.constant(
                base_template, dtype=tf.float32)
            # One-hot scatter matrix (n_summed × base_size) for SNES.
            # reconstruct_params_tf. Built once here to avoid a per-chunk
            # tf.one_hot kernel that fragmented the @tf.function trace.
            _base_flat_size = int(base_template.size)
            _M_np = np.zeros(
                (flat_idx.size, _base_flat_size), dtype=np.float32)
            _M_np[np.arange(flat_idx.size), flat_idx] = 1.0
            self._preprocess_summed_scatter_M = tf.constant(
                _M_np, dtype=tf.float32)
            self._preprocess_kept_template_size = _base_flat_size
            _map_np = self._preprocess_layout["q_raw_to_q_new"]
            self._preprocess_q_to_q_new = tf.constant(_map_np, dtype=tf.int32)
            # Per-type maps (species_pair): one-hot scatter M[T, Q_raw, Q_new]
            # so _W0_preprocess_eff folds via einsum without per-type gather;
            # map==-1 (q_raw not in type t) gives zero rows.
            if _map_np.ndim == 2:
                T_pre_m = int(_map_np.shape[0])
                Q_raw_m = int(_map_np.shape[1])
                Q_new_m = int(self._preprocess_layout["dim_q_new"])
                valid = (_map_np >= 0)
                safe_idx = np.where(valid, _map_np, 0)
                M_np = np.zeros((T_pre_m, Q_raw_m, Q_new_m), dtype=np.float32)
                ti = np.arange(T_pre_m)[:, None]
                qi = np.arange(Q_raw_m)[None, :]
                M_np[ti, qi, safe_idx] = valid.astype(np.float32)
                self._preprocess_scatter = tf.constant(M_np, dtype=tf.float32)
            else:
                self._preprocess_scatter = None
        else:
            self.W_pre_angular = None
            self._preprocess_q_to_q_new = None
            self._preprocess_scatter = None

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
        """
        if not self.descriptor_mixing:
            return W0
        U_full = self._U_full(U_pair)
        if self.descriptor_mixing_per_type:
            return tf.einsum('...tqp,...tqh->...tph', U_full, W0)
        return tf.einsum('...qp,...tqh->...tph', U_full, W0)

    def _W0_preprocess_eff(self, W0: tf.Tensor,
                            W_pre_override: tf.Tensor | None = None) -> tf.Tensor:
        """Fold the angular preprocess contraction into W0, giving raw-dim weights:
            W0_eff[t, q_raw, h] = W_pre[t, q_raw] · W0[t, q_to_q_new[q_raw], h]
        Keeps the matmul at Q_raw while W0 storage stays at Q_new (mirrors
        _W0_eff); backward dq comes out at Q_raw for the raw-W_atom dipole sum.

        Args:
          W0:             [(C,) T, Q_new, H]  weights at contracted dim
          W_pre_override: [(C,) T, Q_raw]     per-type coeffs; None → self.W_pre_angular
        Returns:
          W0_eff: [(C,) T, Q_raw, H]
        """
        if self.descriptor_preprocess_contract == "off":
            return W0
        W_pre = (W_pre_override if W_pre_override is not None
                 else self.W_pre_angular)
        if self.descriptor_preprocess_contract == "nep4_radial":
            return self._W0_preprocess_eff_nep4(W0, W_pre)
        # W0 storage: [(C,) T, Q_new, H]. Gather/scatter along Q_new to
        # produce [(C,) T, Q_raw, H], then weight by W_pre.
        if self._preprocess_scatter is None:
            # 1-D map (angular): shared mapping across t.
            W0_at_qraw = tf.gather(W0, self._preprocess_q_to_q_new, axis=-2)
        else:
            # 2-D map (species_pair / both): per-type scatter via einsum.
            W0_at_qraw = tf.einsum(
                'tqp,...tph->...tqh', self._preprocess_scatter, W0)
        # Multiply by W_pre. per_type=False needs a T-axis singleton inserted
        # (at rank-2) so the broadcast works with or without a candidate dim.
        factor = W_pre[..., tf.newaxis]
        if not self.preprocess_per_type:
            factor = tf.expand_dims(factor, axis=factor.shape.rank - 2)
        return W0_at_qraw * factor

    def _W0_preprocess_eff_nep4(self, W0: tf.Tensor,
                                 c: tf.Tensor) -> tf.Tensor:
        """NEP4 bilinear (rank-1 outer-product) fold of W0.

        Implements g[t, n'', l] = Σ_{n,n'} c[t,s(n),n'',k(n)] · c[t,s(n'),n'',k(n')]
        · p[n,n',l], transforming W0 from [n'',l] storage at Q_new = n_max_out·L
        to the [n,n',l] raw layout at Q_raw so the matmul stays against raw desc.

        Args:
          W0: [(C,) T, n_max_out · L, H]                   weights at Q_new
          c:  [(C,) T_centre, T_neighbour, n_max_out, α]   NEP4 coeffs
        Returns:
          W0_eff: [(C,) T, Q_raw, H]
        """
        T_c = int(self.cfg.num_types)
        T_n = T_c
        n_max_out = int(self._nep4_n_max_out)
        L_ = int(self._nep4_L)
        alpha = int(self._nep4_alpha_max)
        # Flatten (T_neighbour, α) → one axis aligned with nep4_n_global:
        # c [..., T_c, T_n, n_max_out, α] → [..., T_c, T_n·α, n_max_out].
        has_C = (c.shape.rank == 5)
        if has_C:
            c_perm = tf.transpose(c, perm=[0, 1, 2, 4, 3])
            c_flat = tf.reshape(c_perm, [-1, T_c, T_n * alpha, n_max_out])
        else:
            c_perm = tf.transpose(c, perm=[0, 1, 3, 2])
            c_flat = tf.reshape(c_perm, [T_c, T_n * alpha, n_max_out])
        # Gather along the flattened (T_n, α) axis using global-n indices.
        # A_a[..., t, q, n''] = c_flat[..., t, n_global(q), n'']
        # A_b[..., t, q, n''] = c_flat[..., t, n'_global(q), n'']
        A_a = tf.gather(c_flat, self._nep4_n_global, axis=-2)
        A_b = tf.gather(c_flat, self._nep4_np_global, axis=-2)
        AB = A_a * A_b   # [..., T_c, Q_raw, n_max_out]   (rank-1 outer product)
        # Reshape W0 [..., T, n_max_out·L, H] to expose (n_max_out, L),
        # preserving leading batch axes. _nep4_mid_shape is precomputed to avoid
        # a fresh tf.constant per call in the traced path.
        W0_shape = tf.shape(W0)
        H_ = W0_shape[-1]
        leading = W0_shape[:-2]
        new_shape = tf.concat(
            [leading, self._nep4_mid_shape, tf.reshape(H_, [1])], axis=0)
        W0_NLH = tf.reshape(W0, new_shape)   # [..., T, n_max_out, L, H]
        # Gather along the l axis using l_of_q:
        # W0_at_q[..., t, n'', q, h] = W0_NLH[..., t, n'', l_of_q(q), h]
        W0_at_q = tf.gather(W0_NLH, self._nep4_l_of_q, axis=-2)
        # W0_eff[..., T, Q_raw, H] = Σ_{n''} AB[t,q,n''] · W0_at_q[t,n'',q,h].
        W0_eff = tf.einsum('...tqN,...tNqh->...tqh', AB, W0_at_q)
        return W0_eff

    def predict(self, descriptors: tf.Tensor, gradients: tf.Tensor, grad_index: tf.Tensor,
                positions: tf.Tensor, Z: tf.Tensor, box: tf.Tensor,
                atom_mask: tf.Tensor, neighbor_mask: tf.Tensor) -> tf.Tensor:
        """Run the forward pass for a single structure using padded tensors.

        Args:
            descriptors    : [A, dim_q]     padded per-atom SOAP descriptors
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
        # Absorb U_pairᵀ into W0 once so forward and calc_forces share the
        # folded weights (no-op when mixing is off).
        W0_eff = self._W0_eff(self.W0)
        # Expand W0 Q_new → Q_raw when preprocess is on (mirrors score()).
        if self.descriptor_preprocess_contract != "off":
            W0_eff = self._W0_preprocess_eff(W0_eff)

        # Gather per-type weights for each atom
        W0_t = tf.gather(W0_eff, Z)    # [A, dim_q, H]
        b0_t = tf.gather(self.b0, Z)   # [A, H]
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
            if self.descriptor_preprocess_contract != "off":
                W0_pol_eff = self._W0_preprocess_eff(W0_pol_eff)
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
        W0_pol_eff = (self._W0_eff(self.W0_pol)
                      if (self.cfg.target_mode == 2
                          and getattr(self, "W0_pol", None) is not None)
                      else getattr(self, "W0_pol", None))
        # Second fold: with preprocess on, expand W0 Q_new → Q_raw before
        # predict_batch's raw-dim einsum (mirrors validate()/_evaluate_chunk).
        if self.descriptor_preprocess_contract != "off":
            W0_eff = self._W0_preprocess_eff(W0_eff)
            if W0_pol_eff is not None:
                W0_pol_eff = self._W0_preprocess_eff(W0_pol_eff)
        ranges = [(s, min(s + chunk_sz, S_test)) for s in range(0, S_test, chunk_sz)]
        pred_parts: list = []
        for _, _, chunk in prefetched_chunks(
                test_data, ranges,
                pin_to_cpu=self.cfg.pin_data_to_cpu):
            # Pre-reduced data carries _W_atom and has no COO fields; the
            # kernel branch in predict_batch returns before touching them.
            pred_parts.append(self.predict_batch(
                chunk["descriptors"], chunk.get("grad_values"),
                chunk.get("pair_atom"), chunk.get("pair_gidx"),
                chunk.get("pair_struct"),
                chunk["positions"], chunk["Z_int"], chunk["boxes"],
                chunk["atom_mask"],
                W0_eff, self.b0, self.W1, self.b1,
                W0_pol_eff,
                getattr(self, 'b0_pol', None),
                getattr(self, 'W1_pol', None),
                getattr(self, 'b1_pol', None),
                W_atom=chunk.get("_W_atom"),
            ))
            del chunk
        raw_preds = tf.concat(pred_parts, axis=0)
        del pred_parts
        targets = test_data["targets"]

        # Normalize predictions to per-atom space when target scaling is active
        if self.cfg.scale_targets and self.cfg.target_mode in (1, 2) and "num_atoms" in test_data:
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
        if self.cfg.scale_targets and self.cfg.target_mode in (1, 2) and "num_atoms" in test_data:
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
        builder = make_descriptor_builder(cfg_for_load)
        descs, grads, gidx = builder.build_descriptors(dataset)

        # Assemble + pad. Descriptors are now [N_i, TRAIN_DIM_Q] per structure.
        data_dict = assemble_data_dict(
            dataset, ti, descs, grads, gidx, cfg_for_load)
        test_data = pad_and_stack(
            data_dict, num_types=TRAIN_NUM_TYPES, pin_to_cpu=pin_to_cpu)

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
        print(f"    RMSE  = {m['rmse']:.6f}  per atom / component")
        print(f"    R²    = {m['r2']:.6f}")
        print(f"    RRMSE = √(1−R²) = {rrmse_pa:.4%}")
        # Total (only when scale_targets active):
        if "total_rmse" in m and "total_r2" in m:
            rrmse_tot = (1.0 - m["total_r2"]) ** 0.5
            print(f"  TOTAL (per-structure) space:")
            print(f"    RMSE  = {m['total_rmse']:.6f}  per structure / component")
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
                      W_atom: tf.Tensor | None = None) -> tf.Tensor:
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
            W1          : [T, H]      output weights
            b1          : ()          scalar bias
            W0_pol..b1_pol : same shapes, for mode 2 only (None otherwise)
            W_atom      : [B, A, 3, Q] dipole kernel (mode 1) or [B, A, 6, Q]
                          polarizability kernel (mode 2); None → COO fallback

        Returns:
            predictions : [B, T_dim]  T_dim = 1 (PES), 3 (dipole), 6 (pol)
        """
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

        # Single-hidden backward chain (∂U/∂q):
        #   ∂U/∂h1 = W1, ∂U/∂a1 = activation'(h1, z1)·W1, ∂U/∂q = ∂U/∂a1·W0^T
        de_da = self._activation_grad(h1, z1) * W1_t
        de_dq = tf.add_n([
            tf.einsum('bah,qh->baq', de_da, W0_use[t]) * type_masks[t]
            for t in range(self.num_types)
        ])

        B = tf.shape(descriptors)[0]

        if W_atom is not None and self.cfg.target_mode in (1, 2):
            # Precomputed-kernel path (avoids the [P, Q] gather, and [C, P, Q]
            # once pfor vectorises it over candidates). Identical to the COO
            # path by the kernels' definitions:
            #   mode 1: dipole[b,s] = -Σ_{a,q} de_dq[b,a,q]·W_atom[b,a,s,q]
            #   mode 2: same contraction over the 6 rank-2 components, then the
            #           isotropic scalar ANN added on the diagonal.
            contracted = -tf.einsum('baq,basq->bs', de_dq, W_atom)  # [B, 3|6]
            if self.cfg.target_mode == 1:
                return contracted
            return contracted + self._pol_diag_add(
                self._pol_scalar_sum(descriptors, Z, atom_mask,
                                     W0_pol_use, b0_pol, W1_pol, b1_pol))

        # Standard COO path (no kernel supplied: spectroscopy, PES eval).
        box_inv = tf.linalg.inv(boxes)  # [B, 3, 3] — COO branch only
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
                                  W0_pol: tf.Tensor | None = None,
                                  b0_pol: tf.Tensor | None = None,
                                  W1_pol: tf.Tensor | None = None,
                                  b1_pol: tf.Tensor | None = None,
                                  U_pair: tf.Tensor | None = None,
                                  W_pre_angular: tf.Tensor | None = None) -> tf.Tensor:
        """Forward pass for C candidates × B structures using explicit batched GEMMs.

        Replaces vectorized_map for all three target modes.
        Both the input→hidden and hidden→descriptor matmuls are executed as
        single large GEMMs over all C candidates simultaneously:

            Forward:  [B*A, Q] @ [Q, C*H]    → [B*A, C*H] → [C, B, A, H]
            Backward: [C, B*A, H] @ [C, H, Q] → [C, B*A, Q]  (batched GEMM)

        Descriptors are assumed pre-scaled by the caller.

        Args:
            descriptors : [B, A, Q]
            W_atom      : [B, A, 3, Q] dipole kernel (mode 1) or [B, A, 6, Q]
                          polarizability kernel (mode 2); unused for mode 0
            Z           : [B, A]        type indices
            atom_mask   : [B, A]        1.0 real, 0.0 pad
            W0          : [C, T, Q, H]
            b0          : [C, T, H]
            W1          : [C, T, H]
            b1          : [C]
            W0_pol..b1_pol : same shapes, isotropic scalar ANN (mode 2 only)

        Returns:
            predictions : [C, B, T_dim]  T_dim = 1 (PES), 3 (dipole), 6 (pol)
        """
        # Q = dim seen by the matmul: Q_raw after the preprocess fold, else Q_new.
        Q = self.dim_q_forward
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
        if self.descriptor_mixing and U_pair is not None:
            W0 = self._W0_eff(W0, U_pair)
            if W0_pol is not None:
                W0_pol = self._W0_eff(W0_pol, U_pair)

        # Preprocess fold: W0 Q_new → Q_raw so the matmul below is uniform at
        # Q_raw and de_dq comes out at Q_raw for the raw-W_atom sum. Mutually
        # exclusive with mixing.
        if self.descriptor_preprocess_contract != "off":
            W0 = self._W0_preprocess_eff(W0, W_pre_override=W_pre_angular)
            if W0_pol is not None:
                W0_pol = self._W0_preprocess_eff(
                    W0_pol, W_pre_override=W_pre_angular)

        # ── Forward: input→hidden ─────────────────────────────────────────────
        # Per type: [B*A, Q] @ [Q, C*H] → [C, B, A, H]. One GEMM/type, not C.
        desc_flat = tf.reshape(descriptors, [B * A, Q])
        pre_h_terms = []
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
        W1_t_all = tf.reshape(tf.gather(W1, Z_flat, axis=1), [C, B, A, H_final])

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

        # ── Dipole / polarizability: backward matmul ──────────────────────────
        # ∂U/∂a1 = activation'(h1, z1)·W1.
        de_da = self._activation_grad(h1, z1) * W1_t_all      # [C, B, A, H]

        # Apply the type masks on the H-sized tensor and fuse the T per-type
        # GEMMs into one. Masking after the matmul would instead touch T full
        # [C,B,A,Q] tensors and hold them all live for the add_n — Q/H times
        # more traffic for the same result. Concatenating along H turns the
        # sum-over-types into a single contraction.
        de_da_masked = tf.concat(
            [de_da * type_masks[t][tf.newaxis] for t in range(T)],
            axis=-1)                                          # [C, B, A, T*H]
        W0_all = tf.reshape(tf.transpose(W0, [0, 1, 3, 2]),
                            [C, T * H, Q])                    # [C, T*H, Q]
        de_dq = tf.reshape(
            tf.matmul(tf.reshape(de_da_masked, [C, B * A, T * H]), W0_all),
            [C, B, A, Q])                                     # [C, B, A, Q]

        # W_atom [B, A, 3|6, Q]: pred[c,b,s] = -Σ_{a,q} de_dq[c,b,a,q]*W_atom[b,a,s,q]
        contracted = -tf.einsum('cbaq,basq->cbs', de_dq, W_atom)  # [C, B, 3|6]
        if self.cfg.target_mode == 1:
            return contracted

        # ── Polarizability: isotropic scalar ANN on the diagonal ──────────────
        # Same GEMM pattern as the main forward, on the second (scalar) ANN.
        pre_hp_terms = []
        for t in range(T):
            W0p_t     = W0_pol[:, t, :, :]                                         # [C, Q, H]
            W0p_t_mat = tf.reshape(tf.transpose(W0p_t, [1, 0, 2]), [Q, C * H])    # [Q, C*H]
            php_flat  = tf.matmul(desc_flat, W0p_t_mat)                           # [B*A, C*H]
            php       = tf.transpose(tf.reshape(php_flat, [B, A, C, H]), [2, 0, 1, 3])
            pre_hp_terms.append(php * type_masks[t][tf.newaxis])
        pre_hp = tf.add_n(pre_hp_terms)  # [C, B, A, H]

        b0p_all = tf.reshape(tf.gather(b0_pol, Z_flat, axis=1), [C, B, A, H])
        W1p_all = tf.reshape(tf.gather(W1_pol, Z_flat, axis=1), [C, B, A, H_final])

        h_pol = self.activation(pre_hp + b0p_all)
        h_pol = h_pol * atom_mask[tf.newaxis, :, :, tf.newaxis]
        F_pol = tf.reduce_sum(h_pol * W1p_all, axis=3) + b1_pol[:, tf.newaxis, tf.newaxis]
        F_pol = F_pol * atom_mask[tf.newaxis]
        scalar_sum = tf.reduce_sum(F_pol, axis=2)  # [C, B]

        return contracted + self._pol_diag_add(scalar_sum)

    def _scalar_rij_pow(self, rij2: tf.Tensor) -> tf.Tensor:
        """|r_ij|^N per-pair weight from the rij² primitive (N ≥ 1; avoids sqrt
        for even N). N=0 (self-pairs only) is handled by _dipole_pair_weight_*.
            N=1 → √rij²;  N=2 → rij²;  even≥4 → rij²^(N/2);  odd≥3 → rij²^((N-1)/2)·√rij²
        """
        return kernels.scalar_rij_pow(
            int(getattr(self.cfg, "dipole_rij_power", 2)), rij2)

    def _dipole_pair_weight_coo(self, rij2: tf.Tensor,
                                 pair_atom: tf.Tensor,
                                 pair_gidx: tf.Tensor) -> tf.Tensor:
        """Per-pair dipole weight, COO-pair interface. See kernels.pair_weight_coo."""
        return kernels.pair_weight_coo(
            int(getattr(self.cfg, "dipole_rij_power", 2)),
            rij2, pair_atom, pair_gidx)

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
        """[B, A, 3, Q] dipole geometry kernel. See kernels.precompute_dipole_kernel."""
        return kernels.precompute_dipole_kernel(
            int(getattr(self.cfg, "dipole_rij_power", 2)),
            grad_values, pair_struct, pair_atom, pair_gidx, positions, boxes, B, A)

    def _precompute_pol_kernel(self, grad_values: tf.Tensor,
                               pair_struct: tf.Tensor, pair_atom: tf.Tensor,
                               pair_gidx: tf.Tensor, positions: tf.Tensor,
                               boxes: tf.Tensor, B: tf.Tensor,
                               A: tf.Tensor) -> tf.Tensor:
        """[B, A, 6, Q] polarizability geometry kernel. See kernels.precompute_pol_kernel."""
        return kernels.precompute_pol_kernel(
            grad_values, pair_struct, pair_atom, pair_gidx, positions, boxes, B, A)

    def _neighbor_displacements_coo(self, positions: tf.Tensor, boxes: tf.Tensor,
                                    box_inv: tf.Tensor, pair_struct: tf.Tensor,
                                    pair_atom: tf.Tensor,
                                    pair_gidx: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """(dr [P,3], rij2 [P]) with MIC wrapping. See kernels.neighbor_displacements_coo."""
        return kernels.neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)

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

        scalar_sum = self._pol_scalar_sum(descriptors, Z, atom_mask,
                                          W0_pol, b0_pol, W1_pol, b1_pol)  # [B]

        # Tensor part: per-pair outer product, then segment-sum per structure
        pol_outer = -tf.einsum('ki,kj->kij', dr, forces_per_pair)  # [P, 3, 3]
        pol_flat  = tf.reshape(pol_outer, [-1, 9])                  # [P, 9]
        pol_mat_flat = tf.math.unsorted_segment_sum(
            pol_flat, pair_struct, num_segments=B)                  # [B, 9]
        pol_matrix = tf.reshape(pol_mat_flat, [B, 3, 3])

        pol = tf.stack([pol_matrix[:, i, j] for i, j in self._POL_COMPONENTS],
                       axis=1)  # [B, 6]
        return pol + self._pol_diag_add(scalar_sum)

    _POL_COMPONENTS = kernels.POL_COMPONENTS

    @staticmethod
    def _pol_diag_add(scalar_sum: tf.Tensor) -> tf.Tensor:
        """Broadcast the isotropic scalar ANN output onto the xx/yy/zz slots.

        Args:
            scalar_sum : [...]  trailing-axis-free scalar per structure

        Returns:
            [..., 6] with the scalar on the three diagonal components, 0 elsewhere
        """
        zeros = tf.zeros_like(scalar_sum)
        return tf.stack([scalar_sum, scalar_sum, scalar_sum, zeros, zeros, zeros],
                        axis=-1)

    def _pol_scalar_sum(self, descriptors: tf.Tensor, Z: tf.Tensor,
                        atom_mask: tf.Tensor, W0_pol: tf.Tensor,
                        b0_pol: tf.Tensor, W1_pol: tf.Tensor,
                        b1_pol: tf.Tensor) -> tf.Tensor:
        """Isotropic scalar ANN summed over atoms, single weight set.

        Args:
            descriptors : [B, A, Q]
            Z           : [B, A]
            atom_mask   : [B, A]
            W0_pol      : [T, Q, H]   (caller supplies U-absorbed weights)
            b0_pol      : [T, H]
            W1_pol      : [T, H]
            b1_pol      : ()

        Returns:
            scalar_sum : [B]
        """
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
        return tf.reduce_sum(F_pol, axis=1)                     # [B]
