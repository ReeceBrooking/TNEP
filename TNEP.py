from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Callable

from DescriptorBuilder import make_descriptor_builder
from SNES import SNES
from TNEPconfig import TNEPconfig


class TNEP(layers.Layer):
    """Per-type single-hidden-layer ANN for predicting energy, dipole, or polarizability.

    Forward pass per atom i with type t:
        a_i  = q_i @ W0[t] + b0[t]           # [num_neurons]
        h_i  = tanh(a_i)                      # [num_neurons]
        U_i  = h_i · W1[t] + b1              # scalar

    Weights:
        W0 : [num_types, dim_q, num_neurons]  input -> hidden
        b0 : [num_types, num_neurons]         hidden bias
        W1 : [num_types, num_neurons]         hidden -> scalar
        b1 : ()                               global scalar bias

    Prediction modes (cfg.target_mode):
        0 (PES)    : E = -sum_i U_i                                   -> scalar
        1 (Dipole) : μ = -sum_i sum_j |r_ij|² * (dU_i/dr_ij_vec)     -> [3]
        2 (Polar.) : α[6] via dual ANN (scalar + tensor)              -> [6]

    For mode 2 (polarizability), a second "scalar ANN" is added:
        W0_pol, b0_pol, W1_pol, b1_pol
    The scalar ANN computes per-atom F_pol -> isotropic diagonal.
    The tensor ANN (primary W0/b0/W1/b1) computes forces -> anisotropic virial.
    Output: [xx, yy, zz, xy, yz, zx]
    """

    def __init__(self,
                 cfg: TNEPconfig,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

        # Preprocessing contraction (phase 1 of 2): if enabled, run the
        # mutual-exclusion guards then override cfg.dim_q to the
        # contracted output dim BEFORE allocating W0 and downstream
        # Variables that key off cfg.dim_q. Phase 2 below allocates the
        # per-type coefficient Variable W_pre_*.
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
            # nep4_radial collapses the (n, n', species_pair) structure
            # into a flat (n'', l) axis at Q_new = n_max_out · L. Mixing
            # is supported via the l_aware architecture: blocks of size
            # n_max_out per angular momentum l, rotating radial channels
            # within each l. The general preprocess+mixing arch gate
            # below enforces arch='l_aware' (other archs raise).
        if self.descriptor_preprocess_contract != "off":
            # Mixing composes with preprocess only in the l_aware
            # architecture (where blocks are per (post-pair, l_post) and
            # naturally map to the contracted Q_new structure). Other
            # architectures still raise.
            if bool(getattr(cfg, "descriptor_mixing", False)):
                _arch = str(getattr(cfg, "descriptor_mixing_arch", "linear")).lower()
                if _arch != "l_aware":
                    raise NotImplementedError(
                        f"descriptor_mixing_arch={_arch!r} does not compose "
                        f"with descriptor_preprocess_contract in this build. "
                        f"Set arch to 'l_aware' or disable preprocess.")
            from DescriptorBuilderGPU import (
                descriptor_block_layout, descriptor_preprocess_layout)
            _layout_pre = descriptor_block_layout(cfg)
            self._preprocess_layout = descriptor_preprocess_layout(
                cfg, _layout_pre, self.descriptor_preprocess_contract)
            # Override cfg.dim_q so W0 below is allocated at the
            # contracted output dim.
            if not hasattr(cfg, "dim_q_raw") or cfg.dim_q_raw is None:
                cfg.dim_q_raw = int(cfg.dim_q)
            cfg.dim_q = int(self._preprocess_layout["dim_q_new"])
        else:
            self._preprocess_layout = None

        self.dim_q = cfg.dim_q
        # When preprocessing is on, the per-type ANN's W0 stores weights
        # in the CONTRACTED dim (self.dim_q = Q_new) but the forward
        # matmul folds the contraction in and operates on RAW descriptors
        # at Q_raw. Track Q_raw separately for forward-path use.
        if self.descriptor_preprocess_contract != "off":
            self.dim_q_forward = int(cfg.dim_q_raw)
        else:
            self.dim_q_forward = int(cfg.dim_q)
        self.num_types = cfg.num_types
        self.num_neurons = cfg.num_neurons
        self._H_final = cfg.num_neurons
        # Resolve the Keras activation callable. Any name supported by
        # `tf.keras.activations.get` is accepted at construction; the
        # backward derivative path (`_activation_grad`) will raise
        # NotImplementedError at first dipole / pol prediction if the
        # chosen activation isn't plumbed there.
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

        # Validate dipole contraction power (mode 1 only). Caught here
        # so configuration errors surface at model construction rather
        # than at first forward pass.
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

        # Optimizer constructed last so it can capture references to
        # the just-created weight tensors.
        # Optional descriptor-mixing layer (GPUMD c_nk analog).
        # U_pair is shared across central atom types; one mixing
        # square block per unordered neighbour-species pair. Block
        # sizes are pair-dependent (trivial compression mixes radial
        # channels across species pair boundaries), so U_pair is
        # padded to max_block_size and we track each pair's active
        # size separately.
        #
        # For efficient assembly of the full Q×Q mixing matrix, we
        # also precompute per-pair "placement" matrices P_p ∈
        # R^{bs × Q} that map the bs active features of pair p onto
        # their non-contiguous q-indices in the flat descriptor.
        # Then U_full = Σ_p P_p^T · U_pair[..., p, :bs, :bs] · P_p
        # is a clean einsum that broadcasts over any leading batch
        # dims (single-instance and per-candidate paths both work).
        self.descriptor_mixing = bool(getattr(cfg, "descriptor_mixing", False))
        self.descriptor_mixing_per_type = bool(
            getattr(cfg, "descriptor_mixing_per_type", False))
        self.descriptor_mixing_arch = str(
            getattr(cfg, "descriptor_mixing_arch", "linear")).lower()
        if self.descriptor_mixing_arch not in ("linear", "l_aware", "cross_pair_l"):
            raise ValueError(
                f"descriptor_mixing_arch={self.descriptor_mixing_arch!r} not in "
                "('linear', 'l_aware', 'cross_pair_l')")
        if self.descriptor_mixing:
            from DescriptorBuilderGPU import (
                descriptor_block_layout,
                descriptor_post_preprocess_block_layout)
            # When preprocess contraction is on, mixing operates at
            # Q_new (the contracted W0 storage dim), so the block
            # layout it needs is the POST-preprocess one. Otherwise
            # use the raw SOAP-turbo block layout at Q_raw.
            if self.descriptor_preprocess_contract != "off":
                _raw_layout = descriptor_block_layout(cfg)
                self._mix_layout = descriptor_post_preprocess_block_layout(
                    cfg, _raw_layout, self.descriptor_preprocess_contract)
            else:
                self._mix_layout = descriptor_block_layout(cfg)
            self._mix_pair_keys = self._mix_layout["pair_keys"]
            self._mix_num_pairs = len(self._mix_pair_keys)
            self._mix_max_block_size = int(self._mix_layout["max_block_size"])
            self._mix_block_sizes = [
                self._mix_layout["block_sizes"][k] for k in self._mix_pair_keys]
            self._mix_Q = int(self._mix_layout["dim_q"])
            T = cfg.num_types
            # Build per-pair placement matrices P[p] ∈ R^{bs_p × Q}. Used
            # by the "linear" arch; the "l_aware" arch builds finer
            # per-(pair, l) placement matrices below in addition.
            mix_P = []
            for p_idx, k in enumerate(self._mix_pair_keys):
                qidx = self._mix_layout["pair_q_index"][k]   # np.int32 [bs]
                bs = qidx.size
                P = np.zeros((bs, self._mix_Q), dtype=np.float32)
                P[np.arange(bs), qidx] = 1.0
                mix_P.append(tf.constant(P))
            self._mix_P = mix_P
            # Stacked projector [num_pairs, bs, Q] for the linear-arch
            # fast path. Available only when all pairs share the same bs
            # (the common case for fixed alpha_max / l_max). Replaces
            # num_pairs separate einsums with one batched einsum.
            if len(set(self._mix_block_sizes)) == 1:
                bs0 = self._mix_block_sizes[0]
                P_lin_stack = np.zeros(
                    (self._mix_num_pairs, bs0, self._mix_Q),
                    dtype=np.float32)
                for p_idx, k in enumerate(self._mix_pair_keys):
                    qidx = self._mix_layout["pair_q_index"][k]
                    P_lin_stack[p_idx, np.arange(bs0), qidx] = 1.0
                self._mix_P_stack = tf.constant(P_lin_stack)
                self._mix_P_uniform_bs = bs0
            else:
                self._mix_P_stack = None
                self._mix_P_uniform_bs = None

            if self.descriptor_mixing_arch == "linear":
                # U_pair stores the RESIDUAL V = U - I (deviation from
                # identity). The effective per-pair mixing matrix is
                # U_block = I_bs + V_block, so V starts at zero and the
                # model begins bit-identical to a mixing-disabled baseline.
                # Shape:
                #   shared    : [num_pairs, max_bs, max_bs]
                #   per-type  : [T, num_pairs, max_bs, max_bs]
                if self.descriptor_mixing_per_type:
                    shape = (T, self._mix_num_pairs,
                             self._mix_max_block_size, self._mix_max_block_size)
                else:
                    shape = (self._mix_num_pairs,
                             self._mix_max_block_size, self._mix_max_block_size)
                # Build N stacked mixing layers. Layer 0 keeps the legacy
                # name "U_pair" so save/load round-trips for N=1.
                self.U_pair = self.add_weight(
                    name="U_pair",
                    shape=shape,
                    initializer="zeros",
                    trainable=True,
                )
            elif self.descriptor_mixing_arch == "l_aware":
                # Per-(pair, l) residual blocks. Within a pair, only
                # radial channels at the same l mix; cross-l mixing is
                # forbidden by construction. α_eff_per_pair from the
                # descriptor layout gives the radial dimension; L =
                # l_max + 1.
                self._mix_alpha_per_pair = [
                    int(self._mix_layout["alpha_eff_per_pair"][k])
                    for k in self._mix_pair_keys]
                self._mix_max_alpha = int(self._mix_layout["max_alpha_eff"])
                # Use the layout's L_eff when present (post-preprocess
                # layouts collapse l>l_keep into a single slot, so
                # L_eff = l_keep + 1; the raw layout has no L_eff field
                # and uses cfg.l_max+1).
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
                # Stacked projector for the batched _U_full einsum fast
                # path: one tensor [num_pairs * L, α, Q] when α is uniform
                # across pairs (the common case). Replaces num_pairs × L
                # individual einsum launches with a single batched einsum.
                # See _U_full for the contraction.
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
                # V_pair_l shape:
                #   shared    : [num_pairs, L, max_α, max_α]
                #   per-type  : [T, num_pairs, L, max_α, max_α]
                # Padded to max_α so the storage has uniform stride;
                # padded rows/cols stay zero and never affect the math.
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
            else:  # cross_pair_l
                # One [N_l × N_l] residual matrix per angular momentum,
                # where N_l = Σ_p α_eff_p. Mixes radial channels at the
                # same l across all species pairs; cross-l mixing is
                # forbidden by construction. Strictly more expressive
                # than l_aware (l_aware ⊂ cross_pair_l: block-diagonal
                # cross_pair_l recovers l_aware).
                self._mix_L = int(cfg.l_max) + 1
                self._mix_N_per_l = int(self._mix_layout["N_per_l"])
                # Per-l placement matrices P_l ∈ R^{N_l × Q}.
                mix_P_l: list[tf.Tensor] = []
                for l in range(self._mix_L):
                    qidx = self._mix_layout["l_index"][l]   # [N_l]
                    P = np.zeros((self._mix_N_per_l, self._mix_Q),
                                 dtype=np.float32)
                    P[np.arange(self._mix_N_per_l), qidx] = 1.0
                    mix_P_l.append(tf.constant(P))
                self._mix_P_l = mix_P_l
                # Stacked projector [L, N_l, Q] — N_l is uniform across
                # l by descriptor_block_layout's invariant, so the stack
                # is always available for the batched _U_full fast path.
                P_stack = np.stack([P.numpy() for P in mix_P_l], axis=0)
                self._mix_P_l_stack = tf.constant(P_stack)
                # V shape:
                #   shared    : [L, N_l, N_l]
                #   per-type  : [T, L, N_l, N_l]
                # No padding needed — N_l is uniform across l.
                if self.descriptor_mixing_per_type:
                    shape = (T, self._mix_L,
                             self._mix_N_per_l, self._mix_N_per_l)
                else:
                    shape = (self._mix_L,
                             self._mix_N_per_l, self._mix_N_per_l)
                self.U_pair = self.add_weight(
                    name="U_pair",
                    shape=shape,
                    initializer="zeros",
                    trainable=True,
                )
        else:
            self.U_pair = None
            self._mix_P = []
            self._mix_block_sizes = []

        # Preprocessing contraction (phase 2 of 2): Variable allocation
        # and mutual-exclusion guards. cfg.dim_q has already been
        # overridden in phase 1 at the top of __init__ so W0 above is
        # sized at Q_new.
        if self.descriptor_preprocess_contract != "off":
            if self.descriptor_mixing and self.descriptor_mixing_arch != "l_aware":
                raise NotImplementedError(
                    f"descriptor_mixing_arch={self.descriptor_mixing_arch!r} "
                    f"does not compose with descriptor_preprocess_contract in "
                    f"this build. Set arch to 'l_aware' or disable preprocess.")
            init_scheme = str(getattr(cfg, "descriptor_preprocess_init", "mean"))
            T_pre = int(cfg.num_types)
            self.preprocess_per_type = bool(getattr(
                cfg, "descriptor_preprocess_per_type", True))
            if self.descriptor_preprocess_contract == "nep4_radial":
                # c tensor: rank-4 [T_centre, T_neighbour, n_max_out, α].
                # The linear-fold tail machinery below (Q_raw-shaped W_pre,
                # passthrough kept slots, per-q_raw init) does not apply;
                # we allocate c directly and precompute the bilinear fold
                # gather indices for `_W0_preprocess_eff`.
                _lay = self._preprocess_layout
                n_max_out = int(_lay["nep4_n_max_out"])
                alpha_max = int(_lay["nep4_alpha_max"])
                coef_shape = tuple(_lay["coef_shape"])
                full_coef_size = int(_lay["nep4_full_coef_size"])
                # Init policy for the c tensor: force GLOROT regardless of
                # the cfg flag. The 'mean' / 'sum' schemes set c to a
                # constant across all entries, which under the bilinear
                # fold  g[t, n'', l] = Σ_{n,n'} c·c·p  collapses every
                # n'' output channel to the SAME value (the n'' axis
                # drops out of c·c when c is constant). The 60-wide
                # compressed descriptor becomes rank-1 at gen 0 and SNES
                # has no signal to break the symmetry. Glorot's random
                # per-element init breaks that symmetry immediately.
                # The cfg default 'mean' is right for the linear-fold
                # modes (angular/species_pair/both) and for the w_l
                # angular-summed weights below (linear weighted sum, no
                # bilinear symmetry to collapse) — only the c init for
                # nep4_radial gets force-overridden here.
                if init_scheme in ("mean", "sum"):
                    print(
                        "[nep4_radial] forcing GLOROT init for c tensor "
                        f"(cfg requested descriptor_preprocess_init="
                        f"{init_scheme!r}). Constant c init causes "
                        "bilinear-fold symmetry collapse — all n'' "
                        "channels become identical at gen 0. The w_l "
                        "angular-summed slab keeps the requested scheme.")
                # c init is glorot unconditionally; w_l (later block)
                # respects the cfg's init_scheme.
                init_norm = float(_lay["coef_init_norm"])
                fan_in = max(1, alpha_max)
                fan_out = max(1, n_max_out)
                limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
                rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
                init_np = rng.uniform(
                    -limit, limit, size=coef_shape).astype(np.float32)
                if init_scheme not in ("mean", "sum", "glorot"):
                    raise ValueError(
                        f"descriptor_preprocess_init={init_scheme!r} not "
                        f"recognised; expected 'mean', 'sum', or 'glorot'.")
                self.W_pre_angular = tf.Variable(
                    init_np, trainable=False, name="W_pre_angular_nep4",
                    dtype=tf.float32)
                # All c entries are SNES μ-slots (no passthrough). Layout
                # mirrors the linear-fold tail so existing SNES wiring
                # (n_preprocess, flat indices, scatter M) keeps working
                # over the full c tensor.
                smask_full = np.ones(coef_shape, dtype=bool)
                self._preprocess_summed_mask = tf.constant(smask_full, dtype=tf.bool)
                flat_idx = np.arange(full_coef_size, dtype=np.int32)
                self._preprocess_summed_flat_idx = tf.constant(
                    flat_idx, dtype=tf.int32)
                self._preprocess_summed_count = int(full_coef_size)
                # Kept-template is zero (no passthrough); SNES scatters μ
                # values into every position each gen.
                base_template = np.zeros(coef_shape, dtype=np.float32)
                self._preprocess_kept_template = tf.constant(
                    base_template, dtype=tf.float32)
                _M_np = np.eye(full_coef_size, dtype=np.float32)
                self._preprocess_summed_scatter_M = tf.constant(
                    _M_np, dtype=tf.float32)
                self._preprocess_kept_template_size = full_coef_size
                # NEP4-specific precomputed gather indices for the bilinear
                # fold (consumed by `_W0_preprocess_eff`).
                self._nep4_n_max_out = n_max_out
                self._nep4_alpha_max = alpha_max
                self._nep4_L = int(_lay["L"])
                # Optional angular contraction stacked on top of the
                # bilinear fold. L_eff = l_keep + 1 collapses l ≥ l_keep
                # into a single learnable summed channel (mirrors the
                # linear "angular" preprocess mode). When l_keep ≥ L
                # (default), L_eff = L and N_sum_l = 0 → no contraction
                # and no extra learnable weights.
                self._nep4_l_keep = int(_lay["nep4_l_keep"])
                self._nep4_L_eff = int(_lay["nep4_L_eff"])
                self._nep4_N_sum_l = int(_lay["nep4_N_sum_l"])
                # Precompute the [n_max_out, L_eff] mid-shape constant used
                # in `_W0_preprocess_eff_nep4` to reshape W0 from
                # [..., T, n_max_out·L_eff, H] → [..., T, n_max_out, L_eff, H].
                # Without this the fold creates a fresh tf.constant per
                # call inside the @tf.function-traced predict_batch path.
                self._nep4_mid_shape = tf.constant(
                    [self._nep4_n_max_out, self._nep4_L_eff], dtype=tf.int32)
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
                # Angular-contraction weights (trainable summed-channel
                # weights for l ≥ l_keep). Only allocated when l_keep < L;
                # otherwise None and the fold reduces to the bilinear-only
                # case. Init: 'mean' (1/N_sum_l per slot) matches the
                # linear-angular convention; 'sum' (1.0) and 'glorot' (0
                # μ-init, σ at gen 0 supplies spread) also handled.
                #
                # SHARED across centre types. The angular weights pick
                # which l ≥ l_keep channels survive the summation — by
                # the same algebraic argument as mixing (any per-type
                # linear acting on the descriptor is absorbable into
                # W0[t]), per-type w_l adds zero expressive capacity over
                # shared. Sharing collapses N_sum_l·T → N_sum_l SNES
                # dims (saves 12 dims for T=3, N_sum_l=6).
                if self._nep4_N_sum_l > 0:
                    if init_scheme == "mean":
                        w_l_val = 1.0 / float(self._nep4_N_sum_l)
                        w_l_init = np.full(
                            (self._nep4_N_sum_l,), w_l_val, dtype=np.float32)
                    elif init_scheme == "sum":
                        w_l_init = np.ones(
                            (self._nep4_N_sum_l,), dtype=np.float32)
                    else:   # glorot
                        rng_l = np.random.default_rng(
                            int(getattr(cfg, "seed", 0)) ^ 0xA17EC0DE)
                        limit_l = float(np.sqrt(
                            6.0 / (self._nep4_N_sum_l + 1)))
                        w_l_init = rng_l.uniform(
                            -limit_l, limit_l,
                            size=(self._nep4_N_sum_l,)).astype(np.float32)
                    self.W_pre_angular_l = tf.Variable(
                        w_l_init, trainable=False,
                        name="W_pre_angular_nep4_lkeep", dtype=tf.float32)
                    # Precompute two constant projectors used by the fold
                    # to assemble W_ang[L_eff, L] (shared, no T axis):
                    #   _nep4_W_ang_kept: [L_eff, L] identity on the kept
                    #     rows (l_post < l_keep); zeros on the summed row.
                    #   _nep4_w_l_scatter: [N_sum_l, L_eff, L] with a 1 at
                    #     [j, l_keep, l_keep+j]; the einsum
                    #     'j,jpl->pl' against W_pre_angular_l places the
                    #     trainable weights into the summed row.
                    L_raw = self._nep4_L
                    L_eff = self._nep4_L_eff
                    l_keep = self._nep4_l_keep
                    kept_np = np.zeros((L_eff, L_raw), dtype=np.float32)
                    for l_post in range(l_keep):
                        kept_np[l_post, l_post] = 1.0
                    self._nep4_W_ang_kept = tf.constant(kept_np)
                    scat_np = np.zeros(
                        (self._nep4_N_sum_l, L_eff, L_raw), dtype=np.float32)
                    for j in range(self._nep4_N_sum_l):
                        scat_np[j, l_keep, l_keep + j] = 1.0
                    self._nep4_w_l_scatter = tf.constant(scat_np)
                else:
                    self.W_pre_angular_l = None
                    self._nep4_W_ang_kept = None
                    self._nep4_w_l_scatter = None
                # Layout invariant locked in for the L-batched matmul fast
                # path in `_W0_preprocess_eff_nep4`. The trivial-compression
                # walk in DescriptorBuilderGPU emits (n, n', l) with the
                # inner loop over l, so every kept (n, n') pair contributes
                # exactly L consecutive q entries — meaning Q_raw factors
                # exactly as Q_pair_kept · L. We stash Q_pair_kept and
                # check the invariant here so any future layout change
                # surfaces as an assertion fail at __init__, not as a
                # silent reshape mis-alignment in the hot path.
                _l_of_q_np = np.asarray(_lay["nep4_l_of_q"])
                _q_raw = int(_l_of_q_np.size)
                if _q_raw % self._nep4_L != 0:
                    raise AssertionError(
                        f"nep4_radial layout invariant broken: Q_raw={_q_raw}"
                        f" not divisible by L={self._nep4_L}. The L-batched"
                        " matmul fast path assumes l_of_q has period L"
                        " (inner-loop-over-l emit order).")
                _expected_l_of_q = np.tile(
                    np.arange(self._nep4_L, dtype=_l_of_q_np.dtype),
                    _q_raw // self._nep4_L)
                if not np.array_equal(_l_of_q_np, _expected_l_of_q):
                    raise AssertionError(
                        "nep4_radial layout invariant broken: l_of_q is"
                        " not the expected cyclic [0..L-1] pattern. The"
                        " L-batched matmul fast path in"
                        " _W0_preprocess_eff_nep4 requires the inner-loop"
                        "-over-l emit order produced by"
                        " descriptor_preprocess_layout.")
                self._nep4_Q_pair_kept = _q_raw // self._nep4_L
                # Linear-fold-only attributes left at None — branch-checked
                # in `_W0_preprocess_eff`.
                self._preprocess_q_to_q_new = None
                self._preprocess_scatter = None
                self.optimizer = SNES(self)
                return
            Q_raw_pre = int(self._preprocess_layout["coef_shape"][0])

            # Pull the per-q_raw classification from the layout.
            #   summed_mask: True where the q_raw contributes to a SUMMED
            #                output channel (multiple contributors)
            #                → has a learnable W_pre coefficient.
            #                False where the q_raw goes to a KEPT
            #                (passthrough) channel → W_pre fixed at 1.0
            #                (identity scaling).
            #   Shape: 1-D [Q_raw] for angular; 2-D [T, Q_raw] for
            #   species_pair / both.
            _summed = self._preprocess_layout.get("summed_q_raw_mask")
            if _summed is None:
                _summed = np.zeros((Q_raw_pre,), dtype=bool)
            _per_q = self._preprocess_layout.get("coef_init_per_q_raw")
            if _per_q is None:
                _scalar = float(self._preprocess_layout["coef_init_norm"])
                _per_q = np.full((Q_raw_pre,), _scalar, dtype=np.float32)

            # Build the W_pre init tensor honouring kept vs summed:
            # kept entries always start at 1.0 (passthrough); summed
            # entries start at the per-q_raw mean / sum / glorot value.
            def _build_init_2d() -> np.ndarray:
                """Return [T, Q_raw] init (used as the source before
                squeezing T for per_type=False)."""
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
                # Global W_pre [Q_raw]: collapse the T axis.
                # For 1-D layouts (angular) init is identical across t;
                # for 2-D layouts take the max (ghost entries are 0,
                # active entries carry the per-(t, q_raw) init value).
                init_np = np.max(init_2d_np, axis=0).astype(np.float32).copy()
            self.W_pre_angular = tf.Variable(
                init_np, trainable=False, name="W_pre_angular",
                dtype=tf.float32)

            # Stash summed mask for SNES (which exposes only the summed
            # entries to the μ vector). Shape mirrors W_pre's shape.
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
            # Flat indices into W_pre for the summed entries — used by
            # SNES to scatter μ values into the Variable each generation.
            flat_idx = np.flatnonzero(smask_full.reshape(-1)).astype(np.int32)
            self._preprocess_summed_flat_idx = tf.constant(flat_idx, dtype=tf.int32)
            self._preprocess_summed_count = int(flat_idx.size)
            # Base template (kept positions retain their init value, e.g.
            # 1.0; summed positions are zeroed and SNES scatters its μ
            # values into them each generation). Stored as a tf.constant
            # so reconstruction in the candidate path needs only a
            # tensor_scatter_nd_update, never a Variable read.
            base_template = np.where(smask_full, 0.0, init_np).astype(np.float32)
            self._preprocess_kept_template = tf.constant(
                base_template, dtype=tf.float32)
            # Precompute the one-hot scatter matrix used by SNES.
            # reconstruct_params_tf to broadcast summed entries into the
            # full W_pre tensor.  Static shape, ~6 MB for typical CHO
            # angular config (n_summed × base_size). Building this once
            # at __init__ avoids re-issuing a `tf.one_hot` kernel each
            # chunk (which fragmented the @tf.function trace).
            _base_flat_size = int(base_template.size)
            _M_np = np.zeros(
                (flat_idx.size, _base_flat_size), dtype=np.float32)
            _M_np[np.arange(flat_idx.size), flat_idx] = 1.0
            self._preprocess_summed_scatter_M = tf.constant(
                _M_np, dtype=tf.float32)
            self._preprocess_kept_template_size = _base_flat_size
            _map_np = self._preprocess_layout["q_raw_to_q_new"]
            self._preprocess_q_to_q_new = tf.constant(_map_np, dtype=tf.int32)
            # For per-type maps (species_pair), precompute a one-hot
            # scatter matrix M[T, Q_raw, Q_new] so _W0_preprocess_eff can
            # fold via einsum without per-type gather. Entries where the
            # map is -1 (q_raw doesn't involve type t) produce zero rows.
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

        Dispatches on `descriptor_mixing_arch`:
          - "linear" : one [bs_p × bs_p] block per pair, mixing all
                       (n, l) channels of that pair.
                  U_full = I_Q + Σ_p P_pᵀ · V_p · P_p
          - "l_aware": (l_max+1) [α_p × α_p] sub-blocks per pair, one
                       per angular momentum. Cross-l mixing forbidden.
                  U_full = I_Q + Σ_p Σ_l P_{p,l}ᵀ · V_{p,l} · P_{p,l}

        In both cases `tf.eye(Q)` broadcasts across leading batch dims
        so the same code handles single / per-candidate × shared /
        per-type. With V initialised at zero, U_full == I_Q at gen 0.
        """
        V = self.U_pair if U_pair is None else U_pair
        if self.descriptor_mixing_arch == "linear":
            # Fast path: uniform bs across pairs. One batched einsum over
            # P:[num_pairs, bs, Q], V:[..., num_pairs, bs, bs].
            if self._mix_P_stack is not None:
                bs = self._mix_P_uniform_bs
                # Slice V to the active bs×bs (padding stays zero anyway,
                # but the explicit slice avoids touching it).
                V_active = V[..., :bs, :bs]
                # Output q-axes labelled i, m. Contraction axes:
                # p=pair, j=bs (first projector), k=bs (second projector).
                V_full = tf.einsum(
                    'pji,...pjk,pkm->...im',
                    self._mix_P_stack, V_active, self._mix_P_stack)
                return tf.eye(self._mix_Q, dtype=V_full.dtype) + V_full
            # Fallback: non-uniform bs.
            parts = []
            for p_idx, (P, bs) in enumerate(zip(self._mix_P, self._mix_block_sizes)):
                V_block = V[..., p_idx, :bs, :bs]                            # [..., bs, bs]
                placed = tf.einsum('ji,...jk,kl->...il', P, V_block, P)      # [..., Q, Q]
                parts.append(placed)
            V_full = tf.add_n(parts)
            return tf.eye(self._mix_Q, dtype=V_full.dtype) + V_full
        if self.descriptor_mixing_arch == "l_aware":
            # V shape: [..., (T?), num_pairs, L, max_α, max_α].
            # Fast path: uniform α across pairs. Flatten (num_pairs, L)
            # → PL and run one batched einsum over the stacked projector
            # P:[PL, α, Q], V:[..., PL, α, α].
            if self._mix_P_ln_stack is not None:
                alpha = self._mix_P_ln_uniform_alpha
                PL = self._mix_num_pairs * self._mix_L
                # Slice off the (max_α − α) padding rows/cols.
                V_active = V[..., :alpha, :alpha]
                # Reshape (num_pairs, L) → PL. Preserve leading batch
                # dims (which may include candidate axis C and/or type
                # axis T) via dynamic-shape concat.
                new_shape = tf.concat(
                    [tf.shape(V_active)[:-4], [PL, alpha, alpha]], axis=0)
                V_flat = tf.reshape(V_active, new_shape)
                # Output q-axes labelled i, m. Contraction:
                # p=PL (pair, l), j=α (first proj), k=α (second proj).
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
        # cross_pair_l: V shape is [..., (T?), L, N_l, N_l]. N_l is
        # uniform across l, so the stacked projector is always available
        # — one batched einsum, no Python loop.
        # Contraction: a=L, j=N_l (first proj), k=N_l (second proj);
        # output q-axes labelled i, m.
        V_full = tf.einsum(
            'aji,...ajk,akm->...im',
            self._mix_P_l_stack, V, self._mix_P_l_stack)
        return tf.eye(self._mix_Q, dtype=V_full.dtype) + V_full

    def _compose_V_blocks(self, V_list: list) -> tf.Tensor:
        """Compose N stacked-mixing V tensors into ONE equivalent V.

        All three linear arches (linear / l_aware / cross_pair_l) have
        block-disjoint sub-block support — each (pair[, l]) sub-block
        lives in disjoint Q-coordinates, so P_p P_{p'}^T = δ_{pp'} I.
        Composing N layers therefore reduces to a per-sub-block product:

            U_total = I_Q + Σ_p P_p^T [(I + V_{N-1,p}) … (I + V_{0,p}) − I] P_p

        i.e. compose `(I_bs + V_k)` in the SMALL sub-block space (the
        trailing two dims of each V), then return an equivalent
        single-layer V that `_U_full` can fold into W0 in one shot.

        Layer ordering matches `_U_full_composed`: layer 0 is applied
        FIRST to the descriptor (q' = U_total q = U_{N-1} … U_0 q), so the
        sub-block matmul order is `(I + V_{N-1}) … (I + V_0)`.

        Padding (max_bs > bs for non-uniform arches) stays correct: the
        padded slots of V are zero, so (I + V) in those slots is the
        identity, products of identities remain identity, and (· − I)
        zeros them out again.
        """
        bs = V_list[0].shape[-1]
        I_block = tf.eye(bs, dtype=V_list[0].dtype)
        U_running = I_block + V_list[0]
        for k in range(1, len(V_list)):
            U_running = tf.matmul(I_block + V_list[k], U_running)
        return U_running - I_block

    def _W0_eff(self, W0: tf.Tensor,
                U_pair: tf.Tensor | None = None) -> tf.Tensor:
        """Pre-multiply W0 by U_full^T along the Q axis. Equivalent to
        mixing the descriptor (desc' = U_full · desc) but absorbs the
        mixing into the weights so the rest of the forward / backprop /
        dipole sum use raw descriptors and raw grad_values unchanged.

        Shapes (shared U_pair):
            U_pair [num_pairs, bs, bs]       + W0 [T, Q, H]
                                             → W0_eff [T, Q, H]
            U_pair [C, num_pairs, bs, bs]    + W0 [C, T, Q, H]
                                             → W0_eff [C, T, Q, H]
        Shapes (per-central-type U_pair):
            U_pair [T, num_pairs, bs, bs]    + W0 [T, Q, H]
                                             → W0_eff [T, Q, H]
            U_pair [C, T, num_pairs, bs, bs] + W0 [C, T, Q, H]
                                             → W0_eff [C, T, Q, H]

        With the residual V parameterisation (V init = 0 ⇒ U_full = I),
        W0_eff == W0 exactly at generation 0.
        """
        if not self.descriptor_mixing:
            return W0
        U_full = self._U_full(U_pair)
        if self.descriptor_mixing_per_type:
            return tf.einsum('...tqp,...tqh->...tph', U_full, W0)
        return tf.einsum('...qp,...tqh->...tph', U_full, W0)

    def _W0_preprocess_eff(self, W0: tf.Tensor,
                            W_pre_override: tf.Tensor | None = None,
                            W_pre_l_override: tf.Tensor | None = None) -> tf.Tensor:
        """Fold the angular preprocessing contraction into W0.

        Algebra: with `desc' = preprocess(desc_raw)` where
            desc'[t, q_new] = Σ_{q_raw} W_pre[t, q_raw] · 𝟙[q_to_q_new[q_raw] = q_new]
                                       · desc_raw[q_raw],
        the ANN forward `h = (W0)^T · desc'` (per type) is equivalent to
            h[h_idx] = Σ_{q_raw} desc_raw[q_raw] · W_pre[t, q_raw]
                                · W0[t, q_to_q_new[q_raw], h_idx]
        i.e. a per-type weight matrix at the RAW dim:
            W0_eff[t, q_raw, h_idx] = W_pre[t, q_raw]
                                    · W0[t, q_to_q_new[q_raw], h_idx]

        This lets the existing matmul code stay unchanged (operating at
        Q_raw) while W0 storage stays at Q_new — mirroring the mixing-
        layer fold in `_W0_eff`. Backward dq comes out at Q_raw directly,
        so the dipole contraction with the precomputed raw W_atom is also
        unchanged.

        Args:
          W0:             [(C,) T, Q_new, H]   weights at the contracted dim
          W_pre_override: [(C,) T, Q_raw]      per-type preprocess coefficients
                          (with optional leading candidate axis matching W0).
                          When None, falls back to self.W_pre_angular [T, Q_raw].

        Returns:
          W0_eff: [(C,) T, Q_raw, H]  weights at the raw dim
        """
        if self.descriptor_preprocess_contract == "off":
            return W0
        W_pre = (W_pre_override if W_pre_override is not None
                 else self.W_pre_angular)
        if self.descriptor_preprocess_contract == "nep4_radial":
            w_l = (W_pre_l_override if W_pre_l_override is not None
                   else self.W_pre_angular_l)
            return self._W0_preprocess_eff_nep4(W0, W_pre, w_l)
        # W0 storage: [(C,) T, Q_new, H]. Gather/scatter along Q_new to
        # produce [(C,) T, Q_raw, H], then weight by W_pre.
        if self._preprocess_scatter is None:
            # 1-D map (angular): shared mapping across t.
            W0_at_qraw = tf.gather(W0, self._preprocess_q_to_q_new, axis=-2)
        else:
            # 2-D map (species_pair / both): per-type scatter via einsum.
            W0_at_qraw = tf.einsum(
                'tqp,...tph->...tqh', self._preprocess_scatter, W0)
        # Multiply by W_pre.
        #   per_type=True : W_pre [(C,) T, Q_raw] → factor [(C,) T, Q_raw, 1]
        #   per_type=False: W_pre [(C,) Q_raw]    → factor [(C,) Q_raw, 1]
        #                   needs a T-axis singleton inserted (at the index
        #                   of Q_raw in the post-newaxis shape = rank - 2)
        #                   so the broadcast against W0_at_qraw works for
        #                   both with- and without- candidate dim.
        factor = W_pre[..., tf.newaxis]
        if not self.preprocess_per_type:
            factor = tf.expand_dims(factor, axis=factor.shape.rank - 2)
        return W0_at_qraw * factor

    def _W0_preprocess_eff_nep4(self, W0: tf.Tensor,
                                 c: tf.Tensor,
                                 w_l: tf.Tensor | None = None) -> tf.Tensor:
        """NEP4 bilinear (rank-1 outer-product) fold of W0.

        Implements the equation
            g[t, n'', l] = Σ_{n, n'} c[t, s(n), n'', k(n)]
                                  · c[t, s(n'), n'', k(n')]
                                  · p[n, n', l]
        as a transformation of W0 from the [n'', l]-indexed storage at
        Q_new = n_max_out · L_eff to the [n, n', l]-indexed raw-descriptor
        layout at Q_raw, so the existing matmul code can continue to
        operate against the unmodified raw descriptor:
            U = W0_eff[t, q_raw] · desc_raw[q_raw]
              ≡ W0[t, q_new(n'', l_post)] · g[t, n'', l_post]
              ≡ W0[t, q_new(n'', l_post(q))] · Σ_{n,n'} c·c · p

        When the optional angular contraction is active (N_sum_l > 0),
        L_eff = l_keep + 1 < L and `w_l[T, N_sum_l]` supplies the
        trainable weights of the single summed channel that aggregates
        l ≥ l_keep. The fold then expands W0 from L_eff back to L by
        building W_ang[T, L_eff, L] (identity on the kept rows, the
        learned weights on the summed row) and contracting it into the
        weight tensor in a single einsum.

        Args:
          W0: [(C,) T, n_max_out · L_eff, H]   weights at Q_new
          c:  [(C,) T_centre, T_neighbour, n_max_out, α]   NEP4 coeffs
          w_l: [(C,) T_centre, N_sum_l]   angular-summed weights,
               or None when l_keep ≥ L (no angular contraction)

        Returns:
          W0_eff: [(C,) T, Q_raw, H]   weights at the raw descriptor dim
        """
        T_c = int(self.cfg.num_types)
        T_n = T_c
        n_max_out = int(self._nep4_n_max_out)
        L_ = int(self._nep4_L)
        alpha = int(self._nep4_alpha_max)
        Q_pair_kept = int(self._nep4_Q_pair_kept)
        # ── Build AB = c[..,n(q)] · c[..,n'(q)] (the rank-1 cc piece) ────────
        # Rearrange c so the (T_neighbour, α) axes become a single flat
        # axis aligned with the precomputed `nep4_n_global` index map.
        # c [..., T_c, T_n, n_max_out, α] → [..., T_c, T_n, α, n_max_out]
        # then flatten T_n·α → [..., T_c, T_n·α, n_max_out].
        has_C = (c.shape.rank == 5)
        if has_C:
            c_perm = tf.transpose(c, perm=[0, 1, 2, 4, 3])
            c_flat = tf.reshape(c_perm, [-1, T_c, T_n * alpha, n_max_out])
        else:
            c_perm = tf.transpose(c, perm=[0, 1, 3, 2])
            c_flat = tf.reshape(c_perm, [T_c, T_n * alpha, n_max_out])
        A_a = tf.gather(c_flat, self._nep4_n_global, axis=-2)
        A_b = tf.gather(c_flat, self._nep4_np_global, axis=-2)
        AB = A_a * A_b   # [..., T_c, Q_raw, n_max_out]

        # ── L-batched matmul (memory-light) ──────────────────────────────────
        # The layout walk in descriptor_preprocess_layout("nep4_radial") emits
        # (n, n', l) with l as the inner loop, so the q axis factors exactly
        # as Q_raw = Q_pair_kept · L (locked in by the assertion at __init__).
        # That lets us reshape AB's q axis to [Q_pair_kept, L] and contract
        # n'' AND l in a single einsum, without ever materialising the
        # ~Q_raw/L-fold replicated `W0_at_q` intermediate. For the legacy
        # gather path that intermediate was the dominant memory cost (e.g.
        # ~650 MB f32 at C=100, T=3, n_max_out=60, Q_raw=300, H=30); this
        # path peaks at the W0_eff_pl result which is ~Q_pair_kept × L × H /
        # (n_max_out × Q_raw) ≈ 30× smaller.
        W0_shape = tf.shape(W0)
        H_ = W0_shape[-1]
        leading = W0_shape[:-2]
        # W0:     [..., T, n_max_out·L_eff, H]
        # W0_NLp: [..., T, n_max_out, L_eff, H]   (l_post is its own axis)
        new_shape = tf.concat(
            [leading, self._nep4_mid_shape, tf.reshape(H_, [1])], axis=0)
        W0_NLH = tf.reshape(W0, new_shape)
        # ── Optional angular expansion L_eff → L ─────────────────────────────
        # When the bilinear fold is stacked with an angular contraction
        # (N_sum_l > 0 ⇒ L_eff = l_keep + 1 < L), W0 is stored at the
        # collapsed L_eff. To fold it against the raw descriptor we have
        # to expand its l axis back to L using the learned summed-channel
        # weights:
        #   W_ang[t, l_post, l]
        #     = δ(l_post, l)                      for l_post < l_keep,
        #     = w_l[t, l - l_keep]                for l_post = l_keep,
        #                                         l ≥ l_keep,
        #     = 0                                 otherwise.
        # Built without any per-call allocation by fusing the precomputed
        # kept-row identity (`_nep4_W_ang_kept`) with the scattered
        # learned weights (`_nep4_w_l_scatter`).
        if self._nep4_N_sum_l > 0:
            # w_l is shared across centre types — shape [N_sum_l] (single)
            # or [C, N_sum_l] (batched). W_ang ends up [L_eff, L] or
            # [C, L_eff, L]; the W0 expansion einsum broadcasts over t.
            if has_C:
                W_ang_summed = tf.einsum(
                    'Cj,jpl->Cpl', w_l, self._nep4_w_l_scatter)
                W_ang = self._nep4_W_ang_kept + W_ang_summed  # [C, L_eff, L]
                W0_NLH = tf.einsum('Cjl,CtNjh->CtNlh', W_ang, W0_NLH)
            else:
                W_ang_summed = tf.einsum(
                    'j,jpl->pl', w_l, self._nep4_w_l_scatter)
                W_ang = self._nep4_W_ang_kept + W_ang_summed  # [L_eff, L]
                W0_NLH = tf.einsum('jl,tNjh->tNlh', W_ang, W0_NLH)
        # AB:    [..., T_c, Q_raw, n_max_out]
        # AB_pl: [..., T_c, Q_pair_kept, L, n_max_out]   (q axis factored)
        ab_shape = tf.concat(
            [tf.shape(AB)[:-2],
             tf.constant([Q_pair_kept, L_], dtype=tf.int32),
             tf.reshape(tf.shape(AB)[-1], [1])], axis=0)
        AB_pl = tf.reshape(AB, ab_shape)
        # Sum over n'' (index N) for each l; l is shared between operands
        # so the einsum treats it as a batch dim (L independent matmuls).
        #   AB_pl    [..., t, p, l, N]
        #   W0_NLH   [..., t, N, l, h]
        # → W0_eff_pl[..., t, p, l, h]
        W0_eff_pl = tf.einsum('...tplN,...tNlh->...tplh', AB_pl, W0_NLH)
        # Reshape q_pair · L back to flat Q_raw (no copy). `leading` is
        # tf.shape(W0)[:-2] which already includes the T axis (W0 is
        # [..., T, Q_new, H]) — we only need to append [Q_raw, H].
        eff_shape = tf.concat(
            [leading,
             tf.constant([Q_pair_kept * L_], dtype=tf.int32),
             tf.reshape(H_, [1])], axis=0)
        W0_eff = tf.reshape(W0_eff_pl, eff_shape)
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
        # Iterative-encoder front-end: under train_encoder=True, the
        # caller-supplied descriptors/gradients are still Q_raw-shaped.
        # Apply the live encoder before any W0 einsum so W0 (sized at
        # Z) sees the right input. Static-mode callers pass pre-encoded
        # tensors → no-op here.
        _enc = getattr(self.cfg, "_encoder", None)
        if _enc is not None and bool(getattr(self.cfg, "train_encoder", False)):
            _q_raw_enc = int(getattr(self.cfg, "_encoder_q_raw"))
            _enc_chunk = int(getattr(self.cfg, "encoder_chunk_rows", 4096))
            # descriptors: [A, Q_raw] → [A, Z]; single batch is small
            # (one structure), no chunking needed.
            descriptors = _enc.encode(descriptors)
            # gradients: [A, M, 3, Q_raw] → [A, M, 3, Z]. Flatten,
            # chunked-encode (M·3 can be large for big neighbor lists),
            # subtract affine offset, restore shape.
            A_p = int(tf.shape(gradients)[0])
            M_p = int(tf.shape(gradients)[1])
            gv_flat = tf.reshape(gradients, [A_p * M_p * 3, _q_raw_enc])
            gv_z_flat = self._encode_chunked(
                _enc, gv_flat, _q_raw_enc, _enc_chunk)
            b_offset = _enc.encode(
                tf.zeros([1, _q_raw_enc], dtype=tf.float32))
            Z_p = int(tf.shape(gv_z_flat)[-1])
            gradients = tf.reshape(gv_z_flat - b_offset, [A_p, M_p, 3, Z_p])

        # Absorb U_pair^T into W0 (and W0_pol below) once per call so
        # both the forward and calc_forces use the U-folded weights.
        # When descriptor_mixing is disabled, _W0_eff is a no-op.
        W0_eff = self._W0_eff(self.W0)
        # Expand W0 from Q_new back to Q_raw when the preprocess
        # contraction is on (mirrors validate()/score()).
        if self.descriptor_preprocess_contract != "off":
            W0_eff = self._W0_preprocess_eff(W0_eff)

        # Gather per-type weights for each atom
        W0_t = tf.gather(W0_eff, Z)    # [A, dim_q, H]
        b0_t = tf.gather(self.b0, Z)   # [A, H]
        W1_t = tf.gather(self.W1, Z)   # [A, H]

        # Hidden layer: h_i = activation(z_i),  z_i = q_i @ W0[t_i] + b0[t_i]
        # Keep z (pre-activation) for the swish-side backward chain rule;
        # tanh discards it (1 − h² suffices).
        z = tf.einsum('nd,ndh->nh', descriptors, W0_t) + b0_t   # [A, H]
        h = self.activation(z)                                   # [A, H]
        # Mask out padded atoms
        h = h * atom_mask[:, tf.newaxis]                   # [A, H]

        # Target-centering inverse: add the training-set mean back so
        # `predict` returns predictions in original (un-centered) units.
        # The shift depends only on the data-pipeline convention:
        #   - target_mode==1 AND scale_targets : mean is per-atom space,
        #                                        shift by mean * num_atoms
        #   - otherwise (mode 0 energy, mode 2 polarisability, or
        #     mode 1 without scale_targets)   : mean is total space,
        #                                        shift by mean
        # This matches assemble_data_dict's gating (data.py:375), which
        # only divides by num_atoms when target_mode==1 AND scale_targets.
        _do_uncenter = (bool(getattr(self.cfg, "target_centering", False))
                        and getattr(self.cfg, "_target_mean", None) is not None)
        if _do_uncenter:
            _mean_arr = np.asarray(self.cfg._target_mean, dtype=np.float32)
            _shift_per_atom = (self.cfg.target_mode == 1
                               and bool(getattr(self.cfg, "scale_targets", False)))

        if self.cfg.target_mode == 0:
            # PES: E = -sum_i (h_i . W1[t_i] + b1)
            E_per_atom = tf.reduce_sum(h * W1_t, axis=1) + self.b1  # [A]
            E_per_atom = E_per_atom * atom_mask                       # zero padding
            # H-center skip: exclude H atoms from the energy sum. Their
            # descriptor is zero (no builder was called for them) so the
            # bias-driven U_H would otherwise contaminate the total.
            if bool(getattr(self.cfg, "skip_h_centers", False)):
                E_per_atom = E_per_atom * tf.cast(Z != 1, tf.float32)
            E = tf.reduce_sum(E_per_atom)
            out = tf.expand_dims(-E, axis=0)  # [1]
            if _do_uncenter:
                # Energy is always total-space (data pipeline never
                # divides E targets by num_atoms), so shift by `mean`.
                out = out + tf.constant(_mean_arr)
            return out

        # Modes 1 and 2 need forces
        forces = self.calc_forces(h, gradients, W1_t, W0_t, neighbor_mask,
                                  z=z)  # [A, M, 3]

        if self.cfg.target_mode == 1:
            # Dipole contraction. `cfg.dipole_rij_power` selects the
            # weighting:
            #   N >= 1 : μ = -Σ_pair |r_ij|^N · F_ij
            #   N == 0 : μ = -Σ_i de_dq[i] · grad_values[i, i]   (self-only)
            # The dispatcher returns the appropriate per-pair scalar
            # weight for the chosen branch.
            _, rij = self._neighbor_displacements_single(
                positions, box, grad_index)
            rij_n = (self._dipole_pair_weight_padded(tf.square(rij), grad_index)
                     * neighbor_mask)                                     # [A, M]
            dipole_contribs = rij_n[:, :, tf.newaxis] * forces            # [A, M, 3]
            dipole = -tf.reduce_sum(dipole_contribs, axis=[0, 1])         # [3]
            if _do_uncenter:
                if _shift_per_atom:
                    num_atoms = tf.reduce_sum(atom_mask)
                    dipole = dipole + tf.constant(_mean_arr) * num_atoms
                else:
                    dipole = dipole + tf.constant(_mean_arr)
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
            # H-center skip: zero F_pol for H atoms (their descriptor is
            # zero so F_pol would otherwise be bias-driven garbage).
            if bool(getattr(self.cfg, "skip_h_centers", False)):
                F_pol = F_pol * tf.cast(Z != 1, tf.float32)
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
            if _do_uncenter:
                # Polarisability targets are total-space (no scale_targets
                # path for mode=2), so the mean is total-space too.
                pol = pol + tf.constant(_mean_arr)
            return pol

    def _activation_grad(self, h: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Derivative of the activation w.r.t. its input, evaluated at z.

        The dipole / polarisability backward chain rule needs `dh/dz`
        applied as a Hadamard factor to `de/dh`. For tanh, dh/dz = 1 − h²
        depends only on the activation output. For swish/silu, dh/dz also
        depends on the pre-activation z, so callers MUST supply it (or
        pass `z = None` to fall back to the tanh formula — only valid
        when self._activation_name == 'tanh').

        Args:
            h: activation output, shape broadcastable to z.
            z: pre-activation (`W0·q + b0` for the first hidden layer,
               `W0_2·h1 + b0_2` for the second). Required for swish.

        Returns:
            tensor with the same shape as h, equal to dh/dz.

        Formulas:
            tanh : dh/dz = 1 − h²                (h itself encodes z's tanh)
            swish: dh/dz = σ(z) · (1 + z · (1 − σ(z)))
                   equivalently σ(z) + h · (1 − σ(z)).  Both forms cost one
                   sigmoid; the (1+z·(1−σ)) form is used because it avoids
                   the catastrophic-cancellation case at z ≫ 0 where
                   `1 − σ(z) ≈ 0` and `h ≈ z` would multiply to lose
                   precision.
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
        """Compute dU_i/dR_j for every atom i and its neighbours j via chain rule.

        Vectorized version — no Python loops. Uses padded gradient tensors.

        Args:
            h              : [N, H]              hidden activations f(z)
            gradients      : [N, M, 3, dim_q]   padded descriptor gradients
            W1_t           : [N, H]              per-atom output weights
            W0_t           : [N, dim_q, H]       per-atom input weights
            neighbor_mask  : [N, M]              1.0 for real neighbors, 0.0 for padding
            z              : [N, H]              pre-activation (W0·q + b0).
                                                 Required when self._activation_name
                                                 != 'tanh'.

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
        """Train the model.

        Dispatches on `cfg.optimizer`:
          - "snes" (default): population-based SNES, supports non-smooth
            losses and high-dim search distributions. Delegates to the
            SNES instance held on `self.optimizer`.
          - "adam":  gradient-based Adam(W). One forward + one backward
            per epoch via tf.GradientTape. Skips the SNES population
            machinery entirely. See `_fit_adam` for the implementation.

        Args:
            train_data    : dict with keys descriptors, gradients, grad_index,
                            positions, Z_int, targets, boxes (lists over structures)
            val_data      : same structure, used for validation each generation
            plot_callback : optional callable(history, gen) for periodic plotting
            resume_state  : optional dict from `model_io.load_checkpoint`,
                            carries SNES distribution + best-val + history +
                            RNG state. When provided, training continues from
                            `resume_state['last_gen'] + 1`. Not supported for
                            the Adam path yet — will raise if both are set.

        Returns:
            history         : dict with keys generation, train_loss, val_loss (lists)
            final_model     : TNEP model with weights from the last generation
            best_val_model  : TNEP model with weights from the best validation generation
        """
        optimizer = str(getattr(self.cfg, "optimizer", "snes")).lower()
        if optimizer not in ("snes", "adam"):
            raise ValueError(
                f"cfg.optimizer={optimizer!r} not in ('snes', 'adam').")
        if optimizer == "adam":
            return self._fit_adam(train_data, val_data,
                                   plot_callback=plot_callback,
                                   resume_state=resume_state)
        history, final_model, best_val_model = self.optimizer.fit(
            train_data, val_data, plot_callback=plot_callback,
            resume_state=resume_state)
        return history, final_model, best_val_model

    def _encode_chunked(self, encoder, x_flat, q_raw_enc, chunk_rows: int):
        """Apply `encoder.encode` to a [N, q_raw_enc] tensor in chunks.

        The Willatt / bilinear encoders materialise an O(N · G_T² · L)
        dense scatter inside `encode`, which OOMs the GPU for the
        grad_values flat tensor (N = P · 3, often 60k+). Chunking the
        N-axis bounds peak memory to one chunk's worth of dense scatter
        without changing the result. The output is a tf.concat over the
        chunked results; gradients flow back through every chunk to all
        encoder variables, so the iterative-encoder backprop is exact.

        REQUIRES eager mode. If this function is called inside a traced
        `@tf.function`, the chunk loop silently disables itself (N is
        a symbolic tensor with no .numpy()), and the unchunked encode
        OOMs at first batch. Raise loudly instead.
        """
        N = tf.shape(x_flat)[0]
        if chunk_rows is None or chunk_rows <= 0:
            return encoder.encode(x_flat)
        # We need a python-int total to drive the loop; the typical
        # caller (`_forward_loss`) reshapes to a known-eager N.
        if not tf.executing_eagerly():
            raise RuntimeError(
                "_encode_chunked must run in eager mode (Python chunk "
                "loop needs a concrete N). If you've wrapped a caller "
                "in `@tf.function` for graph speed, either move the "
                "encoder application outside the function or replace "
                "this with a `tf.while_loop` over chunks.")
        N_py = int(N.numpy())
        chunks: list = []
        start = 0
        while start < N_py:
            end = min(start + int(chunk_rows), N_py)
            chunks.append(encoder.encode(x_flat[start:end]))
            start = end
        return tf.concat(chunks, axis=0)

    def _encode_chunk_for_inference(self, descriptors_raw, grad_values_raw,
                                     encoder, q_raw_enc, chunk_rows: int):
        """Apply the live encoder to a single chunk's descriptors and
        grad_values. Used by `score()`, `predict()`, and trajectory
        evaluators under iterative-encoder mode. No tape — these paths
        are inference-only.
        """
        S = int(tf.shape(descriptors_raw)[0])
        A = int(tf.shape(descriptors_raw)[1])
        P = int(tf.shape(grad_values_raw)[0])
        d_flat = tf.reshape(descriptors_raw, [S * A, q_raw_enc])
        z_flat = self._encode_chunked(
            encoder, d_flat, q_raw_enc, chunk_rows)
        Z = int(tf.shape(z_flat)[-1])
        d_enc = tf.reshape(z_flat, [S, A, Z])
        gv_flat = tf.reshape(grad_values_raw, [P * 3, q_raw_enc])
        gv_z_flat = self._encode_chunked(
            encoder, gv_flat, q_raw_enc, chunk_rows)
        b_offset = encoder.encode(
            tf.zeros([1, q_raw_enc], dtype=tf.float32))
        gv_enc = tf.reshape(gv_z_flat - b_offset, [P, 3, Z])
        return d_enc, gv_enc

    def _encode_val_data_iterative(self, val_data, encoder, q_raw_enc,
                                    chunk_rows: int = 4096):
        """Apply the live encoder to padded val_data once and return a
        Z-dim version, for the iterative Adam path's validation pass.

        Stateless on val_data — the input dict is not mutated. The
        encoder Variables are read at call time, so subsequent encoder
        updates produce a different result on the next validation tick.

        `chunk_rows` bounds the N-axis of each encode call to avoid the
        multi-GB Willatt scatter blow-up that the unchunked path
        triggers on real val_data (S·A + P·3 rows in one go).
        """
        d_raw = val_data["descriptors"]
        gv_raw = val_data["grad_values"]
        S = int(tf.shape(d_raw)[0])
        A = int(tf.shape(d_raw)[1])
        P = int(tf.shape(gv_raw)[0])
        # Run outside any tape — val is read-only and we want the lowest
        # peak memory possible. The encoder still uses live trainable
        # variables, so the val metric tracks each Adam step.
        d_flat = tf.reshape(d_raw, [S * A, q_raw_enc])
        z_flat = self._encode_chunked(
            encoder, d_flat, q_raw_enc, chunk_rows)
        Z = int(tf.shape(z_flat)[-1])
        d_enc = tf.reshape(z_flat, [S, A, Z])
        gv_flat = tf.reshape(gv_raw, [P * 3, q_raw_enc])
        gv_z_flat = self._encode_chunked(
            encoder, gv_flat, q_raw_enc, chunk_rows)
        b_offset = encoder.encode(
            tf.zeros([1, q_raw_enc], dtype=tf.float32))
        gv_enc = tf.reshape(gv_z_flat - b_offset, [P, 3, Z])
        new = dict(val_data)
        new["descriptors"] = d_enc
        new["grad_values"] = gv_enc
        return new

    def _fit_adam(self, train_data: dict[str, tf.Tensor],
                  val_data: dict[str, tf.Tensor],
                  plot_callback: Callable | None = None,
                  resume_state: dict | None = None) -> tuple:
        """Gradient-based training path. See `fit` for the public API.

        The forward path is the standard `predict_batch` after folding
        any descriptor-mixing and preprocessing into W0 via `_W0_eff` /
        `_W0_preprocess_eff`. Loss = `per_structure_error` on the
        target residual, summed and averaged. Adam updates only the
        Keras-trainable variables (W0/b0/W1/b1, optionally W0_pol etc.,
        and U_pair when descriptor_mixing is on). W_pre_angular is
        registered as non-trainable (it's managed exclusively by SNES)
        so Adam leaves it untouched — the preprocess fold uses the
        frozen init in that case.
        """
        import time
        from loss_functions import per_structure_error, squared_error_per_structure
        from data import prefetched_chunks

        if resume_state is not None:
            raise NotImplementedError(
                "Adam path does not yet support checkpoint resume; clear "
                "resume_state or switch back to cfg.optimizer='snes'.")

        cfg = self.cfg
        n_epochs = int(cfg.num_generations)
        loss_type = str(getattr(cfg, "loss_type", "mse")).lower()
        huber_delta = float(getattr(cfg, "huber_delta", 1e-3))
        val_interval = max(1, int(getattr(cfg, "val_interval", 1)))
        batch_size = getattr(cfg, "batch_size", None)
        S_train = int(train_data["num_atoms"].shape[0])

        # Build Adam / AdamW from cfg.
        wd = float(getattr(cfg, "adam_weight_decay", 0.0))
        if wd > 0.0:
            adam = tf.keras.optimizers.AdamW(
                learning_rate=float(cfg.adam_lr),
                weight_decay=wd,
                beta_1=float(cfg.adam_beta1),
                beta_2=float(cfg.adam_beta2),
                epsilon=float(cfg.adam_epsilon),
                clipnorm=cfg.adam_grad_clip)
        else:
            adam = tf.keras.optimizers.Adam(
                learning_rate=float(cfg.adam_lr),
                beta_1=float(cfg.adam_beta1),
                beta_2=float(cfg.adam_beta2),
                epsilon=float(cfg.adam_epsilon),
                clipnorm=cfg.adam_grad_clip)

        train_vars = list(self.trainable_variables)
        if len(train_vars) == 0:
            raise RuntimeError(
                "Model has no trainable variables — Adam has nothing to "
                "optimise. Check that W0/b0/W1/b1 were built.")

        # Iterative encoder front-end: apply the encoder on each batch
        # inside the GradientTape so its trainable variables (Willatt u,
        # per-l Dense kernels, etc.) co-adapt with TNEP. The encoder's
        # affine offset b = encode(0) is what makes the gradient-side
        # contraction work — for any linear f(x) = J·x + b,
        #   J · gv = f(gv) − f(0)
        # which we evaluate exactly each step via a single extra forward
        # pass on zeros. Static preprocess (train_encoder=False) skips
        # this entire block; the descriptors/grad_values already arrived
        # encoded.
        encoder = getattr(cfg, "_encoder", None)
        iterative_encoder = (encoder is not None
                              and bool(getattr(cfg, "train_encoder", False)))
        if iterative_encoder:
            enc_vars = list(encoder.trainable_variables)
            train_vars = train_vars + enc_vars
            q_raw_enc = int(cfg._encoder_q_raw)
            print(f"[adam] iterative encoder: {len(enc_vars)} encoder "
                  f"variables ({int(sum(np.prod(v.shape) for v in enc_vars)):,d} "
                  f"params) co-train with TNEP.")
        else:
            q_raw_enc = None

        # Reuse SNES.validate for the validation pass. It already
        # handles the descriptor-mixing / preprocess folds, target
        # scaling, chunk streaming, and pol_weights — and runs against
        # the model's live variables when called without mu_tf.
        snes_helper = self.optimizer

        # History schema mirrors SNES output so downstream consumers
        # (plotting, csv, model_io) don't need to special-case.
        history = {
            "generation": [],
            "train_loss": [],
            "train_rmse": [],
            "val_loss": [],
            "L1": [], "L2": [],
            "best_rmse": [], "worst_rmse": [],
            "sigma_min": [], "sigma_max": [],
            "sigma_mean": [], "sigma_median": [],
            "timing": {
                "sample_batch": [],
                "evaluate": [],
                "rank_update": [],
                "validate": [],
                "overhead": [],
            },
        }
        best_val_loss = float("inf")
        best_vars_snapshot: list | None = None
        rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))

        def _sample_batch():
            if batch_size is None:
                return train_data
            idx = rng.choice(S_train, size=int(batch_size), replace=False)
            idx_tf = tf.constant(idx.astype(np.int32))
            struct_keys = ["descriptors", "positions", "Z_int", "boxes",
                           "num_atoms", "targets", "atom_mask"]
            if "types_contained" in train_data:
                struct_keys.append("types_contained")
            batch = {k: tf.gather(train_data[k], idx_tf) for k in struct_keys}
            pair_starts = tf.gather(train_data["struct_ptr"], idx_tf)
            pair_ends = tf.gather(train_data["struct_ptr"], idx_tf + 1)
            pair_ranges = tf.ragged.range(pair_starts, pair_ends)
            flat_pair = tf.cast(pair_ranges.flat_values, tf.int32)
            gv_full = train_data["grad_values"]
            if train_data.get("_gv_disk_backed", False):
                batch["grad_values"] = tf.constant(
                    np.asarray(gv_full[flat_pair.numpy()]))
            else:
                batch["grad_values"] = tf.gather(gv_full, flat_pair)
            batch["pair_atom"] = tf.gather(train_data["pair_atom"], flat_pair)
            batch["pair_gidx"] = tf.gather(train_data["pair_gidx"], flat_pair)
            batch["pair_struct"] = tf.cast(pair_ranges.value_rowids(), tf.int32)
            return batch

        # Per-component weights for the training loss. SNES applies the
        # polarisability shear-weights (mode 2) and an optional per-
        # component inverse-magnitude weighting; the Adam loss MUST
        # match or it ranks a different objective. Reuse the SNES
        # helper's `_pol_weights` so both paths build them identically.
        # `_inv_comp_weights` is not currently materialised by the SNES
        # batch builder (cfg.inverse_weight_mode is dormant), but if it
        # were, the same shape conventions would compose here.
        pol_weights_tf = getattr(snes_helper, "_pol_weights", None)
        loss_comp_w = (pol_weights_tf[tf.newaxis]
                       if pol_weights_tf is not None else None)
        sq_comp_w = loss_comp_w  # SNES uses pol_weights for sq reporting too

        # Chunk size for the iterative encode of grad_values. The
        # Willatt scatter materialises an O(N · G_T² · L) dense tensor
        # per encode call (≈115 KB/row for T=6, αmax=10, L=8); at 4096
        # rows that's ~470 MB before the tape stores activations for
        # backprop. Larger chunks crash the 9.5 GB GPU on realistic P.
        enc_chunk_rows = int(getattr(cfg, "encoder_chunk_rows", 4096))

        def _forward_loss(batch):
            # ── Iterative encoder application ─────────────────────────
            # When train_encoder=True, batch descriptors/grad_values
            # still live in raw Q_raw space — apply the encoder here so
            # gradients flow back through it. Affine descriptors via
            # encoder.encode (includes mean centering); gradients via
            # the linear-part identity J·v = encode(v) − encode(0).
            if iterative_encoder:
                B_b = tf.shape(batch["descriptors"])[0]
                A_b = tf.shape(batch["descriptors"])[1]
                d_flat = tf.reshape(batch["descriptors"], [B_b * A_b, q_raw_enc])
                # Descriptors: B·A typically a few thousand rows — fine
                # in one shot, but chunk anyway to cover the worst case
                # (large val_size or full-batch training).
                z_flat = self._encode_chunked(
                    encoder, d_flat, q_raw_enc, enc_chunk_rows)
                descriptors_use = tf.reshape(
                    z_flat, [B_b, A_b, tf.shape(z_flat)[-1]])
                # Gradients: [P, 3, Q_raw] → [P, 3, Z]. P·3 can run to
                # 100k+ rows; this is the OOM hot-path.
                P_b = tf.shape(batch["grad_values"])[0]
                gv_flat = tf.reshape(batch["grad_values"], [P_b * 3, q_raw_enc])
                gv_enc_with_b = self._encode_chunked(
                    encoder, gv_flat, q_raw_enc, enc_chunk_rows)
                b_offset = encoder.encode(
                    tf.zeros([1, q_raw_enc], dtype=tf.float32))   # [1, Z]
                grad_values_use = tf.reshape(
                    gv_enc_with_b - b_offset,
                    [P_b, 3, tf.shape(gv_enc_with_b)[-1]])
            else:
                descriptors_use = batch["descriptors"]
                grad_values_use = batch["grad_values"]

            # Refold W0 / W0_pol inside the tape so gradients flow back
            # through U_pair and W_pre_angular (if either is trainable).
            W0_eff = self._W0_eff(self.W0)
            W0p = getattr(self, "W0_pol", None)
            b0p = getattr(self, "b0_pol", None)
            W1p = getattr(self, "W1_pol", None)
            b1p = getattr(self, "b1_pol", None)
            if W0p is not None:
                W0p = self._W0_eff(W0p)
            if getattr(self, "descriptor_preprocess_contract", "off") != "off":
                W0_eff = self._W0_preprocess_eff(W0_eff)
                if W0p is not None:
                    W0p = self._W0_preprocess_eff(W0p)
            preds = self.predict_batch(
                descriptors_use, grad_values_use,
                batch["pair_atom"], batch["pair_gidx"], batch["pair_struct"],
                batch["positions"], batch["Z_int"], batch["boxes"],
                batch["atom_mask"],
                W0_eff, self.b0, self.W1, self.b1,
                W0p, b0p, W1p, b1p)
            if cfg.scale_targets and cfg.target_mode == 1:
                num_atoms = tf.reduce_sum(batch["atom_mask"], axis=1)
                preds = preds / tf.maximum(num_atoms, 1.0)[:, tf.newaxis]
            diff = preds - batch["targets"]
            per_struct = per_structure_error(
                diff, loss_type, huber_delta,
                component_weights=loss_comp_w)
            sq_per_struct = squared_error_per_structure(
                diff, component_weights=sq_comp_w)
            return tf.reduce_mean(per_struct), tf.reduce_mean(sq_per_struct), preds

        train_start = time.perf_counter()
        for epoch in range(n_epochs):
            t0 = time.perf_counter()
            batch = _sample_batch()
            t_sample = time.perf_counter() - t0

            t1 = time.perf_counter()
            with tf.GradientTape() as tape:
                loss, sq_loss, _preds = _forward_loss(batch)
            grads = tape.gradient(loss, train_vars)
            adam.apply_gradients(zip(grads, train_vars))
            train_rmse = float(tf.sqrt(tf.maximum(sq_loss, 0.0)))
            t_step = time.perf_counter() - t1

            val_loss = float("inf")
            t_val = 0.0
            if epoch % val_interval == 0 or epoch == n_epochs - 1:
                t2 = time.perf_counter()
                if iterative_encoder:
                    # Apply the CURRENT encoder to val_data so the
                    # validation tracks the live encoder state.
                    # Cheap relative to training: one extra encode
                    # over all val structures per val tick.
                    val_data_enc = self._encode_val_data_iterative(
                        val_data, encoder, q_raw_enc)
                    val_loss = float(snes_helper.validate(val_data_enc))
                else:
                    val_loss = float(snes_helper.validate(val_data))
                t_val = time.perf_counter() - t2
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_vars_snapshot = [tf.identity(v) for v in train_vars]

            history["generation"].append(epoch)
            history["train_loss"].append(float(loss))
            history["train_rmse"].append(train_rmse)
            history["val_loss"].append(val_loss)
            history["L1"].append(0.0)
            history["L2"].append(0.0)
            history["best_rmse"].append(train_rmse)
            history["worst_rmse"].append(train_rmse)
            history["sigma_min"].append(float("nan"))
            history["sigma_max"].append(float("nan"))
            history["sigma_mean"].append(float("nan"))
            history["sigma_median"].append(float("nan"))
            history["timing"]["sample_batch"].append(t_sample)
            history["timing"]["evaluate"].append(t_step)
            history["timing"]["rank_update"].append(0.0)
            history["timing"]["validate"].append(t_val)
            history["timing"]["overhead"].append(
                time.perf_counter() - t0 - t_sample - t_step - t_val)

            if (epoch + 1) % max(1, val_interval) == 0 or epoch == n_epochs - 1:
                print(f"[adam] epoch {epoch + 1}/{n_epochs}  "
                      f"loss={float(loss):.5e}  train_rmse={train_rmse:.5e}  "
                      f"val={val_loss:.5e}  best_val={best_val_loss:.5e}  "
                      f"({time.perf_counter() - train_start:.1f}s elapsed)")
            if plot_callback is not None:
                try:
                    plot_callback(history, epoch)
                except Exception as e:
                    print(f"[adam] plot_callback raised: {e!r}")

        # Restore best-val weights into the model so the returned
        # best_val_model has them. final_model carries the last-epoch
        # state — since the model is one Python object we have to pick:
        # callers overwhelmingly want best_val_model, so we restore
        # those weights and return the same instance for both.
        if best_vars_snapshot is not None:
            for var, snap in zip(train_vars, best_vars_snapshot):
                var.assign(snap)

        return history, self, self

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
        # Streaming chunked scoring. Bounds peak memory to one chunk's
        # gradient slice — and is the only sensible mode when grad_values
        # is disk-backed. With cfg.chunk_prefetch the disk-pipe of chunk N+1
        # overlaps the model forward of chunk N.
        from data import prefetched_chunks
        S_test = test_data["num_atoms"].shape[0]
        chunk_sz = (self.cfg.batch_chunk_size
                    if self.cfg.batch_chunk_size is not None else S_test)
        # Iterative-encoder front-end: test_data here was NOT rewritten
        # by the static preprocess (data.materialize_test_data skips
        # when train_encoder=True), so descriptors / grad_values are
        # still Q_raw-shaped. Apply the live encoder per chunk so the
        # W0 (sized at Z) sees the right input. Static mode already
        # rewrote the data dict at materialize time → no-op here.
        _enc = getattr(self.cfg, "_encoder", None)
        _iter_enc = (_enc is not None
                     and bool(getattr(self.cfg, "train_encoder", False)))
        _q_raw_enc = (int(getattr(self.cfg, "_encoder_q_raw", 0))
                      if _iter_enc else 0)
        _enc_chunk = int(getattr(self.cfg, "encoder_chunk_rows", 4096))
        # Pre-fold U_pair^T into W0 (and W0_pol) once per score call
        # so every chunk forward uses the same already-absorbed
        # weights. No-op when descriptor mixing is disabled.
        W0_eff = self._W0_eff(self.W0)
        W0_pol_eff = (self._W0_eff(self.W0_pol)
                      if (self.cfg.target_mode == 2
                          and getattr(self, "W0_pol", None) is not None)
                      else getattr(self, "W0_pol", None))
        # Mirror the training-path second fold: when the preprocess
        # contraction is on, W0 is stored at Q_new and must be expanded
        # back to Q_raw before `predict_batch`'s einsum, which assumes
        # raw-dim descriptors. See validate()/_evaluate_chunk at lines
        # 1862-1869 for the canonical chain.
        if self.descriptor_preprocess_contract != "off":
            W0_eff = self._W0_preprocess_eff(W0_eff)
            if W0_pol_eff is not None:
                W0_pol_eff = self._W0_preprocess_eff(W0_pol_eff)
        ranges = [(s, min(s + chunk_sz, S_test)) for s in range(0, S_test, chunk_sz)]
        pred_parts: list = []
        for _, _, chunk in prefetched_chunks(
                test_data, ranges,
                pin_to_cpu=self.cfg.pin_data_to_cpu,
                enabled=getattr(self.cfg, "chunk_prefetch", True),
                depth=getattr(self.cfg, "prefetch_depth", 1)):
            if _iter_enc:
                d_chunk, gv_chunk = self._encode_chunk_for_inference(
                    chunk["descriptors"], chunk["grad_values"],
                    _enc, _q_raw_enc, _enc_chunk)
            else:
                d_chunk = chunk["descriptors"]
                gv_chunk = chunk["grad_values"]
            pred_parts.append(self.predict_batch(
                d_chunk, gv_chunk,
                chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
                chunk["positions"], chunk["Z_int"], chunk["boxes"],
                chunk["atom_mask"],
                W0_eff, self.b0, self.W1, self.b1,
                W0_pol_eff,
                getattr(self, 'b0_pol', None),
                getattr(self, 'W1_pol', None),
                getattr(self, 'b1_pol', None),
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

        # Target centering inverse: add the frozen training-set mean
        # back to BOTH preds and targets so all downstream metrics and
        # the returned `preds` are in original (un-centered) units. RMSE
        # and R² are invariant under this shift (both terms get the same
        # offset), but cos_sim and the absolute prediction values are
        # not — they MUST be restored to original units to be meaningful.
        if (bool(getattr(self.cfg, "target_centering", False))
                and getattr(self.cfg, "_target_mean", None) is not None):
            mean_tf = tf.constant(
                np.asarray(self.cfg._target_mean,
                           dtype=np.float32).reshape(1, -1))
            preds = preds + mean_tf
            targets = targets + mean_tf

        diff = preds - targets
        mse = tf.reduce_mean(tf.square(diff))
        rmse = tf.sqrt(tf.maximum(mse, 0.0))

        # Overall R² = 1 - SS_res / SS_tot
        ss_res = tf.reduce_sum(tf.square(diff))
        ss_tot = tf.reduce_sum(tf.square(targets - tf.reduce_mean(targets, axis=0)))
        r2 = 1.0 - ss_res / ss_tot

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

        # Total (un-scaled) metrics when target scaling is active. Both
        # totals must be in ORIGINAL (un-centered) units: `targets` here
        # is per-atom-original (mean added back above), so
        # `targets * num_atoms` is the total-original. The raw network
        # output `raw_preds` is in centered per-atom * num_atoms space —
        # add back the corresponding offset (mean * num_atoms) per
        # structure so total_diff compares two original-unit quantities.
        if self.cfg.scale_targets and self.cfg.target_mode == 1 and "num_atoms" in test_data:
            total_targets = targets * num_atoms_col
            if (bool(getattr(self.cfg, "target_centering", False))
                    and getattr(self.cfg, "_target_mean", None) is not None):
                total_preds = raw_preds + mean_tf * num_atoms_col
            else:
                total_preds = raw_preds
            total_diff = total_preds - total_targets
            total_rmse = tf.sqrt(tf.reduce_mean(tf.square(total_diff)))
            total_ss_res = tf.reduce_sum(tf.square(total_diff))
            total_ss_tot = tf.reduce_sum(tf.square(
                total_targets - tf.reduce_mean(total_targets, axis=0)))
            total_r2 = 1.0 - total_ss_res / total_ss_tot
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
        """Score the model directly on an XYZ file or a directory of them.

        Onboards the data through the standard pipeline (`collect` →
        descriptor build → `assemble_data_dict` → `pad_and_stack`), then
        dispatches to :py:meth:`score`. Useful for one-line evaluation
        on a held-out test set without manually wiring the data dict.

        Species handling:
            The model's training-time `cfg.num_types`, `cfg.types`, and
            `cfg.dim_q` are preserved across the call so per-type ANN
            routing and the descriptor builder produce the same layout
            the model was trained on. Atom types in the test set are
            re-indexed into the training-time species ordering — atoms
            of species the model has never seen raise `KeyError`.

        Args:
            path: path to a single ``.xyz`` file or to a directory.
                When a directory is given, every ``.xyz`` file in it is
                concatenated (sorted by filename) into a single test set.
            allowed_species: optional override of `cfg.allowed_species`.
                ``None`` keeps the training-time filter. Pass an explicit
                list (e.g. ``[1, 8]`` for water) to relax filtering when
                the test set has a strict subset of the training species.
            max_structures: optional cap on the number of structures
                returned by `collect`. ``None`` = no cap.
            filter_mode: optional override of `cfg.filter_mode`
                (``"subset"`` or ``"exact"``). ``None`` keeps the current
                cfg value.
            pin_to_cpu: forwarded to `pad_and_stack`. Default ``True``
                keeps the test data on host RAM (saves GPU memory for
                scoring; the score forward streams chunks to the device).
            plot: when ``True``, generate the standard scoring figures
                (parity plot per component, error-vs-magnitude scatter)
                after scoring. Uses :py:func:`plotting.plot_correlation`
                and :py:func:`plotting.plot_error_vs_magnitude`, which
                honour ``cfg.plot_units`` for unit overrides.
            save_plots: directory to write figures into. ``None`` skips
                saving (figures are only shown if ``show_plots=True``).
                Auto-creates the directory if it doesn't exist.
            show_plots: ``True`` (default) shows figures via the active
                matplotlib backend. Set to ``False`` for headless / batch
                contexts where ``save_plots`` is enough.
            suffix: optional string appended to the figure filenames
                (e.g. ``"water_test"`` produces ``correlation_water_test_…``).
            presentation: when ``True``, switches the figure style to a
                presentation / poster-friendly preset (thick lines,
                large fonts) — currently a no-op style switch (the flag
                is plumbed through but the underlying parity / error
                plots haven't adopted presentation styling yet); flagged
                via ``cfg`` so future plot functions can pick it up.
            shared_axis_scale: when ``True``, the per-component parity
                panels (x, y, z for dipole or 6-component pol) all use
                the SAME x/y range computed from the joint min/max of
                (targets, predictions) across every component. Useful
                when one component dominates the others and per-panel
                autoscaling hides the asymmetry. Default ``False`` keeps
                each panel autoscaled to its own component's range.

        Returns:
            Same as :py:meth:`score`: ``(metrics, preds)`` where
            ``metrics`` is the standard dict (RMSE, R², per-component R²,
            RRMSE, cos similarity) and ``preds`` is an ``[S, T_dim]``
            tensor of model predictions on the loaded set.

        Example:
            >>> model = load_model("models/.../best_val.h5")
            >>> metrics, preds = model.score_from_file(
            ...     "datasets/test_waterbulk.xyz",
            ...     allowed_species=[1, 8],
            ... )
            >>> print(f"water-test RMSE = {float(metrics['rmse']):.4f}")
        """
        import os
        import copy
        from ase.io import read as _ase_read
        from data import collect, assemble_data_dict, pad_and_stack
        from DescriptorBuilder import make_descriptor_builder

        cfg = self.cfg

        # Snapshot training-time species + dim so the descriptor builder
        # and per-type ANN routing keep the layout the model expects.
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

        # Pipeline: collect → build descriptors with TRAINING-TIME species →
        # assemble → pad_and_stack. `collect` overrides num_types from the
        # data; we restore it afterwards so the builder produces dim_q-=165
        # (or whatever the model was trained at) instead of the data-implied dim.
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
        # The builder always emits at the raw SOAP dim (Q_raw); for
        # encoder-front-end runs we then rewrite to Z below.
        print(f"  building descriptors for {len(dataset)} structures ...")
        builder = make_descriptor_builder(cfg_for_load)
        # In encoder mode, ensure the builder sees the raw Q layout (T,
        # alpha_max, l_max, compress_mode) — these have already been
        # cfg-restored from the checkpoint. The builder doesn't know
        # about the encoder; we apply it post-build.
        descs, grads, gidx = builder.build_descriptors(dataset)

        # Encoder front-end: if a pretrained encoder is configured on
        # cfg, apply it to the raw builder output BEFORE pad_and_stack.
        # `setup_encoder_frontend` is idempotent; static or iterative
        # mode is irrelevant here because at score time the encoder is
        # frozen (no training co-adaptation).
        if getattr(cfg_for_load, "encoder_path", None) is not None:
            from encoder_frontend import (
                setup_encoder_frontend, apply_encoder_to_lists)
            setup_encoder_frontend(cfg_for_load)
            descs, grads = apply_encoder_to_lists(
                descs, grads,
                cfg_for_load._encoder_J, cfg_for_load._encoder,
                progress=False)
            # cfg_for_load.dim_q is already TRAIN_DIM_Q (= Z because we
            # restored from the model's trained cfg) — no override
            # needed.

        # Assemble + pad. Descriptors are now [N_i, TRAIN_DIM_Q] per
        # structure (Z when encoder is on, Q_raw otherwise). Pass through
        # q_scaler / target_mean so the test inputs land in the same
        # standardised space the model was trained on — without this,
        # any descriptor_scaling="q_scaler" model silently scores in the
        # wrong space.
        data_dict = assemble_data_dict(
            dataset, ti, descs, grads, gidx, cfg_for_load)
        test_data = pad_and_stack(
            data_dict, num_types=TRAIN_NUM_TYPES, pin_to_cpu=pin_to_cpu,
            q_scaler=getattr(cfg_for_load, "_q_scaler", None),
            target_mean=getattr(cfg_for_load, "_target_mean", None))

        print(f"  scoring on {test_data['descriptors'].shape[0]} structures ...")
        try:
            metrics, preds = self.score(test_data)
            if plot:
                # Lazy import — avoids matplotlib at module import time
                # for non-interactive uses of TNEP.
                from plotting import plot_correlation, plot_error_vs_magnitude
                if save_plots is not None:
                    os.makedirs(save_plots, exist_ok=True)
                targets_np = test_data["targets"].numpy()
                preds_np   = preds.numpy()
                # plot_correlation expects an RRMSE entry alongside RMSE.
                # `score()` doesn't compute it (callers normally inject
                # one because the denominator depends on the per-atom vs
                # total-target convention). Mirror MasterTNEP's
                # convention: divide RMSE by the target std on the same
                # space the score was reported in (per-atom here).
                diff = targets_np - preds_np
                std_overall = max(float(targets_np.std()), 1e-12)
                std_comp = np.maximum(targets_np.std(axis=0), 1e-12)
                rmse_comp = np.sqrt(np.mean(diff ** 2, axis=0))
                metrics_plot = dict(metrics)
                metrics_plot["rrmse"] = float(metrics["rmse"]) / std_overall
                metrics_plot["rrmse_components"] = rmse_comp / std_comp
                # Use the model's cfg for unit handling so cfg.plot_units
                # ("debye" / "e*bohr" / "e*angstrom") is honoured.
                # The `presentation` flag is stashed on the cfg as
                # `_presentation_mode` so plot functions can pick it up
                # via getattr(cfg, "_presentation_mode", False) when
                # they grow presentation-styled paths. For now this is
                # a no-op switch; the kwarg exists for API stability.
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
                    # Restore previous presentation flag (or remove it
                    # if we set it for the first time).
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
        """Print and return a labelled comparison-ready scoring summary.

        Reports metrics in BOTH per-atom space (matches GPUMD's `loss.out`
        `rmse_virial` convention) AND total-dipole space (matches the
        convention used in TNEP / NEP papers like Xu et al for headline
        RMSE / R² values). Use this to compare against published numbers
        without space-convention confusion.

        Also derives an RRMSE = √(1 − R²) for each space, which is the
        centered definition (variance-normalised). Note: this differs from
        the un-centered SNES training RRMSE (`best_rrmse` in history),
        which uses `Σy²` rather than `Σ(y − ȳ)²` in the denominator —
        the two are nearly equal when target means are small but not
        identical.
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

        Returns:
            predictions : [B, T_dim]  where T_dim = 1 (PES), 3 (dipole), 6 (pol)
        """
        box_inv = tf.linalg.inv(boxes)  # [B, 3, 3]

        # predict_batch is a pure forward primitive: caller is
        # responsible for whether W0 / W0_pol are raw or already
        # U-absorbed (via _W0_eff). This keeps the function single-
        # purpose and avoids double-mixing when callers (validate,
        # predict_batch_candidates) have already folded U_pair^T in.
        W0_use = W0
        W0_pol_use = W0_pol

        b0_t = tf.gather(b0, Z)   # [B, A, H]
        W1_t = tf.gather(W1, Z)   # [B, A, H_final]

        # Per-type loop for W0: avoids materialising [B, A, Q, H] (dominant memory cost).
        # b0/W1 only have [B, A, H] so their gathers are fine.
        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(self.num_types)
        ]
        # Pre-activation z1 preserved for the swish-side backward chain.
        z1 = tf.add_n([
            tf.einsum('baq,qh->bah', descriptors, W0_use[t]) * type_masks[t]
            for t in range(self.num_types)
        ]) + b0_t
        h1 = self.activation(z1)
        h1 = h1 * atom_mask[:, :, tf.newaxis]

        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h1 * W1_t, axis=2) + b1  # [B, A]
            E = E * atom_mask
            # H-center skip (see comment in predict): exclude H atoms.
            if bool(getattr(self.cfg, "skip_h_centers", False)):
                E = E * tf.cast(Z != 1, tf.float32)
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

        if self.cfg.target_mode == 1 and W_atom is not None:
            # Precomputed-kernel path: avoids [C, P, Q] inside vectorized_map.
            # Mathematically: dipole[b,s] = -Σ_{a,q} de_dq[b,a,q] * W_atom[b,a,s,q]
            #   = -Σ_p rij²[p] * Σ_q de_dq[struct[p],atom[p],q] * grad_values[p,s,q]
            # (identical to the COO forces path, proven by substituting W_atom definition)
            return -tf.einsum('baq,basq->bs', de_dq, W_atom)  # [B, 3]

        # Standard COO path (used when W_atom is not precomputed: score(), predict())
        forces_per_pair = self._calc_forces_coo(de_dq, grad_values, pair_struct, pair_atom)

        if self.cfg.target_mode == 1:
            return self._dipole_coo(forces_per_pair, pair_struct, pair_atom, pair_gidx,
                                    positions, boxes, box_inv, B)

        elif self.cfg.target_mode == 2:
            # Polarizability's scalar ANN gets the U-absorbed W0_pol_use
            # so it operates in the same learned-feature space as the
            # main ANN. Raw descriptors are passed; the algebra is
            # equivalent to feeding desc_mixed into raw W0_pol.
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
                                  W_pre_angular: tf.Tensor | None = None,
                                  W_pre_angular_l: tf.Tensor | None = None) -> tf.Tensor:
        """Forward pass for C candidates × B structures using explicit batched GEMMs.

        Replaces vectorized_map for target_mode 0 (PES) and 1 (dipole).
        Both the input→hidden and hidden→descriptor matmuls are executed as
        single large GEMMs over all C candidates simultaneously:

            Forward:  [B*A, Q] @ [Q, C*H]    → [B*A, C*H] → [C, B, A, H]
            Backward: [C, B*A, H] @ [C, H, Q] → [C, B*A, Q]  (batched GEMM)

        Descriptors are assumed pre-scaled by the caller.

        Args:
            descriptors : [B, A, Q]
            W_atom      : [B, A, 3, Q]  precomputed dipole kernel (mode 1 only)
            Z           : [B, A]        type indices
            atom_mask   : [B, A]        1.0 real, 0.0 pad
            W0          : [C, T, Q, H]
            b0          : [C, T, H]
            W1          : [C, T, H]
            b1          : [C]

        Returns:
            predictions : [C, B, T_dim]  T_dim = 1 (PES) or 3 (dipole)
        """
        # Q is the dim of the descriptor-axis seen by the matmul. When
        # preprocessing is on, W0 storage lives at Q_new but the matmul
        # operates at Q_raw after the preprocess fold (see
        # _W0_preprocess_eff). dim_q_forward selects the right value.
        Q = self.dim_q_forward
        H = self.num_neurons
        H_final = self._H_final
        T = self.num_types

        B = tf.shape(descriptors)[0]
        A = tf.shape(descriptors)[1]
        C = tf.shape(W0)[0]

        # Type masks [B, A, 1] — independent of C, reused for both matmul directions
        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(T)
        ]

        # Per-candidate descriptor mixing: absorb U_pair^T into W0 so
        # the rest of the forward/backward uses raw descriptors and
        # the de_dq we produce is already in raw-desc space (ready to
        # combine with raw grad_values in the dipole sum). See _W0_eff
        # for the algebraic identity.
        if self.descriptor_mixing and U_pair is not None:
            W0 = self._W0_eff(W0, U_pair)

        # Preprocessing contraction: fold the per-type per-channel
        # coefficients into W0 along the Q_new axis, producing a W0_eff
        # at Q_raw. The matmul code below sees Q = Q_raw uniformly and
        # the dipole backward yields de_dq at Q_raw, ready to contract
        # with the precomputed raw W_atom. Mutually exclusive with
        # mixing/gating (so the two folds never compose in this build).
        if self.descriptor_preprocess_contract != "off":
            W0 = self._W0_preprocess_eff(
                W0,
                W_pre_override=W_pre_angular,
                W_pre_l_override=W_pre_angular_l)

        # ── Forward: input→hidden ─────────────────────────────────────────────
        # Per type: [B*A, Q] @ [Q, C*H] → [B*A, C*H] → [C, B, A, H]
        # One GEMM per type instead of C separate GEMMs inside pfor.
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
        # z1 = pre_h + b0  is the pre-activation; preserve it for the
        # swish-side backward chain rule below.
        z1 = pre_h + b0_t_all
        h1 = self.activation(z1)
        h1 = h1 * atom_mask[tf.newaxis, :, :, tf.newaxis]

        # ── PES ───────────────────────────────────────────────────────────────
        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h1 * W1_t_all, axis=3) + b1[:, tf.newaxis, tf.newaxis]
            E = E * atom_mask[tf.newaxis]
            return -tf.reduce_sum(E, axis=2, keepdims=True)  # [C, B, 1]

        # ── Dipole: backward matmul ───────────────────────────────────────────
        # U = h1 · W1 + b1 ⇒ ∂U/∂h1 = W1, ∂U/∂a1 = activation'(h1, z1)·W1.
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

        # W_atom [B, A, 3, Q]: dipole[c,b,s] = -Σ_{a,q} de_dq[c,b,a,q]*W_atom[b,a,s,q]
        return -tf.einsum('cbaq,basq->cbs', de_dq, W_atom)  # [C, B, 3]

    def _scalar_rij_pow(self, rij2: tf.Tensor) -> tf.Tensor:
        """|r_ij|^N as a scalar per-pair weight (N ≥ 1 only).

        Derived from the rij² primitive without a fresh sqrt for even N:
            N = 1      → sqrt(rij²)
            N = 2      → rij² unchanged (zero ops; default path)
            N even ≥ 4 → tf.pow(rij², N/2)
            N odd  ≥ 3 → tf.pow(rij², (N-1)/2) · sqrt(rij²)

        N = 0 is handled by `_dipole_pair_weight_*` (different algebraic
        branch — restricts the dipole sum to self pairs only), not here.
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
        """Per-pair weight in the dipole sum, COO-pair interface.

            N == 0 : 1 where pair_atom == pair_gidx AND rij² < 1e-20
                     (true zero-image self pair), 0 elsewhere
                     → dipole collapses to -Σ_i de_dq[i] · grad_values[i, i].
                     The rij² guard rejects periodic IMAGES of atom i that
                     appear as its own neighbour with the same atom index
                     but a nonzero displacement vector — including those
                     would double-count the self contribution.
            N >= 1 : |r_ij|^N — self pairs naturally contribute 0 via
                     |r_ii|^N = 0, and periodic-image self pairs are
                     weighted correctly by their nonzero |r|^N.
        """
        N = int(getattr(self.cfg, "dipole_rij_power", 2))
        if N == 0:
            is_self = tf.logical_and(tf.equal(pair_atom, pair_gidx),
                                     rij2 < 1e-20)
            return tf.cast(is_self, rij2.dtype)
        return self._scalar_rij_pow(rij2)

    def _dipole_pair_weight_padded(self, rij2: tf.Tensor,
                                    grad_index: tf.Tensor) -> tf.Tensor:
        """Per-pair weight in the dipole sum, padded [A, M] interface.

        Same dispatch as the COO variant. The "centre" for row i is just
        i itself (the padded layout is [centre A, neighbour-slot M]), so
        self pairs are wherever `grad_index[i, m] == i` AND rij² < 1e-20.
        The rij² guard rejects periodic-image self entries (same atom
        index, nonzero displacement) that would otherwise double-count.

        Contract: the returned weight tensor is only valid AFTER the
        caller multiplies by `neighbor_mask` to zero out padding rows
        (padding rows can have grad_index == 0 == some real centre and
        rij2 == 0, so they look like self pairs). The single-structure
        predict() path applies this mask at ~line 521.
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
        s = tf.einsum('ij,nj->ni', box_inv, positions)       # [A, 3]
        s_j = tf.gather(s, grad_index)                        # [A, M, 3]
        s_i = s[:, tf.newaxis, :]                             # [A, 1, 3]
        ds = s_j - s_i
        ds = ds - tf.round(ds)
        dr = tf.einsum('ij,nmj->nmi', box, ds)                # [A, M, 3]
        rij = tf.linalg.norm(dr, axis=-1)                     # [A, M]
        return dr, rij

    def _precompute_dipole_kernel(self, grad_values: tf.Tensor,
                                  pair_struct: tf.Tensor, pair_atom: tf.Tensor,
                                  pair_gidx: tf.Tensor, positions: tf.Tensor,
                                  boxes: tf.Tensor, B: tf.Tensor,
                                  A: tf.Tensor) -> tf.Tensor:
        """Aggregate rij²-weighted gradients per (structure, atom) — independent of candidates.

        W_atom[b,a,s,q] = Σ_{p: struct[p]=b, atom[p]=a} rij²[p] × grad_values[p,s,q]

        Dipole is then -einsum('baq,basq->bs', de_dq, W_atom) with no P dimension
        inside vectorized_map, eliminating the [C, P, Q] intermediate.

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
        # dipole_rij_power short-circuits:
        #   N == 0 : weight is just the self-pair indicator 1[i==j]. The
        #            data pipeline already filtered the COO list to
        #            self-pairs only (data.py self_pairs_only branch), so
        #            every pair satisfies i==j and weight = 1 everywhere.
        #            Skips the expensive _neighbor_displacements_coo +
        #            tf.linalg.inv(boxes) entirely.
        #   N >= 1 : standard per-pair |r|^N weight (self pairs contribute
        #            zero via |r_ii|=0); needs displacements.
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
        s_c = tf.einsum('kij,kj->ki', binv_k, pos_c)        # [P, 3] fractional
        s_n = tf.einsum('kij,kj->ki', binv_k, pos_n)
        ds  = s_n - s_c
        ds  = ds - tf.round(ds)                              # MIC wrap
        dr  = tf.einsum('kij,kj->ki', box_k, ds)            # [P, 3] Cartesian
        rij2 = tf.reduce_sum(tf.square(dr), axis=-1)         # [P]
        return dr, rij2

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
        # Dipole contraction.
        #   N >= 1 : μ = -Σ_pair |r_ij|^N · F_ij  (scalar weight × force vector)
        #   N == 0 : μ = -Σ_i F_ii (self-only — see _dipole_pair_weight_coo)
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
        # H-center skip: zero F_pol for H atoms (descriptor is zero,
        # so F_pol would be bias-driven). The tensor (virial) part
        # below is automatically H-free because no pairs have center=H.
        if bool(getattr(self.cfg, "skip_h_centers", False)):
            F_pol = F_pol * tf.cast(Z != 1, tf.float32)
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
