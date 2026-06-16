from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import Callable

from DescriptorBuilder import make_descriptor_builder
from SNES import SNES
from TNEPconfig import TNEPconfig


# ---------------------------------------------------------------------------
# Activation registry: name → (canonical key, Glorot gain).
#
# Only activations whose backward derivative is plumbed through the dipole /
# polarisability chain rule below appear here. Adding a new activation
# requires (a) a Keras-recognised forward, (b) a closed-form derivative
# branch in `_activation_grad`, and (c) a gain value. Until that's done,
# unsupported activations fail fast at TNEP construction rather than
# silently breaking the dipole backward.
#
# Glorot gain convention: c_W0 = gain · sqrt(6 / (Q + H)) for uniform init.
# tanh: gain=1 (current default, established in TNEP). swish/silu: gain=1
# (similar slope near origin; PyTorch's Xavier doesn't define silu so we
# pick a conservative value — SNES corrects through σ adaptation anyway).
_ACTIVATION_REGISTRY: dict[str, tuple[str, float]] = {
    "tanh":  ("tanh",  1.0),
    "swish": ("swish", 1.0),
    "silu":  ("swish", 1.0),     # silu is swish, just different name
}


def _normalize_activation_name(name: str) -> str:
    """Map a cfg.activation string to the canonical key used by _activation_grad.

    Raises ValueError when the activation is not in the supported registry —
    not 'unsupported by TF' (most Keras activations work for the forward),
    but specifically 'backward derivative not plumbed here'. Catches the
    silent-incorrect-dipole failure mode where someone sets `cfg.activation
    = 'gelu'` and gets a forward that works but a backward computed as if
    it were tanh.
    """
    key = str(name).lower().strip()
    if key not in _ACTIVATION_REGISTRY:
        supported = sorted(set(v[0] for v in _ACTIVATION_REGISTRY.values()))
        raise ValueError(
            f"cfg.activation={name!r} not supported in this build. "
            f"The dipole / polarisability backward uses a hand-coded "
            f"derivative chain (see TNEP._activation_grad); only activations "
            f"with an entry in _ACTIVATION_REGISTRY are correct end-to-end. "
            f"Supported: {supported}.")
    return _ACTIVATION_REGISTRY[key][0]


def _glorot_gain_for(canonical_name: str) -> float:
    """Per-activation multiplier on the standard sqrt(6/(Q+H)) Glorot bound.

    Applied to c_W0 and c_W1 in SNES._build_mu_init.
    """
    for entry in _ACTIVATION_REGISTRY.values():
        if entry[0] == canonical_name:
            return float(entry[1])
    raise ValueError(f"no Glorot gain for {canonical_name!r}")


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
        # per-type coefficient Variable W_pre_*. See plan doc at
        # docs/superpowers/plans/2026-06-02-descriptor-preprocess-contraction.md.
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
            # into a flat (n'', l) axis, so the post-contraction layout
            # used by descriptor_mixing / descriptor_mixing_output_layer
            # / cross_layer / per_l_ann_heads is undefined. Disallow
            # composition in this first pass — set the offending flag(s)
            # to off, or run nep4_radial standalone.
            _conflicts = []
            if bool(getattr(cfg, "descriptor_mixing", False)):
                _conflicts.append("descriptor_mixing")
            if bool(getattr(cfg, "descriptor_mixing_output_layer", False)):
                _conflicts.append("descriptor_mixing_output_layer")
            if bool(getattr(cfg, "descriptor_mixing_cross_layer", False)):
                _conflicts.append("descriptor_mixing_cross_layer")
            if bool(getattr(cfg, "descriptor_per_l_ann_heads", False)):
                _conflicts.append("descriptor_per_l_ann_heads")
            if bool(getattr(cfg, "descriptor_gating_enabled", False)):
                _conflicts.append("descriptor_gating_enabled")
            if _conflicts:
                raise NotImplementedError(
                    f"descriptor_preprocess_contract='nep4_radial' is "
                    f"mutually exclusive in this build with: "
                    f"{', '.join(_conflicts)}. The bilinear fold collapses "
                    f"the (n, n', species_pair) descriptor structure into "
                    f"a flat (n'', l) axis, so the post-contraction layout "
                    f"those flags assume is undefined. Disable the conflicting "
                    f"flag(s) or run nep4_radial standalone.")
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
            if bool(getattr(cfg, "descriptor_gating_enabled", False)):
                raise NotImplementedError(
                    "descriptor_preprocess_contract is mutually exclusive "
                    "with descriptor_gating_enabled in this build. Set one to off.")
            if bool(getattr(cfg, "descriptor_per_l_ann_heads", False)) \
                    and self.descriptor_preprocess_contract in ("angular", "both"):
                raise NotImplementedError(
                    f"descriptor_preprocess_contract={self.descriptor_preprocess_contract!r} "
                    f"collapses the l axis and is incompatible with "
                    f"descriptor_per_l_ann_heads. Set one to off.")
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
        # Optional second hidden layer width. When None (legacy default), the
        # ANN is single-hidden. When set, an extra [num_neurons, H2] layer is
        # inserted before W1 (whose first-axis size becomes H2 instead of H).
        self.num_neurons_layer_2 = getattr(cfg, "num_neurons_layer_2", None)
        self._H2 = (int(self.num_neurons_layer_2)
                    if self.num_neurons_layer_2 is not None else None)
        # H_final is the actual input dim of W1.
        self._H_final = self._H2 if self._H2 is not None else cfg.num_neurons
        # Validate / normalise activation BEFORE resolving the Keras callable.
        # Unsupported activations fail here (cleaner than a silent
        # wrong-backward bug at first dipole prediction).
        self._activation_name = _normalize_activation_name(cfg.activation)
        self._glorot_gain = _glorot_gain_for(self._activation_name)
        self.activation = tf.keras.activations.get(self._activation_name)
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

        # Optional second hidden layer (per-type). Only created when
        # cfg.num_neurons_layer_2 is set. Identity init is not required —
        # the model only needs to be a valid neural net at gen 0; SNES does
        # the actual exploration. Glorot keeps initial activations balanced.
        if self._H2 is not None:
            self.W0_2 = self.add_weight(
                name="W0_2",
                shape=(cfg.num_types, cfg.num_neurons, self._H2),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b0_2 = self.add_weight(
                name="b0_2",
                shape=(cfg.num_types, self._H2),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.W0_2 = None
            self.b0_2 = None

        # W1 : [num_types, H_final] — hidden-to-scalar weights per type
        # H_final == num_neurons in the legacy single-hidden case; H2 when
        # a second hidden layer is configured.
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

        # Per-(t, l) ANN heads (only allocated when descriptor_per_l_ann_heads).
        # Each head has its own W0_l[T, Q_l, H], b0_l[T, H], W1_l[T, H], b1_l[1].
        # Forward output is the sum over heads. See the plan at
        # docs/superpowers/plans/2026-06-02-per-l-ann-heads.md for rationale.
        self.per_l_heads = bool(getattr(cfg, "descriptor_per_l_ann_heads", False))
        if self.per_l_heads:
            from DescriptorBuilderGPU import descriptor_block_layout
            _layout = descriptor_block_layout(cfg)
            L = int(cfg.l_max) + 1
            self._per_l_L = L
            # Per-l q-index lookup tensors (used by the forward path).
            self._per_l_q_indices = []
            self._per_l_Q_l = []
            for l in range(L):
                qs = []
                for p in _layout["pair_keys"]:
                    if l in _layout["pair_ln_index"][p]:
                        qs.extend(_layout["pair_ln_index"][p][l])
                qs_sorted = sorted(qs)
                self._per_l_q_indices.append(
                    tf.constant(qs_sorted, dtype=tf.int32))
                self._per_l_Q_l.append(len(qs_sorted))
            # Allocate per-l Variables for the primary ANN.
            T_ = int(cfg.num_types)
            H_ = int(cfg.num_neurons)
            self.W0_per_l = [
                self.add_weight(name=f"W0_l{l}",
                                shape=(T_, self._per_l_Q_l[l], H_),
                                initializer="zeros", trainable=False)
                for l in range(L)]
            self.b0_per_l = [
                self.add_weight(name=f"b0_l{l}", shape=(T_, H_),
                                initializer="zeros", trainable=False)
                for l in range(L)]
            self.W1_per_l = [
                self.add_weight(name=f"W1_l{l}", shape=(T_, H_),
                                initializer="zeros", trainable=False)
                for l in range(L)]
            self.b1_per_l = [
                self.add_weight(name=f"b1_l{l}", shape=(),
                                initializer="zeros", trainable=False)
                for l in range(L)]
            if int(cfg.target_mode) == 2:
                self.W0_pol_per_l = [
                    self.add_weight(name=f"W0_pol_l{l}",
                                    shape=(T_, self._per_l_Q_l[l], H_),
                                    initializer="zeros", trainable=False)
                    for l in range(L)]
                self.b0_pol_per_l = [
                    self.add_weight(name=f"b0_pol_l{l}", shape=(T_, H_),
                                    initializer="zeros", trainable=False)
                    for l in range(L)]
                self.W1_pol_per_l = [
                    self.add_weight(name=f"W1_pol_l{l}", shape=(T_, H_),
                                    initializer="zeros", trainable=False)
                    for l in range(L)]
                self.b1_pol_per_l = [
                    self.add_weight(name=f"b1_pol_l{l}", shape=(),
                                    initializer="zeros", trainable=False)
                    for l in range(L)]
            else:
                self.W0_pol_per_l = None
                self.b0_pol_per_l = None
                self.W1_pol_per_l = None
                self.b1_pol_per_l = None
        else:
            self.W0_per_l = None
            self.b0_per_l = None
            self.W1_per_l = None
            self.b1_per_l = None
            self.W0_pol_per_l = None
            self.b0_pol_per_l = None
            self.W1_pol_per_l = None
            self.b1_pol_per_l = None

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
            if _N == 0:
                print(
                    "[EXPERIMENTAL] cfg.dipole_rij_power=0: dipole reduced to "
                    "the per-atom self-pair sum, μ = -Σ_i de_dq[i] · grad_values[i,i]. "
                    "Translation invariance forces the neighbour sum to cancel "
                    "against the self entry, so isolating self gives a non-zero, "
                    "rotation-covariant prediction. Not directly comparable to "
                    "N ≥ 1 (different functional form, not just a different weight)."
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
            # Optional second hidden layer mirror for pol ANN.
            if self._H2 is not None:
                self.W0_2_pol = self.add_weight(
                    name="W0_2_pol",
                    shape=(cfg.num_types, cfg.num_neurons, self._H2),
                    initializer="glorot_uniform",
                    trainable=True,
                )
                self.b0_2_pol = self.add_weight(
                    name="b0_2_pol",
                    shape=(cfg.num_types, self._H2),
                    initializer="zeros",
                    trainable=True,
                )
            else:
                self.W0_2_pol = None
                self.b0_2_pol = None
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
        # Stacked mixing (N independent layers in series) + optional nonlinearity.
        # N=1 + linear (the legacy defaults) is bit-identical to the original
        # single-shared-V_pair path; in particular self.U_pair remains a single
        # tf.Variable referenced by save/load and by _U_full.
        self.descriptor_mixing_n_layers = int(getattr(
            cfg, "descriptor_mixing_n_layers", 1))
        if self.descriptor_mixing_n_layers < 1:
            raise ValueError(
                f"descriptor_mixing_n_layers must be >= 1, got "
                f"{self.descriptor_mixing_n_layers}")
        self.descriptor_mixing_nonlinear = bool(getattr(
            cfg, "descriptor_mixing_nonlinear", False))
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
                self.U_pair_list = []
                for k in range(self.descriptor_mixing_n_layers):
                    name = "U_pair" if k == 0 else f"U_pair_{k}"
                    var = self.add_weight(
                        name=name,
                        shape=shape,
                        initializer="zeros",
                        trainable=True,
                    )
                    self.U_pair_list.append(var)
                self.U_pair = self.U_pair_list[0]
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
                self.U_pair_list = []
                for k in range(self.descriptor_mixing_n_layers):
                    name = "U_pair" if k == 0 else f"U_pair_{k}"
                    var = self.add_weight(
                        name=name,
                        shape=shape,
                        initializer="zeros",
                        trainable=True,
                    )
                    self.U_pair_list.append(var)
                self.U_pair = self.U_pair_list[0]
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
                self.U_pair_list = []
                for k in range(self.descriptor_mixing_n_layers):
                    name = "U_pair" if k == 0 else f"U_pair_{k}"
                    var = self.add_weight(
                        name=name,
                        shape=shape,
                        initializer="zeros",
                        trainable=True,
                    )
                    self.U_pair_list.append(var)
                self.U_pair = self.U_pair_list[0]

            # Optional per-layer bias for nonlinear mixing.
            # b_mix_k shape: [num_types, dim_q] if per_type else [dim_q].
            # Zero-init so model starts close to the linear-mixing case.
            if self.descriptor_mixing_nonlinear:
                if self.descriptor_mixing_per_type:
                    b_mix_shape = (cfg.num_types, self._mix_Q)
                else:
                    b_mix_shape = (self._mix_Q,)
                self.b_mix_list = [
                    self.add_weight(
                        name=f"b_mix_{k}",
                        shape=b_mix_shape,
                        initializer="zeros",
                        trainable=True,
                    )
                    for k in range(self.descriptor_mixing_n_layers)
                ]
            else:
                self.b_mix_list = []
        else:
            self.U_pair = None
            self.U_pair_list = []
            self.b_mix_list = []
            self._mix_P = []
            self._mix_block_sizes = []

        # Optional cross-channel mixing layer: a single [Q × Q] V tensor
        # stored as the residual V_cross = U_cross − I. SNES populates this
        # each candidate via _set_model_params. Zero-initialised so
        # U_cross = I at gen 0 (off-path bit-equivalence).
        self.descriptor_mixing_cross_layer = bool(getattr(
            cfg, "descriptor_mixing_cross_layer", False))
        if self.descriptor_mixing and self.descriptor_mixing_cross_layer:
            Q = int(cfg.dim_q)
            self.V_cross = tf.Variable(
                tf.zeros([Q, Q], dtype=tf.float32),
                trainable=False, name="V_cross")
        else:
            self.V_cross = None

        # Optional output-side (hidden-layer) orthogonal mixing R per type.
        # Stored as residual V_R = R − I; SNES populates this each
        # candidate. Init at zero → R = I at gen 0 (off-path bit-identity).
        # Reconstructed to an orthogonal R via the same Cayley/expm path as
        # U_pair (see self._R_full). Shape: [T, H, H].
        self.descriptor_mixing_output_layer = bool(getattr(
            cfg, "descriptor_mixing_output_layer", False))
        if self.descriptor_mixing_output_layer:
            H_out = int(self.num_neurons)
            T_ = int(self.num_types)
            self.R_pair = tf.Variable(
                tf.zeros([T_, H_out, H_out], dtype=tf.float32),
                trainable=False, name="R_pair")
            self._R_H = H_out
        else:
            self.R_pair = None
            self._R_H = 0

        # Optional per-(species-pair, l, central-type) gating: g[t, p·L+l]
        # is folded into W0 as a per-channel multiplier (channel attention).
        # Allocated as [T, num_pairs·L] flat for fast tf.gather along the
        # (pair, l) axis. Init = cfg.descriptor_gating_init (default 1.0).
        self.descriptor_gating_enabled = bool(getattr(
            cfg, "descriptor_gating_enabled", False))
        if self.descriptor_gating_enabled:
            from DescriptorBuilderGPU import descriptor_block_layout
            _layout = descriptor_block_layout(cfg)
            T_ = int(cfg.num_types)
            num_pairs = len(_layout["pair_keys"])
            L = int(cfg.l_max) + 1
            init = float(getattr(cfg, "descriptor_gating_init", 1.0))
            self.gates_pair_l = tf.Variable(
                tf.fill([T_, num_pairs * L], init),
                trainable=False, name="gates_pair_l")
            # q→(pair·L+l) flat index for fast gather in the forward path.
            pair_idx_of = {p: i for i, p in enumerate(_layout["pair_keys"])}
            q_to_pl = np.full(int(cfg.dim_q), -1, dtype=np.int64)
            for p in _layout["pair_keys"]:
                pi = pair_idx_of[p]
                for l, q_idx_list in _layout["pair_ln_index"][p].items():
                    for q in q_idx_list:
                        q_to_pl[int(q)] = pi * L + int(l)
            self._gating_q_to_pl = tf.constant(q_to_pl)
        else:
            self.gates_pair_l = None
            self._gating_q_to_pl = None

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
            if self.descriptor_gating_enabled:
                raise NotImplementedError(
                    "descriptor_preprocess_contract is mutually exclusive "
                    "with descriptor_gating_enabled in this build. Set one to off.")
            if bool(getattr(cfg, "descriptor_per_l_ann_heads", False)) \
                    and self.descriptor_preprocess_contract in ("angular", "both"):
                raise NotImplementedError(
                    f"descriptor_preprocess_contract={self.descriptor_preprocess_contract!r} "
                    f"collapses the l axis and is incompatible with "
                    f"descriptor_per_l_ann_heads. Set one to off.")
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
                # Init: Glorot-style with fan_in = α (one c-vector contracts
                # over α primitives per centre/neighbour pair). Per-element
                # σ on each c gives a roughly unit-scale bilinear product.
                init_norm = float(_lay["coef_init_norm"])
                if init_scheme == "glorot":
                    fan_in = max(1, alpha_max)
                    fan_out = max(1, n_max_out)
                    limit = float(np.sqrt(6.0 / (fan_in + fan_out)))
                    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)))
                    init_np = rng.uniform(
                        -limit, limit, size=coef_shape).astype(np.float32)
                elif init_scheme in ("mean", "sum"):
                    # Uniform fan-in normalised init. Reasonable starting
                    # point; SNES then searches.
                    init_np = np.full(coef_shape, init_norm, dtype=np.float32)
                else:
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

    def _U_full_composed(self, U_pair: tf.Tensor | list | None = None) -> tf.Tensor:
        """Compose the per-layer effective Q×Q mixing matrices into one.

        Accepts either:
          - None: use self.U_pair_list (the per-layer variables).
          - a tf.Tensor of single-layer V shape: legacy single-layer path
            (used when callers explicitly pass a candidate's V_pair).
          - a list / tuple of tf.Tensors (one per layer): the multi-layer
            candidate-supplied path.

        For a single layer this returns _U_full(U_pair) unchanged so the
        N=1 linear path is bit-identical to the legacy code. For N>1
        linear layers the layers compose:
            q_final = V_N · ... · V_1 · q
            U_composed = V_N · ... · V_1
        Per-type mixing keeps the per-type Q×Q product per central type.
        """
        if not self.descriptor_mixing:
            raise RuntimeError("_U_full_composed called with mixing disabled")
        if U_pair is None:
            U_list_input = self.U_pair_list
        elif isinstance(U_pair, (list, tuple)):
            U_list_input = list(U_pair)
        else:
            # Single tensor — legacy single-layer fold path.
            return self._U_full(U_pair)
        if len(U_list_input) == 1:
            return self._U_full(U_list_input[0])
        # Build per-layer U_full, then multiply right-to-left.
        # Each U_full has shape [..., (T?), Q, Q]; matmul composes on Q.
        U_total = self._U_full(U_list_input[0])
        for k in range(1, len(U_list_input)):
            U_k = self._U_full(U_list_input[k])
            # U_total ← U_k · U_total (apply layer 0 first, then 1, etc.).
            U_total = tf.matmul(U_k, U_total)
        return U_total

    def _R_full(self, R_pair: tf.Tensor | None = None) -> tf.Tensor:
        """Assemble the per-type output-side mixing R = I + V_R.

        R_pair stores the residual V_R = R − I that SNES reconstructs
        each candidate. V_R has shape [(C,) T, H, H]; the returned R has
        the same shape. With V_R initialised at zero, R == I_H at gen 0.

        Args:
          R_pair: residual V_R (defaults to self.R_pair). May carry a
                  leading candidate axis.

        Returns:
          R = I + V_R of shape [(C,) T, H, H].
        """
        V = self.R_pair if R_pair is None else R_pair
        I = tf.eye(self._R_H, dtype=V.dtype)
        return I + V

    def _W0_eff(self, W0: tf.Tensor,
                U_pair: tf.Tensor | list | None = None,
                V_cross_override: tf.Tensor | None = None,
                gates_override: tf.Tensor | None = None) -> tf.Tensor:
        """Pre-multiply W0 by U_full^T along the Q axis. Equivalent
        to mixing the descriptor (desc' = U_full · desc) but absorbs
        the mixing into the weights so the rest of the forward / the
        backprop / the dipole sum use raw descriptors and raw
        grad_values unchanged.

        For N>1 stacked LINEAR layers the layers compose into a single
        Q×Q matrix, which is then folded as usual. NONLINEAR mixing
        cannot be folded — callers using that path must mix descriptors
        explicitly via `_mix_descriptors`.

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
        # Resolve effective per-(type, channel) gates, if gating is enabled.
        # Shape will be [..., T, Q] after the gather; broadcasts over H.
        gates_q = None
        if self.descriptor_gating_enabled:
            gates_src = (gates_override
                         if gates_override is not None
                         else self.gates_pair_l)
            if gates_src is not None and self._gating_q_to_pl is not None:
                gates_q = tf.gather(gates_src, self._gating_q_to_pl, axis=-1)

        if not self.descriptor_mixing:
            # No mixing — if gating is on, fold gate into W0; else passthrough.
            if gates_q is not None:
                return W0 * gates_q[..., :, :, tf.newaxis]
            return W0
        if self.descriptor_mixing_nonlinear:
            raise NotImplementedError(
                "Nonlinear descriptor mixing has no forward path: "
                "`descriptor_mixing_nonlinear=True` allocates a `b_mix_list` "
                "and packs N stacked layers into the SNES parameter vector, "
                "but no consumer applies tanh(U·q + b) in predict / "
                "predict_batch / predict_batch_candidates. Enabling this "
                "flag inflates `dim` with dead-gradient entries that hurt "
                "SNES capacity without contributing to the loss. Implement "
                "an explicit nonlinear mixing layer in the forward pass "
                "before flipping this flag.")
        # Resolve the per-layer V tensors. For N>1 stacked LINEAR layers
        # we compose in the small sub-block space (see `_compose_V_blocks`)
        # so the expensive `_U_full` Q×Q assembly runs ONCE — not N times —
        # and the Q×Q@Q×Q composition matmul disappears entirely. At Q=165
        # this is the difference between ~825M and ~1.85B FLOPs per gen.
        if U_pair is None:
            U_list = self.U_pair_list
        elif isinstance(U_pair, (list, tuple)):
            U_list = list(U_pair)
        else:
            U_list = [U_pair]
        V_eff = U_list[0] if len(U_list) == 1 else self._compose_V_blocks(U_list)
        U_full = self._U_full(V_eff)
        # Optional cross-channel layer: apply U_cross AFTER U_existing.
        # Mathematically: q' = U_cross · U_existing · q
        # In terms of the W0_eff fold: W0_eff = (U_cross · U_existing)^T · W0
        # which we compute as U_combined = U_cross @ U_full and reuse the
        # existing einsum. Q×Q@Q×Q matmul is ~5 MFLOPs at Q=165 — small.
        if (self.descriptor_mixing_cross_layer
                and self.V_cross is not None
                and V_cross_override is None):
            U_cross = tf.eye(self.cfg.dim_q, dtype=U_full.dtype) + self.V_cross
            if U_full.shape.rank == 3 and self.descriptor_mixing_per_type:
                # Per-type: U_full has shape [..., T, Q, Q]; broadcast cross.
                U_full = tf.einsum("qp,...tpr->...tqr", U_cross, U_full)
            else:
                U_full = tf.matmul(U_cross, U_full)
        elif V_cross_override is not None:
            U_cross = tf.eye(self.cfg.dim_q, dtype=U_full.dtype) + V_cross_override
            U_full = tf.matmul(U_cross, U_full)
        if self.descriptor_mixing_per_type:
            W0_eff = tf.einsum('...tqp,...tqh->...tph', U_full, W0)
        else:
            W0_eff = tf.einsum('...qp,...tqh->...tph', U_full, W0)
        # Apply per-(type, q) gating to the post-mixing W0.
        # Order does not matter here: U is block-diagonal in the (pair, l)
        # basis and the gate g_{t, (pair, l)} is constant within each
        # bs×bs block, so the gate is a scalar times the identity inside
        # every block. Scalar × identity commutes with the block rotation
        # (U_block · (g·I) = g · U_block = (g·I) · U_block), hence
        #     U · diag(g_t) = diag(g_t) · U
        # and `gate then rotate` = `rotate then gate` produce identical q'.
        # Folded into W0_eff:
        #     W0_eff = diag(g_t) · Uᵀ · W0 = Uᵀ · diag(g_t) · W0.
        # We multiply by gates_q AFTER the Uᵀ·W0 einsum because it cheaply
        # broadcasts along H; the math is invariant to the order.
        if gates_q is not None:
            W0_eff = W0_eff * gates_q[..., :, :, tf.newaxis]
        return W0_eff

    def _W0_preprocess_eff(self, W0: tf.Tensor,
                            W_pre_override: tf.Tensor | None = None) -> tf.Tensor:
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
                                 c: tf.Tensor) -> tf.Tensor:
        """NEP4 bilinear (rank-1 outer-product) fold of W0.

        Implements the equation
            g[t, n'', l] = Σ_{n, n'} c[t, s(n), n'', k(n)]
                                  · c[t, s(n'), n'', k(n')]
                                  · p[n, n', l]
        as a transformation of W0 from the [n'', l]-indexed storage at
        Q_new = n_max_out · L to the [n, n', l]-indexed raw-descriptor
        layout at Q_raw, so the existing matmul code can continue to
        operate against the unmodified raw descriptor:
            U = W0_eff[t, q_raw] · desc_raw[q_raw]
              ≡ W0[t, q_new(n'', l)] · g[t, n'', l]
              ≡ W0[t, q_new(n'', l(q))] · Σ_{n,n'} c·c · p

        Args:
          W0: [(C,) T, n_max_out · L, H]   weights at Q_new
          c:  [(C,) T_centre, T_neighbour, n_max_out, α]   NEP4 coeffs

        Returns:
          W0_eff: [(C,) T, Q_raw, H]   weights at the raw descriptor dim
        """
        T_c = int(self.cfg.num_types)
        T_n = T_c
        n_max_out = int(self._nep4_n_max_out)
        L_ = int(self._nep4_L)
        alpha = int(self._nep4_alpha_max)
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
        # Gather along the flattened (T_n, α) axis using global-n indices.
        # A_a[..., t, q, n''] = c_flat[..., t, n_global(q), n'']
        # A_b[..., t, q, n''] = c_flat[..., t, n'_global(q), n'']
        A_a = tf.gather(c_flat, self._nep4_n_global, axis=-2)
        A_b = tf.gather(c_flat, self._nep4_np_global, axis=-2)
        AB = A_a * A_b   # [..., T_c, Q_raw, n_max_out]   (rank-1 outer product)
        # Reshape W0 to expose (n_max_out, L). W0 [..., T, n_max_out·L, H].
        W0_shape = tf.shape(W0)
        H_ = W0_shape[-1]
        # Build new shape preserving any leading batch (e.g. candidate) axes.
        leading = W0_shape[:-2]
        new_shape = tf.concat(
            [leading, tf.constant([n_max_out, L_], dtype=W0_shape.dtype),
             tf.reshape(H_, [1])], axis=0)
        W0_NLH = tf.reshape(W0, new_shape)   # [..., T, n_max_out, L, H]
        # Gather along the l axis using l_of_q:
        # W0_at_q[..., t, n'', q, h] = W0_NLH[..., t, n'', l_of_q(q), h]
        W0_at_q = tf.gather(W0_NLH, self._nep4_l_of_q, axis=-2)
        # Combine: sum over n''.
        # AB         [..., T_c, Q_raw, n_max_out]      (no h axis)
        # W0_at_q    [..., T,   n_max_out, Q_raw, H]   (no h axis on n'')
        # Want W0_eff[..., T, Q_raw, H] = Σ_{n''} AB[t, q, n''] · W0_at_q[t, n'', q, h]
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
        if self.descriptor_mixing_output_layer:
            raise NotImplementedError(
                "predict() (single-structure inference path) does not yet "
                "support descriptor_mixing_output_layer. Use predict_batch() / "
                "score() instead, or disable output-side mixing for this call.")
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
            f"{self._activation_name!r}. Extend _ACTIVATION_REGISTRY and add a "
            f"branch here.")

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
        """Train the model using the SNES evolutionary optimizer.

        Args:
            train_data    : dict with keys descriptors, gradients, grad_index,
                            positions, Z_int, targets, boxes (lists over structures)
            val_data      : same structure, used for validation each generation
            plot_callback : optional callable(history, gen) for periodic plotting
            resume_state  : optional dict from `model_io.load_checkpoint`,
                            carries SNES distribution + best-val + history +
                            RNG state. When provided, training continues from
                            `resume_state['last_gen'] + 1`.

        Returns:
            history         : dict with keys generation, train_loss, val_loss (lists)
            final_model     : TNEP model with weights from the last generation
            best_val_model  : TNEP model with weights from the best validation generation
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
        # Streaming chunked scoring. Bounds peak memory to one chunk's
        # gradient slice — and is the only sensible mode when grad_values
        # is disk-backed. With cfg.chunk_prefetch the disk-pipe of chunk N+1
        # overlaps the model forward of chunk N.
        from data import prefetched_chunks
        S_test = test_data["num_atoms"].shape[0]
        chunk_sz = (self.cfg.batch_chunk_size
                    if self.cfg.batch_chunk_size is not None else S_test)
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
            pred_parts.append(self.predict_batch(
                chunk["descriptors"], chunk["grad_values"],
                chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
                chunk["positions"], chunk["Z_int"], chunk["boxes"],
                chunk["atom_mask"],
                W0_eff, self.b0, self.W1, self.b1,
                W0_pol_eff,
                getattr(self, 'b0_pol', None),
                getattr(self, 'W1_pol', None),
                getattr(self, 'b1_pol', None),
                W0_2=getattr(self, "W0_2", None),
                b0_2=getattr(self, "b0_2", None),
                W0_2_pol=getattr(self, "W0_2_pol", None),
                b0_2_pol=getattr(self, "b0_2_pol", None),
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
                      W_atom: tf.Tensor | None = None,
                      W0_2: tf.Tensor | None = None, b0_2: tf.Tensor | None = None,
                      W0_2_pol: tf.Tensor | None = None,
                      b0_2_pol: tf.Tensor | None = None) -> tf.Tensor:
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

        # Output-side mixing: h1_R = h1 · R^T (per-type rotation). Init at
        # zero → R = I → off-path equivalence. h1 (pre-rotation) preserved
        # for the backward chain's activation_grad.
        if self.descriptor_mixing_output_layer:
            R_eff = self._R_full()   # uses self.R_pair, shape [T, H, H]
            h1_R_terms = []
            for t in range(self.num_types):
                R_t = R_eff[t, :, :]                            # [H, H]
                h1_R_t = tf.einsum('bah,Kh->baK', h1, R_t)      # [B, A, H]
                h1_R_terms.append(h1_R_t * type_masks[t])
            h1_R = tf.add_n(h1_R_terms)
        else:
            R_eff = None
            h1_R = h1

        # Optional second hidden layer: a2 = h1_R @ W0_2 + b0_2; h2 = f(a2).
        # Falls back to h2 = h1_R when W0_2 is None (legacy single-hidden path).
        z2 = None
        if W0_2 is not None and b0_2 is not None:
            b0_2_t = tf.gather(b0_2, Z)   # [B, A, H2]
            z2 = tf.add_n([
                tf.einsum('bah,hk->bak', h1_R, W0_2[t]) * type_masks[t]
                for t in range(self.num_types)
            ]) + b0_2_t
            h2 = self.activation(z2)
            h2 = h2 * atom_mask[:, :, tf.newaxis]
        else:
            h2 = h1_R

        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h2 * W1_t, axis=2) + b1  # [B, A]
            E = E * atom_mask
            # H-center skip (see comment in predict): exclude H atoms.
            if bool(getattr(self.cfg, "skip_h_centers", False)):
                E = E * tf.cast(Z != 1, tf.float32)
            E = tf.reduce_sum(E, axis=1, keepdims=True)  # [B, 1]
            return -E

        # de_dq: energy derivative w.r.t. *raw* descriptor (because we
        # absorbed U into W0). No pull-back needed — dipole/pol path
        # below uses raw grad_values directly.
        # Chain rule (two hidden layers):
        #   de/dh2 = W1
        #   de/da2 = f'(z2) * de/dh2
        #   de/dh1_R = de/da2 @ W0_2^T
        #   de/dh1 = de/dh1_R @ R                  ← R-rotation backward
        #   de/da1 = f'(z1) * de/dh1
        #   de/dq  = de/da1 @ W0^T
        if W0_2 is not None and b0_2 is not None:
            de_da2 = self._activation_grad(h2, z2) * W1_t                     # [B, A, H2]
            de_dh1_R = tf.add_n([
                tf.einsum('bak,hk->bah', de_da2, W0_2[t]) * type_masks[t]
                for t in range(self.num_types)
            ])
            if self.descriptor_mixing_output_layer:
                de_dh1_terms = []
                for t in range(self.num_types):
                    R_t = R_eff[t, :, :]
                    de_dh1_t = tf.einsum('baK,Kh->bah', de_dh1_R, R_t)
                    de_dh1_terms.append(de_dh1_t * type_masks[t])
                de_dh1 = tf.add_n(de_dh1_terms)
            else:
                de_dh1 = de_dh1_R
            de_da = self._activation_grad(h1, z1) * de_dh1                    # [B, A, H]
        else:
            # Single-hidden: U = h1_R · W1 + b1, so ∂U/∂h1 = W1 · R.
            if self.descriptor_mixing_output_layer:
                W1_eff_terms = []
                for t in range(self.num_types):
                    R_t = R_eff[t, :, :]
                    W1_eff_t = tf.einsum('baK,Kh->bah', W1_t, R_t)
                    W1_eff_terms.append(W1_eff_t * type_masks[t])
                W1_eff = tf.add_n(W1_eff_terms)
            else:
                W1_eff = W1_t
            de_da = self._activation_grad(h1, z1) * W1_eff
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
                W0_pol_use, b0_pol, W1_pol, b1_pol, B,
                W0_2_pol=W0_2_pol, b0_2_pol=b0_2_pol)

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
                                  U_pair: tf.Tensor | list | None = None,
                                  W0_2: tf.Tensor | None = None,
                                  b0_2: tf.Tensor | None = None,
                                  V_cross: tf.Tensor | None = None,
                                  gates: tf.Tensor | None = None,
                                  W_pre_angular: tf.Tensor | None = None,
                                  R_pair: tf.Tensor | None = None) -> tf.Tensor:
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
        H2 = self._H2
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
        if (self.descriptor_mixing and U_pair is not None) \
                or (self.descriptor_gating_enabled and gates is not None):
            W0 = self._W0_eff(W0, U_pair, V_cross_override=V_cross,
                              gates_override=gates)

        # Preprocessing contraction: fold the per-type per-channel
        # coefficients into W0 along the Q_new axis, producing a W0_eff
        # at Q_raw. The matmul code below sees Q = Q_raw uniformly and
        # the dipole backward yields de_dq at Q_raw, ready to contract
        # with the precomputed raw W_atom. Mutually exclusive with
        # mixing/gating (so the two folds never compose in this build).
        if self.descriptor_preprocess_contract != "off":
            W0 = self._W0_preprocess_eff(W0, W_pre_override=W_pre_angular)

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

        # ── Activation (first hidden layer) ───────────────────────────────────
        # z1 = pre_h + b0  is the pre-activation; preserve it for the
        # swish-side backward chain rule below.
        z1 = pre_h + b0_t_all
        h1 = self.activation(z1)
        h1 = h1 * atom_mask[tf.newaxis, :, :, tf.newaxis]

        # ── Output-side mixing: h1' = h1 · R^T (per-type rotation) ─────────────
        # R = I + V_R where V_R is the per-type residual stored in R_pair.
        # Init at zero → R = I → bit-identical to the off-path. h1 (pre-
        # rotation) is preserved so the backward chain's activation_grad
        # still receives the right (h1, z1) pair.
        if self.descriptor_mixing_output_layer:
            R_eff = self._R_full(R_pair)  # [C, T, H, H]
            h1_R_terms = []
            for t in range(T):
                R_t = R_eff[:, t, :, :]
                h1_R_t = tf.einsum('cbah,cKh->cbaK', h1, R_t)
                h1_R_terms.append(h1_R_t * type_masks[t][tf.newaxis])
            h1_R = tf.add_n(h1_R_terms)
        else:
            R_eff = None
            h1_R = h1

        # ── Optional second hidden layer ───────────────────────────────────────
        z2 = None
        if W0_2 is not None and b0_2 is not None:
            # h1_R: [C, B, A, H]; W0_2: [C, T, H, H2]. The layer-1 [Q,C*H]
            # GEMM-flatten trick doesn't transfer here — the per-type mass is
            # already on the C axis of h1_R, so a per-type einsum + mask sum is
            # the natural fused form (and ditto for the backward chain).
            pre_h2_terms = []
            for t in range(T):
                W0_2_t = W0_2[:, t, :, :]                                         # [C, H, H2]
                ph2 = tf.einsum('cbah,chk->cbak', h1_R, W0_2_t)                  # [C,B,A,H2]
                pre_h2_terms.append(ph2 * type_masks[t][tf.newaxis])
            pre_h2 = tf.add_n(pre_h2_terms)
            b0_2_t_all = tf.reshape(tf.gather(b0_2, Z_flat, axis=1),
                                    [C, B, A, H2])
            z2 = pre_h2 + b0_2_t_all
            h2 = self.activation(z2)
            h2 = h2 * atom_mask[tf.newaxis, :, :, tf.newaxis]
        else:
            h2 = h1_R

        # ── PES ───────────────────────────────────────────────────────────────
        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h2 * W1_t_all, axis=3) + b1[:, tf.newaxis, tf.newaxis]
            E = E * atom_mask[tf.newaxis]
            return -tf.reduce_sum(E, axis=2, keepdims=True)  # [C, B, 1]

        # ── Dipole: backward matmul ───────────────────────────────────────────
        # Chain rule through both hidden layers when H2 is set.
        if W0_2 is not None and b0_2 is not None:
            de_da2 = self._activation_grad(h2, z2) * W1_t_all                     # [C,B,A,H2]
            # de_dh1_R = ∂U / ∂h1_R via the W0_2 chain.
            de_dh1R_terms = []
            for t in range(T):
                W0_2_t = W0_2[:, t, :, :]                                         # [C, H, H2]
                dh1R = tf.einsum('cbak,chk->cbah', de_da2, W0_2_t)               # [C,B,A,H]
                de_dh1R_terms.append(dh1R * type_masks[t][tf.newaxis])
            de_dh1_R = tf.add_n(de_dh1R_terms)                                    # [C,B,A,H]
            # Apply R rotation in the backward direction:
            #   de_dh1[h] = Σ_K de_dh1_R[K] · R[K, h]
            if self.descriptor_mixing_output_layer:
                de_dh1_terms = []
                for t in range(T):
                    R_t = R_eff[:, t, :, :]
                    de_dh1_t = tf.einsum('cbaK,cKh->cbah', de_dh1_R, R_t)
                    de_dh1_terms.append(de_dh1_t * type_masks[t][tf.newaxis])
                de_dh1 = tf.add_n(de_dh1_terms)
            else:
                de_dh1 = de_dh1_R
            de_da = self._activation_grad(h1, z1) * de_dh1                        # [C,B,A,H]
        else:
            # Single-hidden path: U = h1_R · W1 + b1; ∂U/∂h1_R = W1, so
            # ∂U/∂h1 = W1 · R (per-type). Fuse with W1 by computing
            # an effective W1_eff per atom.
            if self.descriptor_mixing_output_layer:
                W1_eff_terms = []
                for t in range(T):
                    R_t = R_eff[:, t, :, :]
                    W1_eff_t = tf.einsum('cbaK,cKh->cbah', W1_t_all, R_t)
                    W1_eff_terms.append(W1_eff_t * type_masks[t][tf.newaxis])
                W1_eff = tf.add_n(W1_eff_terms)
            else:
                W1_eff = W1_t_all
            de_da    = self._activation_grad(h1, z1) * W1_eff   # [C, B, A, H]
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

    def predict_per_l_batch_candidates(self,
                                        descriptors: tf.Tensor,
                                        W_atom: tf.Tensor | None,
                                        Z: tf.Tensor,
                                        atom_mask: tf.Tensor,
                                        W0_per_l: list,
                                        b0_per_l: list,
                                        W1_per_l: list,
                                        b1_per_l: list,
                                        gates: tf.Tensor | None = None) -> tf.Tensor:
        """Per-l-head version of predict_batch_candidates.

        Computes U_i = Σ_l ANN_{Z[i], l}(q_i restricted to l) per atom,
        then runs the standard dipole / PES contraction per-l and sums.

        Inputs are lists of length L (one per angular momentum), each
        carrying the per-candidate tensors:
            W0_per_l[l] : [C, T, Q_l, H]
            b0_per_l[l] : [C, T, H]
            W1_per_l[l] : [C, T, H]
            b1_per_l[l] : [C]

        Returns:
            target_mode 0 (PES)    : [C, B, 1]  (sum over atoms)
            target_mode 1 (dipole) : [C, B, 3]
        Target mode 2 (polarisability) is NOT supported in this first impl.
        """
        if self.cfg.target_mode == 2:
            raise NotImplementedError(
                "predict_per_l_batch_candidates does not yet support "
                "target_mode=2 (polarisability).")
        H = self.num_neurons
        T = self.num_types
        L = self._per_l_L
        Q = self.dim_q
        B = tf.shape(descriptors)[0]
        A = tf.shape(descriptors)[1]
        C = tf.shape(W0_per_l[0])[0]
        Z_flat = tf.reshape(Z, [B * A])

        type_masks = [
            tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
            for t in range(T)]

        # Apply gating to the descriptor (gates flow to channels before
        # per-l partitioning, so just multiply into descriptors here).
        if self.descriptor_gating_enabled and gates is not None:
            gates_q = tf.gather(gates, self._gating_q_to_pl, axis=-1)  # [C,T,Q]
            # gates_q has [C, T, Q]; descriptors [B, A, Q]; to apply
            # type-dependent gates per atom: for each candidate c we need
            # gates_q[c, Z[i], :] for atom i. Build a gated descriptor as:
            #   desc_gated[c, b, a, q] = descriptors[b, a, q] * gates_q[c, Z[b,a], q]
            gates_per_atom = tf.gather(gates_q, Z, axis=1)            # [C, B, A, Q]
            descriptors_for_l = descriptors[tf.newaxis, ...] * gates_per_atom
        else:
            descriptors_for_l = None  # use per-l gather of bare descriptors

        accumulated_target = None   # PES energy or dipole, summed over l
        for l in range(L):
            Q_l = self._per_l_Q_l[l]
            if Q_l == 0:
                continue
            q_indices_l = self._per_l_q_indices[l]
            if descriptors_for_l is not None:
                q_l = tf.gather(descriptors_for_l, q_indices_l, axis=-1)  # [C,B,A,Q_l]
                # q_l_flat for the matmul: need per-candidate batched matmul.
                # Path: per-type GEMMs on [C, B*A, Q_l].
                q_l_flat = tf.reshape(q_l, [C, B * A, Q_l])
                pre_h_l_terms = []
                for t in range(T):
                    W0_l_t = W0_per_l[l][:, t, :, :]                     # [C, Q_l, H]
                    ph = tf.matmul(q_l_flat, W0_l_t)                     # [C, B*A, H]
                    ph = tf.reshape(ph, [C, B, A, H])
                    pre_h_l_terms.append(ph * type_masks[t][tf.newaxis])
                pre_h_l = tf.add_n(pre_h_l_terms)                         # [C, B, A, H]
            else:
                # Bare descriptors (no gating). Slim path: gather q-slice
                # once into [B, A, Q_l], then per-type [B*A, Q_l] @
                # [Q_l, C*H] GEMM (mirrors the predict_batch_candidates
                # trick when gating is off).
                q_l = tf.gather(descriptors, q_indices_l, axis=-1)        # [B, A, Q_l]
                desc_flat = tf.reshape(q_l, [B * A, Q_l])
                pre_h_l_terms = []
                for t in range(T):
                    W0_l_t = W0_per_l[l][:, t, :, :]                     # [C, Q_l, H]
                    W0_l_t_mat = tf.reshape(
                        tf.transpose(W0_l_t, [1, 0, 2]),
                        [Q_l, C * H])                                    # [Q_l, C*H]
                    ph_flat = tf.matmul(desc_flat, W0_l_t_mat)           # [B*A, C*H]
                    ph = tf.transpose(
                        tf.reshape(ph_flat, [B, A, C, H]),
                        [2, 0, 1, 3])                                    # [C,B,A,H]
                    pre_h_l_terms.append(ph * type_masks[t][tf.newaxis])
                pre_h_l = tf.add_n(pre_h_l_terms)

            # Add per-atom bias b0_l[c, Z[a], :].
            b0_l_t_all = tf.reshape(
                tf.gather(b0_per_l[l], Z_flat, axis=1),
                [C, B, A, H])
            W1_l_t_all = tf.reshape(
                tf.gather(W1_per_l[l], Z_flat, axis=1),
                [C, B, A, H])
            # Pre-activation z_l preserved for the swish-side backward chain.
            z_l = pre_h_l + b0_l_t_all
            h_l = self.activation(z_l)
            h_l = h_l * atom_mask[tf.newaxis, :, :, tf.newaxis]

            if self.cfg.target_mode == 0:
                # PES per-l contribution.
                U_per_atom_l = (tf.reduce_sum(h_l * W1_l_t_all, axis=3)
                                + b1_per_l[l][:, tf.newaxis, tf.newaxis])
                U_per_atom_l = U_per_atom_l * atom_mask[tf.newaxis]
                E_l = -tf.reduce_sum(U_per_atom_l, axis=2, keepdims=True)
                accumulated_target = E_l if accumulated_target is None \
                    else accumulated_target + E_l
                continue

            # Dipole branch.
            de_da_l = self._activation_grad(h_l, z_l) * W1_l_t_all        # [C,B,A,H]
            de_da_l_flat = tf.reshape(de_da_l, [C, B * A, H])             # [C,B*A,H]
            de_dq_l_terms = []
            for t in range(T):
                W0_l_t_T = tf.transpose(
                    W0_per_l[l][:, t, :, :], [0, 2, 1])                  # [C, H, Q_l]
                dq_flat = tf.matmul(de_da_l_flat, W0_l_t_T)              # [C,B*A,Q_l]
                dq = tf.reshape(dq_flat, [C, B, A, Q_l])
                de_dq_l_terms.append(dq * type_masks[t][tf.newaxis])
            de_dq_l = tf.add_n(de_dq_l_terms)                             # [C,B,A,Q_l]

            # Restrict W_atom to this l's q-indices and contract.
            W_atom_l = tf.gather(W_atom, q_indices_l, axis=-1)            # [B,A,3,Q_l]
            dipole_l = -tf.einsum(
                'cbaq,basq->cbs', de_dq_l, W_atom_l)                      # [C,B,3]
            accumulated_target = dipole_l if accumulated_target is None \
                else accumulated_target + dipole_l

        return accumulated_target

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
                            B: tf.Tensor,
                            W0_2_pol: tf.Tensor | None = None,
                            b0_2_pol: tf.Tensor | None = None) -> tf.Tensor:
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
        # Optional second hidden layer for the polarizability scalar ANN.
        if W0_2_pol is not None and b0_2_pol is not None:
            b0_2p_t = tf.gather(b0_2_pol, Z)
            h_pol2 = tf.add_n([
                tf.einsum('bah,hk->bak', h_pol, W0_2_pol[t]) * type_masks_p[t]
                for t in range(self.num_types)
            ]) + b0_2p_t
            h_pol2 = self.activation(h_pol2)
            h_pol2 = h_pol2 * atom_mask[:, :, tf.newaxis]
            h_pol = h_pol2
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
