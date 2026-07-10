from __future__ import annotations

import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING, Callable

from loss_functions import squared_error_per_structure

if TYPE_CHECKING:
    from TNEP import TNEP

def _format_duration(seconds: float) -> str:
    """Format a non-negative duration as HH:MM:SS, or as Dd HH:MM:SS
    when it exceeds 24 h. Avoids the day-rollover bug in
    `time.strftime('%H:%M:%S', time.gmtime(seconds))`, which silently
    truncates anything past one day to the hour-of-day component.
    """
    if not (seconds == seconds) or seconds < 0:  # NaN or negative guard
        return "--:--:--"
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _set_model_params(model: TNEP, *params: tf.Tensor) -> None:
    """Assign weight tensors produced by SNES.reconstruct_params_tf into
    the TNEP model's Variables.

    Tail order:
        ANN                            : W0, b0, W1, b1
        + optional pol ANN (mode 2)    : W0_pol, b0_pol, W1_pol, b1_pol
        + optional U_pair (mixing)     : single tensor — the per-pair
                                         skew/residual mixing layer
        + optional W_pre_angular (preprocess_contract != "off")
    """
    params = list(params)
    idx = 0
    model.W0.assign(params[idx]); idx += 1
    model.b0.assign(params[idx]); idx += 1
    model.W1.assign(params[idx]); idx += 1
    model.b1.assign(params[idx]); idx += 1
    if model.cfg.target_mode == 2:
        model.W0_pol.assign(params[idx]); idx += 1
        model.b0_pol.assign(params[idx]); idx += 1
        model.W1_pol.assign(params[idx]); idx += 1
        model.b1_pol.assign(params[idx]); idx += 1
    if (getattr(model, "descriptor_mixing", False)
            and getattr(model, "U_pair", None) is not None
            and idx < len(params)):
        model.U_pair.assign(params[idx]); idx += 1
    if (idx < len(params)
            and getattr(model, "descriptor_preprocess_contract", "off") != "off"
            and getattr(model, "W_pre_angular", None) is not None):
        model.W_pre_angular.assign(params[idx]); idx += 1

class SNES:
    """Separable Natural Evolution Strategy optimizer for TNEP.

    Maintains a diagonal Gaussian search distribution N(mu, diag(sigma^2)) over
    the flattened parameter vector of the TNEP model.  Each generation:
      1. Sample pop_size candidates: z_p = mu + sigma * s_p,  s_p ~ N(0,1)
      2. Evaluate fitness (RMSE) for each candidate on a random batch
      3. Rank candidates by fitness, pair with log-shaped utilities
      4. Update:  mu    <- mu + sigma * sum_p u_p * s_p
                  sigma <- sigma * exp(eta_sigma * sum_p u_p * (s_p^2 - 1))

    All sampling, ranking, and update operations use TensorFlow ops to
    stay on GPU.  Only scalar history values are transferred to CPU.

    Total parameter count = num_types * dim_q * num_neurons   (W0)
                          + num_types * num_neurons            (b0)
                          + num_types * num_neurons            (W1)
                          + 1                                  (b1)
    """

    def __init__(self, model: TNEP) -> None:
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        self.dim_q = self.cfg.dim_q
        self.batch_size = self.cfg.batch_size

        # TF random generator for all stochastic ops in training loop
        if self.cfg.seed is not None:
            self.tf_rng = tf.random.Generator.from_seed(self.cfg.seed)
        else:
            self.tf_rng = tf.random.Generator.from_non_deterministic_state()

        # Total number of trainable parameters
        self.H = int(self.cfg.num_neurons)
        n_W0 = self.cfg.num_types * self.cfg.dim_q * self.H
        n_b0 = self.cfg.num_types * self.H
        n_W1 = self.cfg.num_types * self.H
        n_b1 = 1
        self.n_typed = n_W0 + n_b0 + n_W1
        # Per-type param count (per type t): W0 Q·H + b0 H + W1 H.
        self._n_per_type = self.cfg.dim_q * self.H + 2 * self.H
        self.n_primary = self.n_typed + n_b1
        self._n_W0 = n_W0
        self._n_b0 = n_b0
        self._n_W1 = n_W1
        self._n_b1 = n_b1
        # Mode 2 (polarizability) adds a second ANN with identical shape
        if self.cfg.target_mode == 2:
            self.n_anns_total = 2 * self.n_primary
        else:
            self.n_anns_total = self.n_primary
        # Optional U_pair tail (descriptor-mixing layer). Stored flat
        # at the end of the parameter vector. Block layout is shared
        # across central types (per Phase B): one [bs, bs] matrix per
        # species pair, padded to max_block_size inside the model.
        # SNES sees only the active entries — we pack them densely
        # here and unpack via reconstruct_params_tf.
        if getattr(self.cfg, "descriptor_mixing", False):
            # Reuse TNEP's already-computed mixing layout so SNES and
            # the model agree on (num_pairs, L_eff, α_per_pair, block
            # sizes). When preprocess contraction is on, this is the
            # post-preprocess layout (Q_new, L_eff < L); otherwise it's
            # the raw SOAP-turbo block layout. Previously SNES
            # recomputed the RAW layout here, which gave L = l_max+1
            # even when TNEP allocated U_pair at L_eff < L — SNES then
            # walked dead params and ran expm at the larger batch size.
            from DescriptorBuilderGPU import descriptor_block_layout
            self._mix_layout = getattr(
                self.model, "_mix_layout", None)
            if self._mix_layout is None:
                self._mix_layout = descriptor_block_layout(self.cfg)
            self._mix_per_type = bool(
                getattr(self.cfg, "descriptor_mixing_per_type", False))
            # When the regulariser is "cayley", each [bs, bs] block is
            # parameterised by the upper triangle of a skew-symmetric
            # matrix A (bs(bs-1)/2 free params). The dense block V is
            # then reconstructed via the Cayley transform inside
            # reconstruct_params_tf; SNES sees only the upper-triangle
            # storage. Halves (roughly) the search-space dimensionality
            # and guarantees U lies on the rotation group regardless of
            # any λ — see plan and TNEPconfig docs.
            # Two STRUCTURAL orthogonal parameterisations share the same
            # skew upper-triangle layout (bs(bs-1)/2 entries per block) —
            # they differ only in the map A → U:
            #   "cayley"  : U = (I − A)(I + A)^{−1}  — rational chord; cannot
            #               represent rotations with −1 eigenvalues except
            #               in the limit |A| → ∞
            #   "expm"    : U = exp(A)                — exponential geodesic;
            #               surjective onto SO(n), no Jacobian singularity
            # `_mix_cayley` stays True for either (it gates the skew layout
            # everywhere downstream); `_mix_orth_map` selects the backend.
            _reg_mode = str(getattr(self.cfg,
                                    "descriptor_mixing_regularizer",
                                    "off")).lower()
            self._mix_cayley = _reg_mode in ("cayley", "expm")
            self._mix_orth_map = _reg_mode if self._mix_cayley else None
            # `block_count(bs)` returns the per-block SNES dim for the
            # current arch+regulariser combination.
            def _block_count(bs: int) -> int:
                return (bs * (bs - 1) // 2) if self._mix_cayley else bs * bs

            # n_U_pair packs only the ACTIVE entries — padded rows/cols
            # don't take SNES degrees of freedom. l_aware layout: per
            # pair × (l_max+1), α² entries (or α(α-1)/2 for cayley/expm).
            self._mix_alpha_per_pair = [
                int(self._mix_layout["alpha_eff_per_pair"][k])
                for k in self._mix_layout["pair_keys"]]
            # Use the layout's L_eff when present (post-preprocess
            # layouts collapse l > l_keep into a single slot, so
            # L_eff < L). Falls back to l_max+1 for the raw layout.
            L = int(self._mix_layout.get("L_eff", int(self.cfg.l_max) + 1))
            self._mix_L = L
            per_T_block = int(sum(L * _block_count(a)
                                   for a in self._mix_alpha_per_pair))
            if self._mix_per_type:
                self.n_U_pair = self.cfg.num_types * per_T_block
            else:
                self.n_U_pair = per_T_block
            # Pre-build the Cayley scatter matrices (one per unique block
            # size used by this arch). Doing it at init keeps the
            # construction out of the @tf.function-traced
            # reconstruct_params_tf path — `tf.constant`s live in eager
            # context and are simply captured into the graph at trace
            # time. Without this, the first reconstruct call inside an
            # `@tf.function` triggers a Python-side `hasattr` + dict-add
            # that's at best benign and at worst confuses tracers under
            # XLA recompilation. See review item H2.
            self._cayley_scatter_cache = {}
            if self._mix_cayley:
                # Collect unique block sizes (per-pair α values).
                block_sizes = set(self._mix_alpha_per_pair)
                for bs in block_sizes:
                    if bs <= 1:
                        continue   # 1×1 skew is trivially zero
                    num_upper = bs * (bs - 1) // 2
                    scat = np.zeros((bs, bs, num_upper), dtype=np.float32)
                    k = 0
                    for i in range(bs):
                        for j in range(i + 1, bs):
                            scat[i, j, k] = 1.0
                            scat[j, i, k] = -1.0
                            k += 1
                    self._cayley_scatter_cache[bs] = tf.constant(scat)

            # ── Non-uniform-α `l_aware` + Cayley/expm fast-path gather ───────
            # Without this, the slow branch of `_extract_t_block_l_aware`
            # loops over `num_pairs` pairs and calls `tf.linalg.solve` /
            # `tf.linalg.expm` per pair — and in eager mode each forces a
            # host↔device sync. For the user's CHO+N config (num_pairs=10,
            # alphas [4,7,7,7,4,7,7,4,7,4]) that's 10 syncs per layer per
            # generation, ≈ 60 ms/gen of pure launch overhead at N=2.
            #
            # The fix: zero-pad each pair's α_p×α_p skew-triangle payload
            # into the upper triangle of a max_α×max_α skew matrix. Both
            # parameterisations map a zero-padded skew to identity in the
            # padded rows/cols, so V = U − I = 0 there — bit-equivalent
            # to the per-pair loop. ONE batched solve/expm per layer.
            self._mix_l_aware_cayley_gather = None
            self._mix_l_aware_cayley_max_payload = None
            if (self._mix_cayley
                    and len(set(self._mix_alpha_per_pair)) > 1):
                alpha_per_pair = self._mix_alpha_per_pair
                n_pairs = len(alpha_per_pair)
                L = self._mix_L
                max_alpha = max(alpha_per_pair)
                max_payload = max_alpha * (max_alpha - 1) // 2

                def _encode_upper(i: int, j: int, bs: int) -> int:
                    return i * (bs - 1) - i * (i - 1) // 2 + (j - i - 1)

                def _decode_max(k_max: int) -> tuple[int, int]:
                    for i_cand in range(max_alpha):
                        start = i_cand * (max_alpha - 1) - i_cand * (i_cand - 1) // 2
                        end = (i_cand + 1) * (max_alpha - 1) - (i_cand + 1) * i_cand // 2
                        if start <= k_max < end:
                            j_cand = (k_max - start) + (i_cand + 1)
                            return i_cand, j_cand
                    raise ValueError(f"k_max={k_max} out of range")

                src_pair_offsets = [0]
                for p in range(n_pairs):
                    payload_p = alpha_per_pair[p] * (alpha_per_pair[p] - 1) // 2
                    src_pair_offsets.append(
                        src_pair_offsets[-1] + L * payload_p)
                sentinel = src_pair_offsets[-1]   # = per_T_block; appended-zero idx

                gather_flat = np.full(
                    n_pairs * L * max_payload, sentinel, dtype=np.int32)
                for p in range(n_pairs):
                    alpha_p = alpha_per_pair[p]
                    payload_p = alpha_p * (alpha_p - 1) // 2
                    base = src_pair_offsets[p]
                    for l in range(L):
                        dst_pl = (p * L + l) * max_payload
                        src_pl = base + l * payload_p
                        for k_max in range(max_payload):
                            i, j = _decode_max(k_max)
                            if i < alpha_p and j < alpha_p:
                                k_p = _encode_upper(i, j, alpha_p)
                                gather_flat[dst_pl + k_max] = src_pl + k_p
                self._mix_l_aware_cayley_gather = tf.constant(
                    gather_flat, dtype=tf.int32)
                self._mix_l_aware_cayley_max_payload = max_payload
        else:
            self._mix_per_type = False
            self._mix_cayley = False
            self._mix_orth_map = None
            self._cayley_scatter_cache = {}
            self.n_U_pair = 0

        # Preprocessing contraction tail (descriptor_preprocess_contract).
        # Per-(centre type, raw channel) scalar coefficients fold into W0
        # via _W0_preprocess_eff. T·Q_raw entries — sits at the end of
        # the mu vector after gates.
        self._preprocess_mode = str(getattr(
            self.cfg, "descriptor_preprocess_contract", "off"))
        if self._preprocess_mode != "off":
            # Derive Q_raw from cfg.dim_q_raw if available, else from the
            # model's stored layout. cfg.dim_q has been overridden to Q_new
            # by TNEP.__init__; cfg.dim_q_raw holds the original.
            Q_raw_pre = int(getattr(self.cfg, "dim_q_raw", self.dim_q))
            T_ = int(self.cfg.num_types)
            self._preprocess_Q_raw = Q_raw_pre
            self._preprocess_per_type = bool(
                self.model.preprocess_per_type)
            # NEP4 fold: per-type ranking labels need to index the rank-4
            # c[T_c, T_n, n_max_out, α] tensor via t_centre = flat // (T·n_max_out·α).
            # Override _preprocess_Q_raw to the slab-per-t-centre size so the
            # existing `flat_idx // _preprocess_Q_raw` label formula computes
            # t_centre correctly.
            if self._preprocess_mode == "nep4_radial":
                _n_max_out = int(self.model._nep4_n_max_out)
                _alpha = int(self.model._nep4_alpha_max)
                self._preprocess_Q_raw = T_ * _n_max_out * _alpha
            # n_preprocess now counts ONLY the summed (learnable) W_pre
            # entries — kept (passthrough) positions are fixed at 1.0 in
            # the Variable and don't move through SNES. The TNEP layer
            # built `_preprocess_summed_count` over the same flat layout
            # we'll scatter into below.
            self.n_preprocess = int(
                self.model._preprocess_summed_count)
            # Initial value (uniform across T and Q_raw) for the mean/sum
            # init schemes. Glorot init is set in mu_init directly.
            init_scheme = str(getattr(
                self.cfg, "descriptor_preprocess_init", "mean"))
            if init_scheme == "mean":
                # Prefer the layout's per-q_raw init vector when present
                # (angular: l=0 → 1.0, l>0 → 1/(L-1)). Stored for use in
                # _build_mu_init; the scalar fallback below is kept for
                # modes that don't supply per-slot init.
                self._preprocess_init_per_q_raw = (
                    self.model._preprocess_layout.get("coef_init_per_q_raw")
                    if self.model._preprocess_layout is not None else None)
                if self._preprocess_mode == "species_pair":
                    self._preprocess_init_value = (
                        1.0 / float(max(1, int(self.cfg.num_types) - 1)))
                else:
                    L_ = int(self.cfg.l_max) + 1
                    self._preprocess_init_value = 1.0 / float(L_)
            elif init_scheme == "sum":
                self._preprocess_init_value = 1.0
            elif init_scheme == "glorot":
                # μ-init zero; effective Glorot scaling is applied via σ
                # at gen 0 (a non-zero σ around 0 gives U(-σ√3, σ√3)
                # effective spread at gen 0).
                self._preprocess_init_value = 0.0
                self._preprocess_init_per_q_raw = None
            else:
                raise ValueError(
                    f"descriptor_preprocess_init={init_scheme!r} not "
                    f"recognised in SNES tail init.")
            if init_scheme == "sum":
                self._preprocess_init_per_q_raw = None
            self._preprocess_init_scheme = init_scheme
        else:
            self.n_preprocess = 0
            self._preprocess_Q_raw = 0
            self._preprocess_init_value = 1.0
            self._preprocess_init_scheme = "mean"
            self._preprocess_init_per_q_raw = None

        self.dim = self.n_anns_total + self.n_U_pair + self.n_preprocess

        # Search distribution parameters as tf.Variables (stay on GPU).
        # Initialisation scheme is configurable:
        #   "uniform" (GPUMD default): all ANN entries U(-1, 1).
        #   "glorot"                 : W0 / W1 scaled by sqrt(6 / (fan_in + fan_out)),
        #                              biases zero.
        # In both cases the U_pair tail (residual V = U - I parameterisation)
        # stays at zero so the model begins bit-identical to a mixing-
        # disabled baseline. Sigma is uniform throughout — exploration
        # then expands away from this prior.
        rng = np.random.default_rng(self.cfg.seed)
        mu_init = self._build_mu_init(rng)
        self.mu = tf.Variable(mu_init, trainable=False, name="snes_mu")
        sigma_init_vec = np.full(self.dim, float(self.cfg.init_sigma),
                                 dtype=np.float32)
        # Per-block σ for the mixing tail (cfg.mixing_sigma_scale). Leaves
        # the ANN σ untouched; broadens or damps just the V_pair search.
        mix_scale = float(getattr(self.cfg, "mixing_sigma_scale", 1.0))
        if self.n_U_pair > 0 and mix_scale != 1.0:
            mix_start = self.n_anns_total
            sigma_init_vec[mix_start:mix_start + self.n_U_pair] *= mix_scale
        # Per-block σ for the preprocess tail (cfg.preprocess_sigma_scale).
        # SNES per-gen noise on coefficients near 1/L can overwhelm the
        # optimisation signal at the legacy σ; scaling here decouples it.
        preprocess_scale = float(getattr(
            self.cfg, "preprocess_sigma_scale", 1.0))
        if self.n_preprocess > 0 and preprocess_scale != 1.0:
            pre_start = self.n_anns_total + self.n_U_pair
            sigma_init_vec[pre_start:pre_start + self.n_preprocess] *= preprocess_scale
        self.sigma = tf.Variable(sigma_init_vec, trainable=False, name="snes_sigma")

        auto_pop = int(4 + (3 * np.log(self.dim)))
        self.pop_size = self.cfg.pop_size if self.cfg.pop_size is not None else auto_pop

        # Resolve regularization strengths. Sentinel values:
        #   None : auto = sqrt(dim * 1e-6 / num_types)  (GPUMD formula)
        #   -1   : dynamic adaptation (see _maybe_adapt_lambda).
        # Stored as tf.Variable so per-generation updates don't trigger
        # @tf.function retraces in compute_regularization_tf.
        auto_lambda = float(np.sqrt(self.dim * 1e-6 / self.cfg.num_types))
        self._dyn_lambda_1 = (self.cfg.lambda_1 == -1)
        self._dyn_lambda_2 = (self.cfg.lambda_2 == -1)
        init_lambda_1 = (auto_lambda if (self.cfg.lambda_1 is None or self._dyn_lambda_1)
                         else float(self.cfg.lambda_1))
        init_lambda_2 = (auto_lambda if (self.cfg.lambda_2 is None or self._dyn_lambda_2)
                         else float(self.cfg.lambda_2))
        self.lambda_1 = tf.Variable(init_lambda_1, dtype=tf.float32,
                                    trainable=False, name="lambda_1")
        self.lambda_2 = tf.Variable(init_lambda_2, dtype=tf.float32,
                                    trainable=False, name="lambda_2")

        # V_pair regulariser mode: "off" | "cayley" | "expm". Both
        # non-off modes are STRUCTURAL parameterisations (U is
        # reconstructed from a skew-symmetric A; orthogonality is
        # guaranteed regardless of any λ), so the L1/L2 path is silent
        # on V_pair when either is active.
        self._mix_reg_mode = str(getattr(
            self.cfg, "descriptor_mixing_regularizer", "off")).lower()
        if self._mix_reg_mode not in ("off", "cayley", "expm"):
            raise ValueError(
                f"descriptor_mixing_regularizer={self._mix_reg_mode!r} not "
                "recognised (expected 'off', 'cayley', or 'expm')")

        # Polarizability shear weight: scale off-diagonal components [xy, yz, zx]
        # Targets are [xx, yy, zz, xy, yz, zx] — indices 3,4,5 are off-diagonal
        if cfg.target_mode == 2:
            shear_sq = cfg.lambda_shear ** 2
            self._pol_weights = tf.constant(
                [1.0, 1.0, 1.0, shear_sq, shear_sq, shear_sq], dtype=tf.float32)
        else:
            self._pol_weights = None

        # Per-type ranking: build type_of_variable map [dim] -> type index
        # type 0..T-1 for typed params (W0, b0, W1), type T for b1 (global)
        self._per_type = (cfg.per_type_regularization
                          and cfg.toggle_regularization
                          and cfg.num_types > 1)
        if self._per_type:
            self._type_of_variable = self._build_type_of_variable()
            self._type_of_variable_tf = tf.constant(self._type_of_variable, dtype=tf.int32)

        self.eta_sigma = self.cfg.eta_sigma if self.cfg.eta_sigma is not None else self.compute_eta_sigma()
        self.utilities = tf.constant(self.compute_utilities(), dtype=tf.float32)

        # Effective selection mass mu_eff = 1 / Σ w_i² over the POSITIVE
        # recombination weights (CMA convention; Hansen 2016 Eq. 5).
        recomb_w_np = self._recomb_w.numpy()
        self._mu_eff = float(1.0 / np.sum(recomb_w_np ** 2))

    def compute_regularization(self, param_vector: tf.Tensor | np.ndarray
                               ) -> tuple[float, float, float]:
        """Compute L1 and L2 regularisation penalties.

        For multi-element systems, computes per-type regularisation
        (GPUMD NEP4): each atom type's parameters are penalised
        separately using num_vars/num_types as the denominator, then
        averaged across types and added to a global regularisation
        term over all parameters.

        Args:
            param_vector : [dim] tensor or ndarray — flat parameter vector

        Returns:
            l1     : float — L1 penalty on the ANN block
            l2     : float — L2 penalty on the ANN block
            l_orth : float — 0.0 (retained for API compatibility; no
                     orthogonal soft-penalty path remains since the
                     mixing regulariser is now structural cayley/expm
                     or off).
        """
        pv = tf.cast(param_vector, tf.float32)
        T = self.cfg.num_types
        n_per_type = self._n_per_type  # W0_t + b0_t (+ W0_2_t + b0_2_t) + W1_t

        if T > 1:
            # Per-type regularization: average L1/L2 across types + global term
            total_l1 = tf.constant(0.0)
            total_l2 = tf.constant(0.0)

            for t in range(T):
                type_params = self._extract_type_params(pv, t)
                total_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(type_params)) / n_per_type
                total_l2 += self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(type_params)) / n_per_type)

            # Average per-type + global term over typed params only (excludes b1)
            typed = tf.concat([pv[:self.n_typed]], axis=0)
            n_typed_total = self.n_typed
            if self.cfg.target_mode == 2:
                # Second ANN's typed params (skip b1 of primary ANN)
                typed = tf.concat([typed, pv[self.n_primary:self.n_primary + self.n_typed]], axis=0)
                n_typed_total = 2 * self.n_typed
            l1 = total_l1 / T + self.lambda_1 * tf.reduce_sum(tf.abs(typed)) / n_typed_total
            l2 = total_l2 / T + self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(typed)) / n_typed_total)
        else:
            # Single-type path: when V_pair lives behind a cayley / expm
            # parameterisation, keep its A entries out of the main L1/L2
            # sum — shrinking A toward 0 collapses U toward I and defeats
            # the structural map's purpose.
            ann = pv[:self.n_anns_total] if self._mix_cayley else pv
            ann_n = self.n_anns_total if self._mix_cayley else self.dim
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(ann)) / ann_n
            l2 = self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(ann)) / ann_n)

        return float(l1), float(l2), 0.0

    def _build_mu_init(self, rng: np.random.Generator) -> np.ndarray:
        """Initialise the μ vector with a Glorot/Xavier scheme.

        Returns a [dim]-shaped float32 array. The V_pair tail (when
        descriptor mixing is enabled) is always zeroed so U_full = I
        at gen 0.
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.H
        mu = np.zeros(self.dim, dtype=np.float32)
        # Activation-dependent Glorot gain. gain=1 reproduces the
        # legacy tanh init exactly; other activations may want a
        # different scale, but SNES corrects through σ adaptation
        # so the practical impact is small.
        gain = float(getattr(self.model, "_glorot_gain", 1.0))
        c_W0 = gain * float(np.sqrt(6.0 / (Q + H)))
        c_W1 = gain * float(np.sqrt(6.0 / (H + 1)))

        def _fill_ann(off: int) -> int:
            """Fill one ANN's worth of weights starting at ``off``.
            Layout: [W0(T,Q,H) | b0(T,H) | W1(T,H) | b1(1)].
            """
            n_W0 = T * Q * H
            mu[off:off + n_W0] = rng.uniform(
                -c_W0, c_W0, size=n_W0).astype(np.float32)
            off += n_W0
            # b0 zero
            off += T * H
            n_W1 = T * H
            mu[off:off + n_W1] = rng.uniform(
                -c_W1, c_W1, size=n_W1).astype(np.float32)
            off += n_W1
            # b1 zero
            off += 1
            return off

        off = _fill_ann(0)
        if self.cfg.target_mode == 2:
            off = _fill_ann(off)
        # V_pair tail: residual mixing layer initialised at zero so
        # U_full = I at gen 0 (mu starts from the no-mixing baseline).
        if self.n_U_pair > 0:
            mu[self.n_anns_total:self.n_anns_total + self.n_U_pair] = 0.0
        # Preprocess tail: init per cfg.descriptor_preprocess_init
        # ("mean" → 1/L, "sum" → 1.0, "glorot" → 0.0 with σ providing the
        # spread). μ holds ONLY the summed entries (n_preprocess = number
        # of learnable W_pre slots); the per-summed init values are read
        # off the model's W_pre Variable, which TNEP populated at init.
        if self.n_preprocess > 0:
            pre_start = self.n_anns_total + self.n_U_pair
            W_pre_init = self.model.W_pre_angular.numpy().reshape(-1)
            flat_idx = self.model._preprocess_summed_flat_idx.numpy()
            summed_init = W_pre_init[flat_idx]
            if self._preprocess_init_scheme == "glorot":
                limit = 1e-3
                summed_init = rng.uniform(
                    -limit, limit, size=summed_init.size).astype(np.float32)
            mu[pre_start:pre_start + self.n_preprocess] = summed_init
        return mu

    def _maybe_adapt_lambda(self, gen: int, data_loss: float,
                            l1: float, l2: float, l_orth: float = 0.0) -> None:
        """Rescale lambda_1 / lambda_2 toward `target_ratio · data_loss`.

        Activated only for lambdas that were set to -1 in cfg. Runs at
        the same cadence as `compute_regularization` is sampled (every
        `cfg.lambda_adapt_interval` gens, default 100). The update is

            λ ← clip( λ · (target · data_loss / penalty) ^ damping )

        a multiplicative geometric-mean step: damping<1 means each
        update only partly closes the gap, suppressing oscillation
        around the target ratio. With damping=0.2 the response is
        gentle enough that early-training data-loss spikes don't kick
        λ around. The clamp [cfg.lambda_min, cfg.lambda_max] guards
        against pathological divide-by-zero or runaway adaptation.

        Args:
            gen       : current generation (used to honour `_interval`).
            data_loss : reference signal (best train RMSE this gen).
            l1, l2    : current L1 / L2 penalties (from
                        compute_regularization at this gen).
            l_orth    : retained for API compatibility; ignored.
        """
        if not (self._dyn_lambda_1 or self._dyn_lambda_2):
            return
        interval = max(1, int(getattr(self.cfg, "lambda_adapt_interval", 100)))
        if gen % interval != 0:
            return
        target = float(getattr(self.cfg, "lambda_target_ratio", 0.05))
        damping = float(getattr(self.cfg, "lambda_damping", 0.2))
        lmin = float(getattr(self.cfg, "lambda_min", 1e-8))
        lmax = float(getattr(self.cfg, "lambda_max", 1.0))
        # Floor data_loss so a near-zero reference doesn't blow up the
        # ratio. Anything below 1e-8 means the model has essentially
        # converged on the train set; freezing λ at that point is fine.
        ref = max(float(data_loss), 1e-8)
        if self._dyn_lambda_1 and l1 > 1e-12:
            ratio = (target * ref) / float(l1)
            new = float(self.lambda_1.numpy()) * (ratio ** damping)
            self.lambda_1.assign(float(np.clip(new, lmin, lmax)))
        if self._dyn_lambda_2 and l2 > 1e-12:
            ratio = (target * ref) / float(l2)
            new = float(self.lambda_2.numpy()) * (ratio ** damping)
            self.lambda_2.assign(float(np.clip(new, lmin, lmax)))

    def _cayley_blocks_batched(self, A_upper_stacked: tf.Tensor,
                                bs: int) -> tf.Tensor:
        """Batched Cayley reconstruction across N blocks of size bs.

        Replaces N individual `tf.linalg.solve` launches (one per block)
        with a single batched solve — ~10× faster on consumer GPUs
        because LU on tiny (≤8×8) matrices is launch-bound, not
        arithmetic-bound.

        For each entry in the N-batch:
            A[i,j] = +A_upper[k]   for i < j (k = upper-tri index)
            A[j,i] = −A_upper[k]
            A[i,i] = 0             (scatter has no diagonal)
            U      = (I + A) (I − A)⁻¹
            V      = U − I

        Kept in fp32 throughout: the small (≤8×8) blocks SNES uses keep
        ‖A‖ in the regime where fp32 LU is accurate to a few ULP, and a
        few-ULP departure from exact orthogonality is irrelevant for
        training-time U.

        Args:
            A_upper_stacked : [..., N, bs·(bs-1)/2] flat upper-triangle
                              entries. Leading ... is e.g. population [P].
                              N is the number of blocks being reconstructed
                              together (e.g. num_pairs × L for l_aware).
            bs              : block size (uniform across the N blocks).

        Returns:
            V : [..., N, bs, bs] dense V = U − I, fp32.
                Returns zeros for bs ≤ 1 (1×1 skew is identically 0).
        """
        if bs <= 1:
            shape = tf.concat(
                [tf.shape(A_upper_stacked)[:-1], [bs, bs]], axis=0)
            return tf.zeros(shape, dtype=tf.float32)
        # scatter: [bs, bs, payload] from __init__'s cache.
        scatter = self._cayley_scatter_cache[bs]
        # A[..., n, i, j] = Σ_k scatter[i, j, k] · A_upper[..., n, k]
        A = tf.einsum('ijk,...nk->...nij', scatter, A_upper_stacked)
        I = tf.eye(bs, dtype=A.dtype)
        if self._mix_orth_map == "expm":
            # U = exp(A): geodesic on SO(n). tf.linalg.expm handles batched
            # leading dims natively (Padé approx + squaring/scaling). For
            # the bs ≤ 8 blocks SNES uses, cost is a few small matmuls per
            # block — comparable to the Cayley solve but with no singular
            # Jacobian and full SO(n) coverage (Cayley misses −1 eigvals).
            U = tf.linalg.expm(A)
        else:                                        # "cayley"
            U = tf.linalg.solve(I - A, I + A)
        return U - I

    def _extract_type_params(self, pv: tf.Tensor, t: int) -> tf.Tensor:
        """Extract parameters belonging to atom type t from flat vector.

        Parameter layout (single hidden):
            [W0(T,Q,H) | b0(T,H) | W1(T,H_final) | b1(1)]
        With second hidden layer (H2 set):
            [W0(T,Q,H) | b0(T,H) | W0_2(T,H,H2) | b0_2(T,H2) | W1(T,H_final) | b1]

        Args:
            pv : [dim] flat parameter vector
            t  : type index (0 to num_types-1)

        Returns:
            Concatenated type-t parameters. Length:
              single hidden  : Q*H + H + H_final
              two hidden     : Q*H + H + H*H2 + H2 + H_final
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.H

        # W0 block: [T, Q, H]
        w0_start = t * Q * H
        w0_end = w0_start + Q * H

        # b0 block: after all W0
        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H

        # W1 block: after all b0
        w1_offset = b0_offset + T * H
        w1_start = w1_offset + t * H
        w1_end = w1_start + H

        return tf.concat([
            pv[w0_start:w0_end],
            pv[b0_start:b0_end],
            pv[w1_start:w1_end],
        ], axis=0)

    def _build_type_of_variable(self) -> np.ndarray:
        """Build array mapping each parameter index to its atom type.

        Layout per ANN:
            single-hidden: [W0(T,Q,H) | b0(T,H) | W1(T,H_final) | b1(1)]
            two-hidden   : [W0(T,Q,H) | b0(T,H) | W0_2(T,H,H2) | b0_2(T,H2)
                            | W1(T,H_final) | b1(1)]
        Per-type params get label t. b1 gets label T (global).

        Returns:
            [dim] int array — type label per variable
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.H

        def _ann_types() -> np.ndarray:
            tov = np.empty(self.n_primary, dtype=np.int32)
            offset = 0
            # W0: [T, Q, H] — type t owns contiguous block of Q*H
            for t in range(T):
                tov[offset:offset + Q * H] = t
                offset += Q * H
            # b0: [T, H]
            for t in range(T):
                tov[offset:offset + H] = t
                offset += H
            # W1: [T, H]
            for t in range(T):
                tov[offset:offset + H] = t
                offset += H
            # b1: global (label = T)
            tov[offset] = T
            return tov

        if self.cfg.target_mode == 2:
            ann_tov = np.concatenate([_ann_types(), _ann_types()])
        else:
            ann_tov = _ann_types()

        # U_pair entries route into the per-type ranking system:
        #   - Shared U_pair: all entries get the global label (T),
        #     same ranking signal as the b1 bias. The mixing is
        #     shared across central types so a per-type ranking
        #     wouldn't make physical sense.
        #   - Per-type U_pair: each T-slab gets the corresponding
        #     central type's label (0..T-1). The t-th slab is only
        #     applied to atoms of type t in the forward pass, so it
        #     should be driven by the same fitness signal as that
        #     type's W0/b0/W1 (i.e. structures containing type t).
        # Per-pair rankings (one fitness column per species pair)
        # remain a Tier-2 extension that would require
        # evaluate_population to emit pair-specific RMSE columns on
        # top of the per-type columns.
        tail_labels_parts = []
        # U_pair tail: single layer. With per-type V_pair, T contiguous
        # slabs (one per central type); with shared V_pair, the whole
        # tail routes to the global label T.
        if self.n_U_pair > 0:
            if self._mix_per_type:
                per_T = self.n_U_pair // T
                u_labels = np.empty(self.n_U_pair, dtype=np.int32)
                for t_idx in range(T):
                    u_labels[t_idx * per_T:(t_idx + 1) * per_T] = t_idx
            else:
                u_labels = np.full(self.n_U_pair, T, dtype=np.int32)
            tail_labels_parts.append(u_labels)
        # Preprocess tail: W_pre[t, q_raw] stored t-major in the mu vector
        # → blocks of Q_raw entries belong to a single type.
        if self.n_preprocess > 0:
            # μ holds only summed W_pre entries. Derive per-entry type
            # labels from the flat indices into the W_pre tensor.
            #   per_type=True : W_pre shape [T, Q_raw], t-major flat
            #                   layout → t = flat_idx // Q_raw.
            #   per_type=False: W_pre shape [Q_raw] (no T axis); use
            #                   global label T (model-wide).
            if self._preprocess_per_type:
                flat_idx_np = self.model._preprocess_summed_flat_idx.numpy()
                pre_labels = (flat_idx_np // int(self._preprocess_Q_raw)).astype(np.int32)
            else:
                pre_labels = np.full(self.n_preprocess, T, dtype=np.int32)
            tail_labels_parts.append(pre_labels)
        if tail_labels_parts:
            tail_labels = np.concatenate(tail_labels_parts)
            return np.concatenate([ann_tov, tail_labels])
        return ann_tov

    def _build_per_type_gradients(
        self,
        s: tf.Tensor,
        fitness_per_type_rmse: tf.Tensor,
        samples: tf.Tensor,
    ) -> tf.Tensor:
        """Build composite noise matrix with per-type rankings.

        For each variable v, permutes the P noise vectors according to the
        ranking of type_of_variable[v]'s fitness. The result can be passed
        directly to update() with the standard utilities.

        Args:
            s                    : [P, dim] noise vectors from ask()
            fitness_per_type_rmse: [P, T+1] per-type RMSE (type 0..T-1 from
                                   structures containing that type, type T = global)
            samples              : [P, dim] candidate parameter vectors

        Returns:
            s_sorted : [P, dim] composite noise — each column permuted by
                       its type's ranking
        """
        T = self.cfg.num_types
        n_per_type = self._n_per_type

        # Add per-type regularization to per-type RMSE → [T+1] fitness values
        fitness_per_type = []
        for t in range(T):
            type_params = self._extract_type_params_batched(samples, t)  # [P, n_per_type]
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(type_params), axis=1) / n_per_type
            l2 = self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(type_params), axis=1) / n_per_type)
            fitness_per_type.append(fitness_per_type_rmse[:, t] + l1 + l2)

        # Global ranking (type T): regularize over all typed params, excluding b1
        typed = samples[:, :self.n_typed]
        n_typed_total = self.n_typed
        if self.cfg.target_mode == 2:
            typed = tf.concat([typed, samples[:, self.n_primary:self.n_primary + self.n_typed]], axis=1)
            n_typed_total = 2 * self.n_typed
        global_l1 = self.lambda_1 * tf.reduce_sum(tf.abs(typed), axis=1) / n_typed_total
        global_l2 = self.lambda_2 * tf.sqrt(
            tf.reduce_sum(tf.square(typed), axis=1) / n_typed_total)
        fitness_per_type.append(fitness_per_type_rmse[:, T] + global_l1 + global_l2)

        # Sort each type's fitness independently → [T+1, P] rank indices
        # ranks_per_type[t] = indices that sort type t's fitness ascending
        ranks_per_type = [tf.argsort(f) for f in fitness_per_type]  # list of [P]

        # Build composite s_sorted: for each variable, permute by its type's ranking
        # Gather all T+1 sorted versions of s, then select per-variable
        # s_sorted_all[t] = s permuted by type t's ranking: [P, dim]
        s_sorted_all = tf.stack([tf.gather(s, r) for r in ranks_per_type])  # [T+1, P, dim]

        # type_of_variable[v] tells us which row of s_sorted_all to use for column v
        tov = self._type_of_variable_tf  # [dim] — cached tf.constant
        # We want s_sorted[:, v] = s_sorted_all[tov[v], :, v]
        # Transpose to [dim, T+1, P] so we can index [v, tov[v]] -> [P]
        s_by_var = tf.transpose(s_sorted_all, [2, 0, 1])  # [dim, T+1, P]
        v_indices = tf.range(self.dim)  # [dim]
        gather_idx = tf.stack([v_indices, tov], axis=1)  # [dim, 2] — [v, tov[v]]
        s_selected = tf.gather_nd(s_by_var, gather_idx)  # [dim, P]
        s_sorted = tf.transpose(s_selected)  # [P, dim]

        return s_sorted

    def compute_eta_sigma(self) -> float:
        """Compute the sigma learning rate from per-type parameter dimensionality.

        GPUMD (version != 3) divides by num_types for type-specific ANNs,
        giving a larger step size that accounts for per-type independence.

        Returns:
            eta_sigma : float — controls how fast sigma adapts.
                  eta_sigma = (3 + ln(num)) / (5 * sqrt(num)) / 2
                  where num = dim / num_types
        """
        num = float(self.dim) / self.cfg.num_types
        num = max(num, 1.0)
        eta_sigma = ((3.0 + np.log(num)) / (5.0 * np.sqrt(num))) / 2.0
        return float(eta_sigma)

    def compute_utilities(self) -> np.ndarray:
        """Precompute rank-based utility weights for the population.

        Vanilla Hansen log-rank weights:
          - Top half: positive log-rank weights summing to 1.
          - Bottom half: weight = −1/λ (uniform negative).
          - Zero-mean by construction.

        Also caches ``self._recomb_w`` (positive Hansen weights only,
        sum 1) for any downstream consumer that needs the un-shifted
        weights — μ_eff is computed from this.

        Returns:
            utilities : ndarray [pop_size] — weights indexed by rank (0 = best).
        """
        lam = self.pop_size
        ranks = np.arange(lam) + 1
        raw = np.log((lam * 0.5) + 1.0) - np.log(ranks)
        raw_positive = np.maximum(0.0, raw)
        total_pos = np.sum(raw_positive)
        if total_pos > 0:
            recomb_w = raw_positive / total_pos
        else:
            if self.cfg.debug:
                print("Utility calc failed due to negative total")
            recomb_w = raw_positive
        self._recomb_w = tf.constant(recomb_w.astype(np.float32))

        utilities = recomb_w - 1.0 / lam
        if self.cfg.debug:
            print("utilities = ", utilities)
        return utilities

    def _mirrored_normal(self, shape: tuple[int, int]) -> tf.Tensor:
        """Draw mirrored (antithetic) standard-normal noise.

        `shape = (P, width)`: draws P//2 independent rows ε_i, returns
        them stacked with their mirrors −ε_i, plus one standalone i.i.d.
        row when P is odd so the total leading-dim count is preserved.

        The per-pair antithesis is what cancels the first-order Taylor
        noise in the natural-gradient mean step under Hansen's zero-mean
        utility shaping (Salimans et al. 2017; Brockhoff et al. 2010).
        """
        P, width = int(shape[0]), int(shape[1])
        half = P // 2
        s_half = self.tf_rng.normal(shape=(half, width))
        if P % 2 == 0:
            return tf.concat([s_half, -s_half], axis=0)
        s_extra = self.tf_rng.normal(shape=(1, width))
        return tf.concat([s_half, -s_half, s_extra], axis=0)

    def ask(self) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Sample pop_size candidate parameter vectors via mirrored
        (antithetic) sampling — see `_mirrored_normal`.

        Vanilla separable-NES sampling: ``delta = sigma * s_iso`` per
        coordinate, ``samples = μ + delta``.

        Returns:
            samples : [pop_size, dim] float32 — candidate parameter vectors
            aux     : dict with
                "s_iso" : [pop_size, dim] standard-normal isotropic noise
                "delta" : [pop_size, dim] actual displacement (= sigma·s_iso)
        """
        s_iso = self._mirrored_normal((self.pop_size, self.dim))
        delta = s_iso * self.sigma
        samples = self.mu + delta
        return samples, {"s_iso": s_iso, "delta": delta}

    def update(self, utilities: tf.Tensor, aux: dict[str, tf.Tensor]) -> None:
        """Vanilla separable-NES mean / per-coord sigma update.

        Args:
            utilities : [pop_size] float32 — log-rank weights (best first)
            aux       : dict from ask(), already sorted by fitness:
                "s_iso" : [pop_size, dim] isotropic standard-normal noise
                "delta" : [pop_size, dim] actual displacement (= sigma·s_iso)

        Mutates ``self.mu`` and ``self.sigma`` in place. Sigma is clamped
        to ``cfg.sigma_floor`` after the multiplicative update so a
        near-collapse can't silently kill exploration on long runs.
        """
        s_iso = aux["s_iso"]
        grad_mu = tf.einsum('p,pd->d', utilities, s_iso)
        self.mu.assign_add(self.sigma * grad_mu)
        # NES natural gradient on log-sigma: Σ_p u_p · (s² − 1). This signal
        # is self-equilibrating (too-small σ → best samples are larger-step
        # ones → grad_sigma > 0 → grow; too-large → shrink), which is what
        # makes the multiplicative update stable on long runs.
        grad_sigma = tf.einsum('p,pd->d', utilities, s_iso ** 2 - 1.0)
        new_sigma = self.sigma * tf.exp(self.eta_sigma * grad_sigma)
        floor = getattr(self.cfg, "sigma_floor", 1e-5)
        if floor is not None and floor > 0.0:
            new_sigma = tf.maximum(new_sigma, float(floor))
        self.sigma.assign(new_sigma)

    def fit(self, train_data: dict[str, tf.Tensor], val_data: dict[str, tf.Tensor], plot_callback: Callable | None = None, resume_state: dict | None = None) -> dict:
        """Run the SNES training loop using GPU-batched population evaluation.

        All sampling, ranking, and update operations run on GPU.
        Only scalar metrics are transferred to CPU for history/reporting.

        Args:
            train_data    : dict with padded tensors from pad_and_stack()
            val_data      : same structure
            plot_callback : optional callable(history, gen) — called every
                            cfg.plot_interval generations for periodic plotting

        Returns:
            history : dict with training metrics per generation
        """
        print("Fitting model...")
        cfg = self.cfg
        S_train = train_data["descriptors"].shape[0]

        if resume_state is not None:
            history = resume_state["history"]
            # Make sure the timing sub-dict exists with all expected keys
            # (older checkpoints might not have it).
            history.setdefault("timing", {})
            for k in ("sample_batch", "evaluate", "rank_update",
                      "validate", "overhead"):
                history["timing"].setdefault(k, [])
            self.mu.assign(resume_state["mu"])
            self.sigma.assign(resume_state["sigma"])
            best_mu = tf.constant(resume_state["best_mu"], dtype=tf.float32)
            # `best_sigma` may be absent in pre-2026 checkpoints; fall back
            # to the current self.sigma in that case.
            bsi = resume_state.get("best_sigma")
            best_sigma = (tf.constant(bsi, dtype=tf.float32)
                          if bsi is not None else tf.identity(self.sigma))
            best_val_loss = float(resume_state["best_val_loss"])
            gens_without_improvement = int(resume_state["gens_without_improvement"])
            rng = resume_state.get("rng_state")
            if rng is not None:
                try:
                    self.tf_rng.state.assign(np.asarray(rng))
                except Exception:
                    # Generator-state shape can shift across TF versions;
                    # fall back to keeping the freshly-seeded generator
                    # rather than aborting the resume.
                    pass
            start_gen = int(resume_state["last_gen"]) + 1
            # Offset train_start so the displayed elapsed continues from
            # the checkpointed wall-time rather than restarting at zero.
            prior_elapsed = float(sum(
                sum(history["timing"][k])
                for k in history["timing"]))
            train_start = time.perf_counter() - prior_elapsed
            print(f"  resuming from gen {start_gen} "
                  f"(best_val={best_val_loss:.6f}, "
                  f"history points={len(history['generation'])}, "
                  f"prior elapsed={prior_elapsed:.1f}s)")
        else:
            history = {
                "generation": [],
                "train_loss": [],
                "train_rmse": [],
                "val_loss": [],
                "L1": [],
                "L2": [],
                "best_rmse": [],
                "worst_rmse": [],
                "sigma_min": [],
                "sigma_max": [],
                "sigma_mean": [],
                "sigma_median": [],
                "timing": {
                    "sample_batch": [],
                    "evaluate": [],
                    "rank_update": [],
                    "validate": [],
                    "overhead": [],
                },
            }
            best_val_loss = float('inf')
            best_mu = tf.identity(self.mu)
            best_sigma = tf.identity(self.sigma)
            gens_without_improvement = 0
            start_gen = 0
            train_start = time.perf_counter()

        gen_l1, gen_l2, gen_lorth = 0.0, 0.0, 0.0
        val_fitness = float('inf')
        # True RMSE of the mean μ on train data — the SAME estimator as
        # val_fitness (validate at μ), so train vs val is an apples-to-apples
        # comparison. (The progress bar previously showed avg_fitness here,
        # which is the mean population *fitness* over the sampled candidates
        # rather than the at-μ estimate.)
        train_rmse = float('inf')
        sigma_min = sigma_max = sigma_mean = sigma_median = float(cfg.init_sigma)
        # Last finite RRMSE values, carried into Adam gens (which don't compute
        # a population RRMSE) so the history series stays finite for plotting.
        last_best_rrmse = last_avg_rrmse = 0.0

        # SIGTERM handler: Slurm sends SIGTERM `--time-min` seconds (default
        # 30 s on Mahti) before walltime hits. We flip a flag rather than
        # exiting from the handler so the in-flight generation completes
        # cleanly, then save a final checkpoint at the bottom of the loop
        # and break out. Hooks in only when there's a save_path configured
        # (otherwise nowhere to checkpoint to). The handler is restored at
        # fit() exit so repeated fit() calls don't accumulate handlers.
        import signal as _signal
        _term_requested = {"flag": False}
        _prev_term_handler = None
        if cfg.save_path:
            def _on_term(_sig, _frm):
                _term_requested["flag"] = True
            try:
                _prev_term_handler = _signal.signal(
                    _signal.SIGTERM, _on_term)
            except (ValueError, OSError):
                # signal.signal only works in the main thread; skip the
                # hook in subthreads (e.g. notebook environments).
                _prev_term_handler = None

        for gen in range(start_gen, cfg.num_generations):
            t0 = time.perf_counter()

            # Select batch: None = full train set, int = random subset.
            if cfg.batch_size is None:
                batch_data = train_data
            else:
                batch_idx_tf = tf.argsort(
                    self.tf_rng.uniform(shape=[S_train]))[:cfg.batch_size]
                struct_keys = ["descriptors", "positions", "Z_int", "boxes",
                               "num_atoms", "targets", "atom_mask"]
                if "types_contained" in train_data:
                    struct_keys.append("types_contained")
                batch_data = {
                    key: tf.gather(train_data[key], batch_idx_tf)
                    for key in struct_keys
                }
                # COO pair gather: select pairs belonging to the sampled
                # structures.
                pair_starts = tf.gather(train_data["struct_ptr"], batch_idx_tf)
                pair_ends   = tf.gather(train_data["struct_ptr"], batch_idx_tf + 1)
                pair_ranges = tf.ragged.range(pair_starts, pair_ends)
                flat_pair_idx = tf.cast(pair_ranges.flat_values, tf.int32)
                gv_full = train_data["grad_values"]
                batch_data["grad_values"] = tf.gather(gv_full, flat_pair_idx)
                batch_data["pair_atom"]   = tf.gather(train_data["pair_atom"],   flat_pair_idx)
                batch_data["pair_gidx"]   = tf.gather(train_data["pair_gidx"],   flat_pair_idx)
                batch_data["pair_struct"] = tf.cast(pair_ranges.value_rowids(), tf.int32)
                # Build batch-local struct_ptr for struct_chunk slicing
                batch_pair_counts = tf.cast(pair_ranges.row_lengths(), tf.int32)
                batch_data["struct_ptr"] = tf.concat(
                    [[0], tf.cumsum(batch_pair_counts)], axis=0)

            t1 = time.perf_counter()

            samples, aux = self.ask()

            # Evaluate entire population on GPU
            if self._per_type:
                # Per-type mode: get per-type RMSE [P, T+1], then build composite gradients
                fitness_per_type_rmse = self.evaluate_population(
                    samples, batch_data, return_per_type=True)
                fitness = fitness_per_type_rmse[:, -1]  # global RMSE for reporting
            else:
                fitness = self.evaluate_population(samples, batch_data)

            # GPU→CPU sync. Stack the reductions and pull them in one
            # transfer — five separate `float(reduce_*)` calls would
            # issue five independent device syncs every gen. fitness
            # drives SNES ranking; the rmse/rrmse entries are computed
            # from squared error so they're comparable across runs.
            rmse_pc = self._last_rmse_per_cand
            rrmse_pc = self._last_rrmse_per_cand
            # σ stats are computed on the GPU and folded into the same
            # batched pull as the fitness/RMSE metrics — one device sync
            # per gen, not the previous every-100-gen sampling. Median
            # uses tf.sort which is O(d log d); at d ≈ 50k this is
            # microseconds on GPU.
            sigma_active = self.sigma
            sigma_sorted = tf.sort(sigma_active)
            sigma_med_tf = 0.5 * (
                sigma_sorted[(self.dim - 1) // 2]
                + sigma_sorted[self.dim // 2])
            metrics_gpu = tf.stack([
                tf.reduce_mean(fitness),
                tf.reduce_min(rmse_pc),
                tf.reduce_max(rmse_pc),
                tf.reduce_min(rrmse_pc),
                tf.reduce_mean(rrmse_pc),
                tf.reduce_min(sigma_active),
                tf.reduce_max(sigma_active),
                tf.reduce_mean(sigma_active),
                sigma_med_tf,
            ])
            metrics_np = metrics_gpu.numpy()
            avg_fitness = float(metrics_np[0])
            best_rmse = float(metrics_np[1])
            worst_rmse = float(metrics_np[2])
            best_rrmse = float(metrics_np[3])
            avg_rrmse = float(metrics_np[4])
            sigma_min = float(metrics_np[5])
            sigma_max = float(metrics_np[6])
            sigma_mean = float(metrics_np[7])
            sigma_median = float(metrics_np[8])
            last_best_rrmse, last_avg_rrmse = best_rrmse, avg_rrmse

            t2 = time.perf_counter()

            # Regularisation: matches GPUMD's behaviour. The per-candidate
            # L1+L2 penalty is already computed every gen, in-graph, on
            # the GPU as part of fitness (compute_regularization_tf, called
            # from evaluate_population) — that's the training signal.
            #
            # This block is the *out-of-loop reporting / adaptation* pull,
            # which forces a GPU→CPU sync. GPUMD doesn't pull eager values
            # mid-training at all. We do it every 100 gens for the progress
            # bar / history, and additionally at the user-configured
            # `lambda_adapt_interval` cadence when any λ is in dynamic mode.
            # Between samples the values carry over (history at val gens
            # reads the most recent 100-gen sample).
            need_adapt = (self._dyn_lambda_1 or self._dyn_lambda_2)
            adapt_cadence = max(
                1, int(getattr(cfg, "lambda_adapt_interval", 100)))
            do_adapt_now = need_adapt and (gen % adapt_cadence == 0)
            do_report_now = (gen % 100 == 0)
            if cfg.toggle_regularization and (do_adapt_now or do_report_now):
                gen_l1, gen_l2, gen_lorth = self.compute_regularization(self.mu)
                if do_adapt_now:
                    # Uses best-RMSE in the current population as the
                    # data-loss reference (the mean is dominated by the
                    # worst candidates early in training).
                    self._maybe_adapt_lambda(gen, best_rmse,
                                             gen_l1, gen_l2, gen_lorth)
            elif not cfg.toggle_regularization:
                gen_l1, gen_l2, gen_lorth = 0, 0, 0

            # Rank and update (GPU).
            # Rank BOTH the isotropic noise and the actual displacement
            # by the same per-candidate fitness ordering, then hand the
            # sorted pair to update(). The mean step consumes `delta`
            # (covariance-agnostic) and the sigma step consumes `s_iso`
            # (isotropic component only) — see ask()/update() docs.
            if self._per_type:
                s_iso_sorted = self._build_per_type_gradients(
                    aux["s_iso"], fitness_per_type_rmse, samples)
                delta_sorted = self._build_per_type_gradients(
                    aux["delta"], fitness_per_type_rmse, samples)
            else:
                ranks = tf.argsort(fitness)
                s_iso_sorted = tf.gather(aux["s_iso"], ranks)
                delta_sorted = tf.gather(aux["delta"], ranks)
            aux2 = {"s_iso": s_iso_sorted, "delta": delta_sorted,
                    "fitness": fitness}
            self.update(self.utilities, aux2)

            t3 = time.perf_counter()

            # Validate with updated mean (skip on non-val generations).
            # Clear the intra-gen fold cache every gen so any out-of-loop
            # validate() call (early-stop fallback, etc.) never sees stale
            # cached tensors from a previous gen. `self.mu` Variable
            # identity is preserved across `assign()`, so the id-based
            # cache key alone can't distinguish gens — the per-gen reset
            # is the safety net.
            self._validate_fold_cache = None
            _do_val = (gen % cfg.val_interval == 0) or (gen == cfg.num_generations - 1)
            if _do_val:
                val_fitness = self.validate(val_data, self.mu)
                # Train RMSE at μ, computed identically to val_fitness so the
                # two are directly comparable. Costs
                # one extra forward pass at μ (~1/pop_size of the population
                # eval), only at val ticks. The second validate() reuses
                # the fold from the first via _validate_fold_cache.
                train_rmse = self.validate(train_data, self.mu)

            t4 = time.perf_counter()

            # σ stats are computed every gen as part of the metrics_gpu
            # pull above (zero extra device sync).

            # History is recorded once per val_interval (plus the final
            # generation). Off-val gens contribute only to the progress
            # bar and the early-stopping counter.
            if _do_val:
                history["generation"].append(gen)
                history["train_loss"].append(avg_fitness)   # optimised objective (MSE-based RMSE)
                history.setdefault("train_rmse", []).append(train_rmse)  # at-μ RMSE, comparable to val
                history["val_loss"].append(val_fitness)
                history["L1"].append(gen_l1)
                history["L2"].append(gen_l2)
                history.setdefault("L_orth", []).append(gen_lorth)
                history["best_rmse"].append(best_rmse)
                history["worst_rmse"].append(worst_rmse)
                # RMSE / RRMSE always reported.
                history.setdefault("best_rrmse", []).append(best_rrmse)
                history.setdefault("avg_rrmse", []).append(avg_rrmse)
                history["sigma_min"].append(sigma_min)
                history["sigma_max"].append(sigma_max)
                history["sigma_mean"].append(sigma_mean)
                history["sigma_median"].append(sigma_median)

            # Progress bar
            frac = (gen + 1) / cfg.num_generations
            bar_len = 30
            filled = int(bar_len * frac)
            bar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.perf_counter() - train_start
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            elapsed_str = _format_duration(elapsed)
            eta_str = _format_duration(eta)
            line = (f"\r{bar} {gen + 1}/{cfg.num_generations} "
                    f"train RMSE: {train_rmse:.6f}  "
                    f"val RMSE: {val_fitness:.6f}  "
                    f"best val RMSE: {best_val_loss:.6f}  "
                    f"elapsed: {elapsed_str}  ETA: {eta_str}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
                line += f"  train obj(mse): {avg_fitness:.6f}"
                line += f"  best_RMSE: {best_rmse:.6f}  best_RRMSE: {best_rrmse:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # Early stopping (only update on val generations)
            if _do_val:
                # global best drives best_mu / early stop
                if val_fitness < best_val_loss:
                    best_val_loss = val_fitness
                    best_mu = tf.identity(self.mu)
                    best_sigma = tf.identity(self.sigma)
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1

            if cfg.patience is not None and gens_without_improvement >= cfg.patience:
                print(f"\nEarly stopping at generation {gen + 1} "
                      f"(no improvement for {cfg.patience} generations)")
                # Force a final validation if this generation wasn't a val generation,
                # so best_val_loss is never left at its initial inf sentinel.
                if not _do_val:
                    val_fitness = self.validate(val_data, self.mu)
                    if val_fitness < best_val_loss:
                        best_val_loss = val_fitness
                        best_mu = tf.identity(self.mu)
                        best_sigma = tf.identity(self.sigma)
                # IMPORTANT: do NOT overwrite self.mu/sigma with best
                # here. The post-loop code (after the for-loop) needs
                # the genuine final-gen self.mu to build `final_model`
                # distinct from `best_val_model`. Restoration to best
                # happens AFTER both models are constructed, at the
                # end of fit(). (Older code overwrote here, which made
                # final_model == best_val_model whenever early-stop
                # triggered.)
                break

            t5 = time.perf_counter()

            if _do_val:
                history["timing"]["sample_batch"].append(t1 - t0)
                history["timing"]["evaluate"].append(t2 - t1)
                history["timing"]["rank_update"].append(t3 - t2)
                history["timing"]["validate"].append(t4 - t3)
                history["timing"]["overhead"].append(t5 - t4)

            # Periodic plotting callback
            if gen + 1 < cfg.num_generations:
                if (plot_callback is not None
                        and cfg.plot_interval is not None
                        and (gen + 1) % cfg.plot_interval == 0):
                    # Temporarily restore best params for score()
                    params = self.reconstruct_params_tf(best_mu)
                    _set_model_params(self.model, *params)
                    plot_callback(history, gen + 1)
                    # Restore current mu back into model (training continues)
                    params = self.reconstruct_params_tf(self.mu)
                    _set_model_params(self.model, *params)

            # Periodic checkpoint save (rolling — overwrites previous).
            # cfg.save_path follows the {run_dir}/auto convention used
            # by setup_run_directory, so the checkpoint goes in the
            # parent (run) dir alongside the eventual .h5 model files.
            ci = getattr(cfg, "checkpoint_interval", None)
            if (ci is not None and ci > 0 and cfg.save_path
                    and (gen + 1) % ci == 0
                    and gen + 1 < cfg.num_generations):
                from model_io import save_checkpoint
                run_dir = os.path.dirname(cfg.save_path) or "."
                os.makedirs(run_dir, exist_ok=True)
                ckpt_path = os.path.join(run_dir, "checkpoint.h5")
                ckpt_state = {
                    "mu": self.mu, "sigma": self.sigma,
                    "best_mu": best_mu,
                    "best_val_loss": best_val_loss,
                    "gens_without_improvement": gens_without_improvement,
                    "tf_rng_state": self.tf_rng.state,
                }
                ckpt_state["best_sigma"] = best_sigma
                save_checkpoint(ckpt_path, cfg, ckpt_state, history, gen)
                # Print a one-line note above the in-place progress bar.
                sys.stdout.write(
                    f"\n  checkpoint saved at gen {gen + 1} → {ckpt_path}\n")
                sys.stdout.flush()

            # SIGTERM received (Slurm walltime imminent): flush a final
            # checkpoint regardless of cfg.checkpoint_interval cadence
            # and exit the loop cleanly so the model wrap-up still runs.
            if _term_requested["flag"] and cfg.save_path:
                from model_io import save_checkpoint
                run_dir = os.path.dirname(cfg.save_path) or "."
                os.makedirs(run_dir, exist_ok=True)
                ckpt_path = os.path.join(run_dir, "checkpoint.h5")
                ckpt_state = {
                    "mu": self.mu, "sigma": self.sigma,
                    "best_mu": best_mu, "best_sigma": best_sigma,
                    "best_val_loss": best_val_loss,
                    "gens_without_improvement": gens_without_improvement,
                    "tf_rng_state": self.tf_rng.state,
                }
                save_checkpoint(ckpt_path, cfg, ckpt_state, history, gen)
                sys.stdout.write(
                    f"\n  SIGTERM received — flushed checkpoint at gen "
                    f"{gen + 1} → {ckpt_path}\n")
                sys.stdout.flush()
                break

        # Restore prior SIGTERM handler so a subsequent fit() call (e.g.
        # in a notebook session) starts clean.
        if _prev_term_handler is not None:
            try:
                _signal.signal(_signal.SIGTERM, _prev_term_handler)
            except (ValueError, OSError):
                pass

        print()  # newline after progress bar

        # Build final-gen model (current mu, before restoring best)
        from TNEP import TNEP
        final_model = TNEP(self.cfg)
        final_params = self.reconstruct_params_tf(self.mu)
        _set_model_params(final_model, *final_params)

        # Build best-val model
        best_val_model = TNEP(self.cfg)
        best_val_params = self.reconstruct_params_tf(best_mu)
        _set_model_params(best_val_model, *best_val_params)

        # Restore best into self.model for backward compatibility
        self.mu.assign(best_mu)
        self.sigma.assign(best_sigma)
        _set_model_params(self.model, *best_val_params)

        return history, final_model, best_val_model

    def validate(self, val_data: dict[str, tf.Tensor], mu_tf: tf.Tensor | None = None) -> float:
        """Compute mean RMSE on a subset of validation structures using batched predict.

        Args:
            val_data : dict with padded tensors from pad_and_stack()
            mu_tf    : optional [dim] float32 tensor/Variable — parameter vector.
                       If provided, weights are reconstructed on GPU.
                       If None, uses current model weights.

        Returns:
            fitness : float — mean RMSE
        """
        S_val = val_data["num_atoms"].shape[0]

        # Resolve subsample indices once. None = full val set.
        if self.cfg.val_size is None:
            val_idx_tf = tf.range(S_val, dtype=tf.int32)
        else:
            val_idx_tf = tf.cast(
                tf.argsort(self.tf_rng.uniform(shape=[S_val]))[:self.cfg.val_size],
                tf.int32)

        # Intra-generation cache key: same μ used twice (val + train) at a
        # val gen should not re-fold. Cache holds the post-fold W0/W0p
        # (already absorbed via _W0_eff + _W0_preprocess_eff) keyed by the
        # μ tensor's object identity. fit() invalidates by clearing the
        # cache at the start of each generation.
        _cache = getattr(self, "_validate_fold_cache", None)
        _cache_hit = (mu_tf is not None
                      and _cache is not None
                      and _cache.get("_key") == id(mu_tf))

        if mu_tf is not None and not _cache_hit:
            params = self.reconstruct_params_tf(mu_tf)
            named = self._split_reconstructed(params)
            W0 = named.get("W0"); b0 = named.get("b0")
            W1 = named.get("W1"); b1 = named.get("b1")
            W0p = named.get("W0_pol")
            b0p = named.get("b0_pol")
            W1p = named.get("W1_pol")
            b1p = named.get("b1_pol")
            U_pair_val = named.get("U_pair")
            W_pre_angular_val = named.get("W_pre_angular")
            # Absorb U_pair^T into W0 (and W0_pol).
            if U_pair_val is not None:
                W0 = self.model._W0_eff(W0, U_pair_val)
                if W0p is not None:
                    W0p = self.model._W0_eff(W0p, U_pair_val)
            # Preprocessing fold (composes with mixing under l_aware).
            if (getattr(self.model, "descriptor_preprocess_contract", "off")
                    != "off"):
                W0 = self.model._W0_preprocess_eff(
                    W0, W_pre_override=W_pre_angular_val)
                if W0p is not None:
                    W0p = self.model._W0_preprocess_eff(
                        W0p, W_pre_override=W_pre_angular_val)
            # Stash the folded tensors for the next validate() call this gen.
            self._validate_fold_cache = {
                "_key": id(mu_tf),
                "fold": (W0, b0, W1, b1, W0p, b0p, W1p, b1p),
            }
        elif _cache_hit:
            (W0, b0, W1, b1, W0p, b0p, W1p, b1p) = _cache["fold"]
        else:
            W0, b0, W1, b1 = self.model.W0, self.model.b0, self.model.W1, self.model.b1
            W0p = getattr(self.model, 'W0_pol', None)
            b0p = getattr(self.model, 'b0_pol', None)
            W1p = getattr(self.model, 'W1_pol', None)
            b1p = getattr(self.model, 'b1_pol', None)
            if getattr(self.model, "descriptor_mixing", False):
                W0 = self.model._W0_eff(W0)
                if W0p is not None:
                    W0p = self.model._W0_eff(W0p)
            if (getattr(self.model, "descriptor_preprocess_contract", "off")
                    != "off"):
                W0 = self.model._W0_preprocess_eff(W0)
                if W0p is not None:
                    W0p = self.model._W0_preprocess_eff(W0p)

        # Streaming chunk loop honours batch_chunk_size so tensor
        # materialisation stays bounded. Full-val (val_size=None) takes the
        # contiguous-range path via `prefetched_chunks`; a random val subset
        # falls back to the per-call slice path.
        from data import slice_and_complete_chunk, prefetched_chunks
        N_idx = int(val_idx_tf.shape[0])
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else N_idx

        diff_sq_sum = tf.constant(0.0, dtype=tf.float32)
        diff_count  = tf.constant(0.0, dtype=tf.float32)

        def _consume(chunk, chunk_idx=0):
            nonlocal diff_sq_sum, diff_count
            # Pre-compute W_atom once per (val_data, chunk_idx) and
            # reuse across generations when val_size is None (full
            # set, static chunks). When val_size is set, chunks are
            # random subsets per gen — bypass the cache. Two-level
            # dict: outer keyed by `id(val_data)`, inner by chunk_idx.
            # This lets val_data and train_data each hold their own
            # per-chunk W_atom across gens.
            W_atom_v = None
            if self.cfg.target_mode == 1:
                _can_cache = (self.cfg.val_size is None)
                if _can_cache:
                    _vd_id = id(val_data)
                    _cache_top = getattr(
                        self, "_W_atom_validate_cache", None)
                    if _cache_top is None:
                        self._W_atom_validate_cache = {}
                        _cache_top = self._W_atom_validate_cache
                    _sub = _cache_top.setdefault(_vd_id, {})
                    W_atom_v = _sub.get(chunk_idx)
                if W_atom_v is None:
                    B_arg = chunk["descriptors"].shape[0]
                    A_arg = chunk["descriptors"].shape[1]
                    W_atom_v = self.model._precompute_dipole_kernel(
                        chunk["grad_values"], chunk["pair_struct"],
                        chunk["pair_atom"], chunk["pair_gidx"],
                        chunk["positions"], chunk["boxes"],
                        B_arg, A_arg)
                    if _can_cache:
                        _sub[chunk_idx] = W_atom_v
            preds = self.model.predict_batch(
                chunk["descriptors"], chunk["grad_values"],
                chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
                chunk["positions"], chunk["Z_int"], chunk["boxes"],
                chunk["atom_mask"],
                W0, b0, W1, b1, W0p, b0p, W1p, b1p,
                W_atom=W_atom_v,
            )
            if self.cfg.scale_targets and self.cfg.target_mode == 1:
                num_atoms = tf.reduce_sum(chunk["atom_mask"], axis=1)
                preds = preds / tf.maximum(num_atoms, 1.0)[:, tf.newaxis]
            diff = preds - chunk["targets"]
            if self._pol_weights is not None:
                ds = tf.square(diff) * self._pol_weights
            else:
                ds = tf.square(diff)
            diff_sq_sum += tf.reduce_sum(ds)
            diff_count  += tf.cast(tf.size(ds), tf.float32)

        if self.cfg.val_size is None:
            ranges = [(s, min(s + struct_chunk, N_idx)) for s in range(0, N_idx, struct_chunk)]
            for _ci, (_, _, chunk) in enumerate(prefetched_chunks(
                    val_data, ranges,
                    pin_to_cpu=self.cfg.pin_data_to_cpu)):
                _consume(chunk, _ci)
                del chunk
        else:
            for _ci, s_start in enumerate(range(0, N_idx, struct_chunk)):
                s_end = min(s_start + struct_chunk, N_idx)
                sub_idx = val_idx_tf[s_start:s_end]
                chunk = slice_and_complete_chunk(val_data, sub_idx)
                if self.cfg.pin_data_to_cpu:
                    from data import _gpu_device_ctx
                    with _gpu_device_ctx():
                        chunk = {k: (tf.identity(v) if not k.startswith("_") else v)
                                 for k, v in chunk.items()}
                _consume(chunk, _ci)
                del chunk

        rmse = tf.sqrt(tf.maximum(diff_sq_sum / tf.maximum(diff_count, 1.0), 0.0))
        return float(rmse)

    def _split_reconstructed(self, params: tuple) -> dict:
        """Parse the heterogeneous tuple returned by reconstruct_params_tf
        into a named dict.

        Tail order:
            W0, b0, W1, b1                         (always)
            W0_pol, b0_pol, W1_pol, b1_pol         (target_mode == 2)
            U_pair                                 (descriptor_mixing on)
            W_pre_angular                          (preprocess_contract != "off")
        """
        out: dict = {"U_pair": None, "W_pre_angular": None}
        idx = 0
        out["W0"] = params[idx]; idx += 1
        out["b0"] = params[idx]; idx += 1
        out["W1"] = params[idx]; idx += 1
        out["b1"] = params[idx]; idx += 1
        if self.cfg.target_mode == 2:
            out["W0_pol"] = params[idx]; idx += 1
            out["b0_pol"] = params[idx]; idx += 1
            out["W1_pol"] = params[idx]; idx += 1
            out["b1_pol"] = params[idx]; idx += 1
        if self.n_U_pair > 0 and idx < len(params):
            out["U_pair"] = params[idx]; idx += 1
        if self.n_preprocess > 0 and idx < len(params):
            out["W_pre_angular"] = params[idx]; idx += 1
        return out

    def reconstruct_params_tf(self, param_vectors: tf.Tensor) -> tuple:
        """Reconstruct TNEP weight tensors from flat vectors using TF ops.

        Works inside @tf.function. Handles single [dim] or batched [P, dim] vectors.

        Args:
            param_vectors : [P, dim] or [dim] float32 tensor

        Returns:
            tuple of (W0, b0, W1, b1) and optionally (W0_pol, b0_pol, W1_pol, b1_pol)
            Each has shape [P, ...] for batched input or [...] for single.
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.H

        n_W0 = T * Q * H
        n_b0 = T * H
        n_W1 = T * H
        n_b1 = 1

        is_batched = len(param_vectors.shape) == 2

        def _extract(pv, offset):
            """Extract one ANN's worth of weights from the flat slab.

            Returns (W0, b0, W1, b1, offset).
            """
            W0 = tf.reshape(pv[..., offset:offset + n_W0],
                            [-1, T, Q, H] if is_batched else [T, Q, H])
            offset += n_W0
            b0 = tf.reshape(pv[..., offset:offset + n_b0],
                            [-1, T, H] if is_batched else [T, H])
            offset += n_b0
            W1 = tf.reshape(pv[..., offset:offset + n_W1],
                            [-1, T, H] if is_batched else [T, H])
            offset += n_W1
            b1 = pv[..., offset]
            offset += n_b1
            return W0, b0, W1, b1, offset

        first = _extract(param_vectors, 0)
        offset = first[-1]
        primary = first[:-1]

        if self.cfg.target_mode == 2:
            second = _extract(param_vectors, offset)
            offset = second[-1]
            tail = primary + second[:-1]
        else:
            tail = primary

        if self.n_U_pair > 0:
            # Unpack the descriptor-mixing tail. The per-type variant
            # has T contiguous slabs in the flat tail (same convention
            # as the ANN W0). Inside a slab, the l_aware layout is:
            #   [V_{p=0,l=0} (α²) | V_{p=0,l=1} (α²) | ...
            #    | V_{p=1,l=0} (α²) | ... ]
            U_flat = param_vectors[..., offset:offset + self.n_U_pair]
            U_pair_k = self._reconstruct_one_mixing_layer(
                U_flat, is_batched, self._mix_per_type, T)
            offset += self.n_U_pair
            tail = tail + (U_pair_k,)

        # Optional preprocess tail: W_pre summed entries. μ holds only
        # the learnable (summed) coefficients. Reconstruct the full W_pre
        # tensor by adding the scattered summed values to a base template
        # (kept positions = 1.0 from the model's _preprocess_kept_template,
        # summed positions = 0 in the template, will be filled by SNES).
        if self.n_preprocess > 0:
            pre_flat = param_vectors[..., offset:offset + self.n_preprocess]
            offset += self.n_preprocess
            # All static, precomputed at TNEP.__init__:
            #   M    : [n_summed, base_size] one-hot scatter (tf.constant)
            #   base : [..., kept template shape] (tf.constant, kept=1.0,
            #          summed=0)
            base = self.model._preprocess_kept_template
            M = self.model._preprocess_summed_scatter_M
            base_flat = tf.reshape(base, [-1])
            base_static_shape = base.shape  # static — no tf.shape() call
            if is_batched:
                # pre_flat: [C, n_summed] → summed_contrib [C, base_size]
                summed_contrib = tf.matmul(pre_flat, M)
                full_flat = base_flat[tf.newaxis, :] + summed_contrib
                # Use STATIC base shape so downstream ops can constant-fold
                # the gather/scatter in _W0_preprocess_eff. The leading
                # candidate dim stays dynamic (only -1 needed).
                W_pre = tf.reshape(
                    full_flat,
                    [-1] + base_static_shape.as_list())
            else:
                summed_contrib = tf.matmul(pre_flat[tf.newaxis, :], M)[0]
                full_flat = base_flat + summed_contrib
                W_pre = tf.reshape(full_flat, base_static_shape)
            tail = tail + (W_pre,)

        return tail

    def _reconstruct_one_mixing_layer(self, U_flat: tf.Tensor,
                                        is_batched: bool,
                                        per_type: bool,
                                        T: int) -> tf.Tensor:
        """Reconstruct ONE mixing-layer's U_pair tensor from its flat slab.

        Extracted from the body of reconstruct_params_tf so we can call it
        once per stacked mixing layer. The Cayley fast paths mirror the
        legacy single-layer code exactly.
        """
        max_alpha = max(self._mix_alpha_per_pair)
        L = self._mix_L
        alpha_payloads = [
            (a * (a - 1) // 2) if self._mix_cayley else a * a
            for a in self._mix_alpha_per_pair
        ]
        per_T_block = sum(L * p for p in alpha_payloads)
        uniform_alpha = (
            self._mix_cayley
            and len(set(self._mix_alpha_per_pair)) == 1)

        def _extract_t_block_l_aware(t_idx: int):
            start = t_idx * per_T_block
            slab = U_flat[..., start:start + per_T_block]
            if uniform_alpha:
                alpha_p = self._mix_alpha_per_pair[0]
                payload = alpha_payloads[0]
                n_pairs = len(self._mix_alpha_per_pair)
                n_blocks = n_pairs * L
                new_shape = tf.concat(
                    [tf.shape(slab)[:-1], [n_blocks, payload]], axis=0)
                stacked = tf.reshape(slab, new_shape)
                V = self._cayley_blocks_batched(stacked, alpha_p)
                pad_r = max_alpha - alpha_p
                if pad_r > 0:
                    paddings = [[0, 0]] * (len(V.shape) - 2) + \
                               [[0, pad_r], [0, pad_r]]
                    V = tf.pad(V, paddings)
                out_shape = tf.concat(
                    [tf.shape(V)[:-3],
                     [n_pairs, L, max_alpha, max_alpha]], axis=0)
                return tf.reshape(V, out_shape)

            # Cayley/expm non-uniform-α fast path: zero-pad each pair's
            # skew triangle into the upper triangle of a max_α×max_α skew
            # via the precomputed gather index, then run ONE batched
            # solve/expm at size max_α instead of `num_pairs` per-pair
            # ops. Removes (num_pairs − 1) host↔device syncs per layer in
            # eager mode (~6 ms each) — the dominant overhead at N>1.
            if self._mix_cayley:
                n_pairs = len(self._mix_alpha_per_pair)
                max_payload = self._mix_l_aware_cayley_max_payload
                zero_tail_shape = tf.concat(
                    [tf.shape(slab)[:-1], [1]], axis=0)
                slab_padded = tf.concat(
                    [slab, tf.zeros(zero_tail_shape, dtype=slab.dtype)],
                    axis=-1)
                gathered = tf.gather(
                    slab_padded, self._mix_l_aware_cayley_gather, axis=-1)
                stacked_shape = tf.concat(
                    [tf.shape(slab)[:-1],
                     [n_pairs * L, max_payload]], axis=0)
                stacked = tf.reshape(gathered, stacked_shape)
                V = self._cayley_blocks_batched(stacked, max_alpha)
                out_shape = tf.concat(
                    [tf.shape(V)[:-3],
                     [n_pairs, L, max_alpha, max_alpha]], axis=0)
                return tf.reshape(V, out_shape)

            # Non-cayley/expm ("off" regulariser) non-uniform fallback:
            # no solve/expm to batch, so the Python loop is just reshapes
            # + pads — bounded launch overhead.
            pair_blocks: list = []
            inner_offset = 0
            for alpha_p, payload in zip(self._mix_alpha_per_pair, alpha_payloads):
                pair_payload = L * payload
                sub = slab[..., inner_offset:inner_offset + pair_payload]
                inner_offset += pair_payload
                pad_r = max_alpha - alpha_p
                new_shape = tf.concat(
                    [tf.shape(sub)[:-1],
                     [L, alpha_p, alpha_p]], axis=0)
                l_block = tf.reshape(sub, new_shape)
                if pad_r > 0:
                    paddings = [[0, 0]] * (len(l_block.shape) - 2) + \
                               [[0, pad_r], [0, pad_r]]
                    l_block = tf.pad(l_block, paddings)
                pair_blocks.append(l_block)
            stack_axis_p = -4 if is_batched else 0
            return tf.stack(pair_blocks, axis=stack_axis_p)

        if per_type:
            # Batched-across-T path for the Cayley/expm non-uniform-α case
            # (the hot path under `mixing_per_type=True` + l_aware + expm).
            # Folds the T slabs into the block axis so we issue ONE
            # `tf.linalg.expm` over T·n_pairs·L blocks instead of T
            # separate launches. Each eager expm launch costs ~6 ms (see
            # eager_tf_perf_gotcha memory); avoiding 2 of them per chunk
            # × per mixing layer × per generation adds up.
            if self._mix_cayley:
                # Cayley/expm path — batch the T·n_pairs·L blocks into ONE
                # expm call regardless of whether α is uniform or not.
                # Uniform-α: no zero-pad/gather needed; just reshape.
                # Non-uniform α: pad skew triangles to max_payload via the
                # precomputed gather index, then reshape.
                n_pairs = len(self._mix_alpha_per_pair)
                uniform_alpha_pt = len(set(self._mix_alpha_per_pair)) == 1
                # All T slabs are contiguous in U_flat at offsets
                # [t*per_T_block, (t+1)*per_T_block). Slice out the whole
                # T·per_T_block chunk and reshape so T is its own axis.
                T_total = T * per_T_block
                slabs_all = U_flat[..., :T_total]
                leading = tf.shape(slabs_all)[:-1]
                slab_per_t_shape = tf.concat(
                    [leading, [T, per_T_block]], axis=0)
                slab_per_t = tf.reshape(slabs_all, slab_per_t_shape)
                if uniform_alpha_pt:
                    # Uniform α: every block has the same skew payload
                    # (alpha_payloads[0]). Reshape directly into block
                    # batch without padding/gather.
                    payload = self._mix_alpha_per_pair[0] * (
                        self._mix_alpha_per_pair[0] - 1) // 2
                    stacked_shape = tf.concat(
                        [leading, [T * n_pairs * L, payload]], axis=0)
                    stacked = tf.reshape(slab_per_t, stacked_shape)
                    V = self._cayley_blocks_batched(
                        stacked, self._mix_alpha_per_pair[0])
                else:
                    max_payload = self._mix_l_aware_cayley_max_payload
                    # Zero-pad the skew triangle to max_payload (gather
                    # index applies the trailing-zero trick).
                    zero_tail_shape = tf.concat(
                        [leading, [T, 1]], axis=0)
                    slab_padded = tf.concat(
                        [slab_per_t,
                         tf.zeros(zero_tail_shape, dtype=slab_per_t.dtype)],
                        axis=-1)
                    gathered = tf.gather(
                        slab_padded,
                        self._mix_l_aware_cayley_gather, axis=-1)
                    stacked_shape = tf.concat(
                        [leading, [T * n_pairs * L, max_payload]], axis=0)
                    stacked = tf.reshape(gathered, stacked_shape)
                    V = self._cayley_blocks_batched(stacked, max_alpha)
                # V shape: [..., T·n_pairs·L, max_alpha, max_alpha]
                # Reshape back to [..., T, n_pairs, L, max_alpha, max_alpha].
                final_shape = tf.concat(
                    [leading,
                     [T, n_pairs, L, max_alpha, max_alpha]], axis=0)
                return tf.reshape(V, final_shape)
            # Fall back to the per-T list-comp ONLY for the "off"
            # regulariser (no solve/expm to batch). Cayley/expm paths
            # (uniform-α and non-uniform-α) are handled above with a
            # single batched expm call across T.
            per_t = [_extract_t_block_l_aware(t) for t in range(T)]
            stack_axis = 1 if is_batched else 0
            return tf.stack(per_t, axis=stack_axis)
        return _extract_t_block_l_aware(0)

    @tf.function(reduce_retracing=True)
    def compute_regularization_tf(self, param_vectors: tf.Tensor) -> tf.Tensor:
        """Compute L1+L2 regularization for batched parameter vectors.

        For multi-element systems, computes per-type regularization (GPUMD NEP4):
        each type's parameters are penalized separately, averaged across types,
        then added to a global regularization term.

        Args:
            param_vectors : [P, dim] float32

        Returns:
            reg : [P] float32 — L1 + L2 penalty per candidate
        """
        T = self.cfg.num_types

        if T > 1:
            n_per_type = self._n_per_type  # W0_t + b0_t (+ W0_2_t + b0_2_t) + W1_t

            total_l1 = tf.zeros([tf.shape(param_vectors)[0]])
            total_l2 = tf.zeros([tf.shape(param_vectors)[0]])

            for t in range(T):
                type_params = self._extract_type_params_batched(param_vectors, t)  # [P, n_per_type]
                total_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(type_params), axis=1) / n_per_type
                total_l2 += self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(type_params), axis=1) / n_per_type)

            # Average per-type + global over typed params only (excludes b1)
            typed = param_vectors[:, :self.n_typed]  # [P, n_typed]
            n_typed_total = self.n_typed
            if self.cfg.target_mode == 2:
                typed2 = param_vectors[:, self.n_primary:self.n_primary + self.n_typed]
                typed = tf.concat([typed, typed2], axis=1)
                n_typed_total = 2 * self.n_typed
            global_l1 = self.lambda_1 * tf.reduce_sum(tf.abs(typed), axis=1) / n_typed_total
            global_l2 = self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(typed), axis=1) / n_typed_total)

            reg = total_l1 / T + global_l1 + total_l2 / T + global_l2
        else:
            # Single-type path: keep V_pair out of the main sum when
            # it's held by the cayley / expm parameterisation — those
            # A entries must NEVER be L1/L2-regularised, since pulling
            # A → 0 collapses U → I and defeats the rotation map.
            if self._mix_cayley:
                ann = param_vectors[:, :self.n_anns_total]
                ann_n = self.n_anns_total
            else:
                ann = param_vectors
                ann_n = self.dim
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(ann), axis=1) / ann_n
            l2 = self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(ann), axis=1) / ann_n)
            reg = l1 + l2

        return reg

    def _extract_type_params_batched(self, param_vectors: tf.Tensor, t: int) -> tf.Tensor:
        """Extract type-t parameters from batched flat vectors.

        Returns [P, n_per_type] — see _extract_type_params for layout.
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.H

        w0_start = t * Q * H
        w0_end = w0_start + Q * H

        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H

        w1_offset = b0_offset + T * H
        w1_start = w1_offset + t * H
        w1_end = w1_start + H

        return tf.concat([
            param_vectors[:, w0_start:w0_end],
            param_vectors[:, b0_start:b0_end],
            param_vectors[:, w1_start:w1_end],
        ], axis=1)

    def evaluate_population(self, samples_tf: tf.Tensor, batch_data: dict[str, tf.Tensor],
                            return_per_type: bool = False) -> tf.Tensor:
        """Evaluate all SNES candidates on a batch of structures on GPU.

        Chunks along both population (population_chunk_size) and structure
        (batch_chunk_size) dimensions to limit VRAM usage. Accumulates sum
        of squared errors across structure chunks for correct RMSE.

        Args:
            samples_tf : [P, dim] float32 — all candidate parameter vectors
            batch_data : dict with padded batch tensors:
                descriptors   : [B, A, Q]
                gradients     : [B, A, M, 3, Q]
                grad_index    : [B, A, M]
                positions     : [B, A, 3]
                Z_int         : [B, A]
                boxes         : [B, 3, 3]
                targets       : [B, T_dim]
                atom_mask     : [B, A]
                neighbor_mask : [B, A, M]
            return_per_type : bool — if True, return [P, T+1] per-type RMSE
                (GPUMD-style: each type's RMSE from structures containing that type,
                 plus global RMSE at index T).

        Returns:
            fitness : [P] float32 — RMSE (+ regularization if enabled) per candidate
                      OR [P, T+1] if return_per_type=True
        """
        P = self.pop_size
        # B is the number of structures in the batch. In OTF mode the
        # `descriptors` slot is zero-sized along the Q axis (placeholder),
        # so use `num_atoms` whose leading axis is the structure count in
        # both modes.
        B = batch_data["num_atoms"].shape[0]
        T_dim = batch_data["targets"].shape[1]
        pop_chunk = self.cfg.population_chunk_size if self.cfg.population_chunk_size is not None else P
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else B

        T = self.cfg.num_types
        T_dim_f = tf.cast(T_dim, tf.float32)
        B_f = tf.cast(B, tf.float32)
        eval_chunk_fn = self._evaluate_chunk

        # SS_tot for RRMSE: total target magnitude squared over the whole batch.
        # Independent of weighting choices and candidates; computed once.
        ss_tot_batch = tf.reduce_sum(tf.square(batch_data["targets"]))
        ss_tot_batch = tf.maximum(ss_tot_batch, 1e-12)

        all_fitness = []

        # Streaming evaluation: outer loop = structure chunks, inner = population
        # chunks. Each chunk is built just-in-time by `prefetched_chunks` —
        # purely on-GPU when `_gv_resident_gpu` is set, otherwise via the
        # in-RAM staging path. Per-chunk per-structure errors are reduced
        # into running accumulators so the full [C, B] tensor never
        # materialises.
        from data import prefetched_chunks

        # Pre-compute per-type counts on host (independent of population).
        if return_per_type:
            tc_full = batch_data["types_contained"]                     # [B, T]
            type_counts = tf.maximum(tf.reduce_sum(tc_full, axis=0), 1.0)  # [T]

        # Running accumulators. Two parallel sums per pop-chunk:
        #   total_acc_parts[k]    : [C] — training-loss contribution
        #   sq_acc_parts[k]       : [C] — always-MSE squared-error sum
        # The training loss drives SNES ranking; the sq sum drives the
        # always-on RMSE / RRMSE reporting (stashed on self at the end
        # of this function so the fit loop can pull them).
        total_acc_parts: list = [None] * ((P + pop_chunk - 1) // pop_chunk)
        sq_acc_parts: list = [None] * len(total_acc_parts)
        per_type_acc_parts: list = [None] * len(total_acc_parts) if return_per_type else []

        ranges = [(s, min(s + struct_chunk, B)) for s in range(0, B, struct_chunk)]
        for chunk_idx, (s_start, s_end, chunk) in enumerate(prefetched_chunks(
                batch_data, ranges,
                pin_to_cpu=self.cfg.pin_data_to_cpu)):
            B_chunk = s_end - s_start
            chunk_lo = s_start
            chunk_hi = s_end

            # Slice precomputed [B]-shaped quantities to this chunk's range.
            tc_chunk = tc_full[chunk_lo:chunk_hi] if return_per_type else None

            # Precompute candidate-independent W_atom [B,A,3,Q] ONCE per
            # struct chunk. When cfg.batch_size is None (full-batch
            # training), W_atom depends only on the static train_data
            # (positions, boxes, grad_values, pair_*) and is invariant
            # across generations — cache it on the SNES instance keyed by
            # (chunk_idx, batch_data id) so subsequent generations reuse
            # the same precomputed tensor. With batch_size != None each
            # gen draws a fresh sub-batch → no cache reuse, recompute as
            # before. The cache is invalidated whenever the batch_data
            # identity changes (e.g. resampled batch).
            _W_atom_cached = None
            if self.cfg.target_mode == 1:
                if getattr(self.cfg, "batch_size", None) is None:
                    _bd_id = id(batch_data)
                    _cache = getattr(self, "_W_atom_static_cache", None)
                    if _cache is None or _cache.get("_bd_id") != _bd_id:
                        self._W_atom_static_cache = {"_bd_id": _bd_id}
                        _cache = self._W_atom_static_cache
                    _W_atom_cached = _cache.get(chunk_idx)
                if _W_atom_cached is not None:
                    chunk["_W_atom"] = _W_atom_cached
            if self.cfg.target_mode == 1 and "_W_atom" not in chunk:
                _gv = chunk["grad_values"]
                _ps = chunk["pair_struct"]
                _pa = chunk["pair_atom"]
                _pg = chunk["pair_gidx"]
                _pos = chunk["positions"]
                _boxes = chunk["boxes"]
                _desc = chunk["descriptors"]
                _B_st = _desc.shape[0]
                _A_st = _desc.shape[1]
                _B_arg = _B_st if _B_st is not None else tf.shape(_desc)[0]
                _A_arg = _A_st if _A_st is not None else tf.shape(_desc)[1]
                chunk["_W_atom"] = self.model._precompute_dipole_kernel(
                    _gv, _ps, _pa, _pg, _pos, _boxes, _B_arg, _A_arg)
                # Write through to the per-chunk_idx static cache for
                # reuse on subsequent generations (full-batch only).
                if (getattr(self.cfg, "batch_size", None) is None
                        and getattr(self, "_W_atom_static_cache", None)
                            is not None
                        and self._W_atom_static_cache.get("_bd_id")
                            == id(batch_data)):
                    self._W_atom_static_cache[chunk_idx] = chunk["_W_atom"]

            for pop_idx, p_start in enumerate(range(0, P, pop_chunk)):
                p_end      = min(p_start + pop_chunk, P)
                candidates = samples_tf[p_start:p_end]                  # [C, dim]
                chunk_err, chunk_sq = eval_chunk_fn(candidates, chunk)   # [C, B_chunk] each


                # Accumulate global sums over structures: [C] each.
                global_part = tf.reduce_sum(chunk_err, axis=1)
                sq_part = tf.reduce_sum(chunk_sq, axis=1)
                if total_acc_parts[pop_idx] is None:
                    total_acc_parts[pop_idx] = global_part
                    sq_acc_parts[pop_idx] = sq_part
                else:
                    total_acc_parts[pop_idx] = total_acc_parts[pop_idx] + global_part
                    sq_acc_parts[pop_idx] = sq_acc_parts[pop_idx] + sq_part

                if return_per_type:
                    # Per-type sums: einsum reduces both struct and m axes in
                    # one fused op, yielding [C, T] per chunk.
                    per_type_chunk = tf.einsum('cb,bt->ct', chunk_err, tc_chunk)  # [C, T]
                    if per_type_acc_parts[pop_idx] is None:
                        per_type_acc_parts[pop_idx] = per_type_chunk
                    else:
                        per_type_acc_parts[pop_idx] = per_type_acc_parts[pop_idx] + per_type_chunk

                del chunk_err, chunk_sq
            # When the GPU LRU cache is active it owns the chunk's tensors
            # across generations; otherwise this `del` releases them so the
            # next chunk's stage starts with the freed VRAM.
            del chunk

        # Convert accumulated sums of per-structure errors → fitness.
        # Per-structure error is a sum of squared residuals (MSE), so
        # fitness = sqrt(mean) = RMSE (canonical).
        for pop_idx in range(len(total_acc_parts)):
            global_err = total_acc_parts[pop_idx]                       # [C]
            if return_per_type:
                per_type_err = per_type_acc_parts[pop_idx]              # [C, T]
                per_type_parts = []
                for t in range(T):
                    raw = per_type_err[:, t] / (type_counts[t] * T_dim_f)
                    per_type_parts.append(tf.sqrt(tf.maximum(raw, 0.0)))
                raw_global = global_err / (B_f * T_dim_f)
                per_type_parts.append(tf.sqrt(tf.maximum(raw_global, 0.0)))
                all_fitness.append(tf.stack(per_type_parts, axis=1))    # [C, T+1]
            else:
                raw_global = global_err / (B_f * T_dim_f)
                chunk_fitness = tf.sqrt(tf.maximum(raw_global, 0.0))
                all_fitness.append(chunk_fitness)

        if return_per_type:
            fitness = tf.concat(all_fitness, axis=0)  # [P, T+1]
        else:
            fitness = tf.concat(all_fitness, axis=0)  # [P]

        if not return_per_type and self.cfg.toggle_regularization:
            reg = self.compute_regularization_tf(samples_tf)
            fitness = fitness + reg

        # Stash always-MSE reporting metrics for the fit loop. These
        # are independent of inverse weighting, so RMSE and RRMSE
        # reported in history are comparable across runs.
        sq_per_cand = tf.concat(sq_acc_parts, axis=0)                # [P]
        # RMSE per candidate: sqrt(mean squared error per element).
        # Denominator = B * T_dim so the result matches "RMSE over all
        # vector components of all structures".
        rmse_per_cand = tf.sqrt(tf.maximum(sq_per_cand / (B_f * T_dim_f), 0.0))
        # RRMSE per candidate: sqrt(SS_res / SS_tot).
        rrmse_per_cand = tf.sqrt(tf.maximum(sq_per_cand / ss_tot_batch, 0.0))
        self._last_rmse_per_cand = rmse_per_cand
        self._last_rrmse_per_cand = rrmse_per_cand

        return fitness

    @tf.function(reduce_retracing=True)
    def _evaluate_chunk(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        return self._evaluate_chunk_impl(chunk_samples, batch_data)

    def _evaluate_chunk_impl(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        """Evaluate a chunk of C candidates on B structures.

        Returns a *pair* of per-structure tensors:
          - fitness_err: training-loss contribution (squared error,
              optionally per-component-weighted) — used for SNES
              ranking and the regularised total objective.
          - sq_err: always-MSE squared-error contribution — used for
              RMSE and RRMSE reporting.

        Args:
            chunk_samples : [C, dim]
            batch_data    : dict of [B, ...] tensors

        Returns:
            fitness_err : [C, B]
            sq_err      : [C, B]
        """
        desc        = batch_data["descriptors"]   # [B, A, Q]
        grad_values = batch_data["grad_values"]   # [P, 3, Q]
        pair_atom   = batch_data["pair_atom"]     # [P]
        pair_gidx   = batch_data["pair_gidx"]     # [P]
        pair_struct = batch_data["pair_struct"]   # [P]
        pos         = batch_data["positions"]     # [B, A, 3]
        Z           = batch_data["Z_int"]         # [B, A]
        boxes       = batch_data["boxes"]         # [B, 3, 3]
        targets     = batch_data["targets"]       # [B, T_dim]
        amask       = batch_data["atom_mask"]     # [B, A]

        # Precompute per-atom normalization factor once (not per-candidate)
        _scale_preds = self.cfg.scale_targets and self.cfg.target_mode == 1
        if _scale_preds:
            num_atoms = tf.reduce_sum(amask, axis=1)  # [B]
            inv_num_atoms = 1.0 / tf.maximum(num_atoms, 1.0)  # [B]
            inv_num_atoms = inv_num_atoms[:, tf.newaxis]  # [B, 1]

        # Reconstruct weights for all candidates in chunk. Tail may
        # include a U_pair tensor when cfg.descriptor_mixing=True.
        params = self.reconstruct_params_tf(chunk_samples)
        named = self._split_reconstructed(params)
        W0 = named.get("W0"); b0 = named.get("b0")
        W1 = named.get("W1"); b1 = named.get("b1")
        W0p = named.get("W0_pol")
        b0p = named.get("b0_pol")
        W1p = named.get("W1_pol")
        b1p = named.get("b1_pol")
        # U_pair_cand: the per-candidate mixing tensor (None when no mixing).
        U_pair_cand = named.get("U_pair")
        # W_pre_angular_cand: per-candidate preprocess coefficients (None
        # when descriptor_preprocess_contract='off').
        W_pre_angular_cand = named.get("W_pre_angular")

        if self.cfg.target_mode == 2:
            pol_weights = self._pol_weights  # [6] component weights
            # Pre-absorb U_pair^T into W0 / W0_pol per candidate so the
            # vectorized_map loop body uses raw descriptors and no
            # gradient pull-back. Works because _W0_eff is linear and
            # broadcasts over the leading candidate axis.
            if U_pair_cand is not None:
                W0 = self.model._W0_eff(W0, U_pair_cand)
                W0p = self.model._W0_eff(W0p, U_pair_cand)

            # Per-component weights for the training loss.
            fitness_comp_w = pol_weights[tf.newaxis]  # [1, 6]
            # Squared-error reporting always uses pol_weights only.
            sq_comp_w = pol_weights[tf.newaxis]

            def _forward_one_candidate(args):
                w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p = args
                preds = self.model.predict_batch(
                    desc, grad_values, pair_atom, pair_gidx, pair_struct,
                    pos, Z, boxes, amask,
                    w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p,
                )
                diff = preds - targets  # [B, 6]
                fitness = squared_error_per_structure(
                    diff, component_weights=fitness_comp_w)
                sq = squared_error_per_structure(diff, component_weights=sq_comp_w)
                return tf.stack([fitness, sq], axis=0)  # [2, B]

            stacked = (W0, b0, W1, b1, W0p, b0p, W1p, b1p)
            both = tf.vectorized_map(_forward_one_candidate, stacked)  # [C, 2, B]
            return both[:, 0, :], both[:, 1, :]

        else:

            # Precompute W_atom [B, A, 3, Q] once — independent of all candidates.
            # `evaluate_population` precomputes this per STRUCT chunk and
            # stashes it on `batch_data["_W_atom"]` so the inner POP-chunk
            # loop reuses the same tensor instead of recomputing it pop_chunks
            # times per struct chunk. Falls back to inline compute when the
            # key is absent (other callers: score, etc.).
            W_atom = batch_data.get("_W_atom")
            if W_atom is None:
                B_static = desc.shape[0]
                A_static = desc.shape[1]
                B_arg = B_static if B_static is not None else tf.shape(desc)[0]
                A_arg = A_static if A_static is not None else tf.shape(desc)[1]
                if self.cfg.target_mode == 1:
                    W_atom = self.model._precompute_dipole_kernel(
                        grad_values, pair_struct, pair_atom, pair_gidx,
                        pos, boxes,
                        B_arg, A_arg)

            # Evaluate all C candidates simultaneously using explicit batched GEMMs.
            # predict_batch_candidates executes one GEMM per type in each direction
            # rather than C separate matmuls inside vectorized_map.
            preds = self.model.predict_batch_candidates(
                desc, W_atom, Z, amask, W0, b0, W1, b1,
                U_pair=U_pair_cand,
                W_pre_angular=W_pre_angular_cand)  # [C, B, T_dim]

            if _scale_preds:
                preds = preds * inv_num_atoms[tf.newaxis]  # [C, B, T_dim] * [1, B, 1]

            diff = preds - targets[tf.newaxis]  # [C, B, T_dim]
            fitness = squared_error_per_structure(diff)
            sq = fitness  # unweighted MSE, for RMSE/RRMSE
            return fitness, sq
