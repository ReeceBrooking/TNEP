from __future__ import annotations

import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING, Callable

from loss_functions import per_structure_error, squared_error_per_structure

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
    """Assign weight arrays directly into the TNEP model's tf.Variables.

    Tail conventions produced by SNES.reconstruct_params_tf:
        mode 0/1, no mixing : (W0, b0, W1, b1)
        mode 0/1, mixing    : (W0, b0, W1, b1, U_pair)
        mode 2,  no mixing  : (W0, b0, W1, b1, W0_pol, b0_pol, W1_pol, b1_pol)
        mode 2,  mixing     : (W0, b0, W1, b1, W0_pol, b0_pol, W1_pol, b1_pol, U_pair)
    """
    model.W0.assign(params[0])
    model.b0.assign(params[1])
    model.W1.assign(params[2])
    model.b1.assign(params[3])
    has_pol = len(params) >= 8
    if has_pol:
        model.W0_pol.assign(params[4])
        model.b0_pol.assign(params[5])
        model.W1_pol.assign(params[6])
        model.b1_pol.assign(params[7])
    # The last element is U_pair if the model has descriptor mixing.
    if getattr(model, "descriptor_mixing", False) and len(params) in (5, 9):
        model.U_pair.assign(params[-1])

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
        n_W0 = self.cfg.num_types * self.cfg.dim_q * self.cfg.num_neurons
        n_b0 = self.cfg.num_types * self.cfg.num_neurons
        n_W1 = self.cfg.num_types * self.cfg.num_neurons
        n_b1 = 1
        self.n_typed = n_W0 + n_b0 + n_W1  # per-type params (excludes b1)
        self.n_primary = n_W0 + n_b0 + n_W1 + n_b1
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
            from DescriptorBuilderGPU import descriptor_block_layout
            self._mix_layout = descriptor_block_layout(self.cfg)
            self._mix_block_sizes = [
                self._mix_layout["block_sizes"][k]
                for k in self._mix_layout["pair_keys"]]
            self._mix_per_type = bool(
                getattr(self.cfg, "descriptor_mixing_per_type", False))
            self._mix_arch = str(
                getattr(self.cfg, "descriptor_mixing_arch", "linear")).lower()
            # When the regulariser is "cayley", each [bs, bs] block is
            # parameterised by the upper triangle of a skew-symmetric
            # matrix A (bs(bs-1)/2 free params). The dense block V is
            # then reconstructed via the Cayley transform inside
            # reconstruct_params_tf; SNES sees only the upper-triangle
            # storage. Halves (roughly) the search-space dimensionality
            # and guarantees U lies on the rotation group regardless of
            # any λ — see plan and TNEPconfig docs.
            self._mix_cayley = (
                str(getattr(self.cfg, "descriptor_mixing_regularizer",
                            "off")).lower() == "cayley")
            # `block_count(bs)` returns the per-block SNES dim for the
            # current arch+regulariser combination.
            def _block_count(bs: int) -> int:
                return (bs * (bs - 1) // 2) if self._mix_cayley else bs * bs

            # n_U_pair packs only the ACTIVE entries — padded rows/cols
            # don't take SNES degrees of freedom. Layout depends on arch:
            #   "linear"       : per pair, bs² entries (or bs(bs-1)/2 for cayley).
            #   "l_aware"      : per pair × (l_max+1), α² (or α(α-1)/2).
            #   "cross_pair_l" : per l, N_l² (or N_l(N_l-1)/2).
            if self._mix_arch == "linear":
                per_T_block = int(sum(_block_count(bs)
                                       for bs in self._mix_block_sizes))
            elif self._mix_arch == "l_aware":
                self._mix_alpha_per_pair = [
                    int(self._mix_layout["alpha_eff_per_pair"][k])
                    for k in self._mix_layout["pair_keys"]]
                L = int(self.cfg.l_max) + 1
                self._mix_L = L
                per_T_block = int(sum(L * _block_count(a)
                                       for a in self._mix_alpha_per_pair))
            elif self._mix_arch == "cross_pair_l":
                L = int(self.cfg.l_max) + 1
                self._mix_L = L
                self._mix_N_per_l = int(self._mix_layout["N_per_l"])
                per_T_block = L * _block_count(self._mix_N_per_l)
            else:
                raise ValueError(
                    f"descriptor_mixing_arch={self._mix_arch!r} not in "
                    "('linear', 'l_aware', 'cross_pair_l')")
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
                # Collect unique block sizes across arch variants.
                if self._mix_arch == "linear":
                    block_sizes = set(self._mix_block_sizes)
                elif self._mix_arch == "l_aware":
                    block_sizes = set(self._mix_alpha_per_pair)
                else:                                              # cross_pair_l
                    block_sizes = {self._mix_N_per_l}
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
        else:
            self._mix_block_sizes = []
            self._mix_per_type = False
            self._mix_arch = "linear"
            self._mix_cayley = False
            self._cayley_scatter_cache = {}
            self.n_U_pair = 0
        self.dim = self.n_anns_total + self.n_U_pair

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
        self.sigma = tf.Variable(
            tf.fill([self.dim], self.cfg.init_sigma),
            trainable=False, name="snes_sigma")

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

        # V_pair regulariser mode: "off" | "shrinkage" | "orthogonal".
        # Orthogonal penalty has its own lambda (defaults to the same
        # auto formula scaled by n_U_pair only, since the orth penalty
        # is naturally per-mixing-entry and shouldn't inherit the ANN's
        # dimensionality scaling). -1 sentinel enables dynamic adapt.
        self._mix_reg_mode = str(getattr(
            self.cfg, "descriptor_mixing_regularizer", "off")).lower()
        if self._mix_reg_mode not in ("off", "shrinkage", "orthogonal", "cayley"):
            raise ValueError(
                f"descriptor_mixing_regularizer={self._mix_reg_mode!r} not "
                "recognised (expected 'off', 'shrinkage', 'orthogonal', "
                "or 'cayley')")
        # The "cayley" mode is a *parameterisation*, not a soft penalty:
        # U is reconstructed via the Cayley map from a skew-symmetric A,
        # so it's structurally orthogonal regardless of any λ. The
        # shrinkage / orthogonal soft-penalty paths must remain
        # silent under cayley — the existing dispatches `reg_Vpair` /
        # `reg_Vorth` already gate on the exact strings "shrinkage" /
        # "orthogonal", so this just works.
        if self.n_U_pair > 0:
            auto_lambda_orth = float(np.sqrt(
                self.n_U_pair * 1e-6 / max(self.cfg.num_types, 1)))
        else:
            auto_lambda_orth = 0.0
        cfg_lo = getattr(self.cfg, "lambda_orth", None)
        self._dyn_lambda_orth = (cfg_lo == -1)
        init_lambda_orth = (auto_lambda_orth if (cfg_lo is None or self._dyn_lambda_orth)
                            else float(cfg_lo))
        self._lambda_orth = tf.Variable(init_lambda_orth, dtype=tf.float32,
                                        trainable=False, name="lambda_orth")

        # Loss / weighting flags are read fresh from cfg in
        # evaluate_population — no cached attrs (so live cfg edits
        # work). loss_type ∈ {"mse", "mae", "huber"};
        # inverse_weight_mode ∈ {"none", "vector_magnitude",
        # "per_component"}. See cfg docs.

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

    def compute_regularization(self, param_vector: tf.Tensor | np.ndarray
                               ) -> tuple[float, float, float]:
        """Compute L1, L2, and orthogonal regularisation penalties.

        For multi-element systems, computes per-type regularization
        (GPUMD NEP4): each atom type's parameters are penalised
        separately using num_vars/num_types as the denominator, then
        averaged across types and added to a global regularisation
        term over all parameters.

        The orthogonal penalty (when mode == "orthogonal") is reported
        separately from L2 so it can drive an independent dynamic-λ
        schedule — otherwise lambda_2 would chase a signal it doesn't
        control.

        Args:
            param_vector : [dim] tensor or ndarray — flat parameter vector

        Returns:
            l1     : float — L1 penalty (ANN + shrinkage V_pair when on)
            l2     : float — L2 penalty (ANN + shrinkage V_pair when on)
            l_orth : float — orthogonal penalty on V_pair (0 unless mode=="orthogonal")
        """
        pv = tf.cast(param_vector, tf.float32)
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons
        n_per_type = Q * H + H + H  # W0_t + b0_t + W1_t

        # V_pair regularisation mode (set in __init__ from cfg). The
        # shrinkage path uses lambda_1/2 on the residual tail; the
        # orthogonal path uses lambda_orth on ‖UᵀU - I‖².
        reg_Vpair = (self.n_U_pair > 0 and self._mix_reg_mode == "shrinkage")
        reg_Vorth = (self.n_U_pair > 0 and self._mix_reg_mode == "orthogonal")

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
            # Single-type path: when V_pair is handled separately (by
            # shrinkage, orthogonal, OR Cayley parameterisation), keep
            # it out of the main L1/L2 sum. Cayley is a structural
            # constraint, not a soft penalty — its A entries are NOT
            # to be regularised, since shrinking A toward 0 collapses
            # U toward I and defeats the Cayley map's purpose.
            v_handled = reg_Vpair or reg_Vorth or self._mix_cayley
            ann = pv[:self.n_anns_total] if v_handled else pv
            ann_n = self.n_anns_total if v_handled else self.dim
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(ann)) / ann_n
            l2 = self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(ann)) / ann_n)

        # Optional V_pair tail regularisation (residual mixing layer).
        # Per-type slabs averaged across T (matching the ANN per-type
        # convention); shared V is added as a single global term.
        if reg_Vpair:
            tail = pv[self.n_anns_total:]
            if self._mix_per_type and T > 1:
                per_T = self.n_U_pair // T
                vp_l1 = tf.constant(0.0)
                vp_l2 = tf.constant(0.0)
                for t in range(T):
                    slab = tail[t * per_T:(t + 1) * per_T]
                    vp_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(slab)) / per_T
                    vp_l2 += self.lambda_2 * tf.sqrt(
                        tf.reduce_sum(tf.square(slab)) / per_T)
                l1 = l1 + vp_l1 / T
                l2 = l2 + vp_l2 / T
            else:
                l1 = l1 + self.lambda_1 * tf.reduce_sum(tf.abs(tail)) / self.n_U_pair
                l2 = l2 + self.lambda_2 * tf.sqrt(
                    tf.reduce_sum(tf.square(tail)) / self.n_U_pair)

        # Orthogonal V_pair regularisation (replaces shrinkage when
        # mode == "orthogonal"). The penalty enforces UᵀU = I per
        # block, so U is constrained to the orthogonal group (a pure
        # rotation/reflection of the descriptor basis, no scaling) —
        # which lets the data fit pick whichever rotation works,
        # without anchoring at identity. Reported as a separate
        # signal so its lambda can adapt independently.
        if reg_Vorth:
            tail = pv[self.n_anns_total:]
            l_orth = float(self._lambda_orth.numpy()) * float(
                self._orth_penalty_total(tail))
        else:
            l_orth = 0.0

        return float(l1), float(l2), float(l_orth)

    def _build_mu_init(self, rng: np.random.Generator) -> np.ndarray:
        """Initialise the μ vector according to cfg.mu_init_scheme.

        Returns a [dim]-shaped float32 array. The V_pair tail (when
        descriptor mixing is enabled) is always zeroed so U_full = I
        at gen 0.
        """
        scheme = str(getattr(self.cfg, "mu_init_scheme", "uniform")).lower()
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons
        if scheme == "uniform":
            mu = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        elif scheme == "glorot":
            mu = np.zeros(self.dim, dtype=np.float32)
            c_W0 = float(np.sqrt(6.0 / (Q + H)))
            c_W1 = float(np.sqrt(6.0 / (H + 1)))

            def _fill_ann(off: int) -> int:
                """Fill one ANN's worth of weights starting at `off`.
                Layout: [W0(T,Q,H) | b0(T,H) | W1(T,H) | b1(1)].
                """
                n_W0 = T * Q * H
                mu[off:off + n_W0] = rng.uniform(
                    -c_W0, c_W0, size=n_W0).astype(np.float32)
                off += n_W0
                # b0 zero (T*H entries)
                off += T * H
                n_W1 = T * H
                mu[off:off + n_W1] = rng.uniform(
                    -c_W1, c_W1, size=n_W1).astype(np.float32)
                off += n_W1
                # b1 zero (1 entry)
                off += 1
                return off

            off = _fill_ann(0)
            if self.cfg.target_mode == 2:
                off = _fill_ann(off)
        else:
            raise ValueError(
                f"mu_init_scheme={scheme!r} not in ('uniform', 'glorot')")
        # V_pair tail: always zero (residual mixing layer; U_full = I at init).
        if self.n_U_pair > 0:
            mu[self.n_anns_total:] = 0.0
        return mu

    def _maybe_adapt_lambda(self, gen: int, data_loss: float,
                            l1: float, l2: float, l_orth: float = 0.0) -> None:
        """Rescale lambda_1 / lambda_2 / lambda_orth toward
        `target_ratio · data_loss`.

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
            l_orth    : current orthogonal penalty (mode=="orthogonal").
        """
        if not (self._dyn_lambda_1 or self._dyn_lambda_2 or self._dyn_lambda_orth):
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
        if self._dyn_lambda_orth and l_orth > 1e-12:
            ratio = (target * ref) / float(l_orth)
            new = float(self._lambda_orth.numpy()) * (ratio ** damping)
            self._lambda_orth.assign(float(np.clip(new, lmin, lmax)))

    def _orth_penalty_slab(self, V_tail: tf.Tensor,
                           slab_idx: int = 0) -> tf.Tensor:
        """Compute Σ_p ‖U_pᵀ U_p − I‖²_F for the pair blocks in one
        type-slab of V_tail. In residual form `U = I + V`:

            UᵀU − I = V + Vᵀ + VᵀV

        so the penalty is `‖V + Vᵀ + VᵀV‖²_F` per pair block, summed.
        Block sizes vary (trivial compression), so we loop over pairs
        and pick out each block's `bs²` entries from the flat tail.

        Args:
            V_tail   : [..., n_U_pair] flat residual tail (last axis
                       carries the param entries; any leading axes are
                       broadcast — supports scalar `[n_U_pair]`,
                       population `[P, n_U_pair]`, etc.)
            slab_idx : which T-slab to read (0 for shared U_pair).

        Returns:
            penalty  : `[...]` (leading axes preserved). Normalised by
                       the number of entries in this slab so the value
                       scales like an averaged squared-residual.
        """
        # Cayley parameterisation owns the orthogonality constraint
        # structurally — the V_tail under Cayley contains upper-
        # triangle entries of A (skew-symmetric), NOT the dense V
        # blocks this penalty assumes. Indexing it as bs² per block
        # would silently mis-slice. Hard-fail to catch any future
        # code path that calls into this helper while cayley is on.
        if self._mix_cayley:
            raise RuntimeError(
                "_orth_penalty_slab is not valid under "
                "descriptor_mixing_regularizer='cayley': the V_tail "
                "encodes the upper-triangle of skew-symmetric A, not "
                "dense V blocks. Cayley provides orthogonality as a "
                "structural constraint, so no soft penalty is needed.")
        per_T = (self.n_U_pair // self.cfg.num_types
                 if self._mix_per_type else self.n_U_pair)
        start = slab_idx * per_T
        pen = tf.zeros(tf.shape(V_tail)[:-1])
        offset = 0
        if self._mix_arch == "linear":
            iterator = [(bs,) for bs in self._mix_block_sizes]
        elif self._mix_arch == "l_aware":
            # Each pair contributes L sub-blocks of [α_p × α_p].
            iterator = [(alpha,) for alpha in self._mix_alpha_per_pair
                        for _ in range(self._mix_L)]
        else:  # cross_pair_l: one [N_l × N_l] sub-block per angular momentum
            iterator = [(self._mix_N_per_l,) for _ in range(self._mix_L)]
        for (dim,) in iterator:
            n = dim * dim
            block_flat = V_tail[..., start + offset:start + offset + n]
            new_shape = tf.concat(
                [tf.shape(block_flat)[:-1], [dim, dim]], axis=0)
            V = tf.reshape(block_flat, new_shape)              # [..., dim, dim]
            VtV = tf.matmul(V, V, transpose_a=True)            # [..., dim, dim]
            M = V + tf.linalg.matrix_transpose(V) + VtV
            pen = pen + tf.reduce_sum(tf.square(M), axis=[-2, -1])
            offset += n
        return pen / float(per_T)

    def _orth_penalty_total(self, V_tail: tf.Tensor) -> tf.Tensor:
        """Sum the orthogonal penalty across all T slabs (per-type)
        or compute it once (shared). Output preserves leading dims of
        V_tail. Average across T (per-type) matches the ANN per-type
        convention used by the L1/L2 path.
        """
        T = self.cfg.num_types
        if self._mix_per_type:
            slabs = [self._orth_penalty_slab(V_tail, t) for t in range(T)]
            return tf.add_n(slabs) / float(T)
        return self._orth_penalty_slab(V_tail, 0)

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
        U = tf.linalg.solve(I - A, I + A)
        return U - I

    def _cayley_block(self, A_upper_flat: tf.Tensor, bs: int) -> tf.Tensor:
        """Single-block Cayley helper. Thin wrapper over the batched
        variant. Retained for callers that genuinely need one block
        (e.g. the non-uniform `linear`-arch fallback). Hot paths should
        use `_cayley_blocks_batched` directly.
        """
        if bs <= 1:
            shape = tf.concat(
                [tf.shape(A_upper_flat)[:-1], [bs, bs]], axis=0)
            return tf.zeros(shape, dtype=tf.float32)
        stacked = A_upper_flat[..., tf.newaxis, :]   # [..., 1, payload]
        V = self._cayley_blocks_batched(stacked, bs)
        return V[..., 0, :, :]

    def _extract_type_params(self, pv: tf.Tensor, t: int) -> tf.Tensor:
        """Extract parameters belonging to atom type t from flat vector.

        Parameter layout: [W0(T,Q,H) | b0(T,H) | W1(T,H) | b1(1)]
        Per-type slice: W0[t,:,:] + b0[t,:] + W1[t,:]

        Args:
            pv : [dim] flat parameter vector
            t  : type index (0 to num_types-1)

        Returns:
            [Q*H + H + H] tensor — concatenated type-t parameters
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

        # W0 block: [T, Q, H] flattened → stride T*Q*H, type t starts at t*Q*H
        w0_start = t * Q * H
        w0_end = w0_start + Q * H

        # b0 block: after all W0, [T, H] → type t at offset T*Q*H + t*H
        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H

        # W1 block: after b0, [T, H] → type t at offset T*Q*H + T*H + t*H
        w1_offset = b0_offset + T * H
        w1_start = w1_offset + t * H
        w1_end = w1_start + H

        return tf.concat([pv[w0_start:w0_end], pv[b0_start:b0_end], pv[w1_start:w1_end]], axis=0)

    def _build_type_of_variable(self) -> np.ndarray:
        """Build array mapping each parameter index to its atom type.

        Layout per ANN: [W0(T,Q,H) | b0(T,H) | W1(T,H) | b1(1)]
        W0/b0/W1 params for type t get label t. b1 gets label T (global).

        Returns:
            [dim] int array — type label per variable
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

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
        if self.n_U_pair > 0:
            if self._mix_per_type:
                # Per-T slab size is the same across all arches: n_U_pair
                # is by construction T · per_T. Using `n_U_pair // T`
                # keeps this correct for linear, l_aware, and
                # cross_pair_l. (Earlier `sum(bs² for bs in
                # _mix_block_sizes)` was the linear-only formula and
                # silently mis-labelled slabs for the other arches.)
                per_T = self.n_U_pair // T
                pair_labels = np.empty(self.n_U_pair, dtype=np.int32)
                for t_idx in range(T):
                    pair_labels[t_idx * per_T:(t_idx + 1) * per_T] = t_idx
            else:
                pair_labels = np.full(self.n_U_pair, T, dtype=np.int32)
            return np.concatenate([ann_tov, pair_labels])
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
        P = self.pop_size
        Q = self.dim_q
        H = self.cfg.num_neurons
        n_per_type = Q * H + H + H

        # Optional V_pair tail routing. With per-type V_pair, slab t
        # is owned by label t (per-type ranking); with shared V_pair,
        # the whole tail is owned by the global label T. Dispatched on
        # cfg.descriptor_mixing_regularizer: "off" leaves V_pair alone,
        # "shrinkage" adds L1+L2 of the slab, "orthogonal" adds
        # λ_orth · ‖UᵀU - I‖² per block.
        reg_Vpair = (self.n_U_pair > 0 and self._mix_reg_mode == "shrinkage")
        reg_Vorth = (self.n_U_pair > 0 and self._mix_reg_mode == "orthogonal")
        if reg_Vpair or reg_Vorth:
            V_tail = samples[:, self.n_anns_total:]
            per_T_V = self.n_U_pair // T if self._mix_per_type else None

        # Add per-type regularization to per-type RMSE → [T+1] fitness values
        fitness_per_type = []
        for t in range(T):
            type_params = self._extract_type_params_batched(samples, t)  # [P, n_per_type]
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(type_params), axis=1) / n_per_type
            l2 = self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(type_params), axis=1) / n_per_type)
            if reg_Vpair and self._mix_per_type:
                # Slab t of V_pair routes to label t (shrinkage).
                slab = V_tail[:, t * per_T_V:(t + 1) * per_T_V]
                l1 = l1 + self.lambda_1 * tf.reduce_sum(tf.abs(slab), axis=1) / per_T_V
                l2 = l2 + self.lambda_2 * tf.sqrt(
                    tf.reduce_sum(tf.square(slab), axis=1) / per_T_V)
            if reg_Vorth and self._mix_per_type:
                # Slab t orth penalty routes to label t.
                orth_t = self._orth_penalty_slab(V_tail, slab_idx=t)
                l2 = l2 + self._lambda_orth * orth_t
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
        if reg_Vpair and not self._mix_per_type:
            # Shared V_pair routes to the global label (shrinkage).
            global_l1 = global_l1 + self.lambda_1 * tf.reduce_sum(
                tf.abs(V_tail), axis=1) / self.n_U_pair
            global_l2 = global_l2 + self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(V_tail), axis=1) / self.n_U_pair)
        if reg_Vorth and not self._mix_per_type:
            global_l2 = global_l2 + self._lambda_orth * self._orth_penalty_total(V_tail)
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

        Utilities are log-shaped and zero-centred so that top-ranked
        individuals contribute positive gradient and bottom-ranked
        contribute negative.  Computed once at init, stored as tf.constant.

        Returns:
            utilities : ndarray [pop_size] — weights indexed by rank (0 = best).
        """

        lam = self.pop_size

        ranks = np.arange(lam) + 1

        raw = np.log((lam * 0.5) + 1.0) - np.log(ranks)
        raw = np.maximum(0.0, raw)

        # Normalise to sum=1, then shift to zero-mean
        total = np.sum(raw)
        if total > 0:
            raw /= total
        else:
            if self.cfg.debug:
                print("Utility calc failed due to negative total")
        utilities = raw - 1.0 / lam
        if self.cfg.debug:
            print("utilities = ", utilities)
        return utilities

    def ask(self) -> tuple[tf.Tensor, tf.Tensor]:
        """Sample pop_size candidate parameter vectors from N(mu, diag(sigma^2)).

        Uses **mirrored (antithetic) sampling**: draws pop_size/2 independent
        noise vectors ε_i and pairs each with its mirror −ε_i. The
        first-order Taylor noise in the mean update Σ u_i s_i then
        cancels per pair exactly when u_+ = −u_−, which is what
        Hansen's zero-mean utility shaping produces by construction.
        Net: same evaluation cost, ~1.5–2× lower mean-update variance
        (Salimans et al. 2017; Brockhoff et al. 2010).

        When pop_size is odd we use floor(pop_size/2) pairs plus one
        standalone i.i.d. sample so the total population count is preserved.

        All operations run on GPU via TensorFlow.

        Returns:
            samples : [pop_size, dim] float32 tensor — candidate parameter vectors
            s       : [pop_size, dim] float32 tensor — standard normal noise used
        """
        half = self.pop_size // 2
        s_half = self.tf_rng.normal(shape=(half, self.dim))
        if self.pop_size % 2 == 0:
            s = tf.concat([s_half, -s_half], axis=0)
        else:
            s_extra = self.tf_rng.normal(shape=(1, self.dim))
            s = tf.concat([s_half, -s_half, s_extra], axis=0)
        samples = self.mu + s * self.sigma
        return samples, s

    def update(self, utilities: tf.Tensor, s: tf.Tensor) -> None:
        """Update mu and sigma using fitness-ranked noise vectors.

        All operations run on GPU via TensorFlow.

        Args:
            utilities : [pop_size] float32 tensor — rank-based weights (best first)
            s         : [pop_size, dim] float32 tensor — noise vectors sorted by fitness
                        (s[0] = noise of best individual, s[-1] = worst)

        Mutates self.mu and self.sigma tf.Variables in place. Sigma is
        clamped to a small floor (cfg.sigma_floor, default 1e-5) after
        the multiplicative update so a near-collapse of the search
        distribution can't silently kill exploration on long runs.
        """
        grad_mu = tf.einsum('p,pd->d', utilities, s)
        grad_sigma = tf.einsum('p,pd->d', utilities, s ** 2 - 1.0)

        self.mu.assign_add(self.sigma * grad_mu)
        floor = getattr(self.cfg, "sigma_floor", 1e-5)
        new_sigma = self.sigma * tf.exp(self.eta_sigma * grad_sigma)
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
            best_sigma = tf.constant(resume_state["best_sigma"], dtype=tf.float32)
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
        # Plateau-driven sigma resets (IPOP-style restart, simplified):
        # tracks how many resets have already been fired so we can cap
        # via cfg.max_sigma_resets. Reset counter is per-run (not
        # restored from checkpoint — a fresh attempt to escape any
        # plateau seen so far is fine on resume).
        n_sigma_resets = 0

        gen_l1, gen_l2, gen_lorth = 0.0, 0.0, 0.0
        val_fitness = float('inf')
        sigma_min = sigma_max = sigma_mean = sigma_median = float(cfg.init_sigma)

        for gen in range(start_gen, cfg.num_generations):
            t0 = time.perf_counter()

            samples, s = self.ask()

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
                if train_data.get("_gv_disk_backed", False):
                    # Disk-backed: read this batch's pair slice from the
                    # memmap. ~85 MB at batch=50 / fp32 / Q=645 — one DMA
                    # into the tf.constant; the chunk loop then gathers
                    # that batch tensor per chunk in-GPU.
                    flat_pair_idx_np = flat_pair_idx.numpy()
                    batch_data["grad_values"] = tf.constant(
                        np.asarray(gv_full[flat_pair_idx_np]))
                else:
                    batch_data["grad_values"] = tf.gather(gv_full, flat_pair_idx)
                batch_data["pair_atom"]   = tf.gather(train_data["pair_atom"],   flat_pair_idx)
                batch_data["pair_gidx"]   = tf.gather(train_data["pair_gidx"],   flat_pair_idx)
                batch_data["pair_struct"] = tf.cast(pair_ranges.value_rowids(), tf.int32)
                # Build batch-local struct_ptr for struct_chunk slicing
                batch_pair_counts = tf.cast(pair_ranges.row_lengths(), tf.int32)
                batch_data["struct_ptr"] = tf.concat(
                    [[0], tf.cumsum(batch_pair_counts)], axis=0)

            t1 = time.perf_counter()

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
            # drives SNES ranking (depends on loss_type); the rmse/rrmse
            # entries are ALWAYS computed from squared error so they're
            # comparable across loss-function ablations.
            rmse_pc = self._last_rmse_per_cand
            rrmse_pc = self._last_rrmse_per_cand
            metrics_gpu = tf.stack([
                tf.reduce_mean(fitness),
                tf.reduce_min(rmse_pc),
                tf.reduce_max(rmse_pc),
                tf.reduce_min(rrmse_pc),
                tf.reduce_mean(rrmse_pc),
            ])
            metrics_np = metrics_gpu.numpy()
            avg_fitness = float(metrics_np[0])
            best_rmse = float(metrics_np[1])
            worst_rmse = float(metrics_np[2])
            best_rrmse = float(metrics_np[3])
            avg_rrmse = float(metrics_np[4])

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
            need_adapt = (self._dyn_lambda_1 or self._dyn_lambda_2
                          or self._dyn_lambda_orth)
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

            # Rank and update (GPU)
            if self._per_type:
                s_sorted = self._build_per_type_gradients(s, fitness_per_type_rmse, samples)
            else:
                ranks = tf.argsort(fitness)
                s_sorted = tf.gather(s, ranks)
            self.update(self.utilities, s_sorted)

            t3 = time.perf_counter()

            # Validate with updated mean (skip on non-val generations)
            _do_val = (gen % cfg.val_interval == 0) or (gen == cfg.num_generations - 1)
            if _do_val:
                val_fitness = self.validate(val_data, self.mu)

            t4 = time.perf_counter()

            # Sigma stats: sample every 100 gens to avoid GPU→CPU transfer.
            # Cached scalars are reused on the off-cycles, so the value
            # appended at the next val tick reflects the most recent
            # sample within at most 100 gens of staleness.
            if gen % 100 == 0:
                sigma_np = self.sigma.numpy()
                sigma_min = float(np.min(sigma_np))
                sigma_max = float(np.max(sigma_np))
                sigma_mean = float(np.mean(sigma_np))
                sigma_median = float(np.median(sigma_np))

            # History is recorded once per val_interval (plus the final
            # generation). Off-val gens contribute only to the progress
            # bar and the early-stopping counter.
            if _do_val:
                history["generation"].append(gen)
                history["train_loss"].append(avg_fitness)
                history["val_loss"].append(val_fitness)
                history["L1"].append(gen_l1)
                history["L2"].append(gen_l2)
                history.setdefault("L_orth", []).append(gen_lorth)
                history["best_rmse"].append(best_rmse)
                history["worst_rmse"].append(worst_rmse)
                # RMSE / RRMSE always reported, independent of loss_type.
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
                    f"train RMSE: {avg_fitness:.6f}  "
                    f"val RMSE: {val_fitness:.6f}  "
                    f"best val RMSE: {best_val_loss:.6f}  "
                    f"elapsed: {elapsed_str}  ETA: {eta_str}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
                line += f"  best_RMSE: {best_rmse:.6f}  best_RRMSE: {best_rrmse:.6f}"
                if self._mix_reg_mode == "orthogonal":
                    line += f"  L_orth: {gen_lorth:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # Early stopping (only update on val generations)
            if _do_val:
                if val_fitness < best_val_loss:
                    best_val_loss = val_fitness
                    best_mu = tf.identity(self.mu)
                    best_sigma = tf.identity(self.sigma)
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1

            # Plateau-triggered sigma re-broadening (soft restart).
            # Checked only on val gens — gens_without_improvement
            # increments per val tick, so the patience here is in
            # units of val ticks, not raw gens.
            #
            # Two modes, controlled by cfg.sigma_reset_to_init:
            #
            # (A) "multiply" (default, sigma_reset_to_init=False):
            #     σ ← σ · sigma_reset_factor   (elementwise on the
            #     current sigma vector). Preserves the per-dimension
            #     scale structure SNES has learned, just re-broadens
            #     each dim uniformly. This is the better choice in
            #     high dim because a uniform fresh sigma loses all
            #     direction information.
            #
            # (B) "to_init" (sigma_reset_to_init=True):
            #     σ ← init_sigma · sigma_reset_factor (uniform).
            #     IPOP-style hard reset. Use only when you have a
            #     specific reason to discard learned per-dim scales.
            #
            # μ restoration (plateau_restore_best_mu) is independent
            # and defaults to False — leaving μ where the search has
            # reached usually beats teleporting back to best_μ
            # because the broadened σ around best_μ has no learned
            # direction info to follow.
            reset_patience = getattr(cfg, "plateau_reset_patience", None)
            max_resets = getattr(cfg, "max_sigma_resets", None)
            if (_do_val
                    and reset_patience is not None
                    and gens_without_improvement >= int(reset_patience)
                    and (max_resets is None or n_sigma_resets < int(max_resets))):
                factor = float(getattr(cfg, "sigma_reset_factor", 2.0))
                to_init = bool(getattr(cfg, "sigma_reset_to_init", False))
                if to_init:
                    self.sigma.assign(
                        tf.fill([self.dim], float(cfg.init_sigma) * factor))
                    mode_str = f"σ ← init·{factor:.2f}"
                else:
                    self.sigma.assign(self.sigma * factor)
                    mode_str = f"σ ← σ·{factor:.2f} (preserves per-dim scale)"
                restore_mu = bool(getattr(cfg, "plateau_restore_best_mu", False))
                if restore_mu:
                    self.mu.assign(best_mu)
                gens_without_improvement = 0
                n_sigma_resets += 1
                # Read back current sigma stats for the log line so
                # the user can see what actually happened.
                s_now = self.sigma.numpy()
                s_min = float(np.min(s_now))
                s_mean = float(np.mean(s_now))
                s_max = float(np.max(s_now))
                sys.stdout.write(
                    f"\n  plateau detected at gen {gen + 1}: {mode_str}"
                    f" (σ now min/mean/max = {s_min:.4f}/{s_mean:.4f}/{s_max:.4f})"
                    f" — reset #{n_sigma_resets}"
                    + (f"/{max_resets}" if max_resets is not None else "")
                    + (", μ restored to best" if restore_mu else "")
                    + f", best_val={best_val_loss:.6f}\n")
                sys.stdout.flush()

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
                save_checkpoint(ckpt_path, cfg, {
                    "mu": self.mu, "sigma": self.sigma,
                    "best_mu": best_mu, "best_sigma": best_sigma,
                    "best_val_loss": best_val_loss,
                    "gens_without_improvement": gens_without_improvement,
                    "tf_rng_state": self.tf_rng.state,
                }, history, gen)
                # Print a one-line note above the in-place progress bar.
                sys.stdout.write(
                    f"\n  checkpoint saved at gen {gen + 1} → {ckpt_path}\n")
                sys.stdout.flush()

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
        from data import slice_and_complete_chunk
        S_val = val_data["num_atoms"].shape[0]

        # Resolve subsample indices once. None = full val set.
        if self.cfg.val_size is None:
            val_idx_tf = tf.range(S_val, dtype=tf.int32)
        else:
            val_idx_tf = tf.cast(
                tf.argsort(self.tf_rng.uniform(shape=[S_val]))[:self.cfg.val_size],
                tf.int32)

        if mu_tf is not None:
            params = self.reconstruct_params_tf(mu_tf)
            # Tail of reconstruct_params_tf can be 4 (ANN-only), 5 (ANN + U_pair),
            # 8 (ANN + pol ANN), or 9 (ANN + pol ANN + U_pair). U_pair is the
            # last element when descriptor mixing is enabled.
            has_U = self.n_U_pair > 0
            U_pair_val = params[-1] if has_U else None
            head = params[:-1] if has_U else params
            if self.cfg.target_mode == 2:
                W0, b0, W1, b1, W0p, b0p, W1p, b1p = head
            else:
                W0, b0, W1, b1 = head
                W0p = b0p = W1p = b1p = None
            # Absorb U_pair^T into W0 (and W0_pol) once. predict_batch
            # is a pure forward over the supplied weights (it does not
            # call _W0_eff itself), so callers are responsible for
            # passing U-folded weights — done here for the mu_tf path
            # and in TNEP.score / TNEP.predict for the model-weights
            # path.
            if U_pair_val is not None:
                W0 = self.model._W0_eff(W0, U_pair_val)
                if W0p is not None:
                    W0p = self.model._W0_eff(W0p, U_pair_val)
        else:
            W0, b0, W1, b1 = self.model.W0, self.model.b0, self.model.W1, self.model.b1
            W0p = getattr(self.model, 'W0_pol', None)
            b0p = getattr(self.model, 'b0_pol', None)
            W1p = getattr(self.model, 'W1_pol', None)
            b1p = getattr(self.model, 'b1_pol', None)
            # If the model itself has descriptor mixing, fold U_pair^T
            # into the model's W0 / W0_pol for the validate forward.
            if getattr(self.model, "descriptor_mixing", False):
                W0 = self.model._W0_eff(W0)
                if W0p is not None:
                    W0p = self.model._W0_eff(W0p)

        # Streaming chunk loop honours batch_chunk_size so disk reads (when
        # cache_gradients_to_disk is set) and tensor materialisation stay
        # bounded. Full-val (val_size=None) takes the contiguous-range path
        # via `prefetched_chunks`; a random val subset falls back to the
        # per-call slice path.
        from data import slice_and_complete_chunk, prefetched_chunks
        N_idx = int(val_idx_tf.shape[0])
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else N_idx

        diff_sq_sum = tf.constant(0.0, dtype=tf.float32)
        diff_count  = tf.constant(0.0, dtype=tf.float32)

        def _consume(chunk):
            nonlocal diff_sq_sum, diff_count
            preds = self.model.predict_batch(
                chunk["descriptors"], chunk["grad_values"],
                chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
                chunk["positions"], chunk["Z_int"], chunk["boxes"],
                chunk["atom_mask"],
                W0, b0, W1, b1, W0p, b0p, W1p, b1p,
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
            for _, _, chunk in prefetched_chunks(
                    val_data, ranges,
                    pin_to_cpu=self.cfg.pin_data_to_cpu,
                    enabled=getattr(self.cfg, "chunk_prefetch", True),
                    depth=getattr(self.cfg, "prefetch_depth", 1)):
                _consume(chunk)
                del chunk
        else:
            for s_start in range(0, N_idx, struct_chunk):
                s_end = min(s_start + struct_chunk, N_idx)
                sub_idx = val_idx_tf[s_start:s_end]
                chunk = slice_and_complete_chunk(val_data, sub_idx)
                if self.cfg.pin_data_to_cpu:
                    with tf.device('/GPU:0'):
                        chunk = {k: (tf.identity(v) if not k.startswith("_") else v)
                                 for k, v in chunk.items()}
                _consume(chunk)
                del chunk

        rmse = tf.sqrt(tf.maximum(diff_sq_sum / tf.maximum(diff_count, 1.0), 0.0))
        return float(rmse)

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
        H = self.cfg.num_neurons

        n_W0 = T * Q * H
        n_b0 = T * H
        n_W1 = T * H
        n_b1 = 1

        is_batched = len(param_vectors.shape) == 2

        def _extract(pv, offset):
            W0 = tf.reshape(pv[..., offset:offset + n_W0],
                            [-1, T, Q, H] if is_batched else [T, Q, H])
            offset += n_W0
            b0 = tf.reshape(pv[..., offset:offset + n_b0],
                            [-1, T, H] if is_batched else [T, H])
            offset += n_b0
            W1 = tf.reshape(pv[..., offset:offset + n_W1],
                            [-1, T, H] if is_batched else [T, H])
            offset += n_W1
            b1 = pv[..., offset]  # [P] or scalar
            offset += n_b1
            return W0, b0, W1, b1, offset

        W0, b0, W1, b1, offset = _extract(param_vectors, 0)

        if self.cfg.target_mode == 2:
            W0p, b0p, W1p, b1p, offset = _extract(param_vectors, offset)
            tail = (W0, b0, W1, b1, W0p, b0p, W1p, b1p)
        else:
            tail = (W0, b0, W1, b1)

        if self.n_U_pair > 0:
            # Unpack the descriptor-mixing tail. Layout depends on
            # `_mix_arch`. For both arches the per-type variant has T
            # contiguous slabs in the flat tail (same convention as
            # the ANN W0). Inside a slab, the inner layout differs:
            #   "linear"  : [V_p=0 (bs²) | V_p=1 (bs²) | ... ]
            #   "l_aware" : [V_{p=0,l=0} (α²) | V_{p=0,l=1} (α²) | ...
            #                | V_{p=1,l=0} (α²) | ... ]
            U_flat = param_vectors[..., offset:offset + self.n_U_pair]
            per_type = self._mix_per_type

            if self._mix_arch == "cross_pair_l":
                N_l = self._mix_N_per_l
                L = self._mix_L
                # Per-block payload: dense N_l² entries normally, or
                # only the skew-symmetric upper triangle (N_l(N_l-1)/2)
                # under Cayley parameterisation. Cayley reconstructs
                # the dense V = U_cayley − I inline.
                block_payload = (N_l * (N_l - 1) // 2) if self._mix_cayley else N_l * N_l
                per_T_block = L * block_payload

                def _extract_t_block_cross_pair_l(t_idx: int):
                    """Build [L, N_l, N_l] from slab t. Optional leading
                    batch axis [P] when input is batched.

                    Cayley fast path: reshape the whole slab as
                    [..., L, payload] and run ONE batched solve for all
                    L blocks at once. With non-cayley parameterisation,
                    a single reshape is enough.
                    """
                    start = t_idx * per_T_block
                    slab = U_flat[..., start:start + per_T_block]
                    if self._mix_cayley:
                        # slab: [..., L * payload] → [..., L, payload]
                        new_shape = tf.concat(
                            [tf.shape(slab)[:-1], [L, block_payload]],
                            axis=0)
                        stacked = tf.reshape(slab, new_shape)
                        # One batched solve over L blocks; output
                        # [..., L, N_l, N_l].
                        return self._cayley_blocks_batched(stacked, N_l)
                    if is_batched:
                        return tf.reshape(slab, [-1, L, N_l, N_l])
                    return tf.reshape(slab, [L, N_l, N_l])

                if per_type:
                    per_t = [_extract_t_block_cross_pair_l(t) for t in range(T)]
                    stack_axis = 1 if is_batched else 0
                    U_pair = tf.stack(per_t, axis=stack_axis)
                else:
                    U_pair = _extract_t_block_cross_pair_l(0)
                tail = tail + (U_pair,)
                return tail
            if self._mix_arch == "linear":
                max_bs = max(self._mix_block_sizes)
                # Per-pair payload: bs² normally; bs(bs-1)/2 for Cayley.
                bs_payloads = [
                    (bs * (bs - 1) // 2) if self._mix_cayley else bs * bs
                    for bs in self._mix_block_sizes
                ]
                per_T_block = sum(bs_payloads)

                # Cayley fast path: when all pairs share the same bs,
                # collapse the whole slab into ONE batched solve.
                uniform_bs = (
                    self._mix_cayley
                    and len(set(self._mix_block_sizes)) == 1)

                def _extract_t_block_linear(t_idx: int):
                    start = t_idx * per_T_block
                    slab = U_flat[..., start:start + per_T_block]

                    if uniform_bs:
                        bs0 = self._mix_block_sizes[0]
                        payload = bs_payloads[0]
                        n_pairs = len(self._mix_block_sizes)
                        new_shape = tf.concat(
                            [tf.shape(slab)[:-1], [n_pairs, payload]],
                            axis=0)
                        stacked = tf.reshape(slab, new_shape)
                        V = self._cayley_blocks_batched(stacked, bs0)
                        pad_r = max_bs - bs0
                        if pad_r > 0:
                            paddings = [[0, 0]] * (len(V.shape) - 2) + \
                                       [[0, pad_r], [0, pad_r]]
                            V = tf.pad(V, paddings)
                        return V

                    pair_blocks: list = []
                    inner_offset = 0
                    for bs, payload in zip(self._mix_block_sizes, bs_payloads):
                        block_flat = slab[..., inner_offset:inner_offset + payload]
                        inner_offset += payload
                        pad_r = max_bs - bs
                        if self._mix_cayley:
                            block = self._cayley_block(block_flat, bs)
                        elif is_batched:
                            block = tf.reshape(block_flat, [-1, bs, bs])
                        else:
                            block = tf.reshape(block_flat, [bs, bs])
                        # Pad to max_bs for uniform-stride U_pair storage.
                        if is_batched:
                            block = tf.pad(block, [[0, 0], [0, pad_r], [0, pad_r]])
                        else:
                            block = tf.pad(block, [[0, pad_r], [0, pad_r]])
                        pair_blocks.append(block)
                    stack_axis = -3 if is_batched else 0
                    return tf.stack(pair_blocks, axis=stack_axis)

                if per_type:
                    per_t = [_extract_t_block_linear(t) for t in range(T)]
                    stack_axis = 1 if is_batched else 0
                    U_pair = tf.stack(per_t, axis=stack_axis)
                else:
                    U_pair = _extract_t_block_linear(0)
            else:  # l_aware
                max_alpha = max(self._mix_alpha_per_pair)
                L = self._mix_L
                # Per-(pair, l) payload: α² normally; α(α-1)/2 for Cayley.
                alpha_payloads = [
                    (a * (a - 1) // 2) if self._mix_cayley else a * a
                    for a in self._mix_alpha_per_pair
                ]
                per_T_block = sum(L * p for p in alpha_payloads)

                # Detect the common case: all pairs share the same α.
                # Then the slab is a uniform [num_pairs, L, payload] grid
                # and we can do ONE batched solve for all num_pairs*L
                # Cayley blocks — ~10× faster than the per-block loop.
                uniform_alpha = (
                    self._mix_cayley
                    and len(set(self._mix_alpha_per_pair)) == 1)

                def _extract_t_block_l_aware(t_idx: int):
                    """Build [num_pairs, L, max_α, max_α] from slab t."""
                    start = t_idx * per_T_block
                    slab = U_flat[..., start:start + per_T_block]

                    if uniform_alpha:
                        alpha_p = self._mix_alpha_per_pair[0]
                        payload = alpha_payloads[0]
                        n_pairs = len(self._mix_alpha_per_pair)
                        n_blocks = n_pairs * L
                        # slab: [..., n_pairs*L*payload]
                        # → [..., n_blocks, payload]
                        new_shape = tf.concat(
                            [tf.shape(slab)[:-1], [n_blocks, payload]],
                            axis=0)
                        stacked = tf.reshape(slab, new_shape)
                        V = self._cayley_blocks_batched(stacked, alpha_p)
                        # V: [..., n_blocks, α, α] → [..., n_pairs, L, α, α]
                        pad_r = max_alpha - alpha_p
                        if pad_r > 0:
                            paddings = [[0, 0]] * (len(V.shape) - 2) + \
                                       [[0, pad_r], [0, pad_r]]
                            V = tf.pad(V, paddings)
                        out_shape = tf.concat(
                            [tf.shape(V)[:-3], [n_pairs, L, max_alpha,
                                                max_alpha]], axis=0)
                        return tf.reshape(V, out_shape)

                    # Fallback: non-uniform α across pairs. Group by α.
                    pair_blocks: list = []
                    inner_offset = 0
                    for alpha_p, payload in zip(self._mix_alpha_per_pair,
                                                  alpha_payloads):
                        # For one pair: L sub-blocks each [α_p × α_p].
                        pair_payload = L * payload
                        sub = slab[..., inner_offset:inner_offset + pair_payload]
                        inner_offset += pair_payload
                        pad_r = max_alpha - alpha_p
                        if self._mix_cayley:
                            new_shape = tf.concat(
                                [tf.shape(sub)[:-1], [L, payload]], axis=0)
                            stacked = tf.reshape(sub, new_shape)
                            l_block = self._cayley_blocks_batched(
                                stacked, alpha_p)
                        else:
                            new_shape = tf.concat(
                                [tf.shape(sub)[:-1],
                                 [L, alpha_p, alpha_p]], axis=0)
                            l_block = tf.reshape(sub, new_shape)
                        if pad_r > 0:
                            paddings = [[0, 0]] * (len(l_block.shape) - 2) + \
                                       [[0, pad_r], [0, pad_r]]
                            l_block = tf.pad(l_block, paddings)
                        pair_blocks.append(l_block)
                    # Stack pairs along the num_pairs axis.
                    stack_axis_p = -4 if is_batched else 0
                    return tf.stack(pair_blocks, axis=stack_axis_p)

                if per_type:
                    per_t = [_extract_t_block_l_aware(t) for t in range(T)]
                    stack_axis = 1 if is_batched else 0
                    U_pair = tf.stack(per_t, axis=stack_axis)
                else:
                    U_pair = _extract_t_block_l_aware(0)
            tail = tail + (U_pair,)

        return tail

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
        Q = self.dim_q
        H = self.cfg.num_neurons

        reg_Vpair = (self.n_U_pair > 0 and self._mix_reg_mode == "shrinkage")
        reg_Vorth = (self.n_U_pair > 0 and self._mix_reg_mode == "orthogonal")

        if T > 1:
            n_per_type = Q * H + H + H  # W0_t + b0_t + W1_t

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
            # Single-type path: keep V_pair out of the main sum whenever
            # it's handled separately (shrinkage, orth, OR Cayley).
            # Cayley's A entries must NEVER be L1/L2-regularised — that
            # would pull A → 0 → U → I and collapse the rotation.
            v_handled = reg_Vpair or reg_Vorth or self._mix_cayley
            if v_handled:
                ann = param_vectors[:, :self.n_anns_total]
                ann_n = self.n_anns_total
            else:
                ann = param_vectors
                ann_n = self.dim
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(ann), axis=1) / ann_n
            l2 = self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(ann), axis=1) / ann_n)
            reg = l1 + l2

        # V_pair shrinkage path (mode=="shrinkage"). See
        # compute_regularization() for the rationale.
        if reg_Vpair:
            tail = param_vectors[:, self.n_anns_total:]
            if self._mix_per_type and T > 1:
                per_T = self.n_U_pair // T
                vp_l1 = tf.zeros([tf.shape(param_vectors)[0]])
                vp_l2 = tf.zeros([tf.shape(param_vectors)[0]])
                for t in range(T):
                    slab = tail[:, t * per_T:(t + 1) * per_T]
                    vp_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(slab), axis=1) / per_T
                    vp_l2 += self.lambda_2 * tf.sqrt(
                        tf.reduce_sum(tf.square(slab), axis=1) / per_T)
                reg = reg + vp_l1 / T + vp_l2 / T
            else:
                vp_l1 = self.lambda_1 * tf.reduce_sum(tf.abs(tail), axis=1) / self.n_U_pair
                vp_l2 = self.lambda_2 * tf.sqrt(
                    tf.reduce_sum(tf.square(tail), axis=1) / self.n_U_pair)
                reg = reg + vp_l1 + vp_l2

        # V_pair orthogonal path (mode=="orthogonal"). One scalar
        # penalty per candidate, summed/averaged over slabs. Uses its
        # own lambda so it can be dialled independently.
        if reg_Vorth:
            tail = param_vectors[:, self.n_anns_total:]      # [P, n_U_pair]
            reg = reg + self._lambda_orth * self._orth_penalty_total(tail)

        return reg

    def _extract_type_params_batched(self, param_vectors: tf.Tensor, t: int) -> tf.Tensor:
        """Extract type-t parameters from batched flat vectors.

        Args:
            param_vectors : [P, dim] float32
            t             : type index

        Returns:
            [P, Q*H + H + H] float32 — type-t params for each candidate
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons

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
        loss_type = self.cfg.loss_type
        inv_mode = str(getattr(self.cfg, "inverse_weight_mode", "none")).lower()
        if inv_mode not in ("none", "vector_magnitude", "per_component"):
            raise ValueError(
                f"inverse_weight_mode={inv_mode!r} not in "
                "('none', 'vector_magnitude', 'per_component')")
        use_struct_weight = (inv_mode == "vector_magnitude")
        use_comp_weight = (inv_mode == "per_component")

        # Pick between the standard and XLA-compiled chunk evaluator.
        # XLA fuses the per-chunk eval into a single GPU kernel and is
        # typically 1.5-2× faster, but compiles per unique (B, P) shape
        # — paid as upfront cost in the first generation.
        eval_chunk_fn = (self._evaluate_chunk_xla
                         if getattr(self.cfg, "eval_jit_compile", False)
                         else self._evaluate_chunk)

        eps = float(getattr(self.cfg, "inverse_weight_eps", 1e-4) or 1e-4)
        # Precompute inverse-magnitude weights. Applied to the *fitness*
        # signal only — never to the squared-error accumulator used for
        # RMSE / RRMSE reporting (so reporting metrics stay comparable
        # across loss / weighting ablations).
        inv_weights = None  # vector_magnitude: [B]
        if use_struct_weight:
            tgt_norm_sq = tf.reduce_sum(tf.square(batch_data["targets"]), axis=1)  # [B]
            inv_weights = 1.0 / tf.maximum(tgt_norm_sq, eps)
            inv_weights = inv_weights * (B_f / tf.reduce_sum(inv_weights))
        if use_comp_weight:
            tgt_sq = tf.square(batch_data["targets"])              # [B, T_dim]
            w = 1.0 / tf.maximum(tgt_sq, eps)                       # [B, T_dim]
            # Normalise so total weight = B * T_dim (preserves overall scale).
            w = w * (B_f * T_dim_f / tf.reduce_sum(w))
            # Stash into batch_data so _evaluate_chunk_impl can consume it.
            batch_data["_inv_comp_weights"] = w

        # SS_tot for RRMSE: total target magnitude squared over the whole batch.
        # Independent of weighting choices and candidates; computed once.
        ss_tot_batch = tf.reduce_sum(tf.square(batch_data["targets"]))
        ss_tot_batch = tf.maximum(ss_tot_batch, 1e-12)

        all_fitness = []

        # Streaming evaluation: outer loop = structure chunks, inner = population
        # chunks. Each chunk is built just-in-time by `prefetched_chunks` —
        # purely on-GPU when `_gv_resident_gpu` is set, otherwise via the
        # disk-staging pipe with optional prefetch overlap. Per-chunk
        # per-structure errors are reduced into running accumulators so the
        # full [C, B] tensor never materialises.
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
        # Pad pair arrays to the global per-data max so XLA-compiled eval
        # sees one shape and compiles once. None = no padding.
        pad_pairs_to = (batch_data.get("_max_chunk_pairs")
                        if getattr(self.cfg, "eval_jit_compile", False)
                        else None)
        for chunk_idx, (s_start, s_end, chunk) in enumerate(prefetched_chunks(
                batch_data, ranges,
                pin_to_cpu=self.cfg.pin_data_to_cpu,
                enabled=getattr(self.cfg, "chunk_prefetch", True),
                depth=getattr(self.cfg, "prefetch_depth", 1),
                pad_pairs_to=pad_pairs_to)):
            B_chunk = s_end - s_start
            chunk_lo = s_start
            chunk_hi = s_end

            # Slice precomputed [B]-shaped quantities to this chunk's range.
            inv_chunk = inv_weights[chunk_lo:chunk_hi] if use_struct_weight else None
            tc_chunk = tc_full[chunk_lo:chunk_hi] if return_per_type else None
            # Slice per-component weights too so _evaluate_chunk reads
            # only this chunk's rows. We mutate batch_data inside
            # prefetched_chunks' yielded `chunk` dict (not the outer one).
            if use_comp_weight:
                chunk["_inv_comp_weights"] = batch_data["_inv_comp_weights"][chunk_lo:chunk_hi]

            for pop_idx, p_start in enumerate(range(0, P, pop_chunk)):
                p_end      = min(p_start + pop_chunk, P)
                candidates = samples_tf[p_start:p_end]                  # [C, dim]
                chunk_err, chunk_sq = eval_chunk_fn(candidates, chunk)   # [C, B_chunk] each

                if use_struct_weight:
                    chunk_err = chunk_err * inv_chunk[tf.newaxis, :]
                # sq is INTENTIONALLY left unweighted — RMSE/RRMSE
                # reporting must be comparable across weighting modes.

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
            # next chunk's stage starts with the freed VRAM. The prefetch
            # worker may already hold a strong reference to the next
            # chunk's tensors — that's fine, those are *its* allocation.
            del chunk

        # Convert accumulated sums of per-structure errors → fitness.
        # Aggregation depends on the loss family:
        #   "mse"   : per-structure error is sum of squared residuals;
        #             fitness = sqrt(mean) = RMSE (canonical).
        #   "mae"   : per-structure error is sum of |residuals|;
        #             fitness = mean (already in error units, no sqrt).
        #   "huber" : per-structure error is sum of Huber per-component
        #             values; fitness = mean (already in error units,
        #             not squared — sqrt would scramble the scale).
        sqrt_aggregate = (loss_type == "mse")
        for pop_idx in range(len(total_acc_parts)):
            global_err = total_acc_parts[pop_idx]                       # [C]
            if return_per_type:
                per_type_err = per_type_acc_parts[pop_idx]              # [C, T]
                per_type_parts = []
                for t in range(T):
                    raw = per_type_err[:, t] / (type_counts[t] * T_dim_f)
                    per_type_parts.append(
                        tf.sqrt(tf.maximum(raw, 0.0)) if sqrt_aggregate else raw)
                raw_global = global_err / (B_f * T_dim_f)
                per_type_parts.append(
                    tf.sqrt(tf.maximum(raw_global, 0.0)) if sqrt_aggregate else raw_global)
                all_fitness.append(tf.stack(per_type_parts, axis=1))    # [C, T+1]
            else:
                raw_global = global_err / (B_f * T_dim_f)
                chunk_fitness = (tf.sqrt(tf.maximum(raw_global, 0.0))
                                 if sqrt_aggregate else raw_global)
                all_fitness.append(chunk_fitness)

        if return_per_type:
            fitness = tf.concat(all_fitness, axis=0)  # [P, T+1]
        else:
            fitness = tf.concat(all_fitness, axis=0)  # [P]

        if not return_per_type and self.cfg.toggle_regularization:
            reg = self.compute_regularization_tf(samples_tf)
            fitness = fitness + reg

        # Stash always-MSE reporting metrics for the fit loop. These
        # are independent of `loss_type` / inverse weighting, so RMSE
        # and RRMSE reported in history are comparable across runs.
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

    @tf.function(reduce_retracing=True, jit_compile=True)
    def _evaluate_chunk_xla(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        """XLA-compiled variant of _evaluate_chunk. The body is identical;
        only the @tf.function decorator differs. XLA fuses the dipole
        kernel pre-compute + per-type matmul + reduction into one GPU
        kernel (~1.5-2× faster on Ada/Hopper). Each unique (B, P) shape
        compiles once; with deterministic full-batch chunks the compile
        cost is amortised over the run."""
        return self._evaluate_chunk_impl(chunk_samples, batch_data)

    def _evaluate_chunk_impl(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        """Evaluate a chunk of C candidates on B structures.

        Returns a *pair* of per-structure tensors:
          - fitness_err: training-loss contribution (MSE / MAE / Huber,
              optionally per-component-weighted) — used for SNES
              ranking and the regularised total objective.
          - sq_err: always-MSE squared-error contribution (unweighted
              by the inverse-magnitude scheme) — used for RMSE and
              RRMSE reporting so those metrics remain comparable
              across loss-function ablations.

        Per-component weights from `batch_data["_inv_comp_weights"]`
        (set when `cfg.inverse_weight_mode == "per_component"`) are
        applied to `fitness_err` only. The structure-level inverse
        weighting from `vector_magnitude` mode is applied by the
        caller (after this function returns) so the scaling of fitness
        and reporting can be controlled independently there too.

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
        if self.cfg.target_mode == 2:
            if self.n_U_pair > 0:
                W0, b0, W1, b1, W0p, b0p, W1p, b1p, U_pair_cand = params
            else:
                W0, b0, W1, b1, W0p, b0p, W1p, b1p = params
                U_pair_cand = None
        else:
            if self.n_U_pair > 0:
                W0, b0, W1, b1, U_pair_cand = params
            else:
                W0, b0, W1, b1 = params
                U_pair_cand = None

        loss_type = self.cfg.loss_type
        huber_delta = float(getattr(self.cfg, "huber_delta", 1e-3))
        # Per-component inverse-magnitude weights (optional, applied to
        # the training loss only — never to the squared-error reporting
        # so RMSE stays comparable across weighting schemes).
        comp_w = batch_data.get("_inv_comp_weights")  # [B, T_dim] or None

        if self.cfg.target_mode == 2:
            pol_weights = self._pol_weights  # [6] component weights
            # Pre-absorb U_pair^T into W0 / W0_pol per candidate so the
            # vectorized_map loop body uses raw descriptors and no
            # gradient pull-back. Works because _W0_eff is linear and
            # broadcasts over the leading candidate axis.
            if U_pair_cand is not None:
                W0 = self.model._W0_eff(W0, U_pair_cand)
                W0p = self.model._W0_eff(W0p, U_pair_cand)

            # Combined per-component weights for the training loss:
            # pol_weights × per-component inverse weights (if active).
            # For MAE the legacy code used sqrt(pol_weights), preserved
            # by using sqrt-weighted per_structure_error in mae path.
            fitness_comp_w = pol_weights[tf.newaxis]  # [1, 6]
            if comp_w is not None:
                fitness_comp_w = fitness_comp_w * comp_w
            # Squared-error reporting always uses pol_weights only
            # (so the polarisability metric's shear emphasis is fixed
            # but the per-component-inverse-weighting is ablated).
            sq_comp_w = pol_weights[tf.newaxis]

            def _forward_one_candidate(args):
                w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p = args
                preds = self.model.predict_batch(
                    desc, grad_values, pair_atom, pair_gidx, pair_struct,
                    pos, Z, boxes, amask,
                    w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p,
                )
                diff = preds - targets  # [B, 6]
                fitness = per_structure_error(
                    diff, loss_type, huber_delta, component_weights=fitness_comp_w)
                sq = squared_error_per_structure(diff, component_weights=sq_comp_w)
                return tf.stack([fitness, sq], axis=0)  # [2, B]

            stacked = (W0, b0, W1, b1, W0p, b0p, W1p, b1p)
            both = tf.vectorized_map(_forward_one_candidate, stacked)  # [C, 2, B]
            return both[:, 0, :], both[:, 1, :]

        else:

            # Precompute W_atom [B, A, 3, Q] once — independent of all candidates.
            # Use static shapes when known (XLA needs compile-time dims for
            # the segment_sum's num_segments arg); fall back to tf.shape for
            # truly dynamic graphs.
            B_static = desc.shape[0]
            A_static = desc.shape[1]
            B_arg = B_static if B_static is not None else tf.shape(desc)[0]
            A_arg = A_static if A_static is not None else tf.shape(desc)[1]
            if self.cfg.target_mode == 1:
                W_atom = self.model._precompute_dipole_kernel(
                    grad_values, pair_struct, pair_atom, pair_gidx,
                    pos, boxes,
                    B_arg, A_arg)
            else:
                W_atom = None

            # Evaluate all C candidates simultaneously using explicit batched GEMMs.
            # predict_batch_candidates executes one GEMM per type in each direction
            # rather than C separate matmuls inside vectorized_map.
            preds = self.model.predict_batch_candidates(
                desc, W_atom, Z, amask, W0, b0, W1, b1,
                U_pair=U_pair_cand)  # [C, B, T_dim]

            if _scale_preds:
                preds = preds * inv_num_atoms[tf.newaxis]  # [C, B, T_dim] * [1, B, 1]

            diff = preds - targets[tf.newaxis]  # [C, B, T_dim]
            fitness_comp_w = (comp_w[tf.newaxis] if comp_w is not None else None)
            fitness = per_structure_error(
                diff, loss_type, huber_delta, component_weights=fitness_comp_w)
            sq = squared_error_per_structure(diff)  # unweighted, for RMSE/RRMSE
            return fitness, sq
