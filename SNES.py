from __future__ import annotations

import os
import sys
import time
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING, Callable

from data import gather_nep_pairs
from loss_functions import squared_error_per_structure
from nep_descriptor import nep_pair_kwargs

if TYPE_CHECKING:
    from TNEP import TNEP

def _format_duration(seconds: float) -> str:
    """HH:MM:SS, or Dd HH:MM:SS past 24 h (time.strftime rolls over at 24 h)."""
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
    """Assign reconstruct_params_tf tensors into the model's Variables.

    Tail order (contract): W0,b0,W1,b1 | +experts (nep_descriptor="sum")
    | +c_pair (neighbor readout, _c_learnable) | +U_pair (mixing) | +M_radial
    | +c_r,+c_a (NEP) | +sum_delta (3 unpenalised δ, the very end).
    """
    params = list(params)
    idx = 0
    model.W0.assign(params[idx]); idx += 1
    model.b0.assign(params[idx]); idx += 1
    model.W1.assign(params[idx]); idx += 1
    model.b1.assign(params[idx]); idx += 1
    if model.optimizer.n_expert > 0:
        for k in model.expert_keys:
            getattr(model, k).assign(params[idx]); idx += 1
    if model._neighbor_readout and model._c_learnable:
        model.c_pair.assign(params[idx]); idx += 1
    if (getattr(model, "descriptor_mixing", False)
            and getattr(model, "U_pair", None) is not None
            and idx < len(params)):
        model.U_pair.assign(params[idx]); idx += 1
    if (idx < len(params)
            and getattr(model, "M_radial", None) is not None):
        model.M_radial.assign(params[idx]); idx += 1
    if idx < len(params) and getattr(model, "c_r", None) is not None:
        model.c_r.assign(params[idx]); idx += 1
        model.c_a.assign(params[idx]); idx += 1
    if idx < len(params) and getattr(model, "sum_delta", None) is not None:
        model.sum_delta.assign(params[idx]); idx += 1

class SNES:
    """Separable NES over the flat param vector: N(mu, diag(sigma²)).

    Per generation: sample z_p = mu + sigma·s_p; evaluate RMSE per candidate on
    one batch; rank with log-rank utilities u_p; then
      mu    <- mu + sigma · Σ_p u_p s_p
      sigma <- sigma · exp(eta_sigma · Σ_p u_p (s_p² − 1))
    Param count: W0 T·Q·H + b0 T·H + W1 T·H + b1 1 (T·S under the neighbor readout).
    """

    def __init__(self, model: TNEP) -> None:
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        self.dim_q = self.cfg.dim_q
        self.batch_size = self.cfg.batch_size

        if self.cfg.seed is not None:
            self.tf_rng = tf.random.Generator.from_seed(self.cfg.seed)
        else:
            self.tf_rng = tf.random.Generator.from_non_deterministic_state()

        self.H = int(model.num_neurons)        # G·num_neurons under the rank-2 readout
        # Neighbor readout: W1 gains a channel axis [T, H, Ch]; a packed c_pair
        # table follows b1 — [Ch, T(T+1)/2] "symmetric", [Ch, T²] "ordered",
        # none for "off" (c ≡ 1 frozen).
        self._Ch = (int(model._kernel_channels)
                    if model._neighbor_readout else 1)      # kernel channels → c_pair
        # W1's last axis: Ck ("vector" head) or one per kernel group ("scalar").
        self._Hw = int(model._head_slots)      # head width → W1
        _T = self.cfg.num_types
        # One table per model (model._models), side by side on the pair axis.
        self.n_c = (self._Ch * (_T * _T if model._species_pairs == "ordered"
                                else _T * (_T + 1) // 2) * len(model._models)
                    if model._neighbor_readout and model._c_learnable else 0)
        # Typed ANN blocks: the SOAP ANN plus the "sum" 2b/3b experts, each
        # [W0(T,Q,H) | b0(T,H) | W1(T,H·Hw)] end to end; b1 follows the LAST
        # block so n_typed stays one contiguous slice.
        self._blocks = list(model._ann_blocks)          # AnnBlock rows

        def _block_per_type(b) -> int:
            """Type t's parameters of one block: W0 Q·H + b0 H + W1 H·Hw."""
            return b.dim_in * b.hidden + b.hidden + b.hidden * self._Hw

        per_type = sum(_block_per_type(b) for b in self._blocks)
        n_W0 = self.cfg.num_types * self.cfg.dim_q * self.H
        n_b0 = self.cfg.num_types * self.H
        n_W1 = self.cfg.num_types * self.H * self._Hw
        n_b1 = int(np.prod(model.b1.shape))      # 1, or T·S (neighbor readout)
        self.n_typed = _T * per_type
        self._n_expert = [_T * _block_per_type(b) for b in self._blocks[1:]]   # 0 outside "sum"
        self.n_expert = sum(self._n_expert)
        self.n_primary = self.n_typed + n_b1 + self.n_c
        self._n_W0 = n_W0
        self._n_b0 = n_b0
        self._n_W1 = n_W1
        self._n_b1 = n_b1
        self.n_anns_total = self.n_primary
        # U_pair tail (descriptor mixing): one [bs,bs] block per species pair,
        # active entries only, unpacked in reconstruct_params_tf.
        if getattr(self.cfg, "descriptor_mixing", False):
            # Reuse the model's layout: recomputing it here walks dead params
            # and runs expm at the wrong (larger) block size.
            from descriptor_layout import descriptor_block_layout
            self._mix_layout = getattr(
                self.model, "_mix_layout", None)
            if self._mix_layout is None:
                self._mix_layout = descriptor_block_layout(self.cfg)
            self._mix_per_type = bool(
                getattr(self.cfg, "descriptor_mixing_per_type", False))
            # cayley / expm: structural orthogonal maps (U ∈ SO(n) regardless
            # of λ) sharing the skew upper-triangle storage, bs(bs-1)/2 per
            # block. cayley U = (I − A)(I + A)⁻¹ misses −1 eigenvalues;
            # expm U = exp(A) is surjective onto SO(n).
            _reg_mode = str(getattr(self.cfg,
                                    "descriptor_mixing_regularizer",
                                    "off")).lower()
            self._mix_cayley = _reg_mode in ("cayley", "expm")
            self._mix_orth_map = _reg_mode if self._mix_cayley else None
            def _block_count(bs: int) -> int:
                return (bs * (bs - 1) // 2) if self._mix_cayley else bs * bs

            # Active entries only: per pair × L, α² (α(α−1)/2 for cayley/expm).
            self._mix_alpha_per_pair = [
                int(self._mix_layout["alpha_eff_per_pair"][k])
                for k in self._mix_layout["pair_keys"]]
            L = int(self.cfg.l_max) + 1
            self._mix_L = L
            per_T_block = int(sum(L * _block_count(a)
                                   for a in self._mix_alpha_per_pair))
            if self._mix_per_type:
                self.n_U_pair = self.cfg.num_types * per_T_block
            else:
                self.n_U_pair = per_T_block
            # Scatter matrices (one per block size) built eagerly so the traced
            # reconstruct path captures constants (no dict building under XLA).
            self._cayley_scatter_cache = {}
            if self._mix_cayley:
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

            # Non-uniform α: gather each pair's skew triangle into a max_α×max_α
            # skew so ONE batched solve/expm runs per layer (each eager per-pair
            # op is a ~6 ms host↔device sync). Padded rows/cols give V = 0.
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

        # Learned radial basis tail: M_radial [T, S, α, K], centre-type-major.
        self.n_radial = (int(np.prod(model.M_radial.shape))
                         if getattr(model, "M_radial", None) is not None else 0)
        # NEP contraction tail: c_r [T, T, n_r+1, K_r+1] then c_a, centre-type-major.
        self.n_c_r = (int(np.prod(model.c_r.shape))
                      if getattr(model, "c_r", None) is not None else 0)
        self.n_c_a = (int(np.prod(model.c_a.shape))
                      if getattr(model, "c_a", None) is not None else 0)
        self.n_nep = self.n_c_r + self.n_c_a
        # δ = (δ_soap, δ_2b, δ_3b): the LAST 3 entries, outside n_anns_total so
        # L1/L2 never sees them (the point of cfg.sum_delta).
        self.n_sum_delta = (3 if (bool(getattr(cfg, "sum_delta_learn", False))
                                  and getattr(model, "sum_delta", None) is not None)
                            else 0)
        self.dim = (self.n_anns_total + self.n_U_pair
                    + self.n_radial + self.n_nep + self.n_sum_delta)

        # μ: Glorot W0/W1, zero biases, zero mixing tail (gen 0 ≡ mixing-off).
        # σ: uniform init_sigma, times mixing_sigma_scale on the V_pair tail.
        rng = np.random.default_rng(self.cfg.seed)
        mu_init = self._build_mu_init(rng)
        self.mu = tf.Variable(mu_init, trainable=False, name="snes_mu")
        sigma_init_vec = np.full(self.dim, float(self.cfg.init_sigma),
                                 dtype=np.float32)
        mix_scale = float(getattr(self.cfg, "mixing_sigma_scale", 1.0))
        if self.n_U_pair > 0 and mix_scale != 1.0:
            mix_start = self.n_anns_total
            sigma_init_vec[mix_start:mix_start + self.n_U_pair] *= mix_scale
        self.sigma = tf.Variable(sigma_init_vec, trainable=False, name="snes_sigma")

        auto_pop = int(4 + (3 * np.log(self.dim)))
        self.pop_size = self.cfg.pop_size if self.cfg.pop_size is not None else auto_pop

        # λ sentinels: None → auto sqrt(dim·1e-6/T) (GPUMD); -1 → dynamic.
        # tf.Variable so per-gen updates don't retrace compute_regularization_tf.
        auto_lambda = float(np.sqrt(self.dim * 1e-6 / self.cfg.num_types))
        # GPUMD path: the monitor set is the TEST set; fit() never selects on it.
        self._gpumd_path = bool(getattr(self.cfg, "gpumd_path", False))
        # One λ per model (model._models order): scalar shared, list per model.
        n_models = len(model._models)

        def _lam(v, name):
            if v is None or (np.isscalar(v) and v == -1):
                return np.full(n_models, auto_lambda, dtype=np.float32)
            a = np.atleast_1d(np.asarray(v, dtype=np.float32))
            if a.size == 1:
                return np.full(n_models, float(a[0]), dtype=np.float32)
            if a.size != n_models:
                raise ValueError(
                    f"cfg.{name} has {a.size} entries but the model has "
                    f"{n_models} ({', '.join(model._models)}): give one value, "
                    f"or one per model in that order.")
            return a
        self._dyn_lambda_1 = bool(np.isscalar(self.cfg.lambda_1) and self.cfg.lambda_1 == -1)
        self._dyn_lambda_2 = bool(np.isscalar(self.cfg.lambda_2) and self.cfg.lambda_2 == -1)
        self.lambda_1 = tf.Variable(_lam(self.cfg.lambda_1, "lambda_1"),
                                    trainable=False, name="lambda_1")     # [n_models]
        self.lambda_2 = tf.Variable(_lam(self.cfg.lambda_2, "lambda_2"),
                                    trainable=False, name="lambda_2")
        self._reg_layout = self._build_reg_layout()

        # cayley/expm are structural (orthogonal regardless of λ): L1/L2 skips V_pair.
        self._mix_reg_mode = str(getattr(
            self.cfg, "descriptor_mixing_regularizer", "off")).lower()
        if self._mix_reg_mode not in ("off", "cayley", "expm"):
            raise ValueError(
                f"descriptor_mixing_regularizer={self._mix_reg_mode!r} not "
                "recognised (expected 'off', 'cayley', or 'expm')")

        # Mode 2 targets [xx, yy, zz, xy, yz, zx]: off-diagonals weighted by lambda_shear².
        if cfg.target_mode == 2:
            shear_sq = cfg.lambda_shear ** 2
            self._pol_weights = tf.constant(
                [1.0, 1.0, 1.0, shear_sq, shear_sq, shear_sq], dtype=tf.float32)
        else:
            self._pol_weights = None

        # Per-type ranking: type_of_variable [dim] → 0..T-1, or T for global params.
        self._per_type = (cfg.per_type_regularization
                          and cfg.toggle_regularization
                          and cfg.num_types > 1)
        if self._per_type:
            self._type_of_variable = self._build_type_of_variable()
            self._type_of_variable_tf = tf.constant(self._type_of_variable, dtype=tf.int32)

        self.eta_sigma = self.cfg.eta_sigma if self.cfg.eta_sigma is not None else self.compute_eta_sigma()
        self.utilities = tf.constant(self.compute_utilities(), dtype=tf.float32)

    def _build_reg_layout(self) -> list:
        """Penalty layout, one row per model (model._models order):
        (type_slices [T] of [(start, end)], n_type, global_slices, n_global).
        A model owns its typed ANN block and pair table; b1 and the other tails
        are unpenalised. Each model is normalised by ITS OWN count, so adding
        a model never dilutes another's penalty."""
        T = self.cfg.num_types
        n_tab = self.n_c // len(self.model._models) if self.n_c > 0 else 0
        rows, off = [], 0
        for k, b in enumerate(self._blocks):
            start, tsl = off, [[] for _ in range(T)]
            for n in (b.dim_in * b.hidden, b.hidden, b.hidden * self._Hw):
                for t in range(T):
                    tsl[t].append((off + t * n, off + (t + 1) * n))
                off += T * n
            n_type = b.dim_in * b.hidden + b.hidden + b.hidden * self._Hw
            gsl = [(start, off)]
            if n_tab:
                c0 = self.n_typed + 1 + k * n_tab
                gsl.append((c0, c0 + n_tab))
            rows.append((tsl, n_type, gsl, sum(e - s0 for s0, e in gsl)))
        return rows

    def _reg_terms(self, pv: tf.Tensor) -> tuple:
        """THE penalty: pv [P, dim] → (l1_type [P, T], l2_type [P, T],
        l1_global [P], l2_global [P]), summed over models with each model's own
        λ. GPUMD NEP4 form: L1 = λ1·Σ|θ|/n, L2 = λ2·sqrt(Σθ²/n)."""
        T = self.cfg.num_types
        l1_t, l2_t = [0.0] * T, [0.0] * T
        l1_g = l2_g = 0.0
        for k, (tsl, n_type, gsl, n_glob) in enumerate(self._reg_layout):
            lam1, lam2 = self.lambda_1[k], self.lambda_2[k]
            for t in range(T):
                x = tf.concat([pv[:, s0:e] for s0, e in tsl[t]], axis=1)
                l1_t[t] += lam1 * tf.reduce_sum(tf.abs(x), axis=1) / n_type
                l2_t[t] += lam2 * tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) / n_type)
            g = tf.concat([pv[:, s0:e] for s0, e in gsl], axis=1)
            l1_g += lam1 * tf.reduce_sum(tf.abs(g), axis=1) / n_glob
            l2_g += lam2 * tf.sqrt(tf.reduce_sum(tf.square(g), axis=1) / n_glob)
        return tf.stack(l1_t, axis=1), tf.stack(l2_t, axis=1), l1_g, l2_g

    def compute_regularization(self, param_vector: tf.Tensor | np.ndarray
                               ) -> tuple[float, float, float]:
        """Eager L1/L2 penalty of a [dim] vector → (l1, l2, 0.0); the third
        slot is kept for API compatibility. Multi-type (GPUMD NEP4): per-type
        terms averaged across types + a global term."""
        pv = tf.cast(param_vector, tf.float32)
        T = self.cfg.num_types

        if T > 1:
            l1_t, l2_t, l1_g, l2_g = self._reg_terms(pv[tf.newaxis])
            l1 = tf.reduce_mean(l1_t) + l1_g[0]
            l2 = tf.reduce_mean(l2_t) + l2_g[0]
        else:
            # Single-type: ANN weights only — the mixing tail is excluded
            # (shrinking A → 0 would collapse U → I).
            ann = pv[:self.n_anns_total]
            ann_n = self.n_anns_total
            # ponytail: single-type models use the first strength only;
            # per-model lambdas need the multi-type path (_reg_terms).
            l1 = self.lambda_1[0] * tf.reduce_sum(tf.abs(ann)) / ann_n
            l2 = self.lambda_2[0] * tf.sqrt(tf.reduce_sum(tf.square(ann)) / ann_n)

        return float(l1), float(l2), 0.0

    def _build_mu_init(self, rng: np.random.Generator) -> np.ndarray:
        """μ init [dim] float32: Glorot ANN weights; V_pair tail zero so U = I at gen 0."""
        T = self.cfg.num_types
        mu = np.zeros(self.dim, dtype=np.float32)
        gain = float(getattr(self.model, "_glorot_gain", 1.0))

        def _fill_ann(off: int, blocks: list) -> int:
            """Fill typed blocks then b1 from ``off``; expert blocks (i > 0)
            get W1 = 0 so gen 0 equals the SOAP-only model."""
            for i, b in enumerate(blocks):
                Q, Hb = b.dim_in, b.hidden
                c_W0 = gain * float(np.sqrt(6.0 / (Q + Hb)))
                c_W1 = gain * float(np.sqrt(6.0 / (Hb + self._Hw)))
                n_W0 = T * Q * Hb
                # Plain Glorot even with descriptor_scale: stored weights act
                # on the scaled, centred input.
                mu[off:off + n_W0] = rng.uniform(-c_W0, c_W0, size=n_W0).astype(np.float32)
                off += n_W0
                # b0 zero
                off += T * Hb
                n_W1 = T * Hb * self._Hw
                if i == 0:
                    w1 = rng.uniform(-c_W1, c_W1, size=(T, Hb, self._Hw)).astype(np.float32)
                    mu[off:off + n_W1] = w1.reshape(-1)
                off += n_W1
            # b1 zero
            off += self._n_b1
            return off

        off = _fill_ann(0, self._blocks)
        # c_pair table: init 1.0 (C=1 with c=1 ≈ the single-channel readout).
        if self.n_c > 0:
            mu[off:off + self.n_c] = 1.0
            off += self.n_c
        if self.n_U_pair > 0:
            mu[self.n_anns_total:self.n_anns_total + self.n_U_pair] = 0.0
        # M_radial (M0 = fixed descriptor at gen 0), NEP c_r/c_a (the model's
        # U(−1, 1) draw, GPUMD snes.cu:107-115) and δ: copied from the model.
        off = self.n_anns_total + self.n_U_pair
        for var, n in ((self.model.M_radial, self.n_radial),
                       (self.model.c_r, self.n_c_r),
                       (self.model.c_a, self.n_c_a),
                       (self.model.sum_delta, self.n_sum_delta)):
            if n > 0:
                mu[off:off + n] = var.numpy().reshape(-1); off += n
        return mu

    def _maybe_adapt_lambda(self, gen: int, data_loss: float,
                            l1: float, l2: float) -> None:
        """Dynamic (-1) lambdas only, every lambda_adapt_interval gens:
        λ ← clip(λ · (target_ratio · data_loss / penalty)^damping, λ_min, λ_max).
        damping < 1 suppresses oscillation.
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
        # Floor data_loss so a near-zero reference doesn't blow up the ratio.
        ref = max(float(data_loss), 1e-8)
        if self._dyn_lambda_1 and l1 > 1e-12:
            ratio = (target * ref) / float(l1)
            self.lambda_1.assign(np.clip(self.lambda_1.numpy() * (ratio ** damping), lmin, lmax))
        if self._dyn_lambda_2 and l2 > 1e-12:
            ratio = (target * ref) / float(l2)
            self.lambda_2.assign(np.clip(self.lambda_2.numpy() * (ratio ** damping), lmin, lmax))

    def _cayley_blocks_batched(self, A_upper_stacked: tf.Tensor,
                                bs: int) -> tf.Tensor:
        """Batched Cayley/expm over N uniform skew blocks: A_upper_stacked
        [..., N, bs(bs-1)/2] → V = U − I [..., N, bs, bs] (zeros for bs ≤ 1).

        U = (I+A)(I−A)⁻¹ (cayley) or exp(A) (expm). One batched solve/expm is
        ~10× faster than N launches on tiny ≤8×8 blocks. fp32 throughout.
        """
        if bs <= 1:
            shape = tf.concat(
                [tf.shape(A_upper_stacked)[:-1], [bs, bs]], axis=0)
            return tf.zeros(shape, dtype=tf.float32)
        scatter = self._cayley_scatter_cache[bs]
        # A[..., n, i, j] = Σ_k scatter[i, j, k] · A_upper[..., n, k]
        A = tf.einsum('ijk,...nk->...nij', scatter, A_upper_stacked)
        I = tf.eye(bs, dtype=A.dtype)
        if self._mix_orth_map == "expm":
            # expm batches leading dims natively.
            U = tf.linalg.expm(A)
        else:                                        # "cayley"
            U = tf.linalg.solve(I - A, I + A)
        return U - I


    def _build_type_of_variable(self) -> np.ndarray:
        """Atom-type label per param → [dim] int32: typed W0/b0/W1 rows get t,
        b1 / c_pair / shared mixing / δ get T (global)."""
        T = self.cfg.num_types

        def _ann_types(blocks) -> np.ndarray:
            tov = np.empty(self.n_primary, dtype=np.int32)
            offset = 0
            for b in blocks:
                for n in (b.dim_in * b.hidden, b.hidden, b.hidden * self._Hw):
                    for t in range(T):
                        tov[offset:offset + n] = t
                        offset += n
            tov[offset:] = T          # b1 and c_pair: global
            return tov

        ann_tov = _ann_types(self._blocks)

        # Mixing: shared V_pair → T; per-type → T slabs labelled by central type.
        tail_labels_parts = []
        if self.n_U_pair > 0:
            if self._mix_per_type:
                per_T = self.n_U_pair // T
                u_labels = np.empty(self.n_U_pair, dtype=np.int32)
                for t_idx in range(T):
                    u_labels[t_idx * per_T:(t_idx + 1) * per_T] = t_idx
            else:
                u_labels = np.full(self.n_U_pair, T, dtype=np.int32)
            tail_labels_parts.append(u_labels)
        # M_radial [T, S, α, K] and c_r/c_a [T, T, n, k] are centre-type-major.
        if self.n_radial > 0:
            tail_labels_parts.append(
                np.repeat(np.arange(T, dtype=np.int32), self.n_radial // T))
        for n_c in (self.n_c_r, self.n_c_a):
            if n_c > 0:
                tail_labels_parts.append(
                    np.repeat(np.arange(T, dtype=np.int32), n_c // T))
        # δ: one per expert, not per centre type → global.
        if self.n_sum_delta > 0:
            tail_labels_parts.append(
                np.full(self.n_sum_delta, T, dtype=np.int32))
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
        """Composite noise: each column of s [P, dim] is permuted by the fitness
        ranking of its variable's type (type_of_variable[v]).

        fitness_per_type_rmse [P, T+1] (0..T-1 = structures containing that
        type, T = global); samples [P, dim] supply the penalty. Returns
        s_sorted [P, dim], fed to update() directly.
        """
        T = self.cfg.num_types

        # Per-type penalty onto per-type RMSE, global onto global → [T+1] fitness.
        l1_t, l2_t, l1_g, l2_g = self._reg_terms(samples)
        fitness_per_type = [fitness_per_type_rmse[:, t] + l1_t[:, t] + l2_t[:, t]
                            for t in range(T)]
        fitness_per_type.append(fitness_per_type_rmse[:, T] + l1_g + l2_g)

        ranks_per_type = [tf.argsort(f) for f in fitness_per_type]  # list of [P], ascending
        s_sorted_all = tf.stack([tf.gather(s, r) for r in ranks_per_type])  # [T+1, P, dim]

        # s_sorted[:, v] = s_sorted_all[tov[v], :, v] via gather_nd on [dim, T+1, P].
        tov = self._type_of_variable_tf  # [dim]
        s_by_var = tf.transpose(s_sorted_all, [2, 0, 1])  # [dim, T+1, P]
        v_indices = tf.range(self.dim)  # [dim]
        gather_idx = tf.stack([v_indices, tov], axis=1)  # [dim, 2]
        s_selected = tf.gather_nd(s_by_var, gather_idx)  # [dim, P]
        s_sorted = tf.transpose(s_selected)  # [P, dim]

        return s_sorted

    def compute_eta_sigma(self) -> float:
        """GPUMD σ learning rate: (3 + ln num) / (5·sqrt(num)) / 2, num = dim/num_types."""
        num = float(self.dim) / self.cfg.num_types
        num = max(num, 1.0)
        eta_sigma = ((3.0 + np.log(num)) / (5.0 * np.sqrt(num))) / 2.0
        return float(eta_sigma)

    def compute_utilities(self) -> np.ndarray:
        """Hansen log-rank utilities → [pop_size], rank 0 = best, zero-mean
        (top half positive summing to 1, rest −1/λ). Caches self._recomb_w for μ_eff."""
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
        """Antithetic normal noise (P, width): P//2 rows ε_i stacked with −ε_i
        (+1 i.i.d. row if P odd); cancels first-order noise in the mean step
        (Salimans 2017; Brockhoff 2010)."""
        P, width = int(shape[0]), int(shape[1])
        half = P // 2
        s_half = self.tf_rng.normal(shape=(half, width))
        if P % 2 == 0:
            return tf.concat([s_half, -s_half], axis=0)
        s_extra = self.tf_rng.normal(shape=(1, width))
        return tf.concat([s_half, -s_half, s_extra], axis=0)

    def ask(self) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Sample pop_size candidates, samples = μ + σ·s_iso; mirrored noise
        when cfg.mirrored_sampling, else i.i.d. normal as GPUMD (snes.cu
        gpu_create_population).

        Returns (samples [P, dim], {"s_iso" [P, dim], "delta" [P, dim] = σ·s_iso}).
        """
        s_iso = (self._mirrored_normal((self.pop_size, self.dim))
                 if bool(getattr(self.cfg, "mirrored_sampling", True))
                 else self.tf_rng.normal(shape=(self.pop_size, self.dim)))
        delta = s_iso * self.sigma
        samples = self.mu + delta
        return samples, {"s_iso": s_iso, "delta": delta}

    def update(self, utilities: tf.Tensor, aux: dict[str, tf.Tensor]) -> None:
        """SNES mean / per-coord σ update (mutates mu, sigma); utilities [P]
        best first, aux from ask() sorted by fitness.

        mu += σ·Σ u_p·s_iso;  σ *= exp(η·Σ u_p·(s_iso²−1)), floored at
        cfg.sigma_floor so σ can't collapse on long runs.
        """
        s_iso = aux["s_iso"]
        grad_mu = tf.einsum('p,pd->d', utilities, s_iso)
        self.mu.assign_add(self.sigma * grad_mu)
        # Σ u_p·(s²−1) is self-equilibrating: too-small σ → grad > 0 → grow.
        grad_sigma = tf.einsum('p,pd->d', utilities, s_iso ** 2 - 1.0)
        new_sigma = self.sigma * tf.exp(self.eta_sigma * grad_sigma)
        floor = getattr(self.cfg, "sigma_floor", 1e-5)
        if floor is not None and floor > 0.0:
            new_sigma = tf.maximum(new_sigma, float(floor))
        self.sigma.assign(new_sigma)

    def fit(self, train_data: dict[str, tf.Tensor], val_data: dict[str, tf.Tensor], plot_callback: Callable | None = None, resume_state: dict | None = None) -> dict:
        """SNES training loop → (history, final_model, best_val_model).

        INVARIANT (CRN): one batch_data per generation, shared across the
        population. plot_callback(history, gen) fires every cfg.plot_interval.
        """
        print("Fitting model...")
        cfg = self.cfg
        S_train = train_data["descriptors"].shape[0]

        if resume_state is not None:
            history = resume_state["history"]
            # Ensure the timing sub-dict exists (older checkpoints lack it).
            history.setdefault("timing", {})
            for k in ("sample_batch", "evaluate", "rank_update",
                      "validate", "overhead"):
                history["timing"].setdefault(k, [])
            self.mu.assign(resume_state["mu"])
            self.sigma.assign(resume_state["sigma"])
            best_mu = tf.constant(resume_state["best_mu"], dtype=tf.float32)
            # pre-2026 checkpoints lack best_sigma.
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
                    # Generator-state shape shifts across TF versions; keep the fresh one.
                    pass
            start_gen = int(resume_state["last_gen"]) + 1
            # Offset train_start so elapsed continues from checkpointed wall-time.
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
        # validate() at μ, the same estimator as val_fitness.
        train_rmse = float('inf')
        sigma_min = sigma_max = sigma_mean = sigma_median = float(cfg.init_sigma)

        # SIGTERM (Slurm walltime): flip a flag so the in-flight gen finishes,
        # then checkpoint and break. Hooked only with save_path; restored at exit.
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
                # signal.signal is main-thread only; skip in subthreads.
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
                for _k in ("V_atom", "nep_raw"):
                    if _k in train_data:
                        struct_keys.append(_k)
                batch_data = {
                    key: tf.gather(train_data[key], batch_idx_tf)
                    for key in struct_keys
                }
                # COO pair gather: select pairs of the sampled structures.
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
                # NEP pair stream: same CSR gather + local pointer.
                gather_nep_pairs(train_data, batch_data, batch_idx_tf)

            t1 = time.perf_counter()

            samples, aux = self.ask()

            if self._per_type:
                # Per-type: [P, T+1] RMSE, then composite gradients.
                fitness_per_type_rmse = self.evaluate_population(
                    samples, batch_data, return_per_type=True)
                fitness = fitness_per_type_rmse[:, -1]  # global RMSE for reporting
            else:
                fitness = self.evaluate_population(samples, batch_data)

            # One GPU→CPU pull per gen: stack every scalar (σ stats included).
            rmse_pc = self._last_rmse_per_cand
            rrmse_pc = self._last_rrmse_per_cand
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

            t2 = time.perf_counter()

            # The penalty is already in fitness in-graph; this eager pull (a sync)
            # is for reporting every 100 gens and λ adaptation only.
            need_adapt = (self._dyn_lambda_1 or self._dyn_lambda_2)
            adapt_cadence = max(
                1, int(getattr(cfg, "lambda_adapt_interval", 100)))
            do_adapt_now = need_adapt and (gen % adapt_cadence == 0)
            do_report_now = (gen % 100 == 0)
            if cfg.toggle_regularization and (do_adapt_now or do_report_now):
                gen_l1, gen_l2, gen_lorth = self.compute_regularization(self.mu)
                if do_adapt_now:
                    # best-RMSE as reference: the mean is dominated by bad candidates early.
                    self._maybe_adapt_lambda(gen, best_rmse, gen_l1, gen_l2)
            elif not cfg.toggle_regularization:
                gen_l1, gen_l2, gen_lorth = 0, 0, 0

            # Sort s_iso and delta by the same fitness order, then update().
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

            # Clear the fold cache every gen: self.mu keeps its id across
            # assign(), so the id-keyed cache can't tell gens apart.
            self._validate_fold_cache = None
            _do_val = (gen % cfg.val_interval == 0) or (gen == cfg.num_generations - 1)
            if _do_val:
                val_fitness = self.validate(val_data, self.mu)
            # Train RMSE at μ (reuses the fold cache). GPUMD path: the monitor
            # is the big test set and stays on val_interval; train refreshes every gen.
            if _do_val or self._gpumd_path:
                train_rmse = self.validate(train_data, self.mu)

            t4 = time.perf_counter()

            # History only on val gens (+ final gen).
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
            _mon = "test" if self._gpumd_path else "val"
            line = (f"\r{bar} {gen + 1}/{cfg.num_generations} "
                    f"train RMSE: {train_rmse:.6f}  "
                    f"{_mon} RMSE: {val_fitness:.6f}  "
                    + ("" if self._gpumd_path
                       else f"best val RMSE: {best_val_loss:.6f}  ")
                    + f"elapsed: {elapsed_str}  ETA: {eta_str}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
                line += f"  train obj(mse): {avg_fitness:.6f}"
                line += f"  best_RMSE: {best_rmse:.6f}  best_RRMSE: {best_rrmse:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # GPUMD path: report the monitor, never select on it (patience inert).
            if _do_val and self._gpumd_path:
                best_val_loss = val_fitness
            elif _do_val:
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
                # Final validation on a non-val gen so best_val_loss isn't left at inf.
                if not _do_val:
                    val_fitness = self.validate(val_data, self.mu)
                    if val_fitness < best_val_loss:
                        best_val_loss = val_fitness
                        best_mu = tf.identity(self.mu)
                        best_sigma = tf.identity(self.sigma)
                # INVARIANT: don't restore best into self.mu here — post-loop
                # needs the final-gen μ for final_model.
                break

            t5 = time.perf_counter()

            if _do_val:
                history["timing"]["sample_batch"].append(t1 - t0)
                history["timing"]["evaluate"].append(t2 - t1)
                history["timing"]["rank_update"].append(t3 - t2)
                history["timing"]["validate"].append(t4 - t3)
                history["timing"]["overhead"].append(t5 - t4)

            # Plot callback sees the best params in self.model, then μ is restored.
            if gen + 1 < cfg.num_generations:
                if (plot_callback is not None
                        and cfg.plot_interval is not None
                        and (gen + 1) % cfg.plot_interval == 0):
                    params = self.reconstruct_params_tf(best_mu)
                    _set_model_params(self.model, *params)
                    plot_callback(history, gen + 1)
                    params = self.reconstruct_params_tf(self.mu)
                    _set_model_params(self.model, *params)

            # Rolling checkpoint in the run dir.
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
                sys.stdout.write(
                    f"\n  checkpoint saved at gen {gen + 1} → {ckpt_path}\n")
                sys.stdout.flush()

            # SIGTERM: flush a checkpoint regardless of cadence; wrap-up still runs.
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

        if _prev_term_handler is not None:
            try:
                _signal.signal(_signal.SIGTERM, _prev_term_handler)
            except (ValueError, OSError):
                pass

        print()  # newline after progress bar

        # Build final-gen model (current μ, before restoring best).
        from TNEP import TNEP
        final_model = TNEP(self.cfg)
        final_params = self.reconstruct_params_tf(self.mu)
        _set_model_params(final_model, *final_params)

        if self._gpumd_path:
            # GPUMD writes the CURRENT gen's potential: pin best to the final μ
            # so nothing is selected on the test set.
            best_mu = tf.identity(self.mu)
            best_sigma = tf.identity(self.sigma)

        best_val_model = TNEP(self.cfg)
        best_val_params = self.reconstruct_params_tf(best_mu)
        _set_model_params(best_val_model, *best_val_params)

        # self.model ← best.
        self.mu.assign(best_mu)
        self.sigma.assign(best_sigma)
        _set_model_params(self.model, *best_val_params)

        return history, final_model, best_val_model

    def validate(self, val_data: dict[str, tf.Tensor], mu_tf: tf.Tensor | None = None) -> float:
        """RMSE over (a val_size subsample of) val_data at mu_tf [dim], or at
        the model's current weights when mu_tf is None."""
        S_val = val_data["num_atoms"].shape[0]

        if self.cfg.val_size is None:
            val_idx_tf = tf.range(S_val, dtype=tf.int32)
        else:
            val_idx_tf = tf.cast(
                tf.argsort(self.tf_rng.uniform(shape=[S_val]))[:self.cfg.val_size],
                tf.int32)

        # Fold cache keyed by id(μ): val + train at the same μ shouldn't
        # re-fold; fit() clears it every gen.
        _cache = getattr(self, "_validate_fold_cache", None)
        _cache_hit = (mu_tf is not None
                      and _cache is not None
                      and _cache.get("_key") == id(mu_tf))

        if mu_tf is not None and not _cache_hit:
            params = self.reconstruct_params_tf(mu_tf)
            named = self._split_reconstructed(params)
            W0 = named.get("W0"); b0 = named.get("b0")
            W1 = named.get("W1"); b1 = named.get("b1")
            U_pair_val = named.get("U_pair")
            c_pair_val = named.get("c_pair")
            M_val = named.get("M_radial")
            c_r_val = named.get("c_r"); c_a_val = named.get("c_a")
            experts_val = self._experts_of(named)
            sd_val = named.get("sum_delta")
            # Centring fold (needs the stored W0), then absorb U_pair^T.
            b0 = self.model._b0_eff(b0, W0)
            W0 = self.model._W0_eff(W0, U_pair_val)
            self._validate_fold_cache = {
                "_key": id(mu_tf),
                "fold": (W0, b0, W1, b1, c_pair_val, M_val,
                         c_r_val, c_a_val, experts_val, sd_val),
            }
        elif _cache_hit:
            (W0, b0, W1, b1, c_pair_val, M_val,
             c_r_val, c_a_val, experts_val, sd_val) = _cache["fold"]
        else:
            # None → predict_batch falls back to the model's own Variables.
            W0, b0, W1, b1 = self.model.W0, self.model.b0, self.model.W1, self.model.b1
            c_pair_val = None
            M_val = None
            c_r_val = c_a_val = None
            experts_val = None
            sd_val = None
            b0 = self.model._b0_eff(b0, W0)
            W0 = self.model._W0_eff(W0)

        # Full val streams via prefetched_chunks; a random subset uses the slice path.
        from data import slice_and_complete_chunk, prefetched_chunks
        N_idx = int(val_idx_tf.shape[0])
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else N_idx

        diff_sq_sum = tf.constant(0.0, dtype=tf.float32)
        diff_count  = tf.constant(0.0, dtype=tf.float32)

        def _consume(chunk, chunk_idx=0):
            nonlocal diff_sq_sum, diff_count
            # W_atom cache id(val_data) → chunk_idx, only when val_size is None
            # (static chunks).
            W_atom_v = None
            if self.cfg.target_mode in (1, 2):
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
                    if self.model._neighbor_readout:
                        # V [B,A,D,C,T] from data.py rides in the W_atom slot,
                        # for EVERY nep_descriptor ("replace" included).
                        W_atom_v = chunk["V_atom"]
                    elif self.model._nep_mode != "replace":
                        # replace+tnep: μ is NEP-only, no SOAP kernel.
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
                W0, b0, W1, b1,
                W_atom=W_atom_v, c_pair=c_pair_val, M_radial=M_val,
                c_r=c_r_val, c_a=c_a_val, experts=experts_val,
                sum_delta=sd_val,
                nep_raw=chunk.get("nep_raw"), **nep_pair_kwargs(chunk),
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

    def _experts_of(self, named: dict) -> tuple | None:
        """The "sum" expert tensors in EXPERT_WEIGHTS order (None outside "sum")."""
        return tuple(named[k] for k in self.model.expert_keys) if self.n_expert else None

    def _split_reconstructed(self, params: tuple) -> dict:
        """Named dict of the reconstruct_params_tf tuple (tail order: _set_model_params)."""
        out: dict = {"U_pair": None, "c_pair": None, "M_radial": None,
                     "c_r": None, "c_a": None, "sum_delta": None}
        idx = 0
        out["W0"] = params[idx]; idx += 1
        out["b0"] = params[idx]; idx += 1
        out["W1"] = params[idx]; idx += 1
        out["b1"] = params[idx]; idx += 1
        if self.n_expert:
            for k in self.model.expert_keys:
                out[k] = params[idx]; idx += 1
        if self.n_c > 0:
            out["c_pair"] = params[idx]; idx += 1
        if self.n_U_pair > 0 and idx < len(params):
            out["U_pair"] = params[idx]; idx += 1
        if self.n_radial > 0 and idx < len(params):
            out["M_radial"] = params[idx]; idx += 1
        if self.n_nep > 0 and idx < len(params):
            out["c_r"] = params[idx]; idx += 1
            out["c_a"] = params[idx]; idx += 1
        if self.n_sum_delta > 0 and idx < len(params):
            out["sum_delta"] = params[idx]; idx += 1
        return out

    def reconstruct_params_tf(self, param_vectors: tf.Tensor) -> tuple:
        """[dim] or batched [P, dim] → tuple of weight tensors ([P, ...] when
        batched) in _set_model_params tail order. Works inside @tf.function."""
        T = self.cfg.num_types
        is_batched = len(param_vectors.shape) == 2

        def _extract(pv, offset, blocks):
            """Typed blocks then b1 → (W0, b0, W1, b1, *experts, offset)."""
            ws = []
            for b in blocks:
                Q, Hb = b.dim_in, b.hidden
                w1_shape = [T, Hb, self._Hw] if self.model._neighbor_readout else [T, Hb]
                for shape in ([T, Q, Hb], [T, Hb], w1_shape):
                    n = int(np.prod(shape))
                    ws.append(tf.reshape(pv[..., offset:offset + n],
                                         ([-1] + shape) if is_batched else shape))
                    offset += n
            b1_shape = list(self.model.b1.shape)
            b1 = tf.reshape(pv[..., offset:offset + self._n_b1],
                            ([-1] + b1_shape) if is_batched else b1_shape)
            offset += self._n_b1
            return (*ws[:3], b1, *ws[3:], offset)

        first = _extract(param_vectors, 0, self._blocks)
        offset = first[-1]
        primary = first[:-1]

        if self.n_c > 0:
            n_pairs = self.n_c // self._Ch
            c = tf.reshape(param_vectors[..., offset:offset + self.n_c],
                           [-1, self._Ch, n_pairs] if is_batched
                           else [self._Ch, n_pairs])
            offset += self.n_c
            primary = primary + (c,)

        tail = primary

        if self.n_U_pair > 0:
            # Mixing tail: T contiguous slabs when per-type; inside a slab
            # [V_{p,l}] pair-major, l-minor.
            U_flat = param_vectors[..., offset:offset + self.n_U_pair]
            U_pair_k = self._reconstruct_one_mixing_layer(
                U_flat, is_batched, self._mix_per_type, T)
            offset += self.n_U_pair
            tail = tail + (U_pair_k,)

        # Learned radial basis tail: M_radial [T, S, α, K] (+ leading C).
        if self.n_radial > 0:
            shape = list(self.model.M_radial.shape)
            M = tf.reshape(param_vectors[..., offset:offset + self.n_radial],
                           ([-1] + shape) if is_batched else shape)
            offset += self.n_radial
            tail = tail + (M,)

        # NEP contraction tail: c_r then c_a, [T, T, n+1, K+1] (+ leading C).
        if self.n_nep > 0:
            for var, n_c in ((self.model.c_r, self.n_c_r),
                             (self.model.c_a, self.n_c_a)):
                shape = list(var.shape)
                c = tf.reshape(param_vectors[..., offset:offset + n_c],
                               ([-1] + shape) if is_batched else shape)
                offset += n_c
                tail = tail + (c,)

        # Per-expert amplitude tail: δ [3] (+ leading C when batched).
        if self.n_sum_delta > 0:
            d = tf.reshape(param_vectors[..., offset:offset + self.n_sum_delta],
                           [-1, 3] if is_batched else [3])
            offset += self.n_sum_delta
            tail = tail + (d,)

        return tail

    def _reconstruct_one_mixing_layer(self, U_flat: tf.Tensor,
                                        is_batched: bool,
                                        per_type: bool,
                                        T: int) -> tf.Tensor:
        """Reconstruct one mixing-layer's U_pair tensor from its flat slab."""
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

            # Cayley/expm non-uniform α: gather into max_α×max_α skews, ONE
            # batched solve/expm (see __init__).
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

            # "off" non-uniform fallback: reshapes + pads only, nothing to batch.
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
            # Fold the T slabs into the block axis so ONE solve/expm covers
            # T·n_pairs·L blocks instead of T launches (~6 ms each eager).
            if self._mix_cayley:
                n_pairs = len(self._mix_alpha_per_pair)
                uniform_alpha_pt = len(set(self._mix_alpha_per_pair)) == 1
                T_total = T * per_T_block
                slabs_all = U_flat[..., :T_total]
                leading = tf.shape(slabs_all)[:-1]
                slab_per_t_shape = tf.concat(
                    [leading, [T, per_T_block]], axis=0)
                slab_per_t = tf.reshape(slabs_all, slab_per_t_shape)
                if uniform_alpha_pt:
                    payload = self._mix_alpha_per_pair[0] * (
                        self._mix_alpha_per_pair[0] - 1) // 2
                    stacked_shape = tf.concat(
                        [leading, [T * n_pairs * L, payload]], axis=0)
                    stacked = tf.reshape(slab_per_t, stacked_shape)
                    V = self._cayley_blocks_batched(
                        stacked, self._mix_alpha_per_pair[0])
                else:
                    max_payload = self._mix_l_aware_cayley_max_payload
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
                # V [..., T·n_pairs·L, mα, mα] → [..., T, n_pairs, L, mα, mα].
                final_shape = tf.concat(
                    [leading,
                     [T, n_pairs, L, max_alpha, max_alpha]], axis=0)
                return tf.reshape(V, final_shape)
            # "off" regulariser: per-T list (nothing to batch).
            per_t = [_extract_t_block_l_aware(t) for t in range(T)]
            stack_axis = 1 if is_batched else 0
            return tf.stack(per_t, axis=stack_axis)
        return _extract_t_block_l_aware(0)

    @tf.function(reduce_retracing=True)
    def compute_regularization_tf(self, param_vectors: tf.Tensor) -> tf.Tensor:
        """In-graph L1+L2 penalty, [P, dim] → [P]; same rule as compute_regularization."""
        T = self.cfg.num_types

        if T > 1:
            l1_t, l2_t, l1_g, l2_g = self._reg_terms(param_vectors)
            reg = tf.reduce_mean(l1_t + l2_t, axis=1) + l1_g + l2_g
        else:
            # Single-type: ANN weights only (mixing tail excluded, see compute_regularization).
            ann = param_vectors[:, :self.n_anns_total]
            ann_n = self.n_anns_total
            l1 = self.lambda_1[0] * tf.reduce_sum(tf.abs(ann), axis=1) / ann_n
            l2 = self.lambda_2[0] * tf.sqrt(
                tf.reduce_sum(tf.square(ann), axis=1) / ann_n)
            reg = l1 + l2

        return reg


    def evaluate_population(self, samples_tf: tf.Tensor, batch_data: dict[str, tf.Tensor],
                            return_per_type: bool = False) -> tf.Tensor:
        """RMSE of every candidate samples_tf [P, dim] on one batch → fitness [P]
        (+reg when enabled), or [P, T+1] per-type RMSE (last = global) when
        return_per_type. Chunks over population (population_chunk_size) and
        structures (batch_chunk_size); squared-error sums accumulate across chunks.
        """
        P = self.pop_size
        # num_atoms, not descriptors: its Q axis is a zero-sized placeholder in OTF mode.
        B = batch_data["num_atoms"].shape[0]
        T_dim = batch_data["targets"].shape[1]
        pop_chunk = self.cfg.population_chunk_size if self.cfg.population_chunk_size is not None else P
        struct_chunk = self.cfg.batch_chunk_size if self.cfg.batch_chunk_size is not None else B

        T = self.cfg.num_types
        T_dim_f = tf.cast(T_dim, tf.float32)
        B_f = tf.cast(B, tf.float32)
        eval_chunk_fn = self._evaluate_chunk

        # SS_tot centred per component so RRMSE = RMSE/std(reference) (TNEP
        # paper; R² = 1 − RRMSE²). Raw Σt² would be dominated by α's 44 Å³ trace.
        _t = batch_data["targets"]
        ss_tot_batch = tf.reduce_sum(tf.square(_t - tf.reduce_mean(_t, axis=0, keepdims=True)))
        ss_tot_batch = tf.maximum(ss_tot_batch, 1e-12)

        all_fitness = []

        from data import prefetched_chunks

        if return_per_type:
            tc_full = batch_data["types_contained"]                     # [B, T]
            type_counts = tf.maximum(tf.reduce_sum(tc_full, axis=0), 1.0)  # [T]

        # Per pop-chunk [C] accumulators: total_acc (ranking loss), sq_acc
        # (unweighted MSE for the RMSE/RRMSE report).
        total_acc_parts: list = [None] * ((P + pop_chunk - 1) // pop_chunk)
        sq_acc_parts: list = [None] * len(total_acc_parts)
        per_type_acc_parts: list = [None] * len(total_acc_parts) if return_per_type else []

        ranges = [(s, min(s + struct_chunk, B)) for s in range(0, B, struct_chunk)]
        for chunk_idx, (s_start, s_end, chunk) in enumerate(prefetched_chunks(
                batch_data, ranges,
                pin_to_cpu=self.cfg.pin_data_to_cpu)):
            chunk_lo = s_start
            chunk_hi = s_end

            tc_chunk = tc_full[chunk_lo:chunk_hi] if return_per_type else None

            # Candidate-independent W_atom [B,A,3,Q] once per struct chunk; cached
            # across gens (keyed by chunk_idx, id(batch_data)) only for full batch.
            _W_atom_cached = None
            if self.cfg.target_mode in (1, 2):
                if getattr(self.cfg, "batch_size", None) is None:
                    _bd_id = id(batch_data)
                    _cache = getattr(self, "_W_atom_static_cache", None)
                    if _cache is None or _cache.get("_bd_id") != _bd_id:
                        self._W_atom_static_cache = {"_bd_id": _bd_id}
                        _cache = self._W_atom_static_cache
                    _W_atom_cached = _cache.get(chunk_idx)
                if _W_atom_cached is not None:
                    chunk["_W_atom"] = _W_atom_cached
            if self.cfg.target_mode in (1, 2) and "_W_atom" not in chunk:
                if self.model._neighbor_readout:
                    # V [B,A,D,C,T] from data.py rides in the W_atom slot for
                    # EVERY nep_descriptor: "replace" must not skip it.
                    chunk["_W_atom"] = chunk["V_atom"]
                elif self.model._nep_mode != "replace":
                    # replace+tnep: μ is NEP-only, no SOAP kernel.
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
                # Write through to the static cache ("_W_atom" absent under replace+tnep).
                if ("_W_atom" in chunk
                        and getattr(self.cfg, "batch_size", None) is None
                        and getattr(self, "_W_atom_static_cache", None)
                            is not None
                        and self._W_atom_static_cache.get("_bd_id")
                            == id(batch_data)):
                    self._W_atom_static_cache[chunk_idx] = chunk["_W_atom"]

            for pop_idx, p_start in enumerate(range(0, P, pop_chunk)):
                p_end      = min(p_start + pop_chunk, P)
                candidates = samples_tf[p_start:p_end]                  # [C, dim]
                chunk_err, chunk_sq = eval_chunk_fn(candidates, chunk)   # [C, B_chunk] each


                global_part = tf.reduce_sum(chunk_err, axis=1)
                sq_part = tf.reduce_sum(chunk_sq, axis=1)
                if total_acc_parts[pop_idx] is None:
                    total_acc_parts[pop_idx] = global_part
                    sq_acc_parts[pop_idx] = sq_part
                else:
                    total_acc_parts[pop_idx] = total_acc_parts[pop_idx] + global_part
                    sq_acc_parts[pop_idx] = sq_acc_parts[pop_idx] + sq_part

                if return_per_type:
                    per_type_chunk = tf.einsum('cb,bt->ct', chunk_err, tc_chunk)  # [C, T]
                    if per_type_acc_parts[pop_idx] is None:
                        per_type_acc_parts[pop_idx] = per_type_chunk
                    else:
                        per_type_acc_parts[pop_idx] = per_type_acc_parts[pop_idx] + per_type_chunk

                del chunk_err, chunk_sq
            # Free the chunk's VRAM before the next one stages.
            del chunk

        # Accumulated squared-error sums → RMSE.
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

        # Unweighted reporting metrics for the fit loop (comparable across runs).
        sq_per_cand = tf.concat(sq_acc_parts, axis=0)                # [P]
        rmse_per_cand = tf.sqrt(tf.maximum(sq_per_cand / (B_f * T_dim_f), 0.0))
        # RRMSE: sqrt(SS_res / SS_tot).
        rrmse_per_cand = tf.sqrt(tf.maximum(sq_per_cand / ss_tot_batch, 0.0))
        self._last_rmse_per_cand = rmse_per_cand
        self._last_rrmse_per_cand = rrmse_per_cand

        return fitness

    @tf.function(reduce_retracing=True)
    def _evaluate_chunk(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        return self._evaluate_chunk_impl(chunk_samples, batch_data)

    def _evaluate_chunk_impl(self, chunk_samples: tf.Tensor, batch_data: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        """C candidates [C, dim] on B structures → (fitness_err, sq_err), each
        [C, B]: fitness_err is the (optionally lambda_shear-weighted) squared
        error for ranking, sq_err the unweighted one for RMSE/RRMSE reporting."""
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

        _scale_preds = self.cfg.scale_targets and self.cfg.target_mode == 1
        if _scale_preds:
            num_atoms = tf.reduce_sum(amask, axis=1)  # [B]
            inv_num_atoms = 1.0 / tf.maximum(num_atoms, 1.0)  # [B]
            inv_num_atoms = inv_num_atoms[:, tf.newaxis]  # [B, 1]

        params = self.reconstruct_params_tf(chunk_samples)
        named = self._split_reconstructed(params)
        W0 = named.get("W0"); b0 = named.get("b0")
        W1 = named.get("W1"); b1 = named.get("b1")
        U_pair_cand = named.get("U_pair")

        # W_atom [B,A,3,Q] is candidate-independent: evaluate_population stashes
        # it on batch_data["_W_atom"]; inline fallback for other callers (score).
        W_atom = batch_data.get("_W_atom")
        if W_atom is None and self.cfg.target_mode in (1, 2):
            if self.model._neighbor_readout:
                # V [B,A,D,C,T] in the W_atom slot; "replace" must not skip it.
                W_atom = batch_data["V_atom"]
            elif self.model._nep_mode != "replace":
                # replace+tnep: μ is NEP-only, no SOAP kernel.
                B_static = desc.shape[0]
                A_static = desc.shape[1]
                B_arg = (B_static if B_static is not None
                         else tf.shape(desc)[0])
                A_arg = (A_static if A_static is not None
                         else tf.shape(desc)[1])
                W_atom = self.model._precompute_dipole_kernel(
                    grad_values, pair_struct, pair_atom, pair_gidx,
                    pos, boxes,
                    B_arg, A_arg)

        # One GEMM per type per direction, not C matmuls inside vectorized_map.
        preds = self.model.predict_batch_candidates(
            desc, W_atom, Z, amask, W0, b0, W1, b1,
            U_pair=U_pair_cand,
            c_pair=named.get("c_pair"),
            M_radial=named.get("M_radial"),
            c_r=named.get("c_r"), c_a=named.get("c_a"),
            experts=self._experts_of(named),
            sum_delta=named.get("sum_delta"),
            nep_raw=batch_data.get("nep_raw"),
            **nep_pair_kwargs(batch_data))  # [C, B, T_dim]

        if _scale_preds:
            preds = preds * inv_num_atoms[tf.newaxis]  # [C, B, T_dim] * [1, B, 1]

        diff = preds - targets[tf.newaxis]  # [C, B, T_dim]
        # Mode 2: lambda_shear² on the off-diagonal components (ranking only).
        fitness = squared_error_per_structure(diff, self._pol_weights)
        sq = (fitness if self._pol_weights is None
              else squared_error_per_structure(diff))  # unweighted, for RMSE/RRMSE
        return fitness, sq
