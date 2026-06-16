from __future__ import annotations

import math
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

    Tail conventions produced by SNES.reconstruct_params_tf (legacy /
    extended) — the order of fields per ANN is:
        single hidden : W0, b0, W1, b1
        two hidden    : W0, b0, W0_2, b0_2, W1, b1
    Followed (when mixing enabled) by:
        N==1 layer    : a single U_pair tensor
        N>1 layers    : a list of N U_pair tensors (descriptor_mixing_n_layers)
    And finally (when nonlinear mixing on):
        b_mix_list    : a list of N bias tensors
    Parses each section in order so all four feature combinations work.
    """
    has_h2 = getattr(model, "W0_2", None) is not None
    per_l = getattr(model, "per_l_heads", False)
    params = list(params)
    idx = 0
    if per_l:
        # Per-l head ANN: each ANN supplies 4 list-of-L entries.
        for var_list in (model.W0_per_l, model.b0_per_l,
                          model.W1_per_l, model.b1_per_l):
            entry = params[idx]; idx += 1
            for l, t in enumerate(entry):
                var_list[l].assign(t)
        if model.cfg.target_mode == 2:
            for var_list in (model.W0_pol_per_l, model.b0_pol_per_l,
                              model.W1_pol_per_l, model.b1_pol_per_l):
                entry = params[idx]; idx += 1
                for l, t in enumerate(entry):
                    var_list[l].assign(t)
    else:
        # Primary ANN
        model.W0.assign(params[idx]); idx += 1
        model.b0.assign(params[idx]); idx += 1
        if has_h2:
            model.W0_2.assign(params[idx]); idx += 1
            model.b0_2.assign(params[idx]); idx += 1
        model.W1.assign(params[idx]); idx += 1
        model.b1.assign(params[idx]); idx += 1
        # Polarizability ANN
        if model.cfg.target_mode == 2:
            model.W0_pol.assign(params[idx]); idx += 1
            model.b0_pol.assign(params[idx]); idx += 1
            if has_h2:
                model.W0_2_pol.assign(params[idx]); idx += 1
                model.b0_2_pol.assign(params[idx]); idx += 1
            model.W1_pol.assign(params[idx]); idx += 1
            model.b1_pol.assign(params[idx]); idx += 1
    # Mixing tail (per-layer U_pair + optional b_mix lists)
    if getattr(model, "descriptor_mixing", False) and idx < len(params):
        U_pair_entry = params[idx]; idx += 1
        if isinstance(U_pair_entry, list):
            for k, U_k in enumerate(U_pair_entry):
                model.U_pair_list[k].assign(U_k)
        else:
            model.U_pair_list[0].assign(U_pair_entry)
        # Optional bias list (only present when nonlinear mixing is on).
        if idx < len(params) and getattr(model, "descriptor_mixing_nonlinear", False):
            b_mix_entry = params[idx]; idx += 1
            for k, b_k in enumerate(b_mix_entry):
                model.b_mix_list[k].assign(b_k)
        # Optional cross-channel mixing layer (single [Q, Q] V_cross).
        if (idx < len(params)
                and getattr(model, "descriptor_mixing_cross_layer", False)
                and getattr(model, "V_cross", None) is not None):
            model.V_cross.assign(params[idx]); idx += 1
    # Optional per-(pair, l, type) gating tail. Lives OUTSIDE the mixing
    # block: gating is allowed independent of descriptor_mixing.
    if (idx < len(params)
            and getattr(model, "descriptor_gating_enabled", False)
            and getattr(model, "gates_pair_l", None) is not None):
        model.gates_pair_l.assign(params[idx]); idx += 1
    # Optional preprocess tail (descriptor_preprocess_contract). Per-type
    # coefficients [T, Q_raw] feed _W0_preprocess_eff. Mutually exclusive
    # with mixing/gating but routed at the same set-model-params layer.
    if (idx < len(params)
            and getattr(model, "descriptor_preprocess_contract", "off") != "off"
            and getattr(model, "W_pre_angular", None) is not None):
        model.W_pre_angular.assign(params[idx]); idx += 1
    # Output-side mixing R tail: V_R = R − I of shape [T, H, H].
    if (idx < len(params)
            and getattr(model, "descriptor_mixing_output_layer", False)
            and getattr(model, "R_pair", None) is not None):
        model.R_pair.assign(params[idx]); idx += 1

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
        # Cache widths for new optional second hidden layer.
        self.H = int(self.cfg.num_neurons)
        H2_cfg = getattr(self.cfg, "num_neurons_layer_2", None)
        self.H2 = int(H2_cfg) if H2_cfg is not None else None
        self.H_final = self.H2 if self.H2 is not None else self.H

        # Per-(type, l) ANN heads mode. Replaces the single-ANN-per-type
        # layout with per-l heads, each seeing only Q_l channels.
        self._per_l_heads = bool(getattr(
            self.cfg, "descriptor_per_l_ann_heads", False))
        if self._per_l_heads:
            # Cross-feature guards (restrictions for the first impl).
            if bool(getattr(self.cfg, "descriptor_mixing", False)):
                raise NotImplementedError(
                    "descriptor_per_l_ann_heads + descriptor_mixing is not "
                    "yet supported. Disable one or the other.")
            if self.H2 is not None:
                raise NotImplementedError(
                    "descriptor_per_l_ann_heads + num_neurons_layer_2 is "
                    "not yet supported. Set num_neurons_layer_2=None.")
            from DescriptorBuilderGPU import descriptor_block_layout
            _layout = descriptor_block_layout(self.cfg)
            L = int(self.cfg.l_max) + 1
            self._per_l_L = L
            # Q_l = number of descriptor channels at that l (sum across pairs).
            self._per_l_Q_l = []
            for l in range(L):
                Q_l = sum(
                    len(_layout["pair_ln_index"][p][l])
                    for p in _layout["pair_keys"]
                    if l in _layout["pair_ln_index"][p])
                self._per_l_Q_l.append(int(Q_l))
            # Per-l per-ANN params: W0_l (T·Q_l·H) + b0_l (T·H) + W1_l (T·H) + b1_l (1)
            T_ = int(self.cfg.num_types)
            n_W0 = sum(T_ * Q_l * self.H for Q_l in self._per_l_Q_l)
            n_b0 = T_ * self.H * L
            n_W1 = T_ * self.H * L
            n_b1 = L                                              # one b1 per l
            n_W0_2 = 0; n_b0_2 = 0
            self.n_typed = n_W0 + n_b0 + n_W1                     # excludes b1
            # Per-type param count (per type t, summed over l):
            #   sum_l (Q_l · H + H + H) = Q·H + 2·L·H
            self._n_per_type = (self.cfg.dim_q * self.H
                                + 2 * L * self.H)
            self.n_primary = self.n_typed + n_b1
            self._n_W0 = n_W0
            self._n_b0 = n_b0
            self._n_W0_2 = n_W0_2
            self._n_b0_2 = n_b0_2
            self._n_W1 = n_W1
            self._n_b1 = n_b1
        else:
            n_W0 = self.cfg.num_types * self.cfg.dim_q * self.H
            n_b0 = self.cfg.num_types * self.H
            if self.H2 is not None:
                n_W0_2 = self.cfg.num_types * self.H * self.H2
                n_b0_2 = self.cfg.num_types * self.H2
            else:
                n_W0_2 = 0
                n_b0_2 = 0
            n_W1 = self.cfg.num_types * self.H_final
            n_b1 = 1
            self.n_typed = n_W0 + n_b0 + n_W0_2 + n_b0_2 + n_W1
            self._n_per_type = (self.cfg.dim_q * self.H + self.H
                                + (self.H * self.H2 + self.H2 if self.H2 is not None else 0)
                                + self.H_final)
            self.n_primary = self.n_typed + n_b1
            self._n_W0 = n_W0
            self._n_b0 = n_b0
            self._n_W0_2 = n_W0_2
            self._n_b0_2 = n_b0_2
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
                # Use the layout's L_eff when present (post-preprocess
                # layouts collapse l > l_keep into a single slot, so
                # L_eff < L). Falls back to l_max+1 for the raw layout.
                L = int(self._mix_layout.get("L_eff", int(self.cfg.l_max) + 1))
                self._mix_L = L
                per_T_block = int(sum(L * _block_count(a)
                                       for a in self._mix_alpha_per_pair))
            elif self._mix_arch == "cross_pair_l":
                L = int(self._mix_layout.get("L_eff", int(self.cfg.l_max) + 1))
                self._mix_L = L
                self._mix_N_per_l = int(self._mix_layout["N_per_l"])
                per_T_block = L * _block_count(self._mix_N_per_l)
            else:
                raise ValueError(
                    f"descriptor_mixing_arch={self._mix_arch!r} not in "
                    "('linear', 'l_aware', 'cross_pair_l')")
            if self._mix_per_type:
                self.n_U_pair_per_layer = self.cfg.num_types * per_T_block
            else:
                self.n_U_pair_per_layer = per_T_block
            # Stacked mixing (N layers) multiplies the per-layer count.
            self._mix_n_layers = int(getattr(
                self.cfg, "descriptor_mixing_n_layers", 1))
            self._mix_nonlinear = bool(getattr(
                self.cfg, "descriptor_mixing_nonlinear", False))
            self.n_U_pair = self._mix_n_layers * self.n_U_pair_per_layer
            self._mix_n_U_pair_per_layer = self.n_U_pair_per_layer
            # Optional per-layer descriptor bias for nonlinear mixing.
            if self._mix_nonlinear:
                # b_mix_k shape: [T, dim_q] (per-type) or [dim_q] (shared).
                n_b_per_layer = (self.cfg.num_types if self._mix_per_type else 1) \
                    * self.dim_q
            else:
                n_b_per_layer = 0
            self._mix_n_bias_per_layer = n_b_per_layer
            self.n_U_bias_total = self._mix_n_layers * n_b_per_layer
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
            if (self._mix_arch == "l_aware"
                    and self._mix_cayley
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
            self._mix_block_sizes = []
            self._mix_per_type = False
            self._mix_arch = "linear"
            self._mix_cayley = False
            self._mix_orth_map = None
            self._cayley_scatter_cache = {}
            self.n_U_pair = 0
            self.n_U_pair_per_layer = 0
            self._mix_n_layers = 1
            self._mix_nonlinear = False
            self._mix_n_U_pair_per_layer = 0
            self._mix_n_bias_per_layer = 0
            self.n_U_bias_total = 0

        # Cross-channel mixing layer (a single orthogonal rotation applied
        # AFTER the existing arch). Two modes selected by
        # cfg.descriptor_mixing_cross_mode:
        #   "full"       : R ∈ SO(Q), full per-component rotation.
        #   "block_unit" : R ∈ SO(N_units), rotates (pair, l) units while
        #                  keeping each unit's α components in their slot.
        # Shared across central atom types in both modes.
        self._cross_enabled = (
            bool(getattr(self.cfg, "descriptor_mixing", False))
            and bool(getattr(self.cfg, "descriptor_mixing_cross_layer", False)))
        if self._cross_enabled:
            self._cross_orth_map = str(getattr(
                self.cfg, "descriptor_mixing_cross_regularizer", "expm")).lower()
            if self._cross_orth_map not in ("cayley", "expm"):
                raise ValueError(
                    f"descriptor_mixing_cross_regularizer="
                    f"{self._cross_orth_map!r} must be 'cayley' or 'expm'")
            self._cross_mode = str(getattr(
                self.cfg, "descriptor_mixing_cross_mode", "full")).lower()
            if self._cross_mode not in ("full", "block_unit", "per_l"):
                raise ValueError(
                    f"descriptor_mixing_cross_mode={self._cross_mode!r} "
                    "must be 'full', 'block_unit', or 'per_l'")
            Q = int(self.dim_q)
            self._cross_Q = Q
            if self._cross_mode == "full":
                # Cayley/expm parameterisation: upper-triangle of Q×Q skew.
                self._cross_R_size = Q
                self.n_U_cross = Q * (Q - 1) // 2
            elif self._cross_mode == "block_unit":
                # block_unit: per-component-slot R_k matrices. Each (pair, l)
                # sub-block is a UNIT; for each component slot k, R_k is an
                # orthogonal rotation on the subset of units that have slot
                # k (i.e. α_unit > k). Total parameters:
                #   n_U_cross = Σ_k N_k(N_k-1)/2
                # where N_k = number of units with α > k. Handles non-uniform
                # α correctly (each R_k is exactly orthogonal on its slot).
                from DescriptorBuilderGPU import descriptor_block_layout
                layout = descriptor_block_layout(self.cfg)
                # Sort units in deterministic order: pair-major, l-minor.
                units = []  # list of (pair, l, [q_indices])
                for pair in layout["pair_keys"]:
                    for l, q_idx_list in sorted(
                            layout["pair_ln_index"][pair].items()):
                        units.append((pair, l, list(q_idx_list)))
                max_alpha = max(len(u[2]) for u in units)
                slot_q_indices = [[] for _ in range(max_alpha)]
                for u_idx, (_, _, q_idx) in enumerate(units):
                    for k, q in enumerate(q_idx):
                        slot_q_indices[k].append(q)
                self._cross_slot_N = [len(s) for s in slot_q_indices]
                self._cross_slot_n_payload = [
                    n * (n - 1) // 2 for n in self._cross_slot_N]
                self._cross_slot_offsets = []
                cursor = 0
                for n_pay in self._cross_slot_n_payload:
                    self._cross_slot_offsets.append(cursor)
                    cursor += n_pay
                self.n_U_cross = cursor
                self._cross_slot_q_indices = [
                    tf.constant(np.asarray(s, dtype=np.int64))
                    for s in slot_q_indices]
                self._cross_slot_upper = []
                for n in self._cross_slot_N:
                    if n >= 2:
                        ii, jj = np.triu_indices(n, k=1)
                        self._cross_slot_upper.append((
                            tf.constant(ii.astype(np.int64)),
                            tf.constant(jj.astype(np.int64))))
                    else:
                        self._cross_slot_upper.append((None, None))
                self._cross_max_alpha = max_alpha
                self._cross_R_size = max(self._cross_slot_N)
            else:
                # per_l: rotate pair-units AT EACH l independently, with
                # per-(l, k) R matrices for non-uniform α within an l.
                # Preserves angular-momentum boundaries — the rotation at
                # l=0 never touches l=1 channels and vice-versa.
                #   For each l ∈ {0..l_max}:
                #     For each component slot k ∈ {0..max_α_l−1}:
                #       N_{l,k} = number of pairs at l with α_p > k
                #       R_{l,k} ∈ SO(N_{l,k})
                #   Total params: Σ_l Σ_k N_{l,k}(N_{l,k}−1)/2
                # Flattened representation reuses the per-slot machinery,
                # just with one "slot" per (l, k) pair.
                from DescriptorBuilderGPU import descriptor_block_layout
                layout = descriptor_block_layout(self.cfg)
                pair_keys = layout["pair_keys"]
                # Determine the full l range across all pairs.
                all_ls = sorted({l for p in pair_keys
                                 for l in layout["pair_ln_index"][p]})
                self._cross_per_l_ls = all_ls
                # Per (l, k) bookkeeping — same structure as block_unit's
                # per-slot, but flattened over both l and k.
                slot_q_indices = []         # list of [q_indices] per (l, k)
                slot_meta = []              # list of (l, k) labels for diag
                for l in all_ls:
                    # Pairs that have channels at this l, and their slot
                    # sizes at this l.
                    pairs_at_l = [p for p in pair_keys
                                  if l in layout["pair_ln_index"][p]]
                    max_alpha_l = max(
                        len(layout["pair_ln_index"][p][l])
                        for p in pairs_at_l)
                    for k in range(max_alpha_l):
                        qs = []
                        for p in pairs_at_l:
                            block = layout["pair_ln_index"][p][l]
                            if k < len(block):
                                qs.append(block[k])
                        slot_q_indices.append(qs)
                        slot_meta.append((l, k))
                self._cross_slot_N = [len(s) for s in slot_q_indices]
                self._cross_slot_n_payload = [
                    n * (n - 1) // 2 for n in self._cross_slot_N]
                self._cross_slot_offsets = []
                cursor = 0
                for n_pay in self._cross_slot_n_payload:
                    self._cross_slot_offsets.append(cursor)
                    cursor += n_pay
                self.n_U_cross = cursor
                self._cross_slot_q_indices = [
                    tf.constant(np.asarray(s, dtype=np.int64))
                    for s in slot_q_indices]
                self._cross_slot_upper = []
                for n in self._cross_slot_N:
                    if n >= 2:
                        ii, jj = np.triu_indices(n, k=1)
                        self._cross_slot_upper.append((
                            tf.constant(ii.astype(np.int64)),
                            tf.constant(jj.astype(np.int64))))
                    else:
                        self._cross_slot_upper.append((None, None))
                self._cross_per_l_meta = slot_meta
                self._cross_R_size = max(self._cross_slot_N) if self._cross_slot_N else 0
            # full-mode upper-tri (only used in full mode).
            if self._cross_mode == "full":
                R_size = self._cross_R_size
                i_idx, j_idx = np.triu_indices(R_size, k=1)
                self._cross_upper_i = tf.constant(i_idx.astype(np.int64))
                self._cross_upper_j = tf.constant(j_idx.astype(np.int64))
        else:
            self._cross_orth_map = None
            self._cross_mode = "full"
            self.n_U_cross = 0
            self._cross_Q = 0
            self._cross_R_size = 0

        # Per-(species-pair, l, central-type) gating (descriptor_gating_enabled).
        # Folded into the first-layer W0 via W0_eff[t, c, h] *= g_{t, pair(c), l(c)}.
        self._gating_enabled = bool(getattr(
            self.cfg, "descriptor_gating_enabled", False))
        if self._gating_enabled:
            from DescriptorBuilderGPU import descriptor_block_layout
            layout = descriptor_block_layout(self.cfg)
            pair_keys = layout["pair_keys"]
            L = int(self.cfg.l_max) + 1
            self._gating_num_pairs = len(pair_keys)
            self._gating_L = L
            self.n_gates = (int(self.cfg.num_types)
                            * self._gating_num_pairs * L)
            # Build the q-index → (pair_idx · L + l_idx) lookup once.
            # gate_per_channel[t, q] = gates_flat[t, q_to_pair_l[q]].
            pair_idx_of = {p: i for i, p in enumerate(pair_keys)}
            q_to_pair_l = np.full(int(self.dim_q), -1, dtype=np.int64)
            for p in pair_keys:
                pi = pair_idx_of[p]
                for l, q_idx_list in layout["pair_ln_index"][p].items():
                    for q in q_idx_list:
                        q_to_pair_l[int(q)] = pi * L + int(l)
            if (q_to_pair_l < 0).any():
                missing = int((q_to_pair_l < 0).sum())
                raise RuntimeError(
                    f"descriptor_gating: {missing} q-channels not covered "
                    "by pair_ln_index; layout / dim_q mismatch.")
            self._gating_q_to_pair_l = tf.constant(q_to_pair_l)
            self._gating_init_value = float(getattr(
                self.cfg, "descriptor_gating_init", 1.0))
        else:
            self.n_gates = 0
            self._gating_num_pairs = 0
            self._gating_L = 0
            self._gating_q_to_pair_l = None
            self._gating_init_value = 1.0

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

        # Output-side mixing R tail: per-type orthogonal [H, H] matrix
        # parameterised by skew upper-triangle (T · H · (H − 1) / 2 dims
        # under cayley/expm). Residual V_R = R − I; SNES walks the skew
        # params, _cayley_blocks_batched reconstructs V_R each candidate.
        if bool(getattr(self.model, "descriptor_mixing_output_layer", False)):
            self._R_H = int(self.model._R_H)
            self._R_per_T = int(self.cfg.num_types)
            # Skew param count per block of size H.
            self._R_skew_per_block = self._R_H * (self._R_H - 1) // 2
            self.n_R_pair = self._R_per_T * self._R_skew_per_block
        else:
            self._R_H = 0
            self._R_per_T = 0
            self._R_skew_per_block = 0
            self.n_R_pair = 0

        self.dim = (self.n_anns_total + self.n_U_pair
                    + self.n_U_bias_total + self.n_U_cross
                    + self.n_gates + self.n_preprocess + self.n_R_pair)

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
            mix_end = mix_start + self.n_U_pair + self.n_U_bias_total
            sigma_init_vec[mix_start:mix_end] *= mix_scale
        # Per-block σ for the gating tail (cfg.gating_sigma_scale). Lets
        # gates be explored at a smaller σ than the ANN — important when
        # regularisation is supposed to drive the equilibrium but per-gen
        # noise at the default σ would dominate.
        gate_scale = float(getattr(self.cfg, "gating_sigma_scale", 1.0))
        if self.n_gates > 0 and gate_scale != 1.0:
            g_start = (self.n_anns_total + self.n_U_pair
                       + self.n_U_bias_total + self.n_U_cross)
            sigma_init_vec[g_start:g_start + self.n_gates] *= gate_scale
        # Per-block σ for the preprocess tail (cfg.preprocess_sigma_scale).
        # Same rationale as gating: SNES per-gen noise on coefficients
        # near 1/L can overwhelm the optimisation signal at the legacy σ.
        preprocess_scale = float(getattr(
            self.cfg, "preprocess_sigma_scale", 1.0))
        if self.n_preprocess > 0 and preprocess_scale != 1.0:
            pre_start = (self.n_anns_total + self.n_U_pair
                         + self.n_U_bias_total + self.n_U_cross
                         + self.n_gates)
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

        # V_pair regulariser mode: "off" | "shrinkage" | "orthogonal".
        # Orthogonal penalty has its own lambda (defaults to the same
        # auto formula scaled by n_U_pair only, since the orth penalty
        # is naturally per-mixing-entry and shouldn't inherit the ANN's
        # dimensionality scaling). -1 sentinel enables dynamic adapt.
        self._mix_reg_mode = str(getattr(
            self.cfg, "descriptor_mixing_regularizer", "off")).lower()
        if self._mix_reg_mode not in (
                "off", "shrinkage", "orthogonal", "cayley", "expm"):
            raise ValueError(
                f"descriptor_mixing_regularizer={self._mix_reg_mode!r} not "
                "recognised (expected 'off', 'shrinkage', 'orthogonal', "
                "'cayley', or 'expm')")
        # The "cayley" and "expm" modes are *parameterisations*, not soft
        # penalties: U is reconstructed structurally from a skew-symmetric A
        # (via either the Cayley rational chord or the matrix exponential),
        # so it's exactly orthogonal regardless of any λ. The
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

        # Gating regularisation strengths. Penalty form per type t:
        #   L1_g[t] = λ_g1 · ‖g[t, :] − g_init‖_1 / n_gates_per_type
        #   L2_g[t] = λ_g2 · √(‖g[t, :] − g_init‖_2² / n_gates_per_type)
        # Both default to 0.0; gating is only regularised if either is > 0.
        self._lambda_gate_1 = tf.Variable(
            float(getattr(cfg, "descriptor_gating_lambda_1", 0.0)),
            dtype=tf.float32, trainable=False, name="lambda_gate_1")
        self._lambda_gate_2 = tf.Variable(
            float(getattr(cfg, "descriptor_gating_lambda_2", 0.0)),
            dtype=tf.float32, trainable=False, name="lambda_gate_2")
        # Anti-sparsity floor barrier: hinge penalty on max(0, floor − |g|)².
        # Active only when |g| dips below `floor`; zero otherwise. Prevents
        # input channels from being killed off without constraining gates
        # that grow above the floor.
        self._lambda_gate_floor = tf.Variable(
            float(getattr(cfg, "descriptor_gating_lambda_floor", 0.0)),
            dtype=tf.float32, trainable=False, name="lambda_gate_floor")
        self._gating_floor = tf.constant(
            float(getattr(cfg, "descriptor_gating_floor", 0.2)),
            dtype=tf.float32, name="gating_floor")

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
        # recombination weights (CMA convention; Hansen 2016 Eq. 5). The
        # cached `_recomb_w` (set inside compute_utilities) is exactly that
        # quantity regardless of `snes_active_utilities` — using
        # `self.utilities` directly would silently change mu_eff when
        # active utilities are enabled, which breaks every downstream
        # constant that depends on mu_eff (c_c, c_1, eta_B, etc.).
        recomb_w_np = self._recomb_w.numpy()
        self._mu_eff = float(1.0 / np.sum(recomb_w_np ** 2))

        # ES-upgrade state (lazy — only allocated when the opt-in branches fire).
        # Adam preconditioning of the ES mean gradient (Feature A):
        self._es_m = None
        self._es_v = None
        self._es_t = None
        # Per-coordinate evolution-path cumulation for sigma (Feature B):
        self._grad_sigma_ema = None

        # MSR state (lazy — only materialised when snes_msr_enabled).
        # Paper-faithful (Ait ElHara, Auger, Hansen, GECCO 2013):
        #   z_t = (2/λ)·(K_succ − (λ+1)/2)              # success signal, ∈[−1,1]
        #   p_s ← (1−c_σ)·p_s + c_σ·z_t                  # EMA smoothing
        #   σ   ← σ·exp(p_s / d_σ)                       # exponentiate smoothed
        # d_σ ≈ 2 − 2/n damping, c_σ ≈ 0.3 EMA decay (paper recommendations).
        # `_msr_z_target` is retained for backward compatibility but UNUSED
        # — paper's z formula centers at 0 automatically.
        self._msr_f_prev = None
        self._msr_z_target = float(getattr(cfg, "snes_msr_z_target", 0.5))  # deprecated
        self._msr_c_sigma = float(getattr(cfg, "snes_msr_c_sigma", 0.3))    # EMA decay
        self._msr_clip = float(getattr(cfg, "snes_msr_clip", 0.3))
        self._msr_ps = 0.0          # EMA-smoothed success signal
        self._msr_initialised = False

        # NB: self._recomb_w (CMA recombination weights, sum 1) is populated by
        # compute_utilities() above — do NOT reset it here or the cache is lost.
        # Active CR-FM-NES distance regime needs the MIRRORED negative log-rank
        # weights (sum-1, absolute-value form). Lazy-built on first use in
        # _crfmnes_weights to avoid populating it for non-active-non-crfmnes runs.
        self._neg_recomb_abs = None

        # IPOP / BIPOP restart state.
        # _initial_pop_size : the configured pop_size at __init__ time, used
        #                     as the BIPOP small-regime reference.
        # _restart_count    : number of completed restarts.
        # _bipop_use_large  : BIPOP regime selector — True ⇒ next restart is
        #                     "large" (IPOP-style λ doubling); False ⇒ "small"
        #                     (λ ≈ small_factor · λ_initial). Alternates.
        self._initial_pop_size = int(self.pop_size)
        self._restart_count = 0
        self._bipop_use_large = True

        # CR-FM-NES state (lazy — only materialised when cov_mode="crfmnes").
        # NB: NO `_cr_lf` state — λ_F is per-gen feasible count, recomputed in
        # update() each generation (C0 finding, crfmnes/alg.py:138).
        self._cr_v   = None    # tf.Variable [dim]
        self._cr_D   = None    # tf.Variable [dim]
        self._cr_psg = None    # tf.Variable [dim]  — conjugate path p_σ
        self._cr_pc  = None    # tf.Variable [dim]  — evolution path p_c
        self._cr_sig = None    # tf.Variable scalar — overall step size σ
        self._cr_chi = None    # cached χ_d
        self._cr_h_inv = None  # cached dim-dependent Newton root (see C0)

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
        n_per_type = self._n_per_type  # W0_t + b0_t (+ W0_2_t + b0_2_t) + W1_t

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
            # Bound the slice to the V_pair region only; the nonlinear bias
            # tail (n_U_bias_total > 0 when descriptor_mixing_nonlinear=True)
            # lives in residual form and is excluded from this shrinkage.
            tail = pv[self.n_anns_total : self.n_anns_total + self.n_U_pair]
            if self._mix_per_type and T > 1:
                # reconstruct_params_tf produces layers-major, types-inner:
                #   [L0_t0 | L0_t1 | ... | L0_tT-1 | L1_t0 | ... | LN-1_tT-1]
                # So per-type slab for type t is the union of (L_k, t)
                # sub-blocks across all N layers.
                per_layer = self.n_U_pair_per_layer
                per_T_block = per_layer // T
                vp_l1 = tf.constant(0.0)
                vp_l2 = tf.constant(0.0)
                for t in range(T):
                    if self._mix_n_layers == 1:
                        slab = tail[t * per_T_block:(t + 1) * per_T_block]
                    else:
                        slab = tf.concat(
                            [tail[k * per_layer + t * per_T_block
                                  : k * per_layer + (t + 1) * per_T_block]
                             for k in range(self._mix_n_layers)], axis=0)
                    n_per_t = per_T_block * self._mix_n_layers
                    vp_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(slab)) / n_per_t
                    vp_l2 += self.lambda_2 * tf.sqrt(
                        tf.reduce_sum(tf.square(slab)) / n_per_t)
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
            # See reg_Vpair branch: bound the slice to the V_pair region.
            tail = pv[self.n_anns_total : self.n_anns_total + self.n_U_pair]
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
        H = self.H
        H2 = self.H2
        H_final = self.H_final
        if scheme == "uniform":
            mu = rng.uniform(-1.0, 1.0, size=self.dim).astype(np.float32)
        elif scheme == "glorot":
            mu = np.zeros(self.dim, dtype=np.float32)
            # Activation-dependent Glorot gain (see _ACTIVATION_REGISTRY in
            # TNEP.py). gain=1 reproduces the legacy tanh init exactly; other
            # activations push more variance through, so this scales the
            # uniform bound to match the activation's slope near origin.
            gain = float(getattr(self.model, "_glorot_gain", 1.0))
            c_W0 = gain * float(np.sqrt(6.0 / (Q + H)))
            c_W1 = gain * float(np.sqrt(6.0 / (H_final + 1)))
            c_W0_2 = (gain * float(np.sqrt(6.0 / (H + (H2 or H))))
                      if H2 is not None else 0.0)

            def _fill_ann(off: int) -> int:
                """Fill one ANN's worth of weights starting at `off`.
                Layout: [W0(T,Q,H) | b0(T,H) | (W0_2(T,H,H2) | b0_2(T,H2))?
                          | W1(T,H_final) | b1(1)].
                """
                n_W0 = T * Q * H
                mu[off:off + n_W0] = rng.uniform(
                    -c_W0, c_W0, size=n_W0).astype(np.float32)
                off += n_W0
                # b0 zero
                off += T * H
                if H2 is not None:
                    n_W0_2 = T * H * H2
                    mu[off:off + n_W0_2] = rng.uniform(
                        -c_W0_2, c_W0_2, size=n_W0_2).astype(np.float32)
                    off += n_W0_2
                    # b0_2 zero
                    off += T * H2
                n_W1 = T * H_final
                mu[off:off + n_W1] = rng.uniform(
                    -c_W1, c_W1, size=n_W1).astype(np.float32)
                off += n_W1
                # b1 zero
                off += 1
                return off

            def _fill_ann_per_l(off: int) -> int:
                """Fill one ANN's worth of per-l-head weights.
                Layout per ANN (per l in 0..L-1):
                    W0_l(T, Q_l, H) | b0_l(T, H) | W1_l(T, H) | b1_l(1)
                """
                L = self._per_l_L
                for l in range(L):
                    Q_l = self._per_l_Q_l[l]
                    # Per-head Glorot bound (scaled by the activation gain
                    # to match the outer-ANN init convention).
                    c_W0_l = (gain * float(np.sqrt(6.0 / (Q_l + H)))
                              if Q_l > 0 else 0.0)
                    n_W0_l = T * Q_l * H
                    if n_W0_l > 0:
                        mu[off:off + n_W0_l] = rng.uniform(
                            -c_W0_l, c_W0_l, size=n_W0_l).astype(np.float32)
                    off += n_W0_l
                    # b0_l zero
                    off += T * H
                    # W1_l ~ U(-c_W1, c_W1) with c_W1 = sqrt(6/(H+1))
                    n_W1_l = T * H
                    mu[off:off + n_W1_l] = rng.uniform(
                        -c_W1, c_W1, size=n_W1_l).astype(np.float32)
                    off += n_W1_l
                    # b1_l zero
                    off += 1
                return off

            if self._per_l_heads:
                off = _fill_ann_per_l(0)
                if self.cfg.target_mode == 2:
                    off = _fill_ann_per_l(off)
            else:
                off = _fill_ann(0)
                if self.cfg.target_mode == 2:
                    off = _fill_ann(off)
        else:
            raise ValueError(
                f"mu_init_scheme={scheme!r} not in ('uniform', 'glorot')")
        # V_pair tail (including all N stacked layers) + nonlinear biases:
        # always zero (residual mixing layer; U_full = I at init; biases = 0).
        if self.n_U_pair > 0 or self.n_U_bias_total > 0 or self.n_U_cross > 0:
            mu[self.n_anns_total:self.n_anns_total + self.n_U_pair
               + self.n_U_bias_total + self.n_U_cross] = 0.0
        # Gating tail: init at cfg.descriptor_gating_init (default 1.0)
        # so the gates are identity at gen 0 — model behaviour at gen 0
        # is bit-identical to the no-gating baseline.
        if self.n_gates > 0:
            g_start = (self.n_anns_total + self.n_U_pair
                       + self.n_U_bias_total + self.n_U_cross)
            mu[g_start:g_start + self.n_gates] = self._gating_init_value
        # Preprocess tail: init per cfg.descriptor_preprocess_init
        # ("mean" → 1/L, "sum" → 1.0, "glorot" → 0.0 with σ providing the
        # spread). Per-type ranking uses the [t, q_raw] interleaved layout
        # (t-major, q-inner) so consecutive blocks of Q_raw entries belong
        # to a single type — matches the _set_model_params reshape.
        if self.n_preprocess > 0:
            pre_start = (self.n_anns_total + self.n_U_pair
                         + self.n_U_bias_total + self.n_U_cross
                         + self.n_gates)
            # μ holds ONLY the summed entries (n_preprocess = number of
            # learnable W_pre slots). Read the per-summed init values
            # straight off the model's W_pre Variable, which TNEP already
            # populated with kept positions at 1.0 and summed positions
            # at the per-(t, q_raw)/per-q_raw "mean" / "sum" / "glorot"
            # init. For "glorot" SNES adds a small uniform jitter on top.
            W_pre_init = self.model.W_pre_angular.numpy().reshape(-1)
            flat_idx = self.model._preprocess_summed_flat_idx.numpy()
            summed_init = W_pre_init[flat_idx]
            if self._preprocess_init_scheme == "glorot":
                # μ starts near 0; σ provides the spread at gen 0.
                limit = 1e-3
                summed_init = rng.uniform(
                    -limit, limit, size=summed_init.size).astype(np.float32)
            mu[pre_start:pre_start + self.n_preprocess] = summed_init
        # Output-side R tail: residual init at zero → R = I at gen 0.
        # (The np.zeros background already sets these — explicit for clarity.)
        if self.n_R_pair > 0:
            r_start = (self.n_anns_total + self.n_U_pair
                       + self.n_U_bias_total + self.n_U_cross
                       + self.n_gates + self.n_preprocess)
            mu[r_start:r_start + self.n_R_pair] = 0.0
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
        # V_tail layout (from reconstruct_params_tf) is layers-major,
        # types-inner per layer. Sum across all N layers so stacked mixing
        # doesn't silently leave layers 1..N-1 unconstrained.
        per_layer = self.n_U_pair_per_layer
        per_T_block = (per_layer // self.cfg.num_types
                       if self._mix_per_type else per_layer)
        if self._mix_arch == "linear":
            iterator = [(bs,) for bs in self._mix_block_sizes]
        elif self._mix_arch == "l_aware":
            iterator = [(alpha,) for alpha in self._mix_alpha_per_pair
                        for _ in range(self._mix_L)]
        else:  # cross_pair_l: one [N_l × N_l] sub-block per angular momentum
            iterator = [(self._mix_N_per_l,) for _ in range(self._mix_L)]
        pen = tf.zeros(tf.shape(V_tail)[:-1])
        for k in range(self._mix_n_layers):
            start = k * per_layer + slab_idx * per_T_block
            offset = 0
            for (dim,) in iterator:
                n = dim * dim
                block_flat = V_tail[..., start + offset:start + offset + n]
                new_shape = tf.concat(
                    [tf.shape(block_flat)[:-1], [dim, dim]], axis=0)
                V = tf.reshape(block_flat, new_shape)
                VtV = tf.matmul(V, V, transpose_a=True)
                M = V + tf.linalg.matrix_transpose(V) + VtV
                pen = pen + tf.reduce_sum(tf.square(M), axis=[-2, -1])
                offset += n
        return pen / float(per_T_block * self._mix_n_layers)

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
        H2 = self.H2
        H_final = self.H_final

        # W0 block: [T, Q, H]
        w0_start = t * Q * H
        w0_end = w0_start + Q * H

        # b0 block: after all W0
        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H

        parts = [pv[w0_start:w0_end], pv[b0_start:b0_end]]

        # Optional second hidden layer
        if H2 is not None:
            w0_2_offset = b0_offset + T * H
            w0_2_start = w0_2_offset + t * H * H2
            w0_2_end = w0_2_start + H * H2
            b0_2_offset = w0_2_offset + T * H * H2
            b0_2_start = b0_2_offset + t * H2
            b0_2_end = b0_2_start + H2
            parts.extend([pv[w0_2_start:w0_2_end], pv[b0_2_start:b0_2_end]])
            w1_offset = b0_2_offset + T * H2
        else:
            w1_offset = b0_offset + T * H

        w1_start = w1_offset + t * H_final
        w1_end = w1_start + H_final
        parts.append(pv[w1_start:w1_end])

        return tf.concat(parts, axis=0)

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
        H2 = self.H2
        H_final = self.H_final

        def _ann_types() -> np.ndarray:
            if getattr(self, "_per_l_heads", False):
                # Per-l-head layout per ANN (per l):
                #   W0_l[T, Q_l, H] | b0_l[T, H] | W1_l[T, H] | b1_l[1]
                tov = np.empty(self.n_primary, dtype=np.int32)
                offset = 0
                for l in range(self._per_l_L):
                    Q_l = self._per_l_Q_l[l]
                    # W0_l: per-type
                    for t in range(T):
                        tov[offset:offset + Q_l * H] = t
                        offset += Q_l * H
                    # b0_l: per-type
                    for t in range(T):
                        tov[offset:offset + H] = t
                        offset += H
                    # W1_l: per-type
                    for t in range(T):
                        tov[offset:offset + H] = t
                        offset += H
                    # b1_l: global label per head
                    tov[offset] = T
                    offset += 1
                return tov
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
            if H2 is not None:
                # W0_2: [T, H, H2]
                for t in range(T):
                    tov[offset:offset + H * H2] = t
                    offset += H * H2
                # b0_2: [T, H2]
                for t in range(T):
                    tov[offset:offset + H2] = t
                    offset += H2
            # W1: [T, H_final]
            for t in range(T):
                tov[offset:offset + H_final] = t
                offset += H_final
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
        # U_pair tail: N consecutive layer slabs. Within a layer, the
        # per-type slab pattern is reused.
        if self.n_U_pair > 0:
            n_per_layer = self.n_U_pair_per_layer
            for _ in range(self._mix_n_layers):
                if self._mix_per_type:
                    per_T = n_per_layer // T
                    layer_labels = np.empty(n_per_layer, dtype=np.int32)
                    for t_idx in range(T):
                        layer_labels[t_idx * per_T:(t_idx + 1) * per_T] = t_idx
                else:
                    layer_labels = np.full(n_per_layer, T, dtype=np.int32)
                tail_labels_parts.append(layer_labels)
        # Nonlinear mixing biases (one per layer). Per-type: T slabs
        # of dim_q each; shared: all global label.
        if self.n_U_bias_total > 0:
            n_b_per_layer = self._mix_n_bias_per_layer
            for _ in range(self._mix_n_layers):
                if self._mix_per_type:
                    per_T = n_b_per_layer // T
                    layer_labels = np.empty(n_b_per_layer, dtype=np.int32)
                    for t_idx in range(T):
                        layer_labels[t_idx * per_T:(t_idx + 1) * per_T] = t_idx
                else:
                    layer_labels = np.full(n_b_per_layer, T, dtype=np.int32)
                tail_labels_parts.append(layer_labels)
        # Cross-channel mixing layer: a single rotation shared across all
        # central types, so all entries route to the global label.
        if self.n_U_cross > 0:
            tail_labels_parts.append(
                np.full(self.n_U_cross, T, dtype=np.int32))
        # Gating tail: g[t, :] is the slab for type t, contiguous
        # PL = num_pairs · (l_max+1) entries each. Per-type labels.
        if self.n_gates > 0:
            PL = self._gating_num_pairs * self._gating_L
            gate_labels = np.empty(self.n_gates, dtype=np.int32)
            for t_idx in range(T):
                gate_labels[t_idx * PL:(t_idx + 1) * PL] = t_idx
            tail_labels_parts.append(gate_labels)
        # Preprocess tail: W_pre[t, q_raw] stored t-major in the mu vector
        # → blocks of Q_raw entries belong to a single type. Per-type
        # labels mirror the gating layout.
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
        P = self.pop_size
        Q = self.dim_q
        H = self.cfg.num_neurons
        n_per_type = self._n_per_type

        # Optional V_pair tail routing. With per-type V_pair, slab t
        # is owned by label t (per-type ranking); with shared V_pair,
        # the whole tail is owned by the global label T. Dispatched on
        # cfg.descriptor_mixing_regularizer: "off" leaves V_pair alone,
        # "shrinkage" adds L1+L2 of the slab, "orthogonal" adds
        # λ_orth · ‖UᵀU - I‖² per block.
        reg_Vpair = (self.n_U_pair > 0 and self._mix_reg_mode == "shrinkage")
        reg_Vorth = (self.n_U_pair > 0 and self._mix_reg_mode == "orthogonal")
        # Per-type gating regulariser (L1 / L2 on g − g_init, plus a one-
        # sided anti-sparsity hinge on max(0, floor − |g|)²).
        lg1 = float(self._lambda_gate_1.numpy())
        lg2 = float(self._lambda_gate_2.numpy())
        lgf = float(self._lambda_gate_floor.numpy())
        reg_gate = self.n_gates > 0 and (lg1 > 0.0 or lg2 > 0.0 or lgf > 0.0)
        if reg_gate:
            PL = self._gating_num_pairs * self._gating_L
            g_start = (self.n_anns_total + self.n_U_pair
                       + self.n_U_bias_total + self.n_U_cross)
            g_tail = samples[:, g_start:g_start + self.n_gates]      # [P, T·PL]
            g_init = float(self._gating_init_value)
            g_dev = g_tail - g_init                                  # [P, T·PL]
            # Symmetric in sign: |g| sits inside the hinge, so g = -1.0
            # gets no penalty (still an informative channel) but g ≈ 0
            # does. Squared hinge keeps the penalty smooth at the boundary.
            g_floor_slack = tf.nn.relu(
                self._gating_floor - tf.abs(g_tail))                 # [P, T·PL]
            g_floor_sq = tf.square(g_floor_slack)
            n_per_t = PL
        if reg_Vpair or reg_Vorth:
            # Bound the slice to the V_pair region (excludes nonlinear bias tail).
            V_tail = samples[:, self.n_anns_total
                             : self.n_anns_total + self.n_U_pair]
            if self._mix_per_type:
                # Layers-major, types-inner per layer.
                _vp_per_layer = self.n_U_pair_per_layer
                _vp_per_T_block = _vp_per_layer // T
                _vp_n_per_t = _vp_per_T_block * self._mix_n_layers
            else:
                _vp_per_layer = None
                _vp_per_T_block = None
                _vp_n_per_t = None

        # Add per-type regularization to per-type RMSE → [T+1] fitness values
        fitness_per_type = []
        for t in range(T):
            type_params = self._extract_type_params_batched(samples, t)  # [P, n_per_type]
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(type_params), axis=1) / n_per_type
            l2 = self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(type_params), axis=1) / n_per_type)
            if reg_Vpair and self._mix_per_type:
                # Slab t routes to label t. With stacked layers, type-t's
                # slab is the union of (L_k, t) sub-blocks across all N
                # layers — see compute_regularization() for the layout.
                if self._mix_n_layers == 1:
                    slab = V_tail[:, t * _vp_per_T_block
                                    :(t + 1) * _vp_per_T_block]
                else:
                    slab = tf.concat(
                        [V_tail[:, k * _vp_per_layer + t * _vp_per_T_block
                                  : k * _vp_per_layer + (t + 1) * _vp_per_T_block]
                         for k in range(self._mix_n_layers)], axis=1)
                l1 = l1 + self.lambda_1 * tf.reduce_sum(tf.abs(slab), axis=1) / _vp_n_per_t
                l2 = l2 + self.lambda_2 * tf.sqrt(
                    tf.reduce_sum(tf.square(slab), axis=1) / _vp_n_per_t)
            if reg_Vorth and self._mix_per_type:
                # Slab t orth penalty routes to label t.
                orth_t = self._orth_penalty_slab(V_tail, slab_idx=t)
                l2 = l2 + self._lambda_orth * orth_t
            if reg_gate:
                # Per-type gating slab: gates[t, :] lives at
                #   g_tail[:, t·PL : (t+1)·PL]
                g_slab = g_dev[:, t * PL:(t + 1) * PL]
                if lg1 > 0.0:
                    l1 = l1 + self._lambda_gate_1 * tf.reduce_sum(
                        tf.abs(g_slab), axis=1) / n_per_t
                if lg2 > 0.0:
                    l2 = l2 + self._lambda_gate_2 * tf.sqrt(
                        tf.reduce_sum(tf.square(g_slab), axis=1) / n_per_t)
                if lgf > 0.0:
                    floor_slab = g_floor_sq[:, t * PL:(t + 1) * PL]
                    l2 = l2 + self._lambda_gate_floor * tf.reduce_sum(
                        floor_slab, axis=1) / n_per_t
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
        if reg_gate:
            # Global gating reg: aggregate (g − g_init) penalty across all
            # types/blocks. Normalised by total entries n_gates to match
            # the per-type penalty scale.
            if lg1 > 0.0:
                global_l1 = global_l1 + self._lambda_gate_1 * tf.reduce_sum(
                    tf.abs(g_dev), axis=1) / self.n_gates
            if lg2 > 0.0:
                global_l2 = global_l2 + self._lambda_gate_2 * tf.sqrt(
                    tf.reduce_sum(tf.square(g_dev), axis=1) / self.n_gates)
            if lgf > 0.0:
                global_l2 = global_l2 + self._lambda_gate_floor * tf.reduce_sum(
                    g_floor_sq, axis=1) / self.n_gates
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

    @property
    def _cov_mode_str(self) -> str:
        """Lowercase `snes_cov_mode` string. A `@property` (not a cached
        snapshot) so tests that mutate `cfg.snes_cov_mode` after construction
        keep working, while still avoiding the `str(getattr(...)).lower()`
        round-trip that used to be repeated ~14× per generation across
        `ask()` / `update()` / `fit()`.
        """
        return str(self.cfg.snes_cov_mode).lower()

    def compute_utilities(self) -> np.ndarray:
        """Precompute rank-based utility weights for the population.

        Two modes (selected by `cfg.snes_active_utilities`):

        Vanilla Hansen (default, snes_active_utilities=False):
          - Top half: positive log-rank weights summing to 1.
          - Bottom half: weight = -1/λ (uniform negative).
          - Zero-mean by construction.
          - Throws away the actual *ranking* of bad samples — they all get
            the same small negative weight.

        Active (snes_active_utilities=True; Jastrebski & Arnold 2006,
        Hansen 2016 active CMA, Eqs. 49-53):
          - Top half: same positive log-rank weights as vanilla.
          - Bottom half: MIRRORED log-rank weights with NEGATIVE sign, scaled
            by `cfg.snes_active_alpha` (default 1.0 = symmetric magnitude).
            The worst-ranked sample gets the most-negative weight.
          - Re-centred to zero mean (only changes utilities if alpha != 1).
          - Effect: σ also shrinks along directions where BAD samples spread.
            Half the population now carries useful signal (vs zero info under
            Hansen's floor-at-zero), doubling per-gen information density —
            directly addresses the per-coord σ-spread that signals a
            conditioning bottleneck.

        `self._recomb_w` (cached for CMA path computation) is ALWAYS the
        positive Hansen weights regardless of mode — CMA recombination by
        convention uses positive weights only, so μ_eff and downstream
        constants are unchanged.

        Returns:
            utilities : ndarray [pop_size] — weights indexed by rank (0 = best).
        """
        lam = self.pop_size
        ranks = np.arange(lam) + 1
        raw = np.log((lam * 0.5) + 1.0) - np.log(ranks)
        raw_positive = np.maximum(0.0, raw)

        # Recombination weights for CMA paths — always the positive Hansen
        # weights (sum to 1), used by p_c / p_σ updates in rank-1 + CR-FM-NES.
        total_pos = np.sum(raw_positive)
        if total_pos > 0:
            recomb_w = raw_positive / total_pos
        else:
            if self.cfg.debug:
                print("Utility calc failed due to negative total")
            recomb_w = raw_positive
        self._recomb_w = tf.constant(recomb_w.astype(np.float32))
        # Invalidate the active-distance-regime negative cache so a pop_size
        # change (e.g. IPOP restart) triggers a rebuild on next use.
        self._neg_recomb_abs = None

        active = bool(getattr(self.cfg, "snes_active_utilities", False))
        if not active:
            # Vanilla Hansen: positives sum to 1, all entries shifted by -1/λ.
            # Bottom samples all get -1/λ (no ranking info inside the bottom half).
            utilities = recomb_w - 1.0 / lam
        else:
            # Active mode: bottom half gets MIRRORED negative log-rank weights.
            # raw[i] = log(λ/2+1) - log(i+1) — strictly decreasing; negative for i > λ/2-1.
            # We use |raw| on the bottom half, normalise to sum α, then negate.
            alpha = float(getattr(self.cfg, "snes_active_alpha", 1.0))
            neg_abs = -np.minimum(0.0, raw)   # |bottom-half values|; zeros on top
            total_neg = np.sum(neg_abs)
            neg_w = (alpha * neg_abs / total_neg) if total_neg > 0 else neg_abs
            utilities = recomb_w - neg_w
            # Zero-mean enforcement: if alpha=1 and λ even, sum is already
            # exactly 0; for any other alpha or odd λ this re-centres without
            # affecting CMA paths (which read `_recomb_w` directly, not `utilities`).
            utilities = utilities - np.mean(utilities)

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
        """Sample pop_size candidate parameter vectors from the search
        distribution.

        Uses **mirrored (antithetic) sampling** — see `_mirrored_normal`.

        Sampling covariance:
          - default (snes_cov_mode='none'): isotropic
                delta = sigma * s_iso          (samples = μ + delta)
          - 'crfmnes': CR-FM-NES samples δ = σ_scalar · (D ⊙ y) with
                y = z + (√(1+‖v‖²)−1)·v̄(v̄ᵀz). See `_ensure_crfmnes_state`.

        The returned `delta` is the ACTUAL displacement (samples − μ).
        The natural-gradient mean step Σ_p u_p · delta_p is correct for
        ANY sampling covariance (it's the utility-weighted displacement
        sum), so update() uses delta directly for the mean. The per-dim
        sigma update, by contrast, must use ONLY the isotropic component
        `s_iso` (else sigma absorbs the injected subspace variance and
        drifts) — hence we return both.

        Returns:
            samples : [pop_size, dim] float32 — candidate parameter vectors
            aux     : dict with
                "s_iso" : [pop_size, dim] standard-normal isotropic noise
                "delta" : [pop_size, dim] actual displacement (samples − μ)
        """
        s_iso = self._mirrored_normal((self.pop_size, self.dim))
        # `y` is the v-twisted pre-scale shape sample used by CR-FM-NES; it is
        # only populated in the crfmnes branch below and is returned in `aux`
        # solely when crfmnes is active (update() needs it for the Fisher step).
        y = None
        crfmnes_active = (
            self._cov_mode_str == "crfmnes"
            and self._cr_v is not None
        )
        if crfmnes_active:
            # CR-FM-NES sampling: y = z + (√(1+‖v‖²) − 1)·v̄·(v̄ᵀz);
            # δ = σ_scalar · (D ⊙ y). Mirroring is at z-level (`s_iso`); the
            # v-twist runs after, so y-pairs are NOT exact antithetes — only z
            # is. This matches upstream crfmnes/alg.py:107-114.
            self._ensure_crfmnes_state()
            v = self._cr_v
            v_norm2 = tf.reduce_sum(v * v)
            scale = tf.sqrt(1.0 + v_norm2) - 1.0
            # v̄·(v̄ᵀz) = v·(vᵀz)/‖v‖²; guard against v_norm2 ≈ 0 (then scale → 0
            # and proj is multiplied by 0, so the numerator dominates).
            v_norm2_safe = tf.maximum(v_norm2, 1e-30)
            proj = tf.einsum("pd,d->p", s_iso, v)[:, None] * v[None, :] / v_norm2_safe
            y = s_iso + scale * proj
            delta = self._cr_sig * (self._cr_D * y)
        else:
            delta = s_iso * self.sigma
        samples = self.mu + delta
        aux = {"s_iso": s_iso, "delta": delta}
        if y is not None:
            aux["y"] = y
        return samples, aux

    def _ensure_es_mean_state(self) -> None:
        """Lazily allocate Adam moment buffers for the ES mean gradient.
        Only materialised the first time snes_mean_optimizer == "adam" fires.
        """
        if self._es_m is None:
            self._es_m = tf.Variable(tf.zeros([self.dim], dtype=tf.float32),
                                     trainable=False, name="es_mean_m")
            self._es_v = tf.Variable(tf.zeros([self.dim], dtype=tf.float32),
                                     trainable=False, name="es_mean_v")
            self._es_t = tf.Variable(0, dtype=tf.int64, trainable=False,
                                     name="es_mean_t")

    def _ensure_es_sigma_state(self) -> None:
        """Lazily allocate the EMA buffer for the step-size gradient.

        Only materialised the first time snes_sigma_cumulation is on.
        """
        if self._grad_sigma_ema is None:
            self._grad_sigma_ema = tf.Variable(tf.zeros([self.dim], dtype=tf.float32),
                                               trainable=False, name="es_grad_sigma_ema")

    @staticmethod
    def _solve_h_inv(dim: float) -> float:
        """Newton solve of f(a) = ((1+a²)·exp(a²/2))/0.24 − 10 − dim = 0.

        Ported verbatim from crfmnes/alg.py:12-22 — damped Newton (step
        scale 0.5), tolerance |f(a)| ≤ 1e-10, early-exit if a step moves
        less than 1e-16. The root `a = h_inv(dim)` is dim-dependent and
        enters the distance-weighting α(λ_F) coefficient.
        """
        import math as _math
        f       = lambda a: ((1.0 + a * a) * _math.exp(a * a / 2.0) / 0.24) - 10.0 - dim
        f_prime = lambda a: (1.0 / 0.24) * a * _math.exp(a * a / 2.0) * (3.0 + a * a)
        h = 6.0
        while abs(f(h)) > 1e-10:
            last = h
            h = h - 0.5 * (f(h) / f_prime(h))
            if abs(h - last) < 1e-16:
                break
        return float(h)

    def _ensure_crfmnes_state(self) -> None:
        """Lazily allocate CR-FM-NES Variables and cache the dim-dependent
        constants (χ_d, h_inv).

        Only materialised when cov_mode="crfmnes"; vanilla / rank-1 / guided
        leave these None so the search state stays byte-identical.
        """
        if self._cr_v is None:
            d = float(self.dim)
            self._cr_v   = tf.Variable(tf.zeros([self.dim], dtype=tf.float32),
                                       trainable=False, name="cr_v")
            self._cr_D   = tf.Variable(tf.ones([self.dim], dtype=tf.float32),
                                       trainable=False, name="cr_D")
            self._cr_psg = tf.Variable(tf.zeros([self.dim], dtype=tf.float32),
                                       trainable=False, name="cr_psg")
            self._cr_pc  = tf.Variable(tf.zeros([self.dim], dtype=tf.float32),
                                       trainable=False, name="cr_pc")
            self._cr_sig = tf.Variable(float(self.cfg.init_sigma), dtype=tf.float32,
                                       trainable=False, name="cr_sig")
            self._cr_chi = float(np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d)))
            self._cr_h_inv = float(self._solve_h_inv(d))

    def _crfmnes_constants(self, lambda_F: int, ps_norm: float) -> dict:
        """All CR-FM-NES rates ported verbatim from crfmnes/alg.py.

        Pure function of (dim, λ, λ_F, ‖p_σ‖) so it can be unit-tested
        against the upstream reference. λ_F-dependent rates (c_1, η_B,
        α, η_stag, η_conv) all read `lambda_F`; the three-regime η_σ
        branches on `ps_norm` vs χ_d and 0.1·χ_d.

        Reads cached `self._cr_h_inv` and `self._cr_chi` from
        `_ensure_crfmnes_state()`; reads `self._recomb_w` (populated by
        `compute_utilities()` — same recombination weights the rank-1
        update uses).

        Returns dict with keys:
            mu_eff, eta_m, c_sigma, c_c, c1_cma, c_1,
            eta_B, alpha_dist, eta_sigma, h_inv, chi_d.

        Line refs (crfmnes/alg.py):
            mu_eff   line 60       c_sigma  line 61      c_c      line 62
            c1_cma   line 63       c_1      line 78      eta_B    line 79
            eta_m    line 74       alpha    line 70-71   χ_d      line 65
            η_σ regimes: lines 75-77, branching 150-151
        """
        d   = float(self.dim)
        lam = float(self.pop_size)
        lf  = float(lambda_F)

        # Recompute the CMA recombination weights `w_rank_hat / sum(w_rank_hat)`
        # in float64 (matches upstream crfmnes/alg.py:57-60). The cached
        # self._recomb_w stores the same quantity but in float32 — using it
        # here would inject ~1e-7 relative error into mu_eff and every
        # downstream rate, breaking the 1e-10 tolerance against the
        # upstream-generated reference fixture.
        ranks    = np.arange(1, int(lam) + 1, dtype=np.float64)
        w_hat    = np.log(lam * 0.5 + 1.0) - np.log(ranks)
        w_hat    = np.maximum(0.0, w_hat)
        w64      = w_hat / np.sum(w_hat)
        mu_eff   = float(1.0 / np.sum(w64 * w64))

        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        c_c     = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c1_cma  = 2.0 / ((d + 1.3) ** 2 + mu_eff)

        # λ_F-scaled rates (crfmnes/alg.py:78-79):
        c_1   = c1_cma * (d - 5.0) / 6.0 * (lf / lam)
        eta_B = float(np.tanh((min(0.02 * lf, 3.0 * np.log(d)) + 5.0)
                              / (0.23 * d + 25.0)))

        # α(λ_F) (crfmnes/alg.py:70-71). Cached h_inv is dim-dependent.
        h_inv = self._cr_h_inv
        chi_d = self._cr_chi
        alpha_dist = h_inv * min(1.0, np.sqrt(lam / d)) * np.sqrt(lf / lam)

        # Three-regime η_σ (crfmnes/alg.py:75-77 + branching 150-151):
        if ps_norm >= chi_d:
            eta_sigma = 1.0
        elif ps_norm >= 0.1 * chi_d:
            eta_sigma = float(np.tanh((0.024 * lf + 0.7 * d + 20.0) / (d + 12.0)))
        else:
            eta_sigma = 2.0 * float(np.tanh((0.025 * lf + 0.75 * d + 10.0) / (d + 4.0)))

        return {
            "mu_eff":     mu_eff,
            "eta_m":      1.0,
            "c_sigma":    float(c_sigma),
            "c_c":        float(c_c),
            "c1_cma":     float(c1_cma),
            "c_1":        float(c_1),
            "eta_B":      float(eta_B),
            "alpha_dist": float(alpha_dist),
            "eta_sigma":  float(eta_sigma),
            "h_inv":      float(h_inv),
            "chi_d":      float(chi_d),
        }

    def _crfmnes_weights(self, s_iso_sorted: tf.Tensor, lambda_F: int,
                         ps_norm: float) -> tf.Tensor:
        """Return the active length-λ weight vector for a CR-FM-NES generation.

        Implements the binary regime switch on `‖p_σ‖` (crfmnes/alg.py:149):
            weights = w_dist  if  ps_norm >= χ_d
                      w_rank  otherwise

        Where:
          * `w_rank` is the existing zero-mean log-rank utility set
            (`self.utilities`) — same shaping vanilla SNES uses, and what
            upstream stores as `self.w_rank` (alg.py:59).
          * `w_dist` is `w_rank_hat[i] * exp(α(λ_F) * ‖zᵢ‖)`, renormalised
            to sum-1 and zero-shifted by 1/λ (alg.py:144-147). Upstream's
            `w_rank_hat / sum(w_rank_hat)` is exactly our cached
            `self._recomb_w`, so we multiply the exponential factor in
            BEFORE renormalising (an unnormalised global scalar cancels in
            the renorm anyway, but doing it this way matches the upstream
            structure verbatim).

        Args:
            s_iso_sorted : [pop, dim] — globally-fitness-sorted s_iso (i.e.
                upstream's `z` AFTER the line-127 reorder). In TNEP this is
                supplied by `fit()` as `aux["s_iso_global"]`.
            lambda_F     : per-gen feasible count (= pop_size for
                unconstrained problems). Drives α(λ_F).
            ps_norm      : ‖self._cr_psg‖ — the conjugate-path norm AFTER
                this generation's p_σ update.

        Returns:
            tf.Tensor [pop] float32 — the active (zero-mean) weight vector.

        Note on eager Python-`if`: we deliberately use a host-side `if` on
        `ps_norm` (a Python float) rather than `tf.cond`. This keeps the
        weight build readable and matches how `update()` will consume it
        in C4 (where ps_norm is computed eagerly from `tf.norm(...)`).
        If/when this method is wrapped in `tf.function(jit_compile=True)`,
        revisit and replace with `tf.cond`.
        """
        if self._recomb_w is None:
            self.compute_utilities()
        chi_d = float(self._cr_chi)
        active = bool(getattr(self.cfg, "snes_active_utilities", False))
        if ps_norm < chi_d:
            # Stagnation/early regime → log-rank weights.
            # self.utilities already includes active negatives if cfg enables
            # it (negative recombination on v flows through the Fisher (s,t)
            # update naturally — the bottom samples now SHRINK v along their
            # direction, mirroring how active utilities shrink σ).
            return tf.cast(self.utilities, tf.float32)

        # Move regime → distance-weighted recombination.
        K = self._crfmnes_constants(int(lambda_F), float(ps_norm))
        alpha_dist = tf.constant(float(K["alpha_dist"]), dtype=tf.float32)
        # ‖zᵢ‖ per sample (row-norm); upstream computes ‖z[:,i]‖ over the
        # dim-axis which corresponds to axis=1 here.
        z_norms = tf.norm(tf.cast(s_iso_sorted, tf.float32), axis=1)         # [pop]
        # w_rank_hat / sum cached as self._recomb_w (sum-1, NOT zero-shifted).
        # exp-factor multiplied in before renormalisation matches alg.py:144-147.
        exp_factor = tf.exp(alpha_dist * z_norms)
        w_raw = self._recomb_w * exp_factor

        if not active:
            # Vanilla CR-FM-NES: positive distance weights, zero-mean shift.
            w_dist = w_raw / tf.reduce_sum(w_raw) - (1.0 / float(self.pop_size))
            return tf.cast(w_dist, tf.float32)

        # ACTIVE distance regime: also build mirrored negative distance
        # weights from the Hansen-negative raw log-rank values. The bottom
        # samples that took LARGE steps (high ‖z‖) and were BAD get extra
        # exp-amplified negative weight — directly shrinking _cr_v along the
        # worst-case directions (the "negative recombination" half of full
        # active CMA, ported into CR-FM-NES's distance regime).
        if self._neg_recomb_abs is None:
            # Lazy-build the mirrored absolute negative log-rank weights:
            # |min(0, log(λ/2+1) - log(i))|, normalized to sum to 1.
            lam = self.pop_size
            ranks_np = np.arange(lam) + 1
            raw_np = np.log((lam * 0.5) + 1.0) - np.log(ranks_np)
            neg_abs_np = -np.minimum(0.0, raw_np)
            total_neg = float(neg_abs_np.sum())
            if total_neg > 0.0:
                neg_abs_np = neg_abs_np / total_neg
            self._neg_recomb_abs = tf.constant(neg_abs_np.astype(np.float32))
        active_alpha = float(getattr(self.cfg, "snes_active_alpha", 1.0))
        w_neg_raw = self._neg_recomb_abs * exp_factor
        w_neg_sum = tf.reduce_sum(w_neg_raw)
        # Guard: if all neg_recomb_abs are zero (lambda=2), skip negative half.
        if float(w_neg_sum.numpy()) > 0.0:
            w_pos = w_raw / tf.reduce_sum(w_raw)
            w_neg = active_alpha * w_neg_raw / w_neg_sum
            weights = w_pos - w_neg
            # Zero-mean enforcement.
            weights = weights - tf.reduce_mean(weights)
            return tf.cast(weights, tf.float32)
        # Fallback to vanilla form if no negative entries exist.
        w_dist = w_raw / tf.reduce_sum(w_raw) - (1.0 / float(self.pop_size))
        return tf.cast(w_dist, tf.float32)

    def _active_sigma_vec(self) -> tf.Tensor:
        """Return the active per-coordinate sampling scale, length [dim].

        Under `cov_mode == "crfmnes"` the per-dim `self.sigma` is dead — the
        active sampling scale lives in `_cr_sig * _cr_D`. Under any other
        mode it's just `self.sigma`. Used by σ-reporting, plateau-reset,
        and any future inference-time σ read.
        """
        if (self._cov_mode_str == "crfmnes"
                and self._cr_sig is not None and self._cr_D is not None):
            return self._cr_sig * self._cr_D
        return self.sigma

    def update(self, utilities: tf.Tensor, aux: dict[str, tf.Tensor]) -> None:
        """Update mu and sigma using fitness-ranked samples.

        All operations run on GPU via TensorFlow.

        Args:
            utilities : [pop_size] float32 tensor — rank-based weights (best first)
            aux       : dict from ask(), already sorted by fitness, with
                "s_iso" : [pop_size, dim] isotropic standard-normal noise
                          (s_iso[0] = best individual, s_iso[-1] = worst)
                "delta" : [pop_size, dim] actual displacement (samples − μ),
                          same fitness ordering as s_iso

        Mean step is COVARIANCE-AGNOSTIC: it is the utility-weighted sum of
        actual displacements `Σ_p u_p · delta_p`, which is the natural-
        gradient mean step for any sampling covariance (isotropic OR
        guided). When delta = sigma·s_iso (guided off) this is bit-
        identical to the old `mu += sigma·Σ u_p s_iso`.

        Sigma step uses ONLY the isotropic component s_iso (grad_sigma =
        Σ u_p (s_iso² − 1)): if it used the guided displacement, sigma
        would absorb the injected gradient-subspace variance and drift.

        Mutates self.mu and self.sigma tf.Variables in place. Sigma is
        clamped to a small floor (cfg.sigma_floor, default 1e-5) after
        the multiplicative update so a near-collapse of the search
        distribution can't silently kill exploration on long runs.
        """
        s_iso = aux["s_iso"]
        delta = aux["delta"]
        # Kept for the mean-Adam branch, which preconditions the ES mean
        # gradient (the standard-normal natural gradient, not displacements).
        grad_mu = tf.einsum('p,pd->d', utilities, s_iso)

        # Detect CR-FM-NES mode early. CR-FM-NES owns the mean update (uses
        # regime-switched weights, not static log-rank utilities), so the
        # parent mean dispatch is skipped under crfmnes — the crfmnes branch
        # below advances μ with the active weights.
        crfmnes_mode = (
            self._cov_mode_str == "crfmnes"
        )

        # --- mean update (Feature A: optional Adam preconditioning) ---
        if crfmnes_mode:
            # μ-step deferred to the crfmnes branch (uses regime-switched weights).
            pass
        elif str(getattr(self.cfg, "snes_mean_optimizer", "vanilla")).lower() == "adam":
            self._ensure_es_mean_state()
            b1 = float(self.cfg.snes_mean_beta1); b2 = float(self.cfg.snes_mean_beta2)
            eps = float(self.cfg.snes_mean_epsilon)
            lr = self.cfg.snes_mean_lr
            # None -> 1e-2. Adam normalises the per-dim step magnitude, so the
            # learning rate is decoupled from init_sigma (tying it to init_sigma
            # overshoots ~100x in high dim). ~1e-2 matches the per-dim vanilla
            # step scale empirically; tune per problem.
            lr = float(lr) if lr is not None else 1e-2
            self._es_t.assign_add(1)
            t = tf.cast(self._es_t, tf.float32)
            self._es_m.assign(b1 * self._es_m + (1.0 - b1) * grad_mu)
            self._es_v.assign(b2 * self._es_v + (1.0 - b2) * tf.square(grad_mu))
            m_hat = self._es_m / (1.0 - tf.pow(b1, t))
            v_hat = self._es_v / (1.0 - tf.pow(b2, t))
            self.mu.assign_add(lr * m_hat / (tf.sqrt(v_hat) + eps))
        elif self._cov_mode_str != "none":
            # Rank-1 CMA inflated delta beyond sigma·s_iso, so the
            # covariance-agnostic mean step (utility-weighted sum of actual
            # displacements) is required for the correct natural gradient.
            # (With the rank-1 correction delta != sigma·s_iso, so the
            # vanilla sigma·grad_mu form would be wrong.)
            self.mu.assign_add(tf.einsum('p,pd->d', utilities, delta))
        else:
            # Vanilla path: keep the exact original reduction order
            # (sigma · Σ u_p s_iso) so behaviour is BIT-identical.
            # Algebraically equal to the displacement form above
            # (delta == sigma·s_iso here).
            self.mu.assign_add(self.sigma * grad_mu)

        # --- rank-1 CMA covariance (evolution path + amplitude) ---
        # --- CR-FM-NES (v, D) Fisher update + paths + σ_scalar step ---
        # Ported verbatim from crfmnes/alg.py:138-190.  When active this
        # branch owns the mean, p_σ, p_c, (v, D), det(A) renorm, and σ_scalar
        # updates — the per-dim σ update below is bypassed via early return.
        # f64 internally: the Fisher loop accumulates rounding faster than the
        # constants, and the upstream reference is f64.
        if crfmnes_mode:
            self._ensure_crfmnes_state()
            if "y_global" not in aux:
                raise RuntimeError(
                    "fit() must supply aux['y_global'] for cov_mode='crfmnes'")
            # λ_F = per-gen feasible count (NOT a persistent success counter).
            # crfmnes/alg.py:138. Falls back to pop_size for unit tests that
            # don't supply a "fitness" tensor.
            fitness_t = aux.get("fitness")
            if fitness_t is not None:
                lf = int(tf.reduce_sum(
                    tf.cast(tf.math.is_finite(fitness_t), tf.int32)).numpy())
            else:
                lf = int(self.pop_size)

            # Promote state to f64 for the Fisher loop. We need mu_eff and c_sigma
            # to update psg before evaluating the regime — so first build a
            # "rate-only" K probe with ps_norm=0 (mu_eff, c_sigma, c_c are
            # ps_norm-independent), update psg, then build the real K against
            # POST-update ps_norm (upstream alg.py:141→149 ordering).
            chi_d = float(self._cr_chi)
            v64    = tf.cast(self._cr_v, tf.float64)
            D64    = tf.cast(self._cr_D, tf.float64)
            psg64  = tf.cast(self._cr_psg, tf.float64)
            pc64   = tf.cast(self._cr_pc, tf.float64)
            sig64  = tf.cast(self._cr_sig, tf.float64)
            s_iso_g = aux["s_iso_global"]
            s_iso_g64 = tf.cast(s_iso_g, tf.float64)
            y_g64     = tf.cast(aux["y_global"], tf.float64)
            delta_g64 = tf.cast(aux.get("s_eff_global"), tf.float64) * sig64  # delta = sig * (D*y); s_eff_global = delta/sig

            # ---- 1. p_σ (z-coords, log-rank weighted) — crfmnes/alg.py:141 ----
            # All rates here are ps_norm-independent so a probe with ps_norm=0 works.
            K_probe = self._crfmnes_constants(lf, 0.0)
            c_sigma = tf.constant(K_probe["c_sigma"], dtype=tf.float64)
            mu_eff64 = tf.constant(K_probe["mu_eff"], dtype=tf.float64)
            # CR-FM-NES p_σ uses STANDARD Hansen log-rank utilities
            # (crfmnes/alg.py:141 — upstream w_rank, NOT the active variant).
            # Construct from _recomb_w (positive Hansen, sum=1) shifted by
            # -1/λ. Doing this independently of `self.utilities` insulates
            # CR-FM-NES from `cfg.snes_active_utilities=True`, which would
            # otherwise leak into the p_σ path and break the spec.
            recomb_w64 = tf.cast(self._recomb_w, tf.float64)
            w_rank = recomb_w64 - tf.constant(1.0 / float(self.pop_size),
                                              dtype=tf.float64)
            # z @ w_rank in upstream notation == s_iso_g.T @ w_rank as (dim,) vec
            #   z is (dim, lamb); w_rank is (lamb,). Our s_iso_g is (lamb, dim).
            z_at_w = tf.einsum("p,pd->d", w_rank, s_iso_g64)
            new_psg = (1.0 - c_sigma) * psg64 + tf.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff64) * z_at_w

            # ---- 2. ps_norm (POST-update) drives the regime switch ----
            ps_norm = float(tf.norm(new_psg).numpy())
            K = self._crfmnes_constants(lf, ps_norm)
            # Active weights (regime-switched) — uses POST-update ps_norm.
            weights_f32 = self._crfmnes_weights(s_iso_g, lf, ps_norm)
            weights = tf.cast(weights_f32, tf.float64)

            # ---- 3. mean step + p_c (uses regime-switched weights) ----
            # wxm = (x - m) @ weights — in our notation: einsum('p,pd->d', weights, delta)
            wxm = tf.einsum("p,pd->d", weights, delta_g64)
            c_c = tf.constant(K["c_c"], dtype=tf.float64)
            new_pc = (1.0 - c_c) * pc64 + tf.sqrt(c_c * (2.0 - c_c) * mu_eff64) * wxm / sig64
            eta_m = tf.constant(K["eta_m"], dtype=tf.float64)
            new_mu = tf.cast(self.mu, tf.float64) + eta_m * wxm

            # ---- 4. Fisher (s, t) closed form — crfmnes/alg.py:158-179 ----
            # Build everything column-major to match upstream's (dim, lamb+1)
            # exY shape. Our tensors live as (lamb, dim) so transpose y_g to
            # (dim, lamb), then concat pc/D as the (lamb+1)-th column.
            y_col   = tf.transpose(y_g64)                                       # (dim, lamb)
            pc_over_D = (new_pc / D64)[:, None]                                 # (dim, 1)
            exY     = tf.concat([y_col, pc_over_D], axis=1)                     # (dim, lamb+1)

            # OLD normv, captured BEFORE the v update (crfmnes/alg.py:183).
            normv2 = tf.reduce_sum(v64 * v64)
            normv  = tf.sqrt(normv2)
            normv4 = normv2 * normv2
            normv_safe = tf.maximum(normv, tf.constant(1e-300, dtype=tf.float64))
            vbar = v64 / normv_safe                                              # (dim,)
            vbar_col = vbar[:, None]                                             # (dim, 1)
            gammav  = 1.0 + normv2
            vbarbar = vbar * vbar                                                # (dim,)
            vbarbar_col = vbarbar[:, None]

            yy        = exY * exY                                                # (dim, lamb+1)
            ip_yvbar  = tf.matmul(vbar_col, exY, transpose_a=True)               # (1, lamb+1)
            yvbar     = exY * vbar_col                                           # (dim, lamb+1)
            max_vbarbar = tf.reduce_max(vbarbar)
            max_vbarbar_safe = tf.maximum(max_vbarbar, tf.constant(1e-300, dtype=tf.float64))
            alphavd_inner = tf.sqrt(normv4 + (2.0 * gammav - tf.sqrt(gammav)) / max_vbarbar_safe) / (2.0 + normv2)
            alphavd = tf.minimum(tf.constant(1.0, dtype=tf.float64), alphavd_inner)

            t_mat = exY * ip_yvbar - vbar_col * (ip_yvbar * ip_yvbar + gammav) / 2.0
            b     = -(1.0 - alphavd * alphavd) * normv4 / gammav + 2.0 * alphavd * alphavd
            H     = 2.0 * tf.ones_like(vbar_col) - (b + 2.0 * alphavd * alphavd) * vbarbar_col
            invH  = 1.0 / H
            s_step1 = yy - normv2 / gammav * (yvbar * ip_yvbar) - tf.ones_like(exY)
            ip_vbart = tf.matmul(vbar_col, t_mat, transpose_a=True)              # (1, lamb+1)
            s_step2 = s_step1 - alphavd / gammav * (
                (2.0 + normv2) * (t_mat * vbar_col)
                - normv2 * vbarbar_col @ ip_vbart
            )
            invHvbarbar = invH * vbarbar_col                                     # (dim, 1)
            ip_s_step2invHvbarbar = tf.matmul(invHvbarbar, s_step2, transpose_a=True)  # (1, lamb+1)
            denom_scalar = 1.0 + b * tf.matmul(vbarbar_col, invHvbarbar, transpose_a=True)  # (1,1)
            s_mat = (s_step2 * invH) - b / denom_scalar * (invHvbarbar @ ip_s_step2invHvbarbar)
            ip_svbarbar = tf.matmul(vbarbar_col, s_mat, transpose_a=True)        # (1, lamb+1)
            t_mat = t_mat - alphavd * (
                (2.0 + normv2) * (s_mat * vbar_col)
                - vbar_col @ ip_svbarbar
            )

            # ---- 5. exw = concat([eta_B * weights, c_1]) ----
            eta_B = tf.constant(K["eta_B"], dtype=tf.float64)
            c_1   = tf.constant(K["c_1"],   dtype=tf.float64)
            exw = tf.concat([eta_B * weights, [c_1]], axis=0)                    # (lamb+1,)

            # ---- 6. v, D update — uses OLD normv ----
            normv_old_safe = tf.maximum(normv, tf.constant(1e-300, dtype=tf.float64))
            new_v = v64 + tf.linalg.matvec(t_mat, exw) / normv_old_safe          # (dim,)
            new_D = D64 + tf.linalg.matvec(s_mat, exw) * D64                     # (dim,)

            # ---- 7. det(A) renorm POST-update, with NEW v — crfmnes/alg.py:186 ----
            new_v_norm2 = tf.reduce_sum(new_v * new_v)
            d_f = tf.cast(self.dim, tf.float64)
            log_root = tf.reduce_sum(tf.math.log(new_D)) / d_f \
                + tf.math.log(1.0 + new_v_norm2) / (2.0 * d_f)
            nthrootdetA = tf.exp(log_root)
            new_D = new_D / nthrootdetA

            # ---- 8. σ update — G_σ = sum((z²-1)·weights) / d ----
            G_s = tf.reduce_sum((s_iso_g64 * s_iso_g64 - 1.0) * weights[:, None]) / d_f
            eta_sigma_cr = tf.constant(K["eta_sigma"], dtype=tf.float64)
            new_sig = sig64 * tf.exp(eta_sigma_cr / 2.0 * G_s)
            floor = getattr(self.cfg, "sigma_floor", 1e-5)
            if floor is not None and floor > 0.0:
                new_sig = tf.maximum(new_sig, tf.constant(float(floor), dtype=tf.float64))

            # ---- Cast back to f32 and assign ----
            self._cr_psg.assign(tf.cast(new_psg, tf.float32))
            self._cr_pc.assign(tf.cast(new_pc, tf.float32))
            self.mu.assign(tf.cast(new_mu, tf.float32))
            self._cr_v.assign(tf.cast(new_v, tf.float32))
            self._cr_D.assign(tf.cast(new_D, tf.float32))
            self._cr_sig.assign(tf.cast(new_sig, tf.float32))
            return  # SKIP the per-dim σ update below — CR-FM-NES owns σ.

        # --- sigma update (Feature B: optional cumulation) ---
        # grad_sigma = Σ u_i (s_i² − 1) is the NES natural gradient on log-sigma.
        # It is self-equilibrating (too-small sigma → best samples are the
        # larger-step ones → grad_sigma>0 → grow; too-large → shrink), which is
        # exactly why the vanilla `sigma *= exp(eta·grad_sigma)` update is
        # stable. Cumulation low-pass-filters THIS signal (an EMA) and feeds it
        # through the SAME exp(eta·.) update, so it inherits vanilla's
        # equilibrium and bounded per-gen growth — it adds temporal smoothing
        # without a runaway.
        #
        # (The original separable-CSA law exp(rate·(p²−1)) drove sigma off the
        # MEAN-gradient path grad_mu, which accumulates monotonically downhill
        # with NO negative feedback → sigma compounded to float overflow. Fixed
        # by switching the driving signal to grad_sigma and reusing eta.)
        grad_sigma = tf.einsum('p,pd->d', utilities, s_iso ** 2 - 1.0)
        if bool(getattr(self.cfg, "snes_sigma_cumulation", False)):
            self._ensure_es_sigma_state()
            c = self.cfg.snes_cumulation_c
            c = float(c) if c is not None else 0.2   # EMA decay (smooth ~1/c gens)
            g_ema = (1.0 - c) * self._grad_sigma_ema + c * grad_sigma
            self._grad_sigma_ema.assign(g_ema)
            # Defense-in-depth: clamp the per-gen log-sigma step so no step-size
            # law can blow sigma up in a single generation (≤ e¹ ≈ 2.7× / gen).
            log_step = tf.clip_by_value(self.eta_sigma * g_ema, -1.0, 1.0)
            new_sigma = self.sigma * tf.exp(log_step)
        else:
            new_sigma = self.sigma * tf.exp(self.eta_sigma * grad_sigma)   # canonical vanilla
        floor = getattr(self.cfg, "sigma_floor", 1e-5)
        if floor is not None and floor > 0.0:
            new_sigma = tf.maximum(new_sigma, float(floor))
        self.sigma.assign(new_sigma)

        # --- MSR global σ rescale (Ait ElHara, Auger, Hansen, GECCO 2013) ---
        # Paper-faithful Algorithm 1, applied AFTER the per-coord σ update.
        # Silently no-ops:
        #   - cov_mode="crfmnes" (CR-FM-NES owns σ; early-returned above).
        #   - snes_msr_enabled=False (default).
        # Fitness signal: prefer self._last_rmse_per_cand (always pure RMSE,
        # unaffected by regularisation or per-type fitness composition) over
        # aux["fitness"] which may include L1/L2 penalties.
        if bool(getattr(self.cfg, "snes_msr_enabled", False)):
            fitness_msr_src = getattr(self, "_last_rmse_per_cand", None)
            if fitness_msr_src is None:
                if "fitness" not in aux:
                    raise KeyError(
                        "MSR requires either self._last_rmse_per_cand "
                        "(set by evaluate_population) or aux['fitness'].")
                fitness_msr_src = aux["fitness"]
            fitness_sorted = tf.sort(tf.cast(fitness_msr_src, tf.float32))
            idx_lo = (self.pop_size - 1) // 2
            idx_hi = self.pop_size // 2
            f_med_curr = float(tf.reduce_mean(
                tf.gather(fitness_sorted, [idx_lo, idx_hi])).numpy())
            if self._msr_initialised:
                K_succ = float(tf.reduce_sum(
                    tf.cast(fitness_sorted < self._msr_f_prev, tf.float32)).numpy())
                # Paper's z: centered at 0 under stationarity, range ≈ [−1, 1].
                # E[K_succ] under H0 (i.i.d. populations) = (λ+1)/2 — so the
                # (λ+1)/2 offset removes the Beta-median asymmetry bias.
                lam = float(self.pop_size)
                z = (2.0 / lam) * (K_succ - (lam + 1.0) / 2.0)
                # EMA smoothing (paper Algorithm 1 line 3) — low-pass filter
                # on z. Without this the controller chases noise at full
                # variance every gen.
                c_s = self._msr_c_sigma
                self._msr_ps = (1.0 - c_s) * self._msr_ps + c_s * z
                # Damped exponential update. d_σ ≈ 2 − 2/n in the paper;
                # max(1, ...) guards d=1 corner.
                d_s = max(1.0, 2.0 - 2.0 / float(self.dim))
                log_step = float(np.clip(
                    self._msr_ps / d_s, -self._msr_clip, self._msr_clip))
                self.sigma.assign(self.sigma * float(np.exp(log_step)))
                floor_msr = getattr(self.cfg, "sigma_floor", 1e-5)
                if floor_msr is not None and floor_msr > 0.0:
                    self.sigma.assign(tf.maximum(self.sigma, float(floor_msr)))
            self._msr_f_prev = f_med_curr
            self._msr_initialised = True

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
            # `best_sigma` is legitimately absent under cov_mode="crfmnes"
            # (snapshot skipped — see best_sigma lifecycle under crfmnes).
            # Fall back to self.sigma; the end-of-run restore is also gated on
            # crfmnes so the fallback value is never actually written back to a
            # dead variable when crfmnes is on.
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
            # Restore CR-FM-NES learned state when present. Guarded so
            # pure-SNES checkpoints (no cr_v) resume exactly as before.
            # Dim-mismatch drops the state with a warning (arch changed).
            if resume_state.get("cr_v") is not None:
                cr_v = np.asarray(resume_state["cr_v"], dtype=np.float32)
                if cr_v.shape[0] == self.dim:
                    self._ensure_crfmnes_state()
                    self._cr_v.assign(cr_v)
                    self._cr_D.assign(
                        np.asarray(resume_state["cr_D"], dtype=np.float32))
                    self._cr_psg.assign(
                        np.asarray(resume_state["cr_psg"], dtype=np.float32))
                    self._cr_pc.assign(
                        np.asarray(resume_state["cr_pc"], dtype=np.float32))
                    self._cr_sig.assign(float(resume_state["cr_sig"]))
                else:
                    print(f"  WARNING: checkpoint cr_v dim "
                          f"{cr_v.shape[0]} != model dim {self.dim}; "
                          f"dropping CR-FM-NES state on resume.")
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
            # Skip the σ snapshot under cov_mode="crfmnes" — self.sigma is
            # frozen at init_sigma, so snapshotting it is meaningless. The
            # active scale (_cr_sig·_cr_D) has no "best" snapshot in this
            # design: best_μ alone reproduces the best-val model.
            if self._cov_mode_str != "crfmnes":
                best_sigma = tf.identity(self.sigma)
            else:
                best_sigma = tf.identity(self.sigma)  # carries the (dead) init value; never restored under crfmnes
            gens_without_improvement = 0
            start_gen = 0
            train_start = time.perf_counter()
        # Plateau-driven sigma resets (IPOP-style restart, simplified):
        # tracks how many resets have already been fired so we can cap
        # via cfg.max_sigma_resets. Reset counter is per-run (not
        # restored from checkpoint — a fresh attempt to escape any
        # plateau seen so far is fine on resume).
        n_sigma_resets = 0

        # Eagerly materialise CR-FM-NES state when active so ask() takes
        # the crfmnes branch from gen 0 (and update() sees a valid `y` in
        # aux). Without this, the lazy `self._cr_v is not None` check in
        # ask() short-circuits to vanilla sampling on the first gen, and
        # the fit() y_global gather hits a KeyError. The plateau-reset
        # test materialised state itself; this makes a fresh fit() work.
        if self._cov_mode_str == "crfmnes":
            self._ensure_crfmnes_state()

        # One-time warning: CR-FM-NES co-features. Each of mean-Adam and
        # sigma-cumulation is an UNTESTED combination with cov_mode="crfmnes"
        # — CR-FM-NES owns its own σ_scalar update and Fisher (v, D) state,
        # and stacking another mean/sigma adapter on top of it is not
        # validated by the paper or by any test in this repo.
        if self._cov_mode_str == "crfmnes":
            untested = []
            if str(getattr(cfg, "snes_mean_optimizer", "vanilla")).lower() == "adam":
                untested.append("mean-Adam")
            if bool(getattr(cfg, "snes_sigma_cumulation", False)):
                untested.append("sigma-cumulation")
            if untested:
                print(f"  WARNING: snes_cov_mode='crfmnes' + {', '.join(untested)} "
                      "is an UNTESTED combination (CR-FM-NES owns σ_scalar and "
                      "the v/D Fisher update; stacking another mean/sigma "
                      "adapter on top of it is not validated).")

        gen_l1, gen_l2, gen_lorth = 0.0, 0.0, 0.0
        val_fitness = float('inf')
        # True RMSE of the mean μ on train data — the SAME estimator as
        # val_fitness (validate at μ), so train vs val is an apples-to-apples
        # comparison for ANY loss_type. (The progress bar previously showed
        # avg_fitness here, which is the mean population *fitness* — equal to
        # RMSE only for loss_type="mse"; under huber/mae it is the loss value
        # itself, on a wildly different scale, making "train RMSE" look tiny.)
        train_rmse = float('inf')
        sigma_min = sigma_max = sigma_mean = sigma_median = float(cfg.init_sigma)
        # Last finite RRMSE values, carried into Adam gens (which don't compute
        # a population RRMSE) so the history series stays finite for plotting.
        last_best_rrmse = last_avg_rrmse = 0.0

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
            # drives SNES ranking (depends on loss_type); the rmse/rrmse
            # entries are ALWAYS computed from squared error so they're
            # comparable across loss-function ablations.
            rmse_pc = self._last_rmse_per_cand
            rrmse_pc = self._last_rrmse_per_cand
            # σ stats are computed on the GPU and folded into the same
            # batched pull as the fitness/RMSE metrics — one device sync
            # per gen, not the previous every-100-gen sampling.
            # Median uses tf.sort which is O(d log d); at d ≈ 50k this
            # is microseconds on GPU.
            sigma_active = self._active_sigma_vec()  # [dim]
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

            # Rank and update (GPU).
            if True:
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
                # The CMA evolution path is a CROSS-COORDINATE object, so it
                # must be built from a single GLOBAL fitness ranking of the
                # realized normalized step. Gated on cov_mode != "none" so
                # the baseline SNES run pays nothing for it — was ~5-10 ms
                # of redundant argsort + gather per gen otherwise.
                aux2 = {"s_iso": s_iso_sorted, "delta": delta_sorted,
                        "fitness": fitness}
                cov_mode_str = self._cov_mode_str
                if cov_mode_str != "none":
                    # In the non-per-type case `global_ranks == ranks`, so
                    # the sorted tensors above already match global rank
                    # order — no extra gather needed.
                    if cfg.per_type_regularization:
                        global_ranks = tf.argsort(fitness)
                        delta_global = tf.gather(aux["delta"], global_ranks)
                        s_iso_global = tf.gather(aux["s_iso"], global_ranks)
                    else:
                        global_ranks = ranks
                        delta_global = delta_sorted
                        s_iso_global = s_iso_sorted
                    active_sigma = (self._cr_sig
                                    if cov_mode_str == "crfmnes"
                                    and self._cr_sig is not None
                                    else self.sigma)
                    aux2["s_eff_global"] = delta_global / active_sigma
                    aux2["s_iso_global"] = s_iso_global
                    if cov_mode_str == "crfmnes":
                        aux2["y_global"] = tf.gather(aux["y"], global_ranks)
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
                # two are directly comparable regardless of loss_type. Costs
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
                history["train_loss"].append(avg_fitness)   # optimised objective (loss_type)
                history.setdefault("train_rmse", []).append(train_rmse)  # at-μ RMSE, comparable to val
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
                    f"train RMSE: {train_rmse:.6f}  "
                    f"val RMSE: {val_fitness:.6f}  "
                    f"best val RMSE: {best_val_loss:.6f}  "
                    f"elapsed: {elapsed_str}  ETA: {eta_str}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
                line += f"  train obj(mse): {avg_fitness:.6f}"
                line += f"  best_RMSE: {best_rmse:.6f}  best_RRMSE: {best_rrmse:.6f}"
                if self._mix_reg_mode == "orthogonal":
                    line += f"  L_orth: {gen_lorth:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # Early stopping (only update on val generations)
            if _do_val:
                # global best drives best_mu / early stop
                if val_fitness < best_val_loss:
                    best_val_loss = val_fitness
                    best_mu = tf.identity(self.mu)
                    # Skip σ snapshot under crfmnes — see best_sigma lifecycle.
                    if self._cov_mode_str != "crfmnes":
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
                cov_mode_now = self._cov_mode_str
                if cov_mode_now == "crfmnes" and self._cr_sig is not None:
                    # CR-FM-NES restart: σ_scalar broadens; p_σ and p_c
                    # restart; (v, D) PRESERVED — the learned shape is
                    # still locally informative.
                    self._cr_sig.assign(self._cr_sig * factor)
                    self._cr_psg.assign(tf.zeros([self.dim], dtype=tf.float32))
                    self._cr_pc.assign(tf.zeros([self.dim], dtype=tf.float32))
                    mode_str = (f"crfmnes restart: _cr_sig·={factor:.2f}, "
                                "paths zeroed (v/D preserved)")
                elif to_init:
                    self.sigma.assign(
                        tf.fill([self.dim], float(cfg.init_sigma) * factor))
                    mode_str = f"σ ← init·{factor:.2f}"
                else:
                    self.sigma.assign(self.sigma * factor)
                    mode_str = f"σ ← σ·{factor:.2f} (preserves per-dim scale)"
                # Reset MSR state — the broadened σ samples a different
                # neighbourhood than _msr_f_prev was computed from.
                self._msr_f_prev = None
                self._msr_ps = 0.0
                self._msr_initialised = False
                restore_mu = bool(getattr(cfg, "plateau_restore_best_mu", False))
                if restore_mu:
                    self.mu.assign(best_mu)
                # IPOP / BIPOP: resize population alongside the σ reset.
                # The Auger & Hansen 2005 IPOP-aCMA-ES recipe is "broaden σ +
                # double λ"; BIPOP additionally alternates with a small-pop
                # regime for multimodal landscape coverage (Hansen 2009).
                old_pop, new_pop, restart_label = self._perform_pop_resize_restart()
                pop_msg = ""
                if old_pop != new_pop:
                    pop_msg = (f", {restart_label}: pop {old_pop} → {new_pop}"
                               f" (restart #{self._restart_count})")
                gens_without_improvement = 0
                n_sigma_resets += 1
                # Read back current σ stats for the log line so the user can
                # see what actually happened. Use `_active_sigma_vec()` so the
                # CR-FM-NES restart reports `_cr_sig*_cr_D`, not the dead
                # `self.sigma`.
                s_now = np.asarray(self._active_sigma_vec().numpy()).reshape(-1)
                s_min = float(np.min(s_now))
                s_mean = float(np.mean(s_now))
                s_max = float(np.max(s_now))
                sys.stdout.write(
                    f"\n  plateau detected at gen {gen + 1}: {mode_str}"
                    f" (σ now min/mean/max = {s_min:.4f}/{s_mean:.4f}/{s_max:.4f})"
                    f" — reset #{n_sigma_resets}"
                    + (f"/{max_resets}" if max_resets is not None else "")
                    + (", μ restored to best" if restore_mu else "")
                    + pop_msg
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
                        if self._cov_mode_str != "crfmnes":
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
                # Omit best_sigma under cov_mode="crfmnes" — see best_sigma
                # lifecycle. Old checkpoints with a stray best_sigma resume
                # gracefully via the `.get("best_sigma")` fallback above.
                if self._cov_mode_str != "crfmnes":
                    ckpt_state["best_sigma"] = best_sigma
                # CR-FM-NES learned state (guarded — absent for pure-SNES /
                # cov_mode="none" runs, so those checkpoints stay byte-identical).
                # No `cr_lf` — C0 found λ_F is per-gen feasible count, not
                # persistent state.
                if self._cr_v is not None:
                    ckpt_state["cr_v"]   = self._cr_v
                    ckpt_state["cr_D"]   = self._cr_D
                    ckpt_state["cr_psg"] = self._cr_psg
                    ckpt_state["cr_pc"]  = self._cr_pc
                    ckpt_state["cr_sig"] = float(self._cr_sig.numpy())
                save_checkpoint(ckpt_path, cfg, ckpt_state, history, gen)
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
        # Under cov_mode="crfmnes" self.sigma is dead (frozen at init_sigma);
        # there is no "best_cr_sig" snapshot in this design — the active scale
        # state lives in _cr_sig/_cr_D and is already at its best value via
        # the path machinery.
        if self._cov_mode_str != "crfmnes":
            self.sigma.assign(best_sigma)
        _set_model_params(self.model, *best_val_params)

        return history, final_model, best_val_model

    def _reg_scalar_tf(self, mu: tf.Tensor) -> tf.Tensor:
        """Differentiable L1+L2 regularisation scalar matching SNES exactly.

        Reproduces, as a single differentiable scalar, the SAME L1+L2 penalty
        that SNES ranks/validates on — i.e. the (l1 + l2) total returned by
        ``compute_regularization`` (eager, lines ~322-424) and added to fitness
        by ``compute_regularization_tf`` (lines ~2021-2058). Previously this
        returned only the GLOBAL typed term, so for ``num_types > 1`` (the CHO
        dipole use-case) Adam descended a strictly smaller reg than SNES ranked.

        Formula reproduced (T = num_types, n_per_type = Q*H + H + H):
          T > 1:  reg = (Σ_t λ1·Σ|type_params_t|/n_per_type)/T            (per-type L1)
                      + λ1·Σ|typed|/n_typed_total                          (global  L1)
                      + (Σ_t λ2·sqrt(Σ type_params_t²/n_per_type))/T       (per-type L2)
                      + λ2·sqrt(Σ typed²/n_typed_total)                    (global  L2)
          T == 1: reg = λ1·Σ|ann|/ann_n + λ2·sqrt(Σ ann²/ann_n)
          (+ V_pair shrinkage tail when reg_Vpair, mirroring compute_regularization)

        ``typed`` excludes b1; for target_mode==2 it includes both ANNs' typed
        params (n_typed doubled), exactly as compute_regularization does.

        DELIBERATELY OMITTED: the orthogonal-mixing penalty (l_orth, the third
        element of compute_regularization's return). It is reported/optimised by
        SNES as a SEPARATE signal driving an independent dynamic-λ schedule, and
        replicating ‖UᵀU - I‖² differentiably for Adam here is out of scope. The
        TNEP CHO dipole fixture has descriptor_mixing=False, so n_U_pair == 0 and
        l_orth == 0 anyway (and both V_pair paths below are inert).

        Stays fully differentiable w.r.t. mu (pure TF ops, no .numpy(), no
        python branching on tensor values).

        Returns:
            reg : scalar tf.Tensor — L1 + L2 regularisation penalty.
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.cfg.num_neurons
        n_per_type = self._n_per_type  # W0_t + b0_t (+ W0_2_t + b0_2_t) + W1_t

        reg_Vpair = (self.n_U_pair > 0 and self._mix_reg_mode == "shrinkage")
        reg_Vorth = (self.n_U_pair > 0 and self._mix_reg_mode == "orthogonal")

        if T > 1:
            # Per-type term: λ1/2 over each type's params, averaged across T.
            total_l1 = tf.constant(0.0, tf.float32)
            total_l2 = tf.constant(0.0, tf.float32)
            for t in range(T):
                type_params = self._extract_type_params(mu, t)
                total_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(type_params)) / n_per_type
                total_l2 += self.lambda_2 * tf.sqrt(
                    tf.reduce_sum(tf.square(type_params)) / n_per_type)

            # Global term over typed params only (excludes b1).
            typed = mu[:self.n_typed]
            n_typed_total = self.n_typed
            if self.cfg.target_mode == 2:
                typed = tf.concat(
                    [typed, mu[self.n_primary:self.n_primary + self.n_typed]], axis=0)
                n_typed_total = 2 * self.n_typed
            l1 = total_l1 / T + self.lambda_1 * tf.reduce_sum(tf.abs(typed)) / n_typed_total
            l2 = total_l2 / T + self.lambda_2 * tf.sqrt(
                tf.reduce_sum(tf.square(typed)) / n_typed_total)
        else:
            # Single-type: reg over ANN params only when V_pair is handled
            # separately (shrinkage / orthogonal / Cayley), else all of mu.
            v_handled = reg_Vpair or reg_Vorth or self._mix_cayley
            ann = mu[:self.n_anns_total] if v_handled else mu
            ann_n = float(self.n_anns_total if v_handled else self.dim)
            l1 = self.lambda_1 * tf.reduce_sum(tf.abs(ann)) / ann_n
            l2 = self.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(ann)) / ann_n)

        # V_pair shrinkage tail (mode == "shrinkage"), mirroring
        # compute_regularization. Differentiable. Inert when n_U_pair == 0.
        if reg_Vpair:
            tail = mu[self.n_anns_total:]
            if self._mix_per_type and T > 1:
                per_T = self.n_U_pair // T
                vp_l1 = tf.constant(0.0, tf.float32)
                vp_l2 = tf.constant(0.0, tf.float32)
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

        return l1 + l2

    def _compute_restart_pop_size(self) -> int:
        """Compute the new pop_size for an IPOP/BIPOP restart per cfg.
        Returns the unchanged value when strategy is "none".

        Strategy:
          "ipop"  : pop_size <- min(restart_max_pop, pop_size * restart_pop_factor)
          "bipop" : alternate large (IPOP-like) and small
                    (restart_bipop_small_factor * _initial_pop_size).
        """
        strategy = str(getattr(self.cfg, "restart_strategy", "none")).lower()
        if strategy == "none":
            return int(self.pop_size)
        factor = float(getattr(self.cfg, "restart_pop_factor", 2.0))
        max_pop = getattr(self.cfg, "restart_max_pop", None)
        if strategy == "ipop":
            new_pop = int(round(self.pop_size * factor))
        elif strategy == "bipop":
            if self._bipop_use_large:
                new_pop = int(round(self._initial_pop_size
                                    * (factor ** (self._restart_count // 2 + 1))))
            else:
                small_factor = float(getattr(self.cfg, "restart_bipop_small_factor", 0.25))
                new_pop = max(int(round(self._initial_pop_size * small_factor)), 4)
        else:
            raise ValueError(f"Unknown restart_strategy {strategy!r}")
        if max_pop is not None:
            new_pop = min(new_pop, int(max_pop))
        min_pop_cfg = getattr(self.cfg, "restart_min_pop", None)
        # Effective floor: max(restart_min_pop or 2, absolute floor 2 for the
        # mirrored-sampling antithetic pair).
        floor = max(int(min_pop_cfg) if min_pop_cfg is not None else 2, 2)
        return max(new_pop, floor)

    def _perform_pop_resize_restart(self) -> tuple[int, int, str]:
        """Resize pop_size per the IPOP/BIPOP strategy and rebuild dependent
        state (utilities, _recomb_w, _mu_eff, _neg_recomb_abs cache).

        Does NOT touch σ / evolution paths / μ — those are handled by the
        existing plateau-reset block in fit() so this helper is composable.

        Returns: (old_pop, new_pop, label_for_log).
        """
        old_pop = int(self.pop_size)
        new_pop = self._compute_restart_pop_size()
        strategy = str(getattr(self.cfg, "restart_strategy", "none")).lower()
        if strategy == "none" or new_pop == old_pop:
            return old_pop, new_pop, ""
        regime_label = ""
        if strategy == "bipop":
            regime_label = " (large)" if self._bipop_use_large else " (small)"
            # Flip regime for the NEXT restart.
            self._bipop_use_large = not self._bipop_use_large
        self.pop_size = int(new_pop)
        # Rebuild utility-derived state for the new λ.
        self.utilities = tf.constant(self.compute_utilities(), dtype=tf.float32)
        recomb_w_np = self._recomb_w.numpy()
        self._mu_eff = float(1.0 / np.sum(recomb_w_np ** 2))
        # _neg_recomb_abs auto-invalidated by compute_utilities (cleared to None).
        self._restart_count += 1
        return old_pop, new_pop, f"{strategy.upper()}{regime_label}"

    def validate(self, val_data: dict[str, tf.Tensor], mu_tf: tf.Tensor | None = None) -> float:
        """Compute mean RMSE on a subset of validation structures using batched predict.

        σ-read-site note (C4 audit): this method does NOT read `self.sigma` —
        inference goes through `self.model.predict_batch(... mu ...)` only,
        with no sampling step. A future refactor that introduces a σ-read on
        the inference path MUST route through `self._active_sigma_vec()` so
        the read is correct under `cov_mode="crfmnes"` (where `self.sigma` is
        dead and the active scale lives in `_cr_sig * _cr_D`).

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
            W0_2, b0_2 = named.get("W0_2"), named.get("b0_2")
            W0p = named.get("W0_pol")
            b0p = named.get("b0_pol")
            W1p = named.get("W1_pol")
            b1p = named.get("b1_pol")
            W0_2_pol = named.get("W0_2_pol")
            b0_2_pol = named.get("b0_2_pol")
            U_pair_val = named["U_pair_list"]
            # Gates and V_cross must reach _W0_eff in the validate path too
            # — otherwise the train/val mismatch causes val_RMSE to rise
            # while train improves under the same gates.
            gates_val = named.get("gates")
            V_cross_val = named.get("V_cross")
            W_pre_angular_val = named.get("W_pre_angular")
            # Absorb U_pair^T into W0 (and W0_pol) once when LINEAR. For
            # nonlinear mixing we cannot fold; the forward path would need
            # explicit per-layer descriptor mixing — out of scope here.
            if (U_pair_val is not None
                    or gates_val is not None
                    or V_cross_val is not None) and not getattr(
                    self.model, "descriptor_mixing_nonlinear", False):
                W0 = self.model._W0_eff(
                    W0, U_pair_val,
                    V_cross_override=V_cross_val,
                    gates_override=gates_val)
                if W0p is not None:
                    W0p = self.model._W0_eff(
                        W0p, U_pair_val,
                        V_cross_override=V_cross_val,
                        gates_override=gates_val)
            # Preprocessing fold (mutually exclusive with mixing/gating, so
            # this never composes with the W0_eff fold above).
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
                "fold": (W0, b0, W1, b1, W0_2, b0_2,
                         W0p, b0p, W1p, b1p, W0_2_pol, b0_2_pol),
            }
        elif _cache_hit:
            # Reuse the fold from the previous validate() call this gen.
            (W0, b0, W1, b1, W0_2, b0_2,
             W0p, b0p, W1p, b1p, W0_2_pol, b0_2_pol) = _cache["fold"]
        else:
            W0, b0, W1, b1 = self.model.W0, self.model.b0, self.model.W1, self.model.b1
            W0_2 = getattr(self.model, "W0_2", None)
            b0_2 = getattr(self.model, "b0_2", None)
            W0p = getattr(self.model, 'W0_pol', None)
            b0p = getattr(self.model, 'b0_pol', None)
            W1p = getattr(self.model, 'W1_pol', None)
            b1p = getattr(self.model, 'b1_pol', None)
            W0_2_pol = getattr(self.model, "W0_2_pol", None)
            b0_2_pol = getattr(self.model, "b0_2_pol", None)
            # If the model has mixing OR gating active, fold the appropriate
            # transforms into W0 / W0_pol for the validate forward.
            if (getattr(self.model, "descriptor_mixing", False)
                    or getattr(self.model, "descriptor_gating_enabled", False)
                    ) and not getattr(self.model, "descriptor_mixing_nonlinear", False):
                W0 = self.model._W0_eff(W0)
                if W0p is not None:
                    W0p = self.model._W0_eff(W0p)
            # Preprocessing fold for the no-override branch — uses the
            # current self.model.W_pre_angular Variable implicitly.
            if (getattr(self.model, "descriptor_preprocess_contract", "off")
                    != "off"):
                W0 = self.model._W0_preprocess_eff(W0)
                if W0p is not None:
                    W0p = self.model._W0_preprocess_eff(W0p)

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

        def _consume(chunk, chunk_idx=0):
            nonlocal diff_sq_sum, diff_count
            if self._per_l_heads:
                # Build singleton-candidate per-l tensors and use
                # predict_per_l_batch_candidates with C=1.
                if mu_tf is not None:
                    W0_per_l_v = [t[tf.newaxis] for t in named["W0_per_l"]]
                    b0_per_l_v = [t[tf.newaxis] for t in named["b0_per_l"]]
                    W1_per_l_v = [t[tf.newaxis] for t in named["W1_per_l"]]
                    b1_per_l_v = [t[tf.newaxis] for t in named["b1_per_l"]]
                else:
                    W0_per_l_v = [v[tf.newaxis] for v in self.model.W0_per_l]
                    b0_per_l_v = [v[tf.newaxis] for v in self.model.b0_per_l]
                    W1_per_l_v = [v[tf.newaxis] for v in self.model.W1_per_l]
                    b1_per_l_v = [v[tf.newaxis] for v in self.model.b1_per_l]
                # Precompute W_atom for dipole, same as predict_batch does.
                if self.cfg.target_mode == 1:
                    B_arg = chunk["descriptors"].shape[0]
                    A_arg = chunk["descriptors"].shape[1]
                    W_atom_v = self.model._precompute_dipole_kernel(
                        chunk["grad_values"], chunk["pair_struct"],
                        chunk["pair_atom"], chunk["pair_gidx"],
                        chunk["positions"], chunk["boxes"], B_arg, A_arg)
                else:
                    W_atom_v = None
                preds = self.model.predict_per_l_batch_candidates(
                    chunk["descriptors"], W_atom_v,
                    chunk["Z_int"], chunk["atom_mask"],
                    W0_per_l_v, b0_per_l_v, W1_per_l_v, b1_per_l_v,
                    gates=None)
                preds = tf.squeeze(preds, axis=0)
            else:
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
                    W0_2=W0_2, b0_2=b0_2,
                    W0_2_pol=W0_2_pol, b0_2_pol=b0_2_pol,
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
                    pin_to_cpu=self.cfg.pin_data_to_cpu,
                    enabled=getattr(self.cfg, "chunk_prefetch", True),
                    depth=getattr(self.cfg, "prefetch_depth", 1))):
                _consume(chunk, _ci)
                del chunk
        else:
            for _ci, s_start in enumerate(range(0, N_idx, struct_chunk)):
                s_end = min(s_start + struct_chunk, N_idx)
                sub_idx = val_idx_tf[s_start:s_end]
                chunk = slice_and_complete_chunk(val_data, sub_idx)
                if self.cfg.pin_data_to_cpu:
                    with tf.device('/GPU:0'):
                        chunk = {k: (tf.identity(v) if not k.startswith("_") else v)
                                 for k, v in chunk.items()}
                _consume(chunk, _ci)
                del chunk

        rmse = tf.sqrt(tf.maximum(diff_sq_sum / tf.maximum(diff_count, 1.0), 0.0))
        return float(rmse)

    def _split_reconstructed(self, params: tuple) -> dict:
        """Parse the heterogeneous tuple returned by reconstruct_params_tf
        into a named dict. Order of fields per ANN is:
            single hidden  : W0, b0, W1, b1
            two hidden     : W0, b0, W0_2, b0_2, W1, b1
        Optionally followed (in order) by:
            U_pair         : single tensor (N==1) OR list (N>1)
            b_mix          : list of N tensors (nonlinear mixing only)
        Returns dict keys:
            W0, b0, W0_2, b0_2, W1, b1                       (always)
            W0_pol, b0_pol, W0_2_pol, b0_2_pol, W1_pol, b1_pol (mode==2)
            U_pair, U_pair_list, b_mix_list                  (when mixing)
        """
        has_h2 = self.H2 is not None
        out: dict = {"W0_2": None, "b0_2": None,
                     "W0_2_pol": None, "b0_2_pol": None,
                     "U_pair": None, "U_pair_list": None,
                     "b_mix_list": None,
                     "W0_per_l": None, "b0_per_l": None,
                     "W1_per_l": None, "b1_per_l": None,
                     "W0_pol_per_l": None, "b0_pol_per_l": None,
                     "W1_pol_per_l": None, "b1_pol_per_l": None}
        idx = 0
        if self._per_l_heads:
            # Each ANN contributes 4 list-of-L tensors (W0, b0, W1, b1).
            out["W0_per_l"] = params[idx]; idx += 1
            out["b0_per_l"] = params[idx]; idx += 1
            out["W1_per_l"] = params[idx]; idx += 1
            out["b1_per_l"] = params[idx]; idx += 1
            if self.cfg.target_mode == 2:
                out["W0_pol_per_l"] = params[idx]; idx += 1
                out["b0_pol_per_l"] = params[idx]; idx += 1
                out["W1_pol_per_l"] = params[idx]; idx += 1
                out["b1_pol_per_l"] = params[idx]; idx += 1
        else:
            out["W0"] = params[idx]; idx += 1
            out["b0"] = params[idx]; idx += 1
            if has_h2:
                out["W0_2"] = params[idx]; idx += 1
                out["b0_2"] = params[idx]; idx += 1
            out["W1"] = params[idx]; idx += 1
            out["b1"] = params[idx]; idx += 1
            if self.cfg.target_mode == 2:
                out["W0_pol"] = params[idx]; idx += 1
                out["b0_pol"] = params[idx]; idx += 1
                if has_h2:
                    out["W0_2_pol"] = params[idx]; idx += 1
                    out["b0_2_pol"] = params[idx]; idx += 1
                out["W1_pol"] = params[idx]; idx += 1
                out["b1_pol"] = params[idx]; idx += 1
        if self.n_U_pair > 0 and idx < len(params):
            entry = params[idx]; idx += 1
            if isinstance(entry, list):
                out["U_pair_list"] = entry
                out["U_pair"] = entry[0]  # backward-compat alias
            else:
                out["U_pair"] = entry
                out["U_pair_list"] = [entry]
            if self.n_U_bias_total > 0 and idx < len(params):
                out["b_mix_list"] = params[idx]; idx += 1
        out["V_cross"] = None
        if self.n_U_cross > 0 and idx < len(params):
            out["V_cross"] = params[idx]; idx += 1
        out["gates"] = None
        if self.n_gates > 0 and idx < len(params):
            out["gates"] = params[idx]; idx += 1
        out["W_pre_angular"] = None
        if self.n_preprocess > 0 and idx < len(params):
            out["W_pre_angular"] = params[idx]; idx += 1
        out["R_pair"] = None
        if self.n_R_pair > 0 and idx < len(params):
            out["R_pair"] = params[idx]; idx += 1
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
        H2 = self.H2
        H_final = self.H_final

        n_W0 = T * Q * H
        n_b0 = T * H
        n_W0_2 = T * H * H2 if H2 is not None else 0
        n_b0_2 = T * H2 if H2 is not None else 0
        n_W1 = T * H_final
        n_b1 = 1

        is_batched = len(param_vectors.shape) == 2

        def _extract_per_l(pv, offset):
            """Extract one ANN's worth of per-l-head tensors.
            Returns (W0_list, b0_list, W1_list, b1_list, offset)
            where each *_list is a Python list of L tensors.
            """
            L = self._per_l_L
            W0_list, b0_list, W1_list, b1_list = [], [], [], []
            for l in range(L):
                Q_l = self._per_l_Q_l[l]
                n_W0_l = T * Q_l * H
                W0_l = tf.reshape(
                    pv[..., offset:offset + n_W0_l],
                    [-1, T, Q_l, H] if is_batched else [T, Q_l, H])
                offset += n_W0_l
                n_b0_l = T * H
                b0_l = tf.reshape(
                    pv[..., offset:offset + n_b0_l],
                    [-1, T, H] if is_batched else [T, H])
                offset += n_b0_l
                n_W1_l = T * H
                W1_l = tf.reshape(
                    pv[..., offset:offset + n_W1_l],
                    [-1, T, H] if is_batched else [T, H])
                offset += n_W1_l
                b1_l = pv[..., offset]
                offset += 1
                W0_list.append(W0_l)
                b0_list.append(b0_l)
                W1_list.append(W1_l)
                b1_list.append(b1_l)
            return W0_list, b0_list, W1_list, b1_list, offset

        def _extract(pv, offset):
            """Extract one ANN's weights.

            Returns one of:
              (W0, b0, W1, b1, offset)               — single hidden (legacy)
              (W0, b0, W0_2, b0_2, W1, b1, offset)  — two hidden layers
            """
            W0 = tf.reshape(pv[..., offset:offset + n_W0],
                            [-1, T, Q, H] if is_batched else [T, Q, H])
            offset += n_W0
            b0 = tf.reshape(pv[..., offset:offset + n_b0],
                            [-1, T, H] if is_batched else [T, H])
            offset += n_b0
            if H2 is not None:
                W0_2 = tf.reshape(pv[..., offset:offset + n_W0_2],
                                  [-1, T, H, H2] if is_batched else [T, H, H2])
                offset += n_W0_2
                b0_2 = tf.reshape(pv[..., offset:offset + n_b0_2],
                                  [-1, T, H2] if is_batched else [T, H2])
                offset += n_b0_2
            W1 = tf.reshape(pv[..., offset:offset + n_W1],
                            [-1, T, H_final] if is_batched else [T, H_final])
            offset += n_W1
            b1 = pv[..., offset]  # [P] or scalar
            offset += n_b1
            if H2 is not None:
                return W0, b0, W0_2, b0_2, W1, b1, offset
            return W0, b0, W1, b1, offset

        if self._per_l_heads:
            first = _extract_per_l(param_vectors, 0)
        else:
            first = _extract(param_vectors, 0)
        offset = first[-1]
        primary = first[:-1]

        if self.cfg.target_mode == 2:
            if self._per_l_heads:
                second = _extract_per_l(param_vectors, offset)
            else:
                second = _extract(param_vectors, offset)
            offset = second[-1]
            tail = primary + second[:-1]
        else:
            tail = primary

        if self.n_U_pair > 0:
            # Unpack the descriptor-mixing tail. Layout depends on
            # `_mix_arch`. For both arches the per-type variant has T
            # contiguous slabs in the flat tail (same convention as
            # the ANN W0). Inside a slab, the inner layout differs:
            #   "linear"  : [V_p=0 (bs²) | V_p=1 (bs²) | ... ]
            #   "l_aware" : [V_{p=0,l=0} (α²) | V_{p=0,l=1} (α²) | ...
            #                | V_{p=1,l=0} (α²) | ... ]
            #
            # For N stacked layers the per-layer payload is contiguous:
            #   [layer_0 (n_per_layer) | layer_1 (n_per_layer) | ... ]
            # We iterate N times, each pass extracting one layer's worth.
            U_pair_list: list = []
            n_per_layer = self.n_U_pair_per_layer
            per_type = self._mix_per_type
            for layer_k in range(self._mix_n_layers):
                U_flat = param_vectors[
                    ...,
                    offset + layer_k * n_per_layer:
                    offset + (layer_k + 1) * n_per_layer]
                U_pair_k = self._reconstruct_one_mixing_layer(
                    U_flat, is_batched, per_type, T)
                U_pair_list.append(U_pair_k)
            offset += self.n_U_pair
            # Backward-compat: N==1 returns a single tensor (legacy API).
            if self._mix_n_layers == 1:
                tail = tail + (U_pair_list[0],)
            else:
                tail = tail + (U_pair_list,)
            # Optional per-layer mixing biases (nonlinear path).
            if self.n_U_bias_total > 0:
                b_mix_list: list = []
                n_b_per_layer = self._mix_n_bias_per_layer
                for layer_k in range(self._mix_n_layers):
                    b_flat = param_vectors[
                        ...,
                        offset + layer_k * n_b_per_layer:
                        offset + (layer_k + 1) * n_b_per_layer]
                    if per_type:
                        shape = [-1, T, self.dim_q] if is_batched else [T, self.dim_q]
                    else:
                        shape = [-1, self.dim_q] if is_batched else [self.dim_q]
                    b_mix_list.append(tf.reshape(b_flat, shape))
                offset += self.n_U_bias_total
                tail = tail + (b_mix_list,)

        # Optional cross-channel mixing layer (single [Q × Q] rotation).
        # Lives at the very end of the parameter vector — after U_pair and
        # any nonlinear bias tail. Reconstructed via the existing Cayley/
        # expm forward map but at Q-sized blocks, so the scatter-cache
        # trick (memory O(Q⁴)) is replaced with a direct scatter_nd into
        # a [Q, Q] tensor.
        if self.n_U_cross > 0:
            cross_flat = param_vectors[..., offset:offset + self.n_U_cross]
            offset += self.n_U_cross
            V_cross = self._reconstruct_cross_layer(cross_flat, is_batched)
            tail = tail + (V_cross,)

        # Optional per-(species-pair, l, central-type) gating tail.
        # Stored as a flat [T · num_pairs · L] block, reshaped on read.
        if self.n_gates > 0:
            T_ = int(self.cfg.num_types)
            PL = self._gating_num_pairs * self._gating_L
            gates_flat = param_vectors[..., offset:offset + self.n_gates]
            offset += self.n_gates
            if is_batched:
                gates = tf.reshape(gates_flat, [-1, T_, PL])
            else:
                gates = tf.reshape(gates_flat, [T_, PL])
            tail = tail + (gates,)

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

        # Output-side mixing R tail: skew upper-triangle params per type
        # → V_R = R − I of shape [(C,) T, H, H] via _cayley_blocks_batched.
        if self.n_R_pair > 0:
            r_flat = param_vectors[..., offset:offset + self.n_R_pair]
            offset += self.n_R_pair
            T_ = int(self._R_per_T)
            H_ = int(self._R_H)
            skew_per = int(self._R_skew_per_block)
            # Reshape r_flat to [..., T, skew_per] so the batched Cayley
            # reconstruction treats the T types as the inner block index.
            if is_batched:
                r_stacked = tf.reshape(r_flat, [-1, T_, skew_per])
            else:
                r_stacked = tf.reshape(r_flat, [T_, skew_per])
            V_R = self._cayley_blocks_batched(r_stacked, H_)
            tail = tail + (V_R,)

        return tail

    def _reconstruct_cross_layer(self, A_upper_flat: tf.Tensor,
                                  is_batched: bool) -> tf.Tensor:
        """Reconstruct V_cross = U_cross − I as a [Q, Q] residual.

        Three modes:
          "full"       : R is [Q, Q]; A is upper-tri of skew(Q); V = expm(A) − I.
          "block_unit" : Per-component-slot R_k matrices (one per slot k
                         across all l), each orthogonal on its participating
                         units. The [Q, Q] W is scattered slot-by-slot.
          "per_l"      : Per-(l, k) R_{l,k} matrices, one per "slot"
                         in the FLATTENED (l, k) index. Uses the same per-
                         slot infrastructure as block_unit, just with the
                         slots defined per-l. Preserves angular-momentum
                         separation: l=0 rotation never touches l>0 channels.
        A_upper_flat shape: [P, n_U_cross] or [n_U_cross].
        Returns V_cross shape: [P, Q, Q] or [Q, Q].
        """
        Q = self._cross_Q
        if self._cross_mode == "full":
            R_size = self._cross_R_size
            flat_upper = self._cross_upper_i * R_size + self._cross_upper_j
            if is_batched:
                def _per_cand(payload):
                    buf = tf.scatter_nd(
                        indices=tf.reshape(flat_upper, [-1, 1]),
                        updates=payload,
                        shape=[R_size * R_size])
                    A = tf.reshape(buf, [R_size, R_size])
                    return A - tf.transpose(A)
                A = tf.map_fn(_per_cand, A_upper_flat)
            else:
                buf = tf.scatter_nd(
                    indices=tf.reshape(flat_upper, [-1, 1]),
                    updates=A_upper_flat,
                    shape=[R_size * R_size])
                A = tf.reshape(buf, [R_size, R_size])
                A = A - tf.transpose(A)
            if self._cross_orth_map == "expm":
                R = tf.linalg.expm(A)
            else:
                I_R = tf.eye(R_size, dtype=A.dtype)
                R = tf.linalg.solve(I_R - A, I_R + A)
            return R - tf.eye(Q, dtype=R.dtype)

        # block_unit and per_l modes: build W slot-by-slot from the
        # per-slot R matrices. Both modes share this runtime; they only
        # differ in how "slots" are defined at init time (across-l vs
        # per-l flattened over (l, k)).
        def _build_W(payload):
            """payload: [n_U_cross] flat skew payload."""
            W = tf.eye(Q, dtype=payload.dtype)
            for k, (N_k, n_pay, off, q_idx, upper) in enumerate(zip(
                    self._cross_slot_N,
                    self._cross_slot_n_payload,
                    self._cross_slot_offsets,
                    self._cross_slot_q_indices,
                    self._cross_slot_upper)):
                if N_k < 2:
                    continue
                ii, jj = upper
                A_pay = payload[off:off + n_pay]
                flat_upper = ii * N_k + jj
                buf = tf.scatter_nd(
                    indices=tf.reshape(flat_upper, [-1, 1]),
                    updates=A_pay,
                    shape=[N_k * N_k])
                A_k = tf.reshape(buf, [N_k, N_k])
                A_k = A_k - tf.transpose(A_k)
                if self._cross_orth_map == "expm":
                    R_k = tf.linalg.expm(A_k)
                else:
                    I_k = tf.eye(N_k, dtype=A_k.dtype)
                    R_k = tf.linalg.solve(I_k - A_k, I_k + A_k)
                # Scatter R_k into W at positions (q_idx[i], q_idx[j]).
                # The scatter REPLACES the corresponding identity entries.
                ii_q, jj_q = tf.meshgrid(q_idx, q_idx, indexing="ij")
                idx_pairs = tf.stack(
                    [tf.reshape(ii_q, [-1]),
                     tf.reshape(jj_q, [-1])], axis=1)            # [N_k², 2]
                # First zero out the diagonal entries at these slot
                # positions so the eventual W has the right block in place
                # of the identity row(s).
                W = tf.tensor_scatter_nd_update(
                    W,
                    indices=tf.stack([q_idx, q_idx], axis=1),
                    updates=tf.zeros([N_k], dtype=W.dtype))
                W = tf.tensor_scatter_nd_add(
                    W, idx_pairs, tf.reshape(R_k, [-1]))
            return W - tf.eye(Q, dtype=W.dtype)
        if is_batched:
            V_cross = tf.map_fn(_build_W, A_upper_flat)
        else:
            V_cross = _build_W(A_upper_flat)
        return V_cross

    def _reconstruct_one_mixing_layer(self, U_flat: tf.Tensor,
                                        is_batched: bool,
                                        per_type: bool,
                                        T: int) -> tf.Tensor:
        """Reconstruct ONE mixing-layer's U_pair tensor from its flat slab.

        Extracted from the body of reconstruct_params_tf so we can call it
        once per stacked mixing layer. The arch-dispatch and Cayley fast
        paths mirror the legacy single-layer code exactly.
        """
        if self._mix_arch == "cross_pair_l":
            N_l = self._mix_N_per_l
            L = self._mix_L
            block_payload = (N_l * (N_l - 1) // 2) if self._mix_cayley else N_l * N_l
            per_T_block = L * block_payload

            def _extract_t_block_cross_pair_l(t_idx: int):
                start = t_idx * per_T_block
                slab = U_flat[..., start:start + per_T_block]
                if self._mix_cayley:
                    new_shape = tf.concat(
                        [tf.shape(slab)[:-1], [L, block_payload]], axis=0)
                    stacked = tf.reshape(slab, new_shape)
                    return self._cayley_blocks_batched(stacked, N_l)
                if is_batched:
                    return tf.reshape(slab, [-1, L, N_l, N_l])
                return tf.reshape(slab, [L, N_l, N_l])

            if per_type:
                per_t = [_extract_t_block_cross_pair_l(t) for t in range(T)]
                stack_axis = 1 if is_batched else 0
                return tf.stack(per_t, axis=stack_axis)
            return _extract_t_block_cross_pair_l(0)

        if self._mix_arch == "linear":
            max_bs = max(self._mix_block_sizes)
            bs_payloads = [
                (bs * (bs - 1) // 2) if self._mix_cayley else bs * bs
                for bs in self._mix_block_sizes
            ]
            per_T_block = sum(bs_payloads)
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
                        [tf.shape(slab)[:-1], [n_pairs, payload]], axis=0)
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
                return tf.stack(per_t, axis=stack_axis)
            return _extract_t_block_linear(0)

        # l_aware
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

            # Non-Cayley (orthogonal regularizer) non-uniform fallback:
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
            # Fall back to the per-T list-comp ONLY for the non-Cayley
            # orthogonal regularizer (no expm to batch). Cayley/expm
            # paths (uniform-α and non-uniform-α) are handled above with
            # a single batched expm call across T.
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
        Q = self.dim_q
        H = self.cfg.num_neurons

        reg_Vpair = (self.n_U_pair > 0 and self._mix_reg_mode == "shrinkage")
        reg_Vorth = (self.n_U_pair > 0 and self._mix_reg_mode == "orthogonal")

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
            # See compute_regularization() for the layout rationale; same
            # bounded slice and same layers-major / types-inner unpacking.
            tail = param_vectors[:, self.n_anns_total
                                 : self.n_anns_total + self.n_U_pair]
            if self._mix_per_type and T > 1:
                per_layer = self.n_U_pair_per_layer
                per_T_block = per_layer // T
                vp_l1 = tf.zeros([tf.shape(param_vectors)[0]])
                vp_l2 = tf.zeros([tf.shape(param_vectors)[0]])
                for t in range(T):
                    if self._mix_n_layers == 1:
                        slab = tail[:, t * per_T_block:(t + 1) * per_T_block]
                    else:
                        slab = tf.concat(
                            [tail[:, k * per_layer + t * per_T_block
                                   : k * per_layer + (t + 1) * per_T_block]
                             for k in range(self._mix_n_layers)], axis=1)
                    n_per_t = per_T_block * self._mix_n_layers
                    vp_l1 += self.lambda_1 * tf.reduce_sum(tf.abs(slab), axis=1) / n_per_t
                    vp_l2 += self.lambda_2 * tf.sqrt(
                        tf.reduce_sum(tf.square(slab), axis=1) / n_per_t)
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
            tail = param_vectors[:, self.n_anns_total
                                 : self.n_anns_total + self.n_U_pair]  # [P, n_U_pair]
            reg = reg + self._lambda_orth * self._orth_penalty_total(tail)

        return reg

    def _extract_type_params_batched(self, param_vectors: tf.Tensor, t: int) -> tf.Tensor:
        """Extract type-t parameters from batched flat vectors.

        Returns [P, n_per_type] — see _extract_type_params for layout.
        """
        T = self.cfg.num_types
        Q = self.dim_q
        H = self.H
        H2 = self.H2
        H_final = self.H_final

        w0_start = t * Q * H
        w0_end = w0_start + Q * H

        b0_offset = T * Q * H
        b0_start = b0_offset + t * H
        b0_end = b0_start + H

        parts = [
            param_vectors[:, w0_start:w0_end],
            param_vectors[:, b0_start:b0_end],
        ]

        if H2 is not None:
            w0_2_offset = b0_offset + T * H
            w0_2_start = w0_2_offset + t * H * H2
            w0_2_end = w0_2_start + H * H2
            b0_2_offset = w0_2_offset + T * H * H2
            b0_2_start = b0_2_offset + t * H2
            b0_2_end = b0_2_start + H2
            parts.extend([
                param_vectors[:, w0_2_start:w0_2_end],
                param_vectors[:, b0_2_start:b0_2_end],
            ])
            w1_offset = b0_2_offset + T * H2
        else:
            w1_offset = b0_offset + T * H

        w1_start = w1_offset + t * H_final
        w1_end = w1_start + H_final
        parts.append(param_vectors[:, w1_start:w1_end])

        return tf.concat(parts, axis=1)

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
        loss_type = "mse"
        # Pick between the standard and XLA-compiled chunk evaluator.
        # XLA fuses the per-chunk eval into a single GPU kernel and is
        # typically 1.5-2× faster, but compiles per unique (B, P) shape
        # — paid as upfront cost in the first generation.
        eval_chunk_fn = (self._evaluate_chunk_xla
                         if getattr(self.cfg, "eval_jit_compile", False)
                         else self._evaluate_chunk)

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
        # include a U_pair tensor (or list) when cfg.descriptor_mixing=True.
        params = self.reconstruct_params_tf(chunk_samples)
        named = self._split_reconstructed(params)
        W0 = named.get("W0"); b0 = named.get("b0")
        W1 = named.get("W1"); b1 = named.get("b1")
        W0_2_cand, b0_2_cand = named.get("W0_2"), named.get("b0_2")
        W0p = named.get("W0_pol")
        b0p = named.get("b0_pol")
        W1p = named.get("W1_pol")
        b1p = named.get("b1_pol")
        W0_2_pol_cand = named.get("W0_2_pol")
        b0_2_pol_cand = named.get("b0_2_pol")
        # U_pair_cand: the multi-layer list (or None when no mixing).
        U_pair_cand = named["U_pair_list"]
        # V_cross_cand: per-candidate cross-channel V (or None if disabled).
        V_cross_cand = named.get("V_cross")
        # gates_cand: per-(candidate, type, pair·L+l) gates (or None if disabled).
        gates_cand = named.get("gates")
        # W_pre_angular_cand: per-(candidate, type, q_raw) preprocess
        # coefficients (or None when descriptor_preprocess_contract='off').
        W_pre_angular_cand = named.get("W_pre_angular")
        # Per-l ANN heads — when enabled, predict_per_l_batch_candidates
        # replaces predict_batch_candidates entirely.
        W0_per_l_cand = named.get("W0_per_l")
        b0_per_l_cand = named.get("b0_per_l")
        W1_per_l_cand = named.get("W1_per_l")
        b1_per_l_cand = named.get("b1_per_l")

        loss_type = "mse"
        huber_delta = 0.0
        comp_w = None

        if self.cfg.target_mode == 2:
            pol_weights = self._pol_weights  # [6] component weights
            # Pre-absorb U_pair^T into W0 / W0_pol per candidate so the
            # vectorized_map loop body uses raw descriptors and no
            # gradient pull-back. Works because _W0_eff is linear and
            # broadcasts over the leading candidate axis. Skipped for
            # nonlinear mixing (cannot fold tanh).
            if (U_pair_cand is not None
                    and not getattr(self.model, "descriptor_mixing_nonlinear", False)):
                W0 = self.model._W0_eff(W0, U_pair_cand,
                                        V_cross_override=V_cross_cand,
                                        gates_override=gates_cand)
                W0p = self.model._W0_eff(W0p, U_pair_cand,
                                         V_cross_override=V_cross_cand,
                                         gates_override=gates_cand)

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

            if W0_2_cand is None:
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
            else:
                def _forward_one_candidate(args):
                    (w0, bb0, w0_2, bb0_2,
                     w1, bb1, w0p, bb0p,
                     w0_2_p, bb0_2_p, w1p, bb1p) = args
                    preds = self.model.predict_batch(
                        desc, grad_values, pair_atom, pair_gidx, pair_struct,
                        pos, Z, boxes, amask,
                        w0, bb0, w1, bb1, w0p, bb0p, w1p, bb1p,
                        W0_2=w0_2, b0_2=bb0_2,
                        W0_2_pol=w0_2_p, b0_2_pol=bb0_2_p,
                    )
                    diff = preds - targets
                    fitness = per_structure_error(
                        diff, loss_type, huber_delta, component_weights=fitness_comp_w)
                    sq = squared_error_per_structure(diff, component_weights=sq_comp_w)
                    return tf.stack([fitness, sq], axis=0)

                stacked = (W0, b0, W0_2_cand, b0_2_cand,
                           W1, b1, W0p, b0p,
                           W0_2_pol_cand, b0_2_pol_cand, W1p, b1p)
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

            if self._per_l_heads:
                # Per-l ANN heads path: separate forward pass that loops
                # over l and sums per-l dipole/PES contributions.
                preds = self.model.predict_per_l_batch_candidates(
                    desc, W_atom, Z, amask,
                    W0_per_l_cand, b0_per_l_cand,
                    W1_per_l_cand, b1_per_l_cand,
                    gates=gates_cand)
            else:
                # Evaluate all C candidates simultaneously using explicit batched GEMMs.
                # predict_batch_candidates executes one GEMM per type in each direction
                # rather than C separate matmuls inside vectorized_map.
                preds = self.model.predict_batch_candidates(
                    desc, W_atom, Z, amask, W0, b0, W1, b1,
                    U_pair=U_pair_cand,
                    W0_2=W0_2_cand, b0_2=b0_2_cand,
                    V_cross=V_cross_cand,
                    gates=gates_cand,
                    W_pre_angular=W_pre_angular_cand,
                    R_pair=named.get("R_pair"))  # [C, B, T_dim]

            if _scale_preds:
                preds = preds * inv_num_atoms[tf.newaxis]  # [C, B, T_dim] * [1, B, 1]

            diff = preds - targets[tf.newaxis]  # [C, B, T_dim]
            fitness_comp_w = (comp_w[tf.newaxis] if comp_w is not None else None)
            fitness = per_structure_error(
                diff, loss_type, huber_delta, component_weights=fitness_comp_w)
            sq = squared_error_per_structure(diff)  # unweighted, for RMSE/RRMSE
            return fitness, sq
