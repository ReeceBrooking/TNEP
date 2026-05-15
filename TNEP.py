"""TNEP — GAP (Gaussian Approximation Potential) model.

This file replaces the previous NN-based TNEP. The class name and the
file name are retained so all call sites (`from TNEP import TNEP`,
`model.score(...)`, `model.predict_batch(...)`, etc.) continue to work
unchanged. Internally the per-atom scalar U_i is now computed by a
kernel expansion over a sparse set of training environments:

    U_i = Σ_s α_s · K(q_i, q_s)

with K = δ_t² · (q̂_i · q̂_s)^ζ (L2-normalised polynomial dot-product
kernel; species-masked so atoms only see sparse points of their own
type). The downstream dipole pathway (μ = -Σ r²·∂U/∂r) is identical to
the NN — only ∂U/∂q changes from the NN tanh-Jacobian to the
kernel-trick streaming form.

Phase 1 constraints (lifted later):
- `cfg.target_mode == 2` (polarisability) raises NotImplementedError —
  Phase 3.
- Dummy `W0/b0/W1/b1` (zero-initialised, untrainable) are kept on the
  model so legacy call sites that read `model.W0` etc. continue to
  work without crashing (e.g. `score()`, `spectroscopy._get_fused_predict`).
"""
from __future__ import annotations

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm

from DescriptorBuilder import make_descriptor_builder
from SNES import SNES
from TNEPconfig import TNEPconfig


class TNEP(layers.Layer):
    """GAP-style per-atom scalar predictor with the original TNEP I/O.

    Forward (per atom i, type t = species(i)):
        q̂_i = q_i / ||q_i||
        K(q_i, q_s) = δ_t² · (q̂_i · q̂_s)^ζ · [species(i) == species(s)]
        U_i = Σ_s α_s · K(q_i, q_s)

    Prediction (cfg.target_mode):
        0 (PES)    : E = -Σ_i U_i                                   → scalar
        1 (Dipole) : μ = -Σ_ij r_ij² · ∂U_i/∂r_ij                   → [3]
        2 (Polar.) : Phase 3 — raises NotImplementedError in __init__.

    Persistent state (populated by `self.optimizer.fit`):
        sparse_q          [M, Q]      L2-normalised sparse-point descriptors
        sparse_q_norm     [M]         ||q_s|| (kept for diagnostics)
        sparse_species    [M]  int32  species index of each sparse point
        alpha             [M]         fitted coefficients
        delta_per_species [T]         per-species kernel scale (default 1.0)
    """

    def __init__(self, cfg: TNEPconfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.dim_q = cfg.dim_q
        self.num_types = cfg.num_types
        self.builder = make_descriptor_builder(cfg)

        # ───────────────────────────── Phase 1 mode gate
        # target_mode==2 (polarisability) reuses NN-only attributes
        # (W0_pol/b0_pol/W1_pol/b1_pol) and is not implemented under
        # the GAP rewrite. Phase 3 lifts this.
        if cfg.target_mode == 2:
            raise NotImplementedError(
                "target_mode=2 (polarisability) is deferred to GAP Phase 3. "
                "Use target_mode=1 (dipole) or target_mode=0 (energy) for now.")

        # ───────────────────────────── GAP weights (must be created BEFORE
        # `self.optimizer = SNES(self)` below — the shim reads them).
        M = int(getattr(cfg, "gap_n_sparse", 1000))
        Q = int(cfg.dim_q)
        T = int(cfg.num_types)
        self.M = M

        self.sparse_q = self.add_weight(
            name="sparse_q", shape=(M, Q),
            initializer="zeros", trainable=False)
        self.sparse_q_norm = self.add_weight(
            name="sparse_q_norm", shape=(M,),
            initializer="ones", trainable=False)
        self.sparse_species = self.add_weight(
            name="sparse_species", shape=(M,),
            initializer="zeros", trainable=False, dtype=tf.int32)
        self.alpha = self.add_weight(
            name="alpha", shape=(M,),
            initializer="zeros", trainable=False)
        self.delta_per_species = self.add_weight(
            name="delta_per_species", shape=(T,),
            initializer="ones", trainable=False)

        # Dummy NN-weight zeros, kept only because `spectroscopy.py`
        # passes `model.W0` / `model.b0` / `model.W1` / `model.b1`
        # through to `predict_batch` (which ignores them under GAP).
        self.W0 = self.add_weight(
            name="W0_compat", shape=(T, Q, 1),
            initializer="zeros", trainable=False)
        self.b0 = self.add_weight(
            name="b0_compat", shape=(T, 1),
            initializer="zeros", trainable=False)
        self.W1 = self.add_weight(
            name="W1_compat", shape=(T, 1),
            initializer="zeros", trainable=False)
        self.b1 = self.add_weight(
            name="b1_compat", shape=(),
            initializer="zeros", trainable=False)
        self.W0_pol = None
        self.b0_pol = None
        self.W1_pol = None
        self.b1_pol = None

        # Descriptor-mixing compat stubs — spectroscopy.py guards
        # `model._W0_eff(...)` calls with `model.descriptor_mixing`.
        self.descriptor_mixing = False
        self.U_pair = None

        # ───────────────────────────── Optimizer shim
        # Phase-1 SNES is a thin closed-form GAP fitter — see SNES.py.
        # Constructed last so the shim can read the GAP weights above.
        self.optimizer = SNES(self)

    # ───────────────────────────────────── Compat helpers (no-ops)

    def _W0_eff(self, W0: tf.Tensor,
                U_pair: tf.Tensor | None = None) -> tf.Tensor:
        """Identity no-op. Retained for backwards-compat with callers
        (`score()`, `spectroscopy._get_fused_predict`) that still
        invoke `_W0_eff(model.W0)`. The GAP forward never uses W0; this
        just keeps the call sites alive without modification."""
        return W0

    # ───────────────────────────────────── Kernel forward primitives

    def _kernel_grad_de_dq(self,
                            descriptors: tf.Tensor,
                            Z_int: tf.Tensor,
                            atom_mask: tf.Tensor) -> tf.Tensor:
        """Compute ∂U_i/∂q_i for the gradient pathway.

        Returns `[B, A, Q]`.

        For polynomial dot-product kernel
            K(q_i, q_s) = δ_t² · (q̂_i · q̂_s)^ζ      q̂ = q / ||q||
        the gradient w.r.t. q_i is
            ∂K/∂q_i = (δ_t² · ζ / ||q_i||) · (q̂_i · q̂_s)^(ζ-1) ·
                       (q̂_s − (q̂_i · q̂_s) · q̂_i)
        Contracting with α_s and summing over s (species-masked) gives
            ∂U_i/∂q_i = (1/||q_i||) · ( Σ_s a_s · q̂_s
                                       − (Σ_s a_s · cos_is) · q̂_i )
        where a_s = α_s · δ_t(s)² · ζ · cos_is^(ζ-1) · mask(i, s).

        Streaming form avoids materialising any `[B, A, M, Q]` tensor.

        Args:
            descriptors : [B, A, Q]
            Z_int       : [B, A]    int32 type indices
            atom_mask   : [B, A]    1.0 for real atoms, 0.0 for padding

        Returns:
            de_dq : [B, A, Q]
        """
        cfg = self.cfg
        zeta = int(getattr(cfg, "gap_zeta", 4))
        eps = 1e-12

        # ────── normalise q_i
        q = descriptors                                      # [B, A, Q]
        q_norm = tf.linalg.norm(q, axis=-1)                  # [B, A]
        q_norm_safe = tf.maximum(q_norm, eps)                # [B, A]
        q_hat = q / q_norm_safe[..., tf.newaxis]             # [B, A, Q]

        # ────── kernel cos_is = q̂_i · q̂_s (sparse_q is pre-normalised
        # at fit-time, so we use it directly)
        sparse_q = self.sparse_q                             # [M, Q]
        cos_is = tf.einsum('baq,mq->bam', q_hat, sparse_q)   # [B, A, M]

        # ────── species mask
        Z_b = Z_int[..., tf.newaxis]                         # [B, A, 1]
        sp = self.sparse_species[tf.newaxis, tf.newaxis, :]  # [1, 1, M]
        species_mask = tf.cast(tf.equal(Z_b, sp), tf.float32)  # [B, A, M]
        cos_is = cos_is * species_mask

        # ────── δ_t(s)² per sparse point [M] (broadcast from [T]).
        # `sparse_species` may contain -1 for padded slots (mask kills
        # them downstream); tf.gather with -1 would either wrap or
        # error. Clamp to [0, T-1] for the gather only — masked rows
        # are zeroed by the species_mask above.
        delta_sq_M = tf.square(
            tf.gather(self.delta_per_species,
                      tf.maximum(self.sparse_species, 0)))  # [M]

        # ────── a_s = α_s · δ_t(s)² · ζ · cos^(ζ-1) · species_mask
        # cos^(ζ-1) is safe at cos=0 because we use floats and ζ≥1.
        cos_pow = tf.pow(cos_is, zeta - 1)                   # [B, A, M]
        # Mask AGAIN (pow can be ill-defined at cos=0 for ζ=1; the
        # explicit re-mask zeroes any leak):
        cos_pow = cos_pow * species_mask
        a_s_coeff = self.alpha * delta_sq_M * float(zeta)    # [M]
        a_s = cos_pow * a_s_coeff[tf.newaxis, tf.newaxis, :]  # [B, A, M]

        # ────── ∂U/∂q via two contractions over M:
        #   term1[..., q] = Σ_s a_s · q̂_s[q]   = a_s @ sparse_q
        #   term2[...]    = Σ_s a_s · cos_is   (scalar per atom)
        term1 = tf.einsum('bam,mq->baq', a_s, sparse_q)      # [B, A, Q]
        term2 = tf.einsum('bam,bam->ba', a_s, cos_is)        # [B, A]

        # ∂U_i/∂q_i = (1/||q_i||) · ( term1 − term2 · q̂_i )
        de_dq = (term1 - term2[..., tf.newaxis] * q_hat) / q_norm_safe[..., tf.newaxis]

        # Zero padded atoms.
        de_dq = de_dq * atom_mask[..., tf.newaxis]
        return de_dq

    def _kernel_grad_per_sparse_block(self,
                                       descriptors: tf.Tensor,
                                       Z_int: tf.Tensor,
                                       atom_mask: tf.Tensor,
                                       s_lo: int, s_hi: int) -> tf.Tensor:
        """Per-sparse-point kernel-grad contributions for use in Φ build.

        For each atom i and each sparse point s in [s_lo, s_hi), returns
        the **un-contracted** kernel-gradient column:

            de_dq_per_s[k, i, s, q] = (1/||q_i||) · g_is ·
                                       (q̂_s[q] − cos_is · q̂_i[q])
            g_is = δ_t(s)² · ζ · cos_is^(ζ-1) · species_mask(i, s)

        That is: the contribution to ∂U_i/∂q_i if α_s were a one-hot at
        each s in the block. Used by SNES._build_phi to construct the
        design tensor in O(M / M_sub) blocks rather than O(M) full
        forward passes.

        Args:
            descriptors : [B, A, Q]
            Z_int       : [B, A]
            atom_mask   : [B, A]
            s_lo, s_hi  : sparse-point index range [s_lo, s_hi)

        Returns:
            [B, A, s_hi - s_lo, Q]   float32 (padded atoms zeroed)
        """
        cfg = self.cfg
        zeta = int(getattr(cfg, "gap_zeta", 4))
        eps = 1e-12

        q = descriptors
        q_norm = tf.linalg.norm(q, axis=-1)
        q_norm_safe = tf.maximum(q_norm, eps)
        q_hat = q / q_norm_safe[..., tf.newaxis]                # [B, A, Q]

        sparse_q_block = self.sparse_q[s_lo:s_hi]               # [bs, Q]
        cos_is = tf.einsum('baq,sq->bas', q_hat, sparse_q_block)  # [B, A, bs]
        Z_b = Z_int[..., tf.newaxis]
        sp = self.sparse_species[s_lo:s_hi][tf.newaxis, tf.newaxis, :]
        species_mask = tf.cast(tf.equal(Z_b, sp), tf.float32)   # [B, A, bs]
        cos_is = cos_is * species_mask

        delta_sq_block = tf.square(
            tf.gather(self.delta_per_species,
                      tf.maximum(self.sparse_species[s_lo:s_hi], 0)))  # [bs]
        cos_pow = tf.pow(cos_is, zeta - 1) * species_mask        # [B, A, bs]
        g = cos_pow * (delta_sq_block * float(zeta))[tf.newaxis, tf.newaxis, :]
        # g[B, A, bs]

        # de_dq[..., q] = (1/||q_i||) · g · (q̂_s[q] − cos · q̂_i[q])
        # First term:  g[B, A, bs] · q̂_s[bs, Q] → [B, A, bs, Q]
        term1 = g[..., tf.newaxis] * sparse_q_block[tf.newaxis, tf.newaxis, :, :]
        # Second term: g · cos[B, A, bs] · q̂_i[B, A, Q] → [B, A, bs, Q]
        term2 = (g * cos_is)[..., tf.newaxis] * q_hat[..., tf.newaxis, :]
        de_dq_per_s = (term1 - term2) / q_norm_safe[..., tf.newaxis, tf.newaxis]
        # Zero padded atoms.
        de_dq_per_s = de_dq_per_s * atom_mask[..., tf.newaxis, tf.newaxis]
        return de_dq_per_s

    def _kernel_U(self,
                   descriptors: tf.Tensor,
                   Z_int: tf.Tensor,
                   atom_mask: tf.Tensor) -> tf.Tensor:
        """Compute per-atom U_i = Σ_s α_s K(q_i, q_s) for energy mode.

        Args:
            descriptors : [B, A, Q]
            Z_int       : [B, A]
            atom_mask   : [B, A]
        Returns:
            U : [B, A]   per-atom scalar; padded atoms zero.
        """
        cfg = self.cfg
        zeta = int(getattr(cfg, "gap_zeta", 4))
        eps = 1e-12
        q = descriptors
        q_norm = tf.maximum(tf.linalg.norm(q, axis=-1), eps)
        q_hat = q / q_norm[..., tf.newaxis]
        cos_is = tf.einsum('baq,mq->bam', q_hat, self.sparse_q)
        Z_b = Z_int[..., tf.newaxis]
        sp = self.sparse_species[tf.newaxis, tf.newaxis, :]
        species_mask = tf.cast(tf.equal(Z_b, sp), tf.float32)
        cos_is = cos_is * species_mask
        delta_sq_M = tf.square(
            tf.gather(self.delta_per_species,
                      tf.maximum(self.sparse_species, 0)))
        K = (cos_is ** zeta) * delta_sq_M[tf.newaxis, tf.newaxis, :]
        # U_i = Σ_s α_s · K_is
        U = tf.einsum('bam,m->ba', K, self.alpha)
        return U * atom_mask

    # ───────────────────────────────────── Public forward (batched)

    # ───────────────────────────────────── fit shim (delegates to optimizer)

    def fit(self, train_data: dict, val_data: dict) -> tuple:
        """Run training via the optimizer shim.

        Returns `(history, final_model, best_val_model)`. For the
        closed-form GAP solve, `final_model` and `best_val_model` are
        the same object (no iteration distinguishes them).
        """
        return self.optimizer.fit(train_data, val_data)

    # ───────────────────────────────────── score / score_summary (unchanged)

    def score(self, test_data: dict[str, tf.Tensor]) -> tuple[dict, tf.Tensor]:
        """Evaluate RMSE, R², per-component R², and cosine similarity.

        Streams predictions chunk-by-chunk to bound memory. The chunk
        loop calls `model.predict_batch(...)` — the body of which now
        runs the GAP kernel forward but signature is unchanged.
        """
        from data import prefetched_chunks
        S_test = test_data["num_atoms"].shape[0]
        # Score chunk size: if the user hasn't set batch_chunk_size, cap
        # automatically so we don't try to stage the full grad_values
        # `[P, 3, Q]` cache onto the GPU at once. Even modest l_max/Q
        # combos (Q≳200) at S≳500 push this past 5 GB.
        if self.cfg.batch_chunk_size is not None:
            chunk_sz = self.cfg.batch_chunk_size
        else:
            # Estimate per-struct grad_values bytes: avg_pairs_per_struct
            # × 3 × Q × dtype_size. Aim for ≤1 GB transient per chunk.
            grad = test_data.get("grad_values", None)
            if grad is not None and getattr(grad, "shape", None) is not None \
                    and grad.shape[0] is not None and grad.shape[0] > 0:
                P_total = int(grad.shape[0])
                Q = int(grad.shape[-1]) if grad.shape[-1] is not None else 1
                dtype_size = 4  # fp32 (or int32) — both 4 bytes
                bytes_per_struct = max(
                    1, (P_total // max(1, S_test)) * 3 * Q * dtype_size)
                target_bytes = 1 << 30  # 1 GiB
                chunk_sz = max(1, min(S_test, target_bytes // bytes_per_struct))
            else:
                chunk_sz = S_test
        # _W0_eff is a no-op identity stub under GAP, so passing
        # `self.W0` (zero tensor) through it is harmless — the new
        # `predict_batch` ignores W0/b0/W1/b1 anyway.
        W0_eff = self._W0_eff(self.W0)
        W0_pol_eff = None  # Phase 1: mode-2 raised in __init__
        ranges = [(s, min(s + chunk_sz, S_test)) for s in range(0, S_test, chunk_sz)]
        pred_parts: list = []
        # Per-chunk progress bar. Each unit = one struct-chunk forward
        # pass through predict_batch. Matches Φ-build bar styling.
        pbar = tqdm(total=len(ranges), desc="  score", unit="chunk",
                    leave=False, mininterval=0.1,
                    file=sys.stdout, dynamic_ncols=True)
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
            ))
            del chunk
            pbar.update(1)
        pbar.close()
        raw_preds = tf.concat(pred_parts, axis=0)
        del pred_parts
        targets = test_data["targets"]

        if self.cfg.scale_targets and self.cfg.target_mode == 1 and "num_atoms" in test_data:
            num_atoms = tf.cast(test_data["num_atoms"], tf.float32)
            num_atoms_col = tf.maximum(num_atoms, 1.0)[:, tf.newaxis]
            preds = raw_preds / num_atoms_col
        else:
            preds = raw_preds

        # Target-centering inverse (identical to NN version).
        mean_tf = None
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

        ss_res = tf.reduce_sum(tf.square(diff))
        ss_tot = tf.reduce_sum(tf.square(targets - tf.reduce_mean(targets, axis=0)))
        r2 = 1.0 - ss_res / ss_tot

        ss_res_comp = tf.reduce_sum(tf.square(diff), axis=0)
        ss_tot_comp = tf.reduce_sum(
            tf.square(targets - tf.reduce_mean(targets, axis=0)), axis=0)
        r2_components = 1.0 - ss_res_comp / tf.maximum(ss_tot_comp, 1e-12)

        metrics = {
            "rmse": rmse,
            "r2": r2,
            "r2_components": r2_components,
        }

        # Total-space metrics when scale_targets is active.
        if self.cfg.scale_targets and self.cfg.target_mode == 1 and "num_atoms" in test_data:
            total_targets = targets * num_atoms_col
            if mean_tf is not None:
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

        if self.cfg.target_mode >= 1:
            dot = tf.reduce_sum(preds * targets, axis=1)
            norm_p = tf.linalg.norm(preds, axis=1)
            norm_t = tf.linalg.norm(targets, axis=1)
            cos_sim = dot / tf.maximum(norm_p * norm_t, 1e-12)
            metrics["cos_sim_mean"] = tf.reduce_mean(cos_sim)
            metrics["cos_sim_all"] = cos_sim

        return metrics, preds

    def score_summary(self, test_data: dict[str, tf.Tensor]) -> dict:
        """Print labelled per-atom + total-space metrics (matches GPUMD
        loss.out and NEP-paper conventions). Unchanged from the NN
        version."""
        metrics, _ = self.score(test_data)
        m = {k: (float(v.numpy()) if hasattr(v, "numpy") else float(v))
             for k, v in metrics.items()
             if v is not None and getattr(v, "shape", ()) == ()}
        rrmse_pa = (1.0 - m["r2"]) ** 0.5
        print(f"  PER-ATOM space (matches GPUMD loss.out 'rmse_virial'):")
        print(f"    RMSE  = {m['rmse']:.6f}")
        print(f"    R²    = {m['r2']:.6f}")
        print(f"    RRMSE = √(1−R²) = {rrmse_pa:.4%}")
        if "total_rmse" in m and "total_r2" in m:
            rrmse_tot = (1.0 - m["total_r2"]) ** 0.5
            print(f"  TOTAL-DIPOLE space (matches NEP paper headline RMSE / R²):")
            print(f"    RMSE  = {m['total_rmse']:.6f}")
            print(f"    R²    = {m['total_r2']:.6f}")
            print(f"    RRMSE = √(1−R²) = {rrmse_tot:.4%}")
        if "cos_sim_mean" in m:
            print(f"  Vector quality:")
            print(f"    cos_sim_mean = {m['cos_sim_mean']:.6f}")
        return metrics

    @tf.function(reduce_retracing=True)
    def predict_batch(self, descriptors: tf.Tensor, grad_values: tf.Tensor,
                       pair_atom: tf.Tensor, pair_gidx: tf.Tensor,
                       pair_struct: tf.Tensor,
                       positions: tf.Tensor, Z: tf.Tensor,
                       boxes: tf.Tensor, atom_mask: tf.Tensor,
                       W0: tf.Tensor, b0: tf.Tensor,
                       W1: tf.Tensor, b1: tf.Tensor,
                       W0_pol: tf.Tensor | None = None,
                       b0_pol: tf.Tensor | None = None,
                       W1_pol: tf.Tensor | None = None,
                       b1_pol: tf.Tensor | None = None,
                       W_atom: tf.Tensor | None = None) -> tf.Tensor:
        """Batched forward pass for B structures using COO gradient storage.

        Signature preserved from the NN version so all call sites
        (`score`, `spectroscopy.predict_trajectory_batch`,
        `_get_fused_predict`) continue to work without edits. The
        positional NN-weight args (`W0..b1_pol, W_atom`) are
        **deliberately unused** — GAP reads its state from `self.*`.

        Returns:
            predictions : [B, T_dim]
        """
        # Phase 1: target_mode == 2 raised in __init__, so we only
        # handle modes 0 and 1 here.
        if self.cfg.target_mode == 0:
            U = self._kernel_U(descriptors, Z, atom_mask)   # [B, A]
            E = tf.reduce_sum(U, axis=1, keepdims=True)     # [B, 1]
            return -E

        # Mode 1: dipole via gradient pathway.
        de_dq = self._kernel_grad_de_dq(descriptors, Z, atom_mask)  # [B, A, Q]
        B = tf.shape(descriptors)[0]
        box_inv = tf.linalg.inv(boxes)

        if W_atom is not None:
            # Precomputed-kernel path: dipole[b,s] = -Σ_{a,q} de_dq · W_atom
            # The NN's W_atom encoded `Σ_p rij²[p] · grad_values[p,s,q]`
            # aggregated to [B, A, 3, Q]. Reused verbatim here.
            return -tf.einsum('baq,basq->bs', de_dq, W_atom)

        # Standard COO path with pair-axis chunking via tf.while_loop.
        # Peak transient is `[P, Q]` (gather_nd output, ~2.4 GB at
        # P=2.97M, Q=216, fp32) plus `[P, 3]` forces. Without chunking
        # this can OOM at score time even when the struct-axis chunk is
        # bounded by `cfg.batch_chunk_size`. Chunking along the pair
        # axis bounds it to `[P_sub, Q]` regardless. tf.while_loop is
        # required because `predict_batch` runs inside @tf.function.
        P_total = tf.shape(grad_values)[0]
        P_sub = tf.constant(
            int(getattr(self.cfg, "gap_pair_chunk_size", 100_000)),
            dtype=tf.int32)
        _, rij2 = self._neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)

        def cond(p_lo, _dipole):
            return p_lo < P_total

        def body(p_lo, dipole_acc):
            p_hi = tf.minimum(p_lo + P_sub, P_total)
            ps_sub = pair_struct[p_lo:p_hi]
            pa_sub = pair_atom[p_lo:p_hi]
            ba_sub = tf.stack([ps_sub, pa_sub], axis=1)
            de_dq_pair = tf.gather_nd(de_dq, ba_sub)            # [p_sub, Q]
            gv_sub = grad_values[p_lo:p_hi]                      # [p_sub, 3, Q]
            forces = tf.einsum('kq,kcq->kc', de_dq_pair, gv_sub) # [p_sub, 3]
            contrib = -rij2[p_lo:p_hi, tf.newaxis] * forces
            dipole_acc = dipole_acc + tf.math.unsorted_segment_sum(
                contrib, ps_sub, num_segments=B)
            return p_hi, dipole_acc

        dipole_init = tf.zeros([B, 3], dtype=tf.float32)
        _, dipole = tf.while_loop(
            cond, body,
            loop_vars=[tf.constant(0, dtype=tf.int32), dipole_init],
            parallel_iterations=1,
            shape_invariants=[
                tf.TensorShape([]),
                tf.TensorShape([None, 3]),
            ],
        )
        return dipole

    # ───────────────────────────────────── COO helpers

    def _neighbor_displacements_coo(self, positions: tf.Tensor, boxes: tf.Tensor,
                                     box_inv: tf.Tensor, pair_struct: tf.Tensor,
                                     pair_atom: tf.Tensor,
                                     pair_gidx: tf.Tensor) -> tuple:
        """COO per-pair displacements with minimum-image convention."""
        ba_c = tf.stack([pair_struct, pair_atom], axis=1)
        ba_n = tf.stack([pair_struct, pair_gidx], axis=1)
        pos_c = tf.gather_nd(positions, ba_c)
        pos_n = tf.gather_nd(positions, ba_n)
        box_k = tf.gather(boxes, pair_struct)
        binv_k = tf.gather(box_inv, pair_struct)
        s_c = tf.einsum('kij,kj->ki', binv_k, pos_c)
        s_n = tf.einsum('kij,kj->ki', binv_k, pos_n)
        ds = s_n - s_c
        ds = ds - tf.round(ds)
        dr = tf.einsum('kij,kj->ki', box_k, ds)
        rij2 = tf.reduce_sum(tf.square(dr), axis=-1)
        return dr, rij2
