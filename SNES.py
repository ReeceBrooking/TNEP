"""SNES — GAP closed-form fit shim.

This module provides a thin fitter exposing the `model.optimizer.fit(...)`
interface used by `MasterTNEP._train_model_inner`. Internally there is
no iteration — the GAP α coefficients are obtained via a single
closed-form linear solve:

1. Selects M sparse points (per-species FPS or random) from the
   training-set atomic descriptors.
2. Converts training targets from per-atom-centered space back to
   total-dipole un-centered space.
3. Builds the design tensor Φ ∈ R^{(3·S) × M} via a chunked loop over
   structures, contracting kernel gradients with the existing COO
   `grad_values` pair tensor via `r²·grad_values`-weighted segment sums.
4. Solves the regularised linear system
       (λ I + Φᵀ Σ⁻¹ Φ) α = Φᵀ Σ⁻¹ y
   via QR on the augmented matrix `[Σ⁻¹/² Φ ; √λ I]` in float64.
5. Assigns α (and sparse_q, sparse_q_norm, sparse_species) to the model.

The class is still called `SNES` because callers (TNEP.fit, model.optimizer)
expect that attribute name.
"""
from __future__ import annotations

import math
import sys
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def _rrmse_from_r2(r2: tf.Tensor | float) -> float:
    """RRMSE = √(1 − R²). Clamps R² < 0 to 1 (worst case)."""
    r2_val = float(r2.numpy()) if hasattr(r2, "numpy") else float(r2)
    return float(math.sqrt(max(0.0, 1.0 - r2_val)))


class SNES:
    """GAP closed-form fitter, presenting the .fit(...) interface
    expected by TNEP.fit / MasterTNEP.

    All work happens inside `.fit()`; construction just stashes a
    reference to the model.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.cfg = model.cfg

    # ───────────────────────────────────────── public fit entry

    def fit(self, train_data: dict, val_data: dict) -> tuple:
        """Closed-form GAP fit. Returns (history, final_model, best_val_model).
        final_model and best_val_model are the same object.
        """
        cfg = self.cfg
        model = self.model
        t_start = time.perf_counter()
        print(f"  GAP fit: M={cfg.gap_n_sparse}, ζ={cfg.gap_zeta}, "
              f"sparse_method={cfg.gap_sparse_method}, "
              f"prior_cov={cfg.gap_use_prior_covariance}")

        # 1. Select sparse points (host-side numpy).
        t0 = time.perf_counter()
        sparse_q, sparse_q_norm, sparse_species = self._select_sparse(
            train_data)
        M_actual = int(sparse_q.shape[0])
        if M_actual < int(cfg.gap_n_sparse):
            print(f"  GAP fit: requested {cfg.gap_n_sparse} sparse points but "
                  f"only {M_actual} unique after dedup — using {M_actual}.")
        sparse_q_padded, sparse_qnorm_padded, sparse_species_padded = \
            self._pad_to_model_M(model.M, sparse_q, sparse_q_norm, sparse_species)
        print(f"  [1/5] sparse selection ({cfg.gap_sparse_method}): "
              f"{time.perf_counter() - t0:.1f}s")

        # 2. Convert per-atom-centered targets back to total un-centered.
        y_total = self._restore_total_targets(train_data)   # [S, T_dim]
        T_dim = int(y_total.shape[1])
        y_flat = tf.reshape(y_total, [-1])                  # [S·T_dim]

        # 3. Build Φ via chunked iteration. Assigns sparse points to
        # model first so kernel_grad_de_dq uses them in the build.
        self._assign_sparse_state(
            sparse_q_padded, sparse_qnorm_padded, sparse_species_padded)
        t0 = time.perf_counter()
        Phi = self._build_phi(train_data, T_dim)            # [S·T_dim, M_model]
        print(f"  [2/5] Φ build: {time.perf_counter() - t0:.1f}s "
              f"(shape {tuple(int(x) for x in Phi.shape)})")

        # 4. Per-row noise Σ⁻¹/² (with optional structure-size weight).
        sigma_inv_half = self._sigma_inv_half(train_data, y_total, T_dim)

        # 5. fp64 augmented QR solve.
        t0 = time.perf_counter()
        alpha = self._solve_qr(Phi, y_flat, sigma_inv_half)  # [M_model]
        print(f"  [3/5] fp64 QR solve: {time.perf_counter() - t0:.1f}s")

        # 6. Assign α to model.
        model.alpha.assign(alpha)

        # 7. Train / val metrics via model.score().
        t0 = time.perf_counter()
        print(f"  [4/5] scoring train set...")
        train_metrics, _ = model.score(train_data)
        print(f"        train RMSE = {float(train_metrics['rmse'].numpy()):.4e}  "
              f"R² = {float(train_metrics['r2'].numpy()):.4f}  "
              f"({time.perf_counter() - t0:.1f}s)")
        t0 = time.perf_counter()
        print(f"  [5/5] scoring val set...")
        val_metrics, _ = model.score(val_data)
        print(f"        val RMSE   = {float(val_metrics['rmse'].numpy()):.4e}  "
              f"R² = {float(val_metrics['r2'].numpy()):.4f}  "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"  Total fit time: {time.perf_counter() - t_start:.1f}s")
        history = self._make_history(train_metrics, val_metrics)
        return history, model, model

    # ───────────────────────────────────────── sparse selection

    def _select_sparse(self, train_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select M sparse atomic environments from the training set.

        Returns:
            sparse_q          [M, Q]   L2-normalised (q̂_s)
            sparse_q_norm     [M]      ||q_s|| (pre-normalised magnitude)
            sparse_species    [M] int  species index per sparse point
        """
        cfg = self.cfg
        # Gather all real (un-padded) atomic descriptors + species.
        descriptors = train_data["descriptors"].numpy() if hasattr(
            train_data["descriptors"], "numpy") else np.asarray(train_data["descriptors"])
        Z_int = train_data["Z_int"].numpy() if hasattr(
            train_data["Z_int"], "numpy") else np.asarray(train_data["Z_int"])
        atom_mask = train_data["atom_mask"].numpy() if hasattr(
            train_data["atom_mask"], "numpy") else np.asarray(train_data["atom_mask"])

        S, A_max, Q = descriptors.shape
        flat_q = descriptors.reshape(-1, Q)            # [S·A_max, Q]
        flat_Z = Z_int.reshape(-1)                      # [S·A_max]
        flat_mask = atom_mask.reshape(-1).astype(bool)  # [S·A_max]

        real_q = flat_q[flat_mask]
        real_Z = flat_Z[flat_mask]

        M_target = int(cfg.gap_n_sparse)
        method = str(cfg.gap_sparse_method).lower()
        per_species = bool(cfg.gap_per_species_sparse)
        seed = getattr(cfg, "seed", 0)
        rng = np.random.default_rng(int(seed) if seed is not None else 0)

        # Normalise descriptors for FPS / kernel — sparse points stored
        # as q̂_s, magnitudes preserved separately.
        norms = np.linalg.norm(real_q, axis=1)
        norms_safe = np.maximum(norms, 1e-12)
        real_q_hat = real_q / norms_safe[:, None]

        selected_idx_per_species: dict[int, np.ndarray] = {}

        if per_species:
            unique_species = np.unique(real_Z)
            # Allocate M_target ∝ per-species count, with min 1 each
            counts = {int(t): int(np.sum(real_Z == t)) for t in unique_species}
            total = sum(counts.values())
            M_per_species = {}
            allocated = 0
            for t in unique_species:
                t = int(t)
                share = max(1, int(round(M_target * counts[t] / total)))
                share = min(share, counts[t])
                M_per_species[t] = share
                allocated += share
            # Reconcile rounding: top up or trim to hit exactly M_target.
            if allocated > M_target:
                # Trim from the largest species first.
                sorted_types = sorted(M_per_species, key=lambda t: -M_per_species[t])
                excess = allocated - M_target
                for t in sorted_types:
                    if excess <= 0:
                        break
                    trim = min(M_per_species[t] - 1, excess)
                    M_per_species[t] -= trim
                    excess -= trim
            elif allocated < M_target:
                # Distribute remainder to species with most headroom.
                sorted_types = sorted(M_per_species,
                                       key=lambda t: counts[t] - M_per_species[t],
                                       reverse=True)
                short = M_target - allocated
                for t in sorted_types:
                    if short <= 0:
                        break
                    bump = min(counts[t] - M_per_species[t], short)
                    M_per_species[t] += bump
                    short -= bump
            for t in unique_species:
                t = int(t)
                idx_t = np.where(real_Z == t)[0]
                pick = self._select_indices(
                    real_q_hat[idx_t], M_per_species[t], method, rng)
                selected_idx_per_species[t] = idx_t[pick]
        else:
            pick = self._select_indices(real_q_hat, M_target, method, rng)
            for t in np.unique(real_Z[pick]):
                t = int(t)
                mask = real_Z[pick] == t
                selected_idx_per_species[t] = pick[mask]

        # Concatenate per-species selections.
        all_idx = np.concatenate(list(selected_idx_per_species.values()))
        # Dedup: drop near-duplicates within tol.
        sparse_q_hat = real_q_hat[all_idx]
        sparse_q_norm = norms[all_idx]
        sparse_species = real_Z[all_idx].astype(np.int32)
        keep = self._dedup(sparse_q_hat, float(cfg.gap_dedup_tol))
        return (sparse_q_hat[keep].astype(np.float32),
                sparse_q_norm[keep].astype(np.float32),
                sparse_species[keep])

    def _select_indices(self, q_hat: np.ndarray, n: int,
                         method: str, rng: np.random.Generator) -> np.ndarray:
        """Return indices into q_hat of length ≤ n via FPS or random."""
        N = q_hat.shape[0]
        n = min(n, N)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if method == "random" or N <= n:
            return rng.choice(N, size=n, replace=False)
        # Farthest-point sampling on cosine distance (1 − q̂·q̂').
        # Greedy O(N·n): maintain min-distance-to-selected, pick argmax.
        chosen = np.empty(n, dtype=np.int64)
        chosen[0] = rng.integers(N)
        # Cosine distance can dip slightly negative due to fp32 noise
        # on near-identical normalised vectors; clamp to ≥0 so argmax
        # picks a genuinely distant point.
        min_d = np.maximum(0.0, 1.0 - q_hat @ q_hat[chosen[0]])    # [N]
        for k in range(1, n):
            chosen[k] = int(np.argmax(min_d))
            new_d = np.maximum(0.0, 1.0 - q_hat @ q_hat[chosen[k]])
            min_d = np.minimum(min_d, new_d)
        return chosen

    def _dedup(self, q_hat: np.ndarray, tol: float) -> np.ndarray:
        """Greedy dedup by Euclidean distance on normalised descriptors."""
        if tol <= 0 or q_hat.shape[0] < 2:
            return np.arange(q_hat.shape[0])
        keep = [0]
        for i in range(1, q_hat.shape[0]):
            d = np.linalg.norm(q_hat[i] - q_hat[keep], axis=1)
            if d.min() > tol:
                keep.append(i)
        return np.asarray(keep)

    def _pad_to_model_M(self, M_model: int,
                         sparse_q: np.ndarray,
                         sparse_q_norm: np.ndarray,
                         sparse_species: np.ndarray) -> tuple:
        """Pad/truncate the selected sparse arrays to `M_model` (the
        shape pre-allocated in model.__init__). Padding rows use an
        invalid species index (−1) so the kernel species-mask zeroes
        them out.
        """
        M_sel = sparse_q.shape[0]
        Q = sparse_q.shape[1]
        if M_sel == M_model:
            return sparse_q, sparse_q_norm, sparse_species
        if M_sel > M_model:
            return sparse_q[:M_model], sparse_q_norm[:M_model], sparse_species[:M_model]
        # M_sel < M_model: pad with invalid-species rows.
        pad_n = M_model - M_sel
        q_pad = np.zeros((pad_n, Q), dtype=np.float32)
        n_pad = np.ones(pad_n, dtype=np.float32)
        s_pad = np.full(pad_n, -1, dtype=np.int32)
        return (np.concatenate([sparse_q, q_pad], axis=0),
                np.concatenate([sparse_q_norm, n_pad], axis=0),
                np.concatenate([sparse_species, s_pad], axis=0))

    def _assign_sparse_state(self, sparse_q, sparse_q_norm, sparse_species):
        """Push the chosen sparse-point arrays onto the model."""
        self.model.sparse_q.assign(tf.constant(sparse_q, dtype=tf.float32))
        self.model.sparse_q_norm.assign(tf.constant(sparse_q_norm, dtype=tf.float32))
        self.model.sparse_species.assign(tf.constant(sparse_species, dtype=tf.int32))

    # ───────────────────────────────────────── target conversion

    def _restore_total_targets(self, train_data) -> tf.Tensor:
        """Convert train_data["targets"] back to TOTAL un-centered space.

        Three branches based on (target_mode, scale_targets):
          1. mode=1 AND scale_targets=True:
                 stored = total/N − mean
                 total  = (stored + mean) · N_k
          2. mode=1 AND scale_targets=False:
                 stored = total − mean
                 total  = stored + mean
          3. mode=0 (energy) or mode=2 (polar.; deferred):
                 stored = total − mean (no scale_targets path)
                 total  = stored + mean

        Returns a float32 tensor of shape [S, T_dim].
        """
        cfg = self.cfg
        targets = train_data["targets"]
        # Tensor casts to ensure operations are on tf tensors.
        targets = tf.convert_to_tensor(targets, dtype=tf.float32)
        if targets.shape.ndims == 1:
            targets = tf.expand_dims(targets, axis=-1)
        mean_per_atom = getattr(cfg, "_target_mean", None)
        if (bool(getattr(cfg, "target_centering", False))
                and mean_per_atom is not None):
            mean_tf = tf.constant(
                np.asarray(mean_per_atom, dtype=np.float32).reshape(1, -1))
            out = targets + mean_tf
        else:
            out = targets
        if cfg.target_mode == 1 and bool(getattr(cfg, "scale_targets", False)):
            N = tf.cast(train_data["num_atoms"], tf.float32)[:, None]
            out = out * N
        return out

    # ───────────────────────────────────────── Φ build

    def _build_phi(self, train_data: dict, T_dim: int) -> tf.Tensor:
        """Build the design tensor Φ ∈ R^{(S·T_dim) × M_model}.

        Direct construction: rather than M separate predict_batch calls
        (one per one-hot α), produces all M columns in O(M / M_sub)
        sparse-block sweeps. Each sweep:
          1. Compute the un-contracted kernel-gradient
             `de_dq_per_s [B, A, M_sub, Q]` for the sparse block.
          2. For target_mode=1: contract with `grad_values` via COO to
             get per-pair forces `[P, 3, M_sub]`, then segment-sum the
             `-r²·forces` to dipoles `[B, 3, M_sub]`.
          3. Reshape into `[B·3, M_sub]` rows of Φ.
        For target_mode=0: directly sum `K_iS [B, A, M_sub]` to
        per-structure energy `[B, M_sub]` and flip sign.

        Cost: O(M·B·A·Q) overall, comparable to one full forward pass.
        Far below the O(M)·full-forward of the naive one-hot loop.
        """
        from data import prefetched_chunks
        cfg = self.cfg
        model = self.model
        M = int(model.M)
        S = int(train_data["num_atoms"].shape[0])
        M_sub = max(1, int(getattr(cfg, "gap_sparse_chunk_size", 16)))
        # Structure chunk size for Φ build is independent of
        # cfg.batch_chunk_size: the pair-gather `[P_chunk, M_sub, Q]`
        # is the dominant memory term and needs aggressive chunking
        # even when score-time chunks are unbounded.
        gap_chunk = getattr(cfg, "gap_struct_chunk_size", None)
        if gap_chunk is None:
            chunk_sz = min(S, 100)
        else:
            chunk_sz = min(S, int(gap_chunk))

        # Pre-allocate Φ rows by structure-chunk × T_dim:
        #   For each structure chunk we produce a [B_c, T_dim, M] slice
        #   built up across M_sub blocks. Concatenate over chunks.
        ranges = [(s, min(s + chunk_sz, S)) for s in range(0, S, chunk_sz)]
        Phi_struct_blocks: list = []   # each [B_c · T_dim, M]

        # Progress bar over (struct_chunks × M_sub blocks). Each unit is
        # one (B_c × M_sub × Q) gradient compute, the dominant cost.
        n_struct_chunks = len(ranges)
        n_sparse_blocks = (M + M_sub - 1) // M_sub
        total_units = n_struct_chunks * n_sparse_blocks
        # file=sys.stdout aligns with the rest of the GAP-fit prints so
        # the bar isn't interleaved with TF/CUDA stderr noise.
        # dynamic_ncols redraws on terminal resize. ascii=False uses
        # solid block glyphs; mininterval=0.1 keeps the ETA fresh.
        pbar = tqdm(total=total_units, desc="  Φ build", unit="block",
                    leave=False, mininterval=0.1,
                    file=sys.stdout, dynamic_ncols=True)

        for s_start, s_end, chunk in prefetched_chunks(
                train_data, ranges,
                pin_to_cpu=cfg.pin_data_to_cpu,
                enabled=getattr(cfg, "chunk_prefetch", True),
                depth=getattr(cfg, "prefetch_depth", 1)):
            B_c = int(chunk["num_atoms"].shape[0])

            # Build per-sparse-block columns for this structure chunk.
            phi_slabs: list = []
            # Hoist m_lo-invariant pieces out of the sparse-block loop:
            # box_inv and rij2 depend only on the chunk geometry, not the
            # sparse block, so recomputing them per m_lo wastes work.
            if cfg.target_mode != 0:
                pair_atom = chunk["pair_atom"]
                pair_struct = chunk["pair_struct"]
                pair_gidx = chunk["pair_gidx"]
                grad_values = chunk["grad_values"]            # [P, 3, Q]
                box_inv = tf.linalg.inv(chunk["boxes"])
                _, rij2 = model._neighbor_displacements_coo(
                    chunk["positions"], chunk["boxes"], box_inv,
                    pair_struct, pair_atom, pair_gidx)         # [P]
                P_total = int(tf.shape(grad_values)[0].numpy())
                P_sub = max(1, int(getattr(cfg, "gap_pair_chunk_size", 100_000)))
            for m_lo in range(0, M, M_sub):
                m_hi = min(m_lo + M_sub, M)
                bs = m_hi - m_lo
                if cfg.target_mode == 0:
                    # Φ column for energy mode:
                    #   predict_batch returns -E = -Σ_i U_i for one-hot α at s.
                    #   With α = e_s, U_i = δ²_t(s) · cos_is^ζ · mask, so
                    #   col[k, s] = -Σ_i [ δ²_t(s) · cos_is^ζ · mask ].
                    # We compute K_is_block directly and sum.
                    q = chunk["descriptors"]
                    q_norm = tf.maximum(tf.linalg.norm(q, axis=-1), 1e-12)
                    q_hat = q / q_norm[..., tf.newaxis]
                    sq_block = model.sparse_q[m_lo:m_hi]
                    cos_is = tf.einsum('baq,sq->bas', q_hat, sq_block)
                    Z_b = chunk["Z_int"][..., tf.newaxis]
                    sp = model.sparse_species[m_lo:m_hi][tf.newaxis, tf.newaxis, :]
                    species_mask = tf.cast(tf.equal(Z_b, sp), tf.float32)
                    cos_is = cos_is * species_mask
                    delta_sq_block = tf.square(
                        tf.gather(model.delta_per_species,
                                  tf.maximum(model.sparse_species[m_lo:m_hi], 0)))
                    K_is = (cos_is ** int(cfg.gap_zeta)) * \
                        delta_sq_block[tf.newaxis, tf.newaxis, :]
                    K_is = K_is * chunk["atom_mask"][..., tf.newaxis]
                    # Per-structure E = Σ_i K_is, then predict_batch sign = -E.
                    col = -tf.reduce_sum(K_is, axis=1)  # [B_c, M_sub]
                    # Treat as T_dim=1: reshape to [B_c, 1, M_sub] then flatten.
                    col = col[:, tf.newaxis, :]         # [B_c, 1, M_sub]
                else:
                    # target_mode == 1: dipole. Need the full
                    # gradient-pathway contraction.
                    de_dq_per_s = model._kernel_grad_per_sparse_block(
                        chunk["descriptors"], chunk["Z_int"],
                        chunk["atom_mask"], m_lo, m_hi)  # [B_c, A, bs, Q]

                    # Nested pair-axis chunking. Peak transient memory
                    # `[P_sub, bs, Q]` is the dominant term — bounded
                    # by P_sub. Accumulate dipole_block via
                    # repeated unsorted_segment_sum.
                    dipole_block = tf.zeros([B_c, 3, bs], dtype=tf.float32)
                    for p_lo in range(0, P_total, P_sub):
                        p_hi = min(p_lo + P_sub, P_total)
                        ba_sub = tf.stack([pair_struct[p_lo:p_hi],
                                            pair_atom[p_lo:p_hi]], axis=1)
                        de_dq_pair = tf.gather_nd(de_dq_per_s, ba_sub)
                        # [p_sub, bs, Q]
                        gv_sub = grad_values[p_lo:p_hi]               # [p_sub, 3, Q]
                        forces = tf.einsum('psq,pcq->pcs', de_dq_pair, gv_sub)
                        # [p_sub, 3, bs]
                        contrib = -rij2[p_lo:p_hi, tf.newaxis, tf.newaxis] * forces
                        dipole_block = dipole_block + tf.math.unsorted_segment_sum(
                            contrib, pair_struct[p_lo:p_hi], num_segments=B_c)
                        # Drop pair-chunk transients before the next p_lo
                        # iteration allocates fresh ones. Without these,
                        # the new allocation briefly overlaps the old.
                        del ba_sub, de_dq_pair, gv_sub, forces, contrib
                    col = dipole_block                  # [B_c, 3, M_sub]
                    # Free the [B_c, A, bs, Q] gradient before the next
                    # m_lo iteration calls _kernel_grad_per_sparse_block
                    # again — otherwise the new alloc overlaps the old.
                    del de_dq_per_s
                phi_slabs.append(col)
                pbar.update(1)
            # Concatenate sparse-block slabs along M axis.
            phi_chunk = tf.concat(phi_slabs, axis=-1)   # [B_c, T_dim, M]
            # Reshape to [B_c · T_dim, M] row layout.
            Phi_struct_blocks.append(
                tf.reshape(phi_chunk, [-1, M]))
            del chunk
        pbar.close()
        Phi = tf.concat(Phi_struct_blocks, axis=0)      # [S · T_dim, M]
        return Phi

    # ───────────────────────────────────────── Σ (row noise) construction

    def _sigma_inv_half(self, train_data, y_total, T_dim) -> tf.Tensor:
        """Per-row Σ⁻¹/² for the augmented-QR solve.

        Phase 1: per-species σ via a single global heuristic plus
        optional structure-size weighting (σ_eff = σ · √N_k).
        """
        cfg = self.cfg
        S = int(y_total.shape[0])
        # Global heuristic σ if not user-set.
        if cfg.gap_sigma_E is not None:
            sigma = float(cfg.gap_sigma_E)
        else:
            y_var = float(tf.math.reduce_variance(y_total).numpy())
            sigma = max(1e-6, 0.1 * math.sqrt(y_var))
            print(f"  GAP fit: σ_E heuristic = {sigma:.6e} "
                  f"(0.1·√var(y_total) = 0.1·{math.sqrt(y_var):.4e})")

        # σ_eff per structure = σ · √N_k  (optional structure-size weight)
        if bool(getattr(cfg, "gap_structure_size_weight", True)):
            N = tf.cast(train_data["num_atoms"], tf.float32)        # [S]
            sigma_per_struct = sigma * tf.sqrt(N)                    # [S]
        else:
            sigma_per_struct = tf.fill([S], sigma)                   # [S]
        # Broadcast to per-row (every T_dim component shares the
        # structure's σ).
        sigma_per_row = tf.repeat(sigma_per_struct, repeats=T_dim)   # [S·T_dim]
        return 1.0 / sigma_per_row    # Σ⁻¹/² = 1 / σ (already a sqrt-form)

    # ───────────────────────────────────────── linear solve

    def _solve_qr(self, Phi: tf.Tensor, y_flat: tf.Tensor,
                   sigma_inv_half: tf.Tensor) -> tf.Tensor:
        """Solve (λ I + Φᵀ Σ⁻¹ Φ) α = Φᵀ Σ⁻¹ y via augmented QR in fp64.

        Augmented system:
            [ Σ⁻¹/² · Φ ]      [ Σ⁻¹/² · y ]
            [  √λ · I    ]  α = [    0       ]

        QR-decompose the (R + M) × M matrix A, then α = R⁻¹ · Qᵀ · b.

        Phase 1 always uses ridge form (`gap_use_prior_covariance=False`).
        λ is set from cfg.gap_sigma_E or the heuristic computed in
        _sigma_inv_half (we re-use σ_E as the per-row σ — λ controls the
        prior strength independently and defaults to a small value).
        """
        cfg = self.cfg
        M = int(Phi.shape[1])

        # Weighted Φ rows
        Phi64 = tf.cast(Phi, tf.float64) * tf.cast(
            sigma_inv_half[:, None], tf.float64)
        y64 = tf.cast(y_flat, tf.float64) * tf.cast(sigma_inv_half, tf.float64)

        # Ridge regulariser √λ. **Decoupled** from σ_E (the per-row
        # data fidelity in Σ⁻¹/²) — they control different things:
        #   σ_E : per-row noise variance, data-fidelity weight
        #   λ   : prior strength on α (Phase 1 ridge form)
        lam = float(getattr(cfg, "gap_ridge_lambda", 1e-6))
        sqrt_lam = math.sqrt(max(lam, 1e-12))
        I_aug = sqrt_lam * tf.eye(M, dtype=tf.float64)

        A = tf.concat([Phi64, I_aug], axis=0)                 # [(R + M), M]
        b = tf.concat([y64, tf.zeros([M], dtype=tf.float64)], axis=0)

        # tf.linalg.qr returns Q with shape [..., m, n] for "reduced"
        # QR, R with shape [..., n, n].
        Q, R = tf.linalg.qr(A, full_matrices=False)
        QTb = tf.linalg.matvec(Q, b, transpose_a=True)
        alpha64 = tf.linalg.triangular_solve(R, QTb[:, None], lower=False)[:, 0]
        return tf.cast(alpha64, tf.float32)

    # ───────────────────────────────────────── history dict

    def _make_history(self, train_metrics: dict, val_metrics: dict) -> dict:
        """Single-row history dict for the GAP closed-form fit."""
        train_rmse = float(train_metrics["rmse"].numpy())
        val_rmse = float(val_metrics["rmse"].numpy())
        rrmse_train = _rrmse_from_r2(train_metrics["r2"])
        return {
            "generation":   [0],
            "train_loss":   [train_rmse],
            "val_loss":     [val_rmse],
            "best_rmse":    [train_rmse],
            "worst_rmse":   [train_rmse],
            "best_rrmse":   [rrmse_train],
            "avg_rrmse":    [rrmse_train],
        }
