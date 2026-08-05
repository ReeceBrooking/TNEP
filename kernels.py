"""Candidate-independent geometry kernels for the tensorial (dipole /
polarizability) prediction paths.

These reduce the per-pair descriptor gradients ``∂q_i/∂R_j`` into a per-atom
kernel that carries everything the forward pass can ever ask of them:

    dipole    W[b,a,s,q] = Σ_{p: struct=b, atom=a} |r_ij|^N · grad_values[p,s,q]
    pol       W[b,a,s,q] = Σ_{p: struct=b, atom=a}  dr[p,i_s] · grad_values[p,j_s,q]

    prediction[b,s] = −Σ_{a,q} de_dq[b,a,q] · W[b,a,s,q]

``de_dq`` carries the model parameters; ``W`` is pure geometry. The reduction
sums over each atom's neighbours, shrinking ``[M, 3, Q]`` per atom to
``[3|6, Q]`` — a factor of M for the dipole and M/2 for the polarizability. It
is one-way: the gradients cannot be recovered, but nothing downstream needs
them once ``W`` exists, so the staging path builds and discards them in
batches rather than holding the whole set resident.

Lives outside TNEP so `data.py` can reduce during descriptor staging, before
any model object exists. `TNEP` delegates to these; there is no second copy.
"""
from __future__ import annotations

import tensorflow as tf

# Component order of the [.., 6] polarizability vector as (row, col) of the
# rank-2 tensor: xx, yy, zz, xy, yz, zx. Must match data._extract_target's
# 9→6 flattening (raw[[0, 4, 8, 1, 5, 6]]).
POL_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))


def neighbor_displacements_coo(positions: tf.Tensor, boxes: tf.Tensor,
                               box_inv: tf.Tensor, pair_struct: tf.Tensor,
                               pair_atom: tf.Tensor,
                               pair_gidx: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Per-pair displacements in COO format with minimum-image wrapping.

    Args:
        positions  : [B, A, 3]
        boxes      : [B, 3, 3]
        box_inv    : [B, 3, 3]
        pair_struct: [P]  batch-relative structure index
        pair_atom  : [P]  centre atom index
        pair_gidx  : [P]  neighbour atom index

    Returns:
        dr   : [P, 3]  displacement vectors (neighbour − centre)
        rij2 : [P]     squared distances
    """
    ba_c = tf.stack([pair_struct, pair_atom], axis=1)   # [P, 2]
    ba_n = tf.stack([pair_struct, pair_gidx], axis=1)   # [P, 2]
    pos_c   = tf.gather_nd(positions, ba_c)              # [P, 3]
    pos_n   = tf.gather_nd(positions, ba_n)              # [P, 3]
    box_k   = tf.gather(boxes,   pair_struct)            # [P, 3, 3]
    binv_k  = tf.gather(box_inv, pair_struct)            # [P, 3, 3]
    # ASE row-vector convention: s = r @ inv(cell) → contract the first (row)
    # index of box_inv (='kji'). 'kij' transposes the fractional basis and
    # gives wrong MIC wrapping for triclinic cells.
    s_c = tf.einsum('kji,kj->ki', binv_k, pos_c)        # [P, 3] fractional
    s_n = tf.einsum('kji,kj->ki', binv_k, pos_n)
    ds  = s_n - s_c
    ds  = ds - tf.round(ds)                              # MIC wrap
    dr  = tf.einsum('kji,kj->ki', box_k, ds)            # [P, 3] Cartesian
    rij2 = tf.reduce_sum(tf.square(dr), axis=-1)         # [P]
    return dr, rij2


def scalar_rij_pow(N: int, rij2: tf.Tensor) -> tf.Tensor:
    """|r_ij|^N from the rij² primitive (N ≥ 1; avoids sqrt for even N).
    N=0 (self-pairs only) is handled by pair_weight_coo.
    """
    if N == 1:
        return tf.sqrt(rij2)
    if N == 2:
        return rij2
    if N % 2 == 0:
        return tf.pow(rij2, N // 2)
    return tf.pow(rij2, (N - 1) // 2) * tf.sqrt(rij2)


def pair_weight_coo(N: int, rij2: tf.Tensor, pair_atom: tf.Tensor,
                    pair_gidx: tf.Tensor) -> tf.Tensor:
    """Per-pair dipole weight.

        N == 0 : 1 where pair_atom==pair_gidx AND rij²<1e-20 (a true self
                 pair). The rij² guard rejects periodic self-images (same atom
                 index, nonzero displacement) that would otherwise double-count.
        N >= 1 : |r_ij|^N (self pairs → 0, images weighted correctly).
    """
    if N == 0:
        is_self = tf.logical_and(tf.equal(pair_atom, pair_gidx), rij2 < 1e-20)
        return tf.cast(is_self, rij2.dtype)
    return scalar_rij_pow(N, rij2)


def precompute_dipole_kernel(N: int, grad_values: tf.Tensor,
                             pair_struct: tf.Tensor, pair_atom: tf.Tensor,
                             pair_gidx: tf.Tensor, positions: tf.Tensor,
                             boxes: tf.Tensor, B, A) -> tf.Tensor:
    """W[b,a,s,q] = Σ_{p: struct=b, atom=a} |r_ij|^N · grad_values[p,s,q].

    N == 0: weight is 1 everywhere because the COO list is already filtered to
    self pairs upstream (pad_and_stack(self_pairs_only=True)), so displacements
    and tf.linalg.inv(boxes) are skipped entirely. Passing an unfiltered list
    at N == 0 silently gives the wrong answer.

    Returns:
        [B, A, 3, Q]
    """
    P = tf.shape(grad_values)[0]
    Q = tf.shape(grad_values)[2]
    if N == 0:
        weight = tf.ones([P], dtype=grad_values.dtype)
    else:
        box_inv = tf.linalg.inv(boxes)
        _, rij2 = neighbor_displacements_coo(
            positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)
        weight = pair_weight_coo(N, rij2, pair_atom, pair_gidx)   # [P]
    ba_linear = pair_struct * A + pair_atom               # [P] index into [B*A]
    # One Cartesian component at a time: peak intermediate is [P, Q] rather
    # than [P, 3, Q]. P is the full pair count at N >= 1, so the 3x form is the
    # single largest transient in staging and would dominate the batch budget.
    comps = [
        tf.math.unsorted_segment_sum(
            weight[:, tf.newaxis] * grad_values[:, s, :],  # [P, Q]
            ba_linear, num_segments=B * A)                 # [B*A, Q]
        for s in range(3)
    ]
    return tf.reshape(tf.stack(comps, axis=1), [B, A, 3, Q])


def precompute_pol_kernel(grad_values: tf.Tensor, pair_struct: tf.Tensor,
                          pair_atom: tf.Tensor, pair_gidx: tf.Tensor,
                          positions: tf.Tensor, boxes: tf.Tensor,
                          B, A) -> tf.Tensor:
    """W[b,a,s,q] = Σ_{p: struct=b, atom=a} dr[p,i_s]·grad_values[p,j_s,q],
    with (i_s, j_s) = POL_COMPONENTS.

    Algebraically the same as the COO path (outer product per pair, then
    segment-sum), but with no P axis left for the candidate loop to multiply.

    Returns:
        [B, A, 6, Q]
    """
    Q = tf.shape(grad_values)[2]
    box_inv = tf.linalg.inv(boxes)
    dr, _ = neighbor_displacements_coo(
        positions, boxes, box_inv, pair_struct, pair_atom, pair_gidx)  # [P, 3]
    ba_linear = pair_struct * A + pair_atom               # [P] index into [B*A]
    # One component at a time: peak intermediate is [P, Q] rather than
    # [P, 6, Q]. Mode 2 keeps the full (non-self-filtered) pair list, so P is
    # ~10x the dipole case and the 6x form would dominate memory here.
    comps = [
        tf.math.unsorted_segment_sum(
            dr[:, i:i + 1] * grad_values[:, j, :],        # [P, Q]
            ba_linear, num_segments=B * A)                # [B*A, Q]
        for i, j in POL_COMPONENTS
    ]
    return tf.reshape(tf.stack(comps, axis=1), [B, A, 6, Q])


def kernel_components(cfg) -> int | None:
    """Trailing component count of the kernel for cfg's target mode:
    3 (dipole), 6 (polarizability), or None when the mode has no kernel (PES,
    which needs true per-pair forces and cannot be reduced).
    """
    return {1: 3, 2: 6}.get(int(cfg.target_mode))


def precompute_kernel(cfg, grad_values: tf.Tensor, pair_struct: tf.Tensor,
                      pair_atom: tf.Tensor, pair_gidx: tf.Tensor,
                      positions: tf.Tensor, boxes: tf.Tensor,
                      B, A) -> tf.Tensor:
    """Dispatch to the dipole or polarizability kernel for cfg's target mode."""
    mode = int(cfg.target_mode)
    if mode == 1:
        N = int(getattr(cfg, "dipole_rij_power", 2))
        return precompute_dipole_kernel(N, grad_values, pair_struct, pair_atom,
                                        pair_gidx, positions, boxes, B, A)
    if mode == 2:
        return precompute_pol_kernel(grad_values, pair_struct, pair_atom,
                                     pair_gidx, positions, boxes, B, A)
    raise ValueError(f"target_mode={mode} has no geometry kernel "
                     f"(PES needs per-pair forces).")
