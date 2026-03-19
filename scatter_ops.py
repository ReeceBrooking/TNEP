from __future__ import annotations

import torch


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """Sum src elements by index groups along dim. Replaces torch_scatter dependency.

    Args:
        src      : source tensor
        index    : [N] int64 — group assignment for each element along dim
        dim_size : number of groups (output size along dim)
        dim      : dimension to scatter along (default 0)

    Returns:
        out : tensor with shape[dim] = dim_size, other dims same as src
    """
    if dim < 0:
        dim = src.dim() + dim
    idx = index
    for _ in range(dim):
        idx = idx.unsqueeze(0)
    for _ in range(src.dim() - dim - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(src)
    shape = list(src.shape)
    shape[dim] = dim_size
    out = src.new_zeros(shape)
    return out.scatter_reduce(dim, idx, src, reduce='sum')


def edge_displacements(
    positions: torch.Tensor,
    boxes: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimum image convention displacements for edges only — O(E) not O(N²).

    Inverts box matrices per-structure (S inversions) then gathers,
    rather than per-edge (E inversions).

    Args:
        positions  : [total_atoms, 3] atom positions
        boxes      : [S, 3, 3] lattice vectors per structure
        edge_src   : [E] global index of center atom
        edge_dst   : [E] global index of neighbor atom
        edge_batch : [E] which structure each edge belongs to

    Returns:
        dr  : [E, 3] Cartesian displacement vectors (dst - src, nearest image)
        rij : [E] scalar distances
    """
    Ri = positions[edge_src]                    # [E, 3]
    Rj = positions[edge_dst]                    # [E, 3]
    boxes_inv = torch.linalg.inv(boxes)         # [S, 3, 3] — S inversions, not E
    box = boxes[edge_batch]                     # [E, 3, 3]
    box_inv = boxes_inv[edge_batch]             # [E, 3, 3]
    si = torch.einsum('eij,ej->ei', box_inv, Ri)
    sj = torch.einsum('eij,ej->ei', box_inv, Rj)
    ds = sj - si
    ds = ds - torch.round(ds)                  # MIC wrap to [-0.5, 0.5)
    dr = torch.einsum('eij,ej->ei', box, ds)   # back to Cartesian
    rij = torch.linalg.norm(dr, dim=-1)         # [E]
    return dr, rij
