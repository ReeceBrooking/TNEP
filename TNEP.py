from __future__ import annotations

import numpy as np
import torch
from typing import Callable, TYPE_CHECKING

from scatter_ops import scatter_sum, edge_displacements
from TNEPconfig import TNEPconfig

if TYPE_CHECKING:
    from SNES import SNES


class TNEP:
    """Per-type single-hidden-layer ANN for predicting energy, dipole, or polarizability.

    Forward pass per atom i with type t:
        a_i  = q_i @ W0[t] + b0[t]           # [num_neurons]
        h_i  = tanh(a_i)                      # [num_neurons]
        U_i  = h_i . W1[t] + b1              # scalar

    Weights are plain torch.Tensors (not nn.Parameters) because SNES manages
    them as flat vectors — no autograd needed.

    Prediction modes (cfg.target_mode):
        0 (PES)    : E = -sum_i U_i                                   -> [S, 1]
        1 (Dipole) : mu = -sum_edges |r_ij|^2 * (dU_i/dr_ij_vec)     -> [S, 3]
        2 (Polar.) : alpha[6] via dual ANN (scalar + tensor)          -> [S, 6]
    """

    def __init__(self, cfg: TNEPconfig, device: torch.device | None = None) -> None:
        self.cfg = cfg
        self.dim_q = cfg.dim_q
        self.num_types = cfg.num_types
        self.num_neurons = cfg.num_neurons
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Descriptor scaling
        if hasattr(cfg, 'descriptor_mean') and cfg.descriptor_mean is not None:
            self._descriptor_mean = torch.tensor(
                cfg.descriptor_mean, dtype=torch.float32, device=self.device)
            self._scale_descriptors = True
        else:
            self._scale_descriptors = False

        # Weights — initialised by SNES, stored here for score()/predict
        T, Q, H = cfg.num_types, cfg.dim_q, cfg.num_neurons
        self.W0 = torch.zeros(T, Q, H, dtype=torch.float32, device=self.device)
        self.b0 = torch.zeros(T, H, dtype=torch.float32, device=self.device)
        self.W1 = torch.zeros(T, H, dtype=torch.float32, device=self.device)
        self.b1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if cfg.target_mode == 2:
            self.W0_pol = torch.zeros(T, Q, H, dtype=torch.float32, device=self.device)
            self.b0_pol = torch.zeros(T, H, dtype=torch.float32, device=self.device)
            self.W1_pol = torch.zeros(T, H, dtype=torch.float32, device=self.device)
            self.b1_pol = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.optimizer: SNES | None = None

        # Compiled padded forward pass — built on first use
        self._compiled_padded_fwd: Callable | None = None

    # ------------------------------------------------------------------
    # Flat forward pass (used for validation, scoring, spectroscopy)
    # ------------------------------------------------------------------

    def predict_flat(
        self,
        batch: dict[str, torch.Tensor],
        W0: torch.Tensor,
        b0: torch.Tensor,
        W1: torch.Tensor,
        b1: torch.Tensor,
        W0_pol: torch.Tensor | None = None,
        b0_pol: torch.Tensor | None = None,
        W1_pol: torch.Tensor | None = None,
        b1_pol: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass on flat (concatenated) batch data.

        Returns:
            predictions : [S, T_dim]
        """
        if "desc_scaled" in batch:
            desc = batch["desc_scaled"]
            grads = batch["grads_scaled"]
        else:
            desc = batch["descriptors"]
            grads = batch["gradients"]
            if self._scale_descriptors:
                scale = self._descriptor_mean
                desc = desc / scale
                grads = grads / scale

        Z = batch["Z_int"]
        atom_batch = batch["atom_batch"]
        edge_src = batch["edge_src"]
        S = int(batch["num_atoms"].shape[0])

        # Gather per-type weights
        W0_t = W0[Z]     # [N, Q, H]
        b0_t = b0[Z]     # [N, H]
        W1_t = W1[Z]     # [N, H]

        # Hidden layer
        h = torch.einsum('nq,nqh->nh', desc, W0_t) + b0_t
        h = torch.tanh(h)

        if self.cfg.target_mode == 0:
            E_per_atom = (h * W1_t).sum(dim=1) + b1
            E = scatter_sum(E_per_atom, atom_batch, dim_size=S)
            return -E.unsqueeze(1)

        # Forces
        dtanh = 1.0 - h ** 2
        de_da = dtanh * W1_t
        de_dq = torch.einsum('nh,nqh->nq', de_da, W0_t)
        de_dq_edge = de_dq[edge_src]
        forces = torch.einsum('eq,ecq->ec', de_dq_edge, grads)

        if self.cfg.target_mode == 1:
            rij2 = batch.get("rij2")
            if rij2 is None:
                _, rij = edge_displacements(
                    batch["positions"], batch["boxes"],
                    batch["edge_src"], batch["edge_dst"], batch["edge_batch"])
                rij2 = rij ** 2
            contribs = rij2.unsqueeze(1) * forces
            return -scatter_sum(contribs, batch["edge_batch"], dim_size=S)

        elif self.cfg.target_mode == 2:
            edge_batch = batch["edge_batch"]
            dr = batch.get("dr")
            if dr is None:
                dr, _ = edge_displacements(
                    batch["positions"], batch["boxes"],
                    batch["edge_src"], batch["edge_dst"], edge_batch)
            # Scalar ANN
            W0p_t = W0_pol[Z]
            b0p_t = b0_pol[Z]
            W1p_t = W1_pol[Z]
            h_pol = torch.einsum('nq,nqh->nh', desc, W0p_t) + b0p_t
            h_pol = torch.tanh(h_pol)
            F_pol = (h_pol * W1p_t).sum(dim=1) + b1_pol
            scalar_sum = scatter_sum(F_pol, atom_batch, dim_size=S)
            # Tensor ANN
            pol_outer = -(dr.unsqueeze(2) * forces.unsqueeze(1))
            pol_matrix = scatter_sum(pol_outer, edge_batch, dim_size=S)
            pol = torch.stack([
                pol_matrix[:, 0, 0], pol_matrix[:, 1, 1], pol_matrix[:, 2, 2],
                pol_matrix[:, 0, 1], pol_matrix[:, 1, 2], pol_matrix[:, 2, 0],
            ], dim=1)
            pol[:, 0] += scalar_sum
            pol[:, 1] += scalar_sum
            pol[:, 2] += scalar_sum
            return pol

    # ------------------------------------------------------------------
    # Padded forward pass (used for SNES population evaluation)
    # ------------------------------------------------------------------

    @staticmethod
    def pad_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert a flat batch into padded tensors with static shapes.

        Fully vectorized — no Python loops over atoms or edges.
        This enables torch.compile(fullgraph=True) and efficient batched matmul
        over the population dimension — the same approach that made TF fast.
        """
        S = int(batch["num_atoms"].shape[0])
        desc = batch.get("desc_scaled", batch["descriptors"])
        grads = batch.get("grads_scaled", batch["gradients"])
        Z = batch["Z_int"]
        atom_batch_idx = batch["atom_batch"]
        edge_src = batch["edge_src"]
        Q = desc.shape[1]
        N = desc.shape[0]
        E = edge_src.shape[0]
        device = desc.device

        atom_offsets = batch["atom_offsets"]
        num_atoms = batch["num_atoms"]
        A_max = int(num_atoms.max())

        # Compute local atom index within each structure: local_idx = global - offset[structure]
        atom_local = torch.arange(N, device=device) - atom_offsets[atom_batch_idx]

        # Padded descriptors [S, A_max, Q] and types [S, A_max]
        desc_pad = desc.new_zeros(S, A_max, Q)
        desc_pad[atom_batch_idx, atom_local] = desc
        Z_pad = Z.new_zeros(S, A_max)
        Z_pad[atom_batch_idx, atom_local] = Z
        atom_mask = torch.zeros(S, A_max, dtype=torch.bool, device=device)
        atom_mask[atom_batch_idx, atom_local] = True

        # Count edges per global atom to find M_max and compute local edge index
        edges_per_atom = torch.zeros(N, dtype=torch.int64, device=device)
        edges_per_atom.scatter_add_(0, edge_src, torch.ones(E, dtype=torch.int64, device=device))
        M_max = int(edges_per_atom.max())

        # Local neighbor index within each atom's edge list
        # For each edge, its position among edges sharing the same src atom
        edge_order = torch.zeros(E, dtype=torch.int64, device=device)
        # Cumulative count per src atom via sorting
        sorted_idx = torch.argsort(edge_src, stable=True)
        counts = edges_per_atom[edge_src[sorted_idx]]  # not needed directly
        # Compute position within each group: for edges sorted by src, the position
        # is the index minus the first occurrence of that src
        src_sorted = edge_src[sorted_idx]
        # Use cumsum trick: mark group starts, then cumsum within groups
        group_start = torch.ones(E, dtype=torch.int64, device=device)
        if E > 0:
            same_as_prev = (src_sorted[1:] == src_sorted[:-1])
            group_start[1:] = (~same_as_prev).long()
        cum_pos = torch.cumsum(torch.ones(E, dtype=torch.int64, device=device), dim=0)
        group_cum = torch.cumsum(group_start, dim=0)
        # First edge in each group gets position 0
        # group_offset[i] = cum_pos of the first edge in this group
        group_first_pos = torch.zeros(E, dtype=torch.int64, device=device)
        group_first_pos[0] = 0
        if E > 1:
            new_group = (group_start[1:] == 1).nonzero(as_tuple=True)[0] + 1
            # Simpler approach: scatter to get start position per group
            # and gather back
        # Actually, simplest correct approach for local edge index:
        inv_sorted = torch.empty_like(sorted_idx)
        inv_sorted[sorted_idx] = torch.arange(E, device=device)
        # edges_before[i] = number of edges with same src that come before i in sorted order
        # = position_in_sorted - first_position_of_this_src_in_sorted
        first_pos = torch.zeros(N, dtype=torch.int64, device=device)
        # For each atom, first_pos = min index in sorted_idx where src == atom
        # edges_per_atom is known, so prefix sum gives first_pos
        first_pos[1:] = torch.cumsum(edges_per_atom[:-1], dim=0)
        edge_local = inv_sorted - first_pos[edge_src]

        # Map each edge to (structure, local_atom, local_neighbor)
        edge_struct = batch["edge_batch"]
        edge_src_local = edge_src - atom_offsets[edge_struct]

        # Padded gradients [S, A_max, M_max, 3, Q]
        grads_pad = grads.new_zeros(S, A_max, M_max, 3, Q)
        grads_pad[edge_struct, edge_src_local, edge_local] = grads
        edge_mask = torch.zeros(S, A_max, M_max, dtype=torch.bool, device=device)
        edge_mask[edge_struct, edge_src_local, edge_local] = True

        # rij2 padded [S, A_max, M_max] for dipole mode
        rij2_pad = None
        if "rij2" in batch:
            rij2_pad = batch["rij2"].new_zeros(S, A_max, M_max)
            rij2_pad[edge_struct, edge_src_local, edge_local] = batch["rij2"]

        out = dict(batch)
        out["desc_pad"] = desc_pad
        out["Z_pad"] = Z_pad
        out["atom_mask"] = atom_mask
        out["grads_pad"] = grads_pad
        out["edge_mask"] = edge_mask
        out["A_max"] = A_max
        out["M_max"] = M_max
        if rij2_pad is not None:
            out["rij2_pad"] = rij2_pad
        return out

    def predict_padded_batched(
        self,
        batch: dict[str, torch.Tensor],
        W0: torch.Tensor,
        b0: torch.Tensor,
        W1: torch.Tensor,
        b1: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate P candidates on padded data using dense matmul.

        All shapes are static → torch.compile can fuse everything.
        Currently supports modes 0 (PES) and 1 (dipole).

        Args:
            W0 [P, T, Q, H], b0 [P, T, H], W1 [P, T, H], b1 [P]

        Returns:
            predictions : [P, S, T_dim]
        """
        desc_pad = batch["desc_pad"]          # [S, A, Q]
        Z_pad = batch["Z_pad"]                # [S, A]
        atom_mask = batch["atom_mask"]         # [S, A]
        grads_pad = batch["grads_pad"]         # [S, A, M, 3, Q]
        edge_mask = batch["edge_mask"]         # [S, A, M]

        if self._compiled_padded_fwd is None:
            self._compiled_padded_fwd = torch.compile(
                self._padded_forward_impl,
                fullgraph=True, mode="max-autotune")

        rij2_pad = batch.get("rij2_pad")
        if rij2_pad is None:
            rij2_pad = torch.zeros(1, device=self.device)  # dummy

        return self._compiled_padded_fwd(
            desc_pad, Z_pad, atom_mask, grads_pad, edge_mask,
            rij2_pad, W0, b0, W1, b1)

    def _padded_forward_impl(
        self,
        desc_pad: torch.Tensor,     # [S, A, Q]
        Z_pad: torch.Tensor,        # [S, A]
        atom_mask: torch.Tensor,     # [S, A]
        grads_pad: torch.Tensor,     # [S, A, M, 3, Q]
        edge_mask: torch.Tensor,     # [S, A, M]
        rij2_pad: torch.Tensor,      # [S, A, M] or dummy
        W0: torch.Tensor,            # [P, T, Q, H]
        b0: torch.Tensor,            # [P, T, H]
        W1: torch.Tensor,            # [P, T, H]
        b1: torch.Tensor,            # [P]
    ) -> torch.Tensor:
        """Padded dense forward pass — pure tensor function for torch.compile.

        No scatter, no gather-by-index. All ops are regular dense matmul/einsum
        on static-shape tensors. Masks applied after each step.
        """
        P = W0.shape[0]
        S, A, Q = desc_pad.shape
        H = W0.shape[3]
        target_mode = self.cfg.target_mode

        # Gather per-type weights for each atom position: [P, S, A, ...]
        # W0[p, Z[s,a]] -> [P, S, A, Q, H]
        W0_t = W0[:, Z_pad]       # [P, S, A, Q, H]
        b0_t = b0[:, Z_pad]       # [P, S, A, H]
        W1_t = W1[:, Z_pad]       # [P, S, A, H]

        # Hidden layer: [S, A, Q] @ [P, S, A, Q, H] -> [P, S, A, H]
        h = torch.einsum('saq,psaqh->psah', desc_pad, W0_t) + b0_t
        h = torch.tanh(h)

        if target_mode == 0:
            # PES: sum per-atom energies masked
            E_per_atom = (h * W1_t).sum(dim=3) + b1[:, None, None]  # [P, S, A]
            E_per_atom = E_per_atom * atom_mask.float()              # mask padding
            E = E_per_atom.sum(dim=2)                                # [P, S]
            return -E.unsqueeze(2)                                   # [P, S, 1]

        # Forces via chain rule
        dtanh = 1.0 - h ** 2                                        # [P, S, A, H]
        de_da = dtanh * W1_t                                         # [P, S, A, H]
        # de_dq: [P, S, A, H] @ [P, S, A, H, Q] -> [P, S, A, Q]
        de_dq = torch.einsum('psah,psaqh->psaq', de_da, W0_t)

        # Forces per edge: de_dq[center] . grads[center, nbr]
        # de_dq is [P, S, A, Q], grads_pad is [S, A, M, 3, Q]
        # For each (s, a, m): force = de_dq[p, s, a, :] @ grads[s, a, m, :, :]^T
        # = einsum over Q: [P, S, A, Q] * [S, A, M, 3, Q] -> [P, S, A, M, 3]
        forces = torch.einsum('psaq,samcq->psamc', de_dq, grads_pad)

        if target_mode == 1:
            # Dipole: -sum_edges rij^2 * force, masked
            # rij2_pad [S, A, M], forces [P, S, A, M, 3]
            contribs = rij2_pad[None, :, :, :, None] * forces        # [P, S, A, M, 3]
            contribs = contribs * edge_mask[None, :, :, :, None].float()
            dipole = -contribs.sum(dim=(2, 3))                       # [P, S, 3]
            return dipole

        # Mode 2 not yet implemented in padded path — fall back handled in _evaluate_chunk

    # ------------------------------------------------------------------
    # Training and scoring
    # ------------------------------------------------------------------

    def fit(self, train_data: dict[str, torch.Tensor], val_data: dict[str, torch.Tensor],
            plot_callback: Callable | None = None) -> dict:
        """Train the model using the SNES evolutionary optimizer."""
        from SNES import SNES
        if self.optimizer is None:
            self.optimizer = SNES(self)
        history = self.optimizer.fit(train_data, val_data, plot_callback=plot_callback)
        return history

    def score(self, test_data: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Evaluate RMSE, R², per-component R², and cosine similarity."""
        with torch.no_grad():
            raw_preds = self.predict_flat(
                test_data, self.W0, self.b0, self.W1, self.b1,
                getattr(self, 'W0_pol', None),
                getattr(self, 'b0_pol', None),
                getattr(self, 'W1_pol', None),
                getattr(self, 'b1_pol', None),
            )
        targets = test_data["targets"]

        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            num_atoms = test_data["num_atoms"].float()
            num_atoms_col = num_atoms.clamp(min=1.0).unsqueeze(1)
            preds = raw_preds / num_atoms_col
        else:
            preds = raw_preds

        diff = preds - targets
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)

        ss_res = torch.sum(diff ** 2)
        ss_tot = torch.sum((targets - targets.mean(dim=0)) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        ss_res_comp = torch.sum(diff ** 2, dim=0)
        ss_tot_comp = torch.sum((targets - targets.mean(dim=0)) ** 2, dim=0)
        r2_components = 1.0 - ss_res_comp / ss_tot_comp.clamp(min=1e-12)

        metrics: dict[str, np.ndarray] = {
            "rmse": float(rmse),
            "r2": float(r2),
            "r2_components": r2_components.cpu().numpy(),
        }

        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            total_targets = targets * num_atoms_col
            total_diff = raw_preds - total_targets
            total_rmse = torch.sqrt(torch.mean(total_diff ** 2))
            total_ss_res = torch.sum(total_diff ** 2)
            total_ss_tot = torch.sum((total_targets - total_targets.mean(dim=0)) ** 2)
            total_r2 = 1.0 - total_ss_res / total_ss_tot
            total_ss_res_comp = torch.sum(total_diff ** 2, dim=0)
            total_ss_tot_comp = torch.sum((total_targets - total_targets.mean(dim=0)) ** 2, dim=0)
            total_r2_comp = 1.0 - total_ss_res_comp / total_ss_tot_comp.clamp(min=1e-12)
            metrics["total_rmse"] = float(total_rmse)
            metrics["total_r2"] = float(total_r2)
            metrics["total_r2_components"] = total_r2_comp.cpu().numpy()

        if self.cfg.target_mode >= 1:
            dot = torch.sum(preds * targets, dim=1)
            norm_p = torch.linalg.norm(preds, dim=1)
            norm_t = torch.linalg.norm(targets, dim=1)
            cos_sim = dot / (norm_p * norm_t).clamp(min=1e-12)
            metrics["cos_sim_mean"] = float(cos_sim.mean())
            metrics["cos_sim_all"] = cos_sim.cpu().numpy()

        return metrics, preds.cpu().numpy()
