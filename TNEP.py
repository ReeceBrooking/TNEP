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

        Args:
            batch : dict from collate_flat() with keys:
                descriptors [total_atoms, Q], Z_int [total_atoms],
                positions [total_atoms, 3], atom_batch [total_atoms],
                gradients [total_edges, 3, Q], edge_src [total_edges],
                edge_dst [total_edges], edge_batch [total_edges],
                targets [S, T_dim], boxes [S, 3, 3], num_atoms [S]
            W0..b1 : weight tensors [T,Q,H], [T,H], [T,H], scalar
            W0_pol..b1_pol : mode 2 only

        Returns:
            predictions : [S, T_dim]
        """
        desc = batch["descriptors"]       # [total_atoms, Q]
        Z = batch["Z_int"]                # [total_atoms]
        atom_batch = batch["atom_batch"]   # [total_atoms]
        grads = batch["gradients"]         # [total_edges, 3, Q]
        edge_src = batch["edge_src"]       # [total_edges]
        S = int(batch["num_atoms"].shape[0])

        # Scale descriptors and gradients
        if self._scale_descriptors:
            scale = self._descriptor_mean  # [Q]
            desc = desc / scale
            grads = grads / scale

        # Gather per-type weights for each atom
        W0_t = W0[Z]     # [total_atoms, Q, H]
        b0_t = b0[Z]     # [total_atoms, H]
        W1_t = W1[Z]     # [total_atoms, H]

        # Hidden layer
        h = torch.einsum('nq,nqh->nh', desc, W0_t) + b0_t  # [total_atoms, H]
        h = torch.tanh(h)

        if self.cfg.target_mode == 0:
            # PES: E = -sum_i (h_i . W1[t_i] + b1)
            E_per_atom = torch.sum(h * W1_t, dim=1) + b1    # [total_atoms]
            E = scatter_sum(E_per_atom, atom_batch, dim_size=S)  # [S]
            return -E.unsqueeze(1)  # [S, 1]

        # Modes 1 and 2 need forces
        forces = self._calc_forces_flat(h, grads, W1_t, W0_t, edge_src)  # [total_edges, 3]

        if self.cfg.target_mode == 1:
            return self._dipole_flat(forces, batch, S)

        elif self.cfg.target_mode == 2:
            return self._polarizability_flat(
                desc, forces, batch, S, Z, atom_batch,
                W0_pol, b0_pol, W1_pol, b1_pol)

    def _calc_forces_flat(
        self,
        h: torch.Tensor,
        gradients: torch.Tensor,
        W1_t: torch.Tensor,
        W0_t: torch.Tensor,
        edge_src: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dU_i/dR_j for every edge via chain rule.

        Args:
            h         : [total_atoms, H] hidden activations
            gradients : [total_edges, 3, Q] descriptor gradients
            W1_t      : [total_atoms, H] per-atom output weights
            W0_t      : [total_atoms, Q, H] per-atom input weights
            edge_src  : [total_edges] center atom index

        Returns:
            forces : [total_edges, 3]
        """
        dtanh = 1.0 - h ** 2                                    # [total_atoms, H]
        de_da = dtanh * W1_t                                     # [total_atoms, H]
        de_dq = torch.einsum('nh,nqh->nq', de_da, W0_t)        # [total_atoms, Q]
        de_dq_edge = de_dq[edge_src]                             # [total_edges, Q]
        forces = torch.einsum('eq,ecq->ec', de_dq_edge, gradients)  # [total_edges, 3]
        return forces

    def _dipole_flat(
        self,
        forces: torch.Tensor,
        batch: dict[str, torch.Tensor],
        S: int,
    ) -> torch.Tensor:
        """Dipole prediction from flat data.

        mu = -sum_edges |r_ij|^2 * force_ij

        Returns:
            dipole : [S, 3]
        """
        positions = batch["positions"]
        boxes = batch["boxes"]
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        edge_batch = batch["edge_batch"]

        dr, rij = edge_displacements(positions, boxes, edge_src, edge_dst, edge_batch)
        rij2 = rij ** 2                                          # [E]
        dipole_contribs = rij2.unsqueeze(1) * forces             # [E, 3]
        dipole = -scatter_sum(dipole_contribs, edge_batch, dim_size=S)  # [S, 3]
        return dipole

    def _polarizability_flat(
        self,
        desc: torch.Tensor,
        forces: torch.Tensor,
        batch: dict[str, torch.Tensor],
        S: int,
        Z: torch.Tensor,
        atom_batch: torch.Tensor,
        W0_pol: torch.Tensor,
        b0_pol: torch.Tensor,
        W1_pol: torch.Tensor,
        b1_pol: torch.Tensor,
    ) -> torch.Tensor:
        """Polarizability via dual ANN (GPUMD approach).

        Scalar ANN → isotropic diagonal.
        Tensor ANN (forces) → anisotropic virial.

        Returns:
            pol : [S, 6] — [xx, yy, zz, xy, yz, zx]
        """
        positions = batch["positions"]
        boxes = batch["boxes"]
        edge_src = batch["edge_src"]
        edge_dst = batch["edge_dst"]
        edge_batch = batch["edge_batch"]

        dr, rij = edge_displacements(positions, boxes, edge_src, edge_dst, edge_batch)

        # --- Scalar ANN (isotropic) ---
        W0p_t = W0_pol[Z]     # [total_atoms, Q, H]
        b0p_t = b0_pol[Z]     # [total_atoms, H]
        W1p_t = W1_pol[Z]     # [total_atoms, H]

        h_pol = torch.einsum('nq,nqh->nh', desc, W0p_t) + b0p_t
        h_pol = torch.tanh(h_pol)
        F_pol = torch.sum(h_pol * W1p_t, dim=1) + b1_pol        # [total_atoms]
        scalar_sum = scatter_sum(F_pol, atom_batch, dim_size=S)  # [S]

        # --- Tensor ANN (anisotropic virial) ---
        # pol_outer = -dr_a * force_b for each edge → [E, 3, 3]
        pol_outer = -torch.einsum('ea,eb->eab', dr, forces)      # [E, 3, 3]
        pol_matrix = scatter_sum(pol_outer, edge_batch, dim_size=S)  # [S, 3, 3]

        # Extract 6 unique components: [xx, yy, zz, xy, yz, zx]
        pol = torch.stack([
            pol_matrix[:, 0, 0],
            pol_matrix[:, 1, 1],
            pol_matrix[:, 2, 2],
            pol_matrix[:, 0, 1],
            pol_matrix[:, 1, 2],
            pol_matrix[:, 2, 0],
        ], dim=1)  # [S, 6]

        # Add scalar to diagonal
        pol[:, 0] += scalar_sum
        pol[:, 1] += scalar_sum
        pol[:, 2] += scalar_sum
        return pol

    def fit(self, train_data: dict[str, torch.Tensor], val_data: dict[str, torch.Tensor],
            plot_callback: Callable | None = None) -> dict:
        """Train the model using the SNES evolutionary optimizer.

        Args:
            train_data    : dict from collate_flat()
            val_data      : same structure
            plot_callback : optional callable(history, gen) for periodic plotting

        Returns:
            history : dict with keys generation, train_loss, val_loss (lists)
        """
        from SNES import SNES
        if self.optimizer is None:
            self.optimizer = SNES(self)
        history = self.optimizer.fit(train_data, val_data, plot_callback=plot_callback)
        return history

    def score(self, test_data: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Evaluate RMSE, R², per-component R², and cosine similarity.

        Args:
            test_data : dict from collate_flat()

        Returns:
            metrics : dict with numpy values
            preds   : [S, T] numpy array
        """
        with torch.no_grad():
            raw_preds = self.predict_flat(
                test_data, self.W0, self.b0, self.W1, self.b1,
                getattr(self, 'W0_pol', None),
                getattr(self, 'b0_pol', None),
                getattr(self, 'W1_pol', None),
                getattr(self, 'b1_pol', None),
            )
        targets = test_data["targets"]

        # Per-atom normalization
        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            num_atoms = test_data["num_atoms"].float()         # [S]
            num_atoms_col = num_atoms.clamp(min=1.0).unsqueeze(1)  # [S, 1]
            preds = raw_preds / num_atoms_col
        else:
            preds = raw_preds

        diff = preds - targets
        mse = torch.mean(diff ** 2)
        rmse = torch.sqrt(mse)

        # Overall R²
        ss_res = torch.sum(diff ** 2)
        ss_tot = torch.sum((targets - targets.mean(dim=0)) ** 2)
        r2 = 1.0 - ss_res / ss_tot

        # Per-component R²
        ss_res_comp = torch.sum(diff ** 2, dim=0)
        ss_tot_comp = torch.sum((targets - targets.mean(dim=0)) ** 2, dim=0)
        r2_components = 1.0 - ss_res_comp / ss_tot_comp.clamp(min=1e-12)

        metrics: dict[str, np.ndarray] = {
            "rmse": float(rmse),
            "r2": float(r2),
            "r2_components": r2_components.cpu().numpy(),
        }

        # Total (un-scaled) metrics when target scaling is active
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

        # Cosine similarity for vector targets
        if self.cfg.target_mode >= 1:
            dot = torch.sum(preds * targets, dim=1)
            norm_p = torch.linalg.norm(preds, dim=1)
            norm_t = torch.linalg.norm(targets, dim=1)
            cos_sim = dot / (norm_p * norm_t).clamp(min=1e-12)
            metrics["cos_sim_mean"] = float(cos_sim.mean())
            metrics["cos_sim_all"] = cos_sim.cpu().numpy()

        return metrics, preds.cpu().numpy()
