from __future__ import annotations

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from typing import Callable

from DescriptorBuilder import DescriptorBuilder
from SNES import SNES
from TNEPconfig import TNEPconfig

class TNEP(layers.Layer):
    """Per-type single-hidden-layer ANN for predicting energy, dipole, or polarizability.

    Forward pass per atom i with type t:
        a_i  = q_i @ W0[t] + b0[t]           # [num_neurons]
        h_i  = tanh(a_i)                      # [num_neurons]
        U_i  = h_i · W1[t] + b1              # scalar

    Weights:
        W0 : [num_types, dim_q, num_neurons]  input -> hidden
        b0 : [num_types, num_neurons]         hidden bias
        W1 : [num_types, num_neurons]         hidden -> scalar
        b1 : ()                               global scalar bias

    Prediction modes (cfg.target_mode):
        0 (PES)    : E = -sum_i U_i                                   -> scalar
        1 (Dipole) : μ = -sum_i sum_j |r_ij|² * (dU_i/dr_ij_vec)     -> [3]
        2 (Polar.) : α[6] via dual ANN (scalar + tensor)              -> [6]

    For mode 2 (polarizability), a second "scalar ANN" is added:
        W0_pol, b0_pol, W1_pol, b1_pol
    The scalar ANN computes per-atom F_pol -> isotropic diagonal.
    The tensor ANN (primary W0/b0/W1/b1) computes forces -> anisotropic virial.
    Output: [xx, yy, zz, xy, yz, zx]
    """

    def __init__(self,
                 cfg: TNEPconfig,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg
        self.dim_q = cfg.dim_q
        self.num_types = cfg.num_types
        self.num_neurons = cfg.num_neurons
        self.activation = tf.keras.activations.get(cfg.activation)
        self.builder = DescriptorBuilder(cfg)
        self.optimizer = SNES(self)

        # W0 : [num_types, dim_q, num_neurons] — input-to-hidden weights per type
        self.W0 = self.add_weight(
            name="W0",
            shape=(cfg.num_types, cfg.dim_q, cfg.num_neurons),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b0 : [num_types, num_neurons] — hidden bias per type
        self.b0 = self.add_weight(
            name="b0",
            shape=(cfg.num_types, cfg.num_neurons),
            initializer="zeros",
            trainable=True,
        )

        # W1 : [num_types, num_neurons] — hidden-to-scalar weights per type
        self.W1 = self.add_weight(
            name="W1",
            shape=(cfg.num_types, cfg.num_neurons),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b1 : () — global scalar bias shared across all types
        self.b1 = self.add_weight(
            name="b1",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

        # Scalar ANN for polarizability mode (target_mode == 2)
        if cfg.target_mode == 2:
            self.W0_pol = self.add_weight(
                name="W0_pol",
                shape=(cfg.num_types, cfg.dim_q, cfg.num_neurons),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b0_pol = self.add_weight(
                name="b0_pol",
                shape=(cfg.num_types, cfg.num_neurons),
                initializer="zeros",
                trainable=True,
            )
            self.W1_pol = self.add_weight(
                name="W1_pol",
                shape=(cfg.num_types, cfg.num_neurons),
                initializer="glorot_uniform",
                trainable=True,
            )
            self.b1_pol = self.add_weight(
                name="b1_pol",
                shape=(),
                initializer="zeros",
                trainable=True,
            )

    def predict(self, descriptors: tf.Tensor, gradients: tf.Tensor, grad_index: tf.Tensor,
                positions: tf.Tensor, Z: tf.Tensor, box: tf.Tensor,
                atom_mask: tf.Tensor, neighbor_mask: tf.Tensor) -> tf.Tensor:
        """Run the forward pass for a single structure using padded tensors.

        Args:
            descriptors    : [A, dim_q]     padded per-atom SOAP descriptors
            gradients      : [A, M, 3, dim_q]  padded descriptor gradients
            grad_index     : [A, M]         padded neighbor indices
            positions      : [A, 3]         padded atom positions
            Z              : [A]            padded integer type indices
            box            : [3, 3]         lattice vectors
            atom_mask      : [A]            1.0 for real atoms, 0.0 for padding
            neighbor_mask  : [A, M]         1.0 for real neighbors, 0.0 for padding

        Returns:
            target_mode 0: [1]  total energy
            target_mode 1: [3]  dipole vector
            target_mode 2: [6]  polarizability tensor
        """
        # Gather per-type weights for each atom
        W0_t = tf.gather(self.W0, Z)   # [A, dim_q, H]
        b0_t = tf.gather(self.b0, Z)   # [A, H]
        W1_t = tf.gather(self.W1, Z)   # [A, H]

        # Hidden layer: h_i = activation(q_i @ W0[t_i] + b0[t_i])
        h = tf.einsum('nd,ndh->nh', descriptors, W0_t)  # [A, H]
        h = h + b0_t                                      # [A, H]
        h = self.activation(h)                             # [A, H]
        # Mask out padded atoms
        h = h * atom_mask[:, tf.newaxis]                   # [A, H]

        if self.cfg.target_mode == 0:
            # PES: E = -sum_i (h_i . W1[t_i] + b1)
            E_per_atom = tf.reduce_sum(h * W1_t, axis=1) + self.b1  # [A]
            E_per_atom = E_per_atom * atom_mask                       # zero padding
            E = tf.reduce_sum(E_per_atom)
            return tf.expand_dims(-E, axis=0)  # [1]

        # Modes 1 and 2 need forces
        forces = self.calc_forces(h, gradients, W1_t, W0_t, neighbor_mask)  # [A, M, 3]

        if self.cfg.target_mode == 1:
            # Dipole: μ = -sum_i sum_j |r_ij|^2 * force_ij
            # Gather displacement vectors for each (atom, neighbor) pair
            dr, rij = self.builder.pairwise_displacements(positions, box)
            # dr: [A, A, 3], rij: [A, A]

            # Gather displacements for grad_index neighbors
            A = tf.shape(Z)[0]
            M = tf.shape(grad_index)[1]
            atom_idx = tf.broadcast_to(tf.range(A)[:, tf.newaxis], [A, M])
            # indices: [A, M, 2] — pairs of (atom_i, neighbor_j)
            indices = tf.stack([atom_idx, grad_index], axis=-1)
            dr_gathered = tf.gather_nd(dr, indices)     # [A, M, 3]
            rij_gathered = tf.gather_nd(rij, indices)   # [A, M]

            rij2 = tf.square(rij_gathered)                          # [A, M]
            rij2 = rij2 * neighbor_mask                              # zero padding
            dipole_contribs = rij2[:, :, tf.newaxis] * forces        # [A, M, 3]
            dipole = -tf.reduce_sum(dipole_contribs, axis=[0, 1])   # [3]
            return dipole

        elif self.cfg.target_mode == 2:
            # Polarizability via dual ANN (GPUMD approach)
            dr, rij = self.builder.pairwise_displacements(positions, box)
            A = tf.shape(Z)[0]
            M = tf.shape(grad_index)[1]
            atom_idx = tf.broadcast_to(tf.range(A)[:, tf.newaxis], [A, M])
            indices = tf.stack([atom_idx, grad_index], axis=-1)
            dr_gathered = tf.gather_nd(dr, indices)  # [A, M, 3]

            # --- Scalar ANN (isotropic) ---
            W0p_t = tf.gather(self.W0_pol, Z)  # [A, dim_q, H]
            b0p_t = tf.gather(self.b0_pol, Z)  # [A, H]
            W1p_t = tf.gather(self.W1_pol, Z)  # [A, H]

            h_pol = tf.einsum('nd,ndh->nh', descriptors, W0p_t)  # [A, H]
            h_pol = h_pol + b0p_t
            h_pol = self.activation(h_pol)
            h_pol = h_pol * atom_mask[:, tf.newaxis]
            F_pol = tf.reduce_sum(h_pol * W1p_t, axis=1) + self.b1_pol  # [A]
            F_pol = F_pol * atom_mask
            scalar_sum = tf.reduce_sum(F_pol)

            # --- Tensor ANN (anisotropic virial) ---
            pol_outer = -tf.einsum('nma,nmb->nmab', dr_gathered, forces)  # [A, M, 3, 3]
            pol_outer = pol_outer * neighbor_mask[:, :, tf.newaxis, tf.newaxis]
            pol_matrix = tf.reduce_sum(pol_outer, axis=[0, 1])  # [3, 3]

            # Extract 6 unique components: [xx, yy, zz, xy, yz, zx]
            pol = tf.stack([
                pol_matrix[0, 0],
                pol_matrix[1, 1],
                pol_matrix[2, 2],
                pol_matrix[0, 1],
                pol_matrix[1, 2],
                pol_matrix[2, 0],
            ])

            # Add scalar ANN to diagonal
            pol = pol + tf.stack([scalar_sum, scalar_sum, scalar_sum,
                                  0.0, 0.0, 0.0])
            return pol

    def calc_forces(self, h: tf.Tensor, gradients: tf.Tensor, W1_t: tf.Tensor,
                    W0_t: tf.Tensor, neighbor_mask: tf.Tensor) -> tf.Tensor:
        """Compute dU_i/dR_j for every atom i and its neighbours j via chain rule.

        Vectorized version — no Python loops. Uses padded gradient tensors.

        Args:
            h              : [N, H]              hidden activations tanh(a)
            gradients      : [N, M, 3, dim_q]   padded descriptor gradients
            W1_t           : [N, H]              per-atom output weights
            W0_t           : [N, dim_q, H]       per-atom input weights
            neighbor_mask  : [N, M]              1.0 for real neighbors, 0.0 for padding

        Returns:
            forces : [N, M, 3]  dU_i/dR_j per atom per neighbor
        """
        # dU/dh * dh/da = W1 * (1 - tanh^2(a))
        dtanh = 1.0 - tf.square(h)                               # [N, H]
        de_da = dtanh * W1_t                                      # [N, H]
        # dU/dq = dU/da @ W0^T  ->  [N, dim_q]
        de_dq = tf.einsum('nh,nqh->nq', de_da, W0_t)             # [N, dim_q]
        # Contract dU/dq with dq/dR_j: sum over dim_q
        forces = tf.einsum('nq,nmcq->nmc', de_dq, gradients)     # [N, M, 3]
        # Zero out padded neighbors
        forces = forces * neighbor_mask[:, :, tf.newaxis]
        return forces

    def fit(self, train_data: dict[str, tf.Tensor], val_data: dict[str, tf.Tensor],
            plot_callback: Callable | None = None) -> dict:
        """Train the model using the SNES evolutionary optimizer.

        Args:
            train_data    : dict with keys descriptors, gradients, grad_index,
                            positions, Z_int, targets, boxes (lists over structures)
            val_data      : same structure, used for validation each generation
            plot_callback : optional callable(history, gen) for periodic plotting

        Returns:
            history : dict with keys generation, train_loss, val_loss (lists)
        """
        history = self.optimizer.fit(train_data, val_data, plot_callback=plot_callback)
        return history

    def score(self, test_data: dict[str, tf.Tensor]) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        """Evaluate RMSE, R², per-component R², and cosine similarity.

        Args:
            test_data : dict with padded tensors from pad_and_stack()

        Returns:
            metrics : dict with keys:
                rmse          : scalar float — overall RMSE
                r2            : scalar float — overall R²
                r2_components : [T] tensor — per-component R²
                cos_sim_mean  : scalar float — mean cosine similarity (modes 1,2 only)
                cos_sim_all   : [S] tensor — per-structure cosine similarity (modes 1,2)
            preds : [S, T] tensor of predictions
        """
        preds = self.predict_batch(
            test_data["descriptors"], test_data["gradients"],
            test_data["grad_index"], test_data["positions"],
            test_data["Z_int"], test_data["boxes"],
            test_data["atom_mask"], test_data["neighbor_mask"],
            self.W0, self.b0, self.W1, self.b1,
            getattr(self, 'W0_pol', None),
            getattr(self, 'b0_pol', None),
            getattr(self, 'W1_pol', None),
            getattr(self, 'b1_pol', None),
        )
        targets = test_data["targets"]
        diff = preds - targets
        mse = tf.reduce_mean(tf.square(diff))
        rmse = tf.sqrt(mse)

        # Overall R² = 1 - SS_res / SS_tot
        ss_res = tf.reduce_sum(tf.square(diff))
        ss_tot = tf.reduce_sum(tf.square(targets - tf.reduce_mean(targets, axis=0)))
        r2 = 1.0 - ss_res / ss_tot

        # Per-component R²
        ss_res_comp = tf.reduce_sum(tf.square(diff), axis=0)       # [T]
        ss_tot_comp = tf.reduce_sum(
            tf.square(targets - tf.reduce_mean(targets, axis=0)), axis=0)  # [T]
        r2_components = 1.0 - ss_res_comp / tf.maximum(ss_tot_comp, 1e-12)

        metrics = {
            "rmse": rmse,
            "r2": r2,
            "r2_components": r2_components,
        }

        # Cosine similarity for vector targets (modes 1 and 2)
        if self.cfg.target_mode >= 1:
            dot = tf.reduce_sum(preds * targets, axis=1)          # [S]
            norm_p = tf.linalg.norm(preds, axis=1)                # [S]
            norm_t = tf.linalg.norm(targets, axis=1)              # [S]
            cos_sim = dot / tf.maximum(norm_p * norm_t, 1e-12)    # [S]
            metrics["cos_sim_mean"] = tf.reduce_mean(cos_sim)
            metrics["cos_sim_all"] = cos_sim

        return metrics, preds

    @tf.function
    def predict_batch(self, descriptors: tf.Tensor, gradients: tf.Tensor,
                      grad_index: tf.Tensor, positions: tf.Tensor, Z: tf.Tensor,
                      boxes: tf.Tensor, atom_mask: tf.Tensor, neighbor_mask: tf.Tensor,
                      W0: tf.Tensor, b0: tf.Tensor, W1: tf.Tensor, b1: tf.Tensor,
                      W0_pol: tf.Tensor | None = None, b0_pol: tf.Tensor | None = None,
                      W1_pol: tf.Tensor | None = None, b1_pol: tf.Tensor | None = None) -> tf.Tensor:
        """Batched forward pass for B structures with explicit weight tensors.

        Weights are passed explicitly (not read from self) so this method can
        be used for SNES population evaluation with different candidate weights.

        Args:
            descriptors    : [B, A, Q]        padded descriptors
            gradients      : [B, A, M, 3, Q]  padded descriptor gradients
            grad_index     : [B, A, M]        padded neighbor indices
            positions      : [B, A, 3]        padded positions
            Z              : [B, A]           padded type indices
            boxes          : [B, 3, 3]        lattice vectors
            atom_mask      : [B, A]           atom mask
            neighbor_mask  : [B, A, M]        neighbor mask
            W0             : [T, Q, H]        input weights (shared across batch)
            b0             : [T, H]           hidden bias
            W1             : [T, H]           output weights
            b1             : ()               scalar bias
            W0_pol ... b1_pol : same shapes, for mode 2 only (None otherwise)

        Returns:
            predictions : [B, T_dim]  where T_dim = 1 (PES), 3 (dipole), 6 (pol)
        """
        # Gather per-type weights for each atom in each structure
        W0_t = tf.gather(W0, Z)   # [B, A, Q, H]
        b0_t = tf.gather(b0, Z)   # [B, A, H]
        W1_t = tf.gather(W1, Z)   # [B, A, H]

        # Hidden layer
        h = tf.einsum('bnd,bndh->bnh', descriptors, W0_t)  # [B, A, H]
        h = h + b0_t
        h = self.activation(h)
        h = h * atom_mask[:, :, tf.newaxis]

        if self.cfg.target_mode == 0:
            E = tf.reduce_sum(h * W1_t, axis=2) + b1  # [B, A]
            E = E * atom_mask
            E = tf.reduce_sum(E, axis=1, keepdims=True)  # [B, 1]
            return -E

        # Modes 1 and 2: compute forces
        forces = self._calc_forces_batch(h, gradients, W1_t, W0_t, neighbor_mask)

        if self.cfg.target_mode == 1:
            return self._dipole_batch(forces, positions, boxes, grad_index,
                                      atom_mask, neighbor_mask)

        elif self.cfg.target_mode == 2:
            return self._polarizability_batch(
                descriptors, forces, positions, boxes, Z, grad_index,
                atom_mask, neighbor_mask,
                W0_pol, b0_pol, W1_pol, b1_pol)

        else:
            tf.debugging.assert_equal(True, False, message="Unsupported target_mode")

    def _calc_forces_batch(self, h: tf.Tensor, gradients: tf.Tensor, W1_t: tf.Tensor,
                           W0_t: tf.Tensor, neighbor_mask: tf.Tensor) -> tf.Tensor:
        """Batched calc_forces: [B,A,H] inputs -> [B,A,M,3] forces."""
        dtanh = 1.0 - tf.square(h)                                    # [B, A, H]
        de_da = dtanh * W1_t                                           # [B, A, H]
        de_dq = tf.einsum('bnh,bnqh->bnq', de_da, W0_t)              # [B, A, Q]
        forces = tf.einsum('bnq,bnmcq->bnmc', de_dq, gradients)      # [B, A, M, 3]
        forces = forces * neighbor_mask[:, :, :, tf.newaxis]
        return forces

    def _pairwise_displacements_batch(self, positions: tf.Tensor,
                                      boxes: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Batched pairwise displacements under minimum image convention.

        Args:
            positions : [B, A, 3]
            boxes     : [B, 3, 3]

        Returns:
            dr  : [B, A, A, 3]  displacement vectors
            rij : [B, A, A]     scalar distances
        """
        box_inv = tf.linalg.inv(boxes)                            # [B, 3, 3]
        # Fractional coordinates
        s = tf.einsum('bij,bnj->bni', box_inv, positions)         # [B, A, 3]
        # Pairwise differences in fractional coords
        ds = s[:, tf.newaxis, :, :] - s[:, :, tf.newaxis, :]      # [B, A, A, 3]
        ds = ds - tf.round(ds)                                     # MIC wrap
        # Back to Cartesian
        dr = tf.einsum('bij,bnmj->bnmi', boxes, ds)               # [B, A, A, 3]
        rij = tf.linalg.norm(dr, axis=-1)                          # [B, A, A]
        return dr, rij

    def _dipole_batch(self, forces: tf.Tensor, positions: tf.Tensor, boxes: tf.Tensor,
                      grad_index: tf.Tensor, atom_mask: tf.Tensor,
                      neighbor_mask: tf.Tensor) -> tf.Tensor:
        """Batched dipole prediction.

        Args:
            forces        : [B, A, M, 3]
            positions     : [B, A, 3]
            boxes         : [B, 3, 3]
            grad_index    : [B, A, M]
            atom_mask     : [B, A]
            neighbor_mask : [B, A, M]

        Returns:
            dipole : [B, 3]
        """
        dr, rij = self._pairwise_displacements_batch(positions, boxes)
        B = tf.shape(positions)[0]
        A = tf.shape(positions)[1]
        M = tf.shape(grad_index)[2]

        # Gather displacements for grad_index neighbors
        batch_idx = tf.broadcast_to(
            tf.range(B)[:, tf.newaxis, tf.newaxis], [B, A, M])
        atom_idx = tf.broadcast_to(
            tf.range(A)[tf.newaxis, :, tf.newaxis], [B, A, M])
        indices = tf.stack([batch_idx, atom_idx, grad_index], axis=-1)  # [B,A,M,3]

        rij_gathered = tf.gather_nd(rij, indices)                        # [B, A, M]
        rij2 = tf.square(rij_gathered) * neighbor_mask                   # [B, A, M]

        dipole_contribs = rij2[:, :, :, tf.newaxis] * forces             # [B, A, M, 3]
        dipole = -tf.reduce_sum(dipole_contribs, axis=[1, 2])            # [B, 3]
        return dipole

    def _polarizability_batch(self, descriptors: tf.Tensor, forces: tf.Tensor,
                              positions: tf.Tensor, boxes: tf.Tensor, Z: tf.Tensor,
                              grad_index: tf.Tensor, atom_mask: tf.Tensor,
                              neighbor_mask: tf.Tensor, W0_pol: tf.Tensor,
                              b0_pol: tf.Tensor, W1_pol: tf.Tensor,
                              b1_pol: tf.Tensor) -> tf.Tensor:
        """Batched polarizability via dual ANN (GPUMD approach).

        Scalar ANN (W0_pol..b1_pol) -> isotropic diagonal.
        Tensor ANN (primary, via forces) -> anisotropic virial.

        Args:
            descriptors   : [B, A, Q]
            forces        : [B, A, M, 3]
            positions     : [B, A, 3]
            boxes         : [B, 3, 3]
            Z             : [B, A]
            grad_index    : [B, A, M]
            atom_mask     : [B, A]
            neighbor_mask : [B, A, M]
            W0_pol..b1_pol: scalar ANN weights [T,Q,H] etc.

        Returns:
            pol : [B, 6]  — [xx, yy, zz, xy, yz, zx]
        """
        dr, rij = self._pairwise_displacements_batch(positions, boxes)
        B = tf.shape(positions)[0]
        A = tf.shape(positions)[1]
        M = tf.shape(grad_index)[2]

        batch_idx = tf.broadcast_to(
            tf.range(B)[:, tf.newaxis, tf.newaxis], [B, A, M])
        atom_idx = tf.broadcast_to(
            tf.range(A)[tf.newaxis, :, tf.newaxis], [B, A, M])
        indices = tf.stack([batch_idx, atom_idx, grad_index], axis=-1)
        dr_gathered = tf.gather_nd(dr, indices)  # [B, A, M, 3]

        # Scalar ANN (isotropic)
        W0p_t = tf.gather(W0_pol, Z)  # [B, A, Q, H]
        b0p_t = tf.gather(b0_pol, Z)  # [B, A, H]
        W1p_t = tf.gather(W1_pol, Z)  # [B, A, H]

        h_pol = tf.einsum('bnd,bndh->bnh', descriptors, W0p_t)
        h_pol = h_pol + b0p_t
        h_pol = self.activation(h_pol)
        h_pol = h_pol * atom_mask[:, :, tf.newaxis]
        F_pol = tf.reduce_sum(h_pol * W1p_t, axis=2) + b1_pol  # [B, A]
        F_pol = F_pol * atom_mask
        scalar_sum = tf.reduce_sum(F_pol, axis=1)  # [B]

        # Tensor part: outer product (anisotropic virial)
        pol_outer = -tf.einsum('bnmi,bnmj->bnmij', dr_gathered, forces)  # [B,A,M,3,3]
        pol_outer = pol_outer * neighbor_mask[:, :, :, tf.newaxis, tf.newaxis]
        pol_matrix = tf.reduce_sum(pol_outer, axis=[1, 2])  # [B, 3, 3]

        # Extract 6 components
        pol = tf.stack([
            pol_matrix[:, 0, 0],
            pol_matrix[:, 1, 1],
            pol_matrix[:, 2, 2],
            pol_matrix[:, 0, 1],
            pol_matrix[:, 1, 2],
            pol_matrix[:, 2, 0],
        ], axis=1)  # [B, 6]

        # Add scalar to diagonal
        diag_add = tf.stack([scalar_sum, scalar_sum, scalar_sum,
                             tf.zeros_like(scalar_sum),
                             tf.zeros_like(scalar_sum),
                             tf.zeros_like(scalar_sum)], axis=1)  # [B, 6]
        pol = pol + diag_add
        return pol
