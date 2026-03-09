import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

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
    The scalar ANN computes per-atom F_pol -> diagonal components.
    The tensor ANN (primary W0/b0/W1/b1) computes forces -> off-diagonal + virial.
    Output: [xx, yy, zz, xy, yz, zx]
    """

    def __init__(self,
                 cfg: TNEPconfig,
                 **kwargs):
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

    def predict(self, descriptors, gradients, grad_index, positions, Z, box):
        """Run the forward pass for a single structure.

        Args:
            descriptors : [N, dim_q]         per-atom SOAP descriptors
            gradients   : list of N tensors, each [M_i, 3, dim_q]
                          dq_i/dR_j for each neighbour j (M_i neighbours,
                          3 Cartesian components, dim_q descriptor dims)
            grad_index  : list of N lists, each [M_i] int
                          neighbour atom indices corresponding to gradients
            positions   : [N, 3]             Cartesian atom positions
            Z           : [N]               integer type indices (0..num_types-1)
            box         : [3, 3]             lattice vectors (rows)

        Returns:
            target_mode 0: scalar total energy
            target_mode 1: [3] dipole vector
            target_mode 2: [6] polarizability tensor [xx, yy, zz, xy, yz, zx]
        """
        N = tf.shape(Z)[0]

        # Gather per-type weights for each atom
        W0_t = tf.gather(self.W0, Z)   # [N, dim_q, num_neurons]
        b0_t = tf.gather(self.b0, Z)   # [N, num_neurons]
        W1_t = tf.gather(self.W1, Z)   # [N, num_neurons]

        # Hidden layer: h_i = tanh(q_i @ W0[t_i] + b0[t_i])
        q_exp = tf.expand_dims(descriptors, axis=1)  # [N, 1, dim_q]
        h = tf.matmul(q_exp, W0_t)                   # [N, 1, num_neurons]
        h = tf.squeeze(h, axis=1) + b0_t              # [N, num_neurons]
        h = self.activation(h)                         # [N, num_neurons]

        if self.cfg.target_mode == 0:
            # PES: E = -sum_i (h_i . W1[t_i] + b1)
            E = tf.reduce_sum(h * W1_t, axis=1)  # [N] per-atom energies
            E = E + self.b1
            E = tf.reduce_sum(E)
            return -E
        elif self.cfg.target_mode == 1:
            # Dipole: μ = -sum_i sum_{j!=i} |r_ij|^2 * (dU_i/dr_ij_vec)
            dr, rij = self.builder.pairwise_displacements(
                tf.convert_to_tensor(positions, dtype=tf.float32),
                tf.convert_to_tensor(box, dtype=tf.float32))
            mask = 1.0 - tf.eye(N, dtype=tf.float32)
            rij2 = tf.square(rij) * mask  # [N, N] squared scalar distances, self-terms zeroed

            # dU_i/dr_ij_vec for all atoms and their neighbours
            forces = self.calc_forces(h, gradients, W1_t, W0_t)

            # Assemble dipole: weight each force by squared distance
            dipole = []
            for i in range(len(forces)):
                dipole_i = tf.zeros(3, dtype=tf.float32)
                for j in range(len(forces[i])):
                    neighbor_idx = grad_index[i][j]
                    rij2_val = rij2[i, neighbor_idx]       # scalar |r_ij|^2
                    dipole_i += rij2_val * forces[i][j]    # scalar * [3]
                dipole.append(dipole_i)

            dipole = tf.convert_to_tensor(dipole, dtype=tf.float32)  # [N, 3]
            dipole = -tf.reduce_sum(dipole, axis=0)                  # [3]
            return dipole
        elif self.cfg.target_mode == 2:
            # Polarizability via dual ANN (GPUMD tnep.cu apply_ann_pol)
            # Scalar ANN -> per-atom F_pol -> diagonal polarizability
            # Tensor ANN -> forces via descriptor gradients -> off-diagonal
            dr, rij = self.builder.pairwise_displacements(
                tf.convert_to_tensor(positions, dtype=tf.float32),
                tf.convert_to_tensor(box, dtype=tf.float32))
            mask = 1.0 - tf.eye(N, dtype=tf.float32)

            # --- Scalar ANN (polarizability-specific weights) ---
            W0p_t = tf.gather(self.W0_pol, Z)  # [N, dim_q, H]
            b0p_t = tf.gather(self.b0_pol, Z)  # [N, H]
            W1p_t = tf.gather(self.W1_pol, Z)  # [N, H]

            h_pol = tf.matmul(q_exp, W0p_t)                    # [N, 1, H]
            h_pol = tf.squeeze(h_pol, axis=1) + b0p_t           # [N, H]
            h_pol = self.activation(h_pol)                       # [N, H]
            F_pol = tf.reduce_sum(h_pol * W1p_t, axis=1) + self.b1_pol  # [N]

            # Diagonal: alpha_aa = sum_i(F_pol_i) + sum_ij(-r_ij_a * f_ij_a)
            scalar_sum = tf.reduce_sum(F_pol)  # scalar

            # --- Tensor ANN (primary weights) -> forces ---
            forces = self.calc_forces(h, gradients, W1_t, W0_t)

            # Assemble 6-component polarizability: [xx, yy, zz, xy, yz, zx]
            # Component pairs: (0,0), (1,1), (2,2), (0,1), (1,2), (2,0)
            comp_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]
            pol = tf.zeros(6, dtype=tf.float32)

            for i in range(len(forces)):
                for j_idx in range(len(forces[i])):
                    neighbor = grad_index[i][j_idx]
                    r_vec = dr[i, neighbor]         # [3] displacement vector
                    f_vec = forces[i][j_idx]        # [3] force vector

                    for c, (a, b) in enumerate(comp_pairs):
                        contrib = -r_vec[a] * f_vec[b]
                        pol = pol + tf.one_hot(c, 6, dtype=tf.float32) * contrib

            # Add scalar ANN contribution to diagonal components
            for c in range(3):  # xx, yy, zz
                pol = pol + tf.one_hot(c, 6, dtype=tf.float32) * scalar_sum

            return pol
        else:
            print("target mode not supported")
            return

    def calc_forces(self, h, gradients, W1_t, W0_t):
        """Compute dU_i/dR_j for every atom i and its neighbours j via chain rule.

        Chain rule: dU_i/dR_j = sum_q (dU_i/dq_iq) * (dq_iq/dR_j)

        Args:
            h         : [N, num_neurons]          hidden activations tanh(a)
            gradients : list of N tensors, each [M_i, 3, dim_q]
                        descriptor gradients dq_i/dR_j from quippy
            W1_t      : [N, num_neurons]          per-atom output weights
            W0_t      : [N, dim_q, num_neurons]   per-atom input weights

        Returns:
            forces : list of N tensors, each [M_i, 3]
                     dU_i/dR_j (3-vector per neighbour)
        """
        # dU/dh * dh/da = W1 * (1 - tanh^2(a))
        dtanh = 1.0 - tf.square(h)                               # [N, H]
        de_da = dtanh * W1_t                                      # [N, H]
        # dU/dq = dU/da @ W0^T
        de_da_exp = tf.expand_dims(de_da, axis=1)                 # [N, 1, H]
        de_dq = tf.matmul(de_da_exp, W0_t, transpose_b=True)     # [N, 1, dim_q]
        de_dq = tf.squeeze(de_dq, axis=1)                        # [N, dim_q]

        # Contract dU/dq with dq/dR_j for each atom
        forces = []
        for i in range(len(gradients)):
            # de_dq[i]: [dim_q] broadcasts with gradients[i]: [M_i, 3, dim_q]
            # sum over dim_q (axis=-1) -> [M_i, 3]
            force_i = tf.reduce_sum(de_dq[i] * gradients[i], axis=-1)
            forces.append(force_i)
        return forces

    def fit(self, train_data, val_data):
        """Train the model using the SNES evolutionary optimizer.

        Args:
            train_data : dict with keys descriptors, gradients, grad_index,
                         positions, Z_int, targets, boxes (lists over structures)
            val_data   : same structure, used for validation each generation

        Returns:
            history : dict with keys generation, train_loss, val_loss (lists)
        """
        history = self.optimizer.fit(train_data, val_data)
        return history

    def score(self, test_data):
        """Evaluate RMSE over all structures in test_data.

        Args:
            test_data : dict, same structure as train_data

        Returns:
            rmse : scalar tf.Tensor
                   For PES: RMSE over scalar energies.
                   For dipole: RMSE over all 3 components across structures.
        """
        test_descriptors = test_data["descriptors"]
        test_gradients = test_data["gradients"]
        test_grad_index = test_data["grad_index"]
        test_positions = test_data["positions"]
        test_targets = test_data["targets"]
        test_z = test_data["Z_int"]
        boxes = test_data["boxes"]
        predictions = [self.predict(test_descriptors[i], test_gradients[i], test_grad_index[i], test_positions[i], test_z[i], boxes[i]) for i in range(len(test_descriptors))]
        predictions_tf = tf.convert_to_tensor(predictions, dtype=tf.float32)
        diff = predictions_tf - test_targets
        mse = tf.reduce_mean(tf.square(diff))
        rmse_loss = tf.sqrt(mse)
        return rmse_loss
