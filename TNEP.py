import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Sequential, optimizers, losses
from tensorflow.python.ops.ragged.ragged_math_ops import reduce_sum

from DescriptorBuilder import DescriptorBuilder
from SNES import SNES
from TNEPconfig import TNEPconfig

class TNEP(layers.Layer):
    """
    TensorFlow/Keras implementation of the TNEP per-type 1-hidden-layer ANN.

    - Input: per-atom descriptor q_i (dim_q) and atom type Z_i in [0, num_types)
    - For each type t:
        h_i = act(q_i W0[t] + b0[t])          # hidden layer
        F_i = h_i · w1[t] + b1               # scalar output per atom
    - b1 is a global scalar bias shared by all types (like annmb.b1).
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

        # W0[t] : [dim_q, num_neurons1]  (input -> hidden)
        self.W0 = self.add_weight(
            name="W0",
            shape=(cfg.num_types, cfg.dim_q, cfg.num_neurons),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b0[t] : [num_neurons1]  (hidden bias)
        self.b0 = self.add_weight(
            name="b0",
            shape=(cfg.num_types, cfg.num_neurons),
            initializer="zeros",
            trainable=True,
        )

        # W1[t] : [num_neurons1]  (hidden -> scalar)
        # in the C++ code this is stored as an array of length num_neurons1
        self.W1 = self.add_weight(
            name="W1",
            shape=(cfg.num_types, cfg.num_neurons),
            initializer="glorot_uniform",
            trainable=True,
        )

        # global scalar bias b1 (shared across all types)
        self.b1 = self.add_weight(
            name="b1",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def predict(self, descriptors, gradients, positions, Z, box):
        """
        q : [N, dim_q]  per-atom descriptors
        Z : [N]        integer atom types (0..num_types-1)

        Returns:
            F : [N]  per-atom scalar output (e.g., energy contribution)
        """

        N = tf.shape(Z)[0]

        # Gather per-type parameters - Change to take an input from parameter vector ss?
        # W0_t: [N, dim_q, num_neurons1]
        # b0_t: [N, num_neurons1]
        # W1_t: [N, num_neurons1]
        W0_t = tf.gather(self.W0, Z)   # index by type
        b0_t = tf.gather(self.b0, Z)
        W1_t = tf.gather(self.W1, Z)

        # Hidden layer: h = act(q W0_t + b0_t)
        # q: [N, dim_q]
        # We want: [N, num_neurons1]
        q_exp = tf.expand_dims(descriptors, axis=1)        # [N, 1, dim_q]
        h = tf.matmul(q_exp, W0_t)               # [N, 1, num_neurons1]
        h = tf.squeeze(h, axis=1) + b0_t         # [N, num_neurons1]
        h = self.activation(h)                   # [N, num_neurons1]

        # TODO Partial Force calculations - return Energy and Force predictions
        if self.cfg.target_mode == 0:
            # Output layer: F_i = h_i · W1_t + b1
            # (elementwise dot over last dim)
            E = tf.reduce_sum(h * W1_t, axis=1)  # single value scalar
            E = E + self.b1  # global bias
            E = tf.reduce_sum(E)
            return E
        elif self.cfg.target_mode == 1:
            dr, rij = self.builder.pairwise_displacements(tf.convert_to_tensor(positions, dtype=tf.float32), tf.convert_to_tensor(box, dtype=tf.float32))
            mask = 1.0 - tf.eye(N, dtype=tf.float32)  # [N,N]
            rij2 = tf.square(rij) * mask
            # tanh derivative
            dtanh = 1.0 - tf.square(h)  # [N, H]

            # ∂e_i / ∂a_i
            de_da = dtanh * W1_t  # [N, H]

            # ∂e_i / ∂q_i = W0_t · de_da
            de_da_exp = tf.expand_dims(de_da, axis=1)  # [N, 1, H]
            de_dq = tf.matmul(de_da_exp, W0_t, transpose_b=True)
            de_dq = tf.squeeze(de_dq, axis=1)
            de_dr = tf.einsum(
                "idk,ik->id",
                gradients,
                de_dq
            )  # [N_atoms, 3]
#            print(tf.shape(de_dr))
            dipole = reduce_sum(tf.matmul(rij2, de_dr), axis=0)
#            print(tf.shape(dipole))
            print("dipole calculated == " + str(dipole))
            return dipole
        elif self.cfg.target_mode == 2:
            return pol
        else:
            print("target mode not supported")
            return

    def fit(self, train_data, val_data):
        # needs to init an optimizer, passing itself and the arguments
        optimizer = SNES(self)
        history = optimizer.fit(train_data, val_data)
        # performs n generation loops, calculating fitness and updating parameter values
        return history

    def score(self, test_data):
        test_descriptors = test_data["descriptors"]
        test_positions = test_data["positions"]
        test_targets = test_data["targets"]
        test_z = test_data["Z_int"]
        box = test_data["box"]
        predictions = [self.predict(test_descriptors[i], test_positions[i], test_z, box) for i in range(len(test_descriptors))]
        predictions_tf = tf.convert_to_tensor(predictions, dtype=tf.float32)
        loss = tf.square(predictions_tf - test_targets)
        total_loss = tf.reduce_sum(loss, axis=-1)
        mean_loss = tf.reduce_mean(total_loss)
        rmse_loss = tf.sqrt(mean_loss)
        return rmse_loss