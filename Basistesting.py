import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Sequential, optimizers, losses

# Neighbour List Function
## Enforce PBC by choosing closest possible image
@tf.function
def _minimum_image_displacement(Ri, Rj, box):
    """Return Rj - Ri wrapped by MIC for triclinic/orthorhombic cell.
       Ri,Rj: [...,3], box: [3,3] (rows = lattice vectors)."""
    box_inv = tf.linalg.inv(box)
    si = tf.einsum('ij,bj->bi', box_inv, Ri)  # fractional
    sj = tf.einsum('ij,bj->bi', box_inv, Rj)
    ds = sj - si
    ds -= tf.round(ds)                        # wrap to [-0.5,0.5)
    dr = tf.einsum('ij,bj->bi', box, ds)      # back to Cartesian
    return dr

@tf.function
def pairwise_displacements(R, box):
    """Return dr_ij [N,N,3], r_ij [N,N]."""
    Ri = tf.expand_dims(R, 1)   # [N,1,3]
    Rj = tf.expand_dims(R, 0)   # [1,N,3]
    # Vectorize MIC across pairs
    N = tf.shape(R)[0]
    Ri_t = tf.reshape(tf.tile(R, [N,1]), [N, N, 3])
    Rj_t = tf.transpose(Ri_t, perm=[1,0,2])
    dr_flat = _minimum_image_displacement(tf.reshape(Ri_t, [-1,3]),
                                          tf.reshape(Rj_t, [-1,3]),
                                          box)
    dr = tf.reshape(dr_flat, [N, N, 3])
    rij = tf.linalg.norm(dr + 1e-16, axis=-1)
    return dr, rij#, Ri, Rj, Ri_t, Rj_t, dr_flat

R = tf.convert_to_tensor([[9, 2, 2], [8, 2, 6], [3, 3, 2], [1, 2, 3], [4, 5, 6], [7, 8, 4]], dtype=tf.float32)
box = tf.convert_to_tensor([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=tf.float32)
rc = 5.0

# dr, rij = pairwise_displacements(R, box)

class CosineCutoff(layers.Layer):
    def __init__(self, rc, **kwargs):
        super().__init__(**kwargs)
        self.rc = tf.constant(rc, tf.float32)
    def call(self, r):
        x = tf.clip_by_value(r / self.rc, 0.0, 1.0)
        return 0.5 * (tf.cos(np.pi * x) + 1.0) # * tf.cast(r < self.rc, tf.float32)

# print(cut(rij, rc))

# Descriptor Construction Function
## Radial Basis from Power Series
class RadialBasis(layers.Layer):
    """Polynomial-with-cutoff basis: phi_k(r)= r^k * fc(r), k = 0..K."""
    def __init__(self, K, rc, **kwargs):
        super().__init__(**kwargs)
        self.K = K
        self.cut = CosineCutoff(rc)
    def call(self, r):  # r: [...,]
        fc = self.cut(r)
        feats = [tf.ones_like(r)]
        for k in range(1, self.K + 1):
            feats.append(tf.pow(r, k))
        Phi = tf.stack(feats, axis=-1)       # [..., K+1]
        return Phi * tf.expand_dims(fc, -1), Phi, fc

@tf.function
def angular_basis(R, box, rc, Lmax=2, n_radial=3):
    """
    R: [N, M, 3] neighbor vectors for each central atom
    cutoff: scalar cutoff distance
    Lmax: max angular degree (e.g., 2)
    n_radial: number of radial basis functions
    Returns: angular features [N, M, M, n_radial, Lmax+1]
    """
    # 1. Compute pairwise distances
    #rij = tf.linalg.norm(R, axis=-1)  # [N, M]
    #fc = 0.5 * (tf.cos(np.pi * rij / cutoff) + 1.0) * tf.cast(rij < cutoff, tf.float32)
    dr, rij = pairwise_displacements(R, box)
    fc = CosineCutoff(rc).call(rij)

    # 2. Compute cos(theta_ijk)
    # Expand to compare every pair of neighbors j,k
    Rj = tf.expand_dims(R, 2)  # [N, M, 1, 3]
    Rk = tf.expand_dims(R, 1)  # [N, 1, M, 3]
    dot = tf.reduce_sum(Rj * Rk, axis=-1)  # [N, M, M]
    rij_mag = tf.expand_dims(rij, 2)
    rik_mag = tf.expand_dims(rij, 1)
    cos_theta = dot / (rij_mag * rik_mag + 1e-8)
    cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0) # Why clip?

    # 3. Compute Legendre polynomials up to Lmax
    P = [tf.ones_like(cos_theta)]  # P0(x) = 1
    if Lmax >= 1:
        P.append(cos_theta)  # P1(x) = x
    if Lmax >= 2:
        P.append(0.5 * (3 * cos_theta ** 2 - 1))  # P2(x)
    P = tf.stack(P, axis=-1)  # [N, M, M, Lmax+1]

    # 4. Compute radial terms
    r_terms = [tf.pow(rij_mag, n) * tf.pow(rik_mag, n) for n in range(n_radial)]
    r_terms = tf.stack(r_terms, axis=-1)  # [N, M, M, n_radial]

    # 5. Multiply cutoff * radial * angular
    fc_pair = tf.expand_dims(fc, 2) * tf.expand_dims(fc, 1)  # [N, M, M]
    fc_pair = tf.expand_dims(fc_pair, -1)  # [N, M, M, 1]
    fc_pair = tf.expand_dims(fc_pair, -1)  # [N, M, M, 1, 1]
    features = fc_pair * tf.expand_dims(r_terms, -1) * tf.expand_dims(P, -2)

    # features shape: [N, M, M, n_radial, Lmax+1]
    # needs to be summed for j, k?
    return features

#basis = RadialBasis(3, 5.0)
#print(basis.call(rij))

class DescriptorBuilder(layers.Layer):
    """
    Per-atom descriptors:
      - Radial: summed radial basis over neighbors
      - Angular: 3-body invariants from angular_basis, summed over neighbor pairs
    """
    def __init__(self,
                 K_radial=8,
                 rc=6.0,
                 Lmax=2,
                 n_radial_ang=3,
                 **kwargs):
        super().__init__(**kwargs)
        # Radial 2-body basis
        self.rad = RadialBasis(K_radial, rc)

        # Angular 3-body basis parameters
        self.Lmax = Lmax
        self.n_radial_ang = n_radial_ang
        self.rc = rc

    def call(self, inputs):
        """
        inputs = (R, Z, box)
        R:   [N,3]  Cartesian Å
        Z:   [N]    int types (unused here but kept for extensibility)
        box: [3,3]  lattice vectors

        returns Q: [N, D] per-atom descriptor
        """
        R, Z, box = inputs
        N = tf.shape(R)[0]

        # --- Pairwise displacements and distances (PBC) ---
        dr, rij = pairwise_displacements(R, box)  # dr: [N,N,3], rij: [N,N]

        # Mask to exclude self-interactions i=j
        mask = 1.0 - tf.eye(N, dtype=tf.float32)  # [N,N]

        # ========================
        # 1) Radial 2-body block
        # ========================
        # Phi_r[i,j,k] = r_ij^k * fc(r_ij), k=0..K_radial
        Phi_r = self.rad(rij)                      # [N,N,K_radial+1]
        Phi_r *= tf.expand_dims(mask, -1)          # zero out j=i

        # Sum over neighbors j to get per-atom radial features
        # q_r[i,k] = Σ_j Phi_r[i,j,k]
        q_r = tf.reduce_sum(Phi_r, axis=1)         # [N, K_radial+1]

        # =========================
        # 2) Angular 3-body block
        # =========================
        # Use dr as neighbor vectors per central atom:
        # for each i, neighbors j have vectors dr[i,j,:]
        # angular_basis returns: [N, N, N, n_radial_ang, Lmax+1]
        ang_feat = angular_basis(dr,
                                 cutoff=self.rc,
                                 Lmax=self.Lmax,
                                 n_radial=self.n_radial_ang)  # [N,N,N,n_radial_ang,Lmax+1]

        # Build pair mask to exclude:
        #  - j = i (no self neighbor)
        #  - k = i (no self neighbor)
        mask_ij = tf.expand_dims(mask, 2)          # [N,N,1]
        mask_ik = tf.expand_dims(mask, 1)          # [N,1,N]
        pair_mask = mask_ij * mask_ik              # [N,N,N]

        # Apply mask to angular features
        ang_feat = ang_feat * tf.expand_dims(tf.expand_dims(pair_mask, -1), -1)
        # shape still [N,N,N,n_radial_ang,Lmax+1]

        # Sum over neighbor pairs (j,k) to get per-atom angular invariants
        # q_ang[i, n, l] = Σ_j Σ_k ang_feat[i,j,k,n,l]
        q_ang = tf.reduce_sum(ang_feat, axis=[1, 2])          # [N, n_radial_ang, Lmax+1]

        # Flatten angular channels
        q_ang = tf.reshape(q_ang, [N, -1])                    # [N, n_radial_ang*(Lmax+1)]

        # =========================
        # 3) Concatenate descriptors
        # =========================
        Q = tf.concat([q_r, q_ang], axis=-1)                  # [N, D]
        return Q


class TNEPPerTypeANN(layers.Layer):
    """
    TensorFlow/Keras implementation of the TNEP per-type 1-hidden-layer ANN.

    - Input: per-atom descriptor q_i (dim_q) and atom type Z_i in [0, num_types)
    - For each type t:
        h_i = act(q_i W0[t] + b0[t])          # hidden layer
        F_i = h_i · w1[t] + b1               # scalar output per atom
    - b1 is a global scalar bias shared by all types (like annmb.b1).
    """

    def __init__(self,
                 dim_q: int,
                 num_types: int,
                 num_neurons1: int,
                 activation="tanh",
                 **kwargs):
        super().__init__(**kwargs)
        self.dim_q = dim_q
        self.num_types = num_types
        self.num_neurons1 = num_neurons1
        self.activation = tf.keras.activations.get(activation)

        # W0[t] : [dim_q, num_neurons1]  (input -> hidden)
        self.W0 = self.add_weight(
            name="W0",
            shape=(num_types, dim_q, num_neurons1),
            initializer="glorot_uniform",
            trainable=True,
        )

        # b0[t] : [num_neurons1]  (hidden bias)
        self.b0 = self.add_weight(
            name="b0",
            shape=(num_types, num_neurons1),
            initializer="zeros",
            trainable=True,
        )

        # W1[t] : [num_neurons1]  (hidden -> scalar)
        # in the C++ code this is stored as an array of length num_neurons1
        self.W1 = self.add_weight(
            name="W1",
            shape=(num_types, num_neurons1),
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

    def call(self, q, Z):
        """
        q : [N, dim_q]  per-atom descriptors
        Z : [N]        integer atom types (0..num_types-1)

        Returns:
            F : [N]  per-atom scalar output (e.g., energy contribution)
        """
        # Ensure integer types
        Z = tf.cast(Z, tf.int32)
        N = tf.shape(q)[0]

        # Gather per-type parameters
        # W0_t: [N, dim_q, num_neurons1]
        # b0_t: [N, num_neurons1]
        # W1_t: [N, num_neurons1]
        W0_t = tf.gather(self.W0, Z)   # index by type
        b0_t = tf.gather(self.b0, Z)
        W1_t = tf.gather(self.W1, Z)

        # Hidden layer: h = act(q W0_t + b0_t)
        # q: [N, dim_q]
        # We want: [N, num_neurons1]
        q_exp = tf.expand_dims(q, axis=1)        # [N, 1, dim_q]
        h = tf.matmul(q_exp, W0_t)               # [N, 1, num_neurons1]
        h = tf.squeeze(h, axis=1) + b0_t         # [N, num_neurons1]
        h = self.activation(h)                   # [N, num_neurons1]

        # Output layer: F_i = h_i · W1_t + b1
        # (elementwise dot over last dim)
        F = tf.reduce_sum(h * W1_t, axis=-1)     # [N]
        F = F + self.b1                          # global bias
        return F
