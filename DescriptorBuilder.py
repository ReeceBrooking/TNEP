import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Sequential, optimizers, losses
from TNEPconfig import TNEPconfig

class DescriptorBuilder(layers.Layer):
    """
    Per-atom descriptors:
      - Radial: summed radial basis over neighbors
      - Angular: 3-body invariants from angular_basis, summed over neighbor pairs
    """
    def __init__(self,
                 cfg: TNEPconfig,
                 **kwargs):
        super().__init__(**kwargs)

        # Angular 3-body basis parameters
        self.K_radial = cfg.n_radial
        self.Lmax = cfg.Lmax
        self.n_radial_ang = cfg.n_radial_ang
        self.rc = tf.constant(cfg.rc, tf.float32)
        self.pi = tf.constant(np.pi, tf.float32)

    def build_descriptors(self, R, box):
        """
        inputs = (R, Z, box)
        R:   [N,3]  Cartesian Å
        box: [3,3]  lattice vectors

        returns Q: [N, D] per-atom descriptor
        """
        N = tf.shape(R)[0]

        # --- Pairwise displacements and distances (PBC) ---
        dr, rij = self.pairwise_displacements(R, box)  # dr: [N,N,3], rij: [N,N]

        # Mask to exclude self-interactions i=j
        mask = 1.0 - tf.eye(N, dtype=tf.float32)  # [N,N]

        # ========================
        # 1) Radial 2-body block
        # ========================
        # Chebyshev radial basis
        phi_r = self.radial_basis(rij)                      # [N,N,K_radial+1]
        phi_r *= tf.expand_dims(mask, -1)          # zero out j=i

        # Sum over neighbors j to get per-atom radial features
        # q_r[i,k] = Σ_j Phi_r[i,j,k]
        q_r = tf.reduce_sum(phi_r, axis=1)         # [N, K_radial+1]

        # =========================
        # 2) Angular 3-body block
        # =========================
        # Use dr as neighbor vectors per central atom:
        # for each i, neighbors j have vectors dr[i,j,:]
        # angular_basis returns: [N, N, N, n_radial_ang, Lmax+1]
        ang_feat = self.angular_basis(dr,
                                 rij,
                                      )  # [N,N,N,n_radial_ang,Lmax+1]

        # Build pair mask to exclude:
        #  - j = i (no self neighbor)
        #  - k = i (no self neighbor)
        mask_ij = tf.expand_dims(mask, 2)          # [N,N,1]
        mask_ik = tf.expand_dims(mask, 1)          # [N,1,N]
        pair_mask = mask_ij * mask_ik              # [N,N,N] covers every i=j=k possibility?

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
        descriptors = tf.concat([q_r, q_ang], axis=-1)                  # [N, D]
        return descriptors

    # Neighbour List Function
    ## Enforce PBC by choosing closest possible image
    @tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[3, 3], dtype=tf.float32),
    ]
)
    def _minimum_image_displacement(self, Ri, Rj, box):
        """Return Rj - Ri wrapped by MIC for triclinic/orthorhombic cell.
           Ri,Rj: [...,3], box: [3,3] (rows = lattice vectors)."""
        box_inv = tf.linalg.inv(box)
        si = tf.einsum('ij,bj->bi', box_inv, Ri)  # fractional
        sj = tf.einsum('ij,bj->bi', box_inv, Rj)
        ds = sj - si
        ds -= tf.round(ds)  # wrap to [-0.5,0.5)
        dr = tf.einsum('ij,bj->bi', box, ds)  # back to Cartesian
        return dr

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[3, 3], dtype=tf.float32),
        ]
    )
    def pairwise_displacements(self, R, box):
        """Return dr_ij [N,N,3], r_ij [N,N]."""
        # Vectorize MIC across pairs
        N = tf.shape(R)[0]
        Ri_t = tf.reshape(tf.tile(R, [N, 1]), [N, N, 3])
        Rj_t = tf.transpose(Ri_t, perm=[1, 0, 2])
        dr_flat = self._minimum_image_displacement(tf.reshape(Ri_t, [-1, 3]),
                                              tf.reshape(Rj_t, [-1, 3]),
                                              box)
        dr = tf.reshape(dr_flat, [N, N, 3])
        rij = tf.linalg.norm(dr, axis=-1)
        return dr, rij

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        ]
    )
    def cutoff(self, rij):
        x = tf.clip_by_value(rij / self.rc, 0.0, 1.0)
        return 0.5 * (tf.cos(self.pi * x) + 1.0)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        ]
    )
    def radial_basis(self, rij):
        fc = self.cutoff(rij)
        # scaled distance
        # rcinv = tf.constant(1.0, tf.float32) / self.rc
        t = rij / self.rc # r/rc

        # x = 2*(t-1)^2 - 1
        x = 2.0 * tf.square(t - 1.0) - 1.0

        # Build Chebyshev T_n(x) up to n=K_radial
        # T0 = 1, T1 = x
        T0 = tf.ones_like(x)
        T1 = x

        feats = []

        # n=0: fn0 = fc
        feats.append(fc)

        if self.K_radial >= 1:
            # n=1: fn1 = 0.5*(x+1)*fc
            feats.append(0.5 * (x + 1.0) * fc)

        # n>=2
        for n in range(2, self.K_radial + 1):
            Tn = 2.0 * x * T1 - T0
            fn = 0.5 * (Tn + 1.0) * fc
            feats.append(fn)
            T0, T1 = T1, Tn

        # Stack last axis => [..., K+1]
        phi = tf.stack(feats, axis=-1)
        return phi

    ## Angular Basis
    @tf.function(
         input_signature=[
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None], dtype=tf.float32),
        ]
    )
    def angular_basis(self, dr, rij):
        """
        R: [N, M, 3] neighbor vectors for each central atom
        cutoff: scalar cutoff distance
        Lmax: max angular degree (e.g., 2)
        n_radial: number of radial basis functions
        Returns: angular features [N, M, M, n_radial, Lmax+1]
        """
        # 1. Compute pairwise distances and cutoff
        # dr, rij = pairwise_displacements(R, box)
    #    fc = self.cutoff(rij)

        # 2. Compute cos(theta_ijk)
        # Expand to compare every pair of neighbors j,k
    #    Rj = tf.expand_dims(dr, 2)  # [N, M, 1, 3]
    #    Rk = tf.expand_dims(dr, 1)  # [N, 1, M, 3]
    #    dot = tf.reduce_sum(Rj * Rk, axis=-1)  # [N, M, M]
    #    rij_mag = tf.expand_dims(rij, 2)
    #    rik_mag = tf.expand_dims(rij, 1)
    #    cos_theta = dot / (rij_mag * rik_mag + 1e-8)
    #    cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)  # Why clip?

        # 3. Compute Legendre polynomials up to Lmax
    #    P = [tf.ones_like(cos_theta)]  # P0(x) = 1
    #    if self.Lmax >= 1:
    #        P.append(cos_theta)  # P1(x) = x
    #    if self.Lmax >= 2:
    #        P.append(0.5 * (3 * cos_theta ** 2 - 1))  # P2(x)
    #    P = tf.stack(P, axis=-1)  # [N, M, M, Lmax+1]

        # 4. Compute radial terms
#        r_terms = [tf.pow(rij_mag, n) * tf.pow(rik_mag, n) for n in range(self.n_radial_ang)]  # verify
#        r_terms = tf.stack(r_terms, axis=-1)  # [N, M, M, n_radial]

        # 5. Multiply cutoff * radial * angular
 #       fc_pair = tf.expand_dims(fc, 2) * tf.expand_dims(fc, 1)  # [N, M, M]
 #       fc_pair = tf.expand_dims(fc_pair, -1)  # [N, M, M, 1]
 #       fc_pair = tf.expand_dims(fc_pair, -1)  # [N, M, M, 1, 1]
 #       features = fc_pair * tf.expand_dims(r_terms, -1) * tf.expand_dims(P, -2)  # verify

        r_terms = self.radial_basis(rij)


        # features shape: [N, M, M, n_radial, Lmax+1]
        return features