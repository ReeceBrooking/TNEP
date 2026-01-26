import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Sequential, optimizers, losses
from TNEPconfig import TNEPconfig
from quippy.descriptors import Descriptor

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
        self.cfg = cfg
        self.types = cfg.types
        self.num_types = cfg.num_types
        self.rc = cfg.rc

        base = (
            "soap_turbo l_max=8 "
            "rcut_hard=3.7 rcut_soft=3.2 basis=poly3gauss scaling_mode=polynomial add_species=F "
            "radial_enhancement=1 compress_mode=trivial "
        )

        alpha_max = " alpha_max={"
        atom_sigma_r = " atom_sigma_r={"
        atom_sigma_t = " atom_sigma_t={"
        atom_sigma_r_scaling = " atom_sigma_r_scaling={"
        atom_sigma_t_scaling = " atom_sigma_t_scaling={"
        amplitude_scaling = " amplitude_scaling={"
        central_weight = " central_weight={"

        for a in range(self.num_types):
            alpha_max += "8 "
            atom_sigma_r += "0.5 "
            atom_sigma_t += "0.5 "
            atom_sigma_r_scaling += "0.0 "
            atom_sigma_t_scaling += "0.0 "
            amplitude_scaling += "1.0 "
            central_weight += "1. "

        alpha_max += "}"
        atom_sigma_r += "}"
        atom_sigma_t += "}"
        atom_sigma_r_scaling += "}"
        atom_sigma_t_scaling += "}"
        amplitude_scaling += "}"
        central_weight += "}"

        n_species = " n_species=" + str(self.num_types)

        species_Z = " species_Z={"
        for type in self.types:
            species_Z += str(type) + " "
        species_Z += "}"

        base += species_Z + n_species + alpha_max + atom_sigma_r + atom_sigma_t + atom_sigma_r_scaling + atom_sigma_t_scaling + amplitude_scaling + central_weight

        self.builders = [Descriptor(base + f" central_index={k}") for k in (np.arange(self.num_types, dtype=int) + 1)]
    # TODO Explore SparseTensor Class
    def build_descriptors(self, dataset):
        dataset_descriptors = []
        dataset_gradients = []
        dataset_grad_index = []

        for structure in dataset:
            outs = [b.calc(structure, grad=True) for b in self.builders]

            descriptors = [[] for _ in range(len(structure))]
            gradients = [[] for _ in range(len(structure))]
            grad_indexes = [[] for _ in range(len(structure))]

            for out in outs:
                # print(out)
                data = out.get("data")
                if data is None or data.size == 0 or data.shape[1] == 0:
                    continue
                # print(out["ci"])
                for k in range(len(out["ci"])):
                    descriptors[out["ci"][k] - 1].append(out["data"][k])
                #            for index in out["ci"]:
                #                center_index.append(index)
                #            assert len(center_index) == len(descriptors)
                for j in range(len(out["grad_index_0based"])):
                    center = out["grad_index_0based"][j][0]
                    neighbour = out["grad_index_0based"][j][1]
                    gradients[center].append(out["grad_data"][j])
                    grad_indexes[center].append(neighbour)
            for i in range(len(gradients)):
                gradients[i] = tf.convert_to_tensor(gradients[i], dtype=tf.float32)

            descriptors = tf.convert_to_tensor(descriptors, dtype=tf.float32)
            descriptors = tf.squeeze(descriptors, axis=1)
            #        print(descriptors.shape)
            dataset_descriptors.append(descriptors)
            dataset_gradients.append(gradients)
            dataset_grad_index.append(grad_indexes)
        return dataset_descriptors, dataset_gradients, dataset_grad_index

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

    """
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
 #       R: [N, M, 3] neighbor vectors for each central atom
 #       cutoff: scalar cutoff distance
 #       Lmax: max angular degree (e.g., 2)
 #       n_radial: number of radial basis functions
 #       Returns: angular features [N, M, M, n_radial, Lmax+1]
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
    """