import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from TNEPconfig import TNEPconfig
from quippy.descriptors import Descriptor

class DescriptorBuilder(layers.Layer):
    """Builds SOAP-turbo descriptors and their gradients using quippy.

    Constructs one quippy Descriptor per atom type (central_index), then
    aggregates per-atom descriptors, descriptor gradients, and neighbour
    indices for each structure in a dataset.

    Also provides geometry utilities (pairwise displacements under MIC)
    needed by the dipole prediction branch.
    """

    def __init__(self,
                 cfg: TNEPconfig,
                 **kwargs):
        super().__init__(**kwargs)

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

    def build_descriptors(self, dataset):
        """Compute SOAP descriptors and their gradients for every structure.

        Runs each per-type quippy Descriptor with grad=True, then collects
        results per centre atom.

        Args:
            dataset : list of ase.Atoms structures

        Returns:
            dataset_descriptors : list of tensors, one per structure
                Each tensor has shape [N, dim_q].
            dataset_gradients   : list of (list of N tensors), one per structure
                gradients[s][i] has shape [M_i, 3, dim_q] — the derivative of
                atom i's descriptor w.r.t. each neighbour's position
                (M_i neighbours, 3 Cartesian, dim_q descriptor components).
            dataset_grad_index  : list of (list of N lists), one per structure
                grad_index[s][i] is a list of M_i ints — the atom index of
                each neighbour in gradients[s][i].
        """
        dataset_descriptors = []
        dataset_gradients = []
        dataset_grad_index = []

        for structure in dataset:
            outs = [b.calc(structure, grad=True) for b in self.builders]

            descriptors = [[] for _ in range(len(structure))]
            gradients = [[] for _ in range(len(structure))]
            grad_indexes = [[] for _ in range(len(structure))]

            for out in outs:
                data = out.get("data")
                if data is None or data.size == 0 or data.shape[1] == 0:
                    continue
                # ci is 1-indexed centre atom index from quippy
                for k in range(len(out["ci"])):
                    descriptors[out["ci"][k] - 1].append(out["data"][k])
                # grad_index_0based[j] = [centre, neighbour] (0-indexed)
                for j in range(len(out["grad_index_0based"])):
                    center = out["grad_index_0based"][j][0]
                    neighbour = out["grad_index_0based"][j][1]
                    gradients[center].append(out["grad_data"][j])
                    grad_indexes[center].append(neighbour)

            for i in range(len(gradients)):
                gradients[i] = tf.convert_to_tensor(gradients[i], dtype=tf.float32)

            descriptors = tf.convert_to_tensor(descriptors, dtype=tf.float32)
            descriptors = tf.squeeze(descriptors, axis=1)
            dataset_descriptors.append(descriptors)
            dataset_gradients.append(gradients)
            dataset_grad_index.append(grad_indexes)
        return dataset_descriptors, dataset_gradients, dataset_grad_index

    @staticmethod
    def compute_scaling(descriptors):
        """Compute per-component min and max from a list of descriptor tensors.

        Should be called on the training set only. The returned q_min and
        q_max are then used to scale train, val, and test sets identically.

        Args:
            descriptors : list of [N_i, dim_q] tensors (one per structure)

        Returns:
            q_min : [dim_q] tensor — per-component minimum across all atoms
            q_max : [dim_q] tensor — per-component maximum across all atoms
        """
        all_q = tf.concat(descriptors, axis=0)  # [total_atoms, dim_q]
        q_min = tf.reduce_min(all_q, axis=0)    # [dim_q]
        q_max = tf.reduce_max(all_q, axis=0)    # [dim_q]
        return q_min, q_max

    @staticmethod
    def apply_scaling(descriptors, gradients, q_min, q_max):
        """Scale descriptors to [-1, 1] and apply the same factor to gradients.

        Scaling:  q_scaled = 2 * (q - q_min) / (q_max - q_min) - 1
        The gradient transforms by the chain rule:
            dq_scaled/dR = dq/dR * 2 / (q_max - q_min)

        Components where q_max == q_min (constant) are left at zero.

        Args:
            descriptors : list of [N_i, dim_q] tensors
            gradients   : list of (list of N_i tensors each [M_i, 3, dim_q])
            q_min       : [dim_q] tensor from compute_scaling
            q_max       : [dim_q] tensor from compute_scaling

        Returns:
            scaled_descriptors : same structure, values in [-1, 1]
            scaled_gradients   : same structure, scaled by 2 / (q_max - q_min)
        """
        q_range = q_max - q_min
        # Avoid division by zero for constant components
        safe_range = tf.where(q_range > 0, q_range, tf.ones_like(q_range))
        scale = 2.0 / safe_range  # [dim_q]

        scaled_descriptors = []
        for q in descriptors:
            q_scaled = (q - q_min) * scale - 1.0
            # Zero out constant components (where q_range == 0)
            q_scaled = tf.where(q_range > 0, q_scaled, tf.zeros_like(q_scaled))
            scaled_descriptors.append(q_scaled)

        scaled_gradients = []
        for struct_grads in gradients:
            scaled_struct = []
            for g in struct_grads:
                # g: [M_i, 3, dim_q] — scale along the dim_q axis
                g_scaled = g * scale
                g_scaled = tf.where(q_range > 0, g_scaled, tf.zeros_like(g_scaled))
                scaled_struct.append(g_scaled)
            scaled_gradients.append(scaled_struct)

        return scaled_descriptors, scaled_gradients

    @tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        tf.TensorSpec(shape=[3, 3], dtype=tf.float32),
    ]
)
    def _minimum_image_displacement(self, Ri, Rj, box):
        """Compute displacement vectors Rj - Ri under minimum image convention.

        Args:
            Ri  : [B, 3]  reference positions
            Rj  : [B, 3]  target positions
            box : [3, 3]  lattice vectors (rows)

        Returns:
            dr : [B, 3]  Cartesian displacement vectors wrapped to nearest image
        """
        box_inv = tf.linalg.inv(box)
        si = tf.einsum('ij,bj->bi', box_inv, Ri)  # Cartesian -> fractional
        sj = tf.einsum('ij,bj->bi', box_inv, Rj)
        ds = sj - si
        ds -= tf.round(ds)                         # wrap to [-0.5, 0.5)
        dr = tf.einsum('ij,bj->bi', box, ds)       # fractional -> Cartesian
        return dr

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[3, 3], dtype=tf.float32),
        ]
    )
    def pairwise_displacements(self, R, box):
        """Compute all pairwise displacement vectors and scalar distances under MIC.

        Args:
            R   : [N, 3]  atom positions
            box : [3, 3]  lattice vectors (rows)

        Returns:
            dr  : [N, N, 3]  displacement vectors dr[i,j] = R_j - R_i (nearest image)
            rij : [N, N]     scalar distances |dr[i,j]|
        """
        N = tf.shape(R)[0]
        Ri_t = tf.reshape(tf.tile(R, [N, 1]), [N, N, 3])
        Rj_t = tf.transpose(Ri_t, perm=[1, 0, 2])
        dr_flat = self._minimum_image_displacement(tf.reshape(Ri_t, [-1, 3]),
                                              tf.reshape(Rj_t, [-1, 3]),
                                              box)
        dr = tf.reshape(dr_flat, [N, N, 3])
        rij = tf.linalg.norm(dr, axis=-1)
        return dr, rij

    # TODO: cutoff, radial_basis, and angular_basis were removed — descriptors
    # are now computed entirely by quippy SOAP-turbo via build_descriptors().