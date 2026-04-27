from __future__ import annotations

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from TNEPconfig import TNEPconfig
from quippy.descriptors import Descriptor

from ase import Atoms

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
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.cfg = cfg
        self.types = cfg.types
        self.num_types = cfg.num_types

        base = (
            f"soap_turbo l_max={cfg.l_max} "
            f"rcut_hard={cfg.rcut_hard} rcut_soft={cfg.rcut_soft} "
            f"basis={cfg.basis} scaling_mode={cfg.scaling_mode} "
            f"add_species=F "
            f"radial_enhancement={cfg.radial_enhancement} "
            f"compress_mode={cfg.compress_mode} "
        )
        if cfg.compress_P is not None:
            base += f"compress_P={cfg.compress_P} "

        alpha_max = " alpha_max={"
        atom_sigma_r = " atom_sigma_r={"
        atom_sigma_t = " atom_sigma_t={"
        atom_sigma_r_scaling = " atom_sigma_r_scaling={"
        atom_sigma_t_scaling = " atom_sigma_t_scaling={"
        amplitude_scaling = " amplitude_scaling={"
        central_weight = " central_weight={"

        for a in range(self.num_types):
            alpha_max += f"{cfg.alpha_max} "
            atom_sigma_r += f"{cfg.atom_sigma_r} "
            atom_sigma_t += f"{cfg.atom_sigma_t} "
            atom_sigma_r_scaling += f"{cfg.atom_sigma_r_scaling} "
            atom_sigma_t_scaling += f"{cfg.atom_sigma_t_scaling} "
            amplitude_scaling += f"{cfg.amplitude_scaling} "
            central_weight += f"{cfg.central_weight} "

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

    def build_descriptors(self, dataset: list[Atoms]) -> tuple[list[tf.Tensor], list[list[tf.Tensor]], list[list[list[int]]]]:
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
            cell = structure.cell.array
            if np.allclose(cell, 0) or abs(np.linalg.det(cell)) < 1e-6:
                # Zero or degenerate cell: quippy cannot build a valid PBC neighbour
                # list and returns NaN descriptor gradients.  Replace with a large
                # dummy box (same logic as cell_to_box in data.py) and disable PBC.
                cell = 1000.0 * np.eye(3, dtype=np.float32)
                pbc = False
            else:
                pbc = structure.pbc
            structure = Atoms(
                numbers=structure.numbers,
                positions=structure.positions,
                cell=cell,
                pbc=pbc,
            )
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

