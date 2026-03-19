from __future__ import annotations

import numpy as np
from TNEPconfig import TNEPconfig
from quippy.descriptors import Descriptor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ase import Atoms

class DescriptorBuilder:
    """Builds SOAP-turbo descriptors and their gradients using quippy.

    Constructs one quippy Descriptor per atom type (central_index), then
    aggregates per-atom descriptors, descriptor gradients, and neighbour
    indices for each structure in a dataset.
    """

    def __init__(self,
                 cfg: TNEPconfig) -> None:

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

    def build_descriptors(self, dataset: list[Atoms]) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[list[list[int]]]]:
        """Compute SOAP descriptors and their gradients for every structure.

        Runs each per-type quippy Descriptor with grad=True, then collects
        results per centre atom.

        Args:
            dataset : list of ase.Atoms structures

        Returns:
            dataset_descriptors : list of arrays, one per structure
                Each array has shape [N, dim_q].
            dataset_gradients   : list of (list of N arrays), one per structure
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
                gradients[i] = np.array(gradients[i], dtype=np.float32)

            descriptors = np.array(descriptors, dtype=np.float32)
            descriptors = np.squeeze(descriptors, axis=1)
            dataset_descriptors.append(descriptors)
            dataset_gradients.append(gradients)
            dataset_grad_index.append(grad_indexes)
        return dataset_descriptors, dataset_gradients, dataset_grad_index
