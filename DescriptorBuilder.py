from __future__ import annotations

import os
import threading
import concurrent.futures
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from TNEPconfig import TNEPconfig
from quippy.descriptors import Descriptor

from ase import Atoms


# Per-thread cache of quippy Descriptor objects, keyed by SOAP-string tuple.
# Building Descriptors is expensive (Fortran/C state allocation); caching per
# thread avoids rebuilding for every frame while keeping each thread's
# Descriptor objects isolated (quippy is not guaranteed thread-safe across
# concurrent .calc() calls on the same object).
_thread_local = threading.local()


def _get_thread_builders(soap_strings: list[str]) -> list:
    """Return this thread's cached Descriptor objects, building them on first use."""
    cache = getattr(_thread_local, "cache", None)
    if cache is None:
        cache = _thread_local.cache = {}
    key = tuple(soap_strings)
    builders = cache.get(key)
    if builders is None:
        builders = [Descriptor(s) for s in soap_strings]
        cache[key] = builders
    return builders


def _describe_structure_worker(
    args: tuple[Atoms, list[str]],
) -> tuple[np.ndarray, list[np.ndarray], list[list[int]]]:
    """Compute SOAP descriptors for one structure inside a worker thread.

    Reuses thread-local Descriptor objects across calls — the previous version
    rebuilt them per frame, which dominated runtime for trajectory inference.
    """
    structure, soap_strings = args

    cell = structure.cell.array
    if np.allclose(cell, 0) or abs(np.linalg.det(cell)) < 1e-6:
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

    builders = _get_thread_builders(soap_strings)
    outs = [b.calc(structure, grad=True) for b in builders]

    N = len(structure)
    descriptors  = [[] for _ in range(N)]
    gradients    = [[] for _ in range(N)]
    grad_indexes = [[] for _ in range(N)]

    for out in outs:
        data = out.get("data")
        if data is None or data.size == 0 or data.shape[1] == 0:
            continue
        for k in range(len(out["ci"])):
            descriptors[out["ci"][k] - 1].append(out["data"][k])
        for j in range(len(out["grad_index_0based"])):
            center    = out["grad_index_0based"][j][0]
            neighbour = out["grad_index_0based"][j][1]
            gradients[center].append(out["grad_data"][j])
            grad_indexes[center].append(neighbour)

    descriptors_np = np.array(descriptors, dtype=np.float32).squeeze(axis=1)
    gradients_np   = [np.array(g, dtype=np.float32) for g in gradients]
    return descriptors_np, gradients_np, grad_indexes


def _describe_structure_worker_flat(
    structure: Atoms,
    soap_strings: list[str],
    dim_q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SOAP descriptors and return flat COO arrays directly.

    Skips the per-atom bucketisation that the legacy worker does — quippy
    already gives us per-pair gradients and per-atom descriptors, and our
    consumer (_pack_traj_batch_from_flat) wants flat COO. Avoids ~65,000
    Python-level appends per batch on a 100-frame × 655-atom trajectory.

    Returns:
        descriptors : [N, dim_q]      float32
        grad_values : [P, 3, dim_q]   float32 — one entry per (centre, neighbour) pair
        pair_atom   : [P]             int32   — centre atom index (0-based)
        pair_gidx   : [P]             int32   — neighbour atom index (0-based)
    """
    cell = structure.cell.array
    if np.allclose(cell, 0) or abs(np.linalg.det(cell)) < 1e-6:
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
    N = len(structure)

    builders = _get_thread_builders(soap_strings)
    outs = [b.calc(structure, grad=True) for b in builders]

    descriptors = np.zeros((N, dim_q), dtype=np.float32)
    grad_chunks = []
    pair_atom_chunks = []
    pair_gidx_chunks = []

    for out in outs:
        data = out.get("data")
        if data is None or data.size == 0 or data.shape[1] == 0:
            continue
        # ci is 1-based; data is [N_type, dim_q] — vectorised assign by centre atom
        ci = np.asarray(out["ci"], dtype=np.int32) - 1
        descriptors[ci] = np.asarray(data, dtype=np.float32)

        grad_idx = out.get("grad_index_0based")
        if grad_idx is not None and len(grad_idx) > 0:
            grad_idx = np.asarray(grad_idx, dtype=np.int32)
            grad_chunks.append(np.asarray(out["grad_data"], dtype=np.float32))
            pair_atom_chunks.append(grad_idx[:, 0])
            pair_gidx_chunks.append(grad_idx[:, 1])

    if grad_chunks:
        grad_values = np.concatenate(grad_chunks, axis=0)
        pair_atom   = np.concatenate(pair_atom_chunks)
        pair_gidx   = np.concatenate(pair_gidx_chunks)
    else:
        grad_values = np.zeros((0, 3, dim_q), dtype=np.float32)
        pair_atom   = np.zeros(0, dtype=np.int32)
        pair_gidx   = np.zeros(0, dtype=np.int32)

    return descriptors, grad_values, pair_atom, pair_gidx


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

        self._soap_strings = [
            base + f" central_index={k}"
            for k in (np.arange(self.num_types, dtype=int) + 1)
        ]
        self.builders = [Descriptor(s) for s in self._soap_strings]

        if cfg.num_descriptor_workers is None:
            _slurm = os.environ.get('SLURM_CPUS_PER_TASK')
            self._num_workers = int(_slurm) if _slurm else max(os.cpu_count() // 2, 1)
        else:
            self._num_workers = cfg.num_descriptor_workers

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
        dataset_gradients   = []
        dataset_grad_index  = []

        if self._num_workers <= 1:
            # Serial path — unchanged from original
            for structure in dataset:
                cell = structure.cell.array
                if np.allclose(cell, 0) or abs(np.linalg.det(cell)) < 1e-6:
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

                descriptors  = [[] for _ in range(len(structure))]
                gradients    = [[] for _ in range(len(structure))]
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
                        center    = out["grad_index_0based"][j][0]
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

        else:
            _total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
            omp_per_worker = max(1, _total_cpus // self._num_workers)

            # ThreadPoolExecutor avoids multiprocessing pickling entirely.
            # quippy's Descriptor.calc() is a C extension that releases the GIL,
            # so threads run in true parallel for the compute-heavy part.
            # soap_strings is captured by closure — no serialisation needed.
            soap_strings = self._soap_strings

            def _thread_worker(structure):
                return _describe_structure_worker((structure, soap_strings))

            # OMP_NUM_THREADS is process-wide; setting it before the pool starts
            # means each thread's first quippy OMP context picks up omp_per_worker.
            old_omp = os.environ.get('OMP_NUM_THREADS')
            os.environ['OMP_NUM_THREADS'] = str(omp_per_worker)
            try:
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self._num_workers) as exe:
                    results = list(exe.map(_thread_worker, dataset))
            finally:
                if old_omp is not None:
                    os.environ['OMP_NUM_THREADS'] = old_omp
                else:
                    os.environ.pop('OMP_NUM_THREADS', None)

            for descriptors_np, gradients_np, grad_indexes in results:
                dataset_descriptors.append(tf.convert_to_tensor(descriptors_np, dtype=tf.float32))
                dataset_gradients.append(
                    [tf.convert_to_tensor(g, dtype=tf.float32) for g in gradients_np]
                )
                dataset_grad_index.append(grad_indexes)

        return dataset_descriptors, dataset_gradients, dataset_grad_index

    def build_descriptors_flat(
        self, dataset: list[Atoms],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Compute SOAP descriptors and return flat per-frame COO arrays.

        Optimised for trajectory inference. Skips the per-atom bucketisation
        and tf.Tensor wrapping that the standard build_descriptors() path does
        for compatibility with the training pipeline.

        Args:
            dataset : list of ase.Atoms

        Returns:
            list of (descriptors, grad_values, pair_atom, pair_gidx) tuples,
            one per structure. All arrays are numpy:
                descriptors : [N, dim_q]      float32
                grad_values : [P, 3, dim_q]   float32
                pair_atom   : [P]             int32
                pair_gidx   : [P]             int32
        """
        soap_strings = self._soap_strings
        dim_q = self.cfg.dim_q

        if self._num_workers <= 1:
            return [_describe_structure_worker_flat(s, soap_strings, dim_q)
                    for s in dataset]

        _total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
        omp_per_worker = max(1, _total_cpus // self._num_workers)

        def _thread_worker(structure):
            return _describe_structure_worker_flat(structure, soap_strings, dim_q)

        old_omp = os.environ.get('OMP_NUM_THREADS')
        os.environ['OMP_NUM_THREADS'] = str(omp_per_worker)
        try:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._num_workers) as exe:
                return list(exe.map(_thread_worker, dataset))
        finally:
            if old_omp is not None:
                os.environ['OMP_NUM_THREADS'] = old_omp
            else:
                os.environ.pop('OMP_NUM_THREADS', None)

