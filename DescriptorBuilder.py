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
    args: tuple,
) -> tuple[np.ndarray, list[np.ndarray], list[list[int]]]:
    """Compute SOAP descriptors for one structure inside a worker thread.

    Reuses thread-local Descriptor objects across calls — the previous version
    rebuilt them per frame, which dominated runtime for trajectory inference.

    args = (structure, soap_strings) or (structure, soap_strings, do_grad)
        or (structure, soap_strings, do_grad, skip_indices).
    do_grad defaults to True. `skip_indices` is a tuple of builder
    indices to NOT call (used for skip_h_centers).
    """
    if len(args) == 2:
        structure, soap_strings = args
        do_grad = True
        skip_indices: tuple[int, ...] = ()
    elif len(args) == 3:
        structure, soap_strings, do_grad = args
        skip_indices = ()
    else:
        structure, soap_strings, do_grad, skip_indices = args
        skip_indices = tuple(skip_indices)

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
    # Skip the builders requested by the caller (e.g. H-center skip).
    # These builders' descriptors and gradients never compute.
    outs = [b.calc(structure, grad=do_grad)
            for i, b in enumerate(builders) if i not in skip_indices]

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
        if do_grad and out.get("grad_index_0based") is not None:
            for j in range(len(out["grad_index_0based"])):
                center    = out["grad_index_0based"][j][0]
                neighbour = out["grad_index_0based"][j][1]
                gradients[center].append(out["grad_data"][j])
                grad_indexes[center].append(neighbour)

    # When some builders were skipped, atoms in the skipped types have
    # no descriptor row populated. Pre-fill those rows with zeros so
    # the np.array(...) conversion below doesn't trip on an inhomogeneous
    # list shape. We get Q from the first non-empty row.
    if skip_indices:
        dim_q = next((d_list[0].shape[0] for d_list in descriptors if d_list),
                     None)
        if dim_q is not None:
            for i, d_list in enumerate(descriptors):
                if not d_list:
                    descriptors[i] = [np.zeros(dim_q, dtype=np.float32)]

    descriptors_np = np.array(descriptors, dtype=np.float32).squeeze(axis=1)
    Q = descriptors_np.shape[-1]
    if do_grad:
        # Empty per-atom gradient lists must keep the [0, 3, Q] shape;
        # bare np.array([]) would give shape (0,) and break downstream
        # concat/stack ops.
        gradients_np = [
            (np.array(g, dtype=np.float32) if len(g) > 0
             else np.zeros((0, 3, Q), dtype=np.float32))
            for g in gradients
        ]
    else:
        gradients_np = [np.zeros((0, 3, Q), dtype=np.float32) for _ in range(N)]
    return descriptors_np, gradients_np, grad_indexes


def _describe_structure_worker_flat(
    structure: Atoms,
    soap_strings: list[str],
    dim_q: int,
    do_grad: bool = True,
    skip_indices: tuple[int, ...] = (),
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
    # Skip the builders requested by the caller. Atoms in skipped types
    # get zero descriptors (because `descriptors` is pre-allocated with
    # np.zeros) and have no associated gradient pairs in the output COO.
    outs = [b.calc(structure, grad=do_grad)
            for i, b in enumerate(builders) if i not in skip_indices]

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

        # When `cfg.skip_h_centers` is True, identify which entry in
        # `self.builders` corresponds to H (Z=1) and add it to the
        # skip list. Workers will not call b.calc() for these indices,
        # so the H descriptor and H-centered gradients are never
        # computed. H atoms still appear as neighbours in every other
        # builder's pair list through species_Z, so their geometric/
        # species info still feeds non-H centers.
        self._skip_builder_indices: tuple[int, ...] = ()
        if bool(getattr(cfg, "skip_h_centers", False)):
            self._skip_builder_indices = tuple(
                i for i, z in enumerate(self.types) if int(z) == 1)

    def build_descriptors(
        self,
        dataset: list[Atoms],
        calc_gradients: bool = True,
        batch_frames: int | None = 1,
    ) -> tuple[list[tf.Tensor], list[list[tf.Tensor]], list[list[list[int]]]]:
        """Compute SOAP descriptors and (optionally) their gradients.

        Runs each per-type quippy Descriptor with `grad=calc_gradients`, then
        collects results per centre atom.

        Args:
            dataset        : list of ase.Atoms structures
            calc_gradients : if False, gradients/grad_index are returned as
                             empty per-atom lists (saves the quippy gradient
                             calculation, which is the dominant cost).

        Returns:
            dataset_descriptors : list of tensors, one per structure [N, dim_q]
            dataset_gradients   : per-structure list of N tensors
                                  ([M_i, 3, dim_q] when calc_gradients=True,
                                  [0, 3, dim_q] when False)
            dataset_grad_index  : per-structure list of N lists of int neighbour
                                  indices (empty when calc_gradients=False)
        """
        dataset_descriptors = []
        dataset_gradients   = []
        dataset_grad_index  = []

        if self._num_workers <= 1:
            # Serial path
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
                # Skip the configured-out builders (e.g. H-center skip).
                # Those atoms get zero descriptor rows pre-filled below.
                _skip = self._skip_builder_indices
                outs = [b.calc(structure, grad=calc_gradients)
                        for i, b in enumerate(self.builders) if i not in _skip]

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
                    if calc_gradients and out.get("grad_index_0based") is not None:
                        for j in range(len(out["grad_index_0based"])):
                            center    = out["grad_index_0based"][j][0]
                            neighbour = out["grad_index_0based"][j][1]
                            gradients[center].append(out["grad_data"][j])
                            grad_indexes[center].append(neighbour)

                # Pre-fill rows for skipped-builder atoms with zeros so
                # the convert-to-tensor / squeeze chain below doesn't
                # trip on an inhomogeneous list shape.
                if _skip:
                    dim_q_local = next(
                        (d_list[0].shape[0] for d_list in descriptors if d_list),
                        None)
                    if dim_q_local is not None:
                        for i, d_list in enumerate(descriptors):
                            if not d_list:
                                descriptors[i] = [np.zeros(dim_q_local,
                                                            dtype=np.float32)]

                descriptors = tf.convert_to_tensor(descriptors, dtype=tf.float32)
                descriptors = tf.squeeze(descriptors, axis=1)
                dim_q = int(descriptors.shape[-1])
                if calc_gradients:
                    # Empty per-atom gradient lists become [0,3,Q] tensors
                    # (bare convert_to_tensor on `[]` gives shape (0,) which
                    # breaks downstream concat/stack with `[M,3,Q]` blocks).
                    for i in range(len(gradients)):
                        if len(gradients[i]) > 0:
                            gradients[i] = tf.convert_to_tensor(
                                gradients[i], dtype=tf.float32)
                        else:
                            gradients[i] = tf.zeros(
                                (0, 3, dim_q), dtype=tf.float32)
                else:
                    gradients = [tf.zeros((0, 3, dim_q), dtype=tf.float32) for _ in range(N)]
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

            _skip_indices_local = self._skip_builder_indices

            def _thread_worker(structure):
                return _describe_structure_worker(
                    (structure, soap_strings, calc_gradients, _skip_indices_local))

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
        self,
        dataset: list[Atoms],
        calc_gradients: bool = True,
        batch_frames: int | None = 1,
        memory_budget_bytes: int | None = None,
    ) -> list[tuple]:
        """Compute SOAP descriptors and return flat per-frame COO arrays.

        Args:
            dataset             : list of ase.Atoms
            calc_gradients      : when False, gradient outputs are zero-length
                                  arrays (saves the quippy gradient
                                  computation, the dominant cost).
            batch_frames        : ignored (quippy is per-frame). Accepted for
                                  signature parity with the TF GPU backend.
            memory_budget_bytes : ignored (quippy is per-frame). Accepted for
                                  signature parity with the TF GPU backend.

        Returns:
            list of (descriptors, grad_values, pair_atom, pair_gidx) tuples
            per structure. All arrays are numpy:
                descriptors : [N, dim_q]   float32
                grad_values : [P, 3, dim_q] (or [0,3,dim_q]) float32
                pair_atom   : [P] (or [0]) int32
                pair_gidx   : [P] (or [0]) int32
        """
        soap_strings = self._soap_strings
        dim_q = self.cfg.dim_q

        _skip = self._skip_builder_indices

        if self._num_workers <= 1:
            return [_describe_structure_worker_flat(
                        s, soap_strings, dim_q,
                        do_grad=calc_gradients, skip_indices=_skip)
                    for s in dataset]

        _total_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
        omp_per_worker = max(1, _total_cpus // self._num_workers)

        def _thread_worker(structure):
            return _describe_structure_worker_flat(
                structure, soap_strings, dim_q,
                do_grad=calc_gradients, skip_indices=_skip)

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



# =========================================================================
# Backend dispatcher
# =========================================================================


def make_descriptor_builder(cfg: TNEPconfig, mode: int | None = None):
    """Return the descriptor builder selected by `cfg.descriptor_mode` or `mode`.

    Args:
        cfg  : TNEPconfig instance (provides default mode + hyperparameters)
        mode : optional override (None → use cfg.descriptor_mode)

    Returns:
        - mode 0 : DescriptorBuilder (quippy / Fortran, CPU)
        - mode 1 : DescriptorBuilderGPUTF (TF / GPU, native port)

    The GPU backend is imported lazily so quippy-only deployments don't pull
    in TF on every import of this module.
    """
    selected = cfg.descriptor_mode if mode is None else int(mode)
    if selected == 0:
        return DescriptorBuilder(cfg)
    if selected == 1:
        # Local import to avoid circular dependency at module load time
        from DescriptorBuilderGPU_tf import DescriptorBuilderGPUTF
        return DescriptorBuilderGPUTF(cfg)
    raise ValueError(
        f"Unknown descriptor_mode={selected}. Use 0 (quippy) or 1 (GPU TF)."
    )
