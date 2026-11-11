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


# Per-thread quippy Descriptor cache: .calc() is not thread-safe on a shared object.
_thread_local = threading.local()


def _allocated_cpus() -> int:
    """CPUs granted to this process (twin of MasterTNEP._allocated_cpu_count):
    Slurm vars, then the cpuset mask, else cpu_count()//2 — os.cpu_count()
    ignores the cgroup and over-subscribes partial-node allocations."""
    for _var in ('SLURM_CPUS_PER_TASK', 'SLURM_CPUS_ON_NODE'):
        _val = os.environ.get(_var)
        if _val:
            try:
                return max(int(_val.split('(')[0]), 1)   # guard "128(x2)"
            except ValueError:
                pass
    if os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOBID'):
        try:
            return max(len(os.sched_getaffinity(0)), 1)
        except AttributeError:                           # non-Linux
            pass
    return max((os.cpu_count() or 2) // 2, 1)


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
    """One structure in a worker thread, args = (structure, soap_strings[, do_grad=True])
    → (descriptors [N, Q], per-atom gradients [M_i, 3, Q], per-atom neighbour lists)."""
    if len(args) == 2:
        structure, soap_strings = args
        do_grad = True
    else:
        structure, soap_strings, do_grad = args

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
    outs = [b.calc(structure, grad=do_grad) for b in builders]

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

    descriptors_np = np.array(descriptors, dtype=np.float32).squeeze(axis=1)
    Q = descriptors_np.shape[-1]
    if do_grad:
        # Empty lists must stay [0, 3, Q]; np.array([]) is (0,) and breaks concat.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One structure → flat COO (no per-atom bucketing): descriptors [N, Q],
    grad_values [P, 3, Q], pair_atom [P], pair_gidx [P] (0-based)."""
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
    outs = [b.calc(structure, grad=do_grad) for b in builders]

    descriptors = np.zeros((N, dim_q), dtype=np.float32)
    grad_chunks = []
    pair_atom_chunks = []
    pair_gidx_chunks = []

    for out in outs:
        data = out.get("data")
        if data is None or data.size == 0 or data.shape[1] == 0:
            continue
        ci = np.asarray(out["ci"], dtype=np.int32) - 1  # ci is 1-based
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


def _filter_to_self_pairs(
    descriptors_per_struct: list,
    gradients_per_struct: list,
    grad_index_per_struct: list,
) -> tuple[list, list, list]:
    """Reduce each atom's gradients to the self row ∂q_i/∂r_i: [1, 3, Q] and
    grad_index [i] (a zero row when no self entry). Descriptors pass through."""
    new_grads = []
    new_gidx  = []
    for s, (grads, gidx) in enumerate(
            zip(gradients_per_struct, grad_index_per_struct)):
        atom_grads_kept: list = []
        atom_gidx_kept:  list = []
        for i, (g_i, idx_i) in enumerate(zip(grads, gidx)):
            try:
                k_self = idx_i.index(i) if isinstance(idx_i, list) \
                    else int(np.where(np.asarray(idx_i) == i)[0][0])
            except (ValueError, IndexError):
                dim_q = (g_i.shape[-1] if hasattr(g_i, "shape")
                         else descriptors_per_struct[s].shape[-1])
                atom_grads_kept.append(tf.zeros((1, 3, dim_q),
                                                 dtype=tf.float32))
                atom_gidx_kept.append([i])
                continue
            if hasattr(g_i, "shape") and len(g_i.shape) == 3:
                row = g_i[k_self:k_self + 1]
            else:
                row = tf.convert_to_tensor(
                    np.asarray(g_i)[k_self:k_self + 1], dtype=tf.float32)
            atom_grads_kept.append(row)
            atom_gidx_kept.append([i])
        new_grads.append(atom_grads_kept)
        new_gidx.append(atom_gidx_kept)
    return descriptors_per_struct, new_grads, new_gidx


class DescriptorBuilder(layers.Layer):
    """SOAP-turbo descriptors + gradients via quippy, one Descriptor per centre type."""

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
            # "none" = keyword omitted: soap_turbo rejects the literal
            # (soap_turbo_compress.f90:292).
            + (f"compress_mode={cfg.compress_mode} "
               if str(cfg.compress_mode) != "none" else "")
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

        self._num_workers = _allocated_cpus()

    def build_descriptors(
        self,
        dataset: list[Atoms],
        calc_gradients: bool = True,
        batch_frames: int | None = 1,
    ) -> tuple[list[tf.Tensor], list[list[tf.Tensor]], list[list[list[int]]]]:
        """Per structure: descriptors [N, Q], N gradient tensors [M_i, 3, Q]
        ([0, 3, Q] when calc_gradients=False, which skips the dominant cost),
        N neighbour-index lists."""
        dataset_descriptors = []
        dataset_gradients   = []
        dataset_grad_index  = []

        if self._num_workers <= 1:
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
                outs = [b.calc(structure, grad=calc_gradients)
                        for b in self.builders]

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

                descriptors = tf.convert_to_tensor(descriptors, dtype=tf.float32)
                descriptors = tf.squeeze(descriptors, axis=1)
                dim_q = int(descriptors.shape[-1])
                if calc_gradients:
                    # Empty lists must become [0, 3, Q], not (0,).
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
            _total_cpus = _allocated_cpus()
            omp_per_worker = max(1, _total_cpus // self._num_workers)

            # quippy .calc() releases the GIL: threads give true parallelism, no pickling.
            soap_strings = self._soap_strings

            def _thread_worker(structure):
                return _describe_structure_worker(
                    (structure, soap_strings, calc_gradients))

            # OMP_NUM_THREADS set before the pool so each worker nests omp_per_worker threads.
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

    def build_descriptors_self_only(
        self,
        dataset: list[Atoms],
        batch_size: int | None = 100,
        **build_kwargs,
    ) -> tuple[list[tf.Tensor], list[list[tf.Tensor]], list[list[list[int]]]]:
        """build_descriptors in chunks of batch_size, each filtered to the self
        gradient ([1, 3, Q], grad_index[i] == [i]) before the next chunk
        allocates, bounding peak memory. batch_size=None: single shot."""
        from data import soap_gradients_needed
        if not soap_gradients_needed(self.cfg):
            return self.build_descriptors(dataset, calc_gradients=False,
                                          **build_kwargs)
        if batch_size is None or batch_size >= len(dataset):
            descs, grads, gidx = self.build_descriptors(dataset, **build_kwargs)
            return _filter_to_self_pairs(descs, grads, gidx)

        out_descs: list = []
        out_grads: list = []
        out_gidx:  list = []
        n = len(dataset)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            chunk = dataset[start:end]
            chunk_descs, chunk_grads, chunk_gidx = self.build_descriptors(
                chunk, **build_kwargs)
            filtered_descs, filtered_grads, filtered_gidx = _filter_to_self_pairs(
                chunk_descs, chunk_grads, chunk_gidx)
            out_descs.extend(filtered_descs)
            out_grads.extend(filtered_grads)
            out_gidx.extend(filtered_gidx)
            del chunk_descs, chunk_grads, chunk_gidx
            del filtered_descs, filtered_grads, filtered_gidx
        return out_descs, out_grads, out_gidx

    def build_descriptors_flat(
        self,
        dataset: list[Atoms],
        calc_gradients: bool = True,
        batch_frames: int | None = 1,
        memory_budget_bytes: int | None = None,
    ) -> list[tuple]:
        """Per structure a numpy (descriptors [N, Q], grad_values [P, 3, Q],
        pair_atom [P], pair_gidx [P]) tuple (P = 0 when calc_gradients=False).
        batch_frames / memory_budget_bytes are ignored (GPU-backend parity)."""
        soap_strings = self._soap_strings
        dim_q = self.cfg.dim_q

        if self._num_workers <= 1:
            return [_describe_structure_worker_flat(
                        s, soap_strings, dim_q,
                        do_grad=calc_gradients)
                    for s in dataset]

        _total_cpus = _allocated_cpus()
        omp_per_worker = max(1, _total_cpus // self._num_workers)

        def _thread_worker(structure):
            return _describe_structure_worker_flat(
                structure, soap_strings, dim_q,
                do_grad=calc_gradients)

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


# ── backend dispatcher ──

class _SoapFreeBuilder:
    """Zero-width `descriptors` slot for nep_descriptor="replace", where nothing
    reads SOAP; everything else delegates, so a path that needs SOAP fails on
    the shape rather than reading zeros."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @staticmethod
    def _empty(dataset):
        """calc_gradients=False contract at Q = 0: [N, 0] descriptors, [0, 3, 0] per atom, empty lists."""
        import tensorflow as _tf
        descs, grads, gidx = [], [], []
        for st in dataset:
            n = len(st)
            descs.append(_tf.zeros((n, 0), dtype=_tf.float32))
            grads.append([_tf.zeros((0, 3, 0), dtype=_tf.float32) for _ in range(n)])
            gidx.append([[] for _ in range(n)])
        return descs, grads, gidx

    def build_descriptors(self, dataset, *a, **kw):
        return self._empty(dataset)

    def build_descriptors_self_only(self, dataset, *a, **kw):
        return self._empty(dataset)


def make_descriptor_builder(cfg: TNEPconfig, mode: int | None = None):
    """Builder for `mode` (default cfg.descriptor_mode): 0 → quippy DescriptorBuilder,
    1 → DescriptorBuilderGPUTF; wrapped in _SoapFreeBuilder when SOAP is unused."""
    selected = cfg.descriptor_mode if mode is None else int(mode)
    from descriptor_layout import soap_descriptors_needed
    wrap = (lambda b: b) if soap_descriptors_needed(cfg) else _SoapFreeBuilder
    if selected == 0:
        return wrap(DescriptorBuilder(cfg))
    if selected == 1:
        from DescriptorBuilderGPU_tf import DescriptorBuilderGPUTF  # circular at load
        return wrap(DescriptorBuilderGPUTF(cfg))
    raise ValueError(
        f"Unknown descriptor_mode={selected}. Use 0 (quippy) or 1 (GPU TF)."
    )
