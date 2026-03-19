from __future__ import annotations

import torch
import numpy as np
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from ase.io import read
from ase import Atoms


def collect(cfg: TNEPconfig) -> tuple[list[Atoms], list[np.ndarray]]:
    """Load all structures from train.xyz and assign integer type indices.

    Populates cfg.num_types and cfg.types as a side effect.

    Args:
        cfg : TNEPconfig with data_path set

    Returns:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray [N_i] — integer type index per atom
    """
    dataset = read(cfg.data_path, index=":")
    dataset_types_int = []
    types = []

    for structure in dataset:
        structure_types_int = np.zeros_like(structure.numbers)

        for i in range(len(structure.numbers)):
            z = structure.numbers[i]
            if z not in types:
                types.append(z)
            structure_types_int[i] = types.index(z)

        dataset_types_int.append(structure_types_int)

    cfg.num_types = len(types)
    cfg.types = types
    print("Number of species in raw dataset: " + str(cfg.num_types))
    print("Number of structures in raw dataset: " + str(len(dataset)))

    # Filter by species if configured
    if cfg.allowed_species is not None:
        dataset, dataset_types_int = filter_by_species(dataset, dataset_types_int, allowed_Z=cfg.allowed_species)
        print("After species filter: " + str(len(dataset)) + " structures")

    # Recompute type list and indices after species filtering
    cfg.types = []
    for struct in dataset:
        for z in struct.numbers:
            if z not in cfg.types:
                cfg.types.append(z)
    cfg.num_types = len(cfg.types)
    dataset_types_int = assign_type_indices(dataset, cfg.types)
    print("Species: " + str(cfg.types) + " (" + str(cfg.num_types) + " types)")

    # Filter bad data based on config flags
    dataset, dataset_types_int = filter_bad_data(dataset, dataset_types_int, cfg)

    return dataset, dataset_types_int


def assign_type_indices(dataset: list[Atoms], types: list[int]) -> list[np.ndarray]:
    """Map atoms to type indices using a known type list.

    Unlike collect(), this does not discover types — it uses the provided
    type list (e.g. from a trained model's cfg.types) to assign indices.

    Args:
        dataset : list of ase.Atoms
        types   : list of atomic numbers defining the type ordering

    Returns:
        dataset_types_int : list of ndarray [N_i] — integer type index per atom
    """
    dataset_types_int = []
    for struct in dataset:
        structure_types_int = np.zeros_like(struct.numbers)
        for i, z in enumerate(struct.numbers):
            structure_types_int[i] = types.index(z)
        dataset_types_int.append(structure_types_int)
    return dataset_types_int


def _extract_target(structure: Atoms, target_key: str) -> np.ndarray:
    """Extract target, converting 9-component polarizability to 6-component if needed."""
    if target_key in structure.info:
        raw = np.asarray(structure.info[target_key], dtype=np.float32)
    elif structure.calc is not None and target_key in structure.calc.results:
        raw = np.asarray(structure.calc.results[target_key], dtype=np.float32)
    else:
        raise KeyError(f"'{target_key}' not found in structure.info or calc.results")
    if raw.size == 9:
        # Flattened 3x3 row-major -> unique [xx, yy, zz, xy, yz, zx]
        raw = raw[[0, 4, 8, 1, 5, 6]]
    return np.asarray(raw, dtype=np.float32)


def find_bad_data(dataset: list[Atoms], target_key: str) -> dict[str, list[int]]:
    """Find structures with NaN values or zero-vector targets.

    Args:
        dataset    : list of ase.Atoms
        target_key : key for the target property (e.g. 'dipole', 'pol')

    Returns:
        dict with keys 'nan_positions', 'nan_targets', 'zero_targets',
        each mapping to a list of structure indices.
    """
    nan_positions = []
    nan_targets = []
    zero_targets = []
    missing_targets = []
    for i, structure in enumerate(dataset):
        if np.any(np.isnan(structure.positions)):
            nan_positions.append(i)
        try:
            target = _extract_target(structure, target_key)
        except KeyError:
            missing_targets.append(i)
            continue
        if np.any(np.isnan(target)):
            nan_targets.append(i)
        if np.allclose(target, 0.0):
            zero_targets.append(i)
    return {'nan_positions': nan_positions,
            'nan_targets': nan_targets,
            'zero_targets': zero_targets,
            'missing_targets': missing_targets}


def _gpaw_dipole_worker(args):
    """Compute GPAW dipole for a single structure in a worker process."""
    import os
    os.environ['OMP_NUM_THREADS'] = str(args['omp_threads'])
    os.environ['MKL_NUM_THREADS'] = str(args['omp_threads'])
    from gpaw import GPAW
    mol = args['mol']
    calc = GPAW(mode='lcao', xc='LDA', basis='dzp', h=0.4, txt=None,
                convergence={'energy': 0.01, 'density': 0.01},
                occupations={'name': 'fermi-dirac', 'width': 0.2},
                maxiter=50)
    mol.center(vacuum=4.0)
    return calc.get_dipole_moment(mol)


def find_rigorous_bad_data(
    dataset: list[Atoms],
    target_key: str,
    threshold: float,
) -> list[int]:
    """Recompute targets with GPAW and find structures with low cosine similarity.

    Runs parallel GPAW dipole calculations, compares against dataset targets
    using cosine similarity, and returns indices below the threshold.

    Args:
        dataset    : list of ase.Atoms
        target_key : key for the target property
        threshold  : cosine similarity threshold — structures below this are flagged

    Returns:
        list of structure indices with cosine similarity < threshold
    """
    import os
    from multiprocessing import Pool
    from tqdm import tqdm

    logical = os.cpu_count() or 1
    n_workers = max(logical // 2, 1)
    omp_per_worker = 1

    worker_args = [{'mol': s.copy(), 'omp_threads': omp_per_worker} for s in dataset]
    print(f"  Rigorous filter: running {len(dataset)} GPAW calculations with "
          f"{n_workers} workers ({omp_per_worker} OMP threads each)")

    with Pool(n_workers) as pool:
        target_calcs = list(tqdm(pool.imap(_gpaw_dipole_worker, worker_args),
                                 total=len(worker_args), desc="  GPAW dipoles"))

    target_actuals = [_extract_target(s, target_key) for s in dataset]
    target_calcs = np.array(target_calcs, dtype=np.float32)
    target_actuals = np.stack(target_actuals)

    dot = np.sum(target_calcs * target_actuals, axis=-1)
    norm = np.linalg.norm(target_calcs, axis=-1) * np.linalg.norm(target_actuals, axis=-1)
    similarity = dot / norm

    print(f"  Cosine similarity: mean={float(np.mean(similarity)):.4f}  "
          f"min={float(np.min(similarity)):.4f}")

    bad_indices = [i for i in range(len(dataset)) if float(similarity[i]) < threshold]
    return bad_indices


def filter_bad_data(
    dataset: list[Atoms],
    dataset_types_int: list[np.ndarray],
    cfg: TNEPconfig,
) -> tuple[list[Atoms], list[np.ndarray]]:
    """Remove structures with bad data based on config flags.

    Structures with missing targets are always removed regardless of config,
    since they cannot be used for training.

    Args:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray [N_i] — integer type index per atom
        cfg               : TNEPconfig with filter_nan_positions, filter_nan_targets,
                            filter_zero_targets, filter_rigorous flags

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    target_key = _target_key_for_mode(cfg.target_mode)
    bad = find_bad_data(dataset, target_key)

    bad_indices: set[int] = set()

    # Always remove structures missing the target key
    if bad['missing_targets']:
        bad_indices.update(bad['missing_targets'])
        print(f"  Filtering {len(bad['missing_targets'])} structures with missing '{target_key}' key")

    if not (cfg.filter_nan_positions or cfg.filter_nan_targets or cfg.filter_zero_targets
            or cfg.filter_rigorous) and not bad_indices:
        return dataset, dataset_types_int

    if cfg.filter_nan_positions:
        bad_indices.update(bad['nan_positions'])
        if bad['nan_positions']:
            print(f"  Filtering {len(bad['nan_positions'])} structures with NaN positions")
    if cfg.filter_nan_targets:
        bad_indices.update(bad['nan_targets'])
        if bad['nan_targets']:
            print(f"  Filtering {len(bad['nan_targets'])} structures with NaN targets")
    if cfg.filter_zero_targets:
        bad_indices.update(bad['zero_targets'])
        if bad['zero_targets']:
            print(f"  Filtering {len(bad['zero_targets'])} structures with zero targets")

    if cfg.filter_rigorous:
        rigorous_bad = find_rigorous_bad_data(dataset, target_key, cfg.rigorous_threshold)
        bad_indices.update(rigorous_bad)
        if rigorous_bad:
            print(f"  Filtering {len(rigorous_bad)} structures with cosine similarity < {cfg.rigorous_threshold}")

    if bad_indices:
        dataset = [s for i, s in enumerate(dataset) if i not in bad_indices]
        dataset_types_int = [t for i, t in enumerate(dataset_types_int) if i not in bad_indices]
        print(f"  Removed {len(bad_indices)} bad structures, {len(dataset)} remaining")

    return dataset, dataset_types_int


def _target_key_for_mode(target_mode: int) -> str:
    """Return the Atoms.info key for the given target mode."""
    return {0: "energy", 1: "dipole", 2: "pol"}[target_mode]


def component_labels(target_mode: int, num_components: int) -> list[str]:
    """Return human-readable labels for each target component.

    Args:
        target_mode    : 0 (PES), 1 (dipole), 2 (polarizability)
        num_components : number of output components (fallback for unknown modes)

    Returns:
        list of str labels, one per component
    """
    if target_mode == 0:
        return ["Energy"]
    elif target_mode == 1:
        return ["x", "y", "z"]
    elif target_mode == 2:
        return ["xx", "yy", "zz", "xy", "yz", "zx"]
    return [f"comp {i}" for i in range(num_components)]


def print_score_summary(metrics: dict, cfg: TNEPconfig, prefix: str = "") -> None:
    """Print RMSE, R², per-component R², and cosine similarity from a metrics dict.

    Args:
        metrics : dict from TNEP.score() with rmse, r2, r2_components, etc.
        cfg     : TNEPconfig (used for target_mode and component labels)
        prefix  : str prepended to the first line (e.g. "Model test set" or "External test")
    """
    rmse = float(metrics["rmse"])
    r2 = float(metrics["r2"])
    r2_comp = np.asarray(metrics["r2_components"])
    labels = component_labels(cfg.target_mode, len(r2_comp))

    if "total_rmse" in metrics:
        print(f"\n{prefix} (per-atom) RMSE: {rmse:.4f}")
        print(f"{prefix} (per-atom) R²:   {r2:.4f}")
        print("Per-atom per-component R²:  " + "  ".join(
            f"{lbl}={r2_comp[i]:.4f}" for i, lbl in enumerate(labels)))

        total_rmse = float(metrics["total_rmse"])
        total_r2 = float(metrics["total_r2"])
        total_r2_comp = np.asarray(metrics["total_r2_components"])
        print(f"{prefix} (total)    RMSE: {total_rmse:.4f}")
        print(f"{prefix} (total)    R²:   {total_r2:.4f}")
        print("Total per-component R²:     " + "  ".join(
            f"{lbl}={total_r2_comp[i]:.4f}" for i, lbl in enumerate(labels)))
    else:
        print(f"\n{prefix} RMSE: {rmse:.4f}")
        print(f"{prefix} R²:   {r2:.4f}")
        print("Per-component R²:  " + "  ".join(
            f"{lbl}={r2_comp[i]:.4f}" for i, lbl in enumerate(labels)))

    if "cos_sim_mean" in metrics:
        cos_mean = float(metrics["cos_sim_mean"])
        cos_all = np.asarray(metrics["cos_sim_all"])
        print(f"Cosine similarity:  mean={cos_mean:.4f}  "
              f"min={cos_all.min():.4f}  max={cos_all.max():.4f}  "
              f"std={cos_all.std():.4f}")


def assemble_data_dict(
    dataset: list[Atoms],
    types_int: list[np.ndarray],
    descriptors: list[np.ndarray],
    gradients: list[list[np.ndarray]],
    grad_index: list[list[list[int]]],
    cfg: TNEPconfig,
) -> dict:
    """Assemble a data dict from structures, type indices, and precomputed descriptors.

    Args:
        dataset     : list of ase.Atoms
        types_int   : list of ndarray [N_i] integer type indices
        descriptors : list of [N_i, dim_q] arrays
        gradients   : list of (list of N_i arrays each [M, 3, dim_q])
        grad_index  : list of (list of N_i lists each [M] ints)
        cfg         : TNEPconfig (uses target_mode)

    Returns:
        dict with keys: positions, Z_int, targets, boxes, descriptors, gradients, grad_index
    """
    target_key = _target_key_for_mode(cfg.target_mode)
    targets = [_extract_target(s, target_key) for s in dataset]
    if cfg.scale_targets and cfg.target_mode == 1:
        targets = [t / float(len(s)) for t, s in zip(targets, dataset)]
    return {
        "positions": [np.asarray(s.positions, dtype=np.float32) for s in dataset],
        "Z_int": [np.asarray(t, dtype=np.int64) for t in types_int],
        "targets": targets,
        "boxes": [np.asarray(s.cell.array, dtype=np.float32) for s in dataset],
        "descriptors": descriptors,
        "gradients": gradients,
        "grad_index": grad_index,
    }


def prepare_eval_data(dataset: list[Atoms], cfg: TNEPconfig) -> dict[str, torch.Tensor]:
    """Build type indices, descriptors, and collated data dict for evaluation.

    Convenience function that chains assign_type_indices → build_descriptors →
    assemble_data_dict → collate_flat.

    Args:
        dataset : list of ase.Atoms — structures to evaluate
        cfg     : TNEPconfig from training (carries types, descriptor params, target_mode)

    Returns:
        flat data dict ready for model.score()
    """
    types_int = assign_type_indices(dataset, cfg.types)
    builder = DescriptorBuilder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(dataset)
    if hasattr(cfg, '_descriptor_pca') and cfg._descriptor_pca is not None:
        descriptors, gradients = cfg._descriptor_pca.transform(descriptors, gradients)
    data = assemble_data_dict(dataset, types_int, descriptors, gradients, grad_index, cfg)
    return collate_flat(data)


def split(dataset: list[Atoms], dataset_types_int: list[np.ndarray], cfg: TNEPconfig) -> tuple[dict, dict, dict]:
    """Split dataset into train / test / validation and build SOAP descriptors.

    Uses cfg.indices (shuffled) and cfg.test_ratio to partition. Builds
    descriptors and gradients via DescriptorBuilder for each split.

    Args:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray [N_i] integer type indices
        cfg               : TNEPconfig

    Returns:
        train_data, test_data, val_data : dicts each containing:
            positions   : list of [N_i, 3] arrays
            Z_int       : list of [N_i] int arrays (type indices)
            targets     : list of target arrays (scalar for PES, [3] for dipole)
            boxes       : list of [3, 3] arrays (lattice vectors)
            descriptors : list of [N_i, dim_q] arrays
            gradients   : list of (list of N_i arrays each [M, 3, dim_q])
            grad_index  : list of (list of N_i lists each [M] ints)
    """

    indices = cfg.indices
    n_structures = len(indices)

    builder = DescriptorBuilder(cfg)

    if cfg.test_data_path is not None:
        # External test set: split data_path into train + val only
        n_val = int(cfg.test_ratio * n_structures)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        val_dataset = [dataset[i] for i in val_idx]
        train_dataset = [dataset[i] for i in train_idx]
        val_types_int = [dataset_types_int[i] for i in val_idx]
        train_types_int = [dataset_types_int[i] for i in train_idx]

        # Load and prepare external test set
        from ase.io import read as ase_read
        test_structures = ase_read(cfg.test_data_path, index=":")
        if cfg.allowed_species is not None:
            # Filter before type assignment — test file may contain unknown species
            from ase.data import atomic_numbers
            allowed = set(atomic_numbers[z] if isinstance(z, str) else z
                          for z in cfg.allowed_species)
            test_structures = [s for s in test_structures
                               if set(s.numbers).issubset(allowed)]
        test_dataset = test_structures
        test_types_int = assign_type_indices(test_dataset, cfg.types)
        print(f"External test set: {len(test_dataset)} structures from {cfg.test_data_path}")
    else:
        # Default: three-way split from data_path
        n_test = int(cfg.test_ratio * n_structures)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:(2*n_test)]
        train_idx = indices[(2*n_test):n_structures]

        test_dataset = [dataset[i] for i in test_idx]
        val_dataset = [dataset[i] for i in val_idx]
        train_dataset = [dataset[i] for i in train_idx]
        test_types_int = [dataset_types_int[i] for i in test_idx]
        val_types_int = [dataset_types_int[i] for i in val_idx]
        train_types_int = [dataset_types_int[i] for i in train_idx]

    train_descriptors, train_gradients, train_grad_index = builder.build_descriptors(train_dataset)
    val_descriptors, val_gradients, val_grad_index = builder.build_descriptors(val_dataset)
    test_descriptors, test_gradients, test_grad_index = builder.build_descriptors(test_dataset)

    # PCA compression (after descriptor computation, before scaling)
    if cfg.compress_pca and cfg.compress_P is not None:
        from descriptor_pca import DescriptorPCA
        pca = DescriptorPCA(n_components=cfg.compress_P)
        pca.fit(train_descriptors)
        train_descriptors, train_gradients = pca.transform(train_descriptors, train_gradients)
        val_descriptors, val_gradients = pca.transform(val_descriptors, val_gradients)
        test_descriptors, test_gradients = pca.transform(test_descriptors, test_gradients)
        cfg._descriptor_pca = pca
        print(f"PCA compression: {train_descriptors[0].shape[-1]} components "
              f"({pca.explained_variance_ratio_.sum():.4f} variance explained)")

    if cfg.scale_descriptors:
        all_desc = np.concatenate(train_descriptors, axis=0)  # [total_train_atoms, dim_q]
        dim_q = int(all_desc.shape[-1])

        if cfg.descriptor_scale_mode == "range":
            # GPUMD-style: divide by per-component range (max - min)
            desc_max = np.max(all_desc, axis=0)   # [dim_q]
            desc_min = np.min(all_desc, axis=0)   # [dim_q]
            desc_range = desc_max - desc_min
            # Components with zero range are constant — leave unscaled
            cfg.descriptor_mean = np.where(desc_range > 1e-30, desc_range, 1.0)
            print(f"Descriptor scaling (range): component range = "
                  f"[{desc_range.min():.6f}, {desc_range.max():.6f}], "
                  f"active components = {np.sum(desc_range > 1e-30)}/{dim_q}")

        elif cfg.descriptor_scale_mode == "mean":
            # Mean-based: divide by mean(|x|) * sqrt(dim_q)
            raw_mean = np.mean(np.abs(all_desc), axis=0)  # [dim_q]
            if cfg.descriptor_scale_floor is not None:
                floor = np.max(raw_mean) * cfg.descriptor_scale_floor
                safe_mean = np.maximum(raw_mean, floor)
            else:
                safe_mean = np.maximum(raw_mean, 1e-30)
            cfg.descriptor_mean = safe_mean * np.sqrt(dim_q)
            floor_str = f"{floor:.6f}" if cfg.descriptor_scale_floor is not None else "None"
            print(f"Descriptor scaling (mean): raw |mean| range = "
                  f"[{raw_mean.min():.6f}, {raw_mean.max():.6f}], "
                  f"floor = {floor_str}, effective scale norm = "
                  f"{np.linalg.norm(cfg.descriptor_mean):.6f} (dim_q={dim_q})")

        else:
            raise ValueError(f"Unknown descriptor_scale_mode: {cfg.descriptor_scale_mode!r} "
                             f"(expected 'range' or 'mean')")

        print("  Descriptor scaling applied.")

    train_data = assemble_data_dict(train_dataset, train_types_int, train_descriptors, train_gradients, train_grad_index, cfg)
    test_data = assemble_data_dict(test_dataset, test_types_int, test_descriptors, test_gradients, test_grad_index, cfg)
    val_data = assemble_data_dict(val_dataset, val_types_int, val_descriptors, val_gradients, val_grad_index, cfg)
    n_train = len(train_data["positions"])
    n_test_actual = len(test_data["positions"])
    n_val = len(val_data["positions"])
    if cfg.test_data_path is not None:
        print(f"{n_structures} structures split into train ({n_train}) + val ({n_val})")
        print(f"External test set: {n_test_actual} structures from {cfg.test_data_path}")
    else:
        print(f"{n_structures} structures split into train ({n_train}) + test ({n_test_actual}) + val ({n_val})")
    return train_data, test_data, val_data


def collate_flat(data: dict) -> dict[str, torch.Tensor]:
    """Convert variable-length list data to flat concatenated tensors.

    Convert variable-length list data to flat concatenated tensors.
    All per-atom data is concatenated with batch index arrays for scatter ops.

    Args:
        data : dict from split() / assemble_data_dict() with keys:
            descriptors : list of [N_i, dim_q] arrays
            gradients   : list of (list of N_i arrays each [M_ij, 3, dim_q])
            grad_index  : list of (list of N_i lists each [M_ij] ints)
            positions   : list of [N_i, 3] arrays
            Z_int       : list of [N_i] int arrays (type indices)
            targets     : list of scalar/[3]/[6] arrays
            boxes       : list of [3, 3] arrays

    Returns:
        flat : dict with keys:
            descriptors  : [total_atoms, Q]    float32
            Z_int        : [total_atoms]       int64
            positions    : [total_atoms, 3]    float32
            atom_batch   : [total_atoms]       int64
            gradients    : [total_edges, 3, Q] float32
            edge_src     : [total_edges]       int64
            edge_dst     : [total_edges]       int64
            edge_batch   : [total_edges]       int64
            targets      : [S, T]              float32
            boxes        : [S, 3, 3]           float32
            num_atoms    : [S]                 int64
            atom_offsets : [S+1]               int64
            edge_offsets : [S+1]               int64
    """
    S = len(data["descriptors"])
    all_desc, all_Z, all_pos, atom_batch_list = [], [], [], []
    all_grads, edge_src_list, edge_dst_list, edge_batch_list = [], [], [], []
    atom_offset = 0

    for s in range(S):
        desc_s = np.asarray(data["descriptors"][s], dtype=np.float32)
        N_s = desc_s.shape[0]
        all_desc.append(desc_s)
        all_Z.append(np.asarray(data["Z_int"][s], dtype=np.int64))
        all_pos.append(np.asarray(data["positions"][s], dtype=np.float32))
        atom_batch_list.append(np.full(N_s, s, dtype=np.int64))

        for i in range(N_s):
            g_i = np.asarray(data["gradients"][s][i], dtype=np.float32)  # [M_i, 3, Q]
            nbrs_i = data["grad_index"][s][i]                             # [M_i]
            M_i = g_i.shape[0]
            all_grads.append(g_i)
            edge_src_list.append(np.full(M_i, atom_offset + i, dtype=np.int64))
            edge_dst_list.append(np.array(nbrs_i, dtype=np.int64) + atom_offset)
            edge_batch_list.append(np.full(M_i, s, dtype=np.int64))
        atom_offset += N_s

    # Concatenate all arrays
    desc_cat = np.concatenate(all_desc)
    Z_cat = np.concatenate(all_Z)
    pos_cat = np.concatenate(all_pos)
    ab_cat = np.concatenate(atom_batch_list)
    grads_cat = np.concatenate(all_grads)
    es_cat = np.concatenate(edge_src_list)
    ed_cat = np.concatenate(edge_dst_list)
    eb_cat = np.concatenate(edge_batch_list)

    # Compute offset arrays for O(1) structure chunking
    atom_counts = [np.asarray(data["descriptors"][s]).shape[0] for s in range(S)]
    edge_counts = [sum(np.asarray(data["gradients"][s][i]).shape[0]
                       for i in range(atom_counts[s])) for s in range(S)]
    atom_offsets = np.zeros(S + 1, dtype=np.int64)
    edge_offsets = np.zeros(S + 1, dtype=np.int64)
    np.cumsum(atom_counts, out=atom_offsets[1:])
    np.cumsum(edge_counts, out=edge_offsets[1:])

    # Stack targets — handle scalar (mode 0) vs vector targets
    target_list = [np.asarray(t, dtype=np.float32) for t in data["targets"]]
    if target_list[0].ndim == 0:
        targets_np = np.array(target_list, dtype=np.float32).reshape(-1, 1)
    else:
        targets_np = np.stack(target_list)

    return {
        "descriptors": torch.tensor(desc_cat, dtype=torch.float32),
        "Z_int": torch.tensor(Z_cat, dtype=torch.int64),
        "positions": torch.tensor(pos_cat, dtype=torch.float32),
        "atom_batch": torch.tensor(ab_cat, dtype=torch.int64),
        "gradients": torch.tensor(grads_cat, dtype=torch.float32),
        "edge_src": torch.tensor(es_cat, dtype=torch.int64),
        "edge_dst": torch.tensor(ed_cat, dtype=torch.int64),
        "edge_batch": torch.tensor(eb_cat, dtype=torch.int64),
        "targets": torch.tensor(targets_np, dtype=torch.float32),
        "boxes": torch.tensor(np.stack([np.asarray(b, dtype=np.float32) for b in data["boxes"]]), dtype=torch.float32),
        "num_atoms": torch.tensor(atom_counts, dtype=torch.int64),
        "atom_offsets": torch.tensor(atom_offsets, dtype=torch.int64),
        "edge_offsets": torch.tensor(edge_offsets, dtype=torch.int64),
    }


def select_structure_range(batch: dict[str, torch.Tensor], s_start: int, s_end: int) -> dict[str, torch.Tensor]:
    """O(1) slicing of flat batch using precomputed offsets.

    Args:
        batch   : flat batch dict from collate_flat()
        s_start : first structure index (inclusive)
        s_end   : last structure index (exclusive)

    Returns:
        sub-batch dict with re-indexed atom_batch, edge_src, edge_dst, edge_batch
    """
    a0 = int(batch["atom_offsets"][s_start])
    a1 = int(batch["atom_offsets"][s_end])
    e0 = int(batch["edge_offsets"][s_start])
    e1 = int(batch["edge_offsets"][s_end])
    sub_num_atoms = batch["num_atoms"][s_start:s_end]
    sub_atom_offsets = batch["atom_offsets"][s_start:s_end + 1] - a0
    sub_edge_offsets = batch["edge_offsets"][s_start:s_end + 1] - e0

    return {
        "descriptors": batch["descriptors"][a0:a1],
        "Z_int": batch["Z_int"][a0:a1],
        "positions": batch["positions"][a0:a1],
        "atom_batch": batch["atom_batch"][a0:a1] - s_start,
        "gradients": batch["gradients"][e0:e1],
        "edge_src": batch["edge_src"][e0:e1] - a0,
        "edge_dst": batch["edge_dst"][e0:e1] - a0,
        "edge_batch": batch["edge_batch"][e0:e1] - s_start,
        "targets": batch["targets"][s_start:s_end],
        "boxes": batch["boxes"][s_start:s_end],
        "num_atoms": sub_num_atoms,
        "atom_offsets": sub_atom_offsets,
        "edge_offsets": sub_edge_offsets,
    }


def filter_by_species(dataset: list[Atoms], dataset_types_int: list[np.ndarray], allowed_Z: list[int | str]) -> tuple[list[Atoms], list[np.ndarray]]:
    """Keep only structures whose atoms are all within allowed_Z.

    Args:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray — parallel to dataset
        allowed_Z         : list of int or str — allowed atomic numbers (e.g. [6, 1, 8])
                            or element symbols (e.g. ["C", "H", "O"])

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    from ase.data import atomic_numbers
    allowed = set(atomic_numbers[z] if isinstance(z, str) else z for z in allowed_Z)
    filtered_dataset = []
    filtered_types_int = []
    for struct, types_int in zip(dataset, dataset_types_int):
        if set(struct.numbers).issubset(allowed):
            filtered_dataset.append(struct)
            filtered_types_int.append(types_int)
    return filtered_dataset, filtered_types_int


def print_dipole_statistics(dataset: list[Atoms], target_key: str = "dipole") -> None:
    """Print min/max/mean/std of dipole targets across the dataset.

    Args:
        dataset    : list of ase.Atoms with info[target_key] = [3] array
        target_key : str key in Atoms.info holding the dipole vector
    """
    dipoles = np.array([_extract_target(s, target_key) for s in dataset])
    norms = np.linalg.norm(dipoles, axis=1)
    print("=== Dipole Target Statistics ===")
    print(f"  N structures: {len(dipoles)}")
    print(f"  Component ranges: x=[{dipoles[:,0].min():.4f}, {dipoles[:,0].max():.4f}]  "
          f"y=[{dipoles[:,1].min():.4f}, {dipoles[:,1].max():.4f}]  "
          f"z=[{dipoles[:,2].min():.4f}, {dipoles[:,2].max():.4f}]")
    print(f"  Component means:  x={dipoles[:,0].mean():.4f}  y={dipoles[:,1].mean():.4f}  z={dipoles[:,2].mean():.4f}")
    print(f"  Component stds:   x={dipoles[:,0].std():.4f}  y={dipoles[:,1].std():.4f}  z={dipoles[:,2].std():.4f}")
    print(f"  |μ| range: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"  |μ| mean:  {norms.mean():.4f}  std: {norms.std():.4f}")


def print_polarizability_statistics(dataset: list[Atoms], target_key: str = "pol") -> None:
    """Print min/max/mean/std of polarizability targets across the dataset.

    Args:
        dataset    : list of ase.Atoms with info[target_key] = [6] or [9] array
        target_key : str key in Atoms.info holding the polarizability tensor
    """
    pols = np.array([_extract_target(s, target_key) for s in dataset])
    labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
    print("=== Polarizability Target Statistics ===")
    print(f"  N structures: {len(pols)}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{pols[:,i].min():.4f}, {pols[:,i].max():.4f}]  "
              f"mean={pols[:,i].mean():.4f}  std={pols[:,i].std():.4f}")
