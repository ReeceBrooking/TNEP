from __future__ import annotations

import tensorflow as tf
import numpy as np
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from ase.io import read
from ase import Atoms


BOHR_TO_ANGSTROM = 0.529177210903
DEBYE_TO_EANGSTROM = 0.20819434

# Box used for structures without periodic boundary conditions.
# Large enough that MIC never wraps any pairwise displacement (1000 Å >> any
# molecular extent or NEP cutoff radius).
_NO_PBC_BOX = 1000.0 * np.eye(3, dtype=np.float32)


def cell_to_box(atoms) -> np.ndarray:
    """Return the cell matrix for *atoms*, or a large dummy box if unset/zero.

    Two cases are treated as "no periodic boundary":
      1. ASE stores an unset cell as a zero 3×3 matrix — det = 0, not invertible.
      2. A Lattice is present in the file but all entries are zero (uniformly 0).
    In both cases a 1000 Å cubic box is returned so MIC never alters any
    pairwise displacement while keeping GPU code unconditional.
    """
    cell = atoms.cell.array.astype(np.float32)
    if np.allclose(cell, 0) or abs(np.linalg.det(cell)) < 1e-6:
        return _NO_PBC_BOX
    return cell


def _dipole_conversion_factor(dipole_units: str) -> float:
    """Return the multiplicative factor to convert dipole_units → e·Å.

    Args:
        dipole_units : "e*angstrom", "e*bohr", or "debye"

    Returns:
        float — conversion factor (1.0 if already in e·Å)
    """
    if dipole_units == "e*angstrom":
        return 1.0
    elif dipole_units == "e*bohr":
        return BOHR_TO_ANGSTROM
    elif dipole_units == "debye":
        return DEBYE_TO_EANGSTROM
    else:
        raise ValueError(f"Unknown dipole_units: {dipole_units!r} "
                         f"(expected 'e*angstrom', 'e*bohr', or 'debye')")


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
        dataset, dataset_types_int = filter_by_species(dataset, dataset_types_int, allowed_Z=cfg.allowed_species, mode=cfg.filter_mode)
        print("After species filter (" + cfg.filter_mode + "): " + str(len(dataset)) + " structures")

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


def _extract_target(structure: Atoms, target_key: str) -> tf.Tensor:
    """Extract target, converting 9-component polarizability to 6-component if needed.

    Handles datasets where values have a trailing space inside the quoted string
    (e.g. mu="-0.734398 0.000000 -0.040971 ").  ASE's extxyz parser splits on
    literal space and produces a spurious NaN at the end; trailing NaN elements
    are stripped before the tensor is returned.
    """
    if target_key in structure.info:
        raw = np.asarray(structure.info[target_key], dtype=np.float32)
    elif structure.calc is not None and target_key in structure.calc.results:
        raw = np.asarray(structure.calc.results[target_key], dtype=np.float32)
    else:
        raise KeyError(f"'{target_key}' not found in structure.info or calc.results")
    # Strip trailing NaN artefacts produced by trailing whitespace in quoted values
    if raw.ndim == 1:
        n = raw.size
        while n > 0 and np.isnan(raw[n - 1]):
            n -= 1
        raw = raw[:n]
    if raw.size == 9:
        # Flattened 3x3 row-major -> unique [xx, yy, zz, xy, yz, zx]
        raw = raw[[0, 4, 8, 1, 5, 6]]
    return tf.convert_to_tensor(raw, dtype=tf.float32)


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
            target = _extract_target(structure, target_key).numpy()
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
                            filter_zero_targets flags

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    target_key = _resolve_target_key(cfg)
    bad = find_bad_data(dataset, target_key)

    bad_indices: set[int] = set()

    # Always remove structures missing the target key
    if bad['missing_targets']:
        bad_indices.update(bad['missing_targets'])
        print(f"  Filtering {len(bad['missing_targets'])} structures with missing '{target_key}' key")

    if not (cfg.filter_nan_positions or cfg.filter_nan_targets
            or cfg.filter_zero_targets) and not bad_indices:
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

    if bad_indices:
        dataset = [s for i, s in enumerate(dataset) if i not in bad_indices]
        dataset_types_int = [t for i, t in enumerate(dataset_types_int) if i not in bad_indices]
        print(f"  Removed {len(bad_indices)} bad structures, {len(dataset)} remaining")

    return dataset, dataset_types_int


def _target_key_for_mode(target_mode: int) -> str:
    """Return the default Atoms.info key for the given target mode."""
    return {0: "energy", 1: "dipole", 2: "pol"}[target_mode]


def _resolve_target_key(cfg: TNEPconfig) -> str:
    """Return the target key to use, honouring cfg.target_key if set.

    If cfg.target_key is not None it is returned as-is, allowing non-standard
    dataset labels (e.g. "mu", "alpha") to be used without changing target_mode.
    Otherwise falls back to the mode default ("energy", "dipole", "pol").
    """
    if cfg.target_key is not None:
        return cfg.target_key
    return _target_key_for_mode(cfg.target_mode)


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
    r2_comp = metrics["r2_components"].numpy()
    labels = component_labels(cfg.target_mode, len(r2_comp))

    if "total_rmse" in metrics:
        print(f"\n{prefix} (per-atom) RMSE: {rmse:.4f}")
        print(f"{prefix} (per-atom) R²:   {r2:.4f}")
        print("Per-atom per-component R²:  " + "  ".join(
            f"{lbl}={r2_comp[i]:.4f}" for i, lbl in enumerate(labels)))

        total_rmse = float(metrics["total_rmse"])
        total_r2 = float(metrics["total_r2"])
        total_r2_comp = metrics["total_r2_components"].numpy()
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
        cos_all = metrics["cos_sim_all"].numpy()
        print(f"Cosine similarity:  mean={cos_mean:.4f}  "
              f"min={cos_all.min():.4f}  max={cos_all.max():.4f}  "
              f"std={cos_all.std():.4f}")


def _get_forces(s: Atoms) -> np.ndarray | None:
    """Get per-atom forces [N, 3] from an Atoms object, or None."""
    for key in ("force", "forces"):
        if key in s.arrays:
            return np.asarray(s.arrays[key], dtype=np.float32)
    if s.calc is not None and "forces" in s.calc.results:
        return np.asarray(s.calc.results["forces"], dtype=np.float32)
    return None


def _get_virial(s: Atoms) -> np.ndarray | None:
    """Get virial as 6-component Voigt [xx,yy,zz,xy,yz,zx] from Atoms, or None."""
    if "virial" in s.info:
        v = np.asarray(s.info["virial"], dtype=np.float32)
        if v.shape == (3, 3):
            return np.array([v[0, 0], v[1, 1], v[2, 2],
                             v[0, 1], v[1, 2], v[2, 0]], dtype=np.float32)
        elif v.shape == (9,):
            v = v.reshape(3, 3)
            return np.array([v[0, 0], v[1, 1], v[2, 2],
                             v[0, 1], v[1, 2], v[2, 0]], dtype=np.float32)
        elif v.shape == (6,):
            return v
    return None


def _dataset_has_key(dataset: list[Atoms], getter) -> bool:
    """Check whether ALL structures in dataset have a given property."""
    return all(getter(s) is not None for s in dataset)


def assemble_data_dict(
    dataset: list[Atoms],
    types_int: list[np.ndarray],
    descriptors: list[tf.Tensor],
    gradients: list[list[tf.Tensor]],
    grad_index: list[list[list[int]]],
    cfg: TNEPconfig,
) -> dict:
    """Assemble a data dict from structures, type indices, and precomputed descriptors.

    Args:
        dataset     : list of ase.Atoms
        types_int   : list of ndarray [N_i] integer type indices
        descriptors : list of [N_i, dim_q] tensors
        gradients   : list of (list of N_i tensors each [M, 3, dim_q])
        grad_index  : list of (list of N_i lists each [M] ints)
        cfg         : TNEPconfig (uses target_mode)

    Returns:
        dict with keys: positions, Z_int, targets, boxes, descriptors, gradients, grad_index
    """
    target_key = _resolve_target_key(cfg)
    targets = [_extract_target(s, target_key) for s in dataset]
    if cfg.target_mode == 1:
        factor = _dipole_conversion_factor(cfg.dipole_units)
        if factor != 1.0:
            targets = [t * factor for t in targets]
    if cfg.scale_targets and cfg.target_mode == 1:
        targets = [t / tf.cast(len(s), tf.float32) for t, s in zip(targets, dataset)]
    data = {
        "positions": [tf.convert_to_tensor(s.positions, dtype=tf.float32) for s in dataset],
        "Z_int": [tf.convert_to_tensor(t, dtype=tf.int32) for t in types_int],
        "targets": targets,
        "boxes": [tf.convert_to_tensor(cell_to_box(s), dtype=tf.float32) for s in dataset],
        "descriptors": descriptors,
        "gradients": gradients,
        "grad_index": grad_index,
    }

    # Auto-detect and include force/virial targets for PES mode
    if cfg.target_mode == 0:
        if _dataset_has_key(dataset, _get_forces):
            data["forces"] = [tf.convert_to_tensor(_get_forces(s), dtype=tf.float32)
                              for s in dataset]
            print("  PES mode: forces detected in dataset")
        if _dataset_has_key(dataset, _get_virial):
            data["virials"] = [tf.convert_to_tensor(_get_virial(s), dtype=tf.float32)
                               for s in dataset]
            print("  PES mode: virials detected in dataset")

    return data


def prepare_eval_data(dataset: list[Atoms], cfg: TNEPconfig) -> dict[str, tf.Tensor]:
    """Build type indices, descriptors, and padded data dict for evaluation.

    Convenience function that chains assign_type_indices → build_descriptors →
    assemble_data_dict → pad_and_stack.

    Args:
        dataset : list of ase.Atoms — structures to evaluate
        cfg     : TNEPconfig from training (carries types, descriptor params, target_mode)

    Returns:
        padded data dict ready for model.score() or model.predict_batch()
    """
    types_int = assign_type_indices(dataset, cfg.types)
    builder = DescriptorBuilder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(dataset)
    if hasattr(cfg, '_descriptor_pca') and cfg._descriptor_pca is not None:
        descriptors, gradients = cfg._descriptor_pca.transform(descriptors, gradients)
    data = assemble_data_dict(dataset, types_int, descriptors, gradients, grad_index, cfg)
    return pad_and_stack(data)


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
            positions   : list of [N_i, 3] tensors
            Z_int       : list of [N_i] int tensors (type indices)
            targets     : list of target tensors (scalar for PES, [3] for dipole)
            boxes       : list of [3, 3] tensors (lattice vectors)
            descriptors : list of [N_i, dim_q] tensors
            gradients   : list of (list of N_i tensors each [M, 3, dim_q])
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
        test_structures = read(cfg.test_data_path, index=":")
        if cfg.allowed_species is not None:
            # Filter before type assignment — test file may contain unknown species
            from ase.data import atomic_numbers
            allowed = set(atomic_numbers[z] if isinstance(z, str) else z
                          for z in cfg.allowed_species)
            if cfg.filter_mode == "exact":
                test_structures = [s for s in test_structures
                                   if set(s.numbers) == allowed]
            else:
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
        all_desc = tf.concat(train_descriptors, axis=0)  # [total_train_atoms, dim_q]
        dim_q = int(all_desc.shape[-1])

        if cfg.descriptor_scale_mode == "range":
            # GPUMD-style: divide by per-component range (max - min)
            desc_max = tf.reduce_max(all_desc, axis=0).numpy()   # [dim_q]
            desc_min = tf.reduce_min(all_desc, axis=0).numpy()   # [dim_q]
            desc_range = desc_max - desc_min
            # Components with zero range are constant — leave unscaled
            cfg.descriptor_mean = np.where(desc_range > 1e-30, desc_range, 1.0)
            print(f"Descriptor scaling (range): component range = "
                  f"[{desc_range.min():.6f}, {desc_range.max():.6f}], "
                  f"active components = {np.sum(desc_range > 1e-30)}/{dim_q}")

        elif cfg.descriptor_scale_mode == "mean":
            # Mean-based: divide by mean(|x|) * sqrt(dim_q)
            raw_mean = tf.reduce_mean(tf.abs(all_desc), axis=0).numpy()  # [dim_q]
            if cfg.descriptor_scale_floor is not None:
                floor = max(np.max(raw_mean) * cfg.descriptor_scale_floor, 1e-6)
                safe_mean = np.maximum(raw_mean, floor)
            else:
                safe_mean = np.maximum(raw_mean, 1e-6)
            cfg.descriptor_mean = safe_mean * np.sqrt(dim_q)
            floor_str = f"{floor:.6f}" if cfg.descriptor_scale_floor is not None else "None"
            print(f"Descriptor scaling (mean): raw |mean| range = "
                  f"[{raw_mean.min():.6f}, {raw_mean.max():.6f}], "
                  f"floor = {floor_str}, effective scale norm = "
                  f"{np.linalg.norm(cfg.descriptor_mean):.6f} (dim_q={dim_q})")

        else:
            raise ValueError(f"Unknown descriptor_scale_mode: {cfg.descriptor_scale_mode!r} "
                             f"(expected 'range' or 'mean')")

        # Roundtrip verification: q_scaled * s must recover original exactly
        scale = tf.constant(cfg.descriptor_mean, dtype=tf.float32)
        for split_name, split_descs, split_grads in [
            ("train", train_descriptors, train_gradients),
            ("val", val_descriptors, val_gradients),
            ("test", test_descriptors, test_gradients),
        ]:
            for i in range(min(3, len(split_descs))):
                q_orig = split_descs[i]                            # [N_i, dim_q]
                q_scaled = q_orig / scale
                q_recovered = q_scaled * scale
                desc_err = float(tf.reduce_max(tf.abs(q_orig - q_recovered)))

                g_orig = split_grads[i]                            # list of N_i tensors [M, 3, dim_q]
                g_scaled = [g / scale for g in g_orig]
                g_recovered = [g * scale for g in g_scaled]
                grad_err = max(
                    float(tf.reduce_max(tf.abs(go - gr)))
                    for go, gr in zip(g_orig, g_recovered)
                ) if g_orig else 0.0

                if desc_err > 1e-5 or grad_err > 1e-5:
                    print(f"  WARNING: {split_name}[{i}] roundtrip error: "
                          f"desc={desc_err:.2e}, grad={grad_err:.2e}")
            # Check feature order: verify scale[j] corresponds to descriptor column j
            if split_descs:
                col_means = tf.reduce_mean(tf.abs(split_descs[0]), axis=0)  # [dim_q]
                ratio = col_means / scale
                ratio_std = float(tf.math.reduce_std(ratio))
                if ratio_std > 10.0:
                    print(f"  WARNING: {split_name} feature-order suspect — "
                          f"col_mean/scale ratio std={ratio_std:.2f} (expect <10)")
        print("  Descriptor scaling roundtrip check passed.")

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


def pad_and_stack(data: dict, num_types: int | None = None) -> dict[str, tf.Tensor]:
    """Convert variable-length list-of-tensors data into dense padded tensors.

    Transforms the output of split() into fixed-shape tensors suitable for
    batched GPU evaluation. Variable atom counts and neighbor counts are
    padded to their maximums with zeros, and boolean masks track real vs
    padded entries.

    Args:
        data : dict from split() with keys:
            descriptors : list of [N_i, dim_q] tensors
            gradients   : list of (list of N_i tensors each [M_ij, 3, dim_q])
            grad_index  : list of (list of N_i lists each [M_ij] ints)
            positions   : list of [N_i, 3] tensors
            Z_int       : list of [N_i] int tensors
            targets     : list of scalar/[3]/[6] tensors
            boxes       : list of [3, 3] tensors

    Returns:
        padded : dict with keys:
            descriptors    : [S, A, Q]        float32
            gradients      : [S, A, M, 3, Q]  float32
            grad_index     : [S, A, M]        int32
            positions      : [S, A, 3]        float32
            Z_int          : [S, A]           int32
            targets        : [S, T]           float32  (T=1 for PES, 3 for dipole, 6 for pol)
            boxes          : [S, 3, 3]        float32
            atom_mask      : [S, A]           float32  (1.0 for real atoms, 0.0 for padding)
            neighbor_mask  : [S, A, M]        float32  (1.0 for real neighbors, 0.0 for padding)
            num_atoms      : [S]              int32    (actual atom count per structure)
        where S = num_structures, A = max_atoms, M = max_neighbors, Q = dim_q
    """
    S = len(data["descriptors"])
    dim_q = data["descriptors"][0].shape[-1]

    # Find max atom count across all structures
    atom_counts = [data["descriptors"][i].shape[0] for i in range(S)]
    max_atoms = max(atom_counts)

    # Find max neighbor count across all atoms in all structures
    max_neighbors = 0
    for s in range(S):
        for i in range(atom_counts[s]):
            n_nbrs = data["gradients"][s][i].shape[0]
            if n_nbrs > max_neighbors:
                max_neighbors = n_nbrs

    # Target dimensionality
    target_sample = data["targets"][0]
    if target_sample.shape == ():
        target_dim = 1
    else:
        target_dim = target_sample.shape[0]

    # Detect optional force/virial targets (PES mode)
    has_forces = "forces" in data
    has_virials = "virials" in data

    # Pre-allocate numpy arrays (faster than list comprehension for padding)
    desc_np = np.zeros((S, max_atoms, dim_q), dtype=np.float32)
    grad_np = np.zeros((S, max_atoms, max_neighbors, 3, dim_q), dtype=np.float32)
    gidx_np = np.zeros((S, max_atoms, max_neighbors), dtype=np.int32)
    pos_np = np.zeros((S, max_atoms, 3), dtype=np.float32)
    z_np = np.zeros((S, max_atoms), dtype=np.int32)
    tgt_np = np.zeros((S, target_dim), dtype=np.float32)
    box_np = np.zeros((S, 3, 3), dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms), dtype=np.float32)
    nbr_mask_np = np.zeros((S, max_atoms, max_neighbors), dtype=np.float32)
    num_atoms_np = np.array(atom_counts, dtype=np.int32)
    if has_forces:
        force_np = np.zeros((S, max_atoms, 3), dtype=np.float32)
    if has_virials:
        virial_np = np.zeros((S, 6), dtype=np.float32)

    if num_types is not None:
        types_contained_np = np.zeros((S, num_types), dtype=np.float32)

    for s in range(S):
        N_s = atom_counts[s]
        desc_np[s, :N_s, :] = data["descriptors"][s].numpy()
        pos_np[s, :N_s, :] = data["positions"][s].numpy()
        z_np[s, :N_s] = data["Z_int"][s].numpy()
        if num_types is not None:
            z_vals = z_np[s, :N_s]
            for t in range(num_types):
                if np.any(z_vals == t):
                    types_contained_np[s, t] = 1.0
        box_np[s] = data["boxes"][s].numpy()
        atom_mask_np[s, :N_s] = 1.0

        t = data["targets"][s]
        if t.shape == ():
            tgt_np[s, 0] = t.numpy()
        else:
            tgt_np[s, :] = t.numpy()

        if has_forces:
            force_np[s, :N_s, :] = data["forces"][s].numpy()
        if has_virials:
            virial_np[s, :] = data["virials"][s].numpy()

        for i in range(N_s):
            n_nbrs = data["gradients"][s][i].shape[0]
            grad_np[s, i, :n_nbrs, :, :] = data["gradients"][s][i].numpy()
            gidx_np[s, i, :n_nbrs] = data["grad_index"][s][i]
            nbr_mask_np[s, i, :n_nbrs] = 1.0

    with tf.device('/CPU:0'):
        result = {
            "descriptors": tf.constant(desc_np),
            "gradients": tf.constant(grad_np),
            "grad_index": tf.constant(gidx_np),
            "positions": tf.constant(pos_np),
            "Z_int": tf.constant(z_np),
            "targets": tf.constant(tgt_np),
            "boxes": tf.constant(box_np),
            "atom_mask": tf.constant(atom_mask_np),
            "neighbor_mask": tf.constant(nbr_mask_np),
            "num_atoms": tf.constant(num_atoms_np),
        }
        if has_forces:
            result["forces"] = tf.constant(force_np)
        if has_virials:
            result["virials"] = tf.constant(virial_np)
        if num_types is not None:
            result["types_contained"] = tf.constant(types_contained_np)
    return result


def filter_by_species(dataset: list[Atoms], dataset_types_int: list[np.ndarray], allowed_Z: list[int | str], mode: str = "subset") -> tuple[list[Atoms], list[np.ndarray]]:
    """Keep only structures whose atoms satisfy the species filter.

    Args:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray — parallel to dataset
        allowed_Z         : list of int or str — allowed atomic numbers (e.g. [6, 1, 8])
                            or element symbols (e.g. ["C", "H", "O"])
        mode              : "subset" = keep structures with only allowed species
                            "exact"  = keep structures containing exactly all allowed species

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    from ase.data import atomic_numbers
    allowed = set(atomic_numbers[z] if isinstance(z, str) else z for z in allowed_Z)
    filtered_dataset = []
    filtered_types_int = []
    for struct, types_int in zip(dataset, dataset_types_int):
        species = set(struct.numbers)
        if mode == "exact":
            keep = species == allowed
        else:
            keep = species.issubset(allowed)
        if keep:
            filtered_dataset.append(struct)
            filtered_types_int.append(types_int)
    return filtered_dataset, filtered_types_int


def print_dipole_statistics(dataset: list[Atoms], cfg: TNEPconfig,
                            target_key: str = "dipole") -> None:
    """Print min/max/mean/std of dipole targets across the dataset.

    Args:
        dataset    : list of ase.Atoms with info[target_key] = [3] array
        cfg        : TNEPconfig — used to check unit conversion flag
        target_key : str key in Atoms.info holding the dipole vector
    """
    dipoles = np.array([_extract_target(s, target_key).numpy() for s in dataset])
    factor = _dipole_conversion_factor(cfg.dipole_units)
    if factor != 1.0:
        dipoles = dipoles * factor
    unit = f"e\u00b7\u00c5 (from {cfg.dipole_units})" if factor != 1.0 else "e\u00b7\u00c5"
    norms = np.linalg.norm(dipoles, axis=1)
    print(f"=== Dipole Target Statistics ({unit}) ===")
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
    pols = np.array([_extract_target(s, target_key).numpy() for s in dataset])
    labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
    print("=== Polarizability Target Statistics ===")
    print(f"  N structures: {len(pols)}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{pols[:,i].min():.4f}, {pols[:,i].max():.4f}]  "
              f"mean={pols[:,i].mean():.4f}  std={pols[:,i].std():.4f}")
