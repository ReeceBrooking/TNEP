from __future__ import annotations

import tensorflow as tf
import numpy as np
from TNEPconfig import TNEPconfig
from DescriptorBuilder import make_descriptor_builder
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
            z = int(structure.numbers[i])  # cast np.int64 → Python int
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

    # Recompute type list and indices after species filtering. Coerce
    # to Python int — `struct.numbers` is a numpy array, so the raw
    # entries are np.int64 and would later fail json.dumps in model_io.
    cfg.types = []
    for struct in dataset:
        for z in struct.numbers:
            zi = int(z)
            if zi not in cfg.types:
                cfg.types.append(zi)
    cfg.num_types = len(cfg.types)
    dataset_types_int = assign_type_indices(dataset, cfg.types)
    print("Species: " + str(cfg.types) + " (" + str(cfg.num_types) + " types)")

    # Filter bad data based on config flags
    dataset, dataset_types_int = filter_bad_data(dataset, dataset_types_int, cfg)

    # Per-type structure coverage: fraction of structures that contain
    # at least one atom of each type. Useful for spotting under-represented
    # species early — under-represented types regularise unstably and
    # are typically the first thing to investigate when train RMSE
    # plateaus per-type.
    if dataset_types_int:
        from ase.data import chemical_symbols
        S = len(dataset_types_int)
        for t, z in enumerate(cfg.types):
            n_with_t = sum(1 for ts in dataset_types_int if (ts == t).any())
            sym = chemical_symbols[int(z)] if 0 <= int(z) < len(chemical_symbols) else "?"
            print(f"  Type {t} ({sym}, Z={int(z)}): present in {n_with_t/S:.1%} of structures")

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
    """Remove structures with missing or unusable targets.

    Structures with missing target keys are always removed (they can't
    be trained against). When `cfg.filter_bad_data` is True, also
    remove NaN positions, NaN targets, and zero-vector targets.

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    target_key = _resolve_target_key(cfg)
    bad = find_bad_data(dataset, target_key)

    bad_indices: set[int] = set()
    if bad['missing_targets']:
        bad_indices.update(bad['missing_targets'])
        print(f"  Filtering {len(bad['missing_targets'])} structures with missing '{target_key}' key")

    if cfg.filter_bad_data:
        for kind in ('nan_positions', 'nan_targets', 'zero_targets'):
            if bad[kind]:
                bad_indices.update(bad[kind])
                print(f"  Filtering {len(bad[kind])} structures with {kind.replace('_', ' ')}")

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
    if cfg.target_mode == 1 and getattr(cfg, "convert_dipole_to_eangstrom", True):
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
    builder = make_descriptor_builder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(dataset)
    data = assemble_data_dict(dataset, types_int, descriptors, gradients, grad_index, cfg)
    # Thread q_scaler and target_mean from cfg so the eval dict is in
    # the same scaled+centered space the model was trained on. Without
    # this, downstream metrics that aren't shift-invariant (cos_sim,
    # total_rmse) silently report wrong values when centering is on.
    return pad_and_stack(
        data,
        num_types=cfg.num_types,
        q_scaler=getattr(cfg, "_q_scaler", None),
        target_mean=getattr(cfg, "_target_mean", None))


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

    builder = make_descriptor_builder(cfg)

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

    # Build train + val descriptors. Test descriptors are NOT built here —
    # they're constructed lazily at scoring time by `materialize_test_data`
    # (avoids paying that cost before training, in case the user aborts).
    # When cache_gradients_to_disk is on the GPU TF backend, gradients go
    # straight from per-chunk SOAP output into the NVMe scratch file; the
    # dataset-wide ~10 GB COO is never held in RAM at any point.
    train_streamed = val_streamed = None
    gv_dir = getattr(cfg, "_gradient_cache_path", None)
    if (getattr(cfg, "cache_gradients_to_disk", False)
            and gv_dir is not None and cfg.descriptor_mode == 1):
        os_module = __import__("os")
        train_streamed = builder.build_descriptors_streaming_to_disk(
            train_dataset,
            gv_path=os_module.path.join(gv_dir, "grad_values_train.bin"),
            chunk_size=cfg.descriptor_batch_frames,
            progress_desc="Building train descriptors (streaming)")
        val_streamed = builder.build_descriptors_streaming_to_disk(
            val_dataset,
            gv_path=os_module.path.join(gv_dir, "grad_values_val.bin"),
            chunk_size=cfg.descriptor_batch_frames,
            progress_desc="Building val descriptors (streaming)")
        train_descriptors = train_streamed["descriptors"]
        val_descriptors   = val_streamed["descriptors"]
        # Empty bucket-format placeholders so assemble_data_dict /
        # pad_and_stack's existing field plumbing keeps working.
        # pad_and_stack picks up prebuilt_gv from data['_prebuilt_gv'] and
        # skips its own gradient flatten step.
        train_gradients  = [[] for _ in range(len(train_dataset))]
        val_gradients    = [[] for _ in range(len(val_dataset))]
        train_grad_index = [[] for _ in range(len(train_dataset))]
        val_grad_index   = [[] for _ in range(len(val_dataset))]
    else:
        _kw = {"progress_desc": "Building train descriptors"} if cfg.descriptor_mode == 1 else {}
        train_descriptors, train_gradients, train_grad_index = builder.build_descriptors(train_dataset, **_kw)
        _kw = {"progress_desc": "Building val descriptors"} if cfg.descriptor_mode == 1 else {}
        val_descriptors,   val_gradients,   val_grad_index   = builder.build_descriptors(val_dataset, **_kw)

    train_data = assemble_data_dict(train_dataset, train_types_int, train_descriptors, train_gradients, train_grad_index, cfg)
    val_data   = assemble_data_dict(val_dataset,   val_types_int,   val_descriptors,   val_gradients,   val_grad_index,   cfg)
    if train_streamed is not None:
        # Streaming-to-disk: hand the streamed descriptor / gradient
        # metadata to pad_and_stack via a private dict key so the standard
        # MasterTNEP call path stays unchanged.
        train_data["_prebuilt_gv"] = train_streamed
        val_data["_prebuilt_gv"] = val_streamed
    # Test set: deferred. Stash the raw atoms + per-atom type indices so
    # `materialize_test_data` can build descriptors at scoring time. The
    # rest of train_model treats this dict as opaque until then.
    test_pending = {
        "_pending_test": True,
        "dataset": test_dataset,
        "types_int": test_types_int,
    }
    n_train = len(train_data["positions"])
    n_test  = len(test_dataset)
    n_val   = len(val_data["positions"])
    if cfg.test_data_path is not None:
        print(f"{n_structures} structures split into train ({n_train}) + val ({n_val})")
        print(f"External test set: {n_test} structures from {cfg.test_data_path} (descriptors deferred to scoring)")
    else:
        print(f"{n_structures} structures split into train ({n_train}) + test ({n_test}) + val ({n_val}) "
              f"(test descriptors deferred to scoring)")
    return train_data, test_pending, val_data


def materialize_test_data(test_pending: dict, cfg: 'TNEPconfig',
                          num_types: int | None = None,
                          pin_to_cpu: bool | None = None) -> dict:
    """Build test descriptors on demand and return a ready-to-score data dict.

    Idempotent: if `test_pending` has already been materialised, the cached
    dict is returned unchanged. The cached dict is stashed in
    `test_pending["_built"]` so callers can hold onto the same `test_pending`
    handle across the training loop and scoring without rebuilding.

    Args:
        test_pending : dict from `split()`'s third return value with
                       `_pending_test=True` plus raw `dataset` and
                       `types_int` keys.
        cfg          : TNEPconfig (descriptor backend, target_mode, ...)
        num_types    : passed to `pad_and_stack`. Defaults to cfg.num_types.
        pin_to_cpu   : passed to `pad_and_stack`. Defaults to
                       cfg.pin_data_to_cpu.

    Returns:
        Padded, stacked test_data dict (same shape as train_data / val_data).
    """
    if not test_pending.get("_pending_test", False):
        return test_pending  # already materialised or never deferred
    cached = test_pending.get("_built")
    if cached is not None:
        return cached

    if num_types is None:
        num_types = cfg.num_types
    if pin_to_cpu is None:
        pin_to_cpu = cfg.pin_data_to_cpu

    test_dataset = test_pending["dataset"]
    test_types_int = test_pending["types_int"]

    builder = make_descriptor_builder(cfg)
    gv_dir = getattr(cfg, "_gradient_cache_path", None)
    test_streamed = None
    if (getattr(cfg, "cache_gradients_to_disk", False)
            and gv_dir is not None and cfg.descriptor_mode == 1):
        import os as _os
        test_streamed = builder.build_descriptors_streaming_to_disk(
            test_dataset,
            gv_path=_os.path.join(gv_dir, "grad_values_test.bin"),
            chunk_size=cfg.descriptor_batch_frames,
            progress_desc="Building test descriptors (streaming)")
        test_descriptors = test_streamed["descriptors"]
        test_gradients   = [[] for _ in range(len(test_dataset))]
        test_grad_index  = [[] for _ in range(len(test_dataset))]
    else:
        _kw = {"progress_desc": "Building test descriptors"} if cfg.descriptor_mode == 1 else {}
        test_descriptors, test_gradients, test_grad_index = builder.build_descriptors(
            test_dataset, **_kw)

    test_data = assemble_data_dict(
        test_dataset, test_types_int,
        test_descriptors, test_gradients, test_grad_index, cfg)
    if test_streamed is not None:
        test_data["_prebuilt_gv"] = test_streamed
    test_data = pad_and_stack(
        test_data, num_types=num_types, pin_to_cpu=pin_to_cpu,
        gradient_cache_path=getattr(cfg, "_gradient_cache_path", None),
        cache_tag="test",
        q_scaler=getattr(cfg, "_q_scaler", None),
        target_mean=getattr(cfg, "_target_mean", None))
    # Pre-stage per-chunk pair indices to GPU. Test eval doesn't go
    # through `_evaluate_chunk` (TNEP.score uses model.predict_batch),
    # so XLA padding isn't needed for test data.
    S_test = int(test_data["num_atoms"].shape[0])
    chunk = cfg.batch_chunk_size if cfg.batch_chunk_size is not None else S_test
    test_ranges = [(s, min(s + chunk, S_test)) for s in range(0, S_test, chunk)]
    prestage_chunk_indices(test_data, test_ranges)
    if (getattr(cfg, "use_pinned_buffers", True)
            and test_data.get("_gv_disk_backed", False)):
        n_buffers = max(int(getattr(cfg, "pinned_pool_size", 2)),
                        int(getattr(cfg, "prefetch_depth", 1)) + 1)
        pool = make_pinned_pool_for(
            test_data, batch_chunk_size=cfg.batch_chunk_size,
            n_buffers=n_buffers)
        if pool is not None:
            test_data["_pinned_pool"] = pool
            print(f"  pinned-buffer pool ({len(pool._all)} × "
                  f"{pool.buffer_nbytes/1e6:.0f} MB) attached to test_data")
    if (getattr(cfg, "use_cufile", True)
            and test_data.get("_gv_disk_backed", False)):
        try:
            from cufile_io import (cuFile_available, CuFileHandle,
                                   make_cufile_pool_for)
        except Exception:
            pass
        else:
            if cuFile_available():
                gv = test_data.get("grad_values")
                if hasattr(gv, "filename"):
                    n_cf = max(int(getattr(cfg, "cufile_pool_size", 2)),
                               int(getattr(cfg, "prefetch_depth", 1)) + 1)
                    pool = make_cufile_pool_for(
                        test_data, batch_chunk_size=cfg.batch_chunk_size,
                        n_buffers=n_cf)
                    if pool is not None:
                        try:
                            handle = CuFileHandle(gv.filename)
                            test_data["_cufile_ctx"] = {"handle": handle, "pool": pool}
                            print(f"  cuFile pool ({len(pool._all)} × "
                                  f"{pool.nbytes/1e6:.0f} MB) attached to test_data")
                        except Exception as e:
                            print(f"  cuFile open failed for test: {e}")
    test_pending["_built"] = test_data
    # Free the raw atom list now that descriptors are baked in.
    test_pending["dataset"] = None
    test_pending["types_int"] = None
    return test_data


def _compute_q_scaler(descriptors_list, dim_q: int,
                      eps: float = 1e-30) -> np.ndarray:
    """Compute per-channel range scaler from training-set descriptors.

    Mirrors GPUMD's `find_max_min` kernel (see GPUMD/src/main_nep/tnep.cu):

        For each channel d:
            range_d   = max over all atoms - min over all atoms
            scaler_d  = 1 / max(range_d, eps)

    The eps floor avoids divide-by-zero for channels that are
    constant across the training set (uncommon but possible if SOAP
    params are pathological; such channels carry no information and
    will be killed by L1/L2 regularisation downstream anyway).

    Computed via streaming min/max to avoid a 2x-memory concat —
    important for large training sets.

    Args:
        descriptors_list : list of [N_i, Q] arrays / tensors (one per
                           structure in the training set, pre-padding).
        dim_q            : Q (number of descriptor channels).
        eps              : floor on the range to avoid 1/0.

    Returns:
        scaler : [Q] float32 array of per-channel multipliers.
    """
    chan_min = None
    chan_max = None
    for d in descriptors_list:
        arr = (d.numpy() if hasattr(d, "numpy") else np.asarray(d))
        arr = arr.reshape(-1, dim_q)
        m_min = arr.min(axis=0)
        m_max = arr.max(axis=0)
        if chan_min is None:
            chan_min, chan_max = m_min, m_max
        else:
            chan_min = np.minimum(chan_min, m_min)
            chan_max = np.maximum(chan_max, m_max)
    chan_range = chan_max - chan_min
    safe_range = np.maximum(chan_range, eps)
    return (1.0 / safe_range).astype(np.float32)


def _compute_q_scaler_l_block(descriptors_list, layout: dict,
                              eps: float = 1e-30) -> np.ndarray:
    """Compute per-(species-pair, l) block-pooled range scaler.

    Same shape as `_compute_q_scaler` ([Q] float32) so the rest of the
    pipeline (storage, application via `_apply_q_scaler_np`, save/load)
    is unchanged. The difference is structural: all q-indices that
    belong to the same (pair, l) angular block share one multiplier.

    Rationale: equalises the dominant magnitude variation in SOAP —
    which lives ACROSS l (l=0 components are O(1), l=l_max ~O(0.01))
    — while preserving the WITHIN-block isotropy that the
    descriptor-mixing rotations (l_aware / cross_pair_l with V_pair
    parameterised by Cayley) assume. Per-α components within a
    (pair, l) block are typically already on comparable scales and
    do not need decorrelating.

    Algorithm:
        For each (pair, l) block B = layout["pair_ln_index"][pair][l]:
            range_B   = max(q[atoms, B]) − min(q[atoms, B])    (scalar)
            for each d in B: scaler[d] = 1 / max(range_B, eps)

    Args:
        descriptors_list : list of [N_i, Q] arrays / tensors (one per
                           training structure, pre-padding).
        layout           : dict from `descriptor_block_layout(cfg)` —
                           supplies "pair_keys", "pair_ln_index", "dim_q".
        eps              : floor on the range to avoid 1/0.

    Returns:
        scaler : [Q] float32 array. Entries within the same (pair, l)
                 block are identical; entries across blocks differ.
    """
    dim_q = int(layout["dim_q"])
    # Streaming per-element min/max, same as the per-component path —
    # we then pool across each block's indices in a second pass. Keeping
    # the streaming pass element-wise avoids a per-block reduce inside
    # the loop over training structures, which would scale O(num_blocks).
    chan_min = None
    chan_max = None
    for d in descriptors_list:
        arr = (d.numpy() if hasattr(d, "numpy") else np.asarray(d))
        arr = arr.reshape(-1, dim_q)
        m_min = arr.min(axis=0)
        m_max = arr.max(axis=0)
        if chan_min is None:
            chan_min, chan_max = m_min, m_max
        else:
            chan_min = np.minimum(chan_min, m_min)
            chan_max = np.maximum(chan_max, m_max)
    scaler = np.empty(dim_q, dtype=np.float32)
    pair_keys = layout["pair_keys"]
    pair_ln_index = layout["pair_ln_index"]
    # Track covered indices to catch layout gaps (would indicate a
    # mismatch between descriptor_block_layout and the actual descriptor
    # build — same invariant the layout code asserts internally).
    covered = np.zeros(dim_q, dtype=bool)
    for pair in pair_keys:
        for l, idx in pair_ln_index[pair].items():
            block_min = float(chan_min[idx].min())
            block_max = float(chan_max[idx].max())
            block_range = max(block_max - block_min, eps)
            scaler[idx] = np.float32(1.0 / block_range)
            covered[idx] = True
    if not covered.all():
        missing = np.where(~covered)[0]
        raise RuntimeError(
            f"l_block q_scaler: {missing.size} q-indices not covered by "
            f"pair_ln_index (first few: {missing[:5].tolist()}). "
            f"descriptor_block_layout / dim_q mismatch.")
    return scaler


def _apply_q_scaler_np(desc_np: np.ndarray,
                       grad_values_np: np.ndarray | None,
                       scaler: np.ndarray) -> None:
    """In-place per-channel scaling of `desc_np` and `grad_values_np`.

    Both tensors are multiplied by the same `[Q]`-shape scaler along
    their last axis. The dipole chain rule (∂U/∂r = ∂U/∂q' · diag(s)
    · ∂q/∂r) is satisfied when BOTH descriptor and grad are scaled at
    the data pipeline — see plan §"Why option (A)".

    Args:
        desc_np        : [S, A, Q] float32 padded descriptors.
        grad_values_np : [P, 3, Q] float32 COO gradient values, or
                         None (e.g. streaming path where grad is
                         already on disk).
        scaler         : [Q] float32 per-channel multiplier.
    """
    s = scaler.astype(np.float32)
    desc_np *= s[None, None, :]
    if grad_values_np is not None:
        grad_values_np *= s[None, None, :]


def _compute_target_mean(targets_list, target_dim: int) -> np.ndarray:
    """Compute per-component mean over the training-set targets.

    The mean is a `[T_dim]` vector — one offset per output channel
    (scalar for energy, [3] for dipole, [6] for polarisability). It's
    subtracted from all targets at the data-pipeline level so the
    network learns in a zero-mean output space (see
    `cfg.target_centering` in TNEPconfig). The same mean is added back
    to predictions at the inference boundary so user-facing values stay
    in the original units.

    Why centering matters: the ANN has one scalar output bias `b1`; it
    cannot place an independent per-component offset on the dipole /
    polarisability output. With a non-zero training target mean (e.g.
    anisotropic dataset orientations), the model must encode the offset
    through the W0/W1/SOAP-gradient pathway, consuming capacity that
    should be going to genuine pattern-fitting. Centering moves the
    offset into a frozen training-time constant.

    Args:
        targets_list : list of scalar / [T_dim] arrays / tensors,
                       one per training structure.
        target_dim   : T_dim (1, 3, or 6 for target_mode 0, 1, 2).

    Returns:
        mean : [T_dim] float32 — per-component mean.
    """
    if not targets_list:
        return np.zeros(target_dim, dtype=np.float32)
    acc = np.zeros(target_dim, dtype=np.float64)
    for i, t in enumerate(targets_list):
        arr = (t.numpy() if hasattr(t, "numpy") else np.asarray(t))
        if arr.shape == ():
            if target_dim != 1:
                raise ValueError(
                    f"_compute_target_mean: target[{i}] is scalar but "
                    f"target_dim={target_dim}; cannot mix scalar and "
                    f"vector targets in one list.")
            acc[0] += float(arr)
        else:
            if arr.shape != (target_dim,):
                raise ValueError(
                    f"_compute_target_mean: target[{i}] has shape "
                    f"{arr.shape} but expected ({target_dim},).")
            acc += arr.astype(np.float64)
    return (acc / len(targets_list)).astype(np.float32)


def pad_and_stack(data: dict, num_types: int | None = None,
                  pin_to_cpu: bool = True,
                  gradient_cache_path: str | None = None,
                  cache_tag: str = "data",
                  prebuilt_gv: dict | None = None,
                  q_scaler: np.ndarray | None = None,
                  target_mean: np.ndarray | None = None) -> dict[str, tf.Tensor]:
    """Convert variable-length list-of-tensors data into COO + padded tensors.

    Gradient data is stored in COO (Coordinate) sparse format to avoid the
    O(S * A_max * M_max * 3 * Q) dense allocation. Only real atom-neighbor
    pairs are stored, giving memory proportional to actual neighbor count
    rather than the padded maximum.

    Descriptors, positions, and other per-atom fields remain structure-padded
    as [S, A_max, ...] since their size is dominated by A_max, not M_max.

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
            descriptors : [S, A, Q]       float32  — padded per-atom descriptors
            grad_values : [P, 3, Q]       float32  — COO gradient blocks (P = total pairs)
            pair_struct : [P]             int32    — structure index for each pair
            pair_atom   : [P]             int32    — center atom index for each pair
            pair_gidx   : [P]            int32    — neighbor atom index for each pair
            struct_ptr  : [S+1]          int32    — CSR row pointer: pairs for struct s
                                                     are grad_values[struct_ptr[s]:struct_ptr[s+1]]
            positions   : [S, A, 3]      float32
            Z_int       : [S, A]         int32
            targets     : [S, T]         float32
            boxes       : [S, 3, 3]      float32
            atom_mask   : [S, A]         float32  — 1.0 for real atoms, 0.0 for padding
            num_atoms   : [S]            int32
        where S = num_structures, A = max_atoms, Q = dim_q, P = total atom-neighbor pairs
    """
    # Pick up streamed-to-disk gradient metadata stamped by split() / the
    # streaming test-data path. Explicit prebuilt_gv argument takes
    # precedence so callers can override.
    if prebuilt_gv is None:
        prebuilt_gv = data.get("_prebuilt_gv")

    S = len(data["descriptors"])
    dim_q = data["descriptors"][0].shape[-1]
    atom_counts = [data["descriptors"][i].shape[0] for i in range(S)]
    max_atoms = max(atom_counts)

    target_sample = data["targets"][0]
    target_dim = 1 if target_sample.shape == () else target_sample.shape[0]
    has_forces = "forces" in data
    has_virials = "virials" in data

    # Count pairs per structure to build CSR struct_ptr and size COO arrays.
    # When prebuilt_gv is provided, the gradient COO is already on disk —
    # we still need pair_counts/struct_ptr to slice the on-disk file per
    # chunk, but we don't allocate grad_values_np in RAM.
    if prebuilt_gv is not None:
        pair_counts = list(prebuilt_gv["pair_count_per_struct"])
    else:
        pair_counts = [
            sum(data["gradients"][s][i].shape[0] for i in range(atom_counts[s]))
            for s in range(S)
        ]
    N_pairs_total = sum(pair_counts)
    struct_ptr_np = np.zeros(S + 1, dtype=np.int32)
    for s in range(S):
        struct_ptr_np[s + 1] = struct_ptr_np[s] + pair_counts[s]

    # COO arrays: one entry per real atom-neighbor pair. grad_values_np is
    # only allocated in RAM if we don't already have a streamed disk file.
    if prebuilt_gv is None:
        grad_values_np = np.zeros((N_pairs_total, 3, dim_q), dtype=np.float32)
    pair_struct_np = np.zeros(N_pairs_total, dtype=np.int32)
    pair_atom_np   = np.zeros(N_pairs_total, dtype=np.int32)
    pair_gidx_np   = np.zeros(N_pairs_total, dtype=np.int32)

    # Structure-padded arrays (no M dimension)
    desc_np      = np.zeros((S, max_atoms, dim_q), dtype=np.float32)
    pos_np       = np.zeros((S, max_atoms, 3), dtype=np.float32)
    z_np         = np.zeros((S, max_atoms), dtype=np.int32)
    tgt_np       = np.zeros((S, target_dim), dtype=np.float32)
    box_np       = np.zeros((S, 3, 3), dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms), dtype=np.float32)
    num_atoms_np = np.array(atom_counts, dtype=np.int32)
    if has_forces:
        force_np = np.zeros((S, max_atoms, 3), dtype=np.float32)
    if has_virials:
        virial_np = np.zeros((S, 6), dtype=np.float32)
    if num_types is not None:
        types_contained_np = np.zeros((S, num_types), dtype=np.float32)

    pair_offset = 0
    for s in range(S):
        N_s = atom_counts[s]
        # Descriptors: numpy ndarray (streaming path) or TF tensor (legacy).
        d_s = data["descriptors"][s]
        desc_np[s, :N_s, :]  = d_s if isinstance(d_s, np.ndarray) else d_s.numpy()
        pos_np[s, :N_s, :]   = data["positions"][s].numpy()
        z_np[s, :N_s]        = data["Z_int"][s].numpy()
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

        if prebuilt_gv is not None:
            # Streaming path: pair_atom/pair_gidx already collected per
            # structure (frame-local). Just stamp pair_struct = s and
            # advance the offset by the recorded pair count.
            n_nbrs = pair_counts[s]
            k_end = pair_offset + n_nbrs
            if n_nbrs:
                pair_struct_np[pair_offset:k_end] = s
                pair_atom_np[pair_offset:k_end]   = prebuilt_gv["pair_atom_per_struct"][s]
                pair_gidx_np[pair_offset:k_end]   = prebuilt_gv["pair_gidx_per_struct"][s]
            pair_offset = k_end
        else:
            for i in range(N_s):
                n_nbrs = data["gradients"][s][i].shape[0]
                k_end  = pair_offset + n_nbrs
                grad_values_np[pair_offset:k_end] = data["gradients"][s][i].numpy()
                pair_struct_np[pair_offset:k_end] = s
                pair_atom_np[pair_offset:k_end]   = i
                pair_gidx_np[pair_offset:k_end]   = data["grad_index"][s][i]
                pair_offset += n_nbrs

    # Apply per-channel descriptor scaling (cfg.descriptor_scaling="q_scaler").
    # Multiplying BOTH desc and grad_values by the same s[Q] is the
    # data-pipeline equivalent of GPUMD's "scale q at ANN input + scale
    # Fp at ANN backward" pattern — see plan section "Why scaling
    # gradients is necessary". The streaming path (prebuilt_gv) has
    # grad_values on disk already and is rejected here; rebuild the
    # dataset without streaming if you need scaling.
    if q_scaler is not None:
        if prebuilt_gv is not None:
            raise ValueError(
                "q_scaler is not supported with the streaming "
                "(prebuilt_gv) path because grad_values is already on "
                "disk and cannot be modified in-place. Rebuild without "
                "streaming, or pre-scale the .bin file offline.")
        _apply_q_scaler_np(desc_np, grad_values_np, q_scaler)

    # Per-component target centering (cfg.target_centering=True). One
    # broadcast subtraction along the T_dim axis. Same mean is used for
    # train/val/test so the model sees a consistent zero point; the
    # mean is added back to predictions at the inference boundary so
    # user-facing values stay in the original units.
    if target_mean is not None:
        tm = np.asarray(target_mean, dtype=np.float32).reshape(-1)
        if tm.shape[0] != tgt_np.shape[1]:
            raise ValueError(
                f"target_mean has {tm.shape[0]} components but targets "
                f"have {tgt_np.shape[1]}; mismatch.")
        tgt_np -= tm[np.newaxis, :]

    # Convert each numpy array to a TF tensor then immediately delete the numpy
    # copy so peak RAM stays at ~1x dataset size rather than ~2x.
    # When gradient_cache_path is set, grad_values is the only field that
    # is *not* a TF tensor: it's written to disk and replaced by a numpy
    # memmap. slice_and_complete_chunk reads the per-chunk slice on
    # demand. Everything else stays as TF tensors as before.
    with tf.device('/CPU:0' if pin_to_cpu else '/GPU:0'):
        result = {}
        result["descriptors"] = tf.constant(desc_np);    del desc_np
        if prebuilt_gv is not None:
            # Streaming path: gradient bytes are already on disk. Just
            # memmap them — no in-RAM grad_values_np was ever allocated.
            gv_path = prebuilt_gv["gv_path"]
            gv_shape = prebuilt_gv["gv_shape"]
            gv_dtype = prebuilt_gv["gv_dtype"]
            result["grad_values"] = np.memmap(
                gv_path, dtype=gv_dtype, mode="r", shape=gv_shape)
            result["_gv_disk_backed"] = True
            print(f"  pad_and_stack: grad_values streamed at {gv_path} "
                  f"(shape={gv_shape}, "
                  f"{np.prod(gv_shape) * np.dtype(gv_dtype).itemsize / 1e9:.2f} GB)")
        elif gradient_cache_path is not None:
            import os as _os
            _os.makedirs(gradient_cache_path, exist_ok=True)
            gv_path = _os.path.join(
                gradient_cache_path, f"grad_values_{cache_tag}.bin")
            gv_shape = grad_values_np.shape
            gv_dtype = grad_values_np.dtype
            grad_values_np.tofile(gv_path)
            del grad_values_np
            # Memory-map view: scattered fancy-index reads serve per-chunk
            # slices at NVMe sequential bandwidth. The OS page cache
            # handles repeated access to the same structures.
            result["grad_values"] = np.memmap(
                gv_path, dtype=gv_dtype, mode="r", shape=gv_shape)
            result["_gv_disk_backed"] = True
            print(f"  pad_and_stack: grad_values cached at {gv_path} "
                  f"(shape={gv_shape}, "
                  f"{np.prod(gv_shape) * np.dtype(gv_dtype).itemsize / 1e9:.2f} GB)")
        else:
            result["grad_values"] = tf.constant(grad_values_np); del grad_values_np
        result["pair_struct"] = tf.constant(pair_struct_np); del pair_struct_np
        result["pair_atom"]   = tf.constant(pair_atom_np);   del pair_atom_np
        result["pair_gidx"]   = tf.constant(pair_gidx_np);   del pair_gidx_np
        result["struct_ptr"]  = tf.constant(struct_ptr_np);  del struct_ptr_np
        result["positions"]   = tf.constant(pos_np);         del pos_np
        result["Z_int"]       = tf.constant(z_np);           del z_np
        result["targets"]     = tf.constant(tgt_np);         del tgt_np
        result["boxes"]       = tf.constant(box_np);         del box_np
        result["atom_mask"]   = tf.constant(atom_mask_np);   del atom_mask_np
        result["num_atoms"]   = tf.constant(num_atoms_np);   del num_atoms_np
        if has_forces:
            result["forces"]  = tf.constant(force_np);       del force_np
        if has_virials:
            result["virials"] = tf.constant(virial_np);      del virial_np
        if num_types is not None:
            result["types_contained"] = tf.constant(types_contained_np); del types_contained_np
    return result


def pack_chunk_from_flat(frame_results: list, dim_q: int,
                          max_atoms: int | None = None) -> dict:
    """Pack per-frame TF tensors from build_descriptors_flat(return_tf=True)
    into a chunk-level dict with COO gradients and padded descriptors.

    Each frame_results[s] = (soap_t [N, Q], grad_t [P, 3, Q], pa_t [P], pg_t [P]),
    all TF tensors living on the descriptor builder's compute device. This packer
    concatenates them on-device with per-frame structure indices and pads
    descriptors to `max_atoms` so the resulting dict slots into the chunk
    evaluation path used by SNES._evaluate_chunk.

    Args:
        frame_results : list of per-frame (soap, grad, pa, pg) TF tensors.
        dim_q         : descriptor dimension (Q).
        max_atoms     : explicit padding length for the A axis. None = pad
                        to the chunk's own max(atom_counts). When the chunk
                        is being evaluated against tensors padded to a
                        wider A_max (e.g. the dataset-wide pad used by
                        positions/Z_int), pass that value to keep all
                        per-structure fields shape-consistent.

    Returns a dict with descriptor-shaped fields only:
        descriptors  [B, A_max, Q]   float32
        grad_values  [P, 3, Q]       float32
        pair_atom    [P]             int32   — frame-local center index
        pair_gidx    [P]             int32   — frame-local neighbour index
        pair_struct  [P]             int32   — chunk-local structure index
    """
    S = len(frame_results)
    atom_counts = [int(r[0].shape[0]) for r in frame_results]
    chunk_max_atoms = max(atom_counts) if atom_counts else 0
    if max_atoms is None:
        max_atoms = chunk_max_atoms
    elif max_atoms < chunk_max_atoms:
        raise ValueError(
            f"pack_chunk_from_flat: max_atoms={max_atoms} is smaller than the "
            f"chunk's own max(atom_counts)={chunk_max_atoms}.")
    pair_counts = [int(r[1].shape[0]) for r in frame_results]
    pair_counts_arr = np.array(pair_counts, dtype=np.int32)
    N_pairs = int(pair_counts_arr.sum())

    if N_pairs > 0:
        grad_values_t = tf.concat([r[1] for r in frame_results], axis=0)
        pair_atom_t   = tf.concat([r[2] for r in frame_results], axis=0)
        pair_gidx_t   = tf.concat([r[3] for r in frame_results], axis=0)
        pair_struct_t = tf.repeat(tf.range(S, dtype=tf.int32),
                                   tf.constant(pair_counts_arr, dtype=tf.int32))
    else:
        grad_values_t = tf.zeros((0, 3, dim_q), dtype=tf.float32)
        pair_atom_t   = tf.zeros((0,), dtype=tf.int32)
        pair_gidx_t   = tf.zeros((0,), dtype=tf.int32)
        pair_struct_t = tf.zeros((0,), dtype=tf.int32)

    if S:
        soap_concat = tf.concat([r[0] for r in frame_results], axis=0)
        frame_results.clear()
        desc_ragged = tf.RaggedTensor.from_row_lengths(
            soap_concat, tf.constant(atom_counts, dtype=tf.int64))
        desc_t = desc_ragged.to_tensor(default_value=0.0,
                                        shape=(S, max_atoms, dim_q))
        del soap_concat, desc_ragged
    else:
        desc_t = tf.zeros((0, 0, dim_q), dtype=tf.float32)
        frame_results.clear()

    return {
        "descriptors": desc_t,
        "grad_values": grad_values_t,
        "pair_atom":   pair_atom_t,
        "pair_gidx":   pair_gidx_t,
        "pair_struct": pair_struct_t,
    }


def slice_and_complete_chunk(data: dict, indices,
                              precomputed: dict | None = None) -> dict:
    """Build a chunk dict by slicing per-structure fields from `data`.

    The returned chunk has the same field contract that `SNES._evaluate_chunk`
    and `TNEP.predict_batch` consume:
        descriptors  [B_chunk, A_max, Q]
        grad_values  [P_chunk, 3, Q]
        pair_atom    [P_chunk]    int32  — frame-local centre index
        pair_gidx    [P_chunk]    int32  — frame-local neighbour index
        pair_struct  [P_chunk]    int32  — chunk-local structure index
        positions    [B_chunk, A_max, 3]
        Z_int        [B_chunk, A_max]
        boxes        [B_chunk, 3, 3]
        atom_mask    [B_chunk, A_max]
        num_atoms    [B_chunk]
        targets      [B_chunk, T]
        types_contained [B_chunk, T]   (only present when caller supplied it)

    When grad_values is a numpy memmap (cfg.cache_gradients_to_disk path),
    the chunk's pair slice is pulled from disk and shipped to the GPU as a
    fresh tf.constant; otherwise tf.gather slices the in-memory tensor.
    """
    if isinstance(indices, tf.Tensor):
        idx_tf = tf.cast(indices, tf.int32)
    else:
        idx_tf = tf.constant(np.asarray(indices, dtype=np.int32), dtype=tf.int32)

    chunk: dict = {}
    SMALL_KEYS = ("positions", "Z_int", "boxes", "num_atoms",
                  "targets", "atom_mask", "types_contained",
                  "forces", "virials")
    for k in SMALL_KEYS:
        if k in data:
            chunk[k] = tf.gather(data[k], idx_tf)

    chunk["descriptors"] = tf.gather(data["descriptors"], idx_tf)
    # COO pair gather: pairs for structure idx_tf[i] live in the slice
    # struct_ptr[idx_tf[i]] : struct_ptr[idx_tf[i]+1] of the flat
    # gradient/pair arrays. Build the flat pair-index list via
    # tf.ragged.range and gather. pair_struct is remapped to chunk-local
    # indices [0..B_chunk) via value_rowids(). When `precomputed` is
    # supplied (deterministic full-batch chunks) we skip the ragged
    # build and reuse the cached arrays.
    if precomputed is not None:
        flat_pair_idx_np = precomputed["flat_pair_idx_np"]
        flat_pair_idx_tf = precomputed["flat_pair_idx_tf"]
        chunk["pair_struct"] = precomputed["pair_struct_tf"]
    else:
        ptr = data["struct_ptr"]
        pair_starts = tf.gather(ptr, idx_tf)
        pair_ends   = tf.gather(ptr, idx_tf + 1)
        pair_ranges = tf.ragged.range(pair_starts, pair_ends)
        flat_pair_idx_tf = tf.cast(pair_ranges.flat_values, tf.int32)
        flat_pair_idx_np = None
        chunk["pair_struct"] = tf.cast(pair_ranges.value_rowids(), tf.int32)

    gv = data["grad_values"]
    if data.get("_gv_disk_backed", False):
        # Disk-backed: pull this chunk's pair slice from the memmap.
        # For deterministic contiguous-structure chunks (the full-batch
        # case, which is the dominant cost path) flat_pair_idx is a
        # contiguous arange, so a plain slice into the memmap is a
        # zero-copy view and the subsequent memcpy is *one* sequential
        # memcpy from page cache instead of the much more expensive
        # scatter-gather fancy-index. Random-index batches fall back to
        # the fancy-index path.
        if flat_pair_idx_np is None:
            flat_pair_idx_np = flat_pair_idx_tf.numpy()
        is_contig = (flat_pair_idx_np.size > 0
                     and int(flat_pair_idx_np[-1]) - int(flat_pair_idx_np[0]) + 1
                         == flat_pair_idx_np.size)
        if is_contig:
            lo = int(flat_pair_idx_np[0])
            hi = int(flat_pair_idx_np[-1]) + 1

        pool: "PinnedBufferPool | None" = data.get("_pinned_pool")
        buf = pool.acquire() if pool is not None else None
        if buf is not None:
            # Memcpy directly into pinned host memory; tf.constant from a
            # pinned source takes the async cudaMemcpyAsync path (no
            # driver bounce). The DMA may still be reading the buffer
            # *after* tf.constant returns — so we attach a holder that
            # only releases the buffer back to the pool when the chunk
            # dict itself is garbage-collected (which happens after the
            # consumer has used the GPU tensor and the DMA has been
            # implicitly synchronised by the next op).
            if is_contig:
                src = gv[lo:hi]
            else:
                src = gv[flat_pair_idx_np]
            pinned_view = buf.view_as(gv.dtype, src.shape)
            np.copyto(pinned_view, src, casting="no")
            chunk["grad_values"] = tf.constant(pinned_view)
            chunk["_pinned_holder"] = _PinnedBufferHolder(buf, pool)
        else:
            # Pageable fallback: numpy allocates, tf.constant uses driver
            # bounce buffer. Still correct, just slower.
            if is_contig:
                chunk_grad_np = np.ascontiguousarray(gv[lo:hi])
            else:
                chunk_grad_np = np.asarray(gv[flat_pair_idx_np])
            chunk["grad_values"] = tf.constant(chunk_grad_np)
    else:
        chunk["grad_values"] = tf.gather(gv, flat_pair_idx_tf)
    chunk["pair_atom"]   = tf.gather(data["pair_atom"],   flat_pair_idx_tf)
    chunk["pair_gidx"]   = tf.gather(data["pair_gidx"],   flat_pair_idx_tf)
    return chunk


# ============================================================================
# Chunk staging + prefetch + GPU LRU cache
# ============================================================================

class PinnedBuffer:
    """Page-locked host buffer allocated via cudaMallocHost.

    Pinned host memory enables true async cudaMemcpyAsync from host to
    GPU without a driver-managed bounce buffer (which silently downgrades
    pageable transfers to a synchronous staging copy + async DMA). For
    bulk H2D the resulting bandwidth is ~12-16 GB/s on PCIe Gen4 vs
    ~6-8 GB/s with pageable.

    The numpy `.uint8_view` is a writable view directly over the pinned
    pages; reshape with `view_as(dtype, shape)`.
    """
    def __init__(self, nbytes: int):
        import ctypes as _ct
        cuda = _get_cudart()
        cuda.cudaMallocHost.restype = _ct.c_int
        cuda.cudaMallocHost.argtypes = [_ct.POINTER(_ct.c_void_p), _ct.c_size_t]
        cuda.cudaFreeHost.restype = _ct.c_int
        cuda.cudaFreeHost.argtypes = [_ct.c_void_p]
        self._p = _ct.c_void_p()
        err = cuda.cudaMallocHost(_ct.byref(self._p), _ct.c_size_t(int(nbytes)))
        if err != 0:
            raise RuntimeError(f"cudaMallocHost({nbytes}) failed with code {err}")
        self.nbytes = int(nbytes)
        # Stable raw view; numpy slices/reshapes return views over this.
        self._raw = (_ct.c_uint8 * self.nbytes).from_address(self._p.value)

    def view_as(self, dtype, shape):
        elems = 1
        for d in shape:
            elems *= int(d)
        item = int(np.dtype(dtype).itemsize)
        if elems * item > self.nbytes:
            raise ValueError(
                f"Requested {elems * item} bytes exceeds buffer {self.nbytes}.")
        return np.frombuffer(self._raw, dtype=dtype, count=elems).reshape(shape)

    def __del__(self):
        # Defensive: cudart may have been unloaded by interpreter shutdown.
        try:
            if getattr(self, "_p", None) is not None and self._p.value:
                _get_cudart().cudaFreeHost(self._p)
                self._p.value = 0
        except Exception:
            pass


_cudart = None
def _get_cudart():
    """Lazy CUDA runtime loader. Returns the loaded ctypes handle or
    raises OSError if cudart is unavailable on this system."""
    global _cudart
    if _cudart is None:
        import ctypes as _ct
        for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                _cudart = _ct.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise OSError("Could not load libcudart.so")
    return _cudart


class PinnedBufferPool:
    """FIFO pool of pinned host buffers for the disk-backed staging path.

    Buffers are checked out by `slice_and_complete_chunk`, used as the
    destination for the memmap → host memcpy, and passed to tf.constant
    (which dispatches an async DMA). The buffer is then bundled into the
    chunk dict via `_PinnedBufferHolder`, so it is only returned to the
    pool when the chunk dict is destroyed — by which point the consumer
    has used the GPU tensor and the DMA has been implicitly synchronised
    by the next op on the same stream.

    Pool size of 4 is enough for prefetch depth 1-2 plus a current
    in-flight chunk on the GPU side. acquire() returns None when the
    pool is exhausted; the caller falls back to pageable.
    """
    def __init__(self, n_buffers: int, buffer_nbytes: int):
        from collections import deque
        self.buffer_nbytes = int(buffer_nbytes)
        self._all = [PinnedBuffer(buffer_nbytes) for _ in range(n_buffers)]
        self._free = deque(self._all)

    def acquire(self):
        if not self._free:
            return None
        return self._free.popleft()

    def release(self, buf):
        if buf is None:
            return
        self._free.append(buf)


class _PinnedBufferHolder:
    """Lifetime cookie that returns a pinned buffer to its pool when
    the holder is garbage-collected. Stored inside the chunk dict so the
    buffer's release is tied to the consumer dropping the chunk."""
    __slots__ = ("buf", "pool")

    def __init__(self, buf, pool):
        self.buf = buf
        self.pool = pool

    def __del__(self):
        # Defensive: pool / buf may already be gone at interpreter shutdown.
        try:
            if self.buf is not None and self.pool is not None:
                self.pool.release(self.buf)
        except Exception:
            pass
        self.buf = None
        self.pool = None


def make_pinned_pool_for(data: dict, batch_chunk_size: int,
                          n_buffers: int = 4) -> "PinnedBufferPool | None":
    """Build a pinned-buffer pool sized to hold the worst-case grad slice
    that `slice_and_complete_chunk` will need to stage from `data`.

    Worst-case bytes = max(struct_ptr[s + chunk] - struct_ptr[s]) × 3 × Q
    × itemsize, with `chunk = batch_chunk_size`. Returns None when cudart
    can't be loaded (CUDA-less host) — caller falls back to pageable.
    """
    try:
        _get_cudart()
    except OSError:
        return None
    if "grad_values" not in data or "struct_ptr" not in data:
        return None
    gv = data["grad_values"]
    if gv.shape is None or len(gv.shape) < 3:
        return None
    Q = int(gv.shape[2])
    item = int(np.dtype(gv.dtype).itemsize)
    sp = data["struct_ptr"].numpy() if hasattr(data["struct_ptr"], "numpy") else np.asarray(data["struct_ptr"])
    S = int(sp.shape[0]) - 1
    chunk = int(batch_chunk_size) if batch_chunk_size is not None else S
    chunk = max(1, min(chunk, S))
    # Sliding window max of struct_ptr deltas; small, fast.
    max_pairs = 0
    for s in range(0, S, chunk):
        e = min(s + chunk, S)
        d = int(sp[e]) - int(sp[s])
        if d > max_pairs:
            max_pairs = d
    if max_pairs == 0:
        return None
    # 5% headroom for ragged endcaps + alignment padding.
    nbytes = int(max_pairs * 3 * Q * item * 1.05)
    try:
        return PinnedBufferPool(n_buffers, nbytes)
    except RuntimeError:
        return None


def prestage_chunk_indices(data: dict, ranges: list,
                            pad_to: int | None = None) -> None:
    """Pre-build GPU tensors for the per-chunk pair indices
    (pair_atom, pair_gidx, pair_struct) for each (s, e) in `ranges`.

    For deterministic full-batch chunks, these arrays are constant across
    generations — staging them once at startup eliminates ~3-5 ms/chunk of
    per-gen `tf.constant` + DMA work. Stored on the data dict under
    `_pair_idx_gpu_cache` and consumed by `_stage_finalize_tf` when present.

    `pad_to` is forwarded to `data["_max_chunk_pairs"]` so eval-side
    padding can read it; the *pre-staged* tensors stay unpadded so the
    validate path (which uses raw model.predict_batch with no padding)
    keeps shape-consistent inputs. The eval path applies tf.pad
    on-the-fly when XLA is on.

    Memory cost: ~3 × P_chunk × 4 bytes per chunk; tiny.
    """
    cache: dict = data.setdefault("_pair_idx_gpu_cache", {})
    if pad_to is not None:
        data["_max_chunk_pairs"] = int(pad_to)
    pa_full = data["pair_atom"]
    pg_full = data["pair_gidx"]
    pa_np = pa_full.numpy() if hasattr(pa_full, "numpy") else np.asarray(pa_full)
    pg_np = pg_full.numpy() if hasattr(pg_full, "numpy") else np.asarray(pg_full)
    idx_cache = get_chunk_index_cache()
    with tf.device('/GPU:0'):
        for s, e in ranges:
            key = (int(s), int(e))
            if key in cache:
                continue
            precomp = idx_cache.get(data, s, e)
            flat = precomp["flat_pair_idx_np"]
            pair_struct_np = precomp["pair_struct_tf"].numpy()
            cache[key] = {
                "pair_atom":   tf.constant(pa_np[flat]),
                "pair_gidx":   tf.constant(pg_np[flat]),
                "pair_struct": tf.constant(pair_struct_np),
                "_real_P":     int(flat.shape[0]),
            }


def compute_max_chunk_pairs(data: dict, ranges: list) -> int:
    """Worst-case pair count across the given chunk ranges. Used to pad
    grad_values / pair indices for XLA-compiled eval."""
    sp = data["struct_ptr"].numpy() if hasattr(data["struct_ptr"], "numpy") else np.asarray(data["struct_ptr"])
    m = 0
    for s, e in ranges:
        d = int(sp[int(e)]) - int(sp[int(s)])
        if d > m:
            m = d
    return int(m)


class ChunkIndexCache:
    """Caches the deterministic-per-chunk artefacts that
    slice_and_complete_chunk would otherwise rebuild every call:
    the flat pair-index array (`flat_pair_idx`) and the chunk-local
    `pair_struct` mapping. Keyed by (id(data), s_start, s_end). Cheap
    to build, useful for full-batch where the same chunks repeat every
    generation; harmless on cache miss for finite batches.
    """
    def __init__(self):
        self._cache: dict = {}

    def get(self, data: dict, s_start: int, s_end: int) -> dict:
        key = (id(data), int(s_start), int(s_end))
        item = self._cache.get(key)
        if item is None:
            idx_tf = tf.range(s_start, s_end, dtype=tf.int32)
            ptr = data["struct_ptr"]
            pair_starts = tf.gather(ptr, idx_tf)
            pair_ends   = tf.gather(ptr, idx_tf + 1)
            pair_ranges = tf.ragged.range(pair_starts, pair_ends)
            flat_pair_idx_tf = tf.cast(pair_ranges.flat_values, tf.int32)
            pair_struct_tf   = tf.cast(pair_ranges.value_rowids(), tf.int32)
            # Materialise the int32 indices to numpy so the disk-backed
            # path can do the memmap fancy-index without an extra sync
            # per call. Small (< 100 K ints typically).
            flat_pair_idx_np = flat_pair_idx_tf.numpy()
            item = {
                "flat_pair_idx_tf": flat_pair_idx_tf,
                "flat_pair_idx_np": flat_pair_idx_np,
                "pair_struct_tf":   pair_struct_tf,
            }
            self._cache[key] = item
        return item

    def clear(self):
        self._cache.clear()


# Process-wide singleton; created lazily because tf.range needs TF imported.
_chunk_index_cache: ChunkIndexCache | None = None


def get_chunk_index_cache() -> ChunkIndexCache:
    global _chunk_index_cache
    if _chunk_index_cache is None:
        _chunk_index_cache = ChunkIndexCache()
    return _chunk_index_cache


def _stage_disk_only(data: dict, s_start: int, s_end: int) -> dict:
    """Worker-safe portion of chunk staging: only numpy + memmap, no TF.

    Reads the chunk's gradient pair slice into a pinned (or pageable)
    host buffer and packages every other slice as a small ndarray. The
    main thread then converts this dict to TF tensors via
    `_stage_finalize_tf`. Splitting the staging like this avoids TF
    eager mode's thread-safety pitfalls — calling tf.constant /
    tf.gather from a non-main thread occasionally produces tensors with
    corrupted shape descriptors (rank or dim mismatches) under load,
    which surfaces later as cryptic StridedSlice / shape errors.
    """
    precomputed = get_chunk_index_cache().get(data, s_start, s_end)
    idx_np = np.arange(int(s_start), int(s_end), dtype=np.int32)
    flat_pair_idx_np = precomputed["flat_pair_idx_np"]

    # Slice every CPU-resident structure-padded field into numpy.
    # tf.gather equivalents will run in the main thread.
    out: dict = {"_idx_np": idx_np, "_precomputed": precomputed}
    SMALL_KEYS = ("positions", "Z_int", "boxes", "num_atoms",
                  "targets", "atom_mask", "types_contained",
                  "forces", "virials")
    for k in SMALL_KEYS:
        if k in data:
            v = data[k]
            np_view = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
            out["_np_" + k] = np_view[idx_np]
    desc = data["descriptors"]
    desc_np = desc.numpy() if hasattr(desc, "numpy") else np.asarray(desc)
    out["_np_descriptors"] = desc_np[idx_np]

    # Gradient: disk-backed → memmap copy into pinned (or pageable) numpy.
    # When cuFile is configured *and* the chunk's pair indices form a
    # contiguous range, we issue a direct disk→GPU read instead — pure C
    # call, worker-safe, and ~4× faster than the host-memcpy path on WSL
    # compat-mode (saturates PCIe Gen4 once warm).
    gv = data["grad_values"]
    if data.get("_gv_disk_backed", False):
        is_contig = (flat_pair_idx_np.size > 0
                     and int(flat_pair_idx_np[-1]) - int(flat_pair_idx_np[0]) + 1
                         == flat_pair_idx_np.size)
        if is_contig:
            lo = int(flat_pair_idx_np[0])
            hi = int(flat_pair_idx_np[-1]) + 1

        cf_ctx = data.get("_cufile_ctx")
        if cf_ctx is not None and is_contig:
            cf_pool = cf_ctx["pool"]
            cf_buf = cf_pool.acquire()
            if cf_buf is not None:
                Q = int(gv.shape[2])
                item = int(np.dtype(gv.dtype).itemsize)
                file_offset = int(lo) * 3 * Q * item
                nbytes = int(hi - lo) * 3 * Q * item
                try:
                    cf_ctx["handle"].read(cf_buf.devptr, nbytes, file_offset)
                    out["_cufile_buf"] = cf_buf
                    out["_cufile_pool"] = cf_pool
                    out["_cufile_shape"] = (int(hi - lo), 3, Q)
                    out["_cufile_dtype"] = gv.dtype
                    pa_np = data["pair_atom"].numpy() if hasattr(data["pair_atom"], "numpy") else np.asarray(data["pair_atom"])
                    pg_np = data["pair_gidx"].numpy() if hasattr(data["pair_gidx"], "numpy") else np.asarray(data["pair_gidx"])
                    out["_np_pair_atom"] = pa_np[flat_pair_idx_np]
                    out["_np_pair_gidx"] = pg_np[flat_pair_idx_np]
                    return out
                except Exception:
                    cf_pool.release(cf_buf)
                    # Fall through to pinned/pageable.

        if is_contig:
            src = gv[lo:hi]
        else:
            src = gv[flat_pair_idx_np]
        pool: "PinnedBufferPool | None" = data.get("_pinned_pool")
        buf = pool.acquire() if pool is not None else None
        if buf is not None:
            view = buf.view_as(gv.dtype, src.shape)
            np.copyto(view, src, casting="no")
            out["_np_grad_values"] = view
            out["_pinned_buf"] = buf
            out["_pinned_pool"] = pool
        else:
            out["_np_grad_values"] = np.ascontiguousarray(src)
        # Pair indices: use numpy view of pair_atom/pair_gidx, gathered with
        # the cached flat_pair_idx_np.
        pa_np = data["pair_atom"].numpy() if hasattr(data["pair_atom"], "numpy") else np.asarray(data["pair_atom"])
        pg_np = data["pair_gidx"].numpy() if hasattr(data["pair_gidx"], "numpy") else np.asarray(data["pair_gidx"])
        out["_np_pair_atom"] = pa_np[flat_pair_idx_np]
        out["_np_pair_gidx"] = pg_np[flat_pair_idx_np]
    else:
        # In-RAM: hand back the original tensors / arrays untouched.
        # Main thread will tf.gather them.
        out["_passthrough_grad_values"] = data["grad_values"]
        out["_passthrough_pair_atom"]   = data["pair_atom"]
        out["_passthrough_pair_gidx"]   = data["pair_gidx"]
    return out


def _stage_finalize_tf(data: dict, raw: dict, pin_to_cpu: bool,
                        s_start: int | None = None,
                        s_end: int | None = None,
                        pad_pairs_to: int | None = None) -> dict:
    """Main-thread half of staging: convert the worker's numpy output to
    TF tensors. Cheap (just tf.constant calls), runs in the foreground.

    When `data["_pair_idx_gpu_cache"]` has a pre-staged entry for the
    chunk's (s_start, s_end) range, the deterministic pair-index tensors
    (pair_atom, pair_gidx, pair_struct) are reused from the cache instead
    of being rebuilt each call. The grad_values still goes through
    cuFile / pinned / pageable as usual."""
    chunk: dict = {}
    SMALL_KEYS = ("positions", "Z_int", "boxes", "num_atoms",
                  "targets", "atom_mask", "types_contained",
                  "forces", "virials")
    device = '/GPU:0' if pin_to_cpu else None
    ctx = tf.device(device) if device is not None else _NullCtx()
    pair_idx_cache = data.get("_pair_idx_gpu_cache")
    pair_idx_entry = None
    if pair_idx_cache is not None and s_start is not None and s_end is not None:
        pair_idx_entry = pair_idx_cache.get((int(s_start), int(s_end)))
    with ctx:
        for k in SMALL_KEYS:
            np_key = "_np_" + k
            if np_key in raw:
                chunk[k] = tf.constant(raw[np_key])
        chunk["descriptors"] = tf.constant(raw["_np_descriptors"])
        if pair_idx_entry is not None:
            chunk["pair_struct"] = pair_idx_entry["pair_struct"]
        else:
            chunk["pair_struct"] = (tf.identity(raw["_precomputed"]["pair_struct_tf"])
                                     if pin_to_cpu else
                                     raw["_precomputed"]["pair_struct_tf"])
        if "_cufile_buf" in raw:
            # cuFile path: grad_values already on the GPU, wrap as TF
            # tensor via a manual DLPack capsule whose deleter releases
            # the buffer back to the pool when TF destroys the tensor.
            from cufile_io import build_dlpack_capsule
            cf_buf  = raw["_cufile_buf"]
            cf_pool = raw["_cufile_pool"]
            shape   = raw["_cufile_shape"]
            dtype   = raw["_cufile_dtype"]
            def _release(_buf=cf_buf, _pool=cf_pool):
                _pool.release(_buf)
            capsule = build_dlpack_capsule(cf_buf.devptr, dtype, shape, _release)
            chunk["grad_values"] = tf.experimental.dlpack.from_dlpack(capsule)
            if pair_idx_entry is not None:
                chunk["pair_atom"] = pair_idx_entry["pair_atom"]
                chunk["pair_gidx"] = pair_idx_entry["pair_gidx"]
            else:
                chunk["pair_atom"] = tf.constant(raw["_np_pair_atom"])
                chunk["pair_gidx"] = tf.constant(raw["_np_pair_gidx"])
        elif "_np_grad_values" in raw:
            chunk["grad_values"] = tf.constant(raw["_np_grad_values"])
            if pair_idx_entry is not None:
                chunk["pair_atom"] = pair_idx_entry["pair_atom"]
                chunk["pair_gidx"] = pair_idx_entry["pair_gidx"]
            else:
                chunk["pair_atom"] = tf.constant(raw["_np_pair_atom"])
                chunk["pair_gidx"] = tf.constant(raw["_np_pair_gidx"])
            buf = raw.get("_pinned_buf")
            pool = raw.get("_pinned_pool")
            if buf is not None and pool is not None:
                chunk["_pinned_holder"] = _PinnedBufferHolder(buf, pool)
        else:
            # In-RAM passthrough. When the chunk's pair indices form a
            # contiguous range (the common case for sequential
            # full-batch chunking), use tf.strided_slice — pure GPU
            # slice, no D2D gather of the full chunk. Otherwise (random
            # sub-sampling) fall back to tf.gather.
            flat_pair_idx_np = raw["_precomputed"]["flat_pair_idx_np"]
            is_contig = (flat_pair_idx_np.size > 0
                         and int(flat_pair_idx_np[-1]) - int(flat_pair_idx_np[0]) + 1
                             == flat_pair_idx_np.size)
            # When grad_values is CPU-resident (pin_to_cpu=True),
            # both the slice/gather AND the resulting chunk must stay
            # on CPU until the consumer explicitly moves them — TF's
            # default-device placement otherwise tries to materialise
            # the entire CPU tensor on GPU before slicing, producing
            # a massive D2H→H2D round-trip and OOM at large grad_values.
            gv_dev = raw["_passthrough_grad_values"].device
            slice_dev = '/CPU:0' if pin_to_cpu else (gv_dev or '/GPU:0')
            with tf.device(slice_dev):
                if is_contig:
                    lo = int(flat_pair_idx_np[0])
                    hi = int(flat_pair_idx_np[-1]) + 1
                    gv = raw["_passthrough_grad_values"]
                    chunk["grad_values"] = gv[lo:hi]
                    if pair_idx_entry is not None:
                        chunk["pair_atom"] = pair_idx_entry["pair_atom"]
                        chunk["pair_gidx"] = pair_idx_entry["pair_gidx"]
                    else:
                        chunk["pair_atom"] = raw["_passthrough_pair_atom"][lo:hi]
                        chunk["pair_gidx"] = raw["_passthrough_pair_gidx"][lo:hi]
                else:
                    flat_pair_idx_tf = tf.constant(flat_pair_idx_np)
                    chunk["grad_values"] = tf.gather(
                        raw["_passthrough_grad_values"], flat_pair_idx_tf)
                    chunk["pair_atom"] = tf.gather(
                        raw["_passthrough_pair_atom"], flat_pair_idx_tf)
                    chunk["pair_gidx"] = tf.gather(
                        raw["_passthrough_pair_gidx"], flat_pair_idx_tf)

        # Optional pad of all four pair-aligned tensors to a fixed size
        # so XLA-compiled `_evaluate_chunk` sees one (B, P) shape across
        # all chunks and compiles once. Padded entries contribute zero
        # to the per-structure dipole sum (grad_values pad is zero;
        # pair_struct/atom pad to 0 → zero contribution into segment 0).
        if pad_pairs_to is not None:
            P_real = int(chunk["grad_values"].shape[0])
            if P_real < int(pad_pairs_to):
                npad = int(pad_pairs_to) - P_real
                chunk["grad_values"] = tf.pad(
                    chunk["grad_values"], [[0, npad], [0, 0], [0, 0]])
                chunk["pair_atom"]   = tf.pad(chunk["pair_atom"],   [[0, npad]])
                chunk["pair_gidx"]   = tf.pad(chunk["pair_gidx"],   [[0, npad]])
                chunk["pair_struct"] = tf.pad(chunk["pair_struct"], [[0, npad]])
    return chunk


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


_RESIDENT_SMALL_KEYS = ("positions", "Z_int", "boxes", "num_atoms",
                         "targets", "atom_mask", "types_contained",
                         "forces", "virials")


def move_data_to_gpu(data: dict) -> None:
    """Move every static (non-staging-helper) field in `data` onto
    `/GPU:0`. Used after the GPU-resident grad cache is built so the
    chunk-staging path doesn't have to round-trip through host
    numpy. Tensors already on the GPU are left alone (tf.identity
    inside `with tf.device('/GPU:0')` is a no-op for resident
    tensors). Underscore-prefixed keys (helper objects) are skipped.
    """
    keys = list(_RESIDENT_SMALL_KEYS) + [
        "descriptors", "pair_atom", "pair_gidx", "pair_struct",
        "struct_ptr", "grad_values"]
    with tf.device('/GPU:0'):
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            if hasattr(v, "device"):
                # tf.Tensor: identity-on-GPU is cheap if already there.
                data[k] = tf.identity(v)
            else:
                # numpy / memmap (only valid when not _gv_disk_backed)
                data[k] = tf.constant(np.asarray(v))


def _stage_chunk_resident(data: dict, s_start: int, s_end: int,
                           pad_pairs_to: int | None = None) -> dict:
    """Pure-GPU chunk staging for `_gv_resident_gpu` data dicts.

    All inputs are GPU tensors. Per-chunk work is tf.gather on a
    handful of small [B]-axis fields, a tf.strided_slice on
    grad_values, optional tf.pad — all on-device. No worker thread,
    no numpy, no host↔device traffic. Replaces the
    `_stage_disk_only` + `_stage_finalize_tf` two-phase path
    entirely for the GPU-resident case.
    """
    chunk: dict = {}
    s_lo = int(s_start)
    s_hi = int(s_end)
    pair_idx_cache = data.get("_pair_idx_gpu_cache")
    pair_idx_entry = (pair_idx_cache.get((s_lo, s_hi))
                       if pair_idx_cache is not None else None)
    with tf.device('/GPU:0'):
        idx_tf = tf.range(s_lo, s_hi, dtype=tf.int32)
        for k in _RESIDENT_SMALL_KEYS:
            v = data.get(k)
            if v is None:
                continue
            chunk[k] = tf.gather(v, idx_tf)
        chunk["descriptors"] = tf.gather(data["descriptors"], idx_tf)

        # Grad slice + pair indices.
        precomputed = get_chunk_index_cache().get(data, s_lo, s_hi)
        flat_pair_idx_np = precomputed["flat_pair_idx_np"]
        if flat_pair_idx_np.size > 0:
            lo = int(flat_pair_idx_np[0])
            hi = int(flat_pair_idx_np[-1]) + 1
            is_contig = (hi - lo) == flat_pair_idx_np.size
        else:
            lo = hi = 0
            is_contig = True
        gv = data["grad_values"]
        if is_contig:
            grad_slc = gv[lo:hi]
            if pair_idx_entry is not None:
                pair_atom = pair_idx_entry["pair_atom"]
                pair_gidx = pair_idx_entry["pair_gidx"]
            else:
                pair_atom = data["pair_atom"][lo:hi]
                pair_gidx = data["pair_gidx"][lo:hi]
        else:
            flat_tf = tf.constant(flat_pair_idx_np)
            grad_slc = tf.gather(gv, flat_tf)
            pair_atom = tf.gather(data["pair_atom"], flat_tf)
            pair_gidx = tf.gather(data["pair_gidx"], flat_tf)
        if pair_idx_entry is not None:
            pair_struct = pair_idx_entry["pair_struct"]
        else:
            pair_struct = precomputed["pair_struct_tf"]

        if pad_pairs_to is not None:
            P_real = int(grad_slc.shape[0])
            if P_real < int(pad_pairs_to):
                npad = int(pad_pairs_to) - P_real
                grad_slc = tf.pad(grad_slc, [[0, npad], [0, 0], [0, 0]])
                pair_atom = tf.pad(pair_atom, [[0, npad]])
                pair_gidx = tf.pad(pair_gidx, [[0, npad]])
                pair_struct = tf.pad(pair_struct, [[0, npad]])
        chunk["grad_values"] = grad_slc
        chunk["pair_atom"] = pair_atom
        chunk["pair_gidx"] = pair_gidx
        chunk["pair_struct"] = pair_struct
    return chunk


def stage_chunk(data: dict, s_start: int, s_end: int,
                pin_to_cpu: bool = False,
                pad_pairs_to: int | None = None) -> tuple:
    """Single-thread fallback: do disk-only + finalize-tf in sequence.

    Used for the depth=1, prefetch=False case and as a synchronous
    fallback when the prefetch worker errors out.
    """
    raw = _stage_disk_only(data, s_start, s_end)
    chunk = _stage_finalize_tf(data, raw, pin_to_cpu=pin_to_cpu,
                                s_start=s_start, s_end=s_end,
                                pad_pairs_to=pad_pairs_to)
    return s_start, s_end, chunk


def prefetched_chunks(data: dict, ranges: list, pin_to_cpu: bool,
                      enabled: bool = True,
                      depth: int = 1,
                      pad_pairs_to: int | None = None):
    """Yield (s_start, s_end, chunk) tuples for each (s, e) in `ranges`.

    Three modes, picked by the data dict's state:

    1. **GPU-resident** (`_gv_resident_gpu=True`): chunks are built
       purely on-device via `_stage_chunk_resident`. No worker
       thread, no host↔GPU traffic. Fastest mode — used when the
       grad cache fits in VRAM.

    2. **Disk-backed + prefetch** (`enabled=True`, `depth >= 1`): up
       to `depth` background threads stage chunks N+1..N+depth while
       the consumer evaluates chunk N, overlapping disk→host→GPU
       traffic with GPU compute. Useful when the grad cache is too
       big to fit in VRAM.

    3. **Disk-backed serial** (`enabled=False` or only one range):
       single-threaded staging. Fallback / debugging path.

    Memory cost per pipeline slot in mode 2: one chunk's grad slice
    transient (~few hundred MB at full-batch `batch_chunk_size=500`).
    The pinned / cuFile pool size must be >= depth + 1 — see
    cfg.pinned_pool_size / cfg.cufile_pool_size.
    """
    if not ranges:
        return

    if data.get("_gv_resident_gpu", False):
        for s, e in ranges:
            yield int(s), int(e), _stage_chunk_resident(
                data, int(s), int(e), pad_pairs_to=pad_pairs_to)
        return

    # Pre-warm the chunk-index cache from the main thread so workers
    # never trigger a TF op on cache miss. Calling tf.range / tf.gather
    # / tf.ragged.range from a non-main eager thread occasionally
    # produces tensors with corrupted shape descriptors under
    # concurrency, surfacing later as cryptic StridedSlice / dim
    # errors.
    idx_cache = get_chunk_index_cache()
    for _s, _e in ranges:
        idx_cache.get(data, _s, _e)

    def _stage_worker(s, e):
        return _stage_disk_only(data, s, e)

    def _finalize(s, e, raw):
        return _stage_finalize_tf(data, raw, pin_to_cpu=pin_to_cpu,
                                   s_start=s, s_end=e,
                                   pad_pairs_to=pad_pairs_to)

    depth = max(1, int(depth))
    if not enabled or len(ranges) <= 1:
        for s, e in ranges:
            raw = _stage_worker(s, e)
            yield s, e, _finalize(s, e, raw)
        return

    import concurrent.futures
    n_ranges = len(ranges)
    depth = min(depth, n_ranges)
    with concurrent.futures.ThreadPoolExecutor(max_workers=depth) as pool:
        slots: list = [(ranges[i][0], ranges[i][1],
                        pool.submit(_stage_worker, *ranges[i]))
                       for i in range(depth)]
        next_to_submit = depth
        for i in range(n_ranges):
            slot = i % depth
            s, e, fut = slots[slot]
            try:
                raw = fut.result()
            except BaseException:
                raw = _stage_worker(s, e)
            yield_chunk = _finalize(s, e, raw)

            if next_to_submit < n_ranges:
                ns, ne = ranges[next_to_submit]
                slots[slot] = (ns, ne, pool.submit(_stage_worker, ns, ne))
                next_to_submit += 1
            yield s, e, yield_chunk


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
    convert = getattr(cfg, "convert_dipole_to_eangstrom", True)
    factor = _dipole_conversion_factor(cfg.dipole_units) if convert else 1.0
    if factor != 1.0:
        dipoles = dipoles * factor
    if convert:
        unit = f"e\u00b7\u00c5 (from {cfg.dipole_units})" if factor != 1.0 else "e\u00b7\u00c5"
    else:
        unit = f"{cfg.dipole_units} (no conversion)"
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
