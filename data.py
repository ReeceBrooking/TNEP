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
    """Find structures with NaN or missing targets.

    Args:
        dataset    : list of ase.Atoms
        target_key : key for the target property (e.g. 'dipole', 'pol')

    Returns:
        dict with keys 'nan_positions', 'nan_targets', 'missing_targets',
        each mapping to a list of structure indices.
    """
    nan_positions = []
    nan_targets = []
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
    return {'nan_positions': nan_positions,
            'nan_targets': nan_targets,
            'missing_targets': missing_targets}


def filter_bad_data(
    dataset: list[Atoms],
    dataset_types_int: list[np.ndarray],
    cfg: TNEPconfig,
) -> tuple[list[Atoms], list[np.ndarray]]:
    """Remove untrainable structures (NaN positions, NaN targets, missing targets).

    Always runs. Structures with NaN positions, NaN targets, or a missing
    target key can't be trained against and are dropped. If any are dropped,
    a warning listing the per-category and total counts is printed.

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    target_key = _resolve_target_key(cfg)
    bad = find_bad_data(dataset, target_key)

    bad_indices: set[int] = set()
    for kind in ('nan_positions', 'nan_targets', 'missing_targets'):
        bad_indices.update(bad[kind])

    if bad_indices:
        dataset = [s for i, s in enumerate(dataset) if i not in bad_indices]
        dataset_types_int = [t for i, t in enumerate(dataset_types_int) if i not in bad_indices]
        print(
            f"[filter_bad_data] WARNING: dropped {len(bad_indices)} structures "
            f"(nan_targets={len(bad['nan_targets'])}, "
            f"nan_positions={len(bad['nan_positions'])}, "
            f"missing_targets={len(bad['missing_targets'])})"
        )

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
    _self_only = (cfg.target_mode == 1
                  and int(getattr(cfg, "dipole_rij_power", 2)) == 0)
    return pad_and_stack(
        data,
        num_types=cfg.num_types,
        self_pairs_only=_self_only)


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
    # When `dipole_rij_power == 0` the dipole forward consumes only
    # self-pair gradients ∂q_i/∂r_i. Build in batches and immediately
    # drop the neighbour-gradient rows that the forward will never
    # read — bounds peak memory to one batch's worth of the
    # ~90%-of-COO neighbour-gradient tensor (see
    # `descriptor_self_batch_size` doc).
    _self_only_batch = (int(cfg.descriptor_self_batch_size)
                        if getattr(cfg, "descriptor_self_batch_size", None)
                        is not None else None)
    _self_only = (int(getattr(cfg, "dipole_rij_power", 0)) == 0
                  and cfg.target_mode == 1)
    if _self_only:
        _kw = {"progress_desc": "Building train descriptors (self-only, batched)"} \
            if cfg.descriptor_mode == 1 else {}
        train_descriptors, train_gradients, train_grad_index = \
            builder.build_descriptors_self_only(
                train_dataset, batch_size=_self_only_batch, **_kw)
        _kw = {"progress_desc": "Building val descriptors (self-only, batched)"} \
            if cfg.descriptor_mode == 1 else {}
        val_descriptors, val_gradients, val_grad_index = \
            builder.build_descriptors_self_only(
                val_dataset, batch_size=_self_only_batch, **_kw)
    else:
        _kw = {"progress_desc": "Building train descriptors"} if cfg.descriptor_mode == 1 else {}
        train_descriptors, train_gradients, train_grad_index = builder.build_descriptors(train_dataset, **_kw)
        _kw = {"progress_desc": "Building val descriptors"} if cfg.descriptor_mode == 1 else {}
        val_descriptors,   val_gradients,   val_grad_index   = builder.build_descriptors(val_dataset, **_kw)

    train_data = assemble_data_dict(train_dataset, train_types_int, train_descriptors, train_gradients, train_grad_index, cfg)
    val_data   = assemble_data_dict(val_dataset,   val_types_int,   val_descriptors,   val_gradients,   val_grad_index,   cfg)
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
    _kw = {"progress_desc": "Building test descriptors"} if cfg.descriptor_mode == 1 else {}
    test_descriptors, test_gradients, test_grad_index = builder.build_descriptors(
        test_dataset, **_kw)

    test_data = assemble_data_dict(
        test_dataset, test_types_int,
        test_descriptors, test_gradients, test_grad_index, cfg)
    _self_only = (cfg.target_mode == 1
                  and int(getattr(cfg, "dipole_rij_power", 2)) == 0)
    test_data = pad_and_stack(
        test_data, num_types=num_types, pin_to_cpu=pin_to_cpu,
        self_pairs_only=_self_only)
    # Pre-stage per-chunk pair indices to GPU. Test eval doesn't go
    # through `_evaluate_chunk` (TNEP.score uses model.predict_batch),
    # so XLA padding isn't needed for test data.
    S_test = int(test_data["num_atoms"].shape[0])
    chunk = cfg.batch_chunk_size if cfg.batch_chunk_size is not None else S_test
    test_ranges = [(s, min(s + chunk, S_test)) for s in range(0, S_test, chunk)]
    prestage_chunk_indices(test_data, test_ranges)
    test_pending["_built"] = test_data
    # Free the raw atom list now that descriptors are baked in.
    test_pending["dataset"] = None
    test_pending["types_int"] = None
    return test_data


def pad_and_stack(data: dict, num_types: int | None = None,
                  pin_to_cpu: bool = True,
                  self_pairs_only: bool = False) -> dict[str, tf.Tensor]:
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
    S = len(data["descriptors"])
    dim_q = data["descriptors"][0].shape[-1]
    atom_counts = [data["descriptors"][i].shape[0] for i in range(S)]
    max_atoms = max(atom_counts)

    target_sample = data["targets"][0]
    target_dim = 1 if target_sample.shape == () else target_sample.shape[0]
    has_forces = "forces" in data
    has_virials = "virials" in data

    # Count pairs per structure to build CSR struct_ptr and size COO arrays.
    if self_pairs_only:
        # Self-only: count how many centres have at least one row in
        # data["grad_index"][s][i] equal to i (the centre's own index).
        # Quippy/soap_turbo emits the zero-image self entry FIRST per
        # centre. We keep ONLY that first row; periodic self-images
        # (same atom index, nonzero displacement vector) are
        # intentionally dropped — including them would double-count the
        # self contribution under dipole_rij_power=0.
        # Defensive: an atom with no neighbours and no self entry
        # contributes zero pairs.
        pair_counts = [
            int(sum(1 for i in range(atom_counts[s])
                    if bool(np.any(np.asarray(data["grad_index"][s][i]) == i))))
            for s in range(S)
        ]
    else:
        pair_counts = [
            sum(data["gradients"][s][i].shape[0] for i in range(atom_counts[s]))
            for s in range(S)
        ]
    N_pairs_total = sum(pair_counts)
    struct_ptr_np = np.zeros(S + 1, dtype=np.int32)
    for s in range(S):
        struct_ptr_np[s + 1] = struct_ptr_np[s] + pair_counts[s]

    # COO arrays: one entry per real atom-neighbor pair.
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

        for i in range(N_s):
            gv_full = data["gradients"][s][i]
            gidx_full = np.asarray(data["grad_index"][s][i])
            if self_pairs_only:
                # Keep only the FIRST row where the neighbour index ==
                # centre index. soap_turbo/quippy emit the zero-image
                # self entry first per centre; any subsequent matches
                # are periodic IMAGES of atom i (same atom index,
                # nonzero displacement vector) and would double-count
                # the self contribution under dipole_rij_power=0, so
                # we intentionally drop them. Drops the neighbour
                # pairs entirely — grad_values_np shrinks from
                # O(N·M) per structure to O(N), and N=0 dipole
                # becomes a clean Σ_i de_dq[i] · grad_values[i, i].
                mask = (gidx_full == i)
                if not bool(np.any(mask)):
                    continue
                k0 = int(np.argmax(mask))  # first True index
                gv_arr = (gv_full.numpy()
                          if hasattr(gv_full, "numpy") else np.asarray(gv_full))
                grad_values_np[pair_offset] = gv_arr[k0]
                pair_struct_np[pair_offset] = s
                pair_atom_np[pair_offset]   = i
                pair_gidx_np[pair_offset]   = int(gidx_full[k0])
                pair_offset += 1
            else:
                n_nbrs = gv_full.shape[0]
                k_end  = pair_offset + n_nbrs
                grad_values_np[pair_offset:k_end] = (
                    gv_full.numpy() if hasattr(gv_full, "numpy")
                    else np.asarray(gv_full))
                pair_struct_np[pair_offset:k_end] = s
                pair_atom_np[pair_offset:k_end]   = i
                pair_gidx_np[pair_offset:k_end]   = gidx_full
                pair_offset += n_nbrs

    # Convert each numpy array to a TF tensor then immediately delete the numpy
    # copy so peak RAM stays at ~1x dataset size rather than ~2x.
    # When pin_to_cpu, always pin to CPU. Otherwise pin to GPU when one
    # exists; CPU-only nodes fall back to the implicit CPU placement.
    _dev_ctx = (tf.device('/CPU:0') if pin_to_cpu
                else _gpu_device_ctx())
    with _dev_ctx:
        result = {}
        result["descriptors"] = tf.constant(desc_np);    del desc_np
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


def _is_contiguous_range(arr: np.ndarray) -> bool:
    """True iff `arr` is a strictly monotonic +1 sequence (i.e. a true slice).

    Stronger than checking `arr[-1] - arr[0] + 1 == arr.size`, which falsely
    accepts any permutation whose first/last elements happen to bracket a
    contiguous range. A real contiguous slice has every diff == 1.
    """
    if arr.size == 0:
        return False
    if arr.size == 1:
        return True
    return bool(np.all(np.diff(arr) == 1))


def slice_and_complete_chunk(data: dict, indices) -> dict:
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

    The chunk's gradient pair slice is produced by tf.gather over the
    in-memory grad_values tensor.
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
    # indices [0..B_chunk) via value_rowids().
    ptr = data["struct_ptr"]
    pair_starts = tf.gather(ptr, idx_tf)
    pair_ends   = tf.gather(ptr, idx_tf + 1)
    pair_ranges = tf.ragged.range(pair_starts, pair_ends)
    flat_pair_idx_tf = tf.cast(pair_ranges.flat_values, tf.int32)
    chunk["pair_struct"] = tf.cast(pair_ranges.value_rowids(), tf.int32)

    gv = data["grad_values"]
    chunk["grad_values"] = tf.gather(gv, flat_pair_idx_tf)
    chunk["pair_atom"]   = tf.gather(data["pair_atom"],   flat_pair_idx_tf)
    chunk["pair_gidx"]   = tf.gather(data["pair_gidx"],   flat_pair_idx_tf)
    return chunk


# ============================================================================
# Chunk staging + prefetch + chunk-index caching
# ============================================================================

def prestage_chunk_indices(data: dict, ranges: list) -> None:
    """Pre-build GPU tensors for the per-chunk pair indices
    (pair_atom, pair_gidx, pair_struct) for each (s, e) in `ranges`.

    For deterministic full-batch chunks, these arrays are constant across
    generations — staging them once at startup eliminates ~3-5 ms/chunk of
    per-gen `tf.constant` + DMA work. Stored on the data dict under
    `_pair_idx_gpu_cache` and consumed by `_stage_finalize_tf` when present.

    Memory cost: ~3 × P_chunk × 4 bytes per chunk; tiny.
    """
    cache: dict = data.setdefault("_pair_idx_gpu_cache", {})
    pa_full = data["pair_atom"]
    pg_full = data["pair_gidx"]
    pa_np = pa_full.numpy() if hasattr(pa_full, "numpy") else np.asarray(pa_full)
    pg_np = pg_full.numpy() if hasattr(pg_full, "numpy") else np.asarray(pg_full)
    idx_cache = get_chunk_index_cache()
    with _gpu_device_ctx():
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
            # Materialise the int32 indices to numpy so the staging paths
            # can fancy-index the passthrough pair arrays without an extra
            # device sync per call. Small (< 100 K ints typically).
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
    """Numpy-only first phase of chunk staging (no TF ops).

    Packages the chunk's per-structure slices as small ndarrays and
    passes the in-RAM gradient/pair tensors through untouched.
    `_stage_finalize_tf` then turns this dict into TF tensors. Both
    phases run serially on the main thread; the split just keeps the
    plain-numpy slicing separate from the TF-constant conversion.
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

    # In-RAM: hand back the original tensors / arrays untouched.
    # Main thread will tf.gather them.
    out["_passthrough_grad_values"] = data["grad_values"]
    out["_passthrough_pair_atom"]   = data["pair_atom"]
    out["_passthrough_pair_gidx"]   = data["pair_gidx"]
    return out


def _stage_finalize_tf(data: dict, raw: dict, pin_to_cpu: bool,
                        s_start: int | None = None,
                        s_end: int | None = None) -> dict:
    """Main-thread half of staging: convert the worker's numpy output to
    TF tensors. Cheap (just tf.constant calls), runs in the foreground.

    When `data["_pair_idx_gpu_cache"]` has a pre-staged entry for the
    chunk's (s_start, s_end) range, the deterministic pair-index tensors
    (pair_atom, pair_gidx, pair_struct) are reused from the cache instead
    of being rebuilt each call."""
    chunk: dict = {}
    SMALL_KEYS = ("positions", "Z_int", "boxes", "num_atoms",
                  "targets", "atom_mask", "types_contained",
                  "forces", "virials")
    # pin_to_cpu=True ⇒ raw data is in host RAM; staging uploads each
    # chunk to GPU when one is present (no-op on CPU-only). False ⇒
    # raw data already lives on-device; no device context needed.
    ctx = _gpu_device_ctx() if pin_to_cpu else _NullCtx()
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
        # In-RAM passthrough. When the chunk's pair indices form a
        # contiguous range (the common case for sequential full-batch
        # chunking), use tf.strided_slice — pure GPU slice, no D2D gather
        # of the full chunk. Otherwise (random sub-sampling) fall back to
        # tf.gather.
        flat_pair_idx_np = raw["_precomputed"]["flat_pair_idx_np"]
        is_contig = _is_contiguous_range(flat_pair_idx_np)
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
            chunk["grad_values"] = tf.gather(raw["_passthrough_grad_values"], flat_pair_idx_tf)
            chunk["pair_atom"]   = tf.gather(raw["_passthrough_pair_atom"],   flat_pair_idx_tf)
            chunk["pair_gidx"]   = tf.gather(raw["_passthrough_pair_gidx"],   flat_pair_idx_tf)
    return chunk


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _gpu_device_ctx():
    """Return `tf.device('/GPU:0')` when a GPU is visible, else a no-op
    context. Used by chunk-staging code paths that historically used a
    bare `with tf.device('/GPU:0'):` block. On a CPU-only run (Mahti CPU
    partition) that bare device pin would otherwise raise — TF defaults
    to hard placement, so requesting `/GPU:0` with no GPU registered
    fails immediately. This helper lets the same code path execute
    unchanged on either node type."""
    try:
        if tf.config.list_physical_devices('GPU'):
            return tf.device('/GPU:0')
    except Exception:
        pass
    return _NullCtx()


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
    with _gpu_device_ctx():
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            if hasattr(v, "device"):
                # tf.Tensor: identity-on-GPU is cheap if already there.
                data[k] = tf.identity(v)
            else:
                # numpy → GPU tensor
                data[k] = tf.constant(np.asarray(v))


def _stage_chunk_resident(data: dict, s_start: int, s_end: int) -> dict:
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
    with _gpu_device_ctx():
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

        chunk["grad_values"] = grad_slc
        chunk["pair_atom"] = pair_atom
        chunk["pair_gidx"] = pair_gidx
        chunk["pair_struct"] = pair_struct
    return chunk


def prefetched_chunks(data: dict, ranges: list, pin_to_cpu: bool):
    """Yield (s_start, s_end, chunk) tuples for each (s, e) in `ranges`.

    Chunks are staged serially on the main thread; the name is kept for
    historical reasons (an earlier prefetch thread was removed). Two
    modes, picked by the data dict's state:

    1. **GPU-resident** (`_gv_resident_gpu=True`): chunks are built
       purely on-device via `_stage_chunk_resident`. No host↔GPU
       traffic. Fastest mode — used when the grad cache fits in VRAM.

    2. **In-RAM serial**: chunks are staged via `_stage_disk_only`
       (numpy passthrough) + `_stage_finalize_tf` (TF conversion),
       both on the main thread.
    """
    if not ranges:
        return

    if data.get("_gv_resident_gpu", False):
        for s, e in ranges:
            yield int(s), int(e), _stage_chunk_resident(data, int(s), int(e))
        return

    # Pre-warm the chunk-index cache from the main thread so the
    # staging path never triggers a TF op on cache miss.
    idx_cache = get_chunk_index_cache()
    for _s, _e in ranges:
        idx_cache.get(data, _s, _e)

    for s, e in ranges:
        raw = _stage_disk_only(data, s, e)
        yield s, e, _stage_finalize_tf(data, raw, pin_to_cpu=pin_to_cpu,
                                        s_start=s, s_end=e)


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
