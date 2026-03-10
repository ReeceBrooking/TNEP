from __future__ import annotations

import tensorflow as tf
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
            structure_types_int[i] = np.where(types == z)[0]

        dataset_types_int.append(structure_types_int)

    cfg.num_types = len(types)
    cfg.types = types
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
    """Extract target, converting 9-component polarizability to 6-component if needed."""
    raw = np.asarray(structure.info[target_key], dtype=np.float32)
    if raw.size == 9:
        # Flattened 3x3 row-major -> unique [xx, yy, zz, xy, yz, zx]
        raw = raw[[0, 4, 8, 1, 5, 6]]
    return tf.convert_to_tensor(raw, dtype=tf.float32)


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
    r2_comp = metrics["r2_components"].numpy()
    labels = component_labels(cfg.target_mode, len(r2_comp))

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
    target_key = _target_key_for_mode(cfg.target_mode)
    return {
        "positions": [tf.convert_to_tensor(s.positions, dtype=tf.float32) for s in dataset],
        "Z_int": [tf.convert_to_tensor(t, dtype=tf.int32) for t in types_int],
        "targets": [_extract_target(s, target_key) for s in dataset],
        "boxes": [tf.convert_to_tensor(s.cell.array, dtype=tf.float32) for s in dataset],
        "descriptors": descriptors,
        "gradients": gradients,
        "grad_index": grad_index,
    }


def prepare_eval_data_raw(dataset: list[Atoms], cfg: TNEPconfig) -> dict:
    """Build type indices, descriptors, and raw (un-padded) data dict for evaluation.

    Like prepare_eval_data() but stops before pad_and_stack().

    Args:
        dataset : list of ase.Atoms
        cfg     : TNEPconfig from training

    Returns:
        raw data dict (un-padded) suitable for chunk_raw_data() then pad_and_stack()
    """
    types_int = assign_type_indices(dataset, cfg.types)
    builder = DescriptorBuilder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(dataset)
    return assemble_data_dict(dataset, types_int, descriptors, gradients, grad_index, cfg)


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

    n_test = int(cfg.test_ratio * n_structures)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:(2*n_test)]
    train_idx = indices[(2*n_test):n_structures]

    builder = DescriptorBuilder(cfg)

    test_dataset = [dataset[i] for i in test_idx]
    val_dataset = [dataset[i] for i in val_idx]
    train_dataset = [dataset[i] for i in train_idx]

    test_types_int = [dataset_types_int[i] for i in test_idx]
    val_types_int = [dataset_types_int[i] for i in val_idx]
    train_types_int = [dataset_types_int[i] for i in train_idx]

    train_descriptors, train_gradients, train_grad_index = builder.build_descriptors(train_dataset)
    val_descriptors, val_gradients, val_grad_index = builder.build_descriptors(val_dataset)
    test_descriptors, test_gradients, test_grad_index = builder.build_descriptors(test_dataset)

    """
    # Scale descriptors by inverse range (GPUMD q_scaler convention)
    q_scaler = DescriptorBuilder.compute_q_scaler(train_descriptors)
    train_descriptors, train_gradients = DescriptorBuilder.apply_scaling(
        train_descriptors, train_gradients, q_scaler)
    val_descriptors, val_gradients = DescriptorBuilder.apply_scaling(
        val_descriptors, val_gradients, q_scaler)
    test_descriptors, test_gradients = DescriptorBuilder.apply_scaling(
        test_descriptors, test_gradients, q_scaler)
    """

    train_data = assemble_data_dict(train_dataset, train_types_int, train_descriptors, train_gradients, train_grad_index, cfg)
    test_data = assemble_data_dict(test_dataset, test_types_int, test_descriptors, test_gradients, test_grad_index, cfg)
    val_data = assemble_data_dict(val_dataset, val_types_int, val_descriptors, val_gradients, val_grad_index, cfg)
    print(str(n_structures) + " structures have been split into sets of size:")
    print("Train set: " + str(len(train_data["positions"])) + " structures")
    print("Test set: " + str(len(test_data["positions"])) + " structures")
    print("Validation set: " + str(len(val_data["positions"])) + " structures")
    return train_data, test_data, val_data


def pad_and_stack(data: dict) -> dict[str, tf.Tensor]:
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

    for s in range(S):
        N_s = atom_counts[s]
        desc_np[s, :N_s, :] = data["descriptors"][s].numpy()
        pos_np[s, :N_s, :] = data["positions"][s].numpy()
        z_np[s, :N_s] = data["Z_int"][s].numpy()
        box_np[s] = data["boxes"][s].numpy()
        atom_mask_np[s, :N_s] = 1.0

        t = data["targets"][s]
        if t.shape == ():
            tgt_np[s, 0] = t.numpy()
        else:
            tgt_np[s, :] = t.numpy()

        for i in range(N_s):
            n_nbrs = data["gradients"][s][i].shape[0]
            grad_np[s, i, :n_nbrs, :, :] = data["gradients"][s][i].numpy()
            gidx_np[s, i, :n_nbrs] = data["grad_index"][s][i]
            nbr_mask_np[s, i, :n_nbrs] = 1.0

    return {
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


def estimate_padded_bytes(raw_data: dict, num_structures: int | None = None) -> int:
    """Estimate total bytes for padded tensors from raw (un-padded) data.

    Scans raw data for global max_atoms and max_neighbors, then estimates
    total bytes for num_structures (default: all). Applies 2x safety factor
    (numpy + tf.constant coexist briefly during pad_and_stack()).

    Args:
        raw_data       : dict from split() or assemble_data_dict() (un-padded)
        num_structures : int or None — how many structures to estimate for
                         (None = all structures in raw_data)

    Returns:
        estimated bytes as int
    """
    S = num_structures if num_structures is not None else len(raw_data["descriptors"])
    dim_q = raw_data["descriptors"][0].shape[-1]

    total_S = len(raw_data["descriptors"])
    atom_counts = [raw_data["descriptors"][i].shape[0] for i in range(total_S)]
    max_atoms = max(atom_counts)

    max_neighbors = 0
    for s in range(total_S):
        for i in range(atom_counts[s]):
            n_nbrs = raw_data["gradients"][s][i].shape[0]
            if n_nbrs > max_neighbors:
                max_neighbors = n_nbrs

    target_sample = raw_data["targets"][0]
    target_dim = 1 if target_sample.shape == () else target_sample.shape[0]

    A, M, Q, T = max_atoms, max_neighbors, dim_q, target_dim
    bytes_per_structure = (
        A * Q * 4              # descriptors [A, Q] float32
        + A * M * 3 * Q * 4   # gradients [A, M, 3, Q] float32  (dominant)
        + A * M * 4           # grad_index [A, M] int32
        + A * 3 * 4           # positions [A, 3] float32
        + A * 4               # Z_int [A] int32
        + T * 4               # targets [T] float32
        + 3 * 3 * 4           # boxes [3, 3] float32
        + A * 4               # atom_mask [A] float32
        + A * M * 4           # neighbor_mask [A, M] float32
        + 4                   # num_atoms int32
    )

    # 2x safety: numpy array + tf.constant coexist during pad_and_stack()
    return S * bytes_per_structure * 2


def get_available_memory(cfg: TNEPconfig) -> tuple[int, int]:
    """Return (ram_bytes, vram_bytes).

    If cfg.ram_mb or cfg.gpu_memory_mb is set, those values are used directly
    and auto-detection is skipped entirely. Otherwise:
    RAM: psutil (fallback 8 GB). VRAM: TF experimental (fallback inf).
    """
    if cfg.ram_mb is not None or cfg.gpu_memory_mb is not None:
        ram_bytes = cfg.ram_mb * 1024 * 1024 if cfg.ram_mb is not None else 8 * 1024 ** 3
        vram_bytes = cfg.gpu_memory_mb * 1024 * 1024 if cfg.gpu_memory_mb is not None else float('inf')
        return ram_bytes, vram_bytes

    # Auto-detect RAM
    try:
        import psutil
        ram_bytes = psutil.virtual_memory().available
    except ImportError:
        ram_bytes = 8 * 1024 ** 3

    # Auto-detect VRAM
    try:
        info = tf.config.experimental.get_memory_info('GPU:0')
        if 'limit' in info:
            vram_bytes = info['limit']
        else:
            vram_bytes = float('inf')
    except (RuntimeError, ValueError):
        vram_bytes = float('inf')

    return ram_bytes, vram_bytes


def compute_max_padded_size(raw_data: dict, cfg: TNEPconfig) -> int:
    """Compute max structures that fit in one padded chunk on VRAM.

    Chunk size is determined by VRAM alone (padded tf.constant tensors live on GPU).
    RAM holds the raw data permanently — this is checked separately via
    check_raw_data_ram().

    Args:
        raw_data : dict from split() — should be the combined dataset (all splits
                   concatenated) to get global max_atoms/max_neighbors
        cfg      : TNEPconfig with vram_threshold

    Returns:
        max_structures : int >= 1
    """
    per_struct = estimate_padded_bytes(raw_data, 1)
    _ram_bytes, vram_bytes = get_available_memory(cfg)

    if per_struct <= 0:
        return 1
    if vram_bytes == float('inf'):
        # No VRAM info — fall back to RAM budget as safeguard
        return max(int((_ram_bytes * cfg.ram_threshold) / per_struct), 1)

    return max(int((vram_bytes * cfg.vram_threshold) / per_struct), 1)


def check_raw_data_ram(raw_data: dict, cfg: TNEPconfig) -> None:
    """Warn if the raw (un-padded) dataset may not fit in available RAM.

    Raw data is kept in RAM for the entire run. This estimates its size
    and prints a warning if it exceeds the RAM budget.
    """
    ram_bytes, _vram_bytes = get_available_memory(cfg)
    ram_budget = ram_bytes * cfg.ram_threshold

    # Estimate raw data size: sum of actual tensor bytes (no padding overhead)
    total_bytes = 0
    S = len(raw_data["descriptors"])
    for s in range(S):
        N = raw_data["descriptors"][s].shape[0]
        Q = raw_data["descriptors"][s].shape[-1]
        total_bytes += N * Q * 4  # descriptors
        total_bytes += N * 3 * 4  # positions
        total_bytes += N * 4      # Z_int
        total_bytes += 3 * 3 * 4  # boxes
        # targets
        t = raw_data["targets"][s]
        total_bytes += (1 if t.shape == () else t.shape[0]) * 4
        # gradients + grad_index (variable per atom)
        for i in range(N):
            M_i = raw_data["gradients"][s][i].shape[0]
            total_bytes += M_i * 3 * Q * 4  # gradients
            total_bytes += M_i * 4           # grad_index

    if total_bytes > ram_budget:
        print(f"  WARNING: Raw dataset ~{total_bytes / 1024**3:.1f} GB exceeds "
              f"RAM budget {ram_budget / 1024**3:.1f} GB. "
              f"Consider reducing total_N or increasing ram_threshold.")


def chunk_raw_data(raw_data: dict, chunk_size: int) -> list[dict]:
    """Split raw (un-padded) data dict into sub-dicts of at most chunk_size structures.

    Each chunk has the same keys, with contiguous slices of the structure lists.

    Args:
        raw_data   : dict from split() with list-valued keys
        chunk_size : max structures per chunk

    Returns:
        list of raw data dicts
    """
    S = len(raw_data["descriptors"])
    chunks = []
    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        chunk = {}
        for key in raw_data:
            chunk[key] = raw_data[key][start:end]
        chunks.append(chunk)
    return chunks


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
    dipoles = np.array([s.info[target_key] for s in dataset])
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
    pols = []
    for s in dataset:
        raw = np.asarray(s.info[target_key], dtype=np.float32)
        if raw.size == 9:
            raw = raw[[0, 4, 8, 1, 5, 6]]
        pols.append(raw)
    pols = np.array(pols)
    labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
    print("=== Polarizability Target Statistics ===")
    print(f"  N structures: {len(pols)}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{pols[:,i].min():.4f}, {pols[:,i].max():.4f}]  "
              f"mean={pols[:,i].mean():.4f}  std={pols[:,i].std():.4f}")
