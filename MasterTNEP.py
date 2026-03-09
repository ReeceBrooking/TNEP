import tensorflow as tf
import numpy as np

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from ase.io import read
import matplotlib.pyplot as plt

def collect(cfg : TNEPconfig):
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

    # TODO: type indices depend on encounter order — consider sorting by Z
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

def split(dataset, dataset_types_int, cfg):
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

    if cfg.target_mode == 0:
        target = "energy"
    elif cfg.target_mode == 1:
        target = "dipole"
    elif cfg.target_mode == 2:
        target = "pol"

    def _extract_target(structure, target_key):
        """Extract target, converting 9-component polarizability to 6-component if needed."""
        raw = np.asarray(structure.info[target_key], dtype=np.float32)
        if raw.size == 9:
            # Flattened 3x3 row-major -> unique [xx, yy, zz, xy, yz, zx]
            raw = raw[[0, 4, 8, 1, 5, 6]]
        return tf.convert_to_tensor(raw, dtype=tf.float32)

    def subset(input, descriptors, gradients, grad_index, types_int, target):
        return {
            "positions": [tf.convert_to_tensor(structure.positions, dtype = tf.float32) for structure in input],
            "Z": [structure.numbers for structure in input],
            "Z_int": [tf.convert_to_tensor(structure_types_int, dtype = tf.int32) for structure_types_int in types_int],
            "targets": [_extract_target(structure, target) for structure in input],
            "boxes": [tf.convert_to_tensor(structure.cell.array, dtype = tf.float32) for structure in input],
            "descriptors": descriptors,
            "gradients": gradients,
            "grad_index": grad_index,
        }

    train_data = subset(train_dataset, train_descriptors, train_gradients, train_grad_index, train_types_int, target)
    test_data = subset(test_dataset, test_descriptors, test_gradients, test_grad_index, test_types_int, target)
    val_data = subset(val_dataset, val_descriptors, val_gradients, val_grad_index, val_types_int, target)
    print(str(n_structures) + " structures have been split into sets of size:")
    print("Train set: " + str(len(train_data["positions"])) + " structures")
    print("Test set: " + str(len(test_data["positions"])) + " structures")
    print("Validation set: " + str(len(val_data["positions"])) + " structures")
    return train_data, test_data, val_data

def plot_snes_history(history, logy=False):
    """Plot train and validation RMSE vs generation."""
    g = np.asarray(history["generation"])

    plt.figure()
    plt.plot(g, history["train_loss"], label="Train RMSE")
    plt.plot(g, history["val_loss"], label="Val RMSE")

    plt.xlabel("generation")
    plt.ylabel("fitness (lower is better)")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.title("SNES fitness vs generation")
    plt.show()

def filter_by_species(dataset, dataset_types_int, allowed_Z):
    """Keep only structures whose atoms are all within allowed_Z.

    Args:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray — parallel to dataset
        allowed_Z         : list of int — allowed atomic numbers (e.g. [6, 1, 8])

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    allowed = set(allowed_Z)
    filtered_dataset = []
    filtered_types_int = []
    for struct, types_int in zip(dataset, dataset_types_int):
        if set(struct.numbers).issubset(allowed):
            filtered_dataset.append(struct)
            filtered_types_int.append(types_int)
    return filtered_dataset, filtered_types_int

def print_dipole_statistics(dataset, target_key="dipole"):
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

def print_polarizability_statistics(dataset, target_key="pol"):
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

cfg = TNEPconfig()

# Load raw dataset and assign initial type indices
dataset, dataset_types_int = collect(cfg)
print("Number of species in raw dataset: " + str(cfg.num_types))
print("Number of structures in raw dataset: " + str(len(dataset)))

# Filter to structures containing only C, H, O (Z = 6, 1, 8)
dataset, dataset_types_int = filter_by_species(dataset, dataset_types_int, allowed_Z=[6, 1, 8])
print("After C/H/O filter: " + str(len(dataset)) + " structures")

# Recompute type list and indices after filtering
cfg.num_types = 0
cfg.types = []
for struct in dataset:
    for z in struct.numbers:
        if z not in cfg.types:
            cfg.types.append(z)
cfg.num_types = len(cfg.types)

dataset_types_int = []
for struct in dataset:
    structure_types_int = np.zeros_like(struct.numbers)
    for i in range(len(struct.numbers)):
        z = struct.numbers[i]
        structure_types_int[i] = cfg.types.index(z)
    dataset_types_int.append(structure_types_int)
print("Species after filter: " + str(cfg.types) + " (" + str(cfg.num_types) + " types)")

if cfg.target_mode == 1:
    print_dipole_statistics(dataset)
elif cfg.target_mode == 2:
    print_polarizability_statistics(dataset)

cfg.randomise(dataset)

# Split into train/test/val and build SOAP descriptors (slow)
train_data, test_data, val_data = split(dataset, dataset_types_int, cfg)

# dim_q is determined by the SOAP descriptor size
cfg.dim_q = train_data["descriptors"][0].shape[-1]
print("Dimension of q: " + str(cfg.dim_q))

model = TNEP(cfg)
print("Model Parameters: " + str(model.optimizer.dim))
print("Population Size: " + str(model.optimizer.pop_size))
print("Parameter Natural Log: " + str(np.log(model.optimizer.dim)))
print("Parameter Root " + str(np.sqrt(model.optimizer.dim)))

history = model.fit(train_data, val_data)
print("Model test set RMSE: " + str(model.score(test_data)))
print("Run complete!")

plot_snes_history(history)
