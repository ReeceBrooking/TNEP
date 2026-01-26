import tensorflow as tf
import numpy as np
from numpy.matlib import empty

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from ase.io import read
from ase import Atoms
import matplotlib.pyplot as plt
from quippy.descriptors import Descriptor
from quippy.convert import ase_to_quip

""" 
    1. Collect inputs (Z, R, config) and divide train/test data
    2. Pass into DescriptorBuilder and collect descriptors and dim q
    3. Init TNEP model with inputs
    4. Init SNES optimizer with model and inputs (within model)
    5. SNES model calls FitnessCalc and calculates parameter changes over n generations
    6. New model parameters are outputted and reconstructed from flat vector
    7. Validation step
"""

""" 
    R = [N, 3]
    Z = [N]
    box = [3, 3]
"""

def collect(cfg : TNEPconfig):
    # read R, Z and target values from file
    dataset = read(cfg.data_path, index=":")
    dataset_types_int = []
    types = []

# TODO needs to sort them so that they are in correct atomic order
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

def split(
    dataset,
    dataset_types_int,
    cfg
):
    """
    Split dataset into train and test sets while preserving indexing.

    Returns:
        train_data, test_data (both dicts)
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

    if cfg.target_mode == 0:
        target = "energy"
    if cfg.target_mode == 1:
        target = "dipole"
    elif cfg.target_mode == 2:
        target = "pol"

    def subset(input, descriptors, gradients, grad_index, types_int, target):
        return {
            "positions": [tf.convert_to_tensor(structure.positions, dtype = tf.float32) for structure in input],
            "Z": [structure.numbers for structure in input],
            "Z_int": [tf.convert_to_tensor(structure_types_int, dtype = tf.int32) for structure_types_int in types_int],
            "targets": [tf.convert_to_tensor(structure.info[target], dtype = tf.float32) for structure in input],
            "boxes": [tf.convert_to_tensor(structure.cell.array, dtype = tf.float32) for structure in input],  # shared
            "descriptors": descriptors,
            "gradients": gradients,
            "grad_index": grad_index,
        }

    train_data = subset(train_dataset, train_descriptors, train_gradients, train_grad_index, train_types_int, target)
    test_data = subset(test_dataset, test_descriptors, test_gradients, test_grad_index, test_types_int, target)
    val_data = subset(val_dataset, val_descriptors, val_gradients, val_grad_index, val_types_int, target)
#    lengths = [[len(train_data["Z_int"][i]), len(train_data["positions"][i])] for i in range(len(train_data))]
#    print(lengths)
    print(str(n_structures) + " structures have been split into sets of size:")
    print("Train set: " + str(len(train_data["positions"])) + " structures")
    print("Test set: " + str(len(test_data["positions"])) + " structures")
    print("Validation set: " + str(len(val_data["positions"])) + " structures")
    return train_data, test_data, val_data

def plot_snes_history(history, logy=False):
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

cfg = TNEPconfig()
# Read dataset from train.xyz
dataset, dataset_types_int = collect(cfg)
print("Number of species in dataset: " + str(cfg.num_types))
print("Number of structures in raw dataset: " + str(len(dataset)))
cfg.randomise(dataset)

# Split dataset into train, test, and validation sets
train_data, test_data, val_data = split(dataset, dataset_types_int, cfg)

cfg.dim_q = train_data["descriptors"][0].shape[-1]

model = TNEP(cfg)
history = model.fit(train_data, val_data)
print("Model test set RMSE: " + str(model.score(test_data)))
print("Run complete!")
plot_snes_history(history)


"""
    base = (
        "soap_turbo l_max=8 alpha_max={8 8 8 8 8 8} "
        "atom_sigma_r={0.5 0.5 0.5 0.5 0.5 0.5} atom_sigma_t={0.5 0.5 0.5 0.5 0.5 0.5} "
        "atom_sigma_r_scaling={0.0 0.0 0.0 0.0 0.0 0.0} atom_sigma_t_scaling={0.0 0.0 0.0 0.0 0.0 0.0} "
        "rcut_hard=3.7 rcut_soft=3.2 basis=poly3gauss scaling_mode=polynomial add_species=F "
        "amplitude_scaling={1.0 1.0 1.0 1.0 1.0 1.0} "
        "radial_enhancement=1 compress_mode=trivial central_weight={1. 1. 1. 1. 1. 1.} "
        "species_Z={6 8 16 1 7 17} n_species=6"
    )"""
