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

def collect(file : str):
    # read R, Z and target values from file
    dataset = read(file, index=":")
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

    return dataset, dataset_types_int, types, len(types)

def split(
    dataset,
    descriptors,
    gradients,
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
    val_idx = indices[n_test:2*n_test]
    train_idx = indices[2*n_test:n_structures]

    test_desc = descriptors[:n_test]
    val_desc = descriptors[n_test:2*n_test]
    train_desc = descriptors[2*n_test:n_structures]

    test_grad = gradients[:n_test]
    val_grad = gradients[n_test:2*n_test]
    train_grad = gradients[2*n_test:n_structures]

    target = ""
    if cfg.target_mode == 1:
        target = "dipole"
    elif cfg.target_mode == 2:
        target = "pol"

    def subset(idxs, descriptors, gradients, target):
        return {
            "positions": [tf.convert_to_tensor(dataset[i].positions, dtype = tf.float32) for i in idxs],
            "Z": [dataset[i].numbers for i in idxs],
            "Z_int": [tf.convert_to_tensor(dataset_types_int[i], dtype = tf.int32) for i in idxs],
            "targets": [tf.convert_to_tensor(dataset[i].info[target], dtype = tf.float32) for i in idxs],
            "boxes": [tf.convert_to_tensor(dataset[i].cell.array, dtype = tf.float32) for i in idxs],  # shared
            "descriptors": [tf.convert_to_tensor(descriptor, dtype = tf.float32) for descriptor in descriptors],
            "gradients": [tf.convert_to_tensor(gradient, dtype = tf.float32) for gradient in gradients]
        }

    train_data = subset(train_idx, train_desc, train_grad, target)
    test_data = subset(test_idx, test_desc, test_grad, target)
    val_data = subset(val_idx, val_desc, val_grad, target)

    return train_data, test_data, val_data

def plot_snes_history(history, logy=True):
    g = np.asarray(history["gen"])

    plt.figure()
    plt.plot(g, history["train_mean"], label="Train RMSE")
    plt.plot(g, history["val_mean"], label="Val RMSE")

    plt.xlabel("generation")
    plt.ylabel("fitness (lower is better)")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.title("SNES fitness vs generation")
    plt.show()

def build_descriptors(dataset, types, num_types, cfg):
    base = (
        "soap_turbo l_max=8 "
        "rcut_hard=3.7 rcut_soft=3.2 basis=poly3gauss scaling_mode=polynomial add_species=F "
        "radial_enhancement=1 compress_mode=trivial "
    )

    alpha_max = " alpha_max={"
    atom_sigma_r = " atom_sigma_r={"
    atom_sigma_t = " atom_sigma_t={"
    atom_sigma_r_scaling = " atom_sigma_r_scaling={"
    atom_sigma_t_scaling = " atom_sigma_t_scaling={"
    amplitude_scaling = " amplitude_scaling={"
    central_weight = " central_weight={"

    for a in range(num_types):
        alpha_max += "8 "
        atom_sigma_r += "0.5 "
        atom_sigma_t += "0.5 "
        atom_sigma_r_scaling += "0.0 "
        atom_sigma_t_scaling += "0.0 "
        amplitude_scaling += "1.0 "
        central_weight += "1. "

    alpha_max += "}"
    atom_sigma_r += "}"
    atom_sigma_t += "}"
    atom_sigma_r_scaling += "}"
    atom_sigma_t_scaling += "}"
    amplitude_scaling += "}"
    central_weight += "}"

    n_species = " n_species=" + str(num_types)

    species_Z = " species_Z={"
    for type in types:
        species_Z += str(type) + " "
    species_Z += "}"

    base += species_Z + n_species + alpha_max + atom_sigma_r + atom_sigma_t + atom_sigma_r_scaling + atom_sigma_t_scaling + amplitude_scaling + central_weight

    print(base)

    builders = [Descriptor(base + f" central_index={k}") for k in (np.arange(num_types, dtype=int) + 1)]
    dataset_descriptors = []
    dataset_gradients = []

    for i in cfg.indices:
        outs = [b.calc(dataset[i], grad=True) for b in builders]

        descriptors = []
        center_index = []
        gradients = []
        grad_indexes = []

        for out in outs:
            #print(out)
            data = out.get("data")
            if data is None or data.size == 0 or data.shape[1] == 0:
                continue
            #print(out["ci"])
            for data in out["data"]:
                descriptors.append(data)
            for index in out["ci"]:
                center_index.append(index)
            assert len(center_index) == len(descriptors)
            for j in range(len(out["grad_index_0based"])):
                if out["grad_index_0based"][j][0] not in grad_indexes:
                    grad_indexes.append(out["grad_index_0based"][j][0])
                    gradients.append(out["grad_data"][j])
                else:
                    gradients[-1] += out["grad_data"][j]

        descriptors_sorted = np.zeros_like(descriptors)
        gradients_sorted = np.zeros_like(gradients)
        for k in range(len(center_index)):
            descriptors_sorted[center_index[k] - 1] = descriptors[k]
        for z in range(len(grad_indexes)):
            gradients_sorted[grad_indexes[z]] = gradients[z]
        dataset_descriptors.append(descriptors_sorted)
        dataset_gradients.append(gradients_sorted)
    return dataset_descriptors, dataset_gradients

#def specify_target(dataset, target_mode : int):
#   if target_mode == 1:
#        for i in range(len(dataset["targets"])):
#            dataset["targets"][i] = tf.convert_to_tensor(dataset["targets"][i]["dipole"], dtype = tf.float32)
#    elif target_mode == 2:
#        for i in range(len(dataset["targets"])):
#            dataset["targets"][i] = tf.convert_to_tensor(dataset["targets"][i]["pol"], dtype = tf.float32)
#    return dataset

cfg = TNEPconfig()
# Read dataset from train.xyz
dataset, dataset_types_int, types, num_types = collect(cfg.data_path)
cfg.randomise(dataset)
#print(dataset[0].numbers, dataset[0].positions, dataset[0].symbols, dataset[0].info)
# Split dataset into train, test, and validation sets
descriptors, gradients = build_descriptors(dataset, types, num_types, cfg)
train_data, test_data, val_data = split(dataset, descriptors, gradients, dataset_types_int, cfg)
#print(len(train_data["Z"][0]), tf.shape(train_data["descriptors"][0]))
#for i in range(len(train_data["R"])):
#    assert len(train_data["Z"][i]) == tf.shape(train_data["descriptors"][i])[0]
#    print("Number of atoms and descriptors in structure " + str(i) + " have been verified")

cfg.dim_q = train_data["descriptors"][0].shape[-1]
cfg.num_types = num_types

model = TNEP(cfg)
history = model.fit(train_data, val_data)
#print(model.score(test_data))
#plot_snes_history(history)
print("Run complete!")
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
""" TESTING 
13
Lattice="100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0" Properties=species:S:1:pos:R:3 dipole="0.162 0.0276 0.0046" pol="100.229691 3.510598 0.085722 3.510598 93.638814 -0.07923 0.085722 -0.07923 56.957231" pbc="T T T"
C       47.60422226      49.50684576      49.99673171
C       49.10610401      49.54948269      49.99979367
C       49.84597752      50.71065284      49.99805309
O       49.31389644      51.97280217      49.99249935
S       51.57359926      50.41035399      50.00158503
C       51.30182690      48.68910323      50.00613259
C       49.96279409      48.40102279      50.00436524
H       47.18186935      50.51907360      50.00749184
H       47.21919638      48.97000312      50.87720929
H       47.22173428      48.99045553      49.10288397
H       50.02767742      52.63278090      49.99968036
H       52.15408054      48.01737124      50.01019019
H       49.58113384      47.37959045      50.00673140
"""
"""
Z = [0, 0, 0, 1, 2, 0, 0, 3, 3, 3, 3, 3, 3]

Z = tf.convert_to_tensor(Z)

R = [
    [47.60422226, 49.50684576, 49.99673171],
    [49.10610401, 49.54948269, 49.99979367],
    [49.84597752, 50.71065284, 49.99805309],
    [49.31389644, 51.97280217, 49.99249935],
    [51.57359926, 50.41035399, 50.00158503],
    [51.30182690, 48.68910323, 50.00613259],
    [49.96279409, 48.40102279, 50.00436524],
    [47.18186935, 50.51907360, 50.00749184],
    [47.21919638, 48.97000312, 50.87720929],
    [47.22173428, 48.99045553, 49.10288397],
    [50.02767742, 52.63278090, 49.99968036],
    [52.15408054, 48.01737124, 50.01019019],
    [49.58113384, 47.37959045, 50.00673140]
]

R = tf.convert_to_tensor(R)
#print(R.shape)

box = [
    [100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]
]

box = tf.convert_to_tensor(box)

cfg = TNEPconfig()
descriptors = DescriptorBuilder(cfg).build_descriptors(R, box)
descriptors = tf.expand_dims(descriptors, axis=0) # imitate full dataset dimensions
print("descriptors=", descriptors)
cfg.dim_q = descriptors.shape[-1]
cfg.num_types = 4

model = TNEP(cfg)
predictions = model.predict(descriptors[0], Z)
print("before fitting = ", predictions)

targets = [[-10.0]]
targets = tf.convert_to_tensor(targets)
model = model.fit(descriptors, targets, Z)
predictions = model.predict(descriptors[0], Z)
print("after fitting = ", predictions)
"""