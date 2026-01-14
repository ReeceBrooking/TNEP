import tensorflow as tf
import numpy as np
from TNEP import TNEP
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from ase.io import read
import matplotlib.pyplot as plt

""" 
    1. Collect inputs (Z, R, config) and divide train/test data
    2. Pass into DescriptorBuilder and collect descriptors and dim q
    3. Init TNEP model with inputs
    4. Init SNES optimizer with model and inputs (within model)
    5. SNES model calls FitnessCalc and calculates parameter changes over n generations
    6. New model parameters are outputted and reconstructed from flat vector
    7. Validation step
"""

# inputs = read data file

#cfg = TNEPconfig()
#data = "read cfg.data_path"
""" 
    R = [N, 3]
    Z = [N]
    box = [3, 3]
"""
# Needs to loop through every structure in the dataset and append descriptors to a master array
#descriptors = DescriptorBuilder(cfg).build_descriptors(R, Z, box)
#cfg.dim_q = descriptors.shape[-1]
#cfg.num_types = Z.shape[0]
# Divide train and test data
""" 
    Option 1. Keep all info (Descriptors, Atom type) in one tensor/matrix
    Option 2. Keep descriptors and atoms types separate
"""
#model = TNEP(cfg)
#model = model.fit(train_descriptors, train_targets, cfg)
#test_values = model.predict(test_descriptors)
#model.score(test_values, test_targets)

def collect(file : str):
    # read R, Z and target values from file
    dataset = read(file, index=":")
    dataset_positions = []
    dataset_types = []
    dataset_targets = []
    box = dataset[0].cell.array
    for structure in dataset:
        dataset_positions.append(structure.positions)
        dataset_types.append(structure.numbers)
        dataset_targets.append(structure.info)
    dataset_types_int = dataset_types
    types = []
    for i in range(len(dataset_types)):
        for j in range(len(dataset_types[i])):
            z = dataset_types[i][j]
            if z not in types:
                types.append(z)
            dataset_types_int[i][j] = np.where(types == z)[0]
    num_types = len(types)
    return dataset_positions, dataset_types, dataset_targets, dataset_types_int, box, num_types

def split(
    positions,
    types,
    types_int,
    targets,
    box,
    cfg
):
    """
    Split dataset into train and test sets while preserving indexing.

    Returns:
        train_data, test_data (both dicts)
    """
    if cfg.total_N is not None:
        n_structures = cfg.total_N
    else:
        n_structures = len(positions)
    assert (
        len(types) == len(targets) == len(types_int)
    ), "Dataset lists must have same length"

    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(n_structures)
    rng.shuffle(indices)

    n_test = int(cfg.test_ratio * n_structures)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:2*n_test]
    train_idx = indices[2*n_test:]

    def subset(idxs):
        return {
            "R": [tf.convert_to_tensor(positions[i], dtype = tf.float32) for i in idxs],
            "Z": [types[i] for i in idxs],
            "Z_int": [tf.convert_to_tensor(types_int[i], dtype = tf.int32) for i in idxs],
            "targets": [targets[i] for i in idxs],
            "box": tf.convert_to_tensor(box, dtype = tf.float32),  # shared
            "descriptors": [None] * n_structures
        }

    train_data = subset(train_idx)
    test_data = subset(test_idx)
    val_data = subset(val_idx)

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

def specify_target(dataset, target_mode : int):
    if target_mode == 1:
        for i in range(len(dataset["targets"])):
            dataset["targets"][i] = tf.convert_to_tensor(dataset["targets"][i]["dipole"], dtype = tf.float32)
    elif target_mode == 2:
        for i in range(len(dataset["targets"])):
            dataset["targets"][i] = tf.convert_to_tensor(dataset["targets"][i]["pol"], dtype = tf.float32)
    return dataset

def train_split(dataset_positions, dataset_types, dataset_targets, dataset_types_int, box, cfg):
    train_data, test_data, val_data = split(dataset_positions, dataset_types, dataset_types_int, dataset_targets, box, cfg)
    train_data = specify_target(train_data, cfg.target_mode)
    test_data = specify_target(test_data, cfg.target_mode)
    val_data = specify_target(val_data, cfg.target_mode)
    return train_data, test_data, val_data

cfg = TNEPconfig()
dataset_positions, dataset_types, dataset_targets, dataset_types_int, box, num_types = collect(cfg.data_path)
train_data, test_data, val_data = train_split(dataset_positions, dataset_types, dataset_targets, dataset_types_int, box, cfg)
builder = DescriptorBuilder(cfg)
for i in range(len(train_data["R"])):
    train_data["descriptors"][i] = builder.build_descriptors(train_data["R"][i], train_data["box"])
    print([i + 1], " structure has been described for train set")
for j in range(len(test_data["R"])):
    test_data["descriptors"][j] = builder.build_descriptors(test_data["R"][j], test_data["box"])
    print([j + 1], " structure has been described for test set")

cfg.dim_q = train_data["descriptors"][0].shape[-1]
cfg.num_types = num_types
#print(num_types)
model = TNEP(cfg)
history = model.fit(train_data, val_data)
print(model.score(test_data))
plot_snes_history(history)



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