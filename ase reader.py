from ase.io import read
import tensorflow as tf

structures = read("train.xyz", index=":")
print(structures[0].cell.array)

dataset_positions = []
dataset_types = []
dataset_targets = []

for structure in structures:
    dataset_positions.append(structure.positions)
    dataset_types.append(structure.numbers)
    dataset_targets.append(structure.info)

dataset_positions_tf = tf.convert_to_tensor(dataset_positions[0])
#print(dataset_positions_tf.shape)
#print(dataset_targets[0]["pol"])