import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Sequential, optimizers, losses

# Neighbour List Function
## Enforce PBC by choosing closest possible image
@tf.function
def _minimum_image_displacement(Ri, Rj, box):
    """Return Rj - Ri wrapped by MIC for triclinic/orthorhombic cell.
       Ri,Rj: [...,3], box: [3,3] (rows = lattice vectors)."""
    box_inv = tf.linalg.inv(box)
    si = tf.einsum('ij,bj->bi', box_inv, Ri)  # fractional
    sj = tf.einsum('ij,bj->bi', box_inv, Rj)
    ds = sj - si
    ds -= tf.round(ds)                        # wrap to [-0.5,0.5)
    dr = tf.einsum('ij,bj->bi', box, ds)      # back to Cartesian
    return dr, si, sj, ds

Ri = tf.cast([[1, 2, 3], [4, 5, 6], [7, 8, 4]], tf.float32)
Rj = tf.cast([[9, 2, 2], [8, 2, 6], [3, 3, 2]], tf.float32)
box = tf.cast([[10, 0, 0], [0, 10, 0], [0, 0, 10]], tf.float32)

print(_minimum_image_displacement(Ri, Rj, box))