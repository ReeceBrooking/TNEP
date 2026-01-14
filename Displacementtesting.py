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
    return dr

@tf.function
def pairwise_displacements(R, box):
    """Return dr_ij [N,N,3], r_ij [N,N]."""
    Ri = tf.expand_dims(R, 1)   # [N,1,3]
    Rj = tf.expand_dims(R, 0)   # [1,N,3]
    # Vectorize MIC across pairs
    N = tf.shape(R)[0]
    Ri_t = tf.reshape(tf.tile(R, [N,1]), [N, N, 3])
    Rj_t = tf.transpose(Ri_t, perm=[1,0,2])
    dr_flat = _minimum_image_displacement(tf.reshape(Ri_t, [-1,3]),
                                          tf.reshape(Rj_t, [-1,3]),
                                          box)
    dr = tf.reshape(dr_flat, [N, N, 3])
    rij = tf.linalg.norm(dr + 1e-16, axis=-1)
    return dr, rij#, Ri, Rj, Ri_t, Rj_t, dr_flat

R = tf.convert_to_tensor([[9, 2, 2], [8, 2, 6], [3, 3, 2], [1, 2, 3], [4, 5, 6], [7, 8, 4]], dtype=tf.float32)
box = tf.convert_to_tensor([[10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=tf.float32)
rc = 5.0

print(pairwise_displacements(R, box))

#class CosineCutoff(layers.Layer):
#    def __init__(self, rc, **kwargs):
#        super().__init__(**kwargs)
#        self.rc = tf.constant(rc, tf.float32)
#    def call(self, r):
#        x = tf.clip_by_value(r / self.rc, 0.0, 1.0)
#        return 0.5 * (tf.cos(np.pi * x) + 1.0)# * tf.cast(r < self.rc, tf.float32)

def cut(r, rc):
    x = tf.clip_by_value(r / rc, 0.0, 1.0)
    return 0.5 * (tf.cos(np.pi * x) + 1.0)

dr, rij = pairwise_displacements(R, box)
print(cut(rij, rc))