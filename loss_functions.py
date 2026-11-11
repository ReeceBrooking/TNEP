"""Per-structure prediction-error loss for SNES training."""
from __future__ import annotations

import tensorflow as tf


def squared_error_per_structure(
    diff: tf.Tensor,
    component_weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """Per-structure squared error Σ_k r_k² → shape diff.shape[:-1].

    diff = prediction − target [..., T_dim]; optional component_weights
    [..., T_dim] are applied before the sum (training loss and RMSE/RRMSE metric).
    """
    sq = tf.square(diff)
    if component_weights is not None:
        sq = sq * component_weights
    return tf.reduce_sum(sq, axis=-1)
