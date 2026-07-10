"""Per-structure prediction-error loss for SNES training."""
from __future__ import annotations

import tensorflow as tf


def squared_error_per_structure(
    diff: tf.Tensor,
    component_weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """Per-structure squared error Σ_k r_k² (training loss and RMSE/RRMSE metric).

    Args:
        diff: prediction − target, shape ``[..., T_dim]``.
        component_weights: optional ``[..., T_dim]`` per-component weights,
            applied before the reduction; broadcasts over leading dims.

    Returns:
        per-structure squared error with shape ``diff.shape[:-1]``.
    """
    sq = tf.square(diff)
    if component_weights is not None:
        sq = sq * component_weights
    return tf.reduce_sum(sq, axis=-1)
