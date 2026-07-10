"""Per-structure prediction-error loss for SNES training.

Provides `squared_error_per_structure`, which computes the MSE-style
per-structure squared error and optionally applies per-component
multiplicative weights (used for the per-component inverse-weighting
mode and for polarisability shear weighting). The reduction is always
along the last axis (the T_dim component axis); leading dims (batch /
population) are preserved. The same function serves both the training
loss and the RMSE / RRMSE reporting metric.
"""
from __future__ import annotations

import tensorflow as tf


def squared_error_per_structure(
    diff: tf.Tensor,
    component_weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """Always-MSE per-structure squared error.

    Used both as the training loss and for RMSE / RRMSE reporting.

    Args:
        diff: prediction − target. Shape ``[..., T_dim]``.
        component_weights: optional ``[..., T_dim]`` tensor of
            multiplicative weights applied to each component's
            contribution before the per-structure reduction. Broadcasts
            over leading dims of ``diff``.

    Returns:
        per-structure squared error with shape ``diff.shape[:-1]``.
    """
    sq = tf.square(diff)
    if component_weights is not None:
        sq = sq * component_weights
    return tf.reduce_sum(sq, axis=-1)
