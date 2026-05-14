"""Per-structure prediction-error losses for SNES training.

Provides a single helper `per_structure_error` that dispatches on the
loss type ("mse", "mae", "huber") and optionally applies per-component
multiplicative weights (used for the per-component inverse-weighting
mode and for polarisability shear weighting). The reduction is always
along the last axis (the T_dim component axis); leading dims (batch /
population) are preserved.

The training loss is kept separate from the reporting metric. Callers
that want exact RMSE for reporting should compute the squared-error
contribution separately via `squared_error_per_structure`, regardless
of which `loss_type` is being used for ranking — this keeps RMSE /
RRMSE comparable across loss-function ablations.
"""
from __future__ import annotations

import tensorflow as tf


def per_structure_error(
    diff: tf.Tensor,
    loss_type: str,
    huber_delta: float = 1e-3,
    component_weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """Compute the training-loss cost per structure.

    Args:
        diff: prediction − target. Shape ``[..., T_dim]``.
        loss_type: ``"mse" | "mae" | "huber"``.
        huber_delta: δ for Huber loss. Ignored otherwise.
        component_weights: optional ``[..., T_dim]`` tensor of
            multiplicative weights applied to each component's
            contribution before the per-structure reduction. Broadcasts
            over leading dims of ``diff``.

    Returns:
        per-structure error with shape ``diff.shape[:-1]``.
    """
    if loss_type == "mse":
        per_comp = tf.square(diff)
    elif loss_type == "mae":
        per_comp = tf.abs(diff)
    elif loss_type == "huber":
        if huber_delta <= 0:
            # δ <= 0 makes the linear branch monotonically decreasing
            # in |r|, which inverts the loss surface — silent disaster.
            raise ValueError(
                f"huber_delta must be > 0, got {huber_delta!r}")
        abs_r = tf.abs(diff)
        quad = 0.5 * tf.square(diff)
        lin = huber_delta * (abs_r - 0.5 * huber_delta)
        per_comp = tf.where(abs_r <= huber_delta, quad, lin)
    else:
        raise ValueError(
            f"loss_type={loss_type!r} not in ('mse', 'mae', 'huber')")
    if component_weights is not None:
        per_comp = per_comp * component_weights
    return tf.reduce_sum(per_comp, axis=-1)


def squared_error_per_structure(
    diff: tf.Tensor,
    component_weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """Always-MSE per-structure squared error, for RMSE / RRMSE reporting.

    Identical to ``per_structure_error(diff, "mse", ...)``. Exposed as a
    named function so call sites that need the reporting metric (rather
    than the training loss) make their intent explicit.
    """
    sq = tf.square(diff)
    if component_weights is not None:
        sq = sq * component_weights
    return tf.reduce_sum(sq, axis=-1)
