"""Per-structure prediction-error losses for SNES training.

Provides `per_structure_error` (always MSE) and `squared_error_per_structure`
(also MSE, kept under a separate name so callers that need the *reporting*
metric — RMSE / RRMSE — can express that intent at the call site).
"""
from __future__ import annotations

import tensorflow as tf


def per_structure_error(
    diff: tf.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 0.0,                      # retained for signature compat
    component_weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """Compute the training-loss cost per structure (squared error).

    `loss_type` and `huber_delta` are accepted for backward signature
    compatibility but only `loss_type="mse"` is supported.

    Args:
        diff: prediction − target. Shape ``[..., T_dim]``.
        loss_type: must be ``"mse"`` (only supported value).
        huber_delta: unused.
        component_weights: optional ``[..., T_dim]`` tensor of
            multiplicative weights applied to each component's
            contribution before the per-structure reduction. Broadcasts
            over leading dims of ``diff``.

    Returns:
        per-structure error with shape ``diff.shape[:-1]``.
    """
    if loss_type != "mse":
        raise ValueError(
            f"loss_type={loss_type!r}: only 'mse' is supported in this build.")
    sq = tf.square(diff)
    if component_weights is not None:
        sq = sq * component_weights
    return tf.reduce_sum(sq, axis=-1)


def squared_error_per_structure(
    diff: tf.Tensor,
    component_weights: tf.Tensor | None = None,
) -> tf.Tensor:
    """MSE per-structure squared error (the reporting metric for RMSE/RRMSE)."""
    sq = tf.square(diff)
    if component_weights is not None:
        sq = sq * component_weights
    return tf.reduce_sum(sq, axis=-1)
