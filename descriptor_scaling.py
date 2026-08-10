"""Per-channel search preconditioning for the descriptor axis of W0.

SOAP-turbo L2-normalises each atom's descriptor vector, but individual
channels still differ in magnitude by ~1e4. Two consequences inside SNES:

  1. sigma is uniform per coordinate, so a channel whose W0 row must be
     300x larger to matter is explored at the same step width as every
     other — a random walk rather than a search.
  2. L1/L2 sum |W0| uniformly, so that channel pays a 300x larger penalty
     for an equal contribution to the output.

Both are fixed by rescaling the SEARCH, never the descriptor: q and dq/dR
are untouched, so the representable function class is unchanged and a
useless channel stays as cheap to ignore as it is today. Contrast the
removed 2026-05-13 descriptor scaling, which amplified numerical-dust
channels to unit magnitude inside the forward pass.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-30
VALID_MODES = ("off", "std", "rms", "cv")


def channel_statistic(stats: dict, mode: str) -> np.ndarray:
    """The per-channel scale s_k that the multiplier inverts."""
    if mode == "std":
        return np.asarray(stats["std"], dtype=np.float64)
    if mode == "rms":
        return np.asarray(stats["rms"], dtype=np.float64)
    if mode == "cv":
        # Available but MEASURED to invert the intended direction: dust
        # channels have large cv, hence a small multiplier. See the plan.
        std = np.asarray(stats["std"], dtype=np.float64)
        mean = np.abs(np.asarray(stats["mean"], dtype=np.float64))
        return std / np.maximum(mean, _EPS)
    raise ValueError(
        f"mode={mode!r} has no statistic (expected one of "
        f"{[m for m in VALID_MODES if m != 'off']}; 'off' is handled by the "
        f"caller and must never reach here)")


def channel_multipliers(stats: dict, mode: str, clamp: float,
                        exponent: float,
                        blocks: list | None = None) -> np.ndarray:
    """Per-channel multiplier, unit geometric mean, max/min ratio <= clamp.

    When `blocks` is given (a list of index arrays, one per (pair, l)), the
    statistic is averaged within each block and every channel in the block
    receives the same multiplier. That is required whenever descriptor
    mixing is on: U is block-diagonal over exactly these blocks, so a
    constant-per-block diag(m) commutes with U and the preconditioning is
    exact for all U rather than only at generation 0. Measured within-block
    std spread is a median 24.8x, so per-channel multipliers under mixing
    would drift materially.

    Unit geometric mean keeps cfg.init_sigma (and the Glorot scale under
    mode B) meaning what it meant before the option existed — the
    preconditioner redistributes step size rather than inflating it.

    Clipping to [clamp**-0.5, clamp**+0.5] bounds max/min by exactly clamp,
    and the renormalisation afterwards is a uniform rescale, so it cannot
    reintroduce a wider spread.

    Args:
        stats    : {"mean", "std", "rms"} arrays of length Q
        mode     : one of VALID_MODES, excluding "off"
        clamp    : max ratio between the largest and smallest multiplier
        exponent : 1.0 fully equalises, 0.0 is the identity
        blocks   : optional list of index arrays; one multiplier per block

    Returns:
        [Q] float32, finite and strictly positive
    """
    s = channel_statistic(stats, mode)
    if blocks is not None:
        s_block = np.empty_like(s)
        for idx in blocks:
            s_block[idx] = s[idx].mean()
        s = s_block
    m = (1.0 / np.maximum(s, _EPS)) ** float(exponent)
    m = m / np.exp(np.log(np.maximum(m, _EPS)).mean())      # geomean -> 1
    half = float(clamp) ** 0.5
    m = np.clip(m, 1.0 / half, half)
    m = m / np.exp(np.log(m).mean())                        # renormalise
    return m.astype(np.float32)


def expand_to_w0_coords(m: np.ndarray, num_types: int,
                        num_neurons: int) -> np.ndarray:
    """Lift a [Q] channel multiplier onto the [T*Q*H] flat W0 axis.

    W0 is stored [T, Q, H] row-major (TNEP.py:95-100, SNES.py:1426), so
    channel k owns H consecutive coordinates within each type block.
    """
    return np.tile(np.repeat(np.asarray(m, dtype=np.float32), num_neurons),
                   num_types)
