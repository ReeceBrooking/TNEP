# Descriptor Search Preconditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let SNES reach the weights of low-magnitude descriptor channels, by rescaling the *search geometry* rather than the descriptor — as two independent, opt-in modes.

**Architecture:** Per-channel statistics are collected once over the training split during descriptor staging and frozen on `cfg._descriptor_channel_stats`. Mode A multiplies the per-coordinate SNES `sigma` for the W0 block by `m_k ∝ (1/std_k)^α`. Mode B reparameterises the search space: `μ` holds `Ŵ0`, and `reconstruct_params_tf` emits the effective `W0 = Ŵ0 · m_k`, which makes both the search step *and* the L1/L2 penalty scale-fair. Neither mode touches `q` or `∂q/∂R`.

**Tech Stack:** Python 3.10, TensorFlow 2.20, h5py, numpy, pytest 9. No new dependencies.

---

## Why this is NOT the 2026-05-13 descriptor-scaling attempt

Read [2026-05-13-descriptor-scaling-option-A.md](2026-05-13-descriptor-scaling-option-A.md) first. That plan multiplied **the descriptor and its gradients** by `1/(max − min)`, GPUMD-style. It was implemented, it was empirically bad, and it was removed (the vestige is `descriptor_mean` in `model_io.py:487`, now a no-op).

It handled the gradient chain correctly — that was not the bug. The likely cause is the statistic. Measured on the **current** config (`l_max=4`, `α_max=7`, Q=135, `rcut_hard=4.0`, 250 structures / 24,000 atoms of `train_waterbulk.xyz`):

- per-channel std spans **13,420×**, from 1.6e-05 to 2.1e-01
- **65 of 135 channels** have std below 1% of the maximum

And on the `l_max=7, α_max=7, Q=216` config where the per-channel detail was collected, the smallest channels are numerical dust — e.g. channel 159 has mean −1.5e-06 and std 8.6e-06.

`1/(max − min)` is a *range* normalisation, maximally outlier-sensitive, and it amplifies those dust channels to unit scale. After scaling they enter `z = q·W0` — and, worse, enter the dipole contraction through the equally-amplified `∂q/∂R` — at the same magnitude as real signal. The model must then spend capacity suppressing ~65 channels of amplified noise. That is a plausible mechanism for "massively antagonistic".

**Neither mode here can do that**, for a structural reason rather than a tuning one:

| | 2026-05-13 (descriptor scaling) | This plan (search preconditioning) |
|---|---|---|
| What is multiplied | `q` and `∂q/∂R` | `sigma`, or `μ`'s coordinates |
| Noise channel after scaling | permanently unit-magnitude in the forward pass | still tiny in the forward pass |
| Cost of a useless channel | model must actively suppress it | `Ŵ0_k = 0` and it is gone, as today |
| Model class | changed | **identical** — pure reparameterisation |
| Failure mode | wrong model | wasted sampling budget, nothing more |

Consequences baked into Task 1.2: use **std**, not range; **clamp** the ratio; and **soften** with an exponent (see below).

## On `std/mean` (the coefficient of variation)

The request specified scaling inversely to `std/mean`. Implemented literally, `m_k = 1/CV_k = |mean_k|/std_k`, this **scales the wrong way**. Measured mean multiplier per `l` on the real descriptor set, `clamp` disabled:

| `l` | mean multiplier, `"std"` | mean multiplier, `"cv"` |
|---|---|---|
| 0 | 0.049 | **2.43** |
| 1 | 2.73 | 1.24 |
| 2 | 3.48 | 1.29 |
| 3 | 6.69 | 0.88 |
| 4 | 11.51 | **0.91** |

`cv` gives the *largest* boost to `l=0` — which already carries ~78% of the model's ablation importance — and suppresses `l≥3`, the block we are trying to reach. Several channels also have near-zero mean, so `CV → ∞` and the multiplier collapses to ~0, freezing them.

The mean is absorbed by `b0` and contributes nothing to how much a channel can move the output; **std is the correct statistic** and is the default. `"cv"` is implemented so the claim is falsifiable, `"rms"` as a zero-mean-safe variant. `test_cv_mode_penalises_the_channels_std_mode_boosts` pins the direction.

## Why the exponent defaults to 0.5, not 1.0

The `"std"` column above is the reason. Full equalisation (`exponent=1.0`) cuts `l=0`'s sigma by **20×**, because geometric-mean normalisation means boosting small channels necessarily suppresses large ones. `l=0` carries ~78% of the model's accuracy by ablation, so slowing its search 20-fold against a speculative gain on `l≥3` is a bad trade. At `exponent=0.5` the log-ratio halves: `l=0 → 0.22`, `l=4 → 3.39`. Most of the boost, a quarter of the damage.

## `descriptor_mixing` forces block granularity — measured, not assumed

`TNEP._W0_eff` (`TNEP.py:493-498`) computes `z = q·Uᵀ·W0`, so **`W0`'s Q axis indexes the *mixed* descriptor**, not the raw one. A multiplier measured on raw channels is exact at generation 0 (`V = 0 ⇒ U = I`) and drifts as `U` rotates.

An earlier draft of this plan argued the drift was second-order, because `U`'s blocks are confined within one `(pair, l)` and the multiplier varies mostly *between* `l`. **That argument is wrong.** Measured on the real descriptor set (150 structures, `l_max=4`, `α_max=7`):

```
within-(pair,l)-block std ratio : median 24.8x , max 207.5x  (15 blocks)
between-l std ratio             : 146.2x
```

The within-block spread is the *same order* as the between-`l` spread, so `U` can redistribute magnitude across channels whose multipliers differ by up to 200×. A frozen per-channel multiplier would be substantially invalidated as training proceeds.

**The fix makes it exact rather than approximate.** If every channel in a `(pair, l)` block shares one multiplier, then `M = diag(m)` restricted to that block is `m_block·I`, and since `U` is block-diagonal over the *same* blocks, `M·U = U·M` — the scaling **commutes exactly** with the mixing. Raw-space and mixed-space preconditioning become identical, for all `U`, forever.

So the granularity is chosen automatically, with no new config field:

| `descriptor_mixing` | granularity | rationale |
|---|---|---|
| `True` (default) | per `(pair, l)` block | commutes with `U` exactly |
| `False` | per channel | no `U`, so per-channel is exact and strictly finer |

Block granularity gives up the within-block 24.8× median spread. That is the correct trade: the motivation for this work is that `l≥3` channels are unreachable, and the between-`l` structure (146×) is fully preserved. Task 6.4 re-measures both ratios after the fact so the choice stays evidence-backed.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `descriptor_scaling.py` | **Create** | Pure functions: statistics → clamped, normalised multiplier; expand to the SNES W0 coordinate axis. No TF, no I/O. |
| `conftest.py` | **Create** | Repo root on `sys.path` so `tests/` can import the top-level modules. |
| `TNEPconfig.py` | Modify | Four new fields. |
| `data.py` | Modify | Accumulate per-channel `count/sum/sumsq` across batches in `build_and_reduce`; attach to the returned dict. |
| `SNES.py` | Modify | Resolve modes + guards; mode A sigma scaling; mode B `_build_mu_init` / `reconstruct_params_tf`; add `descriptor_scale` to the checkpoint state dict. |
| `model_io.py` | Modify | Persist in `/snes/descriptor_scale` (**required** for mode-B resume) and `/descriptor/channel_scale` (provenance). Legacy defaults. |
| `scripts/smoke_modes.py` | **Create** | Runnable end-to-end check across target modes and options (Phase 5 needs a real command). |
| `tests/test_descriptor_scaling.py` | **Create** | Multiplier maths, reparameterisation identity, guards, round-trips. |

**Guards — implemented once, in `SNES.__init__`, applying to BOTH modes:**
- both modes non-`"off"` → multiplier applied twice
- mode string not in `VALID_MODES` → typo caught at construction
- `descriptor_preprocess_contract != "off"` → `W0` is at `Q_new`, statistics at `Q_raw`
- `target_mode == 0` → the statistics hook lives in `build_and_reduce`, which PES does not use
- statistics missing → clear message rather than `AttributeError`

---

## Phase 0 — Make the tests importable

### Task 0.1: Root conftest

**Files:** Create `conftest.py`; delete stale `tests/__pycache__/test_descriptor_scaling.*.pyc`

`tests/` currently holds only `__pycache__` and `fixtures` — there is no `conftest.py`, `pytest.ini`, `pyproject.toml`, or `tests/__init__.py`. Under pytest's default prepend import mode, `tests/` goes on `sys.path` but the repo root does not, so `from data import ...` fails.

- [ ] **Step 1: Create `conftest.py` at the repo root**

```python
"""Put the repo root on sys.path so tests can import the top-level modules.

The project has no package layout — data.py, SNES.py, TNEP.py etc. live at
the root — so pytest's prepend import mode would otherwise only see tests/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

- [ ] **Step 2: Remove the stale bytecode from the 2026-05-13 attempt** (housekeeping only — Python will not import it without a sibling `.py`)

```bash
rm -f tests/__pycache__/test_descriptor_scaling.*.pyc
```

- [ ] **Step 3: Verify collection works**

Run: `python -m pytest tests/ -q --collect-only`
Expected: exits 0 (collects nothing yet, but no import errors)

- [ ] **Step 4: Commit**

```bash
git add conftest.py && git commit -m "test: repo root on sys.path for pytest"
```

---

## Phase 1 — The multiplier, in isolation

### Task 1.1: Config fields

**Files:** Modify `TNEPconfig.py` (after the descriptor-preprocess block, ~line 172)

- [ ] **Step 1: Add the fields**

```python
    # ── Search preconditioning ─────────────────────────────────────────
    # SOAP-turbo L2-normalises each atom's descriptor, but per-CHANNEL
    # magnitudes still span ~1e4 (measured: std 1.6e-05 .. 2.1e-01 at
    # l_max=4, alpha_max=7). A channel needing a 300x larger W0 row to
    # matter is explored by SNES at the same sigma as every other, and
    # pays a 300x larger L1 penalty for the same effect. These options fix
    # that WITHOUT touching the descriptor or its gradients — see
    # docs/superpowers/plans/2026-08-06-descriptor-search-preconditioning.md
    # for why descriptor-side scaling (removed, 2026-05-13) is different.
    #
    # Statistic behind the per-channel multiplier m_k ∝ (1/s_k)**exponent:
    #   "off" : no preconditioning (default)
    #   "std" : s_k = std(q_k)                 — recommended
    #   "rms" : s_k = sqrt(mean(q_k^2))        — zero-mean-safe variant
    #   "cv"  : s_k = std(q_k)/|mean(q_k)|     — MEASURED to scale the
    #           OPPOSITE way (boosts l=0, suppresses l>=3); kept only so
    #           the claim stays falsifiable.
    #
    # A: scale SNES's per-coordinate sigma on the W0 block only. Model and
    #    mu semantics unchanged; nothing extra to persist.
    descriptor_sigma_scaling: str = "off"
    # B: reparameterise the search space — mu holds W0_hat, effective
    #    W0 = W0_hat * m_k. Subsumes A AND makes the existing uniform L1/L2
    #    penalty scale-fair. Requires the multiplier in the checkpoint.
    #    Mutually exclusive with A.
    descriptor_weight_reparam: str = "off"
    # Softening. 1.0 fully equalises the perturbation each channel causes
    # in z; 0.0 is a no-op. MEASURED mean multiplier per l:
    #        exponent=1.0            exponent=0.5
    #   l=0    0.049 (sigma /20)       0.22
    #   l=4   11.51                    3.39
    # Full equalisation cuts l=0's sigma 20-fold, and l=0 carries ~78% of
    # the model's ablation importance — a bad trade against a speculative
    # gain on l>=3. 0.5 keeps most of the boost at a quarter of the damage.
    descriptor_scaling_exponent: float = 0.5
    # Max ratio between the largest and smallest multiplier. Enforced by
    # clipping to [clamp**-0.5, clamp**+0.5] about the geometric mean;
    # the later renormalisation is a uniform rescale and so preserves the
    # ratio. The raw unclamped spread at exponent=1.0 is ~13,420x.
    descriptor_scaling_clamp: float = 64.0
```

- [ ] **Step 2: Commit**

```bash
git add TNEPconfig.py && git commit -m "cfg: descriptor search-preconditioning options"
```

### Task 1.2: The multiplier module

**Files:** Create `descriptor_scaling.py`; Create `tests/test_descriptor_scaling.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_descriptor_scaling.py
import numpy as np
import pytest

from descriptor_scaling import (VALID_MODES, channel_multipliers,
                                expand_to_w0_coords)


def _stats(mean, std, rms=None):
    mean = np.asarray(mean, float)
    std = np.asarray(std, float)
    return {"mean": mean, "std": std,
            "rms": np.asarray(rms if rms is not None else std, float)}


def test_std_mode_boosts_small_channels():
    m = channel_multipliers(_stats([1.0, 1.0], [1.0, 0.01]), "std",
                            clamp=1e9, exponent=1.0)
    assert m[1] / m[0] == pytest.approx(100.0, rel=1e-5)


def test_exponent_softens_in_log_space():
    s = _stats([1.0, 1.0], [1.0, 0.01])
    half = channel_multipliers(s, "std", clamp=1e9, exponent=0.5)
    assert half[1] / half[0] == pytest.approx(10.0, rel=1e-5)
    off = channel_multipliers(s, "std", clamp=1e9, exponent=0.0)
    np.testing.assert_allclose(off, np.ones(2), rtol=1e-6)


def test_multipliers_have_unit_geometric_mean():
    rng = np.random.default_rng(0)
    s = rng.uniform(1e-4, 1e-1, size=64)
    m = channel_multipliers(_stats(np.ones(64), s), "std",
                            clamp=1e9, exponent=1.0)
    assert np.exp(np.log(m).mean()) == pytest.approx(1.0, rel=1e-6)


def test_clamp_bounds_max_over_min_ratio():
    """clamp is the max ratio between largest and smallest multiplier."""
    m = channel_multipliers(_stats(np.ones(3), [1.0, 1e-6, 1e3]), "std",
                            clamp=10.0, exponent=1.0)
    assert m.max() / m.min() <= 10.0 + 1e-6
    assert np.exp(np.log(m).mean()) == pytest.approx(1.0, rel=1e-6)


def test_cv_mode_penalises_the_channels_std_mode_boosts():
    """Real measured values: l=0-like (large) vs l=4-like (small).
    "std" boosts the small channel; "cv" boosts the large one instead."""
    s = _stats(mean=[0.328, 7.2e-4], std=[0.257, 7.2e-4], rms=[0.417, 1.0e-3])
    m_std = channel_multipliers(s, "std", clamp=1e9, exponent=1.0)
    m_cv = channel_multipliers(s, "cv", clamp=1e9, exponent=1.0)
    assert m_std[1] > m_std[0]
    assert m_cv[1] < m_cv[0]


def test_zero_std_channel_stays_finite():
    m = channel_multipliers(_stats(np.ones(2), [1.0, 0.0]), "std",
                            clamp=8.0, exponent=1.0)
    assert np.all(np.isfinite(m)) and np.all(m > 0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="no statistic"):
        channel_multipliers(_stats([1.0], [1.0]), "zscore", 8.0, 1.0)


def test_block_granularity_gives_one_multiplier_per_block():
    """Required under descriptor_mixing: a constant-per-block diag(m)
    commutes with the block-diagonal U, making the preconditioning exact
    for all U instead of only at generation 0."""
    s = _stats(np.ones(4), [1.0, 4.0, 0.01, 0.04])
    blocks = [np.array([0, 1]), np.array([2, 3])]
    m = channel_multipliers(s, "std", clamp=1e9, exponent=1.0, blocks=blocks)
    assert m[0] == pytest.approx(m[1])          # same block -> same multiplier
    assert m[2] == pytest.approx(m[3])
    assert m[2] > m[0]                           # small-std block still boosted


def test_expand_to_w0_coords_repeats_per_channel_over_H():
    """W0 is [T, Q, H] row-major, so channel k owns H consecutive
    coordinates within each of the T type blocks."""
    out = expand_to_w0_coords(np.array([2.0, 3.0]), num_types=2, num_neurons=3)
    np.testing.assert_array_equal(out, [2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_descriptor_scaling.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'descriptor_scaling'`

- [ ] **Step 3: Write the implementation**

```python
# descriptor_scaling.py
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_descriptor_scaling.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add descriptor_scaling.py tests/test_descriptor_scaling.py
git commit -m "descriptor_scaling: per-channel multiplier from training statistics"
```

---

## Phase 2 — Collect the statistics during staging

### Task 2.1: Incremental per-channel accumulator

**Files:** Modify `data.py` (add above `build_and_reduce`, which spans `534-621`)

The gradients are discarded batch by batch, so statistics must accumulate incrementally — there is no point at which the whole descriptor set exists.

- [ ] **Step 1: Write the failing test**

```python
def test_channel_stats_match_a_single_pass():
    """Batched accumulation equals a one-shot numpy computation."""
    from data import _accumulate_channel_stats, _finalize_channel_stats
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(7, 5)), rng.normal(size=(11, 5)),
              rng.normal(size=(3, 5))]
    acc = None
    for b in blocks:
        acc = _accumulate_channel_stats(acc, b)
    got = _finalize_channel_stats(acc)
    ref = np.concatenate(blocks, axis=0)
    np.testing.assert_allclose(got["mean"], ref.mean(0), rtol=1e-6)
    np.testing.assert_allclose(got["std"], ref.std(0), rtol=1e-6)
    np.testing.assert_allclose(got["rms"], np.sqrt((ref ** 2).mean(0)), rtol=1e-6)
    assert got["count"] == 21
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_descriptor_scaling.py::test_channel_stats_match_a_single_pass -q`
Expected: FAIL — `ImportError: cannot import name '_accumulate_channel_stats'`

- [ ] **Step 3: Implement**

```python
def _accumulate_channel_stats(acc: dict | None, desc: np.ndarray) -> dict:
    """Fold one [N, Q] block of per-atom descriptors into running sums.

    float64 accumulation: sums run over ~1e4 atoms while channels span 1e4
    in magnitude, so float32 would lose exactly the small channels this
    exists to measure.
    """
    d = np.asarray(desc, dtype=np.float64)
    if acc is None:
        acc = {"n": 0, "sum": np.zeros(d.shape[1], np.float64),
               "sumsq": np.zeros(d.shape[1], np.float64)}
    acc["n"] += d.shape[0]
    acc["sum"] += d.sum(0)
    acc["sumsq"] += (d ** 2).sum(0)
    return acc


def _finalize_channel_stats(acc: dict) -> dict:
    """Running sums -> {"mean", "std", "rms", "count"}."""
    n = max(int(acc["n"]), 1)
    mean = acc["sum"] / n
    msq = acc["sumsq"] / n
    var = np.maximum(msq - mean ** 2, 0.0)
    return {"mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
            "rms": np.sqrt(msq).astype(np.float32),
            "count": int(acc["n"])}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_descriptor_scaling.py -q`
Expected: `9 passed`

### Task 2.2: Hook into `build_and_reduce`

**Files:** Modify `data.py:534` (signature), `~583` (loop body), `~620` (return)

- [ ] **Step 1: Add the parameter and accumulate**

Signature gains `collect_stats: bool = False`. Initialise `stats_acc = None` beside `buffers`. Inside the batch loop, after `pad_and_stack` produces `padded` and **before** the COO keys are popped at `:593-595`:

```python
        if collect_stats:
            # Real atoms only — padded rows are all-zero and would drag every
            # channel's mean and std toward zero.
            dsc = padded["descriptors"].numpy()
            msk = padded["atom_mask"].numpy().astype(bool)
            stats_acc = _accumulate_channel_stats(stats_acc, dsc[msk])
```

- [ ] **Step 2: Attach to the returned dict**

`build_and_reduce` ends with a dict comprehension (`data.py:620-621`), so the stats must be attached to the result — they cannot go through `buffers`, which is preallocated from tensor shapes and wrapped in `tf.constant`:

```python
    with place:
        out = {k: tf.constant(v) for k, v in buffers.items()}
    if collect_stats:
        out["_channel_stats"] = _finalize_channel_stats(stats_acc)
    return out
```

- [ ] **Step 3: Enable for the train split only, and only when needed**

In `data.split()`, the `train_dataset` call gains `collect_stats=_wants_stats(cfg)`; the `val_dataset` call does not. Add beside it:

```python
def _wants_stats(cfg) -> bool:
    """Statistics are only collected when a preconditioning mode needs them.

    Keeps the default path free of both the accumulation cost and the
    cfg attribute (see _serialize_config's handling of runtime extras).
    """
    return (str(getattr(cfg, "descriptor_sigma_scaling", "off")) != "off"
            or str(getattr(cfg, "descriptor_weight_reparam", "off")) != "off")
```

- [ ] **Step 4: Lift onto cfg under an underscore name**

In `MasterTNEP._train_model_inner`, immediately after `split()` returns:

```python
    if "_channel_stats" in train_data:
        # Underscore-prefixed: _serialize_config (model_io.py:127, :131)
        # skips these, and the ndarray values would otherwise blow up
        # json.dumps on the first checkpoint write.
        cfg._descriptor_channel_stats = train_data.pop("_channel_stats")
```

Computing from the training split only is the same discipline as any fitted preprocessing — val and test must not influence it.

- [ ] **Step 5: Verify the default path is untouched**

Run: `python -c "
from TNEPconfig import TNEPconfig
from model_io import _serialize_config
import json
cfg = TNEPconfig(); cfg.types=[8,1]; cfg.num_types=2
cfg._descriptor_channel_stats = {'std': __import__('numpy').ones(4)}
json.dumps(_serialize_config(cfg)); print('serialises OK with stats attached')"`
Expected: `serialises OK with stats attached`

- [ ] **Step 6: Commit**

```bash
git add data.py MasterTNEP.py tests/test_descriptor_scaling.py
git commit -m "data: per-channel descriptor statistics over the train split"
```

---

## Phase 3 — Resolve modes and guards (shared by A and B)

### Task 3.1: One resolution block, above `_build_mu_init`

**Files:** Modify `SNES.__init__`, inserting **before** `mu_init = self._build_mu_init(rng)` at `SNES.py:307`

Both modes must be resolved before `_build_mu_init` (mode B divides there) and before `sigma_init_vec` (mode A multiplies there). Resolving them in two places is how the first draft of this plan ended up referencing `sigma_mode` before assignment — a bug Python's short-circuiting would have hidden until mode B was switched on.

- [ ] **Step 1: Write the failing guard tests**

```python
def _tiny_cfg(**kw):
    from TNEPconfig import TNEPconfig
    from DescriptorBuilderGPU import compute_dim_q
    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.l_max = cfg.alpha_max = 2
    cfg.num_neurons = 3
    cfg.types = [8, 1]; cfg.num_types = 2; cfg.type_map = {8: 0, 1: 1}
    cfg.dim_q = compute_dim_q(cfg)
    cfg.descriptor_mixing = False
    for k, v in kw.items():
        setattr(cfg, k, v)
    q = int(cfg.dim_q)
    cfg._descriptor_channel_stats = {
        "mean": np.ones(q, np.float32), "std": np.ones(q, np.float32),
        "rms": np.ones(q, np.float32), "count": 100}
    return cfg


def test_both_modes_at_once_raises():
    from TNEP import TNEP
    cfg = _tiny_cfg(descriptor_sigma_scaling="std",
                    descriptor_weight_reparam="std")
    with pytest.raises(ValueError, match="mutually exclusive"):
        TNEP(cfg)


def test_pes_mode_raises():
    from TNEP import TNEP
    cfg = _tiny_cfg(target_mode=0, descriptor_sigma_scaling="std")
    with pytest.raises(ValueError, match="target_mode"):
        TNEP(cfg)


def test_missing_statistics_raise_clearly():
    from TNEP import TNEP
    cfg = _tiny_cfg(descriptor_sigma_scaling="std")
    del cfg._descriptor_channel_stats
    with pytest.raises(ValueError, match="statistics"):
        TNEP(cfg)


def test_bad_mode_string_raises():
    from TNEP import TNEP
    with pytest.raises(ValueError, match="not in"):
        TNEP(_tiny_cfg(descriptor_sigma_scaling="zscore"))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_descriptor_scaling.py -k "raises" -q`
Expected: 4 failed with `DID NOT RAISE`

- [ ] **Step 3: Implement the resolution block**

```python
        # ── Search preconditioning: resolve BOTH modes before _build_mu_init
        # (mode B divides there) and before sigma_init_vec (mode A multiplies
        # there). One block so the two can never disagree.
        import descriptor_scaling as dsc
        sigma_mode = str(getattr(self.cfg, "descriptor_sigma_scaling", "off"))
        reparam_mode = str(getattr(self.cfg, "descriptor_weight_reparam", "off"))
        for name, val in (("descriptor_sigma_scaling", sigma_mode),
                          ("descriptor_weight_reparam", reparam_mode)):
            if val not in dsc.VALID_MODES:
                raise ValueError(f"cfg.{name}={val!r} not in {dsc.VALID_MODES}")
        if sigma_mode != "off" and reparam_mode != "off":
            raise ValueError(
                "descriptor_sigma_scaling and descriptor_weight_reparam are "
                "mutually exclusive — mode B already yields a scale-fair "
                "search, so enabling both applies the multiplier twice.")

        self._chan_mult = None
        if sigma_mode != "off" or reparam_mode != "off":
            if self.cfg.target_mode == 0:
                raise ValueError(
                    "Search preconditioning requires target_mode 1 or 2: the "
                    "channel statistics are collected in build_and_reduce, "
                    "which the PES path does not use.")
            if self.cfg.descriptor_preprocess_contract != "off":
                raise ValueError(
                    "Search preconditioning requires "
                    "descriptor_preprocess_contract='off': W0 is stored at "
                    "Q_new while the channel statistics are at Q_raw.")
            # A restored multiplier always wins — a resumed run must never
            # re-derive it from a differently-sampled statistic.
            restored = getattr(self.cfg, "descriptor_scale", None)
            if restored is not None:
                self._chan_mult = np.asarray(restored, np.float32)
            else:
                stats = getattr(self.cfg, "_descriptor_channel_stats", None)
                if stats is None:
                    raise ValueError(
                        "Search preconditioning is enabled but no descriptor "
                        "channel statistics are available. They are collected "
                        "in data.split() when a mode is active — check that "
                        "cfg was set before split() ran.")
                # Block granularity whenever mixing is on: diag(m) must
                # commute with U, which is block-diagonal over (pair, l).
                blocks = None
                if getattr(self.model, "descriptor_mixing", False):
                    from DescriptorBuilderGPU import descriptor_block_layout
                    lay = descriptor_block_layout(self.cfg)
                    blocks = [lay["pair_ln_index"][pk][l]
                              for pk in lay["pair_keys"]
                              for l in sorted(lay["pair_ln_index"][pk])]
                self._chan_mult = dsc.channel_multipliers(
                    stats, sigma_mode if sigma_mode != "off" else reparam_mode,
                    float(self.cfg.descriptor_scaling_clamp),
                    float(self.cfg.descriptor_scaling_exponent),
                    blocks=blocks)
            if len(self._chan_mult) != self.cfg.dim_q:
                raise ValueError(
                    f"channel multiplier has {len(self._chan_mult)} entries "
                    f"but dim_q={self.cfg.dim_q}")
        # Mode B only: mu holds W0_hat, so reconstruct multiplies by this.
        self._w0_reparam = self._chan_mult if reparam_mode != "off" else None
        self._w0_reparam_tf = (tf.constant(self._w0_reparam)
                               if self._w0_reparam is not None else None)
        self._w0_coord_scale = (
            dsc.expand_to_w0_coords(self._chan_mult, self.cfg.num_types,
                                    self.cfg.num_neurons)
            if self._chan_mult is not None else None)
        self._sigma_precondition = (sigma_mode != "off")
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_descriptor_scaling.py -k "raises" -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add SNES.py tests/test_descriptor_scaling.py
git commit -m "SNES: resolve preconditioning modes and guards in one place"
```

---

## Phase 4 — Mode A: scale the search distribution

### Task 4.1: Multiply the W0 region of `sigma_init_vec`

**Files:** Modify `SNES.py:309-323`

- [ ] **Step 1: Write the failing test**

```python
def test_mode_a_scales_only_the_w0_block():
    """sigma is multiplied on W0 coordinates and left alone elsewhere."""
    from TNEP import TNEP
    # Override exponent AND clamp: at the defaults (0.5, 64.0) the softening
    # halves the log-ratio and the clamp then truncates it, giving ~8.9.
    cfg = _tiny_cfg(descriptor_sigma_scaling="std",
                    descriptor_scaling_exponent=1.0,
                    descriptor_scaling_clamp=1e9)
    q, h, t = int(cfg.dim_q), cfg.num_neurons, cfg.num_types
    std = np.ones(q, np.float32); std[0] = 0.01          # one small channel
    cfg._descriptor_channel_stats = {
        "mean": np.ones(q, np.float32), "std": std,
        "rms": np.ones(q, np.float32), "count": 100}
    s = TNEP(cfg).optimizer.sigma.numpy()
    n_w0 = t * q * h
    # channel 0 vs channel 1, first type block
    assert s[0] / s[h] == pytest.approx(100.0, rel=1e-4)
    # b0/W1/b1 coordinates untouched
    np.testing.assert_allclose(s[n_w0:], cfg.init_sigma, rtol=1e-6)


def test_mode_a_scales_the_pol_block_too():
    """target_mode=2 has a second W0 at offset n_primary — the only place the
    offset arithmetic is exercised, and where both review rounds found bugs."""
    from TNEP import TNEP
    cfg = _tiny_cfg(target_mode=2, descriptor_sigma_scaling="std",
                    descriptor_scaling_exponent=1.0,
                    descriptor_scaling_clamp=1e9)
    q, h = int(cfg.dim_q), cfg.num_neurons
    std = np.ones(q, np.float32); std[0] = 0.01
    cfg._descriptor_channel_stats = {
        "mean": np.ones(q, np.float32), "std": std,
        "rms": np.ones(q, np.float32), "count": 100}
    snes = TNEP(cfg).optimizer
    s = snes.sigma.numpy()
    p = snes.n_primary
    assert s[0] / s[h] == pytest.approx(100.0, rel=1e-4)            # main ANN
    assert s[p] / s[p + h] == pytest.approx(100.0, rel=1e-4)        # pol ANN


def test_model_round_trip_restores_the_multiplier(tmp_path):
    """A preconditioned model must be loadable. _load_model_h5 rebuilds cfg
    from JSON (which carries the mode strings) and calls TNEP(cfg); without a
    restored multiplier the Task 3.1 guard would reject every saved model."""
    from TNEP import TNEP
    from model_io import save_model, load_model
    cfg = _tiny_cfg(descriptor_sigma_scaling="std")
    m = TNEP(cfg)
    save_model(m, cfg, path=str(tmp_path))
    import glob
    m2 = load_model(glob.glob(str(tmp_path / "*.h5"))[0])
    np.testing.assert_allclose(m2.optimizer._chan_mult, m.optimizer._chan_mult,
                               rtol=1e-6)


def test_mode_a_off_leaves_sigma_uniform():
    from TNEP import TNEP
    s = TNEP(_tiny_cfg()).optimizer.sigma.numpy()
    np.testing.assert_allclose(s, s[0], rtol=1e-7)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_descriptor_scaling.py -k mode_a -q`
Expected: `test_mode_a_scales_only_the_w0_block` FAILS (`1.0 != 100.0`); the `off` test passes.

- [ ] **Step 3: Implement**

Immediately after `sigma_init_vec` is built:

```python
        if self._sigma_precondition:
            sigma_init_vec[:self._n_W0] *= self._w0_coord_scale
            if self.cfg.target_mode == 2:
                p = self.n_primary
                sigma_init_vec[p:p + self._n_W0] *= self._w0_coord_scale
```

Note `self._n_W0` (`SNES.py:99`) and `self.cfg.num_types` — `SNES` has no `n_W0` or `num_types` attribute.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_descriptor_scaling.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add SNES.py tests/test_descriptor_scaling.py
git commit -m "SNES: per-channel sigma preconditioning on the W0 block (mode A)"
```

Mode A needs **no persistence work**: `sigma` is already saved to and restored from the checkpoint (`MasterTNEP.py:538`), and `μ` keeps its existing meaning.

---

## Phase 5 — Mode B: reparameterise the weights

Mode B is the complete fix. `μ` holds `Ŵ0`; uniform `sigma` and uniform L1/L2 over `Ŵ0` are **both** automatically scale-fair, because a channel's output contribution is `s_k·m_k·Ŵ0_k = const·Ŵ0_k`. `compute_regularization` (`SNES.py:397-424`) and `compute_regularization_tf` (`SNES.py:1640-1671`) slice the raw search vector in both the `T>1` and `T==1` branches, so **neither needs changing** — that is the point of doing it this way rather than adding per-channel weights to the penalty.

### Task 5.1: Divide in `_build_mu_init`, multiply in `reconstruct_params_tf`

**Files:** Modify `SNES.py:428` (`_build_mu_init`), `SNES.py:1402-1486` (`reconstruct_params_tf`)

- [ ] **Step 1: Write the failing test**

```python
def test_mode_b_is_a_pure_reparameterisation():
    """Effective W0 at generation 0 is identical with reparam on and off.

    _build_mu_init emits Glorot/m and reconstruct multiplies by m, so the
    round trip is the identity. If it is not, enabling the option silently
    changes the initial model.
    """
    from TNEP import TNEP
    q = None
    outs = {}
    for mode in ("off", "std"):
        cfg = _tiny_cfg(descriptor_weight_reparam=mode, seed=1234)
        q = int(cfg.dim_q)
        std = np.linspace(0.01, 1.0, q).astype(np.float32)
        cfg._descriptor_channel_stats = {
            "mean": np.ones(q, np.float32), "std": std,
            "rms": np.ones(q, np.float32), "count": 100}
        snes = TNEP(cfg).optimizer
        outs[mode] = snes._split_reconstructed(
            snes.reconstruct_params_tf(snes.mu))["W0"].numpy()
    np.testing.assert_allclose(outs["off"], outs["std"], rtol=2e-5, atol=1e-7)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_descriptor_scaling.py -k mode_b -q`
Expected: FAIL — effective W0 differs by the multiplier

- [ ] **Step 3: Implement**

In `_build_mu_init`, after the Glorot W0 block is filled (reuse the shared helper rather than re-deriving the tile/repeat):

```python
        if self._w0_reparam is not None:
            inv = 1.0 / self._w0_coord_scale
            mu[:self._n_W0] *= inv
            if self.cfg.target_mode == 2:
                p = self.n_primary
                mu[p:p + self._n_W0] *= inv
```

In `reconstruct_params_tf`, inside the closure that reshapes a slice to `[..., T, Q, H]` (so `W0_pol` is covered by the same code):

```python
            if self._w0_reparam_tf is not None:
                W0 = W0 * self._w0_reparam_tf[tf.newaxis, :, tf.newaxis]
```

`[1, Q, 1]` broadcasts against both `[T, Q, H]` and `[C, T, Q, H]`. The constant is hoisted to `__init__` so it is not rebuilt on every trace.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_descriptor_scaling.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add SNES.py tests/test_descriptor_scaling.py
git commit -m "SNES: W0 search-space reparameterisation (mode B)"
```

### Task 5.2: Persist the multiplier

**Files:** Modify `SNES.py:1162, 1181` (the two `ckpt_state` dicts), `model_io.py:278` (`save_checkpoint`), `:339` (`load_checkpoint`), `:155` (`save_model`), `:461` (`_load_model_h5`), `:20` (`_LEGACY_FIELD_DEFAULTS`)

**Required for correctness, not convenience.** Under mode B, `μ` holds `Ŵ0`; resuming without the multiplier reconstructs a different effective `W0` and silently continues training a different model.

`save_checkpoint`'s signature is `(path, cfg, state, history, last_gen)` — there is no `model` argument, so the multiplier travels in the `state` dict SNES already builds.

- [ ] **Step 1: Write the failing tests**

```python
def test_checkpoint_round_trip_preserves_the_multiplier(tmp_path):
    import h5py
    from TNEPconfig import TNEPconfig
    from model_io import save_checkpoint, load_checkpoint
    cfg = _tiny_cfg(descriptor_weight_reparam="std")
    scale = np.linspace(0.5, 2.0, int(cfg.dim_q)).astype(np.float32)
    state = {"mu": np.zeros(4, np.float32), "sigma": np.ones(4, np.float32),
             "best_mu": np.zeros(4, np.float32), "best_sigma": None,
             "best_val_loss": 1.0, "gens_without_improvement": 0,
             "descriptor_scale": scale}
    p = str(tmp_path / "ckpt.h5")
    save_checkpoint(p, cfg, state, {"generation": [0], "val_loss": [1.0]}, 0)
    cfg2, _ = load_checkpoint(p)
    np.testing.assert_array_equal(cfg2.descriptor_scale, scale)


def test_resume_without_saved_multiplier_raises(tmp_path):
    """Hand-craft the broken file: save normally, then delete the dataset.
    The new save_checkpoint can never produce this combination itself."""
    import h5py
    from model_io import save_checkpoint, load_checkpoint
    cfg = _tiny_cfg(descriptor_weight_reparam="std")
    scale = np.ones(int(cfg.dim_q), np.float32)
    state = {"mu": np.zeros(4, np.float32), "sigma": np.ones(4, np.float32),
             "best_mu": np.zeros(4, np.float32), "best_sigma": None,
             "best_val_loss": 1.0, "gens_without_improvement": 0,
             "descriptor_scale": scale}
    p = str(tmp_path / "ckpt.h5")
    save_checkpoint(p, cfg, state, {"generation": [0], "val_loss": [1.0]}, 0)
    with h5py.File(p, "a") as f:
        del f["snes"]["descriptor_scale"]
    with pytest.raises(ValueError, match="descriptor_scale"):
        load_checkpoint(p)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_descriptor_scaling.py -k "round_trip or without_saved" -q`
Expected: 2 failed

- [ ] **Step 3: Implement**

In both `ckpt_state` dicts in `SNES.fit` (the dicts at `SNES.py:1154-1160` and
`:1174-1180`; the `save_checkpoint` calls are eight lines below each). Persist
`_chan_mult`, not `_w0_reparam` — mode A needs it too, both so a resumed run
never re-derives the multiplier and so a saved model can be loaded at all:

```python
                    "descriptor_scale": self._chan_mult,   # None when off
```

In `save_checkpoint`, inside the `/snes` group (`sg`, created at `model_io.py:307`):

```python
        if state.get("descriptor_scale") is not None:
            sg.create_dataset("descriptor_scale",
                              data=np.asarray(state["descriptor_scale"],
                                              dtype=np.float32))
```

In `load_checkpoint`, after the `/snes` reads (`sg = f["snes"]`, `:364`):

Note the indent: `load_checkpoint`'s file body sits at **8 spaces** inside the
`with h5py.File(...)` block (`model_io.py:364`), which closes after `:390`.
Pasting this at 4 spaces would query a closed HDF5 group.

```python
        if "descriptor_scale" in sg:
            cfg.descriptor_scale = sg["descriptor_scale"][:].astype(np.float32)
        elif str(getattr(cfg, "descriptor_weight_reparam", "off")) != "off":
            raise ValueError(
                f"{path!r} was trained with descriptor_weight_reparam="
                f"{cfg.descriptor_weight_reparam!r} but has no "
                f"/snes/descriptor_scale. mu holds the reparameterised W0, so "
                f"resuming without the multiplier would train a different "
                f"model.")
```

In `save_model`, inside the existing `/descriptor` group (`dg`, `model_io.py:223`). This is **required**, not provenance: `_load_model_h5` restores cfg from the config JSON — which does contain the two mode strings, since `_serialize_config` walks `__annotations__` — and then calls `TNEP(cfg)` at `model_io.py:496`. On that path there are no channel statistics (underscore-prefixed, deliberately not serialised), so without a restored multiplier the Task 3.1 guard fires and **every preconditioned model is unloadable**. Write it for both modes:

```python
        if getattr(model.optimizer, "_chan_mult", None) is not None:
            dg.create_dataset("channel_scale",
                              data=model.optimizer._chan_mult)
```

In `_load_model_h5`, **before** `model = TNEP(cfg)` at `model_io.py:496`:

```python
        if "descriptor" in f and "channel_scale" in f["descriptor"]:
            cfg.descriptor_scale = (
                f["descriptor"]["channel_scale"][:].astype(np.float32))
```

In `_LEGACY_FIELD_DEFAULTS`:

```python
    "descriptor_sigma_scaling": "off",
    "descriptor_weight_reparam": "off",
    "descriptor_scaling_exponent": 0.5,
    "descriptor_scaling_clamp": 64.0,
```

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_descriptor_scaling.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add SNES.py model_io.py tests/test_descriptor_scaling.py
git commit -m "model_io: persist the W0 reparameterisation multiplier"
```

---

## Phase 6 — Verification

### Task 6.1: Runnable smoke script

**Files:** Create `scripts/smoke_modes.py`

- [ ] **Step 1: Create it**

```python
"""Short end-to-end run across target modes and preconditioning options.

    MODE=2 OPT=weight_reparam STAT=std python scripts/smoke_modes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TNEPconfig import TNEPconfig                       # noqa: E402
from MasterTNEP import train_model                      # noqa: E402

cfg = TNEPconfig()
cfg.target_mode = int(os.environ.get("MODE", 2))
if cfg.target_mode == 0:
    cfg.data_path, cfg.test_data_path = "datasets/PEStrain.xyz", None
    cfg.allowed_species = None
opt, stat = os.environ.get("OPT"), os.environ.get("STAT", "std")
if opt:
    setattr(cfg, f"descriptor_{opt}", stat)
if os.environ.get("PRE"):                       # exercise the preprocess guard
    cfg.descriptor_preprocess_contract = os.environ["PRE"]
# N unset (or empty) means the WHOLE dataset. total_N=0 is not "no limit":
# TNEPconfig.randomise (TNEPconfig.py:314) truncates to zero structures and
# build_and_reduce then raises on the empty dataset.
_n = os.environ.get("N")
cfg.total_N = int(_n) if _n else None
cfg.num_generations = int(os.environ.get("GENS", 6))
cfg.val_interval = 2
cfg.checkpoint_interval = 10 ** 9
cfg.save_plots = None                           # str | None, not bool
cfg.show_plots = False
cfg.save_path = os.environ.get("OUT", "models/smoke")
train_model(cfg=cfg)
print(f"OK mode={cfg.target_mode} opt={opt or 'off'} stat={stat}")
```

- [ ] **Step 2: Commit**

```bash
git add scripts/smoke_modes.py && git commit -m "scripts: smoke runner for target modes and preconditioning options"
```

### Task 6.2: Defaults must be bit-identical

- [ ] Run `GENS=200 OUT=models/verify_off python scripts/smoke_modes.py` before and after the whole change (`N` unset = full dataset).
- [ ] **Acceptance:** `val_loss` arrays from the two `history.csv` files are `np.array_equal`. Anything else means a default path was disturbed. This is the check that the first draft of this plan would have passed while still harbouring a mode-B `NameError`, so do not treat it as sufficient on its own — Task 6.3 must also pass.

### Task 6.3: Every mode and option combination constructs

- [ ] `MODE ∈ {0,1,2}` with no option → all succeed.
- [ ] `MODE ∈ {1,2}` × `OPT ∈ {sigma_scaling, weight_reparam}` × `STAT=std` → all succeed.
- [ ] `MODE=0 OPT=sigma_scaling` → raises the `target_mode` guard.
- [ ] `MODE=2 OPT=sigma_scaling PRE=species_pair` → raises the preprocess guard.
- [ ] Resume: run 20 generations under `weight_reparam=std`, resume from `checkpoint.h5`, confirm the first resumed `val_loss` matches the last pre-resume value.

### Task 6.4: Confirm the block granularity actually commutes

- [ ] **Write the test** `test_block_multiplier_commutes_with_mixing`: build a `TNEP` with `descriptor_mixing=True` and a non-identity `U_pair`, then assert
  `_W0_eff(diag(m) @ W0) == diag(m) @ _W0_eff(W0)` to fp32 noise. This is the property the whole design now rests on; if it fails, per-channel granularity is unsafe under mixing and the modes must be guarded to `descriptor_mixing=False`.
- [ ] Re-measure both spreads on whatever dataset is actually being trained (the 24.8x / 146.2x figures are `train_waterbulk` at `l_max=4`). If a different dataset shows within-block spread far *below* between-`l`, per-channel granularity becomes safe again and would be strictly better — record the numbers either way.

### Task 6.5: The measurement that decides whether to keep any of this

**This is a hypothesis, not an improvement, until this task says otherwise.**

- [ ] Four runs, identical seed and config apart from the option, 20,000 generations, `target_mode=2` on `train_waterbulk.xyz`:
  `off` · `sigma_scaling="std"` · `weight_reparam="std"` · `weight_reparam="std", exponent=1.0`
- [ ] Record per run: best `val_loss` and the generation it was reached; `sigma_median` **normalised to its own generation-0 value** (mode A shifts gen-0 `sigma_median` outright, because unit *geometric* mean does not preserve the median — comparing raw trajectories would be comparing an offset series against a non-offset one); and the ablation-importance profile of the trained model.
- [ ] Note that `L1`/`L2` in `history.csv` are penalties on `Ŵ0` under mode B and so are **not** comparable across runs; `_maybe_adapt_lambda` (`SNES.py:481-514`) will also settle at a different λ under the dynamic sentinel.
- [ ] **Acceptance:** a mode is kept only if best val RMSE improves by more than the measured run-to-run spread. Measure that spread from repeat `off` runs — do not assume it.
- [ ] **If no mode beats `off`, revert Phases 3-5 and record the negative result.** The prediction under test is that normalised `sigma_median` contracts further than the 9% seen over 30,000 generations, and that `l≥3` channels gain ablation importance. If σ moves but val RMSE does not, the channels are genuinely uninformative and the correct response is lowering `l_max`, not preconditioning.
- [ ] Write findings to `docs/superpowers/notes/2026-08-06-search-preconditioning-results.md`.

---

## Out of scope

- **PES (`target_mode=0`).** Statistics hook lives in `build_and_reduce`. Guarded.
- **`descriptor_preprocess_contract != "off"`.** `W0` at `Q_new` vs statistics at `Q_raw`. Guarded.
- **Per-channel weights in `compute_regularization`.** Mode B makes the existing uniform penalty scale-fair; adding them would double-correct.
- **Re-deriving the multiplier mid-run.** Frozen after first computation; a restored value always wins.
- **Rebalancing across `l` via the mixing layer.** Structurally impossible — every arch has block support within a single `l`, and `expm`/`cayley` are orthogonal hence norm-preserving.

## Acceptance criteria

- `python -m pytest tests/test_descriptor_scaling.py -q` passes in full.
- Defaults bit-identical to the pre-change pipeline (Task 6.2) **and** every option combination constructs (Task 6.3).
- Mode B at generation 0 produces an effective `W0` identical to `"off"` (Task 5.1) — proving reparameterisation, not model change.
- Statistics come from the training split only and are frozen across resume.
- A mode-B checkpoint missing its multiplier raises rather than silently training a different model.
- A model saved under **either** mode loads back successfully (`test_model_round_trip_restores_the_multiplier`).
- Task 6.5 recorded with real numbers, and the code reverted if they do not support keeping it.

## Reusable skills

- @superpowers:test-driven-development for Phases 1-5
- @superpowers:verification-before-completion for Phase 6
