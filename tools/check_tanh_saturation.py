"""Tanh-saturation diagnostic for a trained TNEP model.

Loads a model checkpoint and a dataset, runs the descriptor → W0 → tanh
forward path for every real atom, and reports the fraction of hidden-unit
activations that sit close to ±1 (i.e. the gradient through tanh is dead).

Usage:
    python tools/check_tanh_saturation.py [MODEL_PATH] [-d DATASET_PATH]
                                          [--max-structures N]
                                          [--split {train,test,val,all}]

Examples:
    # Use the model's own cfg.data_path + train-split, default thresholds
    python tools/check_tanh_saturation.py models/.../model.h5

    # Diagnostic on a fresh (untrained) model — useful baseline
    python tools/check_tanh_saturation.py --fresh

    # Override the dataset and limit to 200 structures for speed
    python tools/check_tanh_saturation.py models/.../model.h5 \\
        -d datasets/test.xyz --max-structures 200

Interpretation:
    A trained TNEP that's healthy will have <20% of (atom, unit) activations
    above |h| = 0.99 — the network still has gradient signal to learn from.
    Above ~50% saturation is a red flag: most weights have effectively
    constant outputs through tanh.  Per-unit saturation at 100% (the unit
    *always* fires +1 or always −1) means the unit is dead — usually the
    sign of an over-large b0 or W0 init.  Glorot/Xavier init was designed
    to keep the per-unit pre-activation variance near 1, giving ~5-15%
    saturation at init — anything materially higher at gen 0 is a config
    problem upstream of training.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Headless TF
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf

# Project-local imports
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from TNEPconfig import TNEPconfig                          # noqa: E402
from TNEP import TNEP                                       # noqa: E402
from data import collect, split, pad_and_stack             # noqa: E402
from DescriptorBuilderGPU import compute_dim_q             # noqa: E402


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def load_model_or_fresh(model_path: str | None) -> TNEP:
    """Either load a checkpoint or build an untrained model from defaults."""
    if model_path is None:
        cfg = TNEPconfig()
        cfg.descriptor_mode = 0
        cfg.population_chunk_size = None
        cfg.batch_chunk_size = None
        cfg.pin_data_to_cpu = True
        cfg.cache_gradients_to_disk = False
        cfg.chunk_prefetch = False
        cfg.use_pinned_buffers = False
        cfg.use_cufile = False
        cfg.save_path = None
        cfg.snes_msr_enabled = False
        cfg.mu_init_scheme = "glorot"
        dataset, ti = collect(cfg)
        cfg.randomise(dataset)
        cfg.dim_q = compute_dim_q(cfg)
        return TNEP(cfg)
    from model_io import load_model
    return load_model(model_path)


def build_dataset_for_model(model: TNEP,
                             override_data_path: str | None,
                             split_choice: str,
                             max_structures: int | None) -> dict:
    """Build a padded data dict from the model's cfg or an override path."""
    cfg = model.cfg
    if override_data_path is not None:
        cfg.data_path = override_data_path
    # Force descriptor build via the same builder the model used.
    dataset, ti = collect(cfg)
    # Re-randomise on the actual dataset size (the model's saved split
    # indices may reference a different-sized dataset; this guards against
    # IndexError when the override data is smaller).
    cfg.randomise(dataset)
    if split_choice == "all":
        td = dataset
    else:
        td_train, td_test, td_val = split(dataset, ti, cfg)
        td = {"train": td_train, "test": td_test, "val": td_val}[split_choice]
    if max_structures is not None and len(td) > max_structures:
        td = td[:max_structures]
    return pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)


# ----------------------------------------------------------------------------
# Forward & stats
# ----------------------------------------------------------------------------
def compute_hidden_activations(model: TNEP, data: dict
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pre_h, h, Z) flattened to [N_real, H] / [N_real, H] / [N_real]
    for every real (non-padded) atom in the data dict."""
    descriptors = tf.convert_to_tensor(data["descriptors"], dtype=tf.float32)
    Z = tf.convert_to_tensor(data["Z_int"], dtype=tf.int32)
    atom_mask = tf.convert_to_tensor(data["atom_mask"], dtype=tf.float32)

    # Fold mixing / preprocess into W0 so we operate at raw Q.
    W0 = model.W0                                          # [T, Q_storage, H]
    if model.descriptor_mixing:
        W0_eff = model._W0_eff(W0)                         # [T, Q_raw, H]
    elif model.descriptor_preprocess_contract != "off":
        W0_eff = model._W0_preprocess_eff(W0)              # [T, Q_raw, H]
    else:
        W0_eff = W0                                        # [T, Q_raw, H]

    b0 = model.b0                                          # [T, H]

    # Per-atom forward: pre_h[b, a, h] = Σ_q W0_eff[Z[b,a], q, h] · desc[b, a, q]
    # Gather W0_eff per atom along the T axis.
    W0_per_atom = tf.gather(W0_eff, Z, axis=0)             # [B, A, Q_raw, H]
    b0_per_atom = tf.gather(b0, Z, axis=0)                 # [B, A, H]
    pre_h = tf.einsum("baq,baqh->bah", descriptors, W0_per_atom) + b0_per_atom
    h = tf.tanh(pre_h)

    # Flatten and keep only real atoms.
    real_mask = (atom_mask.numpy() > 0).reshape(-1)
    H = int(h.shape[-1])
    pre_h_flat = pre_h.numpy().reshape(-1, H)[real_mask]
    h_flat = h.numpy().reshape(-1, H)[real_mask]
    Z_flat = Z.numpy().reshape(-1)[real_mask]
    return pre_h_flat, h_flat, Z_flat


def saturation_table(h: np.ndarray, thresholds: list[float]) -> dict:
    """For each threshold, return the fraction of |h| > threshold."""
    abs_h = np.abs(h)
    return {t: float((abs_h > t).mean()) for t in thresholds}


def per_unit_saturation(h: np.ndarray, threshold: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-hidden-unit fraction saturated, plus the sign-bias (mean h)."""
    abs_h = np.abs(h)
    sat_frac = (abs_h > threshold).mean(axis=0)            # [H]
    mean_h = h.mean(axis=0)                                # [H]  in (−1, 1)
    return sat_frac, mean_h


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def emit_report(pre_h: np.ndarray, h: np.ndarray, Z: np.ndarray,
                model: TNEP) -> None:
    thresholds = [0.90, 0.95, 0.99, 0.999]
    H = int(h.shape[1])
    N = int(h.shape[0])

    print("=" * 72)
    print(f"Tanh-saturation report  —  {N} real atoms × {H} hidden units")
    print("=" * 72)

    # 1. Aggregate over all atoms and units.
    overall = saturation_table(h, thresholds)
    print("\n[1] Overall fraction with |h| above threshold")
    for t, frac in overall.items():
        bar = "█" * int(round(40 * frac))
        print(f"     |h| > {t:<5} : {frac * 100:6.2f}%   {bar}")

    # 2. Pre-activation magnitude distribution.
    pre_abs = np.abs(pre_h)
    print("\n[2] Pre-activation magnitude distribution (|W0 · q + b0|)")
    print(f"     mean  = {pre_abs.mean():.3f}")
    print(f"     median= {np.median(pre_abs):.3f}")
    print(f"     p90   = {np.percentile(pre_abs, 90):.3f}")
    print(f"     p99   = {np.percentile(pre_abs, 99):.3f}")
    print(f"     max   = {pre_abs.max():.3f}")
    print("     Interpretation: |z| ≈ 1 → tanh in linear regime;"
          " |z| > 3 → strongly saturating.")

    # 3. Per-species breakdown.
    print("\n[3] Per-species saturation (|h| > 0.99)")
    types_present = sorted(np.unique(Z).tolist())
    type_labels = getattr(model.cfg, "types", None)
    label_of = (lambda t: f"type {t} (Z={type_labels[t]})") if type_labels \
        else (lambda t: f"type {t}")
    print(f"     {'species':<22} {'N_atoms':>10}  {'sat_frac':>10}")
    for t in types_present:
        mask = (Z == t)
        n_t = int(mask.sum())
        if n_t == 0:
            continue
        frac = float((np.abs(h[mask]) > 0.99).mean())
        print(f"     {label_of(t):<22} {n_t:>10}  {frac * 100:>9.2f}%")

    # 4. Per-unit health.
    sat_frac, mean_h = per_unit_saturation(h, threshold=0.99)
    print(f"\n[4] Per-hidden-unit health ({H} units)")
    dead_high = int(((sat_frac > 0.95) & (mean_h > 0.5)).sum())
    dead_low = int(((sat_frac > 0.95) & (mean_h < -0.5)).sum())
    stuck = int(((sat_frac > 0.95)
                 & (mean_h >= -0.5) & (mean_h <= 0.5)).sum())
    healthy = int((sat_frac < 0.20).sum())
    transition = H - dead_high - dead_low - stuck - healthy
    print(f"     dead (always +1, sat>95%, ⟨h⟩>0.5)  : {dead_high}")
    print(f"     dead (always −1, sat>95%, ⟨h⟩<−0.5) : {dead_low}")
    print(f"     stuck (sat>95%, mixed sign)          : {stuck}")
    print(f"     healthy (sat<20%)                    : {healthy}")
    print(f"     intermediate                         : {transition}")

    # 5. List the worst (most-saturated) units.
    worst_n = min(10, H)
    worst_idx = np.argsort(-sat_frac)[:worst_n]
    print(f"\n[5] Top-{worst_n} most-saturated hidden units (|h| > 0.99)")
    print(f"     {'unit':>6}  {'sat_frac':>10}  {'mean(h)':>10}")
    for k in worst_idx:
        print(f"     {int(k):>6}  {sat_frac[k] * 100:>9.2f}%  "
              f"{mean_h[k]:>+10.4f}")

    # 6. Verdict.
    print("\n[6] Verdict")
    sat99 = overall[0.99]
    if sat99 < 0.20:
        verdict = "OK — most units have headroom in the linear regime."
    elif sat99 < 0.50:
        verdict = ("BORDERLINE — many units near saturation. Worth "
                   "re-checking init scale or W0 σ.")
    else:
        verdict = ("CONCERNING — >50% of (atom, unit) activations are "
                   "saturated. Likely causes: b0 too large at init, "
                   "uncentred / unscaled descriptor input, σ exploration "
                   "drove W0 norms too high.")
    print(f"     {verdict}")
    print("=" * 72)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", nargs="?", default=None,
                        help="Path to a saved model (.h5 or .npz). "
                             "Omit to build a fresh untrained model from "
                             "TNEPconfig defaults.")
    parser.add_argument("--fresh", action="store_true",
                        help="Force fresh-model mode even if MODEL is given "
                             "(useful for baseline comparison).")
    parser.add_argument("-d", "--data-path", default=None,
                        help="Override cfg.data_path (XYZ file). "
                             "Defaults to the model's own cfg.data_path.")
    parser.add_argument("--split", choices=["train", "test", "val", "all"],
                        default="train",
                        help="Which split to evaluate on (default: train).")
    parser.add_argument("--max-structures", type=int, default=None,
                        help="Cap the number of structures used.")
    args = parser.parse_args()

    model_path = None if args.fresh else args.model
    print(f"Loading model: {model_path or '(fresh — no checkpoint)'}")
    model = load_model_or_fresh(model_path)

    print(f"Building data ({args.split} split, "
          f"max_structures={args.max_structures})...")
    data = build_dataset_for_model(model, args.data_path,
                                    args.split, args.max_structures)

    print("Running forward through W0 → tanh ...")
    pre_h, h, Z = compute_hidden_activations(model, data)

    emit_report(pre_h, h, Z, model)

    # Second hidden layer if present.
    H2 = getattr(model, "_H2", None)
    if H2 is not None:
        print("\nNOTE: a second hidden layer (W0_2) is configured. The "
              "report above only covers the FIRST tanh. Run a follow-up "
              "by exposing h1 then computing h2 = tanh(W0_2 · h1 + b0_2).")


if __name__ == "__main__":
    main()
