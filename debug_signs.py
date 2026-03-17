"""Diagnostic utilities for debugging cos_sim = -1 sign-flipped dipole predictions.

Run after training a TNEP dipole model (target_mode=1) to identify the root cause
of anti-parallel predictions on a subset of structures.

Usage:
    from debug_signs import *
    from model_io import load_model
    from data import prepare_eval_data
    from ase.io import read

    model, cfg, type_map = load_model("test_model.npz")
    test_structures = read("train.xyz", index=":")
    test_data = prepare_eval_data(test_structures, cfg)

    diag = diagnose_sign_flips(model, test_data)
    check_cells(test_structures)
    if diag["flipped_idx"]:
        characterize_flipped(test_structures, diag["flipped_idx"], diag["good_idx"])
        test_target_negation(diag["preds"], diag["targets"], diag["flipped_idx"])
    verify_gradient_sign(model, test_structures, cfg, structure_idx=0)
"""

import numpy as np
import tensorflow as tf
from ase import Atoms
from collections import Counter

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder


def diagnose_sign_flips(model: TNEP, test_data: dict[str, tf.Tensor]) -> dict:
    """Identify structures with sign-flipped dipole predictions.

    Args:
        model     : trained TNEP model (target_mode=1)
        test_data : padded data dict from prepare_eval_data()

    Returns:
        dict with keys: flipped_idx, good_idx, cos_sim, preds, targets
    """
    metrics, preds = model.score(test_data)
    targets = test_data["targets"].numpy()
    preds = preds.numpy()
    cos_sim = metrics["cos_sim_all"].numpy()

    flipped_idx = list(np.where(cos_sim < -0.9)[0])
    good_idx = list(np.where(cos_sim > 0.9)[0])

    print(f"=== Sign Flip Diagnosis ===")
    print(f"Total structures: {len(cos_sim)}")
    print(f"Flipped (cos_sim < -0.9): {len(flipped_idx)}")
    print(f"Good    (cos_sim >  0.9): {len(good_idx)}")
    print(f"Ambiguous:                {len(cos_sim) - len(flipped_idx) - len(good_idx)}")

    # Histogram
    bins = [(-1.01, -0.9), (-0.9, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 0.9), (0.9, 1.01)]
    labels = ["< -0.9", "-0.9 to -0.5", "-0.5 to 0.0", "0.0 to 0.5", "0.5 to 0.9", "> 0.9"]
    print("\nCosine similarity distribution:")
    for (lo, hi), label in zip(bins, labels):
        count = np.sum((cos_sim >= lo) & (cos_sim < hi))
        bar = "#" * count
        print(f"  {label:>14s}: {count:4d}  {bar}")

    if flipped_idx:
        print(f"\nFlipped structure indices: {flipped_idx[:20]}"
              f"{'...' if len(flipped_idx) > 20 else ''}")
        for i in flipped_idx[:5]:
            t_mag = np.linalg.norm(targets[i])
            p_mag = np.linalg.norm(preds[i])
            print(f"  idx={i}: target |μ|={t_mag:.4f}, pred |μ|={p_mag:.4f}, "
                  f"cos_sim={cos_sim[i]:.4f}")

    return {
        "flipped_idx": flipped_idx,
        "good_idx": good_idx,
        "cos_sim": cos_sim,
        "preds": preds,
        "targets": targets,
    }


def correct_sign_flips(preds: np.ndarray, cos_sim: np.ndarray,
                       threshold: float = -0.9) -> np.ndarray:
    """Negate predictions for structures detected as sign-flipped.

    Args:
        preds     : [S, T] prediction array
        cos_sim   : [S] per-structure cosine similarity
        threshold : cos_sim below this triggers a sign flip correction

    Returns:
        corrected : [S, T] predictions with flipped structures negated
    """
    flipped_mask = cos_sim < threshold
    corrected = preds.copy()
    corrected[flipped_mask] *= -1
    n_flipped = int(np.sum(flipped_mask))
    if n_flipped > 0:
        print(f"  Corrected sign for {n_flipped} flipped predictions")
    return corrected


def check_cells(test_structures: list[Atoms]) -> None:
    """Check whether cell vectors are valid for isolated molecules (hypothesis 2).

    Flags structures with degenerate cells or atoms that could be wrapped by MIC.
    """
    print(f"\n=== Cell / MIC Check ===")
    bad_cell = []
    mic_wrap_risk = []

    for i, s in enumerate(test_structures):
        cell = s.cell[:]
        det = np.linalg.det(cell)

        if abs(det) < 1e-6:
            bad_cell.append(i)
            continue

        # Check if any atom pair distance exceeds half the shortest cell vector length
        cell_lengths = np.linalg.norm(cell, axis=1)
        min_half_cell = np.min(cell_lengths) / 2.0
        positions = s.get_positions()
        if len(positions) > 1:
            # Quick max pairwise distance check via bounding box
            extent = positions.max(axis=0) - positions.min(axis=0)
            max_extent = np.max(extent)
            if max_extent > min_half_cell:
                mic_wrap_risk.append(i)

    print(f"Structures with degenerate cell (|det| < 1e-6): {len(bad_cell)}")
    if bad_cell:
        print(f"  Indices: {bad_cell[:20]}{'...' if len(bad_cell) > 20 else ''}")
    print(f"Structures with MIC wrap risk: {len(mic_wrap_risk)}")
    if mic_wrap_risk:
        print(f"  Indices: {mic_wrap_risk[:20]}{'...' if len(mic_wrap_risk) > 20 else ''}")

    pbc_values = Counter(tuple(s.pbc) for s in test_structures)
    print(f"PBC settings across dataset: {dict(pbc_values)}")


def characterize_flipped(test_structures: list[Atoms],
                         flipped_idx: list[int],
                         good_idx: list[int]) -> None:
    """Compare properties of flipped vs good structures to find patterns."""
    print(f"\n=== Flipped vs Good Structure Comparison ===")

    def _stats(indices, label):
        if not indices:
            print(f"  {label}: no structures")
            return
        sizes = [len(test_structures[i]) for i in indices]
        compositions = Counter()
        dipole_mags = []
        for i in indices:
            s = test_structures[i]
            compositions.update(s.get_chemical_symbols())
            if "dipole" in s.info:
                dipole_mags.append(np.linalg.norm(s.info["dipole"]))

        print(f"  {label} ({len(indices)} structures):")
        print(f"    Atom count: min={min(sizes)}, max={max(sizes)}, "
              f"mean={np.mean(sizes):.1f}")
        print(f"    Element counts: {dict(compositions)}")
        if dipole_mags:
            dipole_mags = np.array(dipole_mags)
            print(f"    Target |μ|: min={dipole_mags.min():.4f}, "
                  f"max={dipole_mags.max():.4f}, mean={dipole_mags.mean():.4f}")

    _stats(flipped_idx, "Flipped")
    _stats(good_idx, "Good")

    # Check if flipped structures share a common composition
    flipped_formulas = Counter(
        test_structures[i].get_chemical_formula() for i in flipped_idx
    )
    good_formulas = Counter(
        test_structures[i].get_chemical_formula() for i in good_idx
    )
    flipped_only = set(flipped_formulas) - set(good_formulas)
    if flipped_only:
        print(f"  Formulas ONLY in flipped set: {flipped_only}")
    else:
        print(f"  No formulas exclusive to flipped set")


def test_target_negation(preds: np.ndarray, targets: np.ndarray,
                         flipped_idx: list[int]) -> None:
    """Test hypothesis 1: are flipped targets simply negated?

    Negates targets for flipped structures and recomputes metrics.
    """
    print(f"\n=== Target Negation Test (Hypothesis 1) ===")

    # Original metrics
    diff_orig = preds - targets
    rmse_orig = np.sqrt(np.mean(diff_orig ** 2))
    dot_orig = np.sum(preds * targets, axis=1)
    norm_p = np.linalg.norm(preds, axis=1)
    norm_t = np.linalg.norm(targets, axis=1)
    cos_orig = dot_orig / np.maximum(norm_p * norm_t, 1e-12)

    # Negate targets for flipped structures
    targets_fixed = targets.copy()
    targets_fixed[flipped_idx] *= -1

    diff_fixed = preds - targets_fixed
    rmse_fixed = np.sqrt(np.mean(diff_fixed ** 2))
    dot_fixed = np.sum(preds * targets_fixed, axis=1)
    norm_t_fixed = np.linalg.norm(targets_fixed, axis=1)
    cos_fixed = dot_fixed / np.maximum(norm_p * norm_t_fixed, 1e-12)

    print(f"  Before negation: RMSE={rmse_orig:.4f}, mean cos_sim={cos_orig.mean():.4f}")
    print(f"  After  negation: RMSE={rmse_fixed:.4f}, mean cos_sim={cos_fixed.mean():.4f}")
    print(f"  RMSE reduction:  {rmse_orig - rmse_fixed:.4f} "
          f"({(rmse_orig - rmse_fixed) / rmse_orig * 100:.1f}%)")

    if rmse_fixed < rmse_orig * 0.5:
        print("  >>> STRONG evidence for target sign inconsistency in the dataset <<<")
    elif rmse_fixed < rmse_orig * 0.8:
        print("  >>> Moderate evidence for target sign inconsistency <<<")
    else:
        print("  >>> Weak/no evidence for target sign inconsistency <<<")


def verify_gradient_sign(model: TNEP, test_structures: list[Atoms],
                         cfg: TNEPconfig, structure_idx: int = 0,
                         epsilon: float = 1e-4) -> None:
    """Test hypothesis 3: does quippy's gradient sign match finite differences?

    Computes numerical gradient via central differences and compares against
    the analytical gradient from quippy's grad_data.

    Args:
        model          : trained TNEP model
        test_structures: list of structures
        cfg            : TNEPconfig
        structure_idx  : which structure to test
        epsilon        : finite difference step size
    """
    print(f"\n=== Gradient Sign Verification (Hypothesis 3) ===")
    print(f"  Structure index: {structure_idx}")
    print(f"  Epsilon: {epsilon}")

    structure = test_structures[structure_idx].copy()
    builder = DescriptorBuilder(cfg)

    # Analytical gradients from quippy
    descs, grads, grad_idx = builder.build_descriptors([structure])
    analytical_grad = grads[0]  # list of N tensors

    # Pick first atom that has neighbors
    test_atom = None
    for atom_i in range(len(structure)):
        if len(grad_idx[0][atom_i]) > 0:
            test_atom = atom_i
            break

    if test_atom is None:
        print("  No atom with neighbors found — cannot verify gradients.")
        return

    neighbor_local_idx = 0  # first neighbor of test_atom
    neighbor_atom = grad_idx[0][test_atom][neighbor_local_idx]
    cart_component = 0  # x-component
    desc_component = 0  # first descriptor component

    analytical_val = analytical_grad[test_atom][neighbor_local_idx, cart_component, desc_component].numpy()

    # Numerical gradient: d(descriptor of test_atom) / d(position of neighbor_atom, cart_component)
    positions = structure.get_positions().copy()

    # Forward
    pos_fwd = positions.copy()
    pos_fwd[neighbor_atom, cart_component] += epsilon
    s_fwd = structure.copy()
    s_fwd.set_positions(pos_fwd)
    descs_fwd, _, _ = builder.build_descriptors([s_fwd])
    q_fwd = descs_fwd[0][test_atom, desc_component].numpy()

    # Backward
    pos_bwd = positions.copy()
    pos_bwd[neighbor_atom, cart_component] -= epsilon
    s_bwd = structure.copy()
    s_bwd.set_positions(pos_bwd)
    descs_bwd, _, _ = builder.build_descriptors([s_bwd])
    q_bwd = descs_bwd[0][test_atom, desc_component].numpy()

    numerical_val = (q_fwd - q_bwd) / (2 * epsilon)

    sign_match = np.sign(analytical_val) == np.sign(numerical_val)
    ratio = analytical_val / numerical_val if abs(numerical_val) > 1e-12 else float('inf')

    print(f"  Test atom: {test_atom}, neighbor atom: {neighbor_atom}")
    print(f"  Cartesian component: {cart_component}, descriptor component: {desc_component}")
    print(f"  Analytical gradient:  {analytical_val:.6e}")
    print(f"  Numerical gradient:   {numerical_val:.6e}")
    print(f"  Ratio (analytical/numerical): {ratio:.4f}")
    print(f"  Sign match: {sign_match}")

    if not sign_match:
        print("  >>> SIGN MISMATCH: quippy grad_data has opposite sign to finite diff <<<")
        print("  >>> This would cause dipole sign flips for affected structures <<<")
    elif abs(ratio - 1.0) > 0.1:
        print(f"  >>> WARNING: magnitude mismatch (ratio={ratio:.4f}), "
              f"but signs agree <<<")
    else:
        print("  >>> Gradient sign and magnitude are consistent <<<")
