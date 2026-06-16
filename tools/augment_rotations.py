"""Rotation-augmented dataset extension for TNEP training.

CAVEAT — read this before using:

The TNEP architecture is ALREADY rotation-equivariant by construction.
SOAP-turbo descriptors are rotation-invariant, the descriptor's
gradient ∂q/∂r is rotation-equivariant (covariant), and the
dipole formula

    μ = -Σ_{i,j} |r_ij|² · (∂U_i / ∂r_ij)

is exactly equivariant when r → R·r (the rotated positions produce
a rotated dipole). In the limit of perfect numerical precision and
no train-time noise, rotation augmentation adds zero information —
each augmented copy produces the same loss as the original after
counter-rotating the prediction.

When rotation augmentation MIGHT still help in practice:

  1. fp32 round-off in the descriptor build and gradient computation
     produces tiny per-orientation asymmetries (~ 1e-6 to 1e-5
     relative). With many training generations and a flexible model,
     these can be "memorised" — i.e. the model learns to exploit
     orientation-specific numerical artefacts of the training-set
     orientations. Rotational copies expose this and force the
     model to ignore the artefacts.

  2. Periodic boundary conditions: the minimum-image convention
     and lattice rotations interact non-trivially with the descriptor
     cutoff in fp32. Rotating cell vectors (which this script does
     NOT do — we only rotate the atomic coordinates within a fixed
     cell) is the part that risks breaking. We deliberately keep
     positions+dipole rotated under the SAME R within a fixed
     unit cell, exploiting MIC symmetry.

  3. As a soft regulariser: training on `N · K` samples instead of
     `N` slows SNES convergence per generation but every generation's
     update is smoother (more orientations per sigma update). Net
     effect on training time: usually slightly slower; net effect
     on final RMSE: small.

  4. As a test-set sanity check: scoring on rotational copies and
     measuring the spread in predicted-dipole magnitude is a direct
     proxy for "how exactly is this model equivariant in fp32?".
     This is the genuinely useful application — measure, don't train.

Recommendation: don't use this for training augmentation unless you
have evidence that the model is overfitting to orientation-specific
descriptor artefacts. DO use it as a post-training diagnostic
(scoring the same molecule at K orientations and looking at the
RMS dipole-magnitude variation across orientations) — that gives a
clean number for "how close to exactly equivariant am I?".
"""

from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ase import Atoms


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly-random rotation matrix from SO(3) using the
    QR decomposition of a random Gaussian matrix (Stewart 1980).
    """
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure det(Q) = +1 (pure rotation, not reflection)
    sign = np.sign(np.diag(R))
    Q = Q * sign[np.newaxis, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def rotate_structure(atoms: Atoms, R: np.ndarray) -> Atoms:
    """Return a rotated copy of `atoms`.

    Positions and lattice vectors are rotated by R; the rotated dipole
    target (if present in `atoms.info`) is also updated by R. Atomic
    numbers and PBC flags are preserved.

    Args:
        atoms : input structure.
        R     : [3, 3] rotation matrix (uniform SO(3) recommended).

    Returns:
        A new ASE Atoms object with rotated coordinates, cell, and
        dipole. The descriptor of the rotated structure will be
        identical (to numerical precision) to the original's; only
        the predicted/target dipole vector rotates.
    """
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ R.T
    if atoms.cell.any():
        rotated.cell = atoms.cell.array @ R.T
    if "dipole" in atoms.info:
        mu = np.asarray(atoms.info["dipole"], dtype=np.float64)
        rotated.info["dipole"] = R @ mu
    if "pol" in atoms.info:
        # Polarizability is a rank-2 tensor: α' = R α Rᵀ
        # Stored as [xx, yy, zz, xy, yz, zx] in TNEP convention.
        pol_flat = np.asarray(atoms.info["pol"], dtype=np.float64)
        alpha = np.zeros((3, 3))
        alpha[0, 0], alpha[1, 1], alpha[2, 2] = pol_flat[0], pol_flat[1], pol_flat[2]
        alpha[0, 1] = alpha[1, 0] = pol_flat[3]
        alpha[1, 2] = alpha[2, 1] = pol_flat[4]
        alpha[2, 0] = alpha[0, 2] = pol_flat[5]
        alpha_rot = R @ alpha @ R.T
        rotated.info["pol"] = np.array([
            alpha_rot[0, 0], alpha_rot[1, 1], alpha_rot[2, 2],
            alpha_rot[0, 1], alpha_rot[1, 2], alpha_rot[2, 0],
        ])
    return rotated


def augment_dataset(structures: list[Atoms],
                    n_rotations: int,
                    seed: int | None = None,
                    include_original: bool = True) -> list[Atoms]:
    """Expand a list of structures by adding `n_rotations` random
    rotations of each.

    Args:
        structures      : input list of Atoms.
        n_rotations     : number of random rotations to add per
                          structure (0 → no augmentation).
        seed            : RNG seed for reproducibility.
        include_original: when True (default), the original structure
                          is kept alongside the rotations. When False,
                          replaces each original with one rotated copy
                          + (n_rotations-1) additional rotations.

    Returns:
        Augmented list of length:
            len(structures) · (n_rotations + 1) if include_original
            len(structures) · n_rotations       otherwise
    """
    rng = np.random.default_rng(seed)
    out: list[Atoms] = []
    for atoms in structures:
        if include_original:
            out.append(atoms)
        for _ in range(n_rotations):
            R = random_rotation_matrix(rng)
            out.append(rotate_structure(atoms, R))
    return out


def measure_equivariance(model, atoms: Atoms,
                          n_rotations: int = 20,
                          seed: int | None = None) -> dict:
    """Diagnostic: score the same structure at `n_rotations` random
    orientations and report how exactly the model is equivariant.

    For a perfectly equivariant model, `R^T · μ_predicted(R · atoms)`
    should be a single fixed vector independent of R. The std across
    rotations measures fp32 / numerical equivariance violation.

    Args:
        model       : trained TNEP model with `.score()` method.
        atoms       : single structure to test.
        n_rotations : number of random orientations to sample.
        seed        : RNG seed.

    Returns:
        dict with keys:
            mean_dipole       : [3] mean predicted dipole (rotated back).
            std_dipole_norm   : float — std of |μ_back| across orientations.
            relative_spread   : std / mean magnitude — dimensionless.
            all_back_rotated  : [n_rotations, 3] all rotated-back predictions.
    """
    from data import prepare_eval_data
    rng = np.random.default_rng(seed)
    backed: list[np.ndarray] = []
    for _ in range(n_rotations):
        R = random_rotation_matrix(rng)
        rotated = rotate_structure(atoms, R)
        # prepare_eval_data threads q_scaler + target_mean from cfg so
        # the eval dict matches the trained model's input space.
        data = prepare_eval_data([rotated], model.cfg)
        _, preds = model.score(data)
        mu_pred = preds[0].numpy()
        # Rotate prediction back to the original frame
        backed.append(R.T @ mu_pred)
    backed_arr = np.asarray(backed)
    mean_dip = backed_arr.mean(axis=0)
    std_norm = np.linalg.norm(backed_arr - mean_dip[np.newaxis, :], axis=1).std()
    rel = std_norm / max(np.linalg.norm(mean_dip), 1e-12)
    return {
        "mean_dipole": mean_dip,
        "std_dipole_norm": float(std_norm),
        "relative_spread": float(rel),
        "all_back_rotated": backed_arr,
    }
