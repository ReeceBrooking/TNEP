"""RMSE / R² / RRMSE between predicted and reference dipoles.

File layout (whitespace-separated, one structure per line):
    cols 1-3 : predicted per-atom dipole (x, y, z)
    cols 4-6 : reference per-atom dipole (x, y, z)
    col  7   : number of atoms in the structure  (optional; if absent,
               only per-atom statistics are reported)

When col 7 is present, the script also reports TOTAL-dipole statistics
(per_atom × N_atoms), which is the directly comparable quantity for
literature benchmarks (Xu et al. JCTC 2024 etc.).

Run from anywhere — paths resolve from the project root.
"""
from __future__ import annotations

import os
import sys
import numpy as np


def _block_stats(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Compute per-component + overall RMSE, R², RRMSE for paired vectors."""
    diff = pred - ref

    rmse_pc = np.sqrt(np.mean(diff**2, axis=0))               # [3]
    rmse_all = float(np.sqrt(np.mean(diff**2)))                # scalar
    rmse_vec = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

    ref_mu_pc = ref.mean(axis=0)
    ss_res_pc = np.sum(diff**2, axis=0)
    ss_tot_pc = np.sum((ref - ref_mu_pc)**2, axis=0)
    r2_pc     = 1.0 - ss_res_pc / np.maximum(ss_tot_pc, 1e-30)

    ref_mu_all = ref.mean()
    ss_res_all = float(np.sum(diff**2))
    ss_tot_all = float(np.sum((ref - ref_mu_all)**2))
    r2_all     = 1.0 - ss_res_all / max(ss_tot_all, 1e-30)

    # RRMSE = RMSE / std(ref). Per-component uses per-component std;
    # overall uses the std of the flattened reference. Algebraically
    # equivalent to sqrt(1 - R²) for the same partition.
    std_pc  = ref.std(axis=0)                     # [3]
    std_all = float(ref.std())                    # scalar (over all 3·N values)
    rrmse_pc  = rmse_pc / np.maximum(std_pc, 1e-30)
    rrmse_all = float(rmse_all / max(std_all, 1e-30))

    norms_p = np.linalg.norm(pred, axis=1)
    norms_r = np.linalg.norm(ref,  axis=1)
    cos_sim = np.sum(pred * ref, axis=1) / np.maximum(norms_p * norms_r, 1e-12)

    return {
        "rmse_pc": rmse_pc, "rmse_all": rmse_all, "rmse_vec": rmse_vec,
        "r2_pc": r2_pc, "r2_all": r2_all,
        "rrmse_pc": rrmse_pc, "rrmse_all": rrmse_all,
        "cos_sim": cos_sim,
    }


def _print_block(label: str, s: dict) -> None:
    print(f"\n  {label}")
    print(f"               {'x':>14} {'y':>14} {'z':>14} {'overall':>14}")
    print(f"  RMSE       : {s['rmse_pc'][0]:>14.6e} {s['rmse_pc'][1]:>14.6e} "
          f"{s['rmse_pc'][2]:>14.6e} {s['rmse_all']:>14.6e}")
    print(f"  R²         : {s['r2_pc'][0]:>14.6f} {s['r2_pc'][1]:>14.6f} "
          f"{s['r2_pc'][2]:>14.6f} {s['r2_all']:>14.6f}")
    print(f"  RRMSE      : {s['rrmse_pc'][0]:>14.6f} {s['rrmse_pc'][1]:>14.6f} "
          f"{s['rrmse_pc'][2]:>14.6f} {s['rrmse_all']:>14.6f}")
    cs = s["cos_sim"]
    print(f"  RMSE vector: {s['rmse_vec']:.6e}    (sqrt of mean |Δμ|²)")
    print(f"  Cosine sim : mean={cs.mean():.6f}  "
          f"median={np.median(cs):.6f}  worst={cs.min():.6f}")


def main(path: str = "datasets/dipole_test.out") -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = path if os.path.isabs(path) else os.path.join(root, path)

    data = np.loadtxt(full)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(
            f"{full}: expected ≥6 columns (pred_xyz | ref_xyz [| N_atoms]), "
            f"got shape {data.shape}")

    pred = data[:, 0:3]
    ref  = data[:, 3:6]
    has_natoms = data.shape[1] >= 7
    natoms = data[:, 6].astype(np.int64) if has_natoms else None

    print(f"File         : {full}")
    print(f"Structures   : {data.shape[0]}")
    if has_natoms:
        print(f"Atom counts  : min={natoms.min()}  max={natoms.max()}  "
              f"mean={natoms.mean():.2f}  total={natoms.sum()}")

    # ── Per-atom block ────────────────────────────────────────────────
    per_atom = _block_stats(pred, ref)
    _print_block("PER-ATOM DIPOLE  (units: as in file)", per_atom)

    # ── Total block (only if N_atoms is available) ────────────────────
    if has_natoms:
        n_col = natoms[:, None].astype(np.float64)
        pred_total = pred * n_col
        ref_total  = ref  * n_col
        total = _block_stats(pred_total, ref_total)
        _print_block("TOTAL DIPOLE     (per-atom × N_atoms)", total)
    else:
        print("\n  [N_atoms column absent — skipping total-dipole stats.]")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "datasets/dipole_test_tnep.out"
    main(p)
