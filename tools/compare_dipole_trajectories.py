"""Compare two dipole trajectory files frame-by-frame.

Reads two text files where each line is `x y z` (3 cols, plain) OR
`frame_idx x y z` (4 cols, GPUMD `dipole.out` style — first column dropped).
The format is auto-detected per file from the column count.

Unit conversion: each file's native unit is named via UNITS_A / UNITS_B
(one of {"e*angstrom", "e*bohr", "debye"}); both are converted to e·Å.

Outputs:
    - summary statistics (per-component + overall RMSE, R², RRMSE, cos-sim)
    - absolute-error plot |Δμ(t)| with per-component lines + total |Δμ|

────────────────────────────────────────────────────────────────────
EDIT THE PARAMETERS BLOCK BELOW, THEN RUN:
    python tools/compare_dipole_trajectories.py
────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os

import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse TNEP's authoritative conversion factor so unit semantics agree
# bit-for-bit with what training/inference does.
from data import _dipole_conversion_factor


# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS — edit these for your run
# ═══════════════════════════════════════════════════════════════════

FILE_A = "plots/BulkWater/water_bulk_traj_dipoles.txt"
FILE_B = "datasets/dipole.out"            # e.g. GPUMD dipole.out

UNITS_A = "e*angstrom"   # one of: "e*angstrom", "e*bohr", "debye"
UNITS_B = "e*bohr"       # one of: "e*angstrom", "e*bohr", "debye"

# If set (e.g. 0.5 or 1.0), the plot x-axis is time in ps (frame_idx × dt_fs/1000).
# If None, x-axis is the integer frame index.
DT_FS: float | None = 1.0

# Where to save the error plot (None = don't save). The directory is auto-created.
SAVE_PLOT: str | None = "plots/BulkWater/dipole_error.png"

# True = interactive plt.show() at the end (set False for headless runs).
SHOW_PLOT = False

# ═══════════════════════════════════════════════════════════════════


def load_dipoles(path: str) -> np.ndarray:
    """Load dipole trajectory. Auto-detects 3-col or 4-col (GPUMD) layout.

    Returns: ndarray of shape [T, 3]. Comment lines starting with '#' are
    skipped (np.loadtxt default behaviour).
    """
    if not os.path.isabs(path):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 3:
        return arr, path
    if arr.shape[1] == 4:
        return arr[:, 1:4], path
    raise ValueError(
        f"{path}: expected 3 or 4 columns per line (got {arr.shape[1]}). "
        f"3-col layout: x y z. 4-col GPUMD layout: frame_idx x y z.")


def _print_block(label: str, pred: np.ndarray, ref: np.ndarray) -> None:
    """Per-component + overall RMSE / R² / RRMSE / cos-sim summary."""
    diff = pred - ref
    rmse_pc = np.sqrt(np.mean(diff**2, axis=0))
    rmse_all = float(np.sqrt(np.mean(diff**2)))

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
    std_pc  = ref.std(axis=0)
    std_all = float(ref.std())
    rrmse_pc  = rmse_pc / np.maximum(std_pc, 1e-30)
    rrmse_all = float(rmse_all / max(std_all, 1e-30))

    norms_p = np.linalg.norm(pred, axis=1)
    norms_r = np.linalg.norm(ref,  axis=1)
    cos_sim = np.sum(pred * ref, axis=1) / np.maximum(norms_p * norms_r, 1e-12)

    print(f"\n  {label}")
    print(f"               {'x':>14} {'y':>14} {'z':>14} {'overall':>14}")
    print(f"  RMSE       : {rmse_pc[0]:>14.6e} {rmse_pc[1]:>14.6e} "
          f"{rmse_pc[2]:>14.6e} {rmse_all:>14.6e}")
    print(f"  R²         : {r2_pc[0]:>14.6f} {r2_pc[1]:>14.6f} "
          f"{r2_pc[2]:>14.6f} {r2_all:>14.6f}")
    print(f"  RRMSE      : {rrmse_pc[0]:>14.6f} {rrmse_pc[1]:>14.6f} "
          f"{rrmse_pc[2]:>14.6f} {rrmse_all:>14.6f}")
    print(f"  Cosine sim : mean={cos_sim.mean():.6f}  "
          f"median={np.median(cos_sim):.6f}  worst={cos_sim.min():.6f}")
    print(f"  |Δμ|       : mean={np.linalg.norm(diff, axis=1).mean():.6e}  "
          f"max={np.linalg.norm(diff, axis=1).max():.6e}")


def plot_error(file_a: str, file_b: str,
               dip_a: np.ndarray, dip_b: np.ndarray,
               dt_fs: float | None,
               save_plot: str | None, show: bool) -> None:
    """Plot |Δμ_x|, |Δμ_y|, |Δμ_z|, and total |Δμ| vs frame index (or time)."""
    diff = dip_a - dip_b
    T = diff.shape[0]
    if dt_fs is not None:
        t = np.arange(T) * dt_fs / 1000.0   # ps
        xlabel = "Time (ps)"
    else:
        t = np.arange(T)
        xlabel = "Frame index"

    abs_err = np.abs(diff)
    vec_err = np.linalg.norm(diff, axis=1)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    for c, label, color in zip(
            range(3), ("|Δμ_x|", "|Δμ_y|", "|Δμ_z|"),
            ("#e74c3c", "#2ecc71", "#3498db")):
        ax_top.plot(t, abs_err[:, c], color=color, linewidth=0.6,
                    alpha=0.85, label=label)
    ax_top.set_ylabel("|Δμ_component| (e·Å)")
    ax_top.set_title(f"Absolute dipole error per frame  "
                     f"({os.path.basename(file_a)}  vs  {os.path.basename(file_b)})")
    ax_top.legend(loc="upper right")
    ax_top.grid(alpha=0.3)

    ax_bot.plot(t, vec_err, color="black", linewidth=0.7, label="|Δμ| (vector)")
    ax_bot.axhline(vec_err.mean(), color="grey", linestyle="--", linewidth=0.8,
                   label=f"mean = {vec_err.mean():.3e}")
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel("|Δμ| (e·Å)")
    ax_bot.legend(loc="upper right")
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    if save_plot is not None:
        # Resolve relative paths against project root, not cwd
        if not os.path.isabs(save_plot):
            save_plot = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                save_plot)
        os.makedirs(os.path.dirname(save_plot) or ".", exist_ok=True)
        fig.savefig(save_plot, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {save_plot}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def compare(file_a: str, file_b: str,
            units_a: str, units_b: str,
            dt_fs: float | None = None,
            save_plot: str | None = None,
            show: bool = True) -> None:
    """Run the full comparison. Kept as a callable so other scripts can
    import this module and invoke programmatically without going through
    the parameters block."""
    dip_a, full_a = load_dipoles(file_a)
    dip_b, full_b = load_dipoles(file_b)
    dip_a = dip_a * _dipole_conversion_factor(units_a)
    dip_b = dip_b * _dipole_conversion_factor(units_b)

    print(f"File A : {full_a}  ({units_a})  shape={dip_a.shape}")
    print(f"File B : {full_b}  ({units_b})  shape={dip_b.shape}")

    if dip_a.shape[0] != dip_b.shape[0]:
        n = min(dip_a.shape[0], dip_b.shape[0])
        print(f"  Length mismatch — truncating both to first {n} frames")
        dip_a = dip_a[:n]
        dip_b = dip_b[:n]

    _print_block("DIPOLE COMPARISON  (units: e·Å, all converted)", dip_a, dip_b)
    plot_error(full_a, full_b, dip_a, dip_b,
               dt_fs=dt_fs, save_plot=save_plot, show=show)


if __name__ == "__main__":
    compare(file_a="plots/BulkWater/water_bulk_traj_dipoles.txt",
            file_b="plots/BulkWater/dipole_gpumd_out.out",
            units_a="e*angstrom",
            units_b="e*bohr",
            dt_fs=1,
            save_plot=SAVE_PLOT,
            show=SHOW_PLOT)
