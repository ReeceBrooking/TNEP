from __future__ import annotations

import os

import numpy as np

# Headless-safe backend selection. On HPC compute nodes there's no
# DISPLAY, and matplotlib's default tk/qt backends will fail at import
# time when no GUI is available. Selecting Agg up front keeps the
# pipeline working in batch jobs while still letting interactive use
# (where DISPLAY is set or MPLBACKEND is user-specified) get the
# default backend.
if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

from TNEPconfig import TNEPconfig
from data import component_labels


def unit_label(cfg: TNEPconfig) -> str:
    """Return the unit string for the current target mode.

    Dipole label follows the chosen training space:
      - `cfg.convert_dipole_to_eangstrom=True` (default) → "e·Å"
      - `False` → the dataset's native unit derived from
        `cfg.dipole_units` ("e·Å", "e·a₀", or "Debye").

    Args:
        cfg : TNEPconfig

    Returns:
        str — "eV" (PES), dipole unit (mode 1), or "Å³" (polarisability).
    """
    if cfg.target_mode == 0:
        return "eV"
    if cfg.target_mode == 2:
        return "\u00c5\u00b3"
    if cfg.target_mode == 1:
        if getattr(cfg, "convert_dipole_to_eangstrom", True):
            return "e\u00b7\u00c5"
        # Dataset-native dipole label when conversion is disabled.
        return {
            "e*angstrom": "e\u00b7\u00c5",
            "e*bohr":     "e\u00b7a\u2080",   # e\u00b7a\u2080
            "debye":      "Debye",
        }.get(getattr(cfg, "dipole_units", "e*angstrom"), "")
    return ""


def _make_plot_filename(cfg: TNEPconfig, plot_name: str) -> str:
    """Generate an automatic filename for a plot based on config and plot name.

    Format (GAP): {plot_name}_{mode}_M{n_sparse}_z{zeta}_l{l_max}.png

    Args:
        cfg       : TNEPconfig
        plot_name : str — short identifier for the plot type

    Returns:
        filename : str — e.g. "correlation_dipole_M1000_z4_l4.png"
    """
    mode_names = {0: "pes", 1: "dipole", 2: "polar"}
    mode = mode_names.get(cfg.target_mode, f"mode{cfg.target_mode}")
    return (f"{plot_name}_{mode}_M{cfg.gap_n_sparse}_z{cfg.gap_zeta}"
            f"_l{cfg.l_max}.png")


def _save_fig(fig: plt.Figure, cfg: TNEPconfig, plot_name: str, save_dir: str | None) -> None:
    """Save a figure to save_dir if set.

    Args:
        fig       : matplotlib Figure
        cfg       : TNEPconfig — used for filename generation
        plot_name : str — short identifier for the plot type
        save_dir  : str or None — directory to save into (None = don't save)
    """
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    filename = _make_plot_filename(cfg, plot_name)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {path}")


def _finish_fig(fig: plt.Figure, cfg: TNEPconfig, plot_name: str,
                save_plots: str | None, show_plots: bool) -> None:
    """Save, show, and/or close a figure.

    Args:
        fig        : matplotlib Figure
        cfg        : TNEPconfig — used for filename generation
        plot_name  : str — short identifier for the plot type
        save_plots : str or None — directory to save into (None = don't save)
        show_plots : bool — True to display interactively
    """
    _save_fig(fig, cfg, plot_name, save_plots)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


def _build_suptitle(cfg: TNEPconfig, suffix: str | None,
                    rmse: float, rrmse: float, r2: float) -> str:
    """Build a suptitle string with mode, suffix label, and metrics.

    Callers in MasterTNEP._plot_eval_set pass suffixes of the form
    `{train,val,test}_{per_atom,total}` or just `{per_atom,total}`.
    """
    mode_names = {0: "PES", 1: "Dipole", 2: "Polarizability"}
    mode = mode_names.get(cfg.target_mode, f"Mode {cfg.target_mode}")
    if suffix:
        parts = []
        if "train" in suffix:
            parts.append("Train")
        elif "val" in suffix:
            parts.append("Validation")
        elif "test" in suffix:
            parts.append("Test")
        if "per_atom" in suffix:
            parts.append("Per Atom")
        elif "total" in suffix:
            parts.append("Total")
        label = f" ({' '.join(parts)})" if parts else f" ({suffix})"
    else:
        label = ""
    units = unit_label(cfg)
    return f"{mode}{label} \u2014 RMSE: {rmse:.4f} {units}, RRMSE: {rrmse:.4f}, R\u00b2: {r2:.4f}"


def plot_correlation(targets: np.ndarray, predictions: np.ndarray, metrics: dict,
                     cfg: TNEPconfig, save_plots: str | None = None,
                     show_plots: bool = True, suffix: str | None = None) -> None:
    """Plot target vs prediction correlation with per-component R².

    For scalar targets (PES), plots a single correlation panel.
    For vector targets (dipole [3] or polarizability [6]), plots one correlation
    panel per component.

    Args:
        targets     : [S, T] numpy array of target values
        predictions : [S, T] numpy array of predicted values
        metrics     : dict from TNEP.score() with r2, rmse, r2_components
        cfg         : TNEPconfig — used to determine target mode and labels
        save_plots  : str or None — directory to save into
        show_plots  : bool — True to display interactively
    """
    T = targets.shape[1]
    rmse = float(metrics["rmse"])
    r2 = float(metrics["r2"])
    r2_comp = metrics["r2_components"].numpy()
    # RRMSE is provided pre-computed by the caller (always derived
    # from total-scale RMSE / total-scale mean-|target|) so the same
    # value appears on both per-atom and total panels — RRMSE is a
    # model-vs-dataset property, not a scale-dependent quantity.
    rrmse = float(metrics["rrmse"])
    rrmse_comp = np.asarray(metrics["rrmse_components"])

    labels = component_labels(cfg.target_mode, T)
    units = unit_label(cfg)

    ncols = min(T, 3)
    nrows = (T + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for i in range(T):
        ax = axes[i // ncols, i % ncols]
        t = targets[:, i]
        p = predictions[:, i]

        ax.scatter(t, p, s=10, alpha=0.6)

        # x = y reference line
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        margin = 0.05 * (hi - lo) if hi > lo else 0.5
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=1, label="x = y")

        ax.set_xlabel(f"Target {labels[i]} ({units})")
        ax.set_ylabel(f"Prediction {labels[i]} ({units})")
        ax.set_title(f"{labels[i]}  R²={r2_comp[i]:.4f}  RRMSE={rrmse_comp[i]:.4f}")
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left')

    # Hide unused subplots
    total_slots = nrows * ncols
    for i in range(T, total_slots):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.suptitle(_build_suptitle(cfg, suffix, rmse, rrmse, r2), fontsize=14)
    plt.tight_layout()
    plot_name = f"correlation_{suffix}" if suffix else "correlation"
    _finish_fig(fig, cfg, plot_name, save_plots, show_plots)


def plot_cosine_similarity(metrics: dict, cfg: TNEPconfig,
                           save_plots: str | None = None,
                           show_plots: bool = True,
                           suffix: str | None = None) -> None:
    """Plot cosine similarity histogram for vector targets.

    Args:
        metrics    : dict from TNEP.score() with cos_sim_all, cos_sim_mean
        cfg        : TNEPconfig
        save_plots : str or None — directory to save into
        show_plots : bool — True to display interactively
        suffix     : str or None — appended to filename
    """
    if "cos_sim_all" not in metrics:
        return

    cos_all = metrics["cos_sim_all"].numpy()
    cos_mean = float(metrics["cos_sim_mean"])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(cos_all, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    ax.axvline(cos_mean, color='red', linestyle='--', linewidth=1.5,
               label=f"Mean = {cos_mean:.4f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_xlim(-1.05, 1.05)
    ax.set_title(f"Cosine similarity distribution  "
                 f"(mean={cos_mean:.4f}, std={cos_all.std():.4f})")
    ax.legend()
    plt.tight_layout()
    plot_name = f"cosine_similarity_{suffix}" if suffix else "cosine_similarity"
    _finish_fig(fig, cfg, plot_name, save_plots, show_plots)


def plot_error_vs_magnitude(targets: np.ndarray, predictions: np.ndarray,
                            cfg: TNEPconfig, save_plots: str | None = None,
                            show_plots: bool = True, suffix: str | None = None) -> None:
    """Plot per-structure absolute error vs target magnitude.

    For vector targets (dipole/polarizability), uses the norm of the full vector.
    Helps diagnose whether errors are proportional to target scale (capacity-limited)
    or constant (accuracy floor from descriptors/data).

    Args:
        targets     : [S, T] numpy array of target values
        predictions : [S, T] numpy array of predicted values
        cfg         : TNEPconfig
        save_plots  : str or None — directory to save into
        show_plots  : bool — True to display interactively
        suffix      : str or None — appended to filename
    """
    T = targets.shape[1]

    if T == 1:
        tgt_mag = np.abs(targets[:, 0])
        abs_err = np.abs(targets[:, 0] - predictions[:, 0])
    else:
        tgt_mag = np.linalg.norm(targets, axis=1)
        abs_err = np.linalg.norm(targets - predictions, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(tgt_mag, abs_err, s=10, alpha=0.5)

    # Fit a linear trend line for reference
    if tgt_mag.max() > 0:
        slope, intercept = np.polyfit(tgt_mag, abs_err, 1)
        x_fit = np.linspace(0, tgt_mag.max(), 100)
        ax.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=1.5,
                label=f"fit: {slope:.4f}x + {intercept:.4f}")

    units = unit_label(cfg)
    ax.set_xlabel(f"Target magnitude ({units})")
    ax.set_ylabel(f"Absolute error ({units})")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    mode_names = {0: "PES", 1: "Dipole", 2: "Polarizability"}
    mode = mode_names.get(cfg.target_mode, f"Mode {cfg.target_mode}")
    label = f" ({suffix})" if suffix else ""
    ax.set_title(f"{mode}{label} — Absolute error vs target magnitude")

    plt.tight_layout()
    plot_name = f"error_vs_mag_{suffix}" if suffix else "error_vs_mag"
    _finish_fig(fig, cfg, plot_name, save_plots, show_plots)
