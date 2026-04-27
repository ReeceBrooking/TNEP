from __future__ import annotations

import re
import numpy as np
import matplotlib.pyplot as plt
import os

from TNEPconfig import TNEPconfig
from data import component_labels


def unit_label(cfg: TNEPconfig) -> str:
    """Return the unit string for the current target mode.

    Dipole targets are always in e·Å after conversion in the data pipeline.

    Args:
        cfg : TNEPconfig

    Returns:
        str — "eV", "e·Å", or "ų"
    """
    return {0: "eV", 1: "e\u00b7\u00c5", 2: "\u00c5\u00b3"}.get(cfg.target_mode, "")


def _make_plot_filename(cfg: TNEPconfig, plot_name: str) -> str:
    """Generate an automatic filename for a plot based on config and plot name.

    Format: {plot_name}_mode{target_mode}_pop{pop_size}_n{num_neurons}_l{l_max}.png

    Args:
        cfg       : TNEPconfig
        plot_name : str — short identifier for the plot type

    Returns:
        filename : str — e.g. "snes_fitness_mode1_pop80_n30_l4.png"
    """
    mode_names = {0: "pes", 1: "dipole", 2: "polar"}
    mode = mode_names.get(cfg.target_mode, f"mode{cfg.target_mode}")
    return (f"{plot_name}_{mode}_pop{cfg.pop_size}_n{cfg.num_neurons}"
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


def plot_snes_history(history: dict, cfg: TNEPconfig,
                      save_plots: str | None = None, show_plots: bool = True) -> None:
    """Plot train and validation RMSE vs generation with best/worst band."""
    g = np.asarray(history["generation"])

    fig = plt.figure()
    plt.plot(g, history["train_loss"], label="Train RMSE")
    plt.plot(g, history["val_loss"], label="Val RMSE")

    if history.get("best_rmse") and history.get("worst_rmse"):
        plt.fill_between(g, history["best_rmse"], history["worst_rmse"],
                         alpha=0.2, label="Best\u2013Worst range")

    units = unit_label(cfg)
    plt.xlabel("Generation")
    plt.ylabel(f"Fitness RMSE ({units})")
    plt.legend()
    plt.title("SNES fitness vs generation")
    _finish_fig(fig, cfg, "snes_fitness", save_plots, show_plots)


def plot_log_val_fitness(history: dict, cfg: TNEPconfig,
                         save_plots: str | None = None, show_plots: bool = True) -> None:
    """Plot natural log of validation fitness vs generation."""
    g = np.asarray(history["generation"]) + 1  # shift so gen 0 → 1 (avoids log(0))
    val = np.asarray(history["val_loss"])
    ln_val = np.log(val)
    ln_g = np.log(g)

    fig = plt.figure()
    plt.plot(ln_g, ln_val, label="ln(Val RMSE)")
    units = unit_label(cfg)
    plt.xlabel("ln(Generation)")
    plt.ylabel(f"ln(Validation RMSE / {units})")
    plt.legend()
    plt.title("Log validation fitness vs generation")
    _finish_fig(fig, cfg, "log_val_fitness", save_plots, show_plots)


def plot_sigma_history(history: dict, cfg: TNEPconfig,
                       save_plots: str | None = None, show_plots: bool = True) -> None:
    """Plot sigma min/max/mean vs generation on log-y scale with reset markers."""
    g = np.asarray(history["generation"])
    if not history.get("sigma_mean"):
        return

    fig = plt.figure()
    plt.plot(g, history["sigma_mean"], label="Sigma mean")
    plt.plot(g, history["sigma_median"], label="Sigma median", linestyle="--")
    plt.plot(g, history["sigma_min"], label="Sigma min", alpha=0.6)
    plt.plot(g, history["sigma_max"], label="Sigma max", alpha=0.6)
    plt.fill_between(g, history["sigma_min"], history["sigma_max"],
                     alpha=0.15, color="blue")

    plt.xlabel("Generation")
    plt.ylabel("\u03c3")
    plt.yscale("log")
    plt.legend()
    plt.title("SNES sigma evolution")
    _finish_fig(fig, cfg, "sigma_evolution", save_plots, show_plots)


def plot_loss_breakdown(history: dict, cfg: TNEPconfig,
                        save_plots: str | None = None,
                        show_plots: bool = True) -> None:
    """Plot total loss with L1, L2 regularization and mean train RMSE on log-x scale."""
    g = np.asarray(history["generation"], dtype=np.float64) + 1  # shift so gen 0 → 1 (avoids log(0))
    train = np.asarray(history["train_loss"])
    val = np.asarray(history["val_loss"])
    l1 = np.maximum(np.asarray(history["L1"]), 0.0)
    l2 = np.maximum(np.asarray(history["L2"]), 0.0)
    total = train + l1 + l2

    units = unit_label(cfg)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(g, train, label=f"Train RMSE ({units})", color="#e74c3c", linewidth=1, alpha=0.8)
    ax.plot(g, val, label=f"Val RMSE ({units})", color="#9b59b6", linewidth=1, alpha=0.8)
    ax.plot(g, total, label="Total loss (RMSE + L1 + L2)", color="#2c3e50", linewidth=1.5,
            linestyle="--")
    ax.plot(g, l1, label="L1 (dimensionless)", color="#3498db", linewidth=1, alpha=0.8)
    ax.plot(g, l2, label="L2 (dimensionless)", color="#2ecc71", linewidth=1, alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Loss breakdown")
    plt.tight_layout()
    _finish_fig(fig, cfg, "loss_breakdown", save_plots, show_plots)


def plot_timing(history: dict, cfg: TNEPconfig,
                save_plots: str | None = None, show_plots: bool = True) -> None:
    """Plot per-generation timing breakdown and aggregate summary."""
    timing = history.get("timing")
    if not timing or not timing.get("evaluate"):
        return

    g = np.arange(len(timing["evaluate"]))
    phases = ["evaluate", "validate", "rank_update", "sample_batch", "overhead"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#95a5a6"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked area chart per generation
    data = [np.array(timing[p]) for p in phases]
    ax1.stackplot(g, *data, labels=phases, colors=colors, alpha=0.8)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Time (s)")
    ax1.legend(loc="upper left")
    ax1.set_title("Per-generation timing breakdown")
    # Cap y-axis at 99th percentile to prevent TF warmup outliers from hiding data
    totals_per_gen = sum(data)
    if len(totals_per_gen) > 1:
        ax1.set_ylim(bottom=0, top=np.percentile(totals_per_gen, 99) * 1.1)

    # Right: horizontal bar chart of totals
    totals = [sum(timing[p]) for p in phases]
    grand_total = max(sum(totals), 1e-9)
    bars = ax2.barh(phases, totals, color=colors)
    ax2.set_xlabel("Total time (s)")
    ax2.set_title(f"Aggregate timing ({grand_total:.2f}s total)")
    for bar, total in zip(bars, totals):
        pct = 100 * total / grand_total
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f" {total:.3f}s ({pct:.1f}%)", va='center')

    plt.tight_layout()
    _finish_fig(fig, cfg, "timing", save_plots, show_plots)


def _build_suptitle(cfg: TNEPconfig, suffix: str | None,
                    rmse: float, rrmse: float, r2: float) -> str:
    """Build a suptitle string with mode, suffix label, and metrics."""
    mode_names = {0: "PES", 1: "Dipole", 2: "Polarizability"}
    mode = mode_names.get(cfg.target_mode, f"Mode {cfg.target_mode}")
    if suffix:
        label_parts = []
        if "best" in suffix:
            label_parts.append("Best Val")
        elif "final" in suffix:
            label_parts.append("Final Gen")
        if "per_atom" in suffix:
            label_parts.append("Per Atom")
        elif "total" in suffix:
            label_parts.append("Total")
        # Detect validation set: "_val_" or trailing "_val" beyond "best_val"
        stripped = suffix.replace("best_val", "", 1)
        if "val" in stripped:
            label_parts.append("Validation")
        elif "test" in stripped:
            label_parts.append("Test")
        gen_match = re.search(r'gen(\d+)', suffix)
        if gen_match:
            label_parts.append(f"Gen {gen_match.group(1)}")
        label = f" ({' '.join(label_parts)})" if label_parts else f" ({suffix})"
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

    # Relative RMSE per component: RMSE_i / mean(|target_i|)
    rrmse_comp = np.array([
        np.sqrt(np.mean((targets[:, i] - predictions[:, i]) ** 2))
        / max(np.mean(np.abs(targets[:, i])), 1e-12)
        for i in range(T)
    ])
    # Overall RRMSE
    rrmse = rmse / max(np.mean(np.abs(targets)), 1e-12)

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
