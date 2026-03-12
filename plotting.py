from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import os

from TNEPconfig import TNEPconfig
from data import component_labels


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
                      save_plots: str | None = None, show_plots: bool = True,
                      logy: bool = False) -> None:
    """Plot train and validation RMSE vs generation with best/worst band."""
    g = np.asarray(history["generation"])

    fig = plt.figure()
    plt.plot(g, history["train_loss"], label="Train RMSE")
    plt.plot(g, history["val_loss"], label="Val RMSE")

    if history.get("best_rmse") and history.get("worst_rmse"):
        plt.fill_between(g, history["best_rmse"], history["worst_rmse"],
                         alpha=0.2, label="Best\u2013Worst range")

    plt.xlabel("generation")
    plt.ylabel("fitness (lower is better)")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.title("SNES fitness vs generation")
    _finish_fig(fig, cfg, "snes_fitness", save_plots, show_plots)


def plot_log_val_fitness(history: dict, cfg: TNEPconfig,
                         save_plots: str | None = None, show_plots: bool = True) -> None:
    """Plot natural log of validation fitness vs generation."""
    g = np.asarray(history["generation"])
    val = np.asarray(history["val_loss"])
    ln_val = np.log(val)
    ln_g = np.log(g)

    fig = plt.figure()
    plt.plot(ln_g, ln_val, label="ln(Val RMSE)")
    plt.xlabel("Ln(Generation)")
    plt.ylabel("ln(Validation RMSE)")
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

    for reset_gen in history.get("sigma_resets", []):
        plt.axvline(x=reset_gen, color="red", linestyle="--", alpha=0.7,
                    label="Sigma reset" if reset_gen == history["sigma_resets"][0] else None)

    plt.xlabel("generation")
    plt.ylabel("sigma")
    plt.yscale("log")
    plt.legend()
    plt.title("SNES sigma evolution")
    _finish_fig(fig, cfg, "sigma_evolution", save_plots, show_plots)


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


def plot_correlation(targets: np.ndarray, predictions: np.ndarray, metrics: dict,
                     cfg: TNEPconfig, save_plots: str | None = None,
                     show_plots: bool = True) -> None:
    """Plot target vs prediction correlation with per-component R² and cosine similarity.

    For scalar targets (PES), plots a single correlation panel.
    For vector targets (dipole [3] or polarizability [6]), plots one correlation
    panel per component plus a cosine similarity histogram.

    Args:
        targets     : [S, T] numpy array of target values
        predictions : [S, T] numpy array of predicted values
        metrics     : dict from TNEP.score() with r2, rmse, r2_components, cos_sim_all
        cfg         : TNEPconfig — used to determine target mode and labels
        save_plots  : str or None — directory to save into
        show_plots  : bool — True to display interactively
    """
    T = targets.shape[1]
    rmse = float(metrics["rmse"])
    r2 = float(metrics["r2"])
    r2_comp = metrics["r2_components"].numpy()

    labels = component_labels(cfg.target_mode, T)

    # Add extra column for cosine similarity histogram on vector targets
    has_cos = cfg.target_mode >= 1 and "cos_sim_all" in metrics
    ncols = min(T, 3)
    nrows = (T + ncols - 1) // ncols
    if has_cos:
        nrows += 1  # extra row for cosine similarity

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

        ax.set_xlabel(f"Target ({labels[i]})")
        ax.set_ylabel(f"Prediction ({labels[i]})")
        ax.set_title(f"{labels[i]}  R²={r2_comp[i]:.4f}")
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left')

    # Hide unused correlation subplots
    cos_row_start = (T + ncols - 1) // ncols  # first row used by cosine sim
    for i in range(T, cos_row_start * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    # Cosine similarity histogram
    if has_cos:
        cos_all = metrics["cos_sim_all"].numpy()
        cos_mean = float(metrics["cos_sim_mean"])

        # Span all columns in the bottom row
        for c in range(ncols):
            axes[nrows - 1, c].set_visible(False)
        ax_cos = fig.add_subplot(nrows, 1, nrows)

        ax_cos.hist(cos_all, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax_cos.axvline(cos_mean, color='red', linestyle='--', linewidth=1.5,
                       label=f"Mean = {cos_mean:.4f}")
        ax_cos.set_xlabel("Cosine similarity")
        ax_cos.set_ylabel("Count")
        ax_cos.set_xlim(-1.05, 1.05)
        ax_cos.set_title(f"Cosine similarity distribution  "
                         f"(mean={cos_mean:.4f}, std={cos_all.std():.4f})")
        ax_cos.legend()

    mode_names = {0: "PES", 1: "Dipole", 2: "Polarizability"}
    mode = mode_names.get(cfg.target_mode, f"Mode {cfg.target_mode}")
    fig.suptitle(f"{mode} — RMSE: {rmse:.4f}, R²: {r2:.4f}", fontsize=14)
    plt.tight_layout()
    _finish_fig(fig, cfg, "correlation", save_plots, show_plots)
