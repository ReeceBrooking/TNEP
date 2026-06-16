"""CR-FM-NES vs vanilla SNES vs rank-1 CMA benchmark harness.

Implements the C7 merge gate from
``docs/superpowers/plans/2026-05-31-crfmnes-snes.md``.

Modes compared:
  - vanilla SNES                  (snes_cov_mode="none", snes_mean_optimizer="vanilla")
  - rank-1 CMA at c1_scale=1000   (snes_cov_mode="rank1", snes_cma_c1_scale=1000)
  - CR-FM-NES                     (snes_cov_mode="crfmnes")

Per-mode: >=3 seeds (configurable via --seeds). Records val_RMSE per generation.

Gate (post-run, computed by this script):
  - Per-baseline-candidate elbow g*: EMA(N=50, alpha=2/(N+1)); per-decade decrease
    compared as EMA(g) vs EMA(round(g/sqrt(10))); criterion: < 1% relative drop;
    earliest g* >= 100.
  - Baseline = vanilla SNES. (Rank-1@1000 is swept as a diagnostic comparison
    point — useful evidence for or against the canonical-rate "numerically
    inert at d=15k" prediction — but is NOT a candidate for baseline selection.
    Earlier draft framed it as a working baseline; that finding was retracted
    after tracing back to an accidental unit-conversion toggle.)
  - Primary: median CR-FM-NES gens-to-target <= 70% * g*_baseline, target val_RMSE
    = baseline EMA(g*) * 1.05, all seeds reach target within num_generations.
  - Stability: gens-to-target CV (std/mean) <= 0.20 over CR-FM-NES seeds.
  - Secondary diagnostic: val_RMSE at baseline's lowest-RMSE wall-time
    >= CR-FM-NES's at same wall-time (i.e. CR-FM-NES does not regress later).

Output:
  - {output_dir}/{date}-crfmnes-vs-snes.md      (markdown report)
  - {output_dir}/{date}-crfmnes-vs-snes.png     (val-RMSE vs gen plot)
  - {output_dir}/{date}-crfmnes-vs-snes-raw.json (per-(mode,seed) val_RMSE series)

CLI:
  --model CONFIG_PATH    path to a TNEPconfig snapshot (config.txt) or "default"
                          for the tiny CHO test model
  --num-generations N    default 5000
  --pop-size P           default 100
  --seeds 0,1,2          comma-separated seeds
  --modes vanilla,rank1,crfmnes
  --output-dir DIR       default docs/benchmarks/
  --sanity               short mode: 500 gens, 1 seed per mode, smaller model
                          (just exercises the gate plumbing end-to-end)

Full-budget run (user, on GPU)::

    python tools/benchmark_crfmnes.py \\
        --model models/<your-recent-good-run>/config.txt \\
        --num-generations 60000 \\
        --pop-size 100 \\
        --seeds 0,1,2,3 \\
        --modes vanilla,rank1,crfmnes \\
        --output-dir docs/benchmarks/

Notes for the full-budget run:
  - The loaded config MUST have ``loss_type = "mse"`` so the recorded ``val_loss``
    series is directly comparable to a val_RMSE target. The harness asserts this
    and refuses to start otherwise. (``validate()`` returns RMSE regardless, but
    forcing loss_type="mse" keeps the optimised objective on the same scale.)
  - The harness disables checkpoint saving (``save_path = None``) and any
    global early stopping (``patience = None``) so every seed sees the full
    generation budget. This is required for elbow detection to be meaningful.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

# Ensure the project root is importable when launched from tools/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CR-FM-NES vs vanilla SNES vs rank-1 CMA merge-gate benchmark"
    )
    p.add_argument(
        "--model", type=str, default="default",
        help='Path to a TNEPconfig snapshot (config.txt) or "default" for '
             'the tiny CHO test model.',
    )
    p.add_argument("--num-generations", type=int, default=5000)
    p.add_argument("--pop-size", type=int, default=100)
    p.add_argument(
        "--seeds", type=str, default="0,1,2",
        help="Comma-separated integer seeds (per mode).",
    )
    p.add_argument(
        "--modes", type=str, default="vanilla,rank1,crfmnes",
        help="Comma-separated subset of {vanilla, rank1, crfmnes}.",
    )
    p.add_argument(
        "--output-dir", type=str, default="docs/benchmarks/",
    )
    p.add_argument(
        "--sanity", action="store_true",
        help="Short sanity run: 500 gens, 1 seed per mode, smaller model. "
             "Exercises the gate plumbing end-to-end; gate verdict is "
             "expected to be 'inconclusive'.",
    )
    return p.parse_args(argv)


# ═══════════════════════════════════════════════════════════════════
# Config loading
# ═══════════════════════════════════════════════════════════════════

# Fields that are derived at runtime from the dataset and should NOT be copied
# from a saved config.txt — they get re-computed by collect()/randomise().
_DERIVED_FIELDS = {
    "dim_q", "types", "type_map", "indices", "num_types",
    "save_path", "save_plots",
}


def _parse_config_txt(path: str) -> dict:
    """Parse a config.txt written by ``model_io.setup_run_directory``.

    Format is ``key = repr(value)`` lines plus ``#`` comments. Values are
    parsed with ``ast.literal_eval``; ndarray placeholder lines are skipped
    (those describe data-derived fields, which we re-build from the dataset).
    """
    out: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " = " not in line:
                continue
            key, val = line.split(" = ", 1)
            key = key.strip()
            val = val.strip()
            if val.startswith("ndarray "):
                continue
            try:
                out[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                # Non-literal repr (e.g. enum, custom class) — skip; the
                # benchmark only needs the standard scalar / list fields.
                continue
    return out


def _make_default_cfg(sanity: bool):
    """Build the tiny CHO test config (matches test_crfmnes / test_lowrank_cma)."""
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    cfg.data_path = "datasets/test.xyz"
    cfg.test_data_path = None
    cfg.allowed_species = [6, 1, 7, 8]
    cfg.filter_mode = "subset"
    cfg.target_mode = 1
    cfg.dipole_units = "e*bohr"
    cfg.scale_targets = True
    cfg.convert_dipole_to_eangstrom = False
    cfg.total_N = 16 if sanity else 60
    cfg.test_ratio = 0.25
    cfg.skip_h_centers = False
    cfg.num_neurons = 8 if sanity else 16
    cfg.descriptor_mode = 0
    cfg.descriptor_mixing = False
    cfg.dipole_rij_power = 2
    cfg.loss_type = "mse"
    return cfg


def _build_cfg_for_mode(
    base_cfg_dict: Optional[dict],
    use_default: bool,
    mode: str,
    seed: int,
    num_generations: int,
    pop_size: int,
    sanity: bool,
):
    """Construct a fresh (cfg, train, val) tuple for a single (mode, seed) run.

    ``mode`` is one of ``"vanilla"``, ``"rank1"``, ``"crfmnes"``.
    """
    from TNEPconfig import TNEPconfig
    from data import collect, split, pad_and_stack
    from DescriptorBuilderGPU import compute_dim_q

    if use_default:
        cfg = _make_default_cfg(sanity)
    else:
        cfg = TNEPconfig()
        for k, v in base_cfg_dict.items():
            if k in _DERIVED_FIELDS:
                continue
            if not hasattr(cfg, k) and k not in cfg.__class__.__annotations__:
                # Field that no longer exists on the dataclass — skip.
                continue
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

    # Force loss_type="mse" so val_loss is directly comparable to a val_RMSE
    # target. validate() always returns RMSE, but the optimised objective
    # tracked in history["train_loss"] also lives on the RMSE scale only when
    # loss_type="mse".
    if cfg.loss_type != "mse":
        raise SystemExit(
            f"[benchmark] cfg.loss_type='{cfg.loss_type}' but the C7 gate "
            f"requires 'mse' so val_loss is val_RMSE. Edit the loaded "
            f"config.txt or use --model default."
        )

    # Common benchmark-safe settings.
    cfg.num_generations = num_generations
    cfg.pop_size = pop_size
    cfg.seed = seed
    cfg.val_interval = 1
    cfg.val_size = None
    cfg.patience = None             # no early stopping; full budget per seed
    cfg.save_path = None            # no checkpoint writes
    cfg.checkpoint_interval = None
    cfg.eval_jit_compile = False
    cfg.population_chunk_size = None
    cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True
    cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False
    cfg.use_pinned_buffers = False
    cfg.use_cufile = False

    # Hold regularisation + guided-ES + per-type ranking constant across modes
    # so the only diff is the cov-mode / mean-optimizer pair.
    cfg.lambda_1 = getattr(cfg, "lambda_1", 0.0)
    cfg.lambda_2 = getattr(cfg, "lambda_2", 0.0)
    cfg.toggle_regularization = False
    cfg.per_type_regularization = False
    cfg.guided_es_enabled = False
    cfg.snes_sigma_cumulation = False
    cfg.optimizer_mode = "snes"
    cfg.snes_mean_optimizer = "vanilla"

    # Mode-specific knobs.
    if mode == "vanilla":
        cfg.snes_cov_mode = "none"
        cfg.snes_cma_c1_scale = 1.0
    elif mode == "rank1":
        cfg.snes_cov_mode = "rank1"
        cfg.snes_cma_c1_scale = 1000.0
    elif mode == "crfmnes":
        cfg.snes_cov_mode = "crfmnes"
        cfg.snes_cma_c1_scale = 1.0
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Build data + finalise dim_q.
    dataset, ti = collect(cfg)
    cfg.randomise(dataset)
    cfg.dim_q = compute_dim_q(cfg)

    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)

    return cfg, train, val


# ═══════════════════════════════════════════════════════════════════
# Run a single (mode, seed)
# ═══════════════════════════════════════════════════════════════════

def _run_one(
    base_cfg_dict: Optional[dict],
    use_default: bool,
    mode: str,
    seed: int,
    num_generations: int,
    pop_size: int,
    sanity: bool,
) -> dict:
    """Run a single (mode, seed) and return a dict with per-gen series + timing."""
    from TNEP import TNEP

    print(f"\n[bench] mode={mode!r} seed={seed} num_generations={num_generations}")
    cfg, train, val = _build_cfg_for_mode(
        base_cfg_dict, use_default, mode, seed,
        num_generations, pop_size, sanity,
    )
    model = TNEP(cfg)

    t0 = time.perf_counter()
    result = model.optimizer.fit(train, val)
    wall = time.perf_counter() - t0

    history = result[0] if isinstance(result, tuple) else result
    gens = np.asarray(history["generation"], dtype=np.int64)
    val_rmse = np.asarray(history["val_loss"], dtype=np.float64)

    print(f"[bench] mode={mode!r} seed={seed} done in {wall:.1f}s "
          f"final_val_rmse={float(val_rmse[-1]) if len(val_rmse) else float('nan'):.6f}")

    return {
        "mode": mode,
        "seed": seed,
        "wall_time_s": wall,
        "generations": gens.tolist(),
        "val_rmse": val_rmse.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════
# Gate computation
# ═══════════════════════════════════════════════════════════════════

def _ema(x: np.ndarray, N: int = 50) -> np.ndarray:
    """Exponential moving average with alpha = 2/(N+1). Returns array of same length."""
    if len(x) == 0:
        return x.copy()
    alpha = 2.0 / (N + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def compute_elbow_gstar(
    val_rmse: np.ndarray,
    N: int = 50,
    min_gen: int = 100,
    decade_drop_threshold: float = 0.01,
) -> int:
    """Earliest generation g >= min_gen where EMA val-RMSE has flattened.

    Implements the C7 elbow rule:
      * EMA smoothing factor alpha = 2/(N+1).
      * "Per-decade decrease" measured by comparing EMA(g) to EMA(floor(g/sqrt(10))),
        then projected over a full decade (multiply the half-decade drop by 2,
        since sqrt(10) spans half a decade in log10).
      * Elbow := first g >= min_gen where 2 * (1 - EMA(g)/EMA(half_dec)) < threshold.
    Returns ``-1`` if no elbow is found within the series.
    """
    val_rmse = np.asarray(val_rmse, dtype=np.float64)
    if len(val_rmse) <= min_gen:
        return -1
    ema = _ema(val_rmse, N=N)
    inv_sqrt10 = 1.0 / np.sqrt(10.0)
    for g in range(min_gen, len(val_rmse)):
        ref_idx = int(np.floor(g * inv_sqrt10))
        if ref_idx < 1:
            continue
        ref = ema[ref_idx]
        cur = ema[g]
        if not (np.isfinite(ref) and np.isfinite(cur)) or ref <= 0.0:
            continue
        half_decade_drop = 1.0 - (cur / ref)
        # Project to full decade — sqrt(10) is half a decade in log space.
        per_decade_drop = 2.0 * half_decade_drop
        if per_decade_drop < decade_drop_threshold:
            return g
    return -1


def compute_gens_to_target(val_rmse: np.ndarray, target: float) -> int:
    """First generation (index into val_rmse) where val_rmse[g] <= target.

    Returns -1 if the target is never reached.
    """
    val_rmse = np.asarray(val_rmse, dtype=np.float64)
    hits = np.where(val_rmse <= target)[0]
    return int(hits[0]) if len(hits) > 0 else -1


def _candidate_gstar_and_target(
    seed_series: list[np.ndarray],
) -> tuple[int, float]:
    """For one candidate (a mode with multiple seeds), compute the per-candidate
    elbow g* by:
      1. Computing each seed's elbow g_s individually.
      2. Taking the median g_s as the candidate g* (drop -1 sentinels first).
      3. Computing the candidate's EMA val-RMSE at g* on each seed and using
         the median as the target reference.
    Returns (g_star_or_-1, ema_val_rmse_at_gstar_or_nan).
    """
    per_seed_gstar = [compute_elbow_gstar(s) for s in seed_series]
    valid = [g for g in per_seed_gstar if g >= 0]
    if not valid:
        return -1, float("nan")
    g_star = int(np.median(valid))
    emas_at_gstar = []
    for s in seed_series:
        if g_star < len(s):
            ema_s = _ema(np.asarray(s))
            emas_at_gstar.append(ema_s[g_star])
    if not emas_at_gstar:
        return g_star, float("nan")
    return g_star, float(np.median(emas_at_gstar))


def select_baseline(
    per_mode_series: dict[str, list[np.ndarray]],
) -> tuple[str, int, float, dict]:
    """Pick the baseline per the C7 procedure: vanilla SNES is the sole
    baseline candidate. Rank-1 is reported as a diagnostic if present in
    `per_mode_series` (its `g_star`/`ema_at_gstar` are still computed and
    surfaced in `details`) but it does NOT participate in baseline selection.

    Args:
        per_mode_series: ``{mode_name: [val_rmse_seed_0, val_rmse_seed_1, ...]}``
            must include "vanilla"; "rank1" optional (diagnostic only).

    Returns:
        (baseline_name, g_star, target_val_rmse, details)
        ``baseline_name`` is ``""`` and ``g_star == -1`` if vanilla didn't elbow.
        ``target_val_rmse = EMA(g*) * 1.05``.
    """
    details = {}
    # Diagnostic computation for rank-1 if present — surfaced in the report
    # but NOT a baseline candidate.
    if "rank1" in per_mode_series:
        g_star_r, ema_r = _candidate_gstar_and_target(per_mode_series["rank1"])
        details["rank1"] = {
            "g_star": g_star_r,
            "ema_val_rmse_at_gstar": ema_r,
            "note": "diagnostic only; not a baseline candidate",
        }
    if "vanilla" not in per_mode_series:
        return "", -1, float("nan"), details
    g_star, ema_at_gstar = _candidate_gstar_and_target(per_mode_series["vanilla"])
    details["vanilla"] = {
        "g_star": g_star,
        "ema_val_rmse_at_gstar": ema_at_gstar,
    }
    if g_star < 0:
        return "", -1, float("nan"), details
    target = ema_at_gstar * 1.05
    return "vanilla", g_star, float(target), details


def evaluate_gate(
    baseline_name: str,
    baseline_gstar: int,
    target_val_rmse: float,
    baseline_series: list[np.ndarray],
    baseline_walls: list[float],
    crfmnes_series: list[np.ndarray],
    crfmnes_walls: list[float],
    num_generations: int,
) -> dict:
    """Evaluate the three gate criteria.

    Primary: median CR-FM-NES gens-to-target <= 0.70 * baseline_gstar AND all
             CR-FM-NES seeds reach target within num_generations.
    Stability: gens-to-target CV (std/mean) <= 0.20 across CR-FM-NES seeds.
    Secondary diagnostic: at the wall-clock time at which the baseline (median
             over its seeds) reaches its lowest val-RMSE, CR-FM-NES's median
             val-RMSE at the same wall time must be <= baseline's. (i.e.
             CR-FM-NES does not regress later.)

    Returns a dict with verdict flags + the quantitative numbers.
    """
    out = {
        "baseline_name": baseline_name,
        "baseline_gstar": baseline_gstar,
        "target_val_rmse": target_val_rmse,
        "primary_pass": False,
        "stability_pass": False,
        "secondary_pass": False,
        "inconclusive": False,
        "crfmnes_gens_to_target": [],
        "crfmnes_gens_to_target_median": float("nan"),
        "crfmnes_gens_to_target_cv": float("nan"),
        "baseline_min_val_rmse": float("nan"),
        "baseline_wall_at_min": float("nan"),
        "crfmnes_val_rmse_at_baseline_wall": float("nan"),
    }
    if baseline_name == "" or baseline_gstar < 0 or not np.isfinite(target_val_rmse):
        out["inconclusive"] = True
        return out

    # CR-FM-NES gens-to-target per seed.
    g2t = [compute_gens_to_target(s, target_val_rmse) for s in crfmnes_series]
    out["crfmnes_gens_to_target"] = g2t
    reached = [g for g in g2t if g >= 0]
    all_reached = (len(reached) == len(g2t))
    if reached:
        med = float(np.median(reached))
        out["crfmnes_gens_to_target_median"] = med
    else:
        med = float("nan")
    primary_pass = all_reached and (med <= 0.70 * baseline_gstar)
    out["primary_pass"] = bool(primary_pass)

    # Stability.
    if len(reached) >= 2:
        arr = np.asarray(reached, dtype=np.float64)
        cv = float(arr.std(ddof=0) / max(arr.mean(), 1e-12))
        out["crfmnes_gens_to_target_cv"] = cv
        out["stability_pass"] = bool(cv <= 0.20)

    # Secondary: compare at the baseline's lowest-RMSE wall-time.
    # We need per-mode median over seeds vs wall time. Each seed has its own
    # wall_per_gen ~= wall_time_s / num_gens; we project that to an absolute
    # wall-time per gen.
    def _interp_series_to_wall(series: np.ndarray, total_wall: float, wall_at: float) -> float:
        """Estimate val_rmse at a given wall-time assuming linear pacing."""
        n = len(series)
        if n == 0 or total_wall <= 0:
            return float("nan")
        # gen index proportional to wall time.
        frac = wall_at / total_wall
        idx = int(np.clip(round(frac * (n - 1)), 0, n - 1))
        return float(series[idx])

    # Baseline: per-seed wall_at_min = (argmin / num_gens) * wall_per_seed,
    # then take the median wall_at_min across baseline seeds.
    base_walls_at_min = []
    base_min_vals = []
    for s, w in zip(baseline_series, baseline_walls):
        if len(s) == 0:
            continue
        idx = int(np.argmin(s))
        base_min_vals.append(float(s[idx]))
        base_walls_at_min.append(w * (idx / max(len(s) - 1, 1)))
    if base_walls_at_min:
        baseline_wall_at_min = float(np.median(base_walls_at_min))
        out["baseline_min_val_rmse"] = float(np.median(base_min_vals))
        out["baseline_wall_at_min"] = baseline_wall_at_min
        # CR-FM-NES val_rmse at the same wall-time (per-seed interp -> median).
        crf_at_wall = [
            _interp_series_to_wall(np.asarray(s), w, baseline_wall_at_min)
            for s, w in zip(crfmnes_series, crfmnes_walls)
        ]
        crf_at_wall = [v for v in crf_at_wall if np.isfinite(v)]
        if crf_at_wall:
            crf_med_at_wall = float(np.median(crf_at_wall))
            out["crfmnes_val_rmse_at_baseline_wall"] = crf_med_at_wall
            out["secondary_pass"] = bool(
                crf_med_at_wall <= out["baseline_min_val_rmse"]
            )

    return out


# ═══════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════

def _generate_plot(
    runs: list[dict],
    output_path: str,
) -> None:
    """Plot val-RMSE vs gen per mode: median + min/max envelope over seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_mode: dict[str, list[dict]] = {}
    for r in runs:
        by_mode.setdefault(r["mode"], []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"vanilla": "tab:blue", "rank1": "tab:orange", "crfmnes": "tab:green"}
    for mode, mode_runs in by_mode.items():
        if not mode_runs:
            continue
        # Align on common gen index range (truncate to shortest).
        min_len = min(len(r["val_rmse"]) for r in mode_runs)
        if min_len == 0:
            continue
        stack = np.stack([np.asarray(r["val_rmse"][:min_len]) for r in mode_runs])
        gens = np.asarray(mode_runs[0]["generations"][:min_len])
        median = np.median(stack, axis=0)
        lo = np.min(stack, axis=0)
        hi = np.max(stack, axis=0)
        color = colors.get(mode, None)
        ax.plot(gens, median, label=f"{mode} (median)", color=color, lw=1.5)
        ax.fill_between(gens, lo, hi, alpha=0.2, color=color)
    ax.set_xlabel("generation")
    ax.set_ylabel("val RMSE")
    ax.set_yscale("log")
    ax.set_title("CR-FM-NES vs vanilla SNES vs rank-1 CMA")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _write_markdown(
    md_path: str,
    runs: list[dict],
    per_mode_series: dict[str, list[np.ndarray]],
    per_mode_walls: dict[str, list[float]],
    baseline_details: dict,
    gate: dict,
    args: argparse.Namespace,
    plot_filename: str,
) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# CR-FM-NES vs vanilla SNES vs rank-1 CMA benchmark")
    lines.append("")
    lines.append(f"Generated: {date_str}")
    lines.append("")
    lines.append("## Run settings")
    lines.append("")
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- num_generations: {args.num_generations}")
    lines.append(f"- pop_size: {args.pop_size}")
    lines.append(f"- seeds: {args.seeds}")
    lines.append(f"- modes: {args.modes}")
    lines.append(f"- sanity: {args.sanity}")
    lines.append("")
    lines.append("## Per-(mode, seed) summary")
    lines.append("")
    lines.append("| mode | seed | final val_RMSE | best val_RMSE | wall (s) | gens recorded |")
    lines.append("|---|---|---|---|---|---|")
    for r in runs:
        vr = r["val_rmse"]
        if vr:
            final_v = vr[-1]
            best_v = min(vr)
        else:
            final_v = float("nan"); best_v = float("nan")
        lines.append(
            f"| {r['mode']} | {r['seed']} | {final_v:.6g} | {best_v:.6g} | "
            f"{r['wall_time_s']:.1f} | {len(vr)} |"
        )
    lines.append("")
    lines.append("## Baseline-candidate elbow analysis")
    lines.append("")
    lines.append("| candidate | per-candidate g* | EMA val_RMSE at g* |")
    lines.append("|---|---|---|")
    for c in ("vanilla", "rank1"):
        d = baseline_details.get(c)
        if d is None:
            lines.append(f"| {c} | — (not run) | — |")
        else:
            g = d["g_star"]
            ema = d["ema_val_rmse_at_gstar"]
            g_str = str(g) if g >= 0 else "no elbow"
            ema_str = f"{ema:.6g}" if np.isfinite(ema) else "—"
            lines.append(f"| {c} | {g_str} | {ema_str} |")
    lines.append("")
    lines.append("## Gate verdict")
    lines.append("")
    if gate.get("inconclusive"):
        lines.append("**INCONCLUSIVE** — no baseline candidate elbowed within "
                     f"{args.num_generations} generations.")
        lines.append("")
        lines.append("Per the C7 plan, extend num_generations up to 2x the original "
                     "budget once; if no elbow is observed, switch to a model where "
                     "convergence is observed. Do NOT lower the elbow threshold.")
    else:
        lines.append(f"- Baseline: **{gate['baseline_name']}**")
        lines.append(f"- Baseline g* = {gate['baseline_gstar']}")
        lines.append(f"- Target val_RMSE = {gate['target_val_rmse']:.6g} "
                     f"(EMA(g*) * 1.05)")
        lines.append("")
        med = gate["crfmnes_gens_to_target_median"]
        cv = gate["crfmnes_gens_to_target_cv"]
        primary_threshold = 0.70 * gate["baseline_gstar"]
        primary_mark = "PASS" if gate["primary_pass"] else "FAIL"
        stability_mark = "PASS" if gate["stability_pass"] else "FAIL"
        secondary_mark = "PASS" if gate["secondary_pass"] else "FAIL"
        lines.append(f"- **Primary** ({primary_mark}): median CR-FM-NES "
                     f"gens-to-target = "
                     f"{med if np.isfinite(med) else '—'} "
                     f"(<= 70% * g* = {primary_threshold:.0f}); "
                     f"per-seed g2t = {gate['crfmnes_gens_to_target']}")
        lines.append(f"- **Stability** ({stability_mark}): CV = "
                     f"{cv if np.isfinite(cv) else '—'} (<= 0.20)")
        lines.append(f"- **Secondary diagnostic** ({secondary_mark}): "
                     f"baseline min val_RMSE = "
                     f"{gate['baseline_min_val_rmse']:.6g} at wall = "
                     f"{gate['baseline_wall_at_min']:.1f}s; CR-FM-NES val_RMSE at "
                     f"same wall = {gate['crfmnes_val_rmse_at_baseline_wall']:.6g}")
        lines.append("")
        all_pass = (gate["primary_pass"] and gate["stability_pass"]
                    and gate["secondary_pass"])
        if all_pass:
            lines.append("**OUTCOME: All three criteria PASS — merge as a "
                         "recommended option.**")
        elif not gate["primary_pass"]:
            lines.append("**OUTCOME: Primary criterion FAILED — leave merged as "
                         "an opt-in research feature; document the negative "
                         "result; flag to redirect effort to Adam-primary "
                         "(GNEP path) per the broader research.**")
        else:
            lines.append("**OUTCOME: Primary passed but stability/secondary "
                         "failed — keep opt-in; treat as 'promising but "
                         "seed-fragile'; investigate constant-tuning before "
                         "recommending.**")
    lines.append("")
    lines.append("## Plot")
    lines.append("")
    lines.append(f"![val-RMSE vs gen]({plot_filename})")
    lines.append("")
    lines.append("## Full-budget run command")
    lines.append("")
    lines.append("```")
    lines.append("python tools/benchmark_crfmnes.py \\")
    lines.append("    --model models/<your-recent-good-run>/config.txt \\")
    lines.append("    --num-generations 60000 \\")
    lines.append("    --pop-size 100 \\")
    lines.append("    --seeds 0,1,2,3 \\")
    lines.append("    --modes vanilla,rank1,crfmnes \\")
    lines.append("    --output-dir docs/benchmarks/")
    lines.append("```")
    lines.append("")
    lines.append("Note: the loaded `config.txt` MUST have `loss_type = \"mse\"` "
                 "so the recorded `val_loss` series is directly comparable to a "
                 "val_RMSE target. The harness asserts this and refuses to start "
                 "otherwise.")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.sanity:
        args.num_generations = 500
        args.seeds = "0"
        if not args.modes:
            args.modes = "vanilla,rank1,crfmnes"
        # Force the tiny CHO model for the sanity run, regardless of --model.
        args.model = "default"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("vanilla", "rank1", "crfmnes"):
            raise SystemExit(f"unknown mode {m!r}; must be in vanilla,rank1,crfmnes")

    use_default = (args.model == "default")
    base_cfg_dict = None if use_default else _parse_config_txt(args.model)

    os.makedirs(args.output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    base_name = f"{date_str}-crfmnes-vs-snes"
    if args.sanity:
        base_name += "-sanity"
    raw_path = os.path.join(args.output_dir, f"{base_name}-raw.json")
    md_path = os.path.join(args.output_dir, f"{base_name}.md")
    png_filename = f"{base_name}.png"
    png_path = os.path.join(args.output_dir, png_filename)

    print(f"[bench] settings: modes={modes} seeds={seeds} "
          f"num_generations={args.num_generations} pop_size={args.pop_size} "
          f"output_dir={args.output_dir} sanity={args.sanity}")

    runs: list[dict] = []
    for mode in modes:
        for seed in seeds:
            try:
                r = _run_one(
                    base_cfg_dict, use_default, mode, seed,
                    args.num_generations, args.pop_size, args.sanity,
                )
                runs.append(r)
            except Exception as exc:
                import traceback
                print(f"[bench] mode={mode!r} seed={seed} FAILED: {exc!r}")
                traceback.print_exc()
                runs.append({
                    "mode": mode, "seed": seed,
                    "wall_time_s": 0.0,
                    "generations": [], "val_rmse": [],
                    "error": repr(exc),
                })

    with open(raw_path, "w") as f:
        json.dump({
            "args": vars(args),
            "runs": runs,
        }, f, indent=2)
    print(f"[bench] wrote raw json: {raw_path}")

    # Assemble per-mode series and walls.
    per_mode_series: dict[str, list[np.ndarray]] = {}
    per_mode_walls: dict[str, list[float]] = {}
    for r in runs:
        per_mode_series.setdefault(r["mode"], []).append(
            np.asarray(r["val_rmse"], dtype=np.float64))
        per_mode_walls.setdefault(r["mode"], []).append(float(r["wall_time_s"]))

    # Baseline selection (over vanilla + rank1 only).
    baseline_name, baseline_gstar, target_val, baseline_details = select_baseline(
        per_mode_series)

    if baseline_name == "":
        gate = evaluate_gate(
            baseline_name="", baseline_gstar=-1,
            target_val_rmse=float("nan"),
            baseline_series=[], baseline_walls=[],
            crfmnes_series=per_mode_series.get("crfmnes", []),
            crfmnes_walls=per_mode_walls.get("crfmnes", []),
            num_generations=args.num_generations,
        )
    else:
        gate = evaluate_gate(
            baseline_name=baseline_name,
            baseline_gstar=baseline_gstar,
            target_val_rmse=target_val,
            baseline_series=per_mode_series[baseline_name],
            baseline_walls=per_mode_walls[baseline_name],
            crfmnes_series=per_mode_series.get("crfmnes", []),
            crfmnes_walls=per_mode_walls.get("crfmnes", []),
            num_generations=args.num_generations,
        )

    try:
        _generate_plot(runs, png_path)
        print(f"[bench] wrote plot: {png_path}")
    except Exception as exc:
        print(f"[bench] plot generation FAILED: {exc!r}")

    _write_markdown(md_path, runs, per_mode_series, per_mode_walls,
                    baseline_details, gate, args, png_filename)
    print(f"[bench] wrote markdown: {md_path}")

    # Stdout summary.
    print()
    print("=" * 72)
    print("BENCHMARK SUMMARY")
    print("=" * 72)
    for c in ("vanilla", "rank1"):
        d = baseline_details.get(c, {})
        g = d.get("g_star", -1)
        ema = d.get("ema_val_rmse_at_gstar", float("nan"))
        print(f"  {c:>10s} g* = {g if g>=0 else 'no elbow':<12} "
              f"EMA(g*) = {ema:.6g}" if np.isfinite(ema) else
              f"  {c:>10s} g* = {g if g>=0 else 'no elbow':<12} EMA(g*) = —")
    if gate.get("inconclusive"):
        print("  GATE: INCONCLUSIVE (baseline never elbowed)")
    else:
        print(f"  BASELINE: {gate['baseline_name']} (g*={gate['baseline_gstar']}, "
              f"target_val_rmse={gate['target_val_rmse']:.6g})")
        print(f"  PRIMARY  : {'PASS' if gate['primary_pass'] else 'FAIL'}")
        print(f"  STABILITY: {'PASS' if gate['stability_pass'] else 'FAIL'}")
        print(f"  SECONDARY: {'PASS' if gate['secondary_pass'] else 'FAIL'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
