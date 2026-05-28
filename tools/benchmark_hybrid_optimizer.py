"""Benchmark the hybrid Adam/SNES optimizer against pure SNES (and optionally
pure Adam) on the same dataset and seed.

Configure the run by editing the Config block below — there is NO CLI argument
parser; the file is meant to be opened and run directly in the IDE (right-click
-> Run File).

Output: a comparison table showing, for each optimizer mode:
    mode | final_val_rmse | best_val_rmse | gens_to_target | wall_time_s

``gens_to_target`` is the first generation index at which the validation RMSE
fell to or below ``target_rmse``, or "—" if the target was never reached.
"""
from __future__ import annotations

import copy
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# Ensure the project root is importable even when the file is launched from
# tools/ directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ═══════════════════════════════════════════════════════════════════
# CONFIG — edit values here, then run the file.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Config:
    # Path to the XYZ dataset used for train/val.
    data_path: str = "datasets/test.xyz"

    # Total number of structures to draw from data_path (before train/val
    # split). Reduce for faster smoke-runs.
    total_N: int = 60

    # Number of SNES/Adam generations to run for each mode.
    num_generations: int = 400

    # Number of hidden neurons in the ANN.
    num_neurons: int = 16

    # SNES population size. Smaller => faster per-gen, noisier gradients.
    pop_size: int = 20

    # Validation RMSE threshold — gens_to_target is the first generation
    # at which val RMSE drops to this value or below.
    target_rmse: float = 0.05

    # Adam plateau patience (val-ticks without improvement before swap to SNES).
    adam_plateau_patience: int = 100

    # SNES plateau patience (val-ticks without improvement before swap back).
    snes_plateau_patience: int = 2000

    # Adam learning rate.
    adam_lr: float = 1e-3

    # If True, also benchmark pure Adam mode (in addition to snes + hybrid).
    run_adam_only: bool = False

    # Random seed — controls dataset split + SNES initialisation.
    seed: int = 0

    # Species filter passed to collect().
    allowed_species: list[int] = field(default_factory=lambda: [6, 1, 7, 8])
    filter_mode: str = "subset"

    # Fraction of total_N held out for validation.
    test_ratio: float = 0.25

    # target_mode: 1 = dipole.
    target_mode: int = 1
    dipole_units: str = "e*bohr"
    dipole_rij_power: int = 2


CFG = Config()


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _build_cfg_for_mode(bench_cfg: Config, mode: str):
    """Return a freshly constructed TNEPconfig for the given optimizer mode.

    All training-relevant fields are taken from bench_cfg; only optimizer_mode
    differs across calls so that the comparison is apples-to-apples.
    """
    from TNEPconfig import TNEPconfig
    from data import collect, split, pad_and_stack
    from DescriptorBuilderGPU import compute_dim_q

    cfg = TNEPconfig()

    # Data
    cfg.data_path = bench_cfg.data_path
    cfg.test_data_path = None
    cfg.allowed_species = list(bench_cfg.allowed_species)
    cfg.filter_mode = bench_cfg.filter_mode
    cfg.total_N = bench_cfg.total_N
    cfg.test_ratio = bench_cfg.test_ratio

    # Target
    cfg.target_mode = bench_cfg.target_mode
    cfg.dipole_units = bench_cfg.dipole_units
    cfg.dipole_rij_power = bench_cfg.dipole_rij_power
    cfg.scale_targets = True
    cfg.convert_dipole_to_eangstrom = False

    # Architecture
    cfg.num_neurons = bench_cfg.num_neurons
    cfg.descriptor_mode = 0
    cfg.descriptor_mixing = False
    cfg.skip_h_centers = False

    # Training
    cfg.pop_size = bench_cfg.pop_size
    cfg.num_generations = bench_cfg.num_generations
    cfg.seed = bench_cfg.seed
    cfg.val_interval = 1
    cfg.val_size = None

    # Optimizer
    cfg.optimizer_mode = mode
    cfg.adam_plateau_patience = bench_cfg.adam_plateau_patience
    cfg.snes_plateau_patience = bench_cfg.snes_plateau_patience
    cfg.adam_lr = bench_cfg.adam_lr
    cfg.hybrid_start = "adam"
    cfg.patience = None  # no global early stopping — run the full budget

    # Performance / IO (minimal, benchmark-safe)
    cfg.population_chunk_size = None
    cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True
    cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False
    cfg.use_pinned_buffers = False
    cfg.use_cufile = False
    cfg.save_path = None
    cfg.checkpoint_interval = None

    # Regularisation off (clean comparison)
    cfg.lambda_1 = 0.0
    cfg.lambda_2 = 0.0
    cfg.toggle_regularization = False
    cfg.per_type_regularization = False

    cfg.eval_jit_compile = False

    # Build data — must happen before compute_dim_q
    dataset, ti = collect(cfg)
    cfg.randomise(dataset)
    cfg.dim_q = compute_dim_q(cfg)

    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)

    return cfg, train, val


def _run_mode(bench_cfg: Config, mode: str) -> dict:
    """Run one optimizer mode and return a result dict with stats."""
    from TNEP import TNEP

    print(f"\n{'='*60}")
    print(f"Running mode: {mode!r}")
    print(f"{'='*60}")

    cfg, train, val = _build_cfg_for_mode(bench_cfg, mode)
    model = TNEP(cfg)

    t_start = time.perf_counter()
    result = model.optimizer.fit(train, val)
    wall_time = time.perf_counter() - t_start

    # fit() returns (history, final_model, best_val_model)
    if isinstance(result, tuple):
        history = result[0]
    else:
        history = result

    val_series = np.array(history["val_loss"], dtype=np.float64)
    gen_series = np.array(history["generation"], dtype=np.int64)

    if len(val_series) == 0:
        return {
            "mode": mode,
            "final_val": float("nan"),
            "best_val": float("nan"),
            "gens_to_target": None,
            "wall_time": wall_time,
        }

    final_val = float(val_series[-1])
    best_val = float(np.min(val_series))

    # gens_to_target: first generation (0-indexed) where val RMSE <= target
    hits = np.where(val_series <= bench_cfg.target_rmse)[0]
    if len(hits) > 0:
        gens_to_target = int(gen_series[hits[0]])
    else:
        gens_to_target = None

    return {
        "mode": mode,
        "final_val": final_val,
        "best_val": best_val,
        "gens_to_target": gens_to_target,
        "wall_time": wall_time,
    }


def _print_table(results: list[dict], target_rmse: float) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = {"mode": 8, "final_val": 14, "best_val": 13, "gens_to_target": 16, "wall_time": 13}
    header = (
        f"{'mode':<{col_w['mode']}} "
        f"{'final_val_rmse':>{col_w['final_val']}} "
        f"{'best_val_rmse':>{col_w['best_val']}} "
        f"{'gens_to_target':>{col_w['gens_to_target']}} "
        f"{'wall_time_s':>{col_w['wall_time']}}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(f"HYBRID vs SNES OPTIMIZER BENCHMARK  (target_rmse={target_rmse})")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        g2t = str(r["gens_to_target"]) if r["gens_to_target"] is not None else "—"
        print(
            f"{r['mode']:<{col_w['mode']}} "
            f"{r['final_val']:>{col_w['final_val']}.6f} "
            f"{r['best_val']:>{col_w['best_val']}.6f} "
            f"{g2t:>{col_w['gens_to_target']}} "
            f"{r['wall_time']:>{col_w['wall_time']}.1f}"
        )
    print("=" * len(header))
    print()


# ═══════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    bench_cfg = CFG

    modes = ["snes", "hybrid"]
    if bench_cfg.run_adam_only:
        modes.append("adam")

    print(f"Benchmark settings:")
    print(f"  data_path       = {bench_cfg.data_path}")
    print(f"  total_N         = {bench_cfg.total_N}")
    print(f"  num_generations = {bench_cfg.num_generations}")
    print(f"  num_neurons     = {bench_cfg.num_neurons}")
    print(f"  pop_size        = {bench_cfg.pop_size}")
    print(f"  target_rmse     = {bench_cfg.target_rmse}")
    print(f"  seed            = {bench_cfg.seed}")
    print(f"  modes           = {modes}")

    results = []
    for mode in modes:
        try:
            r = _run_mode(bench_cfg, mode)
            results.append(r)
        except Exception as exc:
            print(f"\n[ERROR] mode={mode!r} raised: {exc!r}")
            results.append({
                "mode": mode,
                "final_val": float("nan"),
                "best_val": float("nan"),
                "gens_to_target": None,
                "wall_time": 0.0,
            })

    _print_table(results, bench_cfg.target_rmse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
