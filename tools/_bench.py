"""Reliable per-gen perf benchmark.

Runs N gens, discards a `warmup` window (first M), reports median + p95
over the warm window for `evaluate` and `validate` phases. Bypasses
train_model's printed aggregate summary which includes compile cost.

Usage:
    python _bench.py LABEL [TOTAL_N] [GENS] [WARMUP] [extra=key=val ...]
"""

from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import time
import json
import statistics

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def parse_kv_args(rest):
    out = {}
    for a in rest:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        # auto-cast
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                elif v.lower() == "none":
                    v = None
        out[k] = v
    return out


def build_cfg(total_N: int, gens: int, **overrides):
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    cfg.batch_size = None
    cfg.batch_chunk_size = 500
    cfg.pop_size = 80
    cfg.population_chunk_size = 20
    cfg.total_N = total_N
    cfg.test_ratio = 0.3
    cfg.num_generations = gens
    cfg.val_size = None
    cfg.val_interval = 1
    cfg.plot_interval = None
    cfg.save_path = None
    cfg.seed = 42
    cfg.target_mode = 1
    cfg.dipole_units = "e*bohr"
    cfg.chunk_prefetch = True
    cfg.prefetch_depth = 1
    cfg.use_pinned_buffers = True
    cfg.use_cufile = True
    cfg.eval_jit_compile = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def main():
    if len(sys.argv) < 2:
        print("usage: _bench.py LABEL [TOTAL_N=2000] [GENS=40] [WARMUP=10] [k=v ...]")
        sys.exit(2)
    label = sys.argv[1]
    total_N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    warmup = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    overrides = parse_kv_args(sys.argv[5:])

    cfg = build_cfg(total_N, gens, **overrides)

    # Imports here so the prints below show config first.
    from TNEPconfig import TNEPconfig  # noqa
    import MasterTNEP

    captured = {}

    # Patch the inner method on TNEP that calls SNES.fit so we see the
    # history regardless of how SNES wraps it.
    import TNEP as _tnep_mod
    _orig_tnep_fit = _tnep_mod.TNEP.fit

    def _capture_tnep_fit(self, train_data, val_data, plot_callback=None,
                            resume_state=None):
        history, final_model, best_val_model = _orig_tnep_fit(
            self, train_data, val_data, plot_callback=plot_callback,
            resume_state=resume_state)
        # Deep-copy timing arrays since downstream code may clear/replace history.
        import copy as _copy
        timing_snap = _copy.deepcopy(history.get("timing", {}))
        captured["history"] = {"timing": timing_snap}
        print(f"BENCH_CAPTURE: gens recorded = "
              f"{len(timing_snap.get('evaluate', []))}",
              flush=True)
        return history, final_model, best_val_model

    _tnep_mod.TNEP.fit = _capture_tnep_fit

    print(f"\n=== BENCH: {label} | total_N={total_N} gens={gens} "
          f"warmup={warmup} | overrides={overrides} ===", flush=True)
    t0 = time.perf_counter()
    MasterTNEP.train_model(cfg)
    wall = time.perf_counter() - t0

    h = captured.get("history") or {}
    timing = h.get("timing", {})
    if not timing:
        print("NO_TIMING_RECORDED")
        return

    def stats(arr):
        if len(arr) <= warmup:
            return None
        warm = arr[warmup:]
        if not warm:
            return None
        return {
            "n": len(warm),
            "median_ms": 1000 * statistics.median(warm),
            "p95_ms": 1000 * (statistics.quantiles(warm, n=20)[18]
                              if len(warm) >= 20 else max(warm)),
            "min_ms": 1000 * min(warm),
            "max_ms": 1000 * max(warm),
            "mean_ms": 1000 * statistics.mean(warm),
        }

    summary = {
        "label": label,
        "total_N": total_N,
        "gens": gens,
        "warmup": warmup,
        "wall_total_s": round(wall, 2),
        "overrides": overrides,
    }
    for phase in ("evaluate", "validate", "rank_update", "sample_batch", "overhead"):
        s = stats(timing.get(phase, []))
        if s:
            summary[phase] = s

    # Sum of phase medians = approximate steady-state per-gen cost
    phase_medians = [v["median_ms"] for k, v in summary.items()
                     if isinstance(v, dict) and "median_ms" in v]
    summary["sum_of_phase_medians_ms"] = sum(phase_medians)

    print("\nRESULT_JSON " + json.dumps(summary))


if __name__ == "__main__":
    main()
