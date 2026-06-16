"""Quick CLI for inspecting TNEP .h5 model or checkpoint files.

Usage:
    python inspect_h5.py path/to/model.h5
    python inspect_h5.py path/to/checkpoint.h5 --weights      # also dump weight stats
    python inspect_h5.py path/to/checkpoint.h5 --history      # print full history dict
    python inspect_h5.py path/to/checkpoint.h5 --config-only  # cfg only, nothing else

Prints:
- File tree (groups, datasets with shape + dtype, attributes).
- Cfg dict (pretty-printed JSON from /config).
- Optional: per-weight summary statistics (min / max / mean / std).
- Optional: history dict summary (final values + length) or full history.
"""
from __future__ import annotations

import argparse
import json
import sys

import h5py
import numpy as np


def _walk(node: h5py.Group, indent: int = 0) -> None:
    pad = "  " * indent
    if node.attrs:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v.tolist()
                     if isinstance(v, np.ndarray) else v)
                 for k, v in node.attrs.items()}
        print(f"{pad}  [attrs] {attrs}")
    for name, obj in node.items():
        if isinstance(obj, h5py.Group):
            print(f"{pad}{name}/")
            _walk(obj, indent + 1)
        else:
            shape_str = "scalar" if obj.shape == () else f"shape={obj.shape}"
            print(f"{pad}{name}  {shape_str}  dtype={obj.dtype}")
            if obj.attrs:
                a = {k: (v.decode() if isinstance(v, bytes) else v)
                     for k, v in obj.attrs.items()}
                print(f"{pad}  [attrs] {a}")


def _print_config(f: h5py.File) -> None:
    if "config" not in f:
        print("(no /config group)")
        return
    raw = f["config"][()]
    cfg = json.loads(raw)
    print(json.dumps(cfg, indent=2, default=str))


def _weight_stats(f: h5py.File) -> None:
    if "weights" not in f:
        print("(no /weights group)")
        return
    wg = f["weights"]
    print(f"{'name':<10} {'shape':<25} {'min':>10} {'max':>10} "
          f"{'mean':>10} {'std':>10} {'|w|_max':>10}")
    print("-" * 90)
    for name in wg:
        arr = wg[name][()]
        if isinstance(arr, np.ndarray) and arr.size > 0:
            print(f"{name:<10} {str(arr.shape):<25} "
                  f"{arr.min():>10.4f} {arr.max():>10.4f} "
                  f"{arr.mean():>10.4f} {arr.std():>10.4f} "
                  f"{np.abs(arr).max():>10.4f}")
        else:
            print(f"{name:<10} (scalar = {float(arr)})")


def _print_history(f: h5py.File, full: bool = False) -> None:
    if "history" not in f:
        print("(no /history group — this is a model file, not a checkpoint)")
        return
    hg = f["history"]
    print(f"{'key':<20} {'length':>8} {'first':>14} {'last':>14}")
    print("-" * 60)
    for k in hg:
        if k == "timing":
            continue
        arr = hg[k][()]
        first = f"{arr[0]:.6g}" if len(arr) else "—"
        last = f"{arr[-1]:.6g}" if len(arr) else "—"
        print(f"{k:<20} {len(arr):>8} {first:>14} {last:>14}")
    if "timing" in hg:
        print(f"\ntiming phases: {list(hg['timing'].keys())}")
        for phase in hg["timing"]:
            arr = hg["timing"][phase][()]
            print(f"  {phase:<15} sum={arr.sum():.2f}s  mean={arr.mean()*1000:.1f}ms")
    if full:
        print("\nFULL history dump:")
        for k in hg:
            if k == "timing":
                continue
            print(f"\n{k}:")
            print(hg[k][()])


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a TNEP .h5 file.")
    p.add_argument("path", help="Path to model or checkpoint .h5")
    p.add_argument("--weights", action="store_true",
                   help="Print per-weight summary stats (min/max/mean/std).")
    p.add_argument("--history", action="store_true",
                   help="Summarise the /history group (checkpoint only).")
    p.add_argument("--history-full", action="store_true",
                   help="Dump every history entry in full (long output).")
    p.add_argument("--config-only", action="store_true",
                   help="Print only the cfg, skip the file tree.")
    args = p.parse_args()

    with h5py.File(args.path, "r") as f:
        if not args.config_only:
            print(f"=== File: {args.path} ===")
            if f.attrs:
                print(f"[root attrs] {dict(f.attrs)}")
            _walk(f)
            print()
        print("=== Config ===")
        _print_config(f)
        if args.weights:
            print("\n=== Weight stats ===")
            _weight_stats(f)
        if args.history or args.history_full:
            print("\n=== History ===")
            _print_history(f, full=args.history_full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
