"""Inspect the actual scale and distribution of TNEP SOAP-turbo descriptors.

Builds descriptors for a sample of structures from a dataset and reports:

- Overall descriptor statistics (min, max, mean, std, percentiles)
- Per-channel ranges (which channels are tiny, which are huge)
- Per-(species-pair) block aggregates (which pairs dominate)
- Per-l aggregates (radial-only vs high-angular content)
- Implied NEP-style q_scaler = 1 / (max - min) for each channel
- "Pathological" channel flags (constant / near-zero range / extreme outliers)

Useful for:
- Deciding whether per-channel normalisation will help training
- Understanding why the SNES init scale (Glorot etc.) is set where it is
- Diagnosing why certain (n, l, pair) combinations may be under-utilised

Usage:
    python inspect_descriptors.py                          # cfg defaults, 100 structures
    python inspect_descriptors.py --n_structures 500
    python inspect_descriptors.py --data_path my.xyz
    python inspect_descriptors.py --top 20                 # print top-20 widest / tightest
    python inspect_descriptors.py --save_csv stats.csv     # also dump per-channel CSV
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _percentile_row(name: str, x: np.ndarray) -> str:
    return (f"{name:<24} "
            f"min={x.min():>10.4f}  "
            f"p1={np.percentile(x, 1):>10.4f}  "
            f"p50={np.percentile(x, 50):>10.4f}  "
            f"p99={np.percentile(x, 99):>10.4f}  "
            f"max={x.max():>10.4f}  "
            f"mean={x.mean():>10.4f}  "
            f"std={x.std():>10.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_structures", type=int, default=100,
                   help="Number of structures to build descriptors for (default 100).")
    p.add_argument("--data_path", type=str, default=None,
                   help="Override cfg.data_path.")
    p.add_argument("--allowed_species", type=str, default=None,
                   help="Comma-separated atomic numbers, e.g. '6,1,8'. Overrides cfg.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the structure sub-sample.")
    p.add_argument("--top", type=int, default=10,
                   help="Show top-N widest and tightest channels (default 10).")
    p.add_argument("--save_csv", type=str, default=None,
                   help="Optional path to write per-channel statistics as CSV.")
    args = p.parse_args()

    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.allowed_species is not None:
        cfg.allowed_species = [int(s) for s in args.allowed_species.split(",")]

    # Load + filter dataset
    from data import collect
    dataset, dataset_types_int = collect(cfg)
    cfg.type_map = {z: idx for idx, z in enumerate(cfg.types)}
    n_total = len(dataset)
    if n_total == 0:
        print("No structures remain after filtering — aborting.", file=sys.stderr)
        return 1
    n_sample = min(args.n_structures, n_total)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_total, size=n_sample, replace=False)
    sub_structures = [dataset[i] for i in idx]
    sub_types_int = [dataset_types_int[i] for i in idx]

    print(f"\nDataset: {cfg.data_path}")
    print(f"  Total structures (after species filter): {n_total}")
    print(f"  Sampling {n_sample} for descriptor inspection")
    print(f"  Species: {cfg.types}")
    print(f"  Descriptor params: alpha_max={cfg.alpha_max} l_max={cfg.l_max} "
          f"rcut={cfg.rcut_hard} compress={cfg.compress_mode}")

    # Resolve dim_q from cfg analytically (avoids needing a build)
    from DescriptorBuilderGPU import compute_dim_q, descriptor_block_layout
    cfg.dim_q = compute_dim_q(cfg)
    layout = descriptor_block_layout(cfg)
    print(f"  dim_q = {cfg.dim_q}  (channels per atom)")
    print(f"  pair blocks: {layout['pair_keys']}")
    print(f"  block sizes: {[layout['block_sizes'][k] for k in layout['pair_keys']]}")

    # Build descriptors (gradients not needed for stat inspection)
    from DescriptorBuilder import make_descriptor_builder
    builder = make_descriptor_builder(cfg)
    print(f"\nBuilding descriptors for {n_sample} structures...")
    desc_list, _, _ = builder.build_descriptors(
        sub_structures, calc_gradients=False)
    # Concatenate per-atom rows into a flat [N, Q] array
    Q = cfg.dim_q
    flat = np.concatenate([np.asarray(d).reshape(-1, Q) for d in desc_list], axis=0)
    N = flat.shape[0]
    print(f"  Got {N} atoms total ({n_sample} structures, avg {N/n_sample:.1f} atoms/struct)")

    # ===== Overall stats =====
    print("\n" + "=" * 110)
    print("OVERALL (across all atoms and all Q channels)")
    print("=" * 110)
    print(_percentile_row("global", flat))

    # ===== Per-channel ranges =====
    chan_min = flat.min(axis=0)
    chan_max = flat.max(axis=0)
    chan_mean = flat.mean(axis=0)
    chan_std = flat.std(axis=0)
    chan_range = chan_max - chan_min
    # Implied NEP-style q_scaler = 1 / range. Channels with zero range
    # get scaler = inf — flag them separately.
    safe_range = np.maximum(chan_range, 1e-30)
    q_scaler = 1.0 / safe_range

    print("\n" + "=" * 110)
    print("PER-CHANNEL DISTRIBUTION (stats over the Q channels' summary statistics)")
    print("=" * 110)
    print(_percentile_row("chan range (max-min)", chan_range))
    print(_percentile_row("chan std",             chan_std))
    print(_percentile_row("chan |mean|",          np.abs(chan_mean)))
    print(_percentile_row("implied q_scaler",     q_scaler[q_scaler < 1e20]))

    near_const = np.sum(chan_range < 1e-10)
    if near_const:
        print(f"\n  WARNING: {near_const} channel(s) have range < 1e-10 — "
              f"essentially constant across the dataset. These add no "
              f"information; consider whether SOAP params are pathological "
              f"for your structures.")

    # ===== Top-N widest / tightest channels =====
    print("\n" + "=" * 110)
    print(f"TOP {args.top} WIDEST CHANNELS (largest range across atoms)")
    print("=" * 110)
    order_wide = np.argsort(-chan_range)[:args.top]
    print(f"{'idx':>5} {'pair':>8} {'l':>3} {'n':>3} "
          f"{'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    for q in order_wide:
        pk, l_val, n_in_pair = _q_to_meta(int(q), layout, cfg.l_max + 1)
        print(f"{q:>5} {str(pk):>8} {l_val:>3} {n_in_pair:>3} "
              f"{chan_min[q]:>10.4f} {chan_max[q]:>10.4f} "
              f"{chan_mean[q]:>10.4f} {chan_std[q]:>10.4f}")

    print(f"\nTOP {args.top} TIGHTEST CHANNELS (smallest range — most informative-poor)")
    print("=" * 110)
    order_tight = np.argsort(chan_range)[:args.top]
    print(f"{'idx':>5} {'pair':>8} {'l':>3} {'n':>3} "
          f"{'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    for q in order_tight:
        pk, l_val, n_in_pair = _q_to_meta(int(q), layout, cfg.l_max + 1)
        print(f"{q:>5} {str(pk):>8} {l_val:>3} {n_in_pair:>3} "
              f"{chan_min[q]:>10.4g} {chan_max[q]:>10.4g} "
              f"{chan_mean[q]:>10.4g} {chan_std[q]:>10.4g}")

    # ===== Per-pair aggregates =====
    print("\n" + "=" * 110)
    print("PER-PAIR-BLOCK AGGREGATES (mean / max abs across channels in each block)")
    print("=" * 110)
    print(f"{'pair':<8} {'bs':>4} {'mean |q|':>12} {'max |q|':>12} "
          f"{'mean range':>12} {'max range':>12} {'mean std':>12}")
    for pk in layout["pair_keys"]:
        qidx = layout["pair_q_index"][pk]
        bs = qidx.size
        block_vals = flat[:, qidx]
        block_range = chan_range[qidx]
        print(f"{str(pk):<8} {bs:>4} "
              f"{np.abs(block_vals).mean():>12.4f} "
              f"{np.abs(block_vals).max():>12.4f} "
              f"{block_range.mean():>12.4f} "
              f"{block_range.max():>12.4f} "
              f"{chan_std[qidx].mean():>12.4f}")

    # ===== Per-l aggregates =====
    print("\n" + "=" * 110)
    print("PER-L AGGREGATES (channels grouped by angular momentum, all pairs combined)")
    print("=" * 110)
    print(f"{'l':>3} {'N_l':>5} {'mean |q|':>12} {'max |q|':>12} "
          f"{'mean range':>12} {'max range':>12} {'mean std':>12}")
    for l in range(cfg.l_max + 1):
        l_qidx = layout["l_index"][l]
        l_vals = flat[:, l_qidx]
        l_range = chan_range[l_qidx]
        print(f"{l:>3} {l_qidx.size:>5} "
              f"{np.abs(l_vals).mean():>12.4f} "
              f"{np.abs(l_vals).max():>12.4f} "
              f"{l_range.mean():>12.4f} "
              f"{l_range.max():>12.4f} "
              f"{chan_std[l_qidx].mean():>12.4f}")

    # ===== Implications =====
    print("\n" + "=" * 110)
    print("IMPLICATIONS")
    print("=" * 110)
    ratio = chan_range.max() / max(chan_range.min(), 1e-30)
    print(f"  Max/min channel range ratio: {ratio:.2g}")
    if ratio > 100:
        print("    → Wildly different channel scales. Per-channel "
              "normalisation (NEP q_scaler-style) likely helps the bottleneck "
              "W0 spend capacity on meaningful directions rather than scale.")
    elif ratio > 10:
        print("    → Moderate scale spread. Per-channel scaling may "
              "give a small boost (~5–10% RMSE).")
    else:
        print("    → Channels are roughly on the same scale already. "
              "Per-channel normalisation will probably make little "
              "difference.")

    # NEP-style q_scaler magnitude check
    finite_scaler = q_scaler[q_scaler < 1e20]
    print(f"  Implied q_scaler magnitude: "
          f"min={finite_scaler.min():.4f} max={finite_scaler.max():.4f} "
          f"mean={finite_scaler.mean():.4f}")
    print(f"  After q_scaler-style normalisation, all channels would map "
          f"to range ≤ 1 (max-min normalisation, no centring).")

    # ===== CSV dump =====
    if args.save_csv is not None:
        import csv
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["q_idx", "pair_a", "pair_b", "l", "n_in_pair",
                        "min", "max", "mean", "std", "range", "q_scaler"])
            for q in range(Q):
                pk, l_val, n_in_pair = _q_to_meta(int(q), layout, cfg.l_max + 1)
                w.writerow([q, pk[0], pk[1], l_val, n_in_pair,
                            chan_min[q], chan_max[q],
                            chan_mean[q], chan_std[q],
                            chan_range[q],
                            float(q_scaler[q]) if q_scaler[q] < 1e20 else float("inf")])
        print(f"\n  Per-channel CSV written to {args.save_csv}")

    return 0


def _q_to_meta(q: int, layout: dict, L: int) -> tuple:
    """Reverse-lookup: given a flat q-index, return (pair_key, l, n_in_pair).

    Uses the Fortran emit order:
        for each (n, m) pivot pair:
            for l = 0 .. l_max:
                emit one channel
    so within a pair's contiguous channel list, position i has l = i % L.
    """
    for pk in layout["pair_keys"]:
        qidx_in_pair = layout["pair_q_index"][pk]
        if q in qidx_in_pair:
            position = int(np.where(qidx_in_pair == q)[0][0])
            l = position % L
            n_in_pair = position // L     # which (n, m) pivot pair
            return pk, l, n_in_pair
    return (-1, -1), -1, -1


if __name__ == "__main__":
    sys.exit(main())
