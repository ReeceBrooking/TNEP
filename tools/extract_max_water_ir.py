"""Extract Max & Chapados 2009 liquid-water IR optical constants from a
copy-pasted text dump into a clean CSV.

Source: Max, J.-J.; Chapados, C. J. Chem. Phys. 131, 184505 (2009).

The input text is the paper's optical-constants table copied directly out
of the PDF, so it carries layout artefacts:
  - CRLF line endings (Windows)
  - "Page N" headers between table chunks
  - One blank line per page break
The body is 5 space-separated columns: w-number(cm-1), n(H2O), k(H2O),
n(D2O), k(D2O).

This script:
  1. Streams the input, dropping blank and "Page N" lines.
  2. Parses 5-column rows into floats (accepting scientific notation).
  3. Verifies the wavenumber column is strictly increasing and has no
     gaps that would indicate a missing chunk between pages.
  4. Writes a clean CSV ready for use by the IR-overlay plotting code.

Usage:
    python tools/extract_max_water_ir.py [INPUT] [-o OUTPUT]
                                          [--max-step-tolerance FACTOR]

Defaults: reads "Liquid Water IR Max et al.txt" in the project root,
writes "max_water_ir.csv" alongside it. The gap-tolerance factor is
applied as `max_allowed_step = median_step * factor` (default 5×) —
any single jump larger than that is flagged as a likely missing chunk.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HEADER = ["wavenumber_cm1", "n_H2O", "k_H2O", "n_D2O", "k_D2O"]


def is_skippable(line: str) -> bool:
    """True for lines that should be dropped before parsing."""
    s = line.strip()
    if not s:
        return True                                   # blank (CRLF-safe)
    if s.lower().startswith("page"):
        return True                                   # "Page N" header
    return False


def looks_like_header(tokens: list[str]) -> bool:
    """Detect the column-label row (first non-skip line in the dump).

    The header has alphabetic content; data rows are all numeric. Checking
    the first token avoids depending on the exact label spelling.
    """
    try:
        float(tokens[0])
        return False
    except (ValueError, IndexError):
        return True


def parse_input(path: Path) -> tuple[list[tuple[float, ...]], dict]:
    """Stream the file → list of 5-tuples + a stats dict.

    Stats:
        n_blank, n_page, n_header, n_data : line counts by category
        bad_lines                          : list of (line_no, raw, reason)
    """
    rows: list[tuple[float, ...]] = []
    stats = {"n_blank": 0, "n_page": 0, "n_header": 0, "n_data": 0,
             "bad_lines": []}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                stats["n_blank"] += 1
                continue
            if stripped.lower().startswith("page"):
                stats["n_page"] += 1
                continue
            tokens = stripped.split()
            if looks_like_header(tokens):
                stats["n_header"] += 1
                continue
            if len(tokens) != 5:
                stats["bad_lines"].append(
                    (line_no, raw.rstrip(), f"expected 5 columns, got {len(tokens)}"))
                continue
            try:
                row = tuple(float(t) for t in tokens)
            except ValueError as exc:
                stats["bad_lines"].append(
                    (line_no, raw.rstrip(), f"float parse error: {exc}"))
                continue
            rows.append(row)
            stats["n_data"] += 1
    return rows, stats


def check_no_gaps(rows: list[tuple[float, ...]],
                   max_step_factor: float = 5.0) -> dict:
    """Flag jumps in the wavenumber column that look like missing chunks.

    The Max & Chapados step is approximately uniform (~0.96 cm⁻¹). Any
    single jump significantly larger than the median step is suspect —
    almost certainly a page-break boundary where some lines were lost.

    Returns:
        dict with:
            median_step      : median Δν̃ over consecutive rows
            max_step         : largest Δν̃ found
            max_step_at_row  : 0-based row index where max_step ended
            n_jumps          : count of Δν̃ > median_step * max_step_factor
            jumps            : list of (row_idx, w_lo, w_hi, step) for each jump
            non_monotone_idx : list of row indices where Δν̃ ≤ 0 (data error)
    """
    if len(rows) < 2:
        return {"median_step": float("nan"), "max_step": float("nan"),
                "max_step_at_row": -1, "n_jumps": 0, "jumps": [],
                "non_monotone_idx": []}
    import statistics as st
    w = [r[0] for r in rows]
    steps = [w[i + 1] - w[i] for i in range(len(w) - 1)]
    median_step = float(st.median(steps))
    max_step = max(steps)
    max_step_at_row = steps.index(max_step)
    threshold = median_step * max_step_factor
    jumps = [(i, w[i], w[i + 1], steps[i])
             for i, step in enumerate(steps) if step > threshold]
    non_monotone = [i for i, step in enumerate(steps) if step <= 0.0]
    return {"median_step": median_step,
            "max_step": float(max_step),
            "max_step_at_row": int(max_step_at_row),
            "n_jumps": len(jumps),
            "jumps": jumps,
            "non_monotone_idx": non_monotone}


def write_csv(rows: list[tuple[float, ...]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row)


def emit_report(path_in: Path, path_out: Path,
                rows: list[tuple[float, ...]],
                parse_stats: dict, gap_stats: dict,
                max_step_factor: float) -> bool:
    """Print a human-readable report. Returns True iff the data look clean."""
    print(f"\nInput  : {path_in}")
    print(f"Output : {path_out}")
    print(f"\nLine counts:")
    print(f"  data rows    : {parse_stats['n_data']}")
    print(f"  blank lines  : {parse_stats['n_blank']}")
    print(f"  'Page' lines : {parse_stats['n_page']}")
    print(f"  header lines : {parse_stats['n_header']}")
    print(f"  bad lines    : {len(parse_stats['bad_lines'])}")
    for ln, raw, reason in parse_stats["bad_lines"][:5]:
        print(f"     line {ln}: {raw!r}  ({reason})")
    if len(parse_stats["bad_lines"]) > 5:
        print(f"     ... and {len(parse_stats['bad_lines']) - 5} more")

    if not rows:
        print("\nNo data rows parsed — nothing to check.")
        return False

    print(f"\nWavenumber range:")
    print(f"  first  : {rows[0][0]} cm⁻¹")
    print(f"  last   : {rows[-1][0]} cm⁻¹")
    print(f"  count  : {len(rows)}")

    print(f"\nStep-size check (gap detection):")
    print(f"  median step           : {gap_stats['median_step']:.6f} cm⁻¹")
    print(f"  largest step          : {gap_stats['max_step']:.6f} cm⁻¹")
    if gap_stats["max_step_at_row"] >= 0:
        i = gap_stats["max_step_at_row"]
        w_lo = rows[i][0]
        w_hi = rows[i + 1][0]
        print(f"  ... between rows {i} and {i + 1}: "
              f"{w_lo} → {w_hi}")
    print(f"  threshold (× median)  : "
          f"{gap_stats['median_step'] * max_step_factor:.6f} cm⁻¹")
    print(f"  jumps above threshold : {gap_stats['n_jumps']}")
    if gap_stats["jumps"]:
        print("    suspect jumps (likely missing rows at page boundary):")
        for idx, w_lo, w_hi, step in gap_stats["jumps"][:10]:
            print(f"      row {idx}: {w_lo} → {w_hi}  (Δ = {step:.4f})")
        if len(gap_stats["jumps"]) > 10:
            print(f"      ... and {len(gap_stats['jumps']) - 10} more")
    if gap_stats["non_monotone_idx"]:
        print(f"  NON-MONOTONE rows     : {len(gap_stats['non_monotone_idx'])}")
        print("    (wavenumber repeats or decreases — file order is wrong)")

    clean = (gap_stats["n_jumps"] == 0
             and not gap_stats["non_monotone_idx"]
             and not parse_stats["bad_lines"])
    if clean:
        print("\n✓ No gaps, no non-monotone rows, no bad lines. CSV is clean.")
    else:
        print("\n⚠ Issues detected — see above. CSV was written anyway.")
    return clean


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "input", nargs="?",
        default="Liquid Water IR Max et al.txt",
        help="Input text file (default: 'Liquid Water IR Max et al.txt' "
             "in the project root).")
    parser.add_argument(
        "-o", "--output", default="max_water_ir.csv",
        help="Output CSV filename (default: max_water_ir.csv).")
    parser.add_argument(
        "--max-step-tolerance", type=float, default=5.0,
        help="A Δν̃ jump larger than median_step × FACTOR is flagged as a "
             "likely missing chunk. Default 5.0.")
    args = parser.parse_args()

    path_in = Path(args.input)
    if not path_in.is_absolute():
        path_in = Path(__file__).resolve().parent.parent / path_in
    path_out = Path(args.output)
    if not path_out.is_absolute():
        path_out = path_in.parent / path_out

    rows, parse_stats = parse_input(path_in)
    gap_stats = check_no_gaps(rows, max_step_factor=args.max_step_tolerance)
    write_csv(rows, path_out)
    clean = emit_report(path_in, path_out, rows, parse_stats, gap_stats,
                        args.max_step_tolerance)
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
