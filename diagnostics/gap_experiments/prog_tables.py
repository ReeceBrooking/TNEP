"""Markdown tables from the gap_scan logs for a rows file (as the parallel driver uses):
per row the σ-selected (val) stage-A numbers and the fit-500 test result.
    python diagnostics/gap_experiments/prog_tables.py out/rows_round1.txt [out/rows_round2_new.txt ...]"""
import re, sys, os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
A = re.compile(r"RESULT (\w+) (.*) sigma=([\d.e-]+): val ([\d./]+)%? test ([\d./]+)%? ncols=(\d+)")
F = re.compile(r"FINAL (\w+) (.*) sigma=([\d.e-]+): val [\d./]+%? test ([\d./]+)%? ncols=(\d+)")
def score(v):                                           # "1.64" or "15.1/17.6" → selection score
    return sum(float(x) for x in v.split("/"))
logs = {p: open(f"{OUT}/prog_{p}.log").read().splitlines() for p in ("mu", "pol")}
for rows_file in sys.argv[1:]:
    rows = [l.rstrip("\n").split("\t") for l in open(rows_file) if l.strip()]
    for prop in ("mu", "pol"):
        want = [t for p, t in rows if p == prop]
        if not want:
            continue
        best, final = {}, {}
        for l in logs[prop]:
            m = A.match(l)
            if m and m.group(2) in want:
                k = m.group(2); sc = score(m.group(4))
                if k not in best or sc < best[k][0]:
                    best[k] = (sc, m.group(3), m.group(4), m.group(5), m.group(6))
            m = F.match(l)
            if m and m.group(2) in want:
                final[m.group(2)] = (m.group(4), m.group(5))
        unit = "RRMSE %" if prop == "mu" else "diag/off RRMSE %"
        print(f"\n**{os.path.basename(rows_file)} — {prop}** ({unit}; σ selected on val 400/100, then refit on 500)\n")
        print("| row | σ | val (400) | test (400) | test (500) | ncols |\n|---|---|---|---|---|---|")
        for k in want:
            b = best.get(k); f = final.get(k)
            print(f"| {k} | {b[1] if b else '—'} | {b[2] if b else '—'} | {b[3] if b else '—'} | "
                  f"{f[0] if f else '—'} | {f[1] if f else (b[4] if b else '—')} |")
