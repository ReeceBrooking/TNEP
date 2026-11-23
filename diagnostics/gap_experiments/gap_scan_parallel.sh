#!/bin/bash
# Run gap_scan.py rows TWO AT A TIME (each fit ≤ ~6 GB; the WSL has 23 GB; memguard.sh must be running).
# Usage: gap_scan_parallel.sh <rows file> [finals_only]   rows file lines: "<prop>\t<tag>"; σ grid per prop below.
# Stage A: every row × σ on fit 400 → val/test, appended to out/prog_<prop>.log; stage C: best σ per row (val) refit on 500.
cd /home/reece/TNEP; ROWSF=$1; FINALS_ONLY=$2; OUT=diagnostics/gap_experiments/out; MAXJ=${MAXJ:-2}; WD=${WD:-7000000}  # env overrides: MAXJ=1 WD=12000000 for the >7 GB rows
sig() { [ -n "$SIG" ] && echo "$SIG" && return; [ "$1" = mu ] && echo "0.001 0.003" || echo "0.03 0.1 0.3"; }  # SIG="0.3" overrides the grid
onefit() {  # prop tag sigma fit  (foreground; RSS watchdog 7 GB)
  PS_PROP=$1 PS_ROW="$2" PS_SIGMA="$3" PS_FIT="$4" OMP_NUM_THREADS=6 python diagnostics/gap_experiments/gap_scan.py >> $OUT/prog_$1.log 2>&1 &
  local pid=$!; while kill -0 $pid 2>/dev/null; do r=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' '); r=${r:-0}
    if [ "$r" -gt $WD ]; then echo "WATCHDOG kill $2 at ${r} kB" >> $OUT/prog_$1.log; kill -9 $pid; fi; sleep 5; done
}
runq() {  # read "prop\ttag\tsigma\tfit" lines from stdin, MAXJ at a time
  while IFS=$'\t' read -r p t s f; do
    while [ "$(jobs -rp | wc -l)" -ge $MAXJ ]; do sleep 5; done
    onefit "$p" "$t" "$s" "$f" &
  done; wait
}
if [ -z "$FINALS_ONLY" ]; then
  while IFS=$'\t' read -r p t; do for s in $(sig $p); do printf "%s\t%s\t%s\t400\n" "$p" "$t" "$s"; done; done < "$ROWSF" | runq
fi
python - "$ROWSF" "$OUT" <<'EOF2' > $OUT/finals_queue.txt
import re, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
for prop in sorted({p for p, _ in rows}):
    want = {t for p, t in rows if p == prop}; best = {}
    for l in open(f"{sys.argv[2]}/prog_{prop}.log"):
        m = re.match(r"RESULT \w+ (.*) sigma=([\d.e-]+): val ([\d.]+)(?:/([\d.]+))?", l)
        if m and m.group(1) in want:
            sc = float(m.group(3)) + (float(m.group(4)) if m.group(4) else 0.0)
            if m.group(1) not in best or sc < best[m.group(1)][0]: best[m.group(1)] = (sc, m.group(2))
    for t in want:
        if t in best: print(f"{prop}\t{t}\t{best[t][1]}\t500")
EOF2
runq < $OUT/finals_queue.txt
echo "DONE $(basename $ROWSF)" >> $OUT/prog_mu.log; echo "DONE $(basename $ROWSF)" >> $OUT/prog_pol.log
