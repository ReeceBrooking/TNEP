#!/bin/bash
# Drive gap_triplet_tensor_scan.py one fit per process under an RSS watchdog (kill at 10 GB). Stage A: every row at σ 0.1 / 0.3;
# stage C: the best 3 rows by mean val RRMSE refit on 500.  Log: diagnostics/gap_experiments/out/triplet.log
cd /home/reece/TNEP; LOG=diagnostics/gap_experiments/out/triplet.log; : > $LOG
onefit() {  # tag sigma fit
  TT_ROW="$1" TT_SIGMA="$2" TT_FIT="$3" OMP_NUM_THREADS=8 python diagnostics/gap_experiments/gap_triplet_tensor_scan.py >> $LOG 2>&1 &
  local pid=$!; while kill -0 $pid 2>/dev/null; do r=$(ps -o rss= -p $pid 2>/dev/null | tr -d ' '); r=${r:-0}
    if [ "$r" -gt 10000000 ]; then echo "WATCHDOG kill $1 at ${r} kB" >> $LOG; kill -9 $pid; fi; sleep 5; done
}
ROWS=("2groups cross2 +tt M10 s0.5" "2groups cross2 +tt M20 s0.5" "2groups cross2 +tt M20 s1.0" "2groups cross2 +tt M20 pol2" \
      "2groups cross2 +tt M20 s0.5 +induced" "2groups cross2 +tt M20 s0.5 +ang L1")
for t in "${ROWS[@]}"; do for s in 0.1 0.3; do onefit "$t" $s 400; done; done
# select: mean of diag/off val per (row, σ); best σ per row; top 3 rows
python - "$LOG" <<'EOF2' > diagnostics/gap_experiments/out/triplet_finals.txt
import re, sys
best = {}
for l in open(sys.argv[1]):
    m = re.match(r"RESULT (.*) sigma=([\d.]+): val ([\d.]+)/([\d.]+)", l)
    if m:
        t, s, d, o = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)); sc = (d + o) / 2
        if t not in best or sc < best[t][0]: best[t] = (sc, s)
for t in sorted(best, key=lambda t: best[t][0])[:3]: print(f"{t}\t{best[t][1]}")
EOF2
while IFS=$'\t' read -r t s; do onefit "$t" "$s" 500; done < diagnostics/gap_experiments/out/triplet_finals.txt
echo DONE >> $LOG
