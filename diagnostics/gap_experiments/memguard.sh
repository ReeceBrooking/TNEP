#!/bin/bash
# Kill the scan/trajectory jobs if MemAvailable < 2 GB (last line of defence against another WSL OOM).
cd /home/reece/TNEP
while true; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$a" -lt 2000000 ]; then
    for p in $(ps -eo pid,cmd | grep -E "[g]ap_progressive_scan|[r]aman_gap_gpumd" | awk '{print $1}'); do kill -9 $p; done
    echo "$(date +%T) MEMGUARD killed jobs at MemAvailable=${a} kB" >> diagnostics/gap_experiments/out/memguard.log
  fi
  sleep 3
done
