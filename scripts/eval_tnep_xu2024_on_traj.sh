#!/bin/bash
# Reference polarisabilities for the Raman comparison: the Xu et al. 2024 TNEP water
# susceptibility model (Zenodo 10257363, trained on the same frames as datasets/*_waterbulk.xyz,
# with the susceptibility labels of datasets/*_waterbulk_chi.xyz) evaluated with GPUMD's `nep` in prediction mode on the first
# N_FRAMES frames (stride 2, dt 2 fs) of datasets/water_bulk_GPUMD_traj.xyz.
#   scripts/eval_tnep_xu2024_on_traj.sh [N_FRAMES=10000] [WORK=/tmp/tnep_eval]
# Output: plots/water_bulk_GPUMD_traj_polarizabilities_tnep_xu2024_seg<ps>ps.npy ([T, 6] Voigt, per cell).
set -e
N=${1:-10000}; W=${2:-/tmp/tnep_eval}; ROOT=$(cd "$(dirname "$0")/.." && pwd); NEP=${NEP:-$HOME/GPUMD/src/nep}
mkdir -p "$W" && cd "$W"
[ -f nep.txt ] || curl -sL -o nep.txt "https://zenodo.org/records/10257363/files/nep_susceptibility_water_bulk.txt?download=1"
python - "$ROOT" "$N" <<'PY'
import sys, numpy as np
from ase.io import iread, write
root, n = sys.argv[1], int(sys.argv[2]); buf = []
with open("train.xyz", "w") as fh:
    for k, at in enumerate(iread(f"{root}/datasets/water_bulk_GPUMD_traj.xyz", index=":")):
        if k % 2: continue
        at.calc = None; at.info = {"pol": np.zeros(9)}; buf.append(at)           # dummy label: prediction mode only reads it
        if len(buf) == 500: write(fh, buf, format="extxyz"); buf = []
        if (k // 2) + 1 >= n: break
    if buf: write(fh, buf, format="extxyz")
PY
CH=$((2500 * 650)); split -l $CH -d -a 2 train.xyz chunk_                # 2500 frames × (648 + 2) lines per chunk keeps the GPU set small
: > pol_all.out
for c in chunk_*; do
  mkdir -p "run_$c" && mv "$c" "run_$c/train.xyz" && cp nep.txt "run_$c/"
  printf "type 2 O H\nmode 2\nprediction 1\ncutoff 6 4\nn_max 6 6\nbasis_size 10 10\nl_max 4 2 1\nneuron 10\nbatch 2500\n" > "run_$c/nep.in"
  (cd "run_$c" && "$NEP" > nep.log 2>&1) && cat "run_$c/polarizability_train.out" >> pol_all.out
done
python - "$ROOT" "$N" <<'PY'
import sys, numpy as np
root, n = sys.argv[1], int(sys.argv[2])
P = np.loadtxt("pol_all.out")[:, :6] * 648                               # nep writes the per-atom mean; frames are 648 atoms
np.save(f"{root}/plots/water_bulk_GPUMD_traj_polarizabilities_tnep_xu2024_seg{n * 2 // 1000}ps.npy", P)
print("saved", P.shape)
PY
