"""Validation-selected GAP scans for the polarisability readout.
Fit on train frames 0:400, select on 400:500 (val), refit the winner on all
500, score on test. RRMSE = RMSE / std(reference) per the TNEP paper."""
import os, sys, time, itertools, json, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":")
test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for a in train: a.calc = None
write("fit400.xyz", train[:400]); val = train[400:]
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
Tval, Tte = v6(val), v6(test)
def rr(P, T):
    d = np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()); o = np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean())
    return 100 * d / T[:, :3].std(), 100 * o / T[:, 3:].std()
BASE = dict(total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7,
            compress_mode="trivial", dipole_rc=4.0, dipole_channels=10, dipole_radial="chebyshev_warped",
            dipole_cutoff="cosine", dipole_species_pairs="symmetric", gap_sparse_per_species=250,
            gap_kernel_power=2)
def run(tag, frames_path, score_frames, T, **kw):
    cfg = _cfg(data_path=frames_path, **{**BASE, **kw})
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gap_dipole.fit_gap_dipole(cfg); P = gap_dipole.predict(m, score_frames)
    d, o = rr(P, T)
    print(f"RESULT {tag} {json.dumps(kw)} diag {d:.1f}% off {o:.1f}% pooled {np.sqrt((d**2*3+o**2*3)/6):.1f}% ({time.time()-t0:.0f}s)", flush=True)
    return d, o
families = {}
# A. baseline + group delta + lambda_shear at rc 4
grid = [dict(gap_sigma=s, gap_group_delta=[1.0, dt, 1.0], gap_lambda_shear=ls)
        for s in (0.1, 0.3) for dt in (0.3, 1.0, 3.0) for ls in (1.0, 2.0)]
families["A_groups"] = grid
# B. 6 Å kernel + 6 Å kernel-only SOAP scale
families["B_range6"] = [dict(gap_sigma=s, dipole_rc=6.0, gap_extra_rcut=6.0) for s in (0.1, 0.3)]
# C. cross group
families["C_cross"] = [dict(gap_sigma=0.1, gap_cross_channels=cx, gap_group_delta=[1.0, 1.0, dx])
                       for cx in (2, 4) for dx in (0.3, 1.0, 3.0)]
best = {}
for fam, grid in families.items():
    res = []
    for kw in grid:
        try:
            d, o = run(f"val:{fam}", "fit400.xyz", val, Tval, **kw)
            res.append((np.sqrt((d**2 + o**2) / 2), kw))
        except Exception as e:
            print(f"RESULT val:{fam} {json.dumps(kw)} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
    if res:
        best[fam] = min(res, key=lambda t: t[0])[1]
        print(f"BEST {fam}: {json.dumps(best[fam])}", flush=True)
# D. combine the winners, then test-score each winner and the combination on all 500 train frames
combo = {}
for kw in best.values(): combo.update(kw)
print("COMBO", json.dumps(combo), flush=True)
for tag, kw in list(best.items()) + [("D_combo", combo)]:
    try:
        run(f"TEST:{tag}", "/home/reece/TNEP/datasets/train_waterbulk.xyz", test, Tte, **kw)
    except Exception as e:
        print(f"RESULT TEST:{tag} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
