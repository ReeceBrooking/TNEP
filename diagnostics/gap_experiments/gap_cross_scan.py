import os, sys, time, json, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read
import gap_dipole
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); val = train[400:]
test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
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
def run(tag, path, frames, T, **kw):
    cfg = _cfg(data_path=path, **{**BASE, **kw}); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gap_dipole.fit_gap_dipole(cfg); P = gap_dipole.predict(m, frames)
    d, o = rr(P, T)
    print(f"RESULT {tag} Cx={kw['gap_cross_channels']} sigma={kw['gap_sigma']} ncols/sparse={gap_dipole._n_vcols(cfg)}: diag {d:.1f}% off {o:.1f}% pooled {np.sqrt((d**2+o**2)/2):.1f}% ({time.time()-t0:.0f}s)", flush=True)
    return np.sqrt((d ** 2 + o ** 2) / 2)
for cx in (4, 6, 8, 10):
    res = []
    for s in (0.03, 0.1, 0.3):
        try:
            res.append((run("val", "fit400.xyz", val, Tval, gap_cross_channels=cx, gap_sigma=s), s))
        except Exception as e:
            print(f"RESULT val Cx={cx} sigma={s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
    if res:
        s_best = min(res)[1]; print(f"BEST Cx={cx}: sigma={s_best}", flush=True)
        try:
            run("TEST", "/home/reece/TNEP/datasets/train_waterbulk.xyz", test, Tte, gap_cross_channels=cx, gap_sigma=s_best)
        except Exception as e:
            print(f"RESULT TEST Cx={cx} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
