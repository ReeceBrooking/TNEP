import os, sys, time, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); val = train[400:]
test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for a in train: a.calc = None
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
            gap_kernel_power=2, gap_cross_channels=4)
for n in (100, 200, 400):
    write(f"fit{n}_pol.xyz", train[:n])
    res = []
    for s in (0.1, 0.3):
        cfg = _cfg(data_path=f"fit{n}_pol.xyz", **{**BASE, "gap_sigma": s})
        with contextlib.redirect_stdout(io.StringIO()):
            m = gap_dipole.fit_gap_dipole(cfg)
            dv, ov = rr(gap_dipole.predict(m, val), Tval); dt, ot = rr(gap_dipole.predict(m, test), Tte)
        print(f"RESULT N={n} sigma={s}: val diag {dv:.1f}% off {ov:.1f}% | test diag {dt:.1f}% off {ot:.1f}%", flush=True)
        res.append((np.sqrt((dv**2 + ov**2) / 2), s, dt, ot))
    _, s_best, dt, ot = min(res)
    print(f"BEST N={n}: sigma={s_best} → test diag {dt:.1f}% off {ot:.1f}%", flush=True)
