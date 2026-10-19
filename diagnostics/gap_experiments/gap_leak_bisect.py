import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read
import gap_dipole as gd
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); val = train[400:]
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
Tv = v6(val)
def rr(P, T):
    d = np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()); o = np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean())
    return f"{100 * d / T[:, :3].std():.1f}/{100 * o / T[:, 3:].std():.1f}"
POL = dict(data_path="fit400_both.xyz", total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7, compress_mode="trivial",
           dipole_rc=4.0, dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine", dipole_species_pairs="symmetric",
           gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=4, gap_sigma=0.1)
GAPK = dict(gap2b_rcut=3.5, gap3b_rcut=3.5, gap_buffer=0.5, gap2b_n_centres=20, gap2b_sigma=0.3, gap3b_n_centres=50)
def fit(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        m = gd.fit_gap_dipole(_cfg(**{**POL, **kw})); return gd.predict(m, val)
print("RESULT plain:", rr(fit(), Tv), flush=True)
print("RESULT +GAPK:", rr(fit(**GAPK), Tv), flush=True)
for k, v in GAPK.items():
    print(f"RESULT +{k}={v}:", rr(fit(**{k: v}), Tv), flush=True)
