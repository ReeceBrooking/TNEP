import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read
import gap_bond, gap_ewald as ge
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]
mu = lambda fr: np.stack([a.info["dipole"] for a in fr]); Mval, Mte = mu(val), mu(test)
def rr(P, T): r = np.sqrt(((P - T) ** 2).mean()); return 100 * r / T.std()
# sanity: dipole field of one dipole vs analytic point-dipole field in a big box
a0 = train[0]
from ase import Atoms
big = Atoms(positions=[[20, 20, 20], [23, 20, 20]], cell=60 * np.eye(3), pbc=True)
p = np.zeros((2, 3)); p[0] = [0, 0, 0.5]
E = ge.dipole_field(big, p)
r = np.array([3.0, 0, 0]); analytic = (3 * r * (r @ p[0]) / 3.0 ** 5 - p[0] / 3.0 ** 3)
print("DIPOLE-FIELD check at 3 Å: numeric", np.round(E[1], 6), "analytic", np.round(analytic, 6), " self-residual on source", np.round(E[0], 5), flush=True)
cfg = _cfg(data_path="fit400_dip_e.xyz", total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0,
           dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
           gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250,
           l_max=7, alpha_max=7, compress_mode="none", gap_sigma=1e-3)
S = gap_bond.Setup(cfg); print("SETUP done", flush=True)
for split in (True,):
    for mix in (1.0, 0.5):
        t0 = time.time()
        preds = ge.fit_polarise(S, split=split, n_iter=2, mix=mix)
        for k, pred in enumerate(preds):
            print(f"RESULT POLARISE split={split} mix={mix} stage={k}: val {rr(pred(val), Mval):.2f}% test {rr(pred(test), Mte):.2f}% ({time.time()-t0:.0f}s)", flush=True)
