import os, sys, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read
import gap_dipole as gd, gap_bond, gap_ewald as ge
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index="0:50")
for a in test: a.calc = None
POL = dict(data_path="fit400_both.xyz", total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7,
           compress_mode="trivial", dipole_rc=4.0, dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
           dipole_species_pairs="symmetric", gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=4, gap_sigma=0.1)
with contextlib.redirect_stdout(io.StringIO()):
    m = gd.fit_gap_dipole(_cfg(**POL)); P1 = gd.predict(m, test)
S = gap_bond.Setup(_cfg(**POL)); cfg = S.cfg
pred = ge.fit_with_v(S, lambda a, ti: gd._v_atoms(a, ti, cfg)); P2 = pred(test)
print("RESULT production vs gap_bond path, 50 test frames: max|Δ|/max|P| =", f"{np.abs(P1 - P2).max() / np.abs(P1).max():.2e}", " rms P1", np.sqrt((P1**2).mean()).round(3), flush=True)
