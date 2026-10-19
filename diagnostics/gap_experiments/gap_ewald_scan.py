import os, sys, time, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole, gap_bond, gap_ewald
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]; write("fit400_dip_e.xyz", train[:400])
mu = lambda fr: np.stack([a.info["dipole"] for a in fr]); Mval, Mte = mu(val), mu(test)
def rr(P, T): r = np.sqrt(((P - T) ** 2).mean()); return 100 * r / T.std()
cfg = _cfg(data_path="fit400_dip_e.xyz", total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0,
           dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
           gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250,
           l_max=7, alpha_max=7, compress_mode="none", gap_sigma=1e-3)
t0 = time.time(); S = gap_bond.Setup(cfg); print(f"SETUP l7a7_none ({time.time()-t0:.0f}s)", flush=True)
def report(tag, pred, t0):
    print(f"RESULT {tag}: val {rr(pred(val), Mval):.2f}% test {rr(pred(test), Mte):.2f}% ({time.time()-t0:.0f}s)", flush=True)
# ewald feature statistics (sanity)
f_full = gap_ewald.ewald_features(train[:20], None, cfg, False); f_lr = gap_ewald.ewald_features(train[:20], None, cfg, True)
print("EWALD rms |E| full", np.sqrt(np.mean([(E**2).mean() for _, E in f_full])).round(3), " lr-only", np.sqrt(np.mean([(E**2).mean() for _, E in f_lr])).round(3),
      "| std φ full", np.std(np.concatenate([p for p, _ in f_full])).round(3), " lr-only", np.std(np.concatenate([p for p, _ in f_lr])).round(3), flush=True)
for lr in (False, True):
    for dE in (0.3, 1.0, 3.0):
        t0 = time.time()
        try: report(f"FIELD lr_only={lr} delta_E={dE}", gap_ewald.fit_field_column(S, lr, dE), t0)
        except Exception as e: print(f"RESULT FIELD lr_only={lr} delta_E={dE} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
for lr in (False, True):
    for ell in (0.5, 1.0, 2.0):
        t0 = time.time()
        try: report(f"PHIKERNEL lr_only={lr} ell_factor={ell}", gap_ewald.fit_potential_kernel(S, lr, ell), t0)
        except Exception as e: print(f"RESULT PHIKERNEL lr_only={lr} ell_factor={ell} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
t0 = time.time()
try: report("BOND M_b=250 delta_b=3.0", gap_bond.fit_bond_kernel(S, 250, 3.0), t0)
except Exception as e: print(f"RESULT BOND delta_b=3 FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
