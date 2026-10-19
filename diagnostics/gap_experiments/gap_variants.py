import os, sys, time, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole, gap_bond
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]; write("fit400_dip_v.xyz", train[:400])
mu = lambda fr: np.stack([a.info["dipole"] for a in fr]); Mval, Mte = mu(val), mu(test)
def rr(P, T): r = np.sqrt(((P - T) ** 2).mean()); return 100 * r / T.std()
BASE = dict(data_path="fit400_dip_v.xyz", total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0,
            dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
            gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250)
for name, d in {"l4a4_trivial": dict(l_max=4, alpha_max=4, compress_mode="trivial"),
                "l7a7_none": dict(l_max=7, alpha_max=7, compress_mode="none")}.items():
    for sigma in (1e-3,):
        cfg = _cfg(**BASE, **d, gap_sigma=sigma)
        t0 = time.time(); S = gap_bond.Setup(cfg); print(f"SETUP {name} sigma={sigma} n_base={S.n_base} ({time.time()-t0:.0f}s)", flush=True)
        # sanity: production fit vs ALS iter0 (b ≡ 1) must agree
        with contextlib.redirect_stdout(io.StringIO()):
            m = gap_dipole.fit_gap_dipole(_cfg(**BASE, **d, gap_sigma=sigma)); Pprod = gap_dipole.predict(m, val)
        def score(pred, tag, _S=[None]):
            Pv = pred(val); pv = rr(Pv, Mval); pt = rr(pred(test), Mte)
            extra = f" |ALS0−prod|max={np.abs(Pv - Pprod).max():.2e}" if tag.startswith("iter0") else ""
            print(f"RESULT {name} ALS {tag}: val {pv:.2f}% test {pt:.2f}%{extra}", flush=True); return pv
        print(f"RESULT {name} production: val {rr(Pprod, Mval):.2f}%", flush=True)
        t0 = time.time(); gap_bond.fit_bilinear_als(S, n_iter=3, score=score); print(f"ALS done ({time.time()-t0:.0f}s)", flush=True)
        for M_b in (250, 500):
            for db in (0.3, 1.0):
                t0 = time.time()
                try:
                    pred = gap_bond.fit_bond_kernel(S, M_b, delta_b=db)
                    print(f"RESULT {name} BOND M_b={M_b} delta_b={db} ncols={pred.n_columns}: val {rr(pred(val), Mval):.2f}% test {rr(pred(test), Mte):.2f}% ({time.time()-t0:.0f}s)", flush=True)
                except Exception as e:
                    print(f"RESULT {name} BOND M_b={M_b} delta_b={db} FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
