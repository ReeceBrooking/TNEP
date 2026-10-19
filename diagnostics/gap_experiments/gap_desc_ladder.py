import os, sys, time, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]; write("fit400_dip.xyz", train[:400])
mu = lambda fr: np.stack([a.info["dipole"] for a in fr]); Mval, Mte = mu(val), mu(test)
def rr(P, T): r = np.sqrt(((P - T) ** 2).mean()); return r, 100 * r / T.std()
BASE = dict(total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0, dipole_channels=10,
            dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
            gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0)
DESC = {"l4a4_trivial": dict(l_max=4, alpha_max=4, compress_mode="trivial"),
        "l7a7_trivial": dict(l_max=7, alpha_max=7, compress_mode="trivial"),
        "l7a7_none": dict(l_max=7, alpha_max=7, compress_mode="none")}
def run(tag, path, frames, T, **kw):
    cfg = _cfg(data_path=path, **{**BASE, **kw}); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gap_dipole.fit_gap_dipole(cfg); P = gap_dipole.predict(m, frames)
    r, p = rr(P, T); print(f"RESULT {tag} {kw.get('_name','')} M={kw['gap_sparse_per_species']} sigma={kw['gap_sigma']}: RMSE/comp {r:.4f} RRMSE {p:.2f}% ({time.time()-t0:.0f}s)", flush=True); return p
best = None
for name, d in DESC.items():
    res = []
    for s in (3e-4, 1e-3, 3e-3):
        kw = dict(d, gap_sparse_per_species=250, gap_sigma=s, _name=name)
        kw2 = {k: v for k, v in kw.items() if k != "_name"}
        p = run("val", "fit400_dip.xyz", val, Mval, **kw2, _name=name) if False else None
        p = run(f"val {name}", "fit400_dip.xyz", val, Mval, **kw2); res.append((p, s))
    p, s = min(res); print(f"BEST {name}: sigma={s} (val {p:.2f}%)", flush=True)
    pt = run(f"TEST {name}", "/home/reece/TNEP/datasets/train_waterbulk.xyz", test, Mte, **{k: v for k, v in dict(d, gap_sparse_per_species=250, gap_sigma=s).items()})
    if best is None or p < best[0]: best = (p, name, s)
_, name, s = best
print(f"BEST DESC {name}", flush=True)
for M in (500,):
    for s2 in (s / 3, s):
        run(f"val {name}", "fit400_dip.xyz", val, Mval, **dict(DESC[name], gap_sparse_per_species=M, gap_sigma=s2))
    run(f"TEST {name}", "/home/reece/TNEP/datasets/train_waterbulk.xyz", test, Mte, **dict(DESC[name], gap_sparse_per_species=M, gap_sigma=s))
