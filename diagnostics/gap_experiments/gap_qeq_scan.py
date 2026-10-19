import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_bond, gap_dipole as gd, gap_ewald as ge
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
def mk(path): return _cfg(data_path=path, total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0,
    dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
    gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250,
    l_max=7, alpha_max=7, compress_mode="none", gap_sigma=1e-3)
S = gap_bond.Setup(mk("fit400_dip_e.xyz")); cfg = S.cfg; print("SETUP done", flush=True)
cache = {}
def qeq(frames, tis, H):
    out = []
    for a, ti in zip(frames, tis):
        k = (id(a), H)
        if k not in cache: cache[k] = ge.qeq_parts(a, ti, cfg, hardness=H)[1:]
        out.append(cache[k])
    return out
field = lambda frames, tis: [E[:, :, None] for _, E in ge.ewald_features(frames, tis, cfg, False)]
vbase = lambda a, ti: gd._v_atoms(a, ti, cfg)
def report(tag, fn):
    t0 = time.time()
    try:
        pred = fn(); print(f"RESULT {tag}: val {rr(pred(val), Mval):.2f}% test {rr(pred(test), Mte):.2f}% ncols={pred.n_columns} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"RESULT {tag} FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
for H in (5.0, 10.0, 20.0):
    report(f"CHI H={H}", lambda H=H: ge.fit_with_v(S, vbase, lambda fr, tis, H=H: [Vq[:, :, None] for Vq, _ in qeq(fr, tis, H)]))
report("CHI H=10 + FIELD", lambda: ge.fit_with_v(S, vbase, lambda fr, tis: [np.concatenate([Vq[:, :, None], E], 2) for (Vq, _), E in zip(qeq(fr, tis, 10.0), field(fr, tis))]))
for H in (10.0, 40.0):
    vpol = lambda a, ti, H=H: ge.v_polarised(a, ti, cfg, ge.qeq_parts(a, ti, cfg, hardness=H)[2])
    report(f"POLARISE H={H}", lambda vpol=vpol: ge.fit_with_v(S, vpol))
    report(f"POLARISE H={H} + FIELD", lambda vpol=vpol: ge.fit_with_v(S, vpol, field))
