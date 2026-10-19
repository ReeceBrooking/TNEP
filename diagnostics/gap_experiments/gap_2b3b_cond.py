"""2b/3b TurboGAP blocks in the production GAP fit: dipole (with the field column) and polarisability (two-group + cross)."""
import os, sys, time, json, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole as gd
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]; write("fit400_cond.xyz", train[:400])
SPCE = {8: -0.8476, 1: 0.4238}
mu = lambda fr: np.stack([a.info["dipole"] for a in fr])
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
def rr_mu(P, T): r = np.sqrt(((P - T) ** 2).mean()); return f"{100 * r / T.std():.2f}%"
def rr_pol(P, T):
    d = np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()); o = np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean())
    return f"{100 * d / T[:, :3].std():.1f}/{100 * o / T[:, 3:].std():.1f}"
MU = dict(data_path="fit400_cond.xyz", total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0, dipole_channels=10,
          dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric", gap_kernel_power=2,
          convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250, l_max=7, alpha_max=7,
          compress_mode="none", gap_sigma=1e-3, dipole_field=True, dipole_field_charges=SPCE, dipole_field_scale=None)
POL = dict(data_path="fit400_cond.xyz", total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7,
           compress_mode="trivial", dipole_rc=4.0, dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
           dipole_species_pairs="symmetric", gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=4, gap_sigma=0.1)
GAPK = dict(gap2b_rcut=3.5, gap3b_rcut=3.5, gap_buffer=0.5, gap2b_n_centres=20, gap2b_sigma=0.3, gap3b_n_centres=50)
def run(tag, base, rr, Tv, Tt, **kw):
    cfg = _cfg(**{**base, **GAPK, **kw}); t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = gd.fit_gap_dipole(cfg); Pv = gd.predict(m, val); Pt = gd.predict(m, test)
        rec = json.load(open(sorted(__import__("glob").glob("GAP/*.json"), key=os.path.getmtime)[-1]))
        print(f"RESULT {tag}: val {rr(Pv, Tv)} test {rr(Pt, Tt)} ncols={rec['gap']['n_columns']} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"RESULT {tag} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
Mv, Mt, Pv_, Pt_ = mu(val), mu(test), v6(val), v6(test)
print("=== conditioning: jitter and centre spacing (mu, field on, sigma 1e-3)", flush=True)
for jit in (1e-6, 1e-4):
    run(f"mu +2b delta=1 jitter={jit}", MU, rr_mu, Mv, Mt, gap_2b=True, gap_sparse_jitter=jit)
    run(f"mu +3b exp 0.5 delta_3b=0.1 jitter={jit}", MU, rr_mu, Mv, Mt, gap_3b=True, gap3b_kernel="exp", gap3b_sigma=[0.5] * 3, gap_delta_3b=0.1, gap_sparse_jitter=jit)
run("mu +2b M2=8 sigma2=0.5 (well-conditioned)", MU, rr_mu, Mv, Mt, gap_2b=True, gap2b_n_centres=8, gap2b_sigma=0.5)
run("mu +3b M3=20 exp sigma 1.0 delta_3b=0.1", MU, rr_mu, Mv, Mt, gap_3b=True, gap3b_kernel="exp", gap3b_sigma=[1.0] * 3, gap3b_n_centres=20, gap_delta_3b=0.1)
run("pol +2b M2=8 sigma2=0.5 sigma=0.3", POL, rr_pol, Pv_, Pt_, gap_2b=True, gap2b_n_centres=8, gap2b_sigma=0.5, gap_sigma=0.3)
