"""Site-model decomposition (soap / 2b / 3b summed kernels, gap_site_models) in the production GAP fit:
dipole (field column) and polarisability (two-group + cross). argv: mu | pol — run as separate processes.
Stage 1: fit 400, score val for every (row, σ, variant); stage 2: refit the val-best setting of each row on 500, score test.
SITE_SMOKE=1: 20 train frames, one σ, two rows (config validation only)."""
import os, sys, time, json, glob, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole as gd
from test_gap_dipole import _cfg
prop = sys.argv[1]; assert prop in ("mu", "pol")
SMOKE = os.environ.get("SITE_SMOKE") == "1"
os.makedirs(f"diagnostics/gap_experiments/out/site_{prop}", exist_ok=True)
os.chdir(f"diagnostics/gap_experiments/out/site_{prop}")          # GAP/ record dir is per process
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]
F400, F500 = f"fit400_site_{prop}.xyz", f"fit500_site_{prop}.xyz"
write(F400, train[:20] if SMOKE else train[:400]); write(F500, train)
SPCE = {8: -0.8476, 1: 0.4238}
mu = lambda fr: np.stack([a.info["dipole"] for a in fr])
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
def rr_mu(P, T):
    r = 100 * np.sqrt(((P - T) ** 2).mean()) / T.std(); return r, f"{r:.2f}%"
def rr_pol(P, T):
    d = 100 * np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()) / T[:, :3].std()
    o = 100 * np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean()) / T[:, 3:].std()
    return (d + o) / 2, f"{d:.1f}/{o:.1f}"
MU = dict(data_path=F400, total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0, dipole_channels=10,
          dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric", gap_kernel_power=2,
          convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250, l_max=7, alpha_max=7,
          compress_mode="none", gap_sigma=1e-3, dipole_field=True, dipole_field_charges=SPCE, dipole_field_scale=None)
POL = dict(data_path=F400, total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7,
           compress_mode="trivial", dipole_rc=4.0, dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
           dipole_species_pairs="symmetric", gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=4, gap_sigma=0.1)
SITE = dict(gap_site_channels=10, gap_site_lmax=2)
# rows: (tag, fixed kwargs, variants); variant keys d2b/d3b become gap_model_delta entries, the rest pass through.
X3 = dict(gap_cross_channels=4, gap_cross_lmax=1)                   # μ base has no cross group; 3b needs one
ROWS = {
    "mu": [("soap", dict(gap_site_models=["soap"]), [{}]),
           ("2b", dict(gap_site_models=["2b"]), [{}]),
           ("soap+2b", dict(gap_site_models=["soap", "2b"]), [dict(d2b=0.3), dict(d2b=1.0)]),
           ("3b", dict(gap_site_models=["3b"], gap_cross_channels=4), [dict(gap_cross_lmax=1), dict(gap_cross_lmax=2)]),
           ("soap+3b", dict(gap_site_models=["soap", "3b"], **X3), [dict(d3b=0.3), dict(d3b=1.0)]),
           ("soap+2b+3b", dict(gap_site_models=["soap", "2b", "3b"], **X3), [dict(d2b=0.3, d3b=0.3), dict(d2b=1.0, d3b=1.0)])],
    "pol": [("soap", dict(gap_site_models=["soap"]), [{}]),
            ("2b", dict(gap_site_models=["2b"]), [{}]),
            ("soap+2b", dict(gap_site_models=["soap", "2b"]), [dict(d2b=0.3), dict(d2b=1.0)]),
            ("3b", dict(gap_site_models=["3b"]), [dict(gap_cross_lmax=1), dict(gap_cross_lmax=2)]),
            ("soap+3b", dict(gap_site_models=["soap", "3b"]), [dict(gap_cross_lmax=0), dict(gap_cross_lmax=1)]),
            ("soap+2b+3b", dict(gap_site_models=["soap", "2b", "3b"]), [dict(d2b=0.3, d3b=0.3), dict(d2b=1.0, d3b=1.0)])],
}[prop]
SIGMAS = {"mu": (1e-3, 3e-3), "pol": (0.1, 0.3)}[prop]
if SMOKE: ROWS = [ROWS[0], ROWS[-1]]; SIGMAS = SIGMAS[:1]
base, rr, Tv, Tt = (MU, rr_mu, mu(val), mu(test)) if prop == "mu" else (POL, rr_pol, v6(val), v6(test))
def kwargs(fixed, var, sigma, data_path):
    v = dict(var); delta = {"soap": 1.0, "2b": v.pop("d2b", 1.0), "3b": v.pop("d3b", 1.0)}
    return {**base, **SITE, **fixed, **v, "gap_model_delta": delta, "gap_sigma": sigma, "data_path": data_path}
def fit(kw):
    """(val score, val str, test str, ncols, seconds) — the latest GAP/*.json is this fit's record (per-process dir)."""
    cfg = _cfg(**kw); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gd.fit_gap_dipole(cfg); Pv = gd.predict(m, val); Pt = gd.predict(m, test)
    rec = json.load(open(sorted(glob.glob("GAP/*.json"), key=os.path.getmtime)[-1]))
    sv, vs = rr(Pv, Tv); _, ts = rr(Pt, Tt)
    return sv, vs, ts, rec["gap"]["n_columns"], time.time() - t0
def vstr(var): return " ".join(f"{k}={v}" for k, v in var.items()) or "-"
print(f"=== {prop.upper()} site models: stage 1 (fit 400, val select)", flush=True)
best = {}
for tag, fixed, variants in ROWS:
    for var in variants:
        for s in SIGMAS:
            try:
                sv, vs, ts, nc, dt = fit(kwargs(fixed, var, s, F400))
                print(f"RESULT {tag} sigma={s} {vstr(var)}: val {vs} test {ts} ncols={nc} ({dt:.0f}s)", flush=True)
                if tag not in best or sv < best[tag][0]: best[tag] = (sv, s, var)
            except Exception as e:
                print(f"RESULT {tag} sigma={s} {vstr(var)} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
print(f"=== {prop.upper()} site models: stage 2 (refit val-best on 500, score test)", flush=True)
for tag, fixed, _ in ROWS:
    if tag not in best: print(f"FINAL {tag}: no successful stage-1 fit", flush=True); continue
    _, s, var = best[tag]
    try:
        _, _, ts, nc, dt = fit(kwargs(fixed, var, s, F500))
        print(f"FINAL {tag}: test {ts} ncols={nc} sigma={s} {vstr(var)} ({dt:.0f}s)", flush=True)
    except Exception as e:
        print(f"FINAL {tag} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
