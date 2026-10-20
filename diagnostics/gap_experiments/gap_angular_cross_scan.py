"""α: the angular-cross group sym(w⊗v) (gap_angular_cross; w = 3b angular kernel vector, v = dipole kernel vector) —
alone-ish (bond groups at vanishing prior, cross group kept since both share gap_group_delta[2]), on the two-group model,
and with the induced-coupling column.  C_x = 2 throughout (the group is 2·C_x³·L columns per sparse point; C_x = 4 would
exceed the ~40k-column memory budget).  Stage A: fit 400 → val/test at σ ∈ {0.1, 0.3}; stage C: refit 500 → test.
Run ONE process under the RSS watchdog (see README)."""
import os, sys, time, json, glob, contextlib, io, resource
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole as gd
from test_gap_dipole import _cfg
os.makedirs("diagnostics/gap_experiments/out/angcross", exist_ok=True); os.chdir("diagnostics/gap_experiments/out/angcross")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr: a.calc = None
val = train[400:]; F400, F500 = "fit400.xyz", "fit500.xyz"; write(F400, train[:400]); write(F500, train)
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
Tv, Tt = v6(val), v6(test)
def rr(P, T):
    dg = 100 * np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()) / T[:, :3].std(); od = 100 * np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean()) / T[:, 3:].std()
    return (dg + od) / 2, f"{dg:.1f}/{od:.1f}"
SPCE = {8: -0.8476, 1: 0.4238}
POL = dict(total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7, compress_mode="trivial", dipole_rc=4.0,
           dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine", dipole_species_pairs="symmetric",
           gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=2, dipole_field_charges=SPCE)
OFF = {"gap_group_delta": [1e-6, 1e-6, 1.0]}                     # bond groups at vanishing prior amplitude
ANG = lambda L: {"gap_angular_cross": True, "gap_cross_lmax": L}
ROWS = {"bond-off cross2": OFF,
        "bond-off cross2 +ang L1": {**OFF, **ANG(1)},
        "bond-off cross2 +ang L2": {**OFF, **ANG(2)},
        "2groups cross2": {},
        "2groups cross2 +ang L1": ANG(1),
        "2groups cross2 +ang L2": ANG(2),
        "2groups cross2 +ang L1 +induced": {**ANG(1), "pol_field_groups": ["induced"]}}
def run(tag, kw, path, sigma, kind):
    cfg = _cfg(**POL, **kw, data_path=path, gap_sigma=sigma); t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = gd.fit_gap_dipole(cfg); dt = time.time() - t0; Pv, _ = gd.predict_timed(m, val); Pt, ut = gd.predict_timed(m, test)
        rec = json.load(open(sorted(glob.glob("GAP/*.json"), key=os.path.getmtime)[-1]))
        sv, vs = rr(Pv, Tv); _, ts = rr(Pt, Tt)
        print(f"{kind} {tag} sigma={sigma}: val {vs} test {ts} ncols={rec['gap']['n_columns']} fit={dt:.0f}s us/atom={ut:.1f} "
              f"peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f}GB", flush=True); return sv
    except Exception as e:
        print(f"{kind} {tag} sigma={sigma} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
best = {}
print("=== stage A (fit 400 → val/test)", flush=True)
for tag, kw in ROWS.items():
    for s in (0.1, 0.3):
        sv = run(tag, kw, F400, s, "RESULT")
        if sv is not None and (tag not in best or sv < best[tag][0]): best[tag] = (sv, s)
finals = ["2groups cross2"] + sorted((t for t in best if "+ang" in t and "bond-off" not in t), key=lambda t: best[t][0])[:2]
print(f"=== stage C (refit 500 → test) for {finals}", flush=True)
for t in finals:
    if t in best: run(t, ROWS[t], F500, best[t][1], "FINAL")
