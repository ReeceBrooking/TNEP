"""2b site-model and multiscale (gap_extra_rcut) probe on the production GAP fit, with forward-pass timings.
argv: mu | pol — run as separate processes (own GAP/ record dir each).
Stage A: every row at its listed σ on train[:400] → val. Stage B: the best K rows also at the other grid σ.
Stage C: best K rows + soap baseline refit on 500 → test, ncols, fit s, µs/atom (gd.predict_timed).
SITE_SMOKE=1: 20 train frames, two rows. PROBE_ROWS=tag1,tag2 restricts the rows (follow-up runs)."""
import os, sys, time, json, glob, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole as gd
from test_gap_dipole import _cfg
prop = sys.argv[1]; assert prop in ("mu", "pol")
SMOKE = os.environ.get("SITE_SMOKE") == "1"
os.makedirs(f"diagnostics/gap_experiments/out/probe_{prop}", exist_ok=True)
os.chdir(f"diagnostics/gap_experiments/out/probe_{prop}")          # GAP/ record dir is per process
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]
F400, F500 = f"fit400_probe_{prop}.xyz", f"fit500_probe_{prop}.xyz"
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
S = dict(gap_site_models=["soap"]); S2 = dict(gap_site_models=["soap", "2b"])
# rows: (tag, kwargs, stage-A sigmas); key d2b becomes gap_model_delta["2b"], the rest pass through.
ROWS = {
    "mu": [("soap", S, (1e-3, 3e-3)),
           *[(f"soap+2b d{d}", dict(S2, d2b=d), (3e-3,)) for d in (0.1, 0.2, 0.3, 0.5)],
           *[(f"soap+2b rc{rc}", dict(S2, d2b=0.3, gap_site_rcut=rc), (3e-3,)) for rc in (4.0, 5.0, 6.0)],
           *[(f"soap+2b C{c}", dict(S2, d2b=0.3, gap_site_channels=c), (3e-3,)) for c in (6, 20)],
           ("soap+2b M500", dict(S2, d2b=0.3, gap_sparse_per_species=500), (3e-3,)),
           ("soap+2b xi2,4", dict(S2, d2b=0.3, gap_kernel_power=[2, 4]), (3e-3,)),
           *[(f"soap ms{x}", dict(S, gap_extra_rcut=x), (1e-3, 3e-3)) for x in (4.5, 6.0)],
           ("soap+2b ms4.5", dict(S2, d2b=0.3, gap_extra_rcut=4.5), (3e-3,)),
           ("soap+2b rc5 ms4.5", dict(S2, d2b=0.3, gap_site_rcut=5.0, gap_extra_rcut=4.5), (3e-3,)),
           # follow-up (PROBE_ROWS): smaller δ_2b, finer σ, and the raw (un-normalised) 2b kernel that keeps neighbour count
           ("soap+2b d0.05", dict(S2, d2b=0.05), (1.5e-3, 2e-3)),
           ("soap+2b d0.1 s1.5e-3", dict(S2, d2b=0.1), (1.5e-3,)),
           ("soap+2b raw d0.1", dict(S2, d2b=0.1, raw2b=True), (2e-3, 3e-3)),
           ("soap+2b raw d0.3", dict(S2, d2b=0.3, raw2b=True), (3e-3,)),
           # follow-up 2 (PROBE_ROWS): TNEP-like long READOUT basis (dipole_rc) with the 2b descriptor pinned at 3 Å, ± field
           ("soap+2b d0.1 ro4.5", dict(S2, d2b=0.1, dipole_rc=4.5, gap_site_rcut=3.0), (2e-3, 3e-3)),
           ("soap+2b d0.1 ro6.0", dict(S2, d2b=0.1, dipole_rc=6.0, gap_site_rcut=3.0), (2e-3, 3e-3)),
           ("soap+2b d0.1 ro6.0 C20", dict(S2, d2b=0.1, dipole_rc=6.0, gap_site_rcut=3.0, dipole_channels=20), (2e-3, 3e-3)),
           ("soap+2b d0.1 nofield", dict(S2, d2b=0.1, dipole_field=False), (2e-3, 3e-3)),
           ("soap+2b d0.1 ro6.0 nofield", dict(S2, d2b=0.1, dipole_rc=6.0, gap_site_rcut=3.0, dipole_field=False), (2e-3, 3e-3)),
           ("soap+2b d0.1 ro6.0 C20 nofield", dict(S2, d2b=0.1, dipole_rc=6.0, gap_site_rcut=3.0, dipole_channels=20, dipole_field=False), (2e-3, 3e-3)),
           ("soap ro6.0 nofield", dict(S, dipole_rc=6.0, dipole_field=False), (1e-3, 3e-3))],
    "pol": [("soap", S, (0.1,)),
            ("soap+2b d0.1", dict(S2, d2b=0.1), (0.1,)),
            ("soap+2b rc6", dict(S2, d2b=0.3, gap_site_rcut=6.0), (0.1,)),
            *[(f"soap ms{x}", dict(S, gap_extra_rcut=x), (0.1,)) for x in (5.5, 6.5)],
            ("soap+2b rc6 ms5.5", dict(S2, d2b=0.3, gap_site_rcut=6.0, gap_extra_rcut=5.5), (0.1,)),
            # follow-up (PROBE_ROWS): longer second scale, stronger σ
            ("soap ms7.5", dict(S, gap_extra_rcut=7.5), (0.3,)),
            ("soap ms6.5 s0.5", dict(S, gap_extra_rcut=6.5), (0.5,))],
}[prop]
GRID = {"mu": (2e-3, 3e-3, 5e-3), "pol": (0.1, 0.3)}[prop]
K = int(os.environ.get("PROBE_K", {"mu": 4, "pol": 2}[prop]))
if SMOKE: ROWS = [ROWS[0], ROWS[-1]]; K = 1
only = os.environ.get("PROBE_ROWS")
if only: ROWS = [r for r in ROWS if r[0] in only.split(",")]; GRID = ()      # follow-ups: listed σ only
base, rr, Tv, Tt = (MU, rr_mu, mu(val), mu(test)) if prop == "mu" else (POL, rr_pol, v6(val), v6(test))
import gap_models as gm
_normalise = gm._normalise
def kwargs(kw, sigma, data_path):
    v = dict(kw); delta = {"soap": 1.0, "2b": v.pop("d2b", 1.0), "3b": 1.0}
    gm._normalise = (lambda q: q) if v.pop("raw2b", False) else _normalise   # raw2b: polynomial kernel on the un-normalised radial distribution
    return {**base, **SITE, **v, "gap_model_delta": delta, "gap_sigma": sigma, "data_path": data_path}
def fit(kw):
    """(val score, val str, test str, ncols, fit seconds, µs/atom val, µs/atom test) — the latest GAP/*.json is this fit's record."""
    cfg = _cfg(**kw); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gd.fit_gap_dipole(cfg); dt = time.time() - t0
        Pv, uv = gd.predict_timed(m, val); Pt, ut = gd.predict_timed(m, test)
    rec = json.load(open(sorted(glob.glob("GAP/*.json"), key=os.path.getmtime)[-1]))
    sv, vs = rr(Pv, Tv); _, ts = rr(Pt, Tt)
    return sv, vs, ts, rec["gap"]["n_columns"], dt, uv, ut
def run(tag, kw, s, data_path, kind):
    try:
        sv, vs, ts, nc, dt, uv, ut = fit(kwargs(kw, s, data_path))
        print(f"{kind} {tag} sigma={s}: val {vs} test {ts} ncols={nc} fit={dt:.0f}s us/atom={ut:.1f} (val {uv:.1f})", flush=True)
        return sv
    except Exception as e:
        print(f"{kind} {tag} sigma={s} FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
best = {}                                                          # tag -> (val score, sigma)
def record(tag, s, sv):
    if sv is not None and (tag not in best or sv < best[tag][0]): best[tag] = (sv, s)
print(f"=== {prop.upper()} probe: stage A (fit 400 at listed sigma, val)", flush=True)
for tag, kw, sigmas in ROWS:
    for s in sigmas: record(tag, s, run(tag, kw, s, F400, "RESULT"))
top = sorted((t for t in best if t != "soap"), key=lambda t: best[t][0])[:K]
print(f"=== {prop.upper()} probe: stage B (other grid sigmas for {top})", flush=True)
for tag, kw, sigmas in ROWS:
    if tag not in top: continue
    for s in GRID:
        if s not in sigmas: record(tag, s, run(tag, kw, s, F400, "RESULT"))
print(f"=== {prop.upper()} probe: stage C (refit on 500 at selected sigma, test)", flush=True)
for tag, kw, _ in ROWS:
    if tag != "soap" and tag not in top: continue
    if tag not in best: print(f"FINAL {tag}: no successful stage-A fit", flush=True); continue
    run(tag, kw, best[tag][1], F500, "FINAL")
