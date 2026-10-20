"""Round-1 polarisability scan: long-range α column groups (pol_field_groups: induced / grad / cross / quad / qeq)
on the two-group + cross base, alone and combined, then on the 6.5 Å two-scale best model; plus gap_lambda_shear.
Stage A: single scale, fit 400 → val at σ ∈ {0.1, 0.3}: base, each group, all five.  Stage B: pairs of the top-3 groups.
Stage C: refit on 500 → test for base / best single / best pair / all, single scale (selected σ) and gap_extra_rcut=6.5 (σ 0.1 and 0.3),
plus gap_lambda_shear ∈ {0.5, 2} on the two-scale base.  µs/atom from gd.predict_timed.  POL_SMOKE=1: 20 frames, base + all only."""
import os, sys, time, json, glob, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import data as d
import gap_dipole as gd
from test_gap_dipole import _cfg
SMOKE = os.environ.get("POL_SMOKE") == "1"
os.makedirs("diagnostics/gap_experiments/out/polgroups", exist_ok=True)
os.chdir("diagnostics/gap_experiments/out/polgroups")                 # own GAP/ record dir
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr: a.calc = None
val = train[400:]
F400, F500 = "fit400_polgroups.xyz", "fit500_polgroups.xyz"
write(F400, train[:20] if SMOKE else train[:400]); write(F500, train)
SPCE = {8: -0.8476, 1: 0.4238}
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
Tv, Tt = v6(val), v6(test)
def rr(P, T):
    dg = 100 * np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()) / T[:, :3].std()
    od = 100 * np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean()) / T[:, 3:].std()
    return (dg + od) / 2, f"{dg:.1f}/{od:.1f}"
POL = dict(data_path=F400, total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7,
           compress_mode="trivial", dipole_rc=4.0, dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
           dipole_species_pairs="symmetric", gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=4, gap_sigma=0.1,
           dipole_field_charges=SPCE)
# ── cache the (unscaled) group tensors per frame and per group for the whole scan ──
_orig = d.pol_field_groups; _cache = {}
def _cached(atoms, ti, cfg, v_bond):
    groups = list(cfg.pol_field_groups)
    if not groups: return _orig(atoms, ti, cfg, v_bond)
    key0 = (atoms.positions.tobytes(), int(cfg.gap_cross_channels))
    outs = []
    for g in groups:
        k = key0 + (g,)
        if k not in _cache:
            saved = cfg.pol_field_groups; cfg.pol_field_groups = [g]   # compute one group at a time, cache each
            try: _cache[k] = _orig(atoms, ti, cfg, v_bond)
            finally: cfg.pol_field_groups = saved
        outs.append(_cache[k])
    return np.concatenate(outs, axis=2)
d.pol_field_groups = _cached                                        # gap_dipole calls tnep_data.pol_field_groups(...)
GROUPS = ["induced", "grad", "cross", "quad", "qeq"]
def fit(kw, data_path):
    cfg = _cfg(**{**POL, **kw, "data_path": data_path}); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gd.fit_gap_dipole(cfg); dt = time.time() - t0
        Pv, uv = gd.predict_timed(m, val); Pt, ut = gd.predict_timed(m, test)
    rec = json.load(open(sorted(glob.glob("GAP/*.json"), key=os.path.getmtime)[-1]))
    sv, vs = rr(Pv, Tv); _, ts = rr(Pt, Tt)
    return sv, vs, ts, rec["gap"]["n_columns"], dt, ut
def run(tag, kw, data_path, kind):
    try:
        sv, vs, ts, nc, dt, ut = fit(kw, data_path)
        print(f"{kind} {tag} sigma={kw.get('gap_sigma', 0.1)}: val {vs} test {ts} ncols={nc} fit={dt:.0f}s us/atom={ut:.1f}", flush=True)
        return sv
    except Exception as e:
        print(f"{kind} {tag} sigma={kw.get('gap_sigma', 0.1)} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
SIG = (0.1, 0.3)
def scan(tag, kw):
    """stage-A/B row at both σ → (best val score, best σ)."""
    best = None
    for s in SIG:
        sv = run(tag, dict(kw, gap_sigma=s), F400, "RESULT")
        if sv is not None and (best is None or sv < best[0]): best = (sv, s)
    return best
rows = {"base": {}, **{f"+{g}": {"pol_field_groups": [g]} for g in GROUPS}, "+all": {"pol_field_groups": GROUPS}}
if SMOKE: rows = {"base": rows["base"], "+all": rows["+all"]}
IQ = {"pol_field_groups": ["induced", "quad"]}; ALL = {"pol_field_groups": GROUPS}
# POL_FINALS="tag;tag": skip stages A/B and refit only these on 500 → test (follow-ups after the round-1 OOM at '+all ms6.5').
FOLLOW = {"+all ms6.5 s0.1": dict(ALL, gap_extra_rcut=6.5, gap_sigma=0.1),
          "+all ms6.5 s0.3": dict(ALL, gap_extra_rcut=6.5, gap_sigma=0.3),
          "base ms6.5 s0.3 shear0.5": dict(gap_extra_rcut=6.5, gap_lambda_shear=0.5, gap_sigma=0.3),
          "base ms6.5 s0.3 shear2": dict(gap_extra_rcut=6.5, gap_lambda_shear=2.0, gap_sigma=0.3),
          "+induced+quad s0.3 shear3": dict(IQ, gap_lambda_shear=3.0, gap_sigma=0.3),
          "+all s0.3 shear3": dict(ALL, gap_lambda_shear=3.0, gap_sigma=0.3),
          "+induced+quad ms6.5 s0.3 shear3": dict(IQ, gap_extra_rcut=6.5, gap_lambda_shear=3.0, gap_sigma=0.3),
          "+all ms6.5 s0.3 shear3": dict(ALL, gap_extra_rcut=6.5, gap_lambda_shear=3.0, gap_sigma=0.3)}
if os.environ.get("POL_FINALS"):
    for t in os.environ["POL_FINALS"].split(";"): run(t, FOLLOW[t], F500, "FINAL")
    sys.exit(0)
print("=== POL groups: stage A (single scale, fit 400 → val)", flush=True)
best = {t: scan(t, kw) for t, kw in rows.items()}
singles = sorted((t for t in best if t.startswith("+") and t != "+all" and best[t]), key=lambda t: best[t][0])[:3]
print(f"=== stage B: pairs of {singles}", flush=True)
pairs = {}
for i in range(len(singles)):
    for j in range(i + 1, len(singles)):
        t = f"+{singles[i][1:]}+{singles[j][1:]}"; kw = {"pol_field_groups": [singles[i][1:], singles[j][1:]]}
        pairs[t] = kw; best[t] = scan(t, kw)
rows.update(pairs)
best_single = singles[0] if singles else None
best_pair = min(pairs, key=lambda t: best[t][0] if best[t] else 1e9) if pairs else None
finals = [t for t in ["base", best_single, best_pair, "+all"] if t]
print(f"=== stage C: refit on 500 → test for {finals}; single scale at selected σ, two-scale (6.5 Å) at σ 0.1 and 0.3", flush=True)
for t in finals:
    if best.get(t): run(t, dict(rows[t], gap_sigma=best[t][1]), F500, "FINAL")
for t in finals:
    for s in SIG: run(f"{t} ms6.5", dict(rows[t], gap_extra_rcut=6.5, gap_sigma=s), F500, "FINAL")
for lam in (0.5, 2.0):
    run(f"base ms6.5 shear{lam}", dict(gap_extra_rcut=6.5, gap_lambda_shear=lam, gap_sigma=0.3), F500, "FINAL")
