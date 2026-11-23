"""GAP scan rows: the production dipole and polarisability models and their build-up ladders
(docs/gap_architecture_review.md §5.1). One fit per process (memory accumulates across fits):

    PS_PROP=mu|pol PS_ROW="<tag>" PS_SIGMA=<σ> PS_FIT=400|500 python gap_scan.py
    PS_PROP=mu PS_LIST=1 python gap_scan.py          # list the row tags

Stage A (fit 400) prints val and test; stage C (fit 500) prints test. Driver: gap_scan_parallel.sh.
μ rows fit datasets/{train,test}_waterbulk.xyz (dipole labels); α rows the χ files (pol = Vχ = V(ε−1)/4π)."""
import os, sys, time, contextlib, io, resource
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole as gd
from test_gap_dipole import _cfg
prop = os.environ["PS_PROP"]; assert prop in ("mu", "pol")
OUT = f"/home/reece/TNEP/diagnostics/gap_experiments/out/prog_{prop}"
os.makedirs(OUT, exist_ok=True); os.chdir(OUT)                 # own GAP/ record dir
LAB = "_chi" if prop == "pol" else ""
train = read(f"/home/reece/TNEP/datasets/train_waterbulk{LAB}.xyz", index=":"); test = read(f"/home/reece/TNEP/datasets/test_waterbulk{LAB}.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]; F400, F500 = f"fit400_{prop}{LAB}.xyz", f"fit500_{prop}{LAB}.xyz"
for path, frames in ((F400, train[:400]), (F500, train)):          # atomic: parallel rows share these
    if not os.path.exists(path): write(f"{path}.{os.getpid()}", frames); os.replace(f"{path}.{os.getpid()}", path)
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
# μ reference: SOAP-7 (trivial compression, the standard since 2026-10-28), rc 3.0, cosine², warped Chebyshev C 10, M 250,
# full bond vector (û is null for the dipole: 1.47 vs 1.45).
MU = dict(total_N=None, target_mode=1, dipole_unit_vector=False, rcut_hard=3.5, rcut_soft=3.0,
          dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
          convert_dipole_to_eangstrom=False, gap_sparse_per_species=250, l_max=7, alpha_max=7, compress_mode="trivial",
          gap_site_channels=10, gap_site_lmax=2, gap_cross_lmax=1)
# α reference: SOAP-7 trivial 4.0/3.5, rc 4.0, cosine, C 10, M 250, û readout (GAPconfig default).
POL = dict(total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7, compress_mode="trivial",
           dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
           dipole_species_pairs="symmetric", gap_sparse_per_species=250, gap_site_channels=10, gap_site_lmax=2,
           gap_cross_lmax=1)
def summed(models, d2b=1.0, d3b=1.0, M=None, **kw):
    return {"gap_site_models": models, "gap_model_delta": {"soap": 1.0, "2b": d2b, "3b": d3b},
            **({"gap_sparse_per_species": {"soap": 250, **M}} if M else {}), **kw}
def pol3b(M=40, d=0.1, lx=3, C=6, L=1, **kw):
    return summed(["soap", "3b"], d3b=d, M={"3b": M}, gap_cross_lmax=lx, gap_site_lmax=L, gap_site_channels=C, **kw)
MU2B = dict(d2b=0.1, M={"2b": 40})
def tnep(models, d=1.0, **kw):      # QUIP gap_fit dipole model as a site block (test rows; dipole only)
    return {"gap_site_models": models, "gap_model_delta": {"soap": 1.0, "2b": 0.1, "3b": 1.0, "tnep": d},
            "gap_sparse_per_species": {"soap": 250, "2b": 40, "tnep": 250}, **kw}
ROWS = {
    "mu": {"soap": {},
           "tnep": tnep(["tnep"]),
           "soap+tnep d0.1": tnep(["soap", "tnep"], 0.1),
           "soap+tnep d0.3": tnep(["soap", "tnep"], 0.3),
           "soap+tnep d1": tnep(["soap", "tnep"], 1.0),
           "soap+2b +cross2 +tnep d0.3": tnep(["soap", "2b", "tnep"], 0.3, gap_cross_channels=2),
           "soap+2b +cross2 +tnep d1": tnep(["soap", "2b", "tnep"], 1.0, gap_cross_channels=2),
           "soap+tnep N2 d0.1": tnep(["soap", "tnep"], 0.1, gap_tnep_rij_power=2),
           "soap+2b +cross2 +tnep N2 d0.3": tnep(["soap", "2b", "tnep"], 0.3, gap_cross_channels=2, gap_tnep_rij_power=2),
           "soap+2b +cross2 +tnep N2 d1": tnep(["soap", "2b", "tnep"], 1.0, gap_cross_channels=2, gap_tnep_rij_power=2),
           "soap+2b": summed(["soap", "2b"], **MU2B),
           "soap+2b +cross2": summed(["soap", "2b"], gap_cross_channels=2, **MU2B),        # production: 1.47 % (σ 0.003)
           "soap+2b +cross4": summed(["soap", "2b"], gap_cross_channels=4, **MU2B)},
    "pol": {"base": {},
            "cross4": dict(gap_cross_channels=4),
            "cross4 +3b": pol3b(gap_cross_channels=4),
            "cross4 +3b +soap6": pol3b(gap_cross_channels=4, gap_extra_rcut=6.0),
            "winner": pol3b(gap_cross_channels=4, gap_extra_rcut=6.0, gap_lambda_shear=2.0),   # production: 13.0/13.9 (σ 0.1)
            "winner cross6": pol3b(gap_cross_channels=6, gap_extra_rcut=6.0, gap_lambda_shear=2.0)},
}
if os.environ.get("PS_LIST"):                      # list the row tags of a property and exit (driver input)
    print("\n".join(ROWS[prop])); sys.exit(0)
def ncols(m):
    """Design width of a fitted model = its coefficient count (all site models + the extra-rcut block)."""
    return (sum(np.asarray(a).size for mod in m.site_alpha for a in m.site_alpha[mod])
            + sum(np.asarray(a).size for a in (m.alpha2 or [])))
def run(tag, kw, path, sigma, kind):
    cfg = _cfg(**{**(MU if prop == "mu" else POL), **kw}, data_path=path, gap_sigma=sigma); t0 = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            m = gd.fit_gap_dipole(cfg)
            dt = time.time() - t0; Pv, _ = gd.predict_timed(m, val); Pt, ut = gd.predict_timed(m, test)
        rr = rr_mu if prop == "mu" else rr_pol
        _, vs = rr(Pv, mu(val) if prop == "mu" else v6(val)); _, ts = rr(Pt, mu(test) if prop == "mu" else v6(test))
        print(f"{kind} {prop} {tag} sigma={sigma:g}: val {vs} test {ts} ncols={ncols(m)} fit={dt:.0f}s us/atom={ut:.1f} "
              f"peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f}GB", flush=True)
    except Exception as e:
        print(f"{kind} {prop} {tag} sigma={sigma:g} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
tag, sigma, fit_n = os.environ["PS_ROW"], float(os.environ["PS_SIGMA"]), os.environ.get("PS_FIT", "400")
run(tag, ROWS[prop][tag], F400 if fit_n == "400" else F500, sigma, "RESULT" if fit_n == "400" else "FINAL")
