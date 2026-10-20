import os, sys, time, json, glob, contextlib, io, resource
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_dipole as gd
from test_gap_dipole import _cfg
os.makedirs("diagnostics/gap_experiments/out/mubase", exist_ok=True); os.chdir("diagnostics/gap_experiments/out/mubase")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]; write("fit400.xyz", train[:400]); write("fit500.xyz", train)
mu = lambda fr: np.stack([a.info["dipole"] for a in fr]); Mv, Mt = mu(val), mu(test)
rr = lambda P, T: f"{100*np.sqrt(((P-T)**2).mean())/T.std():.2f}%"
BASE = dict(total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0, dipole_channels=10, dipole_radial="chebyshev_warped",
            dipole_cutoff="cosine2", dipole_species_pairs="symmetric", gap_kernel_power=2, convert_dipole_to_eangstrom=False,
            gap_cross_channels=0, gap_sparse_per_species=250, dipole_field=False)
ORIG = dict(BASE, l_max=4, alpha_max=4, compress_mode="trivial")
S7 = dict(BASE, l_max=7, alpha_max=7, compress_mode="none")
def run(tag, kw, path, sigma):
    cfg = _cfg(**kw, data_path=path, gap_sigma=sigma); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gd.fit_gap_dipole(cfg); dt = time.time() - t0; Pv, _ = gd.predict_timed(m, val); Pt, ut = gd.predict_timed(m, test)
    rec = json.load(open(sorted(glob.glob("GAP/*.json"), key=os.path.getmtime)[-1]))
    print(f"{tag} sigma={sigma}: val {rr(Pv, Mv)} test {rr(Pt, Mt)} ncols={rec['gap']['n_columns']} fit={dt:.0f}s us/atom={ut:.1f} "
          f"peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB", flush=True)
for s in (1e-3, 3e-3): run("RESULT original l=a=4 trivial nofield fit400", ORIG, "fit400.xyz", s)
run("FINAL original l=a=4 trivial nofield fit500", ORIG, "fit500.xyz", 1e-3)
for s in (1e-3, 3e-3): run("RESULT soap7 none nofield fit400", S7, "fit400.xyz", s)
run("FINAL soap7 none nofield fit500", S7, "fit500.xyz", 1e-3)
