import os, sys, time, json, glob, contextlib, io
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np, resource
from ase.io import read, write
import gap_dipole as gd
from test_gap_dipole import _cfg
os.makedirs("diagnostics/gap_experiments/out/twogroup", exist_ok=True); os.chdir("diagnostics/gap_experiments/out/twogroup")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr: a.calc = None
val = train[400:]; write("fit400.xyz", train[:400]); write("fit500.xyz", train)
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
Tv, Tt = v6(val), v6(test)
def rr(P, T):
    dg = 100 * np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()) / T[:, :3].std(); od = 100 * np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean()) / T[:, 3:].std()
    return f"{dg:.1f}/{od:.1f}"
POL = dict(total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7, compress_mode="trivial", dipole_rc=4.0,
           dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine", dipole_species_pairs="symmetric",
           gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=0)
def run(tag, path, sigma):
    cfg = _cfg(**POL, data_path=path, gap_sigma=sigma); t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        m = gd.fit_gap_dipole(cfg); dt = time.time() - t0; Pv, _ = gd.predict_timed(m, val); Pt, ut = gd.predict_timed(m, test)
    rec = json.load(open(sorted(glob.glob("GAP/*.json"), key=os.path.getmtime)[-1]))
    print(f"{tag} sigma={sigma}: val {rr(Pv, Tv)} test {rr(Pt, Tt)} ncols={rec['gap']['n_columns']} fit={dt:.0f}s us/atom={ut:.1f} "
          f"peakRSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB", flush=True)
for s in (0.1, 0.3): run("RESULT two-groups fit400", "fit400.xyz", s)
run("FINAL two-groups fit500", "fit500.xyz", 0.1)
