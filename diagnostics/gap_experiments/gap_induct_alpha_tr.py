"""Item 1 (α): Applequist induced-coupling site tensors as an extra kernel group on the two-group + cross model."""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_bond, gap_dipole as gd, gap_ewald as ge
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for a in train: a.calc = None
val = train[400:]; write("fit400_pol_t.xyz", train[:400])
def v6(fr):
    T = np.stack([np.asarray(a.info["pol"], float).reshape(3, 3) for a in fr])
    return np.stack([T[:, 0, 0], T[:, 1, 1], T[:, 2, 2], T[:, 0, 1], T[:, 1, 2], T[:, 2, 0]], 1)
Tval, Tte = v6(val), v6(test)
def rr(P, T):
    d = np.sqrt(((P[:, :3] - T[:, :3]) ** 2).mean()); o = np.sqrt(((P[:, 3:] - T[:, 3:]) ** 2).mean())
    return 100 * d / T[:, :3].std(), 100 * o / T[:, 3:].std()
def mk(sigma):
    return _cfg(data_path="fit400_pol_t.xyz", total_N=None, target_mode=2, rcut_hard=4.0, rcut_soft=3.5, alpha_max=7, l_max=7,
                compress_mode="trivial", dipole_rc=4.0, dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine",
                dipole_species_pairs="symmetric", gap_sparse_per_species=250, gap_kernel_power=2, gap_cross_channels=4, gap_sigma=sigma)
Wc = {}
def W(frames, aO, intra_ex, traceless=True):
    out = []
    for a in frames:
        k = (id(a), aO, intra_ex)
        if k not in Wc:
            w = ge.induced_site_tensors(a, {8: aO}, intra_ex)
            if traceless: w[:, :3] -= w[:, :3].mean(1, keepdims=True)
            Wc[k] = w[:, :, None]
        out.append(Wc[k])
    return out
for sigma in (0.1,):
    S = gap_bond.Setup(mk(sigma)); cfg = S.cfg; print(f"SETUP sigma={sigma}", flush=True)
    vbase = lambda a, ti: gd._v_atoms(a, ti, cfg)
    t0 = time.time(); pred = ge.fit_with_v(S, vbase); dv, ov = rr(pred(val), Tval); dt, ot = rr(pred(test), Tte)
    print(f"RESULT sigma={sigma} BASE two-group+cross: val {dv:.1f}/{ov:.1f} test {dt:.1f}/{ot:.1f} ({time.time()-t0:.0f}s)", flush=True)
    for aO, ie in ((1.0, True), (1.45, True)):
        t0 = time.time()
        try:
            pred = ge.fit_with_v(S, vbase, lambda fr, tis, aO=aO, ie=ie: W(fr, aO, ie))
            dv, ov = rr(pred(val), Tval); dt, ot = rr(pred(test), Tte)
            print(f"RESULT sigma={sigma} INDUCED-TRACELESS aO={aO} intra_excl={ie}: val {dv:.1f}/{ov:.1f} test {dt:.1f}/{ot:.1f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"RESULT sigma={sigma} INDUCED-TRACELESS aO={aO} FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
