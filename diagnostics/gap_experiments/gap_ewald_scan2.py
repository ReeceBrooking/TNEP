import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_bond, gap_ewald as ge
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]; write("fit400_dip_e.xyz", train[:400]); write("fit500_dip_e.xyz", train)
mu = lambda fr: np.stack([a.info["dipole"] for a in fr]); Mval, Mte = mu(val), mu(test)
def rr(P, T): r = np.sqrt(((P - T) ** 2).mean()); return 100 * r / T.std()
def mk(path, sigma=1e-3):
    return _cfg(data_path=path, total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0,
                dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
                gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250,
                l_max=7, alpha_max=7, compress_mode="none", gap_sigma=sigma)
S = gap_bond.Setup(mk("fit400_dip_e.xyz")); print("SETUP done", flush=True)
cfg = S.cfg
def report(tag, fn, S_=None, delta=1.0):
    S_ = S_ or S; t0 = time.time()
    try:
        pred = ge.fit_extra_columns(S_, fn, delta)
        print(f"RESULT {tag}: val {rr(pred(val), Mval):.2f}% test {rr(pred(test), Mte):.2f}% ncols={pred.n_columns} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"RESULT {tag} FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
full = lambda frames, tis: [E[:, :, None] for _, E in ge.ewald_features(frames, tis, cfg, False)]
# A. near-only Coulomb field at several cutoffs
for rc in (3.0, 4.5, 6.0):
    report(f"NEAR cutoff={rc}", lambda frames, tis, rc=rc: [E[:, :, None] for E in ge.coulomb_near(frames, cfg, rc)])
# B. species-split Ewald field (2 columns)
report("SPLIT ewald O|H", lambda frames, tis: ge.ewald_species_split(frames, cfg))
# C. field + anisotropic response T^a·E
for cx in (2, 4):
    report(f"FIELD+TENSOR C_x={cx}", lambda frames, tis, cx=cx: [np.concatenate([E[:, :, None], W], axis=2)
            for (_, E), W in zip(ge.ewald_features(frames, tis, cfg, False),
                                 ge.bond_tensor_dot(frames, tis, cfg, [E for _, E in ge.ewald_features(frames, tis, cfg, False)], cx))])
# D. sigma re-selection for the plain field column
for s in (3e-4, 3e-3):
    S2 = gap_bond.Setup(mk("fit400_dip_e.xyz", s))
    report(f"FIELD sigma={s}", full, S2)
# E. final: field column refit on all 500 → test
S5 = gap_bond.Setup(mk("fit500_dip_e.xyz"))
t0 = time.time(); pred = ge.fit_extra_columns(S5, full, 1.0)
print(f"RESULT FINAL field column, 500 frames: test {rr(pred(test), Mte):.2f}% ({time.time()-t0:.0f}s)", flush=True)
