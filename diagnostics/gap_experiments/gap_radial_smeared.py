"""Item 1 (μ): Applequist screened field, induced dipoles, smeared sources; item 3: radial families with the field column."""
import os, sys, time, copy
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, "/home/reece/TNEP"); sys.path.insert(0, "/home/reece/TNEP/tests")
import numpy as np
from ase.io import read, write
import gap_bond, gap_dipole as gd, gap_ewald as ge
from test_gap_dipole import _cfg
os.chdir("diagnostics/gap_experiments/out")
train = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":"); test = read("/home/reece/TNEP/datasets/test_waterbulk.xyz", index=":")
for fr in (train, test):
    for a in fr:
        if "dipole" not in a.info: a.info["dipole"] = np.asarray(a.calc.results["dipole"], float)
        a.calc = None
val = train[400:]
mu = lambda fr: np.stack([a.info["dipole"] for a in fr]); Mval, Mte = mu(val), mu(test)
def rr(P, T): r = np.sqrt(((P - T) ** 2).mean()); return 100 * r / T.std()
def mk(**kw):
    base = dict(data_path="fit400_dip_e.xyz", total_N=None, target_mode=1, rcut_hard=3.5, rcut_soft=3.0, dipole_rc=3.0,
                dipole_channels=10, dipole_radial="chebyshev_warped", dipole_cutoff="cosine2", dipole_species_pairs="symmetric",
                gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0, gap_sparse_per_species=250,
                l_max=7, alpha_max=7, compress_mode="none", gap_sigma=1e-3, dipole_gauss_sigma_scale=1.0)
    base.update(kw); return _cfg(**base)
S = gap_bond.Setup(mk()); cfg = S.cfg; print("SETUP done", flush=True)
E0c = {}
def E0(frames):
    out = []
    for a in frames:
        if id(a) not in E0c: E0c[id(a)] = ge.ewald_phi_E(a, np.array([ge.SPCE[int(z)] for z in a.numbers]))[1]
        out.append(E0c[id(a)])
    return out
def field(frames, tis): return [E[:, :, None] for E in E0(frames)]
def report(tag, fn):
    t0 = time.time()
    try: pred = fn(); print(f"RESULT {tag}: val {rr(pred(val), Mval):.2f}% test {rr(pred(test), Mte):.2f}% ncols={pred.n_columns} ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e: print(f"RESULT {tag} FAILED {type(e).__name__}: {str(e)[:140]}", flush=True)
vbase = lambda a, ti: gd._v_atoms(a, ti, cfg)
# ── smeared (LODE-like) sources ──
def smeared(frames, sig, with_E0):
    al = 1 / (np.sqrt(2) * sig); out = []
    for a, E in zip(frames, E0(frames)):
        _, Es = ge.ewald_phi_E(a, np.array([ge.SPCE[int(z)] for z in a.numbers]), alpha=al, r_cut=6.0, k_cut=8.6 * al, recip_only=True)
        out.append(np.stack([E, Es], 2) if with_E0 else Es[:, :, None])
    return out
for sig in (0.5, 1.0):
    report(f"SMEARED σ={sig} alone", lambda sig=sig: ge.fit_with_v(S, vbase, lambda fr, tis, sig=sig: smeared(fr, sig, False)))
    report(f"SMEARED σ={sig} + E0", lambda sig=sig: ge.fit_with_v(S, vbase, lambda fr, tis, sig=sig: smeared(fr, sig, True)))
# ── item 3: radial families with the field column ──
for name, kw in {"chebyshev C=10": dict(dipole_radial="chebyshev"), "gaussian C=20 s=0.5": dict(dipole_radial="gaussian", dipole_channels=20, dipole_gauss_sigma_scale=0.5),
                 "gaussian C=20 s=1.0": dict(dipole_radial="gaussian", dipole_channels=20), "bessel C=10": dict(dipole_radial="bessel"),
                 "warped C=20": dict(dipole_channels=20)}.items():
    S2 = gap_bond.Setup(mk(**kw)); c2 = S2.cfg
    report(f"RADIAL {name} + FIELD", lambda S2=S2, c2=c2: ge.fit_with_v(S2, lambda a, ti: gd._v_atoms(a, ti, c2), field))
