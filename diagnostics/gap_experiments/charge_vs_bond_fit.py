"""Charge-only GAP: mu = sum_i (q_i - qbar) r_i, q_i a SOAP sparse-GP scalar. One fit per process.
usage: fit.py DATA(qm7b|cho|qm9) MODELS(charge|soap|soap,charge) M SIGMA [XI] [N_TRAIN]"""
import sys, os, resource, numpy as np
sys.path.insert(0, "/home/reece/TNEP")
from ase.io import read, write
import gap_dipole as gd
from gap_config import GAPconfig

data, models, M, sigma = sys.argv[1], sys.argv[2].split(","), eval(sys.argv[3]), float(sys.argv[4])
xi = int(sys.argv[5]) if len(sys.argv) > 5 else 2
src, key, n_test, to_D = {"qm7b": ("qm7b_shuffle2_dipoles.xyz", "dipole_ccsd", 1443, 2.5417),
                          "cho": ("qm7b_shuffle2_dipoles.xyz", "dipole_ccsd", 480, 2.5417),
                          "qm9": ("qm9_traintest_dipoles.xyz", "dipole_b3lyp", 5000, 1.0)}[data]
SP, CAP, PAIRS = os.environ.get("SPECIES"), int(os.environ.get("CAP", 0)), os.environ.get("PAIRS", "off")
tag = data + (f"_{SP}_{CAP}" if SP or CAP else "")
tr, te = f"{tag}_train.xyz", f"{tag}_test.xyz"
if not os.path.exists(tr):          # file-order split: last n_test frames are the test set
    fr = read(f"/home/reece/TNEP/datasets/{src}", index=":")
    if data == "cho":
        fr = [a for a in fr if set(a.numbers) <= {1, 6, 8}]
    if SP:                          # SPECIES=C,H,N keeps molecules made only of those elements
        fr = [a for a in fr if set(a.get_chemical_symbols()) <= set(SP.split(","))]
    if CAP:                         # CAP=n keeps the first n molecules (matched-size comparisons)
        fr = fr[:CAP]; n_test = int(os.environ.get("NTEST", 480))
    write(tr, fr[:-n_test]); write(te, fr[-n_test:])
key = os.environ.get("KEY", key)
BASE, ROUT = os.environ.get("BASE"), os.environ.get("RESID_OUT")
key0 = key
if BASE:                            # stage 2: fit the stage-1 training residual, add the stage-1 test prediction back
    tr, key = BASE + "_train.xyz", "resid"
if os.environ.get("TRAIN_FILE"):    # cross-fitting: explicit train / held-out files
    tr, te = os.environ["TRAIN_FILE"], os.environ["TEST_FILE"]
cfg = GAPconfig()
for k, v in dict(data_path=tr, test_data_path=None, allowed_species=None, target_key=key, total_N=None,
                 l_max=7, alpha_max=7, rcut_hard=3.5, rcut_soft=3.0, compress_mode="trivial",
                 dipole_radial="chebyshev_warped", dipole_channels=10, dipole_species_pairs=PAIRS,
                 gap_site_models=models, gap_sparse_per_species=M, gap_sigma=sigma, gap_kernel_power=xi).items():
    setattr(cfg, k, v)
cfg.gap_model_delta = {**cfg.gap_model_delta, "charge": float(os.environ.get("DCH", 1.0))}
if os.environ.get("RC"):
    cfg.rcut_hard = float(os.environ["RC"]); cfg.rcut_soft = cfg.rcut_hard - 0.5
if len(sys.argv) > 6:
    cfg.total_N = int(sys.argv[6])
FACT = os.environ.get("FACT")      # "R:Msmall" factorised species-pair table: alpha[m,c,t'] = sum_r A[m,c,r] B_t[r,t']
if FACT:                            # B_t fixed from the SVD of a small pairs-on fit; the big fit stays one linear solve
    import copy
    R, Ms = (int(x) for x in FACT.split(":"))
    c0 = copy.copy(cfg); c0.dipole_species_pairs, c0.gap_site_models, c0.gap_sparse_per_species = "symmetric", ["soap"], Ms
    m0 = gd.fit_gap_dipole(c0); T = int(c0.num_types)
    B = np.stack([np.linalg.svd(al.reshape(-1, T), full_matrices=False)[2][:R] for al in m0.site_alpha["soap"]])   # [T, R, T']
    sv = [np.linalg.svd(al.reshape(-1, T), compute_uv=False) for al in m0.site_alpha["soap"]]
    print("FACT singular values per centre species (normalised):", [np.round(x / x[0], 2).tolist() for x in sv], flush=True)
    del m0
    cfg.dipole_species_pairs = "off"
    def _fact_v(V, atoms, ti, c):   # [N, D, Ck, T'] -> [N, D, Ck*R]
        V = np.asarray(V, dtype=np.float64)
        return np.einsum("ndct,nrt->ndcr", V, B[np.asarray(ti)]).reshape(V.shape[0], V.shape[1], -1)
    gd._finish_v = _fact_v
model = gd.fit_gap_dipole(cfg)
test = read(te, index=":")
pred = gd.predict(model, test)
if ROUT:                            # stage 1: write training frames with resid = target - fitted value (fit order = file order)
    frames = read(tr, index=":"); mu_tr = gd._targets(frames, cfg)
    assert len(frames) == len(model.train_pred)
    for a_, r_ in zip(frames, mu_tr - model.train_pred):
        a_.info["resid"] = r_
    write(ROUT + "_train.xyz", frames); np.save(ROUT + "_test.npy", pred)
cfg.target_key = key0
mu = gd._targets(test, cfg)
if BASE:
    print(f"STAGE2 {tag}: charge correction rms {np.sqrt((pred ** 2).mean()):.4f} on base test residual rms {np.sqrt(((np.load(BASE + '_test.npy') - mu) ** 2).mean()):.4f}")
    pred = pred + np.load(BASE + "_test.npy")
rr = lambda p, m: 100 * np.sqrt(((p - m) ** 2).sum() / ((m - m.mean(0)) ** 2).sum())
mae = np.abs(np.linalg.norm(pred, axis=1) - np.linalg.norm(mu, axis=1)).mean() * to_D
if len(models) > 1:      # ablation: zero one block of the JOINT fit and score what is left
    from dataclasses import replace
    full = rr(pred, mu)
    for m in models:
        part = gd.predict(replace(model, site_alpha={**model.site_alpha, m: [np.zeros_like(x) for x in model.site_alpha[m]]}), test)
        own = pred - part
        print(f"ABLATE {tag} joint {full:.2f} %: without {m} {rr(part, mu):.2f} %; "
              f"{m} block carries rms {np.sqrt((own ** 2).mean()):.4f} of target rms {np.sqrt(((mu - mu.mean(0)) ** 2).mean()):.4f}", flush=True)
if os.environ.get("SAVE"):
    np.savez(os.environ["SAVE"], pred=pred, mu=mu, n=[len(a) for a in test], Z=np.concatenate([a.numbers for a in test]))
print(f"RESULT {tag} {'STAGED ' if BASE else ''}{'FACT=' + FACT + ' ' if FACT else ''}rc={cfg.rcut_hard} pairs={PAIRS} dch={os.environ.get('DCH', 1)} {'+'.join(models)} M{M} sigma{sigma:g} xi{xi} N{len(model.train_pred)}: "
      f"test RRMSE {rr(pred, mu):.2f} %  MAE|mu| {mae:.4f} D  "
      f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)

if os.environ.get("XTEST"):         # transfer: score on another dataset's molecules made only of species this model knows
    xf, xkey = os.environ["XTEST"].split(":")
    known = set(int(z) for z in cfg.types)
    other = [a for a in read(xf, index=":") if set(a.numbers) <= known]
    cfg.target_key = xkey
    xm, xp = gd._targets(other, cfg), gd.predict(model, other)
    sc = (xp * xm).sum() / (xp ** 2).sum()                  # least-squares scale: 1 if units agree
    xmae = np.abs(np.linalg.norm(xp, axis=1) - np.linalg.norm(xm, axis=1)).mean()
    print(f"XTEST {tag} {'+'.join(models)} -> {xf} ({len(other)} molecules): RRMSE {rr(xp, xm):.2f} %  MAE|mu| {xmae:.4f} (file units)  "
          f"best scale {sc:.3f} -> RRMSE {rr(sc * xp, xm):.2f} %;  in-domain {rr(pred, mu):.2f} %", flush=True)
