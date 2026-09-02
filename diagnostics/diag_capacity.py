"""Capacity diagnostics for the single-channel neighbor dipole readout.

All baselines are closed-form least squares on the training set — no ANN,
no SNES. They bracket what the functional form can express:
  A. oracle per-frame s (n dof/frame)     -> verifies V + Laplacian argument (expect ~0)
  B. rigid per-type s, single f_c channel -> what a type-constant field gives (2 params)
  C. rigid per-type, covalent pairs only  -> H-bond pollution probe (2 params)
  D. rigid per-type x 4 radial bins       -> multi-channel rigid (8 params)
"""
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

RC = 4.0          # user's config: dipole_rc=None -> inherits rcut_hard=4.0
BINS = [(0.0, 1.2), (1.2, 2.2), (2.2, 3.0), (3.0, RC)]

def fc(r, rc=RC):
    return np.where(r < rc, (0.5 * (1 + np.cos(np.pi * r / rc))) ** 2, 0.0)

frames = read("datasets/train_waterbulk.xyz", index=":")
print(f"{len(frames)} frames, {len(frames[0])} atoms")

mu = np.array([f.info.get("dipole", f.calc.results.get("dipole") if f.calc else None)
               for f in frames])                             # e*bohr native
print(f"target per-component std: {mu.std(axis=0)}, |mu| mean {np.linalg.norm(mu,axis=1).mean():.2f}")

T = {8: 0, 1: 1}  # O, H
feats = []        # [F, 3, 2*nbins]  per-type-per-bin geometry vectors
oracle_res = []
for f in frames:
    i, j, D = neighbor_list("ijD", f, RC)                    # ordered pairs, D = r_j - r_i
    r = np.linalg.norm(D, axis=1)
    w = fc(r)
    v_pair = -w[:, None] * D                                 # s_i * (r_i - r_j) contribution
    Z = f.numbers
    # features: per (type of i, bin)
    cols = []
    for t in (8, 1):
        for lo, hi in BINS:
            m = (Z[i] == t) & (r >= lo) & (r < hi)
            cols.append(v_pair[m].sum(axis=0))
    feats.append(np.stack(cols, axis=1))                     # [3, 8]

X = np.stack(feats)                                          # [F, 3, 8]
F = len(frames)

def lsq_rmse(cols):
    A = X[:, :, cols].reshape(F * 3, len(cols))
    b = mu.reshape(F * 3)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ c
    rmse = np.sqrt(np.mean((pred - b) ** 2) * 3)             # total-vector RMSE
    return rmse, c

# oracle done properly (per frame, all-atom dof)
res = []
for k, f in enumerate(frames):
    i, j, D = neighbor_list("ijD", f, RC)
    w = fc(np.linalg.norm(D, axis=1))
    v_pair = -w[:, None] * D
    V = np.zeros((len(f), 3))
    np.add.at(V, i, v_pair)
    s_hat, *_ = np.linalg.lstsq(V.T, mu[k], rcond=None)
    res.append(V.T @ s_hat - mu[k])
res = np.array(res)
print(f"A. oracle per-frame s      : RMSE {np.sqrt((res**2).sum(1).mean()):.4f}")


rmse_all, _ = lsq_rmse(list(range(8)))
# single-channel rigid: sum bins per type
Xs = X.reshape(F, 3, 2, 4).sum(axis=3)                       # [F,3,2]
A = Xs.reshape(F * 3, 2); b = mu.reshape(F * 3)
c, *_ = np.linalg.lstsq(A, b, rcond=None)
rmse_single = np.sqrt(np.mean((A @ c - b) ** 2) * 3)
print(f"B. rigid per-type, 1 chan  : RMSE {rmse_single:.4f}   c={c}")

rmse_cov, c_cov = lsq_rmse([0, 4])                           # covalent bin only (r<1.2)
print(f"C. rigid, covalent only    : RMSE {rmse_cov:.4f}   c={c_cov}")

print(f"D. rigid x 4 radial bins   : RMSE {rmse_all:.4f}")
print(f"   (trained ANN plateau ~2.4-2.6; target total std {np.sqrt((mu.var(axis=0)).sum()):.2f})")
