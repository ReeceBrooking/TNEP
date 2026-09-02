"""Step 0: rigid per-type LSQ floor for the screened radial R(r)=r^(z-1)*f_cub(r)."""
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

RC = 4.0
frames = read("datasets/train_waterbulk.xyz", index=":")
mu = np.array([f.info.get("dipole", f.calc.results.get("dipole") if f.calc else None)
               for f in frames])

def fcub(r):
    x = r / RC
    return np.where(r < RC, 1 - 3 * x**2 + 2 * x**3, 0.0)

pairs = []
for f in frames:
    i, j, D = neighbor_list("ijD", f, RC)
    keep = i != j
    pairs.append((i[keep], f.numbers[i[keep]], -D[keep].astype(np.float64)))

F = len(frames)
print("zeta   rigid-LSQ RMSE (total e·bohr)   c_O")
for z in [-4, -3, -2, -1, 0, 1, 2]:
    X = np.zeros((F, 3, 2))
    for k, (i, Zi, d) in enumerate(pairs):
        r = np.linalg.norm(d, axis=1)
        w = r**(z - 1) * fcub(r)
        v = w[:, None] * d
        X[k, :, 0] += v[Zi == 8].sum(0)
        X[k, :, 1] += v[Zi == 1].sum(0)
    A = X.reshape(F * 3, 2)
    c, *_ = np.linalg.lstsq(A, mu.reshape(F * 3), rcond=None)
    rmse = np.sqrt(np.mean((A @ c - mu.reshape(F * 3))**2) * 3)
    print(f"{z:+d}     {rmse:8.4f}                     {c[0]:+.4f}")
print("(reference floors: single-chan f_c rigid 5.50 | covalent-only 1.12 | 4-bin 0.71 | trained C=4 smoke 0.86)")
