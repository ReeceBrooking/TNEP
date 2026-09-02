"""Step 0 for the basis extensions: rigid per-type x per-channel LSQ floors
over radial family x cutoff x C. Pattern of diag_capacity.py baseline D."""
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

RC = 4.0
frames = read("/home/reece/TNEP/datasets/train_waterbulk.xyz", index=":")
mu = np.array([f.info.get("dipole", f.calc.results.get("dipole") if f.calc else None)
               for f in frames])
F = len(frames)

pairs = []
for f in frames:
    i, j, D = neighbor_list("ijD", f, RC)
    keep = i != j
    pairs.append((i[keep], f.numbers[i[keep]], -D[keep].astype(np.float64)))

CUTS = {
    "cosine2": lambda r: ((1 + np.cos(np.pi * r / RC)) / 2) ** 2,
    "cubic":   lambda r: 1 - 3 * (r / RC) ** 2 + 2 * (r / RC) ** 3,
    "cosine":  lambda r: (1 + np.cos(np.pi * r / RC)) / 2,
    "tanh3":   lambda r: np.tanh(1 - r / RC) ** 3 / np.tanh(1.0) ** 3,
}

def radial(name, r, C):
    if name == "chebyshev":
        return np.polynomial.chebyshev.chebvander(2 * r / RC - 1, C - 1)
    if name == "warped":
        return np.polynomial.chebyshev.chebvander(2 * (r / RC - 1) ** 2 - 1, C - 1)
    scale = float(name.split("x")[1])          # "gaussx0.5" etc.
    mus = np.linspace(0, RC, C + 2)[1:-1]
    sig = scale * (mus[1] - mus[0]) if C > 1 else scale * RC / 2
    return np.exp(-((r[:, None] - mus[None, :]) ** 2) / (2 * sig ** 2))

def floor(rname, cname, C):
    X = np.zeros((F, 3, 2 * C))
    for k, (i, Zi, d) in enumerate(pairs):
        r = np.linalg.norm(d, axis=1)
        Rw = radial(rname, r, C) * CUTS[cname](r)[:, None]      # [P, C]
        v = Rw[:, None, :] * d[:, :, None]                       # [P, 3, C]
        for t, Z in enumerate((8, 1)):
            m = Zi == Z
            X[k, :, t * C:(t + 1) * C] += v[m].sum(0)
    A = X.reshape(F * 3, 2 * C)
    c, *_ = np.linalg.lstsq(A, mu.reshape(F * 3), rcond=None)
    return np.sqrt(np.mean((A @ c - mu.reshape(F * 3)) ** 2) * 3)

rads = ["chebyshev", "warped", "gaussx0.5", "gaussx1.0", "gaussx2.0"]
print(f"{'radial':>10} | " + " | ".join(f"{c:>7}" for c in CUTS) + "   (C=4 / C=10, total e·bohr)")
for rname in rads:
    row = []
    for cname in CUTS:
        row.append(f"{floor(rname, cname, 4):5.3f}/{floor(rname, cname, 10):5.3f}")
    print(f"{rname:>10} | " + " | ".join(f"{x:>11}" for x in row))
print("(refs: trained cheb C=4 smoke 0.86 | 4-bin rigid 0.71 | screened z=-2 rigid 2.7)")
