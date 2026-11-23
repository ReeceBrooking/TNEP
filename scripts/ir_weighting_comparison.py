import os, sys, numpy as np
sys.path.insert(0, "/home/reece/TNEP"); os.chdir("/home/reece/TNEP")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from spectroscopy import compute_ir_spectrum
S = os.path.dirname(os.path.abspath(__file__)); T = 298.15
XU = "/home/reece/Work/GPUMD Testing/Water Test (Xu)"
A   = np.load("plots/water_bulk_GPUMD_traj_dipoles.npy").astype(np.float64)          # adapted TNEP, same traj
X   = np.loadtxt(f"{XU}/dipole.out")[-200000:, 1:]                                  # Xu TNEP, same traj (last NVE segment)
X0  = np.loadtxt(f"{XU}/Xu output/dipole.out")[:, 1:]                               # Xu TNEP, Xu's own traj
xu_pub = np.loadtxt(f"{XU}/Xu output/ir_smooth.txt")                                 # Xu's script output (omega^2, box-10)
rows = [l.split() for l in open("/home/reece/Work/Liquid Water IR Max et al.txt").read().splitlines()[1:]]
exp = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows if len(r) >= 3])  # nu, n(H2O), k(H2O)
nu_e, k_e = exp[:, 0], exp[:, 2]; alpha_e = 4*np.pi*nu_e*k_e
assert len(A) == len(X) == 200000

def spec(mu, qc, kind="gaussian"):
    f, I, _, _ = compute_ir_spectrum(mu, dt_fs=1.0, temperature=T, quantum_correction=qc, smooth_k=10, smooth_kind=kind, acf_ratio=0.1)
    return f, I
def norm(f, I, lo=80, hi=2500):            # Xu Fig. 4a convention: unit integral over 80-2500 cm^-1
    m = (f >= lo) & (f <= hi); return I / np.trapz(I[m], f[m])
def on(fgrid, f, I): return np.interp(fgrid, f, I)
def ratio(f, I):                            # stretch (3000-3800) / bend (1500-1800) band integrals
    s = (f>=3000)&(f<=3800); b = (f>=1500)&(f<=1800); return np.trapz(I[s], f[s]) / np.trapz(I[b], f[b])
grid = np.arange(80, 4000, 5.0)
def dist(f1, I1, f2, I2, lo=80, hi=4000):   # L1 distance of unit-integral spectra on [lo,hi] (0 = identical, 2 = disjoint)
    a = on(grid, f1, I1); b = on(grid, f2, I2); m = (grid>=lo)&(grid<=hi)
    a = a[m]/np.trapz(a[m], grid[m]); b = b[m]/np.trapz(b[m], grid[m]); return np.trapz(np.abs(a-b), grid[m])

# sanity: our pipeline vs Xu's script on Xu's own dipoles (omega^2, box smoothing)
f0, I0 = spec(X0, "quadratic", "box")
print(f"pipeline check: ours(omega^2,box10) vs Xu ir_smooth.txt on Xu's dipoles: L1 = {dist(f0, I0, xu_pub[:,0], xu_pub[:,1]):.4f}  (0 = identical)")

print("\nstretch/bend band-integral ratio (3000-3800 / 1500-1800), 80-2500-normalised:")
print(f"  experiment alpha=4*pi*nu*k : {ratio(nu_e, alpha_e):.2f}    experiment k only: {ratio(nu_e, k_e):.2f}    Xu published (omega^2): {ratio(xu_pub[:,0], xu_pub[:,1]):.2f}")
res = {}
for qc in ("quadratic", "bose", "linear", "none"):      # "bose" = the old ω(1−e^−x) (formerly named "harmonic")
    fa, Ia = spec(A, qc); fx, Ix = spec(X, qc); res[qc] = (fa, Ia, fx, Ix)
    print(f"  {qc:9s}: adapted TNEP {ratio(fa, Ia):5.2f}   Xu TNEP (same traj) {ratio(fx, Ix):5.2f}")

print("\nL1 distance between unit-integral spectra on 80-4000 cm^-1 (lower = closer):")
print(f"{'weighting':9s} {'adapted vs Xu(same traj)':>26s} {'adapted vs Xu published':>24s} {'Xu(same traj) vs Xu published':>30s} {'adapted vs exp alpha':>21s} {'Xu vs exp alpha':>16s} {'adapted vs exp k':>17s} {'Xu vs exp k':>12s}")
for qc, (fa, Ia, fx, Ix) in res.items():
    print(f"{qc:9s} {dist(fa,Ia,fx,Ix):26.3f} {dist(fa,Ia,xu_pub[:,0],xu_pub[:,1]):24.3f} {dist(fx,Ix,xu_pub[:,0],xu_pub[:,1]):30.3f} {dist(fa,Ia,nu_e,alpha_e):21.3f} {dist(fx,Ix,nu_e,alpha_e):16.3f} {dist(fa,Ia,nu_e,k_e):17.3f} {dist(fx,Ix,nu_e,k_e):12.3f}")

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
for ax, qc, lab in zip(axes, ("quadratic", "harmonic"), ("ω²·M(ω)  (Xu Eq. 9, GPUMD script)", "ω·(1−e^(−ℏω/kT))·M(ω)  (code's 'harmonic', no QCF)")):
    fa, Ia, fx, Ix = res[qc]
    ax.plot(nu_e, norm(nu_e, alpha_e), color="0.4", lw=3, alpha=0.5, label="experiment, α = 4πν̃k (Max et al.)")
    ax.plot(xu_pub[:,0], norm(xu_pub[:,0], xu_pub[:,1]), "k--", lw=1.2, label="Xu TNEP, Xu's trajectory, Xu's script (ω²)")
    ax.plot(fx, norm(fx, Ix), color="tab:blue", lw=1.5, label="Xu TNEP dipoles, this trajectory")
    ax.plot(fa, norm(fa, Ia), color="tab:red", lw=1.5, label="adapted TNEP dipoles, this trajectory")
    ax.set_title(lab); ax.set_xlim(0, 4000); ax.set_ylim(bottom=0); ax.set_ylabel("intensity (unit integral 80–2500 cm⁻¹)")
axes[0].legend(fontsize=8); axes[1].set_xlabel("wavenumber (cm⁻¹)")
plt.tight_layout(); out = f"{S}/ir_weighting_comparison.png"; plt.savefig(out, dpi=130); print("\nsaved", out)
