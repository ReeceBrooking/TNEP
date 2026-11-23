"""IR absorption of liquid water: experiment vs TNEP vs this work, Xu et al. 2024 Fig. 4(a) presentation
(unit integral over 80–2500 cm⁻¹). All simulated curves use the harmonic quantum correction, i.e. ω²·M(ω).

    experiment : Max & Chapados optical constants (n, k) → α·n = 4π ν̃ k n, the quantity ω²·M is proportional to
                 (α alone is also shown); Downing & Williams as digitised from Fig. 4(a)
    TNEP       : Xu et al.'s published curve (digitised from Fig. 4(a)) and their dipole model run on-the-fly in
                 GPUMD on our 200 ps trajectory (~/Work/GPUMD Testing/Water Test (Xu)/dipole.out)
    this work  : plots/water_bulk_GPUMD_traj_dipoles.npy (the TNEP dipole model trained in this code, same trajectory)

    python scripts/plot_ir_vs_experiment.py      → plots/ir_water_vs_experiment.png"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import spectroscopy as sp

DIG = os.path.join(ROOT, "datasets", "raman_water_xu2024_fig4")
MAX = os.path.expanduser("~/Work/Liquid Water IR Max et al.txt")
XU_DIP = os.path.expanduser("~/Work/GPUMD Testing/Water Test (Xu)/dipole.out")
OURS = os.path.join(ROOT, "plots", "water_bulk_GPUMD_traj_dipoles.npy")
LO, HI, T = 80.0, 2500.0, 298.15


def norm(f, y):
    m = (f >= LO) & (f <= HI)
    return y / np.trapz(y[m], f[m])


def sim(mu, dt_fs):
    f, I, _, _ = sp.compute_ir_spectrum(mu, dt_fs=dt_fs, temperature=T, quantum_correction="harmonic",
                                        smooth_k=10, smooth_kind="gaussian", acf_ratio=5000.0 / (len(mu) * dt_fs))   # 5 ps ACF
    return f, I


def main():
    curves = []
    rows = [l.split() for l in open(MAX).read().splitlines()[1:]]
    e = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows if len(r) >= 3])        # ν̃, n, k
    nu, n, k = e.T
    curves.append(("Max & Chapados 2009 (exp.), α·n", nu, norm(nu, 4 * np.pi * nu * k * n), dict(color="0.35", ls="-.")))
    curves.append(("Max & Chapados 2009 (exp.), α", nu, norm(nu, 4 * np.pi * nu * k), dict(color="0.35", ls=":", lw=0.9)))
    for lab, key, style in (("Downing & Williams 1975 (exp.), digitised Fig. 4a", "downing", dict(color="0.6", ls="--")),
                            ("TNEP, Xu et al. 2024 Fig. 4a (digitised)", "tnep", dict(color="#d73027", lw=2.2, alpha=0.8))):
        d = np.loadtxt(os.path.join(DIG, f"{key}_ir.csv"), delimiter=","); curves.append((lab, d[:, 0], norm(d[:, 0], d[:, 1]), style))
    xu = np.loadtxt(XU_DIP)[-200000:, 1:]                       # 1 fs, the NVE segment
    f, I = sim(xu, 1.0); curves.append(("TNEP dipole model (Xu) on our trajectory, 200 ps", f, norm(f, I), dict(color="#fc8d59", lw=1.2)))
    ours = np.load(OURS).astype(np.float64)
    f, I = sim(ours, 1.0); curves.append(("TNEP (this code), same trajectory", f, norm(f, I), dict(color="black", lw=1.4)))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for lab, f, y, style in curves:
        m = (f >= LO) & (f <= 4000); ax.plot(f[m], y[m], label=lab, **style)
    ax.set_xlim(LO, 4000); ax.set_ylim(0, None); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_xlabel("wavenumber (cm⁻¹)"); ax.set_ylabel("IR intensity (normalised by ∫80–2500 cm⁻¹)")
    ax.set_title("Liquid water IR: experiment vs TNEP vs this work — harmonic QCF (ω²·M), Xu et al. Fig. 4(a) presentation")
    out = os.path.join(ROOT, "plots", "ir_water_vs_experiment.png"); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
    B = lambda f, y, a, b: np.trapz(y[(f >= a) & (f < b)], f[(f >= a) & (f < b)])
    print(f"{'curve':52s} Hbond(80-300)/bend  lib(300-1000)/bend  stretch(2800-3900)/bend")
    for lab, f, y, _ in curves:
        bd = B(f, y, 1400, 1900); print(f"{lab:52s} {B(f, y, 80, 300) / bd:18.2f} {B(f, y, 300, 1000) / bd:19.2f} {B(f, y, 2800, 3900) / bd:24.2f}")


if __name__ == "__main__":
    main()
