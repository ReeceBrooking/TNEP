"""Our GAP Raman spectrum (classical ω²·L footing) overlaid on the experimental curves of Brooker et al. 1989 and
Morawietz et al. 2018 and on Xu et al.'s own TNEP curve, all as digitised from Xu et al. 2024 Fig. 4
(scripts/digitise_xu2024_fig4.py). Presentation as in that figure: anisotropic normalised by ∫80–2500, isotropic
by ∫1000–2500, axis break at 2500 cm⁻¹ with the right panel drawn ×25 (aniso) / ×180 (iso) smaller.

    python scripts/plot_raman_vs_experiment.py      → plots/raman_water_vs_experiment_{aniso,iso}.png"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import spectroscopy as sp

DIG = os.path.join(ROOT, "datasets", "raman_water_xu2024_fig4")
OURS = [("GAP, local α on χ labels (this work, 200 ps)", "plots/water_bulk_GPUMD_traj_polarizabilities_chi.npy", 0.025, dict(color="#1f77b4", lw=1.6))]
DIGITISED = [("Brooker et al. 1989 (exp.)", "brooker", dict(color="0.55", ls="--")),
             ("Morawietz et al. 2018 (exp.)", "morawietz", dict(color="0.3", ls="-.")),
             ("TNEP, Xu et al. 2024 Fig. 4", "tnep", dict(color="#d73027", lw=2.2, alpha=0.8))]
NORM = {"aniso": (80.0, 2500.0, 25.0), "iso": (1000.0, 2500.0, 180.0)}


def norm(f, y, lo, hi):
    m = (f >= lo) & (f <= hi)
    return y / np.trapz(y[m], f[m])


def main():
    ours = {}
    for lab, path, ratio, style in OURS:
        f, _, _, Li, La = sp.compute_raman_spectrum(np.load(os.path.join(ROOT, path)), dt_fs=2.0, acf_ratio=ratio, temperature=298.15, quantum_correction="harmonic")
        ours[lab] = (f, {"iso": Li, "aniso": La}, style)
    for panel, (lo, hi, fac) in NORM.items():
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True,            # one y scale across the break
                                       gridspec_kw={"width_ratios": [2420, 1500], "wspace": 0.04})
        curves = []
        for lab, key, style in DIGITISED:
            d = np.loadtxt(os.path.join(DIG, f"{key}_{panel}.csv"), delimiter=",")
            curves.append((lab, d[:, 0], norm(d[:, 0], d[:, 1], lo, hi), style))
        for lab, (f, L, style) in ours.items():
            curves.append((lab, f, norm(f, L[panel], lo, hi), style))
        for lab, f, y, style in curves:
            m = (f >= lo) & (f <= 2500); axL.plot(f[m], y[m], label=lab, **style)
            m = f >= 2500; axR.plot(f[m], y[m] / fac, **style)
        axL.set_xlim(lo, 2500); axR.set_xlim(2500, 4000); axR.text(0.06, 0.9, f"×{fac:g}", transform=axR.transAxes, fontsize=12)
        for ax in (axL, axR):
            ax.set_ylim(0, None); ax.grid(alpha=0.3)
        axL.tick_params(left=False, labelleft=False); axR.tick_params(left=False, labelleft=False); axL.spines["right"].set_visible(False); axR.spines["left"].set_visible(False)
        axL.set_ylabel(f"{'anisotropic' if panel == 'aniso' else 'isotropic'} Raman intensity (arb. units)")
        fig.legend(*axL.get_legend_handles_labels(), fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False)   # above both panels: never covers a curve
        fig.supxlabel("Wavenumber (cm⁻¹)")
        out = os.path.join(ROOT, "plots", f"raman_water_vs_experiment_{panel}.png"); fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
