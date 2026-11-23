"""Plot a TurboGAP thermo.log (Step, Time, Temperature, E_kin, E_pot, Pressure).

Usage: python scripts/plot_thermo.py [path/to/thermo.log]
Default: datasets/turbogap_waterbulk/thermo.log. Saves <log>_plot.png.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "datasets/turbogap_waterbulk/thermo.log"
step, t, T, ekin, epot, P = np.loadtxt(path).T
t_ps = t / 1000.0  # fs -> ps

def running_mean(y, w=200):
    # 'valid' mode: no fake edge dips; returned series is (w-1) shorter.
    w = min(w, len(y))
    return np.convolve(y, np.ones(w) / w, mode="valid"), t_ps[w // 2:len(y) - (w - 1) + w // 2]

panels = [
    (T, "Temperature (K)", "tab:red"),
    (P, "Pressure (bar)", "tab:purple"),
    (ekin, "E_kin (eV)", "tab:orange"),
    (epot, "E_pot (eV)", "tab:blue"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
for ax, (y, label, color) in zip(axes.flat, panels):
    ax.plot(t_ps, y, color=color, lw=0.3, alpha=0.4)
    ym, tm = running_mean(y)
    ax.plot(tm, ym, color=color, lw=1.2,
            label=f"mean {y.mean():.2f} ± {y.std():.2f}")
    ax.set_ylabel(label)
    ax.legend(loc="best", fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("Time (ps)")
fig.suptitle(path)
fig.tight_layout()
out = path.rsplit(".", 1)[0] + "_plot.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
plt.show()
