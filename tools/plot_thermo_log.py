"""Plot PE_eV and T_K vs time from a thermo log file.

Input file format (whitespace-separated, first line a `#`-prefixed header):
    # step time_ps PE_eV KE_eV E_tot_eV T_K
    0  0.0000  -4221.119516  0.466305  -4220.653212  400.83
    ...

Edit the PARAMETERS block and run:
    python tools/plot_thermo_log.py
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib
if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════

LOG_PATH  = "plots/Ethanol/Ethanol_mace/thermo.log"
SAVE_PLOT = "plots/ethanol_thermo.png"   # None = don't save
SHOW_PLOT = False                         # True = plt.show() at the end

# ═══════════════════════════════════════════════════════════════════


def main(log_path: str = LOG_PATH,
         save_plot: str | None = SAVE_PLOT,
         show_plot: bool = SHOW_PLOT) -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full = log_path if os.path.isabs(log_path) else os.path.join(root, log_path)

    # Columns: step  time_ps  PE_eV  KE_eV  E_tot_eV  T_K
    data = np.loadtxt(full)
    time_ps = data[:, 1]
    pe_eV   = data[:, 2]
    T_K     = data[:, 5]

    print(f"Loaded {len(data)} samples from {full}")
    print(f"  time:  {time_ps[0]:.4f} → {time_ps[-1]:.4f} ps")
    print(f"  PE  :  mean={pe_eV.mean():.4f}  std={pe_eV.std():.4f}  eV")
    print(f"  T   :  mean={T_K.mean():.2f}  std={T_K.std():.2f}  K")

    fig, (ax_pe, ax_T) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_pe.plot(time_ps, pe_eV, color="#e74c3c", linewidth=0.6)
    ax_pe.set_ylabel("Potential energy (eV)")
    ax_pe.grid(alpha=0.3)
    ax_pe.set_title(f"Thermo log — {os.path.basename(full)}")

    ax_T.plot(time_ps, T_K, color="#3498db", linewidth=0.6)
    ax_T.axhline(T_K.mean(), color="grey", linestyle="--", linewidth=0.8,
                 label=f"mean = {T_K.mean():.1f} K")
    ax_T.set_xlabel("Time (ps)")
    ax_T.set_ylabel("Temperature (K)")
    ax_T.legend(loc="upper right")
    ax_T.grid(alpha=0.3)

    plt.tight_layout()
    if save_plot is not None:
        out = save_plot if os.path.isabs(save_plot) else os.path.join(root, save_plot)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {out}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
