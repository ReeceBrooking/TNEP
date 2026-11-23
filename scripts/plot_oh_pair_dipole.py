"""Two-body dipole test: an isolated O–H pair at separation d, dipole along the bond from
dipole GAPs fitted on bulk water with no cross term — one Chebyshev channel (CHANNELS=1,
default) or ten (CHANNELS=10) — against the fixed-charge line q·d. Shows how the readout's cutoff factor, not learned physics, decides
whether μ(2d) > μ(d).   CHANNELS=1|10 python scripts/plot_oh_pair_dipole.py → plots/oh_pair_<C>channel_dipole.{png,csv}"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests")); os.chdir(ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
import gap_dipole as gd
from test_gap_dipole import _cfg

L, Q, D0 = 20.0, 0.4, 0.97                           # box edge (Å), reference partial charge (e), equilibrium O–H (Å)
pair = lambda d: Atoms("OH", positions=[[L / 2] * 3, [L / 2] * 3 + d * np.array([1.0, 0, 0])], cell=L * np.eye(3), pbc=True)
ds = np.round(np.arange(0.6, 3.21, 0.1), 2)
frames = [pair(d) for d in ds]
BASE = dict(data_path="datasets/train_waterbulk.xyz", test_data_path="datasets/test_waterbulk.xyz", total_N=None, target_mode=1,
            rcut_hard=3.5, rcut_soft=3.0, dipole_radial="chebyshev", dipole_species_pairs="symmetric",
            gap_kernel_power=2, convert_dipole_to_eangstrom=False, gap_cross_channels=0, l_max=7, alpha_max=7, compress_mode="trivial",
            gap_sparse_per_species=250, gap_sigma=0.003, dipole_channels=int(os.environ.get("CHANNELS", 1)))
C = BASE["dipole_channels"]; STEM = f"plots/oh_pair_{C}channel_dipole"
VARIANTS = [("cosine² cutoff, rc 3.5 Å", dict(dipole_cutoff="cosine2"), "#2c7fb8"),
            ("cosine cutoff, rc 3.5 Å", dict(dipole_cutoff="cosine"), "#d95f0e"),
            ("cosine² cutoff, rc 4.0 Å", dict(dipole_cutoff="cosine2", rcut_hard=4.0, rcut_soft=3.5), "#31a354")]
rows = {}
for lab, kw, _ in VARIANTS:
    m = gd.fit_gap_dipole(_cfg(**{**BASE, **kw})); rows[lab] = gd.predict(m, frames)[:, 0]
np.savetxt(STEM + ".csv", np.column_stack([ds] + [rows[l] for l, *_ in VARIANTS] + [Q * ds / 0.529177]),
           delimiter=",", header="d_A," + ",".join(f"mu_x_ebohr[{l}]" for l, *_ in VARIANTS) + ",q_d_ebohr(q=0.4e)", comments="")
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(ds, Q * ds / 0.529177, color="0.45", ls="--", lw=1.6, label="fixed charge, μ = q·d (q = 0.4 e)")
for lab, _, col in VARIANTS:
    ax.plot(ds, rows[lab], color=col, lw=2, marker="o", ms=4, label=f"{C}-channel dipole GAP, {lab}")
for d, name in ((D0, "d"), (2 * D0, "2d")):
    ax.axvline(d, color="0.75", lw=1, ls=":"); ax.text(d, ax.get_ylim()[0] * 0.97, name, ha="center", va="bottom", color="0.4")
ax.axhline(0, color="0.8", lw=1)
ax.set_xlabel("O–H separation d (Å)"); ax.set_ylabel("dipole along the bond μ_x (e·bohr)")
ax.set_title(f"Isolated O–H pair: dipole GAPs fitted on bulk water, {C} Chebyshev channel{'s' if C > 1 else ''}, no cross term")
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left" if C > 1 else "lower right")
fig.savefig(STEM + ".png", dpi=150, bbox_inches="tight"); print("saved", STEM + ".png")
