"""Compare two dipole trajectory files (.txt or .dp).

Edit the CONFIG block below, then run the file in your IDE.

Supported formats (auto-detected by extension):
  .txt  3 columns (dx, dy, dz), optional `#`-prefixed header. Typically e*Angstrom.
  .dp   4 columns (dx, dy, dz, |d|), no header. Typically Debye.

SCALE_A / SCALE_B apply a multiplicative factor to each file before comparison
(e.g. 4.80320 to convert e*Angstrom -> Debye).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================ CONFIG (edit these) ============================
FILE_A   = Path("plots/Ethanol/Ethanol_mlatom/ethanol_traj_mlatom_dipoles.txt")
FILE_B   = Path("datasets/ethanol mlatom set/ethanol_traj_dipoles_mlatom.dp")

SCALE_A  = 2.54175   # 1.0 = no rescaling. 4.80320 converts e*Angstrom -> Debye.
SCALE_B  = 1.0

SHOW_PLOT = True                  # True -> render an error-vs-frame figure.
PLOT_OUT  = None                  # Path("plots/dipole_err.png") to save instead of show.
# =============================================================================


def load_dipoles(path: Path) -> np.ndarray:
    """Return an (N, 3) array of (dx, dy, dz) from a .txt or .dp file."""
    suffix = path.suffix.lower()
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if suffix == ".dp":
        if data.shape[1] < 3:
            raise ValueError(f"{path}: expected >=3 columns in .dp, got {data.shape[1]}")
        return data[:, :3]
    if suffix == ".txt":
        if data.shape[1] < 3:
            raise ValueError(f"{path}: expected >=3 columns in .txt, got {data.shape[1]}")
        return data[:, :3]
    # Fallback: take the first three numeric columns.
    return data[:, :3]


def summary(name: str, d: np.ndarray) -> None:
    mag = np.linalg.norm(d, axis=1)
    print(f"\n=== {name} ===")
    print(f"  samples : {len(d)}")
    for i, c in enumerate("xyz"):
        col = d[:, i]
        print(
            f"  d{c}     : mean={col.mean(): .6e}  std={col.std(): .6e}"
            f"  min={col.min(): .6e}  max={col.max(): .6e}"
        )
    print(
        f"  |d|    : mean={mag.mean(): .6e}  std={mag.std(): .6e}"
        f"  min={mag.min(): .6e}  max={mag.max(): .6e}"
    )


def plot_errors(diff: np.ndarray, vec_err: np.ndarray, mag_err: np.ndarray,
                angle_deg: np.ndarray, out: Path | None) -> None:
    frames = np.arange(len(vec_err))
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    for i, c in enumerate("xyz"):
        axes[0].plot(frames, diff[:, i], lw=0.4, label=f"d{c}")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("a - b  (per component)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    axes[1].plot(frames, vec_err, lw=0.4, color="C3")
    axes[1].set_ylabel(r"$\|a - b\|_2$")

    axes[2].plot(frames, mag_err, lw=0.4, color="C2")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].set_ylabel(r"$|a| - |b|$")

    axes[3].plot(frames, angle_deg, lw=0.4, color="C4")
    axes[3].set_ylabel("angle (deg)")
    axes[3].set_xlabel("frame index")

    fig.suptitle("Dipole error vs frame  (A vs B)")
    fig.tight_layout()

    if out is not None:
        fig.savefig(out, dpi=150)
        print(f"\n[plot] saved to {out}")
    else:
        plt.show()
    plt.close(fig)


def compare(a: np.ndarray, b: np.ndarray, plot: bool, plot_out: Path | None) -> None:
    n = min(len(a), len(b))
    if len(a) != len(b):
        print(
            f"\n[warn] length mismatch: A has {len(a)} rows, B has {len(b)}. "
            f"Comparing first {n} rows."
        )
    a, b = a[:n], b[:n]

    diff = a - b                              # (N, 3)
    abs_diff = np.abs(diff)                   # (N, 3)
    vec_err = np.linalg.norm(diff, axis=1)    # (N,) Euclidean per-frame error
    mag_a = np.linalg.norm(a, axis=1)
    mag_b = np.linalg.norm(b, axis=1)
    mag_err = mag_a - mag_b                   # signed magnitude difference

    # Cosine similarity per frame (clip to avoid acos blowups from FP noise).
    denom = mag_a * mag_b
    safe = denom > 0
    cos = np.ones(n)
    cos[safe] = np.clip(np.einsum("ij,ij->i", a[safe], b[safe]) / denom[safe], -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))

    print("\n=== A - B ===")
    print("  per-component absolute error |a_i - b_i|:")
    for i, c in enumerate("xyz"):
        col = abs_diff[:, i]
        print(
            f"    d{c}: mean={col.mean(): .6e}  median={np.median(col): .6e}"
            f"  min={col.min(): .6e}  max={col.max(): .6e}"
            f"  rmse={np.sqrt((col**2).mean()): .6e}"
        )

    print("  vector error ||a - b||_2 (per frame):")
    print(
        f"    mean={vec_err.mean(): .6e}  median={np.median(vec_err): .6e}"
        f"  min={vec_err.min(): .6e}  max={vec_err.max(): .6e}"
        f"  rmse={np.sqrt((vec_err**2).mean()): .6e}"
    )
    imax = int(np.argmax(vec_err))
    imin = int(np.argmin(vec_err))
    print(f"    worst frame: idx={imax}  a={a[imax]}  b={b[imax]}  err={vec_err[imax]:.6e}")
    print(f"    best  frame: idx={imin}  err={vec_err[imin]:.6e}")

    print("  magnitude error |a| - |b|:")
    print(
        f"    mean={mag_err.mean(): .6e}  median={np.median(mag_err): .6e}"
        f"  min={mag_err.min(): .6e}  max={mag_err.max(): .6e}"
        f"  mae={np.abs(mag_err).mean(): .6e}"
    )

    print("  direction error (angle between vectors, degrees):")
    print(
        f"    mean={angle_deg.mean(): .4f}  median={np.median(angle_deg): .4f}"
        f"  min={angle_deg.min(): .4f}  max={angle_deg.max(): .4f}"
    )

    ss_res = (vec_err**2).sum()
    ss_tot = ((a - a.mean(axis=0))**2).sum()
    if ss_tot > 0:
        print(f"  R^2 (A vs B, treating B as prediction of A): {1 - ss_res/ss_tot:.6f}")

    if plot:
        plot_errors(diff, vec_err, mag_err, angle_deg, plot_out)


def main() -> None:
    a = load_dipoles(FILE_A) * SCALE_A
    b = load_dipoles(FILE_B) * SCALE_B

    print(f"A: {FILE_A}  (scale={SCALE_A})")
    print(f"B: {FILE_B}  (scale={SCALE_B})")

    summary("A", a)
    summary("B", b)
    compare(a, b, plot=SHOW_PLOT or PLOT_OUT is not None, plot_out=PLOT_OUT)


if __name__ == "__main__":
    main()
