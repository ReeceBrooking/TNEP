from __future__ import annotations

import os
from typing import TYPE_CHECKING

import tensorflow as tf
import numpy as np

# Headless-safe backend (see plotting.py).
if not os.environ.get("MPLBACKEND") and not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from data import cell_to_box, pack_chunk_from_flat

if TYPE_CHECKING:
    from ase import Atoms
    from TNEP import TNEP


def compute_dipole_acf(dipoles: np.ndarray) -> np.ndarray:
    """Compute the dipole autocorrelation function via the Wiener-Khinchin theorem.

    Uses FFT for O(N log N) efficiency instead of direct O(N²) summation.
    Sums (not averages) over x, y, z components — i.e. computes the dot-
    product ACF ⟨μ(0)·μ(τ)⟩ exactly as in Xu et al., J. Chem. Theory
    Comput. 2024, 20, 3273-3284, Eq. 9.

    Estimator: **biased**. Each lag is normalised by the full trajectory
    length T (not by (T-τ)). This matches GPUMD and Xu's reference
    implementation. The biased form lets the ACF decay smoothly toward
    zero at large τ — the unbiased 1/(T-τ) form would amplify the noisy
    tail (few overlapping pairs at high lag) and leak that noise into
    the spectrum.

    Args:
        dipoles : [T, 3] ndarray — dipole moment trajectory (one per MD frame)

    Returns:
        acf : [T] ndarray — dipole autocorrelation function (e²·Å² units
              if dipoles are in e·Å). NOT normalised to acf[0] = 1.
    """
    T = dipoles.shape[0]
    # Zero-pad to ≥ 2T-1 for linear (non-circular) correlation. Round up
    # to a fast FFT length so NumPy/scipy can hit their O(N log N) paths
    # cleanly instead of a slow prime-factor decomposition. `next_fast_len`
    # lives in scipy.fft on modern releases; fall back to plain 2*T when
    # scipy isn't available.
    try:
        from scipy.fft import next_fast_len as _nfl
        n_fft = int(_nfl(2 * T - 1))
    except ImportError:
        n_fft = 2 * T
    acf = np.zeros(T)
    for dim in range(3):
        d = dipoles[:, dim]
        # Wiener-Khinchin: ACF = IFFT(|FFT(d)|²)
        fd = np.fft.rfft(d, n=n_fft)
        power = np.real(fd * np.conj(fd))
        full_acf = np.fft.irfft(power, n=n_fft)[:T]
        acf += full_acf
    # Biased estimator: divide by T (constant), not (T-τ). Sums (not
    # averages) over the three spatial components per Xu Eq. 9.
    acf /= float(T)
    return acf


def compute_ir_spectrum(dipoles: np.ndarray, dt_fs: float = 1.0, window: str | None = 'hann',
                         max_freq_cm: float = 4000.0, acf_ratio: float = 0.1,
                         smooth_k: int = 10,
                         smooth_kind: str = "gaussian",
                         temperature: float = 300.0,
                         quantum_correction: str = "harmonic",
                         power_dc_cutoff_cm: float = 100.0,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute IR absorption spectrum from a dipole moment trajectory.

    Follows GPUMD / Xu et al., J. Chem. Theory Comput. 2024, 20, 3273–3284:
        1. Subtract mean dipole to remove DC component
        2. Compute dipole autocorrelation function C(τ) = <μ(0)·μ(τ)>
        3. Truncate ACF to first acf_ratio of the trajectory (default 10%)
        4. Apply Hann window and Kronecker doubling factor
        5. Cosine transform to obtain line shape M(ω) (guaranteed non-negative)
        6. IR absorption: σ(ω) ∝ ω · (1 − e^(−ℏω/kT)) · M(ω)
           — the "harmonic" quantum correction. Reduces to ω² · M(ω) only
           in the classical limit ℏω ≪ kT (i.e. ν̃ ≪ 210 cm⁻¹ at 300 K).
           At ν̃ ≈ 3000 cm⁻¹ at 300 K, the classical ω² form overweights
           by a factor of ℏω/kT ≈ 14, masking lower-frequency modes —
           hence the OH/CH stretch always dominates an ω²-weighted plot.
        7. Smooth with a moving average of width smooth_k

    Args:
        dipoles            : [T, 3] ndarray — dipole trajectory (e·Å, one per frame)
        dt_fs              : float — timestep between frames in femtoseconds
        window             : str or None — window function ('hann', 'blackman', or None)
        max_freq_cm        : float — maximum frequency to return in cm⁻¹
        acf_ratio          : float — fraction of trajectory to use as max ACF lag (default 0.1)
        smooth_k           : int — smoothing strength. Higher = smoother
                              spectrum (broader peaks, less noise). For the
                              default smooth_kind="gaussian" this is the
                              FWHM of the Gaussian kernel **in frequency bins**;
                              for smooth_kind="box" it is the moving-average
                              window width in bins. 0 = disable.
                              Typical FWHM: 5-30 cm⁻¹ for clean spectra at
                              dt_fs=0.25 fs (bin width ≈ 0.4-2 cm⁻¹), so
                              smooth_k=10-50 covers most cases.
        smooth_kind        : str — "gaussian" (default; no spectral ringing,
                              recommended) or "box" (moving average; matches
                              GPUMD notebook but has sinc-like sidelobes
                              around sharp peaks).
        temperature        : float — simulation temperature in K (only used when
                              quantum_correction != "classical"). Default 300 K.
        quantum_correction : str — IR absorption weighting:
                              "harmonic"  : ω · (1 − e^(−ℏω/kT)) · M(ω)  — GPUMD default
                              "classical" : ω² · M(ω)                    — Xu Eq. 1 original
                              "quadratic" : alias for "classical" (ω²·M(ω))
                              "linear"    : ω · M(ω)                     — high-freq limit
                              "none"      : M(ω) (power spectrum only)
        power_dc_cutoff_cm : float — frequencies below this are excluded
                              from the power-spectrum peak-normaliser.
                              The raw M(ω) has a huge DC peak that would
                              otherwise crush all vibrational features to
                              ~1 % of full scale. Set 0 to keep the DC
                              bin in the normaliser (matches the IR
                              intensity normaliser, which is naturally
                              ω-suppressed at DC). Default 100 cm⁻¹.
    Returns:
        freq_cm   : [N] ndarray — frequencies in cm⁻¹
        intensity : [N] ndarray — IR absorption intensity (arb. units)
        power     : [N] ndarray — power spectrum (arb. units, M(ω) before ω weighting)
        acf       : [Nmax] ndarray — dipole autocorrelation function
    """
    # Subtract mean to remove DC component before computing ACF
    dipoles = dipoles - dipoles.mean(axis=0)
    acf_full = compute_dipole_acf(dipoles)

    # Truncate ACF — only the first acf_ratio fraction has good statistics.
    # Use multiplicative form so non-reciprocal ratios (e.g. 0.3) and small
    # ratios (e.g. 0.05) work correctly. Guard against acf_ratio<=0.
    if acf_ratio <= 0.0:
        raise ValueError(f"acf_ratio must be > 0, got {acf_ratio}")
    Nmax = max(1, int(len(acf_full) * acf_ratio))
    acf = acf_full[:Nmax]

    # Kronecker doubling: one-sided ACF -> two-sided via factor of 2 (except lag 0)
    kronecker = np.ones(Nmax) * 2.0
    kronecker[0] = 1.0

    # Apply window to reduce spectral leakage
    if window == 'hann':
        w = (np.cos(np.pi * np.arange(Nmax) / Nmax) + 1.0) * 0.5
    elif window == 'blackman':
        w = np.blackman(Nmax)
    else:
        w = np.ones(Nmax)

    acf_prepared = acf * w * kronecker

    # Cosine transform: M(k) = Σ_t acf(t) · cos(2πkt / (2Nmax-1)) is the real
    # part of the rfft of acf_prepared zero-padded to length 2Nmax-1.
    # O(N log N) in C vs O(N²) in Python.
    M_omega = np.fft.rfft(acf_prepared, n=2 * Nmax - 1).real

    # Frequency axis: convert from DCT index to cm⁻¹
    # k-th bin corresponds to frequency k / ((2*Nmax-1) * dt_fs) in 1/fs
    c_cm_per_fs = 2.99792458e-5  # speed of light in cm/fs
    freq_cm = np.arange(Nmax) / ((2 * Nmax - 1) * dt_fs * c_cm_per_fs)

    # IR absorption weighting. The "harmonic" form ω·(1-e^(-ℏω/kT))·M(ω)
    # is what GPUMD uses; the classical ω²·M(ω) form is only valid for
    # ν̃ ≪ kT/ℏ (~210 cm⁻¹ at 300 K) and dramatically overweights high-
    # frequency modes when applied beyond that regime.
    qc = str(quantum_correction).lower()
    hbar_c_eV_cm = 1.23984e-4              # ℏc in eV·cm  (so ℏω [eV] = ℏc · ν̃ [cm⁻¹])
    kT_eV = 8.617333e-5 * float(temperature)
    if qc in ("classical", "quadratic"):    # "quadratic" alias = ω²·M(ω)
        prefactor = freq_cm ** 2
    elif qc == "linear":
        prefactor = freq_cm
    elif qc == "none":
        prefactor = np.ones_like(freq_cm)
    elif qc == "harmonic":
        x = hbar_c_eV_cm * freq_cm / max(kT_eV, 1e-30)   # ℏω/kT  (dimensionless)
        # (1 - e^{-x}) for x→0 is x (gives classical ω² limit); for large
        # x saturates to 1 (gives ω scaling, the high-freq quantum limit).
        prefactor = freq_cm * (1.0 - np.exp(-x))
    else:
        raise ValueError(
            f"quantum_correction must be 'harmonic', 'classical' (alias 'quadratic'), "
            f"'linear', or 'none', got {quantum_correction!r}")
    intensity = prefactor * M_omega
    power = M_omega.copy()

    # Truncate to requested frequency range
    mask = freq_cm <= max_freq_cm
    freq_cm = freq_cm[mask]
    intensity = intensity[mask]
    power = power[mask]

    # Smooth. Higher `smooth_k` → smoother spectrum.
    #   smooth_kind="gaussian" (default): no spectral ringing. Treated as
    #     FWHM in bins; σ = smooth_k / 2.355. Preserves freq_cm length.
    #   smooth_kind="box": moving average of width smooth_k bins. Matches
    #     GPUMD-notebook behaviour but produces sinc-like sidelobes around
    #     sharp peaks. Uses mode='valid' which shortens freq_cm.
    if smooth_k > 1 and len(intensity) > smooth_k:
        sk = str(smooth_kind).lower()
        if sk == "gaussian":
            try:
                from scipy.ndimage import gaussian_filter1d
            except ImportError:                              # graceful fallback
                gaussian_filter1d = None
            if gaussian_filter1d is not None:
                sigma_bins = smooth_k / 2.355                # FWHM → σ
                intensity = gaussian_filter1d(intensity, sigma=sigma_bins,
                                              mode="nearest")
                power     = gaussian_filter1d(power,     sigma=sigma_bins,
                                              mode="nearest")
                # freq_cm unchanged — Gaussian filter preserves alignment.
            else:
                sk = "box"                                   # fall back below
        if sk == "box":
            kernel = np.ones(smooth_k) / smooth_k
            intensity = np.convolve(intensity, kernel, mode='valid')
            power     = np.convolve(power,     kernel, mode='valid')
            # mode='valid' output[i] averages input[i:i+smooth_k], centred at i+(smooth_k-1)/2.
            d_freq = freq_cm[1] - freq_cm[0]
            freq_start = freq_cm[0] + (smooth_k - 1) / 2.0 * d_freq
            freq_cm = freq_start + np.arange(len(intensity)) * d_freq
        elif sk not in ("gaussian", "box"):
            raise ValueError(
                f"smooth_kind must be 'gaussian' or 'box', got {smooth_kind!r}")

    # Normalise both to peak = 1.
    peak = np.max(np.abs(intensity))
    if peak > 0:
        intensity /= peak

    # For the raw power spectrum M(ω), the global max sits at ω ≈ 0
    # (since the ACF C(τ) is largest at τ=0 and the cosine transform of
    # a one-sided decaying function peaks at ω=0). Dividing by that DC
    # peak collapses every vibrational feature to <1 % of full scale —
    # the spectrum looks "empty" on a linear plot. Exclude bins below
    # `power_dc_cutoff_cm` from the normaliser so vibrational peaks are
    # visible. Set to 0 to disable (keeps the original DC-dominated
    # behaviour for users who want raw M(ω) magnitudes).
    if power_dc_cutoff_cm > 0:
        mask_vib = freq_cm >= float(power_dc_cutoff_cm)
    else:
        mask_vib = np.ones_like(freq_cm, dtype=bool)
    peak_vib = float(np.max(np.abs(power[mask_vib]))) if mask_vib.any() else 0.0
    if peak_vib > 0:
        power = power / peak_vib

    return freq_cm, intensity, power, acf


def _ir_plot_basename(trajectory_path: str | None,
                       model_label: str | None) -> str:
    """Build a descriptive plot stem from trajectory + model names.

    e.g. trajectory_path = "datasets/ethanol_nve.traj"
         model_label    = "n50_q165_pop100_CHO"
         → "ethanol_nve_n50_q165_pop100_CHO"

    Any None or empty parts are skipped. Falls back to "ir_spectrum" if
    both are None.
    """
    parts = []
    if trajectory_path:
        parts.append(os.path.splitext(os.path.basename(trajectory_path))[0])
    if model_label:
        parts.append(str(model_label))
    return "_".join(parts) if parts else "ir_spectrum"


def _plot_one_ir_panel(ax_lo, ax_hi, freq_cm: np.ndarray, y: np.ndarray,
                        split_at_cm: float | None,
                        ylabel: str, title: str,
                        invert_y: bool = False) -> None:
    """Plot one IR trace into either a single axis (`ax_hi`, `ax_lo=None`)
    or a broken-axis pair (low/high regions normalised independently).

    When `split_at_cm` is None: plot full range into `ax_hi` only.
    When `split_at_cm` is a float: plot ν̃ ≤ split into `ax_lo` and
    ν̃ > split into `ax_hi`. Each side is renormalised to peak = 1 within
    its window so that high-freq structure isn't crushed by low-freq peaks.
    Diagonal break marks are drawn between the two halves.
    """
    if split_at_cm is None:
        ax_hi.plot(freq_cm, y, color='black', linewidth=0.8)
        ax_hi.set_xlabel("Wavenumber (cm⁻¹)")
        ax_hi.set_ylabel(ylabel)
        ax_hi.set_title(title)
        ax_hi.set_xlim(freq_cm[0], freq_cm[-1])
        ax_hi.invert_xaxis()
        ax_hi.set_ylim(0, 1.05)
        ax_hi.grid(alpha=0.3)
        return

    # Split-region rendering. Two adjacent axes (ax_lo for the lower
    # wavenumber band, ax_hi for the higher) each with independent
    # peak-normalisation; visual "broken-axis" cue between them.
    mask_lo = freq_cm <= split_at_cm
    mask_hi = freq_cm >  split_at_cm
    if not mask_lo.any() or not mask_hi.any():
        # Fall back to single-axis if the split lands outside the data
        _plot_one_ir_panel(None, ax_hi, freq_cm, y, None, ylabel, title)
        return

    f_lo, y_lo = freq_cm[mask_lo], y[mask_lo]
    f_hi, y_hi = freq_cm[mask_hi], y[mask_hi]

    if invert_y:
        # `y` is transmittance built from the GLOBAL-peak absorbance. To
        # renormalise per region, recover the absorbance, peak-normalise
        # it within each region, then re-apply the chosen T(A) mapping.
        a_lo = _transmittance_to_absorbance(y_lo)
        a_hi = _transmittance_to_absorbance(y_hi)
        peak_lo = float(np.max(a_lo)) if a_lo.size else 0.0
        peak_hi = float(np.max(a_hi)) if a_hi.size else 0.0
        if peak_lo > 0:
            a_lo = a_lo / peak_lo
        if peak_hi > 0:
            a_hi = a_hi / peak_hi
        y_lo = _absorbance_to_transmittance(a_lo)
        y_hi = _absorbance_to_transmittance(a_hi)
    else:
        peak_lo = float(np.max(y_lo)) if y_lo.size else 0.0
        peak_hi = float(np.max(y_hi)) if y_hi.size else 0.0
        if peak_lo > 0:
            y_lo = y_lo / peak_lo
        if peak_hi > 0:
            y_hi = y_hi / peak_hi

    # IR convention: high wavenumber on the left. So ax_hi is on the
    # LEFT of the pair, ax_lo on the RIGHT.
    ax_hi.plot(f_hi, y_hi, color='black', linewidth=0.8)
    ax_lo.plot(f_lo, y_lo, color='black', linewidth=0.8)
    ax_hi.set_xlim(f_hi.max(), f_hi.min())     # invert
    ax_lo.set_xlim(f_lo.max(), f_lo.min())
    ax_hi.set_ylim(0, 1.05); ax_lo.set_ylim(0, 1.05)
    ax_hi.set_title(title); ax_hi.set_ylabel(ylabel)
    ax_hi.grid(alpha=0.3); ax_lo.grid(alpha=0.3)
    # Hide the inner spines so the panels look joined.
    ax_hi.spines['right'].set_visible(False)
    ax_lo.spines['left'].set_visible(False)
    ax_lo.tick_params(left=False, labelleft=False)
    # Diagonal break marks (matplotlib broken-axis idiom).
    d = .015
    kwargs = dict(transform=ax_hi.transAxes, color='k', clip_on=False)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax_hi.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_lo.transAxes)
    ax_lo.plot((-d, +d), (-d, +d), **kwargs)
    ax_lo.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    # Shared x-label spans both, centred on the pair.
    ax_hi.set_xlabel("")
    ax_lo.set_xlabel("")
    # Use the figure-level annotation for the combined x-label below.


def _emit_single_ir_figure(freq_cm: np.ndarray, y: np.ndarray,
                            label: str, ylabel: str,
                            title: str, split_at_cm: float | None,
                            stem: str, cfg: TNEPconfig,
                            save_plots: str | None, show_plots: bool,
                            invert_y: bool) -> None:
    """Emit ONE labelled figure (absorbance OR transmittance), saved
    separately so the two are not crammed into one image.
    """
    from plotting import _save_fig
    if split_at_cm is None:
        fig, ax = plt.subplots(figsize=(14, 6))
        _plot_one_ir_panel(None, ax, freq_cm, y, None,
                           ylabel=ylabel, title=title)
    else:
        # Broken-axis pair: [hi | lo] for a single quantity.
        # Width ratios are set in PROPORTION to the wavenumber range each
        # side covers, so the cm⁻¹-per-pixel scale is identical on both
        # halves (i.e. a 50 cm⁻¹ feature has the same on-screen width
        # regardless of which side of the break it sits on).
        f_max = float(freq_cm.max())
        f_min = float(freq_cm.min())
        f_split = float(split_at_cm)
        # Clamp so we never get zero/negative widths from a split outside
        # the data range; fall back to 1:1 in that pathological case.
        hi_extent = max(f_max - f_split, 1.0)
        lo_extent = max(f_split - f_min, 1.0)
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 6),
            gridspec_kw={"width_ratios": [hi_extent, lo_extent],
                         "wspace": 0.06})
        ax_hi, ax_lo = axes
        _plot_one_ir_panel(ax_lo, ax_hi, freq_cm, y, f_split,
                            ylabel=ylabel,
                            title=f"{title}  (split at {int(split_at_cm)} cm⁻¹)",
                            invert_y=invert_y)
    fig.suptitle(label, fontsize=14)
    plt.tight_layout(rect=(0, 0.06 if split_at_cm is not None else 0,
                            1, 0.95))
    if split_at_cm is not None:
        from matplotlib.transforms import Bbox
        bbox = Bbox.union([axes[0].get_position(), axes[1].get_position()])
        fig.text(0.5 * (bbox.x0 + bbox.x1), 0.02,
                 "Wavenumber (cm⁻¹)", ha='center', va='bottom', fontsize=11)
    _save_fig(fig, cfg, stem, save_plots)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


# Module-level state set by `plot_ir_spectrum` and consumed by
# `_plot_one_ir_panel` for per-region transmittance reconstruction.
# Keeps the panel-helper signature stable while letting the outer
# plotter dictate the T(A) convention.
_TRANSMITTANCE_MODE: str = "beer_lambert"
_TRANSMITTANCE_SCALE: float = 1.0


def _absorbance_to_transmittance(A: np.ndarray) -> np.ndarray:
    """A → T using the currently-configured convention.

    "beer_lambert" : T = 10^(−scale·A)   — proper Beer-Lambert form;
                     T=10^(−1) ≈ 0.1 at A=1 with default scale=1.
    "linear"       : T = 1 − A           — visual mirror of absorbance,
                     used by most computational-IR pipelines for display.
    """
    A = np.asarray(A)
    if _TRANSMITTANCE_MODE == "beer_lambert":
        return 10.0 ** (-_TRANSMITTANCE_SCALE * A)
    if _TRANSMITTANCE_MODE == "linear":
        return 1.0 - A
    raise ValueError(f"Unknown transmittance_mode {_TRANSMITTANCE_MODE!r}")


def _transmittance_to_absorbance(T: np.ndarray) -> np.ndarray:
    """T → A, inverse of `_absorbance_to_transmittance` under the
    currently-configured convention. Clipped to keep log/inverse stable
    against FFT round-off producing T slightly above 1 or below 0."""
    T = np.clip(np.asarray(T), 1e-30, 1.0)
    if _TRANSMITTANCE_MODE == "beer_lambert":
        return -np.log10(T) / max(_TRANSMITTANCE_SCALE, 1e-30)
    if _TRANSMITTANCE_MODE == "linear":
        return 1.0 - T
    raise ValueError(f"Unknown transmittance_mode {_TRANSMITTANCE_MODE!r}")


def plot_ir_spectrum(freq_cm: np.ndarray, intensity: np.ndarray, cfg: TNEPconfig,
                     save_plots: str | None = None, show_plots: bool = True,
                     title: str = "IR Spectrum",
                     trajectory_path: str | None = None,
                     model_label: str | None = None,
                     split_at_cm: float | None = None,
                     transmittance_mode: str = "beer_lambert",
                     transmittance_scale: float = 1.0) -> None:
    """Plot IR spectrum as TWO separate, labelled figures.

    Emits the absorbance and transmittance plots as **independent files**
    so neither is cramped. Filename suffixes distinguish them:
        <stem>_absorbance.png
        <stem>_transmittance.png
    (or `_absorbance_split500.png` etc. when `split_at_cm` is set).

    Y-axes both ascend normally:
        - Absorbance     : peaks point UP from a flat 0 baseline.
        - Transmittance  : peaks dip DOWN from a flat 1 baseline.

    Args:
        freq_cm             : [N] ndarray — frequencies in cm⁻¹
        intensity           : [N] ndarray — normalised IR intensity (peak=1)
        cfg                 : TNEPconfig — used for save-directory resolution
        save_plots          : str or None — directory to save into
        show_plots          : bool — True to display interactively
        title               : str — figure title prefix
        trajectory_path     : str or None — appears in filename
        model_label         : str or None — appears in filename
        split_at_cm         : float or None — broken-axis split (per-region
                              peak-normalisation). Default None = full range.
        transmittance_mode  : str — A → T conversion:
                              "beer_lambert" (default): T = 10^(−scale·A).
                                  Physically meaningful Beer-Lambert form.
                                  Peak A=1 maps to T=0.1 with default scale.
                              "linear" : T = 1 − A. Visual mirror only;
                                  not Beer-Lambert correct but used in
                                  many computational-IR pipelines.
        transmittance_scale : float — multiplier on A for Beer-Lambert
                              mode (effective path-length·concentration
                              product). Higher → deeper transmittance
                              dips. Default 1.0 gives ~10 % minimum
                              transmission at the strongest peak.
    """
    # Stash the chosen mode on module state so the per-region renormaliser
    # in `_plot_one_ir_panel` (used by the split-axis layout) can apply
    # the same A↔T mapping when recomputing transmittance per region.
    global _TRANSMITTANCE_MODE, _TRANSMITTANCE_SCALE
    _TRANSMITTANCE_MODE = str(transmittance_mode).lower()
    _TRANSMITTANCE_SCALE = float(transmittance_scale)
    if _TRANSMITTANCE_MODE not in ("beer_lambert", "linear"):
        raise ValueError(
            f"transmittance_mode must be 'beer_lambert' or 'linear', "
            f"got {transmittance_mode!r}")

    absorbance    = intensity
    transmittance = _absorbance_to_transmittance(absorbance)

    base_stem = _ir_plot_basename(trajectory_path, model_label)
    split_tag = f"_split{int(split_at_cm)}" if split_at_cm is not None else ""

    # 1. Absorbance
    _emit_single_ir_figure(
        freq_cm, absorbance,
        label=f"{title} — Absorbance",
        ylabel=("Absorbance (per-region peak-normalised)"
                if split_at_cm is not None
                else "Absorbance (arb. units, peak-normalised)"),
        title="Absorbance",
        split_at_cm=split_at_cm,
        stem=f"{base_stem}_absorbance{split_tag}",
        cfg=cfg, save_plots=save_plots, show_plots=show_plots,
        invert_y=False,
    )

    # 2. Transmittance — y-axis label reflects the chosen mode.
    if _TRANSMITTANCE_MODE == "beer_lambert":
        t_label = (f"Transmittance  T = 10^(−{_TRANSMITTANCE_SCALE:g}·A)"
                   if not split_at_cm
                   else f"Transmittance  (per-region 10^(−{_TRANSMITTANCE_SCALE:g}·A))")
    else:  # "linear"
        t_label = ("Transmittance  (1 − A)"
                   if not split_at_cm
                   else "Transmittance  (per-region 1 − A)")
    _emit_single_ir_figure(
        freq_cm, transmittance,
        label=f"{title} — Transmittance",
        ylabel=t_label,
        title="Transmittance",
        split_at_cm=split_at_cm,
        stem=f"{base_stem}_transmittance{split_tag}",
        cfg=cfg, save_plots=save_plots, show_plots=show_plots,
        invert_y=True,
    )


def ir_spectrum_from_file(
    dipole_path: str,
    dt_fs: float = 1.0,
    save_dir: str | None = None,
    show: bool = True,
    title: str | None = None,
    model_label: str | None = None,
    plot_power: bool = True,
    split_at_cm: float | None = None,
    cfg: TNEPconfig | None = None,
    # ── compute_ir_spectrum knobs promoted to first-class kwargs ────
    window: str | None = 'hann',
    max_freq_cm: float = 4000.0,
    acf_ratio: float = 0.1,
    smooth_k: int = 10,
    smooth_kind: str = "gaussian",
    temperature: float = 300.0,
    quantum_correction: str = "harmonic",
    power_dc_cutoff_cm: float = 100.0,
    # ── transmittance display options ───────────────────────────────
    transmittance_mode: str = "beer_lambert",
    transmittance_scale: float = 1.0,
    **ir_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved dipole trajectory and plot its IR + power spectra.

    Convenience wrapper around `np.load`/`np.loadtxt` + `compute_ir_spectrum`.
    Accepts either the binary `.npy` or the text `.txt` file written by
    `process_trajectory`. Emits the absorbance + transmittance plots as
    separate, labelled files, plus the companion power-spectrum plot.

    Args:
        dipole_path        : path to the dipole file (.npy or .txt). Shape [T, 3].
        dt_fs              : timestep between frames in femtoseconds.
        save_dir           : directory to save plots into. None = don't save.
        show               : True to display interactively.
        title              : optional plot title (defaults to the file basename).
        model_label        : optional model identifier baked into the saved
                             filename (e.g. "n50_q165_CHO"). None = none.
        plot_power         : also produce the companion power-spectrum plot.
        split_at_cm        : if set, the absorbance + transmittance plots
                             are rendered as broken-axis pairs split at
                             this wavenumber (e.g. 500.0). Each side is
                             independently peak-normalised. None = full
                             range.
        cfg                : optional TNEPconfig used by the plotters for unit
                             labels. A minimal default is created if None.

        window             : ACF window — 'hann' (default), 'blackman', or None.
        max_freq_cm        : maximum wavenumber kept on the spectrum (cm⁻¹).
        acf_ratio          : fraction of the trajectory used as max ACF lag.
        smooth_k           : smoothing strength. Higher = smoother spectrum.
                             For Gaussian smoothing this is the FWHM in
                             frequency bins; for box it is the moving-average
                             window. 0 disables.
        smooth_kind        : "gaussian" (default; no ringing) or "box".
        temperature        : simulation T in K. Used by the harmonic quantum
                             correction.
        quantum_correction : IR weighting:
                              "harmonic"  : ω·(1 − e^(−ℏω/kT))·M(ω) (GPUMD)
                              "classical" : ω²·M(ω) (Xu Eq. 1 original)
                              "quadratic" : alias for "classical"
                              "linear"    : ω·M(ω)
                              "none"      : M(ω)
        power_dc_cutoff_cm : exclude bins below this from the power-spectrum
                             peak-normaliser (and from the visible plot
                             range). Set 0 to disable.
        **ir_kwargs        : any additional kwargs forwarded to
                             `compute_ir_spectrum` (future-proofing).

    Returns:
        (freq_cm, intensity, power) — 1-D arrays.
    """
    if dipole_path.lower().endswith(".npy"):
        dipoles = np.load(dipole_path)
    else:
        dipoles = np.loadtxt(dipole_path)
    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"expected dipoles of shape [T, 3], got {dipoles.shape}")

    freq_cm, intensity, power, _acf = compute_ir_spectrum(
        dipoles, dt_fs=dt_fs,
        window=window, max_freq_cm=max_freq_cm, acf_ratio=acf_ratio,
        smooth_k=smooth_k, smooth_kind=smooth_kind,
        temperature=temperature, quantum_correction=quantum_correction,
        power_dc_cutoff_cm=power_dc_cutoff_cm,
        **ir_kwargs,
    )

    # The plotters consult cfg only for unit labels; a fresh TNEPconfig
    # with the default e·Å unit is fine when one isn't supplied.
    if cfg is None:
        cfg = TNEPconfig()

    plot_ir_spectrum(
        freq_cm, intensity, cfg,
        save_plots=save_dir, show_plots=show,
        title=title or f"IR spectrum — {os.path.basename(dipole_path)}",
        trajectory_path=dipole_path, model_label=model_label,
        split_at_cm=split_at_cm,
        transmittance_mode=transmittance_mode,
        transmittance_scale=transmittance_scale,
    )
    if plot_power:
        # Same DC cutoff for the visible plot range as the normaliser,
        # so the two stay consistent.
        plot_power_spectrum(
            freq_cm, power, cfg,
            save_plots=save_dir, show_plots=show,
            title="Power Spectrum  M(ω)",
            trajectory_path=dipole_path, model_label=model_label,
            low_cm_cutoff=power_dc_cutoff_cm,
        )
    return freq_cm, intensity, power


def plot_power_spectrum(freq_cm: np.ndarray, power: np.ndarray, cfg: TNEPconfig,
                        save_plots: str | None = "plots", show_plots: bool = False,
                        title: str = "Power Spectrum",
                        low_cm_cutoff: float = 100.0,
                        use_log: bool = True,
                        trajectory_path: str | None = None,
                        model_label: str | None = None) -> None:
    """Plot the dipole power spectrum M(ω) (no ω weighting).

    M(ω) has a strong DC peak (ω≈0) that dwarfs all vibrational features;
    this plot crops the visible range to ν̃ ≥ `low_cm_cutoff` cm⁻¹ (default
    100) and defaults to a log y-axis so the 2-4 orders of magnitude of
    dynamic range across vibrational modes are visible.

    Args:
        freq_cm         : [N] ndarray — frequencies in cm⁻¹
        power           : [N] ndarray — normalised power spectrum M(ω)
        cfg             : TNEPconfig — used for save-directory resolution
        save_plots      : str or None — directory to save into (None = don't save)
        show_plots      : bool — True to display interactively
        title           : str — plot title
        low_cm_cutoff   : float — drop frequencies below this from the plot
                          (default 100 cm⁻¹). 0 = keep full range.
        use_log         : bool — log y-axis (default True)
        trajectory_path : str or None — used in the saved filename
        model_label     : str or None — used in the saved filename
    """
    from plotting import _save_fig

    mask = freq_cm >= float(low_cm_cutoff)
    f_plot = freq_cm[mask]
    p_plot = power[mask]
    # Guard against zeros / negatives ruining log scale.
    if use_log:
        p_plot = np.maximum(p_plot, 1e-8)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(f_plot, p_plot, color='black', linewidth=0.8)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Power M(ω) (peak-normalised over ν̃ ≥ "
                  f"{int(low_cm_cutoff)} cm⁻¹)")
    ax.set_title(title)
    ax.set_xlim(f_plot[0], f_plot[-1])
    ax.invert_xaxis()
    if use_log:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    stem = _ir_plot_basename(trajectory_path, model_label) + "_power"
    _save_fig(fig, cfg, stem, save_plots)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------
# Phase 3: fused pack + predict @tf.function. Cached per model instance so
# the graph is traced exactly once. Input signature uses [None] dims so a
# changing batch size / atom count / pair count does not retrigger tracing.
# --------------------------------------------------------------------------
_FUSED_PREDICT_CACHE: dict = {}


def _get_fused_predict(model: 'TNEP'):
    """Return (and cache) a per-model fused pack+predict @tf.function.

    The graph captures the model weights via closure, so SNES candidate
    evaluation (which swaps weights) must not use this path — it's
    trajectory-inference-only. Per-model caching keyed by id(model) is
    sufficient for that contract.
    """
    key = id(model)
    cached = _FUSED_PREDICT_CACHE.get(key)
    if cached is not None:
        return cached

    cfg = model.cfg
    dim_q = cfg.dim_q
    sig = [
        tf.TensorSpec(shape=[None, dim_q],     dtype=tf.float32),  # soap_concat
        tf.TensorSpec(shape=[None, 3, dim_q],  dtype=tf.float32),  # grad_concat
        tf.TensorSpec(shape=[None],            dtype=tf.int32),    # pa_concat
        tf.TensorSpec(shape=[None],            dtype=tf.int32),    # pg_concat
        tf.TensorSpec(shape=[None],            dtype=tf.int64),    # atom_counts (Ragged)
        tf.TensorSpec(shape=[None],            dtype=tf.int32),    # pair_counts
        tf.TensorSpec(shape=[None, None, 3],   dtype=tf.float32),  # positions
        tf.TensorSpec(shape=[None, None],      dtype=tf.int32),    # Z
        tf.TensorSpec(shape=[None, 3, 3],      dtype=tf.float32),  # boxes
        tf.TensorSpec(shape=[None, None],      dtype=tf.float32),  # atom_mask
        tf.TensorSpec(shape=[None],            dtype=tf.int32),    # num_atoms
    ]

    # Keras Variables aren't accepted as direct @tf.function args via the
    # TraceTypeBuilder, so we materialise them into tf.constant tensors at
    # trace time and capture via closure. Trajectory inference is fixed-
    # weights, so this is safe; SNES population evaluation must not use
    # this path (it needs per-call weight swapping).
    def _to_tensor(v):
        if v is None:
            return None
        # Keras 3 Variables expose .value (a Tensor); fall back to convert.
        return tf.convert_to_tensor(v.value if hasattr(v, "value") else v)

    # Pre-absorb U_pairᵀ into W0 (and W0_pol) when descriptor mixing is
    # active. `predict_batch` is a pure forward primitive that trusts
    # the caller to do this; the SNES paths handle it explicitly, but
    # the trajectory-inference fused graph previously passed raw W0,
    # silently dropping the learned mixing. Mirrors the score() pattern
    # in TNEP.py.
    if getattr(model, "descriptor_mixing", False) and model.U_pair is not None:
        W0_eff_var = model._W0_eff(model.W0)
        W0p_eff_var = (model._W0_eff(model.W0_pol)
                        if (cfg.target_mode == 2
                            and getattr(model, "W0_pol", None) is not None)
                        else getattr(model, "W0_pol", None))
    else:
        W0_eff_var = model.W0
        W0p_eff_var = getattr(model, "W0_pol", None)
    W0_t  = _to_tensor(W0_eff_var)
    b0_t  = _to_tensor(model.b0)
    W1_t  = _to_tensor(model.W1)
    b1_t  = _to_tensor(model.b1)
    W0p_t = _to_tensor(W0p_eff_var)
    b0p_t = _to_tensor(getattr(model, "b0_pol", None))
    W1p_t = _to_tensor(getattr(model, "W1_pol", None))
    b1p_t = _to_tensor(getattr(model, "b1_pol", None))

    # Per-channel descriptor scaler (cfg._q_scaler). Captured into the
    # fused graph as a constant tensor; multiplies BOTH soap_concat and
    # grad_concat at the top of fused() so descriptors and gradients
    # see the same training-time normalisation. None when scaling is
    # off (descriptor_scaling="none").
    q_scaler_const = (tf.constant(cfg._q_scaler, dtype=tf.float32)
                      if (str(getattr(cfg, "descriptor_scaling", "none")) != "none"
                          and getattr(cfg, "_q_scaler", None) is not None)
                      else None)

    # NOTE: not jit_compile=True. The descriptor-side XLA path (locked
    # compute fns built with jit_compile=True for trajectory) handles the
    # heavy SOAP work. predict_batch internally has shape-dependent stacks
    # in _calc_forces_coo that XLA can't lower (varying P per call), so
    # we keep this graph as a regular @tf.function trace. Per-op launch
    # overhead at this stage is dwarfed by the fused descriptor compute.
    @tf.function(input_signature=sig, reduce_retracing=False)
    def fused(soap_concat, grad_concat, pa_concat, pg_concat,
              atom_counts, pair_counts,
              positions, Z, boxes, atom_mask, num_atoms):
        # Apply per-channel descriptor scaling (cfg._q_scaler) at the
        # same point in the chain as training (pad_and_stack does it
        # before packing). Both `soap_concat` [N, Q] and `grad_concat`
        # [P, 3, Q] get the same `s[Q]` along their last axis — see
        # data._apply_q_scaler_np for the equivalent training-time op.
        if q_scaler_const is not None:
            soap_concat = soap_concat * q_scaler_const[tf.newaxis, :]
            grad_concat = grad_concat * q_scaler_const[tf.newaxis, tf.newaxis, :]
        S = tf.shape(num_atoms)[0]
        # Pad descriptors via scatter_nd. RaggedTensor.to_tensor is the
        # natural choice but the underlying RaggedTensorToTensor op has no
        # XLA kernel, so we build the dense [S, A_max, Q] layout directly
        # from per-row struct/intra indices.
        A_max = tf.shape(atom_mask)[1]
        struct_idx_long = tf.repeat(tf.range(S, dtype=tf.int64), atom_counts)
        cum = tf.concat([[tf.constant(0, dtype=tf.int64)],
                         tf.cumsum(atom_counts)[:-1]], axis=0)
        intra_idx = (tf.range(tf.shape(soap_concat)[0], dtype=tf.int64)
                     - tf.gather(cum, struct_idx_long))
        scatter_idx = tf.stack(
            [tf.cast(struct_idx_long, tf.int32),
             tf.cast(intra_idx, tf.int32)],
            axis=-1)
        descriptors = tf.scatter_nd(
            scatter_idx, soap_concat,
            shape=tf.stack([S, A_max, tf.constant(dim_q, tf.int32)]))
        pair_struct = tf.repeat(tf.range(S, dtype=tf.int32), pair_counts)
        preds = model.predict_batch(
            descriptors, grad_concat, pa_concat, pg_concat, pair_struct,
            positions, Z, boxes, atom_mask,
            W0_t, b0_t, W1_t, b1_t,
            W0p_t, b0p_t, W1p_t, b1p_t,
        )
        # NOTE: predict_batch returns the TOTAL dipole, not per-atom — even
        # when cfg.scale_targets is True. Training stores per-atom *targets*,
        # and TNEP.score() divides raw_preds by N to compare on the same
        # scale (TNEP.py:285-288). So preds here is already the total system
        # dipole; do NOT multiply by num_atoms.
        return preds

    _FUSED_PREDICT_CACHE[key] = fused
    return fused


def _build_fused_inputs(
    frame_results: list,
    frames: list,
    types_int_batch: list[np.ndarray],
    dim_q: int,
    pin_to_cpu: bool = True,
) -> tuple:
    """Concatenate per-frame TF tensors and build host-side padded fields.

    The concats are eager TF ops over a Python list of GPU tensors (one
    kernel each), and the host-built fields (positions, Z, boxes,
    atom_mask, num_atoms) get a single CPU→device push per outer batch.
    Returned tuple matches the input_signature of `_get_fused_predict`.

    Destructive: clears `frame_results` after each field's concat is built,
    so the per-frame device tensors are released as soon as their data is
    folded into the chunk-level concats. This halves the peak VRAM during
    the pack step (peak ~= one concatenation, not concat + originals).
    """
    S = len(frames)
    atom_counts = [int(r[0].shape[0]) for r in frame_results]
    max_atoms = max(atom_counts) if atom_counts else 0
    pair_counts = [int(r[1].shape[0]) for r in frame_results]
    pair_counts_arr = np.array(pair_counts, dtype=np.int32)
    N_pairs = int(pair_counts_arr.sum())

    if S:
        soap_concat = tf.concat([r[0] for r in frame_results], axis=0)
    else:
        soap_concat = tf.zeros((0, dim_q), dtype=tf.float32)
    if N_pairs > 0:
        grad_concat = tf.concat([r[1] for r in frame_results], axis=0)
        pa_concat   = tf.concat([r[2] for r in frame_results], axis=0)
        pg_concat   = tf.concat([r[3] for r in frame_results], axis=0)
    else:
        grad_concat = tf.zeros((0, 3, dim_q), dtype=tf.float32)
        pa_concat   = tf.zeros((0,), dtype=tf.int32)
        pg_concat   = tf.zeros((0,), dtype=tf.int32)
    # All per-frame data is now in the concat tensors; release the
    # caller's list so the per-frame slices get garbage-collected.
    frame_results.clear()

    atom_counts_t = tf.constant(atom_counts, dtype=tf.int64)
    pair_counts_t = tf.constant(pair_counts_arr, dtype=tf.int32)

    pos_np       = np.zeros((S, max_atoms, 3), dtype=np.float32)
    z_np         = np.zeros((S, max_atoms),    dtype=np.int32)
    box_np       = np.zeros((S, 3, 3),         dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms),    dtype=np.float32)
    num_atoms_np = np.array(atom_counts,       dtype=np.int32)
    for s in range(S):
        N_s = atom_counts[s]
        pos_np[s, :N_s]       = frames[s].positions.astype(np.float32)
        z_np[s, :N_s]         = types_int_batch[s]
        box_np[s]             = cell_to_box(frames[s])
        atom_mask_np[s, :N_s] = 1.0

    with tf.device('/CPU:0' if pin_to_cpu else '/GPU:0'):
        positions  = tf.constant(pos_np);       del pos_np
        Z          = tf.constant(z_np);         del z_np
        boxes      = tf.constant(box_np);       del box_np
        atom_mask  = tf.constant(atom_mask_np); del atom_mask_np
        num_atoms  = tf.constant(num_atoms_np); del num_atoms_np

    return (soap_concat, grad_concat, pa_concat, pg_concat,
            atom_counts_t, pair_counts_t,
            positions, Z, boxes, atom_mask, num_atoms)


def _pack_traj_batch_from_tf(
    frame_results: list,
    frames: list,
    types_int_batch: list[np.ndarray],
    dim_q: int,
    pin_to_cpu: bool = True,
) -> dict:
    """TF-tensor variant of pack: avoids the NumPy round-trip on descriptors/grads.

    Each frame_results[s] = (soap_t [N,Q], grad_t [P,3,Q], pa_t [P], pg_t [P]),
    all TF tensors living on the descriptor-builder's compute device. This pack
    concatenates them on-device with global atom-index offsets and returns the
    same dict shape as `_pack_traj_batch_from_flat`. Auxiliary structure-padded
    fields (positions, Z, boxes, atom_mask) still come from the ASE objects on
    the host — they're built once per outer batch and respect pin_to_cpu.

    Returns:
        dict with descriptors [B,A,Q], grad_values [P,3,Q], pair_atom/gidx/struct [P],
        positions [B,A,3], Z_int [B,A], boxes [B,3,3], atom_mask [B,A], num_atoms [B].
    """
    S = len(frames)
    # Cache atom_counts before pack_chunk_from_flat clears frame_results.
    atom_counts = [int(r[0].shape[0]) for r in frame_results]
    max_atoms = max(atom_counts) if atom_counts else 0

    # Descriptor-shaped fields (descriptors, grad_values, pair_*) — shared
    # with the OTF training path via pack_chunk_from_flat.
    chunk = pack_chunk_from_flat(frame_results, dim_q)

    # Structure-padded host-built fields (positions, Z, boxes, atom_mask).
    pos_np = np.zeros((S, max_atoms, 3), dtype=np.float32)
    z_np = np.zeros((S, max_atoms), dtype=np.int32)
    box_np = np.zeros((S, 3, 3), dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms), dtype=np.float32)
    num_atoms_np = np.array(atom_counts, dtype=np.int32)
    for s in range(S):
        N_s = atom_counts[s]
        pos_np[s, :N_s] = frames[s].positions.astype(np.float32)
        z_np[s, :N_s] = types_int_batch[s]
        box_np[s] = cell_to_box(frames[s])
        atom_mask_np[s, :N_s] = 1.0

    with tf.device('/CPU:0' if pin_to_cpu else '/GPU:0'):
        positions_t = tf.constant(pos_np);     del pos_np
        z_t         = tf.constant(z_np);       del z_np
        box_t       = tf.constant(box_np);     del box_np
        atom_mask_t = tf.constant(atom_mask_np); del atom_mask_np
        num_atoms_t = tf.constant(num_atoms_np); del num_atoms_np

    chunk["positions"] = positions_t
    chunk["Z_int"]     = z_t
    chunk["boxes"]     = box_t
    chunk["atom_mask"] = atom_mask_t
    chunk["num_atoms"] = num_atoms_t
    return chunk


def _pack_traj_batch_from_flat(
    frame_results: list,
    frames: list,
    types_int_batch: list[np.ndarray],
    dim_q: int,
    pin_to_cpu: bool = True,
) -> dict:
    """Pack flat per-frame COO arrays into a stacked batch for predict_batch.

    Each frame_results[s] = (descriptors[N,Q], grad_values[P_s,3,Q],
    pair_atom[P_s], pair_gidx[P_s]), all numpy. We concatenate the pair-level
    arrays in one shot and build a struct-index from the per-frame counts —
    no per-atom Python loop, no .numpy() round-trips.

    Returns:
        dict with descriptors [B,A,Q], grad_values [P,3,Q], pair_atom/gidx/struct [P],
        positions [B,A,3], Z_int [B,A], boxes [B,3,3], atom_mask [B,A], num_atoms [B]
        — same layout as data.pad_and_stack() so predict_batch consumes it unchanged.
    """
    S = len(frames)
    atom_counts = [r[0].shape[0] for r in frame_results]
    max_atoms = max(atom_counts)
    pair_counts = np.array([r[1].shape[0] for r in frame_results], dtype=np.int32)
    N_pairs = int(pair_counts.sum())

    # Pair-level COO: single concat per field
    grad_values_np = (np.concatenate([r[1] for r in frame_results], axis=0)
                      if N_pairs else np.zeros((0, 3, dim_q), dtype=np.float32))
    pair_atom_np   = (np.concatenate([r[2] for r in frame_results])
                      if N_pairs else np.zeros(0, dtype=np.int32))
    pair_gidx_np   = (np.concatenate([r[3] for r in frame_results])
                      if N_pairs else np.zeros(0, dtype=np.int32))
    pair_struct_np = np.repeat(np.arange(S, dtype=np.int32), pair_counts)

    # Structure-padded fields
    desc_np      = np.zeros((S, max_atoms, dim_q), dtype=np.float32)
    pos_np       = np.zeros((S, max_atoms, 3),     dtype=np.float32)
    z_np         = np.zeros((S, max_atoms),         dtype=np.int32)
    box_np       = np.zeros((S, 3, 3),              dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms),         dtype=np.float32)
    num_atoms_np = np.array(atom_counts, dtype=np.int32)

    for s in range(S):
        N_s = atom_counts[s]
        desc_np[s, :N_s]      = frame_results[s][0]
        pos_np[s, :N_s]       = frames[s].positions.astype(np.float32)
        z_np[s, :N_s]         = types_int_batch[s]
        box_np[s]             = cell_to_box(frames[s])
        atom_mask_np[s, :N_s] = 1.0

    with tf.device('/CPU:0' if pin_to_cpu else '/GPU:0'):
        result = {}
        result["descriptors"] = tf.constant(desc_np);        del desc_np
        result["grad_values"] = tf.constant(grad_values_np); del grad_values_np
        result["pair_atom"]   = tf.constant(pair_atom_np);   del pair_atom_np
        result["pair_gidx"]   = tf.constant(pair_gidx_np);   del pair_gidx_np
        result["pair_struct"] = tf.constant(pair_struct_np); del pair_struct_np
        result["positions"]   = tf.constant(pos_np);         del pos_np
        result["Z_int"]       = tf.constant(z_np);           del z_np
        result["boxes"]       = tf.constant(box_np);         del box_np
        result["atom_mask"]   = tf.constant(atom_mask_np);   del atom_mask_np
        result["num_atoms"]   = tf.constant(num_atoms_np);   del num_atoms_np
    return result


def predict_trajectory_batch(
    model: TNEP,
    builder: DescriptorBuilder,
    batch_frames: list[Atoms],
    batch_types: list[np.ndarray],
    pin_to_cpu: bool = True,
    descriptor_batch_frames: int | None = 1,
    descriptor_memory_budget_bytes: int | None = None,
    descriptor_precision: str | None = None,
    descriptor_pair_tile_size: int | None = None,
) -> np.ndarray:
    """Run dipole/polarizability prediction on one batch of trajectory frames.

    Build → pack → predict → return. Caller drives the outer batch loop and
    releases batch_frames after the call. predict_batch internally branches on
    cfg.target_mode, so passing the polarizability weights for a mode-1 model
    is harmless (they're never read).

    Args:
        model        : trained TNEP model (target_mode = 1 or 2)
        builder      : reusable DescriptorBuilder (constructed once per trajectory)
        batch_frames : list of ase.Atoms in this batch
        batch_types  : list of [N_i] int arrays — type indices per frame
        pin_to_cpu   : place batch tensors on CPU (transferred to GPU implicitly).
                       Required for trajectories too large to fit in VRAM.
        descriptor_batch_frames : number of frames per descriptor builder graph
                       call (TF GPU mode only). 1 = per-frame; int >= 2 = batched;
                       None = auto-size to descriptor_memory_budget_bytes.
        descriptor_memory_budget_bytes : GPU memory budget (bytes) used by the
                       auto-sizer when descriptor_batch_frames is None. None
                       falls back to the builder's default (6 GiB). Quippy mode
                       and explicit-int batch sizes ignore this field.
        descriptor_precision : "float64" (default, mirrors quippy/Fortran),
                       "float32" (~2× throughput, ~½ VRAM, looser quippy
                       agreement). None falls back to the builder's
                       cfg.descriptor_precision. Quippy mode ignores this.

    Returns:
        [B, 3] for dipole models, [B, 6] for polarizability models.
    """
    cfg = model.cfg
    # Lazy import to avoid an unconditional TF GPU builder import for callers
    # that only ever use the quippy backend.
    try:
        from DescriptorBuilderGPU_tf import DescriptorBuilderGPUTF
        prefers_tf = isinstance(builder, DescriptorBuilderGPUTF)
    except ImportError:
        prefers_tf = False

    if prefers_tf:
        # GPU descriptor builder → fused pack+predict graph (Phase 3).
        # The fused @tf.function pads descriptors, builds pair_struct, runs
        # predict_batch, and applies the dipole scale factor — all in one
        # graph trace, so XLA can fuse across the boundaries that used to be
        # eager-mode op launches.
        # Switch the builder's compute precision before this batch if a
        # trajectory-time override was passed; build_descriptors_flat
        # rebuilds the locked compute fns lazily when precision changes.
        if descriptor_precision is not None:
            builder.set_precision(descriptor_precision)
        if descriptor_pair_tile_size is not None:
            builder.set_pair_tile_size(int(descriptor_pair_tile_size))
        # XLA-JIT is reserved for the (fp32 + pair-tiling) path only.
        #   - fp64 + XLA suffers from severe register pressure (fp64 takes
        #     2× the register space of fp32), causing big spill kernels and
        #     a net slowdown that more than negates the fusion benefit. We
        #     observed ~10× slowdowns on water_bulk fp64 under XLA. Best
        #     to keep fp64 always non-XLA.
        #   - Without pair-tiling the per-call shape varies and XLA
        #     recompiles each batch (~5-7 s ptxas per compile, with
        #     register-spill warnings).
        # With pair-tiling enabled and fp32 selected, pairs are padded to a
        # multiple of pair_tile_size and each tile body has a fixed shape;
        # XLA compiles each (n_atoms_chunk, n_tiles) bucket once and runs
        # at full fused-kernel speed afterward.
        ptile = int(getattr(builder, "_pair_tile_size", 0))
        builder_is_fp32 = (
            getattr(builder, "_real_dtype", None) is not None
            and builder._real_dtype == tf.float32
        )
        use_xla = ptile > 0 and builder_is_fp32
        frame_results = builder.build_descriptors_flat(
            batch_frames,
            batch_frames=descriptor_batch_frames,
            memory_budget_bytes=descriptor_memory_budget_bytes,
            return_tf=True,
            jit_compile=use_xla,
        )
        fused_inputs = _build_fused_inputs(
            frame_results, batch_frames, batch_types, cfg.dim_q,
            pin_to_cpu=pin_to_cpu,
        )
        # _build_fused_inputs cleared frame_results; drop the empty handle.
        del frame_results
        fused_predict = _get_fused_predict(model)
        preds = fused_predict(*fused_inputs)
        # The graph captured / copied its inputs already; drop the chunk-level
        # concat tensors before the .numpy() sync so the next iteration has
        # the full VRAM budget available for the SOAP build.
        del fused_inputs
        out = preds.numpy()
        del preds
        out = _restore_target_mean(out, batch_frames, cfg)
    else:
        # Legacy NumPy pack + eager predict_batch path (quippy and any
        # custom builder that doesn't return TF tensors).
        frame_results = builder.build_descriptors_flat(
            batch_frames,
            batch_frames=descriptor_batch_frames,
            memory_budget_bytes=descriptor_memory_budget_bytes,
        )
        batch = _pack_traj_batch_from_flat(frame_results, batch_frames, batch_types,
                                           cfg.dim_q, pin_to_cpu=pin_to_cpu)
        del frame_results

        # Apply per-channel scaling and mixing absorption in the same
        # order as the fused path so the two backends produce identical
        # predictions. Both transformations are pre-existing training-
        # time operations whose absence at inference would silently
        # corrupt trajectory dipoles.
        if (str(getattr(cfg, "descriptor_scaling", "none")) != "none"
                and getattr(cfg, "_q_scaler", None) is not None):
            s = tf.constant(cfg._q_scaler, dtype=tf.float32)
            batch["descriptors"] = batch["descriptors"] * s[tf.newaxis, tf.newaxis, :]
            batch["grad_values"] = batch["grad_values"] * s[tf.newaxis, tf.newaxis, :]
        if getattr(model, "descriptor_mixing", False) and model.U_pair is not None:
            W0_pred = model._W0_eff(model.W0)
            W0p_pred = (model._W0_eff(model.W0_pol)
                         if (cfg.target_mode == 2
                             and getattr(model, "W0_pol", None) is not None)
                         else getattr(model, "W0_pol", None))
        else:
            W0_pred = model.W0
            W0p_pred = getattr(model, "W0_pol", None)

        preds = model.predict_batch(
            batch["descriptors"], batch["grad_values"],
            batch["pair_atom"], batch["pair_gidx"], batch["pair_struct"],
            batch["positions"], batch["Z_int"], batch["boxes"],
            batch["atom_mask"],
            W0_pred, model.b0, model.W1, model.b1,
            W0p_pred,
            getattr(model, 'b0_pol', None),
            getattr(model, 'W1_pol', None),
            getattr(model, 'b1_pol', None),
        )
        # NOTE: predict_batch returns the TOTAL dipole regardless of
        # cfg.scale_targets. Training stores per-atom *targets*, and
        # TNEP.score() divides raw_preds by N to compare on the same scale
        # (TNEP.py:285-288). So preds is already the total system dipole;
        # no per-atom→total rescaling is needed here.
        out = preds.numpy()
        del batch, preds
        out = _restore_target_mean(out, batch_frames, cfg)
    return out


def _restore_target_mean(out: np.ndarray, batch_frames: list,
                         cfg) -> np.ndarray:
    """Add the frozen training-set target mean back to trajectory
    predictions so the returned array is in original (un-centered) units.

    Mirrors the inverse done in TNEP.score / TNEP.predict. The shift
    depends on whether the data pipeline scaled targets per-atom:
      - target_mode=1, scale_targets=True : `out += mean * num_atoms_i`
        per frame (the training shift was per-atom; raw_pred is total).
      - target_mode=1, scale_targets=False: `out += mean`
      - target_mode=2 (polarisability)    : `out += mean` (mode=2 has
        no scale_targets path in assemble_data_dict, so the mean is
        always in total-space).
    Mode 0 (energy) does not enter this path — predict_trajectory_batch
    is restricted to modes 1/2. No-op when target_centering is off.
    """
    if not (bool(getattr(cfg, "target_centering", False))
            and getattr(cfg, "_target_mean", None) is not None):
        return out
    mean = np.asarray(cfg._target_mean, dtype=np.float32).reshape(1, -1)
    # Mirror assemble_data_dict's gating exactly: per-atom rescaling
    # applies ONLY when target_mode==1 AND cfg.scale_targets is True.
    # Default for missing attr is False (matches data.py:375 semantics).
    if cfg.target_mode == 1 and bool(getattr(cfg, "scale_targets", False)):
        # Per-frame num_atoms scaling. batch_frames length equals out.shape[0].
        n_atoms = np.asarray(
            [len(f) for f in batch_frames], dtype=np.float32).reshape(-1, 1)
        return out + mean * n_atoms
    return out + mean


def _scalar_acf_fft(signal: np.ndarray) -> np.ndarray:
    """Compute autocorrelation of a 1D signal via FFT (Wiener-Khinchin).

    Args:
        signal : [T] ndarray

    Returns:
        acf : [T] ndarray — unnormalised ACF (divided by overlap count)
    """
    T = len(signal)
    n_fft = 2 * T
    fd = np.fft.rfft(signal, n=n_fft)
    power = np.real(fd * np.conj(fd))
    acf = np.fft.irfft(power, n=n_fft)[:T]
    counts = np.arange(T, 0, -1, dtype=np.float64)
    acf /= counts
    return acf



def compute_raman_acfs(polarizabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic and anisotropic polarizability autocorrelation functions.

    Decomposes the polarizability tensor into isotropic (γ) and anisotropic (β)
    parts, then computes their respective ACFs.

    Ref: Xu et al., J. Chem. Theory Comput., 2024, 20, 3273–3284, Eq. 12

    Decomposition:
        γ(t) = (α_xx + α_yy + α_zz) / 3     (isotropic scalar)
        β_ij = α_ij - γ·δ_ij                  (traceless anisotropic tensor)

    ACFs:
        C_iso(τ)   = <γ(0)·γ(τ)>
        C_aniso(τ) = <β_ij(0)·β_ij(τ)>       (full tensor contraction)

    Args:
        polarizabilities : [T, 6] ndarray — [xx, yy, zz, xy, yz, zx] per frame

    Returns:
        acf_iso   : [T] ndarray — isotropic ACF
        acf_aniso : [T] ndarray — anisotropic ACF
    """
    xx = polarizabilities[:, 0]
    yy = polarizabilities[:, 1]
    zz = polarizabilities[:, 2]
    xy = polarizabilities[:, 3]
    yz = polarizabilities[:, 4]
    zx = polarizabilities[:, 5]

    # Isotropic part: γ = Tr(α)/3
    gamma = (xx + yy + zz) / 3.0
    acf_iso = _scalar_acf_fft(gamma)

    # Anisotropic (traceless) part: β_ij = α_ij - γ·δ_ij
    beta_xx = xx - gamma
    beta_yy = yy - gamma
    beta_zz = zz - gamma
    # Off-diagonal β components equal α (since δ_ij = 0 for i≠j)
    beta_xy = xy
    beta_yz = yz
    beta_zx = zx

    # C_aniso(τ) = <β_ij(0)·β_ij(τ)> summed over all i,j
    # Diagonal: β_xx·β_xx + β_yy·β_yy + β_zz·β_zz
    # Off-diagonal (×2 for symmetry): 2(β_xy·β_xy + β_yz·β_yz + β_zx·β_zx)
    acf_aniso = (_scalar_acf_fft(beta_xx)
                 + _scalar_acf_fft(beta_yy)
                 + _scalar_acf_fft(beta_zz)
                 + 2.0 * _scalar_acf_fft(beta_xy)
                 + 2.0 * _scalar_acf_fft(beta_yz)
                 + 2.0 * _scalar_acf_fft(beta_zx))

    return acf_iso, acf_aniso


def compute_raman_spectrum(polarizabilities: np.ndarray, dt_fs: float = 1.0, window: str | None = 'hann',
                           max_freq_cm: float = 4000.0, temperature: float = 300.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Raman spectrum from a polarizability trajectory.

    Follows Xu et al., J. Chem. Theory Comput., 2024, 20, 3273–3284:
        1. Decompose α(t) into isotropic γ(t) and anisotropic β(t)  (Eq. 12)
        2. Compute ACFs: C_iso(τ), C_aniso(τ)
        3. Fourier transform to get line shapes
        4. Assemble parallel/perpendicular spectra  (Eq. 10-11)
        5. Apply Bose-Einstein correction: (n(ω) + 1) / ω

    Polarised (VV) and depolarised (VH) Raman intensities:
        I_VV(ω) ∝ [45·L_iso(ω) + 4·L_aniso(ω)] · (n(ω)+1)/ω
        I_VH(ω) ∝ 3·L_aniso(ω) · (n(ω)+1)/ω

    Args:
        polarizabilities : [T, 6] ndarray — [xx, yy, zz, xy, yz, zx] per frame
        dt_fs            : float — timestep between frames in femtoseconds
        window           : str or None — window function ('hann', 'blackman', None)
        max_freq_cm      : float — maximum frequency in cm⁻¹
        temperature      : float — temperature in Kelvin for Bose-Einstein factor

    Returns:
        freq_cm  : [N] ndarray — frequencies in cm⁻¹
        I_VV     : [N] ndarray — parallel (polarised) Raman intensity
        I_VH     : [N] ndarray — perpendicular (depolarised) Raman intensity
        I_total  : [N] ndarray — total unpolarised Raman intensity
        acf_iso  : [T] ndarray — isotropic ACF
        acf_aniso: [T] ndarray — anisotropic ACF
    """
    acf_iso, acf_aniso = compute_raman_acfs(polarizabilities)
    T = len(acf_iso)

    # Apply window
    if window == 'hann':
        w = np.hanning(T)
    elif window == 'blackman':
        w = np.blackman(T)
    else:
        w = np.ones(T)

    L_iso = np.real(np.fft.rfft(acf_iso * w))
    L_aniso = np.real(np.fft.rfft(acf_aniso * w))

    # Frequency axis: 1/fs -> cm⁻¹
    freq_per_fs = np.fft.rfftfreq(T, d=dt_fs)
    c_cm_per_fs = 2.99792458e-5  # speed of light in cm/fs
    freq_cm = freq_per_fs / c_cm_per_fs

    # Bose-Einstein correction: (n(ω) + 1) / ω
    # n(ω) = 1 / (exp(ℏω/kT) - 1)
    # ℏω in eV: ℏ·c·ν̃ where ν̃ in cm⁻¹
    # ℏc = 1.23984e-4 eV·cm, kT at 300K = 0.02585 eV
    hbar_c_eV_cm = 1.23984e-4  # eV·cm
    kT = 8.617333e-5 * temperature  # eV (k_B in eV/K)

    bose_factor = np.ones_like(freq_cm)
    for i in range(1, len(freq_cm)):  # skip ω=0
        hw = hbar_c_eV_cm * freq_cm[i]  # ℏω in eV
        x = hw / kT
        if x < 500:  # avoid overflow
            n_bose = 1.0 / (np.exp(x) - 1.0)
            bose_factor[i] = (n_bose + 1.0) / freq_cm[i]
        else:
            bose_factor[i] = 0.0
    bose_factor[0] = 0.0  # DC component

    # Raman intensities (Eq. 10-11)
    I_VV = (45.0 * L_iso + 4.0 * L_aniso) * bose_factor
    I_VH = 3.0 * L_aniso * bose_factor
    I_total = I_VV + I_VH

    # Truncate to frequency range
    mask = freq_cm <= max_freq_cm
    freq_cm = freq_cm[mask]
    I_VV = I_VV[mask]
    I_VH = I_VH[mask]
    I_total = I_total[mask]

    # Normalise to peak = 1
    peak = np.max(np.abs(I_total))
    if peak > 0:
        I_VV /= peak
        I_VH /= peak
        I_total /= peak

    return freq_cm, I_VV, I_VH, I_total, acf_iso, acf_aniso


def plot_raman_spectrum(freq_cm: np.ndarray, I_VV: np.ndarray, I_VH: np.ndarray,
                        I_total: np.ndarray, cfg: TNEPconfig,
                        save_plots: str | None = None, show_plots: bool = True,
                        title: str = "Raman Spectrum") -> None:
    """Plot Raman spectrum with parallel, perpendicular, and total components.

    Args:
        freq_cm    : [N] ndarray — frequencies in cm⁻¹
        I_VV       : [N] ndarray — parallel (polarised) intensity
        I_VH       : [N] ndarray — perpendicular (depolarised) intensity
        I_total    : [N] ndarray — total intensity
        cfg        : TNEPconfig — used for filename generation
        save_plots : str or None — directory to save into
        show_plots : bool — True to display interactively
        title      : str — plot title
    """
    from plotting import _finish_fig
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_cm, I_total, color='black', linewidth=0.8, label="Total")
    ax.plot(freq_cm, I_VV, color='blue', linewidth=0.6, alpha=0.7, label="VV (polarised)")
    ax.plot(freq_cm, I_VH, color='red', linewidth=0.6, alpha=0.7, label="VH (depolarised)")
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Raman Intensity (arb. units)")
    ax.set_title(title)
    ax.set_xlim(freq_cm[0], freq_cm[-1])
    ax.set_ylim(1, 0)  # intensity descends from 1 at top to 0 at bottom
    ax.legend()
    plt.tight_layout()
    _finish_fig(fig, cfg, "raman_spectrum", save_plots, show_plots)
