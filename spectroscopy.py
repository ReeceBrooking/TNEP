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
from data import cell_to_box, assign_type_indices
from tqdm import tqdm

if TYPE_CHECKING:
    from ase import Atoms
    from TNEP import TNEP


def compute_dipole_acf(dipoles: np.ndarray) -> np.ndarray:
    """Dipole ACF ⟨μ(0)·μ(τ)⟩ via Wiener-Khinchin (FFT, O(N log N)). Xu et al. 2024, Eq. 9.

    Sums (not averages) over x,y,z. Biased estimator: each lag divided by
    full length T (not T-τ), matching GPUMD/Xu — decays smoothly to zero at
    large τ instead of amplifying the noisy tail.

    Args:
        dipoles : [T, 3] ndarray — dipole trajectory (one per MD frame)

    Returns:
        acf : [T] ndarray — dipole ACF (e²·Å² if dipoles in e·Å). NOT normalised to acf[0]=1.
    """
    T = dipoles.shape[0]
    # Zero-pad to ≥ 2T-1 for linear (non-circular) correlation, rounded up to
    # a fast FFT length. next_fast_len lives in scipy.fft; fall back to 2*T.
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
    # Biased estimator: divide by constant T, not (T-τ).
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
    """IR absorption spectrum from a dipole trajectory. GPUMD / Xu et al. 2024.

    Pipeline: subtract mean dipole → ACF C(τ)=<μ(0)·μ(τ)> → truncate to first
    acf_ratio → Hann window + Kronecker doubling → cosine transform to line
    shape M(ω) ≥ 0 → weight → smooth.

    IR weighting (harmonic): σ(ω) ∝ ω·(1 − e^(−ℏω/kT))·M(ω). Reduces to
    classical ω²·M(ω) only for ℏω ≪ kT (ν̃ ≪ 210 cm⁻¹ at 300 K); beyond that
    the ω² form overweights high-freq modes (~14× at 3000 cm⁻¹), masking
    lower modes.

    Args:
        dipoles            : [T, 3] ndarray — dipole trajectory (e·Å, one per frame)
        dt_fs              : float — timestep between frames in fs
        window             : str or None — 'hann', 'blackman', or None
        max_freq_cm        : float — max frequency returned, cm⁻¹
        acf_ratio          : float — fraction of trajectory used as max ACF lag (default 0.1)
        smooth_k           : int — smoothing strength (higher = smoother). For
                              smooth_kind="gaussian" it is the kernel FWHM in
                              frequency bins; for "box" the moving-average width
                              in bins. 0 = disable.
        smooth_kind        : str — "gaussian" (default; no ringing) or "box"
                              (moving average; matches GPUMD but has sidelobes).
        temperature        : float — simulation T in K (used when quantum_correction
                              != "classical"). Default 300 K.
        quantum_correction : str — IR weighting:
                              "harmonic"  : ω·(1 − e^(−ℏω/kT))·M(ω)  — GPUMD default
                              "classical" : ω²·M(ω)                  — Xu Eq. 1
                              "quadratic" : alias for "classical"
                              "linear"    : ω·M(ω)                   — high-freq limit
                              "none"      : M(ω)
        power_dc_cutoff_cm : float — exclude bins below this from the
                              power-spectrum peak-normaliser (raw M(ω) has a
                              huge DC peak that would crush vibrational features
                              to ~1%). 0 keeps the DC bin. Default 100 cm⁻¹.
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

    # Cosine transform: M(k) = Σ_t acf(t)·cos(2πkt/(2Nmax-1)) = real part of
    # rfft of acf_prepared zero-padded to length 2Nmax-1.
    M_omega = np.fft.rfft(acf_prepared, n=2 * Nmax - 1).real

    # Frequency axis: DCT bin k → k/((2Nmax-1)·dt_fs) in 1/fs → cm⁻¹.
    c_cm_per_fs = 2.99792458e-5  # speed of light in cm/fs
    freq_cm = np.arange(Nmax) / ((2 * Nmax - 1) * dt_fs * c_cm_per_fs)

    # IR absorption weighting (see docstring for the harmonic vs classical forms).
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
        # (1-e^{-x}) → x as x→0 (classical ω² limit), → 1 for large x (ω limit).
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

    # Smooth. "gaussian": FWHM in bins (σ = smooth_k/2.355), preserves freq_cm
    # length. "box": moving average of width smooth_k bins; mode='valid'
    # shortens freq_cm.
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

    # Raw M(ω) peaks at ω≈0, so normalising by that DC peak would collapse
    # vibrational features to <1%. Exclude bins below power_dc_cutoff_cm from
    # the normaliser (0 to disable).
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
    """Build a plot stem from trajectory + model names, e.g.
    ("ethanol_nve.traj", "n50_q165_CHO") → "ethanol_nve_n50_q165_CHO".
    Skips None/empty parts; falls back to "ir_spectrum".
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
    """Plot one IR trace into a single axis (split_at_cm=None → ax_hi only)
    or a broken-axis pair (ν̃ ≤ split → ax_lo, ν̃ > split → ax_hi). Each side
    is renormalised to peak=1 within its window so high-freq structure isn't
    crushed by low-freq peaks; diagonal break marks join the halves.
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

    # Split-region rendering: two adjacent axes, each independently peak-normalised.
    mask_lo = freq_cm <= split_at_cm
    mask_hi = freq_cm >  split_at_cm
    if not mask_lo.any() or not mask_hi.any():
        # Fall back to single-axis if the split lands outside the data
        _plot_one_ir_panel(None, ax_hi, freq_cm, y, None, ylabel, title)
        return

    f_lo, y_lo = freq_cm[mask_lo], y[mask_lo]
    f_hi, y_hi = freq_cm[mask_hi], y[mask_hi]

    if invert_y:
        # y is transmittance from global-peak absorbance. To renormalise per
        # region: recover absorbance, peak-normalise per region, re-apply T(A).
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

    # IR convention: high wavenumber on the left → ax_hi left, ax_lo right.
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
    # x-label added at figure level below (shared across the pair).
    ax_hi.set_xlabel("")
    ax_lo.set_xlabel("")


def _emit_single_ir_figure(freq_cm: np.ndarray, y: np.ndarray,
                            label: str, ylabel: str,
                            title: str, split_at_cm: float | None,
                            stem: str, cfg: TNEPconfig,
                            save_plots: str | None, show_plots: bool,
                            invert_y: bool) -> None:
    """Emit ONE labelled figure (absorbance OR transmittance), saved separately."""
    from plotting import _save_fig
    if split_at_cm is None:
        fig, ax = plt.subplots(figsize=(14, 6))
        _plot_one_ir_panel(None, ax, freq_cm, y, None,
                           ylabel=ylabel, title=title)
    else:
        # Broken-axis pair [hi | lo]. Width ratios proportional to each side's
        # wavenumber range, so cm⁻¹-per-pixel is identical on both halves.
        f_max = float(freq_cm.max())
        f_min = float(freq_cm.min())
        f_split = float(split_at_cm)
        # Clamp against zero/negative widths from a split outside the data range.
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


# Module-level T(A) convention set by `plot_ir_spectrum`, consumed by
# `_plot_one_ir_panel` for per-region transmittance reconstruction.
_TRANSMITTANCE_MODE: str = "beer_lambert"
_TRANSMITTANCE_SCALE: float = 1.0


def _absorbance_to_transmittance(A: np.ndarray) -> np.ndarray:
    """A → T under the configured convention.

    "beer_lambert" : T = 10^(−scale·A)  (T≈0.1 at A=1, scale=1)
    "linear"       : T = 1 − A          (visual mirror, common in comp-IR)
    """
    A = np.asarray(A)
    if _TRANSMITTANCE_MODE == "beer_lambert":
        return 10.0 ** (-_TRANSMITTANCE_SCALE * A)
    if _TRANSMITTANCE_MODE == "linear":
        return 1.0 - A
    raise ValueError(f"Unknown transmittance_mode {_TRANSMITTANCE_MODE!r}")


def _transmittance_to_absorbance(T: np.ndarray) -> np.ndarray:
    """T → A, inverse of `_absorbance_to_transmittance`. Clipped to keep
    log/inverse stable against FFT round-off (T slightly outside [0,1])."""
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
    """Plot IR spectrum as two separate labelled figures (absorbance and
    transmittance), saved as independent files <stem>_absorbance.png /
    <stem>_transmittance.png (or _split500 etc. when split_at_cm is set).
    Absorbance peaks point up from 0; transmittance peaks dip down from 1.

    Args:
        freq_cm             : [N] ndarray — frequencies in cm⁻¹
        intensity           : [N] ndarray — normalised IR intensity (peak=1)
        cfg                 : TNEPconfig — save-directory resolution
        save_plots          : str or None — directory to save into
        show_plots          : bool — display interactively
        title               : str — figure title prefix
        trajectory_path     : str or None — appears in filename
        model_label         : str or None — appears in filename
        split_at_cm         : float or None — broken-axis split with per-region
                              peak-normalisation. None = full range.
        transmittance_mode  : str — A → T conversion:
                              "beer_lambert" (default): T = 10^(−scale·A)
                              "linear" : T = 1 − A (visual mirror only)
        transmittance_scale : float — multiplier on A for Beer-Lambert mode;
                              higher → deeper dips. Default 1.0 gives ~10%
                              minimum transmission at the strongest peak.
    """
    # Stash the chosen mode on module state for the per-region renormaliser.
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
    """Load a saved dipole trajectory (.npy or .txt, shape [T,3]) and plot its
    IR + power spectra. Wraps np.load/np.loadtxt + compute_ir_spectrum; emits
    absorbance + transmittance as separate files plus the power-spectrum plot.

    Args:
        dipole_path        : dipole file (.npy or .txt), shape [T, 3].
        dt_fs              : timestep between frames in fs.
        save_dir           : directory to save plots into (None = don't save).
        show               : display interactively.
        title              : plot title (defaults to file basename).
        model_label        : model id baked into the filename (e.g. "n50_q165_CHO").
        plot_power         : also produce the power-spectrum plot.
        split_at_cm        : if set, render broken-axis pairs split at this
                             wavenumber, each side peak-normalised. None = full range.
        cfg                : TNEPconfig for unit labels (minimal default if None).
        window             : ACF window — 'hann' (default), 'blackman', or None.
        max_freq_cm        : max wavenumber kept (cm⁻¹).
        acf_ratio          : fraction of trajectory used as max ACF lag.
        smooth_k           : smoothing strength (Gaussian FWHM / box width in bins).
        smooth_kind        : "gaussian" (default) or "box".
        temperature        : simulation T in K (harmonic correction).
        quantum_correction : IR weighting — see compute_ir_spectrum.
        power_dc_cutoff_cm : exclude bins below this from the power normaliser
                             and visible range. 0 disables.
        **ir_kwargs        : extra kwargs forwarded to compute_ir_spectrum.

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

    # Plotters consult cfg only for unit labels; a fresh default is fine.
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
        # Same DC cutoff for visible range and normaliser, for consistency.
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

    M(ω) has a strong DC peak (ω≈0) dwarfing vibrational features, so the
    visible range is cropped to ν̃ ≥ low_cm_cutoff and the y-axis defaults to
    log (vibrational modes span 2-4 orders of magnitude).

    Args:
        freq_cm         : [N] ndarray — frequencies in cm⁻¹
        power           : [N] ndarray — normalised power spectrum M(ω)
        cfg             : TNEPconfig — save-directory resolution
        save_plots      : str or None — directory to save into (None = don't save)
        show_plots      : bool — display interactively
        title           : str — plot title
        low_cm_cutoff   : float — drop frequencies below this (default 100; 0 = full range)
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
# Fused pack + predict @tf.function, cached per model instance (traced once).
# [None] dims in the input signature keep batch/atom/pair size changes from
# retriggering tracing.
# --------------------------------------------------------------------------
_FUSED_PREDICT_CACHE: dict = {}


def _get_fused_predict(model: 'TNEP'):
    """Return (and cache) a per-model fused pack+predict @tf.function.

    Captures model weights via closure, so this is trajectory-inference-only:
    SNES candidate evaluation (which swaps weights) must not use it. Cached by id(model).
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

    # Keras Variables can't be direct @tf.function args, so materialise them
    # into tf.constant tensors at trace time and capture via closure (safe
    # since trajectory inference is fixed-weight).
    def _to_tensor(v):
        if v is None:
            return None
        # Keras 3 Variables expose .value (a Tensor); fall back to convert.
        return tf.convert_to_tensor(v.value if hasattr(v, "value") else v)

    # Pre-absorb U_pairᵀ into W0 (and W0_pol) when descriptor mixing is active;
    # predict_batch trusts the caller to do this. Mirrors score() in TNEP.py.
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
    # Optional second hidden layer (None for 1-layer models → single-layer forward).
    Wh_t  = _to_tensor(getattr(model, "Wh", None))
    bh_t  = _to_tensor(getattr(model, "bh", None))
    # Optional charge head.
    W1q_t = _to_tensor(getattr(model, "W1_q", None))
    bq_t  = _to_tensor(getattr(model, "b_q", None))

    # Not jit_compile: predict_batch has shape-dependent stacks in
    # _calc_forces_coo (varying P per call) that XLA can't lower, and the
    # heavy SOAP work is already handled by the descriptor-side XLA path.
    @tf.function(input_signature=sig, reduce_retracing=False)
    def fused(soap_concat, grad_concat, pa_concat, pg_concat,
              atom_counts, pair_counts,
              positions, Z, boxes, atom_mask, num_atoms):
        S = tf.shape(num_atoms)[0]
        # Pad descriptors via scatter_nd to dense [S, A_max, Q] (RaggedTensor
        # .to_tensor has no XLA kernel), using per-row struct/intra indices.
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
            Wh=Wh_t, bh=bh_t,
            W1_q=W1q_t, b_q=bq_t,
        )
        # predict_batch returns the TOTAL dipole (not per-atom) regardless of
        # cfg.scale_targets, so do NOT multiply by num_atoms.
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
    Returned tuple matches the input_signature of `_get_fused_predict`.

    Destructive: clears `frame_results` once its data is folded into the
    concats, halving peak VRAM during the pack step.
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
    # All per-frame data is now in the concats; release the caller's list.
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


def _pack_traj_batch_from_flat(
    frame_results: list,
    frames: list,
    types_int_batch: list[np.ndarray],
    dim_q: int,
    pin_to_cpu: bool = True,
) -> dict:
    """Pack flat per-frame COO arrays (all numpy) into a stacked batch for
    predict_batch: concatenate pair-level arrays and build a struct-index
    from per-frame counts (no per-atom loop).

    Each frame_results[s] = (descriptors[N,Q], grad_values[P_s,3,Q],
    pair_atom[P_s], pair_gidx[P_s]).

    Returns:
        dict with descriptors [B,A,Q], grad_values [P,3,Q], pair_atom/gidx/struct [P],
        positions [B,A,3], Z_int [B,A], boxes [B,3,3], atom_mask [B,A], num_atoms [B]
        — same layout as data.pad_and_stack().
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
    """Run dipole/polarizability prediction on one batch of trajectory frames
    (build → pack → predict → return). Caller drives the outer batch loop.
    predict_batch branches on cfg.target_mode, so passing polarizability
    weights for a mode-1 model is harmless (never read).

    Args:
        model        : trained TNEP model (target_mode = 1 or 2)
        builder      : reusable DescriptorBuilder (one per trajectory)
        batch_frames : list of ase.Atoms in this batch
        batch_types  : list of [N_i] int arrays — type indices per frame
        pin_to_cpu   : place batch tensors on CPU (needed for trajectories too
                       large to fit in VRAM).
        descriptor_batch_frames : frames per descriptor builder graph call (TF
                       GPU only). 1 = per-frame; ≥2 = batched; None = auto-size.
        descriptor_memory_budget_bytes : GPU budget for the auto-sizer when
                       descriptor_batch_frames is None (default 6 GiB). Ignored
                       by quippy and explicit-int batch sizes.
        descriptor_precision : "float64" (default, mirrors quippy) or "float32"
                       (~2× throughput, ~½ VRAM, looser agreement). None →
                       builder default. Ignored by quippy.

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
        # GPU descriptor builder → fused pack+predict graph.
        # Switch compute precision before this batch if overridden;
        # build_descriptors_flat rebuilds the locked compute fns lazily.
        if descriptor_precision is not None:
            builder.set_precision(descriptor_precision)
        if descriptor_pair_tile_size is not None:
            builder.set_pair_tile_size(int(descriptor_pair_tile_size))
        # XLA-JIT only for the (fp32 + pair-tiling) path:
        #   - fp64 + XLA suffers severe register pressure and spill kernels
        #     (~10× slowdown observed on water_bulk); keep fp64 non-XLA.
        #   - Without pair-tiling the per-call shape varies and XLA recompiles
        #     each batch (~5-7 s ptxas per compile).
        # With fp32 + pair-tiling, pairs are padded to a multiple of
        # pair_tile_size (fixed tile shape); XLA compiles each bucket once.
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
        # Graph already captured its inputs; drop the concat tensors before the
        # .numpy() sync so the next SOAP build has the full VRAM budget.
        del fused_inputs
        out = preds.numpy()
        del preds
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

        # Apply mixing absorption in the same order as the fused path so both
        # backends produce identical predictions.
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
            Wh=getattr(model, 'Wh', None), bh=getattr(model, 'bh', None),
            W1_q=getattr(model, 'W1_q', None), b_q=getattr(model, 'b_q', None),
        )
        # predict_batch returns the TOTAL dipole regardless of
        # cfg.scale_targets; no per-atom→total rescaling needed.
        out = preds.numpy()
        del batch, preds
    return out


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
    """Isotropic and anisotropic polarizability ACFs. Xu et al. 2024, Eq. 12.

    Decompose α into isotropic γ = Tr(α)/3 and traceless β_ij = α_ij − γ·δ_ij, then:
        C_iso(τ)   = <γ(0)·γ(τ)>
        C_aniso(τ) = <β_ij(0)·β_ij(τ)>   (full tensor contraction)

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
    """Raman spectrum from a polarizability trajectory. Xu et al. 2024.

    Pipeline: decompose α → ACFs C_iso/C_aniso → FFT to line shapes L_iso/L_aniso
    → assemble VV/VH → Bose-Einstein correction (n(ω)+1)/ω. Intensities (Eq. 10-11):
        I_VV(ω) ∝ [45·L_iso + 4·L_aniso] · (n(ω)+1)/ω    (polarised)
        I_VH(ω) ∝ 3·L_aniso · (n(ω)+1)/ω                 (depolarised)

    Args:
        polarizabilities : [T, 6] ndarray — [xx, yy, zz, xy, yz, zx] per frame
        dt_fs            : float — timestep between frames in fs
        window           : str or None — 'hann', 'blackman', or None
        max_freq_cm      : float — max frequency in cm⁻¹
        temperature      : float — T in K for the Bose-Einstein factor

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

    # Bose-Einstein correction: (n(ω)+1)/ω with n(ω)=1/(exp(ℏω/kT)-1),
    # ℏω [eV] = ℏc·ν̃ [cm⁻¹].
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


# --------------------------------------------------------------------------
# Finite-difference dipole derivatives: ∂μ_α/∂r_iβ and ∂²μ_α/∂r_iβ∂r_jγ.
# The SOAP backends supply first descriptor derivatives only and enter TF as
# tf.constant(rjs, thetas, phis) built in NumPy, so there is no differentiable
# positions → μ path; FD over predict_trajectory_batch is the whole method.
# --------------------------------------------------------------------------


def _fd_stencil(D: int):
    """Yield central-difference points as ((dof, sign), ...), in the order
    _fd_assemble indexes them: base, the 2·D single displacements (shared by
    the first derivative and both second-derivative formulas), then the two
    diagonal points (++, --) per off-diagonal pair.

    1 + 2D + D(D-1) points ≈ D² — half the 4-point mixed formula's cost at the
    same O(h²) error, because the singles are reused.
    """
    yield ()
    for a in range(D):
        yield ((a, 1),)
        yield ((a, -1),)
    for a in range(D):
        for b in range(a + 1, D):
            yield ((a, 1), (b, 1))
            yield ((a, -1), (b, -1))


def _fd_assemble(P: np.ndarray, D: int, h: float):
    """Turn the [n_points, 3] stencil evaluations into (μ, ∂μ, ∂²μ).

        ∂μ/∂x_a    = [f(+a) - f(-a)] / 2h
        ∂²μ/∂x_a²  = [f(+a) - 2f(0) + f(-a)] / h²
        ∂²μ/∂x_a∂x_b = [f(+a+b) + f(-a-b) - f(+a) - f(-a) - f(+b) - f(-b)
                        + 2f(0)] / 2h²
    """
    mu = P[0]
    sp = P[1:1 + 2 * D:2]           # f(+a)
    sm = P[2:1 + 2 * D:2]           # f(-a)
    pp = P[1 + 2 * D::2]            # f(+a+b), pair order = np.triu_indices
    mm = P[2 + 2 * D::2]            # f(-a-b)

    dmu = (sp - sm) / (2.0 * h)

    d2 = np.empty((D, D, 3), dtype=np.float64)
    ia, ib = np.triu_indices(D, 1)
    off = (pp + mm - sp[ia] - sm[ia] - sp[ib] - sm[ib] + 2.0 * mu) / (2.0 * h * h)
    d2[ia, ib] = off
    d2[ib, ia] = off
    d = np.arange(D)
    d2[d, d] = (sp - 2.0 * mu + sm) / (h * h)
    return mu, dmu, d2


def _fd_derivatives(model, frames, dofs, h, builder, batch_size, verbose,
                    **traj_kw) -> np.ndarray:
    """Evaluate the FD stencil for `dofs` on every frame; returns [F, n_points, C].

    One flat stream of (frame, stencil point) over the WHOLE trajectory, fed to
    predict_trajectory_batch in full-size batches — the same batching the
    process_trajectory path relies on. Chunking per frame instead would leave
    the last chunk of every frame short and re-pay the XLA retrace each time.
    """
    from itertools import islice

    D = len(dofs)
    n_points = 1 + 2 * D + D * (D - 1)
    types_per_frame = assign_type_indices(frames, model.cfg.types)

    def _points():
        for fi, frame in enumerate(frames):
            for pts in _fd_stencil(D):
                yield fi, frame, pts

    stream = _points()
    preds = []
    pbar = tqdm(total=len(frames) * n_points, desc="FD evals",
                unit="eval", disable=not verbose)
    while True:
        chunk = list(islice(stream, batch_size))
        if not chunk:
            break
        batch, batch_types = [], []
        for fi, frame, pts in chunk:
            f = frame.copy()
            for a, s in pts:
                i, c = dofs[a]
                f.positions[i, c] += s * h
            batch.append(f)
            batch_types.append(types_per_frame[fi])
        preds.append(predict_trajectory_batch(
            model, builder, batch, batch_types, **traj_kw))
        pbar.update(len(chunk))
    pbar.close()

    # Stream is frame-major, so a flat concat reshapes straight into frames.
    P = np.concatenate(preds, axis=0).astype(np.float64)
    return P.reshape(len(frames), n_points, P.shape[-1])


def dipole_derivatives_fd(
    model: 'TNEP',
    frames,
    atom_idx=None,
    h: float = 0.02,
    builder: DescriptorBuilder | None = None,
    batch_size: int = 64,
    index: str = ':',
    verbose: bool = True,
    **traj_kw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dipoles and their first and second position derivatives for a truncated
    trajectory, by central finite differences over predict_trajectory_batch.

    Args:
        model     : trained dipole model (cfg.target_mode == 1)
        frames    : list of ase.Atoms, or a path to an .xyz read with `index`
        atom_idx  : atoms to differentiate w.r.t. (default: all — see cost).
                    Restrict this: cost is ≈ (3·n_sel)² dipole evaluations PER
                    FRAME, so a 192-atom box is ~3.3e5 evals/frame while one
                    water molecule is 91.
        h         : displacement in Å. Predictions are float32, so the second
                    derivative's roundoff floor scales as ~1e-7/h²; below
                    ~0.01 Å the noise beats the O(h²) truncation error.
        builder   : descriptor builder (default: model.builder)
        **traj_kw : forwarded to predict_trajectory_batch (descriptor_precision,
                    pin_to_cpu, ...)

    Returns:
        mu   : [F, 3]
        dmu  : [F, n_sel, 3, 3]            ∂μ_α/∂r_iβ  (index order i, β, α)
        d2mu : [F, n_sel, 3, n_sel, 3, 3]  ∂²μ_α/∂r_iβ∂r_jγ
    """
    if model.cfg.target_mode != 1:
        raise ValueError(
            f"dipole_derivatives_fd needs a dipole model (target_mode=1), "
            f"got target_mode={model.cfg.target_mode}.")

    if isinstance(frames, str):
        from ase.io import read as _ase_read
        frames = _ase_read(frames, index=index)
    if not isinstance(frames, list):
        frames = list(frames)
    if builder is None:
        builder = model.builder

    n_atoms = len(frames[0])
    if atom_idx is None:
        atom_idx = np.arange(n_atoms)
    atom_idx = np.asarray(atom_idx, dtype=int)
    dofs = [(int(i), c) for i in atom_idx for c in range(3)]
    D = len(dofs)
    n_points = 1 + 2 * D + D * (D - 1)

    if verbose:
        print(f"FD dipole derivatives: {len(frames)} frames × {n_points} "
              f"dipole evaluations ({D} DOF, h={h} Å) = "
              f"{len(frames) * n_points} total")

    P = _fd_derivatives(model, frames, dofs, h, builder, batch_size, verbose,
                        **traj_kw)

    mu_all, dmu_all, d2_all = [], [], []
    for fi in range(len(frames)):
        mu, dmu, d2 = _fd_assemble(P[fi], D, h)
        mu_all.append(mu)
        dmu_all.append(dmu.reshape(len(atom_idx), 3, 3))
        d2_all.append(d2.reshape(len(atom_idx), 3, len(atom_idx), 3, 3))

    return (np.stack(mu_all), np.stack(dmu_all), np.stack(d2_all))
