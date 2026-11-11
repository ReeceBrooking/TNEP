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
from descriptor_layout import descriptor_input_width
from data import cell_to_box

if TYPE_CHECKING:
    from ase import Atoms
    from TNEP import TNEP


def compute_dipole_acf(dipoles: np.ndarray) -> np.ndarray:
    """Dipole ACF ⟨μ(0)·μ(τ)⟩ of dipoles [T, 3] → [T] (Xu et al. 2024, Eq. 9).

    Summed over x,y,z; biased estimator (each lag / T, as GPUMD) so the noisy
    tail decays instead of being amplified. Not normalised to acf[0]=1.
    """
    return _acf_sum(dipoles)


def _acf_sum(x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Σ_k w_k⟨x_k(0)·x_k(τ)⟩ over the columns of x [T, k] via Wiener-Khinchin
    (FFT, O(T log T)); biased estimator (each lag divided by T, as GPUMD/Xu).
    Shared by the dipole ACF (w = 1) and the Raman ACFs (Tr[β(0)β(τ)] weights)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    # Zero-pad to ≥ 2T-1 (linear, not circular, correlation) at a fast FFT length.
    try:
        from scipy.fft import next_fast_len as _nfl
        n_fft = int(_nfl(2 * T - 1))
    except ImportError:
        n_fft = 2 * T
    fd = np.fft.rfft(x, n=n_fft, axis=0)
    acf = np.fft.irfft(np.real(fd * np.conj(fd)), n=n_fft, axis=0)[:T]   # [T, k]
    w = np.ones(x.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
    return (acf @ w) / float(T)


def _acf_lineshape(acf_full: np.ndarray, dt_fs: float, window: str | None,
                   acf_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACF → line shape M(ω) ≥ 0 the GPUMD way: truncate to the first acf_ratio
    of the lags, window (Hann/Blackman/None), Kronecker doubling, cosine
    transform. Returns (freq_cm, M, acf_truncated)."""
    if acf_ratio <= 0.0:
        raise ValueError(f"acf_ratio must be > 0, got {acf_ratio}")
    Nmax = max(1, int(len(acf_full) * acf_ratio))
    acf = acf_full[:Nmax]

    # Kronecker doubling: ×2 except lag 0.
    kronecker = np.ones(Nmax) * 2.0
    kronecker[0] = 1.0

    if window == 'hann':
        w = (np.cos(np.pi * np.arange(Nmax) / Nmax) + 1.0) * 0.5
    elif window == 'blackman':
        w = np.blackman(Nmax)
    else:
        w = np.ones(Nmax)

    acf_prepared = acf * w * kronecker

    # Cosine transform = real part of the rfft zero-padded to 2Nmax-1.
    M_omega = np.fft.rfft(acf_prepared, n=2 * Nmax - 1).real

    # Frequency axis: DCT bin k → k/((2Nmax-1)·dt_fs) in 1/fs → cm⁻¹.
    c_cm_per_fs = 2.99792458e-5  # speed of light in cm/fs
    freq_cm = np.arange(Nmax) / ((2 * Nmax - 1) * dt_fs * c_cm_per_fs)
    return freq_cm, M_omega, acf


def _smooth(freq_cm: np.ndarray, ys: list[np.ndarray], smooth_k: int,
            smooth_kind: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Smooth every spectrum in ys. "gaussian": FWHM smooth_k bins (σ =
    smooth_k/2.355), preserves freq_cm. "box": moving average of width
    smooth_k bins; mode='valid' shortens freq_cm. smooth_k ≤ 1 = no-op."""
    if not (smooth_k > 1 and len(ys[0]) > smooth_k):
        return freq_cm, ys
    sk = str(smooth_kind).lower()
    if sk == "gaussian":
        try:
            from scipy.ndimage import gaussian_filter1d
        except ImportError:                              # graceful fallback
            gaussian_filter1d = None
        if gaussian_filter1d is not None:
            sigma_bins = smooth_k / 2.355                # FWHM → σ
            return freq_cm, [gaussian_filter1d(y, sigma=sigma_bins, mode="nearest") for y in ys]
        sk = "box"                                       # fall back below
    if sk == "box":
        kernel = np.ones(smooth_k) / smooth_k
        ys = [np.convolve(y, kernel, mode='valid') for y in ys]
        # mode='valid' output[i] averages input[i:i+smooth_k], centred at i+(smooth_k-1)/2.
        d_freq = freq_cm[1] - freq_cm[0]
        freq_start = freq_cm[0] + (smooth_k - 1) / 2.0 * d_freq
        return freq_start + np.arange(len(ys[0])) * d_freq, ys
    raise ValueError(f"smooth_kind must be 'gaussian' or 'box', got {smooth_kind!r}")


def _quantum_prefactor(freq_cm: np.ndarray, kind: str, temperature: float, raman: bool = False) -> np.ndarray:
    """Weight turning the classical line shape L_cl(ω) into the observable, x = ℏω/kT.
    The exact prefactors act on the quantum L_q: IR ∝ ω·(1 − e^(−x))·L_q (Xu et al.
    2024 Eq. 8); reduced Raman R(ω) = ω·L_q (Brooker et al. 1988/89). With the
    harmonic QCF L_q = x/(1 − e^(−x))·L_cl (Ramírez et al. 2004, standard for water):

        "harmonic"  : IR ∝ ω²·L_cl;  Raman R ∝ ω²/(1 − e^(−x))·L_cl
        "classical" : ω²·L_cl for both — Xu Eq. 9, the footing of their Fig. 4;
                      = "harmonic" for IR, its x ≫ 1 limit for Raman (10 % above 500 cm⁻¹)
        "bose"      : ω·(1 − e^(−x))·L_cl — exact IR prefactor on the UNCORRECTED
                      ACF (old IR default); underweights the stretch ~14×
        "linear"    : ω·L_cl;   "none": L_cl itself (calorine get_raman_spectrum)."""
    qc = str(kind).lower()
    hbar_c_eV_cm = 1.23984e-4              # ℏc in eV·cm  (so ℏω [eV] = ℏc · ν̃ [cm⁻¹])
    x = hbar_c_eV_cm * freq_cm / max(8.617333e-5 * float(temperature), 1e-30)
    if qc == "harmonic":
        if not raman:
            return freq_cm ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            f = freq_cm ** 2 / (1.0 - np.exp(-x))
        f[x == 0] = 0.0
        return f
    if qc in ("classical", "quadratic"):
        return freq_cm ** 2
    if qc == "bose":
        return freq_cm * (1.0 - np.exp(-x))
    if qc == "linear":
        return freq_cm.astype(np.float64)
    if qc == "none":
        return np.ones_like(freq_cm)
    raise ValueError(f"quantum_correction must be 'harmonic', 'classical' (alias 'quadratic'), "
                     f"'bose', 'linear' or 'none', got {kind!r}")


def compute_ir_spectrum(dipoles: np.ndarray, dt_fs: float = 1.0, window: str | None = 'hann',
                         max_freq_cm: float = 4000.0, acf_ratio: float = 0.1,
                         smooth_k: int = 10,
                         smooth_kind: str = "gaussian",
                         temperature: float = 300.0,
                         quantum_correction: str = "harmonic",
                         power_dc_cutoff_cm: float = 100.0,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IR absorption spectrum from dipoles [T, 3] in e·Å (GPUMD / Xu et al. 2024).

    Pipeline: subtract mean → ACF → truncate to acf_ratio → window + Kronecker
    doubling → cosine transform M(ω) ≥ 0 → quantum weight → smooth → peak=1.
    Default "harmonic" weighting is ω²·M (Xu Eq. 9); see _quantum_prefactor.
    smooth_k: Gaussian FWHM or box width in bins (0 = off; "box" matches GPUMD
    but rings). power_dc_cutoff_cm: bins below it are excluded from the power
    normaliser (raw M(ω) has a huge DC peak; 0 keeps it).

    Returns:
        freq_cm [N] cm⁻¹, intensity [N], power [N] (M(ω) before ω weighting),
        acf [Nmax].
    """
    # Mean removed so the DC component stays out of the ACF.
    dipoles = dipoles - dipoles.mean(axis=0)
    freq_cm, M_omega, acf = _acf_lineshape(compute_dipole_acf(dipoles), dt_fs, window, acf_ratio)

    prefactor = _quantum_prefactor(freq_cm, quantum_correction, temperature)
    intensity = prefactor * M_omega
    power = M_omega.copy()

    mask = freq_cm <= max_freq_cm
    freq_cm = freq_cm[mask]
    intensity = intensity[mask]
    power = power[mask]

    freq_cm, (intensity, power) = _smooth(freq_cm, [intensity, power], smooth_k, smooth_kind)

    # Normalise both to peak = 1.
    peak = np.max(np.abs(intensity))
    if peak > 0:
        intensity /= peak

    # Keep the DC peak of M(ω) out of the power normaliser (0 = disable).
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
    """Plot stem "<traj>_<model>" (empty parts skipped; fallback "ir_spectrum")."""
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
    """One IR trace on a single axis (split_at_cm=None → ax_hi) or a broken-axis
    pair, each side renormalised to peak=1 so low-freq peaks don't crush the rest."""
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

    mask_lo = freq_cm <= split_at_cm
    mask_hi = freq_cm >  split_at_cm
    if not mask_lo.any() or not mask_hi.any():
        # Fall back to single-axis if the split lands outside the data
        _plot_one_ir_panel(None, ax_hi, freq_cm, y, None, ylabel, title)
        return

    f_lo, y_lo = freq_cm[mask_lo], y[mask_lo]
    f_hi, y_hi = freq_cm[mask_hi], y[mask_hi]

    if invert_y:
        # Renormalise in absorbance space, then re-apply T(A).
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
        # Broken-axis pair [hi | lo]; widths ∝ wavenumber range so cm⁻¹/pixel matches.
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


# T(A) convention, set by plot_ir_spectrum and read by _plot_one_ir_panel.
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
    """Absorbance and transmittance as separate figures,
    <stem>_absorbance[_split<N>].png / <stem>_transmittance[_split<N>].png.

    split_at_cm: broken axis with per-region peak-normalisation. transmittance_mode
    "beer_lambert": T = 10^(−scale·A) (scale 1 → ~10 % at the strongest peak);
    "linear": T = 1 − A (visual mirror only).
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

    # 2. Transmittance
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
    # compute_ir_spectrum knobs
    window: str | None = 'hann',
    max_freq_cm: float = 4000.0,
    acf_ratio: float = 0.1,
    smooth_k: int = 10,
    smooth_kind: str = "gaussian",
    temperature: float = 300.0,
    quantum_correction: str = "harmonic",
    power_dc_cutoff_cm: float = 100.0,
    # transmittance display
    transmittance_mode: str = "beer_lambert",
    transmittance_scale: float = 1.0,
    **ir_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a dipole trajectory (.npy/.txt, [T, 3]), compute and plot its IR +
    power spectra (absorbance, transmittance, power as separate files). Knobs
    forward to compute_ir_spectrum / plot_ir_spectrum; title defaults to the
    file basename. Returns (freq_cm, intensity, power)."""
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
    """Plot the dipole power spectrum M(ω) (no ω weighting). The DC peak dwarfs
    the vibrational features, so ν̃ < low_cm_cutoff is cropped (0 = full range)
    and the y-axis defaults to log (modes span 2-4 decades)."""
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


# ── Fused pack+predict @tf.function, cached per model (traced once); [None]
# dims in the signature keep size changes from retracing ──
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
    # Raw-width descriptors (SOAP dim, or S·K·k_max·2 under the learned basis).
    dim_q = descriptor_input_width(cfg)
    # Gradient values are zero-width under the learned basis (pair indices still
    # flow; V is built in-graph).
    grad_q = 0 if cfg.radial_basis_size is not None else dim_q
    sig = [
        tf.TensorSpec(shape=[None, dim_q],     dtype=tf.float32),  # soap_concat
        tf.TensorSpec(shape=[None, 3, grad_q], dtype=tf.float32),  # grad_concat
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

    # Keras Variables can't be @tf.function args: snapshot to tensors, capture by closure.
    def _to_tensor(v):
        if v is None:
            return None
        # Keras 3 Variables expose .value (a Tensor); fall back to convert.
        return tf.convert_to_tensor(v.value if hasattr(v, "value") else v)

    # Pre-absorb U_pairᵀ into W0 (predict_batch trusts the caller; mirrors TNEP.score).
    W0_eff_var = model._W0_eff(model.W0)
    b0_eff_var = model._b0_eff(model.b0, model.W0)
    W0_t  = _to_tensor(W0_eff_var)
    b0_t  = _to_tensor(b0_eff_var)
    W1_t  = _to_tensor(model.W1)
    b1_t  = _to_tensor(model.b1)

    # Not jit_compile: _calc_forces_coo has shape-dependent stacks XLA can't lower.
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
        )
        # TOTAL dipole regardless of cfg.scale_targets — do not multiply by num_atoms.
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
    """Concatenate per-frame TF tensors into the _get_fused_predict input tuple.
    Destructive: clears frame_results once folded in (halves peak VRAM)."""
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
    if S:
        grad_concat = tf.concat([r[1] for r in frame_results], axis=0)
        pa_concat   = tf.concat([r[2] for r in frame_results], axis=0)
        pg_concat   = tf.concat([r[3] for r in frame_results], axis=0)
    else:
        # Width only matters for the trace, which never sees an empty batch.
        grad_concat = tf.zeros((0, 3, dim_q), dtype=tf.float32)
        pa_concat   = tf.zeros((0,), dtype=tf.int32)
        pg_concat   = tf.zeros((0,), dtype=tf.int32)
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
    """Pack flat per-frame COO arrays (numpy) into a stacked batch for predict_batch.
    frame_results[s] = (descriptors[N,Q], grad_values[P_s,3,Q], pair_atom[P_s], pair_gidx[P_s]).
    Returns the data.pad_and_stack() layout: descriptors [B,A,Q], grad_values [P,3,Q],
    pair_atom/gidx/struct [P], positions [B,A,3], Z_int [B,A], boxes [B,3,3],
    atom_mask [B,A], num_atoms [B]."""
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
    """Predict dipole/polarizability for one batch of frames (build → pack →
    predict); caller drives the outer loop. Returns [B, 3] (mode 1) or [B, 6] (mode 2).

    pin_to_cpu: batch tensors on CPU for trajectories too large for VRAM.
    descriptor_batch_frames: frames per builder call (TF GPU only; None = auto-size
    under descriptor_memory_budget_bytes, default 6 GiB). descriptor_precision:
    "float64" (mirrors quippy) or "float32" (~2× throughput, ~½ VRAM). quippy
    ignores all three.
    """
    cfg = model.cfg
    if cfg.nep_descriptor != "off":
        raise NotImplementedError(
            "NEP descriptors: trajectory inference not yet supported")
    # Lazy import: quippy-only callers never load the TF GPU builder.
    try:
        from DescriptorBuilderGPU_tf import DescriptorBuilderGPUTF
        prefers_tf = isinstance(builder, DescriptorBuilderGPUTF)
    except ImportError:
        prefers_tf = False

    if prefers_tf:
        # GPU builder → fused pack+predict graph. Precision/tile overrides go
        # first; build_descriptors_flat rebuilds its compute fns lazily.
        if descriptor_precision is not None:
            builder.set_precision(descriptor_precision)
        if descriptor_pair_tile_size is not None:
            builder.set_pair_tile_size(int(descriptor_pair_tile_size))
        # XLA only for fp32 + pair-tiling: fp64+XLA spills registers (~10× slower);
        # without tiling the shape varies per call and XLA recompiles every batch
        # (~5-7 s each). Tiling pads pairs to fixed buckets, each compiled once.
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
            frame_results, batch_frames, batch_types, descriptor_input_width(cfg),
            pin_to_cpu=pin_to_cpu,
        )
        del frame_results
        fused_predict = _get_fused_predict(model)
        preds = fused_predict(*fused_inputs)
        # Free the concats before the .numpy() sync so the next SOAP build has full VRAM.
        del fused_inputs
        out = preds.numpy()
        del preds
    else:
        # NumPy pack + eager predict_batch (quippy / builders returning numpy).
        frame_results = builder.build_descriptors_flat(
            batch_frames,
            batch_frames=descriptor_batch_frames,
            memory_budget_bytes=descriptor_memory_budget_bytes,
        )
        batch = _pack_traj_batch_from_flat(frame_results, batch_frames, batch_types,
                                           descriptor_input_width(cfg),
                                           pin_to_cpu=pin_to_cpu)
        del frame_results

        # Same mixing absorption as the fused path so both backends agree.
        W0_pred = model._W0_eff(model.W0)
        b0_pred = model._b0_eff(model.b0, model.W0)

        preds = model.predict_batch(
            batch["descriptors"], batch["grad_values"],
            batch["pair_atom"], batch["pair_gidx"], batch["pair_struct"],
            batch["positions"], batch["Z_int"], batch["boxes"],
            batch["atom_mask"],
            W0_pred, b0_pred, model.W1, model.b1,
        )
        # TOTAL dipole regardless of cfg.scale_targets; no rescaling.
        out = preds.numpy()
        del batch, preds
    return out


def compute_raman_acfs(polarizabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic and anisotropic ACFs of polarizabilities [T, 6] (Voigt
    [xx, yy, zz, xy, yz, zx]); Xu et al. 2024 Eq. 12 / calorine. α = γI + β,
    γ = Tr(α)/3: C_iso = ⟨γ(0)γ(τ)⟩, C_aniso = ⟨Tr[β(0)β(τ)]⟩ = Σ_i⟨β_ii²⟩ + 2Σ_{i<j}⟨α_ij²⟩.
    Mean (Rayleigh) removed first; biased FFT estimator as the dipole ACF."""
    a = np.asarray(polarizabilities, dtype=np.float64)
    a = a - a.mean(axis=0)
    gamma = a[:, :3].mean(axis=1)
    beta = np.concatenate([a[:, :3] - gamma[:, None], a[:, 3:]], axis=1)
    return _acf_sum(gamma[:, None]), _acf_sum(beta, weights=[1, 1, 1, 2, 2, 2])


def compute_raman_spectrum(polarizabilities: np.ndarray, dt_fs: float = 1.0, window: str | None = 'hann',
                           max_freq_cm: float = 4000.0, acf_ratio: float = 0.1,
                           smooth_k: int = 10, smooth_kind: str = "gaussian",
                           temperature: float = 300.0, quantum_correction: str = "harmonic",
                           norm_cutoff_cm: float = 100.0,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raman spectrum from polarizabilities [T, 6] (Voigt [xx, yy, zz, xy, yz, zx]):
    the IR pipeline applied to C_iso and C_aniso, then combined (Placzek,
    Long, The Raman Effect, Tab. 5.2(a), with γ² = (3/2)·Tr[ββ]):

        I_VV(ω) ∝ 45·L_iso + 6·L_aniso     (polarised, Z(XX)Z̄)
        I_VH(ω) ∝ 4.5·L_aniso              (depolarised, Z(XY)Z̄)

    so ρ = I_VH/I_VV = 3/4 for a purely anisotropic band. quantum_correction:
    "harmonic" (default) is the reduced spectrum ω²/(1 − e^(−x))·L that experiment
    reports (Brooker et al. 1989); "classical" is ω²·L, the footing of Xu et al.
    2024 Fig. 4 (verified with their own TNEP χ model), equal above ~500 cm⁻¹;
    "none" is bare L(ω) (Eq. 12 / calorine), which keeps the Rayleigh-wing peak
    at ω → 0. norm_cutoff_cm: bins below it are excluded from the peak normaliser
    (the mean-free ACFs still carry slow diffusive weight). Other kwargs as
    compute_ir_spectrum.

    Returns:
        freq_cm, I_VV, I_VH (jointly normalised: max I_VV = 1 above
        norm_cutoff_cm), I_iso, I_aniso (jointly normalised: max = 1).
    """
    C_iso, C_aniso = compute_raman_acfs(polarizabilities)
    freq_cm, L_iso, _ = _acf_lineshape(C_iso, dt_fs, window, acf_ratio)
    _, L_aniso, _ = _acf_lineshape(C_aniso, dt_fs, window, acf_ratio)
    f = _quantum_prefactor(freq_cm, quantum_correction, temperature, raman=True)
    L_iso, L_aniso = L_iso * f, L_aniso * f
    mask = freq_cm <= max_freq_cm
    freq_cm, L_iso, L_aniso = freq_cm[mask], L_iso[mask], L_aniso[mask]
    freq_cm, (L_iso, L_aniso) = _smooth(freq_cm, [L_iso, L_aniso], smooth_k, smooth_kind)
    I_VV = 45.0 * L_iso + 6.0 * L_aniso
    I_VH = 4.5 * L_aniso
    vib = freq_cm >= float(norm_cutoff_cm) if norm_cutoff_cm > 0 else np.ones_like(freq_cm, dtype=bool)
    peak = float(np.max(np.abs(I_VV[vib]))) if vib.any() else 0.0
    if peak > 0:
        I_VV, I_VH = I_VV / peak, I_VH / peak
    peak = float(max(np.max(np.abs(L_iso[vib])), np.max(np.abs(L_aniso[vib])))) if vib.any() else 0.0
    if peak > 0:
        L_iso, L_aniso = L_iso / peak, L_aniso / peak
    return freq_cm, I_VV, I_VH, L_iso, L_aniso


def plot_raman_spectrum(freq_cm: np.ndarray, I_VV: np.ndarray, I_VH: np.ndarray,
                        I_iso: np.ndarray, I_aniso: np.ndarray, cfg: TNEPconfig,
                        save_plots: str | None = None, show_plots: bool = True,
                        title: str = "Raman Spectrum",
                        trajectory_path: str | None = None,
                        model_label: str | None = None,
                        low_cm_cutoff: float = 100.0,
                        split_at_cm: float | None = None) -> None:
    """Two figures, <stem>_raman_iso.png and <stem>_raman_aniso.png (Xu et al.
    2024 Fig. 4); I_VV / I_VH accepted but not plotted. Bins below low_cm_cutoff
    are cropped (the mean-free ACFs keep a relaxational peak at ω ≈ 0; use the
    same value as norm_cutoff_cm). split_at_cm: broken-axis pair as the IR
    plots, each region peak-normalised jointly; filenames get _split<N>."""
    from plotting import _finish_fig
    stem = _ir_plot_basename(trajectory_path, model_label)
    keep = freq_cm >= float(low_cm_cutoff)
    freq_cm, I_VV, I_VH, I_iso, I_aniso = (x[keep] for x in (freq_cm, I_VV, I_VH, I_iso, I_aniso))
    split_tag = f"_split{int(split_at_cm)}" if split_at_cm is not None else ""
    panels = [("raman_iso", [(I_iso, "isotropic  ⟨γ(0)γ(t)⟩", "black", 0.9)]),
              ("raman_aniso", [(I_aniso, "anisotropic  ⟨Tr β(0)β(t)⟩", "black", 0.9)])]
    for name, lines in panels:
        if split_at_cm is None or not ((freq_cm <= split_at_cm).any() and (freq_cm > split_at_cm).any()):
            fig, ax = plt.subplots(figsize=(12, 4.5)); axes, regions = [ax], [np.ones_like(freq_cm, dtype=bool)]
        else:
            lo = freq_cm <= float(split_at_cm)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                                     gridspec_kw={"width_ratios": [max(split_at_cm - freq_cm[0], 1.0),
                                                                   max(freq_cm[-1] - split_at_cm, 1.0)], "wspace": 0.06})
            regions = [lo, ~lo]
        peaks = [max(float(np.max(y[m])) for y, *_ in lines) or 1.0 for m in regions]   # joint per-region normalisers
        for k, (ax, m) in enumerate(zip(axes, regions)):
            peak = peaks[k]
            for y, lab, col, alpha in lines:
                ax.plot(freq_cm[m], y[m] / peak, color=col, linewidth=0.8, alpha=alpha, label=lab)
            ax.set_xlim(freq_cm[m][0], freq_cm[m][-1]); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
            if len(axes) == 2:                                               # broken-axis marks, as the IR plots
                d, kw = .015, dict(color='k', clip_on=False)
                if k == 0:
                    ax.spines['right'].set_visible(False)
                    ax.plot((1 - d, 1 + d), (-d, +d), transform=ax.transAxes, **kw); ax.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax.transAxes, **kw)
                else:
                    ax.spines['left'].set_visible(False); ax.tick_params(left=False, labelleft=False)
                    ax.plot((-d, +d), (-d, +d), transform=ax.transAxes, **kw); ax.plot((-d, +d), (1 - d, 1 + d), transform=ax.transAxes, **kw)
                    # scale change across the break, as Xu et al. Fig. 4: right peak / left peak
                    ax.text(0.05, 0.55, f"×{peaks[1] / peaks[0]:.3g}", transform=ax.transAxes, fontsize=11)
        axes[0].set_ylabel("Raman intensity (arb. units" + (", per-region peak-normalised)" if len(axes) == 2 else ")"))
        axes[-1].legend()
        fig.suptitle(title + (f"  (split at {int(split_at_cm)} cm⁻¹)" if len(axes) == 2 else ""))
        fig.supxlabel("Raman shift (cm⁻¹)") if len(axes) == 2 else axes[0].set_xlabel("Raman shift (cm⁻¹)")
        plt.tight_layout()
        _finish_fig(fig, cfg, f"{stem}_{name}{split_tag}", save_plots, show_plots)


def raman_spectrum_from_file(pol_path: str, dt_fs: float = 1.0, save_dir: str | None = None,
                             show: bool = True, title: str | None = None,
                             model_label: str | None = None, cfg: TNEPconfig | None = None,
                             split_at_cm: float | None = None,
                             **raman_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a saved polarizability trajectory (.npy or .txt, shape [T, 6] Voigt
    [xx, yy, zz, xy, yz, zx]) and plot its Raman spectra — the Raman twin of
    ir_spectrum_from_file. split_at_cm: broken-axis plots (see plot_raman_spectrum).
    **raman_kwargs go to compute_raman_spectrum.
    Returns (freq_cm, I_VV, I_VH, I_iso, I_aniso)."""
    pols = np.load(pol_path) if pol_path.lower().endswith(".npy") else np.loadtxt(pol_path)
    if pols.ndim != 2 or pols.shape[1] != 6:
        raise ValueError(f"expected polarizabilities of shape [T, 6], got {pols.shape}")
    out = compute_raman_spectrum(pols, dt_fs=dt_fs, **raman_kwargs)
    plot_raman_spectrum(*out, cfg or TNEPconfig(), save_plots=save_dir, show_plots=show,
                        title=title or f"Raman spectrum — {os.path.basename(pol_path)}",
                        trajectory_path=pol_path, model_label=model_label, split_at_cm=split_at_cm)
    return out
