from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from data import collate_flat

if TYPE_CHECKING:
    from ase import Atoms
    from TNEP import TNEP


def compute_dipole_acf(dipoles: np.ndarray) -> np.ndarray:
    """Compute the dipole autocorrelation function via the Wiener-Khinchin theorem.

    Uses FFT for O(N log N) efficiency instead of direct O(N²) summation.
    Averages over x, y, z components (isotropic).

    Ref: Xu et al., J. Chem. Theory Comput., 2024, 20, 3273–3284, Eq. 9

    Args:
        dipoles : [T, 3] ndarray — dipole moment trajectory (one per MD frame)

    Returns:
        acf : [T] ndarray — normalised dipole autocorrelation function C(τ)/C(0)
    """
    T = dipoles.shape[0]
    # Zero-pad to avoid circular correlation artefacts
    n_fft = 2 * T
    acf = np.zeros(T)
    for dim in range(3):
        d = dipoles[:, dim]
        # Wiener-Khinchin: ACF = IFFT(|FFT(d)|²)
        fd = np.fft.rfft(d, n=n_fft)
        power = np.real(fd * np.conj(fd))
        full_acf = np.fft.irfft(power, n=n_fft)[:T]
        acf += full_acf
    # Normalise by number of overlapping pairs at each lag
    counts = np.arange(T, 0, -1, dtype=np.float64)
    acf /= counts
    # Average over 3 spatial dimensions
    acf /= 3.0
    return acf


def compute_ir_spectrum(dipoles: np.ndarray, dt_fs: float = 1.0, window: str | None = 'hann', max_freq_cm: float = 4000.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute IR absorption spectrum from a dipole moment trajectory.

    Follows Xu et al., J. Chem. Theory Comput., 2024, 20, 3273–3284:
        1. Compute dipole autocorrelation function C(τ) = <μ(0)·μ(τ)>
        2. Apply optional window function to reduce spectral leakage
        3. Fourier transform: M(ω) = FT[C(τ)]
        4. IR absorption: σ(ω) ∝ ω · M(ω)  (classical approximation, Eq. 8)

    Args:
        dipoles      : [T, 3] ndarray — dipole trajectory (e/Å or Debye, one per frame)
        dt_fs        : float — timestep between frames in femtoseconds
        window       : str or None — window function ('hann', 'blackman', or None)
        max_freq_cm  : float — maximum frequency to return in cm⁻¹

    Returns:
        freq_cm : [N] ndarray — frequencies in cm⁻¹
        intensity : [N] ndarray — IR absorption intensity (arb. units)
        acf : [T] ndarray — dipole autocorrelation function
    """
    acf = compute_dipole_acf(dipoles)
    T = len(acf)

    # Apply window to reduce spectral leakage
    if window == 'hann':
        w = np.hanning(T)
    elif window == 'blackman':
        w = np.blackman(T)
    else:
        w = np.ones(T)
    acf_windowed = acf * w

    # Fourier transform of ACF -> line shape M(ω)
    spectrum = np.fft.rfft(acf_windowed)
    M_omega = np.real(spectrum)

    # Frequency axis: convert from 1/fs to cm⁻¹
    # ν (1/fs) -> ω (cm⁻¹): multiply by 1e15 / (c * 100)
    # where c = 2.998e8 m/s, so c_cm = 2.998e10 cm/s
    freq_per_fs = np.fft.rfftfreq(T, d=dt_fs)  # in 1/fs
    c_cm_per_fs = 2.99792458e-5  # speed of light in cm/fs
    freq_cm = freq_per_fs / c_cm_per_fs  # convert to cm⁻¹

    # IR absorption: σ(ω) ∝ ω · M(ω)  (classical, Eq. 8 simplified)
    intensity = freq_cm * M_omega

    # Truncate to requested frequency range
    mask = freq_cm <= max_freq_cm
    freq_cm = freq_cm[mask]
    intensity = intensity[mask]

    # Normalise to peak = 1
    peak = np.max(np.abs(intensity))
    if peak > 0:
        intensity /= peak

    return freq_cm, intensity, acf


def plot_ir_spectrum(freq_cm: np.ndarray, intensity: np.ndarray, cfg: TNEPconfig,
                     save_plots: str | None = None, show_plots: bool = True,
                     title: str = "IR Spectrum") -> None:
    """Plot IR absorption spectrum.

    Args:
        freq_cm    : [N] ndarray — frequencies in cm⁻¹
        intensity  : [N] ndarray — normalised IR intensity
        cfg        : TNEPconfig — used for filename generation
        save_plots : str or None — directory to save into
        show_plots : bool — True to display interactively
        title      : str — plot title
    """
    from plotting import _finish_fig
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_cm, intensity, color='black', linewidth=0.8)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("IR Intensity (arb. units)")
    ax.set_title(title)
    ax.set_xlim(freq_cm[0], freq_cm[-1])
    ax.invert_xaxis()  # IR convention: high to low wavenumber
    plt.tight_layout()
    _finish_fig(fig, cfg, "ir_spectrum", save_plots, show_plots)


def predict_dipole_trajectory(model: TNEP, trajectory: list[Atoms],
                              dataset_types_int: list[np.ndarray],
                              cfg: TNEPconfig) -> np.ndarray:
    """Predict dipole moments for each frame in an MD trajectory."""
    from data import assemble_data_dict, collate_flat
    builder = DescriptorBuilder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(trajectory)

    dipoles = []
    for s in range(len(trajectory)):
        # Build single-structure data dict and collate
        single_data = assemble_data_dict(
            [trajectory[s]], [dataset_types_int[s]],
            [descriptors[s]], [gradients[s]], [grad_index[s]], cfg)
        batch = collate_flat(single_data)
        # Move to model device
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            mu = model.predict_flat(
                batch, model.W0, model.b0, model.W1, model.b1,
                getattr(model, 'W0_pol', None),
                getattr(model, 'b0_pol', None),
                getattr(model, 'W1_pol', None),
                getattr(model, 'b1_pol', None),
            )
        dipoles.append(mu.cpu().numpy().squeeze(0))

    return np.array(dipoles)


def predict_polarizability_trajectory(model: TNEP, trajectory: list[Atoms],
                                      dataset_types_int: list[np.ndarray],
                                      cfg: TNEPconfig) -> np.ndarray:
    """Predict polarizability tensors for each frame in an MD trajectory."""
    from data import assemble_data_dict, collate_flat
    builder = DescriptorBuilder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(trajectory)

    pols = []
    for s in range(len(trajectory)):
        single_data = assemble_data_dict(
            [trajectory[s]], [dataset_types_int[s]],
            [descriptors[s]], [gradients[s]], [grad_index[s]], cfg)
        batch = collate_flat(single_data)
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        with torch.no_grad():
            pol = model.predict_flat(
                batch, model.W0, model.b0, model.W1, model.b1,
                getattr(model, 'W0_pol', None),
                getattr(model, 'b0_pol', None),
                getattr(model, 'W1_pol', None),
                getattr(model, 'b1_pol', None),
            )
        pols.append(pol.cpu().numpy().squeeze(0))

    return np.array(pols)


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


def _cross_acf_fft(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cross-correlation <a(0)·b(t)> via FFT.

    Args:
        a, b : [T] ndarrays

    Returns:
        ccf : [T] ndarray — unnormalised cross-correlation
    """
    T = len(a)
    n_fft = 2 * T
    fa = np.fft.rfft(a, n=n_fft)
    fb = np.fft.rfft(b, n=n_fft)
    power = np.real(fa * np.conj(fb))
    ccf = np.fft.irfft(power, n=n_fft)[:T]
    counts = np.arange(T, 0, -1, dtype=np.float64)
    ccf /= counts
    return ccf


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
    ax.legend()
    plt.tight_layout()
    _finish_fig(fig, cfg, "raman_spectrum", save_plots, show_plots)
