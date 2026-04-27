from __future__ import annotations

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from data import cell_to_box

if TYPE_CHECKING:
    from ase import Atoms
    from TNEP import TNEP


def _pad_gradients_for_structure(
    gradients: list[tf.Tensor],
    grad_index: list[list[int]],
    N: int,
    dim_q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad per-atom gradients and neighbour indices to uniform shape for a single structure.

    Args:
        gradients  : list of N tensors each [M_i, 3, dim_q]
        grad_index : list of N lists each [M_i] ints
        N          : number of atoms
        dim_q      : descriptor dimension

    Returns:
        grad_padded : [N, max_nbrs, 3, dim_q] float32
        gidx_padded : [N, max_nbrs] int32
        nbr_mask    : [N, max_nbrs] float32
    """
    max_nbrs = max(gradients[i].shape[0] for i in range(N))
    grad_padded = np.zeros((N, max_nbrs, 3, dim_q), dtype=np.float32)
    gidx_padded = np.zeros((N, max_nbrs), dtype=np.int32)
    nbr_mask = np.zeros((N, max_nbrs), dtype=np.float32)
    for i in range(N):
        n_nbrs = gradients[i].shape[0]
        grad_padded[i, :n_nbrs] = gradients[i].numpy()
        gidx_padded[i, :n_nbrs] = grad_index[i]
        nbr_mask[i, :n_nbrs] = 1.0
    return grad_padded, gidx_padded, nbr_mask


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


def compute_ir_spectrum(dipoles: np.ndarray, dt_fs: float = 1.0, window: str | None = 'hann',
                         max_freq_cm: float = 4000.0, acf_ratio: float = 0.1,
                         smooth_k: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute IR absorption spectrum from a dipole moment trajectory.

    Follows the reference GPUMD/NEP notebook (Dr. Nan Xu) and
    Xu et al., J. Chem. Theory Comput., 2024, 20, 3273–3284:
        1. Subtract mean dipole to remove DC component
        2. Compute dipole autocorrelation function C(τ) = <μ(0)·μ(τ)>
        3. Truncate ACF to first acf_ratio of the trajectory (default 10%)
        4. Apply Hann window and Kronecker doubling factor
        5. Cosine transform to obtain line shape M(ω) (guaranteed non-negative)
        6. IR absorption: σ(ω) ∝ ω² · M(ω)  (Eq. 1)
        7. Smooth with a moving average of width smooth_k

    Args:
        dipoles      : [T, 3] ndarray — dipole trajectory (e/Å or Debye, one per frame)
        dt_fs        : float — timestep between frames in femtoseconds
        window       : str or None — window function ('hann', 'blackman', or None)
        max_freq_cm  : float — maximum frequency to return in cm⁻¹
        acf_ratio    : float — fraction of trajectory to use as max ACF lag (default 0.1)
        smooth_k     : int — moving-average smoothing width (default 10, 0 to disable)

    Returns:
        freq_cm   : [N] ndarray — frequencies in cm⁻¹
        intensity : [N] ndarray — IR absorption intensity (arb. units, ω²·M(ω))
        power     : [N] ndarray — power spectrum (arb. units, M(ω) before ω² weighting)
        acf       : [Nmax] ndarray — dipole autocorrelation function
    """
    # Subtract mean to remove DC component before computing ACF
    dipoles = dipoles - dipoles.mean(axis=0)
    acf_full = compute_dipole_acf(dipoles)

    # Truncate ACF — only the first acf_ratio fraction has good statistics
    Nmax = len(acf_full) // int(1.0 / acf_ratio)
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

    # Cosine transform (DCT) to obtain line shape — guaranteed real & non-negative
    # M(k) = Σ_t acf(t) · cos(2πkt / (2Nmax-1))
    t_idx = np.arange(Nmax)
    M_omega = np.zeros(Nmax)
    for k in range(Nmax):
        M_omega[k] = np.sum(acf_prepared * np.cos(2.0 * np.pi * k * t_idx / (2 * Nmax - 1)))

    # Frequency axis: convert from DCT index to cm⁻¹
    # k-th bin corresponds to frequency k / ((2*Nmax-1) * dt_fs) in 1/fs
    c_cm_per_fs = 2.99792458e-5  # speed of light in cm/fs
    freq_cm = np.arange(Nmax) / ((2 * Nmax - 1) * dt_fs * c_cm_per_fs)

    # IR absorption: σ(ω) ∝ ω² · M(ω)  (Xu et al. JCTC 2024, Eq. 1)
    intensity = freq_cm**2 * M_omega
    power = M_omega.copy()

    # Truncate to requested frequency range
    mask = freq_cm <= max_freq_cm
    freq_cm = freq_cm[mask]
    intensity = intensity[mask]
    power = power[mask]

    # Smooth with moving average
    if smooth_k > 1 and len(intensity) > smooth_k:
        kernel = np.ones(smooth_k) / smooth_k
        intensity = np.convolve(intensity, kernel, mode='valid')
        power = np.convolve(power, kernel, mode='valid')
        # mode='valid' output[i] averages input[i:i+smooth_k], centred at i+(smooth_k-1)/2.
        # freq_cm is linear, so compute exact fractional-centre frequencies directly.
        d_freq = freq_cm[1] - freq_cm[0]
        freq_start = freq_cm[0] + (smooth_k - 1) / 2.0 * d_freq
        freq_cm = freq_start + np.arange(len(intensity)) * d_freq

    # Normalise both to peak = 1
    peak = np.max(np.abs(intensity))
    if peak > 0:
        intensity /= peak
    peak = np.max(np.abs(power))
    if peak > 0:
        power /= peak

    return freq_cm, intensity, power, acf


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
    ax.set_ylim(1, 0)  # intensity descends from 1 at top to 0 at bottom
    plt.tight_layout()
    _finish_fig(fig, cfg, "ir_spectrum", save_plots, show_plots)


def plot_power_spectrum(freq_cm: np.ndarray, power: np.ndarray, cfg: TNEPconfig,
                        save_plots: str | None = "plots", show_plots: bool = False,
                        title: str = "Power Spectrum") -> None:
    """Plot the dipole power spectrum M(ω) (before ω² weighting).

    Args:
        freq_cm    : [N] ndarray — frequencies in cm⁻¹
        power      : [N] ndarray — normalised power spectrum M(ω)
        cfg        : TNEPconfig — used for filename generation
        save_plots : str or None — directory to save into
        show_plots : bool — True to display interactively
        title      : str — plot title
    """
    from plotting import _finish_fig
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_cm, power, color='black', linewidth=0.8)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("Power (arb. units)")
    ax.set_title(title)
    ax.set_xlim(freq_cm[0], freq_cm[-1])
    ax.invert_xaxis()
    plt.tight_layout()
    _finish_fig(fig, cfg, "power_spectrum", save_plots, show_plots)


def predict_dipole_trajectory(model: TNEP, trajectory: list[Atoms], dataset_types_int: list[np.ndarray]) -> np.ndarray:
    """Predict dipole moments for each frame in an MD trajectory.

    Args:
        model             : trained TNEP model (target_mode = 1, config via model.cfg)
        trajectory        : list of ase.Atoms — MD frames (ordered in time)
        dataset_types_int : list of [N_i] int arrays — type indices per frame

    Returns:
        dipoles : [T, 3] ndarray — predicted dipole moment per frame
    """
    builder = DescriptorBuilder(model.cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(trajectory)
    actual_dim_q = descriptors[0].shape[-1]

    dipoles = []
    for s in range(len(trajectory)):
        struct = trajectory[s]
        N = len(struct)
        pos = tf.convert_to_tensor(struct.positions, dtype=tf.float32)
        Z = tf.convert_to_tensor(dataset_types_int[s], dtype=tf.int32)
        box = tf.convert_to_tensor(cell_to_box(struct), dtype=tf.float32)
        desc = descriptors[s]
        atom_mask = tf.ones([N], dtype=tf.float32)

        grad_padded, gidx_padded, nbr_mask = _pad_gradients_for_structure(
            gradients[s], grad_index[s], N, actual_dim_q)

        mu = model.predict(
            desc, tf.convert_to_tensor(grad_padded), tf.convert_to_tensor(gidx_padded),
            pos, Z, box, atom_mask, tf.convert_to_tensor(nbr_mask, dtype=tf.float32))
        dipoles.append(mu.numpy())

    return np.array(dipoles)


def predict_polarizability_trajectory(model: TNEP, trajectory: list[Atoms], dataset_types_int: list[np.ndarray]) -> np.ndarray:
    """Predict polarizability tensors for each frame in an MD trajectory.

    Args:
        model             : trained TNEP model (target_mode = 2, config via model.cfg)
        trajectory        : list of ase.Atoms — MD frames (ordered in time)
        dataset_types_int : list of [N_i] int arrays — type indices per frame

    Returns:
        pols : [T, 6] ndarray — predicted polarizability [xx, yy, zz, xy, yz, zx] per frame
    """
    builder = DescriptorBuilder(model.cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(trajectory)
    actual_dim_q = descriptors[0].shape[-1]

    pols = []
    for s in range(len(trajectory)):
        struct = trajectory[s]
        N = len(struct)
        pos = tf.convert_to_tensor(struct.positions, dtype=tf.float32)
        Z = tf.convert_to_tensor(dataset_types_int[s], dtype=tf.int32)
        box = tf.convert_to_tensor(cell_to_box(struct), dtype=tf.float32)
        desc = descriptors[s]
        atom_mask = tf.ones([N], dtype=tf.float32)

        grad_padded, gidx_padded, nbr_mask = _pad_gradients_for_structure(
            gradients[s], grad_index[s], N, actual_dim_q)

        pol = model.predict(
            desc, tf.convert_to_tensor(grad_padded), tf.convert_to_tensor(gidx_padded),
            pos, Z, box, atom_mask, tf.convert_to_tensor(nbr_mask, dtype=tf.float32))
        pols.append(pol.numpy())

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
