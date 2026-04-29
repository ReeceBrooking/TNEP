from __future__ import annotations

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import TYPE_CHECKING
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from data import cell_to_box

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

    # Cosine transform: M(k) = Σ_t acf(t) · cos(2πkt / (2Nmax-1)) is the real
    # part of the rfft of acf_prepared zero-padded to length 2Nmax-1.
    # O(N log N) in C vs O(N²) in Python.
    M_omega = np.fft.rfft(acf_prepared, n=2 * Nmax - 1).real

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


def _pack_traj_batch_coo(
    frames: list,
    types_int_batch: list[np.ndarray],
    descriptors: list,
    gradients: list,
    grad_index: list,
    dim_q: int,
    pin_to_cpu: bool = True,
) -> dict:
    """Pack a batch of trajectory frames into COO format for predict_batch.

    Mirrors pad_and_stack() from data.py: same COO layout, same CPU pinning option,
    same del-numpy-after-tf.constant pattern to keep peak RAM at ~1× rather than 2×.

    Args:
        pin_to_cpu : True = place tensors on CPU (transferred to GPU implicitly during
                     predict_batch). Required for trajectories too large to fit in VRAM.

    Returns:
        dict with descriptors [B,A,Q], grad_values [P,3,Q], pair_atom/gidx/struct [P],
        positions [B,A,3], Z_int [B,A], boxes [B,3,3], atom_mask [B,A] as tf.Tensor
    """
    S = len(frames)
    atom_counts = [descriptors[s].shape[0] for s in range(S)]
    max_atoms = max(atom_counts)

    pair_counts = [
        sum(gradients[s][i].shape[0] for i in range(atom_counts[s]))
        for s in range(S)
    ]
    N_pairs = sum(pair_counts)

    grad_values_np = np.zeros((N_pairs, 3, dim_q), dtype=np.float32)
    pair_struct_np = np.zeros(N_pairs, dtype=np.int32)
    pair_atom_np   = np.zeros(N_pairs, dtype=np.int32)
    pair_gidx_np   = np.zeros(N_pairs, dtype=np.int32)

    desc_np      = np.zeros((S, max_atoms, dim_q), dtype=np.float32)
    pos_np       = np.zeros((S, max_atoms, 3),     dtype=np.float32)
    z_np         = np.zeros((S, max_atoms),         dtype=np.int32)
    box_np       = np.zeros((S, 3, 3),              dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms),         dtype=np.float32)
    num_atoms_np = np.array(atom_counts, dtype=np.int32)

    pair_offset = 0
    for s in range(S):
        N_s = atom_counts[s]
        desc_np[s, :N_s]      = descriptors[s].numpy()
        pos_np[s, :N_s]       = frames[s].positions.astype(np.float32)
        z_np[s, :N_s]         = types_int_batch[s]
        box_np[s]             = cell_to_box(frames[s])
        atom_mask_np[s, :N_s] = 1.0

        for i in range(N_s):
            n_nbrs = gradients[s][i].shape[0]
            k_end  = pair_offset + n_nbrs
            grad_values_np[pair_offset:k_end] = gradients[s][i].numpy()
            pair_struct_np[pair_offset:k_end] = s
            pair_atom_np[pair_offset:k_end]   = i
            pair_gidx_np[pair_offset:k_end]   = grad_index[s][i]
            pair_offset = k_end

    # Convert numpy → tf.Tensor under the chosen device, then drop the numpy
    # buffers immediately so peak RAM stays at ~1× the batch size.
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


def _pack_traj_batch_from_flat(
    frame_results: list,
    frames: list,
    types_int_batch: list[np.ndarray],
    dim_q: int,
    pin_to_cpu: bool = True,
) -> dict:
    """Pack flat per-frame COO arrays into a stacked batch.

    Each frame_results[s] = (descriptors[N,Q], grad_values[P_s,3,Q],
    pair_atom[P_s], pair_gidx[P_s]) all numpy. We concatenate the pair-level
    arrays in one shot and assemble a struct-index from the per-frame counts —
    no per-atom Python loop, no .numpy() round-trips.

    Returns the same dict layout as _pack_traj_batch_coo so predict_batch
    consumes it unchanged.
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


def predict_dipole_batch(
    model: TNEP,
    builder: DescriptorBuilder,
    batch_frames: list[Atoms],
    batch_types: list[np.ndarray],
    pin_to_cpu: bool = True,
) -> np.ndarray:
    """Run dipole prediction on one batch of frames.

    Build → pack → predict → return. Caller is responsible for the outer batch
    loop and for releasing batch_frames after the call.

    Args:
        model        : trained TNEP model (target_mode = 1)
        builder      : reusable DescriptorBuilder (constructed once per trajectory)
        batch_frames : list of ase.Atoms in this batch
        batch_types  : list of [N_i] int arrays — type indices per frame
        pin_to_cpu   : place batch tensors on CPU (transferred to GPU implicitly).
                       Required for trajectories too large to fit in VRAM.

    Returns:
        [B, 3] numpy array — total system dipole per frame in the batch
    """
    cfg = model.cfg
    frame_results = builder.build_descriptors_flat(batch_frames)
    batch = _pack_traj_batch_from_flat(frame_results, batch_frames, batch_types,
                                       cfg.dim_q, pin_to_cpu=pin_to_cpu)
    del frame_results

    preds = model.predict_batch(
        batch["descriptors"], batch["grad_values"],
        batch["pair_atom"], batch["pair_gidx"], batch["pair_struct"],
        batch["positions"], batch["Z_int"], batch["boxes"],
        batch["atom_mask"],
        model.W0, model.b0, model.W1, model.b1,
        None, None, None, None,
    )
    # Training scales dipole targets to per-atom (target / N), so the model
    # outputs per-atom dipole. Multiply by N here to recover the total system
    # dipole that spectroscopy expects.
    if cfg.scale_targets:
        num_atoms_col = tf.cast(batch["num_atoms"], tf.float32)[:, tf.newaxis]
        preds = preds * num_atoms_col
    out = preds.numpy()
    del batch, preds
    return out


def predict_polarizability_batch(
    model: TNEP,
    builder: DescriptorBuilder,
    batch_frames: list[Atoms],
    batch_types: list[np.ndarray],
    pin_to_cpu: bool = True,
) -> np.ndarray:
    """Run polarizability prediction on one batch of frames.

    Build → pack → predict → return. Caller is responsible for the outer batch
    loop and for releasing batch_frames after the call.

    Args:
        model        : trained TNEP model (target_mode = 2)
        builder      : reusable DescriptorBuilder (constructed once per trajectory)
        batch_frames : list of ase.Atoms in this batch
        batch_types  : list of [N_i] int arrays — type indices per frame
        pin_to_cpu   : place batch tensors on CPU (transferred to GPU implicitly).
                       Required for trajectories too large to fit in VRAM.

    Returns:
        [B, 6] numpy array — polarizability [xx, yy, zz, xy, yz, zx] per frame
    """
    cfg = model.cfg
    frame_results = builder.build_descriptors_flat(batch_frames)
    batch = _pack_traj_batch_from_flat(frame_results, batch_frames, batch_types,
                                       cfg.dim_q, pin_to_cpu=pin_to_cpu)
    del frame_results

    preds = model.predict_batch(
        batch["descriptors"], batch["grad_values"],
        batch["pair_atom"], batch["pair_gidx"], batch["pair_struct"],
        batch["positions"], batch["Z_int"], batch["boxes"],
        batch["atom_mask"],
        model.W0, model.b0, model.W1, model.b1,
        getattr(model, 'W0_pol', None),
        getattr(model, 'b0_pol', None),
        getattr(model, 'W1_pol', None),
        getattr(model, 'b1_pol', None),
    )
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
