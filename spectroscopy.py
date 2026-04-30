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


def ir_spectrum_from_file(
    dipole_path: str,
    dt_fs: float = 1.0,
    save_path: str | None = None,
    show: bool = True,
    title: str | None = None,
    **ir_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved dipole trajectory and plot its IR spectrum.

    Convenience wrapper around `np.load`/`np.loadtxt` + `compute_ir_spectrum`
    + a minimal matplotlib plot — no TNEPconfig required. Accepts either the
    binary .npy or the human-readable .txt file written by
    `process_trajectory`.

    Args:
        dipole_path : path to the dipole file (.npy or .txt). Shape [T, 3].
        dt_fs       : MD timestep in femtoseconds.
        save_path   : if given, save the plot here (e.g. "ir.png"). None = do
                      not save.
        show        : True to call plt.show() interactively.
        title       : optional plot title (defaults to the file's basename).
        **ir_kwargs : forwarded to `compute_ir_spectrum`
                      (window, max_freq_cm, acf_ratio, smooth_k).

    Returns:
        (freq_cm, intensity) — 1-D arrays. The autocorrelation and power
        spectrum returned by `compute_ir_spectrum` are computed but not
        returned here; call that function directly if you need them.
    """
    import os
    if dipole_path.lower().endswith(".npy"):
        dipoles = np.load(dipole_path)
    else:
        dipoles = np.loadtxt(dipole_path)
    if dipoles.ndim != 2 or dipoles.shape[1] != 3:
        raise ValueError(
            f"expected dipoles of shape [T, 3], got {dipoles.shape}"
        )
    freq_cm, intensity, _power, _acf = compute_ir_spectrum(
        dipoles, dt_fs=dt_fs, **ir_kwargs)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_cm, intensity, color="black", linewidth=0.8)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("IR Intensity (arb. units)")
    ax.set_title(title or f"IR spectrum — {os.path.basename(dipole_path)}")
    ax.set_xlim(freq_cm[0], freq_cm[-1])
    ax.invert_xaxis()
    ax.set_ylim(1, 0)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return freq_cm, intensity


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

    W0_t  = _to_tensor(model.W0)
    b0_t  = _to_tensor(model.b0)
    W1_t  = _to_tensor(model.W1)
    b1_t  = _to_tensor(model.b1)
    W0p_t = _to_tensor(getattr(model, "W0_pol", None))
    b0p_t = _to_tensor(getattr(model, "b0_pol", None))
    W1p_t = _to_tensor(getattr(model, "W1_pol", None))
    b1p_t = _to_tensor(getattr(model, "b1_pol", None))

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
    atom_counts = [int(r[0].shape[0]) for r in frame_results]
    max_atoms = max(atom_counts) if atom_counts else 0
    pair_counts = [int(r[1].shape[0]) for r in frame_results]
    pair_counts_arr = np.array(pair_counts, dtype=np.int32)
    N_pairs = int(pair_counts_arr.sum())

    if N_pairs > 0:
        # pair_atom / pair_gidx stay frame-local — predict_batch indexes them
        # together with pair_struct, mirroring _pack_traj_batch_from_flat.
        grad_values_t = tf.concat([r[1] for r in frame_results], axis=0)
        pair_atom_t   = tf.concat([r[2] for r in frame_results], axis=0)
        pair_gidx_t   = tf.concat([r[3] for r in frame_results], axis=0)
        pair_struct_t = tf.repeat(tf.range(S, dtype=tf.int32),
                                   tf.constant(pair_counts_arr, dtype=tf.int32))
        # Pair-level data has been concatenated; release per-frame grad/pair
        # tensors held in frame_results. Only the soap_concat below still
        # needs the per-frame soap slices.
    else:
        grad_values_t = tf.zeros((0, 3, dim_q), dtype=tf.float32)
        pair_atom_t = tf.zeros((0,), dtype=tf.int32)
        pair_gidx_t = tf.zeros((0,), dtype=tf.int32)
        pair_struct_t = tf.zeros((0,), dtype=tf.int32)

    # Descriptors: per-frame [N_s, Q] → padded [B, A_max, Q] via RaggedTensor
    # (one concat + one to_tensor, all on-device).
    if S:
        soap_concat = tf.concat([r[0] for r in frame_results], axis=0)
        # Per-frame soap slices are now folded into soap_concat / desc_t —
        # release the caller's list so they GC.
        frame_results.clear()
        desc_ragged = tf.RaggedTensor.from_row_lengths(
            soap_concat, tf.constant(atom_counts, dtype=tf.int64))
        desc_t = desc_ragged.to_tensor(default_value=0.0,
                                        shape=(S, max_atoms, dim_q))
        del soap_concat, desc_ragged
    else:
        desc_t = tf.zeros((0, 0, dim_q), dtype=tf.float32)
        frame_results.clear()

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

    return {
        "descriptors": desc_t,
        "grad_values": grad_values_t,
        "pair_atom":   pair_atom_t,
        "pair_gidx":   pair_gidx_t,
        "pair_struct": pair_struct_t,
        "positions":   positions_t,
        "Z_int":       z_t,
        "boxes":       box_t,
        "atom_mask":   atom_mask_t,
        "num_atoms":   num_atoms_t,
    }


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
        # XLA-JIT only when pair-tiling is active. Without tiling, the per-
        # call shape varies with both n_atoms_chunk AND total pair count P
        # (which changes by handfuls between frames), which forces XLA to
        # recompile (ptxas spill warnings every batch, ~5-7 s/compile).
        # With pair-tiling enabled, pairs are padded to a multiple of
        # pair_tile_size, so the tf.while_loop body has a deterministic
        # per-iteration shape — XLA compiles each (n_atoms_chunk, n_tiles)
        # combination ONCE and reuses across same-sized chunks. For a
        # fixed-size MD trajectory this is one compile total and the per-
        # tile kernel runs at full XLA-fused speed afterward.
        ptile = int(getattr(builder, "_pair_tile_size", 0))
        use_xla = ptile > 0
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
        # NOTE: predict_batch returns the TOTAL dipole regardless of
        # cfg.scale_targets. Training stores per-atom *targets*, and
        # TNEP.score() divides raw_preds by N to compare on the same scale
        # (TNEP.py:285-288). So preds is already the total system dipole;
        # no per-atom→total rescaling is needed here.
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
