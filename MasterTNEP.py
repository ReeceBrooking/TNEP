import tensorflow as tf
import numpy as np

from TNEP import TNEP
from TNEPconfig import TNEPconfig
from DescriptorBuilder import DescriptorBuilder
from ase.io import read
import matplotlib.pyplot as plt
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def collect(cfg : TNEPconfig):
    """Load all structures from train.xyz and assign integer type indices.

    Populates cfg.num_types and cfg.types as a side effect.

    Args:
        cfg : TNEPconfig with data_path set

    Returns:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray [N_i] — integer type index per atom
    """
    dataset = read(cfg.data_path, index=":")
    dataset_types_int = []
    types = []

    for structure in dataset:
        structure_types_int = np.zeros_like(structure.numbers)

        for i in range(len(structure.numbers)):
            z = structure.numbers[i]
            if z not in types:
                types.append(z)
            structure_types_int[i] = np.where(types == z)[0]

        dataset_types_int.append(structure_types_int)

    cfg.num_types = len(types)
    cfg.types = types
    return dataset, dataset_types_int

def split(dataset, dataset_types_int, cfg):
    """Split dataset into train / test / validation and build SOAP descriptors.

    Uses cfg.indices (shuffled) and cfg.test_ratio to partition. Builds
    descriptors and gradients via DescriptorBuilder for each split.

    Args:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray [N_i] integer type indices
        cfg               : TNEPconfig

    Returns:
        train_data, test_data, val_data : dicts each containing:
            positions   : list of [N_i, 3] tensors
            Z_int       : list of [N_i] int tensors (type indices)
            targets     : list of target tensors (scalar for PES, [3] for dipole)
            boxes       : list of [3, 3] tensors (lattice vectors)
            descriptors : list of [N_i, dim_q] tensors
            gradients   : list of (list of N_i tensors each [M, 3, dim_q])
            grad_index  : list of (list of N_i lists each [M] ints)
    """

    indices = cfg.indices
    n_structures = len(indices)

    n_test = int(cfg.test_ratio * n_structures)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:(2*n_test)]
    train_idx = indices[(2*n_test):n_structures]

    builder = DescriptorBuilder(cfg)

    test_dataset = [dataset[i] for i in test_idx]
    val_dataset = [dataset[i] for i in val_idx]
    train_dataset = [dataset[i] for i in train_idx]

    test_types_int = [dataset_types_int[i] for i in test_idx]
    val_types_int = [dataset_types_int[i] for i in val_idx]
    train_types_int = [dataset_types_int[i] for i in train_idx]

    train_descriptors, train_gradients, train_grad_index = builder.build_descriptors(train_dataset)
    val_descriptors, val_gradients, val_grad_index = builder.build_descriptors(val_dataset)
    test_descriptors, test_gradients, test_grad_index = builder.build_descriptors(test_dataset)

    """
    # Scale descriptors by inverse range (GPUMD q_scaler convention)
    q_scaler = DescriptorBuilder.compute_q_scaler(train_descriptors)
    train_descriptors, train_gradients = DescriptorBuilder.apply_scaling(
        train_descriptors, train_gradients, q_scaler)
    val_descriptors, val_gradients = DescriptorBuilder.apply_scaling(
        val_descriptors, val_gradients, q_scaler)
    test_descriptors, test_gradients = DescriptorBuilder.apply_scaling(
        test_descriptors, test_gradients, q_scaler)
    """

    if cfg.target_mode == 0:
        target = "energy"
    elif cfg.target_mode == 1:
        target = "dipole"
    elif cfg.target_mode == 2:
        target = "pol"

    def _extract_target(structure, target_key):
        """Extract target, converting 9-component polarizability to 6-component if needed."""
        raw = np.asarray(structure.info[target_key], dtype=np.float32)
        if raw.size == 9:
            # Flattened 3x3 row-major -> unique [xx, yy, zz, xy, yz, zx]
            raw = raw[[0, 4, 8, 1, 5, 6]]
        return tf.convert_to_tensor(raw, dtype=tf.float32)

    def subset(input, descriptors, gradients, grad_index, types_int, target):
        return {
            "positions": [tf.convert_to_tensor(structure.positions, dtype = tf.float32) for structure in input],
            "Z_int": [tf.convert_to_tensor(structure_types_int, dtype = tf.int32) for structure_types_int in types_int],
            "targets": [_extract_target(structure, target) for structure in input],
            "boxes": [tf.convert_to_tensor(structure.cell.array, dtype = tf.float32) for structure in input],
            "descriptors": descriptors,
            "gradients": gradients,
            "grad_index": grad_index,
        }

    train_data = subset(train_dataset, train_descriptors, train_gradients, train_grad_index, train_types_int, target)
    test_data = subset(test_dataset, test_descriptors, test_gradients, test_grad_index, test_types_int, target)
    val_data = subset(val_dataset, val_descriptors, val_gradients, val_grad_index, val_types_int, target)
    print(str(n_structures) + " structures have been split into sets of size:")
    print("Train set: " + str(len(train_data["positions"])) + " structures")
    print("Test set: " + str(len(test_data["positions"])) + " structures")
    print("Validation set: " + str(len(val_data["positions"])) + " structures")
    return train_data, test_data, val_data

def pad_and_stack(data):
    """Convert variable-length list-of-tensors data into dense padded tensors.

    Transforms the output of split() into fixed-shape tensors suitable for
    batched GPU evaluation. Variable atom counts and neighbor counts are
    padded to their maximums with zeros, and boolean masks track real vs
    padded entries.

    Args:
        data : dict from split() with keys:
            descriptors : list of [N_i, dim_q] tensors
            gradients   : list of (list of N_i tensors each [M_ij, 3, dim_q])
            grad_index  : list of (list of N_i lists each [M_ij] ints)
            positions   : list of [N_i, 3] tensors
            Z_int       : list of [N_i] int tensors
            targets     : list of scalar/[3]/[6] tensors
            boxes       : list of [3, 3] tensors

    Returns:
        padded : dict with keys:
            descriptors    : [S, A, Q]        float32
            gradients      : [S, A, M, 3, Q]  float32
            grad_index     : [S, A, M]        int32
            positions      : [S, A, 3]        float32
            Z_int          : [S, A]           int32
            targets        : [S, T]           float32  (T=1 for PES, 3 for dipole, 6 for pol)
            boxes          : [S, 3, 3]        float32
            atom_mask      : [S, A]           float32  (1.0 for real atoms, 0.0 for padding)
            neighbor_mask  : [S, A, M]        float32  (1.0 for real neighbors, 0.0 for padding)
            num_atoms      : [S]              int32    (actual atom count per structure)
        where S = num_structures, A = max_atoms, M = max_neighbors, Q = dim_q
    """
    S = len(data["descriptors"])
    dim_q = data["descriptors"][0].shape[-1]

    # Find max atom count across all structures
    atom_counts = [data["descriptors"][i].shape[0] for i in range(S)]
    max_atoms = max(atom_counts)

    # Find max neighbor count across all atoms in all structures
    max_neighbors = 0
    for s in range(S):
        for i in range(atom_counts[s]):
            n_nbrs = data["gradients"][s][i].shape[0]
            if n_nbrs > max_neighbors:
                max_neighbors = n_nbrs

    # Target dimensionality
    target_sample = data["targets"][0]
    if target_sample.shape == ():
        target_dim = 1
    else:
        target_dim = target_sample.shape[0]

    # Pre-allocate numpy arrays (faster than list comprehension for padding)
    desc_np = np.zeros((S, max_atoms, dim_q), dtype=np.float32)
    grad_np = np.zeros((S, max_atoms, max_neighbors, 3, dim_q), dtype=np.float32)
    gidx_np = np.zeros((S, max_atoms, max_neighbors), dtype=np.int32)
    pos_np = np.zeros((S, max_atoms, 3), dtype=np.float32)
    z_np = np.zeros((S, max_atoms), dtype=np.int32)
    tgt_np = np.zeros((S, target_dim), dtype=np.float32)
    box_np = np.zeros((S, 3, 3), dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms), dtype=np.float32)
    nbr_mask_np = np.zeros((S, max_atoms, max_neighbors), dtype=np.float32)
    num_atoms_np = np.array(atom_counts, dtype=np.int32)

    for s in range(S):
        N_s = atom_counts[s]
        desc_np[s, :N_s, :] = data["descriptors"][s].numpy()
        pos_np[s, :N_s, :] = data["positions"][s].numpy()
        z_np[s, :N_s] = data["Z_int"][s].numpy()
        box_np[s] = data["boxes"][s].numpy()
        atom_mask_np[s, :N_s] = 1.0

        t = data["targets"][s]
        if t.shape == ():
            tgt_np[s, 0] = t.numpy()
        else:
            tgt_np[s, :] = t.numpy()

        for i in range(N_s):
            n_nbrs = data["gradients"][s][i].shape[0]
            grad_np[s, i, :n_nbrs, :, :] = data["gradients"][s][i].numpy()
            gidx_np[s, i, :n_nbrs] = data["grad_index"][s][i]
            nbr_mask_np[s, i, :n_nbrs] = 1.0

    return {
        "descriptors": tf.constant(desc_np),
        "gradients": tf.constant(grad_np),
        "grad_index": tf.constant(gidx_np),
        "positions": tf.constant(pos_np),
        "Z_int": tf.constant(z_np),
        "targets": tf.constant(tgt_np),
        "boxes": tf.constant(box_np),
        "atom_mask": tf.constant(atom_mask_np),
        "neighbor_mask": tf.constant(nbr_mask_np),
        "num_atoms": tf.constant(num_atoms_np),
    }

def _make_plot_filename(cfg, plot_name):
    """Generate an automatic filename for a plot based on config and plot name.

    Format: {plot_name}_mode{target_mode}_pop{pop_size}_n{num_neurons}_l{l_max}.png

    Args:
        cfg       : TNEPconfig
        plot_name : str — short identifier for the plot type

    Returns:
        filename : str — e.g. "snes_fitness_mode1_pop80_n30_l4.png"
    """
    mode_names = {0: "pes", 1: "dipole", 2: "polar"}
    mode = mode_names.get(cfg.target_mode, f"mode{cfg.target_mode}")
    return (f"{plot_name}_{mode}_pop{cfg.pop_size}_n{cfg.num_neurons}"
            f"_l{cfg.l_max}.png")


def _save_fig(fig, cfg, plot_name):
    """Save a figure if cfg.save_plots is set.

    Args:
        fig       : matplotlib Figure
        cfg       : TNEPconfig
        plot_name : str — short identifier for the plot type
    """
    if cfg.save_plots is None:
        return
    os.makedirs(cfg.save_plots, exist_ok=True)
    filename = _make_plot_filename(cfg, plot_name)
    path = os.path.join(cfg.save_plots, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved: {path}")


def plot_snes_history(history, logy=False, cfg=None):
    """Plot train and validation RMSE vs generation with best/worst band."""
    g = np.asarray(history["generation"])

    fig = plt.figure()
    plt.plot(g, history["train_loss"], label="Train RMSE")
    plt.plot(g, history["val_loss"], label="Val RMSE")

    if history.get("best_rmse") and history.get("worst_rmse"):
        plt.fill_between(g, history["best_rmse"], history["worst_rmse"],
                         alpha=0.2, label="Best\u2013Worst range")

    plt.xlabel("generation")
    plt.ylabel("fitness (lower is better)")
    if logy:
        plt.yscale("log")
    plt.legend()
    plt.title("SNES fitness vs generation")
    if cfg:
        _save_fig(fig, cfg, "snes_fitness")
    plt.show()

def plot_log_val_fitness(history, cfg=None):
    """Plot natural log of validation fitness vs generation."""
    g = np.asarray(history["generation"])
    val = np.asarray(history["val_loss"])
    ln_val = np.log(val)

    fig = plt.figure()
    plt.plot(g, ln_val, label="ln(Val RMSE)")
    plt.xlabel("Generation")
    plt.ylabel("ln(Validation RMSE)")
    plt.legend()
    plt.title("Log validation fitness vs generation")
    if cfg:
        _save_fig(fig, cfg, "log_val_fitness")
    plt.show()

def plot_sigma_history(history, cfg=None):
    """Plot sigma min/max/mean vs generation on log-y scale with reset markers."""
    g = np.asarray(history["generation"])
    if not history.get("sigma_mean"):
        return

    fig = plt.figure()
    plt.plot(g, history["sigma_mean"], label="Sigma mean")
    plt.plot(g, history["sigma_median"], label="Sigma median", linestyle="--")
    plt.plot(g, history["sigma_min"], label="Sigma min", alpha=0.6)
    plt.plot(g, history["sigma_max"], label="Sigma max", alpha=0.6)
    plt.fill_between(g, history["sigma_min"], history["sigma_max"],
                     alpha=0.15, color="blue")

    for reset_gen in history.get("sigma_resets", []):
        plt.axvline(x=reset_gen, color="red", linestyle="--", alpha=0.7,
                    label="Sigma reset" if reset_gen == history["sigma_resets"][0] else None)

    plt.xlabel("generation")
    plt.ylabel("sigma")
    plt.yscale("log")
    plt.legend()
    plt.title("SNES sigma evolution")
    if cfg:
        _save_fig(fig, cfg, "sigma_evolution")
    plt.show()

def plot_timing(history, cfg=None):
    """Plot per-generation timing breakdown and aggregate summary."""
    timing = history.get("timing")
    if not timing or not timing.get("evaluate"):
        return

    g = np.arange(len(timing["evaluate"]))
    phases = ["evaluate", "validate", "rank_update", "sample_batch", "overhead"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#95a5a6"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: stacked area chart per generation
    data = [np.array(timing[p]) for p in phases]
    ax1.stackplot(g, *data, labels=phases, colors=colors, alpha=0.8)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Time (s)")
    ax1.legend(loc="upper left")
    ax1.set_title("Per-generation timing breakdown")

    # Right: horizontal bar chart of totals
    totals = [sum(timing[p]) for p in phases]
    grand_total = max(sum(totals), 1e-9)
    bars = ax2.barh(phases, totals, color=colors)
    ax2.set_xlabel("Total time (s)")
    ax2.set_title(f"Aggregate timing ({grand_total:.2f}s total)")
    for bar, total in zip(bars, totals):
        pct = 100 * total / grand_total
        ax2.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f" {total:.3f}s ({pct:.1f}%)", va='center')

    plt.tight_layout()
    if cfg:
        _save_fig(fig, cfg, "timing")
    plt.show()

def plot_correlation(targets, predictions, metrics, cfg):
    """Plot target vs prediction correlation with per-component R² and cosine similarity.

    For scalar targets (PES), plots a single correlation panel.
    For vector targets (dipole [3] or polarizability [6]), plots one correlation
    panel per component plus a cosine similarity histogram.

    Args:
        targets     : [S, T] numpy array of target values
        predictions : [S, T] numpy array of predicted values
        metrics     : dict from TNEP.score() with r2, rmse, r2_components, cos_sim_all
        cfg         : TNEPconfig — used to determine target mode and labels
    """
    T = targets.shape[1]
    rmse = float(metrics["rmse"])
    r2 = float(metrics["r2"])
    r2_comp = metrics["r2_components"].numpy()

    if cfg.target_mode == 0:
        labels = ["Energy"]
    elif cfg.target_mode == 1:
        labels = ["x", "y", "z"]
    elif cfg.target_mode == 2:
        labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
    else:
        labels = [f"comp {i}" for i in range(T)]

    # Add extra column for cosine similarity histogram on vector targets
    has_cos = cfg.target_mode >= 1 and "cos_sim_all" in metrics
    ncols = min(T, 3)
    nrows = (T + ncols - 1) // ncols
    if has_cos:
        nrows += 1  # extra row for cosine similarity

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for i in range(T):
        ax = axes[i // ncols, i % ncols]
        t = targets[:, i]
        p = predictions[:, i]

        ax.scatter(t, p, s=10, alpha=0.6)

        # x = y reference line
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        margin = 0.05 * (hi - lo) if hi > lo else 0.5
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=1, label="x = y")

        ax.set_xlabel(f"Target ({labels[i]})")
        ax.set_ylabel(f"Prediction ({labels[i]})")
        ax.set_title(f"{labels[i]}  R²={r2_comp[i]:.4f}")
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper left')

    # Hide unused correlation subplots
    cos_row_start = (T + ncols - 1) // ncols  # first row used by cosine sim
    for i in range(T, cos_row_start * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    # Cosine similarity histogram
    if has_cos:
        cos_all = metrics["cos_sim_all"].numpy()
        cos_mean = float(metrics["cos_sim_mean"])

        # Span all columns in the bottom row
        for c in range(ncols):
            axes[nrows - 1, c].set_visible(False)
        ax_cos = fig.add_subplot(nrows, 1, nrows)

        ax_cos.hist(cos_all, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax_cos.axvline(cos_mean, color='red', linestyle='--', linewidth=1.5,
                       label=f"Mean = {cos_mean:.4f}")
        ax_cos.set_xlabel("Cosine similarity")
        ax_cos.set_ylabel("Count")
        ax_cos.set_xlim(-1.05, 1.05)
        ax_cos.set_title(f"Cosine similarity distribution  "
                         f"(mean={cos_mean:.4f}, std={cos_all.std():.4f})")
        ax_cos.legend()

    mode_names = {0: "PES", 1: "Dipole", 2: "Polarizability"}
    mode = mode_names.get(cfg.target_mode, f"Mode {cfg.target_mode}")
    fig.suptitle(f"{mode} — RMSE: {rmse:.4f}, R²: {r2:.4f}", fontsize=14)
    plt.tight_layout()
    _save_fig(fig, cfg, "correlation")
    plt.show()

def compute_dipole_acf(dipoles):
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


def compute_ir_spectrum(dipoles, dt_fs=1.0, window='hann', max_freq_cm=4000.0):
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


def plot_ir_spectrum(freq_cm, intensity, title="IR Spectrum"):
    """Plot IR absorption spectrum.

    Args:
        freq_cm   : [N] ndarray — frequencies in cm⁻¹
        intensity : [N] ndarray — normalised IR intensity
        title     : str — plot title
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freq_cm, intensity, color='black', linewidth=0.8)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("IR Intensity (arb. units)")
    ax.set_title(title)
    ax.set_xlim(freq_cm[0], freq_cm[-1])
    ax.invert_xaxis()  # IR convention: high to low wavenumber
    plt.tight_layout()
    plt.show()


def predict_dipole_trajectory(model, trajectory, dataset_types_int, cfg):
    """Predict dipole moments for each frame in an MD trajectory.

    Args:
        model             : trained TNEP model (target_mode = 1)
        trajectory        : list of ase.Atoms — MD frames (ordered in time)
        dataset_types_int : list of [N_i] int arrays — type indices per frame
        cfg               : TNEPconfig

    Returns:
        dipoles : [T, 3] ndarray — predicted dipole moment per frame
    """
    builder = DescriptorBuilder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(trajectory)

    dipoles = []
    for s in range(len(trajectory)):
        struct = trajectory[s]
        N = len(struct)
        pos = tf.constant(struct.positions, dtype=tf.float32)
        Z = tf.constant(dataset_types_int[s], dtype=tf.int32)
        box = tf.constant(struct.cell.array, dtype=tf.float32)
        desc = descriptors[s]
        atom_mask = tf.ones([N], dtype=tf.float32)

        # Pad gradients/grad_index for this structure
        max_nbrs = max(gradients[s][i].shape[0] for i in range(N))
        grad_padded = np.zeros((N, max_nbrs, 3, cfg.dim_q), dtype=np.float32)
        gidx_padded = np.zeros((N, max_nbrs), dtype=np.int32)
        nbr_mask = np.zeros((N, max_nbrs), dtype=np.float32)
        for i in range(N):
            n_nbrs = gradients[s][i].shape[0]
            grad_padded[i, :n_nbrs] = gradients[s][i].numpy()
            gidx_padded[i, :n_nbrs] = grad_index[s][i]
            nbr_mask[i, :n_nbrs] = 1.0

        mu = model.predict(
            desc, tf.constant(grad_padded), tf.constant(gidx_padded),
            pos, Z, box, atom_mask, tf.constant(nbr_mask, dtype=tf.float32))
        dipoles.append(mu.numpy())

    return np.array(dipoles)


def predict_polarizability_trajectory(model, trajectory, dataset_types_int, cfg):
    """Predict polarizability tensors for each frame in an MD trajectory.

    Args:
        model             : trained TNEP model (target_mode = 2)
        trajectory        : list of ase.Atoms — MD frames (ordered in time)
        dataset_types_int : list of [N_i] int arrays — type indices per frame
        cfg               : TNEPconfig

    Returns:
        pols : [T, 6] ndarray — predicted polarizability [xx, yy, zz, xy, yz, zx] per frame
    """
    builder = DescriptorBuilder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(trajectory)

    pols = []
    for s in range(len(trajectory)):
        struct = trajectory[s]
        N = len(struct)
        pos = tf.constant(struct.positions, dtype=tf.float32)
        Z = tf.constant(dataset_types_int[s], dtype=tf.int32)
        box = tf.constant(struct.cell.array, dtype=tf.float32)
        desc = descriptors[s]
        atom_mask = tf.ones([N], dtype=tf.float32)

        max_nbrs = max(gradients[s][i].shape[0] for i in range(N))
        grad_padded = np.zeros((N, max_nbrs, 3, cfg.dim_q), dtype=np.float32)
        gidx_padded = np.zeros((N, max_nbrs), dtype=np.int32)
        nbr_mask = np.zeros((N, max_nbrs), dtype=np.float32)
        for i in range(N):
            n_nbrs = gradients[s][i].shape[0]
            grad_padded[i, :n_nbrs] = gradients[s][i].numpy()
            gidx_padded[i, :n_nbrs] = grad_index[s][i]
            nbr_mask[i, :n_nbrs] = 1.0

        pol = model.predict(
            desc, tf.constant(grad_padded), tf.constant(gidx_padded),
            pos, Z, box, atom_mask, tf.constant(nbr_mask, dtype=tf.float32))
        pols.append(pol.numpy())

    return np.array(pols)


def _scalar_acf_fft(signal):
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


def _cross_acf_fft(a, b):
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


def compute_raman_acfs(polarizabilities):
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


def compute_raman_spectrum(polarizabilities, dt_fs=1.0, window='hann',
                           max_freq_cm=4000.0, temperature=300.0):
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


def plot_raman_spectrum(freq_cm, I_VV, I_VH, I_total, title="Raman Spectrum"):
    """Plot Raman spectrum with parallel, perpendicular, and total components.

    Args:
        freq_cm : [N] ndarray — frequencies in cm⁻¹
        I_VV    : [N] ndarray — parallel (polarised) intensity
        I_VH    : [N] ndarray — perpendicular (depolarised) intensity
        I_total : [N] ndarray — total intensity
        title   : str — plot title
    """
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
    plt.show()


def save_model(model, cfg, path="tnep_model.npz"):
    """Save trained TNEP model weights and config to a .npz file.

    Args:
        model : trained TNEP model
        cfg   : TNEPconfig used for training
        path  : str — output file path
    """
    data = {
        "W0": model.W0.numpy(),
        "b0": model.b0.numpy(),
        "W1": model.W1.numpy(),
        "b1": model.b1.numpy(),
        "num_types": np.array(cfg.num_types),
        "num_neurons": np.array(cfg.num_neurons),
        "dim_q": np.array(cfg.dim_q),
        "types": np.array(cfg.types),
        "target_mode": np.array(cfg.target_mode),
        "l_max": np.array(cfg.l_max),
        "alpha_max": np.array(cfg.alpha_max),
        "activation": np.array(cfg.activation),
    }
    if cfg.target_mode == 2:
        data["W0_pol"] = model.W0_pol.numpy()
        data["b0_pol"] = model.b0_pol.numpy()
        data["W1_pol"] = model.W1_pol.numpy()
        data["b1_pol"] = model.b1_pol.numpy()

    np.savez(path, **data)
    print(f"Model saved to {path}")


def load_model(path="tnep_model.npz"):
    """Load a trained TNEP model from a .npz file.

    Reconstructs a TNEPconfig and TNEP model with the saved weights.

    Args:
        path : str — path to saved .npz file

    Returns:
        model : TNEP model with loaded weights
        cfg   : TNEPconfig reconstructed from saved parameters
    """
    data = np.load(path, allow_pickle=True)
    cfg = TNEPconfig()
    cfg.num_types = int(data["num_types"])
    cfg.num_neurons = int(data["num_neurons"])
    cfg.dim_q = int(data["dim_q"])
    cfg.types = data["types"].tolist()
    cfg.target_mode = int(data["target_mode"])
    cfg.l_max = int(data["l_max"])
    cfg.alpha_max = int(data["alpha_max"])
    cfg.activation = str(data["activation"])

    model = TNEP(cfg)
    model.W0.assign(data["W0"])
    model.b0.assign(data["b0"])
    model.W1.assign(data["W1"])
    model.b1.assign(data["b1"])

    if cfg.target_mode == 2:
        model.W0_pol.assign(data["W0_pol"])
        model.b0_pol.assign(data["b0_pol"])
        model.W1_pol.assign(data["W1_pol"])
        model.b1_pol.assign(data["b1_pol"])

    print(f"Model loaded from {path} (target_mode={cfg.target_mode}, "
          f"dim_q={cfg.dim_q}, num_types={cfg.num_types})")
    return model, cfg


def filter_by_species(dataset, dataset_types_int, allowed_Z):
    """Keep only structures whose atoms are all within allowed_Z.

    Args:
        dataset           : list of ase.Atoms
        dataset_types_int : list of ndarray — parallel to dataset
        allowed_Z         : list of int — allowed atomic numbers (e.g. [6, 1, 8])

    Returns:
        filtered_dataset, filtered_types_int : filtered parallel lists
    """
    allowed = set(allowed_Z)
    filtered_dataset = []
    filtered_types_int = []
    for struct, types_int in zip(dataset, dataset_types_int):
        if set(struct.numbers).issubset(allowed):
            filtered_dataset.append(struct)
            filtered_types_int.append(types_int)
    return filtered_dataset, filtered_types_int

def print_dipole_statistics(dataset, target_key="dipole"):
    """Print min/max/mean/std of dipole targets across the dataset.

    Args:
        dataset    : list of ase.Atoms with info[target_key] = [3] array
        target_key : str key in Atoms.info holding the dipole vector
    """
    dipoles = np.array([s.info[target_key] for s in dataset])
    norms = np.linalg.norm(dipoles, axis=1)
    print("=== Dipole Target Statistics ===")
    print(f"  N structures: {len(dipoles)}")
    print(f"  Component ranges: x=[{dipoles[:,0].min():.4f}, {dipoles[:,0].max():.4f}]  "
          f"y=[{dipoles[:,1].min():.4f}, {dipoles[:,1].max():.4f}]  "
          f"z=[{dipoles[:,2].min():.4f}, {dipoles[:,2].max():.4f}]")
    print(f"  Component means:  x={dipoles[:,0].mean():.4f}  y={dipoles[:,1].mean():.4f}  z={dipoles[:,2].mean():.4f}")
    print(f"  Component stds:   x={dipoles[:,0].std():.4f}  y={dipoles[:,1].std():.4f}  z={dipoles[:,2].std():.4f}")
    print(f"  |μ| range: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"  |μ| mean:  {norms.mean():.4f}  std: {norms.std():.4f}")

def print_polarizability_statistics(dataset, target_key="pol"):
    """Print min/max/mean/std of polarizability targets across the dataset.

    Args:
        dataset    : list of ase.Atoms with info[target_key] = [6] or [9] array
        target_key : str key in Atoms.info holding the polarizability tensor
    """
    pols = []
    for s in dataset:
        raw = np.asarray(s.info[target_key], dtype=np.float32)
        if raw.size == 9:
            raw = raw[[0, 4, 8, 1, 5, 6]]
        pols.append(raw)
    pols = np.array(pols)
    labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
    print("=== Polarizability Target Statistics ===")
    print(f"  N structures: {len(pols)}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{pols[:,i].min():.4f}, {pols[:,i].max():.4f}]  "
              f"mean={pols[:,i].mean():.4f}  std={pols[:,i].std():.4f}")

cfg = TNEPconfig()

# Load raw dataset and assign initial type indices
dataset, dataset_types_int = collect(cfg)
print("Number of species in raw dataset: " + str(cfg.num_types))
print("Number of structures in raw dataset: " + str(len(dataset)))

# Filter to structures containing only C, H, O (Z = 6, 1, 8)
dataset, dataset_types_int = filter_by_species(dataset, dataset_types_int, allowed_Z=[6, 1, 8])
print("After C/H/O filter: " + str(len(dataset)) + " structures")

# Recompute type list and indices after filtering
cfg.num_types = 0
cfg.types = []
for struct in dataset:
    for z in struct.numbers:
        if z not in cfg.types:
            cfg.types.append(z)
cfg.num_types = len(cfg.types)

dataset_types_int = []
for struct in dataset:
    structure_types_int = np.zeros_like(struct.numbers)
    for i in range(len(struct.numbers)):
        z = struct.numbers[i]
        structure_types_int[i] = cfg.types.index(z)
    dataset_types_int.append(structure_types_int)
print("Species after filter: " + str(cfg.types) + " (" + str(cfg.num_types) + " types)")

if cfg.target_mode == 1:
    print_dipole_statistics(dataset)
elif cfg.target_mode == 2:
    print_polarizability_statistics(dataset)

cfg.randomise(dataset)

# Split into train/test/val and build SOAP descriptors (slow)
train_data, test_data, val_data = split(dataset, dataset_types_int, cfg)

# Convert to padded dense tensors for GPU-batched evaluation
train_data = pad_and_stack(train_data)
test_data = pad_and_stack(test_data)
val_data = pad_and_stack(val_data)

# dim_q is determined by the SOAP descriptor size
cfg.dim_q = train_data["descriptors"][0].shape[-1]
print("Dimension of q: " + str(cfg.dim_q))

model = TNEP(cfg)
print("Model Parameters: " + str(model.optimizer.dim))
print("Population Size: " + str(model.optimizer.pop_size))
print("Parameter Natural Log: " + str(np.log(model.optimizer.dim)))
print("Parameter Root: " + str(np.sqrt(model.optimizer.dim)))

def periodic_plot_callback(history, gen):
    """Called during training at plot_interval to show progress."""
    print(f"\n--- Periodic plots at generation {gen} ---")
    # Score on test set with current best weights
    m, preds = model.score(test_data)
    print(f"  Test RMSE: {float(m['rmse']):.4f}  R²: {float(m['r2']):.4f}")
    plot_snes_history(history)
    plot_log_val_fitness(history)
    plot_sigma_history(history)
    plot_timing(history)
    plot_correlation(test_data["targets"].numpy(), preds.numpy(), m, cfg)

history = model.fit(train_data, val_data,
                    plot_callback=periodic_plot_callback if cfg.plot_interval else None)
metrics, test_preds = model.score(test_data)
test_rmse = float(metrics["rmse"])
test_r2 = float(metrics["r2"])
r2_components = metrics["r2_components"].numpy()
print(f"\nModel test set RMSE: {test_rmse:.4f}")
print(f"Model test set R²:   {test_r2:.4f}")

if cfg.target_mode == 0:
    comp_labels = ["Energy"]
elif cfg.target_mode == 1:
    comp_labels = ["x", "y", "z"]
elif cfg.target_mode == 2:
    comp_labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
else:
    comp_labels = [f"comp {i}" for i in range(len(r2_components))]

print("Per-component R²:  " + "  ".join(
    f"{lbl}={r2_components[i]:.4f}" for i, lbl in enumerate(comp_labels)))

if "cos_sim_mean" in metrics:
    cos_mean = float(metrics["cos_sim_mean"])
    cos_all = metrics["cos_sim_all"].numpy()
    print(f"Cosine similarity:  mean={cos_mean:.4f}  "
          f"min={cos_all.min():.4f}  max={cos_all.max():.4f}  "
          f"std={cos_all.std():.4f}")
print("Run complete!")

if cfg.save_path is not None:
    save_model(model, cfg, cfg.save_path)

timing = history.get("timing", {})
if timing:
    phases = ["sample_batch", "evaluate", "rank_update", "validate", "overhead"]
    grand = sum(sum(timing[p]) for p in phases)
    n_gens = len(timing["evaluate"])
    print(f"\n=== Timing Breakdown ({grand:.2f}s over {n_gens} generations) ===")
    for p in phases:
        t = sum(timing[p])
        avg = t / max(n_gens, 1)
        pct = 100 * t / max(grand, 1e-9)
        print(f"  {p:15s}: {t:.3f}s total ({pct:5.1f}%) | {avg*1000:.1f}ms/gen")

vram = history.get("vram_mb", [])
if vram:
    print(f"\n=== VRAM Usage ===")
    print(f"  Peak: {max(vram):.1f} MB / 12288 MB ({100*max(vram)/12288:.1f}%)")
    print(f"  Last: {vram[-1]:.1f} MB")

plot_snes_history(history, cfg=cfg)
plot_log_val_fitness(history, cfg=cfg)
plot_sigma_history(history, cfg=cfg)
plot_timing(history, cfg=cfg)
plot_correlation(test_data["targets"].numpy(), test_preds.numpy(), metrics, cfg)
