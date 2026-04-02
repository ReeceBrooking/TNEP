from __future__ import annotations

import numpy as np


class TNEPconfig:
    """Holds all hyperparameters and runtime state for a TNEP training run.

    Class-level defaults are overwritten at runtime by MasterTNEP after
    data loading (num_types, types, dim_q, indices).
    """

    data_path: str = "datasets/train.xyz"
    # Separate test dataset (None = split from data_path; str = path to external .xyz)
    test_data_path: str | None = "datasets/test.xyz"
    # Filter dataset to structures containing only these species
    # (None = no filter; list of int or str, e.g. [6, 1, 8] or ["C", "H", "O"])
    allowed_species: list[int | str] | None = ["C", "H", "O"]
    # Species filter mode: "subset" = keep structures with only allowed species,
    # "exact" = keep structures containing exactly all allowed species
    filter_mode: str = "subset"
    # Bad data filtering options
    filter_nan_positions: bool = False
    filter_nan_targets: bool = False
    filter_zero_targets: bool = False
    # Rigorous filtering: recompute targets with GPAW and filter by cosine similarity
    filter_rigorous: bool = False
    rigorous_threshold: float = 0.5
    num_neurons: int = 15
    # Number of structures used in each train step
    batch_size: int | None = None
    # Number of samples made in each train generation
    pop_size: int | None = 80
    # Number of training generations (number of updates to the model)
    num_generations: int = 100
    # Learning rate (None = auto)
    eta_sigma: float | None = None

    # SOAP Turbo descriptor parameters
    l_max: int = 4
    alpha_max: int = 4
    rcut_hard: float = 6.0
    rcut_soft: float = 5.5
    basis: str = "poly3"
    scaling_mode: str = "polynomial"
    radial_enhancement: int = 1
    compress_mode: str = "trivial"
    # Number of compressed radial channels (only used when compress_mode="linear"; None = quippy default)
    compress_P: int | None = None
    # PCA compression of descriptors (applied after SOAP computation, before scaling)
    # Uses compress_P for number of components. Independent of quippy's compress_mode.
    compress_pca: bool = False
    atom_sigma_r: float = 0.5
    atom_sigma_t: float = 0.5
    atom_sigma_r_scaling: float = 0.0
    atom_sigma_t_scaling: float = 0.0
    amplitude_scaling: float = 1.0
    central_weight: float = 1.0

    # L1/L2 regularization strengths (None = auto: sqrt(dim * 1e-6))
    toggle_regularization: bool = True
    lambda_1: float | None = 0.001
    lambda_2: float | None = 0.001
    # Per-type regularization and ranking (GPUMD NEP4 style)
    # Each type's params are regularized separately, creating per-type fitness
    # rankings that drive per-type natural gradient updates.
    # Only effective for multi-element systems (auto-disabled for single-element).
    per_type_regularization: bool = True

    # Early stopping patience (None = disabled)
    patience: int | None = None

    # Stagnation response: apply action after this many gens without val improvement
    # None = disabled; requires stagnation_response to also be set
    sigma_reset_patience: int | None = None
    # 'interpolate' = blend toward init_sigma; 'noise' = additive Gaussian; None = disabled
    stagnation_response: str | None = 'noise'
    # Blending factor for interpolate mode: sigma += alpha * (init_sigma - sigma)
    sigma_interpolate_alpha: float = 0.5
    # Log10 of noise std for noise mode: noise ~ N(0, 10^k)
    sigma_noise_scale: float = -3.0

    # Loss function: "mse" = root-mean-square error, "mae" = mean absolute error
    loss_type: str = "mse"
    # Inverse-magnitude weighting: upweight structures with small targets
    # Each structure's error is scaled by 1 / max(||target||^2, eps).
    # None = disabled (uniform weighting); float = epsilon floor (e.g. 0.01)
    inverse_weight_eps: float | None = None
    # Polarizability off-diagonal weight: loss for components [xy, yz, zx] scaled by lambda_shear^2
    # (GPUMD default 1.0 — equal weighting; <1.0 downweights off-diagonal)
    lambda_shear: float = 1.0

    activation: str = 'tanh'
    # Initial distribution standard deviation
    init_sigma: float = 0.1
    # Seed for randomisation
    seed: int | None = None
    # 0 : PES, 1 : Dipole, 2 : Polarizability
    target_mode: int = 1
    # Test split ratio
    test_ratio: float = 0.3
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N: int | None = 1000
    # Number of structures in each validation step (None = use entire val set)
    val_size: int | None = None
    # Validate every N generations (1 = every gen, 10 = every 10th, etc.)
    val_interval: int = 1
    # Number of SNES candidates to evaluate per GPU chunk (limits VRAM usage).
    # Peak VRAM ≈ C × total_edges × dim_q × 4 bytes (the de_dq_edge tensor).
    # Reduce this if hitting OOM. E.g. with 200k edges, Q=100: C=10→800MB, C=5→400MB.
    population_chunk_size: int | None = 10
    # Number of structures to process per GPU chunk during evaluation (None = all at once)
    batch_chunk_size: int | None = None

    # Periodic plotting interval (None = disabled; int = plot every N generations)
    plot_interval: int | None = 10000

    # Save model after training (None = disabled; "auto" = auto-generate run directory)
    save_path: str | None = "models/auto"
    # Save plots (None = disabled; set automatically by setup_run_directory)
    save_plots: str | None = None
    # Show plots interactively (True = plt.show(), False = close after saving)
    show_plots: bool = False
    # Show extra info in progress bar (L1, L2 regularisation)
    debug: bool = False
    # Negate predictions for structures detected as sign-flipped (cos_sim < -0.9)
    fix_sign_flips: bool = False
    # Scale input descriptors by their training-set statistics (per component)
    scale_descriptors: bool = False
    # Descriptor scaling method: "range" (GPUMD-style 1/(max-min)) or "mean" (mean(|x|)*sqrt(dim_q))
    descriptor_scale_mode: str = "mean"
    # Floor for mean scaling: fraction of max component mean (None = no floor; ignored for range mode)
    descriptor_scale_floor: float | None = 0.001
    # Scale dipole targets by atom count (per-atom dipole training)
    scale_targets: bool = True
    # Input units of dipole targets in the dataset.
    # "e*angstrom" = no conversion needed (already in e·Å)
    # "e*bohr"     = convert from e·bohr to e·Å (multiply by 0.5292)
    # "debye"      = convert from Debye to e·Å  (multiply by 0.2082)
    dipole_units: str = "e*bohr"

    dim_q: int
    num_types: int
    types: list[int] = []
    type_map: dict = {}
    indices: np.ndarray
    descriptor_mean: np.ndarray | None = None

    def randomise(self, dataset: list) -> None:
        """Shuffle dataset indices and truncate to total_N.

        Sets self.indices : ndarray [total_N] of shuffled structure indices.
        """
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(dataset), dtype=int)
        rng.shuffle(indices)
        if self.total_N is not None:
            self.indices = indices[:self.total_N]
        else:
            self.indices = indices
