from __future__ import annotations

import numpy as np


class TNEPconfig:
    """Holds all hyperparameters and runtime state for a TNEP training run.

    Class-level defaults are overwritten at runtime by MasterTNEP after
    data loading (num_types, types, dim_q, indices).
    """

    data_path: str = "train.xyz"
    # Filter dataset to structures containing only these species
    # (None = no filter; list of int or str, e.g. [6, 1, 8] or ["C", "H", "O"])
    species_filter: list[int | str] | None = None
    num_neurons: int = 30
    # Number of structures used in each train step
    batch_size: int | None = 50
    # Number of samples made in each train generation
    pop_size: int | None = 80
    # Number of training generations (number of updates to the model)
    num_generations: int = 20000

    # SOAP Turbo descriptor parameters
    l_max: int = 4
    alpha_max: int = 4
    rcut_hard: float = 3.7
    rcut_soft: float = 3.2
    basis: str = "poly3gauss"
    scaling_mode: str = "polynomial"
    radial_enhancement: int = 1
    compress_mode: str = "trivial"
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

    # Early stopping patience (None = disabled)
    patience: int | None = 5000

    # Sigma reset: reinitialise sigma when it collapses and training stagnates
    # None = disabled; int = generations stagnating + small sigma to trigger reset
    sigma_reset_patience: int | None = None
    # Fraction of init_sigma below which sigma is considered collapsed
    sigma_reset_threshold: float = 0.01

    activation: str = 'tanh'
    # Initial distribution standard deviation
    init_sigma: float = 0.1
    # Seed for randomisation
    seed: int | None = None
    # 0 : PES, 1 : Dipole, 2 : Polarizability
    target_mode: int = 1
    # Test split ratio
    test_ratio: float = 0.2
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N: int | None = None
    # Number of structures in each validation step (None = use entire val set)
    val_size: int | None = None
    # Number of SNES candidates to evaluate per GPU chunk (limits VRAM usage)
    population_chunk_size: int | None = 10
    # Number of structures to process per GPU chunk during evaluation (None = all at once)
    batch_chunk_size: int | None = None
    # Fraction of available RAM to budget for padded tensors (0.0-1.0)
    ram_threshold: float = 0.5
    # Fraction of available VRAM to budget for padded tensors (0.0-1.0)
    vram_threshold: float = 0.8
    # Total RAM in MB (None = auto-detect via psutil; set manually to override)
    ram_mb: int | None = None
    # Total GPU memory in MB (None = auto-detect via TF; set manually if detection fails)
    gpu_memory_mb: int | None = 12288
    # Reset sigma to init_sigma at the start of each chunk in chunked training
    chunk_sigma_reset: bool = True
    # Number of times to cycle through all chunks (None/1 = single pass)
    chunk_cycles: int | None = None

    # Periodic plotting interval (None = disabled; int = plot every N generations)
    plot_interval: int | None = None

    # Save model after training (None = disabled; "auto" = auto-generate name; str = explicit path)
    save_path: str | None = None #"auto"
    # Save final plots to directory (None = disabled; str = directory path)
    save_plots: str | None = None #"plots"
    # Show plots interactively (True = plt.show(), False = close after saving)
    show_plots: bool = True
    # Show extra info in progress bar (L1, L2 regularisation)
    debug: bool = False

    dim_q: int
    num_types: int
    types: list[int] = []
    indices: np.ndarray

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
