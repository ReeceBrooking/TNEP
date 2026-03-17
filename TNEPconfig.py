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
    allowed_species: list[int | str] | None = ["C", "H", "O"]
    # Bad data filtering options
    filter_nan_positions: bool = False
    filter_nan_targets: bool = False
    filter_zero_targets: bool = True
    # Rigorous filtering: recompute targets with GPAW and filter by cosine similarity
    filter_rigorous: bool = False
    rigorous_threshold: float = 0.5
    num_neurons: int = 10
    # Number of structures used in each train step
    batch_size: int | None = 50
    # Number of samples made in each train generation
    pop_size: int | None = 80
    # Number of training generations (number of updates to the model)
    num_generations: int = 1000

    # SOAP Turbo descriptor parameters
    l_max: int = 4
    alpha_max: int = 4
    rcut_hard: float = 3.7
    rcut_soft: float = 3.2
    basis: str = "poly3"
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
    patience: int | None = None

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

    # Periodic plotting interval (None = disabled; int = plot every N generations)
    plot_interval: int | None = None

    # Save model after training (None = disabled; "auto" = auto-generate name; str = explicit path)
    save_path: str | None = None
    # Save final plots to directory (None = disabled; str = directory path)
    save_plots: str | None = None
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
