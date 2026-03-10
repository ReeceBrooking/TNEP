import numpy as np

class TNEPconfig:
    """Holds all hyperparameters and runtime state for a TNEP training run.

    Class-level defaults are overwritten at runtime by MasterTNEP after
    data loading (num_types, types, dim_q, indices).
    """

    data_path: str = "train.xyz"
    num_neurons: int = 30
    # Number of structures used in each train step
    batch_size: int | None = None
    # Number of samples made in each train generation
    pop_size: int | None = 80
    # Number of training generations (number of updates to the model)
    num_generations: int = 5000

    # SOAP Turbo descriptor parameters
    l_max: int = 4
    alpha_max: int = 4

    # Cutoff radius value
    rc: float = 6.0

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
    target_mode : int = 1
    # Test split ratio
    test_ratio : float = 0.2
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N : int = None
    # Number of structures in each validation step (None = use entire val set)
    val_size : int | None = None
    # Number of SNES candidates to evaluate per GPU chunk (limits VRAM usage)
    population_chunk_size: int | None = 20
    # Number of structures to process per GPU chunk during evaluation (None = all at once)
    batch_chunk_size: int | None = 400

    # Periodic plotting interval (None = disabled; int = plot every N generations)
    plot_interval: int | None = 1000

    dim_q: int
    num_types: int
    types = []
    indices : np.ndarray

    def randomise(self, dataset):
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