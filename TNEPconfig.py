import numpy as np

class TNEPconfig:
    data_path: str = "PEStrain.xyz"
    num_neurons: int = 64
    # Number of structures used in each train step
    batch_size: int = 10
    # Number of samples made in each train generation
    pop_size: int = 16
    # Number of training generations (number of updates to the model)
    num_generations: int = 50

    n_radial: int = 3
    n_radial_ang: int = 3
    Lmax: int = 2

    # Cutoff radius value
    rc: float = 6.0

    activation: str = 'tanh'
    # Initial distribution standard deviation
    init_sigma: float = 0.1
    seed: int | None = None
    # 0 : PES, 1 : Dipole, 2 : Polarizability
    target_mode : int = 0
    # Test split ratio
    test_ratio : float = 0.2
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N : int = 100
    # Number of structures in each validation step
    val_size : int = 5

    dim_q: int
    num_types: int
    indices : np.ndarray

    def randomise(self, dataset):
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(dataset))
        rng.shuffle(indices)
        if self.total_N is not None:
            self.indices = indices[:self.total_N]
        else:
            self.indices = indices