class TNEPconfig:
    data_path: str = "train.xyz"
    num_neurons: int = 64
    batch_size: int = 80
    pop_size: int = 8
#    learning_rate: float = 0.001
    num_generations: int = 100
#    epochs: int
    n_radial: int = 3
    n_radial_ang: int = 3
    Lmax: int = 2
    rc: float = 6.0
    activation: str = 'tanh'
    init_sigma: float = 0.1
    seed: int | None = None
    target_mode : int = 1
    test_ratio : float = 0.2
    total_N : int = 100
    val_size : int = 10

    dim_q: int
    num_types: int
