from __future__ import annotations

import numpy as np


class TNEPconfig:
    """Holds all hyperparameters and runtime state for a TNEP training run.

    Runtime-state fields (num_types, types, dim_q, indices) are overwritten
    by MasterTNEP after data loading.

    Sections:
        1. Dataset & targets
        2. Descriptor (SOAP-turbo: geometry, backend, preprocessing)
        3. Network architecture
        4. Loss & regularisation
        5. SNES optimiser
        6. Memory & I/O staging
        7. Output & diagnostics
        8. Runtime state (auto-populated by data load)
    """

    # ═══════════════════════════════════════════════════════════════════
    # 1. DATASET & TARGETS
    # ═══════════════════════════════════════════════════════════════════

    # --- dataset & split -----------------------------------------------
    data_path: str = "datasets/train_waterbulk.xyz"
    # Separate test dataset (None = split from data_path; str = path to external .xyz)
    test_data_path: str | None = "datasets/test_waterbulk.xyz"
    # Filter dataset to structures containing only these species
    # (None = no filter; list of int or str, e.g. [6, 1, 8] or ["C", "H", "O"])
    allowed_species: list[int | str] | None = [6, 1, 8]
    # Species filter mode: "subset" = keep structures with only allowed species,
    # "exact" = keep structures containing exactly all allowed species
    filter_mode: str = "subset"
    # Test split ratio (only used when test_data_path is None)
    test_ratio: float = 0.3
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N: int | None = None
    # Seed for randomisation (dataset shuffle, SNES sampling, etc.)
    seed: int | None = 928375439201

    # --- target type, units, conversions -------------------------------
    # 0 : PES (energy), 1 : Dipole, 2 : Polarizability
    target_mode: int = 1
    # Override the info/results key used to read targets from ASE structures.
    # None = use the default for target_mode ("energy", "dipole", "pol").
    # Set to a custom string to support non-standard dataset labels (e.g. "mu", "alpha").
    target_key: str | None = None
    # Scale dipole targets by atom count (per-atom dipole training)
    scale_targets: bool = True
    # Native units of dipole targets in the dataset: "e*angstrom" (e·Å),
    # "e*bohr" (e·a₀, × 0.5292 → e·Å), "debye" (× 0.2082 → e·Å). Sets the
    # e·Å conversion factor and the plot/stats unit label.
    dipole_units: str = "e*bohr"
    # True = convert dipole targets to e·Å on load; False = train/plot in
    # dataset native units (for RMSE comparable with e·a₀ / Debye references).
    convert_dipole_to_eangstrom: bool = False
    # Polarizability off-diagonal ([xy,yz,zx]) loss weight, scaled by
    # lambda_shear^2 (target_mode=2 only). 1.0 = equal weighting (GPUMD default).
    lambda_shear: float = 1.0

    # Power N of |r_ij| in the dipole contraction (target_mode=1 only):
    #   0 = self-pair-only, μ = −Σ_i de_dq[i]·grad_values[i,i] (distinct form)
    #   1 = first radial moment, μ = −Σ_pair |r_ij| ·F_ij
    #   2 = Xu et al. JCTC 2024 / GPUMD default, μ = −Σ_pair |r_ij|²·F_ij
    #   ≥3 = higher moments (diagnostic)
    # Models must be re-trained if N changes.
    dipole_rij_power: int = 0

    # ═══════════════════════════════════════════════════════════════════
    # 2. DESCRIPTOR (SOAP-turbo)
    # ═══════════════════════════════════════════════════════════════════

    # --- geometric parameters ------------------------------------------
    l_max: int = 7
    alpha_max: int = 7
    rcut_hard: float = 3.0
    rcut_soft: float = 2.5
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

    # --- compute backend (CPU/quippy vs GPU/TF) ------------------------
    # Descriptor backend: 0 = quippy (Fortran, CPU), 1 = native TF/NumPy (GPU).
    # GPU path supports basis="poly3" and compress_mode="trivial" only.
    descriptor_mode: int = 0

    # GPU descriptor compute precision: "float64" (matches Fortran reference)
    # or "float32" (~half VRAM, faster on consumer GPUs, looser agreement).
    # No effect for descriptor_mode=0.
    descriptor_precision: str = "float64"

    # Structures per SOAP graph call: 1 = per-frame (lowest VRAM), int>1 =
    # batched, None = auto-fit to memory budget. Used by training and
    # trajectory inference; process_trajectory kwarg overrides per call.
    descriptor_batch_frames: int | None = 500

    # Pair-tile size for the gradient compute: 0 = single-shot (higher peak
    # VRAM), >0 = tile pairs (cuts VRAM; at fp32 also enables XLA fusion).
    # ~8000 suits ~600-atom systems.
    descriptor_pair_tile_size: int = 8000

    # GPU memory budget (bytes) for the auto-sizer when
    # descriptor_batch_frames is None. None = builder default (6 GiB).
    descriptor_memory_budget_bytes: int | None = None

    # ═══════════════════════════════════════════════════════════════════
    # 3. NETWORK ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════

    # Hidden-layer width of the per-type ANN.
    num_neurons: int = 30
    # Hidden-layer activation (any tf.keras.activations name for the forward
    # pass). Dipole/polarisability training (target_mode 1/2) has a hand-coded
    # backward — only "tanh" and "swish" (alias "silu") supported there;
    # energy training (target_mode 0) accepts any.
    activation: str = 'swish'

    # When True, insert a learnable per-species-pair linear mixing layer
    # between the fixed SOAP-turbo descriptor and the ANN (U_pair shared
    # across central types, identity-init). Analog of NEP's trainable
    # radial-basis coefficients c_nk.
    descriptor_mixing: bool = True
    # When True (with descriptor_mixing), U_pair becomes per-central-type
    # ([T, num_pairs, max_bs, max_bs]) — more expressive, T× the param count.
    descriptor_mixing_per_type: bool = False
    # Parameterisation of each descriptor-mixing block (structural, not a
    # penalty); only used when descriptor_mixing=True:
    #   "off"    : unregularised, bs² params per block
    #   "cayley" : U = (I−A)(I+A)⁻¹, bs·(bs−1)/2 params, no −1-eigenvalue
    #   "expm"   : U = exp(A), surjective onto SO(n), bs·(bs−1)/2 params. RECOMMENDED.
    descriptor_mixing_regularizer: str = "expm"

    # Descriptor preprocessing contraction, before the ANN W0 layer; a learned
    # per-(centre type, raw channel) table contracts the raw SOAP descriptor
    # (Q_raw = raw dim, L = l_max+1). Composes with descriptor_mixing.
    #   "off"          : no preprocessing (W0 sees raw Q_raw)
    #   "angular"      : contract over l per (pair, n_pair); see angular_l_keep
    #   "species_pair" : per-type contraction into SELF vs OTHER pairs, Q_new = 2·max_α·L
    #   "both"         : collapse both axes (smallest Q_new, most info loss)
    #   "nep4_radial"  : NEP4-faithful learned-basis fold, Q_new = n_max_out·L;
    #                    requires compress_mode='trivial'
    descriptor_preprocess_contract: str = "off"
    # NEP4 fold output radial-channel count (nep4_radial mode only). None =
    # auto n_max_out = Q_raw/L so Q_new = Q_raw (dim preserved); small values
    # (≈ α or 2α) match NEP4's compact-basis setting.
    descriptor_nep4_n_max_out: int | None = None
    # Preprocess coefficient init: "mean" (1/N, well-conditioned), "sum" (1.0),
    # or "glorot" (Glorot-uniform).
    descriptor_preprocess_init: str = "mean"
    # Per-tail σ scaling for preprocess coefficients (1.0 = same as ANN; reduce
    # if SNES sampling noise overwhelms the signal).
    preprocess_sigma_scale: float = 1.0
    # Angular contraction threshold (angular/both modes): l < angular_l_keep
    # kept as passthrough channels, l ≥ it summed into one output channel.
    descriptor_preprocess_angular_l_keep: int = 2
    # True = W_pre coefficients per central-atom type; False = global across
    # centre types (symmetric, smallest param count).
    descriptor_preprocess_per_type: bool = False
    # L1/L2 strengths on (coefficient − init): soft prior toward the chosen
    # init, not zero. Both 0.0 = disable.
    descriptor_preprocess_lambda_1: float = 0.0005
    descriptor_preprocess_lambda_2: float = 0.0005

    # ═══════════════════════════════════════════════════════════════════
    # 4. LOSS & REGULARISATION
    # ═══════════════════════════════════════════════════════════════════

    # Master switch for L1/L2 regularisation. False = add 0 reg regardless
    # of lambda settings (lambda values kept so re-enabling mid-run resumes).
    toggle_regularization: bool = True
    # L1/L2 strengths: None = auto sqrt(dim*1e-6/num_types), -1.0 = dynamic
    # (adapt every lambda_adapt_interval gens toward lambda_target_ratio of
    # data RMSE), float = fixed.
    lambda_1: float | None = 0.0005
    lambda_2: float | None = 0.0005
    # Dynamic-λ controls (only when lambda_1 or lambda_2 == -1):
    #   target_ratio : target reg-penalty / data-RMSE ratio
    #   damping      : step exponent, λ_new = λ·(target/r)^d (smaller = gentler)
    #   interval     : gens between recompute/rescale
    #   min/max      : clamp on adapted λ
    lambda_target_ratio: float = 0.05
    lambda_damping: float = 0.2
    lambda_adapt_interval: int = 100
    lambda_min: float = 1e-8
    lambda_max: float = 1.0
    # Per-type regularization + fitness ranking (GPUMD NEP4 style), driving
    # per-type natural-gradient updates. Auto-disabled for single-element systems.
    per_type_regularization: bool = True

    # ═══════════════════════════════════════════════════════════════════
    # 5. SNES OPTIMISER
    # ═══════════════════════════════════════════════════════════════════

    # --- core ----------------------------------------------------------
    # Number of samples made in each train generation
    pop_size: int | None = 100
    # Number of training generations (number of updates to the model)
    num_generations: int = 60000
    # Number of structures used in each train step (None = full batch)
    batch_size: int | None = None
    # Learning rate for sigma (None = auto from canonical SNES heuristic)
    eta_sigma: float | None = None
    # Initial distribution standard deviation
    init_sigma: float = 0.1
    # Lower bound on sigma after each SNES update, preventing σ→0 collapse of
    # the search distribution on long runs. None or 0 = disable; 1e-6 is
    # non-distorting (typical adapted σ is 1e-3 to 1e-1).
    sigma_floor: float | None = 1e-6

    # --- validation ----------------------------------------------------
    # Number of structures in each validation step (None = use entire val set)
    val_size: int | None = None
    # Validate every N generations (1 = every gen, 10 = every 10th, etc.)
    val_interval: int = 1

    # --- early stopping ------------------------------------------------
    # Early stopping patience in val ticks (None = disabled)
    patience: int | None = None

    # ═══════════════════════════════════════════════════════════════════
    # 6. MEMORY & I/O STAGING
    # ═══════════════════════════════════════════════════════════════════

    # --- per-chunk compute ---------------------------------------------
    # SNES candidates evaluated per GPU chunk (limits VRAM): 10 = VRAM-safe
    # baseline for a 12 GB GPU; None = no chunking (whole population at once,
    # for A100-class / CSC GPU profile).
    population_chunk_size: int | None = 10
    # Structures per GPU chunk during evaluation. None = all at once; 2000
    # fits the dev box (CPU CSC profile uses 500, latency-sensitive).
    batch_chunk_size: int | None = 2000

    # Where static training tensors (descriptors, grad_values, positions, pair
    # indices) live: True = host CPU, per-chunk gather with H2D copy each gen
    # (for datasets too big for VRAM); False = /GPU:0, on-device gather
    # (fastest when the working set fits in VRAM).
    pin_data_to_cpu: bool = False

    # --- HPC / Slurm mode ----------------------------------------------
    # Master switch for CSC / Slurm mode (Mahti, Puhti, LUMI). True =
    # _apply_csc_overrides tunes per-chunk sizing for the detected profile.
    # False keeps non-HPC behaviour identical.
    csc_enable: bool = False

    # Force which CSC profile applies (csc_enable=True only):
    #   "auto" = pick from runtime hardware detection
    #   "cpu"  = force CPU profile (pop_chunk=10, batch_chunk=500, pin=True)
    #   "gpu"  = force GPU profile (pop_chunk=None, batch_chunk=2000, pin=False)
    # Canonical impl in _apply_csc_overrides (MasterTNEP.py) — keep both in sync.
    csc_profile: str = "auto"

    # ═══════════════════════════════════════════════════════════════════
    # 7. OUTPUT & DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════

    # Save model after training (None = disabled; "auto" = auto-generate run directory)
    save_path: str | None = "models/auto"
    # Save plots (None = disabled; set automatically by setup_run_directory)
    save_plots: str | None = None
    # Show plots interactively (True = plt.show(), False = close after saving)
    show_plots: bool = False
    # Periodic plotting interval (None = disabled; int = plot every N generations)
    plot_interval: int | None = None

    # Rolling checkpoint interval. None = disabled; int = write
    # {save_path}/checkpoint.h5 every N gens (embeds cfg, SNES distribution,
    # best-val state, history, RNG state — resume via train_model(checkpoint=path)).
    # Requires save_path.
    checkpoint_interval: int | None = 2000

    # Show extra info in progress bar (L1, L2 regularisation)
    debug: bool = False

    # ═══════════════════════════════════════════════════════════════════
    # 8. RUNTIME STATE — populated by MasterTNEP after data load
    # ═══════════════════════════════════════════════════════════════════

    dim_q: int
    num_types: int
    types: list[int] = []
    type_map: dict = {}
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
