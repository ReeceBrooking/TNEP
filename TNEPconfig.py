from __future__ import annotations

import numpy as np


class TNEPconfig:
    """Holds all hyperparameters and runtime state for a TNEP training run.

    Runtime-state fields (num_types, types, dim_q, indices) are overwritten
    by MasterTNEP after data loading.

    Sections:
        1. Dataset & targets
        2. Descriptor (SOAP-turbo: geometry, backend)
        3. Network architecture
        4. Loss & regularisation
        5. SNES optimiser
        6. Memory & I/O staging
        7. Output & diagnostics
        8. Runtime state (auto-populated by data load)
    """

    # ── 1. Dataset & targets ──

    # --- dataset & split -----------------------------------------------
    data_path: str = "datasets/train_waterbulk_chi.xyz"
    # Separate test dataset (None = split from data_path; str = path to external .xyz)
    test_data_path: str | None = "datasets/test_waterbulk_chi.xyz"
    # Keep only structures whose species are in this list (None = no filter; ints or symbols)
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
    target_mode: int = 2
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
    # Mode 2 = rank-2 neighbour readout on two even bond kernels, isotropic
    # R_n·|d|²·I and traceless R_n·(d⊗d − |d|²I/3), each with its own site head
    # (they differ by ~two orders of magnitude, so no shared coefficient).
    # Kernel channels = 2·dipole_channels, group-major; dipole_readout must be "neighbor".
    lambda_shear: float = 1.0

    # Power N of |r_ij| in the dipole contraction (target_mode=1 only):
    #   0 = self-pair-only, μ = −Σ_i de_dq[i]·grad_values[i,i] (distinct form)
    #   1 = first radial moment, μ = −Σ_pair |r_ij| ·F_ij
    #   2 = Xu et al. JCTC 2024 / GPUMD default, μ = −Σ_pair |r_ij|²·F_ij
    #   ≥3 = higher moments (diagnostic)
    # Models must be re-trained if N changes.
    dipole_rij_power: int = 2

    # Dipole readout form (target_mode=1 only): "tnep" = descriptor-gradient form
    # above; "neighbor" = b_ij = Σ_n c_n[p(t_i,t_j)]·R_n(r_ij)·(s_i^n − s_j^n),
    # μ = Σ_{i<j} b_ij·(r_i − r_j) (MIC displacements), R_n per dipole_radial.
    dipole_readout: str = "neighbor"
    # Radial channels C of the neighbor readout (n = 0..C−1); widens W1 to [T, H, C].
    # Models must be re-trained if C changes.
    dipole_channels: int = 20
    # Radial family of the neighbor readout, all × f_c(r) (dipole_cutoff):
    #   "chebyshev"        : T_n(2r/r_c − 1)
    #   "chebyshev_warped" : T_n(2(r/r_c − 1)² − 1), NEP's warp (short-range resolution)
    #   "gaussian"         : exp(−(r − μ_n)²/2σ²), μ_n = C interior points of
    #                        linspace(0, r_c, C+2), σ = dipole_gauss_sigma_scale × spacing
    #   "bessel"           : sin((n+1)πr/r_c)/(r/r_c), spherical j₀ family
    #   "screened"         : r^(ζ−1), single channel (dipole_channels must be 1)
    dipole_radial: str = "chebyshev_warped"
    # Screened-radial exponent ζ (dipole_radial="screened" only); ζ<0 = Coulomb-like
    # suppression of long (H-bond) pairs, ζ>0 inverts the covalent/H-bond weighting.
    dipole_zeta: float = -2.0
    # Gaussian-window width σ = scale × r_c/(C+1) (dipole_radial="gaussian" only).
    dipole_gauss_sigma_scale: float = 1.0
    # Long-range field channel (target_mode=1): one extra kernel channel carrying
    # the Ewald field E_i of fixed reference charges (dipole_field_charges), so
    # μ ⊃ Σ_i s_i^E·E_i. Staged at data assembly only; in-graph V builders refuse.
    dipole_field: bool = False
    dipole_field_charges: dict | None = None        # e.g. SPC/E {8: -0.8476, 1: 0.4238}
    # Scale on E to match the radial channels; None = derive once at the first
    # data assembly as RMS(Σ_t' V)/RMS(E) and pin here (one constant per model).
    dipole_field_scale: float | None = None
    # Cutoff f_c of the neighbour readout: "cosine2" = ((1+cos(πr/r_c))/2)² (C³ at
    # r_c), "cosine" = (1+cos(πr/r_c))/2 (plain NEP, C¹). None = "cosine2".
    dipole_cutoff: str | None = "cosine"
    # Species-pair coefficient table of the neighbor readout: "off" = c ≡ 1 frozen;
    # "symmetric" = learnable c [C, T(T+1)/2], unordered pairs; "ordered" = learnable
    # Ω [C, T²] over (t_center, t_neighbor) — breaks the difference-form antisymmetry.
    # Legacy bool: False→"off", True→"symmetric".
    dipole_species_pairs: str = "symmetric"
    # Head output: "vector" = one site value per radial channel, s [C]; "scalar" =
    # one value broadcast over C (radial resolution then lives only in c[n, pair],
    # so pair with dipole_species_pairs != "off").
    dipole_head: str = "scalar"
    # --- GAP-mode dipole fit (gap_dipole.py) -----------------------------
    # FPS-selected sparse environments per species for the kernel basis.
    gap_sparse_per_species: int = 500
    # Power(s) ξ of the normalized SOAP kernel (D̂_i·D̂_m)^ξ: an int, or a list →
    # one column block per ξ (sum-of-kernels fit). Entries ints ≥ 1.
    gap_kernel_power: int | list[int] = 2
    # Second SOAP length scale for the KERNEL only: a second builder with
    # rcut_hard = gap_extra_rcut (same soft/hard buffer), its own FPS sets and extra
    # Φ column blocks (scales never cross; v unchanged). None = single scale.
    # Multi-scale SOAP: Willatt/Musil/Ceriotti PCCP 2018.
    gap_extra_rcut: float | None = None

    # ── 2. Descriptor (SOAP-turbo) ──

    # --- geometric parameters ------------------------------------------
    l_max: int = 4
    alpha_max: int = 4
    rcut_hard: float = 4.0
    rcut_soft: float = 3.5
    basis: str = "poly3"
    scaling_mode: str = "polynomial"
    radial_enhancement: int = 0
    # "trivial" : soap_turbo's compression (the only mode the GPU builder implements)
    # "none"    : uncompressed power spectrum, N(N+1)/2·(l_max+1), N = num_types·alpha_max
    #             (TurboGAP's default). On quippy the keyword is OMITTED from the
    #             descriptor string: get_compress_indices rejects "none"
    #             (soap_turbo_compress.f90:292).
    # "linear"  : quippy's linear compression, dim = 2·num_types·(l_max+1)
    compress_mode: str = "none"
    atom_sigma_r: float = 0.3
    atom_sigma_t: float = 0.5
    atom_sigma_r_scaling: float = 0.0
    atom_sigma_t_scaling: float = 0.0
    amplitude_scaling: float = 1.0
    central_weight: float = 1.0

    # --- compute backend (CPU/quippy vs GPU/TF) ------------------------
    # Descriptor backend: 0 = quippy (Fortran, CPU), 1 = native TF/NumPy (GPU).
    # GPU path supports basis="poly3" and compress_mode "trivial"/"none" only.
    descriptor_mode: int = 1

    # GPU descriptor compute precision: "float64" (matches Fortran reference)
    # or "float32" (~half VRAM, faster on consumer GPUs, looser agreement).
    # No effect for descriptor_mode=0.
    descriptor_precision: str = "float64"

    # Structures per SOAP graph call: 1 = per-frame (lowest VRAM), int>1 =
    # batched, None = auto-fit to memory budget. Used by training and
    # trajectory inference; process_trajectory kwarg overrides per call.
    descriptor_batch_frames: int | None = 100

    # Pair-tile size for the gradient compute: 0 = single-shot (higher peak
    # VRAM), >0 = tile pairs (cuts VRAM; at fp32 also enables XLA fusion).
    # ~8000 suits ~600-atom systems.
    descriptor_pair_tile_size: int = 8000

    # GPU memory budget (bytes) for the auto-sizer when
    # descriptor_batch_frames is None. None = builder default (6 GiB).
    descriptor_memory_budget_bytes: int | None = None
    # Learnable radial basis (descriptor_mode=1 only): None = fixed SOAP-turbo
    # basis; K ≥ alpha_max = builder emits raw coefficients over K orthonormal
    # poly3 functions, learned combinations M_radial[T, S, alpha_max, K] in the
    # SNES vector (NEP's c_nk; init reproduces the fixed basis). Neighbor dipole
    # readout only — gradient-consuming modes raise NotImplementedError.
    radial_basis_size: int | None = None
    # radial_basis_size only. False = SOAP-turbo species-blocked layout (n_max =
    # S·alpha_max). True = NEP-style: neighbour-species axis summed into alpha_max
    # shared channels, full n ≤ n' spectrum, dim_q = alpha_max(alpha_max+1)/2·(l_max+1).
    radial_basis_species_sum: bool = False
    # radial_basis_size only: True = one M per angular block l (NEP's angular c_nk),
    # L× the coefficients; False shares one M across l.
    radial_basis_per_l: bool = False
    # Power-spectrum footing (radial_basis_size only): "l2" = q/‖q‖ (SOAP-turbo,
    # magnitude discarded); "none" = raw × scale² (quadratic in the density);
    # "sqrt" = sign(q)·√|q| × scale (degree 1, like NEP's radial block).
    radial_basis_norm: str = "l2"
    # Descriptor scale preconditioning: per-channel scale m and shift μ folded into
    # the input layer (W0_eff = Uᵀ·diag(m)·W0, b0_eff = b0 − (m∘μ)·W0: the ANN sees
    # m∘(q − μ)); model class and q/∂q/∂R unchanged, only the search geometry
    # (per-channel step σ·m). Stored weights keep the plain init on the scaled
    # input, so gen 0 is NOT the "off" model (dividing the init by m saturates
    # tanh and inflates L1/L2 by 1/m). m from the gen-0 per-channel std s_k over
    # training atoms: group factor g = 1/median(s_group), times in "channel" mode
    # clip((1/(g·s_k))^0.5, 1/10, 10); dead channels keep m = 1.
    #   "off"     : no scaling or centring
    #   "block"   : group factor only (SOAP l groups + each extra block)
    #   "channel" : group factor × per-channel residual
    #   "minmax"  : GPUMD q_scaler — 1/(max−min) per channel, no centring
    descriptor_scale: str = "off"
    # Runtime: scale m [num_types, dim_q] from the training set, saved in the model config.
    q_scale: list[list[float]] | None = None
    # Runtime: centring μ [num_types, dim_q] (gen-0 descriptor mean), folded into b0.
    q_shift: list[list[float]] | None = None
    # GPUMD NEP descriptors, exact to nep.cu / nep_utilities.cuh (nep_descriptor.py),
    # contracted in-model by learnable c. Defaults = GPUMD's (parameters.cu:83-90).
    #   "off" | "concat" q = [q_soap ‖ q_nep] | "replace" q = q_nep | "sum" summed experts
    nep_descriptor: str = "off"
    nep_n_max_radial: int = 6            # q_n, n = 0..n_max
    nep_n_max_angular: int = 6
    nep_basis_size_radial: int = 10       # Chebyshev k = 0..basis_size
    nep_basis_size_angular: int = 10
    nep_l_max: int = 4                   # GPUMD: the only supported value
    nep_four_body: bool = True           # GPUMD L_max_4body = 2 → q_n222
    nep_five_body: bool = True         # GPUMD L_max_5body = 1 → q_n1111; needs nep_four_body
    nep_rc_radial: float = 6.0
    nep_rc_angular: float = 4.0
    nep_num_neurons: int = 10            # "sum" mode only: width of ANN_2b / ANN_3b
    # Per-expert amplitudes (δ_soap, δ_2b, δ_3b) of the summed head (nep_descriptor=
    # "sum" only). Redundant with rescaling each W1 — meaningful only because δ is
    # excluded from the L1/L2 penalty on W1 (an unpenalised gate per body order).
    sum_delta: list[float] = [1.0, 1.0, 1.0]
    sum_delta_learn: bool = False        # True → the three δ join the trained params

    # ── 3. Network architecture ──

    # Hidden-layer width of the per-type ANN.
    num_neurons: int = 10
    # Hidden activation. Modes 1/2 have a hand-coded backward: "tanh"/"swish"
    # ("silu") only; mode 0 accepts any tf.keras.activations name.
    activation: str = 'tanh'

    # Learnable per-species-pair linear mixing U_pair between the SOAP descriptor
    # and the ANN (identity-init; analogue of NEP's c_nk).
    descriptor_mixing: bool = True
    # Per-central-type U_pair [T, num_pairs, max_bs, max_bs] (T× the params).
    descriptor_mixing_per_type: bool = False
    # Parameterisation of each mixing block (descriptor_mixing only):
    #   "off"    : unconstrained, bs² params
    #   "cayley" : U = (I−A)(I+A)⁻¹, bs(bs−1)/2 params, no −1 eigenvalue
    #   "expm"   : U = exp(A), surjective onto SO(n), bs(bs−1)/2 params (recommended)
    descriptor_mixing_regularizer: str = "expm"

    # ── 4. Loss & regularisation ──

    # Master switch for L1/L2 (False = zero penalty; lambda values kept for re-enabling).
    toggle_regularization: bool = True
    # L1/L2 strength: None = auto sqrt(dim·1e-6/num_types), -1.0 = dynamic (see
    # lambda_* below), float = fixed, list = per model in (soap, 2b, 3b) order.
    # Each model's penalty is normalised by its own parameter count.
    lambda_1: float | list[float] | None = 0.005
    lambda_2: float | list[float] | None = 0.005
    # Dynamic-λ controls (lambda == -1 only): target reg/data-RMSE ratio, step
    # exponent d in λ_new = λ·(target/r)^d, gens between adaptations, clamp.
    lambda_target_ratio: float = 0.05
    lambda_damping: float = 0.2
    lambda_adapt_interval: int = 100
    lambda_min: float = 1e-8
    lambda_max: float = 1.0
    # Per-type regularization + fitness ranking (GPUMD NEP4 style), driving
    # per-type natural-gradient updates. Auto-disabled for single-element systems.
    per_type_regularization: bool = True

    # ── 5. SNES optimiser ──

    # --- core ----------------------------------------------------------
    # "snes" (evolution strategy, NEP default) or "adam" (autodiff through the
    # forward: ANN, c_pair, mixing residual as dense V, M_radial; one step per
    # chunk or sampled batch; no checkpoint/resume). num_generations, batch_size,
    # val_interval, patience and L1/L2 apply to both.
    optimizer: str = "snes"
    # Adam learning rate (optimizer="adam" only).
    adam_lr: float = 1e-3
    # Number of samples made in each train generation
    pop_size: int | None = 100
    # Number of training generations (number of updates to the model)
    num_generations: int = 100000
    # Number of structures used in each train step (None = full batch)
    batch_size: int | None = None
    # Learning rate for sigma (None = auto from canonical SNES heuristic)
    eta_sigma: float | None = None
    # Initial distribution standard deviation
    init_sigma: float = 0.02
    # Antithetic ± pairs. GPUMD does NOT (snes.cu draws independent normals). Pairs
    # halve the mean-gradient variance but share s², so the σ natural gradient
    # Σ_pairs (s² − 1)(u₊ + u₋) cancels where the loss is locally linear and σ
    # stays near init_sigma. False reproduces GPUMD.
    mirrored_sampling: bool = False
    # GPUMD's data procedure (fitness.cu report_error): every frame of data_path
    # trains, no validation split; the external test set is monitored only, never
    # selected on (no best-ever model, so patience is inert). Requires test_data_path.
    gpumd_path: bool = True
    # Lower bound on σ after each SNES update (None or 0 = off); 1e-6 is
    # non-distorting (adapted σ is ~1e-3..1e-1).
    sigma_floor: float | None = 1e-8

    # --- validation ----------------------------------------------------
    # Number of structures in each validation step (None = use entire val set)
    val_size: int | None = None
    # Validate every N generations (1 = every gen, 10 = every 10th, etc.)
    val_interval: int = 100

    # --- early stopping ------------------------------------------------
    # Early stopping patience in val ticks (None = disabled)
    patience: int | None = None

    # ── 6. Memory & I/O staging ──

    # --- per-chunk compute ---------------------------------------------
    # SNES candidates per GPU chunk: 10 = safe on a 12 GB GPU; None = whole population (A100-class).
    population_chunk_size: int | None = 10
    # Structures per GPU chunk in evaluation (None = all at once; CSC CPU profile uses 500).
    batch_chunk_size: int | None = 2000

    # Static training tensors on host CPU (True: per-chunk H2D gather, for datasets
    # too big for VRAM) or on /GPU:0 (False: fastest when they fit).
    pin_data_to_cpu: bool = False

    # --- HPC / Slurm mode ----------------------------------------------
    # CSC / Slurm mode (Mahti, Puhti, LUMI): _apply_csc_overrides tunes chunk sizing per profile.
    csc_enable: bool = False

    # CSC profile (csc_enable only): "auto" = hardware detection; "cpu" = pop_chunk
    # 10, batch_chunk 500, pin=True; "gpu" = pop_chunk None, batch_chunk 2000,
    # pin=False. Canonical impl in MasterTNEP._apply_csc_overrides — keep in sync.
    csc_profile: str = "auto"

    # ── 7. Output & diagnostics ──

    # Save model after training (None = disabled; "auto" = auto-generate run directory)
    save_path: str | None = "models/auto"
    # Save plots (None = disabled; set automatically by setup_run_directory)
    save_plots: str | None = None
    # Show plots interactively (True = plt.show(), False = close after saving)
    show_plots: bool = False
    # Periodic plotting interval (None = disabled; int = plot every N generations)
    plot_interval: int | None = None

    # Rolling checkpoint every N gens to {save_path}/checkpoint.h5 (cfg, SNES state,
    # best-val, history, RNG; resume via train_model(checkpoint=path)). None = off.
    checkpoint_interval: int | None = 2000

    # Show extra info in progress bar (L1, L2 regularisation)
    debug: bool = False

    # ── 8. Runtime state (populated by MasterTNEP after data load) ──

    dim_q: int
    num_types: int
    types: list[int] = []
    type_map: dict = {}
    indices: np.ndarray

    def __init__(self) -> None:
        # Own containers per instance: class-level list/dict defaults are shared.
        self.types = []
        self.type_map = {}
        if type(self).allowed_species is not None:
            self.allowed_species = list(type(self).allowed_species)

    def randomise(self, dataset: list) -> None:
        """Shuffle dataset indices into self.indices [total_N] (truncated to total_N)."""
        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(dataset), dtype=int)
        rng.shuffle(indices)
        if self.total_N is not None:
            self.indices = indices[:self.total_N]
        else:
            self.indices = indices


def neighbor_readout_on(cfg: "TNEPconfig | GAPconfig") -> bool:   # GAPconfig: gap_config.py
    """THE neighbour-readout predicate: mode 2, or mode 1 with dipole_readout="neighbor"."""
    return int(cfg.target_mode) == 2 or (
        int(cfg.target_mode) == 1 and str(cfg.dipole_readout) == "neighbor")


def bond_rank(cfg: "TNEPconfig | GAPconfig") -> int:
    """Rank of the bond tensor in the neighbour readout's geometry kernel:
    1 → d [3] (dipole, one kernel group), 2 → Voigt [6] (polarisability, two
    kernel groups: isotropic |d|²I and traceless d⊗d − |d|²I/3)."""
    return 2 if int(cfg.target_mode) == 2 else 1


def kernel_channels(cfg: "TNEPconfig | GAPconfig") -> int:
    """Channel axis of the staged kernel V[N, D, Ck, T]: the C radial
    channels per kernel group, group-major (Ck = C·bond_rank), plus the
    field channel under dipole_field (last channel, mode 1 only)."""
    return int(cfg.dipole_channels) * bond_rank(cfg) + (
        1 if bool(getattr(cfg, "dipole_field", False)) and int(cfg.target_mode) == 1 else 0)


def resolved_dipole_rc(cfg: "TNEPconfig | GAPconfig") -> float:
    """Readout cutoff r_c: the cutoff of the descriptor feeding the model —
    nep_rc_radial under nep_descriptor="replace", else rcut_hard. Single source for
    TNEP.__init__ and data._neighbor_dipole_V; accepts either config."""
    if str(getattr(cfg, "nep_descriptor", "off")) == "replace":
        return float(cfg.nep_rc_radial)
    return float(cfg.rcut_hard)


def resolved_dipole_cutoff(cfg: TNEPconfig | "GAPconfig") -> str:
    """THE dipole_cutoff=None rule (None → "cosine2"); single source for TNEP.__init__
    and data._neighbor_dipole_V. Accepts either config."""
    return cfg.dipole_cutoff or "cosine2"
