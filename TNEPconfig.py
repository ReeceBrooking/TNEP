from __future__ import annotations

import numpy as np


class TNEPconfig:
    """Holds all hyperparameters and runtime state for a TNEP training run.

    Class-level defaults are overwritten at runtime by MasterTNEP after
    data loading (num_types, types, dim_q, indices).

    Sections (in natural pipeline order — what's the data → how features
    are computed → model → loss → optimiser → memory plumbing → outputs):

        1. Dataset & targets
        2. Descriptor (SOAP-turbo: geometry, backend, preprocessing)
        3. Network architecture
        4. Loss & regularisation
        5. SNES optimiser (core, plateau-reset, validation)
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
    # Native units of dipole targets in the dataset. Used (a) to derive
    # the e·Å conversion factor when `convert_dipole_to_eangstrom=True`,
    # and (b) as the plot-axis / stats unit label when conversion is off.
    # "e*angstrom" = e·Å
    # "e*bohr"     = e·a₀   (× 0.5292 → e·Å)
    # "debye"      = Debye  (× 0.2082 → e·Å)
    dipole_units: str = "e*bohr"
    # When True, dipole targets are converted to e·Å on data load and
    # training / plots / spectra all use e·Å. When False, raw dataset
    # values are passed through unchanged — model trains in dataset
    # native units. Useful when you want loss / RMSE / RRMSE numbers
    # directly comparable with reference values quoted in e·a₀ or Debye.
    convert_dipole_to_eangstrom: bool = False
    # Polarizability off-diagonal weight (target_mode=2 only): loss for
    # components [xy, yz, zx] scaled by lambda_shear^2. GPUMD default 1.0
    # — equal weighting; <1.0 downweights off-diagonal.
    lambda_shear: float = 1.0

    # Power of |r_ij| applied to per-pair forces in the dipole contraction
    # (target_mode=1 only). Two algebraic branches:
    #
    #   N = 0   →  μ = − Σ_i de_dq[i] · grad_values[i, i]
    #             (Sums ONLY the self-pair (i, i) contributions.
    #             A naive μ = -Σ F_ij over all pairs would be identically
    #             zero by translation invariance of q_i, since
    #             Σ_{all j incl. self} ∂q_i/∂R_j = 0. Restricting the sum
    #             to self pairs isolates the centre's own gradient, which
    #             equals −Σ_{j≠i} ∂q_i/∂R_j and is non-zero. The result
    #             is rotation-covariant and a valid dipole-like quantity,
    #             but DIFFERENT in functional form from the N ≥ 1 branches —
    #             not a "lower-power" version of the same formula.)
    #
    #   N = 1   →  μ = − Σ_{pair} |r_ij|   · F_ij  (first radial moment;
    #             Schofield-style virial-like weight)
    #   N = 2   →  μ = − Σ_{pair} |r_ij|²  · F_ij  (Xu et al. JCTC 2024 /
    #             GPUMD default — what the dipole pathway has used to date)
    #   N ≥ 3   →  μ = − Σ_{pair} |r_ij|^N · F_ij  (higher moments;
    #             mostly diagnostic / sensitivity studies)
    #
    # For N ≥ 1, self pairs (i, i) are present in the COO list but
    # contribute zero automatically because |r_ii|^N = 0 — no separate
    # handling needed. The N = 0 branch uses the COO list's self entries
    # exclusively and skips all neighbour pairs.
    #
    # Default 0 enables the self-pair-only formulation. For the
    # standard Xu et al. JCTC 2024 / GPUMD-compatible formula, set
    # `dipole_rij_power = 2`; for the GAP-style first radial moment,
    # set it to 1. Models saved with one N MUST be re-trained if N is
    # changed — the contraction defines what the network's per-atom
    # scalar is mapped to.
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
    # Descriptor backend: 0 = quippy (Fortran, CPU), 1 = native TF/NumPy (GPU when available).
    # The GPU path supports basis="poly3" and compress_mode="trivial" only;
    # falls back to a clear error if other settings are requested.
    descriptor_mode: int = 0

    # Internal precision for the GPU descriptor compute. The Fortran reference
    # uses double-precision throughout; "float64" mirrors that exactly. The
    # opt-in "float32" path keeps roughly half the VRAM and runs faster on
    # consumer GPUs (which are 2-32× more performant in fp32 than fp64), at
    # the cost of slightly looser agreement with quippy. Trajectory-inference
    # outputs are always cast to float32 at the boundary regardless of this
    # setting, so the user-visible difference is dominated by accumulation
    # noise in the radial recursion. Has no effect for descriptor_mode=0.
    descriptor_precision: str = "float64"

    # Number of structures concatenated into a single SOAP graph call.
    #   1     : per-frame (lowest VRAM, highest launch overhead)
    #   int>1 : multi-frame batching, amortises kernel launches
    #   None  : auto — choose the largest batch that fits the memory budget
    # Used by both training (DescriptorBuilder.build_descriptors) and
    # trajectory inference. process_trajectory's `descriptor_batch_frames`
    # kwarg overrides this value for that call only.
    descriptor_batch_frames: int | None = 500

    # Pair-tile size for the gradient compute. 0 = single-shot (lower
    # launch overhead, higher peak VRAM). >0 = tile pairs in chunks of
    # this size (cuts peak VRAM at the dominant [k_max,n_max,P] tensors;
    # at fp32 also auto-enables XLA fusion in the trajectory path).
    # Sensible value for ~600-atom systems: 8000.
    descriptor_pair_tile_size: int = 8000

    # GPU memory budget (bytes) used by the auto-sizer when
    # descriptor_batch_frames is None. None falls back to the builder's
    # default (6 GiB).
    descriptor_memory_budget_bytes: int | None = None

    # Number of parallel workers for SOAP descriptor computation.

    # ═══════════════════════════════════════════════════════════════════
    # 3. NETWORK ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════

    # Hidden-layer width of the per-type ANN.
    num_neurons: int = 30
    # Activation function for the hidden layer. Any name accepted by
    # `tf.keras.activations.get` works for the forward pass. For
    # dipole / polarisability training (target_mode = 1 or 2), the
    # backward derivative is hand-coded — only `tanh` and `swish`
    # (alias `silu`) are fully plumbed. Other activations will raise
    # NotImplementedError at the first force / dipole prediction.
    # Energy training (target_mode = 0) doesn't use the hand-coded
    # backward and works with any Keras activation.
    activation: str = 'swish'

    # When True, insert a learnable per-species-pair linear mixing
    # layer between the (fixed) SOAP-turbo descriptor and the ANN.
    # `desc'[q in block_ab] = U_pair[a,b] @ desc[q in block_ab]` per
    # unordered neighbour-species pair (a, b). Trivial compression
    # makes the block sizes pair-dependent (20 / 35 at alpha_max=4,
    # l_max=4, 3 species) and the q-indices non-contiguous; the
    # implementation gathers/scatters accordingly. U_pair is shared
    # across central atom types — the per-type ANN already
    # differentiates downstream. Identity init so the model starts
    # bit-identical to the no-mixing baseline. This is the closest
    # direct analog to GPUMD/NEP's trainable radial-basis coefficients
    # c_nk (which mix fixed Chebyshev primitives into learned radial
    # functions per species pair).
    descriptor_mixing: bool = True
    # When True (and descriptor_mixing=True), U_pair becomes
    # per-central-atom-type: shape [T, num_pairs, max_bs, max_bs]
    # instead of [num_pairs, max_bs, max_bs]. Each central type t
    # gets its own learned set of pair-mixing matrices, applied to
    # atoms of that type. Strictly more expressive than the shared
    # variant (which is the T=1 case of this) — captures central-
    # type-specific feature selection on top of the pair-block
    # decomposition. Cost: T× the U_pair param count. Identity-init
    # per (t, p) so the model starts bit-identical to the mixing-
    # disabled baseline regardless of T.
    descriptor_mixing_per_type: bool = False
    # Parameterisation applied to each V_pair descriptor-mixing block.
    # Both non-off modes are STRUCTURAL constraints (not soft penalties):
    # SNES walks the upper-triangle of a skew-symmetric A and U is
    # reconstructed from A by a closed-form map. U is exactly orthogonal
    # regardless of any λ.
    #   "off"    : V_pair is unregularised. Each block has bs² trainable
    #              parameters; SNES sigma bounds exploration.
    #   "cayley" : U = (I − A)(I + A)⁻¹  — rational chord. Cannot
    #              represent rotations with a −1 eigenvalue except in
    #              the limit |A| → ∞. bs·(bs−1)/2 free params per block.
    #   "expm"   : U = exp(A)              — exponential geodesic.
    #              Surjective onto SO(n); no Jacobian singularity.
    #              bs·(bs−1)/2 free params per block. RECOMMENDED.
    # Only consulted when descriptor_mixing=True.
    descriptor_mixing_regularizer: str = "expm"

    # Descriptor preprocessing contraction layer. Sits BEFORE the W0
    # layer of the per-type ANN; output becomes the new descriptor
    # input. A learned per-(centre type, raw channel) coefficient
    # table contracts the chosen axis of the raw SOAP descriptor
    # down to a smaller feature vector. NEP-inspired but applied
    # AFTER the SOAP power-spectrum squaring (vs NEP's pre-squaring
    # projection on the density coefficients).
    #
    # Modes (Q_raw = raw SOAP dim, L = l_max+1):
    #   "off"          : (default) no preprocessing; W0 sees raw Q_raw.
    #   "angular"      : Contract over l per (pair, n_pair). l < l_keep
    #                    are kept as passthrough channels; l ≥ l_keep
    #                    summed into one output channel. Output dim
    #                    Q_new = (l_keep + 1 if l_keep < L else 0) ·
    #                    Σ_pair α_eff_per_pair. Coefficients [T, Q_raw].
    #   "species_pair" : Per-central-type contraction into SELF (the
    #                    (t,t) pair) vs OTHER (all (t, j ≠ t) pairs
    #                    summed). Output dim Q_new = 2 · max_α · L.
    #                    Coefficients [T, Q_raw], pair-masked.
    #   "both"         : Collapse BOTH axes — combines "angular" and
    #                    "species_pair". Smallest Q_new; biggest info
    #                    loss; per-(block, l_group) per-type init.
    #   "nep4_radial"  : NEP4-faithful learned-basis fold (rank-1
    #                    outer-product weighting on (n, n')):
    #                      g[t, n'', l] = Σ_{n,n'} c[t,s(n),n'',k(n)]
    #                                          · c[t,s(n'),n'',k(n')]
    #                                          · p[n, n', l]
    #                    Mathematically equivalent to a NEP4 descriptor
    #                    with SOAP-turbo's radial basis as primitives.
    #                    Output dim Q_new = n_max_out · L; n_max_out
    #                    defaults to Q_raw/L so the descriptor dim is
    #                    PRESERVED (pure non-linear transformation).
    #                    Coefficients shape [T_centre, T_neighbour,
    #                    n_max_out, α] — same indexing as NEP4's
    #                    c^{Z_i,Z_j}_{n'',k}. Requires
    #                    compress_mode='trivial'.
    #
    # Descriptor mixing composes with all preprocess modes.
    descriptor_preprocess_contract: str = "off"
    # NEP4 learned-basis fold output radial-channel count. Only consulted
    # when descriptor_preprocess_contract == "nep4_radial". When None
    # (default), the layout auto-picks n_max_out = Q_raw / L so that
    # Q_new = Q_raw (descriptor dim is preserved — pure non-linear
    # transformation). Set explicitly to reduce / expand the descriptor
    # — small values (≈ α or 2α) match NEP4's typical "compact learned
    # basis" setting and give the strongest inductive bias.
    descriptor_nep4_n_max_out: int | None = None
    # Initialisation for preprocess coefficients:
    #   "mean"   : (default) 1/N where N is the contracted-axis size.
    #              For angular mode N = l_max+1, so each coefficient is
    #              1/(l_max+1); gen-0 output ≈ mean across l per channel.
    #              Output magnitude similar to inputs — well conditioned.
    #   "sum"    : 1.0. Gen-0 output = sum across the contracted axis.
    #              Output magnitude grows with N; W0 has to re-scale.
    #   "glorot" : Glorot-uniform: U(-√(6/(fan_in+fan_out)), +√(...)).
    descriptor_preprocess_init: str = "mean"
    # Per-tail σ scaling for preprocess coefficients (analogous to
    # mixing_sigma_scale). Default 1.0 = same as the ANN. Reduce
    # (e.g. 0.1) if SNES sampling noise on preprocess coefficients
    # overwhelms the optimisation signal.
    preprocess_sigma_scale: float = 1.0
    # Angular contraction threshold: l < angular_l_keep are kept as
    # passthrough output channels (no learnable coefficient — the
    # descriptor channel goes straight to its own W0 row). l ≥
    # angular_l_keep are summed into ONE output channel per
    # (pair, n_pair) with L − angular_l_keep learnable coefficients.
    # Default 1 reproduces the original behaviour (l=0 kept, l>0 summed).
    # Only consulted in modes that collapse the l axis ("angular", "both").
    descriptor_preprocess_angular_l_keep: int = 2
    # When True (default), W_pre coefficients are per central-atom type
    # (shape [T, n_summed_q_raw, N]). When False, coefficients are
    # GLOBAL across centre types — symmetric coupling, smallest param count.
    descriptor_preprocess_per_type: bool = False
    # L1/L2 regularisation strengths on (coefficient − init). Penalises
    # deviation from the mean/sum/glorot init (not from zero). Set both
    # to 0.0 to disable; defaults below give a mild soft prior toward
    # the chosen init.
    descriptor_preprocess_lambda_1: float = 0.0005
    descriptor_preprocess_lambda_2: float = 0.0005

    # ═══════════════════════════════════════════════════════════════════
    # 4. LOSS & REGULARISATION
    # ═══════════════════════════════════════════════════════════════════

    # Master switch for the L1/L2 regularisation block. When False:
    # both the per-type fitness path and the main fitness add 0
    # regularisation regardless of lambda_1 / lambda_2 / per-type
    # settings. lambda values are still maintained (and reported in
    # history) so flipping back to True mid-run resumes seamlessly.
    # Useful for ablation studies and for early "warm-up" generations
    # where the data fit dominates.
    toggle_regularization: bool = True
    # L1/L2 regularization strengths.
    #   None  : auto = sqrt(dim * 1e-6 / num_types)
    #   -1.0  : dynamic — adapt every `lambda_adapt_interval` gens so
    #           the L1 (or L2) penalty stays at `lambda_target_ratio`
    #           of the data RMSE. Starts from the auto value.
    #   float : fixed scalar
    lambda_1: float | None = 0.0005
    lambda_2: float | None = 0.0005
    # Dynamic-λ controls (only used when lambda_1 or lambda_2 == -1).
    # `target_ratio`: target ratio of reg-penalty to data RMSE. 0.05
    #   means "keep regularisation at ~5% of the data loss." GPUMD
    #   NEP4 uses a similar target (~0.01–0.1 depending on data size).
    # `damping`: multiplicative step exponent. λ_new = λ * (target/r)^d.
    #   Smaller d → slower adaptation, less oscillation. 0.2 is gentle.
    # `interval`: how often (in gens) to recompute and rescale λ.
    #   Matches the existing per-100-gen reg-sampling cadence by default.
    # `min/max`: safety clamp on adapted λ.
    lambda_target_ratio: float = 0.05
    lambda_damping: float = 0.2
    lambda_adapt_interval: int = 100
    lambda_min: float = 1e-8
    lambda_max: float = 1.0
    # Per-type regularization and ranking (GPUMD NEP4 style).
    # Each type's params are regularized separately, creating per-type fitness
    # rankings that drive per-type natural gradient updates.
    # Only effective for multi-element systems (auto-disabled for single-element).
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
    # Lower bound on sigma after each SNES update. Without a floor,
    # `σ ← σ · exp(η · grad_σ)` can drift toward zero on a long run
    # (especially with the per-type ranking schedule), collapsing the
    # search distribution to a point and silently killing exploration.
    # Set to None or 0 to disable. Typical adapted σ is 1e-3 to 1e-1,
    # so 1e-6 is non-distorting.
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
    # Number of SNES candidates to evaluate per GPU chunk (limits VRAM).
    # Defaults:
    #   10    : Safe baseline that fits a 12 GB consumer GPU (the dev
    #           box) without OOM under typical molecule sizes. Shipping
    #           default since "VRAM-safe" beats "fastest" for new users.
    #   None  : No chunking — the whole population is evaluated in one
    #           shot. Use on A100 40 GB and similar; auto-applied when
    #           cfg.csc_enable=True and the GPU profile is selected.
    population_chunk_size: int | None = 10
    # Number of structures to process per GPU chunk during evaluation.
    # None = all at once. Default 2000 fits the dev box; the CPU CSC
    # profile drops this to 500 because CPU forward passes prefer
    # smaller chunks (per-op latency >> per-FLOP rate vs GPU).
    batch_chunk_size: int | None = 2000

    # Where the static training tensors (descriptors, grad_values,
    # positions, pair indices, etc.) live, and which chunk-staging
    # path the SNES eval / TNEP.score loops use:
    #
    #   True  : tensors stay on host CPU. Per-chunk slice → tf.gather
    #           → implicit H2D copy each gen. Use when the full
    #           dataset is too big for VRAM.
    #   False : tensors live on /GPU:0 — including grad_values, which
    #           is loaded fully onto the GPU at startup. The
    #           chunk-staging path becomes pure on-device
    #           gather/strided_slice — no host round-trip. Fastest
    #           mode when the working set (grad_values + descriptors +
    #           activations) fits in VRAM.
    pin_data_to_cpu: bool = False

    # --- HPC / Slurm mode ----------------------------------------------
    # Master switch for CSC / Slurm-supercomputer mode (Mahti, Puhti,
    # LUMI etc.). When True, `_apply_csc_overrides` tunes the per-chunk
    # sizing (population_chunk_size / batch_chunk_size / pin_data_to_cpu)
    # for the detected profile. Grad_values stays in RAM (or VRAM,
    # depending on `pin_data_to_cpu`).
    # Default False keeps non-HPC behaviour identical.
    csc_enable: bool = False

    # Force which CSC profile (`_apply_csc_overrides`) applies. Only
    # consulted when `csc_enable=True`.
    #   "auto" (default) — pick the profile from runtime hardware
    #                      detection (`_has_gpu` in MasterTNEP).
    #   "cpu"            — force the CPU profile even if a GPU is
    #                      visible. Useful for benchmarking the CPU
    #                      path on a GPU-equipped dev box.
    #   "gpu"            — force the GPU profile. Useful when the
    #                      detection heuristic fails (e.g. cuda
    #                      visible but the GPU is reserved).
    #
    # Profile overrides applied:
    #
    #   GPU profile (A100 40 GB, ~32 cores):
    #       population_chunk_size = None    (no chunking)
    #       batch_chunk_size      = 2000
    #       pin_data_to_cpu       = False
    #
    #   CPU profile (Mahti 128-core EPYC, no GPU):
    #       population_chunk_size = 10
    #       batch_chunk_size      = 500     (CPU latency-sensitive)
    #       pin_data_to_cpu       = True    (no GPU to upload to)
    #
    # See `_apply_csc_overrides` in MasterTNEP.py for the canonical
    # implementation — when these defaults are tuned, update both sides.
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

    # Periodic training checkpoint interval. None = no checkpointing.
    # int = write a rolling checkpoint to `{save_path}/checkpoint.h5`
    # every N generations, overwriting any previous checkpoint at that
    # path. The checkpoint embeds the full cfg, current SNES distribution
    # (mu, sigma), best-val state, full history, RNG state, and last
    # completed gen — enough to resume identically via
    # `train_model(checkpoint=path)`. Requires `save_path` to be set;
    # warned and skipped otherwise.
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
