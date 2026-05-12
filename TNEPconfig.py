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
    allowed_species: list[int | str] | None = [6, 1, 8]
    # Species filter mode: "subset" = keep structures with only allowed species,
    # "exact" = keep structures containing exactly all allowed species
    filter_mode: str = "subset"
    # When True, drop structures with NaN positions, NaN targets, or
    # zero-vector targets. Structures with missing targets are always
    # dropped regardless (they can't be trained against).
    filter_bad_data: bool = False
    num_neurons: int = 15
    # Number of structures used in each train step
    batch_size: int | None = None
    # Number of samples made in each train generation
    pop_size: int | None = 100
    # Number of training generations (number of updates to the model)
    num_generations: int = 150000
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
    atom_sigma_r: float = 0.5
    atom_sigma_t: float = 0.5
    atom_sigma_r_scaling: float = 0.0
    atom_sigma_t_scaling: float = 0.0
    amplitude_scaling: float = 1.0
    central_weight: float = 1.0

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

    # Master switch for CSC / Slurm-supercomputer mode (Mahti, Puhti,
    # LUMI etc.). When True:
    #   - All gradient-caching / IO options are forced off
    #     (`cache_gradients_to_disk`, `chunk_prefetch`,
    #     `use_pinned_buffers`, `use_cufile`). Grad_values stays
    #     in RAM (or VRAM, depending on `pin_data_to_cpu`) — no
    #     NVMe scratch, no pinned-host pool, no cuFile / GDS. CSC
    #     nodes have ample host RAM and the cuFile compat-mode
    #     WSL path doesn't generalise to their kernel / filesystem
    #     stack.
    #   - The Slurm-specific scratch-dir resolver
    #     (`MasterTNEP._resolve_scratch_dir`) is allowed to consult
    #     `$SLURM_TMPDIR` / `$TMPDIR` / `$LOCAL_SCRATCH`. With this
    #     flag False those env-vars are ignored even if set, so a
    #     local dev environment that happens to define `TMPDIR`
    #     doesn't accidentally land scratch there.
    # Default False keeps non-HPC behaviour identical.
    csc_enable: bool = False

    # When True, the bulky grad_values COO tensor is written to a
    # temporary directory on disk (created next to the working directory
    # so it lands on the same filesystem — NVMe in typical setups) and
    # accessed via numpy memory-map. The directory is automatically
    # removed when training ends. For large datasets (S > ~3000 organic
    # structures) grad_values is the dominant memory term — putting it
    # on disk cuts host RAM footprint to <500 MB while preserving
    # precomputed-mode speed (per-chunk disk reads at NVMe sequential
    # bandwidth ~5-10 ms, vs ~50-100 ms/gen for the rest of the
    # training step).
    cache_gradients_to_disk: bool = False

    # Overlap disk → GPU staging of chunks N+1..N+prefetch_depth with
    # GPU evaluation of chunk N. Up to `prefetch_depth` background
    # threads run slice_and_complete_chunk concurrently with the
    # consumer. depth=1 is the simple producer/consumer (one chunk in
    # flight); depth=2 hides both disk read and host→GPU DMA behind
    # compute; depth=3 helps further only when GPU compute > 2× disk
    # pipe. Memory cost: depth × per-chunk grad slice (~few hundred MB
    # at full-batch chunk_size=500 each). Set chunk_prefetch=False to
    # bisect threading issues.
    chunk_prefetch: bool = True
    prefetch_depth: int = 2

    # Number of pinned host buffers in the pool. Must be >=
    # prefetch_depth + 1 (one for the chunk currently held by the
    # consumer, prefetch_depth for in-flight staging). Each buffer is
    # sized to the worst-case chunk grad slice — typically a few hundred
    # MB — and is page-locked, so the total pinned RAM is
    # pinned_pool_size × buffer_bytes. Bump cautiously.
    pinned_pool_size: int = 4

    # When True and libcufile is loadable, the disk-backed gradient cache
    # is read directly from NVMe into pre-allocated GPU buffers via
    # cuFile (NVIDIA GPUDirect Storage). On systems with the nvidia_fs
    # kernel module loaded, this is true zero-copy disk→GPU DMA at full
    # NVMe bandwidth. On WSL or other systems without nvidia_fs, cuFile
    # falls back transparently to compat mode (kernel-stage buffer +
    # CUDA-managed copy) which still saturates PCIe at ~17 GB/s once the
    # page cache is warm — well above the ~3-5 GB/s pinned-host path.
    # Falls back to the pinned path silently if cuFile isn't usable.
    use_cufile: bool = True

    # Number of GPU buffers in the cuFile pool. Each is sized to the
    # worst-case chunk grad slice; total VRAM cost is
    # cufile_pool_size × buffer_bytes. Must be >= prefetch_depth + 1.
    cufile_pool_size: int = 2

    # XLA-compile the per-chunk eval (`_evaluate_chunk`). Fuses the
    # dipole-kernel pre-compute and the per-type matmul + reduction ops
    # into a single GPU kernel — typically 1.5-2× faster on Ada/Hopper.
    # Each unique (B_chunk, P_chunk) shape triggers one XLA compile
    # (~5-10 s the first time that shape is seen); for full-batch
    # deterministic chunks this is a one-shot cost paid in the first
    # generation, then steady-state runs at full XLA speed.
    eval_jit_compile: bool = False

    # Use page-locked (pinned) host buffers for the disk-backed chunk
    # staging path. With pinned source, tf.constant dispatches a true
    # async cudaMemcpyAsync (no driver bounce buffer), saturating PCIe
    # at ~12-16 GB/s instead of the ~6-8 GB/s pageable rate. Buffers
    # are allocated via cudaMallocHost; pool size = 4. Set False if
    # cudart is not loadable (rare) or to bisect a regression.
    use_pinned_buffers: bool = True

    # L1/L2 regularization strengths.
    #   None  : auto = sqrt(dim * 1e-6 / num_types)
    #   -1.0  : dynamic — adapt every `lambda_adapt_interval` gens so
    #           the L1 (or L2) penalty stays at `lambda_target_ratio`
    #           of the data RMSE. Starts from the auto value.
    #   float : fixed scalar
    toggle_regularization: bool = True
    lambda_1: float | None = 0.03
    lambda_2: float | None = 0.03
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
    # Per-type regularization and ranking (GPUMD NEP4 style)
    # Each type's params are regularized separately, creating per-type fitness
    # rankings that drive per-type natural gradient updates.
    # Only effective for multi-element systems (auto-disabled for single-element).
    per_type_regularization: bool = True

    # Early stopping patience (None = disabled)
    patience: int | None = None

    # Plateau-triggered sigma reset (IPOP-CMA-ES style, simplified for
    # SNES). When set to an int N, the search distribution's sigma is
    # re-broadened back toward `init_sigma` after N consecutive val
    # ticks without improvement on best_val_loss. This re-expands the
    # local exploration radius without abandoning best_mu, giving the
    # optimizer a chance to escape a shallow basin.
    #
    # Reference: Auger & Hansen (2005) "A Restart CMA Evolution
    # Strategy With Increasing Population Size" (CEC 2005), which sets
    # `tolstagnation = int(100 + 100·dim^1.5 / popsize)` as the
    # canonical default. For your typical dim ≈ 12k, λ = 100 that's
    # ~1.3M gens — too long to be useful. A more aggressive
    # 200–500 val ticks is more practical on NN training problems
    # where val_interval = 10. None = disabled.
    plateau_reset_patience: int | None = None
    # Multiplier applied to the **current** sigma vector at every
    # plateau reset (default 2.0 — broadens each dimension's search
    # width by 2×). This preserves the per-dimension scale structure
    # that SNES has learned — dimensions where the optimizer
    # tightened sigma stay tighter than dimensions where it didn't.
    # Soft re-broadening like this works better than hard reset in
    # high dimensions because a uniform fresh sigma loses all per-
    # dim information; with dim ~ 20k, the search would just random-
    # walk from best_mu before it could rediscover which directions
    # mattered.
    #
    # Typical values: 1.5–5.0. Try 2.0 first; if the model is deep in
    # a basin and not escaping, bump to 3.0 or 5.0.
    sigma_reset_factor: float = 2.0
    # When True, the multiplier above is applied to `init_sigma`
    # uniformly (i.e. canonical hard reset like IPOP-CMA-ES) rather
    # than to the current sigma. Loses all learned per-dim scale —
    # only set True if you have a specific reason (e.g. you want
    # IPOP-style behaviour or have determined empirically that the
    # learned sigma is corrupted).
    sigma_reset_to_init: bool = False
    # When True, restore mu to best_mu at every sigma reset (warm
    # restart around the best known position). When False (the
    # better default in high dim — see comment on sigma_reset_factor),
    # leave mu where it is so the broadened search continues from
    # the current position. The combination of (mu = best_mu) +
    # (broadened sigma) is the canonical IPOP form, but in our
    # high-dim NN setting it tends to throw away the directional
    # information that SNES has accumulated.
    plateau_restore_best_mu: bool = False
    # Cap on the number of plateau-triggered resets. None = unlimited.
    # Useful with cfg.patience to bound total wall-time: first hits
    # plateau_reset_patience trigger resets; once max_sigma_resets is
    # reached, subsequent plateaus fall through to early stopping
    # (or just continue if patience is None).
    max_sigma_resets: int | None = None

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
    # atoms of that type. Captures central-type-specific feature
    # selection on top of the pair-block decomposition — strictly
    # more expressive than the shared variant (which is the T=1
    # case of this). Cost: T× the U_pair param count
    # (for your typical T=3, that's ~15k extra params on top of
    # the shared 4.9k — bringing the descriptor-mixing layer to
    # roughly 2× the size of the per-type ANN). Identity-init per
    # (t, p) so the model still starts bit-identical to the
    # mixing-disabled baseline.
    descriptor_mixing_per_type: bool = True
    # Regulariser applied to the V_pair descriptor-mixing layer.
    #   "off"        : V_pair is unregularised (default; relies on SNES
    #                  sigma to bound exploration).
    #   "shrinkage"  : L1+L2 on V_pair using cfg.lambda_1 / lambda_2.
    #                  Pulls V → 0 ⇔ U → I — strong "no mixing" prior
    #                  that empirically collapses any learned mixing
    #                  back to the baseline. Useful for ablations only.
    #   "orthogonal" : Frobenius penalty ‖UᵀU − I‖²_F per pair block
    #                  (computed in residual form as ‖V + Vᵀ + VᵀV‖²).
    #                  Minimum is the *orthogonal group*, not identity —
    #                  every rotation/reflection of the descriptor basis
    #                  is at zero penalty. Constrains U to be length-
    #                  preserving (no scaling), preventing column
    #                  collapse and rank deficiency without anchoring at
    #                  no-mixing. Uses cfg.lambda_orth.
    descriptor_mixing_regularizer: str = "off"
    # Strength of the orthogonal-mixing regulariser (used when
    # descriptor_mixing_regularizer == "orthogonal").
    #   None : auto = sqrt(n_U_pair * 1e-6 / num_types)
    #   -1   : dynamic adaptation (see SNES._maybe_adapt_lambda)
    #   float: fixed scalar
    lambda_orth: float | None = None
    # Which optimizer drives `TNEP.fit`. "snes" reproduces the
    # canonical SNES black-box path (no gradient information, rank-
    # based population update). "adam" runs first-order Adam on
    # analytical gradients via tf.GradientTape — the same training
    # regime as the GNEP paper (Huang et al. 2025), which reports
    # ~10× fewer epochs at equal accuracy. Both paths share the same
    # data pipeline, validation, early stopping, history dict, and
    # plot outputs. Checkpointing only works for the SNES path at
    # the moment (an Adam-side checkpoint is straightforward to add
    # once the Adam path is validated end-to-end).
    optimizer: str = "snes"
    # Adam learning rate (only used when optimizer="adam"). Standard
    # default for NN regression heads; reduce to ~1e-4 if loss diverges.
    adam_learning_rate: float = 1e-3
    # Initial distribution standard deviation
    init_sigma: float = 0.1
    # Lower bound on sigma after each SNES update. Without a floor,
    # `σ ← σ · exp(η · grad_σ)` can drift toward zero on a long run
    # (especially with the per-type ranking schedule), collapsing the
    # search distribution to a point and silently killing exploration.
    # 1e-5 keeps the search alive without distorting any normal
    # adaptation (typical adapted σ is 1e-3 to 1e-1). Set to None or 0
    # to disable the floor.
    sigma_floor: float | None = 1e-5
    # Seed for randomisation
    seed: int | None = None
    # 0 : PES, 1 : Dipole, 2 : Polarizability
    target_mode: int = 1
    # Override the info/results key used to read targets from ASE structures.
    # None = use the default for target_mode ("energy", "dipole", "pol").
    # Set to a custom string to support non-standard dataset labels (e.g. "mu", "alpha").
    target_key: str | None = None
    # Test split ratio
    test_ratio: float = 0.3
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N: int | None = 4000
    # Number of structures in each validation step (None = use entire val set)
    val_size: int | None = None
    # Validate every N generations (1 = every gen, 10 = every 10th, etc.)
    val_interval: int = 1
    # Number of SNES candidates to evaluate per GPU chunk (limits VRAM usage).
    # None = no chunking (recommended for A100 at typical molecular sizes).
    # For very large systems (>50k edges per structure), start at 50 and reduce if OOM.
    population_chunk_size: int | None = 10
    # Number of structures to process per GPU chunk during evaluation (None = all at once)
    batch_chunk_size: int | None = 1000
    # Number of parallel workers for SOAP descriptor computation.
    # None = auto (reads SLURM_CPUS_PER_TASK at DescriptorBuilder init time, falls back to 1)
    # 1    = serial (current behaviour, default outside SLURM)
    # N    = use N worker processes
    num_descriptor_workers: int | None = None
    # Where the static training tensors (descriptors, grad_values,
    # positions, pair indices, etc.) live, and which chunk-staging
    # path the SNES eval / TNEP.score loops use:
    #
    #   True  : tensors stay on host CPU. Per-chunk slice → tf.gather
    #           → implicit H2D copy each gen. Required when the full
    #           dataset is too big for VRAM (the disk-backed grad
    #           cache uses pinned-host / cuFile pools to DMA chunks
    #           to GPU on demand).
    #   False : tensors live on /GPU:0 — including grad_values, which
    #           is loaded fully onto the GPU at startup (read from
    #           the disk memmap when cache_gradients_to_disk=True).
    #           The chunk-staging path becomes pure on-device
    #           gather/strided_slice — no host round-trip, no pinned
    #           pool, no cuFile. Fastest mode when the working set
    #           (grad_values + descriptors + activations) fits in
    #           VRAM.
    pin_data_to_cpu: bool = False

    # Periodic plotting interval (None = disabled; int = plot every N generations)
    plot_interval: int | None = None

    # Periodic training checkpoint interval. None (default) = no
    # checkpointing. int = write a rolling checkpoint to
    # `{save_path}/checkpoint.h5` every N generations, overwriting any
    # previous checkpoint at that path. The checkpoint embeds the full
    # cfg, current SNES distribution (mu, sigma), best-val state,
    # full history, RNG state, and last completed gen — enough to
    # resume identically via `train_model(checkpoint=path)`. Requires
    # `save_path` to be set; warned and skipped otherwise.
    checkpoint_interval: int | None = 2000

    # Save model after training (None = disabled; "auto" = auto-generate run directory)
    save_path: str | None = "models/auto"
    # Save plots (None = disabled; set automatically by setup_run_directory)
    save_plots: str | None = None
    # Show plots interactively (True = plt.show(), False = close after saving)
    show_plots: bool = False
    # Show extra info in progress bar (L1, L2 regularisation)
    debug: bool = False
    # Scale dipole targets by atom count (per-atom dipole training)
    scale_targets: bool = True
    # Native units of dipole targets in the dataset. Used (a) to derive
    # the e·Å conversion factor when `convert_dipole_to_eangstrom=True`,
    # and (b) as the plot-axis / stats unit label when conversion is
    # off so the displayed numbers match the file.
    # "e*angstrom" = e·Å
    # "e*bohr"     = e·a₀   (× 0.5292 → e·Å)
    # "debye"      = Debye  (× 0.2082 → e·Å)
    dipole_units: str = "e*bohr"
    # When True (default), dipole targets are converted to e·Å on data
    # load and training / plots / spectra all use e·Å. When False, the
    # raw dataset values are passed through unchanged — the model
    # trains in the dataset's native unit and plot axes label that
    # unit instead. Useful when you want loss / RMSE / RRMSE numbers
    # to be directly comparable with reference values quoted in
    # e·a₀ or Debye.
    convert_dipole_to_eangstrom: bool = False

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
