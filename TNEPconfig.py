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
    allowed_species: list[int | str] | None = None
    # Species filter mode: "subset" = keep structures with only allowed species,
    # "exact" = keep structures containing exactly all allowed species
    filter_mode: str = "subset"
    # When True, drop structures with NaN positions, NaN targets, or
    # zero-vector targets. Structures with missing targets are always
    # dropped regardless (they can't be trained against).
    filter_bad_data: bool = False
    num_neurons: int = 30
    # Number of structures used in each train step
    batch_size: int | None = None
    # Number of samples made in each train generation
    pop_size: int | None = 80
    # Number of training generations (number of updates to the model)
    num_generations: int = 60000
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
    descriptor_mode: int = 1

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
    descriptor_batch_frames: int | None = 100

    # Pair-tile size for the gradient compute. 0 = single-shot (lower
    # launch overhead, higher peak VRAM). >0 = tile pairs in chunks of
    # this size (cuts peak VRAM at the dominant [k_max,n_max,P] tensors;
    # at fp32 also auto-enables XLA fusion in the trajectory path).
    # Sensible value for ~600-atom systems: 8000.
    descriptor_pair_tile_size: int = 0

    # GPU memory budget (bytes) used by the auto-sizer when
    # descriptor_batch_frames is None. None falls back to the builder's
    # default (6 GiB).
    descriptor_memory_budget_bytes: int | None = None

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
    cache_gradients_to_disk: bool = True

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

    # Load the entire disk-backed grad_values into GPU memory once at
    # startup, then slice per-chunk on-GPU instead of staging from
    # disk every generation. Bypasses the cuFile / pinned / pool path
    # entirely — fastest possible read path when the grad cache fits
    # in VRAM. At full-batch full-data perf this saves the residual
    # cuFile read time (which is the dominant remaining cost in the
    # `evaluate` phase). Memory cost: P × 3 × dim_q × 4 bytes (e.g.
    # 333k pairs × 645 dim_q ≈ 2.6 GB at total_N=2000). Disable if
    # the cache exceeds available VRAM minus the model + activations.
    gpu_resident_grad_cache: bool = False

    # Use page-locked (pinned) host buffers for the disk-backed chunk
    # staging path. With pinned source, tf.constant dispatches a true
    # async cudaMemcpyAsync (no driver bounce buffer), saturating PCIe
    # at ~12-16 GB/s instead of the ~6-8 GB/s pageable rate. Buffers
    # are allocated via cudaMallocHost; pool size = 4. Set False if
    # cudart is not loadable (rare) or to bisect a regression.
    use_pinned_buffers: bool = True

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
    # Override the info/results key used to read targets from ASE structures.
    # None = use the default for target_mode ("energy", "dipole", "pol").
    # Set to a custom string to support non-standard dataset labels (e.g. "mu", "alpha").
    target_key: str | None = None
    # Test split ratio
    test_ratio: float = 0.3
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N: int | None = None
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
    # Pin dataset tensors to CPU memory instead of GPU.
    # Required when the full dataset is too large to fit in GPU VRAM (most cases).
    # When True, each training batch is copied CPU→GPU implicitly during evaluation.
    # Set to False only if the entire dataset comfortably fits on the GPU.
    pin_data_to_cpu: bool = True

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
