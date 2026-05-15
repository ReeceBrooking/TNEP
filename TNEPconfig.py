from __future__ import annotations

import numpy as np


class TNEPconfig:
    """Holds all hyperparameters and runtime state for a TNEP training run.

    The model is a Gaussian Approximation Potential (GAP):
    per-atom scalar U_i = Σ_s α_s · K(q_i, q_s) with a polynomial
    dot-product kernel, fitted via closed-form linear regression. There
    is no iterative training loop — only a single closed-form solve.

    Class-level defaults are overwritten at runtime by MasterTNEP after
    data loading (num_types, types, dim_q, indices).
    """

    # ───────────────────────────────────────── Dataset / split
    data_path: str = "datasets/train_waterbulk.xyz"
    # Separate test dataset (None = split from data_path; str = path to external .xyz)
    test_data_path: str | None = "datasets/test_waterbulk.xyz"
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
    # Test split ratio (only used when test_data_path is None)
    test_ratio: float = 0.2
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N: int | None = None
    # Seed for randomisation
    seed: int | None = None

    # ───────────────────────────────────────── SOAP-turbo descriptor parameters
    l_max: int = 7
    alpha_max: int = 7
    rcut_hard: float = 6.0
    rcut_soft: float = 5.5
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

    # Per-channel descriptor scaling applied at data-pipeline level.
    #   "none"     : no scaling (current default).
    #   "q_scaler" : GPUMD-style multiplicative range normalisation.
    #                Each channel d gets s_d = 1 / (max_d - min_d) over
    #                the training set; computed ONCE, frozen for the
    #                run, persisted in /weights/q_scaler. Both
    #                `descriptors` and `grad_values` are multiplied by
    #                s in pad_and_stack so the model code is unchanged.
    descriptor_scaling: str = "none"

    # Granularity of the q_scaler (only consulted when descriptor_scaling=="q_scaler"):
    #   "per_component" : one multiplier per scalar descriptor entry (GPUMD-style).
    #   "l_block"       : one multiplier per (species-pair, l) block,
    #                     shared across α entries within that block.
    q_scaler_granularity: str = "l_block"

    # Per-component target centering. When True, the per-component mean
    # of the training targets is subtracted on load and added back at
    # the inference boundary so user-facing predictions remain in the
    # original units. Persisted in /weights/target_mean.
    target_centering: bool = False

    # ───────────────────────────────────────── GAP hyperparameters

    # Polynomial kernel exponent; body order = 2·ζ+1. Default 4 = TurboGAP convention.
    gap_zeta: int = 4

    # Total number of sparse points across all species.
    gap_n_sparse: int = 4000

    # Sparse-point selection method.
    #   "fps"    : farthest-point sampling (recommended).
    #   "random" : uniform random (debug only — produces inferior fits).
    gap_sparse_method: str = "fps"

    # Split sparse points by atomic species (hard species mask in kernel).
    # Standard GAP / TurboGAP practice — leave True.
    gap_per_species_sparse: bool = True

    # Dedup near-duplicate sparse points within this Euclidean distance
    # (on L2-normalised descriptors). Mirrors gap_fit's behaviour.
    gap_dedup_tol: float = 1e-6

    # Global noise scale σ for the linear-system regulariser.
    # None → heuristic 0.1·√var(y_total). Set explicitly for production.
    gap_sigma_E: float | None = None

    # Multiply each row's σ by √N_k (structure size). Stops large-N
    # structures dominating the loss. Standard GAP "energy weighting".
    gap_structure_size_weight: bool = True

    # Use the GAP prior K_MM in the linear system (DTC form). Phase 2.
    # Phase 1 default = False → simple ridge regression.
    gap_use_prior_covariance: bool = False

    # Ridge regularisation strength λ for the augmented-QR solve.
    # Decoupled from gap_sigma_E (which controls per-row data fidelity
    # in Σ⁻¹/²). Tune up if the fit is unstable or overfits.
    gap_ridge_lambda: float = 1e-6

    # Inner-block size along the M axis for the Φ build. Peak GPU
    # memory during Φ build is dominated by the per-pair gather
    # `[P_chunk, M_sub, Q]` tensor. Reduce if you OOM.
    gap_sparse_chunk_size: int = 8

    # Structure chunk size for the Φ build. Decouples from
    # `cfg.batch_chunk_size` (which controls score/predict chunking)
    # because the Φ-build pair-gather is much heavier per structure.
    # `None` → auto = min(S, 100). Reduce if you OOM on dense
    # neighbour graphs (e.g. bulk water with M ≥ 1000).
    gap_struct_chunk_size: int | None = None

    # Pair-axis chunk size for the inner gather+contract in Φ build
    # and in predict_batch. Peak GPU memory is dominated by the
    # per-pair gather `[P_sub, M_sub, Q]` (Φ build) or `[P_sub, Q]`
    # (predict). Reduce if you OOM.
    gap_pair_chunk_size: int = 25_000

    # ───────────────────────────────────────── Descriptor backend / GPU compute

    # Descriptor backend: 0 = quippy (Fortran, CPU), 1 = native TF/NumPy (GPU when available).
    # The GPU path supports basis="poly3" and compress_mode="trivial" only;
    # falls back to a clear error if other settings are requested.
    descriptor_mode: int = 1

    # Internal precision for the GPU descriptor compute. The Fortran reference
    # uses double-precision throughout; "float64" mirrors that exactly. The
    # opt-in "float32" path keeps roughly half the VRAM and runs faster on
    # consumer GPUs. Has no effect for descriptor_mode=0.
    descriptor_precision: str = "float32"

    # Number of structures concatenated into a single SOAP graph call.
    #   1     : per-frame (lowest VRAM, highest launch overhead)
    #   int>1 : multi-frame batching, amortises kernel launches
    #   None  : auto — choose the largest batch that fits
    #           `descriptor_memory_budget_bytes` (default 6 GiB).
    descriptor_batch_frames: int | None = 10

    # Pair-tile size for the gradient compute. 0 = single-shot.
    # **Lower this** (e.g. 1000-2000) when running with high l_max or
    # α_max — the per-frame peak is dominated by the
    # [k_max · n_max · pair_tile] gradient tensor.
    # Sensible defaults: 8000 for l_max≤4; 2000 for l_max≤6;
    # 1000 for l_max≥7.
    descriptor_pair_tile_size: int = 1000

    # GPU memory budget (bytes) used by the auto-sizer when
    # descriptor_batch_frames is None. None falls back to the builder's
    # default (6 GiB).
    descriptor_memory_budget_bytes: int | None = None

    # Number of parallel workers for SOAP descriptor computation.
    # None = auto (reads SLURM_CPUS_PER_TASK at builder init).
    num_descriptor_workers: int | None = None

    # ───────────────────────────────────────── Data staging / IO

    # Master switch for CSC / Slurm-supercomputer mode. When True:
    #   - All gradient-caching / IO options are forced off
    #     (`cache_gradients_to_disk`, `chunk_prefetch`,
    #     `use_pinned_buffers`, `use_cufile`).
    #   - The Slurm-specific scratch-dir resolver is allowed to consult
    #     `$SLURM_TMPDIR` / `$TMPDIR` / `$LOCAL_SCRATCH`.
    csc_enable: bool = False

    # When True, grad_values is written to a temp dir (on the working
    # directory's filesystem, typically NVMe) and accessed via numpy
    # memmap. Removed automatically at training end.
    cache_gradients_to_disk: bool = True

    # Overlap disk → GPU staging of chunks N+1..N+prefetch_depth with
    # GPU evaluation of chunk N.
    chunk_prefetch: bool = True
    prefetch_depth: int = 1

    # Number of pinned host buffers in the pool. Must be >=
    # prefetch_depth + 1.
    pinned_pool_size: int = 2

    # When True and libcufile is loadable, the disk-backed gradient cache
    # is read directly from NVMe into pre-allocated GPU buffers via
    # cuFile (NVIDIA GPUDirect Storage). Falls back to the pinned path
    # silently if cuFile isn't usable.
    use_cufile: bool = False

    # Number of GPU buffers in the cuFile pool. Must be >=
    # prefetch_depth + 1.
    cufile_pool_size: int = 2

    # Use page-locked (pinned) host buffers for the disk-backed chunk
    # staging path. Set False if cudart is not loadable.
    use_pinned_buffers: bool = True

    # Where the static training tensors live, and which chunk-staging
    # path the score loops use:
    #
    #   True  : tensors stay on host CPU. Per-chunk slice → tf.gather
    #           → implicit H2D copy. Required when the full dataset is
    #           too big for VRAM.
    #   False : tensors live on /GPU:0. The chunk-staging path becomes
    #           pure on-device gather. Fastest mode when the working
    #           set fits in VRAM.
    pin_data_to_cpu: bool = True

    # Number of structures to process per GPU chunk during score
    # (None = auto-derived from per-struct grad_values bytes, capped
    # at 1 GiB per chunk).
    batch_chunk_size: int | None = 200

    # ───────────────────────────────────────── Targets / I/O

    # 0 : PES, 1 : Dipole, 2 : Polarizability (deferred — raises in TNEP.__init__).
    target_mode: int = 1
    # Override the info/results key used to read targets from ASE structures.
    # None = use the default for target_mode ("energy", "dipole", "pol").
    target_key: str | None = None
    # Scale dipole targets by atom count (per-atom dipole training)
    scale_targets: bool = True
    # Native units of dipole targets in the dataset. Used to derive the
    # e·Å conversion factor when `convert_dipole_to_eangstrom=True`.
    # "e*angstrom" = e·Å
    # "e*bohr"     = e·a₀   (× 0.5292 → e·Å)
    # "debye"      = Debye  (× 0.2082 → e·Å)
    dipole_units: str = "e*bohr"
    # When True, dipole targets are converted to e·Å on data load.
    # When False, the raw dataset values are passed through unchanged.
    convert_dipole_to_eangstrom: bool = False

    # Save model after training (None = disabled; "auto" = auto-generate run directory)
    save_path: str | None = "models/auto"
    # Save plots (None = disabled; set automatically by setup_run_directory)
    save_plots: str | None = None
    # Show plots interactively
    show_plots: bool = False

    # ───────────────────────────────────────── Runtime state (set by data load)
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
