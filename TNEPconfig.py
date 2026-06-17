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
        5. Optimiser (SNES or Adam; core, plateau-reset, validation)
        6. Memory & I/O staging
        7. Output & diagnostics
        8. Runtime state (auto-populated by data load)
    """

    # ═══════════════════════════════════════════════════════════════════
    # 1. DATASET & TARGETS
    # ═══════════════════════════════════════════════════════════════════

    # --- dataset & split -----------------------------------------------
    data_path: str = "datasets/train.xyz"
    # Separate test dataset (None = split from data_path; str = path to external .xyz)
    test_data_path: str | None = "datasets/test.xyz"
    # Filter dataset to structures containing only these species
    # (None = no filter; list of int or str, e.g. [6, 1, 8] or ["C", "H", "O"])
    allowed_species: list[int | str] | None = None #[6, 1, 8]
    # Species filter mode: "subset" = keep structures with only allowed species,
    # "exact" = keep structures containing exactly all allowed species
    filter_mode: str = "subset"
    # When True, drop structures with NaN positions, NaN targets, or
    # zero-vector targets. Structures with missing targets are always
    # dropped regardless (they can't be trained against).
    filter_bad_data: bool = False
    # Test split ratio (only used when test_data_path is None)
    test_ratio: float = 0.3
    # None : uses entire dataset, int : defines maximum structures to use in training
    total_N: int | None = None
    # Seed for randomisation (dataset shuffle, SNES sampling, etc.)
    seed: int | None = 928375439201
    # Bitwise-reproducible runs. cfg.seed alone makes runs reproducible on
    # CPU, but GPU reductions (unsorted_segment_sum / atomic adds in the
    # dipole kernel) are non-deterministic across runs even with a fixed
    # seed. When True, MasterTNEP calls tf.config.experimental.
    # enable_op_determinism() so same-seed runs are bitwise identical on GPU
    # too. Cost: slower GPU ops, and a hard error if any op used has no
    # deterministic GPU implementation. Leave False for normal runs.
    deterministic: bool = False

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

    # Skip H atoms as DESCRIPTOR CENTERS:
    #   False (default) — every atom (including H) gets its own SOAP
    #     descriptor and contributes to the predicted total via U_i.
    #   True            — H atoms are NOT used as descriptor centers,
    #     so the quippy / TF descriptor builders are never called for
    #     H-centered windows. H atoms remain in the structure as
    #     NEIGHBOUR species, so their positions still appear in the
    #     species-pair blocks of every non-H center's descriptor.
    # Performance: ~2–3× faster SOAP build and forward pass on typical
    # organic systems (H is 50-70 % of atoms). Memory similar.
    # Caveats:
    #   - Models trained with True are NOT compatible with False
    #     (the network learns a heavy-atom-only scalar that produces
    #     the total μ / E / α — switching back changes the meaning).
    #   - target_mode=0 (energy) loses physical per-H-atom U_i
    #     interpretation; the model still predicts the right TOTAL E.
    #   - target_mode=1 (dipole) is the most natural fit — total μ is
    #     reproduced because non-H atoms' descriptors already encode
    #     H neighbour info via SOAP species-pair blocks.
    skip_h_centers: bool = False

    # ═══════════════════════════════════════════════════════════════════
    # 2. DESCRIPTOR (SOAP-turbo)
    # ═══════════════════════════════════════════════════════════════════

    # --- geometric parameters ------------------------------------------
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
    # None = auto (reads SLURM_CPUS_PER_TASK at DescriptorBuilder init time, falls back to 1)
    # 1    = serial (current behaviour, default outside SLURM)
    # N    = use N worker processes
    num_descriptor_workers: int | None = None

    # --- data-pipeline preprocessing (scaling, centering) --------------
    # Per-channel descriptor scaling applied at data-pipeline level.
    #   "none"     : no scaling (current default; preserves baseline).
    #   "q_scaler" : GPUMD-style multiplicative range normalisation.
    #                Each channel d gets s_d = 1 / (max_d - min_d) over
    #                the training set; computed ONCE, frozen for the
    #                run, persisted in /weights/q_scaler (and /snes/
    #                q_scaler in checkpoints). Both `descriptors` and
    #                `grad_values` are multiplied by s in pad_and_stack
    #                so the entire model code is unchanged (option A
    #                from the implementation plan — algebraically
    #                identical to GPUMD's option B but cleaner to plumb).
    descriptor_scaling: str = "none"

    # Granularity of the q_scaler (only consulted when descriptor_scaling=="q_scaler"):
    #   "per_component" : one multiplier per scalar descriptor entry (GPUMD-style).
    #                     Maximises channel-level decorrelation but breaks the
    #                     within-(pair, l)-block isotropy that l_aware /
    #                     cross_pair_l mixing layers rotate over.
    #   "l_block"       : one multiplier per (species-pair, l) block, shared
    #                     across all α entries within that block. Equalises the
    #                     dominant inter-l magnitude gap (l=0 ~O(1) vs l=l_max
    #                     ~O(0.01)) while preserving intra-block axes. Use this
    #                     when running descriptor_mixing with l_aware/cross_pair_l.
    q_scaler_granularity: str = "l_block"

    # Per-component target centering. When True, the per-component mean
    # of the training targets is computed once, subtracted from every
    # train/val/test target so the network learns in zero-mean output
    # space, and added back at the inference boundary (model.score,
    # model.predict, trajectory inference) so user-facing predictions
    # remain in the original units. Persisted in /weights/target_mean
    # (and /snes/target_mean in checkpoints) — frozen for the run.
    #
    # The network architecture has only one scalar output bias (b1); it
    # cannot place an independent per-component dipole/polarisability
    # offset. With non-zero target means (e.g. anisotropic training
    # data), this forces capacity into shifting the zero point through
    # W0/W1 interactions, leaving less capacity for the genuine
    # structure-dependent pattern. Centering decouples the two.
    #
    # Caveats:
    #   - Rotational equivariance: the saved mean is a constant in the
    #     dataset frame; it does NOT rotate with a rotated input. If the
    #     mean is non-isotropic (norm ≫ 0 for vector / tensor outputs),
    #     the network's centered-space predictions ARE equivariant but
    #     the original-units output (after adding the mean back) is NOT.
    #     Recommended use: only when the training-set mean is near zero
    #     or nearly isotropic — e.g. after rotational augmentation. The
    #     test_rotation_equivariance helper warns when this is violated.
    #   - RRMSE definition shifts: SNES's RRMSE numerator is computed
    #     against `sum(targets^2)` of the centered batch — i.e. it
    #     becomes variance-normalised rather than mean+variance. RRMSE
    #     values are not directly comparable between centered and
    #     un-centered runs on the same dataset.
    target_centering: bool = False

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
    # Linear-mixing architecture variant.
    #   "linear"       : one [bs_p × bs_p] matrix per pair. Mixes all
    #                    (n, l) channels within a pair (cross-l within
    #                    a pair permitted, cross-pair forbidden).
    #                    Block-diagonal in pair only.
    #   "l_aware"      : (l_max+1) separate [α_p × α_p] matrices per
    #                    pair, one per angular momentum. Mixes only
    #                    radial channels at the same l within the
    #                    same pair. Block-diagonal in (pair, l).
    #                    Strictly fewer parameters than "linear" by
    #                    a factor of (l_max+1).
    #   "cross_pair_l" : (l_max+1) separate [N_l × N_l] matrices, one
    #                    per angular momentum, where N_l = Σ_p α_eff_p.
    #                    Mixes radial channels at the same l ACROSS
    #                    species pairs (cross-pair within fixed l
    #                    permitted; cross-l never permitted).
    #                    Block-diagonal in l only.
    #                    ⊃ l_aware (strict superset). Not comparable
    #                    to "linear" — they restrict different axes
    #                    at similar param count.
    # All three use the same `_W0_eff = (I + V_full)ᵀ · W0` absorption.
    # Only consulted when descriptor_mixing=True.
    descriptor_mixing_arch: str = "l_aware"
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
    # l_aware mixing composes with all preprocess modes; other mixing
    # archs (linear, cross_pair_l) raise NotImplementedError when
    # preprocess is on.
    descriptor_preprocess_contract: str = "angular"
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

    # --- pretrained SOAP autoencoder (encoder front-end) ----------------
    # Optional pretrained encoder applied to the SOAP descriptor before
    # it reaches TNEP. None = disabled (TNEP sees raw SOAP). A string
    # is interpreted as a run directory produced by `SoapAutoencoder.py`
    # (i.e. a path under `models/autoencoder/...`) and is loaded via
    # `encoder_io.load_encoder` at data-prep time.
    #
    # When set, the encoder MUST be linear (no hidden layers in the
    # branches) — the descriptor gradient tensor `grad_values[P, 3, Q]`
    # is propagated through the encoder's analytic Jacobian J[Z, Q] via
    #     grad_values_new[p, 3, z] = Σ_q J[z, q] · grad_values[p, 3, q]
    # which is exact only when the encoder is affine. Nonlinear
    # encoders would require batched autodiff per training step and are
    # rejected at load time. Compatible architectures:
    #   - "l_block_pca" with l_block_hidden_dims = ()
    #   - "willatt" (always linear in u)
    #   - "willatt_l_block" with l_block_hidden_dims = ()
    encoder_path: str | None = "models/autoencoder/20260618_011211_willattlblock_Ks4_Kl64/encoder.npz"

    # Controls how the encoder is applied during training:
    #   False (default): STATIC preprocess. After data load, encode the
    #       descriptors AND grad_values once for every split (train /
    #       val / test) and discard the raw versions. TNEP then sees a
    #       Z-dimensional descriptor and a Z-dimensional gradient
    #       tensor. Cheapest at training time; encoder is frozen by
    #       construction (no path to update it).
    #   True: ITERATIVE / live. Keep the encoder in memory; encode each
    #       batch on the fly. The encoder's parameters become exposed
    #       to the optimiser:
    #         · Adam path: all encoder trainable variables join TNEP's
    #           in `self.trainable_variables`; gradients flow back
    #           through the encoder Jacobian to update both.
    #         · SNES path: only the Willatt species projection `u`
    #           (small, [T × K]) joins the μ vector. The per-l Dense
    #           layers stay frozen — SNES can't realistically explore
    #           their hundreds of thousands of dims. With a non-Willatt
    #           encoder under SNES + train_encoder=True, this falls
    #           back to frozen-encoder mode and prints a warning.
    train_encoder: bool = False

    # Row-chunk size for the iterative encoder forward pass. Bounds
    # peak VRAM for the Willatt / bilinear scatter, which materialises
    # an O(N · G_T² · L) dense tensor inside encode(). 4096 rows ≈
    # 470 MB for T=6, αmax=10, L=8; reduce for larger T/αmax/L or
    # smaller GPUs. Ignored when the encoder is absent or static.
    encoder_chunk_rows: int = 4096

    # ═══════════════════════════════════════════════════════════════════
    # 4. LOSS & REGULARISATION
    # ═══════════════════════════════════════════════════════════════════

    # Training loss function (controls what SNES ranks against — NOT what
    # is reported in history; RMSE / RRMSE are always computed alongside
    # and recorded regardless of this setting).
    #   "mse"   : Σ r_k²              standard squared error
    #   "mae"   : Σ |r_k|             absolute error
    #   "huber" : Σ huber(r_k; δ)     quadratic for |r| ≤ δ, linear beyond.
    #             Robust to outlier structures with large residuals.
    #             Does NOT specifically help small-target structures —
    #             use inverse_weight_mode for that.
    loss_type: str = "mse"
    # Transition point between quadratic and linear regimes of Huber loss.
    # Only used when loss_type == "huber". Sensible default ~ expected
    # final RMSE for the dataset (run the loss-tuning sweep to confirm).
    huber_delta: float = 1e-3

    # Inverse-magnitude weighting: upweight small-target structures /
    # components so they aren't dominated by large-target structures in
    # the gradient.
    #   "none"             : uniform (current default)
    #   "vector_magnitude" : w_b ∝ 1 / max(||target_b||², eps)
    #                         — one weight per structure (legacy path).
    #   "per_component"    : w_{b,k} ∝ 1 / max(target_{b,k}², eps)
    #                         — separate weight per (structure, component).
    #                         Useful when individual axes of vector
    #                         targets have systematically small magnitudes.
    inverse_weight_mode: str = "none"
    # Epsilon floor for the inverse-weight denominator. Smaller eps
    # → stronger small-target emphasis but more numerical instability.
    # Only used when inverse_weight_mode != "none".
    inverse_weight_eps: float = 1e-4

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
    # 5. OPTIMISER (SNES or Adam)
    # ═══════════════════════════════════════════════════════════════════

    # Which optimiser drives training. Two options:
    #   "snes" — evolutionary; ranks a population of candidate weight
    #            vectors by per-structure loss and updates μ / σ via the
    #            SNES log-rank gradient. Required for non-smooth losses
    #            (dipole sign-ambiguous contraction, polarisability
    #            shear weighting) and for the population-based search
    #            that escapes shallow basins. Heavy on candidate eval
    #            (P forward passes per generation), light on per-step
    #            memory (no autodiff buffers).
    #   "adam" — gradient descent via tf.GradientTape + Adam(W). One
    #            forward + one backward per epoch. Best for target
    #            modes where the loss is smooth and small models that
    #            train quickly. Skips ALL SNES-specific knobs below
    #            (pop_size, init_sigma, mu_init_scheme, plateau resets,
    #            etc.) and uses the `adam_*` fields below instead.
    #
    # When "adam" is selected, `num_generations` is interpreted as the
    # number of Adam epochs, `batch_size` controls the minibatch size
    # the same way (None = full batch), `val_interval` controls how
    # often validation runs, and `loss_type` / `huber_delta` still
    # select the differentiable training loss.
    optimizer: str = "snes"

    # ── Adam-specific knobs (ignored when optimizer == "snes") ─────────
    # Peak learning rate. For finetuning a near-optimal init (e.g. PCA
    # warm-start), drop to 1e-4 or 1e-5; from-scratch typically 1e-3.
    adam_lr: float = 1e-3
    # Optional decoupled L2 (AdamW). 0.0 disables; positive values
    # apply the standard AdamW decoupled-decay update to all trainable
    # weights (including biases). Note: this is independent of the
    # SNES-side `l1_weight` / `l2_weight` regularisers, which are NOT
    # applied in the Adam path (they're added to the SNES ranking
    # fitness, not to a gradient).
    adam_weight_decay: float = 0.0
    # Global-norm gradient clipping threshold. None disables. Useful
    # against rare large-update epochs when the dipole / polarisability
    # nonlinearity produces sharp loss surface kinks.
    adam_grad_clip: float | None = 1.0
    # Adam β1, β2, ε — defaults match tf.keras.optimizers.Adam.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-7

    # --- core ----------------------------------------------------------
    # Number of samples made in each train generation
    pop_size: int | None = 100
    # Number of training generations (number of updates to the model).
    # When `optimizer == "adam"` this is the number of Adam epochs.
    num_generations: int = 60000
    # Number of structures used in each train step (None = full batch).
    # Adam uses this as the minibatch size; SNES uses it as the per-
    # generation candidate-evaluation batch.
    batch_size: int | None = None
    # Learning rate for sigma (None = auto from canonical SNES heuristic)
    eta_sigma: float | None = None
    # SNES μ vector initialisation scheme.
    #   "uniform" : each ANN entry drawn uniform[-1, 1] (GPUMD default).
    #               The biases come in at the same scale as the weights,
    #               so b0 typically dominates the pre-activation in the
    #               first generation — this relies on SNES exploration
    #               to find a sensible magnitude over the first ~100 gens.
    #   "glorot"  : Glorot/Xavier uniform per weight group:
    #                 W0   ~ U(-c_W0, c_W0),  c_W0 = √(6 / (Q + H))
    #                 W1   ~ U(-c_W1, c_W1),  c_W1 = √(6 / (H + 1))
    #                 b0   = 0
    #                 b1   = 0
    #               Keeps initial pre-activations well-scaled regardless
    #               of Q / H, so SNES doesn't waste generations climbing
    #               out of a saturated-activation regime. The same
    #               scheme is applied to the polarisability ANN's
    #               W0_pol / W1_pol when target_mode == 2. The V_pair
    #               tail (residual mixing layer) stays at zero in both
    #               schemes, so the model is bit-identical to the
    #               mixing-disabled baseline at gen 0 regardless of the
    #               chosen scheme.
    mu_init_scheme: str = "glorot"
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

    # --- early stopping & plateau-triggered sigma reset ----------------
    # Early stopping patience in val ticks (None = disabled)
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
    #
    # Interaction with cfg.patience: plateau resets fire FIRST (they
    # broaden σ and reset the no-improvement counter), so a run that
    # hits plateau_reset_patience repeatedly never reaches early-stop.
    # Setting patience < plateau_reset_patience disables the reset
    # path entirely. cfg.max_sigma_resets caps the number of resets.
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
    # (or just continue if patience is None). No-op when
    # plateau_reset_patience is None — there are no resets to cap.
    max_sigma_resets: int | None = None

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
    # XLA-compile the per-chunk eval (`_evaluate_chunk`). Fuses the
    # dipole-kernel pre-compute and the per-type matmul + reduction ops
    # into a single GPU kernel — typically 1.5-2× faster on Ada/Hopper.
    # Each unique (B_chunk, P_chunk) shape triggers one XLA compile
    # (~5-10 s the first time that shape is seen); for full-batch
    # deterministic chunks this is a one-shot cost paid in the first
    # generation, then steady-state runs at full XLA speed.
    eval_jit_compile: bool = False

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

    # --- HPC / Slurm mode ----------------------------------------------
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
    # Profile overrides applied (both share the same disk/pinned/cuFile
    # disabled common base):
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

    # --- disk-backed grad cache + prefetch ring ------------------------
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
    # bisect threading issues. Force-disabled when csc_enable=True.
    chunk_prefetch: bool = True
    prefetch_depth: int = 2

    # --- pinned-host + cuFile pools ------------------------------------
    # Use page-locked (pinned) host buffers for the disk-backed chunk
    # staging path. With pinned source, tf.constant dispatches a true
    # async cudaMemcpyAsync (no driver bounce buffer), saturating PCIe
    # at ~12-16 GB/s instead of the ~6-8 GB/s pageable rate. Buffers
    # are allocated via cudaMallocHost. Set False if cudart is not
    # loadable (rare) or to bisect a regression. Force-disabled when
    # csc_enable=True (see _apply_csc_overrides).
    use_pinned_buffers: bool = True
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
    # Force-disabled when csc_enable=True (Lustre on Mahti has
    # unreliable GDS support; see _apply_csc_overrides).
    use_cufile: bool = True
    # Number of GPU buffers in the cuFile pool. Each is sized to the
    # worst-case chunk grad slice; total VRAM cost is
    # cufile_pool_size × buffer_bytes. Must be >= prefetch_depth + 1.
    cufile_pool_size: int = 2

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
