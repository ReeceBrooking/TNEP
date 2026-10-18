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
    dipole_rij_power: int = 2

    # Dipole readout form (target_mode=1 only):
    #   "tnep"     : descriptor-gradient form above (dipole_rij_power applies)
    #   "neighbor" : neighbor basis with C radial channels and species-pair
    #                coefficients:
    #                  b_ij = Σ_n c_n[p(t_i,t_j)]·R_n(r_ij)·(s_i^n − s_j^n)
    #                  μ = Σ_{i<j} b_ij·(r_i − r_j)   (MIC displacements)
    #                R_n(r) = T_n(2r/r_c − 1)·f_c(r) (Chebyshev, 1st kind;
    #                f_c per dipole_cutoff)
    dipole_readout: str = "neighbor"
    # Number of Chebyshev radial channels C of the neighbor readout (n =
    # 0..C−1). C=1 with c≡1 reduces to the plain f_c single-channel form.
    # Widens the site head W1 to [T, H, C]. Validated ≥ 1 at model init.
    # Models must be re-trained if C changes.
    dipole_channels: int = 20
    # Radial family of the neighbor readout (f_c per dipole_cutoff):
    #   "chebyshev"        : R_n(r) = T_n(2r/r_c − 1)·f_c(r), C channels
    #   "chebyshev_warped" : R_n(r) = T_n(x_w)·f_c(r), NEP's squared warp
    #                        x_w = 2(r/r_c − 1)² − 1 — resolution
    #                        concentrated at short range, C channels
    #   "gaussian"         : R_n(r) = exp(−(r − μ_n)²/2σ²)·f_c(r), μ_n the
    #                        C interior points of linspace(0, r_c, C+2),
    #                        σ = dipole_gauss_sigma_scale × spacing
    #   "screened"         : R(r) = r^(ζ−1)·f_c(r) — single channel
    #                        (requires dipole_channels == 1).
    dipole_radial: str = "chebyshev_warped"
    # Screened-radial exponent ζ (dipole_radial="screened" only). Default −2
    # from the rigid-charge LSQ scan: floors 2.70 at ζ=−1, 2.72 at ζ=−2,
    # 11.0 at ζ=+1 (ζ>0 inverts the covalent/H-bond weighting). −2 chosen
    # over the marginally-lower −1 deliberately: ~equal floor, stronger
    # H-bond suppression (Coulomb-like r^−2 screening).
    dipole_zeta: float = -2.0
    # Gaussian-window width scale σ/spacing (dipole_radial="gaussian" only):
    # σ = scale × r_c/(C+1) (C=1 ⇒ scale·r_c/2). Step-0 rigid-charge LSQ
    # evidence (diagnostics/diag_basis.py): scale 0.5 lowers the C=4 floor
    # 22% (1.16 vs 1.49) while 1.0 edges it at C=10 — the knob matters.
    dipole_gauss_sigma_scale: float = 1.0
    # Cutoff function f_c of the neighbor readout, crossable with any
    # radial. Smoothness at r_c is a property of the CUTOFF, not the radial:
    #   "cosine2" : ((1+cos(πr/r_c))/2)² — C³ contact at r_c
    #   "cosine"  : (1+cos(πr/r_c))/2 — C¹ contact (plain NEP cutoff)
    #   "cubic"   : 1 − 3x² + 2x³, x = r/r_c — C¹ contact (f″(r_c) ≠ 0)
    #   "tanh3"   : tanh³(1 − r/r_c)/tanh³(1) — C² contact (Behler cutoff,
    #               normalized so f(0) = 1)
    # None = the radial family's traditional form: "cubic" for screened,
    # "cosine2" otherwise (resolved_dipole_cutoff; validated at model init).
    dipole_cutoff: str | None = "cosine"
    # Species-pair coefficient table of the neighbor readout:
    #   "off"       : c ≡ 1 frozen, no parameters.
    #   "symmetric" : learnable c [C, T(T+1)/2], unordered-pair indexed.
    #   "ordered"   : learnable Ω [C, T²], ordered (t_center, t_neighbor)
    #                 indexed — breaks the difference-form antisymmetry (and
    #                 gauge invariance) across species.
    # Legacy bool: False→"off", True→"symmetric" (normalized at model init).
    dipole_species_pairs: str = "symmetric"
    # Head output of the neighbor readout:
    #   "vector" : one site value per radial channel — s [C]
    #   "scalar" : one site value broadcast over the C channels — s [1];
    #              radial resolution then lives only in the learned
    #              species-pair table c[n, pair], so pair it with
    #              dipole_species_pairs != "off". Head width 1 vs C.
    dipole_head: str = "scalar"
    # Neighbor-readout cutoff r_c (Å). None inherits the cutoff of whatever
    # descriptor actually feeds the model: nep_rc_radial under
    # nep_descriptor="replace", else rcut_hard (resolved_dipole_rc, applied at
    # model init). ANY positive radius is allowed — the staged V is built from
    # an image-exact neighbour list, so it is correct past rcut_hard and past
    # L/2. Data assembly WARNS instead of refusing: at r_c ≥ L_min/2 the
    # minimum-image paths (in-graph V, predict) double-count, and at r_c ≥ L_min
    # self-image pairs are dropped so the neighbour sum truncates at the box
    # scale. Note "cosine2" narrows effective support vs the plain cosine.
    dipole_rc: float | None = None

    # --- GAP-mode dipole fit (gap_dipole.py) -----------------------------
    # FPS-selected sparse environments per species for the kernel basis.
    gap_sparse_per_species: int = 500
    # Power(s) ξ of the normalized SOAP dot-product kernel (D̂_i·D̂_m)^ξ.
    # An int is a single kernel; a list (e.g. [1, 2, 4]) is a sum-of-kernels
    # fit — each sparse point contributes one column block PER ξ and the
    # ridge learns the implicit kernel mixture. Entries must be ints ≥ 1.
    gap_kernel_power: int | list[int] = 2
    # Second SOAP length scale for the KERNEL only (gap_dipole.py): a second
    # descriptor builder with rcut_hard = gap_extra_rcut and rcut_soft =
    # gap_extra_rcut − (rcut_hard − rcut_soft) (same buffer width), its own
    # FPS sparse sets, and additional Φ column blocks (scales never cross,
    # like species; the geometry vectors v are unchanged). Typically LARGER
    # than rcut_hard — adds long-sighted environment resolution (multi-scale
    # SOAP ~25%: Willatt/Musil/Ceriotti PCCP 2018). None = single scale.
    gap_extra_rcut: float | None = None

    # ═══════════════════════════════════════════════════════════════════
    # 2. DESCRIPTOR (SOAP-turbo)
    # ═══════════════════════════════════════════════════════════════════

    # --- geometric parameters ------------------------------------------
    l_max: int = 7
    alpha_max: int = 7
    rcut_hard: float = 4.0
    rcut_soft: float = 3.5
    basis: str = "poly3"
    scaling_mode: str = "polynomial"
    radial_enhancement: int = 0
    # "trivial" : soap_turbo's traditional compression recipe (the only mode
    #             the GPU builder, descriptor_mode=1, implements).
    # "none"    : the uncompressed power spectrum, N(N+1)/2*(l_max+1) over the
    #             N = num_types*alpha_max radial channels. This is what
    #             TurboGAP's soap_turbo type defaults to (types.f90 keeps
    #             compress_mode="none" behind its own compress_soap flag).
    #             On the quippy path (descriptor_mode=0) the keyword is OMITTED
    #             from the descriptor string, because soap_turbo's
    #             get_compress_indices rejects the literal "none"
    #             (soap_turbo_compress.f90:292).
    # "linear"  : quippy's linear compression, dim = 2*num_types*(l_max+1).
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
    # Learnable radial basis (GPU builder only, descriptor_mode=1):
    #   None : fixed SOAP-turbo basis (current behaviour)
    #   K    : int ≥ alpha_max. The builder emits raw centre coefficients over
    #          the K orthonormal poly3 functions; the alpha_max radial
    #          functions per (centre type, neighbour species) become learned
    #          combinations M_radial[T, S, alpha_max, K] in the SNES vector
    #          (NEP's c_nk analogue; init reproduces the fixed basis exactly).
    #          Supported with the neighbor dipole readout (no descriptor
    #          gradients); gradient-consuming modes raise NotImplementedError.
    radial_basis_size: int | None = None
    # Species handling of the learned radial channels (radial_basis_size only):
    #   False : keep SOAP-turbo's species-blocked layout and dimensions —
    #           n_max = S·alpha_max, trivial compression, dim_q unchanged.
    #   True  : NEP-style species compression — the neighbour-species axis is
    #           summed into alpha_max shared channels
    #           (c_n = Σ_s Σ_κ M[t,s,n,κ]·c_raw[s,κ]), full (n ≤ n') power
    #           spectrum, dim_q = alpha_max(alpha_max+1)/2·(l_max+1); species
    #           information enters only through M.
    radial_basis_species_sum: bool = False
    # l-dependent learned radial functions (radial_basis_size only): each
    # angular block l = 0..l_max gets its own M[T, S, alpha_max, K, l] —
    # NEP's separate angular c_nk. False shares one M across all l.
    # L× the radial coefficients; init still reproduces the fixed basis.
    radial_basis_per_l: bool = False
    # How the SOAP power spectrum is put on a common footing (radial_basis_size
    # only):
    #   "l2"   : SOAP-turbo's per-atom projection q/‖q‖ — shape only, magnitude
    #            (coordination) discarded, channels coupled through the norm.
    #   "none" : raw spectrum × scale² — magnitude kept, channels independent,
    #            quadratic in the neighbour density (l=0 ≫ l=4, C ≫ H).
    #   "sqrt" : sign(q)·√|q| × scale — degree 1 in the density like NEP's
    #            radial block; halves every log-spread of "none" while
    #            keeping magnitude. Meant with descriptor_scale.
    radial_basis_norm: str = "l2"
    # Descriptor scale preconditioning (plan 2026-09-23). A per-channel scale
    # m and shift μ are folded into the input layer, W0_eff = Uᵀ·diag(m)·W0
    # and b0_eff = b0 − (m∘μ)·W0 (the ANN sees m∘(q − μ)), so the model
    # class is unchanged (any scaled model == an unscaled one with W0' = m∘W0)
    # and q / ∂q/∂R are never touched; only the SNES/Adam search geometry
    # changes (effective per-channel step σ·m). The stored weights keep the
    # plain Glorot / zero init and act on the scaled input, so generation 0
    # is NOT the "off" model (dividing the init by m to make it so starts
    # the tanh layer saturated and inflates L1/L2 by 1/m — measured
    # 2026-10-05). Two levels from the per-channel std s_k of the
    # generation-0 descriptor over training atoms (frozen): a per-group
    # factor g = 1/median(s_group) (unit spread; α = 1, unclamped: group
    # medians are robust) times, in "channel" mode, a per-channel residual
    # clip((1/(g·s_k))^0.5, 1/10, 10) (dust protection). Dead channels
    # keep m = 1.
    #   "off"     : no scaling or centring (bit-identical to before)
    #   "block"   : group factor only (SOAP l groups + each extra block, ≤ 9)
    #   "channel" : group factor × per-channel residual
    #   "minmax"  : GPUMD q_scaler — 1/(max−min) per channel, no centring
    descriptor_scale: str = "block"
    # Runtime: the scale [num_types, dim_q] — one row per centre species,
    # each from that species' atoms — computed by the driver from the
    # training set and saved in the model's config (None = not yet computed).
    q_scale: list[list[float]] | None = None
    # Runtime: per-channel centring μ [num_types, dim_q] (mean of the generation-0
    # descriptor over training atoms), folded into b0 (b0_eff = b0 − (m∘μ)·W0)
    # so the ANN sees m∘(q − μ); computed with q_scale.
    q_shift: list[list[float]] | None = None
    # GPUMD NEP descriptors (multi_descriptor_plan.md §0.3), computed
    # exactly as GPUMD's nep.cu / nep_utilities.cuh from a raw per-atom
    # basis (nep_descriptor.py) contracted in-model by learnable c tensors.
    # Defaults = GPUMD's (parameters.cu:83-90, 118-119).
    #   "off"     : bit-identical to before
    #   "concat"  : q = [q_soap ‖ q_nep]
    #   "replace" : q = q_nep (descriptor_mixing off)
    #   "sum"     : summed experts (§0.4)
    #   "gap_sum" : summed experts on the GAP 2b/3b distance descriptors
    #               (gap_descriptor_plan.md §0.2; no NEP header, no c tensors)
    nep_descriptor: str = "gap_sum"
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
    # Per-expert amplitudes (δ_soap, δ_2b, δ_3b) for the summed head
    #   out = δ_soap·ANN_soap + δ_2b·ANN_2b + δ_3b·ANN_3b
    # Honest statement: δ multiplying an expert's head output is EXACTLY
    # redundant with rescaling that expert's W1 — the model class is unchanged.
    # δ is only meaningful because it is EXCLUDED from the L1/L2 regularisation
    # that acts on W1: an unpenalised per-expert amplitude, i.e. a learned gate
    # on each body order's influence. Ignored unless nep_descriptor is
    # "sum" / "gap_sum".
    sum_delta: list[float] = [1.0, 1.0, 1.0]
    sum_delta_learn: bool = False        # True → the three δ join the trained params
    # TurboGAP 2-body / 3-body distance descriptors (gap_descriptor.py,
    # gap_descriptor_plan.md §0.1): GAP's neighbour sums with FIXED centres
    # (sparse points) and widths, "exp" kernel only (no "pol"). Used by the
    # ANN nep_descriptor="gap_sum" experts and the GAP-mode column blocks.
    gap2b_rcut: float = 6.0              # GAP rcut of the 2b term
    gap3b_rcut: float = 2.0              # GAP rcut of the 3b term
    gap_buffer: float = 0.5              # GAP buffer: cosine taper width inside rcut
    gap2b_n_centres: int = 40            # M2 kernel centres on r
    gap2b_sigma: float = 0.5             # 2b kernel width (Å)
    gap3b_n_centres: int = 100            # M3 kernel centres on q = [r_ij+r_ik, (r_ij−r_ik)², r_jk]
    gap3b_sigma: list[float] = [2.0, 2.0, 2.0]  # per q component
    # 3b kernel on the scaled distance r = |(u − Q)/σ| (gap.f90 kernel_type):
    #   "exp" : exp(−r²/2) — σ is a Gaussian width
    #   "pol" : (1−r)⁴(4r+1) for r < 1, else 0 — σ is the SUPPORT radius
    #           (σ = 2.0 ≈ a Gaussian width of 0.5). TurboGAP's published
    #           potentials use "pol" with σ = 2.0 and rcut = 2.0.
    gap3b_kernel: str = "pol"
    # Runtime: auto-selected by gap_descriptor.select_centres when None
    # (2b uniform grid, 3b farthest-point sampling under cfg.seed) and saved
    # with the model like q_scale; shapes [M2] and [M3, 3].
    gap2b_centres: list[float] | None = None
    gap3b_centres: list[list[float]] | None = None
    # nep_descriptor="gap_sum": which models make up the prediction, any
    # non-empty subset of "soap", "2b", "3b". Each is its own ANN with its own
    # species-pair readout table, μ = Σ_k Σ_i s_k,i·Ω_k(r_ij)·d_ij; the extra
    # models start at zero output, so generation 0 is the first model alone.
    # Leaving "soap" out drops the SOAP network and descriptor entirely (the
    # first selected GAP block feeds the main ANN, like nep_descriptor="replace").
    gap_models: list[str] = ["soap", "2b", "3b"]
    # GAP-mode (gap_dipole.py) additive linear column blocks.
    gap_2b: bool = False
    gap_3b: bool = False

    # ═══════════════════════════════════════════════════════════════════
    # 3. NETWORK ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════

    # Hidden-layer width of the per-type ANN.
    num_neurons: int = 10
    # Hidden-layer activation (any tf.keras.activations name for the forward
    # pass). Dipole/polarisability training (target_mode 1/2) has a hand-coded
    # backward — only "tanh" and "swish" (alias "silu") supported there;
    # energy training (target_mode 0) accepts any.
    activation: str = 'tanh'

    # When True, insert a learnable per-species-pair linear mixing layer
    # between the fixed SOAP-turbo descriptor and the ANN (U_pair shared
    # across central types, identity-init). Analog of NEP's trainable
    # radial-basis coefficients c_nk.
    descriptor_mixing: bool = False
    # When True (with descriptor_mixing), U_pair becomes per-central-type
    # ([T, num_pairs, max_bs, max_bs]) — more expressive, T× the param count.
    descriptor_mixing_per_type: bool = False
    # Parameterisation of each descriptor-mixing block (structural, not a
    # penalty); only used when descriptor_mixing=True:
    #   "off"    : unregularised, bs² params per block
    #   "cayley" : U = (I−A)(I+A)⁻¹, bs·(bs−1)/2 params, no −1-eigenvalue
    #   "expm"   : U = exp(A), surjective onto SO(n), bs·(bs−1)/2 params. RECOMMENDED.
    descriptor_mixing_regularizer: str = "expm"

    # ═══════════════════════════════════════════════════════════════════
    # 4. LOSS & REGULARISATION
    # ═══════════════════════════════════════════════════════════════════

    # Master switch for L1/L2 regularisation. False = add 0 reg regardless
    # of lambda settings (lambda values kept so re-enabling mid-run resumes).
    toggle_regularization: bool = True
    # L1/L2 strengths: None = auto sqrt(dim*1e-6/num_types), -1.0 = dynamic
    # (adapt every lambda_adapt_interval gens toward lambda_target_ratio of
    # data RMSE), float = fixed. A LIST gives each model its own strength,
    # in model order (nep_descriptor="gap_sum": the gap_models selection in
    # soap, 2b, 3b order) — like GAP's per-descriptor priors. Each model's
    # penalty is normalised by its own parameter count either way.
    lambda_1: float | list[float] | None = [0.0005, 0.003, 0.005]
    lambda_2: float | list[float] | None = [0.0005, 0.003, 0.005]
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
    # Training algorithm: "snes" (evolution strategy, the NEP default) or
    # "adam" (gradient descent by autodiff through the forward — trains the
    # ANN, c_pair, the mixing residual U_pair as a dense V, and M_radial;
    # one Adam step per structure chunk (batch_size=None) or per sampled
    # batch; no checkpoint/resume). num_generations, batch_size, val_interval,
    # patience and the
    # L1/L2 settings apply to both.
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
    # Antithetic (mirrored) sampling of the SNES population: candidates come
    # in ± pairs. GPUMD does NOT do this — snes.cu gpu_create_population draws
    # one independent normal per candidate. The pair halves the variance of
    # the MEAN gradient, but both members carry the SAME s², so the σ natural
    # gradient Σ u_p (s_p² − 1) collapses to Σ_pairs (s² − 1)(u₊ + u₋); where
    # the loss is locally linear a good step and its mirror take opposite
    # ranks and those utilities cancel, so σ barely adapts and stays near
    # init_sigma. Set False to reproduce GPUMD's sampler.
    mirrored_sampling: bool = False
    # GPUMD's data procedure (fitness.cu report_error + snes.cu's loop):
    # every frame of data_path trains, there is NO validation split, and the
    # monitor loop runs on the EXTERNAL TEST SET purely to be reported. It
    # never selects: GPUMD writes its potential from the current generation
    # and tracks no best-ever model, so best_val_model == final_model here and
    # cfg.patience is inert. That is what keeps monitoring the test set honest
    # — selecting on it would leak. Requires cfg.test_data_path.
    gpumd_path: bool = True
    # Lower bound on sigma after each SNES update, preventing σ→0 collapse of
    # the search distribution on long runs. None or 0 = disable; 1e-6 is
    # non-distorting (typical adapted σ is 1e-3 to 1e-1).
    sigma_floor: float | None = 1e-8

    # --- validation ----------------------------------------------------
    # Number of structures in each validation step (None = use entire val set)
    val_size: int | None = None
    # Validate every N generations (1 = every gen, 10 = every 10th, etc.)
    val_interval: int = 100

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

    def __init__(self) -> None:
        # Give every instance its OWN mutable containers. Without this, the
        # class-level list/dict defaults are shared objects — an in-place
        # append/assign on a fresh TNEPconfig() would leak into every other
        # instance in the process (e.g. back-to-back configs in a sweep).
        self.types = []
        self.type_map = {}
        if type(self).allowed_species is not None:
            self.allowed_species = list(type(self).allowed_species)

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



# ═══════════════════════════════════════════════════════════════════════════
# GAP kernel-fit config — DELIBERATELY A DUPLICATE of the TNEPconfig fields
# the GAP path needs (declaration, default and comment copied verbatim).
# No shared base class, no inheritance: each model takes only its own
# config, so an ANN experiment value left in TNEPconfig cannot perturb a
# kernel fit. The shared machinery (data.py, DescriptorBuilder*.py,
# gap_descriptor.py, model_io._serialize_config) is duck-typed on
# cfg.<field> and type(cfg).__annotations__, so matching field NAMES flow
# through it untouched — no getattr defaults, no compatibility shims.
# ═══════════════════════════════════════════════════════════════════════════


class GAPconfig:
    """Hyperparameters and runtime state for a GAP-mode dipole fit.

    Runtime-state fields (num_types, types, dim_q, indices) are overwritten
    by data.collect / the descriptor builder during fit_gap_dipole.
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
    # Seed for randomisation (dataset shuffle, FPS, centre selection)
    seed: int | None = 928375439201

    # --- target type, units, conversions -------------------------------
    # 0 : PES (energy), 1 : Dipole, 2 : Polarizability
    target_mode: int = 1
    # Override the info/results key used to read targets from ASE structures.
    # None = use the default for target_mode ("energy", "dipole", "pol").
    # Set to a custom string to support non-standard dataset labels (e.g. "mu", "alpha").
    target_key: str | None = None
    # Scale dipole targets by atom count (per-atom dipole training). Unlike
    # the ANN default (True), the kernel fit requires total dipoles: it
    # solves for μ itself, so a bare GAPconfig() must satisfy _validate.
    scale_targets: bool = False
    # Native units of dipole targets in the dataset: "e*angstrom" (e·Å),
    # "e*bohr" (e·a₀, × 0.5292 → e·Å), "debye" (× 0.2082 → e·Å). Sets the
    # e·Å conversion factor and the plot/stats unit label.
    dipole_units: str = "e*bohr"
    # True = convert dipole targets to e·Å on load; False = train/plot in
    # dataset native units (for RMSE comparable with e·a₀ / Debye references).
    convert_dipole_to_eangstrom: bool = False

    # Power N of |r_ij| in the dipole contraction (target_mode=1 only):
    #   0 = self-pair-only, μ = −Σ_i de_dq[i]·grad_values[i,i] (distinct form)
    #   1 = first radial moment, μ = −Σ_pair |r_ij| ·F_ij
    #   2 = Xu et al. JCTC 2024 / GPUMD default, μ = −Σ_pair |r_ij|²·F_ij
    #   ≥3 = higher moments (diagnostic)
    # Models must be re-trained if N changes.
    dipole_rij_power: int = 0

    # Dipole readout form (target_mode=1 only):
    #   "tnep"     : descriptor-gradient form above (dipole_rij_power applies)
    #   "neighbor" : neighbor basis with C radial channels and species-pair
    #                coefficients:
    #                  b_ij = Σ_n c_n[p(t_i,t_j)]·R_n(r_ij)·(s_i^n − s_j^n)
    #                  μ = Σ_{i<j} b_ij·(r_i − r_j)   (MIC displacements)
    #                R_n(r) = T_n(2r/r_c − 1)·f_c(r) (Chebyshev, 1st kind;
    #                f_c per dipole_cutoff)
    # The kernel fit needs μ linear in the site scalars: "neighbor" only
    # (enforced by gap_dipole._validate).
    dipole_readout: str = "neighbor"
    # Number of Chebyshev radial channels C of the neighbor readout (n =
    # 0..C−1). C=1 with c≡1 reduces to the plain f_c single-channel form.
    # Widens the site head W1 to [T, H, C]. Validated ≥ 1 at model init.
    # Models must be re-trained if C changes.
    dipole_channels: int = 20
    # Radial family of the neighbor readout (f_c per dipole_cutoff):
    #   "chebyshev"        : R_n(r) = T_n(2r/r_c − 1)·f_c(r), C channels
    #   "chebyshev_warped" : R_n(r) = T_n(x_w)·f_c(r), NEP's squared warp
    #                        x_w = 2(r/r_c − 1)² − 1 — resolution
    #                        concentrated at short range, C channels
    #   "gaussian"         : R_n(r) = exp(−(r − μ_n)²/2σ²)·f_c(r), μ_n the
    #                        C interior points of linspace(0, r_c, C+2),
    #                        σ = dipole_gauss_sigma_scale × spacing
    #   "screened"         : R(r) = r^(ζ−1)·f_c(r) — single channel
    #                        (requires dipole_channels == 1).
    dipole_radial: str = "gaussian"
    # Screened-radial exponent ζ (dipole_radial="screened" only). Default −2
    # from the rigid-charge LSQ scan: floors 2.70 at ζ=−1, 2.72 at ζ=−2,
    # 11.0 at ζ=+1 (ζ>0 inverts the covalent/H-bond weighting). −2 chosen
    # over the marginally-lower −1 deliberately: ~equal floor, stronger
    # H-bond suppression (Coulomb-like r^−2 screening).
    dipole_zeta: float = -3.0
    # Gaussian-window width scale σ/spacing (dipole_radial="gaussian" only):
    # σ = scale × r_c/(C+1) (C=1 ⇒ scale·r_c/2). Step-0 rigid-charge LSQ
    # evidence (diagnostics/diag_basis.py): scale 0.5 lowers the C=4 floor
    # 22% (1.16 vs 1.49) while 1.0 edges it at C=10 — the knob matters.
    dipole_gauss_sigma_scale: float = 1.0
    # Cutoff function f_c of the neighbor readout, crossable with any
    # radial. Smoothness at r_c is a property of the CUTOFF, not the radial:
    #   "cosine2" : ((1+cos(πr/r_c))/2)² — C³ contact at r_c
    #   "cosine"  : (1+cos(πr/r_c))/2 — C¹ contact (plain NEP cutoff)
    #   "cubic"   : 1 − 3x² + 2x³, x = r/r_c — C¹ contact (f″(r_c) ≠ 0)
    #   "tanh3"   : tanh³(1 − r/r_c)/tanh³(1) — C² contact (Behler cutoff,
    #               normalized so f(0) = 1)
    # None = the radial family's traditional form: "cubic" for screened,
    # "cosine2" otherwise (resolved_dipole_cutoff; validated at model init).
    dipole_cutoff: str | None = "cosine"
    # Species-pair coefficient table of the neighbor readout:
    #   "off"       : c ≡ 1 frozen, no parameters.
    #   "symmetric" : learnable c [C, T(T+1)/2], unordered-pair indexed.
    #   "ordered"   : learnable Ω [C, T²], ordered (t_center, t_neighbor)
    #                 indexed — breaks the difference-form antisymmetry (and
    #                 gauge invariance) across species.
    # In the GAP fit the distinction collapses: any non-"off" value keeps the
    # neighbour-type axis of V as design columns, which subsumes both tables.
    dipole_species_pairs: str = "symmetric"
    # Neighbor-readout cutoff r_c (Å). None = inherit rcut_hard (resolved at
    # model init). Must be ≤ rcut_hard and satisfy dipole_rc + rcut_hard ≤
    # L_min for every training box (checked at data assembly). Note the
    # "cosine2" cutoff narrows effective support vs the plain cosine.
    dipole_rc: float | None = None

    # --- GAP-mode dipole fit (gap_dipole.py) -----------------------------
    # FPS-selected sparse environments per species for the kernel basis.
    gap_sparse_per_species: int = 40
    # Power(s) ξ of the normalized SOAP dot-product kernel (D̂_i·D̂_m)^ξ.
    # An int is a single kernel; a list (e.g. [1, 2, 4]) is a sum-of-kernels
    # fit — each sparse point contributes one column block PER ξ and the
    # ridge learns the implicit kernel mixture. Entries must be ints ≥ 1.
    gap_kernel_power: int | list[int] = 2
    # Second SOAP length scale for the KERNEL only (gap_dipole.py): a second
    # descriptor builder with rcut_hard = gap_extra_rcut and rcut_soft =
    # gap_extra_rcut − (rcut_hard − rcut_soft) (same buffer width), its own
    # FPS sparse sets, and additional Φ column blocks (scales never cross,
    # like species; the geometry vectors v are unchanged). Typically LARGER
    # than rcut_hard — adds long-sighted environment resolution (multi-scale
    # SOAP ~25%: Willatt/Musil/Ceriotti PCCP 2018). None = single scale.
    gap_extra_rcut: float | None = None
    # δ is each body order's amplitude, exactly gap_fit's `delta`: the
    # kernel carries δ² (gap.f90 `delta**2 * alphas * kernel`), on the design
    # columns AND on the sparse-sparse kernel that regularises the fit, so δ²
    # is the prior variance of that term's site function. TurboGAP's
    # published potentials use 0.5 (2b), 0.1 (SOAP), 0.01 (3b).
    # (the 2b/3b blocks are inert unless gap_2b/gap_3b).
    gap_delta_soap: float = 0.1
    gap_delta_2b: float = 0.5
    gap_delta_3b: float = 0.01
    # Regularisation, as gap_fit: the expected ABSOLUTE error of a dipole
    # component per √atom (gap_fit's default_sigma with
    # sigma_per_atom=T: σ_obs = gap_sigma·√N_atoms), and the jitter added to
    # the sparse-sparse kernel before factorising (gap_fit default 1e-10).
    gap_sigma: float = 1e-8
    gap_sparse_jitter: float = 1e-10

    # ═══════════════════════════════════════════════════════════════════
    # 2. DESCRIPTOR (SOAP-turbo)
    # ═══════════════════════════════════════════════════════════════════

    # --- geometric parameters ------------------------------------------
    l_max: int = 8
    alpha_max: int = 8
    rcut_hard: float = 4.0
    rcut_soft: float = 3.5
    basis: str = "poly3gauss"
    scaling_mode: str = "polynomial"
    radial_enhancement: int = 0
    # "trivial" : soap_turbo's traditional compression recipe (the only mode
    #             the GPU builder, descriptor_mode=1, implements).
    # "none"    : the uncompressed power spectrum, N(N+1)/2*(l_max+1) over the
    #             N = num_types*alpha_max radial channels. This is what
    #             TurboGAP's soap_turbo type defaults to (types.f90 keeps
    #             compress_mode="none" behind its own compress_soap flag).
    #             On the quippy path (descriptor_mode=0) the keyword is OMITTED
    #             from the descriptor string, because soap_turbo's
    #             get_compress_indices rejects the literal "none"
    #             (soap_turbo_compress.f90:292).
    # "linear"  : quippy's linear compression, dim = 2*num_types*(l_max+1).
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
    descriptor_mode: int = 0

    # Structures per SOAP graph call: 1 = per-frame (lowest VRAM), int>1 =
    # batched, None = auto-fit to memory budget. Used by training and
    # trajectory inference; process_trajectory kwarg overrides per call.
    descriptor_batch_frames: int | None = 100

    # Structures per batch of the self-only gradient build (data.py). The
    # kernel fit builds no gradients at all, so this is inert here.
    descriptor_self_batch_size: int | None = None

    # ── FROZEN CONSTANTS — ANN-input modes the shared machinery reads ───
    # data.py / DescriptorBuilderGPU*.py branch on these, so they must hold a
    # definite value, but the kernel fit uses NONE of them: no NEP inputs, no
    # SNES search geometry to precondition, no learned radial basis. They are
    # pinned HERE (not inherited from an ANN config) so that no ANN experiment
    # can perturb a GAP fit. Do not turn them into knobs — the fit is not
    # defined for any other value.
    nep_descriptor: str = "off"
    descriptor_scale: str = "off"
    radial_basis_size: int | None = None
    radial_basis_norm: str = "l2"
    # Read only when nep_descriptor != "off" (the NEP pair-padding
    # displacement in data.assemble_data_dict) — inert here; TNEPconfig's
    # defaults, kept so the read has a value.
    nep_rc_radial: float = 8.0
    nep_rc_angular: float = 4.0

    # --- GAP 2b/3b distance descriptors --------------------------------
    # TurboGAP 2-body / 3-body distance descriptors (gap_descriptor.py,
    # gap_descriptor_plan.md §0.1): GAP's neighbour sums with FIXED centres
    # (sparse points) and widths, "exp" kernel only (no "pol"). Used by the
    # ANN nep_descriptor="gap_sum" experts and the GAP-mode column blocks.
    gap2b_rcut: float = 4.0              # GAP rcut of the 2b term
    gap3b_rcut: float = 2.0              # GAP rcut of the 3b term
    gap_buffer: float = 0.5              # GAP buffer: cosine taper width inside rcut
    gap2b_n_centres: int = 40            # M2 kernel centres on r
    gap2b_sigma: float = 1.0             # 2b kernel width (Å)
    gap3b_n_centres: int = 200            # M3 kernel centres on q = [r_ij+r_ik, (r_ij−r_ik)², r_jk]
    gap3b_sigma: list[float] = [2.0, 2.0, 2.0]  # per q component
    gap3b_kernel: str = "pol"            # "exp" | "pol" (see TNEPconfig)
    # Runtime: auto-selected by gap_descriptor.select_centres when None
    # (2b uniform grid, 3b farthest-point sampling under cfg.seed) and saved
    # with the model like q_scale; shapes [M2] and [M3, 3].
    gap2b_centres: list[float] | None = None
    gap3b_centres: list[list[float]] | None = None
    # GAP-mode (gap_dipole.py) additive linear column blocks.
    gap_2b: bool = True
    gap_3b: bool = True

    # ═══════════════════════════════════════════════════════════════════
    # 6. MEMORY & I/O STAGING
    # ═══════════════════════════════════════════════════════════════════

    # Structures per GPU chunk during evaluation. None = all at once; 2000
    # fits the dev box (CPU CSC profile uses 500, latency-sensitive).
    batch_chunk_size: int | None = 2000

    # Where static training tensors (descriptors, grad_values, positions, pair
    # indices) live: True = host CPU, per-chunk gather with H2D copy each gen
    # (for datasets too big for VRAM); False = /GPU:0, on-device gather
    # (fastest when the working set fits in VRAM).
    pin_data_to_cpu: bool = False

    # ═══════════════════════════════════════════════════════════════════
    # 8. RUNTIME STATE — populated by data.collect / the builder
    # ═══════════════════════════════════════════════════════════════════

    dim_q: int
    num_types: int
    types: list[int] = []
    type_map: dict = {}
    indices: np.ndarray

    def __init__(self) -> None:
        # Give every instance its OWN mutable containers. Without this, the
        # class-level list/dict defaults are shared objects — an in-place
        # append/assign on a fresh GAPconfig() would leak into every other
        # instance in the process (e.g. back-to-back configs in a sweep).
        self.types = []
        self.type_map = {}
        if type(self).allowed_species is not None:
            self.allowed_species = list(type(self).allowed_species)

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

def resolved_dipole_rc(cfg: "TNEPconfig | GAPconfig") -> float:
    """Active neighbour-readout radius — THE dipole_rc=None rule.

    None inherits the cutoff of whatever descriptor actually feeds the model:
    nep_rc_radial under nep_descriptor="replace", where the SOAP block is
    dropped entirely and rcut_hard means nothing to the model, and rcut_hard
    otherwise. Single source: data assembly builds V before the model exists,
    so TNEP.__init__ and data._neighbor_dipole_V must resolve identically.

    Accepts EITHER config — it reads only dipole_rc, nep_descriptor,
    nep_rc_radial and rcut_hard, which both carry."""
    if cfg.dipole_rc is not None:
        return float(cfg.dipole_rc)
    if str(getattr(cfg, "nep_descriptor", "off")) == "replace":
        return float(cfg.nep_rc_radial)
    return float(cfg.rcut_hard)


def resolved_dipole_cutoff(cfg: TNEPconfig | "GAPconfig") -> str:
    """Active neighbor-readout cutoff name — THE dipole_cutoff=None rule
    (None → the radial family's traditional form). Single source: data
    assembly builds V before the model exists, so both TNEP.__init__ and
    data._neighbor_dipole_V resolve through here.

    Accepts EITHER config (TNEPconfig or GAPconfig) — it reads only
    dipole_cutoff/dipole_radial, which both carry, so it is not duplicated."""
    return cfg.dipole_cutoff or (
        "cubic" if cfg.dipole_radial == "screened" else "cosine2")
