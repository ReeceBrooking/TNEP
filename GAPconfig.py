"""Config for the GAP kernel-fit path (gap_dipole.py) — independent of TNEPconfig.

DELIBERATELY A DUPLICATE. Every field the GAP fit needs is copied verbatim
from TNEPconfig (declaration, default and comment); there is NO shared base
class, no inheritance, no mixin. Each model takes only its own config: a GAP
fit reads GAPconfig, the ANN reads TNEPconfig, and an ANN experiment value
left in the working tree's TNEPconfig can no longer perturb a kernel fit.

The shared machinery (data.py, DescriptorBuilder*.py, gap_descriptor.py,
model_io._serialize_config) is duck-typed on `cfg.<field>` and on
`type(cfg).__annotations__`, so a GAPconfig whose field NAMES match flows
through it untouched — no getattr defaults, no compatibility shims.

Sections mirror TNEPconfig's numbering (the ANN-only sections 3-7 are absent):
    1. Dataset & targets (incl. the neighbour readout and the GAP fit)
    2. Descriptor (SOAP-turbo: geometry, backend) + GAP 2b/3b blocks
    6. Memory & I/O staging (only what data.py reads)
    8. Runtime state (auto-populated by data load)
"""
from __future__ import annotations

import numpy as np


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
    # Scale dipole targets by atom count (per-atom dipole training)
    scale_targets: bool = True
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
    # Ridge strength λ, applied RELATIVE to mean(diag(ΦᵀΦ)) — dimensionless
    # (λ_eff = λ · mean_m ‖Φ[:,m]‖²), so its scale is data-independent.
    gap_ridge_lambda: float = 1e-6

    # ═══════════════════════════════════════════════════════════════════
    # 2. DESCRIPTOR (SOAP-turbo)
    # ═══════════════════════════════════════════════════════════════════

    # --- geometric parameters ------------------------------------------
    l_max: int = 4
    alpha_max: int = 4
    rcut_hard: float = 4.0
    rcut_soft: float = 3.5
    basis: str = "poly3"
    scaling_mode: str = "polynomial"
    radial_enhancement: int = 0
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
    gap2b_rcut: float = 3.5              # GAP rcut of the 2b term
    gap3b_rcut: float = 3.5              # GAP rcut of the 3b term
    gap_buffer: float = 0.5              # GAP buffer: cosine taper width inside rcut
    gap2b_n_centres: int = 20            # M2 kernel centres on r
    gap2b_sigma: float = 0.3             # 2b kernel width (Å)
    gap3b_n_centres: int = 50            # M3 kernel centres on q = [r_ij+r_ik, (r_ij−r_ik)², r_jk]
    gap3b_sigma: list[float] = [0.5, 0.5, 0.5]  # per q component
    # Runtime: auto-selected by gap_descriptor.select_centres when None
    # (2b uniform grid, 3b farthest-point sampling under cfg.seed) and saved
    # with the model like q_scale; shapes [M2] and [M3, 3].
    gap2b_centres: list[float] | None = None
    gap3b_centres: list[list[float]] | None = None
    # GAP-mode (gap_dipole.py) additive linear column blocks.
    gap_2b: bool = False
    gap_3b: bool = False

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
