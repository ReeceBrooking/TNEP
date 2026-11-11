"""GAPconfig: configuration of the sparse-kernel (GAP) dipole / polarisability fit
(gap_dipole.py). Its own module so the GAP code never imports the ANN config; the
shared readout rules in TNEPconfig.py are duck-typed on cfg.<field>."""
from __future__ import annotations

import numpy as np


# DELIBERATE DUPLICATE of the TNEPconfig fields the GAP path needs — no shared
# base class, so an ANN experiment value in TNEPconfig cannot perturb a kernel
# fit. The shared machinery (data.py, DescriptorBuilder*.py, model_io) is
# duck-typed on cfg.<field> / __annotations__, so matching names flow through.


class GAPconfig:
    """Hyperparameters and runtime state for a GAP-mode dipole fit.

    Runtime-state fields (num_types, types, dim_q, indices) are overwritten
    by data.collect / the descriptor builder during fit_gap_dipole.
    """

    # ── FROZEN CONSTANTS (plain attributes, not config fields): ANN modes the shared
    # data/descriptor path reads; the kernel fit is defined for exactly these values.
    # Not in __annotations__: not knobs, not serialised. ──
    dipole_rij_power = 0
    dipole_readout = "neighbor"
    scale_targets = False
    descriptor_self_batch_size = None
    batch_chunk_size = 2000
    pin_data_to_cpu = False
    nep_descriptor = "off"
    descriptor_scale = "off"
    radial_basis_size = None
    radial_basis_norm = "l2"
    nep_rc_radial = 8.0
    nep_rc_angular = 4.0

    # ── 1. Dataset & targets ──

    # --- dataset & split -----------------------------------------------
    data_path: str = "datasets/train_waterbulk.xyz"
    # Separate test dataset (None = split from data_path; str = path to external .xyz)
    test_data_path: str | None = "datasets/test_waterbulk.xyz"
    # Keep only structures whose species are in this list (None = no filter; ints or symbols)
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
    # Native units of dipole targets in the dataset: "e*angstrom" (e·Å),
    # "e*bohr" (e·a₀, × 0.5292 → e·Å), "debye" (× 0.2082 → e·Å). Sets the
    # e·Å conversion factor and the plot/stats unit label.
    dipole_units: str = "e*bohr"
    # True = convert dipole targets to e·Å on load; False = train/plot in
    # dataset native units (for RMSE comparable with e·a₀ / Debye references).
    convert_dipole_to_eangstrom: bool = False

    # Radial channels C of the neighbor readout (n = 0..C−1).
    # Models must be re-fitted if C changes.
    dipole_channels: int = 20
    # Radial family of the neighbor readout, all × f_c(r) (dipole_cutoff):
    #   "chebyshev"        : T_n(2r/r_c − 1)
    #   "chebyshev_warped" : T_n(2(r/r_c − 1)² − 1), NEP's warp (short-range resolution)
    #   "gaussian"         : exp(−(r − μ_n)²/2σ²), μ_n = C interior points of
    #                        linspace(0, r_c, C+2), σ = dipole_gauss_sigma_scale × spacing
    #   "screened"         : r^(ζ−1), single channel (dipole_channels must be 1)
    dipole_radial: str = "gaussian"
    # Screened-radial exponent ζ (dipole_radial="screened" only); ζ<0 = Coulomb-like
    # suppression of long (H-bond) pairs, ζ>0 inverts the covalent/H-bond weighting.
    dipole_zeta: float = -3.0
    # Gaussian-window width σ = scale × r_c/(C+1) (dipole_radial="gaussian" only).
    dipole_gauss_sigma_scale: float = 1.0
    # EXPERIMENTAL (out of the production models). Long-range field channel
    # (target_mode=1): one extra kernel channel carrying the Ewald field E_i of
    # fixed reference charges (dipole_field_charges), so μ ⊃ Σ_i s_i^E·E_i.
    # Staged at data assembly only; in-graph V builders refuse.
    dipole_field: bool = False
    dipole_field_charges: dict | None = None        # e.g. SPC/E {8: -0.8476, 1: 0.4238}
    # Scale on E to match the radial channels; None = derive once at the first
    # data assembly as RMS(Σ_t' V)/RMS(E) and pin here (one constant per model).
    dipole_field_scale: float | None = None
    # Cutoff f_c of the neighbour readout: "cosine2" = ((1+cos(πr/r_c))/2)² (C³ at
    # r_c), "cosine" = (1+cos(πr/r_c))/2 (plain NEP, C¹). None = "cosine2".
    dipole_cutoff: str | None = "cosine"
    # Species-pair table: "off" = c ≡ 1; anything else keeps the neighbour-type
    # axis of V as design columns, which subsumes both the ANN readout's symmetric
    # c and ordered Ω tables.
    dipole_species_pairs: str = "symmetric"

    # --- GAP-mode dipole fit (gap_dipole.py) -----------------------------
    # FPS-selected sparse environments per species for the kernel basis.
    gap_sparse_per_species: int = 40
    # Power ξ of the normalised SOAP kernel (D̂_i·D̂_m)^ξ (int ≥ 1).
    gap_kernel_power: int = 2
    # Second SOAP length scale for the KERNEL only: a second builder with
    # rcut_hard = gap_extra_rcut (same soft/hard buffer), its own FPS sets and extra
    # Φ column blocks (scales never cross; v unchanged). None = single scale.
    # Multi-scale SOAP: Willatt/Musil/Ceriotti PCCP 2018.
    gap_extra_rcut: float | None = None
    # Site-function amplitude δ, exactly gap_fit's `delta`: δ² on the design columns
    # AND on the K_MM prior (prior variance of the site function); gap_model_delta
    # scales it per site model.
    gap_delta_soap: float = 0.1
    # --- polarisability (target_mode=2) kernel groups ----------------------
    # Off-diagonal observation weight (GPUMD lambda_shear): xy, yz, zx rows fitted
    # with σ_obs / gap_lambda_shear.
    gap_lambda_shear: float = 1.0
    # Cross (two-neighbour) kernel group, C_x radial channels of v_i^a = Σ_j R_a(r_ij)·d_ij
    # (species-resolved under pair columns). Mode 2: traceless sym(v^a ⊗ v^b), a ≤ b
    # (the d_ij⊗d_ik bond-angle anisotropy); mode 1: (Σ_j R_a d_ij d_ijᵀ)·v^b (the
    # bond-angle charge flux the one-hop difference form cannot carry). 0 = off.
    gap_cross_channels: int = 0
    # --- site models (GAP only): a kernel per body order, summed ------------
    # Each s_i = Σ_m α_m (q̂_i·q̂_m)^ξ on ITS OWN descriptor, FPS set and K_MM prior
    # (never mixed): "soap"; "2b" = radial distribution Σ_{j∈t'} R_n(r_ij); "3b" =
    # angular power spectrum Σ R_n(r_ij) R_n'(r_ik) P_l(cos θ_jik) (gap_models.py).
    # soap/2b read out through the bond basis (+ field channel), 3b through its own
    # angular kernel (gap_cross_lmax) — the only way 3b keeps an uncontracted bond vector.
    gap_site_models: list[str] = ["soap"]
    # Relative amplitude δ per site model (× gap_delta_soap); keys must cover gap_site_models.
    gap_model_delta: dict = {"soap": 1.0, "2b": 1.0, "3b": 1.0}
    # Radial channels C and Legendre order L of the 2b/3b DESCRIPTORS (the
    # readout channels stay dipole_channels). Width: 2b T·C, 3b T(T+1)/2·C²·(L+1).
    gap_site_channels: int = 6
    gap_site_lmax: int = 2
    # Legendre order L of the 3b angular kernel: channels (a ≤ b, l), l = 1..L for μ,
    # 0..L for α (the α l = 0 member is sym(v⊗v) without its j = k term).
    gap_cross_lmax: int = 1
    # gap_fit regularisation: expected ABSOLUTE error of a dipole component per √atom
    # (σ_obs = gap_sigma·√N_atoms; gap_fit default_sigma with sigma_per_atom=T).
    # gap_sparse_jitter is added to K_MM before factorising (gap_fit default 1e-10).
    gap_sigma: float = 1e-8
    # Neighbour-readout geometry from the UNIT bond vector û (rank 1: û; rank 2: I
    # and ûûᵀ − I/3) instead of d, so R_n(r)·f_c(r) carries all r-dependence; û
    # columns are r² smaller, so prefer σ ≈ 0.03–0.1. GAP-only (the ANN keeps d).
    dipole_unit_vector: bool = True
    gap_sparse_jitter: float = 1e-10

    # ── 2. Descriptor (SOAP-turbo) ──

    # --- geometric parameters ------------------------------------------
    l_max: int = 8
    alpha_max: int = 8
    rcut_hard: float = 4.0
    rcut_soft: float = 3.5
    basis: str = "poly3gauss"
    scaling_mode: str = "polynomial"
    radial_enhancement: int = 0
    # "trivial" : soap_turbo's compression (the only mode the GPU builder implements)
    # "none"    : uncompressed power spectrum, N(N+1)/2·(l_max+1), N = num_types·alpha_max
    #             (TurboGAP's default). On quippy the keyword is OMITTED from the
    #             descriptor string: get_compress_indices rejects "none"
    #             (soap_turbo_compress.f90:292).
    # "linear"  : quippy's linear compression, dim = 2·num_types·(l_max+1)
    compress_mode: str = "trivial"
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

    # ── 8. Runtime state (populated by data.collect / the builder) ──

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
