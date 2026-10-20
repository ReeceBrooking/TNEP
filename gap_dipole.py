"""GAP-mode dipole fit for the neighbor readout.

Sparse Gaussian-process alternative to the SNES-trained ANN, exploiting the
readout's linearity in the site scalars (neighbor_dipole_gap_plan.md). The fit
is gap_fit's estimator (libAtoms/GAP gp_predict.F90 gpSparse_fit): observation
error gap_sigma·√N_atoms, penalty αᵀ·K_MM·α per GAP term, δ² on every kernel:

    μ = Σ_i Σ_k s_i^k v_i[:, k],  s_i^k = Σ_m α_m^k (D̂_i · D̂_m)^ξ

with v_i from data._neighbor_dipole_V [N, D, C, T] — any radial
(chebyshev/warped/gaussian/screened; screened requires C = 1, matching the
readout contract) and any cutoff. The k axis of v — and hence the design
columns per sparse point m — depends on dipole_species_pairs:

    "off":         k = channel n only, V summed over neighbor type t′ (K = C);
    anything else: k = (n, t′) kept (K = C·T) — per-(sparse-point, channel,
        neighbor-species) weights absorbed into the linear fit. This STRICTLY
        SUBSUMES both the readout's symmetric c and ordered Ω tables; the
        symmetric/ordered distinction does not exist in the absorbed form
        (any non-"off" value enables the same general columns).

Two multiplicative column extensions (both optional, both compose with
pairs/channels — columns multiply as n_scales × len(ξ) × M_tot·K):
  • sum-of-kernels: gap_kernel_power may be a LIST of ξ — one column block
    per ξ per sparse point (base Gram computed once, raised elementwise);
    the ridge learns the implicit kernel mixture (Deringer Chem. Rev. 2021).
  • multi-scale: gap_extra_rcut adds a second SOAP length scale for the
    KERNEL only (second builder + second FPS sparse sets; scales never
    cross). The geometry vectors v are UNCHANGED (Willatt/Musil/Ceriotti).

Summed site models (gap_site_models, gap_models.py): besides SOAP, a 2b
and/or 3b body-order kernel, each with its own FPS sparse set, prior and
relative amplitude (gap_model_delta); soap/2b read out through the bond
basis, 3b through its own angular geometric kernel (gap_cross_lmax).

Additive (not multiplicative) column blocks: gap_2b / gap_3b append TurboGAP's
2b/3b distance descriptors (gap_descriptor.py, fixed centres auto-selected at
fit) as per-species linear features — plan §0.3.

Columns = M_tot·K can exceed the 3·F equations (underdetermined); the prior
handles it — train RMSE approaches 0 there and test RMSE is the metric.
Takes its own GAPconfig (declared in TNEPconfig.py), never TNEPconfig: the ANN modes
are frozen constants there, so no ANN experiment can reach the fit.
Polarisability (target_mode=2) fits the same way with the rank-2 kernel
(D = 6 Voigt rows per frame instead of 3, and 2·C kernel channels — the
isotropic |d|²I and traceless d⊗d − |d|²I/3 groups; data.bond_geometry):
α = Σ_i Σ_k s_i^k W_i[:, k] — linear in the site scalars exactly as μ is.
Still fixed: dipole_readout="neighbor", scale_targets=False.
NO model persistence — fit writes a config+metrics run record JSON to GAP/."""
from __future__ import annotations

import copy
import json
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.io import read

import data as tnep_data
import gap_models as gm
from DescriptorBuilder import make_descriptor_builder
from gap_descriptor import (f_cut, gap_raw_numpy, gap_slices, select_centres,
                            validate_gap_cfg)
from TNEPconfig import GAPconfig, resolved_dipole_cutoff, bond_rank, kernel_channels


@dataclass
class GapDipoleModel:
    """In-memory GAP dipole model — never persisted (refits in seconds).

    Column/α block order (one contract, shared by fit + predict via
    _phi_frame): site model (cfg.gap_site_models order) → species → ξ →
    (m, k). Per (model, species) the block is [len(xi)·M_t·K_model] flat,
    ξ-major; K_model is the bond-basis width for soap/2b and the angular
    kernel width for 3b. site_sparse / site_alpha are keyed by model name;
    `sparse_desc` / `alpha` are the "soap" entries (read-only views for the
    callers that only know the SOAP model). The scale-2 fields are None
    unless cfg.gap_extra_rcut is set (appended after the site models). Then
    the additive TurboGAP blocks
    (gap_descriptor_plan.md §0.3): alpha_gap2b / alpha_gap3b per species t,
    [W2·K] / [W3·K] flat (s-major), None unless cfg.gap_2b / gap_3b.

    The fitted alpha_gap* values are NOT interpretable per-centre weights.
    The gap block is rank-deficient and badly conditioned (unpopulated
    species-pair slots give identically-zero columns; cond ~1e18 raw, ~1e6
    after column normalisation on the test cfg), so predictions are
    recovered to the ridge floor while α itself wanders along a near-null
    space — measured |Δα| ~0.8 between fits even at 200 frames."""
    cfg: GAPconfig
    builder: object                   # descriptor builder from the fitting cfg
    site_sparse: dict                 # model → per type t: [M_t, Q] unit-norm rows
    site_alpha: dict                  # model → per type t: [len(xi)·M_t·K_model] flat
    xi: tuple[int, ...]               # kernel powers — one column block each
    builder2: object = None           # second-scale builder (gap_extra_rcut)
    sparse_desc2: list[np.ndarray] | None = None
    alpha2: list[np.ndarray] | None = None
    alpha_gap2b: list[np.ndarray] | None = None
    alpha_gap3b: list[np.ndarray] | None = None

    @property
    def sparse_desc(self) -> list[np.ndarray]:
        return self.site_sparse["soap"]

    @property
    def alpha(self) -> list[np.ndarray]:
        return self.site_alpha["soap"]


def _xi_tuple(cfg: GAPconfig) -> tuple[int, ...]:
    """Normalize cfg.gap_kernel_power (int | list[int]) to a tuple.
    Each entry contributes one column block per sparse point (sum-of-kernels:
    the ridge learns the implicit mixture of (D̂·D̂_m)^ξ kernels)."""
    xi = cfg.gap_kernel_power
    xis = (xi,) if not hasattr(xi, "__iter__") else tuple(xi)
    if not xis or any(isinstance(x, bool) or x != int(x) or int(x) < 1
                      for x in xis):
        raise ValueError(
            f"gap_kernel_power entries must be ints >= 1, got {xi!r}")
    return tuple(int(x) for x in xis)


def _validate(cfg: GAPconfig) -> None:
    """Raise unless cfg is a GAP-fittable configuration. Runs FIRST in fit."""
    _xi_tuple(cfg)
    extra = cfg.gap_extra_rcut
    if extra is not None:
        if not extra > 0.0:
            raise ValueError(f"gap_extra_rcut must be > 0, got {extra}")
        if extra <= cfg.rcut_hard:
            warnings.warn(
                f"gap_extra_rcut={extra} <= rcut_hard={cfg.rcut_hard}: legal, "
                f"but the second scale usually adds LONGER-range resolution")
    checks = [
        (cfg.target_mode in (1, 2),
         f"target_mode must be 1 (dipole) or 2 (polarisability), got "
         f"{cfg.target_mode}"),
        (cfg.dipole_readout == "neighbor",
         f"dipole_readout must be 'neighbor', got {cfg.dipole_readout!r}"),
        (int(cfg.dipole_channels) >= 1,
         f"dipole_channels must be >= 1, got {cfg.dipole_channels}"),
        # readout contract: the screened radial is single-channel.
        (cfg.dipole_radial != "screened" or int(cfg.dipole_channels) == 1,
         f"dipole_channels must be 1 for the screened radial, got "
         f"{cfg.dipole_channels}"),
        (not cfg.scale_targets, "scale_targets must be False"),
        (int(cfg.gap_cross_channels) >= 0, "gap_cross_channels must be >= 0"),
        (len(cfg.gap_group_delta) == 3 and all(float(x) > 0 for x in cfg.gap_group_delta),
         "gap_group_delta needs 3 positive entries (isotropic, traceless, cross)"),
        (float(cfg.gap_lambda_shear) > 0, "gap_lambda_shear must be > 0"),
        # nep_descriptor is a frozen constant on GAPconfig, so this cannot
        # fire from a class default — it catches INSTANCE mutation
        # (cfg.nep_descriptor = "concat"), which would otherwise reach
        # data.assemble_data_dict and build NEP inputs the fit never reads.
        (str(cfg.nep_descriptor) == "off",
         "GAP mode needs nep_descriptor='off' (NEP inputs are not part of "
         "the kernel fit)"),
        # Any positive radius, as at model init (47ae1cc): the staged V comes
        # from an image-exact neighbour list, so r_c > rcut_hard is legal.
        (0.0 < float(cfg.dipole_rc or cfg.rcut_hard),
         f"dipole_rc must be > 0, got {cfg.dipole_rc}"),
        (bool(cfg.gap_site_models) and set(cfg.gap_site_models) <= {"soap", "2b", "3b"}
         and len(set(cfg.gap_site_models)) == len(cfg.gap_site_models),
         f"gap_site_models must be a non-empty subset of soap/2b/3b without repeats, "
         f"got {cfg.gap_site_models!r}"),
        (set(cfg.gap_site_models) <= set(cfg.gap_model_delta)
         and all(float(cfg.gap_model_delta[m]) > 0 for m in cfg.gap_site_models),
         "gap_model_delta needs a positive entry for every entry of gap_site_models"),
        (int(cfg.gap_site_channels) >= 1 and int(cfg.gap_site_lmax) >= 0,
         "gap_site_channels >= 1 and gap_site_lmax >= 0 required"),
        (int(cfg.gap_cross_lmax) >= 0, "gap_cross_lmax must be >= 0"),
        (cfg.gap_site_rcut is None or float(cfg.gap_site_rcut) > 0,
         f"gap_site_rcut must be > 0, got {cfg.gap_site_rcut}"),
        (cfg.gap_site_rcut is None or {"2b", "3b"} & set(cfg.gap_site_models),
         "gap_site_rcut only applies to the 2b/3b site models"),
        ("3b" not in cfg.gap_site_models or int(cfg.gap_cross_channels) >= 1,
         "the 3b site model reads out through the angular kernel: gap_cross_channels >= 1 required"),
        ("3b" not in cfg.gap_site_models or int(cfg.target_mode) == 2 or int(cfg.gap_cross_lmax) >= 1,
         "3b dipole kernel needs gap_cross_lmax >= 1 (l = 0 collapses to the one-hop basis)"),
        (set(cfg.gap_site_models) <= {"soap"} or cfg.dipole_radial != "screened",
         "2b/3b site models need a multi-channel radial (screened is single-channel)"),
        (not (cfg.gap_2b or cfg.gap_3b) or list(cfg.gap_site_models)[:1] == ["soap"],
         "the TurboGAP 2b/3b tails read out through the bond basis of the first (soap) site model"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise ValueError("fit_gap_dipole config violation: " + "; ".join(bad))
    if cfg.gap_2b or cfg.gap_3b:
        validate_gap_cfg(cfg)          # centres may still be None: fit selects them


def _descriptors(builder, frames: list[Atoms]) -> list[np.ndarray]:
    """SOAP descriptors, no gradients, float64. Asserts ‖D‖ ≈ 1 (both
    builders already unit-normalize; we must NOT renormalize)."""
    desc, _, _ = builder.build_descriptors(frames, calc_gradients=False)
    out = []
    for D in desc:
        D = np.asarray(D, dtype=np.float64)
        norms = np.linalg.norm(D, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"descriptors not unit-normalized "
            f"(max |norm-1| = {np.abs(norms - 1.0).max():.3g})")
        out.append(D)
    return out


def _collapse_V(V: np.ndarray, pair_columns: bool) -> np.ndarray:
    """[N, D, C, T] → v_i [N, D, K] — the ONE place the column contract is
    encoded (float64, shared by fit + predict). K = C·T with pair columns
    (k = (n, t′), C-major, matching reshape order) else K = C (t′ summed).
    D = 3 (dipole) or 6 (polarisability, Voigt)."""
    V = np.asarray(V, dtype=np.float64)
    if pair_columns:
        return V.reshape(V.shape[0], V.shape[1], -1)
    return V.sum(axis=-1)


def _cross_group(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Third polarisability kernel group [N, 6, K_x] (gap_cross_channels C_x):
    the dipole kernel v_i^a = Σ_j R_a(r_ij) d_ij over its first C_x radial
    channels (species-resolved under pair columns, K₁ = C_x·T′ vectors),
    combined as traceless symmetric products sym(v^a ⊗ v^b) − tr/3 for a ≤ b,
    K_x = K₁(K₁+1)/2. Expanding v⊗v = Σ_j Σ_k R_a R_b d_j⊗d_k shows it carries
    the two-neighbour cross tensors the bond-axis groups lack. Even under
    inversion, covariant, image-exact, linear in the site coefficient."""
    V1 = tnep_data._neighbor_dipole_V(
        atoms, ti, float(cfg.dipole_rc or cfg.rcut_hard),
        int(cfg.gap_cross_channels), int(cfg.num_types),
        str(cfg.dipole_radial), float(cfg.dipole_zeta),
        resolved_dipole_cutoff(cfg), float(cfg.dipole_gauss_sigma_scale),
        rank=1)                                                 # [N, 3, C_x, T]
    v = _collapse_V(V1, gm.pair_columns(cfg))                     # [N, 3, K₁]
    a_idx, b_idx = np.triu_indices(v.shape[2])
    a, b = v[:, :, a_idx], v[:, :, b_idx]                       # [N, 3, K_x]
    sym = 0.5 * (a[:, :, None, :] * b[:, None, :, :]
                 + b[:, :, None, :] * a[:, None, :, :])         # [N, 3, 3, K_x]
    tr3 = (sym[:, 0, 0] + sym[:, 1, 1] + sym[:, 2, 2]) / 3.0
    return np.stack([sym[:, 0, 0] - tr3, sym[:, 1, 1] - tr3, sym[:, 2, 2] - tr3,
                     sym[:, 0, 1], sym[:, 1, 2], sym[:, 2, 0]], axis=1)


def _cross_group_dipole(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Dipole cross group [N, 3, K_x] (gap_cross_channels C_x): the bond-axis
    tensor T_i^a = Σ_j R_a(r_ij) d_ij d_ijᵀ contracted with the dipole kernel
    vector v_i^b = Σ_j R_b(r_ij) d_ij — w_i^{ab} = T^a·v^b, both species-
    resolved under pair columns (K₁ = C_x·T′ each, K_x = K₁²). Expanded it is
    Σ_j Σ_k R_a R_b (d_ij·d_ik) d_ij: the bond-angle charge flux the one-hop
    difference form (s_i − s_j) d_ij cannot carry (the cocycle restriction of
    neighbor_dipole_formulation.md). Odd under inversion, covariant,
    image-exact, linear in the site coefficient."""
    args = (atoms, ti, float(cfg.dipole_rc or cfg.rcut_hard), int(cfg.gap_cross_channels),
            int(cfg.num_types), str(cfg.dipole_radial), float(cfg.dipole_zeta),
            resolved_dipole_cutoff(cfg), float(cfg.dipole_gauss_sigma_scale))
    pair = gm.pair_columns(cfg)
    V2 = tnep_data._neighbor_dipole_V(*args, rank=2)               # [N, 6, 2C_x, T]
    Cx = int(cfg.gap_cross_channels)
    # full d⊗d = traceless group + isotropic group / 3 (Voigt)
    Tv = _collapse_V(V2[:, :, Cx:] + V2[:, :, :Cx] / 3.0, pair)     # [N, 6, K₁]
    v = _collapse_V(tnep_data._neighbor_dipole_V(*args, rank=1), pair)   # [N, 3, K₁]
    xx, yy, zz, xy, yz, zx = (Tv[:, k] for k in range(6))           # each [N, K₁]
    # (T·v)_x = xx v_x + xy v_y + zx v_z, etc.; outer over (a, b) → [N, 3, K₁, K₁]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    o = lambda A, B: A[:, :, None] * B[:, None, :]
    wx = o(xx, vx) + o(xy, vy) + o(zx, vz)
    wy = o(xy, vx) + o(yy, vy) + o(yz, vz)
    wz = o(zx, vx) + o(yz, vy) + o(zz, vz)
    N = len(atoms)
    return np.stack([wx, wy, wz], axis=1).reshape(N, 3, -1)         # [N, 3, K₁²]


def _finish_v(V: np.ndarray, atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Staged kernel [N, D, Ck, T] → design geometry v_i [N, D, K]: collapse
    to pair columns, then scale each polarisability kernel group by its
    gap_group_delta (target_mode=2) and append the cross group (dipole:
    T·v vectors; polarisability: sym(v⊗v) tensors), scaled by
    gap_group_delta[2]. THE one place fit and predict both go through, so
    the column contract cannot drift."""
    v = _collapse_V(V, gm.pair_columns(cfg))
    d_iso, d_tr, d_x = (float(x) for x in cfg.gap_group_delta)
    if int(cfg.target_mode) == 2:
        per_group = v.shape[2] // 2                              # C·T′ channels per group
        v = v.copy()
        v[:, :, :per_group] *= d_iso
        v[:, :, per_group:] *= d_tr
    if int(cfg.gap_cross_channels) > 0:
        cross = (_cross_group if int(cfg.target_mode) == 2 else _cross_group_dipole)
        v = np.concatenate([v, d_x * cross(atoms, ti, cfg)], axis=2)
    return v


def _n_vcols(cfg: GAPconfig) -> int:
    """K: design geometry columns per sparse point (kernel channels × pair
    columns, + the cross group: K₁(K₁+1)/2 symmetric tensor products under
    target_mode=2, K₁² T·v vectors under target_mode=1; K₁ = C_x·T′)."""
    T_ = cfg.num_types if gm.pair_columns(cfg) else 1
    k = kernel_channels(cfg) * T_
    cx = int(cfg.gap_cross_channels)
    if cx > 0:
        k1 = cx * T_
        k += k1 * (k1 + 1) // 2 if int(cfg.target_mode) == 2 else k1 * k1
    return k


def _v_atoms(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Per-atom geometry vectors v_i [N, D, K] per the cfg's radial/cutoff/
    channels, pair-column mode, bond rank (target_mode) and polarisability
    kernel-group settings."""
    V = tnep_data._neighbor_dipole_V(
        atoms, ti, float(cfg.dipole_rc or cfg.rcut_hard),
        int(cfg.dipole_channels), int(cfg.num_types),
        str(cfg.dipole_radial), float(cfg.dipole_zeta),
        resolved_dipole_cutoff(cfg), float(cfg.dipole_gauss_sigma_scale),
        rank=bond_rank(cfg))
    if bool(getattr(cfg, "dipole_field", False)) and int(cfg.target_mode) == 1:
        # same staged channel as assemble_data_dict (scale pinned there at fit)
        V = tnep_data._append_field_channel(V, tnep_data.field_channel(atoms, cfg),
                                            float(cfg.dipole_field_scale))
    return _finish_v(V, atoms, ti, cfg)


def _site_descriptor(name: str, builder, frames: list[Atoms], tis: list, cfg: GAPconfig) -> list[np.ndarray]:
    """Per-frame descriptors of one site model (all unit rows): SOAP from the
    builder, 2b/3b from gap_models (already normalised there)."""
    if name == "soap":
        return _descriptors(builder, frames)
    fn = gm.two_body_descriptor if name == "2b" else gm.three_body_descriptor
    return [fn(a, ti, cfg) for a, ti in zip(frames, tis)]


def _site_geometry(name: str, frames: list[Atoms], tis: list, cfg: GAPconfig, bond_v: list) -> list:
    """Per-frame readout geometry [N, D, K_model]: the bond basis (+ field
    channel) for soap/2b; the angular kernel for 3b."""
    if name != "3b":
        return bond_v
    fn = gm.angular_cross_tensor if int(cfg.target_mode) == 2 else gm.angular_cross_vector
    return [fn(a, ti, cfg) for a, ti in zip(frames, tis)]


def _site_terms(cfg: GAPconfig, builder, frames: list[Atoms], tis: list, vs: list,
                descs: list | None = None) -> dict:
    """model → (per-frame descriptors, per-frame geometry [N, D, K_model], δ)
    in cfg.gap_site_models order — THE per-model inputs shared by fit and
    predict. `descs` reuses already-built SOAP descriptors."""
    d_soap = _deltas(cfg)[0]
    return {m: (descs if m == "soap" and descs is not None
                else _site_descriptor(m, builder, frames, tis, cfg),
                _site_geometry(m, frames, tis, cfg, vs),
                d_soap * float(cfg.gap_model_delta[m]))
            for m in cfg.gap_site_models}


def _frame_scales(site: dict, site_sparse: dict, f: int, descs2, sparse_desc2, v, d_soap) -> list:
    """_phi_frame's scales for frame f: one entry per site model, then the
    extra SOAP scale (gap_extra_rcut) reading out through the bond basis v."""
    scales = [(d[f], site_sparse[m], g[f], dm) for m, (d, g, dm) in site.items()]
    if descs2 is not None:
        scales.append((descs2[f], sparse_desc2, v, d_soap))
    return scales


def _fps(D: np.ndarray, M: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy max-min FPS on unit-norm rows: d² = 2 − 2 D·D_m. Seeded start."""
    M = min(M, len(D))
    idx = np.empty(M, dtype=np.intp)
    idx[0] = rng.integers(len(D))
    d2 = 2.0 - 2.0 * (D @ D[idx[0]])
    for m in range(1, M):
        idx[m] = np.argmax(d2)
        np.minimum(d2, 2.0 - 2.0 * (D @ D[idx[m]]), out=d2)
    return idx


def _phi_frame(scales: list, ti: np.ndarray, xis: tuple[int, ...],
               gap_q: tuple[np.ndarray | None, np.ndarray | None] | None = None,
               tail_deltas: tuple[float, float] = (1.0, 1.0),
               v: np.ndarray | None = None) -> np.ndarray:
    """[D, n_columns] design block for one frame (D = 3 dipole components or
    6 Voigt polarisability components) — THE column contract for fit AND
    predict. scales = [(D, sparse_desc, v, delta)] per site model: each
    model's kernel couples ITS descriptors to ITS sparse points only (models
    never cross, like species), reads out through ITS geometry v [N, D, K]
    and carries δ² on its columns and prior. Block order: model → species → ξ
    → (m, k); the base Gram is computed once per (model, species) and raised
    elementwise to each ξ (sum-of-kernels column blocks).

    gap_q = (q2 [N, W2] | None, q3 [N, W3] | None) appends the additive
    TurboGAP blocks AFTER all model columns: gap2b (species → (s, k)) then
    gap3b (species → (s, k)) — Σ_{i∈t} q[i, s]·v[i, x, k] — i.e. the 2b/3b
    site energies with a linear readout, μ-weighted by the bond basis `v`.

    tail_deltas = (δ_2b, δ_3b) of those tails (_deltas(cfg)[1:]); a site
    model's δ comes ONLY from its scales entry. The 2b/3b kernels carry δ²,
    as in gap.f90 (`delta**2 * alphas * kernel`) and gap_fit's covariance —
    so the columns here are δ²·k and the coefficients α are in gap_fit's
    convention. The same δ² sits on the sparse-sparse kernel K_MM that
    regularises the fit (_prior_factors), which is what makes δ² the prior
    variance of that term's site function."""
    d_2b, d_3b = (float(d) for d in tail_deltas)
    cols = []
    for D, sparse_desc, vm, dm in scales:
        n_rows = vm.shape[1]             # 3 (dipole) or 6 (polarisability)
        for t, Dm in enumerate(sparse_desc):
            sel = ti == t
            G = D[sel] @ Dm.T                           # [n_t, M_t] base Gram
            vt = vm[sel]
            for xi in xis:
                cols.append(dm ** 2 * np.einsum("nm,nxk->xmk", G ** xi, vt)
                            .reshape(n_rows, -1))       # [D, M_t·K]
    assert v is not None or not any(q is not None for q in (gap_q or ())), \
        "_phi_frame: the TurboGAP tails need v (the bond basis)"
    for q, d_q in zip(gap_q or (), (d_2b, d_3b)):
        if q is not None:
            # scales[0][1] is the first model's sparse list: exactly one entry
            # per species (_fps_per_species loops range(cfg.num_types)).
            for t in range(len(scales[0][1])):
                sel = ti == t
                cols.append(d_q ** 2 * np.einsum("ns,nxk->xsk", q[sel], v[sel])
                            .reshape(v.shape[1], -1))   # [D, W·K]
    return np.concatenate(cols, axis=1)


def _deltas(cfg: GAPconfig) -> tuple[float, float, float]:
    """(δ_soap, δ_2b, δ_3b) — the ONE place the amplitudes are read, so fit
    and predict cannot disagree about the column scaling."""
    return (float(cfg.gap_delta_soap), float(cfg.gap_delta_2b),
            float(cfg.gap_delta_3b))


def _gap_centres(cfg: GAPconfig):
    """The 2b / 3b sparse points as gap_fit sees them: (Q2 [M2], w2 [M2],
    Q3 [M3, 3], w3 [M3]). w is the sparse point's own cutoff weight — gap.f90's
    `cutoff(s)` factor (lines 278, 563): f(Q) for a 2b distance, f(r12)·f(r13)
    for a 3b triangle, with r12, r13 recovered from its (sum, squared
    difference) coordinates."""
    Q2 = np.asarray(cfg.gap2b_centres, dtype=np.float64)
    Q3 = np.asarray(cfg.gap3b_centres, dtype=np.float64)
    root = np.sqrt(np.maximum(Q3[:, 1], 0.0))
    w3 = (f_cut(0.5 * (Q3[:, 0] + root), cfg.gap3b_rcut, cfg.gap_buffer)
          * f_cut(0.5 * (Q3[:, 0] - root), cfg.gap3b_rcut, cfg.gap_buffer))
    return Q2, f_cut(Q2, cfg.gap2b_rcut, cfg.gap_buffer), Q3, w3


def _gap_q(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig):
    """Per-frame (q2 | None, q3 | None) float64 for _phi_frame per the
    cfg flags; None when both are off (shared by fit + predict). Each centre's
    column carries its cutoff weight (_gap_centres), as in gap.f90."""
    if not (cfg.gap_2b or cfg.gap_3b):
        return None
    raw = gap_raw_numpy(atoms, ti, cfg).astype(np.float64)
    sl = gap_slices(cfg)
    _, w2, _, w3 = _gap_centres(cfg)
    q2 = raw[:, sl["two_body"]]; q3 = raw[:, sl["three_body"]]
    return ((q2.reshape(len(raw), -1, len(w2)) * w2).reshape(q2.shape) if cfg.gap_2b else None,
            (q3.reshape(len(raw), -1, len(w3)) * w3).reshape(q3.shape) if cfg.gap_3b else None)


def _prior_factors(cfg: GAPconfig, scale_sparse: list, xis: tuple,
                   tail_vcols: int | None = None) -> list:
    """gap_fit's regulariser, one entry per column block in _phi_frame order:
    (n_columns, L) with K_MM + jitter·I = L·Lᵀ the sparse-sparse covariance of
    that GAP term (gp_predict.F90 gpSparse_fit penalises αᵀ·K_MM·α). A block
    is M sparse points × K readout columns, m-major, and the prior is
    independent across the K columns, so its penalty is kron(K_MM, I_K).

    scale_sparse = [(sparse_desc, delta, n_vcols)] per site model, matching
    _phi_frame's scales: one term per (model, species, ξ), K_MM =
    δ²·(D_m·D_m′)^ξ, n_vcols the model's readout width K. The TurboGAP tails
    (cfg.gap_2b / gap_3b) read out through the bond basis of width tail_vcols
    (required only then): one term per (species, neighbour species) for 2b
    and per (species, neighbour pair) for 3b, K_MM = δ²·w_s·w_s′·k(Q_s, Q_s′)
    with the descriptor's own kernel."""
    _, d_2b, d_3b = _deltas(cfg)
    T = cfg.num_types

    def chol(K):
        # gap_fit adds sparse_jitter and factorises (in quad precision). In
        # double a near-singular K_MM can still fail: raise the jitter tenfold
        # until it factorises and say so.
        jit = float(cfg.gap_sparse_jitter)
        while True:
            try:
                L = np.linalg.cholesky(K + jit * np.eye(len(K)))
                if jit > float(cfg.gap_sparse_jitter):
                    warnings.warn(f"K_MM not positive definite at gap_sparse_jitter="
                                  f"{cfg.gap_sparse_jitter:g}; factorised with {jit:g}")
                return L
            except np.linalg.LinAlgError:
                jit *= 10.0

    out = []
    for sparse_desc, dm, k in scale_sparse:
        for Dm in sparse_desc:
            base = Dm @ Dm.T
            for xi in xis:
                out.append((len(Dm) * k, chol(dm ** 2 * base ** xi)))
    if cfg.gap_2b or cfg.gap_3b:
        assert tail_vcols is not None, "tail_vcols required with gap_2b/gap_3b"
        n_vcols = tail_vcols
        Q2, w2, Q3, w3 = _gap_centres(cfg)
        if cfg.gap_2b:
            K2 = d_2b ** 2 * np.outer(w2, w2) * np.exp(
                -0.5 * ((Q2[:, None] - Q2[None]) / cfg.gap2b_sigma) ** 2)
            L2 = chol(K2)
            out += [(len(Q2) * n_vcols, L2)] * (T * T)            # species × neighbour species
        if cfg.gap_3b:
            r = np.sqrt(sum(((Q3[:, None, d] - Q3[None, :, d]) / cfg.gap3b_sigma[d]) ** 2
                            for d in range(3)))
            k3 = (np.where(r < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)
                  if str(getattr(cfg, "gap3b_kernel", "exp")) == "pol" else np.exp(-0.5 * r ** 2))
            L3 = chol(d_3b ** 2 * np.outer(w3, w3) * k3)
            out += [(len(Q3) * n_vcols, L3)] * (T * (T * (T + 1) // 2))   # species × neighbour pair
    return out


def _whiten(X: np.ndarray, factors: list, solve_with_transpose: bool) -> np.ndarray:
    """Apply each block's prior factor along its sparse-point axis.
    X [R, n_columns]. solve_with_transpose=False: Φ → Φ·(L⁻ᵀ ⊗ I) (design in
    whitened coefficients β = (Lᵀ ⊗ I)·α). True: β → α = (L⁻ᵀ ⊗ I)·β.
    Each block's readout width is n_columns / M, so blocks may differ."""
    out, off = np.empty_like(X), 0
    for n, L in factors:
        M = len(L)
        n_vcols = n // M
        blk = X[:, off:off + n].reshape(len(X), M, n_vcols)
        if solve_with_transpose:                               # α[m, k] = (L⁻ᵀ β)[m, k]
            res = np.linalg.solve(L.T, blk[0])[None]
        else:                                                  # Φ′ₖ = Φₖ·L⁻ᵀ  ⇔  Φ′ₖᵀ = L⁻¹·Φₖᵀ
            res = np.linalg.solve(L, blk.transpose(1, 0, 2).reshape(M, -1)).reshape(
                M, len(X), n_vcols).transpose(1, 0, 2)
        out[:, off:off + n] = res.reshape(len(res), n)
        off += n
    assert off == X.shape[1]
    return out


def _targets(frames: list[Atoms], cfg: GAPconfig) -> np.ndarray:
    """Targets [F, D] (dipole [F, 3] unit-converted per cfg, matching
    assemble; polarisability [F, 6] Voigt as stored).

    NOTE: unlike the training path (assemble_data_dict), this applies NO
    box/label guards — a pathological test file scores silently."""
    key = tnep_data._resolve_target_key(cfg)
    mu = np.stack([np.asarray(tnep_data._extract_target(s, key))
                   for s in frames]).astype(np.float64)
    if cfg.target_mode == 1 and getattr(cfg, "convert_dipole_to_eangstrom", True):
        mu = mu * tnep_data._dipole_conversion_factor(cfg.dipole_units)
    return mu


def _rmse_r2(pred: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """Total-vector RMSE √(mean(per-component²)·D) (diag_zeta convention;
    D = 3 dipole, 6 Voigt polarisability), R²."""
    res = pred - mu
    rmse = float(np.sqrt(np.mean(res ** 2) * mu.shape[1]))
    tot = float(np.sum((mu - mu.mean(0)) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / tot if tot > 0.0 else float("nan")
    return rmse, r2


def fit_gap_dipole(cfg: GAPconfig) -> GapDipoleModel:
    """Fit the sparse-kernel dipole model on cfg.data_path; evaluate
    cfg.test_data_path when set; write a run record JSON to GAP/.

    MUTATES cfg in place: `seed` (when None) and `gap2b_centres`/
    `gap3b_centres` (when auto-selected) are filled in, because they must
    travel with the model for `predict` and into the run record.

    Solver memory: the dual ridge solves the [DF, DF] Gram, so Φ itself
    dominates (~1.7 GB at F=1500, 48k columns) and memory is linear in
    columns. Multi-scale (gap_extra_rcut) ≈ 2× descriptor build time, fit
    and predict."""
    _validate(cfg)
    t0 = time.time()

    # As MasterTNEP (~418): resolve the seed BEFORE any RNG is constructed —
    # the subsample draw, FPS and centre selection all consume it, so a seed
    # generated after cfg.randomise would not replay the recorded config.
    if cfg.seed is None:
        cfg.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])

    dataset, types_int = tnep_data.collect(cfg)
    if cfg.total_N is not None:
        cfg.randomise(dataset)
        dataset = [dataset[k] for k in cfg.indices]
        types_int = [types_int[k] for k in cfg.indices]
    tis = [np.asarray(t) for t in types_int]
    # Mutating cfg (here and the seed above) is deliberate: the centres are
    # part of the descriptor definition, so they must travel with the model
    # (model.cfg → predict) and into the run record's "config" block.
    if (cfg.gap_2b or cfg.gap_3b) and (cfg.gap2b_centres is None
                                       or cfg.gap3b_centres is None):
        c2, c3 = select_centres(dataset, cfg)
        cfg.gap2b_centres, cfg.gap3b_centres = c2.tolist(), c3.tolist()

    builder = make_descriptor_builder(cfg)
    descs = _descriptors(builder, dataset)
    # assemble_data_dict: unit-converted targets, box/label guards, V_atom —
    # all built purely from cfg pre-model. Gradients not needed (empty).
    d = tnep_data.assemble_data_dict(
        dataset, types_int, descs, [[]] * len(dataset), [[]] * len(dataset),
        cfg)
    mu = np.stack([np.asarray(t, dtype=np.float64) for t in d["targets"]])
    pair = gm.pair_columns(cfg)
    vs = [_finish_v(V, a, t, cfg) for V, a, t in zip(d["V_atom"], dataset, tis)]
    # Kernel channels (2·C under target_mode=2: isotropic + traceless groups).
    n_vcols = _n_vcols(cfg)

    # Per-species FPS sparse points, per site model in cfg order, then the
    # extra SOAP scale (selected on ITS OWN descriptors — the wider builder)
    # — all from one rng stream, so gap_site_models=["soap"] without
    # gap_extra_rcut is bit-identical to the pre-feature code.
    rng = np.random.default_rng(cfg.seed)
    M_req = int(cfg.gap_sparse_per_species)
    xis = _xi_tuple(cfg)

    def _fps_per_species(scale_descs):
        out = []
        for t in range(cfg.num_types):
            Dt = np.concatenate([D[ti == t]
                                 for D, ti in zip(scale_descs, tis)])
            if not len(Dt):
                raise ValueError(
                    f"no atoms of type Z={cfg.types[t]} in the training "
                    f"data — cannot place sparse points for that species")
            out.append(Dt[_fps(Dt, M_req, rng)].copy())
        return out

    models = list(cfg.gap_site_models)
    site = _site_terms(cfg, builder, dataset, tis, vs, descs)
    site_sparse = {m: _fps_per_species(d) for m, (d, _, _) in site.items()}
    builder2 = descs2 = sparse_desc2 = None
    if cfg.gap_extra_rcut is not None:
        # Second SOAP length scale for the KERNEL only (v is untouched):
        # rcut_hard = gap_extra_rcut, same buffer width rcut_hard − rcut_soft.
        cfg2 = copy.copy(cfg)
        cfg2.rcut_hard = float(cfg.gap_extra_rcut)
        cfg2.rcut_soft = cfg2.rcut_hard - (cfg.rcut_hard - cfg.rcut_soft)
        builder2 = make_descriptor_builder(cfg2)
        descs2 = _descriptors(builder2, dataset)
        sparse_desc2 = _fps_per_species(descs2)

    deltas = _deltas(cfg)
    d_soap = deltas[0]
    # (sparse set, δ, readout width K_model) per site model, then the extra scale
    scale_sparse = [(site_sparse[m], dm, g[0].shape[2]) for m, (_, g, dm) in site.items()]
    if descs2 is not None:
        scale_sparse.append((sparse_desc2, d_soap, n_vcols))
    sizes = [[len(s) for s in group] for group, _, _ in scale_sparse]
    # α block per (model/scale, species): M × len(ξ list) × K columns; then the
    # additive GAP blocks, one per species: W2·K (gap_2b) and W3·K (gap_3b).
    sl = gap_slices(cfg)
    T = cfg.num_types
    w_gap = {k: (sl[k].stop - sl[k].start) * n_vcols if on else 0
             for k, on in (("two_body", cfg.gap_2b),
                           ("three_body", cfg.gap_3b))}
    blocks = ([len(Dm) * len(xis) * k for group, _, k in scale_sparse for Dm in group]
              + [w for w in w_gap.values() if w for _ in range(T)])
    n_columns = sum(blocks)

    # Φ frame-by-frame — the full N_envs × M kernel never exists.
    F = len(dataset)
    D = mu.shape[1]                      # 3 (dipole) or 6 (polarisability)
    Phi = np.empty((F, D, n_columns))
    for f in range(F):
        scales = _frame_scales(site, site_sparse, f, descs2, sparse_desc2, vs[f], d_soap)
        Phi[f] = _phi_frame(scales, tis[f], xis,
                            _gap_q(dataset[f], tis[f], cfg), deltas[1:], v=vs[f])
    Phi_flat = Phi.reshape(D * F, n_columns)

    # An unpopulated species-pair slot (or a centre no triplet reaches) gives
    # identically-zero gap columns: those coefficients are unidentifiable.
    n_gap = sum(w * T for w in w_gap.values())
    if n_gap:
        dead = int(np.count_nonzero(~Phi_flat[:, -n_gap:].any(axis=0)))
        if dead:
            warnings.warn(
                f"{dead}/{n_gap} GAP 2b/3b columns are identically zero "
                f"(unpopulated species-pair slots or unreached centres): "
                f"those coefficients are unidentifiable — the ridge leaves "
                f"them at ~0 and predictions are unaffected")

    # gap_fit's estimator (gp_predict.F90 gpSparse_fit): sparse GP regression,
    #   minimise  Σ_obs ((Φα − μ)/σ_obs)²  +  αᵀ·K_MM·α,
    # σ_obs the absolute error on a total dipole component. gap_fit's default
    # is per-atom regularisation, σ_obs = σ·√N_atoms (gap_fit_module.F90:1149).
    # gap_fit solves the augmented system [Λ^-½·K_NM; chol(K_MM)] by QR. The
    # readout multiplies the columns by K per sparse point, so here the SAME
    # minimiser is taken in whitened coefficients β = (Lᵀ ⊗ I)·α, where the
    # penalty is ‖β‖², and solved in the dual: the Gram is [DF, DF]
    # regardless of n_columns.
    sigma = float(cfg.gap_sigma)
    row_w = np.repeat(1.0 / (sigma * np.sqrt([len(a) for a in dataset])), D)
    if D == 6 and float(cfg.gap_lambda_shear) != 1.0:
        # GPUMD lambda_shear: off-diagonal rows fitted with σ/λ_shear.
        row_w = row_w * np.tile([1.0] * 3 + [float(cfg.gap_lambda_shear)] * 3, F)
    factors = _prior_factors(cfg, scale_sparse, xis, n_vcols)
    Phi_w = _whiten(Phi_flat, factors, False) * row_w[:, None]
    G = Phi_w @ Phi_w.T                                  # [DF, DF]
    G[np.diag_indices_from(G)] += 1.0
    beta = Phi_w.T @ np.linalg.solve(G, mu.reshape(-1) * row_w)
    alpha_flat = _whiten(beta[None], factors, True)[0]
    parts = np.split(alpha_flat, np.cumsum(blocks)[:-1])
    # every α group is one block per species
    groups = [parts[i:i + T] for i in range(0, len(parts), T)]
    site_alpha = dict(zip(models, groups[:len(models)]))
    del groups[:len(models)]
    alpha2 = groups.pop(0) if descs2 is not None else None
    alpha_gap2b = groups.pop(0) if cfg.gap_2b else None
    alpha_gap3b = groups.pop(0) if cfg.gap_3b else None
    assert not groups

    model = GapDipoleModel(cfg=cfg, builder=builder, site_sparse=site_sparse,
                           site_alpha=site_alpha, xi=xis, builder2=builder2,
                           sparse_desc2=sparse_desc2, alpha2=alpha2,
                           alpha_gap2b=alpha_gap2b, alpha_gap3b=alpha_gap3b)

    train_rmse, train_r2 = _rmse_r2(
        (Phi_flat @ alpha_flat).reshape(F, D), mu)
    metrics = {"train_rmse": train_rmse, "train_r2": train_r2,
               "n_train_frames": F, "fit_seconds": time.time() - t0}
    print(f"[gap_dipole] fit models={models} M={sizes} xi={list(xis)} sigma={sigma:g} "
          f"n_columns={n_columns}: "
          f"train RMSE {train_rmse:.4f}, R² {train_r2:.4f} "
          f"({metrics['fit_seconds']:.1f} s)")

    if cfg.test_data_path is not None:
        test_frames = read(cfg.test_data_path, index=":")
        test_rmse, test_r2 = _rmse_r2(predict(model, test_frames),
                                      _targets(test_frames, cfg))
        metrics.update(test_rmse=test_rmse, test_r2=test_r2,
                       n_test_frames=len(test_frames))
        print(f"[gap_dipole] test RMSE {test_rmse:.4f}, R² {test_r2:.4f}")

    from model_io import _serialize_config
    record = {
        "config": _serialize_config(cfg),
        "gap": {"sparse_per_species": M_req, "kernel_power": list(xis),
                "sigma": sigma, "sparse_jitter": float(cfg.gap_sparse_jitter),
                # summed site models (gap_models.py): kernel per body order,
                # each with its own FPS set, prior and relative amplitude
                "site_models": models,
                "model_delta": {m: float(cfg.gap_model_delta[m]) for m in models},
                "site_channels": int(cfg.gap_site_channels),
                "site_lmax": int(cfg.gap_site_lmax),
                "site_rcut": cfg.gap_site_rcut,
                "cross_lmax": int(cfg.gap_cross_lmax),
                "delta_soap": float(cfg.gap_delta_soap),
                "delta_2b": float(cfg.gap_delta_2b),
                "delta_3b": float(cfg.gap_delta_3b),
                # zeta only where it is live — config.dipole_zeta is present
                # in every record but decorative for non-screened radials.
                **({"zeta": float(cfg.dipole_zeta)}
                   if cfg.dipole_radial == "screened" else {}),
                "radial": cfg.dipole_radial,
                "cutoff": resolved_dipole_cutoff(cfg),
                "channels": int(cfg.dipole_channels),
                "kernel_channels": kernel_channels(cfg),
                # True ⇒ per-(sparse-point, channel, neighbor-species)
                # columns; subsumes symmetric c AND ordered Ω (no distinction
                # in the absorbed linear form).
                "pair_columns": pair,
                "cross_channels": int(cfg.gap_cross_channels),
                **({"group_delta": [float(x) for x in cfg.gap_group_delta],
                    "lambda_shear": float(cfg.gap_lambda_shear)}
                   if int(cfg.target_mode) == 2 else {}),
                "extra_rcut": cfg.gap_extra_rcut,
                "n_columns": n_columns,
                # additive TurboGAP 2b/3b blocks (gap_descriptor.py)
                "gap_2b": bool(cfg.gap_2b), "gap_3b": bool(cfg.gap_3b),
                "n_gap_columns": {k: w * T for k, w in w_gap.items()},
                **({k: getattr(cfg, k) for k in (
                    "gap2b_centres", "gap3b_centres", "gap2b_sigma",
                    "gap3b_sigma", "gap2b_rcut", "gap3b_rcut", "gap_buffer")}
                   if cfg.gap_2b or cfg.gap_3b else {}),
                # sparse-set sizes per site model
                "M_per_species": {m: {str(z): n for z, n in zip(cfg.types, sz)}
                                  for m, sz in zip(models, sizes)},
                # scale-2 sparse-set sizes, only when gap_extra_rcut is set
                **({"M_per_species_2": {str(z): n
                                        for z, n in zip(cfg.types, sizes[-1])}}
                   if descs2 is not None else {})},
        "metrics": metrics,
    }
    os.makedirs("GAP", exist_ok=True)
    xtag = "-".join(map(str, xis))
    # Uniquify: same-second fits (λ/ξ scan loops) must not clobber records.
    stem = os.path.join(
        "GAP", f"gap_M{M_req}_x{xtag}_{time.strftime('%Y%m%d_%H%M%S')}")
    path, k = stem + ".json", 1
    while os.path.exists(path):
        path, k = f"{stem}_{k}.json", k + 1
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"[gap_dipole] run record → {path}")
    return model


def predict(model: GapDipoleModel, frames: list[Atoms]) -> np.ndarray:
    """μ [F, 3] (or α [F, 6], target_mode=2) for a list of frames via the
    model's held cfg/builder.

    Routes through _phi_frame — the SAME design-matrix contraction the fit
    uses (μ = Φα) — so fit and predict cannot drift."""
    cfg = model.cfg
    models = list(cfg.gap_site_models)
    tis = [np.asarray(t) for t in tnep_data.assign_type_indices(frames, cfg.types)]
    vs = [_v_atoms(a, ti, cfg) for a, ti in zip(frames, tis)]
    site = _site_terms(cfg, model.builder, frames, tis, vs)
    descs2 = (_descriptors(model.builder2, frames)
              if model.builder2 is not None else None)
    alpha_flat = np.concatenate(
        [a for m in models for a in model.site_alpha[m]] + list(model.alpha2 or [])
        + list(model.alpha_gap2b or []) + list(model.alpha_gap3b or []))
    deltas = _deltas(cfg)
    d_soap = deltas[0]
    out = None
    for f, (atoms, ti, v) in enumerate(zip(frames, tis, vs)):
        scales = _frame_scales(site, model.site_sparse, f, descs2, model.sparse_desc2, v, d_soap)
        row = _phi_frame(scales, ti, model.xi,
                         _gap_q(atoms, ti, cfg), deltas[1:], v=v) @ alpha_flat
        if out is None:
            out = np.empty((len(frames), len(row)))     # D = 3 or 6
        out[f] = row
    return out


def predict_timed(model: GapDipoleModel, frames: list[Atoms]) -> tuple[np.ndarray, float]:
    """predict() plus wall time per atom (µs/atom) of the full forward pass:
    neighbour lists, descriptors, geometry and kernel columns."""
    t0 = time.perf_counter(); P = predict(model, frames)
    return P, 1e6 * (time.perf_counter() - t0) / sum(len(a) for a in frames)


if __name__ == "__main__":
    # IDE / CLI entry: fit on cfg.data_path, evaluate cfg.test_data_path,
    # print metrics, write the GAP/ run record. The GAP-relevant readout
    # fields are pinned HERE rather than read from GAPconfig class defaults,
    # so a working-tree default edit cannot leak into the fit.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # datasets/, GAP/

    cfg = GAPconfig()

    # GAP knobs — tweak freely:
    cfg.gap_sparse_per_species = 40
    cfg.gap_kernel_power = 2
    cfg.gap_sigma = 2e-2

    fit_gap_dipole(cfg)
