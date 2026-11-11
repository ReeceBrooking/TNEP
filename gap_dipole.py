"""GAP-mode dipole / polarisability fit for the neighbor readout.

Sparse-GP alternative to the SNES-trained ANN, exploiting the readout's
linearity in the site scalars. Estimator = gap_fit's (libAtoms/GAP
gp_predict.F90 gpSparse_fit): observation error gap_sigma·√N_atoms, penalty
αᵀ·K_MM·α per term, δ² on every kernel:

    μ = Σ_i Σ_k s_i^k v_i[:, k],  s_i^k = Σ_m α_m^k (D̂_i · D̂_m)^ξ

v_i from data._neighbor_dipole_V [N, D, C, T] (any radial/cutoff; screened
needs C = 1). Polarisability (target_mode=2) is the same fit with D = 6 Voigt
rows and 2·C kernel channels (isotropic |d|²I, traceless d⊗d − |d|²I/3).

Design columns per sparse point: dipole_species_pairs="off" → k = n (V summed
over t′, K = C); anything else → k = (n, t′), K = C·T, which subsumes both the
symmetric c and ordered Ω readout tables. Extensions: gap_extra_rcut (second
SOAP scale for the KERNEL only, own FPS sets, scales never cross, v unchanged),
gap_cross_channels (two-neighbour cross group, _finish_v), gap_site_models
(2b/3b body-order kernels, gap_models.py, each with its own FPS set and prior).

Columns may exceed the D·F equations; the prior handles it (train RMSE → 0,
test RMSE is the metric). Takes GAPconfig only (ANN modes are frozen there).
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
from gap_config import GAPconfig
from TNEPconfig import resolved_dipole_cutoff, resolved_dipole_rc, bond_rank, kernel_channels


@dataclass
class GapDipoleModel:
    """In-memory GAP model — never persisted (refits in seconds).

    Column/α block order (shared by fit + predict via _phi_frame): site model
    (cfg.gap_site_models order) → species → (m, k), each block [M_t·K_model]
    flat (K_model = bond-basis width for soap/2b, angular-kernel width for 3b);
    the scale-2 blocks (gap_extra_rcut) come last. `sparse_desc`/`alpha` are
    the "soap" entries."""
    cfg: GAPconfig
    builder: object                   # descriptor builder from the fitting cfg
    site_sparse: dict                 # model → per type t: [M_t, Q] unit-norm rows
    site_alpha: dict                  # model → per type t: [M_t·K_model] flat
    xi: int                           # kernel power (D̂_i·D̂_m)^ξ
    builder2: object = None           # second-scale builder (gap_extra_rcut)
    sparse_desc2: list[np.ndarray] | None = None
    alpha2: list[np.ndarray] | None = None
    train_pred: np.ndarray | None = None   # Φα on the training frames, [F, D] (fit order)

    @property
    def sparse_desc(self) -> list[np.ndarray]:
        return self.site_sparse["soap"]

    @property
    def alpha(self) -> list[np.ndarray]:
        return self.site_alpha["soap"]


def _sparse_per_model(cfg: GAPconfig) -> dict:
    """model → FPS sparse points per species: an int for every site model, or a
    dict keyed by site model (missing models take the "soap" entry)."""
    M = cfg.gap_sparse_per_species
    if isinstance(M, dict):
        return {m: int(M.get(m, M.get("soap"))) for m in cfg.gap_site_models}
    return {m: int(M) for m in cfg.gap_site_models}


def _validate(cfg: GAPconfig) -> None:
    """Raise unless cfg is a GAP-fittable configuration. Runs FIRST in fit."""
    xi = cfg.gap_kernel_power
    if isinstance(xi, bool) or not isinstance(xi, int) or xi < 1:
        raise ValueError(f"gap_kernel_power must be an int >= 1, got {xi!r}")
    M = cfg.gap_sparse_per_species
    ok = (isinstance(M, dict) and ("soap" in M or set(cfg.gap_site_models) <= set(M))
          and all(int(v) >= 1 for v in M.values())) or (not isinstance(M, dict) and int(M) >= 1)
    if not ok:
        raise ValueError(f"gap_sparse_per_species must be an int >= 1 or a dict keyed by site model "
                         f"(covering gap_site_models or carrying 'soap'), got {M!r}")
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
        (float(cfg.gap_lambda_shear) > 0, "gap_lambda_shear must be > 0"),
        # frozen on GAPconfig, so this catches INSTANCE mutation that would make
        # assemble_data_dict build NEP inputs the fit never reads.
        (str(cfg.nep_descriptor) == "off",
         "GAP mode needs nep_descriptor='off' (NEP inputs are not part of "
         "the kernel fit)"),
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
        ("3b" not in cfg.gap_site_models or int(cfg.gap_cross_channels) >= 1,
         "the 3b site model reads out through the angular kernel: gap_cross_channels >= 1 required"),
        ("3b" not in cfg.gap_site_models or int(cfg.target_mode) == 2 or int(cfg.gap_cross_lmax) >= 1,
         "3b dipole kernel needs gap_cross_lmax >= 1 (l = 0 collapses to the one-hop basis)"),
        (set(cfg.gap_site_models) <= {"soap"} or cfg.dipole_radial != "screened",
         "2b/3b site models need a multi-channel radial (screened is single-channel)"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise ValueError("fit_gap_dipole config violation: " + "; ".join(bad))


def _descriptors(builder, frames: list[Atoms]) -> list[np.ndarray]:
    """SOAP descriptors, no gradients, float64. Asserts ‖D‖ ≈ 1 (both
    builders already unit-normalize; we must NOT renormalize)."""
    if type(builder).__name__ == "DescriptorBuilderGPUTF":
        # flat output with the TF NL: build_descriptors' per-atom packing and
        # brute-force NumPy NL cost ~300 ms/frame for data we never read.
        desc = [f[0] for f in builder.build_descriptors_flat(frames, calc_gradients=False, use_tf_nl=True)]
    else:
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
    """[N, D, C, T] → v_i [N, D, K] — THE one place the column contract is encoded.
    K = C·T with pair columns (k = (n, t′), C-major reshape order) else K = C
    (t′ summed). D = 3 (dipole) or 6 (polarisability, Voigt)."""
    V = np.asarray(V, dtype=np.float64)
    if pair_columns:
        return V.reshape(V.shape[0], V.shape[1], -1)
    return V.sum(axis=-1)


def _cross_vectors(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Dipole kernel vectors of the cross group v_i^a = Σ_j R_a(r_ij) d_ij over
    the first C_x radial channels, [N, 3, K₁] (K₁ = C_x·T′ under pair columns)."""
    V1 = tnep_data._neighbor_dipole_V(
        atoms, ti, resolved_dipole_rc(cfg),
        int(cfg.gap_cross_channels), int(cfg.num_types),
        str(cfg.dipole_radial), float(cfg.dipole_zeta),
        resolved_dipole_cutoff(cfg), float(cfg.dipole_gauss_sigma_scale),
        rank=1, unit=bool(getattr(cfg, "dipole_unit_vector", False)))   # [N, 3, C_x, T]
    return _collapse_V(V1, gm.pair_columns(cfg))


def _cross_group(v: np.ndarray) -> np.ndarray:
    """Third polarisability kernel group [N, 6, K_x]: traceless sym(v^a ⊗ v^b),
    a ≤ b, over v = _cross_vectors(...) [N, 3, K₁]; K_x = K₁(K₁+1)/2. Carries the
    two-neighbour d_j⊗d_k tensors the bond-axis groups lack; even under
    inversion, linear in the site coefficient."""
    a_idx, b_idx = np.triu_indices(v.shape[2])
    a, b = v[:, :, a_idx], v[:, :, b_idx]                       # [N, 3, K_x]
    sym = 0.5 * (a[:, :, None, :] * b[:, None, :, :]
                 + b[:, :, None, :] * a[:, None, :, :])         # [N, 3, 3, K_x]
    tr3 = (sym[:, 0, 0] + sym[:, 1, 1] + sym[:, 2, 2]) / 3.0
    return np.stack([sym[:, 0, 0] - tr3, sym[:, 1, 1] - tr3, sym[:, 2, 2] - tr3,
                     sym[:, 0, 1], sym[:, 1, 2], sym[:, 2, 0]], axis=1)


def _cross_group_dipole(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Dipole cross group [N, 3, K_x]: w^{ab} = T^a·v^b with T^a = Σ_j R_a d_ij d_ijᵀ,
    v^b = Σ_j R_b d_ij (K₁ = C_x·T′ each, K_x = K₁²), i.e. Σ_jk R_a R_b (d_ij·d_ik) d_ij —
    the bond-angle charge flux the one-hop difference form cannot carry. Odd
    under inversion, linear in the site coefficient."""
    args = (atoms, ti, resolved_dipole_rc(cfg), int(cfg.gap_cross_channels),
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
    """Staged kernel [N, D, Ck, T] → design geometry v_i [N, D, K]: pair columns +
    cross group. THE one path fit and predict share, so the contract cannot drift."""
    v = _collapse_V(V, gm.pair_columns(cfg))
    if int(cfg.gap_cross_channels) > 0:
        pol = int(cfg.target_mode) == 2
        cross = _cross_group(_cross_vectors(atoms, ti, cfg)) if pol else _cross_group_dipole(atoms, ti, cfg)
        v = np.concatenate([v, cross], axis=2)
    return v


def _n_vcols(cfg: GAPconfig) -> int:
    """K: design columns per sparse point = kernel channels × pair columns + cross
    group (K₁(K₁+1)/2 under target_mode=2, K₁² under target_mode=1; K₁ = C_x·T′)."""
    T_ = cfg.num_types if gm.pair_columns(cfg) else 1
    k = kernel_channels(cfg) * T_
    cx = int(cfg.gap_cross_channels)
    if cx > 0:
        k1 = cx * T_
        k += k1 * (k1 + 1) // 2 if int(cfg.target_mode) == 2 else k1 * k1
    return k


def _v_atoms(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Per-atom geometry vectors v_i [N, D, K] from cfg (radial, cutoff, channels,
    pair columns, bond rank, field channel)."""
    V = tnep_data._neighbor_dipole_V(
        atoms, ti, resolved_dipole_rc(cfg),
        int(cfg.dipole_channels), int(cfg.num_types),
        str(cfg.dipole_radial), float(cfg.dipole_zeta),
        resolved_dipole_cutoff(cfg), float(cfg.dipole_gauss_sigma_scale),
        rank=bond_rank(cfg), unit=bool(getattr(cfg, "dipole_unit_vector", False)))
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
    d_soap = float(cfg.gap_delta_soap)
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


def _phi_frame(scales: list, ti: np.ndarray, xi: int) -> np.ndarray:
    """[D, n_columns] design block for one frame — THE column contract for fit AND
    predict. scales = [(D, sparse_desc, v, delta)] per site model (then the extra
    SOAP scale); each model's kernel couples its own descriptors to its own sparse
    points, reads out through its v [N, D, K] and carries δ² on its columns
    (gap.f90 `delta**2 * alphas * kernel`). Block order: model → species → (m, k)."""
    cols = []
    for D, sparse_desc, vm, dm in scales:
        n_rows = vm.shape[1]             # 3 (dipole) or 6 (polarisability)
        for t, Dm in enumerate(sparse_desc):
            sel = ti == t
            G = D[sel] @ Dm.T                           # [n_t, M_t] Gram
            vt = vm[sel]
            vflat = vt.reshape(len(vt), -1)             # [n_t, D·K]
            # Σ_n G^ξ[n,m]·v[n,x,k] as ONE GEMM: [M_t, D·K] → [D, M_t·K]
            cols.append(dm ** 2 * ((G ** xi).T @ vflat).reshape(-1, n_rows, vt.shape[2])
                        .transpose(1, 0, 2).reshape(n_rows, -1))           # [D, M_t·K]
    return np.concatenate(cols, axis=1)


def _prior_factors(cfg: GAPconfig, scale_sparse: list, xi: int) -> list:
    """gap_fit's regulariser per column block, in _phi_frame order: (n_columns, L)
    with L·Lᵀ = K_MM + jitter·I, K_MM = δ²·(D_m·D_m′)^ξ per (model, species). The
    prior is independent across the K readout columns, so the penalty is kron(K_MM, I_K)."""
    jit = float(cfg.gap_sparse_jitter)
    return [(len(Dm) * k, np.linalg.cholesky(dm ** 2 * (Dm @ Dm.T) ** xi + jit * np.eye(len(Dm))))
            for sparse_desc, dm, k in scale_sparse for Dm in sparse_desc]


def _whiten(X: np.ndarray, factors: list, solve_with_transpose: bool) -> np.ndarray:
    """Apply each block's prior factor along its sparse-point axis (X [R, n_columns]).
    False: Φ → Φ·(L⁻ᵀ ⊗ I) (whitened coefficients β = (Lᵀ ⊗ I)·α); True: β → α."""
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
    """Targets [F, D] (dipole unit-converted per cfg; polarisability Voigt as stored).
    Unlike assemble_data_dict this applies NO box/label guards."""
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


def _resolve_seed(cfg: GAPconfig) -> None:
    # Resolve the seed BEFORE any RNG exists (subsample, FPS and centre selection
    # all consume it) so the run record replays the fit.
    if cfg.seed is None:
        cfg.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])


def fit_gap_dipole(cfg: GAPconfig) -> GapDipoleModel:
    """Fit on cfg.data_path, evaluate cfg.test_data_path, write a run record JSON
    to GAP/. MUTATES cfg: fills `seed` when None. Memory is linear in columns
    (Φ ~1.7 GB at F=1500, 48k columns); gap_extra_rcut ≈ 2× build/fit/predict."""
    _validate(cfg)
    t0 = time.time()
    _resolve_seed(cfg)

    dataset, types_int = tnep_data.collect(cfg)
    if cfg.total_N is not None:
        cfg.randomise(dataset)
        dataset = [dataset[k] for k in cfg.indices]
        types_int = [types_int[k] for k in cfg.indices]
    tis = [np.asarray(t) for t in types_int]
    builder = make_descriptor_builder(cfg)
    descs = _descriptors(builder, dataset)
    # unit-converted targets, box/label guards, V_atom — all from cfg pre-model; no gradients
    d = tnep_data.assemble_data_dict(
        dataset, types_int, descs, [[]] * len(dataset), [[]] * len(dataset),
        cfg)
    mu = np.stack([np.asarray(t, dtype=np.float64) for t in d["targets"]])
    pair = gm.pair_columns(cfg)
    vs = [_finish_v(V, a, t, cfg) for V, a, t in zip(d["V_atom"], dataset, tis)]
    # Kernel channels (2·C under target_mode=2: isotropic + traceless groups).
    n_vcols = _n_vcols(cfg)

    # FPS per species, per site model in cfg order, then the extra scale (on ITS
    # OWN descriptors) — one rng stream.
    rng = np.random.default_rng(cfg.seed)
    M_req = _sparse_per_model(cfg)
    xi = int(cfg.gap_kernel_power)

    def _fps_per_species(scale_descs, M):
        out = []
        for t in range(cfg.num_types):
            Dt = np.concatenate([D[ti == t]
                                 for D, ti in zip(scale_descs, tis)])
            if not len(Dt):
                raise ValueError(
                    f"no atoms of type Z={cfg.types[t]} in the training "
                    f"data — cannot place sparse points for that species")
            out.append(Dt[_fps(Dt, M, rng)].copy())
        return out

    models = list(cfg.gap_site_models)
    site = _site_terms(cfg, builder, dataset, tis, vs, descs)
    site_sparse = {m: _fps_per_species(d, M_req[m]) for m, (d, _, _) in site.items()}
    builder2 = descs2 = sparse_desc2 = None
    if cfg.gap_extra_rcut is not None:
        # second SOAP scale for the KERNEL only (v untouched), same soft/hard buffer
        cfg2 = copy.copy(cfg)
        cfg2.rcut_hard = float(cfg.gap_extra_rcut)
        cfg2.rcut_soft = cfg2.rcut_hard - (cfg.rcut_hard - cfg.rcut_soft)
        builder2 = make_descriptor_builder(cfg2)
        descs2 = _descriptors(builder2, dataset)
        sparse_desc2 = _fps_per_species(descs2, M_req["soap"])

    d_soap = float(cfg.gap_delta_soap)
    # (sparse set, δ, readout width K_model) per site model, then the extra scale
    scale_sparse = [(site_sparse[m], dm, g[0].shape[2]) for m, (_, g, dm) in site.items()]
    if descs2 is not None:
        scale_sparse.append((sparse_desc2, d_soap, n_vcols))
    sizes = [[len(s) for s in group] for group, _, _ in scale_sparse]
    # α block per (model/scale, species): M × K columns.
    T = cfg.num_types
    blocks = [len(Dm) * k for group, _, k in scale_sparse for Dm in group]
    n_columns = sum(blocks)

    # Φ frame-by-frame — the full N_envs × M kernel never exists.
    F = len(dataset)
    D = mu.shape[1]                      # 3 (dipole) or 6 (polarisability)
    Phi = np.empty((F, D, n_columns))
    for f in range(F):
        scales = _frame_scales(site, site_sparse, f, descs2, sparse_desc2, vs[f], d_soap)
        Phi[f] = _phi_frame(scales, tis[f], xi)
    Phi_flat = Phi.reshape(D * F, n_columns)

    # gap_fit's estimator: minimise Σ_obs ((Φα − μ)/σ_obs)² + αᵀ·K_MM·α, σ_obs =
    # σ·√N_atoms (gap_fit_module.F90:1149). gap_fit QR-solves the augmented system;
    # here the same minimiser is taken in whitened β = (Lᵀ ⊗ I)·α (penalty ‖β‖²)
    # and solved in the dual: the Gram is [DF, DF] regardless of n_columns.
    sigma = float(cfg.gap_sigma)
    row_w = np.repeat(1.0 / (sigma * np.sqrt([len(a) for a in dataset])), D)
    if D == 6 and float(cfg.gap_lambda_shear) != 1.0:
        # GPUMD lambda_shear: off-diagonal rows fitted with σ/λ_shear.
        row_w = row_w * np.tile([1.0] * 3 + [float(cfg.gap_lambda_shear)] * 3, F)
    factors = _prior_factors(cfg, scale_sparse, xi)
    Phi_w = _whiten(Phi_flat, factors, False) * row_w[:, None]
    mu_w = mu.reshape(-1) * row_w
    G = Phi_w @ Phi_w.T                                  # [DF, DF]
    G[np.diag_indices_from(G)] += 1.0
    beta = Phi_w.T @ np.linalg.solve(G, mu_w)
    alpha_flat = _whiten(beta[None], factors, True)[0]
    parts = np.split(alpha_flat, np.cumsum(blocks)[:-1])
    # every α group is one block per species
    groups = [parts[i:i + T] for i in range(0, len(parts), T)]
    site_alpha = dict(zip(models, groups[:len(models)]))
    del groups[:len(models)]
    alpha2 = groups.pop(0) if descs2 is not None else None
    assert not groups

    model = GapDipoleModel(cfg=cfg, builder=builder, site_sparse=site_sparse,
                           site_alpha=site_alpha, xi=xi, builder2=builder2,
                           sparse_desc2=sparse_desc2, alpha2=alpha2,
                           train_pred=(Phi_flat @ alpha_flat).reshape(F, D))

    train_rmse, train_r2 = _rmse_r2(model.train_pred, mu)
    metrics = {"train_rmse": train_rmse, "train_r2": train_r2,
               "n_train_frames": F, "fit_seconds": time.time() - t0}
    print(f"[gap_dipole] fit models={models} M={sizes} xi={xi} sigma={sigma:g} "
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
        "gap": {"sparse_per_species": M_req if isinstance(cfg.gap_sparse_per_species, dict) else M_req[models[0]],
                "kernel_power": xi,
                "sigma": sigma, "sparse_jitter": float(cfg.gap_sparse_jitter),
                "site_models": models,
                "model_delta": {m: float(cfg.gap_model_delta[m]) for m in models},
                "site_channels": int(cfg.gap_site_channels),
                "site_lmax": int(cfg.gap_site_lmax),
                "cross_lmax": int(cfg.gap_cross_lmax),
                "delta_soap": float(cfg.gap_delta_soap),
                # zeta only where live (screened radial)
                **({"zeta": float(cfg.dipole_zeta)}
                   if cfg.dipole_radial == "screened" else {}),
                "radial": cfg.dipole_radial,
                "cutoff": resolved_dipole_cutoff(cfg),
                "channels": int(cfg.dipole_channels),
                "kernel_channels": kernel_channels(cfg),
                # per-(sparse-point, channel, neighbor-species) columns
                "pair_columns": pair,
                "cross_channels": int(cfg.gap_cross_channels),
                **({"lambda_shear": float(cfg.gap_lambda_shear)} if int(cfg.target_mode) == 2 else {}),
                "extra_rcut": cfg.gap_extra_rcut,
                "n_columns": n_columns,
                "M_per_species": {m: {str(z): n for z, n in zip(cfg.types, sz)}
                                  for m, sz in zip(models, sizes)},
                **({"M_per_species_2": {str(z): n
                                        for z, n in zip(cfg.types, sizes[-1])}}
                   if descs2 is not None else {})},
        "metrics": metrics,
    }
    os.makedirs("GAP", exist_ok=True)
    # Uniquify: same-second fits (scan loops) must not clobber records.
    stem = os.path.join(
        "GAP", f"gap_M{M_req[models[0]]}_x{xi}_{time.strftime('%Y%m%d_%H%M%S')}")
    path, k = stem + ".json", 1
    while os.path.exists(path):
        path, k = f"{stem}_{k}.json", k + 1
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"[gap_dipole] run record → {path}")
    return model


def predict(model: GapDipoleModel, frames: list[Atoms]) -> np.ndarray:
    """μ [F, 3] (or α [F, 6]) for frames. soap-only polarisability on orthorhombic
    periodic cells → batched GPU forward (gap_gpu.predict_gpu, matches
    _predict_numpy to 1e-3 Å³); everything else → float64 NumPy."""
    import gap_gpu
    if frames and gap_gpu.gpu_applicable(model) and gap_gpu._cell_ok(frames, resolved_dipole_rc(model.cfg)):
        return gap_gpu.predict_gpu(model, frames)
    return _predict_numpy(model, frames)


def _predict_numpy(model: GapDipoleModel, frames: list[Atoms]) -> np.ndarray:
    """Through _phi_frame — the fit's own contraction (μ = Φα) — so fit and predict cannot drift."""
    cfg = model.cfg
    models = list(cfg.gap_site_models)
    tis = [np.asarray(t) for t in tnep_data.assign_type_indices(frames, cfg.types)]
    vs = [_v_atoms(a, ti, cfg) for a, ti in zip(frames, tis)]
    site = _site_terms(cfg, model.builder, frames, tis, vs)
    descs2 = (_descriptors(model.builder2, frames)
              if model.builder2 is not None else None)
    alpha_flat = np.concatenate([a for m in models for a in model.site_alpha[m]] + list(model.alpha2 or []))
    d_soap = float(cfg.gap_delta_soap)
    out = None
    for f, (atoms, ti, v) in enumerate(zip(frames, tis, vs)):
        scales = _frame_scales(site, model.site_sparse, f, descs2, model.sparse_desc2, v, d_soap)
        row = _phi_frame(scales, ti, model.xi) @ alpha_flat
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
    # CLI entry: fit, evaluate, write the run record. Knobs are pinned here so a
    # working-tree default edit cannot leak into the fit.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # datasets/, GAP/

    cfg = GAPconfig()

    # GAP knobs — tweak freely:
    cfg.gap_sparse_per_species = 40
    cfg.gap_kernel_power = 2
    cfg.gap_sigma = 2e-2

    fit_gap_dipole(cfg)
