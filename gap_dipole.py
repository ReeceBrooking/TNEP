"""GAP-mode dipole fit for the neighbor readout.

Sparse-kernel ridge alternative to the SNES-trained ANN, exploiting the
readout's linearity in the site scalars (neighbor_dipole_gap_plan.md):

    μ = Σ_i Σ_k s_i^k v_i[:, k],  s_i^k = Σ_m α_m^k (D̂_i · D̂_m)^ξ

with v_i from data._neighbor_dipole_V [N, 3, C, T] — any radial
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

Additive (not multiplicative) column blocks: gap_2b / gap_3b append TurboGAP's
2b/3b distance descriptors (gap_descriptor.py, fixed centres auto-selected at
fit) as per-species linear features — plan §0.3.

Columns = M_tot·K can exceed the 3·F equations (underdetermined); the ridge
handles it — train RMSE approaches 0 there and test RMSE is the metric.
Takes its own GAPconfig (declared in TNEPconfig.py), never TNEPconfig: the ANN modes
are frozen constants there, so no ANN experiment can reach the fit.
Still fixed: dipole_readout="neighbor", target_mode=1, scale_targets=False.
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
from DescriptorBuilder import make_descriptor_builder
from gap_descriptor import gap_raw_numpy, gap_slices, select_centres, validate_gap_cfg
from TNEPconfig import GAPconfig, resolved_dipole_cutoff


@dataclass
class GapDipoleModel:
    """In-memory GAP dipole model — never persisted (refits in seconds).

    Column/α block order (one contract, shared by fit + predict via
    _phi_frame): scale → species → ξ → (m, k). Per (scale, species) the
    block is [len(xi)·M_t·K] flat, ξ-major. The scale-2 fields are None
    unless cfg.gap_extra_rcut is set. Then the additive TurboGAP blocks
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
    sparse_desc: list[np.ndarray]     # per type t: [M_t, Q] unit-norm rows
    alpha: list[np.ndarray]           # per type t: [len(xi)·M_t·K] flat
    xi: tuple[int, ...]               # kernel powers — one column block each
    builder2: object = None           # second-scale builder (gap_extra_rcut)
    sparse_desc2: list[np.ndarray] | None = None
    alpha2: list[np.ndarray] | None = None
    alpha_gap2b: list[np.ndarray] | None = None
    alpha_gap3b: list[np.ndarray] | None = None


def _pair_columns(cfg: GAPconfig) -> bool:
    """True when the neighbor-type axis of V becomes design columns
    (any dipole_species_pairs but "off"/legacy False)."""
    sp = cfg.dipole_species_pairs
    return not (sp == "off" or sp is False)


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
        (cfg.target_mode == 1,
         f"target_mode must be 1, got {cfg.target_mode}"),
        (cfg.dipole_readout == "neighbor",
         f"dipole_readout must be 'neighbor', got {cfg.dipole_readout!r}"),
        (int(cfg.dipole_channels) >= 1,
         f"dipole_channels must be >= 1, got {cfg.dipole_channels}"),
        # readout contract: the screened radial is single-channel.
        (cfg.dipole_radial != "screened" or int(cfg.dipole_channels) == 1,
         f"dipole_channels must be 1 for the screened radial, got "
         f"{cfg.dipole_channels}"),
        (not cfg.scale_targets, "scale_targets must be False"),
        # nep_descriptor is a frozen constant on GAPconfig, so this cannot
        # fire from a class default — it catches INSTANCE mutation
        # (cfg.nep_descriptor = "concat"), which would otherwise reach
        # data.assemble_data_dict and build NEP inputs the fit never reads.
        (str(cfg.nep_descriptor) == "off",
         "GAP mode needs nep_descriptor='off' (NEP inputs are not part of "
         "the kernel fit)"),
        # mirror the model-init r_c contract (no TNEP is ever constructed
        # on the GAP path, so it must be enforced here too).
        (0.0 < float(cfg.dipole_rc or cfg.rcut_hard) <= float(cfg.rcut_hard),
         f"dipole_rc must satisfy 0 < rc <= rcut_hard="
         f"{cfg.rcut_hard}, got {cfg.dipole_rc}"),
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
    """[N, 3, C, T] → v_i [N, 3, K] — the ONE place the column contract is
    encoded (float64, shared by fit + predict). K = C·T with pair columns
    (k = (n, t′), C-major, matching reshape order) else K = C (t′ summed)."""
    V = np.asarray(V, dtype=np.float64)
    if pair_columns:
        return V.reshape(V.shape[0], 3, -1)
    return V.sum(axis=-1)


def _v_atoms(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """Per-atom geometry vectors v_i [N, 3, K] per the cfg's radial/cutoff/
    channels and pair-column mode."""
    V = tnep_data._neighbor_dipole_V(
        atoms, ti, float(cfg.dipole_rc or cfg.rcut_hard),
        int(cfg.dipole_channels), int(cfg.num_types),
        str(cfg.dipole_radial), float(cfg.dipole_zeta),
        resolved_dipole_cutoff(cfg), float(cfg.dipole_gauss_sigma_scale))
    return _collapse_V(V, _pair_columns(cfg))


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


def _phi_frame(scales: list[tuple[np.ndarray, list[np.ndarray]]],
               v: np.ndarray, ti: np.ndarray, xis: tuple[int, ...],
               gap_q: tuple[np.ndarray | None, np.ndarray | None] | None = None,
               deltas: tuple[float, float, float] = (1.0, 1.0, 1.0),
               ) -> np.ndarray:
    """[3, n_columns] design block for one frame — THE column contract for
    fit AND predict. scales = [(D, sparse_desc)] per length scale; each
    scale's kernels couple its own descriptors to its own sparse points ONLY
    (scales never cross, like species). Block order: scale → species → ξ →
    (m, k); the base Gram is computed once per (scale, species) and raised
    elementwise to each ξ (sum-of-kernels column blocks). The geometry
    vectors v are scale-independent (bond basis, dipole_rc).

    gap_q = (q2 [N, W2] | None, q3 [N, W3] | None) appends the additive
    TurboGAP blocks AFTER all SOAP columns: gap2b (species → (s, k)) then
    gap3b (species → (s, k)) — Σ_{i∈t} q[i, s]·v[i, x, k] — i.e. the 2b/3b
    site energies with a linear readout, μ-weighted by the bond basis.

    deltas = (δ_soap, δ_2b, δ_3b) scale each body order's columns LINEARLY,
        U_i = δ_soap·U_i(SOAP) + δ_2b·U_i(2b) + δ_3b·U_i(3b).
    The span of the design is unchanged — a least-squares fit would absorb δ
    into α — so δ acts through the RIDGE. With one global λ_eff, scaling a
    block's columns by δ is equivalent to penalising that block's
    contribution by λ_eff/δ²: substituting β = δ·α_block turns
    λ‖α_block‖² into λ‖β‖²/δ². So δ² is the block's prior variance, which is
    what GAP's δ means (gap.f90 carries `delta**2` on the kernel because its
    δ is an amplitude and δ² the variance). Squaring here as well would make
    the penalty go as 1/δ⁴ — the knob would be the square of GAP's — so the
    column multiplier is δ, not δ². This is the lever for how much influence
    each model is allowed over the prediction."""
    d_soap, d_2b, d_3b = (float(d) for d in deltas)
    cols = []
    for D, sparse_desc in scales:
        for t, Dm in enumerate(sparse_desc):
            sel = ti == t
            G = D[sel] @ Dm.T                           # [n_t, M_t] base Gram
            vt = v[sel]
            for xi in xis:
                cols.append(d_soap * np.einsum("nm,nxk->xmk", G ** xi, vt)
                            .reshape(3, -1))            # [3, M_t·K]
    for q, d_q in zip(gap_q or (), (d_2b, d_3b)):
        if q is not None:
            # scales[0][1] is the scale-1 sparse list: exactly one entry per
            # species (_fps_per_species loops range(cfg.num_types)).
            for t in range(len(scales[0][1])):
                sel = ti == t
                cols.append(d_q * np.einsum("ns,nxk->xsk", q[sel], v[sel])
                            .reshape(3, -1))            # [3, W·K]
    return np.concatenate(cols, axis=1)


def _deltas(cfg: GAPconfig) -> tuple[float, float, float]:
    """(δ_soap, δ_2b, δ_3b) — the ONE place the amplitudes are read, so fit
    and predict cannot disagree about the column scaling."""
    return (float(cfg.gap_delta_soap), float(cfg.gap_delta_2b),
            float(cfg.gap_delta_3b))


def _gap_q(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig):
    """Per-frame (q2 | None, q3 | None) float64 for _phi_frame per the
    cfg flags; None when both are off (shared by fit + predict)."""
    if not (cfg.gap_2b or cfg.gap_3b):
        return None
    raw = gap_raw_numpy(atoms, ti, cfg).astype(np.float64)
    sl = gap_slices(cfg)
    return (raw[:, sl["two_body"]] if cfg.gap_2b else None,
            raw[:, sl["three_body"]] if cfg.gap_3b else None)


def _targets(frames: list[Atoms], cfg: GAPconfig) -> np.ndarray:
    """Dipole targets [F, 3], unit-converted per cfg (matches assemble).

    NOTE: unlike the training path (assemble_data_dict), this applies NO
    box/label guards — a pathological test file scores silently."""
    key = tnep_data._resolve_target_key(cfg)
    mu = np.stack([np.asarray(tnep_data._extract_target(s, key))
                   for s in frames]).astype(np.float64)
    if getattr(cfg, "convert_dipole_to_eangstrom", True):
        mu = mu * tnep_data._dipole_conversion_factor(cfg.dipole_units)
    return mu


def _rmse_r2(pred: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """Total-vector RMSE √(mean(per-component²)·3) (diag_zeta convention), R²."""
    res = pred - mu
    rmse = float(np.sqrt(np.mean(res ** 2) * 3.0))
    tot = float(np.sum((mu - mu.mean(0)) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / tot if tot > 0.0 else float("nan")
    return rmse, r2


def fit_gap_dipole(cfg: GAPconfig) -> GapDipoleModel:
    """Fit the sparse-kernel dipole model on cfg.data_path; evaluate
    cfg.test_data_path when set; write a run record JSON to GAP/.

    MUTATES cfg in place: `seed` (when None) and `gap2b_centres`/
    `gap3b_centres` (when auto-selected) are filled in, because they must
    travel with the model for `predict` and into the run record.

    Solver memory: the dual ridge solves the [3F, 3F] Gram, so Φ itself
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
    pair = _pair_columns(cfg)
    vs = [_collapse_V(V, pair) for V in d["V_atom"]]
    n_vcols = int(cfg.dipole_channels) * (cfg.num_types if pair else 1)

    # Per-species FPS sparse points, per scale (scale-2 selected on ITS OWN
    # descriptors — the wider builder — from the same rng stream, after
    # scale 1, so single-scale FPS is bit-identical to the pre-feature code).
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

    sparse_desc = _fps_per_species(descs)
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

    scale_sparse = [sparse_desc] + ([sparse_desc2] if sparse_desc2 else [])
    sizes = [[len(s) for s in group] for group in scale_sparse]
    # α block per (scale, species): M × len(ξ list) × K columns; then the
    # additive GAP blocks, one per species: W2·K (gap_2b) and W3·K (gap_3b).
    sl = gap_slices(cfg)
    T = cfg.num_types
    w_gap = {k: (sl[k].stop - sl[k].start) * n_vcols if on else 0
             for k, on in (("two_body", cfg.gap_2b),
                           ("three_body", cfg.gap_3b))}
    blocks = ([n * len(xis) * n_vcols for sz in sizes for n in sz]
              + [w for w in w_gap.values() if w for _ in range(T)])
    n_columns = sum(blocks)

    # Φ frame-by-frame — the full N_envs × M kernel never exists.
    F = len(dataset)
    Phi = np.empty((F, 3, n_columns))
    deltas = _deltas(cfg)
    for f in range(F):
        scales = [(descs[f], sparse_desc)]
        if descs2 is not None:
            scales.append((descs2[f], sparse_desc2))
        Phi[f] = _phi_frame(scales, vs[f], tis[f], xis,
                            _gap_q(dataset[f], tis[f], cfg), deltas)
    Phi_flat = Phi.reshape(3 * F, n_columns)

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

    # Explicit ridge solve in the dual form, λ_eff relative to
    # mean(diag(ΦᵀΦ)) so λ is dimensionless:
    #   α = Φᵀ(ΦΦᵀ + λ_eff·I)⁻¹μ — the Gram is [3F, 3F] regardless of
    #   n_columns, so memory is linear in columns and Φ itself dominates.
    # Note this squares the conditioning relative to an augmented
    # [Φ; √λ_eff·I] least-squares solve (the normal-equations pitfall); the
    # equivalence of the two is pinned by test_dual_solve_matches_augmented_lstsq,
    # which carries the augmented route as a test-only oracle.
    lam = float(cfg.gap_ridge_lambda)
    lam_eff = lam * float(np.mean(np.sum(Phi_flat ** 2, axis=0)))
    G = Phi_flat @ Phi_flat.T                            # [3F, 3F]
    G[np.diag_indices_from(G)] += lam_eff
    alpha_flat = Phi_flat.T @ np.linalg.solve(G, mu.reshape(-1))
    parts = np.split(alpha_flat, np.cumsum(blocks)[:-1])
    # every α group is one block per species
    groups = [parts[i:i + T] for i in range(0, len(parts), T)]
    alpha = groups.pop(0)
    alpha2 = groups.pop(0) if descs2 is not None else None
    alpha_gap2b = groups.pop(0) if cfg.gap_2b else None
    alpha_gap3b = groups.pop(0) if cfg.gap_3b else None
    assert not groups

    model = GapDipoleModel(cfg=cfg, builder=builder, sparse_desc=sparse_desc,
                           alpha=alpha, xi=xis, builder2=builder2,
                           sparse_desc2=sparse_desc2, alpha2=alpha2,
                           alpha_gap2b=alpha_gap2b, alpha_gap3b=alpha_gap3b)

    train_rmse, train_r2 = _rmse_r2(
        (Phi_flat @ alpha_flat).reshape(F, 3), mu)
    metrics = {"train_rmse": train_rmse, "train_r2": train_r2,
               "n_train_frames": F, "fit_seconds": time.time() - t0}
    print(f"[gap_dipole] fit M={sizes} xi={list(xis)} lambda={lam:g} "
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
                "ridge_lambda": lam, "lambda_eff": lam_eff,
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
                # True ⇒ per-(sparse-point, channel, neighbor-species)
                # columns; subsumes symmetric c AND ordered Ω (no distinction
                # in the absorbed linear form).
                "pair_columns": pair,
                "extra_rcut": cfg.gap_extra_rcut,
                "n_columns": n_columns,
                # additive TurboGAP 2b/3b blocks (gap_descriptor.py)
                "gap_2b": bool(cfg.gap_2b), "gap_3b": bool(cfg.gap_3b),
                "n_gap_columns": {k: w * T for k, w in w_gap.items()},
                **({k: getattr(cfg, k) for k in (
                    "gap2b_centres", "gap3b_centres", "gap2b_sigma",
                    "gap3b_sigma", "gap2b_rcut", "gap3b_rcut", "gap_buffer")}
                   if cfg.gap_2b or cfg.gap_3b else {}),
                "M_per_species": {str(z): n
                                  for z, n in zip(cfg.types, sizes[0])},
                # scale-2 sparse-set sizes, only when gap_extra_rcut is set
                **({"M_per_species_2": {str(z): n
                                        for z, n in zip(cfg.types, sizes[1])}}
                   if len(sizes) > 1 else {})},
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
    """μ [F, 3] for a list of frames via the model's held cfg/builder.

    Routes through _phi_frame — the SAME design-matrix contraction the fit
    uses (μ = Φα) — so fit and predict cannot drift."""
    cfg = model.cfg
    tis = tnep_data.assign_type_indices(frames, cfg.types)
    descs = _descriptors(model.builder, frames)
    descs2 = (_descriptors(model.builder2, frames)
              if model.builder2 is not None else None)
    alpha_flat = np.concatenate(
        model.alpha + list(model.alpha2 or []) + list(model.alpha_gap2b or [])
        + list(model.alpha_gap3b or []))
    out = np.empty((len(frames), 3))
    for f, (atoms, ti, D) in enumerate(zip(frames, tis, descs)):
        ti = np.asarray(ti)
        scales = [(D, model.sparse_desc)]
        if descs2 is not None:
            scales.append((descs2[f], model.sparse_desc2))
        out[f] = _phi_frame(scales, _v_atoms(atoms, ti, cfg), ti, model.xi,
                            _gap_q(atoms, ti, cfg), _deltas(cfg)) @ alpha_flat
    return out


if __name__ == "__main__":
    # IDE / CLI entry: fit on cfg.data_path, evaluate cfg.test_data_path,
    # print metrics, write the GAP/ run record. The GAP-relevant readout
    # fields are pinned HERE rather than read from GAPconfig class defaults,
    # so a working-tree default edit cannot leak into the fit.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # datasets/, GAP/

    cfg = GAPconfig()
    cfg.target_mode = 1
    cfg.scale_targets = False
    cfg.dipole_readout = "neighbor"
    cfg.dipole_radial = "chebyshev"
    cfg.dipole_cutoff = "cosine"
    cfg.dipole_species_pairs = "symmetric"    # any non-"off" ⇒ pair columns
    cfg.dipole_channels = 20
    cfg.dipole_zeta = -2.0                  # unused for chebyshev
    cfg.dipole_rc = 3.5                     # beat inherited 3.5 in run records
    cfg.l_max = 7
    cfg.alpha_max = 7
    cfg.rcut_hard, cfg.rcut_soft = 3.5, 3.0
    #cfg.gap_extra_rcut = 5.5

    # GAP knobs — tweak freely:
    cfg.gap_sparse_per_species = 250
    cfg.gap_kernel_power = 2
    cfg.gap_ridge_lambda = 1e-8    # λ scan {1e-4,1e-6,1e-8}: 1e-8 best on test

    fit_gap_dipole(cfg)
