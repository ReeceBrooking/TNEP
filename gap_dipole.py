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

Columns = M_tot·K can exceed the 3·F equations (underdetermined); the ridge
handles it — train RMSE approaches 0 there and test RMSE is the metric.
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
from TNEPconfig import TNEPconfig, resolved_dipole_cutoff


@dataclass
class GapDipoleModel:
    """In-memory GAP dipole model — never persisted (refits in seconds).

    Column/α block order (one contract, shared by fit + predict via
    _phi_frame): scale → species → ξ → (m, k). Per (scale, species) the
    block is [len(xi)·M_t·K] flat, ξ-major. The scale-2 fields are None
    unless cfg.gap_extra_rcut is set."""
    cfg: TNEPconfig
    builder: object                   # descriptor builder from the fitting cfg
    sparse_desc: list[np.ndarray]     # per type t: [M_t, Q] unit-norm rows
    alpha: list[np.ndarray]           # per type t: [len(xi)·M_t·K] flat
    xi: tuple[int, ...]               # kernel powers — one column block each
    builder2: object = None           # second-scale builder (gap_extra_rcut)
    sparse_desc2: list[np.ndarray] | None = None
    alpha2: list[np.ndarray] | None = None


def _pair_columns(cfg: TNEPconfig) -> bool:
    """True when the neighbor-type axis of V becomes design columns
    (any dipole_species_pairs but "off"/legacy False)."""
    sp = cfg.dipole_species_pairs
    return not (sp == "off" or sp is False)


def _xi_tuple(cfg: TNEPconfig) -> tuple[int, ...]:
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


def _validate(cfg: TNEPconfig) -> None:
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


def _v_atoms(atoms: Atoms, ti: np.ndarray, cfg: TNEPconfig) -> np.ndarray:
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
               v: np.ndarray, ti: np.ndarray,
               xis: tuple[int, ...]) -> np.ndarray:
    """[3, n_columns] design block for one frame — THE column contract for
    fit AND predict. scales = [(D, sparse_desc)] per length scale; each
    scale's kernels couple its own descriptors to its own sparse points ONLY
    (scales never cross, like species). Block order: scale → species → ξ →
    (m, k); the base Gram is computed once per (scale, species) and raised
    elementwise to each ξ (sum-of-kernels column blocks). The geometry
    vectors v are scale-independent (bond basis, dipole_rc)."""
    cols = []
    for D, sparse_desc in scales:
        for t, Dm in enumerate(sparse_desc):
            sel = ti == t
            G = D[sel] @ Dm.T                           # [n_t, M_t] base Gram
            vt = v[sel]
            for xi in xis:
                cols.append(np.einsum("nm,nxk->xmk", G ** xi, vt)
                            .reshape(3, -1))            # [3, M_t·K]
    return np.concatenate(cols, axis=1)


def _targets(frames: list[Atoms], cfg: TNEPconfig) -> np.ndarray:
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


def fit_gap_dipole(cfg: TNEPconfig) -> GapDipoleModel:
    """Fit the sparse-kernel dipole model on cfg.data_path; evaluate
    cfg.test_data_path when set; write a run record JSON to GAP/.

    Solver memory (cfg.gap_solver): "primal" builds the dense augmented
    [3F + n_columns, n_columns] — ~0.6 GB at 8k columns, n_columns² scaling,
    ~18 GB (OOM) at the 48k columns a 2-scale × 3-ξ config reaches; "dual"
    solves the [3F, 3F] Gram instead — Φ itself then dominates (~1.7 GB at
    F=1500, 48k columns), memory linear in columns, same α to numerical
    precision. Multi-scale (gap_extra_rcut) ≈ 2× descriptor build time,
    fit and predict."""
    _validate(cfg)
    t0 = time.time()

    dataset, types_int = tnep_data.collect(cfg)
    if cfg.total_N is not None:
        cfg.randomise(dataset)
        dataset = [dataset[k] for k in cfg.indices]
        types_int = [types_int[k] for k in cfg.indices]
    tis = [np.asarray(t) for t in types_int]

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
    # Columns multiply: n_scales × per-species M × len(ξ list) × K.
    n_columns = sum(sum(sz) for sz in sizes) * len(xis) * n_vcols

    # Φ frame-by-frame — the full N_envs × M kernel never exists.
    F = len(dataset)
    Phi = np.empty((F, 3, n_columns))
    for f in range(F):
        scales = [(descs[f], sparse_desc)]
        if descs2 is not None:
            scales.append((descs2[f], sparse_desc2))
        Phi[f] = _phi_frame(scales, vs[f], tis[f], xis)
    Phi_flat = Phi.reshape(3 * F, n_columns)

    # Explicit ridge solve, λ_eff relative to mean(diag(ΦᵀΦ)) so λ is
    # dimensionless. Two mathematically identical routes (cfg.gap_solver):
    #   "primal": augmented [Φ; √λ_eff·I] lstsq — peak memory ~n_columns²
    #             (the dense identity block + gelsd workspace).
    #   "dual":   α = Φᵀ(ΦΦᵀ + λ_eff·I)⁻¹μ — the Gram is [3F, 3F]
    #             regardless of n_columns; memory linear in columns.
    lam = float(cfg.gap_ridge_lambda)
    lam_eff = lam * float(np.mean(np.sum(Phi_flat ** 2, axis=0)))
    solver = str(getattr(cfg, "gap_solver", "primal"))
    if solver == "primal":
        A = np.vstack([Phi_flat, np.sqrt(lam_eff) * np.eye(n_columns)])
        b = np.concatenate([mu.reshape(-1), np.zeros(n_columns)])
        alpha_flat = np.linalg.lstsq(A, b, rcond=None)[0]
    elif solver == "dual":
        G = Phi_flat @ Phi_flat.T                        # [3F, 3F]
        G[np.diag_indices_from(G)] += lam_eff
        alpha_flat = Phi_flat.T @ np.linalg.solve(G, mu.reshape(-1))
    else:
        raise ValueError(f"unknown gap_solver {solver!r} "
                         f"(expected 'primal' or 'dual')")
    blocks = [n * len(xis) * n_vcols for sz in sizes for n in sz]
    parts = np.split(alpha_flat, np.cumsum(blocks)[:-1])
    alpha, alpha2 = parts[:cfg.num_types], parts[cfg.num_types:] or None

    model = GapDipoleModel(cfg=cfg, builder=builder, sparse_desc=sparse_desc,
                           alpha=alpha, xi=xis, builder2=builder2,
                           sparse_desc2=sparse_desc2, alpha2=alpha2)

    train_rmse, train_r2 = _rmse_r2(
        (Phi_flat @ alpha_flat).reshape(F, 3), mu)
    metrics = {"train_rmse": train_rmse, "train_r2": train_r2,
               "n_train_frames": F, "fit_seconds": time.time() - t0}
    print(f"[gap_dipole] fit M={sizes} xi={list(xis)} lambda={lam:g} "
          f"n_columns={n_columns} solver={solver}: "
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
                "solver": solver,
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
    alpha_flat = np.concatenate(model.alpha + list(model.alpha2 or []))
    out = np.empty((len(frames), 3))
    for f, (atoms, ti, D) in enumerate(zip(frames, tis, descs)):
        ti = np.asarray(ti)
        scales = [(D, model.sparse_desc)]
        if descs2 is not None:
            scales.append((descs2[f], model.sparse_desc2))
        out[f] = _phi_frame(scales, _v_atoms(atoms, ti, cfg), ti,
                            model.xi) @ alpha_flat
    return out


if __name__ == "__main__":
    # IDE / CLI entry: fit on cfg.data_path, evaluate cfg.test_data_path,
    # print metrics, write the GAP/ run record. The GAP-relevant readout
    # fields are pinned HERE rather than read from TNEPconfig class defaults,
    # so working-tree experiment values cannot leak into the fit.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # datasets/, GAP/

    cfg = TNEPconfig()
    cfg.gap_solver = "dual"
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
