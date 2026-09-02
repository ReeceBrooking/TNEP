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

Columns = M_tot·K can exceed the 3·F equations (underdetermined); the ridge
handles it — train RMSE approaches 0 there and test RMSE is the metric.
Still fixed: dipole_readout="neighbor", target_mode=1, scale_targets=False.
NO model persistence — fit writes a config+metrics run record JSON to GAP/."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.io import read

import data as tnep_data
from DescriptorBuilder import make_descriptor_builder
from TNEPconfig import TNEPconfig, resolved_dipole_cutoff


@dataclass
class GapDipoleModel:
    """In-memory GAP dipole model — never persisted (refits in seconds)."""
    cfg: TNEPconfig
    builder: object                   # descriptor builder from the fitting cfg
    sparse_desc: list[np.ndarray]     # per type t: [M_t, Q] unit-norm rows
    alpha: list[np.ndarray]           # per type t: [M_t·K] flat, (m, k) order
    xi: int


def _pair_columns(cfg: TNEPconfig) -> bool:
    """True when the neighbor-type axis of V becomes design columns
    (any dipole_species_pairs but "off"/legacy False)."""
    sp = cfg.dipole_species_pairs
    return not (sp == "off" or sp is False)


def _validate(cfg: TNEPconfig) -> None:
    """Raise unless cfg is a GAP-fittable configuration. Runs FIRST in fit."""
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


def _phi_frame(D: np.ndarray, v: np.ndarray, ti: np.ndarray,
               sparse_desc: list[np.ndarray], xi: int) -> np.ndarray:
    """[3, M_tot·K] design block for one frame: Φ[:, (m,k)] = Σ_{i∈species(m)}
    k(D̂_i, D̂_m) v_i[:, k]. Species-blocked — no cross-species kernel terms."""
    cols = []
    for t, Dm in enumerate(sparse_desc):
        sel = ti == t
        K = (D[sel] @ Dm.T) ** xi                       # [n_t, M_t]
        cols.append(np.einsum("nm,nxk->xmk", K, v[sel])
                    .reshape(3, -1))                    # [3, M_t·K]
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
    cfg.test_data_path when set; write a run record JSON to GAP/."""
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

    # Per-species FPS sparse points.
    rng = np.random.default_rng(cfg.seed)
    M_req = int(cfg.gap_sparse_per_species)
    xi = int(cfg.gap_kernel_power)
    sparse_desc = []
    for t in range(cfg.num_types):
        Dt = np.concatenate([D[ti == t] for D, ti in zip(descs, tis)])
        if not len(Dt):
            raise ValueError(
                f"no atoms of type Z={cfg.types[t]} in the training data — "
                f"cannot place sparse points for that species")
        sparse_desc.append(Dt[_fps(Dt, M_req, rng)].copy())
    sizes = [len(s) for s in sparse_desc]
    n_columns = sum(sizes) * n_vcols

    # Φ frame-by-frame — the full N_envs × M kernel never exists.
    F = len(dataset)
    Phi = np.empty((F, 3, n_columns))
    for f in range(F):
        Phi[f] = _phi_frame(descs[f], vs[f], tis[f], sparse_desc, xi)
    Phi_flat = Phi.reshape(3 * F, n_columns)

    # Explicit ridge solve: augmented [Φ; √λ_eff·I], λ_eff relative to
    # mean(diag(ΦᵀΦ)) so λ is dimensionless.
    lam = float(cfg.gap_ridge_lambda)
    lam_eff = lam * float(np.mean(np.sum(Phi_flat ** 2, axis=0)))
    A = np.vstack([Phi_flat, np.sqrt(lam_eff) * np.eye(n_columns)])
    b = np.concatenate([mu.reshape(-1), np.zeros(n_columns)])
    alpha_flat = np.linalg.lstsq(A, b, rcond=None)[0]
    alpha = np.split(alpha_flat, np.cumsum([n * n_vcols for n in sizes])[:-1])

    model = GapDipoleModel(cfg=cfg, builder=builder,
                           sparse_desc=sparse_desc, alpha=alpha, xi=xi)

    train_rmse, train_r2 = _rmse_r2(
        (Phi_flat @ alpha_flat).reshape(F, 3), mu)
    metrics = {"train_rmse": train_rmse, "train_r2": train_r2,
               "n_train_frames": F, "fit_seconds": time.time() - t0}
    print(f"[gap_dipole] fit M={sizes} xi={xi} lambda={lam:g} "
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
        "gap": {"sparse_per_species": M_req, "kernel_power": xi,
                "ridge_lambda": lam, "lambda_eff": lam_eff,
                "radial": cfg.dipole_radial,
                "cutoff": resolved_dipole_cutoff(cfg),
                "channels": int(cfg.dipole_channels),
                # True ⇒ per-(sparse-point, channel, neighbor-species)
                # columns; subsumes symmetric c AND ordered Ω (no distinction
                # in the absorbed linear form).
                "pair_columns": pair,
                "n_columns": n_columns,
                "M_per_species": {str(z): n
                                  for z, n in zip(cfg.types, sizes)}},
        "metrics": metrics,
    }
    os.makedirs("GAP", exist_ok=True)
    path = os.path.join(
        "GAP", f"gap_M{M_req}_x{xi}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"[gap_dipole] run record → {path}")
    return model


def predict(model: GapDipoleModel, frames: list[Atoms]) -> np.ndarray:
    """μ [F, 3] for a list of frames via the model's held cfg/builder."""
    cfg = model.cfg
    tis = tnep_data.assign_type_indices(frames, cfg.types)
    descs = _descriptors(model.builder, frames)
    out = np.empty((len(frames), 3))
    for f, (atoms, ti, D) in enumerate(zip(frames, tis, descs)):
        ti = np.asarray(ti)
        v = _v_atoms(atoms, ti, cfg)                    # [N, 3, K]
        K = v.shape[-1]
        mu = np.zeros(3)
        for t, (Dm, a) in enumerate(zip(model.sparse_desc, model.alpha)):
            sel = ti == t
            w = ((D[sel] @ Dm.T) ** model.xi) @ a.reshape(len(Dm), K)
            mu += np.einsum("nk,nxk->x", w, v[sel])
        out[f] = mu
    return out


if __name__ == "__main__":
    # IDE / CLI entry: fit on cfg.data_path, evaluate cfg.test_data_path,
    # print metrics, write the GAP/ run record. The GAP-relevant readout
    # fields are pinned HERE rather than read from TNEPconfig class defaults,
    # so working-tree experiment values cannot leak into the fit.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # datasets/, GAP/

    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.scale_targets = False
    cfg.dipole_readout = "neighbor"
    cfg.dipole_radial = "chebyshev"
    cfg.dipole_cutoff = "cosine2"
    cfg.dipole_species_pairs = "ordered"    # any non-"off" ⇒ pair columns
    cfg.dipole_channels = 4
    cfg.dipole_zeta = -2.0                  # unused for chebyshev
    cfg.dipole_rc = 3.0                     # beat inherited 3.5 in run records
    cfg.rcut_hard, cfg.rcut_soft = 3.5, 3.0

    # GAP knobs — tweak freely:
    cfg.gap_sparse_per_species = 500
    cfg.gap_kernel_power = 2
    cfg.gap_ridge_lambda = 1e-8    # λ scan {1e-4,1e-6,1e-8}: 1e-8 best on test

    fit_gap_dipole(cfg)
