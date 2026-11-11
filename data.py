from __future__ import annotations

import tensorflow as tf
import numpy as np
from TNEPconfig import (TNEPconfig, resolved_dipole_cutoff,
                        resolved_dipole_rc, neighbor_readout_on, bond_rank)
from DescriptorBuilder import make_descriptor_builder
from ase.io import read
from ase import Atoms
from ase.neighborlist import neighbor_list
from nep_descriptor import nep_pair_list, nep_raw_numpy, nep_pairs_numpy


BOHR_TO_ANGSTROM = 0.529177210903
DEBYE_TO_EANGSTROM = 0.20819434

# No-PBC box: 1000 Å cube, large enough that MIC never wraps any displacement.
_NO_PBC_BOX = 1000.0 * np.eye(3, dtype=np.float32)


def cell_to_box(atoms) -> np.ndarray:
    """Return the cell matrix for *atoms*, or _NO_PBC_BOX if unset/zero.

    Unset (zero 3×3, det=0) or all-zero lattice → 1000 Å cube so MIC is a no-op
    while GPU code stays unconditional.
    """
    cell = atoms.cell.array.astype(np.float32)
    if np.allclose(cell, 0) or abs(np.linalg.det(cell)) < 1e-6:
        return _NO_PBC_BOX
    return cell


def _dipole_conversion_factor(dipole_units: str) -> float:
    """Factor converting dipole_units ("e*angstrom" | "e*bohr" | "debye") → e·Å."""
    if dipole_units == "e*angstrom":
        return 1.0
    elif dipole_units == "e*bohr":
        return BOHR_TO_ANGSTROM
    elif dipole_units == "debye":
        return DEBYE_TO_EANGSTROM
    else:
        raise ValueError(f"Unknown dipole_units: {dipole_units!r} "
                         f"(expected 'e*angstrom', 'e*bohr', or 'debye')")


def collect(cfg: TNEPconfig) -> tuple[list[Atoms], list[np.ndarray]]:
    """Load cfg.data_path, discover species, filter, drop bad data.

    Side effect: populates cfg.num_types and cfg.types.
    Returns (dataset, dataset_types_int) — Atoms list and parallel [N_i] int type indices.
    """
    dataset = read(cfg.data_path, index=":")
    dataset_types_int = []
    types = []

    for structure in dataset:
        structure_types_int = np.zeros_like(structure.numbers)

        for i in range(len(structure.numbers)):
            z = int(structure.numbers[i])  # cast np.int64 → Python int
            if z not in types:
                types.append(z)
            structure_types_int[i] = types.index(z)

        dataset_types_int.append(structure_types_int)

    cfg.num_types = len(types)
    cfg.types = types
    print("Number of species in raw dataset: " + str(cfg.num_types))
    print("Number of structures in raw dataset: " + str(len(dataset)))

    # Filter by species if configured
    if cfg.allowed_species is not None:
        dataset, dataset_types_int = filter_by_species(dataset, dataset_types_int, allowed_Z=cfg.allowed_species, mode=cfg.filter_mode)
        print("After species filter (" + cfg.filter_mode + "): " + str(len(dataset)) + " structures")

    # Recompute type list/indices after filtering. Coerce to Python int:
    # struct.numbers holds np.int64, which would later fail json.dumps in model_io.
    cfg.types = []
    for struct in dataset:
        for z in struct.numbers:
            zi = int(z)
            if zi not in cfg.types:
                cfg.types.append(zi)
    cfg.num_types = len(cfg.types)
    dataset_types_int = assign_type_indices(dataset, cfg.types)
    print("Species: " + str(cfg.types) + " (" + str(cfg.num_types) + " types)")

    # Filter bad data based on config flags
    dataset, dataset_types_int = filter_bad_data(dataset, dataset_types_int, cfg)

    # Per-type coverage: fraction of structures containing each type. Flags
    # under-represented species early (they regularise unstably).
    if dataset_types_int:
        from ase.data import chemical_symbols
        S = len(dataset_types_int)
        for t, z in enumerate(cfg.types):
            n_with_t = sum(1 for ts in dataset_types_int if (ts == t).any())
            sym = chemical_symbols[int(z)] if 0 <= int(z) < len(chemical_symbols) else "?"
            print(f"  Type {t} ({sym}, Z={int(z)}): present in {n_with_t/S:.1%} of structures")

    return dataset, dataset_types_int


def assign_type_indices(dataset: list[Atoms], types: list[int]) -> list[np.ndarray]:
    """Map atoms to indices into a known `types` list (no discovery) → list of [N_i] int arrays."""
    dataset_types_int = []
    for struct in dataset:
        structure_types_int = np.zeros_like(struct.numbers)
        for i, z in enumerate(struct.numbers):
            structure_types_int[i] = types.index(z)
        dataset_types_int.append(structure_types_int)
    return dataset_types_int


def _extract_target(structure: Atoms, target_key: str) -> tf.Tensor:
    """Extract target tensor; flatten 9-component polarizability to 6 if needed.

    Trailing NaNs are stripped: a trailing space inside a quoted value (e.g.
    mu="... ") makes ASE's extxyz parser emit a spurious NaN at the end.
    """
    if target_key in structure.info:
        raw = np.asarray(structure.info[target_key], dtype=np.float32)
    elif structure.calc is not None and target_key in structure.calc.results:
        raw = np.asarray(structure.calc.results[target_key], dtype=np.float32)
    else:
        raise KeyError(f"'{target_key}' not found in structure.info or calc.results")
    # Strip trailing NaN artefacts produced by trailing whitespace in quoted values
    if raw.ndim == 1:
        n = raw.size
        while n > 0 and np.isnan(raw[n - 1]):
            n -= 1
        raw = raw[:n]
    if raw.size == 9:
        # Flattened 3x3 row-major -> unique [xx, yy, zz, xy, yz, zx]
        raw = raw[[0, 4, 8, 1, 5, 6]]
    return tf.convert_to_tensor(raw, dtype=tf.float32)


def find_bad_data(dataset: list[Atoms], target_key: str) -> dict[str, list[int]]:
    """Structure indices with NaN positions / NaN targets / missing target key, by category."""
    nan_positions = []
    nan_targets = []
    missing_targets = []
    for i, structure in enumerate(dataset):
        if np.any(np.isnan(structure.positions)):
            nan_positions.append(i)
        try:
            target = _extract_target(structure, target_key).numpy()
        except KeyError:
            missing_targets.append(i)
            continue
        # size==0: an all-NaN target that the trailing-NaN strip emptied.
        if target.size == 0 or np.any(np.isnan(target)):
            nan_targets.append(i)
    return {'nan_positions': nan_positions,
            'nan_targets': nan_targets,
            'missing_targets': missing_targets}


def filter_bad_data(
    dataset: list[Atoms],
    dataset_types_int: list[np.ndarray],
    cfg: TNEPconfig,
) -> tuple[list[Atoms], list[np.ndarray]]:
    """Drop untrainable structures (NaN positions/targets, missing target key)."""
    target_key = _resolve_target_key(cfg)
    bad = find_bad_data(dataset, target_key)

    bad_indices: set[int] = set()
    for kind in ('nan_positions', 'nan_targets', 'missing_targets'):
        bad_indices.update(bad[kind])

    if bad_indices:
        dataset = [s for i, s in enumerate(dataset) if i not in bad_indices]
        dataset_types_int = [t for i, t in enumerate(dataset_types_int) if i not in bad_indices]
        print(
            f"[filter_bad_data] WARNING: dropped {len(bad_indices)} structures "
            f"(nan_targets={len(bad['nan_targets'])}, "
            f"nan_positions={len(bad['nan_positions'])}, "
            f"missing_targets={len(bad['missing_targets'])})"
        )

    return dataset, dataset_types_int


def _target_key_for_mode(target_mode: int) -> str:
    """Return the default Atoms.info key for the given target mode."""
    return {0: "energy", 1: "dipole", 2: "pol"}[target_mode]


def _resolve_target_key(cfg: TNEPconfig) -> str:
    """Return cfg.target_key if set (allows labels like "mu"/"alpha"), else the
    mode default ("energy"/"dipole"/"pol")."""
    if cfg.target_key is not None:
        return cfg.target_key
    return _target_key_for_mode(cfg.target_mode)


def component_labels(target_mode: int, num_components: int) -> list[str]:
    """Label per target component for mode 0/1/2 (generic "comp i" otherwise)."""
    if target_mode == 0:
        return ["Energy"]
    elif target_mode == 1:
        return ["x", "y", "z"]
    elif target_mode == 2:
        return ["xx", "yy", "zz", "xy", "yz", "zx"]
    return [f"comp {i}" for i in range(num_components)]


def print_score_summary(metrics: dict, cfg: TNEPconfig, prefix: str = "") -> None:
    """Print RMSE, R², per-component R² and cosine similarity from a TNEP.score() dict."""
    rmse = float(metrics["rmse"])
    r2 = float(metrics["r2"])
    r2_comp = metrics["r2_components"].numpy()
    labels = component_labels(cfg.target_mode, len(r2_comp))

    if "total_rmse" in metrics:
        print(f"\n{prefix} (per-atom) RMSE: {rmse:.4f}")
        print(f"{prefix} (per-atom) R²:   {r2:.4f}")
        print("Per-atom per-component R²:  " + "  ".join(
            f"{lbl}={r2_comp[i]:.4f}" for i, lbl in enumerate(labels)))

        total_rmse = float(metrics["total_rmse"])
        total_r2 = float(metrics["total_r2"])
        total_r2_comp = metrics["total_r2_components"].numpy()
        print(f"{prefix} (total)    RMSE: {total_rmse:.4f}")
        print(f"{prefix} (total)    R²:   {total_r2:.4f}")
        print("Total per-component R²:     " + "  ".join(
            f"{lbl}={total_r2_comp[i]:.4f}" for i, lbl in enumerate(labels)))
    else:
        print(f"\n{prefix} RMSE: {rmse:.4f}")
        print(f"{prefix} R²:   {r2:.4f}")
        print("Per-component R²:  " + "  ".join(
            f"{lbl}={r2_comp[i]:.4f}" for i, lbl in enumerate(labels)))

    if "cos_sim_mean" in metrics:
        cos_mean = float(metrics["cos_sim_mean"])
        cos_all = metrics["cos_sim_all"].numpy()
        print(f"Cosine similarity:  mean={cos_mean:.4f}  "
              f"min={cos_all.min():.4f}  max={cos_all.max():.4f}  "
              f"std={cos_all.std():.4f}")


def _get_forces(s: Atoms) -> np.ndarray | None:
    """Get per-atom forces [N, 3] from an Atoms object, or None."""
    for key in ("force", "forces"):
        if key in s.arrays:
            return np.asarray(s.arrays[key], dtype=np.float32)
    if s.calc is not None and "forces" in s.calc.results:
        return np.asarray(s.calc.results["forces"], dtype=np.float32)
    return None


def _get_virial(s: Atoms) -> np.ndarray | None:
    """Get virial as 6-component Voigt [xx,yy,zz,xy,yz,zx] from Atoms, or None."""
    if "virial" in s.info:
        v = np.asarray(s.info["virial"], dtype=np.float32)
        if v.shape == (3, 3):
            return np.array([v[0, 0], v[1, 1], v[2, 2],
                             v[0, 1], v[1, 2], v[2, 0]], dtype=np.float32)
        elif v.shape == (9,):
            v = v.reshape(3, 3)
            return np.array([v[0, 0], v[1, 1], v[2, 2],
                             v[0, 1], v[1, 2], v[2, 0]], dtype=np.float32)
        elif v.shape == (6,):
            return v
    return None


def _dataset_has_key(dataset: list[Atoms], getter) -> bool:
    """Check whether ALL structures in dataset have a given property."""
    return all(getter(s) is not None for s in dataset)


def assemble_data_dict(
    dataset: list[Atoms],
    types_int: list[np.ndarray],
    descriptors: list[tf.Tensor],
    gradients: list[list[tf.Tensor]],
    grad_index: list[list[list[int]]],
    cfg: TNEPconfig,
) -> dict:
    """List-form data dict: positions [N_i,3], Z_int [N_i], targets, boxes [3,3],
    descriptors [N_i,Q], gradients (N_i × [M,3,Q]), grad_index (N_i × [M] ints);
    plus forces/virials (PES, auto-detected), V_atom (neighbour readout),
    nep_raw / nep_pairs / nep_pad_D (nep_descriptor != "off").

    Dipoles are converted to e·Å and, when cfg.scale_targets, divided by N_atoms.
    """
    target_key = _resolve_target_key(cfg)
    targets = [_extract_target(s, target_key) for s in dataset]
    if cfg.target_mode == 1 and getattr(cfg, "convert_dipole_to_eangstrom", True):
        factor = _dipole_conversion_factor(cfg.dipole_units)
        if factor != 1.0:
            targets = [t * factor for t in targets]
    # Guards need total-cell dipoles (branch quantum e·L): run before per-atom scaling.
    if cfg.target_mode == 1 and neighbor_readout_on(cfg):
        _neighbor_dipole_guards(dataset, targets, cfg)
    if cfg.scale_targets and cfg.target_mode == 1:
        targets = [t / tf.cast(len(s), tf.float32) for t, s in zip(targets, dataset)]
    data = {
        "positions": [tf.convert_to_tensor(s.positions, dtype=tf.float32) for s in dataset],
        "Z_int": [tf.convert_to_tensor(t, dtype=tf.int32) for t in types_int],
        "targets": targets,
        "boxes": [tf.convert_to_tensor(cell_to_box(s), dtype=tf.float32) for s in dataset],
        "descriptors": descriptors,
        "gradients": gradients,
        "grad_index": grad_index,
    }

    # Neighbour readout: data-static geometry kernel V_atom [N, D, C, T].
    if neighbor_readout_on(cfg):
        from tqdm import tqdm
        data["V_atom"] = [
            _neighbor_dipole_V(s, t, resolved_dipole_rc(cfg),
                               int(cfg.dipole_channels), int(cfg.num_types),
                               str(cfg.dipole_radial),
                               float(cfg.dipole_zeta),
                               resolved_dipole_cutoff(cfg),
                               float(cfg.dipole_gauss_sigma_scale),
                               rank=bond_rank(cfg),
                               unit=bool(getattr(cfg, "dipole_unit_vector", False)))
            for s, t in tqdm(list(zip(dataset, types_int)),
                             desc="Building neighbor-readout V", unit="struct",
                             disable=len(dataset) < 50)]
        if bool(getattr(cfg, "dipole_field", False)):
            if cfg.target_mode != 1:
                raise ValueError("dipole_field is a dipole (target_mode=1) channel.")
            Es = [field_channel(s, cfg) for s in dataset]
            if cfg.dipole_field_scale is None:
                rms = lambda xs: float(np.sqrt(np.mean([np.mean(x ** 2) for x in xs])))
                cfg.dipole_field_scale = rms([V.sum(axis=3) for V in data["V_atom"]]) / rms(Es)
                print(f"  dipole_field: dipole_field_scale = {cfg.dipole_field_scale:.4g} "
                      f"(RMS(Σ_t' V)/RMS(E), pinned on cfg)")
            data["V_atom"] = [_append_field_channel(V, E, float(cfg.dipole_field_scale))
                              for V, E in zip(data["V_atom"], Es)]

    # nep_raw [N, W_raw]: GPUMD NEP per-atom basis (contracted in-model by c_r/c_a);
    # nep_pairs: image-exact pair stream when ∂q/∂r is needed. One ase pair list per frame.
    if cfg.nep_descriptor != "off":
        need_pairs = nep_pairs_needed(cfg)
        data["nep_raw"] = []
        if need_pairs:
            data["nep_pairs"] = []
        for s, t in zip(dataset, types_int):
            p = nep_pair_list(s, cfg)
            data["nep_raw"].append(nep_raw_numpy(s, t, cfg, p))
            if need_pairs:
                data["nep_pairs"].append(nep_pairs_numpy(s, cfg, p))
        if need_pairs:
            # Padding-pair displacement (gather_nep_pairs): beyond every cutoff → fc = 0.
            data["nep_pad_D"] = np.array(
                [max(float(cfg.nep_rc_radial), float(cfg.nep_rc_angular)) + 1.0, 0.0, 0.0],
                dtype=np.float32)

    # Auto-detect and include force/virial targets for PES mode
    if cfg.target_mode == 0:
        if _dataset_has_key(dataset, _get_forces):
            data["forces"] = [tf.convert_to_tensor(_get_forces(s), dtype=tf.float32)
                              for s in dataset]
            print("  PES mode: forces detected in dataset")
        if _dataset_has_key(dataset, _get_virial):
            data["virials"] = [tf.convert_to_tensor(_get_virial(s), dtype=tf.float32)
                               for s in dataset]
            print("  PES mode: virials detected in dataset")

    return data


def dipole_self_pairs_only(cfg: TNEPconfig) -> bool:
    """Batches need only self gradients: no SOAP gradients read, or dipole_rij_power=0."""
    return not soap_gradients_needed(cfg) or (
        cfg.target_mode == 1 and int(cfg.dipole_rij_power) == 0)


def descriptor_gradients_needed(cfg: TNEPconfig) -> bool:
    """False only for the neighbour readout, which reads no ∂q/∂r. Gotcha: that
    readout still needs pair_atom/pair_gidx on the trajectory path (V built
    in-graph), which keeps calc_gradients=True and skips only the values."""
    return not neighbor_readout_on(cfg)


def soap_gradients_needed(cfg: TNEPconfig) -> bool:
    """SOAP builder must emit ∂q/∂r: gradients needed AND the SOAP block reaches the ANN."""
    return descriptor_gradients_needed(cfg) and cfg.nep_descriptor != "replace"


def nep_pairs_needed(cfg: TNEPconfig) -> bool:
    """Image-exact NEP pair stream is built: tnep dipole readout with NEP in the ANN."""
    return (cfg.nep_descriptor in ("concat", "replace")
            and cfg.target_mode == 1 and cfg.dipole_readout == "tnep")


def _neighbor_pairs(atoms: Atoms, rc: float):
    """(i, j, D) with D = r_j − r_i for pairs within rc (ASE neighbor_list("ijD")
    contract). Orthorhombic cells with periodic L ≥ 2·rc take a direct
    minimum-image path (ASE bins the whole box: ~100 ms per molecule in a
    100 Å vacuum cell); anything else falls back to ASE."""
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    pbc = np.asarray(atoms.pbc, dtype=bool)
    L = np.diag(cell)
    orthorhombic = np.allclose(cell, np.diag(L)) and np.all(L > 0)
    if not (orthorhombic and np.all(L[pbc] >= 2.0 * rc)):
        return neighbor_list("ijD", atoms, rc)
    pos = np.asarray(atoms.positions, dtype=np.float64)
    if len(pos) > 128:
        # Periodic k-d tree: O(N) instead of N² (12 → 1 ms at 648 atoms);
        # non-periodic axes get a huge box so nothing wraps.
        from scipy.spatial import cKDTree
        box = np.where(pbc, L, 1e6)
        wrapped = np.where(pbc, pos % L, pos - pos.min(0) + 1.0)
        p = cKDTree(wrapped, boxsize=box).query_pairs(rc, output_type="ndarray")
        i = np.concatenate([p[:, 0], p[:, 1]]); j = np.concatenate([p[:, 1], p[:, 0]])
        D = pos[j] - pos[i]
        for a in np.flatnonzero(pbc):
            D[:, a] -= L[a] * np.round(D[:, a] / L[a])
        return i, j, D
    D = pos[None, :, :] - pos[:, None, :]                       # [N, N, 3] r_j − r_i
    for a in np.flatnonzero(pbc):
        D[..., a] -= L[a] * np.round(D[..., a] / L[a])          # minimum image
    r2 = np.einsum('ijk,ijk->ij', D, D)
    np.fill_diagonal(r2, np.inf)                               # no self pair
    i, j = np.nonzero(r2 < rc * rc)
    return i, j, D[i, j]


def bond_geometry(d: np.ndarray, rank: int, unit: bool = False) -> np.ndarray:
    """Bond tensors [P, 3] → [P, D, G]. rank 1: d (D=3, G=1). rank 2: Voigt
    [xx,yy,zz,xy,yz,zx], group 0 isotropic |d|²·I, group 1 traceless
    d⊗d − |d|²I/3 (independent α⊥ / α∥−α⊥ coefficients per bond).
    unit=True uses d/|d|. Numpy twin of TNEP._bond_geometry."""
    if unit:
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
    if rank == 1:
        return d[:, :, None]
    if rank == 2:
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        r2 = x * x + y * y + z * z
        iso = np.stack([r2, r2, r2, 0 * r2, 0 * r2, 0 * r2], axis=1)
        tr = np.stack([x * x, y * y, z * z, x * y, y * z, z * x], axis=1) - iso / 3.0
        return np.stack([iso, tr], axis=2)
    raise ValueError(f"bond rank must be 1 or 2, got {rank}")


def _radial_weights(r: np.ndarray, rc: float, C: int, radial: str, zeta: float,
                    cutoff: str, gauss_sigma_scale: float) -> np.ndarray:
    """R_n(r)·f_c(r) → [P, C] (screened: [P, 1]); numpy twin of TNEP._radial/_fc."""
    if cutoff == "cosine2":
        fc = ((1.0 + np.cos(np.pi * r / rc)) / 2.0) ** 2
    elif cutoff == "cosine":
        fc = (1.0 + np.cos(np.pi * r / rc)) / 2.0
    else:
        raise ValueError(f"unknown dipole_cutoff {cutoff!r}")
    if radial == "screened":
        Rw = (r ** (zeta - 1.0) * fc)[:, None]
    elif radial in ("chebyshev", "chebyshev_warped"):
        xc = (2.0 * (r / rc - 1.0) ** 2 - 1.0 if radial == "chebyshev_warped"
              else 2.0 * r / rc - 1.0)
        Tn = np.polynomial.chebyshev.chebvander(xc, C - 1)          # [P, C]
        Rw = Tn * fc[:, None]
    elif radial == "gaussian":
        mu = np.linspace(0.0, rc, C + 2)[1:-1]
        sigma = gauss_sigma_scale * rc / (C + 1)
        Rw = np.exp(-(r[:, None] - mu) ** 2 / (2.0 * sigma ** 2)) * fc[:, None]
    elif radial == "bessel":
        # spherical Bessel j₀ family: sin((n+1)πr/rc)/(r/rc), finite at r→0
        n1 = np.arange(1, C + 1)
        Rw = np.sin(np.pi * n1[None, :] * r[:, None] / rc) / (r[:, None] / rc) * fc[:, None]
    else:
        raise ValueError(f"unknown dipole_radial {radial!r}")
    return Rw


def _neighbor_dipole_pairs(atoms: Atoms, rc: float, C: int, radial: str,
                           zeta: float, cutoff: str, gauss_sigma_scale: float,
                           rank: int = 1, unit: bool = False):
    """Neighbour pairs (i, j, G) with G[p] = R_n(|d|)·g(d) [P, D, G·C], channel
    = group·C + n (groups/D per bond_geometry), d = r_i − r_j, image pairs
    with self-images dropped. R_n·f_c per _radial_weights. float64; callers cast."""
    i, j, D = _neighbor_pairs(atoms, rc)
    keep = i != j                                  # drop self-image pairs
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)  # ase D = r_j − r_i
    r = np.linalg.norm(d, axis=1)
    Rw = _radial_weights(r, rc, C, radial, zeta, cutoff, gauss_sigma_scale)
    g = bond_geometry(d, rank, unit)                          # [P, D, G]
    P_, D_, G_ = g.shape
    return i, j, np.einsum("pdg,pn->pdgn", g, Rw).reshape(P_, D_, G_ * Rw.shape[1])


def ewald_field(atoms: Atoms, q: np.ndarray, alpha: float = 0.35,
                r_cut: float = 9.0, k_cut: float = 7.0) -> np.ndarray:
    """Ewald field E_i [N, 3] (q/Å²) at each atom from all OTHER charges q;
    tin-foil boundary, k = 0 dropped (neutral cells). α-independent to 1e-5."""
    from scipy.special import erfc
    N, r = len(atoms), atoms.positions
    i, j, d, D = neighbor_list("ijdD", atoms, r_cut)          # D = r_j − r_i
    g = (erfc(alpha * d) / d ** 2 + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * d) ** 2) / d) / d
    E = np.zeros((N, 3)); np.add.at(E, i, -(q[j] * g)[:, None] * D)
    cell = np.asarray(atoms.cell.array); V = abs(np.linalg.det(cell))
    recip = 2 * np.pi * np.linalg.inv(cell).T
    nmax = int(np.ceil(k_cut / np.min(np.linalg.norm(recip, axis=1)))) + 1
    rng_ = np.arange(-nmax, nmax + 1)
    n = np.array(np.meshgrid(rng_, rng_, rng_, indexing="ij")).reshape(3, -1).T
    k = n @ recip; k2 = (k ** 2).sum(1)
    keep = (k2 > 1e-12) & (k2 <= k_cut ** 2); k, k2 = k[keep], k2[keep]
    phase = r @ k.T
    S = (q[:, None] * np.exp(1j * phase)).sum(0)
    w = 4 * np.pi / V * np.exp(-k2 / (4 * alpha ** 2)) / k2
    E -= ((w * (S * np.exp(-1j * phase))).imag[:, :, None] * k[None]).sum(1)
    return E


def field_channel(atoms: Atoms, cfg) -> np.ndarray:
    """E_i [N, 3] of cfg.dipole_field_charges (per species, e) by Ewald."""
    ch = cfg.dipole_field_charges
    if not ch:
        raise ValueError("dipole_field=True needs cfg.dipole_field_charges "
                         "({Z: charge in e} per species, neutral frames).")
    # JSON round-trips (model_io) turn int keys into str: accept both.
    q = np.array([float(ch[int(z)] if int(z) in ch else ch[str(int(z))]) for z in atoms.numbers])
    if abs(q.sum()) > 1e-6 * len(atoms):
        raise ValueError(f"dipole_field: frame has net reference charge {q.sum():+.3g}.")
    return ewald_field(atoms, q)


def _append_field_channel(V: np.ndarray, E: np.ndarray, scale: float) -> np.ndarray:
    """V [N, 3, Ck, T] + E [N, 3] → [N, 3, Ck+1, T]: scale·E in the t′=0 slot of a
    new last channel, so its species-pair coefficient acts as a per-centre scale."""
    extra = np.zeros(V.shape[:2] + (1, V.shape[3]), dtype=V.dtype)
    extra[:, :, 0, 0] = scale * E
    return np.concatenate([V, extra], axis=2)


def _neighbor_dipole_V(atoms: Atoms, types_int: np.ndarray, rc: float,
                       C: int, T: int, radial: str = "chebyshev",
                       zeta: float = -2.0,
                       cutoff: str = "cosine2",
                       gauss_sigma_scale: float = 1.0,
                       rank: int = 1, unit: bool = False) -> np.ndarray:
    """Geometry kernel V_i[:, k, t'] = Σ_{j≠i, |d|<rc, t_j=t'} R_n(|d|)·g_group(d),
    k = group·C + n; float64 accumulate → float32 [N, D, G·C, T]."""
    i, j, G = _neighbor_dipole_pairs(atoms, rc, C, radial, zeta, cutoff,
                                     gauss_sigma_scale, rank, unit)
    D, Ck = G.shape[1], G.shape[2]
    # sparse matmul scatter-add (np.add.at ~10× slower on 30k pairs)
    from scipy.sparse import csr_matrix
    N, P = len(atoms), len(i)
    if P == 0:
        return np.zeros((N, D, Ck, T), dtype=np.float32)
    S = csr_matrix((np.ones(P), (i * T + np.asarray(types_int)[j], np.arange(P))), shape=(N * T, P))
    V = np.asarray(S @ G.reshape(P, -1)).reshape(N, T, D, Ck)
    return V.transpose(0, 2, 3, 1).astype(np.float32)      # [N, D, G·C, T]


def _neighbor_dipole_guards(dataset: list[Atoms], targets: list,
                            cfg: TNEPconfig) -> None:
    """Warnings for the neighbour dipole readout: readout r_c vs box height /
    rcut_hard, net-charged frames, label gaps ≳ e·L (branch-inconsistent Berry
    phases). Warnings only: the staged V is image-exact at any radius."""
    rc, rd = resolved_dipole_rc(cfg), float(cfg.rcut_hard)
    L_min = np.inf
    for k, s in enumerate(dataset):
        if s.pbc.any():
            # Face-to-face cell heights: 1/‖column k of inv(cell)‖.
            h = 1.0 / np.linalg.norm(np.linalg.inv(s.cell[:]), axis=0)
            L = float(h[s.pbc].min())
            L_min = min(L_min, L)
        q = s.arrays.get("charges", s.arrays.get("initial_charges"))
        if q is not None and abs(float(np.sum(q))) > 1e-6 * len(s):
            print(f"  WARNING: structure {k} has net charge "
                  f"{float(np.sum(q)):+.3g}; the origin-free neighbor "
                  f"readout cannot fit its label convention.")
    if np.isfinite(L_min):
        if rc >= L_min:
            print(f"  WARNING: readout r_c={rc} Å ≥ the smallest box height "
                  f"{L_min:.2f} Å, so every neighbour enters through several "
                  f"periodic images and the readout is effectively a lattice "
                  f"sum rather than a local one. Still EXACT: images are "
                  f"enumerated, not minimum-imaged, and the self-image terms "
                  f"the readout drops (i == j) cancel identically — for every "
                  f"lattice vector a in range there is −a, and R_n depends on "
                  f"|a| only. GPUMD keeps those terms because its descriptor "
                  f"is a scalar sum, where they do not cancel; our NEP basis "
                  f"keeps them too (nep_descriptor.nep_pair_list).")
        elif 2.0 * rc >= L_min:
            print(f"  WARNING: readout r_c={rc} Å ≥ half the smallest box height "
                  f"{L_min:.2f} Å. The staged V is image-exact and fine, but "
                  f"the minimum-image paths — the in-graph V fallback and "
                  f"TNEP.predict — double-count bonds whose second image maps "
                  f"onto the same vector. Use the staged V (training and "
                  f"score do); treat predict()/spectroscopy output with care.")
        if rc > rd:
            print(f"  WARNING: readout r_c={rc} Å > rcut_hard={rd} Å, so a "
                  f"readout neighbour can lie outside its partner's SOAP "
                  f"descriptor. Intended under nep_descriptor='replace', "
                  f"where the SOAP block is dropped; the in-graph V fallback "
                  f"cannot serve this radius and will refuse.")
    if np.isfinite(L_min) and len(targets) > 3:
        conv = getattr(cfg, "convert_dipole_to_eangstrom", True)
        quantum = (L_min if conv
                   else L_min / _dipole_conversion_factor(cfg.dipole_units))
        t = np.stack([np.asarray(x) for x in targets])       # [S, 3]
        if float(np.max(np.diff(np.sort(t, axis=0), axis=0))) > 0.6 * quantum:
            print("  WARNING: dipole labels show per-component gaps ≳ e·L — "
                  "possibly branch-inconsistent Berry-phase labels.")


def prepare_eval_data(dataset: list[Atoms], cfg: TNEPconfig) -> dict[str, tf.Tensor]:
    """assign_type_indices → build_descriptors → assemble_data_dict → pad_and_stack."""
    types_int = assign_type_indices(dataset, cfg.types)
    builder = make_descriptor_builder(cfg)
    descriptors, gradients, grad_index = builder.build_descriptors(
        dataset, calc_gradients=soap_gradients_needed(cfg))
    data = assemble_data_dict(dataset, types_int, descriptors, gradients, grad_index, cfg)
    return pad_and_stack(
        data,
        num_types=cfg.num_types,
        self_pairs_only=dipole_self_pairs_only(cfg))


def split(dataset: list[Atoms], dataset_types_int: list[np.ndarray], cfg: TNEPconfig) -> tuple[dict, dict, dict]:
    """Split into train/test/val (cfg.indices, cfg.test_ratio) and build descriptors.

    Returns (train_data, test_pending, val_data): list-form dicts per
    assemble_data_dict; test descriptors are deferred to materialize_test_data.
    An external cfg.test_data_path turns the internal split into train+val only.
    """

    indices = cfg.indices
    n_structures = len(indices)

    builder = make_descriptor_builder(cfg)

    gpumd_path = bool(getattr(cfg, "gpumd_path", False))
    if gpumd_path and cfg.test_data_path is None:
        raise ValueError(
            "cfg.gpumd_path needs cfg.test_data_path: it trains on every frame "
            "of data_path and uses the external test set as the report-only "
            "monitor, which is GPUMD's procedure.")
    if cfg.test_data_path is not None:
        # GPUMD path: no val split, every frame trains.
        n_val = 0 if gpumd_path else int(cfg.test_ratio * n_structures)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        val_dataset = [dataset[i] for i in val_idx]
        train_dataset = [dataset[i] for i in train_idx]
        val_types_int = [dataset_types_int[i] for i in val_idx]
        train_types_int = [dataset_types_int[i] for i in train_idx]

        # Load and prepare external test set
        test_structures = read(cfg.test_data_path, index=":")
        if cfg.allowed_species is not None:
            # Filter before type assignment — test file may contain unknown species
            from ase.data import atomic_numbers
            allowed = set(atomic_numbers[z] if isinstance(z, str) else z
                          for z in cfg.allowed_species)
            if cfg.filter_mode == "exact":
                test_structures = [s for s in test_structures
                                   if set(s.numbers) == allowed]
            else:
                test_structures = [s for s in test_structures
                                   if set(s.numbers).issubset(allowed)]
        test_dataset = test_structures
        test_types_int = assign_type_indices(test_dataset, cfg.types)
        print(f"External test set: {len(test_dataset)} structures from {cfg.test_data_path}")
        if gpumd_path:
            # Monitor set IS the test set (report only; SNES.fit never selects on it).
            val_dataset, val_types_int = test_dataset, test_types_int
    else:
        # Default: three-way split from data_path
        n_test = int(cfg.test_ratio * n_structures)
        test_idx = indices[:n_test]
        val_idx = indices[n_test:(2*n_test)]
        train_idx = indices[(2*n_test):n_structures]

        test_dataset = [dataset[i] for i in test_idx]
        val_dataset = [dataset[i] for i in val_idx]
        train_dataset = [dataset[i] for i in train_idx]
        test_types_int = [dataset_types_int[i] for i in test_idx]
        val_types_int = [dataset_types_int[i] for i in val_idx]
        train_types_int = [dataset_types_int[i] for i in train_idx]

    # Self-only forward (dipole_rij_power=0 / neighbour readout): build in batches
    # keeping only ∂q_i/∂r_i, dropping the ~90 % of COO rows never read.
    _self_only_batch = (int(cfg.descriptor_self_batch_size)
                        if getattr(cfg, "descriptor_self_batch_size", None)
                        is not None else None)
    _self_only = dipole_self_pairs_only(cfg)
    if _self_only:
        _kw = {"progress_desc": "Building train descriptors (self-only, batched)"} \
            if cfg.descriptor_mode == 1 else {}
        train_descriptors, train_gradients, train_grad_index = \
            builder.build_descriptors_self_only(
                train_dataset, batch_size=_self_only_batch, **_kw)
        _kw = {"progress_desc": "Building val descriptors (self-only, batched)"} \
            if cfg.descriptor_mode == 1 else {}
        val_descriptors, val_gradients, val_grad_index = \
            builder.build_descriptors_self_only(
                val_dataset, batch_size=_self_only_batch, **_kw)
    else:
        _kw = {"progress_desc": "Building train descriptors"} if cfg.descriptor_mode == 1 else {}
        train_descriptors, train_gradients, train_grad_index = builder.build_descriptors(train_dataset, **_kw)
        _kw = {"progress_desc": "Building val descriptors"} if cfg.descriptor_mode == 1 else {}
        val_descriptors,   val_gradients,   val_grad_index   = builder.build_descriptors(val_dataset, **_kw)

    train_data = assemble_data_dict(train_dataset, train_types_int, train_descriptors, train_gradients, train_grad_index, cfg)
    val_data   = assemble_data_dict(val_dataset,   val_types_int,   val_descriptors,   val_gradients,   val_grad_index,   cfg)
    test_pending = {
        "_pending_test": True,
        "dataset": test_dataset,
        "types_int": test_types_int,
    }
    n_train = len(train_data["positions"])
    n_test  = len(test_dataset)
    n_val   = len(val_data["positions"])
    if gpumd_path:
        print(f"GPUMD path: all {n_train} structures train (no val split); "
              f"the {n_val}-structure monitor set IS the test set, reported "
              f"every val_interval generations and never selected on.")
    elif cfg.test_data_path is not None:
        print(f"{n_structures} structures split into train ({n_train}) + val ({n_val})")
        print(f"External test set: {n_test} structures from {cfg.test_data_path} (descriptors deferred to scoring)")
    else:
        print(f"{n_structures} structures split into train ({n_train}) + test ({n_test}) + val ({n_val}) "
              f"(test descriptors deferred to scoring)")
    return train_data, test_pending, val_data


def materialize_test_data(test_pending: dict, cfg: 'TNEPconfig',
                          num_types: int | None = None,
                          pin_to_cpu: bool | None = None) -> dict:
    """Build the deferred test set from split() into a padded, scorable dict.
    Idempotent (cached under test_pending["_built"]); num_types/pin_to_cpu
    default to cfg."""
    if not test_pending.get("_pending_test", False):
        return test_pending  # already materialised or never deferred
    cached = test_pending.get("_built")
    if cached is not None:
        return cached

    if num_types is None:
        num_types = cfg.num_types
    if pin_to_cpu is None:
        pin_to_cpu = cfg.pin_data_to_cpu

    test_dataset = test_pending["dataset"]
    test_types_int = test_pending["types_int"]

    builder = make_descriptor_builder(cfg)
    _kw = {"progress_desc": "Building test descriptors"} if cfg.descriptor_mode == 1 else {}
    test_descriptors, test_gradients, test_grad_index = builder.build_descriptors(
        test_dataset, calc_gradients=soap_gradients_needed(cfg), **_kw)

    test_data = assemble_data_dict(
        test_dataset, test_types_int,
        test_descriptors, test_gradients, test_grad_index, cfg)
    test_data = pad_and_stack(
        test_data, num_types=num_types, pin_to_cpu=pin_to_cpu,
        self_pairs_only=dipole_self_pairs_only(cfg))
    # Pre-stage per-chunk pair indices (predict_batch path: no XLA padding).
    S_test = int(test_data["num_atoms"].shape[0])
    chunk = cfg.batch_chunk_size if cfg.batch_chunk_size is not None else S_test
    test_ranges = [(s, min(s + chunk, S_test)) for s in range(0, S_test, chunk)]
    prestage_chunk_indices(test_data, test_ranges)
    test_pending["_built"] = test_data
    test_pending["dataset"] = None
    test_pending["types_int"] = None
    return test_data


def pad_and_stack(data: dict, num_types: int | None = None,
                  pin_to_cpu: bool = True,
                  self_pairs_only: bool = False) -> dict[str, tf.Tensor]:
    """List-form dict (assemble_data_dict) → padded tensors + COO gradients.

    COO (real pairs only) avoids the dense O(S·A·M·3·Q) gradient allocation.
    self_pairs_only keeps one zero-image self entry per centre.

    Returns, with S structures, A = max atoms, Q = dim_q, P = total pairs:
        descriptors [S, A, Q]; positions [S, A, 3]; Z_int [S, A]; targets [S, T];
        boxes [S, 3, 3]; atom_mask [S, A] (1 = real); num_atoms [S];
        grad_values [P, 3, Q]; pair_struct / pair_atom / pair_gidx [P] int32;
        struct_ptr [S+1] CSR (struct s owns grad_values[ptr[s]:ptr[s+1]]);
        types_contained [S, num_types] when num_types given;
        forces [S, A, 3] / virials [S, 6] / V_atom [S, A, ...] / nep_raw [S, A, W] when present;
        nep_pair_atom/gidx [P'], nep_pair_D [P', 3], nep_pair_ang [P'] bool,
        nep_pair_struct [P'], nep_struct_ptr [S+1], nep_pad_D [3] when nep_pairs present.
    """
    S = len(data["descriptors"])
    dim_q = data["descriptors"][0].shape[-1]
    atom_counts = [data["descriptors"][i].shape[0] for i in range(S)]
    max_atoms = max(atom_counts)

    target_sample = data["targets"][0]
    target_dim = 1 if target_sample.shape == () else target_sample.shape[0]
    has_forces = "forces" in data
    has_virials = "virials" in data
    has_V = "V_atom" in data

    # Pairs per structure → CSR struct_ptr + COO sizes.
    if self_pairs_only:
        # One (zero-image) self entry per centre; periodic self-images would double-count.
        pair_counts = [
            int(sum(1 for i in range(atom_counts[s])
                    if bool(np.any(np.asarray(data["grad_index"][s][i]) == i))))
            for s in range(S)
        ]
    else:
        pair_counts = [
            sum(data["gradients"][s][i].shape[0] for i in range(atom_counts[s]))
            for s in range(S)
        ]
    N_pairs_total = sum(pair_counts)
    struct_ptr_np = np.zeros(S + 1, dtype=np.int32)
    for s in range(S):
        struct_ptr_np[s + 1] = struct_ptr_np[s] + pair_counts[s]

    grad_values_np = np.zeros((N_pairs_total, 3, dim_q), dtype=np.float32)
    pair_struct_np = np.zeros(N_pairs_total, dtype=np.int32)
    pair_atom_np   = np.zeros(N_pairs_total, dtype=np.int32)
    pair_gidx_np   = np.zeros(N_pairs_total, dtype=np.int32)

    desc_np      = np.zeros((S, max_atoms, dim_q), dtype=np.float32)
    pos_np       = np.zeros((S, max_atoms, 3), dtype=np.float32)
    z_np         = np.zeros((S, max_atoms), dtype=np.int32)
    tgt_np       = np.zeros((S, target_dim), dtype=np.float32)
    box_np       = np.zeros((S, 3, 3), dtype=np.float32)
    atom_mask_np = np.zeros((S, max_atoms), dtype=np.float32)
    num_atoms_np = np.array(atom_counts, dtype=np.int32)
    if has_forces:
        force_np = np.zeros((S, max_atoms, 3), dtype=np.float32)
    if has_virials:
        virial_np = np.zeros((S, 6), dtype=np.float32)
    if has_V:
        V_np = np.zeros((S, max_atoms) + data["V_atom"][0].shape[1:],
                        dtype=np.float32)
    has_nep = "nep_raw" in data
    if has_nep:
        nep_np = np.zeros((S, max_atoms, data["nep_raw"][0].shape[1]), dtype=np.float32)
    has_nep_pairs = "nep_pairs" in data  # concatenated, CSR by nep_struct_ptr
    if has_nep_pairs:
        nps = data["nep_pairs"]
        counts = np.array([p["nep_pair_atom"].shape[0] for p in nps], dtype=np.int32)
        nep_ptr_np = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        nep_struct_np = np.repeat(np.arange(S, dtype=np.int32), counts)
        nep_cat = {k: np.concatenate([p[k] for p in nps]) for k in _NEP_PAIR_KEYS}
    if num_types is not None:
        types_contained_np = np.zeros((S, num_types), dtype=np.float32)

    pair_offset = 0
    for s in range(S):
        N_s = atom_counts[s]
        d_s = data["descriptors"][s]  # ndarray or TF tensor
        desc_np[s, :N_s, :]  = d_s if isinstance(d_s, np.ndarray) else d_s.numpy()
        pos_np[s, :N_s, :]   = data["positions"][s].numpy()
        z_np[s, :N_s]        = data["Z_int"][s].numpy()
        if num_types is not None:
            z_vals = z_np[s, :N_s]
            for t in range(num_types):
                if np.any(z_vals == t):
                    types_contained_np[s, t] = 1.0
        box_np[s] = data["boxes"][s].numpy()
        atom_mask_np[s, :N_s] = 1.0

        t = data["targets"][s]
        if t.shape == ():
            tgt_np[s, 0] = t.numpy()
        else:
            tgt_np[s, :] = t.numpy()

        if has_forces:
            force_np[s, :N_s, :] = data["forces"][s].numpy()
        if has_virials:
            virial_np[s, :] = data["virials"][s].numpy()
        if has_V:
            V_np[s, :N_s] = data["V_atom"][s]
        if has_nep:
            nep_np[s, :N_s] = data["nep_raw"][s]

        for i in range(N_s):
            gv_full = data["gradients"][s][i]
            gidx_full = np.asarray(data["grad_index"][s][i])
            if self_pairs_only:
                # First self row only (gidx == i): dipole = Σ_i de_dq[i]·grad[i,i].
                mask = (gidx_full == i)
                if not bool(np.any(mask)):
                    continue
                k0 = int(np.argmax(mask))  # first True index
                gv_arr = (gv_full.numpy()
                          if hasattr(gv_full, "numpy") else np.asarray(gv_full))
                grad_values_np[pair_offset] = gv_arr[k0]
                pair_struct_np[pair_offset] = s
                pair_atom_np[pair_offset]   = i
                pair_gidx_np[pair_offset]   = int(gidx_full[k0])
                pair_offset += 1
            else:
                n_nbrs = gv_full.shape[0]
                k_end  = pair_offset + n_nbrs
                grad_values_np[pair_offset:k_end] = (
                    gv_full.numpy() if hasattr(gv_full, "numpy")
                    else np.asarray(gv_full))
                pair_struct_np[pair_offset:k_end] = s
                pair_atom_np[pair_offset:k_end]   = i
                pair_gidx_np[pair_offset:k_end]   = gidx_full
                pair_offset += n_nbrs

    # Convert then delete each numpy copy so peak RAM stays ~1x. pin_to_cpu → CPU, else GPU.
    _dev_ctx = (tf.device('/CPU:0') if pin_to_cpu
                else _gpu_device_ctx())
    with _dev_ctx:
        result = {}
        result["descriptors"] = tf.constant(desc_np);    del desc_np
        result["grad_values"] = tf.constant(grad_values_np); del grad_values_np
        result["pair_struct"] = tf.constant(pair_struct_np); del pair_struct_np
        result["pair_atom"]   = tf.constant(pair_atom_np);   del pair_atom_np
        result["pair_gidx"]   = tf.constant(pair_gidx_np);   del pair_gidx_np
        result["struct_ptr"]  = tf.constant(struct_ptr_np);  del struct_ptr_np
        result["positions"]   = tf.constant(pos_np);         del pos_np
        result["Z_int"]       = tf.constant(z_np);           del z_np
        result["targets"]     = tf.constant(tgt_np);         del tgt_np
        result["boxes"]       = tf.constant(box_np);         del box_np
        result["atom_mask"]   = tf.constant(atom_mask_np);   del atom_mask_np
        result["num_atoms"]   = tf.constant(num_atoms_np);   del num_atoms_np
        if has_forces:
            result["forces"]  = tf.constant(force_np);       del force_np
        if has_virials:
            result["virials"] = tf.constant(virial_np);      del virial_np
        if has_V:
            result["V_atom"]  = tf.constant(V_np);           del V_np
        if has_nep:
            result["nep_raw"]   = tf.constant(nep_np);       del nep_np
        if has_nep_pairs:
            result["nep_struct_ptr"]  = tf.constant(nep_ptr_np)
            result["nep_pair_struct"] = tf.constant(nep_struct_np)
            result["nep_pad_D"]       = tf.constant(data["nep_pad_D"])
            for k in _NEP_PAIR_KEYS:
                result[k] = tf.constant(nep_cat[k])
            del nep_cat
        if num_types is not None:
            result["types_contained"] = tf.constant(types_contained_np); del types_contained_np
    return result


def pack_chunk_from_flat(frame_results: list, dim_q: int,
                          max_atoms: int | None = None) -> dict:
    """Per-frame (soap [N, Q], grad [P, 3, Q], pair_atom [P], pair_gidx [P]) TF
    tensors from build_descriptors_flat → chunk dict {descriptors [B, A, Q],
    grad_values [P, 3, Q], pair_atom/pair_gidx (frame-local), pair_struct
    (chunk-local)} for _evaluate_chunk. max_atoms=None pads to the chunk's own
    max; pass the dataset-wide A_max to match positions/Z_int. Clears frame_results."""
    S = len(frame_results)
    atom_counts = [int(r[0].shape[0]) for r in frame_results]
    chunk_max_atoms = max(atom_counts) if atom_counts else 0
    if max_atoms is None:
        max_atoms = chunk_max_atoms
    elif max_atoms < chunk_max_atoms:
        raise ValueError(
            f"pack_chunk_from_flat: max_atoms={max_atoms} is smaller than the "
            f"chunk's own max(atom_counts)={chunk_max_atoms}.")
    pair_counts = [int(r[1].shape[0]) for r in frame_results]
    pair_counts_arr = np.array(pair_counts, dtype=np.int32)
    N_pairs = int(pair_counts_arr.sum())

    if N_pairs > 0:
        grad_values_t = tf.concat([r[1] for r in frame_results], axis=0)
        pair_atom_t   = tf.concat([r[2] for r in frame_results], axis=0)
        pair_gidx_t   = tf.concat([r[3] for r in frame_results], axis=0)
        pair_struct_t = tf.repeat(tf.range(S, dtype=tf.int32),
                                   tf.constant(pair_counts_arr, dtype=tf.int32))
    else:
        grad_values_t = tf.zeros((0, 3, dim_q), dtype=tf.float32)
        pair_atom_t   = tf.zeros((0,), dtype=tf.int32)
        pair_gidx_t   = tf.zeros((0,), dtype=tf.int32)
        pair_struct_t = tf.zeros((0,), dtype=tf.int32)

    if S:
        soap_concat = tf.concat([r[0] for r in frame_results], axis=0)
        frame_results.clear()
        desc_ragged = tf.RaggedTensor.from_row_lengths(
            soap_concat, tf.constant(atom_counts, dtype=tf.int64))
        desc_t = desc_ragged.to_tensor(default_value=0.0,
                                        shape=(S, max_atoms, dim_q))
        del soap_concat, desc_ragged
    else:
        desc_t = tf.zeros((0, 0, dim_q), dtype=tf.float32)
        frame_results.clear()

    return {
        "descriptors": desc_t,
        "grad_values": grad_values_t,
        "pair_atom":   pair_atom_t,
        "pair_gidx":   pair_gidx_t,
        "pair_struct": pair_struct_t,
    }


def _is_contiguous_range(arr: np.ndarray) -> bool:
    """True iff every diff is +1 (a true slice; the size check alone accepts permutations)."""
    if arr.size == 0:
        return False
    if arr.size == 1:
        return True
    return bool(np.all(np.diff(arr) == 1))


# Structure-padded per-structure fields that ride every chunk.
_RESIDENT_SMALL_KEYS = ("positions", "Z_int", "boxes", "num_atoms",
                         "targets", "atom_mask", "types_contained",
                         "forces", "virials", "V_atom", "nep_raw")


_NEP_PAIR_KEYS = ("nep_pair_atom", "nep_pair_gidx", "nep_pair_D", "nep_pair_ang")


# Significant binary digits kept when bucketing the NEP pair-stream length, so
# the per-shape XLA pair kernels compile ≤ 64 times per octave. Not a power of
# two: that padded a 4.2 M-pair chunk to 2^23 and doubled every per-pair tensor;
# 6 digits bounds the waste at ≤ 3 %.
_PAIR_BUCKET_BITS = 6


def _bucket(n: int, bits: int = _PAIR_BUCKET_BITS) -> int:
    """Smallest multiple of 2^(bitlen(n) − bits) that is ≥ n (n ≥ 0).
    Exact for n ≤ 2^bits, where rounding would cost more than it saves."""
    if n <= (1 << bits):
        return max(n, 1)
    g = 1 << (n.bit_length() - bits)
    return ((n + g - 1) // g) * g


def gather_nep_pairs(data: dict, chunk: dict, idx_tf) -> None:
    """Gather the NEP pair stream for structures idx_tf into chunk (CSR by
    nep_struct_ptr, nep_pair_struct chunk-local); no-op without the stream.
    Eager only. P and the angular count P_a are bucket-padded (_bucket) so the
    per-shape XLA pair kernels compile once per bucket; padding pairs sit at
    D = nep_pad_D beyond every cutoff (fc = 0, contribute nothing), the first
    bucket(P_a) − P_a of them flagged nep_pair_ang. The chunk nep_struct_ptr
    counts real pairs only."""
    ptr = data.get("nep_struct_ptr")
    if ptr is None:
        return
    rng = tf.ragged.range(tf.gather(ptr, idx_tf), tf.gather(ptr, idx_tf + 1))
    flat = tf.cast(rng.flat_values, tf.int32)
    chunk["nep_struct_ptr"] = tf.concat(
        [[0], tf.cumsum(tf.cast(rng.row_lengths(), tf.int32))], axis=0)
    chunk["nep_pad_D"] = data["nep_pad_D"]
    real = {k: tf.gather(data[k], flat) for k in _NEP_PAIR_KEYS}
    real["nep_pair_struct"] = tf.cast(rng.value_rowids(), tf.int32)
    P = int(flat.shape[0])
    P_a = int(tf.reduce_sum(tf.cast(real["nep_pair_ang"], tf.int32)))
    n_ang = _bucket(P_a) - P_a
    n_pad = _bucket(P + n_ang) - P
    zeros = tf.zeros([n_pad], tf.int32)
    pad = dict(nep_pair_atom=zeros, nep_pair_gidx=zeros, nep_pair_struct=zeros,
               nep_pair_D=tf.tile(data["nep_pad_D"][tf.newaxis], [n_pad, 1]),
               nep_pair_ang=tf.concat([tf.ones([n_ang], tf.bool),
                                       tf.zeros([n_pad - n_ang], tf.bool)], 0))
    for k in real:
        chunk[k] = tf.concat([real[k], pad[k]], axis=0)


def slice_and_complete_chunk(data: dict, indices) -> dict:
    """Chunk dict for SNES._evaluate_chunk / TNEP.predict_batch: the
    _RESIDENT_SMALL_KEYS and descriptors gathered on the B axis, grad_values
    [P_chunk, 3, Q] + pair_atom/pair_gidx (frame-local) gathered via
    struct_ptr, pair_struct remapped chunk-local, NEP pair stream likewise."""
    if isinstance(indices, tf.Tensor):
        idx_tf = tf.cast(indices, tf.int32)
    else:
        idx_tf = tf.constant(np.asarray(indices, dtype=np.int32), dtype=tf.int32)

    chunk: dict = {}
    for k in _RESIDENT_SMALL_KEYS:
        if k in data:
            chunk[k] = tf.gather(data[k], idx_tf)

    chunk["descriptors"] = tf.gather(data["descriptors"], idx_tf)
    ptr = data["struct_ptr"]
    pair_starts = tf.gather(ptr, idx_tf)
    pair_ends   = tf.gather(ptr, idx_tf + 1)
    pair_ranges = tf.ragged.range(pair_starts, pair_ends)
    flat_pair_idx_tf = tf.cast(pair_ranges.flat_values, tf.int32)
    chunk["pair_struct"] = tf.cast(pair_ranges.value_rowids(), tf.int32)

    gv = data["grad_values"]
    chunk["grad_values"] = tf.gather(gv, flat_pair_idx_tf)
    chunk["pair_atom"]   = tf.gather(data["pair_atom"],   flat_pair_idx_tf)
    chunk["pair_gidx"]   = tf.gather(data["pair_gidx"],   flat_pair_idx_tf)
    gather_nep_pairs(data, chunk, idx_tf)
    return chunk


# ── chunk staging / prefetch ──

def prestage_chunk_indices(data: dict, ranges: list) -> None:
    """Pre-build GPU pair_atom/pair_gidx/pair_struct per (s, e) chunk into
    data["_pair_idx_gpu_cache"]; saves ~3-5 ms/chunk/gen of tf.constant + DMA."""
    cache: dict = data.setdefault("_pair_idx_gpu_cache", {})
    pa_full = data["pair_atom"]
    pg_full = data["pair_gidx"]
    pa_np = pa_full.numpy() if hasattr(pa_full, "numpy") else np.asarray(pa_full)
    pg_np = pg_full.numpy() if hasattr(pg_full, "numpy") else np.asarray(pg_full)
    idx_cache = get_chunk_index_cache()
    with _gpu_device_ctx():
        for s, e in ranges:
            key = (int(s), int(e))
            if key in cache:
                continue
            precomp = idx_cache.get(data, s, e)
            flat = precomp["flat_pair_idx_np"]
            pair_struct_np = precomp["pair_struct_tf"].numpy()
            cache[key] = {
                "pair_atom":   tf.constant(pa_np[flat]),
                "pair_gidx":   tf.constant(pg_np[flat]),
                "pair_struct": tf.constant(pair_struct_np),
                "_real_P":     int(flat.shape[0]),
            }


class ChunkIndexCache:
    """Per-chunk flat_pair_idx + chunk-local pair_struct, keyed (id(data), s, e)."""
    def __init__(self):
        self._cache: dict = {}

    def get(self, data: dict, s_start: int, s_end: int) -> dict:
        key = (id(data), int(s_start), int(s_end))
        item = self._cache.get(key)
        if item is None:
            idx_tf = tf.range(s_start, s_end, dtype=tf.int32)
            ptr = data["struct_ptr"]
            pair_starts = tf.gather(ptr, idx_tf)
            pair_ends   = tf.gather(ptr, idx_tf + 1)
            pair_ranges = tf.ragged.range(pair_starts, pair_ends)
            flat_pair_idx_tf = tf.cast(pair_ranges.flat_values, tf.int32)
            pair_struct_tf   = tf.cast(pair_ranges.value_rowids(), tf.int32)
            # numpy copy: staging fancy-indexes without a per-call device sync
            flat_pair_idx_np = flat_pair_idx_tf.numpy()
            item = {
                "flat_pair_idx_tf": flat_pair_idx_tf,
                "flat_pair_idx_np": flat_pair_idx_np,
                "pair_struct_tf":   pair_struct_tf,
            }
            self._cache[key] = item
        return item

    def clear(self):
        self._cache.clear()


_chunk_index_cache: ChunkIndexCache | None = None


def get_chunk_index_cache() -> ChunkIndexCache:
    global _chunk_index_cache
    if _chunk_index_cache is None:
        _chunk_index_cache = ChunkIndexCache()
    return _chunk_index_cache


def _stage_disk_only(data: dict, s_start: int, s_end: int) -> dict:
    """Staging phase 1 (numpy): slice per-structure fields; pass grad/pair
    tensors through untouched for _stage_finalize_tf."""
    precomputed = get_chunk_index_cache().get(data, s_start, s_end)
    idx_np = np.arange(int(s_start), int(s_end), dtype=np.int32)
    flat_pair_idx_np = precomputed["flat_pair_idx_np"]

    out: dict = {"_idx_np": idx_np, "_precomputed": precomputed}
    for k in _RESIDENT_SMALL_KEYS:
        if k in data:
            v = data[k]
            np_view = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
            out["_np_" + k] = np_view[idx_np]
    desc = data["descriptors"]
    desc_np = desc.numpy() if hasattr(desc, "numpy") else np.asarray(desc)
    out["_np_descriptors"] = desc_np[idx_np]

    out["_passthrough_grad_values"] = data["grad_values"]
    out["_passthrough_pair_atom"]   = data["pair_atom"]
    out["_passthrough_pair_gidx"]   = data["pair_gidx"]
    return out


def _stage_finalize_tf(data: dict, raw: dict, pin_to_cpu: bool,
                        s_start: int | None = None,
                        s_end: int | None = None) -> dict:
    """Staging phase 2: numpy → tf.constant, reusing data["_pair_idx_gpu_cache"]
    entries when present. pin_to_cpu ⇒ upload each chunk to GPU."""
    chunk: dict = {}
    ctx = _gpu_device_ctx() if pin_to_cpu else _NullCtx()
    pair_idx_cache = data.get("_pair_idx_gpu_cache")
    pair_idx_entry = None
    if pair_idx_cache is not None and s_start is not None and s_end is not None:
        pair_idx_entry = pair_idx_cache.get((int(s_start), int(s_end)))
    with ctx:
        for k in _RESIDENT_SMALL_KEYS:
            np_key = "_np_" + k
            if np_key in raw:
                chunk[k] = tf.constant(raw[np_key])
        chunk["descriptors"] = tf.constant(raw["_np_descriptors"])
        if pair_idx_entry is not None:
            chunk["pair_struct"] = pair_idx_entry["pair_struct"]
        else:
            chunk["pair_struct"] = (tf.identity(raw["_precomputed"]["pair_struct_tf"])
                                     if pin_to_cpu else
                                     raw["_precomputed"]["pair_struct_tf"])
        # Contiguous indices (full-batch) → strided slice; sub-sampling → tf.gather.
        flat_pair_idx_np = raw["_precomputed"]["flat_pair_idx_np"]
        is_contig = _is_contiguous_range(flat_pair_idx_np)
        if is_contig:
            lo = int(flat_pair_idx_np[0])
            hi = int(flat_pair_idx_np[-1]) + 1
            gv = raw["_passthrough_grad_values"]
            chunk["grad_values"] = gv[lo:hi]
            if pair_idx_entry is not None:
                chunk["pair_atom"] = pair_idx_entry["pair_atom"]
                chunk["pair_gidx"] = pair_idx_entry["pair_gidx"]
            else:
                chunk["pair_atom"] = raw["_passthrough_pair_atom"][lo:hi]
                chunk["pair_gidx"] = raw["_passthrough_pair_gidx"][lo:hi]
        else:
            flat_pair_idx_tf = tf.constant(flat_pair_idx_np)
            chunk["grad_values"] = tf.gather(raw["_passthrough_grad_values"], flat_pair_idx_tf)
            chunk["pair_atom"]   = tf.gather(raw["_passthrough_pair_atom"],   flat_pair_idx_tf)
            chunk["pair_gidx"]   = tf.gather(raw["_passthrough_pair_gidx"],   flat_pair_idx_tf)
        gather_nep_pairs(data, chunk, tf.constant(raw["_idx_np"]))
    return chunk


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _gpu_device_ctx():
    """tf.device('/GPU:0') if a GPU is visible, else a no-op context (a bare pin
    raises on CPU-only nodes)."""
    try:
        if tf.config.list_physical_devices('GPU'):
            return tf.device('/GPU:0')
    except Exception:
        pass
    return _NullCtx()



def move_data_to_gpu(data: dict) -> None:
    """Move every static field of `data` onto /GPU:0 (helper `_` keys skipped)."""
    keys = list(_RESIDENT_SMALL_KEYS) + [
        "descriptors", "pair_atom", "pair_gidx", "pair_struct",
        "struct_ptr", "grad_values", "nep_struct_ptr", "nep_pair_struct", "nep_pad_D",
        *_NEP_PAIR_KEYS]
    with _gpu_device_ctx():
        for k in keys:
            v = data.get(k)
            if v is None:
                continue
            if hasattr(v, "device"):
                data[k] = tf.identity(v)
            else:
                data[k] = tf.constant(np.asarray(v))


def _stage_chunk_resident(data: dict, s_start: int, s_end: int) -> dict:
    """On-device chunk staging for `_gv_resident_gpu` dicts: tf.gather on the
    B axis + strided grad_values slice, no host↔device traffic."""
    chunk: dict = {}
    s_lo = int(s_start)
    s_hi = int(s_end)
    pair_idx_cache = data.get("_pair_idx_gpu_cache")
    pair_idx_entry = (pair_idx_cache.get((s_lo, s_hi))
                       if pair_idx_cache is not None else None)
    with _gpu_device_ctx():
        idx_tf = tf.range(s_lo, s_hi, dtype=tf.int32)
        for k in _RESIDENT_SMALL_KEYS:
            v = data.get(k)
            if v is None:
                continue
            chunk[k] = tf.gather(v, idx_tf)
        chunk["descriptors"] = tf.gather(data["descriptors"], idx_tf)

        precomputed = get_chunk_index_cache().get(data, s_lo, s_hi)
        flat_pair_idx_np = precomputed["flat_pair_idx_np"]
        if flat_pair_idx_np.size > 0:
            lo = int(flat_pair_idx_np[0])
            hi = int(flat_pair_idx_np[-1]) + 1
            is_contig = (hi - lo) == flat_pair_idx_np.size
        else:
            lo = hi = 0
            is_contig = True
        gv = data["grad_values"]
        if is_contig:
            grad_slc = gv[lo:hi]
            if pair_idx_entry is not None:
                pair_atom = pair_idx_entry["pair_atom"]
                pair_gidx = pair_idx_entry["pair_gidx"]
            else:
                pair_atom = data["pair_atom"][lo:hi]
                pair_gidx = data["pair_gidx"][lo:hi]
        else:
            flat_tf = tf.constant(flat_pair_idx_np)
            grad_slc = tf.gather(gv, flat_tf)
            pair_atom = tf.gather(data["pair_atom"], flat_tf)
            pair_gidx = tf.gather(data["pair_gidx"], flat_tf)
        if pair_idx_entry is not None:
            pair_struct = pair_idx_entry["pair_struct"]
        else:
            pair_struct = precomputed["pair_struct_tf"]

        chunk["grad_values"] = grad_slc
        chunk["pair_atom"] = pair_atom
        chunk["pair_gidx"] = pair_gidx
        chunk["pair_struct"] = pair_struct
        gather_nep_pairs(data, chunk, idx_tf)
    return chunk


def prefetched_chunks(data: dict, ranges: list, pin_to_cpu: bool):
    """Yield (s, e, chunk) per range: _stage_chunk_resident when
    _gv_resident_gpu, else _stage_disk_only + _stage_finalize_tf."""
    if not ranges:
        return

    if data.get("_gv_resident_gpu", False):
        for s, e in ranges:
            yield int(s), int(e), _stage_chunk_resident(data, int(s), int(e))
        return

    # Pre-warm so staging never hits a TF op on a cache miss.
    idx_cache = get_chunk_index_cache()
    for _s, _e in ranges:
        idx_cache.get(data, _s, _e)

    for s, e in ranges:
        raw = _stage_disk_only(data, s, e)
        yield s, e, _stage_finalize_tf(data, raw, pin_to_cpu=pin_to_cpu,
                                        s_start=s, s_end=e)


def filter_by_species(dataset: list[Atoms], dataset_types_int: list[np.ndarray], allowed_Z: list[int | str], mode: str = "subset") -> tuple[list[Atoms], list[np.ndarray]]:
    """Keep structures whose species are a subset of (mode="subset") or exactly
    (mode="exact") allowed_Z (atomic numbers or symbols)."""
    from ase.data import atomic_numbers
    allowed = set(atomic_numbers[z] if isinstance(z, str) else z for z in allowed_Z)
    filtered_dataset = []
    filtered_types_int = []
    for struct, types_int in zip(dataset, dataset_types_int):
        species = set(struct.numbers)
        if mode == "exact":
            keep = species == allowed
        else:
            keep = species.issubset(allowed)
        if keep:
            filtered_dataset.append(struct)
            filtered_types_int.append(types_int)
    return filtered_dataset, filtered_types_int


def print_dipole_statistics(dataset: list[Atoms], cfg: TNEPconfig,
                            target_key: str = "dipole") -> None:
    """Print min/max/mean/std of the dipole targets (unit-converted per cfg)."""
    dipoles = np.array([_extract_target(s, target_key).numpy() for s in dataset])
    convert = getattr(cfg, "convert_dipole_to_eangstrom", True)
    factor = _dipole_conversion_factor(cfg.dipole_units) if convert else 1.0
    if factor != 1.0:
        dipoles = dipoles * factor
    if convert:
        unit = f"e\u00b7\u00c5 (from {cfg.dipole_units})" if factor != 1.0 else "e\u00b7\u00c5"
    else:
        unit = f"{cfg.dipole_units} (no conversion)"
    norms = np.linalg.norm(dipoles, axis=1)
    print(f"=== Dipole Target Statistics ({unit}) ===")
    print(f"  N structures: {len(dipoles)}")
    print(f"  Component ranges: x=[{dipoles[:,0].min():.4f}, {dipoles[:,0].max():.4f}]  "
          f"y=[{dipoles[:,1].min():.4f}, {dipoles[:,1].max():.4f}]  "
          f"z=[{dipoles[:,2].min():.4f}, {dipoles[:,2].max():.4f}]")
    print(f"  Component means:  x={dipoles[:,0].mean():.4f}  y={dipoles[:,1].mean():.4f}  z={dipoles[:,2].mean():.4f}")
    print(f"  Component stds:   x={dipoles[:,0].std():.4f}  y={dipoles[:,1].std():.4f}  z={dipoles[:,2].std():.4f}")
    print(f"  |μ| range: [{norms.min():.4f}, {norms.max():.4f}]")
    print(f"  |μ| mean:  {norms.mean():.4f}  std: {norms.std():.4f}")


def print_polarizability_statistics(dataset: list[Atoms], target_key: str = "pol") -> None:
    """Print min/max/mean/std of the polarizability targets per Voigt component."""
    pols = np.array([_extract_target(s, target_key).numpy() for s in dataset])
    labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
    print("=== Polarizability Target Statistics ===")
    print(f"  N structures: {len(pols)}")
    for i, lbl in enumerate(labels):
        print(f"  {lbl}: [{pols[:,i].min():.4f}, {pols[:,i].max():.4f}]  "
              f"mean={pols[:,i].mean():.4f}  std={pols[:,i].std():.4f}")
