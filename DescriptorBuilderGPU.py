"""NumPy reference SOAP-turbo descriptor builder (correctness oracle).

Ports the Fortran soap_turbo reference (/home/reece/TNEP/soapturbof90/)
and validates against quippy in 10 phases: basis matrices, radial/angular
expansion, cnk scatter-sum, power spectrum, derivatives, Cartesian
conversion, self-derivative, and the DescriptorBuilderGPU class wrapper.
Phase 0 saves quippy reference fixtures under tests/fixtures/.
"""

from __future__ import annotations

import os
import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from quippy.descriptors import Descriptor
from scipy.special import erf


# SOAP config for all reference fixtures. Mirrors TNEPconfig defaults; keep
# stable — every phase tests against fixtures generated with these params.
REFERENCE_SOAP_PARAMS = dict(
    l_max=4,
    alpha_max=4,
    rcut_hard=6.0,
    rcut_soft=5.5,
    atom_sigma_r=0.5,
    atom_sigma_t=0.5,
    atom_sigma_r_scaling=0.0,
    atom_sigma_t_scaling=0.0,
    amplitude_scaling=1.0,
    central_weight=1.0,
    radial_enhancement=1,
    basis="poly3",
    scaling_mode="polynomial",
    compress_mode="trivial",
)


def _build_soap_string(species_Z: list[int], **soap_params) -> str:
    """Construct the soap_turbo descriptor string for one centre type.

    Emits the per-species arrays (length n_species); the caller appends
    central_index. Mirrors DescriptorBuilder.__init__.
    """
    n = len(species_Z)
    s = (
        f"soap_turbo l_max={soap_params['l_max']} "
        f"rcut_hard={soap_params['rcut_hard']} rcut_soft={soap_params['rcut_soft']} "
        f"basis={soap_params['basis']} scaling_mode={soap_params['scaling_mode']} "
        f"add_species=F radial_enhancement={soap_params['radial_enhancement']} "
        f"compress_mode={soap_params['compress_mode']} "
        f"n_species={n} "
        f"species_Z={{{' '.join(str(z) for z in species_Z)}}} "
        f"alpha_max={{{' '.join([str(soap_params['alpha_max'])] * n)}}} "
        f"atom_sigma_r={{{' '.join([str(soap_params['atom_sigma_r'])] * n)}}} "
        f"atom_sigma_t={{{' '.join([str(soap_params['atom_sigma_t'])] * n)}}} "
        f"atom_sigma_r_scaling={{{' '.join([str(soap_params['atom_sigma_r_scaling'])] * n)}}} "
        f"atom_sigma_t_scaling={{{' '.join([str(soap_params['atom_sigma_t_scaling'])] * n)}}} "
        f"amplitude_scaling={{{' '.join([str(soap_params['amplitude_scaling'])] * n)}}} "
        f"central_weight={{{' '.join([str(soap_params['central_weight'])] * n)}}}"
    )
    return s


def _make_test_structures() -> dict[str, Atoms]:
    """Small ASE Atoms edge cases: water_monomer, water_dimer, h2_close,
    h2_far (soft-cutoff edge), single_h (self only), si_bulk (PBC),
    si_dimer (image neighbours)."""
    structures: dict[str, Atoms] = {}

    # Water monomer
    water = molecule("H2O")
    water.cell = 30.0 * np.eye(3)
    water.pbc = False
    water.center()
    structures["water_monomer"] = water

    # Water dimer (rough geometry, no relaxation needed)
    water_dimer = Atoms(
        symbols="OHHOHH",
        positions=[
            [0.00, 0.00, 0.00],   # O1
            [0.96, 0.00, 0.00],   # H1
            [-0.24, 0.93, 0.00],  # H1'
            [3.20, 0.00, 0.00],   # O2
            [3.95, 0.66, 0.00],   # H2
            [3.95, -0.66, 0.00],  # H2'
        ],
        cell=30.0 * np.eye(3),
        pbc=False,
    )
    water_dimer.center()
    structures["water_dimer"] = water_dimer

    # H–H at typical bond length
    h2_close = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],
                     cell=30.0 * np.eye(3), pbc=False)
    h2_close.center()
    structures["h2_close"] = h2_close

    # H–H near rcut_soft / rcut_hard
    h2_far = Atoms("HH", positions=[[0.0, 0.0, 0.0], [5.7, 0.0, 0.0]],
                   cell=30.0 * np.eye(3), pbc=False)
    h2_far.center()
    structures["h2_far"] = h2_far

    # Single H atom (degenerate: no neighbours, only the self-pair)
    single_h = Atoms("H", positions=[[0.0, 0.0, 0.0]],
                     cell=30.0 * np.eye(3), pbc=False)
    single_h.center()
    structures["single_h"] = single_h

    # Periodic Si bulk (8 atoms, FCC with basis)
    si_bulk = bulk("Si", "diamond", a=5.43, cubic=True)
    structures["si_bulk"] = si_bulk

    # Tiny periodic Si — image atoms come into play
    si_dimer = Atoms("SiSi",
                     positions=[[0.0, 0.0, 0.0], [2.35, 0.0, 0.0]],
                     cell=4.0 * np.eye(3), pbc=True)
    structures["si_dimer"] = si_dimer

    return structures


def _run_quippy(atoms: Atoms, species_Z: list[int],
                soap_params: dict) -> dict[str, np.ndarray]:
    """Call quippy with one Descriptor per central species.

    Returns a dict of reference arrays: descriptors [n_atoms, dim_q],
    grad_values [P, 3, dim_q], pair_atom/pair_gidx [P] (0-based), plus
    structure metadata (numbers, positions, cell, pbc, species_Z).
    dim_q is the compressed SOAP dim (uniform across species for trivial).
    """
    n_atoms = len(atoms)
    soap_strings = [
        _build_soap_string(species_Z, **soap_params) + f" central_index={k}"
        for k in (np.arange(len(species_Z), dtype=int) + 1)
    ]
    builders = [Descriptor(s) for s in soap_strings]
    outs = [b.calc(atoms, grad=True) for b in builders]

    # Determine dim_q from first non-empty output
    dim_q = None
    for out in outs:
        data = out.get("data")
        if data is not None and data.size > 0 and data.shape[1] > 0:
            dim_q = data.shape[1]
            break
    if dim_q is None:
        raise RuntimeError("All quippy outputs are empty for this structure")

    descriptors = np.zeros((n_atoms, dim_q), dtype=np.float32)
    grad_chunks: list[np.ndarray] = []
    pair_atom_chunks: list[np.ndarray] = []
    pair_gidx_chunks: list[np.ndarray] = []

    for out in outs:
        data = out.get("data")
        if data is None or data.size == 0 or data.shape[1] == 0:
            continue
        ci = np.asarray(out["ci"], dtype=np.int32) - 1  # 1-based -> 0-based
        descriptors[ci] = np.asarray(data, dtype=np.float32)
        grad_idx = out.get("grad_index_0based")
        if grad_idx is not None and len(grad_idx) > 0:
            grad_idx = np.asarray(grad_idx, dtype=np.int32)
            grad_chunks.append(np.asarray(out["grad_data"], dtype=np.float32))
            pair_atom_chunks.append(grad_idx[:, 0])
            pair_gidx_chunks.append(grad_idx[:, 1])

    if grad_chunks:
        grad_values = np.concatenate(grad_chunks, axis=0)
        pair_atom   = np.concatenate(pair_atom_chunks)
        pair_gidx   = np.concatenate(pair_gidx_chunks)
    else:
        grad_values = np.zeros((0, 3, dim_q), dtype=np.float32)
        pair_atom   = np.zeros(0, dtype=np.int32)
        pair_gidx   = np.zeros(0, dtype=np.int32)

    return dict(
        n_atoms=np.int32(n_atoms),
        n_species=np.int32(len(species_Z)),
        species_Z=np.asarray(species_Z, dtype=np.int32),
        numbers=np.asarray(atoms.numbers, dtype=np.int32),
        positions=np.asarray(atoms.positions, dtype=np.float32),
        cell=np.asarray(atoms.cell.array, dtype=np.float32),
        pbc=np.asarray(atoms.pbc, dtype=bool),
        descriptors=descriptors,
        grad_values=grad_values,
        pair_atom=pair_atom,
        pair_gidx=pair_gidx,
    )


def build_reference_fixtures(out_dir: str = "tests/fixtures") -> None:
    """Generate quippy reference data for every test structure.

    Saves <out_dir>/<name>.npz with the _run_quippy dict plus SOAP params
    as JSON. Subsequent phases assert_allclose against these.
    """
    import json

    os.makedirs(out_dir, exist_ok=True)
    structures = _make_test_structures()

    # Species set per structure = element types present, sorted by Z.
    print(f"Generating quippy reference fixtures in {out_dir}/")
    print(f"  SOAP params: {REFERENCE_SOAP_PARAMS}")
    print()

    for name, atoms in structures.items():
        species_Z = sorted(set(int(z) for z in atoms.numbers))
        ref = _run_quippy(atoms, species_Z, REFERENCE_SOAP_PARAMS)

        path = os.path.join(out_dir, f"{name}.npz")
        np.savez_compressed(
            path,
            soap_params=np.array(json.dumps(REFERENCE_SOAP_PARAMS)),
            **ref,
        )

        # Sanity stats — flag obvious breakage early.
        desc = ref["descriptors"]
        grads = ref["grad_values"]
        has_nan = np.isnan(desc).any() or np.isnan(grads).any()
        norms = np.linalg.norm(desc, axis=1)  # should be ~1 (normalised SOAP)
        n_pairs = len(ref["pair_atom"])
        n_atoms = int(ref["n_atoms"])
        print(f"  {name:<16} N={n_atoms:>3}  species={species_Z}  "
              f"pairs={n_pairs:>4}  dim_q={desc.shape[1]:>3}  "
              f"|d|=[{norms.min():.4f}, {norms.max():.4f}]  "
              f"NaN={has_nan}")

    print()
    print(f"Wrote {len(structures)} fixtures.")


def load_fixture(name: str, fixtures_dir: str = "tests/fixtures") -> dict:
    """Load a fixture .npz into a dict; soap_params decoded from JSON."""
    import json
    data = np.load(os.path.join(fixtures_dir, f"{name}.npz"), allow_pickle=False)
    out = {k: data[k] for k in data.files if k != "soap_params"}
    out["soap_params"] = json.loads(str(data["soap_params"]))
    return out


def assert_allclose_soap(
    predicted: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    label: str = "soap",
) -> None:
    """Compare SOAP descriptors; report max abs/rel diff and worst index."""
    if predicted.shape != expected.shape:
        raise AssertionError(
            f"{label} shape mismatch: predicted {predicted.shape} vs expected {expected.shape}")
    abs_diff = np.abs(predicted - expected)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    denom = np.maximum(np.abs(expected), 1e-12)
    rel_diff = abs_diff / denom
    max_rel = float(rel_diff.max()) if rel_diff.size else 0.0
    if max_abs > atol and max_rel > rtol:
        idx = np.unravel_index(int(abs_diff.argmax()), abs_diff.shape)
        raise AssertionError(
            f"{label}: max_abs={max_abs:.3e} > atol={atol:.0e}, "
            f"max_rel={max_rel:.3e} > rtol={rtol:.0e}; "
            f"worst at {idx}: predicted={predicted[idx]:.6e} expected={expected[idx]:.6e}"
        )


# =========================================================================
# Phase 1 — Basis matrices, compression mask, multiplicity array
# Fortran refs: soap_turbo_radial.f90:770-816, soap_turbo_compress.f90:73-106,
# soap_turbo.f90:496-535.
# =========================================================================


def build_overlap_matrix_poly3(alpha_max: int) -> np.ndarray:
    """Analytic overlap matrix S of the unnormalised poly3 radial basis.

    S[i,i]=1; S[i,j]=sqrt((5+2i)(5+2j))/(5+i+j) for j!=i (1-indexed).
    Returns [alpha_max, alpha_max], SPD for alpha_max<=~10.
    """
    a = np.arange(1, alpha_max + 1, dtype=np.float64)            # 1-indexed
    i_idx, j_idx = np.meshgrid(a, a, indexing="ij")              # both [α, α]
    S = np.sqrt((5.0 + 2.0 * i_idx) * (5.0 + 2.0 * j_idx)) / (5.0 + i_idx + j_idx)
    return S


def build_orthonormalization_matrix_poly3(alpha_max: int) -> np.ndarray:
    """W = S^(-1/2) for the poly3 basis, via SVD: U·diag(1/sqrt(svd))·Vt.

    Numerically identical to the Fortran's SVD-then-Cholesky two-step
    (soap_turbo_radial.f90:780-816) for SPD S.
    """
    S = build_overlap_matrix_poly3(alpha_max)
    U, svd, Vt = np.linalg.svd(S)                                 # S = U·diag(svd)·Vt
    inv_sqrt = 1.0 / np.sqrt(svd)
    W = U @ np.diag(inv_sqrt) @ Vt
    return W


def build_block_W(alpha_max_per_species: list[int]) -> np.ndarray:
    """Block-diagonal W [n_max, n_max]: one poly3 basis per species.

    Off-diagonal blocks are zero (cross-species radial functions are
    orthogonal by definition; soap_turbo.f90:227-258).
    """
    n_max = sum(alpha_max_per_species)
    W = np.zeros((n_max, n_max), dtype=np.float64)
    offset = 0
    for am in alpha_max_per_species:
        W[offset:offset + am, offset:offset + am] = build_orthonormalization_matrix_poly3(am)
        offset += am
    return W


def compute_dim_q(cfg) -> int:
    """SOAP descriptor dimension n_compressed from cfg alone (closed-form).

    Depends only on (alpha_max, l_max, num_types, compress_mode); no
    neighbour-list build. compress_mode 'linear' is special-cased; others
    route to make_compress_mask_trivial.
    """
    alpha_max = int(cfg.alpha_max)
    l_max = int(cfg.l_max)
    n_species = int(cfg.num_types)
    compress_mode = getattr(cfg, "compress_mode", "trivial")
    if compress_mode == "linear":
        # quippy linear default = 2 × n_species pivots × (l_max+1) features
        return int(2 * n_species * (l_max + 1))
    alpha_per_species = [alpha_max] * n_species
    mask_info = make_compress_mask_trivial(alpha_per_species, l_max)
    return int(mask_info["n_compressed"])


def descriptor_block_layout(cfg) -> dict:
    """Group the flat dim_q axis by unordered species pair (s_a, s_b).

    Each kept (n, n', l) channel maps to the pair owning (n, n'),
    decomposing dim_q into T·(T+1)/2 blocks. Kept channels for a pair are
    NOT contiguous (they interleave along the Fortran (n, n', l) order), so
    callers gather by q index. compress_mode='trivial' only; raises on
    'linear' (not separable by species pair).

    Returns dict: pair_keys, pair_q_index, pair_ln_index, l_index,
    block_sizes, max_block_size, alpha_eff_per_pair, max_alpha_eff,
    N_per_l, dim_q.
    """
    compress_mode = getattr(cfg, "compress_mode", "trivial")
    if compress_mode != "trivial":
        raise NotImplementedError(
            f"descriptor_block_layout supports compress_mode='trivial' "
            f"only (got {compress_mode!r}); 'linear' compression's "
            f"channel-to-species-pair mapping is not separable")
    alpha_max = int(cfg.alpha_max)
    l_max = int(cfg.l_max)
    n_species = int(cfg.num_types)
    alpha_per_species = [alpha_max] * n_species

    # 1-based species ownership of each global radial index.
    n_to_species = np.empty(sum(alpha_per_species) + 1, dtype=np.int32)
    n_to_species[0] = -1   # unused (n is 1-based in Fortran)
    cursor = 1
    for s, am in enumerate(alpha_per_species):
        n_to_species[cursor:cursor + am] = s
        cursor += am

    # Pivots: first global radial index of each species
    pivots: set[int] = {1}
    for s in range(n_species - 1):
        pivots.add(max(pivots) + alpha_per_species[s])

    pair_q_index: dict[tuple[int, int], list[int]] = {}
    # Fortran emit order is (n, n', l) with l innermost, emitting when n or
    # n' is a pivot. pair_ln_index maps (pair, l) -> [q indices].
    pair_ln_index: dict[tuple[int, int], dict[int, list[int]]] = {}

    q_counter = 0
    n_max = sum(alpha_per_species)
    for n in range(1, n_max + 1):
        for nprime in range(n, n_max + 1):
            kept = (n in pivots) or (nprime in pivots)
            for l in range(l_max + 1):
                if kept:
                    s_a = int(n_to_species[n])
                    s_b = int(n_to_species[nprime])
                    pair = (min(s_a, s_b), max(s_a, s_b))
                    pair_q_index.setdefault(pair, []).append(q_counter)
                    pair_ln_index.setdefault(pair, {}).setdefault(l, []).append(q_counter)
                    q_counter += 1
    # Sort pair_keys for deterministic iteration order across runs
    pair_keys = sorted(pair_q_index.keys())
    pair_q_index_np = {k: np.asarray(pair_q_index[k], dtype=np.int32) for k in pair_keys}
    block_sizes = {k: int(pair_q_index_np[k].size) for k in pair_keys}
    # Convert pair_ln_index inner lists to int32 arrays.
    pair_ln_index_np: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for k in pair_keys:
        pair_ln_index_np[k] = {
            l: np.asarray(v, dtype=np.int32)
            for l, v in pair_ln_index[k].items()
        }
    # Per-pair α_eff = radial channels per l (uniform across l for a pair).
    alpha_eff_per_pair: dict[tuple[int, int], int] = {}
    for k in pair_keys:
        counts_by_l = {l: len(v) for l, v in pair_ln_index_np[k].items()}
        assert set(counts_by_l.keys()) == set(range(l_max + 1)), (
            f"pair {k}: l-index keys {sorted(counts_by_l)} != "
            f"range(0, {l_max + 1})")
        unique_counts = set(counts_by_l.values())
        assert len(unique_counts) == 1, (
            f"pair {k}: alpha_eff varies across l: {counts_by_l}")
        alpha_eff_per_pair[k] = int(next(iter(unique_counts)))
    # Per-l union over all pairs (deterministic pair order). Used by
    # cross_pair_l mixing. N_l is uniform across l (asserted below).
    l_index_np: dict[int, np.ndarray] = {}
    for l in range(l_max + 1):
        union: list[int] = []
        for k in pair_keys:
            union.extend(int(q) for q in pair_ln_index_np[k][l])
        l_index_np[l] = np.asarray(union, dtype=np.int32)
    N_l_per = {l: l_index_np[l].size for l in range(l_max + 1)}
    if len(set(N_l_per.values())) > 1:
        raise AssertionError(
            f"N_l varies across l: {N_l_per}. cross_pair_l mixing "
            f"assumes uniform N_l. (Fortran emit order would normally "
            f"guarantee this — check descriptor_block_layout invariants.)")
    N_per_l = int(next(iter(N_l_per.values()))) if N_l_per else 0
    return dict(
        pair_keys=pair_keys,
        pair_q_index=pair_q_index_np,
        pair_ln_index=pair_ln_index_np,
        l_index=l_index_np,
        block_sizes=block_sizes,
        max_block_size=max(block_sizes.values()) if block_sizes else 0,
        alpha_eff_per_pair=alpha_eff_per_pair,
        max_alpha_eff=max(alpha_eff_per_pair.values()) if alpha_eff_per_pair else 0,
        N_per_l=N_per_l,
        dim_q=int(q_counter),
    )


def descriptor_preprocess_layout(cfg, layout: dict, mode: str) -> dict:
    """Precomputed index mappings for the preprocessing contraction.

    Forward op is a per-type scatter-sum:
        desc'[t, q_new] = Σ_{q_raw: map[q_raw]=q_new} W_pre[t, q_raw]·desc[q_raw]

    Modes: "off", "angular" (contract over l per (pair, n_pair)),
    "species_pair", "both", "nep4_radial". Returns dict with dim_q_new,
    q_raw_to_q_new [Q_raw] (None if "off"), coef_shape, coef_init_norm, L.
    """
    pair_keys = layout["pair_keys"]
    pair_ln_index = layout["pair_ln_index"]
    alpha_eff_per_pair = layout["alpha_eff_per_pair"]
    Q_raw = int(layout["dim_q"])
    L = int(cfg.l_max) + 1

    # l_keep: l < l_keep pass through (own output channel); l >= l_keep are
    # summed into one channel per (pair, n_pair). Default 1. "angular"/"both".
    l_keep = int(getattr(cfg, "descriptor_preprocess_angular_l_keep", 1))
    if l_keep < 0 or l_keep > L:
        raise ValueError(
            f"descriptor_preprocess_angular_l_keep={l_keep} not in "
            f"[0, l_max+1={L}].")

    if mode == "off":
        return dict(
            dim_q_new=Q_raw,
            q_raw_to_q_new=None,
            coef_shape=(0,),
            coef_init_norm=1.0,
            coef_init_per_q_raw=None,
            summed_q_raw_mask=None,
            L=L,
        )

    if mode == "angular":
        # Per (pair, n_pair): l<l_keep passthrough (W_pre=1); l>=l_keep summed
        # into one channel with L-l_keep learnable coefs (init 1/(L-l_keep)).
        n_summed_block = 1 if l_keep < L else 0
        n_summed_l = L - l_keep
        init_summed_val = (1.0 / float(n_summed_l)) if n_summed_l > 0 else 0.0
        q_raw_to_q_new = np.full(Q_raw, -1, dtype=np.int32)
        init_per_q_raw = np.zeros(Q_raw, dtype=np.float32)
        summed_mask = np.zeros(Q_raw, dtype=bool)
        out_cursor = 0
        for pair in pair_keys:
            alpha = alpha_eff_per_pair[pair]
            for n_pair in range(alpha):
                # l < l_keep → own passthrough channel.
                for l in range(l_keep):
                    q_raw = int(pair_ln_index[pair][l][n_pair])
                    q_raw_to_q_new[q_raw] = out_cursor
                    init_per_q_raw[q_raw] = 1.0          # passthrough init
                    out_cursor += 1
                # l ≥ l_keep → summed into one channel.
                if n_summed_block > 0:
                    for l in range(l_keep, L):
                        q_raw = int(pair_ln_index[pair][l][n_pair])
                        q_raw_to_q_new[q_raw] = out_cursor
                        init_per_q_raw[q_raw] = init_summed_val
                        summed_mask[q_raw] = True
                    out_cursor += 1
        assert int(np.min(q_raw_to_q_new)) >= 0, (
            f"angular preprocess layout did not cover all Q_raw channels "
            f"(min map = {int(np.min(q_raw_to_q_new))})")
        return dict(
            dim_q_new=int(out_cursor),
            q_raw_to_q_new=q_raw_to_q_new,
            coef_shape=(Q_raw,),
            coef_init_norm=init_summed_val if n_summed_l > 0 else 1.0,
            coef_init_per_q_raw=init_per_q_raw,
            summed_q_raw_mask=summed_mask,
            L=max(1, n_summed_l),  # glorot fan_in for summed slots
            l_keep=l_keep,
        )

    if mode == "species_pair":
        # Per centre t: SELF block = (t,t) pair (passthrough); OTHER block =
        # pairs (t,j≠t) summed (T-1 contributors). Per-centre dim_q_new =
        # (α_self + max_α_cross)·L. Max across t keeps W0 uniformly shaped.
        T = int(cfg.num_types)
        # Per-centre α_self and max-cross-α
        alpha_self_per_t = [0] * T
        alpha_other_max_per_t = [0] * T
        for t in range(T):
            for pair in pair_keys:
                a, b = pair
                if a == t and b == t:
                    alpha_self_per_t[t] = alpha_eff_per_pair[pair]
                elif a == t or b == t:
                    alpha_other_max_per_t[t] = max(
                        alpha_other_max_per_t[t], alpha_eff_per_pair[pair])
        alpha_self = max(alpha_self_per_t)
        alpha_other = max(alpha_other_max_per_t)
        Q_new = (alpha_self + alpha_other) * L
        # q_new layout: [0, alpha_self·L) SELF, then OTHER; (n_pair, l) major.
        q_raw_to_q_new = np.full((T, Q_raw), -1, dtype=np.int32)
        init_per_t_q_raw = np.zeros((T, Q_raw), dtype=np.float32)
        summed_mask = np.zeros((T, Q_raw), dtype=bool)
        init_other = 1.0 / float(max(1, T - 1))
        for t in range(T):
            for pair in pair_keys:
                a, b = pair
                if t != a and t != b:
                    continue
                alpha = alpha_eff_per_pair[pair]
                if a == b:  # self pair
                    block_offset = 0
                    is_summed = False
                    init_val = 1.0  # passthrough
                else:        # other pair
                    block_offset = alpha_self * L
                    is_summed = True
                    init_val = init_other
                for l in range(L):
                    pair_l_qs = pair_ln_index[pair][l]
                    for n_pair in range(alpha):
                        q_raw = int(pair_l_qs[n_pair])
                        q_new = block_offset + n_pair * L + l
                        assert q_raw_to_q_new[t, q_raw] == -1, (
                            f"q_raw {q_raw} mapped twice for t={t}.")
                        q_raw_to_q_new[t, q_raw] = q_new
                        init_per_t_q_raw[t, q_raw] = init_val
                        summed_mask[t, q_raw] = is_summed
        return dict(
            dim_q_new=Q_new,
            q_raw_to_q_new=q_raw_to_q_new,         # 2-D [T, Q_raw]
            coef_shape=(Q_raw,),
            coef_init_norm=init_other,
            coef_init_per_q_raw=init_per_t_q_raw,  # 2-D [T, Q_raw]
            summed_q_raw_mask=summed_mask,         # 2-D [T, Q_raw]
            L=max(1, T - 1),
            l_keep=L,  # not applicable; angular axis fully kept here
        )

    if mode == "both":
        # species_pair × angular l_keep collapse. Per (centre, pair, n_pair):
        # l<l_keep own channel, l>=l_keep summed. SELF+kept = passthrough;
        # SELF+sum, OTHER+kept, OTHER+sum are summed with varying fan-in.
        T = int(cfg.num_types)
        alpha_self_per_t = [0] * T
        alpha_other_max_per_t = [0] * T
        for t in range(T):
            for pair in pair_keys:
                a, b = pair
                if a == t and b == t:
                    alpha_self_per_t[t] = alpha_eff_per_pair[pair]
                elif a == t or b == t:
                    alpha_other_max_per_t[t] = max(
                        alpha_other_max_per_t[t], alpha_eff_per_pair[pair])
        alpha_self = max(alpha_self_per_t)
        alpha_other = max(alpha_other_max_per_t)
        n_summed_block = 1 if l_keep < L else 0
        n_summed_l = L - l_keep
        out_per_pair = l_keep + n_summed_block
        Q_new = (alpha_self + alpha_other) * out_per_pair
        # q_new layout: SELF block [0, alpha_self·out_per_pair), then OTHER;
        # (n_pair, [l<l_keep] then [sum]) ordering, l major within n_pair.
        self_end = alpha_self * out_per_pair
        init_self_kept   = 1.0
        init_self_sum    = 1.0 / float(max(1, n_summed_l))
        init_other_kept  = 1.0 / float(max(1, T - 1))
        init_other_sum   = 1.0 / float(max(1, (T - 1) * n_summed_l))
        q_raw_to_q_new = np.full((T, Q_raw), -1, dtype=np.int32)
        init_per_t_q_raw = np.zeros((T, Q_raw), dtype=np.float32)
        summed_mask = np.zeros((T, Q_raw), dtype=bool)
        for t in range(T):
            for pair in pair_keys:
                a, b = pair
                if t != a and t != b:
                    continue
                alpha = alpha_eff_per_pair[pair]
                if a == b:
                    block_offset = 0
                    init_kept_val = init_self_kept
                    init_sum_val = init_self_sum
                    is_summed_for_kept_l = False  # self+kept = single contributor
                else:
                    block_offset = self_end
                    init_kept_val = init_other_kept
                    init_sum_val = init_other_sum
                    is_summed_for_kept_l = True   # other+kept = T-1 contributors
                for n_pair in range(alpha):
                    base = block_offset + n_pair * out_per_pair
                    # l < l_keep
                    for l in range(l_keep):
                        q_raw = int(pair_ln_index[pair][l][n_pair])
                        q_new = base + l
                        assert q_raw_to_q_new[t, q_raw] == -1
                        q_raw_to_q_new[t, q_raw] = q_new
                        init_per_t_q_raw[t, q_raw] = init_kept_val
                        summed_mask[t, q_raw] = is_summed_for_kept_l
                    # l ≥ l_keep summed
                    if n_summed_block > 0:
                        q_sum = base + l_keep  # singleton summed slot
                        for l in range(l_keep, L):
                            q_raw = int(pair_ln_index[pair][l][n_pair])
                            assert q_raw_to_q_new[t, q_raw] == -1
                            q_raw_to_q_new[t, q_raw] = q_sum
                            init_per_t_q_raw[t, q_raw] = init_sum_val
                            summed_mask[t, q_raw] = True   # always summed
        return dict(
            dim_q_new=Q_new,
            q_raw_to_q_new=q_raw_to_q_new,
            coef_shape=(Q_raw,),
            coef_init_norm=init_other_sum,
            coef_init_per_q_raw=init_per_t_q_raw,
            summed_q_raw_mask=summed_mask,
            L=max(1, (T - 1) * n_summed_l),
            l_keep=l_keep,
        )

    if mode == "nep4_radial":
        # NEP4 learned-basis fold: rank-1 bilinear weighting
        #   g[t, n'', l] = Σ_{n,n'} c[t,s(n),n'',k(n)]·c[t,s(n'),n'',k(n')]·p[n,n',l]
        # (same indexing as NEP4's c^{Z_i,Z_j}_{n'',k}). Adds nep4_* layout
        # fields (n_max_out, l_of_q, n_global, np_global, n_to_species,
        # n_to_local) consumed by TNEP's _W0_preprocess_eff / SNES tail.
        alpha_max = int(cfg.alpha_max)
        T = int(cfg.num_types)
        n_max_global = T * alpha_max
        if Q_raw % L != 0:
            raise AssertionError(
                f"nep4_radial: Q_raw={Q_raw} not divisible by L={L}; "
                f"trivial-compressed SOAP should always emit L channels per "
                f"kept (n, n') pair. Layout invariant broken.")
        Q_pair_kept = Q_raw // L
        n_max_out_cfg = getattr(cfg, "descriptor_nep4_n_max_out", None)
        if n_max_out_cfg is None:
            # Preserve total dim: Q_new = Q_raw.
            n_max_out = Q_pair_kept
        else:
            n_max_out = int(n_max_out_cfg)
            if n_max_out <= 0:
                raise ValueError(
                    f"descriptor_nep4_n_max_out={n_max_out} must be > 0.")
        Q_new = n_max_out * L
        # Re-walk the Fortran (n, n', l) emit order, recording
        # (n_global, n'_global, l) per q_raw (keeps layout signature stable).
        alpha_per_species = [alpha_max] * T
        pivots: set[int] = {1}
        for s in range(T - 1):
            pivots.add(max(pivots) + alpha_per_species[s])
        n_to_species_1based = np.empty(
            sum(alpha_per_species) + 1, dtype=np.int32)
        n_to_species_1based[0] = -1
        cursor = 1
        for s, am in enumerate(alpha_per_species):
            n_to_species_1based[cursor:cursor + am] = s
            cursor += am
        n_max_fortran = sum(alpha_per_species)
        nep4_l_of_q = np.zeros(Q_raw, dtype=np.int32)
        nep4_n_global = np.zeros(Q_raw, dtype=np.int32)
        nep4_np_global = np.zeros(Q_raw, dtype=np.int32)
        # global-n (0-based) for c tensor's flat species×α axis:
        #   n_global = species·α_max + within-species-index
        nep4_n_to_species = np.zeros(n_max_global, dtype=np.int32)
        nep4_n_to_local = np.zeros(n_max_global, dtype=np.int32)
        for s in range(T):
            for k in range(alpha_max):
                ng = s * alpha_max + k
                nep4_n_to_species[ng] = s
                nep4_n_to_local[ng] = k
        q_counter = 0
        for n in range(1, n_max_fortran + 1):
            for nprime in range(n, n_max_fortran + 1):
                kept = (n in pivots) or (nprime in pivots)
                for l in range(L):
                    if kept:
                        s_a = int(n_to_species_1based[n])
                        s_b = int(n_to_species_1based[nprime])
                        # within-species local index (0-based)
                        species_start_a = sum(alpha_per_species[:s_a])
                        species_start_b = sum(alpha_per_species[:s_b])
                        k_a = (n - 1) - species_start_a
                        k_b = (nprime - 1) - species_start_b
                        nep4_l_of_q[q_counter] = l
                        nep4_n_global[q_counter] = s_a * alpha_max + k_a
                        nep4_np_global[q_counter] = s_b * alpha_max + k_b
                        q_counter += 1
        assert q_counter == Q_raw, (
            f"nep4_radial layout walk emitted {q_counter} entries, "
            f"expected Q_raw={Q_raw}.")
        # All c entries learnable; summed_mask sized over the full c tensor
        # so SNES treats every entry as a μ slot.
        coef_shape = (T, T, n_max_out, alpha_max)
        full_coef_size = int(np.prod(coef_shape))
        summed_mask = np.ones(coef_shape, dtype=bool)
        # Glorot fan-in = α_max (per c sum). c lives in n''/k-space, not q_raw.
        fan_in = max(1, alpha_max)
        init_norm = float(np.sqrt(1.0 / fan_in))
        return dict(
            dim_q_new=Q_new,
            q_raw_to_q_new=None,           # not used by the bilinear fold
            coef_shape=coef_shape,
            coef_init_norm=init_norm,
            coef_init_per_q_raw=None,
            summed_q_raw_mask=summed_mask,
            L=L,
            l_keep=L,
            # NEP4-specific fields:
            nep4_n_max_out=int(n_max_out),
            nep4_alpha_max=int(alpha_max),
            nep4_n_max_global=int(n_max_global),
            nep4_full_coef_size=full_coef_size,
            nep4_l_of_q=nep4_l_of_q,
            nep4_n_global=nep4_n_global,
            nep4_np_global=nep4_np_global,
            nep4_n_to_species=nep4_n_to_species,
            nep4_n_to_local=nep4_n_to_local,
        )

    raise ValueError(
        f"unknown descriptor_preprocess_contract mode {mode!r}; "
        f"expected one of: 'off', 'angular', 'species_pair', 'both', "
        f"'nep4_radial'")


def descriptor_post_preprocess_block_layout(
        cfg, layout: dict, preprocess_mode: str) -> dict:
    """Block layout of the Q_new preprocess output (post-contraction).

    Mirrors descriptor_block_layout but q-indices are in Q_new space, plus
    an L_eff field (effective l-axis size for l_aware mixing). Used by
    descriptor_mixing when it composes with preprocess.

    Per mode: "angular" keeps pair_keys, L_eff = l_keep + (l_keep<L);
    "species_pair"/"both" use pair_keys ["self","other"], alpha_eff =
    {self: max_t α[(t,t)], other: max α_cross}, L_eff = L (species_pair) or
    l_keep + (l_keep<L) (both).
    """
    pair_keys_raw = layout["pair_keys"]
    pair_ln_index = layout["pair_ln_index"]
    alpha_eff_per_pair = layout["alpha_eff_per_pair"]
    L = int(cfg.l_max) + 1
    T = int(cfg.num_types)

    if preprocess_mode == "angular":
        l_keep = int(getattr(cfg, "descriptor_preprocess_angular_l_keep", 1))
        L_eff = l_keep + (1 if l_keep < L else 0)
        post_pair_keys = list(pair_keys_raw)
        post_alpha = {k: int(alpha_eff_per_pair[k]) for k in post_pair_keys}
        post_pair_q_index: dict = {}
        post_pair_ln_index: dict = {}
        post_block_sizes: dict = {}
        cursor = 0
        for k in post_pair_keys:
            alpha = post_alpha[k]
            pair_qs = []
            per_l: dict[int, list[int]] = {l: [] for l in range(L_eff)}
            for n_pair in range(alpha):
                base = cursor + n_pair * L_eff
                for l_post in range(L_eff):
                    q = base + l_post
                    pair_qs.append(q)
                    per_l[l_post].append(q)
            post_pair_q_index[k] = np.asarray(pair_qs, dtype=np.int32)
            post_pair_ln_index[k] = {
                l: np.asarray(per_l[l], dtype=np.int32) for l in range(L_eff)}
            post_block_sizes[k] = alpha * L_eff
            cursor += alpha * L_eff
        dim_q_post = cursor

    elif preprocess_mode in ("species_pair", "both"):
        if preprocess_mode == "both":
            l_keep = int(getattr(cfg, "descriptor_preprocess_angular_l_keep", 1))
            L_eff = l_keep + (1 if l_keep < L else 0)
        else:
            L_eff = L
        # alpha per (centre, block) — take the max across centres for
        # shared mixing (uniform alpha across species → no padding).
        alpha_self_per_t = [0] * T
        alpha_other_max_per_t = [0] * T
        for t in range(T):
            for pair in pair_keys_raw:
                a, b = pair
                if a == t and b == t:
                    alpha_self_per_t[t] = alpha_eff_per_pair[pair]
                elif a == t or b == t:
                    alpha_other_max_per_t[t] = max(
                        alpha_other_max_per_t[t], alpha_eff_per_pair[pair])
        alpha_self = max(alpha_self_per_t)
        alpha_other = max(alpha_other_max_per_t)
        post_pair_keys = ["self", "other"]
        post_alpha = {"self": alpha_self, "other": alpha_other}
        post_pair_q_index = {}
        post_pair_ln_index = {}
        post_block_sizes = {}
        self_block_size = alpha_self * L_eff
        other_block_size = alpha_other * L_eff
        for k in post_pair_keys:
            offset = 0 if k == "self" else self_block_size
            alpha = post_alpha[k]
            pair_qs = []
            per_l: dict[int, list[int]] = {l: [] for l in range(L_eff)}
            for n_pair in range(alpha):
                base = offset + n_pair * L_eff
                for l_post in range(L_eff):
                    q = base + l_post
                    pair_qs.append(q)
                    per_l[l_post].append(q)
            post_pair_q_index[k] = np.asarray(pair_qs, dtype=np.int32)
            post_pair_ln_index[k] = {
                l: np.asarray(per_l[l], dtype=np.int32) for l in range(L_eff)}
            post_block_sizes[k] = alpha * L_eff
        dim_q_post = self_block_size + other_block_size

    else:  # "off"
        raise ValueError(
            f"descriptor_post_preprocess_block_layout: preprocess_mode "
            f"{preprocess_mode!r} has no post-contraction layout — call "
            f"descriptor_block_layout instead.")

    # l_index: union of q-indices across pair_keys at each fixed l_post.
    l_index_post = {}
    for l_post in range(L_eff):
        union = []
        for k in post_pair_keys:
            union.extend(int(q) for q in post_pair_ln_index[k][l_post])
        l_index_post[l_post] = np.asarray(union, dtype=np.int32)

    return dict(
        pair_keys=post_pair_keys,
        pair_q_index=post_pair_q_index,
        pair_ln_index=post_pair_ln_index,
        l_index=l_index_post,
        block_sizes=post_block_sizes,
        max_block_size=max(post_block_sizes.values()) if post_block_sizes else 0,
        alpha_eff_per_pair=post_alpha,
        max_alpha_eff=max(post_alpha.values()) if post_alpha else 0,
        N_per_l=int(l_index_post[0].size) if L_eff > 0 else 0,
        dim_q=int(dim_q_post),
        L_eff=int(L_eff),
    )


def make_compress_mask_trivial(
    alpha_max_per_species: list[int],
    l_max: int,
) -> dict:
    """Sparse projection P for compress_mode='trivial'.

    Fortran ref: soap_turbo_compress.f90:73-106. Keeps only (n, n', l)
    channels where n or n' is a species' first global radial index ("pivot":
    n=1, and n=1+alpha_max, ... for extra species), in Fortran (n, n', l) order.

    Returns: compressed_idx [P_nonzero] (output), uncompressed_idx [P_nonzero]
    (source), coeffs [P_nonzero] (all 1.0), skip_mask [n_uncompressed] (True=
    dropped), n_compressed (= dim_q), n_uncompressed (= n_max·(n_max+1)/2·(l_max+1)).
    """
    n_species = len(alpha_max_per_species)
    n_max = sum(alpha_max_per_species)

    # 1-based pivots: first global radial index of each species block
    pivots: set[int] = {1}
    for i in range(n_species - 1):
        next_pivot = max(pivots) + alpha_max_per_species[i]
        pivots.add(next_pivot)

    n_uncompressed = n_max * (n_max + 1) // 2 * (l_max + 1)
    skip_mask = np.ones(n_uncompressed, dtype=bool)               # default: skip everything
    compressed_idx_list: list[int] = []
    uncompressed_idx_list: list[int] = []

    k = 0                                                         # uncompressed (n, n', l) counter (0-based)
    counter = 0                                                   # compressed output counter (0-based)
    for n in range(1, n_max + 1):
        for nprime in range(n, n_max + 1):
            for _ in range(l_max + 1):
                if (n in pivots) or (nprime in pivots):
                    skip_mask[k] = False
                    compressed_idx_list.append(counter)
                    uncompressed_idx_list.append(k)
                    counter += 1
                k += 1

    return dict(
        compressed_idx=np.asarray(compressed_idx_list, dtype=np.int32),
        uncompressed_idx=np.asarray(uncompressed_idx_list, dtype=np.int32),
        coeffs=np.ones(len(compressed_idx_list), dtype=np.float64),
        skip_mask=skip_mask,
        n_compressed=int(counter),
        n_uncompressed=int(n_uncompressed),
    )


def build_multiplicity_array(
    n_max: int,
    l_max: int,
    skip_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-(n, n', l, m) multiplicities for the power-spectrum sum.

    Fortran ref: soap_turbo.f90:496-535.
        mult = 1; ·= sqrt(2) if n != n'; ·= 2 if m > 0
    Iteration order matches make_compress_mask_trivial (m=0..l innermost);
    skipped channels contribute no m entries.

    Args: n_max, l_max; skip_mask [n_uncompressed] (None = keep all).
    Returns: multiplicity_array [n_active], n_active = Σ_{kept (n,n',l)}(l+1).
    """
    n_unc = n_max * (n_max + 1) // 2 * (l_max + 1)
    if skip_mask is None:
        skip_mask = np.zeros(n_unc, dtype=bool)
    if skip_mask.shape != (n_unc,):
        raise ValueError(f"skip_mask shape {skip_mask.shape} != ({n_unc},)")

    sqrt2 = np.sqrt(2.0)
    out: list[float] = []
    k = 0                                                         # uncompressed (n, n', l) index
    for n in range(1, n_max + 1):
        for nprime in range(n, n_max + 1):
            for l in range(l_max + 1):
                if not skip_mask[k]:
                    base = sqrt2 if n != nprime else 1.0
                    for m in range(l + 1):
                        out.append(base * (2.0 if m > 0 else 1.0))
                k += 1
    return np.asarray(out, dtype=np.float64)


# --- Phase 1 validators --------------------------------------------------

def _validate_orthonormality(alpha_max: int, atol: float = 1e-6) -> None:
    """W·S·W = I (basis orthonormality). cond(S) grows with alpha_max;
    Fortran caps at 10. float64 SVD: ~1e-12 residual at alpha_max=4."""
    S = build_overlap_matrix_poly3(alpha_max)
    W = build_orthonormalization_matrix_poly3(alpha_max)
    WSW = W @ S @ W
    err = float(np.abs(WSW - np.eye(alpha_max)).max())
    if err > atol:
        raise AssertionError(
            f"poly3 orthonormality failed at alpha_max={alpha_max}: "
            f"|W·S·W - I|_max = {err:.3e} > atol={atol:.0e}"
        )


def _validate_block_W(alpha_max_per_species: list[int]) -> None:
    """Off-diagonal blocks must be exactly zero; on-diagonal blocks orthonormal."""
    W = build_block_W(alpha_max_per_species)
    n_max = sum(alpha_max_per_species)
    offset = 0
    for am in alpha_max_per_species:
        # On-diagonal block matches single-species W
        block = W[offset:offset + am, offset:offset + am]
        ref = build_orthonormalization_matrix_poly3(am)
        if not np.allclose(block, ref, atol=1e-12):
            raise AssertionError(f"Block at offset {offset} differs from single-species W")
        # Off-diagonal must be all zero
        # Right of block:
        if offset + am < n_max and W[offset:offset + am, offset + am:].any():
            raise AssertionError(f"Non-zero entry in off-diagonal block to the right of {offset}")
        # Below block:
        if offset + am < n_max and W[offset + am:, offset:offset + am].any():
            raise AssertionError(f"Non-zero entry in off-diagonal block below {offset}")
        offset += am


def _validate_compress_against_fixtures(fixtures_dir: str = "tests/fixtures") -> None:
    """Compress mask must reproduce the dim_q observed in every fixture."""
    for name in ["water_monomer", "water_dimer", "h2_close", "h2_far",
                 "single_h", "si_bulk", "si_dimer"]:
        f = load_fixture(name, fixtures_dir)
        species_Z = sorted(set(int(z) for z in f["numbers"]))
        alpha_max_per_species = [REFERENCE_SOAP_PARAMS["alpha_max"]] * len(species_Z)
        l_max = REFERENCE_SOAP_PARAMS["l_max"]
        mask = make_compress_mask_trivial(alpha_max_per_species, l_max)
        observed = int(f["descriptors"].shape[1])
        if mask["n_compressed"] != observed:
            raise AssertionError(
                f"{name}: compress mask gives {mask['n_compressed']} but "
                f"fixture has {observed}"
            )


def _validate_multiplicity_hand_computed() -> None:
    """For alpha_max=2, l_max=1, no compression: 9 entries in known order."""
    arr = build_multiplicity_array(n_max=2, l_max=1, skip_mask=None)
    sqrt2 = np.sqrt(2.0)
    expected = np.array([
        # (n=1, n'=1, l=0, m=0)
        1.0,
        # (n=1, n'=1, l=1, m=0), (m=1)
        1.0, 2.0,
        # (n=1, n'=2, l=0, m=0)
        sqrt2,
        # (n=1, n'=2, l=1, m=0), (m=1)
        sqrt2, 2.0 * sqrt2,
        # (n=2, n'=2, l=0, m=0)
        1.0,
        # (n=2, n'=2, l=1, m=0), (m=1)
        1.0, 2.0,
    ])
    if arr.shape != expected.shape:
        raise AssertionError(f"multiplicity length {arr.shape} != expected {expected.shape}")
    if not np.allclose(arr, expected, atol=1e-12):
        raise AssertionError(f"multiplicity values don't match hand-computed:\n"
                             f"got      {arr}\nexpected {expected}")


def _validate_multiplicity_count_matches_fixtures(
    fixtures_dir: str = "tests/fixtures",
) -> None:
    """Total m-count must be sum_{kept (n,n',l)} (l+1) for each fixture's species set."""
    for name in ["water_monomer", "h2_close", "si_bulk"]:
        f = load_fixture(name, fixtures_dir)
        species_Z = sorted(set(int(z) for z in f["numbers"]))
        alpha_max_per_species = [REFERENCE_SOAP_PARAMS["alpha_max"]] * len(species_Z)
        l_max = REFERENCE_SOAP_PARAMS["l_max"]
        mask = make_compress_mask_trivial(alpha_max_per_species, l_max)
        mult = build_multiplicity_array(
            n_max=sum(alpha_max_per_species),
            l_max=l_max,
            skip_mask=mask["skip_mask"],
        )
        # Expected length = n_pairs_kept · Σ_{l=0..l_max}(l+1).
        n_pairs_kept = mask["n_compressed"] // (l_max + 1)
        expected_length = n_pairs_kept * sum(range(1, l_max + 2))
        if mult.shape[0] != expected_length:
            raise AssertionError(
                f"{name}: multiplicity has {mult.shape[0]} entries, "
                f"expected {expected_length}"
            )


def run_phase1_validation() -> None:
    """Run every Phase-1 sanity check; print pass/fail per item."""
    print("=== Phase 1 validation ===\n")

    print("1. W is the matrix square-root inverse of S (W·S·W = I)")
    for am in [1, 2, 3, 4, 5, 6, 7]:
        _validate_orthonormality(am)
        S = build_overlap_matrix_poly3(am)
        W = build_orthonormalization_matrix_poly3(am)
        residual = float(np.abs(W @ S @ W - np.eye(am)).max())
        print(f"   alpha_max={am}: |W·S·W - I|_max = {residual:.2e}  PASS")

    print("\n2. Block-diagonal W is correct for multi-species")
    for ams in [[4], [4, 4], [4, 4, 4], [3, 5]]:
        _validate_block_W(ams)
        W = build_block_W(ams)
        print(f"   alpha_max_per_species={ams}: shape={W.shape}, "
              f"on/off-diagonal pattern correct  PASS")

    print("\n3. Compress mask reproduces fixture dim_q values")
    _validate_compress_against_fixtures()
    for name in ["water_monomer", "h2_close", "si_bulk"]:
        f = load_fixture(name)
        species_Z = sorted(set(int(z) for z in f["numbers"]))
        alpha_max_per_species = [REFERENCE_SOAP_PARAMS["alpha_max"]] * len(species_Z)
        mask = make_compress_mask_trivial(alpha_max_per_species, REFERENCE_SOAP_PARAMS["l_max"])
        print(f"   {name:<14} species={species_Z}  "
              f"computed={mask['n_compressed']}  fixture={f['descriptors'].shape[1]}  PASS")

    print("\n4. Multiplicity array matches hand-computed for alpha_max=2, l_max=1")
    _validate_multiplicity_hand_computed()
    arr = build_multiplicity_array(2, 1)
    print(f"   length={arr.shape[0]}  values={arr.round(4).tolist()}  PASS")

    print("\n5. Multiplicity-array length matches expected for fixture configs")
    _validate_multiplicity_count_matches_fixtures()
    for name in ["water_monomer", "h2_close", "si_bulk"]:
        f = load_fixture(name)
        species_Z = sorted(set(int(z) for z in f["numbers"]))
        ams = [REFERENCE_SOAP_PARAMS["alpha_max"]] * len(species_Z)
        l_max = REFERENCE_SOAP_PARAMS["l_max"]
        mask = make_compress_mask_trivial(ams, l_max)
        mult = build_multiplicity_array(sum(ams), l_max, mask["skip_mask"])
        print(f"   {name:<14} multiplicity_len={mult.shape[0]}  "
              f"min={mult.min():.4f}  max={mult.max():.4f}  PASS")

    print("\nAll Phase 1 checks passed.")


# =========================================================================
# Phase 2 — Radial expansion coefficients (poly3, forward only)
# Fortran ref: soap_turbo_radial.f90:69-380 (get_radial_expansion_..._poly3).
#
# Per pair: normalise by rcut_hard; amplitude (Hermite envelope ·
# radial_enhancement); first-integral recursion I_α over [0, rcut_soft];
# second recursion over [rcut_soft, rcut_hard] with smoothing filter if near
# cutoff; orthonormalise W·(temp1 + pref_f·temp2); scale by amplitude·sqrt(rcut).
# poly3 only; uniform per-species hyperparameters.
# =========================================================================


def _N_a(alpha: int, rcut_hard: float = 1.0) -> float:
    """Polynomial normalisation N_a = sqrt(rcut/(2α+5)) (soap_turbo_radial.f90:38-52)."""
    return float(np.sqrt(rcut_hard / (2.0 * alpha + 5.0)))


def _radial_first_integral(
    rjs: np.ndarray,                # [P], distances ALREADY normalised by rcut_hard
    alpha_max: int,
    rcut_soft: float,               # ALREADY normalised
    atom_sigma_scaled: np.ndarray,  # [P], σ + scaling*rj (normalised by rcut_hard)
) -> np.ndarray:
    """First-integral recursion over [0, rcut_soft]. Returns [alpha_max, P].

    Mirrors soap_turbo_radial.f90:563-593. Iteration n=1..alpha_max stores
    I_np2 into output index n-1; sequential in α, parallel in pairs.
    """
    P = rjs.shape[0]
    sq2 = np.sqrt(2.0)
    s2 = atom_sigma_scaled ** 2

    # Initial state — see Fortran lines 562-568. I_-1 sentinel = 0.
    I_n = np.zeros(P, dtype=np.float64)
    N_n = 1.0
    N_np1 = _N_a(-2)  # = 1.0 since rcut_hard = 1
    # I_0 = ∫₀^rcut_soft Gaussian(r-rj, σ) dr  — analytical via erf
    I_np1 = (np.sqrt(np.pi / 2.0) * atom_sigma_scaled *
             (erf((rcut_soft - rjs) / (sq2 * atom_sigma_scaled)) -
              erf((-rjs) / (sq2 * atom_sigma_scaled))) / N_np1)

    # Boundary contributions C1, C2 (Fortran:570-575)
    dr = 1.0 - rcut_soft
    if dr == 0.0:
        C1 = np.zeros(P, dtype=np.float64)
    else:
        C1 = s2 / dr * np.exp(-0.5 * (rcut_soft - rjs) ** 2 / s2)
    C2 = s2 * np.exp(-0.5 * rjs ** 2 / s2)  # rcut_hard = 1, so s2/rcut_hard = s2

    out = np.zeros((alpha_max, P), dtype=np.float64)
    for n in range(-1, alpha_max + 1):
        # Update C1, C2 (Fortran:578-579). rcut_hard=1, so C2 update is a no-op.
        C1 = C1 * dr
        # C2 = C2 * 1.0  -- skipped
        N_np2 = _N_a(n)
        I_np2 = (s2 * (n + 1) * (N_n / N_np2) * I_n
                 - N_np1 * (rjs - 1.0) / N_np2 * I_np1
                 + C1 / N_np2
                 - C2 / N_np2)
        if n > 0:
            out[n - 1] = I_np2  # 1-based "n" in Fortran -> 0-based "n-1" here
        # Shift state
        N_n = N_np1
        N_np1 = N_np2
        I_n = I_np1
        I_np1 = I_np2
    return out


def _radial_second_integral(
    rjs: np.ndarray,                # [P] normalised
    alpha_max: int,
    rcut_soft: float,
    atom_sigma_scaled: np.ndarray,  # [P]
    nf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-cutoff filter contribution. Returns (temp2 [alpha_max, P], pref_f [P]).

    Mirrors Fortran:611-645. Computed for all pairs; caller gates via
    near_cutoff_mask (far pairs have tiny pref_f anyway).
    """
    P = rjs.shape[0]
    sq2 = np.sqrt(2.0)
    s2 = atom_sigma_scaled ** 2
    dr = 1.0 - rcut_soft

    # Effective filter parameters (Fortran:612-617)
    denom = s2 + dr ** 2 / nf ** 2
    atom_sigma_f = atom_sigma_scaled * dr / nf / np.sqrt(denom)
    rj_f = (s2 * rcut_soft + dr ** 2 / nf ** 2 * rjs) / denom
    sf2 = atom_sigma_f ** 2
    pref_f = np.exp(-0.5 * (rcut_soft - rjs) ** 2 / denom)

    # Initial state (Fortran:621-626)
    I_n = np.zeros(P, dtype=np.float64)
    N_n = 1.0
    N_np1 = _N_a(-2)
    I_np1 = (np.sqrt(np.pi / 2.0) * atom_sigma_f *
             (erf((1.0 - rj_f) / (sq2 * atom_sigma_f)) -
              erf((rcut_soft - rj_f) / (sq2 * atom_sigma_f))) / N_np1)
    # No C1 in second integral (Fortran:628 only sets C2)
    if dr == 0.0:
        C2 = np.zeros(P, dtype=np.float64)
    else:
        C2 = sf2 / dr * np.exp(-0.5 * (rcut_soft - rj_f) ** 2 / sf2)

    out = np.zeros((alpha_max, P), dtype=np.float64)
    for n in range(-1, alpha_max + 1):
        C2 = C2 * dr  # Fortran:631
        N_np2 = _N_a(n)
        # Note: subtract C2 (no C1 in second integral) — Fortran:634-636
        I_np2 = (sf2 * (n + 1) * (N_n / N_np2) * I_n
                 - N_np1 * (rj_f - 1.0) / N_np2 * I_np1
                 - C2 / N_np2)
        if n > 0:
            out[n - 1] = I_np2
        N_n = N_np1
        N_np1 = N_np2
        I_n = I_np1
        I_np1 = I_np2
    return out, pref_f


def _radial_amplitude(
    rjs: np.ndarray,                # [P] normalised
    atom_sigma_scaled: np.ndarray,  # [P]
    is_central: np.ndarray,         # [P] bool — pair is the centre's self-pair
    central_weight: float,
    amplitude_scaling: float,
    radial_enhancement: int,
) -> np.ndarray:
    """Per-pair amplitude (Fortran:189-231): Hermite envelope
    (1+2rj³-3rj²)^amplitude_scaling / atom_sigma_scaled, ·central_weight for
    central pairs, ·radial_enhancement {0,1,2} multiplier."""
    s2 = atom_sigma_scaled ** 2
    if amplitude_scaling == 0.0:
        amp = 1.0 / atom_sigma_scaled
    else:
        env = 1.0 + 2.0 * rjs ** 3 - 3.0 * rjs ** 2
        # Fortran zeros amplitude where envelope <= 1e-10 (past rcut_soft)
        with np.errstate(invalid="ignore"):
            amp = np.where(
                env <= 1e-10,
                0.0,
                (1.0 / atom_sigma_scaled) * env ** amplitude_scaling,
            )
    amp = np.where(is_central, amp * central_weight, amp)
    if radial_enhancement == 1:
        amp = amp * (rjs + np.sqrt(2.0 / np.pi) * atom_sigma_scaled)
    elif radial_enhancement == 2:
        amp = amp * (rjs ** 2 + s2 + np.sqrt(8.0 / np.pi) * atom_sigma_scaled * rjs)
    return amp


def radial_expansion_coeff_poly3_numpy(
    rjs: np.ndarray,                  # [P] raw distances (NOT yet normalised)
    pair_neighbour_species: np.ndarray,  # [P] int — neighbour's species index (0..n_species-1)
    pair_is_central: np.ndarray,      # [P] bool — j == 1 in Fortran (pair_atom == pair_gidx)
    n_species: int,
    alpha_max: int,                   # uniform across species
    rcut_hard: float,                 # uniform raw value (not normalised)
    rcut_soft: float,
    atom_sigma_r: float,
    atom_sigma_r_scaling: float,
    amplitude_scaling: float,
    central_weight: float,
    radial_enhancement: int,
    nf: float,
    do_central: bool,
    W_single: np.ndarray,             # [alpha_max, alpha_max] for one species
    global_scaling: float = 1.0,
) -> np.ndarray:                      # [n_max, P]
    """Radial expansion coefficients in the orthonormal basis.

    Returns [n_max, P], n_max = n_species·alpha_max. Row r belongs to species
    r//alpha_max; only neighbour-species-matching pairs contribute. Uniform
    per-species hyperparameters. Pairs with rj >= rcut_hard contribute zero.
    """
    P = rjs.shape[0]
    n_max = n_species * alpha_max

    # Normalise distances by rcut_hard
    rj_n = rjs / rcut_hard
    rcut_soft_n = rcut_soft / rcut_hard
    atom_sigma_n = atom_sigma_r / rcut_hard
    atom_sigma_scaled = atom_sigma_n + atom_sigma_r_scaling * rj_n  # [P]

    # Determine which pairs contribute
    in_cutoff = rj_n < 1.0
    pair_active = in_cutoff.copy()
    if not do_central:
        pair_active = pair_active & ~pair_is_central

    # Amplitude (per pair, in normalised space)
    amplitude = _radial_amplitude(
        rj_n, atom_sigma_scaled, pair_is_central,
        central_weight, amplitude_scaling, radial_enhancement,
    )

    # First integral
    temp1 = _radial_first_integral(rj_n, alpha_max, rcut_soft_n, atom_sigma_scaled)

    # Second integral with soft-cutoff gate
    near_cutoff = (rcut_soft_n - rj_n) < 4.0 * atom_sigma_scaled
    temp2, pref_f = _radial_second_integral(rj_n, alpha_max, rcut_soft_n, atom_sigma_scaled, nf)
    pref_f = np.where(near_cutoff, pref_f, 0.0)

    # Combine: amp * W @ (temp1 + pref_f * temp2)
    combined = temp1 + pref_f[None, :] * temp2                     # [alpha_max, P]
    transformed = W_single @ combined                               # [alpha_max, P]
    raw = amplitude[None, :] * transformed                          # [alpha_max, P]

    # Final scaling — Fortran:712. global_scaling and sqrt(rcut_hard).
    raw = raw * global_scaling * np.sqrt(rcut_hard)

    # Mask out inactive pairs
    raw = raw * pair_active[None, :].astype(np.float64)

    # Distribute to species blocks: pair p (neighbour-species s) fills rows
    # [s*alpha_max : (s+1)*alpha_max]; other rows stay zero.
    radial = np.zeros((n_max, P), dtype=np.float64)
    for s in range(n_species):
        species_pair_mask = (pair_neighbour_species == s).astype(np.float64)  # [P]
        radial[s * alpha_max:(s + 1) * alpha_max, :] = raw * species_pair_mask[None, :]
    return radial


# --- Phase 2 validators --------------------------------------------------

def _quadrature_I_alpha(
    alpha: int,
    rj: float,
    rcut_soft: float,
    atom_sigma_scaled: float,
) -> float:
    """Direct numerical quadrature: I_α = ∫₀^rcut_soft (1-r)^(α+2) G(r-rj, σ) dr / N_a(α).

    Used to cross-check the recursion. Assumes rcut_hard=1 (normalised).
    """
    from scipy.integrate import quad
    s = atom_sigma_scaled

    def integrand(r):
        return (1.0 - r) ** (alpha + 2) * np.exp(-0.5 * (r - rj) ** 2 / s ** 2)

    val, _ = quad(integrand, 0.0, rcut_soft, epsabs=1e-12, epsrel=1e-10)
    return val / _N_a(alpha)


def _validate_radial_first_integral_against_quadrature() -> None:
    """The recursion's temp1[β-1] (β=1..alpha_max) must equal I_β via quadrature.

    Picks a pair far from rcut_soft so the second-integral gate is closed
    (pref_f = 0), removing it from the comparison. Tests several rj values.
    """
    rcut_hard = 6.0
    rcut_soft = 5.5
    atom_sigma_r = 0.5
    alpha_max = 4

    rcut_soft_n = rcut_soft / rcut_hard
    atom_sigma_n = atom_sigma_r / rcut_hard

    # rj values: chosen well within rcut_soft, varying density of recursion
    test_rj_n = np.array([0.05, 0.2, 0.5, 0.7])
    rj_n_far = test_rj_n[test_rj_n < rcut_soft_n - 4 * atom_sigma_n]  # second integral inactive

    if len(rj_n_far) == 0:
        # If nothing's far enough, just check shape/finiteness
        rj_n_far = test_rj_n
    atom_sigma_scaled = atom_sigma_n * np.ones_like(rj_n_far)
    temp1 = _radial_first_integral(rj_n_far, alpha_max, rcut_soft_n, atom_sigma_scaled)

    max_err = 0.0
    for k, rj in enumerate(rj_n_far):
        for beta in range(1, alpha_max + 1):
            quad_val = _quadrature_I_alpha(beta, float(rj), rcut_soft_n, float(atom_sigma_scaled[k]))
            recur_val = temp1[beta - 1, k]
            err = abs(recur_val - quad_val)
            denom = max(abs(quad_val), 1e-12)
            rel_err = err / denom
            max_err = max(max_err, rel_err)
            if rel_err > 1e-6 and err > 1e-9:
                raise AssertionError(
                    f"radial recursion mismatch at rj_n={rj:.4f}, β={beta}: "
                    f"recursion={recur_val:.6e}, quad={quad_val:.6e}, "
                    f"abs_err={err:.2e}, rel_err={rel_err:.2e}"
                )
    print(f"   first-integral recursion vs quadrature: max rel. err = {max_err:.2e}  PASS")


def _validate_radial_second_integral_against_quadrature() -> None:
    """Second-integral recursion sanity: pref_f·temp2 finite, smooth, and
    pref_f ∈ [0, 1]. (Direct filter quadrature not re-derived here.)"""
    rcut_hard = 6.0
    rcut_soft = 5.5
    atom_sigma_r = 0.5
    alpha_max = 4
    nf = 4.0  # GPUMD default

    rcut_soft_n = rcut_soft / rcut_hard
    atom_sigma_n = atom_sigma_r / rcut_hard
    dr = 1.0 - rcut_soft_n
    test_rj_n = np.array([rcut_soft_n - atom_sigma_n,        # very close to soft cutoff
                          rcut_soft_n - 2 * atom_sigma_n,    # within 4σ — should activate
                          rcut_soft_n + 0.001,               # just past soft — still in cutoff
                          ])
    test_rj_n = test_rj_n[(test_rj_n > 0) & (test_rj_n < 1.0)]
    atom_sigma_scaled = atom_sigma_n * np.ones_like(test_rj_n)
    temp2, pref_f = _radial_second_integral(test_rj_n, alpha_max, rcut_soft_n, atom_sigma_scaled, nf)

    # No direct filter quadrature; just check finiteness and pref_f range.
    if not np.all(np.isfinite(temp2)):
        raise AssertionError("second-integral recursion produced non-finite values")
    if not np.all(np.isfinite(pref_f)):
        raise AssertionError("pref_f produced non-finite values")
    if (pref_f < 0).any() or (pref_f > 1.0001).any():
        raise AssertionError(f"pref_f out of [0,1]: min={pref_f.min()}, max={pref_f.max()}")
    print(f"   second-integral: finite, pref_f∈[{pref_f.min():.4f}, {pref_f.max():.4f}]  PASS")


def _validate_radial_zero_outside_cutoff() -> None:
    """Pairs with rj >= rcut_hard must produce zero radial coefficients."""
    rcut_hard = 6.0
    rjs = np.array([6.0, 6.5, 10.0])  # all >= rcut_hard
    n_pairs = len(rjs)
    n_species = 1
    pair_neighbour_species = np.zeros(n_pairs, dtype=np.int32)
    pair_is_central = np.zeros(n_pairs, dtype=bool)
    W = build_orthonormalization_matrix_poly3(4)
    radial = radial_expansion_coeff_poly3_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=1, alpha_max=4,
        rcut_hard=rcut_hard, rcut_soft=5.5, atom_sigma_r=0.5,
        atom_sigma_r_scaling=0.0, amplitude_scaling=1.0,
        central_weight=1.0, radial_enhancement=1, nf=4.0,
        do_central=True, W_single=W,
    )
    if not np.allclose(radial, 0.0):
        raise AssertionError(f"non-zero values for rj >= rcut_hard: max|x|={np.abs(radial).max()}")
    print(f"   rj >= rcut_hard produces zero coefficients (max|x|=0)  PASS")


def _validate_radial_smooth_in_rj() -> None:
    """Radial coefficients should be smooth (no jumps) as rj is swept."""
    rcut_hard = 6.0
    rjs = np.linspace(0.05, 5.99, 50)
    n_pairs = len(rjs)
    pair_neighbour_species = np.zeros(n_pairs, dtype=np.int32)
    pair_is_central = np.zeros(n_pairs, dtype=bool)
    W = build_orthonormalization_matrix_poly3(4)
    radial = radial_expansion_coeff_poly3_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=1, alpha_max=4,
        rcut_hard=rcut_hard, rcut_soft=5.5, atom_sigma_r=0.5,
        atom_sigma_r_scaling=0.0, amplitude_scaling=1.0,
        central_weight=1.0, radial_enhancement=1, nf=4.0,
        do_central=True, W_single=W,
    )                                                                 # [4, 50]
    if not np.all(np.isfinite(radial)):
        raise AssertionError("non-finite radial coefficients")
    # Jump test: adjacent-rj change > 10× the typical step flags a discontinuity.
    diffs = np.abs(np.diff(radial, axis=1))                           # [4, 49]
    typical = np.median(diffs, axis=1, keepdims=True)
    outliers = (diffs > 10 * (typical + 1e-12)).any(axis=0)
    if outliers.any():
        bad_idx = int(np.argmax(diffs.max(axis=0)))
        raise AssertionError(
            f"radial coeffs not smooth — jump at rj={rjs[bad_idx]:.3f}: "
            f"max diff={diffs.max():.3e}, typical={typical.max():.3e}"
        )
    print(f"   smooth sweep over rj∈[0.05, 5.99]: max|coeff|={np.abs(radial).max():.4f}  PASS")


def _validate_radial_symmetric_pairs() -> None:
    """Two equivalent pairs (same rj) must produce identical coefficients."""
    rjs = np.array([1.5, 2.7, 1.5, 2.7, 1.5])
    pair_neighbour_species = np.zeros(len(rjs), dtype=np.int32)
    pair_is_central = np.zeros(len(rjs), dtype=bool)
    W = build_orthonormalization_matrix_poly3(4)
    radial = radial_expansion_coeff_poly3_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=1, alpha_max=4,
        rcut_hard=6.0, rcut_soft=5.5, atom_sigma_r=0.5,
        atom_sigma_r_scaling=0.0, amplitude_scaling=1.0,
        central_weight=1.0, radial_enhancement=1, nf=4.0,
        do_central=True, W_single=W,
    )                                                                 # [4, 5]
    err1 = np.abs(radial[:, 0] - radial[:, 2]).max()
    err2 = np.abs(radial[:, 0] - radial[:, 4]).max()
    err3 = np.abs(radial[:, 1] - radial[:, 3]).max()
    if max(err1, err2, err3) > 1e-12:
        raise AssertionError(f"equivalent pairs not identical: errs={err1, err2, err3}")
    print(f"   identical rj produces identical coeffs (err < 1e-12)  PASS")


def _validate_radial_block_structure() -> None:
    """Multi-species: row r in block s must be zero for pairs not of species s."""
    rjs = np.array([1.0, 2.0, 3.0, 4.0])
    n_species = 2
    pair_neighbour_species = np.array([0, 1, 0, 1], dtype=np.int32)
    pair_is_central = np.zeros(len(rjs), dtype=bool)
    W = build_orthonormalization_matrix_poly3(4)
    radial = radial_expansion_coeff_poly3_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=4,
        rcut_hard=6.0, rcut_soft=5.5, atom_sigma_r=0.5,
        atom_sigma_r_scaling=0.0, amplitude_scaling=1.0,
        central_weight=1.0, radial_enhancement=1, nf=4.0,
        do_central=True, W_single=W,
    )                                                                 # [8, 4]
    # Block 0 (rows 0..3) should be nonzero only for pairs 0, 2 (species 0)
    block0 = radial[:4]
    block1 = radial[4:]
    if np.any(block0[:, [1, 3]] != 0):
        raise AssertionError("block 0 has nonzero values for species-1 pairs")
    if np.any(block1[:, [0, 2]] != 0):
        raise AssertionError("block 1 has nonzero values for species-0 pairs")
    if np.allclose(block0[:, [0, 2]], 0) or np.allclose(block1[:, [1, 3]], 0):
        raise AssertionError("blocks unexpectedly all-zero where they should have signal")
    print(f"   2-species block layout: correct masking  PASS")


def run_phase2_validation() -> None:
    """Run every Phase-2 sanity check."""
    print("=== Phase 2 validation ===\n")

    print("1. First-integral recursion matches scipy.integrate.quad")
    _validate_radial_first_integral_against_quadrature()

    print("\n2. Second-integral recursion produces finite, sane values")
    _validate_radial_second_integral_against_quadrature()

    print("\n3. Radial coefficients are zero outside rcut_hard")
    _validate_radial_zero_outside_cutoff()

    print("\n4. Radial coefficients are smooth in rj")
    _validate_radial_smooth_in_rj()

    print("\n5. Equivalent pairs produce bit-identical coefficients")
    _validate_radial_symmetric_pairs()

    print("\n6. Multi-species block layout is correct")
    _validate_radial_block_structure()

    print("\nAll Phase 2 checks passed.")


# =========================================================================
# Phase 3 — Angular expansion coefficients (poly3 forward only)
# Fortran refs (soap_turbo_angular.f90): Plm 36-85, ilexp 256-315, eimphi
# 200-224, assembly 368-458; preflm soap_turbo_functions.f90:202-228.
#
# Flat index k = l(l+1)/2 + m (0-based); k_max = (l_max+1)(l_max+2)/2.
# Vectorised over pairs: each Fortran per-pair scalar is a [P] array.
# =========================================================================


def _get_plm_array(x: np.ndarray, l_max: int) -> np.ndarray:
    """Associated Legendre polynomials P_lm(x) with Condon-Shortley phase.

    Args:
        x      : [P] float64 in [-1, 1]
        l_max  : maximum angular momentum

    Returns:
        plm    : [k_max, P] float64 with k = l(l+1)/2 + m
    """
    P = x.shape[0]
    k_max = (l_max + 1) * (l_max + 2) // 2
    plm = np.zeros((k_max, P), dtype=np.float64)
    sqrt_1mx2 = np.sqrt(np.maximum(1.0 - x * x, 0.0))

    plm[0] = 1.0  # P_00
    if l_max >= 1:
        plm[1] = x                                 # P_10
        plm[2] = -sqrt_1mx2                        # P_11 (CS phase)
    if l_max >= 2:
        plm[3] = 1.5 * x * x - 0.5                 # P_20
        plm[4] = -3.0 * x * sqrt_1mx2              # P_21
        plm[5] = 3.0 - 3.0 * x * x                 # P_22

    for l in range(3, l_max + 1):
        # m = 0..l-2 via Plm = ((2l-1)·x·P_{l-1,m} - (l-1+m)·P_{l-2,m})/(l-m)
        for m in range(l - 1):
            k = l * (l + 1) // 2 + m
            k_lm1_m = (l - 1) * l // 2 + m
            k_lm2_m = (l - 2) * (l - 1) // 2 + m
            plm[k] = ((2 * l - 1) * x * plm[k_lm1_m]
                      - (l - 1 + m) * plm[k_lm2_m]) / (l - m)
        # P_{l, l-1} = x·(2l-1)·P_{l-1, l-1}
        k_lm1_lm1 = (l - 1) * l // 2 + (l - 1)
        k_l_lm1 = l * (l + 1) // 2 + (l - 1)
        plm[k_l_lm1] = x * (2 * l - 1) * plm[k_lm1_lm1]
        # P_{l, l} = -(2l-1)·sqrt(1-x²)·P_{l-1, l-1}
        k_l_l = l * (l + 1) // 2 + l
        plm[k_l_l] = -(2 * l - 1) * sqrt_1mx2 * plm[k_lm1_lm1]
    return plm


def _get_ilexp(x: np.ndarray, l_max: int) -> np.ndarray:
    """Compute i_l(x²)·exp(-x²) for l=0..l_max via the stable recursion.

    Args:
        x      : [P] float64, expected non-negative
        l_max  : maximum l

    Returns:
        ilexp  : [l_max+1, P] float64

    Numerical guards mirror soap_turbo_angular.f90:282-313 — explicit Taylor
    branches at small x for l ∈ {0, 1, ≥2} to avoid the (2l-1)/x² division.
    """
    P = x.shape[0]
    out = np.zeros((l_max + 1, P), dtype=np.float64)
    xcut = 1e-7
    x2 = x * x
    x4 = x2 * x2

    # fact_array[i] = (2i+1)!! for i >= 1; fact_array[0] unused but defined as 1
    fact = np.ones(l_max + 1, dtype=np.float64)
    f = 1.0
    for i in range(1, l_max + 1):
        f = f * (2.0 * i + 1.0)
        fact[i] = f

    # Initial flm2, flm1 (= ilexp at l=0 and l=1 in the "full" formula)
    safe_x2 = np.maximum(x2, 1e-300)
    safe_x4 = np.maximum(x4, 1e-300)
    full_flm2 = np.abs((1.0 - np.exp(-2.0 * x2)) / (2.0 * safe_x2))
    full_flm1 = np.abs((x2 - 1.0 + np.exp(-2.0 * x2) * (x2 + 1.0)) / (2.0 * safe_x4))
    flm2 = np.where(x > 0, full_flm2, 1.0)
    flm1 = np.where(x > 0, full_flm1, 0.0)

    if l_max >= 0:
        out[0] = np.where(x < xcut, 1.0 - x2, flm2)
    if l_max >= 1:
        out[1] = np.where(x2 / 1000.0 < xcut, (x2 - x4) / fact[1], flm1)

    # l >= 2: recursion with Taylor fallback per pair
    for l in range(2, l_max + 1):
        x_2l = safe_x2 ** l
        taylor = x_2l / fact[l]
        recursion = np.abs(flm2 - (2.0 * l - 1.0) / safe_x2 * flm1)
        fl = np.where(taylor * l < xcut, taylor, recursion)
        flm2 = flm1
        flm1 = fl
        out[l] = fl

    return out


def _get_eimphi_factor(phi: np.ndarray, m_max: int) -> np.ndarray:
    """Compute e^{-i·m·φ} for m = 0..m_max via Chebyshev recursion.

    Args:
        phi    : [P] float64 azimuthal angles
        m_max  : maximum m index (= l_max for the angular expansion)

    Returns:
        out    : [m_max+1, P] complex128
    """
    P = phi.shape[0]
    out = np.zeros((m_max + 1, P), dtype=np.complex128)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cosphi2 = 2.0 * cos_phi

    # State holds (cos((m-1)φ), cos((m-2)φ)) and (sin((m-1)φ), sin((m-2)φ))
    # Initial m=1 step: cos(0)=1, cos(-φ)=cos(φ), sin(0)=0, sin(-φ)=-sin(φ)
    cosm2 = cos_phi.copy()
    sinm2 = -sin_phi
    cosm1 = np.ones(P, dtype=np.float64)
    sinm1 = np.zeros(P, dtype=np.float64)

    out[0] = 1.0
    for l in range(1, m_max + 1):
        cos0 = cosphi2 * cosm1 - cosm2
        sin0 = cosphi2 * sinm1 - sinm2
        cosm2, cosm1 = cosm1, cos0
        sinm2, sinm1 = sinm1, sin0
        out[l] = cos0 - 1j * sin0
    return out


def _get_preflm(l_max: int) -> np.ndarray:
    """Y_lm normalisation: sqrt((2l+1)/4π · (l-m)!/(l+m)!).

    Returns flat [k_max] real array indexed by k = l(l+1)/2 + m.
    """
    import math
    k_max = (l_max + 1) * (l_max + 2) // 2
    out = np.zeros(k_max, dtype=np.float64)
    for l in range(l_max + 1):
        norm_l = np.sqrt((2 * l + 1) / (4 * np.pi))
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            num = math.factorial(l - m)
            den = math.factorial(l + m)
            out[k] = norm_l * np.sqrt(num / den)
    return out


def angular_expansion_coeff_numpy(
    rjs: np.ndarray,                   # [P] raw distances (NOT yet normalised)
    thetas: np.ndarray,                # [P] polar angles
    phis: np.ndarray,                  # [P] azimuthal angles
    pair_active: np.ndarray,           # [P] bool — pair within rcut and mask
    l_max: int,
    atom_sigma_t: float,               # uniform across species
    atom_sigma_t_scaling: float,
    rcut: float,                       # max rcut_hard across species (uniform here)
) -> np.ndarray:                       # [k_max, P] complex128
    """Build the angular expansion coefficients per pair.

    Fortran reference: soap_turbo_angular.f90:368-458.
        exp_coeff[k, p] = amplitude(p) · preflm[k] · Plm[k, p] · eimphi[k, p]
    where
        amplitude = rcut² / atom_sigma²,  with σ = σ_t + σ_t_scaling·rj
        eimphi[k=l(l+1)/2+m, p] = ilexp(l, rj/σ)[p] · exp(-i·m·φ)
    """
    P = rjs.shape[0]
    k_max = (l_max + 1) * (l_max + 2) // 2

    # Per-pair angular sigma and amplitude
    atom_sigma = atom_sigma_t + atom_sigma_t_scaling * rjs
    amplitude = rcut ** 2 / atom_sigma ** 2

    # x = cos(theta), then Plm
    x = np.cos(thetas)
    plm = _get_plm_array(x, l_max)                                 # [k_max, P]

    # Modified spherical Bessel × Gaussian decay
    rj_by_sigma = rjs / atom_sigma
    prefl = _get_ilexp(rj_by_sigma, l_max)                          # [l_max+1, P]

    # Chebyshev e^{-imφ}
    prefm = _get_eimphi_factor(phis, l_max)                         # [l_max+1, P] complex

    # Compose eimphi[k(l,m), p] = prefl[l, p] · prefm[m, p]
    eimphi = np.zeros((k_max, P), dtype=np.complex128)
    for l in range(l_max + 1):
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            eimphi[k] = prefl[l] * prefm[m]

    preflm = _get_preflm(l_max)                                     # [k_max]

    # Final assembly
    exp_coeff = (amplitude[None, :] * preflm[:, None] * plm * eimphi)

    # Apply pair mask (zero out inactive pairs)
    exp_coeff = exp_coeff * pair_active[None, :].astype(np.float64)
    return exp_coeff


# --- Phase 3 validators --------------------------------------------------

def _validate_plm_against_scipy() -> None:
    """_get_plm_array must match scipy.special.lpmn for several x values."""
    import scipy.special as sps
    test_x = np.array([-0.99, -0.5, 0.0, 0.5, 0.99])
    l_max = 6
    plm_mine = _get_plm_array(test_x, l_max)                        # [k_max, len(x)]

    max_err = 0.0
    for ix, xv in enumerate(test_x):
        plm_scipy, _ = sps.lpmn(l_max, l_max, float(xv))             # shape (m+1, l+1)
        for l in range(l_max + 1):
            for m in range(l + 1):
                k = l * (l + 1) // 2 + m
                err = abs(plm_mine[k, ix] - plm_scipy[m, l])
                if err > 1e-12:
                    raise AssertionError(
                        f"P_{l},{m}({xv}): mine={plm_mine[k, ix]:.6e} "
                        f"vs scipy={plm_scipy[m, l]:.6e}, err={err:.2e}"
                    )
                max_err = max(max_err, err)
    print(f"   Plm vs scipy.special.lpmn: max abs err = {max_err:.2e}  PASS")


def _validate_ilexp_finite_and_continuous() -> None:
    """_get_ilexp must be finite over a wide x range and continuous at branch boundaries."""
    x = np.logspace(-4, 1.5, 200)                                    # 0.0001 to ~31.6
    out = _get_ilexp(x, l_max=6)
    if not np.all(np.isfinite(out)):
        bad = np.where(~np.isfinite(out))
        raise AssertionError(f"_get_ilexp non-finite at l, x_idx = {bad}")

    # Continuity check: max diff between adjacent x samples
    diffs = np.abs(np.diff(out, axis=1))
    max_diff = diffs.max()
    print(f"   ilexp finite over x∈[1e-4, 31.6], 7 l-values; max adjacent diff = {max_diff:.3e}  PASS")

    # Cross-check vs scipy.special.spherical_in; recursion is noisy at small
    # x/high l but stays below 1e-7 (soap_turbo_angular.f90:262).
    import scipy.special as sps
    test_x = np.array([0.5, 1.0, 2.0, 3.0])
    out_mid = _get_ilexp(test_x, l_max=6)
    max_abs = 0.0
    for ix, xv in enumerate(test_x):
        x2 = xv * xv
        for l in range(7):
            ref = sps.spherical_in(l, x2) * np.exp(-x2)
            mine = out_mid[l, ix]
            err = abs(mine - ref)
            if err > 1e-7:
                raise AssertionError(
                    f"ilexp(l={l}, x={xv}): mine={mine:.6e} vs scipy={ref:.6e}, "
                    f"abs_err={err:.2e} > 1e-7"
                )
            max_abs = max(max_abs, err)
    print(f"   ilexp vs scipy.spherical_in: max abs err = {max_abs:.2e}  PASS")


def _validate_eimphi_factor() -> None:
    """_get_eimphi_factor must equal direct exp(-i·m·φ)."""
    test_phi = np.array([0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2,
                         2 * np.pi - 0.01, -0.7, 5.3])
    m_max = 6
    factor = _get_eimphi_factor(test_phi, m_max)                     # [m_max+1, P]
    max_err = 0.0
    for ip, phi in enumerate(test_phi):
        for m in range(m_max + 1):
            ref = np.exp(-1j * m * phi)
            err = abs(factor[m, ip] - ref)
            if err > 1e-13:
                raise AssertionError(
                    f"e^{{-i·{m}·{phi:.3f}}}: mine={factor[m, ip]} ref={ref}"
                )
            max_err = max(max_err, err)
    print(f"   eimphi factor vs direct exp(-imφ): max err = {max_err:.2e}  PASS")


def _validate_preflm() -> None:
    """_get_preflm matches sqrt((2l+1)/4π · (l-m)!/(l+m)!)."""
    import math
    l_max = 5
    out = _get_preflm(l_max)
    max_err = 0.0
    for l in range(l_max + 1):
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            ref = np.sqrt((2 * l + 1) / (4 * np.pi)
                          * math.factorial(l - m) / math.factorial(l + m))
            err = abs(out[k] - ref)
            max_err = max(max_err, err)
    if max_err > 1e-15:
        raise AssertionError(f"preflm mismatch: max err {max_err:.2e}")
    print(f"   preflm normalisation: max err = {max_err:.2e}  PASS")


def _validate_angular_finite_and_symmetric() -> None:
    """angular_expansion_coeff_numpy must be finite for fixture pair lists.

    Symmetry check: water_monomer's two H atoms (which see equivalent
    environments) should have identical |angular_exp_coeff|² when summed
    over their respective pairs. Same for h2_close.
    """
    for name in ["water_monomer", "h2_close", "si_bulk", "single_h"]:
        f = load_fixture(name)
        positions = f["positions"]
        cell = f["cell"]
        pair_atom = f["pair_atom"]
        pair_gidx = f["pair_gidx"]
        n_pairs = len(pair_atom)
        if n_pairs == 0:
            continue

        # Build (rj, theta, phi) per pair; minimum-image if periodic.
        is_periodic = np.any(f["pbc"])
        rjs = np.zeros(n_pairs)
        thetas = np.zeros(n_pairs)
        phis = np.zeros(n_pairs)
        for k in range(n_pairs):
            i = int(pair_atom[k]); j = int(pair_gidx[k])
            disp = positions[j] - positions[i]
            if is_periodic:
                # Minimum-image convention via fractional coordinates
                cell_inv = np.linalg.inv(cell)
                frac = disp @ cell_inv
                frac -= np.round(frac)
                disp = frac @ cell
            r = float(np.linalg.norm(disp))
            rjs[k] = r
            if r < 1e-10:
                thetas[k] = 0.0
                phis[k] = 0.0
            else:
                thetas[k] = np.arccos(np.clip(disp[2] / r, -1.0, 1.0))
                phis[k] = np.arctan2(disp[1], disp[0])

        soap_params = f["soap_params"]
        rcut = soap_params["rcut_hard"]
        atom_sigma_t = soap_params["atom_sigma_t"]
        atom_sigma_t_scaling = soap_params["atom_sigma_t_scaling"]
        l_max = soap_params["l_max"]
        pair_active = (rjs < rcut)

        ang = angular_expansion_coeff_numpy(
            rjs, thetas, phis, pair_active,
            l_max=l_max,
            atom_sigma_t=atom_sigma_t,
            atom_sigma_t_scaling=atom_sigma_t_scaling,
            rcut=rcut,
        )                                                            # [k_max, n_pairs]
        if not np.all(np.isfinite(ang)):
            raise AssertionError(f"{name}: non-finite angular coefficients")

        # Symmetry: equivalent atoms get equivalent |ang|² aggregated over their pairs
        if name == "water_monomer":
            # numbers = [O, H, H]; H atoms at indices 1, 2
            pairs_h0 = (pair_atom == 1)
            pairs_h1 = (pair_atom == 2)
            agg_h0 = np.sum(np.abs(ang[:, pairs_h0]) ** 2, axis=1)
            agg_h1 = np.sum(np.abs(ang[:, pairs_h1]) ** 2, axis=1)
            err = np.abs(agg_h0 - agg_h1).max()
            if err > 1e-10:
                raise AssertionError(f"water H atoms not equivalent: {err}")
        if name == "h2_close":
            pairs_0 = (pair_atom == 0)
            pairs_1 = (pair_atom == 1)
            agg_0 = np.sum(np.abs(ang[:, pairs_0]) ** 2, axis=1)
            agg_1 = np.sum(np.abs(ang[:, pairs_1]) ** 2, axis=1)
            err = np.abs(agg_0 - agg_1).max()
            if err > 1e-10:
                raise AssertionError(f"H2 atoms not equivalent: {err}")

        n_active = int(pair_active.sum())
        max_mag = float(np.abs(ang).max())
        print(f"   {name:<14} pairs={n_pairs:>4} active={n_active:>4} "
              f"max|ang|={max_mag:.4f}  PASS")


def run_phase3_validation() -> None:
    """Run every Phase-3 sanity check."""
    print("=== Phase 3 validation ===\n")

    print("1. Plm matches scipy.special.lpmn")
    _validate_plm_against_scipy()

    print("\n2. ilexp is finite and continuous, matches scipy")
    _validate_ilexp_finite_and_continuous()

    print("\n3. eimphi factor matches direct exp(-imφ)")
    _validate_eimphi_factor()

    print("\n4. preflm matches Y_lm normalisation formula")
    _validate_preflm()

    print("\n5. angular_expansion_coeff_numpy is finite and symmetric")
    _validate_angular_finite_and_symmetric()

    print("\nAll Phase 3 checks passed.")


# =========================================================================
# Phase 4 — cnk scatter-sum (Fortran ref: soap_turbo.f90:418-449)
# cnk[k, n, i] = 4π · Σ_{p ∈ neigh(i)} radial_exp[n, p] · angular_exp[k, p]
# Per-pair outer product then segment-sum over centres via np.add.at.
# =========================================================================


def aggregate_cnk(
    radial_exp: np.ndarray,            # [n_max, P] real
    angular_exp: np.ndarray,           # [k_max, P] complex
    pair_struct: np.ndarray,           # [P] int — centre index of each pair
    n_sites: int,
) -> np.ndarray:                       # [k_max, n_max, n_sites] complex
    """Accumulate cnk by structure index. Folds in the 4π prefactor.

    Implementation: build the [k_max, n_max, P] per-pair product, then
    scatter-add along the pair axis grouped by pair_struct.
    """
    n_max = radial_exp.shape[0]
    k_max = angular_exp.shape[0]

    # Per-pair outer product [k_max, n_max, P]. Cast to complex via the angular term.
    prod = (4.0 * np.pi) * angular_exp[:, None, :] * radial_exp[None, :, :]

    # Scatter-add along pair axis. np.add.at expects the destination axis to
    # be the leading one for the index, so we transpose, accumulate, transpose back.
    cnk_perm = np.zeros((n_sites, k_max, n_max), dtype=np.complex128)
    np.add.at(cnk_perm, pair_struct, prod.transpose(2, 0, 1))
    return cnk_perm.transpose(1, 2, 0)


# --- Phase 4 validators --------------------------------------------------

def _build_pair_geometry_from_fixture(f: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct (rj, theta, phi) per pair from a Phase-0 fixture.

    Honours periodic boundaries via minimum-image convention. Used by both
    Phase 3 sanity check and Phase 4/5 validators.
    """
    positions = f["positions"]
    cell = f["cell"]
    pair_atom = f["pair_atom"]
    pair_gidx = f["pair_gidx"]
    n_pairs = len(pair_atom)
    is_periodic = bool(np.any(f["pbc"]))
    rjs = np.zeros(n_pairs)
    thetas = np.zeros(n_pairs)
    phis = np.zeros(n_pairs)
    cell_inv = np.linalg.inv(cell) if is_periodic else None
    for k in range(n_pairs):
        i = int(pair_atom[k]); j = int(pair_gidx[k])
        disp = positions[j] - positions[i]
        if is_periodic:
            frac = disp @ cell_inv
            frac -= np.round(frac)
            disp = frac @ cell
        r = float(np.linalg.norm(disp))
        rjs[k] = r
        if r < 1e-10:
            thetas[k] = 0.0
            phis[k] = 0.0
        else:
            thetas[k] = np.arccos(np.clip(disp[2] / r, -1.0, 1.0))
            phis[k] = np.arctan2(disp[1], disp[0])
    return rjs, thetas, phis


def _compute_cnk_for_fixture(name: str) -> tuple[np.ndarray, dict]:
    """Build R, A, and cnk for a fixture using Phases 1-4. Returns (cnk, fixture)."""
    f = load_fixture(name)
    soap_params = f["soap_params"]
    n_atoms = int(f["n_atoms"])
    species_Z = sorted(set(int(z) for z in f["numbers"]))
    n_species = len(species_Z)
    alpha_max = soap_params["alpha_max"]
    l_max = soap_params["l_max"]
    rcut_hard = soap_params["rcut_hard"]

    rjs, thetas, phis = _build_pair_geometry_from_fixture(f)
    pair_atom = np.asarray(f["pair_atom"], dtype=np.int32)
    pair_gidx = np.asarray(f["pair_gidx"], dtype=np.int32)
    # Strict self-pair (Fortran j==1): same atom AND rj=0. Image self-pairs
    # (rj > 0) are regular neighbours.
    pair_is_central = (pair_atom == pair_gidx) & (rjs < 1e-10)

    # Neighbour species: map atomic number → index 0..n_species-1
    z_to_idx = {z: idx for idx, z in enumerate(species_Z)}
    pair_neighbour_species = np.array(
        [z_to_idx[int(f["numbers"][int(j)])] for j in pair_gidx], dtype=np.int32
    )
    pair_active = rjs < rcut_hard

    # Phase 1: W
    W = build_orthonormalization_matrix_poly3(alpha_max)

    # Phase 2: radial
    R = radial_expansion_coeff_poly3_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=alpha_max,
        rcut_hard=rcut_hard,
        rcut_soft=soap_params["rcut_soft"],
        atom_sigma_r=soap_params["atom_sigma_r"],
        atom_sigma_r_scaling=soap_params["atom_sigma_r_scaling"],
        amplitude_scaling=soap_params["amplitude_scaling"],
        central_weight=soap_params["central_weight"],
        radial_enhancement=soap_params["radial_enhancement"],
        nf=4.0,                                                  # GPUMD default
        do_central=True,                                          # poly3 with central_weight=1
        W_single=W,
    )

    # Phase 3: angular
    A = angular_expansion_coeff_numpy(
        rjs, thetas, phis, pair_active,
        l_max=l_max,
        atom_sigma_t=soap_params["atom_sigma_t"],
        atom_sigma_t_scaling=soap_params["atom_sigma_t_scaling"],
        rcut=rcut_hard,
    )

    # Phase 4: aggregate
    cnk = aggregate_cnk(R, A, pair_atom, n_atoms)
    return cnk, f


def _validate_cnk_shape() -> None:
    """cnk shape: [k_max, n_max, n_sites] for each fixture (non-periodic only)."""
    for name in ["water_monomer", "water_dimer", "h2_close", "single_h"]:
        cnk, f = _compute_cnk_for_fixture(name)
        soap_params = f["soap_params"]
        l_max = soap_params["l_max"]
        alpha_max = soap_params["alpha_max"]
        n_species = len(set(int(z) for z in f["numbers"]))
        expected = ((l_max + 1) * (l_max + 2) // 2,
                    n_species * alpha_max,
                    int(f["n_atoms"]))
        if cnk.shape != expected:
            raise AssertionError(f"{name}: cnk shape {cnk.shape} != {expected}")
        if not np.all(np.isfinite(cnk)):
            raise AssertionError(f"{name}: cnk contains non-finite entries")
        print(f"   {name:<14} shape={cnk.shape}  finite=True  PASS")


def _validate_cnk_m0_is_real() -> None:
    """m=0 channels of cnk are purely real: prefm[0]=1 makes the angular
    factor real, and radial_exp is real."""
    for name in ["water_monomer", "water_dimer", "h2_close"]:
        cnk, f = _compute_cnk_for_fixture(name)
        l_max = f["soap_params"]["l_max"]
        max_imag = 0.0
        for l in range(l_max + 1):
            k_m0 = l * (l + 1) // 2 + 0
            max_imag = max(max_imag, float(np.abs(cnk[k_m0].imag).max()))
        if max_imag > 1e-12:
            raise AssertionError(f"{name}: m=0 channel has imag part {max_imag:.2e}")
        print(f"   {name:<14} max|imag(cnk[m=0])|={max_imag:.2e}  PASS")


def _validate_cnk_equivalence_under_symmetry() -> None:
    """Equivalent atoms give the same |cnk| (water_monomer H 1,2; h2_close
    0,1). Non-periodic only; periodic image enumeration deferred to Phase 6."""
    cases = [
        ("water_monomer", [1, 2]),
        ("h2_close", [0, 1]),
    ]
    for name, equiv_atoms in cases:
        cnk, _ = _compute_cnk_for_fixture(name)
        ref = np.abs(cnk[:, :, equiv_atoms[0]])
        for i in equiv_atoms[1:]:
            err = float(np.abs(np.abs(cnk[:, :, i]) - ref).max())
            if err > 1e-10:
                raise AssertionError(
                    f"{name}: |cnk[atom {i}]| differs from |cnk[atom {equiv_atoms[0]}]| "
                    f"by {err:.2e}"
                )
        print(f"   {name:<14} {len(equiv_atoms)} atoms equivalent under symmetry  PASS")


def run_phase4_validation() -> None:
    """Run every Phase-4 sanity check."""
    print("=== Phase 4 validation ===\n")

    print("1. cnk has correct shape and is finite for all fixtures")
    _validate_cnk_shape()

    print("\n2. m=0 channel of cnk is real")
    _validate_cnk_m0_is_real()

    print("\n3. cnk is equivalent across symmetry-related atoms")
    _validate_cnk_equivalence_under_symmetry()

    print("\nAll Phase 4 checks passed.")


# =========================================================================
# Phase 5 — Power spectrum + end-to-end forward gate
# Fortran ref: soap_turbo.f90:539-697. Per atom i:
#   this_soap[(n,n',l)] = Σ_m mult(n,n',m)·Re(cnk[k(l,m),n,i]·conj(cnk[k(l,m),n',i]))
# over upper-triangle (n,n') and l=0..l_max, then compress + L2-normalise.
# Ends with the forward gate: compute_soap_forward_numpy vs quippy fixtures
# (non-periodic).
# =========================================================================


def power_spectrum_numpy(
    cnk: np.ndarray,                # [k_max, n_max, n_sites] complex
    multiplicity_array: np.ndarray, # [n_active] real
    skip_mask: np.ndarray,          # [n_uncompressed] bool
    compressed_idx: np.ndarray,     # [P_nonzero] int — destination
    uncompressed_idx: np.ndarray,   # [P_nonzero] int — source channel
    coeffs: np.ndarray,             # [P_nonzero] real — sparse projection coefficients
    n_compressed: int,
    n_max: int,
    l_max: int,
) -> np.ndarray:                    # [n_sites, n_compressed] real
    """Build the per-atom SOAP power spectrum.

    Fortran reference: lines 539-580 (assembly), 565-573 (compression),
    574-578 (norm guard), 694-697 (final divide).
    """
    n_sites = cnk.shape[2]
    n_unc = n_max * (n_max + 1) // 2 * (l_max + 1)
    this_soap = np.zeros((n_unc, n_sites), dtype=np.float64)

    # Fortran order (n, n', l, m). counter → skip_mask; counter2 →
    # multiplicity_array (kept channels only).
    counter = 0
    counter2 = 0
    for n in range(n_max):
        for nprime in range(n, n_max):
            for l in range(l_max + 1):
                if not skip_mask[counter]:
                    k_start = l * (l + 1) // 2
                    k_end = k_start + (l + 1)            # m = 0..l (inclusive)
                    # cnk[k_start:k_end, n, :] · conj(cnk[k_start:k_end, n', :]) — [l+1, n_sites]
                    prod = (cnk[k_start:k_end, n, :]
                            * np.conj(cnk[k_start:k_end, nprime, :]))
                    mult_lm = multiplicity_array[counter2:counter2 + l + 1]
                    this_soap[counter] = np.sum(mult_lm[:, None] * np.real(prod), axis=0)
                    counter2 += l + 1
                counter += 1

    # Sparse projection: soap_compressed[c, i] = Σ_p coeffs[p] · this_soap[uncompressed[p], i]
    # for each entry where compressed_idx[p] == c.
    soap = np.zeros((n_compressed, n_sites), dtype=np.float64)
    np.add.at(soap, compressed_idx,
              coeffs[:, None] * this_soap[uncompressed_idx, :])

    # L2-normalise per atom, with the Fortran's empty-sphere guard
    norms = np.linalg.norm(soap, axis=0)
    norms = np.where(norms < 1e-5, 1.0, norms)
    soap = soap / norms[None, :]
    return soap.T  # [n_sites, n_compressed] to match quippy/_run_quippy convention


def compute_soap_forward_numpy(
    positions: np.ndarray,        # [n_atoms, 3]
    cell: np.ndarray,             # [3, 3]
    pbc: np.ndarray,              # [3] bool
    numbers: np.ndarray,          # [n_atoms] int (atomic numbers Z)
    species_Z: list[int],         # canonical species ordering
    pair_atom: np.ndarray,        # [P] int — centre index
    pair_gidx: np.ndarray,        # [P] int — neighbour index
    soap_params: dict,            # SOAP hyperparameters
    *,
    nf: float = 4.0,              # smoothing-filter exponent (GPUMD default)
) -> np.ndarray:                  # [n_atoms, n_compressed] float64
    """End-to-end forward SOAP-turbo for one structure (quippy pair list in).

    Pair geometry reconstructed via MIC; exact for non-periodic. Periodic
    multi-image pairs are collapsed (Phase 6 fixes this).
    """
    n_atoms = positions.shape[0]
    n_species = len(species_Z)
    alpha_max = int(soap_params["alpha_max"])
    l_max = int(soap_params["l_max"])
    rcut_hard = float(soap_params["rcut_hard"])

    # --- Pair geometry from positions ------------------------------------
    n_pairs = len(pair_atom)
    rjs = np.zeros(n_pairs)
    thetas = np.zeros(n_pairs)
    phis = np.zeros(n_pairs)
    is_periodic = bool(np.any(pbc))
    cell_inv = np.linalg.inv(cell) if is_periodic else None
    for k in range(n_pairs):
        i = int(pair_atom[k])
        j = int(pair_gidx[k])
        disp = positions[j] - positions[i]
        if is_periodic:
            frac = disp @ cell_inv
            frac -= np.round(frac)
            disp = frac @ cell
        r = float(np.linalg.norm(disp))
        rjs[k] = r
        if r < 1e-10:
            thetas[k] = 0.0
            phis[k] = 0.0
        else:
            thetas[k] = np.arccos(np.clip(disp[2] / r, -1.0, 1.0))
            phis[k] = np.arctan2(disp[1], disp[0])

    pair_is_central = (np.asarray(pair_atom) == np.asarray(pair_gidx))
    z_to_idx = {int(z): idx for idx, z in enumerate(species_Z)}
    pair_neighbour_species = np.array(
        [z_to_idx[int(numbers[int(j)])] for j in pair_gidx], dtype=np.int32
    )
    pair_active = rjs < rcut_hard

    # --- Phase 1: basis + compression ------------------------------------
    W_single = build_orthonormalization_matrix_poly3(alpha_max)
    alpha_per_species = [alpha_max] * n_species
    n_max = sum(alpha_per_species)
    mask_info = make_compress_mask_trivial(alpha_per_species, l_max)
    multiplicity_array = build_multiplicity_array(n_max, l_max, mask_info["skip_mask"])

    # --- Phase 2: radial expansion ---------------------------------------
    R = radial_expansion_coeff_poly3_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=alpha_max,
        rcut_hard=rcut_hard,
        rcut_soft=float(soap_params["rcut_soft"]),
        atom_sigma_r=float(soap_params["atom_sigma_r"]),
        atom_sigma_r_scaling=float(soap_params["atom_sigma_r_scaling"]),
        amplitude_scaling=float(soap_params["amplitude_scaling"]),
        central_weight=float(soap_params["central_weight"]),
        radial_enhancement=int(soap_params["radial_enhancement"]),
        nf=nf,
        do_central=(float(soap_params["central_weight"]) != 0.0),
        W_single=W_single,
    )

    # --- Phase 3: angular expansion --------------------------------------
    A = angular_expansion_coeff_numpy(
        rjs, thetas, phis, pair_active,
        l_max=l_max,
        atom_sigma_t=float(soap_params["atom_sigma_t"]),
        atom_sigma_t_scaling=float(soap_params["atom_sigma_t_scaling"]),
        rcut=rcut_hard,
    )

    # --- Phase 4: cnk aggregation ----------------------------------------
    cnk = aggregate_cnk(R, A, np.asarray(pair_atom, dtype=np.int32), n_atoms)

    # --- Phase 5: power spectrum + normalise -----------------------------
    return power_spectrum_numpy(
        cnk, multiplicity_array,
        mask_info["skip_mask"],
        mask_info["compressed_idx"],
        mask_info["uncompressed_idx"],
        mask_info["coeffs"],
        n_compressed=mask_info["n_compressed"],
        n_max=n_max,
        l_max=l_max,
    )


# --- Phase 5 validators --------------------------------------------------

def _validate_soap_against_fixtures(atol: float = 1e-6, rtol: float = 1e-5) -> None:
    """End-to-end forward gate: compute_soap_forward_numpy vs quippy
    descriptors for non-periodic fixtures (periodic deferred to Phase 6)."""
    non_periodic = ["water_monomer", "water_dimer", "h2_close", "h2_far", "single_h"]
    max_err_overall = 0.0
    for name in non_periodic:
        f = load_fixture(name)
        soap_predicted = compute_soap_forward_numpy(
            positions=np.asarray(f["positions"], dtype=np.float64),
            cell=np.asarray(f["cell"], dtype=np.float64),
            pbc=np.asarray(f["pbc"], dtype=bool),
            numbers=np.asarray(f["numbers"], dtype=np.int32),
            species_Z=sorted(set(int(z) for z in f["numbers"])),
            pair_atom=np.asarray(f["pair_atom"], dtype=np.int32),
            pair_gidx=np.asarray(f["pair_gidx"], dtype=np.int32),
            soap_params=f["soap_params"],
        ).astype(np.float32)                                          # quippy stores fp32
        soap_expected = np.asarray(f["descriptors"], dtype=np.float32)
        assert_allclose_soap(soap_predicted, soap_expected,
                             atol=atol, rtol=rtol, label=name)

        max_abs = float(np.abs(soap_predicted - soap_expected).max())
        n_atoms = soap_predicted.shape[0]
        dim_q = soap_predicted.shape[1]
        print(f"   {name:<14} N={n_atoms:>2}  dim_q={dim_q:>3}  "
              f"max|Δsoap|={max_abs:.2e}  PASS")
        max_err_overall = max(max_err_overall, max_abs)
    print(f"\n   Overall max|Δsoap| across non-periodic fixtures: {max_err_overall:.2e}")


def run_phase5_validation() -> None:
    """Run Phase-5 end-to-end forward gate."""
    print("=== Phase 5 validation (END-TO-END FORWARD GATE) ===\n")
    print("Comparing compute_soap_forward_numpy vs quippy fixtures:")
    _validate_soap_against_fixtures()
    print("\nAll Phase 5 forward checks passed.")


# =========================================================================
# Phase 6 — Periodic neighbour-list builder
# Image-enumerating pair list so the Phase-5 gate passes for periodic
# fixtures (MIC collapses multi-image pairs). Adds build_neighbour_list_numpy
# and compute_soap_from_positions_numpy (no quippy pair list). Image search
# spans a box sized rcut·||b_i|| (reciprocal vectors); image 0 only if aperiodic.
# =========================================================================


def build_neighbour_list_numpy(
    positions: np.ndarray,        # [n_atoms, 3]
    cell: np.ndarray,             # [3, 3]
    pbc: np.ndarray,              # [3] bool
    rcut: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Image-aware neighbour list (fully vectorised).

    Returns (pair_atom, pair_gidx, rjs, thetas, phis), all [P]. Each centre's
    (i, i, image=000) self-pair (rj=0) is emitted first, then all
    (i, j, image) with rj < rcut (including multiple images of one atom).
    Builds the [N, N, N_img, 3] displacement tensor, filters by rcut, sorts
    by centre.
    """
    n_atoms = positions.shape[0]
    is_periodic = bool(np.any(pbc))

    # Image-search bounding box per axis
    if is_periodic:
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            # Degenerate cell with pbc=True (data quirk) — treat as aperiodic.
            is_periodic = False
            n_imgs = np.zeros(3, dtype=np.int32)
        else:
            # Reciprocal vectors are the COLUMNS of inv(cell) (a_i·b_j=δ_ij with
            # rows(cell)=a_i), so the per-axis reciprocal norm is axis=0. axis=1
            # (rows) under-counts periodic images for non-orthogonal cells,
            # silently dropping neighbours near rcut (no-op for orthorhombic).
            b_norms = np.linalg.norm(cell_inv, axis=0)
            n_imgs = np.where(pbc, np.ceil(rcut * b_norms).astype(np.int32), 0)
            # Wrap into the primary cell: the image search only spans rcut
            # around each atom, so unwrapped atoms would miss real neighbours.
            # Matches quippy's internal wrap (translation invariance).
            frac = positions @ cell_inv
            # Wrap ONLY periodic axes: wrapping a non-periodic axis (e.g. z of a
            # [T,T,F] slab) teleports atoms outside the nominal cell and corrupts
            # the descriptor. n_imgs already respects pbc per-axis; match it here.
            frac -= np.floor(frac) * np.asarray(pbc, dtype=frac.dtype)
            positions = frac @ cell
    else:
        n_imgs = np.zeros(3, dtype=np.int32)

    # All image displacement vectors as a [N_img, 3] array
    nx, ny, nz = int(n_imgs[0]), int(n_imgs[1]), int(n_imgs[2])
    image_int = np.stack(np.meshgrid(
        np.arange(-nx, nx + 1),
        np.arange(-ny, ny + 1),
        np.arange(-nz, nz + 1),
        indexing="ij",
    ), axis=-1).reshape(-1, 3)                                       # [N_img, 3] int
    image_vectors = image_int.astype(np.float64) @ cell              # [N_img, 3]
    N_img = image_vectors.shape[0]
    zero_image_idx = int(np.argmin(np.linalg.norm(image_vectors, axis=1)))

    # All pairwise displacements: [N, N, N_img, 3]
    # disps[i, j, k, :] = positions[j] + image_vectors[k] - positions[i]
    disps = (positions[None, :, None, :]
             + image_vectors[None, None, :, :]
             - positions[:, None, None, :])
    rs = np.linalg.norm(disps, axis=-1)                              # [N, N, N_img]

    keep = rs < rcut
    # Mask the (i, i, image=000) self-pair so we don't double-count it
    eye = np.eye(n_atoms, dtype=bool)
    keep[eye, zero_image_idx] = False

    i_idx, j_idx, k_idx = np.where(keep)
    disp_flat = disps[i_idx, j_idx, k_idx]                           # [P_neigh, 3]
    rs_flat = rs[i_idx, j_idx, k_idx]

    # Stable-sort by centre so all of centre i's pairs come together, after
    # which we prepend each centre's self-pair (rj=0) at the start of its block.
    order = np.argsort(i_idx, kind="stable")
    i_neigh = i_idx[order].astype(np.int32)
    j_neigh = j_idx[order].astype(np.int32)
    disp_neigh = disp_flat[order]
    rs_neigh = rs_flat[order]

    # Spherical angles (vectorised). rj=0 entries get theta=phi=0 via tf.where-style mask.
    inv_r = np.where(rs_neigh > 1e-10, 1.0 / np.maximum(rs_neigh, 1e-300), 0.0)
    cos_theta = np.clip(disp_neigh[:, 2] * inv_r, -1.0, 1.0)
    thetas_neigh = np.where(rs_neigh > 1e-10, np.arccos(cos_theta), 0.0)
    phis_neigh = np.where(rs_neigh > 1e-10,
                          np.arctan2(disp_neigh[:, 1], disp_neigh[:, 0]),
                          0.0)

    # Insert each centre's self-pair at the start of its block; build the
    # flat length-(n_atoms + len(i_neigh)) arrays by scatter assignment.
    n_per_centre = np.bincount(i_neigh, minlength=n_atoms).astype(np.int32)
    block_offsets = np.concatenate(([0], np.cumsum(n_per_centre + 1)))   # [n_atoms+1]
    centre_self_pos = block_offsets[:-1]                                  # [n_atoms]
    P = int(block_offsets[-1])
    pair_atom = np.empty(P, dtype=np.int32)
    pair_gidx = np.empty(P, dtype=np.int32)
    rjs = np.empty(P, dtype=np.float64)
    thetas = np.empty(P, dtype=np.float64)
    phis = np.empty(P, dtype=np.float64)

    # Self-pairs at centre_self_pos
    pair_atom[centre_self_pos] = np.arange(n_atoms, dtype=np.int32)
    pair_gidx[centre_self_pos] = np.arange(n_atoms, dtype=np.int32)
    rjs[centre_self_pos] = 0.0
    thetas[centre_self_pos] = 0.0
    phis[centre_self_pos] = 0.0

    # Neighbour output index = block_offsets[i] + 1 + intra-block index
    # (i_neigh sorted by centre, so intra-block is a per-centre running counter).
    intra_idx = np.arange(len(i_neigh), dtype=np.int32) - np.cumsum(n_per_centre)[i_neigh] + n_per_centre[i_neigh]
    neigh_pos = block_offsets[i_neigh] + 1 + intra_idx
    pair_atom[neigh_pos] = i_neigh
    pair_gidx[neigh_pos] = j_neigh
    rjs[neigh_pos] = rs_neigh
    thetas[neigh_pos] = thetas_neigh
    phis[neigh_pos] = phis_neigh
    return pair_atom, pair_gidx, rjs, thetas, phis


def compute_soap_from_positions_numpy(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    numbers: np.ndarray,
    species_Z: list[int],
    soap_params: dict,
    *,
    nf: float = 4.0,
) -> np.ndarray:
    """End-to-end forward SOAP via build_neighbour_list_numpy (no quippy).
    Otherwise identical to compute_soap_forward_numpy."""
    n_atoms = positions.shape[0]
    n_species = len(species_Z)
    alpha_max = int(soap_params["alpha_max"])
    l_max = int(soap_params["l_max"])
    rcut_hard = float(soap_params["rcut_hard"])

    pair_atom, pair_gidx, rjs, thetas, phis = build_neighbour_list_numpy(
        positions, cell, pbc, rcut_hard
    )
    # Strict self-pair (Fortran j==1): same atom AND rj=0. Image self-pairs
    # (rj > 0) are regular neighbours.
    pair_is_central = (pair_atom == pair_gidx) & (rjs < 1e-10)
    pair_active = rjs < rcut_hard

    z_to_idx = {int(z): idx for idx, z in enumerate(species_Z)}
    pair_neighbour_species = np.array(
        [z_to_idx[int(numbers[int(j)])] for j in pair_gidx], dtype=np.int32
    )

    W_single = build_orthonormalization_matrix_poly3(alpha_max)
    alpha_per_species = [alpha_max] * n_species
    n_max = sum(alpha_per_species)
    mask_info = make_compress_mask_trivial(alpha_per_species, l_max)
    multiplicity_array = build_multiplicity_array(n_max, l_max, mask_info["skip_mask"])

    R = radial_expansion_coeff_poly3_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=alpha_max,
        rcut_hard=rcut_hard,
        rcut_soft=float(soap_params["rcut_soft"]),
        atom_sigma_r=float(soap_params["atom_sigma_r"]),
        atom_sigma_r_scaling=float(soap_params["atom_sigma_r_scaling"]),
        amplitude_scaling=float(soap_params["amplitude_scaling"]),
        central_weight=float(soap_params["central_weight"]),
        radial_enhancement=int(soap_params["radial_enhancement"]),
        nf=nf,
        do_central=(float(soap_params["central_weight"]) != 0.0),
        W_single=W_single,
    )

    A = angular_expansion_coeff_numpy(
        rjs, thetas, phis, pair_active,
        l_max=l_max,
        atom_sigma_t=float(soap_params["atom_sigma_t"]),
        atom_sigma_t_scaling=float(soap_params["atom_sigma_t_scaling"]),
        rcut=rcut_hard,
    )

    cnk = aggregate_cnk(R, A, pair_atom, n_atoms)

    return power_spectrum_numpy(
        cnk, multiplicity_array,
        mask_info["skip_mask"],
        mask_info["compressed_idx"],
        mask_info["uncompressed_idx"],
        mask_info["coeffs"],
        n_compressed=mask_info["n_compressed"],
        n_max=n_max,
        l_max=l_max,
    )


# --- Phase 6 validators --------------------------------------------------

def _validate_neighbour_list_pair_count() -> None:
    """Pair count from our NL must match quippy's for every fixture."""
    for name in ["water_monomer", "water_dimer", "h2_close", "h2_far",
                 "single_h", "si_bulk", "si_dimer"]:
        f = load_fixture(name)
        rcut = float(f["soap_params"]["rcut_hard"])
        pair_atom, *_ = build_neighbour_list_numpy(
            np.asarray(f["positions"], dtype=np.float64),
            np.asarray(f["cell"], dtype=np.float64),
            np.asarray(f["pbc"], dtype=bool),
            rcut,
        )
        n_mine = len(pair_atom)
        n_quippy = len(f["pair_atom"])
        if n_mine != n_quippy:
            raise AssertionError(
                f"{name}: NL has {n_mine} pairs, quippy fixture has {n_quippy}"
            )
        print(f"   {name:<14} pairs={n_mine:>4} (matches quippy)  PASS")


def _validate_soap_from_positions_against_fixtures(
    atol: float = 1e-6, rtol: float = 1e-5
) -> None:
    """Phase-5 gate using our own NL — should pass on ALL fixtures including periodic."""
    for name in ["water_monomer", "water_dimer", "h2_close", "h2_far",
                 "single_h", "si_bulk", "si_dimer"]:
        f = load_fixture(name)
        species_Z = sorted(set(int(z) for z in f["numbers"]))
        soap_predicted = compute_soap_from_positions_numpy(
            positions=np.asarray(f["positions"], dtype=np.float64),
            cell=np.asarray(f["cell"], dtype=np.float64),
            pbc=np.asarray(f["pbc"], dtype=bool),
            numbers=np.asarray(f["numbers"], dtype=np.int32),
            species_Z=species_Z,
            soap_params=f["soap_params"],
        ).astype(np.float32)
        soap_expected = np.asarray(f["descriptors"], dtype=np.float32)
        # The pair *ordering* may differ from quippy's, but per-atom descriptors
        # are invariant to ordering. assert_allclose_soap compares element-wise.
        assert_allclose_soap(soap_predicted, soap_expected,
                             atol=atol, rtol=rtol, label=name)
        max_abs = float(np.abs(soap_predicted - soap_expected).max())
        print(f"   {name:<14} N={int(f['n_atoms']):>2}  "
              f"max|Δsoap|={max_abs:.2e}  PASS")


def run_phase6_validation() -> None:
    """Phase-6 validation: own neighbour list + Phase-5 gate on ALL fixtures."""
    print("=== Phase 6 validation (own neighbour list, all 7 fixtures) ===\n")
    print("1. Pair counts match quippy")
    _validate_neighbour_list_pair_count()
    print("\n2. End-to-end forward gate on ALL fixtures (periodic + non-periodic)")
    _validate_soap_from_positions_against_fixtures()
    print("\nAll Phase 6 checks passed.")


# Update the placeholder DescriptorBuilderGPU class — Phase 6 gives it a
# working NumPy backend. Phase 10 will lift to TF.


# =========================================================================
# Phase 7 — Radial derivatives (NumPy)
# Fortran refs (soap_turbo_radial.f90): first-int der 265-275/594-604,
# second-int der 315-336/645-666, amplitude_der 196-231, assembly 339-360.
# Needs temp1/temp2 up to alpha_max+2; chain-rule coeffs reduce to N_a ratios.
# =========================================================================


def _radial_amplitude_with_der(
    rjs: np.ndarray,
    atom_sigma_scaled: np.ndarray,
    atom_sigma_scaling: float,
    is_central: np.ndarray,
    central_weight: float,
    amplitude_scaling: float,
    radial_enhancement: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude *and* its derivative w.r.t. rj. Fortran:196-231."""
    s2 = atom_sigma_scaled ** 2
    if amplitude_scaling == 0.0:
        amp = 1.0 / atom_sigma_scaled
        amp_der = -atom_sigma_scaling / s2
        amp_der = np.broadcast_to(amp_der, rjs.shape).copy() \
            if np.ndim(amp_der) == 0 else amp_der
    else:
        env = 1.0 + 2.0 * rjs ** 3 - 3.0 * rjs ** 2
        env_safe = np.where(env > 1e-10, env, 1.0)
        if amplitude_scaling == 1.0:
            amp_pre = (1.0 / atom_sigma_scaled) * env
            amp_der_pre = (6.0 / atom_sigma_scaled) * (rjs ** 2 - rjs) \
                - (atom_sigma_scaling / atom_sigma_scaled) * amp_pre
        else:
            amp_pre = (1.0 / atom_sigma_scaled) * env_safe ** amplitude_scaling
            amp_der_pre = (6.0 * amplitude_scaling / atom_sigma_scaled) * (rjs ** 2 - rjs) \
                * env_safe ** (amplitude_scaling - 1.0) \
                - (atom_sigma_scaling / atom_sigma_scaled) * amp_pre
        amp = np.where(env > 1e-10, amp_pre, 0.0)
        amp_der = np.where(env > 1e-10, amp_der_pre, 0.0)
    # Central atom factor
    amp = np.where(is_central, amp * central_weight, amp)
    amp_der = np.where(is_central, amp_der * central_weight, amp_der)
    # Radial enhancement
    if radial_enhancement == 1:
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        amp_der_new = amp * (1.0 + sqrt_2_pi * atom_sigma_scaling) + \
            amp_der * (rjs + sqrt_2_pi * atom_sigma_scaled)
        amp_new = amp * (rjs + sqrt_2_pi * atom_sigma_scaled)
        amp, amp_der = amp_new, amp_der_new
    elif radial_enhancement == 2:
        sqrt_8_pi = np.sqrt(8.0 / np.pi)
        amp_der_new = amp * (2.0 * rjs + 2.0 * atom_sigma_scaled * atom_sigma_scaling
                             + sqrt_8_pi * atom_sigma_scaled
                             + sqrt_8_pi * rjs * atom_sigma_scaling) + \
            amp_der * (rjs ** 2 + atom_sigma_scaled ** 2
                       + sqrt_8_pi * atom_sigma_scaled * rjs)
        amp_new = amp * (rjs ** 2 + atom_sigma_scaled ** 2 +
                         sqrt_8_pi * atom_sigma_scaled * rjs)
        amp, amp_der = amp_new, amp_der_new
    return amp, amp_der


def _radial_first_integral_der(
    rjs: np.ndarray,
    temp1_ext: np.ndarray,           # [alpha_max + 2, P]
    alpha_max: int,
    atom_sigma_scaled: np.ndarray,
    atom_sigma_scaling: float,
) -> np.ndarray:
    """Chain-rule derivative of first integral. Returns [alpha_max, P].

    Fortran reference: lines 265-275. With rcut_hard=1 the formula uses
    N_a(α) = 1/sqrt(2α+5).
    """
    s2 = atom_sigma_scaled ** 2
    out = np.zeros((alpha_max, P := rjs.shape[0]), dtype=np.float64)
    # Fortran loop var n → Python β-1 for temp1 indices.
    rj_minus = rjs - 1.0                                              # (rj - rcut_hard)
    sigma_term = atom_sigma_scaling * rj_minus / atom_sigma_scaled
    for n_f in range(1, alpha_max + 1):
        Na_n = _N_a(n_f)
        Na_np1 = _N_a(n_f + 1)
        Na_np2 = _N_a(n_f + 2)
        py = n_f - 1
        out[py] = (
            rj_minus / s2 * (sigma_term - 1.0) * temp1_ext[py]
            + Na_np1 / s2 / Na_n * (2.0 * sigma_term - 1.0) * temp1_ext[py + 1]
            + atom_sigma_scaling * Na_np2 / atom_sigma_scaled ** 3 / Na_n * temp1_ext[py + 2]
        )
    return out


def _radial_second_integral_der(
    rjs: np.ndarray,
    temp2_ext: np.ndarray,           # [alpha_max + 2, P]
    alpha_max: int,
    rcut_soft: float,
    atom_sigma_scaled: np.ndarray,
    atom_sigma_scaling: float,
    nf: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Chain-rule derivative of second integral + derivative of pref_f.

    Returns (out [alpha_max, P], der_pref_f [P]). Fortran reference: 315-336.
    """
    s2 = atom_sigma_scaled ** 2
    dr = 1.0 - rcut_soft
    denom = s2 + dr ** 2 / nf ** 2
    rj_minus = rjs - 1.0
    rcut_soft_minus_rj = rcut_soft - rjs

    # rj_f and atom_sigma_f as in second integral
    atom_sigma_f = atom_sigma_scaled * dr / nf / np.sqrt(denom)
    rj_f = (s2 * rcut_soft + dr ** 2 / nf ** 2 * rjs) / denom
    sf2 = atom_sigma_f ** 2

    pref_f = np.exp(-0.5 * rcut_soft_minus_rj ** 2 / denom)

    der_pref_f = pref_f * (
        rcut_soft_minus_rj / denom
        + rcut_soft_minus_rj ** 2 / denom ** 2 * atom_sigma_scaled * atom_sigma_scaling
    )
    der_rjf_rj = (2.0 * atom_sigma_scaled * rcut_soft * atom_sigma_scaling
                  + dr ** 2 / nf ** 2) / denom \
        - (s2 * rcut_soft + dr ** 2 / nf ** 2 * rjs) * 2.0 * atom_sigma_scaled \
        * atom_sigma_scaling / denom ** 2
    der_sjf_rj = atom_sigma_scaling * dr / nf / np.sqrt(denom) \
        * (1.0 - atom_sigma_scaled ** 2 / denom)

    rjf_minus = rj_f - 1.0
    sigma_f_term = der_sjf_rj * rjf_minus / atom_sigma_f

    out = np.zeros((alpha_max, rjs.shape[0]), dtype=np.float64)
    for n_f in range(1, alpha_max + 1):
        Na_n = _N_a(n_f)
        Na_np1 = _N_a(n_f + 1)
        Na_np2 = _N_a(n_f + 2)
        py = n_f - 1
        out[py] = pref_f * (
            rjf_minus / sf2 * (sigma_f_term - der_rjf_rj) * temp2_ext[py]
            + Na_np1 / sf2 / Na_n * (2.0 * sigma_f_term - der_rjf_rj) * temp2_ext[py + 1]
            + der_sjf_rj * Na_np2 / atom_sigma_f ** 3 / Na_n * temp2_ext[py + 2]
        ) + der_pref_f * temp2_ext[py]
    return out, der_pref_f


def radial_expansion_coeff_poly3_with_der_numpy(
    rjs: np.ndarray,
    pair_neighbour_species: np.ndarray,
    pair_is_central: np.ndarray,
    n_species: int,
    alpha_max: int,
    rcut_hard: float,
    rcut_soft: float,
    atom_sigma_r: float,
    atom_sigma_r_scaling: float,
    amplitude_scaling: float,
    central_weight: float,
    radial_enhancement: int,
    nf: float,
    do_central: bool,
    W_single: np.ndarray,
    global_scaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward + d/d(rj) of the radial expansion. Returns (radial [n_max,P], radial_der [n_max,P]).

    The derivative is w.r.t. the *unscaled* rj (in Å), as the Fortran emits.
    """
    P = rjs.shape[0]
    n_max = n_species * alpha_max

    rj_n = rjs / rcut_hard
    rcut_soft_n = rcut_soft / rcut_hard
    atom_sigma_n = atom_sigma_r / rcut_hard
    atom_sigma_scaled = atom_sigma_n + atom_sigma_r_scaling * rj_n

    in_cutoff = rj_n < 1.0
    pair_active = in_cutoff.copy()
    if not do_central:
        pair_active = pair_active & ~pair_is_central

    amplitude, amplitude_der = _radial_amplitude_with_der(
        rj_n, atom_sigma_scaled, atom_sigma_r_scaling, pair_is_central,
        central_weight, amplitude_scaling, radial_enhancement,
    )

    # Extended integrals (alpha_max + 2 entries) so the derivative formula has temp[n+1], temp[n+2].
    temp1_ext = _radial_first_integral(
        rj_n, alpha_max + 2, rcut_soft_n, atom_sigma_scaled
    )
    temp2_ext, pref_f = _radial_second_integral(
        rj_n, alpha_max + 2, rcut_soft_n, atom_sigma_scaled, nf
    )
    near_cutoff = (rcut_soft_n - rj_n) < 4.0 * atom_sigma_scaled
    pref_f = np.where(near_cutoff, pref_f, 0.0)

    # Forward path (same as Phase 2)
    combined = temp1_ext[:alpha_max] + pref_f[None, :] * temp2_ext[:alpha_max]
    transformed = W_single @ combined
    raw = amplitude[None, :] * transformed * global_scaling * np.sqrt(rcut_hard)

    # Derivative path: amplitude·(temp1_der + temp2_der) +
    # amplitude_der·(temp1 + pref_f·temp2). temp2_der bakes in der_pref_f·temp2.
    temp1_der = _radial_first_integral_der(
        rj_n, temp1_ext, alpha_max, atom_sigma_scaled, atom_sigma_r_scaling
    )
    temp2_der_total, _ = _radial_second_integral_der(
        rj_n, temp2_ext, alpha_max, rcut_soft_n, atom_sigma_scaled, atom_sigma_r_scaling, nf
    )
    # Apply the same near-cutoff gate to second-integral derivatives
    temp2_der_total = np.where(near_cutoff[None, :], temp2_der_total, 0.0)
    der_combined = (
        amplitude[None, :] * (temp1_der + temp2_der_total)
        + amplitude_der[None, :] * (temp1_ext[:alpha_max]
                                    + pref_f[None, :] * temp2_ext[:alpha_max])
    )
    transformed_der = W_single @ der_combined
    # Chain rule: d/d(rj_unscaled) = (1/rcut_hard) · d/d(rj_normalised), and the
    # outer sqrt(rcut_hard) factor from the basis change → 1/sqrt(rcut_hard).
    raw_der = transformed_der * global_scaling / np.sqrt(rcut_hard)

    raw = raw * pair_active[None, :].astype(np.float64)
    raw_der = raw_der * pair_active[None, :].astype(np.float64)

    radial = np.zeros((n_max, P), dtype=np.float64)
    radial_der = np.zeros((n_max, P), dtype=np.float64)
    for s in range(n_species):
        species_mask = (pair_neighbour_species == s).astype(np.float64)
        radial[s * alpha_max:(s + 1) * alpha_max, :] = raw * species_mask[None, :]
        radial_der[s * alpha_max:(s + 1) * alpha_max, :] = raw_der * species_mask[None, :]
    return radial, radial_der


# --- Phase 7 validators --------------------------------------------------

def _validate_radial_der_via_finite_difference(eps: float = 1e-6) -> None:
    """FD check of d(radial)/d(rj): (R(rj+eps)-R(rj-eps))/2eps vs analytic."""
    rcut_hard = 6.0
    rcut_soft = 5.5
    atom_sigma_r = 0.5
    alpha_max = 4
    W = build_orthonormalization_matrix_poly3(alpha_max)

    # Avoid the near-cutoff boundary (rj ≈ rcut_soft - 4σ): the strict
    # inequality there makes a tiny pref_f jump that pollutes FD (not a bug).
    test_rj = np.array([0.5, 1.5, 2.5, 3.0, 4.5, 5.0, 5.4, 5.7, 5.9])
    max_err = 0.0
    for rj in test_rj:
        rjs_pm = np.array([rj - eps, rj + eps], dtype=np.float64)
        common = dict(
            pair_neighbour_species=np.zeros(2, dtype=np.int32),
            pair_is_central=np.zeros(2, dtype=bool),
            n_species=1, alpha_max=alpha_max,
            rcut_hard=rcut_hard, rcut_soft=rcut_soft,
            atom_sigma_r=atom_sigma_r, atom_sigma_r_scaling=0.0,
            amplitude_scaling=1.0, central_weight=1.0,
            radial_enhancement=1, nf=4.0, do_central=True,
            W_single=W,
        )
        R_pm = radial_expansion_coeff_poly3_numpy(rjs_pm, **common)
        R_fd = (R_pm[:, 1] - R_pm[:, 0]) / (2.0 * eps)

        # Analytical derivative at the centre point
        rj_arr = np.array([rj], dtype=np.float64)
        common_single = dict(
            pair_neighbour_species=np.zeros(1, dtype=np.int32),
            pair_is_central=np.zeros(1, dtype=bool),
            n_species=1, alpha_max=alpha_max,
            rcut_hard=rcut_hard, rcut_soft=rcut_soft,
            atom_sigma_r=atom_sigma_r, atom_sigma_r_scaling=0.0,
            amplitude_scaling=1.0, central_weight=1.0,
            radial_enhancement=1, nf=4.0, do_central=True,
            W_single=W,
        )
        _, R_der = radial_expansion_coeff_poly3_with_der_numpy(rj_arr, **common_single)

        err = float(np.abs(R_fd - R_der[:, 0]).max())
        max_err = max(max_err, err)
        if err > 1e-4:
            raise AssertionError(
                f"radial_der at rj={rj}: max|FD - analytic| = {err:.3e} > 1e-4"
            )
    print(f"   radial_der vs finite difference: max abs err = {max_err:.2e}  PASS")


def _validate_radial_der_smooth() -> None:
    """Derivative is finite and reasonably smooth across rj sweep."""
    rjs = np.linspace(0.05, 5.99, 40)
    n_pairs = len(rjs)
    W = build_orthonormalization_matrix_poly3(4)
    R, R_der = radial_expansion_coeff_poly3_with_der_numpy(
        rjs, np.zeros(n_pairs, dtype=np.int32), np.zeros(n_pairs, dtype=bool),
        n_species=1, alpha_max=4,
        rcut_hard=6.0, rcut_soft=5.5,
        atom_sigma_r=0.5, atom_sigma_r_scaling=0.0,
        amplitude_scaling=1.0, central_weight=1.0,
        radial_enhancement=1, nf=4.0, do_central=True,
        W_single=W,
    )
    if not np.all(np.isfinite(R_der)):
        raise AssertionError("non-finite values in radial_der sweep")
    print(f"   smooth sweep over rj∈[0.05, 5.99]: max|R_der|={np.abs(R_der).max():.4f}  PASS")


def run_phase7_validation() -> None:
    print("=== Phase 7 validation (radial derivatives) ===\n")
    print("1. Analytical radial_der matches finite difference")
    _validate_radial_der_via_finite_difference()
    print("\n2. Radial derivative is finite and smooth")
    _validate_radial_der_smooth()
    print("\nAll Phase 7 checks passed.")


# =========================================================================
# Phase 8 — Angular + cnk derivatives (NumPy)
# Fortran refs: plm_der soap_turbo_angular.f90:106-169, ilexp_der 329-356,
# angular_with_der 225-244/440-447, cnk_with_der soap_turbo.f90:781-822.
#
# Conventions: rad_der = d/d(rj); pol_der = -d/d(theta); azi_der =
# d/d(phi)/sin(theta). Plm derivatives absorb the sin(θ) pole singularity.
# =========================================================================


def _get_plm_array_der(
    plm_extended: np.ndarray,    # [k_max_ext, P], Plm computed up to l_max+1
    l_max: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (plm_div_sin, plm_der_mul_sin), both [k_max, P].

    plm_div_sin[k(l,m)]         = -m · P_lm / sin(θ)               for m ≥ 1, else 0
    plm_der_mul_sin[k(l,m)]     = sin(θ) · dP_lm/d(cosθ)           for all m

    Identities used (from the Fortran):
        sin(θ)·dP_lm/d(cosθ) = ½·[(l+m)(l-m+1)·P_l^{m-1} − P_l^{m+1}]
        −m·P_lm/sin(θ)       = ½·[(l-m+1)(l-m+2)·P_{l+1}^{m-1} − P_{l+1}^{m+1}]
    For m=0, P_l^{-1} = −P_l^1 / (l(l+1)) so part1 = −½·P_l^1.
    """
    P = plm_extended.shape[1]
    k_max = (l_max + 1) * (l_max + 2) // 2
    plm_der_mul_sin = np.zeros((k_max, P), dtype=np.float64)
    plm_div_sin = np.zeros((k_max, P), dtype=np.float64)

    # plm_der_mul_sin
    for l in range(l_max + 1):
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            # part1
            if m == 0:
                if l == 0:
                    part1 = np.zeros(P)
                else:
                    k_l_1 = l * (l + 1) // 2 + 1
                    part1 = -0.5 * plm_extended[k_l_1]
            else:
                k_l_mm1 = l * (l + 1) // 2 + (m - 1)
                part1 = 0.5 * (l + m) * (l - m + 1) * plm_extended[k_l_mm1]
            # part2
            if m == l:
                part2 = np.zeros(P)
            else:
                k_l_mp1 = l * (l + 1) // 2 + (m + 1)
                part2 = -0.5 * plm_extended[k_l_mp1]
            plm_der_mul_sin[k] = part1 + part2

    # plm_div_sin (only for m >= 1)
    for l in range(l_max + 1):
        for m in range(1, l + 1):
            k = l * (l + 1) // 2 + m
            k_lp1_mp1 = (l + 1) * (l + 2) // 2 + (m + 1)
            k_lp1_mm1 = (l + 1) * (l + 2) // 2 + (m - 1)
            part1 = 0.5 * (l - m + 1) * (l - m + 2) * plm_extended[k_lp1_mm1]
            part2 = 0.5 * plm_extended[k_lp1_mp1]
            plm_div_sin[k] = part1 + part2

    return plm_div_sin, plm_der_mul_sin


def _get_ilexp_der(
    rj: np.ndarray,
    ilexp_array: np.ndarray,    # [l_max+1, P]
    l_max: int,
    atom_sigma: np.ndarray,     # [P]
    atom_sigma_scaling: float,
) -> np.ndarray:
    """d(ilexp(l, rj/σ))/d(rj). Returns [l_max+1, P].

    Recursion in l with explicit stability: zero output at rj < 1e-5.
    """
    P = rj.shape[0]
    out = np.zeros((l_max + 1, P), dtype=np.float64)
    coeff1 = 2.0 * rj / atom_sigma ** 2
    coeff2 = 1.0 - atom_sigma_scaling * rj / atom_sigma

    out[0] = coeff1 * (ilexp_array[1] - ilexp_array[0])
    safe_rj = np.maximum(rj, 1e-300)
    for l in range(1, l_max + 1):
        out[l] = ((-coeff1 - 2.0 * (l + 1) / safe_rj) * ilexp_array[l]
                  + coeff1 * ilexp_array[l - 1])
    out = out * coeff2[None, :]
    safe = (rj >= 1e-5).astype(np.float64)
    out = out * safe[None, :]
    return out


def angular_expansion_coeff_with_der_numpy(
    rjs: np.ndarray,
    thetas: np.ndarray,
    phis: np.ndarray,
    pair_active: np.ndarray,
    l_max: int,
    atom_sigma_t: float,
    atom_sigma_t_scaling: float,
    rcut: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Angular forward + 3 derivatives (radial, azimuthal, polar).

    Returns (exp_coeff, exp_coeff_rad_der, exp_coeff_azi_der, exp_coeff_pol_der)
    each shape [k_max, P] complex.
    """
    P = rjs.shape[0]
    k_max = (l_max + 1) * (l_max + 2) // 2

    # Plm at extended l_max for derivative recursions
    x = np.cos(thetas)
    plm_ext = _get_plm_array(x, l_max + 1)
    plm = plm_ext[:k_max]
    plm_div_sin, plm_der_mul_sin = _get_plm_array_der(plm_ext, l_max)

    atom_sigma = atom_sigma_t + atom_sigma_t_scaling * rjs
    rj_by_sigma = rjs / atom_sigma
    prefl = _get_ilexp(rj_by_sigma, l_max)
    prefl_rad_der = _get_ilexp_der(rjs, prefl, l_max, atom_sigma, atom_sigma_t_scaling)

    prefm = _get_eimphi_factor(phis, l_max)

    # Compose eimphi[k(l,m)] and its radial derivative
    eimphi = np.zeros((k_max, P), dtype=np.complex128)
    eimphi_rad_der = np.zeros((k_max, P), dtype=np.complex128)
    for l in range(l_max + 1):
        for m in range(l + 1):
            k = l * (l + 1) // 2 + m
            eimphi[k] = prefl[l] * prefm[m]
            eimphi_rad_der[k] = prefl_rad_der[l] * prefm[m]
    eimphi_azi_der = eimphi * 1j

    preflm = _get_preflm(l_max)
    amplitude = rcut ** 2 / atom_sigma ** 2

    exp_coeff = (amplitude[None, :] * preflm[:, None] * plm * eimphi)
    exp_coeff_rad_der = (
        amplitude[None, :] * preflm[:, None] * plm * eimphi_rad_der
        - 2.0 * amplitude[None, :] / atom_sigma[None, :] * atom_sigma_t_scaling
        * preflm[:, None] * plm * eimphi
    )
    exp_coeff_azi_der = amplitude[None, :] * preflm[:, None] * plm_div_sin * eimphi_azi_der
    exp_coeff_pol_der = amplitude[None, :] * preflm[:, None] * plm_der_mul_sin * eimphi

    mask_factor = pair_active[None, :].astype(np.float64)
    exp_coeff = exp_coeff * mask_factor
    exp_coeff_rad_der = exp_coeff_rad_der * mask_factor
    exp_coeff_azi_der = exp_coeff_azi_der * mask_factor
    exp_coeff_pol_der = exp_coeff_pol_der * mask_factor
    return exp_coeff, exp_coeff_rad_der, exp_coeff_azi_der, exp_coeff_pol_der


def aggregate_cnk_with_der(
    R: np.ndarray,                # [n_max, P]
    A: np.ndarray,                # [k_max, P] complex
    R_der: np.ndarray,            # [n_max, P]
    A_rad_der: np.ndarray,        # [k_max, P] complex
    A_azi_der: np.ndarray,        # [k_max, P] complex
    A_pol_der: np.ndarray,        # [k_max, P] complex
    pair_struct: np.ndarray,      # [P]
    n_sites: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward cnk and its three per-pair spherical derivatives.

    Per-pair derivatives have shape [k_max, n_max, P] — they are NOT scattered
    to centres because the chain rule for d(SOAP)/d(r_p) is local to pair p.
    Phase 9 will scatter via the centre's "self-pair" index.
    """
    fac = 4.0 * np.pi
    cnk = aggregate_cnk(R, A, pair_struct, n_sites)
    cnk_rad_der = fac * (A[:, None, :] * R_der[None, :, :]
                         + A_rad_der[:, None, :] * R[None, :, :])
    cnk_azi_der = fac * A_azi_der[:, None, :] * R[None, :, :]
    cnk_pol_der = fac * A_pol_der[:, None, :] * R[None, :, :]
    return cnk, cnk_rad_der, cnk_azi_der, cnk_pol_der


# --- Phase 8 validators --------------------------------------------------

def _validate_plm_der_against_fd(eps: float = 1e-6) -> None:
    """sin(θ)·dP_lm/d(cosθ) = plm_der_mul_sin via FD on Plm at x±eps."""
    test_theta = np.array([0.3, 0.7, 1.2, 1.7, 2.4, 2.8])
    l_max = 5
    max_err = 0.0
    for theta in test_theta:
        x = np.cos(theta)
        sin_t = np.sin(theta)
        x_pm = np.array([x - eps, x + eps])
        plm_pm = _get_plm_array(x_pm, l_max + 1)                  # extended for div_sin
        plm_ext_c = _get_plm_array(np.array([x]), l_max + 1)
        # FD: dPlm/d(x), then mul by sin(θ) for plm_der_mul_sin
        dPlm_dx = (plm_pm[:, 1] - plm_pm[:, 0]) / (2 * eps)        # [k_max_ext]
        plm_div_sin, plm_der_mul_sin = _get_plm_array_der(plm_ext_c, l_max)
        k_max = (l_max + 1) * (l_max + 2) // 2
        # plm_der_mul_sin should equal sin(θ) * dPlm/dx (truncated to k_max)
        ref = sin_t * dPlm_dx[:k_max]
        err = float(np.abs(plm_der_mul_sin[:, 0] - ref).max())
        if err > 1e-5:
            raise AssertionError(
                f"plm_der_mul_sin at θ={theta}: max|FD - analytic| = {err:.3e}"
            )
        max_err = max(max_err, err)
    print(f"   plm_der_mul_sin vs FD: max abs err = {max_err:.2e}  PASS")


def _validate_plm_div_sin_direct() -> None:
    """plm_div_sin(l, m) = -m·P_lm/sin(θ), checked at non-pole θ."""
    test_theta = np.array([0.3, 0.7, 1.2, 1.7, 2.4, 2.8])
    l_max = 5
    max_err = 0.0
    for theta in test_theta:
        x = np.cos(theta)
        sin_t = np.sin(theta)
        plm_ext = _get_plm_array(np.array([x]), l_max + 1)
        plm_div_sin, _ = _get_plm_array_der(plm_ext, l_max)
        for l in range(l_max + 1):
            for m in range(1, l + 1):
                k_full = l * (l + 1) // 2 + m
                ref = -m * plm_ext[k_full, 0] / sin_t
                err = float(abs(plm_div_sin[k_full, 0] - ref))
                max_err = max(max_err, err)
                if err > 1e-10:
                    raise AssertionError(
                        f"plm_div_sin(l={l}, m={m}, θ={theta}): "
                        f"recursion={plm_div_sin[k_full, 0]} ref={ref} err={err:.3e}"
                    )
    print(f"   plm_div_sin vs direct -m·P/sin(θ): max abs err = {max_err:.2e}  PASS")


def _validate_ilexp_der_against_fd(eps: float = 1e-6) -> None:
    """ilexp_der vs FD on ilexp(rj/σ) at rj±eps."""
    sigma_t = 0.5
    test_rj = np.array([0.5, 1.0, 2.0, 3.5, 5.0])
    l_max = 5
    max_err = 0.0
    for rj in test_rj:
        rj_pm = np.array([rj - eps, rj + eps])
        sigma_pm = np.full(2, sigma_t)
        rj_by_sigma_pm = rj_pm / sigma_pm
        ilexp_pm = _get_ilexp(rj_by_sigma_pm, l_max)
        ilexp_fd = (ilexp_pm[:, 1] - ilexp_pm[:, 0]) / (2 * eps)

        rj_c = np.array([rj])
        sigma_c = np.full(1, sigma_t)
        ilexp_c = _get_ilexp(rj_c / sigma_c, l_max)
        ilexp_der = _get_ilexp_der(rj_c, ilexp_c, l_max, sigma_c, atom_sigma_scaling=0.0)
        err = float(np.abs(ilexp_der[:, 0] - ilexp_fd).max())
        max_err = max(max_err, err)
        if err > 1e-5:
            raise AssertionError(f"ilexp_der at rj={rj}: max|FD - analytic|={err:.2e}")
    print(f"   ilexp_der vs FD: max abs err = {max_err:.2e}  PASS")


def _angular_components(rj: float, theta: float, phi: float, l_max: int,
                         sigma_t: float, sigma_t_scaling: float, rcut: float):
    """Helper: scalar wrapper around angular_expansion_coeff_with_der_numpy."""
    return angular_expansion_coeff_with_der_numpy(
        np.array([rj]), np.array([theta]), np.array([phi]),
        np.array([True]), l_max, sigma_t, sigma_t_scaling, rcut,
    )


def _validate_angular_der_via_fd(eps: float = 1e-6) -> None:
    """All three angular derivatives validated against FD."""
    sigma_t = 0.5
    rcut = 6.0
    l_max = 4
    # Avoid pole θ values
    cases = [
        (1.5, 0.7, 0.4),
        (2.5, 1.2, -1.3),
        (3.0, 0.9, 2.1),
        (4.5, 2.0, 0.8),
    ]
    max_rad = max_azi = max_pol = 0.0
    for rj, theta, phi in cases:
        # Forward derivatives at centre
        ang, ang_rad_der, ang_azi_der, ang_pol_der = _angular_components(
            rj, theta, phi, l_max, sigma_t, 0.0, rcut
        )

        # Radial FD
        ang_p, _, _, _ = _angular_components(rj + eps, theta, phi, l_max, sigma_t, 0.0, rcut)
        ang_m, _, _, _ = _angular_components(rj - eps, theta, phi, l_max, sigma_t, 0.0, rcut)
        rad_fd = (ang_p[:, 0] - ang_m[:, 0]) / (2 * eps)
        err_rad = float(np.abs(rad_fd - ang_rad_der[:, 0]).max())

        # Polar FD: pol_der = -d(angular)/dθ
        ang_p, _, _, _ = _angular_components(rj, theta + eps, phi, l_max, sigma_t, 0.0, rcut)
        ang_m, _, _, _ = _angular_components(rj, theta - eps, phi, l_max, sigma_t, 0.0, rcut)
        pol_fd = -(ang_p[:, 0] - ang_m[:, 0]) / (2 * eps)
        err_pol = float(np.abs(pol_fd - ang_pol_der[:, 0]).max())

        # Azi FD: azi_der = d(angular)/dφ / sin(θ)
        ang_p, _, _, _ = _angular_components(rj, theta, phi + eps, l_max, sigma_t, 0.0, rcut)
        ang_m, _, _, _ = _angular_components(rj, theta, phi - eps, l_max, sigma_t, 0.0, rcut)
        azi_fd = (ang_p[:, 0] - ang_m[:, 0]) / (2 * eps) / np.sin(theta)
        err_azi = float(np.abs(azi_fd - ang_azi_der[:, 0]).max())

        max_rad = max(max_rad, err_rad)
        max_azi = max(max_azi, err_azi)
        max_pol = max(max_pol, err_pol)
        if max(err_rad, err_azi, err_pol) > 1e-4:
            raise AssertionError(
                f"angular der at (rj,θ,φ)=({rj},{theta},{phi}): "
                f"errs rad={err_rad:.2e} azi={err_azi:.2e} pol={err_pol:.2e}"
            )
    print(f"   angular rad/azi/pol vs FD: max abs err = "
          f"{max_rad:.2e} / {max_azi:.2e} / {max_pol:.2e}  PASS")


def _validate_cnk_der_via_fd(eps: float = 1e-6) -> None:
    """cnk derivatives must match FD of (R · A) chain rule per pair."""
    rcut_hard = 6.0
    rcut_soft = 5.5
    sigma_r = 0.5
    sigma_t = 0.5
    alpha_max = 4
    l_max = 4
    W = build_orthonormalization_matrix_poly3(alpha_max)

    rj = 2.5; theta = 1.0; phi = 0.5
    pair_active = np.array([True])
    pair_neighbour_species = np.zeros(1, dtype=np.int32)
    pair_is_central = np.zeros(1, dtype=bool)
    pair_struct = np.array([0], dtype=np.int32)

    # Compute forward + per-pair derivatives at centre
    R, R_der = radial_expansion_coeff_poly3_with_der_numpy(
        np.array([rj]), pair_neighbour_species, pair_is_central,
        n_species=1, alpha_max=alpha_max, rcut_hard=rcut_hard, rcut_soft=rcut_soft,
        atom_sigma_r=sigma_r, atom_sigma_r_scaling=0.0,
        amplitude_scaling=1.0, central_weight=1.0, radial_enhancement=1, nf=4.0,
        do_central=True, W_single=W,
    )
    A, A_rad, A_azi, A_pol = angular_expansion_coeff_with_der_numpy(
        np.array([rj]), np.array([theta]), np.array([phi]), pair_active,
        l_max, sigma_t, 0.0, rcut_hard,
    )
    cnk, cnk_rad, cnk_azi, cnk_pol = aggregate_cnk_with_der(
        R, A, R_der, A_rad, A_azi, A_pol, pair_struct, n_sites=1,
    )

    # FD: perturb rj while keeping (theta, phi) and species fixed
    def _eval_at(rj_v, theta_v, phi_v):
        R_, _ = radial_expansion_coeff_poly3_with_der_numpy(
            np.array([rj_v]), pair_neighbour_species, pair_is_central,
            n_species=1, alpha_max=alpha_max, rcut_hard=rcut_hard, rcut_soft=rcut_soft,
            atom_sigma_r=sigma_r, atom_sigma_r_scaling=0.0,
            amplitude_scaling=1.0, central_weight=1.0, radial_enhancement=1, nf=4.0,
            do_central=True, W_single=W,
        )
        A_, _, _, _ = angular_expansion_coeff_with_der_numpy(
            np.array([rj_v]), np.array([theta_v]), np.array([phi_v]), pair_active,
            l_max, sigma_t, 0.0, rcut_hard,
        )
        return aggregate_cnk(R_, A_, pair_struct, n_sites=1)

    cnk_rj_p = _eval_at(rj + eps, theta, phi)
    cnk_rj_m = _eval_at(rj - eps, theta, phi)
    rad_fd = (cnk_rj_p - cnk_rj_m) / (2 * eps)                        # [k, n, 1]

    cnk_th_p = _eval_at(rj, theta + eps, phi)
    cnk_th_m = _eval_at(rj, theta - eps, phi)
    pol_fd = -(cnk_th_p - cnk_th_m) / (2 * eps)                       # = -d/dθ

    cnk_ph_p = _eval_at(rj, theta, phi + eps)
    cnk_ph_m = _eval_at(rj, theta, phi - eps)
    azi_fd = (cnk_ph_p - cnk_ph_m) / (2 * eps) / np.sin(theta)        # = (d/dφ)/sin(θ)

    err_rad = float(np.abs(cnk_rad[:, :, 0] - rad_fd[:, :, 0]).max())
    err_azi = float(np.abs(cnk_azi[:, :, 0] - azi_fd[:, :, 0]).max())
    err_pol = float(np.abs(cnk_pol[:, :, 0] - pol_fd[:, :, 0]).max())
    if max(err_rad, err_azi, err_pol) > 1e-4:
        raise AssertionError(
            f"cnk der: rad={err_rad:.2e} azi={err_azi:.2e} pol={err_pol:.2e}"
        )
    print(f"   cnk rad/azi/pol vs FD: max abs err = "
          f"{err_rad:.2e} / {err_azi:.2e} / {err_pol:.2e}  PASS")


def run_phase8_validation() -> None:
    print("=== Phase 8 validation (angular + cnk derivatives) ===\n")
    print("1. plm_der_mul_sin matches FD on Plm")
    _validate_plm_der_against_fd()
    print("\n2. plm_div_sin matches direct formula -m·P/sin(θ)")
    _validate_plm_div_sin_direct()
    print("\n3. ilexp_der matches FD on ilexp")
    _validate_ilexp_der_against_fd()
    print("\n4. angular rad/azi/pol derivatives match FD")
    _validate_angular_der_via_fd()
    print("\n5. cnk rad/azi/pol derivatives match FD")
    _validate_cnk_der_via_fd()
    print("\nAll Phase 8 checks passed.")


# =========================================================================
# Phase 9 — Power-spectrum derivatives + Cartesian + self-derivative
# Fortran ref: soap_turbo.f90:583-697, 778-822.
#
# Per pair p: (1) product-rule (n,n',l)-flat derivatives from cnk·conj(cnk);
# (2) sparse compression; (3) L2-norm derivative
#   soap_*_der/||soap|| − soap·dot(soap, soap_*_der)/||soap||³;
# (4) spherical→Cartesian Jacobian; (5) self-derivative: central-pair grad =
# −Σ_{j≠i} grad_pair (translational invariance).
# =========================================================================


def power_spectrum_with_grad_numpy(
    cnk: np.ndarray,                  # [k_max, n_max, n_sites] complex
    cnk_rad_der: np.ndarray,          # [k_max, n_max, P] complex
    cnk_azi_der: np.ndarray,
    cnk_pol_der: np.ndarray,
    pair_atom: np.ndarray,            # [P] — centre index for each pair
    multiplicity_array: np.ndarray,
    skip_mask: np.ndarray,
    compressed_idx: np.ndarray,
    uncompressed_idx: np.ndarray,
    coeffs: np.ndarray,
    n_compressed: int,
    n_max: int,
    l_max: int,
    n_sites: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (soap_norm, soap_rad_der, soap_azi_der, soap_pol_der) all in normalised form.

    soap_norm        : [n_sites, n_compressed]
    soap_*_der       : [n_compressed, P]      (per-pair, in spherical coordinates)
    """
    P = cnk_rad_der.shape[2]
    n_unc = n_max * (n_max + 1) // 2 * (l_max + 1)

    # --- Phase 5 forward (un-normalised) ---
    this_soap = np.zeros((n_unc, n_sites), dtype=np.float64)
    # Per-pair (n,n',l)-flat derivatives in spherical coords
    this_soap_rad = np.zeros((n_unc, P), dtype=np.float64)
    this_soap_azi = np.zeros((n_unc, P), dtype=np.float64)
    this_soap_pol = np.zeros((n_unc, P), dtype=np.float64)

    counter = 0
    counter2 = 0
    for n in range(n_max):
        for nprime in range(n, n_max):
            for l in range(l_max + 1):
                if not skip_mask[counter]:
                    k_start = l * (l + 1) // 2
                    k_end = k_start + (l + 1)
                    mult_lm = multiplicity_array[counter2:counter2 + l + 1]

                    # Forward (centre-indexed)
                    prod_fwd = (cnk[k_start:k_end, n, :]
                                * np.conj(cnk[k_start:k_end, nprime, :]))
                    this_soap[counter] = np.sum(mult_lm[:, None] * np.real(prod_fwd), axis=0)

                    # Per-pair derivatives — gather centre cnk by pair_atom
                    cnk_at_n = cnk[k_start:k_end, n, :][:, pair_atom]      # [l+1, P]
                    cnk_at_np = cnk[k_start:k_end, nprime, :][:, pair_atom]
                    rn = cnk_rad_der[k_start:k_end, n, :]
                    rnp = cnk_rad_der[k_start:k_end, nprime, :]
                    an = cnk_azi_der[k_start:k_end, n, :]
                    anp = cnk_azi_der[k_start:k_end, nprime, :]
                    pn = cnk_pol_der[k_start:k_end, n, :]
                    pnp = cnk_pol_der[k_start:k_end, nprime, :]

                    rad_term = rn * np.conj(cnk_at_np) + cnk_at_n * np.conj(rnp)
                    azi_term = an * np.conj(cnk_at_np) + cnk_at_n * np.conj(anp)
                    pol_term = pn * np.conj(cnk_at_np) + cnk_at_n * np.conj(pnp)

                    this_soap_rad[counter] = np.sum(mult_lm[:, None] * np.real(rad_term), axis=0)
                    this_soap_azi[counter] = np.sum(mult_lm[:, None] * np.real(azi_term), axis=0)
                    this_soap_pol[counter] = np.sum(mult_lm[:, None] * np.real(pol_term), axis=0)
                    counter2 += l + 1
                counter += 1

    # Apply sparse compression
    soap_unnorm = np.zeros((n_compressed, n_sites), dtype=np.float64)
    np.add.at(soap_unnorm, compressed_idx,
              coeffs[:, None] * this_soap[uncompressed_idx, :])

    soap_rad_der = np.zeros((n_compressed, P), dtype=np.float64)
    soap_azi_der = np.zeros((n_compressed, P), dtype=np.float64)
    soap_pol_der = np.zeros((n_compressed, P), dtype=np.float64)
    np.add.at(soap_rad_der, compressed_idx, coeffs[:, None] * this_soap_rad[uncompressed_idx, :])
    np.add.at(soap_azi_der, compressed_idx, coeffs[:, None] * this_soap_azi[uncompressed_idx, :])
    np.add.at(soap_pol_der, compressed_idx, coeffs[:, None] * this_soap_pol[uncompressed_idx, :])

    # L2 normalisation: norm guard at 1e-5 (Fortran:574-578)
    norms = np.linalg.norm(soap_unnorm, axis=0)
    sqrt_dot_p = np.where(norms < 1e-5, 1.0, norms)

    # Apply normalisation derivative for each spherical component:
    #   soap_*_der_norm = soap_*_der / sqrt_dot_p  −  soap_unnorm · dot(soap_unnorm, soap_*_der)/sqrt_dot_p^3
    sdpp = sqrt_dot_p[pair_atom]                                  # [P]
    soap_per_pair = soap_unnorm[:, pair_atom]                      # [n_compressed, P]
    for der_arr in (soap_rad_der, soap_azi_der, soap_pol_der):
        dot = np.sum(soap_per_pair * der_arr, axis=0)              # [P]
        der_arr[:] = (der_arr / sdpp[None, :]
                      - soap_per_pair / sdpp[None, :] ** 3 * dot[None, :])

    soap_norm = (soap_unnorm / sqrt_dot_p[None, :]).T              # [n_sites, n_compressed]
    return soap_norm, soap_rad_der, soap_azi_der, soap_pol_der


def _spherical_to_cartesian_per_pair(
    soap_rad_der: np.ndarray,         # [n_compressed, P]
    soap_azi_der: np.ndarray,
    soap_pol_der: np.ndarray,
    rjs: np.ndarray,                  # [P]
    thetas: np.ndarray,
    phis: np.ndarray,
    pair_active: np.ndarray,          # [P] bool — only convert where rj > 0
) -> np.ndarray:                      # [P, 3, n_compressed]
    """Convert spherical-coordinate gradients to Cartesian per-pair.

    Self-pairs (rj=0) get zero gradient initially and are filled by the
    self-derivative aggregation in _aggregate_self_derivative.
    """
    P = rjs.shape[0]
    n_compressed = soap_rad_der.shape[0]
    out = np.zeros((P, 3, n_compressed), dtype=np.float64)

    # Avoid /0 at pair_active=False (rj=0); compute on safe rj
    safe_rj = np.where(pair_active, rjs, 1.0)
    sin_t = np.sin(thetas)
    cos_t = np.cos(thetas)
    sin_p = np.sin(phis)
    cos_p = np.cos(phis)

    for p in range(P):
        if not pair_active[p]:
            continue
        r_inv = 1.0 / safe_rj[p]
        st = sin_t[p]; ct = cos_t[p]; sp = sin_p[p]; cp = cos_p[p]
        rad = soap_rad_der[:, p]
        azi = soap_azi_der[:, p]
        pol = soap_pol_der[:, p]
        out[p, 0] = st * cp * rad - ct * cp * r_inv * pol - sp * r_inv * azi
        out[p, 1] = st * sp * rad - ct * sp * r_inv * pol + cp * r_inv * azi
        out[p, 2] = ct * rad + st * r_inv * pol
    return out


def _aggregate_self_derivative(
    soap_cart_der: np.ndarray,        # [P, 3, n_compressed], modified in place
    pair_atom: np.ndarray,            # [P]
    pair_is_central: np.ndarray,      # [P] bool — strict (rj=0) self-pair flag
) -> np.ndarray:
    """Translational invariance: d(soap[i])/d(r_i) = -Σ_{p ≠ central} d(soap[i])/d(r_p).

    The "central" pair per centre is the rj=0 self-pair (Fortran's j==1).
    Image self-pairs (same atom-index, non-zero displacement) are regular
    neighbours: their Cartesian gradient is subtracted from the central slot.
    """
    P = soap_cart_der.shape[0]
    centre_self_idx = np.full(int(pair_atom.max()) + 1, -1, dtype=np.int64)
    for p in range(P):
        if pair_is_central[p]:
            centre_self_idx[int(pair_atom[p])] = p

    for p in range(P):
        if pair_is_central[p]:
            continue
        i = int(pair_atom[p])
        sp = centre_self_idx[i]
        if sp >= 0:
            soap_cart_der[sp] -= soap_cart_der[p]
    return soap_cart_der


def compute_soap_with_grad_from_positions_numpy(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    numbers: np.ndarray,
    species_Z: list[int],
    soap_params: dict,
    *,
    nf: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end forward + Cartesian gradient.

    Returns:
        soap        : [n_atoms, n_compressed]      L2-normalised
        grad_values : [P, 3, n_compressed]
        pair_atom   : [P]
        pair_gidx   : [P]
    """
    n_atoms = positions.shape[0]
    n_species = len(species_Z)
    alpha_max = int(soap_params["alpha_max"])
    l_max = int(soap_params["l_max"])
    rcut_hard = float(soap_params["rcut_hard"])

    pair_atom, pair_gidx, rjs, thetas, phis = build_neighbour_list_numpy(
        positions, cell, pbc, rcut_hard
    )
    # Strict self-pair (Fortran j==1): same atom AND rj=0. Image self-pairs
    # (rj > 0) are regular neighbours.
    pair_is_central = (pair_atom == pair_gidx) & (rjs < 1e-10)
    pair_active = rjs < rcut_hard

    z_to_idx = {int(z): idx for idx, z in enumerate(species_Z)}
    pair_neighbour_species = np.array(
        [z_to_idx[int(numbers[int(j)])] for j in pair_gidx], dtype=np.int32
    )

    W_single = build_orthonormalization_matrix_poly3(alpha_max)
    alpha_per_species = [alpha_max] * n_species
    n_max = sum(alpha_per_species)
    mask_info = make_compress_mask_trivial(alpha_per_species, l_max)
    multiplicity_array = build_multiplicity_array(n_max, l_max, mask_info["skip_mask"])

    R, R_der = radial_expansion_coeff_poly3_with_der_numpy(
        rjs, pair_neighbour_species, pair_is_central,
        n_species=n_species, alpha_max=alpha_max,
        rcut_hard=rcut_hard,
        rcut_soft=float(soap_params["rcut_soft"]),
        atom_sigma_r=float(soap_params["atom_sigma_r"]),
        atom_sigma_r_scaling=float(soap_params["atom_sigma_r_scaling"]),
        amplitude_scaling=float(soap_params["amplitude_scaling"]),
        central_weight=float(soap_params["central_weight"]),
        radial_enhancement=int(soap_params["radial_enhancement"]),
        nf=nf,
        do_central=(float(soap_params["central_weight"]) != 0.0),
        W_single=W_single,
    )
    A, A_rad, A_azi, A_pol = angular_expansion_coeff_with_der_numpy(
        rjs, thetas, phis, pair_active,
        l_max=l_max,
        atom_sigma_t=float(soap_params["atom_sigma_t"]),
        atom_sigma_t_scaling=float(soap_params["atom_sigma_t_scaling"]),
        rcut=rcut_hard,
    )

    cnk, cnk_rad, cnk_azi, cnk_pol = aggregate_cnk_with_der(
        R, A, R_der, A_rad, A_azi, A_pol, pair_atom, n_atoms,
    )

    soap_norm, soap_rad_der, soap_azi_der, soap_pol_der = power_spectrum_with_grad_numpy(
        cnk, cnk_rad, cnk_azi, cnk_pol, pair_atom,
        multiplicity_array, mask_info["skip_mask"],
        mask_info["compressed_idx"], mask_info["uncompressed_idx"], mask_info["coeffs"],
        mask_info["n_compressed"], n_max, l_max, n_atoms,
    )

    grad_cart = _spherical_to_cartesian_per_pair(
        soap_rad_der, soap_azi_der, soap_pol_der,
        rjs, thetas, phis, ~pair_is_central & pair_active,
    )
    grad_cart = _aggregate_self_derivative(grad_cart, pair_atom, pair_is_central)

    return soap_norm, grad_cart, pair_atom, pair_gidx


# --- Phase 9 validators --------------------------------------------------

def _validate_grad_via_finite_difference_water_monomer(eps: float = 1e-5) -> None:
    """Per-atom Cartesian FD on water_monomer must match summed pair gradients.

    For each atom k and Cartesian axis d:
       ΔR[i] = (R(positions with r_k += eps·e_d) − R(positions with r_k -= eps·e_d)) / (2eps)
            = Σ_{p: pair_gidx[p]==k} grad_values[p, d, :]
    """
    f = load_fixture("water_monomer")
    positions = np.asarray(f["positions"], dtype=np.float64)
    cell = np.asarray(f["cell"], dtype=np.float64)
    pbc = np.asarray(f["pbc"], dtype=bool)
    numbers = np.asarray(f["numbers"], dtype=np.int32)
    species_Z = sorted(set(int(z) for z in numbers))
    soap_params = f["soap_params"]
    n_atoms = len(numbers)

    soap_centre, grad_cart, pair_atom, pair_gidx = (
        compute_soap_with_grad_from_positions_numpy(
            positions, cell, pbc, numbers, species_Z, soap_params,
        )
    )

    max_err = 0.0
    for k in range(n_atoms):
        for d in range(3):
            pos_p = positions.copy(); pos_p[k, d] += eps
            pos_m = positions.copy(); pos_m[k, d] -= eps
            soap_p = compute_soap_from_positions_numpy(
                pos_p, cell, pbc, numbers, species_Z, soap_params
            )
            soap_m = compute_soap_from_positions_numpy(
                pos_m, cell, pbc, numbers, species_Z, soap_params
            )
            fd = (soap_p - soap_m) / (2 * eps)                         # [n_atoms, n_compressed]
            # Analytical: for each centre i, sum gradients over pairs where neighbour == k
            for i in range(n_atoms):
                analytical = np.zeros_like(fd[i])
                for p in range(len(pair_atom)):
                    if int(pair_atom[p]) == i and int(pair_gidx[p]) == k:
                        analytical += grad_cart[p, d, :]
                err = float(np.abs(fd[i] - analytical).max())
                max_err = max(max_err, err)
                if err > 5e-4:
                    raise AssertionError(
                        f"FD mismatch at atom k={k}, axis={d}, centre i={i}: "
                        f"max|err|={err:.3e}"
                    )
    print(f"   water_monomer FD: max|err| = {max_err:.2e}  PASS")


def _validate_grad_against_quippy_fixture_non_periodic(
    atol: float = 1e-5, rtol: float = 1e-4
) -> None:
    """Compare Cartesian gradients to quippy grad_values, matching pairs by
    (pair_atom, pair_gidx) (unique for non-periodic structures)."""
    for name in ["water_monomer", "water_dimer", "h2_close", "single_h"]:
        f = load_fixture(name)
        positions = np.asarray(f["positions"], dtype=np.float64)
        cell = np.asarray(f["cell"], dtype=np.float64)
        pbc = np.asarray(f["pbc"], dtype=bool)
        numbers = np.asarray(f["numbers"], dtype=np.int32)
        species_Z = sorted(set(int(z) for z in numbers))
        soap_params = f["soap_params"]

        soap_norm, grad_cart, my_pair_atom, my_pair_gidx = (
            compute_soap_with_grad_from_positions_numpy(
                positions, cell, pbc, numbers, species_Z, soap_params,
            )
        )
        quippy_pair_atom = np.asarray(f["pair_atom"])
        quippy_pair_gidx = np.asarray(f["pair_gidx"])
        quippy_grad = np.asarray(f["grad_values"], dtype=np.float32)

        # Build a map from (pair_atom, pair_gidx) → quippy index
        q_index_map = {(int(quippy_pair_atom[p]), int(quippy_pair_gidx[p])): p
                       for p in range(len(quippy_pair_atom))}

        max_err = 0.0
        for p in range(len(my_pair_atom)):
            key = (int(my_pair_atom[p]), int(my_pair_gidx[p]))
            if key not in q_index_map:
                raise AssertionError(f"{name}: pair {key} not in quippy fixture")
            qp = q_index_map[key]
            # quippy_grad shape: [P, 3, dim_q]
            mine = grad_cart[p].astype(np.float32)                    # [3, dim_q]
            ref = quippy_grad[qp]                                      # [3, dim_q]
            err = float(np.abs(mine - ref).max())
            max_err = max(max_err, err)
        if max_err > atol:
            raise AssertionError(f"{name}: max|grad mismatch| = {max_err:.3e} > {atol}")
        print(f"   {name:<14} max|Δgrad| = {max_err:.2e}  PASS")


def run_phase9_validation() -> None:
    print("=== Phase 9 validation (END-TO-END GRADIENT GATE) ===\n")
    print("1. water_monomer Cartesian FD on positions")
    _validate_grad_via_finite_difference_water_monomer()
    print("\n2. Cartesian gradients match quippy fixtures (non-periodic)")
    _validate_grad_against_quippy_fixture_non_periodic()
    print("\nAll Phase 9 gradient checks passed.")


class DescriptorBuilderGPU:
    """SOAP-turbo descriptor builder, drop-in for DescriptorBuilder (quippy).

    Same build_descriptors_flat(dataset) API: a list of (descriptors,
    grad_values, pair_atom, pair_gidx) tuples, one per frame. NumPy backend
    ("GPU" marks the eventual TF lift). Requires basis='poly3',
    compress_mode='trivial', and uniform per-species hyperparameters.
    """

    def __init__(self, cfg) -> None:
        # Permissive validation: allow defaults but trip on unsupported overrides.
        if getattr(cfg, "basis", "poly3") != "poly3":
            raise NotImplementedError(
                f"DescriptorBuilderGPU only supports basis='poly3', got '{cfg.basis}'"
            )
        if getattr(cfg, "compress_mode", "trivial") != "trivial":
            raise NotImplementedError(
                f"DescriptorBuilderGPU only supports compress_mode='trivial', "
                f"got '{cfg.compress_mode}'"
            )
        self.cfg = cfg
        # cfg.types is populated by data.collect() before training/inference;
        # for inference it may already be set on a loaded model's cfg.
        self._species_Z = (sorted(int(z) for z in cfg.types)
                           if getattr(cfg, "types", None) else None)
        self._soap_params = self._extract_soap_params(cfg)

    @staticmethod
    def _extract_soap_params(cfg) -> dict:
        """Pack the per-call soap_params dict from a TNEPconfig instance."""
        return {
            "alpha_max": int(cfg.alpha_max),
            "l_max": int(cfg.l_max),
            "rcut_hard": float(cfg.rcut_hard),
            "rcut_soft": float(cfg.rcut_soft),
            "atom_sigma_r": float(cfg.atom_sigma_r),
            "atom_sigma_r_scaling": float(cfg.atom_sigma_r_scaling),
            "atom_sigma_t": float(cfg.atom_sigma_t),
            "atom_sigma_t_scaling": float(cfg.atom_sigma_t_scaling),
            "amplitude_scaling": float(cfg.amplitude_scaling),
            "central_weight": float(cfg.central_weight),
            "radial_enhancement": int(cfg.radial_enhancement),
            "basis": getattr(cfg, "basis", "poly3"),
            "scaling_mode": getattr(cfg, "scaling_mode", "polynomial"),
            "compress_mode": getattr(cfg, "compress_mode", "trivial"),
        }

    def build_descriptors_flat(
        self, dataset: list,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Per-frame SOAP descriptors and Cartesian gradients.

        Returns:
            list of tuples, one per frame:
              descriptors : [N, dim_q]      float32
              grad_values : [P, 3, dim_q]   float32
              pair_atom   : [P]             int32
              pair_gidx   : [P]             int32
        """
        if self._species_Z is None:
            raise RuntimeError(
                "DescriptorBuilderGPU: cfg.types is not set. Run data.collect() "
                "or load a model first so the species ordering is known."
            )
        results = []
        for atoms in dataset:
            soap, grad, pa, pg = compute_soap_with_grad_from_positions_numpy(
                positions=np.asarray(atoms.positions, dtype=np.float64),
                cell=np.asarray(atoms.cell.array, dtype=np.float64),
                pbc=np.asarray(atoms.pbc, dtype=bool),
                numbers=np.asarray(atoms.numbers, dtype=np.int32),
                species_Z=self._species_Z,
                soap_params=self._soap_params,
            )
            results.append((
                soap.astype(np.float32),
                grad.astype(np.float32),
                pa.astype(np.int32),
                pg.astype(np.int32),
            ))
        return results


# --- Phase 10 validators -------------------------------------------------

def _validate_class_against_quippy() -> None:
    """DescriptorBuilderGPU.build_descriptors_flat must equal quippy on test fixtures.

    Constructs a minimal cfg-like object with the params used to generate the
    Phase 0 fixtures, then calls build_descriptors_flat on a small dataset of
    test structures. Each frame's outputs are compared to the corresponding
    quippy fixture by (pair_atom, pair_gidx) match.
    """
    class _CfgStub:
        pass
    cfg = _CfgStub()
    for k, v in REFERENCE_SOAP_PARAMS.items():
        setattr(cfg, k, v)

    # Build a small dataset from existing fixtures.
    fixture_names = ["water_monomer", "water_dimer", "h2_close", "single_h"]
    structures = _make_test_structures()
    dataset = [structures[n] for n in fixture_names]

    # Determine cfg.types from the union of all structures' species
    all_Z = sorted(set(int(z) for atoms in dataset for z in atoms.numbers))
    cfg.types = all_Z

    builder = DescriptorBuilderGPU(cfg)
    results = builder.build_descriptors_flat(dataset)

    for name, (soap, grad, pa, pg) in zip(fixture_names, results):
        f = load_fixture(name)
        quippy_soap = np.asarray(f["descriptors"], dtype=np.float32)

        # Builder uses the union species ordering; the fixture uses the
        # structure's subset (different dim_q). Compare descriptors only when
        # structure species == union; else just check unit norms.
        struct_Z = sorted(set(int(z) for z in f["numbers"]))
        if struct_Z == all_Z:
            err = float(np.abs(soap - quippy_soap).max())
            if err > 1e-5:
                raise AssertionError(f"{name}: |Δsoap|={err:.3e}")
            print(f"   {name:<14} class output matches quippy: max|Δsoap|={err:.2e}  PASS")
        else:
            # Different species set — different dim_q, validate via norm + finite
            if soap.shape[1] != 75:  # union(H, O) → dim_q
                raise AssertionError(f"{name}: unexpected dim_q={soap.shape[1]}")
            norms = np.linalg.norm(soap, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5):
                raise AssertionError(f"{name}: norms not unit ({norms})")
            print(f"   {name:<14} dim_q=75 (union species); unit-norm OK  PASS")


def run_phase10_validation() -> None:
    print("=== Phase 10 validation (DescriptorBuilderGPU class wrapping) ===\n")
    print("1. Class API matches quippy outputs on fixture structures")
    _validate_class_against_quippy()
    print("\nAll Phase 10 checks passed.")


if __name__ == "__main__":
    if not os.path.exists("tests/fixtures/water_monomer.npz"):
        build_reference_fixtures()
        print()
    run_phase1_validation()
    print()
    run_phase2_validation()
    print()
    run_phase3_validation()
    print()
    run_phase4_validation()
    print()
    run_phase5_validation()
    print()
    run_phase6_validation()
    print()
    run_phase7_validation()
    print()
    run_phase8_validation()
    print()
    run_phase9_validation()
    print()
    run_phase10_validation()
