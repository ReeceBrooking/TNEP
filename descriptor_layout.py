"""Descriptor layout rules (dim_q, species-pair blocks, q_scale_*, learned
radial basis) shared by TNEP/SNES/spectroscopy, plus the NumPy SOAP-turbo
tables the TF builder reuses (poly3 orthonormalisation, compress masks,
multiplicity, _N_a, _get_preflm, neighbour list).
"""

from __future__ import annotations

import numpy as np


def build_overlap_matrix_poly3(alpha_max: int) -> np.ndarray:
    """Poly3 overlap S[i,j] = sqrt((5+2i)(5+2j))/(5+i+j), 1-indexed, [α, α]; SPD for α ≲ 10."""
    a = np.arange(1, alpha_max + 1, dtype=np.float64)            # 1-indexed
    i_idx, j_idx = np.meshgrid(a, a, indexing="ij")              # both [α, α]
    S = np.sqrt((5.0 + 2.0 * i_idx) * (5.0 + 2.0 * j_idx)) / (5.0 + i_idx + j_idx)
    return S


def build_orthonormalization_matrix_poly3(alpha_max: int) -> np.ndarray:
    """W = S^(-1/2) via SVD; matches the Fortran SVD+Cholesky (soap_turbo_radial.f90:780-816)."""
    S = build_overlap_matrix_poly3(alpha_max)
    U, svd, Vt = np.linalg.svd(S)                                 # S = U·diag(svd)·Vt
    inv_sqrt = 1.0 / np.sqrt(svd)
    W = U @ np.diag(inv_sqrt) @ Vt
    return W


def compute_dim_q(cfg) -> int:
    """ANN input width: soap_dim (+ NEP block under "concat"; NEP alone under "replace")."""
    from nep_descriptor import nep_dim
    mode = str(getattr(cfg, "nep_descriptor", "off"))
    if mode == "replace":
        return int(nep_dim(cfg))
    return soap_dim(cfg) + (int(nep_dim(cfg)) if mode == "concat" else 0)


def soap_dim(cfg) -> int:
    """Closed-form SOAP width from (alpha_max, l_max, num_types, compress_mode)."""
    alpha_max = int(cfg.alpha_max)
    l_max = int(cfg.l_max)
    n_species = int(cfg.num_types)
    if _radial_species_sum(cfg):
        return alpha_max * (alpha_max + 1) // 2 * (l_max + 1)
    compress_mode = getattr(cfg, "compress_mode", "trivial")
    if compress_mode == "none":
        # Unordered (n, n') over N = T·alpha_max channels × (l_max+1); verified vs quippy.
        N = n_species * alpha_max
        return int(N * (N + 1) // 2 * (l_max + 1))
    if compress_mode == "linear":
        # quippy linear default: 2 × n_species pivots × (l_max+1)
        return int(2 * n_species * (l_max + 1))
    alpha_per_species = [alpha_max] * n_species
    mask_info = make_compress_mask_trivial(alpha_per_species, l_max)
    return int(mask_info["n_compressed"])


def radial_basis_init(alpha_max: int, n_basis: int) -> np.ndarray:
    """M0 [alpha_max, n_basis] = [W_α | 0] · W_K⁻¹: the poly3 orthonormal
    functions in the K-orthonormal set, so M0 @ φ_K == φ_α exactly; identity
    when n_basis == alpha_max. W_K⁻¹ = S_K^{1/2} from the SVD (cond ~ 1e6 at K = 8)."""
    if not alpha_max <= n_basis <= 10:
        raise ValueError(
            f"radial_basis_size={n_basis} must satisfy alpha_max={alpha_max} "
            f"<= K <= 10 (poly3 overlap is SPD only up to ~10).")
    W_a = build_orthonormalization_matrix_poly3(alpha_max)
    U, svd, Vt = np.linalg.svd(build_overlap_matrix_poly3(n_basis))
    W_K_inv = U @ np.diag(np.sqrt(svd)) @ Vt
    pad = np.zeros((alpha_max, n_basis - alpha_max), dtype=np.float64)
    return np.concatenate([W_a, pad], axis=1) @ W_K_inv


def _radial_species_sum(cfg) -> bool:
    """True for the species-summed learned radial basis (spec §4b)."""
    return (getattr(cfg, "radial_basis_size", None) is not None
            and bool(getattr(cfg, "radial_basis_species_sum", False)))


def extra_block_widths(cfg) -> list:
    """Widths of blocks appended after the SOAP blocks (NEP sub-blocks under "concat"/"replace")."""
    if str(getattr(cfg, "nep_descriptor", "off")) in ("concat", "replace"):
        return nep_block_widths(cfg)
    return []


def nep_block_widths(cfg) -> list:
    """NEP sub-block widths in channel order: radial, L1..L4, 4-body, 5-body."""
    from nep_descriptor import nep_slices
    return [sl.stop - sl.start for k, sl in nep_slices(cfg).items() if k != "angular"]


def radial_basis_channel_map(alpha_max: int, l_max: int, n_species: int,
                             species_sum: bool = False) -> dict:
    """Map from the row Gram G[row, n', l] to Fortran (n, n', l) channel order.
    species_sum=False: rows = species pivots, channels where n or n' is a pivot
    (make_compress_mask_trivial order); True: n_max = alpha_max, all n <= n'.

    Returns rows [R], n_max, gather_idx [dim_q] into G flat (R, n_max, L),
    scale [dim_q] (√2 for n != n'), Wm [k_max, L] (2 for m > 0 else 1; k = l(l+1)/2 + m)."""
    L = l_max + 1
    k_max = L * (L + 1) // 2
    if species_sum:
        n_max = alpha_max
        rows = list(range(alpha_max))
    else:
        n_max = alpha_max * n_species
        rows = [s * alpha_max for s in range(n_species)]
    row_of = {n: i for i, n in enumerate(rows)}
    gather: list[int] = []
    scale: list[float] = []
    for n in range(n_max):
        for nprime in range(n, n_max):
            if n not in row_of and nprime not in row_of:
                continue
            i, other = ((row_of[n], nprime) if n in row_of
                        else (row_of[nprime], n))
            for l in range(L):
                gather.append((i * n_max + other) * L + l)
                scale.append(np.sqrt(2.0) if n != nprime else 1.0)
    Wm = np.zeros((k_max, L), dtype=np.float32)
    k = 0
    for l in range(L):
        for m in range(l + 1):
            Wm[k, l] = 2.0 if m > 0 else 1.0
            k += 1
    return dict(
        rows=np.asarray(rows, dtype=np.int32),
        n_max=int(n_max),
        gather_idx=np.asarray(gather, dtype=np.int32),
        scale=np.asarray(scale, dtype=np.float32),
        Wm=Wm,
    )


Q_SCALE_ALPHA = 0.5    # equalisation exponent (0 = no scaling, 1 = full)
Q_SCALE_CLAMP = 10.0   # max boost / damping of any channel


def soap_scale_width(cfg) -> int:
    """Leading SOAP channel count in the scaler vector (offset of the appended blocks)."""
    lay = descriptor_block_layout(cfg)
    return int(sum(len(v) for v in lay["l_index"].values()))


def q_scale_groups(cfg) -> list:
    """descriptor_scale="block" groups: each SOAP l group, then each appended
    block; every channel once. Under "sum" the scaler sees [q_soap ‖ q_experts],
    so the expert blocks follow."""
    lay = descriptor_block_layout(cfg)
    groups = [np.asarray(v, dtype=np.int32) for _, v in sorted(lay["l_index"].items())
              if len(v)]
    start = sum(len(g) for g in groups)
    widths = extra_block_widths(cfg)
    if str(getattr(cfg, "nep_descriptor", "off")) == "sum":
        widths = widths + nep_block_widths(cfg)
    for w in widths:
        groups.append(np.arange(start, start + w, dtype=np.int32))
        start += w
    return groups


def q_scale_model_slices(cfg, Q: int) -> list:
    """Scaler-vector slice per ANN: [0, Q), or under "sum" main slice + one per extra model."""
    if str(getattr(cfg, "nep_descriptor", "off")) != "sum":
        return [slice(0, Q)]
    dq = compute_dim_q(cfg)
    from nep_descriptor import nep_slices
    sls = [nep_slices(cfg)[k] for k in ("radial", "angular")]
    return [slice(0, dq)] + [slice(dq + sl.start, dq + sl.stop) for sl in sls]


def compute_q_scale(q: np.ndarray, Z: np.ndarray, atom_mask: np.ndarray,
                    cfg) -> tuple:
    """Descriptor preconditioning (m [T, Q], μ [T, Q]) from gen-0 q over real
    atoms, one row per centre species (O/H channel spreads differ ~6x).
    Per species μ_k = mean, and from the channel std s_k:
        g_group = 1 / median(s_group)
        r_k     = clip((1 / (g·s_k))^ALPHA, 1/CLAMP, CLAMP)        ("channel")
        m_k     = g_group · r_k ("channel")  or  g_group ("block")
    then each ANN slice is divided by its median input norm (unit length at
    the median). Dead channels / absent species keep m = 1, μ = 0.
    "minmax": GPUMD's species-blind q_scaler, m_k = 1/(max_k − min_k), μ = 0."""
    T = int(cfg.num_types)
    Q = q.shape[-1]
    real = np.asarray(atom_mask).reshape(-1) > 0
    x = np.asarray(q, dtype=np.float64).reshape(-1, Q)[real]
    z = np.asarray(Z).reshape(-1)[real]
    if str(cfg.descriptor_scale) == "minmax":
        m, mu = _q_scale_rows(x, cfg)
        return np.tile(m, (T, 1)), np.tile(mu, (T, 1))
    m = np.ones((T, Q), np.float32)
    mu = np.zeros((T, Q), np.float32)
    for t in range(T):
        if np.any(z == t):
            m[t], mu[t] = _q_scale_rows(x[z == t], cfg)
    return m, mu


def _q_scale_rows(x: np.ndarray, cfg) -> tuple:
    """compute_q_scale's rule on the rows x [N, Q] of one species: (m [Q], μ [Q])."""
    Q = x.shape[-1]
    if str(cfg.descriptor_scale) == "minmax":
        rng = x.max(0) - x.min(0)
        m = 1.0 / np.where(rng > 0, rng, 1.0)
        # GPUMD's q_scaler covers the NEP descriptor only; the SOAP block is
        # already L2-normalised, so pin its channels at 1.
        sw = soap_scale_width(cfg)
        if sw >= Q:
            raise ValueError(
                "descriptor_scale='minmax' is GPUMD's q_scaler, which is "
                "defined on the NEP descriptor only, but this config has no "
                "NEP block (nep_descriptor="
                f"{str(getattr(cfg, 'nep_descriptor', 'off'))!r}) — the "
                "scaler would be a silent no-op. Use 'block' / 'channel' for "
                "the SOAP input, or set nep_descriptor.")
        m[:sw] = 1.0
        return m.astype(np.float32), np.zeros(Q, np.float32)
    mu = x.mean(axis=0)
    s = x.std(axis=0)
    live = s > 0
    if not np.any(live):
        return np.ones(Q, dtype=np.float32), mu.astype(np.float32)
    g = np.ones(Q, dtype=np.float64)
    for idx in q_scale_groups(cfg):
        sg = s[idx][live[idx]]
        if sg.size:
            g[idx] = 1.0 / float(np.median(sg))
    m = g.copy()
    if str(cfg.descriptor_scale) == "channel":
        s1 = s * g
        r = np.clip((1.0 / np.where(live, s1, 1.0)) ** Q_SCALE_ALPHA,
                    1.0 / Q_SCALE_CLAMP, Q_SCALE_CLAMP)
        m = g * r
    # Unit median norm per model so one SNES step σ moves every model's
    # pre-activation by ~σ (per-channel unit std left a 2-body model 12x under-stepped).
    for sl in q_scale_model_slices(cfg, Q):
        n = float(np.median(np.linalg.norm((x[:, sl] - mu[sl]) * np.where(live[sl], m[sl], 0.0), axis=1)))
        if n > 0:
            m[sl] /= n
    m[~live] = 1.0
    return m.astype(np.float32), mu.astype(np.float32)


def soap_descriptors_needed(cfg) -> bool:
    """Whether the SOAP spectrum is built at all: False under "replace", where
    nothing reads it (building it anyway cost ~150 ms/frame of startup)."""
    return str(getattr(cfg, "nep_descriptor", "off")) != "replace"


def staged_descriptor_width(cfg) -> int:
    """Width of the staged `descriptors` slot: 0 when SOAP is skipped, else the builder's."""
    return descriptor_input_width(cfg) if soap_descriptors_needed(cfg) else 0


def descriptor_input_width(cfg) -> int:
    """Builder's emitted width: compressed SOAP dim, or under radial_basis_size
    the flat raw coefficients S·K·k_max·2 (species, κ, (l,m), re/im). For what
    a data dict actually carries use staged_descriptor_width."""
    K = getattr(cfg, "radial_basis_size", None)
    if K is None:
        return soap_dim(cfg)
    L = int(cfg.l_max) + 1
    return int(cfg.num_types) * int(K) * (L * (L + 1) // 2) * 2


def descriptor_block_layout(cfg) -> dict:
    """Group the dim_q axis by unordered species pair (s_a, s_b): T·(T+1)/2
    blocks whose channels interleave along the Fortran (n, n', l) order, so
    callers gather by q index. 'trivial' is a subset mask of 'none'
    (soap_turbo.f90:516); 'linear' raises (not pair-separable).

    Returns pair_keys, pair_q_index, pair_ln_index, l_index, block_sizes,
    max_block_size, alpha_eff_per_pair, max_alpha_eff, N_per_l, dim_q.
    """
    compress_mode = getattr(cfg, "compress_mode", "trivial")
    if compress_mode not in ("trivial", "none"):
        raise NotImplementedError(
            f"descriptor_block_layout supports compress_mode 'trivial' and "
            f"'none' (got {compress_mode!r}); 'linear' compression's "
            f"channel-to-species-pair mapping is not separable")
    alpha_max = int(cfg.alpha_max)
    l_max = int(cfg.l_max)
    n_species = int(cfg.num_types)
    if str(getattr(cfg, "nep_descriptor", "off")) == "replace":
        # NEP block only: no SOAP pair blocks (mixing rejected in this mode).
        return dict(
            pair_keys=[], pair_q_index={}, pair_ln_index={},
            l_index={l: np.zeros(0, dtype=np.int32) for l in range(l_max + 1)},
            block_sizes={}, max_block_size=0, alpha_eff_per_pair={},
            max_alpha_eff=0, N_per_l=0,
            dim_q=sum(extra_block_widths(cfg)),
        )
    if _radial_species_sum(cfg):
        # Species-summed basis: one block, (n, n') outer and l inner.
        L = l_max + 1
        per_l = alpha_max * (alpha_max + 1) // 2
        q = np.arange(per_l * L, dtype=np.int32).reshape(per_l, L)
        ln = {l: q[:, l].copy() for l in range(L)}
        return dict(
            pair_keys=[(0, 0)],
            pair_q_index={(0, 0): q.reshape(-1).copy()},
            pair_ln_index={(0, 0): ln},
            l_index={l: v.copy() for l, v in ln.items()},
            block_sizes={(0, 0): int(per_l * L)},
            max_block_size=int(per_l * L),
            alpha_eff_per_pair={(0, 0): per_l},
            max_alpha_eff=per_l,
            N_per_l=per_l,
            dim_q=int(per_l * L) + sum(extra_block_widths(cfg)),
        )
    alpha_per_species = [alpha_max] * n_species

    # Species owning each 1-based global radial index (n_to_species[0] unused).
    n_to_species = np.empty(sum(alpha_per_species) + 1, dtype=np.int32)
    n_to_species[0] = -1
    cursor = 1
    for s, am in enumerate(alpha_per_species):
        n_to_species[cursor:cursor + am] = s
        cursor += am

    # Pivots: first global radial index of each species
    pivots: set[int] = {1}
    for s in range(n_species - 1):
        pivots.add(max(pivots) + alpha_per_species[s])

    pair_q_index: dict[tuple[int, int], list[int]] = {}
    pair_ln_index: dict[tuple[int, int], dict[int, list[int]]] = {}  # (pair, l) -> [q]

    q_counter = 0
    n_max = sum(alpha_per_species)
    for n in range(1, n_max + 1):
        for nprime in range(n, n_max + 1):
            # 'none' keeps every (n, n'): closed under a radial basis change, unlike 'trivial'.
            kept = (compress_mode == "none"
                    or (n in pivots) or (nprime in pivots))
            for l in range(l_max + 1):
                if kept:
                    s_a = int(n_to_species[n])
                    s_b = int(n_to_species[nprime])
                    pair = (min(s_a, s_b), max(s_a, s_b))
                    pair_q_index.setdefault(pair, []).append(q_counter)
                    pair_ln_index.setdefault(pair, {}).setdefault(l, []).append(q_counter)
                    q_counter += 1
    pair_keys = sorted(pair_q_index.keys())  # deterministic order
    pair_q_index_np = {k: np.asarray(pair_q_index[k], dtype=np.int32) for k in pair_keys}
    block_sizes = {k: int(pair_q_index_np[k].size) for k in pair_keys}
    pair_ln_index_np: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for k in pair_keys:
        pair_ln_index_np[k] = {
            l: np.asarray(v, dtype=np.int32)
            for l, v in pair_ln_index[k].items()
        }
    # α_eff = radial channels per l (uniform across l within a pair).
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
    # Per-l union over pairs (cross_pair_l mixing); N_l uniform across l.
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
        dim_q=int(q_counter) + sum(extra_block_widths(cfg)),  # appended blocks outside every pair
    )


def make_compress_mask_trivial(
    alpha_max_per_species: list[int],
    l_max: int,
) -> dict:
    """Sparse projection for compress_mode='trivial' (soap_turbo_compress.f90:73-106):
    keep (n, n', l) where n or n' is a species pivot (first global radial index).

    Returns compressed_idx / uncompressed_idx / coeffs (all 1.0) [P_nonzero],
    skip_mask [n_uncompressed] (True = dropped), n_compressed, n_uncompressed.
    """
    n_species = len(alpha_max_per_species)
    n_max = sum(alpha_max_per_species)

    pivots: set[int] = {1}  # 1-based
    for i in range(n_species - 1):
        next_pivot = max(pivots) + alpha_max_per_species[i]
        pivots.add(next_pivot)

    n_uncompressed = n_max * (n_max + 1) // 2 * (l_max + 1)
    skip_mask = np.ones(n_uncompressed, dtype=bool)
    compressed_idx_list: list[int] = []
    uncompressed_idx_list: list[int] = []

    k = 0                                                         # uncompressed (n, n', l) index
    counter = 0                                                   # compressed index
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


def make_compress_mask(
    alpha_max_per_species: list[int],
    l_max: int,
    compress_mode: str = "trivial",
) -> dict:
    """'trivial' → make_compress_mask_trivial; 'none' → identity over the full spectrum."""
    if compress_mode == "trivial":
        return make_compress_mask_trivial(alpha_max_per_species, l_max)
    if compress_mode != "none":
        raise NotImplementedError(
            f"GPU SOAP supports compress_mode 'trivial' or 'none', got {compress_mode!r}")
    n_max = sum(alpha_max_per_species)
    n = n_max * (n_max + 1) // 2 * (l_max + 1)
    idx = np.arange(n, dtype=np.int32)
    return dict(compressed_idx=idx, uncompressed_idx=idx.copy(),
                coeffs=np.ones(n, dtype=np.float64),
                skip_mask=np.zeros(n, dtype=bool),
                n_compressed=int(n), n_uncompressed=int(n))


def build_multiplicity_array(
    n_max: int,
    l_max: int,
    skip_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-(n, n', l, m) multiplicities (soap_turbo.f90:496-535): √2 if n != n',
    ×2 if m > 0; m innermost, skipped channels omitted. Returns [n_active],
    n_active = Σ_{kept (n,n',l)} (l+1)."""
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


# Radial expansion reference: soap_turbo_radial.f90:69-380 (poly3 only).
def _N_a(alpha: int, rcut_hard: float = 1.0) -> float:
    """Polynomial normalisation N_a = sqrt(rcut/(2α+5)) (soap_turbo_radial.f90:38-52)."""
    return float(np.sqrt(rcut_hard / (2.0 * alpha + 5.0)))


def _get_preflm(l_max: int) -> np.ndarray:
    """Y_lm normalisation sqrt((2l+1)/4π · (l-m)!/(l+m)!) as flat [k_max],
    k = l(l+1)/2 + m (soap_turbo_functions.f90:202-228)."""
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


def build_neighbour_list_numpy(
    positions: np.ndarray,        # [n_atoms, 3]
    cell: np.ndarray,             # [3, 3]
    pbc: np.ndarray,              # [3] bool
    rcut: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Image-enumerating neighbour list (MIC would collapse multi-image pairs).

    Returns (pair_atom, pair_gidx, rjs, thetas, phis), all [P]: per centre the
    (i, i, image 0) self-pair (rj = 0) first, then every (i, j, image) with
    rj < rcut, sorted by centre.
    """
    n_atoms = positions.shape[0]
    is_periodic = bool(np.any(pbc))

    if is_periodic:
        try:
            cell_inv = np.linalg.inv(cell)
        except np.linalg.LinAlgError:
            # Degenerate cell with pbc=True: treat as aperiodic.
            is_periodic = False
            n_imgs = np.zeros(3, dtype=np.int32)
        else:
            # Reciprocal vectors are the COLUMNS of inv(cell) (axis=0); axis=1
            # under-counts images for non-orthogonal cells.
            b_norms = np.linalg.norm(cell_inv, axis=0)
            n_imgs = np.where(pbc, np.ceil(rcut * b_norms).astype(np.int32), 0)
            # Wrap into the primary cell (image search spans only rcut), periodic
            # axes only: wrapping a slab's non-periodic axis corrupts the descriptor.
            frac = positions @ cell_inv
            frac -= np.floor(frac) * np.asarray(pbc, dtype=frac.dtype)
            positions = frac @ cell
    else:
        n_imgs = np.zeros(3, dtype=np.int32)

    nx, ny, nz = int(n_imgs[0]), int(n_imgs[1]), int(n_imgs[2])
    image_int = np.stack(np.meshgrid(
        np.arange(-nx, nx + 1),
        np.arange(-ny, ny + 1),
        np.arange(-nz, nz + 1),
        indexing="ij",
    ), axis=-1).reshape(-1, 3)                                       # [N_img, 3] int
    image_vectors = image_int.astype(np.float64) @ cell              # [N_img, 3]
    zero_image_idx = int(np.argmin(np.linalg.norm(image_vectors, axis=1)))

    # disps[i, j, k] = positions[j] + image_vectors[k] - positions[i]   [N, N, N_img, 3]
    disps = (positions[None, :, None, :]
             + image_vectors[None, None, :, :]
             - positions[:, None, None, :])
    rs = np.linalg.norm(disps, axis=-1)                              # [N, N, N_img]

    keep = rs < rcut
    eye = np.eye(n_atoms, dtype=bool)  # self-pair is prepended below, not counted here
    keep[eye, zero_image_idx] = False

    i_idx, j_idx, k_idx = np.where(keep)
    disp_flat = disps[i_idx, j_idx, k_idx]                           # [P_neigh, 3]
    rs_flat = rs[i_idx, j_idx, k_idx]

    order = np.argsort(i_idx, kind="stable")  # group by centre
    i_neigh = i_idx[order].astype(np.int32)
    j_neigh = j_idx[order].astype(np.int32)
    disp_neigh = disp_flat[order]
    rs_neigh = rs_flat[order]

    # Spherical angles; rj=0 entries get theta=phi=0.
    inv_r = np.where(rs_neigh > 1e-10, 1.0 / np.maximum(rs_neigh, 1e-300), 0.0)
    cos_theta = np.clip(disp_neigh[:, 2] * inv_r, -1.0, 1.0)
    thetas_neigh = np.where(rs_neigh > 1e-10, np.arccos(cos_theta), 0.0)
    phis_neigh = np.where(rs_neigh > 1e-10,
                          np.arctan2(disp_neigh[:, 1], disp_neigh[:, 0]),
                          0.0)

    # Self-pair first in each centre's block; scatter-assign the flat arrays.
    n_per_centre = np.bincount(i_neigh, minlength=n_atoms).astype(np.int32)
    block_offsets = np.concatenate(([0], np.cumsum(n_per_centre + 1)))   # [n_atoms+1]
    centre_self_pos = block_offsets[:-1]                                  # [n_atoms]
    P = int(block_offsets[-1])
    pair_atom = np.empty(P, dtype=np.int32)
    pair_gidx = np.empty(P, dtype=np.int32)
    rjs = np.empty(P, dtype=np.float64)
    thetas = np.empty(P, dtype=np.float64)
    phis = np.empty(P, dtype=np.float64)

    pair_atom[centre_self_pos] = np.arange(n_atoms, dtype=np.int32)
    pair_gidx[centre_self_pos] = np.arange(n_atoms, dtype=np.int32)
    rjs[centre_self_pos] = 0.0
    thetas[centre_self_pos] = 0.0
    phis[centre_self_pos] = 0.0

    # Neighbour slot = block_offsets[i] + 1 + per-centre running index.
    intra_idx = np.arange(len(i_neigh), dtype=np.int32) - np.cumsum(n_per_centre)[i_neigh] + n_per_centre[i_neigh]
    neigh_pos = block_offsets[i_neigh] + 1 + intra_idx
    pair_atom[neigh_pos] = i_neigh
    pair_gidx[neigh_pos] = j_neigh
    rjs[neigh_pos] = rs_neigh
    thetas[neigh_pos] = thetas_neigh
    phis[neigh_pos] = phis_neigh
    return pair_atom, pair_gidx, rjs, thetas, phis


# Gradient chain (TF builder) reference: soap_turbo.f90:583-697, 778-822.
# Conventions: rad_der = d/d(rj); pol_der = −d/d(theta); azi_der = d/d(phi)/sin(theta);
# self-derivative of centre i = −Σ_{j≠i} grad_pair (translational invariance).
