"""TurboGAP 2-body / 3-body distance descriptors with fixed centres.

Per-atom neighbour sums that, contracted with a linear readout, ARE TurboGAP's
2b/3b site energies (gapf90 gap.f90: get_2b_energy_and_forces 264-281,
get_3b_energy_and_forces 571-647; 3b kernel "exp" or "pol" per cfg.gap3b_kernel).
Exact identities, checked in tests/test_gap_descriptor.py against direct
transcriptions of those loops (delta = 1, cutoff(s) = 1, e0 = 0):

    E2b(sp1, sp2)      = Σ_s α_s (Σ_{i∈sp1} q2[i, sp2, s] + [sp1≠sp2] Σ_{i∈sp2} q2[i, sp1, s])
    E3b(sp0; sp1, sp2) = 2 Σ_s α_s Σ_{i∈sp0} q3[i, pair_index[sp1, sp2], s]

(the 2 is gap.f90's 3b prefactor `pref = 2 δ² α cutoff`, line 563 — q3 does
NOT absorb it). Cutoffs as in the Fortran: 2b f(r_ij); 3b f(r_ij)·f(r_ik),
both r < rcut strict, no cutoff on r_jk = |D_ik − D_ij| (image-exact).

`cfg` here is duck-typed: either a GAPconfig (the gap_dipole kernel fit) or a
TNEPconfig (the ANN nep_descriptor="gap_sum" experts). Only the gap2b_*/gap3b_*
/gap_buffer/num_types/seed fields are read, and both configs carry them.

Layout of gap_raw_numpy's [N, W2 + W3] output:
    q2 [N, T·M2]          neighbour species-major, then centre
    q3 [N, T(T+1)/2·M3]   unordered neighbour-species pair slot (pair_index), then centre
"""
import numpy as np
from ase.neighborlist import neighbor_list

MAX_FPS_SAMPLES = 200_000


def f_cut(r, rcut, buffer):
    """GAP's buffered cosine cutoff (gap.f90:270-275): 1 below rcut−buffer,
    ½(1+cos(π(r−rcut+buffer)/buffer)) in the buffer, 0 at/beyond rcut."""
    r = np.asarray(r, dtype=np.float64)
    taper = 0.5 * (1.0 + np.cos(np.pi * (r - rcut + buffer) / buffer))
    return np.where(r < rcut - buffer, 1.0, np.where(r < rcut, taper, 0.0))


def pair_index(T):
    """[T, T] symmetric packing t ≤ t′ → slot: (0,0)=0, (0,1)=1, … (same
    convention as TNEP._pair_idx "symmetric")."""
    idx = np.zeros((T, T), dtype=np.int64)
    p = 0
    for t in range(T):
        for t2 in range(t, T):
            idx[t, t2] = idx[t2, t] = p
            p += 1
    return idx


def gap_slices(cfg):
    T = cfg.num_types
    w2 = T * cfg.gap2b_n_centres
    w3 = T * (T + 1) // 2 * cfg.gap3b_n_centres
    return {"two_body": slice(0, w2), "three_body": slice(w2, w2 + w3)}


def gap_dim(cfg):
    return gap_slices(cfg)["three_body"].stop


GAP_MODELS = ("soap", "2b", "3b")
GAP_MODEL_SLICE = {"2b": "two_body", "3b": "three_body"}


def gap_models(cfg) -> tuple:
    """cfg.gap_models validated and in canonical order (soap, 2b, 3b). Under
    nep_descriptor="gap_sum" each selected model is its own ANN with its own
    species-pair readout table; the FIRST is the main ANN (so leaving "soap"
    out drops the SOAP network entirely, like nep_descriptor="replace")."""
    sel = list(getattr(cfg, "gap_models", GAP_MODELS))
    if not sel or len(set(sel)) != len(sel) or any(m not in GAP_MODELS for m in sel):
        raise ValueError(f"cfg.gap_models={sel!r}: expected a non-empty subset of {GAP_MODELS}.")
    return tuple(m for m in GAP_MODELS if m in sel)


def gap_main_slice(cfg):
    """Slice of q2‖q3 feeding the MAIN ANN when "soap" is not selected under
    nep_descriptor="gap_sum"; None when the main ANN reads SOAP."""
    if str(getattr(cfg, "nep_descriptor", "off")) != "gap_sum":
        return None
    first = gap_models(cfg)[0]
    return None if first == "soap" else gap_slices(cfg)[GAP_MODEL_SLICE[first]]


def validate_gap_cfg(cfg):
    if cfg.gap2b_n_centres < 1 or cfg.gap3b_n_centres < 1:
        raise ValueError("gap2b_n_centres and gap3b_n_centres must be ≥ 1")
    if cfg.gap2b_sigma <= 0:
        raise ValueError("gap2b_sigma must be > 0")
    s3 = np.asarray(cfg.gap3b_sigma, dtype=np.float64)
    if s3.shape != (3,) or np.any(s3 <= 0):
        raise ValueError("gap3b_sigma must be three positive widths")
    if str(getattr(cfg, "gap3b_kernel", "exp")) not in ("exp", "pol"):
        raise ValueError(f"gap3b_kernel={cfg.gap3b_kernel!r}: expected 'exp' or 'pol'")
    for rc in (cfg.gap2b_rcut, cfg.gap3b_rcut):
        if not 0 < cfg.gap_buffer <= rc:
            raise ValueError(f"gap_buffer must satisfy 0 < buffer ≤ rcut (buffer={cfg.gap_buffer}, rcut={rc})")
    if cfg.gap2b_centres is not None and np.shape(cfg.gap2b_centres) != (cfg.gap2b_n_centres,):
        raise ValueError(f"gap2b_centres must have shape [{cfg.gap2b_n_centres}]")
    if cfg.gap3b_centres is not None and np.shape(cfg.gap3b_centres) != (cfg.gap3b_n_centres, 3):
        raise ValueError(f"gap3b_centres must have shape [{cfg.gap3b_n_centres}, 3]")


# ── geometry ────────────────────────────────────────────────────────────

def _pairs(atoms, rcut):
    """Pairs with 0 < |D| < rcut (strict), sorted by centre: (i, j, D, r)."""
    i, j, D = neighbor_list("ijD", atoms, rcut)
    r = np.linalg.norm(D, axis=1)
    keep = (r > 0) & (r < rcut)
    order = np.argsort(i[keep], kind="stable")
    return i[keep][order], j[keep][order], D[keep][order], r[keep][order]


def _within(pairs, rcut):
    """Subset of a _pairs() tuple with r < rcut (order preserved)."""
    keep = pairs[3] < rcut
    return tuple(x[keep] for x in pairs)


def _triplets(N, pairs, rcut, buffer):
    """All unordered neighbour pairs j < k centred on each atom, both within
    rcut (pairs = _pairs() at any cutoff ≥ rcut). Returns (centre [P], ja [P],
    jb [P], q [P, 3], w [P]) with ja/jb the neighbour atom indices,
    q = [r_ij + r_ik, (r_ij − r_ik)², r_jk] and w = f(r_ij) f(r_ik).
    Vectorised over a padded [N, M] neighbour table."""
    i, j, D, r = _within(pairs, rcut)
    counts = np.bincount(i, minlength=N)
    M = int(counts.max()) if len(i) else 0
    table = np.full((N, M), -1, dtype=np.int64)
    # pairs are sorted by centre: rank within its centre's run = pair index − run start
    table[i, np.arange(len(i)) - np.repeat(np.cumsum(counts) - counts, counts)] = np.arange(len(i))
    mask = table >= 0
    t = np.where(mask, table, 0)                # padded slots point at pair 0; masked out by `valid`
    rt, Dt = r[t], D[t]                         # [N, M], [N, M, 3]
    valid = mask[:, :, None] & mask[:, None, :] & np.triu(np.ones((M, M), dtype=bool), k=1)
    ci, a, b = np.nonzero(valid)
    ra, rb = rt[ci, a], rt[ci, b]
    rjk = np.linalg.norm(Dt[ci, b] - Dt[ci, a], axis=1)
    q = np.stack([ra + rb, (ra - rb) ** 2, rjk], axis=1)
    w = f_cut(ra, rcut, buffer) * f_cut(rb, rcut, buffer)
    return ci, j[t[ci, a]], j[t[ci, b]], q, w


# ── descriptors ─────────────────────────────────────────────────────────

def gap_raw_numpy(atoms, types_int, cfg):
    """[N, W2 + W3] float32 per-atom q2 ‖ q3 (float64 inside)."""
    if cfg.gap2b_centres is None or cfg.gap3b_centres is None:
        raise ValueError("gap2b_centres / gap3b_centres unset: call select_centres first")
    N, T = len(atoms), cfg.num_types
    types_int = np.asarray(types_int)
    Q2 = np.asarray(cfg.gap2b_centres, dtype=np.float64)
    Q3 = np.asarray(cfg.gap3b_centres, dtype=np.float64)
    s3 = np.asarray(cfg.gap3b_sigma, dtype=np.float64)

    pairs = _pairs(atoms, max(cfg.gap2b_rcut, cfg.gap3b_rcut))
    i, j, _, r = _within(pairs, cfg.gap2b_rcut)
    k2 = f_cut(r, cfg.gap2b_rcut, cfg.gap_buffer)[:, None] * np.exp(
        -0.5 * ((r[:, None] - Q2[None]) / cfg.gap2b_sigma) ** 2)
    q2 = np.zeros((N, T, len(Q2)))
    np.add.at(q2, (i, types_int[j]), k2)

    ci, ja, jb, q, w = _triplets(N, pairs, cfg.gap3b_rcut, cfg.gap_buffer)
    # ponytail: per-dimension accumulation keeps the peak at one [P, M3] (P ≈ N·M²/2
    # triplets per frame, ~11k×50 for water at 3.5 Å); chunk over P if frames get large
    d2 = sum(((q[:, d, None] - Q3[None, :, d]) / s3[d]) ** 2 for d in range(3))
    if str(getattr(cfg, "gap3b_kernel", "exp")) == "pol":
        # TurboGAP's cov_pp(r, 3, 1) (gap.f90:760-788): compact support,
        # exactly zero beyond a scaled distance of 1 (σ is the support radius).
        r3 = np.sqrt(d2)
        k3 = w[:, None] * np.where(r3 < 1.0, (1.0 - r3) ** 4 * (4.0 * r3 + 1.0), 0.0)
    else:
        k3 = w[:, None] * np.exp(-0.5 * d2)
    slot = pair_index(T)[types_int[ja], types_int[jb]]
    q3 = np.zeros((N, T * (T + 1) // 2, len(Q3)))
    np.add.at(q3, (ci, slot), k3)

    return np.concatenate([q2.reshape(N, -1), q3.reshape(N, -1)], axis=1).astype(np.float32)


def select_centres(frames, cfg):
    """(centres2b [M2], centres3b [M3, 3]): 2b uniform grid on [0.5, rc2]; 3b
    farthest-point sampling in σ-scaled coordinates over ≤ MAX_FPS_SAMPLES
    training triplets. Deterministic under cfg.seed."""
    rng = np.random.default_rng(cfg.seed)
    c2 = np.linspace(0.5, cfg.gap2b_rcut, cfg.gap2b_n_centres)
    per = -(-MAX_FPS_SAMPLES // len(frames))        # ceil: memory bounded by the cap, not the dataset
    qs = []
    for at in frames:
        qi = _triplets(len(at), _pairs(at, cfg.gap3b_rcut), cfg.gap3b_rcut, cfg.gap_buffer)[3]
        qs.append(qi[rng.choice(len(qi), min(per, len(qi)), replace=False)])
    q = np.unique(np.concatenate(qs), axis=0)
    M3 = cfg.gap3b_n_centres
    if len(q) < M3:
        raise ValueError(f"only {len(q)} distinct triplets for {M3} 3b centres")
    x = q / np.asarray(cfg.gap3b_sigma, dtype=np.float64)
    sel = [int(rng.integers(len(x)))]
    d = ((x - x[sel[0]]) ** 2).sum(1)
    for _ in range(M3 - 1):
        n = int(d.argmax())
        sel.append(n)
        d = np.minimum(d, ((x - x[n]) ** 2).sum(1))
    return c2, q[sel]
