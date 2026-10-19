"""Body-order site-model descriptors and angular geometric kernels for the GAP
dipole/polarisability fit (gap_site_models). Numpy only.

Descriptors (invariant, row-normalised; the GAP kernel is (q̂_i·q̂_m)^ξ on
descriptors of the SAME body order only):
    2b  q_n^{t'}(i)        = Σ_{j∈t'} R_n(r_ij)
    3b  q_{nn'l}^{t't''}(i) = Σ_{j∈t'} Σ_{k∈t'', k≠j} R_n(r_ij) R_n'(r_ik) P_l(cos θ_jik)
with R_n the readout's radial family on dipole_rc (data._radial_weights).

Angular geometric kernels for the 3b model (equivariance by construction: the
angle enters as an invariant weight, the tensor character comes from the
uncontracted bond vectors):
    α : W_i^{(ab,l)} = Σ_{j,k≠j} R_a(r_ij) R_b(r_ik) P_l(cos θ) [sym(d_ij d_ikᵀ)]_traceless
    μ : w_i^{(ab,l)} = Σ_{j,k≠j} R_a(r_ij) R_b(r_ik) P_l(cos θ) d_ij           (odd)
"""
from __future__ import annotations

import numpy as np
from ase import Atoms

import data as tnep_data
from TNEPconfig import GAPconfig, resolved_dipole_cutoff


def legendre(cos: np.ndarray, lmax: int) -> np.ndarray:
    """P_0..P_lmax(cos) by recurrence → [..., lmax+1]."""
    P = [np.ones_like(cos), cos]
    for l in range(1, lmax):
        P.append(((2 * l + 1) * cos * P[l] - l * P[l - 1]) / (l + 1))
    return np.stack(P[:lmax + 1], axis=-1)


def _site_pairs(atoms: Atoms, cfg: GAPconfig, C: int):
    """(i, j, d [P,3], R [P,C]) over image pairs within dipole_rc with the
    readout's radial family evaluated for C channels (data._radial_weights)."""
    rc = float(cfg.dipole_rc or cfg.rcut_hard)
    i, j, D = tnep_data._neighbor_pairs(atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    R = tnep_data._radial_weights(np.linalg.norm(d, axis=1), rc, C, str(cfg.dipole_radial),
                                  float(cfg.dipole_zeta), resolved_dipole_cutoff(cfg),
                                  float(cfg.dipole_gauss_sigma_scale))
    return i, j, d, R


def _normalise(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n > 0, n, 1.0)


def two_body_descriptor(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """[N, T·C] radial distribution per neighbour species, unit rows."""
    C, T = int(cfg.gap_site_channels), int(cfg.num_types)
    i, j, _, R = _site_pairs(atoms, cfg, C)
    q = np.zeros((len(atoms), T, C))
    np.add.at(q, (i, np.asarray(ti)[j]), R)
    return _normalise(q.reshape(len(atoms), -1))


def _triplets(i, j):
    """All ordered neighbour pairs (p, k) of the same centre, p ≠ k, as index
    arrays into the pair list (centres are sorted in ASE's output)."""
    order = np.argsort(i, kind="stable"); i, j = i[order], j[order]
    starts = np.flatnonzero(np.r_[True, i[1:] != i[:-1]]); ends = np.r_[starts[1:], len(i)]
    P, K = [], []
    for s, e in zip(starts, ends):
        idx = np.arange(s, e); pp, kk = np.meshgrid(idx, idx, indexing="ij")
        m = pp != kk; P.append(pp[m]); K.append(kk[m])
    return order, np.concatenate(P), np.concatenate(K)


def three_body_descriptor(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """[N, T(T+1)/2·C·C·(L+1)] angular power spectrum, unit rows."""
    C, L, T = int(cfg.gap_site_channels), int(cfg.gap_site_lmax), int(cfg.num_types)
    i, j, d, R = _site_pairs(atoms, cfg, C)
    order, P, K = _triplets(i, j)
    i, j, d, R = i[order], j[order], d[order], R[order]
    r = np.linalg.norm(d, axis=1)
    cos = np.einsum("pd,pd->p", d[P], d[K]) / (r[P] * r[K])
    Pl = legendre(np.clip(cos, -1.0, 1.0), L)                           # [Q, L+1]
    from gap_descriptor import pair_index
    slot = pair_index(T)[np.asarray(ti)[j[P]], np.asarray(ti)[j[K]]]
    contrib = np.einsum("qa,qb,ql->qabl", R[P], R[K], Pl)                # [Q, C, C, L+1]
    q = np.zeros((len(atoms), T * (T + 1) // 2, C, C, L + 1))
    np.add.at(q, (i[P], slot), contrib)
    # The C×C block is symmetric in (n, n') under the ordered p≠k sum: redundant
    # but harmless columns of an INVARIANT descriptor (kept for a simple layout).
    return _normalise(q.reshape(len(atoms), -1))


def angular_cross_tensor(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """α kernel [N, 6, K_x]: Legendre-weighted traceless sym(d_j ⊗ d_k) over
    channel pairs a ≤ b (species-resolved under pair columns) and l ≤ gap_cross_lmax."""
    Cx, L, T = int(cfg.gap_cross_channels), int(cfg.gap_cross_lmax), int(cfg.num_types)
    pair = str(cfg.dipole_species_pairs) != "off"
    i, j, d, R = _site_pairs(atoms, cfg, Cx)
    order, P, K = _triplets(i, j)
    i, j, d, R = i[order], j[order], d[order], R[order]
    r = np.linalg.norm(d, axis=1)
    cos = np.einsum("pd,pd->p", d[P], d[K]) / (r[P] * r[K])
    Pl = legendre(np.clip(cos, -1.0, 1.0), L)
    tj = np.asarray(ti)[j]
    # species-resolved radial weights: k = (a, t') with t' the partner species
    Tp = T if pair else 1
    Rp = np.zeros((len(R), Cx, Tp)); Rp[np.arange(len(R)), :, (tj if pair else 0)] = R
    Rp = Rp.reshape(len(R), -1)                                           # [P, K1]
    K1 = Rp.shape[1]; au, bu = np.triu_indices(K1)
    S = 0.5 * (d[P][:, :, None] * d[K][:, None, :] + d[K][:, :, None] * d[P][:, None, :])   # [Q, 3, 3]
    tr = np.trace(S, axis1=1, axis2=2) / 3.0
    S[:, 0, 0] -= tr; S[:, 1, 1] -= tr; S[:, 2, 2] -= tr
    voigt = np.stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2], S[:, 0, 1], S[:, 1, 2], S[:, 2, 0]], 1)
    w = 0.5 * (Rp[P][:, au] * Rp[K][:, bu] + Rp[P][:, bu] * Rp[K][:, au])    # [Q, K_x/(L+1)] symmetric in a↔b
    contrib = np.einsum("qd,qc,ql->qdcl", voigt, w, Pl).reshape(len(P), 6, -1)
    W = np.zeros((len(atoms), 6, contrib.shape[2]))
    np.add.at(W, i[P], contrib)
    return W


def angular_cross_vector(atoms: Atoms, ti: np.ndarray, cfg: GAPconfig) -> np.ndarray:
    """μ kernel [N, 3, K_x]: Σ R_a(r_ij) R_b(r_ik) P_l(cos θ) d_ij (odd), l ≥ 1
    (l = 0 collapses to the one-hop basis and is skipped)."""
    Cx, L, T = int(cfg.gap_cross_channels), int(cfg.gap_cross_lmax), int(cfg.num_types)
    pair = str(cfg.dipole_species_pairs) != "off"
    i, j, d, R = _site_pairs(atoms, cfg, Cx)
    order, P, K = _triplets(i, j)
    i, j, d, R = i[order], j[order], d[order], R[order]
    r = np.linalg.norm(d, axis=1)
    cos = np.einsum("pd,pd->p", d[P], d[K]) / (r[P] * r[K])
    Pl = legendre(np.clip(cos, -1.0, 1.0), L)[:, 1:]                      # drop l = 0
    tj = np.asarray(ti)[j]; Tp = T if pair else 1
    Rp = np.zeros((len(R), Cx, Tp)); Rp[np.arange(len(R)), :, (tj if pair else 0)] = R
    Rp = Rp.reshape(len(R), -1)
    w = Rp[P][:, :, None] * Rp[K][:, None, :]                              # [Q, K1, K1] (a on j, b on k)
    contrib = np.einsum("qd,qab,ql->qdabl", d[P], w, Pl).reshape(len(P), 3, -1)
    out = np.zeros((len(atoms), 3, contrib.shape[2]))
    np.add.at(out, i[P], contrib)
    return out
