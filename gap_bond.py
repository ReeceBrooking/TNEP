"""Experimental dipole GAP variants — convex-floor tests, not the production fit.

Both put a site function on the BOND rather than on the atom, breaking the
cocycle restriction of the one-hop difference form (neighbor_dipole_formulation.md
§Properties 2(c)): any per-atom scalar telescopes to (s_i − s_j), so the pair
scalar of the production readout cannot depend on the bond as such.

1. Bond kernel (convex). s_ij^n = Σ_b Γ_b^n k(D_i, D_m_b)·k(D_j, D_m'_b) over
   FPS-selected training bonds b = (m, m′) of each species pair (t ≤ t′), and
   μ = Σ_{ordered i≠j, t_i=t, t_j=t′} R_n(r_ij)·d_ij·s_ij^n. Summing both
   orientations of a same-species bond antisymmetrises the kernel
   automatically (d_ji = −d_ij), so s is a proper bond site function. This is
   the convex superset of the bilinear head (Γ of full rank on sparse bonds).
   Prior: gap_fit's αᵀK_bbα with K_bb the product kernel on the sparse bonds.

2. Bilinear ALS (rank 1, non-convex). s_ij^n = a_i^n b_j^n − a_j^n b_i^n with
   a, b kernel site functions. μ = Σ_{i≠j} R_n d_ij a_i b_j = Σ_i a_i·u_i(b),
   u_i^n = Σ_j R_n(r_ij) d_ij b_j^n, so with b fixed the fit for a is the
   production fit on a neighbour-weighted kernel, and symmetrically for b.
   b ≡ 1 at start reproduces the production model; alternate a ← fit(b),
   b ← fit(a), monitor validation.

Everything else (descriptors, FPS, base design, prior, whitening) is reused
from gap_dipole so the base block is bit-identical to the production fit.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
from ase import Atoms

import data as tnep_data
import gap_dipole as gd
from DescriptorBuilder import make_descriptor_builder
from TNEPconfig import GAPconfig, resolved_dipole_cutoff


# ── shared pieces ──────────────────────────────────────────────────────────

def _pairs(atoms: Atoms, cfg: GAPconfig):
    """(i, j, G [P, 3, C]) ordered pairs within r_c, G = R_n(|d|)·d, d = r_i − r_j."""
    return tnep_data._neighbor_dipole_pairs(
        atoms, float(cfg.dipole_rc or cfg.rcut_hard), int(cfg.dipole_channels),
        str(cfg.dipole_radial), float(cfg.dipole_zeta), resolved_dipole_cutoff(cfg),
        float(cfg.dipole_gauss_sigma_scale), 1)


class Setup:
    """Training frames, descriptors and the per-species FPS sparse set — the
    production fit's first half, kept so every variant shares one base."""

    def __init__(self, cfg: GAPconfig):
        self.cfg = cfg
        with contextlib.redirect_stdout(io.StringIO()):
            self.frames, tis = tnep_data.collect(cfg)
        self.tis = [np.asarray(t) for t in tis]
        self.builder = make_descriptor_builder(cfg)
        self.descs = gd._descriptors(self.builder, self.frames)
        rng = np.random.default_rng(cfg.seed)
        self.sparse, self.sparse_idx = [], []      # idx: into the species-t atom list (frame-major)
        for t in range(cfg.num_types):
            Dt = np.concatenate([D[ti == t] for D, ti in zip(self.descs, self.tis)])
            idx = gd._fps(Dt, int(cfg.gap_sparse_per_species), rng)
            self.sparse.append(Dt[idx].copy()); self.sparse_idx.append(idx)
        self.xis = gd._xi_tuple(cfg)
        self.n_vcols = gd._n_vcols(cfg)
        self.base_factors = gd._prior_factors(cfg, [(self.sparse, gd._deltas(cfg)[0], self.n_vcols)],
                                              self.xis, self.n_vcols)
        self.n_base = sum(n for n, _ in self.base_factors)

    def prep(self, frames: list[Atoms]):
        tis = [np.asarray(t) for t in tnep_data.assign_type_indices(frames, self.cfg.types)]
        return tis, gd._descriptors(self.builder, frames)

    def phi_base(self, atoms, ti, D, v=None) -> np.ndarray:
        """[3, n_base] production design block (optionally with a custom geometry v)."""
        if v is None:
            v = gd._v_atoms(atoms, ti, self.cfg)
        deltas = gd._deltas(self.cfg)
        return gd._phi_frame([(D, self.sparse, v, deltas[0])], ti, self.xis, None, deltas, v=v)

    def row_w(self, frames) -> np.ndarray:
        sigma = float(self.cfg.gap_sigma)
        D = 6 if int(self.cfg.target_mode) == 2 else 3
        return np.repeat(1.0 / (sigma * np.sqrt([len(a) for a in frames])), D)

    def targets(self, frames) -> np.ndarray:
        return gd._targets(frames, self.cfg)


def _solve(blocks_w: list[np.ndarray], y: np.ndarray, row_w: np.ndarray) -> np.ndarray:
    """Dual ridge in whitened coordinates: β = Φ_wᵀ (Φ_w Φ_wᵀ + I)⁻¹ y_w."""
    Phi_w = np.concatenate(blocks_w, axis=1) * row_w[:, None]
    G = Phi_w @ Phi_w.T
    G[np.diag_indices_from(G)] += 1.0
    return Phi_w.T @ np.linalg.solve(G, y * row_w)


def _site_values(D: np.ndarray, ti: np.ndarray, sparse: list, alpha_base: np.ndarray,
                 xis: tuple, n_vcols: int) -> np.ndarray:
    """Kernel site function s_i [N, K] of a fitted base block (ξ list summed)."""
    out = np.zeros((len(D), n_vcols))
    off = 0
    for t, Dm in enumerate(sparse):
        sel = ti == t
        G = D[sel] @ Dm.T
        for xi in xis:
            a = alpha_base[off:off + len(Dm) * n_vcols].reshape(len(Dm), n_vcols)
            out[sel] += (G ** xi) @ a
            off += len(Dm) * n_vcols
    return out


# ── 1. antisymmetric bond kernel ───────────────────────────────────────────

def bond_sparse(setup: Setup, M_b: int, n_frames: int = 60) -> dict:
    """FPS-selected sparse bonds per species pair (t ≤ t′): {(t,t′): (Dm, Dm′)}
    on the concatenated [D_i ‖ D_j] of real bonds within r_c."""
    cfg = setup.cfg
    rng = np.random.default_rng(int(cfg.seed) + 1)
    out = {}
    for t in range(cfg.num_types):
        for t2 in range(t, cfg.num_types):
            feats = []
            for atoms, ti, D in list(zip(setup.frames, setup.tis, setup.descs))[:n_frames]:
                i, j, _ = _pairs(atoms, cfg)
                sel = (ti[i] == t) & (ti[j] == t2)
                feats.append(np.concatenate([D[i[sel]], D[j[sel]]], axis=1))
            F = np.concatenate(feats)
            idx = gd._fps(F, M_b, rng)                     # ordering-safe: rows share norm √2
            Q = D.shape[1]
            out[(t, t2)] = (F[idx, :Q].copy(), F[idx, Q:].copy())
    return out


def _phi_bond(atoms, ti, D, bonds: dict, cfg: GAPconfig, xi: int) -> np.ndarray:
    """[3, Σ_types C·M_b] bond-kernel design block, columns (type, n, b)."""
    i, j, G = _pairs(atoms, cfg)                           # G [P, 3, C]
    cols = []
    for (t, t2), (Dm, Dm2) in bonds.items():
        sel = (ti[i] == t) & (ti[j] == t2)
        kij = ((D[i[sel]] @ Dm.T) ** xi) * ((D[j[sel]] @ Dm2.T) ** xi)      # [P_s, M_b]
        cols.append(np.einsum("pdn,pb->dnb", G[sel], kij).reshape(3, -1))   # n-major
    return np.concatenate(cols, axis=1)


def _bond_factors(bonds: dict, xi: int, delta_b: float, jitter: float) -> list:
    """Per type: Cholesky factor of δ_b²·K_bb, K_bb the (antisymmetrised for
    t = t′) product kernel on the sparse bonds."""
    out = []
    for (t, t2), (Dm, Dm2) in bonds.items():
        K = (Dm @ Dm.T) ** xi * (Dm2 @ Dm2.T) ** xi
        if t == t2:
            K = K - (Dm @ Dm2.T) ** xi * (Dm2 @ Dm.T) ** xi
        K = delta_b ** 2 * K
        jit = jitter
        while True:
            try:
                out.append(np.linalg.cholesky(K + jit * np.eye(len(K)))); break
            except np.linalg.LinAlgError:
                jit *= 10.0
    return out


def _whiten_bond(X: np.ndarray, factors: list, C: int, transpose: bool) -> np.ndarray:
    """Bond block [R, Σ C·M_b] laid out (type, n, b): apply L⁻¹ along b."""
    out, off = np.empty_like(X), 0
    for L in factors:
        M = len(L)
        blk = X[:, off:off + C * M].reshape(len(X), C, M)
        if transpose:                                   # α = L⁻ᵀ β (single row)
            res = np.linalg.solve(L.T, blk[0].T).T[None]
        else:                                           # Φ′ = Φ·L⁻ᵀ along b
            res = np.linalg.solve(L, blk.reshape(-1, M).T).T.reshape(len(X), C, M)
        out[:, off:off + C * M] = res.reshape(len(res), C * M)
        off += C * M
    return out


def fit_bond_kernel(setup: Setup, M_b: int, delta_b: float = 1.0):
    """Production base block + antisymmetric bond-kernel block, one dual ridge.
    Returns a predictor frames → μ [F, 3]."""
    cfg, xi = setup.cfg, setup.xis[0]
    assert len(setup.xis) == 1, "bond kernel: single ξ only"
    bonds = bond_sparse(setup, M_b)
    Lb = _bond_factors(bonds, xi, delta_b, float(cfg.gap_sparse_jitter))
    C = int(cfg.dipole_channels)
    Pb = np.stack([setup.phi_base(a, ti, D) for a, ti, D in zip(setup.frames, setup.tis, setup.descs)])
    Pk = np.stack([_phi_bond(a, ti, D, bonds, cfg, xi) for a, ti, D in zip(setup.frames, setup.tis, setup.descs)])
    F = len(setup.frames)
    Pb, Pk = Pb.reshape(3 * F, -1), Pk.reshape(3 * F, -1)
    beta = _solve([gd._whiten(Pb, setup.base_factors, False),
                   _whiten_bond(Pk, Lb, C, False)],
                  setup.targets(setup.frames).reshape(-1), setup.row_w(setup.frames))
    nb = Pb.shape[1]
    alpha_base = gd._whiten(beta[None, :nb], setup.base_factors, True)[0]
    alpha_bond = _whiten_bond(beta[None, nb:], Lb, C, True)[0]

    def predict(frames):
        tis, descs = setup.prep(frames)
        return np.stack([setup.phi_base(a, ti, D) @ alpha_base + _phi_bond(a, ti, D, bonds, cfg, xi) @ alpha_bond
                         for a, ti, D in zip(frames, tis, descs)])
    predict.n_columns = Pb.shape[1] + Pk.shape[1]
    return predict


# ── 2. bilinear ALS ────────────────────────────────────────────────────────

def _weighted_v(atoms, ti, cfg: GAPconfig, w: np.ndarray, into: str) -> np.ndarray:
    """Neighbour-weighted dipole kernel [N, 3, K] (pair columns, k = (n, t′)):
    into="i": u_i^{n,t′} = Σ_{j∈t′} R_n d_ij · w_j^{n, t_i}   (weights on the partner)
    into="j": w_j^{n,t}  = Σ_{i∈t}  R_n d_ij · w_i^{n, t_j}   (the mirrored half-step)
    w [N, C, T] indexed [atom, channel, species of the OTHER atom]."""
    i, j, G = _pairs(atoms, cfg)
    T, C = int(cfg.num_types), int(cfg.dipole_channels)
    V = np.zeros((len(atoms), T, 3, C))
    if into == "i":
        np.add.at(V, (i, ti[j]), G * w[j, :, ti[i]][:, None, :])
    else:
        np.add.at(V, (j, ti[i]), G * w[i, :, ti[j]][:, None, :])
    return V.transpose(0, 2, 3, 1).reshape(len(atoms), 3, C * T)


def fit_bilinear_als(setup: Setup, n_iter: int = 3, score=None):
    """Rank-1 bilinear bond scalar by alternating base fits. `score(predict)`
    (optional) is called after every half-step for monitoring. Returns the
    predictor of the last iteration (or the best by `score` if it returns a
    number)."""
    cfg = setup.cfg
    T, C = int(cfg.num_types), int(cfg.dipole_channels)
    y, rw = setup.targets(setup.frames).reshape(-1), setup.row_w(setup.frames)
    ones = [np.ones((len(a), C, T)) for a in setup.frames]

    def half(weights, into):
        """Fit the site function whose partner weights are `weights`."""
        Phi = np.stack([setup.phi_base(a, ti, D, _weighted_v(a, ti, cfg, w, into))
                        for a, ti, D, w in zip(setup.frames, setup.tis, setup.descs, weights)])
        Phi = Phi.reshape(-1, Phi.shape[-1])
        beta = _solve([gd._whiten(Phi, setup.base_factors, False)], y, rw)
        return gd._whiten(beta[None], setup.base_factors, True)[0]

    def values(alpha, tis, descs):
        return [_site_values(D, ti, setup.sparse, alpha, setup.xis, setup.n_vcols).reshape(len(D), C, T)
                for ti, D in zip(tis, descs)]

    def make_predict(alpha_a, alpha_b):
        def predict(frames):
            tis, descs = setup.prep(frames)
            A = values(alpha_a, tis, descs)
            B = [np.ones((len(f), C, T)) for f in frames] if alpha_b is None else values(alpha_b, tis, descs)
            out = []
            for f, ti, a, b in zip(frames, tis, A, B):
                u = _weighted_v(f, ti, cfg, b, "i")                   # [N, 3, K]
                out.append(np.einsum("nk,ndk->d", a.reshape(len(f), -1), u))
            return np.stack(out)
        return predict

    alpha_a = half(ones, "i")                        # == production fit
    alpha_b, best = None, None
    pred = make_predict(alpha_a, None)
    if score is not None:
        best = (score(pred, "iter0 a|b=1"), pred)
    for it in range(1, n_iter + 1):
        A = values(alpha_a, setup.tis, setup.descs)
        alpha_b = half(A, "j")
        B = values(alpha_b, setup.tis, setup.descs)
        alpha_a = half(B, "i")
        pred = make_predict(alpha_a, alpha_b)
        if score is not None:
            s = score(pred, f"iter{it}")
            if s is not None and (best is None or s < best[0]):
                best = (s, pred)
    return best[1] if best is not None else pred
