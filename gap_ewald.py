"""Experimental long-range (Ewald) channels for the dipole GAP — convex-floor
tests on top of gap_bond.Setup (not the production fit).

Reference charges q by species (SPC/E: O −0.8476, H +0.4238) give, per
frame, the Ewald potential φ_i and field E_i at every atom. Two ways in, both
linear in the coefficients:

1. field column — E_i (optionally minus the bare Coulomb field of the
   neighbours inside r_c, leaving the beyond-cutoff part only) is an odd,
   covariant, translation-invariant vector: one extra geometry column per
   sparse point, μ ⊃ Σ_i s_i^E E_i — the induced dipole α_i E_i with a learned
   local α. Scaled to the RMS of the bond-kernel columns × delta_E.
2. potential kernel — the site kernel becomes k(D_i,D_m)·exp(−(φ_i−φ_m)²/2ℓ²):
   the site function sees the long-range electrostatic state at zero column
   cost; the prior K_MM carries the same product.
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.special import erfc

import gap_bond as gb
import gap_dipole as gd

SPCE = {8: -0.8476, 1: 0.4238}


def ewald_phi_E(atoms: Atoms, q: np.ndarray, alpha: float = 0.35, r_cut: float = 9.0,
                k_cut: float = 7.0) -> tuple[np.ndarray, np.ndarray]:
    """Ewald potential φ_i [N] and field E_i [N, 3] from all OTHER charges
    (Coulomb units q/Å; self term removed, k = 0 dropped: neutral cells). Real space over the image-exact
    ASE neighbour list, reciprocal over |k| ≤ k_cut."""
    N, r = len(atoms), atoms.positions
    i, j, d, D = neighbor_list("ijdD", atoms, r_cut)          # D = r_j − r_i
    f = erfc(alpha * d) / d
    phi = np.zeros(N); np.add.at(phi, i, q[j] * f)
    g = (erfc(alpha * d) / d ** 2 + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * d) ** 2) / d) / d
    E = np.zeros((N, 3)); np.add.at(E, i, -(q[j] * g)[:, None] * D)   # field at i from j: −q_j g D_ij
    cell = np.asarray(atoms.cell.array); V = abs(np.linalg.det(cell))
    recip = 2 * np.pi * np.linalg.inv(cell).T
    nmax = int(np.ceil(k_cut / np.min(np.linalg.norm(recip, axis=1)))) + 1
    rng_ = np.arange(-nmax, nmax + 1)
    n = np.array(np.meshgrid(rng_, rng_, rng_, indexing="ij")).reshape(3, -1).T
    k = n @ recip
    k2 = (k ** 2).sum(1)
    keep = (k2 > 1e-12) & (k2 <= k_cut ** 2)
    k, k2 = k[keep], k2[keep]
    phase = r @ k.T                                            # [N, K]
    S = (q[:, None] * np.exp(1j * phase)).sum(0)               # [K]
    w = 4 * np.pi / V * np.exp(-k2 / (4 * alpha ** 2)) / k2
    e = np.exp(-1j * phase)
    phi += (w * (S * e)).real.sum(1) - 2 * alpha / np.sqrt(np.pi) * q   # minus the Gaussian self term
    E -= ((w * (S * e)).imag[:, :, None] * k[None]).sum(1)      # E = −∂φ/∂r_i
    return phi, E


def ewald_features(frames, tis, cfg, long_range_only: bool):
    """Per frame (φ [N], E [N, 3]) from SPC/E charges; long_range_only
    subtracts the bare Coulomb contribution of the neighbours inside r_c."""
    rc = float(cfg.dipole_rc or cfg.rcut_hard)
    out = []
    for a in frames:
        q = np.array([SPCE[int(z)] for z in a.numbers])
        phi, E = ewald_phi_E(a, q)
        if long_range_only:
            i, j, d, D = neighbor_list("ijdD", a, rc)
            np.add.at(phi, i, -q[j] / d)
            np.add.at(E, i, (q[j] / d ** 3)[:, None] * D)
        out.append((phi, E))
    return out


# ── 1. field column ─────────────────────────────────────────────────────────

def fit_field_column(setup: gb.Setup, long_range_only: bool, delta_E: float = 1.0):
    cfg = setup.cfg
    feats = ewald_features(setup.frames, setup.tis, cfg, long_range_only)
    vs = [gd._v_atoms(a, ti, cfg) for a, ti in zip(setup.frames, setup.tis)]
    scale = delta_E * np.sqrt(np.mean([(v ** 2).mean() for v in vs]) / np.mean([(E ** 2).mean() for _, E in feats]))
    K1 = setup.n_vcols + 1
    factors = gd._prior_factors(cfg, [setup.sparse], setup.xis, K1)

    def design(frames, tis, descs, feats, vs):
        rows = []
        for a, ti, D, (_, E), v in zip(frames, tis, descs, feats, vs):
            v_aug = np.concatenate([v, scale * E[:, :, None]], axis=2)
            rows.append(gd._phi_frame([(D, setup.sparse)], v_aug, ti, setup.xis, None, gd._deltas(cfg)))
        return np.stack(rows).reshape(3 * len(frames), -1)

    Phi = design(setup.frames, setup.tis, setup.descs, feats, vs)
    beta = gb._solve([gd._whiten(Phi, factors, K1, False)], setup.targets(setup.frames).reshape(-1), setup.row_w(setup.frames))
    alpha = gd._whiten(beta[None], factors, K1, True)[0]

    def predict(frames):
        tis, descs = setup.prep(frames)
        f2 = ewald_features(frames, tis, cfg, long_range_only)
        v2 = [gd._v_atoms(a, ti, cfg) for a, ti in zip(frames, tis)]
        return (design(frames, tis, descs, f2, v2) @ alpha).reshape(len(frames), 3)
    return predict


# ── 2. potential product kernel ─────────────────────────────────────────────

def fit_potential_kernel(setup: gb.Setup, long_range_only: bool, ell_factor: float = 1.0):
    cfg, xis = setup.cfg, setup.xis
    assert len(xis) == 1
    xi = xis[0]
    feats = ewald_features(setup.frames, setup.tis, cfg, long_range_only)
    phi_all = [p for p, _ in feats]
    # φ of the sparse points, per species, through the recorded FPS indices
    phi_m, ell = [], []
    for t in range(cfg.num_types):
        pt = np.concatenate([p[ti == t] for p, ti in zip(phi_all, setup.tis)])
        phi_m.append(pt[setup.sparse_idx[t]]); ell.append(ell_factor * pt.std())
    d_soap = gd._deltas(cfg)[0]
    K = setup.n_vcols

    def kphi(p, pm, t):
        return np.exp(-(p[:, None] - pm[None, :]) ** 2 / (2 * ell[t] ** 2))

    def design(frames, tis, descs, feats, vs):
        rows = []
        for a, ti, D, (p, _), v in zip(frames, tis, descs, feats, vs):
            cols = []
            for t, Dm in enumerate(setup.sparse):
                sel = ti == t
                G = (D[sel] @ Dm.T) ** xi * kphi(p[sel], phi_m[t], t)
                cols.append(d_soap ** 2 * np.einsum("nm,nxk->xmk", G, v[sel]).reshape(3, -1))
            rows.append(np.concatenate(cols, 1))
        return np.stack(rows).reshape(3 * len(frames), -1)

    factors = []
    for t, Dm in enumerate(setup.sparse):
        Kmm = d_soap ** 2 * (Dm @ Dm.T) ** xi * kphi(phi_m[t], phi_m[t], t)
        jit = float(cfg.gap_sparse_jitter)
        while True:
            try:
                factors.append((len(Dm) * K, np.linalg.cholesky(Kmm + jit * np.eye(len(Dm))))); break
            except np.linalg.LinAlgError:
                jit *= 10
    vs = [gd._v_atoms(a, ti, cfg) for a, ti in zip(setup.frames, setup.tis)]
    Phi = design(setup.frames, setup.tis, setup.descs, feats, vs)
    beta = gb._solve([gd._whiten(Phi, factors, K, False)], setup.targets(setup.frames).reshape(-1), setup.row_w(setup.frames))
    alpha = gd._whiten(beta[None], factors, K, True)[0]

    def predict(frames):
        tis, descs = setup.prep(frames)
        f2 = ewald_features(frames, tis, cfg, long_range_only)
        v2 = [gd._v_atoms(a, ti, cfg) for a, ti in zip(frames, tis)]
        return (design(frames, tis, descs, f2, v2) @ alpha).reshape(len(frames), 3)
    return predict


# ── generic extra-vector-column fit ─────────────────────────────────────────

def coulomb_near(frames, cfg, cutoff: float, species: int | None = None):
    """Bare Coulomb field within `cutoff` from SPC/E charges (optionally from
    one species only) — the near part of the Ewald field. Returns [N, 3] per frame."""
    out = []
    for a in frames:
        q = np.array([SPCE[int(z)] for z in a.numbers])
        i, j, d, D = neighbor_list("ijdD", a, cutoff)
        if species is not None:
            keep = a.numbers[j] == species
            i, j, d, D = i[keep], j[keep], d[keep], D[keep]
        E = np.zeros((len(a), 3)); np.add.at(E, i, -(q[j] / d ** 3)[:, None] * D)
        out.append(E)
    return out


def ewald_species_split(frames, cfg):
    """Ewald field from the O charges and from the H charges separately
    (each non-neutral; the dropped k = 0 term is the neutralising background)."""
    out = []
    for a in frames:
        cols = []
        for z in (8, 1):
            q = np.array([SPCE[int(zz)] if zz == z else 0.0 for zz in a.numbers])
            cols.append(ewald_phi_E(a, q)[1])
        out.append(np.stack(cols, axis=2))                   # [N, 3, 2]
    return out


def bond_tensor_dot(frames, tis, cfg, E_list, C_x: int):
    """Anisotropic response columns (Σ_j R_a d dᵀ)·E_i, a = 1..C_x (species-
    resolved under pair columns) → [N, 3, C_x·T′] per frame."""
    import data as tnep_data
    from TNEPconfig import resolved_dipole_cutoff
    out = []
    for a, ti, E in zip(frames, tis, E_list):
        V2 = tnep_data._neighbor_dipole_V(
            a, ti, float(cfg.dipole_rc or cfg.rcut_hard), C_x, int(cfg.num_types),
            str(cfg.dipole_radial), float(cfg.dipole_zeta), resolved_dipole_cutoff(cfg),
            float(cfg.dipole_gauss_sigma_scale), rank=2)
        Tv = gd._collapse_V(V2[:, :, C_x:] + V2[:, :, :C_x] / 3.0, gd._pair_columns(cfg))   # [N, 6, K1]
        xx, yy, zz, xy, yz, zx = (Tv[:, k] for k in range(6))
        ex, ey, ez = E[:, 0, None], E[:, 1, None], E[:, 2, None]
        out.append(np.stack([xx * ex + xy * ey + zx * ez, xy * ex + yy * ey + yz * ez,
                             zx * ex + yz * ey + zz * ez], axis=1))                    # [N, 3, K1]
    return out


def fit_extra_columns(setup: gb.Setup, feature_fn, delta: float = 1.0):
    """Production design + extra geometry columns from feature_fn(frames, tis)
    → list of [N, 3, n_extra]; each extra column scaled to the RMS of the
    bond-kernel columns × delta."""
    cfg = setup.cfg
    vs = [gd._v_atoms(a, ti, cfg) for a, ti in zip(setup.frames, setup.tis)]
    X = feature_fn(setup.frames, setup.tis)
    n_extra = X[0].shape[2]
    rms_v = np.sqrt(np.mean([(v ** 2).mean() for v in vs]))
    scale = delta * rms_v / np.sqrt(np.mean([(x ** 2).mean(axis=(0, 1)) for x in X], axis=0))   # [n_extra]
    K1 = setup.n_vcols + n_extra
    factors = gd._prior_factors(cfg, [setup.sparse], setup.xis, K1)

    def design(frames, tis, descs, X, vs):
        rows = [gd._phi_frame([(D, setup.sparse)], np.concatenate([v, x * scale], axis=2), ti,
                              setup.xis, None, gd._deltas(cfg))
                for ti, D, x, v in zip(tis, descs, X, vs)]
        return np.stack(rows).reshape(3 * len(frames), -1)

    Phi = design(setup.frames, setup.tis, setup.descs, X, vs)
    beta = gb._solve([gd._whiten(Phi, factors, K1, False)], setup.targets(setup.frames).reshape(-1), setup.row_w(setup.frames))
    alpha = gd._whiten(beta[None], factors, K1, True)[0]

    def predict(frames):
        tis, descs = setup.prep(frames)
        v2 = [gd._v_atoms(a, ti, cfg) for a, ti in zip(frames, tis)]
        return (design(frames, tis, descs, feature_fn(frames, tis), v2) @ alpha).reshape(len(frames), 3)
    predict.n_columns = Phi.shape[1]
    return predict


# ── 3. self-consistent induced-dipole field ("polarise") ────────────────────

BOHR = 0.52917721


def dipole_field(atoms: Atoms, p_eA: np.ndarray, delta: float = 0.02) -> np.ndarray:
    """Ewald field at every atom from point dipoles p (e·Å) on the atoms,
    excluding each atom's own dipole: each dipole is two charges ±|p|/δ at
    ±δ/2 along p̂ (exact to O(δ²)); the own pair's bare field −8|p|/δ³ p̂ is
    removed (the smooth Gaussian self part, ~4α³/3√π·p, is neglected)."""
    N = len(atoms)
    norm = np.linalg.norm(p_eA, axis=1)
    live = norm > 1e-12                                  # zero dipoles contribute nothing
    phat = p_eA[live] / norm[live][:, None]
    r = atoms.positions
    aug = Atoms(positions=np.concatenate([r, r[live] + 0.5 * delta * phat, r[live] - 0.5 * delta * phat]),
                cell=atoms.cell, pbc=True)
    q = np.concatenate([np.zeros(N), norm[live] / delta, -norm[live] / delta])
    _, E = ewald_phi_E(aug, q)
    E = E[:N]
    E[live] += 8.0 * (norm[live] / delta ** 3)[:, None] * phat
    return E


def fit_polarise(setup: gb.Setup, split: bool = True, n_iter: int = 1, mix: float = 1.0):
    """Stage 0: field-column fit on the SPC/E Ewald field E⁰. Then, n_iter
    times: induced dipoles p_i = Σ_e s_i^e·scale_e·E_i^e (the model's own
    field-column part), E ← E⁰ + mix·E_dip(p), refit. Prediction replays the
    same chain on new frames. Returns (predict, history of stage predictors)."""
    cfg = setup.cfg
    xi_list, K0 = setup.xis, setup.n_vcols
    base_feats = (lambda frames: ewald_species_split(frames, cfg)) if split else \
                 (lambda frames: [E[:, :, None] for _, E in ewald_features(frames, None, cfg, False)])
    vs_tr = [gd._v_atoms(a, ti, cfg) for a, ti in zip(setup.frames, setup.tis)]
    rms_v = np.sqrt(np.mean([(v ** 2).mean() for v in vs_tr]))
    y, rw = setup.targets(setup.frames).reshape(-1), setup.row_w(setup.frames)

    def fit_stage(X):                                   # X: list [N, 3, n_extra]
        n_extra = X[0].shape[2]
        scale = rms_v / np.sqrt(np.mean([(x ** 2).mean(axis=(0, 1)) for x in X], axis=0))
        K1 = K0 + n_extra
        factors = gd._prior_factors(cfg, [setup.sparse], xi_list, K1)
        Phi = np.stack([gd._phi_frame([(D, setup.sparse)], np.concatenate([v, x * scale], 2), ti, xi_list, None, gd._deltas(cfg))
                        for ti, D, x, v in zip(setup.tis, setup.descs, X, vs_tr)]).reshape(3 * len(X), -1)
        beta = gb._solve([gd._whiten(Phi, factors, K1, False)], y, rw)
        alpha = gd._whiten(beta[None], factors, K1, True)[0]
        return dict(alpha=alpha, scale=scale, K1=K1)

    def induced(stage, frames, tis, descs, X):
        """p_i [N, 3] in e·Å from the stage's field-column site functions."""
        out = []
        for ti, D, x in zip(tis, descs, X):
            s = gb._site_values(D, ti, setup.sparse, stage["alpha"], xi_list, stage["K1"])[:, K0:]   # [N, n_extra]
            out.append(np.einsum("ne,nde->nd", s * stage["scale"], x) * BOHR)
        return out

    def predict_stage(stage, frames, tis, descs, X):
        vs = [gd._v_atoms(a, ti, cfg) for a, ti in zip(frames, tis)]
        Phi = np.stack([gd._phi_frame([(D, setup.sparse)], np.concatenate([v, x * stage["scale"]], 2), ti, xi_list, None, gd._deltas(cfg))
                        for ti, D, x, v in zip(tis, descs, X, vs)])
        return Phi @ stage["alpha"]

    def chain(frames, tis, descs, stages):
        X0 = base_feats(frames); X = X0
        for k, stage in enumerate(stages[:-1]):
            p = induced(stage, frames, tis, descs, X)
            Ed = [dipole_field(a, pi) for a, pi in zip(frames, p)]
            X = [x0 + mix * ed[:, :, None] for x0, ed in zip(X0, Ed)]
        return predict_stage(stages[-1], frames, tis, descs, X), X

    stages = [fit_stage(base_feats(setup.frames))]
    X = base_feats(setup.frames)
    for it in range(n_iter):
        p = induced(stages[-1], setup.frames, setup.tis, setup.descs, X)
        Ed = [dipole_field(a, pi) for a, pi in zip(setup.frames, p)]
        X0 = base_feats(setup.frames)
        X = [x0 + mix * ed[:, :, None] for x0, ed in zip(X0, Ed)]
        stages.append(fit_stage(X))
        print(f"  polarise iter {it + 1}: rms|p| {np.sqrt(np.mean([(pi**2).sum(1).mean() for pi in p])):.3f} e·Å, "
              f"rms|E_dip|/rms|E0| {np.sqrt(np.mean([(e**2).mean() for e in Ed]) / np.mean([(x**2).mean() for x in X0])):.2f}", flush=True)

    def make_predict(k):
        def predict(frames):
            tis, descs = setup.prep(frames)
            return chain(frames, tis, descs, stages[:k + 1])[0]
        return predict
    return [make_predict(k) for k in range(len(stages))]


# ── 4. charge equilibration (QEq) columns — port of ExperimentalDipoleEwald ──
# "chi": Ṽ = −P·L⁺·v_fc, so Σ_i χ_i Ṽ_i ≡ Σ_i q_i r_i with q = −P·χ (bond
# form, PBC-safe); one extra geometry column, χ a kernel site function.
# "polarise": the local charges relax, δq = −P·J·q₀; in bond form every bond
# is built on d_ij − (y_i − y_j), y = −J·Ṽ, no extra column.

from ase.data import covalent_radii
from ase.geometry import get_distances
from scipy.special import erf
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_COULOMB_EV_A = 14.399645


def _fc_np(cutoff, r, rc):
    if cutoff == "cosine2":
        return ((1.0 + np.cos(np.pi * r / rc)) / 2.0) ** 2
    if cutoff == "cosine":
        return (1.0 + np.cos(np.pi * r / rc)) / 2.0
    if cutoff == "cubic":
        x = r / rc; return 1.0 - 3.0 * x * x + 2.0 * x ** 3
    if cutoff == "tanh3":
        return np.tanh(1.0 - r / rc) ** 3 / np.tanh(1.0) ** 3
    raise ValueError(cutoff)


def ewald_point_matrix(positions, cell):
    """Point-charge Ewald interaction matrix J [N, N] (1/Å), tin-foil, α = 7/h_min."""
    cell = np.asarray(cell, float); r = np.asarray(positions, float); N = len(r)
    V = abs(np.linalg.det(cell)); heights = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    alpha = 7.0 / heights.min()
    _, R = get_distances(r, cell=cell, pbc=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(R > 0, erfc(alpha * R) / R, 0.0)
    B = 2.0 * np.pi * np.linalg.inv(cell).T; kmax = 8.6 * alpha
    nmax = np.ceil(kmax * np.linalg.norm(cell, axis=1) / (2.0 * np.pi)).astype(int)
    grids = np.meshgrid(*[np.arange(-m, m + 1) for m in nmax], indexing="ij")
    n = np.stack([g.ravel() for g in grids], 1); n = n[np.any(n != 0, axis=1)]
    K = n @ B; k2 = (K * K).sum(1); sel = k2 <= kmax * kmax; K, k2 = K[sel], k2[sel]
    w = np.exp(-k2 / (4 * alpha * alpha)) / k2
    phase = r @ K.T; c, s = np.cos(phase), np.sin(phase)
    J += (4 * np.pi / V) * ((c * w) @ c.T + (s * w) @ s.T)
    J[np.diag_indices(N)] -= 2 * alpha / np.sqrt(np.pi)
    return J


def qeq_coulomb(atoms, sigma):
    """Gaussian-charge Coulomb matrix (eV/e²), periodic: Ewald + short-range correction."""
    r = atoms.positions.astype(float); N = len(r)
    gam = np.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)
    J = ewald_point_matrix(r, atoms.cell.array)
    i, j, d = neighbor_list("ijd", atoms, 3.5 * np.sqrt(2.0) * gam.max())
    np.add.at(J, (i, j), -erfc(d / (np.sqrt(2.0) * gam[i, j])) / d)
    J[np.diag_indices(N)] += 1.0 / (sigma * np.sqrt(np.pi))
    return _COULOMB_EV_A * J


def qeq_parts(atoms, ti, cfg, hardness=10.0, sigma=None):
    """(J, Ṽ [N, 3], y [N, 3]) — Ṽ = −P·L⁺·v_fc (chi column), y = −J·Ṽ (polarise)."""
    from TNEPconfig import resolved_dipole_cutoff
    rc = float(cfg.dipole_rc or cfg.rcut_hard); N = len(atoms)
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i != j; i, j, d = i[keep], j[keep], -D[keep].astype(float)
    fc = _fc_np(resolved_dipole_cutoff(cfg), np.linalg.norm(d, axis=1), rc)
    assert connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(N, N)), directed=False)[0] == 1
    L = np.zeros((N, N)); np.add.at(L, (i, j), -fc); L[np.diag_indices(N)] = -L.sum(1)
    v = np.zeros((N, 3)); np.add.at(v, i, fc[:, None] * d)
    sig = (np.asarray(sigma, float) if sigma is not None else covalent_radii[np.asarray(cfg.types)])[np.asarray(ti)]
    J = qeq_coulomb(atoms, sig)
    A = J + float(hardness) * np.eye(N)
    ones = np.ones((N, 1))
    kkt = lambda M: np.block([[M, ones], [ones.T, np.zeros((1, 1))]])
    x = np.linalg.solve(kkt(L), np.vstack([v, np.zeros((1, 3))]))[:N]
    Vq = -np.linalg.solve(kkt(A), np.vstack([x, np.zeros((1, 3))]))[:N]
    return J, Vq, -(J @ Vq)


def v_polarised(atoms, ti, cfg, y):
    """Production geometry v_i [N, 3, K] with every bond built on d − (y_i − y_j)
    (radial weights on the true length): V' = V − Σ_j R_n (y_i − y_j)."""
    import data as tnep_data
    from TNEPconfig import resolved_dipole_cutoff
    i, j, G = tnep_data._neighbor_dipole_pairs(
        atoms, float(cfg.dipole_rc or cfg.rcut_hard), int(cfg.dipole_channels), str(cfg.dipole_radial),
        float(cfg.dipole_zeta), resolved_dipole_cutoff(cfg), float(cfg.dipole_gauss_sigma_scale), 1)
    _, _, D = neighbor_list("ijD", atoms, float(cfg.dipole_rc or cfg.rcut_hard))
    # recover R_n from G = d ⊗ R_n using the largest bond component
    d = -D[(np.arange(len(D)) >= 0)]; d = d[np.linalg.norm(d, axis=1) > 0] if False else None
    T, C = int(cfg.num_types), int(cfg.dipole_channels)
    # G[p, :, n] = d_p R_n → R_n = G[p, a, n] / d_p[a] for the component a with max |d|
    Gn = np.abs(G).max(axis=2)                         # [P, 3]
    a = Gn.argmax(1)
    dd = tnep_data._neighbor_pairs(atoms, float(cfg.dipole_rc or cfg.rcut_hard))
    i2, j2, D2 = dd; keep = i2 != j2; d2 = -D2[keep].astype(float)
    assert np.array_equal(i2[keep], i) and np.array_equal(j2[keep], j)
    Rw = G[np.arange(len(G)), a, :] / d2[np.arange(len(G)), a][:, None]   # [P, C]
    shift = (y[i] - y[j])                                                   # [P, 3]
    Gs = G - shift[:, :, None] * Rw[:, None, :]
    V = np.zeros((len(atoms), T, 3, C)); np.add.at(V, (i, np.asarray(ti)[j]), Gs)
    return gd._collapse_V(V.transpose(0, 2, 3, 1), gd._pair_columns(cfg))


def fit_with_v(setup: gb.Setup, v_fn, feature_fn=None, delta: float = 1.0):
    """Production fit with a custom base geometry v_fn(atoms, ti) → [N, 3, K]
    and optional extra columns feature_fn(frames, tis) → [N, 3, n_extra]."""
    cfg = setup.cfg
    vs = [v_fn(a, ti) for a, ti in zip(setup.frames, setup.tis)]
    X = feature_fn(setup.frames, setup.tis) if feature_fn else [np.zeros((len(a), 3, 0)) for a in setup.frames]
    n_extra = X[0].shape[2]
    rms_v = np.sqrt(np.mean([(v ** 2).mean() for v in vs]))
    scale = (delta * rms_v / np.sqrt(np.mean([(x ** 2).mean(axis=(0, 1)) for x in X], axis=0))) if n_extra else np.zeros(0)
    K1 = setup.n_vcols + n_extra
    factors = gd._prior_factors(cfg, [setup.sparse], setup.xis, K1)

    def design(tis, descs, X, vs):
        return np.stack([gd._phi_frame([(D, setup.sparse)], np.concatenate([v, x * scale], 2), ti, setup.xis, None, gd._deltas(cfg))
                         for ti, D, x, v in zip(tis, descs, X, vs)]).reshape(3 * len(vs), -1)
    Phi = design(setup.tis, setup.descs, X, vs)
    beta = gb._solve([gd._whiten(Phi, factors, K1, False)], setup.targets(setup.frames).reshape(-1), setup.row_w(setup.frames))
    alpha = gd._whiten(beta[None], factors, K1, True)[0]

    def predict(frames):
        tis, descs = setup.prep(frames)
        v2 = [v_fn(a, ti) for a, ti in zip(frames, tis)]
        X2 = feature_fn(frames, tis) if feature_fn else [np.zeros((len(a), 3, 0)) for a in frames]
        return (design(tis, descs, X2, v2) @ alpha).reshape(len(frames), 3)
    predict.n_columns = Phi.shape[1]
    return predict
