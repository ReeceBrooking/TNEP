"""Tests for the GAP-mode dipole fit (gap_dipole.py)."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import contextlib
import json
from dataclasses import replace

import numpy as np
import pytest
from ase.io import read, write
from ase.neighborlist import neighbor_list

import data
import gap_dipole
from GAPconfig import GAPconfig

WATER_XYZ = os.path.join(ROOT, "datasets", "train_waterbulk.xyz")


def _cfg(**kw):
    cfg = GAPconfig()
    cfg.data_path = WATER_XYZ
    cfg.test_data_path = None
    cfg.target_mode = 1
    cfg.scale_targets = False
    cfg.convert_dipole_to_eangstrom = False
    cfg.dipole_units = "e*bohr"
    cfg.rcut_hard = 3.5
    cfg.rcut_soft = 3.0
    cfg.seed = 7
    cfg.total_N = 6
    # Pin ALL readout fields so working-tree experiment defaults never leak.
    cfg.dipole_readout = "neighbor"
    cfg.dipole_rc = 3.0
    cfg.dipole_channels = 1
    cfg.dipole_radial = "screened"
    cfg.dipole_zeta = -2.0
    cfg.dipole_cutoff = None                     # resolves to "cubic"
    cfg.dipole_species_pairs = "off"
    cfg.dipole_gauss_sigma_scale = 1.0
    # gap_2b/gap_3b/nep_descriptor need no pinning: GAPconfig has no ANN
    # knobs at all, and its NEP-input fields are frozen constants.
    cfg.gap_sparse_per_species = 40
    cfg.gap_kernel_power = 2
    cfg.gap_ridge_lambda = 1e-6
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


@contextlib.contextmanager
def _chdir(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    with _chdir(tmp_path_factory.mktemp("gap")):    # GAP/ record → tmp
        return gap_dipole.fit_gap_dipole(_cfg())


# ── identity: predict() vs hand-rolled per-atom loop ───────────────────────

def test_predict_identity(fitted):
    m = fitted
    at = read(WATER_XYZ, index="10")
    mu = gap_dipole.predict(m, [at])[0]
    ti = np.array([m.cfg.types.index(int(z)) for z in at.numbers])
    D = np.asarray(m.builder.build_descriptors(
        [at], calc_gradients=False)[0][0], dtype=np.float64)
    rc, zeta = float(m.cfg.dipole_rc), float(m.cfg.dipole_zeta)
    i, j, Dv = neighbor_list("ijD", at, rc)
    keep = i != j
    i, d = i[keep], -Dv[keep].astype(np.float64)    # d = r_i − r_j
    r = np.linalg.norm(d, axis=1)
    x = r / rc
    w = r ** (zeta - 1.0) * (1.0 - 3.0 * x ** 2 + 2.0 * x ** 3)
    v = np.zeros((len(at), 3))
    np.add.at(v, i, w[:, None] * d)
    mu_ref = np.zeros(3)
    for a in range(len(at)):
        s = sum(m.alpha[ti[a]][k]
                * float(D[a] @ m.sparse_desc[ti[a]][k]) ** m.xi[0]
                for k in range(len(m.alpha[ti[a]])))
        mu_ref += s * v[a]
    assert np.allclose(mu, mu_ref, atol=1e-6 * max(np.linalg.norm(mu), 1.0))


# ── chebyshev multi-channel + pair columns: predict() vs hand-rolled ───────

CHEB_PAIRS = dict(dipole_radial="chebyshev", dipole_cutoff="cosine2",
                  dipole_channels=3, dipole_species_pairs="ordered")
GAUSS_PAIRS = dict(dipole_radial="gaussian", dipole_cutoff="cosine2",
                   dipole_channels=3, dipole_species_pairs="ordered",
                   dipole_gauss_sigma_scale=0.5)


def test_chebyshev_pair_identity(tmp_path):
    with _chdir(tmp_path):
        m = gap_dipole.fit_gap_dipole(_cfg(**CHEB_PAIRS))
    at = read(WATER_XYZ, index="10")
    mu = gap_dipole.predict(m, [at])[0]
    ti = np.array([m.cfg.types.index(int(z)) for z in at.numbers])
    D = np.asarray(m.builder.build_descriptors(
        [at], calc_gradients=False)[0][0], dtype=np.float64)
    C, T, rc = 3, len(m.cfg.types), float(m.cfg.dipole_rc)
    i, j, Dv = neighbor_list("ijD", at, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -Dv[keep].astype(np.float64)  # d = r_i − r_j
    r = np.linalg.norm(d, axis=1)
    fc = ((1.0 + np.cos(np.pi * r / rc)) / 2.0) ** 2
    # own Chebyshev: T_n(x) = cos(n·arccos x), x = 2r/rc − 1 ∈ [−1, 1)
    Tn = np.cos(np.arange(C)[None, :] * np.arccos(2.0 * r / rc - 1.0)[:, None])
    v = np.zeros((len(at), 3, C, T))                 # per-(n, t′) accumulation
    for p in range(len(i)):
        for n in range(C):
            v[i[p], :, n, ti[j[p]]] += Tn[p, n] * fc[p] * d[p]
    mu_ref = np.zeros(3)
    for a in range(len(at)):
        t = ti[a]
        beta = m.alpha[t].reshape(-1, C * T)         # [M_t, K], k = n·T + t′
        for mm in range(beta.shape[0]):
            ker = float(D[a] @ m.sparse_desc[t][mm]) ** m.xi[0]
            for n in range(C):
                for tp in range(T):
                    mu_ref += ker * beta[mm, n * T + tp] * v[a, :, n, tp]
    assert np.allclose(mu, mu_ref, atol=1e-6 * max(np.linalg.norm(mu), 1.0))


# ── exactness from a KNOWN α: μ = Φα* recovered at λ → 0 ───────────────────

@pytest.mark.parametrize("overrides", [{}, CHEB_PAIRS, GAUSS_PAIRS],
                         ids=["screened_off", "chebyshev_pairs",
                              "gaussian_pairs"])
def test_exact_recovery_known_alpha(tmp_path, overrides):
    frames = read(WATER_XYZ, index="0:6")
    for f in frames:
        f.calc = None                # else write() collides with info key
        f.info["dipole"] = np.zeros(3)
    p0 = str(tmp_path / "zeros.xyz")
    write(p0, frames)
    kw = dict(total_N=None, gap_sparse_per_species=8,
              gap_ridge_lambda=1e-12, **overrides)
    with _chdir(tmp_path):
        base = gap_dipole.fit_gap_dipole(_cfg(data_path=p0, **kw))
    rng = np.random.default_rng(0)
    star = replace(base, alpha=[rng.normal(size=len(a)) for a in base.alpha])
    mu_star = gap_dipole.predict(star, frames)      # μ = Φα* by construction
    for f, mu in zip(frames, mu_star):
        f.info["dipole"] = mu
    p1 = str(tmp_path / "synth.xyz")
    write(p1, frames)
    with _chdir(tmp_path):                          # same seed → same FPS/Φ
        refit = gap_dipole.fit_gap_dipole(_cfg(data_path=p1, **kw))
    assert np.allclose(gap_dipole.predict(refit, frames), mu_star,
                       atol=1e-8 * max(np.abs(mu_star).max(), 1.0))


# ── species-block isolation ────────────────────────────────────────────────

def test_species_block_isolation(fitted):
    m = fitted
    at = read(WATER_XYZ, index="3")
    tO, tH = m.cfg.types.index(8), m.cfg.types.index(1)
    alpha = list(m.alpha)
    alpha[tH] = np.zeros_like(alpha[tH])
    zero_h = replace(m, alpha=alpha)
    Q = m.sparse_desc[tO].shape[1]
    sparse = list(m.sparse_desc)
    sparse[tH] = np.zeros((0, Q))
    a2 = list(alpha)
    a2[tH] = np.zeros(0)
    o_only = replace(m, sparse_desc=sparse, alpha=a2)
    mu_zero_h = gap_dipole.predict(zero_h, [at])
    assert np.allclose(mu_zero_h, gap_dipole.predict(o_only, [at]),
                       atol=1e-12)
    # and the H block actually contributed something to the full model
    assert not np.allclose(mu_zero_h, gap_dipole.predict(m, [at]), atol=1e-6)


# ── translation + rotation invariance ──────────────────────────────────────

def test_translation_invariance(fitted):
    at = read(WATER_XYZ, index="0")
    mu0 = gap_dipole.predict(fitted, [at])[0]
    at2 = at.copy()
    at2.positions = at2.positions + np.array([1.234, -2.517, 0.771])
    mu1 = gap_dipole.predict(fitted, [at2])[0]
    assert np.allclose(mu0, mu1, atol=1e-4 * max(np.linalg.norm(mu0), 1.0))


def test_rotation_covariance(fitted):
    at = read(WATER_XYZ, index="0")
    mu0 = gap_dipole.predict(fitted, [at])[0]
    a, b, c = 0.7, -0.4, 1.1
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0],
                   [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)],
                   [0, np.sin(c), np.cos(c)]])
    R = Rz @ Ry @ Rx
    at2 = at.copy()
    at2.set_cell(at.cell[:] @ R.T, scale_atoms=False)
    at2.positions = at.positions @ R.T
    mu_rot = gap_dipole.predict(fitted, [at2])[0]
    assert np.allclose(mu_rot, R @ mu0,
                       atol=1e-3 * max(np.linalg.norm(mu0), 1.0))


# ── determinism: same cfg/seed ⇒ identical α ───────────────────────────────

def test_determinism(tmp_path):
    with _chdir(tmp_path):
        m1 = gap_dipole.fit_gap_dipole(_cfg())
        m2 = gap_dipole.fit_gap_dipole(_cfg())
    for a1, a2 in zip(m1.alpha, m2.alpha):
        assert np.array_equal(a1, a2)


# ── pair columns strictly subsume "off": ON fit recovers an OFF model ──────

def test_pair_columns_subsume_off(tmp_path):
    frames = read(WATER_XYZ, index="0:6")
    for f in frames:
        f.calc = None                # else write() collides with info key
        f.info["dipole"] = np.zeros(3)   # OFF-fit targets are irrelevant
    p0 = str(tmp_path / "zeros.xyz")
    write(p0, frames)
    kw = dict(total_N=None, gap_sparse_per_species=8, gap_ridge_lambda=1e-12,
              dipole_radial="chebyshev", dipole_cutoff="cosine2",
              dipole_channels=2)
    with _chdir(tmp_path):
        off = gap_dipole.fit_gap_dipole(
            _cfg(data_path=p0, dipole_species_pairs="off", **kw))
    rng = np.random.default_rng(1)
    off = replace(off, alpha=[rng.normal(size=len(a)) for a in off.alpha])
    mu_off = gap_dipole.predict(off, frames)
    for f, mu in zip(frames, mu_off):
        f.info["dipole"] = mu
    p1 = str(tmp_path / "off_targets.xyz")
    write(p1, frames)
    with _chdir(tmp_path):
        on = gap_dipole.fit_gap_dipole(
            _cfg(data_path=p1, dipole_species_pairs="ordered", **kw))
    assert np.allclose(gap_dipole.predict(on, frames), mu_off,
                       atol=1e-8 * max(np.abs(mu_off).max(), 1.0))


# ── sum-of-kernels: ξ list column blocks ───────────────────────────────────

def test_sum_kernel_identity(tmp_path):
    """predict() with ξ=[1,3] vs hand-rolled numpy: own kernel powers, own
    block order (species → ξ → (m, k))."""
    with _chdir(tmp_path):
        m = gap_dipole.fit_gap_dipole(
            _cfg(**CHEB_PAIRS, gap_kernel_power=[1, 3],
                 gap_sparse_per_species=8))
    at = read(WATER_XYZ, index="10")
    mu = gap_dipole.predict(m, [at])[0]
    ti = np.array([m.cfg.types.index(int(z)) for z in at.numbers])
    D = np.asarray(m.builder.build_descriptors(
        [at], calc_gradients=False)[0][0], dtype=np.float64)
    C, T, rc = 3, len(m.cfg.types), float(m.cfg.dipole_rc)
    i, j, Dv = neighbor_list("ijD", at, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -Dv[keep].astype(np.float64)
    r = np.linalg.norm(d, axis=1)
    fc = ((1.0 + np.cos(np.pi * r / rc)) / 2.0) ** 2
    Tn = np.cos(np.arange(C)[None, :] * np.arccos(2.0 * r / rc - 1.0)[:, None])
    v = np.zeros((len(at), 3, C, T))
    for p in range(len(i)):
        for n in range(C):
            v[i[p], :, n, ti[j[p]]] += Tn[p, n] * fc[p] * d[p]
    K = C * T
    mu_ref = np.zeros(3)
    for a in range(len(at)):
        t = ti[a]
        M_t = len(m.sparse_desc[t])
        beta = m.alpha[t].reshape(2, M_t, K)         # ξ-major, then (m, k)
        for x, xi in enumerate((1, 3)):              # own powers
            for mm in range(M_t):
                ker = float(D[a] @ m.sparse_desc[t][mm]) ** xi
                for n in range(C):
                    for tp in range(T):
                        mu_ref += (ker * beta[x, mm, n * T + tp]
                                   * v[a, :, n, tp])
    assert np.allclose(mu, mu_ref, atol=1e-6 * max(np.linalg.norm(mu), 1.0))


def test_sum_kernel_singleton_list_equals_int(tmp_path):
    """Falsifier for the normalization path: ξ=[2] must equal ξ=2 EXACTLY."""
    with _chdir(tmp_path):
        m_int = gap_dipole.fit_gap_dipole(_cfg(gap_kernel_power=2))
        m_list = gap_dipole.fit_gap_dipole(_cfg(gap_kernel_power=[2]))
    assert m_list.xi == m_int.xi == (2,)
    for a1, a2 in zip(m_int.alpha, m_list.alpha):
        assert np.array_equal(a1, a2)


# ── multi-scale second cutoff ──────────────────────────────────────────────

def test_multiscale_exact_recovery_known_alpha(tmp_path):
    """Pins the two-scale Φ assembly: a known α (both scale blocks) is
    recovered exactly at λ → 0."""
    frames = read(WATER_XYZ, index="0:6")
    for f in frames:
        f.calc = None
        f.info["dipole"] = np.zeros(3)
    p0 = str(tmp_path / "zeros.xyz")
    write(p0, frames)
    kw = dict(total_N=None, gap_sparse_per_species=8, gap_ridge_lambda=1e-12,
              gap_extra_rcut=4.5, gap_kernel_power=[1, 2], **CHEB_PAIRS)
    with _chdir(tmp_path):
        base = gap_dipole.fit_gap_dipole(_cfg(data_path=p0, **kw))
    assert base.builder2 is not None and base.alpha2 is not None
    rng = np.random.default_rng(0)
    star = replace(base,
                   alpha=[rng.normal(size=len(a)) for a in base.alpha],
                   alpha2=[rng.normal(size=len(a)) for a in base.alpha2])
    mu_star = gap_dipole.predict(star, frames)      # μ = Φα* by construction
    for f, mu in zip(frames, mu_star):
        f.info["dipole"] = mu
    p1 = str(tmp_path / "synth.xyz")
    write(p1, frames)
    with _chdir(tmp_path):                          # same seed → same FPS/Φ
        refit = gap_dipole.fit_gap_dipole(_cfg(data_path=p1, **kw))
    assert np.allclose(gap_dipole.predict(refit, frames), mu_star,
                       atol=1e-8 * max(np.abs(mu_star).max(), 1.0))


def test_multiscale_off_default_bitcompat(tmp_path):
    """gap_extra_rcut=None + single-int ξ reproduces a frozen α on the small
    fixture, so the single-scale path cannot drift.

    The hash was re-captured on 2026-09-30 when the augmented-lstsq ("primal")
    solver was removed: the dual ridge is the same estimator but differs in the
    last bits, so the pre-removal baseline no longer applies. It therefore pins
    stability from here forward, not equality with the pre-multiscale code."""
    import hashlib
    with _chdir(tmp_path):
        m = gap_dipole.fit_gap_dipole(_cfg())
    assert m.builder2 is None and m.sparse_desc2 is None and m.alpha2 is None
    h = hashlib.sha256(np.concatenate(m.alpha).tobytes()).hexdigest()
    assert h == ("307cf80b358d1ddece1a767474aaeb48"
                 "9856a6fd8883e4da8e61d0c71bc3d700")


# ── config validation ──────────────────────────────────────────────────────

@pytest.mark.parametrize("kw,msg", [
    (dict(scale_targets=True), "scale_targets"),
    (dict(dipole_channels=4), "dipole_channels"),   # screened ⇒ C == 1
    (dict(dipole_channels=0), "dipole_channels"),
    (dict(dipole_readout="tnep"), "dipole_readout"),
    (dict(target_mode=0), "target_mode"),
    (dict(gap_kernel_power=[0]), "gap_kernel_power"),
    (dict(gap_kernel_power=[]), "gap_kernel_power"),
    (dict(gap_extra_rcut=-1.0), "gap_extra_rcut"),
])
def test_validation_raises(kw, msg):
    cfg = _cfg(data_path="/nonexistent.xyz", **kw)  # raise happens FIRST
    with pytest.raises(ValueError, match=msg):
        gap_dipole.fit_gap_dipole(cfg)


def test_extra_rcut_below_rcut_hard_warns():
    with pytest.warns(UserWarning, match="gap_extra_rcut"):
        gap_dipole._validate(_cfg(gap_extra_rcut=3.0))  # ≤ rcut_hard=3.5


@pytest.mark.parametrize("kw", [
    dict(dipole_species_pairs=False),               # legacy bool "off"
    dict(dipole_radial="chebyshev", dipole_channels=4),
    dict(dipole_cutoff="cosine2"),
    dict(dipole_species_pairs="ordered"),
    dict(dipole_species_pairs="symmetric"),
])
def test_validation_accepts(kw):
    gap_dipole._validate(_cfg(**kw))                # no raise


# ── run record ─────────────────────────────────────────────────────────────

def test_run_record(tmp_path):
    with _chdir(tmp_path):
        gap_dipole.fit_gap_dipole(_cfg(gap_sparse_per_species=10))
        records = os.listdir("GAP")
        assert len(records) == 1 and records[0].startswith("gap_M10_x2_")
        with open(os.path.join("GAP", records[0])) as fh:
            rec = json.load(fh)
    assert rec["config"]["dipole_radial"] == "screened"
    assert rec["config"]["gap_ridge_lambda"] == 1e-6
    assert rec["gap"]["sparse_per_species"] == 10
    assert set(rec["gap"]["M_per_species"]) == {"8", "1"}
    assert rec["gap"]["radial"] == "screened"
    assert rec["gap"]["cutoff"] == "cubic"
    assert rec["gap"]["channels"] == 1
    assert rec["gap"]["pair_columns"] is False
    assert rec["gap"]["n_columns"] == sum(rec["gap"]["M_per_species"].values())
    for key in ("train_rmse", "train_r2", "fit_seconds"):
        assert key in rec["metrics"]


def test_dual_solve_matches_augmented_lstsq(tmp_path):
    """The dual ridge α = Φᵀ(ΦΦᵀ+λI)⁻¹μ the fit uses is a matrix identity
    with the augmented [Φ; √λ·I] least-squares solve. The augmented route is
    no longer in the code (it peaked at ~n_columns² memory), so it lives here
    as a test-only oracle: it is the better-conditioned of the two, and this
    is what pins the production solve."""
    with _chdir(tmp_path):
        model = gap_dipole.fit_gap_dipole(_cfg(**CHEB_PAIRS))
    cfg = model.cfg
    # Rebuild the same design the fit solved, frame by frame (the column
    # contract lives in _phi_frame, so this cannot drift from the fit).
    dataset = read(cfg.data_path, index=":")
    dataset = [dataset[k] for k in cfg.indices]
    tis = data.assign_type_indices(dataset, cfg.types)
    descs = gap_dipole._descriptors(model.builder, dataset)
    scales = [(descs, model.sparse_desc)]
    Phi = np.stack([gap_dipole._phi_frame(
        [(descs[f], model.sparse_desc)], gap_dipole._v_atoms(dataset[f], tis[f], cfg),
        tis[f], model.xi) for f in range(len(dataset))])
    Phi_flat = Phi.reshape(-1, Phi.shape[-1])
    mu = gap_dipole._targets(dataset, cfg)
    lam_eff = (float(cfg.gap_ridge_lambda)
               * float(np.mean(np.sum(Phi_flat ** 2, axis=0))))
    n_cols = Phi_flat.shape[1]
    A = np.vstack([Phi_flat, np.sqrt(lam_eff) * np.eye(n_cols)])
    b = np.concatenate([mu.reshape(-1), np.zeros(n_cols)])
    alpha_aug = np.linalg.lstsq(A, b, rcond=None)[0]
    alpha_fit = np.concatenate(model.alpha)
    # Predictions must agree; the coefficients themselves need not (the
    # design is ill-conditioned, see GapDipoleModel's docstring).
    pred_aug = (Phi_flat @ alpha_aug).reshape(-1, 3)
    pred_fit = (Phi_flat @ alpha_fit).reshape(-1, 3)
    assert np.allclose(pred_aug, pred_fit,
                       rtol=1e-6, atol=1e-8 * np.abs(pred_fit).max())


def test_gap_rejects_non_neighbor_readout():
    # The convex fit needs μ linear in the site scalars, which only the
    # neighbour readout gives, so GAP mode must refuse every other readout.
    with pytest.raises(ValueError, match="dipole_readout"):
        gap_dipole._validate(_cfg(dipole_readout="tnep"))


# ── the split: GAPconfig and TNEPconfig are independent classes ────────────

def test_configs_are_independent():
    """GAPconfig carries no ANN-only field (so an ANN experiment cannot reach
    the kernel fit), TNEPconfig still carries everything the ANN needs, and
    neither is a subclass of the other — shared fields are duplicated."""
    from TNEPconfig import TNEPconfig
    ann_only = ("num_neurons", "activation", "optimizer", "pop_size",
                "descriptor_mixing", "dipole_head")
    for f in ann_only:
        assert not hasattr(GAPconfig, f), f"GAPconfig must not carry {f}"
        assert hasattr(TNEPconfig, f), f"TNEPconfig lost {f}"
    # no inheritance in either direction, and no shared base but object
    assert not issubclass(GAPconfig, TNEPconfig)
    assert not issubclass(TNEPconfig, GAPconfig)
    assert GAPconfig.__bases__ == (object,) == TNEPconfig.__bases__
    # every field GAPconfig declares also exists on TNEPconfig (duplicates,
    # not renames) — except descriptor_self_batch_size, which data.py reads
    # via getattr and TNEPconfig never declared.
    missing = (set(GAPconfig.__annotations__) - set(TNEPconfig.__annotations__)
               - {"descriptor_self_batch_size"})
    assert not missing, missing
