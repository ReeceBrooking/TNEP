"""Tests for the neighbor-basis dipole readout (plan Phase 4, tests 10-18t).

Gradient tests (12, 13, 13b, 17t) are deferred until dipole_gradient exists
(plan Phase 3).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import functools

import numpy as np
import pytest
import tensorflow as tf
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list

from TNEPconfig import TNEPconfig
from TNEP import TNEP
from DescriptorBuilder import make_descriptor_builder
from DescriptorBuilderGPU import compute_dim_q
from data import (prepare_eval_data, assign_type_indices, assemble_data_dict,
                  pad_and_stack, _neighbor_dipole_guards, _neighbor_dipole_V)

WATER_XYZ = os.path.join(ROOT, "datasets", "train_waterbulk.xyz")


def _cfg(**kw):
    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.dipole_readout = "neighbor"
    cfg.rcut_hard = 3.5
    cfg.rcut_soft = 3.0
    cfg.dipole_rc = 3.0
    cfg.num_neurons = 8
    cfg.activation = "tanh"
    cfg.descriptor_mixing = False
    cfg.seed = 7
    cfg.types = [8, 1]
    cfg.num_types = 2
    cfg.type_map = {8: 0, 1: 1}
    # Pin readout fields so working-tree experiment defaults in TNEPconfig
    # never leak into the suite (kwargs still override).
    cfg.dipole_channels = 4
    cfg.dipole_species_pairs = "off"
    cfg.dipole_radial = "chebyshev"
    cfg.dipole_zeta = -2.0
    cfg.dipole_cutoff = None
    cfg.dipole_gauss_sigma_scale = 1.0
    cfg.l_max = 4
    cfg.alpha_max = 4
    cfg.radial_basis_size = None
    cfg.radial_basis_species_sum = False
    cfg.radial_basis_per_l = False
    cfg.optimizer = "snes"
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.dim_q = compute_dim_q(cfg)
    return cfg


def _model(cfg):
    tf.keras.utils.set_random_seed(0)
    return TNEP(cfg)


def _bulk_frame():
    at = read(WATER_XYZ, index="0")
    at.info["dipole"] = np.zeros(3)
    return at


def _mu(model, atoms, cfg):
    """μ through the production eval path (predict_batch, V via W_atom slot)."""
    data = prepare_eval_data([atoms], cfg)
    return model.predict_batch(
        data["descriptors"], data["grad_values"],
        data["pair_atom"], data["pair_gidx"], data["pair_struct"],
        data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
        model.W0, model.b0, model.W1, model.b1,
        W_atom=data.get("V_atom"))[0].numpy()


# Readout variants shared across the parametrized suites. "screened" is the
# user-mandated BASIC form: single channel, c ≡ 1 frozen, ζ = −2.
_MODE_KW = {
    "chebyshev": {},
    "screened": dict(dipole_radial="screened", dipole_channels=1,
                     dipole_zeta=-2.0),
    "ordered": dict(dipole_species_pairs="ordered"),
    "chebyshev_warped": dict(dipole_radial="chebyshev_warped"),
    "gaussian": dict(dipole_radial="gaussian"),
    "cheb_tanh3": dict(dipole_cutoff="tanh3"),
}


@functools.lru_cache(maxsize=8)
def _bulk_setup(mode="chebyshev"):
    """(cfg, model, atoms, padded-data) for the 96-atom water frame — shared.
    "ordered" gets a randomized ASYMMETRIC Ω (symmetric Ω would telescope
    back to the difference form and hide ordered-path bugs)."""
    cfg = _cfg(**_MODE_KW[mode])
    model = _model(cfg)
    if mode == "ordered":
        rng = np.random.default_rng(11)
        model.c_pair.assign(rng.uniform(0.3, 1.7, size=model.c_pair.shape)
                            .astype(np.float32))
    atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    return cfg, model, atoms, data


def _site_scalars_np(model, data):
    """s_i^n replicated in numpy float64 (tanh ANN, b1 excluded by design)."""
    desc = data["descriptors"][0].numpy().astype(np.float64)   # [A, Q]
    Z = data["Z_int"][0].numpy()                                # [A]
    W0 = model.W0.numpy().astype(np.float64)
    b0 = model.b0.numpy().astype(np.float64)
    W1 = model.W1.numpy().astype(np.float64)                    # [T, H, C]
    h = np.tanh(np.einsum('aq,aqh->ah', desc, W0[Z]) + b0[Z])
    return np.einsum('ah,ahn->an', h, W1[Z])                    # [A, C]


def _fc(r, rc):
    return ((1.0 + np.cos(np.pi * r / rc)) / 2.0) ** 2


def _R_np(r, rc, C):
    """R_n(r) = T_n(2r/rc − 1)·f_c(r), Chebyshev recurrence → [P, C]."""
    x = 2.0 * r / rc - 1.0
    Tn = np.empty((len(r), C))
    Tn[:, 0] = 1.0
    if C > 1:
        Tn[:, 1] = x
    for n in range(2, C):
        Tn[:, n] = 2.0 * x * Tn[:, n - 1] - Tn[:, n - 2]
    return Tn * _fc(r, rc)[:, None]


def _pair_index(T):
    """Unordered-pair index table [T, T] matching TNEP._pair_idx."""
    M = np.zeros((T, T), dtype=int)
    p = 0
    for t in range(T):
        for t2 in range(t, T):
            M[t, t2] = M[t2, t] = p
            p += 1
    return M


# ── 10: pair-difference double sum == model's per-atom collapsed form ───────

def test_pair_difference_sum_matches_model():
    # Learned c exercised: dipole_species_pairs=True + randomized table.
    cfg = _cfg(dipole_species_pairs=True)
    model = _model(cfg)
    rng = np.random.default_rng(3)
    model.c_pair.assign(rng.uniform(0.3, 1.7, size=model.c_pair.shape)
                        .astype(np.float32))
    atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    s = _site_scalars_np(model, data)                          # [A, C]
    mu_model = _mu(model, atoms, cfg)

    Zt = data["Z_int"][0].numpy()
    c = model.c_pair.numpy().astype(np.float64)                # [C, P]
    pidx = _pair_index(cfg.num_types)
    i, j, D = neighbor_list("ijD", atoms, float(cfg.dipole_rc))
    keep = i != j
    i, j, D = i[keep], j[keep], D[keep].astype(np.float64)
    d = -D                                       # d = r_i − r_j (+ image)
    r = np.linalg.norm(d, axis=1)
    R = _R_np(r, float(cfg.dipole_rc), int(cfg.dipole_channels))  # [P, C]
    cn = c[:, pidx[Zt[i], Zt[j]]].T                               # [P, C]
    # b_ij = Σ_n c_n[p(t_i,t_j)]·R_n(r_ij)·(s_i^n − s_j^n); ordered pairs
    # count each unordered pair twice with identical term.
    b = np.sum(cn * R * (s[i] - s[j]), axis=1)                    # [P]
    mu_pair = 0.5 * np.sum(b[:, None] * d, axis=0)

    scale = max(np.linalg.norm(mu_model), 1e-6)
    assert np.allclose(mu_model, mu_pair, atol=1e-4 * max(scale, 1.0), rtol=1e-4)


# ── 11: invariance suite ────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["chebyshev", "screened", "ordered",
                                  "gaussian"])
def test_rigid_translation_invariance(mode):
    cfg, model, atoms, _ = _bulk_setup(mode)
    mu0 = _mu(model, atoms, cfg)
    at2 = atoms.copy()
    at2.positions = at2.positions + np.array([1.234, -2.517, 0.771])
    mu1 = _mu(model, at2, cfg)
    assert np.allclose(mu0, mu1, atol=2e-3)


@pytest.mark.parametrize("mode", ["chebyshev", "screened", "ordered",
                                  "gaussian"])
def test_rotation_covariance(mode):
    cfg, model, atoms, _ = _bulk_setup(mode)
    mu0 = _mu(model, atoms, cfg)
    a, b, c = 0.7, -0.4, 1.1
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    R = Rz @ Ry @ Rx
    at2 = atoms.copy()
    at2.set_cell(atoms.cell[:] @ R.T, scale_atoms=False)
    at2.positions = atoms.positions @ R.T
    mu_rot = _mu(model, at2, cfg)
    assert np.allclose(mu_rot, R @ mu0, atol=2e-3)


@pytest.mark.parametrize("mode", ["chebyshev", "screened", "ordered",
                                  "gaussian"])
def test_single_atom_lattice_translation_invariance(mode):
    cfg, model, atoms, _ = _bulk_setup(mode)
    mu0 = _mu(model, atoms, cfg)
    at2 = atoms.copy()
    at2.positions[5] += atoms.cell[:][0]         # +one cell vector, atom 5
    mu1 = _mu(model, at2, cfg)
    assert np.allclose(mu0, mu1, atol=2e-3)


# ── 13c: four-path forward consistency (the drift alarm) ────────────────────

@pytest.mark.parametrize("mode", ["chebyshev", "screened", "chebyshev_warped",
                                  "gaussian", "cheb_tanh3"])
def test_four_path_consistency(mode):
    # The only test cross-checking all three radial dispatch sites (numpy
    # twin, predict inline, _neighbor_V_coo) against each other — catches a
    # ζ/radial mismatch between them. The in-graph/predict legs are the ONLY
    # automated execution of the TF _radial/_fc branches (every μ-comparing
    # test consumes the numpy-built V via W_atom), so the new radials and
    # the tanh3 cross MUST be parametrized here.
    cfg, model, atoms, _ = _bulk_setup(mode)
    # Full pair stream (self_pairs_only=False) so the in-graph V path has
    # the r_d COO list to build V from.
    types_int = assign_type_indices([atoms], cfg.types)
    builder = make_descriptor_builder(cfg)
    desc, grads, gidx = builder.build_descriptors([atoms])
    d = assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg)
    full = pad_and_stack(d, num_types=cfg.num_types, self_pairs_only=False)

    args = (full["descriptors"], full["grad_values"],
            full["pair_atom"], full["pair_gidx"], full["pair_struct"],
            full["positions"], full["Z_int"], full["boxes"], full["atom_mask"],
            model.W0, model.b0, model.W1, model.b1)
    mu_slot = model.predict_batch(*args, W_atom=full["V_atom"])[0].numpy()
    mu_ingraph = model.predict_batch(*args, W_atom=None)[0].numpy()

    mu_cand = model.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        model.W0[tf.newaxis], model.b0[tf.newaxis],
        model.W1[tf.newaxis], model.b1[tf.newaxis])[0, 0].numpy()

    # Single-structure predict(): dense [A, M] neighbor table from the builder.
    gi = gidx[0]
    A = len(atoms)
    M = max(len(lst) for lst in gi)
    grad_index = np.zeros((A, M), dtype=np.int32)
    nmask = np.zeros((A, M), dtype=np.float32)
    for a, lst in enumerate(gi):
        grad_index[a, :len(lst)] = lst
        nmask[a, :len(lst)] = 1.0
    mu_single = model.predict(
        full["descriptors"][0],
        tf.zeros([A, M, 3, cfg.dim_q]),          # unused by neighbor readout
        tf.constant(grad_index), full["positions"][0], full["Z_int"][0],
        full["boxes"][0], full["atom_mask"][0], tf.constant(nmask)).numpy()

    atol = 1e-4 * max(np.linalg.norm(mu_slot), 1.0)
    assert np.allclose(mu_slot, mu_ingraph, atol=atol)
    assert np.allclose(mu_slot, mu_cand, atol=atol)
    assert np.allclose(mu_slot, mu_single, atol=atol)


# ── 13d: branch-consistency label guard ─────────────────────────────────────

def test_branch_inconsistency_warning(capsys):
    cfg = _cfg(convert_dipole_to_eangstrom=True)  # quantum = L_min directly
    frame = _bulk_frame()
    dataset = [frame] * 5
    L = float(frame.cell[0, 0])
    tame = [np.array([0.1 * k, 0.0, 0.0]) for k in range(5)]
    _neighbor_dipole_guards(dataset, tame, cfg)
    assert "branch-inconsistent" not in capsys.readouterr().out
    offset = list(tame)
    offset[2] = offset[2] + np.array([L, 0.0, 0.0])  # one frame off by ~e·L
    _neighbor_dipole_guards(dataset, offset, cfg)
    assert "branch-inconsistent" in capsys.readouterr().out


# ── 14: smoothness at r_c ───────────────────────────────────────────────────

_SCAN_H = 0.03


def _rc_scan(cfg):
    """μ(r) for an OH dimer scanned through cfg.dipole_rc → (rs, mus [K, 3])."""
    model = _model(cfg)
    rc = float(cfg.dipole_rc)
    rs = np.arange(rc - 0.3, rc + 0.3 + 1e-9, _SCAN_H)
    mus = []
    for r in rs:
        at = Atoms("OH", positions=[[10.0, 10.0, 10.0], [10.0 + r, 10.0, 10.0]],
                   cell=20.0 * np.eye(3), pbc=True)
        at.info["dipole"] = np.zeros(3)
        mus.append(_mu(model, at, cfg))
    return rs, np.array(mus)


def _assert_smooth_at_rc(cfg, cutoff):
    """Continuity of μ at r_c, keyed on the CUTOFF (radial-agnostic):
    "cosine2" is a C³ contact — value, μ′ AND μ″ vanish into the
    identically-zero region; "tanh3" is C² — value, μ′, μ″ vanish with
    pinned looser gates; "cubic"/"cosine" are C¹ only — value + μ′ vanish,
    the μ″ jump at r_c is EXPECTED, so d2 is deliberately unasserted."""
    rc, h = float(cfg.dipole_rc), _SCAN_H
    rs, mus = _rc_scan(cfg)
    # Beyond r_c: exactly zero (V has no pairs).
    assert np.allclose(mus[rs >= rc], 0.0, atol=1e-12)
    d1 = (mus[2:] - mus[:-2]) / (2 * h)
    n_mu = np.linalg.norm(mus, axis=1)
    n_d1 = np.linalg.norm(d1, axis=1)
    r_mid = rs[1:-1]
    k_last = np.max(np.where(rs < rc))                    # last point inside
    near = np.abs(r_mid - rc) < 1.5 * h                   # FD points straddling rc
    if cutoff == "cosine2":
        # μ ∝ (rc − r)^4 near rc: value, μ', μ'' all continuous — no jump.
        d2 = (mus[2:] - 2 * mus[1:-1] + mus[:-2]) / h ** 2
        n_d2 = np.linalg.norm(d2, axis=1)
        assert n_mu[k_last] < 1e-3 * n_mu.max()
        assert n_d1[near].max() < 0.05 * n_d1.max()
        assert n_d2[near].max() < 0.10 * n_d2.max()
    elif cutoff == "tanh3":
        # C² contact: μ ~ A(r_c − r)³ near r_c, so the last inside point
        # (s = h = 0.03 into a 0.3 window) has value ratio (h/0.3)³ = 1e-3
        # and d2 ratio h/0.3 = 0.10 — exactly on the cosine2 gates. The
        # pinned looser gates below pass cleanly while a C¹ bug fails by
        # ~4×; float32 FD noise sits ~10³ below signal.
        d2 = (mus[2:] - 2 * mus[1:-1] + mus[:-2]) / h ** 2
        n_d2 = np.linalg.norm(d2, axis=1)
        assert n_mu[k_last] < 5e-3 * n_mu.max()
        assert n_d1[near].max() < 0.05 * n_d1.max()
        assert n_d2[near].max() < 0.25 * n_d2.max()
    else:
        # μ ∝ (rc − r)² near rc: value ~h² and slope ~h at the last inside
        # points relative to their scan maxima.
        assert n_mu[k_last] < 0.02 * n_mu.max()
        assert n_d1[near].max() < 0.15 * n_d1.max()


def test_smoothness_at_rc():
    # dipole_rc strictly inside rcut so the descriptor is smooth at r_c.
    # Default (None) cutoff for chebyshev resolves to "cosine2".
    cfg = _cfg(rcut_hard=4.5, rcut_soft=4.0, dipole_rc=3.0)
    _assert_smooth_at_rc(cfg, "cosine2")


@pytest.mark.parametrize("radial,cutoff", [("chebyshev", "cubic"),
                                           ("screened", "cosine2")])
def test_cross_combo_smoothness_at_rc(radial, cutoff):
    # The two NEW radial×cutoff crosses: smoothness class follows the CUTOFF.
    cfg = _cfg(rcut_hard=4.5, rcut_soft=4.0, dipole_rc=3.0,
               dipole_cutoff=cutoff, **_MODE_KW[radial])
    _assert_smooth_at_rc(cfg, cutoff)


@pytest.mark.parametrize("cutoff", ["cosine", "tanh3"])
def test_new_cutoff_smoothness_at_rc(cutoff):
    # "cosine" C¹ (value + d1 only); "tanh3" C² (value + d1 + d2, pinned).
    cfg = _cfg(rcut_hard=4.5, rcut_soft=4.0, dipole_rc=3.0,
               dipole_cutoff=cutoff)
    _assert_smooth_at_rc(cfg, cutoff)


# ── 15: single H₂O sanity — μ nonzero, along the molecular bisector ─────────

def test_water_monomer_bisector():
    cfg = _cfg()
    model = _model(cfg)
    at = Atoms("OHH",
               positions=[[10.0, 10.0, 10.0],
                          [10.7568, 10.5859, 10.0],
                          [9.2432, 10.5859, 10.0]],     # bisector = +y
               cell=20.0 * np.eye(3), pbc=True)
    at.info["dipole"] = np.zeros(3)
    mu = _mu(model, at, cfg)
    assert np.linalg.norm(mu) > 1e-6
    # Symmetry forces μ ∥ bisector (up to sign) for any weights.
    assert abs(mu[0]) < 1e-3 * abs(mu[1])
    assert abs(mu[2]) < 1e-3 * abs(mu[1])


# ── 16t: gauge — s → s + c shifts μ by c·Σ_i v_i ≡ 0 ────────────────────────

@pytest.mark.parametrize("mode", ["chebyshev", "screened"])
def test_gauge_shift_cancels(mode):
    # "off"/"symmetric" only: with an asymmetric ordered Ω the s → s + g
    # residual Σ g·R·d·(Ω[t,t'] − Ω[t',t]) is nonzero BY DESIGN (gauge
    # invariance is deliberately traded away), so no "ordered" case here.
    cfg, model, atoms, data = _bulk_setup(mode)
    V = data["V_atom"][0].numpy()               # [A, 3, C, T], model's own V
    # Per channel n: shifting s^n uniformly across ALL atoms moves μ by
    # g_n·Σ_{a,t'} V[a,:,n,t'] ≡ 0 (ordered-pair cancellation, c symmetric).
    residual = np.abs(V.sum(axis=(0, 3)))                # [3, C]
    total = np.abs(V).sum(axis=(0, 3))                   # [3, C]
    assert np.all(residual < 1e-4 * total)


# ── 17: species-pair coefficient table ──────────────────────────────────────

def test_same_species_coefficient_zeroes_contribution():
    # Frame of only O atoms → every pair is the OO pair. c[:, OO]=0 must
    # zero μ identically; c=1 must leave it nonzero (random W1 environments).
    cfg = _cfg(dipole_species_pairs=True)
    model = _model(cfg)
    at = Atoms("O4", positions=[[8.0, 8.0, 8.0], [9.1, 8.4, 8.2],
                                [8.3, 9.2, 8.9], [10.0, 9.5, 8.1]],
               cell=20.0 * np.eye(3), pbc=True)
    at.info["dipole"] = np.zeros(3)
    oo = _pair_index(cfg.num_types)[0, 0]         # types [8, 1] → O is type 0
    c = np.ones(model.c_pair.shape, dtype=np.float32)
    model.c_pair.assign(c)
    mu_on = _mu(model, at, cfg)
    c[:, oo] = 0.0
    model.c_pair.assign(c)
    mu_off = _mu(model, at, cfg)
    assert np.linalg.norm(mu_on) > 1e-6
    assert np.allclose(mu_off, 0.0, atol=1e-10)


def test_species_pairs_off_is_frozen_ones():
    # "off": no c parameters in the SNES tail, and the forward equals the
    # learnable model at its c=1 init. Legacy bools (old configs/saves)
    # normalize at model init: False→"off", True→"symmetric".
    from SNES import SNES
    cfg_off = _cfg(dipole_species_pairs=False)             # legacy bool
    model_off = _model(cfg_off)
    assert cfg_off.dipole_species_pairs == "off"           # normalized
    assert not isinstance(model_off.c_pair, tf.Variable)
    cfg_on = _cfg(dipole_species_pairs=True)               # legacy bool
    model_on = _model(cfg_on)                              # same seed → same W
    assert cfg_on.dipole_species_pairs == "symmetric"      # normalized
    atoms = _bulk_frame()
    assert np.allclose(_mu(model_off, atoms, cfg_off),
                       _mu(model_on, atoms, cfg_on), atol=1e-6)
    n_pairs = cfg_on.num_types * (cfg_on.num_types + 1) // 2
    opt_off, opt_on = SNES(model_off), SNES(model_on)
    assert opt_off.n_c == 0
    assert opt_on.n_c == int(cfg_on.dipole_channels) * n_pairs
    assert opt_on.dim == opt_off.dim + opt_on.n_c


# ── 18t: config gates ───────────────────────────────────────────────────────

def test_dipole_channels_lt_1_raises_at_init():
    cfg = _cfg(dipole_channels=0)
    with pytest.raises(ValueError, match="dipole_channels"):
        _model(cfg)


def test_dipole_rc_exceeding_rcut_raises_at_init():
    cfg = _cfg(dipole_rc=5.0)                            # > rcut_hard = 3.5
    with pytest.raises(ValueError, match="dipole_rc"):
        _model(cfg)


def test_dipole_rc_inherits_rcut_hard():
    cfg = _cfg(dipole_rc=None)                           # default: inherit
    _model(cfg)
    assert cfg.dipole_rc == cfg.rcut_hard


def test_charge_aux_weight_raises_at_init():
    cfg = _cfg()
    setattr(cfg, "charge_aux_weight", 0.5)
    with pytest.raises(ValueError, match="charge_aux_weight"):
        _model(cfg)


def test_small_box_raises_at_data_assembly():
    cfg = _cfg()                                         # rc + rcut = 6.5 Å
    at = Atoms("OHH",
               positions=[[1.0, 1.0, 1.0], [1.96, 1.0, 1.0], [0.76, 1.59, 1.0]],
               cell=6.0 * np.eye(3), pbc=True)           # L = 6.0 < 6.5
    with pytest.raises(ValueError, match="box height"):
        _neighbor_dipole_guards([at], [np.zeros(3)], cfg)


# ── screened power-law radial ───────────────────────────────────────────────

def _fcub(r, rc):
    x = r / rc
    return 1.0 - 3.0 * x ** 2 + 2.0 * x ** 3


def test_screened_identity_user_formula():
    # Independent reference implements the USER'S ORIGINAL formula
    # s·r̂·|r|^ζ·f_cub (unit vector times power) — NOT the code's
    # r^(ζ−1)·d rearrangement — so a rearrangement bug cannot self-verify.
    # Basic form: C = 1, c ≡ 1 frozen, ζ = −2.
    cfg, model, atoms, data = _bulk_setup("screened")
    s = _site_scalars_np(model, data)[:, 0]                    # [A]
    mu_model = _mu(model, atoms, cfg)
    rc, zeta = float(cfg.dipole_rc), float(cfg.dipole_zeta)
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)  # d = r_i − r_j
    r = np.linalg.norm(d, axis=1)
    rhat = d / r[:, None]
    mu_ref = np.sum(s[i, None] * rhat
                    * (r ** zeta * _fcub(r, rc))[:, None], axis=0)
    assert np.allclose(mu_model, mu_ref,
                       atol=1e-4 * max(np.linalg.norm(mu_model), 1.0))


def test_screened_smoothness_at_rc():
    # Screened cubic f_cub = (1−x)²(1+2x): f(r_c) = f′(r_c) = 0 but
    # f″(r_c) = 6/r_c² ≠ 0 — C¹ contact only. Default (None) cutoff for
    # screened resolves to "cubic".
    cfg = _cfg(rcut_hard=4.5, rcut_soft=4.0, dipole_rc=3.0,
               dipole_radial="screened", dipole_channels=1, dipole_zeta=-2.0)
    _assert_smooth_at_rc(cfg, "cubic")


def test_screened_no_nan_padded_batch():
    # Padded batch at ζ = −2: the self-pairs-only pair stream feeds the
    # in-graph V builder real r = 0 entries, and padding adds more — the
    # input guard in _radial must keep μ finite (0**(ζ−1) = inf without it).
    cfg, model, _, _ = _bulk_setup("screened")
    small = Atoms("OHH",
                  positions=[[10.0, 10.0, 10.0], [10.96, 10.0, 10.0],
                             [9.76, 10.59, 10.0]],
                  cell=20.0 * np.eye(3), pbc=True)
    small.info["dipole"] = np.zeros(3)
    data = prepare_eval_data([_bulk_frame(), small], cfg)   # padded to 96
    args = (data["descriptors"], data["grad_values"],
            data["pair_atom"], data["pair_gidx"], data["pair_struct"],
            data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
            model.W0, model.b0, model.W1, model.b1)
    assert np.all(np.isfinite(
        model.predict_batch(*args, W_atom=data["V_atom"]).numpy()))
    assert np.all(np.isfinite(
        model.predict_batch(*args, W_atom=None).numpy()))


# ── ordered species-weight table Ω ──────────────────────────────────────────

def test_ordered_identity_asymmetric_omega():
    # Independent numpy ordered-pair sum: μ = Σ_(i,j) s_i·Ω[t_i,t_j]·R·d —
    # only the CENTER's scalar and the ORDERED index, no (s_i − s_j) form.
    cfg, model, atoms, data = _bulk_setup("ordered")
    T = cfg.num_types
    Om = model.c_pair.numpy().astype(np.float64)               # [C, T²]
    Om_sq = Om.reshape(-1, T, T)
    assert not np.allclose(Om_sq, np.swapaxes(Om_sq, 1, 2))    # asymmetric
    s = _site_scalars_np(model, data)                          # [A, C]
    mu_model = _mu(model, atoms, cfg)
    Zt = data["Z_int"][0].numpy()
    i, j, D = neighbor_list("ijD", atoms, float(cfg.dipole_rc))
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    r = np.linalg.norm(d, axis=1)
    R = _R_np(r, float(cfg.dipole_rc), int(cfg.dipole_channels))  # [P, C]
    cn = Om[:, Zt[i] * T + Zt[j]].T                               # [P, C]
    b = np.sum(cn * R * s[i], axis=1)
    mu_ref = np.sum(b[:, None] * d, axis=0)
    assert np.allclose(mu_model, mu_ref,
                       atol=1e-4 * max(np.linalg.norm(mu_model), 1.0))


def test_ordered_omega_one_collapses_to_off():
    # Ω ≡ 1 gathers to the same all-ones [C, T, T] table as "off" — the two
    # forwards are the SAME graph, so the collapse is bit-tight.
    cfg_off = _cfg()
    cfg_ord = _cfg(dipole_species_pairs="ordered")
    model_off = _model(cfg_off)
    model_ord = _model(cfg_ord)                     # same seed → same W; Ω = 1
    atoms = _bulk_frame()
    np.testing.assert_array_equal(_mu(model_off, atoms, cfg_off),
                                  _mu(model_ord, atoms, cfg_ord))


def test_ordered_snes_tail_size():
    from SNES import SNES
    cfg = _cfg(dipole_species_pairs="ordered")
    opt = SNES(_model(cfg))
    assert opt.n_c == int(cfg.dipole_channels) * cfg.num_types ** 2


def test_save_load_roundtrip_ordered_omega(tmp_path):
    # The silent-failure path from the gate inventory: a miskeyed save or
    # load-assign would drop the asymmetric Ω and μ would change silently.
    import model_io
    cfg = _cfg(dipole_species_pairs="ordered")
    model = _model(cfg)
    rng = np.random.default_rng(5)
    model.c_pair.assign(rng.uniform(0.3, 1.7, size=model.c_pair.shape)
                        .astype(np.float32))
    atoms = _bulk_frame()
    mu0 = _mu(model, atoms, cfg)
    p = str(tmp_path / "ordered.h5")
    model_io.save_model(model, cfg, path=p)
    loaded = model_io.load_model(p)
    assert loaded.cfg.dipole_species_pairs == "ordered"
    np.testing.assert_array_equal(loaded.c_pair.numpy(), model.c_pair.numpy())
    assert np.allclose(mu0, _mu(loaded, atoms, loaded.cfg), atol=1e-6)


def test_save_load_roundtrip_screened_basic(tmp_path):
    # Basic screened form (C=1, "off"): saves/loads with no Ω machinery,
    # and dipole_radial/dipole_zeta persist through the config JSON.
    import model_io
    cfg, model, atoms, _ = _bulk_setup("screened")
    mu0 = _mu(model, atoms, cfg)
    p = str(tmp_path / "screened.h5")
    model_io.save_model(model, cfg, path=p)
    loaded = model_io.load_model(p)
    assert loaded.cfg.dipole_radial == "screened"
    assert float(loaded.cfg.dipole_zeta) == -2.0
    assert not isinstance(loaded.c_pair, tf.Variable)
    assert np.allclose(mu0, _mu(loaded, atoms, loaded.cfg), atol=1e-6)


# ── config gates (screened / 3-value field) ─────────────────────────────────

def test_screened_multichannel_raises_at_init():
    # RAISE, not force: data assembly reads cfg.dipole_channels before the
    # model exists, so forcing C=1 here would desync the precomputed V.
    cfg = _cfg(dipole_radial="screened", dipole_channels=4)
    with pytest.raises(ValueError, match="single-channel"):
        _model(cfg)


def test_unknown_radial_raises_at_init():
    cfg = _cfg(dipole_radial="banana")
    with pytest.raises(ValueError, match="dipole_radial"):
        _model(cfg)


def test_numpy_twin_unknown_names_raise():
    # _neighbor_dipole_V runs at data assembly, BEFORE model-init validation
    # — an else-fallthrough there would silently compute the wrong radial.
    at = _bulk_frame()
    t = np.zeros(len(at), dtype=int)
    with pytest.raises(ValueError, match="dipole_radial"):
        _neighbor_dipole_V(at, t, 3.0, 4, 2, radial="banana")
    with pytest.raises(ValueError, match="dipole_cutoff"):
        _neighbor_dipole_V(at, t, 3.0, 4, 2, cutoff="banana")


def test_unknown_species_pairs_raises_at_init():
    cfg = _cfg(dipole_species_pairs="banana")
    with pytest.raises(ValueError, match="dipole_species_pairs"):
        _model(cfg)


def test_legacy_bool_normalizes_at_config_load():
    # model_io leg of the normalization: an old saved JSON carrying a bool
    # must come out of the load path as the 3-value string.
    from model_io import _apply_legacy_field_defaults
    cfg_t = _cfg(dipole_species_pairs=True)
    _apply_legacy_field_defaults(cfg_t, ["dipole_species_pairs"])
    assert cfg_t.dipole_species_pairs == "symmetric"
    cfg_f = _cfg(dipole_species_pairs=False)
    _apply_legacy_field_defaults(cfg_f, ["dipole_species_pairs"])
    assert cfg_f.dipole_species_pairs == "off"


# ── crossable cutoff function (dipole_cutoff) ───────────────────────────────

def test_cheb_cubic_identity():
    # NEW cross combo chebyshev+cubic: independent numpy ordered-pair sum
    # μ = Σ_(i,j) s_i^n·T_n(2r/rc − 1)·f_cub(r)·d with chebvander (not the
    # model's TF recurrence); c ≡ 1 ("off"), C = 4.
    cfg = _cfg(dipole_cutoff="cubic")                       # chebyshev C=4
    model = _model(cfg)
    atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    s = _site_scalars_np(model, data)                       # [A, C]
    mu_model = _mu(model, atoms, cfg)
    rc = float(cfg.dipole_rc)
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    r = np.linalg.norm(d, axis=1)
    Tn = np.polynomial.chebyshev.chebvander(
        2.0 * r / rc - 1.0, int(cfg.dipole_channels) - 1)   # [P, C]
    R = Tn * _fcub(r, rc)[:, None]
    b = np.sum(R * s[i], axis=1)                            # [P]
    mu_ref = np.sum(b[:, None] * d, axis=0)
    assert np.allclose(mu_model, mu_ref,
                       atol=1e-4 * max(np.linalg.norm(mu_model), 1.0))


def test_warped_identity():
    # chebyshev_warped, default (None→cosine2) cutoff: independent numpy
    # ordered-pair sum with chebvander at an INDEPENDENTLY-computed
    # x_w = 2(r/rc − 1)² − 1 (not the model's TF recurrence); c ≡ 1, C = 4.
    cfg = _cfg(dipole_radial="chebyshev_warped")
    model = _model(cfg)
    atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    s = _site_scalars_np(model, data)                       # [A, C]
    mu_model = _mu(model, atoms, cfg)
    rc = float(cfg.dipole_rc)
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    r = np.linalg.norm(d, axis=1)
    xw = 2.0 * (r / rc - 1.0) ** 2 - 1.0
    Tn = np.polynomial.chebyshev.chebvander(
        xw, int(cfg.dipole_channels) - 1)                   # [P, C]
    R = Tn * _fc(r, rc)[:, None]
    b = np.sum(R * s[i], axis=1)                            # [P]
    mu_ref = np.sum(b[:, None] * d, axis=0)
    assert np.allclose(mu_model, mu_ref,
                       atol=1e-4 * max(np.linalg.norm(mu_model), 1.0))


def test_gaussian_identity():
    # gaussian radial at σ-scale 0.5, default (None→cosine2) cutoff:
    # explicit exp formula with μ_n/σ RE-DERIVED here (μ_n = rc·(n+1)/(C+1),
    # σ = scale·rc/(C+1)) — NOT read from the model; c ≡ 1, C = 4.
    cfg = _cfg(dipole_radial="gaussian", dipole_gauss_sigma_scale=0.5)
    model = _model(cfg)
    atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    s = _site_scalars_np(model, data)                       # [A, C]
    mu_model = _mu(model, atoms, cfg)
    rc, C = float(cfg.dipole_rc), int(cfg.dipole_channels)
    mu_n = rc * (np.arange(C) + 1.0) / (C + 1)              # interior points
    sigma = 0.5 * rc / (C + 1)                              # scale × spacing
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    r = np.linalg.norm(d, axis=1)
    R = (np.exp(-(r[:, None] - mu_n) ** 2 / (2.0 * sigma ** 2))
         * _fc(r, rc)[:, None])                             # [P, C]
    b = np.sum(R * s[i], axis=1)                            # [P]
    mu_ref = np.sum(b[:, None] * d, axis=0)
    assert np.allclose(mu_model, mu_ref,
                       atol=1e-4 * max(np.linalg.norm(mu_model), 1.0))


def test_screened_cosine2_identity():
    # NEW cross combo screened+cosine2: reference is the user's ORIGINAL
    # s·r̂·|r|^ζ·f form (unit vector times power) with f = cosine².
    cfg = _cfg(dipole_cutoff="cosine2", **_MODE_KW["screened"])
    model = _model(cfg)
    atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    s = _site_scalars_np(model, data)[:, 0]                 # [A]
    mu_model = _mu(model, atoms, cfg)
    rc, zeta = float(cfg.dipole_rc), float(cfg.dipole_zeta)
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    r = np.linalg.norm(d, axis=1)
    rhat = d / r[:, None]
    mu_ref = np.sum(s[i, None] * rhat
                    * (r ** zeta * _fc(r, rc))[:, None], axis=0)
    assert np.allclose(mu_model, mu_ref,
                       atol=1e-4 * max(np.linalg.norm(mu_model), 1.0))


def test_dipole_cutoff_none_resolution():
    # None resolves at model init to the radial family's traditional form —
    # existing configs/saves keep identical behavior.
    from TNEPconfig import resolved_dipole_cutoff
    cfg_c = _cfg()                                          # chebyshev
    assert _model(cfg_c)._dipole_cutoff == "cosine2"
    assert resolved_dipole_cutoff(cfg_c) == "cosine2"
    cfg_s = _cfg(**_MODE_KW["screened"])
    assert _model(cfg_s)._dipole_cutoff == "cubic"
    assert resolved_dipole_cutoff(cfg_s) == "cubic"
    for rad in ("chebyshev_warped", "gaussian"):        # new radials: cosine2
        cfg_n = _cfg(dipole_radial=rad)
        assert _model(cfg_n)._dipole_cutoff == "cosine2"
        assert resolved_dipole_cutoff(cfg_n) == "cosine2"


def test_unknown_dipole_cutoff_raises_at_init():
    cfg = _cfg(dipole_cutoff="gaussian")
    with pytest.raises(ValueError, match="dipole_cutoff"):
        _model(cfg)


def test_save_load_roundtrip_explicit_cutoff(tmp_path):
    # An explicit non-default cutoff must persist through the config JSON
    # and leave μ unchanged after load.
    import model_io
    cfg = _cfg(dipole_cutoff="cubic")                       # chebyshev+cubic
    model = _model(cfg)
    atoms = _bulk_frame()
    mu0 = _mu(model, atoms, cfg)
    p = str(tmp_path / "cutoff.h5")
    model_io.save_model(model, cfg, path=p)
    loaded = model_io.load_model(p)
    assert loaded.cfg.dipole_cutoff == "cubic"
    assert loaded._dipole_cutoff == "cubic"
    assert np.allclose(mu0, _mu(loaded, atoms, loaded.cfg), atol=1e-6)


def test_save_load_roundtrip_gaussian_tanh3(tmp_path):
    # New cross combo gaussian+tanh3 with a non-default σ-scale: all three
    # fields must persist through the config JSON and leave μ unchanged.
    import model_io
    cfg = _cfg(dipole_radial="gaussian", dipole_cutoff="tanh3",
               dipole_gauss_sigma_scale=0.5)
    model = _model(cfg)
    atoms = _bulk_frame()
    mu0 = _mu(model, atoms, cfg)
    p = str(tmp_path / "gauss_tanh3.h5")
    model_io.save_model(model, cfg, path=p)
    loaded = model_io.load_model(p)
    assert loaded.cfg.dipole_radial == "gaussian"
    assert loaded.cfg.dipole_cutoff == "tanh3"
    assert float(loaded.cfg.dipole_gauss_sigma_scale) == 0.5
    assert np.allclose(mu0, _mu(loaded, atoms, loaded.cfg), atol=1e-6)
