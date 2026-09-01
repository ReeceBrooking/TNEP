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
                  pad_and_stack, _neighbor_dipole_guards)

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


@functools.lru_cache(maxsize=1)
def _bulk_setup():
    """(cfg, model, atoms, padded-data) for the 96-atom water frame — shared."""
    cfg = _cfg()
    model = _model(cfg)
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

def test_rigid_translation_invariance():
    cfg, model, atoms, _ = _bulk_setup()
    mu0 = _mu(model, atoms, cfg)
    at2 = atoms.copy()
    at2.positions = at2.positions + np.array([1.234, -2.517, 0.771])
    mu1 = _mu(model, at2, cfg)
    assert np.allclose(mu0, mu1, atol=2e-3)


def test_rotation_covariance():
    cfg, model, atoms, _ = _bulk_setup()
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


def test_single_atom_lattice_translation_invariance():
    cfg, model, atoms, _ = _bulk_setup()
    mu0 = _mu(model, atoms, cfg)
    at2 = atoms.copy()
    at2.positions[5] += atoms.cell[:][0]         # +one cell vector, atom 5
    mu1 = _mu(model, at2, cfg)
    assert np.allclose(mu0, mu1, atol=2e-3)


# ── 13c: four-path forward consistency (the drift alarm) ────────────────────

def test_four_path_consistency():
    cfg, model, atoms, _ = _bulk_setup()
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

def test_smoothness_at_rc():
    # dipole_rc strictly inside rcut so the descriptor is smooth at r_c.
    cfg = _cfg(rcut_hard=4.5, rcut_soft=4.0, dipole_rc=3.0)
    model = _model(cfg)
    rc = float(cfg.dipole_rc)
    h = 0.03
    rs = np.arange(2.70, 3.30 + 1e-9, h)
    mus = []
    for r in rs:
        at = Atoms("OH", positions=[[10.0, 10.0, 10.0], [10.0 + r, 10.0, 10.0]],
                   cell=20.0 * np.eye(3), pbc=True)
        at.info["dipole"] = np.zeros(3)
        mus.append(_mu(model, at, cfg))
    mus = np.array(mus)                                   # [K, 3]

    # Beyond r_c: exactly zero (V has no pairs).
    assert np.allclose(mus[rs >= rc], 0.0, atol=1e-12)

    d1 = (mus[2:] - mus[:-2]) / (2 * h)
    d2 = (mus[2:] - 2 * mus[1:-1] + mus[:-2]) / h ** 2
    n_mu = np.linalg.norm(mus, axis=1)
    n_d1 = np.linalg.norm(d1, axis=1)
    n_d2 = np.linalg.norm(d2, axis=1)
    r_mid = rs[1:-1]

    # Squared cosine ⇒ μ ∝ (rc − r)^4 near rc: value, μ', μ'' all vanish
    # continuously into the identically-zero region — no jump at r_c.
    k_last = np.max(np.where(rs < rc))                    # last point inside
    assert n_mu[k_last] < 1e-3 * n_mu.max()
    near = np.abs(r_mid - rc) < 1.5 * h                   # FD points straddling rc
    assert n_d1[near].max() < 0.05 * n_d1.max()
    assert n_d2[near].max() < 0.10 * n_d2.max()


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

def test_gauge_shift_cancels():
    cfg, model, atoms, data = _bulk_setup()
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
    # Default dipole_species_pairs=False: no c parameters in the SNES tail,
    # and the forward equals the learnable model at its c=1 init.
    from SNES import SNES
    cfg_off = _cfg()
    assert cfg_off.dipole_species_pairs is False           # class default
    model_off = _model(cfg_off)
    assert not isinstance(model_off.c_pair, tf.Variable)
    cfg_on = _cfg(dipole_species_pairs=True)
    model_on = _model(cfg_on)                              # same seed → same W
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
