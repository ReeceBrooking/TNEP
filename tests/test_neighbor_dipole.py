"""Neighbour-basis readout geometry (data.py): the staged V the GAP fit reads,
its radial/cutoff smoothness, the pair search and the data-assembly guards.
Checked through the ANN forward because that is the cheapest linear consumer of V."""
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
from descriptor_layout import compute_dim_q
from data import prepare_eval_data, _neighbor_dipole_guards, _neighbor_pairs

WATER_XYZ = os.path.join(ROOT, "datasets", "train_waterbulk.xyz")


def _cfg(**kw):
    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.dipole_readout = "neighbor"
    cfg.rcut_hard, cfg.rcut_soft = 3.5, 3.0
    cfg.num_neurons, cfg.activation, cfg.seed = 8, "tanh", 7
    cfg.descriptor_mixing = False
    cfg.types, cfg.num_types, cfg.type_map = [8, 1], 2, {8: 0, 1: 1}
    # Pin every readout/ANN-input field so working-tree experiment defaults never leak in.
    cfg.dipole_channels, cfg.dipole_species_pairs = 4, "off"
    cfg.dipole_radial, cfg.dipole_zeta, cfg.dipole_cutoff = "chebyshev", -2.0, None
    cfg.dipole_gauss_sigma_scale = 1.0
    cfg.l_max = cfg.alpha_max = 4
    cfg.radial_basis_size, cfg.radial_basis_species_sum, cfg.radial_basis_per_l = None, False, False
    cfg.radial_basis_norm, cfg.q_shift, cfg.descriptor_scale, cfg.q_scale = "l2", None, "off", None
    cfg.optimizer, cfg.dipole_head, cfg.nep_descriptor = "snes", "vector", "off"
    cfg.sum_delta, cfg.sum_delta_learn = [1.0, 1.0, 1.0], False
    cfg.lambda_1 = cfg.lambda_2 = 0.0005
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
    """μ (or Voigt α) through predict_batch with the staged V in the W_atom slot."""
    data = prepare_eval_data([atoms], cfg)
    return model.predict_batch(
        data["descriptors"], data["grad_values"],
        data["pair_atom"], data["pair_gidx"], data["pair_struct"],
        data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
        model.W0, model.b0, model.W1, model.b1,
        W_atom=data.get("V_atom"))[0].numpy()


@functools.lru_cache(maxsize=2)
def _bulk_setup(target_mode=1):
    cfg = _cfg(target_mode=target_mode, dipole_species_pairs="symmetric")
    model = _model(cfg)
    rng = np.random.default_rng(3)
    model.c_pair.assign(rng.uniform(0.3, 1.7, size=model.c_pair.shape).astype(np.float32))
    atoms = _bulk_frame()
    return cfg, model, atoms, prepare_eval_data([atoms], cfg)


def _site_scalars_np(model, data):
    """s_i^n in numpy float64 (tanh ANN, b1 excluded by design)."""
    desc = data["descriptors"][0].numpy().astype(np.float64)
    Z = data["Z_int"][0].numpy()
    W0 = model.W0.numpy().astype(np.float64)
    b0 = model.b0.numpy().astype(np.float64)
    W1 = model._masked_W1(model.W1).numpy().astype(np.float64)
    h = np.tanh(np.einsum('aq,aqh->ah', desc, W0[Z]) + b0[Z])
    return np.einsum('ah,ahn->an', h, W1[Z])


def _R_np(r, rc, C):
    """R_n(r) = T_n(2r/rc − 1)·cosine²(r) by the Chebyshev recurrence → [P, C]."""
    x = 2.0 * r / rc - 1.0
    Tn = np.empty((len(r), C))
    Tn[:, 0] = 1.0
    if C > 1:
        Tn[:, 1] = x
    for n in range(2, C):
        Tn[:, n] = 2.0 * x * Tn[:, n - 1] - Tn[:, n - 2]
    return Tn * (((1.0 + np.cos(np.pi * r / rc)) / 2.0) ** 2)[:, None]


def _pair_index(T):
    """Unordered-pair index table [T, T] matching TNEP._pair_idx."""
    M = np.zeros((T, T), dtype=int)
    p = 0
    for t in range(T):
        for t2 in range(t, T):
            M[t, t2] = M[t2, t] = p
            p += 1
    return M


def _rotation():
    a, b, c = 0.7, -0.4, 1.1
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return Rz @ Ry @ Rx


# ── rank 1: the staged V equals the pair-difference double sum ─────────────

def test_pair_difference_sum_matches_model():
    cfg, model, atoms, data = _bulk_setup()
    s = _site_scalars_np(model, data)
    Zt = data["Z_int"][0].numpy()
    c = model.c_pair.numpy().astype(np.float64)
    pidx = _pair_index(cfg.num_types)
    i, j, D = neighbor_list("ijD", atoms, float(cfg.rcut_hard))
    keep = i != j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)        # d = r_i − r_j
    R = _R_np(np.linalg.norm(d, axis=1), float(cfg.rcut_hard), int(cfg.dipole_channels))
    cn = c[:, pidx[Zt[i], Zt[j]]].T
    b = np.sum(cn * R * (s[i] - s[j]), axis=1)
    mu_pair = 0.5 * np.sum(b[:, None] * d, axis=0)
    mu_model = _mu(model, atoms, cfg)
    assert np.allclose(mu_model, mu_pair, atol=1e-4 * max(np.linalg.norm(mu_model), 1.0), rtol=1e-4)


def test_gauge_shift_cancels():
    """s → s + g moves μ by g·Σ V ≡ 0 with symmetric c: Σ_{a,t'} V[a,:,n,t'] vanishes."""
    _, _, _, data = _bulk_setup()
    V = data["V_atom"][0].numpy()                        # [A, 3, C, T]
    assert np.all(np.abs(V.sum(axis=(0, 3))) < 1e-4 * np.abs(V).sum(axis=(0, 3)))


# ── smoothness at r_c, keyed on the cutoff ─────────────────────────────────

@pytest.mark.parametrize("cutoff", ["cosine2", "cosine"])
def test_radial_weights_smooth_at_rc(cutoff):
    """R_n·f_c (the weights the GAP fit and the staged V use): cosine² is a C³
    contact (value, R′, R″ vanish), cosine is C¹ (value and R′ only)."""
    from data import _radial_weights
    rc, h = 3.0, 0.01
    r = np.arange(rc - 0.3, rc + 1e-9, h)
    R = _radial_weights(r, rc, 10, "chebyshev_warped", -2.0, cutoff, 1.0)      # [P, C]
    assert np.allclose(R[-1], 0.0, atol=1e-12)
    d1 = np.abs(np.diff(R, axis=0)).max(1) / h                                  # slope per step
    assert d1[-1] < 0.15 * d1.max()
    if cutoff == "cosine2":
        d2 = np.abs(np.diff(R, 2, axis=0)).max(1) / h ** 2
        assert d2[-1] < 0.15 * d2.max()


# ── data-assembly guards and the pair search ───────────────────────────────

def test_branch_inconsistency_warning(capsys):
    cfg = _cfg(convert_dipole_to_eangstrom=True)
    frame = _bulk_frame()
    dataset = [frame] * 5
    L = float(frame.cell[0, 0])
    tame = [np.array([0.1 * k, 0.0, 0.0]) for k in range(5)]
    _neighbor_dipole_guards(dataset, tame, cfg)
    assert "branch-inconsistent" not in capsys.readouterr().out
    offset = list(tame)
    offset[2] = offset[2] + np.array([L, 0.0, 0.0])
    _neighbor_dipole_guards(dataset, offset, cfg)
    assert "branch-inconsistent" in capsys.readouterr().out


@pytest.mark.parametrize("L,needle", [
    (6.0, "half the smallest box height"),   # rc = 3.5 ≥ L/2: MIC paths
    (3.0, "the smallest box height"),        # rc = 3.5 ≥ L: lattice sum
    (100.0, None),
])
def test_small_box_warns_at_data_assembly(capsys, L, needle):
    cfg = _cfg()
    at = Atoms("OHH", positions=[[1.0, 1.0, 1.0], [1.96, 1.0, 1.0], [0.76, 1.59, 1.0]],
               cell=L * np.eye(3), pbc=True)
    _neighbor_dipole_guards([at], [np.zeros(3)], cfg)
    out = capsys.readouterr().out
    assert ("WARNING: readout r_c" not in out) if needle is None else (needle in out)


def test_neighbor_pairs_fast_path_matches_ase():
    """Direct MIC search == ASE neighbor_list: periodic water (L > 2rc) and a vacuum-boxed CHO molecule."""
    frames = [_bulk_frame()]
    frames.append([a for a in read(os.path.join(ROOT, "datasets", "train.xyz"), index=":10")
                   if set(a.numbers) <= {1, 6, 8}][0])
    for at in frames:
        rc = 3.0
        i, j, D = _neighbor_pairs(at, rc)
        i2, j2, D2 = neighbor_list("ijD", at, rc)
        assert len(i) == len(i2) > 0
        key = lambda i_, j_, D_: sorted(zip(i_.tolist(), j_.tolist(), np.round(D_, 6).tolist()))
        assert key(i, j, D) == key(i2, j2, D2)
        assert np.all(np.diag(at.cell.array)[at.pbc] >= 2 * rc)       # the fast path really was taken


# ── rank 2: α = Σ_{i<j} R_n [c⁰ (s_i+s_j) |d|²I + c¹ (u_i+u_j)(d⊗d − |d|²I/3)] ──

def _voigt(d):
    x, y, z = d.T
    return np.stack([x * x, y * y, z * z, x * y, y * z, z * x], axis=1)


def _voigt_to_mat(v):
    xx, yy, zz, xy, yz, zx = v
    return np.array([[xx, xy, zx], [xy, yy, yz], [zx, yz, zz]])


def test_pol_symmetric_pair_sum_matches_model():
    cfg, model, atoms, data = _bulk_setup(target_mode=2)
    C, rc = cfg.dipole_channels, cfg.rcut_hard
    assert model._bond_rank == 2 and model._kernel_channels == 2 * C
    assert tuple(data["V_atom"].shape) == (1, len(atoms), 6, 2 * C, 2)
    s_all = _site_scalars_np(model, data)                       # [A, 2C]: iso | traceless
    Z = data["Z_int"][0].numpy()
    c = model.c_pair.numpy().astype(np.float64)[:, _pair_index(2)]  # [2C, T, T]
    i, j, D = neighbor_list("ijD", atoms, rc)
    keep = i < j
    i, j, d = i[keep], j[keep], -D[keep].astype(np.float64)
    R = _R_np(np.linalg.norm(d, axis=1), rc, C)
    r2 = (d ** 2).sum(1)
    iso = np.stack([r2, r2, r2, 0 * r2, 0 * r2, 0 * r2], 1)
    a_ref = np.zeros(6)
    for g, geom in enumerate((iso, _voigt(d) - iso / 3.0)):
        s, cg = s_all[:, g * C:(g + 1) * C], c[g * C:(g + 1) * C]
        w = np.einsum("pn,pn->p", R, cg[:, Z[i], Z[j]].T * (s[i] + s[j]))
        a_ref += (w[:, None] * geom).sum(0)
    a_model = _mu(model, atoms, cfg)
    assert np.any(a_ref != 0.0)
    assert np.allclose(a_model, a_ref, rtol=1e-5, atol=1e-5 * np.abs(a_ref).max())


def test_pol_rotation_covariance_and_translation_invariance():
    cfg, model, atoms, _ = _bulk_setup(target_mode=2)
    a0 = _voigt_to_mat(_mu(model, atoms, cfg))
    R = _rotation()
    at2 = atoms.copy()
    at2.set_cell(atoms.cell[:] @ R.T, scale_atoms=False)
    at2.positions = atoms.positions @ R.T
    assert np.allclose(_voigt_to_mat(_mu(model, at2, cfg)), R @ a0 @ R.T, atol=2e-3 * np.abs(a0).max())
    at3 = atoms.copy()
    at3.positions = atoms.positions + np.array([1.3, -0.7, 2.1])
    assert np.allclose(_mu(model, at3, cfg), _mu(model, atoms, cfg), atol=1e-4 * np.abs(a0).max())


# ── Ewald field channel (experimental option): the GAP geometry helper builds the same staged V ──

def test_field_channel_gap_matches_staged_and_in_graph_refuses():
    import gap_dipole
    from test_gap_dipole import _cfg as _gcfg
    spce = {8: -0.8476, 1: 0.4238}
    cfg = _cfg(dipole_field=True, dipole_field_charges=spce, dipole_field_scale=None,
               dipole_species_pairs="symmetric")
    atoms = read(WATER_XYZ, index="2")
    data = prepare_eval_data([atoms], cfg)                     # pins cfg.dipole_field_scale
    model = _model(cfg)
    with pytest.raises(ValueError, match="dipole_field"):
        model.predict_batch(data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
                            data["pair_struct"], data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
                            model.W0, model.b0, model.W1, model.b1, W_atom=None)
    gcfg = _gcfg(dipole_channels=cfg.dipole_channels, dipole_radial="chebyshev",
                 dipole_cutoff=None, dipole_species_pairs="symmetric", dipole_field=True,
                 dipole_field_charges=spce, dipole_field_scale=cfg.dipole_field_scale)
    gcfg.types, gcfg.num_types = [8, 1], 2
    ti = np.array([0 if z == 8 else 1 for z in atoms.numbers])
    v = gap_dipole._v_atoms(atoms, ti, gcfg)                   # [N, 3, (C+1)·T]
    V = data["V_atom"].numpy()[0]
    assert v.shape[2] == (cfg.dipole_channels + 1) * 2
    assert np.allclose(v, V.reshape(96, 3, -1), rtol=1e-5, atol=1e-5 * np.abs(V).max())
    gcfg.dipole_field_charges = {"8": -0.8476, "1": 0.4238}    # str keys (JSON round trip)
    assert np.allclose(gap_dipole._v_atoms(atoms, ti, gcfg), v)
