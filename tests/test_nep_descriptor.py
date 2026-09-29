"""Tests for the multi-descriptor plan (multi_descriptor_plan.md).

Task 1: radial_basis_norm="none" on the FIXED GPU SOAP builder.
Task 3: GPUMD-exact NEP basis / raw builder (nep_descriptor.py).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DescriptorBuilder import make_descriptor_builder
import nep_descriptor as nd
from test_neighbor_dipole import _cfg as _base_cfg, _model, _bulk_frame


def _cfg(**kw):
    # GPU (TF) builder specifically; kwargs still override.
    return _base_cfg(**{"descriptor_mode": 1, "descriptor_precision": "float64", **kw})


def _descriptors(cfg, atoms):
    """[A, Q] numpy descriptors of one frame (forward-only builder path)."""
    desc = make_descriptor_builder(cfg).build_descriptors(
        [atoms], calc_gradients=False)[0][0]
    return np.asarray(desc, dtype=np.float64)


def _self_gradients(cfg, atoms):
    """[A, 3, Q] numpy self gradients ∂q_i/∂r_i (with-grad builder path)."""
    _, grads, _ = make_descriptor_builder(cfg).build_descriptors_self_only([atoms])
    return np.stack([np.asarray(g, dtype=np.float64)[0] for g in grads[0]], axis=0)


def _fd_self_gradient(cfg, atoms, atom, h):
    """[3, Q] central finite difference of q_atom w.r.t. r_atom."""
    out = []
    for c in range(3):
        qs = []
        for s in (+1.0, -1.0):
            at = atoms.copy()
            pos = at.get_positions()
            pos[atom, c] += s * h
            at.set_positions(pos)
            qs.append(_descriptors(cfg, at)[atom])
        out.append((qs[0] - qs[1]) / (2.0 * h))
    return np.stack(out, axis=0)


# ── Task 1: radial_basis_norm="none" on the fixed builder ──────────────────

def test_fixed_soap_norm_none_matches_unnormalised():
    cfg_l2 = _cfg(radial_basis_norm="l2")
    cfg_no = _cfg(radial_basis_norm="none")
    atoms = _bulk_frame()
    q_l2 = _descriptors(cfg_l2, atoms)
    q_no = _descriptors(cfg_no, atoms)                                    # [A, Q]
    n = np.linalg.norm(q_no, axis=1, keepdims=True)
    np.testing.assert_allclose(q_no / n, q_l2, rtol=1e-5, atol=1e-6)      # same direction
    # Magnitude kept and varies with the environment. The raw spectrum is
    # ~4e5 per atom and dominated by the central-atom term, so the spread
    # is modest (O/H ratio ~1.2 on this frame).
    assert np.all(n[:, 0] > 0.0) and n.max() / n.min() > 1.1


@pytest.mark.parametrize("tile", [0, 8000])   # single-shot and pair-tiled gradient paths
def test_fixed_soap_norm_none_gradient_is_unprojected(tile):
    # "tnep" readout so the builder returns descriptor gradients.
    cfg = _cfg(radial_basis_norm="none", dipole_readout="tnep",
               descriptor_pair_tile_size=tile)
    atoms = _bulk_frame()
    g = _self_gradients(cfg, atoms)                                       # [A, 3, Q]
    g_fd = _fd_self_gradient(cfg, atoms, atom=0, h=1e-3)                  # [3, Q]
    # Descriptors leave the builder as float32 at |q| ~ 3e5, so the FD
    # noise floor is eps32·|q|/(2h) ~ 10 on a gradient of scale ~1e5:
    # tolerate 1e-3 of the tensor scale (measured residual ~7e-5 of it).
    np.testing.assert_allclose(g[0], g_fd, rtol=1e-3,
                               atol=1e-3 * np.abs(g_fd).max())
    # Projection identity: applying (I − q̂q̂ᵀ)/‖q‖ to the raw gradient must
    # reproduce the "l2" builder's gradient exactly.
    g_l2 = _self_gradients(_cfg(dipole_readout="tnep",
                                descriptor_pair_tile_size=tile), atoms)
    q = _descriptors(cfg, atoms)[0]; n = np.linalg.norm(q); qh = q / n
    g_proj = (g[0] - (g[0] @ qh)[:, None] * qh) / n
    np.testing.assert_allclose(g_proj, g_l2[0], rtol=1e-5,
                               atol=1e-6 * np.abs(g_l2[0]).max())


def test_fixed_soap_rejects_sqrt():
    with pytest.raises(ValueError, match="radial_basis_norm"):
        _model(_cfg(radial_basis_norm="sqrt"))
    with pytest.raises(ValueError, match="radial_basis_norm"):
        make_descriptor_builder(_cfg(radial_basis_norm="sqrt"))
    # The quippy builder (descriptor_mode=0) ignores radial_basis_norm.
    with pytest.raises(ValueError, match="radial_basis_norm"):
        _model(_cfg(radial_basis_norm="none", descriptor_mode=0))


# ── Task 2: descriptor_scale="minmax" (GPUMD's q_scaler) ───────────────────

def test_q_scale_minmax_is_gpumd_scaler():
    from DescriptorBuilderGPU import compute_q_scale
    q = np.random.default_rng(0).uniform(-3, 5, (50, 7)); mask = np.ones(50)
    q[:, 6] = 2.0                                                       # dead channel
    cfg = _cfg(descriptor_scale="minmax")
    m, mu = compute_q_scale(q, mask, cfg)
    np.testing.assert_allclose(m[:6], 1.0 / (q[:, :6].max(0) - q[:, :6].min(0)), rtol=1e-6)   # nep.cu:347
    assert m[6] == 1.0
    assert np.all(mu == 0.0)                                            # no centring
    assert m.dtype == np.float32 and mu.dtype == np.float32


def test_minmax_threads_into_model_fold():
    from DescriptorBuilderGPU import compute_q_scale
    cfg = _cfg(descriptor_scale="minmax")
    atoms = _bulk_frame()
    q_np = _descriptors(cfg, atoms)
    mask_np = np.ones(q_np.shape[0])
    m, mu = compute_q_scale(q_np, mask_np, cfg)
    cfg.q_scale = [float(v) for v in m]
    cfg.q_shift = [float(v) for v in mu]
    model = _model(cfg)
    np.testing.assert_allclose(model._q_scale.numpy(), m)
    assert np.all(model._q_shift_m.numpy() == 0.0)


# ── Task 3: GPUMD-exact NEP basis and raw builder (nep_descriptor.py) ──────

NEP_TXT = "/home/reece/Work/GPUMD Testing/dipole_bulk_water/nep.txt"


def _types_int(atoms, species):
    """0-based type per atom in the given species order (nep.txt header order)."""
    return np.array([species.index(s) for s in atoms.get_chemical_symbols()])


def test_chebyshev_basis_matches_recursion_reference():
    r = np.linspace(0.1, 7.9, 200); rc, K = 8.0, 6
    f = nd.fn_all(r, rc, K)
    x = 2 * (r / rc - 1) ** 2 - 1; fc_ = 0.5 * (1 + np.cos(np.pi * r / rc))
    ref = np.stack([fc_] + [0.5 * (np.polynomial.chebyshev.Chebyshev.basis(k)(x) + 1) * fc_
                            for k in range(1, K + 1)], 1)
    np.testing.assert_allclose(f, ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(nd.fc(r, rc), fc_, rtol=1e-12)
    assert np.all(nd.fn_all(np.array([8.0, 9.0]), rc, K) == 0.0)
    assert f.dtype == np.float64


def test_real_harmonics_rotation_invariance_of_q():
    rng = np.random.default_rng(1)
    v = rng.standard_normal((30, 3)); w = rng.uniform(0.2, 1, 30)

    def q_of(v):
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        s = (nd.real_harmonics(u[:, 0], u[:, 1], u[:, 2]) * w[:, None]).sum(0)
        return nd.find_q(s[None, None], True, True)[0, 0]

    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    q = q_of(v)
    assert q.shape == (6,) and np.all(q != 0.0)
    np.testing.assert_allclose(q_of(v @ R.T), q, rtol=1e-6, atol=1e-8)
    # The L=1..4 entries are rotation invariant even without the 4b/5b forms.
    assert nd.find_q(np.zeros((2, 3, 24)), False, False).shape == (2, 3, 4)
    assert nd.find_q(np.zeros((2, 3, 24)), True, False).shape == (2, 3, 5)


def test_real_harmonics_L1_is_unit_vector():
    # Z_COEFFICIENT_1 = {{0,1},{1,0}} (nep_utilities.cuh:48): n1 = 0 picks
    # Z[0][1]·z¹ → s[0] = z; n1 = 1 picks Z[1][0]·z⁰ = 1 times (x + iy)¹ →
    # s[1] = x, s[2] = y.  So the L=1 block is (z, x, y).
    assert nd.Z_COEFFICIENT[1] == [[0.0, 1.0], [1.0, 0.0]]
    rng = np.random.default_rng(2)
    u = rng.standard_normal((10, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = nd.real_harmonics(u[:, 0], u[:, 1], u[:, 2])
    assert P.shape == (10, 24)
    np.testing.assert_allclose(P[:, :3], u[:, [2, 0, 1]], rtol=1e-12)
    # L=2, n1=0: Z_2[0] = {-1, 0, 3} → 3z² − 1 (Legendre P2 up to scale).
    np.testing.assert_allclose(P[:, 3], 3 * u[:, 2] ** 2 - 1, rtol=1e-12)


def test_nep_raw_width_and_shape():
    cfg = _cfg(nep_descriptor="concat")
    atoms = _bulk_frame()
    raw = nd.nep_raw_numpy(atoms, _types_int(atoms, ["O", "H"]), cfg)
    T, Kr, Ka = 2, cfg.nep_basis_size_radial, cfg.nep_basis_size_angular
    W = T * (Kr + 1) + T * (Ka + 1) * 24
    assert raw.shape == (len(atoms), W) and raw.dtype == np.float32
    assert nd.nep_raw_width(cfg) == W
    assert nd.nep_dim(cfg) == (cfg.nep_n_max_radial + 1) + (cfg.nep_n_max_angular + 1) * (4 + 1 + 0)
    sl = nd.nep_slices(cfg)
    assert sl["radial"] == slice(0, cfg.nep_n_max_radial + 1)
    assert sl["angular"] == slice(cfg.nep_n_max_radial + 1, nd.nep_dim(cfg))
    cfg5 = _cfg(nep_descriptor="concat", nep_five_body=True)
    assert nd.nep_dim(cfg5) == nd.nep_dim(cfg) + cfg.nep_n_max_angular + 1
    # Every neighbour contributes fc ≥ 0 to raw_r[:, :, 0]: no atom is empty.
    assert np.all(raw[:, [0, Kr + 1]].sum(1) > 0)
    # Sub-blocks partition the angular slice, in GPUMD l-major order.
    for c, subs in [(cfg, ["L1", "L2", "L3", "L4", "four_body"]),
                    (cfg5, ["L1", "L2", "L3", "L4", "four_body", "five_body"])]:
        sl = nd.nep_slices(c)
        assert set(sl) == {"radial", "angular", *subs}
        cols = [k for s in subs for k in range(nd.nep_dim(c))[sl[s]]]
        assert cols == list(range(nd.nep_dim(c))[sl["angular"]])
        assert all(sl[s].stop - sl[s].start == c.nep_n_max_angular + 1 for s in subs)


@pytest.mark.parametrize("kw", [dict(nep_l_max=3),
                                dict(nep_five_body=True, nep_four_body=False),
                                dict(nep_rc_angular=0.0)])
def test_validate_nep_cfg_rejects_bad_fields(kw):
    nd.validate_nep_cfg(_cfg())
    with pytest.raises(ValueError):
        nd.validate_nep_cfg(_cfg(**kw))


def test_nep_contract_permutation_invariance():
    cfg = _cfg(nep_descriptor="concat", nep_five_body=True)
    atoms = _bulk_frame(); species = ["O", "H"]
    rng = np.random.default_rng(3)
    T, nr, Kr = 2, cfg.nep_n_max_radial + 1, cfg.nep_basis_size_radial + 1
    na, Ka = cfg.nep_n_max_angular + 1, cfg.nep_basis_size_angular + 1
    c_r = rng.uniform(-1, 1, (T, T, nr, Kr)); c_a = rng.uniform(-1, 1, (T, T, na, Ka))

    def q_of(at):
        t = _types_int(at, species)
        return nd.nep_contract_numpy(nd.nep_raw_numpy(at, t, cfg), t, c_r, c_a, cfg)

    q = q_of(atoms)
    assert q.shape == (len(atoms), nd.nep_dim(cfg))
    perm = rng.permutation(len(atoms))
    np.testing.assert_allclose(q_of(atoms[perm]), q[perm], rtol=1e-5, atol=1e-6)
    # Rigid rotation (positions and cell) leaves every component unchanged.
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    rot = atoms.copy(); rot.set_cell(atoms.cell[:] @ R.T); rot.set_positions(atoms.positions @ R.T)
    np.testing.assert_allclose(q_of(rot), q, rtol=1e-5, atol=1e-6)


def test_nep_descriptor_matches_calorine_on_users_nep_txt():
    """Oracle: calorine.nep.get_descriptors == NEP_CPU find_descriptor_small_box,
    which writes g_descriptor = q · q_scaler (nep.cpp, `if (calculating_descriptor)`):
    the SCALED descriptors. Run under the calorine venv:
      CUDA_VISIBLE_DEVICES='' <venv>/bin/python -m pytest tests/test_nep_descriptor.py -k calorine
    """
    calorine = pytest.importorskip("calorine")
    from calorine.nep import get_descriptors
    model = nd.read_nep_txt(NEP_TXT)
    assert model["species"] == ["O", "H"]
    cfg = _cfg(nep_descriptor="concat", **{k: v for k, v in model.items() if k.startswith("nep_")})
    assert (cfg.nep_rc_radial, cfg.nep_rc_angular, cfg.nep_basis_size_radial,
            cfg.nep_four_body, cfg.nep_five_body) == (6.0, 4.0, 10, True, True)
    atoms = _bulk_frame()
    t = _types_int(atoms, model["species"])
    raw = nd.nep_raw_numpy(atoms, t, cfg)
    q = nd.nep_contract_numpy(raw, t, model["c_r"], model["c_a"], cfg)
    theirs = get_descriptors(atoms, NEP_TXT)
    assert theirs.shape == q.shape == (len(atoms), nd.nep_dim(cfg)) == (96, 49)
    ours = q * model["q_scaler"]
    np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-5 * np.abs(theirs).max())
    # And the raw q does NOT match — calorine really returns the scaled form.
    assert not np.allclose(q, theirs, rtol=1e-3, atol=1e-3 * np.abs(theirs).max())
