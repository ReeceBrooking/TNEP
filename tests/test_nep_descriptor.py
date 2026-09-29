"""Tests for the multi-descriptor plan (multi_descriptor_plan.md).

Task 1: radial_basis_norm="none" on the FIXED GPU SOAP builder.
Task 3: GPUMD-exact NEP basis / raw builder (nep_descriptor.py).
Task 4: in-model contraction nep_from_raw, c variables, _ann_inputs, dim bookkeeping.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DescriptorBuilder import make_descriptor_builder
import nep_descriptor as nd
from data import assign_type_indices, prepare_eval_data
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


# ── Task 4: nep_from_raw, c variables, _ann_inputs, dim bookkeeping ────────

def _nep_inputs(cfg, atoms):
    """(raw [A, W_raw], types [A]) of one frame in cfg.types order."""
    types = np.asarray(assign_type_indices([atoms], cfg.types)[0])
    return nd.nep_raw_numpy(atoms, types, cfg), types


def test_nep_from_raw_matches_numpy_contraction():
    cfg = _cfg(nep_descriptor="concat", nep_five_body=True); m = _model(cfg)
    atoms = _bulk_frame()
    raw, types = _nep_inputs(cfg, atoms)
    R, Zt = tf.constant(raw[None]), tf.constant(types[None])
    q_tf = m.nep_from_raw(R, Zt, m.c_r, m.c_a)[0].numpy()
    q_np = nd.nep_contract_numpy(raw, types, m.c_r.numpy(), m.c_a.numpy(), cfg)
    assert q_tf.shape == (len(atoms), nd.nep_dim(cfg)) and q_tf.dtype == np.float32
    np.testing.assert_allclose(q_tf, q_np, rtol=1e-5, atol=1e-6 * np.abs(q_np).max())
    # c defaults to the model's own variables.
    np.testing.assert_array_equal(m.nep_from_raw(R, Zt)[0].numpy(), q_tf)
    # Per-candidate c [Cc, T, T, n, k] on candidate-independent raw →
    # [Cc, ..., A, dim]; doubling c scales each block by its degree in c.
    c_r2 = tf.stack([m.c_r, 2.0 * m.c_r]); c_a2 = tf.stack([m.c_a, 2.0 * m.c_a])
    q2 = m.nep_from_raw(R, Zt, c_r2, c_a2).numpy()
    assert q2.shape == (2, 1, len(atoms), nd.nep_dim(cfg))
    np.testing.assert_allclose(q2[0, 0], q_tf, rtol=1e-5, atol=1e-6 * np.abs(q_tf).max())
    sl = nd.nep_slices(cfg)
    for k, deg in [("radial", 1), ("L1", 2), ("L2", 2), ("L3", 2), ("L4", 2),
                   ("four_body", 3), ("five_body", 4)]:
        ref = 2.0 ** deg * q_tf[:, sl[k]]
        np.testing.assert_allclose(q2[1, 0][:, sl[k]], ref, rtol=1e-5,
                                   atol=1e-6 * np.abs(ref).max())


def test_nep_descriptor_ordering_is_gpumd():
    cfg = _cfg(nep_descriptor="concat"); m = _model(cfg); atoms = _bulk_frame()
    raw, types = _nep_inputs(cfg, atoms)
    R, Zt = tf.constant(raw[None]), tf.constant(types[None])
    q = m.nep_from_raw(R, Zt)[0].numpy()
    sl = nd.nep_slices(cfg)
    assert np.all(np.abs(q).max(0) > 0.0)                       # every column live
    # [radial ‖ angular]: zero c_a kills exactly the angular/4b columns, zero c_r the radial.
    q_no_a = m.nep_from_raw(R, Zt, m.c_r, tf.zeros_like(m.c_a))[0].numpy()
    np.testing.assert_array_equal(q_no_a[:, sl["radial"]], q[:, sl["radial"]])
    assert np.all(q_no_a[:, sl["angular"]] == 0.0)
    q_no_r = m.nep_from_raw(R, Zt, tf.zeros_like(m.c_r), m.c_a)[0].numpy()
    assert np.all(q_no_r[:, sl["radial"]] == 0.0)
    np.testing.assert_array_equal(q_no_r[:, sl["angular"]], q[:, sl["angular"]])
    # l-major (ln = l·(n_max_a+1) + n): keeping only c_a[..., n0, :] leaves
    # column n0 of EVERY L / 4-body block and zeroes the rest.
    n0, na = 2, cfg.nep_n_max_angular + 1
    c_a = np.zeros(m.c_a.shape, np.float32); c_a[:, :, n0] = m.c_a.numpy()[:, :, n0]
    q_n0 = m.nep_from_raw(R, Zt, m.c_r, tf.constant(c_a))[0].numpy()
    others = [n for n in range(na) if n != n0]
    for k in ("L1", "L2", "L3", "L4", "four_body"):
        blk = q_n0[:, sl[k]]
        assert blk.shape[1] == na and np.all(blk[:, others] == 0.0)
        np.testing.assert_allclose(blk[:, n0], q[:, sl[k]][:, n0], rtol=1e-6)


def test_nep_c_init_uniform_pm1_and_shapes():
    cfg = _cfg(nep_descriptor="concat"); m = _model(cfg)
    T, nr, Kr = 2, cfg.nep_n_max_radial, cfg.nep_basis_size_radial
    na, Ka = cfg.nep_n_max_angular, cfg.nep_basis_size_angular
    assert tuple(m.c_r.shape) == (T, T, nr + 1, Kr + 1)
    assert tuple(m.c_a.shape) == (T, T, na + 1, Ka + 1)
    for c in (m.c_r, m.c_a):
        assert isinstance(c, tf.Variable) and not c.trainable and c.dtype == tf.float32
        v = c.numpy()
        assert v.min() > -1.0 and v.max() < 1.0 and v.std() > 0.4   # uniform(−1, 1): σ 0.577
    m2 = _model(_cfg(nep_descriptor="concat"))                     # same seed → same c
    np.testing.assert_array_equal(m2.c_r.numpy(), m.c_r.numpy())
    np.testing.assert_array_equal(m2.c_a.numpy(), m.c_a.numpy())
    m3 = _model(_cfg(nep_descriptor="concat", seed=8))
    assert not np.array_equal(m3.c_r.numpy(), m.c_r.numpy())
    m_off = _model(_cfg())
    assert m_off.c_r is None and m_off.c_a is None


def test_ann_inputs_modes():
    from DescriptorBuilderGPU import compute_dim_q
    atoms = _bulk_frame()
    cfg0 = _cfg(); dim_soap = cfg0.dim_q
    desc = tf.constant(_descriptors(cfg0, atoms)[None].astype(np.float32))   # [1, A, dim_soap]
    for mode in ("off", "concat", "replace"):
        cfg = _cfg(nep_descriptor=mode); m = _model(cfg)
        raw, types = _nep_inputs(cfg, atoms)
        R, Zt = tf.constant(raw[None]), tf.constant(types[None])
        x = m._ann_inputs(desc, R, Zt)
        assert int(x.shape[-1]) == cfg.dim_q == compute_dim_q(cfg)
        if mode == "off":
            assert x is desc and cfg.dim_q == dim_soap
            assert m._ann_inputs(desc, None, Zt) is desc
            continue
        q_nep = m.nep_from_raw(R, Zt).numpy()
        if mode == "concat":
            assert cfg.dim_q == dim_soap + nd.nep_dim(cfg)
            np.testing.assert_array_equal(x[..., :dim_soap].numpy(), desc.numpy())
            np.testing.assert_array_equal(x[..., dim_soap:].numpy(), q_nep)
        else:
            assert cfg.dim_q == nd.nep_dim(cfg)
            np.testing.assert_array_equal(x.numpy(), q_nep)
        with pytest.raises(ValueError, match="nep_raw"):
            m._ann_inputs(desc, None, Zt)
    with pytest.raises(ValueError, match="descriptor_mixing"):
        _model(_cfg(nep_descriptor="replace", descriptor_mixing=True))
    with pytest.raises(ValueError, match="descriptor_preprocess_contract"):
        _model(_cfg(nep_descriptor="replace", descriptor_preprocess_contract="angular"))
    with pytest.raises(NotImplementedError):
        _model(_cfg(nep_descriptor="sum"))
    with pytest.raises(ValueError, match="nep_descriptor"):
        _model(_cfg(nep_descriptor="bogus"))


def test_nep_rejects_descriptor_scale_until_task7():
    with pytest.raises(NotImplementedError, match="descriptor_scale"):
        _model(_cfg(nep_descriptor="concat", descriptor_scale="minmax"))


def test_forward_sites_take_nep_raw():
    """predict_batch / predict_batch_candidates consume nep_raw through
    _ann_inputs; the candidate path with per-candidate c reproduces the batched one."""
    cfg = _cfg(nep_descriptor="concat"); m = _model(cfg); atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    raw, _ = _nep_inputs(cfg, atoms)
    R = tf.constant(raw[None])
    args = (data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
            data["pair_struct"], data["positions"], data["Z_int"], data["boxes"],
            data["atom_mask"], m.W0, m.b0, m.W1, m.b1)
    with pytest.raises(ValueError, match="nep_raw"):
        m.predict_batch(*args, W_atom=data.get("V_atom"))
    mu = m.predict_batch(*args, W_atom=data.get("V_atom"), nep_raw=R).numpy()
    assert mu.shape == (1, 3) and np.all(np.isfinite(mu))
    # nep_raw changes the prediction (the NEP block reaches the ANN).
    mu0 = m.predict_batch(*args, W_atom=data.get("V_atom"), nep_raw=tf.zeros_like(R)).numpy()
    assert not np.allclose(mu, mu0)
    C = 2
    tile = lambda v: tf.tile(v[tf.newaxis], [C] + [1] * len(v.shape))
    mu_c = m.predict_batch_candidates(
        data["descriptors"], data.get("V_atom"), data["Z_int"], data["atom_mask"],
        tile(m.W0), tile(m.b0), tile(m.W1), tile(m.b1),
        nep_raw=R, c_r=tile(m.c_r), c_a=tile(m.c_a)).numpy()
    assert mu_c.shape == (C, 1, 3)
    np.testing.assert_allclose(mu_c[0], mu, rtol=1e-5, atol=1e-6 * np.abs(mu).max())
    np.testing.assert_allclose(mu_c[1], mu, rtol=1e-5, atol=1e-6 * np.abs(mu).max())
    # Shared c (None → model variables) on the candidate path gives the same μ.
    mu_s = m.predict_batch_candidates(
        data["descriptors"], data.get("V_atom"), data["Z_int"], data["atom_mask"],
        tile(m.W0), tile(m.b0), tile(m.W1), tile(m.b1), nep_raw=R).numpy()
    np.testing.assert_allclose(mu_s[1], mu, rtol=1e-5, atol=1e-6 * np.abs(mu).max())


def test_nep_concat_with_learned_soap_basis():
    """soap_from_raw reshapes to the SOAP width, not the (wider) ANN width."""
    cfg = _cfg(nep_descriptor="concat", radial_basis_size=4); m = _model(cfg)
    atoms = _bulk_frame()
    data = prepare_eval_data([atoms], cfg)
    raw, _ = _nep_inputs(cfg, atoms)
    R = tf.constant(raw[None])
    x = m._ann_inputs(m.soap_from_raw(data["descriptors"], data["Z_int"]), R, data["Z_int"])
    assert int(x.shape[-1]) == cfg.dim_q == m._soap_dim + nd.nep_dim(cfg)

    def mu_of(M, c_r, c_a):
        return m.predict_batch(
            data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
            data["pair_struct"], data["positions"], data["Z_int"], data["boxes"],
            data["atom_mask"], m.W0, m.b0, m.W1, m.b1, W_atom=data.get("V_atom"),
            M_radial=M, nep_raw=R, c_r=c_r, c_a=c_a).numpy()

    rng = np.random.default_rng(5)
    M2 = tf.constant(np.stack([m.M_radial.numpy(),
                               rng.standard_normal(m.M_radial.shape).astype(np.float32)]))
    c_r2 = tf.stack([m.c_r, 0.5 * m.c_r]); c_a2 = tf.stack([m.c_a, -m.c_a])
    mu = np.stack([mu_of(M2[i], c_r2[i], c_a2[i]) for i in range(2)])   # [2, 1, 3]
    assert mu.shape == (2, 1, 3) and np.all(np.isfinite(mu)) and not np.allclose(mu[0], mu[1])
    C = 2
    tile = lambda v: tf.tile(v[tf.newaxis], [C] + [1] * len(v.shape))
    mu_c = m.predict_batch_candidates(
        data["descriptors"], data.get("V_atom"), data["Z_int"], data["atom_mask"],
        tile(m.W0), tile(m.b0), tile(m.W1), tile(m.b1),
        M_radial=M2, nep_raw=R, c_r=c_r2, c_a=c_a2).numpy()
    np.testing.assert_allclose(mu_c, mu, rtol=1e-5, atol=1e-5 * np.abs(mu).max())
    # Per-candidate M with SHARED c (None → model variables): _ann_inputs
    # broadcasts the rank-3 NEP block onto the rank-4 SOAP descriptors.
    x = m._ann_inputs(m.soap_from_raw(data["descriptors"], data["Z_int"], M2), R, data["Z_int"])
    assert tuple(x.shape) == (2, 1, len(atoms), cfg.dim_q)
    mu_s = np.stack([mu_of(M2[i], None, None) for i in range(2)])
    mu_sc = m.predict_batch_candidates(
        data["descriptors"], data.get("V_atom"), data["Z_int"], data["atom_mask"],
        tile(m.W0), tile(m.b0), tile(m.W1), tile(m.b1), M_radial=M2, nep_raw=R).numpy()
    np.testing.assert_allclose(mu_sc, mu_s, rtol=1e-5, atol=1e-5 * np.abs(mu_s).max())


def test_nep_rejects_descriptor_gradient_readouts():
    with pytest.raises(NotImplementedError, match="neighbor/bilinear"):
        _model(_cfg(nep_descriptor="concat", dipole_readout="tnep"))
    with pytest.raises(NotImplementedError, match="neighbor/bilinear"):
        _model(_cfg(nep_descriptor="replace", target_mode=0))


def test_compute_dim_q_and_layout_per_mode():
    from DescriptorBuilderGPU import (compute_dim_q, descriptor_block_layout,
                                      q_scale_groups, descriptor_input_width)
    off = _cfg(); cc = _cfg(nep_descriptor="concat")
    rp = _cfg(nep_descriptor="replace", nep_five_body=True)
    ds, dn = compute_dim_q(off), nd.nep_dim(cc)
    assert compute_dim_q(cc) == ds + dn and compute_dim_q(rp) == nd.nep_dim(rp)
    assert compute_dim_q(_cfg(nep_descriptor="sum")) == ds
    # The builder's width is the SOAP width in every mode.
    assert descriptor_input_width(off) == descriptor_input_width(cc) == descriptor_input_width(rp) == ds

    def covers(cfg, n_groups):
        g = q_scale_groups(cfg)
        assert len(g) == n_groups
        assert sorted(np.concatenate(g).tolist()) == list(range(compute_dim_q(cfg)))
        return g

    L = off.l_max + 1
    g_off = covers(off, L); g_cc = covers(cc, L + 1 + 4 + 1); g_rp = covers(rp, 1 + 4 + 2)
    assert all(np.array_equal(a, b) for a, b in zip(g_off, g_cc[:L]))
    for g, k in zip(g_cc[L:], ["radial", "L1", "L2", "L3", "L4", "four_body"]):
        assert g.tolist() == [ds + i for i in range(dn)[nd.nep_slices(cc)[k]]]
    for g, k in zip(g_rp, ["radial", "L1", "L2", "L3", "L4", "four_body", "five_body"]):
        assert g.tolist() == list(range(nd.nep_dim(rp))[nd.nep_slices(rp)[k]])
    lo, lc, lr = (descriptor_block_layout(c) for c in (off, cc, rp))
    assert lc["pair_keys"] == lo["pair_keys"]
    assert all(np.array_equal(lc["pair_q_index"][k], lo["pair_q_index"][k]) for k in lo["pair_keys"])
    assert lo["dim_q"] == ds and lc["dim_q"] == ds + dn
    assert max(int(v.max()) for v in lc["pair_q_index"].values()) < ds   # NEP block outside the pairs
    assert lr["pair_keys"] == [] and lr["dim_q"] == nd.nep_dim(rp) and lr["max_block_size"] == 0
    # "replace" drops the SOAP output, learned-basis extras included.
    rp2 = _cfg(nep_descriptor="replace", radial_basis_size=4, radial_basis_two_body=True)
    assert compute_dim_q(rp2) == descriptor_block_layout(rp2)["dim_q"] == nd.nep_dim(rp2)
    covers(rp2, 1 + 4 + 1)
