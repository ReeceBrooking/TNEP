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
from test_q_scale import _predict_eff


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
        x = m._ann_inputs(desc, R, Zt)[0]
        assert int(x.shape[-1]) == cfg.dim_q == compute_dim_q(cfg)
        if mode == "off":
            assert x is desc and cfg.dim_q == dim_soap
            assert m._ann_inputs(desc, None, Zt)[0] is desc
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
    m = _model(_cfg(nep_descriptor="sum"))           # main ANN sees SOAP only
    raw, types = _nep_inputs(m.cfg, atoms)
    assert m._ann_inputs(desc, tf.constant(raw[None]), tf.constant(types[None]))[0] is desc
    with pytest.raises(ValueError, match="nep_raw"):
        m._ann_inputs(desc, None, tf.constant(types[None]))
    with pytest.raises(ValueError, match="nep_descriptor"):
        _model(_cfg(nep_descriptor="bogus"))


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
    x = m._ann_inputs(m.soap_from_raw(data["descriptors"], data["Z_int"]), R, data["Z_int"])[0]
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
    x = m._ann_inputs(m.soap_from_raw(data["descriptors"], data["Z_int"], M2), R, data["Z_int"])[0]
    assert tuple(x.shape) == (2, 1, len(atoms), cfg.dim_q)
    mu_s = np.stack([mu_of(M2[i], None, None) for i in range(2)])
    mu_sc = m.predict_batch_candidates(
        data["descriptors"], data.get("V_atom"), data["Z_int"], data["atom_mask"],
        tile(m.W0), tile(m.b0), tile(m.W1), tile(m.b1), M_radial=M2, nep_raw=R).numpy()
    np.testing.assert_allclose(mu_sc, mu_s, rtol=1e-5, atol=1e-5 * np.abs(mu_s).max())


def test_nep_rejects_descriptor_gradient_readouts():
    # Phase B lifts the guard for the tnep dipole readout only (Task 11).
    with pytest.raises(NotImplementedError, match="target_mode"):
        _model(_cfg(nep_descriptor="replace", target_mode=0))
    with pytest.raises(NotImplementedError, match="target_mode"):
        _model(_cfg(nep_descriptor="concat", target_mode=2))
    with pytest.raises(NotImplementedError, match="target_mode"):
        _model(_cfg(nep_descriptor="sum", dipole_readout="tnep", dipole_rij_power=2))


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


# ── Task 5: nep_raw data plumbing + every forward call site ────────────────

def _forward_args(m, d):
    return (d["descriptors"], d["grad_values"], d["pair_atom"], d["pair_gidx"],
            d["pair_struct"], d["positions"], d["Z_int"], d["boxes"], d["atom_mask"],
            m.W0, m.b0, m.W1, m.b1)


def test_nep_raw_rides_data_dict_and_padding():
    cfg = _cfg(nep_descriptor="concat"); atoms = _bulk_frame()
    d = prepare_eval_data([atoms, atoms], cfg)                   # two frames → [2, A, W_raw]
    assert tuple(d["nep_raw"].shape) == (2, len(atoms), nd.nep_raw_width(cfg))
    assert d["nep_raw"].dtype == tf.float32
    raw, _ = _nep_inputs(cfg, atoms)
    np.testing.assert_array_equal(d["nep_raw"][0].numpy(), raw)
    np.testing.assert_array_equal(d["nep_raw"][1].numpy(), raw)
    # Padding: a shorter frame is zero-filled beyond its atom count.
    small = atoms[:10]; small.info["dipole"] = np.zeros(3)
    d2 = prepare_eval_data([atoms, small], cfg)
    assert tuple(d2["nep_raw"].shape) == (2, len(atoms), nd.nep_raw_width(cfg))
    assert np.all(d2["nep_raw"][1, 10:].numpy() == 0.0)
    raw_s, _ = _nep_inputs(cfg, small)
    np.testing.assert_array_equal(d2["nep_raw"][1, :10].numpy(), raw_s)
    # Chunk slicing carries it (SMALL_KEYS).
    from data import slice_and_complete_chunk
    ch = slice_and_complete_chunk(d2, [1])
    np.testing.assert_array_equal(ch["nep_raw"].numpy(), d2["nep_raw"][1:2].numpy())
    assert "nep_raw" not in prepare_eval_data([atoms], _cfg(nep_descriptor="off"))


@pytest.mark.parametrize("mode", ["concat", "replace"])
@pytest.mark.parametrize("readout", ["neighbor", "bilinear"])
def test_nep_four_path_consistency(mode, readout):
    cfg = _cfg(nep_descriptor=mode, dipole_readout=readout)
    _check_four_path(cfg, _model(cfg))


def _check_four_path(cfg, m):
    """predict / predict_batch (V slot + in-graph COO) / predict_batch_candidates agree."""
    from data import assemble_data_dict, pad_and_stack
    atoms = _bulk_frame()
    types_int = assign_type_indices([atoms], cfg.types)
    builder = make_descriptor_builder(cfg)
    desc, grads, gidx = builder.build_descriptors([atoms])
    d = assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg)
    full = pad_and_stack(d, num_types=cfg.num_types, self_pairs_only=False)
    kw = dict(nbr_index=full.get("nbr_index"), nbr_G=full.get("nbr_G"),
              nep_raw=full["nep_raw"])
    args = _forward_args(m, full)
    mu_slot = m.predict_batch(*args, W_atom=full["V_atom"], **kw)[0].numpy()
    mu_ingraph = m.predict_batch(*args, W_atom=None, nep_raw=full["nep_raw"])[0].numpy()
    mu_cand = m.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        m.W0[tf.newaxis], m.b0[tf.newaxis], m.W1[tf.newaxis], m.b1[tf.newaxis],
        **kw)[0, 0].numpy()
    gi = gidx[0]; A = len(atoms); M = max(len(lst) for lst in gi)
    grad_index = np.zeros((A, M), dtype=np.int32); nmask = np.zeros((A, M), dtype=np.float32)
    for a, lst in enumerate(gi):
        grad_index[a, :len(lst)] = lst; nmask[a, :len(lst)] = 1.0
    mu_single = m.predict(
        full["descriptors"][0], tf.zeros([A, M, 3, cfg.dim_q]),
        tf.constant(grad_index), full["positions"][0], full["Z_int"][0],
        full["boxes"][0], full["atom_mask"][0], tf.constant(nmask),
        nep_raw=full["nep_raw"][0]).numpy()
    assert np.all(np.isfinite(mu_slot)) and np.abs(mu_slot).max() > 0
    atol = 1e-4 * max(np.linalg.norm(mu_slot), 1.0)
    assert np.allclose(mu_slot, mu_ingraph, atol=atol)
    assert np.allclose(mu_slot, mu_cand, atol=atol)
    assert np.allclose(mu_slot, mu_single, atol=atol)


def test_nep_snes_validate_and_candidates_smoke():
    cfg = _cfg(nep_descriptor="concat", pop_size=4, num_generations=2, batch_size=1,
               val_interval=1)
    m = _model(cfg); atoms = _bulk_frame()
    d = prepare_eval_data([atoms, atoms], cfg)
    snes = m.optimizer
    fit = snes.evaluate_population(tf.stack([snes.mu, snes.mu]), d).numpy()
    assert fit.shape == (2,) and np.all(np.isfinite(fit))
    assert np.isfinite(snes.validate(d, snes.mu)) and np.isfinite(snes.validate(d))
    missing = {k: v for k, v in d.items() if k != "nep_raw"}
    with pytest.raises(ValueError, match="nep_raw missing"):
        snes.validate(missing)
    with pytest.raises(ValueError, match="nep_raw missing"):
        snes.evaluate_population(tf.stack([snes.mu]), missing)
    # Minibatch fit path (struct_keys gather) + validate: two generations.
    history, _, _ = m.fit(d, d)
    assert np.all(np.isfinite(history["val_loss"]))


def test_nep_adam_step_smoke():
    cfg = _cfg(nep_descriptor="concat", optimizer="adam", adam_lr=1e-2,
               num_generations=2, val_interval=1, batch_chunk_size=1)
    m = _model(cfg); atoms = _bulk_frame()
    d = prepare_eval_data([atoms, atoms], cfg)
    c_r0 = m.c_r.numpy().copy()
    history, _, _ = m.fit(d, d)
    assert len(history["val_loss"]) == 2 and np.all(np.isfinite(history["val_loss"]))
    assert np.all(np.isfinite(history["train_loss"]))
    assert not np.array_equal(m.c_r.numpy(), c_r0)   # c is in Adam's tape


def test_score_takes_nep_raw():
    cfg = _cfg(nep_descriptor="replace"); m = _model(cfg); atoms = _bulk_frame()
    d = prepare_eval_data([atoms, atoms], cfg)
    metrics, preds = m.score(d)
    assert np.isfinite(float(metrics["rmse"])) and preds.shape == (2, 3)


def test_spectroscopy_rejects_nep():
    import spectroscopy
    cfg = _cfg(nep_descriptor="concat"); m = _model(cfg); atoms = _bulk_frame()
    types = assign_type_indices([atoms], cfg.types)
    with pytest.raises(NotImplementedError, match="trajectory inference"):
        spectroscopy.predict_trajectory_batch(m, make_descriptor_builder(cfg), [atoms], types)


# ── Task 6: c_r / c_a in the SNES tail (after M_radial), Adam, save/load ────

def _mu_nep(model, atoms, cfg):
    """μ through predict_batch with nep_raw (the eval path for a NEP model)."""
    d = prepare_eval_data([atoms], cfg)
    return model.predict_batch(*_forward_args(model, d), W_atom=d.get("V_atom"),
                               nbr_index=d.get("nbr_index"), nbr_G=d.get("nbr_G"),
                               nep_raw=d["nep_raw"])[0].numpy()


def test_nep_c_tail_roundtrip_snes():
    from SNES import SNES, _set_model_params
    from data import assemble_data_dict, pad_and_stack
    cfg = _cfg(nep_descriptor="concat", dipole_readout="bilinear", dipole_pair_rank=1)
    m = _model(cfg); opt = SNES(m)
    n_c_r = int(np.prod(m.c_r.shape)); n_c_a = int(np.prod(m.c_a.shape)); n_c = n_c_r + n_c_a
    assert opt.n_nep == n_c
    assert opt.dim == opt.n_anns_total + opt.n_U_pair + opt.n_preprocess + opt.n_radial + n_c
    named = opt._split_reconstructed(opt.reconstruct_params_tf(opt.mu))
    np.testing.assert_array_equal(named["c_r"].numpy(), m.c_r.numpy())   # μ-init carries the model's draw
    np.testing.assert_array_equal(named["c_a"].numpy(), m.c_a.numpy())
    tail = opt.mu.numpy()[-n_c:]
    np.testing.assert_array_equal(tail[:n_c_r], m.c_r.numpy().ravel())   # tail order: c_r then c_a
    np.testing.assert_array_equal(tail[n_c_r:], m.c_a.numpy().ravel())
    # Round trip: perturb μ → assign → predict_batch(c=None) == candidates(μ) with per-candidate c.
    atoms = _bulk_frame()
    types_int = assign_type_indices([atoms], cfg.types)
    desc, grads, gidx = make_descriptor_builder(cfg).build_descriptors([atoms])
    d = assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg)
    full = pad_and_stack(d, num_types=cfg.num_types, self_pairs_only=True)
    mu_vec = opt.mu + 0.05 * tf.random.stateless_normal(opt.mu.shape, [3, 4])
    _set_model_params(m, *opt.reconstruct_params_tf(mu_vec))
    n = opt._split_reconstructed(opt.reconstruct_params_tf(mu_vec))
    np.testing.assert_array_equal(m.c_r.numpy(), n["c_r"].numpy())
    assert not np.array_equal(m.c_r.numpy(), named["c_r"].numpy())        # assign actually moved c
    kw = dict(nbr_index=full["nbr_index"], nbr_G=full["nbr_G"], nep_raw=full["nep_raw"])
    mu_model = m.predict_batch(*_forward_args(m, full), W_atom=full["V_atom"], **kw)[0].numpy()
    nb = opt._split_reconstructed(opt.reconstruct_params_tf(mu_vec[tf.newaxis]))
    mu_cand = m.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        nb["W0"], nb["b0"], nb["W1"], nb["b1"], c_pair=nb["c_pair"],
        c_r=nb["c_r"], c_a=nb["c_a"], **kw)[0, 0].numpy()
    np.testing.assert_allclose(mu_cand, mu_model, rtol=1e-5, atol=1e-6)
    assert np.any(mu_cand != 0.0)
    # A different c in the candidate path must change μ (the kwarg is live).
    mu_other = m.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        nb["W0"], nb["b0"], nb["W1"], nb["b1"], c_pair=nb["c_pair"],
        c_r=2.0 * nb["c_r"], c_a=nb["c_a"], **kw)[0, 0].numpy()
    assert not np.allclose(mu_other, mu_model, atol=1e-6)
    # The SNES fit and validate call sites forward per-candidate c: doubling
    # the c tail of μ must change both the population fitness and val RMSE.
    d = prepare_eval_data([atoms, atoms], cfg)
    mu_c = opt.mu.numpy().copy(); mu_c[-n_c:] *= 2.0
    fit = opt.evaluate_population(tf.stack([opt.mu, tf.constant(mu_c)]), d).numpy()
    assert np.all(np.isfinite(fit)) and fit[0] != fit[1]
    assert opt.validate(d, tf.constant(mu_c)) != opt.validate(d, opt.mu)
    # off: no tail.
    assert SNES(_model(_cfg(nep_descriptor="off"))).n_nep == 0


def test_nep_c_per_type_labels():
    from SNES import SNES
    cfg = _cfg(nep_descriptor="concat"); m = _model(cfg); opt = SNES(m)
    T = cfg.num_types
    n_c_r = int(np.prod(m.c_r.shape)); n_c_a = int(np.prod(m.c_a.shape))
    tov = np.asarray(opt._type_of_variable)
    assert tov.shape == (opt.dim,)
    tail = tov[-(n_c_r + n_c_a):]
    np.testing.assert_array_equal(tail[:n_c_r], np.repeat(np.arange(T), n_c_r // T))
    np.testing.assert_array_equal(tail[n_c_r:], np.repeat(np.arange(T), n_c_a // T))


@pytest.mark.parametrize("types", [[8, 1], [8]])
def test_nep_c_excluded_from_regularisation(types):
    from SNES import SNES
    kw = dict(nep_descriptor="concat", lambda_1=0.1, lambda_2=0.1)
    if len(types) == 1:
        kw.update(types=[8], num_types=1, type_map={8: 0})
    cfg = _cfg(**kw); m = _model(cfg); opt = SNES(m)
    mu = opt.mu.numpy().copy()
    mu2 = mu.copy(); mu2[-opt.n_nep:] *= 10.0
    assert opt.compute_regularization(mu) == opt.compute_regularization(mu2)
    l1, l2, _ = opt.compute_regularization(mu)
    assert l1 > 0 and l2 > 0
    np.testing.assert_allclose(opt.compute_regularization_tf(tf.constant(mu[None])).numpy(),
                               opt.compute_regularization_tf(tf.constant(mu2[None])).numpy())


def test_save_load_roundtrip_nep(tmp_path):
    import h5py, model_io
    cfg = _cfg(nep_descriptor="concat", nep_five_body=True, nep_n_max_radial=3,
               nep_rc_radial=4.5, q_scale="minmax")
    m = _model(cfg); atoms = _bulk_frame()
    m.W1.assign(tf.random.stateless_normal(m.W1.shape, [5, 6]))
    mu0 = _mu_nep(m, atoms, cfg)
    p = str(tmp_path / "nep.h5")
    model_io.save_model(m, cfg, path=p)
    loaded = model_io.load_model(p)
    for k in ("nep_descriptor", "nep_five_body", "nep_n_max_radial", "nep_rc_radial",
              "nep_rc_angular", "nep_basis_size_radial", "q_scale"):
        assert getattr(loaded.cfg, k) == getattr(cfg, k), k
    np.testing.assert_array_equal(loaded.c_r.numpy(), m.c_r.numpy())
    np.testing.assert_array_equal(loaded.c_a.numpy(), m.c_a.numpy())
    np.testing.assert_allclose(_mu_nep(loaded, atoms, loaded.cfg), mu0, atol=1e-6)
    # A save without c_r must not silently reset c to its init.
    with h5py.File(p, "a") as f:
        del f["weights"]["c_r"]
    with pytest.raises(ValueError, match="c_r"):
        model_io.load_model(p)


def test_adam_trains_c():
    from adam import _trainable
    cfg = _cfg(nep_descriptor="concat", optimizer="adam", adam_lr=1e-2,
               num_generations=2, val_interval=1, batch_chunk_size=1)
    m = _model(cfg); atoms = _bulk_frame()
    d = prepare_eval_data([atoms, atoms], cfg)
    assert set(_trainable(m)) >= {"c_r", "c_a"}
    c_r0 = m.c_r.numpy().copy(); c_a0 = m.c_a.numpy().copy()
    history, final, best = m.fit(d, d)
    assert np.all(np.isfinite(history["train_loss"]))
    for mdl in (final, best):
        assert np.all(np.isfinite(mdl.c_r.numpy())) and np.all(np.isfinite(mdl.c_a.numpy()))
    assert not np.array_equal(final.c_r.numpy(), c_r0)
    assert not np.array_equal(final.c_a.numpy(), c_a0)
    assert "c_r" not in _trainable(_model(_cfg(nep_descriptor="off")))


# ── Task 7: descriptor_scale groups / gen-0 scaler input / mixing exclusion ──

def test_q_scale_groups_cover_nep_block_once():
    from DescriptorBuilderGPU import compute_dim_q, q_scale_groups
    n_soap = len(q_scale_groups(_cfg(nep_descriptor="off", descriptor_scale="block")))
    for mode, extra in (("concat", n_soap), ("replace", 0)):
        cfg = _cfg(nep_descriptor=mode, descriptor_scale="block")
        groups = q_scale_groups(cfg)
        assert sorted(np.concatenate(groups).tolist()) == list(range(compute_dim_q(cfg)))
        assert len(groups) == extra + 1 + 4 + 1          # radial, L1..L4, four_body


def test_minmax_scaler_uses_c_equal_one():
    """The scaler input is the assembled ANN vector: c ≡ 1 for minmax (GPUMD
    fitness.cu:165-175), the model's real gen-0 c for block/channel."""
    from TNEP import gen0_ann_inputs
    atoms = _bulk_frame()
    cfg = _cfg(nep_descriptor="concat", descriptor_scale="minmax")
    d = prepare_eval_data([atoms, atoms], cfg)
    q_mm = gen0_ann_inputs(cfg, d)
    m = _model(_cfg(nep_descriptor="concat"))
    ones = m._ann_inputs(d["descriptors"], d["nep_raw"], d["Z_int"],
                         c_r=tf.ones_like(m.c_r), c_a=tf.ones_like(m.c_a))[0].numpy()
    real = m._ann_inputs(d["descriptors"], d["nep_raw"], d["Z_int"])[0].numpy()
    assert q_mm.shape == (2, len(atoms), cfg.dim_q)
    np.testing.assert_array_equal(q_mm, ones)
    q_bl = gen0_ann_inputs(_cfg(nep_descriptor="concat", descriptor_scale="block"), d)
    np.testing.assert_array_equal(q_bl, real)
    assert not np.allclose(q_mm, q_bl)
    # "off" NEP with the fixed basis: the builder output itself.
    cfg0 = _cfg(descriptor_scale="block"); d0 = prepare_eval_data([atoms], cfg0)
    np.testing.assert_array_equal(gen0_ann_inputs(cfg0, d0), d0["descriptors"].numpy())


@pytest.mark.parametrize("scale", ["block", "channel", "minmax"])
@pytest.mark.parametrize("mode", ["concat", "replace"])
def test_scaled_nep_model_is_gen0_reparameterisation(mode, scale):
    from DescriptorBuilderGPU import compute_q_scale
    from TNEP import gen0_ann_inputs
    atoms = _bulk_frame(); small = atoms[:12]; small.info["dipole"] = np.zeros(3)
    cfg_off = _cfg(nep_descriptor=mode)
    cfg_on = _cfg(nep_descriptor=mode, descriptor_scale=scale)
    d = prepare_eval_data([atoms, small], cfg_off)
    m_np, mu_np = compute_q_scale(gen0_ann_inputs(cfg_on, d), d["atom_mask"].numpy(), cfg_on)
    cfg_on.q_scale, cfg_on.q_shift = [float(v) for v in m_np], [float(v) for v in mu_np]
    assert len(cfg_on.q_scale) == cfg_on.dim_q and (mu_np == 0).all() == (scale == "minmax")
    m_off, m_on = _model(cfg_off), _model(cfg_on)
    np.testing.assert_array_equal(m_on.c_r.numpy(), m_off.c_r.numpy())
    np.testing.assert_allclose(m_on.W0.numpy() * m_np[None, :, None], m_off.W0.numpy(), rtol=1e-6)
    np.testing.assert_allclose(_predict_eff(m_on, d), _predict_eff(m_off, d), rtol=1e-4, atol=1e-6)
    # Any scaled model == unscaled model with W0' = m∘W0, b0' = b0 − (m∘μ)·W0.
    rng = np.random.default_rng(1)
    W = rng.normal(size=m_on.W0.shape).astype(np.float32)
    B = rng.normal(size=m_on.b0.shape).astype(np.float32)
    m_on.W0.assign(W); m_on.b0.assign(B)
    m_off.W0.assign(W * m_np[None, :, None]); m_off.b0.assign(B - np.einsum('q,tqh->th', m_np * mu_np, W))
    np.testing.assert_allclose(_predict_eff(m_on, d), _predict_eff(m_off, d), rtol=1e-4, atol=1e-6)


def test_nep_block_excluded_from_mixing():
    cfg = _cfg(nep_descriptor="concat", descriptor_mixing=True); m = _model(cfg)
    ds = _cfg().dim_q
    m.U_pair.assign(0.3 * np.random.default_rng(2).normal(size=m.U_pair.shape).astype(np.float32))
    W0, W0_eff = m.W0.numpy(), m._W0_eff(m.W0).numpy()
    np.testing.assert_array_equal(W0_eff[:, ds:], W0[:, ds:])
    assert not np.allclose(W0_eff[:, :ds], W0[:, :ds])


# ── Task 8: nep_descriptor="sum" — ANN_soap + ANN_2b + ANN_3b summed into the head ──

# Deliberately NOT imported from nep_descriptor: an independent pin on the
# SNES / Adam / save-format order of the expert tensors.
EXPERTS = ("W0_r", "b0_r", "W1_r", "W0_a", "b0_a", "W1_a")


def _sum_cfg(**kw):
    return _cfg(**{"nep_descriptor": "sum", "nep_num_neurons": 6, "dipole_pair_rank": 2, **kw})


def _randomise_experts(m, seed=11):
    """Live experts: the gen-0 W1_r = W1_a = 0 would hide every expert bug."""
    for i, k in enumerate(EXPERTS):
        v = getattr(m, k)
        v.assign(0.5 * tf.random.stateless_normal(v.shape, [seed, i]))


def test_sum_mode_gen0_identity():
    """A fresh sum-mode model equals the SOAP-only model with the same seed
    (W1_r = W1_a = 0); the SNES μ-init does the same."""
    from SNES import SNES
    atoms = _bulk_frame()
    cfg_off, cfg_sum = _cfg(), _sum_cfg()
    m_off, m_sum = _model(cfg_off), _model(cfg_sum)
    assert cfg_sum.dim_q == cfg_off.dim_q                       # main ANN input = SOAP
    for k in ("W0", "b0", "W1"):
        np.testing.assert_array_equal(getattr(m_sum, k).numpy(), getattr(m_off, k).numpy())
    sl = nd.nep_slices(cfg_sum); He, Hw, T = 6, m_sum._head_width, 2
    assert tuple(m_sum.W0_r.shape) == (T, sl["radial"].stop, He)
    assert tuple(m_sum.W0_a.shape) == (T, sl["angular"].stop - sl["angular"].start, He)
    assert tuple(m_sum.W1_r.shape) == tuple(m_sum.W1_a.shape) == (T, He, Hw)
    assert np.all(m_sum.W1_r.numpy() == 0) and np.all(m_sum.W1_a.numpy() == 0)
    assert np.any(m_sum.W0_r.numpy() != 0) and np.any(m_sum.W0_a.numpy() != 0)
    d_off, d_sum = prepare_eval_data([atoms], cfg_off), prepare_eval_data([atoms], cfg_sum)
    np.testing.assert_allclose(_predict_eff(m_sum, d_sum), _predict_eff(m_off, d_off), rtol=1e-6)
    # SNES μ-init: same SOAP slab, zero expert W1, same candidate prediction.
    o_off, o_sum = SNES(m_off), SNES(m_sum)
    n_off, n_sum = (o._split_reconstructed(o.reconstruct_params_tf(o.mu)) for o in (o_off, o_sum))
    for k in ("W0", "b0", "W1", "b1"):
        np.testing.assert_array_equal(n_sum[k].numpy(), n_off[k].numpy())
    assert np.all(n_sum["W1_r"].numpy() == 0) and np.all(n_sum["W1_a"].numpy() == 0)
    assert np.any(n_sum["W0_r"].numpy() != 0) and np.all(n_sum["b0_r"].numpy() == 0)
    np.testing.assert_allclose(o_sum.validate(d_sum, o_sum.mu), o_off.validate(d_off, o_off.mu), rtol=1e-5)


def test_sum_mode_experts_add():
    """Neighbor readout (K=0): μ is linear in the head, so μ(all) = μ(SOAP) + μ(2b) + μ(3b)."""
    cfg = _sum_cfg(); m = _model(cfg); atoms = _bulk_frame()
    _randomise_experts(m)
    d = prepare_eval_data([atoms], cfg)
    W1, W1_r, W1_a = (getattr(m, k).numpy().copy() for k in ("W1", "W1_r", "W1_a"))

    def mu_with(**on):
        for k, v in (("W1", W1), ("W1_r", W1_r), ("W1_a", W1_a)):
            getattr(m, k).assign(v if on.get(k, True) else np.zeros_like(v))
        return _predict_eff(m, d)[0]

    mu_all = mu_with()
    mu_soap = mu_with(W1_r=False, W1_a=False)
    mu_2b = mu_with(W1=False, W1_a=False)
    mu_3b = mu_with(W1=False, W1_r=False)
    for v in (mu_soap, mu_2b, mu_3b):
        assert np.linalg.norm(v) > 1e-6
    np.testing.assert_allclose(mu_all, mu_soap + mu_2b + mu_3b, rtol=1e-5, atol=1e-6)
    assert not np.allclose(mu_all, mu_soap, atol=1e-6)


@pytest.mark.parametrize("readout", ["neighbor", "bilinear"])
def test_sum_mode_four_path(readout):
    cfg = _sum_cfg(dipole_readout=readout); m = _model(cfg)
    _randomise_experts(m)
    _check_four_path(cfg, m)


def test_sum_mode_snes_layout_and_roundtrip():
    """Typed slab [W0|b0|W1 | W0_r|b0_r|W1_r | W0_a|b0_a|W1_a], then b1, then c_pair."""
    from SNES import SNES, _set_model_params
    from data import assemble_data_dict, pad_and_stack
    cfg = _sum_cfg(dipole_readout="bilinear", dipole_pair_rank=1, dipole_species_pairs="symmetric")
    m = _model(cfg); opt = SNES(m)
    T, H, He, Hw, Q = 2, cfg.num_neurons, 6, m._head_width, cfg.dim_q
    sl = nd.nep_slices(cfg); dr, da = sl["radial"].stop, sl["angular"].stop - sl["angular"].start
    n_main = T * (Q * H + H + H * Hw)
    n_r, n_a = T * (dr * He + He + He * Hw), T * (da * He + He + He * Hw)
    assert opt.n_typed == n_main + n_r + n_a
    assert opt._n_per_type == opt.n_typed // T
    assert opt.n_c > 0 and opt.n_primary == opt.n_typed + 1 + opt.n_c
    assert opt.dim == opt.n_anns_total + opt.n_nep
    mu = opt.mu.numpy()
    assert int(opt._extract_type_params(opt.mu, 0).shape[0]) == opt._n_per_type
    assert tuple(opt._extract_type_params_batched(opt.mu[tf.newaxis], 1).shape) == (1, opt._n_per_type)
    named = opt._split_reconstructed(opt.reconstruct_params_tf(opt.mu))
    # Slab positions: experts right after W1, b1 at n_typed, c_pair after b1.
    np.testing.assert_array_equal(mu[n_main:n_main + T * dr * He], named["W0_r"].numpy().ravel())
    np.testing.assert_array_equal(mu[n_main + n_r:n_main + n_r + T * da * He], named["W0_a"].numpy().ravel())
    assert mu[opt.n_typed] == named["b1"].numpy()
    np.testing.assert_array_equal(mu[opt.n_typed + 1:opt.n_typed + 1 + opt.n_c], named["c_pair"].numpy().ravel())
    assert np.all(named["W1_r"].numpy() == 0) and np.all(named["W1_a"].numpy() == 0)
    # Per-type labels: each expert block is typed by its centre type.
    tov = np.asarray(opt._type_of_variable)
    np.testing.assert_array_equal(tov[n_main:n_main + T * dr * He], np.repeat(np.arange(T), dr * He))
    np.testing.assert_array_equal(tov[n_main + n_r:n_main + n_r + T * da * He], np.repeat(np.arange(T), da * He))
    assert tov[opt.n_typed] == T
    # Per-type extraction picks type t's rows of every block: after type 0's
    # W0 / b0 / W1 rows of the SOAP block come its W0_r rows.
    tp0 = opt._extract_type_params(opt.mu, 0).numpy()
    off = Q * H + H + H * Hw
    np.testing.assert_array_equal(tp0[off:off + dr * He], mu[n_main:n_main + dr * He])
    # Round trip: perturb μ → assign → predict_batch(model weights) == candidates(μ).
    atoms = _bulk_frame()
    types_int = assign_type_indices([atoms], cfg.types)
    desc, grads, gidx = make_descriptor_builder(cfg).build_descriptors([atoms])
    full = pad_and_stack(assemble_data_dict([atoms], types_int, desc, grads, gidx, cfg),
                         num_types=cfg.num_types, self_pairs_only=True)
    mu_vec = opt.mu + 0.05 * tf.random.stateless_normal(opt.mu.shape, [3, 4])
    _set_model_params(m, *opt.reconstruct_params_tf(mu_vec))
    n = opt._split_reconstructed(opt.reconstruct_params_tf(mu_vec))
    for k in EXPERTS:
        np.testing.assert_array_equal(getattr(m, k).numpy(), n[k].numpy())
    assert np.any(m.W1_r.numpy() != 0)
    kw = dict(nbr_index=full["nbr_index"], nbr_G=full["nbr_G"], nep_raw=full["nep_raw"])
    mu_model = m.predict_batch(*_forward_args(m, full), W_atom=full["V_atom"], **kw)[0].numpy()
    nb = opt._split_reconstructed(opt.reconstruct_params_tf(mu_vec[tf.newaxis]))
    experts = tuple(nb[k] for k in EXPERTS)
    mu_cand = m.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        nb["W0"], nb["b0"], nb["W1"], nb["b1"], c_pair=nb["c_pair"],
        c_r=nb["c_r"], c_a=nb["c_a"], experts=experts, **kw)[0, 0].numpy()
    np.testing.assert_allclose(mu_cand, mu_model, rtol=1e-5, atol=1e-6)
    assert np.any(mu_cand != 0.0)
    # Liveness: two candidates differing only in the expert slabs.
    d = prepare_eval_data([atoms, atoms], cfg)
    mu_e = mu_vec.numpy().copy(); mu_e[n_main:opt.n_typed] *= 3.0
    fit = opt.evaluate_population(tf.stack([mu_vec, tf.constant(mu_e)]), d).numpy()
    assert np.all(np.isfinite(fit)) and fit[0] != fit[1]
    assert opt.validate(d, tf.constant(mu_e)) != opt.validate(d, mu_vec)


def test_sum_mode_regularisation_includes_experts():
    from SNES import SNES
    for types in ([8, 1], [8]):
        kw = dict(lambda_1=0.1, lambda_2=0.1)
        if len(types) == 1:
            kw.update(types=[8], num_types=1, type_map={8: 0})
        cfg = _sum_cfg(**kw); m = _model(cfg); opt = SNES(m)
        T, H, Hw, Q = cfg.num_types, cfg.num_neurons, m._head_width, cfg.dim_q
        n_main = T * (Q * H + H + H * Hw)
        mu = opt.mu.numpy().copy(); mu2 = mu.copy()
        mu2[n_main:n_main + T * m.W0_r.shape[1] * 6] *= 10.0     # W0_r slab
        assert opt.compute_regularization(mu) != opt.compute_regularization(mu2)
        assert not np.allclose(opt.compute_regularization_tf(tf.constant(mu[None])).numpy(),
                               opt.compute_regularization_tf(tf.constant(mu2[None])).numpy())


@pytest.mark.parametrize("scale", ["block", "channel", "minmax"])
def test_sum_mode_scaled_is_gen0_reparameterisation(scale):
    """gen0_ann_inputs is the full [q_soap ‖ q_nep]; the SOAP / radial / angular
    slices fold into W0, W0_r, W0_a; a scaled model equals the unscaled one."""
    from DescriptorBuilderGPU import compute_q_scale, q_scale_groups
    from TNEP import gen0_ann_inputs
    atoms = _bulk_frame(); small = atoms[:12]; small.info["dipole"] = np.zeros(3)
    cfg_off, cfg_on = _sum_cfg(), _sum_cfg(descriptor_scale=scale)
    d = prepare_eval_data([atoms, small], cfg_off)
    ds, dn = cfg_off.dim_q, nd.nep_dim(cfg_off)
    q0 = gen0_ann_inputs(cfg_on, d)
    assert q0.shape == (2, len(atoms), ds + dn)
    groups = q_scale_groups(cfg_on)
    assert sorted(np.concatenate(groups).tolist()) == list(range(ds + dn))
    assert len(groups) == len(q_scale_groups(_cfg(descriptor_scale=scale))) + 1 + 4 + 1
    m_np, mu_np = compute_q_scale(q0, d["atom_mask"].numpy(), cfg_on)
    cfg_on.q_scale, cfg_on.q_shift = [float(v) for v in m_np], [float(v) for v in mu_np]
    m_off, m_on = _model(cfg_off), _model(cfg_on)
    sl = nd.nep_slices(cfg_on); m_r, m_a = m_np[ds:][sl["radial"]], m_np[ds:][sl["angular"]]
    np.testing.assert_allclose(m_on.W0.numpy() * m_np[None, :ds, None], m_off.W0.numpy(), rtol=1e-6)
    np.testing.assert_allclose(m_on.W0_r.numpy() * m_r[None, :, None], m_off.W0_r.numpy(), rtol=1e-6)
    np.testing.assert_allclose(m_on.W0_a.numpy() * m_a[None, :, None], m_off.W0_a.numpy(), rtol=1e-6)
    for m in (m_off, m_on):
        _randomise_experts(m)                      # same draw on both → live experts
    for k, mk in (("r", m_r), ("a", m_a)):     # re-fold the shared draw into m_on
        getattr(m_on, f"W0_{k}").assign(getattr(m_off, f"W0_{k}").numpy() / mk[None, :, None])
        mu_k = (m_np * mu_np)[ds:][sl["radial" if k == "r" else "angular"]]
        getattr(m_on, f"b0_{k}").assign(getattr(m_off, f"b0_{k}").numpy()
                                        + np.einsum('q,tqh->th', mu_k, getattr(m_on, f"W0_{k}").numpy()))
    np.testing.assert_allclose(_predict_eff(m_on, d), _predict_eff(m_off, d), rtol=1e-4, atol=1e-6)
    assert np.abs(_predict_eff(m_on, d)).max() > 0


def test_sum_mode_save_load(tmp_path):
    import h5py, model_io
    cfg = _sum_cfg(); m = _model(cfg); atoms = _bulk_frame()
    _randomise_experts(m)
    mu0 = _mu_nep(m, atoms, cfg)
    p = str(tmp_path / "sum.h5")
    model_io.save_model(m, cfg, path=p)
    loaded = model_io.load_model(p)
    assert loaded.cfg.nep_descriptor == "sum" and loaded.cfg.nep_num_neurons == 6
    for k in EXPERTS:
        np.testing.assert_array_equal(getattr(loaded, k).numpy(), getattr(m, k).numpy())
    np.testing.assert_allclose(_mu_nep(loaded, atoms, loaded.cfg), mu0, atol=1e-6)
    with h5py.File(p, "a") as f:
        del f["weights"]["W0_r"]
    with pytest.raises(ValueError, match="W0_r"):
        model_io.load_model(p)


def test_sum_mode_adam_smoke():
    from adam import _trainable, _ann_core
    cfg = _sum_cfg(optimizer="adam", adam_lr=1e-2, num_generations=2, val_interval=1,
                   batch_chunk_size=1, dipole_species_pairs="symmetric")
    m = _model(cfg); atoms = _bulk_frame()
    keys = list(_trainable(m))
    assert keys[keys.index("b1") + 1:keys.index("b1") + 7] == list(EXPERTS)
    assert keys.index("W1_a") < keys.index("c_pair")
    assert len(_ann_core(m)) == 9
    d = prepare_eval_data([atoms, atoms], cfg)
    history, final, best = m.fit(d, d)
    assert np.all(np.isfinite(history["train_loss"]))
    for mdl in (final, best):
        assert np.all(np.isfinite(mdl.W1_r.numpy())) and np.any(mdl.W1_r.numpy() != 0)
        assert np.any(mdl.W1_a.numpy() != 0)


# ── Task 10 (Phase B): per-pair basis derivatives + image-exact pair stream ─

def _fd(f, x, h=1e-6):
    """Central FD of f: [P] → [P, ...] w.r.t. its single vector argument.
    h=1e-6 in float64: h² truncation ≈ 1e-12, eps/h roundoff ≈ 1e-10 → rtol 1e-6 is safe."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def test_fn_and_fnp_matches_fd_of_fn_all():
    rc, K = 8.0, 6
    r = np.linspace(0.1, rc - 0.01, 300)
    fn, fnp = nd.fn_and_fnp(r, rc, K)
    assert fn.shape == fnp.shape == (300, K + 1) and fn.dtype == fnp.dtype == np.float64
    np.testing.assert_array_equal(fn, nd.fn_all(r, rc, K))
    np.testing.assert_allclose(fnp, _fd(lambda x: nd.fn_all(x, rc, K), r), rtol=1e-6, atol=1e-8)
    # Beyond the cutoff both vanish (find_fc_and_fcp else-branch).
    fn0, fnp0 = nd.fn_and_fnp(np.array([rc, rc + 1.0]), rc, K)
    assert np.all(fn0 == 0.0) and np.all(fnp0 == 0.0)


def test_real_harmonics_grad_matches_fd():
    rng = np.random.default_rng(3)
    u = rng.standard_normal((40, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    g = nd.real_harmonics_grad(u[:, 0], u[:, 1], u[:, 2])
    assert g.shape == (40, 3, 24) and g.dtype == np.float64
    h = 1e-6
    for c in range(3):   # x̂, ŷ, ẑ perturbed as FREE variables (no renormalisation)
        def P(uc, c=c):
            v = u.copy(); v[:, c] = uc
            return nd.real_harmonics(v[:, 0], v[:, 1], v[:, 2])
        np.testing.assert_allclose(g[:, c], _fd(P, u[:, c], h), rtol=1e-6, atol=1e-8)


def test_tf_basis_twins_match_numpy():
    rng = np.random.default_rng(4)
    rc, K = 8.0, 6
    r = rng.uniform(0.1, rc + 0.5, 100)   # includes r ≥ rc
    u = rng.standard_normal((100, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    fn, fnp = nd.fn_and_fnp(r, rc, K)
    fn_t, fnp_t = nd.fn_and_fnp_tf(tf.constant(r, tf.float32), rc, K)
    assert fn_t.dtype == fnp_t.dtype == tf.float32
    np.testing.assert_allclose(fn_t.numpy(), fn, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fnp_t.numpy(), fnp, rtol=1e-5, atol=1e-6)
    ut = [tf.constant(u[:, c], tf.float32) for c in range(3)]
    P_t = nd.real_harmonics_tf(*ut); g_t = nd.real_harmonics_grad_tf(*ut)
    assert P_t.dtype == g_t.dtype == tf.float32
    np.testing.assert_allclose(P_t.numpy(), nd.real_harmonics(*u.T), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(g_t.numpy(), nd.real_harmonics_grad(*u.T), rtol=1e-5, atol=1e-6)


def test_nep_pairs_numpy_resums_to_raw_and_is_image_exact():
    from ase.neighborlist import neighbor_list
    cfg = _cfg(nep_descriptor="concat"); atoms = _bulk_frame()
    t = _types_int(atoms, ["O", "H"]); N, T = len(atoms), 2
    rc_r, rc_a = cfg.nep_rc_radial, cfg.nep_rc_angular
    Kr, Ka = cfg.nep_basis_size_radial, cfg.nep_basis_size_angular
    p = nd.nep_pairs_numpy(atoms, cfg)
    assert set(p) == {"nep_pair_atom", "nep_pair_gidx", "nep_pair_D", "nep_pair_ang"}
    P = p["nep_pair_atom"].shape[0]
    assert (p["nep_pair_atom"].dtype, p["nep_pair_gidx"].dtype, p["nep_pair_D"].dtype, p["nep_pair_ang"].dtype) == \
        (np.int32, np.int32, np.float32, np.bool_)
    assert p["nep_pair_D"].shape == (P, 3) and p["nep_pair_ang"].shape == (P,)
    assert np.all(np.diff(p["nep_pair_atom"]) >= 0)                 # sorted by centre
    r = np.linalg.norm(p["nep_pair_D"].astype(np.float64), axis=1)
    assert np.all(r > 0.0) and np.all(r < rc_r)                 # no self image, within rc_r
    L = float(atoms.cell[0, 0])
    assert rc_r > L / 2                                         # MIC would be WRONG here:
    # several images of one (i, j) are in the list — no MIC construction can produce that.
    assert len(set(zip(p["nep_pair_atom"].tolist(), p["nep_pair_gidx"].tolist()))) < P
    assert int(p["nep_pair_ang"].sum()) == int(np.sum(np.linalg.norm(
        neighbor_list("D", atoms, rc_a), axis=1) > 0.0))        # ang = the rc_a list
    # Re-summing the stream reproduces nep_raw_numpy.
    raw = nd.nep_raw_numpy(atoms, t, cfg)
    raw_r = np.zeros((N, T, Kr + 1)); raw_a = np.zeros((N, T, Ka + 1, 24))
    tj = t[p["nep_pair_gidx"]]
    np.add.at(raw_r, (p["nep_pair_atom"], tj), nd.fn_all(r, rc_r, Kr))
    a = p["nep_pair_ang"]; u = p["nep_pair_D"][a].astype(np.float64) / r[a][:, None]
    np.add.at(raw_a, (p["nep_pair_atom"][a], tj[a]),
              nd.fn_all(r[a], rc_a, Ka)[:, :, None] * nd.real_harmonics(*u.T)[:, None, :])
    ours = np.concatenate([raw_r.reshape(N, -1), raw_a.reshape(N, -1)], 1)
    # atol: the stream stores float32 D (raw sums float64 D) → ~5e-7 on cancelling sums.
    np.testing.assert_allclose(ours, raw, rtol=1e-6, atol=2e-6)


def test_nep_pair_stream_rides_data_dict_and_every_slicer():
    from data import (prepare_eval_data, slice_and_complete_chunk, prefetched_chunks,
                      move_data_to_gpu, prestage_chunk_indices)
    # dipole_readout="tnep" needs the stream; the Phase B guard lives in TNEP.__init__,
    # not in the data path, so no model is built here.
    cfg = _cfg(nep_descriptor="concat", dipole_readout="tnep")
    full = _bulk_frame(); small = full[:12]; small.info["dipole"] = np.zeros(3)
    d = prepare_eval_data([full, small], cfg)
    p0, p1 = nd.nep_pairs_numpy(full, cfg), nd.nep_pairs_numpy(small, cfg)
    P0, P1 = p0["nep_pair_atom"].shape[0], p1["nep_pair_atom"].shape[0]
    np.testing.assert_array_equal(d["nep_struct_ptr"].numpy(), [0, P0, P0 + P1])
    assert d["nep_pair_D"].dtype == tf.float32 and d["nep_pair_ang"].dtype == tf.bool
    assert d["nep_pair_atom"].dtype == d["nep_pair_gidx"].dtype == d["nep_pair_struct"].dtype == tf.int32
    np.testing.assert_array_equal(d["nep_pair_struct"].numpy(), [0] * P0 + [1] * P1)
    np.testing.assert_array_equal(d["nep_pair_atom"].numpy()[P0:], p1["nep_pair_atom"])
    np.testing.assert_array_equal(d["nep_pair_D"].numpy()[:P0], p0["nep_pair_D"])
    np.testing.assert_array_equal(d["nep_pair_ang"].numpy()[P0:], p1["nep_pair_ang"])

    def check_frame1(ch):
        n = int(ch["nep_pair_atom"].shape[0])                      # bucket-padded (power of two)
        assert n >= P1 and n & (n - 1) == 0
        np.testing.assert_array_equal(ch["nep_struct_ptr"].numpy(), [0, P1])   # real pairs only
        assert np.all(ch["nep_pair_struct"].numpy() == 0)          # chunk-local
        np.testing.assert_array_equal(ch["nep_pair_atom"].numpy()[:P1], p1["nep_pair_atom"])
        np.testing.assert_array_equal(ch["nep_pair_gidx"].numpy()[:P1], p1["nep_pair_gidx"])
        np.testing.assert_array_equal(ch["nep_pair_D"].numpy()[:P1], p1["nep_pair_D"])
        np.testing.assert_array_equal(ch["nep_pair_ang"].numpy()[:P1], p1["nep_pair_ang"])
        np.testing.assert_array_equal(ch["nep_pair_D"].numpy()[P1:], np.tile([[cfg.nep_rc_radial + 1, 0, 0]], (n - P1, 1)))
        assert np.all(ch["nep_pair_atom"].numpy()[P1:] == 0) and np.all(ch["nep_pair_gidx"].numpy()[P1:] == 0)

    check_frame1(slice_and_complete_chunk(d, [1]))
    ranges = [(0, 1), (1, 2)]
    prestage_chunk_indices(d, ranges)
    for pin in (True, False):                                      # in-RAM / GPU-resident stagers
        dd = dict(d)
        if not pin:
            dd["_gv_resident_gpu"] = True
            move_data_to_gpu(dd)
        chunks = [c for _, _, c in prefetched_chunks(dd, ranges, pin_to_cpu=pin)]
        np.testing.assert_array_equal(chunks[0]["nep_pair_atom"].numpy()[:P0], p0["nep_pair_atom"])
        check_frame1(chunks[1])
    # Both frames: struct index remapped, pairs concatenated in order.
    ch = slice_and_complete_chunk(d, [1, 0])
    np.testing.assert_array_equal(ch["nep_pair_struct"].numpy()[:P1 + P0], [0] * P1 + [1] * P0)
    np.testing.assert_array_equal(ch["nep_pair_D"].numpy()[P1:P1 + P0], p0["nep_pair_D"])
    # Not built without descriptor gradients, or with NEP off.
    assert "nep_struct_ptr" not in prepare_eval_data([small], _cfg(nep_descriptor="concat"))
    assert "nep_struct_ptr" not in prepare_eval_data([small], _cfg(dipole_readout="tnep"))


# ── Task 11 (Phase B): per-candidate NEP pair forces for dipole_readout="tnep" ─

def _tnep_cfg(mode, **kw):
    return _cfg(**{"nep_descriptor": mode, "dipole_readout": "tnep", "dipole_rij_power": 2, **kw})


def _small_frames(n, seed=0):
    """12-atom truncations of the bulk frame (cell kept → images at rc_r = 8 Å > L/2)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        at = _bulk_frame()[:12]
        at.set_positions(at.get_positions() + 0.05 * rng.standard_normal((12, 3)))
        at.info["dipole"] = rng.standard_normal(3)
        out.append(at)
    return out


def _tnep_kw(d):
    return dict(nep_raw=d["nep_raw"], **nd.nep_pair_kwargs(d))


def _mu_batch(m, d, W_atom=None, weights=None, c=None):
    """μ [B, 3] via predict_batch (W_atom None → SOAP part through the COO path)."""
    W0, b0, W1, b1 = weights if weights is not None else (m.W0, m.b0, m.W1, m.b1)
    c_r, c_a = c if c is not None else (None, None)
    return m.predict_batch(
        d["descriptors"], d["grad_values"], d["pair_atom"], d["pair_gidx"], d["pair_struct"],
        d["positions"], d["Z_int"], d["boxes"], d["atom_mask"], W0, b0, W1, b1,
        W_atom=W_atom, c_r=c_r, c_a=c_a, **_tnep_kw(d)).numpy()


def test_parameter_count_matches_gpumd_formula():
    """SNES.dim for the faithful GPUMD TNEP configuration equals GPUMD's
    number_of_variables (parameters.cu:203-235): typed slab (dim+2)·H per
    type + shared b1 + c_r/c_a — no extra parameters anywhere."""
    cfg = _tnep_cfg("replace", descriptor_mixing=False,
                    descriptor_preprocess_contract="off", descriptor_scale="off",
                    nep_n_max_radial=6, nep_n_max_angular=6, nep_basis_size_radial=6,
                    nep_basis_size_angular=6, nep_four_body=True, nep_five_body=False)
    m = _model(cfg)
    assert int(m.optimizer.dim) == nd.gpumd_num_params(cfg)
    assert nd.gpumd_num_params(cfg) == 1097   # = 2*(42+2)*8 + 1 + 4*(49+49): T=2, dim=7+7·5, H=8


@pytest.mark.parametrize("four_body", [True, False])
def test_nep_pair_forces_match_fd(four_body):
    """F_p = ∂U_i/∂D_p (D = r_j − r_i) for U = Σ_{i,d} Fp[i,d] q_d with Fp frozen:
    FD of U w.r.t. atom a's position == Σ_{p: j_p=a} F_p − Σ_{p: i_p=a} F_p."""
    cfg = _tnep_cfg("replace", nep_four_body=four_body); m = _model(cfg)
    atoms = _small_frames(1)[0]
    d = prepare_eval_data([atoms], cfg)
    types = np.asarray(assign_type_indices([atoms], cfg.types)[0]); N = len(atoms)
    rng = np.random.default_rng(5)
    c_r = rng.uniform(-1, 1, m.c_r.shape).astype(np.float32)
    c_a = rng.uniform(-1, 1, m.c_a.shape).astype(np.float32)
    Fp = rng.standard_normal((N, nd.nep_dim(cfg))).astype(np.float32)
    _, s = m.nep_from_raw(d["nep_raw"], d["Z_int"], tf.constant(c_r), tf.constant(c_a), return_s=True)
    F = m._nep_pair_forces(tf.constant(Fp)[None, None], s[None], tf.constant(c_r), tf.constant(c_a),
                           d["Z_int"], **nd.nep_pair_kwargs(d))
    i, j = d["nep_pair_atom"].numpy(), d["nep_pair_gidx"].numpy()
    assert tuple(F.shape) == (1, i.shape[0], 3)
    F = F[0].numpy().astype(np.float64)

    def U(pos):
        at = atoms.copy(); at.set_positions(pos)
        raw = nd.nep_raw_numpy(at, types, cfg, dtype=np.float64)
        return float(np.sum(Fp.astype(np.float64) * nd.nep_contract_numpy(raw, types, c_r, c_a, cfg)))

    # h = 1e-3 in float64 FD (truncation ~h² = 1e-6 relative, roundoff eps/h ~ 1e-13);
    # the analytic side is float32 TF summed over ~30 pairs → rtol 1e-4 is the float32 floor.
    h = 1e-3
    for a in (0, 5):                                     # an O and an H
        analytic = F[j == a].sum(0) - F[i == a].sum(0)
        fd = np.zeros(3)
        for x in range(3):
            pos_p, pos_m = atoms.get_positions(), atoms.get_positions()
            pos_p[a, x] += h; pos_m[a, x] -= h
            fd[x] = (U(pos_p) - U(pos_m)) / (2.0 * h)
        assert np.abs(fd).max() > 1e-2
        np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-4 * np.abs(fd).max())


def _mu_reference(m, cfg, d):
    """Independent μ: U = Σ_i ANN(q_i) with q_nep built in-graph from the pair
    displacements D as free variables; μ = −Σ_p |D_p|² ∂U/∂D_p (+ the SOAP COO
    part from ∂U/∂q_soap under concat). Uses only basis VALUES — no derivative code."""
    A, T = int(d["Z_int"].shape[1]), cfg.num_types
    Z = d["Z_int"][0]; i = d["nep_pair_atom"]; j = d["nep_pair_gidx"]
    ang = tf.cast(d["nep_pair_ang"], tf.float32)
    D = tf.Variable(d["nep_pair_D"]); q_soap = tf.Variable(d["descriptors"][0])
    Kr, Ka = cfg.nep_basis_size_radial, cfg.nep_basis_size_angular
    with tf.GradientTape(persistent=True) as tape:
        r = tf.norm(D, axis=1); u = D / r[:, None]
        seg = i * T + tf.gather(Z, j)
        fr = nd.fn_and_fnp_tf(r, cfg.nep_rc_radial, Kr)[0]
        raw_r = tf.math.unsorted_segment_sum(fr, seg, A * T)
        fa = nd.fn_and_fnp_tf(r, cfg.nep_rc_angular, Ka)[0] * ang[:, None]
        Pc = nd.real_harmonics_tf(u[:, 0], u[:, 1], u[:, 2])
        raw_a = tf.math.unsorted_segment_sum(fa[:, :, None] * Pc[:, None, :], seg, A * T)
        raw = tf.concat([tf.reshape(raw_r, [A, -1]), tf.reshape(raw_a, [A, -1])], 1)
        q = m.nep_from_raw(raw[None], Z[None])[0]
        if cfg.nep_descriptor == "concat":
            q = tf.concat([q_soap, q], 1)
        h1 = m.activation(tf.einsum('aq,aqh->ah', q, tf.gather(m.W0, Z)) + tf.gather(m.b0, Z))
        U = tf.reduce_sum(h1 * tf.gather(m.W1, Z)) + tf.cast(A, tf.float32) * m.b1
    np.testing.assert_allclose(raw.numpy(), d["nep_raw"][0].numpy(), rtol=1e-5, atol=1e-5)
    F = tape.gradient(U, D)
    mu = -tf.reduce_sum(tf.reduce_sum(tf.square(D), 1)[:, None] * F, 0)
    if cfg.nep_descriptor == "concat":
        de_dq = tape.gradient(U, q_soap)[None]
        mu += m._dipole_coo(m._calc_forces_coo(de_dq, d["grad_values"], d["pair_struct"], d["pair_atom"]),
                            d["pair_struct"], d["pair_atom"], d["pair_gidx"], d["positions"],
                            d["boxes"], tf.linalg.inv(d["boxes"]), 1)[0]
    return mu.numpy()


@pytest.mark.parametrize("mode", ["concat", "replace"])
def test_nep_tnep_mu_matches_autodiff_reference(mode):
    cfg = _tnep_cfg(mode); m = _model(cfg)
    m.W1.assign(m.W1 + 0.5)                              # live output weights
    atoms = _small_frames(1, seed=1)[0]
    d = prepare_eval_data([atoms], cfg)
    mu = _mu_batch(m, d)[0]
    ref = _mu_reference(m, cfg, d)
    assert np.linalg.norm(ref) > 1e-3
    np.testing.assert_allclose(mu, ref, rtol=1e-4, atol=1e-4 * np.linalg.norm(ref))


@pytest.mark.parametrize("mode", ["concat", "replace"])
def test_nep_tnep_candidates_match_batch(mode):
    cfg = _tnep_cfg(mode); m = _model(cfg)
    d = prepare_eval_data(_small_frames(2, seed=2), cfg)
    rng = np.random.default_rng(3)
    def two(v):                                          # two distinct candidates
        v = v.numpy(); return tf.constant(np.stack([v, 0.7 * v + 0.1 * rng.standard_normal(v.shape)]).astype(np.float32))
    W0, b0, W1, b1 = (two(getattr(m, k)) for k in ("W0", "b0", "W1", "b1"))
    W1 = W1 + 0.5; c_r, c_a = two(m.c_r), two(m.c_a)
    W_atom = m._precompute_dipole_kernel(d["grad_values"], d["pair_struct"], d["pair_atom"], d["pair_gidx"],
                                         d["positions"], d["boxes"], 2, int(d["Z_int"].shape[1]))
    mu_c = m.predict_batch_candidates(d["descriptors"], W_atom, d["Z_int"], d["atom_mask"],
                                      W0, b0, W1, b1, c_r=c_r, c_a=c_a, **_tnep_kw(d)).numpy()
    assert mu_c.shape == (2, 2, 3) and not np.allclose(mu_c[0], mu_c[1])
    for c in range(2):
        w = (W0[c], b0[c], W1[c], b1[c]); cc = (c_r[c], c_a[c])
        mu_k = _mu_batch(m, d, W_atom=W_atom, weights=w, c=cc)
        mu_coo = _mu_batch(m, d, W_atom=None, weights=w, c=cc)
        atol = 1e-5 * max(np.abs(mu_k).max(), 1.0)
        np.testing.assert_allclose(mu_c[c], mu_k, atol=atol)
        np.testing.assert_allclose(mu_coo, mu_k, atol=atol)


def test_tnep_soap_only_paths_agree():
    """NEP off: the SOAP tnep readout is untouched — W_atom path == COO path == candidates."""
    cfg = _cfg(dipole_readout="tnep", dipole_rij_power=2); m = _model(cfg)
    m.W1.assign(m.W1 + 0.5)
    d = prepare_eval_data(_small_frames(2, seed=4), cfg)
    args = _forward_args(m, d)
    W_atom = m._precompute_dipole_kernel(d["grad_values"], d["pair_struct"], d["pair_atom"], d["pair_gidx"],
                                         d["positions"], d["boxes"], 2, int(d["Z_int"].shape[1]))
    mu_k = m.predict_batch(*args, W_atom=W_atom).numpy()
    mu_coo = m.predict_batch(*args).numpy()
    mu_c = m.predict_batch_candidates(d["descriptors"], W_atom, d["Z_int"], d["atom_mask"],
                                      m.W0[None], m.b0[None], m.W1[None], m.b1[None])[0].numpy()
    assert np.abs(mu_k).max() > 1e-4
    np.testing.assert_allclose(mu_coo, mu_k, atol=1e-5 * np.abs(mu_k).max())
    np.testing.assert_allclose(mu_c, mu_k, atol=1e-5 * np.abs(mu_k).max())


@pytest.mark.parametrize("mode", ["concat", "replace"])
def test_nep_tnep_snes_roundtrip_and_minibatch(mode):
    cfg = _tnep_cfg(mode, pop_size=4, num_generations=2, batch_size=2, val_interval=1)
    m = _model(cfg)
    d = prepare_eval_data(_small_frames(3, seed=5), cfg)
    snes = m.optimizer
    fit = snes.evaluate_population(tf.stack([snes.mu, snes.mu + 0.1]), d).numpy()
    assert fit.shape == (2,) and np.all(np.isfinite(fit)) and fit[0] != fit[1]
    assert np.isfinite(snes.validate(d, snes.mu))
    missing = {k: v for k, v in d.items() if not k.startswith("nep_pair") and k != "nep_struct_ptr"}
    with pytest.raises(ValueError, match="nep_pair"):
        snes.validate(missing)
    with pytest.raises(ValueError, match="nep_pair"):
        snes.evaluate_population(tf.stack([snes.mu]), missing)
    history, _, _ = m.fit(d, d)                          # batch_size=2: the mini-batch CSR twin
    assert len(history["val_loss"]) == 2 and np.all(np.isfinite(history["val_loss"]))
    metrics, preds = m.score(d)
    assert np.isfinite(float(metrics["rmse"])) and preds.shape == (3, 3)


def test_nep_tnep_adam_smoke():
    cfg = _tnep_cfg("replace", optimizer="adam", adam_lr=1e-2, num_generations=2, val_interval=1,
                    batch_chunk_size=2)
    m = _model(cfg)
    d = prepare_eval_data(_small_frames(2, seed=6), cfg)
    c_a0 = m.c_a.numpy().copy()
    history, _, _ = m.fit(d, d)
    assert np.all(np.isfinite(history["train_loss"])) and np.all(np.isfinite(history["val_loss"]))
    assert not np.array_equal(m.c_a.numpy(), c_a0)      # c_a in the tape through the pair forces


def test_nep_tnep_guards():
    with pytest.raises(ValueError, match="dipole_rij_power"):
        _model(_tnep_cfg("concat", dipole_rij_power=0))
    with pytest.raises(ValueError, match="radial_basis_size"):   # M_radial would ride the genome unused
        _model(_cfg(nep_descriptor="replace", radial_basis_size=4))
    cfg = _tnep_cfg("replace"); m = _model(cfg)
    atoms = _small_frames(1)[0]; d = prepare_eval_data([atoms], cfg); A = len(atoms)
    with pytest.raises(NotImplementedError, match="predict_batch"):
        m.predict(d["descriptors"][0], tf.zeros([A, 1, 3, cfg.dim_q]), tf.zeros([A, 1], tf.int32),
                  d["positions"][0], d["Z_int"][0], d["boxes"][0], d["atom_mask"][0],
                  tf.zeros([A, 1]), nep_raw=d["nep_raw"][0])


def test_nep_pair_padding_is_exact_and_bucketed():
    """Chunks bucket-pad the pair stream (P and the angular count P_a to powers of
    two) so the XLA kernels compile once per bucket: μ is unchanged, and two
    mini-batches with different true pair counts share one bucket."""
    from data import slice_and_complete_chunk
    cfg = _tnep_cfg("concat"); m = _model(cfg); m.W1.assign(m.W1 + 0.5)
    d = prepare_eval_data(_small_frames(3, seed=5), cfg)          # full dict: unpadded
    ptr = d["nep_struct_ptr"].numpy(); P = int(ptr[-1]); P_a = int(d["nep_pair_ang"].numpy().sum())
    ch = slice_and_complete_chunk(d, [0, 1, 2])
    Pp = int(ch["nep_pair_D"].shape[0]); Pap = int(ch["nep_pair_ang"].numpy().sum())
    assert Pp > P and Pp & (Pp - 1) == 0 and Pap > P_a and Pap & (Pap - 1) == 0
    np.testing.assert_array_equal(ch["nep_pair_D"].numpy()[P:], np.tile([[cfg.nep_rc_radial + 1, 0, 0]], (Pp - P, 1)))
    assert np.all(ch["nep_pair_ang"].numpy()[P:P + Pap - P_a]) and not np.any(ch["nep_pair_ang"].numpy()[P + Pap - P_a:])
    mu, mu_pad = _mu_batch(m, d), _mu_batch(m, ch)
    assert np.abs(mu).max() > 1e-4
    np.testing.assert_allclose(mu_pad, mu, atol=1e-6)
    # Different true pair counts (258 vs 256 here) → same bucket for P and P_a.
    n0, n1 = ptr[1] - ptr[0], ptr[2] - ptr[1]
    assert n0 != n1
    c0, c1 = slice_and_complete_chunk(d, [0]), slice_and_complete_chunk(d, [1])
    assert int(c0["nep_pair_D"].shape[0]) == int(c1["nep_pair_D"].shape[0])
    assert int(c0["nep_pair_ang"].numpy().sum()) == int(c1["nep_pair_ang"].numpy().sum())
    np.testing.assert_allclose(_mu_batch(m, c0)[0], mu[0], atol=1e-6)



def test_replace_tnep_skips_soap_gradients_and_w_atom(monkeypatch, tmp_path):
    """replace+tnep: μ is NEP-only, so the SOAP gradient stream is not built
    (zero-width, like the neighbour readout) and SNES never forms W_atom;
    concat keeps both streams. μ is unchanged by the SOAP stream's width."""
    from data import (soap_gradients_needed, nep_pairs_needed, descriptor_gradients_needed,
                      assemble_data_dict, pad_and_stack)
    cfg, cc = _tnep_cfg("replace"), _tnep_cfg("concat")
    assert descriptor_gradients_needed(cfg) and nep_pairs_needed(cfg) and not soap_gradients_needed(cfg)
    assert soap_gradients_needed(cc) and nep_pairs_needed(cc)
    assert not nep_pairs_needed(_cfg(nep_descriptor="replace")) and not nep_pairs_needed(_cfg(dipole_readout="tnep"))
    frames = _small_frames(2, seed=7)
    d = prepare_eval_data(frames, cfg)
    assert int(d["grad_values"].shape[0]) == 0 and "nep_struct_ptr" in d
    dc = prepare_eval_data(frames, cc)
    assert int(dc["grad_values"].shape[0]) > 0 and "nep_struct_ptr" in dc

    m = _model(cfg); m.W1.assign(m.W1 + 0.5)
    # Same frames with the full SOAP COO stream forced: identical μ (never read).
    b = make_descriptor_builder(cfg)
    full = pad_and_stack(assemble_data_dict(frames, assign_type_indices(frames, cfg.types),
                                            *b.build_descriptors(frames, calc_gradients=True), cfg),
                         num_types=cfg.num_types)
    assert int(full["grad_values"].shape[0]) > 0
    mu = _mu_batch(m, d)
    assert np.abs(mu).max() > 1e-4
    np.testing.assert_array_equal(mu, _mu_batch(m, full))
    mu_c = m.predict_batch_candidates(d["descriptors"], None, d["Z_int"], d["atom_mask"],
                                      m.W0[None], m.b0[None], m.W1[None], m.b1[None],
                                      c_r=m.c_r[None], c_a=m.c_a[None], **_tnep_kw(d))[0].numpy()
    np.testing.assert_allclose(mu_c, mu, atol=1e-5 * np.abs(mu).max())

    def no_kernel(*a, **k):
        raise AssertionError("W_atom kernel must not be built under nep_descriptor='replace'")
    monkeypatch.setattr(m, "_precompute_dipole_kernel", no_kernel)
    snes = m.optimizer
    fit = snes.evaluate_population(tf.stack([snes.mu, snes.mu + 0.1]), d).numpy()
    assert fit.shape == (2,) and np.all(np.isfinite(fit)) and fit[0] != fit[1]
    assert np.isfinite(snes.validate(d, snes.mu))
    assert np.isfinite(snes.validate(d, snes.mu))      # cached path (val_size None)
    # score_from_file builds with the same gating (no SOAP gradients) and scores.
    from ase.io import write
    write(str(tmp_path / "f.xyz"), frames)
    _, preds = m.score_from_file(str(tmp_path / "f.xyz"), plot=False)
    _, preds_d = m.score(d)                            # same consumer, empty-stream dict
    np.testing.assert_allclose(preds.numpy(), preds_d.numpy(), atol=1e-5 * np.abs(preds_d).max())
