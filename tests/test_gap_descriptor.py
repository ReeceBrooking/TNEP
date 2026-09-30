"""Tests for gap_descriptor.py (gap_descriptor_plan.md Task 1): TurboGAP 2-body
and 3-body distance descriptors with fixed centres, exact against direct NumPy
transcriptions of the gapf90 gap.f90 energy loops (gapf90_gap.f90.ref).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pytest
from ase.neighborlist import neighbor_list

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_neighbor_dipole import _cfg as _base_cfg, _bulk_frame
from data import assign_type_indices
import gap_descriptor as gd

RCUT, BUFFER = 3.5, 1.0     # buffer 1.0 so the taper is exercised
SIGMA2, SIGMA3 = 0.3, [0.5, 0.4, 0.6]


def _cfg(**kw):
    return _base_cfg(**{"gap2b_rcut": RCUT, "gap3b_rcut": RCUT, "gap_buffer": BUFFER,
                        "gap2b_n_centres": 7, "gap2b_sigma": SIGMA2,
                        "gap3b_n_centres": 9, "gap3b_sigma": list(SIGMA3), **kw})


def _frame():
    at = _bulk_frame()
    return at, assign_type_indices([at], [8, 1])[0]


def _random_centres(rng, cfg):
    cfg.gap2b_centres = rng.uniform(0.5, RCUT, cfg.gap2b_n_centres).tolist()
    lo, hi = np.array([1.0, 0.0, 0.5]), np.array([2 * RCUT, RCUT ** 2, 2 * RCUT])
    cfg.gap3b_centres = rng.uniform(lo, hi, (cfg.gap3b_n_centres, 3)).tolist()
    return cfg


# ── gap.f90 transcriptions (delta = 1, cutoff(s) = 1, e0 = 0, exp kernel) ──

def _fcut_f90(r, rcut, buffer):
    # No r ≥ rcut branch: callers gate on r < rcut first, as gap.f90:271/604 do.
    if r < rcut - buffer:
        return 1.0
    return (np.cos(np.pi * (r - rcut + buffer) / buffer) + 1.0) / 2.0


def _pairs(atoms, rcut):
    i, j, D = neighbor_list("ijD", atoms, rcut)
    r = np.linalg.norm(D, axis=1)
    # r > 0 drops the self entry — the Fortran's `k = k + 1` before `do j = 2, n_neigh(i)`.
    keep = r > 0
    return i[keep], j[keep], D[keep], r[keep]


def _gap2b_energy_f90(atoms, sp, alphas, Qs, sigma, sp1, sp2, rcut, buffer):
    """get_2b_energy_and_forces energy loop (gap.f90:250-281), summed over sites."""
    ii, jj, _, rr = _pairs(atoms, rcut)
    E = 0.0
    for a, b, r in zip(ii, jj, rr):
        if sp[a] != sp1 and sp[a] != sp2:
            continue
        if not ((sp[a] == sp1 and sp[b] == sp2) or (sp[a] == sp2 and sp[b] == sp1)):
            continue
        if r < rcut:
            fcut = _fcut_f90(r, rcut, buffer)
            for s in range(len(alphas)):
                E += alphas[s] * fcut * np.exp(-0.5 * (r - Qs[s]) ** 2 / sigma ** 2)
    return E


def _gap3b_energy_f90(atoms, sp, alphas, Qs, sigma, sp0, sp1, sp2, rcut, buffer):
    """get_3b_energy_and_forces energy loop (gap.f90:566-647), summed over sites."""
    ii, jj, DD, rr = _pairs(atoms, rcut)
    Qs, sigma = np.asarray(Qs, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    pref = 2.0 * np.asarray(alphas, dtype=np.float64)
    E = 0.0
    for i in range(len(atoms)):
        if sp[i] != sp0:
            continue
        ks = np.nonzero(ii == i)[0]
        for a, k in enumerate(ks):
            r12 = rr[k]
            nsk = sp[jj[k]]
            if (nsk != sp1 and nsk != sp2) or r12 > rcut:
                continue
            xyz12 = DD[k]
            for k2 in ks[a + 1:]:
                r13 = rr[k2]
                nsk2 = sp[jj[k2]]
                if not (((nsk == sp1 and nsk2 == sp2) or (nsk == sp2 and nsk2 == sp1)) and r13 < rcut):
                    continue
                xyz13 = DD[k2]
                r23 = np.sqrt(np.sum((xyz13 - xyz12) ** 2))
                if r12 < rcut and r13 < rcut:
                    fcut = _fcut_f90(r12, rcut, buffer) * _fcut_f90(r13, rcut, buffer)
                    q = np.array([r12 + r13, (r12 - r13) ** 2, r23])
                    d = np.sqrt(np.sum((q - Qs) ** 2 / sigma ** 2, axis=1))
                    E += fcut * np.sum(pref * np.exp(-0.5 * d ** 2))
    return E


# ── exactness ───────────────────────────────────────────────────────────

def test_taper_is_exercised():
    at, _ = _frame()
    _, _, _, r = _pairs(at, RCUT)
    assert np.any((r >= RCUT - BUFFER) & (r < RCUT))
    assert np.any(r < RCUT - BUFFER)


@pytest.mark.parametrize("sp1,sp2", [(0, 0), (0, 1), (1, 1)])
def test_gap2b_matches_f90(sp1, sp2):
    rng = np.random.default_rng(11)
    cfg = _random_centres(rng, _cfg())
    at, ti = _frame()
    alphas = rng.normal(size=cfg.gap2b_n_centres)
    q2 = gd.gap_raw_numpy(at, ti, cfg)[:, gd.gap_slices(cfg)["two_body"]].astype(np.float64)
    q2 = q2.reshape(len(at), cfg.num_types, cfg.gap2b_n_centres)
    E_ours = alphas @ q2[ti == sp1, sp2].sum(0)
    if sp1 != sp2:
        E_ours += alphas @ q2[ti == sp2, sp1].sum(0)
    E_f90 = _gap2b_energy_f90(at, ti, alphas, cfg.gap2b_centres, cfg.gap2b_sigma,
                              sp1, sp2, RCUT, BUFFER)
    assert abs(E_f90) > 1e-3
    np.testing.assert_allclose(E_ours, E_f90, rtol=1e-6)


@pytest.mark.parametrize("sp0,sp1,sp2", [(0, 1, 1), (0, 0, 1), (1, 0, 1), (1, 1, 1)])
def test_gap3b_matches_f90(sp0, sp1, sp2):
    rng = np.random.default_rng(13)
    cfg = _random_centres(rng, _cfg())
    at, ti = _frame()
    alphas = rng.normal(size=cfg.gap3b_n_centres)
    q3 = gd.gap_raw_numpy(at, ti, cfg)[:, gd.gap_slices(cfg)["three_body"]].astype(np.float64)
    q3 = q3.reshape(len(at), -1, cfg.gap3b_n_centres)
    p = gd.pair_index(cfg.num_types)[sp1, sp2]
    E_ours = 2.0 * (alphas @ q3[ti == sp0, p].sum(0))
    E_f90 = _gap3b_energy_f90(at, ti, alphas, cfg.gap3b_centres, cfg.gap3b_sigma,
                              sp0, sp1, sp2, RCUT, BUFFER)
    assert abs(E_f90) > 1e-3
    np.testing.assert_allclose(E_ours, E_f90, rtol=1e-6)


# ── invariances / layout ────────────────────────────────────────────────

def test_rigid_motion_and_permutation_invariance():
    rng = np.random.default_rng(5)
    cfg = _random_centres(rng, _cfg())
    at, ti = _frame()
    q = gd.gap_raw_numpy(at, ti, cfg)
    assert q.dtype == np.float32 and q.shape == (len(at), gd.gap_dim(cfg))
    moved = at.copy()
    moved.rotate(37.0, rng.normal(size=3), rotate_cell=True)
    moved.translate(rng.normal(size=3))
    np.testing.assert_allclose(gd.gap_raw_numpy(moved, ti, cfg), q, rtol=1e-6, atol=1e-6)
    perm = rng.permutation(len(at))
    np.testing.assert_allclose(gd.gap_raw_numpy(at[perm], ti[perm], cfg), q[perm],
                               rtol=1e-6, atol=1e-6)


def test_pair_index_and_widths():
    for T in (1, 2, 3):
        pi = gd.pair_index(T)
        assert pi.shape == (T, T) and np.array_equal(pi, pi.T)
        assert sorted(pi[np.triu_indices(T)]) == list(range(T * (T + 1) // 2))
    cfg = _cfg()
    T, M2, M3 = cfg.num_types, cfg.gap2b_n_centres, cfg.gap3b_n_centres
    sl = gd.gap_slices(cfg)
    assert sl["two_body"] == slice(0, T * M2)
    assert sl["three_body"] == slice(T * M2, T * M2 + T * (T + 1) // 2 * M3)
    assert gd.gap_dim(cfg) == T * M2 + T * (T + 1) // 2 * M3


def test_f_cut_form():
    r = np.array([0.0, RCUT - BUFFER, RCUT - BUFFER / 2, RCUT, RCUT + 1])
    np.testing.assert_allclose(gd.f_cut(r, RCUT, BUFFER), [1.0, 1.0, 0.5, 0.0, 0.0], atol=1e-15)


# ── centre selection / validation ───────────────────────────────────────

def test_select_centres():
    cfg = _cfg(seed=3)
    at, _ = _frame()
    c2, c3 = gd.select_centres([at], cfg)
    assert c2.shape == (cfg.gap2b_n_centres,) and c2[0] == 0.5 and c2[-1] == RCUT
    assert c3.shape == (cfg.gap3b_n_centres, 3)
    assert len(np.unique(c3, axis=0)) == cfg.gap3b_n_centres
    assert np.all(c3[:, 0] >= 0) and np.all(c3[:, 0] <= 2 * RCUT)
    assert np.all(c3[:, 1] >= 0) and np.all(c3[:, 1] <= RCUT ** 2)
    assert np.all(c3[:, 2] >= 0) and np.all(c3[:, 2] <= 2 * RCUT)
    c2b, c3b = gd.select_centres([at], _cfg(seed=3))
    np.testing.assert_array_equal(c2b, c2)
    np.testing.assert_array_equal(c3b, c3)
    _, c3c = gd.select_centres([at], _cfg(seed=4))
    assert not np.array_equal(c3c, c3)


@pytest.mark.parametrize("bad", [
    {"gap2b_n_centres": 0}, {"gap3b_n_centres": -1},
    {"gap2b_sigma": 0.0}, {"gap3b_sigma": [0.5, -0.1, 0.5]}, {"gap3b_sigma": [0.5, 0.5]},
    {"gap_buffer": RCUT + 0.1}, {"gap_buffer": 0.0},
    {"gap2b_centres": [1.0, 2.0]}, {"gap3b_centres": [[1.0, 2.0]] * 9},
])
def test_validate_rejects(bad):
    cfg = _cfg(**bad)
    with pytest.raises(ValueError):
        gd.validate_gap_cfg(cfg)


def test_validate_accepts_default_and_with_centres():
    gd.validate_gap_cfg(_cfg())
    gd.validate_gap_cfg(_random_centres(np.random.default_rng(0), _cfg()))


# ── Task 2: nep_descriptor="gap_sum" — q2 ‖ q3 as the sum-model experts ─────
import tensorflow as tf
from data import prepare_eval_data
from test_neighbor_dipole import _model
from test_nep_descriptor import (EXPERTS, _check_four_path, _forward_args,
                                 _randomise_experts)
from test_q_scale import _predict_eff

_C2 = [0.6, 1.1, 1.6, 2.1, 2.7, 3.3]
_C3 = [[2.0, 0.1, 1.5], [3.0, 0.5, 2.0], [4.0, 1.0, 2.8], [5.0, 2.0, 3.1],
       [3.5, 0.2, 2.5], [4.5, 1.5, 3.4], [2.5, 0.3, 1.0], [5.5, 3.0, 2.2]]


def _gap_cfg(**kw):
    """gap_sum with explicit centres so no test needs the driver's selection."""
    return _cfg(**{"nep_descriptor": "gap_sum", "nep_num_neurons": 6,
                   "gap2b_n_centres": 6, "gap3b_n_centres": 8,
                   "gap2b_centres": list(_C2), "gap3b_centres": [list(c) for c in _C3], **kw})


def test_gap_sum_widths_and_data_dict():
    import nep_descriptor as nd
    from DescriptorBuilderGPU import nep_block_widths
    cfg = _gap_cfg(); at, ti = _frame()
    W2, W3 = 2 * 6, 3 * 8
    assert gd.gap_dim(cfg) == W2 + W3 == nd.nep_raw_width(cfg)
    assert nep_block_widths(cfg) == [W2, W3]
    d = prepare_eval_data([at, at[:12]], cfg)
    assert tuple(d["nep_raw"].shape) == (2, len(at), W2 + W3)
    np.testing.assert_array_equal(d["nep_raw"][0].numpy(), gd.gap_raw_numpy(at, ti, cfg))
    assert np.all(d["nep_raw"][1, 12:].numpy() == 0.0)
    assert "nep_pairs" not in d


def test_gap_sum_gen0_identity():
    from SNES import SNES
    at, _ = _frame()
    cfg_off, cfg_g = _cfg(), _gap_cfg()
    m_off, m_g = _model(cfg_off), _model(cfg_g)
    assert cfg_g.dim_q == cfg_off.dim_q and m_g.c_r is None and m_g.c_a is None
    for k in ("W0", "b0", "W1"):
        np.testing.assert_array_equal(getattr(m_g, k).numpy(), getattr(m_off, k).numpy())
    T, He, Hw = 2, 6, m_g._head_slots
    assert tuple(m_g.W0_r.shape) == (T, 12, He) and tuple(m_g.W0_a.shape) == (T, 24, He)
    assert tuple(m_g.W1_r.shape) == tuple(m_g.W1_a.shape) == (T, He, Hw)
    assert np.all(m_g.W1_r.numpy() == 0) and np.all(m_g.W1_a.numpy() == 0)
    d_off, d_g = prepare_eval_data([at], cfg_off), prepare_eval_data([at], cfg_g)
    np.testing.assert_allclose(_predict_eff(m_g, d_g), _predict_eff(m_off, d_off), rtol=1e-6)
    o_off, o_g = SNES(m_off), SNES(m_g)
    assert o_g.n_nep == 0
    n_off, n_g = (o._split_reconstructed(o.reconstruct_params_tf(o.mu)) for o in (o_off, o_g))
    for k in ("W0", "b0", "W1", "b1"):
        np.testing.assert_array_equal(n_g[k].numpy(), n_off[k].numpy())
    assert np.all(n_g["W1_r"].numpy() == 0) and np.all(n_g["W1_a"].numpy() == 0)
    np.testing.assert_allclose(o_g.validate(d_g, o_g.mu), o_off.validate(d_off, o_off.mu), rtol=1e-5)


def test_gap_sum_experts_add():
    cfg = _gap_cfg(); m = _model(cfg); at, _ = _frame()
    _randomise_experts(m)
    d = prepare_eval_data([at], cfg)
    W1, W1_r, W1_a = (getattr(m, k).numpy().copy() for k in ("W1", "W1_r", "W1_a"))

    def mu_with(**on):
        for k, v in (("W1", W1), ("W1_r", W1_r), ("W1_a", W1_a)):
            getattr(m, k).assign(v if on.get(k, True) else np.zeros_like(v))
        return _predict_eff(m, d)[0]

    mu_all = mu_with()
    mu_soap, mu_2b, mu_3b = (mu_with(W1_r=False, W1_a=False), mu_with(W1=False, W1_a=False),
                             mu_with(W1=False, W1_r=False))
    for v in (mu_soap, mu_2b, mu_3b):
        assert np.linalg.norm(v) > 1e-6
    np.testing.assert_allclose(mu_all, mu_soap + mu_2b + mu_3b, rtol=1e-5, atol=1e-6)


def test_gap_sum_four_path():
    cfg = _gap_cfg(dipole_readout="neighbor"); m = _model(cfg)
    _randomise_experts(m)
    _check_four_path(cfg, m)


def test_gap_sum_snes_layout_and_roundtrip():
    from SNES import SNES, _set_model_params
    cfg = _gap_cfg(dipole_readout="neighbor", dipole_species_pairs="symmetric")
    m = _model(cfg); opt = SNES(m); at, _ = _frame()
    T, H, He, Hw, Q = 2, cfg.num_neurons, 6, m._head_slots, cfg.dim_q
    n_main = T * (Q * H + H + H * Hw)
    n_r, n_a = T * (12 * He + He + He * Hw), T * (24 * He + He + He * Hw)
    assert opt.n_typed == n_main + n_r + n_a and opt.n_nep == 0
    assert opt._n_per_type == opt.n_typed // T
    assert opt.dim == opt.n_anns_total
    assert int(opt._extract_type_params(opt.mu, 0).shape[0]) == opt._n_per_type
    mu_vec = opt.mu + 0.05 * tf.random.stateless_normal(opt.mu.shape, [3, 4])
    _set_model_params(m, *opt.reconstruct_params_tf(mu_vec))
    n = opt._split_reconstructed(opt.reconstruct_params_tf(mu_vec))
    for k in EXPERTS:
        np.testing.assert_array_equal(getattr(m, k).numpy(), n[k].numpy())
    assert n["c_r"] is None and np.any(m.W1_r.numpy() != 0)
    full = prepare_eval_data([at], cfg)
    kw = dict(nep_raw=full["nep_raw"])
    mu_model = m.predict_batch(*_forward_args(m, full), W_atom=full["V_atom"], **kw)[0].numpy()
    nb = opt._split_reconstructed(opt.reconstruct_params_tf(mu_vec[tf.newaxis]))
    mu_cand = m.predict_batch_candidates(
        full["descriptors"], full["V_atom"], full["Z_int"], full["atom_mask"],
        nb["W0"], nb["b0"], nb["W1"], nb["b1"], c_pair=nb["c_pair"],
        experts=tuple(nb[k] for k in EXPERTS), **kw)[0, 0].numpy()
    np.testing.assert_allclose(mu_cand, mu_model, rtol=1e-5, atol=1e-6)
    assert np.any(mu_cand != 0.0)
    d = prepare_eval_data([at, at], cfg)
    mu_e = mu_vec.numpy().copy(); mu_e[n_main:opt.n_typed] *= 3.0
    fit = opt.evaluate_population(tf.stack([mu_vec, tf.constant(mu_e)]), d).numpy()
    assert np.all(np.isfinite(fit)) and fit[0] != fit[1]
    assert opt.validate(d, tf.constant(mu_e)) != opt.validate(d, mu_vec)


def test_gap_sum_save_load(tmp_path):
    import model_io
    cfg = _gap_cfg(); m = _model(cfg); at, _ = _frame()
    _randomise_experts(m)
    mu0 = _predict_eff(m, prepare_eval_data([at], cfg))
    p = str(tmp_path / "gap_sum.h5")
    model_io.save_model(m, cfg, path=p)
    loaded = model_io.load_model(p)
    lc = loaded.cfg
    assert lc.nep_descriptor == "gap_sum" and loaded.c_r is None
    np.testing.assert_allclose(lc.gap2b_centres, _C2)
    np.testing.assert_allclose(lc.gap3b_centres, _C3)
    for k in EXPERTS:
        np.testing.assert_array_equal(getattr(loaded, k).numpy(), getattr(m, k).numpy())
    np.testing.assert_allclose(_predict_eff(loaded, prepare_eval_data([at], lc)), mu0, atol=1e-6)


def test_gap_sum_scaled_block_is_gen0_reparameterisation():
    from DescriptorBuilderGPU import compute_q_scale, q_scale_groups
    from TNEP import gen0_ann_inputs
    at, _ = _frame(); small = at[:12]; small.info["dipole"] = np.zeros(3)
    cfg_off, cfg_on = _gap_cfg(), _gap_cfg(descriptor_scale="block")
    d = prepare_eval_data([at, small], cfg_off)
    ds, W2, W3 = cfg_off.dim_q, 12, 24
    q0 = gen0_ann_inputs(cfg_on, d)
    assert q0.shape == (2, len(at), ds + W2 + W3)
    np.testing.assert_array_equal(q0[:, :, ds:], d["nep_raw"].numpy())
    groups = q_scale_groups(cfg_on)
    assert sorted(np.concatenate(groups).tolist()) == list(range(ds + W2 + W3))
    assert len(groups) == len(q_scale_groups(_cfg(descriptor_scale="block"))) + 2
    m_np, mu_np = compute_q_scale(q0, d["atom_mask"].numpy(), cfg_on)
    cfg_on.q_scale, cfg_on.q_shift = [float(v) for v in m_np], [float(v) for v in mu_np]
    m_off, m_on = _model(cfg_off), _model(cfg_on)
    sl = gd.gap_slices(cfg_on); m_r, m_a = m_np[ds:][sl["two_body"]], m_np[ds:][sl["three_body"]]
    np.testing.assert_allclose(m_on.W0.numpy() * m_np[None, :ds, None], m_off.W0.numpy(), rtol=1e-6)
    for m in (m_off, m_on):
        _randomise_experts(m)
    for k, mk, s in (("r", m_r, sl["two_body"]), ("a", m_a, sl["three_body"])):
        getattr(m_on, f"W0_{k}").assign(getattr(m_off, f"W0_{k}").numpy() / mk[None, :, None])
        mu_k = (m_np * mu_np)[ds:][s]
        getattr(m_on, f"b0_{k}").assign(getattr(m_off, f"b0_{k}").numpy()
                                        + np.einsum('q,tqh->th', mu_k, getattr(m_on, f"W0_{k}").numpy()))
    np.testing.assert_allclose(_predict_eff(m_on, d), _predict_eff(m_off, d), rtol=1e-4, atol=1e-6)
    assert np.abs(_predict_eff(m_on, d)).max() > 0


def test_gap_sum_guards():
    from TNEP import TNEP
    with pytest.raises(NotImplementedError, match="descriptor gradients"):
        TNEP(_gap_cfg(dipole_readout="tnep"))
    with pytest.raises(ValueError, match="select_centres"):
        TNEP(_gap_cfg(gap2b_centres=None))
    with pytest.raises(ValueError, match="select_centres"):
        TNEP(_gap_cfg(gap3b_centres=None))
    with pytest.raises(ValueError, match="gap2b_sigma"):
        TNEP(_gap_cfg(gap2b_sigma=0.0))


def test_gap_sum_adam_smoke():
    from adam import _trainable, _ann_core
    cfg = _gap_cfg(optimizer="adam", adam_lr=1e-2, num_generations=2, val_interval=1,
                   batch_chunk_size=1, dipole_species_pairs="symmetric")
    m = _model(cfg); at, _ = _frame()
    keys = list(_trainable(m))
    assert all(k in keys for k in EXPERTS) and "c_r" not in keys
    assert len(_ann_core(m)) == 9
    d = prepare_eval_data([at, at], cfg)
    history, final, best = m.fit(d, d)
    assert np.all(np.isfinite(history["train_loss"]))
    for mdl in (final, best):
        assert np.any(mdl.W1_r.numpy() != 0) and np.any(mdl.W1_a.numpy() != 0)


def test_gap_sum_driver_centres():
    """select_centres → cfg → model: the driver's path without MasterTNEP."""
    cfg = _gap_cfg(gap2b_centres=None, gap3b_centres=None, seed=5)
    at, _ = _frame()
    c2, c3 = gd.select_centres([at, at[:12]], cfg)
    cfg.gap2b_centres, cfg.gap3b_centres = c2.tolist(), c3.tolist()
    m = _model(cfg)
    d = prepare_eval_data([at], cfg)
    assert np.all(np.isfinite(_predict_eff(m, d)))


# ── Task 3: GAP mode — gap_2b / gap_3b additive column blocks (gap_dipole) ──
from dataclasses import replace
from ase.io import read, write
import json
import gap_dipole
from test_gap_dipole import _cfg as _dip_cfg, _chdir, WATER_XYZ

GAP_KW = dict(gap_2b=True, gap_3b=True, gap2b_n_centres=4, gap3b_n_centres=6,
              gap_sparse_per_species=8)


def _six_frames(tmp_path, name="six.xyz", mu=None):
    frames = read(WATER_XYZ, index="0:6")
    for k, f in enumerate(frames):
        f.calc = None
        f.info["dipole"] = np.zeros(3) if mu is None else mu[k]
    p = str(tmp_path / name)
    write(p, frames)
    return frames, p


@pytest.fixture(scope="module")
def gap_fitted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gapmode")
    with _chdir(tmp):
        base = gap_dipole.fit_gap_dipole(_dip_cfg(**{**GAP_KW, "gap_2b": False, "gap_3b": False}))
        m = gap_dipole.fit_gap_dipole(_dip_cfg(**GAP_KW))
        rec = json.load(open(os.path.join("GAP", sorted(os.listdir("GAP"))[-1])))
    return base, m, rec


def _frame_inputs(m, at):
    ti = np.array([m.cfg.types.index(int(z)) for z in at.numbers])
    D = np.asarray(m.builder.build_descriptors([at], calc_gradients=False)[0][0], dtype=np.float64)
    v = gap_dipole._v_atoms(at, ti, m.cfg)
    raw = gd.gap_raw_numpy(at, ti, m.cfg).astype(np.float64)
    sl = gd.gap_slices(m.cfg)
    return ti, D, v, raw[:, sl["two_body"]], raw[:, sl["three_body"]]


def test_gapmode_column_count(gap_fitted):
    base, m, _ = gap_fitted
    cfg, T, K = m.cfg, m.cfg.num_types, 1
    sl = gd.gap_slices(cfg)
    W2, W3 = sl["two_body"].stop, sl["three_body"].stop - sl["two_body"].stop
    n_soap = sum(len(a) for a in base.alpha)
    assert [len(a) for a in m.alpha] == [len(a) for a in base.alpha]
    assert [len(a) for a in m.alpha_gap2b] == [W2 * K] * T
    assert [len(a) for a in m.alpha_gap3b] == [W3 * K] * T
    assert cfg.gap2b_centres is not None and cfg.gap3b_centres is not None
    at = read(WATER_XYZ, index="10")
    ti, D, v, q2, q3 = _frame_inputs(m, at)
    phi = gap_dipole._phi_frame([(D, m.sparse_desc)], v, ti, m.xi, (q2, q3))
    assert phi.shape == (3, n_soap + T * W2 * K + T * W3 * K)


def test_gapmode_phi_blocks_explicit_loop(gap_fitted):
    _, m, _ = gap_fitted
    at = read(WATER_XYZ, index="10")
    ti, D, v, q2, q3 = _frame_inputs(m, at)
    scales = [(D, m.sparse_desc)]
    soap = gap_dipole._phi_frame(scales, v, ti, m.xi)
    assert np.array_equal(soap, gap_dipole._phi_frame(scales, v, ti, m.xi, gap_q=None))
    full = gap_dipole._phi_frame(scales, v, ti, m.xi, (q2, q3))
    assert np.array_equal(full[:, :soap.shape[1]], soap)
    blocks = []
    for q in (q2, q3):
        for t in range(m.cfg.num_types):
            b = np.zeros((3, q.shape[1], v.shape[2]))
            for i in np.flatnonzero(ti == t):
                for s in range(q.shape[1]):
                    for k in range(v.shape[2]):
                        b[:, s, k] += q[i, s] * v[i, :, k]
            blocks.append(b.reshape(3, -1))
    assert np.allclose(full[:, soap.shape[1]:], np.concatenate(blocks, axis=1), rtol=1e-12, atol=1e-12)
    # 2b-only / 3b-only drop the other block
    only2 = gap_dipole._phi_frame(scales, v, ti, m.xi, (q2, None))
    assert np.array_equal(only2, full[:, :soap.shape[1] + sum(b.shape[1] for b in blocks[:2])])


@pytest.mark.parametrize("flags", [dict(gap_2b=True, gap_3b=False), dict(gap_2b=True, gap_3b=True)])
def test_gapmode_exact_recovery_known_alpha(tmp_path, flags):
    frames, p0 = _six_frames(tmp_path)
    kw = dict(total_N=None, gap_ridge_lambda=1e-12, **{**GAP_KW, **flags})
    with _chdir(tmp_path):
        base = gap_dipole.fit_gap_dipole(_dip_cfg(data_path=p0, **kw))
    assert (base.alpha_gap3b is None) == (not flags["gap_3b"])
    rng = np.random.default_rng(0)
    star = replace(base, alpha=[np.zeros_like(a) for a in base.alpha],
                   alpha_gap2b=[rng.normal(size=len(a)) for a in base.alpha_gap2b],
                   alpha_gap3b=None if base.alpha_gap3b is None
                   else [rng.normal(size=len(a)) for a in base.alpha_gap3b])
    mu_star = gap_dipole.predict(star, frames)
    _, p1 = _six_frames(tmp_path, "synth.xyz", mu_star)
    with _chdir(tmp_path):
        refit = gap_dipole.fit_gap_dipole(_dip_cfg(data_path=p1, **kw))
    assert np.allclose(gap_dipole.predict(refit, frames), mu_star,
                       atol=1e-8 * max(np.abs(mu_star).max(), 1.0))


def test_gapmode_predict_matches_fit_design(tmp_path):
    """predict() on the TRAINING frames reproduces the fit's own Φα (its
    recorded train RMSE) — guards the three coupled edits of plan §0.3."""
    mu = np.random.default_rng(3).normal(size=(6, 3))
    frames, p1 = _six_frames(tmp_path, "rand.xyz", mu)
    cfg = _dip_cfg(data_path=p1, total_N=None, **GAP_KW)
    with _chdir(tmp_path):
        m = gap_dipole.fit_gap_dipole(cfg)
        rec = json.load(open(os.path.join("GAP", os.listdir("GAP")[0])))
    pred = gap_dipole.predict(m, frames)
    rmse, _ = gap_dipole._rmse_r2(pred, gap_dipole._targets(frames, cfg))
    assert np.isclose(rmse, rec["metrics"]["train_rmse"], rtol=1e-8, atol=1e-12)
    # and the gap blocks are live: zeroing them changes the prediction
    off = replace(m, alpha_gap2b=[np.zeros_like(a) for a in m.alpha_gap2b],
                  alpha_gap3b=[np.zeros_like(a) for a in m.alpha_gap3b])
    assert not np.allclose(gap_dipole.predict(off, frames), pred, atol=1e-8)


def test_gapmode_run_record(gap_fitted):
    _, m, rec = gap_fitted
    g = rec["gap"]
    assert g["gap_2b"] is True and g["gap_3b"] is True
    assert g["gap2b_centres"] == m.cfg.gap2b_centres and len(g["gap2b_centres"]) == 4
    assert g["gap3b_centres"] == m.cfg.gap3b_centres and np.shape(g["gap3b_centres"]) == (6, 3)
    assert g["gap2b_sigma"] == m.cfg.gap2b_sigma and g["gap3b_sigma"] == list(m.cfg.gap3b_sigma)
    assert g["gap2b_rcut"] == m.cfg.gap2b_rcut and g["gap3b_rcut"] == m.cfg.gap3b_rcut
    assert g["gap_buffer"] == m.cfg.gap_buffer
    T = m.cfg.num_types
    assert g["n_gap_columns"] == {"two_body": T * 2 * 4, "three_body": T * 3 * 6}
    assert g["n_columns"] == sum(len(a) for a in m.alpha + m.alpha_gap2b + m.alpha_gap3b)


def test_gapmode_validate():
    gap_dipole._validate(_dip_cfg(gap_2b=True))                    # centres None: filled in fit
    gap_dipole._validate(_dip_cfg(gap_3b=True, gap2b_centres=[1.0] * 20))
    gap_dipole._validate(_dip_cfg(gap2b_sigma=0.0))                # flags off: gap fields unused
    with pytest.raises(ValueError, match="gap2b_sigma"):
        gap_dipole._validate(_dip_cfg(gap_2b=True, gap2b_sigma=0.0))
    with pytest.raises(ValueError, match="gap3b_centres"):
        gap_dipole._validate(_dip_cfg(gap_3b=True, gap3b_centres=[[1.0, 2.0]]))


def test_gapmode_flags_off_alpha_gap_fields_are_none(gap_fitted):
    base, _, _ = gap_fitted
    assert base.alpha_gap2b is None and base.alpha_gap3b is None


def test_gapmode_zero_columns_warn(tmp_path):
    """A 3b cutoff short enough to leave species-pair slots unpopulated gives
    identically-zero gap columns — unidentifiable coefficients, warned about."""
    with _chdir(tmp_path), pytest.warns(UserWarning, match="identically zero"):
        gap_dipole.fit_gap_dipole(_dip_cfg(**{**GAP_KW, "gap3b_rcut": 1.2}))
