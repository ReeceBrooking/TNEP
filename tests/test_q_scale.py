"""Descriptor scale preconditioning (cfg.descriptor_scale): the invariants."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import tensorflow as tf

import DescriptorBuilderGPU as DB
from DescriptorBuilderGPU import compute_q_scale, q_scale_groups
from test_radial_basis import _cfg, _cho_frames, _model, _eval_data, _predict

FULL = dict(radial_basis_size=8, radial_basis_per_l=True, radial_basis_two_body=True,
            radial_basis_four_body=True, radial_basis_five_body=True,
            descriptor_mixing=True, descriptor_mixing_regularizer="expm",
            dipole_species_pairs="symmetric")


def _q0(cfg, frames):
    m0 = _model(_cfg(**{k: v for k, v in vars(cfg).items() if k in FULL or k == "radial_basis_species_sum"}))
    d = _eval_data(cfg, frames)
    return m0.soap_from_raw(d["descriptors"], d["Z_int"]).numpy(), d["atom_mask"].numpy(), d


def test_compute_q_scale_rules(monkeypatch):
    cfg = _cfg(descriptor_scale="channel", **FULL)
    q, mask, _ = _q0(cfg, _cho_frames(6))
    m = compute_q_scale(q, mask, cfg)
    assert m.shape == (cfg.dim_q,) and np.all((m >= 0.1) & (m <= 10.0))
    # alpha = 0 -> no scaling; padded atoms ignored; dead channel -> 1; dust clamps.
    monkeypatch.setattr(DB, "Q_SCALE_ALPHA", 0.0)
    np.testing.assert_array_equal(compute_q_scale(q, mask, cfg), np.ones(cfg.dim_q, np.float32))
    monkeypatch.setattr(DB, "Q_SCALE_ALPHA", 0.5)
    q2 = q.copy(); q2[mask == 0] = 1e9
    np.testing.assert_array_equal(compute_q_scale(q2, mask, cfg), m)
    q3 = q.copy(); q3[..., 0] = 0.0; q3[..., 1] = q3[..., 1].mean() + 1e-7 * np.random.default_rng(0).normal(size=q3[..., 1].shape)
    m3 = compute_q_scale(q3, mask, cfg)
    assert m3[0] == 1.0 and m3[1] == 10.0
    # block mode: constant within each group, groups cover every channel once.
    cfg_b = _cfg(descriptor_scale="block", **FULL)
    groups = q_scale_groups(cfg_b)
    assert sorted(np.concatenate(groups).tolist()) == list(range(cfg_b.dim_q))
    mb = compute_q_scale(q, mask, cfg_b)
    for g in groups:
        assert np.allclose(mb[g], mb[g][0])
    print("\nblock scales:", np.round([mb[g][0] for g in groups], 3))


@pytest.mark.parametrize("mode", ["block", "channel"])
def test_scaling_is_a_reparameterisation_identical_at_gen0(mode):
    frames = _cho_frames(6)
    cfg_off = _cfg(**FULL)
    q, mask, data = _q0(cfg_off, frames)
    cfg_on = _cfg(descriptor_scale=mode, **FULL)
    cfg_on.q_scale = [float(v) for v in compute_q_scale(q, mask, cfg_on)]
    m_off, m_on = _model(cfg_off), _model(cfg_on)
    m = np.asarray(cfg_on.q_scale, np.float32)
    # I3: same seed -> stored W0 differ by 1/m, effective weights and predictions identical.
    np.testing.assert_allclose(m_on.W0.numpy() * m[None, :, None], m_off.W0.numpy(), rtol=1e-6)
    np.testing.assert_allclose(_predict_eff(m_on, data), _predict_eff(m_off, data), rtol=1e-5, atol=1e-7)
    mu_on, mu_off = m_on.optimizer.mu.numpy(), m_off.optimizer.mu.numpy()
    assert abs(m_on.optimizer.validate(data, tf.constant(mu_on)) - m_off.optimizer.validate(data, tf.constant(mu_off))) < 1e-6
    # I2: any scaled model == unscaled model with W0' = m * W0 (mixing on, arbitrary weights).
    rng = np.random.default_rng(1)
    W = rng.normal(size=m_on.W0.shape).astype(np.float32)
    V = 0.1 * rng.normal(size=m_on.U_pair.shape).astype(np.float32)
    m_on.W0.assign(W); m_off.W0.assign(W * m[None, :, None])
    m_on.U_pair.assign(V); m_off.U_pair.assign(V)
    np.testing.assert_allclose(_predict_eff(m_on, data), _predict_eff(m_off, data), rtol=1e-5, atol=1e-7)
    # candidates path agrees with the single path on the scaled model.
    snes = m_on.optimizer
    pv = tf.stack([snes.mu, snes.mu + 0.02])
    named = snes._split_reconstructed(snes.reconstruct_params_tf(pv))
    preds = m_on.predict_batch_candidates(
        data["descriptors"], data["V_atom"], data["Z_int"], data["atom_mask"],
        named["W0"], named["b0"], named["W1"], named["b1"],
        U_pair=named["U_pair"], c_pair=named["c_pair"], M_radial=named["M_radial"]).numpy()
    one = snes._split_reconstructed(snes.reconstruct_params_tf(pv[1]))
    single = m_on.predict_batch(
        data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
        data["pair_struct"], data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
        m_on._W0_eff(one["W0"], one["U_pair"]), one["b0"], one["W1"], one["b1"],
        W_atom=data["V_atom"], c_pair=one["c_pair"], M_radial=one["M_radial"]).numpy()
    np.testing.assert_allclose(preds[1], single, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError):
        _model(_cfg(descriptor_scale=mode, **FULL))     # q_scale not computed


def _predict_eff(model, data):
    return model.predict_batch(
        data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
        data["pair_struct"], data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
        model._W0_eff(model.W0), model.b0, model.W1, model.b1,
        W_atom=data["V_atom"]).numpy()


def test_unnormalised_spectrum_option():
    """normalize=False: same direction as the normalised descriptor per atom,
    O(1) magnitudes carrying coordination, extras unchanged, scaler finite."""
    frames = _cho_frames(6)
    cfg_n = _cfg(**FULL)
    cfg_u = _cfg(radial_basis_normalize=False, **FULL)
    m_n, m_u = _model(cfg_n), _model(cfg_u)
    data = _eval_data(cfg_n, frames)
    qn = m_n.soap_from_raw(data["descriptors"], data["Z_int"]).numpy()
    qu = m_u.soap_from_raw(data["descriptors"], data["Z_int"]).numpy()
    n_extra = 3 * 12
    mask = data["atom_mask"].numpy() > 0
    soap_n, soap_u = qn[mask][:, :-n_extra], qu[mask][:, :-n_extra]
    norms = np.linalg.norm(soap_u, axis=1, keepdims=True)
    np.testing.assert_allclose(soap_u / norms, soap_n, rtol=1e-4, atol=1e-6)     # same shape
    assert 1e-3 < np.median(norms) < 10.0 and norms.std() / norms.mean() > 0.05  # O(1), varies with environment
    np.testing.assert_allclose(qu[..., -n_extra:], qn[..., -n_extra:], rtol=1e-6)  # extras untouched
    assert np.all(qu[~mask] == 0.0)
    m = compute_q_scale(qu, data["atom_mask"].numpy(), _cfg(descriptor_scale="channel",
                                                             radial_basis_normalize=False, **FULL))
    assert np.all(np.isfinite(m)) and m.shape == (cfg_u.dim_q,)
    print("\nunnormalised SOAP norms: median %.3f  min %.3f  max %.3f" % (np.median(norms), norms.min(), norms.max()))
    with pytest.raises(ValueError):
        _model(_cfg(radial_basis_normalize=False))
