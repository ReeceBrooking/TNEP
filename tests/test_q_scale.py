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
from TNEP import gen0_ann_inputs
from test_radial_basis import _cfg, _cho_frames, _model, _eval_data, _predict

FULL = dict(radial_basis_size=8, radial_basis_per_l=True,
            descriptor_mixing=True, descriptor_mixing_regularizer="expm",
            dipole_species_pairs="symmetric")


def _q0(cfg, frames):
    d = _eval_data(cfg, frames)
    return gen0_ann_inputs(cfg, d), d["atom_mask"].numpy(), d


def _Z(d):
    return d["Z_int"].numpy()


def test_compute_q_scale_rules(monkeypatch):
    cfg = _cfg(descriptor_scale="channel", **FULL)
    q, mask, d = _q0(cfg, _cho_frames(6))
    Z = _Z(d)
    m, mu = compute_q_scale(q, Z, mask, cfg)
    mb, _ = compute_q_scale(q, Z, mask, _cfg(descriptor_scale="block", **FULL))
    assert m.shape == mu.shape == (cfg.num_types, cfg.dim_q)
    groups = q_scale_groups(cfg)
    assert sorted(np.concatenate(groups).tolist()) == list(range(cfg.dim_q))
    present = [t for t in range(cfg.num_types) if np.any((mask > 0) & (Z == t))]
    assert len(present) >= 2
    for t in present:                       # each row from its own centre species only
        x = q[(mask > 0) & (Z == t)]
        np.testing.assert_allclose(mu[t], x.mean(0), rtol=1e-5, atol=1e-7)   # centring = channel mean
        s = x.std(0)
        g = np.ones(cfg.dim_q)
        for idx in groups:
            g[idx] = 1.0 / np.median(s[idx][s[idx] > 0])                     # group median std -> 1 ...
        live = s > 0
        kb = mb[t][live] / g[live]
        np.testing.assert_allclose(kb, kb[0], rtol=1e-5)                     # ... block = that, times ONE constant
        r = m[t][live] / g[live]; r = r / np.exp(np.median(np.log(r)))
        assert np.all((r >= 0.1 / 10 - 1e-6) & (r <= 10.0 * 10 + 1e-6))      # residual bounded by the clamp
        for mm in (m[t], mb[t]):                                             # unit-norm input for the ANN
            np.testing.assert_allclose(np.median(np.linalg.norm((x - mu[t]) * mm, axis=1)), 1.0, rtol=1e-4)
    assert not np.allclose(mb[present[0]], mb[present[1]])                   # species really differ
    # padded atoms ignored; zero-std channel -> 1; dust residual clamps at 10 on top of its group factor.
    q2 = q.copy(); q2[mask == 0] = 1e9
    np.testing.assert_array_equal(compute_q_scale(q2, Z, mask, cfg)[0], m)
    q3 = q.copy(); q3[..., 0] = 0.0
    q3[..., 1] = q3[..., 1].mean() + 1e-7 * np.random.default_rng(0).normal(size=q3[..., 1].shape)
    m3, _ = compute_q_scale(q3, Z, mask, cfg)
    gb3, _ = compute_q_scale(q3, Z, mask, _cfg(descriptor_scale="block", **FULL))   # group factor alone
    t = present[0]
    assert m3[t, 0] == 1.0
    # the dust channel is boosted far more than an ordinary one, limited by the clamp
    # (compare ratios within each mode: both carry their own overall norm constant)
    assert (m3[t, 1] / m3[t, 2]) / (gb3[t, 1] / gb3[t, 2]) > 5.0
    # a species with no atoms keeps the identity row
    absent = compute_q_scale(q, np.where(Z == present[0], present[1], Z), mask, cfg)
    assert np.all(absent[0][present[0]] == 1.0) and np.all(absent[1][present[0]] == 0.0)
    print("\nblock scales per species:", [np.round([mb[t][gr][0] for gr in groups], 3) for t in present])


@pytest.mark.parametrize("mode", ["block", "channel"])   # minmax needs a NEP block: test_nep_descriptor
def test_scaling_folds_into_W0_and_starts_unsaturated(mode):
    frames = _cho_frames(6)
    cfg_off = _cfg(**FULL)
    q, mask, data = _q0(cfg_off, frames)
    cfg_on = _cfg(descriptor_scale=mode, **FULL)
    m_np, mu_np = compute_q_scale(q, _Z(data), mask, cfg_on)
    cfg_on.q_scale, cfg_on.q_shift = m_np.tolist(), mu_np.tolist()
    m_off, m_on = _model(cfg_off), _model(cfg_on)
    m = np.asarray(cfg_on.q_scale, np.float32)
    # I3: the stored init is the plain Glorot / zero one whatever the scaler (never
    # divided by m), in the Keras variables AND in the SNES mean; it acts on the
    # scaled, centred input, so the tanh layer does not start saturated.
    np.testing.assert_array_equal(m_on.W0.numpy(), m_off.W0.numpy())
    assert np.abs(m_on.b0.numpy()).max() == 0 and np.abs(m_off.b0.numpy()).max() == 0
    np.testing.assert_array_equal(m_on.optimizer.mu.numpy(), m_off.optimizer.mu.numpy())
    real = mask > 0
    x = (q[real] - mu_np[_Z(data)[real]]) * m_np[_Z(data)[real]]
    g0 = m_on.optimizer._split_reconstructed(m_on.optimizer.reconstruct_params_tf(m_on.optimizer.mu))
    pre = np.einsum('nq,nqh->nh', x, g0["W0"].numpy()[_Z(data)[real]])
    assert np.mean(np.abs(np.tanh(pre)) > 0.99) < 0.2
    assert 0.3 < np.median(np.linalg.norm(x, axis=1)) < 5.0      # the input vector has norm ~1
    # I2: any scaled model == unscaled model with W0' = m*W0, b0' = b0 - (m*mu).W0 (mixing on).
    rng = np.random.default_rng(1)
    W = rng.normal(size=m_on.W0.shape).astype(np.float32); B = rng.normal(size=m_on.b0.shape).astype(np.float32)
    V = 0.1 * rng.normal(size=m_on.U_pair.shape).astype(np.float32)
    m_on.W0.assign(W); m_on.b0.assign(B); m_on.U_pair.assign(V)
    m_off.W0.assign(W * m[:, :, None]); m_off.b0.assign(B - np.einsum('tq,tqh->th', m * mu_np, W)); m_off.U_pair.assign(V)
    np.testing.assert_allclose(_predict_eff(m_on, data), _predict_eff(m_off, data), rtol=1e-4, atol=1e-6)
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
        m_on._W0_eff(one["W0"], one["U_pair"]), m_on._b0_eff(one["b0"], one["W0"]), one["W1"], one["b1"],
        W_atom=data["V_atom"], c_pair=one["c_pair"], M_radial=one["M_radial"]).numpy()
    np.testing.assert_allclose(preds[1], single, rtol=1e-4, atol=1e-6)
    with pytest.raises(ValueError):
        _model(_cfg(descriptor_scale=mode, **FULL))     # q_scale not computed


def _predict_eff(model, data):
    return model.predict_batch(
        data["descriptors"], data["grad_values"], data["pair_atom"], data["pair_gidx"],
        data["pair_struct"], data["positions"], data["Z_int"], data["boxes"], data["atom_mask"],
        model._W0_eff(model.W0), model._b0_eff(model.b0, model.W0), model.W1, model.b1,
        W_atom=data.get("V_atom"), nep_raw=data.get("nep_raw")).numpy()


@pytest.mark.parametrize("norm", ["none", "sqrt"])
def test_norm_options(norm):
    """none: same direction as l2, O(1) norms varying with environment.
    sqrt: degree 1 (doubling the coefficients doubles q), O(1)."""
    frames = _cho_frames(6)
    cfg_n = _cfg(**FULL)
    cfg_u = _cfg(radial_basis_norm=norm, **FULL)
    m_n, m_u = _model(cfg_n), _model(cfg_u)
    data = _eval_data(cfg_n, frames)
    qn = m_n.soap_from_raw(data["descriptors"], data["Z_int"]).numpy()
    qu = m_u.soap_from_raw(data["descriptors"], data["Z_int"]).numpy()
    mask = data["atom_mask"].numpy() > 0
    soap_n, soap_u = qn[mask], qu[mask]
    norms = np.linalg.norm(soap_u, axis=1)
    if norm == "none":
        np.testing.assert_allclose(soap_u / norms[:, None], soap_n, rtol=1e-4, atol=1e-6)
    else:
        q2 = m_u.soap_from_raw(2.0 * data["descriptors"], data["Z_int"]).numpy()[mask]
        np.testing.assert_allclose(q2, 2.0 * soap_u, rtol=1e-4, atol=1e-6)      # degree 1
        np.testing.assert_allclose(np.sign(soap_u), np.sign(soap_n), atol=0)      # signs kept
    assert 1e-2 < np.median(norms) < 10.0 and norms.std() / norms.mean() > 0.03
    assert np.all(qu[~mask] == 0.0)
    m, mu = compute_q_scale(qu, _Z(data), data["atom_mask"].numpy(), _cfg(descriptor_scale="channel", radial_basis_norm=norm, **FULL))
    assert np.all(np.isfinite(m)) and np.all(np.isfinite(mu))
    print(f"\n{norm}: SOAP norms median {np.median(norms):.3f} range {norms.min():.3f}-{norms.max():.3f}; l0/l4 mean|q| ratio "
          f"{np.abs(soap_u[:, :10]).mean() / np.abs(soap_u[:, 40:50]).mean():.1f}")
    # Fixed builder: "none" is supported (tests/test_nep_descriptor.py), "sqrt" is not.
    if norm == "sqrt":
        with pytest.raises(ValueError):
            _model(_cfg(radial_basis_norm=norm))
    else:
        _model(_cfg(radial_basis_norm=norm))
