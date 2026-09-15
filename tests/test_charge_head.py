import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import pytest
import numpy as np
import tensorflow as tf

from test_second_layer import _tiny_cfg, _build

MOL_BOX = tf.zeros([2, 3, 3])  # ignored by the q·r readout


def _charge_cfg(kspace=False, **kw):
    cfg = _tiny_cfg(**kw)
    cfg.charge_head = True
    cfg.charge_head_kspace = kspace
    return cfg


def _rand_geometry(seed=0, B=2, A=5):
    g = tf.random.Generator.from_seed(seed)
    q = g.normal([B, A])                      # deliberately non-neutral
    pos = g.normal([B, A, 3]) * 3.0
    mask = tf.ones([B, A])
    return q, pos, mask


# ------------------------------------------------------------------ #
#  Readout-helper symmetry tests (no dataset needed)                  #
# ------------------------------------------------------------------ #

def test_molecular_translation_invariance():
    m, _, _ = _build(_charge_cfg(num_hidden_layers=1, mixing=False))
    q, pos, mask = _rand_geometry()
    mu0 = m._charge_dipole_from_q(q, pos, MOL_BOX, mask)
    mu1 = m._charge_dipole_from_q(q, pos + tf.constant([5.0, -3.0, 11.0]),
                                  MOL_BOX, mask)
    assert np.allclose(mu0.numpy(), mu1.numpy(), atol=1e-4)


def test_molecular_rotation_equivariance():
    m, _, _ = _build(_charge_cfg(num_hidden_layers=1, mixing=False))
    q, pos, mask = _rand_geometry(seed=1)
    th = 0.83
    R = tf.constant([[np.cos(th), -np.sin(th), 0.0],
                     [np.sin(th), np.cos(th), 0.0],
                     [0.0, 0.0, 1.0]], tf.float32)
    mu0 = m._charge_dipole_from_q(q, pos, MOL_BOX, mask)
    mu_rot = m._charge_dipole_from_q(q, tf.einsum('ij,baj->bai', R, pos),
                                     MOL_BOX, mask)
    assert np.allclose(mu_rot.numpy(),
                       tf.einsum('ij,bj->bi', R, mu0).numpy(), atol=1e-4)


def test_periodic_lattice_translation_single_valued():
    """Shifting ONE atom by a lattice vector must not change P(k) — the
    property Σq·r lacks and the k-space readout exists to provide."""
    m, _, _ = _build(_charge_cfg(kspace=True, num_hidden_layers=1,
                                 mixing=False))
    q, pos, mask = _rand_geometry(seed=2)
    L = 12.5
    boxes = tf.eye(3, batch_shape=[2]) * L
    shift = tf.tensor_scatter_nd_update(
        tf.zeros([2, 5, 3]), [[0, 3, 0]], [L])   # atom 3 of struct 0, +L in x
    mu0 = m._charge_dipole_from_q(q, pos, boxes, mask)
    mu1 = m._charge_dipole_from_q(q, pos + shift, boxes, mask)
    assert np.allclose(mu0.numpy(), mu1.numpy(), atol=1e-4)


def test_kspace_matches_qr_in_large_box_limit():
    mk, _, _ = _build(_charge_cfg(kspace=True, num_hidden_layers=1,
                                  mixing=False))
    q, pos, mask = _rand_geometry(seed=4)
    pos = pos / 3.0                              # ~1 Å scale, kr ≈ 0.01
    big = tf.eye(3, batch_shape=[2]) * 900.0
    mu_k = mk._charge_dipole_from_q(q, pos, big, mask)
    mm, _, _ = _build(_charge_cfg(num_hidden_layers=1, mixing=False))
    mu_r = mm._charge_dipole_from_q(q, pos, MOL_BOX, mask)
    # sin-expansion error is O((kr)²) ≈ 1e-4 relative here.
    assert np.allclose(mu_k.numpy(), mu_r.numpy(), atol=1e-3)


def test_non_diagonal_periodic_box_raises():
    m, _, _ = _build(_charge_cfg(kspace=True, num_hidden_layers=1,
                                 mixing=False))
    q, pos, mask = _rand_geometry(seed=5)
    tilted = tf.eye(3, batch_shape=[2]) * 10.0 + tf.scatter_nd(
        [[0, 0, 1]], [1.5], [2, 3, 3])
    with pytest.raises(tf.errors.InvalidArgumentError):
        m._charge_dipole_from_q(q, pos, tilted, mask)


def test_padding_atoms_do_not_contribute():
    m, _, _ = _build(_charge_cfg(num_hidden_layers=1, mixing=False))
    q, pos, _ = _rand_geometry(seed=6)
    mask = tf.constant([[1.0, 1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0, 0.0]])
    mu0 = m._charge_dipole_from_q(q, pos, MOL_BOX, mask)
    # Corrupt padded entries: result must be unchanged.
    q2 = q + (1.0 - mask) * 100.0
    pos2 = pos + (1.0 - mask)[:, :, tf.newaxis] * 77.0
    mu1 = m._charge_dipole_from_q(q2, pos2, MOL_BOX, mask)
    assert np.allclose(mu0.numpy(), mu1.numpy(), atol=1e-4)


# ------------------------------------------------------------------ #
#  End-to-end model / optimizer-layout tests                          #
# ------------------------------------------------------------------ #

def test_full_prediction_translation_invariant():
    cfg = _charge_cfg(num_hidden_layers=1, mixing=False)
    model, train, _ = _build(cfg)
    for v in (model.W0, model.b0, model.W1, model.W1_q, model.b_q):
        v.assign(tf.random.stateless_normal(v.shape, [3, 4], stddev=0.3))
    from Adam import Adam
    opt = Adam(model); opt._precompute_W_atom(train)

    def _pred(positions):
        return model.predict_batch(
            train["descriptors"], train["grad_values"], train["pair_atom"],
            train["pair_gidx"], train["pair_struct"], positions,
            train["Z_int"], train["boxes"], train["atom_mask"],
            model.W0, model.b0, model.W1, model.b1,
            W_atom=train["_W_atom"], W1_q=model.W1_q, b_q=model.b_q)

    p0 = _pred(train["positions"])
    p1 = _pred(train["positions"] + tf.constant([3.0, -7.0, 2.0]))
    assert np.allclose(p0.numpy(), p1.numpy(), atol=1e-3)
    # And the head actually changes the prediction vs zeroed head weights:
    model.W1_q.assign(tf.zeros_like(model.W1_q))
    assert not np.allclose(p0.numpy(), _pred(train["positions"]).numpy(),
                           atol=1e-6)


def test_missing_charge_weights_raise():
    cfg = _charge_cfg(num_hidden_layers=1, mixing=False)
    model, train, _ = _build(cfg)
    from Adam import Adam
    Adam(model)._precompute_W_atom(train)
    with pytest.raises(ValueError, match="W1_q"):
        model.predict_batch(
            train["descriptors"], train["grad_values"], train["pair_atom"],
            train["pair_gidx"], train["pair_struct"], train["positions"],
            train["Z_int"], train["boxes"], train["atom_mask"],
            model.W0, model.b0, model.W1, model.b1,
            W_atom=train["_W_atom"])


def test_snes_layout_and_candidates_parity():
    """Genome → reconstruct → _set_model_params → predict_batch must equal
    the batched predict_batch_candidates path on the same flat vector.
    Mixing ON so the U_pair tail sits after the charge block."""
    cfg = _charge_cfg(num_hidden_layers=1, mixing=True, optimizer="snes")
    model, train, _ = _build(cfg)
    from SNES import SNES, _set_model_params
    from Adam import Adam
    opt = SNES(model)
    rng = np.random.default_rng(0)
    pv = rng.normal(0.0, 0.2, size=opt.dim).astype(np.float32)
    pv_tf = tf.constant(pv)
    # Per-type slice length and type-of-variable map must match bookkeeping.
    assert int(opt._extract_type_params(pv_tf, 0).shape[0]) == opt._n_per_type
    assert opt._build_type_of_variable().shape[0] == opt.dim

    params = opt.reconstruct_params_tf(pv_tf)
    _set_model_params(model, *params)
    Adam(model)._precompute_W_atom(train)   # reuse W_atom precompute helper
    W0_eff = model._W0_eff(model.W0)
    preds_single = model.predict_batch(
        train["descriptors"], train["grad_values"], train["pair_atom"],
        train["pair_gidx"], train["pair_struct"], train["positions"],
        train["Z_int"], train["boxes"], train["atom_mask"],
        W0_eff, model.b0, model.W1, model.b1,
        W_atom=train["_W_atom"], W1_q=model.W1_q, b_q=model.b_q)

    batched = opt.reconstruct_params_tf(pv_tf[tf.newaxis])
    named = opt._split_reconstructed(batched)
    preds_cand = model.predict_batch_candidates(
        train["descriptors"], train["_W_atom"], train["Z_int"],
        train["atom_mask"], named["W0"], named["b0"], named["W1"],
        named["b1"], U_pair=named["U_pair"],
        positions=train["positions"], boxes=train["boxes"],
        W1_q=named["W1_q"], b_q=named["b_q"])
    assert np.allclose(preds_single.numpy(), preds_cand[0].numpy(),
                       atol=1e-4, rtol=1e-4)


def test_snes_trains_with_charge_head():
    cfg = _charge_cfg(num_hidden_layers=1, mixing=True, optimizer="snes")
    model, train, val = _build(cfg)
    from SNES import SNES
    history, final_model, best = SNES(model).fit(train, val)
    assert np.all(np.isfinite(history["val_loss"]))
    m, _ = best.score(val)
    assert np.isfinite(float(m["rmse"]))


def test_adam_trains_with_charge_head():
    cfg = _charge_cfg(num_hidden_layers=2, mixing=True)
    cfg.num_generations = 30
    cfg.adam_learning_rate = 1e-2
    model, train, val = _build(cfg)
    from Adam import Adam
    opt = Adam(model)
    l0 = float(opt._batch_loss(train))
    history, final_model, best = opt.fit(train, val)
    assert float(opt._batch_loss(train)) < l0
    assert final_model.W1_q is not None
    m, _ = best.score(val)
    assert np.isfinite(float(m["rmse"]))


def _synthetic_charges(batch):
    """Type-dependent fake Mulliken labels, masked."""
    per_type = tf.constant([-0.8, 0.4, 0.1, 0.3])
    return tf.gather(per_type, batch["Z_int"]) * batch["atom_mask"]


def test_return_charges_matches_manual():
    cfg = _charge_cfg(num_hidden_layers=1, mixing=False)
    model, train, _ = _build(cfg)
    from Adam import Adam
    opt = Adam(model); opt._precompute_W_atom(train)
    _, q = opt._forward(train, return_charges=True)
    # Manual: q = h·W1_q[t] + b_q[t] with h the (masked) hidden layer.
    Z = train["Z_int"]
    z1 = tf.add_n([
        tf.einsum('baq,qh->bah', train["descriptors"], model.W0[t])
        * tf.cast(tf.equal(Z, t), tf.float32)[:, :, tf.newaxis]
        for t in range(model.num_types)]) + tf.gather(model.b0, Z)
    h = model.activation(z1) * train["atom_mask"][:, :, tf.newaxis]
    q_ref = (tf.reduce_sum(h * tf.gather(model.W1_q, Z), axis=2)
             + tf.gather(model.b_q, Z))
    assert np.allclose(q.numpy(), q_ref.numpy(), atol=1e-5)


def test_charge_aux_training_reduces_charge_error():
    cfg = _charge_cfg(num_hidden_layers=1, mixing=False)
    cfg.charge_aux_weight = 1.0
    cfg.num_generations = 60
    cfg.adam_learning_rate = 1e-2
    model, train, val = _build(cfg)
    train["charges"] = _synthetic_charges(train)
    val["charges"] = _synthetic_charges(val)
    from Adam import Adam
    opt = Adam(model)
    opt._precompute_W_atom(train)

    def charge_mse():
        _, q = opt._forward(train, return_charges=True)
        mask = train["atom_mask"]
        return float(tf.reduce_sum(tf.square(q - train["charges"]) * mask)
                     / tf.reduce_sum(mask))

    e0 = charge_mse()
    opt.fit(train, val)
    assert charge_mse() < e0


def test_aux_without_charge_head_raises():
    cfg = _tiny_cfg(num_hidden_layers=1, mixing=False)
    cfg.charge_head = False
    cfg.charge_aux_weight = 0.1
    with pytest.raises(ValueError, match="charge_head"):
        _build(cfg)


def test_aux_without_charge_labels_raises():
    cfg = _charge_cfg(num_hidden_layers=1, mixing=False)
    cfg.charge_aux_weight = 0.1
    model, train, val = _build(cfg)   # dataset has no charges column
    from Adam import Adam
    with pytest.raises(ValueError, match="charges"):
        Adam(model).fit(train, val)


def test_model_io_roundtrip_charge_head(tmp_path):
    import model_io
    cfg = _charge_cfg(num_hidden_layers=1, mixing=False)
    model, train, _ = _build(cfg)
    for v in (model.W0, model.b0, model.W1, model.W1_q, model.b_q):
        v.assign(tf.random.stateless_normal(v.shape, [5, 6], stddev=0.3))
    from Adam import Adam
    Adam(model)._precompute_W_atom(train)

    def _pred(m):
        return m.predict_batch(
            train["descriptors"], train["grad_values"], train["pair_atom"],
            train["pair_gidx"], train["pair_struct"], train["positions"],
            train["Z_int"], train["boxes"], train["atom_mask"],
            m.W0, m.b0, m.W1, m.b1,
            W_atom=train["_W_atom"], W1_q=m.W1_q, b_q=m.b_q).numpy()

    before = _pred(model)
    p = str(tmp_path / "m.h5")
    model_io.save_model(model, cfg, path=p)
    loaded = model_io.load_model(p)
    assert loaded.charge_head and loaded.W1_q is not None
    assert np.allclose(before, _pred(loaded), atol=1e-6)
