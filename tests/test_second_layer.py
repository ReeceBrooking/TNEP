import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import pytest
import numpy as np
import tensorflow as tf

# Reuse the tiny-model fixture pattern from tests/test_adam.py, adding num_hidden_layers.
def _tiny_cfg(num_hidden_layers=2, target_mode=1, optimizer="adam", mixing=True):
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    cfg.data_path = 'datasets/test.xyz'; cfg.test_data_path = None
    cfg.allowed_species = [6, 1, 7, 8]; cfg.filter_mode = 'subset'
    cfg.target_mode = target_mode; cfg.dipole_units = 'e*bohr'
    cfg.scale_targets = True; cfg.convert_dipole_to_eangstrom = False
    cfg.total_N = 16; cfg.test_ratio = 0.25
    cfg.num_neurons = 8; cfg.num_hidden_layers = num_hidden_layers
    cfg.descriptor_mode = 0; cfg.descriptor_mixing = mixing
    cfg.descriptor_mixing_regularizer = "expm"; cfg.descriptor_preprocess_contract = "off"
    cfg.pop_size = 8; cfg.num_generations = 3; cfg.batch_size = None
    cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.save_path = None; cfg.checkpoint_interval = None
    cfg.lambda_1 = 0.0; cfg.lambda_2 = 0.0; cfg.seed = 0
    cfg.val_interval = 1; cfg.val_size = None
    cfg.toggle_regularization = False; cfg.per_type_regularization = False
    cfg.dipole_rij_power = 0; cfg.optimizer = optimizer
    return cfg

def _build(cfg):
    from data import collect, split, pad_and_stack
    from DescriptorBuilderGPU import compute_dim_q
    from TNEP import TNEP
    dataset, ti = collect(cfg); cfg.randomise(dataset); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    return TNEP(cfg), train, val

def test_config_default_is_one_layer():
    from TNEPconfig import TNEPconfig
    assert TNEPconfig().num_hidden_layers == 1

def test_guard_num_hidden_layers_range():
    with pytest.raises(ValueError):
        _build(_tiny_cfg(num_hidden_layers=3))

def test_guard_mode2_rejected():
    with pytest.raises(ValueError):
        _build(_tiny_cfg(num_hidden_layers=2, target_mode=2))

def test_guard_snes_rejected():
    # SNES + 2 layers must raise at TNEP construction (guard reads cfg.optimizer).
    with pytest.raises(ValueError):
        _build(_tiny_cfg(num_hidden_layers=2, optimizer="snes"))

def test_wh_bh_created_with_correct_shapes():
    cfg = _tiny_cfg(num_hidden_layers=2, mixing=False)
    model, _, _ = _build(cfg)
    T, H = cfg.num_types, cfg.num_neurons
    assert tuple(model.Wh.shape) == (T, H, H)
    assert tuple(model.bh.shape) == (T, H)

def test_wh_bh_absent_when_one_layer():
    model, _, _ = _build(_tiny_cfg(num_hidden_layers=1, mixing=False))
    assert getattr(model, "Wh", None) is None
    assert getattr(model, "bh", None) is None


def _autodiff_reference_dipole(model, batch):
    """Reference dipole using AUTODIFF de_dq through a plain 2-layer energy,
    contracted with the SAME precomputed W_atom that predict_batch uses."""
    m = model
    desc = tf.identity(batch["descriptors"])          # [B,A,Q]
    Z = batch["Z_int"]; amask = batch["atom_mask"]
    W0 = m._W0_eff(m.W0)                               # fold mixing like the caller
    type_masks = [tf.cast(tf.equal(Z, t), tf.float32)[:, :, None] for t in range(m.num_types)]
    with tf.GradientTape() as tape:
        tape.watch(desc)
        b0_t = tf.gather(m.b0, Z); W1_t = tf.gather(m.W1, Z)
        z1 = tf.add_n([tf.einsum('baq,qh->bah', desc, W0[t]) * type_masks[t]
                       for t in range(m.num_types)]) + b0_t
        h1 = m.activation(z1) * amask[:, :, None]
        bh_t = tf.gather(m.bh, Z)
        z2 = tf.add_n([tf.einsum('bah,hg->bag', h1, m.Wh[t]) * type_masks[t]
                       for t in range(m.num_types)]) + bh_t
        h2 = m.activation(z2) * amask[:, :, None]
        U = (tf.reduce_sum(h2 * W1_t, axis=2) + m.b1) * amask   # [B,A] local energies
        Usum = tf.reduce_sum(U)
    de_dq = tape.gradient(Usum, desc)                 # [B,A,Q] — exact per-atom de_dq
    W_atom = batch["_W_atom"]
    return -tf.einsum('baq,basq->bs', de_dq, W_atom)  # [B,3]


@pytest.mark.parametrize("activation", ["tanh", "swish"])
def test_predict_batch_analytic_de_dq_matches_autodiff(activation):
    cfg = _tiny_cfg(num_hidden_layers=2, mixing=False)
    cfg.activation = activation
    model, train, _ = _build(cfg)
    # randomize weights so the check isn't trivially satisfied at init
    for v in (model.W0, model.b0, model.Wh, model.bh, model.W1):
        v.assign(tf.random.stateless_normal(v.shape, [1, 2], stddev=0.3))
    from Adam import Adam
    opt = Adam(model); opt._precompute_W_atom(train)
    ref = _autodiff_reference_dipole(model, train)
    got = model.predict_batch(
        train["descriptors"], train["grad_values"], train["pair_atom"],
        train["pair_gidx"], train["pair_struct"], train["positions"],
        train["Z_int"], train["boxes"], train["atom_mask"],
        model._W0_eff(model.W0), model.b0, model.W1, model.b1,
        Wh=model.Wh, bh=model.bh, W_atom=train["_W_atom"])
    assert np.allclose(got.numpy(), ref.numpy(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("activation", ["tanh", "swish"])
def test_calc_forces_two_layer_matches_autodiff(activation):
    cfg = _tiny_cfg(num_hidden_layers=2, mixing=False); cfg.activation = activation
    model, _, _ = _build(cfg)                     # for model.activation + calc_forces
    N, H, Q, M = 5, cfg.num_neurons, cfg.dim_q, 4
    g = tf.random.Generator.from_seed(0)
    q     = g.normal([N, Q]);   W0_t = g.normal([N, Q, H]); b0_t = g.normal([N, H])
    Wh_t  = g.normal([N, H, H]); bh_t = g.normal([N, H]);    W1_t = g.normal([N, H])
    grads = g.normal([N, M, 3, Q]); nmask = tf.ones([N, M])
    # autodiff reference de_dq of the local energy U = sum_h h2*W1 (b1 drops out of d/dq)
    with tf.GradientTape() as tape:
        tape.watch(q)
        z1 = tf.einsum('nq,nqh->nh', q, W0_t) + b0_t; h1 = model.activation(z1)
        z2 = tf.einsum('nh,nhg->ng', h1, Wh_t) + bh_t; h2 = model.activation(z2)
        Us = tf.reduce_sum(tf.reduce_sum(h2 * W1_t, axis=1))
    de_dq_auto = tape.gradient(Us, q)                       # [N,Q]
    ref = tf.einsum('nq,nmcq->nmc', de_dq_auto, grads)
    # calc_forces recomputes nothing internally - it receives h1,z1,h2,z2:
    z1 = tf.einsum('nq,nqh->nh', q, W0_t) + b0_t; h1 = model.activation(z1)
    z2 = tf.einsum('nh,nhg->ng', h1, Wh_t) + bh_t; h2 = model.activation(z2)
    got = model.calc_forces(h1, grads, W1_t, W0_t, nmask, z=z1,
                            Wh_t=Wh_t, h2=h2, z2=z2)
    assert np.allclose(got.numpy(), ref.numpy(), atol=1e-5, rtol=1e-4)


def test_adam_trains_two_layer_and_scores():
    cfg = _tiny_cfg(num_hidden_layers=2, mixing=True)
    cfg.num_generations = 30; cfg.adam_learning_rate = 1e-2
    model, train, val = _build(cfg)
    from Adam import Adam
    opt = Adam(model)
    l0 = float(opt._batch_loss(train))
    history, final_model, best = opt.fit(train, val)
    assert float(opt._batch_loss(train)) < l0
    m, _ = best.score(val); assert float(m["rmse"]) < 1.0
    assert tuple(final_model.Wh.shape) == (cfg.num_types, cfg.num_neurons, cfg.num_neurons)

def test_adam_reg_includes_wh_bh():
    cfg = _tiny_cfg(num_hidden_layers=2, mixing=False)
    cfg.lambda_1 = 0.01; cfg.lambda_2 = 0.01
    model, train, _ = _build(cfg)
    from Adam import Adam
    opt = Adam(model)
    tp = opt._type_params(0)
    H = cfg.num_neurons; Q = cfg.dim_q
    # W0(Q*H) + b0(H) + Wh(H*H) + bh(H) + W1(H)
    assert int(tp.shape[0]) == Q * H + H + H * H + H + H


def test_predict_trajectory_batch_uses_second_layer():
    """Regression test for the bug where spectroscopy.predict_trajectory_batch
    (legacy quippy/eager path) called model.predict_batch(...) without threading
    Wh/bh, silently skipping the second hidden layer during trajectory/IR
    inference for a trained 2-layer dipole model.

    Strategy: build a 2-layer model with randomised Wh/bh, run real trajectory
    frames through predict_trajectory_batch, and compare against a 1-layer
    model that shares the identical W0/b0/W1/b1. If Wh/bh were dropped (the
    bug), predict_batch would fall back to its single-layer branch and the
    two predictions would be numerically identical to the 1-layer model's —
    exactly what the bug produced. With the fix, the second layer is a
    nontrivial nonlinear transform, so the outputs must differ.
    """
    import ase.io
    from data import assign_type_indices
    from spectroscopy import predict_trajectory_batch

    cfg = _tiny_cfg(num_hidden_layers=2, mixing=False)
    model, _, _ = _build(cfg)
    for v in (model.W0, model.b0, model.Wh, model.bh, model.W1):
        v.assign(tf.random.stateless_normal(v.shape, [7, 8], stddev=0.3))

    allowed = set(cfg.types)
    frames = [f for f in ase.io.read(cfg.data_path, index=':6')
              if set(f.numbers.tolist()) <= allowed]
    assert len(frames) >= 2, "need at least 2 in-species frames from test.xyz"
    batch_types = assign_type_indices(frames, cfg.types)

    dip_2layer = predict_trajectory_batch(
        model, model.builder, frames, batch_types,
        pin_to_cpu=True, descriptor_batch_frames=1)

    cfg1 = _tiny_cfg(num_hidden_layers=1, mixing=False)
    model1, _, _ = _build(cfg1)
    assert cfg1.types == cfg.types  # same data_path/allowed_species => same order
    model1.W0.assign(model.W0); model1.b0.assign(model.b0)
    model1.W1.assign(model.W1); model1.b1.assign(model.b1)
    assert getattr(model1, "Wh", None) is None

    dip_1layer = predict_trajectory_batch(
        model1, model1.builder, frames, batch_types,
        pin_to_cpu=True, descriptor_batch_frames=1)

    assert np.all(np.isfinite(dip_2layer)) and dip_2layer.shape == (len(frames), 3)
    # The buggy code (Wh/bh dropped) collapses to exactly the 1-layer forward;
    # the fix makes the second layer's nonlinearity actually matter.
    assert not np.allclose(dip_2layer, dip_1layer, atol=1e-6), (
        "predict_trajectory_batch output is identical whether or not Wh/bh "
        "are present — the second hidden layer is being silently dropped")


def test_model_io_roundtrip_two_layer(tmp_path):
    import model_io
    cfg = _tiny_cfg(num_hidden_layers=2, mixing=False)
    model, train, _ = _build(cfg)
    from Adam import Adam
    opt = Adam(model); opt._precompute_W_atom(train)
    pred_before = model.predict_batch(
        train["descriptors"], train["grad_values"], train["pair_atom"],
        train["pair_gidx"], train["pair_struct"], train["positions"],
        train["Z_int"], train["boxes"], train["atom_mask"],
        model._W0_eff(model.W0), model.b0, model.W1, model.b1,
        Wh=model.Wh, bh=model.bh, W_atom=train["_W_atom"]).numpy()
    p = str(tmp_path / "m.h5")
    model_io.save_model(model, cfg, path=p)
    loaded = model_io.load_model(p)
    opt2 = Adam(loaded); opt2._precompute_W_atom(train)
    pred_after = loaded.predict_batch(
        train["descriptors"], train["grad_values"], train["pair_atom"],
        train["pair_gidx"], train["pair_struct"], train["positions"],
        train["Z_int"], train["boxes"], train["atom_mask"],
        loaded._W0_eff(loaded.W0), loaded.b0, loaded.W1, loaded.b1,
        Wh=loaded.Wh, bh=loaded.bh, W_atom=train["_W_atom"]).numpy()
    assert np.allclose(pred_before, pred_after, atol=1e-6)
