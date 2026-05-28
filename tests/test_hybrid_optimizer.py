from __future__ import annotations
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope="module")
def tiny_model():
    """Tiny CHO dipole model + padded train/val dicts, seed-fixed."""
    from TNEPconfig import TNEPconfig
    from data import collect, split, pad_and_stack
    from DescriptorBuilderGPU import compute_dim_q
    from TNEP import TNEP
    cfg = TNEPconfig()
    cfg.data_path = 'datasets/test.xyz'; cfg.test_data_path = None
    cfg.allowed_species = [6, 1, 7, 8]; cfg.filter_mode = 'subset'
    cfg.target_mode = 1; cfg.dipole_units = 'e*bohr'; cfg.scale_targets = True
    cfg.convert_dipole_to_eangstrom = False
    cfg.total_N = 16; cfg.test_ratio = 0.25; cfg.skip_h_centers = False
    cfg.num_neurons = 8; cfg.descriptor_mode = 0; cfg.descriptor_mixing = False
    cfg.pop_size = 8; cfg.num_generations = 1
    cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False; cfg.use_pinned_buffers = False; cfg.use_cufile = False
    cfg.save_path = None; cfg.checkpoint_interval = None
    cfg.lambda_1 = 0.0; cfg.lambda_2 = 0.0  # zero reg for clean FD check
    cfg.seed = 0; cfg.eval_jit_compile = False
    cfg.val_interval = 1; cfg.val_size = None
    cfg.toggle_regularization = False; cfg.per_type_regularization = False
    cfg.dipole_rij_power = 2
    dataset, ti = collect(cfg)
    cfg.randomise(dataset); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    model = TNEP(cfg)
    return model, train, val


def test_loss_grad_is_finite(tiny_model):
    model, train, _ = tiny_model
    snes = model.optimizer
    loss, grad = snes._loss_and_grad(train)
    assert np.isfinite(float(loss))
    g = grad.numpy()
    assert g.shape == (snes.dim,)
    assert np.all(np.isfinite(g))
    assert np.linalg.norm(g) > 0.0


def test_loss_grad_matches_finite_difference(tiny_model):
    model, train, _ = tiny_model
    snes = model.optimizer
    loss0, grad = snes._loss_and_grad(train)
    grad = grad.numpy()
    mu0 = snes.mu.numpy().copy()
    rng = np.random.default_rng(0)
    idxs = rng.choice(snes.dim, size=8, replace=False)
    eps = 1e-3
    for i in idxs:
        for sign in (+1, -1):
            mu = mu0.copy(); mu[i] += sign * eps
            snes.mu.assign(mu)
            l, _ = snes._loss_and_grad(train)
            if sign == +1: lp = float(l)
            else: lm = float(l)
        snes.mu.assign(mu0)
        fd = (lp - lm) / (2 * eps)
        assert abs(fd - grad[i]) < 1e-2 * (abs(grad[i]) + 1.0), \
            f"param {i}: analytic {grad[i]:.4e} vs FD {fd:.4e}"


def test_adam_state_allocated(tiny_model):
    model, train, _ = tiny_model
    snes = model.optimizer
    snes._ensure_adam_state()
    assert snes.adam_m.shape == (snes.dim,)
    assert float(tf.reduce_sum(tf.abs(snes.adam_m))) == 0.0
    assert int(snes.adam_t.numpy()) == 0
    assert snes.adam_v.shape == (snes.dim,)


def test_optimizer_config_defaults():
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    assert cfg.optimizer_mode == "snes"           # default unchanged
    assert cfg.adam_plateau_patience == 100
    assert cfg.snes_plateau_patience == 2000
    assert cfg.hybrid_start == "adam"
    assert cfg.adam_lr == 1e-3
    assert cfg.adam_reset_moments_on_entry is True
    assert cfg.hybrid_handoff_sigma is None


def test_loss_grad_includes_regularisation(tiny_model):
    model, train, _ = tiny_model
    snes = model.optimizer
    # fixture has toggle_regularization=False; flip it on with nonzero lambdas
    import tensorflow as tf
    snes.cfg.toggle_regularization = True
    snes.lambda_1.assign(0.01); snes.lambda_2.assign(0.01)
    loss_reg, grad_reg = snes._loss_and_grad(train)
    snes.cfg.toggle_regularization = False
    loss_noreg, grad_noreg = snes._loss_and_grad(train)
    assert float(loss_reg) > float(loss_noreg)            # reg adds positive penalty
    assert float(tf.norm(grad_reg - grad_noreg)) > 0.0     # reg changes the gradient
    # restore fixture state for other tests (module-scoped fixture!)
    snes.cfg.toggle_regularization = False
    snes.lambda_1.assign(0.0); snes.lambda_2.assign(0.0)


def test_adam_steps_reduce_loss(tiny_model):
    import numpy as np
    model, train, _ = tiny_model
    snes = model.optimizer
    mu_save = snes.mu.numpy().copy()
    try:
        snes.cfg.adam_lr = 5e-3
        snes._ensure_adam_state()
        l0, _ = snes._loss_and_grad(train)
        for _ in range(50):
            snes._adam_step(train)
        l1, _ = snes._loss_and_grad(train)
        assert float(l1) < float(l0), f"Adam did not reduce loss: {float(l0)} -> {float(l1)}"
    finally:
        snes.mu.assign(mu_save)
        # reset Adam moments so a later test starting fresh isn't polluted
        snes.adam_m.assign(np.zeros(snes.dim, dtype=np.float32))
        snes.adam_v.assign(np.zeros(snes.dim, dtype=np.float32))
        snes.adam_t.assign(0)


def test_hybrid_early_stop_guard():
    import pytest
    from TNEPconfig import TNEPconfig
    from data import collect, split, pad_and_stack
    from DescriptorBuilderGPU import compute_dim_q
    from TNEP import TNEP
    cfg = TNEPconfig()
    cfg.data_path = 'datasets/test.xyz'; cfg.test_data_path = None
    cfg.allowed_species = [6, 1, 7, 8]; cfg.filter_mode = 'subset'
    cfg.target_mode = 1; cfg.total_N = 16; cfg.test_ratio = 0.25
    cfg.num_neurons = 8; cfg.descriptor_mode = 0; cfg.descriptor_mixing = False
    cfg.pop_size = 8; cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False; cfg.use_pinned_buffers = False; cfg.use_cufile = False
    cfg.save_path = None; cfg.checkpoint_interval = None; cfg.seed = 0
    cfg.optimizer_mode = "hybrid"
    cfg.patience = 50            # < snes_plateau_patience (2000) -> must raise
    cfg.snes_plateau_patience = 2000
    dataset, ti = collect(cfg); cfg.randomise(dataset); cfg.dim_q = compute_dim_q(cfg)
    with pytest.raises(ValueError):
        TNEP(cfg)   # SNES.__init__ guard fires during model construction
