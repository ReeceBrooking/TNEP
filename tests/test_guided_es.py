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
    # Pin the optimizer baseline so tests are hermetic regardless of the
    # class-level defaults a user may flip for experiments (mean-Adam /
    # cumulation / hybrid / guided-ES). Individual tests opt into a feature
    # explicitly.
    cfg.optimizer_mode = "snes"
    cfg.snes_mean_optimizer = "vanilla"
    cfg.snes_sigma_cumulation = False
    cfg.guided_es_enabled = False
    dataset, ti = collect(cfg)
    cfg.randomise(dataset); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    model = TNEP(cfg)
    return model, train, val


def test_guided_buffer_orthonormal(tiny_model):
    import numpy as np, tensorflow as tf
    model, train, _ = tiny_model
    snes = model.optimizer
    snes.cfg.guided_es_enabled = True
    snes.cfg.guided_es_k = 4
    for _ in range(6):                     # push more than k gradients
        snes._refresh_guided_subspace(train)
    U = snes._U
    assert U.shape == (snes.dim, 4)
    G = (tf.transpose(U) @ U).numpy()
    assert np.allclose(G, np.eye(4), atol=1e-4), "U columns not orthonormal"


def test_guided_alpha0_matches_vanilla(tiny_model):
    # guided OFF => ask() draws identically to vanilla and update() is bit-identical.
    import numpy as np, tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.guided_es_enabled = False
    mu0 = snes.mu.numpy().copy(); sig0 = snes.sigma.numpy().copy()
    samples, aux = snes.ask()
    # mean step must equal Σ u_p (samples - mu)
    u = tf.constant(snes.compute_utilities(), tf.float32)
    snes.update(u, aux)
    expected_mu = mu0 + np.einsum('p,pd->d', u.numpy(), samples.numpy() - mu0)
    assert np.allclose(snes.mu.numpy(), expected_mu, atol=1e-5)


def test_guided_sampling_inflates_subspace_variance(tiny_model):
    import numpy as np, tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.guided_es_enabled = True
    snes.cfg.guided_es_k = 2
    snes.cfg.guided_es_alpha = 4.0          # large so subspace variance clearly dominates
    e = np.zeros((snes.dim, 2), np.float32); e[0, 0] = 1.0; e[1, 1] = 1.0
    snes._U = tf.constant(e)                 # planted orthonormal subspace (dims 0,1)
    samples, aux = snes.ask()
    d = (samples.numpy() - snes.mu.numpy())
    var_sub = 0.5 * (np.var(d[:, 0]) + np.var(d[:, 1]))
    var_bulk = np.var(d[:, 5])
    assert var_sub > 2.0 * var_bulk, f"subspace var {var_sub} not > 2x bulk {var_bulk}"
    snes.cfg.guided_es_enabled = False       # restore module fixture


def test_guided_update_runs_end_to_end(tiny_model):
    model, train, val = tiny_model
    snes = model.optimizer
    snes.cfg.guided_es_enabled = True; snes.cfg.guided_es_k = 3; snes.cfg.guided_es_alpha = 0.5
    snes._refresh_guided_subspace(train)     # populate U
    snes.cfg.num_generations = 20; snes.cfg.patience = None
    hist = snes.fit(train, val)
    h = hist[0] if isinstance(hist, tuple) else hist
    tl = h["train_loss"] if isinstance(h, dict) else h
    import numpy as np
    assert np.all(np.isfinite(tl))
    snes.cfg.guided_es_enabled = False; snes.cfg.num_generations = 1


def test_guided_config_defaults():
    # guided_es_enabled is a user-tunable run toggle (flipped on for
    # experiments), so assert the fields exist with valid types/ranges rather
    # than pinning the exact default — matching the other config-field tests.
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    assert isinstance(cfg.guided_es_enabled, bool)
    assert isinstance(cfg.guided_es_k, int) and cfg.guided_es_k >= 1
    assert cfg.guided_es_alpha >= 0
    assert isinstance(cfg.guided_es_grad_interval, int) and cfg.guided_es_grad_interval >= 1


def test_guided_autorefresh_in_fit(tiny_model):
    # With guided enabled, fit() must refresh the subspace itself — _U should
    # become populated during the run WITHOUT any manual _refresh call.
    model, train, val = tiny_model
    snes = model.optimizer
    snes._U = None   # reset any state left by earlier tests in this module session
    assert snes._U is None
    snes.cfg.guided_es_enabled = True
    snes.cfg.guided_es_k = 3
    snes.cfg.guided_es_grad_interval = 5
    snes.cfg.num_generations = 12
    snes.cfg.patience = None
    try:
        snes.fit(train, val)
        assert snes._U is not None, "fit() did not auto-refresh the guided subspace"
        assert snes._U.shape[1] >= 1
    finally:
        snes.cfg.guided_es_enabled = False
        snes.cfg.num_generations = 1
        snes._U = None   # reset shared fixture state


def test_guided_disabled_no_refresh(tiny_model):
    # guided OFF: fit() must NEVER touch the subspace (stays None, zero cost).
    model, train, val = tiny_model
    snes = model.optimizer
    snes._U = None
    snes.cfg.guided_es_enabled = False
    snes.cfg.num_generations = 5; snes.cfg.patience = None
    snes.fit(train, val)
    assert snes._U is None
    snes.cfg.num_generations = 1


def test_guided_with_per_type_ranking():
    # Coverage for the danger-zone: guided sampling + per-type ranking, where
    # _build_per_type_gradients must permute BOTH s_iso and delta by the same
    # per-type rankings. A mismatched permutation would desync the mean/sigma
    # steps and typically diverge or NaN. Builds its own model with per-type
    # ON from construction (so _type_of_variable is properly initialised).
    import numpy as np
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
    cfg.pop_size = 12; cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False; cfg.use_pinned_buffers = False; cfg.use_cufile = False
    cfg.save_path = None; cfg.checkpoint_interval = None
    cfg.lambda_1 = 0.001; cfg.lambda_2 = 0.001
    cfg.seed = 0; cfg.eval_jit_compile = False; cfg.val_interval = 5; cfg.val_size = None
    cfg.dipole_rij_power = 2; cfg.optimizer_mode = "snes"; cfg.patience = None
    cfg.snes_mean_optimizer = "vanilla"; cfg.snes_sigma_cumulation = False
    # per-type ranking ON (num_types=4 here so the per-type path is active)
    cfg.per_type_regularization = True; cfg.toggle_regularization = True
    # guided ON
    cfg.guided_es_enabled = True; cfg.guided_es_k = 3
    cfg.guided_es_alpha = 0.5; cfg.guided_es_grad_interval = 5
    cfg.num_generations = 25
    ds, ti = collect(cfg); cfg.randomise(ds); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(ds, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    snes = TNEP(cfg).optimizer
    assert snes._per_type, "per-type ranking must be active for this test"
    hist = snes.fit(train, val)
    h = hist[0] if isinstance(hist, tuple) else hist
    tl = h["train_loss"] if isinstance(h, dict) else h
    assert np.all(np.isfinite(tl)), "per-type guided run went non-finite"
    assert float(np.max(snes.sigma.numpy())) < 100.0 * float(cfg.init_sigma), \
        "sigma exploded under per-type guided"
