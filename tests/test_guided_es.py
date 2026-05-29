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


def test_guided_config_defaults():
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    assert cfg.guided_es_enabled is False
    assert isinstance(cfg.guided_es_k, int) and cfg.guided_es_k >= 1
    assert cfg.guided_es_alpha >= 0
    assert isinstance(cfg.guided_es_grad_interval, int) and cfg.guided_es_grad_interval >= 1
