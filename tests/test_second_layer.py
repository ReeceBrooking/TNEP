import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import pytest

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
    cfg.pop_size = 8; cfg.num_generations = 3
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
