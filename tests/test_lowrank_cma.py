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
    # class-level defaults a user may flip for experiments.
    cfg.optimizer_mode = "snes"
    cfg.snes_mean_optimizer = "vanilla"
    cfg.snes_sigma_cumulation = False
    cfg.guided_es_enabled = False
    cfg.snes_cov_mode = "none"
    cfg.snes_cma_c1_scale = 1.0
    dataset, ti = collect(cfg)
    cfg.randomise(dataset); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    model = TNEP(cfg)
    return model, train, val


def test_cma_constants_match_reference(tiny_model):
    import numpy as np
    model, _, _ = tiny_model
    snes = model.optimizer
    n = snes.dim
    cc, c1, mu_eff = snes._cma_constants()
    raw = snes._recomb_w.numpy()
    assert np.isclose(raw.sum(), 1.0, atol=1e-6)
    assert np.isclose(mu_eff, 1.0/np.sum(raw**2), rtol=1e-5)
    assert np.isclose(cc, (4 + mu_eff/n)/(n + 4 + 2*mu_eff/n), rtol=1e-6)
    assert np.isclose(c1, 2.0/((n + 1.3)**2 + mu_eff), rtol=1e-6)


def test_cma_config_defaults():
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    assert cfg.snes_cov_mode in ("none", "rank1", "lowrank")
    assert cfg.snes_cma_c1_scale > 0


def test_rank1_sampling_inflates_pc_variance(tiny_model):
    import numpy as np, tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "rank1"
    v = np.zeros((snes.dim,), np.float32); v[0] = 1.0
    snes._cma_pc = tf.Variable(v, trainable=False)
    snes._cma_a1 = tf.Variable(2.0, dtype=tf.float32, trainable=False)
    _, aux = snes.ask()
    d = aux["delta"].numpy(); s = aux["s_iso"].numpy()
    assert np.var(d[:, 0]) > 2.0 * np.var(d[:, 5])      # correction lands in delta
    assert np.var(s[:, 0]) < 2.0 * np.var(s[:, 5])      # s_iso NOT inflated (sigma-isolated)
    snes.cfg.snes_cov_mode = "none"


def test_rank1_none_bit_identical(tiny_model):
    import numpy as np
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "none"
    st = snes.tf_rng.state.numpy().copy()
    _, aux_a = snes.ask()
    snes.tf_rng.state.assign(st)
    _, aux_b = snes.ask()
    assert np.array_equal(aux_a["delta"].numpy(), aux_b["delta"].numpy())


def test_rank1_delta_is_true_displacement(tiny_model):
    import numpy as np
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "rank1"
    import tensorflow as tf, numpy as np
    v = np.zeros((snes.dim,), np.float32); v[3] = 1.0
    snes._cma_pc = tf.Variable(v, trainable=False)
    snes._cma_a1 = tf.Variable(1.0, dtype=tf.float32, trainable=False)
    samples, aux = snes.ask()
    assert np.allclose((samples - snes.mu).numpy(), aux["delta"].numpy(), atol=1e-6)
    snes.cfg.snes_cov_mode = "none"


def test_rank1_evolution_path_recurrence(tiny_model):
    import numpy as np, tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "rank1"
    snes._cma_pc = None; snes._cma_a1 = None
    snes._ensure_cma_state()
    pc0 = snes._cma_pc.numpy().copy()
    _, aux = snes.ask()                          # no global key -> raw-order fallback
    u = tf.constant(snes.compute_utilities(), tf.float32)
    sig = snes.sigma.numpy()
    s_eff = aux["delta"].numpy() / sig
    snes.update(u, aux)
    cc, c1, mu_eff = snes._cma_constants()
    w = snes._recomb_w.numpy()
    mean_step = np.einsum('p,pd->d', w, s_eff)
    expected = (1-cc)*pc0 + np.sqrt(cc*(2-cc)*mu_eff)*mean_step
    assert np.allclose(snes._cma_pc.numpy(), expected, atol=1e-5)
    assert np.isclose(float(snes._cma_a1.numpy()),
                      np.sqrt(c1*snes.cfg.snes_cma_c1_scale), rtol=1e-5)
    snes.cfg.snes_cov_mode = "none"


def test_rank1_none_update_bit_identical(tiny_model):
    import numpy as np, tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "none"
    # Reset any CMA state left by an earlier test (module-scoped fixture).
    snes._cma_pc = None; snes._cma_a1 = None
    # Known init values well above sigma_floor so the floor cannot clamp.
    snes.mu.assign(tf.ones([snes.dim], tf.float32) * 0.5)
    snes.sigma.assign(tf.ones([snes.dim], tf.float32) * 0.1)
    mu0 = snes.mu.numpy().copy()
    sigma0 = snes.sigma.numpy().copy()
    _, aux = snes.ask()
    u = tf.constant(snes.compute_utilities(), tf.float32)
    s_iso = aux["s_iso"].numpy()
    # Manual vanilla reference from the SAME aux.
    grad_mu = np.einsum('p,pd->d', u.numpy(), s_iso)
    grad_sigma = np.einsum('p,pd->d', u.numpy(), s_iso**2 - 1.0)
    exp_mu = mu0 + sigma0 * grad_mu
    exp_sigma = sigma0 * np.exp(float(snes.eta_sigma) * grad_sigma)
    snes.update(u, aux)
    assert snes._cma_pc is None
    assert np.allclose(snes.mu.numpy(), exp_mu, atol=1e-5)
    assert np.allclose(snes.sigma.numpy(), exp_sigma, atol=1e-5)


def _build_cho_cfg(per_type: bool):
    """Self-contained tiny CHO cfg builder (mirrors test_guided_es). When
    per_type=True, per-type regularization is ON from construction so the
    per-type ranking path is active (num_types=4)."""
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    cfg.data_path = 'datasets/test.xyz'; cfg.test_data_path = None
    cfg.allowed_species = [6, 1, 7, 8]; cfg.filter_mode = 'subset'
    cfg.target_mode = 1; cfg.dipole_units = 'e*bohr'; cfg.scale_targets = True
    cfg.convert_dipole_to_eangstrom = False
    cfg.total_N = 16; cfg.test_ratio = 0.25; cfg.skip_h_centers = False
    cfg.num_neurons = 8; cfg.descriptor_mode = 0; cfg.descriptor_mixing = False
    cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False; cfg.use_pinned_buffers = False; cfg.use_cufile = False
    cfg.save_path = None; cfg.checkpoint_interval = None
    cfg.seed = 0; cfg.eval_jit_compile = False; cfg.val_size = None
    cfg.dipole_rij_power = 2; cfg.patience = None
    # Pin optimizer baseline; opt into rank-1 only.
    cfg.optimizer_mode = "snes"
    cfg.snes_mean_optimizer = "vanilla"
    cfg.snes_sigma_cumulation = False
    cfg.guided_es_enabled = False
    cfg.snes_cov_mode = "rank1"
    cfg.snes_cma_c1_scale = 1.0
    if per_type:
        cfg.lambda_1 = 0.001; cfg.lambda_2 = 0.001
        cfg.per_type_regularization = True; cfg.toggle_regularization = True
    else:
        cfg.lambda_1 = 0.0; cfg.lambda_2 = 0.0
        cfg.per_type_regularization = False; cfg.toggle_regularization = False
    return cfg


def _materialise(cfg):
    """Run the data pipeline and build the TNEP model. Returns (snes, train, val)."""
    from data import collect, split, pad_and_stack
    from DescriptorBuilderGPU import compute_dim_q
    from TNEP import TNEP
    ds, ti = collect(cfg); cfg.randomise(ds); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(ds, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    snes = TNEP(cfg).optimizer
    return snes, train, val


def test_rank1_bounded_long_run():
    # MANDATORY divergence guard: a LONG rank-1 run on a tiny model must stay
    # finite, with sigma bounded and the evolution path norm bounded. This is
    # the regression guard for the sigma-cumulation >1e38 divergence family.
    import numpy as np, time
    cfg = _build_cho_cfg(per_type=False)
    cfg.pop_size = 100
    cfg.num_generations = 300
    cfg.val_interval = 25
    snes, train, val = _materialise(cfg)
    t0 = time.perf_counter()
    hist = snes.fit(train, val)
    elapsed = time.perf_counter() - t0
    tl = hist["train_loss"] if isinstance(hist, dict) else (hist[0]["train_loss"])
    max_sigma = float(np.max(snes.sigma.numpy()))
    pc_norm = float(np.linalg.norm(snes._cma_pc.numpy()))
    print(f"\n[rank1_bounded_long_run] gens={cfg.num_generations} "
          f"elapsed={elapsed:.2f}s max_sigma={max_sigma:.6g} "
          f"(init_sigma={float(cfg.init_sigma):.6g}) ||p_c||={pc_norm:.6g}")
    assert np.all(np.isfinite(tl)), "rank-1 run went non-finite"
    assert max_sigma < 100.0 * float(cfg.init_sigma), "sigma exploded"
    assert pc_norm < 1e3, "evolution path unbounded"


def test_rank1_per_type_pc_uses_global_ranking():
    # The real guard a finiteness test cannot provide: in per-type mode, p_c
    # must be built from the GLOBAL fitness ranking of delta/sigma, NOT any
    # per-type column permutation. We do one controlled generation mirroring
    # fit()'s per-type path and assert _cma_pc matches the GLOBAL-ranked
    # recurrence exactly. If update() were rewired to feed the per-type-permuted
    # tensor into the evolution path, this allclose would fail.
    import numpy as np, tensorflow as tf
    cfg = _build_cho_cfg(per_type=True)
    cfg.pop_size = 12
    cfg.num_generations = 1
    cfg.val_interval = 1
    snes, train, _ = _materialise(cfg)
    assert snes._per_type, "per-type must be active"
    snes.cfg.snes_cov_mode = "rank1"
    snes._cma_pc = None; snes._cma_a1 = None
    snes._ensure_cma_state()
    pc0 = snes._cma_pc.numpy().copy()

    samples, aux = snes.ask()
    fitness_per_type_rmse = snes.evaluate_population(
        samples, batch_data=train, return_per_type=True)
    fitness = fitness_per_type_rmse[:, -1].numpy()      # GLOBAL rmse column

    sig = snes.sigma.numpy()
    # Expected p_c uses the GLOBAL ranking of delta/sigma.
    global_ranks = np.argsort(fitness)
    s_eff_global = aux["delta"].numpy()[global_ranks] / sig
    cc, c1, mu_eff = snes._cma_constants()   # also materialises _recomb_w if None
    w = snes._recomb_w.numpy()
    mean_step = np.einsum('p,pd->d', w, s_eff_global)
    expected_pc = (1 - cc) * pc0 + np.sqrt(cc * (2 - cc) * mu_eff) * mean_step

    # Drive update() exactly as fit() does for the per-type branch: the
    # mean/sigma steps consume the per-type-PERMUTED tensors, while the
    # evolution path consumes the GLOBALLY-ranked s_eff_global.
    s_iso_sorted = snes._build_per_type_gradients(
        aux["s_iso"], fitness_per_type_rmse, samples)
    delta_sorted = snes._build_per_type_gradients(
        aux["delta"], fitness_per_type_rmse, samples)
    s_eff_g_tf = tf.gather(
        aux["delta"], tf.argsort(fitness_per_type_rmse[:, -1])) / snes.sigma
    snes.update(snes.utilities,
                {"s_iso": s_iso_sorted, "delta": delta_sorted,
                 "s_eff_global": s_eff_g_tf})

    got_pc = snes._cma_pc.numpy()
    print(f"\n[rank1_per_type_pc] ||expected||={np.linalg.norm(expected_pc):.6g} "
          f"||got||={np.linalg.norm(got_pc):.6g} "
          f"max_abs_diff={np.max(np.abs(got_pc - expected_pc)):.3g}")
    assert np.allclose(got_pc, expected_pc, atol=1e-4), \
        "p_c not built from GLOBAL ranking"


def _minimal_state(snes):
    """Build a minimal-but-valid save_checkpoint state dict from a model's
    optimizer, mirroring the required keys fit() supplies."""
    return {
        "mu": snes.mu,
        "sigma": snes.sigma,
        "best_mu": snes.mu,
        "best_sigma": snes.sigma,
        "best_val_loss": 1.23,
        "gens_without_improvement": 0,
    }


def _minimal_history():
    return {
        "generation": [], "train_loss": [], "val_loss": [],
        "sigma_mean": [], "timing": {},
    }


def test_rank1_checkpoint_round_trip(tmp_path):
    import numpy as np, tensorflow as tf
    import model_io
    cfg = _build_cho_cfg(per_type=False)
    cfg.snes_cov_mode = "rank1"
    snes, _, _ = _materialise(cfg)
    # Populate the learned rank-1 state with known non-zero values.
    snes._cma_pc = None; snes._cma_a1 = None
    snes._ensure_cma_state()
    known_pc = (np.arange(snes.dim, dtype=np.float32) + 1.0) * 0.01
    known_a1 = 0.7531
    snes._cma_pc.assign(known_pc)
    snes._cma_a1.assign(np.float32(known_a1))
    # Build a ckpt_state the way fit() does (guarded cma keys present).
    state = _minimal_state(snes)
    state["cma_pc"] = snes._cma_pc
    state["cma_a1"] = float(snes._cma_a1.numpy())
    path = str(tmp_path / "ckpt_rank1.h5")
    model_io.save_checkpoint(path, snes.cfg, state, _minimal_history(), 0)
    _, resume = model_io.load_checkpoint(path)
    assert np.allclose(np.asarray(resume["cma_pc"], np.float32), known_pc)
    assert np.isclose(resume["cma_a1"], known_a1, rtol=1e-6)


def test_none_checkpoint_has_no_cma_datasets(tmp_path):
    import h5py
    import model_io
    cfg = _build_cho_cfg(per_type=False)
    cfg.snes_cov_mode = "none"
    snes, _, _ = _materialise(cfg)
    assert snes._cma_pc is None  # pure-SNES: no CMA state allocated
    # fit()'s guard omits the cma keys when _cma_pc is None.
    state = _minimal_state(snes)
    path = str(tmp_path / "ckpt_none.h5")
    model_io.save_checkpoint(path, snes.cfg, state, _minimal_history(), 0)
    with h5py.File(path, "r") as f:
        assert "cma_pc" not in f["snes"], \
            "pure-SNES checkpoint must not gain CMA datasets"
