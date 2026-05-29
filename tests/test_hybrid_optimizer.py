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
    # cumulation / hybrid). Individual tests opt into a feature explicitly.
    cfg.optimizer_mode = "snes"
    cfg.snes_mean_optimizer = "vanilla"
    cfg.snes_sigma_cumulation = False
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


def test_optimizer_config_fields_exist_and_valid():
    # These are user-tunable RUN settings (mode, patiences, lrs get flipped
    # for experiments), so assert the fields exist with valid types/ranges
    # rather than pinning exact default values that legitimately change.
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    assert cfg.optimizer_mode in ("snes", "adam", "hybrid")
    assert cfg.hybrid_start in ("adam", "snes")
    assert isinstance(cfg.adam_plateau_patience, int) and cfg.adam_plateau_patience > 0
    assert isinstance(cfg.snes_plateau_patience, int) and cfg.snes_plateau_patience > 0
    assert isinstance(cfg.adam_lr, float) and cfg.adam_lr > 0
    assert isinstance(cfg.adam_reset_moments_on_entry, bool)
    assert cfg.hybrid_handoff_sigma is None or cfg.hybrid_handoff_sigma > 0
    # mean-Adam + cumulation fields (this feature)
    assert cfg.snes_mean_optimizer in ("vanilla", "adam")
    assert cfg.snes_mean_lr is None or cfg.snes_mean_lr > 0
    assert isinstance(cfg.snes_sigma_cumulation, bool)


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


def test_reg_scalar_matches_snes(tiny_model):
    import tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.toggle_regularization = True
    snes.lambda_1.assign(0.01); snes.lambda_2.assign(0.01)
    mu = snes.mu
    adam_reg = float(snes._reg_scalar_tf(mu))
    # compute_regularization returns (l1, l2, l_orth); SNES adds l1+l2 to
    # fitness. l_orth (orthogonal-mixing penalty) is a separate signal and is
    # deliberately NOT reproduced by Adam; the fixture has descriptor_mixing
    # =False so l_orth == 0 anyway (asserted below).
    l1, l2, l_orth = snes.compute_regularization(mu)
    assert float(l_orth) == 0.0, "fixture should have no orthogonal mixing penalty"
    snes_reg = float(l1) + float(l2)
    assert abs(adam_reg - snes_reg) < 1e-5, f"adam reg {adam_reg} != snes reg {snes_reg}"
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


def test_schedule_fsm_transitions(tiny_model):
    model, _, _ = tiny_model
    snes = model.optimizer
    snes._opt_mode = "hybrid"; snes.cfg.optimizer_mode = "hybrid"
    snes.cfg.adam_plateau_patience = 3
    snes.cfg.snes_plateau_patience = 5
    snes._opt_phase = "adam"; snes._phase_best = float("inf"); snes._hybrid_cycles = 0
    assert snes._advance_schedule(gwi=2) == "adam"
    assert snes._advance_schedule(gwi=3) == "snes"   # Adam->SNES swap
    assert snes._opt_phase == "snes"
    assert snes._advance_schedule(gwi=4) == "snes"
    assert snes._advance_schedule(gwi=5) == "adam"   # SNES->Adam swap
    assert snes._opt_phase == "adam"
    assert snes._hybrid_cycles == 1
    # restore module-scoped fixture state
    snes._opt_mode = "snes"; snes.cfg.optimizer_mode = "snes"; snes._opt_phase = "adam"


def test_hybrid_run_end_to_end(tiny_model):
    model, train, val = tiny_model
    snes = model.optimizer
    snes.cfg.optimizer_mode = "hybrid"; snes._opt_mode = "hybrid"
    snes.cfg.hybrid_start = "adam"; snes._opt_phase = "adam"
    snes.cfg.adam_plateau_patience = 5
    snes.cfg.snes_plateau_patience = 10
    snes.cfg.num_generations = 40
    snes.cfg.adam_lr = 5e-3
    snes.cfg.patience = None
    try:
        hist = snes.fit(train, val)
        if isinstance(hist, tuple): hist = hist[0]
        tl = hist["train_loss"] if isinstance(hist, dict) else hist
        assert tl[-1] <= tl[0] + 1e-6
    finally:
        # restore fixture
        snes.cfg.optimizer_mode = "snes"; snes._opt_mode = "snes"; snes._opt_phase = "adam"
        snes.cfg.num_generations = 1; snes.cfg.patience = None
        if snes.adam_m is not None:
            snes.adam_m.assign(np.zeros(snes.dim, dtype=np.float32))
            snes.adam_v.assign(np.zeros(snes.dim, dtype=np.float32))
            snes.adam_t.assign(0)


def test_snes_mode_unchanged():
    # Fresh model so adam_m is guaranteed unallocated (module fixture may
    # have been touched by earlier Adam tests).
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
    cfg.pop_size = 8; cfg.num_generations = 3
    cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False; cfg.use_pinned_buffers = False; cfg.use_cufile = False
    cfg.save_path = None; cfg.checkpoint_interval = None
    cfg.lambda_1 = 0.0; cfg.lambda_2 = 0.0
    cfg.seed = 0; cfg.eval_jit_compile = False
    cfg.val_interval = 1; cfg.val_size = None
    cfg.toggle_regularization = False; cfg.per_type_regularization = False
    cfg.dipole_rij_power = 2; cfg.patience = None
    cfg.optimizer_mode = "snes"   # explicit: don't depend on the class default
    dataset, ti = collect(cfg)
    cfg.randomise(dataset); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    model = TNEP(cfg)
    snes = model.optimizer
    assert snes._opt_mode == "snes"
    assert snes.adam_m is None          # Adam state never allocated in pure SNES
    snes.fit(train, val)                # must run without touching Adam paths
    assert snes.adam_m is None


def test_hybrid_ends_on_snes(tiny_model):
    model, train, val = tiny_model
    snes = model.optimizer
    snes.cfg.optimizer_mode = "hybrid"; snes._opt_mode = "hybrid"
    snes.cfg.hybrid_start = "adam"; snes._opt_phase = "adam"
    snes.cfg.adam_plateau_patience = 3
    snes.cfg.snes_plateau_patience = 6
    snes.cfg.num_generations = 30
    snes.cfg.hybrid_tail_polish_gens = 5     # final 5 gens forced to SNES
    snes.cfg.adam_lr = 5e-3
    snes.cfg.patience = None
    snes.fit(train, val)
    assert snes._last_phase == "snes", \
        f"hybrid run ended in phase {snes._last_phase!r}, expected 'snes'"
    # restore fixture
    snes.cfg.optimizer_mode = "snes"; snes._opt_mode = "snes"; snes._opt_phase = "adam"


def test_hybrid_state_checkpoint_roundtrip(tiny_model, tmp_path):
    import numpy as np
    from model_io import save_checkpoint, load_checkpoint
    model, _, _ = tiny_model
    snes = model.optimizer
    snes._ensure_adam_state()
    snes.adam_m.assign(np.full(snes.dim, 0.123, dtype=np.float32))
    snes.adam_v.assign(np.full(snes.dim, 0.456, dtype=np.float32))
    snes.adam_t.assign(7)
    snes._opt_phase = "snes"
    snes._phase_best = 0.0042
    snes._hybrid_cycles = 3
    state = {
        "mu": snes.mu, "sigma": snes.sigma,
        "best_mu": snes.mu, "best_sigma": snes.sigma,
        "best_val_loss": 0.01, "gens_without_improvement": 0,
        "tf_rng_state": snes.tf_rng.state,
        "adam_m": snes.adam_m, "adam_v": snes.adam_v, "adam_t": int(snes.adam_t.numpy()),
        "opt_phase": snes._opt_phase, "phase_best": snes._phase_best,
        "hybrid_cycles": snes._hybrid_cycles,
    }
    ckpt = str(tmp_path / "checkpoint.h5")
    save_checkpoint(ckpt, snes.cfg, state, {"train_loss": [0.1], "val_loss": [0.1]}, last_gen=5)
    cfg2, rs = load_checkpoint(ckpt)
    assert np.allclose(rs["adam_m"], 0.123)
    assert np.allclose(rs["adam_v"], 0.456)
    assert int(rs["adam_t"]) == 7
    assert rs["opt_phase"] == "snes"
    assert abs(rs["phase_best"] - 0.0042) < 1e-9
    assert int(rs["hybrid_cycles"]) == 3
    # restore module-scoped fixture state
    snes._opt_phase = "adam"; snes._phase_best = float("inf"); snes._hybrid_cycles = 0
    snes.adam_m.assign(np.zeros(snes.dim, dtype=np.float32))
    snes.adam_v.assign(np.zeros(snes.dim, dtype=np.float32))
    snes.adam_t.assign(0)


def test_defaults_leave_es_state_unallocated():
    # Fresh model so the shared module fixture (touched by the opt-in tests
    # below) can't leak allocated ES state into this default-cfg check.
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
    cfg.pop_size = 8; cfg.num_generations = 3
    cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False; cfg.use_pinned_buffers = False; cfg.use_cufile = False
    cfg.save_path = None; cfg.checkpoint_interval = None
    cfg.lambda_1 = 0.0; cfg.lambda_2 = 0.0
    cfg.seed = 0; cfg.eval_jit_compile = False
    cfg.val_interval = 1; cfg.val_size = None
    cfg.toggle_regularization = False; cfg.per_type_regularization = False
    cfg.dipole_rij_power = 2
    cfg.optimizer_mode = "snes"            # pure SNES loop
    # Explicitly OFF — this test asserts the lazy ES state is never touched
    # when both features are disabled (don't inherit a user's class default).
    cfg.snes_mean_optimizer = "vanilla"
    cfg.snes_sigma_cumulation = False
    cfg.patience = None
    dataset, ti = collect(cfg)
    cfg.randomise(dataset); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(dataset, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    model = TNEP(cfg)
    snes = model.optimizer
    snes.fit(train, val)
    # Defaults (vanilla mean step, memoryless sigma) must never touch the
    # new branches, so their lazy state stays unallocated.
    assert snes._es_m is None
    assert snes._grad_sigma_ema is None


def test_mean_adam_reduces_loss(tiny_model):
    model, train, val = tiny_model
    snes = model.optimizer
    snes.cfg.snes_mean_optimizer = "adam"
    snes.cfg.num_generations = 40
    snes.cfg.patience = None
    try:
        hist = snes.fit(train, val)
        if isinstance(hist, tuple): hist = hist[0]
        tl = hist["train_loss"] if isinstance(hist, dict) else hist
        best = min(tl)
        print(f"[mean_adam] train_loss {tl[0]:.6e} -> {tl[-1]:.6e} (best {best:.6e})")
        # Adam-preconditioned mean + adaptive sigma can oscillate at the tail,
        # so success = reached a clearly lower loss at some point (>=10% better),
        # not that the final gen is monotone below the start.
        assert best <= 0.9 * tl[0], \
            f"Adam mean made no real progress: start {tl[0]:.4e}, best {best:.4e}"
    finally:
        snes.cfg.snes_mean_optimizer = "vanilla"
        snes.cfg.num_generations = 1


def test_sigma_cumulation_runs_and_stays_positive(tiny_model):
    model, train, val = tiny_model
    snes = model.optimizer
    snes.cfg.snes_sigma_cumulation = True
    snes.cfg.num_generations = 40
    snes.cfg.patience = None
    try:
        hist = snes.fit(train, val)
        if isinstance(hist, tuple): hist = hist[0]
        tl = hist["train_loss"] if isinstance(hist, dict) else hist
        print(f"[sigma_cumulation] train_loss {tl[0]:.6e} -> {tl[-1]:.6e}")
        assert float(tf.reduce_min(snes.sigma)) > 0.0
        assert tl[-1] <= tl[0], f"sigma cumulation did not reduce loss: {tl[0]} -> {tl[-1]}"
    finally:
        snes.cfg.snes_sigma_cumulation = False
        snes.cfg.num_generations = 1


def test_mu_eff_positive(tiny_model):
    model, _, _ = tiny_model
    snes = model.optimizer
    print(f"[mu_eff] = {snes._mu_eff}")
    assert snes._mu_eff > 1.0


def test_sigma_cumulation_bounded_long_run():
    # Regression for the cumulation sigma-explosion: over many generations of
    # sustained selection the old per-coordinate exp(rate*(p^2-1)) law drove
    # sigma off the mean-gradient path (no negative feedback) and compounded
    # to ~1e38 -> NaN. Must use pop_size=100 (the user's setting) — the bug is
    # population-dependent and does NOT trigger at the fixture's pop_size=8.
    # The corrected EMA-of-grad_sigma law keeps sigma bounded + loss finite.
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
    cfg.pop_size = 100; cfg.population_chunk_size = None; cfg.batch_chunk_size = None
    cfg.pin_data_to_cpu = True; cfg.cache_gradients_to_disk = False
    cfg.chunk_prefetch = False; cfg.use_pinned_buffers = False; cfg.use_cufile = False
    cfg.save_path = None; cfg.checkpoint_interval = None
    cfg.lambda_1 = 0.0; cfg.lambda_2 = 0.0
    cfg.toggle_regularization = False; cfg.per_type_regularization = False
    cfg.seed = 0; cfg.eval_jit_compile = False; cfg.val_interval = 10; cfg.val_size = None
    cfg.dipole_rij_power = 0; cfg.optimizer_mode = "snes"; cfg.patience = None
    cfg.num_generations = 300
    cfg.snes_sigma_cumulation = True
    ds, ti = collect(cfg); cfg.randomise(ds); cfg.dim_q = compute_dim_q(cfg)
    td, _, vd = split(ds, ti, cfg)
    train = pad_and_stack(td, num_types=cfg.num_types, pin_to_cpu=True)
    val = pad_and_stack(vd, num_types=cfg.num_types, pin_to_cpu=True)
    snes = TNEP(cfg).optimizer
    hist = snes.fit(train, val)
    h = hist[0] if isinstance(hist, tuple) else hist
    tl = h["train_loss"] if isinstance(h, dict) else h
    sig_max = float(tf.reduce_max(snes.sigma))
    print(f"[cumulation pop100 300gen] loss {tl[0]:.3e}->{tl[-1]:.3e} "
          f"max {max(tl):.3e} | sigma_max {sig_max:.3e}")
    assert np.all(np.isfinite(tl)), "train_loss went non-finite (divergence)"
    assert sig_max < 100.0 * float(cfg.init_sigma), \
        f"sigma exploded: max={sig_max:.3e} (init_sigma={cfg.init_sigma})"
    assert min(tl) <= tl[0], "cumulation made no progress"


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


def test_huber_train_rmse_is_true_rmse(tiny_model):
    # Regression: under Huber the reported/recorded train RMSE must be a true
    # RMSE at μ (comparable to val RMSE), NOT the mean Huber objective — which
    # is ~delta-scale and previously made "train RMSE" look spuriously tiny
    # next to the at-μ val RMSE.
    model, train, val = tiny_model
    snes = model.optimizer
    snes.cfg.loss_type = "huber"; snes.cfg.huber_delta = 1e-3
    snes.cfg.optimizer_mode = "snes"; snes._opt_mode = "snes"
    snes.cfg.num_generations = 30; snes.cfg.patience = None
    snes.cfg.val_interval = 5
    try:
        hist = snes.fit(train, val)
        h = hist[0] if isinstance(hist, tuple) else hist
        train_rmse = h["train_rmse"][-1]      # at-μ RMSE (the comparable metric)
        train_obj = h["train_loss"][-1]        # mean Huber objective (~delta scale)
        assert np.isfinite(train_rmse) and train_rmse > 0
        # it is a genuine RMSE, orders larger than the tiny Huber loss value
        assert train_rmse > 5.0 * train_obj, \
            f"train RMSE ({train_rmse:.3e}) ~ Huber objective ({train_obj:.3e}) — mislabeled?"
        # and on the same scale as the val RMSE (within a sane factor), not 100x off
        val_rmse = h["val_loss"][-1]
        assert 0.02 < train_rmse / val_rmse < 50.0, \
            f"train RMSE {train_rmse:.3e} not on the same scale as val RMSE {val_rmse:.3e}"
    finally:
        snes.cfg.loss_type = "mse"; snes.cfg.num_generations = 1; snes.cfg.val_interval = 1
