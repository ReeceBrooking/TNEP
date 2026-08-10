import numpy as np
import tensorflow as tf
import pytest

from descriptor_scaling import (VALID_MODES, channel_multipliers,
                                expand_to_w0_coords)


def _stats(mean, std, rms=None):
    mean = np.asarray(mean, float)
    std = np.asarray(std, float)
    return {"mean": mean, "std": std,
            "rms": np.asarray(rms if rms is not None else std, float)}


def test_std_mode_boosts_small_channels():
    m = channel_multipliers(_stats([1.0, 1.0], [1.0, 0.01]), "std",
                            clamp=1e9, exponent=1.0)
    assert m[1] / m[0] == pytest.approx(100.0, rel=1e-5)


def test_exponent_softens_in_log_space():
    s = _stats([1.0, 1.0], [1.0, 0.01])
    half = channel_multipliers(s, "std", clamp=1e9, exponent=0.5)
    assert half[1] / half[0] == pytest.approx(10.0, rel=1e-5)
    off = channel_multipliers(s, "std", clamp=1e9, exponent=0.0)
    np.testing.assert_allclose(off, np.ones(2), rtol=1e-6)


def test_multipliers_have_unit_geometric_mean():
    rng = np.random.default_rng(0)
    s = rng.uniform(1e-4, 1e-1, size=64)
    m = channel_multipliers(_stats(np.ones(64), s), "std",
                            clamp=1e9, exponent=1.0)
    assert np.exp(np.log(m).mean()) == pytest.approx(1.0, rel=1e-6)


def test_clamp_bounds_max_over_min_ratio():
    """clamp is the max ratio between largest and smallest multiplier."""
    m = channel_multipliers(_stats(np.ones(3), [1.0, 1e-6, 1e3]), "std",
                            clamp=10.0, exponent=1.0)
    assert m.max() / m.min() <= 10.0 + 1e-6
    assert np.exp(np.log(m).mean()) == pytest.approx(1.0, rel=1e-6)


def test_cv_mode_penalises_the_channels_std_mode_boosts():
    """Real measured values: a large-std channel and a small-std one.
    "std" boosts the small channel; "cv" boosts the large one instead."""
    s = _stats(mean=[0.328, 7.2e-4], std=[0.257, 7.2e-4], rms=[0.417, 1.0e-3])
    m_std = channel_multipliers(s, "std", clamp=1e9, exponent=1.0)
    m_cv = channel_multipliers(s, "cv", clamp=1e9, exponent=1.0)
    assert m_std[1] > m_std[0]
    assert m_cv[1] < m_cv[0]


def test_zero_std_channel_stays_finite():
    m = channel_multipliers(_stats(np.ones(2), [1.0, 0.0]), "std",
                            clamp=8.0, exponent=1.0)
    assert np.all(np.isfinite(m)) and np.all(m > 0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="no statistic"):
        channel_multipliers(_stats([1.0], [1.0]), "zscore", 8.0, 1.0)


def test_block_granularity_gives_one_multiplier_per_block():
    """Required under descriptor_mixing: a constant-per-block diag(m)
    commutes with the block-diagonal U, making the preconditioning exact
    for all U instead of only at generation 0."""
    s = _stats(np.ones(4), [1.0, 4.0, 0.01, 0.04])
    blocks = [np.array([0, 1]), np.array([2, 3])]
    m = channel_multipliers(s, "std", clamp=1e9, exponent=1.0, blocks=blocks)
    assert m[0] == pytest.approx(m[1])
    assert m[2] == pytest.approx(m[3])
    assert m[2] > m[0]


def test_expand_to_w0_coords_repeats_per_channel_over_H():
    """W0 is [T, Q, H] row-major, so channel k owns H consecutive
    coordinates within each of the T type blocks."""
    out = expand_to_w0_coords(np.array([2.0, 3.0]), num_types=2, num_neurons=3)
    np.testing.assert_array_equal(out, [2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3])


def test_channel_stats_match_a_single_pass():
    """Batched accumulation equals a one-shot numpy computation."""
    from data import _accumulate_channel_stats, _finalize_channel_stats
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(7, 5)), rng.normal(size=(11, 5)),
              rng.normal(size=(3, 5))]
    acc = None
    for b in blocks:
        acc = _accumulate_channel_stats(acc, b)
    got = _finalize_channel_stats(acc)
    ref = np.concatenate(blocks, axis=0)
    np.testing.assert_allclose(got["mean"], ref.mean(0), rtol=1e-6)
    np.testing.assert_allclose(got["std"], ref.std(0), rtol=1e-6)
    np.testing.assert_allclose(got["rms"], np.sqrt((ref ** 2).mean(0)), rtol=1e-6)
    assert got["count"] == 21


def _tiny_cfg(**kw):
    from TNEPconfig import TNEPconfig
    from DescriptorBuilderGPU import compute_dim_q
    cfg = TNEPconfig()
    cfg.target_mode = 1
    cfg.l_max = cfg.alpha_max = 2
    cfg.num_neurons = 3
    cfg.types = [8, 1]; cfg.num_types = 2; cfg.type_map = {8: 0, 1: 1}
    cfg.dim_q = compute_dim_q(cfg)
    cfg.descriptor_mixing = False
    # Pin the preconditioning fields rather than inheriting the class
    # defaults: these are live experiment knobs, and a test suite that
    # changes behaviour when someone edits TNEPconfig is not a test suite.
    cfg.descriptor_sigma_scaling = "off"
    cfg.descriptor_weight_reparam = "off"
    cfg.descriptor_scaling_exponent = 0.5
    cfg.descriptor_scaling_clamp = 64.0
    for k, v in kw.items():
        setattr(cfg, k, v)
    q = int(cfg.dim_q)
    cfg._descriptor_channel_stats = {
        "mean": np.ones(q, np.float32), "std": np.ones(q, np.float32),
        "rms": np.ones(q, np.float32), "count": 100}
    return cfg


def test_both_modes_at_once_raises():
    from TNEP import TNEP
    cfg = _tiny_cfg(descriptor_sigma_scaling="std",
                    descriptor_weight_reparam="std")
    with pytest.raises(ValueError, match="mutually exclusive"):
        TNEP(cfg)


def test_pes_mode_raises():
    from TNEP import TNEP
    cfg = _tiny_cfg(target_mode=0, descriptor_sigma_scaling="std")
    with pytest.raises(ValueError, match="target_mode"):
        TNEP(cfg)


def test_missing_statistics_raise_clearly():
    from TNEP import TNEP
    cfg = _tiny_cfg(descriptor_sigma_scaling="std")
    del cfg._descriptor_channel_stats
    with pytest.raises(ValueError, match="statistics"):
        TNEP(cfg)


def test_bad_mode_string_raises():
    from TNEP import TNEP
    with pytest.raises(ValueError, match="not in"):
        TNEP(_tiny_cfg(descriptor_sigma_scaling="zscore"))


def test_mode_a_scales_only_the_w0_block():
    """sigma is multiplied on W0 coordinates and left alone elsewhere."""
    from TNEP import TNEP
    # Override exponent AND clamp: at the defaults (0.5, 64.0) the softening
    # halves the log-ratio and the clamp then truncates it, giving ~8.9.
    cfg = _tiny_cfg(descriptor_sigma_scaling="std",
                    descriptor_scaling_exponent=1.0,
                    descriptor_scaling_clamp=1e9)
    q, h, t = int(cfg.dim_q), cfg.num_neurons, cfg.num_types
    std = np.ones(q, np.float32); std[0] = 0.01          # one small channel
    cfg._descriptor_channel_stats = {
        "mean": np.ones(q, np.float32), "std": std,
        "rms": np.ones(q, np.float32), "count": 100}
    s = TNEP(cfg).optimizer.sigma.numpy()
    n_w0 = t * q * h
    # channel 0 vs channel 1, first type block
    assert s[0] / s[h] == pytest.approx(100.0, rel=1e-4)
    # b0/W1/b1 coordinates untouched
    np.testing.assert_allclose(s[n_w0:], cfg.init_sigma, rtol=1e-6)


def test_mode_a_scales_the_pol_block_too():
    """target_mode=2 has a second W0 at offset n_primary — the only place the
    offset arithmetic is exercised, and where both review rounds found bugs."""
    from TNEP import TNEP
    cfg = _tiny_cfg(target_mode=2, descriptor_sigma_scaling="std",
                    descriptor_scaling_exponent=1.0,
                    descriptor_scaling_clamp=1e9)
    q, h = int(cfg.dim_q), cfg.num_neurons
    std = np.ones(q, np.float32); std[0] = 0.01
    cfg._descriptor_channel_stats = {
        "mean": np.ones(q, np.float32), "std": std,
        "rms": np.ones(q, np.float32), "count": 100}
    snes = TNEP(cfg).optimizer
    s = snes.sigma.numpy()
    p = snes.n_primary
    assert s[0] / s[h] == pytest.approx(100.0, rel=1e-4)            # main ANN
    assert s[p] / s[p + h] == pytest.approx(100.0, rel=1e-4)        # pol ANN


def test_mode_a_off_leaves_sigma_uniform():
    from TNEP import TNEP
    s = TNEP(_tiny_cfg()).optimizer.sigma.numpy()
    np.testing.assert_allclose(s, s[0], rtol=1e-7)


def test_mode_b_is_a_pure_reparameterisation():
    """Effective W0 at generation 0 is identical with reparam on and off.

    _build_mu_init emits Glorot/m and reconstruct multiplies by m, so the
    round trip is the identity. If it is not, enabling the option silently
    changes the initial model.
    """
    from TNEP import TNEP
    outs = {}
    for mode in ("off", "std"):
        cfg = _tiny_cfg(descriptor_weight_reparam=mode, seed=1234)
        q = int(cfg.dim_q)
        std = np.linspace(0.01, 1.0, q).astype(np.float32)
        cfg._descriptor_channel_stats = {
            "mean": np.ones(q, np.float32), "std": std,
            "rms": np.ones(q, np.float32), "count": 100}
        snes = TNEP(cfg).optimizer
        outs[mode] = snes._split_reconstructed(
            snes.reconstruct_params_tf(snes.mu))["W0"].numpy()
    np.testing.assert_allclose(outs["off"], outs["std"], rtol=2e-5, atol=1e-7)


def test_checkpoint_round_trip_preserves_the_multiplier(tmp_path):
    from model_io import save_checkpoint, load_checkpoint
    cfg = _tiny_cfg(descriptor_weight_reparam="std")
    scale = np.linspace(0.5, 2.0, int(cfg.dim_q)).astype(np.float32)
    state = {"mu": np.zeros(4, np.float32), "sigma": np.ones(4, np.float32),
             "best_mu": np.zeros(4, np.float32), "best_sigma": None,
             "best_val_loss": 1.0, "gens_without_improvement": 0,
             "descriptor_scale": scale}
    p = str(tmp_path / "ckpt.h5")
    save_checkpoint(p, cfg, state, {"generation": [0], "val_loss": [1.0]}, 0)
    cfg2, _ = load_checkpoint(p)
    np.testing.assert_array_equal(cfg2.descriptor_scale, scale)


def test_resume_without_saved_multiplier_raises(tmp_path):
    """Hand-craft the broken file: save normally, then delete the dataset.
    The new save_checkpoint can never produce this combination itself."""
    import h5py
    from model_io import save_checkpoint, load_checkpoint
    cfg = _tiny_cfg(descriptor_weight_reparam="std")
    scale = np.ones(int(cfg.dim_q), np.float32)
    state = {"mu": np.zeros(4, np.float32), "sigma": np.ones(4, np.float32),
             "best_mu": np.zeros(4, np.float32), "best_sigma": None,
             "best_val_loss": 1.0, "gens_without_improvement": 0,
             "descriptor_scale": scale}
    p = str(tmp_path / "ckpt.h5")
    save_checkpoint(p, cfg, state, {"generation": [0], "val_loss": [1.0]}, 0)
    with h5py.File(p, "a") as f:
        del f["snes"]["descriptor_scale"]
    with pytest.raises(ValueError, match="descriptor_scale"):
        load_checkpoint(p)


def test_model_round_trip_restores_the_multiplier(tmp_path):
    """A preconditioned model must be loadable at all."""
    from TNEP import TNEP
    from model_io import save_model, load_model
    import glob
    cfg = _tiny_cfg(descriptor_sigma_scaling="std")
    m = TNEP(cfg)
    save_model(m, cfg, path=str(tmp_path))
    m2 = load_model(glob.glob(str(tmp_path / "*.h5"))[0])
    np.testing.assert_allclose(m2.optimizer._chan_mult, m.optimizer._chan_mult,
                               rtol=1e-6)


def test_block_multiplier_commutes_with_mixing():
    """The property the whole design rests on.

    W0's Q axis indexes the MIXED descriptor (z = q.U^T.W0), while the
    multiplier is measured on RAW channels. Those coincide only at
    generation 0 unless diag(m) commutes with U. U is block-diagonal over
    (pair, l), and block granularity gives every channel in a block the
    same multiplier, so diag(m) restricted to a block is m*I and the two
    commute exactly -- for all U, not just U = I.

    Measured within-(pair,l) std spread is a median 24.8x, so a per-channel
    multiplier under mixing would drift materially. If this test fails,
    block granularity is not engaging and mode A/B must be guarded to
    descriptor_mixing=False.
    """
    from TNEP import TNEP
    cfg = _tiny_cfg(descriptor_mixing=True, descriptor_sigma_scaling="std",
                    descriptor_scaling_exponent=1.0,
                    descriptor_scaling_clamp=1e9)
    q = int(cfg.dim_q)
    rng = np.random.default_rng(3)
    cfg._descriptor_channel_stats = {
        "mean": np.ones(q, np.float32),
        "std": rng.uniform(1e-3, 1e-1, q).astype(np.float32),
        "rms": np.ones(q, np.float32), "count": 100}
    model = TNEP(cfg)
    m = model.optimizer._chan_mult
    assert m is not None

    # Give U a non-identity value; V=0 would make U=I and pass trivially.
    model.U_pair.assign(
        rng.normal(size=model.U_pair.shape).astype(np.float32) * 0.2)

    W0 = model.W0.numpy()                       # [T, Q, H]
    scaled = W0 * m[np.newaxis, :, np.newaxis]  # diag(m) applied first
    lhs = model._W0_eff(tf.constant(scaled)).numpy()
    rhs = model._W0_eff(model.W0).numpy() * m[np.newaxis, :, np.newaxis]
    denom = max(float(np.abs(rhs).max()), 1e-12)
    assert float(np.abs(lhs - rhs).max()) / denom < 1e-5, (
        "diag(m) does not commute with the mixing U -- block granularity "
        "is not engaging")


def test_separate_pol_rotation_grows_the_parameter_vector():
    """The second rotation occupies its own tail, exactly n_U_pair wide."""
    from TNEP import TNEP
    shared = TNEP(_tiny_cfg(target_mode=2, descriptor_mixing=True)).optimizer
    sep = TNEP(_tiny_cfg(target_mode=2, descriptor_mixing=True,
                         descriptor_mixing_separate_pol=True)).optimizer
    assert shared.n_U_pair_pol == 0
    assert sep.n_U_pair_pol == sep.n_U_pair > 0
    assert sep.dim == shared.dim + shared.n_U_pair


def test_separate_pol_rotation_is_identity_at_gen_zero():
    """Both rotations init to zero (U = I), so enabling the option must not
    change generation-0 predictions."""
    from TNEP import TNEP
    outs = {}
    for sep in (False, True):
        cfg = _tiny_cfg(target_mode=2, descriptor_mixing=True, seed=7,
                        descriptor_mixing_separate_pol=sep)
        snes = TNEP(cfg).optimizer
        named = snes._split_reconstructed(
            snes.reconstruct_params_tf(snes.mu))
        outs[sep] = (named["W0"].numpy(), named["W0_pol"].numpy())
    np.testing.assert_allclose(outs[False][0], outs[True][0], rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(outs[False][1], outs[True][1], rtol=2e-5, atol=1e-7)


def test_separate_pol_rotation_is_actually_independent():
    """With the option on, perturbing only the pol tail must change W0_pol's
    effective weights and leave the tensor head's untouched."""
    from TNEP import TNEP
    cfg = _tiny_cfg(target_mode=2, descriptor_mixing=True, seed=7,
                    descriptor_mixing_separate_pol=True)
    snes = TNEP(cfg).optimizer
    mu = snes.mu.numpy().copy()
    start = snes.n_anns_total + snes.n_U_pair
    mu[start:start + snes.n_U_pair_pol] += 0.3          # pol rotation only
    base = snes._split_reconstructed(snes.reconstruct_params_tf(snes.mu))
    pert = snes._split_reconstructed(
        snes.reconstruct_params_tf(tf.constant(mu)))
    np.testing.assert_allclose(base["W0"].numpy(), pert["W0"].numpy(),
                               rtol=1e-6, atol=1e-8)
    assert not np.allclose(base["U_pair_pol"].numpy(),
                           pert["U_pair_pol"].numpy())
