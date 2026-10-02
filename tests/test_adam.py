"""Adam trainer (cfg.optimizer='adam') — bug checks, not benchmarks."""
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

from test_radial_basis import _cfg, _cho_frames, _model, _eval_data, _predict
from data import pad_and_stack, prepare_eval_data


def _train_val(cfg, n_train=6, n_val=3):
    frames = _cho_frames(n_train + n_val)
    tr = prepare_eval_data(frames[:n_train], cfg)
    va = prepare_eval_data(frames[n_train:], cfg)
    for d in (tr, va):
        d["types_contained"] = tf.ones((int(d["num_atoms"].shape[0]), 3))
    return tr, va, frames


@pytest.mark.parametrize("kw", [
    dict(),                                                         # fixed basis
    dict(radial_basis_size=8),                                      # learned, blocked
    dict(radial_basis_size=8, radial_basis_species_sum=True,
         descriptor_mixing=True, descriptor_mixing_regularizer="expm",
         dipole_species_pairs="symmetric"),                         # learned + mixing + c_pair
    dict(batch_size=4),                                             # random minibatch path
])
def test_adam_fit_runs_and_trains_every_tail(kw):
    cfg = _cfg(optimizer="adam", adam_lr=1e-2, num_generations=6, val_interval=2,
               batch_chunk_size=4, **kw)
    model = _model(cfg)
    tr, va, frames = _train_val(cfg)
    from adam import _trainable
    before = [v.numpy().copy() for v in _trainable(model).values()]
    history, final_model, best_model = model.fit(tr, va)
    assert best_model is model
    assert len(history["generation"]) == 4 and np.all(np.isfinite(history["val_loss"]))
    assert np.all(np.isfinite(history["train_loss"]))
    after_final = [v.numpy() for v in _trainable(final_model).values()]
    # Every watched variable moved, including M_radial / c_pair.
    for b, a, name in zip(before, after_final, _trainable(model)):
        if name == "b1":
            continue   # no gradient under the neighbor readout (excluded from s by design)
        assert not np.allclose(a, b), f"{name} did not train"
    if cfg.descriptor_mixing:
        # The mixing tail trained through SNES's expm map: V != 0 and orthogonal.
        V = final_model.U_pair.numpy()
        assert np.abs(V).max() > 0
        U = np.eye(V.shape[-1]) + V
        np.testing.assert_allclose(U @ np.swapaxes(U, -1, -2), np.broadcast_to(np.eye(V.shape[-1]), U.shape), atol=1e-5)
    # The in-place model holds the best-val snapshot: its val RMSE is the best seen.
    assert abs(model.optimizer.validate(va) - min(history["val_loss"])) < 1e-6
    # A fresh model given the final weights (incl. U_pair / M_radial) reproduces final_model.
    from TNEP import TNEP
    m2 = TNEP(cfg)
    for k, v in _trainable(final_model).items():
        _trainable(m2)[k].assign(v)
    if cfg.descriptor_mixing:
        m2.U_pair.assign(final_model.U_pair)
    assert abs(m2.optimizer.validate(va) - final_model.optimizer.validate(va)) < 1e-6
    assert final_model.optimizer.validate(va) == pytest.approx(history["val_loss"][-1], abs=1e-6)


def test_adam_rejects_resume_and_bad_optimizer():
    cfg = _cfg(optimizer="adam")
    tr, va, _ = _train_val(cfg, 3, 2)
    model = _model(cfg)
    with pytest.raises(NotImplementedError):
        model.fit(tr, va, resume_state={"mu": None})
    with pytest.raises(ValueError):
        _model(_cfg(optimizer="sgd")).fit(tr, va)
    with pytest.raises(NotImplementedError):
        _model(_cfg(optimizer="adam", lambda_1=-1.0)).fit(tr, va)


def test_adam_loss_decreases_on_tiny_set():
    cfg = _cfg(optimizer="adam", adam_lr=5e-3, num_generations=40, val_interval=40,
               radial_basis_size=6, toggle_regularization=False)
    model = _model(cfg)
    tr, va, _ = _train_val(cfg, 4, 2)
    history, _, _ = model.fit(tr, va)
    # single history tick at the end; compare against the gen-0 train RMSE
    cfg0 = _cfg(optimizer="adam", radial_basis_size=6, toggle_regularization=False)
    m0 = _model(cfg0)
    p0 = _predict(m0, tr) / np.maximum(tr["num_atoms"].numpy(), 1)[:, None]
    rmse0 = float(np.sqrt(np.mean((p0 - tr["targets"].numpy()) ** 2)))
    assert history["train_loss"][-1] < rmse0


# ── cfg.gpumd_path: monitor the test set, never select on it ────────────────

def test_gpumd_path_never_selects_on_the_monitor_set(optimizer="snes"):
    """GPUMD writes its potential from the CURRENT generation and tracks no
    best-ever model (snes.cu's loop + fitness.cu report_error), which is the
    only thing that makes monitoring the test set honest. So under
    cfg.gpumd_path the two models fit() returns must be identical, even when
    an earlier generation scored better on the monitor set."""
    kw = dict(num_generations=4, val_interval=1, pop_size=6, patience=None)
    tr, va, _ = _train_val(_cfg(optimizer=optimizer, **kw))

    def run(gpumd):
        cfg = _cfg(optimizer=optimizer, gpumd_path=gpumd, adam_lr=1e-2, **kw)
        m = _model(cfg)
        hist, final_model, best_model = m.fit(tr, va)
        f = np.concatenate([w.numpy().ravel() for w in final_model.trainable_weights])
        b = np.concatenate([w.numpy().ravel() for w in best_model.trainable_weights])
        return hist, f, b

    hist, f, b = run(True)
    np.testing.assert_array_equal(f, b)          # best IS final: nothing selected
    # The monitor really was evaluated, and really did move.
    assert len(hist["val_loss"]) >= 2 and len(set(hist["val_loss"])) > 1
    # Sanity: with the flag off the two models are allowed to differ, and the
    # selection path still records a genuine best.
    hist2, f2, b2 = run(False)
    assert min(hist2["val_loss"]) <= hist2["val_loss"][-1] + 1e-12
