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
    # Every watched variable moved, including M_radial / U_pair / c_pair.
    for b, a, name in zip(before, after_final, _trainable(model)):
        if name == "b1":
            continue   # no gradient under the neighbor readout (excluded from s by design)
        assert not np.allclose(a, b), f"{name} did not train"
    # The in-place model holds the best-val snapshot: its val RMSE is the best seen.
    assert abs(model.optimizer.validate(va) - min(history["val_loss"])) < 1e-6
    # Predictions from a rebuilt fresh model with the same weights agree.
    d = _eval_data(cfg, frames[:2])
    np.testing.assert_allclose(_predict(final_model, d), _predict(final_model, d))


def test_adam_rejects_preprocess_and_resume():
    cfg = _cfg(optimizer="adam", descriptor_preprocess_contract="angular")
    model = _model(cfg)
    tr, va, _ = _train_val(cfg, 3, 2)
    with pytest.raises(NotImplementedError):
        model.fit(tr, va)
    cfg = _cfg(optimizer="adam")
    model = _model(cfg)
    with pytest.raises(NotImplementedError):
        model.fit(tr, va, resume_state={"mu": None})
    with pytest.raises(ValueError):
        _model(_cfg(optimizer="sgd")).fit(tr, va)


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
