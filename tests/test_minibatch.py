import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
import tensorflow as tf

from test_second_layer import _tiny_cfg, _build


def test_sample_minibatch_coo_consistency():
    """sample_minibatch on a synthetic dict: struct keys gathered on axis 0,
    COO pair arrays re-indexed per structure, batch-local pair_struct/struct_ptr."""
    from SNES import sample_minibatch
    # 3 structures with 2, 1, 3 pairs respectively.
    train = {
        "descriptors": tf.constant([[0.0], [1.0], [2.0]]),
        "targets": tf.constant([[10.0], [11.0], [12.0]]),
        "struct_ptr": tf.constant([0, 2, 3, 6]),
        "grad_values": tf.constant([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]),
        "pair_atom": tf.constant([0, 1, 0, 0, 1, 2]),
        "pair_gidx": tf.constant([5, 6, 7, 8, 9, 10]),
    }
    batch = sample_minibatch(train, tf.constant([2, 0]))
    np.testing.assert_array_equal(batch["descriptors"].numpy(), [[2.0], [0.0]])
    np.testing.assert_array_equal(batch["targets"].numpy(), [[12.0], [10.0]])
    # structure 2's pairs (rows 3,4,5) then structure 0's (rows 0,1)
    np.testing.assert_array_equal(
        batch["grad_values"].numpy().ravel(), [3.0, 4.0, 5.0, 0.0, 1.0])
    np.testing.assert_array_equal(batch["pair_atom"].numpy(), [0, 1, 2, 0, 1])
    np.testing.assert_array_equal(batch["pair_gidx"].numpy(), [8, 9, 10, 5, 6])
    np.testing.assert_array_equal(batch["pair_struct"].numpy(), [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(batch["struct_ptr"].numpy(), [0, 3, 5])
    assert "pair_atom" in batch and "_W_atom" not in batch


def test_adam_minibatch_trains():
    """Adam honors cfg.batch_size: minibatch steps run (varying pair counts
    through the compiled loss/grad fn) and reduce the full-train loss."""
    cfg = _tiny_cfg(num_hidden_layers=2, mixing=True)
    cfg.batch_size = 4
    cfg.num_generations = 30
    cfg.adam_learning_rate = 1e-2
    model, train, val = _build(cfg)
    from Adam import Adam
    opt = Adam(model)
    l0 = float(opt._batch_loss(train))
    history, final_model, best = opt.fit(train, val)
    assert float(opt._batch_loss(train)) < l0
    assert np.all(np.isfinite(history["val_loss"]))
    m, _ = best.score(val)
    assert float(m["rmse"]) < 1.0
