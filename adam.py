"""Adam trainer for TNEP (cfg.optimizer = "adam").

Gradient-based alternative to SNES: tf.GradientTape through predict_batch,
so every learnable tensor the forward reads — the ANN weights, c_pair, the
descriptor-mixing residual U_pair (trained as a dense V; the cayley/expm
skew parameterisations are SNES-only) and the learned radial basis
M_radial — is trained by autodiff, including through soap_from_raw.

Same (history, final_model, best_val_model) contract and history keys as
SNES.fit so the driver, plots and save_history work unchanged (sigma
columns are 0). Not supported: checkpoint/resume, descriptor_preprocess_contract.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import tensorflow as tf

from loss_functions import squared_error_per_structure


def _tf_var(v):
    """Keras-3 layer weights wrap a tf.Variable in `.value`; the tape, the
    optimizer and tf.function arguments need the tf.Variable itself."""
    return v.value if isinstance(getattr(v, "value", None), tf.Variable) else v


def _trainable(model) -> dict:
    """name -> tf.Variable for everything predict_batch reads, in a fixed
    order (the best-val snapshot relies on it). M_radial is a non-trainable
    tf.Variable by construction (SNES owns it), so the tape watches the list
    explicitly."""
    vs = {"W0": model.W0, "b0": model.b0, "W1": model.W1, "b1": model.b1}
    if model._neighbor_readout and model._c_learnable:
        vs["c_pair"] = model.c_pair
    if model.cfg.target_mode == 2:
        vs.update(W0_pol=model.W0_pol, b0_pol=model.b0_pol,
                  W1_pol=model.W1_pol, b1_pol=model.b1_pol)
    if model.descriptor_mixing and model.U_pair is not None:
        vs["U_pair"] = model.U_pair
    if model.M_radial is not None:
        vs["M_radial"] = model.M_radial
    return {k: _tf_var(v) for k, v in vs.items()}


def _ann_core(model) -> list:
    """Weights the L1/L2 penalty covers — the ANN(s) only, matching SNES's
    exclusion of the c_pair / U_pair / M_radial tails."""
    core = [model.W0, model.b0, model.W1]
    if model.cfg.target_mode == 2:
        core += [model.W0_pol, model.b0_pol, model.W1_pol]
    return [_tf_var(v) for v in core]


def fit_adam(model, train_data: dict, val_data: dict,
             plot_callback: Callable | None = None) -> tuple:
    cfg = model.cfg
    if model.descriptor_preprocess_contract != "off":
        raise NotImplementedError(
            "optimizer='adam' does not support descriptor_preprocess_contract "
            "(its summed-entry masking is SNES-specific); set it to 'off'.")
    from data import prefetched_chunks, slice_and_complete_chunk

    snes = model.optimizer            # resolved lambdas, validate(), tf_rng
    named = _trainable(model)
    variables = list(named.values())
    core = _ann_core(model)
    n_core = float(sum(int(np.prod(v.shape)) for v in core))
    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg.adam_lr))
    scale_preds = bool(cfg.scale_targets and cfg.target_mode == 1)
    pol_w = snes._pol_weights
    reg_on = bool(cfg.toggle_regularization)
    T_dim = int(train_data["targets"].shape[1])
    S_train = int(train_data["num_atoms"].shape[0])

    def _forward(chunk):
        # Only tf.Variables / tensors go into the forward (Keras variables
        # are not valid tf.function arguments).
        U = named.get("U_pair")
        W0 = model._W0_eff(named["W0"], U) if U is not None else named["W0"]
        W0p = None
        if cfg.target_mode == 2:
            W0p = (model._W0_eff(named["W0_pol"], U) if U is not None
                   else named["W0_pol"])
        preds = model.predict_batch(
            chunk["descriptors"], chunk["grad_values"],
            chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
            chunk["positions"], chunk["Z_int"], chunk["boxes"],
            chunk["atom_mask"],
            W0, named["b0"], named["W1"], named["b1"],
            W0p, named.get("b0_pol"), named.get("W1_pol"), named.get("b1_pol"),
            W_atom=chunk.get("V_atom"),
            c_pair=named.get("c_pair"),
            M_radial=named.get("M_radial"))
        if scale_preds:
            num_atoms = tf.reduce_sum(chunk["atom_mask"], axis=1)
            preds = preds / tf.maximum(num_atoms, 1.0)[:, tf.newaxis]
        return preds

    @tf.function(reduce_retracing=True)
    def _step(chunk):
        with tf.GradientTape() as tape:
            tape.watch(variables)
            diff = _forward(chunk) - chunk["targets"]
            mse = tf.reduce_mean(squared_error_per_structure(diff, pol_w)) / T_dim
            l1 = l2 = tf.constant(0.0)
            if reg_on:
                flat = tf.concat([tf.reshape(v, [-1]) for v in core], axis=0)
                l1 = snes.lambda_1 * tf.reduce_sum(tf.abs(flat)) / n_core
                l2 = snes.lambda_2 * tf.sqrt(tf.reduce_sum(tf.square(flat)) / n_core)
            loss = mse + l1 + l2
        grads = tape.gradient(loss, variables)
        # b1 has no gradient under the neighbor readout (excluded from s by
        # design); skip None grads instead of letting Keras warn every step.
        opt.apply_gradients([(g, v) for g, v in zip(grads, variables)
                             if g is not None])
        return mse, l1, l2

    struct_chunk = cfg.batch_chunk_size if cfg.batch_chunk_size is not None else S_train
    full_ranges = [(s, min(s + struct_chunk, S_train))
                   for s in range(0, S_train, struct_chunk)]

    def _batches():
        """One generation's Adam steps: every train chunk in order (full batch)
        or one random batch of cfg.batch_size structures."""
        if cfg.batch_size is None:
            for _, _, chunk in prefetched_chunks(
                    train_data, full_ranges, pin_to_cpu=cfg.pin_data_to_cpu):
                yield chunk
        else:
            idx = tf.argsort(snes.tf_rng.uniform(shape=[S_train]))[:cfg.batch_size]
            yield slice_and_complete_chunk(train_data, idx)

    history = {k: [] for k in (
        "generation", "train_loss", "train_rmse", "val_loss", "L1", "L2",
        "best_rmse", "worst_rmse",
        "sigma_min", "sigma_max", "sigma_mean", "sigma_median")}
    history["timing"] = {k: [] for k in (
        "sample_batch", "evaluate", "rank_update", "validate", "overhead")}
    best_val = float("inf")
    best_snapshot = [v.numpy() for v in variables]
    gens_without_improvement = 0
    print(f"Fitting model with Adam (lr={cfg.adam_lr}, "
          f"{sum(int(np.prod(v.shape)) for v in variables)} trainable params)...")

    for gen in range(cfg.num_generations):
        t0 = time.perf_counter()
        mses, l1s, l2s = [], [], []
        for chunk in _batches():
            mse, l1, l2 = _step(chunk)
            mses.append(mse); l1s.append(l1); l2s.append(l2)
        train_rmse = float(np.sqrt(np.mean([float(m) for m in mses])))
        gen_l1, gen_l2 = float(np.mean([float(x) for x in l1s])), float(np.mean([float(x) for x in l2s]))
        t1 = time.perf_counter()

        do_val = (gen % cfg.val_interval == 0) or (gen == cfg.num_generations - 1)
        if do_val:
            val_rmse = snes.validate(val_data)
            t2 = time.perf_counter()
            if val_rmse < best_val:
                best_val = val_rmse
                best_snapshot = [v.numpy() for v in variables]
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1
            history["generation"].append(gen)
            history["train_loss"].append(train_rmse)
            history["train_rmse"].append(train_rmse)
            history["val_loss"].append(val_rmse)
            history["L1"].append(gen_l1)
            history["L2"].append(gen_l2)
            history["best_rmse"].append(train_rmse)
            history["worst_rmse"].append(train_rmse)
            for k in ("sigma_min", "sigma_max", "sigma_mean", "sigma_median"):
                history[k].append(0.0)
            history["timing"]["sample_batch"].append(0.0)
            history["timing"]["evaluate"].append(t1 - t0)
            history["timing"]["rank_update"].append(0.0)
            history["timing"]["validate"].append(t2 - t1)
            history["timing"]["overhead"].append(0.0)
            print(f"gen {gen + 1}/{cfg.num_generations}  train RMSE: {train_rmse:.6f}"
                  f"  val RMSE: {val_rmse:.6f}  best val RMSE: {best_val:.6f}"
                  f"  L1: {gen_l1:.2e}  L2: {gen_l2:.2e}", flush=True)
            if plot_callback is not None and cfg.plot_interval and gen > 0 \
                    and gen % cfg.plot_interval == 0:
                plot_callback(history, gen)
            if cfg.patience is not None and gens_without_improvement >= cfg.patience:
                print(f"Early stopping at gen {gen} "
                      f"(no improvement for {cfg.patience} validation ticks)")
                break

    # final_model = a fresh model with the last weights; the in-place model is
    # restored to best-val (same convention as SNES.fit).
    from TNEP import TNEP
    final_model = TNEP(cfg)
    for dst, src in zip(_trainable(final_model).values(), variables):
        dst.assign(src)
    for v, snap in zip(variables, best_snapshot):
        v.assign(snap)
    return history, final_model, model
