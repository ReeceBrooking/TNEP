"""Adam trainer for TNEP (cfg.optimizer = "adam").

Gradient-based alternative to SNES: tf.GradientTape through predict_batch,
so every learnable tensor the forward reads — the ANN weights, c_pair, the
descriptor-mixing tail (the same flat skew/dense parameters SNES searches,
mapped to V by SNES._reconstruct_one_mixing_layer, so cayley/expm stay
structural), the learned radial basis M_radial and the NEP contraction
coefficients c_r / c_a — is trained by autodiff,
including through soap_from_raw. Objective = RMSE + L1 + L2 on the ANN core,
the same quantity SNES ranks on.

Same (history, final_model, best_val_model) contract and history keys as
SNES.fit so the driver, plots and save_history work unchanged (sigma
columns are 0). Not supported: checkpoint/resume, dynamic lambda (-1).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import tensorflow as tf

from loss_functions import squared_error_per_structure
from nep_descriptor import EXPERT_WEIGHTS, nep_pair_kwargs


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
    if model._experts_on:
        vs.update({k: getattr(model, k) for k in EXPERT_WEIGHTS})
    if model._neighbor_readout and model._c_learnable:
        vs["c_pair"] = model.c_pair
    if model.cfg.target_mode == 2:
        vs.update(W0_pol=model.W0_pol, b0_pol=model.b0_pol,
                  W1_pol=model.W1_pol, b1_pol=model.b1_pol)
    if model.M_radial is not None:
        vs["M_radial"] = model.M_radial
    if (getattr(model, "sum_delta", None) is not None
            and bool(getattr(model.cfg, "sum_delta_learn", False))):
        vs["sum_delta"] = model.sum_delta          # δ: unpenalised (_ann_core skips it)
    if model.c_r is not None:
        vs["c_r"] = model.c_r
        vs["c_a"] = model.c_a
    return {k: _tf_var(v) for k, v in vs.items()}


def _ann_core(model) -> list:
    """Weights the L1/L2 penalty covers — the ANN(s) only (b1 and the
    c_pair / U_pair / M_radial / c_r / c_a tails excluded, as in SNES's multi-type
    per-type penalty)."""
    core = [model.W0, model.b0, model.W1]
    if model._experts_on:
        core += list(model.expert_weights)
    if model.cfg.target_mode == 2:
        core += [model.W0_pol, model.b0_pol, model.W1_pol]
    return [_tf_var(v) for v in core]


def fit_adam(model, train_data: dict, val_data: dict,
             plot_callback: Callable | None = None) -> tuple:
    cfg = model.cfg
    if cfg.lambda_1 == -1 or cfg.lambda_2 == -1:
        raise NotImplementedError(
            "optimizer='adam' does not support dynamic lambda (-1); give "
            "lambda_1/lambda_2 a value or None (SNES auto value).")
    from data import prefetched_chunks, slice_and_complete_chunk

    snes = model.optimizer            # resolved lambdas, validate(), tf_rng
    named = _trainable(model)
    # Descriptor mixing: train SNES's flat tail (skew upper triangles under
    # cayley/expm, dense entries under "off") and map it to V exactly as
    # SNES does, so the parameterisation stays structural. V = 0 at init.
    mix_flat = None
    if model.descriptor_mixing and model.U_pair is not None:
        mix_flat = tf.Variable(tf.zeros([snes.n_U_pair], tf.float32),
                               trainable=False, name="mix_flat")
        named["mix_flat"] = mix_flat

    def _mixing_V():
        return snes._reconstruct_one_mixing_layer(
            mix_flat, False, snes._mix_per_type, cfg.num_types)

    def _sync_U_pair():
        """Write the current V into model.U_pair (validate/score/save read it)."""
        if mix_flat is not None:
            _tf_var(model.U_pair).assign(_mixing_V())

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
        U = _mixing_V() if mix_flat is not None else None
        b0 = model._b0_eff(named["b0"], named["W0"])    # centring fold (stored W0)
        W0 = model._W0_eff(named["W0"], U)     # U fold (if mixing) + q-scale fold
        b0p = model._b0_eff(named["b0_pol"], named["W0_pol"]) if cfg.target_mode == 2 else None
        W0p = model._W0_eff(named["W0_pol"], U) if cfg.target_mode == 2 else None
        preds = model.predict_batch(
            chunk["descriptors"], chunk["grad_values"],
            chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
            chunk["positions"], chunk["Z_int"], chunk["boxes"],
            chunk["atom_mask"],
            W0, b0, named["W1"], named["b1"],
            W0p, b0p, named.get("W1_pol"), named.get("b1_pol"),
            W_atom=chunk.get("V_atom"),
            c_pair=named.get("c_pair"),
            M_radial=named.get("M_radial"),
            c_r=named.get("c_r"), c_a=named.get("c_a"),
            experts=(tuple(map(_tf_var, model.expert_weights))   # Keras vars → tf.Variables
                     if model._experts_on else None),
            sum_delta=named.get("sum_delta"),
            nep_raw=chunk.get("nep_raw"), **nep_pair_kwargs(chunk))
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
            # SNES ranks on RMSE + penalty; use the same objective so the
            # lambda scale means the same thing under both optimizers.
            loss = tf.sqrt(mse + 1e-12) + l1 + l2
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
    last_plot = 0
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
            _sync_U_pair()
            val_rmse = snes.validate(val_data)
            t2 = time.perf_counter()
            if not np.isfinite(val_rmse) or not np.isfinite(train_rmse):
                print(f"Adam diverged at gen {gen} (train RMSE {train_rmse}, "
                      f"val RMSE {val_rmse}); stopping — lower adam_lr. "
                      f"Returning the best-val weights so far.")
                break
            if val_rmse < best_val:
                best_val = val_rmse
                best_snapshot = [v.numpy() for v in variables]
                gens_without_improvement = 0
            else:
                gens_without_improvement += 1
            history["generation"].append(gen)
            history["train_loss"].append(train_rmse + gen_l1 + gen_l2)  # objective, as SNES
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
            # Fires on the val tick at/after each plot_interval boundary
            # (val_interval need not divide plot_interval).
            if (plot_callback is not None and cfg.plot_interval
                    and (gen + 1) // cfg.plot_interval > last_plot):
                last_plot = (gen + 1) // cfg.plot_interval
                plot_callback(history, gen + 1)
            if cfg.patience is not None and gens_without_improvement >= cfg.patience:
                print(f"Early stopping at gen {gen} "
                      f"(no improvement for {cfg.patience} validation ticks)")
                break

    # final_model = a fresh model with the last weights; the in-place model is
    # restored to best-val (same convention as SNES.fit).
    from TNEP import TNEP
    final_model = TNEP(cfg)
    _sync_U_pair()
    for k, dst in _trainable(final_model).items():
        dst.assign(named[k])
    if mix_flat is not None:
        _tf_var(final_model.U_pair).assign(_tf_var(model.U_pair))
    for v, snap in zip(variables, best_snapshot):
        v.assign(snap)
    _sync_U_pair()
    return history, final_model, model
