"""First-order Adam trainer for TNEP — analytical-gradient alternative to SNES.

Same fit() interface as `SNES`, so `TNEP.fit` can dispatch between the
two via `cfg.optimizer`. Adam computes the gradient of the loss with
respect to the trainable weights via tf.GradientTape and updates them
with `tf.keras.optimizers.Adam`. The model is differentiable through
`predict_batch`, so no manual gradient derivation is needed.

Compared to `SNES`:
  - Each step is one forward + one backward through one chunk (mini-
    batch SGD-like), rather than one full-batch evaluation of a
    `pop_size`-sized population of perturbed weights.
  - No mu / sigma — Adam mutates the model weights in place.
  - Chunks are cycled deterministically: step `g` uses chunk
    `g % num_chunks`. This is equivalent to standard NN training with
    a per-step batch == one of the `batch_chunk_size` slabs.

Reports the same history dict shape as SNES so the plotting pipeline
keeps working without changes. SNES-specific entries (sigma_min/max/
mean/median, best_rmse, worst_rmse) are left as empty lists; the plot
helpers gate on emptiness and skip those panels.
"""
from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from TNEP import TNEP


def _format_duration(seconds: float) -> str:
    """Match SNES._format_duration so the progress bar reads identical
    in both modes (HH:MM:SS, or Dd HH:MM:SS past 24 h)."""
    if not (seconds == seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _clone_model(src: "TNEP", cfg) -> "TNEP":
    """Deep-copy a TNEP model's trainable weights into a fresh
    instance. Used to materialise final_gen and best_val models at
    the end of training (mirrors what SNES.fit returns)."""
    from TNEP import TNEP
    dst = TNEP(cfg)
    dst.W0.assign(src.W0)
    dst.b0.assign(src.b0)
    dst.W1.assign(src.W1)
    dst.b1.assign(src.b1)
    if cfg.target_mode == 2:
        dst.W0_pol.assign(src.W0_pol)
        dst.b0_pol.assign(src.b0_pol)
        dst.W1_pol.assign(src.W1_pol)
        dst.b1_pol.assign(src.b1_pol)
    return dst


class AdamTrainer:
    """First-order Adam trainer with the same interface as SNES."""

    def __init__(self, model: "TNEP") -> None:
        self.model = model
        self.cfg = model.cfg
        # Use the modern Keras 3 API; falls back to the legacy module
        # for older TF installs.
        try:
            self._opt = tf.keras.optimizers.Adam(
                learning_rate=float(self.cfg.adam_learning_rate))
        except AttributeError:
            self._opt = tf.optimizers.Adam(
                learning_rate=float(self.cfg.adam_learning_rate))

        # ANN trainables and (optional) U_pair are tracked separately
        # so the regulariser dispatch can apply different priors to
        # each. The full `_trainables` list is what tape.gradient and
        # the Adam apply step use; `_ann_trainables` is what L1/L2
        # iterates over when descriptor_mixing_regularizer != "shrinkage"
        # (so V_pair isn't pulled toward zero unless the user opts in).
        self._ann_trainables = [model.W0, model.b0, model.W1, model.b1]
        if self.cfg.target_mode == 2:
            self._ann_trainables.extend(
                [model.W0_pol, model.b0_pol, model.W1_pol, model.b1_pol])
        self._trainables = list(self._ann_trainables)
        self._has_Upair = getattr(model, "descriptor_mixing", False)
        if self._has_Upair:
            # When descriptor mixing is enabled, U_pair joins the
            # trainables list so tape.gradient produces a U_pair grad
            # alongside the ANN weight gradients. _forward_loss applies
            # _W0_eff(W0, U_pair) so U_pair flows into the forward pass.
            self._trainables.append(model.U_pair)

        # Surface a couple of fields that MasterTNEP / external tooling
        # reads off the optimizer for run-summary printouts. SNES.dim
        # is the total parameter count; mirror that here. pop_size is
        # meaningless for Adam (no population), report 1.
        self.dim = int(sum(int(np.prod(w.shape)) for w in self._trainables))
        self.pop_size = 1

        # Polarizability shear weighting — mirrors SNES._pol_weights.
        if self.cfg.target_mode == 2:
            shear_sq = self.cfg.lambda_shear ** 2
            self._pol_weights = tf.constant(
                [1.0, 1.0, 1.0, shear_sq, shear_sq, shear_sq], dtype=tf.float32)
        else:
            self._pol_weights = None
        self._use_mae = (self.cfg.loss_type == "mae")

        # L1/L2 strengths — resolved the same way as SNES so reg has the same scale.
        # Sentinels mirror SNES: None → auto, -1 → dynamic (adapted in fit loop).
        # Stored as tf.Variable so per-step graph captures don't restage on update.
        T = self.cfg.num_types
        n_per_type = self.cfg.dim_q * self.cfg.num_neurons + 2 * self.cfg.num_neurons
        self._n_per_type = float(n_per_type)
        auto_lambda = float(np.sqrt(
            (T * n_per_type + 1) * 1e-6 / max(T, 1)))
        self._dyn_lambda_1 = (self.cfg.lambda_1 == -1)
        self._dyn_lambda_2 = (self.cfg.lambda_2 == -1)
        init_lambda_1 = (auto_lambda if (self.cfg.lambda_1 is None or self._dyn_lambda_1)
                         else float(self.cfg.lambda_1))
        init_lambda_2 = (auto_lambda if (self.cfg.lambda_2 is None or self._dyn_lambda_2)
                         else float(self.cfg.lambda_2))
        self.lambda_1 = tf.Variable(init_lambda_1, dtype=tf.float32,
                                    trainable=False, name="lambda_1")
        self.lambda_2 = tf.Variable(init_lambda_2, dtype=tf.float32,
                                    trainable=False, name="lambda_2")

        # V_pair regulariser mode (mirrors SNES):
        #   "off"        : V_pair excluded from L1/L2.
        #   "shrinkage"  : V_pair included in L1/L2 (legacy default).
        #   "orthogonal" : V_pair penalised by ‖UᵀU - I‖² per block,
        #                  scaled by lambda_orth (its own dynamic option).
        self._mix_reg_mode = str(getattr(
            self.cfg, "descriptor_mixing_regularizer", "off")).lower()
        if self._mix_reg_mode not in ("off", "shrinkage", "orthogonal"):
            raise ValueError(
                f"descriptor_mixing_regularizer={self._mix_reg_mode!r} not "
                "recognised (expected 'off', 'shrinkage', or 'orthogonal')")
        if self._has_Upair:
            n_U_pair = int(np.prod(model.U_pair.shape))
            auto_lambda_orth = float(np.sqrt(
                n_U_pair * 1e-6 / max(T, 1)))
        else:
            auto_lambda_orth = 0.0
        cfg_lo = getattr(self.cfg, "lambda_orth", None)
        self._dyn_lambda_orth = (cfg_lo == -1)
        init_lambda_orth = (auto_lambda_orth if (cfg_lo is None or self._dyn_lambda_orth)
                            else float(cfg_lo))
        self._lambda_orth = tf.Variable(init_lambda_orth, dtype=tf.float32,
                                        trainable=False, name="lambda_orth")

    # ----- forward + loss --------------------------------------------------

    def _forward_loss(self, chunk: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Run predict_batch on `chunk` and return (loss, train_rmse).

        Both quantities are scalar `tf.Tensor`. `loss` is what we
        differentiate (includes regularization); `train_rmse` is the
        unregularised metric reported in history.
        """
        m = self.model
        # Apply U_pair^T to W0 (and W0_pol) so the gradient w.r.t.
        # U_pair flows through tape.gradient — predict_batch itself
        # is a pure forward over the supplied weights.
        W0 = m._W0_eff(m.W0)
        if self.cfg.target_mode == 2:
            W0p = m._W0_eff(m.W0_pol)
            b0p = m.b0_pol
            W1p, b1p = m.W1_pol, m.b1_pol
        else:
            W0p = b0p = W1p = b1p = None
        preds = m.predict_batch(
            chunk["descriptors"], chunk["grad_values"],
            chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
            chunk["positions"], chunk["Z_int"], chunk["boxes"],
            chunk["atom_mask"],
            W0, m.b0, m.W1, m.b1,
            W0p, b0p, W1p, b1p,
        )
        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            num_atoms = tf.reduce_sum(chunk["atom_mask"], axis=1)
            preds = preds / tf.maximum(num_atoms, 1.0)[:, tf.newaxis]
        diff = preds - chunk["targets"]
        if self._pol_weights is not None:
            sq = tf.square(diff) * self._pol_weights
        else:
            sq = tf.square(diff)
        if self._use_mae:
            sq = tf.abs(diff)
        train_rmse = tf.sqrt(tf.maximum(tf.reduce_mean(sq), 0.0))

        loss = tf.reduce_mean(sq)
        if self.cfg.toggle_regularization:
            # L1/L2 always cover the ANN trainables. U_pair (V) is
            # included only when mode == "shrinkage". For mode ==
            # "orthogonal", we add the Frobenius orthogonality penalty
            # ‖UᵀU - I‖² instead (computed in residual form). For
            # mode == "off", V_pair is unregularised.
            if self._mix_reg_mode == "shrinkage" and self._has_Upair:
                reg_weights = self._trainables
            else:
                reg_weights = self._ann_trainables
            l1 = sum(tf.reduce_sum(tf.abs(w)) for w in reg_weights)
            l2 = tf.sqrt(tf.reduce_sum(
                tf.stack([tf.reduce_sum(tf.square(w)) for w in reg_weights])))
            loss = loss + self.lambda_1 * l1 + self.lambda_2 * l2
            if self._mix_reg_mode == "orthogonal" and self._has_Upair:
                loss = loss + self._lambda_orth * self._orth_penalty(m.U_pair)
        return loss, train_rmse

    def _orth_penalty(self, U_pair: tf.Tensor) -> tf.Tensor:
        """Frobenius orthogonality penalty in residual form.

        U_pair stores V = U - I, so UᵀU - I = V + Vᵀ + VᵀV. We
        compute that quantity per active [bs, bs] sub-block (the
        padded rows/cols stay zero by construction and never see the
        descriptor) and sum the squared Frobenius norms. Normalised
        by total active entries so the value is on a comparable scale
        to the L2 contribution.

        U_pair shape:
            shared    : [num_pairs, max_bs, max_bs]
            per-type  : [T, num_pairs, max_bs, max_bs]
        """
        m = self.model
        block_sizes = m._mix_block_sizes
        n_total = sum(bs * bs for bs in block_sizes)
        if m.descriptor_mixing_per_type:
            n_total *= m.cfg.num_types
            U_view = U_pair  # [T, P, bs_max, bs_max]
        else:
            U_view = U_pair[tf.newaxis, ...]  # [1, P, bs_max, bs_max]
        pen = tf.constant(0.0)
        for t in range(int(U_view.shape[0])):
            for p_idx, bs in enumerate(block_sizes):
                V = U_view[t, p_idx, :bs, :bs]               # [bs, bs]
                VtV = tf.matmul(V, V, transpose_a=True)
                M = V + tf.linalg.matrix_transpose(V) + VtV
                pen = pen + tf.reduce_sum(tf.square(M))
        return pen / float(n_total)

    def _train_step(self, chunk: dict) -> tuple[tf.Tensor, tf.Tensor]:
        """Forward + backward + Adam update.

        Not decorated with `@tf.function` — `predict_batch` is already
        a tf.function, and wrapping a second tf.function around it
        triggers Keras 3's trace-type canonicalisation to materialise
        the Variables via `__array__`, which fails inside a graph
        context. The inner predict_batch + tape.gradient compose
        cleanly in eager mode and still produce a compiled forward
        pass — the only thing we lose is the (small) Python-side
        dispatch for the Adam apply step.
        """
        with tf.GradientTape() as tape:
            loss, train_rmse = self._forward_loss(chunk)
        grads = tape.gradient(loss, self._trainables)
        self._opt.apply_gradients(zip(grads, self._trainables))
        return loss, train_rmse

    # ----- dynamic-λ -------------------------------------------------------

    def _maybe_adapt_lambda(self, gen: int, data_loss: float,
                            l1: float, l2: float,
                            l_orth: float = 0.0) -> None:
        """Rescale λ_1 / λ_2 / λ_orth toward `target_ratio · data_loss`.

        Mirrors the SNES adaptation rule so Adam and SNES runs with
        the same cfg follow the same λ schedule. Only the lambdas set
        to -1 in cfg are touched. See SNES._maybe_adapt_lambda for the
        derivation of the multiplicative damped step.
        """
        if not (self._dyn_lambda_1 or self._dyn_lambda_2 or self._dyn_lambda_orth):
            return
        interval = max(1, int(getattr(self.cfg, "lambda_adapt_interval", 100)))
        if gen % interval != 0:
            return
        target = float(getattr(self.cfg, "lambda_target_ratio", 0.05))
        damping = float(getattr(self.cfg, "lambda_damping", 0.2))
        lmin = float(getattr(self.cfg, "lambda_min", 1e-8))
        lmax = float(getattr(self.cfg, "lambda_max", 1.0))
        ref = max(float(data_loss), 1e-8)
        if self._dyn_lambda_1 and l1 > 1e-12:
            ratio = (target * ref) / float(l1)
            new = float(self.lambda_1.numpy()) * (ratio ** damping)
            self.lambda_1.assign(float(np.clip(new, lmin, lmax)))
        if self._dyn_lambda_2 and l2 > 1e-12:
            ratio = (target * ref) / float(l2)
            new = float(self.lambda_2.numpy()) * (ratio ** damping)
            self.lambda_2.assign(float(np.clip(new, lmin, lmax)))
        if self._dyn_lambda_orth and l_orth > 1e-12:
            ratio = (target * ref) / float(l_orth)
            new = float(self._lambda_orth.numpy()) * (ratio ** damping)
            self._lambda_orth.assign(float(np.clip(new, lmin, lmax)))

    # ----- validate --------------------------------------------------------

    def _validate(self, val_data: dict) -> float:
        """Mean RMSE on val_data (or random subset when cfg.val_size set).

        Mirrors `SNES.validate` for label parity. Reuses the streaming
        `prefetched_chunks` loop to keep memory bounded.
        """
        from data import prefetched_chunks
        S_val = int(val_data["num_atoms"].shape[0])
        N_idx = S_val if self.cfg.val_size is None else min(self.cfg.val_size, S_val)
        struct_chunk = (self.cfg.batch_chunk_size
                        if self.cfg.batch_chunk_size is not None else N_idx)
        diff_sq_sum = tf.constant(0.0, dtype=tf.float32)
        diff_count = tf.constant(0.0, dtype=tf.float32)
        m = self.model
        if self.cfg.target_mode == 2:
            W0p, b0p, W1p, b1p = m.W0_pol, m.b0_pol, m.W1_pol, m.b1_pol
        else:
            W0p = b0p = W1p = b1p = None

        ranges = [(s, min(s + struct_chunk, N_idx))
                  for s in range(0, N_idx, struct_chunk)]
        for _, _, chunk in prefetched_chunks(
                val_data, ranges,
                pin_to_cpu=self.cfg.pin_data_to_cpu,
                enabled=getattr(self.cfg, "chunk_prefetch", True),
                depth=getattr(self.cfg, "prefetch_depth", 1)):
            preds = m.predict_batch(
                chunk["descriptors"], chunk["grad_values"],
                chunk["pair_atom"], chunk["pair_gidx"], chunk["pair_struct"],
                chunk["positions"], chunk["Z_int"], chunk["boxes"],
                chunk["atom_mask"],
                m.W0, m.b0, m.W1, m.b1,
                W0p, b0p, W1p, b1p,
            )
            if self.cfg.scale_targets and self.cfg.target_mode == 1:
                na = tf.reduce_sum(chunk["atom_mask"], axis=1)
                preds = preds / tf.maximum(na, 1.0)[:, tf.newaxis]
            d = preds - chunk["targets"]
            if self._pol_weights is not None:
                ds = tf.square(d) * self._pol_weights
            else:
                ds = tf.square(d)
            diff_sq_sum += tf.reduce_sum(ds)
            diff_count += tf.cast(tf.size(ds), tf.float32)
            del chunk

        rmse = tf.sqrt(tf.maximum(diff_sq_sum / tf.maximum(diff_count, 1.0), 0.0))
        return float(rmse)

    # ----- main loop -------------------------------------------------------

    def fit(self,
            train_data: dict,
            val_data: dict,
            plot_callback: Callable | None = None,
            resume_state: dict | None = None) -> tuple[dict, "TNEP", "TNEP"]:
        """Adam training loop. Same signature/returns as SNES.fit.

        `resume_state` is accepted for interface compatibility but
        ignored in this prototype — the Adam checkpoint format is a
        follow-up. A warning is printed if resume is attempted.
        """
        if resume_state is not None:
            print("  Adam: resume_state given but Adam checkpoint format "
                  "not yet implemented — starting from scratch")

        print("Fitting model (Adam)...")
        cfg = self.cfg
        S_train = int(train_data["num_atoms"].shape[0])
        chunk_size = (cfg.batch_chunk_size
                      if cfg.batch_chunk_size is not None else S_train)
        train_ranges = [(s, min(s + chunk_size, S_train))
                        for s in range(0, S_train, chunk_size)]
        n_chunks = max(1, len(train_ranges))

        from data import prefetched_chunks

        history = {
            "generation": [],
            "train_loss": [],
            "val_loss": [],
            "L1": [],
            "L2": [],
            # SNES-specific, kept for plot-pipeline compatibility:
            "best_rmse": [],
            "worst_rmse": [],
            "sigma_min": [],
            "sigma_max": [],
            "sigma_mean": [],
            "sigma_median": [],
            "timing": {
                "sample_batch": [],
                "evaluate": [],
                "rank_update": [],
                "validate": [],
                "overhead": [],
            },
        }

        # Best-val snapshot (we copy weight tensors, not references).
        best_val_loss = float("inf")
        best_weights = [tf.identity(w) for w in self._trainables]
        gens_without_improvement = 0
        val_fitness = float("inf")
        gen_l1, gen_l2, gen_lorth = 0.0, 0.0, 0.0

        # Cache the chunk objects so we don't re-stage every step (each
        # chunk is identical across steps when the data is GPU-resident,
        # so prefetched_chunks already returns instantly). Materialise
        # once up front, then index per step — saves the
        # prefetched_chunks generator dispatch on each step.
        train_chunks: list[dict] = []
        for _, _, chunk in prefetched_chunks(
                train_data, train_ranges,
                pin_to_cpu=cfg.pin_data_to_cpu,
                enabled=False,  # single-pass build; no need to overlap.
                depth=1):
            train_chunks.append(chunk)

        train_start = time.perf_counter()

        for gen in range(cfg.num_generations):
            t0 = time.perf_counter()

            chunk = train_chunks[gen % n_chunks]

            t1 = time.perf_counter()

            loss, train_rmse = self._train_step(chunk)

            t2 = time.perf_counter()

            # No "rank+update" stage in Adam — t3 marker stays at t2
            # so the timing slot for `rank_update` is ~0 (kept for
            # plot consistency).
            t3 = t2

            _do_val = (gen % cfg.val_interval == 0) or (gen == cfg.num_generations - 1)
            if _do_val:
                val_fitness = self._validate(val_data)

            t4 = time.perf_counter()

            # Sample reg metrics every 100 steps to mirror SNES cadence
            # (cheap here but stay consistent for plot scale). The
            # weights used for the L1/L2 sum match the dispatch in
            # _forward_loss: shrinkage includes U_pair, otherwise just
            # the ANN trainables.
            if cfg.toggle_regularization and gen % 100 == 0:
                reg_weights = (self._trainables
                               if (self._mix_reg_mode == "shrinkage"
                                   and self._has_Upair)
                               else self._ann_trainables)
                l1 = float(sum(tf.reduce_sum(tf.abs(w)) for w in reg_weights))
                l2 = float(tf.sqrt(tf.reduce_sum(tf.stack(
                    [tf.reduce_sum(tf.square(w)) for w in reg_weights]))))
                gen_l1 = float(self.lambda_1.numpy()) * l1 / self._n_per_type
                gen_l2 = float(self.lambda_2.numpy()) * l2 / np.sqrt(self._n_per_type)
                if self._mix_reg_mode == "orthogonal" and self._has_Upair:
                    gen_lorth = float(self._lambda_orth.numpy()) * float(
                        self._orth_penalty(self.model.U_pair))
                else:
                    gen_lorth = 0.0
                # Dynamic-λ adaptation (any λ set to -1 in cfg).
                self._maybe_adapt_lambda(gen, float(train_rmse),
                                         gen_l1, gen_l2, gen_lorth)
            elif not cfg.toggle_regularization:
                gen_l1 = gen_l2 = gen_lorth = 0.0

            if _do_val:
                history["generation"].append(gen)
                history["train_loss"].append(float(train_rmse))
                history["val_loss"].append(val_fitness)
                history["L1"].append(gen_l1)
                history["L2"].append(gen_l2)
                history.setdefault("L_orth", []).append(gen_lorth)
                # Adam has no population spread; report current loss
                # in best/worst slots so the plot doesn't show NaNs.
                history["best_rmse"].append(float(train_rmse))
                history["worst_rmse"].append(float(train_rmse))

            # Progress bar
            frac = (gen + 1) / cfg.num_generations
            bar_len = 30
            filled = int(bar_len * frac)
            bar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.perf_counter() - train_start
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            line = (f"\r{bar} {gen + 1}/{cfg.num_generations} "
                    f"train RMSE: {float(train_rmse):.6f}  "
                    f"val RMSE: {val_fitness:.6f}  "
                    f"best val RMSE: {best_val_loss:.6f}  "
                    f"elapsed: {_format_duration(elapsed)}  "
                    f"ETA: {_format_duration(eta)}")
            if cfg.debug:
                line += f"  L1: {gen_l1:.6f}  L2: {gen_l2:.6f}"
                if self._mix_reg_mode == "orthogonal":
                    line += f"  L_orth: {gen_lorth:.6f}"
            sys.stdout.write(line)
            sys.stdout.flush()

            # Early stopping: update best snapshot on val improvement.
            if _do_val:
                if val_fitness < best_val_loss:
                    best_val_loss = val_fitness
                    best_weights = [tf.identity(w) for w in self._trainables]
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1

            if cfg.patience is not None and gens_without_improvement >= cfg.patience:
                print(f"\nEarly stopping at generation {gen + 1} "
                      f"(no improvement for {cfg.patience} generations)")
                break

            t5 = time.perf_counter()

            if _do_val:
                history["timing"]["sample_batch"].append(t1 - t0)
                history["timing"]["evaluate"].append(t2 - t1)
                history["timing"]["rank_update"].append(t3 - t2)
                history["timing"]["validate"].append(t4 - t3)
                history["timing"]["overhead"].append(t5 - t4)

            # Periodic plot callback (same hook as SNES).
            if (gen + 1 < cfg.num_generations
                    and plot_callback is not None
                    and cfg.plot_interval is not None
                    and (gen + 1) % cfg.plot_interval == 0):
                # Swap in best weights for plotting, then restore.
                snapshot = [tf.identity(w) for w in self._trainables]
                for w, b in zip(self._trainables, best_weights):
                    w.assign(b)
                plot_callback(history, gen + 1)
                for w, s in zip(self._trainables, snapshot):
                    w.assign(s)

        print()  # newline after progress bar

        # Final-gen model = current weights at exit.
        final_model = _clone_model(self.model, cfg)
        # Best-val model = the snapshot taken at lowest val seen.
        for w, b in zip(self._trainables, best_weights):
            w.assign(b)
        best_val_model = _clone_model(self.model, cfg)

        return history, final_model, best_val_model
