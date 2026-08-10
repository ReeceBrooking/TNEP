from __future__ import annotations
import contextlib
import sys
import time
import numpy as np
import tensorflow as tf
from typing import TYPE_CHECKING
from SNES import _set_model_params, _format_duration, sample_minibatch
if TYPE_CHECKING:
    from TNEP import TNEP


def _build_mixing_scatter(model):
    """Precompute block layout + upper-tri scatter for A→skew.

    Returns (n_A, scatter_indices[int32 (n_A, rank)], block_shape[list],
             max_alpha[int]).
      block_shape = list(model.U_pair.shape) == [(T,) num_pairs, L, max_a, max_a].
    A holds one entry per active upper-tri position (i<j<alpha_p) of every block;
    scatter places them into U_pair's axis order. NOTE: A is an independent,
    zero-initialised Variable (not tied to any SNES genome), so the *ordering* of
    A's entries is FREE — any consistent bijection works, because result models
    rebuild V from A through this same scatter. Only two things matter:
    (1) fill exactly i<j<alpha_p per block; (2) use U_pair's axis order.
    """
    per_type = bool(getattr(model, "descriptor_mixing_per_type", False))
    T = int(model.cfg.num_types)
    P = int(model._mix_num_pairs)
    L = int(model._mix_L)
    alpha = list(model._mix_alpha_per_pair)        # per-pair active block size
    max_a = int(model._mix_max_alpha)
    idx = []
    type_range = range(T) if per_type else [None]
    for t in type_range:
        for p in range(P):
            for l in range(L):
                for i in range(alpha[p]):
                    for j in range(i + 1, alpha[p]):
                        idx.append(([t, p, l, i, j] if per_type else [p, l, i, j]))
    n_A = len(idx)
    block_shape = list(model.U_pair.shape)
    scatter = tf.constant(np.asarray(idx, dtype=np.int32)) if n_A else \
        tf.zeros([0, len(block_shape)], tf.int32)
    return n_A, scatter, block_shape, max_a


def _sentinel0(x) -> float:
    """Sentinel-aware coerce: None or -1 → 0.0, else float(x)."""
    return 0.0 if x is None or x == -1 else float(x)


def _mask_wpre_grad(g, idx):
    """Zero every entry of the flattened W_pre_angular gradient except those at
    `idx` (the summed-flat indices), reshaped back to g's shape."""
    if g is None:
        return g
    shape = tf.shape(g)
    flat = tf.reshape(g, [-1])
    kept = tf.gather(flat, idx)
    masked_flat = tf.scatter_nd(idx[:, tf.newaxis], kept, tf.shape(flat))
    return tf.reshape(masked_flat, shape)


class Adam:
    def __init__(self, model: "TNEP") -> None:
        self.model = model
        self.cfg = model.cfg
        cfg = self.cfg
        self._mixing = bool(getattr(model, "descriptor_mixing", False))
        self._preprocess = str(
            getattr(cfg, "descriptor_preprocess_contract", "off")).lower() != "off"
        # --- mixing generator A (active upper-tri entries), init 0 ---
        if self._mixing:
            self._n_A, self._scatter, self._block_shape, self._max_alpha = \
                _build_mixing_scatter(model)
            self.A = tf.Variable(tf.zeros([self._n_A], tf.float32),
                                 trainable=True, name="adam_mix_A")
        else:
            self._n_A = 0
            self.A = None

        # --- layout counts (read from Variable shapes) ---
        self._n_W0 = int(tf.size(model.W0))
        self._n_b0 = int(tf.size(model.b0))
        self._n_W1 = int(tf.size(model.W1))
        self._n_b1 = int(tf.size(model.b1))
        self._n_Wh = int(tf.size(model.Wh)) if model.num_hidden_layers == 2 else 0
        self._n_bh = int(tf.size(model.bh)) if model.num_hidden_layers == 2 else 0
        self.n_U_pair = self._n_A if self._mixing else 0
        self.n_preprocess = (int(model._preprocess_summed_flat_idx.shape[0])
                             if self._preprocess else 0)

        # --- trainable Variable lists ---
        # _ann_vars excludes A / W_pre (W_pre handled specially — watched + masked).
        # Order W0,b0,[Wh,bh],W1,b1(+pol): Wh/bh (2-layer only) sit between the
        # first-layer and output-layer params so the tape/reg/snapshot see them,
        # but they are NEVER part of the core tail handed to _set_model_params
        # (see _snapshot/_params_from_snapshot below — that's the SNES-shared
        # contract and must stay W0,b0,W1,b1(+pol)(+V)(+W_pre)).
        self._ann_vars = [model.W0, model.b0]
        if model.num_hidden_layers == 2:
            self._ann_vars += [model.Wh, model.bh]
        self._ann_vars += [model.W1, model.b1]
        if cfg.target_mode == 2:
            self._ann_vars += [model.W0_pol, model.b0_pol,
                               model.W1_pol, model.b1_pol]
        self._vars = list(self._ann_vars)
        if self._mixing:
            self._vars.append(self.A)
        # Vars the tape watches each step (W_pre is trainable=False → watched
        # explicitly). Fixed list so the compiled loss/grad fn traces once.
        self._watched = list(self._vars) + (
            [self.model.W_pre_angular] if self._preprocess else [])

        # total trainable params (Variable sizes + A entries + summed W_pre)
        self.dim = (sum(int(tf.size(v)) for v in self._ann_vars)
                    + self._n_A + self.n_preprocess)
        self.pop_size = None

        # amsgrad: the converged NequIP/MACE choice (max-of-second-moment
        # variant, monotone effective step) — no config knob on purpose.
        self._keras = tf.keras.optimizers.Adam(
            cfg.adam_learning_rate, cfg.adam_beta_1,
            cfg.adam_beta_2, cfg.adam_epsilon, amsgrad=True)

        # --- weight EMA (Polyak averaging), opt-in via cfg.adam_use_ema ---
        # Shadow copies of every watched var, initialised to the starting weights.
        # Updated eagerly each step; swapped in for eval / result models so the
        # returned model uses the averaged (settled) weights, not the raw iterate.
        self._ema_on = bool(getattr(cfg, "adam_use_ema", False))
        self._ema_momentum = float(getattr(cfg, "adam_ema_momentum", 0.999))
        if self._ema_on:
            self._ema = [tf.Variable(tf.identity(v), trainable=False,
                                     name=f"adam_ema_{i}")
                         for i, v in enumerate(self._watched)]
        else:
            self._ema = None

    def _reconstruct_V(self, A=None) -> tf.Tensor:
        """A → V = expm(skew) − I, padded to model.U_pair shape (0 if no mixing).

        `A` defaults to the live `self.A` Variable; pass a snapshot tensor to
        rebuild V from stored best-val values.
        """
        if not self._mixing:
            return None
        if A is None:
            A = self.A
        # scatter A into padded upper triangle U, skew = U - Uᵀ, expm, minus eye.
        upper = tf.scatter_nd(self._scatter, A, self._block_shape)  # padded
        skew = upper - tf.linalg.matrix_transpose(upper)
        U = tf.linalg.expm(skew)                              # batched over blocks
        V = U - tf.eye(self._max_alpha, batch_shape=self._block_shape[:-2])
        return V

    def _forward(self, batch):
        """Differentiable per-atom-normalized prediction for a staged batch dict."""
        m = self.model
        V = self._reconstruct_V()
        W0_eff = m._W0_eff(m.W0, U_pair=V)
        if self._preprocess:
            W0_eff = m._W0_preprocess_eff(W0_eff)
        # predict_batch is itself a @tf.function; when _forward is compiled, its
        # arg-canonicalisation calls __array__ on raw Keras Variables and fails.
        # Pass tensor reads instead (gradient still flows to the Variables).
        # convert_to_tensor on every weight: _W0_eff returns the raw W0 Variable
        # when mixing is OFF, and b0/W1/b1/pol are always Variables — the nested
        # predict_batch @tf.function can't canonicalize Variable args in graph.
        W0_eff = tf.convert_to_tensor(W0_eff)
        b0 = tf.convert_to_tensor(m.b0)
        W1 = tf.convert_to_tensor(m.W1)
        b1 = tf.convert_to_tensor(m.b1)
        Wh = bh = None
        if m.num_hidden_layers == 2:
            Wh = tf.convert_to_tensor(m.Wh)
            bh = tf.convert_to_tensor(m.bh)
        pol = (None, None, None, None)
        if self.cfg.target_mode == 2:
            W0p = m._W0_eff(m.W0_pol, U_pair=V)
            if self._preprocess:
                W0p = m._W0_preprocess_eff(W0p)
            pol = (tf.convert_to_tensor(W0p), tf.convert_to_tensor(m.b0_pol),
                   tf.convert_to_tensor(m.W1_pol), tf.convert_to_tensor(m.b1_pol))
        # Arg order copied from score's predict_batch call (TNEP.py:805-815).
        # NOTE type index is batch["Z_int"] (not "Z"). W_atom precomputed in fit.
        raw = m.predict_batch(
            batch["descriptors"], batch["grad_values"], batch["pair_atom"],
            batch["pair_gidx"], batch["pair_struct"], batch["positions"],
            batch["Z_int"], batch["boxes"], batch["atom_mask"],
            W0_eff, b0, W1, b1, *pol, Wh=Wh, bh=bh, W_atom=batch.get("_W_atom"))
        # Per-atom normalization — MUST match SNES.py:1310-1312 / score TNEP.py:820-825,
        # since stored `targets` are per-atom-scaled when scale_targets & mode 1.
        if self.cfg.scale_targets and self.cfg.target_mode == 1:
            num_atoms = tf.reduce_sum(batch["atom_mask"], axis=1)
            raw = raw / tf.maximum(num_atoms, 1.0)[:, tf.newaxis]
        return raw

    def _type_params(self, t):
        """Flattened per-type ANN params (W0,b0,[Wh,bh,]W1[+pol]), excluding b1 —
        the grouping SNES regularizes (SNES.py compute_regularization_tf)."""
        m = self.model
        parts = [tf.reshape(m.W0[t], [-1]), m.b0[t]]
        if m.num_hidden_layers == 2:
            parts += [tf.reshape(m.Wh[t], [-1]), m.bh[t]]
        parts.append(m.W1[t])
        if self.cfg.target_mode == 2:
            parts += [tf.reshape(m.W0_pol[t], [-1]), m.b0_pol[t], m.W1_pol[t]]
        return tf.concat(parts, 0)

    def _regularization(self):
        """L1+L2 penalty matching SNES's NORMALIZED scale (per-type mean +
        global, L2 = sqrt(mean(w²))). Raw Σ|w|/Σw² would be ~1000x too strong
        vs the data loss and collapse the weights. b1 and the mixing generator
        A are excluded (SNES excludes them too)."""
        l1 = _sentinel0(self.cfg.lambda_1)
        l2 = _sentinel0(self.cfg.lambda_2)
        if not l1 and not l2:
            return tf.constant(0.0, tf.float32)
        T = int(self.cfg.num_types)
        if T > 1:
            total_l1 = tf.constant(0.0); total_l2 = tf.constant(0.0)
            for t in range(T):
                tp = self._type_params(t)
                n = tf.cast(tf.size(tp), tf.float32)
                total_l1 += l1 * tf.reduce_sum(tf.abs(tp)) / n
                total_l2 += l2 * tf.sqrt(tf.reduce_sum(tf.square(tp)) / n)
            typed = tf.concat([self._type_params(t) for t in range(T)], 0)
            nt = tf.cast(tf.size(typed), tf.float32)
            global_l1 = l1 * tf.reduce_sum(tf.abs(typed)) / nt
            global_l2 = l2 * tf.sqrt(tf.reduce_sum(tf.square(typed)) / nt)
            return total_l1 / T + global_l1 + total_l2 / T + global_l2
        ann = tf.concat([tf.reshape(w, [-1]) for w in self._ann_vars], 0)
        n = tf.cast(tf.size(ann), tf.float32)
        return (l1 * tf.reduce_sum(tf.abs(ann)) / n
                + l2 * tf.sqrt(tf.reduce_sum(tf.square(ann)) / n))

    def _batch_loss(self, batch):
        loss = tf.reduce_mean(tf.square(self._forward(batch) - batch["targets"]))
        return loss + self._regularization()

    @tf.function(reduce_retracing=True)
    def _loss_and_grads(self, batch):
        # Compiled once (traces on the first call) so the expensive part — the
        # tape, expm reconstruction, forward and backward — runs in-graph
        # instead of eagerly (~100x faster after the first step). The Keras
        # Adam apply stays eager (it hits an eager-only path in graph mode).
        with tf.GradientTape() as tape:
            if self._preprocess:
                tape.watch(self.model.W_pre_angular)     # trainable=False → watch
            loss = self._batch_loss(batch)
        # Replace None grads (vars with no gradient path — e.g. the b1 output
        # bias in dipole mode) with zeros: no warning, clean tf.function output,
        # and Adam leaves them untouched. The `is None` test is trace-time.
        raw_grads = tape.gradient(loss, self._watched)
        grads = [g if g is not None else tf.zeros_like(v)
                 for g, v in zip(raw_grads, self._watched)]
        if self._preprocess:                              # mask to summed entries
            grads[-1] = _mask_wpre_grad(
                grads[-1], self.model._preprocess_summed_flat_idx)
        return loss, grads

    def _train_step(self, batch):
        loss, grads = self._loss_and_grads(batch)         # graph
        self._keras.apply_gradients(zip(grads, self._watched))  # eager
        if self._ema_on:
            m = self._ema_momentum
            for e, v in zip(self._ema, self._watched):    # e ← m·e + (1−m)·v
                e.assign(m * e + (1.0 - m) * v)
        return loss

    @contextlib.contextmanager
    def _eval_weights(self):
        """Temporarily load EMA weights into the live vars for evaluation /
        snapshotting, restoring the raw iterate on exit. No-op when EMA is off,
        so the non-EMA path is byte-identical to before."""
        if not self._ema_on:
            yield
            return
        backup = [tf.identity(v) for v in self._watched]
        for v, e in zip(self._watched, self._ema):
            v.assign(e)
        try:
            yield
        finally:
            for v, b in zip(self._watched, backup):
                v.assign(b)

    # ------------------------------------------------------------------ #
    #  Snapshot / result-model helpers                                    #
    # ------------------------------------------------------------------ #
    def _snapshot(self) -> dict:
        """Copy the current values (not references) of the trainable state so a
        best-val snapshot does not track live weights.

        `core` is EXACTLY the tail `_set_model_params` expects — W0,b0,W1,b1
        (+pol) — built directly from the model, NOT from `_ann_vars` (which now
        also holds Wh/bh for 2-layer models and must never reach
        `_set_model_params`, a contract shared with SNES). Wh/bh get their own
        keys and are restored onto the model separately.
        """
        m = self.model
        core = [tf.identity(m.W0), tf.identity(m.b0),
                tf.identity(m.W1), tf.identity(m.b1)]
        if self.cfg.target_mode == 2:
            core += [tf.identity(m.W0_pol), tf.identity(m.b0_pol),
                     tf.identity(m.W1_pol), tf.identity(m.b1_pol)]
        snap = {
            "core": core,
            "Wh": tf.identity(m.Wh) if m.num_hidden_layers == 2 else None,
            "bh": tf.identity(m.bh) if m.num_hidden_layers == 2 else None,
            "A": tf.identity(self.A) if self._mixing else None,
            "W_pre": (tf.identity(self.model.W_pre_angular)
                      if self._preprocess else None),
        }
        return snap

    def _params_from_snapshot(self, snap: dict) -> list:
        """Build the conditional positional params list for `_set_model_params`.

        Tail order must match SNES._set_model_params: W0,b0,W1,b1 | +pol (mode 2)
        | +U_pair=V (mixing) | +W_pre_angular (preprocess). Uses snap["core"]
        (never snap["ann"], which may also contain Wh/bh) so Wh/bh can never
        leak into `_set_model_params`.
        """
        params = list(snap["core"])                       # W0,b0,W1,b1 (+pol)
        if self._mixing:
            params.append(self._reconstruct_V(snap["A"]))   # V, not U
        if self._preprocess:
            params.append(snap["W_pre"])
        return params

    def _restore_hidden_layer(self, model, snap: dict) -> None:
        """Assign Wh/bh from a snapshot onto `model` (2-layer only; no-op else).

        Kept separate from `_set_model_params` because Wh/bh are NOT part of
        its positional contract (shared with SNES).
        """
        if snap.get("Wh") is not None:
            model.Wh.assign(snap["Wh"])
            model.bh.assign(snap["bh"])

    def _model_from_snapshot(self, snap: dict):
        """Fresh TNEP with snapshot weights applied."""
        from TNEP import TNEP                              # runtime import (cycle)
        new_model = TNEP(self.cfg)
        _set_model_params(new_model, *self._params_from_snapshot(snap))
        self._restore_hidden_layer(new_model, snap)
        return new_model

    # ------------------------------------------------------------------ #
    #  Training loop                                                      #
    # ------------------------------------------------------------------ #
    def _precompute_W_atom(self, batch) -> None:
        """Geometry-only, weights-independent W_atom for the full batch (mode 1)."""
        if self.cfg.target_mode != 1 or "_W_atom" in batch:
            return
        desc = batch["descriptors"]
        B = desc.shape[0] if desc.shape[0] is not None else tf.shape(desc)[0]
        A = desc.shape[1] if desc.shape[1] is not None else tf.shape(desc)[1]
        batch["_W_atom"] = self.model._precompute_dipole_kernel(
            batch["grad_values"], batch["pair_struct"], batch["pair_atom"],
            batch["pair_gidx"], batch["positions"], batch["boxes"], B, A)

    def fit(self, train_data, val_data, plot_callback=None, resume_state=None):
        """Adam training. Returns (history, final_model, best_val_model).

        Mirrors SNES.fit's return contract so it drops into TNEP.fit unchanged.
        cfg.batch_size = None trains full-batch; an int samples that many
        structures per step (same sampler as SNES.fit).
        """
        if resume_state is not None:
            raise NotImplementedError(
                "Adam optimizer does not support resume; "
                "use optimizer='snes' for checkpointing.")

        cfg = self.cfg
        # W_atom is geometry-only (weights-independent): precompute ONCE on the
        # full train set; sample_minibatch gathers the per-structure rows along
        # with the rest. score() recomputes its own W_atom internally, so val
        # doesn't strictly need it, but precompute for symmetry. Mode 2 uses
        # the COO path (no W_atom).
        self._precompute_W_atom(train_data)
        self._precompute_W_atom(val_data)

        S_train = int(train_data["targets"].shape[0])
        minibatch = cfg.batch_size is not None and int(cfg.batch_size) < S_train
        if minibatch:
            rng = (tf.random.Generator.from_seed(cfg.seed)
                   if cfg.seed is not None
                   else tf.random.Generator.from_non_deterministic_state())

        history = {"generation": [], "train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_snap = None
        gens_without_improvement = 0
        val_interval = max(1, int(getattr(cfg, "val_interval", 1)))
        n_gen = int(cfg.num_generations)
        train_start = time.perf_counter()
        last_train = float("nan")   # last recorded train/val RMSE (updated on val ticks)
        last_val = float("nan")

        for gen in range(n_gen):
            if minibatch:
                idx = tf.argsort(rng.uniform(shape=[S_train]))[:cfg.batch_size]
                batch = sample_minibatch(train_data, idx)
            else:
                batch = train_data
            loss = self._train_step(batch)

            if (gen + 1) % val_interval == 0 or gen == n_gen - 1:
                # Evaluate + snapshot under EMA weights (no-op if EMA off), so
                # val RMSE, the plotted train RMSE, and best/final models all use
                # the averaged iterate.
                with self._eval_weights():
                    # Sync trained mixing into the model so score() (reads
                    # self.U_pair) sees it — else val uses identity mixing.
                    if self._mixing:
                        self.model.U_pair.assign(self._reconstruct_V())
                    m, _ = self.model.score(val_data)
                    vloss = float(m["rmse"])
                    # Reg-free FULL-train RMSE, same scale as val rmse (plot
                    # parity) — always the whole train set, not the minibatch.
                    train_rmse = float(tf.sqrt(tf.reduce_mean(tf.square(
                        self._forward(train_data) - train_data["targets"]))))
                    snap = self._snapshot()
                # All three history keys appended TOGETHER, only on val ticks, so
                # they stay equal-length (mirrors SNES.fit history recording).
                history["generation"].append(gen)
                history["train_loss"].append(train_rmse)
                history["val_loss"].append(vloss)
                last_train, last_val = train_rmse, vloss
                if vloss < best_val:
                    best_val = vloss
                    best_snap = snap
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1
                    # Plateau LR decay (opt-in): cut lr each time the stall count
                    # reaches a multiple of the decay patience (SNES sigma-anneal
                    # analog). Floored at adam_lr_min.
                    if (cfg.adam_lr_decay and gens_without_improvement > 0
                            and gens_without_improvement
                            % int(cfg.adam_lr_decay_patience) == 0):
                        new_lr = max(float(self._keras.learning_rate)
                                     * float(cfg.adam_lr_decay_factor),
                                     float(cfg.adam_lr_min))
                        self._keras.learning_rate = new_lr
                if (cfg.patience is not None
                        and gens_without_improvement >= cfg.patience):
                    break

            # Progress bar (mirrors SNES.fit), rendered every gen.
            frac = (gen + 1) / n_gen
            bar_len = 30
            bar = "█" * int(bar_len * frac) + "░" * (bar_len - int(bar_len * frac))
            elapsed = time.perf_counter() - train_start
            eta = elapsed / frac * (1 - frac) if frac > 0 else 0
            sys.stdout.write(
                f"\r{bar} {gen + 1}/{n_gen} "
                f"train RMSE: {last_train:.6f}  val RMSE: {last_val:.6f}  "
                f"best val RMSE: {best_val:.6f}  "
                f"elapsed: {_format_duration(elapsed)}  ETA: {_format_duration(eta)}")
            sys.stdout.flush()

            if plot_callback is not None:
                plot_callback(history)

        print()  # newline after the progress bar

        # Build result models: final = snapshot-of-now; best = best_snap (fall
        # back to current if val never ran). Snapshot under EMA weights (no-op if
        # EMA off) so returned models use the averaged iterate.
        with self._eval_weights():
            final_model = self._model_from_snapshot(self._snapshot())
            if best_snap is None:
                best_snap = self._snapshot()
        best_val_model = self._model_from_snapshot(best_snap)

        # Restore best into self.model (incl. U_pair=V_best) for compatibility.
        _set_model_params(self.model, *self._params_from_snapshot(best_snap))
        self._restore_hidden_layer(self.model, best_snap)
        if self._mixing:
            self.A.assign(best_snap["A"])

        return history, final_model, best_val_model


def make_optimizer(model: "TNEP"):
    if str(getattr(model.cfg, "optimizer", "snes")).lower() == "adam":
        return Adam(model)
    from SNES import SNES
    return SNES(model)
