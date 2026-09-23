# Descriptor scale preconditioning that cannot regress — plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** give the ANN / SNES a well-conditioned view of the descriptor (the SOAP block spans a 50× per-channel spread; the learned-basis extras sit at its top end) in a way that is provably a pure reparameterisation of today's model, identical at generation 0, and continuously tunable back to the unscaled run.

**Why every previous scaler regressed (facts from the repo history):**
- 2026-05-13 (removed): multiplied `q` **and** `∂q/∂R` by `1/(max − min)` per channel. Range is outlier-sensitive; 65 of 135 channels had std < 1 % of the max, so numerical-dust channels were amplified to unit scale and entered both the hidden layer and the dipole contraction through the amplified gradients. The model class itself changed (inputs differ at fixed weights), and there was no path back to the unscaled run.
- 2026-08-06 (planned, never built): σ-scaling / μ-reparameterisation with std, clamp and exponent 0.5. This plan is that idea, reduced to one mechanism with hard equivalence tests.

## Invariants (each is a test, not a hope)

| # | Invariant | How it is guaranteed |
|---|---|---|
| I1 | `descriptor_scale="off"` is bit-identical to today | the fold is skipped entirely; no tensor is touched |
| I2 | Pure reparameterisation | `q` and `∂q/∂R` are never modified. The scale `m [Q]` is folded into the input weights in raw-descriptor space, `W0_eff = diag(m)·W0`, after the mixing / preprocess folds. Any scaled model is exactly the unscaled model with `W0' = m ∘ W0`: same model class, all readouts (including gradient-consuming ones) unchanged |
| I3 | Identical function at generation 0 | both weight initialisations (Keras `W0` and SNES μ Glorot) are divided by `m`, so `W0_eff` at init equals the unscaled init; tested to 1e-6 at every entry point |
| I4 | Continuous path back | `m_k = clip((s_ref / s_k)^α, 1/c, c)`; `α = 0 ⇒ m ≡ 1 ⇒` the whole run (same seed) is identical to `"off"`; worst case is bounded by α |
| I5 | No moving target | `s_k` = per-channel std over real training atoms of the **initial** descriptor (`M0`), computed once, frozen, saved with the model. Under the learned basis it is a preconditioner, not a normaliser, so it never chases `M` |
| I6 | Dust cannot be amplified past `c` | clamp `c = 10` caps the boost of a quiet channel at 10× and the damping of a loud one at 10× (full equalisation would cut the `l = 0` block's step 20×, measured 2026-08-06); channels with `s_k = 0` are set to `m = 1` — dead channels are left alone, never boosted |
| I7 | One fold site | every forward reaches `W0` through `TNEP._W0_eff`; the nine current call sites (several conditional on mixing) are made unconditional first, and a test pins that all entry points agree |

Two granularities, one knob: `descriptor_scale = "off" | "block" | "channel"`.
`"block"` uses one scalar per group (each SOAP `l` group from `descriptor_block_layout["l_index"]`, plus each extra block: 2-body / 4-body / 5-body) — ≤ 9 scalars, the safe first step. `"channel"` is per channel. `α` is the fixed constant 0.5 and `c = 10` (from the 2026-08-06 measurements); no further knobs.

Interactions, stated: L1/L2 act on the stored `W0 = W0'/m`, so channels boosted by `m` are penalised less per unit of effective weight — the "scale-fair" penalty of the earlier plan; bounded by α. SNES keeps uniform σ; the effective per-channel step in original units becomes `σ·m_k`. Adam sees the same reparameterised geometry. `descriptor_preprocess_contract != "off"` is rejected with scaling (its fold rewrites `W0` at a different width).

## File map

| File | Change |
|---|---|
| `TNEPconfig.py` | `descriptor_scale: str = "off"`; runtime field `q_scale: list[float] | None = None` (serialised in the config JSON, so `save_model` / `load_model` / checkpoints carry it for free) |
| `DescriptorBuilderGPU.py` | `compute_q_scale(q, atom_mask, cfg) -> np.ndarray [Q]` (pure numpy; block map from `descriptor_block_layout` + `radial_extra_count`) |
| `TNEP.py` | `_W0_eff`: multiply by `m` in raw-q space after the U fold; `__init__`: `self._q_scale` constant, divide the Keras `W0` (`W0_pol`) init by `m`; make every call site unconditional |
| `SNES.py` | `_build_mu_init`: divide the W0 (and W0_pol) Glorot block by `m`; `validate` / `_evaluate_chunk_impl`: unconditional `_W0_eff` |
| `adam.py`, `spectroscopy.py`, `TNEP.predict_batch_candidates` | unconditional `_W0_eff` |
| `MasterTNEP.py` | compute `cfg.q_scale` from the training set before the real model is built (fixed basis: from `train_data["descriptors"]`; learned basis: a throwaway `TNEP(cfg)` at `M0` with scaling off, `soap_from_raw` over the train chunks); print `min / median / max m` per block |
| `model_io.py` | `_LEGACY_FIELD_DEFAULTS`: `descriptor_scale: "off"`, `q_scale: None` |
| `tests/test_q_scale.py` | new |

---

### Task 1: one fold site

- [ ] Make all nine `_W0_eff` call sites unconditional (`_W0_eff` already returns `W0` unchanged when mixing is off): `TNEP.predict_batch_candidates:1660`, `SNES.validate:1299/1326`, `SNES._evaluate_chunk_impl:1992`, `adam._forward:106/109`; leave `TNEP.predict/score` and `spectroscopy` (already unconditional).
- [ ] Test `test_every_entry_point_agrees`: one model (mixing on, learned basis on), same structures through `predict_batch` (score path), `predict_batch_candidates` (C = 1), `SNES.validate(mu)`, `adam._forward`, and the spectroscopy fused path; all predictions equal to 1e-5. (This is the regression net for everything below.)
- [ ] Run all suites; commit.

### Task 2: the statistic

- [ ] `compute_q_scale(q [N, Q], mask [N], cfg)`:
  ```python
  s = q[mask > 0].std(axis=0)                       # real atoms only
  s_ref = np.median(s[s > 0])
  m = np.clip((s_ref / np.where(s > 0, s, s_ref)) ** ALPHA, 1.0 / CLAMP, CLAMP)
  m[s == 0] = 1.0                                   # dead channels: leave alone
  if cfg.descriptor_scale == "block":               # one value per group
      for idx in groups(cfg): m[idx] = np.clip((s_ref / np.median(s[idx])) ** ALPHA, 1/CLAMP, CLAMP)
  ```
  with `ALPHA = 0.5`, `CLAMP = 10.0` module constants; groups = SOAP `l` groups + extra blocks.
- [ ] Tests: `ALPHA = 0 ⇒ m ≡ 1`; a channel with std 1e-6 clamps at `CLAMP`, not 1e3; a zero-std channel gives 1; block mode is constant within each group and covers every channel exactly once; padded atoms excluded (set them to 1e9 and check `m` unchanged).
- [ ] Commit.

### Task 3: the fold and the init

- [ ] `TNEP.__init__`: `self._q_scale = tf.constant(cfg.q_scale)` when `descriptor_scale != "off"` (shape check `== dim_q_forward`; raise if `cfg.q_scale is None` — the driver must have computed it); divide `W0`/`W0_pol` Keras init by `m` along the raw-q axis; reject preprocess contraction.
- [ ] `_W0_eff`: after the U fold, `W0 = W0 * m[None, :, None]` (broadcasts over the candidate axis).
- [ ] `SNES._build_mu_init`: divide the W0 (and W0_pol) Glorot block by `m` (reshape `[T, Q, H]`, divide along Q).
- [ ] Tests (the invariants): I3 — scaled model vs unscaled model with the same seed: predictions identical to 1e-6 at every entry point of Task 1; I4 — `ALPHA=0` monkeypatched: `evaluate_population` fitness and `validate` identical to the unscaled model for a 2-candidate population; I2 — set arbitrary `W0` on the scaled model, copy `W0 * m` into the unscaled model, predictions identical; Adam: one `_step` on both models with `m ≡ 1` gives identical weights.
- [ ] Commit.

### Task 4: driver, persistence

- [ ] `MasterTNEP._train_model_inner`: after `pad_and_stack`, if `descriptor_scale != "off"` and `cfg.q_scale is None`: get the gen-0 descriptor of the train set (fixed basis: `train_data["descriptors"]`; learned basis: throwaway `TNEP(cfg)` with `descriptor_scale="off"`, `soap_from_raw` chunk-wise on the GPU), call `compute_q_scale`, set `cfg.q_scale = m.tolist()`, print per-block `min/median/max`. Checkpoint resume: `cfg.q_scale` is in the saved config, so it is reused, never recomputed.
- [ ] `model_io._LEGACY_FIELD_DEFAULTS`: both fields. `load_model` needs nothing else (the vector rides in the config JSON).
- [ ] Tests: save → load → identical predictions; a loaded model's `cfg.q_scale` equals the trained one; `descriptor_scale != "off"` with `q_scale=None` raises at `TNEP(cfg)`.
- [ ] GPU smoke: `train_model` 5 generations, `"block"` and `"channel"`, learned basis with all extras; `Run complete!`.
- [ ] Commit.

### Task 5: acceptance (short, same seed, falsifiable — default stays "off" regardless)

- [ ] Plumbing proof: `ALPHA = 0` run vs `"off"` run, 20 generations, same seed: `history["val_loss"]` identical to the last digit. If not, the fold is wrong somewhere; stop.
- [ ] `"block"` vs `"off"`, then `"channel"` vs `"off"`: 300 generations, same seed, full CHO train set, learned basis + extras on (the configuration that motivated this). Accept a mode only if its val RMSE at 300 generations is ≤ the `"off"` run's. Record the printed numbers in this file. Whatever the outcome, `descriptor_scale` defaults to `"off"`, so nothing changes for existing runs.

## Non-goals
Scaling `q` or `∂q/∂R`; per-channel range statistics; re-estimating `m` during training; any new σ-per-block knob (the reparameterisation already moves the search geometry); applying to preprocess-contract modes.
