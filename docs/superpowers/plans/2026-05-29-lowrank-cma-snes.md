# Low-Rank Covariance for SNES (#5) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give SNES a **diagonal + low-rank** sampling covariance whose low-rank factor is **learned internally** from evolution paths (CMA-style), capturing parameter correlations the per-dim diagonal cannot — recovering much of full-CMA's conditioning at `O(d·k)` (full CMA is infeasible at `d≈15k`: a dense `C` is ~1.8 GB and its update is `O(d²)`).

**Architecture:** Reuse the existing SNES sampling substrate — `ask()` already returns `(samples, aux={s_iso, delta})`, the mean step is covariance-agnostic (`μ += Σ u_p δ_p`), and the σ step consumes **only** `s_iso`. We add a learned low-rank correction to the **displacement** `delta` (exactly where guided-ES adds its correction), keeping `s_iso` correction-free: `delta = σ⊙s_iso + Σ_j a_j ξ_j (σ⊙p_c^(j))`, so the realized normalized draw is `s_eff = delta/σ = s_iso + Σ_j a_j ξ_j p_c^(j)` and `Σ = diag(σ²)·(I + Σ_j a_j² p_c^(j) p_c^(j)ᵀ)`. **The σ update keeps seeing `s_iso` only** — the rank correction never leaks into the diagonal, the proven guided-ES discipline (a per-coordinate σ cannot remove rank-1 directional variance, so letting σ chase it biases σ upward along `p_c`; we avoid that by isolation, not by hoping a damping constant saves us). The CMA evolution path is driven by the **global**-fitness-ranked realized step `delta/σ` (never the per-type-permuted tensor — `p_c` is a cross-coordinate object and per-type column shuffling would corrupt its direction). Default OFF ⇒ vanilla SNES stays **bit-identical**.

**Tech Stack:** TensorFlow 2 (eager + tf.Variable state), the existing `ask()`/`update()`/`fit()`/`_build_per_type_gradients`, `model_io.save_checkpoint`/`load_checkpoint`.

**Scope of THIS plan:** Build the **rank-1 stepping stone** (Tasks B0–B4) end-to-end: config, learned evolution path `p_c`, rank-1 sampling, the rank-1 covariance update, a **mandatory bounded-covariance long-run test**, and checkpoint round-trip. The **LM-CMA rank-k** extension (Task B5) is specified in design + decision-gate form only and re-planned into TDD tasks once rank-1 proves out on a benchmark (see "Decision gate before rank-k").

---

## Why a stepping stone, and the calibration trap (read first)

The earlier σ-cumulation divergence in this project (an uncalibrated step-size law that compounded to float overflow) is the cautionary tale: **a covariance-adaptation law with wrong damping constants has the same failure mode.** Two rules follow, both enforced below:

1. **Port reference constants verbatim; do not derive.** Rank-1 constants come from Hansen's CMA-ES tutorial (Hansen 2016, *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772). Rank-k constants (Task B5) come from Loshchilov 2014 (*A Computationally Efficient Limited Memory CMA-ES*, LM-CMA) — **not** invented.
2. **A bounded-covariance long-run test is mandatory** (Task B3): assert `σ` and the low-rank factor norm stay bounded over ≥300 gens at pop=100. No merge without it.

**`d≈15k` scaling caveat (must be surfaced, not silently absorbed):** the canonical rank-1 learning rate `c_1 = 2/((d+1.3)² + μ_eff)` is ≈ `9e-9` at `d=15000`, and `c_c ≈ 4/d ≈ 2.7e-4`. So with *faithfully ported* CMA constants the rank-1 correction adapts **extremely slowly** — it is a **correctness** stepping stone (proves the path/sampling/checkpoint machinery is bounded and wired right), **not** a performance win on its own. Real conditioning gains require LM-CMA's constants, which scale with the stored-vector count `m` (not `d²`). The plan therefore (a) ports the canonical constants for B0–B4 so the stone is provably faithful+bounded, and (b) exposes a single optional override `snes_cma_c1_scale` (default 1.0 = canonical) so a benchmark can probe larger `c_1` deliberately, with the bounded test as the guardrail.

## Background the implementer must know (current code)

- **`ask()`** ([SNES.py:939](../../../SNES.py)) draws mirrored `s_iso = _mirrored_normal((pop,dim))`, sets `delta = s_iso * sigma`, optionally adds the guided term `gamma·(eps_k @ Uᵀ)` to `delta`, returns `(mu+delta, {"s_iso": s_iso, "delta": delta})`.
- **`update(utilities, aux)`** ([SNES.py:1007](../../../SNES.py)): `grad_mu = Σ u_p s_iso_p`; mean step branches — Adam / guided-displacement (`μ += Σ u_p delta_p`) / vanilla-exact (`μ += sigma·grad_mu`); σ step uses `grad_sigma = Σ u_p (s_iso²−1)` (cumulation EMA optional), then floor. **The mean step is covariance-agnostic via `delta`; the σ step uses `s_iso` only.**
- **`fit()`** ([SNES.py:1219](../../../SNES.py)): per gen — optional guided refresh, `samples, aux = self.ask()`, evaluate, rank. **Ranking permutes BOTH `aux["s_iso"]` and `aux["delta"]`** by the same fitness order (per-type via `_build_per_type_gradients` on each; non-per-type via `tf.gather(..., ranks)`), then `self.update(self.utilities, {"s_iso":…, "delta":…})`. The mean displacement consumed by the evolution path must be computed **inside `update()`** from these already-sorted, utility-weighted tensors.
- **Lazy state** lives in `__init__` ([SNES.py:300-321](../../../SNES.py)) next to the Adam/guided buffers.
- **Checkpoint**: `fit()` builds `ckpt_state` ([SNES.py:1590](../../../SNES.py)) with guarded optional keys (hybrid Adam state pattern); `model_io.save_checkpoint` ([model_io.py:294](../../../model_io.py)) writes them under group `snes`; `load_checkpoint` ([model_io.py:373](../../../model_io.py)) reads them back guarded. **Follow the `adam_m` guarded pattern exactly** so pure-SNES checkpoints stay byte-identical.
- **`compute_utilities()`** ([SNES.py:890](../../../SNES.py)) returns zero-mean log-rank weights `u`; internally it builds `raw` (normalised to sum 1, the CMA recombination weight vector) then subtracts `1/λ`. **Expose `raw` directly** — add `self._recomb_w = tf.constant(raw, tf.float32)` in `compute_utilities` and read μ_eff from it, rather than algebraically inverting the shift (`max(u+1/λ,0)` happens to recover it but is fragile). `μ_eff = 1/Σ raw_i²`.

## Test conventions

Mirror `tests/test_guided_es.py`: CPU-only (`CUDA_VISIBLE_DEVICES=''`), a module-scoped `tiny_model` fixture that **pins the optimizer baseline** (`optimizer_mode="snes"`, `snes_mean_optimizer="vanilla"`, `snes_sigma_cumulation=False`, `guided_es_enabled=False`, and the new `snes_cov_mode="none"`) so tests are hermetic regardless of class-default toggles the user flips for experiments. Each test opts into the feature explicitly. Config-defaults test asserts **types/ranges**, not exact default values.

---

## File structure

- **Modify `TNEPconfig.py`** — add the low-rank CMA config block.
- **Modify `SNES.py`** — lazy state in `__init__`; `_cma_constants()` helper (ported); `_ensure_cma_state()`; rank-1 branch in `ask()`; evolution-path + rank-1 covariance update in `update()`; checkpoint save block; resume restore block.
- **Modify `model_io.py`** — guarded save/load of CMA state under group `snes`.
- **Create `tests/test_lowrank_cma.py`** — constants, sampling covariance, vanilla bit-identity, bounded long-run, checkpoint round-trip, per-type composability.

---

# Rank-1 stepping stone (BUILD THIS)

### Task B0: Config + ported constants + lazy state

**Files:** Modify `TNEPconfig.py`, `SNES.py`; Create `tests/test_lowrank_cma.py`.

- [ ] **Step 1 — failing test** for config defaults and the ported constants helper.

```python
def test_cma_constants_match_reference(tiny_model):
    import numpy as np
    model, _, _ = tiny_model
    snes = model.optimizer
    n = snes.dim
    cc, c1, mu_eff = snes._cma_constants()
    raw = snes._recomb_w.numpy()                 # pre-shift recombination weights
    assert np.isclose(raw.sum(), 1.0, atol=1e-6)
    assert np.isclose(mu_eff, 1.0/np.sum(raw**2), rtol=1e-5)
    assert np.isclose(cc, (4 + mu_eff/n)/(n + 4 + 2*mu_eff/n), rtol=1e-6)
    assert np.isclose(c1, 2.0/((n + 1.3)**2 + mu_eff), rtol=1e-6)

def test_cma_config_defaults():
    from TNEPconfig import TNEPconfig
    cfg = TNEPconfig()
    assert cfg.snes_cov_mode in ("none", "rank1", "lowrank")
    assert cfg.snes_cma_c1_scale > 0
```

- [ ] **Step 2** run → FAIL.
- [ ] **Step 3** add config (new block, near the guided-ES block):

```python
    # --- Low-rank covariance (CMA-style learned correction) ------------
    #   "none"   : vanilla per-dim diagonal SNES (bit-identical).
    #   "rank1"  : add a single learned evolution-path direction p_c to the
    #              sampling covariance (CMA-ES rank-1 update; O(d)).
    #   "lowrank": LM-CMA rank-k (Task B5; not yet wired — treated as "rank1"
    #              until implemented, with a one-time warning).
    #   The σ (diagonal) update is UNCHANGED — the rank correction is excluded
    #   from grad_sigma, exactly as guided-ES does, so the diagonal can't
    #   absorb the injected directional variance. Default "none".
    snes_cov_mode: str = "none"
    snes_cma_c1_scale: float = 1.0   # multiplies the ported rank-1 rate c_1
                                     # (1.0 = canonical Hansen; >1 probes faster
                                     #  adaptation — guarded by the bounded test)
```

  Add lazy state in `__init__` next to the guided block:

```python
        # Low-rank CMA state (lazy — only materialised when snes_cov_mode != "none").
        self._cma_pc = None          # tf.Variable [dim] rank-1 evolution path (normalized units)
        self._cma_a1 = None          # tf.Variable scalar — current rank-1 sampling amplitude √(c1·scale)·... (see B2)
```

  Add the ported-constants helper (cache the scalars; they depend only on dim/pop):

```python
def _cma_constants(self) -> tuple[float, float, float]:
    """Ported CMA-ES rank-1 constants (Hansen 2016, arXiv:1604.00772,
    Eq. 24 + 57). μ_eff = 1/Σ w_i² over the recombination weights
    self._recomb_w (the pre-shift positive log-rank weights, sum 1)."""
    n = float(self.dim)
    w = self._recomb_w.numpy()
    mu_eff = float(1.0 / np.sum(w**2))
    c_c = (4.0 + mu_eff/n) / (n + 4.0 + 2.0*mu_eff/n)
    c_1 = 2.0 / ((n + 1.3)**2 + mu_eff)
    return float(c_c), float(c_1), mu_eff
```

  In `compute_utilities`, before returning, cache the pre-shift weights:
  `self._recomb_w = tf.constant(raw.astype(np.float32))` (where `raw` is the sum-1 vector just before the `- 1/lam` shift).

- [ ] **Step 4** run → PASS.
- [ ] **Step 5** commit: `feat(snes): low-rank CMA config + ported rank-1 constants`.

### Task B1: Rank-1 sampling in `ask()`

**Files:** Modify `SNES.py`; Test.

- [ ] **Step 1 — failing test:** with `snes_cov_mode="rank1"` and a planted `p_c`, the **displacement** `delta` has inflated variance along `p_c` while **`s_iso` is unchanged** (σ-isolated); with `"none"` (or `_cma_pc is None`) the draw is **bit-identical** to vanilla for a fixed RNG state.

```python
def test_rank1_sampling_inflates_pc_variance(tiny_model):
    import numpy as np, tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "rank1"
    v = np.zeros((snes.dim,), np.float32); v[0] = 1.0          # planted unit p_c
    snes._cma_pc = tf.Variable(v, trainable=False)
    snes._cma_a1 = tf.Variable(2.0, dtype=tf.float32, trainable=False)  # large amp
    _, aux = snes.ask()
    d = aux["delta"].numpy()
    s = aux["s_iso"].numpy()
    sig0 = float(snes.sigma.numpy()[0])
    assert np.var(d[:, 0]) > 2.0 * np.var(d[:, 5])             # correction lands in delta
    # σ-isolation: s_iso variance is the plain unit normal, NOT inflated along p_c
    assert np.var(s[:, 0]) < 2.0 * np.var(s[:, 5])
    snes.cfg.snes_cov_mode = "none"

def test_rank1_none_bit_identical(tiny_model):
    import numpy as np
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "none"
    st = snes.tf_rng.state.numpy().copy()
    _, aux_a = snes.ask()
    snes.tf_rng.state.assign(st)
    _, aux_b = snes.ask()
    assert np.array_equal(aux_a["delta"].numpy(), aux_b["delta"].numpy())
```

- [ ] **Step 2** run → FAIL.
- [ ] **Step 3** in `ask()`, add the rank-1 correction to **`delta`** (exactly where guided-ES adds its term), leaving `s_iso` correction-free. After `delta = s_iso * self.sigma` and alongside the guided branch:

```python
        if str(getattr(self.cfg, "snes_cov_mode", "none")).lower() != "none" \
                and self._cma_pc is not None:
            xi = self._mirrored_normal((self.pop_size, 1))        # [P,1] mirrored scalar
            # delta += a1·ξ·(σ⊙p_c)  ⇒  realized normalized step delta/σ = s_iso + a1·ξ·p_c.
            delta = delta + self._cma_a1 * xi * (self.sigma * self._cma_pc)[None, :]
```

  **Key design point (document in the docstring):** the correction lives in `delta`, so `Σ = diag(σ²)·(I + a1²·p_c p_cᵀ)` and the mean step (`Σ u_p δ_p`) is the exact covariance-agnostic natural-gradient step, while the σ step (which reads `s_iso`) is **provably unaffected** — same isolation guided-ES relies on, for the same reason (σ scales all coords uniformly and cannot cancel rank-1 directional variance; coupling it would bias σ upward along `p_c`, the σ-cumulation failure family).

  rank-1 and guided both add to `delta` independently and may coexist, but **rank-1 + guided is untested** — add a one-time warning in `fit()` if both are co-enabled.

- [ ] **Step 4** run → PASS (both tests). Add `test_rank1_mean_step_covariance_agnostic` (μ moves by `Σ u_p δ_p`, asserting the displacement form is used when cov_mode != none).
- [ ] **Step 5** commit.

### Task B2: Evolution-path + rank-1 covariance update in `update()`

**Files:** Modify `SNES.py`; Test.

The evolution path is a **cross-coordinate** object, so it must be built from a **single global ranking** of the realized normalized step `delta/σ` — never the per-type-permuted tensor (whose columns carry different rankings, which would corrupt `p_c`'s direction). `update()` therefore reads `aux["s_eff_global"]` (the globally-fitness-ranked `delta/σ`), which `fit()` supplies (Step 3b). For a direct `update()` call with no global key (unit tests), fall back to `aux["delta"]/σ` in raw sample order.

- [ ] **Step 1 — failing test:** after one `update()` with `cov_mode="rank1"`, `p_c` equals the ported cumulation recurrence applied to the recombination-weighted mean of the realized normalized step; `_cma_a1 == sqrt(c1·scale)`. With `cov_mode="none"`, `_cma_pc` stays None and μ/σ are bit-identical to the vanilla reference.

```python
def test_rank1_evolution_path_recurrence(tiny_model):
    import numpy as np, tensorflow as tf
    model, _, _ = tiny_model
    snes = model.optimizer
    snes.cfg.snes_cov_mode = "rank1"
    snes._ensure_cma_state()
    pc0 = snes._cma_pc.numpy().copy()
    _, aux = snes.ask()                          # no global key → raw-order fallback
    u = tf.constant(snes.compute_utilities(), tf.float32)
    sig = snes.sigma.numpy()
    s_eff = aux["delta"].numpy() / sig           # realized normalized step
    snes.update(u, aux)
    cc, c1, mu_eff = snes._cma_constants()
    w = snes._recomb_w.numpy()                   # recombination weights, sum 1
    mean_step = np.einsum('p,pd->d', w, s_eff)
    expected = (1-cc)*pc0 + np.sqrt(cc*(2-cc)*mu_eff)*mean_step
    assert np.allclose(snes._cma_pc.numpy(), expected, atol=1e-5)
    assert np.isclose(float(snes._cma_a1.numpy()),
                      np.sqrt(c1*snes.cfg.snes_cma_c1_scale), rtol=1e-5)
    snes.cfg.snes_cov_mode = "none"
```

  Note the test uses `snes._recomb_w` (the sum-1 recombination weights), NOT `max(utilities,0)` — they coincide numerically but `_recomb_w` is the canonical CMA weight set the implementation must use.

- [ ] **Step 2** run → FAIL.
- [ ] **Step 3a** add `_ensure_cma_state()` (lazy-allocate `_cma_pc = zeros([dim])`, `_cma_a1 = 0.0`). In `update()`, AFTER the mean step and BEFORE the σ step, add the rank-1 block (guarded on `cov_mode != "none"`):

```python
        if str(getattr(self.cfg, "snes_cov_mode", "none")).lower() != "none":
            self._ensure_cma_state()
            cc, c1, mu_eff = self._cma_constants()
            c1 = c1 * float(getattr(self.cfg, "snes_cma_c1_scale", 1.0))
            # Realized normalized step, GLOBALLY ranked (cross-coordinate p_c
            # must not use per-type column shuffles). fit() supplies it; unit
            # tests fall back to the raw-order displacement.
            s_eff = aux.get("s_eff_global")
            if s_eff is None:
                s_eff = delta / self.sigma
            # Recombination-weighted mean of normalized steps (CMA Eq. 24).
            mean_step = tf.einsum('p,pd->d', self._recomb_w, s_eff)   # [dim]
            new_pc = (1.0 - cc)*self._cma_pc + tf.sqrt(cc*(2.0-cc)*mu_eff)*mean_step
            self._cma_pc.assign(new_pc)
            # Rank-1 covariance C ← (1-c1)C + c1 p_c p_cᵀ realised on the
            # SAMPLING side as added variance c1 along p_c: amplitude a1 = √c1.
            # (NOTE: the (1−c1) shrink of the diagonal base is omitted —
            #  negligible at canonical c1≈9e-9; revisit if snes_cma_c1_scale≫1.)
            self._cma_a1.assign(tf.sqrt(tf.maximum(c1, 0.0)))
```

  Verify the `√(c_c(2−c_c)μ_eff)` factor and the sum-1 weights match Hansen Eq. (24). h_σ (Heaviside stall guard, needs the conjugate path p_σ) is **simplified to 1** for the stepping stone — document this; p_σ/h_σ is a B5 refinement, and the bounded test (B3) covers the residual risk. The σ step below is **unchanged** (`grad_sigma = Σ u_p (s_iso²−1)`, correction-free).

- [ ] **Step 3b** in `fit()`, where the population is ranked (SNES branch, ~line 1373): compute the **global** ranking `global_ranks = tf.argsort(fitness)` (the global RMSE column already drives reporting), and pass `aux2["s_eff_global"] = tf.gather(aux["delta"], global_ranks) / self.sigma` into the dict handed to `update()` — for BOTH the per-type and non-per-type branches. In the non-per-type branch this is the same ordering as the diagonal step (consistent); in per-type it keeps `p_c` on the coherent global ranking while σ/mean stay per-type.

- [ ] **Step 4** run → PASS. Add `test_rank1_none_update_bit_identical` (cov_mode none → μ/σ match a vanilla reference manual computation, and `_cma_pc is None`).
- [ ] **Step 5** commit.

### Task B3: MANDATORY bounded-covariance long-run test + per-type composability

**Files:** Test only (`tests/test_lowrank_cma.py`).

- [ ] **Step 1 — bounded long-run test** (the cumulation-divergence guard). Run ≥300 gens at a realistic pop on the tiny model with `cov_mode="rank1"`; assert finiteness and that σ and `‖p_c‖` stay bounded.

```python
def test_rank1_bounded_long_run():
    # Own hermetic model (pop=100, 300 gens) — the cumulation bug's regression guard.
    ... build tiny CHO model like test_guided_es, cfg.pop_size=100,
        cfg.num_generations=300, cfg.snes_cov_mode="rank1", patience=None ...
    snes = TNEP(cfg).optimizer
    hist = snes.fit(train, val)
    tl = hist["train_loss"]
    assert np.all(np.isfinite(tl)), "rank-1 run went non-finite"
    assert float(np.max(snes.sigma.numpy())) < 100.0*float(cfg.init_sigma), "sigma exploded"
    assert float(np.linalg.norm(snes._cma_pc.numpy())) < 1e3, "evolution path unbounded"
```

- [ ] **Step 2 — per-type composability + p_c-coherence test:** `cov_mode="rank1"` + `per_type_regularization=True` must run finite with σ bounded AND build `p_c` from the **global** ranking, not the per-type-permuted columns. Model built with per-type ON from construction (mirror `test_guided_with_per_type_ranking`). Beyond finiteness, assert the coherence property directly: run one gen with per-type on, capture `aux["delta"]` and the global `fitness`, and verify `_cma_pc` matches the recurrence driven by `tf.gather(delta, argsort(fitness))/σ` — NOT by any per-type permutation. (A finiteness-only test would pass even with a corrupted `p_c`; this assertion is the actual guard.)
- [ ] **Step 3** run both → PASS. Run full `tests/test_lowrank_cma.py`, `tests/test_guided_es.py`, `tests/test_hybrid_optimizer.py` — all green (vanilla untouched).
- [ ] **Step 4** commit: `test(snes): bounded-covariance long-run + per-type rank-1 guards`.

### Task B4: Checkpoint round-trip for the learned state

**Files:** Modify `SNES.py` (save block + resume restore), `model_io.py` (save + load); Test.

- [ ] **Step 1 — failing test:** save a checkpoint mid-run with `cov_mode="rank1"`, reload, and assert `_cma_pc`/`_cma_a1` round-trip exactly; and that a pure-SNES (`cov_mode="none"`) checkpoint contains **no** CMA datasets (byte-compat guard — assert the `snes` group has no `cma_pc` key).

- [ ] **Step 2** run → FAIL.
- [ ] **Step 3**:
  - In `fit()` `ckpt_state` block, add (guarded like `adam_m`): `if self._cma_pc is not None: ckpt_state["cma_pc"]=self._cma_pc; ckpt_state["cma_a1"]=float(self._cma_a1.numpy())`.
  - In `fit()` resume block, add: `if resume_state.get("cma_pc") is not None: self._ensure_cma_state(); self._cma_pc.assign(resume_state["cma_pc"]); self._cma_a1.assign(resume_state["cma_a1"])`.
  - In `model_io.save_checkpoint`, after the `adam_m` block: `if state.get("cma_pc") is not None: sg.create_dataset("cma_pc", data=_np(state["cma_pc"])); sg.attrs["cma_a1"]=float(state["cma_a1"])`.
  - In `model_io.load_checkpoint`, after the `adam_m` block: `if "cma_pc" in sg: resume_state["cma_pc"]=sg["cma_pc"][:]; resume_state["cma_a1"]=float(sg.attrs["cma_a1"])`.
  - **dim-mismatch guard:** on resume, if `cma_pc` length != `self.dim`, drop it with a warning (arch changed) rather than asserting — mirrors guided's rebuild-on-resume tolerance.
- [ ] **Step 4** run → PASS. Confirm `tests/test_hybrid_optimizer.py` checkpoint tests still pass (pure-SNES checkpoint unchanged).
- [ ] **Step 5** commit: `feat(snes): checkpoint round-trip for rank-1 CMA evolution path`.

---

# Task B5 (DESIGN ONLY — re-plan into TDD after the gate): LM-CMA rank-k

Generalise the single `p_c` to `m` stored evolution-path vectors `{p_c^(j)}` updated at intervals, and sample `s_eff = s_iso + Σ_j a_j ξ_j p_c^(j)` with the **Cholesky-factor action applied implicitly** (never form the matrix). Port the update/coefficients from **Loshchilov 2014 (LM-CMA)** exactly — its learning rates scale with `m`, not `d²`, which is what makes the conditioning gain real at `d≈15k`. State to checkpoint extends to the `m`-vector store + their step-indices.

**Decision gate before building rank-k (all must hold):**
1. The rank-1 stone is bounded (B3 green) AND a benchmark with `snes_cma_c1_scale>1` shows a **val-RMSE** improvement over vanilla on a real run (evidence that off-diagonal structure helps — same gate as the guided-ES experiment).
2. Descriptor/data/per-type capacity bottlenecks are ruled out (the diagonal is genuinely the limiter).
3. You are prepared to **port** the LM-CMA reference, not derive it.

If the gate fails (diagonal isn't the bottleneck, or dominant curvature isn't low-rank), **stop** — rank-k buys little and adds the most invasive state.

---

## Caveats carried from the design (keep visible)

- **Most invasive change** — touches core sampling AND a genuine covariance *learning* update; highest correctness risk of the optimizer features. Bounded test (B3) is non-negotiable.
- **No external signal** — unlike guided-ES it discovers the subspace from ranked samples only ⇒ slow warm-up.
- **Calibration-sensitive** — ported constants only; `snes_cma_c1_scale` is the one deliberate override, guarded by B3.
- **Genuinely changes checkpoint state** — `p_c` (and rank-k: the vector store) are part of the search state; B4 persists them. Pure-SNES checkpoints stay byte-identical (guarded keys).
- **σ-isolation (decided)** — rank-1's correction rides in `delta`, NOT `s_iso`, so the σ (diagonal) update is provably unaffected — the same discipline guided-ES uses. (Earlier draft folded it into `s_iso`; rejected in review because a per-coordinate σ cannot cancel rank-1 directional variance, so coupling would bias σ upward along `p_c` — the σ-cumulation failure family — and the canonical-`c1` bounded test would not exercise the dangerous `c1_scale≫1` regime.)
- **Evolution path uses the global ranking** — `p_c` is cross-coordinate; it is driven by the global-fitness-ranked realized step, never the per-type column permutation (which would corrupt its direction). The B3 per-type test asserts this coherence directly, not just finiteness.
- **(1−c1) diagonal shrink omitted** — only the additive `c1 p_c p_cᵀ` term is realized on the sampling side; the `(1−c1)` shrink of the diagonal base is dropped (negligible at canonical c1≈9e-9, revisit if `snes_cma_c1_scale≫1`).
- **Combining with mean-Adam / σ-cumulation / guided is uncharted** — start with rank-1 alone; `fit()` warns if multiple covariance features are co-enabled.
- **Diminishing-returns risk at the gate** — confirm the diagonal is the bottleneck before rank-k.
