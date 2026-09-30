# GAP 2-body / 3-body distance descriptors — ANN sum experts and GAP-mode column blocks

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TurboGAP's distance-only 2-body and 3-body descriptors (`gapf90/gap.f90`, `get_2b_energy_and_forces` lines 210-341 and `get_3b_energy_and_forces` lines 489-747 in the user's main checkout; a copy is at `gapf90_gap.f90.ref` in this worktree) as (a) the two experts of the ANN sum model, alongside the SOAP ANN, and (b) additive linear column blocks in the GAP-mode dipole fit (`gap_dipole.py`).

**Architecture:** One new module `gap_descriptor.py` defines the per-atom descriptors exactly as GAP's neighbour sums with FIXED centres (GAP's sparse points) and widths: `q2[i, t′, s] = Σ_{j∈t′} f_cut(r_ij)·exp(−(r_ij−Q_s)²/2σ²)` and `q3[i, p, s] = Σ_{j<k∈p} f(r_ij) f(r_ik)·exp(−½|q_jk − Q_s|²_σ)` with `q_jk = [r_ij+r_ik, (r_ij−r_ik)², r_jk]` and GAP's buffered cosine cutoff. With a linear readout these ARE TurboGAP's 2b+3b site energies (exactness test transcribes gap.f90's energy loops). Built once at data time (NumPy) into the existing raw-expert data slot (`nep_raw`, renamed in docstrings to "raw expert input"), so every SNES/Adam/score/validate site that already threads `nep_raw` carries it unchanged. A new `nep_descriptor="gap_sum"` mode feeds q2 to ANN_2b and q3 to ANN_3b through the existing sum-mode expert machinery (no learned c tensors, no NEP contraction). In GAP mode, `gap_2b`/`gap_3b` flags append linear column blocks (per centre species, columns = centres × geometry columns) to `_phi_frame`.

**Tech Stack:** NumPy (descriptor build), TensorFlow (forward, unchanged experts), ase `neighbor_list`, pytest. Reference: `gapf90_gap.f90.ref`.

---

## 0. Design

### 0.1 Descriptors (`gap_descriptor.py`), GAP-exact

Cutoff (`gap.f90:270-275`): f(r) = 1 for r < r_cut − buffer; ½(1 + cos(π (r − r_cut + buffer)/buffer)) for r_cut − buffer ≤ r < r_cut; 0 beyond.

2-body (`gap.f90:264-281`, kernel `exp(−½ (r − Q_s)²/σ²)`), per atom i, per NEIGHBOUR species t′, per centre s:

    q2[i, t′, s] = Σ_{j: t_j = t′, r_ij < rc2} f(r_ij; rc2, buf) · exp(−(r_ij − Q2_s)² / (2 σ2²))

GAP's per-species-pair term (sp1, sp2) equals Σ_s α_s (Σ_{i∈sp1} q2[i, sp2, s] + [sp1 ≠ sp2]·Σ_{i∈sp2} q2[i, sp1, s]) — for sp1 == sp2 the Fortran filter collapses to ONE clause (no mirrored term). Our layout keeps the centre species implicit (the ANN is per type; in GAP mode the columns are per centre species) and the neighbour species explicit; GAP's symmetric term is the tied-α special case. Width W2 = T·M2. NOTE the 2b Fortran prefactor is δ²·α (no 2), the 3b prefactor is 2·δ²·α (`gap.f90:563`): the exactness identities are E2b = Σ_s α_s (…) and E3b = 2·Σ_s α_s q3 — do NOT insert a 2 into q3.

3-body (`gap.f90:571-647`, kernel_type "exp"): triplets centred on i over unordered neighbour pairs j < k, both within rc3, cutoff f(r_ij)·f(r_ik) only (no cutoff on r_jk), descriptor q = [r_ij + r_ik, (r_ij − r_ik)², r_jk] (symmetric in j↔k), per unordered neighbour-species pair p = {t_j, t_k} (packed like `TNEP._pair_idx` symmetric: T(T+1)/2 slots), per centre s:

    q3[i, p, s] = Σ_{j<k ∈ p} f(r_ij) f(r_ik) · exp(−½ Σ_d (q_d − Q3_{s,d})² / σ3_d²)

Width W3 = T(T+1)/2 · M3. (`pol` kernel / cov_pp NOT implemented — exp only; note in docs.)

r_jk is |D_ik − D_ij| from the two centre displacements (image-exact, as `xyz23 = xyz13 − xyz12` in gap.f90:604). The T(T+1)/2 symmetric pair packing lives in `gap_descriptor.pair_index(T)` (shared by the tests); do not import `TNEP._pair_idx`, which exists only under the neighbor readout.

Centres and widths are config: `gap2b_sigma` (float), `gap3b_sigma` (3 floats), `gap2b_n_centres` (M2), `gap3b_n_centres` (M3), plus runtime `gap2b_centres: list|None`, `gap3b_centres: list[list]|None` (stored in cfg, saved with the model like `q_scale`). Auto-selection when None, by `gap_descriptor.select_centres(frames, cfg)`. In `MasterTNEP._train_model_inner` it runs AFTER seed resolution (~417-421; FPS uses cfg.seed) and `cfg.randomise` (~428) and BEFORE `split(...)` (~432, which calls `assemble_data_dict` and fills the raw slot), on the TRAINING frames only (`cfg.indices` minus the val/test split — or note the leak if simpler), guarded by `if cfg.gap2b_centres is None` so a checkpoint/resume cfg keeps its centres. In `fit_gap_dipole` the analogue is after the `cfg.indices` subsetting (~236) and before Φ is built; `_validate` runs first and must not require centres. Selection: 2b → uniform grid on [0.5 Å, rc2]; 3b → farthest-point sampling (σ-scaled Euclidean) over a random subsample (≤ 200k) of the training triplet descriptors. Also `gap2b_rcut`, `gap3b_rcut`, `gap_buffer` (GAP's `rcut`/`buffer`).

`gap_raw_numpy(atoms, types_int, cfg) -> [N, W2 + W3] float32`, `gap_slices(cfg) -> {"two_body": slice, "three_body": slice}`, `gap_dim(cfg)`.

### 0.2 ANN: `nep_descriptor="gap_sum"`

Same machinery as "sum": SOAP ANN + ANN_2b + ANN_3b with zero-init heads. Differences: the raw slot holds q2‖q3 (fixed, no contraction), expert inputs are `gap_slices` into it, no c_r/c_a. Implementation hooks in TNEP.py: a model flag `self._experts_on = mode in ("sum", "gap_sum")` replacing the ten `_nep_mode == "sum"` checks (TNEP.py 316/1246/1386/2006/2227/2817, adam.py 41/61/131, model_io.py 263/532); `self._expert_in` built from `gap_slices` under gap_sum; `_expert_outputs` uses `q = nep_raw` directly (no `nep_from_raw`) under gap_sum; `nep_raw_width(cfg)` returns `gap_dim(cfg)` under gap_sum AND `TNEP._nep_dim` is set to `gap_dim(cfg)` under gap_sum (TNEP.py ~300/321: `_scale_width += _nep_dim` feeds the per-expert `_scale_fold` slices — a NEP dim there would mismatch the gap widths under `descriptor_scale`); `validate_nep_cfg` accepts "gap_sum" (no NEP header validation; requires gap widths > 0; rejects tnep like sum); `c_r = c_a = None`; `gen0_ann_inputs` returns `[q_soap ‖ gap_raw]` under gap_sum for the per-expert scale fold; `q_scale_groups` (DescriptorBuilderGPU.py ~481: `if cfg.nep_descriptor == "sum"`) → `in ("sum", "gap_sum")` with `nep_block_widths` dispatching to the two gap sub-blocks (two_body, three_body) under gap_sum; `extra_block_widths` (~405) is concat/replace only and must NOT change. Data: `assemble_data_dict` (data.py ~394) gets an `elif cfg.nep_descriptor == "gap_sum"` branch that fills `nep_raw` with `gap_raw_numpy` and never touches `nep_pair_list`/`nep_raw_numpy`; everything downstream is width-agnostic (SNES struct_keys, adam `_trainable` c-guards, padding by shape, model_io c-guards, `nep_pairs_needed` False, `compute_dim_q` SOAP-only). Complete `_experts_on` site list: TNEP.py 316/1246/1386/2006/2227/2817, adam.py 41/61/131, model_io.py 263/532 (model flag), plus the cfg-string sites DescriptorBuilderGPU.py 481 and the MasterTNEP.py ~497 info line.

### 0.3 GAP mode: `gap_2b: bool = False`, `gap_3b: bool = False`

`_phi_frame` gains, after the SOAP blocks, per centre species t: `cols.append(einsum("ns,nxk->xsk", q2[sel_t], v[sel_t]).reshape(3, −1))` for the 2b block (all T′·M2 columns of q2 for that species) and likewise q3. Column contract order: scale → species → ξ → (m,k), THEN gap2b (species → (s,k)), THEN gap3b (species → (s,k)). Fit and predict both go through `_phi_frame`, whose signature grows to `(scales, v, ti, xis, gap_q=None)` with `gap_q = (q2 [N,W2] | None, q3 [N,W3] | None)` per frame. Three edits that MUST land together (else predict silently drops the block): (a) `fit_gap_dipole` builds q2/q3 per frame itself via `gap_raw_numpy` (GAP mode has `nep_descriptor="off"`, so the data dict has no raw slot) and extends the α split `blocks` and `GapDipoleModel` with `alpha_gap2b`/`alpha_gap3b`; (b) `predict` calls `gap_raw_numpy` per frame and passes `gap_q`; (c) `alpha_flat = concat(alpha + alpha2 + alpha_gap2b + alpha_gap3b)` in the same column order at both ends. `fit_gap_dipole` calls `select_centres` when centres are None and stores them in cfg; the run record JSON records the flags, centres and σ. λ_eff by column norms already handles scale. `_validate`: `nep_descriptor` must stay "off" (unchanged) — the GAP descriptors are a GAP-mode feature, not a NEP one.

### 0.4 Out of scope

`pol` kernel; learned centres/σ; GAP 2b/3b under the tnep readout (no descriptor gradients — validator rejects like "sum"); trajectory inference.

### 0.5 Files

| File | Change |
|---|---|
| `gap_descriptor.py` (new) | cutoff, q2, q3, dims/slices, `gap_raw_numpy`, `select_centres`, NumPy transcription helpers for the tests |
| `TNEPconfig.py` | `gap2b_rcut=3.5`, `gap3b_rcut=3.5`, `gap_buffer=0.5`, `gap2b_n_centres=20`, `gap2b_sigma=0.3`, `gap3b_n_centres=50`, `gap3b_sigma=[0.5,0.5,0.5]`, `gap2b_centres=None`, `gap3b_centres=None`, `gap_2b=False`, `gap_3b=False`; `nep_descriptor` doc gains "gap_sum" |
| `nep_descriptor.py` | `validate_nep_cfg` accepts gap_sum; `nep_raw_width`/`nep_block_widths` dispatch on gap_sum (lazy import of gap_descriptor) |
| `TNEP.py` | `_experts_on`, `_expert_in` from gap_slices, `_expert_outputs` source switch, c tensors None, `gen0_ann_inputs` |
| `adam.py`, `model_io.py` | `_experts_on` instead of `== "sum"` |
| `data.py` | assembly gate for gap_sum |
| `MasterTNEP.py` | centre auto-selection before assembly; info line |
| `gap_dipole.py` | flags, centre selection, column blocks, run record |
| `scripts/smoke_train.py` | `--nep gap_sum` |
| `tests/test_gap_descriptor.py` (new) | exactness vs gap.f90 transcription, invariances, centres, ANN gap_sum (gen-0 identity, four-path, SNES layout, save/load, scaler), GAP-mode columns/fit |

---

## Task 1: `gap_descriptor.py` + config + exactness tests

- [ ] Tests first (`tests/test_gap_descriptor.py`, helpers imported from `test_neighbor_dipole`): (a) `_gap2b_energy_f90(atoms, alphas, sp1, sp2, cfg)` and `_gap3b_energy_f90(atoms, alphas, sp0, sp1, sp2, cfg)` — direct NumPy transcriptions of the gap.f90 ENERGY loops (delta=1, cutoff(s)=1, e0=0, exp kernel, the same neighbour list) — and assert `E2b == Σ_s α_s (Σ_{i∈sp1} q2[i,sp2,s] + [sp1≠sp2]·Σ_{i∈sp2} q2[i,sp1,s])` and `E3b == 2·Σ_s α_s Σ_{i∈sp0} q3[i,{sp1,sp2},s]` (the 2 is the Fortran `pref`) (rtol 1e-6, random α, random centres, both same- and mixed-species pairs, buffer inside the cutoff so the taper is exercised); (b) rotation/translation/permutation invariance of q2/q3; (c) widths and slices; (d) `select_centres`: 2b grid endpoints, 3b FPS returns M3 distinct rows within the data's bounding box and is deterministic under cfg.seed; (e) `validate` rejects M ≤ 0, σ ≤ 0, buffer > rcut.
- [ ] Implement; `gap_raw_numpy` vectorised over pairs (2b: `np.add.at` per (i, t_j)) and over the padded [N, M] neighbour table for triplets (broadcast M×M with j<k mask; M ≈ 20 at 3.5 Å); float64 inside, float32 out.
- [ ] Commit: `feat: gap_descriptor.py — TurboGAP 2-body/3-body distance descriptors with fixed centres (exact vs gap.f90 energy loops)`.

## Task 2: ANN `gap_sum`

- [ ] Tests: gen-0 identity vs "off" (same seed); experts add (zero SOAP W1 → expert-only); four-path consistency (neighbor + bilinear); SNES layout (n_typed includes experts, no c tail: `n_nep == 0`), candidates == assign+predict_batch, evaluate/validate liveness when only expert weights differ; save/load round trip (centres in cfg round-trip); `descriptor_scale="block"` reparameterisation identity at gen 0 via `gen0_ann_inputs`; `gap_sum` + tnep rejected; Adam 2 steps move W1_r.
- [ ] Implement per §0.2 (`_experts_on` flag everywhere; grep `== "sum"`); MasterTNEP centre auto-selection + info line; `scripts/smoke_train.py --nep gap_sum`.
- [ ] Full suite alone (233 + new, 1 skipped) and a CPU smoke `--readout neighbor --nep gap_sum` and `--readout bilinear --pair-rank 1 --nep gap_sum --scale block`.
- [ ] Commit: `feat: nep_descriptor="gap_sum" — GAP 2b/3b distance descriptors as the sum-model experts`.

## Task 3: GAP mode column blocks

- [ ] Tests (`tests/test_gap_descriptor.py`, using `tests/test_gap_dipole.py`'s `_cfg`): column count = SOAP columns + Σ_t (T′·M2)·K + Σ_t (T(T+1)/2·M3)·K; `_phi_frame` blocks are the q2/q3 · v contractions (compare against an explicit loop on one frame); `fit_gap_dipole` with `gap_2b=True, gap_3b=True` on 6 frames runs, predicts, records flags/centres/σ in the JSON, and `predict` via `_phi_frame` == fit's design (no drift); synthetic μ built from the 2b columns of the same design is recovered to numerical precision by the ridge fit with `gap_2b=True`; the no-drift check compares `predict(model, frames)` against `Phi_flat @ alpha_flat` on the training frames (the Φ the fit solved), which is what guards the three coupled edits of §0.3.
- [ ] Implement per §0.3; `_validate` gains sanity checks (flags need centres or auto-selection; nep off).
- [ ] Commit: `feat: gap_dipole — gap_2b / gap_3b additive column blocks (TurboGAP 2b+3b as linear features)`.

## Task 4: docs + memory

- [ ] `neighbor_dipole_method.md` §2.5: one paragraph on `gap_sum` and the GAP-mode blocks (forms, fixed centres, exactness vs gap.f90, no accuracy claims). Memory note by the coordinator.
