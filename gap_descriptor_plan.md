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

GAP's per-species-pair term (sp1, sp2) equals Σ_s α_s (q2[i∈sp1, sp2, s] + q2[i∈sp2, sp1, s]) — our layout keeps the centre species implicit (the ANN is per type; in GAP mode the columns are per centre species) and the neighbour species explicit. Width W2 = T·M2.

3-body (`gap.f90:571-647`, kernel_type "exp"): triplets centred on i over unordered neighbour pairs j < k, both within rc3, cutoff f(r_ij)·f(r_ik) only (no cutoff on r_jk), descriptor q = [r_ij + r_ik, (r_ij − r_ik)², r_jk] (symmetric in j↔k), per unordered neighbour-species pair p = {t_j, t_k} (packed like `TNEP._pair_idx` symmetric: T(T+1)/2 slots), per centre s:

    q3[i, p, s] = Σ_{j<k ∈ p} f(r_ij) f(r_ik) · exp(−½ Σ_d (q_d − Q3_{s,d})² / σ3_d²)

Width W3 = T(T+1)/2 · M3. (`pol` kernel / cov_pp NOT implemented — exp only; note in docs.)

r_jk is |D_ik − D_ij| from the two centre displacements (image-exact, as `xyz23 = xyz13 − xyz12` in gap.f90:604).

Centres and widths are config: `gap2b_sigma` (float), `gap3b_sigma` (3 floats), `gap2b_n_centres` (M2), `gap3b_n_centres` (M3), plus runtime `gap2b_centres: list|None`, `gap3b_centres: list[list]|None` (stored in cfg, saved with the model like `q_scale`). Auto-selection when None, by `gap_descriptor.select_centres(frames, cfg)` run by the driver BEFORE data assembly: 2b → uniform grid on [0.5 Å, rc2]; 3b → farthest-point sampling (σ-scaled Euclidean) over a random subsample (≤ 200k) of the training triplet descriptors. Also `gap2b_rcut`, `gap3b_rcut`, `gap_buffer` (GAP's `rcut`/`buffer`).

`gap_raw_numpy(atoms, types_int, cfg) -> [N, W2 + W3] float32`, `gap_slices(cfg) -> {"two_body": slice, "three_body": slice}`, `gap_dim(cfg)`.

### 0.2 ANN: `nep_descriptor="gap_sum"`

Same machinery as "sum": SOAP ANN + ANN_2b + ANN_3b with zero-init heads. Differences: the raw slot holds q2‖q3 (fixed, no contraction), expert inputs are `gap_slices` into it, no c_r/c_a. Implementation hooks in TNEP.py: a model flag `self._experts_on = mode in ("sum", "gap_sum")` replacing the ten `_nep_mode == "sum"` checks (TNEP.py 316/1246/1386/2006/2227/2817, adam.py 41/61/131, model_io.py 263/532); `self._expert_in` built from `gap_slices` under gap_sum; `_expert_outputs` uses `q = nep_raw` directly (no `nep_from_raw`) under gap_sum; `nep_raw_width(cfg)` returns `gap_dim(cfg)` under gap_sum; `validate_nep_cfg` accepts "gap_sum" (no NEP header validation; requires gap widths > 0; rejects tnep like sum); `c_r = c_a = None`; `gen0_ann_inputs` returns `[q_soap ‖ gap_raw]` under gap_sum for the per-expert scale fold; `extra_block_widths`/`nep_block_widths` → gap sub-blocks under gap_sum (two_body, three_body) for `q_scale_groups`. Data: `assemble_data_dict` fills `nep_raw` with `gap_raw_numpy` under gap_sum (gate `cfg.nep_descriptor == "gap_sum"`); everything downstream unchanged.

### 0.3 GAP mode: `gap_2b: bool = False`, `gap_3b: bool = False`

`_phi_frame` gains, after the SOAP blocks, per centre species t: `cols.append(einsum("ns,nxk->xsk", q2[sel_t], v[sel_t]).reshape(3, −1))` for the 2b block (all T′·M2 columns of q2 for that species) and likewise q3. Column contract order: scale → species → ξ → (m,k), THEN gap2b (species → (s,k)), THEN gap3b (species → (s,k)). Fit and predict both go through `_phi_frame`; `fit_gap_dipole` calls `select_centres` when centres are None and stores them in cfg; the run record JSON records the flags, centres and σ. λ_eff by column norms already handles scale. `_validate`: `nep_descriptor` must stay "off" (unchanged) — the GAP descriptors are a GAP-mode feature, not a NEP one.

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

- [ ] Tests first (`tests/test_gap_descriptor.py`, helpers imported from `test_neighbor_dipole`): (a) `_gap2b_energy_f90(atoms, alphas, sp1, sp2, cfg)` and `_gap3b_energy_f90(atoms, alphas, sp0, sp1, sp2, cfg)` — direct NumPy transcriptions of the gap.f90 ENERGY loops (delta=1, cutoff(s)=1, e0=0, exp kernel, the same neighbour list) — and assert `Σ_s α_s·q2[i, sp2, s]` summed over atoms i of species sp1 plus the mirrored term equals the 2b transcription; `Σ_s α_s q3[i,{sp1,sp2},s]` over i∈sp0 equals the 3b transcription (rtol 1e-6, random α, random centres, both same- and mixed-species pairs, buffer inside the cutoff so the taper is exercised); (b) rotation/translation/permutation invariance of q2/q3; (c) widths and slices; (d) `select_centres`: 2b grid endpoints, 3b FPS returns M3 distinct rows within the data's bounding box and is deterministic under cfg.seed; (e) `validate` rejects M ≤ 0, σ ≤ 0, buffer > rcut.
- [ ] Implement; `gap_raw_numpy` vectorised over pairs (2b: `np.add.at` per (i, t_j)) and over the padded [N, M] neighbour table for triplets (broadcast M×M with j<k mask; M ≈ 20 at 3.5 Å); float64 inside, float32 out.
- [ ] Commit: `feat: gap_descriptor.py — TurboGAP 2-body/3-body distance descriptors with fixed centres (exact vs gap.f90 energy loops)`.

## Task 2: ANN `gap_sum`

- [ ] Tests: gen-0 identity vs "off" (same seed); experts add (zero SOAP W1 → expert-only); four-path consistency (neighbor + bilinear); SNES layout (n_typed includes experts, no c tail: `n_nep == 0`), candidates == assign+predict_batch, evaluate/validate liveness when only expert weights differ; save/load round trip (centres in cfg round-trip); `descriptor_scale="block"` reparameterisation identity at gen 0 via `gen0_ann_inputs`; `gap_sum` + tnep rejected; Adam 2 steps move W1_r.
- [ ] Implement per §0.2 (`_experts_on` flag everywhere; grep `== "sum"`); MasterTNEP centre auto-selection + info line; `scripts/smoke_train.py --nep gap_sum`.
- [ ] Full suite alone (233 + new, 1 skipped) and a CPU smoke `--readout neighbor --nep gap_sum` and `--readout bilinear --pair-rank 1 --nep gap_sum --scale block`.
- [ ] Commit: `feat: nep_descriptor="gap_sum" — GAP 2b/3b distance descriptors as the sum-model experts`.

## Task 3: GAP mode column blocks

- [ ] Tests (`tests/test_gap_descriptor.py`, using `tests/test_gap_dipole.py`'s `_cfg`): column count = SOAP columns + Σ_t (T′·M2)·K + Σ_t (T(T+1)/2·M3)·K; `_phi_frame` blocks are the q2/q3 · v contractions (compare against an explicit loop on one frame); `fit_gap_dipole` with `gap_2b=True, gap_3b=True` on 6 frames runs, predicts, records flags/centres/σ in the JSON, and `predict` via `_phi_frame` == fit's design (no drift); 2b-only linear identity: with SOAP disabled?? — SOAP cannot be disabled in gap mode (keep) — instead assert that adding the 2b block with α chosen to reproduce a synthetic TurboGAP 2b site-energy profile fits it to numerical precision (build synthetic μ from the same columns).
- [ ] Implement per §0.3; `_validate` gains sanity checks (flags need centres or auto-selection; nep off).
- [ ] Commit: `feat: gap_dipole — gap_2b / gap_3b additive column blocks (TurboGAP 2b+3b as linear features)`.

## Task 4: docs + memory

- [ ] `neighbor_dipole_method.md` §2.5: one paragraph on `gap_sum` and the GAP-mode blocks (forms, fixed centres, exactness vs gap.f90, no accuracy claims). Memory note by the coordinator.
