# Prune: remove the descriptor-preprocess contraction, the learned-basis NEP extras, and the bilinear readout

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Delete three unused feature families — config options AND all their model/optimiser/IO/test code — as a clean break (no shims, no deprecation paths). User-requested; these are dead experiment branches.

**Baseline:** branch `Prune` from ExperimentalDescriptorLearning @ 3e50c0d. `CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q --ignore=tests/test_charge_head.py` → **275 passed, 1 skipped** (the skip is the calorine oracle). Run pytest ALONE, in the foreground, `timeout 900`. Never run anything in `/home/reece/TNEP`.

**Principle:** deletion only. Where a removal leaves a function with one branch, collapse it. Where it leaves a config field unused, delete the field and its comment. Nothing is kept "in case". After each task the suite must be green with no skips other than calorine, and no test may be deleted except those that exist solely to exercise a removed feature (say which, and why, in the report).

---

## What must NOT be touched (traps)

- `nep_four_body` / `nep_five_body` / `nep_l_max` / `nep_n_max_*` / `nep_basis_size_*`: the **GPUMD NEP header** (`nep_descriptor` modes). These STAY. Only `radial_basis_two_body` / `radial_basis_four_body` / `radial_basis_five_body` (the learned-SOAP-basis extras) go.
- `gap_slices`' `"two_body"` / `"three_body"` keys and everything in `gap_descriptor.py`: the TurboGAP 2b/3b feature. STAYS.
- `dipole_readout="neighbor"` and `"tnep"`, `dipole_channels`, `dipole_species_pairs`, `dipole_radial`, `dipole_cutoff`, `dipole_rc`, `dipole_zeta`, `dipole_gauss_sigma_scale`, `dipole_head`: STAY. Only the **bilinear** readout value and its rank/pair-term machinery go.
- `descriptor_mixing` (`U_pair`) and `descriptor_scale` / `q_scale` / `q_shift`: STAY. Only the **preprocess contraction** (`W_pre_angular`) goes.
- `radial_basis_size` / `radial_basis_species_sum` / `radial_basis_per_l` / `radial_basis_norm` (the learned SOAP radial basis itself): STAYS.

---

## Task 1: remove the learned-basis NEP extras (`radial_basis_two_body` / `_four_body` / `_five_body`)

**Files:** `TNEPconfig.py`, `TNEP.py`, `DescriptorBuilderGPU.py`, `MasterTNEP.py`, `model_io.py`, `nep_descriptor.py` (only if it references them), tests (`test_radial_basis.py`, `test_nep_descriptor.py`, `test_q_scale.py`, `test_neighbor_dipole.py`, `test_gap_descriptor.py`).

- [ ] Delete the three config fields and their comment blocks.
- [ ] `DescriptorBuilderGPU.py`: delete `radial_extra_count`, `_radial_extras_on`, and the extras contribution to `compute_dim_q` / `soap_dim` / `extra_block_widths` / `q_scale_groups` / `descriptor_block_layout`. `extra_block_widths` keeps only its NEP (concat/replace) contribution; if it then has a single source, keep the function (it is still called) but drop the extras branch.
- [ ] `TNEP.py`: delete `_rb_two_body` / `_rb_four_body` / `_rb_five_body`, the `_rb_two_body_scale` constant if only used by them, the Wigner-3j / bispectrum / |c_{n1m}|² blocks in `soap_from_raw` and the `extras` list and its concat, and the validation raises (`l_max ≥ 2` for four-body etc.).
- [ ] `MasterTNEP.py`, `model_io.py`: drop the extras from prints / dim bookkeeping / any saved-shape reasoning.
- [ ] Tests: delete only tests whose sole purpose is these extras (name them); fix `_cfg` pins that set them.
- [ ] Suite green. Commit: `refactor: remove the learned-basis NEP extras (radial_basis_two/four/five_body)`.

## Task 2: remove the bilinear readout

**Files:** `TNEPconfig.py`, `TNEP.py`, `SNES.py`, `data.py`, `adam.py`, `MasterTNEP.py`, `DescriptorBuilderGPU.py`, `DescriptorBuilderGPU_tf.py`, `nep_descriptor.py`, `gap_dipole.py`, `scripts/smoke_train.py`, tests.

- [ ] Config: delete `dipole_pair_rank`; remove `"bilinear"` from the `dipole_readout` doc and from every accepted-value list (`("tnep", "neighbor", "bilinear")` → `("tnep", "neighbor")`, `in ("neighbor", "bilinear")` → `== "neighbor"`).
- [ ] `TNEP.py`: delete `_pair_rank`, `_head_b_start`, `_bilinear_padded`, `_bilinear_coo`, the module-level `_bilinear_padded_xla`, the `nbr_index`/`nbr_G` kwargs on `predict` / `predict_batch` / `predict_batch_candidates` and their pair-term additions. `_split_head` collapses to `s = out` (the head is now S values: `_head_width = _head_slots`, `dipole_head` "vector" → C, "scalar" → 1 broadcast). Keep `dipole_head` and its broadcast.
- [ ] `data.py`: delete `_neighbor_dipole_nbr`, the `nbr_index`/`nbr_G` assembly, their `pad_and_stack` allocation/fill/emit, and their entries in `_RESIDENT_SMALL_KEYS`. `_neighbor_dipole_pairs` STAYS (V uses it).
- [ ] `SNES.py`: drop the b-slice zeroing in `_build_mu_init` (the head has no b-slice now), the `nbr_index`/`nbr_G` in `struct_keys` and in the validate/fit call sites.
- [ ] `adam.py`, `MasterTNEP.py` (label `[T·H·C(1+2K)]` → `[T·H·C]` / `[T·H·1]`, info line), `DescriptorBuilderGPU*.py` comments, `nep_descriptor.py` (any "bilinear" in messages), `gap_dipole.py` (the `_validate` bilinear rejection becomes unnecessary — `dipole_readout` must be `"neighbor"` there already; delete the now-dead check only if it is literally about bilinear), `scripts/smoke_train.py` (`--readout` choices, `--pair-rank`, `--head` stays).
- [ ] Tests: delete the bilinear-only tests and parametrisations (name them); every `readout in ("neighbor", "bilinear")` parametrisation drops to neighbour; `_cfg` pins lose `dipole_pair_rank`.
- [ ] Suite green. Commit: `refactor: remove the bilinear dipole readout (dipole_pair_rank, pair-term kernels, nbr_index/nbr_G plumbing)`.

## Task 3: remove the descriptor-preprocess contraction

**Files:** `TNEPconfig.py`, `TNEP.py`, `SNES.py`, `DescriptorBuilderGPU.py`, `model_io.py`, `MasterTNEP.py`, `adam.py`, `spectroscopy.py`, `nep_descriptor.py`, tests.

- [ ] Config: delete `descriptor_preprocess_contract`, `descriptor_nep4_n_max_out`, `descriptor_preprocess_init`, `preprocess_sigma_scale`, `descriptor_preprocess_angular_l_keep`, `descriptor_preprocess_per_type`, `descriptor_preprocess_lambda_1`, `descriptor_preprocess_lambda_2` and their comments.
- [ ] `DescriptorBuilderGPU.py`: delete `descriptor_preprocess_layout`, `descriptor_post_preprocess_block_layout`, the nep4 fold helpers, and any `dim_q_raw` bookkeeping that exists only for this.
- [ ] `TNEP.py`: delete `descriptor_preprocess_contract` reads, `_preprocess_layout`, `W_pre_angular`, `_preprocess_summed_flat_idx`, `_W0_preprocess_eff`, `_W0_preprocess_eff_nep4`, `dim_q_forward`'s raw-vs-new distinction (it becomes `cfg.dim_q`), `cfg.dim_q_raw`, and every `if self.descriptor_preprocess_contract != "off"` fold at the forward sites and in `score`/`score_from_file`. Every `_W0_eff`-then-`_W0_preprocess_eff` chain collapses to `_W0_eff`.
- [ ] `SNES.py`: delete `n_preprocess`, the W_pre tail (assign / counts / μ-init / labels / reconstruct / split / validate / fit call), `preprocess_sigma_scale` in σ-init, the preprocess λ terms in `compute_regularization`, and `_preprocess_init_scheme` / `_preprocess_per_type` / `_preprocess_Q_raw`.
- [ ] `model_io.py`: delete the `W_pre_angular` save/load and its guards. **Old saves that contain a `W_pre_angular` dataset must fail loudly** with a message saying the preprocess contraction was removed — do not silently ignore the dataset.
- [ ] `adam.py`, `MasterTNEP.py` (compression-ratio print), `spectroscopy.py` (the W0 preprocess fold in the fused predictor), `nep_descriptor.py` (the concat+preprocess and replace+preprocess guards go away entirely).
- [ ] Tests: delete preprocess-only tests (name them); fix `_cfg` pins.
- [ ] Suite green. Commit: `refactor: remove the descriptor-preprocess contraction (W_pre_angular, nep4 fold, 8 config fields)`.

## Task 4: verification

- [ ] `grep -rn "preprocess\|W_pre\|nep4\|bilinear\|pair_rank\|nbr_index\|nbr_G\|radial_basis_two_body\|radial_basis_four_body\|radial_basis_five_body" --include=*.py .` → only unrelated hits (e.g. `gap_descriptor`'s `two_body` slice key, `nep_four_body`). Report the full list.
- [ ] Full suite green; report the count and which tests were deleted with the reason.
- [ ] Smokes on CPU: `scripts/smoke_train.py --readout neighbor --nep off`, `--readout tnep --nep replace --scale minmax`, `--readout neighbor --nep gap_sum`, `--readout neighbor --nep sum` — all SMOKE OK.
- [ ] The calorine oracle still passes under `/tmp/claude-1000/-home-reece-TNEP/bb5da32c-98ca-4c1a-bbfd-b9106da2eeff/scratchpad/calorine-venv/bin/python -m pytest tests/test_nep_descriptor.py -q -k calorine`.
- [ ] Report the net line count removed (`git diff --stat 3e50c0d..HEAD`).
