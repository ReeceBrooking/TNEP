# Multi-Descriptor TNEP (SOAP norm/standardisation · GPUMD NEP descriptors · summed experts) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) let the fixed SOAP descriptor run un-normalised and standardised; (2) add GPUMD's NEP descriptors (2-body radial, 3-body angular, optional 4/5-body) with GPUMD-exact basis functions, learned species-pair coefficients c and GPUMD's min–max scaler, usable either concatenated with SOAP or replacing it (a faithful GPUMD TNEP); (3) add a summed-experts mode where separate ANNs on the NEP 2-body block, the NEP 3-body block and the SOAP block have their head outputs added before the readout.

**Architecture:** Work lands on a new branch `MultiDescriptor` = ExperimentalDescriptorLearning + merge of ExperimentalDipole (bilinear readout, perf, filename commits). The NEP descriptor is split GPUMD-style into a candidate-independent **raw** part built once at data time from the ase pair list (per-atom sums of Chebyshev basis × real-harmonic polynomials, resolved by neighbour species) and an **in-model** linear contraction with learned c tensors that ride the SNES tail exactly like `M_radial` (and are tape-watched under Adam). `nep_descriptor` ∈ {off, concat, replace, sum} decides how the resulting q enters the ANN(s). Phase A (Tasks 0–9) covers the neighbour and bilinear readouts (no descriptor gradients). Phase B (Tasks 10–12) ports GPUMD's analytic ∂q/∂r so `dipole_readout="tnep"` with `dipole_rij_power=2` reproduces GPUMD's TNEP dipole μ = −Σ_pairs |r_ij|² F_ij.

**Tech Stack:** TensorFlow 2 (eager + tf.function/XLA), NumPy, ase `neighbor_list`, h5py, pytest. Reference source: `/home/reece/GPUMD` (commit fee6340c), files `src/main_nep/nep.cu`, `src/main_nep/parameters.cu`, `src/main_nep/snes.cu`, `src/utilities/nep_utilities.cuh`. Optional exactness oracle: `pip install calorine` (bundles NEP_CPU) with the user's trained `nep.txt` files under `/home/reece/Work/GPUMD Testing/`.

---

## 0. Design (read before any task)

### 0.1 Branch

`git checkout -b MultiDescriptor ExperimentalDescriptorLearning && git merge ExperimentalDipole`. `git merge-tree` reports content conflicts in `SNES.py`, `TNEP.py`, `data.py`, `tests/test_neighbor_dipole.py` (MasterTNEP/TNEPconfig auto-merge). Resolution rules in Task 0.

### 0.2 Item 1 — SOAP normalisation off + standardisation (one knob each, both already half-exist)

- **Normalisation:** `cfg.radial_basis_norm` ("l2" | "none" | "sqrt") already exists but is gated to the learned basis (`TNEP.soap_from_raw`). Drop the gate: the **fixed** GPU builder honours `"none"` by skipping the per-atom division `soap_unnorm / sqrt_dot_p` at `DescriptorBuilderGPU_tf.py` (DL lines 762, 989 and the tiled-gradient twin at 1166; the gradient projection term that subtracts q(q·∂q) must be skipped too). `"sqrt"` stays learned-basis-only (its ∂/∂r is singular at q=0); the fixed builder raises on it. No new field.
- **Standardisation:** `cfg.descriptor_scale` ("off" | "block" | "channel") already exists — per-channel std scale + mean shift folded into W0/b0 at generation 0 (`compute_q_scale`, `TNEP._W0_eff`). Add one mode `"minmax"` = GPUMD's `q_scaler`: m_k = 1/(max_k − min_k) over ALL training atoms, **no centring** (μ = 0) — `nep.cu:311-348` (`find_max_min` takes `min` across training batches and GPUs; over one batch = the global range). Applies to the full vector the ANN(s) see (SOAP, NEP or both); in "sum" mode the fold is applied per expert (§0.4).
  GPUMD computes `q_scaler` ONCE, before training, from descriptors evaluated with **every c ≡ 1** (`fitness.cu:165-175`: `dummy_solution` = `para.initial_para` = 1.0), not from the initial random c. To be faithful, the minmax scaler is computed from q assembled with `c_r = c_a = 1` (Task 7); "block"/"channel" keep using the model's actual gen-0 q.

### 0.3 Item 2 — GPUMD NEP descriptors, exactly

Config (defaults = GPUMD's, `parameters.cu:83-90, 118-119`):

```python
nep_descriptor: str = "off"          # "off" | "concat" | "replace" | "sum"
nep_n_max_radial: int = 6            # q_n, n = 0..n_max
nep_n_max_angular: int = 6
nep_basis_size_radial: int = 6       # Chebyshev k = 0..basis_size
nep_basis_size_angular: int = 6
nep_l_max: int = 4                   # GPUMD: "the only supported value"; validate == 4
nep_four_body: bool = True           # GPUMD L_max_4body = 2 default → q_n222
nep_five_body: bool = False          # GPUMD L_max_5body = 0 default → q_n1111; requires nep_four_body (find_q reaches the 5-body branch only via num_L ≥ L_max+2) — validate at init
nep_rc_radial: float = 8.0
nep_rc_angular: float = 4.0
nep_num_neurons: int = 30            # "sum" mode only: width of ANN_2b and ANN_3b
```

**Basis (`nep_utilities.cuh:339-347, 500-517`):** with rc the pair cutoff (GPUMD averages per-type cutoffs; we use one value per block),
- f_c(r) = ½(1 + cos(π r/rc)) for r < rc, else 0
- x = 2 (r/rc − 1)² − 1 (NEP's squared warp), Chebyshev T_k(x) by recursion
- f_0(r) = f_c(r); f_k(r) = ½ (T_k(x) + 1) f_c(r), k = 1..basis_size

**Radial block (`nep.cu:110-155`):** q_n^i = Σ_j g_n(r_ij), g_n(r_ij) = Σ_k c_r[t_i, t_j, n, k] f_k(r_ij; rc_r). Learned c_r [T, T, n_max_r+1, K_r+1] (GPUMD index `(n*(K+1)+k)*T² + t1*T + t2`).

**Angular block (`nep.cu:157-215`, `accumulate_s_one<L>` at `nep_utilities.cuh:846-895`, `find_q_one<L>` at 930-942 (`calculate_s_one<L>` at 650-664 is its derivative twin) with `C3B` constants at 19-39, the `Z_COEFFICIENT_L` tables BELOW them (lines 48+), `complex_product`):**
- s^i_{n,·} = Σ_j g_n(r_ij; c_a, rc_a) · P_L(r̂_ij) where P_L is GPUMD's 2L+1 real polynomial components for L = 1..4 (24 components, s index L²−1 … L²+2L−1; port `accumulate_s_one<1..4>` and its coefficient tables VERBATIM, including the `complex_product` iteration of (x+iy)^{n1})
- q^i_{nL} = C3B[L²−1]·(s_{n,L²−1})² + 2·Σ_{k=1}^{2L} C3B[L²−1+k]·(s_{n,L²−1+k})² (`find_q_one<L>`, lines 930-942: the m≠0 components carry a factor 2 — do NOT drop it; `calculate_s_one` mirrors it with `Fp_factor = 2·Fp` for k=0 and ×2 again for k≥1)
- 4-body q_{n222} and 5-body q_{n1111} from the same s per `find_q` at `nep_utilities.cuh:944-1000` (C4B, C5B constants at 40-46) — port verbatim
- Descriptor order = GPUMD's: [q_0..q_{n_max_r}] ‖ angular l-major (`ln = l*(n_max_a+1)+n`, l = 0..3 for L = 1..4) ‖ 4-body (n_max_a+1) ‖ 5-body (n_max_a+1). dim_nep = (n_max_r+1) + (n_max_a+1)·(4 + four_body + five_body) (`parameters.cu:203-211`).

**Split into raw (data time) and contraction (in-model), so c is learnable:**

```
raw_r[i, t', k]        = Σ_{j∈t', r<rc_r} f_k(r_ij; rc_r)                    [N, T, K_r+1]
raw_a[i, t', k, c]     = Σ_{j∈t', r<rc_a} f_k(r_ij; rc_a) · P_c(r̂_ij)        [N, T, K_a+1, 24]
q_r[i, n]              = Σ_{t',k} c_r[t_i, t', n, k] · raw_r[i, t', k]
s[i, n, c]             = Σ_{t',k} c_a[t_i, t', n, k] · raw_a[i, t', k, c]
q_a[i, n, L], q_4b, q_5b = GPUMD quadratic/cubic/quartic forms of s (find_q)
```

`nep_raw` = flatten([raw_r ‖ raw_a]) per atom, width W_raw = T·(K_r+1) + T·(K_a+1)·24 (T=2, K=6: 14 + 336 = 350 floats/atom). Rides the dense per-structure keys next to `V_atom`/`nbr_G`.

**Initialisation (`snes.cu:107-115`):** GPUMD draws EVERY variable, c included, uniform in (−1, 1) with σ₀. We init c_r, c_a uniform(−1, 1) (faithful) and keep our Glorot ANN init. c is excluded from L1/L2 like `M_radial` (GPUMD regularises all variables; note the deviation in the info line).

**Scaler:** `descriptor_scale="minmax"` computed once before training from q assembled with c ≡ 1 (GPUMD `fitness.cu:165-175`), frozen. Under `radial_basis_size` (learned SOAP basis) the SOAP part uses its own gen-0 M0 (exact fixed basis) — consistent with the same "initial parameters" rule.

**Modes:**
- `"concat"`: q = [q_soap ‖ q_nep]; dim_q = dim_soap + dim_nep. The NEP block is an appended extra outside the SOAP species-pair blocks (like the learned-basis extras): excluded from descriptor mixing / preprocess contraction; `q_scale_groups` gets one group per NEP sub-block (radial, each L, 4b, 5b).
- `"replace"`: q = q_nep; dim_q = dim_nep; `descriptor_mixing` and `descriptor_preprocess_contract` must be off (raise). The SOAP builder still runs at data time (cached, one-off) and its output is dropped in-model — ponytail: the faithful path pays one SOAP build; skipping the builder is a follow-up if trajectory inference needs it.
- `"sum"` (item 3, §0.4).
- `"off"`: bit-identical to today.

**One assembly helper:** `TNEP._ann_inputs(descriptors, nep_raw, Z, c_r, c_a) -> tf.Tensor` returns the [.., dim_q] the ANN consumes, called at the three forward sites right where `soap_from_raw` is called today (DL `TNEP.py:1021, 1595, 1760`). This is the single place that knows about modes.

### 0.4 Item 3 — summed experts (`nep_descriptor="sum"`)

Three per-type single-hidden-layer ANNs, outputs (full head width `_head_width`) added:

```
out_i = ANN_soap(q_soap; W0,b0,W1) + ANN_2b(q_r; W0_r,b0_r,W1_r) + ANN_3b(q_ang; W0_a,b0_a,W1_a)
```

then `_split_head(out)` as today. q_r = radial block (n_max_r+1 wide), q_ang = angular (+4b/5b) block. Widths `nep_num_neurons`. In "sum" mode `compute_dim_q` = dim_soap (the main ANN's input); the experts' input widths come from `nep_slices(cfg)`. **descriptor_scale in sum mode:** `compute_q_scale` runs on the full `[q_soap ‖ q_nep]` vector; the SOAP slice folds into W0/b0 as today and the radial / angular slices fold into `W0_r/b0_r` and `W0_a/b0_a` through the same `_W0_eff`-style fold (three folds, one helper). `cfg.q_scale`/`q_shift` are stored at the full width. **Gen-0 identity:** W1_r, W1_a zero-initialised in μ so generation 0 equals the SOAP-only model. SNES layout: the two extra ANNs sit INSIDE the typed slab, before b1: `[W0|b0|W1 | W0_r|b0_r|W1_r | W0_a|b0_a|W1_a | b1 | c_pair | ...]` so `n_typed` stays one contiguous slice (`_global_typed` slices `pv[:, :n_typed]`, DL `SNES.py:407-418`). Typed blocks labelled by centre type, included in per-type L1/L2; mirror every `W0_pol` site for the variables and every `n_typed`-arithmetic site for the offsets. Adam: added to the trainable list. b1: shared global bias only (each expert's own bias would be redundant).

### 0.5 Phase B — descriptor gradients for the faithful TNEP readout

`dipole_readout="tnep"`, `dipole_rij_power=2` already implements μ = −Σ_pairs |r_ij|² F_ij (GPUMD `tnep.cu:576-580`). It needs ∂q/∂r_ij per pair. With learned c the gradient is candidate-dependent, so the static per-pair tensors are the basis derivatives and the c-contraction happens in-model:

```
dfr[p, 3, k]      = ∂f_k(r_ij; rc_r)/∂r_j   = f_k'(r) r̂                       (find_fn_and_fnp, nep_utilities.cuh:519-540)
dfa[p, 3, k, c]   = ∂[f_k(r; rc_a) P_c(r̂)]/∂r_j = f_k' P_c r̂ + f_k ∇P_c        (accumulate_f12_one<L>, nep_utilities.cuh:665-760)
∂q_r[i,n]/∂r_j    = Σ_k c_r[t_i,t_j,n,k] dfr[p,:,k]
∂q_a[i,n,L]/∂r_j  = Σ_c w_c C3B[c] s[i,n,c] Σ_k c_a[t_i,t_j,n,k] dfa[p,:,k,c],  w_c = 2 for the m=0 component of each L, 4 for the others   (+ 4b/5b chain terms from find_q's polynomials; cf. calculate_s_one, nep_utilities.cuh:650-664)
F_ij              = Σ_d de_dq[i,d] ∂q_d/∂r_j   →  μ = −Σ_pairs |r_ij|² F_ij
```

Per candidate: contract de_dq [Cc, A, dim_nep] with c and s first → per-(atom, neighbour-type) tensor [Cc, A, T, (K+1)·25], gather per pair, dot with the static [P, 3, (K+1)·25]. Same shape class as the old tnep path's [Cc, P, Q]. Pairs at rc_r = 8 Å need the MIC/image guard (`_neighbor_dipole_guards` pattern): with L ≈ 12 Å the 8 Å list has multiple images — the ase image list is exact, the in-graph MIC forms are NOT; Phase B keeps the pair stream from ase with explicit image offsets.

### 0.6 Out of scope

- NEP's own ANN details beyond one hidden layer (GPUMD's `num_hidden_layers=2` maps onto the separate depth plan).
- Per-type cutoffs (GPUMD averages `rc[t1], rc[t2]`); one rc per block here.
- ZBL, charge modes, temperature dimension.
- Trajectory inference (spectroscopy) for NEP descriptors — needs the raw builder in-graph; follow-up.

### 0.7 Files

| File | Change |
|---|---|
| `TNEPconfig.py` | `nep_*` fields; `descriptor_scale` gains `"minmax"`; `radial_basis_norm` docs |
| `DescriptorBuilderGPU_tf.py` | fixed builder honours `radial_basis_norm="none"` (3 sites) |
| `DescriptorBuilderGPU.py` | `compute_q_scale` minmax; `nep_dim(cfg)`, `nep_raw_width(cfg)`; `descriptor_input_width` / `compute_dim_q` / `descriptor_block_layout` / `q_scale_groups` aware of the NEP block |
| `nep_descriptor.py` (new) | GPUMD-exact basis + real-harmonic tables + `find_q` forms in NumPy AND TF (one module, two twins); raw builder `nep_raw_numpy(atoms, types, cfg)` |
| `data.py` | `nep_raw` dense key (assembly, `pad_and_stack`, every `SMALL_KEYS` tuple) |
| `TNEP.py` | `c_r`, `c_a` variables; `nep_from_raw`; `_ann_inputs`; sum-mode ANNs + forward additions at 3 sites |
| `SNES.py` | c tail after `M_radial` (all `M_radial` sites); sum-mode ANN blocks (all `W0_pol` sites); thread `nep_raw` |
| `adam.py` | c_r/c_a tape-watched; sum-mode weights trainable; `nep_raw` threaded |
| `model_io.py` | save/load c_r, c_a, sum-mode weights |
| `MasterTNEP.py` | gen-0 minmax scaler; breakdown rows; info line |
| `tests/test_nep_descriptor.py` (new) | exactness, invariance, ordering, tails, gen-0 identities |

---

## Task 0: Branch and merge

**Files:** git only (conflicts in `SNES.py`, `TNEP.py`, `data.py`, `tests/test_neighbor_dipole.py`)

- [ ] **Step 1: Create the branch and merge**

```bash
git checkout -b MultiDescriptor ExperimentalDescriptorLearning
git merge ExperimentalDipole      # expect conflicts in the four files above
```

- [ ] **Step 2: Resolve, keeping BOTH sides' features.** Known overlaps:
  - `SNES.__init__`: DL adds `self.n_radial` (M_radial tail); Dipole adds `self._Hw` and the `_Ch`/`_Hw` split. Keep both; `n_W1` uses `_Hw`.
  - `SNES._set_model_params` / `_split_reconstructed` / `reconstruct_params_tf` / `_build_mu_init` / `_build_type_of_variable`: tail order is `W0,b0,W1,b1 | c_pair | pol | U_pair | W_pre_angular | M_radial`. Dipole's b-slice zeroing goes inside `_fill_ann`; DL's `M_radial` fill stays at the end.
  - `SNES.validate` / fit `predict_batch_candidates` call: pass BOTH `M_radial=` and `nbr_index=/nbr_G=`.
  - `TNEP.predict_batch` / `predict_batch_candidates` / `predict`: DL inserts `soap_from_raw` before the hidden layer; Dipole changes the neighbour branch (`_split_head`, `_bilinear_*`). Both apply, at different places in each function.
  - `data.py`: DL's `_neighbor_dipole_V` perf rewrite (direct MIC search, commit 01c5202) vs Dipole's `_neighbor_dipole_pairs` refactor — keep DL's fast pair search as the body of `_neighbor_dipole_pairs` (it must still return `(i, j, G)`), and Dipole's `_neighbor_dipole_nbr`. `SMALL_KEYS` tuples: union.
  - `tests/test_neighbor_dipole.py`: union of tests; `_cfg` pins the union of fields (`radial_basis_size=None`, `dipole_pair_rank=2`, `descriptor_scale="off"`, ...).

- [ ] **Step 3: Run both suites**

Run: `CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q`
Expected: every test that passed on either branch passes here (DL's learned-basis tests + Dipole's 67 neighbour tests + 18 GAP tests). Fix until green — do NOT skip tests to get there.

- [ ] **Step 4: Commit the merge**

```bash
git commit -m "merge: ExperimentalDipole (bilinear readout, perf, filenames) into ExperimentalDescriptorLearning → MultiDescriptor"
```

---

## Task 1: `radial_basis_norm="none"` for the fixed SOAP builder

**Files:**
- Modify: `DescriptorBuilderGPU_tf.py` (normalisation sites ~762, ~989, ~1166 and the gradient projection that uses `sqrt_dot_p`/`sdpp`), `TNEP.py` (drop the `radial_basis_size` gate on `radial_basis_norm`, keep the `"sqrt"` gate)
- Test: `tests/test_nep_descriptor.py` (new file; put item-1 tests here too)

- [ ] **Step 1: Failing tests**

```python
def test_fixed_soap_norm_none_matches_unnormalised():
    cfg_l2 = _cfg(radial_basis_norm="l2"); cfg_no = _cfg(radial_basis_norm="none")
    atoms = _bulk_frame()
    q_l2 = _descriptors(cfg_l2, atoms); q_no = _descriptors(cfg_no, atoms)      # [A, Q]
    n = np.linalg.norm(q_no, axis=1, keepdims=True)
    np.testing.assert_allclose(q_no / n, q_l2, rtol=1e-5, atol=1e-6)            # same direction
    assert np.all(n[:, 0] > 0.9)                                                 # magnitude kept (O has |q| ≫ 1)

def test_fixed_soap_norm_none_gradient_is_unprojected():
    # ∂q/∂r of the un-normalised descriptor by finite difference on one atom vs builder gradients
    cfg = _cfg(radial_basis_norm="none"); atoms = _bulk_frame()
    g = _self_gradients(cfg, atoms)                                               # [A, 3, Q] from build_descriptors_self_only
    g_fd = _fd_self_gradient(cfg, atoms, atom=0, h=1e-4)
    np.testing.assert_allclose(g[0], g_fd, rtol=1e-3, atol=1e-5)

def test_fixed_soap_rejects_sqrt():
    with pytest.raises(ValueError, match="radial_basis_norm"):
        _model(_cfg(radial_basis_norm="sqrt"))
```

- [ ] **Step 2: Run → fail** (`radial_basis_norm` is currently rejected without `radial_basis_size`).
- [ ] **Step 3: Implement.** Builder: thread a `normalize: bool` through `_compute_soap_with_grad_inner_tf` and the tiled twin; when False, `soap_norm = soap_unnorm` and the gradient skips the projection term (grad of q̂ = (I − q̂ q̂ᵀ)/‖q‖ · grad q → just grad q). TNEP: `_rb_norm` read for both builders; raise only for `"sqrt"` without `radial_basis_size`.
- [ ] **Step 4: Run the new tests + `tests/test_neighbor_dipole.py`** — the neighbour suite stays green (default "l2").
- [ ] **Step 5: Commit** `feat: radial_basis_norm="none" for the fixed SOAP builder (skip L2 projection, un-projected gradients)`

---

## Task 2: `descriptor_scale="minmax"` (GPUMD q_scaler)

**Files:** `DescriptorBuilderGPU.py:449-480` (`compute_q_scale`), `TNEPconfig.py` (mode doc), `MasterTNEP.py:~499-517` (already computes at gen 0)

- [ ] **Step 1: Failing test**

```python
def test_q_scale_minmax_is_gpumd_scaler():
    q = np.random.default_rng(0).uniform(-3, 5, (50, 7)); mask = np.ones(50)
    cfg = _cfg(descriptor_scale="minmax")
    m, mu = compute_q_scale(q, mask, cfg)
    np.testing.assert_allclose(m, 1.0 / (q.max(0) - q.min(0)), rtol=1e-6)   # nep.cu:347
    assert np.all(mu == 0.0)                                                # no centring
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** in `compute_q_scale`: `if cfg.descriptor_scale == "minmax": rng = x.max(0) - x.min(0); m = where(rng > 0, 1/rng, 1); return m, zeros`. Accept the mode wherever `descriptor_scale` is validated.
- [ ] **Step 4: Run tests; Step 5: Commit** `feat: descriptor_scale="minmax" — GPUMD's q_scaler (1/(max−min), no centring)`

---

## Task 3: GPUMD-exact NEP basis and raw builder (`nep_descriptor.py`)

**Files:**
- Create: `nep_descriptor.py`
- Test: `tests/test_nep_descriptor.py`

Module contents (NumPy first; the TF twin in Task 4 shares the constant tables):

```python
# Constants ported VERBATIM from /home/reece/GPUMD/src/utilities/nep_utilities.cuh
C3B = np.array([...80 values, lines 19-39...]); C4B = np.array([... 5 values, line 40]); C5B = np.array([..., line 46])
Z_COEFFICIENT = {1: [[..]], 2: [[..]], 3: [[..]], 4: [[..]]}   # the Z_COEFFICIENT_L tables (L ≤ 4 only)

def fc(r, rc): ...                     # ½(1+cos πr/rc) for r<rc else 0     (line 339)
def fn_all(r, rc, K):                  # [P, K+1]: f_0 = fc, f_k = ½(T_k(x)+1) fc, x = 2(r/rc−1)²−1   (line 500-517)
def real_harmonics(xhat, yhat, zhat):  # [P, 24]: GPUMD's P_c for L=1..4 in s-index order L²−1.., via accumulate_s_one<L> (846-895)
def find_q(s, four_body, five_body):   # s [.., n, 24] → q_a [.., n, 4 (+1) (+1)] per find_q / find_q_one (944-1000, 930-942)
def nep_raw_numpy(atoms, types_int, cfg) -> np.ndarray   # [N, W_raw] = flatten(raw_r [N,T,Kr+1] ‖ raw_a [N,T,Ka+1,24])
def nep_dim(cfg) -> int; def nep_raw_width(cfg) -> int; def nep_slices(cfg) -> dict   # radial / angular / 4b / 5b column ranges
```

- [ ] **Step 1: Failing tests**

```python
def test_chebyshev_basis_matches_recursion_reference():
    r = np.linspace(0.1, 7.9, 200); rc, K = 8.0, 6
    f = fn_all(r, rc, K)
    x = 2*(r/rc - 1)**2 - 1; fc_ = 0.5*(1 + np.cos(np.pi*r/rc))
    ref = np.stack([fc_] + [0.5*(np.polynomial.chebyshev.Chebyshev.basis(k)(x) + 1)*fc_ for k in range(1, K+1)], 1)
    np.testing.assert_allclose(f, ref, rtol=1e-6, atol=1e-9)
    assert np.all(fn_all(np.array([8.0, 9.0]), rc, K) == 0.0)

def test_real_harmonics_rotation_invariance_of_q():
    # Σ_c C3B[c] P_c(r̂)² summed over a random neighbour cloud must be rotation invariant.
    rng = np.random.default_rng(1); v = rng.standard_normal((30, 3)); w = rng.uniform(0.2, 1, 30)
    def q_of(v):
        s = (real_harmonics(*(v / np.linalg.norm(v, axis=1, keepdims=True)).T) * w[:, None]).sum(0)
        return find_q(s[None, None], True, True)[0, 0]
    R = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    np.testing.assert_allclose(q_of(v @ R.T), q_of(v), rtol=1e-6, atol=1e-8)

def test_nep_raw_width_and_shape():
    cfg = _cfg(nep_descriptor="concat")
    atoms = _bulk_frame(); raw = nep_raw_numpy(atoms, _types(atoms, cfg), cfg)
    T, Kr, Ka = 2, cfg.nep_basis_size_radial, cfg.nep_basis_size_angular
    assert raw.shape == (len(atoms), T*(Kr+1) + T*(Ka+1)*24)
    assert nep_dim(cfg) == (cfg.nep_n_max_radial+1) + (cfg.nep_n_max_angular+1)*(4 + 1 + 0)   # four_body default on

@pytest.mark.skipif(not _has_calorine(), reason="pip install calorine for the GPUMD oracle")
def test_nep_descriptor_matches_calorine_on_users_nep_txt():
    # Parse /home/reece/Work/GPUMD Testing/dipole_bulk_water/nep.txt (c tensors + q_scaler),
    # build raw with our cfg mirroring its header (n_max, basis_size, rc), contract with ITS c,
    # apply ITS q_scaler, compare with calorine.nep.get_descriptors(atoms, nep_txt). atol 1e-5.
    ...
```

(The calorine test is the only true exactness oracle. `calorine` is NOT installed in this env; ask the human to `pip install calorine` BEFORE starting Task 3 so the oracle runs during the port, not after. If they decline, say so in the Task 3 commit message: exactness then rests on transcription + the invariance tests.)

- [ ] **Step 2: Run → fail (module missing).**
- [ ] **Step 3: Implement** by transcription from the cited lines. `real_harmonics` reproduces `accumulate_s_one<L>` for L=1..4 exactly: for each L, `s_index = L²−1`; loop n1 = 0..L, n2 from `(L+n1)%2` to L−n1 step 2 accumulating `Z_COEFFICIENT_L[n1][n2]·z^n2`; n1 = 0 → one component; n1 ≥ 1 → two components (×Re, ×Im of (x+iy)^{n1}). `nep_raw_numpy` uses `ase.neighborlist.neighbor_list("ijD", atoms, max(rc_r, rc_a))` once, filters per block cutoff, `np.add.at` per (i, t_j).
- [ ] **Step 4: Run tests. Step 5: Commit** `feat: nep_descriptor.py — GPUMD-exact Chebyshev basis, real-harmonic tables, find_q forms, raw builder`

---

## Task 4: In-model contraction `TNEP.nep_from_raw` + c variables + dim bookkeeping

**Files:** `TNEP.py` (init: `c_r`, `c_a`, validation; `nep_from_raw`; `_ann_inputs`), `nep_descriptor.py` (TF twin `find_q_tf`, constants as `tf.constant`), `DescriptorBuilderGPU.py` (`compute_dim_q`: += nep_dim under "concat", == nep_dim under "replace", == dim_soap under "sum" (main ANN input); `descriptor_block_layout`: NEP block is an appended extra under "concat", and under "replace" the layout has NO SOAP blocks (mixing is rejected there anyway) — `q_scale_groups` must then return NEP sub-groups only; `descriptor_input_width` unchanged — it is the builder's SOAP width, never dim_q). DL `MasterTNEP.py:427` sets `cfg.dim_q = compute_dim_q(cfg)` and DL `TNEP.py:222` checks `q_scale` against dim_q, so both must see the mode-correct dim., `TNEPconfig.py` (fields from §0.3)

- [ ] **Step 1: Failing tests**

```python
def test_nep_from_raw_matches_numpy_contraction():
    cfg = _cfg(nep_descriptor="concat"); m = _model(cfg); atoms = _bulk_frame()
    raw = nep_raw_numpy(atoms, _types(atoms, cfg), cfg); Z = _types(atoms, cfg)
    q_tf = m.nep_from_raw(tf.constant(raw[None]), tf.constant(Z[None]), m.c_r, m.c_a)[0].numpy()
    q_np = _nep_contract_numpy(raw, Z, m.c_r.numpy(), m.c_a.numpy(), cfg)   # straightforward loops
    np.testing.assert_allclose(q_tf, q_np, rtol=1e-5, atol=1e-6)
    assert q_tf.shape == (len(atoms), nep_dim(cfg))

def test_nep_descriptor_ordering_is_gpumd():
    # radial first, then angular l-major, then 4-body: zero c_a ⇒ angular/4b columns are exactly 0 and radial unchanged.
    ...

def test_nep_c_init_uniform_pm1_and_shapes():
    cfg = _cfg(nep_descriptor="concat"); m = _model(cfg)
    T, nr, Kr, na, Ka = 2, cfg.nep_n_max_radial, cfg.nep_basis_size_radial, cfg.nep_n_max_angular, cfg.nep_basis_size_angular
    assert tuple(m.c_r.shape) == (T, T, nr+1, Kr+1) and tuple(m.c_a.shape) == (T, T, na+1, Ka+1)
    for c in (m.c_r, m.c_a): v = c.numpy(); assert v.min() >= -1 and v.max() <= 1 and v.std() > 0.4   # uniform(−1,1): std 0.577

def test_ann_inputs_modes():
    atoms = _bulk_frame()
    for mode, width in (("off", "dim_soap"), ("concat", "dim_soap+nep"), ("replace", "nep")):
        cfg = _cfg(nep_descriptor=mode); m = _model(cfg)
        x = m._ann_inputs(desc, nep_raw, Z, m.c_r, m.c_a)      # shapes per mode; "off" bit-identical to desc
    with pytest.raises(ValueError, match="descriptor_mixing"):
        _model(_cfg(nep_descriptor="replace", descriptor_mixing=True))
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** `nep_from_raw(raw [.., A, W_raw], Z [.., A], c_r, c_a)`: split raw into raw_r [.., A, T, Kr+1] and raw_a [.., A, T, Ka+1, 24]; gather c by centre type `tf.gather(c_r, Z)` → [.., A, T, n, k]; `q_r = einsum('...atk,...atnk->...an')`; `s = einsum('...atkc,...atnk->...anc')`; `q_a = find_q_tf(s)`; concat in GPUMD order. `c_r`, `c_a` are `tf.Variable(trainable=False)` like `M_radial` (SNES assigns; Adam tape-watches). `_ann_inputs` implements §0.3 modes; `"replace"` raises at init if mixing/preprocess on; `nep_l_max != 4` raises.
- [ ] **Step 4: Run tests. Step 5: Commit** `feat: nep_from_raw — learned species-pair c contraction, GPUMD descriptor order; nep_descriptor modes off/concat/replace`

---

## Task 5: Data plumbing for `nep_raw` and the three forward sites

**Files:** `data.py` (assembly next to `V_atom`; `pad_and_stack` `[S, A_max, W_raw]`; EVERY `SMALL_KEYS` tuple + `_RESIDENT_SMALL_KEYS` — grep `"V_atom"` rather than counting), `TNEP.py` (`predict`, `predict_batch`, `predict_batch_candidates` + `score` chunk call: `nep_raw=` kwarg → `_ann_inputs`), `SNES.py` (`struct_keys`, validate/fit calls), `adam.py` (call site)

- [ ] **Step 1: Failing test** — four-path consistency for NEP modes:

```python
@pytest.mark.parametrize("mode", ["concat", "replace"])
def test_nep_four_path_consistency(mode):
    # predict / predict_batch(padded) / predict_batch_candidates(Cc=1) agree, neighbor readout, K=0 and K=2.
    ...
```

- [ ] **Step 2: Run → fail (`nep_raw` kwarg unknown).**
- [ ] **Step 3: Implement** by copying the `nbr_index`/`nbr_G` pattern exactly (Task 2 and 4 of `neighbor_dipole_bilinear_plan.md`): assembly gated on `cfg.nep_descriptor != "off"`; `pad_and_stack` fills `[s, :N_s]`; keys added to every tuple; `SNES.struct_keys` loop; all `predict_batch`/`predict_batch_candidates` call sites pass `nep_raw=chunk.get("nep_raw")`. In `predict` (single structure) the raw is passed in too (`nep_raw` arg) — the padded [A, M] table there does not reach 8 Å.
- [ ] **Step 4: Run `tests/test_nep_descriptor.py tests/test_neighbor_dipole.py`. Step 5: Commit** `feat: nep_raw rides the dense per-structure keys; NEP inputs assembled at all forward sites`

---

## Task 6: SNES + Adam tail for c_r, c_a; save/load; breakdown

**Files:** `SNES.py` (every `M_radial` site: `_set_model_params:61`, counts `319-323`, `_build_mu_init:533`, `_build_type_of_variable:706-710`, validate `1307-1333, 1398`, `_split_reconstructed:1439-1460`, `reconstruct_params_tf`, fit call `2061`), `adam.py:44-51, 110-119`, `model_io.py` (save `255`, `_load_weights:444-507`), `MasterTNEP.py` (breakdown rows, info line)

- [ ] **Step 1: Failing tests**

```python
def test_nep_c_tail_roundtrip_snes():
    cfg = _cfg(nep_descriptor="concat", dipole_readout="bilinear", dipole_pair_rank=1); m = _model(cfg); opt = SNES(m)
    n_c = np.prod(m.c_r.shape) + np.prod(m.c_a.shape)
    assert opt.dim == opt.n_anns_total + opt.n_U_pair + opt.n_preprocess + opt.n_radial + n_c
    named = opt._split_reconstructed(opt.reconstruct_params_tf(opt.mu))
    np.testing.assert_array_equal(named["c_r"].numpy(), m.c_r.numpy()); ...   # μ-init carries the model's uniform(−1,1) draw
    # candidates(mu) == predict_batch after _set_model_params (both with nep_raw) — same pattern as the bilinear round-trip test
    ...
def test_nep_c_per_type_labels():
    # type_of_variable for the c tail is centre-type-major (t1 axis) like M_radial
def test_save_load_roundtrip_nep(tmp_path): ...   # c_r, c_a, mode, q_scale round-trip; μ identical
def test_adam_trains_c(): ...                      # two Adam steps change c_r (tape-watched), loss finite — DL branch pattern from tests of fit_adam
```

- [ ] **Step 2: Run → fail. Step 3: Implement** by mirroring each `M_radial` site with `c_r` then `c_a` appended AFTER `M_radial` (tail order: `... | M_radial | c_r | c_a`). μ-init: copy the model's variables (already uniform(−1,1)). Labels: centre-type-major. Excluded from L1/L2 (single-type branch already slices `pv[:n_anns_total]`). Breakdown rows `NEP c_r [T·T·(n+1)·(K+1)]`, `NEP c_a [...]`; info line prints mode, dims, cutoffs and "c init U(−1,1) per GPUMD; c excluded from L1/L2 (GPUMD includes it)".
- [ ] **Step 4: Run all tests. Step 5: Commit** `feat: NEP c_r/c_a in the SNES tail (after M_radial) and Adam; save/load; breakdown`

---

## Task 7: Scaler + mixing interplay for the NEP block

**Files:** `DescriptorBuilderGPU.py` (`q_scale_groups`, `descriptor_block_layout` extras), `MasterTNEP.py` (gen-0 q from `_ann_inputs`, not the builder output)

- [ ] **Step 1: Failing tests**

```python
def test_q_scale_groups_cover_nep_block_once():
    cfg = _cfg(nep_descriptor="concat", descriptor_scale="block")
    groups = q_scale_groups(cfg); flat = np.concatenate(groups)
    assert sorted(flat) == list(range(compute_dim_q(cfg)))                      # each channel exactly once
    assert len(groups) == n_soap_groups + 1 + 4 + 1                              # radial, L=1..4, 4-body
    cfg_r = _cfg(nep_descriptor="replace", descriptor_scale="block")
    assert len(q_scale_groups(cfg_r)) == 1 + 4 + 1                              # no SOAP groups under replace
def test_minmax_scaler_uses_c_equal_one():
    # MasterTNEP's scaler input must be the concatenated/replaced vector assembled with c_r = c_a = 1
    # (GPUMD fitness.cu:165-175), not the SOAP builder output and not the random initial c.
    # "block"/"channel" use the model's actual gen-0 q (assert the two differ for a random c).
def test_nep_block_excluded_from_mixing():
    cfg = _cfg(nep_descriptor="concat", descriptor_mixing=True); m = _model(cfg)
    # U_pair acts on SOAP columns only: W0_eff rows for NEP columns == W0 rows
```

- [ ] **Step 2-5:** implement (NEP block = one more "extra" family in `radial_extra_count`-style bookkeeping: generalise to `extra_block_widths(cfg) -> list[int]`), run, commit `feat: NEP block in descriptor_scale groups and outside mixing; gen-0 scaler from the assembled ANN input`

---

## Task 8: Summed experts (`nep_descriptor="sum"`)

**Files:** `TNEP.py` (variables `W0_r,b0_r,W1_r,W0_a,b0_a,W1_a`; 3 forward sites add the two expert outputs before `_split_head`), `SNES.py` (every `W0_pol` site: counts, `_extract_type_params` (+batched), `_build_type_of_variable`, `_build_mu_init` (W1_r/W1_a zero), `reconstruct_params_tf`, `_split_reconstructed`, `_set_model_params`, validate/fit calls), `adam.py` (trainables), `model_io.py`, `MasterTNEP.py` (rows)

- [ ] **Step 1: Failing tests**

```python
def test_sum_mode_gen0_identity():
    # sum-mode μ-init with W1_r = W1_a = 0 gives μ identical to nep_descriptor="off" with the same SOAP-ANN weights
def test_sum_mode_experts_add():
    # zero the SOAP ANN's W1 → μ equals ANN_2b + ANN_3b alone; zero W1_a → equals ANN_2b alone (linearity of the head)
def test_sum_mode_four_path(): ...         # predict / batch / candidates agree with K=2
def test_sum_mode_snes_layout_and_roundtrip(): ...   # n_typed includes both experts; per-type extract length; candidates(mu)==assign+predict_batch
def test_sum_mode_save_load(tmp_path): ...
```

- [ ] **Step 2: Run → fail. Step 3: Implement.** Expert inputs come from `nep_slices(cfg)` applied to `nep_from_raw` output (radial slice → ANN_2b, angular+4b+5b slice → ANN_3b); in "sum" mode `_ann_inputs` returns the SOAP block only for the main ANN and the forwards call `_expert_outputs(q_nep, Z, ...)` to add `[.., A, head_width]`. Per-type loop pattern of `predict_batch` (type masks) for the batched site; batched-GEMM pattern for candidates. The per-expert descriptor-scale fold reuses `_W0_eff` (one helper taking (W0, b0, scale slice, shift slice)); do not write three copies. SNES layout: `[W0|b0|W1 | W0_r|b0_r|W1_r | W0_a|b0_a|W1_a | b1] | c_pair | pol | ...` — n_typed grows and stays contiguous; `_extract_type_params` concatenates the type-t slabs of all three ANNs (per-type L1/L2 denominator `_n_per_type` updated).
- [ ] **Step 4: Run all tests. Step 5: Commit** `feat: nep_descriptor="sum" — separate 2-body / 3-body / SOAP experts summed into the head; gen-0 identical to SOAP-only`

---

## Task 9: Smokes (short; no A/B)

- [ ] **Step 1:** 8-generation SNES on 60 bulk-water frames for each of: `concat` (neighbor readout), `replace` + `descriptor_scale="minmax"` (neighbor readout), `sum` (bilinear K=1), each printing the breakdown and finishing with a saved model. Add `scripts/smoke_train.py` (readout, nep mode, K, generations, frames as CLI args; save_path under a temp dir) so smokes are reproducible from the repo.
- [ ] **Step 2:** Adam smoke (`optimizer="adam"`, 20 steps) for `concat` and `sum` — loss decreases, c_r changes.
- [ ] **Step 3:** If `calorine` is installable in the env, run `test_nep_descriptor_matches_calorine_on_users_nep_txt` and record the max abs error in the commit message; otherwise state plainly that GPUMD exactness rests on the transcription + invariance tests.
- [ ] **Step 4:** Method note in `neighbor_dipole_method.md` (new §2.5 "ANN inputs: SOAP / NEP / sum") + memory update. Commit `docs: multi-descriptor inputs`.

---

## Phase B — faithful GPUMD TNEP readout (descriptor gradients)

### Task 10: Per-pair basis derivatives (numpy) + FD tests

**Files:** `nep_descriptor.py` (`fn_and_fnp`, `real_harmonics_grad` porting `accumulate_f12_one<L>` at `nep_utilities.cuh:665-760`), `data.py` (COO stream: `nep_pair_atom/gidx/struct/offset`, `nep_dfr [P,3,Kr+1]`, `nep_dfa [P,3,Ka+1,24]`, `nep_pair_t`), tests: FD of `raw_r`, `raw_a` w.r.t. one atom's position vs the summed per-pair derivatives (both signs: centre and neighbour).

### Task 11: In-model per-candidate force contraction for `dipole_readout="tnep"`

**Files:** `TNEP.py` (`_nep_pair_forces(de_dq_nep, s, c_r, c_a, Z, pair tensors)` per §0.5, used by `predict_batch` and `predict_batch_candidates` in the tnep branch; the SOAP part keeps the existing `grad_values` path and the two force streams add), `SNES.py`/`adam.py` (thread the pair tensors). Tests: μ from analytic pair forces vs finite-difference ∂U/∂r summed with |r_ij|² weights on a 3-molecule cell (`dipole_rij_power=2`), for `replace` mode with and without 4-body; K=0 (fixed c) and random c.

### Task 12: Faithful GPUMD TNEP smoke

`nep_descriptor="replace"`, `descriptor_scale="minmax"`, `dipole_readout="tnep"`, `dipole_rij_power=2`, tanh, 30 neurons, `init_sigma` 0.1 (GPUMD's σ₀), 8 generations on 60 frames. Print the parameter count and check it equals GPUMD's `number_of_variables` formula (`parameters.cu:216-235`) for the same header (ANN per type (dim+2)·N + 1 shared bias, + c count) up to our c_pair/b-slice absences.

---

## Validation the plan does NOT do

Per the standing rule (tests + short smokes only), no long SNES runs. The comparisons the user wants — SOAP vs NEP-replace vs concat vs sum — are 300-generation arms at `init_sigma=0.02` (see the σ finding of 2026-09-28), run by the user.
