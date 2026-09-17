# Learned SOAP-turbo Radial Basis — Brainstorm & Design

Date: 2026-09-17
Branch: ExperimentalDescriptorLearning

## 1. Problem

NEP (Fan et al., GPUMD) trains the radial descriptor functions together with
the network: each radial channel is a learned combination of a fixed basis,

    g_n(r) = Σ_k c_nk^{(t_i, t_j)} f_k(r),     n ≤ n_max, k ≤ basis_size

and the `c_nk` sit in the SNES parameter vector next to the ANN weights
(`GPUMD/src/main_nep/nep.cu:141-149`, `parameters.cu:231-235`). That is the
reason NEP uses SNES at all: it lets the descriptor itself be optimised without
back-propagating through the descriptor code.

TNEP swapped the NEP basis for SOAP-turbo but kept the descriptor fixed: it is
built once by quippy / `DescriptorBuilderGPU_tf.py`, and SNES only ever sees
the ANN (plus post-hoc linear "mixing" of the fixed descriptor). This design
brings the learnable radial basis back, inside the GPU SOAP-turbo builder.

## 2. Where the learnable coefficients go in SOAP-turbo

The SOAP-turbo pipeline (`soap_turbo.f90`, ported in `DescriptorBuilderGPU_tf.py`)
for centre `i`, neighbour `j` of species `s`:

1. Radial expansion: `r_n(j) = A_j · Σ_κ W[n,κ] I_κ(j)`, with `I_κ` the analytic
   Gaussian-smeared poly3 integrals, `W = S^{-1/2}` the fixed orthonormalisation.
2. Angular: `y_k(j)`, k = (l, m) flat.
3. Centre coefficients: `c_{n k}(i) = 4π Σ_{j∈i} r_n(j) y_k(j)`, species-blocked
   in n (n_max = S·α).
4. Power spectrum: `p_{n n' l}(i) = Σ_m mult · Re(c_{n l m} c*_{n' l m})`.
5. Trivial compression (keep channels with n or n' a species pivot), then
   per-atom L2 normalisation.

Steps 1–3 are **linear** in the radial functions. So if the radial functions
become learned combinations of a larger orthonormal basis, everything up to the
power spectrum can be pre-computed once and contracted per candidate:

    c_raw[i, s, κ, k] = 4π Σ_{j∈i, species(j)=s} A_j · φ_κ(r_j) · y_k(j)      κ ≤ K
    c[i, (s,n), k]    = Σ_κ M[t_i, s, n, κ] · c_raw[i, s, κ, k]              n ≤ α

with `φ_κ` the K orthonormalised poly3 functions (`W_K` applied to `I_1..I_K`)
and `M ∈ ℝ^{T×S×α×K}` the learned coefficients — the exact analogue of NEP's
`c_nk^{(t_i,t_j)}` with `f_k ↔ φ_κ`. Steps 4–5 then run unchanged on `c`, so
`dim_q`, the trivial compression, the block layout used by `descriptor_mixing`,
the preprocess contraction and `W0` are all untouched.

`c_raw` is exactly what the existing builder computes internally with
`alpha_max = K` before the power spectrum (`aggregate_cnk_tf`), so the builder
change is: stop after step 3 and emit `cnk`.

### Initialisation that reproduces the fixed descriptor exactly

Löwdin orthonormalisation is not nested (`W_K[:α,:α] ≠ W_α`), so `M = [I | 0]`
would NOT reproduce the current α-basis. Instead

    M0 = [W_α | 0_{α×(K−α)}] · W_K^{-1}

which satisfies `M0 φ^K = φ^α` exactly (both bases span the poly3 functions
`I_1..I_α`). Since `φ^α` are orthonormal and lie in span(`φ^K`), the rows of
`M0` are orthonormal vectors in ℝ^K: entries are O(1), well conditioned, and
SNES's `init_sigma` perturbs coefficients of an orthonormal basis. Generation 0
must reproduce the fixed-basis descriptor to fp32 precision — this is the
primary correctness test. With `K = α`, `M0 = I`.

### Why not the alternatives

| Option | Verdict |
|---|---|
| Re-run the full SOAP graph per candidate per generation | ~pop_size × descriptor-build cost per gen (minutes); unusable. |
| Learn a bilinear transform of the *uncompressed* power spectrum `p_raw` (`M p_raw Mᵀ`) | Algebraically identical for K = α, but cannot enlarge the basis (the whole point), needs the uncompressed `p_raw` stored (390 vs 360 values/atom), and the existing `nep4_radial` preprocess already covers the K = α diagonal case. |
| NEP-style species-summed channels (`c[i, n, k] = Σ_s Σ_κ M[t,s,n,κ] c_raw[i,s,κ,k]`, no species blocks) | More NEP-faithful and smaller `dim_q`, but changes the descriptor layout every downstream consumer assumes (mixing blocks, preprocess, saved models). Deferred — noted as a follow-up; the chosen design is a strict superset in information. |
| **Chosen**: species-blocked learned radial mixing of a K-sized orthonormal basis, power spectrum formed per candidate in-model | Preserves every downstream layout; per-candidate cost is two small einsums; exact gen-0 equivalence. |

## 3. Design

### 3.1 Config (one knob)

```python
# Learnable radial basis (GPU builder only, descriptor_mode=1):
#   None : fixed SOAP-turbo basis (current behaviour)
#   K    : int ≥ alpha_max. The builder emits raw centre coefficients over the
#          K orthonormal poly3 functions; the α radial functions per
#          (centre type, neighbour species) become learned combinations
#          M[T, S, α, K] in the SNES vector (init reproduces the fixed basis).
radial_basis_size: int | None = None
```

Gate everywhere: `cfg.radial_basis_size is not None`. Validation at model
init: requires `descriptor_mode == 1` (the builder itself already enforces
`basis == "poly3"` and `compress_mode == "trivial"`), `alpha_max ≤ K ≤ 10`
(the poly3 overlap is SPD only up to ~10, and `M0` needs `W_K⁻¹`).

### 3.2 Builder (`DescriptorBuilderGPU_tf.py`)

- `_build_cfg_cache` gains `n_basis = K` and a second orthonormalisation
  `W_K` (the existing `W_np` at α stays for the fixed path).
- New forward-only inner `_compute_cnk_inner_tf`: radial (at `alpha_max=K`,
  `W_single=W_K`) → angular → `aggregate_cnk_tf` → returns `cnk`
  `[k_max, S·K, N]` complex, re-laid out as float32 `[N, S·K·k_max·2]`
  (order: species s, κ, k, (re, im)).
- `DescriptorBuilderGPUTF.build_descriptors_flat / build_descriptors` emit
  this in the descriptor slot when the knob is on. Per-frame tuple stays
  `(desc, grad, pa, pg)`. `calc_gradients=True` under the knob: for the
  neighbor readout (which needs the pair index stream on the trajectory path,
  where V is built in-graph, but no gradient values) it returns real `pa/pg`
  with zero-width `grad [P, 3, 0]`; for gradient-consuming modes it raises
  `NotImplementedError` until §5 lands.
- Prerequisite bug: `DescriptorBuilderGPUTF` lacks `build_descriptors_self_only`
  (the neighbor readout crashes with `descriptor_mode=1` today). Add it:
  neighbor readout → `build_descriptors(calc_gradients=False)`; otherwise
  build with gradients and filter to self pairs via the existing
  `_filter_to_self_pairs` helper. Under the neighbor readout this yields
  **P = 0** pairs (`grad_values [0, 3, W_in]`, empty pair arrays) — declared a
  supported invariant: the neighbor forwards never read `grad_values`, and
  both staging paths (`_stage_chunk_resident` has an explicit size-0 branch;
  `_stage_finalize_tf` takes the empty-gather branch since
  `_is_contiguous_range([]) is False`) are exercised by a test.

### 3.3 Data flow — the `descriptors` slot carries raw coefficients

The padded `descriptors` tensor `[S, A, W_in]` already flows through
`pad_and_stack`, `slice_and_complete_chunk`, `_stage_*`, `prefetched_chunks`
with a generic last dimension. Under the knob `W_in = S·K·k_max·2` instead of
`dim_q`; **no staging code changes**. The model reshapes it back on entry.
`cfg.dim_q` remains the compressed SOAP width (what `W0` sees).

Sites that hardcode the input width to `dim_q` are switched to one cfg-only
helper `descriptor_input_width(cfg)` (= `S·K·k_max·2` under the knob, else the
raw compressed width). It is the width of the *slot*; the existing
`TNEP.dim_q_forward` stays the width of the W0 matmul after the preprocess
fold — different things, both documented: the `built_dim_q` cross-check in `MasterTNEP`,
`spectroscopy._get_fused_predict` (TensorSpec + scatter shape) and the two
pack helpers' `dim_q` argument.

### 3.4 Model (`TNEP.py`)

- New weight `M_radial` `[T, S, α, K]` (init `M0` tiled over T), and constants:
  `_rb_pivot_rows` (the S pivot rows of the `n` axis), `_rb_Wm [k_max, L]`
  (`Wm[k,l] = (2 if m(k)>0 else 1)·[l(k)=l]`), and a gather map from the
  `dim_q` Fortran channel order to `(pivot s, n', l)` plus the √2 factor for
  n ≠ n' — all derived from `make_compress_mask_trivial` / the same loop that
  `_build_kept_triples` uses, so the channel order is provably identical to the
  fixed builder's.
- `soap_from_raw(raw, M)`:
  ```
  raw [..., A, W_in] → [..., A, S, K, k_max, 2]
  per type t:  cc_t = einsum('(c)snq,...asqkr->(c)...asnkr', M[(c),t], raw) · mask_t
  cc  [(C,) B, A, S·α, k_max, 2]
  G   = einsum('...pkr,...nkr,kl->...pnl', cc[pivots], cc, Wm)   # [(C,) B, A, S, S·α, L]
  q   = gather(G.reshape(..., S·S·α·L), kept_idx) · sqrt2_fac     # [(C,) B, A, dim_q]
  q  /= ‖q‖  where ‖q‖ ≥ 1e-5, else left un-normalised             # SOAP-turbo's guard
  ```
  All real float32 einsums; no linalg ops (no eager sync cost, see
  `eager_tf_perf_gotcha`). Memory per candidate ≈ 180 floats/atom for `G`.
- `predict`, `predict_batch`, `predict_batch_candidates` call it on entry when
  the knob is on (`M_radial` kwarg, default `self.M_radial`).
  `predict_batch_candidates` needs one new branch: descriptors are then
  per-candidate `[C, B, A, Q]`, so the input→hidden product becomes a batched
  matmul `[C, B·A, Q] @ [C, Q, H]` per type instead of the shared-descriptor
  GEMM. Everything after `pre_h` is unchanged (mixing / preprocess folds act on
  `W0` in Q-space and compose as before).

### 3.5 SNES (`SNES.py`)

- Tail order becomes `[ANN | c_pair | pol ANN | U_pair | W_pre | M_radial]`;
  `n_radial = T·S·α·K`; `dim += n_radial`.
- `reconstruct_params_tf` reshapes the tail to `[(C,) T, S, α, K]`;
  `_split_reconstructed` exposes it as `"M_radial"`; `_set_model_params`
  assigns it.
- μ init: `M0` tiled; σ init: `init_sigma` (GPUMD uses one `sigma0` for c_nk
  and ANN alike). No extra knob.
- `_build_type_of_variable`: label = centre type `t` (per-type fitness ranking
  drives each centre type's radial functions, matching how NEP4 ranks).
- L1/L2: excluded, like the other tails with a non-zero init prior (`U_pair`,
  `W_pre`). `_extract_type_params` / `_global_typed` already ignore tails.
- `_evaluate_chunk_impl` and `validate` pass `M_radial` through (`validate`'s
  intra-generation `_validate_fold_cache` tuple must carry it too). The mode-2
  `vectorized_map` path is out of Phase 1 scope (mode 2 raises under the knob).

### 3.6 I/O (`model_io.py`)

Save/load the `M_radial` dataset; `_LEGACY_FIELD_DEFAULTS["radial_basis_size"]
= None`. Checkpoints already persist the full μ; the shape check on resume
catches a K mismatch.

### 3.7 Reporting

`_print_param_breakdown` adds a `radial basis  M [T·S·α·K]` row and
`_train_model_inner` prints one line describing the learned basis.

## 4. Issues and risks considered

1. **Conditioning of the search.** Learning coefficients of the *raw* poly3
   integrals would be hopeless (W entries span orders of magnitude). Using the
   orthonormal K-basis with orthonormal-row `M0` keeps every coordinate O(1).
2. **σ scale.** `init_sigma = 0.1` on O(1) orthonormal coefficients is the same
   relative perturbation GPUMD applies to c_nk ∈ [−1, 1]. Revisit only if the
   A/B shows gen-0 val RMSE blowing up.
3. **Precision.** The fixed builder forms the power spectrum in fp64 and casts
   to fp32; here `c_raw` is stored fp32 and the Gram is fp32. Expected
   gen-0 agreement ~1e-6 relative; the equivalence test pins an explicit
   tolerance (atol 1e-5 on unit-norm descriptors).
4. **Memory.** Per pop-chunk: `cc` ≈ 360 and `G` ≈ 180 floats per atom per
   candidate. For `batch_chunk_size=2000`, A_max ≈ 30, `population_chunk_size=10`:
   ≈ 1.3 GB transient. Acceptable on 12 GB; the two chunk knobs already bound it.
   Raw storage: `S·K·k_max·2 = 720` floats/atom at K=8 (vs 165 today).
5. **The `descriptors` slot semantics change** under the knob. Documented on
   `TNEP.soap_from_raw` and the builder; one property (`descriptor_input_width`)
   is the only place downstream code needs to know.
6. **Central-atom term, soft cutoff, amplitude, radial enhancement** all live
   inside `c_raw` (steps 1–3), so they are inherited unchanged.
7. **Compression pivots.** "First radial index of each species" is now the
   first *learned* function of each block; `M` can choose what it is. Fine.
8. **Mixing / preprocess composition.** Both act in Q-space after
   `soap_from_raw`; `nep4_radial` stays mutually exclusive with mixing as
   today. No new exclusions needed.
9. **Per-type ranking of M.** Labelled by centre type (like per-type W0).
10. **quippy CPU builder** cannot emit `c_raw`; the knob requires
    `descriptor_mode=1` and raises otherwise.
11. **Retracing.** `predict_batch_candidates` is `@tf.function(reduce_retracing)`;
    the new branch is static (Python bool), so one extra trace at most.
12. **Gradient-consuming modes** (PES forces, `dipole_readout="tnep"`,
    polarizability) need `∂q/∂r` through the learned basis — §5. Phase 1 raises
    a clear `NotImplementedError` for them under the knob; the neighbor readout
    (the current default) reads no descriptor gradients and is fully supported.
13. **K < α** is rejected (basis must contain the α learned functions at init).
14. **Existing saved models** are unaffected (knob absent → `None`).

## 4b. Second option — species compression (`radial_basis_species_sum`)

`radial_basis_species_sum: bool = False` selects how the learned radial
channels treat neighbour species (only meaningful with `radial_basis_size`):

- **False (default) — keep SOAP-turbo's dimensions.** Species-blocked as in
  §2: `c[i,(s,n),k] = Σ_κ M[t_i,s,n,κ] c_raw[i,s,κ,k]`, n_max = S·α, trivial
  compression, `dim_q` unchanged (165 for S=3, α=4, l_max=4).
- **True — NEP-style species compression.** The neighbour-species axis is
  summed into α shared channels, exactly as NEP's `g_n = Σ_j Σ_k c_nk^{(t_i,t_j)} f_k(r_ij)`:
  `c[i,n,k] = Σ_s Σ_κ M[t_i,s,n,κ] c_raw[i,s,κ,k]`, n_max = α. The power
  spectrum is the full upper triangle over the α channels (no compression
  needed — it is already small): `dim_q = α(α+1)/2 · (l_max+1)` (50 for α=4,
  l_max=4), Fortran (n, n', l) order with the √2 rule for n ≠ n', per-atom
  L2 norm. Species information enters only through `M`, so `dim_q` no longer
  grows with the number of species. Init: `M0` tiled over `s` (every species
  contributes the same α functions at gen 0 — a species-agnostic SOAP;
  SNES noise breaks the symmetry). There is no fixed-basis descriptor with
  these dimensions, so gen-0 equivalence is not defined for this mode; its
  reference test is an independent NumPy power spectrum over the summed
  channels.

The raw builder output is identical in both modes; only `soap_from_raw`'s
contraction, `compute_dim_q`, `radial_basis_channel_map` (rows = all α
channels, keep all n ≤ n') and `descriptor_block_layout` (one block
`(0,0)` with α(α+1)/2 channels per l, so `descriptor_mixing` still works)
change. `descriptor_preprocess_contract != "off"` is rejected in this mode
(its layouts assume species-pair blocks).

## 5. Phase 2 — descriptor gradients through the learned basis (designed, deferred)

Every gradient consumer contracts `de_dq[i, q] · ∂q̂_i/∂r_p` into a per-pair
force `F_p`. With `q̂ = q/‖q‖` and `q` bilinear in `c = M c_raw`:

    E_i          = (de_dq̂_i − q̂_i (q̂_i · de_dq̂_i)) / ‖q_i‖        # un-normalise, per atom (‖q‖ ≥ 1e-5 guard as forward)
    Ẽ_i[n,n',l]  = symmetric expansion of E_i over kept channels:
                   Ẽ[n,n] = E[(n,n,l)];  Ẽ[n,n'] = Ẽ[n',n] = E[(n,n',l)] / √2   (n ≠ n')
                   (the stored channel is √2·p_{nn'}, so the symmetric double
                   sum divides by √2; `mult_k` below is the m-multiplicity only,
                   1 or 2 — NOT the codebase's `multiplicity_array`, which bakes
                   the √2 in and would double-count)
    A_i[n,k]     = Σ_{n'} Ẽ_i[n,n',l(k)] · c_i[n',k]                 # complex [N, S·α, k_max]
    B_i[s,κ,k]   = Σ_n M[t_i,s,n,κ] · A_i[(s,n),k]                   # [N, S, K, k_max]
    F_p[d]       = 2 Σ_{κ,k} mult_k · Re( dc_raw[p,d,κ,k] · conj B_{i(p)}[s(p),κ,k] )

`dc_raw[p, d, κ, k]` is the Cartesian derivative of the pair's own species
block of `c_raw` — candidate-independent, stored once per pair
(`[P, 3, K, k_max, 2]` float32, ~2.9 KB/pair at K=8 vs 2 KB/pair today). Per
candidate the only per-pair work is the last contraction (same order as
today's `Σ_q de_dq · grad_values`). The self pair is `F_{ii} = −Σ_j F_{ij}`
(translational invariance) in full-COO mode; the self-only mode
(`dipole_rij_power=0`) stores the pre-summed self block `[N, 3, S, K, k_max, 2]`
instead. The precomputed candidate-independent dipole kernel `W_atom` no longer
applies under the knob (it depends on M); the tnep readout uses the per-pair
path. Three consumers change: `predict_batch`'s `_calc_forces_coo` call (its
only call site; `_polarizability_coo` consumes its output), the padded
`calc_forces` used by `predict`, and `predict_batch_candidates`' `W_atom`
contraction.

## 6. Validation protocol (actual outputs only)

1. `tests/test_radial_basis.py` (no stored fixtures exist in the repo — every
   reference is built live)
   - gen-0 equivalence on CHO frames from `datasets/train.xyz`: with K = α and
     K = 2α, `soap_from_raw(builder raw, M0)` ≈ the fixed TF builder's
     descriptors (atol 2e-5), which are themselves validated against quippy by
     `DescriptorBuilderGPU_tf.run_phase11_validation`.
   - P = 0 staging invariant under the neighbor readout (§3.2).
   - M0 rows orthonormal; K = α ⇒ M0 = I.
   - SNES round trip: `reconstruct_params_tf(mu_init)["M_radial"] == M0`;
     `_set_model_params` then `score` matches the fixed-basis model's score
     on a small CHO subset at gen 0.
   - save → load → identical predictions.
2. End-to-end A/B on CHO (`train.xyz` / `test.xyz`, `allowed_species=[6,1,8]`),
   `descriptor_mode=1`, same seed, same generation budget, three runs:
   fixed basis; K = α; K = 2α. Record the printed val/test RMSE and per-gen
   timing verbatim.
