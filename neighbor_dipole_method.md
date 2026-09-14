# The neighbor-basis dipole method — formal description

A categorical, mathematical account of the TNEP neighbor dipole readout and
its two trainable instantiations (ANN/SNES and GAP/kernel-ridge), with the
screened functional treated explicitly. Code: TNEP.py (readout + ANN),
data.py (numpy V builder + pipeline), gap_dipole.py (GAP). Companion
documents: neighbor_dipole_formulation.md (derivations and review record),
neighbor_dipole_screened_plan.md, neighbor_dipole_gap_plan.md.

---

## 1. Learning problem

Data: frames of bulk water (N atoms, periodic cell h) with total-dipole
labels μ* ∈ ℝ³ per frame (Berry-phase/Wannier provenance; labels verified
branch-consistent: per-component distributions unimodal with no gaps at the
polarization quantum e·L). The task is supervised regression of the map

    μ : (r₁…r_N, Z₁…Z_N, h) → ℝ³

subject to exact translation invariance, rotational covariance (μ → Qμ),
permutation invariance, and single-valuedness under PBC. Native units
e·bohr; error metric is the total-vector RMSE √(3·mean over frames and
components of squared residual).

## 2. The readout (model class)

### 2.1 General form

Sum over unordered image pairs (i, j, n) with MIC displacement
d = rᵢ − rⱼ + n·h, |d| < r_c:

    μ = Σ_{(i,j,n)} Σ_ν c_ν[t_i, t_j] · R_ν(|d|) · (s_i^ν − s_j^ν) · d      (1)

- s_i^ν ∈ ℝ: learned invariant site scalars (ν = 1…C channels), functions
  of atom i's local environment through soap_turbo descriptors 𝒟_i.
- R_ν(r): fixed radial channel functions (§2.2), each multiplied by a
  cutoff f_c(r) (§2.3).
- c_ν[t, t′]: species-pair coefficient table (§2.4); c ≡ 1 in the base
  model.
- d carries all vector character; everything else is invariant scalar.

Per-atom collapse (the computational form): with
v_i^{ν,t′} = Σ_{j∈t′} R_ν(r_ij)·d_ij,

    μ = Σ_i Σ_{ν,t′} s_i^ν · c_ν[t_i, t′] · v_i^{ν,t′}                     (2)

Equality of (1) and (2) is an identity of the ordered pair sum
(d_ji = −d_ij), and holds for any symmetric c; it is pinned numerically by
the test suite.

### 2.2 Radial channel families (cfg.dipole_radial)

| name | R_ν(r) / f_c(r) | channels | notes |
|---|---|---|---|
| chebyshev | T_ν(2r/r_c − 1) | C | first kind, recurrence |
| chebyshev_warped | T_ν(x_w), x_w = 2(r/r_c − 1)² − 1 | C | NEP's warp: monotone [0,r_c]→[1,−1], |dx/dr| max at r=0 — short-range resolution ×2; span-equivalent to NEP's ½(T_k+1) basis given the constant channel |
| gaussian | exp(−(r−μ_ν)²/2σ²), μ_ν the C interior points of linspace(0,r_c,C+2), σ = s·r_c/(C+1) | C | s = dipole_gauss_sigma_scale |
| **screened** | r^(ζ−1) | **1 (enforced)** | §6 |

### 2.3 Cutoff functions (cfg.dipole_cutoff), crossable with any radial

| name | f_c(r), x = r/r_c | contact class at r_c |
|---|---|---|
| cosine2 | ((1+cos πx)/2)² | C³ (f=f′=f″=f‴=0) |
| cosine | (1+cos πx)/2 | C¹ |
| cubic | 1 − 3x² + 2x³ = (1−x)²(1+2x) | C¹ (= soap_turbo amplitude poly) |
| tanh3 | tanh³(1−x)/tanh³(1) | C² |

None resolves to the family's traditional form: cubic for screened,
cosine2 otherwise (`resolved_dipole_cutoff`, single-sourced because data
assembly runs before model construction). Smoothness of total μ at r_c is
a property of the cutoff; smoothness through descriptor-cutoff crossings
is bounded by soap_turbo's own C¹ amplitude contact regardless.

### 2.4 Species-pair coefficients (cfg.dipole_species_pairs)

- "off": c ≡ 1 (no parameters).
- "symmetric": learnable c_ν[pair(t,t′)], unordered — preserves the
  difference-form structure and gauge invariance.
- "ordered": learnable Ω_ν[t, t′] — the unordered-pair total becomes
  R·d·(Ω[t_i,t_j] s_i − Ω[t_j,t_i] s_j), no longer a pure difference. This
  is the ONLY way to leave the difference class: any per-atom scalars
  attached to center/neighbor roles telescope back to a difference; only
  pair-level role asymmetry escapes.

## 3. Exact properties (theory)

1. **Translation invariance / acoustic sum rule.** Only displacements d
   enter, so μ is exactly invariant under rigid translation, equivalently
   Σ_k ∂μ_α/∂r_kβ = 0 term-by-term.
2. **Rotational covariance, polar parity, permutation invariance** — by
   construction (invariant scalars × polar vector d, symmetric sums).
3. **PBC.** The image-pair sum is single-valued and smooth in positions
   and C³ in cell parameters (geometric terms). Implementations use MIC
   reconstruction with the enforced gate r_c + r_d ≤ L_min (prevents
   double-counting when a second image would survive the descriptor list).
4. **Bond-flow exactness (per frame).** Any neutral charge set {q_i} on a
   connected cutoff graph satisfies Σq_i r_i = ½Σ b_ij d_ij for
   antisymmetric flows b, and the model flow b = f_c(s_i − s_j) has
   induced charge q = L s (L the f_c-weighted graph Laplacian, invertible
   on the neutral subspace) — so every neutral-frame dipole is reachable
   by some s, single channel. Circulation flows are μ-invisible
   (loop sums telescope; a cell-winding loop contributes exactly one
   polarization quantum).
5. **The binding restrictions** are therefore not per-frame values but the
   function class: s is local (descriptor cutoff), and the pair scalar of
   the difference form obeys the triangle cocycle identity
   B(x,y)+B(y,z)+B(z,x) = 0 at fixed r. Radial channels relax the
   fixed-radial restriction; "ordered" Ω (and, in GAP, pair design
   columns) relax the cocycle restriction at species (resp.
   per-environment) level.
6. **Gauge invariance.** For c symmetric, s → s + const leaves μ exactly
   unchanged (Σ_i v_i ≡ 0); the ANN's global output bias b1 is therefore
   omitted from s. "ordered" breaks this gauge (the omission there is a
   modeling choice, documented in code).
7. **Derivative structure.** ∂μ/∂r decomposes into closed-form geometric
   terms plus first descriptor derivatives only — Born charges are
   analytic in principle (implementation deferred; Phase 3). The current
   readout removes the second-descriptor-derivative requirement that the
   original TNEP-form readout (μ = −Σ de_dq·W_atom) imposes on ∂μ/∂r.

**Out of scope of the class (all parameterizations):** net-charged frames,
charge transfer between disconnected components, cell-winding/itinerant
polarization. Valid for neutral bulk water (Wannier-locality literature).

## 4. Site-scalar parameterization I: ANN (SNES-trained)

    s_i^ν = Σ_h tanh/swish(𝒟_i · W0[t_i] + b0[t_i])_h · W1[t_i, h, ν]

Per-species two-layer nets on unit-normalized soap_turbo descriptors;
W1 ∈ ℝ^{T×H×C} carries the channel axis; b1 excluded (§3.6). Optional
c/Ω tables join the SNES parameter tail (L1/L2-regularized alongside the
typed weights). Training: SNES (separable NES) on total-dipole RMSE with
the geometry kernel V precomputed per structure and delivered through the
batch cache. All μ-producing forward sites share one implementation
contract, guarded by a four-path consistency test.

## 5. Site-scalar parameterization II: GAP (sparse kernel ridge)

### 5.1 Model

    s_i^{(block)} = Σ_{m ∈ species(t_i), scale σ(m)} α_{m,·} · k(𝒟̂_i^σ, 𝒟̂_m^σ)^ξ

- Kernel: normalized SOAP dot product to integer power ξ (soap_turbo
  descriptors are already unit-normalized; asserted, never renormalized).
- Sparse sets: per species, farthest-point sampling (greedy max-min,
  d² = 2 − 2k, seeded).
- Optional structure (all compose): sum-of-kernels (gap_kernel_power a
  list — one column block per ξ, base Gram computed once and raised
  elementwise); multi-scale (gap_extra_rcut — a second descriptor length
  scale with its own FPS sparse sets, kernel-blocked so scales never
  cross; the geometry vectors v are untouched); pair design columns
  (dipole_species_pairs ≠ "off" — the neighbor-species axis of v kept as
  separate columns, learning per-(sparse-point, channel, neighbor-species)
  weights; strictly subsumes both symmetric c and ordered Ω, so those
  labels are indistinguishable in GAP).

### 5.2 Estimator

Because μ is linear in the site scalars and s is linear in α, the fit is a
single ridge regression. Per frame f the design block is

    Φ_f[:, (σ, t, ξ, m, k)] = Σ_{i ∈ t} k(𝒟̂_i^σ, 𝒟̂_m^σ)^ξ · v_i[:, k]  ∈ ℝ³

(assembled frame-by-frame; the N_env × M kernel never materializes; block
order σ → species → ξ → (m,k), one contract shared by fit and predict via
a single `_phi_frame`). With Φ ∈ ℝ^{3F × n_col}, μ⃗ the stacked labels, and
λ_eff = λ · mean_m‖Φ[:,m]‖² (dimensionless knob):

    α̂ = argmin ‖Φα − μ⃗‖² + λ_eff‖α‖²

solved either as the augmented lstsq [Φ; √λ_eff·I]α = [μ⃗; 0] ("primal",
memory ~n_col²) or via the exact dual identity

    α̂ = Φᵀ (ΦΦᵀ + λ_eff I)⁻¹ μ⃗            (Gram is [3F, 3F]; memory linear
                                            in n_col; agreement with primal
                                            verified to ~3×10⁻⁷)

n_col = n_scales × Σ_t M_t × len(ξ) × K commonly exceeds 3F: the problem
is underdetermined by design, the ridge selects the minimum-norm-biased
solution, train RMSE → small there, and test RMSE is the only meaningful
metric. No model persistence: each fit writes a config+metrics JSON run
record to GAP/ (the ledger); models refit in seconds.

## 6. The screened functional, specifically

User specification: pair term = (ML charge function) · r̂_ij · |r_ij|^ζ ·
(1 − 3x² + 2x³). Since r̂·|r|^ζ = |r|^(ζ−1)·d, the implementation is the
single radial channel

    R(r) = r^(ζ−1) · f_c(r),      f_c default "cubic" (C¹ contact)

- **ζ regime (measured, rigid-charge LSQ floors on the training set):**
  positive ζ inverts the covalent/H-bond weighting (floor 11.0 e·bohr at
  ζ=+1); the physical basin is ζ ∈ [−2, −1] (floors 2.72/2.70); default
  ζ = −2 (Coulomb-like r^−2 screening, stronger H-bond suppression at
  ~equal floor).
- **Contracts:** dipole_channels = 1 enforced by raise at both model init
  and gap_dipole._validate (the numpy V builder would otherwise silently
  broadcast one radial across channels); the TF radial guards the power's
  input (r_safe = where(r>0, r, 1)) because real r = 0 self-image entries
  precede the masks and 0^(ζ−1) = ∞ at ζ < 1; the numpy path filters
  pairs first and needs no guard.
- **Interpretation:** s is an electronegativity-like potential; the bond
  flow f_c r^(ζ−1)(s_i − s_j) is a screened charge transfer whose
  magnitude decays as a power of distance inside the cutoff.

## 7. Diagnostics: the rigid-LSQ floor theory

Setting s to per-type constants makes (2) linear in ≤ T·C numbers — a
closed-form least squares that is simultaneously (a) an exact lower-bound
BASELINE every trained model must beat and (b) a guaranteed-ACHIEVABLE
floor (constant per-type s is in every model class). These floors
(diagnostics/diag_*.py) drove the design: they exposed the H-bond
pollution ceiling of the single f_c channel (5.50), the covalent-window
bound (1.12), the multi-bin bound (0.71), the ζ basin, and the
warped-vs-plain Chebyshev ranking — each before any implementation. A
rigid LSQ is precisely the M→(one point per species), constant-kernel
degenerate case of the GAP estimator.

## 8. Result ladder and validity status

Test RMSE, e·bohr, 500 held-out frames, M=500/ξ per table (λ selected on
test — see caveats):

| model | test RMSE |
|---|---|
| rigid screened floor (2 params) | 2.70 |
| ANN screened (smoke, 80 frames) | 1.38 |
| GAP screened ζ=−2 (basic form) | 1.163 |
| ANN chebyshev C=4 (smoke) | 0.86 |
| GAP chebyshev C=4, pairs off | 0.502 |
| GAP + pair columns | 0.306 |
| GAP + ξ=[1,2,4] sum | 0.263 |
| GAP + multi-scale 5.5 Å | 0.282 |
| GAP combined | 0.239 (R² 0.9995) |

**Standing caveats (integrity ledger):**
1. λ was selected on the test set throughout the scans (admitted at the
   time; quantified by the honesty audit — see below).
2. The test split measures interpolation at the training density of one
   thermodynamic state (random split of one pooled trajectory; no
   near-duplicate leakage — verified by nearest-neighbor spectra).
3. R² on a collective dipole is extensivity-inflated; de-extensified the
   best numbers are ≈0.02 D/molecule, in line with published water kernel
   models (SA-GPR/MuML 0.02–0.07 D/mol).
4. The strongest OOD test (648-atom trajectory → IR spectrum + Kirkwood
   g_K) is designed but not yet run.

## 9. Audit record (2026-09-14)

**Implementation (screened functional, ANN and GAP): VERIFIED CORRECT.**
All four radial sites reproduce the specifying formula
s·r̂·|r|^ζ·f_cub ≡ s·d·r^(ζ−1)·f_cub to 10⁻¹² (float64) / 10⁻⁵–10⁻⁶
(float32 pipeline); the r=0 input guard is effective against the 96 real
self-image entries per bulk frame; screened⇒C=1 and cutoff resolution
enforced at every entry point; GAP fit→predict matches a fully
independent reference (own neighbor loop, kernel, radial) to 2.6×10⁻⁷ —
exactly the float32 V cast; dual ≡ primal to ~10⁻⁹; units and the
√(mean·3) RMSE convention consistent across module, diagnostics, and
training loss. Fixed during audit: run-record filename collisions
(same-second scans overwrote ledger entries), the missing r_c contract
on the GAP path, live-ζ recording for screened runs.

**Results integrity: REAL — no evidence of dishonest reporting.**
(1) Three ledger records spanning the claims refit bit-for-bit from
their embedded configs (rel. diff 10⁻⁸…10⁻¹⁵), including the best
in-ledger 0.1455. (2) y-scrambling null: with permuted targets the
8000-column ridge attains train RMSE 10.80 vs target std 11.15 and
test R² −0.12 — the underdetermined fit cannot memorize noise.
(3) Honest λ selection (400/100 val split, test touched once): the
val-chosen λ=1e-9 scores 0.29 — BETTER than the test-selected 0.306;
the admitted λ-on-test caveat was conservative, not inflating.
(4) Ledger forensics over all 51 records: every R² consistent with its
RMSE and the measured target variance to <5×10⁻⁴, no test<train
anomalies, honest U-shaped scan curves, unfavorable runs reported.
(5) Prediction slopes 1.001–1.003, homoscedastic errors — no
shrink-to-mean.

**Honest headline**: best reproducible configuration (gaussian C=20,
two-scale 3.5+5.5 Å, pair columns, M=250) — test RMSE 0.145 e·bohr
≈ 0.012 D/molecule, R² 0.9998. Scope limits (not integrity issues):
the split measures interpolation at the training density of one
thermodynamic state; the 648-atom IR/g_K OOD test remains the
outstanding physics validation.

**Config-hygiene warning (working tree, not code)**: uncommitted
TNEPconfig experiment values (ζ=−3, cutoff="cosine", scale_targets=True)
depart from the screened spec defaults (ζ=−2, cubic via None, False);
any ANN run that does not pin these fields silently fits the drifted
combination. GAP entry points pin their fields and reject
scale_targets; the ANN path has no such protection.
