# GAP dipole / polarisability experiment drivers (2026-10-19)

Convex-floor scans behind the ExperimentalPolarisability work. Each script
fits on train frames 0:400, selects σ on 400:500, and scores test; run from
the repo root (`python diagnostics/gap_experiments/<script>.py`), output in
`diagnostics/gap_experiments/out/`. Modules under test: `gap_dipole.py`
(production), `gap_bond.py` and `gap_ewald.py` (experimental).

| script | question | outcome (test RRMSE) |
|---|---|---|
| gap_scan.py / gap_cross_scan.py | α: group δ, λ_shear, 6 Å, cross sym(v⊗v) group width | cross C_x=4: 14.9/16.0 (paper 16.3/20.4); width saturates at 4 |
| gap_pol_lc.py | α learning curve | 100→500 frames: diag 24.4→14.9, ∝ N^−0.33, data-limited |
| gap_desc_ladder.py | μ: l=α=4 trivial / 7 trivial / 7 none; M 250→500 | 1.76 → 1.64 → 1.56; M saturated |
| gap_variants.py | μ: bilinear ALS, antisymmetric bond kernel | ALS degrades val; bond kernel ≤ 2% gain |
| gap_ewald_scan*.py | μ: Ewald field column, φ product kernel, near/far split | field column 1.56 → 1.17 (1.08 on 500 frames); near-only or far-only ≈ 0 gain; φ kernel harmful |
| gap_polarise.py | μ: self-consistent induced-dipole field | see memory notes |
