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
| gap_qeq_scan.py | μ: QEq "chi" column (Ṽ = −P·L⁺·v_fc, identity verified 1e-14 on a cluster) and "polarise" kernel, ± field column | chi 1.69–1.70 at H = 5/10/20 (baseline 1.67), chi+field 1.19 vs field 1.17; polarise 4.85 (H=10) / 3.59 (H=40), harmful |
| gap_induct_mu.py / gap_radial_smeared.py | μ: Applequist screened field, induced p, smeared (LODE-like) sources, radial families with the field column | screened field 1.64 (loses the plain field's 1.17); smearing σ=0.5 1.17, σ=1.0 1.20 — point-charge field is right |
| gap_induct_alpha.py / _tr.py | α: Applequist induced-coupling site tensors Σ_j R_ij as a kernel group | full tensor 16.7/15.2 (diag worse); TRACELESS 14.5/15.0 vs base 15.3/16.3 — both improve |
| gap_2b3b_scan.py / _delta.py / _sigma.py / _cond.py | TurboGAP 2b/3b blocks on μ (field on) and α (two-group + cross): amplitudes, σ, jitter, centre spacing | μ +2b neutral at δ≤0.1, worse at δ≥1; +3b 1.75–3.6 % at every setting (train RMSE 1e-4: interpolates); α +2b 21–27 %, +3b 30–64 % vs 15–16 % base; jitter and well-conditioned centres change nothing |
| gap_leak_bisect.py / gap_path_check.py | audit: do 2b/3b hyperparameters leak into the base; do the production and experimental fit paths agree | no leak (all 15.1/17.6); paths agree to 6e-12. A one-row discrepancy was a race on a shared training file between concurrent scans |
| gap_site_models_scan.py (`mu` / `pol`, separate processes and record dirs) | NEP-style site-model decomposition (`gap_site_models`: soap / 2b / 3b, each its own kernel + FPS set; 3b via its angular kernel): does splitting the body orders beat the single SOAP site function? | μ: soap+2b (δ_2b 0.3) 1.08 → **1.00 %**, soap+3b 1.02, soap+2b+3b 1.05 (δ = 1 worse than the base in every case); 2b-only 1.76, 3b-only 2.65. α: nothing beats 14.9/16.0 — soap+3b 14.9/17.0, soap+2b+3b 15.5/15.9, soap+2b 16.1/17.0; 2b-only 24.2/19.2; 3b-only has no isotropic group (diagonal undefined, off-diag 18.6) |

### gap_site_models_scan.py results (2026-10-20; val-selected on train 400/100, refit on 500, test RRMSE)

Common: gap_site_channels=10, gap_site_lmax=2, M=250 per species per model, ξ=2; μ base = l=α=7 uncompressed SOAP, rc 3.0, Ewald field column; α base = l=α=7 trivial, rc 4.0, cross group C_x=4. σ grid μ {1e-3, 3e-3}, α {0.1, 0.3}. Unlisted δ = 1.0.

| tag | μ: selected (σ, δ, …) | μ test | ncols | fit s | α: selected (σ, δ, …) | α test diag/off | ncols | fit s |
|---|---|---|---|---|---|---|---|---|
| soap | 1e-3 | 1.08 % | 11000 | 158 | 0.1 | 14.9/16.0 | 38000 | 81 |
| 2b | 3e-3 | 1.76 % | 11000 | 118 | 0.3 | 24.2/19.2 | 38000 | 46 |
| soap+2b | 3e-3, δ_2b 0.3 | **1.00 %** | 22000 | 144 | 0.3, δ_2b 0.3 | 16.1/17.0 | 76000 | 120 |
| 3b | 3e-3, C_x 4, l_x 1 | 2.65 % | 32000 | 133 | 0.3, l_x 2 | (5001)/18.6 | 54000 | 134 |
| soap+3b | 3e-3, C_x 4, l_x 1, δ_3b 0.3 | 1.02 % | 75000 | 170 | 0.3, l_x 1 | 14.9/17.0 | 74000 | 174 |
| soap+2b+3b | 3e-3, C_x 4, l_x 1, δ 0.3 | 1.05 % | 118000 | 166 | 0.3, δ 0.3 | 15.5/15.9 | 112000 | 216 |

The 3b-only α row has no isotropic group (its angular kernel is traceless), so the diagonal is the mean-free prediction and is not a result. Every decomposed α row preferred the larger σ (0.3) while the base preferred 0.1: the extra kernels add variance the 2400 rows cannot pin down. The μ 3b rows also carry the T·v cross group on the soap/2b readout (C_x = 4, forced by the 3b kernel, while the μ base has C_x = 0), which §5 of the docs found neutral (1.76 → 1.71 % without the field).
