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
| gap_2b_probe.py (`mu` / `pol`; `PROBE_ROWS=` for follow-ups; prints µs/atom from `gap_dipole.predict_timed`) | push the 2b site model (δ, range `gap_site_rcut`, channels, M, ξ, raw kernel) and multiscale (`gap_extra_rcut`) alone and stacked; forward-pass cost per model | μ: soap+2b at δ_2b 0.05–0.1, σ 1.5–2e-3 → **0.96–0.97 %** (soap 1.08) at +2–7 % µs/atom; longer 2b range, more/fewer channels, M 500, ξ [2,4], raw kernel and multiscale all worse or neutral. α: 2b hurts at every setting; a second SOAP scale helps the diagonal — 5.5 Å 14.2/16.1 (2.5× cost), **6.5 Å 13.9/16.2** (3.2×), 7.5 Å 14.0/16.2 (4×); off-diagonal unchanged |
| gap_pol_groups_scan.py (`POL_FINALS=` for follow-ups) | α: long-range column groups `pol_field_groups` (induced / grad / cross / quad / qeq) alone, paired, all five, and on the 6.5 Å two-scale model | induced is the only group that moves the off-diagonal: base 14.9/16.0 → **+induced 14.0/14.6** (500 frames, +500 columns, no cost); +induced+quad 13.9/14.5; all five 13.9/14.3; grad / cross / qeq null. Stacked on the 6.5 Å scale: 13.3/15.5 (+induced, σ 0.3), 13.0/15.2 (+induced+quad). Two rows (all-five two-scale, shear-weighted variants) not run: the process OOM-killed WSL twice |
| gap_mu_baselines.py / gap_pol_twogroup.py | the ORIGINAL models on the 500-frame protocol, for the docs' baselines (run alone under an 8 GB RSS watchdog) | μ original (l=α=4 trivial, bond basis, no field) 1.76 % (1.86 on 400), 0.28 ms/atom; SOAP-7 uncompressed no field 1.56 % (1.67), 0.43 ms/atom. α two bond groups only 16.6/19.9 (16.9/20.2 on 400), 0.63 ms/atom. Peak RSS 2.3–4.1 GB for 10k–20k columns ≈ 200 kB per column at 3000 rows: the 90k-column model needs ~18 GB, which is the first OOM |
| gap_angular_cross_scan.py / gap_triplet_tensor_scan.py + _driver.sh | α: two triplet-form groups — sym(w⊗v) of the 3b angular vector and the dipole kernel vector (`gap_angular_cross`), and Σ k_s(q_ijk) sym(d_ij⊗d_ik) on the TurboGAP 3b centres (`gap_triplet_tensor`); C_x = 2 base for memory; the driver runs ONE FIT PER PROCESS under a 10 GB RSS watchdog | both null: on the 2-channel base (fit 500: 15.1/16.8) w⊗v L1+induced 14.4/15.2 (vs 14.0/14.6 for induced on the 4-channel base), triplet tensor 15.5/17.0 (M10), 16.3/17.7 (M20), +induced 15.5/16.5, in-sample error collapses 9.1 → 4.9–5.9 (interpolation). Memory: identical 35–45k-column fits peak at 3.9–4.8 GB in a fresh process vs 9–11 GB late in a long one — memory accumulates across fits inside a process; always one process per fit |

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

### gap_2b_probe.py results (2026-10-20; stage A = fit 400 → val, stage C = val-selected σ refit on 500 → test; µs/atom = `predict_timed` on the test set, two processes sharing 16 cores at OMP 8, so relative numbers are the reliable ones)

μ (base = soap, l=α=7 uncompressed, rc 3.0, Ewald field; 2b: gap_site_channels 10 on rc 3.0 unless listed; σ_A = 3e-3 unless listed):

| tag | σ | val (400) | test (400) | test (500, selected σ) | ncols | fit s | µs/atom |
|---|---|---|---|---|---|---|---|
| soap | 1e-3 | 1.28 % | 1.17 % | 1.08 % | 11000 | 64 | 1445 |
| soap+2b δ 0.05 | 1.5e-3 | 1.07 | 1.04 | **0.96 %** | 22000 | 71 | 1541 |
| soap+2b δ 0.1 | 2e-3 (1.5e-3) | 1.07 (1.07) | 1.05 | **0.97 %** (0.97) | 22000 | 76 | 1468 |
| soap+2b δ 0.2 | 2e-3 | 1.08 | 1.07 | 0.99 % | 22000 | 67 | 1483 |
| soap+2b δ 0.3 / 0.5 | 3e-3 | 1.10 / 1.13 | 1.09 / 1.12 | (1.00 % from the site-model scan) | 22000 | 60 | 1533 / 1658 |
| soap+2b δ 0.3, gap_site_rcut 4.0 / 5.0 / 6.0 | 3e-3 | 1.23 / 1.38 / 1.49 | 1.21 / 1.36 / 1.43 | — | 22000 | 49–77 | 1608 / 2150 / 2292 |
| soap+2b δ 0.3, C 6 / 20 | 3e-3 | 1.21 / 1.29 | 1.15 / 1.31 | — | 22000 | 58 | 1547 / 1527 |
| soap+2b δ 0.3, M 500 | 3e-3 | 1.10 | 1.09 | 1.00 % | 44000 | 71 | 1508 |
| soap+2b δ 0.3, ξ [2, 4] | 3e-3 | 1.18 | 1.20 | — | 44000 | 59 | 1660 |
| soap+2b raw kernel (un-normalised q⁽²⁾), δ 0.1 / 0.3 | 2e-3 / 3e-3 | 1.17 / 1.16 | 1.07 / 1.07 | 1.00 / 1.01 % | 22000 | 66 | 1468 |
| soap, gap_extra_rcut 4.5 | 1e-3 / 3e-3 | 1.24 / 1.18 | 1.33 / 1.21 | — | 22000 | 80 | 2244 |
| soap, gap_extra_rcut 6.0 | 1e-3 / 3e-3 | 1.60 / 1.48 | 1.55 / 1.43 | — | 22000 | 97 | 2679 |
| soap+2b δ 0.3, gap_extra_rcut 4.5 | 5e-3 (3e-3) | 1.04 (1.07) | 1.10 (1.12) | 1.00 % | 33000 | 92 | 2228 |
| soap+2b δ 0.3, gap_site_rcut 5.0, gap_extra_rcut 4.5 | 3e-3 | 1.31 | 1.36 | — | 33000 | 99 | 2699 |

α (base = soap, l=α=7 trivial, rc 4.0, cross C_x 4; σ_A = 0.1):

| tag | σ | val (400) | test (400) | test (500, selected σ) | ncols | fit s | µs/atom |
|---|---|---|---|---|---|---|---|
| soap | 0.1 | 15.1/17.6 | 15.3/16.3 | 14.9/16.0 | 38000 | 40 | 764 |
| soap+2b δ 0.1 | 0.1 | 16.2/17.8 | 16.6/16.6 | — | 76000 | 50 | 929 |
| soap+2b δ 0.3, gap_site_rcut 6.0 | 0.1 | 16.0/18.0 | 16.1/16.2 | — | 76000 | 71 | 1796 |
| soap, gap_extra_rcut 5.5 | 0.1 / 0.3 | 14.7/17.1 / 14.1/17.5 | 15.2/16.0 / 14.6/16.6 | 14.2/16.1 (σ 0.3) | 76000 | 101 | 1935 |
| soap, gap_extra_rcut 6.5 | 0.1 / 0.3 / 0.5 | 14.4/17.9 / 14.0/18.1 / 14.1/19.2 | 15.3/16.1 / 14.6/16.7 / 14.7/17.7 | **13.9/16.2** (σ 0.3); 14.1/17.2 (σ 0.5) | 76000 | 137 | 2444 |
| soap, gap_extra_rcut 7.5 | 0.3 | 13.4/18.1 | 14.7/16.7 | 14.0/16.2 | 76000 | 141 | 3034 |
| soap+2b δ 0.3, gap_site_rcut 6.0, gap_extra_rcut 5.5 | 0.1 | 16.1/17.6 | 15.8/16.0 | — | 114000 | 118 | 3056 |

Reading: the 2b gain is a small-amplitude short-range prior — it wants δ_2b ≤ 0.1, the readout cutoff (every longer `gap_site_rcut` is monotonically worse), 10 channels, the normalised kernel, and it is saturated at M 250; multiscale does not help μ alone (the Ewald field already carries the long range) and does not stack with 2b on test. For α the opposite: 2b never helps, a second SOAP scale lifts the diagonal by a point (saturating at 6.5 Å; σ moves 0.1 → 0.3) and leaves the off-diagonal alone — the anisotropy stays a form limit (§4.3 of the docs). Logs: out/probe_{mu,pol}.log, out/probe2_{mu,pol}.log; records out/probe_{mu,pol}/GAP/.

### gap_2b_probe.py follow-up 2 (2026-10-20): a TNEP-like long READOUT basis (`dipole_rc`), 2b descriptor pinned at 3 Å (`gap_site_rcut=3.0`), ± Ewald field

Question: the published TNEP reads the dipole out through the derivative of a learned 6 Å two-body basis; does a 6 Å bond basis in the GAP recover the field column's gain, or add to it? σ_A ∈ {2e-3, 3e-3}; all rows soap+2b δ_2b 0.1 unless listed; "train" = RRMSE on the 500 training frames of the refit.

| readout r_c, C | field | val (400) | test (400) | train (500) | test (500) | ncols | fit s | µs/atom |
|---|---|---|---|---|---|---|---|---|
| 3.0, 10 (best model) | on | 1.07 | 1.05 | 0.60 | **0.97 %** | 22000 | 76 | 1468 |
| 4.5, 10 | on | 1.26 | 1.19 | 0.43 | 1.11 % | 22000 | 66 | 1481 |
| 6.0, 10 | on | 1.64 | 1.56 | 0.24 | 1.41 % | 22000 | 98 | 2278 |
| 6.0, 20 | on | 1.40 | 1.53 | 0.03 | 1.46 % | 42000 | 102 | 2256 |
| 3.0, 10 | off | 1.64 | 1.56 | 1.05 | 1.49 % | 20000 | 22 | 452 |
| 6.0, 10 | off | 1.67 | 1.56 | 0.26 | 1.42 % | 20000 | 54 | 1211 |
| 6.0, 20 | off | 1.47 | 1.58 | 0.04 | 1.50 % | 40000 | 59 | 1264 |
| 6.0, 10, soap only | off | 2.03 | 2.00 | 0.59 | 1.89 % | 10000 | 52 | 1151 |

Reading: widening the readout is monotonically worse (0.97 → 1.11 → 1.41 %) and the training error collapses (0.60 → 0.03 %): with independent site functions per (channel, species) the 6 Å bond basis has enough freedom to interpolate 500 frames and generalises worse. At 6 Å the field column stops mattering (1.41 with, 1.42 without), i.e. the long bond basis can fit what the field explains on the training set but not transferably. The TNEP's 6 Å readout works because its site functions are tied — all channels are ∂U_i/∂q of one 1097-parameter network — not because of the range itself; in a kernel model long range has to enter as a fixed physics column with one site function (the field). Cost: the Ewald field column is ≈ 1.0 ms/atom of the 1.47 ms/atom forward pass (452 µs/atom without it) — the single largest cost item in the best dipole model, and a pure-Python Ewald sum.

### gap_pol_groups_scan.py results (2026-10-20; fit 400 → val/test at σ ∈ {0.1, 0.3}, then refit on 500 → test; base = two bond groups + cross C_x 4, l=α=7 trivial, rc 4.0, M 250; SPC/E charges, α_O 1.45 Å³)

| model | fit 400 test σ 0.1 | fit 400 test σ 0.3 | fit 500 test σ 0.1 | fit 500 test σ 0.3 | ncols | µs/atom |
|---|---|---|---|---|---|---|
| base | 15.3/16.3 | 15.1/17.7 | 14.9/16.0 | | 38000 | 857 |
| +induced | 14.5/14.9 | 14.4/16.5 | **14.0/14.6** | | 38500 | 802 |
| +grad | 15.3/16.2 | 15.1/17.5 | | | 38500 | |
| +cross | 15.4/16.4 | 14.7/17.4 | | | 42000 | |
| +quad | 15.0/16.2 | 14.9/17.5 | | | 39000 | |
| +qeq | 15.4/16.4 | 14.9/17.8 | | | 39000 | |
| +induced+quad | 14.2/14.8 | 14.1/16.2 | 13.9/14.5 | | 39500 | 821 |
| +induced+qeq | 14.6/15.0 | 14.1/16.5 | | | 39500 | |
| +quad+qeq | 15.3/16.3 | 14.8/17.5 | | | 40000 | |
| +all five | 14.4/14.8 | 13.7/15.9 | 13.9/14.3 | | 45000 | 837 |
| base, gap_extra_rcut 6.5 | | | 14.7/15.6 | 13.9/16.2 | 76000 | 2500 |
| +induced, gap_extra_rcut 6.5 | | | 14.0/14.7 | 13.3/15.5 | 77000 | 2500 |
| +induced+quad, gap_extra_rcut 6.5 | | | 13.9/14.6 | **13.0**/15.2 | 79000 | 2571 |

Val selected σ 0.1 for every off-diagonal and 0.3 for every diagonal. Log: out/polgroups.log. Not run: all five on the two-scale model (90 000 columns; the process took WSL down), and the gap_lambda_shear rows meant to decouple the two σ preferences (second crash before its first fit; cause not yet identified — the solver is dual, so it is not the column count).

### Triplet-form α groups (2026-10-21; C_x = 2 base = two bond groups + cross; fit 400 → val/test at σ 0.1 / 0.3; refit 500 → test)

| model | val σ 0.1 | test σ 0.1 (400) | test σ 0.3 (400) | test (500, σ 0.1) | ncols | µs/atom |
|---|---|---|---|---|---|---|
| 2groups cross2 (reference) | 15.3/19.0 | 15.5/17.2 | 15.4/18.7 | 15.1/16.8 | 25000 | 658 |
| + angular cross sym(w⊗v), L 1 | 14.7/18.4 | 15.6/16.8 | 15.4/18.2 | | 33000 | 752 |
| + angular cross, L 2 | 14.7/18.2 | 15.3/16.7 | 15.2/18.1 | (killed by the watchdog at 11.4 GB in the long process) | 41000 | 844 |
| + angular cross L 1 + induced | 13.5/17.6 | 14.8/15.5 | 14.8/17.0 | 14.4/15.2 | 33500 | 1703 |
| + triplet tensor M10 σ 0.5 exp | 15.0/19.0 | 15.9/17.4 | 15.6/18.5 | 15.5/17.0 | 35000 | 897 |
| + triplet tensor M20 σ 0.5 exp | 15.3/18.9 | 16.6/17.8 | 15.9/17.9 | 16.3/17.7 | 45000 | 1053 |
| + triplet tensor M20 σ 1.0 exp | 18.3/20.2 | 18.1/18.7 | 16.8/17.8 | | 45000 | 1096 |
| + triplet tensor M20 pol σ 2 | 15.7/19.2 | 16.8/17.9 | 16.0/17.9 | | 45000 | 1644 |
| + triplet tensor M20 + induced | 13.9/18.1 | 15.7/16.5 | 15.1/16.8 | 15.5/16.5 | 45500 | 1932 |
| + triplet tensor M20 + angular cross L1 | 15.6/18.7 | 17.3/17.6 | 16.4/17.5 | | 53000 | 1159 |

"bond-off" rows (bond groups at prior amplitude 1e-6) are invalid — the column scaling and prior scaling together blow the whitening up (4600 % diagonal) — and are omitted. Logs: out/angcross.log, out/triplet.log.
