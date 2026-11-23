# GAP dipole / polarisability experiment drivers (2026-10-19)

> 2026-11-11: the scan scripts below were pruned; only `gap_scan.py` (the former
> `gap_progressive_scan.py`: production rows and ladders, `PS_*` env protocol), its driver
> `gap_scan_parallel.sh`, `memguard.sh` and `prog_tables.py` remain. Rows naming other
> scripts are a record of results, not runnable.

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
| gap_progressive_scan.py + gap_progressive_driver.sh (`mu` / `pol`; `ROWS_OVERRIDE`, `FINALS_ONLY`; 8 GB watchdog, run_all_sequential.sh / chain_*.sh for one-job-at-a-time orchestration) | progressive (stage-wise residual) GAP `gap_dipole.fit_gap_progressive` vs the summed fit, in both orders; the dipole T·v cross group (`gap_cross_channels`, the sym(v⊗v) analogue for a vector) on the best LOCAL dipole model; the 2b Chebyshev site model on the local α base. No Ewald columns (user). | μ: cross C_x=2 on soap+2b 1.49 → **1.45 %** (C_x=4 1.48; prior amplitude 0.3 same 1.45; on SOAP alone null 1.56 → 1.56). Progressive is never better than summed: 2b→soap 1.69–1.96, soap→2b 1.52–1.66 (best σ_2b 0.03–0.1, between soap-only 1.56 and summed 1.49), any 3b stage 1.87–1.91. α: 2b→soap 19.5–28.1/16.5–21.1 (catastrophic), soap→2b converges to the base from above (σ_2b 1: 15.2/15.9 vs 14.9/16.0), soap→3b 15.4/16.7; round 2 below |

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

### gap_progressive_scan.py results (2026-10-26; fit 400 → val/test for σ selection, refit 500 → test; no Ewald columns)

Progressive = `fit_gap_dipole(cfg, mu_offset)` per stage in `gap_site_models` order, each stage on the residual of the
previous ones (own σ via `fit_gap_progressive(cfg, sigma={model: σ})`), prediction = stage sum. "sX" is the σ of the
non-SOAP stage(s); the row σ is the SOAP stage's.

μ (base = best local model, docs §3.2: SOAP-7 uncompressed, rc 3.0, cosine², C 10, M 250, soap+2b δ_2b 0.1; μ SOAP-only 1.56 %):

| row | σ (soap) | val 400 | test 400 | test 500 | ncols |
|---|---|---|---|---|---|
| summed soap+2b δ0.1 (reference) | 3e-3 | 1.64 | 1.56 | **1.49** | 20000 |
| + cross C_x=2 (T·v, `gap_cross_channels=2`) | 3e-3 | 1.63 | 1.52 | **1.45** | 36000 |
| + cross C_x=4 | 3e-3 | 1.64 | 1.55 | 1.48 | 84000 |
| + cross C_x=2, `gap_group_delta=[1,1,0.3]` | 3e-3 | 1.63 | 1.53 | 1.45 | 36000 |
| SOAP + cross C_x=2 (no 2b) | 1e-3 | 1.85 | 1.69 | 1.56 | 18000 |
| prog 2b→soap s0.003 / 0.01 / 0.03 | 1e-3–3e-3 | 2.05 / 1.91 / 1.81 | 1.94 / 1.82 / 1.74 | — / — / 1.69 | 20000 |
| prog 2b→3b→soap s0.01 / 0.03 (C_x 4, l_x 1) | 1e-3 / 3e-3 | 2.01 / 1.79 | 1.98 / 1.71 | 1.88 / 1.66 | 116000 |
| prog soap→2b s0.001 / 0.003 / 0.01 / 0.03 / 0.1 | 1e-3–3e-3 | 1.86 / 1.78 / 1.71 / 1.68 / 1.72 | 1.76 / 1.69 / 1.62 / 1.60 / 1.60 | 1.66 / 1.61 / 1.55 / **1.52** / 1.53 | 20000 |
| prog soap→3b s0.003 (C_x 4, l_x 1) | 1e-3 | 2.03 | 1.96 | 1.87 | 74000 |
| prog soap→2b→3b s0.003 | 1e-3 | 2.03 | 1.99 | 1.91 | 116000 |

α (base = docs §4.2 "base": SOAP-7 trivial, rc 4.0, cosine, C 10, M 250, cross C_x 4; diag/off-diag RRMSE):

| row | σ (soap) | val 400 | test 400 | test 500 | ncols |
|---|---|---|---|---|---|
| base (reference) | 0.1 | 15.1/17.6 | 15.3/16.3 | **14.9/16.0** | 38000 |
| prog 2b→soap s0.1 / 0.3 / 1 | 0.1 | 27.0/23.5 / 22.9/20.3 / 20.1/18.4 | 29.7/22.4 / 24.9/18.8 / 21.4/17.1 | 28.1/21.1 / 22.6/17.8 / 19.5/16.5 | 76000 |
| prog 2b→3b→soap s0.3 (l_x 1) | 0.3 | 22.9/20.5 | 25.0/19.4 | 23.0/18.7 | 112000 |
| prog soap→2b s0.1 / 0.3 / 1 | 0.1 | 17.5/21.4 / 16.3/18.8 / 15.5/17.7 | 18.0/20.3 / 16.5/17.4 / 15.7/16.4 | 17.3/19.1 / 15.8/16.7 / 15.2/15.9 | 76000 |
| prog soap→3b s0.3 (l_x 1) | 0.1 | 15.3/17.5 | 15.9/17.2 | 15.4/16.7 | 74000 |
| (summed soap+2b, from docs §4.4/§7, not rerun: 77k columns) | 0.3 | 16.2/17.8 (δ 0.1) | | 16.1/17.0 (δ 0.3) | 76000 |
| round 2: prog soap→2b s1 C4 / s1 rc3 / s3 (smoother, shorter, weaker residual 2b stage) | 0.1 | 15.0/17.5 / 15.7/17.5 / 15.2/17.6 | 15.3/16.3 / 15.7/16.5 / 15.4/16.3 | 14.9/15.9 / 15.3/16.2 / 15.0/16.0 — all within noise of the base | 76000 |

Reading: a body-order stage fitted FIRST takes a smooth component the SOAP stage then cannot undo (the 2b stage has no
anisotropy, so for α it destroys the diagonal: 29.7 → 21.4 as it is weakened); fitted LAST on the SOAP residual it is a
weak learner that converges to SOAP-only from above and never reaches the summed fit — the summed fit's gain is the joint
solve using the 2b kernel as a second regulariser, not a two-body model. Any 3b residual stage interpolates (in-sample
0.2–0.5 %, test 1.9 %). The dipole T·v cross group is the one positive: a small, repeatable 1.49 → 1.45 % on the summed
local model (nothing on SOAP alone, so it works with the 2b prior), saturating at two channels.

### gap_progressive_scan.py rounds 1–3 (2026-10-27): low-M 2b/3b stages, grouped stages, 2b/3b/cross parameter search (no Ewald)

Driver `gap_scan_parallel.sh <rows file>` (two fits at a time under `memguard.sh`, 7 GB per-fit watchdog; rows listed by
`PS_LIST=1`), tables by `prog_tables.py`. New fit features used here: `gap_sparse_per_species` as a dict (per site model
FPS size; "M40" = 40 points for the 2b/3b model, SOAP keeps 250) and `fit_gap_progressive(stages=[[...], [...]])` (a stage
may be a summed group, e.g. soap+2b jointly, then 3b on the residual; σ keyed by the stage name). Bases as the previous
section: μ soap+2b δ_2b 0.1 (1.49 %), α two bond groups + cross C_x 4 (14.9/16.0). `ncols` of progressive rows is
unreliable (parallel fits share the record directory); metrics are computed in-process and exact.

**rows_round1.txt — mu** (RRMSE %; σ selected on val 400/100, then refit on 500)

| row | σ | val (400) | test (400) | test (500) | ncols |
|---|---|---|---|---|---|
| soap+2b d0.1 +cross2 dx0.3 | 0.003 | 1.63 | 1.53 | 1.45 | 36000 |
| prog soap>2b(M40) s0.003 | 0.003 | 1.76 | 1.68 | 1.60 | 11600 |
| prog soap>2b(M40) s0.01 | 0.003 | 1.71 | 1.62 | 1.54 | 3200 |
| prog soap>2b(M40) s0.03 | 0.003 | 1.69 | 1.60 | 1.52 | 13200 |
| prog 2b(M40)>soap s0.01 | 0.003 | 1.88 | 1.82 | 1.74 | 11600 |
| prog 2b(M40)>soap s0.03 | 0.003 | 1.80 | 1.74 | 1.68 | 11600 |
| prog 2b(M40)>soap s0.1 | 0.003 | 1.83 | 1.72 | 1.66 | 11600 |
| prog soap+2b>3b(M40) s0.01 | 0.001 | 1.72 | 1.62 | 1.54 | 32160 |
| prog soap+2b>3b(M40) s0.03 | 0.003 | 1.87 | 1.75 | 1.69 | 44320 |
| prog soap+2b>3b(M40) s0.1 | 0.001 | 2.32 | 2.15 | 2.08 | 46160 |
| soap+2b(M40) d0.1 | 0.003 | 1.64 | 1.56 | 1.49 | 11600 |
| soap+2b(M40) d0.3 | 0.003 | 1.69 | 1.62 | 1.53 | 11600 |
| soap+2b(M40) d1 | 0.003 | 1.77 | 1.71 | 1.61 | 22160 |
| soap+2b+3b(M40) d3b0.1 L1 C6 | 0.003 | 1.78 | 1.58 | 1.49 | 22160 |
| soap+2b+3b(M40) d3b0.1 L1 C10 | 0.003 | 1.65 | 1.53 | 1.46 | 1280 |
| soap+2b+3b(M40) d3b0.1 L2 C6 | 0.003 | 1.78 | 1.58 | 1.50 | 22160 |
| soap+2b+3b(M40) d3b0.1 L2 C10 | 0.003 | 1.65 | 1.53 | 1.46 | 22160 |
| soap+2b+3b(M40) d3b0.3 L1 C6 | 0.003 | 1.77 | 1.60 | 1.53 | 22160 |
| soap+2b+3b(M40) d3b0.3 L1 C10 | 0.003 | 1.71 | 1.57 | 1.49 | 22160 |
| soap+2b+3b(M40) d3b0.3 L2 C6 | 0.003 | 1.78 | 1.60 | 1.53 | 22160 |
| soap+2b+3b(M40) d3b0.3 L2 C10 | 0.003 | 1.71 | 1.57 | 1.50 | 22160 |
| soap+2b d0.1 +cross1 dx0.3 | 0.003 | 1.64 | 1.54 | 1.47 | 56000 |
| soap+2b d0.1 +cross1 dx1 | 0.003 | 1.64 | 1.54 | 1.47 | 1600 |
| soap+2b d0.1 +cross1 dx3 | 0.003 | 1.63 | 1.56 | 1.49 | 24000 |
| soap+2b d0.1 +cross2 dx1 | 0.003 | 1.63 | 1.52 | 1.45 | 10000 |
| soap+2b d0.1 +cross2 dx3 | 0.003 | 1.65 | 1.55 | 1.49 | 36000 |
| soap+2b d0.1 +cross3 dx0.3 | 0.003 | 1.63 | 1.53 | 1.45 | 20880 |
| soap+2b d0.1 +cross3 dx1 | 0.003 | 1.63 | 1.54 | 1.47 | 56000 |
| soap+2b d0.1 +cross3 dx3 | 0.003 | 1.65 | 1.60 | 1.55 | 56000 |

**rows_round1.txt — pol** (diag/off RRMSE %; σ selected on val 400/100, then refit on 500)

| row | σ | val (400) | test (400) | test (500) | ncols |
|---|---|---|---|---|---|
| prog soap>2b(M40) s0.3 | 0.1 | 16.2/18.7 | 16.4/17.4 | 15.7/16.6 | 49840 |
| prog soap>2b(M40) s1 | 0.1 | 15.5/17.7 | 15.7/16.4 | 15.2/16.0 | 44080 |
| prog soap>2b(M40) s3 | 0.1 | 15.2/17.6 | 15.4/16.3 | 15.0/16.0 | 49840 |
| prog 2b(M40)>soap s1 | 0.1 | 20.6/18.4 | 22.1/17.1 | 20.3/16.6 | 82080 |
| prog 2b(M40)>soap s3 | 0.1 | 18.0/18.3 | 20.3/16.8 | 18.8/16.5 | 84640 |
| prog soap>3b(M40) s0.3 | 0.1 | 14.9/17.9 | 15.7/16.9 | 15.2/16.2 | 52400 |
| prog soap>3b(M40) s1 | 0.1 | 15.0/17.4 | 15.3/16.1 | 14.8/15.7 | 49520 |
| soap+2b(M40) d0.03 | 0.1 | 15.2/17.6 | 15.6/16.3 | 15.1/15.9 | 44080 |
| soap+2b(M40) d0.1 | 0.1 | 16.2/17.8 | 16.6/16.6 | 15.8/16.0 | 44080 |
| soap+2b(M40) d0.3 | 0.3 | 16.7/18.8 | 16.8/17.5 | 16.1/17.0 | 38000 |
| soap+3b(M40) d3b0.1 lx1 L1 | 0.1 | 15.1/17.2 | 15.3/16.1 | 14.8/15.5 | 43760 |
| soap+3b(M40) d3b0.1 lx1 L2 | 0.1 | 15.2/17.0 | 15.3/16.0 | 14.7/15.5 | 43760 |
| soap+3b(M40) d3b0.1 lx2 L1 | 0.1 | 14.9/16.6 | 15.2/15.8 | 14.6/15.3 | 44080 |
| soap+3b(M40) d3b0.1 lx2 L2 | 0.1 | 15.0/16.4 | 15.2/15.8 | 14.6/15.3 | 5760 |
| soap+3b(M40) d3b0.3 lx1 L1 | 0.3 | 14.7/18.1 | 14.8/16.8 | 14.5/16.2 | 25000 |
| soap+3b(M40) d3b0.3 lx1 L2 | 0.3 | 14.8/18.0 | 14.8/16.6 | 14.4/16.2 | 43760 |
| soap+3b(M40) d3b0.3 lx2 L1 | 0.3 | 14.5/17.4 | 14.8/16.5 | 14.4/16.1 | 46640 |
| soap+3b(M40) d3b0.3 lx2 L2 | 0.3 | 14.5/17.3 | 14.7/16.4 | 14.3/16.0 | 46640 |
| base cross2 dx0.3 | 0.1 | 15.4/19.3 | 15.6/17.4 | 15.2/17.0 | 25000 |
| base cross2 dx1 | 0.1 | 15.3/19.0 | 15.5/17.2 | 15.1/16.8 | 25000 |
| base cross2 dx3 | 0.1 | 15.2/18.9 | 15.5/17.0 | 15.1/16.7 | 25000 |
| base cross4 dx0.3 | 0.1 | 15.1/17.9 | 15.4/16.7 | 14.9/16.3 | 38000 |
| base cross4 dx1 | 0.1 | 15.1/17.6 | 15.3/16.3 | 14.9/16.0 | 38000 |
| base cross4 dx3 | 0.1 | 15.1/17.8 | 15.4/16.5 | 14.9/16.0 | 6080 |
| base cross6 dx1 | 0.1 | 15.0/17.8 | 15.3/16.5 | 14.8/16.1 | 43760 |

**rows_round2_new.txt — mu** (RRMSE %; σ selected on val 400/100, then refit on 500)

| row | σ | val (400) | test (400) | test (500) | ncols |
|---|---|---|---|---|---|
| soap+2b(M40) d0.1 +cross1 | 0.003 | 1.64 | 1.54 | 1.47 | 13920 |
| soap+2b(M40) d0.1 +cross2 | 0.003 | 1.63 | 1.52 | 1.45 | 20880 |
| soap+2b(M20) d0.1 +cross2 | 0.003 | 1.62 | 1.51 | 1.44 | 22160 |
| soap+2b(M80) d0.1 +cross2 | 0.003 | 1.63 | 1.52 | 1.45 | 20880 |
| soap+2b(M40) d0.05 +cross2 | 0.003 | 1.62 | 1.50 | 1.44 | 20880 |
| soap+2b(M40) d0.2 +cross2 | 0.003 | 1.67 | 1.57 | 1.48 | 20880 |
| prog soap+2b>3b(M40) s0.001 | 0.001 | 1.82 | 1.69 | 1.62 | 43680 |
| prog soap+2b>3b(M40) s0.003 | 0.001 | 1.72 | 1.57 | 1.52 | 23440 |
| prog 3b(M40)>soap+2b s0.03 | 0.001 | 2.90 | 2.61 | 2.55 | 43040 |
| prog 3b(M40)>soap+2b s0.1 | 0.001 | 3.96 | 3.53 | 3.41 | 23440 |
| soap+2b+3b(M40) d3b0.03 L1 C10 | 0.003 | 1.64 | 1.52 | 1.45 | 22160 |
| soap+2b+3b(M20) d3b0.1 L1 C10 | 0.003 | 1.65 | 1.53 | 1.45 | 21520 |
| soap+2b+3b(M20) d3b0.03 L1 C10 | 0.003 | 1.64 | 1.52 | 1.45 | 20880 |

**rows_round2_new.txt — pol** (diag/off RRMSE %; σ selected on val 400/100, then refit on 500)

| row | σ | val (400) | test (400) | test (500) | ncols |
|---|---|---|---|---|---|
| soap+3b(M40) d3b0.03 lx2 C10 | 0.1 | 15.0/16.9 | 15.1/15.8 | 14.6/15.3 | 46640 |
| soap+3b(M40) d3b0.1 lx3 C10 | 0.1 | 14.9/16.5 | 15.2/15.9 | 14.6/15.3 | 8640 |
| soap+3b(M40) d3b0.03 lx3 C10 | 0.1 | 15.0/16.8 | 15.1/15.7 | 14.6/15.2 | 46640 |
| soap+3b(M20) d3b0.1 lx2 C10 | 0.1 | 15.0/16.9 | 15.2/15.9 | 14.7/15.4 | 42320 |
| soap+3b(M80) d3b0.1 lx2 C10 | 0.1 | 14.8/16.3 | 15.1/15.8 | 14.6/15.4 | 41040 |
| soap+3b(M40) d3b0.1 lx2 C6 | 0.1 | 14.6/15.8 | 14.9/15.3 | 14.4/14.9 | 42320 |
| soap+3b(M40) d3b0.3 lx2 C10 | 0.3 | 14.5/17.4 | 14.8/16.5 | 14.4/16.1 | 46640 |
| prog soap>3b(M40) lx2 s1 | 0.1 | 14.8/16.8 | 15.2/15.9 | 14.7/15.5 | 58160 |
| prog soap>3b(M40) lx2 s3 | 0.1 | 15.0/17.0 | 15.2/16.0 | 14.7/15.6 | 46640 |
| soap+2b(M20) d0.03 | 0.1 | 15.2/17.6 | 15.6/16.3 | 15.0/16.0 | 41040 |
| soap+2b(M40) d0.01 | 0.1 | 15.0/17.6 | 15.3/16.3 | 14.9/16.0 | 46640 |
| soap+2b(M40)+3b(M40) d2b0.03 d3b0.1 lx2 | 0.1 | 15.1/16.7 | 15.5/15.9 | 14.9/15.2 | 8640 |

**rows_round3.txt — mu** (RRMSE %; σ selected on val 400/100, then refit on 500)

| row | σ | val (400) | test (400) | test (500) | ncols |
|---|---|---|---|---|---|
| soap+2b(M40)+3b(M40) d3b0.1 L1 C6 lx2 cross2 | 0.003 | 1.78 | 1.59 | 1.51 | 23440 |
| soap+2b(M40)+3b(M40) d3b0.1 L1 C6 lx2 cross1 | 0.003 | 1.78 | 1.59 | 1.52 | 2560 |
| prog soap+2b>3b(M40) C6 lx2 s0.01 | 0.003 | 1.90 | 1.72 | 1.65 | 38000 |

**rows_round3.txt — pol** (diag/off RRMSE %; σ selected on val 400/100, then refit on 500)

| row | σ | val (400) | test (400) | test (500) | ncols |
|---|---|---|---|---|---|
| soap+3b(M40) d3b0.1 lx2 C4 | 0.1 | 14.6/15.9 | 14.8/15.1 | 14.4/14.7 | 46640 |
| soap+3b(M40) d3b0.15 lx2 C6 | 0.1 | 14.5/15.7 | 14.8/15.3 | 14.4/14.9 | 46640 |
| soap+3b(M40) d3b0.07 lx2 C6 | 0.1 | 14.7/16.1 | 14.9/15.3 | 14.5/15.0 | 46640 |
| soap+3b(M80) d3b0.1 lx2 C6 | 0.1 | 14.5/15.8 | 14.8/15.2 | 14.4/14.8 | 77720 |
| soap+3b(M40) d3b0.1 lx3 C6 | 0.1 | 14.6/15.6 | 14.8/15.1 | 14.4/14.7 | 46640 |
| soap+3b(M40) d3b0.1 lx2 C6 L2 | 0.1 | 14.7/16.0 | 14.9/15.3 | 14.4/14.9 | 46640 |
| soap+3b(M40) d3b0.1 lx2 C6 cross6 | 0.1 | 14.5/16.0 | 14.8/15.7 | 14.3/15.1 | 46640 |
| prog soap>3b(M40) lx2 C6 s1 | 0.3 | 14.1/16.8 | 14.5/16.1 | 14.2/15.7 | 55280 |


**Reading.**
- *Progressive fits (all orderings, M 40 or 250, grouped stages):* never better than the summed fit. μ: best 1.52 (soap→2b(M40),
  σ_2b 0.03) vs 1.45–1.49 summed; soap+2b jointly then 3b on the residual 1.52–1.65 (and 1.62 at σ_3b 0.001, i.e. it only
  gets worse as the stage is allowed to fit); 3b first 2.5–3.4; 2b first 1.66–1.74. α: soap→3b(M40) 14.7/15.5 (l_x 2, C 10)
  and 14.2/15.7 (C 6, σ 0.3) — the progressive 3b stage captures part of the gain of the summed 3b model but not the
  off-diagonal; soap→2b(M40) is neutral (15.0/16.0), 2b first 18.8–20.3 diag.
- *2b term:* 40 (even 20) sparse points reproduce the 250-point 2b model exactly for μ (1.49; with cross 1.44–1.45) at a fifth
  of the columns; δ_2b 0.05–0.1 is the plateau, 0.2 and above worse. For α the 2b model is neutral at δ ≤ 0.03 and harmful
  above, at any M.
- *3b term:* for α it is the result of this scan — SOAP + 3b site model with M 40, δ_3b 0.1, l_x 2 (or 3), 4–6 channels:
  **14.4/14.7–14.9** (fit 500 → test) vs 14.9/16.0, the first local term to move the off-diagonal (the M 250 version in the
  earlier scan was 14.9/17.0: the 3b kernel needs few sparse points, with 250 it interpolates). δ_3b 0.03 and 0.3, M 20/80,
  site_lmax 2, 10 channels and adding cross C_x 6 are all within ±0.2 of it. For μ the same term is null to slightly harmful
  (1.45–1.46 with the T·v cross group present, 1.51 with the α-optimal C 6 / l_x 2 setting).
- *Cross group (μ):* C_x 1/2/3 at δ_x 0.3–1 are all 1.44–1.47 (floor 1.45); δ_x 3 hurts (1.49–1.55); on SOAP alone it is null.
  (α): C_x 2/4/6 × δ_x 0.3–3 move the base by ±0.3 at most; C_x 4, δ_x 1 remains the base.
- *Cheapest models:* μ soap + 2b(M20–40, δ 0.05–0.1) + cross C_x 2 → 1.44–1.45 % at ~24k columns; α soap + 3b(M40, δ 0.1,
  l_x 2, C 4) → 14.4/14.7 at ~47k columns.

### Raman footing and intensity balance (2026-10-27): why the <1000 cm⁻¹ bands looked too strong, settled with the authors' own TNEP model

Question: on the local α GAP the anisotropic libration/H-bond bands were ~2× the bend and the OH stretch only ×14 the
low-frequency maximum, where Xu et al. 2024 Fig. 4 shows bend ≈ libration and a ×25 axis break. Not statistics (40 ps
blocks agree to 2–4 %; ACF truncation 2–20 ps changes nothing). The paper writes Eq. 12 without any ω factor and
publishes no analysis script, so the footing had to be inferred. Decisive test: their TNEP susceptibility model
(Zenodo 10257363, `nep_susceptibility_water_bulk.txt`, trained on the SAME 500+500 frames as `datasets/*_waterbulk.xyz`)
evaluated on the first 20 ps (10 000 frames, stride 2) of our trajectory with the local GPUMD binary in prediction mode
(`scripts/eval_tnep_xu2024_on_traj.sh`; ~1 s per 2500 frames on the GPU). Band ratios of that series under each
candidate weighting (peak heights, 5 ps Hann-truncated ACF, 8 cm⁻¹ Gaussian smoothing):

| footing | aniso: stretch ÷ max(80–2500) | aniso lib ÷ bend | aniso H-bond ÷ lib | iso: stretch ÷ bend | iso lib ÷ bend |
|---|---|---|---|---|---|
| none (bare L, calorine) | 1.1 | 12 | — | 49 | — |
| harmonic ω(1−e^(−x))·L (old "reduced" default) | 8.7 | 3.1 | 0.81 | 96 | 0.27 |
| **classical ω²·L (Xu Eq. 9)** | **46** | **1.2** | **0.42** | **188** | **0.15** |
| Fig. 4 TNEP curve, read off the zoomed PDF | ≈47 (3.0 × 25 ÷ 1.6) | ≈1.1 | ≈0.45 | ≈150 (×180 label) | ≈0.2 |

Only ω² reproduces the figure, on every ratio, so Fig. 4 is on the classical ω²·L footing (the same prefactor as
their IR Eq. 9). `compute_raman_spectrum` now defaults to `quantum_correction="classical"`, and IR and Raman share one
`_quantum_prefactor` with the IR vocabulary (classical / harmonic / linear / none; the calorine x/(1−e^(−x)) option is
gone). Plots: `plots/water_bulk_GPUMD_traj_polarizabilities_{gap_local,tnep_xu2024_seg20ps}_classical_raman_{iso,aniso}[_split2500]_*.png`.

GAP vs TNEP on the identical 20 ps frames (ω²): per-component correlation 0.95–0.98 of the mean-free series.
GAP: aniso stretch ×72 (200 ps: ×67), lib/bend 0.99, H-bond/lib 0.56, iso stretch/bend 193, iso lib/bend 0.24.
TNEP: ×46, 1.20, 0.42, 188, 0.15. The experimental curves in Fig. 4 read the same way give aniso stretch ≈ ×80–120 of
the libration (Brooker ≈ 83, Morawietz ≈ 120; baselines of the offset curves are approximate), so the GAP's stretch
weight is the closer one. The difference between the models is in the diffuse intermolecular part (<1000 cm⁻¹), not
in the per-frame response: `raman_response_probe.py` regresses test-set reference and predicted α on
bond-polarisability features (stretch Σδr(ûû−I/3), rigid Σ(ûû−I/3), isotropic Σδr and Σδθ); intensity ∝ slope².

**Label convention found on the way.** `datasets/{train,test}_waterbulk.xyz` are byte-identical (md5) to
`dipole_bulk_water/{train,test}.xyz` in the Zenodo record — the DIPOLE archive, whose `pol` column is
(3V/4π)(ε−1)(ε+2)⁻¹ / 1.13205 (Clausius–Mossotti form, reproduced component-wise to 4e-5; the constant is not
explained anywhere in the record). The SUSCEPTIBILITY archive (`susceptibility_water_bulk/`, the data the Raman model
`nep_susceptibility_water_bulk.txt` was trained on) holds the same 500+500 frames with `mu`, `epsilon` (= ε − 1) and
`pol` = V(ε−1)/4π exactly — the quantity the paper's Eq. 10–12 use for extended systems. Mean ratio 0.693, fluctuation slope 0.545
(= 3/(ε+2)² × 3 / 1.132). At 648 atoms the transform is linear (CM-transformed TNEP equals the GAP in mean and slope
to 1 %, band ratios unchanged), so for spectra it is a scale factor; for anything absolute (α per molecule, ε) it is the
wrong quantity. χ-label copies on the identical frames: `datasets/{train,test}_waterbulk_chi.xyz`.

**Response probe, α GAP variants (test set, 500 frames; slope_pred/slope_ref per feature; intensity ∝ slope²):**

| variant | RRMSE diag/off | stretch aniso | rigid aniso | stretch iso | bend iso | aniso (stretch/rigid)² | iso (stretch/bend)² |
|---|---|---|---|---|---|---|---|
| base (SOAP, cross 4, M 250, σ 0.1) | 14.9/16.0 | 0.994 | 0.971 | 1.003 | 0.989 | 1.05 | 1.03 |
| σ 0.03 | 16.6/16.0 | 0.995 | 0.977 | 1.003 | 0.959 | 1.04 | 1.09 |
| σ 0.3 | 14.9/17.4 | 0.990 | 0.963 | 1.011 | 1.021 | 1.06 | 0.98 |
| best3b (SOAP + 3b M40 δ0.1) | 14.4/14.7 | 0.998 | 0.976 | 1.003 | 0.989 | 1.04 | 1.03 |
| dipole_rc 2.5 | 23.0/32.7 | 1.011 | 0.935 | 1.005 | 1.016 | 1.17 | 0.98 |
| cross 0 | 16.6/19.9 | 0.994 | 0.973 | 1.011 | 0.982 | 1.05 | 1.06 |
| traceless prior 0.3 | 15.3/17.1 | 0.991 | 0.965 | 1.011 | 0.982 | 1.06 | 1.06 |
| χ labels, base | 17.0/15.9 | 0.995 | 0.976 | 0.995 | 0.938 | 1.04 | 1.13 |
| χ labels, best3b | 16.6/14.7 | 0.999 | 0.981 | 0.995 | 0.938 | 1.04 | 1.13 |
| TNEP (Xu et al.) on the χ test set | 16.3/20.4 | 0.997 | 0.996 | 0.998 | 0.971 | 1.00 | 1.06 |

Every variant reproduces the DFT response slopes to 1–4 % (rc 2.5 loses 6 % of the rigid response and a third of
its accuracy), so none of the model knobs moves the stretch/libration balance by more than ~5 % — there is nothing
to "fix" in the GAP at the per-frame level, and the band balance is set by the footing. **χ-label GAP, 20 ps
segment:** std per component equal to the TNEP's (1.99/1.89/1.93/1.56/1.48/1.22 vs 1.98/1.85/1.93/1.57/1.51/1.25),
corr 0.96–0.98, slope 0.97; off-diagonal series identical to the CM-label GAP (corr 1.000, slope 1.85 = the label
scale) — band ratios ×71 / lib-bend 0.98 / H-bond-lib 0.57 / iso 198, unchanged from the CM labels. The GAP–TNEP
difference lives in 100–1000 cm⁻¹: bare-L integrals of (GAPχ − TNEP) are 9–20 % of the TNEP's anisotropic
H-bond/libration bands and 100–140 % of its (tiny) isotropic intermolecular bands, but 1–2 % of the stretch. Which
model is right there needs reference data (below); the experimental Fig. 4 curves favour the GAP's higher stretch
weight. Plots: `plots/water_bulk_GPUMD_traj_polarizabilities_gapchi_seg20ps_classical_raman_{iso,aniso}[_split2500]_*.png`.

**IR footing, same test.** The authors' TNEP dipole model (`nep_dipole_bulk_water.txt`) on the same 20 ps, band
peaks relative to the libration (Fig. 4(a) TNEP curve read off the zoomed PDF: H-bond 0.35, bend 0.62, stretch 2.96):
classical ω²·M gives 0.34 / 0.62 / 3.13, harmonic ω(1−e^(−x))·M gives 0.76 / 0.27 / 0.66. Fig. 4(a) is on ω² too, as
the text says ("via Eq. 9"); `compute_ir_spectrum` keeps its "harmonic" default, which is the physically weighted
absorption, not the figure's footing. Series: `plots/water_bulk_GPUMD_traj_dipoles_tnep_xu2024_seg20ps.npy`.

**Experimental reference curves (2026-10-27).** Neither Brooker et al. 1989 (J. Raman Spectrosc. 20, 683) nor
Morawietz et al. 2018 (JPCL 9, 851) publishes numerical spectra (Brooker: no SI; Morawietz: methods-only SI PDF;
Pattenaude et al. 2018 digitised Brooker from the figures themselves). What exists is Xu et al.'s adaptation of both
in Fig. 4, and that figure is vector graphics in the arXiv PDF, so `scripts/digitise_xu2024_fig4.py` reads the curve
paths exactly (axis calibration from the tick marks; vertical offsets removed as the right-panel minimum; right panel
×25 / ×180 undone) into `datasets/raman_water_xu2024_fig4/<source>_<panel>.csv` for Brooker, Morawietz, TNEP, T-EANN, DP,
MB-pol (Raman) and Downing, Max, TNEP, DP, MB-pol, TIP3P (IR). Check: the digitised TNEP curve gives stretch/libration
46.7, iso stretch/bend 134 (the figure's ×180 label is the axis factor, not the peak ratio) — our evaluation of their
model on our trajectory gave 46.2 / 188 (the iso value is sensitive to the 1650 peak height, 2.5e-3 vs 1.9e-3 in the
overlay). `scripts/plot_raman_vs_experiment.py` → `plots/raman_water_vs_experiment_{aniso,iso}.png`, Xu et al.'s
presentation (∫80–2500 / ∫1000–2500 normalisation, break at 2500, right panel ÷25 / ÷180). Readings (anisotropic, ω², band
integrals relative to the bend band 1400–1900; H-bond 80–300 / libration 300–1000 / stretch 2800–3900):
Brooker 0.33 / 1.15 / 60, Morawietz 0.09 / 0.82 / 67, TNEP (figure) 0.20 / 1.86 / 78, TNEP model on our trajectory
0.33 / 2.19 / 80, **GAP 0.37 / 1.85 / 109**. The GAP matches the TNEP on libration/bend but its stretch band is 1.4×
stronger relative to bend and libration than the TNEP's and 1.6–1.8× stronger than both experiments — the earlier
peak-height reading (GAP "closer to experiment") was wrong: the experimental libration peaks are low because the
band is broad; on integrals, experiment and TNEP agree (∫stretch/∫80–2500 = 15.6–16.7) and the GAP is the outlier
(24.3). Both simulations put ~1.6–2× too much integrated libration relative to the bend (classical nuclei). The
response probe does not see this (both models reproduce the DFT collective-stretch slope), so the excess is in what
the bond-polarisability features miss (R²(β) 0.43), e.g. the intermolecularly coupled symmetric-stretch part of the
anisotropic band (Brooker: ~50 % of it). A reference α(t) is still needed to say which model is right vs DFT.
Isotropic: all curves agree on the stretch band; experiment is broader at 1650.

**Quantum-correction bookkeeping (2026-10-27, after the user's question).** The exact prefactors act on the QUANTUM
line shape L_q: IR α(ω)n(ω) ∝ ω(1−e^(−x))·L_q (Xu Eq. 8); the experimental reduced Raman spectrum R(ω) =
I·(ω₀−ω)⁻⁴·ω(1−e^(−x)) (Brooker 1988/89) equals ω·L_q because I ∝ (ω₀−ω)⁴(n+1)L_q. Classical MD gives L_cl, and
the harmonic quantum correction L_q = x/(1−e^(−x))·L_cl (Ramírez et al. 2004) — the standard choice for water — then
yields IR ∝ ω²·L_cl and Raman R ∝ ω²/(1−e^(−x))·L_cl. So Xu's "ω²" IS the harmonic-corrected result (exact for
harmonic modes), not a low-frequency approximation, and the weighting this code used to call "harmonic",
ω(1−e^(−x))·L_cl, is the exact prefactor applied to the UNCORRECTED classical ACF (no QCF at all) — it underweights
the stretch ~14× relative to the libration, which is why it "did not perform as well". No label correction is
involved anywhere. Renamed accordingly in `_quantum_prefactor`: "harmonic" = ω² (IR) / ω²/(1−e^(−x)) (Raman, the
reduced-spectrum form; equal to ω² within 10 % above 500 cm⁻¹, ×1.6 at 200 cm⁻¹), "classical" = ω² (Xu Eq. 9 /
Fig. 4 footing), "bose" = the old ω(1−e^(−x)), "linear", "none". Defaults: IR "harmonic" (now ω²; MasterTNEP's
process_trajectory therefore changes), Raman "harmonic". Confirmed against Xu's own script found locally
(`~/Work/GPUMD Testing/Water Test (Xu)/calculate_IR.py`: mean-free ACF, first 10 % of lags, Hann, Kronecker, cosine
transform, ×ω², box-10 smoothing — identical to compute_ir_spectrum "classical"), and that folder's run.in is the
trajectory's: NPT/NVT equilibration then NVE at 298.15 K, dt 0.5 fs, dump every 1 fs. GAP anisotropic integrals vs the
bend under the Raman "harmonic" form: H-bond 0.64, libration 1.96, stretch 109 (ω²: 0.37 / 1.85 / 109).

**Implementation check and comparison plots (2026-10-27).** Exact port of Xu's `calculate_IR.py` reproduces his
`ir_smooth.txt` to L1 = 0.000; our `compute_ir_spectrum("harmonic")` (ω²) on the same dipoles differs from it by L1
0.058, which drops to 0.004 after shifting his grid by 4.5 bins — his `np.convolve(mode='valid')` smoothing is not
re-centred (3.6 cm⁻¹ shift); unsmoothed the two agree to 0.006 (biased vs unbiased ACF normalisation). Prefactor forms
pinned by `test_quantum_prefactor_forms`. Plots (unit integral 80–2500, Fig. 4 presentation):
`plots/ir_water_vs_experiment.png` — Max & Chapados optical constants (α·n and α), Downing & Williams and Xu's TNEP
curve digitised from Fig. 4(a), Xu's dipole model run in GPUMD on our 200 ps trajectory, and this code's TNEP dipoles;
`plots/raman_water_vs_experiment_{aniso,iso}.png` — Brooker, Morawietz, Xu's TNEP (digitised), Xu's χ model on our
20 ps, and the GAP (reduced form, harmonic QCF). IR band integrals vs the bend (H-bond / libration / stretch):
Max α·n 0.57 / 3.70 / 11.2, Downing 0.53 / 3.90 / 12.0, Xu TNEP published 0.55 / 4.50 / 14.3, Xu model on our
trajectory 0.54 / 4.05 / 13.1, this code's TNEP 0.56 / 4.21 / 13.6 — IR agrees with experiment to ~10–20 % on every
band, which is the known classical-nuclei residual; the Raman stretch excess of the GAP (1.4× vs TNEP) has no IR analogue.

### GAP trajectory prediction performance (2026-10-27): 99 min → 24 min for 100 000 frames of 648 atoms

Baseline (quippy SOAP, 8 forked CPU workers): 59 ms/frame wall, 99 min for the 100 000-frame stride-2 trajectory; the
TNEP ANN path does the full 200 000 frames in 15 min (4.5 ms/frame). Single-process profile of `gap_dipole.predict`
before: 399 ms/frame = quippy SOAP 297 (serial Fortran, no OpenMP) + `_phi_frame` einsum 82 + bond-basis readout 29.

Changes (all equivalence-checked: predictions agree with the quippy path to max |Δα| 3e-6 Å³ on 32 trajectory frames;
120 GAP tests pass; the 4 neighbour-readout smoothness/guard tests that fail also fail on HEAD):
1. SOAP on the GPU (`descriptor_mode=1`, `basis="poly3"` — the GPU builder lacks poly3gauss; poly3 gives the same test
   RMSE, 0.2624 vs 0.2629). `_descriptors` now calls `build_descriptors_flat(use_tf_nl=True)`: the bucketised
   `build_descriptors` spent 25 ms/frame packing per-atom gradient lists we never read, and its default NumPy
   neighbour list is a brute-force N² scan (262 ms per 648-atom frame); the TF/MIC list is ~1 ms. 270 → 17 ms/frame.
2. `_phi_frame`: the `einsum("nm,nxk->xmk")` contraction ran as loops (14 ms per species block); one GEMM
   `(G^ξ)ᵀ @ v.reshape(n, D·K)` is < 1 ms. 82 → 1 ms/frame.
3. `data._neighbor_pairs`: periodic `cKDTree(boxsize)` for cells with > 128 atoms instead of the N² MIC scan (12 → 1.6 ms);
   `data._neighbor_dipole_V`: the `np.add.at` scatter (12 ms) is a sparse CSR matmul (0.4 ms).
4. `scripts/raman_gap_gpumd_traj.py`: spawned (not forked — CUDA) workers sharing the GPU, each with its own model and
   a hard VRAM cap (`RAMAN_VRAM_MB`, default 2500); with memory growth 4 processes filled the card and thrashed
   (160 ms/frame). OMP threads partitioned 16 / workers. quippy's per-frame leak is gone with quippy.

Single process now: 27 ms/frame (GPU descriptors 17, CPU readout + kernels 10). 5-minute samples on the trajectory
(rate incl. start-up, read from the script's ETA line, runs killed afterwards):

| workers (OMP threads each) | ms/frame wall | GPU util / VRAM | 100 000 frames (stride 2) | 200 000 frames |
|---|---|---|---|---|
| quippy, 8 forked CPU workers (before) | 59 | — | 99 min (measured) | ~200 min |
| GPU, 1 (16) | 32.2 | 22 % / 4.2 GB | 54 min | 107 min |
| GPU, 2 (8) | 17.1 | 35 % / 6.7 GB | 29 min | 57 min |
| **GPU, 3 (5)** | **14.2** | 66 % / 9.7 GB | **24 min** | **47 min** |
| GPU, 4 (4) | no frames in 5 min | 100 % / 11.7 GB | saturated card | — |
| TNEP ANN path (user's run) | ~4.5 | — | ~7.5 min | 15 min |

Default is now 3 workers (`RAMAN_WORKERS`). The 4-worker run never produced a progress line: four 2.5 GB caps plus
contexts fill the 12 GB card.

**Why it is still slower than the ANN, and what would close the gap.** The TNEP inference path is one XLA-compiled TF
graph per batch of frames — neighbour list, SOAP, ANN readout all on the GPU, ~4.4 ms/frame. The GAP forward is the
same descriptor call plus a Python/NumPy per-frame readout: bond basis V [N, 6, K] (ASE-contract pair search, radial
basis, scatter), cross group, per-species kernel Gram D·Dₘᵀ and its power, then the GEMM — about 10 ms of CPU per
frame that cannot overlap the GPU inside one process, plus TF's per-call overhead and host↔device copies on the
descriptor side (the SOAP kernels themselves are a few ms; `build_descriptors_flat` costs 15 ms/frame at batch 4–16).
The kernel evaluation itself is cheap (648×250 Gram per species). Fixes, in order of payoff: (i) port the bond-basis
readout and the kernel contraction to TF and batch frames, so one GPU graph does descriptors + V + Gram + GEMM —
this is what the ANN path already does and should bring the GAP to the same ~5 ms/frame; (ii) reuse the descriptor
builder's neighbour list for V instead of a second pair search; (iii) `tf.function`/XLA the descriptor forward with a
fixed pair-tile shape to cut per-call overhead; (iv) a poly3gauss radial basis in the GPU builder if that basis is
ever needed (poly3 is equivalent here).

**GPU readout port (2026-10-27, `gap_gpu.py`).** The soap-only polarisability forward — all-pairs MIC neighbour list,
bond basis, cross group, per-species kernel Gram and readout — is one `tf.function` per batch of frames;
`gap_dipole.predict` dispatches to it for that model on orthorhombic MIC-safe cells and keeps the NumPy
`_predict_numpy` (the fit's contraction) for everything else. Risk assessment before building it: batch-boundary
artefacts (a periodic per-chunk error becomes a line at 1/(8 × 2 fs) = 2083 cm⁻¹), float32 cancellation across
hundreds of atoms, neighbour-list edge differences, fit/predict drift. Findings: TF32 — TensorFlow's default for
float32 matmul on this card — gave 300 Å³ errors on a 290 Å³ cell, and true float32 still left rms 0.03 Å³ with a
visible batch-8 component in the error's FFT (×2.5 the median); float32 SOAP descriptors alone cost 1 Å³. The forward
therefore runs in float64 on the GPU (a few hundred MFLOP per batch, <1 ms): max |Δα| 3e-6 Å³ against the NumPy path
on 2000 contiguous frames, identical band ratios, no batch-period signature (0.35). `tests/test_gap_gpu.py` pins
GPU == NumPy at 1e-4 Å³; all 67 GAP tests pass. XLA (`jit_compile=True`) on the descriptor builder is 20–40× slower
(retraces per pair count) and batch 32 OOMs — batch 8 with `descriptor_batch_frames=4` inside.

| workers (OMP each) | ms/frame wall | GPU util / VRAM | 100 000 frames | 200 000 frames |
|---|---|---|---|---|
| quippy, 8 CPU workers (original) | 59 | — | 99 min | ~200 min |
| GPU descriptors + NumPy readout, 3 | 14.2 | 66 % / 9.7 GB | 24 min | 47 min |
| **full GPU forward, 1 (16)** | 18.5 | 40 % / 4.9 GB | 31 min | 62 min |
| **full GPU forward, 2 (8)** | 13.8 | 68 % / 7.5 GB | 23 min | 46 min |
| **full GPU forward, 3 (5)** | **13.2** | 85 % / 10.2 GB | **22 min** | **44 min** |
| TNEP ANN path | ~4.5 | — | ~7.5 min | 15 min |

Single-process the forward went 27 → 16 ms/frame (readout ~1 ms), so one worker now does what three did before; but with
2–3 workers the card is 70–85 % busy and the wall time barely moves: the limit is now the SOAP builder itself — complex128
expansion-coefficient tensors of [pairs, 36, 14] for alpha 7 / l 7, ~15 ms per 648-atom frame, and it must stay float64
(float32 descriptors feed the large sparse-GP coefficients and cancel badly). Closing the remaining 3× to the ANN would
need a cheaper descriptor (lower alpha/l or a radial-basis cut, i.e. a model change), a float32-safe reparameterisation of
the kernel coefficients, or the builder's pair tiling and complex casts reworked for the forward-only path.

**Why the ANN path is 3× faster with the same SOAP builder (2026-10-27).** Default workers reverted to 2 (3 adds
<1 ms/frame). Measured breakdown of the builder's forward-only call at the GAP settings (batch 4, float64):
TF neighbour list incl. host copy 7.7 ms/frame + SOAP compute 6.0 + host bookkeeping/upload 2.5 = 16 ms; batch 8 is
19 ms (compute 9.3), larger batches OOM because the forward-only path has no pair tiling. SOAP size is not the cost:
dim_q 75 (alpha 4/l 4, the TNEP's) runs 12 ms/frame, dim_q 216 (alpha 7/l 7) 15 ms — overhead-dominated.
The ANN trajectory path (`spectroscopy.predict_trajectory_batch`) calls the same builder differently: float32,
`descriptor_pair_tile_size` (pairs padded to a multiple of the tile → fixed shapes) and `jit_compile=True` (XLA, one
compile per shape), 40 frames per call, then a fused ANN graph — the whole frame is a few compiled kernels, ~4.5 ms.
None of that transfers to the GAP as it stands: the sparse-GP coefficients are large and cancel across the 648 atoms,
so float32 descriptors (1e-7 relative noise) already give 1 Å³ errors on a 290 Å³ cell — frame-to-frame white noise
above the 0.86 Å³ fluctuation we are resolving — and float64 forbids the float32-only XLA/tiled path. The per-frame
NL+bookkeeping overhead (10 of the 16 ms) is the builder's API, not SOAP; the user has a fix for the builder side,
so it is not pursued here. A GAP-side route would be a better-conditioned fit (larger σ / fewer sparse points / jitter)
that tolerates float32 — a model change.

### Mean prior E_0 (2026-10-27, `gap_mean_prior`)

One constant target vector over the dataset, added to every structure's prediction and fitted JOINTLY with the
kernels as an unpenalised intercept of the ridge: D extra columns (row (f, x) carries 1 in column x) with no prior
penalty. Frisch–Waugh–Lovell makes that exact without touching the dual solve: residualise the weighted rows of Φ and
μ against the (weighted) intercept column, solve the ridge as before, then E_0 = weighted LS of the leftover
residual — the dataset mean of the residual (exactly so when every frame has the same N, as here). Fitted residuals
therefore sum to zero over frames, which the plain ridge does not give (`test_mean_prior_fits_residual_and_adds_n_e0_back`).
The kernels keep part of the mean (E_0 ≈ 0.67 × the raw mean on a 6-frame fit), because the isotropic bond-basis
columns carry a constant-like part the ridge is happy to use. E_0 is intensive by construction (a 648-atom cell gets
the 96-atom E_0) — harmless for spectra, which remove the mean; wrong for absolute values of extensive targets; a per-atom
variant (column N_f, predict + N·E_0) is the gap_fit-style alternative if that is ever needed. GPU path adds E_0 after
the forward (`predict`), record field `mean_prior_e0`.

| model (fit 500 → test 500) | diag/off RRMSE | with E_0 |
|---|---|---|
| base (SOAP, cross 4, M 250) | 14.9 / 16.0 | 14.7 / 16.0 |
| best3b (SOAP + 3b M40) | 14.4 / 14.7 | 14.1 / 14.7 |

Response slopes (stretch/rigid/bend) unchanged to 0.5 %. The diagonal gains 0.2–0.3 points because the fit no longer
spends penalised coefficients on the 45 Å³ isotropic mean; the off-diagonal (mean ≈ 0) is untouched.

**Dipole.** Fit 500 → test, σ 0.003: SOAP alone 1.59 % → 1.60 % with E_0; soap+2b(M40) δ0.1 + cross 2 (the best μ model)
1.45 % → 1.45 %. Null, as expected: the training-set dipole mean is within one standard error of zero
(|mean|/std ≤ 0.06, SE 0.29 e·bohr on a 6.4 e·bohr spread) and the test set's mean differs from it by about as much with
the opposite sign, so E_0 is sampling noise at the 0.3 e·bohr level — 5 % of the residual RMSE, invisible in the RRMSE.
Use the prior for the polarisability, not the dipole.

### Best polarisability models on the χ labels (2026-10-27; rows "chi …", σ ∈ {0.1, 0.3} on fit 400 → val/test, refit 500 → test)

| model (χ labels) | σ | test (500) diag/off RRMSE % |
|---|---|---|
| base (SOAP, cross 4, M 250) | 0.1 | 17.0 / 15.9 |
| base + E_0 | 0.1 | 17.0 / 15.8 |
| base cross 6 + E_0 | 0.1 | 17.0 / 15.9 |
| base poly3 (GPU basis) + E_0 | 0.1 | 17.0 / 15.8 |
| SOAP + 3b (M40, δ 0.1, l_x 2, C 4) [± E_0] | 0.1 | 16.6 / 14.7 |
| SOAP + 3b (M40, δ 0.1, l_x 2, C 6) + E_0 | 0.1 | 16.7 / 14.8 |
| **SOAP + 3b (M40, δ 0.15, l_x 3, C 6) + E_0** | 0.1 | **16.6 / 14.6** |
| SOAP + 2b + 3b (M40, δ 0.03 / 0.1, l_x 2) + E_0 | 0.1 | 16.7 / 14.8 |
| TNEP χ model (Xu et al.) on the same test set | — | 16.3 / 20.4 |

σ 0.3 always lost on the off-diagonal (+1.3) for ≤ 0.4 on the diagonal. E_0 is null on χ (the SOAP+bond columns carry the
mean comfortably once the 3b term is present); poly3 ≡ poly3gauss; cross 6 ≡ cross 4. The 3b site model is again the one
real gain (off-diagonal 15.9 → 14.6).

**A better χ predictor is the CM-trained model mapped through the exact inverse Clausius–Mossotti relation** (cross-scoring,
same frames, `scratchpad chi_vs_cm.py`): base CM-trained → χ 15.4 / 15.9 (vs 17.0 / 15.8 χ-trained); best3b CM-trained → χ
14.9 / 14.7; best3b + E_0 CM-trained → χ **14.7 / 14.7** — the best χ numbers of all, 1.9 points better on the diagonal than
any χ-trained fit, off-diagonal identical. The CM relation Vχ = Nα/(1 − 4πρα/3) divides out the collective Lorentz local-field
enhancement analytically before the local site-sum kernel sees the target and restores it after; the local model cannot
learn that non-local isotropic part itself (the same physics the Ewald/induced columns address). Off-diagonal has no such
enhancement at this order, hence unchanged. The inverse map is ε − 1 = 3A(I − A)⁻¹ with A = 4π·1.13205·α_CM/(3V), Vχ = V(ε−1)/4π,
so prediction needs the cell volume; for spectra at 648 atoms the map is linear and changes nothing.

**CM clean-up (2026-10-27).** All polarisability run records fitted on the CM-form `pol` column (406 JSON files under
GAP/ and the scan's record dir), the 200 ps CM-based trajectory prediction and its plots were deleted. The datasets stay
separate: `datasets/{train,test}_waterbulk.xyz` are the untouched Zenodo dipole-archive files (md5 9abb291e / ece965e1;
their `pol` column is the CM form and is NOT to be used for polarisability fits), `datasets/{train,test}_waterbulk_chi.xyz`
the same frames with `pol` = Vχ. Polarisability work from here on uses the χ files (probe variants "chi …", scan rows
"chi …", `RAMAN_DATA/RAMAN_TEST` on the trajectory script). Every α number in this README above the "Best polarisability
models on the χ labels" section was measured on the CM column. The 200 ps trajectory is being re-predicted with the χ base
model (`plots/water_bulk_GPUMD_traj_polarizabilities_chi.npy`). `tests/test_two_atom_dipole.py`: an O–H pair with a fixed
partial charge, μ = q·d — the fitted dipole GAP reproduces q·d at d and 2d within 5 % and the 2× ratio within 10 %.

**Two-atom O–H check on trained models (2026-10-27, `tests/test_two_atom_dipole.py`, strict xfail).** Isolated O–H pair in a
20 Å box, μ_x along the bond (e·bohr): gradient-readout TNEP n30_q216 (R² 0.9998) +0.52 at 0.97 Å, +1.37 at 1.5 Å, −0.47 at 1.94 Å,
→ 0 beyond 3 Å; gradient-readout TNEP n30_q75 (Sep) +0.19 → +0.02 (monotonic decay to the cutoff, no flip); bulk-fitted dipole GAP (soap+2b+cross) +0.82 at 0.97 Å, +0.33 at 1.94 Å, 0 at the 3 Å cutoff. A fixed partial
charge would give q·d, i.e. +0.73 → +1.44. Both bulk-trained models grow with d only up to ~1.5 Å: the training set contains
O–H distances only as covalent bonds (0.9–1.05 Å) and hydrogen bonds (1.6–2.2 Å), so a stretched pair is read as an O···H
hydrogen bond, and a local model carries no charge to make μ keep growing — the gradient readout's radial derivatives vanish at
the cutoff by construction, so μ → 0 with separation whatever the training; only a charge-like term q_i·r_i (the charge head)
could pass. The test asserts the physical expectation and is marked strict xfail with this reason.
Single-channel follow-up (user's request): SOAP-only dipole GAP with ONE Chebyshev channel and no cross term, fitted on bulk
water. cosine² cutoff (rc 3): +0.87 at d, +0.31 at 2d — fails; cosine cutoff: +2.10 → +2.79 — passes, but with one channel
the pair dipole is μ = c(env)·f_c(d)·d and d·f_c(d) for the cosine cutoff happens to peak near 1.9 Å (cosine² peaks near
1.4 Å), so the pass is the cutoff shape, not a learned charge; bulk test R² drops to 0.878 (0.986 cosine², 0.9998 full model).
rc 4 Å single channel: sign change off the training manifold. Conclusion unchanged: μ ∝ d needs a charge-like term.

**χ-label 200 ps trajectory (2026-10-27).** `plots/water_bulk_GPUMD_traj_polarizabilities_chi.npy` (base χ model: SOAP, cross 4,
M 250, σ 0.1; 100 000 frames at 2 fs, 23 min with 2 GPU workers). Mean 431 Å³, std 2.1 / 1.5 Å³ (diag / off) — the TNEP's
scale, as it should be now that the labels agree. Anisotropic band integrals vs the bend, ω² footing: H-bond 0.37, libration
1.82, stretch 107, ∫stretch/∫80–2500 = 24.0 — within 2 % of the CM-era numbers (0.37 / 1.85 / 109 / 24.3), confirming that the
label convention never touched the spectra. Plots regenerated: `..._chi_gap_local_classical_raman_{iso,aniso}[_split2500]_*.png`
and the overlays `plots/raman_water_vs_experiment_{aniso,iso}.png` now show this χ-trained model.
Plots `plots/oh_pair_{1,10}channel_dipole.png` (+ .csv; `scripts/plot_oh_pair_dipole.py`, CHANNELS=1|10): μ_x(d) for the three
cutoff variants against q·d. Ten channels (bulk test R² 0.9997–0.9998, i.e. the real readout): all three variants give ~0.4 e·bohr
at the bond, a hump to 1.2–1.4 near 1.4 Å, a sign change at 1.8–2.0 Å (the hydrogen-bond O···H distance, learned with the
opposite sign) and zero by 2.5–3 Å — the cutoff choice no longer matters once the radial basis can resolve the two distance
populations of the training set. The single-channel "pass" with the cosine cutoff was the one-function case where the shape is
fixed by f_c(d)·d alone.
Hard cutoff (user's request; `dipole_cutoff="none"`, f_c ≡ 1, the neighbour list alone selects pairs; GAP readout only):
`plots/oh_pair_nocutoff_dipole.png` / .csv. One channel: the readout becomes μ = c(env)·d exactly, and it IS linear in d —
but with c < 0 (−1.7 at 1 Å, −5.1 at 2 Å) and a 55 % bulk RRMSE: one global coefficient cannot serve bonds and hydrogen
bonds at once, so it fits neither. Ten channels (bulk RRMSE 2.2–2.6 %, versus 1.5–1.6 % with cosine²): the same
two-population curve as before — positive at the bond, hump at 1.4 Å, sign change at 1.8 Å — now ending in a step to zero
at 3 Å instead of a smooth decay. Removing the cutoff function changes the edge behaviour and costs bulk accuracy; it does
not change what the model learned, because the sign change comes from the data (hydrogen-bond pairs), not from f_c.
SOAP cutoff vs readout cutoff (hard readout, 10 ch): readout 3.0/SOAP 5.0 keeps the sign change at 1.8 Å; readout 4.5/SOAP 5.0
decays to ~0 from 2.4 Å although both radii are far away (+0.40 at 1 Å, −0.21 at 2 Å, −0.05 at 2.4, 0.00 at 4 Å). The decay is
the learned pair weight at hydrogen-bond and second-shell distances (no net transfer in water), not either cutoff; cutoffs
only enforce the exact zero beyond rc.

### χ fitting suite (2026-10-28; rows "chi S1…S5", `gap_progressive_scan.py` POL_R6/R7, driver `gap_scan_parallel.sh`; σ ∈ {0.03, 0.1, 0.3} on fit 400 → val/test, best-val σ refit on 500 → test; diag/off RRMSE %)

Isolated search on `datasets/{train,test}_waterbulk_chi.xyz` (no comparison with the retired CM models). Stage 1 = base
iso/aniso readout without the cross term, one option at a time; stage 2 = cross term; stage 3 = other options on base + cross 4;
stage 4 = multi-scale SOAP (`gap_extra_rcut`); stage 5 = combinations of the stage winners. 67 + 15 rows, 300 fits, ~5 h with two
workers (rows above 7 GB RSS — cross 8, angular cross, triplet tensor, every extra_rcut 6 + 3b row — one at a time under a 12.5 GB
watchdog; `MAXJ=1 WD=12500000`). Angular cross l_x 2 and the extra_rcut 6 + 3b + angular-cross row exceed 13 GB alone and were dropped.

**Stage 1 (no cross; base 18.7/19.8 at fit 500).** Only cutoffs matter: readout r_c 3.0 → 21.8/24.5, 3.5 → 21.0/21.2, 4.0 (base)
18.7/19.8, 4.5 with SOAP 4.5/4.0 → 17.4/18.4, 5.0 with SOAP 5.0/4.5 → **16.8/17.7**; SOAP 5.0/4.5 alone 16.9/19.4 (the diagonal
is the SOAP radius, the off-diagonal the readout radius). Everything else is within ±0.5 of base: channels 6/14/20 (18.2/20.4,
19.9/19.8, 20.1/21.1), radial chebyshev 18.8/19.6, gaussian 19.3/19.6, bessel 20.7/19.7, cutoffs cosine2/cubic/tanh3 18.7–19.1/19.6–19.9,
group deltas 18.6–20.6/19.5–19.8, α/l 5/5 7/4 4/7 9/7 all 18.6–19.0/19.8–20.2, compress none 18.3/20.2, ξ 1 / 4 / 1+2+4 19.8/19.4,
18.9/19.6, 19.2/19.4, M 500 = M 250 (18.7/19.8), pairs off 19.7/22.5.

**Stage 2 (cross term).** cross 1 → 17.5/17.4, 2 → 17.3/16.7, **4 → 17.0/15.9**, 6 → 17.0/16.0, 8 → 17.0/16.4; δ_x 0.3 or 3 on
cross 4/6 within 0.2. The cross term is the single biggest off-diagonal gain (−3.9).

**Stage 3 (on base + cross 4 = 17.0/15.9).** 3b(M40) site model: δ 0.1 l_x 3 C 6 → **16.6/14.6** (best 3b; δ 0.05–0.3, l_x 1–2, C 4–10,
M 80, L 2 all 16.6–16.8/14.7–15.2); 2b(M40) δ 0.03 → 17.1/15.8 (2b alone adds nothing the cross term lacks); 2b + 3b = 3b alone
(16.6/14.8). Angular cross l_x 1 16.5/16.0 (102 k columns, 9 GB); triplet tensor 18.3/16.5 (worse); mean prior E_0 17.0/15.8
(null); λ_shear 0.5 → 17.4/16.8, **2 → 16.9/15.5** (down-weighting the diagonal rows helps the off-diagonal at no diagonal cost).

**Stage 4 (multi-scale SOAP, second SOAP at `gap_extra_rcut`).** 2.5 Å → 17.1/16.9 (worse), 5 → 15.7/16.1, 5.5 → 15.2/16.1,
**6 → 14.9/16.2** (the long-range SOAP is the only diagonal gain: −2.1 at no column cost, 20 k); 5.5 at M 150 15.3/16.1;
5.5 + 3b(M40, l_x 2) 14.8/15.2.

**Stage 5 (combinations).** r_c 5 + SOAP 5 + cross 4 15.9/16.0; r_c 5 + SOAP 5 + 3b l_x 3 14.6/14.9; extra_rcut 5.5 + 3b l_x 3
14.6/14.9; extra_rcut 6 + 3b l_x 3 14.3/14.9; + r_c 5 15.5/14.9 (the long readout hurts once the long SOAP is present);
+ mean prior 14.2/14.9; + cross 6 δ_x 0.3 14.2/14.6 (143 k columns, 10 GB); **+ λ_shear 2 → 14.1/14.3** — the suite's best:
SOAP 4.0/3.5 + second SOAP 6.0, readout r_c 4, 10 warped-Chebyshev channels, cross 4, 3b(M40, δ 0.1, l_x 3, C 6), λ_shear 2,
σ 0.3, 87 520 columns, 8.4 GB. Base → best: 18.7/19.8 → 14.1/14.3 (−4.6 / −5.5); the TNEP χ model scores 16.3/20.4 on the same
test frames and convention (see the RMSE note below). σ 0.3 won every stage-4/5 row (the long SOAP columns want more ridge), 0.1
everything else.

**RMSE conventions.** All water frames have 96 atoms / 32 molecules, so per-atom, per-molecule and per-cell RMSEs differ by the
constants 96 and 32 applied to prediction and label alike; the RRMSEs here (RMSE / label spread) are invariant to that. GPUMD prints
one per-atom RMSE pooled over the 6 components; our "total" per-cell RMSE is √D times the per-component one — the two absolute
numbers need both conversions (÷N and ÷√D) before comparison, the relative ones none. The 16.3/20.4 TNEP cross-score was computed
per cell from GPUMD's per-atom output on the same frames and is like-for-like.

**Unit-vector readout (`dipole_unit_vector=True`, rows "chi U …"; bond tensors from û = d/|d|: rank 1 û, rank 2 I and ûûᵀ − I/3, so
R_n(r)·f_c(r) carries all of the r-dependence instead of r·R_n and r²·R_n).** Threaded through the four geometry sites
(`data.assemble_data_dict` V_atom, `data.bond_geometry`, `gap_dipole._v_atoms/_cross_vectors`, `gap_gpu._bond_basis`); the first
run missed the fit's V_atom site and scored 900 % on the diagonal (fit on d, predict on û) — now guarded by
`test_unit_vector_geometry_used_by_fit_and_predict`. Test (500), unit vs d:

| model | d (standard) | û (unit vector) |
|---|---|---|
| base, no cross | 18.7/19.8 | **17.7/19.5** |
| base, 20 channels | 20.1/21.1 | 20.0/19.4 |
| base, plain Chebyshev | 18.8/19.6 | **17.7/19.2** |
| base + cross 4 | 17.0/15.9 | **15.8/15.6** |
| + 3b(M40) l_x 3 | 16.6/14.6 | **15.4/14.8** |
| extra_rcut 6 + 3b l_x 3 | 14.3/14.9 | **13.2/14.3** |

The unit vector wins the diagonal by 1.0–1.2 at every stage and ties or wins the off-diagonal; it prefers less ridge (σ 0.03,
0.1 with the long SOAP) because its columns are r² smaller. A plausible reason: with d the iso group's column is Σ_j r²f_c(r)T_n(r),
which weights the second shell ∝ r² before the learned radial function sees it, whereas with û the learned R_n(r) alone decides the
shell weighting. Follow-up rows "chi U best" (winner + unit) and "chi UE best + all" (+ Ewald) below.

**Last combinations and the Ewald interest test (rows "chi S6 / chi E", σ 0.3 only, best = extra_rcut 6 + 3b l_x 3 + λ_shear 2 = 14.1/14.3).**
best + cross 6 δ_x 0.3 → 13.9/14.1 (143 k columns, 10 GB; −0.2 for 1.6× the columns); best + E_0 → 14.0/14.3 (null).
Ewald field groups of SPC/E reference charges (`pol_field_groups`, `dipole_field_charges={8: −0.8476, 1: 0.4238}`,
`pol_induced_alpha` {O 1.45, H 0}; scales pinned on the fit frames): + induced **13.4/13.6**, + grad 14.1/14.4, + cross 14.1/14.3,
+ quad 13.8/14.2, + qeq 13.9/14.4, + all five **13.1/13.4** (102 k columns). As in the CM-era groups scan, the traceless Applequist
induced-coupling tensor is the one long-range column that carries information (−0.7/−0.7); the field gradient, field-cross and
QEq columns are null on their own, and the full set is worth −1.0/−0.9 over the local winner. Kept for interest only (Ewald
models remain out of scope for the trajectory work).

**Final (rows "chi U best", "chi UE best + all"; σ ∈ {0.1, 0.3}).** The winner with the unit-vector readout: **13.0/13.9** (σ 0.1;
σ 0.3 14.1/15.4 — û wants less ridge), i.e. base 18.7/19.8 → 13.0/13.9 over the whole suite, and 3.3/6.5 points ahead of the TNEP χ
model on the same test frames (16.3/20.4). With all five Ewald groups on top (interest only): **12.0/13.0**. Cost of the winner:
87 520 columns, 7.0 GB fit RSS, 2.2 ms/atom numpy prediction (the GPU path needs `basis="poly3"` and does not yet carry the
extra_rcut second SOAP or the 3b site model, so trajectory prediction with this model is a quippy/numpy job — see "GAP trajectory
prediction performance").

**`dipole_unit_vector=True` is the GAPconfig default from 2026-10-28** (user's call after the suite). `gap_progressive_scan.py` pins
`dipole_unit_vector=False` in its MU/POL reference dicts so every non-"U" row above stays reproducible; new GAP fits elsewhere
(probe, trajectory script, tests via `_cfg`) now use û unless they ask for d. The GPU forward honours the flag from the model's cfg.

**Stage 2 with the unit-vector readout (rows "chi U cross …", σ grid {0.03, 0.1, 0.3}; test 500, û vs d).** cross 1 16.4/16.6 vs
17.5/17.4; 2 16.1/16.2 vs 17.3/16.7; 4 **15.8/15.6** vs 17.0/15.9; 6 15.8/15.8 vs 17.0/16.0; 8 15.8/16.2 vs 17.0/16.4; cross 4 δ_x 0.3
16.0/15.8, δ_x 3 15.8/15.8; cross 6 δ_x 0.3 15.8/15.6; cross 2 δ_x 3 16.0/16.1. Same shape as with d (knee at 4 channels, δ_x
irrelevant, 8 channels slightly worse off-diagonal), shifted by −1.1 to −1.3 on the diagonal and −0.2 to −0.8 off; σ 0.03 won every
row (σ 0.3 is 2–3 points worse with û).

**Column counts (design width n = Σ_blocks M_block·K_block; water T′ = 2 species columns, M = 250/species for SOAP blocks).**
Measured by refitting each row on 12 frames (`scratchpad ncols.py`; the `ncols=` in the scan log before 2026-10-28 was read from
the newest GAP/ record and is wrong whenever two workers ran — the scan now counts the fitted model's own coefficients).

| block | K per sparse point | scaling | water example |
|---|---|---|---|
| bond basis (SOAP block) | 2·C·T′ | linear in channels C | C 10 → 40; C 6 → 24; C 20 → 80; pairs off (T′ 1) → 20 |
| + cross group | K₁(K₁+1)/2, K₁ = C_x·T′ | quadratic in C_x | C_x 1/2/4/6/8 → 3/10/36/78/136 |
| ξ list | × number of ξ | linear | ξ 1+2+4 → ×3 |
| second SOAP (`gap_extra_rcut`) | same K as the first block, own M | doubles the SOAP block | base+cross 4: 38 000 → 76 000 |
| 2b site model (M 40) | same readout K as SOAP | + 80·K | + 6 080 on base+cross 4 |
| 3b site model (M 40) | K₁(K₁+1)/2·(L_x+1) | linear in L_x, quadratic in C_x; descriptor C/L free | l_x 1/2/3 → 72/108/144 → + 5 760 / 8 640 / 11 520; M 80 doubles |
| angular cross | + 2·K₁²·L_x on every SOAP block | quadratic in C_x, linear in L_x | l_x 1 → +64 000; l_x 2 → +128 000 (13 GB) |
| triplet tensor | + 100 on every SOAP block | — | + 50 000 |
| Ewald groups | induced 1, grad 1, quad 2, qeq 2, cross 2·C_x (per SOAP block) | — | all five on the winner → + 14 000 |

Rows: base 20 000; cross 1/2/4/6/8 → 21 500 / 25 000 / 38 000 / 59 000 / 88 000; 3b(M40) l_x 1/2/3 on base+cross 4 → 43 760 /
46 640 / 49 520 (C 10 or L 2 descriptors: unchanged; M 80 → 55 280); 2b(M40) 44 080; angular cross l_x 1 / 2 → 102 000 / 166 000;
triplet 88 000; extra_rcut 6 → 76 000; winner (extra_rcut 6 + 3b l_x 3) 87 520; winner + cross 6 → 142 960; winner + Ewald
induced/grad 88 520, quad/qeq 89 520, cross 95 520, all 101 520. M 500 → 40 000 (the 12-frame refit caps at 35 360 for lack of
oxygen atoms). Fit cost: the design is (frames·6) × n and the solve is O(n²) memory — 87 520 columns ≈ 7–8 GB, 143 000 ≈ 10 GB,
166 000 > 13 GB.

### Architecture-review follow-ups (2026-10-28; rows "U soap+2b(M40) d0.1 +cross2", "chi CM U cross 4", "chi CM U best"; σ selected on val 400/100, refit 500 → test)

Two of the three one-fit checks suggested in `docs/gap_architecture_review.md` (the third, cross 2 + field column, was skipped as a field term).

**Unit-vector readout for the dipole** (`dipole_unit_vector=True` on the best local μ model soap+2b(M40) δ0.1 + cross 2; σ grid 0.0003 … 0.03):
σ 0.0003 / 0.001 / 0.003 / 0.01 / 0.03 → test(400) 1.92 / 1.62 / 1.53 / 1.63 / 1.88 %; best σ 0.003 refit on 500 → **1.47 %** vs **1.45 %** with d.
Null (within the 0.03 σ-to-σ noise, slightly worse). The dipole's radial weighting r·R_n(r) is right as it is; û stays a polarisability-only gain.
The GAPconfig default remains True; dipole fits should set `dipole_unit_vector=False` (the scan's MU dict does).

**Clausius–Mossotti label map** (fit the kernels to α_CM = 3V·A/4π, A = B(3I+B)⁻¹, B = 4π·Vχ/V per frame; predict back through B = 3A(I−A)⁻¹;
implemented as a temporary `gap_cm_map` flag, since REMOVED):

| model (χ labels, test 500 diag/off) | plain | CM map | σ (map) |
|---|---|---|---|
| û base + cross 4 | 15.8 / 15.6 | 14.2 / 15.6 | 0.03 |
| û winner (extra_rcut 6 + 3b l_x3 + λ_shear 2) | 13.0 / 13.9 | 12.6 / 13.8 | 0.1 |

The map does what the review predicted (an exact non-local isotropic correction: −1.6 on the base's diagonal, off-diagonal untouched) but once
the 6 Å SOAP block is present it is worth only −0.4 on the diagonal — the long SOAP has already learned the collective local-field part.
User's call: not worth a label-space transform in the model; feature removed, numbers kept here. The CM-era cross-scoring gain (14.7 vs 16.6)
was on a base without the long SOAP, consistent with this.

### Round 12 (2026-10-28): hyperparameters by evidence (`gap_evidence`) and û inside the 3b kernel

**`gap_evidence`** maximises the sparse-GP log marginal likelihood over σ per component class (dipole: one; polarisability:
diagonal / off-diagonal, which subsumes λ_shear) and a prior amplitude δ per column group (each site model / SOAP scale ×
{iso, traceless, cross, rest}; 3b as one group), starting from the row σ, λ_shear and δ = 1. Implementation: the prior-whitened
design P is split into the groups, G_k = P_k P_kᵀ stored once (6F×6F each), −log p = ½[y_wᵀ(G_W+I)⁻¹y_w + log|G_W+I| − 2Σlog W]
with G_W = W(Σδ_k²G_k)W, L-BFGS-B over the log parameters with numeric gradients (one Cholesky per evaluation); the final solve
reuses the same path with the selected W and δ. Selected values in the run record (`gap.evidence`) and `model.hyper`.
Rows (σ selected by the optimiser, so one stage-A fit; refit on 500):

| row | start | selected | test 400 | test 500 | reference (grid) |
|---|---|---|---|---|---|
| chi EV U best (û winner) | σ 0.1, λ 2 | σ_diag 0.018, σ_off 0.010; δ iso 0.15, traceless 0.40, cross 0.07, 3b 0.38, soap6 iso/tr/cross 0.11/0.13/0.09 | 13.5/14.4 | **13.0/14.0** | 13.0/13.9 (σ 0.1, λ 2) |
| EV soap+2b(M40) d0.1 +cross2 (μ, d readout) | σ 0.003 | σ 0.0084; δ soap bond 1.45, cross 0.21, 2b bond 1.09, cross 0.20 | 1.56 | **1.49** | 1.45 (σ 0.003) |

Converged (28 / 24 iterations, −log p 149 → −3382 and −511 → −2496). The optimiser rediscovers the hand-tuned structure — a
diag/off noise ratio of 1.75 (λ_shear 2), a down-weighted cross group, σ/δ_iso ≈ 0.12 ≈ the grid's 0.1 — and lands within noise
of the grid for α (−0.0/+0.1) and 0.04 worse for μ, where it prefers a 3× larger σ than the test set does (the dipole fit is
capacity-limited and the evidence trades test accuracy for smoothness, the usual misspecification bias). Cost +40 % fit time and
+2 GB (the stored Grams). Verdict: a mature replacement for the σ grid and λ_shear (one unattended step, no re-tuning when the
column set changes) but **no gain in model performance**; left in the code as an option, off by default.

**û inside the 3b angular kernel** (`gap_models.angular_cross_tensor` normalising d_ij, d_ik; the fifth geometry site): the û
winner re-scored 13.1/14.1 (fit 500; 13.7/14.4 on 400) vs 13.0/13.9 with the full d_ij d_ikᵀ. Reverted; the kernel keeps its
r-weighting (comment at the site).

### Round 13 (2026-10-28): the dipole ladder with the compressed (trivial) l=α=7 SOAP (rows "trivial …"; σ ∈ {0.0003, 0.001, 0.003, 0.01} on fit 400 → val, best-val σ refit on 500 → test)

| step | compress none (ladder above) | compress trivial | σ (trivial) | µs/atom none / trivial |
|---|---|---|---|---|
| SOAP-7 alone | 1.56 | 1.64 | 0.001 | 430 / 293 |
| + 2b(M40) δ 0.1 | 1.49 | 1.52 | 0.003 | 450 / 295 |
| + cross C_x 2 | **1.45** | **1.47** | 0.003 | 410 / 306 |

Compression costs 0.08 on SOAP alone and 0.02–0.03 once the 2b and cross terms are present (within ~noise of 0.03), for a 25–30 %
cheaper forward pass (the 840- vs ~200-component kernel). The compressed SOAP-alone row wants the smaller σ 0.001 (less
capacity → less ridge), the composite rows 0.003 as before. The uncompressed production model stays; the trivial one is the
drop-in if forward cost matters.
**Decision (user, 2026-10-28): trivial compression is the standard for both GAP models.** `GAPconfig.compress_mode` defaults to
"trivial"; the scan's MU dict uses it (every μ row above round 13 was fitted with compress none); production dipole model =
trivial SOAP-7 + 2b(M40, δ 0.1) + cross C_x 2, σ 0.003 → 1.47 % at 0.31 ms/atom.

### Cleanup (2026-10-28, `docs/gap_cleanup.md` Tier 1 + cutoffs + GAPconfig + artefacts)

Removed from the code (every result they produced stays in this README and in `docs/gap_dipole_polarisability_models.md`):
`gap_ewald.py`, `gap_bond.py`; the TurboGAP 2b/3b GAP-fit column blocks (`gap_2b`/`gap_3b` and the `gap2b_*`/`gap3b_*`
knobs on GAPconfig — `gap_descriptor.py` stays for the ANN's `gap_sum`); `gap_triplet_tensor`; the `gap_angular_cross` column
group (the angular kernel functions stay: the 3b site model is built on them); progressive fits (`fit_gap_progressive`);
`gap_mean_prior`; `gap_evidence`; the ξ-list sum of kernels (`gap_kernel_power` is an int); `gap_group_delta`; `gap_site_rcut`
(2b/3b descriptors use `dipole_rc`); the `cubic`, `tanh3` and `none` cutoffs in all three twins (TNEP._fc, data._radial_weights,
gap_gpu._radial; `dipole_cutoff=None` → `cosine2` for every radial, screened included); GAPconfig's ANN-only fields are
frozen plain class attributes (not fields, not serialised). Artefacts: 27 one-off scan/probe/chain scripts, the run-record
directories under out/, the queue files and all GAP/*.json records (the logs in out/ stay). The scan driver is now
`gap_scan.py` (rows = the two production models and their ladders). Tests of the removed features deleted; the GAP suites
(test_gap_dipole/gpu/models/descriptor, two_atom) and the neighbour-readout suite were re-run green.
Also removed (user, 2026-10-28): the polarisability Ewald groups `pol_field_groups` / `pol_field_scale` / `pol_induced_alpha`
with their data.py machinery (Ewald dipole tensor, Applequist response, induced site tensors, QEq site tensors, molecule
pairing). The 12.0/13.0 and 13.4/13.6 rows above are the record of them. The dipole Ewald field column (`dipole_field`,
`dipole_field_charges`, `dipole_field_scale`) stays as an experimental option outside the production models.
**Module partition (2026-10-28).** `GAPconfig` moved from TNEPconfig.py to `gap_config.py`; `gap_models` no longer imports the
TurboGAP descriptor module (own `pair_index`); that ANN-only module is renamed `turbogap_descriptor.py` (tests
`test_turbogap_descriptor.py`). GAP modules: gap_config, gap_dipole, gap_models, gap_gpu. Shared on purpose: data.py (reading,
splits, V builder, radial weights, targets, field channel), DescriptorBuilder, model_io._serialize_config, TNEPconfig's readout rules.
**TurboGAP descriptors removed (user, 2026-10-28).** `turbogap_descriptor.py` (the gap.f90 2b/3b distance descriptors) and
its tests are gone together with the ANN's `nep_descriptor="gap_sum"` mode and the TNEPconfig `gap2b_*`/`gap3b_*`/`gap_buffer`/
`gap_models`/`gap_2b`/`gap_3b` fields: neither production model used them (the GAP fit dropped the column blocks in Tier 1, the
ANN default is nep_descriptor="off"). The ANN keeps "off" / "concat" / "replace" / "sum" (NEP experts). Results: docs §7 and the
rounds above.
**Descriptor modules (2026-10-28).** `DescriptorBuilderGPU.py` → `descriptor_layout.py` (layout/config helpers and the NumPy
reference pieces the TF builder and tests use; 2 683 lines of never-runnable fixture validation and the unused NumPy builder
class removed). `DescriptorBuilderGPU_tf.py` is the single GPU builder (phase-11 validation and `__main__` removed).
