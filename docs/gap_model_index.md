# GAP / SNES model index for the report (verification)

Every test number in `docs/gap_dipole_polarisability_models.md` comes from a scan log under
`diagnostics/gap_experiments/out/` (untracked, local); each fit also writes a run record
`<scan>/GAP/gap_M*_x2_<timestamp>.json` holding the full config, the sparse-set sizes, column count
and fit time (records do not hold test metrics — those are the `FINAL …` lines in the logs). The
500-frame records below are matched to the report rows by their config fields. Test spreads used
for RMSE: μ component 6.24 e·a₀/cell; α diag 0.887, off-diag 0.468 Å³/cell (`datasets/test_waterbulk.xyz`).

## Dipole (test RRMSE; RMSE e·a₀/cell)

| report row | log (FINAL line) | record (out/…/GAP/) | RRMSE | RMSE |
|---|---|---|---|---|
| §3.1 original, l=α=4 trivial, no field | mubase.log `FINAL original l=a=4 trivial nofield fit500` | mubase/gap_M250_x2_20261020_221435.json | 1.76 % | 0.110 |
| l=α=7 uncompressed, no field | mubase.log `FINAL soap7 none nofield fit500` | mubase/gap_M250_x2_20261020_221629.json | 1.56 % | 0.097 |
| §3.2 best local: + 2b δ 0.1, no field | probe3_mu.log `FINAL soap+2b d0.1 nofield sigma=0.003` | probe_mu/gap_M250_x2_20261020_184640.json | 1.49 % | 0.093 |
| l=α=7 + Ewald field | probe_mu.log `FINAL soap sigma=0.001` (also site_mu.log) | probe_mu/gap_M250_x2_20261020_121351.json | 1.08 % | 0.067 |
| §3.3 best: + field + 2b δ 0.1, σ 2e-3 | probe_mu.log `FINAL soap+2b d0.1 sigma=0.002` | probe_mu/gap_M250_x2_20261020_121623.json | 0.97 % | 0.060 |
| same, δ 0.05, σ 1.5e-3 | probe2_mu.log `FINAL soap+2b d0.05 sigma=0.0015` | probe_mu/gap_M250_x2_20261020_124349.json | 0.96 % | 0.060 |
| site-model scan: soap+2b δ 0.3 (field) | site_mu.log `FINAL soap+2b` | site_mu/gap_M250_x2_20261020_012440.json | 1.00 % | 0.062 |
| soap+3b δ 0.3 (field) | site_mu.log `FINAL soap+3b` | site_mu/gap_M250_x2_20261020_012927.json | 1.02 % | |
| soap+2b+3b δ 0.3 (field) | site_mu.log `FINAL soap+2b+3b` | site_mu/gap_M250_x2_20261020_013215.json | 1.05 % | |
| 6 Å readout, field / no field | probe3_mu.log `FINAL … ro6.0 …` | probe_mu/…_184039 (field), …_184758 (no field) | 1.41 / 1.42 % | |

## Polarisability (test RRMSE diag/off; RMSE Å³/cell)

| report row | log (FINAL line) | record (out/…/GAP/) | RRMSE | RMSE |
|---|---|---|---|---|
| original two bond groups | twogroup.log `FINAL two-groups fit500` | twogroup/gap_M250_x2_20261020_221132.json | 16.6 / 19.9 | 0.147 / 0.093 |
| + cross group C_x 4 (base) | polgroups.log `FINAL base sigma=0.1` | polgroups/gap_M250_x2_20261020_204520.json | 14.9 / 16.0 | 0.132 / 0.075 |
| **+ induced coupling (recommended)** | polgroups.log `FINAL +induced sigma=0.1` | polgroups/gap_M250_x2_20261020_204645.json | 14.0 / 14.6 | 0.124 / 0.068 |
| + induced + quad | polgroups.log `FINAL +induced+quad sigma=0.1` | polgroups/gap_M250_x2_20261020_204808.json | 13.9 / 14.5 | 0.123 / 0.068 |
| + all five groups | polgroups.log `FINAL +all sigma=0.1` | polgroups/gap_M250_x2_20261020_204934.json | 13.9 / 14.3 | 0.123 / 0.067 |
| + 6.5 Å scale, σ 0.3 | probe_pol.log `FINAL soap ms6.5 sigma=0.3` (also polgroups.log) | probe_pol/gap_M250_x2_20261020_112848.json | 13.9 / 16.2 | 0.123 / 0.076 |
| + induced + quad + 6.5 Å, σ 0.3 / 0.1 | polgroups.log `FINAL +induced+quad ms6.5 sigma=…` | polgroups/…_211428 (σ 0.3), …_210950 (σ 0.1) | 13.0 / 15.2 · 13.9 / 14.6 | 0.115 / 0.071 · 0.123 / 0.068 |
| + angular cross L1 + induced (C_x 2) | angcross.log `FINAL 2groups cross2 +ang L1 +induced` | angcross/gap_M250_x2_20261020_233513.json | 14.4 / 15.2 | |
| + triplet tensor M10 / M20 / M20+induced (C_x 2) | triplet.log `FINAL …` | triplet/…_000756, …_000944, …_000522 | 15.5/17.0 · 16.3/17.7 · 15.5/16.5 | |
| SNES two-group ANN, gen 20k | out/snes_score_n15_q840_pop100_20261018_231703.log (`Final-gen test set` block) | models/n15_q840_pop100_20261018_231703/checkpoint.h5 | 35.7 / 27.5 | 0.317 / 0.129 |

The SNES RRMSEs are sqrt(1 − R²) per component from the pipeline's final-generation scoring
(R² xx/yy/zz 0.874/0.872/0.883, xy/yz/zx 0.910/0.925/0.921), pooled with the per-component test
spreads; the pooled RMSE printed by the pipeline is 0.2395 Å³. Do not use the "Best-val" block of
that log: for GPUMD-path runs there is no validation split and best_μ is not a trained model.
