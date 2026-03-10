# Gamma-Ray Burst Substructure: A QPO-Driven Model with Adaptive Bayesian Blocks and Jet Dynamics

## Abstract
We present a corrected simulation and recovery workflow for GRB temporal substructure using a hybrid flux model: a FRED pulse envelope multiplied by QPO modulation, plus additive spike transients and background. The previous derivative-as-rate bug was removed, and all synthetic scenarios were regenerated from the corrected rate model. We then benchmarked significance-calibrated detection (phase-randomized surrogates, `alpha=0.05`) across three detector variants on a fixed injection grid. The full balanced Sprint 3 run (March 9, 2026; `n=40`, `B=0.0-0.4`, `p0_scale=0.5/1.0/2.0`, `n_surrogates=200`) did not pass the predefined balanced acceptance bar for `mid` and `boat_drift`, so current claims are limited to methods calibration and reproducibility, not observational QPO confirmation.

## 1. Model
The corrected photon-rate model is:

\[
F(t) = A e^{-(t-t_0)/\tau} \left(1 - e^{-(t-t_0)/\tau_r}\right)
\times \left[1 + B \cos\left(2\pi f_{\mathrm{QPO}} t + \phi\right)\right]
+ S_{\mathrm{spikes}}(t) + R_{\mathrm{bg}}
\]

Counts are sampled as:

\[
C(t) \sim \mathrm{Poisson}\left(F(t)\,\Delta t\right)
\]

This draft explicitly uses flux rate \(F(t)\), not \(dF/dt\), as the Poisson intensity.

## 2. Recovery-Test Results (Corrected Simulation)
Table 1 is generated from `run_core_refresh.py` outputs.

| GRB Type | Knots | Residual QPO | Residual FRED | FRED Knots | Edge (%) | f0 (Hz) | p0 | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Short 70 | 3 | 370.7044 | 399.1322 | 3 | 0.0000 | 0.4100 | 0.0179 | 17070 |
| Short 100 | 4 | 416.0669 | 480.6622 | 3 | 33.3300 | 0.4100 | 0.0178 | 17100 |
| Weak | 3 | 163.8950 | 165.5218 | 3 | 0.0000 | 0.4000 | 0.0186 | 17200 |
| Mid | 48 | 185.6391 | 318.0615 | 20 | 140.0000 | 0.4100 | 0.0165 | 17300 |
| BOAT Drift | 1996 | 271.3523 | 4048.2894 | 73 | 2634.2500 | 0.4120 | 0.0104 | 17400 |
| BOAT Transient | 1801 | 264.1373 | 3135.6183 | 71 | 2436.6200 | 0.4080 | 0.0109 | 17500 |

## 3. Adaptive BB Prior Sensitivity
We evaluated BOAT sensitivity by sweeping \(p_0\) around the adaptive baseline
\(p_0 = 0.02 \times e^{-0.00008\,\mathrm{Rate}} \times 0.95\) and recording knot counts.
Artifacts:

- Sensitivity CSV: `outputs/core_refresh/bb_sensitivity_boat.csv`
- Sensitivity Figure: `../figures/bb_sensitivity.png`

## 4. AIC Comparison
AIC snapshot for one representative burst (BOAT drift):

| model | log_likelihood | n_params | aic | delta_aic |
| --- | --- | --- | --- | --- |
| FRED+QPO+Spikes | 22608819.3552 | 10 | -45217618.7103 | 0.0000 |
| FRED+QPO | 21381608.8813 | 8 | -42763201.7626 | 2454416.9477 |
| FRED | 21360293.9660 | 5 | -42720577.9320 | 2497040.7783 |

## 5. Figures
- `../figures/short_70.png`
- `../figures/short_100.png`
- `../figures/weak.png`
- `../figures/mid.png`
- `../figures/boat_drift.png`
- `../figures/boat_transient.png`
- `../figures/bb_sensitivity.png`

## 6. Conclusions
This project now has a reproducible, corrected simulation baseline and a transparent significance-calibration framework, but the current detector family is not yet sufficient for strong positive recovery claims in the hardest target regimes. In the full balanced Sprint 3 benchmark, none of the evaluated variants met the pre-registered acceptance bar (`TPR_sig >= 0.6` at `B>=0.2` with `FPR_sig <= 0.1` at `B=0`) for `mid` and `boat_drift`. The most defensible interpretation is therefore a negative/mixed methods result: calibration appears controlled, sensitivity remains limited, and further detector development should be treated as an open methods problem rather than evidence of astrophysical QPO existence. Immediate next work is a focused Sprint 4 sequence that prioritizes sensitivity gains while explicitly protecting false-positive calibration.

## 7. Sprint 4 Candidate Experiments (Prioritized)
| Rank | Experiment | Expected TPR Gain | FPR Risk | Priority Rationale | Immediate Success Check |
| --- | --- | --- | --- | --- | --- |
| 1 | Window-and-band optimization on injection grid (fixed surrogate null) | High | Low-Medium | Biggest near-term leverage without changing null calibration model | `mid` or `boat_drift` reaches `TPR_sig >= 0.2` while `FPR_sig <= 0.1` |
| 2 | Multi-taper or Welch-style peak statistic under same surrogate test | Medium-High | Low-Medium | Reduces single-FFT variance and should improve peak stability | Improve target-group `TPR_sig - FPR_sig` vs current baseline |
| 3 | Transient-focused tiled windows with multiple-testing correction | Medium | Medium | Better match for nonstationary injections, especially `boat_transient` | `boat_transient` TPR increases without crossing `FPR_sig > 0.1` |
| 4 | Detrend model sweep (order/robust fit) with strict holdout seeds | Medium | Medium | Current detrend path helps little in targets; may need better trend removal | Target scenarios improve over current detrended variant by >=0.05 TPR |
| 5 | Real-data bridge pilot (limited TTE subset, methods claim only) | Low (simulation TPR) / High (external validity) | Medium | Validates portability and failure modes before broader claims | Complete pilot with calibrated p-value report and no claim inflation |

## Data Artifacts
- Metrics CSV: `outputs/core_refresh/scenario_metrics.csv`
- Table 1 CSV: `outputs/core_refresh/table1_source.csv`
- AIC CSV: `outputs/core_refresh/aic_comparison.csv`

<!-- BENCHMARK_SECTION_START -->
## Injection-Recovery and False-Positive Benchmark
This benchmark quantifies recovery behavior under controlled injections and explicitly separates methods validation from observational existence claims. Primary detection uses FFT band-peak significance in `0.35-0.45 Hz`, calibrated with phase-randomized surrogates and thresholded at `alpha=0.05`. Legacy frequency-hit rates are retained in CSV outputs for backward comparison.

| scenario | B | recovery_rate_sig | false_positive_rate_sig | recovery_rate_sig_ci_low | recovery_rate_sig_ci_high | median_knots | iqr_knots | median_residual_gain_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| boat_drift | 0.0000 | 0.0250 | 0.0250 | 0.0044 | 0.1288 | 667.0000 | 38.7500 | 92.5980 |
| boat_drift | 0.2000 | 0.0000 | nan | 0.0000 | 0.0876 | 763.5000 | 33.5000 | 92.9091 |
| boat_drift | 0.3000 | 0.0000 | nan | 0.0000 | 0.0876 | 807.0000 | 21.0000 | 93.2329 |
| boat_drift | 0.4000 | 0.0000 | nan | 0.0000 | 0.0876 | 838.5000 | 25.2500 | 93.6229 |
| boat_transient | 0.0000 | 0.0250 | 0.0250 | 0.0044 | 0.1288 | 637.0000 | 30.5000 | 91.5593 |
| boat_transient | 0.2000 | 0.0250 | nan | 0.0044 | 0.1288 | 642.0000 | 28.5000 | 91.5612 |
| boat_transient | 0.3000 | 0.0250 | nan | 0.0044 | 0.1288 | 643.5000 | 34.7500 | 91.5580 |
| boat_transient | 0.4000 | 0.0000 | nan | 0.0000 | 0.0876 | 643.0000 | 31.0000 | 91.5563 |
| mid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0876 | 20.0000 | 1.0000 | 2.2405 |
| mid | 0.2000 | 0.0000 | nan | 0.0000 | 0.0876 | 37.0000 | 4.0000 | 28.4035 |
| mid | 0.3000 | 0.0000 | nan | 0.0000 | 0.0876 | 51.0000 | 4.5000 | 43.3344 |
| mid | 0.4000 | 0.0000 | nan | 0.0000 | 0.0876 | 64.0000 | 5.0000 | 54.1982 |
| short_100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 12.0773 |
| short_100 | 0.2000 | 0.0000 | nan | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 11.6935 |
| short_100 | 0.3000 | 0.0000 | nan | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 11.3735 |
| short_100 | 0.4000 | 0.0000 | nan | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 11.4755 |
| short_70 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 6.7764 |
| short_70 | 0.2000 | 0.0000 | nan | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 6.9825 |
| short_70 | 0.3000 | 0.0000 | nan | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 7.1792 |
| short_70 | 0.4000 | 0.0000 | nan | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 7.5082 |
| weak | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 1.4076 |
| weak | 0.2000 | 0.0000 | nan | 0.0000 | 0.0876 | 3.0000 | 0.0000 | 1.1786 |
| weak | 0.3000 | 0.0250 | nan | 0.0044 | 0.1288 | 3.0000 | 0.0000 | 1.1418 |
| weak | 0.4000 | 0.0250 | nan | 0.0044 | 0.1288 | 3.0000 | 0.0000 | 1.3371 |

### Benchmark Artifacts
- `outputs/rigor_benchmark/run_level_results.csv`
- `outputs/rigor_benchmark/recovery_summary.csv`
- `../figures/recovery_heatmap_sig.png`
- `../figures/fpr_vs_p0_sig.png`
- `../figures/knot_stability.png`
- `../figures/pvalue_distribution.png`

This section is methods validation only and should not be interpreted as direct observational proof of QPO existence in real bursts.
<!-- DETECTOR_VARIANT_SECTION_START -->
### Detector Variant Comparison
We evaluated three significance-calibrated detector variants under a shared surrogate framework: global tapered FFT, transient-window FFT, and detrended FFT. Pass/fail was pre-registered using a balanced bar for `mid` and `boat_drift`: `TPR_sig >= 0.6` at `B>=0.2` and `FPR_sig <= 0.1` at `B=0`.

Sprint 3 outcome: **FAIL**

| detector_variant | scenario | TPR_sig | FPR_sig | delta_tpr_vs_baseline | delta_fpr_vs_baseline | passes_balanced_bar |
| --- | --- | --- | --- | --- | --- | --- |
| detrended_fft_sig | boat_drift | 0.0000 | 0.0000 | 0.0000 | -0.0083 | False |
| detrended_fft_sig | mid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| global_tapered_fft_sig | boat_drift | 0.0000 | 0.0083 | 0.0000 | 0.0000 | False |
| global_tapered_fft_sig | mid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |
| windowed_fft_sig | boat_drift | 0.0000 | 0.0083 | 0.0000 | 0.0000 | False |
| windowed_fft_sig | mid | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False |

Best-Variant Selector:

| selector_scope | selector_key | best_detector_variant | TPR_sig | FPR_sig | score_tpr_minus_fpr | passes_balanced_bar |
| --- | --- | --- | --- | --- | --- | --- |
| scenario | boat_drift | detrended_fft_sig | 0.0000 | 0.0000 | 0.0000 | False |
| scenario | boat_transient | global_tapered_fft_sig | 0.0556 | 0.0083 | 0.0472 | False |
| scenario | mid | detrended_fft_sig | 0.0000 | 0.0000 | 0.0000 | False |
| scenario | short_100 | detrended_fft_sig | 0.0806 | 0.0750 | 0.0056 | False |
| scenario | short_70 | global_tapered_fft_sig | 0.0083 | 0.0000 | 0.0083 | False |
| scenario | weak | global_tapered_fft_sig | 0.0083 | 0.0000 | 0.0083 | False |
| target_group | mid+boat_drift | detrended_fft_sig | 0.0000 | 0.0000 | 0.0000 | False |

Artifacts:
- `outputs/detector_variants/variant_comparison.csv`
- `outputs/detector_variants/best_variant_selector.csv`
- `../figures/variant_tpr_fpr_tradeoff.png`
- `../figures/variant_roc_like_grid.png`

Interpretation is conservative: this remains methods validation and calibration, not observational proof of astrophysical QPO existence.
<!-- DETECTOR_VARIANT_SECTION_END -->


<!-- BENCHMARK_SECTION_END -->




<!-- TTE_BRIDGE_SECTION_START -->
## Real-TTE Bridge Pilot (Methods Validation)
We ran a small curated TTE bridge pilot using the same significance-calibrated FFT detector (phase-randomized surrogates) to evaluate portability and calibration behavior on real bursts.

| burst_id | p_value | detected_sig | peak_freq_obs | null_empirical_fpr_alpha | detection_mode |
| --- | --- | --- | --- | --- | --- |
| bn090709630_n3 | 0.1244 | False | 0.3583 | 0.0125 | global |
| bn170817529_n5 | 0.6418 | False | 0.3625 | 0.0500 | global |

Summary:
| n_bursts | n_detected_sig | detected_fraction | median_p_value | mean_null_empirical_fpr_alpha | alpha | calibration_status | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 | 0.0000 | 0.3831 | 0.0312 | 0.0500 | calibrated_or_conservative | Methods calibration check only; this pilot is not observational proof of QPO existence. |

Robustness sweep (seeds x bin widths x bands):
- `18` runs total, `0` runs with significance detections, `max_detected_fraction=0.0000`.
- Null calibration remained controlled (`max mean_null_empirical_fpr_alpha=0.0750`, threshold check `<= 1.5 * alpha`).
- Solidify decision: `LOCK_MIXED_NEGATIVE_METHODS_CLAIM`.

Artifacts:
- `outputs/sprint5_tte_bridge/tte_bridge_results.csv`
- `outputs/sprint5_tte_bridge/tte_bridge_summary.csv`
- `outputs/sprint5_tte_solidify/solidify_summary_matrix.csv`
- `outputs/sprint5_tte_solidify/solidify_decision.csv`
- `../figures/sprint5_tte_pvalues.png`
- `../figures/sprint5_tte_null_calibration.png`
- `../figures/sprint5_tte_solidify_detected_fraction.png`
- `../figures/sprint5_tte_solidify_null_fpr.png`

Interpretation remains methods-first: this pilot does not claim observational confirmation of QPOs.
<!-- TTE_BRIDGE_SECTION_END -->


## References
- Chattopadhyay, T., Misra, R., & Bhattacharyya, S. (2022). *The Astrophysical Journal*, 935, 157. https://doi.org/10.3847/1538-4357/ac7d5a
- Kumar, P., & Zhang, B. (2015). *Physics Reports*, 561, 1-109. https://doi.org/10.1016/j.physrep.2014.09.008
- Scargle, J. D., Norris, J. P., Jackson, B., & Chiang, J. (2013). *The Astrophysical Journal*, 764, 167. https://doi.org/10.1088/0004-637X/764/2/167
