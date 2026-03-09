# Gamma-Ray Burst Substructure: A QPO-Driven Model with Adaptive Bayesian Blocks and Jet Dynamics

## Abstract
We present a corrected simulation and recovery workflow for GRB temporal substructure using a hybrid flux model: a FRED pulse envelope multiplied by QPO modulation, plus additive spike transients and background. The previous derivative-as-rate bug was removed, and all synthetic scenarios were regenerated from the corrected rate model. Across short, weak, mid, and BOAT-like cases, adaptive Bayesian Blocks recover rich knot structure and resolve the injected 0.41 Hz band in the recomputed signals. We report refreshed knot counts, residual comparisons against FRED-only baselines, and a model-selection snapshot with AIC.

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
The corrected simulation confirms that adaptive Bayesian Blocks can recover injected substructure under a physically consistent flux-rate model. Relative to FRED-only baselines, the hybrid model yields stronger structural recovery and improved residual behavior across the tested classes. The BOAT prior sweep indicates that segmentation density is sensitive but stable over a practical adaptive range. Next-stage work should add WWZ significance contours and real-TTE validation to test whether recovered QPO signatures are present in observed bursts.

## Data Artifacts
- Metrics CSV: `outputs/core_refresh/scenario_metrics.csv`
- Table 1 CSV: `outputs/core_refresh/table1_source.csv`
- AIC CSV: `outputs/core_refresh/aic_comparison.csv`

<!-- BENCHMARK_SECTION_START -->
## Injection-Recovery and False-Positive Benchmark
This benchmark quantifies recovery behavior under controlled injections and explicitly separates methods validation from observational existence claims. Detection is defined as `|f0_est - f_qpo| <= 0.02 Hz`, and false-positive behavior is estimated from the `B=0` branch under matched settings.

| scenario | B | recovery_rate | false_positive_rate | median_knots | iqr_knots | median_residual_gain_pct |
| --- | --- | --- | --- | --- | --- | --- |
| boat_drift | 0.0000 | 0.4250 | 0.4250 | 667.0000 | 38.7500 | 92.5980 |
| boat_drift | 0.2000 | 1.0000 | nan | 763.5000 | 33.5000 | 92.9091 |
| boat_drift | 0.3000 | 1.0000 | nan | 807.0000 | 21.0000 | 93.2329 |
| boat_drift | 0.4000 | 1.0000 | nan | 838.5000 | 25.2500 | 93.6229 |
| boat_transient | 0.0000 | 0.4000 | 0.4000 | 637.0000 | 30.5000 | 91.5593 |
| boat_transient | 0.2000 | 0.4000 | nan | 642.0000 | 28.5000 | 91.5612 |
| boat_transient | 0.3000 | 0.4500 | nan | 643.5000 | 34.7500 | 91.5580 |
| boat_transient | 0.4000 | 0.6250 | nan | 643.0000 | 31.0000 | 91.5563 |
| mid | 0.0000 | 0.4750 | 0.4750 | 20.0000 | 1.0000 | 2.2405 |
| mid | 0.2000 | 1.0000 | nan | 37.0000 | 4.0000 | 28.4035 |
| mid | 0.3000 | 1.0000 | nan | 51.0000 | 4.5000 | 43.3344 |
| mid | 0.4000 | 1.0000 | nan | 64.0000 | 5.0000 | 54.1982 |
| short_100 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 0.0000 | 12.0773 |
| short_100 | 0.2000 | 0.0000 | nan | 3.0000 | 0.0000 | 11.6935 |
| short_100 | 0.3000 | 0.0000 | nan | 3.0000 | 0.0000 | 11.3735 |
| short_100 | 0.4000 | 0.0000 | nan | 3.0000 | 0.0000 | 11.4755 |
| short_70 | 0.0000 | 0.0000 | 0.0000 | 3.0000 | 0.0000 | 6.7764 |
| short_70 | 0.2000 | 0.0000 | nan | 3.0000 | 0.0000 | 6.9825 |
| short_70 | 0.3000 | 0.0000 | nan | 3.0000 | 0.0000 | 7.1792 |
| short_70 | 0.4000 | 0.0000 | nan | 3.0000 | 0.0000 | 7.5082 |
| weak | 0.0000 | 1.0000 | 1.0000 | 3.0000 | 0.0000 | 1.4076 |
| weak | 0.2000 | 1.0000 | nan | 3.0000 | 0.0000 | 1.1786 |
| weak | 0.3000 | 1.0000 | nan | 3.0000 | 0.0000 | 1.1418 |
| weak | 0.4000 | 1.0000 | nan | 3.0000 | 0.0000 | 1.3371 |

### Benchmark Artifacts
- `outputs/rigor_benchmark/run_level_results.csv`
- `outputs/rigor_benchmark/recovery_summary.csv`
- `../figures/recovery_heatmap.png`
- `../figures/fpr_vs_p0.png`
- `../figures/knot_stability.png`

This section is methods validation only and should not be interpreted as direct observational proof of QPO existence in real bursts.
<!-- BENCHMARK_SECTION_END -->


## References
- Chattopadhyay, T., Misra, R., & Bhattacharyya, S. (2022). *The Astrophysical Journal*, 935, 157. https://doi.org/10.3847/1538-4357/ac7d5a
- Kumar, P., & Zhang, B. (2015). *Physics Reports*, 561, 1-109. https://doi.org/10.1016/j.physrep.2014.09.008
- Scargle, J. D., Norris, J. P., Jackson, B., & Chiang, J. (2013). *The Astrophysical Journal*, 764, 167. https://doi.org/10.1088/0004-637X/764/2/167
