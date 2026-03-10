# Sprint 4 Candidate Experiments (Ranked)

## Goal
Recover meaningful true positives while keeping significance-calibrated false positives controlled.

## Ranking Table
| Rank | Experiment | Expected TPR Gain | FPR Risk | Est. Effort | Why This Order | Primary Go/No-Go Metric |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Window-and-band optimization on existing injection grid | High | Low-Medium | 1-2 days | Highest expected sensitivity gain without changing surrogate calibration core | `mid` or `boat_drift` reaches `TPR_sig >= 0.2` with `FPR_sig <= 0.1` |
| 2 | Multi-taper/Welch peak statistic under same phase-randomized null | Medium-High | Low-Medium | 2-3 days | Likely to stabilize peak estimates and reduce noise volatility | Target-group mean `TPR_sig - FPR_sig` improves vs Sprint 3 baseline |
| 3 | Transient tiled-window detector with family-wise/FDR control | Medium | Medium | 2-4 days | Better match for short-lived/partial-window periodic structure | `boat_transient` TPR rises and global FPR remains <= 0.1 |
| 4 | Detrending model sweep (order + robust trend fit + holdout seeds) | Medium | Medium | 1-2 days | Current detrending was not enough; this isolates trend-removal failure modes | `mid` and `boat_drift` each gain >=0.05 absolute TPR vs current detrended run |
| 5 | Real-TTE bridge pilot (small curated set, methods framing only) | Low (sim TPR) / High (external validity) | Medium | 3-5 days | Confirms transfer behavior and failure patterns before broader scope | Complete p-value calibration report with explicit negative/mixed framing |

## Execution Policy
- Execute one item at a time in rank order.
- Reuse fixed seed grids and surrogate settings for comparability.
- After each item, re-score against the same balanced bar:
  - `mid`, `boat_drift`: `TPR_sig >= 0.6` at `B>=0.2` and `FPR_sig <= 0.1` at `B=0`.
- Stop and document if no item shows material movement (`delta TPR_sig < 0.05`) in target scenarios.

## Baseline Reference (for deltas)
- Source run: Sprint 3 full balanced benchmark on March 9, 2026.
- Artifacts:
  - `outputs/detector_variants/variant_comparison.csv`
  - `outputs/detector_variants/best_variant_selector.csv`
  - `docs/sprint3_decision_log.md`

## Item 1 Status (March 9, 2026)
- Full Item 1 sweep completed with `n_replicates=40` and `n_surrogates=200` across:
  - Bands: `0.30-0.40`, `0.35-0.45`, `0.38-0.48`
  - Window padding: `5s`, `10s`, `20s`
- Result: no target-group TPR lift (`delta TPR = 0.0` vs baseline) and no balanced-bar passes.
- Decision: move to Item 2 (multi-taper/Welch statistic) as the next logical experiment.
- Artifacts:
  - `outputs/sprint4_window_band/candidate_summary.csv`
  - `outputs/sprint4_window_band/best_candidate.csv`
  - `docs/sprint4_window_band_log.md`

## Item 2 Status (March 9, 2026)
- Full Item 2 run completed with `n_replicates=40` and `n_surrogates=200` using the same benchmark grid.
- Compared `windowed_fft_sig` baseline vs `welch_fft_sig` under the same phase-randomized surrogate null.
- Target-group result (`mid` + `boat_drift`):
  - `delta_score (TPR-FPR) = -0.0153` (worse than baseline)
  - `delta_tpr = +0.0014`
  - `delta_fpr = +0.0167`
- Decision: Item 2 did not improve target-group score; move to Item 3 (transient tiled-window + multiple-testing control).
- Artifacts:
  - `outputs/sprint4_welch_compare/welch_comparison.csv`
  - `outputs/sprint4_welch_compare/welch_decision.csv`
  - `docs/sprint4_welch_log.md`
