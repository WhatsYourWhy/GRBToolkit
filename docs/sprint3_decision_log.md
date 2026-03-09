# Sprint 3 Decision Log

## Hypothesis
Detector variants can recover materially higher true-positive rates while preserving significance-calibrated false-positive control under shared surrogate null modeling.

## Configuration
- Timestamp: 2026-03-09 05:39:33
- Variants: global_tapered_fft_sig, windowed_fft_sig, detrended_fft_sig
- Scenarios: short_70, short_100, weak, mid, boat_drift, boat_transient
- B grid: 0.0, 0.1, 0.2, 0.3, 0.4
- p0 scale grid: 0.5, 1.0, 2.0
- Replicates per cell: 40
- Surrogates: 200
- Alpha: 0.05
- Frequency band: 0.35-0.45 Hz
- Window padding (s): 10.0
- Detrend order: 1
- Runtime (s): 2021.8

## Outcomes
- Sprint outcome: **FAIL**

| detector_variant | scenario | TPR_sig | FPR_sig | passes_balanced_bar |
| --- | --- | --- | --- | --- |
| detrended_fft_sig | boat_drift | 0.0000 | 0.0000 | False |
| detrended_fft_sig | mid | 0.0000 | 0.0000 | False |
| global_tapered_fft_sig | boat_drift | 0.0000 | 0.0083 | False |
| global_tapered_fft_sig | mid | 0.0000 | 0.0000 | False |
| windowed_fft_sig | boat_drift | 0.0000 | 0.0083 | False |
| windowed_fft_sig | mid | 0.0000 | 0.0000 | False |

## Best-Variant Selector
| selector_scope | selector_key | best_detector_variant | TPR_sig | FPR_sig | score_tpr_minus_fpr | passes_balanced_bar |
| --- | --- | --- | --- | --- | --- | --- |
| scenario | boat_drift | detrended_fft_sig | 0.0000 | 0.0000 | 0.0000 | False |
| scenario | boat_transient | global_tapered_fft_sig | 0.0556 | 0.0083 | 0.0472 | False |
| scenario | mid | detrended_fft_sig | 0.0000 | 0.0000 | 0.0000 | False |
| scenario | short_100 | detrended_fft_sig | 0.0806 | 0.0750 | 0.0056 | False |
| scenario | short_70 | global_tapered_fft_sig | 0.0083 | 0.0000 | 0.0083 | False |
| scenario | weak | global_tapered_fft_sig | 0.0083 | 0.0000 | 0.0083 | False |
| target_group | mid+boat_drift | detrended_fft_sig | 0.0000 | 0.0000 | 0.0000 | False |

## Pass/Fail Rule
- Balanced bar target scenarios: `mid`, `boat_drift`
- Pass threshold: `TPR_sig >= 0.6` at `B>=0.2` and `FPR_sig <= 0.1` at `B=0`

## Next-Step Decision
Pivot to negative/mixed methods framing and real-data bridge planning.
