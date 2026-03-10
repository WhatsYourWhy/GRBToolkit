# Sprint 5 TTE Solidify Log

## Sweep Summary (Top 12 by detected fraction, then calibration)
| run_id | seed | bin_width_s | freq_band_min | freq_band_max | band_label | n_bursts | n_detected_sig | detected_fraction | median_p_value | mean_null_empirical_fpr_alpha | calibration_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R002 | 701000 | 0.0500 | 0.3000 | 0.4000 | 0.30-0.40 | 2 | 0 | 0.0000 | 0.2811 | 0.0250 | calibrated_or_conservative |
| R004 | 701000 | 0.1000 | 0.3000 | 0.4000 | 0.30-0.40 | 2 | 0 | 0.0000 | 0.3010 | 0.0250 | calibrated_or_conservative |
| R012 | 702000 | 0.0200 | 0.3000 | 0.4000 | 0.30-0.40 | 2 | 0 | 0.0000 | 0.2836 | 0.0250 | calibrated_or_conservative |
| R003 | 701000 | 0.0500 | 0.3500 | 0.4500 | 0.35-0.45 | 2 | 0 | 0.0000 | 0.3831 | 0.0312 | calibrated_or_conservative |
| R005 | 701000 | 0.1000 | 0.3500 | 0.4500 | 0.35-0.45 | 2 | 0 | 0.0000 | 0.3035 | 0.0312 | calibrated_or_conservative |
| R017 | 702000 | 0.1000 | 0.3500 | 0.4500 | 0.35-0.45 | 2 | 0 | 0.0000 | 0.3159 | 0.0312 | calibrated_or_conservative |
| R015 | 702000 | 0.0500 | 0.3500 | 0.4500 | 0.35-0.45 | 2 | 0 | 0.0000 | 0.3856 | 0.0375 | calibrated_or_conservative |
| R000 | 701000 | 0.0200 | 0.3000 | 0.4000 | 0.30-0.40 | 2 | 0 | 0.0000 | 0.2886 | 0.0437 | calibrated_or_conservative |
| R001 | 701000 | 0.0200 | 0.3500 | 0.4500 | 0.35-0.45 | 2 | 0 | 0.0000 | 0.4378 | 0.0437 | calibrated_or_conservative |
| R013 | 702000 | 0.0200 | 0.3500 | 0.4500 | 0.35-0.45 | 2 | 0 | 0.0000 | 0.4229 | 0.0437 | calibrated_or_conservative |
| R006 | 701500 | 0.0200 | 0.3000 | 0.4000 | 0.30-0.40 | 2 | 0 | 0.0000 | 0.3109 | 0.0500 | calibrated_or_conservative |
| R008 | 701500 | 0.0500 | 0.3000 | 0.4000 | 0.30-0.40 | 2 | 0 | 0.0000 | 0.2687 | 0.0500 | calibrated_or_conservative |

## Decision
| n_runs | n_runs_with_detection | max_detected_fraction | mean_detected_fraction | max_null_empirical_fpr_alpha | mean_null_empirical_fpr_alpha | alpha | stable_non_detection | stable_calibration | recommended_action | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0 | 0.0000 | 0.0000 | 0.0750 | 0.0434 | 0.0500 | True | True | LOCK_MIXED_NEGATIVE_METHODS_CLAIM | No significance detections across sweep and null calibration remained controlled. |

## Interpretation Guardrail
- This sweep is a methods-validation robustness check.
- Use these results to calibrate claims and uncertainty language, not to assert astrophysical detection.
