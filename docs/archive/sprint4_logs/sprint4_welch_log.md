# Sprint 4 Item 2 Log (Welch Compare)

## Comparison Summary
| scenario | detector_variant | tpr_sig | fpr_sig | score_tpr_minus_fpr | delta_score_vs_windowed | delta_tpr_vs_windowed | delta_fpr_vs_windowed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boat_drift | windowed_fft_sig | 0.0000 | 0.0083 | -0.0083 | 0.0000 | 0.0000 | 0.0000 |
| boat_drift | welch_fft_sig | 0.0000 | 0.0417 | -0.0417 | -0.0333 | 0.0000 | 0.0333 |
| boat_transient | windowed_fft_sig | 0.0361 | 0.0333 | 0.0028 | 0.0000 | 0.0000 | 0.0000 |
| boat_transient | welch_fft_sig | 0.0639 | 0.0417 | 0.0222 | 0.0194 | 0.0278 | 0.0083 |
| mid | windowed_fft_sig | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mid | welch_fft_sig | 0.0028 | 0.0000 | 0.0028 | 0.0028 | 0.0028 | 0.0000 |
| short_100 | windowed_fft_sig | 0.0056 | 0.0000 | 0.0056 | 0.0000 | 0.0000 | 0.0000 |
| short_100 | welch_fft_sig | 0.0056 | 0.0000 | 0.0056 | 0.0000 | 0.0000 | 0.0000 |
| short_70 | windowed_fft_sig | 0.0083 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 0.0000 |
| short_70 | welch_fft_sig | 0.0083 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 0.0000 |
| weak | windowed_fft_sig | 0.0083 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 0.0000 |
| weak | welch_fft_sig | 0.0111 | 0.0000 | 0.0111 | 0.0028 | 0.0028 | 0.0000 |
| target_group | windowed_fft_sig | 0.0000 | 0.0042 | -0.0042 | 0.0000 | 0.0000 | 0.0000 |
| target_group | welch_fft_sig | 0.0014 | 0.0208 | -0.0194 | -0.0153 | 0.0014 | 0.0167 |

## Decision
| baseline_variant | item2_variant | target_score_windowed | target_score_welch | delta_score | delta_tpr | delta_fpr | improves_target_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| windowed_fft_sig | welch_fft_sig | -0.0042 | -0.0194 | -0.0153 | 0.0014 | 0.0167 | False |

## Recommended Next Step
No score gain; move to next detector family.
