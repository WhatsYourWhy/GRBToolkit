# Sprint 4 Item 3 Log (Tiled Compare)

## Comparison Summary
| scenario | detector_variant | tpr_sig | fpr_sig | score_tpr_minus_fpr | delta_score_vs_windowed | delta_tpr_vs_windowed | delta_fpr_vs_windowed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| boat_drift | windowed_fft_sig | 0.0000 | 0.0083 | -0.0083 | 0.0000 | 0.0000 | 0.0000 |
| boat_drift | tiled_window_fft_sig | 0.0000 | 0.0000 | 0.0000 | 0.0083 | 0.0000 | -0.0083 |
| boat_transient | windowed_fft_sig | 0.0361 | 0.0333 | 0.0028 | 0.0000 | 0.0000 | 0.0000 |
| boat_transient | tiled_window_fft_sig | 0.0417 | 0.0000 | 0.0417 | 0.0389 | 0.0056 | -0.0333 |
| mid | windowed_fft_sig | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mid | tiled_window_fft_sig | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| short_100 | windowed_fft_sig | 0.0056 | 0.0000 | 0.0056 | 0.0000 | 0.0000 | 0.0000 |
| short_100 | tiled_window_fft_sig | 0.0056 | 0.0000 | 0.0056 | 0.0000 | 0.0000 | 0.0000 |
| short_70 | windowed_fft_sig | 0.0083 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 0.0000 |
| short_70 | tiled_window_fft_sig | 0.0083 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 0.0000 |
| weak | windowed_fft_sig | 0.0083 | 0.0000 | 0.0083 | 0.0000 | 0.0000 | 0.0000 |
| weak | tiled_window_fft_sig | 0.0056 | 0.0000 | 0.0056 | -0.0028 | -0.0028 | 0.0000 |
| target_group | windowed_fft_sig | 0.0000 | 0.0042 | -0.0042 | 0.0000 | 0.0000 | 0.0000 |
| target_group | tiled_window_fft_sig | 0.0000 | 0.0000 | 0.0000 | 0.0042 | 0.0000 | -0.0042 |

## Decision
| baseline_variant | item3_variant | target_score_windowed | target_score_tiled | delta_score | delta_tpr | delta_fpr | improves_target_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| windowed_fft_sig | tiled_window_fft_sig | -0.0042 | 0.0000 | 0.0042 | 0.0000 | -0.0042 | True |

## Recommended Next Step
Keep tiled-window path and refine correction/window policy.
