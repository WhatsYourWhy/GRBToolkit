# Sprint 4 Window/Band Optimization Log

## Baseline
- Candidate: `C04`
- Band: `0.35-0.45 Hz`
- Window padding: `10.0 s`
- Target mean TPR: `0.0000`
- Target mean FPR: `0.0042`
- Target mean score (`TPR-FPR`): `-0.0042`

## Best Candidate
- Candidate: `C00`
- Band: `0.30-0.40 Hz`
- Window padding: `5.0 s`
- Target mean TPR: `0.0000`
- Target mean FPR: `0.0042`
- Delta TPR vs baseline: `0.0000`
- Delta FPR vs baseline: `0.0000`
- Passes balanced bar: `False`

## Top Candidates
| candidate_id | freq_band_min | freq_band_max | window_padding_s | tpr_sig_target_mean | fpr_sig_target_mean | score_tpr_minus_fpr | passes_balanced_bar | delta_tpr_vs_baseline | delta_fpr_vs_baseline | is_baseline | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C00 | 0.3000 | 0.4000 | 5.0000 | 0.0000 | 0.0042 | -0.0042 | False | 0.0000 | 0.0000 | False | 1 |
| C01 | 0.3000 | 0.4000 | 10.0000 | 0.0000 | 0.0042 | -0.0042 | False | 0.0000 | 0.0000 | False | 2 |
| C02 | 0.3000 | 0.4000 | 20.0000 | 0.0000 | 0.0042 | -0.0042 | False | 0.0000 | 0.0000 | False | 3 |
| C03 | 0.3500 | 0.4500 | 5.0000 | 0.0000 | 0.0042 | -0.0042 | False | 0.0000 | 0.0000 | False | 4 |
| C04 | 0.3500 | 0.4500 | 10.0000 | 0.0000 | 0.0042 | -0.0042 | False | 0.0000 | 0.0000 | True | 5 |
| C05 | 0.3500 | 0.4500 | 20.0000 | 0.0000 | 0.0042 | -0.0042 | False | 0.0000 | 0.0000 | False | 6 |
| C06 | 0.3800 | 0.4800 | 5.0000 | 0.0000 | 0.0125 | -0.0125 | False | 0.0000 | 0.0083 | False | 7 |
| C07 | 0.3800 | 0.4800 | 10.0000 | 0.0000 | 0.0125 | -0.0125 | False | 0.0000 | 0.0083 | False | 8 |
| C08 | 0.3800 | 0.4800 | 20.0000 | 0.0000 | 0.0125 | -0.0125 | False | 0.0000 | 0.0083 | False | 9 |
