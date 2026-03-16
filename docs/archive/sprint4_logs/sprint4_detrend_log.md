# Sprint 4 Item 4 Log (Detrend Sweep)

## Baseline Summary
| split | tpr_sig_target | fpr_sig_target | score_tpr_minus_fpr_target | passes_balanced_bar |
| --- | --- | --- | --- | --- |
| train | 0.0000 | 0.0042 | -0.0042 | False |
| holdout | 0.0000 | 0.0208 | -0.0208 | False |

## Holdout Candidate Ranking (Top 10)
| split | candidate_id | detector_variant | detrend_order | tpr_sig_target | fpr_sig_target | score_tpr_minus_fpr_target | delta_tpr | delta_fpr | delta_score | passes_balanced_bar | is_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | D00 | detrended_fft_sig | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0208 | 0.0208 | False | False |
| holdout | D01 | detrended_fft_sig | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0208 | 0.0208 | False | False |
| holdout | D02 | detrended_fft_sig | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0208 | 0.0208 | False | False |

## Decision
| best_candidate_id | best_detrend_order | holdout_delta_tpr | holdout_delta_fpr | holdout_delta_score | holdout_tpr_target | holdout_fpr_target | train_delta_tpr | train_delta_fpr | train_delta_score | meets_delta_tpr_bar | meets_fpr_bar | meets_delta_score_bar | meets_train_consistency | recommended_action | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D00 | 1 | 0.0000 | -0.0208 | 0.0208 | 0.0000 | 0.0000 | 0.0000 | -0.0042 | 0.0042 | False | True | True | True | STOP_OR_PIVOT | No holdout candidate reached minimum target-group TPR gain threshold. |
