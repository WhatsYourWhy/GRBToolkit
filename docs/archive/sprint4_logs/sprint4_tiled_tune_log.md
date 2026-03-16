# Sprint 4 Item 3 Tuning Log

## Baseline (Windowed FFT)
- Target TPR: `0.0000`
- Target FPR: `0.0000`
- Target score (`TPR-FPR`): `0.0000`

## Candidate Ranking (Top 10)
| candidate_id | tile_window_s | tile_step_s | tile_correction_method | tile_max_windows | tpr_sig_target | fpr_sig_target | score_tpr_minus_fpr_target | delta_tpr | delta_fpr | delta_score | passes_balanced_bar | rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T00 | 40.0000 | 10.0000 | bh | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 1 |
| T01 | 40.0000 | 10.0000 | bonferroni | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 2 |
| T02 | 40.0000 | 20.0000 | bh | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 3 |
| T03 | 40.0000 | 20.0000 | bonferroni | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 4 |
| T04 | 60.0000 | 10.0000 | bh | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 5 |
| T05 | 60.0000 | 10.0000 | bonferroni | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 6 |
| T06 | 60.0000 | 20.0000 | bh | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 7 |
| T07 | 60.0000 | 20.0000 | bonferroni | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | 8 |

## Stop/Go Decision
| best_candidate_id | best_delta_tpr | best_delta_fpr | best_delta_score | delta_score | best_fpr_target | best_tpr_target | meets_delta_tpr_bar | meets_fpr_bar | meets_delta_score_bar | recommended_action | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | False | True | False | STOP_OR_PIVOT | No candidate reached minimum TPR gain threshold. |
