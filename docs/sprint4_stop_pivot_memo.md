# Sprint 4 Stop/Pivot Memo (March 10, 2026)

## Decision
- Final Sprint 4 status: **STOP_OR_PIVOT**
- Gate outcome: no tested detector family in Sprint 4 delivered meaningful target-group TPR lift (`mid`, `boat_drift`) under the pre-registered significance framework.

## Evidence Snapshot

### Item 1: Window/Band Optimization
- Result: no target-group movement.
- Baseline target score (`TPR-FPR`): `-0.0042`
- Best candidate delta: `delta_tpr = 0.0000`, `delta_fpr = 0.0000`, `delta_score = 0.0000`
- Source: `docs/sprint4_window_band_log.md`

### Item 2: Welch vs Windowed FFT
- Result: score worsened on target group.
- Delta vs baseline: `delta_tpr = +0.0014`, `delta_fpr = +0.0167`, `delta_score = -0.0153`
- Source: `docs/sprint4_welch_log.md`

### Item 3: Tiled Windows + Multiple-Testing Control
- Result: reduced FPR but no TPR gain.
- Delta vs baseline: `delta_tpr = 0.0000`, `delta_fpr = -0.0042`, `delta_score = +0.0042`
- Tuning sweep outcome: `STOP_OR_PIVOT` (no candidate reached minimum TPR-gain threshold)
- Sources:
  - `docs/sprint4_tiled_log.md`
  - `docs/sprint4_tiled_tune/tiled_tune_decision.csv`
  - `docs/sprint4_tiled_tune_log.md`

### Item 4: Detrend Sweep (Train/Holdout)
- Result: holdout FPR improved, TPR unchanged.
- Best holdout candidate (`D00`, order `1`):
  - `holdout_delta_tpr = 0.0000`
  - `holdout_delta_fpr = -0.0208`
  - `holdout_delta_score = +0.0208`
- Gate status: failed TPR lift bar, action = `STOP_OR_PIVOT`
- Sources:
  - `outputs/sprint4_detrend_sweep/detrend_decision.csv`
  - `docs/sprint4_detrend_log.md`

## Interpretation
- Sprint 4 delivered calibration tightening in places (lower FPR), but did not recover meaningful true-positive sensitivity in the target regimes.
- Claims should remain conservative: this is a methods/calibration result, not observational QPO confirmation.

## Next Step (Approved Pivot)
- Move to **Item 5: Real-TTE bridge pilot** with explicit methods-validation framing.
- Objective: portability + calibration behavior on curated real bursts, with no claim inflation.
