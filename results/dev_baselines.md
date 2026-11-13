# Baselines on the development set

Git commit `9449908c9f268a1c14e319f0bf89449ce2ce171c` (with uncommitted changes). Dataset rows: 5,849 development trials.

Development trials: 5,849 (primary completion before 2021-01-01). No test-set trial is read.

_Plain predictors of the headline label (any treatment arm beats control), to show what the benchmark reads. Nothing here is tuned. Start year is deliberately not an input to any model._

Calibration gap is mean predicted minus observed success, in points (positive: too optimistic). Accuracy should be read against the always-majority column. The leaky and results-structure models are deliberately wrong: they read fields that exist only after the trial ended or are taken from posted results.

Intervals are 95% percentile intervals from 1,000 bootstrap resamples of lead drugs: each resample draws whole drugs, with all their trials (a trial with no known lead drug counts as its own). Trials that share a sponsor are still treated as independent, so the intervals are somewhat too narrow.

## A. Time-forward validation

Trained on 3,945 trials finishing before 2017-01-01; scored on 1,904 finishing from 2017-01-01 to 2020-12-31.

1,904 trials; observed success rate 53.6%.

_Every model on the same trials: how well it ranks trials (AUROC), how far off its average is, and how it compares with always saying the common answer._

| Model | Mean predicted | Calibration gap (points) | AUROC | Brier | Accuracy at 0.5 | Always-majority accuracy |
|---|---|---|---|---|---|---|
| constant | 60.8% | +7.2 (4.3 to 10.0) | 0.500 (0.500 to 0.500) | 0.254 | 53.6% (50.7 to 56.5) | 53.6% |
| phase lookup | 59.3% | +5.8 (3.0 to 8.5) | 0.614 (0.592 to 0.636) | 0.239 | 61.7% (59.4 to 64.0) | 53.6% |
| registration | 58.3% | +4.7 (2.4 to 7.1) | 0.713 (0.686 to 0.739) | 0.218 | 64.7% (62.2 to 67.3) | 53.6% |
| registration, no modality | 58.0% | +4.5 (2.1 to 6.8) | 0.714 (0.688 to 0.739) | 0.217 | 64.2% (61.7 to 66.7) | 53.6% |
| leaky | 58.2% | +4.6 (2.3 to 6.9) | 0.739 (0.714 to 0.763) | 0.211 | 66.4% (64.0 to 68.7) | 53.6% |
| results structure | 57.6% | +4.0 (1.7 to 6.4) | 0.715 (0.689 to 0.740) | 0.217 | 65.1% (62.7 to 67.6) | 53.6% |

_How much better one model ranks trials than another, using the same resamples._

| AUROC difference | Estimate (95% interval) |
|---|---|
| registration minus phase lookup | +0.100 (0.076 to 0.124) |
| leaky minus registration | +0.026 (0.012 to 0.038) |
| registration minus registration, no modality | -0.001 (-0.006 to 0.003) |

_The registration model within subsets of these trials. modality, phase, sponsor class are inputs to the model, so within-subset AUROC is lower partly because the model's strongest input is held fixed. Not model inputs: control kind (it comes from the results section); enrollment (actual, post-hoc) (it is the actual figure, recorded after the trial)._

| Grouping | Subset | n | Observed success | AUROC (95% interval) | Calibration gap (points, 95% interval) |
|---|---|---|---|---|---|
| control kind | placebo | 1598 | 55.1% | 0.719 (0.692 to 0.745) | +4.5 (2.1 to 6.9) |
| control kind | active | 246 | 46.7% | 0.700 (0.627 to 0.765) | +5.6 (-0.4 to 11.8) |
| enrollment (actual, post-hoc) | <100 | 524 | 41.2% | 0.647 (0.598 to 0.694) | +5.4 (1.1 to 9.3) |
| enrollment (actual, post-hoc) | 100-299 | 618 | 50.2% | 0.691 (0.649 to 0.734) | +6.6 (2.7 to 10.4) |
| enrollment (actual, post-hoc) | 300-999 | 602 | 64.1% | 0.735 (0.687 to 0.778) | +3.3 (-0.6 to 7.7) |
| enrollment (actual, post-hoc) | 1000+ | 160 | 67.5% | 0.639 (0.551 to 0.734) | +0.6 (-7.3 to 7.7) |
| modality | small molecule | 1412 | 52.5% | 0.707 (0.672 to 0.738) | +4.3 (1.8 to 7.1) |
| modality | biologic | 413 | 57.6% | 0.743 (0.686 to 0.796) | +7.2 (2.2 to 12.5) |
| modality | advanced | 24 | 62.5% | not computed (n < 30) | not computed (n < 30) |
| modality | mixed | 41 | 43.9% | 0.595 (0.413 to 0.784) | +2.3 (-10.2 to 19.1) |
| modality | vaccine | 12 | 58.3% | not computed (n < 30) | not computed (n < 30) |
| phase | phase2 | 856 | 41.0% | 0.659 (0.617 to 0.698) | +4.9 (1.7 to 7.9) |
| phase | phase3 | 1048 | 63.8% | 0.686 (0.649 to 0.723) | +4.6 (1.5 to 7.7) |
| sponsor class | industry | 1535 | 59.7% | 0.689 (0.661 to 0.718) | +4.3 (1.8 to 6.9) |
| sponsor class | not industry | 369 | 28.2% | 0.548 (0.485 to 0.613) | +6.4 (1.3 to 10.7) |

Trials in no subset row (values not shown): control kind: no treatment (23), unspecified (37); modality: missing (2). Control kinds other than placebo and active are not shown: 'no treatment' and 'unspecified' are too few to compare and are not the placebo-against-active contrast.

Bootstrap resamples skipped for holding one class only, headline tables: at most 0 of 1000 in any row. Bootstrap resamples skipped for holding one class only, subset table: at most 0 of 1000 in any row.

## B. Five-fold cross-validation

All development trials, 5 stratified shuffled folds (seed 0); each trial is predicted by models that did not see it. Subsets are not repeated here.

5,849 trials; observed success rate 58.4%.

_Every model on the same trials: how well it ranks trials (AUROC), how far off its average is, and how it compares with always saying the common answer._

| Model | Mean predicted | Calibration gap (points) | AUROC | Brier | Accuracy at 0.5 | Always-majority accuracy |
|---|---|---|---|---|---|---|
| constant | 58.4% | +0.0 (-1.9 to 2.0) | 0.500 (0.486 to 0.514) | 0.243 | 58.4% (56.4 to 60.3) | 58.4% |
| phase lookup | 58.4% | +0.0 (-1.7 to 1.8) | 0.606 (0.591 to 0.622) | 0.231 | 62.5% (61.2 to 63.8) | 58.4% |
| registration | 58.5% | +0.0 (-1.3 to 1.5) | 0.729 (0.711 to 0.745) | 0.207 | 68.4% (66.8 to 69.9) | 58.4% |
| registration, no modality | 58.4% | +0.0 (-1.3 to 1.5) | 0.729 (0.712 to 0.746) | 0.207 | 68.1% (66.6 to 69.5) | 58.4% |
| leaky | 58.5% | +0.0 (-1.2 to 1.3) | 0.757 (0.741 to 0.771) | 0.198 | 70.0% (68.5 to 71.4) | 58.4% |
| results structure | 58.5% | +0.0 (-1.3 to 1.4) | 0.735 (0.718 to 0.752) | 0.205 | 68.9% (67.4 to 70.3) | 58.4% |

_How much better one model ranks trials than another, using the same resamples._

| AUROC difference | Estimate (95% interval) |
|---|---|
| registration minus phase lookup | +0.123 (0.105 to 0.139) |
| leaky minus registration | +0.028 (0.021 to 0.035) |
| registration minus registration, no modality | -0.001 (-0.002 to 0.001) |

The constant model's AUROC is 0.4996 rather than 0.5 because each fold predicts a slightly different constant (its own training rate) and the out-of-fold predictions are pooled.

Bootstrap resamples skipped for holding one class only, headline tables: at most 0 of 1000 in any row.

## C. Time-forward against cross-validated, on the same trials

_Both columns score the same 1,904 trials (finished 2017 to 2020). Time-forward models were trained on earlier trials only. Cross-validated models were trained on a random 80% of all development trials, so they had seen trials from the same years. Differences are cross-validated minus time-forward, from shared bootstrap resamples. Section B scores more trials than A, so its numbers cannot be compared with A's._

| Model | AUROC time-forward | AUROC cross-validated | AUROC difference (95% interval) | Gap time-forward (points) | Gap cross-validated (points) | Gap difference (points, 95% interval) |
|---|---|---|---|---|---|---|
| constant | 0.500 | 0.517 | +0.017 (-0.009 to 0.040) | +7.2 | +4.8 | -2.3 (-2.3 to -2.3) |
| phase lookup | 0.614 | 0.610 | -0.003 (-0.016 to 0.009) | +5.8 | +3.9 | -1.9 (-1.9 to -1.9) |
| registration | 0.713 | 0.713 | +0.000 (-0.004 to 0.005) | +4.7 | +3.1 | -1.7 (-1.8 to -1.5) |
| registration, no modality | 0.714 | 0.714 | -0.001 (-0.005 to 0.003) | +4.5 | +3.0 | -1.5 (-1.6 to -1.3) |
| leaky | 0.739 | 0.743 | +0.004 (-0.001 to 0.009) | +4.6 | +2.6 | -2.0 (-2.2 to -1.8) |
| results structure | 0.715 | 0.717 | +0.002 (-0.003 to 0.007) | +4.0 | +2.7 | -1.4 (-1.5 to -1.2) |

For the registration model, ranking is 0.713 cross-validated against 0.713 time-forward, but the calibration gap is +3.1 points against +4.7. A random split hides how over-optimistic the model is; it does not flatter its ranking.

## D. What the registration model leans on

_Each row refits the registration model without that one column (trained before 2017, scored on the same trials as A) and shows how much AUROC it loses; a loss near zero means the other columns cover for it, not that the column says nothing, since related columns (phase and sponsor, modality and vaccine) can stand in for each other._

| Column dropped | AUROC without it | AUROC lost (95% interval) |
|---|---|---|
| phase | 0.668 | +0.045 (0.028 to 0.062) |
| sponsor_class | 0.693 | +0.020 (0.013 to 0.028) |
| therapeutic_area | 0.699 | +0.014 (0.002 to 0.025) |
| n_registered_arms | 0.700 | +0.013 (0.004 to 0.023) |
| intervention_model | 0.710 | +0.003 (-0.002 to 0.006) |
| allocation | 0.713 | +0.000 (-0.001 to 0.002) |
| is_vaccine | 0.713 | +0.000 (-0.000 to 0.000) |
| masking | 0.713 | -0.000 (-0.004 to 0.004) |
| modality | 0.713 | -0.000 (-0.005 to 0.004) |
| healthy_volunteers | 0.714 | -0.000 (-0.002 to 0.001) |
| has_dmc | 0.714 | -0.001 (-0.007 to 0.005) |

Logistic regression converged in every fit; a fit that failed to converge would have stopped the run.
