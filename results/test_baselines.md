# Baselines on the held-out test set

Git commit `2bc89fe3e89c56a7893941dc89caa129339892a2` (clean).

Fit on all 5,849 development trials. Trials scored: 1,236 test, 492 late.

Access note: Pre-registered baseline ladder, first and only run before any new modelling. Pre-registration emailed Wed 11 Nov 2026 06:21 GMT and locked at tag prereg.

## Access to the held-out labels

The test labels have been opened 1 time(s), counting this run. Every opening is listed, so a re-run cannot quietly replace an earlier result.

| Time (UTC) | Commit | Note |
|---|---|---|
| 2026-11-11T06:48:33+00:00 | `2bc89fe3e89c` | Pre-registered baseline ladder, first and only run before any new modelling. Pre-registration emailed Wed 11 Nov 2026 06:21 GMT and locked at tag prereg. |

Calibration gap is mean predicted minus observed success, in points (positive: too optimistic). Accuracy should be read against the always-majority column. The leaky and results-structure models are deliberately wrong: they read fields that exist only after the trial ended or are taken from posted results.

Intervals are 95% percentile intervals from 1,000 bootstrap resamples of trials. The bootstrap treats trials as independent; they are not (the same drugs and sponsors recur), so the intervals are somewhat too narrow.

## Primary test set (finished 2021 to 2023)

1,236 trials; observed success rate 50.6%.

_Every model on the same trials: how well it ranks trials (AUROC), how far off its average is, and how it compares with always saying the common answer._

| Model | Mean predicted | Calibration gap (points) | AUROC | Brier | Accuracy at 0.5 | Always-majority accuracy |
|---|---|---|---|---|---|---|
| constant | 58.4% | +7.8 (5.2 to 10.8) | 0.500 (0.500 to 0.500) | 0.256 | 50.6% (47.6 to 53.2) | 50.6% |
| phase lookup | 57.4% | +6.8 (4.2 to 9.7) | 0.630 (0.604 to 0.657) | 0.238 | 63.0% (60.5 to 65.8) | 50.6% |
| registration | 55.3% | +4.6 (2.3 to 7.3) | 0.730 (0.702 to 0.759) | 0.213 | 68.2% (65.5 to 70.7) | 50.6% |
| registration, no modality | 55.1% | +4.4 (2.1 to 7.2) | 0.731 (0.702 to 0.760) | 0.212 | 67.6% (64.7 to 70.1) | 50.6% |
| leaky | 52.5% | +1.9 (-0.4 to 4.6) | 0.779 (0.751 to 0.804) | 0.194 | 72.6% (70.1 to 75.1) | 50.6% |
| results structure | 54.7% | +4.1 (1.8 to 6.8) | 0.740 (0.711 to 0.766) | 0.209 | 68.2% (65.5 to 70.7) | 50.6% |

_How much better one model ranks trials than another, using the same resamples._

| AUROC difference | Estimate (95% interval) |
|---|---|
| registration minus phase lookup | +0.101 (0.073 to 0.126) |
| leaky minus registration | +0.048 (0.031 to 0.066) |
| registration minus registration, no modality | -0.000 (-0.004 to 0.003) |

_The registration model within subsets of these trials. modality is an input to the model, so within-subset AUROC is lower partly because the model's strongest input is held fixed. Not model inputs: control kind (it comes from the results section); enrollment (actual, post-hoc) (it is the actual figure, recorded after the trial)._

| Grouping | Subset | n | Observed success | AUROC (95% interval) | Calibration gap (points, 95% interval) |
|---|---|---|---|---|---|
| control kind | placebo | 1053 | 50.8% | 0.745 (0.714 to 0.774) | +6.0 (3.3 to 8.7) |
| control kind | active | 150 | 50.7% | 0.680 (0.595 to 0.760) | -3.6 (-10.8 to 4.2) |
| enrollment (actual, post-hoc) | <100 | 346 | 32.1% | 0.654 (0.594 to 0.713) | +11.1 (6.3 to 16.1) |
| enrollment (actual, post-hoc) | 100-299 | 405 | 47.7% | 0.708 (0.656 to 0.754) | +6.7 (2.2 to 11.5) |
| enrollment (actual, post-hoc) | 300-999 | 367 | 69.2% | 0.677 (0.618 to 0.740) | -4.6 (-8.9 to -0.3) |
| enrollment (actual, post-hoc) | 1000+ | 118 | 57.6% | 0.762 (0.665 to 0.846) | +7.0 (-0.8 to 15.5) |
| modality | small molecule | 891 | 49.0% | 0.740 (0.709 to 0.771) | +4.8 (1.8 to 7.7) |
| modality | biologic | 282 | 53.5% | 0.707 (0.642 to 0.761) | +6.1 (0.7 to 11.9) |
| modality | advanced | 24 | 66.7% | not computed (n < 30) | not computed (n < 30) |
| modality | mixed | 18 | 50.0% | not computed (n < 30) | not computed (n < 30) |
| modality | vaccine | 15 | 73.3% | not computed (n < 30) | not computed (n < 30) |

Trials in no subset row (values not shown): control kind: no treatment (13), unspecified (20); modality: missing (6). Control kinds other than placebo and active are not shown: 'no treatment' and 'unspecified' are too few to compare and are not the placebo-against-active contrast.

Bootstrap resamples skipped for holding one class only, headline tables: at most 0 of 1000 in any row. Bootstrap resamples skipped for holding one class only, subset table: at most 0 of 1000 in any row.

## Late test set (finished 2024 to 2025)

492 trials; observed success rate 47.8%.

_Every model on the same trials: how well it ranks trials (AUROC), how far off its average is, and how it compares with always saying the common answer._

| Model | Mean predicted | Calibration gap (points) | AUROC | Brier | Accuracy at 0.5 | Always-majority accuracy |
|---|---|---|---|---|---|---|
| constant | 58.4% | +10.7 (6.2 to 15.1) | 0.500 (0.500 to 0.500) | 0.261 | 47.8% (43.3 to 52.2) | 47.8% |
| phase lookup | 57.1% | +9.4 (4.9 to 13.6) | 0.646 (0.608 to 0.688) | 0.238 | 64.4% (60.6 to 68.7) | 47.8% |
| registration | 55.4% | +7.6 (3.7 to 11.5) | 0.733 (0.691 to 0.777) | 0.219 | 66.7% (62.8 to 70.7) | 47.8% |
| registration, no modality | 55.2% | +7.4 (3.4 to 11.3) | 0.738 (0.694 to 0.779) | 0.218 | 66.1% (62.0 to 69.9) | 47.8% |
| leaky | 50.4% | +2.7 (-1.3 to 6.3) | 0.778 (0.737 to 0.819) | 0.194 | 71.1% (67.1 to 75.2) | 47.8% |
| results structure | 54.8% | +7.0 (3.0 to 11.0) | 0.742 (0.699 to 0.784) | 0.214 | 66.7% (62.6 to 70.7) | 47.8% |

_How much better one model ranks trials than another, using the same resamples._

| AUROC difference | Estimate (95% interval) |
|---|---|
| registration minus phase lookup | +0.087 (0.050 to 0.127) |
| leaky minus registration | +0.045 (0.018 to 0.076) |
| registration minus registration, no modality | -0.004 (-0.011 to 0.001) |

_The registration model within subsets of these trials. modality is an input to the model, so within-subset AUROC is lower partly because the model's strongest input is held fixed. Not model inputs: control kind (it comes from the results section); enrollment (actual, post-hoc) (it is the actual figure, recorded after the trial)._

| Grouping | Subset | n | Observed success | AUROC (95% interval) | Calibration gap (points, 95% interval) |
|---|---|---|---|---|---|
| control kind | placebo | 429 | 49.2% | 0.737 (0.687 to 0.782) | +7.7 (3.1 to 11.9) |
| control kind | active | 51 | 37.3% | 0.697 (0.527 to 0.860) | +9.3 (-3.8 to 21.0) |
| enrollment (actual, post-hoc) | <100 | 126 | 38.9% | 0.716 (0.611 to 0.806) | +6.9 (-1.6 to 14.5) |
| enrollment (actual, post-hoc) | 100-299 | 184 | 45.7% | 0.662 (0.577 to 0.742) | +8.9 (2.0 to 15.8) |
| enrollment (actual, post-hoc) | 300-999 | 142 | 52.1% | 0.806 (0.732 to 0.871) | +10.6 (3.3 to 17.9) |
| enrollment (actual, post-hoc) | 1000+ | 40 | 70.0% | 0.680 (0.487 to 0.849) | -6.5 (-19.5 to 7.0) |
| modality | small molecule | 344 | 48.5% | 0.756 (0.705 to 0.809) | +5.1 (0.1 to 10.0) |
| modality | biologic | 113 | 42.5% | 0.652 (0.546 to 0.748) | +18.0 (8.9 to 26.5) |
| modality | advanced | 16 | 62.5% | not computed (n < 30) | not computed (n < 30) |
| modality | mixed | 13 | 61.5% | not computed (n < 30) | not computed (n < 30) |
| modality | vaccine | 4 | 25.0% | not computed (n < 30) | not computed (n < 30) |

Trials in no subset row (values not shown): control kind: no treatment (5), unspecified (7); modality: missing (2). Control kinds other than placebo and active are not shown: 'no treatment' and 'unspecified' are too few to compare and are not the placebo-against-active contrast.

Bootstrap resamples skipped for holding one class only, headline tables: at most 0 of 1000 in any row. Bootstrap resamples skipped for holding one class only, subset table: at most 0 of 1000 in any row.

## Extended test set (finished 2021 to 2025)

1,728 trials; observed success rate 49.8%.

_Every model on the same trials: how well it ranks trials (AUROC), how far off its average is, and how it compares with always saying the common answer._

| Model | Mean predicted | Calibration gap (points) | AUROC | Brier | Accuracy at 0.5 | Always-majority accuracy |
|---|---|---|---|---|---|---|
| constant | 58.4% | +8.6 (6.4 to 10.9) | 0.500 (0.500 to 0.500) | 0.257 | 49.8% (47.5 to 52.0) | 49.8% |
| phase lookup | 57.3% | +7.5 (5.4 to 9.8) | 0.634 (0.610 to 0.660) | 0.238 | 63.4% (61.0 to 65.9) | 49.8% |
| registration | 55.3% | +5.5 (3.4 to 7.7) | 0.731 (0.708 to 0.754) | 0.215 | 67.8% (65.7 to 70.0) | 49.8% |
| registration, no modality | 55.1% | +5.3 (3.2 to 7.4) | 0.733 (0.709 to 0.755) | 0.214 | 67.1% (65.0 to 69.3) | 49.8% |
| leaky | 51.9% | +2.1 (0.1 to 4.2) | 0.779 (0.757 to 0.800) | 0.194 | 72.2% (70.1 to 74.2) | 49.8% |
| results structure | 54.7% | +4.9 (2.8 to 7.0) | 0.740 (0.716 to 0.763) | 0.211 | 67.8% (65.8 to 70.0) | 49.8% |

_How much better one model ranks trials than another, using the same resamples._

| AUROC difference | Estimate (95% interval) |
|---|---|
| registration minus phase lookup | +0.097 (0.074 to 0.120) |
| leaky minus registration | +0.048 (0.034 to 0.062) |
| registration minus registration, no modality | -0.002 (-0.004 to 0.001) |

_The registration model within subsets of these trials. modality is an input to the model, so within-subset AUROC is lower partly because the model's strongest input is held fixed. Not model inputs: control kind (it comes from the results section); enrollment (actual, post-hoc) (it is the actual figure, recorded after the trial)._

| Grouping | Subset | n | Observed success | AUROC (95% interval) | Calibration gap (points, 95% interval) |
|---|---|---|---|---|---|
| control kind | placebo | 1482 | 50.3% | 0.743 (0.717 to 0.765) | +6.5 (4.1 to 9.0) |
| control kind | active | 201 | 47.3% | 0.682 (0.608 to 0.752) | -0.3 (-6.6 to 6.1) |
| enrollment (actual, post-hoc) | <100 | 472 | 33.9% | 0.673 (0.622 to 0.722) | +10.0 (5.7 to 14.4) |
| enrollment (actual, post-hoc) | 100-299 | 589 | 47.0% | 0.694 (0.647 to 0.735) | +7.4 (3.7 to 11.1) |
| enrollment (actual, post-hoc) | 300-999 | 509 | 64.4% | 0.719 (0.672 to 0.765) | -0.4 (-4.0 to 3.7) |
| enrollment (actual, post-hoc) | 1000+ | 158 | 60.8% | 0.738 (0.655 to 0.818) | +3.6 (-3.3 to 10.7) |
| modality | small molecule | 1235 | 48.9% | 0.745 (0.717 to 0.771) | +4.9 (2.2 to 7.3) |
| modality | biologic | 395 | 50.4% | 0.689 (0.637 to 0.746) | +9.5 (4.7 to 14.2) |
| modality | advanced | 40 | 65.0% | 0.707 (0.518 to 0.871) | -2.7 (-16.7 to 11.5) |
| modality | mixed | 31 | 54.8% | 0.733 (0.519 to 0.900) | -1.7 (-18.8 to 14.8) |
| modality | vaccine | 19 | 63.2% | not computed (n < 30) | not computed (n < 30) |

Trials in no subset row (values not shown): control kind: no treatment (18), unspecified (27); modality: missing (8). Control kinds other than placebo and active are not shown: 'no treatment' and 'unspecified' are too few to compare and are not the placebo-against-active contrast.

Bootstrap resamples skipped for holding one class only, headline tables: at most 0 of 1000 in any row. Bootstrap resamples skipped for holding one class only, subset table: at most 0 of 1000 in any row.
