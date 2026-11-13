# The second scoring of the test set (TEST_PLAN_2.md)

Plan: `TEST_PLAN_2.md` at commit `441e3142f3c0eb041246d2a47d3ee50452f307ce`, unchanged since. Code: commit `441e3142f3c0eb041246d2a47d3ee50452f307ce`, clean tree, checked at the start and again just before the opening.

Scored: 1,236 test trials (2021 to 2023), observed success 50.6 per 100. Trained on 5,849 development trials.

Models: 40 (the fixed list in `final2.MODELS`). Fitting and predicting took 82 seconds, before any label was read; the dress rehearsal of the whole shared path on made-up labels 5 seconds.

Access note: TEST_PLAN_2.md: the second scoring of the test set (items 1 to 8), one opening, which also supplies the late-set labels (2024 to 2025) that item 7 needs. Second and last scoring of the test set: the eight items of TEST_PLAN_2.md, at commit 441e314

The test labels have been opened 4 time(s), counting this run. Every opening of the held-out folder is listed (also in evals/heldout_log.jsonl).

| Time (UTC) | Commit | File | Note |
|---|---|---|---|
| 2026-11-11T06:48:33+00:00 | `2bc89fe3e89c` | data/heldout/test_labels.parquet | Pre-registered baseline ladder, first and only run before any new modelling. Pre-registration emailed Wed 11 Nov 2026 06:21 GMT and locked at tag prereg. |
| 2026-11-11T19:36:42+00:00 | `69dd31b822da` | data/heldout/test_labels.parquet | final run per FINAL_PLAN.md |
| 2026-11-11T19:36:59+00:00 | `3399dff3049f` | data/heldout/test_labels.parquet | scorecard for the seven pre-registered predictions, same session as the final run |
| 2026-11-12T17:46:39+00:00 | `441e3142f3c0` | data/heldout/test_labels.parquet | TEST_PLAN_2.md: the second scoring of the test set (items 1 to 8), one opening, which also supplies the late-set labels (2024 to 2025) that item 7 needs. Second and last scoring of the test set: the eight items of TEST_PLAN_2.md, at commit 441e314 |

How to read this. Differences are first-named minus second-named, AUROC to three places, taken within each resample (paired). Calibration gap is mean predicted minus observed success, per 100 trials (positive: too optimistic). Intervals are 95% percentile intervals from 1,000 resamples of lead drugs (all of a drug's trials together; a trial with no known lead drug is its own); trials sharing a sponsor are still treated as independent, so they are somewhat too narrow. A result is met when its interval is above zero.

The SHA-256 of every input file is in the JSON (`input_sha256`).

## Item 1: Hierarchical model (partially pooled effects for target, condition, sponsor, drug, area) against the registration baseline

**Expected (TEST_PLAN_2.md, verbatim):** about +0.03 AUROC

**Result:** hierarchical: full hierarchy minus logistic: registration, AUROC: +0.038 (+0.018 to +0.057).

**Verdict:** by the plan's rule (interval wholly on the expected side of zero), **met**; against the stated range, **no range stated in the plan**.

| Model | AUROC (95% interval) | AUROC minus logistic: registration | Calibration gap per 100 trials | Brier |
|---|---|---|---|---|
| logistic: registration | 0.730 (0.698 to 0.761) | (baseline) | +4.6 (+1.9 to +7.3) | 0.213 |
| hierarchical: registration only | 0.733 (0.700 to 0.764) | +0.002 (-0.000 to +0.005) | +4.7 (+2.0 to +7.4) | 0.212 |
| hierarchical: + target | 0.752 (0.720 to 0.782) | +0.021 (+0.007 to +0.036) | +4.8 (+2.1 to +7.5) | 0.206 |
| hierarchical: + condition | 0.746 (0.716 to 0.776) | +0.016 (+0.002 to +0.031) | +3.7 (+1.2 to +6.5) | 0.207 |
| hierarchical: + sponsor | 0.749 (0.715 to 0.781) | +0.018 (+0.007 to +0.032) | +4.8 (+2.2 to +7.5) | 0.208 |
| hierarchical: full hierarchy | 0.768 (0.738 to 0.796) | +0.038 (+0.018 to +0.057) | +4.0 (+1.4 to +6.6) | 0.199 |
| hierarchical: full + precedent + genetics | 0.774 (0.742 to 0.802) | +0.043 (+0.023 to +0.064) | +4.5 (+2.0 to +7.1) | 0.197 |
| hierarchical: registration only [2020 links] | 0.733 (0.700 to 0.764) | +0.002 (-0.000 to +0.005) | +4.7 (+2.0 to +7.4) | 0.212 |
| hierarchical: + target [2020 links] | 0.747 (0.715 to 0.778) | +0.017 (+0.005 to +0.028) | +4.0 (+1.2 to +6.7) | 0.207 |
| hierarchical: + condition [2020 links] | 0.746 (0.716 to 0.776) | +0.016 (+0.002 to +0.031) | +3.7 (+1.2 to +6.5) | 0.207 |
| hierarchical: + sponsor [2020 links] | 0.749 (0.715 to 0.781) | +0.018 (+0.007 to +0.032) | +4.8 (+2.2 to +7.5) | 0.208 |
| hierarchical: full hierarchy [2020 links] | 0.765 (0.734 to 0.794) | +0.034 (+0.016 to +0.053) | +3.6 (+1.1 to +6.3) | 0.200 |
| hierarchical: full + precedent + genetics [2020 links] | 0.766 (0.735 to 0.795) | +0.036 (+0.017 to +0.055) | +3.6 (+1.0 to +6.1) | 0.199 |

## Item 2: Registration plus the precedent flag, for three link versions (today's; targets dated to June 2020; drug names and targets dated). Headline: the strict flag (experimental arms only) on the fully dated version

**Expected (TEST_PLAN_2.md, verbatim):** strict version +0.005 to +0.01

**Result:** logistic: registration + precedent (2020 names and links, experimental arms only) minus logistic: registration, AUROC: -0.000 (-0.008 to +0.009).

**Verdict:** by the plan's rule (interval wholly on the expected side of zero), **not met**; against the stated range, **outside, but the interval overlaps the range**.

| Model | AUROC (95% interval) | AUROC minus logistic: registration | Calibration gap per 100 trials | Brier |
|---|---|---|---|---|
| logistic: registration | 0.730 (0.698 to 0.761) | (baseline) | +4.6 (+1.9 to +7.3) | 0.213 |
| logistic: registration + precedent (today's links) | 0.741 (0.709 to 0.771) | +0.011 (+0.000 to +0.023) | +4.6 (+1.9 to +7.2) | 0.209 |
| logistic: registration + precedent (2020 links) | 0.729 (0.696 to 0.760) | -0.001 (-0.010 to +0.009) | +3.8 (+1.1 to +6.6) | 0.212 |
| logistic: registration + precedent (2020 names and links) | 0.731 (0.698 to 0.762) | +0.001 (-0.007 to +0.010) | +4.1 (+1.5 to +6.8) | 0.212 |
| logistic: registration + precedent (today's links, experimental arms only) | 0.741 (0.709 to 0.772) | +0.011 (+0.002 to +0.021) | +4.8 (+2.1 to +7.4) | 0.209 |
| logistic: registration + precedent (2020 links, experimental arms only) | 0.730 (0.696 to 0.761) | -0.001 (-0.009 to +0.008) | +4.1 (+1.3 to +6.8) | 0.212 |
| logistic: registration + precedent (2020 names and links, experimental arms only) | 0.730 (0.697 to 0.761) | -0.000 (-0.008 to +0.009) | +4.0 (+1.3 to +6.8) | 0.212 |

## Item 3: HOW (registration without therapeutic area), WHAT (disease, target, precedent, genetics, biology), and both

**Expected (TEST_PLAN_2.md, verbatim):** both beats HOW by +0.015 to +0.03; WHAT alone about 0.66

**Result:** logistic: HOW + WHAT (2020 names and links) minus logistic: HOW (2020 names and links), AUROC: +0.039 (+0.021 to +0.058).

**Verdict:** by the plan's rule (interval wholly on the expected side of zero), **met**; against the stated range, **outside, but the interval overlaps the range**.

**WHAT alone (2020 names and links):** AUROC 0.689 (0.652 to 0.723); the plan states no range ("about 0.66"), so there is no verdict.

### Today's links

| Model | AUROC (95% interval) | AUROC minus logistic: registration | Calibration gap per 100 trials | Brier |
|---|---|---|---|---|
| logistic: registration | 0.730 (0.698 to 0.761) | (baseline) | +4.6 (+1.9 to +7.3) | 0.213 |
| logistic: HOW (today's links) | 0.717 (0.685 to 0.747) | -0.013 (-0.025 to -0.000) | +5.4 (+2.5 to +8.2) | 0.218 |
| logistic: WHAT (today's links) | 0.707 (0.670 to 0.739) | -0.024 (-0.058 to +0.010) | +4.7 (+2.0 to +7.5) | 0.221 |
| logistic: WHAT + phase (today's links) | 0.737 (0.704 to 0.769) | +0.007 (-0.019 to +0.034) | +4.9 (+2.3 to +7.7) | 0.211 |
| logistic: HOW + WHAT (today's links) | 0.763 (0.731 to 0.792) | +0.033 (+0.017 to +0.049) | +3.7 (+1.2 to +6.4) | 0.201 |
| logistic: WHAT (today's links, own lead drug excluded) | 0.703 (0.665 to 0.736) | -0.028 (-0.061 to +0.007) | +4.9 (+2.2 to +7.7) | 0.222 |
| logistic: WHAT + phase (today's links, own lead drug excluded) | 0.734 (0.701 to 0.766) | +0.004 (-0.022 to +0.031) | +5.1 (+2.5 to +7.8) | 0.212 |
| logistic: HOW + WHAT (today's links, own lead drug excluded) | 0.762 (0.730 to 0.791) | +0.031 (+0.015 to +0.047) | +3.8 (+1.2 to +6.5) | 0.201 |

Paired differences from HOW (same resamples):

| Model | AUROC minus logistic: HOW (today's links) |
|---|---|
| logistic: WHAT (today's links) | -0.011 (-0.049 to +0.028) |
| logistic: WHAT + phase (today's links) | +0.020 (-0.010 to +0.048) |
| logistic: HOW + WHAT (today's links) | +0.046 (+0.027 to +0.065) |
| logistic: WHAT (today's links, own lead drug excluded) | -0.015 (-0.053 to +0.024) |
| logistic: WHAT + phase (today's links, own lead drug excluded) | +0.017 (-0.013 to +0.045) |
| logistic: HOW + WHAT (today's links, own lead drug excluded) | +0.044 (+0.026 to +0.063) |

### 2020 links

| Model | AUROC (95% interval) | AUROC minus logistic: registration | Calibration gap per 100 trials | Brier |
|---|---|---|---|---|
| logistic: registration | 0.730 (0.698 to 0.761) | (baseline) | +4.6 (+1.9 to +7.3) | 0.213 |
| logistic: HOW (2020 links) | 0.717 (0.685 to 0.747) | -0.013 (-0.025 to -0.000) | +5.4 (+2.5 to +8.2) | 0.218 |
| logistic: WHAT (2020 links) | 0.679 (0.640 to 0.712) | -0.051 (-0.084 to -0.017) | +3.5 (+0.7 to +6.4) | 0.228 |
| logistic: WHAT + phase (2020 links) | 0.722 (0.688 to 0.755) | -0.009 (-0.034 to +0.016) | +4.2 (+1.4 to +7.0) | 0.215 |
| logistic: HOW + WHAT (2020 links) | 0.752 (0.720 to 0.781) | +0.022 (+0.009 to +0.036) | +2.9 (+0.3 to +5.5) | 0.204 |
| logistic: WHAT (2020 links, own lead drug excluded) | 0.672 (0.633 to 0.706) | -0.058 (-0.092 to -0.022) | +3.7 (+0.9 to +6.6) | 0.229 |
| logistic: WHAT + phase (2020 links, own lead drug excluded) | 0.718 (0.683 to 0.751) | -0.013 (-0.038 to +0.014) | +4.4 (+1.6 to +7.2) | 0.217 |
| logistic: HOW + WHAT (2020 links, own lead drug excluded) | 0.749 (0.718 to 0.778) | +0.019 (+0.006 to +0.033) | +3.0 (+0.4 to +5.6) | 0.205 |

Paired differences from HOW (same resamples):

| Model | AUROC minus logistic: HOW (2020 links) |
|---|---|
| logistic: WHAT (2020 links) | -0.039 (-0.078 to -0.001) |
| logistic: WHAT + phase (2020 links) | +0.004 (-0.026 to +0.035) |
| logistic: HOW + WHAT (2020 links) | +0.035 (+0.017 to +0.053) |
| logistic: WHAT (2020 links, own lead drug excluded) | -0.045 (-0.084 to -0.006) |
| logistic: WHAT + phase (2020 links, own lead drug excluded) | +0.000 (-0.031 to +0.031) |
| logistic: HOW + WHAT (2020 links, own lead drug excluded) | +0.032 (+0.014 to +0.051) |

### 2020 names and links

| Model | AUROC (95% interval) | AUROC minus logistic: registration | Calibration gap per 100 trials | Brier |
|---|---|---|---|---|
| logistic: registration | 0.730 (0.698 to 0.761) | (baseline) | +4.6 (+1.9 to +7.3) | 0.213 |
| logistic: HOW (2020 names and links) | 0.717 (0.685 to 0.747) | -0.013 (-0.025 to -0.000) | +5.4 (+2.5 to +8.2) | 0.218 |
| logistic: WHAT (2020 names and links) | 0.689 (0.652 to 0.723) | -0.041 (-0.075 to -0.006) | +3.0 (+0.2 to +5.9) | 0.225 |
| logistic: WHAT + phase (2020 names and links) | 0.729 (0.697 to 0.761) | -0.001 (-0.025 to +0.025) | +3.7 (+1.1 to +6.5) | 0.213 |
| logistic: HOW + WHAT (2020 names and links) | 0.757 (0.725 to 0.785) | +0.026 (+0.013 to +0.041) | +2.6 (-0.1 to +5.2) | 0.203 |
| logistic: WHAT (2020 names and links, own lead drug excluded) | 0.683 (0.645 to 0.717) | -0.047 (-0.081 to -0.011) | +3.1 (+0.3 to +6.0) | 0.226 |
| logistic: WHAT + phase (2020 names and links, own lead drug excluded) | 0.726 (0.693 to 0.759) | -0.004 (-0.029 to +0.023) | +3.9 (+1.2 to +6.7) | 0.214 |
| logistic: HOW + WHAT (2020 names and links, own lead drug excluded) | 0.755 (0.724 to 0.783) | +0.024 (+0.010 to +0.039) | +2.6 (+0.0 to +5.3) | 0.203 |

Paired differences from HOW (same resamples):

| Model | AUROC minus logistic: HOW (2020 names and links) |
|---|---|
| logistic: WHAT (2020 names and links) | -0.028 (-0.068 to +0.012) |
| logistic: WHAT + phase (2020 names and links) | +0.012 (-0.018 to +0.040) |
| logistic: HOW + WHAT (2020 names and links) | +0.039 (+0.021 to +0.058) |
| logistic: WHAT (2020 names and links, own lead drug excluded) | -0.034 (-0.074 to +0.007) |
| logistic: WHAT + phase (2020 names and links, own lead drug excluded) | +0.009 (-0.022 to +0.037) |
| logistic: HOW + WHAT (2020 names and links, own lead drug excluded) | +0.037 (+0.019 to +0.056) |

## Item 4: Phase 2 success by third of the WHAT score

**Expected (TEST_PLAN_2.md, verbatim):** top third beats bottom third by 15 to 25 per 100 trials

**Result:** Phase 2: top third minus bottom third of logistic: WHAT (2020 names and links), per 100: +27.8 (+18.0 to +39.0) per 100 trials.

**Verdict:** by the plan's rule (interval wholly on the expected side of zero), **met**; against the stated range, **outside, but the interval overlaps the range**.

### logistic: WHAT (today's links)

|  | Bottom third | Middle third | Top third | Top minus bottom |
|---|---|---|---|---|
| Phase 2 | 23.7 (16.9 to 30.0), n=186 | 32.8 (25.8 to 40.1), n=186 | 52.4 (44.2 to 60.8), n=185 | +28.8 (+17.7 to +39.4) |
| Phase 3 | 40.8 (34.5 to 47.2), n=223 | 64.5 (57.8 to 70.7), n=231 | 81.8 (75.8 to 86.9), n=225 | +41.0 (+32.1 to +48.9) |

Survival (`whathow_report.survival`): top third minus bottom third +41.7 per 100; among trials alike in phase, sponsor class and HOW third (12 of 16 strata with both tails) +32.9 per 100 (+25.1 to +40.1), so 79% survives (68% to 87%, lead-drug resamples).

### logistic: WHAT (today's links, own lead drug excluded)

|  | Bottom third | Middle third | Top third | Top minus bottom |
|---|---|---|---|---|
| Phase 2 | 24.2 (17.6 to 30.6), n=186 | 31.6 (24.6 to 38.8), n=187 | 53.3 (45.0 to 61.7), n=184 | +29.1 (+18.1 to +39.7) |
| Phase 3 | 42.3 (35.7 to 49.1), n=227 | 63.7 (56.6 to 70.4), n=226 | 81.4 (75.6 to 86.4), n=226 | +39.1 (+30.2 to +47.2) |

Survival (`whathow_report.survival`): top third minus bottom third +40.3 per 100; among trials alike in phase, sponsor class and HOW third (12 of 16 strata with both tails) +32.2 per 100 (+24.3 to +39.1), so 80% survives (70% to 88%, lead-drug resamples).

### logistic: WHAT (2020 links)

|  | Bottom third | Middle third | Top third | Top minus bottom |
|---|---|---|---|---|
| Phase 2 | 24.6 (17.7 to 31.1), n=187 | 33.0 (25.8 to 40.2), n=185 | 51.4 (42.9 to 59.7), n=185 | +26.8 (+17.0 to +37.8) |
| Phase 3 | 44.1 (37.2 to 50.8), n=227 | 64.2 (57.6 to 70.1), n=226 | 79.2 (72.8 to 84.6), n=226 | +35.2 (+25.9 to +43.9) |

Survival (`whathow_report.survival`): top third minus bottom third +35.9 per 100; among trials alike in phase, sponsor class and HOW third (12 of 16 strata with both tails) +27.1 per 100 (+19.7 to +34.2), so 75% survives (64% to 85%, lead-drug resamples).

### logistic: WHAT (2020 links, own lead drug excluded)

|  | Bottom third | Middle third | Top third | Top minus bottom |
|---|---|---|---|---|
| Phase 2 | 25.1 (18.3 to 31.6), n=187 | 32.6 (25.6 to 39.7), n=184 | 51.1 (42.7 to 59.6), n=186 | +25.9 (+15.8 to +37.2) |
| Phase 3 | 45.4 (38.6 to 52.2), n=227 | 62.8 (56.4 to 69.3), n=226 | 79.2 (72.7 to 84.3), n=226 | +33.8 (+24.7 to +42.7) |

Survival (`whathow_report.survival`): top third minus bottom third +35.2 per 100; among trials alike in phase, sponsor class and HOW third (13 of 16 strata with both tails) +25.9 per 100 (+18.7 to +33.1), so 73% survives (62% to 83%, lead-drug resamples).

### logistic: WHAT (2020 names and links)

|  | Bottom third | Middle third | Top third | Top minus bottom |
|---|---|---|---|---|
| Phase 2 | 25.7 (18.7 to 32.2), n=187 | 29.7 (22.7 to 36.6), n=185 | 53.5 (45.6 to 61.5), n=185 | +27.8 (+18.0 to +39.0) |
| Phase 3 | 42.0 (35.6 to 48.8), n=226 | 66.5 (60.1 to 72.2), n=227 | 78.8 (72.0 to 84.2), n=226 | +36.7 (+26.8 to +45.4) |

Survival (`whathow_report.survival`): top third minus bottom third +37.6 per 100; among trials alike in phase, sponsor class and HOW third (13 of 16 strata with both tails) +29.3 per 100 (+22.2 to +36.5), so 78% survives (68% to 87%, lead-drug resamples).

### logistic: WHAT (2020 names and links, own lead drug excluded)

|  | Bottom third | Middle third | Top third | Top minus bottom |
|---|---|---|---|---|
| Phase 2 | 25.8 (18.9 to 32.5), n=186 | 29.6 (22.8 to 36.3), n=186 | 53.5 (45.3 to 61.9), n=185 | +27.7 (+17.6 to +38.5) |
| Phase 3 | 42.3 (35.4 to 49.6), n=220 | 65.8 (59.5 to 71.6), n=234 | 78.7 (72.2 to 84.1), n=225 | +36.4 (+26.8 to +44.9) |

Survival (`whathow_report.survival`): top third minus bottom third +38.1 per 100; among trials alike in phase, sponsor class and HOW third (13 of 16 strata with both tails) +29.9 per 100 (+22.5 to +37.3), so 79% survives (68% to 87%, lead-drug resamples).

## Item 5: Split study with the test set as the "later" holdout, against random holdouts of the same size

**Expected (TEST_PLAN_2.md, verbatim):** a random split flatters the baseline by about 0.03 AUROC and hides the over-prediction

**Result:** logistic: registration: random-by-trial holdout minus the later trials, AUROC: -0.001 (-0.053 to +0.049).

**Verdict:** by the plan's rule (interval wholly on the expected side of zero), **not met**; against the stated range, **no range stated in the plan**.

**Over-prediction:** the baseline's calibration gap is +4.6 per 100 on the later trials and +0.2 on a random-by-trial holdout (-0.3 by drug). Hidden: **met (by direction only: no interval)**.

Pool 5,849 trials; later holdout 1,236; 20 random seeds each.

| Model | Later trials | Random by trial | Random by drug | By trial minus later | By drug minus later | Gap per 100: later / by trial / by drug |
|---|---|---|---|---|---|---|
| lookup: phase | 0.630 | 0.615 | 0.617 | -0.015 (-0.059 to +0.030) | -0.013 (-0.060 to +0.035) | +6.8 / -0.0 / -0.3 |
| logistic: registration | 0.730 | 0.729 | 0.735 | -0.001 (-0.053 to +0.049) | +0.005 (-0.053 to +0.060) | +4.6 / +0.2 / -0.3 |
| memory: lead drug | 0.633 | 0.672 | 0.500 | +0.039 (-0.020 to +0.100) | -0.133 (-0.172 to -0.091) | +9.5 / +2.2 / -0.4 |
| ladder: registration + endpoint + history + genetics + words | 0.748 | 0.775 | 0.764 | +0.027 (-0.021 to +0.074) | +0.016 (-0.036 to +0.065) | +3.0 / +0.3 / -0.7 |
| text: drug names only | 0.625 | 0.673 | 0.604 | +0.048 (-0.012 to +0.107) | -0.021 (-0.101 to +0.043) | +4.6 / -0.3 / -3.4 |

## Item 6: Per-trial predictions saved for the baseline and the models above

**Expected (TEST_PLAN_2.md, verbatim):** none; needed to draw development and test curves side by side

Saved `outputs/final2/predictions.parquet`: 1,236 trials, 40 models, with outcomes. Git ignores it: True. It must never be committed.

## Item 7: Intervals for the earlier results regenerated with resampling by lead drug

**Expected (TEST_PLAN_2.md, verbatim):** wider; no verdict changes

**Result:** the earlier reports regenerated by `final` and `scorecard` with lead-drug resampling, against the same code with single-trial resampling (which reproduces the earlier intervals). Wider: 143 intervals; narrower: 26; lines with a changed verdict: 2.

**Verdict:** wider, **met**; no verdict changes, **not met**.

Full reports: `results/test_final2_item7_final.md`, `results/test_final2_item7_scorecard.md`.

Only rows of the first run are compared: the model families added since (precedent, hierarchical, whathow) are left out of the regenerated reports. The guard's one read also supplied the late-set labels (2024 to 2025) that regenerating the first run's report needs.

- final: 135 wider, 24 narrower, 157 unchanged; median width ratio 1.08.

- scorecard: 8 wider, 2 narrower, 16 unchanged; median width ratio 1.07.

**Which verdicts changed (2).** The plan expected none; these are the rows where it did not hold.

| Report | Row | Single-trial interval (earlier) | Lead-drug interval | Change |
|---|---|---|---|---|
| final | neural network: registration | -0.000 to +0.018 | +0.001 to +0.018 | included zero, now excludes it |
| scorecard | 3 | therapeutic_area | 0.001 to 0.024 | 0.000 to 0.025 | excluded zero, now includes it |

## Item 8: A biology-only score, within phase: (a) a count of six risk flags fixed in advance (target expressed in nearly every tissue; high expression in vital organs; gene essential to cells; intracellular target; drug activates its target; low drug-likeness), trials with two or more flags against trials with none; (b) a logistic model on biology columns only, top third against bottom third. No phase, sponsor, disease area, design, precedent, genetics or earlier outcome goes in. Thresholds are the ones fixed on trials before 2015; the logistic model is refitted on all development trials by the declared recipe

**Expected (TEST_PLAN_2.md, verbatim):** (a) 2 to 10 fewer successes per 100 with two or more flags; (b) 5 to 15 more in the top third. On development trials from 2015 to 2020 these were 8.1 fewer (14.6 to 1.7) and 16.6 more (10 to 23), and I expect them to shrink again. On trials whose targets appear in no development trial: same direction, too few to say more

Within phase: each difference is computed inside Phase 2 and inside Phase 3 and pooled, so phase cannot explain it; the phases are also shown apart below. Under the 2020 names and links version there are no chemistry columns, so two of the six flags (drug activates its target; low drug-likeness) cannot fire.

**(a) two or more flags minus none, today's links, pooled over phase:** -10.1 (-20.8 to +1.2) per 100 (n = 238 against 290). **Verdict:** by the plan's rule (interval wholly on the expected side of zero), **not met**; against the stated range, **outside, but the interval overlaps the range**.

**(b) top third minus bottom third, today's links, pooled over phase:** +23.9 (+13.3 to +33.7) per 100 (n = 230 against 248). **Verdict:** by the plan's rule (interval wholly on the expected side of zero), **met**; against the stated range, **outside, but the interval overlaps the range**.

Every link version (today's is the headline; the other two are sensitivity rows):

| Links | Contrast | Estimate (95% interval) | Interval | Range |
|---|---|---|---|---|
| today's links | (a) two or more flags minus none | -10.1 (-20.8 to +1.2) per 100 (n = 238 against 290) | not met | outside, but the interval overlaps the range |
| today's links | (b) top third minus bottom third | +23.9 (+13.3 to +33.7) per 100 (n = 230 against 248) | met | outside, but the interval overlaps the range |
| 2020 links | (a) two or more flags minus none | -2.9 (-16.0 to +10.1) per 100 (n = 169 against 209) | not met | inside the predicted range |
| 2020 links | (b) top third minus bottom third | +21.3 (+8.7 to +33.0) per 100 (n = 169 against 196) | met | outside, but the interval overlaps the range |
| 2020 names and links | (a) two or more flags minus none | -1.6 (-15.2 to +11.5) per 100 (n = 142 against 267) | not met | outside, but the interval overlaps the range |
| 2020 names and links | (b) top third minus bottom third | +23.5 (+10.1 to +35.9) per 100 (n = 149 against 212) | met | outside, but the interval overlaps the range |

Trials whose targets appear in no development trial (the plan claims only the direction):

| Links | Contrast | Trials | Estimate (95% interval) | Direction |
|---|---|---|---|---|
| today's links | (a) two or more flags minus none | 30 | not estimated: fewer than 15 trials on a side (n = 14 against 9) | too few trials to say |
| today's links | (b) top third minus bottom third | 30 | not estimated: fewer than 15 trials on a side (n = 5 against 8) | too few trials to say |
| 2020 links | (a) two or more flags minus none | 5 | not estimated: fewer than 15 trials on a side (n = 3 against 1) | too few trials to say |
| 2020 links | (b) top third minus bottom third | 5 | not estimated: fewer than 15 trials on a side (n = 0 against 4) | too few trials to say |
| 2020 names and links | (a) two or more flags minus none | 6 | not estimated: fewer than 15 trials on a side (n = 4 against 1) | too few trials to say |
| 2020 names and links | (b) top third minus bottom third | 6 | not estimated: fewer than 15 trials on a side (n = 0 against 5) | too few trials to say |

### Today's links: success per 100 by flag count, within phase

| Phase | Group | Trials | Success per 100 (95% interval) |
|---|---|---|---|
| phase2 | 0 | 88 | 36.4 (25.7 to 47.1) |
| phase2 | 1 | 84 | 46.4 (34.1 to 59.0) |
| phase2 | 2+ | 86 | 27.9 (18.5 to 37.9) |
| phase3 | 0 | 202 | 68.8 (59.5 to 77.2) |
| phase3 | 1 | 154 | 59.1 (51.0 to 66.7) |
| phase3 | 2+ | 152 | 57.9 (47.2 to 67.3) |

### Today's links: success per 100 by third of the score, within phase

| Phase | Group | Trials | Success per 100 (95% interval) |
|---|---|---|---|
| phase2 | low | 95 | 33.7 (24.2 to 43.6) |
| phase2 | mid | 96 | 29.2 (19.0 to 39.8) |
| phase2 | high | 67 | 52.2 (39.0 to 64.5) |
| phase3 | low | 153 | 47.7 (39.3 to 56.3) |
| phase3 | mid | 192 | 64.6 (56.1 to 72.4) |
| phase3 | high | 163 | 74.2 (64.9 to 82.3) |

766 scored trials with a mapped target; logistic fitted on 4,244 development trials; thirds cut at 0.562 and 0.664.

### 2020 links: success per 100 by flag count, within phase

| Phase | Group | Trials | Success per 100 (95% interval) |
|---|---|---|---|
| phase2 | 0 | 63 | 28.6 (16.1 to 40.3) |
| phase2 | 1 | 46 | 43.5 (26.8 to 60.0) |
| phase2 | 2+ | 57 | 28.1 (17.2 to 39.7) |
| phase3 | 0 | 146 | 63.0 (51.7 to 73.4) |
| phase3 | 1 | 118 | 52.5 (43.4 to 61.2) |
| phase3 | 2+ | 112 | 58.9 (45.7 to 69.3) |

### 2020 links: success per 100 by third of the score, within phase

| Phase | Group | Trials | Success per 100 (95% interval) |
|---|---|---|---|
| phase2 | low | 69 | 30.4 (20.0 to 42.0) |
| phase2 | mid | 53 | 32.1 (18.4 to 47.1) |
| phase2 | high | 44 | 36.4 (20.5 to 52.5) |
| phase3 | low | 127 | 43.3 (33.3 to 52.1) |
| phase3 | mid | 124 | 61.3 (50.5 to 70.5) |
| phase3 | high | 125 | 71.2 (60.7 to 80.6) |

542 scored trials with a mapped target; logistic fitted on 3,811 development trials; thirds cut at 0.570 and 0.673.

### 2020 names and links: success per 100 by flag count, within phase

| Phase | Group | Trials | Success per 100 (95% interval) |
|---|---|---|---|
| phase2 | 0 | 83 | 27.7 (16.9 to 38.9) |
| phase2 | 1 | 42 | 50.0 (33.3 to 65.9) |
| phase2 | 2+ | 46 | 28.3 (16.3 to 40.9) |
| phase3 | 0 | 184 | 60.9 (51.6 to 69.4) |
| phase3 | 1 | 100 | 56.0 (45.9 to 65.5) |
| phase3 | 2+ | 96 | 58.3 (43.4 to 70.9) |

### 2020 names and links: success per 100 by third of the score, within phase

| Phase | Group | Trials | Success per 100 (95% interval) |
|---|---|---|---|
| phase2 | low | 69 | 29.0 (18.3 to 40.0) |
| phase2 | mid | 56 | 28.6 (15.5 to 42.6) |
| phase2 | high | 46 | 45.7 (29.8 to 62.5) |
| phase3 | low | 143 | 46.2 (36.6 to 54.4) |
| phase3 | mid | 134 | 61.9 (52.2 to 70.8) |
| phase3 | high | 103 | 72.8 (60.5 to 83.1) |

551 scored trials with a mapped target; logistic fitted on 4,070 development trials; thirds cut at 0.576 and 0.669.
