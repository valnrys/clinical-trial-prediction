# WHAT versus HOW (development trials)

Can what a trial tries to do predict its outcome beyond how it is run? Two blocks of predictors, compared model against model.

- **HOW**: the registration columns less therapeutic area (phase, sponsor class, allocation, masking, intervention model, arm count, modality, vaccine, healthy volunteers, data monitoring committee).
- **WHAT**: therapeutic area; success rates for the trial's target, its disease and the pair (earlier trials only, see below); how many trials back each rate; the strict precedent flag (experimental-arm drugs only, no comparator fallback); genetic support when the trial started; generic target biology; flags for "has a mapped target" and "has a mapped disease".
- **WHAT + phase** and **HOW + WHAT** add phase, and all of HOW. All WHAT models are logistic regressions with a fixed penalty (C = 0.1).

**HOW here is the registration baseline WITHOUT therapeutic area.** Against the full registration baseline (which has it) the gain of HOW + WHAT is smaller, and most of that gain is the precedent flag: see the rows for the registration baseline plus the strict precedent flag alone, and HOW + WHAT minus that, in each table.

**The rule for group rates.** A trial's rate counts only labelled training trials whose results were first posted before that trial's start, and never the trial itself. The training rows follow the same rule. Each rate is (successes + 5) / (trials + 10): pulled toward 0.5 by a fixed ten trials' worth, with nothing tuned on a label. The outcomes counted are dated; the keys that group trials are not (see below).

**Which links.** The drug-to-target links come from a release published after the trials started, so the WHAT models are not "honest" (the group rates included: their grouping keys come from the link version and today's disease mapping). The 2020 versions use the 20.06 release (the last development year's); the third also matches the registry's drug names to 20.06's names (a short list of generic names such as insulin or saline is never matched) and leaves the chemistry columns empty, because they are keyed by today's drug ids. The disease mapping and the registry record are today's in every version, and so is `lead_drug`, which the lead-drug resampling and the own-drug exclusion use, even in the third version.

**Test-set caution.** The 20.06 links postdate most development trials' starts but predate many test trials' starts, so gains from the dated versions may not carry over to the test set.

Scoring is the repo's own (three rounds; grouped folds by lead drug). Intervals resample whole lead drugs, drawn once for all rounds; differences are paired on the same resamples. The correlation, the share and the thirds use the same draws of drugs, with the thirds and strata fixed from the predictions and cut inside each round or fold. Where a lead drug is unknown the trial is its own cluster; pairs of trials where either lead drug is unknown are kept in the group rates. Success is per 100 trials.

## Today's links (Open Targets 26.09)

### Time-ordered rounds

2,795 scored trials. Mean within-round AUROC, 95% interval resampling whole lead drugs.

| model | AUROC |
|---|---|
| HOW | 0.697 (0.676 to 0.720) |
| WHAT | 0.669 (0.644 to 0.694) |
| WHAT + phase | 0.692 (0.668 to 0.716) |
| HOW + WHAT | 0.722 (0.700 to 0.745) |
| registration baseline (with therapeutic area) | 0.709 (0.686 to 0.731) |
| registration baseline + strict precedent flag alone | 0.719 (0.697 to 0.741) |

Paired differences (same resamples):

| comparison | AUROC difference |
|---|---|
| HOW + WHAT minus HOW | +0.025 (+0.011 to +0.039) |
| HOW + WHAT minus the registration baseline | +0.014 (+0.003 to +0.024) |
| registration baseline + strict precedent flag alone, minus the registration baseline | +0.010 (+0.004 to +0.016) |
| HOW + WHAT minus (registration baseline + strict precedent flag alone) | +0.003 (−0.005 to +0.012) |
| WHAT minus HOW | −0.029 (−0.057 to −0.003) |
| WHAT + phase minus HOW | −0.005 (−0.026 to +0.016) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.332 (0.286 to 0.381) pooled, 0.332 (0.286 to 0.380) on average within a round or fold.
- WHAT's top third succeeds 33.9 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class and HOW third within the same round, the gap is 24.4 per 100 (19.6 to 29.7): **72% survives** (64% to 80%, lead-drug resamples). 37 of 50 strata had trials in both tails.
- WHAT's top third succeeds 33.9 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same round, the gap is 24.1 per 100 (17.0 to 31.4): **71% survives** (52% to 89%, lead-drug resamples). 123 of 359 strata had trials in both tails.

Success per 100 trials by third of WHAT, within phase (lead-drug 95% intervals):

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 28.9 (24.8 to 32.9), n=415 | 40.1 (35.2 to 45.0), n=414 | 56.9 (51.5 to 61.7), n=413 | +28.0 (+21.3 to +33.9) |
| Phase 3 | 49.5 (44.7 to 54.1), n=519 | 61.2 (56.2 to 65.8), n=516 | 81.9 (77.9 to 85.3), n=518 | +32.3 (+26.1 to +38.6) |

With the scored trial's own lead drug left out of every group rate (training rows too). Difference is from HOW:

| model | AUROC | minus HOW |
|---|---|---|
| WHAT, own lead drug excluded | 0.666 (0.642 to 0.691) | −0.031 (−0.060 to −0.006) |
| HOW + WHAT, own lead drug excluded | 0.721 (0.699 to 0.744) | +0.024 (+0.010 to +0.038) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.326 (0.278 to 0.376) pooled, 0.326 (0.279 to 0.374) on average within a round or fold.
- WHAT's top third succeeds 33.3 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class and HOW third within the same round, the gap is 24.0 per 100 (19.0 to 29.6): **72% survives** (63% to 80%, lead-drug resamples). 37 of 50 strata had trials in both tails.
- WHAT's top third succeeds 33.3 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same round, the gap is 23.9 per 100 (16.5 to 31.6): **72% survives** (53% to 91%, lead-drug resamples). 123 of 359 strata had trials in both tails.

Success per 100 by third of WHAT, own lead drug excluded:

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 28.7 (24.4 to 32.7), n=415 | 39.9 (34.9 to 44.7), n=414 | 57.4 (51.8 to 62.3), n=413 | +28.7 (+21.8 to +35.2) |
| Phase 3 | 49.7 (45.0 to 54.2), n=519 | 61.6 (56.7 to 66.4), n=518 | 81.4 (77.4 to 84.9), n=516 | +31.7 (+25.3 to +37.7) |

### Grouped 10-fold by lead drug

5,849 scored trials. Mean within-fold AUROC, 95% interval resampling whole lead drugs.

| model | AUROC |
|---|---|
| HOW | 0.712 (0.694 to 0.727) |
| WHAT | 0.666 (0.645 to 0.682) |
| WHAT + phase | 0.696 (0.677 to 0.711) |
| HOW + WHAT | 0.734 (0.716 to 0.749) |
| registration baseline (with therapeutic area) | 0.727 (0.708 to 0.742) |
| registration baseline + strict precedent flag alone | 0.734 (0.715 to 0.749) |

Paired differences (same resamples):

| comparison | AUROC difference |
|---|---|
| HOW + WHAT minus HOW | +0.022 (+0.012 to +0.031) |
| HOW + WHAT minus the registration baseline | +0.007 (−0.000 to +0.014) |
| registration baseline + strict precedent flag alone, minus the registration baseline | +0.007 (+0.003 to +0.012) |
| HOW + WHAT minus (registration baseline + strict precedent flag alone) | −0.000 (−0.006 to +0.005) |
| WHAT minus HOW | −0.046 (−0.066 to −0.028) |
| WHAT + phase minus HOW | −0.016 (−0.031 to −0.003) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.302 (0.269 to 0.338) pooled, 0.307 (0.271 to 0.336) on average within a round or fold.
- WHAT's top third succeeds 31.4 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class and HOW third within the same fold, the gap is 20.2 per 100 (16.6 to 23.6): **64% survives** (58% to 71%, lead-drug resamples). 107 of 148 strata had trials in both tails.
- WHAT's top third succeeds 31.4 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same fold, the gap is 16.8 per 100 (10.3 to 22.0): **54% survives** (34% to 67%, lead-drug resamples). 291 of 1010 strata had trials in both tails.

Success per 100 trials by third of WHAT, within phase (lead-drug 95% intervals):

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 31.8 (28.5 to 35.4), n=800 | 42.6 (39.0 to 46.1), n=789 | 60.8 (57.1 to 64.5), n=785 | +29.0 (+23.3 to +33.8) |
| Phase 3 | 54.5 (51.1 to 57.8), n=1162 | 67.4 (64.1 to 70.4), n=1161 | 81.2 (78.0 to 84.0), n=1152 | +26.7 (+22.3 to +30.7) |

With the scored trial's own lead drug left out of every group rate (training rows too). Difference is from HOW:

| model | AUROC | minus HOW |
|---|---|---|
| WHAT, own lead drug excluded | 0.665 (0.645 to 0.682) | −0.047 (−0.067 to −0.028) |
| HOW + WHAT, own lead drug excluded | 0.734 (0.716 to 0.750) | +0.023 (+0.012 to +0.032) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.303 (0.269 to 0.338) pooled, 0.307 (0.272 to 0.337) on average within a round or fold.
- WHAT's top third succeeds 31.5 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class and HOW third within the same fold, the gap is 20.6 per 100 (17.1 to 24.0): **65% survives** (59% to 72%, lead-drug resamples). 106 of 148 strata had trials in both tails.
- WHAT's top third succeeds 31.5 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same fold, the gap is 16.5 per 100 (10.2 to 21.7): **52% survives** (33% to 66%, lead-drug resamples). 290 of 1010 strata had trials in both tails.

Success per 100 by third of WHAT, own lead drug excluded:

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 31.9 (28.7 to 35.6), n=795 | 42.0 (38.3 to 45.6), n=793 | 61.1 (57.4 to 64.8), n=786 | +29.1 (+23.4 to +33.8) |
| Phase 3 | 55.1 (51.7 to 58.4), n=1160 | 66.7 (63.5 to 69.6), n=1161 | 81.2 (77.9 to 84.0), n=1154 | +26.1 (+21.7 to +30.2) |

## 2020 links (Open Targets 20.06 targets, today's drug match)

### Time-ordered rounds

2,795 scored trials. Mean within-round AUROC, 95% interval resampling whole lead drugs.

| model | AUROC |
|---|---|
| HOW | 0.697 (0.676 to 0.720) |
| WHAT | 0.661 (0.636 to 0.686) |
| WHAT + phase | 0.687 (0.663 to 0.711) |
| HOW + WHAT | 0.719 (0.696 to 0.742) |
| registration baseline (with therapeutic area) | 0.709 (0.686 to 0.731) |
| registration baseline + strict precedent flag alone | 0.715 (0.693 to 0.737) |

Paired differences (same resamples):

| comparison | AUROC difference |
|---|---|
| HOW + WHAT minus HOW | +0.022 (+0.009 to +0.034) |
| HOW + WHAT minus the registration baseline | +0.010 (+0.001 to +0.020) |
| registration baseline + strict precedent flag alone, minus the registration baseline | +0.007 (+0.001 to +0.013) |
| HOW + WHAT minus (registration baseline + strict precedent flag alone) | +0.003 (−0.003 to +0.011) |
| WHAT minus HOW | −0.037 (−0.063 to −0.011) |
| WHAT + phase minus HOW | −0.010 (−0.031 to +0.011) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.339 (0.292 to 0.387) pooled, 0.339 (0.292 to 0.387) on average within a round or fold.
- WHAT's top third succeeds 33.0 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class and HOW third within the same round, the gap is 23.3 per 100 (18.7 to 28.5): **71% survives** (63% to 78%, lead-drug resamples). 36 of 50 strata had trials in both tails.
- WHAT's top third succeeds 33.0 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same round, the gap is 22.3 per 100 (15.1 to 30.6): **68% survives** (49% to 89%, lead-drug resamples). 112 of 359 strata had trials in both tails.

Success per 100 trials by third of WHAT, within phase (lead-drug 95% intervals):

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 31.9 (27.4 to 36.5), n=417 | 38.0 (33.2 to 42.8), n=411 | 56.0 (50.9 to 61.2), n=414 | +24.1 (+17.0 to +30.9) |
| Phase 3 | 50.9 (46.0 to 55.6), n=517 | 59.2 (54.3 to 64.1), n=520 | 82.6 (78.8 to 85.9), n=516 | +31.7 (+25.7 to +37.8) |

With the scored trial's own lead drug left out of every group rate (training rows too). Difference is from HOW:

| model | AUROC | minus HOW |
|---|---|---|
| WHAT, own lead drug excluded | 0.658 (0.634 to 0.683) | −0.039 (−0.066 to −0.013) |
| HOW + WHAT, own lead drug excluded | 0.719 (0.696 to 0.742) | +0.021 (+0.008 to +0.034) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.333 (0.285 to 0.381) pooled, 0.333 (0.284 to 0.381) on average within a round or fold.
- WHAT's top third succeeds 32.2 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class and HOW third within the same round, the gap is 22.6 per 100 (17.7 to 27.8): **70% survives** (62% to 78%, lead-drug resamples). 37 of 50 strata had trials in both tails.
- WHAT's top third succeeds 32.2 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same round, the gap is 22.6 per 100 (14.7 to 30.7): **70% survives** (49% to 92%, lead-drug resamples). 116 of 359 strata had trials in both tails.

Success per 100 by third of WHAT, own lead drug excluded:

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 30.8 (26.2 to 35.4), n=412 | 39.2 (34.4 to 44.0), n=418 | 55.8 (50.3 to 60.9), n=412 | +25.0 (+17.9 to +31.9) |
| Phase 3 | 51.1 (46.3 to 55.8), n=517 | 59.6 (54.8 to 64.4), n=520 | 82.0 (78.2 to 85.4), n=516 | +30.9 (+25.0 to +37.3) |

### Grouped 10-fold by lead drug

5,849 scored trials. Mean within-fold AUROC, 95% interval resampling whole lead drugs.

| model | AUROC |
|---|---|
| HOW | 0.712 (0.694 to 0.727) |
| WHAT | 0.661 (0.639 to 0.678) |
| WHAT + phase | 0.692 (0.673 to 0.707) |
| HOW + WHAT | 0.731 (0.713 to 0.746) |
| registration baseline (with therapeutic area) | 0.727 (0.708 to 0.742) |
| registration baseline + strict precedent flag alone | 0.733 (0.714 to 0.748) |

Paired differences (same resamples):

| comparison | AUROC difference |
|---|---|
| HOW + WHAT minus HOW | +0.020 (+0.009 to +0.029) |
| HOW + WHAT minus the registration baseline | +0.004 (−0.002 to +0.011) |
| registration baseline + strict precedent flag alone, minus the registration baseline | +0.006 (+0.002 to +0.010) |
| HOW + WHAT minus (registration baseline + strict precedent flag alone) | −0.001 (−0.007 to +0.004) |
| WHAT minus HOW | −0.051 (−0.071 to −0.032) |
| WHAT + phase minus HOW | −0.020 (−0.036 to −0.007) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.306 (0.271 to 0.341) pooled, 0.311 (0.275 to 0.341) on average within a round or fold.
- WHAT's top third succeeds 30.1 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class and HOW third within the same fold, the gap is 19.1 per 100 (15.6 to 22.8): **64% survives** (56% to 70%, lead-drug resamples). 100 of 148 strata had trials in both tails.
- WHAT's top third succeeds 30.1 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same fold, the gap is 15.7 per 100 (9.5 to 21.1): **52% survives** (33% to 69%, lead-drug resamples). 272 of 1010 strata had trials in both tails.

Success per 100 trials by third of WHAT, within phase (lead-drug 95% intervals):

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 31.3 (28.1 to 34.9), n=792 | 43.3 (39.2 to 46.8), n=793 | 60.3 (56.5 to 64.2), n=789 | +29.0 (+23.5 to +34.1) |
| Phase 3 | 56.1 (52.6 to 59.3), n=1161 | 65.0 (61.5 to 68.2), n=1160 | 81.9 (78.9 to 84.7), n=1154 | +25.8 (+21.7 to +30.0) |

With the scored trial's own lead drug left out of every group rate (training rows too). Difference is from HOW:

| model | AUROC | minus HOW |
|---|---|---|
| WHAT, own lead drug excluded | 0.661 (0.640 to 0.678) | −0.051 (−0.071 to −0.032) |
| HOW + WHAT, own lead drug excluded | 0.732 (0.713 to 0.747) | +0.020 (+0.009 to +0.029) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.306 (0.272 to 0.341) pooled, 0.311 (0.275 to 0.340) on average within a round or fold.
- WHAT's top third succeeds 30.6 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class and HOW third within the same fold, the gap is 19.9 per 100 (16.2 to 23.7): **65% survives** (58% to 72%, lead-drug resamples). 101 of 148 strata had trials in both tails.
- WHAT's top third succeeds 30.6 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same fold, the gap is 16.3 per 100 (10.1 to 21.3): **53% survives** (34% to 68%, lead-drug resamples). 276 of 1010 strata had trials in both tails.

Success per 100 by third of WHAT, own lead drug excluded:

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 31.7 (28.6 to 35.3), n=794 | 42.7 (38.9 to 46.4), n=793 | 60.5 (56.8 to 64.4), n=787 | +28.7 (+23.3 to +33.5) |
| Phase 3 | 56.0 (52.7 to 59.3), n=1164 | 64.9 (61.5 to 68.1), n=1160 | 82.1 (79.1 to 84.9), n=1151 | +26.1 (+21.9 to +30.2) |

## 2020 names and links (drug matched by 20.06 names, 20.06 targets)

### Time-ordered rounds

2,795 scored trials. Mean within-round AUROC, 95% interval resampling whole lead drugs.

| model | AUROC |
|---|---|
| HOW | 0.697 (0.676 to 0.720) |
| WHAT | 0.663 (0.639 to 0.688) |
| WHAT + phase | 0.690 (0.666 to 0.714) |
| HOW + WHAT | 0.723 (0.700 to 0.745) |
| registration baseline (with therapeutic area) | 0.709 (0.686 to 0.731) |
| registration baseline + strict precedent flag alone | 0.716 (0.694 to 0.737) |

Paired differences (same resamples):

| comparison | AUROC difference |
|---|---|
| HOW + WHAT minus HOW | +0.025 (+0.013 to +0.038) |
| HOW + WHAT minus the registration baseline | +0.014 (+0.005 to +0.023) |
| registration baseline + strict precedent flag alone, minus the registration baseline | +0.007 (+0.001 to +0.013) |
| HOW + WHAT minus (registration baseline + strict precedent flag alone) | +0.007 (−0.000 to +0.015) |
| WHAT minus HOW | −0.035 (−0.063 to −0.009) |
| WHAT + phase minus HOW | −0.007 (−0.029 to +0.015) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.330 (0.284 to 0.378) pooled, 0.330 (0.285 to 0.378) on average within a round or fold.
- WHAT's top third succeeds 31.0 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class and HOW third within the same round, the gap is 21.4 per 100 (16.7 to 26.7): **69% survives** (60% to 78%, lead-drug resamples). 38 of 50 strata had trials in both tails.
- WHAT's top third succeeds 31.0 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same round, the gap is 20.4 per 100 (12.3 to 29.0): **66% survives** (43% to 89%, lead-drug resamples). 115 of 359 strata had trials in both tails.

Success per 100 trials by third of WHAT, within phase (lead-drug 95% intervals):

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 30.0 (25.7 to 34.5), n=413 | 39.9 (34.9 to 44.6), n=416 | 55.9 (50.5 to 61.0), n=413 | +25.9 (+18.9 to +32.2) |
| Phase 3 | 51.3 (46.4 to 55.9), n=519 | 58.1 (52.5 to 63.1), n=518 | 83.3 (79.7 to 86.6), n=516 | +32.1 (+26.3 to +38.1) |

With the scored trial's own lead drug left out of every group rate (training rows too). Difference is from HOW:

| model | AUROC | minus HOW |
|---|---|---|
| WHAT, own lead drug excluded | 0.661 (0.637 to 0.686) | −0.037 (−0.065 to −0.011) |
| HOW + WHAT, own lead drug excluded | 0.722 (0.700 to 0.744) | +0.025 (+0.012 to +0.038) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.323 (0.275 to 0.372) pooled, 0.323 (0.275 to 0.372) on average within a round or fold.
- WHAT's top third succeeds 30.6 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class and HOW third within the same round, the gap is 21.4 per 100 (16.7 to 26.7): **70% survives** (61% to 79%, lead-drug resamples). 38 of 50 strata had trials in both tails.
- WHAT's top third succeeds 30.6 per 100 more than its bottom third (thirds cut within each round). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same round, the gap is 20.7 per 100 (13.3 to 28.7): **68% survives** (46% to 90%, lead-drug resamples). 113 of 359 strata had trials in both tails.

Success per 100 by third of WHAT, own lead drug excluded:

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 30.4 (26.1 to 34.7), n=414 | 40.2 (35.3 to 45.0), n=415 | 55.2 (49.8 to 60.2), n=413 | +24.8 (+17.8 to +31.1) |
| Phase 3 | 51.4 (46.6 to 56.0), n=519 | 58.0 (52.7 to 63.2), n=517 | 83.2 (79.5 to 86.6), n=517 | +31.7 (+25.9 to +37.6) |

### Grouped 10-fold by lead drug

5,849 scored trials. Mean within-fold AUROC, 95% interval resampling whole lead drugs.

| model | AUROC |
|---|---|
| HOW | 0.712 (0.694 to 0.727) |
| WHAT | 0.663 (0.642 to 0.680) |
| WHAT + phase | 0.695 (0.676 to 0.710) |
| HOW + WHAT | 0.734 (0.715 to 0.748) |
| registration baseline (with therapeutic area) | 0.727 (0.708 to 0.742) |
| registration baseline + strict precedent flag alone | 0.733 (0.715 to 0.749) |

Paired differences (same resamples):

| comparison | AUROC difference |
|---|---|
| HOW + WHAT minus HOW | +0.022 (+0.012 to +0.030) |
| HOW + WHAT minus the registration baseline | +0.007 (+0.000 to +0.013) |
| registration baseline + strict precedent flag alone, minus the registration baseline | +0.006 (+0.002 to +0.011) |
| HOW + WHAT minus (registration baseline + strict precedent flag alone) | +0.000 (−0.005 to +0.005) |
| WHAT minus HOW | −0.049 (−0.069 to −0.031) |
| WHAT + phase minus HOW | −0.017 (−0.033 to −0.003) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.304 (0.271 to 0.336) pooled, 0.308 (0.273 to 0.336) on average within a round or fold.
- WHAT's top third succeeds 29.9 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class and HOW third within the same fold, the gap is 19.1 per 100 (15.7 to 22.4): **64% survives** (56% to 71%, lead-drug resamples). 103 of 148 strata had trials in both tails.
- WHAT's top third succeeds 29.9 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same fold, the gap is 14.4 per 100 (8.8 to 19.3): **48% survives** (30% to 63%, lead-drug resamples). 274 of 1010 strata had trials in both tails.

Success per 100 trials by third of WHAT, within phase (lead-drug 95% intervals):

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 32.5 (29.4 to 36.1), n=794 | 41.1 (37.4 to 45.0), n=793 | 61.4 (57.9 to 65.0), n=787 | +28.9 (+23.7 to +33.5) |
| Phase 3 | 56.3 (53.0 to 59.5), n=1165 | 64.7 (61.1 to 67.9), n=1155 | 82.0 (79.1 to 84.6), n=1155 | +25.7 (+21.5 to +29.6) |

With the scored trial's own lead drug left out of every group rate (training rows too). Difference is from HOW:

| model | AUROC | minus HOW |
|---|---|---|
| WHAT, own lead drug excluded | 0.663 (0.642 to 0.680) | −0.049 (−0.069 to −0.031) |
| HOW + WHAT, own lead drug excluded | 0.734 (0.715 to 0.748) | +0.022 (+0.012 to +0.031) |

- Correlation of HOW's and WHAT's out-of-sample log-odds (lead-drug 95% interval): 0.305 (0.273 to 0.337) pooled, 0.309 (0.273 to 0.336) on average within a round or fold.
- WHAT's top third succeeds 30.1 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class and HOW third within the same fold, the gap is 19.5 per 100 (15.9 to 22.9): **65% survives** (57% to 72%, lead-drug resamples). 102 of 148 strata had trials in both tails.
- WHAT's top third succeeds 30.1 per 100 more than its bottom third (thirds cut within each fold). Among trials of the same phase, sponsor class, therapeutic area and HOW third within the same fold, the gap is 14.6 per 100 (8.8 to 19.3): **48% survives** (30% to 64%, lead-drug resamples). 273 of 1010 strata had trials in both tails.

Success per 100 by third of WHAT, own lead drug excluded:

|  | bottom third | middle third | top third | top minus bottom |
|---|---|---|---|---|
| Phase 2 | 32.5 (29.5 to 36.1), n=800 | 41.4 (37.7 to 45.0), n=787 | 61.1 (57.6 to 64.7), n=787 | +28.6 (+23.2 to +33.3) |
| Phase 3 | 56.3 (53.1 to 59.5), n=1164 | 65.3 (61.6 to 68.7), n=1158 | 81.4 (78.6 to 84.1), n=1153 | +25.2 (+21.1 to +29.2) |
