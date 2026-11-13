# Earlier results regenerated with lead-drug intervals (TEST_PLAN_2.md item 7)

The first run's code and models: the model families added since (precedent, hierarchical, whathow) are left out, so every row is a row of the first run. Only the bootstrap differs: each resample draws whole lead drugs instead of single trials.

Trained 53 models on 5,849 development trials.

Differences are first-named minus second-named. AUROC differences are paired: taken within each bootstrap resample. Calibration gap is mean predicted minus observed success, in points (positive means over-prediction). Intervals are 95% percentile intervals from 1,000 resamples of lead drugs (all of a drug's trials together), except the adjusted differences of E1 and E3, which use 2,000 resamples of lead drugs; trials that share a sponsor are not independent, so intervals are somewhat too narrow. The expectation for each item is in FINAL_PLAN.md, written before this run.

## R1 (reference): Does the registry's registration model reproduce the pre-registered one?

Matches to three decimals. | Set | Printed by the baseline run | This run |
|---|---|---|
| Test set (finished 2021 to 2023) | 0.730 | 0.730 |
| Late set (finished 2024 to 2025) | 0.733 | 0.733 |

## P1 (primary): Does the best honest model built here beat the pre-registered baseline?

### Test set (finished 2021 to 2023)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| P1 | ladder: registration + endpoint + history + genetics + words minus logistic: registration | +0.018 (-0.001 to +0.039) | AUROC, 1,236 trials, success 50.6% |
| P1 | AUROC of ladder: registration + endpoint + history + genetics + words | 0.748 (0.716 to 0.780) | 1,236 trials, success 50.6% |
| P1 | Brier score of ladder: registration + endpoint + history + genetics + words (lower is better) | 0.205 (no interval) | 1,236 trials, success 50.6% |
| P1 | calibration gap of ladder: registration + endpoint + history + genetics + words: mean predicted minus observed | +3.0 points (+0.5 to +5.7) | 1,236 trials, success 50.6% |
| P1 | AUROC of logistic: registration | 0.730 (0.698 to 0.761) | 1,236 trials, success 50.6% |
| P1 | Brier score of logistic: registration (lower is better) | 0.213 (no interval) | 1,236 trials, success 50.6% |
| P1 | calibration gap of logistic: registration: mean predicted minus observed | +4.6 points (+1.9 to +7.3) | 1,236 trials, success 50.6% |

### Test and late sets together (2021 to 2025)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| P1 | ladder: registration + endpoint + history + genetics + words minus logistic: registration | +0.026 (+0.008 to +0.045) | AUROC, 1,728 trials, success 49.8% |
| P1 | AUROC of ladder: registration + endpoint + history + genetics + words | 0.757 (0.729 to 0.783) | 1,728 trials, success 49.8% |
| P1 | Brier score of ladder: registration + endpoint + history + genetics + words (lower is better) | 0.202 (no interval) | 1,728 trials, success 49.8% |
| P1 | calibration gap of ladder: registration + endpoint + history + genetics + words: mean predicted minus observed | +3.6 points (+1.4 to +6.0) | 1,728 trials, success 49.8% |
| P1 | AUROC of logistic: registration | 0.731 (0.702 to 0.759) | 1,728 trials, success 49.8% |
| P1 | Brier score of logistic: registration (lower is better) | 0.215 (no interval) | 1,728 trials, success 49.8% |
| P1 | calibration gap of logistic: registration: mean predicted minus observed | +5.5 points (+3.1 to +8.1) | 1,728 trials, success 49.8% |

## S1 (secondary): How does every other registered model do, scored once?

### Honest models

| Model | AUROC (95% interval) | AUROC minus registration | Brier | Gap (points) |
|---|---|---|---|---|
| ladder: boosted trees, endpoint + history + genetics | 0.755 (0.722 to 0.785) | +0.024 (+0.009 to +0.041) | 0.202 | +3.1 |
| biology: ladder, all four tables + science | 0.753 (0.720 to 0.783) | +0.022 (+0.002 to +0.044) | 0.204 | +3.2 |
| ladder: registration + words | 0.752 (0.721 to 0.784) | +0.022 (+0.004 to +0.042) | 0.203 | +3.6 |
| ladder: registration + endpoint + history + genetics + words | 0.748 (0.716 to 0.780) | +0.018 (-0.001 to +0.039) | 0.205 | +3.0 |
| embeddings: general vector plus registration plus words (uses a pretrained model) | 0.748 (0.714 to 0.780) | +0.017 (-0.010 to +0.045) | 0.206 | +3.3 |
| embeddings: general vector plus registration (uses a pretrained model) | 0.747 (0.714 to 0.778) | +0.016 (-0.008 to +0.041) | 0.206 | +3.3 |
| random forest: registration | 0.744 (0.712 to 0.776) | +0.014 (+0.001 to +0.027) | 0.209 | +4.9 |
| logistic: registration + pooled sponsor | 0.743 (0.711 to 0.773) | +0.012 (+0.006 to +0.019) | 0.210 | +4.7 |
| biology: registration + all sources | 0.742 (0.709 to 0.772) | +0.012 (+0.001 to +0.022) | 0.209 | +4.6 |
| boosted trees: registration, tuned | 0.741 (0.708 to 0.771) | +0.010 (+0.001 to +0.019) | 0.209 | +4.9 |
| boosted trees: registration | 0.740 (0.706 to 0.771) | +0.010 (-0.000 to +0.019) | 0.210 | +5.0 |
| neural network: registration | 0.740 (0.708 to 0.771) | +0.009 (+0.001 to +0.018) | 0.208 | +4.4 |
| embeddings: biomedical vector plus registration plus words (uses a pretrained model) | 0.739 (0.708 to 0.771) | +0.009 (-0.017 to +0.039) | 0.209 | +2.7 |
| logistic: last 8 years only | 0.738 (0.705 to 0.768) | +0.007 (+0.001 to +0.013) | 0.209 | +3.3 |
| embeddings: biomedical vector plus registration (uses a pretrained model) | 0.736 (0.706 to 0.767) | +0.006 (-0.021 to +0.036) | 0.209 | +2.3 |
| biology: registration + expression | 0.736 (0.703 to 0.767) | +0.005 (-0.002 to +0.012) | 0.211 | +4.4 |
| logistic: separate by phase | 0.735 (0.701 to 0.766) | +0.005 (-0.005 to +0.015) | 0.210 | +4.6 |
| logistic: recency-weighted | 0.735 (0.703 to 0.766) | +0.005 (+0.000 to +0.010) | 0.209 | +2.8 |
| biology: registration + annotations | 0.735 (0.703 to 0.765) | +0.004 (-0.000 to +0.009) | 0.211 | +4.4 |
| ladder: registration + genetics | 0.734 (0.701 to 0.764) | +0.004 (-0.003 to +0.010) | 0.212 | +4.8 |
| logistic: registration + interactions | 0.734 (0.701 to 0.765) | +0.003 (-0.005 to +0.011) | 0.211 | +4.7 |
| logistic: last 4 years only | 0.734 (0.701 to 0.765) | +0.003 (-0.006 to +0.013) | 0.209 | +1.0 |
| biology: registration + DepMap | 0.733 (0.699 to 0.764) | +0.003 (-0.001 to +0.006) | 0.212 | +4.7 |
| biology: registration + networks | 0.732 (0.699 to 0.764) | +0.002 (-0.001 to +0.005) | 0.212 | +4.7 |
| ladder: registration + endpoint | 0.732 (0.699 to 0.763) | +0.002 (-0.010 to +0.012) | 0.211 | +2.9 |
| logistic: registration | 0.730 (0.698 to 0.761) | (baseline) | 0.213 | +4.6 |
| logistic: trend-shifted intercept | 0.730 (0.698 to 0.761) | +0.000 (+0.000 to +0.000) | 0.213 | -5.7 |
| biology: registration + chemistry | 0.730 (0.697 to 0.761) | -0.001 (-0.005 to +0.003) | 0.213 | +4.9 |
| logistic: registration, tuned penalty | 0.730 (0.697 to 0.761) | -0.001 (-0.004 to +0.002) | 0.213 | +4.7 |
| ladder: registration + history | 0.729 (0.696 to 0.759) | -0.001 (-0.007 to +0.005) | 0.213 | +4.4 |
| logistic: separate for company and other sponsors | 0.724 (0.692 to 0.756) | -0.006 (-0.015 to +0.004) | 0.214 | +4.7 |
| text: words | 0.709 (0.675 to 0.743) | -0.021 (-0.050 to +0.012) | 0.218 | +4.4 |
| text: words without drug names | 0.708 (0.674 to 0.743) | -0.022 (-0.051 to +0.011) | 0.219 | +4.4 |
| embeddings: general vector alone (uses a pretrained model) | 0.707 (0.672 to 0.742) | -0.023 (-0.056 to +0.010) | 0.220 | +4.2 |
| embeddings: biomedical vector alone (uses a pretrained model) | 0.704 (0.672 to 0.736) | -0.027 (-0.058 to +0.008) | 0.221 | +2.2 |
| lookup: phase x sponsor class | 0.684 (0.654 to 0.712) | -0.047 (-0.069 to -0.024) | 0.225 | +6.5 |
| memory: lead sponsor | 0.656 (0.620 to 0.690) | -0.074 (-0.111 to -0.038) | 0.240 | +8.7 |
| lookup: phase | 0.630 (0.601 to 0.657) | -0.101 (-0.129 to -0.072) | 0.238 | +6.8 |
| text: drug names only | 0.625 (0.585 to 0.665) | -0.105 (-0.147 to -0.063) | 0.239 | +4.6 |
| lookup: constant | 0.500 (0.500 to 0.500) | -0.230 (-0.261 to -0.198) | 0.256 | +7.8 |

### Not honest (deliberately leaky)

| Model | AUROC (95% interval) | AUROC minus registration | Brier | Gap (points) |
|---|---|---|---|---|
| leaky: registration + post-hoc fields | 0.779 (0.749 to 0.807) | +0.048 (+0.030 to +0.067) | 0.194 | +1.9 |
| ladder: all four tables plus after-the-start columns (not honest) | 0.756 (0.724 to 0.786) | +0.026 (+0.007 to +0.047) | 0.201 | +2.2 |
| leaky: registration + results structure | 0.740 (0.707 to 0.770) | +0.010 (+0.001 to +0.017) | 0.209 | +4.1 |
| text: words with summary (leaky) | 0.716 (0.683 to 0.749) | -0.014 (-0.044 to +0.018) | 0.216 | +4.2 |
| memory: lead drug | 0.633 (0.593 to 0.672) | -0.097 (-0.139 to -0.057) | 0.245 | +9.5 |

Not ranked here (not meaningful on all trials: they predict one constant where a trial has no mapped target; scored on mapped trials in `results/dev_science.md`): biology: logistic, all sources, biology: boosted trees, all sources, biology: logistic, DepMap only, biology: logistic, expression only, biology: logistic, networks only, biology: logistic, annotations only, biology: logistic, chemistry only.

Skipped (scored on the sampled trials only: see E4): llm: haiku (identified), llm: haiku (masked), llm: haiku (blinded), llm: sonnet (identified), llm: sonnet (masked), llm: sonnet (blinded).

## S2 (secondary): What does each table of information add to registration details?

### Test set (finished 2021 to 2023)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S2 | + endpoint minus logistic: registration | +0.002 (-0.010 to +0.012) | AUROC, 1,236 trials, success 50.6% |
| S2 | + history minus logistic: registration | -0.001 (-0.007 to +0.005) | AUROC, 1,236 trials, success 50.6% |
| S2 | + genetics minus logistic: registration | +0.004 (-0.003 to +0.010) | AUROC, 1,236 trials, success 50.6% |
| S2 | + words minus logistic: registration | +0.022 (+0.004 to +0.042) | AUROC, 1,236 trials, success 50.6% |
| S2 | all four tables minus logistic: registration | +0.018 (-0.001 to +0.039) | AUROC, 1,236 trials, success 50.6% |

### Test and late sets together (2021 to 2025)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S2 | + endpoint minus logistic: registration | +0.004 (-0.007 to +0.013) | AUROC, 1,728 trials, success 49.8% |
| S2 | + history minus logistic: registration | -0.002 (-0.007 to +0.004) | AUROC, 1,728 trials, success 49.8% |
| S2 | + genetics minus logistic: registration | +0.004 (-0.002 to +0.008) | AUROC, 1,728 trials, success 49.8% |
| S2 | + words minus logistic: registration | +0.026 (+0.010 to +0.045) | AUROC, 1,728 trials, success 49.8% |
| S2 | all four tables minus logistic: registration | +0.026 (+0.008 to +0.045) | AUROC, 1,728 trials, success 49.8% |

## S3 (secondary): Does the trend-shifted intercept reduce the over-prediction?

This is an arithmetic check, not a test. The trend-shifted model moves only the intercept of the registration model, so its mean prediction sits a fixed distance below the registration model's (that distance needs no labels) and its gap is the registration model's gap plus that shift, whatever the labels say.

### Test set (finished 2021 to 2023)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S3 | logistic: registration: mean predicted minus observed | +4.6 points (+1.9 to +7.3) | observed success 50.6% |
| S3 | logistic: trend-shifted intercept: mean predicted minus observed | -5.7 points (-8.4 to -2.9) | observed success 50.6% |
| S3 | logistic: trend-shifted intercept minus logistic: registration: mean prediction, no labels needed | -10.3 points (no interval) | 1,236 trials, success 50.6% |
| S3 | logistic: trend-shifted intercept minus logistic: registration | +0.000 (+0.000 to +0.000) | AUROC, 1,236 trials, success 50.6% |

### Late set (finished 2024 to 2025)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S3 | logistic: registration: mean predicted minus observed | +7.6 points (+3.4 to +12.2) | observed success 47.8% |
| S3 | logistic: trend-shifted intercept: mean predicted minus observed | -2.7 points (-6.9 to +1.7) | observed success 47.8% |
| S3 | logistic: trend-shifted intercept minus logistic: registration: mean prediction, no labels needed | -10.3 points (no interval) | 492 trials, success 47.8% |
| S3 | logistic: trend-shifted intercept minus logistic: registration | +0.000 (+0.000 to +0.000) | AUROC, 492 trials, success 47.8% |

## S4 (secondary): Do the public science sources help, and does the positive control show?

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S4 | biology: registration + all sources minus logistic: registration, mapped trials | +0.019 (+0.003 to +0.036) | AUROC, 764 trials, success 53.8% |
| S4 | control: registration + genetic support minus logistic: registration, mapped trials | +0.006 (-0.002 to +0.016) | AUROC, 764 trials, success 53.8% |
| S4 | biology: ladder, all four tables + science minus ladder: registration + endpoint + history + genetics + words, mapped trials | +0.012 (-0.001 to +0.022) | AUROC, 764 trials, success 53.8% |
| S4 | biology: registration + all sources minus logistic: registration, all trials | +0.012 (+0.001 to +0.022) | AUROC, 1,236 trials, success 50.6% |
| S4 | control: registration + genetic support minus logistic: registration, all trials | +0.004 (-0.001 to +0.008) | AUROC, 1,236 trials, success 50.6% |

## S5 (secondary): Do the primary and ladder comparisons hold on the late set?

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S5 | ladder: registration + endpoint + history + genetics + words minus logistic: registration | +0.047 (+0.018 to +0.076) | AUROC, 492 trials, success 47.8% |
| S5 | AUROC of ladder: registration + endpoint + history + genetics + words | 0.780 (0.735 to 0.822) | 492 trials, success 47.8% |
| S5 | Brier score of ladder: registration + endpoint + history + genetics + words (lower is better) | 0.196 (no interval) | 492 trials, success 47.8% |
| S5 | calibration gap of ladder: registration + endpoint + history + genetics + words: mean predicted minus observed | +5.1 points (+1.1 to +9.4) | 492 trials, success 47.8% |
| S5 | AUROC of logistic: registration | 0.733 (0.686 to 0.777) | 492 trials, success 47.8% |
| S5 | Brier score of logistic: registration (lower is better) | 0.219 (no interval) | 492 trials, success 47.8% |
| S5 | calibration gap of logistic: registration: mean predicted minus observed | +7.6 points (+3.4 to +12.2) | 492 trials, success 47.8% |
| S5 | + endpoint minus logistic: registration | +0.010 (-0.009 to +0.026) | AUROC, 492 trials, success 47.8% |
| S5 | + history minus logistic: registration | -0.003 (-0.013 to +0.007) | AUROC, 492 trials, success 47.8% |
| S5 | + genetics minus logistic: registration | +0.005 (-0.005 to +0.016) | AUROC, 492 trials, success 47.8% |
| S5 | + words minus logistic: registration | +0.035 (+0.007 to +0.063) | AUROC, 492 trials, success 47.8% |
| S5 | all four tables minus logistic: registration | +0.047 (+0.018 to +0.076) | AUROC, 492 trials, success 47.8% |

## E1 (exploratory): Do biomarker endpoints succeed more, and survival endpoints less?

### Test set (finished 2021 to 2023)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E1 | biomarker first primary endpoint minus all other endpoint types, crude | +11.9 points (+4.8 to +18.8) | 232 against 1,004; success 60.3% against 48.4% |
| E1 | biomarker first primary endpoint minus all other endpoint types, adjusted | +6.4 points (-1.6 to +15.4) | 232 against 1,004; success 60.3% against 48.4%; adjusted for phase, sponsor type, area, drug type |
| E1 | survival first primary endpoint minus all other endpoint types, crude | -11.4 points (-18.7 to -3.8) | 195 against 1,041; success 41.0% against 52.4% |
| E1 | survival first primary endpoint minus all other endpoint types, adjusted | -14.6 points (-23.9 to -5.0) | 195 against 1,041; success 41.0% against 52.4%; adjusted for phase, sponsor type, area, drug type |

### Late set (finished 2024 to 2025)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E1 | biomarker first primary endpoint minus all other endpoint types, crude | +31.7 points (+21.3 to +40.7) | 106 against 386; success 72.6% against 40.9% |
| E1 | biomarker first primary endpoint minus all other endpoint types, adjusted | +19.9 points (+7.7 to +33.3) | 106 against 386; success 72.6% against 40.9%; adjusted for phase, sponsor type, area, drug type |
| E1 | survival first primary endpoint minus all other endpoint types, crude | -22.0 points (-32.8 to -9.0) | 63 against 429; success 28.6% against 50.6% |
| E1 | survival first primary endpoint minus all other endpoint types, adjusted | -33.3 points (-53.8 to -15.4) | 63 against 429; success 28.6% against 50.6%; adjusted for phase, sponsor type, area, drug type |

## E2 (exploratory): Do Phase 3 trials succeed more after a successful Phase 2?

### Test set (finished 2021 to 2023)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E2 | Phase 3 after a successful Phase 2 minus after a failed Phase 2 | +3.4 points (-17.3 to +33.1) | 51 against 12; success 78.4% against 75.0% |

### Late set (finished 2024 to 2025)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E2 | Phase 3 after a successful Phase 2 minus after a failed Phase 2 | +33.3 points (-7.9 to +61.2) | 27 against 6; success 66.7% against 33.3% |

## E3 (exploratory): Do single-target drugs with a broadly expressed target succeed less?

### Test set (finished 2021 to 2023)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E3 | expr_n_tissues, top third minus bottom third, single-target drugs, adjusted | -11.4 points (-27.1 to +5.6) | 124 against 169; success 46.0% against 68.6%; top third is more than 46 tissues, bottom third is 14 or fewer |

### Late set (finished 2024 to 2025)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E3 | expr_n_tissues, top third minus bottom third, single-target drugs, adjusted | +13.2 points (-16.3 to +43.5) | 58 against 63; success 51.7% against 57.1%; top third is more than 46 tissues, bottom third is 14 or fewer |

## E4 (exploratory): How do the language models do, and how much of it is recall?

### Sampled test trials

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E4 | AUROC of haiku (identified) | 0.813 (0.762 to 0.860) | 300 sampled trials |
| E4 | AUROC of haiku (masked) | 0.801 (0.747 to 0.848) | 300 sampled trials |
| E4 | AUROC of haiku (blinded) | 0.788 (0.733 to 0.837) | 300 sampled trials |
| E4 | AUROC of sonnet (identified) | 0.854 (0.808 to 0.896) | 300 sampled trials |
| E4 | AUROC of sonnet (masked) | 0.845 (0.794 to 0.887) | 300 sampled trials |
| E4 | AUROC of sonnet (blinded) | 0.812 (0.755 to 0.858) | 300 sampled trials |
| E4 | AUROC of logistic: registration | 0.737 (0.675 to 0.792) | 300 sampled trials |
| E4 | AUROC of ladder: registration + words | 0.772 (0.714 to 0.829) | 300 sampled trials |
| E4 | haiku (identified) minus haiku (blinded) | +0.025 (-0.011 to +0.064) | 300 sampled trials |
| E4 | haiku (masked) minus haiku (blinded) | +0.012 (-0.025 to +0.053) | 300 sampled trials |
| E4 | haiku (identified) minus haiku (masked) | +0.012 (-0.013 to +0.039) | 300 sampled trials |
| E4 | sonnet (identified) minus sonnet (blinded) | +0.042 (-0.002 to +0.087) | 300 sampled trials |
| E4 | sonnet (masked) minus sonnet (blinded) | +0.032 (-0.010 to +0.075) | 300 sampled trials |
| E4 | sonnet (identified) minus sonnet (masked) | +0.010 (-0.009 to +0.029) | 300 sampled trials |
| E4 | haiku (identified) minus logistic: registration | +0.076 (+0.014 to +0.144) | 300 sampled trials |
| E4 | haiku (identified) minus ladder: registration + words | +0.041 (-0.017 to +0.101) | 300 sampled trials |
| E4 | haiku (masked) minus logistic: registration | +0.064 (-0.002 to +0.130) | 300 sampled trials |
| E4 | haiku (masked) minus ladder: registration + words | +0.028 (-0.035 to +0.090) | 300 sampled trials |
| E4 | haiku (blinded) minus logistic: registration | +0.052 (-0.012 to +0.117) | 300 sampled trials |
| E4 | haiku (blinded) minus ladder: registration + words | +0.016 (-0.042 to +0.070) | 300 sampled trials |
| E4 | sonnet (identified) minus logistic: registration | +0.118 (+0.055 to +0.189) | 300 sampled trials |
| E4 | sonnet (identified) minus ladder: registration + words | +0.082 (+0.022 to +0.140) | 300 sampled trials |
| E4 | sonnet (masked) minus logistic: registration | +0.108 (+0.045 to +0.176) | 300 sampled trials |
| E4 | sonnet (masked) minus ladder: registration + words | +0.072 (+0.014 to +0.131) | 300 sampled trials |
| E4 | sonnet (blinded) minus logistic: registration | +0.076 (+0.015 to +0.136) | 300 sampled trials |
| E4 | sonnet (blinded) minus ladder: registration + words | +0.040 (-0.013 to +0.089) | 300 sampled trials |
| E4 | AUROC of blinded haiku, guess names the drug, drug not named | 0.730 (0.659 to 0.800) | 212 trials, success 43% |
| E4 | blinded haiku, guess names the drug, drug not named, minus logistic: registration | -0.002 (-0.085 to +0.075) | 212 trials, success 43% |
| E4 | blinded haiku, guess names the drug, drug not named, minus ladder: registration + words | -0.020 (-0.092 to +0.049) | 212 trials, success 43% |
| E4 | AUROC of blinded haiku, guess names the drug, drug named | 0.862 (0.779 to 0.936) | 88 trials, success 70% |
| E4 | blinded haiku, guess names the drug, drug named, minus logistic: registration | +0.118 (+0.002 to +0.244) | 88 trials, success 70% |
| E4 | blinded haiku, guess names the drug, drug named, minus ladder: registration + words | +0.057 (-0.057 to +0.168) | 88 trials, success 70% |
| E4 | AUROC of blinded sonnet, guess names the drug, drug not named | 0.750 (0.666 to 0.822) | 171 trials, success 42% |
| E4 | blinded sonnet, guess names the drug, drug not named, minus logistic: registration | +0.026 (-0.065 to +0.114) | 171 trials, success 42% |
| E4 | blinded sonnet, guess names the drug, drug not named, minus ladder: registration + words | -0.017 (-0.088 to +0.056) | 171 trials, success 42% |
| E4 | AUROC of blinded sonnet, guess names the drug, drug named | 0.879 (0.820 to 0.932) | 129 trials, success 64% |
| E4 | blinded sonnet, guess names the drug, drug named, minus logistic: registration | +0.137 (+0.051 to +0.223) | 129 trials, success 64% |
| E4 | blinded sonnet, guess names the drug, drug named, minus ladder: registration + words | +0.111 (+0.036 to +0.184) | 129 trials, success 64% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named | 0.731 (0.656 to 0.799) | 200 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, minus logistic: registration | -0.005 (-0.084 to +0.079) | 200 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, minus ladder: registration + words | -0.021 (-0.089 to +0.043) | 200 trials, success 42% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named | 0.854 (0.767 to 0.927) | 100 trials, success 68% |
| E4 | blinded haiku, guess or reason names it, drug named, minus logistic: registration | +0.122 (+0.012 to +0.241) | 100 trials, success 68% |
| E4 | blinded haiku, guess or reason names it, drug named, minus ladder: registration + words | +0.057 (-0.049 to +0.163) | 100 trials, success 68% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named | 0.750 (0.668 to 0.833) | 154 trials, success 42% |
| E4 | blinded sonnet, guess or reason names it, drug not named, minus logistic: registration | +0.040 (-0.061 to +0.143) | 154 trials, success 42% |
| E4 | blinded sonnet, guess or reason names it, drug not named, minus ladder: registration + words | -0.020 (-0.096 to +0.058) | 154 trials, success 42% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named | 0.866 (0.806 to 0.919) | 146 trials, success 61% |
| E4 | blinded sonnet, guess or reason names it, drug named, minus logistic: registration | +0.109 (+0.034 to +0.177) | 146 trials, success 61% |
| E4 | blinded sonnet, guess or reason names it, drug named, minus ladder: registration + words | +0.103 (+0.033 to +0.170) | 146 trials, success 61% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial | 0.732 (0.656 to 0.798) | 195 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial, minus logistic: registration | +0.000 (-0.080 to +0.073) | 195 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial, minus ladder: registration + words | -0.019 (-0.091 to +0.040) | 195 trials, success 42% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named, said it recognised the trial | n/a (n/a to n/a) | too few to score; 5 trials, success 80% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it recognised the trial, minus logistic: registration | n/a (n/a to n/a) | too few to score; 5 trials, success 80% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it recognised the trial, minus ladder: registration + words | n/a (n/a to n/a) | too few to score; 5 trials, success 80% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named, said it did not recognise the trial | 0.870 (0.765 to 0.950) | 59 trials, success 61% |
| E4 | blinded haiku, guess or reason names it, drug named, said it did not recognise the trial, minus logistic: registration | +0.158 (+0.027 to +0.306) | 59 trials, success 61% |
| E4 | blinded haiku, guess or reason names it, drug named, said it did not recognise the trial, minus ladder: registration + words | +0.074 (-0.042 to +0.199) | 59 trials, success 61% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named, said it recognised the trial | 0.859 (0.707 to 0.971) | 41 trials, success 78% |
| E4 | blinded haiku, guess or reason names it, drug named, said it recognised the trial, minus logistic: registration | +0.115 (-0.017 to +0.288) | 41 trials, success 78% |
| E4 | blinded haiku, guess or reason names it, drug named, said it recognised the trial, minus ladder: registration + words | +0.064 (-0.133 to +0.276) | 41 trials, success 78% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial | 0.772 (0.681 to 0.853) | 119 trials, success 41% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial, minus logistic: registration | +0.024 (-0.082 to +0.123) | 119 trials, success 41% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial, minus ladder: registration + words | -0.028 (-0.111 to +0.052) | 119 trials, success 41% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named, said it recognised the trial | 0.687 (0.486 to 0.859) | 35 trials, success 43% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it recognised the trial, minus logistic: registration | +0.130 (-0.112 to +0.392) | 35 trials, success 43% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it recognised the trial, minus ladder: registration + words | +0.037 (-0.147 to +0.235) | 35 trials, success 43% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial | 0.850 (0.745 to 0.931) | 58 trials, success 59% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial, minus logistic: registration | +0.088 (-0.001 to +0.191) | 58 trials, success 59% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial, minus ladder: registration + words | +0.067 (-0.012 to +0.151) | 58 trials, success 59% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named, said it recognised the trial | 0.874 (0.799 to 0.935) | 88 trials, success 62% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it recognised the trial, minus logistic: registration | +0.112 (+0.003 to +0.223) | 88 trials, success 62% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it recognised the trial, minus ladder: registration + words | +0.114 (+0.021 to +0.209) | 88 trials, success 62% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named by neither model | 0.693 (0.605 to 0.784) | 136 trials, success 37% |
| E4 | blinded haiku, guess or reason names it, drug named by neither model, minus logistic: registration | -0.004 (-0.116 to +0.105) | 136 trials, success 37% |
| E4 | blinded haiku, guess or reason names it, drug named by neither model, minus ladder: registration + words | -0.032 (-0.122 to +0.063) | 136 trials, success 37% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named by neither model | 0.710 (0.619 to 0.804) | 136 trials, success 37% |
| E4 | blinded sonnet, guess or reason names it, drug named by neither model, minus logistic: registration | +0.013 (-0.104 to +0.128) | 136 trials, success 37% |
| E4 | blinded sonnet, guess or reason names it, drug named by neither model, minus ladder: registration + words | -0.014 (-0.109 to +0.078) | 136 trials, success 37% |

### Sampled late trials

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E4 | AUROC of haiku (identified) | 0.853 (0.803 to 0.896) | 300 sampled trials |
| E4 | AUROC of haiku (masked) | 0.870 (0.822 to 0.907) | 300 sampled trials |
| E4 | AUROC of haiku (blinded) | 0.831 (0.780 to 0.872) | 300 sampled trials |
| E4 | AUROC of sonnet (identified) | 0.847 (0.800 to 0.888) | 300 sampled trials |
| E4 | AUROC of sonnet (masked) | 0.846 (0.800 to 0.888) | 300 sampled trials |
| E4 | AUROC of sonnet (blinded) | 0.808 (0.756 to 0.851) | 300 sampled trials |
| E4 | AUROC of logistic: registration | 0.701 (0.641 to 0.760) | 300 sampled trials |
| E4 | AUROC of ladder: registration + words | 0.769 (0.714 to 0.821) | 300 sampled trials |
| E4 | haiku (identified) minus haiku (blinded) | +0.022 (-0.014 to +0.060) | 300 sampled trials |
| E4 | haiku (masked) minus haiku (blinded) | +0.039 (+0.004 to +0.077) | 300 sampled trials |
| E4 | haiku (identified) minus haiku (masked) | -0.017 (-0.048 to +0.010) | 300 sampled trials |
| E4 | sonnet (identified) minus sonnet (blinded) | +0.039 (-0.001 to +0.082) | 300 sampled trials |
| E4 | sonnet (masked) minus sonnet (blinded) | +0.038 (+0.004 to +0.076) | 300 sampled trials |
| E4 | sonnet (identified) minus sonnet (masked) | +0.001 (-0.022 to +0.025) | 300 sampled trials |
| E4 | haiku (identified) minus logistic: registration | +0.151 (+0.091 to +0.212) | 300 sampled trials |
| E4 | haiku (identified) minus ladder: registration + words | +0.084 (+0.024 to +0.143) | 300 sampled trials |
| E4 | haiku (masked) minus logistic: registration | +0.168 (+0.108 to +0.229) | 300 sampled trials |
| E4 | haiku (masked) minus ladder: registration + words | +0.101 (+0.042 to +0.160) | 300 sampled trials |
| E4 | haiku (blinded) minus logistic: registration | +0.129 (+0.066 to +0.186) | 300 sampled trials |
| E4 | haiku (blinded) minus ladder: registration + words | +0.062 (+0.008 to +0.112) | 300 sampled trials |
| E4 | sonnet (identified) minus logistic: registration | +0.146 (+0.087 to +0.206) | 300 sampled trials |
| E4 | sonnet (identified) minus ladder: registration + words | +0.078 (+0.018 to +0.135) | 300 sampled trials |
| E4 | sonnet (masked) minus logistic: registration | +0.145 (+0.084 to +0.203) | 300 sampled trials |
| E4 | sonnet (masked) minus ladder: registration + words | +0.077 (+0.019 to +0.135) | 300 sampled trials |
| E4 | sonnet (blinded) minus logistic: registration | +0.107 (+0.044 to +0.164) | 300 sampled trials |
| E4 | sonnet (blinded) minus ladder: registration + words | +0.039 (-0.016 to +0.089) | 300 sampled trials |
| E4 | AUROC of blinded haiku, guess names the drug, drug not named | 0.772 (0.703 to 0.834) | 197 trials, success 42% |
| E4 | blinded haiku, guess names the drug, drug not named, minus logistic: registration | +0.131 (+0.038 to +0.224) | 197 trials, success 42% |
| E4 | blinded haiku, guess names the drug, drug not named, minus ladder: registration + words | +0.053 (-0.030 to +0.131) | 197 trials, success 42% |
| E4 | AUROC of blinded haiku, guess names the drug, drug named | 0.885 (0.806 to 0.945) | 103 trials, success 59% |
| E4 | blinded haiku, guess names the drug, drug named, minus logistic: registration | +0.110 (+0.019 to +0.205) | 103 trials, success 59% |
| E4 | blinded haiku, guess names the drug, drug named, minus ladder: registration + words | +0.045 (-0.034 to +0.127) | 103 trials, success 59% |
| E4 | AUROC of blinded sonnet, guess names the drug, drug not named | 0.791 (0.719 to 0.856) | 178 trials, success 44% |
| E4 | blinded sonnet, guess names the drug, drug not named, minus logistic: registration | +0.108 (+0.022 to +0.194) | 178 trials, success 44% |
| E4 | blinded sonnet, guess names the drug, drug not named, minus ladder: registration + words | +0.016 (-0.058 to +0.092) | 178 trials, success 44% |
| E4 | AUROC of blinded sonnet, guess names the drug, drug named | 0.818 (0.742 to 0.891) | 122 trials, success 52% |
| E4 | blinded sonnet, guess names the drug, drug named, minus logistic: registration | +0.101 (-0.004 to +0.202) | 122 trials, success 52% |
| E4 | blinded sonnet, guess names the drug, drug named, minus ladder: registration + words | +0.046 (-0.034 to +0.128) | 122 trials, success 52% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named | 0.772 (0.704 to 0.832) | 191 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, minus logistic: registration | +0.125 (+0.036 to +0.206) | 191 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, minus ladder: registration + words | +0.050 (-0.028 to +0.128) | 191 trials, success 42% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named | 0.883 (0.810 to 0.944) | 109 trials, success 57% |
| E4 | blinded haiku, guess or reason names it, drug named, minus logistic: registration | +0.121 (+0.027 to +0.213) | 109 trials, success 57% |
| E4 | blinded haiku, guess or reason names it, drug named, minus ladder: registration + words | +0.045 (-0.036 to +0.126) | 109 trials, success 57% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named | 0.790 (0.718 to 0.863) | 164 trials, success 45% |
| E4 | blinded sonnet, guess or reason names it, drug not named, minus logistic: registration | +0.102 (+0.008 to +0.191) | 164 trials, success 45% |
| E4 | blinded sonnet, guess or reason names it, drug not named, minus ladder: registration + words | +0.012 (-0.072 to +0.086) | 164 trials, success 45% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named | 0.820 (0.746 to 0.887) | 136 trials, success 51% |
| E4 | blinded sonnet, guess or reason names it, drug named, minus logistic: registration | +0.108 (+0.013 to +0.205) | 136 trials, success 51% |
| E4 | blinded sonnet, guess or reason names it, drug named, minus ladder: registration + words | +0.049 (-0.022 to +0.126) | 136 trials, success 51% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial | 0.772 (0.704 to 0.832) | 191 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial, minus logistic: registration | +0.125 (+0.036 to +0.206) | 191 trials, success 42% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial, minus ladder: registration + words | +0.050 (-0.028 to +0.128) | 191 trials, success 42% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named, said it recognised the trial | n/a (n/a to n/a) | too few to score; 0 trials, success nan% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it recognised the trial, minus logistic: registration | n/a (n/a to n/a) | too few to score; 0 trials, success nan% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it recognised the trial, minus ladder: registration + words | n/a (n/a to n/a) | too few to score; 0 trials, success nan% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named, said it did not recognise the trial | 0.886 (0.783 to 0.964) | 55 trials, success 51% |
| E4 | blinded haiku, guess or reason names it, drug named, said it did not recognise the trial, minus logistic: registration | +0.151 (+0.020 to +0.300) | 55 trials, success 51% |
| E4 | blinded haiku, guess or reason names it, drug named, said it did not recognise the trial, minus ladder: registration + words | +0.028 (-0.079 to +0.145) | 55 trials, success 51% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named, said it recognised the trial | 0.890 (0.792 to 0.963) | 54 trials, success 63% |
| E4 | blinded haiku, guess or reason names it, drug named, said it recognised the trial, minus logistic: registration | +0.101 (+0.001 to +0.208) | 54 trials, success 63% |
| E4 | blinded haiku, guess or reason names it, drug named, said it recognised the trial, minus ladder: registration + words | +0.065 (-0.046 to +0.175) | 54 trials, success 63% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial | 0.781 (0.694 to 0.868) | 117 trials, success 42% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial, minus logistic: registration | +0.099 (-0.013 to +0.209) | 117 trials, success 42% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial, minus ladder: registration + words | +0.023 (-0.078 to +0.123) | 117 trials, success 42% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named, said it recognised the trial | 0.801 (0.665 to 0.918) | 47 trials, success 53% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it recognised the trial, minus logistic: registration | +0.144 (-0.040 to +0.317) | 47 trials, success 53% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it recognised the trial, minus ladder: registration + words | -0.008 (-0.142 to +0.144) | 47 trials, success 53% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial | 0.869 (0.764 to 0.949) | 58 trials, success 53% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial, minus logistic: registration | +0.170 (+0.024 to +0.325) | 58 trials, success 53% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial, minus ladder: registration + words | +0.142 (+0.025 to +0.264) | 58 trials, success 53% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named, said it recognised the trial | 0.806 (0.697 to 0.896) | 78 trials, success 49% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it recognised the trial, minus logistic: registration | +0.075 (-0.059 to +0.208) | 78 trials, success 49% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it recognised the trial, minus ladder: registration + words | -0.013 (-0.114 to +0.092) | 78 trials, success 49% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named by neither model | 0.770 (0.688 to 0.844) | 141 trials, success 43% |
| E4 | blinded haiku, guess or reason names it, drug named by neither model, minus logistic: registration | +0.090 (-0.009 to +0.193) | 141 trials, success 43% |
| E4 | blinded haiku, guess or reason names it, drug named by neither model, minus ladder: registration + words | +0.005 (-0.076 to +0.088) | 141 trials, success 43% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named by neither model | 0.758 (0.673 to 0.835) | 141 trials, success 43% |
| E4 | blinded sonnet, guess or reason names it, drug named by neither model, minus logistic: registration | +0.078 (-0.026 to +0.183) | 141 trials, success 43% |
| E4 | blinded sonnet, guess or reason names it, drug named by neither model, minus ladder: registration + words | -0.007 (-0.097 to +0.081) | 141 trials, success 43% |

## E5 (exploratory): The seven pre-registered predictions

Not computed here. The seven pre-registered predictions are scored by `uv run python -m elman_exercise.scorecard test`, a separate command in the same session.
