# REHEARSAL on development data. This is NOT the final run.

**Every number below is from development trials.** Dev trials that finished 2017 to 2020 stand in for the test set, and every model was trained on dev trials that finished before 2017. The 300 sampled development trials stand in for the sampled test and late trials of the language models. There is no stand-in for the late set (S5). The held-out labels were not opened. This runs the same code as the final run, to check the code, not to say anything about the plan.

Git commit `b4621c5d26f3a2eb8624b35ccd36091912ee31c8` (with uncommitted changes).

Trained 53 models on 3,945 dev trials (finishing before 2017) in 28 seconds. Scored:

- development trials finishing 2017 to 2020: 1,904 trials, observed success 53.6%.

Differences are first-named minus second-named. AUROC differences are paired: taken within each bootstrap resample. Calibration gap is mean predicted minus observed success, in points (positive means over-prediction). Intervals are 95% percentile intervals from 1,000 resamples of trials, except the adjusted differences of E1 and E3, which use 2,000 resamples of lead drugs; trials that share a drug or sponsor are not independent, so intervals are somewhat too narrow. The expectation for each item is in FINAL_PLAN.md, written before this run.

## R1 (reference): Does the registry's registration model reproduce the pre-registered one?

Not checked here: the figures to match are those of the real test and late sets.

## P1 (primary): Does the best honest model built here beat the pre-registered baseline?

### Development trials finishing 2017 to 2020 (stand-in for the test set)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| P1 | ladder: registration + endpoint + history + genetics + words minus logistic: registration | +0.037 (+0.023 to +0.050) | AUROC, 1,904 trials, success 53.6% |
| P1 | AUROC of ladder: registration + endpoint + history + genetics + words | 0.750 (0.728 to 0.769) | 1,904 trials, success 53.6% |
| P1 | Brier score of ladder: registration + endpoint + history + genetics + words (lower is better) | 0.204 (no interval) | 1,904 trials, success 53.6% |
| P1 | calibration gap of ladder: registration + endpoint + history + genetics + words: mean predicted minus observed | +2.5 points (+0.6 to +4.4) | 1,904 trials, success 53.6% |
| P1 | AUROC of logistic: registration | 0.713 (0.692 to 0.735) | 1,904 trials, success 53.6% |
| P1 | Brier score of logistic: registration (lower is better) | 0.218 (no interval) | 1,904 trials, success 53.6% |
| P1 | calibration gap of logistic: registration: mean predicted minus observed | +4.7 points (+2.7 to +6.7) | 1,904 trials, success 53.6% |

## S1 (secondary): How does every other registered model do, scored once?

### Honest models

| Model | AUROC (95% interval) | AUROC minus registration | Brier | Gap (points) |
|---|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.750 (0.728 to 0.769) | +0.037 (+0.023 to +0.050) | 0.204 | +2.5 |
| biology: ladder, all four tables + science | 0.747 (0.725 to 0.767) | +0.034 (+0.019 to +0.048) | 0.205 | +2.2 |
| ladder: registration + words | 0.740 (0.718 to 0.761) | +0.027 (+0.014 to +0.040) | 0.209 | +4.1 |
| ladder: boosted trees, endpoint + history + genetics | 0.740 (0.717 to 0.760) | +0.027 (+0.014 to +0.038) | 0.208 | +3.4 |
| embeddings: biomedical vector plus registration plus words (uses a pretrained model) | 0.733 (0.713 to 0.756) | +0.020 (-0.000 to +0.041) | 0.213 | +3.4 |
| ladder: registration + endpoint | 0.727 (0.705 to 0.747) | +0.013 (+0.006 to +0.021) | 0.212 | +2.7 |
| embeddings: biomedical vector plus registration (uses a pretrained model) | 0.725 (0.703 to 0.747) | +0.012 (-0.008 to +0.033) | 0.215 | +3.6 |
| embeddings: general vector plus registration plus words (uses a pretrained model) | 0.725 (0.703 to 0.747) | +0.012 (-0.009 to +0.033) | 0.215 | +4.1 |
| boosted trees: registration, tuned | 0.724 (0.701 to 0.745) | +0.011 (-0.000 to +0.021) | 0.214 | +4.9 |
| random forest: registration | 0.724 (0.702 to 0.745) | +0.011 (-0.000 to +0.021) | 0.214 | +4.9 |
| neural network: registration | 0.723 (0.701 to 0.744) | +0.010 (+0.003 to +0.016) | 0.215 | +5.3 |
| boosted trees: registration | 0.722 (0.700 to 0.744) | +0.009 (-0.000 to +0.018) | 0.215 | +4.7 |
| logistic: separate by phase | 0.718 (0.697 to 0.740) | +0.005 (-0.002 to +0.013) | 0.216 | +5.0 |
| logistic: registration + interactions | 0.717 (0.696 to 0.739) | +0.004 (-0.002 to +0.010) | 0.217 | +4.8 |
| ladder: registration + genetics | 0.716 (0.695 to 0.739) | +0.003 (-0.000 to +0.007) | 0.217 | +4.9 |
| logistic: separate for company and other sponsors | 0.716 (0.694 to 0.738) | +0.003 (-0.005 to +0.010) | 0.217 | +4.8 |
| logistic: registration + pooled sponsor | 0.715 (0.694 to 0.737) | +0.002 (-0.001 to +0.006) | 0.218 | +4.8 |
| logistic: registration, tuned penalty | 0.714 (0.693 to 0.736) | +0.001 (-0.003 to +0.005) | 0.218 | +4.7 |
| ladder: registration + history | 0.714 (0.692 to 0.735) | +0.001 (-0.003 to +0.005) | 0.218 | +4.5 |
| biology: registration + expression | 0.713 (0.692 to 0.735) | +0.000 (-0.005 to +0.005) | 0.217 | +4.4 |
| logistic: registration | 0.713 (0.692 to 0.735) | (baseline) | 0.218 | +4.7 |
| logistic: trend-shifted intercept | 0.713 (0.692 to 0.735) | +0.000 (+0.000 to +0.000) | 0.215 | -0.4 |
| biology: registration + annotations | 0.712 (0.692 to 0.733) | -0.001 (-0.004 to +0.002) | 0.218 | +4.6 |
| biology: registration + DepMap | 0.712 (0.691 to 0.734) | -0.001 (-0.004 to +0.003) | 0.219 | +4.8 |
| text: words without drug names | 0.712 (0.689 to 0.734) | -0.001 (-0.025 to +0.022) | 0.218 | +5.5 |
| text: words | 0.712 (0.689 to 0.734) | -0.001 (-0.026 to +0.022) | 0.219 | +5.7 |
| logistic: recency-weighted | 0.712 (0.691 to 0.734) | -0.001 (-0.004 to +0.002) | 0.218 | +4.2 |
| biology: registration + networks | 0.711 (0.690 to 0.733) | -0.002 (-0.005 to +0.001) | 0.218 | +4.6 |
| biology: registration + all sources | 0.710 (0.689 to 0.731) | -0.003 (-0.012 to +0.006) | 0.219 | +4.3 |
| logistic: last 4 years only | 0.710 (0.689 to 0.731) | -0.003 (-0.010 to +0.003) | 0.219 | +4.4 |
| logistic: last 8 years only | 0.709 (0.688 to 0.731) | -0.004 (-0.007 to -0.000) | 0.219 | +4.5 |
| biology: registration + chemistry | 0.709 (0.688 to 0.730) | -0.004 (-0.010 to +0.001) | 0.219 | +4.6 |
| embeddings: general vector plus registration (uses a pretrained model) | 0.706 (0.684 to 0.728) | -0.007 (-0.026 to +0.014) | 0.221 | +4.9 |
| embeddings: biomedical vector alone (uses a pretrained model) | 0.697 (0.673 to 0.719) | -0.016 (-0.040 to +0.006) | 0.224 | +5.2 |
| embeddings: general vector alone (uses a pretrained model) | 0.677 (0.654 to 0.700) | -0.037 (-0.061 to -0.012) | 0.230 | +6.0 |
| lookup: phase x sponsor class | 0.669 (0.648 to 0.691) | -0.044 (-0.060 to -0.026) | 0.228 | +6.4 |
| memory: lead sponsor | 0.618 (0.594 to 0.643) | -0.095 (-0.121 to -0.068) | 0.246 | +8.3 |
| lookup: phase | 0.614 (0.593 to 0.634) | -0.100 (-0.121 to -0.078) | 0.239 | +5.8 |
| text: drug names only | 0.585 (0.560 to 0.610) | -0.128 (-0.158 to -0.099) | 0.247 | +5.2 |
| lookup: constant | 0.500 (0.500 to 0.500) | -0.213 (-0.235 to -0.192) | 0.254 | +7.2 |

### Not honest (deliberately leaky)

| Model | AUROC (95% interval) | AUROC minus registration | Brier | Gap (points) |
|---|---|---|---|---|
| ladder: all four tables plus after-the-start columns (not honest) | 0.755 (0.734 to 0.774) | +0.042 (+0.028 to +0.055) | 0.202 | +2.5 |
| leaky: registration + post-hoc fields | 0.739 (0.717 to 0.759) | +0.026 (+0.013 to +0.039) | 0.211 | +4.6 |
| text: words with summary (leaky) | 0.723 (0.700 to 0.744) | +0.010 (-0.014 to +0.032) | 0.217 | +6.5 |
| leaky: registration + results structure | 0.715 (0.693 to 0.737) | +0.002 (-0.005 to +0.009) | 0.217 | +4.0 |
| memory: lead drug | 0.620 (0.596 to 0.644) | -0.093 (-0.122 to -0.063) | 0.245 | +8.6 |

Not ranked here (not meaningful on all trials: they predict one constant where a trial has no mapped target; scored on mapped trials in `results/dev_science.md`): biology: logistic, all sources, biology: boosted trees, all sources, biology: logistic, DepMap only, biology: logistic, expression only, biology: logistic, networks only, biology: logistic, annotations only, biology: logistic, chemistry only.

Skipped (scored on the sampled trials only: see E4): llm: haiku (identified), llm: haiku (masked), llm: haiku (blinded), llm: sonnet (identified), llm: sonnet (masked), llm: sonnet (blinded).

## S2 (secondary): What does each table of information add to registration details?

### Development trials finishing 2017 to 2020 (stand-in for the test set)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S2 | + endpoint minus logistic: registration | +0.013 (+0.006 to +0.021) | AUROC, 1,904 trials, success 53.6% |
| S2 | + history minus logistic: registration | +0.001 (-0.003 to +0.005) | AUROC, 1,904 trials, success 53.6% |
| S2 | + genetics minus logistic: registration | +0.003 (-0.000 to +0.007) | AUROC, 1,904 trials, success 53.6% |
| S2 | + words minus logistic: registration | +0.027 (+0.014 to +0.040) | AUROC, 1,904 trials, success 53.6% |
| S2 | all four tables minus logistic: registration | +0.037 (+0.023 to +0.050) | AUROC, 1,904 trials, success 53.6% |

## S3 (secondary): Does the trend-shifted intercept reduce the over-prediction?

This is an arithmetic check, not a test. The trend-shifted model moves only the intercept of the registration model, so its mean prediction sits a fixed distance below the registration model's (that distance needs no labels) and its gap is the registration model's gap plus that shift, whatever the labels say.

### Development trials finishing 2017 to 2020 (stand-in for the test set)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S3 | logistic: registration: mean predicted minus observed | +4.7 points (+2.7 to +6.7) | observed success 53.6% |
| S3 | logistic: trend-shifted intercept: mean predicted minus observed | -0.4 points (-2.4 to +1.6) | observed success 53.6% |
| S3 | logistic: trend-shifted intercept minus logistic: registration: mean prediction, no labels needed | -5.1 points (no interval) | 1,904 trials, success 53.6% |
| S3 | logistic: trend-shifted intercept minus logistic: registration | +0.000 (+0.000 to +0.000) | AUROC, 1,904 trials, success 53.6% |

## S4 (secondary): Do the public science sources help, and does the positive control show?

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| S4 | biology: registration + all sources minus logistic: registration, mapped trials | -0.002 (-0.015 to +0.010) | AUROC, 1,320 trials, success 56.7% |
| S4 | control: registration + genetic support minus logistic: registration, mapped trials | +0.003 (-0.000 to +0.007) | AUROC, 1,320 trials, success 56.7% |
| S4 | biology: ladder, all four tables + science minus ladder: registration + endpoint + history + genetics + words, mapped trials | -0.000 (-0.010 to +0.009) | AUROC, 1,320 trials, success 56.7% |
| S4 | biology: registration + all sources minus logistic: registration, all trials | -0.003 (-0.012 to +0.006) | AUROC, 1,904 trials, success 53.6% |
| S4 | control: registration + genetic support minus logistic: registration, all trials | +0.003 (+0.000 to +0.005) | AUROC, 1,904 trials, success 53.6% |

## S5 (secondary): Do the primary and ladder comparisons hold on the late set?

Not run in the rehearsal: there is no stand-in for the late set. The code path is exercised by the synthetic test (`tests/test_final.py`).

## E1 (exploratory): Do biomarker endpoints succeed more, and survival endpoints less?

### Development trials finishing 2017 to 2020 (stand-in for the test set)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E1 | biomarker first primary endpoint minus all other endpoint types, crude | +18.4 points (+12.9 to +23.6) | 362 against 1,542; success 68.5% against 50.1% |
| E1 | biomarker first primary endpoint minus all other endpoint types, adjusted | +16.0 points (+9.4 to +23.0) | 362 against 1,542; success 68.5% against 50.1%; adjusted for phase, sponsor type, area, drug type |
| E1 | survival first primary endpoint minus all other endpoint types, crude | -11.0 points (-17.1 to -4.7) | 287 against 1,617; success 44.3% against 55.2% |
| E1 | survival first primary endpoint minus all other endpoint types, adjusted | -10.6 points (-18.3 to -2.3) | 287 against 1,617; success 44.3% against 55.2%; adjusted for phase, sponsor type, area, drug type |

## E2 (exploratory): Do Phase 3 trials succeed more after a successful Phase 2?

### Development trials finishing 2017 to 2020 (stand-in for the test set)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E2 | Phase 3 after a successful Phase 2 minus after a failed Phase 2 | +64.3 points (+32.5 to +83.9) | 53 against 10; success 94.3% against 30.0% |

## E3 (exploratory): Do single-target drugs with a broadly expressed target succeed less?

### Development trials finishing 2017 to 2020 (stand-in for the test set)

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E3 | expr_n_tissues, top third minus bottom third, single-target drugs, adjusted | -15.4 points (-29.0 to -2.7) | 258 against 265; success 45.3% against 70.9%; top third is more than 45 tissues, bottom third is 14 or fewer |

## E4 (exploratory): How do the language models do, and how much of it is recall?

### Sampled development trials

| Item | Comparison | Estimate (95% interval) | Trials and notes |
|---|---|---|---|
| E4 | AUROC of haiku (identified) | 0.877 (0.838 to 0.914) | 300 sampled trials |
| E4 | AUROC of haiku (masked) | 0.860 (0.816 to 0.903) | 300 sampled trials |
| E4 | AUROC of haiku (blinded) | 0.799 (0.751 to 0.848) | 300 sampled trials |
| E4 | AUROC of sonnet (identified) | 0.916 (0.881 to 0.946) | 300 sampled trials |
| E4 | AUROC of sonnet (masked) | 0.907 (0.872 to 0.940) | 300 sampled trials |
| E4 | AUROC of sonnet (blinded) | 0.822 (0.773 to 0.868) | 300 sampled trials |
| E4 | AUROC of logistic: registration | 0.716 (0.660 to 0.767) | 300 sampled trials |
| E4 | AUROC of ladder: registration + words | 0.749 (0.693 to 0.798) | 300 sampled trials |
| E4 | haiku (identified) minus haiku (blinded) | +0.078 (+0.036 to +0.120) | 300 sampled trials |
| E4 | haiku (masked) minus haiku (blinded) | +0.061 (+0.022 to +0.101) | 300 sampled trials |
| E4 | haiku (identified) minus haiku (masked) | +0.017 (-0.017 to +0.049) | 300 sampled trials |
| E4 | sonnet (identified) minus sonnet (blinded) | +0.094 (+0.056 to +0.134) | 300 sampled trials |
| E4 | sonnet (masked) minus sonnet (blinded) | +0.085 (+0.048 to +0.123) | 300 sampled trials |
| E4 | sonnet (identified) minus sonnet (masked) | +0.009 (-0.009 to +0.028) | 300 sampled trials |
| E4 | haiku (identified) minus logistic: registration | +0.160 (+0.104 to +0.219) | 300 sampled trials |
| E4 | haiku (identified) minus ladder: registration + words | +0.128 (+0.074 to +0.185) | 300 sampled trials |
| E4 | haiku (masked) minus logistic: registration | +0.144 (+0.087 to +0.206) | 300 sampled trials |
| E4 | haiku (masked) minus ladder: registration + words | +0.111 (+0.057 to +0.170) | 300 sampled trials |
| E4 | haiku (blinded) minus logistic: registration | +0.082 (+0.026 to +0.142) | 300 sampled trials |
| E4 | haiku (blinded) minus ladder: registration + words | +0.050 (-0.004 to +0.109) | 300 sampled trials |
| E4 | sonnet (identified) minus logistic: registration | +0.199 (+0.145 to +0.256) | 300 sampled trials |
| E4 | sonnet (identified) minus ladder: registration + words | +0.167 (+0.115 to +0.223) | 300 sampled trials |
| E4 | sonnet (masked) minus logistic: registration | +0.191 (+0.133 to +0.246) | 300 sampled trials |
| E4 | sonnet (masked) minus ladder: registration + words | +0.158 (+0.107 to +0.213) | 300 sampled trials |
| E4 | sonnet (blinded) minus logistic: registration | +0.106 (+0.051 to +0.165) | 300 sampled trials |
| E4 | sonnet (blinded) minus ladder: registration + words | +0.073 (+0.024 to +0.126) | 300 sampled trials |
| E4 | AUROC of blinded haiku, guess names the drug, drug not named | 0.755 (0.689 to 0.819) | 209 trials, success 52% |
| E4 | blinded haiku, guess names the drug, drug not named, minus logistic: registration | +0.062 (-0.013 to +0.138) | 209 trials, success 52% |
| E4 | blinded haiku, guess names the drug, drug not named, minus ladder: registration + words | +0.007 (-0.063 to +0.072) | 209 trials, success 52% |
| E4 | AUROC of blinded haiku, guess names the drug, drug named | 0.880 (0.798 to 0.944) | 91 trials, success 56% |
| E4 | blinded haiku, guess names the drug, drug named, minus logistic: registration | +0.117 (+0.031 to +0.207) | 91 trials, success 56% |
| E4 | blinded haiku, guess names the drug, drug named, minus ladder: registration + words | +0.126 (+0.035 to +0.216) | 91 trials, success 56% |
| E4 | AUROC of blinded sonnet, guess names the drug, drug not named | 0.805 (0.742 to 0.867) | 165 trials, success 51% |
| E4 | blinded sonnet, guess names the drug, drug not named, minus logistic: registration | +0.119 (+0.047 to +0.209) | 165 trials, success 51% |
| E4 | blinded sonnet, guess names the drug, drug not named, minus ladder: registration + words | +0.064 (-0.006 to +0.139) | 165 trials, success 51% |
| E4 | AUROC of blinded sonnet, guess names the drug, drug named | 0.843 (0.774 to 0.902) | 135 trials, success 56% |
| E4 | blinded sonnet, guess names the drug, drug named, minus logistic: registration | +0.096 (+0.013 to +0.175) | 135 trials, success 56% |
| E4 | blinded sonnet, guess names the drug, drug named, minus ladder: registration + words | +0.089 (+0.012 to +0.164) | 135 trials, success 56% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named | 0.756 (0.682 to 0.817) | 204 trials, success 52% |
| E4 | blinded haiku, guess or reason names it, drug not named, minus logistic: registration | +0.060 (-0.015 to +0.137) | 204 trials, success 52% |
| E4 | blinded haiku, guess or reason names it, drug not named, minus ladder: registration + words | +0.006 (-0.064 to +0.079) | 204 trials, success 52% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named | 0.877 (0.799 to 0.939) | 96 trials, success 54% |
| E4 | blinded haiku, guess or reason names it, drug named, minus logistic: registration | +0.122 (+0.037 to +0.208) | 96 trials, success 54% |
| E4 | blinded haiku, guess or reason names it, drug named, minus ladder: registration + words | +0.127 (+0.044 to +0.223) | 96 trials, success 54% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named | 0.807 (0.738 to 0.873) | 150 trials, success 49% |
| E4 | blinded sonnet, guess or reason names it, drug not named, minus logistic: registration | +0.134 (+0.051 to +0.219) | 150 trials, success 49% |
| E4 | blinded sonnet, guess or reason names it, drug not named, minus ladder: registration + words | +0.081 (+0.007 to +0.157) | 150 trials, success 49% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named | 0.838 (0.770 to 0.900) | 150 trials, success 57% |
| E4 | blinded sonnet, guess or reason names it, drug named, minus logistic: registration | +0.089 (+0.008 to +0.174) | 150 trials, success 57% |
| E4 | blinded sonnet, guess or reason names it, drug named, minus ladder: registration + words | +0.072 (-0.001 to +0.144) | 150 trials, success 57% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial | 0.759 (0.687 to 0.819) | 197 trials, success 52% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial, minus logistic: registration | +0.057 (-0.014 to +0.129) | 197 trials, success 52% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it did not recognise the trial, minus ladder: registration + words | +0.002 (-0.064 to +0.068) | 197 trials, success 52% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug not named, said it recognised the trial | n/a (n/a to n/a) | too few to score; 7 trials, success 57% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it recognised the trial, minus logistic: registration | n/a (n/a to n/a) | too few to score; 7 trials, success 57% |
| E4 | blinded haiku, guess or reason names it, drug not named, said it recognised the trial, minus ladder: registration + words | n/a (n/a to n/a) | too few to score; 7 trials, success 57% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named, said it did not recognise the trial | 0.766 (0.639 to 0.875) | 64 trials, success 48% |
| E4 | blinded haiku, guess or reason names it, drug named, said it did not recognise the trial, minus logistic: registration | +0.017 (-0.097 to +0.133) | 64 trials, success 48% |
| E4 | blinded haiku, guess or reason names it, drug named, said it did not recognise the trial, minus ladder: registration + words | +0.011 (-0.103 to +0.122) | 64 trials, success 48% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named, said it recognised the trial | 1.000 (1.000 to 1.000) | 32 trials, success 66% |
| E4 | blinded haiku, guess or reason names it, drug named, said it recognised the trial, minus logistic: registration | +0.182 (+0.057 to +0.345) | 32 trials, success 66% |
| E4 | blinded haiku, guess or reason names it, drug named, said it recognised the trial, minus ladder: registration + words | +0.212 (+0.067 to +0.401) | 32 trials, success 66% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial | 0.799 (0.705 to 0.881) | 111 trials, success 45% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial, minus logistic: registration | +0.107 (+0.009 to +0.213) | 111 trials, success 45% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it did not recognise the trial, minus ladder: registration + words | +0.032 (-0.049 to +0.115) | 111 trials, success 45% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug not named, said it recognised the trial | 0.787 (0.607 to 0.936) | 39 trials, success 62% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it recognised the trial, minus logistic: registration | +0.237 (+0.060 to +0.418) | 39 trials, success 62% |
| E4 | blinded sonnet, guess or reason names it, drug not named, said it recognised the trial, minus ladder: registration + words | +0.235 (+0.056 to +0.407) | 39 trials, success 62% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial | 0.758 (0.628 to 0.871) | 60 trials, success 52% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial, minus logistic: registration | +0.085 (-0.037 to +0.207) | 60 trials, success 52% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it did not recognise the trial, minus ladder: registration + words | +0.056 (-0.064 to +0.173) | 60 trials, success 52% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named, said it recognised the trial | 0.885 (0.805 to 0.945) | 90 trials, success 60% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it recognised the trial, minus logistic: registration | +0.074 (-0.038 to +0.178) | 90 trials, success 60% |
| E4 | blinded sonnet, guess or reason names it, drug named, said it recognised the trial, minus ladder: registration + words | +0.065 (-0.031 to +0.154) | 90 trials, success 60% |
| E4 | AUROC of blinded haiku, guess or reason names it, drug named by neither model | 0.746 (0.653 to 0.819) | 131 trials, success 50% |
| E4 | blinded haiku, guess or reason names it, drug named by neither model, minus logistic: registration | +0.080 (-0.016 to +0.175) | 131 trials, success 50% |
| E4 | blinded haiku, guess or reason names it, drug named by neither model, minus ladder: registration + words | +0.003 (-0.080 to +0.083) | 131 trials, success 50% |
| E4 | AUROC of blinded sonnet, guess or reason names it, drug named by neither model | 0.799 (0.712 to 0.866) | 131 trials, success 50% |
| E4 | blinded sonnet, guess or reason names it, drug named by neither model, minus logistic: registration | +0.133 (+0.039 to +0.224) | 131 trials, success 50% |
| E4 | blinded sonnet, guess or reason names it, drug named by neither model, minus ladder: registration + words | +0.056 (-0.021 to +0.130) | 131 trials, success 50% |

## E5 (exploratory): The seven pre-registered predictions

Not computed here. The seven pre-registered predictions are scored by `uv run python -m trial_benchmark.scorecard test`, a separate command in the same session.

## Check against results/dev_models.md

45 models compared. The largest difference is 0.000. `dev_models.md` scores the round trained before 2017 on trials finishing 2017 to 2018, with the same training trials as this rehearsal, so agreement to the third decimal is expected.

| Model | dev_models.md | this rehearsal | Difference |
|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.771 | 0.771 | -0.000 |
| biology: ladder, all four tables + science | 0.763 | 0.763 | -0.000 |
| ladder: registration + words | 0.761 | 0.761 | -0.000 |
| embeddings: biomedical vector plus registration plus words | 0.755 | 0.755 | +0.000 |
| ladder: boosted trees, endpoint + history + genetics | 0.747 | 0.747 | +0.000 |
| embeddings: biomedical vector plus registration | 0.745 | 0.745 | +0.000 |
| embeddings: general vector plus registration plus words | 0.758 | 0.758 | -0.000 |
| boosted trees: registration, tuned | 0.729 | 0.729 | +0.000 |
| ladder: registration + endpoint | 0.742 | 0.742 | -0.000 |
| boosted trees: registration | 0.728 | 0.728 | -0.000 |
| text: words | 0.737 | 0.737 | +0.000 |
| neural network: registration | 0.733 | 0.733 | -0.000 |
| random forest: registration | 0.730 | 0.730 | +0.000 |
| text: words without drug names | 0.736 | 0.736 | -0.000 |
| logistic: registration + interactions | 0.729 | 0.729 | -0.000 |
| logistic: registration + pooled sponsor | 0.728 | 0.728 | +0.000 |
| logistic: separate for company and other sponsors | 0.725 | 0.725 | -0.000 |
| embeddings: biomedical vector alone | 0.718 | 0.718 | -0.000 |
| ladder: registration + genetics | 0.732 | 0.732 | -0.000 |
| logistic: separate by phase | 0.728 | 0.728 | +0.000 |
| ladder: registration + history | 0.727 | 0.727 | -0.000 |
| biology: registration + expression | 0.725 | 0.725 | -0.000 |
| logistic: recency-weighted | 0.727 | 0.727 | -0.000 |
| biology: registration + networks | 0.723 | 0.723 | -0.000 |
| logistic: registration | 0.727 | 0.727 | -0.000 |
| logistic: trend-shifted intercept | 0.727 | 0.727 | -0.000 |
| logistic: registration, tuned penalty | 0.729 | 0.729 | -0.000 |
| biology: registration + DepMap | 0.724 | 0.724 | -0.000 |
| embeddings: general vector plus registration | 0.740 | 0.740 | -0.000 |
| biology: registration + all sources | 0.719 | 0.719 | -0.000 |
| logistic: last 4 years only | 0.726 | 0.726 | -0.000 |
| logistic: last 8 years only | 0.723 | 0.723 | +0.000 |
| biology: registration + annotations | 0.726 | 0.726 | -0.000 |
| biology: registration + chemistry | 0.722 | 0.722 | -0.000 |
| embeddings: general vector alone | 0.708 | 0.708 | -0.000 |
| lookup: phase x sponsor class | 0.679 | 0.679 | +0.000 |
| memory: lead sponsor | 0.637 | 0.637 | +0.000 |
| lookup: phase | 0.626 | 0.626 | +0.000 |
| text: drug names only | 0.598 | 0.598 | +0.000 |
| lookup: constant | 0.500 | 0.500 | +0.000 |
| ladder: all four tables plus after-the-start columns (not honest) | 0.773 | 0.773 | +0.000 |
| leaky: registration + post-hoc fields | 0.746 | 0.746 | +0.000 |
| text: words with summary (leaky) | 0.747 | 0.747 | +0.000 |
| leaky: registration + results structure | 0.736 | 0.736 | +0.000 |
| memory: lead drug | 0.660 | 0.660 | -0.000 |
