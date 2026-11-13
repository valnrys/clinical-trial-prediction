# The hierarchical model on the development trials

Development trials only (completed before 2021). No test or late trial is used.

A Bayesian logistic model of trial success: the registration columns plus partially pooled effects for therapeutic area, lead sponsor, condition, target and lead drug (`models/hierarchical.py`). A group that has few trials is pulled towards the average, so it learns from what is common and cannot just memorise. Every score is out of sample: each fit is made inside its round and sees only trials completed before it.

Where the targets and precedent flags came from: today links, repo; 2020 links, repo ("repo" is `science.links` and `precedent`). "2020" dates only the drug-to-target step (Open Targets release 20.06, mid 2020); the trial-to-drug match, lead drug and MeSH condition are today's. The 2020 genetics columns are rebuilt from the 2020 investigational targets, dated to each trial's start year.

## AUROC gain over the registration logistic (2795 validation trials)

The registration logistic scores 0.709 (95% interval, drugs resampled, 0.686 to 0.731). The table is the paired gain over it: mean AUROC within each of the three time-ordered rounds, each variant fitted inside each round on trials completed before it. A model's prediction is the posterior mean of its probability of success. The drug-clustered interval resamples whole lead drugs (a drug with trials in two rounds is in or out of both) and is the one to trust; the last column resamples single trials as if they were independent, for comparison.

| variant | links | AUROC | gain | 95% interval, drugs resampled | 95% interval, trials resampled |
|---|---|---|---|---|---|
| registration only | today | 0.710 | +0.001 | -0.000 to +0.003 | -0.000 to +0.003 |
| + target | today | 0.723 | +0.014 | +0.006 to +0.022 | +0.006 to +0.022 |
| + condition | today | 0.722 | +0.014 | +0.004 to +0.024 | +0.005 to +0.022 |
| + sponsor | today | 0.710 | +0.002 | -0.006 to +0.009 | -0.006 to +0.009 |
| full hierarchy | today | 0.734 | +0.025 | +0.014 to +0.038 | +0.015 to +0.036 |
| full + precedent + genetics | today | 0.741 | +0.032 | +0.020 to +0.045 | +0.021 to +0.043 |
| registration only | 2020 | 0.710 | +0.001 | -0.000 to +0.003 | -0.000 to +0.003 |
| + target | 2020 | 0.718 | +0.009 | +0.001 to +0.018 | +0.001 to +0.016 |
| + condition | 2020 | 0.722 | +0.014 | +0.004 to +0.024 | +0.005 to +0.022 |
| + sponsor | 2020 | 0.710 | +0.002 | -0.006 to +0.009 | -0.006 to +0.009 |
| full hierarchy | 2020 | 0.732 | +0.024 | +0.012 to +0.036 | +0.013 to +0.035 |
| full + precedent + genetics | 2020 | 0.737 | +0.028 | +0.016 to +0.041 | +0.018 to +0.039 |

## How much each group matters

The full hierarchy fitted on all development trials. Each group has one spread (SD, in log-odds). "Per 100" turns it into successes per 100 trials between a group one SD below average and one SD above it, for a trial at the overall success rate (58.4 per 100). Read the drug and target rows together, not apart: a drug and its targets overlap, so the model can give credit to either. For targets the per-100 figure is the trial-level one (the SD is first scaled for averaging over a trial's targets).

- today links: 27% of trials have no target and get no target effect; a trial with several targets gets the average of their effects, so its effect varies by 0.84 times the SD in the table.
- 2020 links: 35% of trials have no target and get no target effect; a trial with several targets gets the average of their effects, so its effect varies by 0.84 times the SD in the table.

| group | sd, log-odds (today) | per 100 (today) | sd, log-odds (2020) | per 100 (2020) |
|---|---|---|---|---|
| target | 0.77 (0.55 to 1.00) | 30.2 (21.9 to 38.7) | 0.65 (0.41 to 0.89) | 25.6 (16.4 to 34.7) |
| condition | 0.73 (0.59 to 0.89) | 34.2 (27.8 to 40.9) | 0.73 (0.58 to 0.90) | 34.2 (27.6 to 41.3) |
| lead sponsor | 0.56 (0.31 to 0.81) | 26.3 (14.8 to 37.6) | 0.56 (0.33 to 0.82) | 26.5 (15.9 to 38.0) |
| lead drug | 0.58 (0.37 to 0.77) | 27.3 (17.6 to 35.8) | 0.64 (0.45 to 0.83) | 30.3 (21.7 to 38.4) |
| therapeutic area | 0.28 (0.10 to 0.54) | 13.6 (5.0 to 25.5) | 0.29 (0.12 to 0.54) | 14.0 (5.6 to 25.8) |

## Precedent and genetics

From the full hierarchy plus precedent and genetics, fitted on all development trials. "Per 100" is, for each posterior draw, the average over the development trials of the change in predicted probability of success when the column goes from no to yes (or, for the genetic score, up one SD), group effects included, in sample, times 100. These coefficients have a tight prior (SD 0.2 in log-odds), so a small number is partly the prior's doing.

| column | links | unit | per 100 (95% interval) | odds ratio |
|---|---|---|---|---|
| genetic_score_at_start | today | +1 SD | +2.1 (+0.1 to +3.9) | 1.13 (1.00 to 1.27) |
| supported_at_start | today | yes vs no | +0.9 (-5.2 to +7.1) | 1.08 (0.73 to 1.53) |
| precedent | today | yes vs no | +7.7 (+4.9 to +10.2) | 1.58 (1.34 to 1.83) |
| genetic_score_at_start | 2020 | +1 SD | +1.6 (-0.2 to +3.4) | 1.10 (0.99 to 1.22) |
| supported_at_start | 2020 | yes vs no | +0.9 (-5.1 to +6.9) | 1.08 (0.74 to 1.52) |
| precedent | 2020 | yes vs no | +7.6 (+4.6 to +10.3) | 1.57 (1.32 to 1.83) |

For comparison, the plain registration logistic with the same precedent flag added, fitted on all development trials and read the same way (average change in predicted probability, no to yes, in sample, times 100; no interval):

| links | per 100 |
|---|---|
| today | +11.3 |
| 2020 | +10.9 |

## Named examples

Largest effects among levels with at least 10 training trials, full hierarchy, today links, all development trials. Each is the change in successes per 100 trials for an otherwise average trial, as a posterior mean with its 95% interval. Sparse levels are shrunk, but these are still the largest of the levels shown (118 conditions, 399 targets), picked by size, so they overstate what a fresh trial of that kind would show; most intervals include zero (109 of 118 conditions, 399 of 399 targets).

### Conditions (first MeSH term): success higher than average

| name | training trials | per 100 (95% interval) |
|---|---|---|
| Hypercholesterolemia | 56 | +25.3 (+9.6 to +35.8) |
| Arthritis, Psoriatic | 41 | +22.9 (+7.4 to +34.3) |
| Peanut Hypersensitivity | 11 | +21.5 (+1.0 to +35.0) |
| Hot Flashes | 15 | +21.3 (+0.4 to +35.4) |
| Angioedemas, Hereditary | 15 | +21.1 (+0.3 to +34.9) |

### Conditions (first MeSH term): success lower than average

| name | training trials | per 100 (95% interval) |
|---|---|---|
| Depressive Disorder, Major | 93 | -29.7 (-41.0 to -17.0) |
| Alzheimer Disease | 68 | -23.4 (-37.1 to -8.1) |
| Anemia, Sickle Cell | 18 | -23.1 (-43.4 to +0.6) |
| Neoplasm Metastasis | 14 | -21.2 (-41.1 to +1.4) |
| Stomach Neoplasms | 19 | -20.2 (-38.8 to +1.1) |

### Targets: success higher than average

| name | training trials | per 100 (95% interval) |
|---|---|---|
| PGR | 29 | +18.6 (-2.0 to +32.7) |
| JAK2 | 71 | +18.3 (-9.5 to +36.0) |
| GUCY2C | 11 | +16.3 (-9.4 to +34.3) |
| CALCRL | 20 | +16.3 (-7.3 to +32.9) |
| IL4R | 21 | +16.3 (-8.0 to +33.7) |

### Targets: success lower than average

| name | training trials | per 100 (95% interval) |
|---|---|---|
| PRKCB | 15 | -20.4 (-44.7 to +7.7) |
| IL13 | 24 | -19.7 (-40.3 to +3.2) |
| EPOR | 14 | -18.7 (-40.5 to +6.4) |
| INSR | 26 | -18.2 (-38.7 to +4.6) |
| PDE9A | 13 | -17.9 (-43.6 to +10.4) |

## Sampler checks

Four chains per fit. Divergences should be 0; the worst R-hat (over every parameter) is acceptable below 1.05 and clean below 1.01; the smallest effective sample size (over every parameter) should be in the hundreds. Over 48 fits the worst R-hat is 1.030; 13 fits are above 1.01 and 0 are at or above 1.05. Divergences in total: 0.

| links | variant | split | divergences | worst R-hat | smallest effective n |
|---|---|---|---|---|---|
| today | registration only | round0 | 0 | 1.003 | 1608 |
| today | registration only | round1 | 0 | 1.002 | 1509 |
| today | registration only | round2 | 0 | 1.002 | 1503 |
| today | registration only | full | 0 | 1.004 | 1302 |
| today | + target | round0 | 0 | 1.004 | 1260 |
| today | + target | round1 | 0 | 1.003 | 1368 |
| today | + target | round2 | 0 | 1.002 | 1375 |
| today | + target | full | 0 | 1.003 | 1251 |
| today | + condition | round0 | 0 | 1.001 | 1308 |
| today | + condition | round1 | 0 | 1.003 | 1256 |
| today | + condition | round2 | 0 | 1.008 | 1156 |
| today | + condition | full | 0 | 1.007 | 966 |
| today | + sponsor | round0 | 0 | 1.005 | 771 |
| today | + sponsor | round1 | 0 | 1.013 | 567 |
| today | + sponsor | round2 | 0 | 1.004 | 701 |
| today | + sponsor | full | 0 | 1.003 | 739 |
| today | full hierarchy | round0 | 0 | 1.027 | 216 |
| today | full hierarchy | round1 | 0 | 1.007 | 244 |
| today | full hierarchy | round2 | 0 | 1.012 | 326 |
| today | full hierarchy | full | 0 | 1.017 | 322 |
| today | full + precedent + genetics | round0 | 0 | 1.014 | 328 |
| today | full + precedent + genetics | round1 | 0 | 1.022 | 238 |
| today | full + precedent + genetics | round2 | 0 | 1.015 | 195 |
| today | full + precedent + genetics | full | 0 | 1.009 | 326 |
| 2020 | registration only | round0 | 0 | 1.003 | 1608 |
| 2020 | registration only | round1 | 0 | 1.002 | 1509 |
| 2020 | registration only | round2 | 0 | 1.002 | 1503 |
| 2020 | registration only | full | 0 | 1.004 | 1302 |
| 2020 | + target | round0 | 0 | 1.003 | 1443 |
| 2020 | + target | round1 | 0 | 1.004 | 1487 |
| 2020 | + target | round2 | 0 | 1.001 | 1577 |
| 2020 | + target | full | 0 | 1.001 | 1459 |
| 2020 | + condition | round0 | 0 | 1.001 | 1308 |
| 2020 | + condition | round1 | 0 | 1.003 | 1256 |
| 2020 | + condition | round2 | 0 | 1.008 | 1156 |
| 2020 | + condition | full | 0 | 1.007 | 966 |
| 2020 | + sponsor | round0 | 0 | 1.005 | 771 |
| 2020 | + sponsor | round1 | 0 | 1.013 | 567 |
| 2020 | + sponsor | round2 | 0 | 1.004 | 701 |
| 2020 | + sponsor | full | 0 | 1.003 | 739 |
| 2020 | full hierarchy | round0 | 0 | 1.019 | 242 |
| 2020 | full hierarchy | round1 | 0 | 1.016 | 211 |
| 2020 | full hierarchy | round2 | 0 | 1.007 | 351 |
| 2020 | full hierarchy | full | 0 | 1.014 | 380 |
| 2020 | full + precedent + genetics | round0 | 0 | 1.019 | 234 |
| 2020 | full + precedent + genetics | round1 | 0 | 1.030 | 234 |
| 2020 | full + precedent + genetics | round2 | 0 | 1.009 | 364 |
| 2020 | full + precedent + genetics | full | 0 | 1.010 | 308 |
