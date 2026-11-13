# REHEARSAL on development data. This is NOT the test set.

**Every number below is from development trials.** Dev trials that finished 2017 to 2020 stand in for the test set, and the models were trained on dev trials that finished before 2017. The held-out labels were not opened. It runs the same code as the real scorecard, to check the code, not to say anything about the predictions.

Git commit `1e0522039f4ce1ee01faa48ac0a052ad9ad63c0b` (with uncommitted changes).

Trained on 3,945 dev trials; scored 1,904 trials finishing from 2017-01-01 up to 2021-01-01. Recruitment-stop population: 14,492 finished drug trials in the same period.

**How each row is judged.** The observed value is compared, unrounded, with the predicted range (both ends count as inside, unless the range says "below"). *Inside the predicted range*: the value is in it. *Outside, but the interval overlaps the range*: the value is out, but the 95% interval reaches into the range. *Outside*: the value and the whole interval are out. Where the email gives no range the row is reported without a verdict. Prediction 2b is not judged by this rule; it has its own, stated under it.

Intervals: bootstrap over trials for AUROCs and the over-prediction (as in results/dev_baselines.md); Wilson for a rate; Newcombe for a difference of rates; sponsor-resampled for the adjusted recruitment ratio (as in results/dev_failures.md). All but the last treat trials as independent, which they are not, so they are somewhat too narrow.

## The predictions

| # | Prediction | Predicted | Observed (95% interval) | Verdict | Notes |
|---|---|---|---|---|---|
| 1a | Only 45 to 50% of test trials succeed (down from 58% in training) | 45.0% to 50.0% | 53.6% (51.3% to 55.8%) | outside | 1,904 trials |
| 1b | The registration model over-predicts by 5 to 9 per 100 (mean predicted minus observed) | +5.0 to +9.0 points | +4.7 points (+2.7 to +6.7) | outside, but the interval overlaps the range | mean predicted 58.3% |
| 2a | Registration model AUROC is 0.66 to 0.72 | 0.660 to 0.720 | 0.713 (0.692 to 0.735) | inside the predicted range |  |
| 3 | Leaked fields raise AUROC by 0.02 to 0.05 (leaky minus registration) | +0.020 to +0.050 | +0.026 (+0.013 to +0.039) | inside the predicted range |  |
| 4a | Biologics succeed a little more than small molecules, a gap like 63% vs 58% (biologic minus small molecule) | no range stated | +5.1 points (-0.3 to +10.5) | no range in the email | 413 biologic against 1,412 small molecule; the interval does not exclude zero |
| 4b | Telling the model the drug type barely improves AUROC, by less than 0.01 (registration minus registration without modality) | below +0.010 | -0.001 (-0.005 to +0.002) | inside the predicted range | the model did no better with modality |
| 5a | Active-comparator trials succeed 3 to 10 per 100 less than placebo-controlled ones (active minus placebo) | -10.0 to -3.0 points | -8.4 points (-15.0 to -1.7) | inside the predicted range | 246 active against 1,598 placebo |
| 5b | Trials under 100 participants succeed 18 to 28 per 100 less than those with 300 or more (under 100 minus 300 or more) | -28.0 to -18.0 points | -23.6 points (-28.9 to -18.1) | inside the predicted range | 524 under 100 against 762 300 or more |
| 6a | Genetic support adds 5 to 13 successes per 100 trials, among trials mapped to a target (supported minus not supported) | +5.0 to +13.0 points | +14.6 points (+9.1 to +20.0) | outside, but the interval overlaps the range | 461 supported against 832 not supported |
| 6b | Weaker for drugs with 5 or fewer earlier trials than for those with more (supported minus not supported, within each) | no range stated | +1.9 points (-10.5 to +13.9) | no range in the email | 91 supported against 176 not supported; point estimate lower than the +17.9 for drugs with more than 5 earlier trials (370 supported against 656 not supported; interval +11.7 to +23.8); the intervals overlap |
| 7a | Outside oncology, rare-disease trials stop for recruitment about 1.2 times as often (rare over not rare, adjusted for phase, sponsor type and era) | no range stated | 1.18 (0.92 to 1.50) | no range in the email | email: about 1.2; training-set figure 1.19 (1.00 to 1.40). All finished trials in the period: 1,440 rare, 5,672 not rare, outside oncology; crude ratio 1.18 (0.91 to 1.51) |
| 7b | Among trials that report a result, rare-disease trials succeed 8 to 18 per 100 less (rare minus not rare) | -18.0 to -8.0 points | -11.1 points (-17.5 to -4.5) | inside the predicted range | 276 rare-disease against 1,114 not rare |

## Prediction 2b: which columns matter most

**2b. Predicted: phase and sponsor_class matter most. Observed: phase and sponsor_class. Matches.** Rule for this row only: the two named columns must be the two biggest losses, and both must be positive. The second loss is not clearly above the third (paired interval of the difference includes zero).

_Same method as results/dev_baselines.md section D: refit the registration model without one column, score the same trials, and rank the columns by the AUROC lost._

| Rank | Column dropped | AUROC without it | AUROC lost (95% interval) |
|---|---|---|---|
| 1 | phase | 0.668 | +0.045 (0.029 to 0.061) |
| 2 | sponsor_class | 0.693 | +0.020 (0.012 to 0.028) |
| 3 | therapeutic_area | 0.699 | +0.014 (0.004 to 0.024) |
| 4 | n_registered_arms | 0.700 | +0.013 (0.002 to 0.021) |
| 5 | intervention_model | 0.710 | +0.003 (-0.001 to 0.006) |
| 6 | allocation | 0.713 | +0.000 (-0.001 to 0.002) |
| 7 | is_vaccine | 0.713 | +0.000 (-0.000 to 0.000) |
| 8 | masking | 0.713 | -0.000 (-0.003 to 0.003) |
| 9 | modality | 0.713 | -0.000 (-0.004 to 0.003) |
| 10 | healthy_volunteers | 0.714 | -0.000 (-0.002 to 0.001) |
| 11 | has_dmc | 0.714 | -0.001 (-0.007 to 0.005) |
