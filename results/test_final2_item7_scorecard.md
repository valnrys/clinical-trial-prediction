# Earlier results regenerated with lead-drug intervals (TEST_PLAN_2.md item 7)

The first run's code and models: the model families added since (precedent, hierarchical, whathow) are left out, so every row is a row of the first run. Only the bootstrap differs: each resample draws whole lead drugs instead of single trials.

**How each row is judged.** The observed value is compared, unrounded, with the predicted range (both ends count as inside, unless the range says "below"). *Inside the predicted range*: the value is in it. *Outside, but the interval overlaps the range*: the value is out, but the 95% interval reaches into the range. *Outside*: the value and the whole interval are out. Where the email gives no range the row is reported without a verdict. Prediction 2b is not judged by this rule; it has its own, stated under it.

Intervals: bootstrap over lead drugs for AUROCs and the over-prediction (as in results/dev_baselines.md); Wilson for a rate; Newcombe for a difference of rates; sponsor-resampled for the adjusted recruitment ratio (as in results/dev_failures.md). The Wilson and Newcombe intervals treat trials as independent, which they are not, so they are somewhat too narrow.

## The predictions

| # | Prediction | Predicted | Observed (95% interval) | Verdict | Notes |
|---|---|---|---|---|---|
| 1a | Only 45 to 50% of test trials succeed (down from 58% in training) | 45.0% to 50.0% | 50.6% (47.9% to 53.4%) | outside, but the interval overlaps the range | 1,236 trials |
| 1b | The registration model over-predicts by 5 to 9 per 100 (mean predicted minus observed) | +5.0 to +9.0 points | +4.6 points (+1.9 to +7.3) | outside, but the interval overlaps the range | mean predicted 55.3% |
| 2a | Registration model AUROC is 0.66 to 0.72 | 0.660 to 0.720 | 0.730 (0.698 to 0.761) | outside, but the interval overlaps the range |  |
| 3 | Leaked fields raise AUROC by 0.02 to 0.05 (leaky minus registration) | +0.020 to +0.050 | +0.048 (+0.030 to +0.067) | inside the predicted range |  |
| 4a | Biologics succeed a little more than small molecules, a gap like 63% vs 58% (biologic minus small molecule) | no range stated | +4.5 points (-2.2 to +11.1) | no range in the email | 282 biologic against 891 small molecule; the interval does not exclude zero |
| 4b | Telling the model the drug type barely improves AUROC, by less than 0.01 (registration minus registration without modality) | below +0.010 | -0.000 (-0.004 to +0.004) | inside the predicted range | the model did no better with modality |
| 5a | Active-comparator trials succeed 3 to 10 per 100 less than placebo-controlled ones (active minus placebo) | -10.0 to -3.0 points | -0.1 points (-8.6 to +8.3) | outside, but the interval overlaps the range | 150 active against 1,053 placebo |
| 5b | Trials under 100 participants succeed 18 to 28 per 100 less than those with 300 or more (under 100 minus 300 or more) | -28.0 to -18.0 points | -34.3 points (-40.5 to -27.6) | outside, but the interval overlaps the range | 346 under 100 against 485 300 or more |
| 6a | Genetic support adds 5 to 13 successes per 100 trials, among trials mapped to a target (supported minus not supported) | +5.0 to +13.0 points | +17.3 points (+9.8 to +24.4) | outside, but the interval overlaps the range | 258 supported against 468 not supported |
| 6b | Weaker for drugs with 5 or fewer earlier trials than for those with more (supported minus not supported, within each) | no range stated | +2.9 points (-16.5 to +21.4) | no range in the email | 37 supported against 69 not supported; point estimate lower than the +19.7 for drugs with more than 5 earlier trials (221 supported against 399 not supported; interval +11.6 to +27.4); the intervals overlap |
| 7a | Outside oncology, rare-disease trials stop for recruitment about 1.2 times as often (rare over not rare, adjusted for phase, sponsor type and era) | no range stated | 1.00 (0.70 to 1.33) | no range in the email | email: about 1.2; training-set figure 1.19 (1.00 to 1.40). All finished trials in the period: 1,114 rare, 4,677 not rare, outside oncology; crude ratio 0.96 (0.73 to 1.26) |
| 7b | Among trials that report a result, rare-disease trials succeed 8 to 18 per 100 less (rare minus not rare) | -18.0 to -8.0 points | -0.7 points (-8.8 to +7.5) | outside, but the interval overlaps the range | 180 rare-disease against 670 not rare |

## Prediction 2b: which columns matter most

**2b. Predicted: phase and sponsor_class matter most. Observed: phase and sponsor_class. Matches.** Rule for this row only: the two named columns must be the two biggest losses, and both must be positive. The second loss is not clearly above the third (paired interval of the difference includes zero).

_Same method as results/dev_baselines.md section D: refit the registration model without one column, score the same trials, and rank the columns by the AUROC lost._

| Rank | Column dropped | AUROC without it | AUROC lost (95% interval) |
|---|---|---|---|
| 1 | phase | 0.673 | +0.057 (0.036 to 0.078) |
| 2 | sponsor_class | 0.711 | +0.020 (0.009 to 0.031) |
| 3 | therapeutic_area | 0.717 | +0.013 (0.000 to 0.025) |
| 4 | n_registered_arms | 0.719 | +0.012 (0.000 to 0.023) |
| 5 | has_dmc | 0.726 | +0.004 (-0.003 to 0.012) |
| 6 | intervention_model | 0.729 | +0.002 (-0.002 to 0.006) |
| 7 | healthy_volunteers | 0.729 | +0.001 (0.000 to 0.002) |
| 8 | is_vaccine | 0.730 | +0.000 (-0.001 to 0.001) |
| 9 | modality | 0.730 | -0.000 (-0.004 to 0.004) |
| 10 | allocation | 0.731 | -0.000 (-0.003 to 0.003) |
| 11 | masking | 0.734 | -0.004 (-0.007 to 0.000) |
