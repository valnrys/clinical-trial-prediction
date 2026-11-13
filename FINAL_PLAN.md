# Final run: what will be computed

_Written and committed before the final run, so it is timestamped before any result is seen.
Anything not listed here is not computed on the test or late labels. The run refuses if this file
is uncommitted, changed since its commit, or still has an unfinished-section marker in it._

## What was already seen

The pre-registered baselines were scored on the test and late sets once, through the guarded
labels file (`results/test_baselines.md`, one logged access). The project lead saw these:

- **Test set (1,236 trials).** Success rate 50.6%. Registration model AUROC 0.730 (0.702 to
  0.759), over-predicting success by 4.6 points (2.3 to 7.3). The leaky model 0.779; modality added
  nothing (-0.000). The constant, phase-lookup and results-structure models were also scored
  (results structure 0.740). Accuracy at the 0.5 threshold was reported. The registration model's
  AUROC, success rate and calibration within subsets by control kind, enrolment band and modality.
- **Late set (492 trials).** Success rate 47.8%. Registration AUROC 0.733. Calibration gap +7.6
  points (3.7 to 11.5).
- **Test and late together (1,728 trials).** Registration AUROC 0.731.

Every modelling choice since was made on development validation only (trials finishing 2017 to
2020, trained on earlier ones). The tables, models and analyses below were fixed before the
final labels were read, except the trend-shifted model (S3), which was built after the +4.6 was
seen. Nothing else is computed on test labels in this run. The seven pre-registered predictions
are scored by `scorecard test` as a separate command (E5).

## How it is computed

Every model in the registry at the commit of the run is trained on all 5,849 development trials,
predicts the unlabelled test (1,236) and late (492) trials, and only then are the labels read,
once, for both. Differences are first-named minus second-named. AUROC differences are paired
bootstrap: 1,000 resamples of trials, seed 20261110, 95% percentile intervals. Intervals treat
trials as independent, which they are not (drugs and sponsors recur), so they are somewhat too
narrow. The adjusted differences (E1, E3) resample lead drugs instead (2,000 resamples) and are
adjusted for phase, sponsor type, therapeutic area and drug type. "Development" figures are the
three-round mean in `results/dev_models.md` unless stated; the run itself is one round, so expect
small differences. Every table reports the trial count and success rate of each set or subset.
Before the labels are opened the whole plan runs once on made-up labels, so a bug cannot surface
after the opening.

## REFERENCE CHECK

### R1. Does the registry's registration model reproduce the pre-registered one?

"logistic: registration" in the registry gives predictions identical to the pre-registered
registration model, so this run must print AUROC **0.730 on the test set and 0.733 on the late
set**, to three decimals. If it does not, something has drifted and nothing else in the report
should be trusted. The run marks R1 as failed in that case.

## PRIMARY

### P1. Does the best honest model built here beat the pre-registered baseline?

"ladder: registration + endpoint + history + genetics + words" minus "logistic: registration",
AUROC, on the 1,236 test trials, both trained on all development trials. The model is named here,
not chosen after seeing results. Also reported: AUROC, Brier score and calibration gap of both
models, and the same gain on the test and late sets together (1,728 trials).

**Decision rule.** "Beats the baseline" means the lower end of the 95% interval is above zero.

**Development: +0.036 (+0.025 to +0.048)** (0.745 against 0.709); by round +0.027, +0.044,
+0.038; grouped cross-validation +0.032. **Expect less on the test set, roughly +0.02 to +0.035.**
The primary model was the top honest model on development validation, so its development gain
carries some selection optimism. The registration baseline also scored higher on the test set
(0.730) than on development validation (0.709), which leaves less room.

## SECONDARY

### S1. Every other registered model, once, in one table

Honest and not honest apart; ranked and not ranked as the registry says; pretrained models
flagged. AUROC with interval, minus registration, Brier score, calibration gap. No model is
chosen after seeing this table: the registry at the run commit is the frozen list. Development:
`results/dev_models.md` (honest models span 0.500 to 0.745).

### S2. The information ladder, rung by rung (each minus registration, AUROC)

On the test set and on the test and late sets together. **Development:** + endpoint +0.010
(+0.003 to +0.016); + history +0.002 (-0.001 to +0.005); + genetics +0.003 (-0.001 to +0.006);
+ words +0.030 (+0.020 to +0.040); all four tables +0.036 (+0.025 to +0.048). Expected: words and
endpoint carry the gain; history and genetics add little.

### S3. Calibration: an arithmetic check, not a test

Its answer is already fixed by numbers that have been seen, so it is reported as arithmetic.
Trained on all development trials, the trend-shifted model's mean prediction is **10.3 points
below** the registration model's on both the test set (44.9% against 55.3%) and the late set
(computed on the unlabelled features). The registration model's gap is +4.6 on the test set
(seen), so the trend-shifted model's gap must be about **-5.7: it over-corrects**, and misses by
more than the model it was meant to fix. On the late set (registration +7.6, seen) the arithmetic
gives about **-2.7**. AUROC is identical (it only moves the intercept).

History, stated plainly: the trend-shifted model was built on development data after the +4.6 on
the test set had been seen. On development validation it reduced the gap from about +3.8 to about
-1.2. Projecting the fall in the success rate forward over-shoots, because the fall did not
continue at that rate. Reported on the test and late sets: both models' gaps, the shift, and the
AUROC difference.

### S4. Public science sources, and a positive control

AUROC difference, test set. **Development**, trials with a mapped target (764 of the test
trials): "biology: registration + all sources" minus registration -0.001 (-0.010 to +0.008); positive
control "control: registration + genetic support" minus registration +0.003 (-0.000 to +0.006);
"biology: ladder, all four tables + science" minus the ladder without science -0.003 (-0.010 to
+0.004). On all trials (the first two only): sources -0.002 (-0.008 to +0.005); control +0.002
(-0.000 to +0.004). Expected: sources add nothing; the control adds a little. The control runs
through the same code path, so a test that cannot see it cannot see a source either.

### S5. The primary and ladder comparisons on the late set (492 trials, 2024 to 2025)

P1 and S2 again on the late set alone, with Brier score and calibration gap. No development
figure; intervals will be wider than on the test set. Same direction expected as P1 and S2.

## EXPLORATORY

**Direction only; no claim of significance; several of these were noticed on development data
among many comparisons.** Each is reported on the test set and on the late set, with counts and
success rates for every group.

### E1. Endpoint type

Biomarker first primary endpoints succeed more than all other types; survival endpoints less.
Crude (Newcombe interval) and adjusted, in points. **Development:** biomarker minus rest crude
+18.5 (+15.7 to +21.3), adjusted +13.3 (+9.4 to +17.2); survival minus rest crude -15.7 (-19.4 to
-11.9), adjusted -13.6 (-18.3 to -8.8). Counts (features only), biomarker and survival: test 232
and 195; late 106 and 63.

### E2. Phase 3 after a labelled Phase 2 of the same drug and disease

Success after a successful Phase 2 minus success after a failed one, Phase 3 trials, in points.
**Development:** 89.6% of 77 against 47.6% of 21, +42.0 (+20.1 to +61.9). Expected positive.
Counts (features only), after a success and after a failure: test 51 and 12; late 27 and 6. The
intervals will be wide.

### E3. Single-target drugs with a broadly expressed target

`expr_n_tissues`, top third minus bottom third, single-target drugs, adjusted. The definition and
code are those of the candidate section of `science/report.py`; cut points are fixed on the
development trials (bottom third 14 or fewer tissues, top third more than 46). **Development:**
-10.0 (-17.1 to -3.3) points. Expected negative. **This is the one pattern, out of the science
features examined, that stood out after the fact.** Counts (features only), top and bottom: test
124 and 169; late 58 and 63.

### E4. Language models (Claude Haiku and Sonnet; identified, masked and blinded prompts)

AUROC of each model and prompt on the 300 sampled test and the 300 sampled late trials,
separately, against "logistic: registration" and "ladder: registration + words" on the same
trials (trained on all development trials). Contrasts: identified minus blinded, masked minus
blinded, identified minus masked; the blinded prompt on trials the model did and did not name
(both definitions of "named" in `llm_report.py`); the four-cell table (named by the stricter
definition, crossed with the model's own "I recognise this trial"); and the row for trials
neither model named. **Development (300 trials):** identified 0.916 (Sonnet) and 0.877 (Haiku);
masked 0.907 and 0.860; blinded 0.822 and 0.799; registration 0.716; ladder with words 0.749.
**Every period is inside these models' training data**, so a high score cannot be told from
recall, and the identified prompt least of all. They are pretrained models and are flagged as such.

### E5. The seven pre-registered predictions

Not computed here. Scored by `scorecard test`, a separate command.

## Order of the run

1. Commit everything, this plan included, so the tree is clean.
2. `uv run python -m trial_benchmark.final check`. It opens nothing. It runs every label-free step
   and the dress rehearsal on made-up labels.
3. `uv run python -m trial_benchmark.final test --note "..."`. One logged opening, for both sets.
   It writes `results/test_final.md`.
4. Commit `results/test_final.md` and `evals/heldout_log.jsonl`. Otherwise the next opening's log
   line says the code was dirty.
5. `uv run python -m trial_benchmark.scorecard test --note "..."` (E5). Commit its report.

## Not computed

No other analysis, subset, model or threshold is computed on test or late labels in this run.
