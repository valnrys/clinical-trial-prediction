# Second test-set plan

Written 12 November 2026, before any of the models below has been scored on the
test set (trials finished 2021 to 2023). It adds to `FINAL_PLAN.md`; it does not
replace it.

**Rule.** Every model here is defined and fitted on development data only. The
run happens once, from a committed state. No model, feature or definition is
changed afterwards in response to what the test set shows. The guard logs the
read as before.

**What I already know about the test set.** The baseline's results (success
50.6%, AUROC 0.730), the first final run and the scorecard, all in `results/`.
The models below were built knowing those.

## What is scored, and what I expect

| # | What is scored | Expectation |
|---|---|---|
| 1 | Hierarchical model (partially pooled effects for target, condition, sponsor, drug, area) against the registration baseline | about +0.03 AUROC |
| 2 | Registration plus the precedent flag, for three link versions (today's; targets dated to June 2020; drug names and targets dated). Headline: the strict flag (experimental arms only) on the fully dated version | strict version +0.005 to +0.01 |
| 3 | HOW (registration without therapeutic area), WHAT (disease, target, precedent, genetics, biology), and both | both beats HOW by +0.015 to +0.03; WHAT alone about 0.66 |
| 4 | Phase 2 success by third of the WHAT score | top third beats bottom third by 15 to 25 per 100 trials |
| 5 | Split study with the test set as the "later" holdout, against random holdouts of the same size | a random split flatters the baseline by about 0.03 AUROC and hides the over-prediction |
| 6 | Per-trial predictions saved for the baseline and the models above | none; needed to draw development and test curves side by side |
| 7 | Intervals for the earlier results regenerated with resampling by lead drug | wider; no verdict changes |
| 8 | A biology-only score, within phase: (a) a count of six risk flags fixed in advance (target expressed in nearly every tissue; high expression in vital organs; gene essential to cells; intracellular target; drug activates its target; low drug-likeness), trials with two or more flags against trials with none; (b) a logistic model on biology columns only, top third against bottom third. No phase, sponsor, disease area, design, precedent, genetics or earlier outcome goes in. Thresholds are the ones fixed on trials before 2015; the logistic model is refitted on all development trials by the declared recipe | (a) 2 to 10 fewer successes per 100 with two or more flags; (b) 5 to 15 more in the top third. On development trials from 2015 to 2020 these were 8.1 fewer (14.6 to 1.7) and 16.6 more (10 to 23), and I expect them to shrink again. On trials whose targets appear in no development trial: same direction, too few to say more |

## How each item is judged

Written before the run, so that nothing is settled afterwards.

- **The rule.** A difference counts as met when its 95% interval lies wholly on
  the expected side of zero. Intervals resample lead drugs. Where an expectation
  gives a range, the result is also reported as inside or outside it. Where it
  says "about", only the interval rule applies.
- **Link versions.** Three are scored and reported everywhere: today's; targets
  dated to June 2020; drug names and targets dated. The headline for items 2, 3
  and 4 is the fully dated version. The headline for item 1 is the full
  hierarchy on today's links, the version it was developed on; the hierarchy is
  not fitted on the fully dated version. The headline for item 8 is today's
  links, as its recipe declared, with the other two as sensitivity rows. Today's
  links can carry hindsight; the dated rows are there to show how much.
- **Item 3.** "Both beats HOW" is the paired difference between HOW + WHAT and
  HOW. The difference from the full registration baseline is reported beside it.
- **Item 4.** Thirds are cut within the scored trials and within phase. Phase 3
  and the share of the gap that survives within phase, sponsor type and third
  of the HOW score are reported too, without a verdict.
- **Item 5.** "Flatters" is the random-split AUROC minus the later-trials AUROC
  for the baseline, random by trial, with an interval that combines the spread
  across the 20 random splits and the sampling of trials. "Hides the over-prediction" is met if the over-prediction on
  later trials is above zero and smaller under a random split. The random
  holdouts mix test trials into training by design; nothing from them is used in
  any other item.
- **Item 7.** "Wider" is met if more intervals widen than narrow. A verdict
  change is any interval, of any kind, that excluded zero and now includes it,
  or the reverse; every one is listed. Only rows already in `results/test_final.md` and
  `results/test_scorecard.md` are compared; models added since are not part of
  it. Regenerating the scorecard reads the late set's labels (2024 to 2025) as
  well as the test set's, as the first run did; it is one logged read.
- **Item 8.** "Within phase" means the difference is computed inside each phase
  and pooled, so phase cannot explain it; the two phases are also shown apart.
  The score uses no phase, sponsor, disease-area label, design, precedent,
  genetics or earlier outcome. Four of its inputs describe the target in the
  trial's own disease tissue or cancer lineage, and its chemistry inputs exist
  only for small molecules, so it partly reflects the kind of disease and drug.
- **One run.** The script refuses to run a second time. If it fails part-way,
  the failure and any re-run are reported as such.

Item 8 judges biology on its own terms: a difference in success rate within a
phase, in successes per 100 trials, not a gain in AUROC over the registration
baseline. The baseline contains phase, and reaching Phase 3 is itself the result
of earlier success, so "does not beat the baseline" and "says nothing about the
chance of success" are different claims. The six flags were chosen after seeing
all development data, 2015 to 2020 included, so the test set is their first
clean test. The low drug-likeness flag did nothing on development data and stays
in as declared.

## Not scored on the test set

Protein language model embeddings, the graph neural network and link-prediction
models, the Bayesian network. They added nothing on development data, and the
report says so.

## Known weaknesses going in

- Drug-to-target links and drug names are dated to June 2020, which is after
  many development trials started. The disease mapping and the registry record
  itself are today's.
- The precedent finding shrank from about 24 to about 15 successes per 100 as
  links were dated; it may shrink further on the test set.
- Reweighting towards all finished trials lowers the baseline's development
  AUROC from 0.709 to about 0.67.
