# Beliefs, limits and decisions

A working record, kept as we go. The pre-registration is distilled from the
beliefs; the report's interpretation leans on the limits.

**How to amend:** edit any entry in place. Each entry is a short block with
the same fields, so add, reword or delete freely. When a belief changes, keep
the old line and add the new one under the next stage, so the movement stays
visible. Number new entries on from the last one.

**Stages of a belief**

- **Before data**: written before either of us has seen any success rate.
- **After development set**: after looking at pre-2021 outcomes.
- **After test set**: after the pre-registration is sent and the test labels are opened.

_Status on 11 Nov 2026: the test labels have been opened (three logged reads); "After test set" lines are filled in where the scorecard or final run gives a number. Entries B20 to B27 were added after the final run. "Before" and "after development" lines for B1 to B19 were not backfilled here (see BELIEFS-CURRENT.md for the development numbers)._

_Status on 10 Nov 2026: John confirmed the "before data" beliefs below as our joint starting position. The development set was first looked at after that._

---

## Decisions so far

| # | Decision | Why |
|---|---|---|
| D1 | Trials: interventional Phase 2 or 3, with a drug, biological or genetic intervention, completed or terminated, with a p-value posted on a primary outcome (10,945 trials), narrowed by D3a to those with a treatment-versus-control comparison (7,593). | The only population where "met its primary endpoint" can be read from the record by rule. |
| D2 | Split by primary completion date. Development: before 2021. **Primary test set: 2021 to 2023** (1,236 eligible trials). Extended test set: 2021 to 2025 (1,728). | Mirrors real use. 2024 to 2025 are thin and skewed to fast reporters, so they are reported separately, not in the headline. |
| D3 | Two labels: (a) met primary endpoint, binary; (b) strength of evidence, a z-score, where an estimate or p-value allows. | Raw effect sizes are in each trial's own units and do not compare. |
| D3a | Only treatment-versus-control comparisons count: two groups, exactly one a control, not a non-inferiority or equivalence test, and the treatment not an established reference drug, on the first primary outcome that has such a comparison. | A read of 40 multi-comparison trials showed that "first-listed", "any comparison" and "all comparisons" each misread common layouts (lowest dose first; safety endpoints; dose-versus-dose rows). Replaces the earlier first / any / all plan. Decided before any outcome rate was seen. |
| D3b | **Headline label: any treatment arm beats control.** Strict variant: all arms beat control. Check: the highest dose beats control, on trials whose arms form a clean dose ladder. | "Any arm" is defined for every eligible trial and is how a dose-ranging trial is usually read. Its known bias is that more arms give more chances. "Max dose" is one comparison per trial with no multiplicity, but is defined for only 1,056 trials. |
| D4 | The reference configuration is primary: any-arm label, all modalities, 2021 to 2023, modality-agnostic model. | Named in advance so the headline is not chosen after the fact. |
| D5 | Variants are measured as differences from D4, one axis at a time: label (all arms; max dose), population (modality groups, extended test set), model (modality as a feature). | About eight numbers, not a grid. |
| D6 | Modality groups: small molecule, biologic, advanced (cell, gene, oligonucleotide). Vaccines get their own flag. | Vaccine trials mostly test immune response in healthy volunteers, a different kind of success. |
| D7 | Modality is assigned from the experimental arm's drug, by name stem, using only the trial's own record. Trials with mixed modalities in the experimental arm are counted and kept as their own group for now. | Keeps "information beyond the trial's record" a clean no. Mixed trials are about 5% of trials with a primary p-value (502 of 10,945); decide later whether to exclude. |
| D8 | No pretrained model in the labels or the baseline. | Keeps the first one-line answer a clean no. Models did read pre-2021 samples to audit the labelling rule (see below); the report will say so. |
| D9 | Trials against an active comparator stay in where the registry marks which arm is the comparator. Each trial's control is tagged placebo, active, no treatment or unspecified. | Beating an established drug is a harder test than beating placebo, so the two are reported as subsets. |
| D10 | A biomarker primary endpoint counts. The label is "met its own pre-specified primary endpoint". | Separating biomarker from clinical endpoints needs a judgement on every outcome title. Stated as a limit (L3). |
| D11 | Add-on designs with no placebo arm (drug plus background versus background alone) are mostly left out. | Inferring them from the arms' intervention lists was wrong about as often as right on the audited sample. |
| D12 | The report leads with three subsets: placebo versus active control, trial size, and modality. Oncology is not one of them. | Each has a mechanism that can be stated in a sentence. Oncology is only 14% of eligible trials, because the label needs a control arm and a p-value and so drops single-arm and many add-on designs; what is left is not representative of oncology. |
| D13 | For each treatment arm, only its first-listed comparison with control counts. | 463 pre-2021 trials test an arm against control more than once on the chosen outcome. A read of 15 found the repeats are sensitivity analyses, other timepoints, other populations, and in one case a baseline balance test. Counting any of them gives an arm several chances, which is the "any comparison" rule D3a rejected. Confirmed by John, 10 Nov. |
| D14 | A trial with both a placebo and an active comparator is labelled against placebo. More generally, one control group per trial: the placebo if there is one, otherwise the first control listed. | About 50 eligible pre-2021 trials mix kinds of control. Without this, the label means "beat placebo or beat the active drug", and the trial fits neither side of the placebo-versus-active subset. Confirmed by John, 10 Nov. |
| D15 | The strength-of-evidence label (D3b) is parked. | The binary label, three subsets and the baseline ladder fill two pages. If built later, it should use the same single comparison that decides the binary label, not the smallest p-value across arms, which is biased by selection. Confirmed by John, 10 Nov. |
| D16 | The pre-registration carries the forecast that the model over-predicts success on the test set by 8 to 12 points (a restatement of B11) and that modality adds under 0.01 AUROC (B13). The forecast about ranking on the 2024 to 2025 set (B12) stays in this log only. | With 492 trials the ranking score there is uncertain by about ±0.05, so "about the same" could not be confirmed either way. Agreed with John, 10 Nov. |

### Open decisions

- Trials with a control comparison on more than one primary outcome (about a fifth): held at the first eligible outcome for now.
- Whether trials stopped for futility, which rarely post a p-value, are added as failures from the "why stopped" text.
- Which registration-time fields the baseline may use, and which are treated as leaky.
- Whether mixed-modality trials stay in.
- After the final run (11 Nov): the owner's placeholders in REPORT.md (date, active hours, assistant account); regenerating results from a clean commit; whether to add a remote and push the `prereg` tag; overnight work on training data only (no test reads), asking which patterns could be told to someone starting a programme.

---

## Counts we are working from

Structure counts only, from exploration scripts run against the control rule
in `src/elman_exercise/comparisons.py`. The dataset build will regenerate them
from committed code, and they may shift slightly.

| | Development (<2021) | Test 2021–23 | 2024–25 | Total |
|---|---|---|---|---|
| Any p-value on a primary outcome | 8,619 | 1,657 | 669 | 10,945 |
| **Eligible: a treatment-versus-control comparison on a primary** | 5,860 | 1,239 | 494 | 7,593 |
| ...one treatment arm | 4,077 | 932 | 368 | 5,377 |
| ...two or more treatment arms | 1,783 | 307 | 126 | 2,216 |
| ......of which a clean dose ladder | 848 | 141 | 67 | 1,056 |

From the label code (`labels.py`): 7,577 of the 7,593 get a determined headline
label (5,849 development, 1,236 primary test, 492 in 2024 to 2025); 16 are
undetermined because the p-value does not settle it. With one control per
trial (D14): placebo 6,383, active comparator 981, unspecified 143, no
treatment 70. Clean dose ladders: 1,058.

Before D14, eligible trials by kind of control: placebo 6,336; active comparator 982;
unspecified (recognised only by a phrase such as "control") 143; no treatment
70; more than one kind 62. Per comparison, the control is recognised by a
placebo title in 83%, by registered arm type in 15%, and by a control phrase in 2%.

Modality groups, counted on the rebuilt dataset (7,577 trials; feature counts
only). "Vaccine" is the modality of trials whose chosen interventions are all
vaccines; the vaccine flag is also set on 4 biologic or small-molecule trials
that give a vaccine alongside another drug. Placebo-like interventions
(placebo, vehicle, sham, saline, dummy) are ignored. Trials with nothing but
those, or only a genetic test, have no modality.

| Group | Development (<2021) | Test 2021–23 | 2024–25 | Total |
|---|---|---|---|---|
| Small molecule | 4,638 | 891 | 344 | 5,873 |
| Biologic | 993 | 282 | 113 | 1,388 |
| Advanced (cell, gene, oligonucleotide) | 49 | 24 | 16 | 89 |
| Vaccine only | 40 | 15 | 4 | 59 |
| Mixed in the experimental arm | 108 | 18 | 13 | 139 |
| No modality | 21 | 6 | 2 | 29 |

Vaccine flag set: 44 development, 17 test 2021–23, 4 in 2024 to 2025.
Inside "advanced", test 2021–23: 1 cell, 3 gene, 20 oligonucleotide.

### How the control rule was checked (pre-2021 trials only, p-values removed)

Models audited the rule; they do not produce labels. Haiku read 60 trials from
each of four groups (240 in all): three kinds of trial an early version of the
rule rejected, and one sample it accepted. Sonnet independently read the first
20 of each group (80), and the two agreed on 75 of those 80. A reviewer model
also hand-read 25 trials accepted by arm type and 25 by control phrase.

The rule was then changed in response, so these samples are no longer an
independent test of it. A fresh hand-read sample is still owed.

Against the current rule:

| Reader's verdict | Trials | Current rule |
|---|---|---|
| Accepted by the early rule, and correct | 55 | 53 still accepted |
| Accepted by the early rule, but comparing populations | 1 | Now rejected |
| Accepted, primary outcome a biomarker or feasibility measure | 4 | Still accepted (see D10) |
| Rejected, really treatment vs placebo, sham or vehicle | 50 | 45 now accepted |
| Rejected, really an add-on design | 34 | 0 accepted (see D11) |
| Rejected, head-to-head, dose-vs-dose, or not a treatment comparison | 93 | 90 still rejected |
| Rejected, reader could not tell | 3 | Still rejected |

Problems the review found in real records and the rule now handles: an
established reference drug "beating placebo" counted as the new treatment
succeeding (over a hundred development-set trials); a comparator matched
against a group whose title matched no registered arm; and "highest dose"
picked across different drugs or dosing schedules.

A reviewer model then read a fresh sample of 40 trials accepted by the
second version of the rule: 20 recognised by placebo title (none wrong), 10 by
arm type (2 wrong: one with the roles inverted, since fixed, and one comparing
patient populations), 10 by control phrase (none wrong, 2 using historical
controls). It also checked 20 dose-ladder picks, all correct. The rule changed
slightly after that read too.

Proportions from samples of 60 are good to about ±13 points.

---

## Beliefs

### B1. Overall success rate
- **Belief:** 50 to 60% of eligible trials have at least one treatment arm beat control at p < 0.05 (the headline label, D3b).
- **Confidence:** low.
- **Why:** trials that post a formal comparison are mostly randomised and mostly completed, so they are a favourable slice. Trials that failed early rarely get this far.
- **After development set:**
- **After test set (11 Nov 2026):** Success rate 50.6% (47.9 to 53.4) on 1,236 test trials, just above the pre-registered 45 to 50% (outside; interval overlaps).

### B2. Phase 3 succeeds more often than Phase 2
- **Belief:** Phase 3 is about 10 points higher.
- **Confidence:** medium.
- **Why:** Phase 3 trials are run only after a positive Phase 2, and are powered for the endpoint.
- **After development set:**
- **After test set:**

### B3. The three endpoint rules
- **Belief (rewritten for D3b, still before data):** "any arm" is the most lenient and "all arms" the strictest, by construction. The rules agree on the 70% of trials with one treatment arm. On multi-arm trials, "any" and "all" differ by 20 to 30 points; across all trials by 6 to 9. On dose ladders, the highest dose succeeds almost as often as "any dose" (within 5 points). Under "any arm", success rises with the number of arms. The model's ranking score (AUROC) moves by less than 0.02 between rules.
- **Confidence:** medium on the ordering and the gaps, low on the ranking score.
- **Why:** in a dose ladder the top dose is usually the one that works if any does; low doses are often included to find the floor. More arms give more chances at p < 0.05.
- **After development set:**
- **After test set:**

### B4. Biologics versus small molecules
- **Belief:** biologics meet their primary endpoint 5 to 10 points more often.
- **Confidence:** low to medium.
- **Why:** antibodies are specific to a validated target and have fewer off-target failures. Industry reports show biologics clearing development about twice as often, though that measures progression between phases, not endpoints.
- **After development set:**
- **After test set:**

### B5. Advanced modalities
- **Belief:** with 20 test trials (16 of them oligonucleotides), the interval on the success rate is wider than ±15 points, and no claim about cell, gene or oligonucleotide therapies can be tested.
- **Confidence:** high.
- **Why:** arithmetic on the counts above.
- **After development set:**
- **After test set:**

### B6. Vaccines
- **Belief:** the highest success rate of any group, above 75%.
- **Confidence:** medium.
- **Why:** immune-response endpoints against placebo are close to guaranteed for a working platform; many are non-inferiority lot-consistency trials.
- **After development set:**
- **After test set:**

### B7. Oncology versus the rest
- **Belief:** oncology succeeds 5 to 15 points less often.
- **Confidence:** low to medium.
- **Why:** survival endpoints against active standard of care are hard to beat. Countering this, single-arm oncology trials drop out of the labelled set, leaving the larger randomised ones.
- **After development set:**
- **After test set:**

### B8. Industry versus other sponsors
- **Belief:** industry-sponsored trials succeed about 10 points more often.
- **Confidence:** medium.
- **Why:** more selective about what reaches Phase 2 and 3, better powered, and under a legal duty to post results either way.
- **After development set:**
- **After test set:**

### B9. How well a simple model predicts
- **Belief:** on the primary test set, a lookup by phase alone scores an AUROC of 0.53 to 0.58. A simple model on registration-time fields (phase, disease area, sponsor type, planned size, design) scores 0.60 to 0.68.
- **Confidence:** medium.
- **Why:** these fields carry base-rate information, not information about the drug. Published claims well above this tend to use different labels or fields set after the trial ended.
- **After development set:**
- **After test set (11 Nov 2026):** Registration model AUROC 0.730 (0.702 to 0.759) against the pre-registered 0.66 to 0.72 (outside; interval overlaps). Top two columns were phase and sponsor class, as predicted.

### B10. Leaky fields
- **Belief:** adding fields edited after the trial ended (actual enrolment, status, duration) lifts AUROC by 0.03 to 0.08.
- **Confidence:** low.
- **Why:** these fields mostly reveal early termination, and terminated trials rarely post a p-value, so much of the leak is already selected out of this population. The lift would be far larger on a completed-versus-terminated label.
- **After development set:**
- **After test set (11 Nov 2026):** Leaky minus registration +0.048 (+0.031 to +0.066), inside the pre-registered 0.02 to 0.05.

### B11. Drift over time
- **Belief:** the model scores 0.02 to 0.04 lower on the 2021 to 2023 test set than under cross-validation on the development set.
- **Confidence:** medium.
- **Why:** sponsors, disease mix and modalities shift, and the test window includes the pandemic.
- **After development set:**
- **After test set:**

### B12. Primary versus extended test set
- **Belief (superseded by B12a on the success rate):** ~~the success rate in 2024 to 2025 is 0 to 5 points higher than in 2021 to 2023.~~ Still held: AUROC on the extended set is within 0.02 of the primary.
- **Confidence:** low.
- **Why:** sponsors may post good news sooner. Against that, the legal deadline makes compliant sponsors post on time whatever the result.
- **After development set:**
- **After test set:**

### B12a. The test set has a lower success rate than the development set
- **Belief:** the success rate in 2021 to 2023 is 3 to 8 points lower than before 2021, and 2024 to 2025 is lower again. This revises B12, which guessed fast reporters would skew positive.
- **Confidence:** medium.
- **Why:** a count of trial status (not outcomes) showed terminated trials are 8.7% of eligible trials before 2021, 17% in 2021 to 2023 and 21% in 2024 to 2025. Completed trials take longer to post results, so recent years over-represent trials that stopped early, and those rarely succeed.
- **After development set:**
- **After test set (11 Nov 2026):** Over-prediction +4.6 per 100 (+2.3 to +7.3) against the pre-registered +5 to +9 (outside; interval overlaps). Test success rate 50.6%.

### B15. Placebo-controlled versus active-controlled
- **Belief:** trials against an active comparator succeed 15 to 25 points less often than trials against placebo. The model ranks them worse too: AUROC at least 0.03 lower within the active-controlled subset.
- **Confidence:** medium on the success rate, low on the model.
- **Why:** beating an established drug is a harder test than beating nothing. Nothing in the registration fields says how good the comparator is, so the model has less to go on. Only 150 active-controlled test trials, so the interval will be wide.
- **After development set:**
- **After test set (11 Nov 2026):** Active minus placebo -0.1 per 100 (-8.6 to +8.3) on 150 active against 1,053 placebo, against the pre-registered -10 to -3 (outside; interval overlaps). No effect seen.

### B16. Trial size
- **Belief:** success rises with size: trials of 1,000 or more succeed 15 to 25 points more often than trials under 100. The model ranks small trials better than large ones.
- **Confidence:** medium on the success rate, low on the model.
- **Why:** large trials are powered for their endpoint and are run only on drugs that have already shown an effect. Small trials are a mix of exploratory and under-powered studies, with more variety for a model to separate. Size is recorded after the trial (L2): a trial that under-recruited shows up as small.
- **After development set:**
- **After test set (11 Nov 2026):** Under 100 minus 300 or more: -34.3 per 100 (-40.5 to -27.6), against the pre-registered -28 to -18 (outside; interval overlaps). Further than predicted.

### B17. Modality is the most interesting subset and the hardest for the model
- **Belief:** within small molecules and within biologics, AUROC is within 0.02 of the overall figure, and no claim about a difference between them survives its interval. Advanced modalities cannot be assessed (B5).
- **Confidence:** medium.
- **Why:** modality comes from name stems, which blur the groups (L7), and the baseline uses no information about the drug itself.
- **After development set:**
- **After test set:**

### B18. Human genetic evidence for the drug's target
- **Belief (before any look at how genetic support relates to success, in either set):** trials whose target has human genetic evidence for the trial's disease succeed 5 to 15 points more often than mapped trials without it. The effect is smaller than the published "about twice as likely to be approved".
- **Confidence:** low to medium.
- **Why:** the published finding is about approval across a whole programme; our label is one trial meeting its primary endpoint, among trials that reported a comparison. Much of the advantage of genetic support may be spent before Phase 2. Disease area will confound it: metabolic disease has both strong genetics and an 81% success rate.
- **What we know about coverage (counts only, no outcomes):** about 69% of trials map to a target and disease through Open Targets. Of those, about 18% have genetic evidence dated before the trial finished (35% using today's evidence, which is hindsight). That is roughly 590 development trials and 200 test trials with support, so the test set can detect a gap of about 8 points or more.
- **After development set:**
- **After test set (11 Nov 2026):** Supported minus not supported, mapped trials: +17.3 per 100 (+9.8 to +24.4; 258 against 468), against the pre-registered +5 to +13 (outside; interval overlaps). Larger than predicted, but only +0.004 AUROC as a model input. See B20.

### B18a. Genetic evidence: definition, grading, and why hindsight is not disqualifying
- **Definition (fixed before any outcome was joined, agreed with John 10 Nov):** a trial has genetic support if any of its drug's targets has a genetic-association score above zero for the trial's disease in Open Targets. Two versions: counting all evidence known today, and counting only evidence dated before the year the trial started.
- **Why both:** they answer different questions. Today's evidence is the best measurement of the biological fact: if targets with genetic support succeeded more often even before anyone knew of that support, genetics matters (John's point). Evidence dated before the trial started is what a predictor could have used at the time. The gap between the two is itself a finding.
- **Graded version:** none, then three bands of the positive scores, with cut points taken from the development set's scores before looking at outcomes. Belief: success rises with each band. If the type of evidence can be separated, rare-variant (Mendelian) support does more than common-variant (GWAS) support.
- **Confidence:** low to medium on the trend; low on the type of evidence.
- **One bias that remains with today's evidence:** targets of successful drugs attract more research, so some genetic evidence may exist because the drug worked. Genome-wide studies are not aimed at drug targets, so we expect this to be small.
- **After development set:**
- **After test set:**

### B19. Rare diseases and failing to recruit
- **Belief (John's, before data):** trials in rare diseases are more likely to stop because they cannot recruit enough patients.
- **Claude's estimate:** among finished trials, the share stopped for recruitment is 1.3 to 2 times higher in rare diseases. But among rare-disease trials that do report a comparison, the success rate is no lower, and may be higher, because rare diseases are often genetically defined with large treatment effects.
- **Confidence:** medium on recruitment, low on the second part.
- **Why it matters for the benchmark:** our label covers only trials that reported a treatment-versus-control comparison. A trial that never recruited is a failure the benchmark cannot see (L1). If that kind of failure is concentrated in rare diseases, a predictor that scores well here could still be badly wrong about which rare-disease programmes will deliver a result.
- **After development set:**
- **After test set (11 Nov 2026):** Rare-disease recruitment stops, ratio 1.00 (0.70 to 1.33), no range pre-registered; no effect seen. Rare minus not rare success -0.7 per 100 (-8.8 to +7.5) against the pre-registered -18 to -8 (outside; interval overlaps).

### B13. Does knowing the modality help the model?
- **Belief:** adding modality as a feature improves AUROC by less than 0.01.
- **Confidence:** medium.
- **Why:** phase, disease area and sponsor type already carry most of what modality would add, and 77% of trials are small molecules.
- **After development set:**
- **After test set (11 Nov 2026):** Registration minus registration without modality: -0.000 (-0.004 to +0.003). Inside the pre-registered "below 0.01".

### B14. Strength of evidence is harder to predict than success
- **Belief:** a model predicting the z-score explains less than 5% of its variance.
- **Confidence:** low.
- **Why:** the z-score depends on sample size and endpoint choice as much as on the drug.
- **After development set:**
- **After test set:**

### B20. Public biology and ranking (supersedes the development reading)
- **Belief (before the test set):** ~~public biology adds nothing to ranking~~ (development: -0.001, -0.010 to +0.008).
- **After development set:** -0.001 (-0.010 to +0.008).
- **After test set (11 Nov 2026):** nothing in development; +0.012 (+0.003 to +0.021) on test, all trials, and +0.019 (+0.005 to +0.034) on 764 mapped trials. A lead, not a finding: one secondary comparison, unseen in development. Genetic support as a model input: +0.004 (-0.001 to +0.008).

### B21. Phase 3 after a successful Phase 2
- **Belief (before the test set):** ~~Phase 3 after a successful Phase 2 succeeds far more~~ (development: 89.6% of 77 against 47.6% of 21, +42.0).
- **After test set (11 Nov 2026):** did not hold. 78.4% of 51 against 75.0% of 12. Too few in the failed group to tell.

### B22. Broadly expressed single targets
- **Belief (before the test set):** lower success (development: -10.0, -17.1 to -3.3, adjusted). The one pattern among the science features that stood out after the fact.
- **After test set (11 Nov 2026):** -11.4 (-27.1 to +5.6) on test; reversed on late (+13.2). Unresolved.

### B23. Language models: recall or reasoning
- **Belief (before the test set):** every period is inside these models' training data, so a high score cannot be told from recall.
- **After test set (11 Nov 2026):** on the 136 test trials neither model named, Sonnet is +0.013 (-0.104 to +0.128) over registration: too few to tell an edge from none. Cannot separate recall from reasoning.

### B24. How far development validation can be trusted
- **Belief (before the test set):** the primary model gains +0.036 on development (carrying some selection optimism); expected +0.02 to +0.035 on test.
- **After test set (11 Nov 2026):** +0.018 (-0.001 to +0.038). Development was optimistic by about half for the primary model. The decision rule (lower end above zero) is narrowly not met. Test and late together +0.026 (+0.010 to +0.042); late alone +0.047 (+0.017 to +0.079). Words carried the gain (+0.022).

### B25. The trend-shifted intercept
- **Belief (before the test set):** built on development data after the +4.6 had been seen; arithmetic said it would over-correct, to about -5.7.
- **After test set (11 Nov 2026):** over-corrects, -5.7 (against +4.6 for registration).

### B26. Genetic support: rate gap against model gain
- **Belief (before the test set):** +5 to +13 per 100 (B18).
- **After test set (11 Nov 2026):** +17.3 per 100 (+9.8 to +24.4), larger than predicted. As a model input only +0.004 AUROC. Phase and disease area may carry part of the gap.

### B27. Endpoint type
- **Belief (before the test set):** survival endpoints succeed less (development: -13.6 adjusted); biomarker endpoints more (+13.3 adjusted).
- **After test set (11 Nov 2026):** survival held: -14.6 adjusted (-23.9 to -5.0); late -33.3. Biomarker: through zero on test (+6.4), held on late (+19.9, +7.7 to +33.3).

---

## Limits

Each one says what it means for interpretation, and whether we can measure it.

### L1. Only about 10% of finished trials can be labelled
- **Limit:** 75,905 completed or terminated trials with a drug-like intervention; 69,685 with a completion date to 2025; 35,954 with results; 7,593 with a treatment-versus-control comparison; 7,577 with a determined label (10% of the 75,905).
- **Consequence:** this is a benchmark of trials that reported a formal comparison. Scores say nothing about trials that never reported, and failures are under-counted.
- **Measurable?** Partly: we can compare labelled and unlabelled trials on phase, sponsor, size and status.

### L2. We see each record only as it is today
- **Limit:** the API serves the current version. Sponsors edit records after the trial ends, including outcomes, enrolment and dates.
- **Consequence:** a "registration-time" field may not be what was registered. Any model using the record can be reading the future.
- **Measurable?** Partly. Enrolment is marked "actual" in every eligible record, so the planned size is gone. Registered history exists on the website but not in this API.

### L3. Meeting a primary endpoint is not the drug working
- **Limit:** statistical significance on one endpoint is not clinical benefit, approval or commercial success. The endpoint may be a biomarker the sponsor chose, which no regulator has accepted as evidence of benefit.
- **Consequence:** the benchmark cannot test claims about approval or about whether a drug helps patients.
- **Measurable?** No, not from this source.

### L4. Direction of benefit is not recorded
- **Limit:** the record does not say whether higher or lower is better.
- **Consequence:** a significant result in the wrong direction counts as a success.
- **Measurable?** By hand on a sample. In 40 multi-comparison trials read before 2021: readable from the record in 15, only with domain knowledge in 17, not at all in 8. No rule recovers it in general.

### L5. Sponsors choose what is listed first, and how many tests to run
- **Limit:** of trials with a primary p-value, 31% have more than one primary outcome and 47% report more than one p-value; 53% have one or the other.
- **Consequence:** "first-listed" follows the sponsor's ordering; "any" rewards running more tests; "all" penalises it.
- **Measurable?** Yes: run all three rules and compare.

### L6. Results arrive late, and not at random
- **Limit:** fewer than half of trials post within two years of finishing.
- **Consequence:** recent years over-represent fast reporters. The newest trials, the ones a time-split most wants, are the least representative.
- **Measurable?** Yes: compare the primary and extended test sets.

### L7. Modality comes from drug-name stems
- **Limit:** "small molecule" means "no biologic stem", so it includes peptides, hormones and company-coded drugs of any kind.
- **Consequence:** modality contrasts are blurred, most likely towards no difference.
- **Measurable?** By hand on a sample.

Modality is read from the record as it is today, and records gain generic and
brand names after a drug succeeds (NCT02597127 registered as "ALN-PCSSC", now
also "Inclisiran"), so whether a drug gets a stem-based modality partly depends
on what happened later. The stem rule also uses a short hand-made list of brand
names and ordinary words that would otherwise match.

### L8. Advanced modalities are too few
- **Limit:** 24 trials in the primary test set, of which 1 is a cell and 3 are gene therapies.
- **Consequence:** no claim about cell, gene or oligonucleotide therapies can be tested here.
- **Measurable?** Yes: the interval width shows it.

### L9. A model may already know the answer
- **Limit:** results from 2021 to 2023 are public, and in the training data of any recent language model.
- **Consequence:** a time split protects only against models trained before the test window. For newer models the benchmark measures recall, not prediction.
- **Measurable?** Not here. It needs trials whose results are not yet out, scored later.

### L10. Trials are not independent
- **Limit:** the same drug, sponsor and programme appear in several trials, on both sides of the split.
- **Consequence:** uncertainty intervals that treat trials as independent are too narrow, and a model can score by recognising a programme.
- **Measurable?** Partly: resample by sponsor, or check drugs that appear in both development and test sets.

### L11. p < 0.05 is not every trial's threshold
- **Limit:** some trials pre-specify a different alpha, one-sided tests, or non-inferiority margins.
- **Consequence:** some labels disagree with what the sponsor would call success.
- **Measurable?** Partly: the record marks non-inferiority tests (6% of analyses; 500 trials have nothing else). Stated alphas of 0.2, one-sided 0.1379 and 0.025 all appeared in a sample of 40, and 17% of analyses mention a multiplicity adjustment in free text.

### L12. One snapshot
- **Limit:** built from the registry as it stood on 10 Nov 2026.
- **Consequence:** a rebuild later gives slightly different trials and labels. The exact trial list will be committed.
- **Measurable?** Yes, by rebuilding.

### L13. Many posted p-values are not a treatment-versus-control test
- **Limit:** before 2021, 8.8% of primary analyses test a single group (change from baseline), 4.4% are omnibus tests across three or more arms, and about 4% compare two experimental arms such as two doses. In 15% the roles of the groups cannot be determined. Some primaries are safety, pharmacokinetic or feasibility measures with incidental p-values.
- **Consequence:** a significant p-value can mean "the dose groups differ" or "the drug raised bleeding", not "the drug worked". One sampled trial's second primary was major bleeding, significant in the direction of harm.
- **Measurable?** Partly: group count is reliable. The control is recognised mostly by a placebo title; trials with no recognisable control are left out and counted.

### L14. "First-listed" often means the lowest dose
- **Limit:** in trials with more than one primary outcome or p-value (53%), the first analysis follows arm order. In 15 of 17 dose-ranging trials read, that was the lowest dose against control.
- **Consequence:** a first-listed rule would under-count success in dose-ranging trials; it was a defensible primary test in only 16 of 40 such trials read. This is why the label is "any arm" (D3b), not "first-listed".
- **Measurable?** Yes: compare "any arm", "all arms" and "highest dose".

### L15. The registered protocol is not an independent check on the results
- **Limit:** the first primary outcome in the results section matched the first in the protocol section in 100% of trials, and counts agreed in 99.8%.
- **Consequence:** the protocol section has evidently been brought into line with the results, so it cannot show whether the primary outcome was switched after registration. This is L2 seen directly.
- **Measurable?** Not from this API.

### L16. Some posted p-values are not usable numbers
- **Limit:** 29% are bounds such as "<0.001"; a small number are above 1, equal to zero, bounded above 0.05 (such as "<0.2"), or text. Posted values are also rounded, so a posted "0.05" may hide 0.046. (Seen as formats in pre-2021 records; no rate of significance was computed.)
- **Consequence:** bounds below 0.05 are fine for the binary label but give only a floor for the strength-of-evidence label. The unusable ones are excluded and counted.
- **Measurable?** Yes, counted at build time.

### L17. Some kinds of real treatment comparison are left out
- **Limit:** head-to-head trials where the record does not mark the comparator (roughly 600), add-on designs without a placebo arm (roughly 350), and placebo controls named by an abbreviation the rule does not know.
- **Consequence:** the benchmark leans towards placebo-controlled trials (83% of eligible trials), and under-represents oncology-style add-on designs.
- **Measurable?** Partly: the estimates above come from the audited samples.

### L18. Known ways the control rule is still wrong
- **Limit:** a reference drug registered as "OTHER", or under a title matching no registered arm, still counts as a treatment (a handful of trials). Arms that are patient populations, not treatments, cannot be told apart by rule. Controls titled "Placebo/Drug X" (delayed start) are accepted, which is right only if the endpoint falls in the placebo period (about 50 trials). A few controls are historical, not randomised. A dose arm mistyped as an active comparator is dropped as a reference when another arm remains.
- **Consequence:** a small share of labels, on the order of a few per cent, answer a slightly different question from "did the new treatment beat a concurrent control".
- **Measurable?** By a fresh hand-read sample, which is still owed.

### L19. Fields we treat as known at registration can be edited afterwards
- **Limit:** therapeutic area is derived from MeSH terms for the current conditions; the number of registered arms can fall when arms are dropped; start year is the actual one, not the planned one; sponsor class changes when a company is acquired. The number of arms compared and the kind of control come from the results section, so they are not used as model inputs.
- **Consequence:** a model that uses these fields may be reading a little of the future, and a field may differ from what a sponsor could have known at registration.
- **Measurable?** Not from this API, which serves only the current record.

Therapeutic areas are assigned by a fixed priority over MeSH ancestors, so a trial in two areas goes to the first. "Other" hides coherent groups (kidney and urology, eye, pain, blood), and 652 trials have no MeSH terms.
