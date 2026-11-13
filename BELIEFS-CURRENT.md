# What we currently believe

**In short:** a model can look good at predicting trials without knowing anything about the drug. Two things do most of the work. One is how often trials of that kind succeed anyway: 58% of ours do, so "always say yes" is right 58% of the time. The other is hints about the answer hidden in the data: the registry record is edited after a trial ends, and may say it was stopped early. So a claim such as "90% accuracy" means little until you know what counted as success, which trials were scored, when the model's information was frozen, and what a trivial guess would have scored.

We expect a plain model, using only what is known when a trial is registered, to rank trials modestly better than chance, mostly by knowing who sponsors the trial and what phase it is.

The benchmark is the ruler; our baseline is one plain predictor that shows the ruler works. This page is our position as of 10 Nov 2026. The full working record, with every decision and its reasoning, is in [BELIEFS.md](BELIEFS.md).

**Updated 11 Nov 2026:** the test set has now been opened. What it changed is in the last section, "After the final test-set run". The sections above are left as they were.

---

## What we are measuring

**The label, in one sentence:** did at least one treatment arm beat its control group at p < 0.05 on the trial's first primary outcome that has a treatment-versus-control comparison?

**Which trials:** finished Phase 2 or 3 trials of a drug, biologic or genetic therapy that posted a p-value comparing a treatment arm with a control. That is 7,577 trials, about 11% of the roughly 70,000 finished trials of this kind. The rest cannot be labelled by rule.

**The time split:** by the date the trial finished collecting its primary outcome.

| Set | Finished | Trials | Status |
|---|---|---:|---|
| Development | before 2021 | 5,849 | Outcomes examined |
| Test (headline) | 2021 to 2023 | 1,236 | Unseen |
| Late | 2024 to 2025 | 492 | Unseen, reported separately |

---

## What the development set showed

Our "before data" guesses were agreed before any success rate was seen. Then we looked at the 5,849 development trials. Several guesses were wrong.

| Belief | What we said first | Development set |
|---|---|---|
| Overall success | 50 to 60% | **58.4%** |
| Phase 3 vs Phase 2 | about 10 points higher | **23 points** (67.6% vs 44.9%) |
| Industry vs other sponsors | about 10 points higher | **about 30 points** (64.6% vs about 33%) |
| Active comparator vs placebo | 15 to 25 points lower | **7 points** (52.7% vs 59.7%) |
| Size: under 100 vs large | 15 to 25 points | 43% under 100, 69% at 300 or more, **flat above 300** |
| Biologic vs small molecule | 5 to 10 points higher | **6 points** (63.1% vs 57.5%) |
| Vaccines | above 75% | **53.8%** on 39 trials. Wrong. |
| Oncology vs the rest | 5 to 15 points lower | **about 17 points lower** (44.1%) |
| Any arm vs all arms | 6 to 9 points apart | **6 points** (58.4% vs 52.0%) |
| Highest dose vs any dose, on dose ladders | within 5 points | **3 points** (69.8% vs 73.1%) |

**The biggest surprise was sponsor type:** industry trials succeed about twice as often as academic and government ones.

---

## What we predict for the unseen 2021 to 2023 test set

These go into a short pre-registration email before the test set is opened.

**Base rate around 45 to 50%, lower than the development set.** Confidence: medium. The rate already falls year by year inside the development set: about 60% for 2009 to 2014, 53 to 58% for 2015 to 2019, 49.8% for 2020. Terminated trials succeed 25% of the time against 62% for completed ones, and they are 9% of the development set but 17% of the test set (a count of status, not outcomes), because completed trials take longer to post results.

**A simple model scores an AUROC of about 0.65 to 0.72.** AUROC is the chance the model ranks a random successful trial above a random failed one; 0.5 is a coin toss. Confidence: medium. The model uses only registration-time fields, and we expect sponsor type and phase to do most of the work. We first guessed 0.60 to 0.68; the size of the sponsor and phase gaps moved us up.

**The model is too optimistic on the test set, by 8 to 12 points.** Confidence: medium. Trained on older trials, it will predict success for about 58% of trials where the truth is nearer 47%. This follows from the falling base rate. We are less sure whether its ranking gets worse: the test set has more academic and terminated trials, and that variety could make ranking easier.

**Telling the model the drug's modality barely helps: under 0.01 AUROC.** Confidence: medium. Once the model knows sponsor, phase and disease area, "antibody" or "small molecule" adds little. If so, modality differences in success are mostly about who runs those trials, not the molecule.

**A deliberately leaky model does clearly better, by more than on the development set.** Confidence: medium. Fields edited after the trial ended, such as actual enrolment and final status, reveal early termination. Terminated trials are a bigger share of the test set, so the leak is worth more there.

### The three subsets the report leads with

**Placebo vs active control:** a small gap, in line with the 7 points seen. Only about 150 active-controlled test trials, so the interval will be wide.

**Trial size:** small trials do worse; above 300 participants, size makes little difference.

**Modality:** biologics succeed about 5 points more often than small molecules. With 282 biologic trials in the test set, the uncertainty on that gap is about 7 points either way, so it will probably not be distinguishable from zero. Nothing can be said about cell, gene or oligonucleotide therapies: there are 24 in the test set.

---

## The limits a sceptic will raise first

**Only 11% of finished trials can be labelled.** Failures that never reported a comparison are invisible, so success is over-counted.

**We see each record as it is today.** Sponsors edit records after the trial ends; even enrolment is the actual figure, not the planned one.

**Meeting a primary endpoint is not the drug working.** The endpoint may be a biomarker. The benchmark says nothing about approval or patient benefit.

**The record does not say which direction is better.** A significant result in the wrong direction counts as a success.

**Trials are not independent.** The same drug and sponsor appear on both sides of the split, so intervals are too narrow and a model can score by recognising a programme.

**A language model may already know the answer.** 2021 to 2023 results are in recent models' training data; for them the time split tests recall, not prediction.

The other limits, including the p < 0.05 threshold and known errors in the control rule, are in [BELIEFS.md](BELIEFS.md#limits).

---

## Still undecided

- **Genetic evidence for the drug's target.** We are scoping whether human genetic support for a target (from Open Targets) is linked to trial success. Our starting view: yes, by a modest margin, smaller than the published "about twice as likely to be approved". (Superseded 11 Nov: on test the gap was larger, but it adds little as a model input. See the last section.)
- Which registration-time fields the baseline may use, and which count as leaky.
- Whether trials stopped for futility, which rarely post a p-value, are added as failures from the "why stopped" text.
- Whether mixed-modality trials stay in, and whether to use more than the first eligible primary outcome.
- A fresh hand-read sample to check the control rule is still owed.

---

## After the final test-set run (11 Nov 2026)

**The main result: the best honest model narrowly did not clearly beat the baseline.** The model was named in advance: registration plus four tables (endpoint type, the drug's earlier trials, genetic support, the record's words). On the 1,236 test trials it gains **+0.018 AUROC** over registration alone (-0.001 to +0.038). Our rule was that the lower end must be above zero, so it is **narrowly not met**. We expected +0.02 to +0.035; development had estimated +0.036.

- **Test and late together** (1,728 trials): +0.026 (+0.010 to +0.042).
- **Late set alone** (492 trials): +0.047 (+0.017 to +0.079).
- **Words carried the gain:** +0.022 on their own.

The run was made at commit 69dd31b (the second logged read of the test labels, 19:36 UTC 11 Nov). The scorecard was made at 3399dff (the third logged read). Results are committed as 3399dff and 0ae4027. The report was revised after the Opus and Gemini reviews at 825ba44 and f85a814.

### Beliefs that changed

| Belief | Before | After the test set |
|---|---|---|
| Public biology and ranking | Nothing: -0.001 in development | Nothing in development; **+0.012 (+0.003 to +0.021)** on test, all trials, and +0.019 on mapped trials. A lead, not a finding. |
| Phase 3 after a successful Phase 2 | Succeeds far more | Did not hold on test: 78.4% of 51 against 75.0% of 12. Too few failures to tell. |
| Broadly expressed single targets | Lower success | -11.4 (-27.1 to +5.6) on test; reversed on late. Unresolved. |
| Language models | Could be recall or reasoning | On the 136 test trials neither model named, Sonnet is +0.013 (-0.104 to +0.128) over registration. Cannot separate recall from reasoning. |
| Development validation | A fair guide | Optimistic by about half for the primary model (+0.036 expected, +0.018 seen). |
| Trend-shifted intercept | Would fix the over-prediction | Over-corrects: -5.7. |
| Genetic support, success-rate gap | Modest | **+17.3 per 100** (+9.8 to +24.4), larger than predicted. But only +0.004 AUROC as a model input. |
| Endpoint type | Survival endpoints succeed less | Held: -14.6 adjusted. |

### Scorecard of the seven pre-registered predictions

- **Inside the range (3):** the gain from leaky fields, the modality contribution (under 0.01), and the top two columns (phase, sponsor type).
- **Outside, with the interval overlapping the range (7):** success rate, over-prediction, baseline AUROC, active comparator, trial size, genetic support, rare-disease success.
- **No range stated (3):** biologic against small molecule, genetic support in newer drugs, rare-disease recruitment.
- **Flat results:** active comparator, rare-disease success and rare-disease recruitment showed no effect. Small trials and genetic support went our way but further than predicted.

### Still open

- The owner's placeholders in REPORT.md: date, active hours, assistant account.
- Regenerate the results from a clean commit.
- Decide on a remote, and on pushing the `prereg` tag.
- Overnight work is under way on training data only, asking which patterns could be told to someone starting a programme. No test reads.
