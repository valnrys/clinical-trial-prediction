# A benchmark for claims that AI can predict clinical-trial success

John McGonigle, 13 November 2026

**In short.**
Most of the predictive signal of trial success can be matched or beaten by a model considering only information on how it was registered, AUROC 0.730. Information about the trial derived from external sources, such as the disease, the target, and factors about its biology, can add only a moderate gain in AUROC, 0.026 to 0.043, and alone is less predictive. This isn't a paradox, it's a problem. Much of what decides a drug discovery programme's success is likely already determined before a trial is registered.

## 1. Method

I defined a successful trial as one where at least one treatment arm beat the control at p < 0.05 on the primary outcome. Starting with every Phase 2 and 3 trial on ClinicalTrials.gov (~133,500), I kept the 7,577 that tested a drug, biologic or gene therapy, completed before the end of 2025, and reported a p-value comparing at least one treatment arm with a control arm on the primary outcome.

I split the data chronologically by completion date: a training set of trials completing before 2021 (5,849), a test set completing 2021 to 2023 (1,236), and 2024 and 2025 held in reserve (492). I then compared this with a classic random train/test split that ignores completion date, first inside the training set (treating 2019 and 2020 as the 'future') and then on the test set.

## 2. Models

The baseline for this analysis was a logistic regression using only parameters available at trial registration: phase, sponsor type, disease area and basic trial design. Note, sponsors can, and do, update these values after registration. I tried about 50 models: simple floors (the training success rate, phase alone), variants of the baseline, tree and neural models, graph-based models, text and language models, and deliberately leaky ones to size the effect of leakage. In my first test round, the best honest models gained about 0.02 AUROC; in the logistic models nearly all of it came from the words of the trial record. A hierarchical model, with its own effect for each disease, drug, target and sponsor, gained 0.038 on the test set (0.025 on the training set) without reading any text, which suggests the words were standing in for what is being treated and how. Graph-based approaches were explored and a graph neural network over the same links produced a similar result on the training set (0.021), but was harder to interpret, so it wasn't taken to the test.

## 3. Sources of evidence

To explore the predictive potential of publicly available scientific data, I first utilised Open Targets to link each trial to its drug, the drug's target genes and the disease. Using this backbone, I linked seven sources of evidence: human genetic evidence for the target in the disease (Open Targets), disease prevalence (Orphanet), cancer-cell dependency (DepMap), tissue expression (GTEx and the Human Protein Atlas), genetic perturbation screen results and protein interactions (BioGRID), target class, location and tolerance of damage (Open Targets), and any publicly known chemistry of the drug (ChEMBL). Most were tested alone, added to the baseline, and combined with the others.

## 4. Results

**Much of the informative variance may be fixed by the time a trial is registered.** On the test set the registration baseline ranks trials at AUROC 0.730 (95% interval 0.698 to 0.761). In fact, phase alone accounts for 0.630. In comparison, a hierarchical model with its own effect for each disease, drug, target and sponsor reached 0.768, a gain of 0.038 (0.018 to 0.057). Adding precedent and genetic support made it the best of about 50 models tested, at 0.774, a gain of 0.043 (0.023 to 0.064). When the baseline was provided with fields edited after the trial, it scored 0.779. Thus, we must be careful not to conflate predictive power with hindsight.

**Predicting purely on the basis of disease biology is less predictive than the baseline, but contains novel signal.** The disease, the target and its biology, with no registration details, rank trials at 0.689. Added to the baseline they gain 0.026 (0.013 to 0.041). A score built from target biology alone, fixed before the test, split Phase 2 trials into a top third that succeeded 52 times in 100 and a bottom third that succeeded 34. In Phase 3 it was 74 against 48. On the training set, among comparable single-target trials, targets expressed in nearly every tissue saw 10 fewer successes per 100, intracellular targets 12 fewer, and drugs that activate their target 15 fewer. Across all trials with a known target, human genetic support added about 5.

**Control type showed no difference in trial success, whereas trial size did.** My intuition had been that trials run against an active (and often approved) drug, and smaller trials, would have a lower success rate. This is because proving value against a working drug is more challenging than beating a placebo, and smaller trials are likely to have less power to detect a difference. On the test set, placebo-controlled trials succeeded 50.8 times in 100, almost identical to 50.7 for active-controlled trials. Trials with fewer than 100 patients succeeded 32.1 times in 100, whereas trials with 1,000 or more succeeded 57.6 times. The baseline was also weakest on the smallest trials (AUROC 0.654, against 0.762 for the largest). Both groupings use fields recorded after the trial, so they could not feed a model.

### What this analysis can and can't do

The benchmark employed can test whether a model ranks trials well and whether its probabilities can be trusted, on controlled Phase 2 and 3 drug trials that report results, provided the predictions were fixed before each trial finished. It cannot test claims about approval, patient benefit, or trials that never post a comparison.

**A claim of "90% accuracy"** can be tested by this approach, but, as demonstrated by this analysis, can mean very little alone. Here, giving every trial the same answer scores 50.6% and the baseline 68.2%. Accuracy also depends on how success is defined, so achieving 90% might say more about the choice of label than about the model.

**It could mislead** because every trial here has already finished, so hindsight can leak in. Fields edited after the trial added 0.048 AUROC. A precedent flag built from undated sources added 0.011 that vanished once they were dated. A random holdout hid the baseline's over-confidence: 0.2 per 100 against 4.6 on later trials.

**Critically, public data cannot tell us** how unreported trials ended, or what a vendor's model saw in training.

{{figure: forest}}

{{figure: curves}}

**My prediction scorecard**

{{table: predictions}}

## 5. Limits

- Only trials that report are here; industry and Phase 3 trials are over-represented.
- A result significant in the wrong direction could be recorded as a success, and more arms mean more chances.
- Registrations and most databases are based on today's data and thus the information likely wouldn't be available at trial design.
- We are judging marathon finalists by their last ten miles: most programmes never reach a trial.

## 6. Time, and next

**Active time:** 8 hours. **Next:** In short, fill in the missing data. This analysis was restricted by the strict criterion for inclusion. Using an LLM-based workflow, with multiple rounds of validation, it should be possible to recover trials where values exist outside the registry. Inclusion of more structured and dense data sources, such as UK Biobank, could also improve predictive power.

## Three, almost, one-line answers

1. **Pretrained model in the labels or baseline?** No. Each label comes from a fixed rule applied to the trial's posted results, and the baseline is a logistic regression. Language models and text embeddings appear only as comparisons, and are labelled as such. Claude read a sample of trials to check the rule, but never decided a label.
2. **Information beyond each trial's own record?** Not in the labels or baseline. Only in marked extras: Open Targets, Orphanet, the drug's earlier trials, and five biology databases.
3. **Coding assistants?** Claude Code: Sonnet to write the code. Opus and Gemini for adversarial review of every change. These tools enabled a scope and breadth of work that would have been impossible without them given the 8-hour (one-day) time allowance. Where they hindered, the agents got obsessed with measurement as the aim rather than translatability.
