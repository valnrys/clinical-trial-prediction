# Pre-registration

**Emailed:** Wed 11 Nov 2026, 06:21 GMT
**Commit / tag:** `prereg`

Written before inspecting any test-set results. The email is the independent
timestamp; this file is the same text, kept with the code.

_Everything below the line is the email._

---

**Subject:** Pre-registration: clinical-trial outcome benchmark

Hi,

Here is a quick breakdown of how I'm framing the problem, my general approach, and a few predictions. Note that this email was written before I looked at the test-set outcomes, but after exploring the training data.

**Design and approach**

The results of this analysis will depend heavily on how we define trial success, so the definition I'm using is: a successful trial is one where at least one treatment arm beat the control at p < 0.05 on the primary outcome.

Beginning with every Phase 2 and 3 trial on ClinicalTrials.gov (about 133,500), I narrowed the dataset down to 7,577. I filtered for trials testing a drug, biologic, or genetic therapy that had finished by the end of 2025, and specifically those that posted a p-value comparing a treatment arm with a control arm on a primary outcome. That last requirement is very stringent: only about 1 in 10 finished trials meets this criterion. This substantially reduces the available data, but given the time constraints, I wanted a dataset from which I could draw strictly data-driven conclusions, rather than using LLM-inferred information.

For the train/test split, I chose to separate the data chronologically to investigate if past trials can predict more recent ones. Based on some initial exploration, I defined the training set as trials completed before 2021 (5,849 trials), and the test set covers 2021 to 2023 (1,236 trials). I've set aside trials finishing in 2024 and 2025 (492) to report separately. Results are due on the registry a year after a trial finishes, but looking into reporting rates, in practice about a third arrive more than two years after, so it's likely that the 2024–2025 data is still incomplete.

The baseline model I have chosen is deliberately simple: a logistic regression using only the parameters available at trial registration (phase, sponsor type, disease area, and basic design). Because this model is blind to any specific drug and post-registration events, it gives us a clean look at how far background facts alone can take us, and any model that claims to predict trial outcomes should beat it. I'll also compare this baseline against a couple of more rudimentary models, such as guessing the average, or using trial phase alone, and against a version that includes post-trial edits, to get a measure of how much performance gets artificially inflated by leaked information.

**Predictions for the test set**

1. **A drop in the overall success rate.** Exploring the data, the success rate has steadily fallen since 2014. My prediction is that only 45–50% of the test-set trials will succeed, down from 58% in the training set. As such, we would expect a model trained on older trials to over-predict success rates, predicting 5 to 9 more successes per 100 trials than occur.

2. **The baseline will perform modestly well.** I anticipate an AUROC of between 0.66 and 0.72, with most of the signal coming from just two features: the trial's phase and whether the sponsor is a company (as opposed to an academic sponsor).

3. **Data leakage will inflate scores.** For fields that can be edited after the trial ends, I expect leaked information to raise the AUROC by roughly 0.02 to 0.05.

4. **Drug modality will be a poor predictor of success.** In the training set, biologics trials were marginally more successful than small molecule trials (63% vs 58%). I expect a similar gap in the test set, but the test set is likely too small to show it is statistically significant. Separately, I expect that telling the model the drug type will barely improve how well it ranks trials (less than 0.01 on AUROC), because phase, sponsor and disease area already carry most of that information.

5. **The type of control will be a weak signal, and trial size a strong one.** Trials run against another active treatment succeeded less often than placebo-controlled ones in the training set (53% vs 60%), and I expect that penalty (roughly 3 to 10 fewer successes per 100 trials) to persist. I also expect trial size to be a strong predictor of success. Trials with fewer than 100 participants in the training set had a 43% success rate compared to 69% for trials with 300 or more. I expect a similar trend for small trials in the test set, with between 18 and 28 fewer successes per 100 trials.

6. **Human genetic support will boost success rates.** This prediction is something of a personal indulgence of mine. The published claim is that drugs with human genetic support are about 2.6 times as likely to reach approval. I'm interested in how much of that shows up at the level of a single trial. Mapping trials to drug targets and diseases via Open Targets shows a clear edge: 68% success with genetic support vs 57% without in the training set. I expect a similar boost (5–13 more successes per 100 trials) in the test set, though likely weaker for newer drugs lacking historical trial data. Since this relies on external data, I'll report it alongside the baseline rather than as part of it.

7. **Rare diseases will underperform.** From my personal experience, I expect rare diseases to fare worse. Outside of oncology, I expect rare-disease trials to be halted for lack of recruitment more often than other trials: about 1.2 times as often, as in the training set. Also, for those trials that do report a result, I anticipate a lower success rate (in the training set it was 47% vs 62%, and I expect 8–18 fewer successes per 100 trials). We only have about a hundred rare-disease trials in the test set, so I will likely lack the statistical power to confirm this, but my prior from working in a rare disease company is that this will be a factor.

John
