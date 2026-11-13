# A language model as predictor: how much is design, how much is recall?

Development trials only (finishing 2017 to 2020). No test or late trial is scored here.

## What this shows

1. **The scores.** On 300 development trials finished 2017 to 2020, Sonnet reading the registration record scored AUROC 0.92 and Haiku 0.88, against 0.72 for the registration baseline and 0.75 for the best model built on registration details and wording.
2. **Most of that cannot be separated from recall.** With drug names removed, Sonnet still named the drug in 45% of trials by its guess (50% by the stricter rule) and Haiku in 30% (32%). Sonnet's stated reasons often cite a named trial and its result: 53 of 300 blinded replies claim knowledge of an outcome. On trials a model both named and said it recognised, it scored best of any group: Sonnet 0.88 on 90 trials and Haiku 1.00 on 32 trials.
3. **Without the name, a smaller edge.** On trials it did not name, with names removed, Sonnet still ranked outcomes better than the registration model but not clearly better than the registration-plus-words model (the ladder); Haiku showed no clear edge. The split is exploratory, and "not named" is counted three ways. Gaps in AUROC (Sonnet's interval against the ladder only just clears zero under the stricter rule):

   - Sonnet. Guess alone, 165 trials: +0.119 (+0.047 to +0.209) over registration and +0.064 (−0.006 to +0.139) over the ladder. Stricter rule, 150 trials: +0.134 (+0.051 to +0.219) over registration and +0.081 (+0.007 to +0.157) over the ladder. Reviewer's hand-corrected labels, 131 trials: +0.117 (+0.034 to +0.204) over registration and about +0.057 (−0.019 to +0.135) over the ladder.
   - Haiku. Guess alone, 209 trials: +0.062 (−0.013 to +0.138) over registration and +0.007 (−0.063 to +0.072) over the ladder. Stricter rule, 204 trials: +0.060 (−0.015 to +0.137) over registration and +0.006 (−0.064 to +0.079) over the ladder. Reviewer's hand-corrected labels, 197 trials: +0.049 (−0.032 to +0.128) over registration and about −0.014 over the ladder.

   This mixes design and drug-class knowledge with whatever recall survives blinding. It is not evidence of reasoning from design.

4. **What would settle it.** A claim that a language model forecasts trial outcomes needs trials that finished after its training cut-off. None of the three periods here qualifies (both models report a cut-off of June 2026).

## Limits

- **Blinding lowers recall; it does not remove it.** A drug-class word in an outcome, an unusual endpoint, an abbreviation or a named scale can still point to the drug, and a model can recognise a programme from its conditions, design and endpoint with no drug name at all. Evidence is in the leak check section.
- **Blinding also removes legitimate knowledge.** A forecaster at the start of a trial knows the drug's class, its earlier results and who the comparator is. So a gap between prompts is recall plus that knowledge, and no gap here is a measure of recall alone.
- **The not-named split is exploratory.** The model's own output chooses it. It is harder for the baseline, and not free of recall: the automatic drug-guess match misjudges about one in six of the replies it calls not named (see the hand read below).
- **Known blinding gaps, unfixed.** The stored predictions were made from prompts built by the blinding rules as they stood, so the rules are frozen (a test re-renders a sample of stored prompts and compares their hashes). Gaps left in: a drug name followed by a colon or a dash suffix can be left unblinded (9 of 300 development prompts show a word of a tested drug's name beside a colon or dash); a spaced company code whose letters are an ignored word is not read as a code (0 of 300); "Placebo for X" is treated as a control, but in an add-on design X is also given (3 of 300 trials list a "Placebo for ..." intervention); a date written with a comma ("March 3, 2019") can survive the post-hoc clean-up (0 of 300); the short name of an industry collaborator can survive (not counted); an abbreviation glossary in a description that names the drug can survive (not counted); doubled placeholders such as "Drug B Drug B" appear (3 of 300). Counts are of the 300 sampled development trials.
- **No clean period.** All three periods (2017-2020, 2021-2023, 2024-2025) are inside these models' training data (each reports a knowledge cut-off of June 2026). A fall in score from the early to the late period reflects how much had been published before the cut-off, mixed with changes in trial mix. The late period is the least contaminated, not clean. Nothing here is an out-of-sample estimate of forecasting skill.
- **The model is told the date and its own cut-off.** The command line adds an environment block with today's date and the model's knowledge cut-off, and a block with the account's email. They are the same on every call, so they do not affect comparisons, but they are not nothing (see `llm_predictor`).
- **Post-hoc text.** Sponsors rewrite outcome time frames when they post results (cut-off dates, achieved follow-up, termination). The blinded prompt cuts the clauses a rule finds; masked and identified prompts keep them. The scores without flagged trials are below. A rule cannot find every rewrite.
- **Self-reported recognition is weak evidence.** A model saying it recognises a trial is not proof of memory, and one saying it does not is not proof of its absence. A correct drug guess is stronger, but a design can make a drug easy to guess without any memory of the trial.
- **Not repeatable to the digit.** The command line has no temperature setting, so a repeat of the same batch gives somewhat different answers (see run-to-run below).
- **Batches of ten, frozen.** Each call holds ten trials, so a model sees neighbours. The batches and the order in them were fixed once, are mixed across the three periods, and are the same for every model and variant.
- **Arm descriptions.** The protocol record's arm text sometimes states a planned number of participants. That is a plan, not the final enrolment, which is never shown.
- **Intervals** resample trials as if independent; trials sharing a drug or sponsor are not.

## What this sample can resolve

| comparison on the 300 development trials | what the sample resolves |
|---|---|
| one AUROC, 300 trials | about plus or minus 0.05 |
| identified minus masked | can detect about 0.03 |
| identified minus blinded, or masked minus blinded | can detect about 0.06 to 0.08 |
| blinded minus registration | can detect about 0.08 to 0.10 |
| a subset of 100 to 200 trials against a baseline | about 0.10 or more |
| share recognised, by completion year | only differences of 15 to 20 points |

## Scores on the same 300 sampled trials

0 of 300 sampled trials are left out for lack of a successful prediction from every model and variant. Success rate in these trials: 53.0%. The two baselines are trained on development trials finishing before 2017. Calibration gap: mean predicted minus observed, in points; positive is too optimistic.

| model | AUROC (95% interval) | Brier | calibration gap, points |
|---|---|---|---|
| haiku (identified) | 0.877 (0.838 to 0.914) | 0.138 (0.117 to 0.160) | -1.0 (-5.1 to 3.1) |
| haiku (masked) | 0.860 (0.816 to 0.903) | 0.147 (0.124 to 0.170) | -0.8 (-4.9 to 3.3) |
| haiku (blinded) | 0.799 (0.751 to 0.848) | 0.181 (0.160 to 0.204) | 3.0 (-1.8 to 7.7) |
| sonnet (identified) | 0.916 (0.881 to 0.946) | 0.115 (0.095 to 0.136) | -0.5 (-4.1 to 3.4) |
| sonnet (masked) | 0.907 (0.872 to 0.940) | 0.121 (0.100 to 0.140) | -0.4 (-4.2 to 3.5) |
| sonnet (blinded) | 0.822 (0.773 to 0.868) | 0.172 (0.150 to 0.194) | 3.1 (-1.4 to 7.7) |
| logistic: registration | 0.716 (0.660 to 0.767) | 0.219 (0.200 to 0.241) | 6.4 (1.2 to 11.7) |
| ladder: registration + words | 0.749 (0.693 to 0.798) | 0.206 (0.186 to 0.228) | 5.3 (0.3 to 10.3) |

### Calibration

The baselines' over-confidence is mostly the fall in the success rate between the years they learn from and these: trials finishing before 2017 succeeded 61% of the time, these trials 53%.

Given the names, the language models are under-confident at the top of the scale (table below); blinded, they are about right. Calibration says nothing about recall versus reasoning.

| model | trials given 80% or more | mean probability given | share that succeeded |
|---|---|---|---|
| haiku (identified) | 94 | 90% | 94% |
| haiku (masked) | 89 | 90% | 92% |
| haiku (blinded) | 90 | 88% | 88% |
| sonnet (identified) | 115 | 88% | 96% |
| sonnet (masked) | 106 | 89% | 95% |
| sonnet (blinded) | 98 | 87% | 88% |
| logistic: registration | 48 | 85% | 88% |
| ladder: registration + words | 54 | 87% | 87% |

## Paired differences

Within the same resamples. The first rows are the contrasts between prompts. Identified minus blinded is recall plus the legitimate drug knowledge a forecaster has at the start (the drug's class, its earlier results, who the comparator is), not a floor on recall. Masked minus blinded is drug knowledge, known before the start and recalled afterwards, together. Identified minus masked is what the id, title, sponsor name, acronym and study ids add. The rest are against the two baselines: the blinded prompt, in which recall is reduced, not removed.

| pair | AUROC difference | Brier difference (negative is better) |
|---|---|---|
| haiku (identified) minus haiku (blinded) | +0.078 (+0.036 to +0.120) | −0.043 (−0.063 to −0.024) |
| haiku (masked) minus haiku (blinded) | +0.061 (+0.022 to +0.101) | −0.034 (−0.052 to −0.017) |
| haiku (identified) minus haiku (masked) | +0.017 (−0.017 to +0.049) | −0.009 (−0.025 to +0.007) |
| sonnet (identified) minus sonnet (blinded) | +0.094 (+0.056 to +0.134) | −0.057 (−0.077 to −0.038) |
| sonnet (masked) minus sonnet (blinded) | +0.085 (+0.048 to +0.123) | −0.051 (−0.070 to −0.032) |
| sonnet (identified) minus sonnet (masked) | +0.009 (−0.009 to +0.028) | −0.006 (−0.017 to +0.004) |
| haiku (identified) minus logistic: registration | +0.160 (+0.104 to +0.219) | −0.081 (−0.107 to −0.057) |
| haiku (identified) minus ladder: registration + words | +0.128 (+0.074 to +0.185) | −0.068 (−0.094 to −0.044) |
| haiku (masked) minus logistic: registration | +0.144 (+0.087 to +0.206) | −0.072 (−0.099 to −0.048) |
| haiku (masked) minus ladder: registration + words | +0.111 (+0.057 to +0.170) | −0.058 (−0.084 to −0.033) |
| haiku (blinded) minus logistic: registration | +0.082 (+0.026 to +0.142) | −0.038 (−0.061 to −0.016) |
| haiku (blinded) minus ladder: registration + words | +0.050 (−0.004 to +0.109) | −0.024 (−0.048 to −0.003) |
| sonnet (identified) minus logistic: registration | +0.199 (+0.145 to +0.256) | −0.104 (−0.128 to −0.080) |
| sonnet (identified) minus ladder: registration + words | +0.167 (+0.115 to +0.223) | −0.091 (−0.115 to −0.067) |
| sonnet (masked) minus logistic: registration | +0.191 (+0.133 to +0.246) | −0.098 (−0.122 to −0.074) |
| sonnet (masked) minus ladder: registration + words | +0.158 (+0.107 to +0.213) | −0.085 (−0.109 to −0.062) |
| sonnet (blinded) minus logistic: registration | +0.106 (+0.051 to +0.165) | −0.047 (−0.070 to −0.025) |
| sonnet (blinded) minus ladder: registration + words | +0.073 (+0.024 to +0.126) | −0.034 (−0.054 to −0.015) |

## The blinded prompt on trials the model could not name (exploratory)

Each model's blinded reply is sorted by whether it named the tested drug, under two definitions. "guess names the drug": its `drug_guess` matches a name, other name or company code of the tested intervention (case aside; a guess that is only a class word such as "monoclonal antibody" does not count, nor does a drug that a control arm takes too). "guess or reason names it": that, or the `reason` text holds such a name, or it claims a result and names a trial the prompt does not hold. "drug not named" includes a null guess. Each row is paired with both baselines on the same trials. This is an exploratory split, chosen by the model's own output; the intervals are wide because the subsets are small.

| definition of named | model, blinded prompt | subset | trials | success rate | AUROC | registration AUROC, same trials | difference | ladder AUROC, same trials | difference |
|---|---|---|---|---|---|---|---|---|---|
| guess names the drug | haiku | drug not named | 209 | 52% | 0.755 (0.689 to 0.819) | 0.693 (0.621 to 0.758) | +0.062 (−0.013 to +0.138) | 0.748 (0.677 to 0.808) | +0.007 (−0.063 to +0.072) |
| guess names the drug | haiku | drug named | 91 | 56% | 0.880 (0.798 to 0.944) | 0.763 (0.657 to 0.857) | +0.117 (+0.031 to +0.207) | 0.754 (0.648 to 0.847) | +0.126 (+0.035 to +0.216) |
| guess names the drug | sonnet | drug not named | 165 | 51% | 0.805 (0.742 to 0.867) | 0.687 (0.604 to 0.764) | +0.119 (+0.047 to +0.209) | 0.741 (0.673 to 0.814) | +0.064 (−0.006 to +0.139) |
| guess names the drug | sonnet | drug named | 135 | 56% | 0.843 (0.774 to 0.902) | 0.747 (0.665 to 0.828) | +0.096 (+0.013 to +0.175) | 0.754 (0.676 to 0.832) | +0.089 (+0.012 to +0.164) |
| guess or reason names it | haiku | drug not named | 204 | 52% | 0.756 (0.682 to 0.817) | 0.696 (0.625 to 0.760) | +0.060 (−0.015 to +0.137) | 0.750 (0.685 to 0.812) | +0.006 (−0.064 to +0.079) |
| guess or reason names it | haiku | drug named | 96 | 54% | 0.877 (0.799 to 0.939) | 0.755 (0.647 to 0.849) | +0.122 (+0.037 to +0.208) | 0.750 (0.648 to 0.842) | +0.127 (+0.044 to +0.223) |
| guess or reason names it | sonnet | drug not named | 150 | 49% | 0.807 (0.738 to 0.873) | 0.673 (0.588 to 0.760) | +0.134 (+0.051 to +0.219) | 0.726 (0.644 to 0.810) | +0.081 (+0.007 to +0.157) |
| guess or reason names it | sonnet | drug named | 150 | 57% | 0.838 (0.770 to 0.900) | 0.750 (0.673 to 0.822) | +0.089 (+0.008 to +0.174) | 0.767 (0.691 to 0.837) | +0.072 (−0.001 to +0.144) |

**The automatic match misses about one in six.** A reviewer read every blinded reply it calls not named, with a language model's help (a model-assisted hand read, not a computed figure). Of the 159 replies of Sonnet's that the match left (it now leaves 165, after the tightening described above), 28 did name the drug or trial: 16 by a synonym, brand or company code that is not in the alias list (nirsevimab for MEDI8897, erenumab for AMG334, melatonin for Circadin) and 12 in the reason text instead of the guess field. Of Haiku's 209, 12 did. So the not-named subset still contains trials the model identified. With those moved to the named group the reviewer found, against registration: Sonnet 0.807 against 0.690, +0.117 (+0.034 to +0.204), on 131 trials; Haiku +0.049 (−0.032 to +0.128) on 197 trials. Against the ladder: Sonnet about +0.057 (−0.019 to +0.135), Haiku about −0.014.

The stricter rule recomputes this, without the hand read:

- Haiku: it moves 5 trials to named (the reviewer: 12), leaving 204 not named (the reviewer: 197). Its edge over registration is +0.060 (−0.015 to +0.137) (the reviewer: +0.049 (−0.032 to +0.128)).
- Sonnet: it moves 15 trials to named (the reviewer: 28), leaving 150 not named (the reviewer: 131). Its edge over registration is +0.134 (+0.051 to +0.219) (the reviewer: +0.117 (+0.034 to +0.204)).

So the rule gets the edge about right but finds only about half of the trials the reviewer found. It catches an alias or a trial acronym in the reason; it cannot catch a synonym written in plain words.

### By whether the model also said it recognised the trial (stricter definition)

| model | drug | said it recognised the trial | trials | success rate | AUROC | registration AUROC, same trials | difference | ladder AUROC, same trials | difference |
|---|---|---|---|---|---|---|---|---|---|
| Haiku | drug not named | said no | 197 | 52% | 0.759 (0.687 to 0.819) | 0.702 (0.624 to 0.766) | +0.057 (−0.014 to +0.129) | 0.757 (0.687 to 0.821) | +0.002 (−0.064 to +0.068) |
| Haiku | drug not named | said yes | 7 | 57% | too few to score | too few to score | too few to score | too few to score | too few to score |
| Haiku | drug named | said no | 64 | 48% | 0.766 (0.639 to 0.875) | 0.749 (0.618 to 0.868) | +0.017 (−0.097 to +0.133) | 0.755 (0.633 to 0.866) | +0.011 (−0.103 to +0.122) |
| Haiku | drug named | said yes | 32 | 66% | 1.000 (1.000 to 1.000) | 0.818 (0.655 to 0.943) | +0.182 (+0.057 to +0.345) | 0.788 (0.599 to 0.933) | +0.212 (+0.067 to +0.401) |
| Sonnet | drug not named | said no | 111 | 45% | 0.799 (0.705 to 0.881) | 0.692 (0.579 to 0.784) | +0.107 (+0.009 to +0.213) | 0.766 (0.677 to 0.844) | +0.032 (−0.049 to +0.115) |
| Sonnet | drug not named | said yes | 39 | 62% | 0.787 (0.607 to 0.936) | 0.550 (0.340 to 0.745) | +0.237 (+0.060 to +0.418) | 0.553 (0.360 to 0.739) | +0.235 (+0.056 to +0.407) |
| Sonnet | drug named | said no | 60 | 52% | 0.758 (0.628 to 0.871) | 0.673 (0.532 to 0.810) | +0.085 (−0.037 to +0.207) | 0.702 (0.567 to 0.832) | +0.056 (−0.064 to +0.173) |
| Sonnet | drug named | said yes | 90 | 60% | 0.885 (0.805 to 0.945) | 0.810 (0.715 to 0.896) | +0.074 (−0.038 to +0.178) | 0.820 (0.731 to 0.902) | +0.065 (−0.031 to +0.154) |

- Haiku: the gap is against registration above zero in 3 of 3, clearly in 1; against the ladder above zero in 3 of 3, clearly in 1.
- Sonnet: the gap is against registration above zero in 4 of 4, clearly in 2; against the ladder above zero in 4 of 4, clearly in 1.
- Clearly above zero against both baselines in: Haiku, drug named, said yes (32 trials); Sonnet, drug not named, said yes (39 trials).

The reviewer's reading of Sonnet, from hand-corrected labels: the advantage does not concentrate in recognised trials. It is present everywhere and certain nowhere. In trials not named and not recognised Sonnet is +0.122 (+0.019 to +0.227) against registration on 100 trials, but about +0.040 (−0.053 to +0.133) against the ladder. Here the cells are small, and the gaps that clear zero against both baselines sit in cells of a few dozen trials.

### Trials neither model named (stricter definition)

| model, blinded prompt | trials | success rate | AUROC | registration AUROC, same trials | difference | ladder AUROC, same trials | difference |
|---|---|---|---|---|---|---|---|
| Haiku | 131 | 50% | 0.746 (0.653 to 0.819) | 0.666 (0.574 to 0.755) | +0.080 (−0.016 to +0.175) | 0.743 (0.657 to 0.820) | +0.003 (−0.080 to +0.083) |
| Sonnet | 131 | 50% | 0.799 (0.712 to 0.866) | 0.666 (0.574 to 0.755) | +0.133 (+0.039 to +0.224) | 0.743 (0.657 to 0.820) | +0.056 (−0.021 to +0.130) |

### How the not-named trials differ from the named ones (stricter definition)

| model | subset | trials | phase 3 | success rate | median enrolment | industry sponsor | mean completion year | registration AUROC on these trials |
|---|---|---|---|---|---|---|---|---|
| Haiku | drug not named | 204 | 48% | 52% | 191 | 85% | 2018.6 | 0.696 (0.625 to 0.760) |
| Haiku | drug named | 96 | 71% | 54% | 320 | 79% | 2018.6 | 0.755 (0.647 to 0.849) |
| Sonnet | drug not named | 150 | 46% | 49% | 194 | 85% | 2018.6 | 0.673 (0.588 to 0.760) |
| Sonnet | drug named | 150 | 64% | 57% | 246 | 81% | 2018.5 | 0.750 (0.673 to 0.822) |

The subset is chosen by the model's own output, is harder for the baseline, and is not free of recall.

### Replies that claim to know an outcome

A rule-based phrase list on the blinded reply's reason: recall, met its, did not meet, was positive, was negative, failed, reported, published (and a few close forms).

| model | blinded replies claiming an outcome | among the not-named (guess names the drug) | among the not-named (guess or reason names it) |
|---|---|---|---|
| Haiku | 40 of 300 | 16 of 209 | 14 of 204 |
| Sonnet | 53 of 300 | 20 of 165 | 10 of 150 |

What the confident, correct calls on not-named trials rest on was read by a reviewer, with a language model's help, in 21 of Sonnet's replies (a model-assisted read of 21, not a computed figure). About half rest on the drug class plus an easy endpoint against placebo (blood sugar, cholesterol, weight, haemoglobin, lung function) or on the design. About half rest on a remembered result of the same or a similar programme.

## Without the trials with a post-hoc phrase

18 of 300 scored trials have a post-hoc phrase in arm or outcome text (cut-off dates, achieved follow-up, termination). The same scores without them:

| model | AUROC (95% interval) | Brier | calibration gap, points |
|---|---|---|---|
| haiku (identified) | 0.867 (0.821 to 0.909) | 0.143 (0.120 to 0.168) | -1.1 (-5.6 to 3.3) |
| haiku (masked) | 0.865 (0.820 to 0.909) | 0.146 (0.123 to 0.168) | 0.0 (-4.5 to 4.4) |
| haiku (blinded) | 0.799 (0.748 to 0.849) | 0.182 (0.159 to 0.203) | 3.5 (-1.5 to 8.3) |
| sonnet (identified) | 0.907 (0.871 to 0.940) | 0.120 (0.100 to 0.140) | -0.2 (-4.2 to 3.7) |
| sonnet (masked) | 0.904 (0.868 to 0.938) | 0.124 (0.103 to 0.144) | 0.0 (-3.9 to 4.2) |
| sonnet (blinded) | 0.826 (0.775 to 0.870) | 0.170 (0.149 to 0.194) | 3.4 (-1.5 to 8.0) |
| logistic: registration | 0.713 (0.645 to 0.772) | 0.221 (0.199 to 0.244) | 6.6 (1.3 to 12.3) |
| ladder: registration + words | 0.740 (0.678 to 0.793) | 0.209 (0.188 to 0.233) | 5.4 (0.1 to 11.0) |

| pair | AUROC difference | Brier difference (negative is better) |
|---|---|---|
| haiku (identified) minus haiku (blinded) | +0.068 (+0.024 to +0.109) | −0.039 (−0.060 to −0.015) |
| haiku (masked) minus haiku (blinded) | +0.065 (+0.028 to +0.108) | −0.036 (−0.055 to −0.018) |
| sonnet (identified) minus sonnet (blinded) | +0.081 (+0.043 to +0.124) | −0.051 (−0.072 to −0.031) |
| sonnet (masked) minus sonnet (blinded) | +0.078 (+0.042 to +0.118) | −0.046 (−0.066 to −0.028) |

Trials flagged, by period (all sampled trials):

| period | sampled trials | with a post-hoc phrase |
|---|---|---|
| 2017-2020 | 300 | 18 |
| 2021-2023 | 300 | 11 |
| 2024-2025 | 300 | 5 |

## By whether the model said it recognised the trial

The model's own answer to "do you know this specific trial or its result?", prompt variant as in the row. Many models say yes to most trials, which makes the yes group most of the sample.

| model | said yes: trials | success rate | AUROC | said no: trials | success rate | AUROC |
|---|---|---|---|---|---|---|
| haiku (identified) | 145 | 59% | 0.907 (0.857 to 0.951) | 155 | 48% | 0.865 (0.795 to 0.919) |
| haiku (masked) | 125 | 62% | 0.902 (0.844 to 0.954) | 175 | 47% | 0.833 (0.765 to 0.891) |
| haiku (blinded) | 39 | 64% | 0.969 (0.910 to 1.000) | 261 | 51% | 0.760 (0.696 to 0.815) |
| sonnet (identified) | 203 | 57% | 0.940 (0.905 to 0.972) | 97 | 44% | 0.839 (0.749 to 0.915) |
| sonnet (masked) | 211 | 56% | 0.934 (0.896 to 0.966) | 89 | 46% | 0.833 (0.739 to 0.912) |
| sonnet (blinded) | 129 | 60% | 0.857 (0.788 to 0.915) | 171 | 47% | 0.784 (0.717 to 0.853) |

## How often the model said it recognised the trial, and named the drug

| model | share it said it recognised | share where its drug guess names the tested drug |
|---|---|---|
| haiku (identified) | 48% | 95% |
| haiku (masked) | 42% | 96% |
| haiku (blinded) | 13% | 30% |
| sonnet (identified) | 68% | 96% |
| sonnet (masked) | 70% | 97% |
| sonnet (blinded) | 43% | 45% |

## Share recognised, by completion year

| completion year | trials | haiku (identified) | haiku (masked) | haiku (blinded) | sonnet (identified) | sonnet (masked) | sonnet (blinded) |
|---|---|---|---|---|---|---|---|
| 2017 | 65 | 51% | 45% | 8% | 69% | 68% | 48% |
| 2018 | 71 | 42% | 34% | 11% | 68% | 73% | 41% |
| 2019 | 85 | 49% | 42% | 18% | 67% | 69% | 45% |
| 2020 | 79 | 51% | 46% | 14% | 67% | 71% | 39% |

## Leak check: what blinding leaves behind

Name removal cannot blind a model to a trial it has read about. Three pieces of evidence:

- **Leak check.** After blinding, 27 of 300 sampled development trials (9%) still show a word or company code from one of their own intervention names or other names. That undercounts: it cannot see an abbreviation ("MF" for mometasone), a brand or device name that was never listed, or a programme recognised from its design.
- **Hand read.** Reading 30 blinded prompts (`results/llm_blinded_read.csv`, made with an earlier version of the blinding rules), the builder found the drug obvious in 13 and partly in 5. A reviewer reading a fresh 30 found it obvious in about 15 and partly in 8.
- **The model's own guess.** On blinded prompts the model's `drug_guess` names the tested drug in: haiku (blinded) 30%, sonnet (blinded) 45%.

## Run-to-run variation

| model | trials asked twice | correlation | mean absolute change | recognised flipped |
|---|---|---|---|---|
| haiku (blinded) | 4 | 1.00 | 0.013 | 0% |
| sonnet (blinded) | 4 | 0.98 | 0.037 | 0% |
