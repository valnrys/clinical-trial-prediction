# Models on the development trials

Development trials only (completed before 2021). No test or late trial is used.

**Read this first.** With about 2,795 validation trials, an AUROC difference under about 0.01 is a tie. Intervals are bootstrap intervals that resample whole lead drugs; trials that share a sponsor are not independent, so they are somewhat too narrow. Every difference is against "logistic: registration", on the same resamples. Calibration gap: mean predicted minus observed success rate, in points; positive means too optimistic.

The AUROC shown is the mean of the AUROCs within each round (or fold), and its interval resamples lead drugs inside each round. Each round's model scores on its own scale, so ranking all rounds together would reward or punish a model for how its scores shift between rounds rather than for how well it ranks trials. Brier score and calibration gap are averages over trials, so they are pooled.

## The models

- **lookup: constant** (honest): Says the training success rate for every trial; knows nothing about the trial.
- **lookup: phase** (honest): Says the training success rate of the trial's phase.
- **lookup: phase x sponsor class** (honest): Says the training success rate of the trial's phase and sponsor class, pulled towards the phase rate when few trials share the cell.
- **logistic: registration** (honest): The pre-registered baseline: logistic regression on what is known at registration.
- **logistic: registration + interactions** (honest): The registration logistic plus phase-by-sponsor and phase-by-area terms.
- **logistic: registration, tuned penalty** (honest): The registration logistic with penalty strength and L1/L2 mix chosen on an inner split.
- **boosted trees: registration** (honest): Gradient-boosted shallow trees on the registration columns, fixed settings.
- **random forest: registration** (honest): A random forest of 500 trees on the registration columns, fixed settings.
- **boosted trees: registration, tuned** (honest): Boosted trees on the registration columns, depth and step size chosen on an inner split.
- **neural network: registration** (honest): A small two-layer neural network on the registration columns; training length chosen on an inner split.
- **logistic: last 4 years only** (honest): The registration logistic fitted on only the newest 4 years of training trials.
- **logistic: last 8 years only** (honest): The registration logistic fitted on only the newest 8 years of training trials.
- **logistic: recency-weighted** (honest): The registration logistic with newer training trials counting more (half-life 4 years).
- **logistic: trend-shifted intercept** (honest): The registration logistic, shifted to the success rate projected from the yearly trend.
- **logistic: separate for company and other sponsors** (honest): One registration logistic for industry-sponsored trials and another for all the rest.
- **logistic: separate by phase** (honest): One registration logistic for Phase 2 trials and another for Phase 3.
- **memory: lead sponsor** (honest): Says the lead sponsor's success rate in the training trials (shrunk towards the rate of trials that have one), nothing else.
- **memory: lead drug** (NOT honest): Says the lead drug's success rate in the training trials (shrunk towards the rate of trials that have one), nothing else.
- **logistic: registration + pooled sponsor** (honest): The registration logistic plus the lead sponsor's shrunk success rate in trials that finished in earlier years.
- **ladder: registration + endpoint** (honest): The registration logistic plus features of the primary endpoint.
- **ladder: registration + history** (honest): The registration logistic plus the sponsor's and drug's record before the trial.
- **ladder: registration + genetics** (honest): The registration logistic plus genetic evidence dated before the trial started.
- **ladder: registration + words** (honest): The registration logistic plus words of the title, conditions and primary outcomes.
- **ladder: registration + endpoint + history + genetics + words** (honest): The registration logistic plus all four tables, honest columns only.
- **ladder: boosted trees, endpoint + history + genetics** (honest): Boosted trees on the registration columns plus endpoint, history and genetics, honest columns only.
- **ladder: all four tables plus after-the-start columns (not honest)** (NOT honest): The all-tables logistic that also reads columns, and the summary, that may include later knowledge.
- **text: words** (honest): Logistic regression on words and two-word phrases of the title, conditions and primary outcomes.
- **text: drug names only** (honest): Logistic regression on the intervention names alone: it can only work by recognising drugs.
- **text: words without drug names** (honest): As the words model, with each trial's own drug names removed from its text.
- **text: words with summary (leaky)** (NOT honest): The words model plus the brief summary, which sponsors can rewrite with results.
- **embeddings: general vector alone** (honest, uses a pretrained model): Logistic regression on the general sentence vector of the registration text.
- **embeddings: general vector plus registration** (honest, uses a pretrained model): Registration columns beside the general sentence vector, then a logistic regression.
- **embeddings: general vector plus registration plus words** (honest, uses a pretrained model): Registration columns, the general sentence vector and word features, then a logistic regression.
- **embeddings: biomedical vector alone** (honest, uses a pretrained model): Logistic regression on the biomedical sentence vector of the registration text.
- **embeddings: biomedical vector plus registration** (honest, uses a pretrained model): Registration columns beside the biomedical sentence vector, then a logistic regression.
- **embeddings: biomedical vector plus registration plus words** (honest, uses a pretrained model): Registration columns, the biomedical sentence vector and word features, then a logistic regression.
- **biology: logistic, all sources** (honest): Logistic regression on the science columns alone (no registration columns). Defined on trials with a mapped target; the others get one constant.
- **biology: boosted trees, all sources** (honest): Boosted trees on the science columns alone (no registration columns). Defined on trials with a mapped target; the others get one constant.
- **biology: logistic, DepMap only** (honest): Logistic regression on the DepMap columns alone. Defined on trials with a mapped target; the others get one constant.
- **biology: logistic, expression only** (honest): Logistic regression on the expression columns alone. Defined on trials with a mapped target; the others get one constant.
- **biology: logistic, networks only** (honest): Logistic regression on the networks columns alone. Defined on trials with a mapped target; the others get one constant.
- **biology: logistic, annotations only** (honest): Logistic regression on the annotations columns alone. Defined on trials with a mapped target; the others get one constant.
- **biology: logistic, chemistry only** (honest): Logistic regression on the chemistry columns alone. Defined on trials with a mapped target; the others get one constant.
- **biology: registration + DepMap** (honest): The registration logistic plus the DepMap columns.
- **biology: registration + expression** (honest): The registration logistic plus the expression columns.
- **biology: registration + networks** (honest): The registration logistic plus the networks columns.
- **biology: registration + annotations** (honest): The registration logistic plus the annotations columns.
- **biology: registration + chemistry** (honest): The registration logistic plus the chemistry columns.
- **biology: registration + all sources** (honest): The registration logistic plus every science column.
- **biology: ladder, all four tables + science** (honest): The information ladder's all-tables logistic plus every science column.
- **llm: haiku (identified)** (honest, uses a pretrained model): Claude Haiku reads the protocol record with the registry id, title and sponsor name, then gives a probability. Defined on the sampled trials only; the others get one constant. Recall is not bounded: the model may have read about the trial.
- **llm: haiku (masked)** (honest, uses a pretrained model): Claude Haiku reads the protocol record with no registry id, title or sponsor name (the sponsor class only), then gives a probability. Defined on the sampled trials only; the others get one constant. Recall is not bounded: the model may have read about the trial.
- **llm: haiku (blinded)** (honest, uses a pretrained model): Claude Haiku reads the protocol record with no registry id, title, sponsor name or start date, and every intervention name replaced by a placeholder, then gives a probability. Defined on the sampled trials only; the others get one constant. Recall is not bounded: the model may have read about the trial.
- **llm: sonnet (identified)** (honest, uses a pretrained model): Claude Sonnet reads the protocol record with the registry id, title and sponsor name, then gives a probability. Defined on the sampled trials only; the others get one constant. Recall is not bounded: the model may have read about the trial.
- **llm: sonnet (masked)** (honest, uses a pretrained model): Claude Sonnet reads the protocol record with no registry id, title or sponsor name (the sponsor class only), then gives a probability. Defined on the sampled trials only; the others get one constant. Recall is not bounded: the model may have read about the trial.
- **llm: sonnet (blinded)** (honest, uses a pretrained model): Claude Sonnet reads the protocol record with no registry id, title, sponsor name or start date, and every intervention name replaced by a placeholder, then gives a probability. Defined on the sampled trials only; the others get one constant. Recall is not bounded: the model may have read about the trial.
- **logistic: registration + precedent (today's links)** (NOT honest): The registration logistic plus one yes/no column: an earlier Phase 3 trial of another molecule on the same target in the same disease (today's links).
- **logistic: registration + precedent (2020 links)** (NOT honest): The registration logistic plus one yes/no column: an earlier Phase 3 trial of another molecule on the same target in the same disease (2020 links).
- **logistic: registration + precedent (2020 names and links)** (NOT honest): The registration logistic plus one yes/no column: an earlier Phase 3 trial of another molecule on the same target in the same disease (2020 names and links).
- **logistic: registration + precedent (today's links, experimental arms only)** (NOT honest): The registration logistic plus one yes/no column: an earlier Phase 3 trial of another molecule on the same target in the same disease (today's links; experimental-arm drugs on both sides).
- **logistic: registration + precedent (2020 links, experimental arms only)** (NOT honest): The registration logistic plus one yes/no column: an earlier Phase 3 trial of another molecule on the same target in the same disease (2020 links; experimental-arm drugs on both sides).
- **logistic: registration + precedent (2020 names and links, experimental arms only)** (NOT honest): The registration logistic plus one yes/no column: an earlier Phase 3 trial of another molecule on the same target in the same disease (2020 names and links; experimental-arm drugs on both sides).
- **leaky: registration + post-hoc fields** (NOT honest): The registration logistic plus fields that are edited after the trial ends.
- **leaky: registration + results structure** (NOT honest): The registration logistic plus arm count and control kind, read from posted results.

Not ranked in any table below (not meaningful on all trials: they predict one constant where a trial has no mapped target; scored on mapped trials in `results/dev_science.md`): biology: logistic, all sources, biology: boosted trees, all sources, biology: logistic, DepMap only, biology: logistic, expression only, biology: logistic, networks only, biology: logistic, annotations only, biology: logistic, chemistry only.

Not ranked in any table below (predictions exist only for the sampled trials and every other trial gets one constant; scored on the sampled development trials in `results/dev_llm.md`): llm: haiku (identified), llm: haiku (masked), llm: haiku (blinded), llm: sonnet (identified), llm: sonnet (masked), llm: sonnet (blinded).

## Time-ordered check (2795 validation trials, three rounds)

Rounds: train before 2015, score 2015-2016; train before 2017, score 2017-2018; train before 2019, score 2019-2020.

### Honest models

| model | mean AUROC (95% interval) | AUROC minus registration | Brier | calibration gap (pts) |
|---|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.745 (0.723 to 0.765) | +0.036 (+0.024 to +0.049) | 0.206 | +2.7 |
| biology: ladder, all four tables + science | 0.743 (0.720 to 0.764) | +0.034 (+0.020 to +0.047) | 0.207 | +2.4 |
| ladder: registration + words | 0.739 (0.717 to 0.760) | +0.030 (+0.018 to +0.043) | 0.209 | +3.8 |
| embeddings: biomedical vector plus registration plus words | 0.738 (0.717 to 0.759) | +0.029 (+0.013 to +0.049) | 0.210 | +3.4 |
| ladder: boosted trees, endpoint + history + genetics | 0.735 (0.714 to 0.755) | +0.026 (+0.016 to +0.037) | 0.209 | +3.1 |
| embeddings: biomedical vector plus registration | 0.730 (0.708 to 0.753) | +0.021 (+0.005 to +0.040) | 0.213 | +3.4 |
| embeddings: general vector plus registration plus words | 0.728 (0.706 to 0.749) | +0.019 (+0.002 to +0.037) | 0.215 | +3.7 |
| boosted trees: registration, tuned | 0.719 (0.697 to 0.741) | +0.011 (+0.002 to +0.019) | 0.215 | +4.3 |
| ladder: registration + endpoint | 0.718 (0.695 to 0.740) | +0.010 (+0.003 to +0.017) | 0.215 | +2.2 |
| boosted trees: registration | 0.718 (0.695 to 0.741) | +0.009 (+0.002 to +0.017) | 0.216 | +4.2 |
| text: words | 0.718 (0.693 to 0.741) | +0.009 (−0.012 to +0.031) | 0.216 | +5.4 |
| neural network: registration | 0.718 (0.695 to 0.741) | +0.009 (+0.004 to +0.015) | 0.216 | +4.4 |
| random forest: registration | 0.717 (0.695 to 0.740) | +0.008 (−0.001 to +0.018) | 0.215 | +4.3 |
| text: words without drug names | 0.716 (0.691 to 0.740) | +0.007 (−0.014 to +0.029) | 0.217 | +5.3 |
| logistic: registration + interactions | 0.713 (0.691 to 0.734) | +0.004 (−0.000 to +0.010) | 0.217 | +4.0 |
| logistic: registration + pooled sponsor | 0.712 (0.691 to 0.734) | +0.004 (+0.000 to +0.007) | 0.218 | +3.9 |
| logistic: separate for company and other sponsors | 0.712 (0.690 to 0.735) | +0.004 (−0.003 to +0.011) | 0.218 | +4.3 |
| embeddings: biomedical vector alone | 0.711 (0.690 to 0.734) | +0.003 (−0.017 to +0.024) | 0.219 | +4.1 |
| ladder: registration + genetics | 0.711 (0.688 to 0.733) | +0.003 (−0.001 to +0.006) | 0.218 | +4.0 |
| logistic: separate by phase | 0.711 (0.690 to 0.733) | +0.003 (−0.004 to +0.009) | 0.218 | +4.2 |
| ladder: registration + history | 0.711 (0.688 to 0.733) | +0.002 (−0.001 to +0.005) | 0.218 | +3.6 |
| biology: registration + expression | 0.710 (0.687 to 0.732) | +0.001 (−0.004 to +0.005) | 0.218 | +3.6 |
| logistic: recency-weighted | 0.709 (0.686 to 0.732) | +0.000 (−0.003 to +0.003) | 0.219 | +3.4 |
| biology: registration + networks | 0.709 (0.686 to 0.731) | +0.000 (−0.003 to +0.003) | 0.219 | +3.7 |
| logistic: registration | 0.709 (0.686 to 0.731) | (reference) | 0.219 | +3.8 |
| logistic: trend-shifted intercept | 0.709 (0.686 to 0.731) | +0.000 (+0.000 to +0.000) | 0.217 | −1.2 |
| logistic: registration, tuned penalty | 0.709 (0.686 to 0.730) | −0.000 (−0.005 to +0.005) | 0.219 | +3.9 |
| biology: registration + DepMap | 0.708 (0.686 to 0.730) | −0.000 (−0.003 to +0.002) | 0.219 | +3.9 |
| embeddings: general vector plus registration | 0.708 (0.684 to 0.732) | −0.001 (−0.017 to +0.016) | 0.223 | +3.6 |
| biology: registration + all sources | 0.707 (0.683 to 0.731) | −0.002 (−0.009 to +0.006) | 0.219 | +3.5 |
| logistic: last 4 years only | 0.707 (0.682 to 0.729) | −0.002 (−0.008 to +0.003) | 0.219 | +3.1 |
| logistic: last 8 years only | 0.706 (0.683 to 0.728) | −0.002 (−0.005 to +0.000) | 0.219 | +3.6 |
| biology: registration + annotations | 0.706 (0.683 to 0.728) | −0.003 (−0.006 to +0.001) | 0.219 | +3.7 |
| biology: registration + chemistry | 0.705 (0.683 to 0.727) | −0.004 (−0.008 to +0.001) | 0.220 | +3.7 |
| embeddings: general vector alone | 0.682 (0.658 to 0.707) | −0.026 (−0.048 to −0.004) | 0.230 | +4.7 |
| lookup: phase x sponsor class | 0.666 (0.644 to 0.688) | −0.043 (−0.059 to −0.025) | 0.227 | +5.4 |
| memory: lead sponsor | 0.626 (0.599 to 0.653) | −0.083 (−0.111 to −0.056) | 0.242 | +7.6 |
| lookup: phase | 0.610 (0.590 to 0.630) | −0.099 (−0.119 to −0.079) | 0.238 | +5.2 |
| text: drug names only | 0.607 (0.582 to 0.631) | −0.102 (−0.128 to −0.073) | 0.242 | +4.7 |
| lookup: constant | 0.500 (0.500 to 0.500) | −0.209 (−0.231 to −0.186) | 0.252 | +6.6 |

### Not honest (deliberately leaky)

| model | mean AUROC (95% interval) | AUROC minus registration | Brier | calibration gap (pts) |
|---|---|---|---|---|
| ladder: all four tables plus after-the-start columns (not honest) | 0.749 (0.727 to 0.770) | +0.040 (+0.028 to +0.053) | 0.204 | +2.5 |
| leaky: registration + post-hoc fields | 0.737 (0.716 to 0.757) | +0.029 (+0.019 to +0.039) | 0.211 | +3.6 |
| text: words with summary (leaky) | 0.729 (0.705 to 0.751) | +0.020 (+0.000 to +0.041) | 0.214 | +5.7 |
| logistic: registration + precedent (today's links) | 0.724 (0.701 to 0.745) | +0.015 (+0.008 to +0.021) | 0.214 | +3.9 |
| logistic: registration + precedent (today's links, experimental arms only) | 0.719 (0.697 to 0.741) | +0.010 (+0.005 to +0.016) | 0.215 | +3.8 |
| logistic: registration + precedent (2020 links) | 0.718 (0.696 to 0.739) | +0.009 (+0.003 to +0.016) | 0.215 | +3.4 |
| logistic: registration + precedent (2020 links, experimental arms only) | 0.716 (0.693 to 0.738) | +0.007 (+0.002 to +0.014) | 0.216 | +3.5 |
| logistic: registration + precedent (2020 names and links) | 0.716 (0.693 to 0.737) | +0.007 (+0.002 to +0.014) | 0.216 | +3.5 |
| logistic: registration + precedent (2020 names and links, experimental arms only) | 0.716 (0.693 to 0.736) | +0.007 (+0.001 to +0.013) | 0.216 | +3.4 |
| leaky: registration + results structure | 0.712 (0.690 to 0.734) | +0.004 (−0.002 to +0.009) | 0.218 | +3.2 |
| memory: lead drug | 0.638 (0.612 to 0.663) | −0.071 (−0.098 to −0.042) | 0.242 | +8.2 |

### AUROC within each round

#### Honest models

| model | 2015-2016 | 2017-2018 | 2019-2020 |
|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.727 | 0.771 | 0.738 |
| biology: ladder, all four tables + science | 0.727 | 0.763 | 0.739 |
| ladder: registration + words | 0.727 | 0.761 | 0.729 |
| embeddings: biomedical vector plus registration plus words | 0.733 | 0.755 | 0.726 |
| ladder: boosted trees, endpoint + history + genetics | 0.723 | 0.747 | 0.735 |
| embeddings: biomedical vector plus registration | 0.727 | 0.745 | 0.717 |
| embeddings: general vector plus registration plus words | 0.715 | 0.758 | 0.711 |
| boosted trees: registration, tuned | 0.715 | 0.729 | 0.714 |
| ladder: registration + endpoint | 0.702 | 0.742 | 0.712 |
| boosted trees: registration | 0.710 | 0.728 | 0.716 |
| text: words | 0.713 | 0.737 | 0.703 |
| neural network: registration | 0.707 | 0.733 | 0.714 |
| random forest: registration | 0.707 | 0.730 | 0.714 |
| text: words without drug names | 0.709 | 0.736 | 0.703 |
| logistic: registration + interactions | 0.707 | 0.729 | 0.704 |
| logistic: registration + pooled sponsor | 0.704 | 0.728 | 0.705 |
| logistic: separate for company and other sponsors | 0.705 | 0.725 | 0.707 |
| embeddings: biomedical vector alone | 0.719 | 0.718 | 0.698 |
| ladder: registration + genetics | 0.701 | 0.732 | 0.701 |
| logistic: separate by phase | 0.697 | 0.728 | 0.709 |
| ladder: registration + history | 0.702 | 0.727 | 0.704 |
| biology: registration + expression | 0.702 | 0.725 | 0.701 |
| logistic: recency-weighted | 0.702 | 0.727 | 0.699 |
| biology: registration + networks | 0.702 | 0.723 | 0.702 |
| logistic: registration | 0.700 | 0.727 | 0.700 |
| logistic: trend-shifted intercept | 0.700 | 0.727 | 0.700 |
| logistic: registration, tuned penalty | 0.702 | 0.729 | 0.695 |
| biology: registration + DepMap | 0.701 | 0.724 | 0.700 |
| embeddings: general vector plus registration | 0.693 | 0.740 | 0.690 |
| biology: registration + all sources | 0.704 | 0.719 | 0.699 |
| logistic: last 4 years only | 0.702 | 0.726 | 0.692 |
| logistic: last 8 years only | 0.701 | 0.723 | 0.695 |
| biology: registration + annotations | 0.696 | 0.726 | 0.696 |
| biology: registration + chemistry | 0.698 | 0.722 | 0.695 |
| embeddings: general vector alone | 0.676 | 0.708 | 0.663 |
| lookup: phase x sponsor class | 0.662 | 0.679 | 0.657 |
| memory: lead sponsor | 0.624 | 0.637 | 0.617 |
| lookup: phase | 0.604 | 0.626 | 0.600 |
| text: drug names only | 0.618 | 0.598 | 0.604 |
| lookup: constant | 0.500 | 0.500 | 0.500 |

#### Not honest (deliberately leaky)

| model | 2015-2016 | 2017-2018 | 2019-2020 |
|---|---|---|---|
| ladder: all four tables plus after-the-start columns (not honest) | 0.731 | 0.773 | 0.742 |
| leaky: registration + post-hoc fields | 0.732 | 0.746 | 0.734 |
| text: words with summary (leaky) | 0.728 | 0.747 | 0.710 |
| logistic: registration + precedent (today's links) | 0.714 | 0.745 | 0.712 |
| logistic: registration + precedent (today's links, experimental arms only) | 0.709 | 0.739 | 0.709 |
| logistic: registration + precedent (2020 links) | 0.713 | 0.737 | 0.704 |
| logistic: registration + precedent (2020 links, experimental arms only) | 0.710 | 0.734 | 0.704 |
| logistic: registration + precedent (2020 names and links) | 0.709 | 0.737 | 0.702 |
| logistic: registration + precedent (2020 names and links, experimental arms only) | 0.708 | 0.735 | 0.703 |
| leaky: registration + results structure | 0.708 | 0.736 | 0.693 |
| memory: lead drug | 0.629 | 0.660 | 0.624 |

## Grouped cross-validation (5849 dev trials, 10 folds)

No drug is on both sides of a fold, so every validation drug is unseen and a model that memorises drugs gets nothing from remembering. But the split ignores time: a model may be trained on trials that finished after the ones it predicts, which a forecaster cannot do. That helps any model that learns from later trials of the same condition or sponsor, so scores here can run higher than in the time-ordered check. The drift models are not meaningful here: with no time cut, their "most recent years" are just the newest years of the training folds.

### Honest models

| model | mean AUROC (95% interval) | AUROC minus registration | Brier | calibration gap (pts) |
|---|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.759 (0.741 to 0.774) | +0.032 (+0.023 to +0.042) | 0.196 | −0.2 |
| biology: ladder, all four tables + science | 0.756 (0.738 to 0.771) | +0.029 (+0.019 to +0.039) | 0.197 | −0.2 |
| ladder: registration + words | 0.754 (0.735 to 0.769) | +0.027 (+0.018 to +0.036) | 0.198 | −0.3 |
| ladder: boosted trees, endpoint + history + genetics | 0.752 (0.734 to 0.767) | +0.025 (+0.018 to +0.033) | 0.198 | −0.1 |
| embeddings: biomedical vector plus registration plus words | 0.748 (0.730 to 0.761) | +0.021 (+0.008 to +0.033) | 0.201 | −0.4 |
| embeddings: biomedical vector plus registration | 0.743 (0.726 to 0.757) | +0.016 (+0.005 to +0.028) | 0.203 | −0.1 |
| embeddings: general vector plus registration plus words | 0.739 (0.719 to 0.755) | +0.012 (−0.001 to +0.025) | 0.204 | −0.5 |
| ladder: registration + endpoint | 0.738 (0.719 to 0.753) | +0.011 (+0.006 to +0.016) | 0.204 | −0.1 |
| boosted trees: registration, tuned | 0.736 (0.717 to 0.751) | +0.009 (+0.005 to +0.014) | 0.204 | −0.1 |
| boosted trees: registration | 0.736 (0.717 to 0.751) | +0.009 (+0.005 to +0.014) | 0.203 | −0.1 |
| random forest: registration | 0.733 (0.714 to 0.749) | +0.007 (+0.000 to +0.013) | 0.204 | −0.1 |
| neural network: registration | 0.733 (0.714 to 0.747) | +0.006 (+0.001 to +0.010) | 0.205 | −0.8 |
| logistic: separate for company and other sponsors | 0.731 (0.712 to 0.746) | +0.004 (+0.000 to +0.009) | 0.205 | −0.1 |
| text: words | 0.730 (0.712 to 0.746) | +0.004 (−0.011 to +0.018) | 0.206 | −0.3 |
| logistic: registration + pooled sponsor | 0.730 (0.711 to 0.746) | +0.003 (+0.001 to +0.006) | 0.206 | +0.0 |
| text: words without drug names | 0.730 (0.711 to 0.746) | +0.003 (−0.012 to +0.018) | 0.206 | −0.3 |
| logistic: registration + interactions | 0.729 (0.711 to 0.744) | +0.002 (−0.001 to +0.006) | 0.206 | −0.1 |
| ladder: registration + history | 0.728 (0.710 to 0.743) | +0.002 (−0.001 to +0.004) | 0.207 | −0.1 |
| logistic: registration, tuned penalty | 0.728 (0.709 to 0.743) | +0.001 (−0.001 to +0.004) | 0.207 | −0.0 |
| logistic: separate by phase | 0.728 (0.709 to 0.743) | +0.001 (−0.003 to +0.005) | 0.206 | −0.1 |
| ladder: registration + genetics | 0.728 (0.709 to 0.743) | +0.001 (−0.001 to +0.003) | 0.207 | −0.1 |
| biology: registration + expression | 0.728 (0.709 to 0.744) | +0.001 (−0.003 to +0.004) | 0.207 | −0.1 |
| logistic: registration | 0.727 (0.708 to 0.742) | (reference) | 0.207 | −0.1 |
| logistic: trend-shifted intercept | 0.727 (0.708 to 0.742) | +0.000 (+0.000 to +0.000) | 0.217 | −10.2 |
| biology: registration + networks | 0.726 (0.708 to 0.741) | −0.000 (−0.002 to +0.001) | 0.207 | −0.1 |
| biology: registration + DepMap | 0.726 (0.708 to 0.742) | −0.001 (−0.002 to +0.001) | 0.207 | −0.1 |
| logistic: recency-weighted | 0.725 (0.707 to 0.741) | −0.001 (−0.004 to +0.001) | 0.208 | −1.7 |
| biology: registration + chemistry | 0.725 (0.706 to 0.740) | −0.002 (−0.005 to +0.001) | 0.207 | −0.1 |
| biology: registration + all sources | 0.724 (0.705 to 0.740) | −0.003 (−0.008 to +0.002) | 0.208 | −0.0 |
| logistic: last 8 years only | 0.724 (0.705 to 0.738) | −0.003 (−0.006 to −0.000) | 0.208 | −1.3 |
| biology: registration + annotations | 0.723 (0.705 to 0.739) | −0.004 (−0.006 to −0.001) | 0.208 | −0.1 |
| embeddings: biomedical vector alone | 0.723 (0.704 to 0.738) | −0.004 (−0.018 to +0.011) | 0.209 | −0.3 |
| embeddings: general vector plus registration | 0.722 (0.703 to 0.737) | −0.005 (−0.017 to +0.007) | 0.210 | −0.4 |
| logistic: last 4 years only | 0.720 (0.702 to 0.735) | −0.006 (−0.011 to −0.002) | 0.210 | −3.4 |
| embeddings: general vector alone | 0.687 (0.667 to 0.704) | −0.040 (−0.055 to −0.023) | 0.221 | −0.5 |
| lookup: phase x sponsor class | 0.670 (0.655 to 0.684) | −0.057 (−0.067 to −0.043) | 0.218 | +0.2 |
| memory: lead sponsor | 0.633 (0.612 to 0.653) | −0.094 (−0.113 to −0.073) | 0.231 | +1.4 |
| lookup: phase | 0.612 (0.598 to 0.624) | −0.115 (−0.128 to −0.097) | 0.231 | −0.0 |
| text: drug names only | 0.597 (0.579 to 0.614) | −0.130 (−0.148 to −0.108) | 0.238 | −3.1 |
| lookup: constant | 0.500 (0.500 to 0.500) | −0.227 (−0.242 to −0.208) | 0.243 | +0.0 |

### Not honest (deliberately leaky)

| model | mean AUROC (95% interval) | AUROC minus registration | Brier | calibration gap (pts) |
|---|---|---|---|---|
| ladder: all four tables plus after-the-start columns (not honest) | 0.762 (0.744 to 0.778) | +0.035 (+0.025 to +0.045) | 0.195 | −0.2 |
| leaky: registration + post-hoc fields | 0.755 (0.737 to 0.769) | +0.028 (+0.021 to +0.035) | 0.198 | −0.1 |
| text: words with summary (leaky) | 0.740 (0.721 to 0.756) | +0.013 (−0.001 to +0.027) | 0.203 | −0.5 |
| logistic: registration + precedent (today's links) | 0.737 (0.719 to 0.752) | +0.010 (+0.005 to +0.016) | 0.204 | −0.1 |
| logistic: registration + precedent (2020 links) | 0.735 (0.716 to 0.749) | +0.008 (+0.003 to +0.013) | 0.204 | −0.1 |
| logistic: registration + precedent (today's links, experimental arms only) | 0.734 (0.715 to 0.749) | +0.007 (+0.003 to +0.012) | 0.205 | −0.1 |
| leaky: registration + results structure | 0.734 (0.715 to 0.750) | +0.007 (+0.003 to +0.011) | 0.205 | −0.1 |
| logistic: registration + precedent (2020 names and links) | 0.734 (0.715 to 0.749) | +0.007 (+0.003 to +0.012) | 0.205 | −0.1 |
| logistic: registration + precedent (2020 names and links, experimental arms only) | 0.733 (0.715 to 0.749) | +0.006 (+0.002 to +0.011) | 0.205 | −0.1 |
| logistic: registration + precedent (2020 links, experimental arms only) | 0.733 (0.715 to 0.748) | +0.006 (+0.002 to +0.010) | 0.205 | −0.1 |
| memory: lead drug | 0.500 (0.500 to 0.500) | −0.227 (−0.242 to −0.208) | 0.243 | +0.0 |

## Seen versus unseen drugs (time-ordered rounds)

A drug is seen if it is the lead drug of at least one trial in that round's training trials.

Validation trials: 869 with an unseen drug, 1725 with a seen drug, 201 with no known lead drug (left out).

### Honest models

| model | unseen: AUROC (95% interval) | unseen: minus registration | seen: AUROC (95% interval) | seen: minus registration |
|---|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.681 (0.642 to 0.720) | +0.045 (+0.020 to +0.068) | 0.779 (0.754 to 0.806) | +0.036 (+0.020 to +0.052) |
| biology: ladder, all four tables + science | 0.679 (0.639 to 0.719) | +0.043 (+0.016 to +0.071) | 0.776 (0.750 to 0.804) | +0.033 (+0.016 to +0.050) |
| ladder: boosted trees, endpoint + history + genetics | 0.669 (0.630 to 0.708) | +0.033 (+0.011 to +0.056) | 0.769 (0.742 to 0.796) | +0.025 (+0.013 to +0.039) |
| embeddings: biomedical vector plus registration plus words | 0.668 (0.630 to 0.707) | +0.032 (−0.007 to +0.069) | 0.771 (0.744 to 0.799) | +0.028 (+0.006 to +0.048) |
| embeddings: biomedical vector plus registration | 0.665 (0.627 to 0.703) | +0.029 (−0.009 to +0.065) | 0.762 (0.734 to 0.790) | +0.019 (−0.003 to +0.040) |
| ladder: registration + words | 0.665 (0.626 to 0.703) | +0.029 (+0.007 to +0.050) | 0.776 (0.751 to 0.804) | +0.033 (+0.019 to +0.048) |
| boosted trees: registration, tuned | 0.661 (0.624 to 0.697) | +0.025 (+0.007 to +0.045) | 0.747 (0.721 to 0.776) | +0.003 (−0.007 to +0.014) |
| ladder: registration + endpoint | 0.655 (0.615 to 0.692) | +0.018 (+0.004 to +0.033) | 0.752 (0.725 to 0.781) | +0.009 (+0.001 to +0.017) |
| boosted trees: registration | 0.653 (0.615 to 0.689) | +0.017 (+0.001 to +0.033) | 0.749 (0.722 to 0.779) | +0.006 (−0.003 to +0.015) |
| random forest: registration | 0.653 (0.615 to 0.689) | +0.017 (−0.001 to +0.034) | 0.745 (0.717 to 0.774) | +0.001 (−0.010 to +0.012) |
| neural network: registration | 0.650 (0.612 to 0.687) | +0.014 (+0.001 to +0.025) | 0.750 (0.723 to 0.780) | +0.007 (+0.000 to +0.013) |
| logistic: registration, tuned penalty | 0.647 (0.608 to 0.683) | +0.011 (+0.000 to +0.021) | 0.740 (0.713 to 0.768) | −0.004 (−0.010 to +0.003) |
| embeddings: general vector plus registration plus words | 0.645 (0.603 to 0.686) | +0.008 (−0.029 to +0.044) | 0.770 (0.743 to 0.798) | +0.027 (+0.005 to +0.047) |
| ladder: registration + genetics | 0.641 (0.600 to 0.680) | +0.005 (−0.002 to +0.011) | 0.745 (0.717 to 0.775) | +0.002 (−0.003 to +0.007) |
| biology: registration + networks | 0.641 (0.601 to 0.677) | +0.004 (−0.001 to +0.010) | 0.740 (0.713 to 0.770) | −0.003 (−0.006 to +0.000) |
| biology: registration + all sources | 0.638 (0.596 to 0.677) | +0.002 (−0.015 to +0.018) | 0.739 (0.709 to 0.769) | −0.004 (−0.013 to +0.004) |
| logistic: recency-weighted | 0.637 (0.597 to 0.674) | +0.001 (−0.005 to +0.007) | 0.744 (0.716 to 0.775) | +0.001 (−0.003 to +0.004) |
| logistic: last 4 years only | 0.637 (0.599 to 0.676) | +0.001 (−0.010 to +0.011) | 0.742 (0.714 to 0.774) | −0.001 (−0.008 to +0.006) |
| ladder: registration + history | 0.637 (0.597 to 0.674) | +0.001 (−0.004 to +0.005) | 0.745 (0.718 to 0.775) | +0.002 (−0.002 to +0.007) |
| logistic: registration | 0.636 (0.596 to 0.674) | (reference) | 0.743 (0.716 to 0.773) | (reference) |
| logistic: trend-shifted intercept | 0.636 (0.596 to 0.674) | +0.000 (+0.000 to +0.000) | 0.743 (0.716 to 0.773) | +0.000 (+0.000 to +0.000) |
| biology: registration + expression | 0.636 (0.594 to 0.674) | −0.000 (−0.009 to +0.008) | 0.743 (0.714 to 0.773) | −0.000 (−0.006 to +0.005) |
| logistic: last 8 years only | 0.636 (0.595 to 0.674) | −0.000 (−0.007 to +0.006) | 0.742 (0.713 to 0.772) | −0.002 (−0.005 to +0.002) |
| logistic: registration + interactions | 0.635 (0.595 to 0.672) | −0.001 (−0.013 to +0.011) | 0.749 (0.721 to 0.779) | +0.006 (+0.000 to +0.012) |
| biology: registration + DepMap | 0.635 (0.595 to 0.674) | −0.001 (−0.007 to +0.005) | 0.743 (0.714 to 0.772) | −0.001 (−0.005 to +0.003) |
| biology: registration + chemistry | 0.635 (0.594 to 0.673) | −0.001 (−0.012 to +0.008) | 0.737 (0.708 to 0.767) | −0.007 (−0.012 to −0.002) |
| logistic: separate for company and other sponsors | 0.634 (0.593 to 0.670) | −0.002 (−0.016 to +0.012) | 0.745 (0.718 to 0.774) | +0.002 (−0.006 to +0.011) |
| logistic: separate by phase | 0.634 (0.594 to 0.671) | −0.002 (−0.017 to +0.013) | 0.746 (0.717 to 0.775) | +0.002 (−0.005 to +0.010) |
| text: words | 0.633 (0.594 to 0.673) | −0.003 (−0.044 to +0.039) | 0.763 (0.735 to 0.792) | +0.019 (−0.006 to +0.045) |
| embeddings: biomedical vector alone | 0.632 (0.593 to 0.673) | −0.004 (−0.046 to +0.039) | 0.750 (0.720 to 0.778) | +0.007 (−0.017 to +0.031) |
| logistic: registration + pooled sponsor | 0.632 (0.592 to 0.670) | −0.004 (−0.011 to +0.003) | 0.748 (0.721 to 0.778) | +0.005 (+0.000 to +0.010) |
| text: words without drug names | 0.632 (0.591 to 0.671) | −0.005 (−0.045 to +0.036) | 0.761 (0.733 to 0.790) | +0.018 (−0.007 to +0.043) |
| biology: registration + annotations | 0.631 (0.590 to 0.669) | −0.005 (−0.012 to +0.002) | 0.741 (0.712 to 0.770) | −0.002 (−0.007 to +0.002) |
| embeddings: general vector plus registration | 0.626 (0.583 to 0.668) | −0.011 (−0.044 to +0.023) | 0.751 (0.722 to 0.780) | +0.008 (−0.014 to +0.028) |
| lookup: phase x sponsor class | 0.614 (0.577 to 0.650) | −0.022 (−0.056 to +0.012) | 0.684 (0.660 to 0.710) | −0.059 (−0.080 to −0.039) |
| embeddings: general vector alone | 0.593 (0.552 to 0.636) | −0.043 (−0.084 to +0.001) | 0.729 (0.698 to 0.760) | −0.014 (−0.042 to +0.013) |
| lookup: phase | 0.582 (0.549 to 0.613) | −0.054 (−0.093 to −0.014) | 0.619 (0.597 to 0.643) | −0.124 (−0.149 to −0.101) |
| memory: lead sponsor | 0.538 (0.496 to 0.577) | −0.098 (−0.144 to −0.054) | 0.658 (0.624 to 0.691) | −0.086 (−0.120 to −0.053) |
| text: drug names only | 0.536 (0.495 to 0.575) | −0.100 (−0.150 to −0.048) | 0.640 (0.606 to 0.671) | −0.103 (−0.137 to −0.071) |
| lookup: constant | 0.500 (0.500 to 0.500) | −0.136 (−0.174 to −0.096) | 0.500 (0.500 to 0.500) | −0.243 (−0.273 to −0.216) |

### Not honest (deliberately leaky)

| model | unseen: AUROC (95% interval) | unseen: minus registration | seen: AUROC (95% interval) | seen: minus registration |
|---|---|---|---|---|
| ladder: all four tables plus after-the-start columns (not honest) | 0.688 (0.649 to 0.727) | +0.052 (+0.027 to +0.077) | 0.780 (0.754 to 0.807) | +0.037 (+0.021 to +0.052) |
| leaky: registration + post-hoc fields | 0.667 (0.627 to 0.706) | +0.031 (+0.010 to +0.054) | 0.772 (0.747 to 0.799) | +0.028 (+0.016 to +0.042) |
| logistic: registration + precedent (today's links) | 0.658 (0.620 to 0.695) | +0.021 (+0.010 to +0.034) | 0.756 (0.730 to 0.784) | +0.012 (+0.003 to +0.021) |
| leaky: registration + results structure | 0.656 (0.616 to 0.694) | +0.020 (+0.006 to +0.033) | 0.741 (0.712 to 0.773) | −0.002 (−0.009 to +0.005) |
| text: words with summary (leaky) | 0.655 (0.616 to 0.693) | +0.019 (−0.018 to +0.057) | 0.767 (0.739 to 0.796) | +0.024 (+0.000 to +0.047) |
| logistic: registration + precedent (today's links, experimental arms only) | 0.651 (0.613 to 0.689) | +0.015 (+0.006 to +0.025) | 0.751 (0.724 to 0.780) | +0.007 (−0.000 to +0.015) |
| logistic: registration + precedent (2020 links) | 0.648 (0.610 to 0.686) | +0.012 (+0.002 to +0.022) | 0.752 (0.725 to 0.780) | +0.008 (−0.001 to +0.016) |
| logistic: registration + precedent (2020 names and links, experimental arms only) | 0.646 (0.607 to 0.684) | +0.010 (+0.001 to +0.018) | 0.748 (0.720 to 0.777) | +0.005 (−0.003 to +0.012) |
| logistic: registration + precedent (2020 links, experimental arms only) | 0.645 (0.606 to 0.683) | +0.009 (−0.001 to +0.017) | 0.749 (0.722 to 0.778) | +0.006 (−0.002 to +0.013) |
| logistic: registration + precedent (2020 names and links) | 0.643 (0.604 to 0.680) | +0.007 (−0.003 to +0.016) | 0.749 (0.722 to 0.778) | +0.006 (−0.002 to +0.014) |
| memory: lead drug | 0.500 (0.500 to 0.500) | −0.136 (−0.174 to −0.096) | 0.674 (0.641 to 0.703) | −0.070 (−0.103 to −0.041) |


## Skipped: table missing

None.
