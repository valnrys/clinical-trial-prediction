# Does a precedent on the same target help predict trial success?

Development trials only (completed before 2021). No test or late outcome is read.

## What is being tested

The flag says: before this trial started, another molecule acting on the same target had already started a Phase 3 trial in the same disease. Only the fact that the earlier trial had started is used, never how it ended. The earlier trial must be a pure Phase 3 (a combined Phase 2/3 record does not count; it can be a Phase 2 relabelled later) and not withdrawn. Month-only dates are read conservatively: the earlier trial's latest possible start and first-submission days must both be before the scored trial's earliest possible start day. The exact rule is in `precedent.py`. Rows marked "with Phase 2/3" in section 1 are the sensitivity version that also counts combined Phase 2/3 records.

Three versions, each dating more of the chain. **Today's links:** Open Targets 26.09 for both the trial-to-drug match and the drug-to-target links. **Targets dated:** the same drug match, targets from Open Targets 20.06 (2 July 2020); a drug gets a target in the catalogue mostly because it went far, so today's links favour drugs that later succeeded. **Names and targets dated (the headline):** registry intervention names are matched only to the names in the 20.06 drug table (preferred names, synonyms, trade names; four or more characters, whole-word phrases; a short list of generic names such as insulin, saline or water is never matched), and targets are 20.06's, so a drug catalogued after June 2020 cannot be found. Targets come from the experimental-arm (investigational) drugs of the trial. The main flag falls back to every matched drug when none matches an experimental arm; the strict flag does not, and also needs the earlier trial's drug to be an experimental-arm drug. In every version a salt or brand form inherits the targets of its molecule family.

**Still today's in all three versions:** the registry record itself (intervention names, arms, phase, status and start dates, as they are now; a sponsor can have edited them since), Open Targets' disease mapping (26.09), and `lead_drug` (the drug trials are clustered by in the intervals; today's match even in the names version). **Still today's in the first two:** the trial-to-drug name match. In the third, the 20.06 names cover the table's parent molecules only: a salt or brand that 20.06 lists as a child of a parent is not matched on its own. No version rebuilds what could have been known at the start, so the models that read the flags are not marked honest. **Test-set caution:** the 20.06 links postdate most development trials' starts but predate many test trials' starts, so gains from the dated versions may not carry over to the test set.

## 1. Paired gain over the registration baseline

AUROC is the chance a randomly chosen success is ranked above a randomly chosen failure (0.5 is a coin flip). Mean of three time-ordered rounds (scored 2015-16, 2017-18, 2019-20, each trained on earlier completions), and the repo's grouped 10-fold by lead drug. Differences are paired against `logistic: registration`. Intervals come from `validate.summarise`: the first gain column resamples trials; the others resample whole lead drugs, drawn once for all rounds, because one drug's trials are not independent.

| model | mean AUROC, three rounds (95% interval, lead drugs resampled) | gain over registration, trials resampled | gain over registration, lead drugs resampled | gain over registration, grouped 10-fold, lead drugs resampled |
|---|---|---|---|---|
| logistic: registration | 0.709 (0.686 to 0.731) | (reference) | (reference) | (reference) |
| logistic: registration + precedent (today's links) | 0.724 (0.702 to 0.746) | +0.015 (+0.010 to +0.021) | +0.015 (+0.008 to +0.022) | +0.010 (+0.005 to +0.016) |
| logistic: registration + precedent (2020 links) | 0.718 (0.696 to 0.740) | +0.009 (+0.004 to +0.015) | +0.009 (+0.003 to +0.016) | +0.007 (+0.003 to +0.012) |
| logistic: registration + precedent (2020 names and links) | 0.716 (0.694 to 0.737) | +0.008 (+0.003 to +0.013) | +0.008 (+0.002 to +0.014) | +0.007 (+0.002 to +0.011) |
| logistic: registration + precedent (today's links, experimental arms only) | 0.719 (0.697 to 0.741) | +0.010 (+0.005 to +0.016) | +0.010 (+0.004 to +0.016) | +0.007 (+0.003 to +0.012) |
| logistic: registration + precedent (2020 links, experimental arms only) | 0.715 (0.693 to 0.737) | +0.007 (+0.002 to +0.012) | +0.007 (+0.001 to +0.013) | +0.006 (+0.002 to +0.010) |
| logistic: registration + precedent (2020 names and links, experimental arms only) | 0.716 (0.694 to 0.737) | +0.007 (+0.002 to +0.012) | +0.007 (+0.001 to +0.013) | +0.006 (+0.002 to +0.011) |
| logistic: registration + precedent (today's links, experimental arms only, with Phase 2/3) | 0.719 (0.697 to 0.741) | +0.011 (+0.005 to +0.015) | +0.011 (+0.005 to +0.016) | +0.007 (+0.003 to +0.012) |
| logistic: registration + precedent (2020 links, experimental arms only, with Phase 2/3) | 0.716 (0.693 to 0.738) | +0.007 (+0.003 to +0.012) | +0.007 (+0.002 to +0.013) | +0.006 (+0.002 to +0.010) |
| logistic: registration + precedent (2020 names and links, experimental arms only, with Phase 2/3) | 0.716 (0.693 to 0.737) | +0.007 (+0.002 to +0.012) | +0.007 (+0.001 to +0.013) | +0.006 (+0.002 to +0.011) |

How many development trials each flag marks:

| flag | development trials flagged | per 100 trials |
|---|---|---|
| precedent_today | 2,251 | 38.5 |
| precedent_2020 | 2,017 | 34.5 |
| precedent_exp_today | 1,983 | 33.9 |
| precedent_exp_2020 | 1,789 | 30.6 |
| precedent_2020_names | 2,024 | 34.6 |
| precedent_exp_2020_names | 1,937 | 33.1 |
| precedent_exp_today_p23 | 2,022 | 34.6 |
| precedent_exp_2020_p23 | 1,830 | 31.3 |
| precedent_exp_2020_names_p23 | 1,984 | 33.9 |

## 2. Phase 2 trials that are the drug's first in the disease

Successes per 100 trials (any arm), with and without a precedent, for Phase 2 trials with no earlier trial of the same molecule in the same disease. A raw comparison, not adjusted for anything else. Intervals resample whole lead drugs; the last column is an interval on the difference. **First table: a fixed population.** "First in disease" is decided once, by TODAY's drug match (Open Targets 26.09), so every row describes the same trials and the rows differ only in the flag. **Second table:** each row's own link version decides "first in disease" (today's drug match for the first four rows; for the 20.06-names rows, only registry Phase 2 and 3 records matched by 20.06 name can be the earlier trial), so the populations differ from row to row and from the first table.

**Fixed population (first in disease by today's drug match):**

| links | with precedent: successes per 100 trials (95% interval, lead drugs resampled) | without precedent | difference per 100 (95% interval) |
|---|---|---|---|
| today's links | 61.9 (53.3 to 69.8) of 155 | 36.5 (32.6 to 40.3) of 690 | +25.4 (+16.4 to +34.1) |
| 2020 links (targets dated) | 62.9 (51.7 to 72.6) of 97 | 38.4 (34.6 to 42.1) of 748 | +24.5 (+13.2 to +34.9) |
| 2020 names and links (drug match and targets dated) | 55.9 (46.0 to 65.5) of 102 | 39.2 (35.3 to 42.9) of 743 | +16.7 (+6.8 to +27.1) |
| today's links, experimental arms only | 60.6 (51.3 to 68.5) of 132 | 37.6 (33.9 to 41.3) of 713 | +23.0 (+13.1 to +31.6) |
| 2020 links, experimental arms only | 60.0 (48.8 to 70.0) of 85 | 39.1 (35.3 to 42.7) of 760 | +20.9 (+9.4 to +31.1) |
| 2020 names and links, experimental arms only | 57.6 (46.8 to 67.0) of 92 | 39.2 (35.4 to 42.8) of 753 | +18.4 (+7.5 to +28.4) |

**Each row's own link version decides first in disease:**

| links | with precedent: successes per 100 trials (95% interval, lead drugs resampled) | without precedent | difference per 100 (95% interval) |
|---|---|---|---|
| today's links | 61.9 (53.3 to 69.8) of 155 | 36.5 (32.6 to 40.3) of 690 | +25.4 (+16.4 to +34.1) |
| 2020 links (targets dated) | 62.9 (51.7 to 72.6) of 97 | 38.4 (34.6 to 42.1) of 748 | +24.5 (+13.2 to +34.9) |
| 2020 names and links (drug match and targets dated) | 61.1 (52.5 to 70.1) of 113 | 39.2 (35.6 to 43.0) of 762 | +21.8 (+12.5 to +30.9) |
| today's links, experimental arms only | 60.6 (51.3 to 68.5) of 132 | 37.6 (33.9 to 41.3) of 713 | +23.0 (+13.1 to +31.6) |
| 2020 links, experimental arms only | 60.0 (48.8 to 70.0) of 85 | 39.1 (35.3 to 42.7) of 760 | +20.9 (+9.4 to +31.1) |
| 2020 names and links, experimental arms only | 60.9 (52.0 to 70.0) of 110 | 39.3 (35.8 to 43.2) of 765 | +21.6 (+12.1 to +30.9) |

## 3. Funnel

| step | trials | change from today's links |
|---|---|---|
| development trials | 5,849 |  |
| with an investigational drug (a drug of an experimental arm that Open Targets lists) | 4,923 |  |
| with a target gene, today's links (26.09) | 4,244 |  |
| with a target gene, 2020 links (20.06) | 3,811 | 433 lost, 0 gained |
| with an investigational drug found by 20.06 names (an experimental-arm name matches a 20.06 drug name; comparators are not counted) | 5,014 | 5,176 trials have any 20.06-named drug, comparators included |
| with a target gene, 2020 names and links (20.06 names, 20.06 targets) | 4,070 | 427 of today's lost, 253 gained |

Counts only. A trial with no target under the 2020 links is one whose drug the 2020 catalogue did not yet know, or gave no target. In the flag tables below, the 2020 names version counts every trial with a 20.06-named drug (comparators included) in its first rows. All versions use one salt rule (a salt inherits its molecule family's targets), so the target counts here and in the flag tables agree.

**main flag (with the fallback)**

| step (development trials) | today's links | targets dated (2020 links) | names and targets dated (2020 names and links) |
|---|---|---|---|
| trials | 5,849 | 5,849 | 5,849 |
| has a drug listed (Open Targets's list, or the 20.06 name match) | 5,490 | 5,490 | 5,176 |
| has a drug of the kind used | 5,490 | 5,490 | 5,176 |
| ... that has a target in this release | 4,687 | 4,197 | 4,185 |
| ... and has a disease | 4,109 | 3,686 | 3,668 |
| ... and has a start date | 4,107 | 3,685 | 3,667 |
| ... and a Phase 3 trial shares a target and disease with another molecule (any date) | 2,929 | 2,618 | 2,605 |
| ... that started before it | 2,319 | 2,086 | 2,096 |
| ... and was first submitted before it (the flag) | 2,251 | 2,017 | 2,024 |

**experimental arms only, no fallback**

| step (development trials) | today's links | targets dated (2020 links) | names and targets dated (2020 names and links) |
|---|---|---|---|
| trials | 5,849 | 5,849 | 5,849 |
| has a drug listed (Open Targets's list, or the 20.06 name match) | 5,490 | 5,490 | 5,176 |
| has a drug of the kind used | 4,923 | 4,923 | 5,014 |
| ... that has a target in this release | 4,244 | 3,811 | 4,070 |
| ... and has a disease | 3,716 | 3,340 | 3,564 |
| ... and has a start date | 3,716 | 3,340 | 3,563 |
| ... and a Phase 3 trial shares a target and disease with another molecule (any date) | 2,618 | 2,342 | 2,500 |
| ... that started before it | 2,039 | 1,846 | 2,005 |
| ... and was first submitted before it (the flag) | 1,983 | 1,789 | 1,937 |

## 4. Checks

- today links: KEYNOTE-010 (NCT01905657, pembrolizumab, lung cancer, 2013): the nivolumab Phase 3 trials NCT01642004, NCT01673867 are found as precedent.
- today links: NCT01642004 (CheckMate, nivolumab, 2012; the first PD-1 Phase 3 trials in lung cancer): no PD-1 precedent, as it should be. It is flagged on 15 other target(s) by the main flag, because its registry record lists docetaxel as an experimental-arm intervention, so the comparator is read as the trial's own drug.
- today links: NCT01673867 (CheckMate, nivolumab, 2012; the first PD-1 Phase 3 trials in lung cancer): no PD-1 precedent, as it should be. It is flagged on 0 other target(s) by the main flag.
- 2020 links: KEYNOTE-010 (NCT01905657, pembrolizumab, lung cancer, 2013): the nivolumab Phase 3 trials NCT01642004, NCT01673867 are found as precedent.
- 2020 links: NCT01642004 (CheckMate, nivolumab, 2012; the first PD-1 Phase 3 trials in lung cancer): no PD-1 precedent, as it should be. It is flagged on 15 other target(s) by the main flag, because its registry record lists docetaxel as an experimental-arm intervention, so the comparator is read as the trial's own drug.
- 2020 links: NCT01673867 (CheckMate, nivolumab, 2012; the first PD-1 Phase 3 trials in lung cancer): no PD-1 precedent, as it should be. It is flagged on 0 other target(s) by the main flag.
- 2020_names links: KEYNOTE-010 (NCT01905657, pembrolizumab, lung cancer, 2013): the nivolumab Phase 3 trials NCT01642004, NCT01673867 are found as precedent.
- 2020_names links: NCT01642004 (CheckMate, nivolumab, 2012; the first PD-1 Phase 3 trials in lung cancer): no PD-1 precedent, as it should be. It is flagged on 15 other target(s) by the main flag, because its registry record lists docetaxel as an experimental-arm intervention, so the comparator is read as the trial's own drug.
- 2020_names links: NCT01673867 (CheckMate, nivolumab, 2012; the first PD-1 Phase 3 trials in lung cancer): no PD-1 precedent, as it should be. It is flagged on 0 other target(s) by the main flag.
- today links: 500 of 183,269 pairs rechecked one by one in plain Python. Recomputed from raw data: the Phase 3, withdrawn, start and first-submission rules (from the registry's own date strings), the shared disease and both drugs in Open Targets' own list for their trial. Not independent: the target links and which drug is the trial's experimental one are read from the same tables the flag used: no problems.
- 2020 links: 500 of 158,207 pairs rechecked one by one in plain Python. Recomputed from raw data: the Phase 3, withdrawn, start and first-submission rules (from the registry's own date strings), the shared disease and both drugs in Open Targets' own list for their trial. Not independent: the target links and which drug is the trial's experimental one are read from the same tables the flag used: no problems.
- 2020_names links: 500 of 170,739 pairs rechecked one by one in plain Python. Recomputed from raw data: the Phase 3, withdrawn, start and first-submission rules (from the registry's own date strings), the shared disease and (no Open Targets list to check the 20.06 name match against). Not independent: the target links and which drug is the trial's experimental one are read from the same tables the flag used: no problems.

Leakage: `tests/test_precedent.py` fails if a Phase 3 trial that starts after the scored trial, starts or is first registered on the scored trial's start day, or is first registered after it, sets the flag.
