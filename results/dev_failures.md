# Two kinds of failure: development set (trials finishing before 2021)


A trial can fail by not delivering a result at all (it stopped, often because it could not recruit), or by delivering a result that is negative. Our benchmark label needs a posted treatment-versus-control comparison, so it only sees the second kind. This file measures the first kind on every finished pre-2021 drug trial, asks whether rare-disease trials stop for recruitment more often and how much of that is cancer, and then looks at the second kind among the labelled trials. See BELIEFS.md B19 and limit L1.


**In short.** Rare-flagged trials stop for recruitment more often in the crude numbers (ratio 1.44), but most of that is because rare-flagged trials are mostly cancer trials (61.1% against 23.1% of the rest). Allowing for phase, sponsor, era and oncology the overall ratio is 0.97 (0.88 to 1.08). Outside oncology the adjusted ratio is 1.19 (1.00 to 1.40), which the interval cannot separate from no excess. For the second kind of failure, a negative result, rare-flagged labelled trials succeed less often, and the gap survives adjustment for phase, sponsor type and therapeutic area, smaller: -10.7 points (-15.6 to -5.6). The rare flag is imperfect (section 8), and none of this shows that rarity causes either gap.


Population: 55,329 Phase 2 or 3 trials with a drug, biological or genetic intervention, status completed, terminated or withdrawn, primary completion before 2021. Nothing from 2021 onwards was read, and the code refuses any table that holds it. Crude intervals treat trials as independent, so they are somewhat too narrow; adjusted intervals resample sponsors.

## 1. What share of trials can the benchmark speak about?

|  | trials | share of finished | 95% interval |
|---|---|---|---|
| Finished drug trials (completed, terminated or withdrawn) | 55,329 | 100% |  |
| Completed | 44,765 | 80.9% | 80.6% to 81.2% |
| Terminated | 8,285 | 15.0% | 14.7% to 15.3% |
| Withdrawn | 2,279 | 4.1% | 4.0% to 4.3% |
| Posted results on ClinicalTrials.gov | 27,584 | 49.9% | 49.4% to 50.3% |
| In our labelled benchmark | 5,849 | 10.6% | 10.3% to 10.8% |

Of the finished drug trials that ClinicalTrials.gov lists, how many ended early, how many posted results, and how many our benchmark can speak about.

| Status | trials | posted results | in our benchmark |
|---|---|---|---|
| Completed | 44,765 | 51.0% (50.5% to 51.4%) | 11.9% (11.6% to 12.2%) |
| Terminated | 8,285 | 57.5% (56.4% to 58.5%) | 6.1% (5.6% to 6.7%) |
| Withdrawn | 2,279 | 0.0% (0.0% to 0.2%) | 0.0% (0.0% to 0.2%) |

The same two shares within each status. All 5,849 labelled trials are in this population.

## 2. Why trials stopped

| Reason | terminated | withdrawn | both | share of both (95% interval) |
|---|---|---|---|---|
| stopped early for benefit | 12 | 0 | 12 | 0.1% (0.1% to 0.2%) |
| recruitment | 2,587 | 356 | 2,943 | 27.9% (27.0% to 28.7%) |
| efficacy or futility | 779 | 22 | 801 | 7.6% (7.1% to 8.1%) |
| safety | 316 | 24 | 340 | 3.2% (2.9% to 3.6%) |
| funding or business | 882 | 393 | 1,275 | 12.1% (11.5% to 12.7%) |
| other | 2,567 | 971 | 3,538 | 33.5% (32.6% to 34.4%) |
| no reason given | 1,142 | 513 | 1,655 | 15.7% (15.0% to 16.4%) |
| all | 8,285 | 2,279 | 10,564 | 100% |

Why trials stopped, by a keyword reading of the sponsor's free text. On a fresh hand read of 60 stopped trials the rules were wrong 10 times (16.7%; see section 8).

## 3. Rare versus not rare: the crude numbers

| Group | trials | share |
|---|---|---|
| rare | 10,723 | 19.4% |
| not rare | 29,119 | 52.6% |
| unclear | 6,966 | 12.6% |
| unmapped | 8,521 | 15.4% |

How many trials can be called rare or not. 28.0% cannot: unclear means every disease has an Orphanet entry but no usable prevalence class; unmapped means Open Targets lists no disease for the trial.

|  | trials | terminated or withdrawn | withdrawn | stopped for recruitment | recruitment, share of those stopped |
|---|---|---|---|---|---|
| rare | 10,723 | 22.9% (22.1% to 23.7%) | 4.4% (4.0% to 4.8%) | 6.9% (6.4% to 7.3%) | 29.9% (28.1% to 31.7%) |
| not rare | 29,119 | 17.2% (16.8% to 17.6%) | 4.0% (3.7% to 4.2%) | 4.8% (4.5% to 5.0%) | 27.7% (26.5% to 28.9%) |
| ratio, rare over not rare |  | 1.33 (1.28 to 1.39) | 1.11 (1.00 to 1.24) | 1.44 (1.32 to 1.57) |  |

The crude numbers: rare-flagged trials are stopped, and stopped for recruitment, more often than the rest. The ratio rows compare rates among all finished trials in each group; intervals treat trials as independent. With whole sponsors resampled the crude recruitment ratio is 1.44 (1.28 to 1.61). These are not adjusted for anything. The next section asks how much of this is something else about rare-flagged trials.

## 4. How much of that is cancer?

### 4.1 The oncology split

|  | trials | oncology | industry sponsor |
|---|---|---|---|
| rare | 10,723 | 61.1% | 39.3% |
| not rare | 29,119 | 23.1% | 54.6% |

Share of each group that is oncology (therapeutic area from the trial's MeSH terms) or industry-sponsored.

|  | rare trials | rare: stopped for recruitment | not rare trials | not rare: stopped for recruitment | crude ratio |
|---|---|---|---|---|---|
| oncology | 6,556 | 8.3% (7.6% to 9.0%) | 6,721 | 9.2% (8.5% to 9.9%) | 0.90 (0.80 to 1.00) |
| not oncology | 4,167 | 4.6% (4.0% to 5.3%) | 22,398 | 3.4% (3.2% to 3.7%) | 1.35 (1.16 to 1.58) |

The same crude comparison inside oncology and outside it. Cancer trials stop for recruitment at a different rate from other trials, and most rare-flagged trials are cancer trials, so part of the crude ratio in section 3 is cancer, not rarity.

### 4.2 Adjusted ratio

| Rare over not rare | crude (trials independent) | crude (sponsors resampled) | adjusted for phase, sponsor type, era and oncology (sponsors resampled) |
|---|---|---|---|
| stopped for recruitment | 1.44 (1.32 to 1.57) | 1.44 (1.28 to 1.61) | 0.97 (0.88 to 1.08) |
| terminated or withdrawn | 1.33 (1.28 to 1.39) | 1.33 (1.24 to 1.43) | 1.07 (1.01 to 1.13) |

Mantel-Haenszel ratios over strata of phase (2, 2/3, 3), sponsor type (industry or not), era of primary completion (before 2010, 2010 to 2014, 2015 to 2020) and oncology. Intervals resample lead sponsors (2,000 draws). The adjusted recruitment ratio is 0.97; its interval includes 1, so these data show no overall gap in stopping for recruitment once cancer and the other factors are allowed for. Anything not in the strata is not adjusted for.

### 4.3 Outside oncology

| Ratio of stopped for recruitment, rare over not rare | rare trials | not rare trials | crude (trials independent) | adjusted for phase, sponsor type and era (sponsors resampled) |
|---|---|---|---|---|
| outside oncology | 4,167 | 22,398 | 1.35 (1.16 to 1.58) | 1.19 (1.00 to 1.40) |
| inside oncology | 6,556 | 6,721 | 0.90 (0.80 to 1.00) | 0.87 (0.77 to 0.98) |

Outside oncology the adjusted ratio is 1.19 (1.00 to 1.40), so the data are consistent with an excess of about that size and with none. It is the one figure in this file that could be a rare-disease effect, and it is the least firm: it moves with how the rare flag is drawn (section 4.5) and rests on a flag that is imperfect (section 8). Inside oncology the adjusted ratio is below or near 1.

### 4.4 By phase and sponsor

|  | rare trials | rare: stopped for recruitment | not rare trials | not rare: stopped for recruitment | crude ratio | adjusted ratio (sponsors resampled) |
|---|---|---|---|---|---|---|
| phase2 | 8,091 | 7.7% (7.2% to 8.3%) | 16,285 | 6.1% (5.7% to 6.5%) | 1.26 (1.15 to 1.39) | 0.94 (0.85 to 1.06) |
| phase2/3 | 323 | 5.6% (3.6% to 8.6%) | 1,448 | 3.9% (3.1% to 5.1%) | 1.42 (0.85 to 2.37) | 1.32 (0.75 to 2.12) |
| phase3 | 2,309 | 4.0% (3.3% to 4.9%) | 11,386 | 2.9% (2.6% to 3.3%) | 1.37 (1.09 to 1.71) | 1.05 (0.81 to 1.36) |
| industry sponsor | 4,214 | 3.3% (2.8% to 3.9%) | 15,904 | 2.0% (1.8% to 2.2%) | 1.69 (1.39 to 2.05) | 1.22 (0.95 to 1.58) |
| other sponsor | 6,509 | 9.1% (8.5% to 9.9%) | 13,215 | 8.1% (7.7% to 8.6%) | 1.13 (1.02 to 1.24) | 0.92 (0.82 to 1.02) |

Crude and adjusted ratios by phase and by sponsor type. Phase rows are adjusted for sponsor type, era and oncology; sponsor rows for phase, era and oncology. Compare each crude ratio with its adjusted one.

### 4.5 Robustness

| How trials are grouped | rare | not rare | crude ratio | adjusted ratio | adjusted ratio outside oncology |
|---|---|---|---|---|---|
| as above | 10,723 | 29,119 | 1.44 (1.32 to 1.57) | 0.97 (0.88 to 1.08) | 1.19 (1.00 to 1.40) |
| unclear counted as rare | 17,689 | 29,119 | 1.36 (1.26 to 1.47) | 1.01 (0.93 to 1.10) | 1.14 (0.99 to 1.30) |
| unclear counted as not rare | 10,723 | 36,085 | 1.37 (1.26 to 1.49) | 0.94 (0.85 to 1.04) | 1.19 (1.00 to 1.40) |
| rare only if every disease is rare; the rest not rare | 8,787 | 31,055 | 1.42 (1.30 to 1.56) | 1.01 (0.92 to 1.12) | 1.12 (0.92 to 1.35) |
| rare only if every disease is rare; the rest left out | 8,787 | 29,119 | 1.46 (1.33 to 1.60) | 0.98 (0.88 to 1.09) | 1.14 (0.94 to 1.35) |
| prevalence measure ranked below geography (earlier rule) | 10,412 | 29,430 | 1.49 (1.37 to 1.63) | 1.00 (0.90 to 1.11) | 1.33 (1.11 to 1.56) |

Does the answer depend on how trials we cannot classify, or trials with several diseases, are counted, or on how the rare flag is drawn? Ratios are of stopping for recruitment, rare over not rare. 'Left out' drops the mixed trials (some diseases rare, some not) from both groups; 'the rest not rare' moves them to not rare, so every classified trial is counted. The last row redoes the whole classification with the earlier rule for choosing among Orphanet's prevalence entries (geography before kind of measure). Across these rows the adjusted ratio outside oncology runs from 1.12 to 1.33.

## 5. Within rare diseases, by prevalence band

| By the rarest disease in the trial | trials | terminated or withdrawn | stopped for recruitment | recruitment, outside oncology (trials: rate) | recruitment, inside oncology (trials: rate) | adjusted ratio against not rare |
|---|---|---|---|---|---|---|
| under 1 per million | 448 | 12.1% (9.4% to 15.4%) | 2.5% (1.4% to 4.3%) | 304: 1.0% (0.3% to 2.9%) | 144: 5.6% (2.8% to 10.6%) | 0.44 (0.19 to 0.75) |
| 1 to 9 per million | 716 | 21.6% (18.8% to 24.8%) | 8.9% (7.1% to 11.3%) | 386: 6.0% (4.0% to 8.8%) | 330: 12.4% (9.3% to 16.4%) | 1.33 (1.00 to 1.70) |
| 1 to 9 per 100,000 | 3,912 | 22.3% (21.0% to 23.6%) | 6.3% (5.6% to 7.1%) | 1,751: 4.5% (3.6% to 5.6%) | 2,161: 7.8% (6.7% to 9.0%) | 0.89 (0.75 to 1.05) |
| 1 to 5 per 10,000 | 5,647 | 24.4% (23.3% to 25.5%) | 7.3% (6.7% to 8.0%) | 1,726: 5.1% (4.2% to 6.2%) | 3,921: 8.3% (7.5% to 9.2%) | 0.98 (0.86 to 1.12) |

Within rare diseases, do the rarer ones stop for recruitment more often? A trial's band is the Orphanet class of the RAREST of its rare diseases, the natural reading of 'rarer diseases fail to recruit more'. The last column compares each band with all not-rare trials, adjusted for phase, sponsor type, era and oncology. There is no steady rise with rarity: the adjusted ratios from the rarest band to the commonest rare band are 0.44, 1.33, 0.89, 0.98. The two rarest bands hold few trials and their intervals are wide; the recruitment columns split by oncology show the same.

| By the commonest rare disease in the trial | trials | terminated or withdrawn | stopped for recruitment | recruitment, outside oncology (trials: rate) | recruitment, inside oncology (trials: rate) | adjusted ratio against not rare |
|---|---|---|---|---|---|---|
| under 1 per million | 205 | 13.7% (9.6% to 19.0%) | 2.9% (1.3% to 6.2%) | 166: 1.8% (0.6% to 5.2%) | 39: 7.7% (2.7% to 20.3%) | 0.55 (0.11 to 1.06) |
| 1 to 9 per million | 575 | 22.3% (19.1% to 25.8%) | 9.9% (7.7% to 12.6%) | 353: 6.5% (4.4% to 9.6%) | 222: 15.3% (11.2% to 20.6%) | 1.58 (1.21 to 2.02) |
| 1 to 9 per 100,000 | 3,629 | 21.5% (20.2% to 22.8%) | 5.8% (5.1% to 6.6%) | 1,842: 4.0% (3.2% to 5.0%) | 1,787: 7.6% (6.5% to 8.9%) | 0.87 (0.72 to 1.04) |
| 1 to 5 per 10,000 | 6,314 | 24.1% (23.1% to 25.2%) | 7.3% (6.7% to 8.0%) | 1,806: 5.1% (4.2% to 6.3%) | 4,508: 8.2% (7.4% to 9.0%) | 0.96 (0.85 to 1.09) |

The same table with the earlier definition of a band, the class of the trial's COMMONEST rare disease (this and the table above both use the current choice of prevalence entry). Only trials with several rare diseases of different classes move between rows.

| Earlier rule in full: geography first, commonest disease | trials | terminated or withdrawn | stopped for recruitment | recruitment, outside oncology (trials: rate) | recruitment, inside oncology (trials: rate) | adjusted ratio against not rare |
|---|---|---|---|---|---|---|
| under 1 per million | 212 | 13.7% (9.7% to 19.0%) | 2.8% (1.3% to 6.0%) | 173: 1.7% (0.6% to 5.0%) | 39: 7.7% (2.7% to 20.3%) | 0.54 (0.10 to 1.09) |
| 1 to 9 per million | 657 | 22.1% (19.1% to 25.4%) | 9.3% (7.3% to 11.7%) | 370: 6.5% (4.4% to 9.5%) | 287: 12.9% (9.5% to 17.3%) | 1.41 (1.07 to 1.81) |
| 1 to 9 per 100,000 | 4,968 | 23.0% (21.8% to 24.2%) | 6.8% (6.1% to 7.5%) | 1,779: 4.6% (3.7% to 5.6%) | 3,189: 8.0% (7.1% to 9.0%) | 0.93 (0.80 to 1.06) |
| 1 to 5 per 10,000 | 4,575 | 24.5% (23.2% to 25.7%) | 7.2% (6.5% to 8.0%) | 1,534: 5.4% (4.4% to 6.7%) | 3,041: 8.1% (7.2% to 9.1%) | 1.00 (0.87 to 1.14) |

The band table as it stood before the two changes: prevalence entries chosen with geography ahead of the kind of measure, and the band set by the commonest rare disease. Both changes move trials between bands and in or out of the rare group.

## 6. How many patients

| Status | group | trials | median enrolment | middle half |
|---|---|---|---|---|
| Completed | rare | 8,063 | 52 | 27 to 128 |
| Completed | not rare | 23,837 | 114 | 44 to 320 |
| Terminated | rare | 1,964 | 16 | 6 to 42 |
| Terminated | not rare | 3,828 | 24 | 8 to 84 |

Actual enrolment (the planned figure is not in the record). Withdrawn trials never enrolled anyone and are left out. Rare-flagged trials are smaller whether or not they finish, which fits a small pool of patients; this table is not adjusted for oncology or anything else.

## 7. The second kind of failure, in the labelled trials

| Group | labelled trials | success rate (95% interval) |
|---|---|---|
| rare | 677 | 46.7% (42.9% to 50.4%) |
| not rare | 3,533 | 62.2% (60.6% to 63.8%) |
| unclear | 820 | 55.1% (51.7% to 58.5%) |
| unmapped | 819 | 55.2% (51.8% to 58.6%) |

Success (`any_arm`) among the labelled development trials, by rare-disease group. 1,639 of 5,849 labelled trials (28.0%) are unclear or unmapped and are left out of the rare-versus-not-rare comparisons below.

|  | rare trials | rare: success | not rare trials | not rare: success | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| all labelled trials | 677 | 46.7% (42.9% to 50.4%) | 3,533 | 62.2% (60.6% to 63.8%) | -15.5 | -19.6 to -11.4 |
| phase 2 | 276 | 30.8% (25.6% to 36.5%) | 1,444 | 49.8% (47.2% to 52.4%) | -19.0 | -24.8 to -12.8 |
| phase 3 | 401 | 57.6% (52.7% to 62.3%) | 2,089 | 70.8% (68.8% to 72.7%) | -13.1 | -18.4 to -8.0 |
| oncology | 244 | 47.1% (41.0% to 53.4%) | 392 | 44.6% (39.8% to 49.6%) | +2.5 | -5.4 to +10.4 |
| not oncology | 433 | 46.4% (41.8% to 51.1%) | 3,141 | 64.4% (62.7% to 66.0%) | -18.0 | -22.9 to -13.0 |

The second kind of failure. Among trials that did report a comparison, rare-flagged trials succeeded less often: 46.7% against 62.2%, a crude gap of -15.5 points. Adjusted for phase, sponsor type and therapeutic area (Mantel-Haenszel, sponsors resampled) the gap is -10.7 points (-15.6 to -5.6): smaller than the crude figure, and still below zero. Phase 3 here includes combined phase 2/3 trials. The crude intervals treat trials as independent.

| Largest conditions (first condition listed) | labelled rare trials | success |
|---|---|---|
| cystic fibrosis | 44 | 50.0% (35.8% to 64.2%) |
| systemic lupus erythematosus | 21 | 47.6% (28.3% to 67.6%) |
| idiopathic pulmonary fibrosis | 17 | 17.6% (6.2% to 41.0%) |
| ovarian cancer | 14 | 42.9% (21.4% to 67.4%) |
| amyotrophic lateral sclerosis | 13 | 7.7% (1.4% to 33.3%) |
| multiple myeloma | 12 | 41.7% (19.3% to 68.0%) |
| pulmonary arterial hypertension | 12 | 41.7% (19.3% to 68.0%) |
| hepatocellular carcinoma | 9 | 44.4% (18.9% to 73.3%) |
| malaria | 9 | 66.7% (35.4% to 87.9%) |
| small cell lung cancer | 9 | 55.6% (26.7% to 81.1%) |

The ten largest first-listed conditions hold 23.6% of the labelled rare-flagged trials. Dropping every trial with one of these ten conditions, the adjusted gap is -8.3 points (-13.4 to -2.9). Conditions are the registry's free text, so near-duplicates (two spellings of one cancer) are counted apart.

| By the rarest disease in the trial | labelled trials | success rate (95% interval) |
|---|---|---|
| under 1 per million | 15 | 40.0% (19.8% to 64.3%) |
| 1 to 9 per million | 46 | 63.0% (48.6% to 75.5%) |
| 1 to 9 per 100,000 | 233 | 47.6% (41.3% to 54.0%) |
| 1 to 5 per 10,000 | 383 | 44.4% (39.5% to 49.4%) |

Success within rare-flagged trials by prevalence band. The two rarest bands hold too few trials to say anything.

## 8. What this does and does not show

**Planned enrolment is not in the record.** The registry shows the actual figure, so "failed to recruit" rests on the sponsor's free-text reason. A trial that ran short of patients may give another reason, or none (15.7% of stopped trials give none).

**The stop-reason rules are crude.** The first version of the rules was read on 40 random stopped trials and 5 were wrong (12.5%); the rules were then rewritten (negated safety or efficacy phrases removed, recruitment needing a failure word, 'interim analysis' alone not counting, a separate stopped-for-benefit category). The rewritten rules were then read on 60 fresh stopped trials that gave a reason (drawn with seed 31337 from the trials sorted by id; `results/stop_reason_read.csv`; read by a language model, not a person) and were wrong on 10 (16.7%, 95% interval 9.3% to 28.0%). For recruitment: of 18 trials the rules called recruitment, 0 were not; of 24 that were recruitment, the rules missed 6. The rules were not changed after that read, so these are their error rates. They are rules over free text: a trial that ran short of patients may give another reason, or none. 'Other' is a catch-all for reasons we did not try to sort.

**The rare flag is imperfect.** A reviewer (a language model, not a person) read 40 rare-flagged trials and found about 4 to 6 clearly wrong (for example head-and-neck cancer, a DTP/polio vaccine trial, Alzheimer's disease, a generic "leukemia") and 7 debatable (for example ovarian cancer and renal cell carcinoma, which have rare subtypes but are common overall). A trial is flagged rare if any of its Open Targets diseases is, and Open Targets lists related or background conditions as well as the one studied. A wrongly flagged trial dilutes any real difference. 61.1% of rare-flagged trials are oncology, so the flag matters most where most of the rare trials are.

**Rare is tangled up with other things.** Across all finished trials 61.1% of rare-flagged trials are oncology against 23.1% of the rest, and 39.3% have an industry sponsor against 54.6%. Section 4 allows for phase, sponsor type, era and oncology. Other differences (the kind of drug, trial size, country) are not allowed for, and nothing here says rarity causes any gap that remains.

**28.0% of trials cannot be called rare or not.** They are the unclear and unmapped rows in section 3. In the labelled set 1,639 of 5,849 trials (28.0%) are unclear or unmapped and are left out of the rare-versus-not-rare comparison in section 7. Counting unclear trials either way changes the ratios little (section 4.5); unmapped trials are never varied.

**"Not rare" does not mean "common".** A numeric prevalence for common diseases is not available from a public source by a simple join. "Not rare" means no disease of the trial is in an Orphanet rare class, or Orphanet has no entry for it. Where Orphanet gives several figures for one disease, the kind of measure (point prevalence, then birth or lifetime prevalence, then incidence) is preferred over where it applies; the earlier order, geography first, is in the robustness table.

**Open Targets lists diseases, not necessarily the one the trial targets.** A trial's disease list can include related or background conditions.

**The wider population is not the labelled benchmark.** The benchmark covers 10.6% of finished trials (section 1). This file shows how big the part we cannot see is and where it concentrates, not what those trials would have scored.

