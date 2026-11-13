# Our success label against two published benchmarks: development set


Do other benchmarks mean the same thing by success? Compared on the trials that are in both our development set (primary completion before 2021) and the other benchmark. Nothing from 2021 onwards was read: the benchmark files are filtered to our development trial ids before any label column is read, and the code refuses anything else. The benchmark files are not in this repository (see the module docstring for where to get them).


## 1. How each benchmark defines success

**CTO** (read in the paper, arXiv 2406.10292). Trials that were terminated, withdrawn, suspended or no longer available are failures, trials marked approved for marketing are successes, and a completed trial with any significant primary p-value (below 0.05) is a success. The rest are labelled by a weak-supervision model that combines several signals, including a language model's reading of publications, news headlines, links to later trials and approvals, and trial features. The `_rf` files used here hold a random forest's probability (`pred_proba`; read in the paper: the forest is fitted to TOP's human-annotated labels from those signals; that `pred_proba` is that forest's output is our reading of the file name). One of its inputs is a p-value signal (`pvalues`: 1 if any primary p-value is below 0.05, 0 if not, -1 if unknown; read in the paper). The files give a probability, not a label, and no cut-off: any yes-or-no here is our choice.

**TOP** (Fu et al.). The TOP repository's README says only that `label` is "0 (failure) or 1 (success)" and that "Outcome labels are provided by IQVIA". The CTO paper, citing TOP among others, describes the outcome as "whether the trial achieves its primary endpoints and can progress to the next stage of drug development" (arXiv 2406.10292). We could not verify more about how TOP's labels were made.

**Ours.** At least one treatment arm beat its control at p < 0.05 on the first primary outcome that has a treatment-versus-control comparison (BELIEFS-CURRENT.md). It does not check which direction is better, or whether the p-value is one-sided.

## 2. What CTO's probability looks like on these trials

**CTO phase 2 file**

| CTO probability | trials | terminated (our record) | CTO p-value signal says significant | our label: success |
|---|---|---|---|---|
| below 0.1 | 266 | 99.6% | 22.9% | 20.3% |
| 0.1 to 0.55 | 5 | 0.0% | 0.0% | 0.0% |
| 0.55 to 0.75 | 1,109 | 0.2% | 0.0% | 5.7% |
| 0.75 to 0.9 | 0 | n/a | n/a | n/a |
| 0.9 and above | 1,176 | 0.0% | 100.0% | 89.4% |

**CTO phase 3 file**

| CTO probability | trials | terminated (our record) | CTO p-value signal says significant | our label: success |
|---|---|---|---|---|
| below 0.1 | 261 | 100.0% | 34.1% | 29.9% |
| 0.1 to 0.55 | 1 | 0.0% | 0.0% | 100.0% |
| 0.55 to 0.75 | 887 | 0.0% | 0.0% | 5.0% |
| 0.75 to 0.9 | 0 | n/a | n/a | n/a |
| 0.9 and above | 2,289 | 0.0% | 100.0% | 96.4% |

Among these trials CTO's probability falls in three groups and almost nowhere else: below 0.1, which is the terminated trials (CTO counts those as failures); 0.55 to 0.75, completed trials whose p-value signal says not significant, where CTO's model is unsure; and 0.9 and above, completed trials whose p-value signal says significant. At a cut-off of 0.5, CTO's label is in effect whether the trial completed.

**Who is in the overlap.** Our set holds only trials that posted a treatment-versus-control p-value, so stopped trials, which CTO counts as failures, are under-represented. Of all finished pre-2021 drug trials, 11.9% of completed, 6.1% of terminated and 0.0% of withdrawn trials are in our set.

|  | all finished pre-2021 trials in the file | terminated or withdrawn | CTO probability at or above 0.5 | our overlap | terminated or withdrawn | CTO probability at or above 0.5 |
|---|---|---|---|---|---|---|
| CTO phase 2 file | 35,544 | 21.8% | 78.2% | 2,556 | 10.4% | 89.6% |
| CTO phase 3 file | 21,623 | 14.6% | 85.5% | 3,438 | 7.6% | 92.4% |

So the share CTO calls success at a cut-off of 0.5 is higher in our overlap than across all these trials, partly because stopped trials are missing, but it is high across all of them too: at 0.5 CTO's label mostly separates completed trials from stopped ones. A figure like '90% successes' is a property of the cut-off and of which trials are counted, and does not measure how well CTO's model does. 198 trials appear in both CTO files; each file is compared on its own, so they are counted in both rows of the next section and the two rows are not independent.

## 3. Agreement

|  | overlap | our success | their success | raw agreement | kappa (95% interval) |
|---|---|---|---|---|---|
| CTO phase 2 file, cut at 0.5 | 2,556 | 45.7% | 89.6% | 51.9% | 0.10 (0.08 to 0.12) |
| CTO phase 2 file, cut at 0.7 | 2,556 | 45.7% | 46.7% | 90.4% | 0.81 (0.78 to 0.83) |
| CTO phase 2 file, confident trials only (below 0.1 or above 0.9), cut at 0.5 | 1,442 | 76.6% | 81.6% | 87.6% | 0.63 (0.58 to 0.67) |
| CTO phase 3 file, cut at 0.5 | 3,438 | 67.7% | 92.4% | 70.8% | 0.16 (0.14 to 0.19) |
| CTO phase 3 file, cut at 0.7 | 3,438 | 67.7% | 70.1% | 90.5% | 0.78 (0.76 to 0.80) |
| CTO phase 3 file, confident trials only (below 0.1 or above 0.9), cut at 0.5 | 2,550 | 89.6% | 89.8% | 93.7% | 0.66 (0.61 to 0.70) |
| TOP | 2,515 | 63.3% | 69.1% | 87.4% | 0.72 (0.69 to 0.75) |

On the trials both cover, how CTO's probability is turned into a yes or no changes its agreement with our label from slight to strong: kappa is 0.10 at 0.5 and 0.81 at 0.7 (CTO phase 2 file); 0.16 at 0.5 and 0.78 at 0.7 (CTO phase 3 file). A benchmark's "accuracy" depends on its label definition and on the threshold, which is the reason to state both. Neither cut-off is CTO's: the files give none. 0.5 was our first try and 0.7 was chosen after seeing the groups in section 2, so the second view is shown to make the dependence visible, not as the better reading of CTO. The third view keeps only the trials CTO is confident about.

The real disagreement is on completed trials without a significant p-value: CTO's model weakly leans towards success (probabilities of 0.55 to 0.75) and we count them as failures. Kappa near 0 means no more agreement than chance; near 1 means the same labels. Trials listed more than once within a file: phase 2 file 108, phase 3 file 147. Of those, trials whose copies fall on different sides of a cut-off, dropped: phase 2 file 0, phase 3 file 2. TOP trials dropped because their copies disagree: 0.

**CTO phase 2 file, cut at 0.7** (rows: our label, columns: theirs)

|  | CTO phase 2 file: success | CTO phase 2 file: failure |
|---|---|---|
| our label: success | 1,058 | 110 |
| our label: failure | 135 | 1,253 |

**CTO phase 3 file, cut at 0.7** (rows: our label, columns: theirs)

|  | CTO phase 3 file: success | CTO phase 3 file: failure |
|---|---|---|
| our label: success | 2,207 | 122 |
| our label: failure | 203 | 906 |

**TOP** (rows: our label, columns: theirs)

|  | TOP: success | TOP: failure |
|---|---|---|
| our label: success | 1,507 | 86 |
| our label: failure | 232 | 690 |

The two-by-two tables: each trial counted once, by what each label says.

## 4. By phase and by sponsor

|  | overlap | our success | their success | raw agreement | kappa (95% interval) |
|---|---|---|---|---|---|
| CTO phase 2 file (cut at 0.7): phase 2 | 2,357 | 45.1% | 46.4% | 90.9% | 0.82 (0.79 to 0.84) |
| CTO phase 2 file (cut at 0.7): phase 3 | 199 | 52.3% | 49.7% | 84.4% | 0.69 (0.59 to 0.78) |
| CTO phase 2 file (cut at 0.7): industry sponsor | 1,786 | 50.8% | 52.6% | 92.3% | 0.85 (0.82 to 0.87) |
| CTO phase 2 file (cut at 0.7): other sponsor | 770 | 33.8% | 32.9% | 86.1% | 0.69 (0.63 to 0.74) |
| CTO phase 3 file (cut at 0.7): phase 3 | 3,438 | 67.7% | 70.1% | 90.5% | 0.78 (0.76 to 0.80) |
| CTO phase 3 file (cut at 0.7): industry sponsor | 2,994 | 72.8% | 75.6% | 91.2% | 0.77 (0.74 to 0.80) |
| CTO phase 3 file (cut at 0.7): other sponsor | 444 | 33.3% | 33.3% | 86.0% | 0.69 (0.61 to 0.76) |
| TOP: phase 2 | 850 | 48.9% | 60.5% | 81.9% | 0.64 (0.59 to 0.69) |
| TOP: phase 3 | 1,665 | 70.7% | 73.6% | 90.2% | 0.76 (0.72 to 0.79) |
| TOP: industry sponsor | 2,274 | 66.9% | 71.7% | 88.6% | 0.73 (0.70 to 0.76) |
| TOP: other sponsor | 241 | 29.9% | 44.8% | 75.9% | 0.50 (0.39 to 0.60) |

The same comparison inside our phase and sponsor groups (groups under 20 trials are left out). CTO is cut at 0.7 here. Phase is ours: combined phase 2/3 trials count as phase 3.

## 5. How well does CTO's probability rank our label?

|  | overlap | AUROC against our label (95% interval) |
|---|---|---|
| CTO phase 2 file | 2,556 | 0.894 (0.880 to 0.908) |
| CTO phase 3 file | 3,438 | 0.924 (0.913 to 0.934) |

AUROC is the chance that a random trial with our success label gets a higher CTO probability than a random trial without it (0.5 is a coin toss). CTO's model takes a p-value as an input, and our label is a p-value rule, so a high value is partly the same signal seen twice, not independent confirmation.

|  | overlap | our success | their success | raw agreement | kappa (95% interval) |
|---|---|---|---|---|---|
| CTO phase 2 file: their p-value signal | 2,519 | 45.7% | 49.1% | 92.9% | 0.86 (0.84 to 0.88) |
| CTO phase 3 file: their p-value signal | 3,411 | 67.9% | 69.7% | 96.3% | 0.91 (0.90 to 0.93) |

CTO's own p-value signal against our label, on trials where CTO has one. This is the closest like-for-like test of whether the two p-value rules read trials the same way.

## 6. What the disagreements show about our own label

These are trials where our label says success and another benchmark says failure (for CTO, a probability below 0.5, which is the terminated trials). 29 different trials were read (30 were drawn, 15 from each benchmark, and one was drawn from both). A language model read each registry record; it was not checked by a person. The 5 wrong-direction trials and the one-sided trial were then checked against the raw registry record and are marked 'registry record'; judgements marked 'model recall, unverified' rest on the reader's memory of the trial, not on the record. The trials were drawn from the disagreements, so this is a description of why labels differ, not an error rate for either label.

| What the read found | trials |
|---|---|
| terminated but significant | 17 |
| significant in the wrong direction (our label wrong) | 5 |
| their label says failure, reason not determinable from the registry record | 4 |
| p below 0.05 but said to be above the trial's own threshold | 2 |
| one-sided p-value accepted by our label | 1 |
| all | 29 |

17 of these trials were stopped early yet posted a significant primary result, which CTO or TOP counts as a failure and we count as a success: a difference in the question, not an error in either label. For 4 completed trials with a significant result, TOP says failure, and the registry record does not show why; TOP's files do not say either. 6 are errors in our label, checked against the registry record: 5 had a significant result in the wrong direction (the registry has no field for which direction is better, BELIEFS.md L4), and 1 counted a one-sided p-value. 2 more were below 0.05 but, as the reader recalls, above the trial's own threshold (L11); those two rest on recall and are unverified.

The one-sided case shows a weakness: our label accepts any p-value below 0.05 and does not ask whether it is one-sided or two-sided. Among the 5,849 development trials with a deciding comparison, 242 (4.1%) have a deciding comparison whose p-value comment or statistical comment says one-sided (or one-tailed). That is a count of mentions, not of labels that would change.

| Trial | direction | category | note | source of judgement |
|---|---|---|---|---|
| NCT00428597 | ours success / CTO failure | terminated but significant | Sunitinib v placebo in pancreatic islet cell tumours; terminated; PFS p=0.000118. | model read of registry record |
| NCT00748189 | ours success / CTO failure | terminated but significant | Ofatumumab + chlorambucil v chlorambucil; terminated; PFS p<0.001. | model read of registry record |
| NCT02492763 | ours success / CTO failure | terminated but significant | MK-8521 v placebo and liraglutide; terminated; HbA1c p=0.017 and 0.018 on two of four pairs. | model read of registry record |
| NCT01390220 | ours success / CTO failure | terminated but significant | Intranasal midazolam v placebo for seizure clusters; terminated; treatment success p=0.0109. | model read of registry record |
| NCT03530345 | ours success / CTO failure | terminated but significant | Neridronic acid v placebo in CRPS; terminated (sponsor decision); pain score p=0.0111. | model read of registry record |
| NCT01569152 | ours success / CTO failure | terminated but significant | MK-8457 + methotrexate v placebo; terminated; ACR response p<0.001. | model read of registry record |
| NCT02527343 | ours success / CTO failure | terminated but significant | Volanesorsen v placebo; terminated once enough data had accrued to decide on development; p=0.0009. | model read of registry record |
| NCT04034355 | ours success / CTO failure | terminated but significant | PledOx v placebo for oxaliplatin neuropathy; terminated after a hold on recruitment and dosing; p=0.028. | model read of registry record |
| NCT00611455 | ours success / CTO failure | terminated but significant | Ofatumumab v placebo in RA; terminated when GSK refocused on subcutaneous delivery; ACR20 p<0.001. | model read of registry record |
| NCT00762359 | ours success / CTO failure | terminated but significant | Lansoprazole v gefarnate for aspirin-induced ulcers; terminated; ulcer rate p<0.0001. | model read of registry record |
| NCT00864097 | ours success / CTO failure | terminated but significant | Tanezumab + diclofenac v placebo + diclofenac in knee OA; terminated; pain and function p from 0.002 to 0.039. | model read of registry record |
| NCT01854762 | ours success / CTO failure | terminated but significant | Raltegravir v lopinavir/ritonavir to prevent mother-to-child HIV transmission; stopped on IRB advice because the arms differed; p<0.01. | model read of registry record |
| NCT02847650 | ours success / TOP failure | terminated but significant | PF-06649751 v placebo in early Parkinson's; the why-stopped text cites lack of efficacy in moderate or advanced disease; p=0.0407 on this record's primary outcome. | model read of registry record |
| NCT01278745 | ours success / TOP failure | terminated but significant | Rituximab v placebo for cardiac allograft vasculopathy; terminated because accrual goals could not be met; p=0.0019. | model read of registry record |
| NCT01937364 | ours success / TOP failure | terminated but significant | Baclofen v placebo for alcohol withdrawal; terminated because enrolment was too limited; p=0.0334. | model read of registry record |
| NCT01975389 | ours success / TOP failure | terminated but significant | Bococizumab v placebo for cardiovascular events; terminated; p=0.021. | model read of registry record |
| NCT00129220 | ours success / TOP failure | their label says failure, reason not determinable from the registry record | Olanzapine v haloperidol and placebo in mania; completed; p<0.001 on the primary outcome; TOP calls it a failure. | model read of registry record |
| NCT00453063 | ours success / TOP failure | their label says failure, reason not determinable from the registry record | Mometasone nasal spray v placebo; completed; nasal symptoms p=0.001 (a second primary p=0.304); TOP calls it a failure. | model read of registry record |
| NCT00447278 | ours success / TOP failure | their label says failure, reason not determinable from the registry record | Atomoxetine v other standard care; completed; p<0.001 on function; TOP calls it a failure. | model read of registry record |
| NCT01495000 | ours success / TOP failure | their label says failure, reason not determinable from the registry record | Denosumab v placebo in Indian women with osteoporosis; completed; bone density p<0.0001; TOP calls it a failure. | model read of registry record |
| NCT01151137 | ours success / CTO and TOP failure | significant in the wrong direction (our label wrong) | Dronedarone v placebo in permanent atrial fibrillation; stopped for safety concerns; first co-primary outcome events 43 on dronedarone v 19 on placebo, HR 2.29 (1.34 to 3.94), p=0.0019: significant against the drug. | registry record |
| NCT02556203 | ours success / CTO failure | significant in the wrong direction (our label wrong) | Rivaroxaban-based v antiplatelet-based strategy; terminated for an imbalance in favour of the comparator; the first primary outcome has no p-value, so the label used the second: 105 v 78 events, HR 1.35 (1.01 to 1.81), p=0.042, more events on rivaroxaban. | registry record |
| NCT00768300 | ours success / CTO failure | significant in the wrong direction (our label wrong) | Ambrisentan v placebo in IPF; terminated for lack of efficacy; time to death or progression HR 1.74 (1.14 to 2.66), p=0.010: worse on ambrisentan. | registry record |
| NCT03061812 | ours success / TOP failure | significant in the wrong direction (our label wrong) | Rova-T v topotecan in small-cell lung cancer; completed; overall survival median 6.34 v 8.57 months (Rova-T v topotecan), HR 1.46 (1.17 to 1.82), p=0.0051: worse on Rova-T. | registry record |
| NCT00594568 | ours success / TOP failure | significant in the wrong direction (our label wrong) | Semagacestat v placebo in Alzheimer's; first primary outcome ADAS-Cog11 at 76 weeks, where higher is worse: change 6.19 on placebo v 7.68 on 140 mg, p=0.045: worse on the drug. | registry record |
| NCT01924533 | ours success / TOP failure | p below 0.05 but said to be above the trial's own threshold | Olaparib + paclitaxel v placebo in gastric cancer; overall survival p=0.0262 in one analysis (0.2458 in the other); the reader recalls the trial missed its own stricter threshold. | model recall, unverified |
| NCT01191801 | ours success / TOP failure | one-sided p-value accepted by our label | Vosaroxin + cytarabine v placebo + cytarabine in relapsed AML; overall survival median 7.5 v 6.1 months, HR 0.87, p=0.0305, which the record labels 'Unstratified one sided P-Value', so the two-sided p is about 0.06. | registry record |
| NCT00490139 | ours success / TOP failure | p below 0.05 but said to be above the trial's own threshold | ALTTO, lapatinib and trastuzumab arms; disease-free survival p=0.048 for one pair; the reader recalls an adjusted threshold of about 0.025 for the two comparisons. | model recall, unverified |
| NCT00562965 | ours success / TOP failure | terminated but significant | Inotuzumab + rituximab v investigator's choice; terminated (reason given only as 'see detailed description'); 29 of 978 planned participants randomised; PFS HR 0.19 (0.04 to 1.02) favours treatment, p=0.0358; the interval crosses 1 while p<0.05. | registry record |
