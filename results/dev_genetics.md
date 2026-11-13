# Genetic evidence and trial success: development set

**What "genetic evidence" is here.** The `genetic_association` datatype of Open Targets release 26.09, and nothing else. Checked on the release's own evidence tables (`datatypeId` column under https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.09/output/evidence_*/): the datatype is built from five sources, ClinVar germline (eva), gene burden tests, GWAS credible sets, Orphanet and UniProt curated variants. ClinGen, Gene2Phenotype, Genomics England PanelApp and UniProt literature are in a separate datatype (`genetic_literature`) and are not counted, so much curated Mendelian evidence is excluded. Clinical precedence (known drugs) is a separate datatype and is excluded. "Supported" means a score above zero, a low bar: ClinVar entries below "pathogenic" still score (drug response, association and protective 0.9; likely pathogenic 0.7; uncertain significance, conflicting and other 0.3, per https://platform-docs.opentargets.org/evidence), so some support is pharmacogenomic or of uncertain significance. Scores are "indirect": evidence on narrower disease terms is rolled up to broader ones. Drugs are every drug Open Targets lists for the trial, including comparators and background therapy. The results apply to trials Open Targets could map to a target and disease (section 8).

**The "dated before start" version.** "Evidence dated before the trial started, as curated today": the history is rebuilt from publication years using today's curation, today's gene-assignment model and today's disease ontology. Against the archived 2020 release it agrees on which pairs have no evidence, but on positives only about half the time. So it is not what a predictor could have seen at the time.

Development trials (primary completion before 2021): 5,849. Success means `any_arm`. Wilson and Newcombe intervals are 95% and treat trials as independent, so they are too narrow (section 10 gives a wider one). Definitions are in BELIEFS.md B18 and B18a; the cut points for bands were fixed before outcomes were examined: today 0.608 and 0.839, dated before start 0.547 and 0.782.

**Which analyses were pre-specified.** Only the definition of support, its two versions, and the three bands with their cut points (sections 2 and 3) were written down (BELIEFS.md B18, B18a) before any outcome was joined. Sections 1 and 4 to 7 were run in the first look at outcomes and were not recorded beforehand, so treat them as exploratory. Sections 8 to 15 were added after the development result was seen, in response to review, and are labelled as such.

## 1. Trials that map to a target and disease

Do trials we can link to a target and disease succeed at a different rate from the rest?

|  | n | success rate | 95% interval |
|---|---|---|---|
| Mapped to a target and disease | 4,142 | 61.1% | 59.6% to 62.5% |
| Not mapped | 1,707 | 52.0% | 49.6% to 54.4% |

## 2. Supported versus not supported (mapped trials; pre-specified)

Among mapped trials, how much more often do those with genetic support succeed? The dated-before-start version leaves out 2 mapped trials with no start year. Success rates are per trial; the interval is Newcombe's and treats trials as independent.

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| Evidence today | 1,430 | 68.1% | 2,712 | 57.3% | +10.8 | +7.7 to +13.8 |
| Evidence dated before start | 423 | 74.0% | 3,717 | 59.6% | +14.4 | +9.8 to +18.7 |

## 3. By strength of evidence (pre-specified)

Does success rise from none to low to middle to high genetic score? Today's evidence:

|  | n | success rate | 95% interval |
|---|---|---|---|
| none | 2,712 | 57.3% | 55.5% to 59.2% |
| low | 476 | 58.0% | 53.5% to 62.3% |
| middle | 484 | 77.3% | 73.3% to 80.8% |
| high | 470 | 68.9% | 64.6% to 73.0% |

Evidence dated before the trial started, as curated today:

|  | n | success rate | 95% interval |
|---|---|---|---|
| none | 3,717 | 59.6% | 58.0% to 61.1% |
| low | 152 | 78.3% | 71.1% to 84.1% |
| middle | 131 | 74.0% | 65.9% to 80.8% |
| high | 140 | 69.3% | 61.2% to 76.3% |

## 4. Within subgroups

Is the gap the same in each phase, era, sponsor class and large therapeutic area? Evidence today:

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| Phase 2 | 512 | 52.9% | 1,092 | 43.2% | +9.7 | +4.5 to +14.9 |
| Phase 3 | 918 | 76.6% | 1,620 | 66.9% | +9.7 | +6.1 to +13.2 |
| Completed before 2015 | 747 | 69.1% | 1,467 | 62.0% | +7.1 | +2.9 to +11.2 |
| Completed 2015 to 2020 | 683 | 67.1% | 1,245 | 51.9% | +15.2 | +10.6 to +19.6 |
| Industry sponsor | 1,242 | 73.0% | 2,214 | 63.3% | +9.7 | +6.5 to +12.9 |
| Other sponsor | 188 | 35.6% | 498 | 30.9% | +4.7 | -3.0 to +12.8 |
| Area: oncology | 241 | 41.5% | 495 | 44.0% | -2.5 | -10.0 to +5.1 |
| Area: metabolic_endocrine | 320 | 87.5% | 242 | 78.5% | +9.0 | +2.8 to +15.4 |
| Area: neurology | 127 | 58.3% | 361 | 55.1% | +3.1 | -6.9 to +12.8 |
| Area: respiratory | 126 | 71.4% | 272 | 63.2% | +8.2 | -1.9 to +17.5 |
| Area: psychiatry | 129 | 56.6% | 235 | 53.6% | +3.0 | -7.7 to +13.4 |

Evidence dated before the trial started, as curated today:

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| Phase 2 | 147 | 60.5% | 1,457 | 44.9% | +15.7 | +7.2 to +23.6 |
| Phase 3 | 276 | 81.2% | 2,260 | 69.0% | +12.1 | +6.8 to +16.7 |
| Completed before 2015 | 164 | 78.7% | 2,048 | 63.2% | +15.5 | +8.3 to +21.4 |
| Completed 2015 to 2020 | 259 | 71.0% | 1,669 | 55.1% | +15.9 | +9.7 to +21.6 |
| Industry sponsor | 388 | 76.8% | 3,066 | 65.5% | +11.3 | +6.6 to +15.6 |
| Other sponsor | 35 | 42.9% | 651 | 31.6% | +11.2 | -4.1 to +27.9 |
| Area: oncology | 63 | 49.2% | 673 | 42.6% | +6.6 | -6.0 to +19.1 |
| Area: metabolic_endocrine | 132 | 88.6% | 430 | 82.1% | +6.5 | -0.8 to +12.4 |
| Area: neurology | 44 | 40.9% | 444 | 57.4% | -16.5 | -30.5 to -1.1 |
| Area: respiratory | 30 | 73.3% | 368 | 65.2% | +8.1 | -10.3 to +21.6 |
| Area: psychiatry | 21 | 71.4% | 343 | 53.6% | +17.8 | -4.2 to +33.5 |

## 5. Adjusted odds ratios

After allowing for the other things that differ between the groups, how large is the effect as an odds ratio? Logistic column: logistic regression (scikit-learn, unpenalised, converged), adjusted for phase, sponsor class (industry or not), therapeutic area and start year (linear). Mantel-Haenszel column: stratified by phase x sponsor class x era (before 2015 or not) x therapeutic area, pooling the strata under the assumption that the odds ratio is the same in every stratum (so it is not assumption-free, and it ignores strata with only one kind of trial). The two adjust for the same things in different ways, and agree closely only if the effect is similar across strata. Both intervals treat trials as independent.

|  | n | crude odds ratio | adjusted odds ratio (logistic) | Mantel-Haenszel |
|---|---|---|---|---|
| Evidence today | 4,140 | 1.59 | 1.35 (1.16 to 1.56) | 1.37 (1.18 to 1.60) |
| Evidence dated before start | 4,140 | 1.93 | 1.49 (1.16 to 1.91) | 1.54 (1.19 to 1.98) |

## 6. Sensitivity: trials with three targets or fewer

Does the gap survive when we drop trials whose many targets give many chances of a hit?

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| Evidence today, three targets or fewer | 955 | 69.9% | 2,243 | 58.8% | +11.1 | +7.5 to +14.6 |
| Evidence dated before start, three targets or fewer | 248 | 71.8% | 2,949 | 61.3% | +10.5 | +4.3 to +16.0 |

## 7. Open Targets drug type against our modality (all splits, features only)

Do two independent ways of naming a trial's drug class agree? Agreement over the 6,755 trials where both are informative (antibody and protein count as biologic): 82.6%.

| our modality \ Open Targets type | antibody | cell | gene | mixed | oligonucleotide | other | protein | small molecule | unknown |
|---|---|---|---|---|---|---|---|---|---|
| (missing) | 0 | 0 | 0 | 2 | 0 | 1 | 5 | 12 | 9 |
| advanced | 0 | 9 | 1 | 4 | 50 | 0 | 0 | 1 | 24 |
| biologic | 777 | 2 | 2 | 287 | 2 | 21 | 100 | 20 | 177 |
| mixed | 31 | 0 | 1 | 92 | 0 | 2 | 5 | 5 | 3 |
| small molecule | 97 | 1 | 1 | 203 | 13 | 14 | 493 | 4554 | 497 |
| vaccine | 1 | 0 | 0 | 1 | 0 | 25 | 0 | 2 | 30 |

## 8. Who is mapped (features and development outcomes only)

**Results here apply to mapped trials only.** Drugs with curated targets are more likely to be ones that progressed, so mapped trials are not a random sample, and mapping coverage is lower for recent trials. The last two columns are success rates within each group.

|  | n | mapped share | success, mapped | success, not mapped |
|---|---|---|---|---|
| Industry sponsor | 4,712 | 73.3% | 66.8% | 58.5% |
| Other sponsor | 1,137 | 60.3% | 32.2% | 33.9% |
| Area: oncology | 874 | 84.2% | 43.2% | 48.6% |
| Area: metabolic_endocrine | 699 | 80.4% | 83.6% | 70.1% |
| Area: neurology | 665 | 73.4% | 55.9% | 40.7% |
| Area: other | 554 | 52.9% | 65.5% | 52.9% |
| Area: respiratory | 490 | 81.2% | 65.8% | 47.8% |
| Area: (missing) | 470 | 50.9% | 58.2% | 55.8% |
| Area: psychiatry | 452 | 80.5% | 54.7% | 47.7% |
| Area: cardiovascular | 413 | 70.0% | 59.2% | 46.0% |
| Area: infectious | 390 | 35.6% | 51.8% | 51.4% |
| Area: immunology | 305 | 84.6% | 69.8% | 55.3% |
| Area: digestive | 210 | 73.8% | 56.1% | 45.5% |
| Area: dermatology | 188 | 62.8% | 73.7% | 68.6% |
| Area: musculoskeletal | 139 | 74.1% | 76.7% | 41.7% |
| Started before 2010 | 2,021 | 73.1% | 60.0% | 52.8% |
| Started 2010 to 2014 | 2,095 | 72.1% | 61.4% | 54.5% |
| Started 2015 to 2019 | 1,687 | 66.9% | 62.3% | 48.5% |
| Started 2020 | 44 | 56.8% | 44.0% | 57.9% |
| Start year missing | 2 | 100.0% | 100.0% | n/a |

Success rate of mapped minus not-mapped trials, within phase x sponsor cells:

|  | n mapped | rate | n not mapped | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| phase 2, industry | 1,194 | 51.8% | 499 | 45.5% | +6.3 | +1.0 to +11.4 |
| phase 2, other sponsor | 410 | 30.5% | 271 | 35.8% | -5.3 | -12.6 to +1.9 |
| phase 3, industry | 2,262 | 74.7% | 757 | 67.1% | +7.6 | +3.9 to +11.5 |
| phase 3, other sponsor | 276 | 34.8% | 180 | 31.1% | +3.7 | -5.2 to +12.2 |

## 9. Added after the development result was seen: investigational drug only

Support must come through a drug matched to the trial's treatment under test (registry interventions in experimental arms, see `genetics.py` for the matching rule and its failure modes) rather than a comparator or background drug. Development set only.

Evidence today: 1,430 supported trials; 1,196 are supported through a drug matched to the treatment under test and 234 (16.4%) only through another listed drug. Evidence dated before start: 423 supported trials; 307 are supported through a drug matched to the treatment under test and 116 (27.4%) only through another listed drug.

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| today: investigational support vs all other mapped | 1,196 | 68.3% | 2,946 | 58.1% | +10.2 | +7.0 to +13.3 |
| today: same, dropping trials supported only through other drugs | 1,196 | 68.3% | 2,712 | 57.3% | +11.0 | +7.7 to +14.2 |
| today: same, only trials whose investigational drug has a target | 1,196 | 68.3% | 2,398 | 56.3% | +12.0 | +8.6 to +15.2 |
| dated before start: investigational support vs all other mapped | 307 | 73.0% | 3,833 | 60.1% | +12.9 | +7.4 to +17.8 |
| dated before start: same, dropping trials supported only through other drugs | 307 | 73.0% | 3,717 | 59.6% | +13.4 | +7.9 to +18.3 |
| dated before start: same, only trials whose investigational drug has a target | 307 | 73.0% | 3,317 | 58.7% | +14.2 | +8.7 to +19.2 |

## 10. Added after the development result was seen: interval for repeated drugs

Cluster bootstrap, 1,000 resamples, seed fixed: whole lead-drug groups (first investigational drug, or first drug) are resampled, so trials of one drug are not counted as independent. It is wider because trials of the same drug share a target, a sponsor and often a result, so there is less independent information than trials suggest. Development set only.

|  | n supported | rate | n not | rate | difference (points) | 95% interval (trials independent) | 95% interval (cluster bootstrap by lead drug) | lead drugs |
|---|---|---|---|---|---|---|---|---|
| Evidence today | 1,430 | 68.1% | 2,712 | 57.3% | +10.8 | +7.7 to +13.8 | +6.6 to +14.6 | 1,245 |
| Evidence dated before start | 423 | 74.0% | 3,717 | 59.6% | +14.4 | +9.8 to +18.7 | +8.1 to +19.5 | 1,243 |

## 11. Added after the development result was seen: concentration

How many targets carry the supported trials, which, and does the difference survive dropping the five biggest? A trial is counted under each target with a positive score for its disease. Development set only.

Evidence today: the supported trials carry 308 distinct supported targets.

| target | supported trials |
|---|---|
| GLP1R (ENSG00000112164) | 119 |
| DRD2 (ENSG00000149295) | 93 |
| EGFR (ENSG00000146648) | 75 |
| NDUFA13 (ENSG00000186010) | 75 |
| NDUFS4 (ENSG00000164258) | 74 |
| NDUFS3 (ENSG00000213619) | 73 |
| NDUFA10 (ENSG00000130414) | 71 |
| NDUFB8 (ENSG00000166136) | 71 |
| NDUFB2 (ENSG00000090266) | 70 |
| GPD2 (ENSG00000115159) | 70 |

Evidence dated before start: the supported trials carry 109 distinct supported targets.

| target | supported trials |
|---|---|
| GPD2 (ENSG00000115159) | 70 |
| IL12B (ENSG00000113302) | 31 |
| INSR (ENSG00000171105) | 29 |
| CFTR (ENSG00000001626) | 26 |
| TYK2 (ENSG00000105397) | 25 |
| PCSK9 (ENSG00000169174) | 18 |
| ABCC8 (ENSG00000006071) | 17 |
| IL23A (ENSG00000110944) | 16 |
| KCNJ11 (ENSG00000187486) | 15 |
| HTR2A (ENSG00000102468) | 14 |

Headline difference after dropping every trial with any of the top five targets among its drug targets (supported or not):

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| today: dropping every trial that touches the top 5 targets | 1,071 | 67.5% | 2,639 | 57.6% | +9.9 | +6.5 to +13.3 |
| dated before start: dropping every trial that touches the top 5 targets | 240 | 62.5% | 3,633 | 59.5% | +3.0 | -3.5 to +9.1 |

## 12. Added after the development result was seen: established versus newer drugs

Earlier trials of the same lead drug: ClinicalTrials.gov trials in Open Targets' report (any phase) that list the drug and started strictly earlier. Start dates and drug ids only, no outcomes. Development set only.

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| today: 5 or fewer earlier trials of the lead drug | 314 | 65.0% | 602 | 60.0% | +5.0 | -1.6 to +11.4 |
| today: more than 5 earlier trials | 1,116 | 69.0% | 2,108 | 56.5% | +12.4 | +9.0 to +15.8 |
| dated before start: 5 or fewer earlier trials of the lead drug | 108 | 69.4% | 808 | 60.6% | +8.8 | -1.0 to +17.4 |
| dated before start: more than 5 earlier trials | 315 | 75.6% | 2,909 | 59.3% | +16.3 | +11.0 to +21.1 |

## 13. Added after the development result was seen: stricter threshold

Supported means the middle or high band (score above the existing lower cut point). Chosen after seeing that the low band looked like none. Development set only.

|  | n supported | rate | n not | rate | difference (points) | 95% interval |
|---|---|---|---|---|---|---|
| today: middle or high band vs none or low | 954 | 73.2% | 3,188 | 57.4% | +15.7 | +12.4 to +18.9 |
| dated before start: middle or high band vs none or low | 271 | 71.6% | 3,869 | 60.3% | +11.3 | +5.4 to +16.6 |

## 14. Added after the development result was seen: what each band is made of

The target of each trial's highest-scoring pair, most common first. Development set only.

| version | band | trials | top 5 targets (highest-scoring target, trials) |
|---|---|---|---|
| today | low | 476 | CHRM3 (ENSG00000133019) 56; PDE4A (ENSG00000065989) 30; OPRM1 (ENSG00000112038) 17; EGFR (ENSG00000146648) 14; GRIN2A (ENSG00000183454) 14 |
| today | middle | 484 | GLP1R (ENSG00000112164) 112; DRD2 (ENSG00000149295) 70; GPD2 (ENSG00000115159) 45; AGTR1 (ENSG00000144891) 20; ESR1 (ENSG00000091831) 19 |
| today | high | 470 | IL12B (ENSG00000113302) 44; EGFR (ENSG00000146648) 42; PCSK9 (ENSG00000169174) 41; TYK2 (ENSG00000105397) 34; CFTR (ENSG00000001626) 26 |
| dated before start | low | 152 | GPD2 (ENSG00000115159) 57; HTR2A (ENSG00000102468) 14; AR (ENSG00000169083) 13; IL12B (ENSG00000113302) 11; ESR1 (ENSG00000091831) 6 |
| dated before start | middle | 131 | INSR (ENSG00000171105) 27; KCNJ11 (ENSG00000187486) 13; IL12B (ENSG00000113302) 12; TYK2 (ENSG00000105397) 10; GCK (ENSG00000106633) 6 |
| dated before start | high | 140 | CFTR (ENSG00000001626) 24; PCSK9 (ENSG00000169174) 15; TYK2 (ENSG00000105397) 11; APP (ENSG00000142192) 8; IL12B (ENSG00000113302) 7 |

## 15. Added after the development result was seen: comparison with Minikel et al.

Minikel et al. 2024 (Nature, "Refining the impact of genetic evidence on clinical success") estimate that drug mechanisms with genetic support are 2.6 times as likely to succeed. In the text I read (the PubMed Central full text), that figure is not labelled as cumulative, but their Methods define Phase I to launch as the product of the per-phase relative successes, so it is probably a cumulative Phase I to launch ratio. Their definition is stricter than ours: a genetic association mapped to the target gene (for GWAS, at least 50% locus-to-gene share) for a trait at least 0.8 similar to the indication. They report the effect as strongest in Phases II and III and weakest in Phase I, but the per-phase ratios are shown only in figures (Fig. 2b to d) and I could not read the numbers, so I do not quote them. Ours is a different quantity: a per-trial ratio of primary-endpoint success, 68.1/57.3 = 1.19 with today's evidence and 74.0/59.6 = 1.24 with evidence dated before start, among trials that posted a comparison, with a permissive definition. The two figures are not on the same scale and should not be read as agreement or disagreement. Not verified: Minikel's per-phase values and the confidence interval for 2.6.
