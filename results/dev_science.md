# Do public science databases help predict trial success?

Development trials only (completed before 2021). No test or late trial outcome is read.

## Read this first

Five public sources about a trial's drug target (DepMap, tissue expression, protein networks, protein annotations, drug chemistry) give 23 features. They exist for the 1,973 of 2,795 validation trials whose drug has a known target. Two tests were run: does adding them improve how well a model ranks trials, and, feature by feature, do trials at the top and bottom of a feature differ in success rate? Every difference in this report is printed with its raw sign: first-named group minus second-named group, so a negative number means the first group did worse.

- **Adding the sources to a model built on registration details did not improve how well it ranked trials.** Registration plus all five sources scores −0.001 AUROC against registration alone on the mapped validation trials (95% interval −0.010 to +0.008); on all 2,795 trials, −0.002 (−0.008 to +0.005). One source at a time: DepMap −0.002; expression +0.001; networks −0.000; annotations −0.003; chemistry −0.004.
- **The same test barely registers human genetic support, a signal established in published studies.** As a model increment it scores +0.003 AUROC (−0.000 to +0.006), so the model test cannot reliably see it, although as a single contrast it shows +5.3 (+0.0 to +10.7) points (all trials with a value; single-target drugs: +0.1 (−7.2 to +7.4), an interval that includes zero). So "adds less than about 0.01 AUROC" means these sources do not change how well trials are ranked overall. It does not mean they carry no information. The per-feature differences are the more sensitive test (section 6).
- **Feature by feature, one pattern stood out after the fact. It is a hypothesis for new data, not a finding.** For single-target drugs, a target expressed in nearly every tissue goes with lower success (in `expr_n_tissues`, on all trials with a value the top third is exactly "expressed in all 51 tissues"; among single-target drugs it is 47 to 51 tissues): −10.0 (−17.1 to −3.3) points, top third minus bottom third, adjusted for phase, sponsor type, therapeutic area and drug type. Two things weaken it. On all trials with a value the figure is −4.8 (−9.3 to −0.5) with registration groups alone, but −3.7 (−8.7 to +0.9) once drug type is accounted for (the interval includes zero): antibodies are rarely broadly expressed and succeed more. And with target class added as well, the single-target figure is −8.8 (−22.3 to +4.6) (the interval includes zero). It was noticed on the full development sample, so the confirm period of section 4 is part of what it was noticed on; confirming it needs new data or the test set (section 5).
  Context: there are 38 contrasts (one per feature, one per category level); 4 have an interval that excludes zero on all trials with a value (`expr_vital_max` −5.9; `chem_qed` +7.8; `chem_action_type`, activator −15.2; `chem_action_type`, agonist +5.0). They are correlated, so counting how many to expect by chance would mislead. With the cut points fixed before 2015, 28 of 38 kept their sign on 2015 to 2020 (about half would with no effect at all), and 1 survives with a confirm interval that excludes zero (`target_class`, Transporter +10.3); about 1 would be expected by luck. A second pattern, in cancer trials (a target the trial's cancer lineage depends on more than other lineages), is a direction worth testing; it is not distinguishable from zero here. All trials with a value: −5.7 (−18.3 to +3.1); drugs with at most three targets: −2.5 (−20.3 to +15.1); single-target drugs: not measurable (section 5).
- **What the joins were checked against, and what was not.** About thirty hand-picked genes and two drugs were checked against known biology and all passed; that catches a plainly broken join and says nothing about the ordinary gene. A spot check followed 25 random mapped trials from their target lists through the raw DepMap, GTEx, HPA, BioGRID and ORCS files and recomputed every value with separate code: 575 of 575 values agreed (section 2). Not checked: which drug a trial tested and its targets (Open Targets), the disease-to-tissue and disease-to-cancer-lineage rules, and the annotation and chemistry files beyond the spot check's reading of the slim copies the build wrote.
- **Part of "adds nothing" is the price of more columns.** Adding columns can lower AUROC slightly: the largest drop of any one source is −0.004 (registration + chemistry), because every extra column is fitted on only about 2,251 to 3,579 training trials that have a chemistry value.
- **Two limits on everything here.** Which drug a trial tested, and its target, come from today's Open Targets mapping, which favours drugs that progressed far enough to be catalogued. And Open Targets lists a drug's whole protein complex as targets, so the number of targets, which is hindsight, leaks into the maximum or minimum of every gene-level feature (`science` explains). Section 3 repeats the comparison on drugs with at most three targets and with one target.

## 1. Which trials have science data

Trials by split. "Investigational drug": Open Targets lists a drug matched to the treatment under test (`genetics.py`'s rule). "Target": that drug has at least one known target gene. The source columns count trials with at least one value from that source; every science column, chemistry included, exists only for trials with a target. Saved when the table was built (`science_funnel.csv`). For the test and late splits these are counts of trials only.

| split | trials | investigational drug | target | depmap value | expression value | networks value | annotations value | chemistry value |
|---|---|---|---|---|---|---|---|---|
| dev | 5,849 | 4,923 | 4,226 | 4,204 | 4,226 | 4,226 | 4,226 | 4,224 |
| late | 492 | 404 | 292 | 291 | 292 | 292 | 292 | 290 |
| test | 1,236 | 1,020 | 764 | 759 | 763 | 764 | 764 | 761 |

## 2. Do the sources say what biology says?

Known answers asked of the real data before any feature was tested. A failure stops the build. They cover about thirty hand-picked genes and two drugs; they do not touch the ordinary gene, the disease matching, or the choice of drug.

| check | result | what was found |
|---|---|---|
| DepMap: ribosomal and polymerase genes common essential, INS/ALB/TNNT2 not | passed | flags {'RPL11': True, 'RPL5': True, 'RPS14': True, 'RPS19': True, 'TNNT2': False, 'ALB': False, 'INS': False, 'POLR2B': True, 'POLR2E': True, 'POLR2L': True} |
| DepMap: SOX10 (skin), PAX8 (ovary), MYB (myeloid) more essential in their own lineage | passed | matched minus other lineages' mean effect {'SOX10': -1.04, 'PAX8': -0.58, 'MYB': -0.97} (needs below -0.3) |
| Expression: INS pancreas, ALB liver, TNNT2 heart, HMGCR and PCSK9 liver are top tissues | passed | rank among 51 tissues {'INS/pancreas': 1, 'ALB/liver': 1, 'TNNT2/heart': 1, 'HMGCR/liver': 1, 'PCSK9/liver': 1} |
| Expression: INS tissue-specific, ACTB and GAPDH broad (GTEx tau) | passed | tau {'ACTB': 0.19, 'GAPDH': 0.19, 'INS': 0.99} |
| Networks: TP53 in the top 1% for BioGRID partners and for papers | passed | 2206 partners (above 0.11% of genes), 1678 papers (above 0.00%); log2 partners per paper 0.39469007515155935 |
| Networks: ribosomal genes hit in most viability screens, olfactory receptors in few | passed | median hit fraction: ribosomal 0.7185792349726776, olfactory 0.0 |
| Annotations: EGFR, BRAF, ABL1 are kinases; INS secreted, EGFR membrane, TP53 nuclear | passed | kinase {'EGFR': True, 'BRAF': True, 'ABL1': True}; location {'INS': 'secreted', 'EGFR': 'membrane', 'TP53': 'nuclear'} |
| Annotations: ribosomal genes less tolerant of damage than olfactory receptors and the typical gene | passed | scores {'constraint: ribosomal': 0.9, 'constraint: olfactory': -0.89, 'constraint: all genes': -0.0, 'mouse knockout: ribosomal': 0.81, 'mouse knockout: all genes': 0.45} (higher means less tolerant) |
| Chemistry: aspirin weighs 180 with no rule-of-five violation; atorvastatin is an inhibitor | passed | aspirin mass 180.16, violations 0.0; atorvastatin action inhibitor |
| Join: trials of atorvastatin or rosuvastatin have HMGCR as a target | passed | 18 of 18 trials |

**Spot check from the raw files.** 25 development trials with a target, drawn at random (seed 7), each followed from its target list to the table value (`trace.py`). It re-reads the raw DepMap, GTEx, HPA, BioGRID and ORCS files with its own loops, separate from the build. It reads the protein annotations and the chemistry from the slim files the build wrote, not from the raw Open Targets and ChEMBL downloads. It reuses the build's gene-id mapping and its disease rules, and starts from the build's own list of each trial's target genes. It does not cover the step from trial to drug to target, or the disease-to-lineage and disease-to-tissue rules (those have their own unit tests). So it checks the join from target gene to value, the choice among several targets and the empty values, for 25 trials. A missing value on both sides counts as agreement.

| source | values compared | of which not missing | agreed |
|---|---|---|---|
| DepMap | 125 | 77 | 125 |
| expression | 125 | 121 | 125 |
| networks | 75 | 75 | 75 |
| annotations | 100 | 97 | 100 |
| chemistry | 150 | 91 | 150 |


## 3. Feature by feature

Success rate (the label: at least one arm beat control on its primary outcome) of trials at the top of a feature against the bottom, development trials with a value for it. Every difference is the first-named group minus the second-named group, as the "compared" column says, with its raw sign. "Adjusted" is a Mantel-Haenszel average of the difference within groups sharing phase, sponsor type, therapeutic area and drug type (modality); the interval resamples lead drugs. Numbers are percentage points. The thirds are cut at the one-third and two-thirds quantiles (a tied value stays on the lower side). For text features each level is compared with the others (levels with 100 or more trials). Direction was not chosen in advance here; mind that lower means stronger for the DepMap effect columns and higher means more specific for tau. The last two columns repeat the comparison on drugs with at most three targets and with one target, which removes the part of each feature that is the number of targets.

| feature | compared (difference = first group minus second) | trials (first, second) | success rate (first, second) | all: adjusted difference, points (95% interval) | at most 3 targets | one target |
|---|---|---|---|---|---|---|
| `depmap_common_essential` | yes minus no | 286, 3,918 | 46%, 61% | −5.6 (−12.4 to +0.6) | −8.6 (−17.7 to +1.0) | −10.5 (−23.8 to +2.3) |
| `depmap_mean_effect` | top third minus bottom third | 1,400, 1,414 | 62%, 54% | +0.7 (−3.7 to +5.2) | +1.3 (−3.4 to +6.2) | +2.8 (−3.1 to +10.0) |
| `depmap_fraction_dependent` | top third minus bottom third | 1,277, 1,584 | 53%, 65% | −2.2 (−6.8 to +1.8) | −1.3 (−5.9 to +2.9) | −0.2 (−7.0 to +5.0) |
| `depmap_lineage_effect` | top third minus bottom third | 123, 140 | 40%, 41% | +6.8 (−5.3 to +20.6) | +1.0 (−19.9 to +16.2) | +0.4 (−26.1 to +24.5) |
| `depmap_lineage_minus_other` | top third minus bottom third | 138, 139 | 38%, 47% | −5.7 (−18.3 to +3.1) | −2.5 (−20.3 to +15.1) | too few |
| `expr_tau` | top third minus bottom third | 1,384, 1,398 | 67%, 54% | +2.8 (−1.8 to +7.8) | +5.8 (−0.5 to +10.9) | +7.4 (+1.0 to +14.5) |
| `expr_n_tissues` | top third minus bottom third | 1,283, 1,442 | 54%, 68% | −3.7 (−8.7 to +0.9) | −8.5 (−13.9 to −3.1) | −10.0 (−17.1 to −3.3) |
| `expr_vital_max` | top third minus bottom third | 1,400, 1,430 | 53%, 66% | −5.9 (−11.1 to −1.1) | −7.0 (−12.8 to −1.9) | −10.6 (−17.8 to −3.6) |
| `expr_disease_tissue` | top third minus bottom third | 1,228, 1,289 | 58%, 62% | +1.5 (−3.1 to +6.1) | +3.5 (−1.6 to +8.7) | −0.8 (−7.7 to +5.6) |
| `expr_disease_tissue_top3` | yes minus no | 1,833, 2,008 | 62%, 57% | +1.6 (−1.6 to +5.4) | +1.3 (−2.7 to +5.4) | +2.2 (−3.1 to +7.5) |
| `net_viability_hit_fraction` | top third minus bottom third | 1,406, 1,440 | 52%, 62% | −1.9 (−6.0 to +2.5) | −2.5 (−6.9 to +2.2) | −1.4 (−7.6 to +4.2) |
| `net_other_hit_fraction` | top third minus bottom third | 1,372, 1,410 | 56%, 65% | −2.6 (−7.1 to +2.1) | −1.1 (−5.7 to +3.7) | −3.8 (−10.3 to +2.3) |
| `net_partners_per_paper` | top third minus bottom third | 1,376, 1,435 | 64%, 59% | +3.7 (−0.6 to +8.5) | +4.2 (−0.2 to +9.4) | +2.8 (−3.1 to +9.3) |
| `target_class`, Enzyme | Enzyme minus other levels | 1,043, 3,124 | 55%, 62% | −3.0 (−7.3 to +1.1) | −5.8 (−11.3 to −1.1) | −7.6 (−14.0 to −2.1) |
| `target_class`, Ion channel | Ion channel minus other levels | 278, 3,889 | 52%, 61% | −2.9 (−9.0 to +3.9) | −4.2 (−11.7 to +3.7) | −2.0 (−10.4 to +8.4) |
| `target_class`, Membrane receptor | Membrane receptor minus other levels | 1,023, 3,144 | 63%, 59% | −0.4 (−4.5 to +3.8) | −0.7 (−5.2 to +3.8) | −0.9 (−6.4 to +4.9) |
| `target_class`, Secreted protein | Secreted protein minus other levels | 475, 3,692 | 69%, 59% | +2.6 (−5.2 to +10.4) | +3.4 (−5.3 to +11.9) | +4.5 (−6.6 to +14.8) |
| `target_class`, Transcription factor | Transcription factor minus other levels | 272, 3,895 | 60%, 60% | +4.8 (−2.9 to +11.7) | +5.8 (−1.7 to +12.9) | +9.2 (−0.3 to +17.2) |
| `target_class`, Transporter | Transporter minus other levels | 288, 3,879 | 69%, 59% | +5.1 (−0.3 to +11.5) | +5.4 (−0.3 to +11.7) | +7.0 (+0.3 to +14.8) |
| `target_class`, multiple | multiple minus other levels | 522, 3,645 | 57%, 60% | +1.5 (−3.0 to +6.3) | +5.3 (−1.4 to +12.6) | too few |
| `target_location`, intracellular | intracellular minus other levels | 201, 4,023 | 43%, 61% | −7.8 (−15.7 to +0.1) | −9.8 (−18.4 to −1.1) | −12.1 (−21.2 to −2.1) |
| `target_location`, membrane | membrane minus other levels | 2,500, 1,724 | 62%, 58% | −0.6 (−4.2 to +3.4) | −0.1 (−4.3 to +4.0) | +1.1 (−3.9 to +6.4) |
| `target_location`, multiple | multiple minus other levels | 702, 3,522 | 58%, 60% | +2.7 (−1.9 to +7.4) | +2.6 (−4.7 to +9.7) | too few |
| `target_location`, nuclear | nuclear minus other levels | 381, 3,843 | 57%, 60% | +3.7 (−2.6 to +9.7) | +5.3 (−1.1 to +11.8) | +8.4 (+0.4 to +15.4) |
| `target_location`, secreted | secreted minus other levels | 440, 3,784 | 64%, 59% | −2.4 (−9.3 to +3.9) | −2.1 (−9.2 to +4.6) | −3.9 (−11.5 to +4.0) |
| `constraint` | top third minus bottom third | 1,347, 1,399 | 57%, 63% | +2.9 (−1.4 to +7.7) | +2.3 (−2.7 to +7.5) | −3.0 (−9.8 to +2.9) |
| `mouse_knockout` | top third minus bottom third | 1,363, 1,368 | 56%, 65% | −1.5 (−5.9 to +3.1) | +0.4 (−4.7 to +5.8) | +0.8 (−6.5 to +7.1) |
| `chem_mol_weight` | top third minus bottom third | 1,007, 1,009 | 61%, 55% | +1.5 (−4.0 to +7.2) | −0.3 (−6.1 to +6.0) | +0.3 (−7.6 to +8.5) |
| `chem_logp` | top third minus bottom third | 956, 983 | 61%, 57% | +2.5 (−2.9 to +7.7) | +0.2 (−6.2 to +6.2) | −1.7 (−10.8 to +5.8) |
| `chem_polar_surface_area` | top third minus bottom third | 939, 964 | 60%, 59% | −2.9 (−9.2 to +2.8) | −3.9 (−10.5 to +2.6) | +6.0 (−1.8 to +13.8) |
| `chem_qed` | top third minus bottom third | 943, 998 | 60%, 56% | +7.8 (+2.1 to +13.1) | +7.2 (+0.5 to +13.5) | +0.7 (−7.6 to +8.9) |
| `chem_rule_of_five_violations` | top third minus bottom third | 715, 2,157 | 57%, 59% | −1.2 (−6.4 to +3.4) | −2.3 (−7.9 to +3.6) | +0.5 (−6.7 to +7.3) |
| `chem_action_type`, activator | activator minus other levels | 129, 4,053 | 47%, 61% | −15.2 (−24.3 to −6.8) | −17.7 (−29.4 to −7.6) | −14.9 (−26.3 to −3.2) |
| `chem_action_type`, agonist | agonist minus other levels | 819, 3,363 | 66%, 59% | +5.0 (+0.2 to +9.5) | +6.6 (+1.7 to +11.5) | +6.6 (+0.0 to +12.9) |
| `chem_action_type`, antagonist | antagonist minus other levels | 692, 3,490 | 57%, 61% | −1.0 (−5.7 to +4.6) | −2.3 (−8.4 to +3.7) | −0.2 (−7.9 to +7.0) |
| `chem_action_type`, inhibitor | inhibitor minus other levels | 2,060, 2,122 | 61%, 59% | +0.8 (−2.9 to +4.5) | −0.3 (−4.8 to +4.1) | −0.3 (−5.6 to +5.5) |
| `chem_action_type`, multiple | multiple minus other levels | 193, 3,989 | 59%, 60% | −0.0 (−7.3 to +7.6) | +4.6 (−6.0 to +14.5) | too few |
| `chem_action_type`, other | other minus other levels | 289, 3,893 | 55%, 61% | −4.8 (−13.1 to +2.5) | −4.8 (−15.5 to +3.9) | −7.8 (−20.8 to +2.5) |

How each feature is built, and how several targets become one value:

- `depmap_common_essential`: target is on DepMap's common-essential list (any target)
- `depmap_mean_effect`: mean CRISPR gene effect over all cell lines; lower means more dependent (most dependent target)
- `depmap_fraction_dependent`: share of cell lines that depend on the gene (most dependent target)
- `depmap_lineage_effect`: cancer trials: mean gene effect in the trial's cancer lineage; lower means more dependent (most dependent tumour-intrinsic target)
- `depmap_lineage_minus_other`: cancer trials: that mean minus the mean in other lineages; lower means more specific to the lineage
- `expr_tau`: tissue specificity, GTEx; higher means one tissue (lowest tau, the broadest target)
- `expr_n_tissues`: number of tissues expressing the gene, HPA (highest)
- `expr_vital_max`: highest expression in heart, liver, kidney, brain or blood, GTEx (highest)
- `expr_disease_tissue`: expression in the disease's matched tissue, HPA (best target)
- `expr_disease_tissue_top3`: matched tissue is among the gene's top three (any target)
- `net_viability_hit_fraction`: share of independent viability screens where the gene is a hit (highest)
- `net_other_hit_fraction`: share of other knockout screens where the gene is a hit (highest)
- `net_partners_per_paper`: log2 of BioGRID physical partners per paper; high means large screens, not centrality (highest)
- `target_class`: level-1 protein class (shared class, else 'multiple')
- `target_location`: secreted, membrane, nuclear or intracellular (shared, else 'multiple')
- `constraint`: genetic constraint, higher is less tolerant (highest)
- `mouse_knockout`: mouse-knockout score, higher is less tolerant (highest)
- `chem_mol_weight`: lead drug: molecular weight
- `chem_logp`: lead drug: logP
- `chem_polar_surface_area`: lead drug: polar surface area
- `chem_qed`: lead drug: drug-likeness (QED)
- `chem_rule_of_five_violations`: lead drug: rule-of-five violations
- `chem_action_type`: lead drug: inhibitor, antagonist, agonist, activator, other or 'multiple'

`depmap_common_essential` flags mostly tubulin binders, metformin (complex I subunits) and statins (HMGCR). It says cells in culture need the target gene; it does not say the drug is toxic or that the disease depends on the gene.

## 4. Explore on early trials, confirm on later ones

For every contrast the cut points were fixed on trials finishing before 2015; the same contrast was then measured on trials finishing 2015 to 2020. Both periods print the raw difference (first group minus second, as in the compared column), so the two columns use one sign convention, and "same sign" says whether the confirm period points the same way as the explore period. The two periods share drugs, so this guards against sampling noise, not against drugs that are new.

| feature | compared (difference = first group minus second) | explore: before 2015, points | confirm: 2015 to 2020, points (95% interval) | same sign | same sign and confirm interval excludes zero |
|---|---|---|---|---|---|
| `depmap_common_essential` | yes minus no | −5.8 | −4.1 (−15.2 to +5.8) | yes | no |
| `depmap_mean_effect` | top third minus bottom third | +2.1 | +0.1 (−6.3 to +7.7) | yes | no |
| `depmap_fraction_dependent` | top third minus bottom third | −3.1 | −2.9 (−10.2 to +3.2) | yes | no |
| `depmap_lineage_effect` | top third minus bottom third | −4.6 | +8.9 (−8.2 to +28.5) | no | no |
| `depmap_lineage_minus_other` | top third minus bottom third | −10.4 | −7.3 (−29.0 to +8.1) | yes | no |
| `expr_tau` | top third minus bottom third | +3.8 | +2.4 (−5.1 to +9.3) | yes | no |
| `expr_n_tissues` | top third minus bottom third | −7.2 | −4.4 (−12.0 to +2.4) | yes | no |
| `expr_vital_max` | top third minus bottom third | −7.7 | −2.3 (−9.8 to +4.3) | yes | no |
| `expr_disease_tissue` | top third minus bottom third | −1.5 | +5.1 (−1.3 to +12.0) | no | no |
| `expr_disease_tissue_top3` | yes minus no | +1.2 | +1.9 (−4.0 to +7.4) | yes | no |
| `net_viability_hit_fraction` | top third minus bottom third | −4.4 | +2.3 (−4.2 to +8.5) | no | no |
| `net_other_hit_fraction` | top third minus bottom third | −1.4 | −3.4 (−10.1 to +3.1) | yes | no |
| `net_partners_per_paper` | top third minus bottom third | +2.9 | +3.0 (−3.4 to +9.5) | yes | no |
| `target_class`, Enzyme | Enzyme minus other levels | −3.2 | −1.5 (−8.2 to +4.5) | yes | no |
| `target_class`, Ion channel | Ion channel minus other levels | −1.6 | −2.3 (−11.7 to +6.9) | yes | no |
| `target_class`, Membrane receptor | Membrane receptor minus other levels | −2.2 | +0.7 (−5.6 to +7.6) | no | no |
| `target_class`, Secreted protein | Secreted protein minus other levels | +13.2 | −5.1 (−16.8 to +6.8) | no | no |
| `target_class`, Transcription factor | Transcription factor minus other levels | +4.9 | +3.1 (−8.3 to +15.2) | yes | no |
| `target_class`, Transporter | Transporter minus other levels | +2.0 | +10.3 (+0.0 to +22.9) | yes | yes |
| `target_class`, multiple | multiple minus other levels | +0.8 | +1.5 (−5.5 to +8.2) | yes | no |
| `target_location`, intracellular | intracellular minus other levels | −4.1 | −10.0 (−21.5 to +1.4) | yes | no |
| `target_location`, membrane | membrane minus other levels | −1.0 | −0.6 (−5.7 to +4.8) | yes | no |
| `target_location`, multiple | multiple minus other levels | +1.1 | +5.5 (−2.1 to +12.3) | yes | no |
| `target_location`, nuclear | nuclear minus other levels | +2.5 | +2.9 (−6.1 to +12.6) | yes | no |
| `target_location`, secreted | secreted minus other levels | +1.2 | −4.2 (−12.6 to +4.3) | no | no |
| `constraint` | top third minus bottom third | +3.5 | +5.1 (−2.1 to +11.6) | yes | no |
| `mouse_knockout` | top third minus bottom third | −3.2 | −4.0 (−11.5 to +3.9) | yes | no |
| `chem_mol_weight` | top third minus bottom third | +6.3 | −1.3 (−9.9 to +7.9) | no | no |
| `chem_logp` | top third minus bottom third | +5.3 | −0.4 (−8.6 to +7.7) | no | no |
| `chem_polar_surface_area` | top third minus bottom third | −2.3 | −3.3 (−12.5 to +7.5) | yes | no |
| `chem_qed` | top third minus bottom third | +7.3 | +5.9 (−2.7 to +14.5) | yes | no |
| `chem_rule_of_five_violations` | top third minus bottom third | −0.8 | −1.8 (−9.0 to +5.8) | yes | no |
| `chem_action_type`, activator | activator minus other levels | −17.6 | −11.1 (−23.6 to +0.9) | yes | no |
| `chem_action_type`, agonist | agonist minus other levels | +4.6 | +4.1 (−3.6 to +11.7) | yes | no |
| `chem_action_type`, antagonist | antagonist minus other levels | −4.5 | +2.4 (−4.5 to +9.9) | no | no |
| `chem_action_type`, inhibitor | inhibitor minus other levels | +2.1 | +0.3 (−5.4 to +6.0) | yes | no |
| `chem_action_type`, multiple | multiple minus other levels | +6.7 | −8.3 (−19.2 to +3.2) | no | no |
| `chem_action_type`, other | other minus other levels | −6.6 | −2.2 (−13.8 to +7.8) | yes | no |

28 of 38 contrasts kept their sign; 1 has the same sign and a confirm interval that excludes zero.

## 5. Two patterns that stood out after the fact

These two patterns were noticed on the full development sample by reviewers reading earlier versions of these results. They were not specified in advance. The numbers below are not a test of them: the first column is the sample they were noticed on, and the confirm period is part of it, so the explore and confirm columns show only whether the sign is the same in both halves. They are hypotheses for the test set or new data, not findings.

- (a) A target expressed broadly across tissues goes with lower success. Top third minus bottom third: more tissues and higher vital-organ expression give negative numbers; lower tau (broad) gives a positive number, because tau is high for one tissue. In `expr_n_tissues`, on all trials with a value the top third is exactly "expressed in all 51 tissues"; among single-target drugs it is 47 to 51 tissues.
- (b) A direction worth testing; not distinguishable from zero here: in cancer trials, a target the trial's cancer lineage depends on more than other lineages, tumour-intrinsic targets only (`depmap_lineage_minus_other`, a negative number means more specific; 416 trials with a value).

| feature | trials | adjusted for | all development trials with a value, points (95% interval) | explore: before 2015 | confirm: 2015 to 2020, points (95% interval) | same sign |
|---|---|---|---|---|---|---|
| `expr_n_tissues` | all trials with a value | phase, sponsor, area, drug type | −3.7 (−8.7 to +0.9) | −7.2 | −4.4 (−12.0 to +2.4) | yes |
| `expr_n_tissues` | all trials with a value | phase, sponsor, area | −4.8 (−9.3 to −0.5) | −8.4 | −6.0 (−12.7 to +0.2) | yes |
| `expr_n_tissues` | all trials with a value | phase, sponsor, area, drug type, target class | −7.2 (−18.7 to +2.1) | −7.8 | −7.3 (−21.4 to +5.0) | yes |
| `expr_n_tissues` | one target | phase, sponsor, area, drug type | −10.0 (−17.1 to −3.3) | −11.1 | −10.7 (−20.1 to −1.5) | yes |
| `expr_n_tissues` | one target | phase, sponsor, area | −10.1 (−16.8 to −3.6) | −10.6 | −11.8 (−20.6 to −3.5) | yes |
| `expr_n_tissues` | one target | phase, sponsor, area, drug type, target class | −8.8 (−22.3 to +4.6) | −11.7 | −9.0 (−25.1 to +7.0) | yes |
| `expr_vital_max` | all trials with a value | phase, sponsor, area, drug type | −5.9 (−11.1 to −1.1) | −7.7 | −2.3 (−9.8 to +4.3) | yes |
| `expr_vital_max` | all trials with a value | phase, sponsor, area | −6.0 (−10.6 to −1.6) | −7.6 | −2.3 (−9.1 to +3.9) | yes |
| `expr_vital_max` | all trials with a value | phase, sponsor, area, drug type, target class | −6.5 (−15.0 to +0.9) | −7.0 | −8.0 (−19.5 to +2.8) | yes |
| `expr_vital_max` | one target | phase, sponsor, area, drug type | −10.6 (−17.8 to −3.6) | −10.2 | −10.1 (−19.6 to −0.5) | yes |
| `expr_vital_max` | one target | phase, sponsor, area | −9.3 (−16.1 to −2.8) | −8.9 | −8.2 (−17.7 to +0.1) | yes |
| `expr_vital_max` | one target | phase, sponsor, area, drug type, target class | −12.2 (−23.3 to −1.6) | −13.0 | −16.8 (−30.3 to −1.5) | yes |
| `expr_tau` | all trials with a value | phase, sponsor, area, drug type | +2.8 (−1.8 to +7.8) | +3.8 | +2.4 (−5.1 to +9.3) | yes |
| `expr_tau` | all trials with a value | phase, sponsor, area | +3.5 (−0.9 to +7.9) | +5.3 | +3.2 (−3.1 to +9.2) | yes |
| `expr_tau` | all trials with a value | phase, sponsor, area, drug type, target class | +1.5 (−6.8 to +10.1) | +0.6 | +1.0 (−11.8 to +14.3) | yes |
| `expr_tau` | one target | phase, sponsor, area, drug type | +7.4 (+1.0 to +14.5) | +8.6 | +7.0 (−2.9 to +17.7) | yes |
| `expr_tau` | one target | phase, sponsor, area | +7.2 (+1.3 to +14.2) | +8.1 | +7.9 (−1.2 to +17.7) | yes |
| `expr_tau` | one target | phase, sponsor, area, drug type, target class | +3.6 (−8.0 to +16.7) | +7.0 | +5.0 (−13.9 to +22.6) | yes |
| `depmap_lineage_minus_other` | all trials with a value | phase, sponsor, area, drug type | −5.7 (−18.3 to +3.1) | −10.4 | −7.3 (−29.0 to +8.1) | yes |
| `depmap_lineage_minus_other` | at most 3 targets | phase, sponsor, area, drug type | −2.5 (−20.3 to +15.1) | −5.6 | −19.1 (−44.7 to +2.4) | yes |
| `depmap_lineage_minus_other` | one target | phase, sponsor, area, drug type | too few | too few | too few |  |

Drug type matters for (a): antibodies are rarely in the broadly expressed group and succeed more, so part of the all-trials figure is drug type. Single-target drugs hold up better; adding target class as well widens the interval. Single-target drugs also remove the complex-expansion problem, in which a maximum or minimum over many genes tracks the number of targets. Cancer trials whose drug has a single tumour-intrinsic target are too few for the lineage feature to be measured there.

## 6. A positive control, and what each test can see

Human genetic support (`supported_at_start`, from `genetics.parquet`) is a signal established in published studies: drug targets with human genetic evidence have tended to succeed more often. In this data its single-target interval is wide (+0.1 (−7.2 to +7.4)). It is blanked where a trial has no target, so it has the same missing values as the science columns, and then sent through the same code as the science features.

| contrast (yes minus no) | all trials with a value | at most 3 targets | one target | explore, before 2015 | confirm, 2015 to 2020 | same sign |
|---|---|---|---|---|---|---|
| genetic support | +5.3 (+0.0 to +10.7) | +3.5 (−2.6 to +9.4) | +0.1 (−7.2 to +7.4) | +3.1 | +7.9 (+0.4 to +14.4) | yes |

As a model increment over registration, on the mapped validation trials (`control: registration + genetic support`): +0.003 AUROC (−0.000 to +0.006).

What follows. The per-feature intervals above are about ±4.9 points wide in the middle, so a source could show a difference of roughly that size or more. An effect the size of genetic support (+5.3 points) is at the edge of what the per-feature test can see, and the model test cannot reliably see it. The upper end of the interval for all sources together (+0.008 AUROC) therefore bounds effects that change the overall ranking, not the contrasts a single feature might carry. Neither test shows that a source carries no information.

## 7. Models

Scored as in `results/dev_models.md`: three time-ordered rounds (train before 2015, score 2015-2016; before 2017, 2017-2018; before 2019, 2019-2020), mean of the AUROCs within each round, bootstrap intervals (which resample trials, not drugs, so they are somewhat too narrow), and every difference taken against "logistic: registration" on the same resamples. No model sees a missing-value flag for a science column: each science column is centred on its training mean over trials that have a value and missing is zero, so a trial with no target adds nothing to the score (`models/biology.py`). The models that read science columns only are fitted on trials with at least one science value; for any other trial every column is zero and they predict one constant, the model's own prediction at the centring point. Their scores are therefore given on mapped trials only.

### On the 1,973 validation trials with a mapped target

| model | mean AUROC (95% interval) | minus registration (95% interval) | validation trials with a value |
|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.759 (0.737 to 0.781) | +0.034 (+0.021 to +0.048) |  |
| biology: ladder, all four tables + science | 0.756 (0.734 to 0.778) | +0.031 (+0.016 to +0.046) |  |
| control: registration + genetic support | 0.728 (0.704 to 0.751) | +0.003 (−0.000 to +0.006) |  |
| biology: registration + expression | 0.726 (0.703 to 0.750) | +0.001 (−0.004 to +0.006) |  |
| logistic: registration | 0.725 (0.702 to 0.750) | (reference) |  |
| biology: registration + networks | 0.725 (0.701 to 0.749) | −0.000 (−0.003 to +0.003) |  |
| biology: registration + all sources | 0.724 (0.701 to 0.747) | −0.001 (−0.010 to +0.008) |  |
| biology: registration + DepMap | 0.724 (0.700 to 0.747) | −0.002 (−0.005 to +0.002) |  |
| biology: registration + annotations | 0.722 (0.698 to 0.747) | −0.003 (−0.007 to +0.001) |  |
| biology: registration + chemistry | 0.721 (0.699 to 0.746) | −0.004 (−0.010 to +0.002) |  |
| biology: boosted trees, all sources | 0.646 (0.622 to 0.670) | −0.079 (−0.108 to −0.053) |  |
| biology: logistic, all sources | 0.608 (0.583 to 0.634) | −0.117 (−0.146 to −0.089) |  |
| biology: logistic, expression only | 0.582 (0.557 to 0.605) | −0.144 (−0.172 to −0.114) | 1,973 of 1,973 |
| biology: logistic, annotations only | 0.559 (0.534 to 0.585) | −0.166 (−0.194 to −0.136) | 1,973 of 1,973 |
| biology: logistic, chemistry only | 0.538 (0.511 to 0.562) | −0.187 (−0.221 to −0.154) | 1,973 of 1,973 |
| biology: logistic, networks only | 0.538 (0.514 to 0.563) | −0.188 (−0.220 to −0.155) | 1,973 of 1,973 |
| biology: logistic, DepMap only | 0.528 (0.500 to 0.554) | −0.197 (−0.230 to −0.165) | 1,958 of 1,973 |

### On all 2,795 validation trials

For the science-only models a score on all trials would compare a constant with real predictions, so it is not given.

| model | mean AUROC (95% interval) | minus registration (95% interval) | validation trials with a value |
|---|---|---|---|
| ladder: registration + endpoint + history + genetics + words | 0.745 (0.727 to 0.762) | +0.036 (+0.025 to +0.048) |  |
| biology: ladder, all four tables + science | 0.743 (0.724 to 0.759) | +0.034 (+0.022 to +0.046) |  |
| control: registration + genetic support | 0.711 (0.691 to 0.729) | +0.002 (−0.000 to +0.004) |  |
| biology: registration + expression | 0.710 (0.690 to 0.728) | +0.001 (−0.003 to +0.005) |  |
| biology: registration + networks | 0.709 (0.689 to 0.727) | +0.000 (−0.002 to +0.002) |  |
| logistic: registration | 0.709 (0.689 to 0.727) | (reference) |  |
| biology: registration + DepMap | 0.708 (0.689 to 0.727) | −0.000 (−0.003 to +0.002) |  |
| biology: registration + all sources | 0.707 (0.688 to 0.724) | −0.002 (−0.008 to +0.005) |  |
| biology: registration + annotations | 0.706 (0.686 to 0.724) | −0.003 (−0.006 to +0.000) |  |
| biology: registration + chemistry | 0.705 (0.685 to 0.722) | −0.004 (−0.009 to −0.000) |  |
| biology: boosted trees, all sources | not meaningful (a constant for unmapped trials) | | |
| biology: logistic, all sources | not meaningful (a constant for unmapped trials) | | |
| biology: logistic, expression only | not meaningful (a constant for unmapped trials) | | |
| biology: logistic, annotations only | not meaningful (a constant for unmapped trials) | | |
| biology: logistic, chemistry only | not meaningful (a constant for unmapped trials) | | |
| biology: logistic, networks only | not meaningful (a constant for unmapped trials) | | |
| biology: logistic, DepMap only | not meaningful (a constant for unmapped trials) | | |

The ladder rung with science (`biology: ladder, all four tables + science`) against the same rung without science (`ladder: registration + endpoint + history + genetics + words`), on mapped trials: AUROC −0.003 (−0.010 to +0.004). The gain of that rung over plain registration comes from the endpoint, history, genetics and word tables it already includes.

### What the science-only models can read

The boosted trees can see which sources are empty for a trial. No chemistry value, for example, usually means the lead drug is a biologic, so the science-only trees partly encode drug type. Their lead over the logistic model on mapped trials is +0.038 AUROC (+0.017 to +0.059). The number of targets does not explain it. The science columns say "several targets rather than one" almost perfectly (AUROC 0.975, mapped development trials, five folds that keep each drug together), yet the number of targets alone ranks success at only 0.519 (0.494 to 0.543), and the trees lead the logistic among one-target trials as well as among several-target trials (table).

| validation trials | trials | trees AUROC | logistic AUROC |
|---|---|---|---|
| one target | 1,105 | 0.649 (0.617 to 0.682) | 0.618 (0.586 to 0.654) |
| several targets | 868 | 0.642 (0.609 to 0.679) | 0.596 (0.560 to 0.635) |
