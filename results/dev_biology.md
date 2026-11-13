# Biology only: what do target and drug biology say about success?

Development trials only (finished before 2021). Per 100 trials; 95% intervals from a bootstrap over lead drugs. The recipe (the six flags, their thresholds, the grouping, the logistic settings and the statistic) was written down before any outcome was read in the analysis that tested it, but **the six flags themselves were chosen after looking at all development data**, 2015 to 2020 included (`dev_science.md`). It is ported unchanged (`biology_score.py`); `TEST_PLAN_2.md` item 8 scores it once on the sealed test set. Written by `python -m trial_benchmark.biology_report`.

**What it is.** Two scores, both read within phase. (a) Six risk flags, counted: target expressed in nearly every tissue; high expression in vital organs; gene essential to cells; intracellular target; the drug turns its target up; low drug-likeness. Trials with 2 or more flags against none. (b) A logistic model on biology columns only, top third against bottom third. No phase, sponsor, area label, design, precedent, genetics or earlier outcome goes in. **But the score is not blind to disease or drug type.** Four inputs describe the target in the trial's own disease tissue or cancer lineage (`expr_disease_tissue`, `expr_disease_tissue_top3`, `depmap_lineage_effect`, `depmap_lineage_minus_other`; the lineage ones exist almost only for cancer trials), and the chemistry columns exist only for small molecules (biologics have none, so the low-drug-likeness flag is off for them). So the score partly encodes drug type and disease.

**Design.** Cut points and the model come from trials finishing before 2015 ('explore', 2266 trials with a mapped target). They are then applied unchanged to trials finishing 2015 to 2020 ('confirm', 1978 trials). Nothing is tuned.

**Headline (today's links, 2015-2020).** 2+ flags minus 0: **-8.1 (-14.6 to -1.7)**. Logistic top minus bottom third: **+16.6 (+10.0 to +23.0)**. Holding disease area fixed, the flag gap is -4.5 (-11.4 to +2.3), whose interval includes zero, so it is not distinguishable from no gap, and the measured increment over registration columns is below. The groups are not monotonic: phase2 middle logistic third 49 against top third 46; phase3 one flag 60 against two or more 64 (tables below). The separation is mainly the bottom third against the rest.

## Thresholds

Fixed from trials that finished before 2015 and have a mapped target under today's links (top third for the two expression flags, bottom third for drug-likeness). The constants in `biology_score.py` are copied from the pre-declared plan; the right-hand column recomputes them from the data now, and a test checks the two agree. Method: the thresholds are Polars `quantile` (nearest rank) of the pre-2015 trials that have a value (2/3 for the expression flags, 1/3 for drug-likeness). The logistic model's tertile cuts are `numpy.quantile` (linear) of its fitted scores on the same trials. The two methods differ; both are part of the declared recipe and neither was changed. Discrete columns tie at a cut, so the groups are only roughly thirds.

| constant | in code | recomputed now | meaning |
|---|---|---|---|
| N_TISSUES_CUT | 49.0 | 49.0 | more than this many of 51 tissues = broad |
| VITAL_CUT | 5.786964829327251 | 5.786964829327251 | vital-organ max expression above this |
| QED_CUT | 0.51 | 0.51 | drug-likeness at or below this |

## The score within phase

### 2015-2020, today's links (Open Targets 26.09)

Successes per 100 by flags, within phase (2+ minus 0, pooled over phase: **-8.1 (-14.6 to -1.7)**; n 667 vs 778):

| phase | 0 flags | 1 flag | 2+ flags |
|---|---|---|---|
| phase2 | 45 (39-52), n=249 | 41 (35-48), n=247 | 37 (32-43), n=299 |
| phase3 | 72 (67-77), n=529 | 60 (54-67), n=286 | 64 (58-70), n=368 |

Successes per 100 by logistic third, within phase (high minus low, pooled over phase: **+16.6 (+10.0 to +23.0)**; n 610 vs 723):

| phase | low | mid | high |
|---|---|---|---|
| phase2 | 33 (28-38), n=353 | 49 (42-56), n=242 | 46 (38-54), n=200 |
| phase3 | 59 (53-65), n=370 | 62 (56-69), n=403 | 79 (73-83), n=410 |

Success per 100 by number of flags (2015-2020):

| phase | 0 | 1 | 2 | 3 | 4+ |
|---|---|---|---|---|---|
| phase2 | 45 (39-52), n=249 | 41 (35-48), n=247 | 41 (33-49), n=183 | 34 (23-45), n=76 | 28 (14-43), n=40 |
| phase3 | 72 (67-77), n=529 | 60 (54-67), n=286 | 66 (57-73), n=256 | 66 (55-78), n=77 | 49 (30-64), n=35 |

Logistic coefficients on the six flags (fitted before 2015; negative = fewer successes): f_broad -0.55, f_vital -0.34, f_essential -0.30, f_intracellular -0.31, f_activator -0.08, f_lowqed -0.16.

### Reference: before 2015 (where cut points and model came from; the model is in-sample)

Successes per 100 by flags, within phase (2+ minus 0, pooled over phase: **-16.5 (-22.3 to -10.9)**; n 689 vs 961):

| phase | 0 flags | 1 flag | 2+ flags |
|---|---|---|---|
| phase2 | 56 (50-62), n=278 | 49 (43-56), n=244 | 37 (29-44), n=255 |
| phase3 | 77 (73-81), n=683 | 70 (64-75), n=372 | 62 (57-67), n=434 |

Successes per 100 by logistic third, within phase (high minus low, pooled over phase: **+29.8 (+24.1 to +35.3)**; n 754 vs 757):

| phase | low | mid | high |
|---|---|---|---|
| phase2 | 31 (25-37), n=309 | 52 (46-58), n=254 | 66 (59-73), n=214 |
| phase3 | 58 (53-63), n=448 | 69 (65-73), n=501 | 85 (81-88), n=540 |

### 2015-2020, 2020 links (Open Targets 20.06 targets, today's drug match)

Successes per 100 by flags, within phase (2+ minus 0, pooled over phase: **-7.4 (-15.0 to -0.6)**; n 545 vs 680):

| phase | 0 flags | 1 flag | 2+ flags |
|---|---|---|---|
| phase2 | 43 (35-51), n=183 | 40 (33-48), n=171 | 36 (29-43), n=217 |
| phase3 | 71 (65-76), n=497 | 61 (54-68), n=258 | 64 (56-70), n=328 |

Successes per 100 by logistic third, within phase (high minus low, pooled over phase: **+14.4 (+7.0 to +21.2)**; n 506 vs 640):

| phase | low | mid | high |
|---|---|---|---|
| phase2 | 34 (28-39), n=276 | 49 (41-57), n=160 | 41 (32-51), n=135 |
| phase3 | 60 (53-66), n=364 | 62 (55-69), n=348 | 78 (72-82), n=371 |

### 2015-2020, 2020 names and links (drug matched by 20.06 names; no chemistry flags)

Successes per 100 by flags, within phase (2+ minus 0, pooled over phase: **-6.2 (-13.9 to +1.3)**; n 469 vs 868):

| phase | 0 flags | 1 flag | 2+ flags |
|---|---|---|---|
| phase2 | 41 (35-48), n=251 | 40 (33-47), n=185 | 36 (29-44), n=192 |
| phase3 | 71 (66-76), n=617 | 60 (53-67), n=245 | 64 (57-72), n=277 |

Successes per 100 by logistic third, within phase (high minus low, pooled over phase: **+16.6 (+10.1 to +23.0)**; n 582 vs 726):

| phase | low | mid | high |
|---|---|---|---|
| phase2 | 33 (27-39), n=291 | 45 (37-54), n=162 | 44 (37-51), n=175 |
| phase3 | 59 (53-65), n=435 | 63 (56-69), n=297 | 79 (73-84), n=407 |

## Within each phase, and the declared 7-flag secondary

2015-2020, today's links. Gap = 2+ flags minus 0 flags, per 100 trials. AUROC is the chance a trial that succeeded outranks one that did not (0.5 = coin flip); for the flags the score is minus the flag count, so fewer flags ranks higher. Intervals bootstrap lead drugs. 7 flags = the six plus 'no genetic support at start' (declared as a secondary; genetic support is deliberately not in the primary score).

| phase | 2+ minus 0 | AUROC, minus 6 flags | AUROC, minus 7 flags | AUROC, logistic |
|---|---|---|---|---|
| phase2 | -7.9 (-17.2 to +1.1) | 0.544 (0.505 to 0.584) | 0.561 (0.522 to 0.601) | 0.575 (0.532 to 0.617) |
| phase3 | -8.3 (-16.6 to +0.0) | 0.552 (0.506 to 0.599) | 0.552 (0.508 to 0.597) | 0.616 (0.575 to 0.662) |

## Targets and drugs the score has not met

Unseen targets: 275 of 1978 trials from 2015 to 2020 have no target gene in any pre-2015 trial. New lead drugs: 848 of 1978 have a known lead drug that is in no pre-2015 trial. Per 100 trials, today's links, pooled over phase.

| subset (2015-2020) | trials | 2+ flags minus 0 | top minus bottom third |
|---|---|---|---|
| all 2015-2020 trials with a target | 1978 | -8.1 (-14.6 to -1.7) | +16.6 (+10.0 to +23.0) |
| unseen targets (no target gene in any pre-2015 trial) | 275 | -12.6 (-28.0 to +0.8) | +19.6 (+5.4 to +33.6) |
| targets seen before 2015 | 1703 | -7.7 (-14.7 to -0.2) | +16.4 (+9.2 to +23.5) |
| lead drug in no pre-2015 trial | 848 | -5.6 (-15.3 to +3.9) | +13.3 (+3.2 to +23.0) |
| single-target small molecules | 688 | -14.5 (-26.1 to -3.2) | +23.8 (+13.2 to +34.3) |
| biologics | 501 | -13.5 (-31.1 to +2.4) | +17.1 (+4.4 to +31.0) |
| oncology | 351 | -2.9 (-16.0 to +11.3) | +18.2 (+0.4 to +36.8) |
| not oncology | 1441 | -5.8 (-13.9 to +2.1) | +14.3 (+6.4 to +22.0) |

0 trials in the confirm period have no known lead drug; they are left out of the new-lead-drug subset (unknown is not new) and each is its own cluster in every interval. Every trial here has a mapped target, so each is in exactly one of the 'unseen' and 'seen' groups. Read the intervals, not just the signs: the 2+ flags interval includes zero in 5 of 8 subsets, the top-third interval in 0. Replication is weaker on new lead drugs (see the sign-agreement count below).

## How much is just disease area?

Compared among trials that share phase (and the other named things), 2015 to 2020, today's links. Per 100 trials.

| adjusted for | 2+ flags minus 0 | top minus bottom third |
|---|---|---|
| phase only | -8.1 (-14.6 to -1.7) | +16.6 (+10.0 to +23.0) |
| phase + area | -4.5 (-11.4 to +2.3) | +11.6 (+4.8 to +18.6) |
| phase + sponsor + area + modality | -4.1 (-12.2 to +2.6) | +10.3 (+3.7 to +18.0) |

About 45% of the flag-count gap is disease area (oncology trials carry more flags and succeed less often): it goes from -8.1 to -4.5 when area is held fixed. The logistic gap goes from +16.6 to +11.6. After the area adjustment the flag-count interval includes zero, so it is not distinguishable from no gap; the logistic interval excludes zero.

## Is it just precedent or genetic support in disguise?

2015 to 2020, today's links, trials where the other flag is known. 'Phase only' pools over phase; 'phase + other' also pools over the other flag. Per 100 trials.

**strict precedent (another molecule on the same target had already started a Phase 3 trial in the same disease)** (n = 1978; 890 yes)

| contrast | phase only | phase + other flag (or biology group) |
|---|---|---|
| biology 2+ flags minus 0 | -8.1 (-14.6 to -1.7) | -7.8 (-14.3 to -1.6) |
| biology top minus bottom third | +16.6 (+10.0 to +23.0) | +15.0 (+8.7 to +21.0) |
| the other flag, yes minus no | +17.9 (+12.9 to +23.2) | +17.5 (+12.4 to +22.8) |

Share of trials with the other flag, by biology group: 0 flags: 0.49, 1 flags: 0.40, 2+ flags: 0.44.

**genetic support when the trial started** (n = 1724; 199 yes)

| contrast | phase only | phase + other flag (or biology group) |
|---|---|---|
| biology 2+ flags minus 0 | -8.7 (-15.9 to -1.6) | -8.9 (-16.0 to -2.1) |
| biology top minus bottom third | +16.5 (+9.7 to +23.1) | +16.4 (+9.7 to +23.0) |
| the other flag, yes minus no | +16.5 (+8.1 to +24.1) | +16.7 (+8.4 to +24.2) |

Share of trials with the other flag, by biology group: 0 flags: 0.12, 1 flags: 0.08, 2+ flags: 0.14.

Reading: the biology gap barely moves when precedent or genetic support is held fixed, and theirs barely moves when biology is held fixed. They measure different things.

## Each biology column on its own

Difference in successes per 100 trials, top third minus bottom third of the column (cut points from trials before 2015), or yes minus no, or one category against the rest. Pooled within the named groups; interval in brackets; **bold** = interval excludes zero. Today's links unless a column says otherwise. 'full' = phase, sponsor, area and modality. An interval counts as excluding zero (and is bold) only if it still does at the printed one-decimal precision. 'Explore to confirm: same sign' compares the 'explore' column with the 'confirm' column (the pooled 'phase only' column covers both periods and is not used for it). Last column: in how many of the variants the interval excludes zero. The intervals are bootstrapped over lead drugs; the columns are not independent tests.

| feature | n (top/yes, bottom/no) | phase only | phase + area | full | P2 only | P3 only | single target | single, full | 2020 links | 2020 names | explore | confirm | confirm, single | explore to confirm: same sign | CI excl. 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `depmap_mean_effect` | 1313/1423 | **+7.0 (+1.8, +11.9)** | +1.8 (-2.5, +6.2) | +0.8 (-3.6, +5.0) | +6.2 (-0.7, +13.1) | **+7.5 (+1.4, +13.2)** | +6.6 (-1.0, +13.5) | +2.6 (-3.6, +9.9) | **+8.1 (+3.0, +13.2)** | **+7.8 (+2.2, +12.9)** | **+12.0 (+5.3, +18.5)** | +0.5 (-6.3, +7.3) | +2.1 (-7.9, +11.1) | yes | 5 |
| `depmap_fraction_dependent` | 1283/1597 | **-9.9 (-14.5, -5.2)** | -3.4 (-7.7, +0.7) | -2.1 (-6.4, +1.8) | **-11.0 (-17.8, -3.7)** | **-9.3 (-14.6, -3.6)** | **-9.3 (-16.5, -1.7)** | -1.6 (-7.7, +4.4) | **-10.0 (-15.1, -5.1)** | **-9.4 (-14.5, -3.8)** | **-14.6 (-20.9, -8.1)** | -4.3 (-11.1, +2.5) | -7.5 (-15.8, +1.7) | yes | 7 |
| `depmap_lineage_effect` | 155/137 | +2.7 (-6.9, +12.9) | +2.7 (-6.9, +12.9) | +2.3 (-7.4, +14.3) | -1.4 (-16.5, +11.8) | +5.5 (-9.8, +21.8) | -1.9 (-19.6, +19.1) | +3.5 (-21.8, +24.0) | +3.0 (-7.0, +13.7) | +4.0 (-7.0, +14.8) | -4.2 (-18.3, +12.0) | +8.4 (-7.3, +24.2) | +1.8 (-24.8, +29.9) | no | 0 |
| `depmap_lineage_minus_other` | 128/144 | -9.6 (-21.2, +0.1) | -9.6 (-21.2, +0.1) | -6.3 (-19.1, +2.0) | -0.5 (-17.0, +13.2) | -14.9 (-30.4, +0.0) | -12.2 (-32.4, +9.2) | +1.6 (-25.6, +27.7) | -10.2 (-22.3, +1.6) | -8.4 (-19.3, +2.0) | -8.2 (-22.4, +3.9) | -10.4 (-25.5, +5.6) | – | yes | 0 |
| `expr_tau` | 1387/1546 | **+11.3 (+5.9, +16.6)** | **+4.8 (+0.4, +9.9)** | +2.6 (-2.2, +7.8) | **+13.1 (+5.7, +19.8)** | **+10.2 (+4.8, +15.7)** | **+15.5 (+7.9, +22.3)** | **+7.3 (+0.7, +14.6)** | **+10.4 (+5.5, +15.7)** | **+8.0 (+2.6, +13.2)** | **+13.6 (+7.2, +19.1)** | **+8.5 (+2.1, +15.2)** | **+15.2 (+4.7, +25.2)** | yes | 11 |
| `expr_n_tissues` | 1453/1445 | **-13.1 (-18.1, -8.2)** | **-7.5 (-12.2, -2.8)** | **-5.1 (-9.8, -0.2)** | **-13.9 (-20.8, -6.7)** | **-12.6 (-18.4, -7.0)** | **-18.9 (-25.6, -12.3)** | **-10.3 (-17.4, -4.1)** | **-12.2 (-17.6, -6.8)** | **-9.7 (-15.0, -4.5)** | **-14.8 (-21.1, -8.4)** | **-11.1 (-17.9, -4.0)** | **-18.9 (-28.3, -10.5)** | yes | 12 |
| `expr_vital_max` | 1444/1385 | **-10.8 (-15.6, -6.2)** | **-6.9 (-11.6, -2.4)** | -4.8 (-9.8, +0.0) | **-7.4 (-14.5, -0.7)** | **-12.8 (-18.4, -7.2)** | **-8.8 (-15.2, -1.8)** | **-10.6 (-17.8, -4.1)** | **-10.8 (-15.8, -5.7)** | **-9.5 (-14.7, -4.2)** | **-13.5 (-19.8, -7.0)** | **-7.3 (-14.2, -0.4)** | -8.7 (-17.2, +0.3) | yes | 10 |
| `expr_disease_tissue` | 1232/1375 | -3.2 (-7.9, +1.8) | +1.0 (-3.1, +5.1) | +2.1 (-2.3, +6.4) | -1.1 (-8.0, +5.9) | -4.4 (-10.0, +1.3) | +2.7 (-4.2, +9.2) | -0.4 (-6.9, +6.2) | -4.0 (-9.0, +1.1) | -2.1 (-7.1, +3.0) | -5.6 (-11.5, +0.4) | -0.7 (-7.5, +5.6) | +4.0 (-5.3, +12.3) | yes | 0 |
| `net_viability_hit_fraction` | 1492/1453 | **-7.1 (-12.1, -2.3)** | -2.9 (-7.1, +1.5) | -1.7 (-5.8, +2.7) | -1.8 (-8.4, +4.9) | **-10.6 (-16.0, -5.2)** | -1.9 (-9.0, +5.3) | +0.6 (-5.6, +6.1) | **-8.7 (-13.6, -3.6)** | **-7.4 (-12.4, -2.4)** | **-11.6 (-17.6, -5.4)** | -1.7 (-7.6, +4.7) | +1.7 (-7.2, +10.3) | yes | 5 |
| `net_other_hit_fraction` | 1467/1428 | **-7.3 (-12.5, -1.7)** | -3.1 (-7.2, +1.1) | -2.2 (-6.4, +2.1) | **-8.4 (-15.3, -1.3)** | **-6.7 (-13.2, -0.9)** | -7.2 (-14.8, +0.0) | -4.4 (-10.4, +1.3) | **-7.8 (-13.7, -2.1)** | **-6.3 (-12.0, -1.3)** | **-8.8 (-15.6, -1.7)** | -5.6 (-12.7, +1.4) | **-11.0 (-19.9, -2.1)** | yes | 7 |
| `net_partners_per_paper` | 1282/1646 | +3.9 (-1.5, +8.6) | +3.5 (-1.0, +8.0) | +3.3 (-1.0, +7.8) | +6.6 (-0.4, +14.0) | +2.2 (-3.8, +8.2) | **+8.8 (+2.5, +15.7)** | +3.8 (-1.6, +9.9) | +2.2 (-3.4, +8.0) | +2.0 (-3.7, +8.0) | +4.7 (-2.5, +11.4) | +2.2 (-4.7, +9.1) | +3.1 (-5.7, +11.8) | yes | 1 |
| `constraint` | 1350/1418 | **-5.5 (-10.5, -0.4)** | +2.4 (-2.2, +6.9) | +3.2 (-1.2, +8.0) | -6.2 (-13.1, +0.4) | -5.0 (-10.9, +0.7) | **-12.1 (-18.0, -5.6)** | -3.4 (-9.6, +2.4) | -4.8 (-10.0, +0.5) | -2.8 (-7.8, +2.1) | **-6.9 (-13.1, -0.3)** | -3.9 (-10.3, +2.4) | **-11.2 (-19.2, -3.5)** | yes | 4 |
| `mouse_knockout` | 1000/1432 | **-10.6 (-15.7, -5.4)** | -4.3 (-8.9, +0.7) | -3.3 (-7.8, +1.8) | **-11.3 (-18.6, -4.4)** | **-10.1 (-16.3, -4.1)** | -6.4 (-13.5, +0.8) | +1.8 (-5.8, +8.3) | **-10.6 (-16.6, -4.6)** | **-10.7 (-16.3, -5.0)** | **-10.2 (-16.5, -3.7)** | **-13.2 (-20.6, -5.8)** | -8.4 (-17.3, +0.3) | yes | 7 |
| `chem_mol_weight` | 1058/959 | **+6.4 (+1.3, +12.0)** | **+6.4 (+0.9, +11.8)** | +1.5 (-4.2, +7.2) | +6.4 (-1.4, +14.2) | +6.4 (-0.3, +12.8) | +6.3 (-1.8, +14.5) | +0.8 (-6.8, +9.1) | +5.6 (-0.1, +11.3) | – | **+7.5 (+0.6, +14.3)** | +5.9 (-2.1, +13.2) | +0.5 (-11.9, +12.1) | yes | 3 |
| `chem_logp` | 936/993 | +3.4 (-2.6, +9.3) | **+6.1 (+1.0, +11.5)** | +2.3 (-3.4, +7.4) | +0.6 (-8.3, +9.1) | +5.1 (-1.0, +12.8) | -5.7 (-14.3, +3.8) | -2.7 (-10.6, +4.5) | +5.3 (-0.9, +12.4) | – | +3.6 (-4.0, +11.3) | +2.9 (-5.4, +11.4) | -4.0 (-15.2, +6.9) | yes | 1 |
| `chem_polar_surface_area` | 1009/871 | +1.0 (-4.8, +7.0) | -1.1 (-7.3, +4.9) | -4.5 (-10.0, +1.1) | +2.7 (-6.1, +11.0) | +0.1 (-7.0, +6.9) | **+10.9 (+2.0, +19.3)** | +5.4 (-2.2, +13.6) | +0.3 (-6.7, +6.5) | – | +1.3 (-6.6, +9.0) | +2.3 (-6.0, +11.4) | +5.1 (-7.7, +18.9) | yes | 1 |
| `chem_qed` | 900/1002 | +3.0 (-2.7, +8.9) | +5.0 (-0.5, +10.4) | **+8.3 (+2.9, +13.5)** | +2.9 (-5.1, +11.5) | +3.0 (-4.2, +10.0) | -2.5 (-10.3, +5.7) | -0.1 (-8.7, +7.7) | +3.4 (-3.1, +10.1) | – | +4.2 (-2.9, +11.1) | +1.1 (-8.0, +9.2) | -3.2 (-15.7, +8.5) | yes | 1 |
| `chem_rule_of_five_violations` | 717/2172 | -1.1 (-6.4, +4.3) | +0.3 (-4.5, +5.6) | -1.2 (-6.2, +3.8) | -0.6 (-8.3, +7.2) | -1.5 (-8.1, +5.0) | -0.9 (-7.7, +6.6) | +0.4 (-6.1, +7.4) | -1.4 (-7.5, +4.3) | – | -1.7 (-9.5, +5.7) | -0.1 (-7.0, +6.7) | +0.1 (-10.0, +10.4) | yes | 0 |
| `depmap_common_essential` | 286/3936 | **-14.7 (-21.4, -8.6)** | **-7.7 (-13.7, -1.7)** | -5.7 (-12.7, +0.7) | **-18.8 (-27.1, -9.5)** | **-12.2 (-20.1, -4.4)** | **-23.8 (-36.1, -11.8)** | -10.5 (-23.5, +1.8) | **-14.4 (-21.1, -8.5)** | **-12.0 (-17.6, -6.3)** | **-18.2 (-26.5, -10.4)** | **-10.4 (-19.4, -1.3)** | -14.0 (-38.2, +13.1) | yes | 9 |
| `expr_disease_tissue_top3` | 1839/2020 | +3.8 (+0.0, +7.6) | +2.3 (-1.2, +5.7) | +1.4 (-1.8, +4.8) | +5.1 (-0.6, +11.2) | +3.0 (-1.6, +7.8) | **+7.4 (+1.4, +12.6)** | +2.0 (-3.5, +7.3) | +2.7 (-1.3, +6.6) | +2.4 (-2.1, +6.4) | +4.6 (-0.5, +10.3) | +2.4 (-3.0, +7.8) | **+7.6 (+0.3, +14.8)** | yes | 2 |
| genetic support at start | 307/3409 | **+12.9 (+6.8, +18.4)** | **+9.3 (+3.6, +14.3)** | **+6.1 (+0.5, +11.4)** | **+19.4 (+10.0, +28.7)** | **+9.4 (+2.1, +16.4)** | +7.0 (-1.9, +15.7) | -1.0 (-10.5, +8.0) | **+11.7 (+5.2, +17.7)** | **+10.4 (+4.0, +16.7)** | **+9.9 (+0.8, +17.8)** | **+16.5 (+8.2, +24.1)** | +10.2 (-1.0, +20.9) | yes | 9 |
| `target_class` = Enzyme | 1055/3130 | -4.0 (-9.6, +1.3) | -2.9 (-7.4, +1.3) | -2.6 (-6.8, +1.4) | **-7.3 (-14.1, -1.2)** | -1.8 (-8.2, +4.3) | **-7.8 (-15.2, -1.1)** | **-7.1 (-12.9, -1.4)** | -3.2 (-8.8, +2.9) | -3.4 (-9.5, +1.9) | -7.0 (-14.5, +0.1) | -0.7 (-7.3, +6.2) | -7.7 (-16.2, +0.5) | yes | 3 |
| `target_class` = Ion channel | 277/3908 | **-7.4 (-13.1, -1.3)** | -5.0 (-10.7, +1.0) | -3.1 (-9.0, +3.7) | -6.5 (-14.4, +2.1) | -8.0 (-16.2, +0.0) | -3.0 (-10.7, +6.1) | -2.0 (-10.3, +8.2) | **-6.9 (-13.4, -0.2)** | **-10.8 (-17.1, -3.7)** | -4.9 (-12.7, +3.8) | -9.4 (-19.2, +0.1) | -0.8 (-11.3, +10.8) | yes | 3 |
| `target_class` = Membrane receptor | 1028/3157 | +3.0 (-1.8, +7.5) | +0.9 (-3.0, +4.9) | -0.6 (-4.5, +3.5) | **+9.2 (+2.7, +15.7)** | -0.6 (-6.1, +4.9) | +2.6 (-3.7, +8.7) | -1.1 (-6.3, +4.9) | +1.6 (-3.3, +6.4) | +2.3 (-2.6, +7.2) | +3.2 (-2.4, +8.7) | +1.9 (-5.6, +8.4) | +1.4 (-7.9, +10.8) | yes | 1 |
| `target_class` = Secreted protein | 475/3710 | **+8.9 (+1.3, +15.2)** | +4.8 (-2.6, +11.3) | +2.6 (-6.1, +10.6) | -2.2 (-11.7, +7.5) | **+14.4 (+6.2, +21.0)** | +5.4 (-4.5, +13.3) | +4.5 (-7.3, +14.2) | **+10.7 (+3.3, +17.4)** | **+9.9 (+2.3, +16.8)** | **+13.3 (+5.3, +20.0)** | +7.3 (-2.2, +15.4) | +3.0 (-8.2, +12.3) | yes | 5 |
| `target_class` = Transcription factor | 271/3914 | +0.4 (-7.4, +7.6) | -1.1 (-9.5, +7.4) | +4.7 (-2.7, +11.4) | +0.5 (-9.8, +11.1) | +0.4 (-8.6, +9.3) | +1.1 (-7.8, +9.4) | +9.2 (-0.3, +17.2) | +1.7 (-6.6, +9.6) | +1.8 (-5.7, +8.9) | +2.6 (-5.4, +10.3) | -4.0 (-15.5, +8.3) | -2.5 (-16.6, +12.7) | no | 0 |
| `target_class` = Transporter | 288/3897 | **+7.3 (+0.4, +14.5)** | +4.6 (-0.4, +10.8) | +4.9 (-0.1, +11.3) | **+14.1 (+2.1, +25.5)** | +4.2 (-3.7, +11.8) | +7.9 (-1.9, +17.1) | +6.5 (-0.3, +14.5) | +7.7 (-0.7, +14.9) | +7.4 (-0.6, +14.6) | +2.9 (-7.3, +13.2) | **+13.1 (+2.7, +22.6)** | **+13.2 (+0.9, +26.8)** | yes | 4 |
| `target_class` = multiple | 525/3660 | -4.4 (-9.1, +0.5) | -0.2 (-4.2, +4.3) | +1.6 (-2.6, +6.2) | -2.0 (-10.3, +6.8) | -5.6 (-11.5, +1.1) | – | – | **-5.7 (-10.9, -0.5)** | -3.3 (-8.0, +1.7) | -3.4 (-9.3, +3.7) | -6.1 (-12.9, +0.8) | – | yes | 1 |
| `target_location` = intracellular | 201/4041 | **-15.2 (-22.8, -7.6)** | **-9.4 (-17.2, -1.5)** | -7.8 (-15.8, +1.3) | **-20.7 (-30.4, -8.9)** | **-10.5 (-20.8, -2.2)** | **-20.0 (-28.4, -11.8)** | **-11.9 (-21.0, -2.0)** | **-14.8 (-23.4, -6.2)** | **-14.6 (-23.3, -6.2)** | **-16.8 (-26.1, -8.0)** | **-13.5 (-24.4, -0.7)** | **-15.6 (-26.8, -2.3)** | yes | 11 |
| `target_location` = membrane | 2516/1726 | +2.8 (-1.4, +7.0) | +1.1 (-2.5, +5.1) | -0.6 (-4.3, +3.1) | +5.0 (-0.5, +10.6) | +1.5 (-3.9, +6.5) | **+5.8 (+0.1, +11.8)** | +1.1 (-3.9, +6.6) | +2.9 (-1.8, +7.7) | +2.4 (-1.7, +6.6) | +4.1 (-0.9, +9.1) | +0.8 (-4.7, +6.3) | +4.9 (-2.4, +14.1) | yes | 1 |
| `target_location` = multiple | 705/3537 | -2.2 (-7.9, +3.1) | +0.9 (-4.2, +5.8) | +2.8 (-2.3, +7.3) | -0.9 (-8.2, +6.6) | -3.0 (-8.7, +3.3) | – | – | -1.8 (-7.1, +3.9) | -0.1 (-5.1, +4.7) | -5.2 (-11.6, +1.4) | +1.4 (-5.9, +9.0) | – | no | 0 |
| `target_location` = nuclear | 380/3862 | -0.9 (-7.8, +5.7) | -0.6 (-7.2, +6.1) | +3.5 (-2.4, +9.3) | +1.6 (-6.9, +9.9) | -2.7 (-11.7, +4.5) | +1.1 (-6.8, +8.3) | **+8.4 (+0.5, +15.0)** | -2.8 (-10.2, +4.6) | -2.9 (-10.1, +3.7) | +1.6 (-6.0, +9.0) | -4.2 (-13.9, +5.6) | -0.3 (-12.9, +13.2) | no | 1 |
| `target_location` = secreted | 440/3802 | +4.0 (-3.9, +10.2) | +1.1 (-5.7, +7.2) | -2.4 (-9.3, +4.0) | -1.2 (-10.2, +7.2) | +6.8 (-3.5, +14.8) | -1.1 (-10.0, +6.7) | -4.0 (-11.1, +3.6) | +4.9 (-2.8, +12.3) | +3.4 (-4.5, +10.3) | +6.1 (-2.7, +13.7) | +4.4 (-5.0, +12.2) | -1.1 (-13.3, +8.4) | yes | 0 |
| `chem_action_type` = activator | 129/4069 | **-11.5 (-21.7, -2.3)** | **-13.2 (-22.7, -3.8)** | **-15.0 (-23.8, -5.5)** | -13.8 (-26.2, +0.9) | -9.5 (-23.9, +2.4) | -8.0 (-20.8, +4.6) | **-14.9 (-26.8, -2.7)** | **-13.8 (-27.9, -2.8)** | – | -8.6 (-25.2, +6.9) | **-12.8 (-23.7, -2.5)** | -10.1 (-26.1, +3.3) | yes | 6 |
| `chem_action_type` = agonist | 822/3376 | **+5.9 (+0.6, +11.1)** | +2.0 (-2.9, +7.3) | **+4.8 (+0.1, +9.4)** | **+12.9 (+5.7, +20.3)** | +2.1 (-4.0, +8.5) | **+6.9 (+0.1, +13.7)** | **+6.6 (+0.1, +12.5)** | +4.8 (-1.1, +10.3) | – | **+6.5 (+0.5, +12.0)** | +3.6 (-5.6, +12.0) | +3.2 (-7.1, +12.3) | yes | 6 |
| `chem_action_type` = antagonist | 692/3506 | -3.1 (-8.5, +2.3) | -1.0 (-6.2, +4.6) | -1.3 (-6.1, +4.3) | +0.9 (-6.3, +8.2) | -5.8 (-12.7, +0.7) | -2.1 (-9.3, +5.3) | -0.4 (-7.9, +7.2) | -3.4 (-9.3, +2.1) | – | -4.1 (-10.4, +2.4) | -2.1 (-10.0, +5.5) | +0.7 (-9.8, +10.2) | yes | 0 |
| `chem_action_type` = inhibitor | 2070/2128 | +1.3 (-3.1, +5.6) | +1.6 (-2.6, +5.4) | +1.0 (-3.0, +4.9) | **-6.3 (-12.0, -0.9)** | **+5.8 (+0.6, +10.5)** | -0.1 (-5.8, +5.7) | -0.2 (-5.6, +4.9) | +1.3 (-3.1, +6.0) | – | -0.6 (-6.2, +4.9) | +4.4 (-1.1, +9.7) | +0.2 (-6.8, +8.3) | no | 2 |
| `chem_action_type` = multiple | 196/4002 | -2.5 (-10.5, +5.5) | +1.6 (-6.3, +9.6) | +0.6 (-7.2, +8.0) | +5.8 (-8.3, +19.0) | -6.4 (-16.3, +4.3) | – | – | -2.7 (-11.5, +6.6) | – | +3.4 (-5.9, +14.1) | **-11.2 (-22.2, -0.2)** | – | no | 1 |
| `chem_action_type` = other | 289/3909 | -5.9 (-13.4, +2.2) | -3.3 (-11.3, +4.6) | -4.8 (-13.4, +2.6) | -2.9 (-14.2, +8.8) | -7.6 (-15.8, +1.5) | **-10.8 (-21.4, -0.2)** | -7.8 (-20.5, +2.7) | -2.3 (-11.4, +7.0) | – | -6.0 (-15.7, +4.2) | -5.7 (-16.2, +4.0) | -4.2 (-17.8, +9.1) | yes | 1 |

Explore to confirm, 33 of 39 columns keep their sign.

**Most robust** (interval excludes zero in at least 9 of the variants): `expr_n_tissues` (12), `expr_tau` (11), `target_location` = intracellular (11), `expr_vital_max` (10), `depmap_common_essential` (9), genetic support at start (9).

## Genuinely nothing

Columns whose interval excludes zero in none of the variants above. Not 'proved absent': the intervals are wide. But nothing here, in any slice, distinguishes them from zero:

- `depmap_lineage_effect` (phase-only +2.7)
- `depmap_lineage_minus_other` (phase-only -9.6)
- `expr_disease_tissue` (phase-only -3.2)
- `chem_rule_of_five_violations` (phase-only -1.1)
- `target_class` = Transcription factor (phase-only +0.4)
- `target_location` = multiple (phase-only -2.2)
- `target_location` = secreted (phase-only +4.0)
- `chem_action_type` = antagonist (phase-only -3.1)

The low-drug-likeness flag is not on this list as a column (see `chem_qed`), but as a flag it did little: with it on, the pooled gap (2015-2020, phase only) is -2.9 (-9.3 to +3.2). It stays in the score because the score was declared with it.

## Is the pattern across columns more than luck?

With 39 contrasts, some would keep their sign by chance; the shuffled-outcome count says how many. **Caution:** the shuffle moves outcomes between individual trials, so it ignores that trials of one lead drug share biology and tend to share outcomes. That makes the shuffled counts too tightly spread, so these P values are optimistic. The lead-drug bootstrap intervals in the tables are the numbers to trust; this section is a rough guide.

- **phase only**: 33 of 39 contrasts keep their sign from explore to confirm. If outcomes were shuffled (within phase and period, 2000 draws, keeping the correlation between columns) the count would be 19.5 on average, 95% range 12 to 27; P(null >= observed) = 0.0005.
  - Confirm restricted to the 848 trials whose lead drug is in no pre-2015 trial: **23 of 39** keep their sign; shuffled average 19.5, 95% range 12 to 27; P = 0.2280.
- Primary (2+ minus 0 flags, 2015 to 2020, phase only): -8.1 per 100; permutation P (one-sided, null <= observed) = 0.0005, two-sided 0.0010.
- **phase + area**: 26 of 39 contrasts keep their sign from explore to confirm. If outcomes were shuffled (within phase and area and period, 2000 draws, keeping the correlation between columns) the count would be 19.6 on average, 95% range 12 to 27; P(null >= observed) = 0.0610.
  - Confirm restricted to the 848 trials whose lead drug is in no pre-2015 trial: **24 of 39** keep their sign; shuffled average 19.6, 95% range 13 to 27; P = 0.1475.

## What it adds to models without biology (the measured increment)

The flag count, then the logistic score, added to a phase-only model and to the registration model (`baselines`), all fitted on pre-2015 trials and scored on 2015-2020 trials with a mapped target, today's links. 'Spread within phase' = successes per 100 in the third of trials the model ranks highest minus the lowest third, in Phase 2 / Phase 3. Change columns are paired differences against the model without biology, bootstrapped over lead drugs. The logistic score for the pre-2015 rows is in-sample (it was fitted on them), so its added value on the training rows is optimistic; the 2015-2020 scoring is out of sample. A phase-only model gives every trial in a phase the same prediction, so its spread within a phase is only the tie-break (a fixed tiny jitter) and means nothing.

| model | AUROC | spread within phase 2 / 3 | change in AUROC | change in spread, phase 2 / phase 3 |
|---|---|---|---|---|
| phase only | 0.627 | -6.4 / +4.6 |  |  |
| phase only + flag count | 0.651 | +10.6 / +9.1 | +0.024 (+0.006 to +0.039) | +17.0 (+2.3 to +27.6) / +4.6 (-7.7 to +14.0) |
| phase only + logistic biology score | 0.660 | +14.3 / +20.6 | +0.033 (+0.008 to +0.056) | +20.8 (+7.6 to +32.3) / +16.0 (+4.2 to +25.9) |
| registration | 0.719 | +27.9 / +34.0 |  |  |
| registration + flag count | 0.720 | +27.9 / +31.7 | +0.001 (-0.004 to +0.006) | +0.0 (-3.4 to +3.7) / -2.3 (-4.4 to +2.3) |
| registration + logistic biology score | 0.717 | +27.9 / +31.7 | -0.002 (-0.013 to +0.009) | +0.0 (-5.3 to +6.1) / -2.3 (-6.4 to +2.7) |

## How little AUROC can an effect this size move? (a simulation of best cases)

This section is a **simulation, not a measurement of the biology score**; the measured increment is the previous section. AUROC is the chance that a trial that succeeded ranks above one that did not (0.5 = coin flip). A steady gap of several per 100 on a flag that applies to a minority of trials is a small move in AUROC. 'Share' and 'difference' are measured on the 2015-2020 trials (the difference is the phase-pooled gap, flag on minus off; the last row is an assumed example). The simulated column adds that effect, known exactly, to the registration model's predictions, so it is a best case: real use would also have to estimate the effect.

| effect | share of trials | difference, per 100 | AUROC on its own (formula) | AUROC gain on top of registration (simulated) |
|---|---|---|---|---|
| activator | 0.04 | -12.6 | 0.510 | +0.0017 |
| intracellular target | 0.05 | -13.5 | 0.513 | +0.0022 |
| common-essential gene | 0.06 | -10.4 | 0.513 | +0.0017 |
| broad expression | 0.36 | -6.2 | 0.531 | +0.0028 |
| genetic support (for comparison) | 0.12 | +16.5 | 0.536 | +0.0079 |
| 2+ flags vs 0 flags (the score) | 0.46 | -8.1 | 0.543 | +0.0048 |
| hypothetical: half the trials, 15 points | 0.50 | -15.0 | 0.580 | +0.0159 |

Base rate used for the shift and the formula: p = 0.63, the pre-2015 success rate. Formula: AUROC = 0.5 + 0.5 q (1 - q) |d| / (p (1 - p)). 300 simulated draws per row.

## Honest limits

- **The features were chosen after seeing all development data.** `dev_science.md` showed the six flags, and it used 2015 to 2020 as well as before. So the 2015-2020 'confirm' here is a fresh test of the combined score, its thresholds and its grouping, not of the choice of features. The real confirmation is the sealed test set.
- **Replication is weaker on new lead drugs.** Of 39 per-column contrasts, 33 keep their sign from before 2015 to 2015-2020 (shuffled-outcome 97.5th percentile 27). Restricted to trials whose lead drug is new since 2015 it is 23 of 39 (P = 0.23: a shuffle gives that often). That is weaker support than the full-sample count.
- **The score is not blind to disease or drug type** (see 'What it is'): four inputs describe the target in the trial's own disease tissue or cancer lineage, and chemistry exists only for small molecules.
- **A large share of the flag-count gap is disease area.** Holding area fixed takes 2+ flags minus 0 from -8.1 to -4.5 per 100 (interval -11.4 to +2.3).
- **It adds almost nothing once registration columns are known.** An effect this size moves AUROC by a few thousandths on top of the registration model in the simulation, and the measured increment (above) agrees: a few hundredths of AUROC over phase alone, nothing measurable over the registration model.
- **Today's links are a release from after the trials started.** Everything built from them is 'known after the start', not known at the start; the 2020 versions are the closer-in-time check.
- The tables read here (`whathow_<version>`, `precedent`) are the repo's current builds.
