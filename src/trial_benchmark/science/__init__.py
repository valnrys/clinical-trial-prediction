"""Do public science databases help predict whether a trial succeeds?

The question behind this package: once a trial's design is known, does anything we can look
up about its drug's target (is the gene essential to cancer cells, where is it expressed, how
connected is the protein, what class of protein is it, what does the molecule look like) tell
us more about whether the trial will succeed? Each source gets one module and a handful of
features, all joined into one table, `data/interim/science.parquet`, with one row per trial.

    python -m trial_benchmark.science.table    # download, join, check -> science.parquet
    python -m trial_benchmark.science.report   # results/dev_science.md (development trials only)

Modules: `targets` (trial -> investigational drug -> target genes; gene identifiers),
`depmap`, `expression`, `networks`, `annotations`, `chemistry` (one per source), `fetch`
(checked downloads), `table` (the join), `checks` (known-biology checks), `trace` (a spot check of random trials against the raw files) and `report`.

The limits all the sources share
--------------------------------
**Which drug, and which targets, are today's.** Which drug a trial tested, and which genes that
drug acts on, come from today's Open Targets mapping (release 26.09). A drug is catalogued, and
given targets, mostly because it progressed far enough to be worth cataloguing, so "this trial
has a mapped target" carries hindsight, and trials of drugs that stalled early are somewhat more
often unmapped. That limit is stated here once and is not repeated on each feature.

**A drug's targets are often a whole complex, and the number of targets shows up in the
features.** Open Targets lists every gene of a protein complex the drug binds: pregabalin has
26 calcium-channel subunits, riluzole 10 sodium channels. A trial's value is then a maximum or
a minimum over those genes, and the more genes there are, the more extreme that value is by
chance alone. In these data several features correlate with the number of targets (Spearman
0.3 to 0.5 in size), and the number of targets is itself "after the start" knowledge this package
deliberately does not read. Trials whose drug has more than 5 targets succeeded 46 to 49% of
the time (6 to 10 and more than 10 targets) against 61% for 1 to 5 targets. So part of any gene-level feature is the target count
in disguise. The report repeats its per-feature comparison on single-target drugs and on drugs
with at most three targets to show how much.

How the features are dated (see `dating`)
----------------------------------------
- A measurement of a gene, tissue or molecule (an essentiality score, an expression level, a
  molecular weight) does not depend on any trial's result: TIMELESS. It does not mean the
  number existed when the trial began: DepMap, GTEx and the screen collections were published
  years after many of these trials started. If anything that flatters the features.
- A feature that also reads the trial's own conditions (the matched cancer lineage, the
  matched tissue) is PROTOCOL: current registry text, which sponsors can edit.
- `has_target` and `n_target_genes` (does the drug have a known target, and how many) are
  AFTER_START: the hindsight above. Honest models must not read them.

Science columns exist only for trials with a target
---------------------------------------------------
Every science column, chemistry included, is empty for a trial with no mapped target. A model
that is told "this value is missing" has been told the trial is unmapped, which is hindsight.
`models.biology` therefore never shows a model a missing value or a missing-value flag for a
science column: it centres each column on its training mean (over trials that have a value) and
fills missing with zero, so a trial with no value adds nothing to a linear score. Models that
read science columns only are defined on trials that have a value and predict one constant for
the rest. `models.biology` says exactly how.
"""

from trial_benchmark.science import annotations, chemistry, depmap, expression, networks, targets

SOURCES = {
    "depmap": depmap,
    "expression": expression,
    "networks": networks,
    "annotations": annotations,
    "chemistry": chemistry,
}
# Column -> dating, for the whole table (every module lists its own columns).
FEATURES: dict[str, str] = {
    **targets.FEATURES,
    **{name: label for module in SOURCES.values() for name, label in module.FEATURES.items()},
}
