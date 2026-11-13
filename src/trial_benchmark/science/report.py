"""Write results/dev_science.md: do the public science databases predict trial success?

    uv run python -m trial_benchmark.science.report

Development trials only (completed before 2021). The test and late trials appear in one place
only, the join funnel saved when the table was built, as counts of trials with feature values;
no outcome is read for them. Nothing here opens the held-out labels.

What it measures:

1. The join funnel: how many trials get a drug, a target and a value from each source.
2. The known-biology checks (`checks.py`), and a spot check of 25 random trials from the raw
   files to the table (`trace.py`).
3. For each feature, the adjusted difference in success rate between trials at the top and the
   bottom of the feature (the top and bottom third of its values, or the two sides of a yes/no,
   or one level of a category against the rest). "Adjusted" means compared within groups of
   trials that share phase, sponsor type, therapeutic area and drug type (modality;
   `adjusted.py`, a Mantel-Haenszel average), with an interval that resamples whole lead drugs, because the
   trials of one drug are alike and should not count as independent. Repeated for drugs with at
   most three targets and with one target.
4. An explore / confirm split in time: the cut points of every contrast are fixed on trials
   finishing before 2015, then the same contrast is measured on 2015 to 2020. Every difference
   is printed with its raw sign, always first-named group minus second-named group.
5. A positive control: human genetic support, a signal established in published studies, is blanked where
   the trial has no target (so it has the same missing values as the science columns) and sent
   through the same per-feature and model code. It shows what each test can and cannot see.
6. The models of `models/biology.py`, scored the way every model in this project is
   (`validate.py`: three time-ordered rounds, mean within-round AUROC).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict

from trial_benchmark import metrics, validate
from trial_benchmark.adjusted import N_RESAMPLES, adjusted
from trial_benchmark.models import biology
from trial_benchmark.models.linear import REGISTRATION_NAME
from trial_benchmark.models.registry import ModelSpec, all_specs
from trial_benchmark.science import checks, table, trace
from trial_benchmark.science import targets as tg

REPORT_PATH = Path(__file__).resolve().parents[3] / "results" / "dev_science.md"
LADDER_NAME = "ladder: registration + endpoint + history + genetics + words"
LADDER_SCIENCE = "biology: ladder, all four tables + science"
TREES = "biology: boosted trees, all sources"
LOGISTIC_ALL = "biology: logistic, all sources"
STRATA_BASE = ["phase", "sponsor_class", "therapeutic_area"]
STRATA = [*STRATA_BASE, "modality"]  # drug type is known at registration
CLUSTER = "lead_drug"
MIN_GROUP = 30  # fewest trials in each of the two groups compared
MIN_LEVEL = 100  # a category level needs this many trials to be compared with the rest
EXPLORE_BEFORE = 2015  # trials finishing before this year explore; 2015 to 2020 confirm
EXPLORE_RESAMPLES = 250  # the explore step only needs the direction
CONTROL = "control_genetic_support"
CONTROL_SPEC = "control: registration + genetic support"
POPULATIONS = {"all": "all trials with a value", "few": "at most 3 targets", "single": "one target"}
# What the candidate signals are adjusted for: the main strata, without drug type, and with target class.
ADJUSTMENTS = {
    "main": ("phase, sponsor, area, drug type", STRATA),
    "no_modality": ("phase, sponsor, area", STRATA_BASE),
    "with_class": ("phase, sponsor, area, drug type, target class", [*STRATA, "target_class"]),
}

# One plain line per feature: what it is, and how several targets become one value.
DESCRIPTIONS = {
    "depmap_common_essential": "target is on DepMap's common-essential list (any target)",
    "depmap_mean_effect": "mean CRISPR gene effect over all cell lines; lower means more dependent (most dependent target)",
    "depmap_fraction_dependent": "share of cell lines that depend on the gene (most dependent target)",
    "depmap_lineage_effect": "cancer trials: mean gene effect in the trial's cancer lineage; lower means more dependent (most dependent tumour-intrinsic target)",
    "depmap_lineage_minus_other": "cancer trials: that mean minus the mean in other lineages; lower means more specific to the lineage",
    "expr_tau": "tissue specificity, GTEx; higher means one tissue (lowest tau, the broadest target)",
    "expr_n_tissues": "number of tissues expressing the gene, HPA (highest)",
    "expr_vital_max": "highest expression in heart, liver, kidney, brain or blood, GTEx (highest)",
    "expr_disease_tissue": "expression in the disease's matched tissue, HPA (best target)",
    "expr_disease_tissue_top3": "matched tissue is among the gene's top three (any target)",
    "net_viability_hit_fraction": "share of independent viability screens where the gene is a hit (highest)",
    "net_other_hit_fraction": "share of other knockout screens where the gene is a hit (highest)",
    "net_partners_per_paper": "log2 of BioGRID physical partners per paper; high means large screens, not centrality (highest)",
    "target_class": "level-1 protein class (shared class, else 'multiple')",
    "target_location": "secreted, membrane, nuclear or intracellular (shared, else 'multiple')",
    "constraint": "genetic constraint, higher is less tolerant (highest)",
    "mouse_knockout": "mouse-knockout score, higher is less tolerant (highest)",
    "chem_mol_weight": "lead drug: molecular weight",
    "chem_logp": "lead drug: logP",
    "chem_polar_surface_area": "lead drug: polar surface area",
    "chem_qed": "lead drug: drug-likeness (QED)",
    "chem_rule_of_five_violations": "lead drug: rule-of-five violations",
    "chem_action_type": "lead drug: inhibitor, antagonist, agonist, activator, other or 'multiple'",
}
# The two candidate patterns the reviewers noticed on the full sample (see section 5).
BROAD_EXPRESSION = ("expr_n_tissues", "expr_vital_max", "expr_tau")
HEADLINE_BROAD = "expr_n_tissues"
LINEAGE = ("depmap_lineage_minus_other",)


# ---------------------------------------------------------------- the data


def wanted_specs() -> list[ModelSpec]:
    """The biology specs, the control, the registration logistic and the ladder rung without science."""
    wanted = [
        s
        for s in all_specs()
        if s.family == "biology" or s.name in (REGISTRATION_NAME, LADDER_NAME)
    ]
    specs, missing = validate.split_specs(wanted)
    if missing:
        raise ValueError(f"Tables missing for {sorted(missing)}.")
    return [*specs, control_spec()]


def load_frame(specs: list[ModelSpec]) -> pl.DataFrame:
    """Development trials with every table the specs read, plus the positive-control column."""
    frame = validate.load_development_with({t for s in specs for t in s.tables})
    return frame.with_columns(
        pl.when(pl.col("has_target")).then(pl.col("supported_at_start")).alias(CONTROL)
    )


class Result(NamedTuple):
    n_exposed: int
    n_reference: int
    rate_exposed: float
    rate_reference: float
    difference: float  # adjusted, as a share (0.05 is 5 points)
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def same_sign(first: Result, second: Result) -> bool:
    """True if both differences point the same way (a difference of exactly zero does not)."""
    return first.difference * second.difference > 0


class Contrast(NamedTuple):
    feature: str
    compared: str
    level: str | None = None  # the text level set against the rest


# ---------------------------------------------------------------- one contrast


def contrasts_for(frame: pl.DataFrame, names: list[str]) -> list[Contrast]:
    """One contrast per number or yes/no feature, one per text level with enough trials."""
    found = []
    for name in names:
        if frame.schema[name] == pl.String:
            counts = frame[name].drop_nulls().value_counts().filter(pl.col("count") >= MIN_LEVEL)
            found += [Contrast(name, f"{v} minus other levels", v) for v in sorted(counts[name])]
        elif frame.schema[name] == pl.Boolean:
            found.append(Contrast(name, "yes minus no"))
        else:
            found.append(Contrast(name, "top third minus bottom third"))
    return found


def groups(
    frame: pl.DataFrame, contrast: Contrast, cuts: tuple[float, float] | None = None
) -> tuple[pl.DataFrame, tuple[float, float] | None]:
    """The trials compared, with a yes/no column `exposed`, and the cut points used (numbers).

    Top third is strictly above the upper cut and bottom third at or below the lower one, so a
    feature with many tied values (zero, say) splits cleanly into "zero" against "more".
    """
    name = contrast.feature
    known = frame.filter(pl.col(name).is_not_null())
    if contrast.level is not None:
        return known.with_columns((pl.col(name) == contrast.level).alias("exposed")), None
    if frame.schema[name] == pl.Boolean:
        return known.with_columns(pl.col(name).alias("exposed")), None
    low, high = cuts or (known[name].quantile(1 / 3), known[name].quantile(2 / 3))
    part = known.filter((pl.col(name) > high) | (pl.col(name) <= low))
    return part.with_columns((pl.col(name) > high).alias("exposed")), (low, high)


def estimate(
    frame: pl.DataFrame,
    contrast: Contrast,
    cuts: tuple[float, float] | None = None,
    resamples: int = N_RESAMPLES,
    strata: list[str] = STRATA,
) -> tuple[Result | None, tuple[float, float] | None]:
    """The adjusted difference (first group minus second) for one contrast, or None if too small."""
    part, used = groups(frame, contrast, cuts)
    n1 = int(part["exposed"].sum())
    n0 = part.height - n1
    if min(n1, n0) < MIN_GROUP:
        return None, used
    rate = {
        flag: part.filter(pl.col("exposed") == flag)["any_arm"].mean() for flag in (True, False)
    }
    difference, low, high = adjusted(
        part, "exposed", "any_arm", strata, CLUSTER, measure="difference", n_resamples=resamples
    )
    return Result(n1, n0, rate[True], rate[False], difference, low, high), used


def explore_confirm(
    frame: pl.DataFrame, contrast: Contrast, strata: list[str] = STRATA
) -> tuple[Result, Result] | None:
    """(explore result, confirm result), both with their raw signs, or None.

    Cut points are fixed on trials finishing before `EXPLORE_BEFORE`; the same contrast is then
    measured on the later ones. Both differences are first group minus second group; use
    `same_sign` to ask whether the sign held.
    """
    year = frame["primary_completion_date"].dt.year()
    early = frame.filter(year < EXPLORE_BEFORE)
    first, cuts = estimate(early, contrast, resamples=EXPLORE_RESAMPLES, strata=strata)
    if first is None:
        return None
    second, _ = estimate(frame.filter(year >= EXPLORE_BEFORE), contrast, cuts, strata=strata)
    if second is None:
        return None
    return first, second


def subsets(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """The trials compared: every trial with a value, those with at most 3 targets, with one."""
    targets = pl.col("n_target_genes")
    return {
        "all": frame,
        "few": frame.filter(targets.is_between(1, 3)),
        "single": frame.filter(targets == 1),
    }


# ---------------------------------------------------------------- text helpers


def points(value: float) -> str:
    return f"{100 * value:+.1f}".replace("-", "−")


def cell(result: Result | None) -> str:
    if result is None:
        return "too few"
    return f"{points(result.difference)} ({points(result.low)} to {points(result.high)})"


def label(contrast: Contrast) -> str:
    return f"`{contrast.feature}`" + ("" if contrast.level is None else f", {contrast.level}")


# ---------------------------------------------------------------- the tables


def per_feature(frame: pl.DataFrame, names: list[str]) -> list[tuple[Contrast, dict]]:
    """Every contrast on all trials, on at most three targets, and on one target."""
    parts = subsets(frame)
    return [
        (c, {key: estimate(part, c)[0] for key, part in parts.items()})
        for c in contrasts_for(frame, names)
    ]


def feature_table(rows: list[tuple[Contrast, dict]]) -> list[str]:
    lines = [
        "| feature | compared (difference = first group minus second) | trials (first, second) | success rate (first, second) | all: adjusted difference, points (95% interval) | at most 3 targets | one target |",
        "|---|---|---|---|---|---|---|",
    ]
    for contrast, results in rows:
        everyone = results["all"]
        trials, rates = ("too few or too tied", "") if everyone is None else (
            f"{everyone.n_exposed:,}, {everyone.n_reference:,}",
            f"{everyone.rate_exposed:.0%}, {everyone.rate_reference:.0%}",
        )  # fmt: skip
        lines.append(
            f"| {label(contrast)} | {contrast.compared} | {trials} | {rates} | {cell(everyone)} | "
            f"{cell(results['few'])} | {cell(results['single'])} |"
        )
    return lines


def confirm_rows(frame: pl.DataFrame, names: list[str]) -> list[tuple[Contrast, tuple | None]]:
    return [(c, explore_confirm(frame, c)) for c in contrasts_for(frame, names)]


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def survives(first: Result, second: Result) -> bool:
    """The sign held and the confirm interval excludes zero."""
    return same_sign(first, second) and second.excludes_zero()


def confirm_table(rows) -> list[str]:
    lines = [
        "| feature | compared (difference = first group minus second) | explore: before 2015, points | confirm: 2015 to 2020, points (95% interval) | same sign | same sign and confirm interval excludes zero |",
        "|---|---|---|---|---|---|",
    ]
    for contrast, pair in rows:
        if pair is None:
            lines.append(f"| {label(contrast)} | {contrast.compared} | too few | | | |")
            continue
        first, second = pair
        lines.append(
            f"| {label(contrast)} | {contrast.compared} | {points(first.difference)} | {cell(second)} | "
            f"{yes_no(same_sign(first, second))} | {yes_no(survives(first, second))} |"
        )
    return lines


def confirm_summary(rows) -> tuple[int, int, list[str]]:
    """(contrasts measured, sign held, those that survive, as text with the raw confirm sign)."""
    done = [(c, pair) for c, pair in rows if pair is not None]
    held = sum(1 for _, (first, second) in done if same_sign(first, second))
    clear = [
        f"{label(c)} {points(second.difference)}"
        for c, (first, second) in done
        if survives(first, second)
    ]
    return len(done), held, clear


def candidate_rows(frame: pl.DataFrame) -> dict:
    """(feature, population, adjustment) -> (full result, explore result, confirm result).

    The broad-expression features are measured on all trials with a value and on single-target
    drugs, under each adjustment; the lineage feature on all three populations with the main
    adjustment. A part that is too small is None.
    """
    parts = subsets(frame)
    wanted = [(f, ("all", "single"), tuple(ADJUSTMENTS)) for f in BROAD_EXPRESSION]
    wanted += [(f, ("all", "few", "single"), ("main",)) for f in LINEAGE]
    rows = {}
    for feature, populations, adjustments in wanted:
        for population in populations:
            for adjustment in adjustments:
                strata = ADJUSTMENTS[adjustment][1]
                for contrast in contrasts_for(parts[population], [feature]):
                    full, _ = estimate(parts[population], contrast, strata=strata)
                    pair = explore_confirm(parts[population], contrast, strata)
                    rows[(feature, population, adjustment)] = (full, *(pair or (None, None)))
    return rows


def candidate_table(rows: dict) -> list[str]:
    lines = [
        "| feature | trials | adjusted for | all development trials with a value, points (95% interval) | explore: before 2015 | confirm: 2015 to 2020, points (95% interval) | same sign |",
        "|---|---|---|---|---|---|---|",
    ]
    for (feature, population, adjustment), (full, explore, confirm) in rows.items():
        first = "too few" if explore is None else points(explore.difference)
        agree = "" if explore is None or confirm is None else yes_no(same_sign(explore, confirm))
        lines.append(
            f"| `{feature}` | {POPULATIONS[population]} | {ADJUSTMENTS[adjustment][0]} | "
            f"{cell(full)} | {first} | {cell(confirm)} | {agree} |"
        )
    return lines


def candidate_cell(rows: dict, feature: str, population: str, adjustment: str = "main") -> str:
    full = rows[(feature, population, adjustment)][0]
    return "too few trials to measure" if full is None else cell(full)


def includes_zero_result(result: Result | None) -> bool:
    return result is None or not result.excludes_zero()


def interval_verdict(rows: dict, feature: str, population: str, adjustment: str = "main") -> str:
    full = rows[(feature, population, adjustment)][0]
    return f"the interval {'includes' if includes_zero_result(full) else 'excludes'} zero"


def top_third_range(frame: pl.DataFrame, feature: str) -> tuple[int, int]:
    """(lowest, highest) value of the top third of a count feature."""
    part, _ = groups(frame, Contrast(feature, "top third minus bottom third"))
    top = part.filter(pl.col("exposed"))[feature]
    return int(top.min()), int(top.max())


def top_third_text(frame: pl.DataFrame, feature: str) -> str:
    """What the top third of a count feature is, on all trials with a value and on one-target drugs."""
    low_all, high_all = top_third_range(frame, feature)
    low_one, high_one = top_third_range(subsets(frame)["single"], feature)
    everyone = (
        f'exactly "expressed in all {high_all} tissues"'
        if low_all == high_all
        else f"expressed in {low_all} to {high_all} tissues"
    )
    return (
        f"on all trials with a value the top third is {everyone}; among single-target drugs "
        f"it is {low_one} to {high_one} tissues"
    )


def plural(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


# ---------------------------------------------------------------- the models


def control_spec() -> ModelSpec:
    return ModelSpec(
        CONTROL_SPEC,
        "biology",
        "Positive control: the registration logistic plus human genetic support, blank without a target.",
        lambda train: biology.fit_with_registration(train, [CONTROL]),
        tables=(biology.TABLE, "genetics"),
    )


def evaluate_models(specs: list[ModelSpec], frame: pl.DataFrame) -> dict:
    """Run the harness on the specs and score all validation trials and the mapped ones."""
    result = validate.run_time_ordered(specs, frame)
    rows = np.concatenate([validate.time_round(frame, a, b)[1] for a, b in validate.ROUNDS])
    scored = frame[rows]
    pairs = [(LADDER_SCIENCE, LADDER_NAME), (TREES, LOGISTIC_ALL)]
    populations = {}
    for name, mask in (
        ("all", np.ones(len(rows), dtype=bool)),
        ("mapped", scored["has_target"].to_numpy()),
    ):
        predictions = {n: p[mask] for n, p in result.predictions.items()}
        rounds = result.round_of[mask]
        evaluation = validate.summarise(result.outcomes[mask], predictions, rounds)
        _, extra = validate.mean_auroc_bootstrap(
            result.outcomes[mask],
            {n: predictions[n] for pair in pairs for n in pair},
            rounds,
            pairs,
        )
        in_part = scored.filter(pl.Series(mask))
        counts = {
            f"biology: logistic, {source} only": int(
                biology.has_value(in_part, biology.honest_columns((key,))).sum()
            )
            for key, source in biology.SOURCE_LABELS.items()
        }
        populations[name] = (int(mask.sum()), evaluation, extra, counts)
    return {
        "specs": specs,
        "populations": populations,
        "training": training_counts(frame),
        "target_count": target_count_check(result, scored, frame),
    }


def training_counts(frame: pl.DataFrame) -> dict[str, list[int]]:
    """Per round, the training trials in all, and those with a value from each source."""
    train = [frame[validate.time_round(frame, first, last)[0]] for first, last in validate.ROUNDS]
    counts = {"all": [part.height for part in train]}
    for key, source in biology.SOURCE_LABELS.items():
        columns = biology.honest_columns((key,))
        counts[source] = [int(biology.has_value(part, columns).sum()) for part in train]
    counts["any"] = [int(biology.has_value(part, biology.ALL).sum()) for part in train]
    return counts


def target_count_check(result, scored: pl.DataFrame, frame: pl.DataFrame) -> dict:
    """Is the boosted trees' lead over the logistic just the number of targets?

    How well the science columns say "several targets rather than one"; how well the number of
    targets alone ranks success on mapped validation trials; and the two models' AUROC among
    one-target and among several-target trials separately.
    """
    mapped = scored["has_target"].to_numpy()
    count = scored["n_target_genes"].fill_null(0).to_numpy()
    several = count > 1
    names = (TREES, LOGISTIC_ALL)
    alone = validate.summarise(
        result.outcomes[mapped],
        {"fewer targets": -count[mapped].astype(float)},
        result.round_of[mapped],
    )
    check = {
        "columns_say_several": columns_predict_several(frame),
        "count_alone": validate.auroc_cell(alone, "fewer targets"),
        "within": {},
    }
    for label, mask in (
        ("one target", mapped & (count == 1)),
        ("several targets", mapped & several),
    ):
        evaluation = validate.summarise(
            result.outcomes[mask],
            {n: result.predictions[n][mask] for n in names},
            result.round_of[mask],
        )
        check["within"][label] = (
            int(mask.sum()),
            {n: validate.auroc_cell(evaluation, n) for n in names},
        )
    return check


def columns_predict_several(frame: pl.DataFrame) -> float:
    """How well the science columns alone say "several targets, not one", on mapped dev trials.

    Boosted trees, five folds that keep each lead drug on one side, the AUROC of the held-out
    predictions. It is high because a maximum or minimum over many genes behaves differently
    from a single gene's value; that is the leak the report warns about.
    """
    mapped = frame.filter(pl.col("has_target"))
    centring = biology.fit_centring(mapped, biology.ALL)
    features = centring.features()
    columns = centring.apply(mapped).select(features.boolean + features.numeric).to_numpy()
    several = (mapped["n_target_genes"] > 1).to_numpy()
    held_out = cross_val_predict(
        HistGradientBoostingClassifier(max_depth=3),
        columns,
        several,
        cv=GroupKFold(5),
        groups=validate.drug_groups(mapped),
        method="predict_proba",
    )[:, 1]
    return metrics.auroc(several, held_out)


def is_science_only(name: str) -> bool:
    return name.startswith(("biology: logistic,", "biology: boosted trees"))


def model_table(specs, population, which: str) -> list[str]:
    n, evaluation, _, source_counts = population
    order = sorted((s.name for s in specs), key=lambda name: -evaluation.aurocs[name].value)
    lines = [
        "| model | mean AUROC (95% interval) | minus registration (95% interval) | validation trials with a value |",
        "|---|---|---|---|",
    ]
    for name in order:
        if which == "all" and is_science_only(name):
            lines.append(f"| {name} | not meaningful (a constant for unmapped trials) | | |")
            continue
        values = source_counts.get(name)
        lines.append(
            f"| {name} | {validate.auroc_cell(evaluation, name)} | "
            f"{validate.difference_cell(name, evaluation)} | "
            f"{'' if values is None else f'{values:,} of {n:,}'} |"
        )
    return lines


def difference(evaluation, name: str) -> tuple[float, float, float]:
    d = evaluation.differences[name]
    return d.value, d.interval.low, d.interval.high


# ---------------------------------------------------------------- the report


def read_first(parts: dict) -> list[str]:
    """The top of the report: what was tested, what it shows, what it cannot show."""
    plus = validate.plus_minus
    m, a, c = parts["m_diff"], parts["a_diff"], parts["control_model"]
    survivors = parts["survivors"]
    spot_ok = parts["spot_agreed"] == parts["spot_total"]
    cand = parts["candidates"]
    worst_value, worst_name = parts["worst"]
    low, high = parts["training"][worst_name]
    return [
        "# Do public science databases help predict trial success?",
        "",
        "Development trials only (completed before 2021). No test or late trial outcome is read.",
        "",
        "## Read this first",
        "",
        (
            f"Five public sources about a trial's drug target (DepMap, tissue expression, protein "
            f"networks, protein annotations, drug chemistry) give {parts['n_features']} features. They "
            f"exist for the {parts['n_mapped']:,} of {parts['n_all']:,} validation trials whose drug has a "
            "known target. Two tests were run: does adding them improve how well a model ranks trials, "
            "and, feature by feature, do trials at the top and bottom of a feature differ in success rate? "
            "Every difference in this report is printed with its raw sign: first-named group minus "
            "second-named group, so a negative number means the first group did worse."
        ),
        "",
        (
            f"- **Adding the sources to a model built on registration details did not improve how well "
            f"it ranked trials.** Registration plus all five sources scores {plus(m[0])} AUROC against "
            f"registration alone on the mapped validation trials (95% interval {plus(m[1])} to "
            f"{plus(m[2])}); on all {parts['n_all']:,} trials, {plus(a[0])} ({plus(a[1])} to {plus(a[2])}). "
            f"One source at a time: {parts['per_source']}."
        ),
        (
            f"- **The same test barely registers human genetic support, a signal established in "
            f"published studies.** As a model increment it scores {plus(c[0])} AUROC ({plus(c[1])} to "
            f"{plus(c[2])}), so the model test cannot reliably see it, although as a single contrast "
            f"it shows {cell(parts['control_feature'])} points (all trials with a value; "
            f'{parts["control_single_text"]}). So "adds less than about 0.01 AUROC" means these '
            "sources do not change how well trials are ranked overall. It does not mean they carry "
            "no information. The per-feature differences are the more sensitive test (section 6)."
        ),
        (
            f"- **Feature by feature, one pattern stood out after the fact. It is a hypothesis for new "
            f"data, not a finding.** For single-target drugs, a target expressed in nearly every "
            f"tissue goes with lower success (in `{HEADLINE_BROAD}`, {parts['top_third']}): "
            f"{cand['a_single']} points, top third minus bottom third, adjusted for phase, sponsor "
            f"type, therapeutic area and drug type. "
            f"Two things weaken it. On all trials with a value the figure is {cand['a_all_no_modality']} "
            f"with registration groups alone, but {cand['a_all']} once drug type is accounted for "
            f"({cand['a_all_verdict']}): antibodies are rarely broadly expressed and succeed more. "
            f"And with target class added as well, the single-target figure is {cand['a_single_class']} "
            f"({cand['a_single_class_verdict']}). It was noticed on the full development sample, so the confirm period of "
            "section 4 is part of what it was noticed on; confirming it needs new data or the test "
            "set (section 5)."
        ),
        (
            f"  Context: there are {parts['n_done']} contrasts (one per feature, one per category "
            f"level); {len(parts['excluding'])} have an interval that excludes zero on all trials "
            f"with a value ({'; '.join(parts['excluding']) or 'none'}). They are correlated, so counting "
            f"how many to expect by chance would mislead. With the cut points fixed before "
            f"{EXPLORE_BEFORE}, {parts['held']} of {parts['n_confirm']} kept their sign on {EXPLORE_BEFORE} "
            f"to 2020 (about half would with no effect at all), and {plural(len(survivors), 'survives', 'survive')} "
            f"with a confirm interval that excludes zero ({'; '.join(survivors) if survivors else 'none'}); "
            f"about {parts['n_confirm'] / 40:.0f} would be expected by luck. A second pattern, in "
            f"cancer trials (a target the trial's cancer lineage depends on more than other lineages), "
            f"is a direction worth testing; it is not distinguishable from zero here. "
            f"All trials with a value: {cand['b_all']}; drugs with at most three targets: "
            f"{cand['b_few']}; single-target drugs: not measurable (section 5)."
        ),
        (
            "- **What the joins were checked against, and what was not.** "
            + (
                "About thirty hand-picked genes and two drugs were checked against known biology and "
                "all passed; that catches a plainly broken join and says nothing about the ordinary "
                "gene. "
                if not parts["failed"]
                else f"These known-biology checks FAILED: {parts['failed']}. Do not rely on this report. "
            )
            + (
                "A spot check followed 25 random mapped trials from their target lists through the "
                "raw DepMap, GTEx, HPA, BioGRID and ORCS files and recomputed every value with "
                f"separate code: {parts['spot_agreed']} of {parts['spot_total']} values agreed"
                + (" (section 2). " if spot_ok else "; the disagreements are listed in section 2. ")
            )
            + "Not checked: which drug a trial tested and its targets (Open Targets), the "
            "disease-to-tissue and disease-to-cancer-lineage rules, and the annotation and chemistry "
            "files beyond the spot check's reading of the slim copies the build wrote."
        ),
        (
            f'- **Part of "adds nothing" is the price of more columns.** Adding columns can lower AUROC '
            f"slightly: the largest drop of any one source is {plus(worst_value)} "
            f"(registration + {worst_name}), because every extra column is fitted on only about "
            f"{low:,} to {high:,} training trials that have a {worst_name} value."
        ),
        (
            "- **Two limits on everything here.** Which drug a trial tested, and its target, come from "
            "today's Open Targets mapping, which favours drugs that progressed far enough to be "
            "catalogued. And Open Targets lists a drug's whole protein complex as targets, so the number "
            "of targets, which is hindsight, leaks into the maximum or minimum of every gene-level "
            "feature (`science` explains). Section 3 repeats the comparison on drugs with at most three "
            "targets and with one target."
        ),
        "",
    ]


def write_report(frame, funnel, results, spot, feature_rows, confirmed, candidates, models) -> str:
    plus = validate.plus_minus
    mapped = models["populations"]["mapped"]
    everyone = models["populations"]["all"]
    science_rows = [r for r in feature_rows if r[0].feature != CONTROL]
    control_rows = [r for r in feature_rows if r[0].feature == CONTROL]
    science_confirmed = [r for r in confirmed if r[0].feature != CONTROL]
    control_confirmed = [r for r in confirmed if r[0].feature == CONTROL]
    combined = "biology: registration + all sources"
    done = [r["all"] for _, r in science_rows if r["all"] is not None]
    n_confirm, held, survivors = confirm_summary(science_confirmed)
    control_model = difference(mapped[1], CONTROL_SPEC)
    control_feature = control_rows[0][1]["all"]
    control_single = control_rows[0][1]["single"]
    parts = {
        "n_features": len(biology.ALL),
        "n_mapped": mapped[0],
        "n_all": everyone[0],
        "m_diff": difference(mapped[1], combined),
        "a_diff": difference(everyone[1], combined),
        "control_model": control_model,
        "control_feature": control_feature,
        "control_single_text": (
            "single-target drugs: "
            + cell(control_single)
            + (", an interval that includes zero" if includes_zero_result(control_single) else "")
        ),
        "per_source": "; ".join(
            f"{name} {plus(difference(mapped[1], f'biology: registration + {name}')[0])}"
            for name in biology.SOURCE_LABELS.values()
        ),
        "n_done": len(done),
        "excluding": [
            f"{label(c)} {points(r['all'].difference)}"
            for c, r in science_rows
            if r["all"] is not None and r["all"].excludes_zero()
        ],
        "candidates": {
            "a_single": candidate_cell(candidates, HEADLINE_BROAD, "single"),
            "a_all": candidate_cell(candidates, HEADLINE_BROAD, "all"),
            "a_all_no_modality": candidate_cell(candidates, HEADLINE_BROAD, "all", "no_modality"),
            "a_all_verdict": interval_verdict(candidates, HEADLINE_BROAD, "all"),
            "a_single_class": candidate_cell(candidates, HEADLINE_BROAD, "single", "with_class"),
            "a_single_class_verdict": interval_verdict(
                candidates, HEADLINE_BROAD, "single", "with_class"
            ),
            "b_all": candidate_cell(candidates, LINEAGE[0], "all"),
            "b_few": candidate_cell(candidates, LINEAGE[0], "few"),
            "b_single": candidate_cell(candidates, LINEAGE[0], "single"),
        },
        "top_third": top_third_text(frame, HEADLINE_BROAD),
        "n_lineage": frame["depmap_lineage_minus_other"].drop_nulls().len(),
        "n_confirm": n_confirm,
        "held": held,
        "survivors": survivors,
        "failed": [r.name for r in results if not r.passed],
        "spot_total": sum(c.values for c in spot),
        "spot_agreed": sum(c.agreed for c in spot),
        "worst": min(
            (difference(mapped[1], f"biology: registration + {name}")[0], name)
            for name in biology.SOURCE_LABELS.values()
        ),
        "training": {
            name: (min(counts), max(counts)) for name, counts in models["training"].items()
        },
    }
    half_width = float(np.median([50 * (r.high - r.low) for r in done]))
    explore, confirm = control_confirmed[0][1] if control_confirmed[0][1] else (None, None)
    control_agree = "" if explore is None else yes_no(same_sign(explore, confirm))
    check = models["target_count"]
    trees_lead = mapped[2][TREES]
    funnel_lines = [
        "## 1. Which trials have science data",
        "",
        (
            'Trials by split. "Investigational drug": Open Targets lists a drug matched to the '
            'treatment under test (`genetics.py`\'s rule). "Target": that drug has at least one '
            "known target gene. The source columns count trials with at least one value from that "
            "source; every science column, chemistry included, exists only for trials with a target. "
            "Saved when the table was built (`science_funnel.csv`). For the test and late splits "
            "these are counts of trials only."
        ),
        "",
        *funnel_table(funnel),
        "",
    ]
    check_lines = [
        "## 2. Do the sources say what biology says?",
        "",
        (
            "Known answers asked of the real data before any feature was tested. A failure stops "
            "the build. They cover about thirty hand-picked genes and two drugs; they do not touch "
            "the ordinary gene, the disease matching, or the choice of drug."
        ),
        "",
        "| check | result | what was found |",
        "|---|---|---|",
        *[f"| {r.name} | {'passed' if r.passed else 'FAILED'} | {r.detail} |" for r in results],
        "",
        (
            "**Spot check from the raw files.** 25 development trials with a target, drawn at random "
            f"(seed {trace.SEED}), each followed from its target list to the table value (`trace.py`). "
            "It re-reads the raw DepMap, GTEx, HPA, BioGRID and ORCS files with its own loops, "
            "separate from the build. It reads the protein annotations and the chemistry from the slim "
            "files the build wrote, not from the raw Open Targets and ChEMBL downloads. It reuses the "
            "build's gene-id mapping and its disease rules, and starts from the build's own list of "
            "each trial's target genes. It does not cover the step from trial to drug to target, or "
            "the disease-to-lineage and disease-to-tissue rules (those have their own unit tests). "
            "So it checks the join from target gene to value, the choice among several targets and "
            "the empty values, for 25 trials. A missing value on both sides counts as agreement."
        ),
        "",
        "| source | values compared | of which not missing | agreed |",
        "|---|---|---|---|",
        *[f"| {c.source} | {c.values} | {c.present} | {c.agreed} |" for c in spot],
        "",
        *[f"- disagreement: {m}" for c in spot for m in c.mismatches],
        "",
    ]
    feature_lines = [
        "## 3. Feature by feature",
        "",
        (
            "Success rate (the label: at least one arm beat control on its primary outcome) of "
            "trials at the top of a feature against the bottom, development trials with a value for "
            "it. Every difference is the first-named group minus the second-named group, as the "
            '"compared" column says, with its raw sign. "Adjusted" is a Mantel-Haenszel average of '
            "the difference within groups sharing phase, sponsor type, therapeutic area and drug "
            "type (modality); the interval resamples lead drugs. Numbers are percentage points. "
            "The thirds are cut at the one-third and two-thirds quantiles (a tied value stays on "
            "the lower side). For text features each level is compared with the others (levels "
            "with 100 or more trials). Direction was not chosen in advance here; mind that lower "
            "means stronger for the DepMap effect columns and higher means more specific for tau. "
            "The last two columns repeat the comparison on drugs with at most three targets and "
            "with one target, which removes the part of each feature that is the number of targets."
        ),
        "",
        *feature_table(science_rows),
        "",
        "How each feature is built, and how several targets become one value:",
        "",
        *[f"- `{name}`: {DESCRIPTIONS[name]}" for name in biology.honest_columns()],
        "",
        (
            "`depmap_common_essential` flags mostly tubulin binders, metformin (complex I subunits) "
            "and statins (HMGCR). It says cells in culture need the target gene; it does not say the "
            "drug is toxic or that the disease depends on the gene."
        ),
        "",
    ]
    confirm_lines = [
        "## 4. Explore on early trials, confirm on later ones",
        "",
        (
            f"For every contrast the cut points were fixed on trials finishing before "
            f"{EXPLORE_BEFORE}; the same contrast was then measured on trials finishing {EXPLORE_BEFORE} "
            "to 2020. Both periods print the raw difference (first group minus second, as in the "
            'compared column), so the two columns use one sign convention, and "same sign" says '
            "whether the confirm period points the same way as the explore period. The two periods "
            "share drugs, so this guards against sampling noise, not against drugs that are new."
        ),
        "",
        *confirm_table(science_confirmed),
        "",
        (
            f"{held} of {n_confirm} contrasts kept their sign; {plural(len(survivors), 'has', 'have')} "
            "the same sign and a confirm interval that excludes zero."
        ),
        "",
    ]
    candidate_lines = [
        "## 5. Two patterns that stood out after the fact",
        "",
        (
            "These two patterns were noticed on the full development sample by reviewers reading earlier "
            "versions of these results. They were not specified in advance. The numbers below are not "
            "a test of them: the first column is the sample they were noticed on, and the confirm "
            "period is part of it, so the explore and confirm columns show only whether the sign is "
            "the same in both halves. They are hypotheses for the test set or new data, not findings."
        ),
        "",
        (
            "- (a) A target expressed broadly across tissues goes with lower success. Top third minus "
            "bottom third: more tissues and higher vital-organ expression give negative numbers; "
            "lower tau (broad) gives a positive number, because tau is high for one tissue. In "
            f"`{HEADLINE_BROAD}`, {parts['top_third']}."
        ),
        (
            "- (b) A direction worth testing; not distinguishable from zero here: in cancer trials, a "
            "target the trial's cancer lineage depends on more than other lineages, tumour-intrinsic "
            f"targets only (`depmap_lineage_minus_other`, a negative number means more specific; "
            f"{parts['n_lineage']:,} trials with a value)."
        ),
        "",
        *candidate_table(candidates),
        "",
        (
            "Drug type matters for (a): antibodies are rarely in the broadly expressed group and "
            "succeed more, so part of the all-trials figure is drug type. Single-target drugs hold up "
            "better; adding target class as well widens the interval. Single-target drugs also remove "
            "the complex-expansion problem, in which a maximum or minimum over many genes tracks the "
            "number of targets. Cancer trials whose drug has a single tumour-intrinsic target are too "
            "few for the lineage feature to be measured there."
        ),
        "",
    ]
    control_lines = [
        "## 6. A positive control, and what each test can see",
        "",
        (
            "Human genetic support (`supported_at_start`, from `genetics.parquet`) is a signal "
            "established in published studies: drug targets with human genetic evidence have "
            "tended to succeed more often. In this data its single-target interval is wide "
            f"({cell(control_single)}). It is blanked where a trial has no target, so it has the "
            "same missing values as the science columns, and then sent through the same code as the "
            "science features."
        ),
        "",
        "| contrast (yes minus no) | all trials with a value | at most 3 targets | one target | explore, before 2015 | confirm, 2015 to 2020 | same sign |",
        "|---|---|---|---|---|---|---|",
        (
            f"| genetic support | {cell(control_feature)} | {cell(control_rows[0][1]['few'])} | "
            f"{cell(control_single)} | "
            f"{'too few' if explore is None else points(explore.difference)} | {cell(confirm)} | "
            f"{control_agree} |"
        ),
        "",
        (
            f"As a model increment over registration, on the mapped validation trials (`{CONTROL_SPEC}`): "
            f"{plus(control_model[0])} AUROC ({plus(control_model[1])} to {plus(control_model[2])})."
        ),
        "",
        (
            f"What follows. The per-feature intervals above are about ±{half_width:.1f} points wide in "
            f"the middle, so a source could show a difference of roughly that size or more. An effect "
            f"the size of genetic support ({points(control_feature.difference)} points) is at the edge "
            "of what the per-feature test can see, and the model test cannot reliably see it. The upper "
            f"end of the interval for all sources together ({plus(parts['m_diff'][2])} AUROC) therefore "
            "bounds effects that change the overall ranking, not the contrasts a single feature might "
            "carry. Neither test shows that a source carries no information."
        ),
        "",
    ]
    model_lines = [
        "## 7. Models",
        "",
        (
            "Scored as in `results/dev_models.md`: three time-ordered rounds (train before 2015, "
            "score 2015-2016; before 2017, 2017-2018; before 2019, 2019-2020), mean of the AUROCs "
            "within each round, bootstrap intervals (which resample trials, not drugs, so they are "
            "somewhat too narrow), and every difference taken against "
            '"logistic: registration" on the same resamples. No model sees a missing-value flag '
            "for a science column: each science column is centred on its training mean over trials "
            "that have a value and missing is zero, so a trial with no target adds nothing to the "
            "score (`models/biology.py`). The models that read science columns only are fitted on "
            "trials with at least one science value; for any other trial every column is zero and "
            "they predict one constant, the model's own prediction at the centring point. Their "
            "scores are therefore given on mapped trials only."
        ),
        "",
        f"### On the {mapped[0]:,} validation trials with a mapped target",
        "",
        *model_table(models["specs"], mapped, "mapped"),
        "",
        f"### On all {everyone[0]:,} validation trials",
        "",
        (
            "For the science-only models a score on all trials would compare a constant with real "
            "predictions, so it is not given."
        ),
        "",
        *model_table(models["specs"], everyone, "all"),
        "",
        (
            f"The ladder rung with science (`{LADDER_SCIENCE}`) against the same rung without science "
            f"(`{LADDER_NAME}`), on mapped trials: AUROC {plus(mapped[2][LADDER_SCIENCE].value)} "
            f"({plus(mapped[2][LADDER_SCIENCE].interval.low)} to {plus(mapped[2][LADDER_SCIENCE].interval.high)}). "
            "The gain of that rung over plain registration comes from the endpoint, history, "
            "genetics and word tables it already includes."
        ),
        "",
        "### What the science-only models can read",
        "",
        (
            "The boosted trees can see which sources are empty for a trial. No chemistry value, for "
            "example, usually means the lead drug is a biologic, so the science-only trees partly "
            "encode drug type. Their lead over the logistic model on mapped trials is "
            f"{plus(trees_lead.value)} AUROC ({plus(trees_lead.interval.low)} to "
            f"{plus(trees_lead.interval.high)}). The number of targets does not explain it. The "
            f'science columns say "several targets rather than one" almost perfectly (AUROC '
            f"{check['columns_say_several']:.3f}, mapped development trials, five folds that keep each "
            f"drug together), yet the number of targets alone ranks success at only "
            f"{check['count_alone']}, and the trees lead the logistic among one-target trials as "
            f"well as among several-target trials (table)."
        ),
        "",
        "| validation trials | trials | trees AUROC | logistic AUROC |",
        "|---|---|---|---|",
        *[
            f"| {name} | {n:,} | {cells[TREES]} | {cells[LOGISTIC_ALL]} |"
            for name, (n, cells) in check["within"].items()
        ],
        "",
    ]
    return "\n".join(
        [*read_first(parts), *funnel_lines, *check_lines, *feature_lines, *confirm_lines,
         *candidate_lines, *control_lines, *model_lines]
    )  # fmt: skip


def funnel_table(funnel: pl.DataFrame) -> list[str]:
    header = ["split", *funnel.columns[1:]]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in funnel.iter_rows():
        lines.append(
            "| " + " | ".join(f"{v:,}" if isinstance(v, int) else str(v) for v in row) + " |"
        )
    return lines


def main() -> None:
    specs = wanted_specs()
    frame = load_frame(specs)
    funnel = pl.read_csv(table.FUNNEL_PATH)
    drugs, targets, conditions = tg.build_inputs()
    results = checks.run_all(drugs, targets)
    print("known-biology checks:", "all passed" if all(r.passed for r in results) else "FAILED",
          file=sys.stderr)  # fmt: skip
    spot = trace.run(pl.read_parquet(table.OUT_PATH), drugs, targets, conditions)
    print("spot check:\n" + trace.render(spot), file=sys.stderr)
    names = [*biology.honest_columns(), CONTROL]
    feature_rows = per_feature(frame, names)
    confirmed = confirm_rows(frame, names)
    candidates = candidate_rows(frame)
    models = evaluate_models(specs, frame)
    REPORT_PATH.write_text(
        write_report(frame, funnel, results, spot, feature_rows, confirmed, candidates, models)
    )
    print(f"wrote {REPORT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
