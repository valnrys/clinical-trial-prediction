"""The biology-only score: how likely is a trial to succeed, judged from target and drug biology alone?

Pre-declared recipe, ported unchanged from the scratch analysis whose plan was written before any
outcome was read (`TEST_PLAN_2.md`, item 8). Nothing here is tuned; do not tune it.

Two scores, both within phase:

1. A count of six risk flags (0 to 6, unweighted; a missing value switches a flag off). Trials are
   grouped as 0, 1, or 2 or more flags. Statistic: success per 100 trials with 2+ flags minus
   with 0 flags, Mantel-Haenszel pooled over phase (phase is the only stratum).
2. A logistic model (L2, C = 1.0, nothing tuned) on biology columns only, fitted on training
   trials. Inputs are the six flags, 18 numeric biology columns (standardised on the training
   set, missing filled with the training mean, no missing-value indicators), 2 yes/no columns and
   the one-hot target class, target location and drug action. No phase, sponsor, disease area,
   design, modality, precedent, earlier outcome or genetic support. Trials are grouped in thirds
   cut at the training rows' own fitted-score tertiles. Statistic: success per 100 in the top
   third minus the bottom third, pooled over phase.

Quantiles. The three flag thresholds are Polars `quantile` (nearest rank) of the pre-2015 trials
with a value; the logistic tertile cuts are `numpy.quantile` (linear) of the training rows' fitted
scores. The two methods differ and both are part of the declared recipe, so neither is changed.
Many biology columns are discrete (for example tissue counts), so ties at a cut put the same value
in the same group and the groups are only roughly thirds.

Intervals are 95% percentile intervals from a bootstrap that resamples whole lead drugs (a trial
with no lead drug is its own cluster), 2,000 resamples.

Outcome safety. `feature_frame` and `flag_frame` never see an outcome. `fit_logistic` takes the
training outcomes as an argument and uses them only to fit; the score of any frame is a function
of that frame's biology columns alone. The statistics functions take plain arrays, so they can
score any set the caller supplies; nothing in this module reads a label from disk.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from trial_benchmark import adjusted as adj
from trial_benchmark.opentargets import INTERIM

# ---------------------------------------------------------------- the declared thresholds

EXPLORE_BEFORE = 2015  # "before 2015" = primary completion year below this
# Copied from the pre-declared plan. Each was fixed on development trials that finished before
# 2015 and have a mapped investigational target under today's links (Open Targets 26.09), before
# any outcome was read. `recompute_thresholds` rebuilds them; a test checks they still match.
N_TISSUES_CUT = 49.0  # broad: expressed in more than 49 of 51 HPA tissues (top third, 2/3 quantile)
VITAL_CUT = (
    5.786964829327251  # vital-organ max GTEx expression above this (top third, 2/3 quantile)
)
QED_CUT = 0.51  # low drug-likeness: QED at or below this (bottom third, 1/3 quantile)

FLAGS = (
    "f_broad",  # target present in nearly every tissue
    "f_vital",  # high expression in heart, liver, kidney, brain or blood
    "f_essential",  # gene needed by cells (DepMap common-essential)
    "f_intracellular",  # target sits inside the cell
    "f_activator",  # the drug turns its target up
    "f_lowqed",  # low drug-likeness (small molecules only; biologics have none, so flag off)
)

NUMERIC = (
    "depmap_mean_effect",
    "depmap_fraction_dependent",
    "depmap_lineage_effect",
    "depmap_lineage_minus_other",
    "expr_tau",
    "expr_n_tissues",
    "expr_vital_max",
    "expr_disease_tissue",
    "net_viability_hit_fraction",
    "net_other_hit_fraction",
    "net_partners_per_paper",
    "constraint",
    "mouse_knockout",
    "chem_mol_weight",
    "chem_logp",
    "chem_polar_surface_area",
    "chem_qed",
    "chem_rule_of_five_violations",
)
BOOLEAN = ("depmap_common_essential", "expr_disease_tissue_top3")
CATEGORICAL = ("target_class", "target_location", "chem_action_type")
# The only columns the logistic model may read. The six flags are functions of these.
INPUT_COLUMNS = (*NUMERIC, *BOOLEAN, *CATEGORICAL)
# Extra columns `feature_frame` returns for convenience; the model never reads them.
EXTRA_COLUMNS = ("n_target_genes", "lead_drug")

SEED = 20261112
N_RESAMPLES = 2000
MIN_GROUP = 15  # a contrast needs at least this many trials on each side
GROUPS = ("0", "1", "2+")
THIRDS = ("low", "mid", "high")


# ---------------------------------------------------------------- the benchmark's biology inputs


@cache
def target_table(version: str) -> pl.DataFrame:
    """`science.links.investigational_targets(version)`, read once. Every split; no label."""
    from trial_benchmark.science import links

    return links.investigational_targets(version)


def feature_frame(version: str = "today") -> pl.DataFrame:
    """Biology inputs for every benchmark trial with a mapped investigational target.

    One row per trial, all splits, no outcome. Columns: `nct_id`, the model's `INPUT_COLUMNS`,
    `n_target_genes` and `lead_drug` (from the genetics table; used only as the bootstrap
    cluster). `version` is a link version of `science.links`: "today", "2020" or "2020_names"
    (the last has no chemistry columns). Built from the repo's science tables with the trial's
    targets and drug actions taken from `links.investigational_targets(version)`.
    """
    from trial_benchmark import whathow

    targets = target_table(version)
    table = whathow.biology_columns(version, targets)
    leads = pl.read_parquet(INTERIM / "genetics.parquet", columns=["nct_id", "lead_drug"])
    return (
        table.filter(pl.col("nct_id").is_in(targets["nct_id"].implode()))
        .join(leads, on="nct_id", how="left", validate="1:1")
        .select("nct_id", *INPUT_COLUMNS, *EXTRA_COLUMNS)
        .sort("nct_id")
    )


# ---------------------------------------------------------------- the six flags


def _flag_expressions(
    n_tissues: float = N_TISSUES_CUT, vital: float = VITAL_CUT, qed: float = QED_CUT
) -> dict[str, pl.Expr]:
    return {
        "f_broad": pl.col("expr_n_tissues") > n_tissues,
        "f_vital": pl.col("expr_vital_max") > vital,
        "f_essential": pl.col("depmap_common_essential") == True,
        "f_intracellular": pl.col("target_location") == "intracellular",
        "f_activator": pl.col("chem_action_type") == "activator",
        "f_lowqed": pl.col("chem_qed") <= qed,
    }


def flag_frame(features: pl.DataFrame) -> pl.DataFrame:
    """`nct_id`, the six boolean flags (`FLAGS`, in order) and `n_flags` (0 to 6).

    A missing value never raises a flag. Needs no outcome.
    """
    flags = [e.fill_null(False).alias(k) for k, e in _flag_expressions().items()]
    out = features.select("nct_id", *flags)
    return out.with_columns(pl.sum_horizontal(*FLAGS).cast(pl.Int64).alias("n_flags"))


def flag_groups(n_flags: np.ndarray) -> np.ndarray:
    """ "0", "1" or "2+" for each flag count."""
    n = np.asarray(n_flags)
    return np.where(n == 0, "0", np.where(n == 1, "1", "2+"))


# ---------------------------------------------------------------- recomputing the thresholds


def pre_2015_ids() -> set[str]:
    """Development trials finishing before 2015 (dates only; no outcome is read)."""
    dev = pl.read_parquet(INTERIM / "dev.parquet", columns=["nct_id", "primary_completion_date"])
    early = dev.filter(pl.col("primary_completion_date").dt.year() < EXPLORE_BEFORE)
    return set(early["nct_id"].to_list())


def recompute_thresholds(features: pl.DataFrame, early_ids: Iterable[str]) -> dict[str, float]:
    """The three cut points from the pre-2015 trials of `features`.

    `features` is a `feature_frame` (trials with a mapped target); `early_ids` are the ids of the
    trials that finished before 2015. Broad expression and vital-organ expression: the top third
    (2/3 quantile of trials with a value). Low drug-likeness: the bottom third (1/3 quantile).
    """
    early = features.filter(pl.col("nct_id").is_in(list(early_ids)))

    def cut(column: str, q: float) -> float:
        return float(early[column].drop_nulls().quantile(q))

    return {
        "N_TISSUES_CUT": cut("expr_n_tissues", 2 / 3),
        "VITAL_CUT": cut("expr_vital_max", 2 / 3),
        "QED_CUT": cut("chem_qed", 1 / 3),
    }


# ---------------------------------------------------------------- the logistic model


def _design(
    frame: pl.DataFrame,
    stats: dict[str, tuple[float, float]],
    levels: dict[str, list[str]],
) -> np.ndarray:
    """Flags, standardised numerics (missing = training mean), yes/no columns, one-hot categories."""
    parts = [frame.select(list(_flag_expressions().values())).fill_null(False).to_numpy()]
    parts[0] = parts[0].astype(float)
    for column in NUMERIC:
        mean, std = stats[column]
        x = frame[column].cast(pl.Float64).to_numpy()
        x = np.where(np.isnan(x), mean, x)
        parts.append(((x - mean) / (std or 1.0))[:, None])
    for column in BOOLEAN:
        parts.append(frame[column].cast(pl.Float64).fill_null(0.0).to_numpy()[:, None])
    for column in CATEGORICAL:
        if not levels[column]:
            continue
        values = frame[column].to_numpy()
        parts.append(np.column_stack([(values == lv).astype(float) for lv in levels[column]]))
    return np.hstack(parts)


@dataclass
class FittedScore:
    """A fitted biology-only logistic model. `score` needs no outcome."""

    model: LogisticRegression
    stats: dict[str, tuple[float, float]]
    levels: dict[str, list[str]]
    tertile_cuts: tuple[float, float]  # from the training rows' own fitted scores
    input_columns: tuple[str, ...] = INPUT_COLUMNS

    def score(self, frame: pl.DataFrame) -> np.ndarray:
        """Predicted success chance for each row, from its biology columns alone."""
        x = _design(frame.select(list(self.input_columns)), self.stats, self.levels)
        return self.model.predict_proba(x)[:, 1]

    def flag_coefficients(self) -> dict[str, float]:
        return dict(zip(FLAGS, (float(c) for c in self.model.coef_[0][: len(FLAGS)]), strict=True))


def fit_logistic(train: pl.DataFrame, outcomes) -> FittedScore:
    """Fit on `train` (a `feature_frame`-like table) with the aligned yes/no `outcomes`.

    L2 penalty, C = 1.0, nothing tuned. Only `INPUT_COLUMNS` are read from `train`; a column
    that is empty in the training rows (the chemistry columns under "2020_names") contributes 0.
    """
    y = np.asarray(outcomes).astype(int)
    if len(y) != train.height:
        raise ValueError("outcomes must have one entry per training row")
    train = train.select(list(INPUT_COLUMNS))
    stats: dict[str, tuple[float, float]] = {}
    for column in NUMERIC:
        x = train[column].cast(pl.Float64)
        mean, std = x.mean(), x.std()
        stats[column] = (0.0, 1.0) if mean is None else (float(mean), float(std or 1.0))
    levels = {c: sorted(train[c].drop_nulls().unique().to_list()) for c in CATEGORICAL}
    model = LogisticRegression(C=1.0, max_iter=5000).fit(_design(train, stats, levels), y)
    fitted = FittedScore(model, stats, levels, (np.nan, np.nan))
    q1, q2 = np.quantile(fitted.score(train), [1 / 3, 2 / 3])
    fitted.tertile_cuts = (float(q1), float(q2))
    return fitted


def thirds(score: np.ndarray, cuts: tuple[float, float]) -> np.ndarray:
    """ "low" (score <= first cut), "mid" (<= second cut) or "high" for each score."""
    s = np.asarray(score, dtype=float)
    return np.where(s <= cuts[0], "low", np.where(s <= cuts[1], "mid", "high"))


# ---------------------------------------------------------------- unseen targets


def unseen_targets(
    scored_ids: Iterable[str], training_ids: Iterable[str], version: str
) -> set[str]:
    """Scored trials none of whose target genes appear in any training trial.

    A trial with no mapped target is not counted (it has nothing to be unseen).
    """
    targets = target_table(version).select("nct_id", "target_id").unique()
    training = set(training_ids)
    seen = set(targets.filter(pl.col("nct_id").is_in(list(training)))["target_id"].to_list())
    scored = set(scored_ids)
    mine = targets.filter(pl.col("nct_id").is_in(list(scored)))
    any_seen = mine.group_by("nct_id").agg(pl.col("target_id").is_in(list(seen)).any().alias("s"))
    return set(any_seen.filter(~pl.col("s"))["nct_id"].to_list())


# ---------------------------------------------------------------- the statistics


@dataclass(frozen=True)
class Contrast:
    """A difference in successes per 100 trials with a 95% interval; n of each side."""

    estimate: float
    low: float
    high: float
    n_exposed: int
    n_reference: int

    def text(self) -> str:
        if np.isnan(self.estimate):
            return "n/a"
        e, lo, hi = (round(float(v), 1) + 0.0 for v in (self.estimate, self.low, self.high))
        return f"{e:+.1f} ({lo:+.1f} to {hi:+.1f})"


def _missing(label: object) -> bool:
    """None or a float NaN: a trial with no known lead drug is its own cluster."""
    return label is None or (isinstance(label, float | np.floating) and bool(np.isnan(label)))


def mh_difference(
    outcomes,
    exposed: np.ndarray,
    keep: np.ndarray,
    phase,
    lead_drug,
    extra_strata: dict[str, np.ndarray] | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> Contrast:
    """Mantel-Haenszel exposed minus reference over the rows in `keep`, per 100, pooled over phase
    (and over `extra_strata` if given), with a lead-drug cluster-bootstrap interval.

    Plain arrays in, so any exposure can be tested the same way (the report uses it for single
    features and for independence checks)."""
    columns = {
        "y": np.asarray(outcomes).astype(bool).tolist(),
        "e": np.asarray(exposed).astype(bool).tolist(),
        "phase": np.asarray(phase).astype(str).tolist(),
        "lead_drug": [None if _missing(v) else str(v) for v in lead_drug],
        **{k: np.asarray(v).astype(str).tolist() for k, v in (extra_strata or {}).items()},
    }
    schema = {"lead_drug": pl.String}
    frame = pl.DataFrame(columns, schema_overrides=schema)
    frame = frame.filter(pl.Series(np.asarray(keep).astype(bool)))
    n1 = int(frame["e"].sum())
    n0 = frame.height - n1
    if n1 < MIN_GROUP or n0 < MIN_GROUP:
        return Contrast(np.nan, np.nan, np.nan, n1, n0)
    strata = ["phase", *(extra_strata or {})]
    d, lo, hi = adj.adjusted(
        frame,
        "e",
        "y",
        strata,
        "lead_drug",
        measure="difference",
        n_resamples=n_resamples,
        seed=seed,
    )
    return Contrast(100 * d, 100 * lo, 100 * hi, n1, n0)


def flag_contrast(
    outcomes, n_flags, phase, lead_drug, extra_strata: dict[str, np.ndarray] | None = None
) -> Contrast:
    """2 or more flags minus 0 flags, per 100 trials, pooled over phase, with an interval.

    Trials with exactly 1 flag are left out. Expected sign: negative. `extra_strata` (optional)
    adds more variables to pool over, for adjusted versions.
    """
    group = flag_groups(np.asarray(n_flags))
    return mh_difference(
        outcomes, group == "2+", group != "1", phase, lead_drug, extra_strata=extra_strata
    )


def thirds_contrast(
    outcomes,
    score,
    cuts: tuple[float, float],
    phase,
    lead_drug,
    extra_strata: dict[str, np.ndarray] | None = None,
) -> Contrast:
    """Top third minus bottom third of the logistic score, per 100 trials, pooled over phase.

    `cuts` are the training tertiles (`FittedScore.tertile_cuts`). Expected sign: positive.
    """
    group = thirds(np.asarray(score), cuts)
    return mh_difference(
        outcomes, group == "high", group != "mid", phase, lead_drug, extra_strata=extra_strata
    )


def rate(outcomes, lead_drug, n_resamples: int = N_RESAMPLES, seed: int = SEED):
    """(successes per 100, low, high, n) with a lead-drug cluster-bootstrap interval."""
    y = np.asarray(outcomes).astype(float)
    if len(y) == 0:
        return (np.nan, np.nan, np.nan, 0)
    names = [f"__row{i}" if _missing(v) else str(v) for i, v in enumerate(lead_drug)]
    cluster = np.unique(np.array(names), return_inverse=True)[1]
    k = cluster.max() + 1
    successes = np.bincount(cluster, weights=y, minlength=k)
    counts = np.bincount(cluster, minlength=k).astype(float)
    weights = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n_resamples)
    draws = (weights @ successes) / (weights @ counts)
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return (100 * float(y.mean()), 100 * float(low), 100 * float(high), len(y))


def by_group(outcomes, groups, phase, lead_drug, n_resamples: int = N_RESAMPLES) -> pl.DataFrame:
    """Successes per 100 with a 95% interval, and n, for each group within each phase.

    Columns: phase, group, n, per_100, low, high. Groups are those present in `groups`
    (ordered "0", "1", "2+" / "low", "mid", "high" if those are the values).
    """
    y = np.asarray(outcomes).astype(bool)
    g = np.asarray(groups).astype(str)
    p = np.asarray(phase).astype(str)
    lead = np.asarray(lead_drug, dtype=object)
    order = {v: i for i, v in enumerate((*GROUPS, *THIRDS))}
    rows = []
    for ph in sorted(set(p)):
        for grp in sorted(set(g), key=lambda v: (order.get(v, 99), v)):
            m = (p == ph) & (g == grp)
            if not m.any():
                continue
            r = rate(y[m], lead[m], n_resamples=n_resamples)
            rows.append(
                {"phase": ph, "group": grp, "n": r[3], "per_100": r[0], "low": r[1], "high": r[2]}
            )
    return pl.DataFrame(
        rows,
        schema={
            "phase": pl.String,
            "group": pl.String,
            "n": pl.Int64,
            "per_100": pl.Float64,
            "low": pl.Float64,
            "high": pl.Float64,
        },
    )
