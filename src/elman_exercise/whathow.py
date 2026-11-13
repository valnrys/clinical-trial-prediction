"""WHAT versus HOW: what is being attempted, against how the trial is run.

    uv run python -m elman_exercise.whathow     # builds the tables below (no outcome is read)

A trial's predictors split into two blocks.

- HOW: the registration columns less therapeutic area: phase, sponsor class, allocation, masking,
  intervention model, arm count, modality, vaccine, healthy volunteers, data monitoring committee.
  (`baselines.registration_features_without("therapeutic_area")`. No column from the results
  section is in it.)
- WHAT: therapeutic area; a success rate for the trial's target, for its disease and for the pair,
  each pulled toward 0.5 when few earlier trials back it; how many trials back each rate; the
  STRICT precedent flag (`precedent_exp_<version>`: experimental-arm drugs only, no comparator
  fallback, so a trial with no investigational target is never flagged); genetic support when the trial started; generic target biology (dependency,
  expression, network, constraint, chemistry); and yes/no flags for "has a mapped target" and
  "has a mapped disease".

The rule that matters. A group rate for a scored trial T counts only trials U that are

- in the labelled pool (the training trials of the split; never a trial of the scored set),
- not T itself, and
- first posted with results strictly before T's earliest possible start day.

Whether U succeeded is only knowable once its results are posted, so a forecaster standing at T's
start could not have used any later U. The rule holds in the time-ordered rounds and in the grouped
folds alike, and for the training rows' own features as well as the scored ones. Optionally
(`exclude="lead"`) every U with the same lead drug as T is also dropped, so no rate can be the drug
itself. The rate is `(successes + K * 0.5) / (n + K)` with K = 10 and 0.5 fixed in advance: no label
is read to choose either.

Link versions ("today", "2020", "2020_names", `science.links`) change which drug-to-target links
give a trial its targets, and so its target keys, mapped flag, biology columns, genetic support and
precedent flag. All versions share today's trial-to-disease mapping and today's registry record,
and `lead_drug` (the drug used for clusters and the own-drug exclusion) is today's match in every
version. In the "2020_names" version the chemistry columns are left empty: they are keyed by
today's drug ids, which a 20.06 name match does not give.

Tables written to `data/interim/` (build them once with the command above):

- `whathow_<version>.parquet`: one row per trial, every split, columns named `wh_*`. Holds the
  start date (earliest possible), the results-first-posted date (latest possible; development
  trials only, null for others), the flags, genetic support and biology columns.
- `whathow_keys_<version>.parquet`: (nct_id, kind, key) with kind target, disease or pair.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import polars as pl

from elman_exercise import baselines as bl
from elman_exercise import dataset, genetics
from elman_exercise.dating import AFTER_START, PROTOCOL, TIMELESS
from elman_exercise.history import date_range
from elman_exercise.opentargets import INTERIM, OT_DIR
from elman_exercise.science import annotations, chemistry, depmap, expression, links, networks
from elman_exercise.science import table as science_table
from elman_exercise.science import targets as tg

K = 10.0  # pseudo-count: a rate behaves as if it began with K trials at the prior
BASE = 0.5  # the prior success rate, fixed in advance
KINDS = ("target", "disease", "pair")
LEAD_DRUG = "lead_drug"
PREFIX = "wh_"
HOW = bl.registration_features_without("therapeutic_area")

SCIENCE_NUMERIC = (
    "depmap_common_essential",
    "depmap_mean_effect",
    "depmap_fraction_dependent",
    "depmap_lineage_effect",
    "depmap_lineage_minus_other",
    "expr_tau",
    "expr_n_tissues",
    "expr_vital_max",
    "expr_disease_tissue",
    "expr_disease_tissue_top3",
    "net_viability_hit_fraction",
    "net_other_hit_fraction",
    "net_partners_per_paper",
    "constraint",
    "mouse_knockout",
    "n_target_genes",
    "chem_mol_weight",
    "chem_logp",
    "chem_polar_surface_area",
    "chem_qed",
    "chem_rule_of_five_violations",
)
SCIENCE_CATEGORICAL = ("target_class", "target_location", "chem_action_type")
FLAG_COLUMNS = ("precedent", "supported_at_start", "has_target", "has_disease")
DATE_COLUMNS = ("start_first", "posted_last")
TABLE_COLUMNS = (*DATE_COLUMNS, *FLAG_COLUMNS, *SCIENCE_NUMERIC, *SCIENCE_CATEGORICAL)

# How each WHAT column is dated (see `dating`). The links are a release from after the trials
# started, so everything built from them is "after the start". The group rates are only half
# dated: the outcomes they count are dated by the rule above, but the keys that group trials
# (the target, from the link version, and the disease, from today's mapping) are not, so they are
# "after the start" too, not DATED. The date columns only serve the rule.
FEATURES: dict[str, str] = {
    "therapeutic_area": PROTOCOL,
    **dict.fromkeys((f"rate_{k}" for k in KINDS), AFTER_START),
    **dict.fromkeys((f"n_{k}" for k in KINDS), AFTER_START),
    "precedent": AFTER_START,
    "supported_at_start": AFTER_START,
    "has_target": AFTER_START,
    "has_disease": AFTER_START,
    **dict.fromkeys((*SCIENCE_NUMERIC, *SCIENCE_CATEGORICAL), AFTER_START),
    "start_first": TIMELESS,
    "posted_last": TIMELESS,
}

DATES_PATH = INTERIM / "whathow_dates.parquet"
CONDITIONS_PATH = INTERIM / "whathow_conditions.parquet"


def table_name(version: str) -> str:
    return f"whathow_{version}"


def keys_name(version: str) -> str:
    return f"whathow_keys_{version}"


def column(name: str) -> str:
    """The name of a WHAT table column inside a frame."""
    return PREFIX + name


# ---------------------------------------------------------------- group rates


def group_rates(
    scored: pl.DataFrame,
    pool: pl.DataFrame,
    keys: pl.DataFrame,
    exclude: str | None = None,
    k: float = K,
    base: float = BASE,
) -> pl.DataFrame:
    """For each scored trial: `rate_<kind>`, `n_<kind>`, `logit_<kind>` and `logn_<kind>`.

    `scored` has nct_id, `wh_start_first` and `lead_drug`. `pool` is the labelled trials a rate may
    count: nct_id, `wh_posted_last`, `lead_drug` and `y` (success, 0 or 1). `keys` is
    (nct_id, kind, key) for every trial of either frame.

    A pool trial U counts toward scored trial T only if U has T's key of that kind, U is not T,
    and U's results were first posted strictly before T's start (a trial with no date on either
    side counts for nobody). With `exclude="lead"`, U must also have a different lead drug (an
    unknown lead drug on either side excludes nothing). The rate is
    `(successes + k * base) / (n + k)`; with nothing to count it is `base` and n is 0.
    """
    if exclude not in (None, "lead"):
        raise ValueError(f"exclude must be None or 'lead', not {exclude!r}")
    target = scored.select(
        "nct_id", pl.col(column("start_first")).alias("start"), pl.col(LEAD_DRUG).alias("lead")
    )
    source = pool.select(
        pl.col("nct_id").alias("u"),
        pl.col(column("posted_last")).alias("u_posted"),
        pl.col(LEAD_DRUG).alias("u_lead"),
        pl.col("y").cast(pl.Float64),
    )
    joined = (
        keys.join(target, on="nct_id")
        .join(keys.rename({"nct_id": "u"}).join(source, on="u"), on=["kind", "key"])
        .filter(pl.col("u") != pl.col("nct_id"))
        .filter(pl.col("u_posted") < pl.col("start"))
    )
    if exclude == "lead":
        joined = joined.filter(
            pl.col("lead").is_null()
            | pl.col("u_lead").is_null()
            | (pl.col("lead") != pl.col("u_lead"))
        )
    counts = (
        joined.select("nct_id", "kind", "u", "y")
        .unique()
        .group_by("nct_id", "kind")
        .agg(pl.col("y").sum().alias("successes"), pl.len().alias("n"))
    )
    out = scored.select("nct_id")
    for kind in KINDS:
        part = counts.filter(pl.col("kind") == kind).select(
            "nct_id",
            ((pl.col("successes") + k * base) / (pl.col("n") + k)).alias(f"rate_{kind}"),
            pl.col("n").cast(pl.Float64).alias(f"n_{kind}"),
        )
        out = out.join(part, on="nct_id", how="left").with_columns(
            pl.col(f"rate_{kind}").fill_null(base), pl.col(f"n_{kind}").fill_null(0.0)
        )
    return out.with_columns(
        *[
            (pl.col(f"rate_{x}") / (1 - pl.col(f"rate_{x}"))).log().alias(f"logit_{x}")
            for x in KINDS
        ],
        *[pl.col(f"n_{x}").log1p().alias(f"logn_{x}") for x in KINDS],
    )


def pool_of(train: pl.DataFrame) -> pl.DataFrame:
    """The labelled trials of `train` as a rate pool (label: `any_arm`, the headline label)."""
    outcomes = bl.training_outcomes(train)
    return train.select("nct_id", column("posted_last"), LEAD_DRUG).with_columns(
        pl.Series("y", outcomes.astype(float))
    )


def what_frame(
    scored: pl.DataFrame,
    pool: pl.DataFrame,
    keys: pl.DataFrame,
    exclude: str | None = None,
) -> pl.DataFrame:
    """`scored` with the group-rate columns added. Works for any scored set, a test set included.

    The scored frame is never read for a label. The pool is whatever labelled trials the caller
    allows (for the final test, the development trials). The scored frame must carry `lead_drug`:
    a missing column would turn the own-drug exclusion into a no-op, so it is refused. A trial
    whose lead drug is null (no investigational drug matched) is kept; on either side of a pair a
    null lead drug excludes nobody.
    """
    if LEAD_DRUG not in scored.columns:
        raise ValueError(
            f"the scored frame has no `{LEAD_DRUG}` column; join the `genetics` table "
            "(without it the own-drug exclusion and the drug clustering would silently do nothing)"
        )
    rates = group_rates(scored, pool, keys, exclude)
    return scored.join(rates, on="nct_id", how="left", validate="1:1")


def column_groups() -> tuple[list[str], list[str], list[str]]:
    """(text, yes/no, number) columns of the WHAT block as named in a frame."""
    text = ["therapeutic_area", *(column(c) for c in SCIENCE_CATEGORICAL)]
    yes_no = [column(c) for c in FLAG_COLUMNS]
    numbers = [column(c) for c in SCIENCE_NUMERIC]
    numbers += [f"logit_{x}" for x in KINDS] + [f"logn_{x}" for x in KINDS]
    return text, yes_no, numbers


def what_columns() -> set[str]:
    text, yes_no, numbers = column_groups()
    return {*text, *yes_no, *numbers}


def how_columns() -> set[str]:
    return set(HOW.columns)


# ---------------------------------------------------------------- building the tables


def start_and_posted(path=DATES_PATH, dev_ids: set[str] | None = None) -> pl.DataFrame:
    """(nct_id, start_first, posted_last) for every downloaded trial.

    `start_first` is the earliest day the registry start date could mean; `posted_last` the latest
    day the results-first-posted date could mean, kept for development trials only (the one place a
    posting date is used is deciding whether a labelled trial's result was known). Null where the
    registry has no date. Cached in `data/interim/whathow_dates.parquet`.
    """
    if path.exists():
        return pl.read_parquet(path)
    from elman_exercise.download import iter_studies, nct_id

    if dev_ids is None:
        dev_ids = set(pl.read_parquet(INTERIM / "dev.parquet", columns=["nct_id"])["nct_id"])
    get = dataset._get
    rows = []
    for study in iter_studies():
        ident = nct_id(study).upper()
        status = ("protocolSection", "statusModule")
        first, _ = date_range(get(study, *status, "startDateStruct", "date"))
        posted = None
        if ident in dev_ids:
            _, posted = date_range(get(study, *status, "resultsFirstPostDateStruct", "date"))
        rows.append((ident, first, posted))
    frame = pl.DataFrame(
        rows,
        schema={"nct_id": pl.String, "start_first": pl.Date, "posted_last": pl.Date},
        orient="row",
    )
    frame.write_parquet(path)
    return frame


def trial_disease_ids(ids: list[str]) -> pl.DataFrame:
    """(nct_id, diseaseId): Open Targets' disease mapping, today's."""
    reports = pl.read_parquet(OT_DIR / "clinical_report.parquet", columns=["id", "diseases"])
    return genetics.trial_diseases(reports, ids)


def link_keys(targets: pl.DataFrame, diseases: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, kind, key): target, disease and "target|disease" keys of each trial."""
    target = targets.select("nct_id", pl.col("target_id").alias("key")).unique()
    disease = diseases.select("nct_id", pl.col("diseaseId").alias("key")).unique()
    pair = target.join(disease, on="nct_id", suffix="_d").select(
        "nct_id", (pl.col("key") + "|" + pl.col("key_d")).alias("key")
    )
    return pl.concat(
        [
            target.with_columns(pl.lit("target").alias("kind")),
            disease.with_columns(pl.lit("disease").alias("kind")),
            pair.unique().with_columns(pl.lit("pair").alias("kind")),
        ]
    ).select("nct_id", "kind", "key")


def genetic_support(
    targets: pl.DataFrame, diseases: pl.DataFrame, trials: pl.DataFrame
) -> pl.DataFrame:
    """(nct_id, supported_at_start): did any (target, disease) pair of the trial have genetic
    evidence dated before its start year? Same rule as `genetics.pair_scores`, with the
    investigational drugs' targets of the chosen link version. Null where there is no pair or no
    start year."""
    associations = pl.read_parquet(OT_DIR / "genetic_association.parquet")
    pairs = (
        targets.select("nct_id", pl.col("target_id").alias("targetId"))
        .unique()
        .join(diseases, on="nct_id")
    )
    scored = genetics.pair_scores(pairs, trials.select("nct_id", "start_year"), associations)
    return (
        scored.group_by("nct_id")
        .agg(pl.col("at_start").max().alias("score"))
        .select("nct_id", (pl.col("score") > 0).alias("supported_at_start"))
    )


def biology_columns(version: str, targets: pl.DataFrame) -> pl.DataFrame:
    """The generic target biology columns (`science.table` pieces) for one link version.

    Same pieces as `data/interim/science.parquet`, with the trial's targets and the drug's action
    words taken from `links.investigational_targets(version)`. The lead drug's chemistry is read
    from today's tables by its (today's) id, which the "2020_names" version does not have (its
    drugs carry 20.06 ids), so there the chemistry columns are left empty rather than mixing two
    id systems.
    """
    trials = tg.load_trials()
    conditions = read_conditions(trials["nct_id"].to_list())
    hgnc = tg.read_hgnc()
    mapping = tg.entrez_to_ensembl(hgnc)
    drug_targets = targets.select(
        "nct_id", pl.col("drug_id").alias("drugId"), pl.col("target_id").alias("targetId")
    ).unique()
    words = (
        targets.select(
            pl.col("drug_id").alias("drugId"),
            pl.col("target_id").alias("targetId"),
            pl.col("action_type").str.split("; ").alias("actionType"),
        )
        .explode("actionType")
        .drop_nulls()
        .with_columns(pl.col("actionType").str.to_uppercase())
        .unique()
    )
    mechanism = (
        words.group_by("drugId", "actionType")
        .agg(pl.col("targetId").unique().alias("targets"))
        .select("actionType", pl.concat_list("drugId").alias("chemblIds"), "targets")
    )
    properties = pl.read_csv(chemistry.CHEMISTRY_DIR / chemistry.PROPERTIES).rename(
        {"drug_id": "drugId", **{v: f"chem_{v}" for v in chemistry.CHEMBL_FIELDS.values()}}
    )
    chem = properties.join(
        chemistry.drug_actions(mechanism), on="drugId", how="full", coalesce=True
    )
    chem = chem.filter(pl.any_horizontal(pl.col(c).is_not_null() for c in chemistry.FEATURES))
    leads = (
        pl.read_parquet(INTERIM / "genetics.parquet", columns=["nct_id", LEAD_DRUG])
        .drop_nulls()
        .select("nct_id", pl.col(LEAD_DRUG).alias("drugId"), pl.lit(True).alias("is_lead"))
    )
    intrinsic = depmap.tumour_intrinsic(drug_targets, words, depmap.host_directed_ids(hgnc))
    pieces = [
        depmap.trial_features(
            drug_targets,
            intrinsic,
            trials,
            conditions,
            depmap.gene_table(mapping=mapping),
            depmap.lineage_table(mapping=mapping),
        ),
        expression.trial_features(
            drug_targets,
            trials,
            conditions,
            expression.gene_table(),
            expression.matched_tissue_table(),
        ),
        networks.trial_features(drug_targets, networks.gene_table(mapping=mapping)),
        annotations.trial_features(drug_targets, annotations.gene_table()),
    ]
    if version != "2020_names":
        pieces.append(chemistry.trial_features(leads, drug_targets, chem.sort("drugId")))
    table = science_table.assemble(trials, drug_targets, pieces).drop("split", "has_target")
    if version == "2020_names":
        empty = [pl.lit(None, pl.String if c == "chem_action_type" else pl.Float64).alias(c)
                 for c in chemistry.FEATURES]  # fmt: skip
        table = table.with_columns(empty)
    return table


def read_conditions(ids: list[str], path=CONDITIONS_PATH) -> pl.DataFrame:
    """Conditions, MeSH terms and ancestors per trial (one registry pass, cached)."""
    if path.exists():
        return pl.read_parquet(path)
    _, conditions = tg.read_registry(ids)
    conditions.write_parquet(path)
    return conditions


def build_tables(version: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(table, keys) for one link version; writes both under `data/interim/`. No outcome is read."""
    targets = links.investigational_targets(version)
    trials = tg.load_trials().select("nct_id")
    start_year = pl.concat(
        [
            pl.read_parquet(INTERIM / f, columns=["nct_id", "start_year"])
            for f in ("dev.parquet", "test_features.parquet")
        ]
    )
    diseases = trial_disease_ids(trials["nct_id"].to_list())
    keys = link_keys(targets, diseases)
    flag = pl.read_parquet(
        INTERIM / "precedent.parquet", columns=["nct_id", f"precedent_exp_{version}"]
    )
    table = (
        trials.join(start_and_posted(), on="nct_id", how="left")
        .join(flag.rename({f"precedent_exp_{version}": "precedent"}), on="nct_id", how="left")
        .join(genetic_support(targets, diseases, start_year), on="nct_id", how="left")
        .join(biology_columns(version, targets), on="nct_id", how="left")
        .with_columns(
            pl.col("nct_id").is_in(targets["nct_id"].implode()).alias("has_target"),
            pl.col("nct_id").is_in(diseases["nct_id"].implode()).alias("has_disease"),
        )
        .select("nct_id", *(pl.col(c).alias(column(c)) for c in TABLE_COLUMNS))
        .sort("nct_id")
    )
    table.write_parquet(INTERIM / f"{table_name(version)}.parquet")
    keys.write_parquet(INTERIM / f"{keys_name(version)}.parquet")
    return table, keys


@cache
def load_keys(version: str) -> pl.DataFrame:
    """The (nct_id, kind, key) table for `version`, read once."""
    return pl.read_parquet(INTERIM / f"{keys_name(version)}.parquet")


# ---------------------------------------------------------------- small helpers for reports


def thirds(values: np.ndarray) -> np.ndarray:
    """0, 1, 2 for the bottom, middle and top third by rank.

    Tied values get the average of their ranks, so they always land in the same third (the thirds
    are then not exactly equal in size when many values tie).
    """
    ordered = np.sort(values)
    below = np.searchsorted(ordered, values, side="left")  # values strictly smaller
    up_to = np.searchsorted(ordered, values, side="right")  # values smaller or equal
    ranks = (below + 1 + up_to) / 2  # a tie takes the average of the ranks it spans
    return np.clip(np.floor((ranks - 1) * 3 / len(values)).astype(int), 0, 2)


def main() -> None:
    for version in links.VERSIONS:
        table, keys = build_tables(version)
        print(f"{version}: {table.height:,} trials, {keys.height:,} keys")


if __name__ == "__main__":
    main()
