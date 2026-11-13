"""Precedent: before this trial started, had another molecule on the same target already started
a Phase 3 trial in the same disease?

    uv run python -m elman_exercise.precedent    # -> data/interim/precedent.parquet

The flag, for a trial T (all splits; features only, no outcome of any trial is read):

    there is a Phase 3 trial U, and a drug in U, and a target gene, and a disease, such that
      - T's drug and U's drug both act on that target;
      - T and U both carry that disease (Open Targets' own disease id, matched exactly);
      - U's drug is a different molecule from every drug listed for T (parent molecules compared,
        so a salt of T's own drug is not "another molecule");
      - U is another trial, was not withdrawn (a withdrawn record never started), its LATEST
        possible start day is before T's EARLIEST possible start day (a month alone, "2015-03",
        means the whole month), and its first-submitted date, on its latest possible day, is also
        before T's earliest possible start day.

Only the fact that U had started is used. Whether U succeeded is never read, so no outcome can
leak. Phase 3 means the registry record's CURRENT phase list is exactly PHASE3. A combined
Phase 2/3 record is not counted: it can be a Phase 2 start that was later relabelled. The
sensitivity columns (`SENSITIVITY_FLAGS`) count a Phase 2/3 record as a Phase 3 as well, to show
how much the headline depends on that choice.

Flags, each a yes/no per trial:

- `precedent_today`, `precedent_2020`: T's drugs are the experimental-arm drugs if any listed drug
  matches an experimental-arm name, otherwise every listed drug (the fallback the first prototype
  used). U's drugs are every drug listed for U. Drug-to-target links are today's (Open Targets
  26.09) or June 2020's (20.06): see `science.links`. Salts inherit their family's targets in
  both (`links.MERGE_SALTS`).
- `precedent_exp_today`, `precedent_exp_2020`: the stricter version. Both T's and U's drugs must
  be matched to an experimental arm; no fallback, so a comparator never supplies a target. This is
  the version `whathow` uses.
- `precedent_2020_names`, `precedent_exp_2020_names`: the same two flags with the trial-to-drug step
  dated as well: registry intervention names are matched only to the names in the 20.06 drug table
  (`links.match_names_2006`; a short stoplist of generic names such as insulin or saline is
  never matched), and the targets are 20.06's. Its names cover the table's parent molecules only.
  Both drug sets come from that match.
- `precedent_exp_<version>_p23`: the strict flag counting a Phase 2/3 record as a Phase 3 too.

`first_in_disease` is a plain column used only to describe results: True when no earlier trial
(any phase) in Open Targets' ClinicalTrials.gov report lists the same parent molecule (today's
drug match) and one of the same diseases, and the trial has a drug. "Earlier" is the same
conservative rule as above: latest possible start before T's earliest possible start, first
submitted before that start (where the registry holds the record; a trial only Open Targets
knows has no submission date and is judged by its start date alone), and not withdrawn. A
month-level start in Open Targets is that month's last day (checked against the registry), so
it is read as the latest possible day. Null if the trial has no drug listed.
`first_in_disease_2020_names` is the same question asked of each row's own link version for the
names version: only registry Phase 2 and 3 records can be earlier trials there, matched by 20.06 name.

Still today's, in the first two link versions: which drug each trial tested (name match against
the registry as it is now). Still today's in all three: the trial's disease (Open Targets' current
mapping), `lead_drug`, and the registry's phase, status and dates (a sponsor can have edited them
since). A trial with no start date gets no flag. Dating: the flags depend on a release published
after the trials started, so they carry the label "after the start" (`FEATURES`) and the models
that read them are not marked honest.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from elman_exercise import genetics
from elman_exercise.dating import AFTER_START
from elman_exercise.opentargets import INTERIM, OT_DIR
from elman_exercise.science import links
from elman_exercise.science import targets as tg

TABLE = "precedent"
TABLE_PATH = INTERIM / f"{TABLE}.parquet"
FLAGS = ("precedent_today", "precedent_2020", "precedent_exp_today", "precedent_exp_2020")
# The third link version dates the trial-to-drug name match too (`science.links`, "2020_names").
NAMES_FLAGS = ("precedent_2020_names", "precedent_exp_2020_names")
ALL_FLAGS = (*FLAGS, *NAMES_FLAGS)
# The strict flag again, with a combined Phase 2/3 record also counted as a Phase 3 precedent.
SENSITIVITY_FLAGS = tuple(f"precedent_exp_{v}_p23" for v in ("today", "2020", "2020_names"))
FIRST_NAMES = "first_in_disease_2020_names"
FEATURES: dict[str, str] = dict.fromkeys((*ALL_FLAGS, *SENSITIVITY_FLAGS), AFTER_START)
PHASE3 = "PHASE3"

PAIR_COLUMNS = [
    "nct_id",
    "t_drug",
    "target_id",
    "disease",
    "u_nct",
    "u_drug",
    "u_start_last",
    "u_submit_last",
]
SAMPLE = 500  # pairs rechecked one by one by `check_pairs` in the report


# ---------------------------------------------------------------- inputs


def listed_drugs(
    reports: pl.DataFrame,
    molecules: pl.DataFrame,
    registry: pl.DataFrame,
    parent_of: dict[str, str],
) -> pl.DataFrame:
    """(nct_id, drug_id, canon, is_exp, is_main) for every drug Open Targets lists for a registry trial.

    `is_exp`: the drug matches an experimental-arm intervention name (`targets.investigational_drugs`).
    `is_main`: `is_exp`, or any listed drug when the trial has no match at all (the fallback).
    `canon` is the drug's parent molecule if it has one, else itself.
    """
    ids = registry["nct_id"].to_list()
    experimental = dict(registry.select("nct_id", "exp_names").rows())
    named = genetics.trial_drug_names(reports, ids).unique(
        ["nct_id", "drugId"], maintain_order=True
    )
    exp = tg.investigational_drugs(ids, reports, molecules, experimental).select(
        "nct_id", "drugId", pl.lit(True).alias("is_exp")
    )
    table = (
        named.join(exp, on=["nct_id", "drugId"], how="left")
        .with_columns(pl.col("is_exp").fill_null(False))
        .with_columns(pl.col("is_exp").any().over("nct_id").alias("has_exp"))
        .with_columns(
            (pl.col("is_exp") | ~pl.col("has_exp")).alias("is_main"),
            pl.col("drugId").replace_strict(parent_of, default=pl.col("drugId")).alias("canon"),
        )
    )
    return table.select("nct_id", pl.col("drugId").alias("drug_id"), "canon", "is_exp", "is_main")


def trial_diseases(reports: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, disease): Open Targets' disease ids for each trial."""
    return (
        reports.select(pl.col("id").alias("nct_id"), "diseases")
        .explode("diseases")
        .select("nct_id", pl.col("diseases").struct.field("diseaseId").alias("disease"))
        .drop_nulls()
        .unique()
    )


@dataclass(frozen=True)
class Inputs:
    """Everything the flag reads, as tables, so a test can supply made-up ones."""

    registry: pl.DataFrame  # nct_id, phases, withdrawn, start_first/last, submit_first/last
    drugs: pl.DataFrame  # listed_drugs()
    diseases: pl.DataFrame  # trial_diseases()
    scored: tuple[str, ...]  # the trials to flag
    # The same kind of table as `drugs`, matched by 20.06 names (`names_listed_drugs`); drug ids
    # are 20.06 ids. Only the "2020_names" version reads it.
    names_drugs: pl.DataFrame | None = None


def names_listed_drugs(matched: pl.DataFrame) -> pl.DataFrame:
    """`listed_drugs`-shaped table from `links.match_names_2006`: (nct_id, drug_id, canon, is_exp, is_main).

    A drug's 20.06 entry is its own molecule (`canon` is the id). `is_main` falls back to every
    matched drug when a trial has no experimental-arm match, as in the other versions.
    """
    return (
        matched.select("nct_id", "drug_id", "is_exp")
        .with_columns(pl.col("drug_id").alias("canon"))
        .with_columns(pl.col("is_exp").any().over("nct_id").alias("has_exp"))
        .with_columns((pl.col("is_exp") | ~pl.col("has_exp")).alias("is_main"))
        .select("nct_id", "drug_id", "canon", "is_exp", "is_main")
    )


def inputs_for_version(inputs: Inputs, version: str) -> Inputs:
    """`inputs` with the trial-to-drug table that belongs to `version`."""
    if version != "2020_names":
        return inputs
    if inputs.names_drugs is None:
        raise ValueError("these inputs carry no 20.06 name match")
    return Inputs(
        inputs.registry, inputs.names_drugs, inputs.diseases, inputs.scored, inputs.names_drugs
    )


def load_inputs(ot_dir: Path = OT_DIR) -> tuple[Inputs, pl.DataFrame]:
    """The real inputs for every benchmark trial, and Open Targets' full report (for `first_in_disease`)."""
    links.download()
    registry = links.registry_table()
    reports = pl.read_parquet(
        ot_dir / "clinical_report.parquet", columns=["id", "drugs", "diseases", "trialStartDate"]
    )
    molecules = pl.read_parquet(ot_dir / "drug_molecule.parquet")
    family = pl.read_parquet(links.OT2006_DIR / links.SLIM_FAMILY_2609)
    parent_of = {d: p for d, p in family.select("drug_id", "parent_id").rows() if p}
    drugs = listed_drugs(reports, molecules, registry, parent_of)
    names_drugs = names_listed_drugs(links.trial_drugs_2020_names())
    ids = tuple(links.trial_ids())
    inputs = Inputs(registry, drugs, trial_diseases(reports), ids, names_drugs)
    return inputs, reports


# ---------------------------------------------------------------- the pairs behind a flag


def phase3_trials(registry: pl.DataFrame, include_combined: bool = False) -> pl.DataFrame:
    """(u_nct, u_start_last, u_submit_last): the records that count as an earlier Phase 3.

    The phase list must be exactly PHASE3 (with `include_combined`, PHASE3 alone or PHASE2 and
    PHASE3 together) and the record must not be withdrawn.
    """
    is_phase3 = pl.col("phases").list.contains(PHASE3)
    if not include_combined:
        is_phase3 = is_phase3 & (pl.col("phases").list.len() == 1)
    return registry.filter(is_phase3 & ~pl.col("withdrawn").fill_null(False)).select(
        pl.col("nct_id").alias("u_nct"),
        pl.col("start_last").alias("u_start_last"),
        pl.col("submit_last").alias("u_submit_last"),
    )


def precedent_pairs(
    inputs: Inputs,
    drug_links: pl.DataFrame,
    strict: bool = False,
    include_combined: bool = False,
) -> pl.DataFrame:
    """Every (trial T, earlier Phase 3 trial U) pair that makes T's flag true, with the shared
    target and disease: columns `PAIR_COLUMNS`. One row per T drug, U drug, target and disease.

    `drug_links` is (drug_id, target_id, ...) from `links.drug_targets`. `strict` asks for
    experimental-arm drugs on both sides, with no fallback. `include_combined` also counts a
    Phase 2/3 record as a Phase 3 (the sensitivity version). Dates: U's latest possible start and
    first-submission days must both be strictly before T's earliest possible start day.
    """
    target_of = drug_links.select("drug_id", "target_id").unique()
    t_drugs = inputs.drugs.filter(pl.col("nct_id").is_in(list(inputs.scored)))
    t_all = t_drugs.group_by("nct_id").agg(
        pl.col("drug_id").alias("t_listed"), pl.col("canon").alias("t_canon")
    )
    own = t_drugs.filter(pl.col("is_exp" if strict else "is_main")).select(
        "nct_id", pl.col("drug_id").alias("t_drug")
    )
    t_side = (
        own.join(target_of.rename({"drug_id": "t_drug"}), on="t_drug")
        .join(inputs.diseases, on="nct_id")
        .join(
            inputs.registry.select(
                "nct_id", pl.col("start_first").alias("t_start_first")
            ).drop_nulls(),
            on="nct_id",
        )
        .join(t_all, on="nct_id")
    )

    phase3 = phase3_trials(inputs.registry, include_combined)
    u_drugs = inputs.drugs.filter(pl.col("is_exp")) if strict else inputs.drugs
    u_side = (
        u_drugs.select(
            pl.col("nct_id").alias("u_nct"),
            pl.col("drug_id").alias("u_drug"),
            pl.col("canon").alias("u_canon"),
        )
        .join(phase3, on="u_nct")
        .join(target_of.rename({"drug_id": "u_drug"}), on="u_drug")
        .join(inputs.diseases.rename({"nct_id": "u_nct"}), on="u_nct")
    )
    joined = t_side.join(u_side, on=["target_id", "disease"])
    return (
        joined.filter(
            (pl.col("u_nct") != pl.col("nct_id"))
            & (pl.col("u_start_last") < pl.col("t_start_first"))
            & (pl.col("u_submit_last") < pl.col("t_start_first"))
            & ~pl.col("t_listed").list.contains(pl.col("u_drug"))
            & ~pl.col("t_canon").list.contains(pl.col("u_canon"))
        )
        .select(PAIR_COLUMNS)
        .unique()
        .sort(PAIR_COLUMNS)
    )


def flag_from_pairs(pairs: pl.DataFrame, scored: Iterable[str], name: str) -> pl.DataFrame:
    """(nct_id, <name>): True if the trial has any pair."""
    flagged = set(pairs["nct_id"])
    return pl.DataFrame(
        {"nct_id": list(scored), name: [n in flagged for n in scored]},
        schema={"nct_id": pl.String, name: pl.Boolean},
    )


def precedent_flags(
    inputs: Inputs, versions: dict[str, pl.DataFrame] | None = None
) -> pl.DataFrame:
    """The flag columns for every scored trial. `versions` maps a link version to its drug links.

    By default all three versions are run; "2020_names" needs `inputs.names_drugs` and is left
    out when that is missing. The columns are `ALL_FLAGS` (those of the versions run), then the
    Phase 2/3 sensitivity columns of the strict flag.
    """
    if versions is None:
        versions = {v: links.drug_targets(v, merge_salts=links.MERGE_SALTS) for v in links.VERSIONS}
        if inputs.names_drugs is None:
            versions.pop("2020_names")
    scored = sorted(inputs.scored)
    table = pl.DataFrame({"nct_id": scored}, schema={"nct_id": pl.String})
    for strict in (False, True):
        for version, drug_links in versions.items():
            name = f"precedent_{'exp_' if strict else ''}{version}"
            version_inputs = inputs_for_version(inputs, version)
            pairs = precedent_pairs(version_inputs, drug_links, strict)
            table = table.join(flag_from_pairs(pairs, scored, name), on="nct_id")
    for version, drug_links in versions.items():
        name = f"precedent_exp_{version}_p23"
        pairs = precedent_pairs(inputs_for_version(inputs, version), drug_links, True, True)
        table = table.join(flag_from_pairs(pairs, scored, name), on="nct_id")
    keep = [c for c in (*ALL_FLAGS, *SENSITIVITY_FLAGS) if c in table.columns]
    return table.select("nct_id", *keep)


# ---------------------------------------------------------------- first trial of the drug in the disease


def earlier_than_start(others: pl.DataFrame, mine: pl.DataFrame) -> pl.DataFrame:
    """The (nct_id) of `mine` trials that have an earlier trial of the same molecule and disease.

    `mine`: nct_id, canon, disease, t_start. `others`: o_nct, canon, disease, o_last (latest
    possible start day), o_submit_last (null when no submission date is known), o_withdrawn.
    """
    return (
        mine.join(others, on=["canon", "disease"])
        .filter(
            (pl.col("o_nct") != pl.col("nct_id"))
            & (pl.col("o_last") < pl.col("t_start"))
            & (pl.col("o_submit_last").is_null() | (pl.col("o_submit_last") < pl.col("t_start")))
            & ~pl.col("o_withdrawn").fill_null(False)
        )
        .select("nct_id")
        .unique()
    )


def first_flag(
    scored: list[str], mine: pl.DataFrame, earlier: pl.DataFrame, name: str
) -> pl.DataFrame:
    """(nct_id, <name>): True unless `earlier` lists the trial; null if `mine` does not know it."""
    known = mine.select("nct_id").unique().with_columns(pl.lit(True).alias("known"))
    found = earlier.with_columns(pl.lit(False).alias(name))
    return (
        pl.DataFrame({"nct_id": scored}, schema={"nct_id": pl.String})
        .join(known, on="nct_id", how="left")
        .join(found, on="nct_id", how="left")
        .select(
            "nct_id",
            pl.when(pl.col("known").is_null())
            .then(None)
            .otherwise(pl.col(name).fill_null(True))
            .alias(name),
        )
    )


def trials_to_describe(inputs: Inputs, drugs: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, canon, disease, t_start): the scored trials' main drugs, diseases and start day."""
    return (
        drugs.filter(pl.col("nct_id").is_in(list(inputs.scored)) & pl.col("is_main"))
        .select("nct_id", "canon")
        .unique()
        .join(inputs.diseases, on="nct_id")
        .join(
            inputs.registry.select("nct_id", pl.col("start_first").alias("t_start")).drop_nulls(),
            on="nct_id",
        )
    )


def first_in_disease(inputs: Inputs, reports: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, first_in_disease): no earlier trial lists the trial's parent molecule and disease.

    Today's drug match. "Earlier": the other trial's latest possible start is before this trial's
    earliest possible start; where the registry holds the other trial it must also have been
    first submitted (latest possible day) before then and must not be withdrawn; a trial the
    registry does not hold uses Open Targets' own start date (a month-level date there is the
    month's last day, so it is the latest possible day) and has no submission check. Null for a
    trial with no drug listed, no disease or no start date.
    """
    scored = list(inputs.scored)
    mine = trials_to_describe(inputs, inputs.drugs)
    all_drugs = (
        reports.select(pl.col("id").alias("o_nct"), "drugs", "trialStartDate")
        .explode("drugs")
        .select(
            "o_nct",
            pl.col("drugs").struct.field("drugId").alias("o_drug"),
            pl.col("trialStartDate").alias("o_ot_start"),
        )
        .drop_nulls("o_drug")
    )
    parent = dict(inputs.drugs.select("drug_id", "canon").unique().iter_rows())
    other = (
        all_drugs.with_columns(
            pl.col("o_drug").replace_strict(parent, default=pl.col("o_drug")).alias("canon")
        )
        .join(inputs.diseases.rename({"nct_id": "o_nct"}), on="o_nct")
        .join(
            inputs.registry.select(
                pl.col("nct_id").alias("o_nct"),
                "start_last",
                pl.col("submit_last").alias("o_submit_last"),
                pl.col("withdrawn").alias("o_withdrawn"),
            ),
            on="o_nct",
            how="left",
        )
        .select(
            "o_nct",
            "canon",
            "disease",
            pl.coalesce("start_last", "o_ot_start").alias("o_last"),
            "o_submit_last",
            "o_withdrawn",
        )
    )
    return first_flag(scored, mine, earlier_than_start(other, mine), "first_in_disease")


def first_in_disease_names(inputs: Inputs) -> pl.DataFrame:
    """(nct_id, first_in_disease_2020_names): the same question under the 20.06 name match.

    Both the trial's drugs and the earlier trials' drugs come from `inputs.names_drugs`, so only
    registry Phase 2 and 3 records can be the earlier trial (there is no 20.06 name match for the
    rest of Open Targets' trials). The date rules are those of `first_in_disease`.
    """
    names = inputs_for_version(inputs, "2020_names").drugs
    mine = trials_to_describe(inputs, names)
    other = (
        names.select(pl.col("nct_id").alias("o_nct"), "canon")
        .unique()
        .join(inputs.diseases.rename({"nct_id": "o_nct"}), on="o_nct")
        .join(
            inputs.registry.select(
                pl.col("nct_id").alias("o_nct"),
                pl.col("start_last").alias("o_last"),
                pl.col("submit_last").alias("o_submit_last"),
                pl.col("withdrawn").alias("o_withdrawn"),
            ),
            on="o_nct",
        )
    )
    return first_flag(list(inputs.scored), mine, earlier_than_start(other, mine), FIRST_NAMES)


# ---------------------------------------------------------------- funnel and checks


def funnel(inputs: Inputs, drug_links: pl.DataFrame, strict: bool = False) -> list[tuple[str, int]]:
    """How many scored trials survive each step of the flag, in order. Counts only; no outcomes.

    The steps: has a drug listed (by Open Targets, or by the 20.06 name match in the names
    version); has a drug of the kind used (experimental arm, or with the fallback); the drug has a
    target; the trial has a disease; the trial has a start date; some Phase 3 trial shares a target
    and disease with a different molecule (whenever it started); the sharing trial started before;
    and was first submitted before; flag.
    """
    scored = set(inputs.scored)
    kind = "is_exp" if strict else "is_main"
    stages: list[tuple[str, set[str]]] = [("trials", scored)]
    listed = set(inputs.drugs.filter(pl.col("nct_id").is_in(list(scored)))["nct_id"])
    stages.append(("has a drug listed (Open Targets's list, or the 20.06 name match)", listed))
    own = inputs.drugs.filter(pl.col("nct_id").is_in(list(scored)) & pl.col(kind))
    stages.append(("has a drug of the kind used", set(own["nct_id"])))
    target_of = drug_links.select("drug_id", "target_id").unique()
    with_target = own.join(target_of, on="drug_id")
    stages.append(("... that has a target in this release", set(with_target["nct_id"])))
    diseased = set(with_target["nct_id"]) & set(inputs.diseases["nct_id"])
    stages.append(("... and has a disease", diseased))
    dated = diseased & set(inputs.registry.drop_nulls("start_first")["nct_id"])
    stages.append(("... and has a start date", dated))
    for label, frame_in in (
        (
            "... and a Phase 3 trial shares a target and disease with another molecule (any date)",
            replace_dates(inputs, far_past=True),
        ),
        (
            "... that started before it",
            replace_dates(inputs, submit_free=True),
        ),
        ("... and was first submitted before it (the flag)", inputs),
    ):
        pairs = precedent_pairs(frame_in, drug_links, strict)
        stages.append((label, set(pairs["nct_id"]) & dated))
    return [(label, len(members)) for label, members in stages]


def replace_dates(inputs: Inputs, far_past: bool = False, submit_free: bool = False) -> Inputs:
    """Inputs with the date rules relaxed, to count how many trials each date rule removes.

    `far_past`: every Phase 3 trial counts as started and submitted long ago.
    `submit_free`: the first-submitted rule is dropped, the start rule stays.
    The name table is carried over.
    """
    long_ago = pl.lit(date(1900, 1, 1))
    registry = inputs.registry
    if far_past:
        registry = registry.with_columns(
            pl.when(pl.col("phases").list.contains(PHASE3))
            .then(long_ago)
            .otherwise(pl.col("start_last"))
            .alias("start_last"),
            long_ago.alias("submit_first"),
            long_ago.alias("submit_last"),
        )
    if submit_free:
        registry = registry.with_columns(
            long_ago.alias("submit_first"), long_ago.alias("submit_last")
        )
    return Inputs(registry, inputs.drugs, inputs.diseases, inputs.scored, inputs.names_drugs)


def plain_day_bounds(text: str | None) -> tuple[date, date] | None:
    """The first and last day a registry date string could mean, in plain Python.

    Written separately from `history.date_range` so a check does not share its code.
    """
    match = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", (text or "").strip())
    if not match:
        return None
    year, month, day = int(match[1]), int(match[2]), match[3]
    if day:
        return date(year, month, int(day)), date(year, month, int(day))
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def check_pairs(
    inputs: Inputs,
    drug_links: pl.DataFrame,
    pairs: pl.DataFrame,
    sample: int = 300,
    reports: pl.DataFrame | None = None,
    drugs_in_reports: bool = True,
) -> list[str]:
    """Recheck a sample of pairs one row at a time, with plain Python.

    Returns a list of problems; empty means every sampled pair meets every part of the rule.

    What is recomputed from raw data, not from the tables the flag was built from: when the
    registry table carries the raw `start` and `first_submit` strings, the start and submission
    bounds are re-derived from them (`plain_day_bounds`) instead of reading `start_last` and
    friends; the phase list and withdrawn status are the registry's own columns; with `reports`
    (Open Targets' raw clinical report) the two trials' diseases and drugs are looked up in it
    rather than in `inputs.diseases` and `inputs.drugs`. What is NOT independent: the targets
    come from `drug_links`, and which drug is the trial's experimental one comes from `inputs.drugs`.
    The 20.06 name version has no Open Targets report to check drugs against.
    """
    registry = {r["nct_id"]: r for r in inputs.registry.iter_rows(named=True)}
    targets: dict[str, set[str]] = {}
    for drug, target in drug_links.select("drug_id", "target_id").iter_rows():
        targets.setdefault(drug, set()).add(target)
    sampled = pairs.sample(min(sample, pairs.height), seed=0)
    wanted = set(sampled["nct_id"]) | set(sampled["u_nct"])
    diseases: dict[str, set[str]] = {}
    listed: dict[str, list[tuple[str, str]]] = {}
    for nct, drug, canon in inputs.drugs.select("nct_id", "drug_id", "canon").iter_rows():
        listed.setdefault(nct, []).append((drug, canon))
    reported: dict[str, set[str]] = {}
    if reports is None:
        for nct, disease in inputs.diseases.iter_rows():
            diseases.setdefault(nct, set()).add(disease)
    else:
        for nct, drugs_, diseases_ in (
            reports.filter(pl.col("id").is_in(list(wanted)))
            .select("id", "drugs", "diseases")
            .iter_rows()
        ):
            reported[nct] = {d["drugId"] for d in drugs_ or [] if d["drugId"]}
            diseases[nct] = {d["diseaseId"] for d in diseases_ or [] if d["diseaseId"]}
    problems = []
    for row in sampled.iter_rows(named=True):
        t, u = registry[row["nct_id"]], registry[row["u_nct"]]
        where = f"{row['nct_id']} <- {row['u_nct']}"
        if row["target_id"] not in targets.get(row["t_drug"], set()):
            problems.append(f"{where}: target not a target of the trial's drug")
        if row["target_id"] not in targets.get(row["u_drug"], set()):
            problems.append(f"{where}: target not a target of the earlier trial's drug")
        if row["disease"] not in diseases.get(row["nct_id"], set()) or row[
            "disease"
        ] not in diseases.get(row["u_nct"], set()):
            problems.append(f"{where}: disease not shared")
        if (
            reports is not None
            and drugs_in_reports
            and (
                row["t_drug"] not in reported.get(row["nct_id"], set())
                or row["u_drug"] not in reported.get(row["u_nct"], set())
            )
        ):
            problems.append(f"{where}: a drug is not in Open Targets' list for its trial")
        if PHASE3 not in u["phases"] or len(u["phases"]) != 1 or u.get("withdrawn"):
            problems.append(f"{where}: earlier trial is not a Phase 3 that went ahead")
        if "start" in t and "first_submit" in u:
            t_start, u_start = plain_day_bounds(t["start"]), plain_day_bounds(u["start"])
            u_sub = plain_day_bounds(u["first_submit"])
            ok = bool(t_start and u_start and u_sub)
            ok = ok and u_start[1] < t_start[0] and u_sub[1] < t_start[0]
        else:
            ok = u["start_last"] < t["start_first"] and u["submit_last"] < t["start_first"]
        if not ok:
            problems.append(f"{where}: earlier trial did not start and register before")
        t_canon = {c for _, c in listed.get(row["nct_id"], [])}
        t_ids = {d for d, _ in listed.get(row["nct_id"], [])}
        u_canon = dict(listed.get(row["u_nct"], [])).get(row["u_drug"])
        if row["u_drug"] in t_ids or u_canon in t_canon:
            problems.append(f"{where}: earlier drug is the trial's own molecule")
    return problems


# ---------------------------------------------------------------- build


def build(path: Path = TABLE_PATH) -> pl.DataFrame:
    """Compute the flags and `first_in_disease` for every benchmark trial and write the table."""
    inputs, reports = load_inputs()
    table = (
        precedent_flags(inputs)
        .join(first_in_disease(inputs, reports), on="nct_id")
        .join(first_in_disease_names(inputs), on="nct_id")
    )
    table.write_parquet(path)
    return table


def main() -> None:
    table = build()
    print(f"wrote {TABLE_PATH} ({table.height:,} trials)")
    for flag in (*ALL_FLAGS, *SENSITIVITY_FLAGS):
        print(f"  {flag}: {int(table[flag].sum()):,} trials flagged")


if __name__ == "__main__":
    main()
