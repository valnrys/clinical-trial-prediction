"""Human genetic evidence for each trial's drug target, from Open Targets.

    uv run python -m elman_exercise.opentargets   # download first
    uv run python -m elman_exercise.genetics      # -> data/interim/genetics.parquet
    uv run python -m elman_exercise.genetics --check-timeseries

The definition is fixed in BELIEFS.md (B18, B18a): a trial has genetic support if
any target of its drug has a genetic-association score above zero for the trial's
disease. Two versions: with all evidence known today, and with only evidence dated
before the year the trial started ("evidence dated before the trial started, as
curated today": the history is rebuilt from publication years using today's curation,
today's gene-assignment model and today's disease ontology; against the archived 2020
release it agrees on which pairs have no evidence but on positives only about half the
time, so it is not what a predictor could have seen at the time).

The table holds features only, for every trial in every split. It reads no label
and no outcome column. The registry is read only for arms and interventions.

What "genetic evidence" is
--------------------------
The `genetic_association` **datatype** of Open Targets release 26.09, nothing else.
Checked on the release itself (the `datatypeId` column of each evidence table under
https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.09/output/evidence_*/):
that datatype is built from five sources only: ClinVar germline (`eva`), gene burden
tests (`gene_burden`), GWAS credible sets (`gwas_credible_sets`), Orphanet
(`orphanet`) and UniProt curated variants (`uniprot_variants`). ClinGen,
Gene2Phenotype, Genomics England PanelApp and UniProt literature sit in a separate
datatype, `genetic_literature`, and are NOT counted, so much curated Mendelian
evidence is excluded. Clinical precedence (`known_drug`, datatype `clinical`),
somatic mutation, literature and animal-model evidence are excluded too.

"Supported" means score > 0, which is a low bar. ClinVar entries below "pathogenic"
score above zero: "drug response", "association" and "protective" 0.9, "likely
pathogenic" 0.7, and "uncertain significance", "conflicting" and "other" 0.3
(https://platform-docs.opentargets.org/evidence). So some of the support is
pharmacogenomic or of uncertain significance. Scores are "indirect": evidence on
narrower disease terms is rolled up to broader ones, and not down.

What it counts, and what it does not
------------------------------------
- **Which drugs.** Every drug Open Targets lists for the trial's ClinicalTrials.gov
  record. That includes comparators and background therapy: Open Targets' report
  does not say which arm a drug was in. A trial of drug A against active drug B gets
  the targets of both. This blurs the signal, and it favours trials listing many
  drugs. The `*_investigational` columns are the sensitivity version: support
  counts only through a drug matched to the treatment under test (below).
- **Which disease.** Open Targets' own mapping of the trial's conditions to
  disease terms (EFO or MONDO), not ours.
- **Pairs.** Every (target, disease) combination: each target of each drug,
  against each mapped disease. The trial's score is the highest over its pairs.
  A maximum over many pairs rises with the number of pairs, so `few_targets`
  (three targets or fewer) is there for a sensitivity check.
- **Not mapped is not "no evidence".** A trial with no (target, disease) pair has
  null scores. A trial with pairs but no genetic evidence scores 0.

The investigational-drug flag
-----------------------------
Open Targets' drug names for the trial (the name as the registry wrote it, and the
standard molecule name) are matched to the names and other names of the registry
interventions chosen by `dataset.experimental_interventions` (drug-like
interventions given to an experimental arm and to no comparator arm; if none, to
any experimental arm; if none, all). A drug matches when, after lower-casing and
reducing to letters and digits, its name is a whole-word phrase inside an
intervention name or the reverse (at least 3 characters). Failure modes: brand,
salt or code names that differ between the two sources (missed, so real
investigational support is lost); a combination written as one intervention
("A + B") marks both parts; a background drug given to the experimental arm only
is counted as investigational; single-arm trials with no arm types fall back to
every drug; a trial where nothing matches has no investigational drug at all.

What the history column can be trusted for
------------------------------------------
Each association row carries a `timeseries` with one score per year, run through to
2026. `--check-timeseries` tests it; the numbers are in the project report. In
short: each year's score counts only evidence dated up to that year; the entry
with no year repeats today's score; the first entry of every row is 0 and sits in
the year before the first evidence (or in 2009, if there is none earlier), so a
year with no entry means "no evidence yet". But the history is not an exact
replay: about 0.7% of rows end on a score different from today's, and about 5%
dip at some point (scores re-weighted), though a dip is almost never to zero. And
evidence is dated by the year its source was published, not the year it became
known to the field.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from elman_exercise.dating import AFTER_START, DATED
from elman_exercise.opentargets import GENETIC, INTERIM, OT_DIR

OUT_PATH = INTERIM / "genetics.parquet"
TABLE = "genetics"

# How each column a model might use is dated (see `dating`). Being mapped to Open
# Targets at all partly reflects hindsight: a drug gets a standard identifier and a
# target list only if it progressed far enough to be curated, so anything that
# depends on the mapping is AFTER_START, even if the value itself looks timeless.
FEATURES: dict[str, str] = {
    "mapped": AFTER_START,  # whether the drug could be linked at all: hindsight
    "n_targets": AFTER_START,  # a count from today's mapping; null or zero when unmapped
    "n_pairs": AFTER_START,  # target-disease pairs found today
    "few_targets": AFTER_START,  # derived from n_targets
    "drug_type": AFTER_START,  # taken from the Open Targets drug record, which exists only if mapped
    "investigational_mapped": AFTER_START,  # same mapping hindsight
    "lead_drug": AFTER_START,  # an Open Targets drug id: the identity itself comes from the mapping
    "genetic_score_today": AFTER_START,  # all evidence known today
    "supported_today": AFTER_START,
    "supported_today_investigational": AFTER_START,
    "genetic_score_at_start": DATED,  # evidence dated before the start year (still today's curation)
    "supported_at_start": DATED,
    "supported_at_start_investigational": DATED,
    # Kept "dated", as history.py labels its own counts: the COUNT uses only trials that
    # started strictly before this one (by start dates), so nothing from after this trial
    # began is in it. Which drug is "the lead drug" comes from today's mapping, the same limit
    # history.py states; that limit is stated there once and is not repeated per column.
    "lead_drug_earlier_trials": DATED,
}
FEW_TARGETS = 3

# Grading, fixed before any outcome was examined.
# Set on 2026-11-10 from the terciles of the positive scores of development-set
# trials (features only, no labels joined). Not to be changed after outcomes are
# looked at (BELIEFS.md, B18a). Bands: 0 = none; up to the first cut = low; up to
# the second = middle; above = high.
# Development-set sizes at these cuts: today none 2,712 / low 476 / middle 484 / high 470;
# at start none 3,717 / low 152 / middle 131 / high 140.
TODAY_CUTS = (0.6082967170759085, 0.8394763380649297)
START_CUTS = (0.5474670453683177, 0.7820825516900308)
BANDS = ("none", "low", "middle", "high")

DRUG_TYPE_GROUPS = {
    "small molecule": "small molecule",
    "antibody": "antibody",
    "protein": "protein",
    "enzyme": "protein",
    "oligonucleotide": "oligonucleotide",
    "cell": "cell",
    "gene": "gene",
    "vaccine": "vaccine",
    "oligosaccharide": "other",
    "unknown": "unknown",
}


def band(score: float | None, cuts: tuple[float, float]) -> str | None:
    """Band of a score: none (0), low, middle or high. None stays None."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    if score <= 0:
        return "none"
    low, high = cuts
    if score <= low:
        return "low"
    return "middle" if score <= high else "high"


def tercile_cuts(scores: list[float]) -> tuple[float, float]:
    """The two cut points that split the positive scores into thirds."""
    positive = sorted(s for s in scores if s is not None and s > 0)
    if len(positive) < 3:
        raise ValueError("Need at least three positive scores to cut into terciles.")
    n = len(positive)
    return positive[n // 3 - 1], positive[2 * n // 3 - 1]


def collapse_drug_types(types: list[str | None]) -> str:
    """One label from the Open Targets drug types of a trial's drugs.

    Unknown types are ignored if any drug has a known one. Two or more known groups
    give "mixed". Nothing known gives "unknown".
    """
    groups = {
        DRUG_TYPE_GROUPS.get(t.strip().lower(), "other") for t in types if t and t.strip()
    } - {"unknown"}
    if not groups:
        return "unknown"
    return groups.pop() if len(groups) == 1 else "mixed"


# --- the join ---------------------------------------------------------------------


def trial_drugs(reports: pl.DataFrame, ids: list[str]) -> pl.DataFrame:
    """(nct_id, drugId) for every drug Open Targets lists for these trials."""
    return (
        reports.filter(pl.col("id").is_in(ids))
        .select(pl.col("id").alias("nct_id"), "drugs")
        .explode("drugs")
        .select("nct_id", pl.col("drugs").struct.field("drugId").alias("drugId"))
        .drop_nulls()
        .unique()
    )


def trial_diseases(reports: pl.DataFrame, ids: list[str]) -> pl.DataFrame:
    """(nct_id, diseaseId) for the diseases Open Targets maps the trials' conditions to."""
    return (
        reports.filter(pl.col("id").is_in(ids))
        .select(pl.col("id").alias("nct_id"), "diseases")
        .explode("diseases")
        .select("nct_id", pl.col("diseases").struct.field("diseaseId").alias("diseaseId"))
        .drop_nulls()
        .unique()
    )


def trial_pairs(
    drugs: pl.DataFrame, diseases: pl.DataFrame, drug_targets: pl.DataFrame
) -> pl.DataFrame:
    """(nct_id, targetId, diseaseId): each target of each drug against each disease."""
    targets = drugs.join(drug_targets.select("drugId", "targetId").unique(), on="drugId")
    return targets.select("nct_id", "targetId").unique().join(diseases, on="nct_id")


def long_history(associations: pl.DataFrame) -> pl.DataFrame:
    """One row per (target, disease, year, score): the score history, undated entry dropped."""
    return (
        associations.select("targetId", "diseaseId", "timeseries")
        .explode("timeseries")
        .select(
            "targetId",
            "diseaseId",
            pl.col("timeseries").struct.field("year").alias("year"),
            pl.col("timeseries").struct.field("associationScore").alias("score"),
        )
        .drop_nulls(["year", "score"])
    )


def pair_scores(
    pairs: pl.DataFrame, trials: pl.DataFrame, associations: pl.DataFrame
) -> pl.DataFrame:
    """Each trial-pair with its score today and its score the year before the trial started.

    `trials` has nct_id and start_year (ours). The at-start score is the pair's score
    in the latest history year strictly before the start year. No such year means no
    evidence was dated that early, so 0 (a history begins at 0). A pair with no genetic
    row at all scores 0 both ways. A trial with no start year has a null at-start score.
    """
    current = associations.select(
        "targetId", "diseaseId", pl.col("associationScore").alias("today")
    )
    history = long_history(associations)
    keyed = pairs.join(trials.select("nct_id", "start_year"), on="nct_id")
    earlier = (
        keyed.join(history, on=["targetId", "diseaseId"])
        .filter(pl.col("year") < pl.col("start_year"))
        .sort("year")
        .group_by("nct_id", "targetId", "diseaseId", maintain_order=True)
        .agg(pl.col("score").last().alias("at_start"))
    )
    scored = (
        keyed.join(current, on=["targetId", "diseaseId"], how="left")
        .join(earlier, on=["nct_id", "targetId", "diseaseId"], how="left")
        .with_columns(
            pl.col("today").fill_null(0.0),
            pl.col("at_start").fill_null(0.0),
        )
    )
    return scored.with_columns(
        pl.when(pl.col("start_year").is_null())
        .then(None)
        .otherwise(pl.col("at_start"))
        .alias("at_start")
    )


def trial_drug_types(drugs: pl.DataFrame, drug_molecule: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, drug_type): the collapsed Open Targets drug type of each trial's drugs."""
    typed = drugs.join(
        drug_molecule.select(pl.col("id").alias("drugId"), "drugType"), on="drugId", how="left"
    )
    return (
        typed.group_by("nct_id")
        .agg(pl.col("drugType").alias("types"))
        .select(
            "nct_id",
            pl.col("types")
            .map_elements(collapse_drug_types, return_dtype=pl.String)
            .alias("drug_type"),
        )
    )


def trial_drug_names(reports: pl.DataFrame, ids: list[str]) -> pl.DataFrame:
    """(nct_id, drugId, source_name) for each drug listed for these trials, in list order."""
    return (
        reports.filter(pl.col("id").is_in(ids))
        .select(pl.col("id").alias("nct_id"), "drugs")
        .explode("drugs")
        .select(
            "nct_id",
            pl.col("drugs").struct.field("drugId").alias("drugId"),
            pl.col("drugs").struct.field("drugFromSource").alias("source_name"),
        )
        .drop_nulls("drugId")
    )


def normalise_name(text: str | None) -> str:
    """Lower-case letters and digits only, single-spaced: the form names are compared in."""
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def names_match(a: str, b: str) -> bool:
    """True when one normalised name is a whole-word phrase inside the other (3+ characters)."""
    if not a or not b:
        return False
    short, long = sorted((a, b), key=len)
    return len(short) >= 3 and f" {short} " in f" {long} "


def experimental_names(studies: Iterable[dict], ids: set[str]) -> dict[str, list[str]]:
    """NCT id -> normalised names of the registry interventions that stand for the treatment
    under test (`dataset.experimental_interventions`). Reads arms and interventions only."""
    from elman_exercise.dataset import experimental_interventions
    from elman_exercise.download import nct_id

    out: dict[str, list[str]] = {}
    for study in studies:
        key = nct_id(study).upper()
        if key not in ids:
            continue
        names = {
            normalise_name(n)
            for i in experimental_interventions(study)
            for n in [i.get("name"), *(i.get("otherNames") or [])]
        } - {""}
        out[key] = sorted(names)
    return out


def investigational_drugs(
    named: pl.DataFrame, drug_molecule: pl.DataFrame, experimental: dict[str, list[str]]
) -> pl.DataFrame:
    """(nct_id, drugId) for the drugs Open Targets lists that match an experimental-arm
    intervention name of the same trial (see the module docstring for the rule)."""
    molecule = dict(zip(drug_molecule["id"], drug_molecule["name"], strict=True))
    keep = []
    for nct, drug, source in named.select("nct_id", "drugId", "source_name").rows():
        candidates = {normalise_name(source), normalise_name(molecule.get(drug))} - {""}
        if any(names_match(c, e) for c in candidates for e in experimental.get(nct, [])):
            keep.append((nct, drug))
    return pl.DataFrame(keep, schema={"nct_id": pl.String, "drugId": pl.String}, orient="row")


def lead_drugs(named: pl.DataFrame, investigational: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, lead_drug): the first investigational drug in Open Targets' list order,
    or the first drug if none matched."""
    inv = set(investigational.rows())
    lead: dict[str, str] = {}
    first: dict[str, str] = {}
    for nct, drug in named.select("nct_id", "drugId").rows():
        first.setdefault(nct, drug)
        if (nct, drug) in inv:
            lead.setdefault(nct, drug)
    rows = [(n, lead.get(n, d)) for n, d in first.items()]
    return pl.DataFrame(rows, schema={"nct_id": pl.String, "lead_drug": pl.String}, orient="row")


def earlier_trial_counts(reports: pl.DataFrame, leads: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, lead_drug_earlier_trials): ClinicalTrials.gov trials in Open Targets' report
    (any phase, not only ours) that list the lead drug and started strictly earlier. Start
    dates and drug ids only. Null if the trial has no start date there."""
    listed = (
        reports.select(pl.col("id").alias("other"), "trialStartDate", "drugs")
        .explode("drugs")
        .select(
            "other",
            pl.col("trialStartDate").alias("other_start"),
            pl.col("drugs").struct.field("drugId").alias("lead_drug"),
        )
        .drop_nulls(["lead_drug", "other_start"])
        .unique()
    )
    starts = reports.select(pl.col("id").alias("nct_id"), pl.col("trialStartDate").alias("start"))
    keyed = leads.join(starts.unique("nct_id"), on="nct_id", how="left")
    counts = (
        keyed.join(listed, on="lead_drug")
        .filter(pl.col("other_start") < pl.col("start"))
        .group_by("nct_id")
        .agg(pl.col("other").n_unique().alias("n"))
    )
    return keyed.join(counts, on="nct_id", how="left").select(
        "nct_id",
        pl.when(pl.col("start").is_null())
        .then(None)
        .otherwise(pl.col("n").fill_null(0))
        .cast(pl.Int64)
        .alias("lead_drug_earlier_trials"),
    )


def top_targets(scored: pl.DataFrame, column: str, name: str) -> pl.DataFrame:
    """(nct_id, <name>): the target of each trial's highest-scoring pair, when that score is
    above zero (ties go to the smaller Ensembl id)."""
    return (
        scored.sort([column, "targetId"], descending=[True, False], nulls_last=True)
        .group_by("nct_id", maintain_order=True)
        .agg(pl.col(column).first().alias("_s"), pl.col("targetId").first().alias("_t"))
        .select("nct_id", pl.when(pl.col("_s") > 0).then(pl.col("_t")).alias(name))
    )


def check_genetic_only(associations: pl.DataFrame) -> None:
    """Fail unless every kept association row is from the genetic_association datatype."""
    if "aggregationValue" not in associations.columns:
        raise ValueError("The genetic file has no aggregationValue column: download it again.")
    other = associations.filter(pl.col("aggregationValue") != GENETIC)
    if other.height or associations["aggregationValue"].null_count():
        raise ValueError(
            f"{other.height} association rows are not {GENETIC} "
            f"(found {sorted(set(other['aggregationValue'].drop_nulls()))})."
        )


def build_table(
    trials: pl.DataFrame,
    reports: pl.DataFrame,
    drug_targets: pl.DataFrame,
    associations: pl.DataFrame,
    drug_molecule: pl.DataFrame,
    experimental: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """One row per trial. `trials` needs nct_id, split and start_year.

    `experimental` maps NCT id to normalised experimental-arm intervention names; without
    it no drug counts as investigational."""
    ids = trials["nct_id"].to_list()
    drugs = trial_drugs(reports, ids)
    diseases = trial_diseases(reports, ids)
    scored = pair_scores(trial_pairs(drugs, diseases, drug_targets), trials, associations)
    named = trial_drug_names(reports, ids)
    inv = investigational_drugs(named, drug_molecule, experimental or {})
    scored_inv = pair_scores(trial_pairs(inv, diseases, drug_targets), trials, associations)
    per_trial = scored.group_by("nct_id").agg(
        pl.col("targetId").n_unique().alias("n_targets"),
        pl.len().alias("n_pairs"),
        pl.col("today").max().alias("genetic_score_today"),
        pl.col("at_start").max().alias("genetic_score_at_start"),
        pl.col("targetId").unique().sort().alias("target_ids"),
        pl.col("targetId").filter(pl.col("today") > 0).unique().sort()
        .alias("supported_targets_today"),
        pl.col("targetId").filter(pl.col("at_start") > 0).unique().sort()
        .alias("supported_targets_at_start"),
    )  # fmt: skip
    per_inv = scored_inv.group_by("nct_id").agg(
        pl.len().alias("n_pairs_inv"),
        pl.col("today").max().alias("score_today_inv"),
        pl.col("at_start").max().alias("score_at_start_inv"),
    )
    leads = lead_drugs(named, inv)
    table = (
        trials.select("nct_id", "split")
        .join(per_trial, on="nct_id", how="left")
        .join(per_inv, on="nct_id", how="left")
        .join(top_targets(scored, "today", "top_target_today"), on="nct_id", how="left")
        .join(top_targets(scored, "at_start", "top_target_at_start"), on="nct_id", how="left")
        .join(leads, on="nct_id", how="left")
        .join(earlier_trial_counts(reports, leads), on="nct_id", how="left")
        .join(trial_drug_types(drugs, drug_molecule), on="nct_id", how="left")
        .with_columns(
            pl.col("n_pairs").is_not_null().alias("mapped"),
            pl.col("n_targets").fill_null(0).cast(pl.Int64),
            pl.col("n_pairs").fill_null(0).cast(pl.Int64),
            pl.col("drug_type").fill_null("unknown"),
        )
        .with_columns(
            (pl.col("genetic_score_today") > 0).alias("supported_today"),
            (pl.col("genetic_score_at_start") > 0).alias("supported_at_start"),
            pl.when(pl.col("mapped")).then(pl.col("n_targets") <= FEW_TARGETS).alias("few_targets"),
            pl.when(pl.col("mapped")).then(pl.col("n_pairs_inv").fill_null(0) > 0)
            .alias("investigational_mapped"),
            pl.when(pl.col("mapped")).then(pl.col("score_today_inv").fill_null(0.0) > 0)
            .alias("supported_today_investigational"),
            pl.when(pl.col("mapped") & pl.col("genetic_score_at_start").is_not_null())
            .then(pl.col("score_at_start_inv").fill_null(0.0) > 0)
            .alias("supported_at_start_investigational"),
        )
    )  # fmt: skip
    return table.select(
        "nct_id",
        "split",
        "mapped",
        "n_targets",
        "n_pairs",
        "genetic_score_today",
        "genetic_score_at_start",
        "supported_today",
        "supported_at_start",
        "drug_type",
        "few_targets",
        "investigational_mapped",
        "supported_today_investigational",
        "supported_at_start_investigational",
        "lead_drug",
        "lead_drug_earlier_trials",
        "target_ids",
        "supported_targets_today",
        "supported_targets_at_start",
        "top_target_today",
        "top_target_at_start",
    ).sort("nct_id")


# --- reading and writing --------------------------------------------------------------


def load_trials() -> pl.DataFrame:
    """nct_id, split and start_year for every trial. Feature columns only: no labels are read."""
    columns = ["nct_id", "split", "start_year"]
    frames = [
        pl.read_parquet(INTERIM / name, columns=columns)
        for name in ("dev.parquet", "test_features.parquet")
    ]
    return pl.concat(frames)


def build(out_dir: Path = OT_DIR, out_path: Path = OUT_PATH) -> pl.DataFrame:
    from elman_exercise.download import iter_studies
    from elman_exercise.opentargets import WANTED_COLUMNS

    reports = pl.read_parquet(
        out_dir / "clinical_report.parquet", columns=list(WANTED_COLUMNS["clinical_report"])
    )
    drug_targets = pl.read_parquet(out_dir / "clinical_target.parquet")
    molecules = pl.read_parquet(out_dir / "drug_molecule.parquet")
    associations = pl.read_parquet(out_dir / "genetic_association.parquet")
    check_genetic_only(associations)
    trials = load_trials()
    experimental = experimental_names(iter_studies(), set(trials["nct_id"].to_list()))
    table = build_table(trials, reports, drug_targets, associations, molecules, experimental)
    table.write_parquet(out_path)
    return table


# --- checking the history column -------------------------------------------------------


def check_timeseries(associations: pl.DataFrame, sample: int = 500, seed: int = 0) -> str:
    """Plain-language checks that the yearly history means what we assume it means."""
    rows = associations.sample(min(sample, associations.height), seed=seed)
    rows = rows.with_columns(
        pl.col("timeseries").list.eval(pl.element().struct.field("year")).alias("years")
    )
    n = rows.height
    last_equals_today = undated_equals_today = monotone = first_positive = 0
    first_years: list[int] = []
    last_years: list[int] = []
    for record in rows.iter_rows(named=True):
        today = record["associationScore"]
        dated = sorted(
            (e["year"], e["associationScore"])
            for e in record["timeseries"]
            if e["year"] is not None
        )
        undated = [e["associationScore"] for e in record["timeseries"] if e["year"] is None]
        scores = [s for _, s in dated]
        last_equals_today += bool(dated) and math.isclose(scores[-1], today, abs_tol=1e-9)
        undated_equals_today += bool(undated) and math.isclose(undated[0], today, abs_tol=1e-9)
        monotone += all(b >= a - 1e-9 for a, b in itertools.pairwise(scores))
        if dated:
            first_years.append(dated[0][0])
            last_years.append(dated[-1][0])
            first_positive += scores[0] > 0
    return "\n".join(
        [
            f"Timeseries check on {n} random genetic-association pairs:",
            f"  latest dated year equals today's score:      {last_equals_today}/{n}",
            f"  the undated entry equals today's score:      {undated_equals_today}/{n}",
            f"  scores never fall from one year to the next: {monotone}/{n}",
            f"  earliest year in history: {min(first_years)}; latest: {max(last_years)}",
            f"  pairs already positive in their earliest year: {first_positive}/{n}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-timeseries", action="store_true")
    args = parser.parse_args()
    if args.check_timeseries:
        print(check_timeseries(pl.read_parquet(OT_DIR / "genetic_association.parquet")))
        return
    table = build()
    print(f"wrote {OUT_PATH} ({table.height} trials)")
    print(table.group_by("split").agg(pl.col("mapped").mean().alias("mapped_share")).sort("split"))


if __name__ == "__main__":
    main()
