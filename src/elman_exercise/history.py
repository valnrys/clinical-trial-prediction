"""Programme history: what earlier trials of the same drug had shown before a trial began.

    uv run python -m elman_exercise.history    # -> data/interim/history.parquet

The question: if a Phase 2 of this drug, in this disease, had already reported
when the trial started, does that tell us how the trial will go? On development
data, Phase 3 trials that follow a Phase 2 that succeeded win far more often than
those that follow one that failed. The mere existence of an earlier Phase 2 adds
nothing. So the features are built to carry the earlier result, and nothing else.

The two timing rules (the whole point)
--------------------------------------
- An earlier trial U counts for trial T only if U STARTED BEFORE T started.
- U's label counts as KNOWN only if U's results were FIRST POSTED BEFORE T started.
  That is what an outsider could have read. We do not use "U's primary completion
  came before T started": that assumes knowledge of a result before it is public,
  which an outsider lacks. (The scouting tried it; it is deliberately not built.)

Both are strict and cautious. The same day does not count. A date given as year and
month stands for every day of that month, and "before" holds only if it holds for
every one of those days (on both sides). So a U starting "2015-03" is not before a T
starting "2015-03-15". Trials with no start date get null features.

Labels come ONLY from data/interim/dev.parquet (`any_arm`). A test or late Phase 2
is never labelled here. So test and late trials have thinner history by
construction: their features rest on development-period Phase 2 results only, and
a Phase 2 from the test period never supplies a label. `report()` prints how many
trials would have needed a sealed result (a count only).

A trial registered after the fact (entered once it was already under way) still counts as
earlier if its start date is before T's: the rule uses start dates, not registration dates.

The join, in plain words
------------------------
Same drug: the two trials share a drug key. A key is either an Open Targets drug id
that Open Targets matched to an experimental-arm intervention of the trial (the rule in
genetics.py), or the normalised name of an experimental-arm intervention (lower
case, doses, salts and dosage words removed). For trial T, a key is dropped if more
than 50 trials that started before T share it: it is then a background drug or a
generic word. Only trials that started before T are counted, because a drug reaches
50 trials partly by succeeding, and counting later trials would use the future.

Same programme: same drug, and the two trials share a condition: a MeSH condition
term from the registry, or an Open Targets disease id.

An "earlier Phase 2" is PHASE2 or PHASE1/PHASE2. A PHASE2/PHASE3 trial is a Phase 3
here and is never counted as an earlier Phase 2.

Combination trials are left out of the label join on both sides, because a
combination is not the same drug as one of its parts. They still count in the
label-free counts. A trial is a combination if an intervention name joins drugs
("A + B", "A/B", "A and B", "A with B"; doses such as "10 mg/mL" and "taken with
food" do not count), or if one experimental arm gets more than one distinct drug.

The stricter join also needs the same lead sponsor. Hand-reading
(results/history_pairs_read.csv) found the plain join about 80 to 90 percent right;
the errors are a different disease that shares a broad condition term, and a
combination matched to one of its parts.

A limit that remains
--------------------
Every date comparison is strict, but WHO is who is not dated. Drug and disease
identities come from today's Open Targets tables and today's MeSH terms. A company
code used in Phase 2 links to the later generic name partly because the drug
progressed far enough to get one. So the link itself carries some hindsight, and the
features are called DATED only in the sense of the timing rules above.

Deliberately NOT built: anything about the drug's target. Open Targets' target
mapping is today's, so it would carry hindsight.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from datetime import date

import polars as pl

from elman_exercise import dataset, genetics
from elman_exercise.dataset import ROOT
from elman_exercise.dating import DATED

TABLE = "history"
OUT_PATH = ROOT / "data" / "interim" / "history.parquet"
TRIALS_PATH = ROOT / "trials.csv"
DEV_PATH = ROOT / "data" / "interim" / "dev.parquet"
MAX_EARLIER_PER_KEY = 50

FEATURES = dict.fromkeys(
    (
        "prior_phase2_labelled",
        "prior_phase2_succeeded",
        "prior_phase2_same_sponsor",
        "prior_phase2_same_sponsor_succeeded",
        "n_earlier_trials_same_drug",
        "n_earlier_trials_same_programme",
    ),
    DATED,
)

# A dose or strength: "10 mg", "1.5%", "5 mg/kg", "10 mg/mL", or a bare "mg/kg".
_PER = r"(?:\s*/\s*(?:kg|ml|l|dl|m2|m²|g|mg|day|d|h|hr|min|wk|week)\b)"
_UNIT = r"(?:mg|mcg|ug|µg|g|kg|ml|l|iu|u|units?)"
_DOSE = re.compile(
    rf"\b\d+(?:[.,]\d+)?\s*{_UNIT}{_PER}*\b|\b\d+(?:[.,]\d+)?\s*%|\b{_UNIT}{_PER}+",
    re.IGNORECASE,
)
_JOINERS = re.compile(r"\+|/|\bplus\b|\band\b|\bwith\b", re.IGNORECASE)
_COMBINATION_WORDS = re.compile(r"combination|fixed.dose", re.IGNORECASE)
_NOT_A_DRUG = {"food", "meal", "meals", "water", "breakfast", "placebo", "vehicle", "saline"}
# Salts, dosage forms and filler words that do not change which drug it is.
_WORDS_DROPPED = {
    "hydrochloride", "hcl", "sodium", "potassium", "calcium", "sulfate", "sulphate", "phosphate",
    "acetate", "tablet", "tablets", "capsule", "capsules", "injection", "oral", "solution",
    "suspension", "cream", "gel", "extended", "qd", "bid", "tid", "daily", "release", "dose",
    "low", "high", "and", "plus", "with", "the", "of", "for",
    # Delivery devices and their brand types: "Ellipta device" is not a drug.
    "device", "inhaler", "pen", "autoinjector", "syringe", "prefilled", "ellipta", "diskus",
    "respimat", "turbuhaler", "handihaler", "breezhaler", "pmdi",
}  # fmt: skip


# ------------------------------------------------------------ dates


def date_range(text: str | None) -> tuple[date | None, date | None]:
    """The first and last day a registry date could stand for: a month alone is the whole month."""
    first = dataset.parse_date(text)
    if first is None:
        return None, None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()):
        return first, first
    return first, first.replace(day=calendar.monthrange(first.year, first.month)[1])


# ------------------------------------------------------------ names


def clean_name(text: str | None) -> str:
    """Brackets, doses, salts and dosage words removed, in `genetics.normalise_name` form.

    History needs its own step on top of `genetics.normalise_name`: that function only
    lower-cases and strips punctuation, so "Ruxolitinib 1.5% Cream QD" would not match
    "Ruxolitinib". Here the extra words go before the shared normalisation."""
    text = _DOSE.sub(" ", re.sub(r"\([^)]*\)", " ", (text or "").lower()))
    words = genetics.normalise_name(text).split()
    return " ".join(w for w in words if w not in _WORDS_DROPPED)


def drug_name(text: str | None) -> str:
    """A drug name reduced to what identifies it; empty if nothing useful is left."""
    name = clean_name(text)
    return name if len(name) >= 4 and not name.isdigit() else ""


def names_a_combination(name: str | None) -> bool:
    """True if one intervention name joins two drugs ("A + B", "A/B", "A and B", "A with B").

    Doses are removed first, so "10 mg/mL" and "5 mg/kg" are not joins, and a side that is
    not a drug ("taken with food") does not count."""
    name = name or ""
    if _COMBINATION_WORDS.search(name):
        return True
    sides = (
        clean_name(side) for side in _JOINERS.split(re.sub(r"\([^)]*\)", " ", _DOSE.sub(" ", name)))
    )
    drugs = [s for s in sides if s and not s.isdigit() and not _NOT_A_DRUG & set(s.split())]
    return len(drugs) > 1


def is_combination(interventions: list[dict], experimental_arms: Iterable[str] = ()) -> bool:
    """True if a name joins drugs, or one experimental arm gets more than one distinct drug.

    `interventions` are the treatment-under-test interventions of a trial (placebo and
    background drugs already removed, `dataset.experimental_interventions`), each with a
    "name" and the "armGroupLabels" it is given in."""
    if any(names_a_combination(i.get("name")) for i in interventions):
        return True
    for arm in experimental_arms:
        given = {
            clean_name(i.get("name"))
            for i in interventions
            if arm in (i.get("armGroupLabels") or [])
        }
        if len(given - {""}) > 1:
            return True
    return False


# ------------------------------------------------------------ pairs


def _keys(registry: pl.DataFrame, column: str) -> pl.DataFrame:
    """(nct_id, key), one row per pair."""
    return (
        registry.select("nct_id", pl.col(column).alias("key")).explode("key").drop_nulls().unique()
    )


def _earlier_shared(
    registry: pl.DataFrame, column: str, trial_ids: list[str], cap: int | None
) -> pl.DataFrame:
    """(t, u, t_start): trial t (from trial_ids) and a trial u that shares a key with it and
    started strictly before it. A key shared with more than `cap` such earlier trials is
    dropped for t, which counts only what t could have seen."""
    keys = _keys(registry, column)
    t_side = (
        keys.filter(pl.col("nct_id").is_in(trial_ids))
        .join(registry.select("nct_id", pl.col("start").alias("t_start")), on="nct_id")
        .rename({"nct_id": "t"})
    )
    u_side = keys.join(
        registry.select("nct_id", pl.col("start_end").alias("u_end")), on="nct_id"
    ).rename({"nct_id": "u"})
    pairs = (
        t_side.join(u_side, on="key")
        .filter(pl.col("t") != pl.col("u"))
        .filter(pl.col("u_end") < pl.col("t_start"))  # every day of u's start is before t's
    )
    if cap is not None:
        pairs = pairs.filter(pl.len().over("t", "key") <= cap)
    return pairs.select("t", "u", "t_start").unique()


def earlier_pairs(
    registry: pl.DataFrame, trial_ids: list[str], max_earlier: int | None = MAX_EARLIER_PER_KEY
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(drug, programme): (t, u, t_start) pairs where u started strictly before t, for the
    same drug, and for the same drug and an overlapping condition."""
    drug = _earlier_shared(registry, "drug_keys", trial_ids, max_earlier)
    condition = _earlier_shared(registry, "condition_keys", trial_ids, None)
    return drug, drug.join(condition.select("t", "u"), on=["t", "u"])


def public_phase2_pairs(programme: pl.DataFrame, registry: pl.DataFrame) -> pl.DataFrame:
    """(t, u, t_start, t_sponsor, u_sponsor): earlier Phase 2 trials u of t's programme whose
    results were first posted before t started, with neither trial a combination."""
    u = registry.select(
        pl.col("nct_id").alias("u"),
        "is_phase2",
        pl.col("results_end").alias("u_results_end"),
        pl.col("sponsor").alias("u_sponsor"),
        pl.col("combination").alias("u_combination"),
    )
    t = registry.select(
        pl.col("nct_id").alias("t"),
        pl.col("sponsor").alias("t_sponsor"),
        pl.col("combination").alias("t_combination"),
    )
    return (
        programme.join(u, on="u")
        .join(t, on="t")
        .filter(pl.col("is_phase2") & (pl.col("u_results_end") < pl.col("t_start")))
        .filter(~pl.col("u_combination") & ~pl.col("t_combination"))
        .select("t", "u", "t_start", "t_sponsor", "u_sponsor")
    )


def labelled_phase2_pairs(
    registry: pl.DataFrame,
    dev_labels: pl.DataFrame,
    trial_ids: list[str],
    max_earlier: int | None = MAX_EARLIER_PER_KEY,
) -> pl.DataFrame:
    """(t, u, u_succeeded, same_sponsor) for public earlier Phase 2 trials u whose label
    comes from development data only."""
    _, programme = earlier_pairs(registry, trial_ids, max_earlier)
    dev = dev_labels.filter(pl.col("split") == "dev").select(
        pl.col("nct_id").alias("u"), pl.col("any_arm").alias("u_succeeded")
    )
    return (
        public_phase2_pairs(programme, registry)
        .join(dev, on="u")
        .with_columns(
            (pl.col("u_sponsor") == pl.col("t_sponsor")).fill_null(False).alias("same_sponsor")
        )
        .select("t", "u", "u_succeeded", "same_sponsor")
    )


# ------------------------------------------------------------ features


def build_features(
    registry: pl.DataFrame,
    dev_labels: pl.DataFrame,
    trials: pl.DataFrame,
    max_earlier: int | None = MAX_EARLIER_PER_KEY,
) -> pl.DataFrame:
    """One row per trial in `trials` (nct_id, split): the FEATURES.

    registry: nct_id, start, start_end, results_end, sponsor, is_phase2, combination,
        drug_keys, condition_keys for every study in the registry (earlier trials need
        not be in the benchmark). `start` and `start_end` are the first and last day the
        start date could mean; `results_end` is the last day the first-posted date could mean.
    dev_labels: nct_id, split, any_arm. Rows whose split is not "dev" are ignored.
    """
    ids = trials["nct_id"].to_list()
    drug, programme = earlier_pairs(registry, ids, max_earlier)
    labelled = labelled_phase2_pairs(registry, dev_labels, ids, max_earlier)

    def count(pairs: pl.DataFrame, name: str) -> pl.DataFrame:
        return pairs.group_by("t").agg(pl.col("u").n_unique().alias(name))

    def verdict(pairs: pl.DataFrame, prefix: str) -> pl.DataFrame:
        return pairs.group_by("t").agg(
            pl.col("u_succeeded").any().alias(f"{prefix}_succeeded"),
            pl.lit(True).alias(f"{prefix}_labelled"),
        )

    out = trials.select("nct_id", "split").join(
        registry.select("nct_id", "start"), on="nct_id", how="left"
    )
    for table in (
        count(drug, "n_earlier_trials_same_drug"),
        count(programme, "n_earlier_trials_same_programme"),
        verdict(labelled, "main"),
        verdict(labelled.filter(pl.col("same_sponsor")), "strict"),
    ):
        out = out.join(table.rename({"t": "nct_id"}), on="nct_id", how="left")
    known = pl.col("start").is_not_null()
    return out.select(
        "nct_id",
        "split",
        pl.when(known)
        .then(pl.col("main_labelled").fill_null(False))
        .alias("prior_phase2_labelled"),
        pl.col("main_succeeded").alias("prior_phase2_succeeded"),
        pl.when(known)
        .then(pl.col("strict_labelled").fill_null(False))
        .alias("prior_phase2_same_sponsor"),
        pl.col("strict_succeeded").alias("prior_phase2_same_sponsor_succeeded"),
        pl.when(known)
        .then(pl.col("n_earlier_trials_same_drug").fill_null(0))
        .alias("n_earlier_trials_same_drug"),
        pl.when(known)
        .then(pl.col("n_earlier_trials_same_programme").fill_null(0))
        .alias("n_earlier_trials_same_programme"),
    )


# ------------------------------------------------------------ reading the registry


def read_registry() -> pl.DataFrame:
    """One pass over every downloaded study, plus the Open Targets drug and disease ids.

    Reads dates, sponsor, phase, interventions and conditions. The results section is
    only checked for its first-posted date, never for outcomes."""
    from elman_exercise.download import iter_studies, nct_id
    from elman_exercise.opentargets import OT_DIR, WANTED_COLUMNS

    get = dataset._get
    rows = []
    for study in iter_studies():
        interventions = dataset.experimental_interventions(study)
        arms = get(study, "protocolSection", "armsInterventionsModule", "armGroups") or []
        status = ("protocolSection", "statusModule")
        start, start_end = date_range(get(study, *status, "startDateStruct", "date"))
        phases = get(study, "protocolSection", "designModule", "phases") or []
        sponsor = get(study, "protocolSection", "sponsorCollaboratorsModule", "leadSponsor", "name")
        rows.append(
            {
                "nct_id": nct_id(study).upper(),
                "start": start,
                "start_end": start_end,
                "results_end": date_range(
                    get(study, *status, "resultsFirstPostDateStruct", "date")
                )[1],
                "sponsor": (sponsor or "").lower().strip() or None,
                "is_phase2": "PHASE2" in phases and "PHASE3" not in phases,
                "combination": is_combination(
                    interventions, [a["label"] for a in arms if a.get("type") == "EXPERIMENTAL"]
                ),
                "names": [
                    n for i in interventions for n in [i.get("name"), *(i.get("otherNames") or [])]
                ],
                "mesh": [
                    m["id"]
                    for m in get(study, "derivedSection", "conditionBrowseModule", "meshes") or []
                    if m.get("id")
                ],
            }
        )
    registry = pl.DataFrame(rows, infer_schema_length=None)
    reports = pl.read_parquet(
        OT_DIR / "clinical_report.parquet", columns=list(WANTED_COLUMNS["clinical_report"])
    ).with_columns(pl.col("id").str.to_uppercase())
    molecules = pl.read_parquet(OT_DIR / "drug_molecule.parquet")
    return with_keys(registry, reports, molecules)


def with_keys(
    registry: pl.DataFrame, reports: pl.DataFrame, molecules: pl.DataFrame
) -> pl.DataFrame:
    """Turn names and Open Targets ids into the drug_keys and condition_keys columns."""
    ids = registry["nct_id"].to_list()
    experimental = {
        nct: sorted({genetics.normalise_name(n) for n in names} - {""})
        for nct, names in registry.select("nct_id", "names").rows()
    }
    named = genetics.trial_drug_names(reports, ids)
    ot_drugs = genetics.investigational_drugs(named, molecules, experimental)
    ot_diseases = genetics.trial_diseases(reports, ids)

    def grouped(table: pl.DataFrame, column: str, name: str) -> pl.DataFrame:
        return table.group_by("nct_id").agg(("ot:" + pl.col(column)).alias(name))

    return (
        registry.join(grouped(ot_drugs, "drugId", "ot_drug"), on="nct_id", how="left")
        .join(grouped(ot_diseases, "diseaseId", "ot_disease"), on="nct_id", how="left")
        .with_columns(
            pl.struct("names", "ot_drug")
            .map_elements(
                lambda r: (
                    sorted({"name:" + drug_name(n) for n in r["names"]} - {"name:"})
                    + (r["ot_drug"] or [])
                ),
                return_dtype=pl.List(pl.String),
            )
            .alias("drug_keys"),
            pl.struct("mesh", "ot_disease")
            .map_elements(
                lambda r: ["mesh:" + m for m in r["mesh"]] + (r["ot_disease"] or []),
                return_dtype=pl.List(pl.String),
            )
            .alias("condition_keys"),
        )
        .select(
            "nct_id",
            "start",
            "start_end",
            "results_end",
            "sponsor",
            "is_phase2",
            "combination",
            "drug_keys",
            "condition_keys",
        )
    )


# ------------------------------------------------------------ diagnostics (not used by the features)


def sealed_history_trials(
    registry: pl.DataFrame, trials: pl.DataFrame, max_earlier: int | None = MAX_EARLIER_PER_KEY
) -> list[str]:
    """Ids of trials whose same-programme history includes a usable earlier Phase 2 from the
    sealed test or late splits (the same filter as the labelled pairs). Their labels would be
    needed to complete the feature, and are not used. Ids only; no outcomes."""
    _, programme = earlier_pairs(registry, trials["nct_id"].to_list(), max_earlier)
    sealed = trials.filter(pl.col("split").is_in(["test", "late"])).select(
        pl.col("nct_id").alias("u")
    )
    pairs = public_phase2_pairs(programme, registry).join(sealed, on="u")
    return sorted(pairs["t"].unique().to_list())


def report(
    table: pl.DataFrame, dev: pl.DataFrame, registry: pl.DataFrame, trials: pl.DataFrame
) -> None:
    """Coverage, the dev Phase 3 check (success by whether an earlier Phase 2 succeeded), and
    how many trials would have needed a sealed label."""
    for split in ("dev", "test", "late"):
        part = table.filter(pl.col("split") == split)
        print(
            f"{split:<5} {part.height:>5} trials, {part['prior_phase2_labelled'].sum():>4} with a labelled Phase 2"
        )
    phase3 = table.join(
        dev.filter(pl.col("split") == "dev").select("nct_id", "phase", "any_arm"), on="nct_id"
    ).filter(pl.col("phase") == "phase3")
    for value in (True, False, None):
        group = phase3.filter(
            pl.col("prior_phase2_succeeded").is_null()
            if value is None
            else pl.col("prior_phase2_succeeded") == value
        )
        print(
            f"Dev Phase 3, prior_phase2_succeeded={value}: {group.height} trials, success {group['any_arm'].mean():.1%}"
        )
    print(
        f"Trials whose history would need a sealed label (left unused): {len(sealed_history_trials(registry, trials))}"
    )


def main() -> None:
    trials = pl.read_csv(TRIALS_PATH)
    dev = pl.read_parquet(DEV_PATH, columns=["nct_id", "split", "phase", "any_arm"])
    registry = read_registry()
    table = build_features(registry, dev.select("nct_id", "split", "any_arm"), trials)
    table.write_parquet(OUT_PATH)
    report(table, dev, registry, trials)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}: {table.shape}")


if __name__ == "__main__":
    main()
