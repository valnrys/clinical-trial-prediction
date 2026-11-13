"""Turn the raw download into the benchmark tables.

Reads the raw studies only through `download.iter_studies()`, applies the
eligibility rules in a fixed order (counting what each one removes), splits by
primary completion date, and writes:

    data/interim/dev.parquet            development rows, with labels
    data/interim/test_features.parquet  test and late rows, NO labels
    data/heldout/test_labels.parquet    their labels (guarded; this module never reads it)
    trials.csv                          nct_id and split of every eligible trial, no labels;
                                        kept if it exists, unless --update-trial-list
    flow.md                             the eligibility flow and counts per split

    uv run python -m trial_benchmark.dataset [--update-trial-list]

Columns come in four groups, kept apart because the difference is the point
of the benchmark. `columns_in_group(name)` returns the names in a group, so a
baseline asks for "design" and cannot pick up the others by accident:

- design: things a sponsor could know when the trial was registered;
- results structure (`control_kind`, `n_arms`): read from the results section
  (which groups were compared), not from registration. Fine for slicing
  results; not to be used as model inputs;
- post-hoc (leaky) fields: edited or only known after the trial ended, so a
  model that uses them is reading the future;
- labels.

Even the design group is the record as it stands today (limit L2): sponsors
can edit modality, therapeutic area, phase, arm count and sponsor class.

The public return value of `build` and `make_tables` carries no test or late
label; those are written only to data/heldout/test_labels.parquet.

Missing values stay null. Nothing is imputed here.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from trial_benchmark import download
from trial_benchmark.labels import TrialLabel, label_trial

ROOT = Path(__file__).resolve().parents[2]

LAST_COMPLETION = date(2025, 12, 31)
TEST_START = date(2021, 1, 1)
LATE_START = date(2024, 1, 1)
DRUG_LIKE_TYPES = {"DRUG", "BIOLOGICAL", "GENETIC"}
FINISHED_STATUSES = {"COMPLETED", "TERMINATED"}

IDENTIFIER_COLUMNS = ("nct_id", "split", "primary_completion_date")
DESIGN_COLUMNS = (
    "phase",
    "sponsor_class",
    "allocation",
    "masking",
    "intervention_model",
    "n_registered_arms",
    "modality",
    "is_vaccine",
    "therapeutic_area",
    "start_year",
    "healthy_volunteers",
    "has_dmc",
)
# From the results section (which groups were compared), not from registration.
RESULTS_STRUCTURE_COLUMNS = ("control_kind", "n_arms")
LEAKY_COLUMNS = (
    "enrollment",
    "overall_status",
    "why_stopped_present",
    "duration_months",
    "n_locations",
    "results_lag_months",
)
LABEL_COLUMNS = ("any_arm", "all_arms", "highest_dose", "has_dose_ladder")

COLUMN_GROUPS = {
    "identifier": IDENTIFIER_COLUMNS,
    "design": DESIGN_COLUMNS,
    "results_structure": RESULTS_STRUCTURE_COLUMNS,
    "post_hoc": LEAKY_COLUMNS,
    "labels": LABEL_COLUMNS,
}


def columns_in_group(group: str) -> tuple[str, ...]:
    """The column names in one named group (the keys of COLUMN_GROUPS)."""
    if group not in COLUMN_GROUPS:
        raise ValueError(f"Unknown column group {group!r}; choose from {sorted(COLUMN_GROUPS)}.")
    return COLUMN_GROUPS[group]


SCHEMA = {
    "nct_id": pl.String,
    "split": pl.String,
    "primary_completion_date": pl.Date,
    "phase": pl.String,
    "sponsor_class": pl.String,
    "allocation": pl.String,
    "masking": pl.String,
    "intervention_model": pl.String,
    "n_registered_arms": pl.Int64,
    "modality": pl.String,
    "is_vaccine": pl.Boolean,
    "therapeutic_area": pl.String,
    "start_year": pl.Int64,
    "healthy_volunteers": pl.Boolean,
    "has_dmc": pl.Boolean,
    "control_kind": pl.String,
    "n_arms": pl.Int64,
    "enrollment": pl.Int64,
    "overall_status": pl.String,
    "why_stopped_present": pl.Boolean,
    "duration_months": pl.Float64,
    "n_locations": pl.Int64,
    "results_lag_months": pl.Float64,
    "any_arm": pl.Boolean,
    "all_arms": pl.Boolean,
    "highest_dose": pl.Boolean,
    "has_dose_ladder": pl.Boolean,
}
assert set(SCHEMA) == {
    *IDENTIFIER_COLUMNS,
    *DESIGN_COLUMNS,
    *RESULTS_STRUCTURE_COLUMNS,
    *LEAKY_COLUMNS,
    *LABEL_COLUMNS,
}

DAYS_PER_MONTH = 30.4375


# ---------------------------------------------------------------- reading fields


def _get(record: dict, *path: str):
    """Follow a path of keys through nested dicts; None if any step is missing."""
    for key in path:
        if not isinstance(record, dict):
            return None
        record = record.get(key)
    return record


def parse_date(text: str | None) -> date | None:
    """Read "YYYY-MM-DD" or "YYYY-MM". A month alone counts as the first of that month."""
    match = re.fullmatch(r"(\d{4})-(\d{2})(?:-(\d{2}))?", (text or "").strip())
    if not match:
        return None
    year, month, day = match.groups()
    try:
        return date(int(year), int(month), int(day or 1))
    except ValueError:
        return None


def _date_at(study: dict, struct: str) -> date | None:
    return parse_date(_get(study, "protocolSection", "statusModule", struct, "date"))


def primary_completion(study: dict) -> date | None:
    return _date_at(study, "primaryCompletionDateStruct")


def _months_between(start: date | None, end: date | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).days / DAYS_PER_MONTH, 2)


# ---------------------------------------------------------------- eligibility


def _interventions(study: dict) -> list[dict]:
    return _get(study, "protocolSection", "armsInterventionsModule", "interventions") or []


def has_drug_like_intervention(study: dict) -> bool:
    return any(item.get("type") in DRUG_LIKE_TYPES for item in _interventions(study))


def is_finished(study: dict) -> bool:
    return _get(study, "protocolSection", "statusModule", "overallStatus") in FINISHED_STATUSES


def completed_in_window(study: dict) -> bool:
    # Checked 2026-11-10: every eligible trial's primary completion date has type ACTUAL.
    pcd = primary_completion(study)
    return pcd is not None and pcd <= LAST_COMPLETION


# Each step is (description, test). A trial must pass them in this order; the
# flow table counts it against the first one it fails. The last two need the
# label, so they are applied in `first_failed_step`.
FLOW_STEPS = (
    ("Phase 2 or 3 interventional trials in the download", None),
    ("Has a drug, biological or genetic intervention", has_drug_like_intervention),
    ("Status completed or terminated", is_finished),
    ("Primary completion date present, readable and on or before 2025-12-31", completed_in_window),
    ("Has posted results", lambda study: bool(study.get("hasResults"))),
    ("Has a treatment-versus-control comparison on a primary outcome", "label"),
    ("Headline label (any arm) is determined", "determined"),
)


def first_failed_step(study: dict) -> tuple[int | None, TrialLabel | None]:
    """The index in FLOW_STEPS of the first rule the trial fails, or None if eligible.

    Also returns the label when it was worked out, so it is not computed twice.
    """
    label = None
    for index, (_, test) in enumerate(FLOW_STEPS):
        if test is None:
            continue
        if test == "label":
            label = label_trial(study)
            passed = label is not None
        elif test == "determined":
            passed = label is not None and label.any_arm is not None
        else:
            passed = test(study)
        if not passed:
            return index, label
    return None, label


def split_of(pcd: date) -> str:
    """dev before 2021; test 2021 to 2023; late 2024 to 2025."""
    if pcd < TEST_START:
        return "dev"
    return "test" if pcd < LATE_START else "late"


# ---------------------------------------------------------------- modality

# Real drug names are listed beside the stems so the intent is plain. First match wins.
# Names are cut into plain words first (hyphens split them), so "car t" is two words.
# The short stems need two letters before them, so "VEC-162" or a bare "cel" do not match.
_STEMS = (
    ("cell", r"\w{2,}cel\b|car t\b|stem cell|cell therapy"),  # tisagenlecleucel
    ("gene", r"\w{2,}(gene|vec)\b|aav|gene therapy|lentivir"),  # onasemnogene abeparvovec
    ("oligonucleotide", r"\w*(rsen|siran)\b|antisense|sirna"),  # nusinersen, inclisiran
    ("vaccine", r"vaccin"),
    ("antibody", r"\w+mab\b"),  # adalimumab
    ("fusion protein", r"\w+cept\b"),  # etanercept
)
_STEM_PATTERNS = tuple(
    (kind, re.compile(rf"\b(?:{stems})", re.IGNORECASE)) for kind, stems in _STEMS
)
# Words that end like a stem but are not biologics: ordinary words, and small-molecule
# brand names found by reading the stem matches on pre-2021 trials. Dropped from the
# name before matching. Short and by hand, so other lookalikes will get through (limit L7).
_NOT_DRUG_WORDS = frozenset(
    """
    excel parcel cancel intercept concept except accept precept worsen
    cellcept aricept gleevec glivec sprycel epicept benagene surgicel oxycel
    """.split()  # noqa: SIM905
)
_GENE_RELATED = re.compile(r"\bgene-related\b", re.IGNORECASE)
# Interventions that are not the treatment under test, dropped before a modality is chosen:
# placebo-like controls, and genetic tests (a "genotyping" typed GENETIC is not a gene therapy).
_NOT_A_TREATMENT = re.compile(
    r"\b(placebo|vehicle|sham|saline|dummy)\b|genotyp|genetic (test|screen)|pharmacogen",
    re.IGNORECASE,
)
# Known errors, left as they are: company codes with a stem-like tail (nothing marks a
# Merck vaccine code such as V114 as a vaccine); CD34+ cells typed BIOLOGICAL land in
# "biologic", not "advanced"; and anything a biologic or vaccine has only under a code name.
_BY_INTERVENTION_TYPE = {
    "GENETIC": "gene",
    "BIOLOGICAL": "other biological",
    "DRUG": "small molecule",
}
_GROUP_OF_KIND = {
    "cell": "advanced",
    "gene": "advanced",
    "oligonucleotide": "advanced",
    "antibody": "biologic",
    "fusion protein": "biologic",
    "other biological": "biologic",
    "small molecule": "small molecule",
}
COMPARATOR_ARM_TYPES = {
    "PLACEBO_COMPARATOR",
    "ACTIVE_COMPARATOR",
    "SHAM_COMPARATOR",
    "NO_INTERVENTION",
}


def intervention_kind(intervention: dict) -> str:
    """One intervention's kind, from its name and other names, then its registered type.

    The last resort is the registered type, so "small molecule" really means
    "a drug with no biologic stem in its name": peptides, hormones and
    company-coded biologics (e.g. "ABC-123") land there (limit L7).
    """
    names = [intervention.get("name") or "", *(intervention.get("otherNames") or [])]
    text = _GENE_RELATED.sub(" ", " ; ".join(names))
    words = [w for w in re.findall(r"\w+", text) if w.lower() not in _NOT_DRUG_WORDS]
    text = " ".join(words)
    for kind, pattern in _STEM_PATTERNS:
        if pattern.search(text):
            return kind
    return _BY_INTERVENTION_TYPE[intervention["type"]]


def experimental_interventions(study: dict) -> list[dict]:
    """The drug-like interventions that stand for the treatment under test.

    Prefer those given to an experimental arm and to no comparator arm. If
    there are none, those given to any experimental arm. If still none, all.
    Placebo-like interventions (placebo, vehicle, sham, saline, dummy) and genetic
    tests are never the treatment, so they are dropped first.
    """
    module = _get(study, "protocolSection", "armsInterventionsModule") or {}
    drugs = [
        i
        for i in module.get("interventions") or []
        if i.get("type") in DRUG_LIKE_TYPES and not _NOT_A_TREATMENT.search(i.get("name") or "")
    ]
    arms = module.get("armGroups") or []
    experimental = {a.get("label") for a in arms if a.get("type") == "EXPERIMENTAL"}
    comparator = {a.get("label") for a in arms if a.get("type") in COMPARATOR_ARM_TYPES}

    def labels(item: dict) -> set:
        return set(item.get("armGroupLabels") or [])

    only_experimental = [
        i for i in drugs if labels(i) & experimental and not labels(i) & comparator
    ]
    if only_experimental:
        return only_experimental
    any_experimental = [i for i in drugs if labels(i) & experimental]
    return any_experimental or drugs


def modality_of(study: dict) -> tuple[str | None, bool | None]:
    """(modality, is_vaccine) for a trial, using only its own record.

    modality is "small molecule", "biologic", "advanced" (cell, gene or
    oligonucleotide), "mixed" if the chosen interventions span more than one of
    those, or "vaccine" if every chosen intervention is a vaccine. A vaccine
    given alongside other drugs does not set the modality; the others do.
    """
    kinds = [intervention_kind(i) for i in experimental_interventions(study)]
    if not kinds:
        return None, None
    is_vaccine = "vaccine" in kinds
    groups = {_GROUP_OF_KIND[kind] for kind in kinds if kind != "vaccine"}
    if not groups:
        return "vaccine", True
    return (groups.pop() if len(groups) == 1 else "mixed"), is_vaccine


# ---------------------------------------------------------------- therapeutic area

# First match wins. Names are MeSH tree terms as they occur in conditionBrowseModule.
AREAS = (
    ("oncology", {"Neoplasms"}),
    ("infectious", {"Infections", "Virus Diseases", "Bacterial Infections and Mycoses"}),
    ("cardiovascular", {"Cardiovascular Diseases"}),
    ("neurology", {"Nervous System Diseases"}),
    ("psychiatry", {"Mental Disorders"}),
    ("metabolic_endocrine", {"Nutritional and Metabolic Diseases", "Endocrine System Diseases"}),
    ("respiratory", {"Respiratory Tract Diseases"}),
    ("immunology", {"Immune System Diseases"}),
    ("musculoskeletal", {"Musculoskeletal Diseases"}),
    ("digestive", {"Digestive System Diseases"}),
    ("dermatology", {"Skin and Connective Tissue Diseases"}),
)


def mesh_terms(study: dict) -> set[str]:
    module = _get(study, "derivedSection", "conditionBrowseModule") or {}
    return {
        item["term"]
        for key in ("ancestors", "meshes")
        for item in module.get(key) or []
        if item.get("term")
    }


def therapeutic_area(study: dict) -> str | None:
    """The first area in AREAS whose MeSH term is present; "other" if none; None with no MeSH terms."""
    terms = mesh_terms(study)
    if not terms:
        return None
    return next((area for area, names in AREAS if names & terms), "other")


# ---------------------------------------------------------------- the row


def phase_of(study: dict) -> str | None:
    """phase3 for PHASE3 and PHASE2/PHASE3; phase2 for PHASE2 and PHASE1/PHASE2."""
    phases = _get(study, "protocolSection", "designModule", "phases") or []
    if "PHASE3" in phases:
        return "phase3"
    return "phase2" if "PHASE2" in phases else None


def sponsor_class_of(study: dict) -> str | None:
    """INDUSTRY, NIH and OTHER as registered; government, network and the rest grouped."""
    kind = _get(study, "protocolSection", "sponsorCollaboratorsModule", "leadSponsor", "class")
    if kind is None:
        return None
    return kind if kind in {"INDUSTRY", "NIH", "OTHER"} else "OTHER_GOV_OR_NETWORK"


def _healthy_volunteers(study: dict) -> bool | None:
    value = _get(study, "protocolSection", "eligibilityModule", "healthyVolunteers")
    return value if isinstance(value, bool) else None


def make_row(study: dict, label: TrialLabel) -> dict:
    """One eligible trial as a flat row. Assumes it passed every eligibility rule."""
    protocol = study["protocolSection"]
    design = _get(protocol, "designModule", "designInfo") or {}
    pcd = primary_completion(study)
    start = _date_at(study, "startDateStruct")
    posted = _date_at(study, "resultsFirstPostDateStruct")
    arms = _get(protocol, "armsInterventionsModule", "armGroups")
    locations = _get(protocol, "contactsLocationsModule", "locations")
    modality, is_vaccine = modality_of(study)
    has_dmc = _get(protocol, "oversightModule", "oversightHasDmc")
    return {
        "nct_id": download.nct_id(study),
        "split": split_of(pcd),
        "primary_completion_date": pcd,
        "phase": phase_of(study),
        "sponsor_class": sponsor_class_of(study),
        "allocation": design.get("allocation"),
        "masking": _get(design, "maskingInfo", "masking"),
        "intervention_model": design.get("interventionModel"),
        "n_registered_arms": len(arms) if arms is not None else None,
        "control_kind": label.control_kind,
        "n_arms": label.n_arms,
        "modality": modality,
        "is_vaccine": is_vaccine,
        "therapeutic_area": therapeutic_area(study),
        "start_year": start.year if start else None,
        "healthy_volunteers": _healthy_volunteers(study),
        "has_dmc": has_dmc if isinstance(has_dmc, bool) else None,
        "enrollment": _get(protocol, "designModule", "enrollmentInfo", "count"),
        "overall_status": _get(protocol, "statusModule", "overallStatus"),
        "why_stopped_present": bool(_get(protocol, "statusModule", "whyStopped")),
        "duration_months": _months_between(start, pcd),
        "n_locations": len(locations) if locations is not None else None,
        "results_lag_months": _months_between(pcd, posted),
        "any_arm": label.any_arm,
        "all_arms": label.all_arms,
        "highest_dose": label.highest_dose,
        "has_dose_ladder": label.has_dose_ladder,
    }


# ---------------------------------------------------------------- the build


@dataclass
class Tables:
    """What callers get back. Test and late rows carry no labels (the label columns are null)."""

    flow: list[tuple[str, int, int]]  # (step, trials remaining, trials removed at this step)
    rows: pl.DataFrame  # one row per eligible trial; labels only on development rows


def _derive(studies: Iterable[dict]) -> tuple[list[tuple[str, int, int]], pl.DataFrame]:
    """Apply the eligibility rules in order, counting what each removes.

    Returns the flow and the full rows, labels included. Private: only
    `write_tables` may see the held-out labels.
    """
    removed_at: Counter[int] = Counter()
    total = 0
    rows = []
    for study in studies:
        total += 1
        failed, label = first_failed_step(study)
        if failed is None:
            rows.append(make_row(study, label))
        else:
            removed_at[failed] += 1
    flow = []
    remaining = total
    for index, (description, _) in enumerate(FLOW_STEPS):
        remaining -= removed_at[index]
        flow.append((description, remaining, removed_at[index]))
    return flow, pl.DataFrame(rows, schema=SCHEMA).sort("nct_id")


def without_held_out_labels(rows: pl.DataFrame) -> pl.DataFrame:
    """The same rows with every label set to null outside the development split."""
    return rows.with_columns(
        pl.when(pl.col("split") == "dev").then(pl.col(c)).otherwise(None).alias(c)
        for c in LABEL_COLUMNS
    )


def make_tables(studies: Iterable[dict]) -> Tables:
    """Eligibility flow and rows, with test and late labels removed."""
    flow, rows = _derive(studies)
    return Tables(flow=flow, rows=without_held_out_labels(rows))


def split_counts(rows: pl.DataFrame) -> dict[str, int]:
    counts = dict(rows.group_by("split").len().iter_rows())
    return {name: counts.get(name, 0) for name in ("dev", "test", "late")}


def flow_markdown(flow: list[tuple[str, int, int]], counts: dict[str, int]) -> str:
    lines = ["# Eligibility flow", "", "| Step | Trials remaining | Removed |", "|---|---:|---:|"]
    lines += [f"| {step} | {left:,} | {removed:,} |" for step, left, removed in flow]
    lines += ["", "# Eligible trials per split", "", "| Split | Trials |", "|---|---:|"]
    lines += [f"| {name} | {n:,} |" for name, n in counts.items()]
    lines += [f"| test + late (extended test set) | {counts['test'] + counts['late']:,} |", ""]
    return "\n".join(lines)


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def compare_trial_list(old: pl.DataFrame, new: pl.DataFrame) -> dict[str, int]:
    """How a newly derived (nct_id, split) list differs from the committed one."""
    before = dict(old.select("nct_id", "split").iter_rows())
    after = dict(new.select("nct_id", "split").iter_rows())
    return {
        "added": len(after.keys() - before.keys()),
        "removed": len(before.keys() - after.keys()),
        "moved": sum(1 for k in before.keys() & after.keys() if before[k] != after[k]),
    }


def write_trial_list(
    rows: pl.DataFrame, root: Path = ROOT, update: bool = False
) -> dict[str, int] | None:
    """Write trials.csv, the exact benchmark membership, unless one already exists.

    An existing file is replaced only when `update` is true. Returns how the new
    list differs from the existing one (None when there was no file).
    """
    path = root / "trials.csv"
    new = rows.select("nct_id", "split").sort("nct_id")
    difference = compare_trial_list(pl.read_csv(path), new) if path.exists() else None
    if difference is None or update:
        new.write_csv(path)
    return difference


def write_tables(flow: list[tuple[str, int, int]], rows: pl.DataFrame, root: Path = ROOT) -> None:
    """Write the parquet files and flow.md. `rows` has labels; held-out ones go only to heldout/."""
    not_labels = [c for c in rows.columns if c not in LABEL_COLUMNS]
    held_out = rows.filter(pl.col("split") != "dev")
    _write_parquet(rows.filter(pl.col("split") == "dev"), root / "data/interim/dev.parquet")
    _write_parquet(held_out.select(not_labels), root / "data/interim/test_features.parquet")
    _write_parquet(
        held_out.select("nct_id", "split", *LABEL_COLUMNS),
        root / "data/heldout/test_labels.parquet",
    )
    (root / "flow.md").write_text(flow_markdown(flow, split_counts(rows)))


def build(
    studies: Iterator[dict] | None = None, root: Path = ROOT, update_trial_list: bool = False
) -> Tables:
    """Build and write every table, and print how trials.csv would change.

    trials.csv is the benchmark's membership, so an existing one is kept unless
    `update_trial_list` is true. Returns the flow and rows with test and late
    labels removed.
    """
    flow, rows = _derive(download.iter_studies() if studies is None else studies)
    write_tables(flow, rows, root)
    difference = write_trial_list(rows, root, update_trial_list)
    if difference is None:
        print("trials.csv written (there was none).")
    else:
        verb = "rewritten" if update_trial_list else "KEPT (pass --update-trial-list to rewrite)"
        print(
            f"trials.csv {verb}. Newly derived list: {difference['added']} added, "
            f"{difference['removed']} removed, {difference['moved']} moved split."
        )
    return Tables(flow=flow, rows=without_held_out_labels(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update-trial-list",
        action="store_true",
        help="overwrite trials.csv with the newly derived list",
    )
    args = parser.parse_args()
    tables = build(update_trial_list=args.update_trial_list)
    print(flow_markdown(tables.flow, split_counts(tables.rows)))


if __name__ == "__main__":
    main()
