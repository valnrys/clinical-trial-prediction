"""Is a trial about a rare disease? Open Targets disease ids joined to Orphanet prevalence.

    uv run python -m elman_exercise.rare_disease    # downloads and parses; ~16 MB + ~6 MB

Open Targets lists the diseases each ClinicalTrials.gov trial studies, and many of
those diseases carry an Orphanet code. Orphanet's public prevalence file (Orphadata
product 9, https://www.orphadata.com/data/xml/en_product9_prev.xml, licence CC-BY 4.0,
(c) Orphanet) gives each rare disease a prevalence class, such as "1-9 / 100 000".

The rules, all fixed here:

- A disease with several Orphanet entries gets ONE class. Entries are ranked first by
  what they measure (point prevalence, then prevalence at birth or lifetime, then annual
  incidence), then by where they apply (worldwide, then Europe, then the United States,
  then anywhere). Entries with no usable class ("Unknown", "Not yet documented") are
  dropped first. Still tied: the more common class wins, the cautious choice for a
  "rare" label. The measure comes first because the rare cut-off (1-5 per 10,000) is a
  prevalence threshold, and an annual incidence is a different quantity. An earlier
  version put geography first, so a worldwide incidence could beat a European
  prevalence; `measure_first=False` still gives that rule, for the robustness table.
- A disease is **rare** if that class is 1-5 per 10,000 or lower. Open Targets may map
  one disease to several Orphanet codes; the most common of their classes is used.
- A disease with an Orphanet entry but no usable class is **unclear**. A disease with no
  Orphanet entry at all counts as not rare: Orphanet covers rare diseases, so absence
  is read as "common", though it may also mean "not catalogued".
- A trial is **rare** if any of its diseases is rare. Its band is the class of its
  rarest rare disease. It is **unclear** if every one of
  its diseases is unclear. Otherwise it is **not rare**. A trial Open Targets lists no
  disease for is **unmapped**.

There is no public numeric prevalence for common diseases to join on, so "not rare"
means "not in Orphanet's rare classes", not "known to be common".

Downloads land in data/raw/orphanet/ (git-ignored with the rest of data/raw/):
`en_product9_prev.xml`, the slim `prevalence.parquet`, and `manifest.json` (source, licence,
date, sizes and checksum). Open Targets outcome-bearing columns are never read: the
disease table is read for two columns, id and external references.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from elman_exercise import opentargets
from elman_exercise.dataset import ROOT

ORPHANET_URL = "https://www.orphadata.com/data/xml/en_product9_prev.xml"
LICENCE = "CC-BY-4.0 (Orphanet, https://www.orphadata.com)"
ORPHANET_DIR = ROOT / "data" / "raw" / "orphanet"

# Prevalence classes from rarest to commonest. Rare means this rank or lower.
CLASS_RANK = {
    "<1 / 1 000 000": 0,
    "1-9 / 1 000 000": 1,
    "1-9 / 100 000": 2,
    "1-5 / 10 000": 3,
    "6-9 / 10 000": 4,
    ">1 / 1000": 5,
}
RARE_MAX_RANK = 3
BAND_NAMES = {
    0: "under 1 per million",
    1: "1 to 9 per million",
    2: "1 to 9 per 100,000",
    3: "1 to 5 per 10,000",
}
# Higher is preferred. Anything not listed scores 0.
GEOGRAPHY_PREFERENCE = {"Worldwide": 3, "Europe": 2, "United States": 1}
TYPE_PREFERENCE = {
    "Point prevalence": 3,
    "Lifetime Prevalence": 2,
    "Prevalence at birth": 2,
    "Annual incidence": 1,
}

RARE, NOT_RARE, UNCLEAR, UNMAPPED = "rare", "not rare", "unclear", "unmapped"
GROUPS = (RARE, NOT_RARE, UNCLEAR, UNMAPPED)

PREVALENCE_SCHEMA = {
    "orpha_code": pl.String,
    "prevalence_type": pl.String,
    "geography": pl.String,
    "prevalence_class": pl.String,
}


# ---------------------------------------------------------------- Orphanet


def parse_prevalence(xml_path: Path) -> pl.DataFrame:
    """Every prevalence entry in the Orphadata file: code, type, geography and class (or null)."""
    rows = []

    def name_of(entry: ET.Element, tag: str) -> str | None:
        node = entry.find(tag)
        return node.findtext("Name") if node is not None else None

    for _, element in ET.iterparse(xml_path, events=("end",)):
        if element.tag != "Disorder":
            continue
        code = element.findtext("OrphaCode")
        for entry in element.iter("Prevalence"):
            rows.append(
                {
                    "orpha_code": code,
                    "prevalence_type": name_of(entry, "PrevalenceType"),
                    "geography": name_of(entry, "PrevalenceGeographic"),
                    "prevalence_class": name_of(entry, "PrevalenceClass"),
                }
            )
        element.clear()
    return pl.DataFrame(rows, schema=PREVALENCE_SCHEMA)


def one_class_per_disease(prevalence: pl.DataFrame, measure_first: bool = True) -> pl.DataFrame:
    """One (orpha_code, class_rank) per Orphanet code that has a usable class.

    See the module docstring for the order of preference. `measure_first=False` ranks
    geography above the kind of measure (the earlier rule).
    """
    order = ["what", "where", "class_rank"] if measure_first else ["where", "what", "class_rank"]
    usable = prevalence.filter(pl.col("prevalence_class").is_in(list(CLASS_RANK)))
    return (
        usable.with_columns(
            class_rank=pl.col("prevalence_class").replace_strict(CLASS_RANK, return_dtype=pl.Int64),
            where=pl.col("geography").replace_strict(
                GEOGRAPHY_PREFERENCE, default=0, return_dtype=pl.Int64
            ),
            what=pl.col("prevalence_type").replace_strict(
                TYPE_PREFERENCE, default=0, return_dtype=pl.Int64
            ),
        )
        .sort(order, descending=True)
        .group_by("orpha_code", maintain_order=True)
        .first()
        .select("orpha_code", "class_rank")
    )


def download_orphanet(out_dir: Path = ORPHANET_DIR, url: str = ORPHANET_URL) -> Path:
    """Fetch the XML (resumable), parse it to prevalence.parquet, and write a manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / Path(url).name
    opentargets.download(url, xml_path)
    prevalence = parse_prevalence(xml_path)
    prevalence.write_parquet(out_dir / "prevalence.parquet")
    manifest = {
        "source": url,
        "licence": LICENCE,
        "retrieved": datetime.now(UTC).isoformat(timespec="seconds"),
        "xml_bytes": xml_path.stat().st_size,
        "xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "prevalence_rows": prevalence.height,
        "orphanet_codes": prevalence["orpha_code"].n_unique(),
        "columns_kept": list(PREVALENCE_SCHEMA),
        "open_targets_disease_table": {
            "release": opentargets.RELEASE,
            "columns_read": list(opentargets.DISEASE_COLUMNS),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return xml_path


# ---------------------------------------------------------------- Open Targets side


def disease_orphanet_codes(disease: pl.DataFrame) -> pl.DataFrame:
    """(disease_id, orpha_code) pairs, from Orphanet cross-references and Orphanet's own ids.

    `disease` has the Open Targets columns `id` and `dbXRefs`.
    """
    crossref = (
        disease.explode("dbXRefs")
        .filter(pl.col("dbXRefs").str.starts_with("Orphanet:"))
        .select(
            pl.col("id").alias("disease_id"),
            pl.col("dbXRefs").str.replace("Orphanet:", "").alias("orpha_code"),
        )
    )
    own = disease.filter(pl.col("id").str.starts_with("Orphanet_")).select(
        pl.col("id").alias("disease_id"),
        pl.col("id").str.replace("Orphanet_", "").alias("orpha_code"),
    )
    return pl.concat([crossref, own]).unique()


def trial_diseases(reports: pl.DataFrame, nct_ids: list[str]) -> pl.DataFrame:
    """(nct_id, disease_id) for the given trials, from the Open Targets clinical report.

    `reports` has `id` (the NCT id) and `diseases` (a list of structs with `diseaseId`).
    """
    return (
        reports.with_columns(nct_id=pl.col("id").str.to_uppercase())
        .filter(pl.col("nct_id").is_in(nct_ids))
        .explode("diseases")
        .select("nct_id", pl.col("diseases").struct.field("diseaseId").alias("disease_id"))
        .drop_nulls("disease_id")
        .unique()
    )


def classify_trials(
    diseases: pl.DataFrame, codes: pl.DataFrame, classes: pl.DataFrame
) -> pl.DataFrame:
    """One row per trial that has any disease: its counts of diseases by kind, and its group.

    `diseases` is (nct_id, disease_id); `codes` is (disease_id, orpha_code); `classes` is
    (orpha_code, class_rank). Columns returned: nct_id, n_diseases, n_rare, n_unclear, group,
    band (the class of the trial's RAREST rare disease, 0 to 3, else null), band_commonest
    (the class of its commonest rare disease, for comparison), all_rare (every disease of the
    trial is rare).
    """
    per_disease = (
        diseases.join(codes, on="disease_id", how="left")
        .join(classes, on="orpha_code", how="left")
        .group_by("nct_id", "disease_id")
        .agg(
            has_entry=pl.col("orpha_code").is_not_null().any(),
            rank=pl.col("class_rank").max(),
        )
        .with_columns(
            kind=pl.when(pl.col("rank") <= RARE_MAX_RANK)
            .then(pl.lit("rare"))
            .when(pl.col("rank").is_not_null())
            .then(pl.lit("common"))
            .when(pl.col("has_entry"))
            .then(pl.lit("unclear"))
            .otherwise(pl.lit("no entry"))
        )
    )
    per_trial = per_disease.group_by("nct_id").agg(
        n_diseases=pl.len(),
        n_rare=(pl.col("kind") == "rare").sum(),
        n_unclear=(pl.col("kind") == "unclear").sum(),
        band=pl.col("rank").filter(pl.col("kind") == "rare").min(),
        band_commonest=pl.col("rank").filter(pl.col("kind") == "rare").max(),
    )
    return per_trial.with_columns(
        group=pl.when(pl.col("n_rare") > 0)
        .then(pl.lit(RARE))
        .when(pl.col("n_unclear") == pl.col("n_diseases"))
        .then(pl.lit(UNCLEAR))
        .otherwise(pl.lit(NOT_RARE)),
        all_rare=pl.col("n_rare") == pl.col("n_diseases"),
    ).sort("nct_id")


def rare_flags(
    nct_ids: list[str],
    orphanet_dir: Path = ORPHANET_DIR,
    ot_dir: Path = opentargets.OT_DIR,
    measure_first: bool = True,
) -> pl.DataFrame:
    """For each trial id: its rare-disease group, band and all_rare, from the files on disk.

    Trials with no Open Targets disease are "unmapped". Reads only named columns.
    """
    reports = pl.read_parquet(ot_dir / "clinical_report.parquet", columns=["id", "diseases"])
    disease = pl.read_parquet(ot_dir / "disease.parquet", columns=list(opentargets.DISEASE_COLUMNS))
    prevalence = pl.read_parquet(orphanet_dir / "prevalence.parquet")
    per_trial = classify_trials(
        trial_diseases(reports, nct_ids),
        disease_orphanet_codes(disease),
        one_class_per_disease(prevalence, measure_first),
    )
    return (
        pl.DataFrame({"nct_id": nct_ids}, schema={"nct_id": pl.String})
        .join(per_trial, on="nct_id", how="left")
        .with_columns(pl.col("group").fill_null(UNMAPPED))
    )


def main() -> None:
    xml_path = download_orphanet()
    print(f"Orphanet prevalence: {xml_path.name}, parsed to prevalence.parquet")
    print(f"Open Targets disease table: {opentargets.fetch_disease_table().name}")


if __name__ == "__main__":
    main()
