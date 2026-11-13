"""Which genes does a trial's treatment act on? The join every science source shares.

    trial -> investigational drug(s) -> target genes (Ensembl ids)

- **Drugs.** Open Targets (release 26.09) lists every drug it links to the trial's
  ClinicalTrials.gov record, comparators and background drugs included. We keep only the
  ones matched to the treatment under test, with the rule in `genetics.py`
  (`investigational_drugs`): drug-like interventions of an experimental arm, matched by name.
  A trial where no listed drug matches has no investigational drug, and so no science row.
- **Targets.** Every target Open Targets lists for those drugs, with no disease required
  (unlike `genetics.py`, which needs a target-disease pair).
- **Lead drug.** The first investigational drug in Open Targets' list order. Chemistry and
  mechanism of action are read from it alone: a combination's other drugs are not looked at.
- **Gene identifiers.** Open Targets, GTEx and the Human Protein Atlas use Ensembl gene ids.
  DepMap, BioGRID and ORCS use Entrez ids. The two are joined through the HGNC table
  (approved genes with both ids), from a dated archive file. An id that HGNC links to more
  than one id on the other side is not joined. Nothing is ever joined on a gene symbol.

The trial's conditions (registry text and MeSH terms) are read here too, because the
disease-matched features of `depmap` and `expression` need them.

Aggregation over a drug's several targets is stated per feature in each source module.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl

from trial_benchmark import genetics
from trial_benchmark.dating import AFTER_START
from trial_benchmark.opentargets import INTERIM, OT_DIR, WANTED_COLUMNS
from trial_benchmark.science.fetch import RemoteFile, fetch_all

RAW = OT_DIR.parent
HGNC_DIR = RAW / "hgnc"
# A dated archive file, not HGNC's live "latest" file, so the same checksum holds next month.
HGNC_FILE = RemoteFile(
    "hgnc_complete_set_2026-10-06.txt",
    "https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/"
    "hgnc_complete_set_2026-10-06.txt",
    16973125,
    "8bf6f686e640dbc864a346484626953621b0500d900d26cd085a35c2782d725d",
)
HGNC_RELEASE = "monthly archive 2026-10-06"
HGNC_LICENCE = "CC0 (HGNC, https://www.genenames.org)"

# Whether a trial has any investigational target, and how many, are known only because the drug
# was catalogued (and its target list widened) later: hindsight. Models must not read them
# (see the package docstring).
FEATURES: dict[str, str] = {"has_target": AFTER_START, "n_target_genes": AFTER_START}


# ---------------------------------------------------------------- gene identifiers


def download_hgnc() -> Path:
    return fetch_all(HGNC_DIR, HGNC_RELEASE, HGNC_LICENCE, [HGNC_FILE])[0]


def read_hgnc(path: Path | None = None) -> pl.DataFrame:
    """Approved HGNC genes: hgnc_id, symbol, entrez_id (Int64), ensembl_gene_id."""
    path = path or HGNC_DIR / HGNC_FILE.name
    table = pl.read_csv(path, separator="\t", infer_schema_length=0, quote_char=None)
    return (
        table.filter(pl.col("status") == "Approved")
        .select(
            "hgnc_id",
            "symbol",
            pl.col("entrez_id").cast(pl.Int64, strict=False),
            "ensembl_gene_id",
        )
        .sort("hgnc_id")
    )


def entrez_to_ensembl(hgnc: pl.DataFrame) -> pl.DataFrame:
    """(entrez_id, ensembl_gene_id) pairs where each id appears in exactly one pair.

    An id on either side that HGNC links to two ids on the other side is left out, so a join
    through this table can neither lose a gene's rows twice nor count them twice.
    """
    both = (
        hgnc.drop_nulls(["entrez_id", "ensembl_gene_id"])
        .select("entrez_id", "ensembl_gene_id")
        .unique()
    )
    return both.filter(
        (pl.len().over("entrez_id") == 1) & (pl.len().over("ensembl_gene_id") == 1)
    ).sort("entrez_id")


def with_ensembl(frame: pl.DataFrame, entrez_column: str, mapping: pl.DataFrame) -> pl.DataFrame:
    """Add `ensembl_gene_id` to a table keyed by Entrez id. Unmapped ids get null."""
    return frame.join(mapping, left_on=entrez_column, right_on="entrez_id", how="left")


# ---------------------------------------------------------------- trials


def load_trials() -> pl.DataFrame:
    """nct_id, split and therapeutic_area for every trial, from the feature files. No labels."""
    columns = ["nct_id", "split", "therapeutic_area"]
    return pl.concat(
        [
            pl.read_parquet(INTERIM / name, columns=columns)
            for name in ("dev.parquet", "test_features.parquet")
        ]
    ).sort("nct_id")


def read_registry(ids: Iterable[str]) -> tuple[dict[str, list[str]], pl.DataFrame]:
    """One pass over the registry: experimental-arm intervention names, and conditions.

    Returns (names by NCT id, a table of nct_id, conditions, mesh, ancestors). Reads arms,
    interventions and condition fields only.
    """
    from trial_benchmark.download import iter_studies, nct_id

    wanted = set(ids)
    names: dict[str, list[str]] = {}
    rows = []
    for study in iter_studies():
        key = nct_id(study).upper()
        if key not in wanted:
            continue
        names.update(genetics.experimental_names([study], {key}))
        protocol = study.get("protocolSection", {})
        browse = study.get("derivedSection", {}).get("conditionBrowseModule", {}) or {}
        rows.append(
            {
                "nct_id": key,
                "conditions": protocol.get("conditionsModule", {}).get("conditions") or [],
                "mesh": [m["term"] for m in browse.get("meshes", []) if m.get("term")],
                "ancestors": [m["term"] for m in browse.get("ancestors", []) if m.get("term")],
            }
        )
    schema = {"nct_id": pl.String} | dict.fromkeys(
        ("conditions", "mesh", "ancestors"), pl.List(pl.String)
    )
    return names, pl.DataFrame(rows, schema=schema).sort("nct_id")


# ---------------------------------------------------------------- trial -> drugs -> targets


def investigational_drugs(
    ids: list[str],
    reports: pl.DataFrame,
    molecules: pl.DataFrame,
    experimental: dict[str, list[str]],
) -> pl.DataFrame:
    """(nct_id, drugId, is_lead): the drugs matched to the treatment under test.

    `reports` is Open Targets' clinical report (id, drugs); `molecules` its drug table (id,
    name); `experimental` the registry's experimental-arm names. `is_lead` marks the first
    investigational drug in Open Targets' list order, whether or not it has a known target.
    """
    named = genetics.trial_drug_names(reports, ids)
    matched = genetics.investigational_drugs(named, molecules, experimental).unique(
        maintain_order=True
    )  # a drug can be listed twice under different names
    leads = genetics.lead_drugs(named, matched).join(
        matched, left_on=["nct_id", "lead_drug"], right_on=["nct_id", "drugId"], how="semi"
    )
    lead_flags = leads.select(
        "nct_id", pl.col("lead_drug").alias("drugId"), pl.lit(True).alias("is_lead")
    )
    return (
        matched.join(lead_flags, on=["nct_id", "drugId"], how="left")
        .with_columns(pl.col("is_lead").fill_null(False))
        .sort("nct_id", "drugId")
    )


def investigational_targets(drugs: pl.DataFrame, drug_targets: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, drugId, targetId): every target Open Targets lists for the investigational drugs."""
    return (
        drugs.select("nct_id", "drugId")
        .join(drug_targets.select("drugId", "targetId").drop_nulls().unique(), on="drugId")
        .sort("nct_id", "drugId", "targetId")
    )


def build_inputs(ot_dir: Path = OT_DIR) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(investigational drugs, their targets, conditions) for every trial in the benchmark."""
    ids = load_trials()["nct_id"].to_list()
    reports = pl.read_parquet(
        ot_dir / "clinical_report.parquet", columns=list(WANTED_COLUMNS["clinical_report"])
    )
    drug_targets = pl.read_parquet(ot_dir / "clinical_target.parquet")
    molecules = pl.read_parquet(ot_dir / "drug_molecule.parquet")
    experimental, conditions = read_registry(ids)
    drugs = investigational_drugs(ids, reports, molecules, experimental)
    return drugs, investigational_targets(drugs, drug_targets), conditions


def target_counts(targets: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, n_target_genes, has_target): how many distinct target genes each trial has."""
    return (
        targets.group_by("nct_id")
        .agg(pl.col("targetId").n_unique().alias("n_target_genes"))
        .with_columns(pl.lit(True).alias("has_target"))
    )


MULTIPLE = "multiple"


def agree_or_multiple(column: str) -> pl.Expr:
    """One level per trial from its targets' levels: the shared level, or "multiple" if they differ.

    Targets with no level are ignored. No level at all gives null.
    """
    levels = pl.col(column).drop_nulls().unique()
    return (
        pl.when(levels.len() == 0)
        .then(None)
        .when(levels.len() == 1)
        .then(levels.first())
        .otherwise(pl.lit(MULTIPLE))
        .alias(column)
    )
