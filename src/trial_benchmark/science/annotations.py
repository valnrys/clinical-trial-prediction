"""What kind of protein is the target? Class, location, and how tolerant the gene is of damage.

    uv run python -m trial_benchmark.science.annotations    # download and slim the Open Targets tables

Source: Open Targets Platform release 26.09 (same release as `opentargets.py`), tables
`target` and `target_prioritisation`. CC0. Kept in data/raw/annotations/. The prioritisation
table also carries the target's maximum clinical stage, which is about how far drugs got and
so belongs to the label's neighbourhood: it is downloaded for a moment and never kept
(`PRIORITISATION_COLUMNS` is the whole list that is read).

Features, one row per trial, from the targets of its investigational drug(s):

- `target_class`: the target's level-1 class (Enzyme, Membrane receptor, Ion channel, ...).
  A target with several level-1 classes, or several targets with different classes, give
  "multiple".
- `target_location`: secreted, membrane, nuclear or intracellular. A protein can have many
  locations; the first rule that applies wins: any cell-surface location (the cell or plasma
  membrane, with or without a qualifier such as "apical" or "postsynaptic", or a bare
  "membrane") gives membrane, else any "Secreted" gives secreted, else any nuclear location
  gives nuclear, else any other location (including the membranes of organelles) gives
  intracellular. Several targets that differ give "multiple".
- `constraint`: Open Targets' genetic-constraint score (gnomAD), sign flipped so that higher
  means the gene tolerates damage less. Several targets: the highest (a drug is as
  constrained as its least tolerant target).
- `mouse_knockout`: Open Targets' mouse-knockout score (IMPC), sign flipped the same way.
  Several targets: the highest.

All four describe the gene, not the trial, so they are TIMELESS in the sense of `dating`.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from trial_benchmark.dating import TIMELESS
from trial_benchmark.opentargets import BASE_URL, RELEASE
from trial_benchmark.science import targets as tg
from trial_benchmark.science.fetch import RemoteFile, fetch, write_manifest

ANNOTATIONS_DIR = tg.RAW / "annotations"
LICENCE = "CC0 (Open Targets Platform)"
TARGET_PART = "part-0000{i}-ddfc1f8c-60a1-40fe-885a-2c1dd3f9a9c1-c000.zstd.parquet"
TARGET_FILES = [
    RemoteFile(
        TARGET_PART.format(i=i),
        f"{BASE_URL}target/{TARGET_PART.format(i=i)}",
        size,
        digest,
    )
    for i, (size, digest) in enumerate(
        [
            (7165956, "80186f1c9919d5d1351f009253480fad93b3bb3e2de506d8948ed2126ac48cab"),
            (7196171, "a777ec38d5fbf6e6475d85158141c0a01f2c32cadc7f8769256eaab6e034e702"),
            (7270990, "3621b2510eb3f9da9b533f40559b4f4de390fffc57935726a2b0eebcc40aa028"),
        ]
    )
]
PRIORITISATION_PART = "part-00000-4c96f461-f5af-4b65-878f-9e76a0f36f21-c000.zstd.parquet"
PRIORITISATION_FILE = RemoteFile(
    PRIORITISATION_PART,
    f"{BASE_URL}target_prioritisation/{PRIORITISATION_PART}",
    1551695,
    "ed9e2a9f2ed4c4c55ad601f726f522a2b8440c38a6e24d313d85101c8a4a7e1e",
)
# The only columns ever read. maxClinicalStage and the safety-event flag are not on the list.
TARGET_COLUMNS = ("id", "targetClass", "subcellularLocations")
PRIORITISATION_COLUMNS = ("targetId", "geneticConstraint", "mouseKOScore")
TARGET_SLIM = "target_annotations.parquet"
PRIORITISATION_SLIM = "target_tolerance.parquet"

FEATURES: dict[str, str] = dict.fromkeys(
    ("target_class", "target_location", "constraint", "mouse_knockout"), TIMELESS
)

# The first part of a location string ("Cell membrane ; Multi-pass membrane protein") names
# the place. Surface: a bare "membrane", the plasma or cell membrane (also "apical cell
# membrane", "postsynaptic cell membrane"), the cell surface. Not "nuclear membrane" etc.
SURFACE = re.compile(
    r"^(membrane($|,|\s)|plasma membrane|cell surface|cell membrane)|\b(cell|plasma) membrane$"
)
# "Nucleus" and "nucleolus" are nuclear; "perinuclear region" (cytoplasm around the nucleus) is not.
NUCLEAR = re.compile(r"\bnucle")


# ---------------------------------------------------------------- download


def download(directory: Path = ANNOTATIONS_DIR) -> None:
    """Fetch the Open Targets parts, keep the named columns only, delete the full files."""
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / TARGET_SLIM).exists():
        pieces = []
        for file in TARGET_FILES:
            path = fetch(file, directory)
            pieces.append(pl.read_parquet(path, columns=list(TARGET_COLUMNS)))
            path.unlink()
        pl.concat(pieces).write_parquet(directory / TARGET_SLIM)
    if not (directory / PRIORITISATION_SLIM).exists():
        path = fetch(PRIORITISATION_FILE, directory)
        pl.read_parquet(path, columns=list(PRIORITISATION_COLUMNS)).write_parquet(
            directory / PRIORITISATION_SLIM
        )
        path.unlink()
    write_manifest(
        directory, f"Open Targets {RELEASE}", LICENCE, [*TARGET_FILES, PRIORITISATION_FILE]
    )


# ---------------------------------------------------------------- per gene


def location_class(locations: list[str] | None) -> str | None:
    """secreted / membrane / nuclear / intracellular from a protein's location strings."""
    texts = [t.lower() for t in locations or [] if t]
    if not texts:
        return None
    if any(SURFACE.search(t.split(";")[0].strip()) for t in texts):
        return "membrane"
    if any(t.startswith("secreted") for t in texts):
        return "secreted"
    if any(NUCLEAR.search(t) for t in texts):
        return "nuclear"
    return "intracellular"


def level_one_class(classes: list[dict] | None) -> str | None:
    """The target's level-1 class label, "multiple" if it has several, null if none."""
    labels = {c["label"] for c in classes or [] if c["level"] == "l1" and c["label"]}
    if len(labels) > 1:
        return tg.MULTIPLE
    return next(iter(labels), None)


def gene_table(directory: Path = ANNOTATIONS_DIR) -> pl.DataFrame:
    """One row per Ensembl gene: target_class, target_location, constraint, mouse_knockout."""
    proteins = pl.read_parquet(directory / TARGET_SLIM)
    rows = [
        {
            "targetId": row["id"],
            "target_class": level_one_class(row["targetClass"]),
            "target_location": location_class(
                [s["location"] for s in row["subcellularLocations"] or []]
            ),
        }
        for row in proteins.iter_rows(named=True)
    ]
    classes = pl.DataFrame(
        rows,
        schema={"targetId": pl.String, "target_class": pl.String, "target_location": pl.String},
    )
    tolerance = pl.read_parquet(directory / PRIORITISATION_SLIM).select(
        "targetId",
        (-pl.col("geneticConstraint")).alias("constraint"),
        (-pl.col("mouseKOScore")).alias("mouse_knockout"),
    )
    return classes.join(tolerance, on="targetId", how="full", coalesce=True).sort("targetId")


# ---------------------------------------------------------------- per trial


def trial_features(targets: pl.DataFrame, genes: pl.DataFrame) -> pl.DataFrame:
    """One row per trial with an investigational target. Aggregation rules in the docstring."""
    joined = targets.select("nct_id", "targetId").unique().join(genes, on="targetId", how="left")
    return (
        joined.group_by("nct_id")
        .agg(
            tg.agree_or_multiple("target_class"),
            tg.agree_or_multiple("target_location"),
            pl.col("constraint").max(),
            pl.col("mouse_knockout").max(),
        )
        .sort("nct_id")
    )


def main() -> None:
    download()
    print(gene_table().describe())


if __name__ == "__main__":
    main()
