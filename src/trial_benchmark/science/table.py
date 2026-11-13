"""Join every science source into one table: data/interim/science.parquet.

    uv run python -m trial_benchmark.science.table

One row per trial (all 7,577, every split), with `split`, the two target-count columns and the
feature columns of each source (`science.FEATURES` says how each is dated). A column is null
where the trial has no investigational target, or the target is not in that source. No label is
read anywhere. The command downloads and checks the source files first (`--no-download` skips
that), builds the table, runs the known-biology checks (`checks.py`) and stops with a message,
writing nothing, if one fails. Otherwise it writes the table and the join funnel by split
(`science_funnel.csv`, counts only, which the report reads).
"""

from __future__ import annotations

import argparse

import polars as pl

from trial_benchmark import science
from trial_benchmark.opentargets import INTERIM
from trial_benchmark.science import annotations, checks, chemistry, depmap, expression, networks
from trial_benchmark.science import targets as tg

OUT_PATH = INTERIM / "science.parquet"
FUNNEL_PATH = INTERIM / "science_funnel.csv"
TABLE = "science"


def assemble(
    trials: pl.DataFrame, targets: pl.DataFrame, pieces: list[pl.DataFrame]
) -> pl.DataFrame:
    """Left-join the per-source tables onto the trial list. `trials` has nct_id and split."""
    counts = tg.target_counts(targets)
    table = trials.select("nct_id", "split").join(counts, on="nct_id", how="left")
    for piece in pieces:
        table = table.join(piece, on="nct_id", how="left", validate="1:1")
    return table.with_columns(
        pl.col("has_target").fill_null(False), pl.col("n_target_genes").fill_null(0)
    ).sort("nct_id")


def funnel(table: pl.DataFrame, drugs: pl.DataFrame, targets: pl.DataFrame) -> pl.DataFrame:
    """Trials by split at each step: all, with an investigational drug, with a target, with a
    value from each source (any of its columns). Counts only."""
    with_drug = drugs.select("nct_id").unique().with_columns(pl.lit(True).alias("has_drug"))
    counts = [pl.len().alias("trials"), pl.col("has_drug").sum().alias("investigational drug"),
              pl.col("has_target").sum().alias("target")]  # fmt: skip
    for name, module in science.SOURCES.items():
        columns = list(module.FEATURES)
        counts.append(
            pl.any_horizontal(pl.col(c).is_not_null() for c in columns).sum().alias(f"{name} value")
        )
    return (
        table.join(with_drug, on="nct_id", how="left")
        .with_columns(pl.col("has_drug").fill_null(False))
        .group_by("split")
        .agg(counts)
        .sort("split")
    )


def build_all(
    download: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(table, drugs, targets). Downloads and checks the source files first."""
    drugs, targets, conditions = tg.build_inputs()
    if download:
        tg.download_hgnc()
        annotations.download()
        depmap.download()
        expression.download()
        networks.download()
        chemistry.download(drugs.filter(pl.col("is_lead"))["drugId"].to_list())
    trials = tg.load_trials()
    hgnc = tg.read_hgnc()
    mapping = tg.entrez_to_ensembl(hgnc)
    intrinsic = depmap.tumour_intrinsic(
        targets, chemistry.target_actions(), depmap.host_directed_ids(hgnc)
    )
    pieces = [
        depmap.trial_features(
            targets,
            intrinsic,
            trials,
            conditions,
            depmap.gene_table(mapping=mapping),
            depmap.lineage_table(mapping=mapping),
        ),
        expression.trial_features(
            targets, trials, conditions, expression.gene_table(), expression.matched_tissue_table()
        ),
        networks.trial_features(targets, networks.gene_table(mapping=mapping)),
        annotations.trial_features(targets, annotations.gene_table()),
        chemistry.trial_features(drugs, targets, chemistry.drug_table()),
    ]
    return assemble(trials, targets, pieces), drugs, targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-download", action="store_true", help="use the files already on disk")
    args = parser.parse_args()
    table, drugs, targets = build_all(download=not args.no_download)
    unexpected = set(table.columns) - set(science.FEATURES) - {"nct_id", "split"}
    if unexpected:
        raise ValueError(f"Columns with no dating: {sorted(unexpected)}")
    results = checks.run_all(drugs, targets)
    print(checks.render(results))
    failed = [r.name for r in results if not r.passed]
    if failed:
        raise SystemExit(f"Known-biology checks failed: {failed}. Nothing was written.")
    counts = funnel(table, drugs, targets)
    table.write_parquet(OUT_PATH)
    counts.write_csv(FUNNEL_PATH)
    print(f"\nwrote {OUT_PATH} ({table.height} trials, {table.width} columns)\n")
    print(counts)


if __name__ == "__main__":
    main()
