"""How often is the gene a hit in screens, and how connected is its protein? BioGRID and ORCS.

    uv run python -m trial_benchmark.science.networks    # download, then summarise

Two public collections from BioGRID (https://thebiogrid.org, MIT licence): ORCS release 2.0.18,
which gathers published CRISPR screens, and BioGRID release 5.0.263, which gathers published
protein interactions. Both name genes by Entrez id; the HGNC table gives the Ensembl id.

ORCS screens are counted only if they are human knockout screens that list the whole library
(`FULL_SIZE_AVAILABLE`), so that "not a hit" is known and not just unreported. A gene needs
at least ten viability screens, or five other screens, to get a fraction.

- `net_viability_hit_fraction`: share of viability screens (a time course of cell growth, read
  for genes whose loss slows growth) in which the gene is a hit. Screens from the DepMap
  family (Meyers, Behan, Aguirre, Bandopadhayay, Thandapani, or analysed with CERES) are left
  out, because `depmap.py` already reads those cells, so this is a second, independent look.
- `net_other_hit_fraction`: share of the other knockout screens (drug response, virus, protein
  levels, ...) in which the gene is a hit. Viability screens run for gain of growth are not
  counted as either.
- `net_partners_per_paper`: log2 of (distinct human physical interaction partners in BioGRID
  divided by the papers that report them). It is not a centrality score. A high value means the
  partners came from few large studies (pull-down or two-hybrid screens that list hundreds of
  partners at once), a low value means many small studies each found a few. TP53 scores low,
  PD-1 high. Self-interactions are left out of both counts.

Several targets: the highest value of each (a drug is as consequential, or as central, as its
most consequential target).

Limit. Both collections grow with the literature, so the counts include screens and papers
published after a trial began. They describe the gene as the literature now knows it: TIMELESS
in the sense that no trial's result enters them, but not what a sponsor could have read.
"""

from __future__ import annotations

import tarfile
import tempfile
import zipfile
from pathlib import Path

import polars as pl

from trial_benchmark.dating import TIMELESS
from trial_benchmark.science import targets as tg
from trial_benchmark.science.fetch import RemoteFile, fetch_all

NETWORKS_DIR = tg.RAW / "networks"
RELEASE = "BioGRID 5.0.263; BioGRID ORCS 2.0.18"
LICENCE = "MIT (BioGRID, Tyers Lab)"
ARCHIVE = "https://downloads.thebiogrid.org/Download/"
BIOGRID = RemoteFile(
    "BIOGRID-ORGANISM-5.0.263.tab3.zip",
    ARCHIVE + "BioGRID/Release-Archive/BIOGRID-5.0.263/BIOGRID-ORGANISM-5.0.263.tab3.zip",
    189706838,
    "03bf5036bcf3026511ead173f06025983122d98a61b8bfba3e04e414e01dbeeb",
)
ORCS = RemoteFile(
    "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz",
    ARCHIVE + "BioGRID-ORCS/Release-Archive/BIOGRID-ORCS-2.0.18/"
    "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz",
    752653348,
    "39222a9650eed083edf193debe45eedc4aabc779ca04ea70107b6bd1efd9b8d7",
)
FILES = [BIOGRID, ORCS]
HUMAN_MEMBER = "BIOGRID-ORGANISM-Homo_sapiens-5.0.263.tab3.txt"
INDEX_MEMBER = "BIOGRID-ORCS-SCREEN_INDEX-2.0.18.index.tab.txt"
ORCS_SUMMARY = "orcs_gene_summary.parquet"
BIOGRID_SUMMARY = "biogrid_gene_summary.parquet"

FEATURES: dict[str, str] = dict.fromkeys(
    ("net_viability_hit_fraction", "net_other_hit_fraction", "net_partners_per_paper"), TIMELESS
)
MIN_VIABILITY_SCREENS = 10
MIN_OTHER_SCREENS = 5
DEPMAP_FAMILY_AUTHORS = {
    "Meyers RM (2017)", "Behan FM (2019)", "Aguirre AJ (2016)", "Bandopadhayay P (2019)",
    "Thandapani P (2022)",
}  # fmt: skip
VIABILITY_PHENOTYPES = {"cell proliferation", "viability", "cell viability"}
LOSS_SELECTION = {"Negative Selection", "Positive and Negative Selection"}
VIABILITY, OTHER = "viability", "other"


# ---------------------------------------------------------------- ORCS


def screen_groups(index: pl.DataFrame) -> pl.DataFrame:
    """(screen_id, group): which screens count, and as what.

    Human knockout screens listing the whole library only. "viability" is a growth time course
    read for loss of growth, outside the DepMap family; "other" is every knockout screen whose
    phenotype is not growth or viability (drug response, virus, protein levels ...). A growth
    or viability screen that is not counted as "viability" is counted as neither.
    """
    growth_phenotype = pl.col("PHENOTYPE").is_in(list(VIABILITY_PHENOTYPES))
    growth = (pl.col("EXPERIMENTAL_SETUP") == "Timecourse") & growth_phenotype
    family = pl.col("AUTHOR").is_in(list(DEPMAP_FAMILY_AUTHORS)) | (pl.col("ANALYSIS") == "CERES")
    eligible = index.filter(
        (pl.col("LIBRARY_METHODOLOGY") == "Knockout") & (pl.col("FULL_SIZE_AVAILABLE") == "Yes")
    )
    return eligible.select(
        pl.col("#SCREEN_ID").cast(pl.Int64).alias("screen_id"),
        pl.when(growth & pl.col("SCREEN_TYPE").is_in(list(LOSS_SELECTION)) & ~family)
        .then(pl.lit(VIABILITY))
        .when(~growth_phenotype)
        .then(pl.lit(OTHER))
        .alias("group"),
    ).drop_nulls("group")


def count_hits(screens: list[pl.DataFrame], groups: pl.DataFrame) -> pl.DataFrame:
    """Per Entrez id: screens assayed and screens with a hit, for each group.

    `screens` holds one table per screen: screen_id, entrez_id, hit (True/False); a gene listed
    twice in a screen counts once, as a hit if any row is a hit.
    """
    kept = (
        pl.concat(screens)
        .group_by("screen_id", "entrez_id")
        .agg(pl.col("hit").any())
        .join(groups, on="screen_id")
    )
    return (
        kept.group_by("entrez_id", "group")
        .agg(pl.len().alias("n_screens"), pl.col("hit").sum().alias("n_hits"))
        .pivot(on="group", index="entrez_id", values=["n_screens", "n_hits"])
        .sort("entrez_id")
    )


def read_orcs(path: Path) -> pl.DataFrame:
    """Hit counts per Entrez id from the ORCS archive, read screen by screen without unpacking."""
    index, screens = None, []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if name == INDEX_MEMBER:
                index = pl.read_csv(
                    archive.extractfile(member).read(), separator="\t", infer_schema_length=0,
                    quote_char=None,
                )  # fmt: skip
            elif ".screen.tab.txt" in name and member.isfile():
                table = pl.read_csv(
                    archive.extractfile(member).read(), separator="\t", infer_schema_length=0,
                    quote_char=None, columns=["#SCREEN_ID", "IDENTIFIER_ID", "IDENTIFIER_TYPE", "HIT"],
                )  # fmt: skip
                screens.append(
                    table.filter(pl.col("IDENTIFIER_TYPE") == "ENTREZ_GENE").select(
                        pl.col("#SCREEN_ID").cast(pl.Int64).alias("screen_id"),
                        pl.col("IDENTIFIER_ID").cast(pl.Int64).alias("entrez_id"),
                        (pl.col("HIT") == "YES").alias("hit"),
                    )
                )
    if index is None:
        raise ValueError(f"{path.name} has no screen index ({INDEX_MEMBER}).")
    return count_hits(screens, screen_groups(index))


# ---------------------------------------------------------------- BioGRID


def read_biogrid(path: Path) -> pl.DataFrame:
    """Per Entrez id: degree (distinct physical partners) and pubs (papers reporting them).

    Human-human physical interactions only; an interaction of a gene with itself counts for
    neither its degree nor its papers, and a row whose partner has no Entrez id ("-") is dropped.
    """
    with tempfile.TemporaryDirectory() as scratch:
        with zipfile.ZipFile(path) as archive:
            archive.extract(HUMAN_MEMBER, scratch)
        table = (
            pl.scan_csv(
                Path(scratch) / HUMAN_MEMBER, separator="\t", quote_char=None, infer_schema=False
            )
            .select(
                pl.col("Entrez Gene Interactor A").alias("a"),
                pl.col("Entrez Gene Interactor B").alias("b"),
                pl.col("Experimental System Type").alias("kind"),
                pl.col("Publication Source").alias("paper"),
                pl.col("Organism ID Interactor A").alias("organism_a"),
                pl.col("Organism ID Interactor B").alias("organism_b"),
            )
            .filter(
                (pl.col("kind") == "physical")
                & (pl.col("organism_a") == "9606")
                & (pl.col("organism_b") == "9606")
            )
            .collect()
        )
    return degree_and_papers(table.select("a", "b", "paper"))


def degree_and_papers(interactions: pl.DataFrame) -> pl.DataFrame:
    """From rows of (a, b, paper): per gene, distinct partners and papers. Self-pairs dropped."""
    pairs = interactions.with_columns(
        pl.col("a").cast(pl.Int64, strict=False), pl.col("b").cast(pl.Int64, strict=False)
    ).drop_nulls(["a", "b"])  # BioGRID writes "-" for an interactor with no Entrez id
    both_ways = pl.concat(
        [
            pairs.select(pl.col("a").alias("gene"), pl.col("b").alias("partner"), "paper"),
            pairs.select(pl.col("b").alias("gene"), pl.col("a").alias("partner"), "paper"),
        ]
    ).filter(pl.col("gene") != pl.col("partner"))
    return (
        both_ways.group_by("gene")
        .agg(
            pl.col("partner").n_unique().alias("degree"),
            pl.col("paper").n_unique().alias("papers"),
        )
        .rename({"gene": "entrez_id"})
        .sort("entrez_id")
    )


# ---------------------------------------------------------------- per gene and per trial


def download(directory: Path = NETWORKS_DIR) -> None:
    fetch_all(directory, RELEASE, LICENCE, FILES)
    if not (directory / ORCS_SUMMARY).exists():
        read_orcs(directory / ORCS.name).write_parquet(directory / ORCS_SUMMARY)
    if not (directory / BIOGRID_SUMMARY).exists():
        read_biogrid(directory / BIOGRID.name).write_parquet(directory / BIOGRID_SUMMARY)


def gene_table(directory: Path = NETWORKS_DIR, mapping: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per Ensembl gene with the three network measures (null where too few screens)."""
    if mapping is None:
        mapping = tg.entrez_to_ensembl(tg.read_hgnc())
    orcs = pl.read_parquet(directory / ORCS_SUMMARY).with_columns(
        pl.col(
            "n_screens_viability", "n_screens_other", "n_hits_viability", "n_hits_other"
        ).fill_null(0)
    )
    orcs = orcs.select(
        "entrez_id",
        pl.when(pl.col("n_screens_viability") >= MIN_VIABILITY_SCREENS)
        .then(pl.col("n_hits_viability") / pl.col("n_screens_viability"))
        .alias("net_viability_hit_fraction"),
        pl.when(pl.col("n_screens_other") >= MIN_OTHER_SCREENS)
        .then(pl.col("n_hits_other") / pl.col("n_screens_other"))
        .alias("net_other_hit_fraction"),
    )
    biogrid = pl.read_parquet(directory / BIOGRID_SUMMARY).select(
        "entrez_id", (pl.col("degree") / pl.col("papers")).log(2).alias("net_partners_per_paper")
    )
    genes = orcs.join(biogrid, on="entrez_id", how="full", coalesce=True)
    return (
        tg.with_ensembl(genes, "entrez_id", mapping)
        .drop_nulls("ensembl_gene_id")
        .sort("ensembl_gene_id")
    )


def trial_features(targets: pl.DataFrame, genes: pl.DataFrame) -> pl.DataFrame:
    """One row per trial with a target in either collection; highest value over its targets."""
    joined = (
        targets.select("nct_id", "targetId")
        .unique()
        .join(genes, left_on="targetId", right_on="ensembl_gene_id")
    )
    return joined.group_by("nct_id").agg(pl.col(name).max() for name in FEATURES).sort("nct_id")


def main() -> None:
    tg.download_hgnc()
    download()
    print(gene_table().describe())


if __name__ == "__main__":
    main()
