"""Known-biology checks: does each source say what everyone knows it should?

A join can be wrong in a way that still produces a number, and a null result built on a wrong
join says nothing. So before any feature is tested, each source is asked questions with
well-known answers: ribosomes are essential, insulin is made in the pancreas, albumin in the
liver, a statin's target is a liver enzyme, p53 is among the most studied and most connected
proteins, EGFR is a kinase, aspirin weighs 180. Each check returns True or False with the
numbers behind it. `table.py` runs them all and stops, writing nothing, if any fails.

What the checks cover, and what they do not: about thirty hand-picked genes and two drugs, all
well known. They catch a join that is plainly broken (wrong file, wrong id, shifted columns,
a sign flipped). They say nothing about the ordinary gene nobody picked, about the disease
matching, or about whether the drug-to-target step picked the right drug; the spot check in
`trace.py` samples random trials for the first of those and the unit tests cover rules.

Gene symbols are used here only to pick which genes to look at (through the HGNC table, to
their Ensembl ids). No feature is ever joined on a symbol.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from trial_benchmark.science import annotations, chemistry, depmap, expression, networks
from trial_benchmark.science import targets as tg

RIBOSOMAL = ("RPL5", "RPL11", "RPS14", "RPS19")
POLYMERASES = ("POLR2B", "POLR2E", "POLR2L")  # POLR2A itself is not in the DepMap matrix
NOT_ESSENTIAL = ("INS", "ALB", "TNNT2")
OLFACTORY = ("OR2A4", "OR1A1", "OR5A1", "OR2W1")
STATIN_TARGET = "HMGCR"
STATINS = {"CHEMBL1487": "atorvastatin", "CHEMBL1496": "rosuvastatin"}
TISSUE_OF = {
    "INS": "pancreas",
    "ALB": "liver",
    "TNNT2": "heart",
    "HMGCR": "liver",
    "PCSK9": "liver",
}
# (gene, lineage): a cancer lineage that depends on the gene far more than other lineages do.
LINEAGE_DEPENDENCIES = (("SOX10", "Skin"), ("PAX8", "Ovary/Fallopian Tube"), ("MYB", "Myeloid"))
KINASES = ("EGFR", "BRAF", "ABL1")


@dataclass(frozen=True)
class Result:
    name: str
    passed: bool
    detail: str


def ids_of(symbols: tuple[str, ...], hgnc: pl.DataFrame) -> dict[str, str]:
    """Ensembl ids for approved HGNC symbols. A symbol that is not found is left out."""
    found = hgnc.filter(pl.col("symbol").is_in(list(symbols))).drop_nulls("ensembl_gene_id")
    return dict(zip(found["symbol"], found["ensembl_gene_id"], strict=True))


def _value(table: pl.DataFrame, key: str, ensembl: str, column: str):
    row = table.filter(pl.col(key) == ensembl)
    return row[column][0] if row.height else None


# ---------------------------------------------------------------- the checks


def common_essential(genes: pl.DataFrame, hgnc: pl.DataFrame) -> Result:
    """Ribosomal and polymerase genes are common essential; insulin, albumin, troponin are not."""
    ids = ids_of((*RIBOSOMAL, *POLYMERASES, *NOT_ESSENTIAL), hgnc)
    flag = {s: _value(genes, "ensembl_gene_id", e, "common_essential") for s, e in ids.items()}
    wrong = [s for s in (*RIBOSOMAL, *POLYMERASES) if flag.get(s) is not True]
    wrong += [s for s in NOT_ESSENTIAL if flag.get(s) is not False]  # absent is a failure too
    return Result(
        "DepMap: ribosomal and polymerase genes common essential, INS/ALB/TNNT2 not",
        not wrong,
        f"flags {flag}" if not wrong else f"wrong or missing: {wrong}",
    )


def lineage_dependencies(genes: pl.DataFrame, lineages: pl.DataFrame, hgnc: pl.DataFrame) -> Result:
    """SOX10 in skin, PAX8 in ovary and MYB in myeloid lines are more dependent than elsewhere."""
    ids = ids_of(tuple(g for g, _ in LINEAGE_DEPENDENCIES), hgnc)
    differences = {}
    for symbol, lineage in LINEAGE_DEPENDENCIES:
        gene = ids.get(symbol, "")
        row = lineages.filter((pl.col("ensembl_gene_id") == gene) & (pl.col("lineage") == lineage))
        overall = genes.filter(pl.col("ensembl_gene_id") == gene)
        if row.is_empty() or overall.is_empty():
            differences[symbol] = None
            continue
        n_in, n_all = row["n_lines"][0], overall["n_lines"][0]
        in_mean = row["sum_effect"][0] / n_in
        out_mean = (overall["mean_effect"][0] * n_all - row["sum_effect"][0]) / (n_all - n_in)
        differences[symbol] = round(in_mean - out_mean, 2)
    ok = all(d is not None and d < -0.3 for d in differences.values())
    return Result(
        "DepMap: SOX10 (skin), PAX8 (ovary), MYB (myeloid) more essential in their own lineage",
        ok,
        f"matched minus other lineages' mean effect {differences} (needs below -0.3)",
    )


def tissue_of_origin(matched: pl.DataFrame, hgnc: pl.DataFrame) -> Result:
    """INS pancreas, ALB liver, TNNT2 heart, HMGCR and PCSK9 liver: the tissue is the gene's top one."""
    ids = ids_of(tuple(TISSUE_OF), hgnc)
    ranks = {}
    for symbol, tissue in TISSUE_OF.items():
        row = matched.filter(
            (pl.col("ensembl_gene_id") == ids.get(symbol, "")) & (pl.col("tissue") == tissue)
        )
        ranks[f"{symbol}/{tissue}"] = row["best_rank"][0] if row.height else None
    return Result(
        "Expression: INS pancreas, ALB liver, TNNT2 heart, HMGCR and PCSK9 liver are top tissues",
        all(r == 1 for r in ranks.values()),
        f"rank among 51 tissues {ranks}",
    )


def tissue_specificity(genes: pl.DataFrame, hgnc: pl.DataFrame) -> Result:
    """Insulin is tissue-specific (tau above 0.9); actin and GAPDH are housekeeping (below 0.4)."""
    ids = ids_of(("INS", "ACTB", "GAPDH"), hgnc)
    tau = {s: _value(genes, "ensembl_gene_id", e, "gtex_tau") for s, e in ids.items()}
    ok = (
        None not in tau.values()
        and len(tau) == 3
        and tau["INS"] > 0.9
        and tau["ACTB"] < 0.4
        and tau["GAPDH"] < 0.4
    )
    return Result("Expression: INS tissue-specific, ACTB and GAPDH broad (GTEx tau)", ok,
                  f"tau {({s: None if v is None else round(v, 2) for s, v in tau.items()})}")  # fmt: skip


def studied_and_connected(genes: pl.DataFrame, raw: pl.DataFrame, hgnc: pl.DataFrame) -> Result:
    """TP53 is in the top 1% of genes by both interaction partners and papers."""
    tp53 = ids_of(("TP53",), hgnc)["TP53"]
    entrez = hgnc.filter(pl.col("symbol") == "TP53")["entrez_id"][0]
    connected = raw.filter(pl.col("degree") > 0)
    row = connected.filter(pl.col("entrez_id") == entrez)
    if row.is_empty():
        return Result("Networks: TP53 highly connected and highly studied", False, "TP53 missing")
    degree_share = (connected["degree"] > row["degree"][0]).mean()
    papers_share = (connected["papers"] > row["papers"][0]).mean()
    ratio = _value(genes, "ensembl_gene_id", tp53, "net_partners_per_paper")
    return Result(
        "Networks: TP53 in the top 1% for BioGRID partners and for papers",
        degree_share < 0.01 and papers_share < 0.01,
        f"{row['degree'][0]} partners (above {degree_share:.2%} of genes), {row['papers'][0]} papers "
        f"(above {papers_share:.2%}); log2 partners per paper {ratio}",
    )


def screen_hits(genes: pl.DataFrame, hgnc: pl.DataFrame) -> Result:
    """Ribosomal genes are hits in most viability screens; olfactory receptors in almost none."""
    ids = ids_of((*RIBOSOMAL, *OLFACTORY), hgnc)

    def median_of(symbols: tuple[str, ...]) -> float | None:
        values = [
            _value(genes, "ensembl_gene_id", ids[s], "net_viability_hit_fraction")
            for s in symbols
            if s in ids
        ]
        values = [v for v in values if v is not None]
        return float(pl.Series(values).median()) if values else None

    ribosomal, olfactory = median_of(RIBOSOMAL), median_of(OLFACTORY)
    ok = ribosomal is not None and olfactory is not None and ribosomal >= 0.5 and olfactory <= 0.05
    return Result(
        "Networks: ribosomal genes hit in most viability screens, olfactory receptors in few",
        ok,
        f"median hit fraction: ribosomal {ribosomal}, olfactory {olfactory}",
    )


def protein_classes(directory, hgnc: pl.DataFrame, genes: pl.DataFrame) -> Result:
    """EGFR, BRAF, ABL1 are kinases (level-2 class); INS is secreted, EGFR membrane, TP53 nuclear."""
    ids = ids_of((*KINASES, "INS", "TP53"), hgnc)
    classes = pl.read_parquet(directory / annotations.TARGET_SLIM)
    kinase = {}
    for symbol in KINASES:
        row = classes.filter(pl.col("id") == ids.get(symbol, ""))
        labels = [c["label"] for c in row["targetClass"][0]] if row.height else []
        kinase[symbol] = "Kinase" in labels
    place = {
        s: _value(genes, "targetId", ids.get(s, ""), "target_location")
        for s in ("INS", "EGFR", "TP53")
    }
    ok = all(kinase.values()) and place == {
        "INS": "secreted",
        "EGFR": "membrane",
        "TP53": "nuclear",
    }
    return Result(
        "Annotations: EGFR, BRAF, ABL1 are kinases; INS secreted, EGFR membrane, TP53 nuclear",
        ok,
        f"kinase {kinase}; location {place}",
    )


def tolerance(genes: pl.DataFrame, hgnc: pl.DataFrame) -> Result:
    """Ribosomal genes are less tolerant of damage than the typical gene, and than olfactory
    receptors (constraint; olfactory receptors have no mouse-knockout score)."""
    ids = ids_of((*RIBOSOMAL, *OLFACTORY), hgnc)

    def mean_of(symbols: tuple[str, ...], column: str) -> float | None:
        values = [_value(genes, "targetId", ids[s], column) for s in symbols if s in ids]
        values = [v for v in values if v is not None]
        return round(float(pl.Series(values).mean()), 2) if values else None

    scores = {
        "constraint: ribosomal": mean_of(RIBOSOMAL, "constraint"),
        "constraint: olfactory": mean_of(OLFACTORY, "constraint"),
        "constraint: all genes": round(float(genes["constraint"].median()), 2),
        "mouse knockout: ribosomal": mean_of(RIBOSOMAL, "mouse_knockout"),
        "mouse knockout: all genes": round(float(genes["mouse_knockout"].median()), 2),
    }
    present = all(v is not None for v in scores.values())
    ok = (
        present
        and scores["constraint: ribosomal"] > scores["constraint: olfactory"]
        and scores["constraint: ribosomal"] > scores["constraint: all genes"]
        and scores["mouse knockout: ribosomal"] > scores["mouse knockout: all genes"]
    )
    return Result(
        "Annotations: ribosomal genes less tolerant of damage than olfactory receptors and the typical gene",
        ok,
        f"scores {scores} (higher means less tolerant)",
    )


def known_molecules(table: pl.DataFrame) -> Result:
    """Aspirin weighs 180 and breaks no rule of five; atorvastatin is an inhibitor."""
    aspirin = table.filter(pl.col("drugId") == "CHEMBL25")
    statin = table.filter(pl.col("drugId") == "CHEMBL1487")
    if aspirin.is_empty() or statin.is_empty():
        return Result("Chemistry: aspirin and atorvastatin", False, "drug not in the table")
    weight = aspirin["chem_mol_weight"][0]
    violations = aspirin["chem_rule_of_five_violations"][0]
    action = statin["chem_action_type"][0]
    return Result(
        "Chemistry: aspirin weighs 180 with no rule-of-five violation; atorvastatin is an inhibitor",
        abs(weight - 180.16) < 0.1 and violations == 0 and action == "inhibitor",
        f"aspirin mass {weight}, violations {violations}; atorvastatin action {action}",
    )


def statin_trials_reach_hmgcr(
    drugs: pl.DataFrame, targets: pl.DataFrame, hgnc: pl.DataFrame
) -> Result:
    """The drug -> target join: trials whose drug is atorvastatin or rosuvastatin end at HMGCR.

    The trials are the ones the join already attached those drugs to, so this tests the
    drug-to-target step and the Ensembl ids, not whether the name matching found every statin trial.
    """
    hmgcr = ids_of((STATIN_TARGET,), hgnc)[STATIN_TARGET]
    statin_trials = drugs.filter(pl.col("drugId").is_in(list(STATINS)))["nct_id"].unique().to_list()
    reached = targets.filter(
        pl.col("nct_id").is_in(statin_trials) & pl.col("drugId").is_in(list(STATINS))
        & (pl.col("targetId") == hmgcr)
    )["nct_id"].n_unique()  # fmt: skip
    share = reached / max(len(statin_trials), 1)
    return Result(
        "Join: trials of atorvastatin or rosuvastatin have HMGCR as a target",
        len(statin_trials) >= 10 and share >= 0.95,
        f"{reached} of {len(statin_trials)} trials",
    )


# ---------------------------------------------------------------- run


def run_all(drugs: pl.DataFrame, targets: pl.DataFrame) -> list[Result]:
    """Load the real source tables and run every check on them and on the trial joins."""
    hgnc = tg.read_hgnc()
    mapping = tg.entrez_to_ensembl(hgnc)
    depmap_genes = depmap.gene_table(mapping=mapping)
    expression_genes = expression.gene_table()
    network_genes = networks.gene_table(mapping=mapping)
    annotation_genes = annotations.gene_table()
    raw_degree = pl.read_parquet(networks.NETWORKS_DIR / networks.BIOGRID_SUMMARY)
    return [
        common_essential(depmap_genes, hgnc),
        lineage_dependencies(depmap_genes, depmap.lineage_table(mapping=mapping), hgnc),
        tissue_of_origin(expression.matched_tissue_table(), hgnc),
        tissue_specificity(expression_genes, hgnc),
        studied_and_connected(network_genes, raw_degree, hgnc),
        screen_hits(network_genes, hgnc),
        protein_classes(annotations.ANNOTATIONS_DIR, hgnc, annotation_genes),
        tolerance(annotation_genes, hgnc),
        known_molecules(chemistry.drug_table()),
        statin_trials_reach_hmgcr(drugs, targets, hgnc),
    ]


def render(results: list[Result]) -> str:
    return "\n".join(
        f"[{'ok' if r.passed else 'FAIL'}] {r.name}\n       {r.detail}" for r in results
    )
