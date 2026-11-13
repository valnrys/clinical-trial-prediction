"""Spot check: follow random trials from the raw files to the numbers in the science table.

    python -m elman_exercise.science.trace      # 25 random mapped development trials

The known-biology checks (`checks.py`) look at about thirty hand-picked genes. This asks a
different question: for trials drawn at random, does every value in `science.parquet` equal what
you get by reading the source files again, with loops written separately from the table build
(plain loops and direct lookups, no shared aggregation)?

What it re-reads: the raw DepMap, GTEx, HPA, BioGRID and ORCS files. What it does not: the
protein annotations and the chemistry are read from the slim files the build wrote, not from
the raw Open Targets and ChEMBL downloads. It reuses the build's gene-id mapping and its
lineage and tissue rule functions, and starts from the build's own list of target genes for each
trial. It covers 25 trials. It does not cover the step from trial to drug to target (see
`checks.statin_trials_reach_hmgcr` and the unit tests) or whether the disease-mapping rules are
right (they have unit tests).

A pass means the join from target gene to value, the aggregation over several targets and the
null handling agree with a second reading for these trials. It says nothing about the rules
themselves being right.
"""

from __future__ import annotations

import gzip
import io
import math
import re
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from elman_exercise.science import annotations, chemistry, depmap, expression, networks
from elman_exercise.science import targets as tg

SAMPLE = 25
# The cell-surface rule, written out again here rather than imported (a test checks the two agree).
SURFACE = re.compile(
    r"^(membrane($|,|\s)|plasma membrane|cell surface|cell membrane)|\b(cell|plasma) membrane$"
)
SEED = 7


@dataclass(frozen=True)
class Check:
    source: str
    values: int  # table values compared (a missing value on both sides counts as agreement)
    present: int  # of those, how many are a real number or label, not "missing"
    agreed: int
    mismatches: tuple[str, ...]


def same(table_value, recomputed) -> bool:
    if table_value is None or recomputed is None:
        return table_value is None and recomputed is None
    if isinstance(recomputed, float):
        return math.isclose(float(table_value), recomputed, rel_tol=1e-9, abs_tol=1e-12)
    return table_value == recomputed


def compare(source: str, table: dict[str, dict], expected: dict[str, dict]) -> Check:
    """`table` and `expected` map nct_id -> {feature: value}."""
    values, present, mismatches = 0, 0, []
    for nct, features in expected.items():
        for name, recomputed in features.items():
            values += 1
            present += recomputed is not None
            if not same(table[nct][name], recomputed):
                mismatches.append(f"{nct} {name}: table {table[nct][name]}, raw {recomputed}")
    return Check(source, values, present, values - len(mismatches), tuple(mismatches))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


# ---------------------------------------------------------------- DepMap


def read_columns(path: Path, entrez: set[int]) -> dict[str, dict[int, list]]:
    """{model id: {entrez id: value or None}} for the named genes, straight from a DepMap CSV."""
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split(",")
    wanted = {i: int(h.rsplit("(", 1)[1].rstrip(")")) for i, h in enumerate(header) if i}
    columns = [header[i] for i, e in wanted.items() if e in entrez]
    table = pl.read_csv(path, columns=[header[0], *columns], infer_schema_length=0)
    by_name = {header[i]: e for i, e in wanted.items()}
    return {
        row[header[0]]: {
            by_name[c]: (None if row[c] in (None, "") else float(row[c])) for c in columns
        }
        for row in table.iter_rows(named=True)
    }


def depmap_values(
    sample: dict[str, list[str]], lineages: dict[str, list[str]], intrinsic: dict[str, list[str]]
) -> dict[str, dict]:
    """The five DepMap features for each sampled trial, recomputed from the raw files."""
    directory = depmap.DEPMAP_DIR
    mapping = tg.entrez_to_ensembl(tg.read_hgnc())
    entrez_of = dict(zip(mapping["ensembl_gene_id"], mapping["entrez_id"], strict=True))
    genes = {g for targets in sample.values() for g in targets if g in entrez_of}
    wanted = {entrez_of[g] for g in genes}
    effect = read_columns(directory / depmap.EFFECT.name, wanted)
    dependency = read_columns(directory / depmap.DEPENDENCY.name, wanted)
    essential = {
        int(s.rsplit("(", 1)[1].rstrip(")"))
        for s in pl.read_csv(directory / depmap.COMMON_ESSENTIAL.name)["Essentials"]
    }
    models = pl.read_csv(directory / depmap.MODEL.name, infer_schema_length=0)
    lineage_of = dict(
        zip(models["ModelID"], models["OncotreeLineage"].fill_null("Unknown"), strict=True)
    )
    out = {}
    for nct, targets in sample.items():
        stats = {}
        for gene in targets:
            if gene not in entrez_of:
                continue
            e = entrez_of[gene]
            scores = [row[e] for row in effect.values() if row[e] is not None]
            probabilities = [row[e] for row in dependency.values() if row[e] is not None]
            stats[gene] = (
                mean(scores),
                sum(p > 0.5 for p in probabilities) / len(probabilities) if probabilities else None,
                e in essential,
            )
        scored = {g: v for g, v in stats.items() if v[0] is not None}
        best = min(scored.items(), key=lambda kv: (kv[1][0], kv[0])) if scored else None
        row = {
            "depmap_common_essential": any(v[2] for v in stats.values()) if stats else None,
            "depmap_mean_effect": best[1][0] if best else None,
            "depmap_fraction_dependent": best[1][1] if best else None,
            "depmap_lineage_effect": None,
            "depmap_lineage_minus_other": None,
        }
        matched_models = {m for m in effect if lineage_of[m] in lineages.get(nct, [])}
        if 1 <= len(set(lineages.get(nct, []))) <= depmap.MAX_LINEAGES:
            candidates = []
            for gene in intrinsic.get(nct, []):
                if gene not in entrez_of:
                    continue
                e = entrez_of[gene]
                inside = [effect[m][e] for m in matched_models if effect[m][e] is not None]
                outside = [
                    v[e] for m, v in effect.items() if m not in matched_models and v[e] is not None
                ]
                if len(inside) >= depmap.MIN_LINES and outside:
                    candidates.append((mean(inside), gene, mean(inside) - mean(outside)))
            if candidates:
                best_line = min(candidates, key=lambda c: (c[0], c[1]))
                row["depmap_lineage_effect"], _, row["depmap_lineage_minus_other"] = best_line
        out[nct] = row
    return out


# ---------------------------------------------------------------- expression


def expression_values(
    sample: dict[str, list[str]], tissues: dict[str, list[str]]
) -> dict[str, dict]:
    directory = expression.EXPRESSION_DIR
    with gzip.open(directory / expression.GTEX.name) as handle:
        gtex = pl.read_csv(handle.read(), separator="\t", skip_rows=2)
    gtex = {
        row["Name"].split(".")[0]: row
        for row in gtex.iter_rows(named=True)
        if "PAR_Y" not in row["Name"]
    }
    whole = [
        c for c in next(iter(gtex.values()))
        if c not in ("Name", "Description") and not c.startswith("Cells_")
        and c not in expression.GTEX_NOT_WHOLE
    ]  # fmt: skip
    vital = [c for group in expression.VITAL for c in expression.TISSUES[group][0]]
    with zipfile.ZipFile(directory / expression.HPA.name) as archive:
        long = pl.read_csv(io.BytesIO(archive.read(archive.namelist()[0])), separator="\t")
    hpa: dict[str, dict[str, float]] = defaultdict(dict)
    for gene, tissue, ntpm in long.select("Gene", "Tissue", "nTPM").iter_rows():
        hpa[gene][tissue] = ntpm
    out = {}
    for nct, genes in sample.items():
        taus, vitals, counts, disease, top3 = [], [], [], [], []
        for gene in genes:
            if gene in gtex:
                values = [math.log2(gtex[gene][c] + 1) for c in whole]
                if max(gtex[gene][c] for c in whole) >= expression.EXPRESSED:
                    peak = max(values)
                    taus.append(sum(1 - v / peak for v in values) / (len(values) - 1))
                vitals.append(math.log2(max(gtex[gene][c] for c in vital) + 1))
            if gene in hpa:
                counts.append(sum(v >= expression.EXPRESSED for v in hpa[gene].values()))
                for group in tissues.get(nct, []):
                    members = expression.TISSUES[group][1]
                    disease.append(math.log2(max(hpa[gene].get(t, 0.0) for t in members) + 1))
                    rank = min(
                        1 + sum(w > hpa[gene].get(t, 0.0) for w in hpa[gene].values())
                        for t in members
                    )
                    expressed = max(hpa[gene].get(t, 0.0) for t in members) >= expression.EXPRESSED
                    top3.append(expressed and rank <= expression.TOP_RANKS)
        out[nct] = {
            "expr_tau": min(taus) if taus else None,
            "expr_n_tissues": max(counts) if counts else None,
            "expr_vital_max": max(vitals) if vitals else None,
            "expr_disease_tissue": max(disease) if disease else None,
            "expr_disease_tissue_top3": any(top3) if disease else None,
        }
    return out


# ---------------------------------------------------------------- networks


def network_values(sample: dict[str, list[str]]) -> dict[str, dict]:
    """Viability and other hit fractions from ORCS, partners per paper from BioGRID."""
    directory = networks.NETWORKS_DIR
    mapping = tg.entrez_to_ensembl(tg.read_hgnc())
    entrez_of = dict(zip(mapping["ensembl_gene_id"], mapping["entrez_id"], strict=True))
    wanted = {entrez_of[g] for genes in sample.values() for g in genes if g in entrez_of}
    # ORCS: for every screen, whether each wanted gene was a hit (the index may come last).
    hit_in: dict[int, dict[int, bool]] = defaultdict(dict)
    group_of: dict[int, str] = {}
    with tarfile.open(directory / networks.ORCS.name, "r:gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if name == networks.INDEX_MEMBER:
                index = pl.read_csv(
                    archive.extractfile(member).read(),
                    separator="\t",
                    infer_schema_length=0,
                    quote_char=None,
                )
                group_of = screen_group_of(index)
            elif ".screen.tab.txt" in name and member.isfile():
                table = pl.read_csv(
                    archive.extractfile(member).read(),
                    separator="\t",
                    infer_schema_length=0,
                    quote_char=None,
                )
                screen = int(table["#SCREEN_ID"][0])
                for row in table.iter_rows(named=True):
                    if row["IDENTIFIER_TYPE"] != "ENTREZ_GENE":
                        continue
                    gene = int(row["IDENTIFIER_ID"])
                    if gene in wanted:
                        hit_in[screen][gene] = (
                            hit_in[screen].get(gene, False) or row["HIT"] == "YES"
                        )
    hits: dict[tuple[int, str], dict[int, bool]] = {
        (screen, group_of[screen]): genes for screen, genes in hit_in.items() if screen in group_of
    }
    fractions: dict[int, dict[str, float | None]] = {}
    for gene in wanted:
        fractions[gene] = {}
        for group, minimum in (
            (networks.VIABILITY, networks.MIN_VIABILITY_SCREENS),
            (networks.OTHER, networks.MIN_OTHER_SCREENS),
        ):
            results = [v[gene] for (s, g), v in hits.items() if g == group and gene in v]
            fractions[gene][group] = (
                sum(results) / len(results) if len(results) >= minimum else None
            )
    # BioGRID: distinct partners and papers for the wanted genes, self-pairs dropped.
    columns = [
        "Entrez Gene Interactor A", "Entrez Gene Interactor B", "Experimental System Type",
        "Publication Source", "Organism ID Interactor A", "Organism ID Interactor B",
    ]  # fmt: skip
    with (
        zipfile.ZipFile(directory / networks.BIOGRID.name) as archive,
        archive.open(networks.HUMAN_MEMBER) as handle,
    ):
        biogrid = pl.read_csv(
            handle.read(), separator="\t", quote_char=None, infer_schema_length=0, columns=columns
        )
    physical = biogrid.filter(
        (pl.col("Experimental System Type") == "physical")
        & (pl.col("Organism ID Interactor A") == "9606")
        & (pl.col("Organism ID Interactor B") == "9606")
    )
    partners: dict[int, set] = defaultdict(set)
    papers: dict[int, set] = defaultdict(set)
    for a, b, paper in physical.select(
        "Entrez Gene Interactor A", "Entrez Gene Interactor B", "Publication Source"
    ).iter_rows():
        if a == b:
            continue
        for gene, other in ((a, b), (b, a)):
            if gene.isdigit() and other.isdigit() and int(gene) in wanted:
                partners[int(gene)].add(other)
                papers[int(gene)].add(paper)
    out = {}
    for nct, genes in sample.items():
        known = [entrez_of[g] for g in genes if g in entrez_of]
        per_paper = [math.log2(len(partners[e]) / len(papers[e])) for e in known if partners.get(e)]
        via = [
            fractions[e][networks.VIABILITY]
            for e in known
            if fractions[e][networks.VIABILITY] is not None
        ]
        other = [
            fractions[e][networks.OTHER] for e in known if fractions[e][networks.OTHER] is not None
        ]
        out[nct] = {
            "net_viability_hit_fraction": max(via) if via else None,
            "net_other_hit_fraction": max(other) if other else None,
            "net_partners_per_paper": max(per_paper) if per_paper else None,
        }
    return out


def screen_group_of(index: pl.DataFrame) -> dict[int, str]:
    """Screen id -> "viability" or "other", written out screen by screen (see networks)."""
    result = {}
    for row in index.iter_rows(named=True):
        if row["LIBRARY_METHODOLOGY"] != "Knockout" or row["FULL_SIZE_AVAILABLE"] != "Yes":
            continue
        growth_phenotype = row["PHENOTYPE"] in networks.VIABILITY_PHENOTYPES
        family = row["AUTHOR"] in networks.DEPMAP_FAMILY_AUTHORS or row["ANALYSIS"] == "CERES"
        if (
            growth_phenotype
            and row["EXPERIMENTAL_SETUP"] == "Timecourse"
            and row["SCREEN_TYPE"] in networks.LOSS_SELECTION
            and not family
        ):
            result[int(row["#SCREEN_ID"])] = networks.VIABILITY
        elif not growth_phenotype:
            result[int(row["#SCREEN_ID"])] = networks.OTHER
    return result


# ---------------------------------------------------------------- annotations and chemistry


def annotation_values(sample: dict[str, list[str]]) -> dict[str, dict]:
    directory = annotations.ANNOTATIONS_DIR
    proteins = {
        r["id"]: r
        for r in pl.read_parquet(directory / annotations.TARGET_SLIM).iter_rows(named=True)
    }
    tolerance = {
        r["targetId"]: r
        for r in pl.read_parquet(directory / annotations.PRIORITISATION_SLIM).iter_rows(named=True)
    }
    out = {}
    for nct, genes in sample.items():
        classes, places, constraint, knockout = set(), set(), [], []
        for gene in genes:
            row = proteins.get(gene)
            if row:
                labels = {c["label"] for c in row["targetClass"] or [] if c["level"] == "l1"}
                if labels:
                    classes.add("multiple" if len(labels) > 1 else next(iter(labels)))
                texts = [s["location"].lower() for s in row["subcellularLocations"] or []]
                if texts:
                    firsts = [t.split(";")[0].strip() for t in texts]
                    surface = any(SURFACE.search(f) for f in firsts)
                    if surface:
                        places.add("membrane")
                    elif any(t.startswith("secreted") for t in texts):
                        places.add("secreted")
                    elif any(re.search(r"\bnucle", t) for t in texts):
                        places.add("nuclear")
                    else:
                        places.add("intracellular")
            if gene in tolerance:
                if tolerance[gene]["geneticConstraint"] is not None:
                    constraint.append(-tolerance[gene]["geneticConstraint"])
                if tolerance[gene]["mouseKOScore"] is not None:
                    knockout.append(-tolerance[gene]["mouseKOScore"])
        out[nct] = {
            "target_class": ("multiple" if len(classes) > 1 else next(iter(classes)))
            if classes
            else None,
            "target_location": ("multiple" if len(places) > 1 else next(iter(places)))
            if places
            else None,
            "constraint": max(constraint) if constraint else None,
            "mouse_knockout": max(knockout) if knockout else None,
        }
    return out


def chemistry_values(leads: dict[str, str | None]) -> dict[str, dict]:
    directory = chemistry.CHEMISTRY_DIR
    properties = {
        r["drug_id"]: r for r in pl.read_csv(directory / chemistry.PROPERTIES).iter_rows(named=True)
    }
    actions: dict[str, set] = defaultdict(set)
    for row in pl.read_parquet(directory / chemistry.MECHANISM_SLIM).iter_rows(named=True):
        for drug in row["chemblIds"] or []:
            if row["actionType"]:
                actions[drug].add(chemistry.ACTION_GROUPS.get(row["actionType"], chemistry.OTHER))
    out = {}
    for nct, drug in leads.items():
        row = properties.get(drug, {})
        group = actions.get(drug, set())
        values = {f"chem_{name}": row.get(name) for name in chemistry.CHEMBL_FIELDS.values()}
        values["chem_action_type"] = (
            ("multiple" if len(group) > 1 else next(iter(group))) if group else None
        )
        out[nct] = values
    return out


# ---------------------------------------------------------------- run


def run(
    table: pl.DataFrame,
    drugs: pl.DataFrame,
    targets: pl.DataFrame,
    conditions: pl.DataFrame,
    n: int = SAMPLE,
    seed: int = SEED,
) -> list[Check]:
    """Draw `n` mapped development trials and recompute their values from the raw files.

    `drugs`, `targets` and `conditions` are the trial joins of `targets.build_inputs()`.
    """
    chosen = table.filter((pl.col("split") == "dev") & pl.col("has_target")).sample(n, seed=seed)
    rows = {r["nct_id"]: r for r in chosen.iter_rows(named=True)}
    ids = list(rows)
    sample = {
        nct: sorted(targets.filter(pl.col("nct_id") == nct)["targetId"].unique()) for nct in ids
    }
    area = tg.load_trials().select("nct_id", "therapeutic_area")
    info = {
        r["nct_id"]: r
        for r in conditions.join(area, on="nct_id")
        .filter(pl.col("nct_id").is_in(ids))
        .iter_rows(named=True)
    }
    lineages = {
        nct: depmap.lineages_for(r["mesh"], r["conditions"])
        if r["therapeutic_area"] == "oncology"
        else []
        for nct, r in info.items()
    }
    tissues = {
        nct: expression.disease_tissues(
            r["mesh"], r["ancestors"], r["conditions"], r["therapeutic_area"]
        )
        for nct, r in info.items()
    }
    pairs = depmap.tumour_intrinsic(
        targets.filter(pl.col("nct_id").is_in(ids)),
        chemistry.target_actions(),
        depmap.host_directed_ids(tg.read_hgnc()),
    )
    intrinsic = {nct: sorted(pairs.filter(pl.col("nct_id") == nct)["targetId"]) for nct in ids}
    leads = {
        nct: next(iter(drugs.filter((pl.col("nct_id") == nct) & pl.col("is_lead"))["drugId"]), None)
        for nct in ids
    }
    return [
        compare("DepMap", rows, depmap_values(sample, lineages, intrinsic)),
        compare("expression", rows, expression_values(sample, tissues)),
        compare("networks", rows, network_values(sample)),
        compare("annotations", rows, annotation_values(sample)),
        compare("chemistry", rows, chemistry_values(leads)),
    ]


def render(checks: list[Check]) -> str:
    return "\n".join(
        f"{c.source}: {c.agreed} of {c.values} values agree ({c.present} not missing)"
        + "".join(f"\n  {m}" for m in c.mismatches)
        for c in checks
    )


def main() -> None:
    from elman_exercise.science import table as science_table

    print(render(run(pl.read_parquet(science_table.OUT_PATH), *tg.build_inputs())))


if __name__ == "__main__":
    main()
