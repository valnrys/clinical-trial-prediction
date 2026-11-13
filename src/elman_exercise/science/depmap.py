"""Does the cancer-cell screen say the target is essential? DepMap 24Q4.

    uv run python -m elman_exercise.science.depmap    # download, then summarise the two matrices

DepMap (https://depmap.org, release 24Q4, CC BY 4.0, via figshare) knocks out each gene in
about 1,100 cancer cell lines with CRISPR and scores how much the line grows less. We use
two gene-by-cell-line matrices (about 430 MB each): `CRISPRGeneEffect` (Chronos score;
0 is no effect, -1 is the typical effect of a known essential gene) and
`CRISPRGeneDependency` (the probability the line depends on the gene), plus DepMap's own
list of common essential genes. The matrices are read one cell line at a time and reduced
to per-gene and per-lineage summaries; nothing larger than one row is held in memory.

Features, one row per trial with an investigational target found in DepMap. Several targets:

- `depmap_common_essential`: any target is on DepMap's common-essential list. It says that
  cells in culture need the target gene. It does not say the drug is toxic, or that the
  trial's disease depends on the gene. On these trials it mostly flags tubulin binders,
  metformin (complex I subunits) and statins (HMGCR): drugs whose target every cell needs.
- `depmap_mean_effect`: mean gene effect over all lines, for the most dependent target (the
  one with the lowest mean effect).
- `depmap_fraction_dependent`: share of lines (with a dependency probability) whose probability
  is above 0.5, for that same target.

Open Targets expands a drug's target to whole protein complexes (pregabalin has 26 calcium-channel
subunits), so "the most dependent target" is a minimum over many genes for such drugs, and the
more targets a drug has, the lower that minimum tends to be (`science` says more on this).

Cancer trials only (the project's `oncology` area), when the registry's conditions map to one to three
DepMap lineages (the rules in `ONCOLOGY_RULES`, MeSH terms first, then the condition text), and
not a supportive-care trial in cancer patients (pain, nausea, mucositis ...):

- `depmap_lineage_effect`: mean gene effect in the matched lineages' cell lines, for the
  target with the lowest such mean. Needs at least five matched lines.
- `depmap_lineage_minus_other`: that same target's matched-lineage mean minus its mean in
  all other lines (negative: more dependent in the trial's cancer than elsewhere).

These two read tumour-intrinsic targets only: a target the drug turns down (Open Targets'
action is an inhibitor, antagonist, blocker ...) that is not on `HOST_DIRECTED`, a list written
before any outcome was examined of targets on the host rather than the tumour cell (immune
checkpoints, VEGF and its receptors, hormone axes, bone and symptom targets, antibody
antigens). A cell line cannot depend on those, so their value is about zero by construction.
A trial with no such target gets no lineage value.

Limits. A lineage is a coarse match: "melanoma" matches every skin line. A cancer lineage
is tested against all other lineages, which are mostly different cancers. The cell lines are
a cancer-cell view, with no immune system, blood vessels or hormones.
The gene scores are TIMELESS properties of the gene; the lineage match reads the trial's
own conditions, so those two are PROTOCOL.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import polars as pl

from elman_exercise.dating import PROTOCOL, TIMELESS
from elman_exercise.science import targets as tg
from elman_exercise.science.fetch import RemoteFile, fetch_all

DEPMAP_DIR = tg.RAW / "depmap"
RELEASE = "DepMap Public 24Q4"
LICENCE = "CC BY 4.0 (Broad Institute DepMap)"
FIGSHARE = "https://ndownloader.figshare.com/files/"
EFFECT = RemoteFile(
    "CRISPRGeneEffect.csv", FIGSHARE + "51064667", 428678699,
    "3d8f3ec6dbf2db7ff834b79b508622ec0b226f3518003fe96ecf5a4fcf167e3b",
)  # fmt: skip
DEPENDENCY = RemoteFile(
    "CRISPRGeneDependency.csv", FIGSHARE + "51064631", 421115594,
    "76f2df90bc2a0d70bac667dda83a052d04a1220cdc70e01ce0f6508d59ebea40",
)  # fmt: skip
COMMON_ESSENTIAL = RemoteFile(
    "CRISPRInferredCommonEssentials.csv", FIGSHARE + "51064916", 20795,
    "0352de5335ae3bf72250dd136d1b0df87a6b72f653df34c2560cce17c0e7d928",
)  # fmt: skip
MODEL = RemoteFile(
    "Model.csv", FIGSHARE + "51065297", 645696,
    "b7a0c1385e6cef30132b56aff61f1261d11e3f490490b355c430d32ee0dbdcfa",
)  # fmt: skip
FILES = [EFFECT, DEPENDENCY, COMMON_ESSENTIAL, MODEL]
GENE_SUMMARY = "gene_summary.parquet"
LINEAGE_SUMMARY = "lineage_summary.parquet"

FEATURES: dict[str, str] = {
    "depmap_common_essential": TIMELESS,
    "depmap_mean_effect": TIMELESS,
    "depmap_fraction_dependent": TIMELESS,
    "depmap_lineage_effect": PROTOCOL,
    "depmap_lineage_minus_other": PROTOCOL,
}
DEPENDENT = 0.5  # dependency probability above this counts as "dependent"
MIN_LINES = 5  # matched lineages need at least this many cell lines
MAX_LINEAGES = 3  # a condition that maps to more lineages than this is too vague to match

# ---------------------------------------------------------------- condition -> lineage
# (name, regex on lower-cased text, DepMap OncotreeLineage names). MeSH terms are tried first and
# the registry's condition text second, only if no MeSH term matched. Written before looking at
# any outcome; read it as the whole decision.
GENERIC_BLOOD = "leukaemia_any"
ONCOLOGY_RULES = [
    ("breast", r"breast|mammary", ["Breast"]),
    ("lung", r"lung", ["Lung"]),
    ("mesothelioma", r"mesothelioma", ["Pleura"]),
    ("prostate", r"prostat", ["Prostate"]),
    ("colorectal", r"colorectal|colonic|colon\b|rectal|rectum|anus|anal\b|bowel|appendic", ["Bowel"]),
    ("oesophagogastric", r"stomach|gastric|gastro-?oesophageal|gastroesophageal|esophag|oesophag", ["Esophagus/Stomach"]),
    ("pancreas", r"pancrea", ["Pancreas"]),
    ("ovary", r"ovarian|ovary|fallopian|peritoneal neoplasm|primary peritoneal", ["Ovary/Fallopian Tube"]),
    ("uterus", r"endometri|uterine neoplasm|uterine cancer|uterine carcinoma|uterine sarcoma|carcinosarcoma", ["Uterus"]),
    ("cervix", r"cervical cancer|cervical neoplasm|uterine cervical neoplasm|cervix|cervical carcinoma", ["Cervix"]),
    ("vulva_vagina", r"vulva|vagin", ["Vulva/Vagina"]),
    ("skin", r"basal cell|merkel|skin neoplasm|skin cancer|cutaneous squamous", ["Skin"]),
    ("head_neck", r"head and neck|oropharyn|laryn|nasopharyn|oral cancer|mouth neoplasm|tongue|hypopharyn|salivary|pharyngeal|oral cavity|oral squamous", ["Head and Neck"]),
    ("brain", r"glioblastoma|glioma|gliosarcoma|astrocytoma|brain neoplasm|brain tumou?r|central nervous system neoplasm|medulloblastoma|ependymoma|oligodendroglioma|meningioma|cns tumou?r", ["CNS/Brain"]),
    ("neuroblastoma", r"neuroblastoma", ["Peripheral Nervous System"]),
    ("kidney", r"renal cell|kidney neoplasm|kidney cancer|wilms|renal carcinoma|renal cancer", ["Kidney"]),
    ("bladder", r"bladder|urothelial|transitional cell|urinary tract neoplasm|ureter", ["Bladder/Urinary Tract"]),
    ("liver", r"hepatocellular|liver neoplasm|liver cancer|hepatoma|hepatic carcinoma", ["Liver"]),
    ("biliary", r"cholangio|biliary|bile duct|gallbladder", ["Biliary Tract"]),
    ("thyroid", r"thyroid", ["Thyroid"]),
    ("bone", r"osteosarcoma|osteogenic sarcoma|ewing|bone neoplasm|chondrosarcoma|bone cancer|chordoma", ["Bone"]),
    ("soft_tissue", r"liposarcoma|leiomyosarcoma|rhabdomyosarcoma|soft tissue|gastrointestinal stromal|synovial sarcoma|fibrosarcoma|rhabdoid", ["Soft Tissue"]),
    ("eye", r"uveal|retinoblastoma|eye neoplasm|choroid|ocular melanoma|conjunctival melanoma", ["Eye"]),
    ("testis", r"testic|testis|germ cell", ["Testis"]),
    ("myeloma", r"myeloma|plasma cell|plasmacytoma|waldenstr", ["Lymphoid"]),
    ("lymphoid", r"lymphoma|hodgkin|lymphocytic leuk|lymphoblastic|burkitt|mycosis fungoides|sezary|lymphoproliferative|hairy cell|leukemia, (lymphocytic|lymphoid|[bt]-cell|prolymphocytic)", ["Lymphoid"]),
    ("myeloid", r"myeloid|myelogenous|myeloblastic|myelodysplas|myeloproliferative|myelofibrosis|polycythemia vera|essential thrombocythemia|thrombocythemia, essential|monocytic|erythroblastic|megakaryoblastic|myelomonocytic|promyelocytic", ["Myeloid"]),
    (GENERIC_BLOOD, r"^leukemia$|leukaemia|hematologic neoplasm|haematolog.* malig|hematologic malig|blood cancer", ["Lymphoid", "Myeloid"]),
]  # fmt: skip
# Acronyms, matched case-sensitively on the original text so that the English word "all" and
# "gist" do not count. "ALL" is also skipped when a capitalised phrase follows that makes it the
# word ("ALL SOLID TUMORS", "ALL STAGES"). Each also counts as a cancer word.
ACRONYM_RULES = [
    ("lung", r"\b(NSCLC|SCLC)\b", ["Lung"]),
    ("liver", r"\bHCC\b", ["Liver"]),
    ("kidney", r"\bRCC\b", ["Kidney"]),
    ("soft_tissue", r"\bGIST\b", ["Soft Tissue"]),
    ("lymphoid", r"\bCLL\b|\bALL\b(?!\s+(?i:solid|stages?|types?|cancers?|tumou?rs?|comers|patients|grades?))",
     ["Lymphoid"]),
    ("myeloid", r"\b(AML|CML|MDS)\b", ["Myeloid"]),
]  # fmt: skip
# Bare "sarcoma" and "melanoma" count only when no other rule matched the same text, so that
# Ewing, uterine, osteogenic or myeloid sarcoma and uveal melanoma keep their own lineage.
BARE_RULES = [
    ("soft_tissue", r"\bsarcoma\b", ["Soft Tissue"]),
    ("skin", r"melanoma", ["Skin"]),
]
# A text that looks like a cancer word but names a benign or pre-cancerous growth.
NOT_A_TUMOUR = re.compile(
    r"keratosis|precancer|(?<!myelo)dysplasia|condylom|leiomyoma|polycystic|fibroid|myoma"
    r"|hyperplasia|benign|papilloma|wart|neurofibromatosis|tuberous|\bcysts?\b|haemangioma"
    r"|hemangioma",
    re.IGNORECASE,
)
# A condition that marks a supportive-care trial (the drug treats a symptom or side effect, and
# cancer is only who the patients are). Any one of them, in the MeSH terms or the condition
# text, stops the trial getting a tumour lineage.
SUPPORTIVE = re.compile(
    r"menorrhagia|hot flash|mucositis|stomatitis|cachexia|nausea|vomiting|\bpain\b"
    r"|fatigue|thrombosis|thromboembol|neurotoxic|infection|mastalgia|gynecomastia|lymphedema"
    r"|hypercalc",
    re.IGNORECASE,
)
# Blood counts that are low because of treatment are supportive care, but low counts are also
# what a myelodysplastic or myeloproliferative disease is ("Anemia, Refractory" is the MeSH name
# for one). So these veto a trial only when no myeloid condition is named beside them.
CYTOPENIA = re.compile(
    r"neutropeni|(?<!refractory )anemia(?!,\s*refractory)|(?<!refractory )anaemia|thrombocytopeni",
    re.IGNORECASE,
)
CANCER_WORD = re.compile(
    r"neoplas|cancer|carcinom|tumou?r|malignan|sarcoma|lymphoma|leuk|myeloma|melanoma|glioma"
    r"|blastoma|metasta|mesothelioma|myelodysplas|myeloproliferative|myelofibrosis|hodgkin"
    r"|waldenstr|astrocytoma|stromal|polycythemia|thrombocythemia|plasmacytoma|chordoma|ewing"
    r"|mycosis fungoides|sezary|hairy cell|meningioma|ependymoma|hepatoma|rhabdoid",
    re.IGNORECASE,
)
_RULES = [
    (name, re.compile(pattern, re.IGNORECASE), lineages)
    for name, pattern, lineages in ONCOLOGY_RULES
]
_ACRONYMS = [(name, re.compile(pattern), lineages) for name, pattern, lineages in ACRONYM_RULES]
_BARE = [
    (name, re.compile(pattern, re.IGNORECASE), lineages) for name, pattern, lineages in BARE_RULES
]


def _matches(texts: list[str]) -> list[tuple[str, list[str]]]:
    hits = []
    for text in texts:
        if NOT_A_TUMOUR.search(text):
            continue
        found = [(name, lineages) for name, pattern, lineages in _RULES if pattern.search(text)]
        found += [(name, lineages) for name, pattern, lineages in _ACRONYMS if pattern.search(text)]
        if not found and CANCER_WORD.search(text):
            found = [(name, lineages) for name, pattern, lineages in _BARE if pattern.search(text)]
        if found and (CANCER_WORD.search(text) or any(p.search(text) for _, p, _ in _ACRONYMS)):
            hits += found
    return hits


def _lineages(hits: list[tuple[str, list[str]]]) -> list[str]:
    return sorted({lineage for _, lineages in hits for lineage in lineages})


def _supportive(texts: list[str], found: list[list[tuple[str, list[str]]]]) -> bool:
    """True if the trial is about a symptom or side effect, not the cancer."""
    if any(SUPPORTIVE.search(text) for text in texts):
        return True
    myeloid = any(name == "myeloid" for hits in found for name, _ in hits)
    return not myeloid and any(CYTOPENIA.search(text) for text in texts)


def lineages_for(mesh: list[str], conditions: list[str]) -> list[str]:
    """DepMap lineages for a trial's cancer, from its MeSH terms, else its condition text.

    Empty for a supportive-care trial (`SUPPORTIVE`, and `CYTOPENIA` unless a myeloid condition
    is named) and when nothing matches. A generic "leukemia" term is used only when no specific
    term in either place names a cancer, so it never widens a specific condition.
    """
    found = [_matches(texts) for texts in (mesh, conditions)]
    if _supportive([*mesh, *conditions], found):
        return []
    for hits in found:
        specific = [h for h in hits if h[0] != GENERIC_BLOOD]
        if specific:
            return _lineages(specific)
    return _lineages(found[0] or found[1])


# ---------------------------------------------------------------- host-directed targets
# Targets of drugs that act on the patient rather than on the tumour cell. Symbols are used
# here only to pick the genes (through HGNC, to Ensembl ids); nothing is joined on a symbol.
HOST_DIRECTED = {
    "immune checkpoint and co-stimulation, immune cell surface": (
        "PDCD1 CD274 PDCD1LG2 CTLA4 LAG3 TIGIT HAVCR2 CD27 TNFRSF9 TNFRSF4 TNFRSF18 ICOS CD40 CD40LG "
        "CD47 SIRPA IDO1 CD28 VTCN1 CD80 CD86 CD70 CSF1R CCR4 CCR8 CD3E CD3D CD3G IL2RA IL2RB IL15 "
        "IL2 IL7R IFNAR1 IFNAR2 TLR7 TLR8 TLR9 TLR4 STING1 CXCR4 CCR5 SIGLEC15 CD96 TNFRSF1B C5AR1 "
        "CSF2RA CSF3R CSF2 CSF3 EPOR"
    ),
    "angiogenesis": (
        "VEGFA VEGFB VEGFC VEGFD KDR FLT1 FLT4 ANGPT1 ANGPT2 TEK PGF NRP1 NRP2 DLL4 PDGFB"
    ),
    "hormone axis": (
        "ESR1 ESR2 AR CYP19A1 CYP17A1 SRD5A1 SRD5A2 GNRHR GNRH1 PGR NR3C1 CYP11A1 SHBG GPER1 "
        "LHCGR FSHR SSTR2 SSTR5 TSHR"
    ),
    "bone, symptom and supportive-care targets": (
        "TNFSF11 TNFRSF11A TNFRSF11B TACR1 HTR3A HTR3B DRD2 HRH1 HRH2 PTH1R CALCR ALOX5 PTGS1 "
        "PTGS2 OPRM1 OPRD1 OPRK1 SLC12A1 SLC12A3 SCN9A CACNA2D1"
    ),
    "antibody and cell-therapy antigens": (
        "MS4A1 CD19 CD38 CD22 CD33 CD79B CD79A TNFRSF17 SLAMF7 CD52 TNFRSF8 FCGR3A FCGR2B CD74 "
        "CD37 CD123 IL3RA CD7 CD5 GPRC5D FCRL5 CD276 CD24 EPCAM TACSTD2 MSLN FOLR1 CLDN18 CEACAM5 "
        "DLL3 NECTIN4 TNFRSF10A TNFRSF10B CLDN6 ROR1 FOLH1 STEAP1 KLK3 GUCY2C"
    ),
}
# Open Targets action types for a drug that turns its target's activity down.
LOWERING_ACTIONS = {
    "INHIBITOR", "ANTAGONIST", "BLOCKER", "NEGATIVE ALLOSTERIC MODULATOR", "INVERSE AGONIST",
    "DEGRADER", "DISRUPTING AGENT", "ANTISENSE INHIBITOR", "SEQUESTERING AGENT",
    "NEUTRALISING ANTIBODY", "NEGATIVE MODULATOR", "RNAI INHIBITOR", "ALLOSTERIC ANTAGONIST",
    "BINDING AGENT", "CROSS-LINKING AGENT",
}  # fmt: skip


def host_directed_ids(hgnc: pl.DataFrame) -> list[str]:
    """Ensembl ids of the `HOST_DIRECTED` genes."""
    symbols = [s for group in HOST_DIRECTED.values() for s in group.split()]
    found = hgnc.filter(pl.col("symbol").is_in(symbols)).drop_nulls("ensembl_gene_id")
    return sorted(set(found["ensembl_gene_id"]))


def tumour_intrinsic(
    targets: pl.DataFrame, actions: pl.DataFrame, host_ids: list[str]
) -> pl.DataFrame:
    """(nct_id, targetId) for targets the drug turns down that are not host-directed.

    `targets` has nct_id, drugId, targetId; `actions` is (drugId, targetId, actionType).
    """
    lowering = (
        actions.filter(pl.col("actionType").is_in(sorted(LOWERING_ACTIONS)))
        .select("drugId", "targetId")
        .unique()
    )
    return (
        targets.join(lowering, on=["drugId", "targetId"])
        .filter(~pl.col("targetId").is_in(host_ids))
        .select("nct_id", "targetId")
        .unique()
        .sort("nct_id", "targetId")
    )


# ---------------------------------------------------------------- download and summarise


def download(directory: Path = DEPMAP_DIR) -> None:
    fetch_all(directory, RELEASE, LICENCE, FILES)
    if not (directory / GENE_SUMMARY).exists():
        genes, lineages = summarise(
            directory / EFFECT.name, directory / DEPENDENCY.name, directory / MODEL.name
        )
        genes.write_parquet(directory / GENE_SUMMARY)
        lineages.write_parquet(directory / LINEAGE_SUMMARY)


def entrez_of(header: str) -> int:
    """Entrez id from a DepMap column name such as "TP53 (7157)"."""
    return int(header.rsplit("(", 1)[1].rstrip(")"))


def _rows(path: Path):
    """(model id, values) for each cell line of a DepMap matrix; empty cells are NaN."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        yield [entrez_of(h) for h in header[1:]], None
        for row in reader:
            yield row[0], np.array([x or "nan" for x in row[1:]], dtype=float)


def summarise(
    effect_path: Path, dependency_path: Path, model_path: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reduce the two matrices to a per-gene table and a per-gene-and-lineage table.

    Gene table: entrez_id, n_lines (lines with an effect score), mean_effect, fraction_dependent
    (of the lines with a dependency probability). Lineage table: lineage, entrez_id, n_lines,
    sum_effect (the sum lets the caller average over several lineages, and over "all other"
    lines, exactly). A model with no lineage is kept as "Unknown". Models missing from
    Model.csv stop the run.
    """
    models = pl.read_csv(model_path, infer_schema_length=0).select("ModelID", "OncotreeLineage")
    models = models.with_columns(pl.col("OncotreeLineage").fill_null("Unknown"))
    lineage_of = dict(zip(models["ModelID"], models["OncotreeLineage"], strict=True))
    effect_rows, dependency_rows = _rows(effect_path), _rows(dependency_path)
    entrez, _ = next(effect_rows)
    other_entrez, _ = next(dependency_rows)
    if entrez != other_entrez:
        raise ValueError("The effect and dependency matrices list different genes.")
    n_genes = len(entrez)
    count, total = np.zeros(n_genes), np.zeros(n_genes)
    scored, dependent = np.zeros(n_genes), np.zeros(n_genes)  # lines with a probability; above 0.5
    by_lineage: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (model, effect), (other, probability) in zip(effect_rows, dependency_rows, strict=True):
        if model != other:
            raise ValueError(f"Row order differs between the matrices at {model} and {other}.")
        lineage = lineage_of[model]
        seen = ~np.isnan(effect)
        count += seen
        total += np.where(seen, effect, 0.0)
        scored += ~np.isnan(probability)
        dependent += np.nan_to_num(probability) > DEPENDENT
        n_line, sum_line = by_lineage.setdefault(lineage, (np.zeros(n_genes), np.zeros(n_genes)))
        n_line += seen
        sum_line += np.where(seen, effect, 0.0)
    genes = pl.DataFrame(
        {
            "entrez_id": entrez,
            "n_lines": count.astype(int),
            "mean_effect": np.where(count > 0, total / np.maximum(count, 1), np.nan),
            "fraction_dependent": np.where(scored > 0, dependent / np.maximum(scored, 1), np.nan),
        }
    ).fill_nan(None)
    lineages = pl.concat(
        pl.DataFrame(
            {"lineage": name, "entrez_id": entrez, "n_lines": n.astype(int), "sum_effect": s}
        )
        for name, (n, s) in sorted(by_lineage.items())
    )
    return genes, lineages


def common_essential_ids(path: Path) -> list[int]:
    """Entrez ids of DepMap's inferred common-essential genes."""
    return [entrez_of(h) for h in pl.read_csv(path)["Essentials"].to_list()]


# ---------------------------------------------------------------- per gene and per trial


def gene_table(directory: Path = DEPMAP_DIR, mapping: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per Ensembl gene: n_lines, mean_effect, fraction_dependent, common_essential."""
    if mapping is None:
        mapping = tg.entrez_to_ensembl(tg.read_hgnc())
    essential = set(common_essential_ids(directory / COMMON_ESSENTIAL.name))
    genes = pl.read_parquet(directory / GENE_SUMMARY).with_columns(
        pl.col("entrez_id").is_in(list(essential)).alias("common_essential")
    )
    return tg.with_ensembl(genes, "entrez_id", mapping).drop_nulls("ensembl_gene_id")


def lineage_table(
    directory: Path = DEPMAP_DIR, mapping: pl.DataFrame | None = None
) -> pl.DataFrame:
    if mapping is None:
        mapping = tg.entrez_to_ensembl(tg.read_hgnc())
    lineages = pl.read_parquet(directory / LINEAGE_SUMMARY)
    return tg.with_ensembl(lineages, "entrez_id", mapping).drop_nulls("ensembl_gene_id")


def gene_features(targets: pl.DataFrame, genes: pl.DataFrame) -> pl.DataFrame:
    """The three gene-level features per trial (aggregation rules in the module docstring)."""
    joined = (
        targets.select("nct_id", "targetId")
        .unique()
        .join(genes, left_on="targetId", right_on="ensembl_gene_id")
    )
    return (
        joined.sort("mean_effect", "targetId", nulls_last=True)
        .group_by("nct_id", maintain_order=True)
        .agg(
            pl.col("common_essential").any().alias("depmap_common_essential"),
            pl.col("mean_effect").first().alias("depmap_mean_effect"),
            pl.when(pl.col("mean_effect").first().is_not_null())
            .then(pl.col("fraction_dependent").first())
            .alias("depmap_fraction_dependent"),
        )
    )


def lineage_features(
    targets: pl.DataFrame,
    trial_lineages: pl.DataFrame,
    genes: pl.DataFrame,
    lineages: pl.DataFrame,
) -> pl.DataFrame:
    """The two matched-lineage features per trial.

    `targets` is (nct_id, targetId): the tumour-intrinsic targets. `trial_lineages` has nct_id
    and a list column `lineages` (cancer trials only). Per
    (trial, target): matched mean = sum over matched lineages / lines; other mean = (all
    lines' sum minus matched sum) / (all lines minus matched lines). The trial takes the target
    with the lowest matched mean.
    """
    pairs = targets.select("nct_id", "targetId").unique()
    matched_lines = (
        trial_lineages.filter(pl.col("lineages").list.len().is_between(1, MAX_LINEAGES))
        .explode("lineages")
        .rename({"lineages": "lineage"})
        .join(pairs, on="nct_id")
        .join(lineages, left_on=["targetId", "lineage"], right_on=["ensembl_gene_id", "lineage"])
        .group_by("nct_id", "targetId")
        .agg(
            pl.col("n_lines").sum().alias("n_matched"),
            pl.col("sum_effect").sum().alias("sum_matched"),
        )
        .filter(pl.col("n_matched") >= MIN_LINES)
        .join(
            genes.select(
                pl.col("ensembl_gene_id").alias("targetId"),
                pl.col("n_lines").alias("n_all"),
                (pl.col("mean_effect") * pl.col("n_lines")).alias("sum_all"),
            ),
            on="targetId",
        )
    )
    with_means = matched_lines.with_columns(
        (pl.col("sum_matched") / pl.col("n_matched")).alias("matched_mean"),
        (
            (pl.col("sum_all") - pl.col("sum_matched")) / (pl.col("n_all") - pl.col("n_matched"))
        ).alias("other_mean"),
    ).filter(pl.col("n_all") > pl.col("n_matched"))
    return (
        with_means.sort("matched_mean", "targetId")
        .group_by("nct_id", maintain_order=True)
        .agg(
            pl.col("matched_mean").first().alias("depmap_lineage_effect"),
            (pl.col("matched_mean") - pl.col("other_mean"))
            .first()
            .alias("depmap_lineage_minus_other"),
        )
    )


def trial_features(
    targets: pl.DataFrame,
    intrinsic: pl.DataFrame,
    trials: pl.DataFrame,
    conditions: pl.DataFrame,
    genes: pl.DataFrame,
    lineages: pl.DataFrame,
) -> pl.DataFrame:
    """One row per trial with a target in DepMap. `trials` has nct_id and therapeutic_area.

    The gene-level features read every target; the lineage features read the `intrinsic`
    (nct_id, targetId) pairs only.
    """
    cancer = trials.filter(pl.col("therapeutic_area") == "oncology").join(conditions, on="nct_id")
    mapped = cancer.with_columns(
        pl.struct("mesh", "conditions")
        .map_elements(
            lambda row: lineages_for(row["mesh"], row["conditions"]),
            return_dtype=pl.List(pl.String),
        )
        .alias("lineages")
    ).select("nct_id", "lineages")
    return (
        gene_features(targets, genes)
        .join(lineage_features(intrinsic, mapped, genes, lineages), on="nct_id", how="left")
        .sort("nct_id")
    )


def main() -> None:
    tg.download_hgnc()
    download()
    print(gene_table().describe())


if __name__ == "__main__":
    main()
