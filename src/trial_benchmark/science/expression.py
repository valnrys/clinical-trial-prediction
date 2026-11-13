"""Is the target switched on in healthy tissue, and in the tissue the disease affects?

    uv run python -m trial_benchmark.science.expression    # download and check the two files

Two public atlases of normal tissue: GTEx v10 (median TPM per gene in 68 tissue columns,
https://gtexportal.org, open access) and the Human Protein Atlas release 25.1 (consensus RNA,
nTPM in 51 tissues, https://www.proteinatlas.org, CC BY-SA 4.0). Genes are joined on Ensembl
gene id (GTEx's version suffix is dropped).

Gene-level features, one row per trial with an investigational target (several targets:
the rule in brackets):

- `expr_tau`: tissue specificity (Yanai tau, 0 = equal everywhere, 1 = one tissue) from GTEx
  on log2(TPM + 1) across the 49 whole-tissue columns, null when no tissue reaches 1 TPM
  [lowest tau: the broadest target is the riskiest].
- `expr_n_tissues`: number of HPA tissues where the gene reaches 1 nTPM [highest].
- `expr_vital_max`: highest GTEx log2(TPM + 1) in heart, liver, kidney, brain or blood
  (the tissues where on-target toxicity usually stops drugs) [highest].

Disease-matched features (the trial's conditions are turned into tissues by `DISEASE_TISSUE`
below, a table written before any outcome was examined):

- `expr_disease_tissue`: HPA log2(nTPM + 1) in the matched tissue [the best-matched target:
  highest].
- `expr_disease_tissue_top3`: whether the matched tissue is among the gene's three highest of
  the 51 tissues, ties sharing a rank and only tissues where the gene reaches 1 nTPM counting
  [true if any target].

Open Targets expands a drug's target to whole protein complexes, so the gene-level features take
a maximum or minimum over many genes for such drugs, and the more targets, the more extreme
the value (`science` says more on this). A drug with a broad target list tends to look broadly
expressed.

Limits. The disease-to-tissue map is coarse and some diseases have no sensible tissue
(joint, bone, systemic infection): those trials get no matched value. A cancer is matched to
the normal tissue it grew from. GTEx has no bone marrow or lymph node, so "blood" there is
whole blood, and the disease-matched features use HPA, which has them. The tissue scores are
TIMELESS properties of the gene; the disease match reads the trial's conditions, so it is
PROTOCOL.
"""

from __future__ import annotations

import gzip
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

from trial_benchmark.dating import PROTOCOL, TIMELESS
from trial_benchmark.science import targets as tg
from trial_benchmark.science.fetch import RemoteFile, fetch_all

EXPRESSION_DIR = tg.RAW / "expression"
RELEASE = "GTEx v10; Human Protein Atlas 25.1"
LICENCE = "GTEx: open access; HPA: CC BY-SA 4.0"
GTEX = RemoteFile(
    "gtex_v10_median_tpm.gct.gz",
    "https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/"
    "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz",
    8846936,
    "b951fd32aae5cad48435026e07225860596a411a75efd0519e00f330b7a98294",
)
HPA = RemoteFile(
    "hpa_rna_tissue_consensus.tsv.zip",
    "https://v25.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip",
    5293680,
    "b001903da23f802c8dfbd1b6b3a8c20b831fedd4932f4bb721b06622d83ae60b",
)
FILES = [GTEX, HPA]

FEATURES: dict[str, str] = {
    "expr_tau": TIMELESS,
    "expr_n_tissues": TIMELESS,
    "expr_vital_max": TIMELESS,
    "expr_disease_tissue": PROTOCOL,
    "expr_disease_tissue_top3": PROTOCOL,
}
EXPRESSED = 1.0  # TPM or nTPM at which a gene counts as expressed in a tissue
TOP_RANKS = 3

# GTEx v10 has 68 tissue columns. The cultured cell lines and the sub-compartment columns
# (mucosa, islets, ...) are left out of tau, which uses the 49 whole-tissue columns.
GTEX_NOT_WHOLE = {
    "Colon_Transverse_Mixed_Cell", "Colon_Transverse_Mucosa", "Colon_Transverse_Muscularis",
    "Liver_Hepatocyte", "Liver_Mixed_Cell", "Liver_Portal_Tract", "Pancreas_Acini",
    "Pancreas_Islets", "Pancreas_Mixed_Cell", "Small_Intestine_Terminal_Ileum_Lymphode_Aggregate",
    "Small_Intestine_Terminal_Ileum_Mixed_Cell", "Stomach_Mixed_Cell", "Stomach_Mucosa",
    "Stomach_Muscularis",
}  # fmt: skip

# Tissue group -> (GTEx columns, HPA tissues). One table, every line visible.
TISSUES: dict[str, tuple[list[str], list[str]]] = {
    "lung": (["Lung"], ["lung"]),
    "liver": (["Liver"], ["liver"]),
    "heart": (["Heart_Left_Ventricle", "Heart_Atrial_Appendage"], ["heart muscle"]),
    "vessel": (["Artery_Aorta", "Artery_Coronary", "Artery_Tibial"], ["blood vessel", "smooth muscle"]),
    "kidney": (["Kidney_Cortex", "Kidney_Medulla"], ["kidney"]),
    "brain": (
        ["Brain_Amygdala", "Brain_Anterior_cingulate_cortex_BA24", "Brain_Caudate_basal_ganglia",
         "Brain_Cerebellar_Hemisphere", "Brain_Cerebellum", "Brain_Cortex", "Brain_Frontal_Cortex_BA9",
         "Brain_Hippocampus", "Brain_Hypothalamus", "Brain_Nucleus_accumbens_basal_ganglia",
         "Brain_Putamen_basal_ganglia", "Brain_Substantia_nigra"],
        ["amygdala", "basal ganglia", "cerebellum", "cerebral cortex", "hippocampal formation",
         "hypothalamus", "midbrain"],
    ),
    "nerve": (["Nerve_Tibial", "Brain_Spinal_cord_cervical_c-1"], ["spinal cord"]),
    "stomach": (["Stomach"], ["stomach"]),
    "small_intestine": (["Small_Intestine_Terminal_Ileum"], ["small intestine", "duodenum"]),
    "colon": (["Colon_Sigmoid", "Colon_Transverse"], ["colon", "rectum"]),
    "esophagus": (
        ["Esophagus_Gastroesophageal_Junction", "Esophagus_Mucosa", "Esophagus_Muscularis"],
        ["esophagus"],
    ),
    "pancreas": (["Pancreas"], ["pancreas"]),
    "skin": (["Skin_Not_Sun_Exposed_Suprapubic", "Skin_Sun_Exposed_Lower_leg"], ["skin"]),
    "breast": (["Breast_Mammary_Tissue"], ["breast"]),
    "prostate": (["Prostate"], ["prostate"]),
    "ovary": (["Ovary"], ["ovary"]),
    "uterus": (
        ["Uterus", "Cervix_Ectocervix", "Cervix_Endocervix", "Vagina", "Fallopian_Tube"],
        ["endometrium", "cervix", "vagina", "fallopian tube"],
    ),
    "testis": (["Testis"], ["testis"]),
    "bladder": (["Bladder"], ["urinary bladder"]),
    "thyroid": (["Thyroid"], ["thyroid gland"]),
    "adrenal": (["Adrenal_Gland"], ["adrenal gland"]),
    "pituitary": (["Pituitary"], ["pituitary gland"]),
    "muscle": (["Muscle_Skeletal"], ["skeletal muscle"]),
    "adipose": (["Adipose_Subcutaneous", "Adipose_Visceral_Omentum"], ["adipose tissue"]),
    # GTEx has no bone marrow, lymph node, thymus or tonsil: whole blood and spleen stand in.
    "immune_blood": (
        ["Whole_Blood", "Spleen"],
        ["bone marrow", "lymph node", "spleen", "thymus", "tonsil", "appendix"],
    ),
    "salivary_oral": (["Minor_Salivary_Gland"], ["salivary gland", "tongue"]),
    "eye": ([], ["retina"]),
}  # fmt: skip
VITAL = ("heart", "liver", "kidney", "brain", "immune_blood")

# ---------------------------------------------------------------- disease -> tissue
METABOLIC = ["pancreas", "liver", "muscle", "adipose"]
# MeSH term -> tissue groups. A cancer is matched to its normal tissue of origin.
SPECIFIC = {
    "Lung Neoplasms": ["lung"], "Breast Neoplasms": ["breast"], "Prostatic Neoplasms": ["prostate"],
    "Colorectal Neoplasms": ["colon"], "Colonic Neoplasms": ["colon"], "Rectal Neoplasms": ["colon"],
    "Stomach Neoplasms": ["stomach"], "Esophageal Neoplasms": ["esophagus"],
    "Liver Neoplasms": ["liver"], "Carcinoma, Hepatocellular": ["liver"],
    "Pancreatic Neoplasms": ["pancreas"], "Kidney Neoplasms": ["kidney"],
    "Urinary Bladder Neoplasms": ["bladder"], "Ovarian Neoplasms": ["ovary"],
    "Uterine Cervical Neoplasms": ["uterus"], "Endometrial Neoplasms": ["uterus"],
    "Skin Neoplasms": ["skin"], "Melanoma": ["skin"],
    "Thyroid Neoplasms": ["thyroid"], "Brain Neoplasms": ["brain"], "Glioma": ["brain"],
    "Glioblastoma": ["brain"], "Head and Neck Neoplasms": ["salivary_oral"],
    "Testicular Neoplasms": ["testis"], "Leukemia": ["immune_blood"], "Lymphoma": ["immune_blood"],
    "Multiple Myeloma": ["immune_blood"], "Myelodysplastic Syndromes": ["immune_blood"],
    "Hematologic Neoplasms": ["immune_blood"], "Intestinal Neoplasms": ["small_intestine", "colon"],
    "Lung Diseases": ["lung"], "Asthma": ["lung"], "Pulmonary Disease, Chronic Obstructive": ["lung"],
    "Cystic Fibrosis": ["lung"], "Hypertension, Pulmonary": ["lung", "vessel"],
    "Heart Diseases": ["heart"], "Heart Failure": ["heart"], "Myocardial Ischemia": ["heart"],
    "Atrial Fibrillation": ["heart"], "Arrhythmias, Cardiac": ["heart"],
    "Vascular Diseases": ["vessel"], "Hypertension": ["vessel", "heart", "kidney"],
    "Atherosclerosis": ["vessel"], "Cerebrovascular Disorders": ["brain", "vessel"],
    "Thrombosis": ["vessel"], "Venous Thrombosis": ["vessel"],
    "Liver Diseases": ["liver"], "Hepatitis": ["liver"], "Dyslipidemias": ["liver"],
    "Hypercholesterolemia": ["liver"], "Hyperlipidemias": ["liver"],
    "Kidney Diseases": ["kidney"], "Renal Insufficiency": ["kidney"], "Diabetic Nephropathies": ["kidney"],
    "Stomach Diseases": ["stomach"], "Gastroesophageal Reflux": ["esophagus", "stomach"],
    "Esophageal Diseases": ["esophagus"], "Peptic Ulcer": ["stomach", "small_intestine"],
    "Duodenal Diseases": ["small_intestine"],
    "Colonic Diseases": ["colon"], "Inflammatory Bowel Diseases": ["small_intestine", "colon"],
    "Pancreatic Diseases": ["pancreas"], "Biliary Tract Diseases": ["liver"],
    "Brain Diseases": ["brain"], "Epilepsy": ["brain"],
    "Neurodegenerative Diseases": ["brain"], "Migraine Disorders": ["brain"],
    "Headache Disorders": ["brain"], "Peripheral Nervous System Diseases": ["nerve"],
    "Neuralgia": ["nerve", "brain"], "Spinal Cord Diseases": ["nerve"],
    "Sleep Wake Disorders": ["brain"], "Mental Disorders": ["brain"],
    "Skin Diseases": ["skin"], "Psoriasis": ["skin"], "Dermatitis": ["skin"], "Acne Vulgaris": ["skin"],
    "Muscular Diseases": ["muscle"], "Muscular Dystrophies": ["muscle"],
    "Breast Diseases": ["breast"], "Prostatic Diseases": ["prostate"],
    "Urinary Bladder Diseases": ["bladder"], "Urinary Bladder, Overactive": ["bladder"],
    "Endometriosis": ["uterus", "ovary"], "Polycystic Ovary Syndrome": ["ovary"],
    "Thyroid Diseases": ["thyroid"], "Adrenal Gland Diseases": ["adrenal"],
    "Pituitary Diseases": ["pituitary"], "Eye Diseases": ["eye"], "Mouth Diseases": ["salivary_oral"],
    "Periodontal Diseases": ["salivary_oral"],
    "Diabetes Mellitus": METABOLIC, "Obesity": [*METABOLIC, "brain"], "Insulin Resistance": METABOLIC,
    "Arthritis, Rheumatoid": ["immune_blood"], "Lupus Erythematosus, Systemic": ["immune_blood"],
    "Immunologic Deficiency Syndromes": ["immune_blood"], "Hematologic Diseases": ["immune_blood"],
    "Anemia": ["immune_blood"], "Lymphatic Diseases": ["immune_blood"],
    "HIV Infections": ["immune_blood"], "Graft vs Host Disease": ["immune_blood"],
    "Sepsis": ["immune_blood"],
}  # fmt: skip
# Terms where we decided there is no sensible single tissue. Umbrella terms cover almost every
# disease below them, so they count only when they are one of the trial's own MeSH terms (their
# ancestors must not blank a trial that has a real organ match). Branch terms count for the
# trial's own term or any ancestor, so a child of a tissue-less branch ("Sarcoma" under
# "Neoplasms, Connective and Soft Tissue") is not handed to the broader rules below.
UMBRELLA_NO_TISSUE = {
    "Infections", "Neoplasms", "Metabolic Diseases", "Nutrition Disorders", "Signs and Symptoms",
    "Chemically-Induced Disorders", "Osteoarthritis", "Osteoporosis",
}  # fmt: skip
BRANCH_NO_TISSUE = {
    "Bone Diseases", "Joint Diseases", "Bacterial Infections", "Virus Diseases",
    "Parasitic Diseases", "Mycoses", "Neoplasms, Connective and Soft Tissue",
    "Neuroendocrine Tumors", "Wounds and Injuries", "Pregnancy Complications",
}  # fmt: skip
# Broader MeSH branches (and symptoms), used only when no specific or no-tissue term matched.
SYSTEM = {
    "Central Nervous System Diseases": ["brain"], "Pain": ["nerve", "brain"],
    "Intestinal Diseases": ["small_intestine", "colon"],
    "Genital Diseases, Female": ["uterus", "ovary"], "Genital Diseases, Male": ["prostate", "testis"],
    "Respiratory Tract Diseases": ["lung"], "Nervous System Diseases": ["brain"],
    "Cardiovascular Diseases": ["heart", "vessel"],
    "Digestive System Diseases": ["stomach", "small_intestine", "colon", "liver"],
    "Gastrointestinal Diseases": ["stomach", "small_intestine", "colon"],
    "Skin and Connective Tissue Diseases": ["skin"], "Urologic Diseases": ["kidney", "bladder", "prostate"],
    "Urogenital Diseases": ["kidney", "bladder", "uterus", "prostate"],
    "Immune System Diseases": ["immune_blood"],
    "Otorhinolaryngologic Diseases": ["salivary_oral"],
    "Endocrine System Diseases": ["pancreas", "thyroid", "adrenal", "pituitary"],
    "Nutritional and Metabolic Diseases": METABOLIC,
}  # fmt: skip
# Words in the registered condition text, for trials the MeSH lookup left unmatched.
TEXT_RULES = [
    (r"psoria|eczema|dermatitis|acne|hidradenitis|alopecia|vitiligo|urticaria|pruritus|rosacea", ["skin"]),
    (r"\bhiv\b|hiv-1|human immunodeficiency", ["immune_blood"]),
    (r"macular|retin|glaucoma|uveitis|dry eye|ocular|myopia|conjunctivitis", ["eye"]),
    (r"multiple sclerosis|alzheimer|parkinson|epilep|seizure|migraine|dementia|neuropath|narcolepsy|insomnia|huntington|\bals\b|amyotrophic", ["brain"]),
    (r"schizophren|depress|bipolar|anxiety|adhd|attention deficit|autism|ptsd|psychosis|smoking|opioid|alcohol", ["brain"]),
    (r"lung cancer|nsclc|non[- ]small|mesothelioma", ["lung"]), (r"breast cancer", ["breast"]),
    (r"prostate cancer", ["prostate"]),
    (r"gastric|stomach|gastroesophageal junction", ["stomach"]), (r"renal cell|kidney cancer", ["kidney"]),
    (r"hepatocellular|liver cancer|biliary", ["liver"]),
    (r"pancrea.*(cancer|carcinoma|adenocarcinoma)", ["pancreas"]), (r"melanoma", ["skin"]),
    (r"bladder cancer|urothelial", ["bladder"]),
    (r"leuk|lymphoma|myeloma|myelofibrosis|polycythemia|thrombocytop|neutropenia|sickle|hemophilia|anemia|anaemia", ["immune_blood"]),
    (r"lupus|rheumatoid|sjogren|vasculitis|graft", ["immune_blood"]),
    (r"cholesterol|lipid|triglycerid|hyperchylomicron", ["liver"]),
    (r"hypertension|heart failure|cardiac|cardio|myocardial|angina|atrial|thromboembol", ["heart", "vessel"]),
    (r"diabet|obes|insulin|nash|nafld|metabolic syndrome", METABOLIC),
    (r"hepatitis|cirrhosis|liver", ["liver"]), (r"kidney|renal|nephr|dialysis", ["kidney"]),
    (r"copd|asthma|pneumonia|covid|influenza|rsv|respiratory syncytial|pulmonary|bronch|cystic fibrosis|cough", ["lung"]),
    (r"crohn|colitis|constipation|gastroparesis|gerd|reflux|ibs\b|irritable bowel|dyspepsia|diarrh", ["stomach", "small_intestine", "colon"]),
    (r"overactive bladder|incontinence|prostatic hyperplasia", ["bladder", "prostate"]),
]  # fmt: skip
_TEXT = [(re.compile(pattern), tissues) for pattern, tissues in TEXT_RULES]
# Last resort: the project's therapeutic area, for areas that are one organ system.
AREA = {
    "neurology": ["brain"], "psychiatry": ["brain"], "respiratory": ["lung"],
    "cardiovascular": ["heart", "vessel"], "dermatology": ["skin"], "immunology": ["immune_blood"],
    "digestive": ["stomach", "small_intestine", "colon", "liver"],
}  # fmt: skip
# A broad MeSH term is dropped when its only reason for matching is one of these children
# (breast diseases sit under "Skin Diseases", pulmonary hypertension under "Hypertension").
GENUINE_SKIN = {
    "Psoriasis", "Dermatitis", "Acne Vulgaris", "Skin Neoplasms", "Melanoma",
    "Skin Diseases, Papulosquamous", "Skin Diseases, Eczematous", "Skin Diseases, Genetic",
    "Skin Diseases, Infectious", "Skin Diseases, Vascular", "Skin Diseases, Vesiculobullous",
    "Urticaria", "Alopecia",
}  # fmt: skip
GENUINE_VASCULAR = {
    "Hypertension", "Atherosclerosis", "Cerebrovascular Disorders", "Thrombosis", "Venous Thrombosis",
    "Aneurysm", "Peripheral Vascular Diseases", "Vasculitis",
}  # fmt: skip
SPURIOUS = {  # broad term -> (children that do not count, children that do)
    "Skin Diseases": ({"Breast Diseases", "Skin Diseases, Metabolic"}, GENUINE_SKIN),
    "Vascular Diseases": (
        {"Multiple Myeloma", "Paraproteinemias", "Blood Protein Disorders"},
        GENUINE_VASCULAR,
    ),
    "Hypertension": (
        {"Hypertension, Pulmonary"},
        {"Hypertension, Renal", "Hypertension, Malignant", "Hypertension, Pregnancy-Induced",
         "Hypertension, Portal"},
    ),
}  # fmt: skip


def _union(tables: list[list[str]]) -> list[str]:
    return sorted({tissue for tissues in tables for tissue in tissues})


def disease_tissues(
    mesh: list[str], ancestors: list[str], conditions: list[str], area: str | None
) -> list[str]:
    """Tissue groups for a trial. Empty when the table decides there is no sensible tissue.

    In order, the first step that matches decides, so a broad parent never widens a specific
    match: (1) the trial's own MeSH terms in `SPECIFIC`; (2) their ancestor terms in `SPECIFIC`;
    (3) an own umbrella term or any branch term in `UMBRELLA_NO_TISSUE` / `BRANCH_NO_TISSUE`,
    which leaves the trial unmatched; (4) own, then ancestor
    terms in `SYSTEM`; (5) the condition text; (6) the therapeutic area.
    """
    own, parents = set(mesh), set(ancestors)
    blocked: set[str] = set()  # tissues of a broad term dropped as spurious; later steps skip them
    for terms in (own, parents):
        specific = {t: v for t, v in SPECIFIC.items() if t in terms}
        for broad, (spurious, genuine) in SPURIOUS.items():
            seen = own | parents
            if broad in specific and seen & spurious and not seen & genuine:
                blocked.update(specific.pop(broad))
        if specific:
            return _union(list(specific.values()))
    if own & UMBRELLA_NO_TISSUE or (own | parents) & BRANCH_NO_TISSUE:
        return []

    def allowed(groups: list[list[str]]) -> list[str]:
        """The union of the tissue groups that share nothing with a dropped broad term."""
        return _union([g for g in groups if not blocked & set(g)])

    for terms in (own, parents):
        system = allowed([v for t, v in SYSTEM.items() if t in terms])
        if system:
            return system
    text = " | ".join(conditions).lower()
    by_text = allowed([tissues for pattern, tissues in _TEXT if pattern.search(text)])
    if by_text:
        return by_text
    return allowed([AREA.get(area or "", [])])


# ---------------------------------------------------------------- download and gene tables


def download(directory: Path = EXPRESSION_DIR) -> None:
    fetch_all(directory, RELEASE, LICENCE, FILES)


def read_gtex(path: Path) -> pl.DataFrame:
    """GTEx median TPM: Name (Ensembl, no version) and one column per tissue. Y-chromosome
    copies of pseudo-autosomal genes (PAR_Y) are dropped so each gene appears once."""
    with gzip.open(path) as handle:
        table = pl.read_csv(handle.read(), separator="\t", skip_rows=2)
    return (
        table.filter(~pl.col("Name").str.contains("PAR_Y"))
        .with_columns(pl.col("Name").str.split(".").list.first().alias("Name"))
        .drop("Description")
    )


def read_hpa(path: Path) -> pl.DataFrame:
    """HPA consensus RNA as a wide table: Gene (Ensembl) and one nTPM column per tissue."""
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(archive.namelist()[0])
    long = pl.read_csv(io.BytesIO(raw), separator="\t", infer_schema_length=0).with_columns(
        pl.col("nTPM").cast(pl.Float64)
    )
    return long.pivot(on="Tissue", index="Gene", values="nTPM").fill_null(0.0).sort("Gene")


def tau(log_expression: np.ndarray) -> np.ndarray:
    """Yanai tau per row of a genes-by-tissues matrix of log2(TPM + 1). NaN for all-zero rows."""
    peak = log_expression.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 - log_expression / peak[:, None]).sum(axis=1) / (log_expression.shape[1] - 1)


def gene_table(directory: Path = EXPRESSION_DIR) -> pl.DataFrame:
    """One row per Ensembl gene with gtex_tau, hpa_n_tissues and gtex_vital_max."""
    gtex = read_gtex(directory / GTEX.name)
    whole = [c for c in gtex.columns if c != "Name" and not c.startswith("Cells_")
             and c not in GTEX_NOT_WHOLE]  # fmt: skip
    values = gtex.select(whole).to_numpy().astype(float)
    expressed = values.max(axis=1) >= EXPRESSED
    gene_tau = np.where(expressed, tau(np.log2(values + 1)), np.nan)
    vital_columns = [c for group in VITAL for c in TISSUES[group][0]]
    vital = gtex.select(vital_columns).to_numpy().astype(float).max(axis=1)
    gtex_part = pl.DataFrame(
        {
            "ensembl_gene_id": gtex["Name"],
            "gtex_tau": gene_tau,
            "gtex_vital_max": np.log2(vital + 1),
        }
    ).fill_nan(None)
    hpa = read_hpa(directory / HPA.name)
    tissue_columns = [c for c in hpa.columns if c != "Gene"]
    hpa_part = pl.DataFrame(
        {
            "ensembl_gene_id": hpa["Gene"],
            "hpa_n_tissues": (hpa.select(tissue_columns).to_numpy() >= EXPRESSED).sum(axis=1),
        }
    )
    return gtex_part.join(hpa_part, on="ensembl_gene_id", how="full", coalesce=True).sort(
        "ensembl_gene_id"
    )


def matched_tissue_table(directory: Path = EXPRESSION_DIR) -> pl.DataFrame:
    """(ensembl_gene_id, tissue, log2_ntpm, best_rank) in HPA for every tissue group.

    The expression of a group is its highest member tissue; its rank is the best (smallest)
    rank any member has among the gene's 51 tissues, 1 being the highest, counting only tissues
    where the gene reaches `EXPRESSED` (a tissue below it ranks last).
    """
    hpa = read_hpa(directory / HPA.name)
    names = [c for c in hpa.columns if c != "Gene"]
    values = hpa.select(names).to_numpy()
    # Rank 1 is the highest. Ties share the best rank, and a tissue where the gene is below the
    # expression threshold is never placed among the top ones.
    ranks = (values[:, None, :] > values[:, :, None]).sum(axis=2) + 1
    ranks = np.where(values >= EXPRESSED, ranks, len(names) + 1)
    frames = []
    for group, (_, members) in TISSUES.items():
        index = [names.index(m) for m in members]
        frames.append(
            pl.DataFrame(
                {
                    "ensembl_gene_id": hpa["Gene"],
                    "tissue": group,
                    "log2_ntpm": np.log2(values[:, index].max(axis=1) + 1),
                    "best_rank": ranks[:, index].min(axis=1),
                }
            )
        )
    return pl.concat(frames)


# ---------------------------------------------------------------- per trial


def gene_features(targets: pl.DataFrame, genes: pl.DataFrame) -> pl.DataFrame:
    joined = (
        targets.select("nct_id", "targetId")
        .unique()
        .join(genes, left_on="targetId", right_on="ensembl_gene_id")
    )
    return joined.group_by("nct_id").agg(
        pl.col("gtex_tau").min().alias("expr_tau"),
        pl.col("hpa_n_tissues").max().alias("expr_n_tissues"),
        pl.col("gtex_vital_max").max().alias("expr_vital_max"),
    )


def matched_features(
    targets: pl.DataFrame, trial_tissues: pl.DataFrame, matched: pl.DataFrame
) -> pl.DataFrame:
    """`trial_tissues` has nct_id and a list column `tissues`. A target's value for a trial is
    its best over the trial's tissue groups; the trial takes the best over its targets."""
    pairs = (
        trial_tissues.explode("tissues")
        .drop_nulls("tissues")
        .rename({"tissues": "tissue"})
        .join(targets.select("nct_id", "targetId").unique(), on="nct_id")
        .join(
            matched,
            left_on=["targetId", "tissue"],
            right_on=["ensembl_gene_id", "tissue"],
        )
    )
    return pairs.group_by("nct_id").agg(
        pl.col("log2_ntpm").max().alias("expr_disease_tissue"),
        (pl.col("best_rank") <= TOP_RANKS).any().alias("expr_disease_tissue_top3"),
    )


def trial_features(
    targets: pl.DataFrame,
    trials: pl.DataFrame,
    conditions: pl.DataFrame,
    genes: pl.DataFrame,
    matched: pl.DataFrame,
) -> pl.DataFrame:
    """One row per trial with a target. `trials` has nct_id and therapeutic_area."""
    labelled = trials.select("nct_id", "therapeutic_area").join(conditions, on="nct_id")
    tissues = labelled.select(
        "nct_id",
        pl.struct("mesh", "ancestors", "conditions", "therapeutic_area")
        .map_elements(
            lambda row: disease_tissues(
                row["mesh"], row["ancestors"], row["conditions"], row["therapeutic_area"]
            ),
            return_dtype=pl.List(pl.String),
        )
        .alias("tissues"),
    )
    return (
        gene_features(targets, genes)
        .join(matched_features(targets, tissues, matched), on="nct_id", how="left")
        .sort("nct_id")
    )


def main() -> None:
    download()
    print(gene_table().describe())


if __name__ == "__main__":
    main()
