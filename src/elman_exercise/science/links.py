"""Trial -> investigational drug -> target genes, under today's links or under June 2020 links.

    uv run python -m elman_exercise.science.links    # download the 20.06 drug table, build the tables

    from elman_exercise.science.links import investigational_targets
    investigational_targets("today")   # Open Targets 26.09
    investigational_targets("2020")    # Open Targets 20.06 (2 July 2020)

Why two versions. A drug gets a target in Open Targets mostly because it got far enough to be
catalogued, so today's links favour drugs that later succeeded. The 2020 version keeps the
trial-to-drug step as it is today and replaces only the drug-to-target step with what the
20.06 release said. Release 20.06 is the oldest that still has a usable mechanism table.

Which drugs. The investigational ones only: drug-like interventions of an experimental arm,
matched by name to the drugs Open Targets lists for the trial (`targets.investigational_drugs`).
Comparators and background drugs Open Targets lists for the same trial are not used, and a
trial where nothing matches has no rows. There is no fallback to "every listed drug".

Which targets. `today`: every target Open Targets 26.09 lists for the drug, and a salt or brand
form also gets the targets of its whole molecule family (`MERGE_SALTS`; one rule for every use of
`drug_targets` in the benchmark, the same merging the 2020 match does through its drug ids; the
older `science.parquet` does not merge, so its target counts differ a little). `2020`: the targets of the drug's entry in the 20.06
drug table. A 26.09 drug is matched to a 20.06 entry through its own id, its parent's id and the
child ids either release lists (a salt or a brand form shares its parent's targets); a drug with
no 20.06 entry has no 2020 targets. Action type is Open Targets' word for what the drug does to
the target (inhibitor, antagonist, ...); 20.06 words come from the 20.06 table, today's from the
26.09 mechanism table (`chemistry.target_actions`), null when the table has no word. Several
different words for one (drug, target) are joined with "; ", sorted.

A third version, "2020_names", also dates the trial-to-drug step: registry intervention names are
matched only to the names in the 20.06 drug table (preferred name, synonyms, trade names; four or
more characters, not all digits, not in the generic-name stoplist `GENERIC_NAMES`; whole-word
phrase), so a drug catalogued after June 2020 cannot be found. Its names cover the table's parent
molecules only. `drug_id` there is a 20.06 id.

Still today's in the first two versions: which drug the trial tested (name match against the registry as
it is now, against Open Targets' drug names as they are now) and the trial's disease (precedent
only). The 2020 version fixes only the drug-to-target step. In all three versions the registry record
itself and Open Targets' disease mapping are today's, and so is `lead_drug` (the drug that
benchmark trials are clustered by and that the own-drug exclusion uses), even for "2020_names".

Only the columns below are read from the 20.06 drug table. Its approval year, phase and
indication columns describe how far a drug got and are not read. The full files are fetched,
checked against the recorded size and sha256 and slimmed, as `annotations` does; `download` then
deletes each full file (copies fetched earlier may still be on disk, and are not read).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import polars as pl

from elman_exercise import genetics
from elman_exercise.dating import AFTER_START
from elman_exercise.history import date_range
from elman_exercise.opentargets import INTERIM, OT_DIR, WANTED_COLUMNS
from elman_exercise.science import targets as tg
from elman_exercise.science.chemistry import target_actions
from elman_exercise.science.fetch import RemoteFile, fetch, write_manifest

Version = Literal["today", "2020", "2020_names"]
VERSIONS: tuple[str, ...] = ("today", "2020", "2020_names")
# The versions whose drug match is today's. "2020_names" matches registry names to the 20.06 drug
# table's own names, so its drug ids are 20.06 parent ids, not 26.09 ids.
TODAYS_DRUG_MATCH: tuple[str, ...] = ("today", "2020")
# One rule for salts in every use of `drug_targets`: a salt or brand form inherits its molecule
# family's targets. (The 2020 versions always do, through their drug match.)
MERGE_SALTS = True

OT2006_DIR = tg.RAW / "opentargets_2006"
OT2006_RELEASE = "Open Targets Platform 20.06 (output of 2 July 2020)"
LICENCE = (
    "Open Targets Platform data: CC0 1.0, and ChEMBL, a source of its drug content, CC BY-SA 3.0 "
    "(both as stated on https://platform-docs.opentargets.org/licence, read 12 Nov 2026; that is "
    "the current page, the 20.06 download folder carries no licence file)"
)
BASE_2006 = (
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/20.06/output/parquet/drug_parquet/"
)
_RUN_2006 = "cf32eb48-d514-4e5f-b74a-da02f4a9b3d1"
DRUG_2006_FILES = [
    RemoteFile(
        f"drug_2006_part_{i}.parquet",
        f"{BASE_2006}part-0000{i}-{_RUN_2006}-c000.snappy.parquet",
        size,
        digest,
    )
    for i, (size, digest) in enumerate(
        [
            (888323, "8f3e5858080c37c92708413c7d57fea6c646896b7643949e6ff3b3e74f443764"),
            (890350, "f9775539c4dac0b84c5e6d55932fc2c75520441b6329a63733049c3c40d818fa"),
            (892198, "eff8cfedfe42b969fc43a82b5be2e3d7d61fc2735866536a628fc47c3725924b"),
            (866912, "6ab8f2584b1c7e7546fe3b3eb0f3f611919ea476ab152ddbfa8464c8508b56a3"),
            (894716, "ab2d3bb507f8f6658c2826a56a547b63b09e354be66f5be935cdd8b72849a2b1"),
            (886030, "f6d70147be06d0792302d063bf3a81dc5a52370caee4cb38ea65f635480a4128"),
            (789045, "fc4bd91200b2a9083ef3f37a9825d432e2f4cc1f23612dbcf8559b4cc93ddbd7"),
        ]
    )
]
# The 26.09 drug table, for parent and child ids only. Its stage and approval columns are not read.
MOLECULE_2609 = RemoteFile(
    "drug_molecule_2609.parquet",
    "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.09/output/drug_molecule/"
    "00000000.parquet",
    6867467,
    "be65749aa1dbcc68d0930f8ac18126cec5c2bc2c0f232f11e8530dad4ba0585c",
)
DRUG_2006_COLUMNS = ("id", "child_chembl_ids", "mechanisms_of_action")
MOLECULE_2609_COLUMNS = ("id", "parentId", "childChemblIds")
SLIM_TARGETS_2006 = "targets_2006.parquet"  # drug_id, target_id, action_type
SLIM_DRUGS_2006 = "drugs_2006.parquet"  # drug_id, child_ids
SLIM_FAMILY_2609 = "family_2609.parquet"  # drug_id, parent_id, child_ids
SLIM_NAMES_2006 = "names_2006.parquet"  # drug_id, name (preferred name, synonyms, trade names)
DRUG_2006_NAME_COLUMNS = ("id", "pref_name", "synonyms", "trade_names")
MIN_NAME_LENGTH = 4  # a 20.06 name shorter than this, or all digits, is not used to match
# Generic substances and vehicles that a 20.06 entry lists as a name, but that a registry
# intervention names as a diluent, background or element, not as the drug under test. Never used
# to match. Chosen by reading the 20.06 names of up to eight characters that hit more than 50
# registry records; only plainly non-specific words are listed (no brand, no named drug).
GENERIC_NAMES: frozenset[str] = frozenset(
    {
        "insulin", "saline", "water", "oxygen", "glucose", "alcohol", "calcium", "potassium",
        "placebo", "vehicle", "salt", "same", "target", "depo", "diamine", "tartrate", "isotonic",
        "starch", "lactate", "lactose", "nitrogen", "hydrogen", "carbon", "silver", "syrup",
        "nitrate", "sucrose", "dextrose", "fluoride", "iodine", "ethanol", "gelatin", "platinum",
        "pd l1", "f 18", "ringer s",
    }
)  # fmt: skip

TRIAL_DRUGS = INTERIM / "investigational_drugs.parquet"
TRIAL_NAMES = INTERIM / "precedent_registry_names.parquet"  # nct_id, all_names
TRIAL_DRUGS_2006 = INTERIM / "trial_drugs_2020_names.parquet"  # nct_id, drug_id, is_exp
REGISTRY_SLIM = INTERIM / "precedent_registry.parquet"

# Which drug-to-target links a trial sees is decided by a release published after the trial
# started (and a catalogue that favours drugs that went on): hindsight. See `dating`.
FEATURES: dict[str, str] = {"target_links_today": AFTER_START, "target_links_2020": AFTER_START}

OUTPUT = {
    "nct_id": pl.String,
    "drug_id": pl.String,
    "target_id": pl.String,
    "action_type": pl.String,
}


# ---------------------------------------------------------------- download


def download(directory: Path = OT2006_DIR) -> None:
    """Fetch the 20.06 drug table and the 26.09 parent table, keep named columns, delete the rest."""
    directory.mkdir(parents=True, exist_ok=True)
    wanted = (SLIM_TARGETS_2006, SLIM_DRUGS_2006, SLIM_NAMES_2006)
    if not all((directory / n).exists() for n in wanted):
        parts = []
        for file in DRUG_2006_FILES:
            path = fetch(file, directory)
            columns = [*DRUG_2006_COLUMNS, *DRUG_2006_NAME_COLUMNS[1:]]
            parts.append(pl.read_parquet(path, columns=columns))
            path.unlink()
        drugs = pl.concat(parts)
        for name, build in (
            (SLIM_DRUGS_2006, slim_drugs_2006),
            (SLIM_TARGETS_2006, slim_targets_2006),
            (SLIM_NAMES_2006, slim_names_2006),
        ):
            if not (directory / name).exists():  # an existing slim file is never rewritten
                build(drugs).write_parquet(directory / name)
    if not (directory / SLIM_FAMILY_2609).exists():
        path = fetch(MOLECULE_2609, directory)
        (
            pl.read_parquet(path, columns=list(MOLECULE_2609_COLUMNS))
            .rename({"id": "drug_id", "parentId": "parent_id", "childChemblIds": "child_ids"})
            .write_parquet(directory / SLIM_FAMILY_2609)
        )
        path.unlink()
    write_manifest(directory, OT2006_RELEASE, LICENCE, [*DRUG_2006_FILES, MOLECULE_2609])


def slim_drugs_2006(drugs: pl.DataFrame) -> pl.DataFrame:
    """(drug_id, child_ids): every drug in the 20.06 table and the ids it lists as children."""
    return drugs.select(
        pl.col("id").alias("drug_id"), pl.col("child_chembl_ids").alias("child_ids")
    )


def slim_names_2006(drugs: pl.DataFrame) -> pl.DataFrame:
    """(drug_id, name): the preferred name, synonyms and trade names of each 20.06 drug, normalised.

    Names shorter than `MIN_NAME_LENGTH` characters or made only of digits are left out. The
    generic-name stoplist is applied when matching (`name_lookup`), not here, so a slimmed file on
    disk never has to be rebuilt.
    """
    rows = []
    for drug, preferred, synonyms, trade in drugs.select(
        "id", "pref_name", "synonyms", "trade_names"
    ).rows():
        # a null preferred name or a null entry in a list gives "", which the length rule drops
        raw = [preferred, *(synonyms or []), *(trade or [])]
        names = {genetics.normalise_name(n) for n in raw if n is not None}
        rows += [(drug, n) for n in sorted(names) if len(n) >= MIN_NAME_LENGTH and not n.isdigit()]
    return pl.DataFrame(rows, schema={"drug_id": pl.String, "name": pl.String}, orient="row")


def slim_targets_2006(drugs: pl.DataFrame) -> pl.DataFrame:
    """(drug_id, target_id, action_type): one row per mechanism and target gene."""
    return (
        drugs.select(pl.col("id").alias("drug_id"), "mechanisms_of_action")
        .explode("mechanisms_of_action")
        .drop_nulls("mechanisms_of_action")
        .select(
            "drug_id",
            pl.col("mechanisms_of_action").struct.field("action_type").alias("action_type"),
            pl.col("mechanisms_of_action").struct.field("target_components").alias("components"),
        )
        .explode("components")
        .drop_nulls("components")
        .select(
            "drug_id",
            pl.col("components").struct.field("ensembl").alias("target_id"),
            "action_type",
        )
        .drop_nulls("target_id")
        .unique()
        .sort("drug_id", "target_id", "action_type", nulls_last=True)
    )


# ---------------------------------------------------------------- matching drugs across releases


def entries_2006(drugs_2006: pl.DataFrame) -> dict[str, set[str]]:
    """Any id the 20.06 table knows -> every entry it stands for.

    An id that has its own row stands for that entry. An id a row lists as a child also stands for
    the row that lists it, even when it has a row of its own (a salt with its own catalogue row
    still shares its parent's targets).
    """
    known: dict[str, set[str]] = {i: {i} for i in drugs_2006["drug_id"]}
    for entry, children in drugs_2006.select("drug_id", "child_ids").rows():
        for child in children or []:
            known.setdefault(child, set()).add(entry)
    return known


def map_to_2006(family_2609: pl.DataFrame, drugs_2006: pl.DataFrame) -> pl.DataFrame:
    """(drug_id, drug_2020): each 26.09 drug and the 20.06 entries it stands for.

    The candidates are the drug, its parent, and the children either release lists for the drug
    or its parent. A candidate that has its own 20.06 row and is also listed as a child of another
    row stands for both. A drug with no candidate in the 20.06 table is absent from the result.
    """
    known = entries_2006(drugs_2006)
    parent_of = {d: p for d, p in family_2609.select("drug_id", "parent_id").rows() if p}
    kids = {d: c or [] for d, c in family_2609.select("drug_id", "child_ids").rows()}
    rows = []
    for drug in family_2609["drug_id"]:
        parent = parent_of.get(drug)
        candidates = {drug, parent, *kids.get(drug, []), *(kids.get(parent, []) if parent else [])}
        found = {e for c in candidates if c in known for e in known[c]}
        rows += [(drug, e) for e in sorted(found)]
    return pl.DataFrame(rows, schema={"drug_id": pl.String, "drug_2020": pl.String}, orient="row")


def join_action_words(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per (drug_id, target_id): action types, if several, sorted and joined by "; "."""
    return (
        frame.group_by("drug_id", "target_id")
        .agg(pl.col("action_type").drop_nulls().unique().sort().str.join("; ").alias("action_type"))
        .with_columns(pl.col("action_type").replace("", None))
        .sort("drug_id", "target_id")
    )


def inherit_from_parent(
    table: pl.DataFrame, parent_of: dict[str, str], drug_ids: Iterable[str] = ()
) -> pl.DataFrame:
    """Give every drug the targets of its whole molecule family (the parent and all its salts).

    A drug's family is its parent molecule if it has one, else itself. Every drug in `drug_ids`
    (and every drug already in `table`) gets every target listed for any member of its family.
    """
    canon = pl.DataFrame(
        [(d, parent_of.get(d, d)) for d in {*table["drug_id"], *drug_ids, *parent_of}],
        schema={"drug_id": pl.String, "canon": pl.String},
        orient="row",
    )
    by_family = table.join(canon, on="drug_id").select("canon", "target_id", "action_type").unique()
    spread = canon.join(by_family, on="canon").select("drug_id", "target_id", "action_type")
    return join_action_words(pl.concat([table, spread]))


# ---------------------------------------------------------------- drug -> targets


def drug_targets(version: Version, merge_salts: bool = MERGE_SALTS) -> pl.DataFrame:
    """(drug_id, target_id, action_type) for every drug, in today's or the 2020 link version.

    `drug_id` is a 26.09 drug id in both versions, so either table joins to Open Targets' trial
    reports; for "2020_names" it is a 20.06 drug id instead (its drugs are matched by 20.06 name).
    `merge_salts` (today only; the 2020 version always merges through its drug match)
    lets a salt or brand form inherit its parent molecule's targets.
    """
    if version == "today":
        links = pl.read_parquet(OT_DIR / "clinical_target.parquet").rename(
            {"drugId": "drug_id", "targetId": "target_id"}
        )
        links = links.select("drug_id", "target_id").drop_nulls().unique()
        actions = target_actions().rename(
            {"drugId": "drug_id", "targetId": "target_id", "actionType": "action_type"}
        )
        table = links.join(actions, on=["drug_id", "target_id"], how="left")
        table = join_action_words(table)
        if merge_salts:
            family = pl.read_parquet(OT2006_DIR / SLIM_FAMILY_2609)
            parent_of = {d: p for d, p in family.select("drug_id", "parent_id").rows() if p}
            table = inherit_from_parent(table, parent_of, family["drug_id"])
        return table
    if version == "2020_names":
        moa = pl.read_parquet(OT2006_DIR / SLIM_TARGETS_2006)
        return join_action_words(moa.select("drug_id", "target_id", "action_type"))
    if version == "2020":
        family = pl.read_parquet(OT2006_DIR / SLIM_FAMILY_2609)
        drugs = pl.read_parquet(OT2006_DIR / SLIM_DRUGS_2006)
        moa = pl.read_parquet(OT2006_DIR / SLIM_TARGETS_2006).rename({"drug_id": "drug_2020"})
        table = map_to_2006(family, drugs).join(moa, on="drug_2020")
        return join_action_words(table.select("drug_id", "target_id", "action_type"))
    raise ValueError(f"version must be one of {VERSIONS}, not {version!r}")


# ---------------------------------------------------------------- the registry, read once


def date_columns(registry: pl.DataFrame) -> pl.DataFrame:
    """Add `start_first` and `start_last` (earliest and latest day the start date could mean) and
    `submit_first` and `submit_last` (the same for the first-submitted date). A month alone stands
    for the whole month; a full date is a single day."""
    rows = []
    for nct, start, submitted in registry.select("nct_id", "start", "first_submit").rows():
        first, last = date_range(start)
        sub_first, sub_last = date_range(submitted)
        rows.append((nct, first, last, sub_first, sub_last))
    dates = pl.DataFrame(
        rows,
        schema={
            "nct_id": pl.String,
            "start_first": pl.Date,
            "start_last": pl.Date,
            "submit_first": pl.Date,
            "submit_last": pl.Date,
        },
        orient="row",
    )
    return registry.join(dates, on="nct_id", how="left")


def build_registry_table(path: Path = REGISTRY_SLIM) -> pl.DataFrame:
    """One row per downloaded Phase 2 or 3 record: phases, start and first-submitted dates, whether
    the record is WITHDRAWN (it never started, so it is no precedent), and the experimental-arm
    intervention names. Arms, interventions, dates and that one status word; no stop reason or
    result field is read. Written to `data/interim/precedent_registry.parquet`."""
    from elman_exercise.download import iter_studies, nct_id

    rows = []
    for study in iter_studies():
        key = nct_id(study).upper()
        protocol = study.get("protocolSection", {})
        status = protocol.get("statusModule", {})
        rows.append(
            {
                "nct_id": key,
                "phases": protocol.get("designModule", {}).get("phases") or [],
                "start": (status.get("startDateStruct") or {}).get("date"),
                "first_submit": status.get("studyFirstSubmitDate"),
                "withdrawn": status.get("overallStatus") == "WITHDRAWN",
                "exp_names": genetics.experimental_names([study], {key}).get(key, []),
            }
        )
    schema = {
        "nct_id": pl.String,
        "phases": pl.List(pl.String),
        "start": pl.String,
        "first_submit": pl.String,
        "withdrawn": pl.Boolean,
        "exp_names": pl.List(pl.String),
    }
    table = date_columns(pl.DataFrame(rows, schema=schema)).sort("nct_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)
    return table


REGISTRY_COLUMNS = ("withdrawn", "submit_last")


def registry_table(path: Path = REGISTRY_SLIM) -> pl.DataFrame:
    """The slim registry table; rebuilt if the file on disk predates a column the code needs."""
    if path.exists():
        table = pl.read_parquet(path)
        if all(c in table.columns for c in REGISTRY_COLUMNS):
            return table
    return build_registry_table(path)


# ---------------------------------------------------------------- trial -> drug by 20.06 names


def build_registry_names(path: Path = TRIAL_NAMES) -> pl.DataFrame:
    """(nct_id, all_names): the normalised names of every intervention of every Phase 2 or 3 record.

    Names and other names only; read for the 20.06 name match, which also looks at comparators
    so that a trial with no experimental-arm match can fall back to its listed drugs.
    """
    from elman_exercise.download import iter_studies, nct_id

    rows = []
    for study in iter_studies():
        module = study.get("protocolSection", {}).get("armsInterventionsModule") or {}
        names = {
            genetics.normalise_name(n)
            for i in module.get("interventions") or []
            for n in [i.get("name"), *(i.get("otherNames") or [])]
        } - {""}
        rows.append({"nct_id": nct_id(study).upper(), "all_names": sorted(names)})
    table = pl.DataFrame(rows, schema={"nct_id": pl.String, "all_names": pl.List(pl.String)})
    table = table.sort("nct_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)
    return table


def registry_names(path: Path = TRIAL_NAMES) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else build_registry_names(path)


def name_lookup(names_2006: pl.DataFrame) -> dict[str, set[str]]:
    """Normalised 20.06 name -> the 20.06 drug ids that carry it (generic names left out)."""
    lookup: dict[str, set[str]] = {}
    for drug, name in names_2006.select("drug_id", "name").rows():
        if name not in GENERIC_NAMES:
            lookup.setdefault(name, set()).add(drug)
    return lookup


def drugs_named_in(names: Iterable[str], lookup: dict[str, set[str]], longest: int) -> set[str]:
    """20.06 drugs one of whose names is a whole-word phrase inside one of `names`."""
    found: set[str] = set()
    for name in names:
        words = name.split()
        for i in range(len(words)):
            for j in range(i + 1, min(len(words), i + longest) + 1):
                found |= lookup.get(" ".join(words[i:j]), set())
    return found


def match_names_2006(registry: pl.DataFrame, names_2006: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, drug_id, is_exp): which 20.06 drugs a trial's registry names point to.

    `registry` has `nct_id`, `exp_names` (experimental-arm names) and `all_names` (every
    intervention). A drug found in an experimental-arm name has `is_exp` True; a drug found only
    elsewhere in the record (a comparator, a background drug) has it False. `drug_id` is the 20.06
    entry's own id (a parent molecule: a salt or brand listed as a child there is not matched on
    its own). Nothing from today's Open Targets is read, so a drug added to the catalogue after
    June 2020 cannot be found.
    """
    lookup = name_lookup(names_2006)
    longest = max((len(n.split()) for n in lookup), default=1)
    rows = []
    for nct, experimental, everything in registry.select("nct_id", "exp_names", "all_names").rows():
        in_exp = drugs_named_in(experimental or [], lookup, longest)
        in_all = drugs_named_in(everything or [], lookup, longest)
        rows += [(nct, d, True) for d in sorted(in_exp)]
        rows += [(nct, d, False) for d in sorted(in_all - in_exp)]
    schema = {"nct_id": pl.String, "drug_id": pl.String, "is_exp": pl.Boolean}
    return pl.DataFrame(rows, schema=schema, orient="row")


def trial_drugs_2020_names() -> pl.DataFrame:
    """(nct_id, drug_id, is_exp) for every Phase 2 or 3 record, matched by 20.06 names; cached."""
    if not TRIAL_DRUGS_2006.exists():
        download()
        registry = (
            registry_table()
            .select("nct_id", "exp_names")
            .join(registry_names(), on="nct_id", how="left")
        )
        names = pl.read_parquet(OT2006_DIR / SLIM_NAMES_2006)
        match_names_2006(registry, names).write_parquet(TRIAL_DRUGS_2006)
    return pl.read_parquet(TRIAL_DRUGS_2006)


# ---------------------------------------------------------------- trial -> investigational drugs


def trial_ids() -> list[str]:
    """NCT ids of every trial in the benchmark, all splits (ids only, no labels)."""
    return tg.load_trials()["nct_id"].to_list()


def build_trial_drugs(
    ids: Iterable[str], registry: pl.DataFrame, ot_dir: Path = OT_DIR
) -> pl.DataFrame:
    """(nct_id, drug_id): the investigational drugs of these trials (`targets.investigational_drugs`)."""
    ids = list(ids)
    reports = pl.read_parquet(
        ot_dir / "clinical_report.parquet", columns=list(WANTED_COLUMNS["clinical_report"])
    )
    molecules = pl.read_parquet(ot_dir / "drug_molecule.parquet")
    wanted = registry.filter(pl.col("nct_id").is_in(ids))
    experimental = dict(wanted.select("nct_id", "exp_names").rows())
    drugs = tg.investigational_drugs(ids, reports, molecules, experimental)
    return (
        drugs.select("nct_id", pl.col("drugId").alias("drug_id")).unique().sort("nct_id", "drug_id")
    )


def trial_drugs() -> pl.DataFrame:
    """The benchmark's investigational drugs, built once and kept in `data/interim/`."""
    if not TRIAL_DRUGS.exists():
        build_trial_drugs(trial_ids(), registry_table()).write_parquet(TRIAL_DRUGS)
    return pl.read_parquet(TRIAL_DRUGS)


# ---------------------------------------------------------------- the public function


def join_trial_targets(trial_drug: pl.DataFrame, links: pl.DataFrame) -> pl.DataFrame:
    """(nct_id, drug_id, target_id, action_type): each trial's drugs joined to their targets."""
    return (
        trial_drug.select("nct_id", "drug_id")
        .join(links.select("drug_id", "target_id", "action_type"), on="drug_id")
        .select(*OUTPUT)
        .unique()
        .sort("nct_id", "drug_id", "target_id")
    )


def investigational_targets(version: Version) -> pl.DataFrame:
    """One row per (nct_id, drug_id, target_id) for the investigational drugs of the benchmark.

    Columns: `nct_id`, `drug_id` (an Open Targets 26.09 drug id), `target_id` (Ensembl gene id),
    `action_type` (null if the release gives none). Every split; no label is read.
    `version` is "today" (Open Targets 26.09), "2020" (20.06 targets, today's drug match) or
    "2020_names" (20.06 targets and the drug matched by 20.06 names only; `drug_id` is then a 20.06
    drug id). A trial with no
    investigational drug, or none with a target in that release, has no rows.
    """
    if version not in VERSIONS:
        raise ValueError(f"version must be one of {VERSIONS}, not {version!r}")
    if version != "today":
        download()
    if version == "2020_names":
        ids = set(trial_ids())
        matched = trial_drugs_2020_names().filter(pl.col("is_exp") & pl.col("nct_id").is_in(ids))
        return join_trial_targets(matched, drug_targets(version, merge_salts=MERGE_SALTS))
    return join_trial_targets(trial_drugs(), drug_targets(version, merge_salts=MERGE_SALTS))


def main() -> None:
    download()
    for version in VERSIONS:
        table = investigational_targets(version)
        print(
            f"{version}: {table.height:,} rows, {table['nct_id'].n_unique():,} trials, "
            f"{table['target_id'].n_unique():,} targets"
        )


if __name__ == "__main__":
    main()
