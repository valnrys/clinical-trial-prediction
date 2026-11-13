"""What kind of molecule is the drug, and what does it do to its target? ChEMBL and Open Targets.

    uv run python -m trial_benchmark.science.chemistry    # needs data/raw/opentargets and the registry

Small-molecule properties are ChEMBL release 37's calculated values (https://www.ebi.ac.uk/chembl,
CC BY-SA 3.0), fetched from its public REST API for the trials' lead drugs only and saved,
with a checksum, in data/raw/chemistry/. The action type is Open Targets 26.09's
`drug_mechanism_of_action` table (CC0). Only the columns below are read; ChEMBL's maximum
phase, approval and marketed-route fields are never requested, because they describe how far
a drug got.

The lead drug is the trial's first investigational drug (`targets.investigational_drugs`).
A combination's other drugs are not looked at.

Which trials get a value. Chemistry describes the molecule, so it could exist for a drug with
no known target. It is given only to trials that have a target, the rule every science column
follows (see `science`): a column that was present for some target-less trials would show a
model which trials are target-less. A drug with no ChEMBL properties and no mechanism (most
biologics) gives the trial no chemistry row.

- `chem_mol_weight`, `chem_logp`, `chem_polar_surface_area`, `chem_qed`
  (drug-likeness, 0 to 1), `chem_rule_of_five_violations`: ChEMBL's values for the lead drug.
  Null for biologics and anything ChEMBL gives no properties for.
- `chem_action_type`: what the drug does to its target: inhibitor, antagonist, agonist,
  activator or other (`ACTION_GROUPS`). A drug whose mechanism rows fall in different groups
  gives "multiple". Null if Open Targets lists no mechanism.

All describe the molecule, so they are TIMELESS in the sense of `dating`.
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

from trial_benchmark.dating import TIMELESS
from trial_benchmark.opentargets import BASE_URL
from trial_benchmark.opentargets import RELEASE as OT_RELEASE
from trial_benchmark.science import targets as tg
from trial_benchmark.science.fetch import USER_AGENT, RemoteFile, fetch, sha256_of, write_manifest

CHEMISTRY_DIR = tg.RAW / "chemistry"
CHEMBL_RELEASE = "ChEMBL_37"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/"
LICENCE = "ChEMBL: CC BY-SA 3.0; Open Targets: CC0"
MECHANISM = RemoteFile(
    "drug_mechanism_of_action.parquet",
    f"{BASE_URL}drug_mechanism_of_action/00000000.parquet",
    322830,
    "8484accd3383837e0eaac2164eed89991daf58dc93f0a599072fb7e5ef4f6c28",
)
MECHANISM_COLUMNS = ("actionType", "chemblIds", "targets")
MECHANISM_SLIM = "mechanism_of_action.parquet"
PROPERTIES = "chembl_properties.csv"
# The only ChEMBL fields requested. No phase, approval or route field is on this list.
CHEMBL_FIELDS = {
    "mw_freebase": "mol_weight",
    "alogp": "logp",
    "psa": "polar_surface_area",
    "qed_weighted": "qed",
    "num_ro5_violations": "rule_of_five_violations",
}
# The fetched file depends only on the pinned releases and on the lead drugs, which are fixed
# by the trials. Rows and sha256 of the saved CSV, so a changed upstream value is noticed.
EXPECTED_PROPERTIES = (1965, "f77c3583d30acdde6b50258ecdd84b143979920e76521b76790e45db0b0fb8a4")
BATCH = 50
RETRIES = 15

FEATURES: dict[str, str] = dict.fromkeys(
    (
        "chem_mol_weight",
        "chem_logp",
        "chem_polar_surface_area",
        "chem_qed",
        "chem_rule_of_five_violations",
        "chem_action_type",
    ),
    TIMELESS,
)
ACTION_GROUPS = {
    "INHIBITOR": "inhibitor", "ANTISENSE INHIBITOR": "inhibitor", "RNAI INHIBITOR": "inhibitor",
    "ANTAGONIST": "antagonist", "BLOCKER": "antagonist", "INVERSE AGONIST": "antagonist",
    "NEGATIVE ALLOSTERIC MODULATOR": "antagonist", "NEGATIVE MODULATOR": "antagonist",
    "ALLOSTERIC ANTAGONIST": "antagonist",
    "AGONIST": "agonist", "PARTIAL AGONIST": "agonist",
    "ACTIVATOR": "activator", "OPENER": "activator", "POSITIVE ALLOSTERIC MODULATOR": "activator",
    "POSITIVE MODULATOR": "activator",
}  # fmt: skip
OTHER = "other"


# ---------------------------------------------------------------- download


def _get_json(url: str) -> dict:
    """GET and parse JSON, retrying: the ChEMBL service answers 500 now and then."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except (OSError, json.JSONDecodeError):
            time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"{url}: no answer after {RETRIES} attempts.")


def _number(value) -> float | None:
    """ChEMBL sends numbers as text, and null for a property it has not calculated."""
    return None if value in (None, "") else float(value)


def fetch_properties(drug_ids: list[str], path: Path) -> None:
    """Save ChEMBL's calculated properties for these molecules as a CSV (drug_id + fields)."""
    release = _get_json(CHEMBL_API + "status.json").get("chembl_db_version")
    if release is not None and release != CHEMBL_RELEASE:
        raise RuntimeError(f"The ChEMBL service is now {release}; this code pins {CHEMBL_RELEASE}.")
    if release is None:  # the status page is sometimes served without a version
        print(
            f"  ChEMBL did not name its release; the saved file's checksum decides ({CHEMBL_RELEASE})."
        )
    rows = {}
    for start in range(0, len(drug_ids), BATCH):
        query = urllib.parse.urlencode(
            {
                "molecule_chembl_id__in": ",".join(drug_ids[start : start + BATCH]),
                "only": "molecule_chembl_id,molecule_properties",
                "limit": 100,
            }
        )
        for molecule in _get_json(f"{CHEMBL_API}molecule.json?{query}")["molecules"]:
            found = molecule.get("molecule_properties") or {}
            rows[molecule["molecule_chembl_id"]] = [_number(found.get(f)) for f in CHEMBL_FIELDS]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["drug_id", *CHEMBL_FIELDS.values()])
    writer.writerows([drug, *rows[drug]] for drug in sorted(rows))
    path.write_text(buffer.getvalue())


def download(lead_drugs: list[str], directory: Path = CHEMISTRY_DIR) -> None:
    """Fetch the mechanism table (slimmed) and the ChEMBL properties of the lead drugs."""
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / MECHANISM_SLIM).exists():
        full = fetch(MECHANISM, directory)
        pl.read_parquet(full, columns=list(MECHANISM_COLUMNS)).write_parquet(
            directory / MECHANISM_SLIM
        )
        full.unlink()
    path = directory / PROPERTIES
    if not path.exists():
        fetch_properties(sorted(set(lead_drugs)), path)
    expected_rows, expected_sha = EXPECTED_PROPERTIES
    rows = len(path.read_text().splitlines()) - 1
    if (rows, sha256_of(path)) != (expected_rows, expected_sha):
        raise RuntimeError(
            f"{path.name}: {rows} rows with sha256 {sha256_of(path)}, expected "
            f"{expected_rows} rows with {expected_sha}. Delete it to fetch again."
        )
    write_manifest(
        directory,
        f"{CHEMBL_RELEASE}; Open Targets {OT_RELEASE}",
        LICENCE,
        [
            MECHANISM,
            RemoteFile(PROPERTIES, CHEMBL_API + "molecule.json", path.stat().st_size, expected_sha),
        ],
    )


# ---------------------------------------------------------------- per drug


def action_group(action_type: str | None) -> str | None:
    return None if action_type is None else ACTION_GROUPS.get(action_type, OTHER)


def drug_actions(mechanism: pl.DataFrame) -> pl.DataFrame:
    """(drugId, chem_action_type): one group per drug, or "multiple" if its mechanisms differ."""
    rows = (
        mechanism.select("actionType", pl.col("chemblIds").alias("drugId"))
        .explode("drugId")
        .drop_nulls(["actionType", "drugId"])
        .with_columns(
            pl.col("actionType").map_elements(action_group, return_dtype=pl.String).alias("group")
        )
    )
    return (
        rows.group_by("drugId")
        .agg(pl.col("group").alias("chem_action_type"))
        .with_columns(pl.col("chem_action_type").list.unique().alias("chem_action_type"))
        .with_columns(
            pl.when(pl.col("chem_action_type").list.len() == 1)
            .then(pl.col("chem_action_type").list.first())
            .otherwise(pl.lit(tg.MULTIPLE))
            .alias("chem_action_type")
        )
        .sort("drugId")
    )


def target_actions(directory: Path = CHEMISTRY_DIR) -> pl.DataFrame:
    """(drugId, targetId, actionType): what Open Targets says each drug does to each target."""
    return (
        pl.read_parquet(directory / MECHANISM_SLIM)
        .select(
            pl.col("chemblIds").alias("drugId"), pl.col("targets").alias("targetId"), "actionType"
        )
        .explode("drugId")
        .explode("targetId")
        .drop_nulls()
        .unique()
        .sort("drugId", "targetId", "actionType")
    )


def drug_table(directory: Path = CHEMISTRY_DIR) -> pl.DataFrame:
    """One row per drug with any value: the five ChEMBL properties and the action group."""
    properties = pl.read_csv(directory / PROPERTIES).rename(
        {"drug_id": "drugId", **{v: f"chem_{v}" for v in CHEMBL_FIELDS.values()}}
    )
    actions = drug_actions(pl.read_parquet(directory / MECHANISM_SLIM))
    table = properties.join(actions, on="drugId", how="full", coalesce=True)
    return table.filter(pl.any_horizontal(pl.col(c).is_not_null() for c in FEATURES)).sort("drugId")


def trial_features(drugs: pl.DataFrame, targets: pl.DataFrame, table: pl.DataFrame) -> pl.DataFrame:
    """One row per trial that has a target and a lead drug with any chemistry value."""
    with_target = targets.select("nct_id").unique()
    leads = drugs.filter(pl.col("is_lead")).select("nct_id", "drugId")
    return (
        leads.join(with_target, on="nct_id").join(table, on="drugId").drop("drugId").sort("nct_id")
    )


def main() -> None:
    drugs, _, _ = tg.build_inputs()
    download(drugs.filter(pl.col("is_lead"))["drugId"].to_list())
    print(drug_table().describe())


if __name__ == "__main__":
    main()
