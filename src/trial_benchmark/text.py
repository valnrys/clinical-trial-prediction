"""The words of each trial's registration, one column per source field.

    uv run python -m trial_benchmark.text      # -> data/interim/text.parquet

One row per trial in trials.csv (all splits), with `nct_id`, `split` and six text
columns. Each is read from the PROTOCOL section of the raw record only. The results
section is never opened. Missing fields are null, not empty text.

The columns are kept apart so a model can choose which to read:

- title: the official title, or the brief title if there is no official one.
- conditions: the registered conditions, joined with "; ".
- interventions: the names of the registered interventions, joined with "; ".
- primary_outcomes: the registered primary outcome measures (names only), joined with "; ".
- eligibility: the inclusion and exclusion criteria text.
- summary: the brief summary.

How each column is dated
------------------------
Every column except the summary is **protocol**: read from the current version of
the record (the API serves nothing older), which sponsors can edit after the trial
starts. The pre-registered baseline reads the same record, so honest models may use
these columns (see `dating`). The summary is **after the start**.

Most exposed first:

1. summary. Sponsors sometimes rewrite it once results are in ("the study met its
   primary endpoint"). That would put the answer into the feature. The share of dev
   summaries with result-like phrases is printed by `main()`. Honest models must
   not read it.
2. title. The official title is occasionally reworded late, but rarely with results.
3. primary_outcomes. Measures are sometimes tidied after the fact (units, time
   frames). The measure itself seldom changes, as the registry logs outcome changes.
4. interventions, conditions, eligibility. Edited now and then (a drug renamed,
   criteria amended), but not usually with knowledge of the result.

The claim for the protocol columns is "not obviously contaminated", not "known
on the start date". `FEATURES` records the dating of each column for the models.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import polars as pl

from trial_benchmark.dataset import ROOT
from trial_benchmark.dating import AFTER_START, HONEST, PROTOCOL
from trial_benchmark.opentargets import INTERIM

TABLE = "text"
OUT_PATH = INTERIM / "text.parquet"
TRIALS_PATH = ROOT / "trials.csv"
SEPARATOR = "; "

FEATURES: dict[str, str] = {
    "title": PROTOCOL,
    "conditions": PROTOCOL,
    "interventions": PROTOCOL,
    "primary_outcomes": PROTOCOL,
    "eligibility": PROTOCOL,
    "summary": AFTER_START,  # the riskiest: may have been rewritten with results
}
ALL_COLUMNS = tuple(FEATURES)  # everything the table holds, summary included
WORD_FIELDS = tuple(c for c, d in FEATURES.items() if d in HONEST)  # the default for models

# Phrases that sound like a result rather than a plan. Checked on dev by `main()`.
RESULT_PHRASES: dict[str, str] = {
    "met its primary endpoint": r"\bmet (its|the) primary",
    "primary endpoint met / not met": r"\bprimary (end ?point|objective|outcome) (was|were) (not )?met",
    "was terminated": r"\b(was|were|been) (terminated|stopped early|halted|discontinued)",
    "did not show": r"\bdid not (show|meet|demonstrate|achieve)",
    "results showed": r"\bresults? (showed|show|demonstrated|indicated)",
    "failed to": r"\bfailed to\b",
}


def _clean(text: str | None) -> str | None:
    """The text with surrounding space removed, or None if nothing is left."""
    if text is None:
        return None
    text = text.strip()
    return text or None


def _joined(items: Iterable[str | None]) -> str | None:
    """The non-empty items joined with "; ", or None if there are none."""
    kept = [c for c in map(_clean, items) if c]
    return SEPARATOR.join(kept) if kept else None


def extract(study: dict) -> dict[str, str | None]:
    """The six text columns of one raw record, from its protocol section only."""
    protocol = study["protocolSection"]
    identification = protocol.get("identificationModule") or {}
    interventions = (protocol.get("armsInterventionsModule") or {}).get("interventions") or []
    outcomes = (protocol.get("outcomesModule") or {}).get("primaryOutcomes") or []
    return {
        "nct_id": identification["nctId"],
        "title": _clean(identification.get("officialTitle"))
        or _clean(identification.get("briefTitle")),
        "conditions": _joined((protocol.get("conditionsModule") or {}).get("conditions") or []),
        "interventions": _joined(i.get("name") for i in interventions),
        "primary_outcomes": _joined(o.get("measure") for o in outcomes),
        "eligibility": _clean((protocol.get("eligibilityModule") or {}).get("eligibilityCriteria")),
        "summary": _clean((protocol.get("descriptionModule") or {}).get("briefSummary")),
    }


def build_table(trials: pl.DataFrame, studies: Iterator[dict]) -> pl.DataFrame:
    """One row per trial in `trials` (nct_id, split), in its order, with the text columns."""
    wanted = set(trials["nct_id"].to_list())
    rows = [
        extract(s)
        for s in studies
        if s["protocolSection"]["identificationModule"]["nctId"] in wanted
    ]
    text = pl.DataFrame(rows, schema={"nct_id": pl.String, **dict.fromkeys(ALL_COLUMNS, pl.String)})
    table = trials.select("nct_id", "split").join(text, on="nct_id", how="left")
    if text.height != trials.height:
        raise ValueError(f"{trials.height} trials listed but {text.height} records found.")
    return table


def result_like_shares(summaries: pl.Series) -> dict[str, float]:
    """Share of non-null summaries matching each result-like phrase, and any of them."""
    present = summaries.drop_nulls()
    shares = {
        label: float(present.str.contains(f"(?i){pattern}").mean())
        for label, pattern in RESULT_PHRASES.items()
    }
    anything = "|".join(f"(?:{p})" for p in RESULT_PHRASES.values())
    shares["any of these"] = float(present.str.contains(f"(?i){anything}").mean())
    return shares


def build(out_path: Path = OUT_PATH) -> pl.DataFrame:
    from trial_benchmark.download import iter_studies

    table = build_table(pl.read_csv(TRIALS_PATH), iter_studies())
    table.write_parquet(out_path)
    return table


def main() -> None:
    table = build()
    print(f"wrote {OUT_PATH} ({table.height} trials)")
    dev = table.filter(pl.col("split") == "dev")
    print(
        f"share of dev summaries with result-like phrases ({dev['summary'].drop_nulls().len()} summaries):"
    )
    for label, share in result_like_shares(dev["summary"]).items():
        print(f"  {label}: {share:.1%}")
    print("share of trials missing each column:")
    for column in ALL_COLUMNS:
        print(f"  {column}: {table[column].null_count() / table.height:.1%}")


if __name__ == "__main__":
    main()
