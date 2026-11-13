"""A language model as predictor: how much of its score is reasoning, how much is recall?

    uv run python -m elman_exercise.llm_predictor sample           # fix sample and batches -> results/llm_sample.csv
    uv run python -m elman_exercise.llm_predictor check-isolation --model haiku
    uv run python -m elman_exercise.llm_predictor run --model haiku sonnet --variant identified masked blinded
    uv run python -m elman_exercise.llm_predictor run --batches 1-2        # only whole frozen batches 1 and 2
    uv run python -m elman_exercise.llm_predictor leak-check       # what blinding leaves behind

This is a model under test. It is not part of the labels or the baseline. A Claude model is
shown a plain-text copy of a trial's registration record and asked how likely it is that at
least one treatment arm beat control at p < 0.05 on the primary outcome. Its training data may
contain the trial's published result, so the question for the report is how much of its score
comes from the design and how much from recall.

The recall probe is three prompts per trial:

- "identified": the registry id (NCT number), the official title and the lead sponsor's name,
  which are what a model would use to recognise a specific trial.
- "masked": none of those three, and no trial acronym or organisation study id; the sponsor class
  (industry, hospital, ...) only. Drug names, conditions, arms and outcomes stay in both, because
  removing them would change the task, not just the recognition. A model can still recognise a
  trial from a drug and design alone, so "masked" lowers recall; it does not remove it.
- "blinded": the masked prompt with every active drug, test or comparator, replaced by a
  placeholder ("Drug A"), in arm, condition and outcome text too, the start date dropped and
  post-hoc phrases cut out of arm and outcome text. Doses, schedules, design and outcome measures
  stay. See `llm_blinding` and `llm_posthoc` for exactly what is replaced and what is not.

Every prompt also asks for `drug_guess`: the main drug being tested, if the model can tell or guess
it. In the blinded prompt a correct guess means the model identified the trial's drug from the
design alone. The report scores the trials it could not name separately: that is the cleanest
number this study gives. Blinding lowers recall; it does not remove it, and name removal cannot
blind a model to a trial it has read about. The contrasts between variants mix recall with the
legitimate knowledge a forecaster has when a trial starts (the drug's class, its earlier results);
`llm_report` words them that way.

What the model sees (`PROMPT_FIELDS`, all read from the PROTOCOL section of the raw record)
-------------------------------------------------------------------------------------------
Phase, sponsor class, start date (not in "blinded"), design (allocation, model, masking, purpose), conditions,
interventions, arms (label, type, description), primary outcomes with time frames, and sex,
age limits and healthy-volunteer rule. Never shown: the results section, the brief summary
(sponsors rewrite it once results are in), status, why stopped, any completion date, or the
enrolment (the real figure is known only after the trial). All are "protocol" fields in the
sense of `dating`: the current version of the record, which a sponsor can edit after the start.
Text is cut to `MAX_TEXT` characters after names are replaced and masked, never before.

Isolation
---------
Each call is text in, text out, through the Claude Code command line (`claude -p`). What is
switched off:

- `--tools ""`: no tools at all (no web, no files, no shell). `check-isolation` asks the model to
  fetch a URL and list files and reads the tool list in the tool's own start-up message. Every
  `run` does this check first and stops if any tool, MCP server or skill is available.
- `--safe-mode`, `--setting-sources ""`, `--strict-mcp-config`, `--disable-slash-commands`: no
  user or project settings, memory, CLAUDE.md, MCP servers, skills or plugins are loaded.
- `--no-session-persistence`: nothing is saved or carried between calls.
- The call runs from a fresh empty temporary directory outside the repository, with a minimal
  environment: only PATH, HOME, USER and LOGNAME (found by testing: without USER and LOGNAME the
  tool cannot find its login).

What is NOT switched off. `--system-prompt` gives our own `SYSTEM_PROMPT` in place of the tool's
default one, but the tool still adds two blocks to the model's context, found by asking the model
to quote what it was given:

- an environment block: the empty temporary working directory, that it is not a git repository,
  the operating system and shell, the model's own name and id, the model's knowledge cut-off
  (both models report "June 2026") and today's date;
- a short block holding the account's email address and a line saying the context was attached
  automatically.

These are the same for every call, variant and trial (the date is the day of the call), so they
cannot explain a gap between variants. They are part of the conditions of every score: the model
is told its own cut-off and the date, and the prompt says nothing else about when the trial ran.
The calls go to the same provider and are made with the same account as everything else in this
project; no email address or other personal detail is stored in this repository.

The exact model id is read from the tool's JSON output (`modelUsage`) for every call. A call
fails if it is not the id asked for. The command line has no temperature setting, so a repeat of
the same batch can differ: `--run 1` repeats a batch and the report measures how much.

Sampling, batches and outputs
-----------------------------
`sample` draws a fixed random sample of 300 trials from each of three periods (development
trials finishing 2017 to 2020, test 2021 to 2023, late 2024 to 2025), stratified by phase so the
mix matches the period, and writes the ids to `results/llm_sample.csv` before any prediction is
made. The same file fixes the batches: every batch of ten trials holds trials of all three
periods, in a seeded shuffled order, and the batch and the position in it are columns of the file.
They are the same for every model, variant and run: a trial always has the same nine neighbours
in the same place. `--split` and `--limit` choose whole batches; a batch that failed is asked
again as the same batch. Predictions are features: no label is read to make them, so test and late
trials may be predicted. Scoring them belongs to the final logged run, not to this module.

`data/interim/llm_predictions.parquet`: one row per trial, model, variant and run, with the
probability, whether the model said it recognised the trial, its drug guess, its reason, the exact
model id, the batch and position, the sha256 of the prompt and (for a failed batch) the error
instead. Raw responses are in `data/interim/llm_raw/`, one file per call, each holding the exact
prompt sent, its sha256, the system prompt, the command line, the model id reported and a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path

import numpy as np
import polars as pl

from elman_exercise.dataset import ROOT
from elman_exercise.dating import PROTOCOL
from elman_exercise.download import iter_studies
from elman_exercise.llm_blinding import blind_record, leaked_tokens
from elman_exercise.llm_posthoc import flagged, strip_record

INTERIM = ROOT / "data" / "interim"
SAMPLE_PATH = ROOT / "results" / "llm_sample.csv"
PREDICTIONS_PATH = INTERIM / "llm_predictions.parquet"
RAW_DIR = INTERIM / "llm_raw"
DEV_PATH = INTERIM / "dev.parquet"
TEST_FEATURES_PATH = INTERIM / "test_features.parquet"

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}  # short name -> pinned id
VARIANTS = ("identified", "masked", "blinded")
PERIODS = {  # period label -> (split, first year, last year)
    "2017-2020": ("dev", 2017, 2020),
    "2021-2023": ("test", 2021, 2023),
    "2024-2025": ("late", 2024, 2025),
}
PER_PERIOD = 300
SAMPLE_SEED = 20261111
BATCH_SIZE = 10
CALL_TIMEOUT_SECONDS = 900
KEY_COLUMNS = ["nct_id", "model", "variant", "run"]
CALL_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME")  # all the subprocess gets

# Every field the prompt can show, and how it is dated. All are read from the protocol section.
PROMPT_FIELDS: dict[str, str] = {
    "phase": PROTOCOL,
    "sponsor_class": PROTOCOL,
    "start_date": PROTOCOL,
    "design": PROTOCOL,
    "conditions": PROTOCOL,
    "interventions": PROTOCOL,
    "arms": PROTOCOL,
    "primary_outcomes": PROTOCOL,
    "eligibility": PROTOCOL,
    "registry_id": PROTOCOL,  # identified only
    "title": PROTOCOL,  # identified only
    "sponsor_name": PROTOCOL,  # identified only
}
IDENTIFYING_FIELDS = ("registry_id", "title", "sponsor_name")  # shown in "identified" only
NOT_SHOWN = {  # variant -> the prompt fields it leaves out
    "identified": (),
    "masked": IDENTIFYING_FIELDS,
    "blinded": (*IDENTIFYING_FIELDS, "start_date"),
}

SYSTEM_PROMPT = (
    "You are an expert in clinical trial design and drug development. You will be given "
    "registration records of clinical trials and asked to estimate how likely each is to have "
    "had a positive result. Answer only from the text you are given and what you already know. "
    "Reply with the JSON requested and nothing else."
)

TASK = """\
Below are {n} clinical trial registration records. For each one, estimate the probability that \
the trial was positive, meaning at least one treatment arm beat its control arm at p < 0.05 on \
the primary outcome. Results are not shown. Judge each trial on its own.

For each trial also say whether you recognise it: true only if you believe you know this \
specific trial or its published result, false if you are judging from the design alone.

Reply with a JSON array and nothing else, one object per trial, in order:
[{{"trial": 1, "probability": 0.55, "recognised": false, "drug_guess": "name or null", \
"reason": "One sentence."}}, ...]
"probability" is a number from 0 to 1. "recognised" is true or false. "drug_guess" is the name of \
the main drug being tested, if you can tell or guess it, otherwise null. "reason" is one sentence.

{records}"""

MAX_ARMS = 8
MAX_OUTCOMES = 5
MAX_TEXT = 300
NO_GUESS = {"", "null", "none", "unknown", "n/a", "na"}


# ---------------------------------------------------------------- the record


def _clean(value: str | None) -> str | None:
    value = " ".join((value or "").split())
    return value or None


def _words(value: str | None) -> str | None:
    """An enum such as PLACEBO_COMPARATOR as plain words."""
    return value.replace("_", " ").lower() if value else None


def _cut(text: str | None, limit: int = MAX_TEXT) -> str | None:
    """`text` cut to `limit` characters. Done last, after names are replaced and masked."""
    if text is None or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def extract_record(study: dict) -> dict:
    """The protocol-only fields of one raw record that the prompt may show.

    Reads the protocol section and nothing else. In particular it never opens the results
    section, the derived section, the brief summary, the status module (except the start date),
    the enrolment or any completion date. Text is kept whole here; `render_trial` cuts it.
    `post_hoc` says whether a sponsor's after-the-fact phrase is in the outcome or arm text.
    """
    protocol = study["protocolSection"]
    ident = protocol.get("identificationModule") or {}
    sponsors = protocol.get("sponsorCollaboratorsModule") or {}
    sponsor = sponsors.get("leadSponsor") or {}
    design = protocol.get("designModule") or {}
    info = design.get("designInfo") or {}
    arms_module = protocol.get("armsInterventionsModule") or {}
    eligibility = protocol.get("eligibilityModule") or {}
    start = (protocol.get("statusModule") or {}).get("startDateStruct") or {}
    arm_groups = arms_module.get("armGroups") or []
    outcomes = (protocol.get("outcomesModule") or {}).get("primaryOutcomes") or []
    other_ids = [i.get("id") for i in ident.get("secondaryIdInfos") or []]
    record = {
        "nct_id": ident["nctId"],
        "title": _clean(ident.get("officialTitle")) or _clean(ident.get("briefTitle")),
        "acronym": _clean(ident.get("acronym")),
        "own_ids": [
            i
            for i in map(_clean, [(ident.get("orgStudyIdInfo") or {}).get("id"), *other_ids])
            if i and not i.startswith("NCT")
        ],
        "sponsor_name": _clean(sponsor.get("name")),
        "sponsor_class": _words(sponsor.get("class")),
        "collaborators": [
            n
            for n in map(_clean, (c.get("name") for c in sponsors.get("collaborators") or []))
            if n
        ],
        "phase": " / ".join(_words(p) for p in design.get("phases") or []) or None,
        "start_date": _clean(start.get("date")),
        "design": {
            "allocation": _words(info.get("allocation")),
            "model": _words(info.get("interventionModel")),
            "masking": _words((info.get("maskingInfo") or {}).get("masking")),
            "purpose": _words(info.get("primaryPurpose")),
            "n_arms": len(arm_groups),
        },
        "conditions": [
            c
            for c in map(_clean, (protocol.get("conditionsModule") or {}).get("conditions") or [])
            if c
        ],
        "interventions": [
            f"{_words(i.get('type')) or 'other'}: {_clean(i.get('name'))}"
            for i in arms_module.get("interventions") or []
            if _clean(i.get("name"))
        ],
        "intervention_details": [  # for blinding only: the names a model could recognise
            {
                "type": _words(i.get("type")),
                "name": _clean(i.get("name")),
                "other_names": [n for n in map(_clean, i.get("otherNames") or []) if n],
            }
            for i in arms_module.get("interventions") or []
            if _clean(i.get("name"))
        ],
        "arms": [
            {
                "label": _clean(a.get("label")),
                "type": _words(a.get("type")),
                "description": _clean(a.get("description")),
                "interventions": [_clean(n) for n in a.get("interventionNames") or [] if _clean(n)],
            }
            for a in arm_groups
        ],
        "primary_outcomes": [
            {"measure": _clean(o.get("measure")), "time_frame": _clean(o.get("timeFrame"))}
            for o in outcomes
        ],
        "eligibility": {
            "sex": _words(eligibility.get("sex")),
            "minimum_age": _clean(eligibility.get("minimumAge")),
            "maximum_age": _clean(eligibility.get("maximumAge")),
            "healthy_volunteers": eligibility.get("healthyVolunteers"),
        },
    }
    return {**record, "post_hoc": flagged(record)}


def _design_line(design: dict) -> str | None:
    parts = [
        design["allocation"],
        design["model"],
        design["masking"] and f"masking: {design['masking']}",
    ]
    parts += [design["purpose"] and f"purpose: {design['purpose']}"]
    parts += [design["n_arms"] and f"{design['n_arms']} arms"]
    return "; ".join(p for p in parts if p) or None


def _eligibility_line(eligibility: dict) -> str | None:
    ages = [eligibility["minimum_age"], eligibility["maximum_age"]]
    healthy = eligibility["healthy_volunteers"]
    parts = [
        eligibility["sex"] and f"sex: {eligibility['sex']}",
        any(ages) and f"age: {ages[0] or 'no minimum'} to {ages[1] or 'no maximum'}",
        healthy is not None and f"healthy volunteers: {'yes' if healthy else 'no'}",
    ]
    return "; ".join(p for p in parts if p) or None


def _arm_line(arm: dict) -> str:
    line = f"- {_cut(arm['label']) or 'unnamed'}"
    if arm["type"]:
        line += f" ({arm['type']})"
    if arm["description"]:
        line += f": {_cut(arm['description'])}"
    if arm["interventions"]:
        line += f" [{'; '.join(arm['interventions'])}]"
    return line


# ---------------------------------------------------------------- masking

# Words that end a company name without being the company: "Novartis Pharmaceuticals" is Novartis.
LEGAL_WORDS = frozenset(
    """
    inc llc ltd limited corp corporation co company gmbh ag sa plc pharmaceuticals pharmaceutical
    pharma therapeutics laboratories laboratory biosciences development research and kg bv nv lp
    usa us international group holdings
    """.split()  # noqa: SIM905
)
# Words of an industry sponsor's name that are not the company's own and are left unmasked on their own.
COMMON_WORDS = LEGAL_WORDS | {"medical", "health", "science", "sciences", "institute", "american"}
MIN_MASKED_WORD = 4


def sponsor_names(record: dict) -> set[str]:
    """Every text that stands for the sponsor or a collaborator, longest first when used.

    The name as written, and without legal endings ("Eli Lilly and Company" -> "Eli Lilly"). For an
    industry sponsor also each distinctive word of it ("Lilly"), because arm text uses the short
    name. For other sponsors only the whole names, so "University" is not removed from a text.
    """
    found = set()
    for name in [record["sponsor_name"], *record["collaborators"]]:
        if not name:
            continue
        words = name.replace(",", " ").split()
        while len(words) > 1 and words[-1].lower().strip(".&") in LEGAL_WORDS | {"&"}:
            words.pop()
        found |= {name, " ".join(words)}
        if name == record["sponsor_name"] and record["sponsor_class"] == "industry":
            found |= {
                w
                for w in re.findall(r"[A-Za-z]+", " ".join(words))
                if len(w) >= MIN_MASKED_WORD and w.lower() not in COMMON_WORDS
            }
    return found


def _whole_word(texts: Iterable[str]) -> re.Pattern | None:
    """A pattern for any of `texts` as whole words, case aside, or None if there are none."""
    ordered = sorted({t for t in texts if t}, key=lambda t: (-len(t), t))
    if not ordered:
        return None
    body = "|".join(r"\s+".join(re.escape(w) for w in t.split()) for t in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def mask(text: str | None, record: dict) -> str | None:
    """Remove the sponsor's names, the trial's acronym and study ids, and any registry id.

    Only whole words are replaced, never part of a longer word.
    """
    if text is None:
        return None
    acronym = [record["acronym"]] if record["acronym"] and len(record["acronym"]) >= 3 else []
    for texts, replacement in (
        (sponsor_names(record), "the sponsor"),
        (acronym, "[study name removed]"),
        ([i for i in record["own_ids"] if len(i) >= 4], "[id removed]"),
    ):
        if pattern := _whole_word(texts):
            text = pattern.sub(replacement, text)
    return re.sub(r"NCT\d{8}", "[id removed]", text)


def map_text(record: dict, change: Callable[[str | None], str | None]) -> dict:
    """A copy of `record` with `change` applied to every free-text field the prompt shows."""
    return {
        **record,
        "conditions": [change(c) for c in record["conditions"]],
        "interventions": [change(i) for i in record["interventions"]],
        "arms": [
            {
                **a,
                "label": change(a["label"]),
                "description": change(a["description"]),
                "interventions": [change(n) for n in a["interventions"]],
            }
            for a in record["arms"]
        ],
        "primary_outcomes": [
            {"measure": change(o["measure"]), "time_frame": change(o["time_frame"])}
            for o in record["primary_outcomes"]
        ],
    }


def render_trial(record: dict, variant: str, label: int) -> str:
    """One trial as plain text under the neutral label "Trial <label>".

    The "masked" variant leaves out the registry id, title and sponsor name, and also replaces
    the sponsor's names, the acronym, the study ids and any registry id if they turn up inside
    an arm, condition or outcome text. The "blinded" variant is the masked one with every active
    drug (test or comparator) replaced by a placeholder, no start date and the post-hoc phrases
    cut out. The order is fixed: replace names, cut post-hoc clauses, mask, then cut to length.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, not {variant!r}")
    if variant == "blinded":
        record = strip_record(blind_record(record))
    if variant != "identified":
        record = map_text(record, lambda text: mask(text, record))
    lines = [f"Trial {label}"]
    if variant == "identified":
        lines += [f"Registry id: {record['nct_id']}", f"Title: {record['title']}"]
    sponsor = record["sponsor_class"] or "unknown"
    if variant == "identified" and record["sponsor_name"]:
        sponsor = f"{record['sponsor_name']} ({sponsor})"
    lines += [
        f"Phase: {record['phase'] or 'unknown'}",
        f"Lead sponsor: {sponsor}",
    ]
    if variant != "blinded":
        lines.append(f"Start date: {record['start_date'] or 'unknown'}")
    for name, text in (
        ("Design", _design_line(record["design"])),
        ("Conditions", "; ".join(record["conditions"]) or None),
        ("Interventions", "; ".join(record["interventions"]) or None),
        ("Eligibility", _eligibility_line(record["eligibility"])),
    ):
        if text:
            lines.append(f"{name}: {text}")
    if record["arms"]:
        lines.append("Arms:")
        lines += [_arm_line(a) for a in record["arms"][:MAX_ARMS]]
    if record["primary_outcomes"]:
        lines.append("Primary outcomes:")
        for outcome in record["primary_outcomes"][:MAX_OUTCOMES]:
            frame = _cut(outcome["time_frame"])
            when = f" (time frame: {frame})" if frame else ""
            lines.append(f"- {_cut(outcome['measure']) or 'unnamed outcome'}{when}")
    return "\n".join(lines)


def leaked_words(record: dict) -> list[str]:
    """Words of this trial's intervention names still in its blinded text (see `llm_blinding`)."""
    return leaked_tokens(render_trial(record, "blinded", 1), record)


def leak_counts(records: Iterable[dict]) -> dict[str, int]:
    """How many trials still show a word or code of their own intervention names once blinded."""
    records = list(records)
    return {"trials": len(records), "any": sum(bool(leaked_words(r)) for r in records)}


def render_prompt(records: list[dict], variant: str) -> str:
    """The user message for one batch: the task, then the trials labelled 1, 2, 3, ..."""
    rendered = "\n\n".join(render_trial(r, variant, i) for i, r in enumerate(records, start=1))
    return TASK.format(n=len(records), records=rendered)


# ---------------------------------------------------------------- the answer


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_response(text: str, n: int) -> list[dict]:
    """The model's answer as one dict per trial, in order. Raises ValueError if anything is off.

    Strict: the reply must be a JSON array (one surrounding code fence is tolerated) of exactly
    `n` objects labelled 1..n, each with a probability between 0 and 1, a true/false
    `recognised`, a `drug_guess` that is text or null, and a `reason` string. A bad reply is never
    repaired or guessed. An empty guess, or one that says "null" or "unknown", is no guess.
    """
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    items = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(items, list) or len(items) != n:
        raise ValueError(f"expected a JSON array of {n} objects")
    parsed = []
    for expected, item in enumerate(items, start=1):
        if not isinstance(item, dict) or item.get("trial") != expected:
            raise ValueError(f"item {expected} is not labelled trial {expected}")
        probability, recognised, reason = (
            item.get(k) for k in ("probability", "recognised", "reason")
        )
        is_number = isinstance(probability, int | float) and not isinstance(probability, bool)
        require(is_number and 0 <= probability <= 1, f"trial {expected}: probability is not 0 to 1")
        require(isinstance(recognised, bool), f"trial {expected}: recognised is not true or false")
        require(isinstance(reason, str), f"trial {expected}: reason is not text")
        require("drug_guess" in item, f"trial {expected}: drug_guess is missing")
        guess = item["drug_guess"]
        require(
            guess is None or isinstance(guess, str), f"trial {expected}: drug_guess is not text"
        )
        guess = (guess or "").strip()
        parsed.append(
            {
                "probability": float(probability),
                "recognised": recognised,
                "drug_guess": None if guess.lower() in NO_GUESS else guess,
                "reason": reason,
            }
        )
    return parsed


# ---------------------------------------------------------------- the sample


def largest_remainder(shares: dict[str, float], total: int) -> dict[str, int]:
    """Whole numbers close to `shares * total` that add to `total`."""
    exact = {k: v * total for k, v in shares.items()}
    counts = {k: int(v) for k, v in exact.items()}
    by_remainder = sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))
    for key in by_remainder[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def draw_sample(frame: pl.DataFrame, n: int = PER_PERIOD, seed: int = SAMPLE_SEED) -> pl.DataFrame:
    """`n` trials from `frame` (nct_id, split, primary_completion_date, phase), phase mix kept.

    Seeded, and independent of row order: trials are sorted by id before drawing. Every period
    is drawn from the same seed, so the sample does not change if another period changes. A trial
    with no phase is drawn as its own group.
    """
    frame = frame.with_columns(pl.col("phase").fill_null("unknown"))
    chosen = []
    for period, (split, first, last) in PERIODS.items():
        year = pl.col("primary_completion_date").dt.year()
        pool = frame.filter((pl.col("split") == split) & year.is_between(first, last))
        pool = pool.sort("nct_id")
        sizes = pool.group_by("phase").len()
        counts = largest_remainder(
            {p: c / pool.height for p, c in zip(sizes["phase"], sizes["len"], strict=True)}, n
        )
        rng = np.random.default_rng(seed)
        for phase, count in sorted(counts.items()):
            ids = pool.filter(pl.col("phase") == phase)["nct_id"].to_list()
            for nct_id in rng.choice(ids, size=count, replace=False):
                chosen.append({"nct_id": str(nct_id), "split": split, "period": period})
    return pl.DataFrame(chosen).sort("period", "nct_id")


def freeze_batches(
    sample: pl.DataFrame, seed: int = SAMPLE_SEED, size: int = BATCH_SIZE
) -> pl.DataFrame:
    """`sample` with a `batch` (1, 2, ...) and a `position` (1 to `size`) for every trial.

    Each period is shuffled with the seed, the periods are dealt out in turn so every batch holds
    trials of all three, and the order inside a batch is shuffled again. This is done once, from the
    whole sample, and written to the file: the same for every model, variant and run.
    """
    rng = np.random.default_rng(seed)
    shuffled = []
    for period in PERIODS:
        ids = sorted(sample.filter(pl.col("period") == period)["nct_id"].to_list())
        shuffled.append([ids[i] for i in rng.permutation(len(ids))])
    dealt = [i for turn in zip_longest(*shuffled) for i in turn if i is not None]
    rows = []
    for number, start in enumerate(range(0, len(dealt), size), start=1):
        group = dealt[start : start + size]
        for position, i in enumerate(rng.permutation(len(group)), start=1):
            rows.append({"nct_id": group[i], "batch": number, "position": position})
    return sample.join(pl.DataFrame(rows), on="nct_id", how="left", validate="1:1").sort(
        "batch", "position"
    )


def load_sampling_frame() -> pl.DataFrame:
    """Id, split, completion date and phase of every trial. Features only: no label is read."""
    columns = ["nct_id", "split", "primary_completion_date", "phase"]
    return pl.concat(
        [
            pl.read_parquet(DEV_PATH, columns=columns),
            pl.read_parquet(TEST_FEATURES_PATH, columns=columns),
        ]
    )


def read_sample(path: Path = SAMPLE_PATH) -> pl.DataFrame:
    return pl.read_csv(path)


def sample_batches(sample: pl.DataFrame) -> dict[int, list[str]]:
    """Batch number -> its trial ids in position order, as frozen in the sample file."""
    ordered = sample.sort("batch", "position")
    return {
        int(number): group["nct_id"].to_list()
        for (number,), group in ordered.group_by("batch", maintain_order=True)
    }


def parse_batch_numbers(text: str) -> set[int]:
    """ "1-2" -> {1, 2}; "5" -> {5}; "1-3,7" -> {1, 2, 3, 7}."""
    numbers: set[int] = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        numbers |= set(range(int(first), int(last or first) + 1))
    return numbers


def choose_batches(
    sample: pl.DataFrame,
    split: str | None = None,
    limit: int | None = None,
    only: set[int] | None = None,
) -> dict[int, list[str]]:
    """The frozen batches to work on. Whole batches only: a batch is never cut or rebuilt.

    `only` keeps the numbered batches, `split` the batches holding at least one trial of that
    split, `limit` the first N of what is left.
    """
    in_split = set(sample.filter(pl.col("split") == split)["batch"]) if split else None
    chosen = {
        number: ids
        for number, ids in sample_batches(sample).items()
        if (only is None or number in only) and (in_split is None or number in in_split)
    }
    return dict(list(chosen.items())[:limit])


# ---------------------------------------------------------------- calling the model


def claude_command(
    model_id: str, system_prompt: str = SYSTEM_PROMPT, output: str = "json"
) -> list[str]:
    """The command line for one isolated, tool-free call. The prompt goes in on stdin."""
    command = [
        "claude", "-p",
        "--model", model_id,
        "--tools", "",
        "--safe-mode",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--system-prompt", system_prompt,
        "--output-format", output,
    ]  # fmt: skip
    return command + (["--verbose"] if output == "stream-json" else [])


def call_environment() -> dict[str, str]:
    """The only environment variables the subprocess gets (those of `CALL_ENVIRONMENT` that are set)."""
    return {k: os.environ[k] for k in CALL_ENVIRONMENT if k in os.environ}


def call_claude(prompt: str, model_id: str, output: str = "json") -> str:
    """Run one call from a fresh empty temporary directory and return its standard output."""
    with tempfile.TemporaryDirectory(prefix="llm-call-") as empty:
        done = subprocess.run(
            claude_command(model_id, output=output),
            input=prompt,
            capture_output=True,
            text=True,
            cwd=empty,
            env=call_environment(),
            timeout=CALL_TIMEOUT_SECONDS,
            check=False,
        )
    if done.returncode != 0:
        raise RuntimeError(f"claude exited {done.returncode}: {done.stderr.strip()[:500]}")
    return done.stdout


def read_output(stdout: str, expected_model_id: str) -> tuple[str, str]:
    """(answer text, model id) from the tool's JSON output; refuses any use of tools.

    Raises RuntimeError unless the tool reports exactly the model id that was asked for.
    """
    out = json.loads(stdout)
    if out.get("is_error"):
        raise RuntimeError(f"claude reported an error: {str(out.get('result'))[:300]}")
    if out.get("permission_denials") or out.get("num_turns") != 1:
        raise RuntimeError("the model tried to use a tool or took more than one turn")
    tool_use = (out.get("usage") or {}).get("server_tool_use") or {}
    if any(tool_use.values()):
        raise RuntimeError(f"server tools were used: {tool_use}")
    reported = sorted(out["modelUsage"])
    if reported != [expected_model_id]:
        raise RuntimeError(f"asked for {expected_model_id} but the tool reports {reported}")
    return out["result"], reported[0]


PROBE = (
    "Do two things. First fetch https://example.com and tell me its page title. Second, list "
    "the files in your current directory. If you cannot do either, say plainly which tools you lack."
)


def probe_isolation(model_id: str) -> dict:
    """Ask for a fetch and a file listing, and report the tools the tool itself says exist.

    Raises RuntimeError if any tool, MCP server, skill or plug-in command is available, or if
    the tool's start-up message names a model other than the one asked for.
    """
    stdout = call_claude(PROBE, model_id, output="stream-json")
    events = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    start = next(e for e in events if e.get("type") == "system" and "tools" in e)
    final = next(e for e in events if e.get("type") == "result")
    found = {k: start.get(k) for k in ("tools", "mcp_servers", "skills", "slash_commands")}
    if any(found.values()) or final.get("permission_denials"):
        raise RuntimeError(f"Isolation failed. The model has access to: {found}")
    if start["model"] != model_id:
        raise RuntimeError(f"asked for {model_id} but the tool reports {start['model']}")
    return {**found, "model": start["model"], "answer": final["result"]}


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def predict_batch(
    batch: int,
    nct_ids: list[str],
    records: dict[str, dict],
    model: str,
    variant: str,
    run: int,
    call=call_claude,
) -> tuple[list[dict], dict]:
    """Ask for one frozen batch, retrying once if the call or the parse fails.

    Returns (one dict per trial, the raw record of the call). The raw record holds the exact prompt
    and its sha256, the system prompt, the command line, the environment variable names, every
    attempt's output with a time, and the model id the tool reported. Each row dict has `error`
    set, and no probability, if both attempts failed: a failure is recorded, never guessed.
    """
    prompt = render_prompt([records[i] for i in nct_ids], variant)
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    raw = {
        "batch": batch,
        "model": model,
        "model_requested": MODELS[model],
        "variant": variant,
        "run": run,
        "nct_ids": nct_ids,
        "prompt": prompt,
        "prompt_sha256": sha,
        "system_prompt": SYSTEM_PROMPT,
        "command": claude_command(MODELS[model]),
        "environment_variables": sorted(call_environment()),
        "model_id_reported": None,
        "attempts": [],
    }
    for _ in range(2):
        attempt: dict = {"started": now()}
        raw["attempts"].append(attempt)
        try:
            stdout = call(prompt, MODELS[model])
            attempt["stdout"] = stdout
            text, model_id = read_output(stdout, MODELS[model])
            parsed = parse_response(text, len(nct_ids))
        except (RuntimeError, ValueError, subprocess.TimeoutExpired, KeyError) as error:
            attempt["error"] = f"{type(error).__name__}: {error}"[:500]
            continue
        raw["model_id_reported"] = model_id
        rows = [
            {"nct_id": i, "model": model, "variant": variant, "run": run, **p, "model_id": model_id,
             "batch": batch, "position": position, "prompt_sha256": sha, "error": None}
            for position, (i, p) in enumerate(zip(nct_ids, parsed, strict=True), start=1)
        ]  # fmt: skip
        return rows, raw
    error = raw["attempts"][-1]["error"]
    rows = [
        {"nct_id": i, "model": model, "variant": variant, "run": run, "probability": None,
         "recognised": None, "drug_guess": None, "reason": None, "model_id": None, "batch": batch,
         "position": position, "prompt_sha256": sha, "error": error}
        for position, i in enumerate(nct_ids, start=1)
    ]  # fmt: skip
    return rows, raw


# ---------------------------------------------------------------- the predictions table

TABLE_SCHEMA = {
    "nct_id": pl.String, "model": pl.String, "variant": pl.String, "run": pl.Int64,
    "probability": pl.Float64, "recognised": pl.Boolean, "drug_guess": pl.String,
    "reason": pl.String, "model_id": pl.String, "batch": pl.Int64, "position": pl.Int64,
    "prompt_sha256": pl.String, "error": pl.String,
}  # fmt: skip


def read_predictions(path: Path = PREDICTIONS_PATH) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=TABLE_SCHEMA)


def add_predictions(table: pl.DataFrame, rows: list[dict]) -> pl.DataFrame:
    """`table` plus `rows`; a new row replaces an older one for the same trial, model, variant, run."""
    new = pl.DataFrame(rows, schema=TABLE_SCHEMA)
    kept = table.join(new.select(KEY_COLUMNS), on=KEY_COLUMNS, how="anti")
    return pl.concat([kept, new]).sort(KEY_COLUMNS)


def pending(table: pl.DataFrame, order: list[str], model: str, variant: str, run: int) -> list[str]:
    """The ids in `order` with no successful prediction yet for this model, variant and run."""
    done = table.filter(
        (pl.col("model") == model) & (pl.col("variant") == variant) & (pl.col("run") == run)
        & pl.col("probability").is_not_null()
    )["nct_id"]  # fmt: skip
    finished = set(done.to_list())
    return [i for i in order if i not in finished]


def save_atomic(table: pl.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    table.write_parquet(tmp)
    os.replace(tmp, path)


def load_records(nct_ids: Iterable[str]) -> dict[str, dict]:
    """The protocol-only record of each trial, found by one pass over the raw download (~30 s)."""
    wanted = set(nct_ids)
    found = {}
    for study in iter_studies():
        nct_id = study["protocolSection"]["identificationModule"]["nctId"]
        if nct_id in wanted:
            found[nct_id] = extract_record(study)
    if missing := wanted - set(found):
        raise ValueError(f"{len(missing)} sampled trials are not in the raw download")
    return found


def run_all(
    models: list[str], variants: list[str], split: str | None, limit: int | None,
    only: set[int] | None, run: int, workers: int,
    predictions_path: Path = PREDICTIONS_PATH, call: Callable = call_claude,
) -> None:  # fmt: skip
    """Predict every frozen batch that has a trial with no successful prediction yet. Resumable.

    A batch is asked as the whole batch it was frozen as, never rebuilt from what is left, so a
    trial's companions and position are the same for every model, variant and run.
    """
    sample = read_sample()
    chosen = choose_batches(sample, split, limit, only)
    for model in models:
        print(f"isolation check, {model}: ", end="", flush=True)
        probe_isolation(MODELS[model])
        print("no tools, no MCP servers, no skills")
    records = load_records([i for ids in chosen.values() for i in ids])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    table = read_predictions(predictions_path)
    jobs = [
        (number, ids, model, variant)
        for model in models
        for variant in variants
        for number, ids in chosen.items()
        if pending(table, ids, model, variant, run)
    ]
    print(f"{len(jobs)} calls to make")
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(predict_batch, number, ids, records, model, variant, run, call)
            for number, ids, model, variant in jobs
        ]
        for done, future in enumerate(futures, start=1):
            rows, raw = future.result()
            name = f"{raw['model']}_{raw['variant']}_run{run}_b{raw['batch']:03d}.json"
            (RAW_DIR / name).write_text(json.dumps(raw, indent=1))
            table = add_predictions(table, rows)
            save_atomic(table, predictions_path)
            failed = " FAILED" if rows[0]["error"] else ""
            print(f"  {done}/{len(jobs)} {raw['model']} {raw['variant']} batch {raw['batch']}"
                  f"{failed} ({time.time() - started:.0f}s)", flush=True)  # fmt: skip


# ---------------------------------------------------------------- command line


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sample", help="draw the sample and batches, write results/llm_sample.csv")
    check = commands.add_parser("check-isolation", help="prove the model has no tools")
    check.add_argument("--model", choices=MODELS, required=True)
    commands.add_parser("leak-check", help="how often blinding leaves an intervention name word")
    run = commands.add_parser("run", help="predict frozen batches; skips those already done")
    run.add_argument("--model", nargs="+", choices=MODELS, default=list(MODELS))
    run.add_argument("--variant", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    run.add_argument(
        "--split", choices=("dev", "test", "late"), help="only batches with this split"
    )
    run.add_argument("--limit", type=int, help="only the first N batches")
    run.add_argument("--batches", help='only these batches, e.g. "1-2" or "1-3,7"')
    run.add_argument(
        "--run", type=int, default=0, help="repeat number; 1 re-asks to measure variation"
    )
    run.add_argument("--workers", type=int, default=4, help="calls in parallel")
    args = parser.parse_args()

    if args.command == "sample":
        if SAMPLE_PATH.exists():
            raise SystemExit(
                f"{SAMPLE_PATH} exists. The sample is fixed once; delete it to redraw."
            )
        sample = draw_sample(load_sampling_frame())
        records = load_records(sample["nct_id"])
        flags = pl.DataFrame(
            {"nct_id": list(records), "post_hoc": [r["post_hoc"] for r in records.values()]}
        )
        sample = freeze_batches(sample.join(flags, on="nct_id", validate="1:1"))
        sample.select("nct_id", "split", "period", "post_hoc", "batch", "position").write_csv(
            SAMPLE_PATH
        )
        print(sample.group_by("period").len().sort("period"))
    elif args.command == "leak-check":
        sample = read_sample()
        records = load_records(sample["nct_id"])
        for period in PERIODS:
            ids = sample.filter(pl.col("period") == period)["nct_id"]
            flagged_now = sum(records[i]["post_hoc"] for i in ids)
            print(period, leak_counts(records[i] for i in ids), "post-hoc phrase:", flagged_now)
    elif args.command == "check-isolation":
        print(json.dumps(probe_isolation(MODELS[args.model]), indent=1))
    else:
        only = parse_batch_numbers(args.batches) if args.batches else None
        run_all(args.model, args.variant, args.split, args.limit, only, args.run, args.workers)


if __name__ == "__main__":
    sys.exit(main())
