"""What kind of thing a trial's first primary outcome measures, and how long it takes.

    uv run python -m trial_benchmark.endpoint   # -> data/interim/endpoint.parquet

One row per trial, for every split. Features only: it reads the protocol section of
each record and never the results section, so it holds no outcome and no label.

The features
------------
- `endpoint_type`: the first registered primary outcome, filed into one of eight
  kinds (`TYPES`). A rule-based reader looks at the outcome's title first and, only
  if the title matches nothing, at the start of its description. Anything it cannot
  place is "other / unclear": a wrong confident answer costs more than an honest one.
- `time_frame_weeks`: the outcome's time frame, in weeks. When the text lists several
  times ("Week 4, 8 and 12") the longest is taken. Null when no time can be read.
- `n_primary_outcomes`: how many primary outcomes are registered.
- `registered_before_start`: the record was first submitted before the trial started.
  A month-only start date counts as the first of that month, so this is true only when
  registration was clearly earlier. Null when either date is missing.

All four are "protocol"
-----------------------
The record is the current version, not the one filed at registration (see `dating`:
the pre-registered baseline reads the same record, so honest models may use these). Sponsors can
reword an outcome, or add and drop primary outcomes, when they post results. So these
features could carry information from after the trial. What bounds the worry: on the
development set the gap in success rate between biomarker first-primaries and the rest is
about the same whether the trial was registered before it started (70.5% against 51.1%,
3,545 trials) or after (76.0% against 58.7%, 2,302 trials); for survival endpoints it is
12.5 points lower before and 20.1 points lower after. Crude gaps, with no adjustment.
`registered_before_start` is there so an analysis can restrict to the first group. That is
a bound, not proof: an outcome reworded after the trial started could still be in the
record, whenever it was first registered.

How the reader was checked
--------------------------
Rules were written, then tuned on one random sample of 60 development trials, then
frozen and scored on a second, fresh sample of 60. In both, the "true" type was a
reading by a language model (Claude), not by a person. A model's reading is
consistent but can share the rules' blind spots, so treat the error rates as rough.
The second sample, with the reading beside the assigned type, is in
results/endpoint_type_read.csv so a person can audit it.

Boundary calls the rules make on purpose
----------------------------------------
- A response or remission defined on a named scale (PASI 75, ACR20, DAS28 remission)
  is a symptom or function scale, not an event.
- Tumour response, remission, cure, relapse and similar are clinical events.
- Bone density and other quantities read off a scan or X-ray are imaging.
- Lung function, glucose, blood pressure, antibodies and other tests of the body
  are laboratory or physiological biomarkers.
- Areas under the curve of glucose or C-peptide are biomarkers; of a drug level, pharmacokinetic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date

import polars as pl

from trial_benchmark import dev_summary
from trial_benchmark.dataset import ROOT, parse_date
from trial_benchmark.dating import PROTOCOL
from trial_benchmark.download import iter_studies

TABLE = "endpoint"
OUT_PATH = ROOT / "data" / "interim" / "endpoint.parquet"
TRIALS_PATH = ROOT / "trials.csv"

FEATURES = {
    "endpoint_type": PROTOCOL,
    "time_frame_weeks": PROTOCOL,
    "n_primary_outcomes": PROTOCOL,
    "registered_before_start": PROTOCOL,
}

SURVIVAL = "survival or time-to-event"
EVENT = "clinical event"
SYMPTOM = "symptom or function scale"
BIOMARKER = "laboratory or physiological biomarker"
PHARMACOKINETIC = "pharmacokinetic"
IMAGING = "imaging"
SAFETY = "safety or tolerability"
OTHER = "other / unclear"
TYPES = (SURVIVAL, EVENT, SYMPTOM, BIOMARKER, PHARMACOKINETIC, IMAGING, SAFETY, OTHER)

DESCRIPTION_CHARS = 300  # how much of the description is read when the title says nothing


# ---------------------------------------------------------------- the rules


def phrases(*fragments: str) -> re.Pattern:
    """Whole words or phrases, any case. Write a stem as `tolerab\\w*`."""
    return re.compile(r"\b(?:" + "|".join(fragments) + r")\b", re.IGNORECASE)


def acronyms(*fragments: str) -> re.Pattern:
    """Whole acronyms, exact case, so that eGFR (a blood test) is not EGFR (a gene)."""
    return re.compile(r"\b(?:" + "|".join(fragments) + r")\b")


# Each rule is (type, pattern). The first rule that matches the text wins, so the order
# is part of the logic. Every pattern has an example of what it is for.
RULES: list[tuple[str, re.Pattern]] = [
    # "Area under the glucose curve", "FEV1 AUC": the body's response to a drug, not the drug's level.
    (
        BIOMARKER,
        re.compile(
            r"\bAUC\b.*\b(?:glucose|c-peptide|insulin|FEV1)|(?:glucose|c-peptide|insulin|FEV1)\b.*\bAUC\b",
            re.IGNORECASE,
        ),
    ),
    # "Cmax of the study drug", "plasma concentration", "bioequivalence".
    (
        PHARMACOKINETIC,
        phrases(
            r"pharmacokinetic\w*",
            r"(?:trough|peak) (?:drug )?(?:concentration|level)s?",
            r"(?:plasma|serum|blood) drug (?:concentration|level)s?",
            r"(?:plasma|serum|blood) (?:concentration|level)s? of (?:the )?(?:study |investigational )?(?:drug|product|medication|treatment)",
            r"half[- ]life",
            r"bioavailability",
            r"bioequivalence",
            r"time to (?:peak|maximum) concentration",
        ),
    ),
    (PHARMACOKINETIC, acronyms("Cmax", "Cmin", "Ctrough", "Tmax", "PK")),
    # An area under the curve is a drug exposure only in a drug context: "AUC0-24", "AUC of plasma drug",
    # not "pain AUC" or "viral load AUC".
    (
        PHARMACOKINETIC,
        re.compile(
            r"\bAUC(?:last|inf|tau|0|\d|\()"
            r"|(?=.*(?i:\b(?:drug|plasma|serum|concentration|exposure|dose)\b)).*\bAUC\b",
            re.DOTALL,
        ),
    ),
    # "Number of participants with adverse events", "dose-limiting toxicity", "tolerability".
    (
        SAFETY,
        phrases(
            r"adverse (?:event|events|experience|experiences|reaction|reactions|effect|effects)",
            r"safety",
            r"tolerab\w*",
            r"toxicit\w+",
            r"side effects?",
            r"treatment[- ]emergent",
            r"discontinu\w+ (?:due to|because of)",
        ),
    ),
    (SAFETY, acronyms("AEs?", "SAEs?", "TEAEs?", "DLTs?")),
    # A relapse, progression or death counted together as one composite: "disease progression, relapse or death".
    (SURVIVAL, phrases(r"progression,? (?:relapse,? )?or death", r"relapse,? or death")),
    # "Overall survival", "progression-free survival", "time to first seizure", "all-cause mortality".
    (
        SURVIVAL,
        phrases(
            r"survival",
            r"time[- ]to",
            r"time from [\w ]+ to",
            r"mortality",
            r"(?:progression|relapse|recurrence|event|disease|failure)[- ]free",
            r"duration of (?:response|remission|survival)",
            r"alive",
            r"hazard ratio",
        ),
    ),
    (SURVIVAL, acronyms("PFS", "OS", "DFS", "EFS", "RFS", "TTP", "TTF")),
    # Counted outcomes that the symptom-scale words below would otherwise swallow: "asthma exacerbation rate".
    (
        EVENT,
        phrases(
            r"exacerbations?",
            r"hospitali[sz]ations?",
            r"relapses?",
            r"recurrences?",
            r"fractures?",
            r"strokes?",
            r"myocardial infarction",
            r"major adverse cardiovascular events?",
            r"seizures?",
            r"bleeding",
            r"infections?",
            r"pregnanc\w+",
            r"live birth",
        ),
    ),
    (EVENT, acronyms("MACE", "SSIs?")),
    # "Radiographic global impression of change", "modified total Sharp score", "bone mineral density by DXA".
    (
        IMAGING,
        phrases(
            r"radiograph\w*",
            r"x-ray",
            r"magnetic resonance",
            r"computed tomography",
            r"ultrasound",
            r"ultrasonograph\w*",
            r"echocardiograph\w*",
            r"ejection fraction",
            r"angiograph\w*",
            r"optical coherence",
            r"(?:central )?(?:retinal|subfield|macular) thickness",
            r"bone mineral (?:density|content)",
            r"(?:total )?sharp score",
            r"(?:tumou?r|lesion|infarct) (?:size|volume)",
            r"body composition",
            r"fat mass",
            r"lean (?:body )?mass",
        ),
    ),
    (IMAGING, acronyms("MRI", "CT", "PET", "OCT", "DXA", "DEXA", "LVEF", "mTSS")),
    # "PANSS total score", "ACR20 response" written out, "pain intensity", "quality of life".
    (
        SYMPTOM,
        phrases(
            r"scores?",
            r"scales?",
            r"questionnaires?",
            r"inventory",
            r"(?:severity|activity|disability|impact|symptom) index",
            r"rating",
            r"diary",
            r"quality of life",
            r"symptoms?",
            r"pain",
            r"itch\w*",
            r"pruritus",
            r"fatigue",
            r"depress\w+",
            r"anxiety",
            r"cognit\w+",
            r"dyspn(?:o)?ea",
            r"cough",
            r"nausea",
            r"vomiting",
            r"emesis",
            r"hot flush(?:es)?",
            r"relief",
            r"fever",
            r"global impression",
            r"activities of daily living",
            r"functional (?:status|independence|ability|capacity|outcome)",
            r"physical function\w*",
            r"disability",
            r"American College of Rheumatology(?: criteria)?(?: \w+){0,3} (?:20|50|70)",
            r"(?:6|six)[- ]?minute walk\w*",
            r"disease activity",
        ),
    ),
    (
        SYMPTOM,
        acronyms(
            r"PANSS\w*",
            r"HAM-?D\w*",
            r"MADRS",
            r"ADAS\S*",
            r"ACR ?(?:20|50|70)",
            r"PASI\S*",
            r"IGA",
            r"VAS",
            r"NRS",
            r"EASI\S*",
            r"CGI\S*",
            r"NIHSS",
            r"mRS",
            r"ADLs?",
            r"SF-?36",
            r"WOMAC",
            r"BDI\S*",
            r"UPDRS\S*",
            r"DAS28\S*",
            r"6MWT",
            r"QoL",
            r"PROs?",
            r"SBM",
            r"SBMs",
        ),
    ),
    # "HbA1c", "LDL-C", "eGFR", "FEV1", "serum creatinine", "HIV-1 RNA", "antibody titre", "polysomnogram".
    (
        BIOMARKER,
        phrases(
            r"hba1c",
            r"a1c",
            r"glycated\w*",
            r"glycosylated\w*",
            r"glucose",
            r"c-peptide",
            r"insulin",
            r"ldl(?:-c)?",
            r"hdl(?:-c)?",
            r"cholesterol",
            r"triglycerides?",
            r"lipids?",
            r"blood pressure",
            r"heart rate",
            r"serum",
            r"plasma",
            r"urin\w+",
            r"blood",
            r"creatinine",
            r"glomerular filtration",
            r"proteinuria",
            r"albumin\w*",
            r"hemoglobin",
            r"haemoglobin",
            r"platelets?",
            r"neutrophils?",
            r"viral load",
            r"virolog\w+",
            r"HIV(?:-1)? RNA",
            r"HBV DNA",
            r"HCV RNA",
            r"antibod\w+",
            r"titer",
            r"titre",
            r"seroconversion",
            r"seroprotection",
            r"seroresponse",
            r"immunogenicity",
            r"geometric mean (?:titer|titre|concentration)s?",
            r"biops\w+",
            r"histolog\w+",
            r"polysomnogra\w+",
            r"lung function",
            r"forced expiratory volume",
            r"forced vital capacity",
            r"peak (?:expiratory )?flow",
            r"intraocular pressure",
            r"body weight",
            r"weight loss",
            r"body mass index",
            r"bone turnover",
            r"testosterone",
            r"estradiol",
            r"ferritin",
            r"uric acid",
            r"urate",
            r"calcium",
            r"phosph\w+",
            r"potassium",
            r"sodium",
            r"biomarkers?",
            r"anti-?F?Xa",
            r"culture conversion",
            r"sputum",
            r"parasit\w+",
            r"hormone\w*",
            r"concentrations?",
            r"(?<!-)levels?",  # but not "2-level responders"
            r"minimal residual disease",
            r"cytogenetic response",
            r"molecular response",
            r"electrocardiogra\w+",
            r"QTc?",
        ),
    ),
    (
        BIOMARKER,
        acronyms(
            "FEV1",
            "FVC",
            "eGFR",
            "GFR",
            "iGFR",
            "PSA",
            "CRP",
            "INR",
            "ALT",
            "AST",
            "BMI",
            "SVR\\w*",
            "IOP",
            "CD4",
            "ECG",
            "SBP",
            "DBP",
            "IgE",
            "IgG",
        ),
    ),
    # "Incidence of influenza", "wound closure", "smoking abstinence", "treatment failure", "cured".
    (
        EVENT,
        phrases(
            r"incidence",
            r"occurrence",
            r"cure[ds]?",
            r"heal\w+",
            r"closure",
            r"abstinence",
            r"eradication",
            r"failure",
        ),
    ),
]

# Words that say an outcome is a yes/no result but not of what. "Percentage of responders"
# could be a response on a pain scale, a tumour response or a lab target, so the rules above
# get a go at the description first, and these only decide if the description says nothing.
# Explicit named responses and remissions are clinical events, and are read from the title
# before the description gets a say: "Objective Response Rate" is an event even if the
# description mentions a scan or a blood test.
EXPLICIT_RESPONSE_RULES: list[tuple[str, re.Pattern]] = [
    (
        EVENT,
        phrases(
            r"(?:objective|overall|complete|partial|tumou?r|clinical|h(?:a)?ematologic(?:al)?|"
            r"pathologic(?:al)?|disease) (?:response|remission)s?",
            r"(?:response|remission) rates?",
        ),
    ),
    (EVENT, acronyms("ORR", "CRR", "pCR")),
]

VAGUE_RULES: list[tuple[str, re.Pattern]] = [
    (EVENT, phrases(r"respon\w+", r"remission", r"success")),
]


def classify_text(text: str, rules: list[tuple[str, re.Pattern]] = RULES) -> str | None:
    """The first rule's type that matches `text`, or None when no rule matches."""
    for kind, pattern in rules:
        if pattern.search(text):
            return kind
    return None


def classify_outcome(outcome: dict) -> str:
    """The type of one registered primary outcome.

    Order of trust: the title, then explicit responses and remissions in the title
    ("Objective Response Rate"), then the description, then the vague words
    ("response") in the title, then in the description. Nothing matching gives
    "other / unclear".
    """
    title = outcome.get("measure") or ""
    description = (outcome.get("description") or "")[:DESCRIPTION_CHARS]
    return (
        classify_text(title)
        or classify_text(title, EXPLICIT_RESPONSE_RULES)
        or classify_text(description)
        or classify_text(title, VAGUE_RULES)
        or classify_text(description, VAGUE_RULES)
        or OTHER
    )


# ---------------------------------------------------------------- time frame

WEEKS_PER_UNIT = {
    "hour": 1 / 168,
    "day": 1 / 7,
    "week": 1.0,
    "month": 365.25 / 12 / 7,
    "year": 365.25 / 7,
}
_UNIT = r"(?:hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)"
_NUMBER = r"\d+(?:\.\d+)?"
# One number, with a unit word before it ("Week 4") or after it ("4 weeks") or neither.
_ITEM = rf"(?:({_UNIT})\b\s*)?({_NUMBER})(?:\s*-?\s*({_UNIT})\b)?"
# "1, 2 and 4", "0-24", "12 to 24", "Day 1 to Day 28": a short list or range of items.
_SEPARATOR = r"\s*(?:,|-|–|to|and|or|&)\s*"
_RUN = re.compile(rf"{_ITEM}(?:{_SEPARATOR}{_ITEM})*", re.IGNORECASE)
_ITEM_RE = re.compile(_ITEM, re.IGNORECASE)


def _unit_name(word: str) -> str:
    """ "Weeks" and "wks" are both "week"."""
    word = word.lower()
    for name in ("hour", "hr", "day", "week", "wk", "month", "mo", "year", "yr"):
        if word.startswith(name):
            return {"hr": "hour", "wk": "week", "mo": "month", "yr": "year"}.get(name, name)
    raise ValueError(f"not a time unit: {word}")


def _times_in_run(run: str) -> list[float]:
    """Weeks for each number in one list. A number takes its own unit word, else the
    unit named before it in the list, else the next one named after it.
    """
    items = [(m[2], m[1] or m[3]) for m in _ITEM_RE.finditer(run)]
    weeks, previous = [], None
    for index, (number, unit) in enumerate(items):
        unit = unit or previous
        if unit is None:
            unit = next((u for _, u in items[index:] if u), None)
        previous = unit or previous
        if unit:
            weeks.append(float(number) * WEEKS_PER_UNIT[_unit_name(unit)])
    return weeks


def time_frame_weeks(text: str | None) -> float | None:
    """The longest time in a time-frame text, in weeks. None when it holds no time.

    Reads "12 weeks", "up to 5 years", "Week 0, 4 and 12", "Day 1 to Day 28",
    "0-24 hours" and "baseline to month 6". A number followed by its own unit belongs
    to that unit: "Year 1 to 18 months" is 1 year and 18 months, and "Week 4, 24 hours"
    is 4 weeks and 24 hours. Times with no unit ("Cycle 3", "through study
    completion") are not read.
    """
    if not text:
        return None
    times = [w for run in _RUN.finditer(text) for w in _times_in_run(run.group())]
    return max(times) if times else None


# ---------------------------------------------------------------- one trial


def _first_submit(study: dict) -> date | None:
    return parse_date(study["protocolSection"]["statusModule"].get("studyFirstSubmitDate"))


def _start(study: dict) -> date | None:
    start = study["protocolSection"]["statusModule"].get("startDateStruct") or {}
    return parse_date(start.get("date"))


def make_row(study: dict) -> dict:
    """The features of one study. Reads the protocol section only."""
    protocol = study["protocolSection"]
    outcomes = (protocol.get("outcomesModule") or {}).get("primaryOutcomes") or []
    first = outcomes[0] if outcomes else None
    submitted, started = _first_submit(study), _start(study)
    return {
        "nct_id": protocol["identificationModule"]["nctId"],
        "endpoint_type": classify_outcome(first) if first else None,
        "time_frame_weeks": time_frame_weeks(first.get("timeFrame")) if first else None,
        "n_primary_outcomes": len(outcomes),
        "registered_before_start": (submitted < started if submitted and started else None),
    }


SCHEMA = {
    "nct_id": pl.String,
    "split": pl.String,
    "endpoint_type": pl.String,
    "time_frame_weeks": pl.Float64,
    "n_primary_outcomes": pl.Int64,
    "registered_before_start": pl.Boolean,
}


def build(studies: Iterable[dict], splits: dict[str, str]) -> pl.DataFrame:
    """One row per trial listed in `splits` (nct_id -> split), in the order of `splits`."""
    rows = {}
    for study in studies:
        row = make_row(study)
        if row["nct_id"] in splits:
            rows[row["nct_id"]] = {**row, "split": splits[row["nct_id"]]}
    missing = splits.keys() - rows.keys()
    if missing:
        raise ValueError(f"{len(missing)} trials in trials.csv are missing from the download.")
    return pl.DataFrame([rows[n] for n in splits], schema=SCHEMA)


# ---------------------------------------------------------------- summary


def coverage_text(table: pl.DataFrame) -> str:
    """How many trials in each split have each feature. Counts only; no outcomes."""
    lines = [
        "Trials with the feature present",
        f"  {'':<26}" + "".join(f"{s:>8}" for s in ("dev", "test", "late")),
    ]
    for column in FEATURES:
        counts = [
            table.filter(pl.col("split") == s)[column].is_not_null().sum()
            for s in ("dev", "test", "late")
        ]
        lines.append(f"  {column:<26}" + "".join(f"{c:>8}" for c in counts))
    totals = [table.filter(pl.col("split") == s).height for s in ("dev", "test", "late")]
    lines.append(f"  {'all trials':<26}" + "".join(f"{t:>8}" for t in totals))
    return "\n".join(lines)


def dev_rates_text(table: pl.DataFrame) -> str:
    """Success rate by endpoint type, on development trials only (95% Wilson intervals)."""
    dev = dev_summary.load_dev().join(
        table.select("nct_id", "endpoint_type"), on="nct_id", how="left"
    )
    return dev_summary.by_group(
        dev,
        "Development set: success by first-primary endpoint type",
        dev["endpoint_type"].to_list(),
        order=TYPES,
    )


def main() -> None:
    splits = dict(pl.read_csv(TRIALS_PATH).select("nct_id", "split").iter_rows())
    table = build(iter_studies(), splits)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(OUT_PATH)
    print(f"Wrote {OUT_PATH} ({table.height} trials)\n")
    print(coverage_text(table))
    print()
    print(dev_rates_text(table))


if __name__ == "__main__":
    main()
