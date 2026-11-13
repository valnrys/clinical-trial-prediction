"""Two kinds of failure, measured on trials that finished before 2021.

    uv run python -m elman_exercise.rare_disease       # once: Orphanet + Open Targets diseases
    uv run python -m elman_exercise.failures_summary   # writes results/dev_failures.md

A trial can fail by not delivering a result (it stopped, often because it could not
recruit), or by delivering a result that is negative. Our benchmark label needs a posted
treatment-versus-control comparison, so it only sees the second kind. This module
measures the first kind on the wider population of finished drug trials and asks whether
rare-disease trials stop for recruitment more often, then looks at the second kind in
our labelled development set.

The population is every trial in the raw download with a primary completion date
before 2021-01-01, a drug, biological or genetic intervention, and status completed,
terminated or withdrawn. The date is checked before the status is read, so a trial
from 2021 onwards is never looked at. `report` also refuses any table that holds one.
That is the held-out rule: nothing from 2021 onwards is read, counted or summarised.

Stop reasons come from `stopping.py` (fixed keyword rules over free text, with a
measured error rate). The rare flag comes from `rare_disease.py`. Intervals are 95%.

Two kinds of interval are used. Crude comparisons get the usual Wilson and log intervals,
which treat trials as independent (they are not, limit L10), so they are somewhat too
narrow. Adjusted comparisons are Mantel-Haenszel estimates (`adjusted.py`) with an
interval from resampling lead sponsors, which does not treat a sponsor's trials as
independent.
"""

from __future__ import annotations

import csv
import itertools
import math
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import polars as pl

from elman_exercise import dataset, download
from elman_exercise.adjusted import adjusted, adjusted_text
from elman_exercise.dataset import ROOT, TEST_START
from elman_exercise.dev_summary import Z_95, check_is_development, load_dev, wilson_interval
from elman_exercise.genetics_summary import md_table, newcombe_difference, pct, points
from elman_exercise.rare_disease import (
    BAND_NAMES,
    NOT_RARE,
    RARE,
    UNCLEAR,
    UNMAPPED,
    rare_flags,
)
from elman_exercise.stopping import NO_REASON, REASONS, RECRUITMENT, classify_stop_reason

RESULTS_PATH = ROOT / "results" / "dev_failures.md"
STOP_READ_PATH = ROOT / "results" / "stop_reason_read.csv"
STATUSES = ("COMPLETED", "TERMINATED", "WITHDRAWN")
LABEL = "any_arm"
PHASES = ("phase2", "phase2/3", "phase3")
CLUSTER = "sponsor_name"
ADJUST_FOR = ["phase", "sponsor", "era", "oncology"]
ERAS = (("before 2010", 2010), ("2010 to 2014", 2015), ("2015 to 2020", 2021))
SUCCESS_STRATA = ["phase", "industry", "therapeutic_area"]

POPULATION_SCHEMA = {
    "nct_id": pl.String,
    "primary_completion_date": pl.Date,
    "status": pl.String,
    "why_stopped": pl.String,
    "phase": pl.String,
    "industry": pl.Boolean,
    "enrollment": pl.Int64,
    "has_results": pl.Boolean,
    "therapeutic_area": pl.String,
    "sponsor_name": pl.String,
    "first_condition": pl.String,
}


# ---------------------------------------------------------------- the population


def phase_of(study: dict) -> str:
    """phase2, phase3, phase2/3 (a combined trial), or unknown (neither phase registered)."""
    phases = dataset._get(study, "protocolSection", "designModule", "phases") or []
    has_two, has_three = "PHASE2" in phases, "PHASE3" in phases
    if has_two and has_three:
        return "phase2/3"
    if has_two:
        return "phase2"
    return "phase3" if has_three else "unknown"


def population_row(
    study: dict, first_day: date | None = None, end_day: date = TEST_START
) -> dict | None:
    """One finished drug trial as a row, or None if it is not in the population.

    By default the window is every trial finishing before 2021. `first_day` and `end_day`
    set another window (first day included, end day excluded). The date is checked first,
    so the status of a trial outside the window (or one with no date) is never read.
    """
    completed = dataset.primary_completion(study)
    if completed is None or completed >= end_day or (first_day and completed < first_day):
        return None
    status = dataset._get(study, "protocolSection", "statusModule", "overallStatus")
    if status not in STATUSES or not dataset.has_drug_like_intervention(study):
        return None
    protocol = study["protocolSection"]
    sponsor = dataset._get(protocol, "sponsorCollaboratorsModule", "leadSponsor", "class")
    conditions = dataset._get(protocol, "conditionsModule", "conditions") or []
    return {
        "nct_id": download.nct_id(study),
        "primary_completion_date": completed,
        "status": status,
        "why_stopped": dataset._get(protocol, "statusModule", "whyStopped"),
        "phase": phase_of(study),
        "industry": sponsor == "INDUSTRY",
        "enrollment": dataset._get(protocol, "designModule", "enrollmentInfo", "count"),
        "has_results": bool(study.get("hasResults")),
        "therapeutic_area": dataset.therapeutic_area(study),
        "sponsor_name": dataset._get(protocol, "sponsorCollaboratorsModule", "leadSponsor", "name"),
        "first_condition": conditions[0].strip().lower() if conditions else None,
    }


def build_population(
    studies: Iterable[dict], first_day: date | None = None, end_day: date = TEST_START
) -> pl.DataFrame:
    """Every finished drug trial among `studies` in the window (default: before 2021), one row each."""
    rows = (population_row(study, first_day, end_day) for study in studies)
    return pl.DataFrame([r for r in rows if r], schema=POPULATION_SCHEMA).sort("nct_id")


def check_pre_2021(population: pl.DataFrame) -> None:
    """Refuse a table that holds a trial finishing on or after 2021-01-01, or an undated one."""
    if population.height == 0:
        raise ValueError("The population has no trials.")
    dates = population["primary_completion_date"]
    if dates.null_count() > 0:
        raise ValueError("Refusing to summarise: some trials have no primary completion date.")
    if dates.max() >= TEST_START:
        raise ValueError("Refusing to summarise: the data holds trials from 2021 or later.")


def era_of(year: int) -> str:
    return next(name for name, end in ERAS if year < end)


def add_groups(
    population: pl.DataFrame, flags: pl.DataFrame, later_era: str | None = None
) -> pl.DataFrame:
    """Add the stop reason, the early-stop flags, era, oncology and the rare-disease group.

    The eras end in 2020, so by default a table with a trial from 2021 on is refused. Give
    `later_era` to accept such trials: they all get that one era name.
    """
    if later_era is None:
        check_pre_2021(population)

    def era(year: int) -> str:
        return later_era if later_era and year >= TEST_START.year else era_of(year)

    return (
        population.join(flags, on="nct_id", how="left")
        .with_columns(
            reason=pl.struct("status", "why_stopped").map_elements(
                lambda r: (
                    None if r["status"] == "COMPLETED" else classify_stop_reason(r["why_stopped"])
                ),
                return_dtype=pl.String,
            ),
            era=pl.col("primary_completion_date")
            .dt.year()
            .map_elements(era, return_dtype=pl.String),
            oncology=(pl.col("therapeutic_area") == "oncology").fill_null(False),
        )
        .with_columns(
            stopped=pl.col("status") != "COMPLETED",
            withdrawn=pl.col("status") == "WITHDRAWN",
            recruitment=(pl.col("reason") == RECRUITMENT).fill_null(False),
            sponsor=pl.when(pl.col("industry")).then(pl.lit("industry")).otherwise(pl.lit("other")),
        )
    )


# ---------------------------------------------------------------- statistics and formatting


def ratio_interval(k1: int, n1: int, k0: int, n0: int) -> tuple[float, float, float]:
    """Risk ratio (group 1 over group 0) with the usual 95% log interval.

    Not a number if group 0 has no events (the ratio is undefined) or a group is empty.
    If group 1 has none, the ratio is 0 and the interval is from 0 up to the upper end of
    the interval on counts with 0.5 added to each cell (the Haldane-Anscombe correction).
    """
    if n1 == 0 or n0 == 0 or k0 == 0:
        return math.nan, math.nan, math.nan
    if k1 == 0:
        _, _, high = ratio_interval(1, n1 + 1, k0 + 1, n0 + 1)
        return 0.0, 0.0, high
    ratio = (k1 / n1) / (k0 / n0)
    spread = math.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return ratio, ratio * math.exp(-Z_95 * spread), ratio * math.exp(Z_95 * spread)


def rate_text(k: int, n: int) -> str:
    """ "8.2% (7.7% to 8.8%)", or n/a if there are no trials."""
    if n == 0:
        return "n/a"
    low, high = wilson_interval(k, n)
    return f"{pct(k / n)} ({pct(low)} to {pct(high)})"


def ratio_text(k1: int, n1: int, k0: int, n0: int) -> str:
    ratio, low, high = ratio_interval(k1, n1, k0, n0)
    return "n/a" if math.isnan(ratio) else f"{ratio:.2f} ({low:.2f} to {high:.2f})"


def count(frame: pl.DataFrame, flag: str) -> int:
    return int(frame[flag].sum())


def split_groups(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(rare, not rare) trials."""
    return frame.filter(pl.col("group") == RARE), frame.filter(pl.col("group") == NOT_RARE)


def two_groups(frame: pl.DataFrame) -> pl.DataFrame:
    """Rare and not rare trials only, with the boolean column `is_rare` the estimators use."""
    return frame.filter(pl.col("group").is_in([RARE, NOT_RARE])).with_columns(
        is_rare=pl.col("group") == RARE
    )


def adjusted_ratio(frame: pl.DataFrame, outcome: str, strata: list[str]) -> tuple[float, ...]:
    """Mantel-Haenszel rare-over-not-rare ratio and sponsor-clustered interval."""
    return adjusted(two_groups(frame), "is_rare", outcome, strata, CLUSTER)


def adjusted_ratio_text(frame: pl.DataFrame, outcome: str, strata: list[str]) -> str:
    return adjusted_text(two_groups(frame), "is_rare", outcome, strata, CLUSTER)


# ---------------------------------------------------------------- sections


def funnel_tables(population: pl.DataFrame, benchmark_ids: set[str]) -> str:
    """How many finished trials completed, stopped, posted results, and are in the benchmark."""
    total = population.height
    in_benchmark = population.filter(pl.col("nct_id").is_in(list(benchmark_ids)))
    first_rows = [
        ["Finished drug trials (completed, terminated or withdrawn)", f"{total:,}", "100%", ""]
    ]
    for status in STATUSES:
        n = population.filter(pl.col("status") == status).height
        first_rows.append([status.capitalize(), f"{n:,}", *_share(n, total)])
    results = count(population, "has_results")
    first_rows.append(
        ["Posted results on ClinicalTrials.gov", f"{results:,}", *_share(results, total)]
    )
    first_rows.append(
        [
            "In our labelled benchmark",
            f"{in_benchmark.height:,}",
            *_share(in_benchmark.height, total),
        ]
    )
    by_status = []
    for status in STATUSES:
        part = population.filter(pl.col("status") == status)
        by_status.append(
            [
                status.capitalize(),
                f"{part.height:,}",
                rate_text(count(part, "has_results"), part.height),
                rate_text(
                    part.filter(pl.col("nct_id").is_in(list(benchmark_ids))).height, part.height
                ),
            ]
        )
    missing = len(benchmark_ids - set(population["nct_id"].to_list()))
    note = (
        f"All {len(benchmark_ids):,} labelled trials are in this population."
        if missing == 0
        else f"{missing:,} labelled trials are not in this population."
    )
    return "\n\n".join(
        [
            md_table(["", "trials", "share of finished", "95% interval"], first_rows),
            (
                "Of the finished drug trials that ClinicalTrials.gov lists, how many ended early, how many "
                "posted results, and how many our benchmark can speak about."
            ),
            md_table(["Status", "trials", "posted results", "in our benchmark"], by_status),
            "The same two shares within each status. " + note,
        ]
    )


def _share(k: int, n: int) -> list[str]:
    low, high = wilson_interval(k, n)
    return [pct(k / n), f"{pct(low)} to {pct(high)}"]


def reason_table(population: pl.DataFrame, hand_read: dict | None) -> str:
    """Stop-reason distribution among terminated and withdrawn trials."""
    stopped = population.filter(pl.col("stopped"))
    terminated = stopped.filter(pl.col("status") == "TERMINATED")
    withdrawn = stopped.filter(pl.col("status") == "WITHDRAWN")
    rows = []
    for reason in REASONS:
        n_all = stopped.filter(pl.col("reason") == reason).height
        n_term = terminated.filter(pl.col("reason") == reason).height
        n_with = withdrawn.filter(pl.col("reason") == reason).height
        rows.append(
            [
                reason,
                f"{n_term:,}",
                f"{n_with:,}",
                f"{n_all:,}",
                rate_text(n_all, stopped.height),
            ]
        )
    rows.append(
        ["all", f"{terminated.height:,}", f"{withdrawn.height:,}", f"{stopped.height:,}", "100%"]
    )
    error = (
        f"On a fresh hand read of {hand_read['n']} stopped trials the rules were wrong "
        f"{hand_read['wrong']} times ({pct(hand_read['wrong'] / hand_read['n'])}; see section 8)."
        if hand_read
        else "The error rate of these rules has not been measured."
    )
    return "\n\n".join(
        [
            md_table(
                ["Reason", "terminated", "withdrawn", "both", "share of both (95% interval)"], rows
            ),
            f"Why trials stopped, by a keyword reading of the sponsor's free text. {error}",
        ]
    )


def group_size_table(population: pl.DataFrame) -> str:
    rows = []
    for group in (RARE, NOT_RARE, UNCLEAR, UNMAPPED):
        n = population.filter(pl.col("group") == group).height
        rows.append([group, f"{n:,}", pct(n / population.height)])
    unknown = population.filter(pl.col("group").is_in([UNCLEAR, UNMAPPED])).height
    return "\n\n".join(
        [
            md_table(["Group", "trials", "share"], rows),
            (
                f"How many trials can be called rare or not. {pct(unknown / population.height)} cannot: "
                "unclear means every disease has an Orphanet entry but no usable prevalence class; "
                "unmapped means Open Targets lists no disease for the trial."
            ),
        ]
    )


def crude_ratios(population: pl.DataFrame) -> dict[str, tuple[float, float, float]]:
    rare, other = split_groups(population)
    return {
        flag: ratio_interval(count(rare, flag), rare.height, count(other, flag), other.height)
        for flag in ("stopped", "withdrawn", "recruitment")
    }


def contrast_table(population: pl.DataFrame) -> str:
    """Rare versus not rare, crude: stopped early, withdrawn, stopped for recruitment, ratios."""
    rare, other = split_groups(population)
    rows = []
    for name, part in ((RARE, rare), (NOT_RARE, other)):
        stopped = part.filter(pl.col("stopped"))
        rows.append(
            [
                name,
                f"{part.height:,}",
                rate_text(count(part, "stopped"), part.height),
                rate_text(count(part, "withdrawn"), part.height),
                rate_text(count(part, "recruitment"), part.height),
                rate_text(count(stopped, "recruitment"), stopped.height),
            ]
        )
    rows.append(
        [
            "ratio, rare over not rare",
            "",
            ratio_text(count(rare, "stopped"), rare.height, count(other, "stopped"), other.height),
            ratio_text(
                count(rare, "withdrawn"), rare.height, count(other, "withdrawn"), other.height
            ),
            ratio_text(
                count(rare, "recruitment"), rare.height, count(other, "recruitment"), other.height
            ),
            "",
        ]
    )
    header = [
        "",
        "trials",
        "terminated or withdrawn",
        "withdrawn",
        "stopped for recruitment",
        "recruitment, share of those stopped",
    ]
    crude = adjusted_ratio_text(population, "recruitment", [])
    return "\n\n".join(
        [
            md_table(header, rows),
            (
                "The crude numbers: rare-flagged trials are stopped, and stopped for recruitment, more "
                "often than the rest. The ratio rows compare rates among all finished trials in each group; "
                "intervals treat trials as independent. With whole sponsors resampled the crude "
                f"recruitment ratio is {crude}. These are not adjusted for anything. The next section "
                "asks how much of this is something else about rare-flagged trials."
            ),
        ]
    )


def oncology_split_table(population: pl.DataFrame) -> str:
    """Section 4.1: rare-flagged trials are mostly cancer trials, and cancer trials stop more."""
    rare, other = split_groups(population)
    share_rows = [
        [
            name,
            f"{part.height:,}",
            pct(count(part, "oncology") / part.height),
            pct(count(part, "industry") / part.height),
        ]
        for name, part in ((RARE, rare), (NOT_RARE, other))
    ]
    split_rows = []
    for label, part in (
        ("oncology", population.filter(pl.col("oncology"))),
        ("not oncology", population.filter(~pl.col("oncology"))),
    ):
        r, o = split_groups(part)
        split_rows.append(
            [
                label,
                f"{r.height:,}",
                rate_text(count(r, "recruitment"), r.height),
                f"{o.height:,}",
                rate_text(count(o, "recruitment"), o.height),
                ratio_text(count(r, "recruitment"), r.height, count(o, "recruitment"), o.height),
            ]
        )
    return "\n\n".join(
        [
            md_table(["", "trials", "oncology", "industry sponsor"], share_rows),
            "Share of each group that is oncology (therapeutic area from the trial's MeSH terms) or industry-sponsored.",
            md_table(
                [
                    "",
                    "rare trials",
                    "rare: stopped for recruitment",
                    "not rare trials",
                    "not rare: stopped for recruitment",
                    "crude ratio",
                ],
                split_rows,
            ),
            (
                "The same crude comparison inside oncology and outside it. Cancer trials stop for "
                "recruitment at a different rate from other trials, and most rare-flagged trials are cancer "
                "trials, so part of the crude ratio in section 3 is cancer, not rarity."
            ),
        ]
    )


def adjusted_overall_table(population: pl.DataFrame) -> str:
    """Section 4.2: the ratio with phase, sponsor, era and oncology allowed for."""
    crude = crude_ratios(population)
    rows = []
    for label, flag in (
        ("stopped for recruitment", "recruitment"),
        ("terminated or withdrawn", "stopped"),
    ):
        c, cl, ch = crude[flag]
        rows.append(
            [
                label,
                f"{c:.2f} ({cl:.2f} to {ch:.2f})",
                adjusted_ratio_text(population, flag, []),
                adjusted_ratio_text(population, flag, ADJUST_FOR),
            ]
        )
    est, low, high = adjusted_ratio(population, "recruitment", ADJUST_FOR)
    if low <= 1 <= high:
        reading = (
            "its interval includes 1, so these data show no overall gap in stopping for recruitment "
            "once cancer and the other factors are allowed for."
        )
    else:
        reading = (
            f"its interval ({low:.2f} to {high:.2f}) excludes 1, so some overall gap remains after "
            "adjustment."
        )
    return "\n\n".join(
        [
            md_table(
                [
                    "Rare over not rare",
                    "crude (trials independent)",
                    "crude (sponsors resampled)",
                    "adjusted for phase, sponsor type, era and oncology (sponsors resampled)",
                ],
                rows,
            ),
            (
                "Mantel-Haenszel ratios over strata of phase (2, 2/3, 3), sponsor type (industry or not), "
                "era of primary completion (before 2010, 2010 to 2014, 2015 to 2020) and oncology. "
                "Intervals resample lead sponsors (2,000 draws). The adjusted recruitment ratio is "
                f"{est:.2f}; {reading} Anything not in the strata is not adjusted for."
            ),
        ]
    )


def outside_oncology_table(population: pl.DataFrame) -> str:
    """Section 4.3: the ratio inside and outside oncology, adjusted for phase, sponsor and era."""
    strata = ["phase", "sponsor", "era"]
    rows = []
    for label, part in (
        ("outside oncology", population.filter(~pl.col("oncology"))),
        ("inside oncology", population.filter(pl.col("oncology"))),
    ):
        r, o = split_groups(part)
        rows.append(
            [
                label,
                f"{r.height:,}",
                f"{o.height:,}",
                ratio_text(count(r, "recruitment"), r.height, count(o, "recruitment"), o.height),
                adjusted_ratio_text(part, "recruitment", strata),
            ]
        )
    est, low, high = adjusted_ratio(population.filter(~pl.col("oncology")), "recruitment", strata)
    verdict = (
        "so there is an excess of recruitment stops outside oncology"
        if low > 1
        else "so the data are consistent with an excess of about that size and with none"
    )
    return "\n\n".join(
        [
            md_table(
                [
                    "Ratio of stopped for recruitment, rare over not rare",
                    "rare trials",
                    "not rare trials",
                    "crude (trials independent)",
                    "adjusted for phase, sponsor type and era (sponsors resampled)",
                ],
                rows,
            ),
            (
                f"Outside oncology the adjusted ratio is {est:.2f} ({low:.2f} to {high:.2f}), {verdict}. "
                "It is the one figure in this file that could be a rare-disease effect, and it is the least "
                "firm: it moves with how the rare flag is drawn (section 4.5) and rests on a flag that is "
                "imperfect (section 8). Inside oncology the adjusted ratio is below or near 1."
            ),
        ]
    )


def subgroup_row(label: str, part: pl.DataFrame, strata: list[str]) -> list[str]:
    rare, other = split_groups(part)
    return [
        label,
        f"{rare.height:,}",
        rate_text(count(rare, "recruitment"), rare.height),
        f"{other.height:,}",
        rate_text(count(other, "recruitment"), other.height),
        ratio_text(
            count(rare, "recruitment"), rare.height, count(other, "recruitment"), other.height
        ),
        adjusted_ratio_text(part, "recruitment", strata),
    ]


SUBGROUP_HEADER = [
    "",
    "rare trials",
    "rare: stopped for recruitment",
    "not rare trials",
    "not rare: stopped for recruitment",
    "crude ratio",
    "adjusted ratio (sponsors resampled)",
]


def phase_sponsor_table(population: pl.DataFrame) -> str:
    """Section 4.4: crude and adjusted recruitment ratio by phase and by sponsor type."""
    rows = []
    for phase in PHASES:
        rows.append(
            subgroup_row(
                phase,
                population.filter(pl.col("phase") == phase),
                ["sponsor", "era", "oncology"],
            )
        )
    for sponsor in ("industry", "other"):
        rows.append(
            subgroup_row(
                f"{sponsor} sponsor",
                population.filter(pl.col("sponsor") == sponsor),
                ["phase", "era", "oncology"],
            )
        )
    return "\n\n".join(
        [
            md_table(SUBGROUP_HEADER, rows),
            (
                "Crude and adjusted ratios by phase and by sponsor type. Phase rows are adjusted for sponsor "
                "type, era and oncology; sponsor rows for phase, era and oncology. Compare each crude "
                "ratio with its adjusted one."
            ),
        ]
    )


def regrouped(
    population: pl.DataFrame, variant: str, alternative: pl.DataFrame | None
) -> pl.DataFrame | None:
    """The same trials with the rare / not-rare split redrawn (None if it cannot be)."""
    group = pl.col("group")
    if variant == "unclear counted as rare":
        return population.with_columns(
            pl.when(group == UNCLEAR).then(pl.lit(RARE)).otherwise(group).alias("group")
        )
    if variant == "unclear counted as not rare":
        return population.with_columns(
            pl.when(group == UNCLEAR).then(pl.lit(NOT_RARE)).otherwise(group).alias("group")
        )
    mixed = (group == RARE) & ~pl.col("all_rare")
    if variant == "rare only if every disease is rare; the rest not rare":
        return population.with_columns(
            pl.when(mixed).then(pl.lit(NOT_RARE)).otherwise(group).alias("group")
        )
    if variant == "rare only if every disease is rare; the rest left out":
        return population.with_columns(
            pl.when(mixed).then(pl.lit("mixed")).otherwise(group).alias("group")
        )
    if variant == "prevalence measure ranked below geography (earlier rule)":
        if alternative is None:
            return None
        flags = alternative.select("nct_id", "group", "all_rare", "band", "band_commonest")
        return population.drop("group", "all_rare", "band", "band_commonest").join(
            flags, on="nct_id", how="left"
        )
    return population


ROBUSTNESS_VARIANTS = (
    "as above",
    "unclear counted as rare",
    "unclear counted as not rare",
    "rare only if every disease is rare; the rest not rare",
    "rare only if every disease is rare; the rest left out",
    "prevalence measure ranked below geography (earlier rule)",
)


def robustness_table(population: pl.DataFrame, alternative: pl.DataFrame | None) -> str:
    rows = []
    for variant in ROBUSTNESS_VARIANTS:
        redrawn = regrouped(population, variant, alternative)
        if redrawn is None:
            continue
        rare, other = split_groups(redrawn)
        rows.append(
            [
                variant,
                f"{rare.height:,}",
                f"{other.height:,}",
                ratio_text(
                    count(rare, "recruitment"),
                    rare.height,
                    count(other, "recruitment"),
                    other.height,
                ),
                adjusted_ratio_text(redrawn, "recruitment", ADJUST_FOR),
                adjusted_ratio_text(
                    redrawn.filter(~pl.col("oncology")), "recruitment", ["phase", "sponsor", "era"]
                ),
            ]
        )
    outside = [float(row[5].split(" ")[0]) for row in rows if row[5] != "n/a"]
    return "\n\n".join(
        [
            md_table(
                [
                    "How trials are grouped",
                    "rare",
                    "not rare",
                    "crude ratio",
                    "adjusted ratio",
                    "adjusted ratio outside oncology",
                ],
                rows,
            ),
            (
                "Does the answer depend on how trials we cannot classify, or trials with several diseases, "
                "are counted, or on how the rare flag is drawn? Ratios are of stopping for recruitment, rare "
                "over not rare. 'Left out' drops the mixed trials (some diseases rare, some not) from both "
                "groups; 'the rest not rare' moves them to not rare, so every classified trial is counted. "
                "The last row redoes the whole classification with the earlier rule for choosing among "
                "Orphanet's prevalence entries (geography before kind of measure). Across these rows the "
                f"adjusted ratio outside oncology runs from {min(outside, default=math.nan):.2f} to "
                f"{max(outside, default=math.nan):.2f}."
            ),
        ]
    )


def band_table(population: pl.DataFrame, column: str, title: str) -> str:
    """Recruitment stops by band for rare-flagged trials, split by oncology, and against not rare."""
    rare = population.filter(pl.col("group") == RARE)
    rows = []
    for band, name in BAND_NAMES.items():
        part = rare.filter(pl.col(column) == band)
        outside = part.filter(~pl.col("oncology"))
        inside = part.filter(pl.col("oncology"))
        versus = population.filter(
            (pl.col("group") == NOT_RARE) | ((pl.col("group") == RARE) & (pl.col(column) == band))
        )
        rows.append(
            [
                name,
                f"{part.height:,}",
                rate_text(count(part, "stopped"), part.height),
                rate_text(count(part, "recruitment"), part.height),
                f"{outside.height:,}: {rate_text(count(outside, 'recruitment'), outside.height)}",
                f"{inside.height:,}: {rate_text(count(inside, 'recruitment'), inside.height)}",
                adjusted_ratio_text(versus, "recruitment", ADJUST_FOR),
            ]
        )
    return md_table(
        [
            title,
            "trials",
            "terminated or withdrawn",
            "stopped for recruitment",
            "recruitment, outside oncology (trials: rate)",
            "recruitment, inside oncology (trials: rate)",
            "adjusted ratio against not rare",
        ],
        rows,
    )


def band_estimates(population: pl.DataFrame, column: str) -> list[float]:
    """Adjusted rare-over-not-rare recruitment ratio for each band, rarest first."""
    estimates = []
    for band in BAND_NAMES:
        versus = population.filter(
            (pl.col("group") == NOT_RARE) | ((pl.col("group") == RARE) & (pl.col(column) == band))
        )
        estimates.append(adjusted_ratio(versus, "recruitment", ADJUST_FOR)[0])
    return estimates


def band_section(population: pl.DataFrame, alternative: pl.DataFrame | None = None) -> str:
    estimates = band_estimates(population, "band")
    rising = all(a > b for a, b in itertools.pairwise(estimates))
    reading = (
        "The adjusted ratio rises steadily with rarity."
        if rising
        else "There is no steady rise with rarity: the adjusted ratios from the rarest band to the "
        f"commonest rare band are {', '.join(f'{e:.2f}' for e in estimates)}."
    )
    parts = [
        band_table(population, "band", "By the rarest disease in the trial"),
        (
            "Within rare diseases, do the rarer ones stop for recruitment more often? A trial's band is "
            "the Orphanet class of the RAREST of its rare diseases, the natural reading of 'rarer diseases "
            "fail to recruit more'. The last column compares each band with all not-rare trials, adjusted "
            f"for phase, sponsor type, era and oncology. {reading} The two rarest bands hold few trials "
            "and their intervals are wide; the recruitment columns split by oncology show the same."
        ),
        band_table(population, "band_commonest", "By the commonest rare disease in the trial"),
        (
            "The same table with the earlier definition of a band, the class of the trial's COMMONEST "
            "rare disease (this and the table above both use the current choice of prevalence entry). "
            "Only trials with several rare diseases of different classes move between rows."
        ),
    ]
    if alternative is not None:
        before = regrouped(population, ROBUSTNESS_VARIANTS[-1], alternative)
        parts += [
            band_table(
                before, "band_commonest", "Earlier rule in full: geography first, commonest disease"
            ),
            (
                "The band table as it stood before the two changes: prevalence entries chosen with "
                "geography ahead of the kind of measure, and the band set by the commonest rare disease. "
                "Both changes move trials between bands and in or out of the rare group."
            ),
        ]
    return "\n\n".join(parts)


def enrolment_table(population: pl.DataFrame) -> str:
    rows = []
    for status in ("COMPLETED", "TERMINATED"):
        for group in (RARE, NOT_RARE):
            sizes = population.filter(
                (pl.col("group") == group)
                & (pl.col("status") == status)
                & pl.col("enrollment").is_not_null()
            )["enrollment"]
            if sizes.len() == 0:
                rows.append([status.capitalize(), group, "0", "n/a", "n/a"])
                continue
            rows.append(
                [
                    status.capitalize(),
                    group,
                    f"{sizes.len():,}",
                    f"{sizes.median():.0f}",
                    f"{sizes.quantile(0.25):.0f} to {sizes.quantile(0.75):.0f}",
                ]
            )
    return "\n\n".join(
        [
            md_table(["Status", "group", "trials", "median enrolment", "middle half"], rows),
            (
                "Actual enrolment (the planned figure is not in the record). Withdrawn trials never enrolled "
                "anyone and are left out. Rare-flagged trials are smaller whether or not they finish, which "
                "fits a small pool of patients; this table is not adjusted for oncology or anything else."
            ),
        ]
    )


# ---------------------------------------------------------------- the labelled trials


def with_flags(dev: pl.DataFrame, population: pl.DataFrame) -> pl.DataFrame:
    """Labelled development trials with their rare-disease group, band, sponsor and first condition."""
    flags = population.select("nct_id", "group", "band", CLUSTER, "first_condition")
    return dev.join(flags, on="nct_id", how="left").with_columns(
        pl.col("group").fill_null(UNMAPPED)
    )


def success_text(part: pl.DataFrame) -> str:
    return rate_text(count(part, LABEL), part.height)


def success_group_table(dev: pl.DataFrame) -> str:
    rows = []
    for group in (RARE, NOT_RARE, UNCLEAR, UNMAPPED):
        part = dev.filter(pl.col("group") == group)
        rows.append([group, f"{part.height:,}", success_text(part)])
    unknown = dev.filter(pl.col("group").is_in([UNCLEAR, UNMAPPED])).height
    return "\n\n".join(
        [
            md_table(["Group", "labelled trials", "success rate (95% interval)"], rows),
            (
                "Success (`any_arm`) among the labelled development trials, by rare-disease group. "
                f"{unknown:,} of {dev.height:,} labelled trials ({pct(unknown / dev.height)}) are unclear "
                "or unmapped and are left out of the rare-versus-not-rare comparisons below."
            ),
        ]
    )


def success_contrast_row(label: str, part: pl.DataFrame) -> list[str]:
    rare, other = split_groups(part)
    s1, s0 = count(rare, LABEL), count(other, LABEL)
    diff, low, high = newcombe_difference(s1, rare.height, s0, other.height)
    return [
        label,
        f"{rare.height:,}",
        success_text(rare),
        f"{other.height:,}",
        success_text(other),
        points(diff),
        f"{points(low)} to {points(high)}",
    ]


def success_frame(dev: pl.DataFrame) -> pl.DataFrame:
    """Rare and not-rare labelled trials with the boolean columns the estimators need."""
    return two_groups(dev.filter(pl.col(LABEL).is_not_null())).with_columns(
        success=pl.col(LABEL).cast(pl.Boolean),
        industry=pl.col("sponsor_class") == "INDUSTRY",
    )


def adjusted_success_gap(dev: pl.DataFrame) -> tuple[float, float, float]:
    """Mantel-Haenszel success difference (rare minus not rare), in proportion, with interval."""
    return adjusted(
        success_frame(dev), "is_rare", "success", SUCCESS_STRATA, CLUSTER, measure="difference"
    )


def success_contrast_table(dev: pl.DataFrame) -> str:
    oncology = pl.col("therapeutic_area") == "oncology"
    parts = [
        ("all labelled trials", dev),
        ("phase 2", dev.filter(pl.col("phase") == "phase2")),
        ("phase 3", dev.filter(pl.col("phase") == "phase3")),
        ("oncology", dev.filter(oncology)),
        ("not oncology", dev.filter(~oncology.fill_null(False))),
    ]
    rows = [success_contrast_row(label, part) for label, part in parts]
    est, low, high = adjusted_success_gap(dev)
    crude = rows[0]
    return "\n\n".join(
        [
            md_table(
                [
                    "",
                    "rare trials",
                    "rare: success",
                    "not rare trials",
                    "not rare: success",
                    "difference (points)",
                    "95% interval",
                ],
                rows,
            ),
            (
                f"The second kind of failure. Among trials that did report a comparison, rare-flagged "
                f"trials succeeded less often: {crude[2].split(' (')[0]} against {crude[4].split(' (')[0]}, "
                f"a crude gap of {crude[5]} points. Adjusted for phase, sponsor type and therapeutic area "
                f"(Mantel-Haenszel, sponsors resampled) the gap is {points(est)} points "
                f"({points(low)} to {points(high)}): smaller than the crude figure, and still below zero. "
                "Phase 3 here includes combined phase 2/3 trials. The crude intervals treat trials as "
                "independent."
            ),
        ]
    )


def condition_table(dev: pl.DataFrame) -> str:
    """The largest conditions among labelled rare trials, and the gap with them removed."""
    frame = success_frame(dev)
    rare = frame.filter(pl.col("is_rare"))
    top = (
        rare.filter(pl.col("first_condition").is_not_null())
        .group_by("first_condition")
        .agg(n=pl.len(), k=pl.col("success").sum())
        .sort(["n", "first_condition"], descending=[True, False])
        .head(10)
    )
    rows = [
        [r["first_condition"], f"{r['n']:,}", rate_text(r["k"], r["n"])]
        for r in top.iter_rows(named=True)
    ]
    share = top["n"].sum() / rare.height
    trimmed = frame.filter(
        ~pl.col("first_condition").is_in(top["first_condition"].to_list())
        | pl.col("first_condition").is_null()
    )
    est, low, high = adjusted(
        trimmed, "is_rare", "success", SUCCESS_STRATA, CLUSTER, measure="difference"
    )
    return "\n\n".join(
        [
            md_table(
                ["Largest conditions (first condition listed)", "labelled rare trials", "success"],
                rows,
            ),
            (
                f"The ten largest first-listed conditions hold {pct(share)} of the labelled rare-flagged "
                f"trials. Dropping every trial with one of these ten conditions, the adjusted gap is "
                f"{points(est)} points ({points(low)} to {points(high)}). Conditions are the registry's "
                "free text, so near-duplicates (two spellings of one cancer) are counted apart."
            ),
        ]
    )


def success_band_table(dev: pl.DataFrame) -> str:
    rare = dev.filter(pl.col("group") == RARE)
    rows = []
    for band, name in BAND_NAMES.items():
        part = rare.filter(pl.col("band") == band)
        rows.append([name, f"{part.height:,}", success_text(part)])
    return "\n\n".join(
        [
            md_table(
                [
                    "By the rarest disease in the trial",
                    "labelled trials",
                    "success rate (95% interval)",
                ],
                rows,
            ),
            "Success within rare-flagged trials by prevalence band. The two rarest bands hold too few trials to say anything.",
        ]
    )


def load_stop_reason_read(path: Path = STOP_READ_PATH) -> dict | None:
    """Counts from the hand read of stopped trials (results/stop_reason_read.csv), or None.

    Columns: nct_id, why_stopped, rule, truth, note. `rule` is what the keyword rules say and
    `truth` is what a reader of the reason says. A wrong reading is rule != truth.
    """
    if not Path(path).exists():
        return None
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "n": len(rows),
        "wrong": sum(r["rule"] != r["truth"] for r in rows),
        "recruitment_called": sum(r["rule"] == RECRUITMENT for r in rows),
        "recruitment_false_positive": sum(
            r["rule"] == RECRUITMENT and r["truth"] != RECRUITMENT for r in rows
        ),
        "recruitment_true": sum(r["truth"] == RECRUITMENT for r in rows),
        "recruitment_false_negative": sum(
            r["truth"] == RECRUITMENT and r["rule"] != RECRUITMENT for r in rows
        ),
    }


FIRST_READ = (40, 5)  # the first version of the rules, read on 40 stopped trials, 5 wrong


def stop_rule_paragraph(hand_read: dict | None) -> str:
    n0, w0 = FIRST_READ
    first = (
        f"The first version of the rules was read on {n0} random stopped trials and {w0} were wrong "
        f"({pct(w0 / n0)}); the rules were then rewritten (negated safety or efficacy phrases removed, "
        "recruitment needing a failure word, 'interim analysis' alone not counting, a separate "
        "stopped-for-benefit category). "
    )
    if not hand_read:
        return first + "The rewritten rules have not been read on fresh trials."
    n, wrong = hand_read["n"], hand_read["wrong"]
    low, high = wilson_interval(wrong, n)
    fp, called = hand_read["recruitment_false_positive"], hand_read["recruitment_called"]
    fn, true = hand_read["recruitment_false_negative"], hand_read["recruitment_true"]
    return (
        first
        + f"The rewritten rules were then read on {n} fresh stopped trials that gave a reason (drawn "
        "with seed 31337 from the trials sorted by id; `results/stop_reason_read.csv`; read by a language "
        f"model, not a person) and were wrong on {wrong} ({pct(wrong / n)}, 95% interval {pct(low)} "
        f"to {pct(high)}). For recruitment: of {called} trials the rules called recruitment, {fp} were "
        f"not; of {true} that were recruitment, the rules missed {fn}. The rules were not changed after "
        "that read, so these are their error rates. They are rules over free text: a trial that ran short "
        "of patients may give another reason, or none. 'Other' is a catch-all for reasons we did not try "
        "to sort."
    )


def count_reason(population: pl.DataFrame, reason: str) -> int:
    return population.filter(pl.col("reason") == reason).height


def limits_text(
    population: pl.DataFrame, labelled: pl.DataFrame, hand_read: dict | None = None
) -> str:
    """Section 8: what the numbers do not show. Figures come from the data where they can."""
    unknown = population.filter(pl.col("group").is_in([UNCLEAR, UNMAPPED])).height
    rare, other = split_groups(population)
    stopped = population.filter(pl.col("stopped"))
    no_reason = count_reason(population, NO_REASON)
    unknown_dev = labelled.filter(pl.col("group").is_in([UNCLEAR, UNMAPPED])).height
    return f"""\
**Planned enrolment is not in the record.** The registry shows the actual figure, so "failed to recruit" rests on the sponsor's free-text reason. A trial that ran short of patients may give another reason, or none ({pct(no_reason / stopped.height)} of stopped trials give none).

**The stop-reason rules are crude.** {stop_rule_paragraph(hand_read)}

**The rare flag is imperfect.** A reviewer (a language model, not a person) read 40 rare-flagged trials and found about 4 to 6 clearly wrong (for example head-and-neck cancer, a DTP/polio vaccine trial, Alzheimer's disease, a generic "leukemia") and 7 debatable (for example ovarian cancer and renal cell carcinoma, which have rare subtypes but are common overall). A trial is flagged rare if any of its Open Targets diseases is, and Open Targets lists related or background conditions as well as the one studied. A wrongly flagged trial dilutes any real difference. {pct(count(rare, "oncology") / rare.height)} of rare-flagged trials are oncology, so the flag matters most where most of the rare trials are.

**Rare is tangled up with other things.** Across all finished trials {pct(count(rare, "oncology") / rare.height)} of rare-flagged trials are oncology against {pct(count(other, "oncology") / other.height)} of the rest, and {pct(count(rare, "industry") / rare.height)} have an industry sponsor against {pct(count(other, "industry") / other.height)}. Section 4 allows for phase, sponsor type, era and oncology. Other differences (the kind of drug, trial size, country) are not allowed for, and nothing here says rarity causes any gap that remains.

**{pct(unknown / population.height)} of trials cannot be called rare or not.** They are the unclear and unmapped rows in section 3. In the labelled set {unknown_dev:,} of {labelled.height:,} trials ({pct(unknown_dev / labelled.height)}) are unclear or unmapped and are left out of the rare-versus-not-rare comparison in section 7. Counting unclear trials either way changes the ratios little (section 4.5); unmapped trials are never varied.

**"Not rare" does not mean "common".** A numeric prevalence for common diseases is not available from a public source by a simple join. "Not rare" means no disease of the trial is in an Orphanet rare class, or Orphanet has no entry for it. Where Orphanet gives several figures for one disease, the kind of measure (point prevalence, then birth or lifetime prevalence, then incidence) is preferred over where it applies; the earlier order, geography first, is in the robustness table.

**Open Targets lists diseases, not necessarily the one the trial targets.** A trial's disease list can include related or background conditions.

**The wider population is not the labelled benchmark.** The benchmark covers {pct(labelled.height / population.height)} of finished trials (section 1). This file shows how big the part we cannot see is and where it concentrates, not what those trials would have scored.
"""


def summary_text(population: pl.DataFrame, labelled: pl.DataFrame) -> str:
    """The findings in plain words, built from the numbers below so they cannot drift from them."""
    rare, other = split_groups(population)
    crude = crude_ratios(population)["recruitment"][0]
    est, low, high = adjusted_ratio(population, "recruitment", ADJUST_FOR)
    out_est, out_low, out_high = adjusted_ratio(
        population.filter(~pl.col("oncology")), "recruitment", ["phase", "sponsor", "era"]
    )
    gap, gap_low, gap_high = adjusted_success_gap(labelled)
    out_reading = (
        "an excess the interval excludes 1 for"
        if out_low > 1
        else "which the interval cannot separate from no excess"
    )
    share = "most" if (est - 1) < 0.5 * (crude - 1) else "only part"
    return (
        "**In short.** "
        f"Rare-flagged trials stop for recruitment more often in the crude numbers (ratio {crude:.2f}), "
        f"but {share} of that is because rare-flagged trials are mostly cancer trials "
        f"({pct(count(rare, 'oncology') / rare.height)} against {pct(count(other, 'oncology') / other.height)} "
        f"of the rest). Allowing for phase, sponsor, era and oncology the overall ratio is {est:.2f} "
        f"({low:.2f} to {high:.2f}). Outside oncology the adjusted ratio is {out_est:.2f} "
        f"({out_low:.2f} to {out_high:.2f}), {out_reading}. "
        "For the second kind of failure, a negative result, rare-flagged labelled trials succeed less "
        "often, and the gap survives adjustment for phase, sponsor type and therapeutic area, smaller: "
        f"{points(gap)} points ({points(gap_low)} to {points(gap_high)}). "
        "The rare flag is imperfect (section 8), and none of this shows that rarity causes either gap.\n"
    )


def report(
    population: pl.DataFrame,
    dev: pl.DataFrame,
    alternative: pl.DataFrame | None = None,
    hand_read: dict | None = None,
) -> str:
    """The whole results file. Refuses data holding a trial from 2021 or later.

    `alternative` is the rare-disease table built with the earlier choice among Orphanet
    entries (for the robustness table); `hand_read` is the output of `load_stop_reason_read`.
    """
    check_pre_2021(population)
    check_is_development(dev)
    labelled = with_flags(dev, population)
    sections = [
        "# Two kinds of failure: development set (trials finishing before 2021)\n",
        (
            "A trial can fail by not delivering a result at all (it stopped, often because it could not "
            "recruit), or by delivering a result that is negative. Our benchmark label needs a posted "
            "treatment-versus-control comparison, so it only sees the second kind. This file measures "
            "the first kind on every finished pre-2021 drug trial, asks whether rare-disease trials "
            "stop for recruitment more often and how much of that is cancer, and then looks at the second "
            "kind among the labelled trials. See BELIEFS.md B19 and limit L1.\n"
        ),
        summary_text(population, labelled),
        (
            f"Population: {population.height:,} Phase 2 or 3 trials with a drug, biological or genetic "
            "intervention, status completed, terminated or withdrawn, primary completion before 2021. "
            "Nothing from 2021 onwards was read, and the code refuses any table that holds it. "
            "Crude intervals treat trials as independent, so they are somewhat too narrow; adjusted "
            "intervals resample sponsors."
        ),
        "## 1. What share of trials can the benchmark speak about?\n\n"
        + funnel_tables(population, set(dev["nct_id"])),
        "## 2. Why trials stopped\n\n" + reason_table(population, hand_read),
        "## 3. Rare versus not rare: the crude numbers\n\n"
        + group_size_table(population)
        + "\n\n"
        + contrast_table(population),
        "## 4. How much of that is cancer?\n\n"
        "### 4.1 The oncology split\n\n"
        + oncology_split_table(population)
        + "\n\n### 4.2 Adjusted ratio\n\n"
        + adjusted_overall_table(population)
        + "\n\n### 4.3 Outside oncology\n\n"
        + outside_oncology_table(population)
        + "\n\n### 4.4 By phase and sponsor\n\n"
        + phase_sponsor_table(population)
        + "\n\n### 4.5 Robustness\n\n"
        + robustness_table(population, alternative),
        "## 5. Within rare diseases, by prevalence band\n\n"
        + band_section(population, alternative),
        "## 6. How many patients\n\n" + enrolment_table(population),
        "## 7. The second kind of failure, in the labelled trials\n\n"
        + success_group_table(labelled)
        + "\n\n"
        + success_contrast_table(labelled)
        + "\n\n"
        + condition_table(labelled)
        + "\n\n"
        + success_band_table(labelled),
        "## 8. What this does and does not show\n\n" + limits_text(population, labelled, hand_read),
    ]
    return "\n\n".join(sections) + "\n"


def main(out_path: Path = RESULTS_PATH) -> None:
    population = build_population(download.iter_studies())
    ids = population["nct_id"].to_list()
    population = add_groups(population, rare_flags(ids))
    alternative = rare_flags(ids, measure_first=False)
    text = report(population, load_dev(), alternative, load_stop_reason_read())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
