"""Success rates on the development set (trials that finished before 2021).

    uv run python -m trial_benchmark.dev_summary

Reads only data/interim/dev.parquet and refuses anything else, so this script
cannot show a test-set outcome. Each cell shows how many trials, the share that
met the label, and a 95% Wilson interval. The interval treats trials as
independent, which they are not (limit L10), so it is somewhat too narrow.

Rates here are descriptive. They are the first look at outcomes, and so belong
in BELIEFS.md under "After development set".
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from trial_benchmark.dataset import ROOT, TEST_START

DEV_PATH = ROOT / "data" / "interim" / "dev.parquet"
MISSING = "(missing)"
Z_95 = 1.96


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for a proportion. (nan, nan) when n is 0."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def load_dev(path: Path = DEV_PATH) -> pl.DataFrame:
    """Read the development file, and refuse anything that is not purely development data."""
    if Path(path).resolve() != DEV_PATH.resolve():
        raise ValueError(f"This script reads only {DEV_PATH}, not {path}.")
    frame = pl.read_parquet(path)
    check_is_development(frame)
    return frame


def check_is_development(frame: pl.DataFrame) -> None:
    if frame.height == 0:
        raise ValueError("The development file has no rows.")
    dates = frame["primary_completion_date"]
    if dates.null_count() > 0:
        raise ValueError("Refusing to summarise: some trials have no primary completion date.")
    if set(frame["split"].unique()) != {"dev"} or dates.max() >= TEST_START:
        raise ValueError("Refusing to summarise: the data holds trials from 2021 or later.")


ENROLLMENT_ORDER = ("<100", "100-299", "300-999", "1000+")


def enrollment_band(n: int | None) -> str:
    if n is None:
        return MISSING
    if n < 100:
        return "<100"
    if n < 300:
        return "100-299"
    return "300-999" if n < 1000 else "1000+"


def arms_band(n: int | None) -> str:
    if n is None:
        return MISSING
    return str(n) if n < 3 else "3+"


def cell(outcomes: list[bool]) -> tuple[int, int, float, float, float]:
    """(n, successes, rate, low, high) for a list of True/False outcomes."""
    n, successes = len(outcomes), sum(outcomes)
    low, high = wilson_interval(successes, n)
    return n, successes, (successes / n if n else math.nan), low, high


def format_cell(label: str, outcomes: list[bool], width: int) -> str:
    n, successes, rate, low, high = cell(outcomes)
    if n == 0:
        return f"  {label:<{width}} {0:>6}"
    return f"  {label:<{width}} {n:>6}  {successes:>6}  {rate:6.1%}  [{low:6.1%}, {high:6.1%}]"


def by_group(
    frame: pl.DataFrame,
    title: str,
    groups: list,
    outcome: str = "any_arm",
    order: tuple[str, ...] | None = None,
) -> str:
    """One table: the rate of `outcome` within each group. `groups` has one value per row.

    `order` fixes the row order; names it does not list follow, in natural order.
    """
    labelled = frame.with_columns(pl.Series("_group", groups, dtype=pl.String).fill_null(MISSING))
    known = labelled.filter(pl.col(outcome).is_not_null())
    rank = {name: i for i, name in enumerate(order or ())}
    names = sorted(
        known["_group"].unique().to_list(),
        key=lambda name: (rank.get(name, len(rank)), _natural_order(name)),
    )
    width = max([len(name) for name in names] + [5])
    header = f"  {'':<{width}} {'n':>6}  {'met':>6}  {'rate':>6}  95% interval"
    lines = [f"{title}", header]
    for name in names:
        outcomes = known.filter(pl.col("_group") == name)[outcome].to_list()
        lines.append(format_cell(name, outcomes, width))
    return "\n".join(lines)


def _natural_order(name: str) -> tuple:
    """Order bands and years sensibly: numbers by value, the rest alphabetically, missing last."""
    digits = "".join(ch for ch in name.split("-")[0] if ch.isdigit())
    return (name == MISSING, int(digits) if digits else 0, name)


def overall_section(frame: pl.DataFrame) -> str:
    width = 34
    lines = [
        "Overall (trials with the label determined)",
        f"  {'':<{width}} {'n':>6}  {'met':>6}  {'rate':>6}  95% interval",
    ]
    for column in ("any_arm", "all_arms"):
        outcomes = frame.filter(pl.col(column).is_not_null())[column].to_list()
        lines.append(format_cell(column, outcomes, width))
    ladder = frame.filter(
        pl.col("has_dose_ladder")
        & pl.col("highest_dose").is_not_null()
        & pl.col("any_arm").is_not_null()
    )
    lines.append("  On dose-ladder trials only:")
    for column in ("any_arm", "highest_dose"):
        lines.append(format_cell(column, ladder[column].to_list(), width))
    return "\n".join(lines)


def summary(frame: pl.DataFrame) -> str:
    """The whole printout. Refuses data that is not purely development data."""
    check_is_development(frame)

    def column(name: str) -> list:
        return [None if v is None else str(v) for v in frame[name].to_list()]

    sections = [
        f"Development set: {frame.height} trials, primary completion before {TEST_START}.",
        overall_section(frame),
        "Headline label (any_arm) by subgroup",
        by_group(frame, "By phase", column("phase")),
        by_group(frame, "By control kind (from the results section)", column("control_kind")),
        by_group(frame, "By modality", column("modality")),
        by_group(frame, "By vaccine flag", column("is_vaccine")),
        by_group(frame, "By sponsor class", column("sponsor_class")),
        by_group(frame, "By therapeutic area", column("therapeutic_area")),
        "Post-hoc fields: known only after the trial ended; for slicing, not prediction.",
        by_group(
            frame,
            "By enrollment (actual, post-hoc)",
            [enrollment_band(n) for n in frame["enrollment"].to_list()],
            order=ENROLLMENT_ORDER,
        ),
        by_group(frame, "By overall status (post-hoc)", column("overall_status")),
        "Results-derived field: taken from the results section, not from registration.",
        by_group(
            frame,
            "By number of treatment arms compared (from the results section)",
            [arms_band(n) for n in frame["n_arms"].to_list()],
        ),
        by_group(
            frame,
            "By primary completion year",
            [str(d.year) for d in frame["primary_completion_date"]],
        ),
    ]
    return "\n\n".join(sections)


def main() -> None:
    print(summary(load_dev()))


if __name__ == "__main__":
    main()
