"""Does genetic support for the drug's target go with trial success? Development set only.

    uv run python -m elman_exercise.genetics_summary

Joins data/interim/genetics.parquet to the development trials and writes
results/dev_genetics.md. It reads the development file through
`dev_summary.load_dev` and refuses anything holding a trial that completed on or
after 2021-01-01, so it cannot show a test-set outcome. The one table that touches
other splits (Open Targets drug type against our modality) uses features only.

The grading cut points are constants in `genetics.py`, fixed before any outcome
was looked at. The Wilson and Newcombe intervals treat trials as independent, which
they are not (limit L10), so they are too narrow; section 10 adds a cluster bootstrap
by lead drug. Sections 9 onwards were added after the development result was seen.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from elman_exercise.dataset import ROOT
from elman_exercise.dev_summary import (
    MISSING,
    Z_95,
    check_is_development,
    load_dev,
    wilson_interval,
)
from elman_exercise.genetics import BANDS, OUT_PATH, START_CUTS, TODAY_CUTS, band
from elman_exercise.opentargets import INTERIM, OT_DIR

RESULTS_PATH = ROOT / "results" / "dev_genetics.md"
LABEL = "any_arm"
START = "dated before start"
VERSIONS = {"today": "supported_today", START: "supported_at_start"}
SCORE_COLUMNS = {"today": "genetic_score_today", START: "genetic_score_at_start"}
CUTS = {"today": TODAY_CUTS, START: START_CUTS}
INVESTIGATIONAL = {
    "today": "supported_today_investigational",
    START: "supported_at_start_investigational",
}
SUPPORTED_TARGETS = {"today": "supported_targets_today", START: "supported_targets_at_start"}
TOP_TARGET = {"today": "top_target_today", START: "top_target_at_start"}
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261110
ESTABLISHED_AFTER = 5  # more than this many earlier trials of the lead drug = "established"
START_CAVEAT = (
    '"Evidence dated before the trial started, as curated today": the history is rebuilt '
    "from publication years using today's curation, today's gene-assignment model and today's "
    "disease ontology. Against the archived 2020 release it agrees on which pairs have no "
    "evidence, but on positives only about half the time. So it is not what a predictor could "
    "have seen at the time."
)
ERA_SPLIT_YEAR = 2015
MAX_ITER = 2000
MH_METHOD = "Mantel-Haenszel over phase x sponsor x era x area strata"
N_AREAS = 5


# --- small statistics ------------------------------------------------------------------------


def newcombe_difference(s1: int, n1: int, s0: int, n0: int) -> tuple[float, float, float]:
    """Rate in group 1 minus group 0, in proportion, with Newcombe's 95% interval.

    Built from the two Wilson intervals, so it behaves at small counts. All nan if
    either group is empty.
    """
    if n1 == 0 or n0 == 0:
        return math.nan, math.nan, math.nan
    p1, p0 = s1 / n1, s0 / n0
    l1, u1 = wilson_interval(s1, n1)
    l0, u0 = wilson_interval(s0, n0)
    low = (p1 - p0) - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    high = (p1 - p0) + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return p1 - p0, low, high


def mantel_haenszel(strata: list[tuple[int, int, int, int]]) -> tuple[float, float, float]:
    """Pooled odds ratio over strata of (a, b, c, d) = (exposed yes/no, unexposed yes/no).

    Returns (odds ratio, low, high); the interval is the Robins-Breslow-Greenland one.
    Strata missing one of the four cells still count; empty strata are skipped.
    """
    r = s = pr = ps_qr = qs = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0:
            continue
        r_i, s_i = a * d / n, b * c / n
        p_i, q_i = (a + d) / n, (b + c) / n
        r, s = r + r_i, s + s_i
        pr += p_i * r_i
        ps_qr += p_i * s_i + q_i * r_i
        qs += q_i * s_i
    if r == 0 or s == 0:
        return math.nan, math.nan, math.nan
    log_or = math.log(r / s)
    variance = pr / (2 * r**2) + ps_qr / (2 * r * s) + qs / (2 * s**2)
    half = Z_95 * math.sqrt(variance)
    return math.exp(log_or), math.exp(log_or - half), math.exp(log_or + half)


def design_matrix(frame: pl.DataFrame, flag: str) -> tuple[np.ndarray, list[str]]:
    """Predictors for the adjusted model: the flag, phase, sponsor class, area, start year.

    Sponsor class is industry versus everything else; therapeutic area is a set of
    yes/no columns with a missing group; start year is centred and in decades.
    """
    columns: dict[str, np.ndarray] = {flag: frame[flag].cast(pl.Float64).to_numpy()}
    columns["phase 3 (vs 2)"] = (frame["phase"] == "phase3").cast(pl.Float64).to_numpy()
    columns["industry sponsor"] = (frame["sponsor_class"] == "INDUSTRY").cast(pl.Float64).to_numpy()
    areas = frame["therapeutic_area"].fill_null(MISSING)
    counts = areas.value_counts().sort("count", descending=True)["therapeutic_area"].to_list()
    for area in counts[1:]:  # the most common area is the reference
        columns[f"area: {area}"] = (areas == area).cast(pl.Float64).to_numpy()
    years = frame["start_year"].cast(pl.Float64).to_numpy()
    columns["start year (per decade)"] = (years - years.mean()) / 10
    return np.column_stack(list(columns.values())), list(columns)


def adjusted_odds_ratio(frame: pl.DataFrame, flag: str) -> tuple[float, float, float, str]:
    """Odds ratio for the flag from logistic regression on the label, with a 95% interval.

    Uses scikit-learn if it is installed. The fit is deliberately UNPENALISED: an odds
    ratio with a Wald interval from the information matrix describes the maximum-likelihood
    estimate, and a penalty would shrink the coefficient and make that interval wrong.
    The fit must converge (checked; a non-converged fit raises). Otherwise falls back to
    the Mantel-Haenszel pooled odds ratio. The last item says which was used.
    """
    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return (*mh_adjusted(frame, flag), MH_METHOD)
    x, _ = design_matrix(frame, flag)
    y = frame[LABEL].cast(pl.Int64).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=np.inf, max_iter=MAX_ITER).fit(x, y)
    if int(model.n_iter_[0]) >= MAX_ITER:
        raise RuntimeError("The unpenalised logistic regression did not converge.")
    p = model.predict_proba(x)[:, 1]
    design = np.column_stack([np.ones(len(y)), x])
    information = design.T @ (design * (p * (1 - p))[:, None])
    se = math.sqrt(np.linalg.inv(information)[1, 1])
    beta = float(model.coef_[0][0])
    return (math.exp(beta), math.exp(beta - Z_95 * se), math.exp(beta + Z_95 * se),
            "logistic regression (scikit-learn, unpenalised, converged)")  # fmt: skip


def strata_of(frame: pl.DataFrame) -> pl.Series:
    """Phase x sponsor class (industry or not) x era of primary completion x therapeutic
    area (a missing area is its own stratum)."""
    era = pl.when(pl.col("primary_completion_date").dt.year() < ERA_SPLIT_YEAR)
    era = era.then(pl.lit("early")).otherwise(pl.lit("late"))
    industry = (
        pl.when(pl.col("sponsor_class") == "INDUSTRY").then(pl.lit("ind")).otherwise(pl.lit("oth"))
    )
    area = pl.col("therapeutic_area").fill_null(MISSING)
    return frame.select(pl.concat_str([pl.col("phase"), industry, era, area], separator="|"))[
        "phase"
    ]


def mh_adjusted(frame: pl.DataFrame, flag: str) -> tuple[float, float, float]:
    strata = []
    keyed = frame.with_columns(pl.Series("_stratum", strata_of(frame)))
    for _, part in keyed.group_by("_stratum"):
        a = part.filter(pl.col(flag) & pl.col(LABEL)).height
        b = part.filter(pl.col(flag) & ~pl.col(LABEL)).height
        c = part.filter(~pl.col(flag) & pl.col(LABEL)).height
        d = part.filter(~pl.col(flag) & ~pl.col(LABEL)).height
        strata.append((a, b, c, d))
    return mantel_haenszel(strata)


# --- tables -----------------------------------------------------------------------------------


def pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1%}"


def points(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{100 * x:+.1f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def rate_row(label: str, frame: pl.DataFrame) -> list[str]:
    outcomes = frame[LABEL].to_list()
    n, s = len(outcomes), sum(outcomes)
    low, high = wilson_interval(s, n)
    rate = s / n if n else math.nan
    return [label, f"{n:,}", pct(rate), f"{pct(low)} to {pct(high)}"]


def rate_table(groups: list[tuple[str, pl.DataFrame]]) -> str:
    return md_table(["", "n", "success rate", "95% interval"], [rate_row(n, f) for n, f in groups])


def difference_row(label: str, frame: pl.DataFrame, flag: str) -> list[str]:
    """Supported versus not within `frame`: counts, rates, difference in points with interval."""
    yes, no = frame.filter(pl.col(flag)), frame.filter(~pl.col(flag))
    s1, s0 = int(yes[LABEL].sum()), int(no[LABEL].sum())
    diff, low, high = newcombe_difference(s1, yes.height, s0, no.height)
    return [
        label,
        f"{yes.height:,}",
        pct(s1 / yes.height if yes.height else math.nan),
        f"{no.height:,}",
        pct(s0 / no.height if no.height else math.nan),
        points(diff),
        f"{points(low)} to {points(high)}",
    ]


DIFF_HEADER = ["", "n supported", "rate", "n not", "rate", "difference (points)", "95% interval"]


def usable(frame: pl.DataFrame, flag: str) -> pl.DataFrame:
    """Trials with the label determined and the flag known (so mapped, and dated if at start)."""
    return frame.filter(pl.col(LABEL).is_not_null() & pl.col(flag).is_not_null())


def mapped_table(frame: pl.DataFrame) -> str:
    groups = [
        ("Mapped to a target and disease", frame.filter(pl.col("mapped"))),
        ("Not mapped", frame.filter(~pl.col("mapped"))),
    ]
    return rate_table(groups)


def supported_table(frame: pl.DataFrame) -> str:
    rows = [
        difference_row(f"Evidence {v}", usable(frame, flag), flag) for v, flag in VERSIONS.items()
    ]
    return md_table(DIFF_HEADER, rows)


def graded_table(frame: pl.DataFrame, version: str) -> str:
    column = SCORE_COLUMNS[version]
    cuts = CUTS[version]
    known = usable(frame, VERSIONS[version])
    bands = [band(s, cuts) for s in known[column].to_list()]
    labelled = known.with_columns(pl.Series("_band", bands, dtype=pl.String))
    groups = [(b, labelled.filter(pl.col("_band") == b)) for b in BANDS]
    return rate_table(groups)


def strata_tables(frame: pl.DataFrame, flag: str) -> str:
    known = usable(frame, flag)
    era = known["primary_completion_date"].dt.year()
    areas = (
        known.filter(pl.col("therapeutic_area").is_not_null())["therapeutic_area"]
        .value_counts()
        .sort("count", descending=True)["therapeutic_area"]
        .to_list()[:N_AREAS]
    )
    cuts: list[tuple[str, pl.DataFrame]] = [
        ("Phase 2", known.filter(pl.col("phase") == "phase2")),
        ("Phase 3", known.filter(pl.col("phase") == "phase3")),
        (f"Completed before {ERA_SPLIT_YEAR}", known.filter(era < ERA_SPLIT_YEAR)),
        (f"Completed {ERA_SPLIT_YEAR} to 2020", known.filter(era >= ERA_SPLIT_YEAR)),
        ("Industry sponsor", known.filter(pl.col("sponsor_class") == "INDUSTRY")),
        ("Other sponsor", known.filter(pl.col("sponsor_class") != "INDUSTRY")),
    ]
    cuts += [(f"Area: {a}", known.filter(pl.col("therapeutic_area") == a)) for a in areas]
    return md_table(DIFF_HEADER, [difference_row(label, part, flag) for label, part in cuts])


def adjusted_table(frame: pl.DataFrame) -> tuple[str, list[str]]:
    rows, methods = [], []
    for version, flag in VERSIONS.items():
        known = usable(frame, flag).filter(pl.col("start_year").is_not_null())
        known = known.filter(pl.col("phase").is_not_null() & pl.col("sponsor_class").is_not_null())
        crude = mh_crude(known, flag)
        odds, low, high, method = adjusted_odds_ratio(known, flag)
        mh = mh_adjusted(known, flag)
        rows.append([
            f"Evidence {version}", f"{known.height:,}", f"{crude:.2f}",
            f"{odds:.2f} ({low:.2f} to {high:.2f})",
            f"{mh[0]:.2f} ({mh[1]:.2f} to {mh[2]:.2f})",
        ])  # fmt: skip
        methods.append(method)
    header = ["", "n", "crude odds ratio", "adjusted odds ratio (logistic)", "Mantel-Haenszel"]
    return md_table(header, rows), methods


def mh_crude(frame: pl.DataFrame, flag: str) -> float:
    a = frame.filter(pl.col(flag) & pl.col(LABEL)).height
    b = frame.filter(pl.col(flag) & ~pl.col(LABEL)).height
    c = frame.filter(~pl.col(flag) & pl.col(LABEL)).height
    d = frame.filter(~pl.col(flag) & ~pl.col(LABEL)).height
    return (a * d) / (b * c) if b * c else math.nan


def few_targets_table(frame: pl.DataFrame) -> str:
    few = frame.filter(pl.col("few_targets").fill_null(False))
    rows = [difference_row(f"Evidence {v}, three targets or fewer", usable(few, f), f)
            for v, f in VERSIONS.items()]  # fmt: skip
    return md_table(DIFF_HEADER, rows)


# --- drug type against our modality -------------------------------------------------------------

OT_TO_MODALITY = {
    "small molecule": "small molecule",
    "antibody": "biologic",
    "protein": "biologic",
    "oligonucleotide": "advanced",
    "cell": "advanced",
    "gene": "advanced",
    "vaccine": "vaccine",
    "mixed": "mixed",
}


def modality_agreement(frame: pl.DataFrame) -> tuple[str, float, int]:
    """Cross-table of our modality (rows) against Open Targets' drug type (columns).

    `frame` has `modality` and `drug_type`. Agreement is counted over trials where both
    say something comparable: Open Targets type not unknown or other, ours not missing.
    A vaccine counts as agreeing when we flag it a vaccine, whatever our modality says.
    """
    rows = sorted(set(frame["modality"].fill_null(MISSING)))
    cols = sorted(set(frame["drug_type"]))
    counts = (
        frame.with_columns(pl.col("modality").fill_null(MISSING))
        .group_by("modality", "drug_type")
        .len()
    )
    lookup = {(m, d): n for m, d, n in counts.iter_rows()}
    table = md_table(
        ["our modality \\ Open Targets type", *cols],
        [[m, *[str(lookup.get((m, d), 0)) for d in cols]] for m in rows],
    )
    comparable = frame.filter(
        pl.col("drug_type").is_in(list(OT_TO_MODALITY)) & pl.col("modality").is_not_null()
    )
    mapped = comparable["drug_type"].replace_strict(OT_TO_MODALITY, return_dtype=pl.String)
    agree = int((mapped == comparable["modality"]).sum())
    return table, agree / comparable.height if comparable.height else math.nan, comparable.height


# --- added after the development result was seen -----------------------------------------------------


def cluster_bootstrap_difference(
    frame: pl.DataFrame, flag: str, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float, float, int]:
    """Supported minus not-supported success rate, with a percentile interval from a cluster
    bootstrap: whole lead-drug groups are resampled with replacement, so trials of one drug
    stay together. Returns (difference, low, high, number of clusters). Fixed seed."""
    known = usable(frame, flag).filter(pl.col("lead_drug").is_not_null())
    grouped = (
        known.group_by("lead_drug")
        .agg(
            pl.col(flag).sum().alias("n1"),
            (pl.col(flag) & pl.col(LABEL)).sum().alias("s1"),
            (~pl.col(flag)).sum().alias("n0"),
            (~pl.col(flag) & pl.col(LABEL)).sum().alias("s0"),
        )
        .sort("lead_drug")
    )
    counts = grouped.select("n1", "s1", "n0", "s0").to_numpy().astype(float)
    k = counts.shape[0]
    n1, s1, n0, s0 = counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_resamples):
        weights = np.bincount(rng.integers(0, k, size=k), minlength=k)
        r1, r2, r3, r4 = weights @ counts
        if r1 > 0 and r3 > 0:
            diffs.append(r2 / r1 - r4 / r3)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return s1 / n1 - s0 / n0, float(low), float(high), k


def drop_top_targets(frame: pl.DataFrame, targets: list[str]) -> pl.DataFrame:
    """Trials none of whose drug targets is in `targets`."""
    hit = pl.col("target_ids").list.eval(pl.element().is_in(targets)).list.any()
    return frame.filter(~hit.fill_null(False))


def target_counts(frame: pl.DataFrame, version: str) -> pl.DataFrame:
    """(target, n): supported trials (usable, label known) per supported target."""
    known = usable(frame, VERSIONS[version]).filter(pl.col(VERSIONS[version]))
    return (
        known.select(pl.col(SUPPORTED_TARGETS[version]).alias("target"))
        .explode("target")
        .drop_nulls()
        .group_by("target")
        .len()
        .rename({"len": "n"})
        .sort(["n", "target"], descending=[True, False])
    )


def name_target(target: str, symbols: dict[str, str]) -> str:
    return f"{symbols[target]} ({target})" if symbols.get(target) else target


def cluster_row(label: str, frame: pl.DataFrame, flag: str) -> list[str]:
    diff, low, high, k = cluster_bootstrap_difference(frame, flag)
    return [label, f"{k:,}", points(diff), f"{points(low)} to {points(high)}"]


def headline_with_cluster(frame: pl.DataFrame) -> str:
    """The headline difference with the independence-assuming and the cluster interval."""
    rows = []
    for v, flag in VERSIONS.items():
        known = usable(frame, flag)
        base = difference_row(f"Evidence {v}", known, flag)
        _, low, high, k = cluster_bootstrap_difference(frame, flag)
        rows.append([*base, f"{points(low)} to {points(high)}", f"{k:,}"])
    return md_table([*DIFF_HEADER[:-1], "95% interval (trials independent)",
                     "95% interval (cluster bootstrap by lead drug)", "lead drugs"], rows)  # fmt: skip


def investigational_section(frame: pl.DataFrame) -> str:
    rows, notes = [], []
    for v, flag in VERSIONS.items():
        inv = INVESTIGATIONAL[v]
        known = usable(frame, flag)
        any_sup = known.filter(pl.col(flag))
        via_inv = any_sup.filter(pl.col(inv))
        only_other = any_sup.height - via_inv.height
        notes.append(
            f"Evidence {v}: {any_sup.height:,} supported trials; {via_inv.height:,} are supported "
            f"through a drug matched to the treatment under test and {only_other:,} "
            f"({only_other / any_sup.height:.1%}) only through another listed drug."
        )
        clean = known.filter(~(pl.col(flag) & ~pl.col(inv)))
        both = clean.filter(pl.col("investigational_mapped"))
        rows += [
            difference_row(f"{v}: investigational support vs all other mapped", known, inv),
            difference_row(f"{v}: same, dropping trials supported only through other drugs",
                           clean, inv),
            difference_row(f"{v}: same, only trials whose investigational drug has a target",
                           both, inv),
        ]  # fmt: skip
    return "\n\n".join([" ".join(notes), md_table(DIFF_HEADER, rows)])


def concentration_section(frame: pl.DataFrame, symbols: dict[str, str]) -> str:
    out = []
    for v, flag in VERSIONS.items():
        counts = target_counts(frame, v)
        top = counts.head(10)
        out.append(
            f"Evidence {v}: the supported trials carry {counts.height:,} distinct supported targets."
        )
        out.append(md_table(
            ["target", "supported trials"],
            [[name_target(t, symbols), f"{n:,}"] for t, n in top.iter_rows()],
        ))  # fmt: skip
    rows = []
    for v, flag in VERSIONS.items():
        top5 = target_counts(frame, v).head(5)["target"].to_list()
        rows.append(difference_row(f"{v}: dropping every trial that touches the top 5 targets",
                                   usable(drop_top_targets(frame, top5), flag), flag))  # fmt: skip
    out.append("Headline difference after dropping every trial with any of the top five targets "
               "among its drug targets (supported or not):")  # fmt: skip
    out.append(md_table(DIFF_HEADER, rows))
    return "\n\n".join(out)


def established_section(frame: pl.DataFrame) -> str:
    rows = []
    for v, flag in VERSIONS.items():
        known = usable(frame, flag).filter(pl.col("lead_drug_earlier_trials").is_not_null())
        few = pl.col("lead_drug_earlier_trials") <= ESTABLISHED_AFTER
        rows.append(difference_row(f"{v}: {ESTABLISHED_AFTER} or fewer earlier trials of the "
                                   "lead drug", known.filter(few), flag))  # fmt: skip
        rows.append(difference_row(f"{v}: more than {ESTABLISHED_AFTER} earlier trials",
                                   known.filter(~few), flag))  # fmt: skip
    return md_table(DIFF_HEADER, rows)


def strict_section(frame: pl.DataFrame) -> str:
    rows = []
    for v, flag in VERSIONS.items():
        known = usable(frame, flag)
        bands = [band(x, CUTS[v]) for x in known[SCORE_COLUMNS[v]].to_list()]
        strict = known.with_columns(
            pl.Series("_strict", [b in ("middle", "high") for b in bands], dtype=pl.Boolean)
        )
        rows.append(difference_row(f"{v}: middle or high band vs none or low", strict, "_strict"))
    return md_table(DIFF_HEADER, rows)


def band_makeup_section(frame: pl.DataFrame, symbols: dict[str, str]) -> str:
    rows = []
    for v, flag in VERSIONS.items():
        known = usable(frame, flag)
        bands = [band(x, CUTS[v]) for x in known[SCORE_COLUMNS[v]].to_list()]
        labelled = known.with_columns(pl.Series("_band", bands, dtype=pl.String))
        for b in BANDS[1:]:
            part = labelled.filter(pl.col("_band") == b)
            top = (
                part.group_by(TOP_TARGET[v]).len().drop_nulls().sort(["len", TOP_TARGET[v]],
                                                                    descending=[True, False])
                .head(5)
            )  # fmt: skip
            listed = "; ".join(f"{name_target(t, symbols)} {n}" for t, n in top.iter_rows())
            rows.append([v, b, f"{part.height:,}", listed])
    return md_table(["version", "band", "trials", "top 5 targets (highest-scoring target, trials)"],
                    rows)  # fmt: skip


# --- who is mapped -------------------------------------------------------------------------------------

YEAR_BANDS = (("before 2010", 0, 2010), ("2010 to 2014", 2010, 2015),
              ("2015 to 2019", 2015, 2020), ("2020", 2020, 2100))  # fmt: skip
MAPPED_HEADER = ["", "n", "mapped share", "success, mapped", "success, not mapped"]


def mapped_row(label: str, frame: pl.DataFrame) -> list[str]:
    mapped, rest = frame.filter(pl.col("mapped")), frame.filter(~pl.col("mapped"))

    def rate(f: pl.DataFrame) -> str:
        return pct(f[LABEL].mean()) if f.height else "n/a"

    return [label, f"{frame.height:,}", pct(mapped.height / frame.height if frame.height else
                                            math.nan), rate(mapped), rate(rest)]  # fmt: skip


def mapped_coverage_tables(frame: pl.DataFrame) -> str:
    groups: list[tuple[str, pl.DataFrame]] = [
        ("Industry sponsor", frame.filter(pl.col("sponsor_class") == "INDUSTRY")),
        ("Other sponsor", frame.filter(pl.col("sponsor_class") != "INDUSTRY")),
    ]
    areas = frame["therapeutic_area"].fill_null(MISSING).value_counts().sort(
        "count", descending=True)["therapeutic_area"].to_list()  # fmt: skip
    groups += [
        (f"Area: {a}", frame.filter(pl.col("therapeutic_area").fill_null(MISSING) == a))
        for a in areas
    ]
    for label, lo, hi in YEAR_BANDS:
        groups.append(
            (f"Started {label}", frame.filter(pl.col("start_year").is_between(lo, hi - 1)))
        )
    groups.append(("Start year missing", frame.filter(pl.col("start_year").is_null())))
    first = md_table(MAPPED_HEADER, [mapped_row(label, f) for label, f in groups])
    cells = []
    for phase in ("phase2", "phase3"):
        for industry in (True, False):
            sponsor = pl.col("sponsor_class") == "INDUSTRY"
            part = frame.filter((pl.col("phase") == phase) & (sponsor if industry else ~sponsor))
            name = f"{phase[:-1]} {phase[-1]}, {'industry' if industry else 'other sponsor'}"
            row = difference_row(name, part, "mapped")
            cells.append(row)
    second = md_table(["", "n mapped", "rate", "n not mapped", "rate", "difference (points)",
                       "95% interval"], cells)  # fmt: skip
    return (
        first
        + "\n\nSuccess rate of mapped minus not-mapped trials, within phase x sponsor cells:\n\n"
        + second
    )


# --- the report ------------------------------------------------------------------------------------

DEFINITION = (
    '**What "genetic evidence" is here.** The `genetic_association` datatype of Open Targets '
    "release 26.09, and nothing else. Checked on the release's own evidence tables "
    "(`datatypeId` column under https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.09/"
    "output/evidence_*/): the datatype is built from five sources, ClinVar germline (eva), gene "
    "burden tests, GWAS credible sets, Orphanet and UniProt curated variants. ClinGen, "
    "Gene2Phenotype, Genomics England PanelApp and UniProt literature are in a separate "
    "datatype (`genetic_literature`) and are not counted, so much curated Mendelian evidence is "
    "excluded. Clinical precedence (known drugs) is a separate datatype and is excluded. "
    '"Supported" means a score above zero, a low bar: ClinVar entries below "pathogenic" '
    "still score (drug response, association and protective 0.9; likely pathogenic 0.7; "
    "uncertain significance, conflicting and other 0.3, per "
    "https://platform-docs.opentargets.org/evidence), so some support is pharmacogenomic or of "
    'uncertain significance. Scores are "indirect": evidence on narrower disease terms is '
    "rolled up to broader ones. Drugs are every drug Open Targets lists for the trial, "
    "including comparators and background therapy. The results apply to trials Open Targets "
    "could map to a target and disease (section 8)."
)

MINIKEL = (
    'Minikel et al. 2024 (Nature, "Refining the impact of genetic evidence on clinical '
    'success") estimate that drug mechanisms with genetic support are 2.6 times as likely to '
    "succeed. In the text I read (the PubMed Central full text), that figure is not labelled as "
    "cumulative, but their Methods define Phase I to launch as the product of the per-phase "
    "relative successes, so it is probably a cumulative Phase I to launch ratio. Their definition "
    "is stricter than ours: a genetic association mapped to the target gene (for GWAS, at least "
    "50% locus-to-gene share) for a trait at least 0.8 similar to the indication. They report the "
    "effect as strongest in Phases II and III and weakest in Phase I, but the per-phase ratios "
    "are shown only in figures (Fig. 2b to d) and I could not read the numbers, so I do not quote "
    "them. Ours is a different quantity: a per-trial ratio of primary-endpoint success, "
    "68.1/57.3 = 1.19 with today's evidence and 74.0/59.6 = 1.24 with evidence dated before "
    "start, among trials that posted a comparison, with a permissive definition. The two figures "
    "are not on the same scale and should not be read as agreement or disagreement. Not "
    "verified: Minikel's per-phase values and the confidence interval for 2.6."
)


def with_genetics(dev: pl.DataFrame, genetics: pl.DataFrame) -> pl.DataFrame:
    """Development trials with their genetic features. Every trial must have a row."""
    check_is_development(dev)
    joined = dev.join(genetics.drop("split"), on="nct_id", how="left")
    if joined["mapped"].null_count():
        raise ValueError("Some development trials have no row in the genetics table.")
    return joined


def report(
    frame: pl.DataFrame,
    agreement: tuple[str, float, int],
    symbols: dict[str, str] | None = None,
) -> str:
    """The whole markdown report. Refuses data that is not purely development data."""
    check_is_development(frame)
    symbols = symbols or {}
    mapped = frame.filter(pl.col("mapped"))
    adjusted, methods = adjusted_table(frame)
    table, rate, n_compared = agreement
    start_unknown = int(mapped["supported_at_start"].is_null().sum())
    parts = [
        "# Genetic evidence and trial success: development set\n",
        DEFINITION,
        (f'**The "dated before start" version.** {START_CAVEAT}'),
        (
            f"Development trials (primary completion before 2021): {frame.height:,}. "
            f"Success means `{LABEL}`. Wilson and Newcombe intervals are 95% and treat trials as "
            "independent, so they are too narrow (section 10 gives a wider one). Definitions are "
            "in BELIEFS.md B18 and B18a; the cut points for bands were fixed before outcomes were "
            f"examined: today {TODAY_CUTS[0]:.3f} and {TODAY_CUTS[1]:.3f}, dated before start "
            f"{START_CUTS[0]:.3f} and {START_CUTS[1]:.3f}."
        ),
        (
            "**Which analyses were pre-specified.** Only the definition of support, its two "
            "versions, and the three bands with their cut points (sections 2 and 3) were written "
            "down (BELIEFS.md B18, B18a) before any outcome was joined. Sections 1 and 4 to 7 were "
            "run in the first look at outcomes and were not recorded beforehand, so treat them as "
            "exploratory. Sections 8 to 15 were added after the development result was seen, "
            "in response to review, and are labelled as such."
        ),
        "## 1. Trials that map to a target and disease\n",
        "Do trials we can link to a target and disease succeed at a different rate from the rest?\n",
        mapped_table(frame),
        "## 2. Supported versus not supported (mapped trials; pre-specified)\n",
        (
            "Among mapped trials, how much more often do those with genetic support succeed? "
            f"The dated-before-start version leaves out {start_unknown:,} mapped trials with no "
            "start year. Success rates are per trial; the interval is Newcombe's and treats trials "
            "as independent.\n"
        ),
        supported_table(frame),
        "## 3. By strength of evidence (pre-specified)\n",
        "Does success rise from none to low to middle to high genetic score? Today's evidence:\n",
        graded_table(frame, "today"),
        "\nEvidence dated before the trial started, as curated today:\n",
        graded_table(frame, START),
        "## 4. Within subgroups\n",
        (
            "Is the gap the same in each phase, era, sponsor class and large therapeutic area? "
            "Evidence today:\n"
        ),
        strata_tables(frame, VERSIONS["today"]),
        "\nEvidence dated before the trial started, as curated today:\n",
        strata_tables(frame, VERSIONS[START]),
        "## 5. Adjusted odds ratios\n",
        (
            "After allowing for the other things that differ between the groups, how large is the "
            f"effect as an odds ratio? Logistic column: {methods[0]}, adjusted for phase, sponsor "
            "class (industry or not), therapeutic area and start year (linear). Mantel-Haenszel "
            "column: stratified by phase x sponsor class x era (before 2015 or not) x therapeutic "
            "area, pooling the strata under the assumption that the odds ratio is the same in "
            "every stratum (so it is not assumption-free, and it ignores strata with only one "
            "kind of trial). The two adjust for the same things in different ways, and agree "
            "closely only if the effect is similar across strata. Both intervals treat trials as "
            "independent.\n"
        ),
        adjusted,
        "## 6. Sensitivity: trials with three targets or fewer\n",
        "Does the gap survive when we drop trials whose many targets give many chances of a hit?\n",
        few_targets_table(frame),
        "## 7. Open Targets drug type against our modality (all splits, features only)\n",
        (
            "Do two independent ways of naming a trial's drug class agree? "
            f"Agreement over the {n_compared:,} trials where both are informative "
            f"(antibody and protein count as biologic): {pct(rate)}.\n"
        ),
        table,
        "## 8. Who is mapped (features and development outcomes only)\n",
        (
            "**Results here apply to mapped trials only.** Drugs with curated targets are more "
            "likely to be ones that progressed, so mapped trials are not a random sample, and "
            "mapping coverage is lower for recent trials. The last two columns are success rates "
            "within each group.\n"
        ),
        mapped_coverage_tables(frame),
        "## 9. Added after the development result was seen: investigational drug only\n",
        (
            "Support must come through a drug matched to the trial's treatment under test "
            "(registry interventions in experimental arms, see `genetics.py` for the matching rule "
            "and its failure modes) rather than a comparator or background drug. Development "
            "set only.\n"
        ),
        investigational_section(frame),
        "## 10. Added after the development result was seen: interval for repeated drugs\n",
        (
            "Cluster bootstrap, 1,000 resamples, seed fixed: whole lead-drug groups (first "
            "investigational drug, or first drug) are resampled, so trials of one drug are not "
            "counted as independent. It is wider because trials of the same drug share a target, a "
            "sponsor and often a result, so there is less independent information than trials "
            "suggest. Development set only.\n"
        ),
        headline_with_cluster(frame),
        "## 11. Added after the development result was seen: concentration\n",
        (
            "How many targets carry the supported trials, which, and does the difference survive "
            "dropping the five biggest? A trial is counted under each target with a positive "
            "score for its disease. Development set only.\n"
        ),
        concentration_section(frame, symbols),
        "## 12. Added after the development result was seen: established versus newer drugs\n",
        (
            "Earlier trials of the same lead drug: ClinicalTrials.gov trials in Open Targets' "
            "report (any phase) that list the drug and started strictly earlier. Start dates and "
            "drug ids only, no outcomes. Development set only.\n"
        ),
        established_section(frame),
        "## 13. Added after the development result was seen: stricter threshold\n",
        (
            "Supported means the middle or high band (score above the existing lower cut point). "
            "Chosen after seeing that the low band looked like none. Development set only.\n"
        ),
        strict_section(frame),
        "## 14. Added after the development result was seen: what each band is made of\n",
        (
            "The target of each trial's highest-scoring pair, most common first. Development "
            "set only.\n"
        ),
        band_makeup_section(frame, symbols),
        "## 15. Added after the development result was seen: comparison with Minikel et al.\n",
        MINIKEL,
    ]
    return "\n\n".join(part.strip("\n") for part in parts) + "\n"


def load_drug_type_frame(genetics: pl.DataFrame) -> pl.DataFrame:
    """Our modality and Open Targets' drug type for every trial. Feature columns only."""
    ours = pl.concat(
        [pl.read_parquet(INTERIM / n, columns=["nct_id", "modality"])
         for n in ("dev.parquet", "test_features.parquet")]
    )  # fmt: skip
    return ours.join(genetics.select("nct_id", "drug_type"), on="nct_id")


def load_symbols() -> dict[str, str]:
    """Ensembl id -> approved gene symbol, from the target table (two columns) if downloaded."""
    path = OT_DIR / "target.parquet"
    if not path.exists():
        return {}
    table = pl.read_parquet(path, columns=["id", "approvedSymbol"]).drop_nulls()
    return dict(zip(table["id"], table["approvedSymbol"], strict=True))


def main(out_path: Path = RESULTS_PATH) -> None:
    genetics = pl.read_parquet(OUT_PATH)
    frame = with_genetics(load_dev(), genetics)
    text = report(frame, modality_agreement(load_drug_type_frame(genetics)), load_symbols())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
