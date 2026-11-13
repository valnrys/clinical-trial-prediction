"""Write results/dev_biology.md: does target and drug biology alone say anything about success?

    uv run python -m trial_benchmark.biology_report

Development trials only (primary completion before 2021), through the guarded loader in
`validate`. No test or late trial's outcome is read. The score is `biology_score` (the recipe
declared in advance; nothing is tuned here). Design, as in the scratch analysis it ports:

- explore = trials finishing before 2015. The three cut points and the logistic model come from
  these trials only.
- confirm = trials finishing 2015 to 2020, scored with those fixed cut points and that fitted
  model. (The six flags themselves were chosen after seeing all development data, 2015 to 2020
  included, so this confirms the combined score, not the choice of features.)
- Population: trials with at least one mapped investigational target under the link version.
- Outcome: success of any arm (`any_arm`). Rates are per 100 trials. Intervals are 95%,
  bootstrapped over lead drugs (a trial with no lead drug is its own cluster).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from trial_benchmark import adjusted as adj
from trial_benchmark import baselines as bl
from trial_benchmark import biology_score as bs
from trial_benchmark import metrics, validate, whathow

REPORT_PATH = validate.ROOT / "results" / "dev_biology.md"
VERSIONS = ("today", "2020", "2020_names")
LINK_TITLE = {
    "today": "today's links (Open Targets 26.09)",
    "2020": "2020 links (Open Targets 20.06 targets, today's drug match)",
    "2020_names": "2020 names and links (drug matched by 20.06 names; no chemistry flags)",
}
PHASES = ("phase2", "phase3")
ADJUSTMENTS = {
    "phase only": [],
    "phase + area": ["therapeutic_area"],
    "phase + sponsor + area + modality": ["sponsor_class", "therapeutic_area", "modality"],
}
GRID_RESAMPLES = 1000
MIN_LEVEL = 100  # a category level needs this many trials to get its own contrast
PERMUTATIONS = 2000


# ---------------------------------------------------------------- the analysis frame


def load(version: str) -> pl.DataFrame:
    """Development trials with a mapped target under `version`, scored by the declared recipe.

    Adds: y, year, period (explore / confirm), the flags, n_flags, group, p_bio and third. The
    logistic model is fitted on the explore rows only.
    """
    dev = validate.load_development_with({whathow.table_name(version), "precedent"})
    features = bs.feature_frame(version).drop("lead_drug")
    f = dev.join(features, on="nct_id", how="inner", validate="1:1")
    if f["primary_completion_date"].dt.year().max() > 2020:
        raise ValueError("A trial finishing after 2020 reached the development analysis.")
    f = f.with_columns(
        y=pl.col("any_arm").cast(pl.Boolean),
        year=pl.col("primary_completion_date").dt.year(),
        supported=pl.col("wh_supported_at_start"),
        precedent_strict=pl.col(f"precedent_exp_{version}"),
    ).with_columns(
        period=pl.when(pl.col("year") < bs.EXPLORE_BEFORE)
        .then(pl.lit("explore"))
        .otherwise(pl.lit("confirm"))
    )
    f = f.join(bs.flag_frame(f), on="nct_id", validate="1:1")
    train = f.filter(pl.col("period") == "explore")
    fitted = bs.fit_logistic(train, train["y"].to_numpy())
    p_bio = fitted.score(f)
    f = f.with_columns(
        p_bio=pl.Series(p_bio),
        group=pl.Series(bs.flag_groups(f["n_flags"].to_numpy())),
        third=pl.Series(bs.thirds(p_bio, fitted.tertile_cuts)),
    )
    return f.with_columns(
        pl.lit(fitted.tertile_cuts[0]).alias("_cut1"), pl.lit(fitted.tertile_cuts[1]).alias("_cut2")
    )


def cuts_of(f: pl.DataFrame) -> tuple[float, float]:
    return (float(f["_cut1"][0]), float(f["_cut2"][0]))


def confirm(f: pl.DataFrame) -> pl.DataFrame:
    return f.filter(pl.col("period") == "confirm")


# ---------------------------------------------------------------- small formatting helpers


def rate_text(row: dict) -> str:
    return f"{row['per_100']:.0f} ({row['low']:.0f}-{row['high']:.0f}), n={row['n']}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return [*out, ""]


def flag_stats(f: pl.DataFrame, extra: list[str] | None = None) -> bs.Contrast:
    return bs.flag_contrast(f["y"], f["n_flags"], f["phase"], f["lead_drug"], _strata(f, extra))


def thirds_stats(f: pl.DataFrame, extra: list[str] | None = None) -> bs.Contrast:
    return bs.thirds_contrast(
        f["y"], f["p_bio"], cuts_of(f), f["phase"], f["lead_drug"], _strata(f, extra)
    )


def _strata(f: pl.DataFrame, extra: list[str] | None) -> dict[str, np.ndarray] | None:
    if not extra:
        return None
    return {c: f[c].fill_null("NA").to_numpy() for c in extra}


def score_tables(f: pl.DataFrame, title: str) -> list[str]:
    """The flag-count table and the logistic-thirds table within phase, and both contrasts."""
    out = [f"### {title}", ""]
    for column, names, label, stat in (
        ("group", bs.GROUPS, "flags", flag_stats),
        ("third", bs.THIRDS, "logistic third", thirds_stats),
    ):
        t = bs.by_group(f["y"], f[column], f["phase"], f["lead_drug"])
        rows = []
        for ph in PHASES:
            cells = []
            for g in names:
                r = t.filter((pl.col("phase") == ph) & (pl.col("group") == g))
                cells.append(rate_text(r.row(0, named=True)) if r.height else "n/a")
            rows.append([ph, *cells])
        both = stat(f)
        head = [
            "phase",
            *(("1 flag" if g == "1" else f"{g} flags") if label == "flags" else g for g in names),
        ]
        out.append(
            f"Successes per 100 by {label}, within phase "
            f"({'2+ minus 0' if label == 'flags' else 'high minus low'}, pooled over phase: "
            f"**{both.text()}**; n {both.n_exposed} vs {both.n_reference}):"
        )
        out.append("")
        out += table(head, rows)
    return out


def count_table(f: pl.DataFrame) -> list[str]:
    rows = []
    for ph in PHASES:
        g = f.filter(pl.col("phase") == ph).with_columns(c=pl.col("n_flags").clip(0, 4))
        cells = []
        for k in range(5):
            sub = g.filter(pl.col("c") == k)
            r = bs.rate(sub["y"], sub["lead_drug"].to_list())
            cells.append(rate_text({"per_100": r[0], "low": r[1], "high": r[2], "n": r[3]}))
        rows.append([ph, *cells])
    return table(["phase", "0", "1", "2", "3", "4+"], rows)


# ---------------------------------------------------------------- thresholds


def thresholds_section(f: pl.DataFrame) -> list[str]:
    early = bs.pre_2015_ids()
    recomputed = bs.recompute_thresholds(f.select("nct_id", *bs.INPUT_COLUMNS), early)
    rows = [
        [name, f"{getattr(bs, name)!r}", f"{recomputed[name]!r}", why]
        for name, why in (
            ("N_TISSUES_CUT", "more than this many of 51 tissues = broad"),
            ("VITAL_CUT", "vital-organ max expression above this"),
            ("QED_CUT", "drug-likeness at or below this"),
        )
    ]
    return [
        "## Thresholds",
        "",
        (
            "Fixed from trials that finished before 2015 and have a mapped target under today's links "
            "(top third for the two expression flags, bottom third for drug-likeness). The constants in "
            "`biology_score.py` are copied from the pre-declared plan; the right-hand column recomputes "
            "them from the data now, and a test checks the two agree. Method: the thresholds are Polars "
            "`quantile` (nearest rank) of the pre-2015 trials that have a value (2/3 for the "
            "expression flags, 1/3 for drug-likeness). The logistic model's tertile cuts are "
            "`numpy.quantile` (linear) of its fitted scores on the same trials. The two methods "
            "differ; both are part of the declared recipe and neither was changed. Discrete "
            "columns tie at a cut, so the groups are only roughly thirds."
        ),
        "",
        *table(["constant", "in code", "recomputed now", "meaning"], rows),
    ]


# ---------------------------------------------------------------- crosses and adjustments


def adjustment_section(c: pl.DataFrame, facts: dict) -> list[str]:
    rows = []
    for name, extra in ADJUSTMENTS.items():
        fc, tc = flag_stats(c, extra), thirds_stats(c, extra)
        facts[f"flag {name}"], facts[f"third {name}"] = fc, tc
        rows.append([name, fc.text(), tc.text()])
    return [
        "## How much is just disease area?",
        "",
        (
            "Compared among trials that share phase (and the other named things), 2015 to 2020, "
            "today's links. Per 100 trials."
        ),
        "",
        *table(["adjusted for", "2+ flags minus 0", "top minus bottom third"], rows),
        (
            f"About {100 * (1 - facts['flag phase + area'].estimate / facts['flag phase only'].estimate):.0f}% "
            f"of the flag-count gap is disease area (oncology trials carry more flags and "
            f"succeed less often): it goes from {facts['flag phase only'].estimate:+.1f} to "
            f"{facts['flag phase + area'].estimate:+.1f} when area is held fixed. The logistic gap goes "
            f"from {facts['third phase only'].estimate:+.1f} to "
            f"{facts['third phase + area'].estimate:+.1f}. After the area adjustment the flag-count "
            f"interval {_zero_word(facts['flag phase + area'])}; the logistic interval "
            f"{_zero_word(facts['third phase + area'])}."
        ),
        "",
    ]


def independence_section(c: pl.DataFrame) -> list[str]:
    out = [
        "## Is it just precedent or genetic support in disguise?",
        "",
        (
            "2015 to 2020, today's links, trials where the other flag is known. 'Phase only' pools "
            "over phase; 'phase + other' also pools over the other flag. Per 100 trials."
        ),
        "",
    ]
    for column, label in (
        (
            "precedent_strict",
            (
                "strict precedent (another molecule on the same target had "
                "already started a Phase 3 trial in the same disease)"
            ),
        ),
        ("supported", "genetic support when the trial started"),
    ):
        g = c.filter(pl.col(column).is_not_null())
        other = g[column].cast(pl.Int8).to_numpy()
        strata = {"other": other}
        flags_alone = flag_stats(g)
        flags_adj = bs.flag_contrast(g["y"], g["n_flags"], g["phase"], g["lead_drug"], strata)
        logit_alone = thirds_stats(g)
        logit_adj = bs.thirds_contrast(
            g["y"], g["p_bio"], cuts_of(g), g["phase"], g["lead_drug"], strata
        )
        group = g["group"].to_numpy()
        yes_alone = bs.mh_difference(
            g["y"], other == 1, np.ones(g.height, bool), g["phase"], g["lead_drug"]
        )
        yes_adj = bs.mh_difference(
            g["y"],
            other == 1,
            np.ones(g.height, bool),
            g["phase"],
            g["lead_drug"],
            {"group": group},
        )
        out += [
            f"**{label}** (n = {g.height}; {int(other.sum())} yes)",
            "",
            *table(
                ["contrast", "phase only", "phase + other flag (or biology group)"],
                [
                    ["biology 2+ flags minus 0", flags_alone.text(), flags_adj.text()],
                    ["biology top minus bottom third", logit_alone.text(), logit_adj.text()],
                    ["the other flag, yes minus no", yes_alone.text(), yes_adj.text()],
                ],
            ),
        ]
        share = ", ".join(
            f"{k} flags: {g.filter(pl.col('group') == k)[column].cast(pl.Float64).mean():.2f}"
            for k in bs.GROUPS
        )
        out += [f"Share of trials with the other flag, by biology group: {share}.", ""]
    out.append(
        "Reading: the biology gap barely moves when precedent or genetic support is held fixed, "
        "and theirs barely moves when biology is held fixed. They measure different things."
    )
    return [*out, ""]


# ---------------------------------------------------------------- per-feature table


def contrast_list(f: pl.DataFrame) -> list[tuple[str, str | None, str]]:
    """(column, level, kind) for each single-feature contrast, as in `science.report`."""
    out: list[tuple[str, str | None, str]] = [(c, None, "num") for c in bs.NUMERIC]
    out += [(c, None, "bool") for c in (*bs.BOOLEAN, "supported")]
    for c in bs.CATEGORICAL:
        counts = f[c].drop_nulls().value_counts().filter(pl.col("count") >= MIN_LEVEL)
        out += [(c, v, "cat") for v in sorted(counts[c].to_list())]
    return out


def exposed(g: pl.DataFrame, column: str, level, kind: str, cuts) -> pl.DataFrame:
    known = g.filter(pl.col(column).is_not_null())
    if kind == "cat":
        return known.with_columns(e=pl.col(column) == level)
    if kind == "bool":
        return known.with_columns(e=pl.col(column).cast(pl.Boolean))
    low, high = cuts
    return known.filter((pl.col(column) > high) | (pl.col(column) <= low)).with_columns(
        e=pl.col(column) > high
    )


def mh(g: pl.DataFrame, strata: list[str], n: int = GRID_RESAMPLES) -> tuple[float, float, float]:
    """Mantel-Haenszel exposed minus not, per 100, from a frame with `e` and `y`."""
    n1 = int(g["e"].sum())
    if n1 < bs.MIN_GROUP or g.height - n1 < bs.MIN_GROUP:
        return (np.nan, np.nan, np.nan)
    d, lo, hi = adj.adjusted(
        g, "e", "y", strata, "lead_drug", measure="difference", n_resamples=n, seed=bs.SEED
    )
    return (100 * d, 100 * lo, 100 * hi)


FULL = ["phase", "sponsor_class", "therapeutic_area", "modality"]
VARIANTS = [
    ("phase only", {"strata": ["phase"]}),
    ("phase + area", {"strata": ["phase", "therapeutic_area"]}),
    ("full", {"strata": FULL}),
    ("P2 only", {"strata": ["phase"], "phase": "phase2"}),
    ("P3 only", {"strata": ["phase"], "phase": "phase3"}),
    ("single target", {"strata": ["phase"], "single": True}),
    ("single, full", {"strata": FULL, "single": True}),
    ("2020 links", {"strata": ["phase"], "version": "2020"}),
    ("2020 names", {"strata": ["phase"], "version": "2020_names"}),
    ("explore", {"strata": ["phase"], "period": "explore"}),
    ("confirm", {"strata": ["phase"], "period": "confirm"}),
    ("confirm, single", {"strata": ["phase"], "period": "confirm", "single": True}),
]


def _r1(x: float) -> float:
    """Round to one decimal as printed, with no negative zero."""
    return round(float(x), 1) + 0.0


def excludes_zero(t: tuple[float, float, float]) -> bool:
    """True only if the interval still excludes zero at the printed precision (one decimal)."""
    return _r1(t[1]) > 0 or _r1(t[2]) < 0


def cell_text(t: tuple[float, float, float]) -> str:
    if np.isnan(t[0]):
        return "–"
    s = f"{_r1(t[0]):+.1f} ({_r1(t[1]):+.1f}, {_r1(t[2]):+.1f})"
    return f"**{s}**" if excludes_zero(t) else s


def feature_grid(frames: dict[str, pl.DataFrame]) -> tuple[list[str], list[dict]]:
    """Every single-feature contrast in the twelve variants. Cut points for a numeric feature are
    the explore-period tertiles of the same population and link version."""
    contrasts = contrast_list(frames["today"])
    lines = [
        "| feature | n (top/yes, bottom/no) | "
        + " | ".join(v for v, _ in VARIANTS)
        + " | explore to confirm: same sign | CI excl. 0 |",
        "|" + "---|" * (len(VARIANTS) + 3),
    ]
    summary = []
    for column, level, kind in contrasts:
        cells, estimates, n_text = [], [], ""
        for name, spec in VARIANTS:
            f = frames[spec.get("version", "today")]
            if spec.get("single"):
                f = f.filter(pl.col("n_target_genes") == 1)
            col = "supported" if column == "supported" else column
            if f[col].is_null().all():
                cells.append("–")
                estimates.append(None)
                continue
            cuts = None
            if kind == "num":
                x = f.filter(pl.col("period") == "explore")[col].drop_nulls()
                cuts = (x.quantile(1 / 3), x.quantile(2 / 3))
            g = exposed(f, col, level, kind, cuts)
            if "period" in spec:
                g = g.filter(pl.col("period") == spec["period"])
            if "phase" in spec:
                g = g.filter(pl.col("phase") == spec["phase"])
            result = mh(g, spec["strata"])
            if name == "phase only":
                n1 = int(g["e"].sum())
                n_text = f"{n1}/{g.height - n1}"
            cells.append(cell_text(result))
            estimates.append(None if np.isnan(result[0]) else result)
        valid = [t for t in estimates if t is not None]
        if not valid or estimates[0] is None:
            continue
        names = [v for v, _ in VARIANTS]
        early, late = estimates[names.index("explore")], estimates[names.index("confirm")]
        agrees = (
            None if early is None or late is None else bool(np.sign(early[0]) == np.sign(late[0]))
        )
        excludes = sum(excludes_zero(t) for t in valid)
        name = "genetic support at start" if column == "supported" else f"`{column}`"
        name += f" = {level}" if level else ""
        lines.append(
            f"| {name} | {n_text} | "
            + " | ".join(cells)
            + f" | {'n/a' if agrees is None else ('yes' if agrees else 'no')} | {excludes} |"
        )
        summary.append(
            {
                "feature": name,
                "base": estimates[0][0],
                "agrees": agrees,
                "excl": excludes,
            }
        )
    return lines, summary


def features_section(frames: dict[str, pl.DataFrame]) -> list[str]:
    lines, summary = feature_grid(frames)
    c = confirm(frames["today"])
    qed_flag = bs.mh_difference(
        c["y"], c["f_lowqed"], np.ones(c.height, bool), c["phase"], c["lead_drug"]
    )
    both = [s_ for s_ in summary if s_["agrees"] is not None]
    n_agree = sum(s_["agrees"] for s_ in both)
    nothing = [s for s in summary if s["excl"] == 0]
    robust = sorted((s for s in summary if s["excl"] >= 9), key=lambda s: -s["excl"])
    out = [
        "## Each biology column on its own",
        "",
        (
            "Difference in successes per 100 trials, top third minus bottom third of the column "
            "(cut points from trials before 2015), or yes minus no, or one category against the rest. "
            "Pooled within the named groups; interval in brackets; **bold** = interval excludes zero. "
            "Today's links unless a column says otherwise. 'full' = phase, sponsor, area and modality. "
            "An interval counts as excluding zero (and is bold) only if it still does at the printed "
            "one-decimal precision. 'Explore to confirm: same sign' compares the 'explore' column with "
            "the 'confirm' column (the pooled 'phase only' column covers both periods and is not "
            "used for it). Last column: in how many of the variants the interval excludes zero. The "
            "intervals are bootstrapped over lead drugs; the columns are not independent tests."
        ),
        "",
        *lines,
        "",
        f"Explore to confirm, {n_agree} of {len(both)} columns keep their sign.",
        "",
        "**Most robust** (interval excludes zero in at least 9 of the variants): "
        + (", ".join(f"{s['feature']} ({s['excl']})" for s in robust) or "none")
        + ".",
        "",
        "## Genuinely nothing",
        "",
        (
            "Columns whose interval excludes zero in none of the variants above. Not 'proved absent': "
            "the intervals are wide. But nothing here, in any slice, distinguishes them from zero:"
        ),
        "",
        *(f"- {s['feature']} (phase-only {s['base']:+.1f})" for s in nothing),
        "",
        (
            "The low-drug-likeness flag is not on this list as a column (see `chem_qed`), but as a "
            f"flag it did little: with it on, the pooled gap (2015-2020, phase only) is "
            f"{qed_flag.text()}. It stays in the score because the score was "
            "declared with it."
        ),
        "",
    ]
    return out


# ---------------------------------------------------------------- sign agreement (permutations)


def _mh_many(e, mask, strata_onehot, outcomes):
    """MH risk difference for every column of `outcomes`. e: 1/0, nan = not compared."""
    use = mask & ~np.isnan(e)
    s, eu = strata_onehot[use], e[use]
    n1, n0 = s.T @ eu, s.T @ (1 - eu)
    total = n1 + n0
    k1 = (s * eu[:, None]).T @ outcomes[use]
    k0 = (s * (1 - eu)[:, None]).T @ outcomes[use]
    ok = (n1 > 0) & (n0 > 0)
    safe = np.where(total > 0, total, 1)
    num = np.where(ok[:, None], (k1 * n0[:, None] - k0 * n1[:, None]) / safe[:, None], 0).sum(0)
    den = np.where(ok, n1 * n0 / safe, 0).sum()
    return num / den if den > 0 else np.full(outcomes.shape[1], np.nan)


def sign_agreement(f: pl.DataFrame, facts: dict) -> list[str]:
    """Explore-to-confirm sign agreement across all single-feature contrasts, against a
    permutation null that keeps the correlation between features (outcomes shuffled within
    phase x period, or phase x area x period)."""
    rng = np.random.default_rng(7)
    contrasts = contrast_list(f)
    early = f.filter(pl.col("period") == "explore")
    period = (f["period"] == "confirm").to_numpy()
    y = f["y"].to_numpy().astype(float)
    phase = f["phase"].to_numpy().astype(str)
    area = f["therapeutic_area"].fill_null("NA").to_numpy().astype(str)

    def expo(column, level, kind):
        x = f[column]
        e = np.full(f.height, np.nan)
        if kind in ("cat", "bool"):
            known = x.is_not_null().to_numpy()
            values = x.to_numpy()[known]
            e[known] = (values == level).astype(float) if kind == "cat" else values.astype(float)
        else:
            xe = early[column].drop_nulls()
            low, high = xe.quantile(1 / 3), xe.quantile(2 / 3)
            v = x.cast(pl.Float64).to_numpy()
            e[v > high] = 1.0
            e[v <= low] = 0.0
        return e

    def onehot(keys):
        u, inv = np.unique(keys, return_inverse=True)
        return np.eye(len(u))[inv]

    def permuted(block_keys):
        out = np.empty((len(y), PERMUTATIONS + 1))
        out[:, 0] = y
        blocks = [np.flatnonzero(block_keys == b) for b in np.unique(block_keys)]
        for j in range(1, PERMUTATIONS + 1):
            col = y.copy()
            for idx in blocks:
                col[idx] = y[rng.permutation(idx)]
            out[:, j] = col
        return out

    exps = [expo(*c) for c in contrasts]
    new = period & new_lead_mask(f)
    p = np.char.add(phase, period.astype(str))
    lines = []
    for label, strata_keys, block in (
        ("phase only", phase, p),
        (
            "phase + area",
            np.char.add(phase, area),
            np.char.add(np.char.add(phase, area), period.astype(str)),
        ),
    ):
        s = onehot(strata_keys)
        shuffled = permuted(block)
        ex = np.array([_mh_many(e, ~period, s, shuffled) for e in exps])
        co = np.array([_mh_many(e, period, s, shuffled) for e in exps])
        agree = (np.sign(ex) == np.sign(co)).sum(0)
        obs, null = agree[0], agree[1:]
        facts[f"agree {label}"] = (int(obs), len(contrasts), int(np.percentile(null, 97.5)))
        lines.append(
            f"- **{label}**: {obs} of {len(contrasts)} contrasts keep their sign from explore to "
            f"confirm. If outcomes were shuffled (within {'phase' if label == 'phase only' else 'phase and area'} "
            f"and period, {PERMUTATIONS} draws, keeping the correlation between columns) the count "
            f"would be {null.mean():.1f} on average, 95% range {np.percentile(null, 2.5):.0f} to "
            f"{np.percentile(null, 97.5):.0f}; P(null >= observed) = {(null >= obs).mean():.4f}."
        )
        co_new = np.array([_mh_many(e, new, s, shuffled) for e in exps])
        ag_new = (np.sign(ex) == np.sign(co_new)).sum(0)
        ok = ~np.isnan(co_new[:, 0])
        facts[f"agree new {label}"] = (
            int(ag_new[0]),
            int(ok.sum()),
            float((ag_new[1:] >= ag_new[0]).mean()),
        )
        lines.append(
            f"  - Confirm restricted to the {int(new.sum())} trials whose lead drug is in no "
            f"pre-2015 trial: **{ag_new[0]} of {int(ok.sum())}** keep their sign; shuffled average "
            f"{ag_new[1:].mean():.1f}, 95% range {np.percentile(ag_new[1:], 2.5):.0f} to "
            f"{np.percentile(ag_new[1:], 97.5):.0f}; P = {(ag_new[1:] >= ag_new[0]).mean():.4f}."
        )
        if label == "phase only":
            group = f["group"].to_numpy()
            e = np.where(group == "2+", 1.0, np.where(group == "0", 0.0, np.nan))
            d = _mh_many(e, period, s, shuffled)
            lines.append(
                f"- Primary (2+ minus 0 flags, 2015 to 2020, phase only): {100 * d[0]:+.1f} per 100; "
                f"permutation P (one-sided, null <= observed) = {(d[1:] <= d[0]).mean():.4f}, "
                f"two-sided {(np.abs(d[1:]) >= abs(d[0])).mean():.4f}."
            )
    return [
        "## Is the pattern across columns more than luck?",
        "",
        (
            "With 39 contrasts, some would keep their sign by chance; the shuffled-outcome count says "
            "how many. **Caution:** the shuffle moves outcomes between individual trials, so it "
            "ignores that trials of one lead drug share biology and tend to share outcomes. That "
            "makes the shuffled counts too tightly spread, so these P values are optimistic. The "
            "lead-drug bootstrap intervals in the tables are the numbers to trust; this section "
            "is a rough guide."
        ),
        "",
        *lines,
        "",
    ]


# ---------------------------------------------------------------- how little AUROC


def _codes(f: pl.DataFrame) -> np.ndarray:
    return metrics.cluster_codes(np.asarray(f["lead_drug"].to_list(), dtype=object), f.height)


def _cluster_auroc(y, x, codes, n: int = 1000, seed: int = bs.SEED):
    """AUROC with a lead-drug cluster-bootstrap interval."""
    if len(set(y)) < 2:
        return (np.nan, np.nan, np.nan)
    point = roc_auc_score(y, x)
    members = [np.flatnonzero(codes == i) for i in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if len(set(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], x[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return (point, float(low), float(high))


def auc_text(t) -> str:
    return f"{t[0]:.3f} ({t[1]:.3f} to {t[2]:.3f})"


def declared_extras(c: pl.DataFrame) -> list[str]:
    """Declared alongside the pooled statistic: the 2+ minus 0 gap within each phase, and the AUROC
    of minus the flag count (and of the logistic score) within each phase. Also the declared 7-flag
    secondary (the six flags plus 'no genetic support at start', unweighted)."""
    rows = []
    for ph in PHASES:
        g = c.filter(pl.col("phase") == ph)
        y = g["y"].to_numpy()
        codes = _codes(g)
        gap = flag_stats(g)
        a6 = _cluster_auroc(y, -g["n_flags"].to_numpy().astype(float), codes)
        nogen = (g["supported"] == False).fill_null(False).cast(pl.Int64)
        a7 = _cluster_auroc(y, -(g["n_flags"] + nogen).to_numpy().astype(float), codes)
        al = _cluster_auroc(y, g["p_bio"].to_numpy(), codes)
        rows.append([ph, gap.text(), auc_text(a6), auc_text(a7), auc_text(al)])
    return [
        "## Within each phase, and the declared 7-flag secondary",
        "",
        (
            "2015-2020, today's links. Gap = 2+ flags minus 0 flags, per 100 trials. AUROC is the "
            "chance a trial that succeeded outranks one that did not (0.5 = coin flip); for the "
            "flags the score is minus the flag count, so fewer flags ranks higher. Intervals "
            "bootstrap lead drugs. 7 flags = the six plus 'no genetic support at start' (declared "
            "as a secondary; genetic support is deliberately not in the primary score)."
        ),
        "",
        *table(
            [
                "phase",
                "2+ minus 0",
                "AUROC, minus 6 flags",
                "AUROC, minus 7 flags",
                "AUROC, logistic",
            ],
            rows,
        ),
    ]


def _spread(y: np.ndarray, s: np.ndarray, jitter: np.ndarray) -> float:
    """Top third minus bottom third of the predictions, per 100 (ties broken by a fixed jitter)."""
    order = np.argsort(s + jitter[: len(s)])
    n = len(s) // 3
    return 100 * float(y[order[-n:]].mean() - y[order[:n]].mean())


def increment_section(f: pl.DataFrame) -> list[str]:
    """The declared increment: the flag count and the logistic score added to a phase-only model and
    to the registration model. Fitted on pre-2015 trials, scored on 2015-2020 trials, paired
    cluster bootstrap (same resamples for both models)."""
    train = f.filter(pl.col("period") == "explore")
    c = confirm(f)
    train = train.with_columns(
        n_flags_f=pl.col("n_flags").cast(pl.Float64),
        logit_bio=(pl.col("p_bio") / (1 - pl.col("p_bio"))).log(),
    )
    c = c.with_columns(
        n_flags_f=pl.col("n_flags").cast(pl.Float64),
        logit_bio=(pl.col("p_bio") / (1 - pl.col("p_bio"))).log(),
    )
    y = c["y"].to_numpy().astype(float)
    phase = c["phase"].to_numpy()
    members = metrics.cluster_members(np.asarray(c["lead_drug"].to_list(), dtype=object), c.height)
    rng = np.random.default_rng(bs.SEED)
    picks = [
        np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        for _ in range(1000)
    ]
    jitter = np.random.default_rng(1).uniform(0, 1e-9, c.height)
    registration = bl.FEATURES[bl.REGISTRATION]
    phase_only = bl.Features(categorical=("phase",), boolean=(), numeric=())

    def fit_predict(feat):
        pipe = bl.build_pipeline(feat).fit(train, train["y"].to_numpy().astype(bool))
        return pipe.predict_proba(c)[:, 1]

    def within(score, ph):
        m = phase == ph
        return _spread(y[m], score[m], jitter[m])

    rows = []
    for base_name, base in (("phase only", phase_only), ("registration", registration)):
        sb = fit_predict(base)
        for add_name, extra in (
            ("", None),
            (" + flag count", ("n_flags_f",)),
            (" + logistic biology score", ("logit_bio",)),
        ):
            feat = (
                base
                if extra is None
                else bl.Features(base.categorical, base.boolean, base.numeric + extra)
            )
            sc = sb if extra is None else fit_predict(feat)
            auc = roc_auc_score(y, sc)
            spreads = f"{within(sc, 'phase2'):+.1f} / {within(sc, 'phase3'):+.1f}"
            if extra is None:
                rows.append([base_name, f"{auc:.3f}", spreads, "", ""])
                continue
            d_auc, d_p2, d_p3 = [], [], []
            for idx in picks:
                yy = y[idx]
                d_auc.append(roc_auc_score(yy, sc[idx]) - roc_auc_score(yy, sb[idx]))
                for store, ph in ((d_p2, "phase2"), (d_p3, "phase3")):
                    m = phase[idx] == ph
                    store.append(
                        _spread(yy[m], sc[idx][m], jitter[idx][m])
                        - _spread(yy[m], sb[idx][m], jitter[idx][m])
                    )

            def ci(v):
                lo, hi = np.percentile(v, [2.5, 97.5])
                return lo, hi

            lo, hi = ci(d_auc)
            change = f"{auc - roc_auc_score(y, sb):+.3f} ({lo:+.3f} to {hi:+.3f})"
            sp = []
            for ph, d in (("phase2", d_p2), ("phase3", d_p3)):
                lo2, hi2 = ci(d)
                sp.append(f"{within(sc, ph) - within(sb, ph):+.1f} ({lo2:+.1f} to {hi2:+.1f})")
            rows.append([f"{base_name}{add_name}", f"{auc:.3f}", spreads, change, " / ".join(sp)])
    return [
        "## What it adds to models without biology (the measured increment)",
        "",
        (
            "The flag count, then the logistic score, added to a phase-only model and to the "
            "registration model (`baselines`), all fitted on pre-2015 trials and scored on "
            "2015-2020 trials with a mapped target, today's links. 'Spread within phase' = successes "
            "per 100 in the third of trials the model ranks highest minus the lowest third, in "
            "Phase 2 / Phase 3. Change columns are paired differences against the model without "
            "biology, bootstrapped over lead drugs. The logistic score for the pre-2015 rows is "
            "in-sample (it was fitted on them), so its added value on the training rows is "
            "optimistic; the 2015-2020 scoring is out of sample. A phase-only model gives every trial "
            "in a phase the same prediction, so its spread within a phase is only the tie-break "
            "(a fixed tiny jitter) and means nothing."
        ),
        "",
        *table(
            [
                "model",
                "AUROC",
                "spread within phase 2 / 3",
                "change in AUROC",
                "change in spread, phase 2 / phase 3",
            ],
            rows,
        ),
    ]


def auroc_section(f: pl.DataFrame) -> list[str]:
    """A SIMULATION of best cases: what an effect of a given size and reach could add to AUROC.

    Not a measurement of the biology score (see `increment_section` for that). The effect sizes
    and shares are measured on the 2015-2020 trials (phase-pooled difference, share of trials with
    the flag); the base predictions are the registration model fitted on pre-2015 trials, shifted
    so their average equals the PRE-2015 success rate (the confirm-period rate is not used). The
    effect is then added with its size known exactly and outcomes are drawn from the result.
    """
    train = f.filter(pl.col("period") == "explore")
    c = confirm(f)
    pipe = bl.build_pipeline(bl.FEATURES[bl.REGISTRATION]).fit(train, train["y"].to_numpy())
    base = pipe.predict_proba(c)[:, 1]
    p = float(train["y"].mean())
    logit = np.log(base / (1 - base))
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() < p:
            lo = mid
        else:
            hi = mid
    base = 1 / (1 + np.exp(-(logit + mid)))
    rng = np.random.default_rng(3)

    def measured(column: str, values=None):
        g = c if values is None else c.filter(pl.col(column).is_not_null())
        exposed_ = (g[column] == True) if values is None else g[column].cast(pl.Boolean)
        r = bs.mh_difference(g["y"], exposed_, np.ones(g.height, bool), g["phase"], g["lead_drug"])
        return float(exposed_.mean()), r.estimate

    spec = [
        ("activator", *measured("f_activator")),
        ("intracellular target", *measured("f_intracellular")),
        ("common-essential gene", *measured("f_essential")),
        ("broad expression", *measured("f_broad")),
        ("genetic support (for comparison)", *measured("supported", values=True)),
    ]
    two = c.filter(pl.col("group") != "1")
    share_two = float((two["group"] == "2+").mean())
    spec.append(("2+ flags vs 0 flags (the score)", share_two, flag_stats(c).estimate))
    spec.append(("hypothetical: half the trials, 15 points", 0.5, -15.0))  # an assumed example
    rows = []
    for name, q, d in spec:
        d /= 100
        alone = 0.5 + 0.5 * q * (1 - q) * abs(d) / (p * (1 - p))
        gains = []
        for _ in range(300):
            flag = rng.random(len(base)) < q
            pt = np.clip(base + d * (flag - q), 0.01, 0.99)
            y = rng.random(len(base)) < pt
            gains.append(roc_auc_score(y, pt) - roc_auc_score(y, base))
        rows.append([name, f"{q:.2f}", f"{100 * d:+.1f}", f"{alone:.3f}", f"{np.mean(gains):+.4f}"])
    return [
        "## How little AUROC can an effect this size move? (a simulation of best cases)",
        "",
        (
            "This section is a **simulation, not a measurement of the biology score**; the measured "
            "increment is the previous section. AUROC is the chance that a trial that succeeded "
            "ranks above one that did not (0.5 = coin flip). A steady gap of several per 100 on a "
            "flag that applies to a minority of trials is a small move in AUROC. 'Share' and "
            "'difference' are measured on the 2015-2020 trials (the difference is the phase-pooled "
            "gap, flag on minus off; the last row is an assumed example). The simulated column adds "
            "that effect, known exactly, to the registration model's predictions, so it is a best "
            "case: real use would also have to estimate the effect."
        ),
        "",
        *table(
            [
                "effect",
                "share of trials",
                "difference, per 100",
                "AUROC on its own (formula)",
                "AUROC gain on top of registration (simulated)",
            ],
            rows,
        ),
        (
            f"Base rate used for the shift and the formula: p = {p:.2f}, the pre-2015 success rate. "
            "Formula: AUROC = 0.5 + 0.5 q (1 - q) |d| / (p (1 - p)). 300 simulated draws per row."
        ),
        "",
    ]


# ---------------------------------------------------------------- subsets


def new_lead_mask(f: pl.DataFrame, rows: pl.DataFrame | None = None):
    """True for trials whose lead drug is KNOWN and is in no pre-2015 trial. A trial with no lead
    drug is not a "new lead drug" (it is simply unknown), so it is left out of the subset.

    Without `rows`, a numpy mask over `f`; with `rows`, a polars expression-ready Series over them.
    """
    seen = list(f.filter(pl.col("period") == "explore")["lead_drug"].drop_nulls().to_list())
    target = f if rows is None else rows
    mask = target["lead_drug"].is_not_null() & ~target["lead_drug"].is_in(seen).fill_null(False)
    return mask.to_numpy() if rows is None else mask


def subset_lines(f: pl.DataFrame, version: str):
    """Unseen targets and new lead drugs (both against the explore trials)."""
    c = confirm(f)
    explore_ids = f.filter(pl.col("period") == "explore")["nct_id"].to_list()
    unseen = bs.unseen_targets(c["nct_id"].to_list(), explore_ids, version)
    targeted = bs.target_table(version)["nct_id"].unique().to_list()
    subsets = {
        "all 2015-2020 trials with a target": c,
        "unseen targets (no target gene in any pre-2015 trial)": c.filter(
            pl.col("nct_id").is_in(list(unseen))
        ),
        "targets seen before 2015": c.filter(
            pl.col("nct_id").is_in(targeted) & ~pl.col("nct_id").is_in(list(unseen))
        ),
        "lead drug in no pre-2015 trial": c.filter(new_lead_mask(f, c)),
        "single-target small molecules": c.filter(
            (pl.col("n_target_genes") == 1) & (pl.col("modality") == "small molecule")
        ),
        "biologics": c.filter(pl.col("modality") == "biologic"),
        "oncology": c.filter(pl.col("therapeutic_area") == "oncology"),
        "not oncology": c.filter(pl.col("therapeutic_area") != "oncology"),
    }
    rows, contrasts = [], []
    for name, g in subsets.items():
        fc, tc = flag_stats(g), thirds_stats(g)
        rows.append([name, str(g.height), fc.text(), tc.text()])
        contrasts.append((fc, tc))
    return rows, contrasts


def subsets_section(f: pl.DataFrame, version: str) -> list[str]:
    c = confirm(f)
    explore_ids = f.filter(pl.col("period") == "explore")["nct_id"].to_list()
    unseen = bs.unseen_targets(c["nct_id"].to_list(), explore_ids, version)
    new_lead = c.filter(new_lead_mask(f, c))
    no_lead = c["lead_drug"].null_count()
    rows, contrasts = subset_lines(f, version)
    flag_zero = sum(not excludes_zero((a.estimate, a.low, a.high)) for a, _ in contrasts)
    third_zero = sum(not excludes_zero((b.estimate, b.low, b.high)) for _, b in contrasts)
    out = [
        "## Targets and drugs the score has not met",
        "",
        (
            f"Unseen targets: {len(unseen)} of {c.height} trials from 2015 to 2020 have no target gene "
            f"in any pre-2015 trial. New lead drugs: {new_lead.height} of {c.height} have a known lead drug "
            "that is in no pre-2015 trial. Per 100 trials, today's links, pooled over phase."
        ),
        "",
        *table(
            ["subset (2015-2020)", "trials", "2+ flags minus 0", "top minus bottom third"],
            rows,
        ),
        (
            f"{no_lead} trials in the confirm period have no known lead drug; they are left out of "
            "the new-lead-drug subset (unknown is not new) and each is its own cluster in every "
            "interval. Every trial here has a mapped target, so each is in exactly one of the "
            "'unseen' and 'seen' groups. "
            f"Read the intervals, not just the signs: the 2+ flags interval includes zero in "
            f"{flag_zero} of {len(contrasts)} subsets, the top-third interval in {third_zero}. "
            "Replication is weaker on new lead drugs (see the sign-agreement count below)."
        ),
        "",
    ]
    return out


# ---------------------------------------------------------------- the report


def limits_section(facts: dict) -> list[str]:
    kept, total, upper = facts["agree phase only"]
    new_kept, new_total, new_p = facts["agree new phase only"]
    flag_phase, flag_area = facts["flag phase only"], facts["flag phase + area"]
    return [
        "## Honest limits",
        "",
        (
            "- **The features were chosen after seeing all development data.** `dev_science.md` "
            "showed the six flags, and it used 2015 to 2020 as well as before. So the 2015-2020 "
            "'confirm' here is a fresh test of the combined score, its thresholds and its grouping, "
            "not of the choice of features. The real confirmation is the sealed test set."
        ),
        (
            f"- **Replication is weaker on new lead drugs.** Of {total} per-column contrasts, {kept} "
            f"keep their sign from before 2015 to 2015-2020 (shuffled-outcome 97.5th percentile "
            f"{upper}). Restricted to trials whose lead drug is new since 2015 it is {new_kept} of "
            f"{new_total} (P = {new_p:.2f}: a shuffle gives that often). That is weaker support than the "
            "full-sample count."
        ),
        (
            "- **The score is not blind to disease or drug type** (see 'What it is'): four inputs "
            "describe the target in the trial's own disease tissue or cancer lineage, and chemistry "
            "exists only for small molecules."
        ),
        (
            "- **A large share of the flag-count gap is disease area.** Holding area fixed takes 2+ flags "
            f"minus 0 from {flag_phase.estimate:+.1f} to {flag_area.estimate:+.1f} per 100 "
            f"(interval {flag_area.low:+.1f} to {flag_area.high:+.1f})."
        ),
        (
            "- **It adds almost nothing once registration columns are known.** An effect this size "
            "moves AUROC by a few thousandths on top of the registration model in the simulation, "
            "and the measured increment (above) agrees: a few hundredths of AUROC over phase alone, "
            "nothing measurable over the registration model."
        ),
        (
            "- **Today's links are a release from after the trials started.** Everything built from "
            "them is 'known after the start', not known at the start; the 2020 versions are the "
            "closer-in-time check."
        ),
        "- The tables read here (`whathow_<version>`, `precedent`) are the repo's current builds.",
        "",
    ]


def monotonic_note(c: pl.DataFrame) -> str:
    """Where the groups are out of order, with the rates (per 100)."""
    t = bs.by_group(c["y"], c["third"], c["phase"], c["lead_drug"], n_resamples=50)
    g = bs.by_group(c["y"], c["group"], c["phase"], c["lead_drug"], n_resamples=50)

    def v(table_, ph, name):
        return float(
            table_.filter((pl.col("phase") == ph) & (pl.col("group") == name))["per_100"][0]
        )

    notes = []
    for ph in PHASES:
        if v(t, ph, "mid") > v(t, ph, "high"):
            notes.append(
                f"{ph} middle logistic third {v(t, ph, 'mid'):.0f} against top third {v(t, ph, 'high'):.0f}"
            )
        if v(g, ph, "2+") > v(g, ph, "1"):
            notes.append(
                f"{ph} one flag {v(g, ph, '1'):.0f} against two or more {v(g, ph, '2+'):.0f}"
            )
    return "; ".join(notes) or "no group is out of order"


def _zero_word(c: bs.Contrast) -> str:
    return (
        "excludes zero"
        if excludes_zero((c.estimate, c.low, c.high))
        else "includes zero, so it is not distinguishable from no gap"
    )


def report() -> str:
    frames = {}
    for version in VERSIONS:
        try:
            frames[version] = load(version)
        except FileNotFoundError as error:
            print(f"{version}: {error}", file=sys.stderr)
    today = frames["today"]
    c = confirm(today)
    explore = today.filter(pl.col("period") == "explore")
    lines = [
        "# Biology only: what do target and drug biology say about success?",
        "",
        (
            "Development trials only (finished before 2021). Per 100 trials; 95% intervals from a "
            "bootstrap over lead drugs. The recipe (the six flags, their thresholds, the grouping, the "
            "logistic settings and the statistic) was written down before any outcome was read "
            "in the analysis that tested it, but **the six flags themselves were chosen after "
            "looking at all development data**, 2015 to 2020 included (`dev_science.md`). It is "
            "ported unchanged (`biology_score.py`); `TEST_PLAN_2.md` item 8 scores it once on the "
            "sealed test set. Written by `python -m trial_benchmark.biology_report`."
        ),
        "",
        (
            "**What it is.** Two scores, both read within phase. (a) Six risk flags, counted: target "
            "expressed in nearly every tissue; high expression in vital organs; gene essential to "
            "cells; intracellular target; the drug turns its target up; low drug-likeness. Trials "
            "with 2 or more flags against none. (b) A logistic model on biology columns only, top "
            "third against bottom third. No phase, sponsor, area label, design, precedent, genetics or "
            "earlier outcome goes in. **But the score is not blind to disease or drug type.** Four "
            "inputs describe the target in the trial's own disease tissue or cancer lineage "
            "(`expr_disease_tissue`, `expr_disease_tissue_top3`, `depmap_lineage_effect`, "
            "`depmap_lineage_minus_other`; the lineage ones exist almost only for cancer trials), "
            "and the chemistry columns exist only for small molecules (biologics have none, so "
            "the low-drug-likeness flag is off for them). So the score partly encodes drug type "
            "and disease."
        ),
        "",
        (
            "**Design.** Cut points and the model come from trials finishing before 2015 ('explore', "
            f"{explore.height} trials with a mapped target). They are then applied unchanged to trials "
            f"finishing 2015 to 2020 ('confirm', {c.height} trials). Nothing is tuned."
        ),
        "",
        (
            "**Headline (today's links, 2015-2020).** 2+ flags minus 0: "
            f"**{flag_stats(c).text()}**. Logistic top minus bottom third: "
            f"**{thirds_stats(c).text()}**. Holding disease area fixed, the flag gap is "
            f"{flag_stats(c, ['therapeutic_area']).text()}, whose interval {_zero_word(flag_stats(c, ['therapeutic_area']))}, "
            "and the measured increment over registration columns is below. The groups are not "
            "monotonic: " + monotonic_note(c) + " (tables below). The separation is mainly the "
            "bottom third against the rest."
        ),
        "",
        *thresholds_section(today),
        "## The score within phase",
        "",
        *score_tables(c, f"2015-2020, {LINK_TITLE['today']}"),
        "Success per 100 by number of flags (2015-2020):",
        "",
        *count_table(c),
        "Logistic coefficients on the six flags (fitted before 2015; negative = fewer successes): "
        + ", ".join(
            f"{k} {v:+.2f}"
            for k, v in bs.fit_logistic(explore, explore["y"].to_numpy())
            .flag_coefficients()
            .items()
        )
        + ".",
        "",
        *score_tables(
            explore,
            "Reference: before 2015 (where cut points and model came from; the model is in-sample)",
        ),
    ]
    for version in ("2020", "2020_names"):
        if version in frames:
            lines += score_tables(confirm(frames[version]), f"2015-2020, {LINK_TITLE[version]}")
    lines += declared_extras(c)
    lines += subsets_section(today, "today")
    facts: dict = {}
    lines += adjustment_section(c, facts)
    lines += independence_section(c)
    lines += features_section(frames)
    lines += sign_agreement(today, facts)
    lines += increment_section(today)
    lines += auroc_section(today)
    lines += limits_section(facts)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    args.out.write_text(report())
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
