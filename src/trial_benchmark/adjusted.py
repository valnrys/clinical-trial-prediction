"""Adjusted group comparisons: Mantel-Haenszel estimates with a sponsor-clustered bootstrap.

Two groups ("exposed" and "not exposed") are compared on a yes/no outcome, inside strata
of other variables (phase, sponsor type, era, and so on), and the within-stratum
comparisons are pooled. This answers "would the gap remain among trials that look alike
on those variables?", which a crude comparison cannot.

- `measure="ratio"`: Mantel-Haenszel risk ratio, exposed over not exposed.
- `measure="difference"`: Mantel-Haenszel risk difference (exposed minus not exposed), in
  proportion. Strata with no trial in one of the groups carry no weight.

The interval is a percentile bootstrap that resamples whole clusters (here, lead
sponsors) with replacement, so a sponsor with hundreds of trials counts as one draw, not
hundreds. That is wider than an interval that treats trials as independent, and fairer:
the trials of one sponsor are alike. Trials with no cluster value are each their own
cluster. Everything is fixed by the seed.

The estimate is only as good as the strata. Anything that is not a stratum variable is
not adjusted for.
"""

from __future__ import annotations

import numpy as np
import polars as pl

N_RESAMPLES = 2000
SEED = 20261110


def _cells(
    frame: pl.DataFrame, exposed: str, outcome: str, strata: list[str], cluster: str
) -> np.ndarray:
    """Counts shaped (clusters, strata, 4): exposed n, exposed events, other n, other events."""
    n = frame.height
    if strata:
        keys = frame.select(
            pl.concat_str(
                [pl.col(c).cast(pl.String).fill_null("NA") for c in strata], separator="|"
            )
        )
        stratum = np.unique(keys.to_series().to_numpy(), return_inverse=True)[1]
    else:
        stratum = np.zeros(n, dtype=int)
    names = frame[cluster].to_list()
    labelled = [f"__row{i}" if name is None else str(name) for i, name in enumerate(names)]
    group = np.unique(np.array(labelled), return_inverse=True)[1]
    is_exposed = frame[exposed].to_numpy().astype(bool)
    event = frame[outcome].fill_null(False).to_numpy().astype(float)
    cells = np.zeros((group.max() + 1, stratum.max() + 1, 4))
    column = np.where(is_exposed, 0, 2)
    np.add.at(cells, (group, stratum, column), 1.0)
    np.add.at(cells, (group, stratum, column + 1), event)
    return cells


def _estimate(totals: np.ndarray, measure: str) -> np.ndarray:
    """Pooled estimate for each row of `totals` shaped (..., strata, 4)."""
    n1, k1, n0, k0 = (totals[..., i] for i in range(4))
    size = n1 + n0
    with np.errstate(divide="ignore", invalid="ignore"):
        if measure == "ratio":
            top = np.where(size > 0, k1 * n0 / size, 0.0).sum(axis=-1)
            bottom = np.where(size > 0, k0 * n1 / size, 0.0).sum(axis=-1)
            return np.where(bottom > 0, top / bottom, np.nan)
        both = (n1 > 0) & (n0 > 0)
        top = np.where(both, (k1 * n0 - k0 * n1) / size, 0.0).sum(axis=-1)
        weight = np.where(both, n1 * n0 / size, 0.0).sum(axis=-1)
        return np.where(weight > 0, top / weight, np.nan)


def adjusted(
    frame: pl.DataFrame,
    exposed: str,
    outcome: str,
    strata: list[str],
    cluster: str,
    measure: str = "ratio",
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """(estimate, low, high): pooled over `strata`, with a cluster-bootstrap 95% interval.

    `exposed` and `outcome` are boolean columns. All nan if the frame has no usable trials.
    """
    if measure not in ("ratio", "difference"):
        raise ValueError("measure must be 'ratio' or 'difference'")
    if frame.height == 0:
        return float("nan"), float("nan"), float("nan")
    cells = _cells(frame, exposed, outcome, strata, cluster)
    point = float(_estimate(cells.sum(axis=0), measure))
    rng = np.random.default_rng(seed)
    k = cells.shape[0]
    flat = cells.reshape(k, -1)
    draws = []
    for start in range(0, n_resamples, 250):
        size = min(250, n_resamples - start)
        weights = rng.multinomial(k, np.full(k, 1 / k), size=size).astype(float)
        totals = (weights @ flat).reshape(size, cells.shape[1], 4)
        draws.append(_estimate(totals, measure))
    values = np.concatenate(draws)
    if np.isnan(values).all():
        return point, float("nan"), float("nan")
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return point, float(low), float(high)


def adjusted_text(*args, **kwargs) -> str:
    """The same as `adjusted`, as "1.02 (0.93 to 1.13)" for a ratio."""
    estimate, low, high = adjusted(*args, **kwargs)
    if np.isnan(estimate):
        return "n/a"
    if kwargs.get("measure") == "difference":
        return f"{100 * estimate:+.1f} ({100 * low:+.1f} to {100 * high:+.1f})"
    return f"{estimate:.2f} ({low:.2f} to {high:.2f})"
