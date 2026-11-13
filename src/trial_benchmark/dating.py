"""How each feature is dated: could it have been known when the trial started?

Every feature table lists its columns with one of four labels. The label says
what the value depends on, so a model can be restricted to honest features.

- TIMELESS: a property that does not depend on any trial's result (for
  example, the count of words in a title is fixed by the title itself).
- DATED: restricted to information from before the trial started.
- PROTOCOL: read from the trial's own protocol record, current version.
  Sponsors can edit that record after the trial starts, so these values may
  have been changed later.
- AFTER_START: may include what was learned after the trial started.

The pre-registered baseline itself reads the current protocol record. So
"protocol" features stand on the same footing as the baseline, and honest
models may use them, with that caveat stated here once. "After the start"
features are not allowed in honest models.
"""

TIMELESS = "timeless"  # a property that does not depend on any trial's result
DATED = "dated"  # restricted to information from before the trial started
PROTOCOL = "protocol"  # from the trial's own protocol record, current version; sponsors can edit it
AFTER_START = "after the start"  # may include what was learned after the trial started
ALL = (TIMELESS, DATED, PROTOCOL, AFTER_START)
HONEST = frozenset({TIMELESS, DATED, PROTOCOL})


def check(features: dict[str, str]) -> None:
    """Raise ValueError if any feature carries a label that is not in ALL."""
    unknown = {name: label for name, label in features.items() if label not in ALL}
    if unknown:
        raise ValueError(f"unknown dating labels: {unknown}")
