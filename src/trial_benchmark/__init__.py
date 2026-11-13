"""A benchmark for clinical-trial outcome prediction."""

from __future__ import annotations

import argparse
import sys

from trial_benchmark.heldout import (
    HeldoutLocked,
    access_log,
    lock_prereg,
    open_heldout,
    prereg_locked,
    prereg_unchanged,
)

__all__ = [
    "HeldoutLocked",
    "access_log",
    "lock_prereg",
    "open_heldout",
    "prereg_locked",
    "prereg_unchanged",
]


def _status() -> None:
    if not prereg_locked():
        print("Pre-registration: not locked. Held-out data is closed.")
    elif prereg_unchanged():
        print("Pre-registration: LOCKED and unchanged since the tag.")
    else:
        print("Pre-registration: LOCKED, but the file has CHANGED since the tag.")
    log = access_log()
    print(f"Held-out accesses: {len(log)}")
    for entry in log:
        dirty = " (uncommitted changes)" if entry["dirty"] else ""
        print(f"  {entry['at']}  {entry['commit'][:8]}{dirty}  {entry['path']}: {entry['note']}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="trial-benchmark", description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "lock-prereg"],
        help="status (default): show the lock and every held-out access. "
        "lock-prereg: tag the committed pre-registration, opening the held-out data.",
    )
    args = parser.parse_args()
    if args.command == "status":
        _status()
        return
    try:
        commit = lock_prereg()
    except RuntimeError as error:
        sys.exit(str(error))
    print(f"Pre-registration locked at {commit[:8]}. Held-out data is now open.")
