"""A tripwire on the held-out test labels.

The firm rule is that the pre-registration is sent before any
test-set result is inspected. This module makes the *final evaluation* follow
that rule mechanically:

- `lock_prereg()` tags the commit that holds the finished PREREGISTRATION.md.
- `open_heldout()` refuses to hand over a held-out file until that tag exists
  and the pre-registration is unchanged since it, and it writes every access
  to evals/heldout_log.jsonl.

What it is not: a seal. The raw download contains every trial's outcome, so
nothing here can stop someone computing test-set results another way, and git
tags and local logs can be rewritten. The independent proof of timing is the
pre-registration email. This guards against accident, and gives the report an
honest count of how often the test labels were opened.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PREREG_TAG = "prereg"
PREREG_FILE = "PREREGISTRATION.md"
PLACEHOLDER = "TODO"
LOG_FILE = Path("evals") / "heldout_log.jsonl"
HELDOUT_DIR = Path("data") / "heldout"


class HeldoutLocked(RuntimeError):
    """Raised when held-out data is requested without a valid, locked pre-registration."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def repo_root() -> Path:
    return Path(_git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel"))


def prereg_locked(root: Path | None = None) -> bool:
    root = root or repo_root()
    return bool(_git(root, "tag", "--list", PREREG_TAG))


def prereg_unchanged(root: Path | None = None) -> bool:
    """True if PREREGISTRATION.md, as it is on disk, is identical to the tagged version."""
    root = root or repo_root()
    prereg = root / PREREG_FILE
    if not prereg_locked(root) or not prereg.is_file():
        return False
    # Compare the bytes on disk, not git's view of the file, which can be told to look away.
    tagged = _git(root, "show", f"{PREREG_TAG}:{PREREG_FILE}")
    return prereg.read_text().strip() == tagged


def lock_prereg(root: Path | None = None) -> str:
    """Tag the current commit as the pre-registration. Returns the commit hash."""
    root = root or repo_root()
    if prereg_locked(root):
        raise RuntimeError(f"Already locked: tag '{PREREG_TAG}' exists.")
    prereg = root / PREREG_FILE
    if not prereg.is_file():
        raise RuntimeError(f"{PREREG_FILE} not found.")
    if PLACEHOLDER in prereg.read_text():
        raise RuntimeError(f"{PREREG_FILE} still contains '{PLACEHOLDER}'. Finish it first.")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError(
            "The working tree has uncommitted changes. Commit everything first, so the tag "
            "records the code and the pre-registration together."
        )
    _git(root, "tag", "-a", PREREG_TAG, "-m", "Pre-registration locked")
    return _git(root, "rev-parse", "HEAD")


def open_heldout(relative_path: str, note: str, root: Path | None = None) -> Path:
    """Return the path to a held-out file, and log the access.

    `note` says why the test labels are being opened (e.g. "final eval, baseline v1").
    """
    root = root or repo_root()
    if not prereg_locked(root):
        raise HeldoutLocked(
            "Held-out data is locked until the pre-registration is sent. Finish "
            f"{PREREG_FILE}, commit it, email it, then run `uv run trial-benchmark lock-prereg`."
        )
    if not prereg_unchanged(root):
        raise HeldoutLocked(
            f"{PREREG_FILE} has changed since the '{PREREG_TAG}' tag. Restore it; "
            "deviations from the plan belong in the report."
        )
    if not note.strip():
        raise ValueError("Give a note saying why the held-out data is being opened.")
    heldout = (root / HELDOUT_DIR).resolve()
    path = (heldout / relative_path).resolve()
    if not path.is_relative_to(heldout):
        raise ValueError(f"{relative_path} is outside {HELDOUT_DIR}.")
    if not path.is_file():
        raise FileNotFoundError(path)

    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git(root, "rev-parse", "HEAD"),
        # Uncommitted code changes at the time of access. The log itself is
        # excluded, or every access after the first would count as dirty.
        "dirty": bool(_git(root, "status", "--porcelain", "--", ".", f":(exclude){LOG_FILE}")),
        "prereg_commit": _git(root, "rev-parse", f"{PREREG_TAG}^{{commit}}"),
        "path": str(HELDOUT_DIR / path.relative_to(heldout)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "note": note,
    }
    log = root / LOG_FILE
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return path


def access_log(root: Path | None = None) -> list[dict]:
    log = (root or repo_root()) / LOG_FILE
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
