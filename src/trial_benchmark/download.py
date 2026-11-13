"""Download raw trial records from the ClinicalTrials.gov v2 API.

Pulls a deliberately wide superset (every interventional Phase 2 or Phase 3
study, any status, any intervention type) and stores the records untouched.
Eligibility, labels and the split are applied later, locally, so every
exclusion can be counted.

The API serves only the *current* version of each record, and its index is
refreshed about once a day. A pull must therefore finish inside one snapshot:
resuming after the snapshot has changed is refused, because page tokens would
then walk a different index and silently skip or repeat studies, and a pull
that crossed a refresh without stopping is rejected when it is read. A rebuild on
a later day gives a slightly different dataset; the snapshot time is in the
manifest.

    uv run python -m trial_benchmark.download            # full pull, resumable
    uv run python -m trial_benchmark.download --max-pages 2
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

API = "https://clinicaltrials.gov/api/v2"
QUERY = "AREA[StudyType]INTERVENTIONAL AND (AREA[Phase]PHASE2 OR AREA[Phase]PHASE3)"
PAGE_SIZE = 200
PAUSE_SECONDS = 1.0
USER_AGENT = "trial-outcome-benchmark/0.1 (research; python-urllib)"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_WAIT_SECONDS = 300.0
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "data" / "raw" / "ctgov"


def get_json(path: str, params: dict | None = None, attempts: int = 6) -> dict:
    """GET a JSON document, retrying only failures that can plausibly pass next time."""
    url = f"{API}/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        wait = 5.0 * 2**attempt
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            with error:
                if error.code not in RETRY_STATUSES or attempt == attempts - 1:
                    raise RuntimeError(f"HTTP {error.code} from {url}") from error
                retry_after = error.headers.get("Retry-After", "")
            if retry_after.isdigit():
                wait = max(wait, float(retry_after))
        except (OSError, http.client.HTTPException, ValueError):
            # Dropped connections, timeouts, and bodies cut off mid-JSON or mid-character.
            if attempt == attempts - 1:
                raise
        time.sleep(min(wait, MAX_WAIT_SECONDS))
    raise AssertionError("unreachable")


def _write_atomic(path: Path, write) -> None:
    """Write through a temporary file so a kill mid-write cannot leave a torn file."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _page_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"page_{index:05d}.jsonl.gz"


def _write_page(path: Path, studies: list[dict]) -> None:
    def write(tmp: Path) -> None:
        with gzip.open(tmp, "wt") as handle:
            for study in studies:
                handle.write(json.dumps(study) + "\n")

    _write_atomic(path, write)


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    _write_atomic(
        out_dir / "manifest.json",
        lambda tmp: tmp.write_text(json.dumps(manifest, indent=2)),
    )


def nct_id(study: dict) -> str:
    return study["protocolSection"]["identificationModule"]["nctId"]


def _read_pages(out_dir: Path, pages: int) -> Iterator[dict]:
    # Read by index from the manifest, never by glob: a stray page file left
    # by an interrupted run must not be picked up.
    for index in range(pages):
        with gzip.open(_page_path(out_dir, index), "rt") as handle:
            for line in handle:
                yield json.loads(line)


def problems(manifest: dict) -> list[str]:
    """Reasons not to trust a pull. Empty means complete, counted and from one snapshot."""
    if not manifest["complete"]:
        return ["the pull is not complete"]
    check = manifest.get("verification")
    if check is None:
        return ["the pull was never verified"]
    found = []
    if not check["matches_total_count"]:
        found.append(
            f"{check['unique_studies']} unique studies on disk, "
            f"but the API reported {check['total_count_at_start']}"
        )
    if manifest.get("data_timestamp_at_end") != manifest["data_timestamp"]:
        found.append(
            f"the API snapshot changed during the pull ({manifest['data_timestamp']} -> "
            f"{manifest.get('data_timestamp_at_end')})"
        )
    return found


def iter_studies(out_dir: Path = DEFAULT_OUT) -> Iterator[dict]:
    """Yield each downloaded study once. Refuses a pull that is partial or failed its checks."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    if found := problems(manifest):
        raise RuntimeError(
            f"The pull in {out_dir} cannot be used: {'; '.join(found)}. "
            "Delete the folder and download again."
        )
    seen: set[str] = set()
    for study in _read_pages(out_dir, manifest["pages"]):
        key = nct_id(study)
        if key not in seen:
            seen.add(key)
            yield study


def verify(out_dir: Path, manifest: dict) -> dict:
    """Count what is on disk against what the API said it would send. Makes no network call."""
    ids = [nct_id(study) for study in _read_pages(out_dir, manifest["pages"])]
    unique = len(set(ids))
    return {
        "records_on_disk": len(ids),
        "unique_studies": unique,
        "duplicates": len(ids) - unique,
        "total_count_at_start": manifest.get("total_count"),
        "matches_total_count": unique == manifest.get("total_count"),
    }


def _new_manifest() -> dict:
    return {
        "api": API,
        "query": QUERY,
        "page_size": PAGE_SIZE,
        "data_timestamp": get_json("version")["dataTimestamp"],
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pages": 0,
        "studies": 0,
        "next_page_token": None,
        "complete": False,
    }


def _check_resumable(manifest: dict, out_dir: Path) -> None:
    for key, current in (("query", QUERY), ("page_size", PAGE_SIZE)):
        if manifest[key] != current:
            raise RuntimeError(
                f"{out_dir} holds a partial pull made with a different {key}. "
                "Delete the folder and start again."
            )
    live = get_json("version")["dataTimestamp"]
    if live != manifest["data_timestamp"]:
        raise RuntimeError(
            f"The API snapshot changed ({manifest['data_timestamp']} -> {live}) since this "
            f"pull began, so it cannot be resumed safely. Delete {out_dir} and start again."
        )


def download(out_dir: Path = DEFAULT_OUT, max_pages: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not manifest["complete"]:
            _check_resumable(manifest, out_dir)
    else:
        manifest = _new_manifest()

    pages_this_run = 0
    while not manifest["complete"]:
        if max_pages is not None and pages_this_run >= max_pages:
            break
        params = {"filter.advanced": QUERY, "pageSize": PAGE_SIZE}
        if manifest["pages"] == 0:
            params["countTotal"] = "true"
        if manifest["next_page_token"]:
            params["pageToken"] = manifest["next_page_token"]
        page = get_json("studies", params)
        if "totalCount" in page:
            manifest["total_count"] = page["totalCount"]

        # Page first, manifest second: if the run dies in between, the same
        # page index is fetched again with the same token and overwritten.
        studies = page.get("studies", [])
        _write_page(_page_path(out_dir, manifest["pages"]), studies)
        manifest["pages"] += 1
        manifest["studies"] += len(studies)
        manifest["next_page_token"] = page.get("nextPageToken")
        if not manifest["next_page_token"]:
            # Read the snapshot time again now, while the pull is still in
            # progress, so a refresh during the run is caught.
            manifest["data_timestamp_at_end"] = get_json("version")["dataTimestamp"]
            manifest["complete"] = True
            manifest["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        _save_manifest(out_dir, manifest)

        pages_this_run += 1
        if manifest["pages"] % 25 == 0 or manifest["complete"]:
            total = manifest.get("total_count", "?")
            print(f"{manifest['studies']:>7} / {total} studies", flush=True)
        if not manifest["complete"]:
            time.sleep(PAUSE_SECONDS)

    if manifest["complete"] and "verification" not in manifest:
        manifest["verification"] = verify(out_dir, manifest)
        _save_manifest(out_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()
    manifest = download(args.out, args.max_pages)
    if not manifest["complete"]:
        print(f"{manifest['studies']} studies so far (partial; rerun to resume)")
        return
    check = manifest["verification"]
    print(
        f"{check['unique_studies']} unique studies, {check['duplicates']} duplicates, "
        f"API total at start {check['total_count_at_start']}"
    )
    if found := problems(manifest):
        sys.exit(f"This pull cannot be used: {'; '.join(found)}. Delete {args.out} and rerun.")


if __name__ == "__main__":
    main()
