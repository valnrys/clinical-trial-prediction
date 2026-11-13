"""Download one public file, resumably, and check it against the size and sha256 recorded in code.

Every source module lists its files as `RemoteFile`s: a URL plus the exact size and sha256
of the release it pins. A file that is already on disk is checked, not trusted. A wrong
size or checksum deletes the file and raises, so a changed upstream file is never read
silently. A half-finished download is kept as `<name>.part` and resumed.

No account, key or personal detail is sent: a plain GET with a generic User-Agent.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (generic research client)"
CHUNK = 1 << 20
RETRIES = 8


@dataclass(frozen=True)
class RemoteFile:
    name: str  # file name on disk, inside the source's folder
    url: str
    size: int  # bytes
    sha256: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, partial: Path, size: int) -> None:
    """Append to `partial` until it holds `size` bytes, resuming after a dropped connection."""
    for attempt in range(1, RETRIES + 1):
        try:
            _download_once(url, partial, size)
            return
        except (OSError, http.client.IncompleteRead) as error:
            print(f"    interrupted ({error}); attempt {attempt} of {RETRIES}")
    raise RuntimeError(f"{url}: still incomplete after {RETRIES} attempts; run again to resume.")


def _download_once(url: str, partial: Path, size: int) -> None:
    have = partial.stat().st_size if partial.exists() else 0
    if have > size:
        partial.unlink()
        have = 0
    if have == size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if have:
        request.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        if have and response.status != 206:  # the server ignored the resume request
            have = 0
        with partial.open("ab" if have else "wb") as out:
            while block := response.read(CHUNK):
                out.write(block)
    if partial.stat().st_size < size:  # a stream that ends early does not raise by itself
        raise OSError(f"connection closed after {partial.stat().st_size} of {size} bytes")


def fetch(file: RemoteFile, directory: Path) -> Path:
    """Make sure `directory/file.name` exists and matches the recorded size and sha256."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / file.name
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        print(f"  fetching {file.name} ({file.size / 1e6:.1f} MB)")
        _download(file.url, partial, file.size)
        partial.replace(target)
    found = target.stat().st_size
    if found != file.size or (digest := sha256_of(target)) != file.sha256:
        target.unlink()
        raise RuntimeError(
            f"{file.url}: got {found} bytes"
            + ("" if found != file.size else f" with sha256 {digest}")
            + f", expected {file.size} bytes with sha256 {file.sha256}. Deleted; run again."
        )
    return target


def write_manifest(directory: Path, release: str, licence: str, files: list[RemoteFile]) -> None:
    """A small record of what is in the folder: release, licence, and each file's URL and checksum."""
    manifest = {
        "release": release,
        "licence": licence,
        "checked": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [
            {"name": f.name, "url": f.url, "size": f.size, "sha256": f.sha256} for f in files
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def fetch_all(directory: Path, release: str, licence: str, files: list[RemoteFile]) -> list[Path]:
    """Fetch and check every file of one source, then write its manifest."""
    paths = [fetch(f, directory) for f in files]
    write_manifest(directory, release, licence, files)
    return paths
