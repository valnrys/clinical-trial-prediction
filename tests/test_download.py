import urllib.error

import pytest

from trial_benchmark import download as dl


def study(n: int) -> dict:
    return {"protocolSection": {"identificationModule": {"nctId": f"NCT{n:08d}"}}}


class FakeApi:
    """Serves `pages` of studies in order, the way the real API pages with tokens."""

    def __init__(self, pages: list[list[dict]], timestamp: str = "2026-11-10T10:00:05"):
        self.pages = pages
        self.timestamp = timestamp
        self.total = sum(len(page) for page in pages)
        self.requests: list[dict] = []

    def __call__(self, path: str, params: dict | None = None) -> dict:
        if path == "version":
            return {"dataTimestamp": self.timestamp}
        self.requests.append(params)
        index = int(params.get("pageToken", "0"))
        page = {"studies": self.pages[index]}
        if "countTotal" in params:
            page["totalCount"] = self.total
        if index + 1 < len(self.pages):
            page["nextPageToken"] = str(index + 1)
        return page


@pytest.fixture
def api(monkeypatch):
    fake = FakeApi([[study(1), study(2)], [study(3), study(4)], [study(5)]])
    monkeypatch.setattr(dl, "get_json", fake)
    monkeypatch.setattr(dl, "PAUSE_SECONDS", 0)
    return fake


def ids(out_dir):
    return [dl.nct_id(s) for s in dl.iter_studies(out_dir)]


def test_a_full_pull_gets_every_study_and_verifies(api, tmp_path):
    manifest = dl.download(tmp_path)
    assert manifest["complete"]
    assert manifest["verification"]["unique_studies"] == 5
    assert manifest["verification"]["matches_total_count"]
    assert ids(tmp_path) == [f"NCT{n:08d}" for n in range(1, 6)]


def test_an_interrupted_pull_resumes_where_it_stopped(api, tmp_path):
    partial = dl.download(tmp_path, max_pages=1)
    assert not partial["complete"]
    assert "verification" not in partial
    dl.download(tmp_path)
    assert len(api.requests) == 3
    assert ids(tmp_path) == [f"NCT{n:08d}" for n in range(1, 6)]


def test_a_partial_pull_cannot_be_read(api, tmp_path):
    dl.download(tmp_path, max_pages=1)
    with pytest.raises(RuntimeError, match="not complete"):
        list(dl.iter_studies(tmp_path))


def test_a_crash_after_the_page_write_refetches_that_page(api, tmp_path):
    dl.download(tmp_path, max_pages=1)
    # The page for index 1 reached disk, but the manifest was never updated.
    dl._write_page(dl._page_path(tmp_path, 1), [study(99)])
    dl.download(tmp_path)
    assert ids(tmp_path) == [f"NCT{n:08d}" for n in range(1, 6)]


def test_resume_is_refused_if_the_api_snapshot_changed(api, tmp_path):
    dl.download(tmp_path, max_pages=1)
    api.timestamp = "2026-11-11T10:00:05"
    with pytest.raises(RuntimeError, match="snapshot changed"):
        dl.download(tmp_path)


def test_resume_is_refused_if_the_query_changed(api, tmp_path, monkeypatch):
    dl.download(tmp_path, max_pages=1)
    monkeypatch.setattr(dl, "QUERY", "AREA[Phase]PHASE1")
    with pytest.raises(RuntimeError, match="different query"):
        dl.download(tmp_path)


def test_stray_page_files_are_ignored(api, tmp_path):
    dl.download(tmp_path)
    dl._write_page(dl._page_path(tmp_path, 7), [study(99)])
    assert "NCT00000099" not in ids(tmp_path)


def test_a_failed_write_leaves_the_existing_file_intact(tmp_path):
    target = tmp_path / "manifest.json"
    target.write_text("good")

    def failing_write(tmp):
        tmp.write_text("half")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        dl._write_atomic(target, failing_write)
    assert target.read_text() == "good"


def test_a_pull_that_crossed_a_snapshot_refresh_cannot_be_read(api, tmp_path, monkeypatch):
    dl.download(tmp_path, max_pages=2)
    # The refresh lands after the resume check has passed, before the last page.
    real = api.__call__

    def refresh_then_serve(path, params=None):
        if path == "studies":
            api.timestamp = "2026-11-11T10:00:05"
        return real(path, params)

    monkeypatch.setattr(dl, "get_json", refresh_then_serve)
    manifest = dl.download(tmp_path)
    assert manifest["complete"]
    with pytest.raises(RuntimeError, match="snapshot changed during the pull"):
        list(dl.iter_studies(tmp_path))


def test_a_pull_with_the_wrong_count_cannot_be_read(monkeypatch, tmp_path):
    fake = FakeApi([[study(1), study(2)], [study(2), study(3)]])
    monkeypatch.setattr(dl, "get_json", fake)
    monkeypatch.setattr(dl, "PAUSE_SECONDS", 0)
    check = dl.download(tmp_path)["verification"]
    assert check["duplicates"] == 1
    with pytest.raises(RuntimeError, match="3 unique studies on disk, but the API reported 4"):
        list(dl.iter_studies(tmp_path))


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "error", {}, None)


def flaky_urlopen(failures: list[Exception]):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, *args):
            return b'{"ok": true}'

    def urlopen(request, timeout):
        calls.append(request)
        if len(calls) <= len(failures):
            raise failures[len(calls) - 1]
        return Response()

    return urlopen, calls


def test_a_bad_request_fails_at_once(monkeypatch):
    urlopen, calls = flaky_urlopen([http_error(400)])
    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        dl.get_json("studies")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure",
    [
        http_error(429),
        http_error(503),
        ConnectionResetError(),
        dl.http.client.IncompleteRead(b""),
    ],
)
def test_transient_failures_are_retried(monkeypatch, failure):
    urlopen, calls = flaky_urlopen([failure])
    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda seconds: None)
    assert dl.get_json("studies") == {"ok": True}
    assert len(calls) == 2


def test_retries_stop_after_the_last_attempt_and_back_off(monkeypatch):
    urlopen, calls = flaky_urlopen([http_error(503) for _ in range(3)])
    sleeps = []
    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        dl.get_json("studies", attempts=3)
    assert len(calls) == 3
    assert sleeps == [5.0, 10.0]


def test_retry_after_is_honoured_up_to_a_cap(monkeypatch):
    slow_down = urllib.error.HTTPError("http://x", 429, "slow down", {"Retry-After": "9999"}, None)
    urlopen, _ = flaky_urlopen([slow_down])
    sleeps = []
    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl.time, "sleep", sleeps.append)
    dl.get_json("studies")
    assert sleeps == [dl.MAX_WAIT_SECONDS]
