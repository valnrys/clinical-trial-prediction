import subprocess

import pytest

from elman_exercise.heldout import (
    HeldoutLocked,
    access_log,
    lock_prereg,
    open_heldout,
    prereg_locked,
    prereg_unchanged,
)


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A small git repo shaped like this one: data ignored, an unfinished pre-registration."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / ".gitignore").write_text("data/\n")
    (tmp_path / "PREREGISTRATION.md").write_text("# Pre-registration\n\nTODO\n")
    heldout = tmp_path / "data" / "heldout"
    heldout.mkdir(parents=True)
    (heldout / "test.parquet").write_bytes(b"labels")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def finish_prereg(repo):
    (repo / "PREREGISTRATION.md").write_text("# Pre-registration\n\nI expect X because Y.\n")
    git(repo, "commit", "-q", "-am", "prereg")


@pytest.fixture
def locked(repo):
    finish_prereg(repo)
    lock_prereg(repo)
    return repo


def test_heldout_is_closed_before_the_prereg_is_locked(repo):
    with pytest.raises(HeldoutLocked):
        open_heldout("test.parquet", "peek", root=repo)
    assert access_log(repo) == []


def test_an_unlocked_prereg_is_not_reported_as_unchanged(repo):
    assert not prereg_unchanged(repo)


def test_cannot_lock_an_unfinished_prereg(repo):
    with pytest.raises(RuntimeError, match="TODO"):
        lock_prereg(repo)
    assert not prereg_locked(repo)


def test_cannot_lock_with_uncommitted_changes(repo):
    (repo / "PREREGISTRATION.md").write_text("# Pre-registration\n\nFinished, not committed.\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        lock_prereg(repo)


def test_cannot_lock_with_uncommitted_code(repo):
    finish_prereg(repo)
    (repo / "split.py").write_text("# changed after the plan was written\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        lock_prereg(repo)


def test_cannot_lock_without_a_prereg_file(repo):
    git(repo, "rm", "-q", "PREREGISTRATION.md")
    git(repo, "commit", "-q", "-m", "remove")
    with pytest.raises(RuntimeError, match="not found"):
        lock_prereg(repo)


def test_cannot_lock_twice(locked):
    with pytest.raises(RuntimeError, match="Already locked"):
        lock_prereg(locked)


def test_heldout_opens_and_logs_once_locked(locked):
    path = open_heldout("test.parquet", "final eval", root=locked)
    assert path.read_bytes() == b"labels"
    (entry,) = access_log(locked)
    assert entry["note"] == "final eval"
    assert entry["path"] == "data/heldout/test.parquet"
    assert len(entry["sha256"]) == 64
    assert entry["commit"] == entry["prereg_commit"]


def test_dirty_flag_ignores_the_log_itself(locked):
    open_heldout("test.parquet", "first", root=locked)
    open_heldout("test.parquet", "second", root=locked)
    assert [entry["dirty"] for entry in access_log(locked)] == [False, False]


def test_dirty_flag_reports_uncommitted_code(locked):
    (locked / "model.py").write_text("# tweak\n")
    open_heldout("test.parquet", "eval with a tweak", root=locked)
    assert access_log(locked)[0]["dirty"] is True


def test_editing_the_prereg_after_locking_closes_the_heldout(locked):
    (locked / "PREREGISTRATION.md").write_text("# Pre-registration\n\nI always expected Z.\n")
    assert not prereg_unchanged(locked)
    with pytest.raises(HeldoutLocked, match="changed"):
        open_heldout("test.parquet", "eval", root=locked)
    git(locked, "commit", "-q", "-am", "rewrite history")
    with pytest.raises(HeldoutLocked, match="changed"):
        open_heldout("test.parquet", "eval", root=locked)
    assert access_log(locked) == []


def test_an_access_needs_a_reason(locked):
    with pytest.raises(ValueError, match="note"):
        open_heldout("test.parquet", "  ", root=locked)


def test_a_missing_file_is_not_logged(locked):
    with pytest.raises(FileNotFoundError):
        open_heldout("typo.parquet", "eval", root=locked)
    assert access_log(locked) == []


@pytest.mark.parametrize("escape", ["../raw/x.parquet", "/etc/passwd"])
def test_paths_outside_the_heldout_folder_are_refused(locked, escape):
    with pytest.raises(ValueError, match="outside"):
        open_heldout(escape, "eval", root=locked)
    assert access_log(locked) == []


def test_an_edit_hidden_from_git_is_still_caught(locked):
    git(locked, "update-index", "--assume-unchanged", "PREREGISTRATION.md")
    (locked / "PREREGISTRATION.md").write_text("# Pre-registration\n\nI always expected Z.\n")
    with pytest.raises(HeldoutLocked, match="changed"):
        open_heldout("test.parquet", "eval", root=locked)
