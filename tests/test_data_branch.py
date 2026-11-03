"""Runs the publish script against a local bare repository."""
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "data_branch.sh"


def script(*args):
    return subprocess.run(["bash", str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def git(remote, *args):
    return subprocess.run(["git", "-C", str(remote), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def remote(tmp_path):
    path = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(path)], check=True)
    return path


def publish(remote, folder, text):
    folder.mkdir(exist_ok=True)
    (folder / "forecast.json").write_text(text)
    return script("publish", remote, folder)


def test_restore_starts_empty_when_the_branch_does_not_exist(remote, tmp_path):
    out = tmp_path / "data"
    assert script("restore", remote, out).returncode == 0
    assert list(out.iterdir()) == []


def test_restore_stops_when_the_remote_is_unreachable(tmp_path):
    out = tmp_path / "data"
    assert script("restore", tmp_path / "missing.git", out).returncode != 0
    assert not out.exists()


def test_publish_then_restore_round_trip(remote, tmp_path):
    assert publish(remote, tmp_path / "first", "{}").returncode == 0
    out = tmp_path / "data"
    assert script("restore", remote, out).returncode == 0
    assert (out / "forecast.json").read_text() == "{}"
    assert not (out / ".git").exists()


def test_each_publish_replaces_the_branch_with_one_commit(remote, tmp_path):
    publish(remote, tmp_path / "data", "1")
    assert publish(remote, tmp_path / "data", "2").returncode == 0
    assert git(remote, "rev-list", "--count", "data") == "1"
    assert git(remote, "show", "data:forecast.json") == "2"


def test_publish_refuses_a_directory_without_a_report(remote, tmp_path):
    empty = tmp_path / "data"
    empty.mkdir()
    assert script("publish", remote, empty).returncode != 0
    assert git(remote, "branch", "--list") == ""
