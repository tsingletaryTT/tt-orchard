"""Fetching the model snapshot (orchard/fetch.py).

The supervisor never downloads, and agents run offline, so `bringup` fetches the snapshot first. The
download runs `hf download` as a subprocess with a scrubbed environment: the harness never uses the
operator's Hugging Face token. The subprocess is injected, so no test touches the network."""
import os
import subprocess
from pathlib import Path

import pytest

from orchard import fetch

REV = "2f3de3dd85f379784083b0814d997ab627200f0c"
FILES = ["config.json", "model-00001-of-00002.safetensors", "sub/dir.bin"]


class Runner:
    """Stands in for subprocess.run; records the call and optionally lays down the snapshot."""

    def __init__(self, hf_home, files=FILES, code=0, stderr=""):
        self.calls, self.hf_home, self.files, self.code, self.stderr = [], hf_home, files, code, stderr

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        if self.code == 0:
            snap = fetch.snapshot_dir(self.hf_home, "Cloudflare/clef", REV)
            for f in self.files:
                (snap / f).parent.mkdir(parents=True, exist_ok=True)
                (snap / f).write_text("x")
        return subprocess.CompletedProcess(argv, self.code, stdout="", stderr=self.stderr)


def go(tmp_path, runner=None, which=lambda n: "/usr/bin/hf", env=None):
    runner = runner or Runner(tmp_path)
    path = fetch.fetch_snapshot("Cloudflare/clef", REV, tmp_path, files=FILES, run=runner, which=which,
                                env=env or {})
    return path, runner


def test_the_snapshot_directory_follows_the_hub_cache_layout(tmp_path):
    assert fetch.snapshot_dir(tmp_path, "Cloudflare/clef", REV) == \
        tmp_path / "hub" / "models--Cloudflare--clef" / "snapshots" / REV


def test_the_download_is_pinned_to_the_revision_the_preflight_looked_at(tmp_path):
    path, runner = go(tmp_path)
    argv = runner.calls[0][0]
    assert argv[:3] == ["/usr/bin/hf", "download", "Cloudflare/clef"]
    assert argv[argv.index("--revision") + 1] == REV
    assert argv[argv.index("--cache-dir") + 1] == str(tmp_path / "hub")
    assert path == fetch.snapshot_dir(tmp_path, "Cloudflare/clef", REV)


def test_the_environment_carries_no_token_and_cannot_find_the_token_file(tmp_path):
    _, runner = go(tmp_path, env={"HF_TOKEN": "hf_x", "HUGGING_FACE_HUB_TOKEN": "hf_y", "PATH": "/bin",
                                  "HF_HUB_OFFLINE": "1", "HF_TOKEN_PATH": "/home/u/.cache/huggingface/token"})
    env = runner.calls[0][1]["env"]
    assert "HF_TOKEN" not in env and "HUGGING_FACE_HUB_TOKEN" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert not Path(env["HF_TOKEN_PATH"]).exists()
    assert env["HF_HOME"] == str(tmp_path) and env["PATH"] == "/bin"
    assert "HF_HUB_OFFLINE" not in env              # the download needs the network


def test_the_argument_list_has_no_token_flag(tmp_path):
    _, runner = go(tmp_path)
    assert "--token" not in runner.calls[0][0]


def test_a_missing_hf_executable_is_a_clear_error(tmp_path):
    with pytest.raises(fetch.FetchError, match="hf"):
        go(tmp_path, which=lambda n: None)


def test_a_failed_download_quotes_the_end_of_the_error_output(tmp_path):
    runner = Runner(tmp_path, code=1, stderr="\n".join(f"line {i}" for i in range(40)))
    with pytest.raises(fetch.FetchError) as e:
        go(tmp_path, runner=runner)
    assert "line 39" in str(e.value) and "line 0" not in str(e.value)


def test_a_snapshot_missing_a_listed_file_is_an_incomplete_download(tmp_path):
    runner = Runner(tmp_path, files=FILES[:2])
    with pytest.raises(fetch.FetchError, match="sub/dir.bin"):
        go(tmp_path, runner=runner)


def test_an_already_complete_snapshot_is_not_downloaded_again(tmp_path):
    snap = fetch.snapshot_dir(tmp_path, "Cloudflare/clef", REV)
    for f in FILES:
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        (snap / f).write_text("x")
    runner = Runner(tmp_path)
    path, _ = go(tmp_path, runner=runner)
    assert path == snap and runner.calls == []


def test_a_partial_snapshot_is_completed_by_running_the_download_again(tmp_path):
    snap = fetch.snapshot_dir(tmp_path, "Cloudflare/clef", REV)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("x")
    path, runner = go(tmp_path)
    assert len(runner.calls) == 1 and (path / "sub" / "dir.bin").exists()
