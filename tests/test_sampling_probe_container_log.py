# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The sampling probe in a container-backed swap check (issue 107, review of PR 109). A container's `docker logs` are copied to server.log only by
`Container.save_log`, which the container script ran after `measure()` finished. The probe reads server.log around each request, so it saw a missing or
stale file and lost the drafter's session lines. The probe now asks the process to refresh the log before it takes each starting offset and after each
request. The fake container here keeps its log on the "docker side" and copies it to the file only when `save_log` is called, as the real one does."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchard import sampling_probe as sp  # noqa: E402
from test_weights_swap_templates import load_swap_template  # noqa: E402

LINE = ("[dflash2-serve] slot 0 session: 4 iters, 18 committed tokens, accept 3.75/7 -> 4.50 tok/iter, 21.3 tok/s in-model (incl. seed); "
        "histogram [0, 2, 0, 0, 0, 0, 1, 1]")


class FakeContainer:
    """Its log grows on the docker side when a request ends; the file changes only on save_log()."""

    def __init__(self, log_path):
        self.log_path, self.docker_side, self.saves = log_path, ["boot line one", "boot line two"], 0

    def request_done(self):
        self.docker_side.append(f"(EngineCore) INFO {LINE}")

    def save_log(self):
        self.saves += 1
        self.log_path.write_text("\n".join(self.docker_side) + "\n", encoding="utf-8")


@pytest.fixture
def sc(monkeypatch):
    mod = load_swap_template()
    state = {}

    def fake_post(port, path, body):
        state["container"].request_done()
        return {"choices": [{"text": "word " * 60, "finish_reason": "length"}], "usage": {"completion_tokens": 96}}

    monkeypatch.setattr(mod, "post_json", fake_post)
    return mod, state


def test_each_probe_request_keeps_its_own_session_line_when_the_log_is_refreshed(sc, tmp_path):
    mod, state = sc
    log = tmp_path / "server.log"
    c = state["container"] = FakeContainer(log)
    rec = mod.run_sampling_probe({"sampling_log_settle_s": 0}, 1, "m", log, refresh=c.save_log)
    for name in sp.ENTRY_NAMES:
        assert rec["requests"][name]["drafter_log"].count("[dflash2-serve]") == 1, (name, rec["requests"][name]["drafter_log"])
    assert sp.summarize(rec)["greedy"]["drafter_accept_rate"] == pytest.approx(15 / 28)


def test_without_a_refresh_a_container_log_is_stale_and_the_lines_are_lost(sc, tmp_path):
    mod, state = sc
    log = tmp_path / "server.log"
    state["container"] = FakeContainer(log)
    rec = mod.run_sampling_probe({"sampling_log_settle_s": 0}, 1, "m", log)
    assert all(rec["requests"][n]["drafter_log"] == "" for n in sp.ENTRY_NAMES)


def test_the_log_is_refreshed_before_each_start_offset_and_after_each_request(sc, tmp_path):
    mod, state = sc
    log = tmp_path / "server.log"
    c = state["container"] = FakeContainer(log)
    mod.run_sampling_probe({"sampling_log_settle_s": 0}, 1, "m", log, refresh=c.save_log)
    assert c.saves == 8            # four requests, a refresh before and after each


def test_a_refresh_that_fails_is_ignored_and_the_probe_still_records_the_requests(sc, tmp_path):
    mod, state = sc
    log = tmp_path / "server.log"
    state["container"] = FakeContainer(log)

    def broken():
        raise OSError("docker is gone")

    rec = mod.run_sampling_probe({"sampling_log_settle_s": 0}, 1, "m", log, refresh=broken)
    assert all(rec["requests"][n]["accepted"] for n in sp.ENTRY_NAMES)


def test_measure_hands_a_container_s_save_log_to_the_probe(sc, tmp_path, monkeypatch):
    mod, state = sc
    log = tmp_path / "server.log"
    c = state["container"] = FakeContainer(log)

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return type("E", (), {"ids": [1]})()

        def decode(self, ids):
            return "x"

    monkeypatch.setattr(mod, "wait_healthy", lambda *a, **k: 0.1)
    monkeypatch.setattr(mod, "complete", lambda *a, **k: "x")
    monkeypatch.setattr(mod, "STAGE_DIR", tmp_path)
    m = mod.measure({"port": 1, "sampling_probe": True, "sampling_log_settle_s": 0}, c, log, Tok(), [1], [1] * 32)
    for name in sp.ENTRY_NAMES:
        assert m["sampling_probe"]["requests"][name]["drafter_log"].count("[dflash2-serve]") == 1


# ---- the real Container.save_log against a fake `docker` whose stdout and stderr interleave over time ---------------------------------

FAKE_DOCKER = """#!/usr/bin/env bash
# `docker logs <cid>`: replay LOG (lines "out<TAB>text" or "err<TAB>text") to the stream each names, in order, like the real CLI.
[ "$1" = "logs" ] || exit 0
while IFS=$'\\t' read -r stream text; do
  if [ "$stream" = "err" ]; then echo "$text" >&2; else echo "$text"; fi
done < "$FAKE_DOCKER_LOG"
"""


@pytest.fixture
def real_container(tmp_path, monkeypatch):
    import importlib.util
    from test_container_template import TEMPLATES
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "docker"
    exe.write_text(FAKE_DOCKER)
    exe.chmod(0o755)
    docker_log = tmp_path / "docker-log.tsv"
    docker_log.write_text("out\tboot one\nerr\tboot warning\n")
    monkeypatch.setenv("PATH", f"{bin_dir}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(docker_log))
    spec = importlib.util.spec_from_file_location("serve_and_compare_container_under_test", TEMPLATES / "serve_and_compare_container.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, docker_log


def test_the_saved_log_keeps_stdout_and_stderr_in_the_order_the_container_wrote_them(real_container, tmp_path):
    mod, docker_log = real_container
    docker_log.write_text("out\ta\nerr\tb\nout\tc\nerr\td\n")
    log = tmp_path / "server.log"
    mod.Container("abc", log).save_log()
    assert log.read_text().splitlines() == ["a", "b", "c", "d"]


def test_a_log_that_grows_in_both_streams_stays_append_stable_for_the_probe(sc, real_container, tmp_path):
    """Each request makes the container write the drafter's session line on stdout, then a long warning on stderr, then another stdout line. With the streams written one
    after the other, new stdout pushed the saved stderr down, so a starting offset taken earlier pointed into the wrong place."""
    mod, state = sc
    cmod, docker_log = real_container
    log = tmp_path / "server.log"
    c = cmod.Container("abc", log)

    def fake_post(port, path, body):
        with open(docker_log, "a", encoding="utf-8") as f:
            f.write(f"out\t(EngineCore) INFO {LINE}\n" + "err\t" + "warning from a library, a long line of text " * 3 + "\n" + "out\trequest answered\n")
        return {"choices": [{"text": "word " * 60, "finish_reason": "length"}], "usage": {"completion_tokens": 96}}

    mod.post_json = fake_post
    rec = mod.run_sampling_probe({"sampling_log_settle_s": 0}, 1, "m", log, refresh=c.save_log)
    for name in sp.ENTRY_NAMES:
        text = rec["requests"][name]["drafter_log"]
        assert text.count("[dflash2-serve]") == 1, (name, text)
    assert sp.summarize(rec)["greedy"]["drafter_accept_rate"] == pytest.approx(15 / 28)


def test_a_plain_popen_has_no_save_log_and_measure_still_works(sc, tmp_path, monkeypatch):
    mod, state = sc
    log = tmp_path / "server.log"
    state["container"] = FakeContainer(log)

    class Plain:
        pass

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return type("E", (), {"ids": [1]})()

        def decode(self, ids):
            return "x"

    monkeypatch.setattr(mod, "wait_healthy", lambda *a, **k: 0.1)
    monkeypatch.setattr(mod, "complete", lambda *a, **k: "x")
    monkeypatch.setattr(mod, "STAGE_DIR", tmp_path)
    m = mod.measure({"port": 1, "sampling_probe": True, "sampling_log_settle_s": 0}, Plain(), log, Tok(), [1], [1] * 32)
    assert all(m["sampling_probe"]["requests"][n]["accepted"] for n in sp.ENTRY_NAMES)
