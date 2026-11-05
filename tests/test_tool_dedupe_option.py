# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `--no-tool-dedupe` option (docs/tool-result-dedupe.md): parsed by the supervisor and by bringup, recorded in run_start only when off, kept by a
resume, refused when a resume would turn it off for a run that started with it on, and passed to the agent's tools."""
import pytest

from orchard import bringup_config
from orchard.supervisor import parse
from run_fakes import argv
from test_supervisor import Rig


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def start_entry(rig):
    return next(e for e in rig.entries() if e["event"] == "run_start")["data"]


def test_the_supervisor_parses_the_option_and_the_default_is_on(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).no_tool_dedupe is False and parse(a + ["--no-tool-dedupe"]).no_tool_dedupe is True


def test_bringup_passes_the_option_only_when_it_is_off(tmp_path):
    from test_bringup_config import load
    cfg = load(tmp_path)
    assert "--no-tool-dedupe" not in bringup_config.supervisor_argv(cfg, "org/model")
    argv_off = bringup_config.supervisor_argv(cfg, "org/model", tool_dedupe=False)
    assert "--no-tool-dedupe" in argv_off and parse(argv_off).no_tool_dedupe is True


def test_a_default_run_records_nothing_about_the_option_and_dedupes(rig):
    rig.run()
    assert "tool_dedupe" not in start_entry(rig)


def test_a_run_started_with_the_option_records_it(rig):
    rig.args.no_tool_dedupe = True
    rig.run()
    assert start_entry(rig)["tool_dedupe"] is False


def built(rig, pid=200):
    """The supervisor `build` hands over for the rig's args on the rig's ledger, not run: what a resume would use."""
    from orchard.ledger import Ledger
    from orchard.supervisor import build
    rig.m.owner_pid = pid
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        return build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=pid), coder=rig.coder_cls(rig.m), versions={"tt_model": "test"},
                     clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers)


def test_a_resume_without_the_option_keeps_the_recorded_off_and_a_resume_that_turns_it_off_is_refused(tmp_path):
    (tmp_path / "off").mkdir()
    off = Rig(tmp_path / "off")
    off.args.no_tool_dedupe = True
    off.args.unattended = False
    off.run()
    off.args.no_tool_dedupe = False
    assert built(off).tool_dedupe is False                       # the recorded off stands; nothing is refused
    off.close()
    (tmp_path / "on").mkdir()
    on = Rig(tmp_path / "on")
    on.args.unattended = False
    on.run()
    assert built(on).tool_dedupe is True
    on.args.no_tool_dedupe = True
    with pytest.raises(ValueError, match="tool-result dedupe"):
        built(on)
    on.close()


def test_the_supervisor_gives_its_tools_the_choice(rig, monkeypatch):
    from orchard import supervisor
    seen = []
    real = supervisor.Tools

    def spy(*a, **kw):
        seen.append(kw.get("dedupe"))
        return real(*a, **kw)
    monkeypatch.setattr(supervisor, "Tools", spy)
    rig.args.no_tool_dedupe = True
    rig.run()
    assert seen and set(seen) == {False}
