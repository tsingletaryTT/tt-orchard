# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 789: `--fidelity-tokens N`, the operator's way to ask stage 1 for a longer sampled reference.

The supervisor cannot write stage 1's `reference_config.json` (the stage agent does, from the skill), so the value travels as a run fact: the
supervisor records it in run_start, the stage contexts list it, and the reference-gate skill tells the agent to copy it into the config. A run
that does not name the option has the files it had before; a resume keeps the recorded value and refuses another."""
import pytest

from orchard import bringup_config as bc
from orchard import context, full_port, supervisor
from orchard.supervisor import Control, parse
from run_fakes import argv
from test_bringup_config import load
from test_cli import conf, run as cli_run  # noqa: F401  (the `conf` fixture)
from test_supervisor import Rig


def test_check_fidelity_tokens_takes_zero_or_the_cap_range_and_refuses_the_rest():
    assert full_port.check_fidelity_tokens(0) == 0 and full_port.check_fidelity_tokens("4000") == 4000
    assert full_port.check_fidelity_tokens(16) == 16 and full_port.check_fidelity_tokens(full_port.POSITIONS_LIMIT) == full_port.POSITIONS_LIMIT
    for bad in (1, 15, 8193, -1, True, 4000.0, "x", "", None, "1" * 7):
        with pytest.raises(ValueError, match="fidelity"):
            full_port.check_fidelity_tokens(bad)


def test_the_stage_1_limits_and_the_cap_limit_agree():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rg", full_port.TEMPLATE.parent.parent / "reference-gate-templates" / "reference_gate.py")
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    assert (rg.FIDELITY_MIN, rg.FIDELITY_MAX) == (full_port.MIN_FIDELITY_TOKENS, full_port.POSITIONS_LIMIT)


def test_parse_has_the_option_default_none_and_checks_the_range(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).fidelity_tokens is None
    assert parse(a + ["--fidelity-tokens", "4000"]).fidelity_tokens == 4000
    assert parse(a + ["--fidelity-tokens", "0"]).fidelity_tokens == 0
    for bad in ("15", "8193", "-1", "abc", "6.5", ""):
        with pytest.raises(SystemExit):
            parse(a + ["--fidelity-tokens", bad])


def test_the_bringup_config_adds_the_flag_only_when_named(tmp_path):
    cfg = load(tmp_path)
    assert "--fidelity-tokens" not in bc.supervisor_argv(cfg, "Cloudflare/clef")
    out = bc.supervisor_argv(cfg, "Cloudflare/clef", fidelity_tokens=4000)
    assert out[out.index("--fidelity-tokens") + 1] == "4000" and supervisor.parse(out).fidelity_tokens == 4000


def test_the_cli_passes_the_flag_to_the_supervisor_only_when_given_and_checks_the_range(conf, capsys):
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef"])
    assert "--fidelity-tokens" not in rec.argv
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--fidelity-tokens", "4000"])
    assert code == 0 and rec.argv[rec.argv.index("--fidelity-tokens") + 1] == "4000"
    for bad in ("15", "8193", "abc"):
        with pytest.raises(SystemExit):
            cli_run(conf, ["bringup", "Cloudflare/clef", "--fidelity-tokens", bad])
    assert "from 16 to 8192" in capsys.readouterr().err


def test_the_run_facts_list_the_sampled_reference_length_only_when_the_run_names_it(tmp_path):
    assert "fidelity reference tokens" not in context.facts_from({"model": "m"}, tmp_path)
    assert context.facts_from({"model": "m", "fidelity_tokens": 4000}, tmp_path)["fidelity reference tokens"] == 4000


def test_the_skill_tells_the_stage_agent_to_copy_the_fact_into_the_config():
    from pathlib import Path
    text = (Path(full_port.__file__).parent / "skills" / "reference-gate.md").read_text()
    assert '"fidelity_tokens"' in text and "fidelity reference tokens" in text


# ---- the supervisor ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.args.full_port = True
    yield r
    r.close()


def run_start(rig):
    return next(e for e in rig.entries() if e["event"] == "run_start")["data"]


@pytest.fixture
def built(monkeypatch):
    import test_supervisor
    from orchard.supervisor import build
    made = []
    monkeypatch.setattr(test_supervisor, "build", lambda *a, **k: made.append(build(*a, **k)) or made[-1])
    return made


def start(rig, **kw):
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run(**kw)


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_run_with_no_option_records_nothing_new(rig, built):
    start(rig)
    assert "fidelity_tokens" not in run_start(rig) and built[-1].fidelity_tokens == 0


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_the_value_is_recorded_in_run_start_when_named(rig, built):
    rig.args.fidelity_tokens = 4000
    start(rig)
    assert run_start(rig)["fidelity_tokens"] == 4000 and built[-1].fidelity_tokens == 4000


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_zero_is_the_same_as_not_naming_it(rig, built):
    rig.args.fidelity_tokens = 0
    start(rig)
    assert "fidelity_tokens" not in run_start(rig)


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_resume_keeps_the_recorded_value_and_refuses_another(rig, built):
    rig.args.fidelity_tokens = 4000
    start(rig)
    rig.args.fidelity_tokens = None
    start(rig, pid=200)
    assert built[-1].fidelity_tokens == 4000
    rig.args.fidelity_tokens = 4000
    start(rig, pid=201)
    rig.args.fidelity_tokens = 2000
    with pytest.raises(ValueError, match="fidelity reference of"):
        rig.run(pid=202)


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_run_that_started_without_it_refuses_a_resume_that_names_a_length(rig, built):
    start(rig)
    rig.args.fidelity_tokens = 4000
    with pytest.raises(ValueError, match="fidelity reference of"):
        rig.run(pid=200)
    rig.args.fidelity_tokens = 0                              # the run's own value: accepted
    start(rig, pid=201)
