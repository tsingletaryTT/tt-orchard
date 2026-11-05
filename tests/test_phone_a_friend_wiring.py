# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The path from `bringup.toml` to the supervisor's helper: `bringup` passes `--telegraph-config` when the helper is on, the supervisor's `run`
parser takes it, and `build` makes a `Telegraph` from it (orchard/phone_a_friend.py). Each link is checked on its own, so one that is cut shows
up here and not as a helper that is configured and never called (run 8, 2026-11-04)."""
import pytest

from orchard import bringup_config as bc
from orchard import supervisor
from orchard import telegraph as tg
from test_bringup_config import MINIMAL, load
from test_cli import conf, run  # noqa: F401  (the fixture and the runner of the CLI tests)

ON = '\n[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["small"]\nkey_env = "TG_KEY"\n'
OFF = '\n[telegraph]\nenabled = false\n'


def test_the_argv_names_the_config_file_only_when_the_helper_is_on(tmp_path):
    on = bc.supervisor_argv(load(tmp_path, MINIMAL + ON), "Cloudflare/clef", telegraph_config=tmp_path / "bringup.toml")
    assert on[on.index("--telegraph-config") + 1] == str(tmp_path / "bringup.toml")
    off = bc.supervisor_argv(load(tmp_path, MINIMAL + OFF), "Cloudflare/clef", telegraph_config=tmp_path / "bringup.toml")
    assert "--telegraph-config" not in off
    assert "--telegraph-config" not in bc.supervisor_argv(load(tmp_path, MINIMAL), "Cloudflare/clef", telegraph_config=tmp_path / "x.toml")
    assert "--telegraph-config" not in bc.supervisor_argv(load(tmp_path, MINIMAL + ON), "Cloudflare/clef")        # no path given


def test_the_supervisor_parser_takes_the_option(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path, MINIMAL + ON), "Cloudflare/clef", telegraph_config=tmp_path / "bringup.toml")
    assert supervisor.parse(argv).telegraph_config == str(tmp_path / "bringup.toml")
    assert supervisor.parse(bc.supervisor_argv(load(tmp_path), "Cloudflare/clef")).telegraph_config is None


def test_bringup_hands_the_config_file_to_the_supervisor_when_the_helper_is_on(conf):
    conf.write_text(conf.read_text() + ON)
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"])
    assert rec.argv is not None and rec.argv[rec.argv.index("--telegraph-config") + 1] == str(conf)


def test_bringup_passes_nothing_when_the_helper_is_off(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"])
    assert rec.argv is not None and "--telegraph-config" not in rec.argv


def test_build_makes_a_telegraph_from_the_named_file_and_none_without_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCHARD_RUN_DIR", raising=False)
    assert supervisor._telegraph_from(None) is None
    off = tmp_path / "off.toml"
    off.write_text(OFF)
    assert supervisor._telegraph_from(str(off), environ={}, home=tmp_path) is None            # off in the table: no helper
    on = tmp_path / "on.toml"
    on.write_text(ON + f'dir = "{tmp_path}/state"\n')
    tel = supervisor._telegraph_from(str(on), environ={}, home=tmp_path)
    assert isinstance(tel, tg.Telegraph) and tel.cfg.enabled and str(tel.dir) == f"{tmp_path}/state"


def test_a_file_that_cannot_be_read_refuses_the_start_and_names_the_option(tmp_path):
    with pytest.raises(ValueError, match="--telegraph-config"):
        supervisor._telegraph_from(str(tmp_path / "missing.toml"), environ={}, home=tmp_path)
    bad = tmp_path / "bad.toml"
    bad.write_text("[telegraph\n")
    with pytest.raises(ValueError, match="--telegraph-config"):
        supervisor._telegraph_from(str(bad), environ={}, home=tmp_path)


def test_a_state_directory_that_cannot_be_chosen_refuses_the_start(tmp_path):
    on = tmp_path / "on.toml"
    on.write_text(ON)                                                                           # no `dir` in the table
    with pytest.raises(ValueError, match="agent shell"):
        supervisor._telegraph_from(str(on), environ={"ORCHARD_RUN_DIR": "/run"}, home=tmp_path)
