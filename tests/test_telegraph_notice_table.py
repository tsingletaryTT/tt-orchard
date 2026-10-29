"""Every command a notice shows, for every state, on every surface that shows one.

Two rules are checked on the same table, so a new surface or a new state cannot skip them:
  1. Every option that changes what a command does is carried: the directory, the config selectors, the scope, and (on `db migrate`
     commands only) the count of accepted lines. A migrate-only option is never put on a status or restore command.
  2. A state never gets advice it forbids: a ledger that was already moved (lost database, old writer) is never told to migrate, a
     half-finished migration is never told to start with a dry run, and `--dry-run-first` is offered only for an unmigrated ledger.
Temporary directories only."""
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import cli, db_cmd
from orchard import setup_machine as sm
from orchard import telegraph as tg
from orchard import telegraph_notice as tn
from test_cli import conf, run as cli_run                         # noqa: F401  (conf is a fixture)
from test_setup_machine import Machine
from test_spend_cmd import BASE
from test_telegraph_cmd import run as run_tg
from test_telegraph_notice_places import FakeServer, panel_cfg
from test_telegraph_view import app_for

STATES = ["unmigrated", "half-finished", "lost-database", "old-writer"]
MOVED = ("lost-database", "old-writer")


def make_state(d: Path, kind: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    if kind in ("unmigrated", "half-finished", "old-writer"):
        (d / "telegraph.jsonl").write_text(json.dumps({"call_id": "a", "problem": "p", "model": "m", "outcome": "ok"}) + "\n")
    if kind == "half-finished":
        (d / "MIGRATING-TO-SQLITE").write_text(json.dumps({"scope": "team"}))
    if kind in MOVED:
        (d / "MIGRATED-TO-SQLITE").write_text("{}")
    if kind == "old-writer":
        (d / "project.db").write_bytes(b"x")
    return d


def commands_in(text: str) -> list[str]:
    return [c for c in re.findall(r"`(tt-orchard [^`]+)`", text)]


def check(text: str, kind: str, required: list[str], accept: bool):
    found = commands_in(text)
    assert found, f"no command in the output for {kind}:\n{text}"
    for c in found:
        for option in required:
            assert option in c, f"{option!r} missing from {c!r}"
        is_migrate = c.startswith("tt-orchard db migrate")
        assert ("--accept-malformed 2" in c) == (accept and is_migrate), c           # on migrate commands only
        if kind in MOVED:
            assert not is_migrate, f"a moved ledger is told to migrate: {c}"
        if kind == "half-finished":
            assert "--dry-run" not in c, f"a half-finished migration is told to start with a dry run: {c}"
        if kind != "unmigrated":
            assert "--dry-run-first" not in c
    if kind != "unmigrated":
        assert "--dry-run-first" not in text


# ---------------------------------------------------------------------------------------------------- the surfaces

def surface_module(tmp_path, kind):
    d = make_state(tmp_path / "state", kind)
    args = SimpleNamespace(dir=str(d), telegraph_config="/c/t.toml", config="/c/b.toml", scope="team", accept_malformed=2)
    n = tn.telegraph_state_notice(tg.Config(), directory=str(d), environ={}, home=tmp_path, flags=tn.flags_from_args(args),
                                  migrate_flags=tn.migrate_flags_from_args(args))
    return n.text(), [f"--dir {d}", "--telegraph-config /c/t.toml", "--config /c/b.toml", "--scope team"], True


def surface_usage(tmp_path, kind):
    d = make_state(tmp_path / "state", kind)
    t = tmp_path / "tg.toml"
    t.write_text("x = 1\n")
    cfg = tmp_path / "c.toml"
    cfg.write_text("x = 1\n")
    code, text = run_tg(["usage", "--dir", str(d), "--telegraph-config", str(t), "--config", str(cfg)], tmp_path)
    return text, [f"--dir {d}", f"--telegraph-config {t}", f"--config {cfg}"], False


def surface_spend(tmp_path, kind):
    d = make_state(tmp_path / "state", kind)
    cfg = tmp_path / "bringup.toml"
    cfg.write_text(BASE)
    out = io.StringIO()
    cli.main(["spend", str(tmp_path / "nothing"), "--dir", str(d), "--config", str(cfg), "--telegraph-config", str(cfg)],
             env={"HOME": str(tmp_path / "home")}, stdout=out)
    return out.getvalue(), [f"--dir {d}", f"--telegraph-config {cfg}", f"--config {cfg}"], False


def surface_bringup(tmp_path, kind, conf):
    d = make_state(tmp_path / "elsewhere", kind)
    conf.write_text(conf.read_text() + f'[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["m"]\nkey_env = "K"\ndir = "{d}"\n')
    home = tmp_path / "home"
    home.mkdir()
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], env={"HOME": str(home)})
    return out, [f"--dir {d}", f"--config {conf}"], False


def surface_ui(tmp_path, kind, monkeypatch):
    d = make_state(tmp_path / "elsewhere", kind)
    home = tmp_path / "home"
    home.mkdir()
    cfg_tg = tg.Config(enabled=True, endpoint="https://llm.example.org/v1", models=("m",), key_env="K", dir=d)
    monkeypatch.setattr("orchard.webui.WebApp", lambda **kw: SimpleNamespace(hostname="h", toplike_close_all=lambda: None))
    monkeypatch.setattr("orchard.webui.make_server", lambda *a, **k: FakeServer())
    monkeypatch.setattr(cli, "find_config", lambda *a, **k: Path("/x/bringup.toml"))
    monkeypatch.setattr(cli.bringup_config, "load", lambda p: SimpleNamespace(runs_root=Path("/x"), telegraph=cfg_tg))
    out = io.StringIO()
    cli._ui(SimpleNamespace(host=None, lan=False, port=1, config="/c/custom.toml", toplike=None), {"HOME": str(home)}, home, out)
    return out.getvalue(), [f"--dir {d}", "--config /c/custom.toml"], False


def surface_panel(tmp_path, kind):
    d = make_state(tmp_path / "state", kind)
    root = tmp_path / "runs"
    root.mkdir()
    app = app_for(root, panel_cfg(d))
    out = app.telegraph()
    m = out["migration"]
    text = (" ".join(f"`{c}`" for c in m["commands"]) + " " + m["summary"] + " " + " ".join(m["lines"]) + " " + str(out["error"]) + " "
            + out["meter"]["text"] + " " + out["meter"]["title"])
    return text, [f"--dir {d}", f"--config {app.config_path}"], False


def surface_setup(tmp_path, kind):
    m = Machine(tmp_path)
    d = make_state(m.home / "elsewhere", kind)
    (m.checkout / "config" / "bringup.toml").write_text(
        f'runs_root = "runs"\n[coder]\ntarget = "pkg/coder"\nport = 8000\nchips = 4\n'
        f'[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["m"]\nkey_env = "K"\ndir = "{d}"\n')
    step = next(s for s in sm.plan(m.env(), sm.Options()) if s.name == "telegraph ledger")
    return step.detail, [f"--dir {d}"], False


def surface_guided(tmp_path, kind):
    d = make_state(tmp_path / "state", kind)
    out = io.StringIO()
    args = cli._parser().parse_args(["db", "migrate", "telegraph", "--dir", str(d), "--dry-run-first", "--scope", "team",
                                     "--accept-malformed", "2", "--config", str(tmp_path / "c.toml"),
                                     "--telegraph-config", str(tmp_path / "t.toml")])
    db_cmd.main(args, out=out, environ={}, home=tmp_path)                   # no terminal: it refuses or prints the commands to run
    return out.getvalue(), [f"--dir {d}", "--scope team", f"--config {tmp_path / 'c.toml'}",
                            f"--telegraph-config {tmp_path / 't.toml'}"], True


SURFACES = {"module": surface_module, "usage": surface_usage, "spend": surface_spend, "panel": surface_panel,
            "setup": surface_setup, "guided": surface_guided}


@pytest.mark.parametrize("kind", STATES)
@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_every_surface_carries_every_selector_and_never_advises_a_forbidden_step(tmp_path, kind, surface):
    text, required, accept = SURFACES[surface](tmp_path, kind)
    check(text, kind, required, accept)


@pytest.mark.parametrize("kind", STATES)
def test_bringup_carries_every_selector_and_never_advises_a_forbidden_step(tmp_path, conf, kind):
    text, required, accept = surface_bringup(tmp_path, kind, conf)
    check(text, kind, required, accept)


@pytest.mark.parametrize("kind", STATES)
def test_the_ui_log_carries_every_selector_and_never_advises_a_forbidden_step(tmp_path, monkeypatch, kind):
    text, required, accept = surface_ui(tmp_path, kind, monkeypatch)
    check(text, kind, required, accept)
