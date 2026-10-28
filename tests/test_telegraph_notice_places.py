"""Where the Telegraph state notice is shown: `bringup`, `ui` (the startup log and the Telegraph panel), `spend` and `setup`.

Each place shows it once, near the top, only when the directory needs something, and never changes the directory. Every directory is
a temporary one, and HOME is a temporary one, so the real Telegraph state directory is never looked at."""
import io
import json
import os
import re
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import cli
from orchard import telegraph as tg
from orchard import telegraph_migrate as mig
from test_cli import Pipe, conf, run as cli_run, signals            # noqa: F401  (conf is a fixture)
from test_setup_machine import Machine
from orchard import setup_machine as sm
from test_telegraph_view import app_for
from test_spend_cmd import BASE, real_ledger

RECORD = {"call_id": "a1", "problem": "p", "model": "small", "outcome": "ok", "tokens_in": 10, "tokens_out": 5,
          "cost_usd": 0.5, "usage_known": True, "ts": 1.0}
FIRST = "Run `tt-orchard db migrate telegraph --dry-run`"
THEN = "then `tt-orchard db migrate telegraph`"


def old_ledger(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "telegraph.jsonl").write_text(json.dumps(RECORD) + "\n")
    return directory


def snapshot(directory):
    out = {}
    for p in sorted(directory.rglob("*")):
        st = os.lstat(p)
        out[str(p.relative_to(directory))] = (stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime_ns, p.read_bytes() if p.is_file() else None)
    return out


def home_with_old_ledger(tmp_path):
    home = tmp_path / "home"
    return home, old_ledger(home / ".cache" / "tt-orchard" / "telegraph")


# ------------------------------------------------------------------------------------------------ bringup

def test_bringup_shows_the_notice_once_before_the_checks_and_changes_nothing(conf, tmp_path):
    home, d = home_with_old_ledger(tmp_path)
    before = snapshot(d)
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], env={"HOME": str(home)})
    start = "telegraph: Run `tt-orchard db migrate telegraph --dry-run "
    assert code == 0 and out.count(start) == 1 and out.index(start) < out.index("hub")
    assert f"--config {conf}`" in out and "Nothing is deleted" in out       # the commands select the config the run used
    assert snapshot(d) == before


def test_bringup_of_someone_who_never_used_telegraph_prints_no_notice(conf, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], env={"HOME": str(home)})
    assert code == 0 and "telegraph" not in out.lower().replace("tt-orchard bringup", "") and "db migrate" not in out


def test_bringup_does_not_stop_for_the_notice(conf, tmp_path):
    home, d = home_with_old_ledger(tmp_path)
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef"], env={"HOME": str(home)})
    assert rec.argv is not None                              # the run started: the notice warns and never blocks


# ------------------------------------------------------------------------------------------------ ui

class FakeServer:
    stopping = SimpleNamespace(set=lambda: None)

    def serve_forever(self, poll_interval=0.5):
        return None

    def server_close(self):
        return None


def start_ui(monkeypatch, tmp_path, cfg_telegraph=None):
    home, d = home_with_old_ledger(tmp_path)
    monkeypatch.setattr("orchard.webui.WebApp", lambda **kw: SimpleNamespace(hostname="h", toplike_close_all=lambda: None))
    monkeypatch.setattr("orchard.webui.make_server", lambda *a, **k: FakeServer())
    monkeypatch.setattr(cli, "find_config", lambda *a, **k: Path("/x/bringup.toml"))
    monkeypatch.setattr(cli.bringup_config, "load",
                        lambda p: SimpleNamespace(runs_root=Path("/x"), telegraph=cfg_telegraph or tg.Config()))
    out = io.StringIO()
    args = SimpleNamespace(host=None, lan=False, port=1, config=None, toplike=None)
    code = cli._ui(args, {"HOME": str(home)}, home, out)
    return code, out.getvalue(), d


def test_the_ui_startup_log_shows_the_notice_once_after_the_address(monkeypatch, tmp_path):
    code, text, d = start_ui(monkeypatch, tmp_path)
    assert code == 0 and text.count(FIRST) == 1
    assert text.index("tt-orchard ui") < text.index(FIRST) < text.index("Ctrl-C")
    assert snapshot(d) == {"telegraph.jsonl": snapshot(d)["telegraph.jsonl"]}


def test_the_ui_startup_log_has_no_notice_for_a_directory_that_needs_nothing(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("orchard.webui.WebApp", lambda **kw: SimpleNamespace(hostname="h", toplike_close_all=lambda: None))
    monkeypatch.setattr("orchard.webui.make_server", lambda *a, **k: FakeServer())
    monkeypatch.setattr(cli, "find_config", lambda *a, **k: Path("/x/bringup.toml"))
    monkeypatch.setattr(cli.bringup_config, "load", lambda p: SimpleNamespace(runs_root=Path("/x"), telegraph=tg.Config()))
    out = io.StringIO()
    cli._ui(SimpleNamespace(host=None, lan=False, port=1, config=None, toplike=None), {"HOME": str(home)}, home, out)
    assert "db migrate" not in out.getvalue()


# ------------------------------------------------------------------------------------------------ the Telegraph panel

@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


def panel_cfg(directory, enabled=True):
    return tg.load({"enabled": enabled, "endpoint": "https://llm.example.org/v1", "models": ["small"], "key_env": "TG_KEY",
                    "dir": str(directory)}, base=directory.parent)


def test_the_panel_carries_the_notice_with_the_exact_commands_and_the_meter_says_what_to_run(root, tmp_path):
    d = old_ledger(tmp_path / "ledger")
    before = snapshot(d)
    out = app_for(root, panel_cfg(d), environ={"TG_KEY": "k3y-VALUE-0123456789abcdef"}).telegraph()
    m = out["migration"]
    assert m["kind"] == "unmigrated"
    assert m["commands"] == [f"tt-orchard db migrate telegraph --dry-run --dir {d}", f"tt-orchard db migrate telegraph --dir {d}"]
    assert m["summary"].startswith("Run `tt-orchard db migrate telegraph --dry-run --dir ") and any("Nothing is deleted" in line for line in m["lines"])
    assert "db migrate telegraph" in out["meter"]["text"] and "spend unknown" in out["meter"]["text"]
    assert "Run `tt-orchard db migrate telegraph --dry-run" in out["meter"]["title"]
    assert out["error"].startswith("Run `")                  # the first words of the old message now say what to do
    assert snapshot(d) == before


def test_the_panel_hides_the_directory_path_the_way_the_rest_of_the_page_does(root, tmp_path):
    d = old_ledger(tmp_path / "ledger")
    app = app_for(root, panel_cfg(d))
    app.redact = lambda t: str(t).replace(str(tmp_path), "~")
    blob = json.dumps(app.telegraph())
    assert str(tmp_path) not in blob and "~/ledger" in blob


def test_the_panel_has_no_notice_for_a_migrated_directory_and_none_for_telegraph_that_was_never_used(root, tmp_path):
    d = old_ledger(tmp_path / "ledger")
    mig.migrate(d)
    assert app_for(root, panel_cfg(d)).telegraph()["migration"] is None
    assert app_for(root, tg.Config(), environ={}).telegraph()["migration"] is None


def test_the_page_script_draws_the_notice_and_its_commands():
    js = (Path(__file__).resolve().parents[1] / "orchard" / "web" / "app.js").read_text()
    html = (Path(__file__).resolve().parents[1] / "orchard" / "web" / "index.html").read_text()
    assert 'id="tg-migration"' in html and "t.migration" in js and "tg-migration" in js
    assert "commands" in js[js.index("tg-migration"):]


# ------------------------------------------------------------------------------------------------ spend

def run_spend(tmp_path, d, *extra):
    cfg = tmp_path / "bringup.toml"
    cfg.write_text(BASE)
    out = io.StringIO()
    code = cli.main(["spend", str(tmp_path / "nothing"), *extra, "--dir", str(d), "--config", str(cfg), "--telegraph-config", str(cfg)],
                    env={"HOME": str(tmp_path / "home")}, stdout=out)
    return code, out.getvalue()


def test_spend_puts_the_notice_first_and_still_says_the_helper_spend_is_unknown(tmp_path):
    d = old_ledger(tmp_path / "tg")
    before = snapshot(d)
    code, text = run_spend(tmp_path, d)
    lines = text.splitlines()
    assert lines[0].startswith("telegraph: Run `tt-orchard db migrate telegraph --dry-run --dir ") and text.index("db migrate") < text.index("REAL helper spend")
    assert "the helper spend is unknown" in text and "--dir" in lines[0]
    assert snapshot(d) == before


def test_spend_json_keeps_its_two_keys_and_carries_the_notice_inside_the_real_part(tmp_path):
    d = old_ledger(tmp_path / "tg")
    code, text = run_spend(tmp_path, d, "--json")
    data = json.loads(text)
    assert set(data) == {"real_helper_spend", "local_virtual_spend"}
    assert data["real_helper_spend"]["migration"]["kind"] == "unmigrated" and "error" in data["real_helper_spend"]


def test_spend_of_a_good_ledger_prints_no_notice(tmp_path):
    real_ledger(tmp_path / "tg")
    code, text = run_spend(tmp_path, tmp_path / "tg")
    assert "db migrate" not in text and "$0.2500" in text


# ------------------------------------------------------------------------------------------------ setup

def test_setup_lists_the_move_as_a_todo_step_with_the_commands_and_runs_nothing_for_it(tmp_path):
    m = Machine(tmp_path)
    d = old_ledger(m.home / ".cache" / "tt-orchard" / "telegraph")
    before = snapshot(d)
    steps = {s.name: s for s in sm.plan(m.env(), sm.Options())}
    step = steps["telegraph ledger"]
    assert step.status == sm.TODO and step.actions == []
    assert "tt-orchard db migrate telegraph --dry-run" in step.detail and "tt-orchard db migrate telegraph" in step.detail
    ran_before = len(m.ran)
    out = []
    sm.main(["--yes"], env=m.env(), ask=lambda q: True, say=out.append)
    assert not any("migrate" in " ".join(a) for a in m.ran[ran_before:])      # setup never migrates by itself
    assert any("telegraph ledger" in line for line in out) and snapshot(d) == before


def test_setup_has_no_such_step_when_the_directory_needs_nothing(tmp_path):
    m = Machine(tmp_path)
    assert "telegraph ledger" not in {s.name for s in sm.plan(m.env(), sm.Options())}


# ------------------------------------------------------------------------------------------------ commands name the ledger they describe

def test_bringup_commands_name_a_directory_that_the_config_chose(conf, tmp_path):
    chosen = old_ledger(tmp_path / "elsewhere dir")
    conf.write_text(conf.read_text() + f'[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["m"]\nkey_env = "K"\ndir = "{chosen}"\n')
    home = tmp_path / "home"
    home.mkdir()
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], env={"HOME": str(home)})
    assert f"--dir '{chosen}'" in out.splitlines()[1] and f"--config {conf}" in out.splitlines()[1]


def test_the_ui_log_commands_carry_the_selected_config(monkeypatch, tmp_path):
    home, d = home_with_old_ledger(tmp_path)
    monkeypatch.setattr("orchard.webui.WebApp", lambda **kw: SimpleNamespace(hostname="h", toplike_close_all=lambda: None))
    monkeypatch.setattr("orchard.webui.make_server", lambda *a, **k: FakeServer())
    monkeypatch.setattr(cli, "find_config", lambda *a, **k: Path("/x/bringup.toml"))
    monkeypatch.setattr(cli.bringup_config, "load", lambda p: SimpleNamespace(runs_root=Path("/x"), telegraph=tg.Config()))
    out = io.StringIO()
    cli._ui(SimpleNamespace(host=None, lan=False, port=1, config="/c/my config.toml", toplike=None), {"HOME": str(home)}, home, out)
    assert "--config '/c/my config.toml'" in out.getvalue()


def test_the_panel_commands_name_the_directory_redacted_and_quoted_for_a_shell(root, tmp_path):
    d = old_ledger(tmp_path / "my ledger")
    app = app_for(root, panel_cfg(d))
    app.home = str(tmp_path)
    app.redact = lambda t: str(t).replace(str(tmp_path), "~")
    m = app.telegraph()["migration"]
    assert m["commands"][0] == "tt-orchard db migrate telegraph --dry-run --dir ~/'my ledger'"
    assert str(tmp_path) not in json.dumps(m)


def test_the_meter_does_not_tell_a_lost_database_or_an_old_writer_to_migrate(root, tmp_path):
    d = tmp_path / "ledger"
    d.mkdir()
    (d / "MIGRATED-TO-SQLITE").write_text("{}")
    out = app_for(root, panel_cfg(d)).telegraph()
    assert out["migration"]["kind"] == "lost-database" and "db migrate" not in out["meter"]["text"]
    assert "db status" in out["meter"]["title"]
