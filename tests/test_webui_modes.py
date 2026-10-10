# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The web UI and the two run modes: a mode per bring-up, the lab's chips in the machine panel, the mode in
Settings, a preflight that follows a settings save, and "Check machines" (read-only readiness checks).

The tests that matter most prove what must never happen: a check running one of its fixes, a lab-mode run
started without a lab, and a preflight judging a run by a config the operator has since changed.
"""
import json
import time
from pathlib import Path

import pytest

from orchard import bringup_config, machine_checks, webui
from orchard.setup_machine import FAIL, OK, TODO, Action, Step
from test_settings import CONFIG, ollama, write_config
from test_webui import Served

GOZER_LAB = """grain: chip   (2 boards, 2 chips)
board 000004033192101E  (p150a)
  chip 0  0000:01:00.0  FREE
board 0000040331921021  (p150a)
  chip 1  0000:03:00.0  HELD             orchard:supervisor pid 77
"""
LAB = '\n[lab]\nhost = "node4"\nroot = "/srv/orchard"\n'


def config(tmp_path, *, mode=None, lab=False):
    cfg_path = write_config(tmp_path / "config", "coder-next")
    text = cfg_path.read_text()
    if mode:
        text = f'mode = "{mode}"\n' + text
    if lab:
        text += LAB
    cfg_path.write_text(text)
    (tmp_path / "config" / "runs").mkdir(exist_ok=True)
    return cfg_path


class Spawns:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, *, log_path, cwd):
        self.calls.append(list(argv))
        return 4242


def app_for(cfg_path, **kw):
    cfg = bringup_config.load(cfg_path)
    seen = []

    def preflight_for(c):
        def run(model, base):
            seen.append(c)
            return []
        return run
    kw.setdefault("lab_status", lambda lab: (True, GOZER_LAB, ""))
    app = webui.WebApp(runs_root=cfg.runs_root, config_path=cfg_path, cfg=cfg, token="tok", gozer_status=lambda: "",
                       settings_dir=CONFIG, cpu_models=ollama("qwen3-coder:30b"), is_installed=lambda t: True,
                       hostname="box", home="/home/operator", preflight_for=preflight_for, spawn=Spawns(), **kw)
    return app, seen


# ---- a mode per bring-up --------------------------------------------------------------------------

def test_the_preflight_follows_a_settings_save(tmp_path):
    app, seen = app_for(config(tmp_path))
    app.preflight("org/model")
    app.save_settings("27b", "qwen3-coder:30b")
    app.preflight("org/model")
    assert [c.coder.port for c in seen] == [8001, 8000]


def test_a_bring_up_in_local_mode_from_a_lab_config_carries_no_lab(tmp_path):
    app, seen = app_for(config(tmp_path, mode="lab", lab=True))
    app.preflight("org/model", mode="local")
    assert seen[-1].mode == "local" and seen[-1].lab is None
    app.bringup("org/model", mode="local")
    argv = app.spawn.calls[-1]
    assert argv[argv.index("--mode") + 1] == "local"


def test_lab_mode_without_a_lab_table_is_a_400_and_starts_nothing(tmp_path):
    app, seen = app_for(config(tmp_path))
    with Served(app) as s:
        code, _, body = s.post("/api/bringup", {"model": "org/model", "mode": "lab"}, token="tok")
        assert code == 400 and "[lab]" in json.loads(body)["error"]
        assert s.post("/api/preflight", {"model": "org/model", "mode": "remote"}, token="tok")[0] == 400
    assert app.spawn.calls == []


def test_meta_says_the_configs_mode_and_lab(tmp_path):
    app, _ = app_for(config(tmp_path, mode="lab", lab=True))
    assert app.meta()["config"]["mode"] == "lab" and app.meta()["config"]["lab"] == "node4"


# ---- the lab's chips in the machine panel ----------------------------------------------------------

def _settled(app):
    deadline = time.time() + 5
    while time.time() < deadline:
        m = app.machine()
        if not m["lab"].get("asking"):
            return m
        time.sleep(0.02)
    raise AssertionError("the lab never answered")


def test_the_machine_panel_shows_the_labs_chips_in_lab_mode_and_asks_once_per_interval(tmp_path):
    calls = []

    def lab_status(lab):
        calls.append(lab.host)
        return True, GOZER_LAB, ""
    app, _ = app_for(config(tmp_path, mode="lab", lab=True), lab_status=lab_status)
    first = app.machine()["lab"]
    assert first["asking"] is True and first["boards"] == []          # the first paint never waits for ssh
    m = _settled(app)
    assert m["lab"]["host"] == "node4" and m["lab"]["ok"] is True
    states = [c["state"] for b in m["lab"]["boards"] for c in b["chips"]]
    assert states == ["FREE", "HELD"]
    app.machine()
    assert calls == ["node4"]


def test_a_local_config_has_no_lab_group(tmp_path):
    app, _ = app_for(config(tmp_path, mode="local", lab=True))
    assert "lab" not in app.machine()


def test_a_lab_that_does_not_answer_says_so(tmp_path):
    app, _ = app_for(config(tmp_path, mode="lab", lab=True),
                     lab_status=lambda lab: (False, "", "ssh: connect to host node4: No route to host"))
    app.machine()
    m = _settled(app)["lab"]
    assert m["ok"] is False and "No route" in m["error"] and m["boards"] == []


# ---- the mode in Settings --------------------------------------------------------------------------

def test_settings_shows_and_switches_the_mode(tmp_path):
    cfg_path = config(tmp_path, mode="local", lab=True)
    app, _ = app_for(cfg_path)
    cur = app.settings()
    assert cur["mode"] == "local" and cur["lab"] == "node4"
    app.save_settings("coder-next", "qwen3-coder:30b", mode="lab")
    assert bringup_config.load(cfg_path).mode == "lab"
    assert "[lab]" in cfg_path.read_text() and app.cfg.mode == "lab"


def test_settings_refuses_lab_mode_without_a_lab_table(tmp_path):
    cfg_path = config(tmp_path)
    before = cfg_path.read_text()
    app, _ = app_for(cfg_path)
    with Served(app) as s:
        code, _, body = s.post("/api/settings", {"layout": "coder-next", "cpu_model": "qwen3-coder:30b",
                                                 "mode": "lab"}, token="tok")
    assert code == 400 and cfg_path.read_text() == before


# ---- Check machines ---------------------------------------------------------------------------------

def test_check_machines_runs_in_the_background_one_at_a_time(tmp_path):
    started = []

    def runner(cfg, **kw):
        started.append(kw.get("config_path"))
        time.sleep(0.3)
        return [{"name": "this box", "steps": [{"name": "python", "status": "ok", "detail": "3.12", "fix": []}]}]
    app, _ = app_for(config(tmp_path), checks=machine_checks.ChecksJob(runner))
    with Served(app) as s:
        assert s.post("/api/checks", {}, token="tok")[0] == 200
        assert s.post("/api/checks", {}, token="tok")[0] == 409             # one job at a time
        deadline = time.time() + 5
        while time.time() < deadline and not json.loads(s.request("GET", "/api/checks")[2]).get("finished"):
            time.sleep(0.05)
        got = json.loads(s.request("GET", "/api/checks")[2])
    assert started == [app.config_path] and got["boxes"][0]["steps"][0]["status"] == "ok"


def test_a_check_lists_its_fix_and_never_runs_it(tmp_path):
    """setup's steps carry actions (an install, a clone); Check machines must only describe them."""
    ran = []
    fix = Action("run", (["git", "clone", "https://example.invalid/tt-gozer.git"],))

    def fake_plan(env, opts):
        return [Step("gozer", TODO, "gozer is not installed", [fix]), Step("python", OK, "3.12")]
    boxes = machine_checks.run_checks(bringup_config.load(config(tmp_path)), setup_plan=fake_plan,
                                      env=type("E", (), {"run": staticmethod(lambda *a, **k: ran.append(a) or (0, "", ""))})())
    steps = {s["name"]: s for s in boxes[0]["steps"]}
    assert steps["gozer"]["fix"] == ["git clone https://example.invalid/tt-gozer.git"] and steps["python"]["fix"] == []
    assert not ran
    assert [b["name"] for b in boxes] == ["this box"]                      # no [lab]: one box


def test_with_a_lab_both_boxes_are_checked(tmp_path):
    def fake_plan(env, opts):
        return [Step("python", OK, "3.12")]

    def fake_lab_plan(lab, **kw):
        return [Step("sfpi", FAIL, "the lab has SFPI 7.61.0", [])]
    boxes = machine_checks.run_checks(bringup_config.load(config(tmp_path, mode="local", lab=True)),
                                      setup_plan=fake_plan, lab_plan=fake_lab_plan)
    assert [b["name"] for b in boxes] == ["this box", "lab node4"]
    assert boxes[1]["steps"][0]["status"] == "fail"



def test_the_config_row_reports_the_file_the_page_uses(tmp_path):
    cfg_path = config(tmp_path)

    def fake_plan(env, opts):
        return [Step("config", TODO, "writes bringup.toml for a QuietBox 2", [Action("write", ("/x", ""))])]
    boxes = machine_checks.run_checks(bringup_config.load(cfg_path), setup_plan=fake_plan, config_path=cfg_path)
    row = boxes[0]["steps"][0]
    assert row["status"] == "ok" and str(cfg_path) in row["detail"] and row["fix"] == []
