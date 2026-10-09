# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/settings.py: the Settings view's model choices (which chip layout, which CPU stand-in).

A layout is one of the QuietBox 2 presets that setup writes (config/bringup.qb2-*.toml and
config/tiers.qb2-*.toml): the coder package the run boots on the chips, and the model both chip tiers
name. The tests that matter most prove what must never happen: a saved config that the supervisor's own
loaders refuse, a half-written pair of files, the stage map or comments being rewritten, and a model
name the machine does not have.
"""
import json
import tomllib
from pathlib import Path

import pytest

from orchard import bringup_config, settings, tiers
from orchard.setup_machine import render

CONFIG = Path(settings.__file__).resolve().parent.parent / "config"


def write_config(d: Path, preset: str) -> Path:
    """config/bringup.toml and config/tiers.toml as setup writes them for `preset`."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "bringup.toml").write_text(render((CONFIG / f"bringup.qb2-{preset}.toml").read_text(),
                                           RUNS_ROOT=str(d / "runs"), REFERENCE_PYTHON="/usr/bin/python3",
                                           SKILLS_LINE=""))
    (d / "tiers.toml").write_text((CONFIG / f"tiers.qb2-{preset}.toml").read_text())
    return d / "bringup.toml"


def ollama(*names):
    return lambda: list(names)


def installed(*targets):
    return lambda target: target in targets


def test_the_layouts_are_the_presets_setup_ships():
    found = {p["name"]: p for p in settings.presets(CONFIG)}
    assert set(found) == {"coder-next", "27b"}
    cn, big = found["coder-next"], found["27b"]
    assert cn["coder"]["target"] == "raahemnabeel/qwen3-coder-next-blackhole" and cn["coder"]["chips"] == 2
    assert cn["chip_model"] == "Qwen/Qwen3-Coder-Next"
    assert big["coder"]["target"] == "mando2222/qwen3.8-27b-dflash2-p300x2-q4kv" and big["coder"]["chips"] == 4
    assert big["chip_model"] == "Qwen/Qwen3.8-27B"
    assert all(p["note"] for p in found.values())


def test_the_current_settings_name_the_layout_and_each_roles_model(tmp_path):
    cfg = write_config(tmp_path, "coder-next")
    cur = settings.current(cfg, CONFIG, cpu_models=ollama("qwen3-coder:30b", "llama3.2:3b"),
                           is_installed=installed("raahemnabeel/qwen3-coder-next-blackhole"))
    assert cur["layout"] == "coder-next"
    roles = {r["role"]: r for r in cur["roles"]}
    assert roles["grafter"]["model"] == "Qwen/Qwen3-Coder-Next" and roles["grafter"]["tier"] == "small"
    assert roles["head grower"]["model"] == "Qwen/Qwen3-Coder-Next" and roles["head grower"]["tier"] == "large"
    assert roles["seasonal hand"]["model"] == "qwen3-coder:30b" and roles["seasonal hand"]["tier"] == "cpu"
    assert cur["cpu_models"] == ["qwen3-coder:30b", "llama3.2:3b"]
    layouts = {p["name"]: p for p in cur["layouts"]}
    assert layouts["coder-next"]["installed"] is True and layouts["27b"]["installed"] is False


def test_switching_to_the_27b_layout_writes_what_the_preset_says_and_nothing_else(tmp_path):
    cfg = write_config(tmp_path, "coder-next")
    tiers_before = (tmp_path / "tiers.toml").read_text()
    stage_map = tiers_before[tiers_before.index("[stages.0]"):]
    out = settings.apply(cfg, CONFIG, layout="27b", cpu_model="llama3.2:3b",
                         cpu_models=ollama("qwen3-coder:30b", "llama3.2:3b"),
                         is_installed=installed("mando2222/qwen3.8-27b-dflash2-p300x2-q4kv"))
    b = bringup_config.load(cfg)
    assert (b.coder.target, b.coder.port, b.coder.chips, b.coder.profile) == \
        ("mando2222/qwen3.8-27b-dflash2-p300x2-q4kv", 8000, 4, "batch8-dflash2")
    t = tomllib.loads((tmp_path / "tiers.toml").read_text())["tiers"]
    assert t["large"]["model"] == t["small"]["model"] == "Qwen/Qwen3.8-27B"
    assert t["large"]["endpoint"].endswith(":8000/v1") and t["cpu"]["model"] == "llama3.2:3b"
    tiers.load(tmp_path / "tiers.toml")                                   # the supervisor's loader accepts it
    after = (tmp_path / "tiers.toml").read_text()
    assert after[after.index("[stages.0]"):] == stage_map                 # the stage map is not touched
    assert after.count("#") == tiers_before.count("#")                    # nor any comment
    assert out["layout"] == "27b" and len(out["backups"]) == 2
    assert all(Path(p).is_file() for p in out["backups"])


def test_the_round_trip_back_restores_the_original_files(tmp_path):
    cfg = write_config(tmp_path, "coder-next")
    before = (cfg.read_text(), (tmp_path / "tiers.toml").read_text())
    kw = dict(cpu_models=ollama("qwen3-coder:30b"), is_installed=lambda t: True)
    settings.apply(cfg, CONFIG, layout="27b", cpu_model="qwen3-coder:30b", **kw)
    settings.apply(cfg, CONFIG, layout="coder-next", cpu_model="qwen3-coder:30b", **kw)
    assert (cfg.read_text(), (tmp_path / "tiers.toml").read_text()) == before


@pytest.mark.parametrize("change,match", [
    (dict(layout="nope"), "no layout"),
    (dict(cpu_model="mystery:1b"), "not in ollama"),
    (dict(installed=False), "not installed"),
])
def test_a_choice_the_machine_cannot_serve_is_refused_and_nothing_is_written(tmp_path, change, match):
    cfg = write_config(tmp_path, "coder-next")
    before = (cfg.read_text(), (tmp_path / "tiers.toml").read_text())
    with pytest.raises(settings.SettingsError, match=match):
        settings.apply(cfg, CONFIG, layout=change.get("layout", "27b"),
                       cpu_model=change.get("cpu_model", "qwen3-coder:30b"),
                       cpu_models=ollama("qwen3-coder:30b"),
                       is_installed=lambda t: change.get("installed", True))
    assert (cfg.read_text(), (tmp_path / "tiers.toml").read_text()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bringup.toml", "tiers.toml"]


def test_a_config_the_loaders_refuse_is_never_written(tmp_path, monkeypatch):
    cfg = write_config(tmp_path, "coder-next")
    before = (cfg.read_text(), (tmp_path / "tiers.toml").read_text())

    def refuse(path):
        raise tiers.TierConfigError("planted refusal")
    monkeypatch.setattr(settings.tiers, "load", refuse)
    with pytest.raises(settings.SettingsError, match="planted refusal"):
        settings.apply(cfg, CONFIG, layout="27b", cpu_model="qwen3-coder:30b",
                       cpu_models=ollama("qwen3-coder:30b"), is_installed=lambda t: True)
    assert (cfg.read_text(), (tmp_path / "tiers.toml").read_text()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bringup.toml", "tiers.toml"]


def test_an_unknown_current_coder_shows_as_custom(tmp_path):
    cfg = write_config(tmp_path, "coder-next")
    cfg.write_text(cfg.read_text().replace("raahemnabeel/qwen3-coder-next-blackhole", "someone/other-coder"))
    cur = settings.current(cfg, CONFIG, cpu_models=ollama(), is_installed=lambda t: False)
    assert cur["layout"] is None and cur["coder"]["target"] == "someone/other-coder"


def test_set_values_edits_one_table_and_keeps_the_rest():
    text = '# top\n[a]\nx = 1  # keep me? no: the line is rewritten\ny = "old"\n\n[b]\ny = "other"\n'
    out = settings.set_values(text, "a", {"y": "new", "z": 3})
    assert tomllib.loads(out) == {"a": {"x": 1, "y": "new", "z": 3}, "b": {"y": "other"}}
    assert out.startswith("# top\n[a]\nx = 1")
    with pytest.raises(settings.SettingsError):
        settings.set_values(text, "missing", {"y": 1})


# ---- the page's side: GET and POST /api/settings ------------------------------------------------------

def web_app(tmp_path, preset="coder-next"):
    from orchard import webui
    cfg_path = write_config(tmp_path / "config", preset)
    (tmp_path / "config" / "runs").mkdir()
    return webui.WebApp(runs_root=tmp_path / "config" / "runs", config_path=cfg_path,
                        cfg=bringup_config.load(cfg_path), token="tok", gozer_status=lambda: "",
                        settings_dir=CONFIG, cpu_models=ollama("qwen3-coder:30b", "llama3.2:3b"),
                        is_installed=lambda t: True, hostname="box", home="/home/operator")


def test_the_page_reads_and_saves_settings_with_its_token(tmp_path):
    from test_webui import Served
    app = web_app(tmp_path)
    with Served(app) as s:
        code, _, body = s.request("GET", "/api/settings")
        got = json.loads(body)
        assert code == 200 and got["layout"] == "coder-next" and got["running"] == []
        code, _, body = s.post("/api/settings", {"layout": "27b", "cpu_model": "qwen3-coder:30b"}, token="tok")
        assert code == 200 and "next run" in json.loads(body)["note"]
        assert json.loads(s.request("GET", "/api/settings")[2])["layout"] == "27b"
        assert json.loads(s.request("GET", "/api/meta")[2])["config"]["coder"]["chips"] == 4   # meta follows


def test_saving_needs_the_token_and_a_bad_choice_is_a_400_that_writes_nothing(tmp_path):
    from test_webui import Served
    app = web_app(tmp_path)
    before = app.config_path.read_text()
    with Served(app) as s:
        assert s.post("/api/settings", {"layout": "27b", "cpu_model": "qwen3-coder:30b"}, token="wrong")[0] == 403
        code, _, body = s.post("/api/settings", {"layout": "27b", "cpu_model": "mystery"}, token="tok")
        assert code == 400 and "not in ollama" in json.loads(body)["error"]
    assert app.config_path.read_text() == before


def test_tt_model_list_marks_what_is_installed():
    from types import SimpleNamespace

    from orchard import webui
    out = ("  ✓ raahemnabeel/qwen3-coder-next-blackhole   container  blackhole  image bd0ca927a46b\n"
           "  ✗ mando2222/qwen3.8-27b-dflash2-p300x2-q4kv  container  blackhole  needs 4 chips\n")
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(stdout=out)
    check = webui.TtModelList(run=run, clock=lambda: 0.0)
    assert check("raahemnabeel/qwen3-coder-next-blackhole") is True
    assert check("mando2222/qwen3.8-27b-dflash2-p300x2-q4kv") is False
    assert check("raahemnabeel/qwen3-coder-next") is False          # a prefix is not the package
    assert len(calls) == 1                                           # one tt-model list per minute
