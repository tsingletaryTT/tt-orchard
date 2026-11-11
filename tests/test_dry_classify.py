"""A dry classification (issue 705): stage 0's measurement of a model against its nearest model, with nothing else started.

The module copies the shipped stage 0 template into a scratch directory, runs it there with a minimal environment and reads the
delta.json it writes. Everything below runs on the fake snapshots of tests/test_delta_triage_template.py (real template, real
`tokenizers` package, headers written by hand). A fake runner stands in for the subprocess where the case is a failure."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

if importlib.util.find_spec("tokenizers") is None:
    pytest.skip("SKIPPED: the `tokenizers` package is not importable, so the template runs in these tests did not run.",
                allow_module_level=True)

from orchard import dry_classify  # noqa: E402
from test_delta_triage_template import BASE_REV, MODEL_REV, make_snapshot, tokenizer_json  # noqa: E402,F401  (a fixture, used by name)


@pytest.fixture
def pair(tmp_path, tokenizer_json):
    """Hemmingway-1's shape: a text-only fine-tune against a vision-language nearest model (a weights-only delta)."""
    model = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False, tokenizer=tokenizer_json)
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    return model, nearest


def run(pair, **kw):
    model, nearest = pair
    return dry_classify.classify(model_id="Altworld/Hemmingway-1", model_snapshot=str(model), nearest_id="Qwen/Qwen3.8-27B",
                                 nearest_snapshot=str(nearest), **kw)


# ---- the result ----------------------------------------------------------------------------------------------------------------------

def test_a_weights_only_delta_reports_its_class_path_and_nearest_model(pair):
    r = run(pair)
    assert r["ok"] is True and r["class"] == "weights-only" and r["path"] == "weights-only"
    assert r["model"].startswith("Altworld/Hemmingway-1") and r["nearest_model"].startswith("Qwen/Qwen3.8-27B")
    assert isinstance(r["reasons"], list) and isinstance(r["differences"], list) and r["differences"]
    assert all(set(d) == {"area", "finding"} for d in r["differences"])


def test_a_changed_decisive_config_value_is_a_full_port_and_the_reason_is_listed(tmp_path, tokenizer_json):
    model = make_snapshot(tmp_path / "new", MODEL_REV, nested=False, vision=False, tokenizer=tokenizer_json,
                          config_change={"num_attention_heads": 8})
    nearest = make_snapshot(tmp_path / "base", BASE_REV, nested=True, vision=True, tokenizer=tokenizer_json)
    r = run((model, nearest))
    assert r["ok"] is True and r["path"] == "full-port"
    assert r["reasons"] and any("num_attention_heads" in x or "head" in x for x in r["reasons"])


def test_the_result_is_plain_json(pair):
    r = run(pair)
    assert json.loads(json.dumps(r)) == r


# ---- nothing else starts, nothing is left behind ------------------------------------------------------------------------------------

def test_the_scratch_directory_is_removed_and_the_snapshots_are_untouched(pair, tmp_path, monkeypatch):
    seen = {}
    real = subprocess.run

    def spy(argv, **kw):
        seen["cwd"] = kw.get("cwd")
        return real(argv, **kw)
    monkeypatch.setattr(dry_classify.subprocess, "run", spy)
    before = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    run(pair)
    assert seen["cwd"] and not Path(seen["cwd"]).exists()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == before


def test_the_subprocess_gets_a_minimal_environment_with_the_hub_offline(pair, monkeypatch):
    seen = {}
    real = subprocess.run

    def spy(argv, **kw):
        seen["env"] = kw.get("env")
        seen["argv"] = argv
        return real(argv, **kw)
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    monkeypatch.setattr(dry_classify.subprocess, "run", spy)
    run(pair)
    assert seen["env"]["HF_HUB_OFFLINE"] == "1" and "HF_TOKEN" not in seen["env"]
    assert seen["argv"][0] == sys.executable and "-I" in seen["argv"]


def test_no_coder_lease_or_container_module_is_touched(pair):
    """The module imports none of the adapters that start a coder, take a lease or run a container."""
    src = Path(dry_classify.__file__).read_text()
    for name in ("gozer", "docker", "coder", "adapter", "supervisor", "ttnn", "ttml"):
        assert f"import {name}" not in src and f"from orchard import {name}" not in src and f"orchard.{name}" not in src.replace("orchard.stages", "")


# ---- refusals ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 3, "", "a\0b", "/no/such/dir"])
def test_a_snapshot_that_is_not_a_directory_is_refused_with_a_reason(pair, bad):
    model, nearest = pair
    r = dry_classify.classify(model_id="m/x", model_snapshot=bad, nearest_id="n/y", nearest_snapshot=str(nearest))
    assert r["ok"] is False and "model snapshot" in r["problem"]
    r = dry_classify.classify(model_id="m/x", model_snapshot=str(model), nearest_id="n/y", nearest_snapshot=bad)
    assert r["ok"] is False and "nearest snapshot" in r["problem"]


@pytest.mark.parametrize("bad", [None, 3, "", "x" * 400, "a\nb"])
def test_a_model_id_that_is_not_a_short_line_is_refused(pair, bad):
    model, nearest = pair
    r = dry_classify.classify(model_id=bad, model_snapshot=str(model), nearest_id="n/y", nearest_snapshot=str(nearest))
    assert r["ok"] is False and "model id" in r["problem"]


def test_a_template_that_exits_non_zero_is_a_problem_with_its_last_stderr_line(pair):
    def runner(argv, **kw):
        return subprocess.CompletedProcess(argv, 2, "", "delta_triage: triage_config.json is missing ['x']\n")
    r = run(pair, runner=runner)
    assert r["ok"] is False and "triage_config.json is missing" in r["problem"]


def test_a_template_that_times_out_is_a_problem(pair):
    def runner(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 1)
    r = run(pair, runner=runner)
    assert r["ok"] is False and "did not finish" in r["problem"]


def test_a_delta_json_that_is_not_an_object_is_a_problem(pair):
    def runner(argv, **kw):
        (Path(kw["cwd"]) / "delta.json").write_text("[1]")
        return subprocess.CompletedProcess(argv, 0, "", "")
    r = run(pair, runner=runner)
    assert r["ok"] is False and "delta.json" in r["problem"]


def test_a_class_outside_the_known_list_is_reported_as_unknown_with_the_value_cut(pair):
    def runner(argv, **kw):
        (Path(kw["cwd"]) / "delta.json").write_text(json.dumps({"class": "x" * 500, "path": "full-port", "model": "m", "nearest_model": "n",
                                                                 "path_reasons": ["r" * 900] * 40, "differences": [{"area": "a", "finding": "f" * 900}] * 40}))
        return subprocess.CompletedProcess(argv, 0, "", "")
    r = run(pair, runner=runner)
    assert r["ok"] is True and r["class"] == "unknown" and len(r["reasons"]) <= dry_classify.MAX_ITEMS
    assert all(len(x) <= dry_classify.MAX_TEXT for x in r["reasons"]) and all(len(d["finding"]) <= dry_classify.MAX_TEXT for d in r["differences"])


# ---- the text ----------------------------------------------------------------------------------------------------------------------

def test_the_text_lists_class_route_nearest_reasons_and_the_command_with_left_bars_only(pair):
    r = run(pair)
    text = dry_classify.render(r, command="python3 -m orchard.supervisor run --model m")
    lines = text.splitlines()
    assert lines[0].startswith("║") and any("class" in l and "weights-only" in l for l in lines)
    assert any("nearest" in l for l in lines) and lines[-1].startswith("command") and "python3 -m orchard.supervisor run" in lines[-1]
    assert "╚══" in lines and not any(l.rstrip().endswith("║") for l in lines)
    assert all(len(l) <= dry_classify.WIDTH for l in lines if l.startswith("║"))


def test_the_text_of_a_problem_says_so_and_still_prints_the_command():
    text = dry_classify.render({"ok": False, "problem": "the model snapshot is not a directory"}, command="cmd")
    assert "not a directory" in text and text.splitlines()[-1] == "command   cmd"


# ---- the commands -------------------------------------------------------------------------------------------------------------------

import io  # noqa: E402

from orchard import cli, fetch, preflight as pf  # noqa: E402
from test_cli import Pipe, Recorder, conf, run as cli_run, signals  # noqa: E402,F401


def test_bringup_dry_run_for_a_model_that_is_not_downloaded_says_so_and_downloads_nothing(conf):
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"])
    assert code == 0 and rec.argv is None and rec.fetched == []
    assert "stage 0" in out and "not classified" in out and "the model is not downloaded" in out
    assert "python3 -m orchard.supervisor run --model Cloudflare/clef" in out


def test_bringup_dry_run_json_prints_one_object_and_only_that(conf):
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--json"])
    d = json.loads(out)
    assert code == 0 and d["ok"] is False and "the model is not downloaded" in d["problem"] and d["command"].startswith("python3 -m orchard.supervisor run")
    assert rec.argv is None and rec.fetched == []


def test_bringup_dry_run_with_local_snapshots_prints_the_classification_through_the_module(conf, pair, monkeypatch):
    model, nearest = pair
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"ok": True, "model": "Cloudflare/clef@r", "nearest_model": "Qwen/Qwen3.8-27B@s", "class": "full-port", "path": "full-port",
                "reasons": ["num_attention_heads differs"], "differences": [{"area": "config", "finding": "heads differ"}]}
    monkeypatch.setattr("orchard.dry_classify.classify", fake)
    real = pf.run_preflight

    def with_base(*a, **kw):
        checks = real(*a, **kw)
        return [pf.Check("base", "ok", "Qwen/Qwen3.8-27B", data={"base": "Qwen/Qwen3.8-27B", "snapshot": str(nearest)}) if c.name == "base" else c
                for c in checks]
    monkeypatch.setattr(pf, "run_preflight", with_base)
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=signals(local_snapshot=lambda m: str(model)))
    assert code == 0 and rec.argv is None and rec.fetched == []
    assert seen["model_snapshot"] == str(model) and seen["nearest_snapshot"] == str(nearest) and seen["nearest_id"] == "Qwen/Qwen3.8-27B"
    assert "class full-port" in out and "num_attention_heads differs" in out and "python3 -m orchard.supervisor run" in out


def test_supervisor_run_dry_run_classifies_the_inputs_and_opens_no_run_directory(pair, tmp_path, capsys):
    from orchard import supervisor
    model, nearest = pair
    run_dir = tmp_path / "no-such-run"
    code = supervisor.main(["run", "--model", "Altworld/Hemmingway-1", "--run-dir", str(run_dir), "--tiers", "x.toml", "--coder-target", "t",
                            "--coder-port", "8011", "--coder-chips", "2", "--input", f"model={model}", "--input", f"base={nearest}",
                            "--dry-run", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert code == 0 and d["ok"] is True and d["class"] == "weights-only" and not run_dir.exists()
    assert "--dry-run" not in d["command"] and "--json" not in d["command"] and "supervisor run" in d["command"]


def test_supervisor_run_dry_run_without_a_base_input_is_refused_with_a_reason(pair, tmp_path, capsys):
    from orchard import supervisor
    model, _nearest = pair
    code = supervisor.main(["run", "--model", "m/x", "--run-dir", str(tmp_path / "r"), "--tiers", "x.toml", "--coder-target", "t",
                            "--coder-port", "8011", "--coder-chips", "2", "--input", f"model={model}", "--dry-run"])
    out = capsys.readouterr().out
    assert code == supervisor.EXIT_REFUSED and "not classified" in out and "nearest" in out


# ---- machine-state blocks do not stop a dry classification (coordinator follow-up on issue 705) -----------------------------------------------

def _with_local_base(monkeypatch, model, nearest, result=None):
    """A preflight that finds the nearest model on disk, and a classifier that records what it was asked."""
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return result or {"ok": True, "model": "Cloudflare/clef@r", "nearest_model": "Qwen/Qwen3.8-27B@s", "class": "weights-only",
                          "path": "weights-only", "reasons": [], "differences": []}
    monkeypatch.setattr("orchard.dry_classify.classify", fake)
    real = pf.run_preflight

    def with_base(*a, **kw):
        return [pf.Check("base", "ok", "Qwen/Qwen3.8-27B", data={"base": "Qwen/Qwen3.8-27B", "snapshot": str(nearest)}) if c.name == "base" else c
                for c in real(*a, **kw)]
    monkeypatch.setattr(pf, "run_preflight", with_base)
    return seen


def test_a_full_disk_and_a_busy_port_do_not_stop_the_classification(conf, pair, monkeypatch):
    model, nearest = pair
    seen = _with_local_base(monkeypatch, model, nearest)
    sig = signals(local_snapshot=lambda m: str(model), free_gb=lambda p: 1.0, port_in_use=lambda p: True)
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=sig)
    assert code == 0 and rec.argv is None and rec.fetched == []
    assert seen["model_snapshot"] == str(model) and "class weights-only" in out
    assert "would block a real run" in out and "disk-full" in out and "coder-unusable" in out
    assert "frost before planting" not in out


def test_json_carries_a_would_block_list_and_the_classification(conf, pair, monkeypatch):
    model, nearest = pair
    _with_local_base(monkeypatch, model, nearest)
    sig = signals(local_snapshot=lambda m: str(model), free_gb=lambda p: 1.0, port_in_use=lambda p: True)
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--json"], sig=sig)
    d = json.loads(out)
    assert code == 0 and d["ok"] is True and d["class"] == "weights-only"
    assert {w["reason"] for w in d["would_block"]} >= {"disk-full", "coder-unusable"}
    assert all(set(w) == {"check", "reason", "detail"} for w in d["would_block"])


def test_a_dry_run_with_no_machine_blocks_has_an_empty_would_block_list(conf, pair, monkeypatch):
    model, nearest = pair
    _with_local_base(monkeypatch, model, nearest)
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--json"], sig=signals(local_snapshot=lambda m: str(model)))
    assert code == 0 and json.loads(out)["would_block"] == [] and "would block" not in out


def test_a_block_that_makes_classification_impossible_still_stops_a_dry_run(conf):
    sig = signals(hub_info=lambda m: (None, "no such repository"))
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=sig)
    assert code == 2 and "frost before planting" in out and "class " not in out and rec.argv is None


def test_a_real_run_still_stops_on_a_machine_block(conf):
    sig = signals(free_gb=lambda p: 1.0)
    code, out, rec = cli_run(conf, ["bringup", "Cloudflare/clef"], sig=sig)
    assert code == 2 and "disk-full" in out and rec.argv is None and "would block a real run" not in out


def test_the_text_lists_each_would_block_line_with_left_bars_only():
    text = dry_classify.render({"ok": True, "model": "m", "nearest_model": "n", "class": "weights-only", "path": "weights-only",
                                "reasons": [], "differences": []}, command="cmd",
                               would_block=[{"check": "disk", "reason": "disk-full", "detail": "needs 80 GB, 75 GB free"}])
    lines = text.splitlines()
    assert any("would block a real run" in l and "disk-full" in l and "75 GB free" in l for l in lines)
    assert not any(l.rstrip().endswith("║") for l in lines)
