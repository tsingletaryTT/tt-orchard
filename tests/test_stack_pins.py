# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0's stack freshness record (orchard/stack_pins.py). No network and no real git or gh: the runner and the interpreter probe are
fakes. A checkout is a temporary directory; the fake answers what git and GitHub would say about it."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from orchard import safeio, skillsfresh, stack_pins as sp

NOW = 1_793_000_000.0           # a fixed "now"
DAY = 86400
HEX = lambda c: c * 40          # noqa: E731


def iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeUpstream:
    """A git and gh stand-in. `repos` maps a checkout directory to its facts. Records every argv in `calls`."""

    def __init__(self, repos: dict):
        self.repos, self.calls = repos, []

    def __call__(self, argv, timeout, env=None):
        self.calls.append(list(argv))
        if argv[0] == "gh":
            return self._gh(argv)
        root = argv[2]
        r = self.repos.get(root)
        if r is None:
            return 128, "", "not a git repository"
        rest = argv[3:]
        if rest[:2] == ["rev-parse", "--show-toplevel"]:
            return 0, root + "\n", ""
        if rest[:2] == ["rev-parse", "HEAD"]:
            return 0, r["head"] + "\n", ""
        if rest[0] == "symbolic-ref":
            return 0, r.get("branch", "main") + "\n", ""
        if rest[0] == "ls-tree":
            return 0, "", ""
        if rest[:2] == ["remote", "get-url"]:
            return 0, r.get("origin", "https://github.com/tenstorrent/repo.git") + "\n", ""
        if rest[0] == "ls-remote":
            if r.get("down"):
                return 128, "", "fatal: unable to access: could not resolve host"
            return 0, f"{r['tip']}\t{rest[-1]}\n", ""
        if rest[0] == "cat-file":
            return (0 if r.get("local_objects") else 1), "", ""
        if rest[:3] == ["rev-list", "--left-right", "--count"]:
            return 0, f"{r.get('ahead', 0)}\t{r['behind']}\n", ""
        if rest[0] == "rev-list" and "--format=%ct" in rest and "-n" not in rest:
            return 0, "".join(f"commit {i:040x}\n{int(t)}\n" for i, t in enumerate(r["missing_times"])), ""
        if rest[0] == "rev-list" and "-n" in rest:
            return 0, f"commit {r['tip']}\n{int(r['tip_time'])}\n", ""
        return 1, "", "unexpected"

    def _gh(self, argv):
        path = argv[2]
        for r in self.repos.values():
            if r.get("down"):
                continue
            if "/compare/" in path and r["head"] in path:
                return 0, json.dumps({"ahead_by": r["behind"], "behind_by": r.get("ahead", 0),
                                      "merge_base_commit": {"commit": {"committer": {"date": iso(r["base_time"])}}}}), ""
            if path.endswith("/commits/" + r["tip"]):
                return 0, json.dumps({"commit": {"committer": {"date": iso(r["tip_time"])}}}), ""
        return 1, "", "no answer"


def repo(tmp_path, name, *, behind=0, days=0.0, up="main", head=None, tip=None, **kw):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    import hashlib
    head = head or hashlib.sha1(name.encode()).hexdigest()
    tip = tip or (head if behind == 0 else hashlib.sha1((name + "-tip").encode()).hexdigest())
    base = NOW - 100 * DAY
    return str(d), {"head": head, "tip": tip, "behind": behind, "base_time": base, "tip_time": base + days * DAY, **kw}


def config(tmp_path, **cfg):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


def probe_says(version, classes=None, importable=True):
    calls = []

    def run(argv, timeout, env=None):
        calls.append(argv)
        wanted = argv[4:]
        got = {c: (classes or {}).get(c, False) for c in wanted}
        return 0, json.dumps({"importable": importable, "version": version if importable else None, "classes": got if importable else {}}) + "\n", ""
    run.calls = calls
    return run


@pytest.fixture
def py(tmp_path):
    p = tmp_path / "ref-python"
    p.write_text("#!/bin/sh\n")
    p.chmod(0o755)
    return str(p)


def full(tmp_path, py, *, metal=None, vllm=None, skills=None, model=None):
    """Sources and a fake upstream where every part is current unless the test overrides it."""
    parts = {}
    for key, given, name in (("metal", metal, "tt-metal"), ("vllm", vllm, "vllm"), ("skills", skills, "skills")):
        d, facts = given or repo(tmp_path, name)
        parts[key] = d
        parts[key + "_facts"] = facts
    fake = FakeUpstream({parts[k]: parts[k + "_facts"] for k in ("metal", "vllm", "skills")})
    src = sp.Sources(model_dir=model or config(tmp_path, model_type="llama", architectures=["LlamaForCausalLM"], transformers_version="4.40.0"),
                     ref_python=py, tt_metal_dir=parts["metal"], vllm_dir=parts["vllm"], skills_dirs=(parts["skills"],))
    return src, fake


def build(src, fake, probe, *, new_run=True, offline=False):
    return sp.build_record(src, new_run=new_run, run=fake, probe_run=probe, cache=skillsfresh.Cache(Path("/nonexistent/cache.json"), write=False),
                           offline=offline, now=NOW)


GOOD = lambda: probe_says("4.57.6", {"LlamaForCausalLM": True})     # noqa: E731
names = lambda items: [i["name"] for i in items]                     # noqa: E731


# ---- every part current ------------------------------------------------------------------------------------------------

def test_everything_current_is_ok_with_no_hazard_and_no_advisory(tmp_path, py):
    src, fake = full(tmp_path, py)
    rec = build(src, fake, GOOD())
    assert rec["severity"] == "ok" and rec["hazards"] == [] and rec["advisories"] == []
    assert rec["parts"]["tt_metal"]["state"] == skillsfresh.CURRENT
    assert rec["parts"]["tt_vllm_plugin"]["state"] == skillsfresh.CURRENT
    assert [p["state"] for p in rec["parts"]["skills"]] == [skillsfresh.CURRENT]
    assert rec["parts"]["transformers"]["state"] == skillsfresh.CURRENT and rec["parts"]["transformers"]["installed"] == "4.57.6"
    assert rec["mode"] == "new"


# ---- the concrete case from this week: qwen4_exp needs transformers 5.19.0 -------------------------------------------

QWEN4 = dict(model_type="qwen4_exp", architectures=["Qwen4ExpForCausalLM"])


def test_a_reference_interpreter_below_5_19_0_is_critical_for_qwen4_exp(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = build(src, fake, probe_says("4.57.6", {"Qwen4ExpForCausalLM": False}))
    assert rec["severity"] == "critical"
    assert names(rec["hazards"]) == [sp.H_BELOW, sp.H_CLASS]
    below, cls = rec["hazards"]
    assert "4.57.6" in below["reason"] and "5.19.0" in below["reason"]
    assert "Qwen4ExpForCausalLM" in cls["reason"]
    t = rec["parts"]["transformers"]
    assert t["state"] == skillsfresh.BEHIND and t["checkpoint_minimum"] == "5.19.0" and "KNOWN_MINIMUMS" in t["minimum_source"]


def test_5_19_0_with_the_class_present_passes_for_qwen4_exp(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = build(src, fake, probe_says("5.19.0", {"Qwen4ExpForCausalLM": True}))
    assert rec["severity"] == "ok" and rec["parts"]["transformers"]["state"] == skillsfresh.CURRENT


def test_the_class_missing_alone_is_critical_even_when_the_version_is_enough(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = build(src, fake, probe_says("5.19.0", {"Qwen4ExpForCausalLM": False}))
    assert names(rec["hazards"]) == [sp.H_CLASS]


def test_the_config_s_own_transformers_version_is_the_minimum_and_the_larger_one_wins(tmp_path, py):
    model = config(tmp_path, model_type="qwen4_exp", architectures=["X"], transformers_version="5.21.0")
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("5.19.0", {"X": True}))
    assert names(rec["hazards"]) == [sp.H_BELOW] and "5.21.0" in rec["hazards"][0]["reason"]


@pytest.mark.parametrize("installed,minimum,critical", [
    ("4.51.0", "4.51.0", False), ("4.50.9", "4.51.0", True), ("4.51.1", "4.51.0", False), ("5.0.0", "4.99.9", False),
    ("4.51.0", "4.51.0.dev0", False),       # a release satisfies its own dev minimum
    ("4.51.0rc1", "4.51.0", True),          # a pre-release is below the release
    ("4.51", "4.51.0", False), ("10.0.0", "9.9.9", False),      # compared as numbers
    ("4.57.6+cu121", "4.57.0", False)])
def test_version_comparison(tmp_path, py, installed, minimum, critical):
    model = config(tmp_path, model_type="llama", architectures=["A"], transformers_version=minimum)
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says(installed, {"A": True}))
    assert (rec["severity"] == "critical") is critical


def test_transformers_that_cannot_be_imported_is_critical(tmp_path, py):
    src, fake = full(tmp_path, py)
    rec = build(src, fake, probe_says(None, importable=False))
    assert names(rec["hazards"]) == [sp.H_NOT_IMPORTABLE]


# ---- a new run and a resume --------------------------------------------------------------------------------------------

def test_with_resume_the_same_findings_are_advisories(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = build(src, fake, probe_says("4.57.6", {"Qwen4ExpForCausalLM": False}), new_run=False)
    assert rec["severity"] == "advisory" and rec["hazards"] == [] and rec["mode"] == "resume"
    assert names(rec["advisories"]) == [sp.H_BELOW, sp.H_CLASS]
    assert all(a["would_be_critical"] for a in rec["advisories"])


# ---- checkouts ----------------------------------------------------------------------------------------------------------

def test_a_tt_metal_checkout_65_commits_and_42_days_behind_is_critical(tmp_path, py):
    src, fake = full(tmp_path, py, metal=repo(tmp_path, "tt-metal", behind=65, days=42))
    rec = build(src, fake, GOOD())
    assert names(rec["hazards"]) == [sp.H_TT_METAL]
    assert "65 commits" in rec["hazards"][0]["reason"] and "42 days" in rec["hazards"][0]["reason"]
    assert rec["parts"]["tt_metal"]["state"] == skillsfresh.BEHIND and rec["parts"]["tt_metal"]["behind"] == 65
    assert rec["parts"]["tt_metal"]["behind_days"] == 42.0


def test_the_vllm_plugin_and_the_skills_clone_get_the_same_rule(tmp_path, py):
    src, fake = full(tmp_path, py, vllm=repo(tmp_path, "vllm", behind=65, days=43), skills=repo(tmp_path, "skills", behind=4, days=20))
    rec = build(src, fake, GOOD())
    assert names(rec["hazards"]) == [sp.H_VLLM, sp.H_SKILLS]


@pytest.mark.parametrize("days,critical", [(0.0, False), (2.0, False), (7.0, False), (7.1, True), (8.0, True)])
def test_the_seven_day_line(tmp_path, py, days, critical):
    src, fake = full(tmp_path, py, metal=repo(tmp_path, "tt-metal", behind=3, days=days))
    rec = build(src, fake, GOOD())
    assert (sp.H_TT_METAL in names(rec["hazards"])) is critical
    assert rec["parts"]["tt_metal"]["state"] == skillsfresh.BEHIND          # a small gap is still reported as behind


def test_the_age_comes_from_local_objects_when_the_tip_is_here(tmp_path, py):
    m = repo(tmp_path, "tt-metal", behind=2, days=0, local_objects=True)
    m[1]["missing_times"] = [NOW - 30 * DAY, NOW - 29 * DAY]
    m[1]["tip_time"] = NOW - 10 * DAY
    src, fake = full(tmp_path, py, metal=m)
    rec = build(src, fake, GOOD())
    assert rec["parts"]["tt_metal"]["behind_days"] == 20.0 and names(rec["hazards"]) == [sp.H_TT_METAL]
    assert not any(c[0] == "gh" for c in fake.calls)           # no GitHub call was needed


def test_a_gap_whose_age_cannot_be_told_is_an_advisory_never_a_pass(tmp_path, py):
    m = repo(tmp_path, "tt-metal", behind=65, days=42)
    src, fake = full(tmp_path, py, metal=m)
    broken = lambda argv, timeout, env=None: (1, "", "no answer") if argv[0] == "gh" and "/commits/" in argv[2] else fake(argv, timeout, env)  # noqa: E731
    rec = build(src, broken, GOOD())
    assert rec["hazards"] == [] and rec["severity"] == "advisory"
    assert names(rec["advisories"]) == [sp.H_TT_METAL + "_age_unknown"]


def test_an_unreachable_upstream_is_an_advisory_for_each_part_and_never_a_silent_pass(tmp_path, py):
    src, fake = full(tmp_path, py, metal=repo(tmp_path, "tt-metal", down=True), vllm=repo(tmp_path, "vllm", down=True),
                     skills=repo(tmp_path, "skills", down=True))
    rec = build(src, fake, GOOD())
    assert rec["severity"] == "advisory" and rec["hazards"] == []
    assert names(rec["advisories"]) == [sp.H_TT_METAL + "_unchecked", sp.H_VLLM + "_unchecked", sp.H_SKILLS + "_unchecked"]
    assert rec["parts"]["tt_metal"]["state"] == skillsfresh.UNKNOWN


def test_offline_makes_no_network_call_and_every_remote_part_is_unknown(tmp_path, py):
    src, fake = full(tmp_path, py)
    rec = build(src, fake, GOOD(), offline=True)
    assert not any(c[0] == "gh" or "ls-remote" in c for c in fake.calls)
    assert rec["severity"] == "advisory" and rec["parts"]["tt_metal"]["state"] == skillsfresh.UNKNOWN


def test_parts_nobody_named_are_unknown_advisories(tmp_path):
    rec = sp.build_record(sp.Sources(), new_run=True, run=FakeUpstream({}), probe_run=GOOD(), now=NOW)
    assert rec["severity"] == "advisory" and rec["hazards"] == []
    assert names(rec["advisories"]) == [sp.H_TT_METAL + "_unchecked", sp.H_VLLM + "_unchecked", sp.H_SKILLS + "_unchecked", "transformers_unchecked"]


def test_ahead_is_not_a_finding_and_differs_is_an_advisory(tmp_path, py):
    ahead = repo(tmp_path, "tt-metal", behind=0, ahead=3, head=HEX("a"), tip=HEX("b"))
    src, fake = full(tmp_path, py, metal=ahead)
    assert build(src, fake, GOOD())["severity"] == "ok"
    # no local objects and no compare answer: how far apart cannot be counted
    differs = repo(tmp_path, "vllm", behind=5, days=1)
    src, fake = full(tmp_path, py, vllm=differs)
    nogh = lambda argv, timeout, env=None: (1, "", "no answer") if argv[0] == "gh" else fake(argv, timeout, env)   # noqa: E731
    rec = build(src, nogh, GOOD())
    assert rec["parts"]["tt_vllm_plugin"]["state"] == skillsfresh.DIFFERS and names(rec["advisories"]) == [sp.H_VLLM + "_differs"]


def test_the_vllm_plugin_is_compared_with_the_named_upstream_branch(tmp_path, py):
    src, fake = full(tmp_path, py)
    src = sp.Sources(**{**src.__dict__, "vllm_branch": "dev"})
    build(src, fake, GOOD())
    asked = [c[-1] for c in fake.calls if "ls-remote" in c]
    assert "refs/heads/dev" in asked and asked.count("refs/heads/main") == 2


def test_the_home_directory_never_reaches_the_record(tmp_path, py, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    src, fake = full(tmp_path, py, metal=repo(tmp_path, "tt-metal", behind=65, days=42))
    text = json.dumps(build(src, fake, GOOD()))
    assert str(tmp_path) not in text and "~/tt-metal" in text


# ---- hostile and wrong-shaped inputs ----------------------------------------------------------------------------------

@pytest.mark.parametrize("model_dir", [None, "", "no/such/dir", "a\0b", 7])
def test_a_checkpoint_directory_that_cannot_be_used_gives_an_unchecked_advisory(tmp_path, py, model_dir):
    src, fake = full(tmp_path, py)
    src = sp.Sources(**{**src.__dict__, "model_dir": model_dir})
    rec = build(src, fake, GOOD())
    assert rec["parts"]["transformers"]["state"] == skillsfresh.UNKNOWN and "transformers_unchecked" in names(rec["advisories"])
    assert rec["hazards"] == []


@pytest.mark.parametrize("cfg", [
    {"model_type": None, "architectures": None, "transformers_version": None},
    {"model_type": [], "architectures": "Llama", "transformers_version": {}},
    {"model_type": "x" * 500, "architectures": [None, 3, "a b", "a\0b", "x" * 500], "transformers_version": 5},
    {"architectures": ["../../etc/passwd", "-c", "A;B"], "transformers_version": "not a version"}])
def test_wrong_typed_config_fields_never_raise_and_never_reach_the_interpreter_as_class_names(tmp_path, py, cfg):
    src, fake = full(tmp_path, py, model=config(tmp_path, **cfg))
    probe = probe_says("4.57.6")
    rec = build(src, fake, probe)
    assert rec["severity"] in ("advisory", "ok") and rec["hazards"] == []
    for argv in probe.calls:
        assert all(sp.CLASS_NAME.match(a) for a in argv[4:])


@pytest.mark.parametrize("kind", ["fifo", "zero_link", "oversize", "not_json", "list"])
def test_a_hostile_config_file_is_refused_without_blocking(tmp_path, py, kind):
    d = tmp_path / "model"
    d.mkdir()
    c = d / "config.json"
    if kind == "fifo":
        os.mkfifo(c)
    elif kind == "zero_link":
        c.symlink_to("/dev/zero")
    elif kind == "oversize":
        c.write_bytes(b'{"a": "' + b"x" * (model_config_max() + 10) + b'"}')
    elif kind == "not_json":
        c.write_text("{nan")
    else:
        c.write_text("[]")
    src, fake = full(tmp_path, py, model=str(d))
    rec = build(src, fake, GOOD())
    assert rec["parts"]["transformers"]["state"] == skillsfresh.UNKNOWN


def model_config_max() -> int:
    from orchard import model_config
    return model_config.MAX_CONFIG_BYTES


@pytest.mark.parametrize("python", [None, "", "a\0b", "/no/such/python", "/tmp"])
def test_an_unusable_interpreter_is_unknown(tmp_path, python):
    out = sp.probe_interpreter(python, ["A"], run=lambda *a, **k: pytest.fail("must not run"))
    assert out["state"] == "unknown"


@pytest.mark.parametrize("answer", [
    (0, "", ""), (0, "not json", ""), (0, "[]", ""), (0, json.dumps({"importable": "yes", "classes": {}}), ""),
    (0, json.dumps({"importable": True, "version": None, "classes": {"A": True}}), ""),
    (0, json.dumps({"importable": True, "version": "5.0.0", "classes": {}}), ""),      # a class answer missing
    (0, json.dumps({"importable": True, "version": "5.0.0", "classes": {"A": "yes"}}), ""),
    (0, '{"importable": true, "version": NaN, "classes": {"A": true}}', ""), (1, "", "boom"), (124, "", "timed out")])
def test_a_probe_answer_that_is_not_usable_is_unknown(py, answer):
    out = sp.probe_interpreter(py, ["A"], run=lambda *a, **k: answer)
    assert out["state"] == "unknown"


def test_a_runner_that_raises_is_a_failed_probe(py):
    def boom(*a, **k):
        raise RuntimeError("x")
    assert sp.probe_interpreter(py, [], run=boom)["state"] == "unknown"


def test_warnings_before_the_answer_do_not_hide_it(py):
    out = sp.probe_interpreter(py, ["A"], run=lambda *a, **k: (0, "FutureWarning: x\n" + json.dumps({"importable": True, "version": "5.1.0", "classes": {"A": True}}) + "\n", ""))
    assert out["state"] == "answered" and out["version"] == "5.1.0"


# ---- the probe script itself -------------------------------------------------------------------------------------------

def run_script(monkeypatch, capsys, fake_module, classes):
    monkeypatch.setitem(sys.modules, "transformers", fake_module) if fake_module is not None else monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.setattr(sys, "argv", ["-c", *classes])
    exec(compile(sp.PROBE_SCRIPT, "<probe>", "exec"), {})
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_the_probe_script_reports_the_version_and_which_classes_exist(monkeypatch, capsys):
    mod = types.ModuleType("transformers")
    mod.__spec__ = importlib.machinery.ModuleSpec("transformers", None)       # find_spec needs the spec a real module has
    mod.__version__ = "5.19.0"
    mod.Qwen4ExpForCausalLM = object
    out = run_script(monkeypatch, capsys, mod, ["Qwen4ExpForCausalLM", "NoSuchClass"])
    assert out == {"importable": True, "version": "5.19.0", "classes": {"Qwen4ExpForCausalLM": True, "NoSuchClass": False}}


def test_the_probe_script_treats_a_class_whose_import_raises_as_missing(monkeypatch, capsys):
    class Lazy(types.ModuleType):
        def __getattr__(self, name):
            raise ImportError("backend missing")
    mod = Lazy("transformers")
    mod.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    mod.__version__ = "5.0.0"
    assert run_script(monkeypatch, capsys, mod, ["Z"])["classes"] == {"Z": False}


def test_the_probe_script_reports_not_importable_when_transformers_is_absent(monkeypatch, capsys):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    out = run_script(monkeypatch, capsys, None, ["A"])
    assert out["importable"] is False and out["version"] is None


def test_the_probe_script_runs_under_isolated_mode_in_a_real_interpreter():
    """The script compiles and prints one JSON object under `python -I`, with the original runner (the suite's guard replaces the module's)."""
    spec = importlib.util.spec_from_file_location("skillsfresh_real", skillsfresh.__file__)
    real = importlib.util.module_from_spec(spec)
    sys.modules["skillsfresh_real"] = real         # a dataclass in the module looks its own module up by name
    try:
        spec.loader.exec_module(real)
    finally:
        sys.modules.pop("skillsfresh_real", None)
    out = sp.probe_interpreter(sys.executable, ["DoesNotExistForCausalLM"], run=real._raw_run)
    assert out["state"] == "answered" and out["classes"].get("DoesNotExistForCausalLM", False) is False


def test_the_probe_never_names_ttnn_or_ttml():
    assert "ttnn" not in sp.PROBE_SCRIPT and "ttml" not in sp.PROBE_SCRIPT


# ---- the record file and the gate -------------------------------------------------------------------------------------

def written(tmp_path, py, **kw):
    src, fake = full(tmp_path, py, **kw)
    rec = build(src, fake, GOOD())
    stage = tmp_path / "stage0"
    stage.mkdir(exist_ok=True)
    return stage, sp.write_record(stage, rec), rec


def test_the_gate_accepts_the_recorded_file(tmp_path, py):
    stage, sha, _ = written(tmp_path, py)
    assert sp.gate_reasons(stage, sha) == []
    assert sha == __import__("hashlib").sha256((stage / sp.FILE).read_bytes()).hexdigest()


def test_the_gate_refuses_a_file_with_no_ledger_entry(tmp_path, py):
    stage, _, _ = written(tmp_path, py)
    assert "ledger holds no entry" in sp.gate_reasons(stage, None)[0]


def test_the_gate_refuses_a_missing_file(tmp_path, py):
    stage, sha, _ = written(tmp_path, py)
    (stage / sp.FILE).unlink()
    assert sp.gate_reasons(stage, sha) and "cannot be used" in sp.gate_reasons(stage, sha)[0]


def test_the_gate_refuses_a_rewritten_file(tmp_path, py):
    stage, sha, rec = written(tmp_path, py)
    (stage / sp.FILE).write_text(json.dumps({**rec, "severity": "ok", "advisories": [], "note": "edited"}))
    assert "hash changed" in sp.gate_reasons(stage, sha)[0]


def test_the_gate_refuses_a_link_in_place_of_the_file(tmp_path, py):
    stage, sha, _ = written(tmp_path, py)
    other = tmp_path / "other.json"
    other.write_bytes((stage / sp.FILE).read_bytes())
    (stage / sp.FILE).unlink()
    (stage / sp.FILE).symlink_to(other)
    assert sp.gate_reasons(stage, sha) and "cannot be used" in sp.gate_reasons(stage, sha)[0]


def test_the_gate_does_not_block_on_a_fifo(tmp_path, py):
    stage, sha, _ = written(tmp_path, py)
    (stage / sp.FILE).unlink()
    os.mkfifo(stage / sp.FILE)
    assert "cannot be used" in sp.gate_reasons(stage, sha)[0]


@pytest.mark.parametrize("edit", [
    lambda r: None, lambda r: [], lambda r: {**r, "schema": 2}, lambda r: {**r, "severity": "fine"}, lambda r: {**r, "mode": None},
    lambda r: {**r, "hazards": [{}]}, lambda r: {**r, "hazards": None}, lambda r: {**r, "advisories": [{"name": 1, "reason": "x"}]},
    lambda r: {**r, "parts": []}])
def test_the_gate_refuses_a_record_of_the_wrong_shape(tmp_path, py, edit):
    stage, _, rec = written(tmp_path, py)
    data = json.dumps(edit(rec)).encode()
    (stage / sp.FILE).write_bytes(data)
    assert sp.gate_reasons(stage, __import__("hashlib").sha256(data).hexdigest())


def test_the_gate_names_each_critical_hazard(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = build(src, fake, probe_says("4.57.6", {"Qwen4ExpForCausalLM": False}))
    stage = tmp_path / "stage0"
    stage.mkdir()
    sha = sp.write_record(stage, rec)
    reasons = sp.gate_reasons(stage, sha)
    assert len(reasons) == 2 and sp.H_BELOW in reasons[0] and sp.H_CLASS in reasons[1]


def test_the_unchecked_record_is_an_advisory_and_passes_the_gate(tmp_path):
    rec = sp.unchecked_record("no stack check was supplied", new_run=True)
    stage = tmp_path / "s"
    stage.mkdir()
    sha = sp.write_record(stage, rec)
    assert rec["severity"] == "advisory" and rec["hazards"] == [] and sp.gate_reasons(stage, sha) == []


def test_the_record_is_written_all_or_nothing_and_never_through_a_link(tmp_path, py):
    stage = tmp_path / "s"
    stage.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    (stage / sp.FILE).symlink_to(victim)
    sp.write_record(stage, sp.unchecked_record("x", new_run=True))
    assert victim.read_text() == "keep" and not (stage / sp.FILE).is_symlink()


def test_make_check_reads_the_offline_variable(tmp_path, py, monkeypatch):
    src, fake = full(tmp_path, py)
    stage = tmp_path / "s"
    stage.mkdir()
    monkeypatch.setenv(sp.OFFLINE_ENV, "1")
    rec, sha = sp.make_check(src, run=fake, probe_run=GOOD())(stage, True)
    assert not any(c[0] == "gh" or "ls-remote" in c for c in fake.calls) and rec["parts"]["tt_metal"]["state"] == skillsfresh.UNKNOWN
    assert sp.gate_reasons(stage, sha) == []


def test_a_known_minimum_larger_than_the_config_s_version_wins(tmp_path, py):
    model = config(tmp_path, model_type="qwen4_exp", architectures=["Qwen4ExpForCausalLM"], transformers_version="4.40.0")
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("5.0.0", {"Qwen4ExpForCausalLM": True}))
    assert names(rec["hazards"]) == [sp.H_BELOW] and "5.19.0" in rec["hazards"][0]["reason"]
    assert rec["parts"]["transformers"]["checkpoint_minimum"] == "5.19.0"


@pytest.mark.parametrize("bad", ["a b", "-c", "A;B", "../x", "", "a\0b", "x" * 200, 5, None, ["A"]])
def test_only_plain_class_names_reach_the_interpreter_whoever_calls_the_probe(py, bad):
    seen = []
    answer = (0, json.dumps({"importable": True, "version": "5.0.0", "classes": {"Good": True}}), "")
    sp.probe_interpreter(py, ["Good", bad], run=lambda argv, t, e=None: (seen.append(argv), answer)[1])
    assert seen and seen[0][4:] == ["Good"]


# ---- skillsfresh additions -------------------------------------------------------------------------------------------

def test_check_clone_refuses_a_branch_name_it_will_not_ask_about(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    fake = FakeUpstream({str(d): {"head": HEX("a"), "tip": HEX("a"), "behind": 0}})
    runner = skillsfresh.ReadOnlyRunner(fake)
    for bad in ("", "-x", "a..b", "a b", "a;b", "x" * 200, "a/", "a.lock"):
        f = skillsfresh.check_clone(str(d), source="s", runner=runner, cache=skillsfresh.Cache(write=False), update=skillsfresh.skills_update,
                                    upstream_branch=bad)
        assert f.state == skillsfresh.UNKNOWN and "branch name" in f.detail
    assert not any("ls-remote" in c for c in fake.calls)


def test_behind_days_is_left_alone_unless_asked_for(tmp_path):
    d, facts = repo(tmp_path, "c", behind=5, days=30)
    fake = FakeUpstream({d: facts})
    f = skillsfresh.check_clone(d, source="s", runner=skillsfresh.ReadOnlyRunner(fake), cache=skillsfresh.Cache(write=False), update=skillsfresh.skills_update)
    assert f.state == skillsfresh.BEHIND and f.behind_days is None
    assert not any("/commits/" in c[2] for c in fake.calls if c[0] == "gh")
