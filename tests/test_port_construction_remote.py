# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The effective construction for a model that ships its own code (issue 781).

`microsoft/Aion-1.0-Instruct-2.3B` carries `modeling_aion_instruct.py`. The script used `trust_remote_code=False`, transformers refused, the
script exited with code 1, the ledger said only "the script exited with code 1", and `--thin-port` printed "thin port not used". The script
now takes stage 1's route (`reference_gate.allow_remote_code`): when transformers refuses, credential-like environment variables are removed,
the hub goes offline, HOME moves to an empty scratch directory, every `.py` file is listed with its sha256, and the load runs again with
`trust_remote_code=True`. A model without custom code is loaded as before, with no trust flag.

The fakes stand in for `torch` and `transformers`. One test runs the real reference interpreter on a tiny model with custom code that it builds
in a temporary directory, and skips with the name of what it lacks."""
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import port_construction as pc
from orchard import supervisor, thin_port
from orchard.commands import CommandResult
from test_port_construction import Mod, load_script, _reference_python

REFUSAL = ("The repository contains custom code which must be executed to correctly load the model. "
           "Please pass the argument `trust_remote_code=True` to allow custom code to be run.")
SHIPPED_TEMPLATES = Path(pc.SCRIPT).resolve().parent.parent / "reference-gate-templates"


@pytest.fixture(autouse=True)
def restore_environment():
    """The route changes os.environ directly, so the whole environment is saved and restored."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


class FakeConfig:
    architectures = ["NotInTransformersForCausalLM"]


class FakeRemoteConfig(FakeConfig):
    auto_map = {"AutoModelForCausalLM": "modeling_tiny_remote.TinyRemoteForCausalLM"}      # what a config with custom code carries


def install_fakes(monkeypatch, *, refuse: bool, calls: list, fail_with=None, config_error=None):
    """A `torch` with a `device` context and a `transformers` whose loaders refuse custom code unless trust_remote_code=True is passed. A call
    that passes no flag is recorded as None: the real transformers asks on the terminal for None, which must never happen in the script."""
    torch = types.ModuleType("torch")
    torch.device = lambda name: contextlib.nullcontext()
    transformers = types.ModuleType("transformers")

    def check(kind, trust):
        calls.append((kind, trust))
        if refuse and not trust:
            raise ValueError(REFUSAL)

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, trust_remote_code=None):
            check("config", trust_remote_code)
            if config_error is not None:
                raise config_error
            return FakeRemoteConfig() if refuse else FakeConfig()

    class AutoModelForCausalLM:
        @staticmethod
        def from_config(config, trust_remote_code=None):
            check("model", trust_remote_code)
            if fail_with is not None:
                raise fail_with
            return Mod().add("layer", Mod())

    transformers.AutoConfig, transformers.AutoModelForCausalLM, transformers.AutoModel = AutoConfig, AutoModelForCausalLM, None
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def snapshot_with_code(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({"model_type": "tiny_remote", "architectures": ["NotInTransformersForCausalLM"]}))
    (snap / "modeling_tiny_remote.py").write_text("class TinyRemote: pass\n")
    return snap


# ---- the route in the script ----------------------------------------------------------------------------------------------------------

def test_a_refusal_is_answered_with_stage_1s_route_and_a_second_load_with_trust_remote_code(tmp_path, monkeypatch):
    snap = snapshot_with_code(tmp_path)
    calls = []
    install_fakes(monkeypatch, refuse=True, calls=calls)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("KEEP_ME", "1")
    home_before = os.environ.get("HOME")
    script = load_script()
    model = script.build_model(str(snap))
    assert model is not None
    assert calls == [("config", False), ("config", True), ("model", True)]
    assert "HF_TOKEN" not in os.environ and os.environ["KEEP_ME"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HOME"] != home_before
    report = script.remote_code_report()
    assert report["used"] is True and "HF_TOKEN" in report["scrubbed"]       # other credential-like names of this machine may be listed too
    assert report["files"] == {"modeling_tiny_remote.py": hashlib.sha256(b"class TinyRemote: pass\n").hexdigest()}
    assert "hf_secret_value" not in json.dumps(report)                 # the names of scrubbed variables are recorded and never their values


def test_a_model_without_custom_code_is_loaded_without_the_trust_flag_and_the_route_is_not_used(tmp_path, monkeypatch):
    snap = snapshot_with_code(tmp_path)
    calls = []
    install_fakes(monkeypatch, refuse=False, calls=calls)
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    script = load_script()
    script.build_model(str(snap))
    assert calls == [("config", False), ("model", None)]                # a plain model is built as before: no flag
    assert os.environ["HF_TOKEN"] == "hf_secret_value"                  # nothing was scrubbed
    assert script.remote_code_report() is None


def test_the_scratch_home_and_the_audit_are_not_written_beside_the_shipped_template(tmp_path, monkeypatch):
    snap = snapshot_with_code(tmp_path)
    install_fakes(monkeypatch, refuse=True, calls=[])
    script = load_script()
    before = sorted(p.name for p in SHIPPED_TEMPLATES.iterdir())
    script.build_model(str(snap))
    assert sorted(p.name for p in SHIPPED_TEMPLATES.iterdir()) == before
    assert not Path(os.environ["HOME"]).is_relative_to(SHIPPED_TEMPLATES)


def test_another_value_error_from_the_config_load_is_not_taken_for_a_refusal(tmp_path, monkeypatch):
    snap = snapshot_with_code(tmp_path)
    calls = []
    install_fakes(monkeypatch, refuse=False, calls=calls, config_error=ValueError("some other problem"))
    script = load_script()
    with pytest.raises(ValueError, match="some other problem"):
        script.build_model(str(snap))
    assert calls == [("config", False)]                                   # no second load
    assert script.remote_code_report() is None and os.environ.get("HF_HUB_OFFLINE") != "1"


def test_main_prints_the_result_with_the_remote_code_report(tmp_path, monkeypatch, capsys):
    snap = snapshot_with_code(tmp_path)
    install_fakes(monkeypatch, refuse=True, calls=[])
    script = load_script()
    assert script.main(["x", str(snap)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith(script.MARK + " "))
    assert json.loads(line[len(script.MARK) + 1:])["remote_code"]["used"] is True


def test_a_failure_prints_the_exception_class_and_whether_the_models_code_was_in_use_and_nothing_else(tmp_path, monkeypatch, capsys):
    snap = snapshot_with_code(tmp_path)
    install_fakes(monkeypatch, refuse=True, calls=[], fail_with=KeyError("a secret path /home/x/.ssh/id_rsa"))
    script = load_script()
    assert script.main(["x", str(snap)]) == 3
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith(script.ERROR_MARK + " "))
    assert json.loads(line[len(script.ERROR_MARK) + 1:]) == {"error": "RuntimeError", "remote_code": True}
    assert ".ssh" not in out


# ---- what the supervisor's module reads back -------------------------------------------------------------------------------------------

def base_data():
    """A minimal, valid script result."""
    return {"model_class": "TinyRemoteForCausalLM", "modules_total": 2, "classes_left_out": 0, "modules_cut": False,
            "classes": [{"class": "Linear", "signature": [], "variants": [{"count": 1, "paths": ["a"], "attrs": {"bias": True}}],
                         "variants_left_out": 0}]}


def with_report(**kw):
    report = {"used": True, "files": {"modeling_tiny_remote.py": "a" * 64}, "files_total": 1, "scrubbed": ["HF_TOKEN"]}
    report.update(kw)
    return {**base_data(), "remote_code": report}


def ok_run(data):
    return lambda argv, timeout, env=None: CommandResult(tuple(argv), 0, pc.MARK + " " + json.dumps(data) + "\n", "")


def test_a_result_with_a_remote_code_report_is_kept_and_the_file_and_summary_say_so():
    text, flags, summary, why = pc.for_snapshot("/py", "/snap", runner=ok_run(with_report()))
    assert not why, why
    intro = text.split("## Resolved vs class default", 1)[0]
    assert "ships its own code (1 .py file" in intro and "HF_TOKEN" in intro        # the file says the model's code ran, and which names were scrubbed
    assert summary["remote_code"] == {"files": {"modeling_tiny_remote.py": "a" * 64}, "files_total": 1, "scrubbed": ["HF_TOKEN"]}


def test_a_result_without_a_report_has_no_remote_code_sentence_and_no_summary_key():
    text, flags, summary, why = pc.for_snapshot("/py", "/snap", runner=ok_run(base_data()))
    assert not why and "ships its own code" not in text and "remote_code" not in summary


@pytest.mark.parametrize("report", [
    {"used": True, "files": {"../x.py": "a" * 64}, "files_total": 1, "scrubbed": []},                      # a path that leaves the snapshot
    {"used": True, "files": {"/abs.py": "a" * 64}, "files_total": 1, "scrubbed": []},
    {"used": True, "files": {"a/../x.py": "a" * 64}, "files_total": 1, "scrubbed": []},
    {"used": True, "files": {"a//x.py": "a" * 64}, "files_total": 1, "scrubbed": []},
    {"used": False, "files": {}, "files_total": 0, "scrubbed": []},                                         # a report that does not say used
    {"used": True, "files": {"m.py": "not a hash"}, "files_total": 1, "scrubbed": []},
    {"used": True, "files": {"m.py": "a" * 64}, "files_total": "1", "scrubbed": []},
    {"used": True, "files": {f"m{i}.py": "a" * 64 for i in range(65)}, "files_total": 65, "scrubbed": []},    # over the cap
    {"used": True, "files": {}, "files_total": 0, "scrubbed": ["has space"]},
    {"used": True, "files": {}, "files_total": 0, "scrubbed": ["X"] * 41},
    {"used": True, "files": [], "files_total": 0, "scrubbed": []},
    ["not", "an", "object"],
])
def test_a_malformed_remote_code_report_is_refused(report):
    data = {**base_data(), "remote_code": report}
    text, flags, summary, why = pc.for_snapshot("/py", "/snap", runner=ok_run(data))
    assert text == "" and "refused" in why


def test_the_words_the_stage_1_hasher_gives_for_a_file_it_cannot_hash_are_accepted():
    for word in ("unreadable", "not a regular file", "over the size limit"):
        data = with_report(files={"m.py": word})
        assert not pc.for_snapshot("/py", "/snap", runner=ok_run(data))[3]


def failing_run(stdout, code=3):
    return lambda argv, timeout, env=None: CommandResult(tuple(argv), code, stdout, "")


def test_a_failed_run_names_the_exception_class_the_script_reported():
    out = pc.ERROR_MARK + ' {"error": "RuntimeError", "remote_code": true}\n'
    why = pc.run("/py", "/snap", runner=failing_run(out))[1]
    assert why.startswith("the script exited with code 3") and "RuntimeError" in why and "the model's own code" in why


def test_a_failed_run_without_a_report_line_keeps_the_old_reason():
    assert pc.run("/py", "/snap", runner=failing_run("", code=1))[1] == "the script exited with code 1"


@pytest.mark.parametrize("line", [
    pc.ERROR_MARK + ' {"error": "Bad Name; rm -rf /", "remote_code": false}',
    pc.ERROR_MARK + ' {"error": "' + "A" * 200 + '", "remote_code": false}',
    pc.ERROR_MARK + ' {"error": 5, "remote_code": false}',
    pc.ERROR_MARK + " not json",
    pc.ERROR_MARK + ' {"error": "KeyError", "remote_code": "yes"}',
    pc.ERROR_MARK + ' ["KeyError"]',
])
def test_a_hostile_error_line_gives_the_old_reason_and_quotes_nothing(line):
    why = pc.run("/py", "/snap", runner=failing_run(line + "\n"))[1]
    assert why == "the script exited with code 3"


def test_the_supervisors_written_decision_carries_the_remote_code_report_and_a_failure_carries_the_cause(tmp_path):
    class Ledger:
        def __init__(self):
            self.entries = []

        def append(self, event, stage, **kw):
            self.entries.append(kw)

    stage = tmp_path / "stage"
    stage.mkdir()
    ledger = Ledger()
    written, summary = supervisor.write_effective_construction(ledger, tmp_path, SimpleNamespace(number=2), stage, tmp_path, "/py",
                                                              runner=ok_run(with_report()))
    assert written and summary["remote_code"]["files_total"] == 1
    assert next(k for k in ledger.entries if k["decision"] == "effective construction written")["remote_code"]["files"] == {
        "modeling_tiny_remote.py": "a" * 64}
    ledger = Ledger()
    out = pc.ERROR_MARK + ' {"error": "AttributeError", "remote_code": true}\n'
    written, _ = supervisor.write_effective_construction(ledger, tmp_path, SimpleNamespace(number=2), stage, tmp_path, "/py", runner=failing_run(out))
    assert written is False and "AttributeError" in ledger.entries[0]["problem"]


# ---- the real reference interpreter on a tiny model with custom code ---------------------------------------------------------------------

CONFIGURATION = '''
from transformers import PretrainedConfig


class TinyRemoteConfig(PretrainedConfig):
    model_type = "tiny_remote"

    def __init__(self, hidden_size=16, vocab_size=32, eps=1e-5, **kwargs):
        self.hidden_size, self.vocab_size, self.eps = hidden_size, vocab_size, eps
        super().__init__(**kwargs)
'''

MODELING = '''
import torch
from torch import nn
from transformers import PreTrainedModel
from .configuration_tiny_remote import TinyRemoteConfig


class TinyRemoteNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight, self.variance_epsilon = nn.Parameter(torch.ones(hidden_size)), eps


class TinyRemoteForCausalLM(PreTrainedModel):
    config_class = TinyRemoteConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = TinyRemoteNorm(config.hidden_size, eps=config.eps)          # resolved 1e-5 against the class default 1e-6
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
'''


def tiny_remote_snapshot(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(json.dumps({
        "model_type": "tiny_remote", "architectures": ["TinyRemoteForCausalLM"], "hidden_size": 16, "vocab_size": 32, "eps": 1e-5,
        "auto_map": {"AutoConfig": "configuration_tiny_remote.TinyRemoteConfig",
                     "AutoModelForCausalLM": "modeling_tiny_remote.TinyRemoteForCausalLM"}}))
    (snap / "configuration_tiny_remote.py").write_text(CONFIGURATION)
    (snap / "modeling_tiny_remote.py").write_text(MODELING)
    return snap


@pytest.mark.skipif(_reference_python() is None, reason="needs the reference interpreter ($ORCHARD_REFERENCE_PYTHON or the default venv)")
@pytest.mark.timeout(300)
def test_the_real_reference_interpreter_builds_the_construction_of_a_model_with_custom_code_and_reports_its_files(tmp_path):
    snap = tiny_remote_snapshot(tmp_path)
    python = _reference_python()
    text, flags, summary, why = pc.for_snapshot(python, snap)
    assert not why, why
    assert any("TinyRemoteNorm" in f and "1e-05" in f for f in flags)               # the resolved eps against the class default
    assert set(summary["remote_code"]["files"]) == {"configuration_tiny_remote.py", "modeling_tiny_remote.py"}
    assert summary["remote_code"]["files"]["modeling_tiny_remote.py"] == hashlib.sha256((snap / "modeling_tiny_remote.py").read_bytes()).hexdigest()
    assert pc.for_snapshot(python, snap)[0] == text                                  # two runs, the same bytes
    # the thin-port step's own call of the same script: it printed "thin port not used: the effective construction could not be built"
    header = json.dumps({"embed.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (snap / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
    inputs = thin_port.gather_inputs(run_dir=tmp_path, stage_dir=tmp_path, snapshot=snap, python=python)
    assert inputs.construction["model_class"] == "TinyRemoteForCausalLM"
