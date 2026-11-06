# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The effective construction of the reference model (issue 541, orchard/port_construction.py and
orchard/skills/full-port-templates/effective_construction.py).

The model case is run 18 of the tiny Qwen4 rehearsal: the gated norm's class default is `activation="silu"`, the model's constructor passed
`config.output_gate_type` (sigmoid), and the coder wrote silu. The file must show the resolved value and flag it against the default.
Everything here runs on fakes and on the CPU. The test that uses the real reference interpreter skips with the name of what it lacks."""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import port_construction as pc
from orchard import port_facts, supervisor
from orchard.commands import CommandResult
from test_port_facts import CONFIG, TENSORS, snapshot_with

SCRIPT = pc.SCRIPT


def load_script():
    spec = importlib.util.spec_from_file_location("effective_construction_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- fake modules (no torch): what `named_modules()` and `vars()` give for the pieces of the tiny model --------------------------

class Mod:
    def __init__(self, **kw):
        self._modules = {}
        self.training = True
        self.__dict__.update(kw)

    def add(self, name, child):
        self._modules[name] = child
        return self

    def named_modules(self, prefix=""):
        yield prefix, self
        for name, child in self._modules.items():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)


class RMSNormGated(Mod):
    def __init__(self, hidden, eps=1e-6, activation="silu", **kw):         # the class default the excerpt shows
        super().__init__(**kw)
        self.activation, self.variance_epsilon = activation, eps          # stored as the real class stores them


class RMSNorm(Mod):
    def __init__(self, hidden, group_size=None, eps=1e-6, **kw):
        super().__init__(**kw)
        self.group_size, self.eps = group_size, eps


class Conv1d(Mod):
    def __init__(self, c_in, c_out, k, stride=1, padding=0, dilation=1, **kw):
        super().__init__(**kw)
        self.stride, self.padding, self.dilation = stride if isinstance(stride, tuple) else (stride,), \
            padding if isinstance(padding, tuple) else (padding,), dilation if isinstance(dilation, tuple) else (dilation,)


class GatedDeltaNet(Mod):
    pass


class Model(Mod):
    pass


def fake_model(gate="sigmoid"):
    """Three linear-attention layers whose gated norm resolved `gate`, and a PLE conv with dilation 3 (the other two defects the spike found)."""
    model = Model()
    layers = Mod()
    model.add("layers", layers)
    for i in range(3):
        layer = Mod()
        attn = GatedDeltaNet(layer_idx=i, activation="silu", conv_kernel_size=4)      # `activation` here is the conv's, a different thing
        attn.add("norm", RMSNormGated(32, eps=1e-6, activation=gate, layer_idx=i))
        attn.add("conv1d", Conv1d(320, 320, 4, padding=3, groups=320))
        layer.add("linear_attn", attn)
        layer.add("hc_norm", RMSNorm(64, group_size=64, eps=1e-6))
        layers.add(str(i), layer)
    ple = Mod().add("conv1d", Conv1d(256, 256, 4, dilation=3))
    layers.add("3", Mod().add("ple", ple))
    return model


def described(**kw):
    return pc.clean(load_script().describe(fake_model(**kw)))


# ---- the script: what is read off a constructed model ------------------------------------------------------------------------------

def test_the_script_reads_the_resolved_activation_of_the_gated_norm_not_the_class_default():
    data = described()
    norm = next(c for c in data["classes"] if c["class"] == "RMSNormGated")
    assert norm["variants"][0]["attrs"]["activation"] == "sigmoid"
    assert {"name": "activation", "default": "silu", "shown": True} in norm["signature"]


def test_modules_of_every_layer_fall_in_one_group_and_the_layer_number_is_a_pattern():
    norm = next(c for c in described()["classes"] if c["class"] == "RMSNormGated")
    assert len(norm["variants"]) == 1 and norm["variants"][0]["count"] == 3
    assert norm["variants"][0]["paths"] == ["layers.{i}.linear_attn.norm"] and "layer_idx" not in norm["variants"][0]["attrs"]


def test_a_function_valued_attribute_is_named_and_a_tensor_like_object_is_left_out():
    script = load_script()
    attrs = script.attributes(Mod(act_fn=len, weight=object(), scale=0.5, shape=(1, 2)))
    assert attrs["act_fn"] == "<callable len>" and attrs["scale"] == 0.5 and attrs["shape"] == [1, 2] and "weight" not in attrs


def test_the_scripts_output_is_the_same_bytes_every_time():
    script = load_script()
    one = json.dumps(script.describe(fake_model()), sort_keys=True)
    assert one == json.dumps(script.describe(fake_model()), sort_keys=True)


def test_the_caps_bound_the_output(monkeypatch):
    script = load_script()
    monkeypatch.setattr(script, "MAX_CLASSES", 2)
    big = Model()
    for i in range(10):
        big.add(f"m{i}", type(f"C{i}", (Mod,), {})())
    out = script.describe(big)
    assert len(out["classes"]) == 2 and out["classes_left_out"] == 9


# ---- resolved vs class default ----------------------------------------------------------------------------------------------------

def rows(data):
    return {(r["class"], r["param"]): r for r in pc.differences(data)}


def test_the_gated_norm_activation_is_flagged_resolved_sigmoid_against_default_silu():
    row = rows(described())[("RMSNormGated", "activation")]
    assert row["resolved"] == "sigmoid" and row["default"] == "silu" and row["modules"] == 3
    line = pc.flag_line(row)
    assert "RMSNormGated.activation: resolved 'sigmoid', class default 'silu'" in line and "layers.{i}.linear_attn.norm" in line


def test_a_resolved_value_equal_to_the_default_is_not_flagged_even_when_stored_as_a_one_element_list_or_under_an_alias():
    flagged = rows(described())
    assert ("RMSNormGated", "eps") not in flagged             # eps is stored as variance_epsilon and equals the default
    assert not any(k[0] == "Conv1d" and k[1] == "stride" for k in flagged)     # (1,) equals 1
    assert ("RMSNorm", "eps") not in flagged


def test_an_eps_stored_under_an_alias_is_compared_with_the_constructor_default_and_flagged_when_it_differs():
    model = fake_model()
    model.add("odd", RMSNormGated(32, eps=1e-5))                      # stored as variance_epsilon; the signature calls it eps
    row = rows(pc.clean(load_script().describe(model)))[("RMSNormGated", "eps")]
    assert row["attr"] == "variance_epsilon" and row["resolved"] == 1e-5 and row["default"] == 1e-6
    assert "eps (stored as variance_epsilon)" in pc.flag_line(row)


def test_the_ple_conv_dilation_and_the_grouped_norm_are_flagged():
    flagged = rows(described())
    assert flagged[("Conv1d", "dilation")]["resolved"] == [3] and flagged[("RMSNorm", "group_size")]["resolved"] == 64


def test_when_the_gate_resolves_to_the_class_default_nothing_is_flagged_for_it():
    assert ("RMSNormGated", "activation") not in rows(described(gate="silu"))


def test_a_class_whose_modules_disagree_gives_a_row_for_each_value():
    model = fake_model()
    model.add("extra", RMSNormGated(32, activation="gelu"))
    got = [r for r in pc.differences(pc.clean(load_script().describe(model))) if r["class"] == "RMSNormGated"]
    assert sorted(r["resolved"] for r in got) == ["gelu", "sigmoid"]


# ---- the file ---------------------------------------------------------------------------------------------------------------------------

def test_the_file_lists_the_flags_first_and_one_line_per_group_of_modules():
    text, flags = pc.render(described())
    assert text.index("## Resolved vs class default") < text.index("## Modules")
    assert any("RMSNormGated.activation: resolved 'sigmoid'" in f for f in flags)
    assert "RMSNormGated x3 at layers.{i}.linear_attn.norm: activation='sigmoid'" in text


def test_the_file_is_deterministic_whatever_order_the_data_arrives_in():
    raw = json.loads(json.dumps(load_script().describe(fake_model())))
    shuffled = json.loads(json.dumps(raw))
    shuffled["classes"].reverse()
    for c in shuffled["classes"]:
        c["variants"].reverse()
    assert pc.render(pc.clean(raw))[0].encode() == pc.render(pc.clean(shuffled))[0].encode()


def test_the_summary_pins_the_bytes_of_the_file():
    text, flags = pc.render(described())
    s = pc.summary(text, flags, described())
    import hashlib
    assert s["sha256"] == hashlib.sha256(text.encode()).hexdigest() and s["bytes"] == len(text.encode())
    assert s["resolved_vs_default_total"] == len(flags) and len(s["resolved_vs_default"]) <= pc.SUMMARY_FLAGS


# ---- hostile output from the interpreter ---------------------------------------------------------------------------------------------

def ok_run(data):
    out = pc.MARK + " " + json.dumps(data)
    return lambda argv, timeout, env=None: CommandResult(tuple(argv), 0, out + "\n", "")


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(classes="x"),
    lambda d: d["classes"][0].update({"class": "a b; rm -rf"}),
    lambda d: d["classes"][0]["variants"][0]["attrs"].update({"x": {"nested": 1}}),
    lambda d: d["classes"][0]["variants"][0].update({"count": -1}),
    lambda d: d["classes"][0]["variants"][0].update({"count": 10 ** 30}),
    lambda d: d["classes"][0]["variants"][0]["attrs"].update({"x\x00": 1}),
    lambda d: d["classes"][0]["variants"][0]["attrs"].update({"x": [[1]]}),
    lambda d: d["classes"][0]["signature"].append({"name": 5, "default": 1}),
])
def test_output_of_the_wrong_type_or_shape_is_refused_with_a_reason(mutate):
    data = json.loads(json.dumps(load_script().describe(fake_model())))
    mutate(data)
    text, flags, summary, why = pc.for_snapshot("/py", "/snap", runner=ok_run(data))
    assert text == "" and summary == {} and "refused" in why


def test_a_nan_and_an_out_of_range_number_are_refused():
    data = json.loads(json.dumps(load_script().describe(fake_model())))
    data["classes"][0]["variants"][0]["attrs"]["x"] = 10 ** 30
    assert pc.for_snapshot("/py", "/snap", runner=ok_run(data))[3]
    nan = pc.MARK + ' {"classes": [{"class": "A", "signature": [], "variants": [{"count": 1, "paths": [], "attrs": {"x": NaN}}]}]}'
    assert "refused" in pc.for_snapshot("/py", "/snap", runner=lambda a, t, env=None: CommandResult(tuple(a), 0, nan, ""))[3]


@pytest.mark.parametrize("result,word", [
    (CommandResult(("p",), 1, "", "boom"), "exited"),
    (CommandResult(("p",), 0, "no marker here\n", ""), "0 result lines"),
    (CommandResult(("p",), 0, pc.MARK + " not json\n", ""), "not JSON"),
    (CommandResult(("p",), 0, pc.MARK + " [1]\n", ""), "not an object"),
    (CommandResult(("p",), 0, (pc.MARK + " {}\n") * 2, ""), "2 result lines"),
    (CommandResult(("p",), None, "", "", timed_out=True), "exited"),
])
def test_a_failed_run_gives_a_reason_and_no_text(result, word):
    text, flags, summary, why = pc.for_snapshot("/py", "/snap", runner=lambda a, t, env=None: result)
    assert text == "" and word in why


def test_no_interpreter_is_a_reason():
    assert pc.for_snapshot(None, "/snap")[3] == "no reference interpreter"


def test_the_interpreter_runs_isolated_offline_and_without_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    seen = {}

    def runner(argv, timeout, env=None):
        seen.update(argv=argv, env=env, timeout=timeout)
        return CommandResult(tuple(argv), 1, "", "")
    pc.run("/py", "/snap", runner=runner)
    assert seen["argv"][:2] == ["/py", "-I"] and "HF_TOKEN" not in seen["env"] and seen["env"]["HF_HUB_OFFLINE"] == "1"
    assert "ttnn" not in " ".join(seen["argv"])


# ---- the facts file carries the section and pins the sibling file -----------------------------------------------------------------

def facts_with_construction(tmp_path):
    text, flags, summary, why = pc.for_snapshot("/py", "/snap", runner=ok_run(json.loads(json.dumps(load_script().describe(fake_model())))))
    assert not why
    return port_facts.build(snapshot_with(tmp_path, TENSORS), construction=summary), text, summary


def test_the_facts_file_has_an_effective_construction_section_with_the_flags_and_the_hash(tmp_path):
    facts, text, summary = facts_with_construction(tmp_path)
    section = facts["effective_construction"]
    assert section["file"] == pc.FILE_NAME and section["sha256"] == summary["sha256"]
    assert any("activation: resolved 'sigmoid'" in f for f in section["resolved_vs_default"])
    assert "elide" in section["note"] and len(json.dumps(facts, indent=2)) + 1 <= port_facts.MAX_CHARS


def test_the_facts_file_is_the_same_bytes_for_the_same_inputs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _, _ = facts_with_construction(tmp_path / "a")
    b, _, _ = facts_with_construction(tmp_path / "b")
    assert json.dumps(a, indent=2) == json.dumps(b, indent=2)


def test_a_summary_that_names_another_file_or_a_bad_hash_is_not_carried(tmp_path):
    snap = snapshot_with(tmp_path, TENSORS)
    assert "effective_construction" not in port_facts.build(snap, construction={"file": "other.txt"})
    assert "effective_construction" not in port_facts.build(snap, construction="x")
    bad = port_facts.build(snap, construction={"file": pc.FILE_NAME, "sha256": "zz", "bytes": True, "resolved_vs_default": [3, "ok"]})
    assert bad["effective_construction"]["sha256"] is None and bad["effective_construction"]["bytes"] is None
    assert bad["effective_construction"]["resolved_vs_default"] == ["ok"]


def test_the_task_text_points_at_the_section_only_when_the_file_exists():
    assert pc.FILE_NAME in port_facts.prepare_note(facts=True, skeleton=True, construction=True)
    assert "Resolved vs class default" in port_facts.prepare_note(facts=True, skeleton=True, construction=True)
    assert pc.FILE_NAME not in port_facts.prepare_note(facts=True, skeleton=True)


def test_the_excerpts_header_points_at_the_file():
    from orchard import port_excerpts
    assert pc.FILE_NAME in Path(port_excerpts.__file__).read_text()


# ---- the supervisor's step: the file is written and pinned in the ledger ------------------------------------------------------------

def test_the_supervisor_writes_the_file_and_the_ledger_pins_its_hash_and_the_facts_pin_the_same_hash(tmp_path):
    entries = []
    ledger = SimpleNamespace(append=lambda *a, **k: entries.append((a, k)))
    stage = tmp_path / "stages" / "2"
    stage.mkdir(parents=True)
    data = json.loads(json.dumps(load_script().describe(fake_model())))
    written, summary = supervisor.write_effective_construction(ledger, tmp_path, SimpleNamespace(number=2), stage, tmp_path, "/py", runner=ok_run(data))
    assert written
    decision = next(k for a, k in entries if k.get("decision") == "effective construction written")
    import hashlib
    on_disk = hashlib.sha256((stage / pc.FILE_NAME).read_bytes()).hexdigest()
    assert decision["construction"]["sha256"] == on_disk == summary["sha256"]
    assert decision["construction"]["path"] == "stages/2/" + pc.FILE_NAME and decision["resolved_vs_default"] >= 1


def test_a_failing_script_leaves_no_file_and_a_decision_with_the_reason(tmp_path):
    entries = []
    ledger = SimpleNamespace(append=lambda *a, **k: entries.append((a, k)))
    stage = tmp_path / "stages" / "2"
    stage.mkdir(parents=True)
    written, summary = supervisor.write_effective_construction(
        ledger, tmp_path, SimpleNamespace(number=2), stage, tmp_path, "/py", runner=lambda a, t, env=None: CommandResult(tuple(a), 3, "", "x"))
    assert not written and summary is None and not (stage / pc.FILE_NAME).exists()
    assert entries[0][1]["decision"] == "effective construction not written" and "code 3" in entries[0][1]["problem"]


def test_no_reference_interpreter_means_no_step_and_no_ledger_entry(tmp_path):
    entries = []
    ledger = SimpleNamespace(append=lambda *a, **k: entries.append((a, k)))
    assert supervisor.write_effective_construction(ledger, tmp_path, SimpleNamespace(number=2), tmp_path, tmp_path, None) == (False, None) and not entries


def test_the_seed_step_carries_the_section_into_the_facts_and_the_task(tmp_path, monkeypatch):
    snap = snapshot_with(tmp_path, TENSORS)
    sd = tmp_path / "stages" / "2"
    sd.mkdir(parents=True)
    data = json.loads(json.dumps(load_script().describe(fake_model())))
    real = pc.for_snapshot
    monkeypatch.setattr(pc, "for_snapshot", lambda python, snapshot, **kw: real(python, snapshot, runner=ok_run(data)))
    entries = []
    stub = SimpleNamespace(inputs={"model": str(snap), "reference_python": "/nonexistent-python"}, run_dir=tmp_path,
                           ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    supervisor.Supervisor._seed_full_port(stub, SimpleNamespace(number=2), sd)
    facts = json.loads((sd / "port_facts.json").read_text())
    import hashlib
    assert facts["effective_construction"]["sha256"] == hashlib.sha256((sd / pc.FILE_NAME).read_bytes()).hexdigest()
    assert stub._port_seed["construction"] is True
    decisions = [k.get("decision") for a, k in entries]
    assert "effective construction written" in decisions and "port facts written" in decisions


# ---- the real reference interpreter on the tiny checkpoint ----------------------------------------------------------------------------

def _reference_python():
    """The reference interpreter: $ORCHARD_REFERENCE_PYTHON, else the default venv location. None when neither exists."""
    for c in (os.environ.get("ORCHARD_REFERENCE_PYTHON"), str(Path.home() / "orchard-venvs" / "reference" / "bin" / "python")):
        if c and Path(c).is_file():
            return c
    return None


def _tiny_snapshot():
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--local--tiny-qwen4-dense" / "snapshots"
    return next((p for p in sorted(root.glob("*")) if (p / "config.json").is_file()), None) if root.is_dir() else None


@pytest.mark.skipif(_reference_python() is None or _tiny_snapshot() is None,
                    reason="needs the reference interpreter ($ORCHARD_REFERENCE_PYTHON or the default venv) and the cached tiny Qwen4 checkpoint (models--local--tiny-qwen4-dense)")
@pytest.mark.timeout(300)
def test_the_tiny_qwen4_shows_sigmoid_for_the_gated_norm_in_the_real_reference():
    python = _reference_python()
    text, flags, summary, why = pc.for_snapshot(python, _tiny_snapshot())
    assert not why, why
    assert any("RMSNormGated.activation: resolved 'sigmoid', class default 'silu'" in f for f in flags)
    assert any("Conv1d.dilation: resolved [3]" in f for f in flags) and any("RMSNorm.group_size: resolved 64" in f for f in flags)
    again = pc.for_snapshot(python, _tiny_snapshot())[0]
    assert again == text                                    # two runs, the same bytes
