# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0's gate holds a run whose bundle ships orchard-capabilities.json to a delta.json that applied that file (orchard/bundle_capabilities.py).

The stage 0 agent writes triage_config.json, and the capabilities are applied only when it names `nearest_bundle_dir`. A forgotten key
would call the release weights-only by the rule without the file while stage 2 serves the release's own config (prepare_swap.py), so the
gate compares delta.json's `capabilities` record with the bundle's file as the supervisor reads it."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from orchard import bundle_capabilities as bc

CAPS = json.dumps({"schema": 1, "config_keys": {"num_hidden_layers": {"type": "int"}}})


def bundle(tmp_path: Path, text: str | None = CAPS) -> Path:
    d = tmp_path / "models" / "mando2222" / "qwen4"
    d.mkdir(parents=True)
    if text is not None:
        (d / bc.FILE).write_text(text)
    return d


def record(d: Path, **change) -> dict:
    rec = {"bundle_dir": str(d), "file": bc.FILE, "present": True, "applied": True,
           "sha256": hashlib.sha256((d / bc.FILE).read_bytes()).hexdigest() if (d / bc.FILE).is_file() else None, "problems": []}
    rec.update(change)
    return rec


def delta(path="weights-only", **extra) -> dict:
    return {"model": "Qwen/Qwen4-27B@" + "1" * 40, "nearest_model": "local/franken@" + "2" * 40, "path": path, **extra}


def test_no_bundle_input_checks_nothing(tmp_path):
    assert bc.stage0_problems(delta(), None) == []
    assert bc.stage0_problems(delta(), "") == []


def test_no_bundle_input_but_an_applied_record_fails(tmp_path):
    # The stage 0 agent writes triage_config.json, so it can point nearest_bundle_dir at a folder it wrote and get `applied` true. A run
    # with no bundle input has no file the supervisor vouches for, so a present or applied record must not pass (PR 659 review).
    d = bundle(tmp_path)
    for bad in (record(d), {**record(d), "applied": False}, {**record(d), "present": False, "applied": True}):
        got = bc.stage0_problems(delta(capabilities=bad), None)
        assert got and bc.FILE in got[0], bad
        assert bc.stage0_problems(delta(capabilities=bad), "") == got
    assert bc.stage0_problems(delta(capabilities={**record(d), "present": False, "applied": False}), None) == []     # a recorded 'none'


def test_a_bundle_without_the_file_and_a_delta_without_a_record_pass(tmp_path):
    d = bundle(tmp_path, None)
    assert bc.stage0_problems(delta(), str(d)) == []


def test_a_bundle_with_the_file_needs_the_record(tmp_path):
    d = bundle(tmp_path)
    problems = bc.stage0_problems(delta(), str(d))
    assert problems and "nearest_bundle_dir" in problems[0] and bc.FILE in problems[0]


def test_the_matching_record_passes(tmp_path):
    d = bundle(tmp_path)
    assert bc.stage0_problems(delta(capabilities=record(d)), str(d)) == []


def test_a_record_with_another_sha256_fails(tmp_path):
    d = bundle(tmp_path)
    problems = bc.stage0_problems(delta(capabilities=record(d, sha256="0" * 64)), str(d))
    assert problems and "sha256" in problems[0]


def test_the_file_changed_after_the_triage_fails(tmp_path):
    d = bundle(tmp_path)
    rec = record(d)
    (d / bc.FILE).write_text(CAPS.replace("int", "number"))
    assert bc.stage0_problems(delta(capabilities=rec), str(d))


def test_a_record_for_another_bundle_directory_fails(tmp_path):
    d = bundle(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    problems = bc.stage0_problems(delta(capabilities=record(d, bundle_dir=str(other))), str(d))
    assert problems and "directory" in problems[0]


def test_the_same_directory_through_a_link_passes(tmp_path):
    d = bundle(tmp_path)
    link = tmp_path / "link"
    os.symlink(d, link)
    assert bc.stage0_problems(delta(capabilities=record(d, bundle_dir=str(link))), str(d)) == []


def test_a_file_that_was_not_applied_must_leave_the_path_full_port(tmp_path):
    d = bundle(tmp_path)
    bad = record(d, applied=False, problems=["schema must be 1"])
    assert bc.stage0_problems(delta(path="full-port", capabilities=bad), str(d)) == []
    problems = bc.stage0_problems(delta(path="weights-only", capabilities=bad), str(d))
    assert problems and "full-port" in problems[0]


def test_a_record_that_claims_capabilities_for_a_bundle_without_the_file_fails(tmp_path):
    d = bundle(tmp_path, None)
    problems = bc.stage0_problems(delta(capabilities=record(d, sha256="0" * 64)), str(d))
    assert problems and "no " in problems[0]


def test_a_record_that_says_absent_for_a_bundle_with_the_file_fails(tmp_path):
    d = bundle(tmp_path)
    assert bc.stage0_problems(delta(capabilities=record(d, present=False, applied=False, sha256=None)), str(d))


@pytest.mark.parametrize("bad", [None, [], 3, "x", {"present": "yes"}])
def test_a_record_of_the_wrong_shape_fails_with_a_named_reason(tmp_path, bad):
    d = bundle(tmp_path)
    problems = bc.stage0_problems(delta(capabilities=bad), str(d))
    assert problems


@pytest.mark.parametrize("field, value", [("sha256", None), ("sha256", 5), ("sha256", ["x"]), ("bundle_dir", None), ("bundle_dir", "a\0b"),
                                          ("bundle_dir", 7), ("applied", "true"), ("problems", "x")])
def test_each_field_of_the_wrong_type_fails(tmp_path, field, value):
    d = bundle(tmp_path)
    assert bc.stage0_problems(delta(capabilities=record(d, **{field: value})), str(d))


def test_a_linked_file_in_the_bundle_fails(tmp_path):
    d = bundle(tmp_path, None)
    real = tmp_path / "elsewhere.json"
    real.write_text(CAPS)
    os.symlink(real, d / bc.FILE)
    problems = bc.stage0_problems(delta(capabilities=record(d, sha256="0" * 64)), str(d))
    assert problems and "link" in problems[0]


def test_a_fifo_in_the_bundle_fails_without_blocking(tmp_path):
    d = bundle(tmp_path, None)
    os.mkfifo(d / bc.FILE)
    assert bc.stage0_problems(delta(), str(d))


def test_an_oversized_file_fails(tmp_path):
    d = bundle(tmp_path, "x" * (bc.MAX_BYTES + 1))
    problems = bc.stage0_problems(delta(), str(d))
    assert problems and "bytes" in problems[0]


def test_a_bundle_input_that_is_not_a_directory_fails(tmp_path):
    problems = bc.stage0_problems(delta(), str(tmp_path / "gone"))
    assert problems and "directory" in problems[0]


def test_the_limit_matches_the_triage_template():
    import importlib.util
    path = Path(bc.__file__).parent / "skills" / "delta-triage-templates" / "delta_triage.py"
    spec = importlib.util.spec_from_file_location("delta_triage_limit", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CAPABILITIES_MAX_BYTES == bc.MAX_BYTES and mod.CAPABILITIES_FILE == bc.FILE


def test_gate_reasons_reads_delta_json_safely(tmp_path):
    d = bundle(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    assert any("delta.json" in r for r in bc.gate_reasons(stage, str(d)))           # missing
    (stage / "delta.json").write_text('{"path": NaN}')
    assert any("delta.json" in r for r in bc.gate_reasons(stage, str(d)))           # refused by the safe reader
    (stage / "delta.json").write_text(json.dumps(delta(capabilities=record(d))))
    assert bc.gate_reasons(stage, str(d)) == []
    assert any(bc.FILE in x for x in bc.gate_reasons(stage, None))          # no bundle input: the same record is refused (PR 659 review)
    (stage / "delta.json").write_text(json.dumps(delta()))
    assert bc.gate_reasons(stage, None) == []
    (stage / "delta.json").write_text('{"path": NaN}')
    assert bc.gate_reasons(stage, None) == []          # no bundle input and an unreadable delta.json: stage 0's own gate judges that file


# ---- the supervisor's stage 0 gate ---------------------------------------------------------------------------------------------

def test_the_supervisor_adds_the_check_to_stage_0s_gate_only_when_the_run_names_a_bundle(tmp_path):
    from test_supervisor import Rig
    from orchard.ledger import Ledger
    from orchard.stages import STAGES, GateResult
    from orchard.supervisor import build, parse
    from run_fakes import argv
    d = bundle(tmp_path)
    r = Rig(tmp_path)
    try:
        stage = r.run_dir / "stages" / "0"
        stage.mkdir(parents=True)
        (stage / "delta.json").write_text(json.dumps(delta()))
        ok = lambda s, rd: GateResult(True)          # noqa: E731
        spec0 = next(s for s in STAGES if s.number == 0)
        for extra, fails in (([], False), (["--input", f"bundle={d}"], True)):
            args = parse(argv(tmp_path, r.tiers, r.chip_server.endpoint) + extra)
            with Ledger(r.run_dir / "ledger.jsonl") as led:
                sup = build(args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m), home=r.home, containers=r.containers)
                got = sup._with_capabilities(ok)(stage, r.run_dir)
                assert got.ok is (not fails), got.reasons
                if fails:
                    assert any(bc.FILE in x for x in got.reasons)
                    full = sup._gate(spec0)(stage, r.run_dir)          # the gate the stage really uses carries the check
                    assert any(bc.FILE in x for x in full.reasons)
        (stage / "delta.json").write_text(json.dumps(delta(capabilities=record(d))))
        args = parse(argv(tmp_path, r.tiers, r.chip_server.endpoint) + ["--input", f"bundle={d}"])
        with Ledger(r.run_dir / "ledger.jsonl") as led:
            sup = build(args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m), home=r.home, containers=r.containers)
            assert sup._with_capabilities(ok)(stage, r.run_dir).ok
    finally:
        r.close()


def test_the_supervisor_refuses_an_applied_record_when_the_run_has_no_bundle_input(tmp_path):
    from test_supervisor import Rig
    from orchard.ledger import Ledger
    from orchard.stages import STAGES, GateResult
    from orchard.supervisor import build, parse
    from run_fakes import argv
    d = bundle(tmp_path)
    r = Rig(tmp_path)
    try:
        stage = r.run_dir / "stages" / "0"
        stage.mkdir(parents=True)
        (stage / "delta.json").write_text(json.dumps(delta(capabilities=record(d))))
        ok = lambda s, rd: GateResult(True)          # noqa: E731
        spec0 = next(s for s in STAGES if s.number == 0)
        args = parse(argv(tmp_path, r.tiers, r.chip_server.endpoint))
        with Ledger(r.run_dir / "ledger.jsonl") as led:
            sup = build(args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m), home=r.home, containers=r.containers)
            got = sup._with_capabilities(ok)(stage, r.run_dir)
            assert not got.ok and any(bc.FILE in x for x in got.reasons), got.reasons
            assert any(bc.FILE in x for x in sup._gate(spec0)(stage, r.run_dir).reasons)       # the gate stage 0 really uses carries it
            (stage / "delta.json").write_text(json.dumps(delta()))
            assert sup._with_capabilities(ok)(stage, r.run_dir).ok
    finally:
        r.close()


# ---- the evaluator the release-day check calls ---------------------------------------------------------------------------------

def _caps_example() -> dict:
    sys_path_tests()
    from test_delta_triage_capabilities import DROP_CHECK_EXAMPLE
    return json.loads(json.dumps(DROP_CHECK_EXAMPLE))


def sys_path_tests():
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


QWEN4_CFG = {"model_type": "qwen4_exp_text", "architectures": ["Qwen4ExpForCausalLM"], "hidden_size": 5120, "num_attention_heads": 24,
             "head_dim": 256, "num_key_value_heads": 4, "intermediate_size": 17408, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
             "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 248320, "hc_count": 4,
             "hc_lowrank": 640, "output_gate_type": "silu", "hidden_act": "silu", "num_hidden_layers": 20}


def test_config_reasons_is_empty_for_a_covered_release_and_names_each_problem_otherwise():
    caps = _caps_example()
    assert bc.config_reasons(caps, QWEN4_CFG) == []
    got = bc.config_reasons(caps, {**QWEN4_CFG, "num_key_value_heads": 3, "architectures": ["Qwen4ForCausalLM"]})
    assert any("num_key_value_heads" in r for r in got) and any("Qwen4ForCausalLM" in r for r in got)
    assert any("model_type" in r for r in bc.config_reasons(caps, {**QWEN4_CFG, "model_type": "qwen4_text"}))


def test_config_reasons_with_the_bundles_own_config_needs_every_difference_covered():
    caps = _caps_example()
    release = {**QWEN4_CFG, "num_hidden_layers": 64}
    assert bc.config_reasons(caps, release) == []                                        # no dev config: rules only
    got = bc.config_reasons(caps, release, QWEN4_CFG)
    assert any("num_hidden_layers" in r and "does not declare" in r for r in got), got
    caps["keys"]["num_hidden_layers"] = {"min": 4}
    assert bc.config_reasons(caps, release, QWEN4_CFG) == []


@pytest.mark.parametrize("caps, release, dev, words", [
    ({"schema": 2}, QWEN4_CFG, None, "cannot be used"),
    (None, QWEN4_CFG, None, "cannot be used"),
    ("default", [], None, "not a JSON object"),
    ("default", QWEN4_CFG, "x", "not a JSON object"),
])
def test_config_reasons_names_inputs_it_cannot_use(caps, release, dev, words):
    got = bc.config_reasons(_caps_example() if caps == "default" else caps, release, dev)
    assert len(got) == 1 and words in got[0], got


def test_config_reasons_says_what_decide_path_says_about_the_config():
    """The release-day check and stage 0 must not disagree: for each pair the config side of decide_path equals config_reasons."""
    sys_path_tests()
    import test_delta_triage_capabilities as tc
    pairs = [(tc.cfg(), tc.cfg()), (tc.cfg(num_key_value_heads=4), tc.cfg()), (tc.cfg(rms_norm_eps=1e-5), tc.cfg()),
             (tc.cfg(layers=6), tc.cfg(layers=2)), (tc.cfg(model_type="qwen4_text"), tc.cfg()),
             (tc.cfg(rope_parameters={"rope_theta": 0.5}), tc.cfg(rope_parameters={"rope_theta": 1.0})),
             (tc.cfg(architectures=["Other"]), tc.cfg())]
    for m, b in pairs:
        reasons, _ = tc.judge(m, b)
        config_side = [r for r in reasons if r.startswith(("text config", "the new model's architecture"))]
        assert config_side == bc.config_reasons(tc.caps(), m, b), (m, b)
