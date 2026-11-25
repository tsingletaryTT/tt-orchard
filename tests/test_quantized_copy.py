# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 63: a quantized copy of a supported model (FP8, NVFP4, GPTQ, AWQ) is a full port to stage 0, and the pause says so as a full port only.

Nobody has served a quantized checkpoint on a Tenstorrent chip with this harness, so the class stays `full-port` until a hardware test of one format says what
the runtime accepts, converts or refuses. What the harness can do without a board is say, in the pause, what stage 0 saw: tensors in a quantized format that the
nearest model keeps in bf16. `orchard/quantized_copy.py` reads that from stage 0's own evidence (tensor-compare.json) and the model's `quantization_config`.
It is a hint in the pause text and a ledger decision. It never changes the class or the path."""
import json
from pathlib import Path

import pytest

from orchard import blocked, quantized_copy as qc, status, supervisor

BF16 = {"BF16": 400}
MODEL, NEAREST = "Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.8-27B"


def det(model_dtypes=None, base_dtypes=None, only_in_model=(), quant_config=None, nearest=NEAREST):
    return qc.detect(model_dtypes=model_dtypes or {}, base_dtypes=base_dtypes if base_dtypes is not None else BF16, only_in_model=list(only_in_model),
                     quant_config=quant_config, nearest_model=nearest)


# ---- the format -------------------------------------------------------------------------------------------------------

def test_fp8_weights_with_scales_against_a_bf16_base_are_an_fp8_copy():
    got = det({"BF16": 150, "F8_E4M3": 250}, only_in_model=[f"model.layers.{i}.mlp.up_proj.weight_scale_inv" for i in range(3)])
    assert got["format"] == "fp8" and got["nearest"] == NEAREST and any("F8_E4M3" in s for s in got["signals"])


def test_a_float8_dtype_alone_is_enough_for_fp8():
    assert det({"BF16": 10, "F8_E5M2": 5})["format"] == "fp8"


@pytest.mark.parametrize("config,fmt", [({"quant_method": "fp8"}, "fp8"), ({"quant_method": "FP8"}, "fp8"), ({"quant_method": "gptq", "bits": 4}, "gptq"),
                                        ({"quant_method": "awq"}, "awq"), ({"quant_method": "modelopt", "quant_algo": "NVFP4"}, "nvfp4"),
                                        ({"quant_method": "nvfp4"}, "nvfp4"), ({"quant_method": "mxfp8"}, "mxfp8"),
                                        ({"quant_method": "compressed-tensors"}, "compressed-tensors")])
def test_the_models_quantization_config_names_the_format(config, fmt):
    assert det({"BF16": 400}, quant_config=config)["format"] == fmt


def test_gptq_tensor_names_without_a_config_are_a_gptq_copy():
    got = det({"BF16": 100, "I32": 300}, only_in_model=["model.layers.0.mlp.up_proj.qweight", "model.layers.0.mlp.up_proj.qzeros", "model.layers.0.mlp.up_proj.g_idx"])
    assert got["format"] == "gptq"


def test_nvfp4_scale_tensors_with_packed_bytes_are_an_nvfp4_copy():
    got = det({"BF16": 100, "U8": 300, "F8_E4M3": 300}, only_in_model=["model.layers.0.mlp.up_proj.weight_scale_2", "model.layers.0.mlp.up_proj.input_scale"])
    assert got["format"] == "nvfp4"


def test_the_config_wins_over_the_tensor_names():
    got = det({"BF16": 100, "F8_E4M3": 300}, only_in_model=["a.weight_scale_inv"], quant_config={"quant_method": "gptq"})
    assert got["format"] == "gptq"


# ---- what is not a quantized copy ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("model_dtypes,base_dtypes,only", [
    ({"BF16": 400}, BF16, []),                                          # the same dtype
    ({"F16": 400}, BF16, []),                                           # a float16 copy
    ({"BF16": 400}, BF16, ["model.layers.0.lora_A.weight"]),            # an adapter, which has no quantization
    ({"F8_E4M3": 300}, {"F8_E4M3": 300}, ["x.weight_scale_inv"]),       # the nearest model is quantized the same way
])
def test_a_model_the_nearest_model_matches_in_kind_is_not_a_quantized_copy(model_dtypes, base_dtypes, only):
    assert det(model_dtypes, base_dtypes, only) is None


def test_a_nearest_model_that_is_itself_quantized_gives_none():
    assert det({"F8_E4M3": 300, "BF16": 10}, {"F8_E4M3": 300, "BF16": 10}, ["a.weight_scale_inv"]) is None


@pytest.mark.parametrize("bad", [None, [], "x", 3])
def test_a_quantization_config_that_is_not_an_object_is_ignored(bad):
    assert det({"BF16": 400}, quant_config=bad) is None


def test_an_unknown_quant_method_is_named_as_given_but_bounded():
    got = det({"BF16": 400}, quant_config={"quant_method": "weird-fmt-" + "x" * 500})
    assert got["format"].startswith("weird-fmt-") and len(got["format"]) <= 40


@pytest.mark.parametrize("method", [None, 5, ["fp8"], "", "  ", "a\nb", "a\0b"])
def test_a_quant_method_that_is_not_a_clean_word_is_no_signal(method):
    assert det({"BF16": 400}, quant_config={"quant_method": method}) is None


def test_the_signals_are_bounded_and_say_what_was_seen():
    got = det({"BF16": 1, "F8_E4M3": 2}, only_in_model=[f"l{i}.weight_scale_inv" for i in range(5000)])
    assert len(got["signals"]) <= 4 and all(len(s) <= 200 for s in got["signals"])


def test_the_sentence_names_the_format_the_nearest_model_and_the_open_question():
    text = qc.sentence({"format": "fp8", "nearest": NEAREST, "signals": ["F8_E4M3 tensors"]})
    assert "quantized copy" in text and "FP8" in text and NEAREST in text and "hardware" in text and "issue 63" in text
    # the words status.classify_pause looks for must not turn it into another kind of pause
    for banned in ("scoring route", "audio path", "restricted head", "full-port route", "failed after escalation", "is not escalated"):
        assert banned not in text


def test_the_pause_reason_keeps_the_full_port_prefix_and_classifies_as_a_full_port():
    reason = supervisor.full_port_reason({"format": "gptq", "nearest": NEAREST, "signals": ["qweight"]})
    assert reason.startswith(supervisor.FULL_PORT) and "quantized copy" in reason
    assert supervisor.is_full_port_pause(reason) and supervisor.is_full_port_pause(supervisor.FULL_PORT)
    assert status.classify_pause({"reason": reason})[0] == "full_port"
    assert blocked.block_code({"decision": "pause", "reason": reason}) == "needs-new-model-code"


def test_a_reason_that_only_quotes_the_prefix_later_is_not_a_full_port_pause():
    assert not supervisor.is_full_port_pause("blocked: " + supervisor.FULL_PORT) and not supervisor.is_full_port_pause(supervisor.FULL_PORT + "x")
    assert supervisor.full_port_reason(None) == supervisor.FULL_PORT


# ---- from a run directory -------------------------------------------------------------------------------------------------

def make_run(tmp_path, model_dtypes, only_in_model, quant_config=None, base_dtypes=None, nearest=NEAREST):
    run, snap = tmp_path / "run", tmp_path / "snap"
    ev = run / "stages" / "0" / "evidence"
    ev.mkdir(parents=True)
    snap.mkdir()
    (ev / "tensor-compare.json").write_text(json.dumps({"model": {"dtype_histogram": model_dtypes}, "base": {"dtype_histogram": base_dtypes or BF16},
                                                       "raw_name_sets": {"common": 1, "only_in_model": only_in_model, "only_in_base": []}}))
    (run / "stages" / "0" / "delta.json").write_text(json.dumps({"model": MODEL, "nearest_model": nearest + "@abc", "path": "full-port", "class": "full-port"}))
    cfg = {"architectures": ["Qwen3_5ForCausalLM"]}
    if quant_config is not None:
        cfg["quantization_config"] = quant_config
    (snap / "config.json").write_text(json.dumps(cfg))
    return run, snap


def test_from_run_reads_the_evidence_and_the_config(tmp_path):
    run, snap = make_run(tmp_path, {"BF16": 100, "F8_E4M3": 300}, ["a.weight_scale_inv"], {"quant_method": "fp8"})
    got = qc.from_run(run, snap)
    assert got["format"] == "fp8" and got["nearest"] == NEAREST            # the revision after @ is not part of the id


def test_from_run_gives_none_for_an_unquantized_pair(tmp_path):
    run, snap = make_run(tmp_path, {"BF16": 400}, [])
    assert qc.from_run(run, snap) is None


@pytest.mark.parametrize("breakage,expected", [("no_evidence", None), ("bad_json", None), ("link", None), ("no_delta", "fp8"), ("no_config", "fp8"), ("config_list", "fp8")])
def test_from_run_never_raises_on_a_run_it_cannot_read(tmp_path, breakage, expected):
    """An unreadable evidence file gives none; a missing delta or config only loses a signal (the tensors still say FP8)."""
    run, snap = make_run(tmp_path, {"BF16": 100, "F8_E4M3": 300}, ["a.weight_scale_inv"], {"quant_method": "fp8"})
    ev = run / "stages" / "0" / "evidence" / "tensor-compare.json"
    if breakage == "no_evidence":
        ev.unlink()
    elif breakage == "bad_json":
        ev.write_text("{nope")
    elif breakage == "no_delta":
        (run / "stages" / "0" / "delta.json").unlink()
    elif breakage == "no_config":
        (snap / "config.json").unlink()
    elif breakage == "config_list":
        (snap / "config.json").write_text("[]")
    else:
        ev.unlink()
        ev.symlink_to(tmp_path / "elsewhere")
    got = qc.from_run(run, snap)
    assert (got["format"] if got else None) == expected


def test_from_run_without_evidence_does_not_turn_a_config_alone_into_an_answer(tmp_path):
    run, snap = make_run(tmp_path, {"BF16": 400}, [], {"quant_method": "gptq"})
    (run / "stages" / "0" / "evidence" / "tensor-compare.json").unlink()
    assert qc.from_run(run, snap) is None


def test_the_models_config_alone_is_enough_when_the_tensors_say_nothing(tmp_path):
    run, snap = make_run(tmp_path, {"BF16": 400}, [], {"quant_method": "gptq"})
    got = qc.from_run(run, snap)
    assert got["format"] == "gptq" and got["signals"] == ["quantization_config says gptq"]


def test_an_evidence_file_over_the_bound_is_not_read(tmp_path):
    run, snap = make_run(tmp_path, {"BF16": 100, "F8_E4M3": 300}, ["a.weight_scale_inv"])
    path = run / "stages" / "0" / "evidence" / "tensor-compare.json"
    with open(path, "ab") as f:
        f.write(b" " * (64 * 1024 * 1024 + 8))          # valid JSON with trailing space, a few bytes over 64 MiB
    assert qc.from_run(run, snap) is None


# ---- the pause in a run -----------------------------------------------------------------------------------------------------

def quantized_run(tmp_path, *, quant=True):
    from run_fakes import DELTA, FILES
    from test_supervisor import Rig
    from test_supervisor_full_port import make, never_wait, with_snapshot
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    if quant:
        (tmp_path / "snapshot" / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    r.args.unattended = True
    compare = {"model": {"dtype_histogram": {"BF16": 100, "F8_E4M3": 300} if quant else {"BF16": 400}}, "base": {"dtype_histogram": BF16},
               "raw_name_sets": {"common": 1, "only_in_model": ["a.weight_scale_inv"] if quant else [], "only_in_base": []}}
    make(r, over={(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "path": "full-port", "class": "full-port"},
                               "evidence/tensor-compare.json": compare}})
    never_wait(r)
    return r


def test_a_quantized_copy_pauses_with_the_note_and_keeps_the_full_port_class(tmp_path):
    from orchard.supervisor import EXIT_BLOCKED
    r = quantized_run(tmp_path)
    try:
        assert r.run() == EXIT_BLOCKED
        pauses = [d["reason"] for d in r.decisions() if d.get("decision") == "pause"]
        assert len(pauses) == 1 and pauses[0].startswith(supervisor.FULL_PORT + ". ") and "FP8 quantized copy" in pauses[0] and "Qwen/Qwen3.8-27B" in pauses[0]
        noted = [d for d in r.decisions() if d.get("decision") == supervisor.QUANTIZED_COPY_NOTED]
        assert len(noted) == 1 and noted[0]["format"] == "fp8" and noted[0]["nearest"] == "Qwen/Qwen3.8-27B" and noted[0]["signals"]
        end = r.ends(0)[0]
        assert end["class"] == "full-port" and end["path"] == "full-port"
        assert blocked.block_code({"decision": "pause", "reason": pauses[0]}) == "needs-new-model-code"
    finally:
        r.close()


def test_a_full_port_model_that_is_not_quantized_keeps_the_plain_message_and_notes_nothing(tmp_path):
    from orchard.supervisor import EXIT_BLOCKED
    r = quantized_run(tmp_path, quant=False)
    try:
        assert r.run() == EXIT_BLOCKED
        assert [d["reason"] for d in r.decisions() if d.get("decision") == "pause"] == [supervisor.FULL_PORT]
        assert not [d for d in r.decisions() if d.get("decision") == supervisor.QUANTIZED_COPY_NOTED]
    finally:
        r.close()


def test_the_pause_after_the_note_is_still_an_acknowledged_full_port_pause_on_a_resume(tmp_path):
    reason = supervisor.full_port_reason({"format": "fp8", "nearest": NEAREST, "signals": ["x"]})
    assert supervisor.is_full_port_pause(reason)


# ---- more of what detect and the sentence promise -------------------------------------------------------------------------------------

def test_nvfp4_scale_names_win_over_block_scale_names_even_when_there_are_fewer():
    names = [f"l{i}.weight_scale_inv" for i in range(5)] + ["l0.weight_scale_2"]
    assert det({"BF16": 10, "U8": 50, "F8_E4M3": 50}, only_in_model=names)["format"] == "nvfp4"


def test_block_scale_names_alone_are_an_fp8_copy():
    got = det({"BF16": 400}, only_in_model=["l0.weight_scale_inv", "l1.weight_scale_inv"])
    assert got["format"] == "fp8" and got["signals"] and "2 tensors only the model has end in weight_scale_inv" in got["signals"][0]


def test_a_quant_algo_that_is_very_long_is_cut_in_the_signal():
    got = det({"BF16": 400}, quant_config={"quant_method": "modelopt", "quant_algo": "NVFP4" + "x" * 5000})
    assert got["format"] == "nvfp4" and all(len(s) <= qc.SIGNAL_CHARS for s in got["signals"])


def test_names_that_are_not_strings_are_skipped():
    got = det({"BF16": 400}, only_in_model=[None, 5, ["a"], "l0.weight_scale_inv"])
    assert got["format"] == "fp8"


@pytest.mark.parametrize("fmt,article", [("fp8", "an"), ("nvfp4", "an"), ("mxfp8", "an"), ("awq", "an"), ("gptq", "a"), ("compressed-tensors", "a")])
def test_the_sentence_uses_the_right_article(fmt, article):
    text = qc.sentence({"format": fmt, "nearest": NEAREST, "signals": []})
    assert f"looks like {article} {qc.LABELS[fmt]} quantized copy" in text


def test_the_sentence_says_the_class_stays_a_full_port():
    assert "the class stays full port until a hardware test" in qc.sentence({"format": "fp8", "nearest": NEAREST, "signals": []})


def test_a_run_whose_model_config_alone_says_quantized_gets_the_note(tmp_path):
    from orchard.supervisor import EXIT_BLOCKED
    r = quantized_run(tmp_path, quant=False)
    try:
        (tmp_path / "snapshot" / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "gptq"}}))
        assert r.run() == EXIT_BLOCKED
        pauses = [d["reason"] for d in r.decisions() if d.get("decision") == "pause"]
        assert len(pauses) == 1 and "a GPTQ quantized copy" in pauses[0]
    finally:
        r.close()


def cache_snapshot(tmp_path, config):
    """A snapshot laid out as the Hugging Face cache does: files in snapshots/<revision> are links into the model directory's blobs/."""
    model_dir = tmp_path / "models--Qwen--Qwen3.8-27B-FP8"
    snap, blobs = model_dir / "snapshots" / ("a" * 40), model_dir / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "abc").write_text(json.dumps(config))
    (snap / "config.json").symlink_to(blobs / "abc")
    return snap


def test_a_config_stored_as_a_hugging_face_cache_link_is_read(tmp_path):
    """In the cache every file of a snapshot is a link into blobs/. The real FP8 copy's quantization_config was lost until the link was followed."""
    run, _ = make_run(tmp_path, {"BF16": 400}, [])
    snap = cache_snapshot(tmp_path, {"quantization_config": {"quant_method": "fp8"}})
    assert qc.from_run(run, snap)["format"] == "fp8"


def test_a_config_link_that_leaves_the_snapshot_and_its_cache_is_not_followed(tmp_path):
    run, _ = make_run(tmp_path, {"BF16": 400}, [])
    snap = cache_snapshot(tmp_path, {})
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    (snap / "config.json").unlink()
    (snap / "config.json").symlink_to(elsewhere)
    assert qc.from_run(run, snap) is None


def test_a_link_in_the_run_directory_is_still_refused(tmp_path):
    run, snap = make_run(tmp_path, {"BF16": 100, "F8_E4M3": 300}, ["a.weight_scale_inv"])
    delta = run / "stages" / "0" / "delta.json"
    real = tmp_path / "real-delta.json"
    real.write_text(delta.read_text().replace(NEAREST, "other/model"))
    delta.unlink()
    delta.symlink_to(real)
    assert qc.from_run(run, snap)["nearest"] == "the nearest model"          # the delta was refused, so its nearest model is not read
