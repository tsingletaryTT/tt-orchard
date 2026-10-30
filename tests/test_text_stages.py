# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The gates of the text stages recompute what the result files claim (orchard/text_stages.py). Each test takes the files of a finished fake run, damages
one thing, and expects the gate to name it. The undamaged run passes all three gates, so a damaged one that passes would be a gate that checks nothing."""
import json
import shutil
from pathlib import Path

import pytest

from orchard import text_runtime as rt, text_stages as st
from test_supervisor_text_runtime import CONFIG, MODEL, bench_json, make_environment, trig  # noqa: F401  (the fixture)
from text_fixtures import base_arrays, perturb, write_run
from orchard import text_reference as tr


@pytest.fixture
def done(trig):
    make_environment(trig)
    assert trig.run() == 0
    return trig


def dirs(r):
    d = r.run_dir
    return d / "stages" / "1", d / "stages" / "2", d / "stages" / "6", d


def gates(r):
    s1, s2, s6, d = dirs(r)
    return (st.reference_problems(s1, d), st.chip_problems(s2, d, s1), st.bench_problems(s6, d, s1))


def edit(path, fn):
    data = json.loads(Path(path).read_text())
    fn(data)
    Path(path).write_text(json.dumps(data))


def test_an_undamaged_run_passes_every_gate(done):
    assert gates(done) == ([], [], [])


# ---- stage 1 --------------------------------------------------------------------------------------------------------------

def test_a_recorded_floor_that_differs_from_the_arrays_is_refused(done):
    s1, *_ = dirs(done)
    edit(s1 / rt.REFERENCE_FILE, lambda d: d["floor"]["floor_bfloat16"].update(worst=0.5))
    assert any("floor" in p for p in gates(done)[0])


def test_a_changed_array_is_refused_by_its_hash(done):
    s1, *_ = dirs(done)
    f = next((s1 / "reference" / "bfloat16" / "arrays").glob("layer__layer_01.npy"))
    f.write_bytes(f.read_bytes()[:-4] + b"\x00\x00\x00\x00")
    assert any("sha256" in p or "does not match" in p for p in gates(done)[0])


def test_a_floor_run_on_another_token_path_is_refused(done):
    s1, _, _, d = dirs(done)
    other = base_arrays()
    other[("other", tr.TOKENS)] = other[("other", tr.TOKENS)][::-1].copy()
    shutil.rmtree(s1 / "reference" / "bfloat16")
    write_run(s1 / "reference" / "bfloat16", other, dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    assert any("token path" in p for p in gates(done)[0])


def test_a_reference_run_with_a_block_missing_is_refused(done):
    s1, _, _, d = dirs(done)
    arrays = {k: v for k, v in base_arrays().items() if k[1] != "layer_03"}
    shutil.rmtree(s1 / "reference" / "float32")
    write_run(s1 / "reference" / "float32", arrays, dtype="float32", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    assert any("layer_03" in p for p in gates(done)[0])


def test_a_run_of_another_configuration_is_refused(done):
    s1, *_ = dirs(done)
    (done.snap / "config.json").write_text(json.dumps(dict(CONFIG, rms_norm_eps=1e-6)))
    shutil.rmtree(s1 / "reference" / "bfloat16")
    write_run(s1 / "reference" / "bfloat16", perturb(base_arrays(), 0.01, seed=2), dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT,
              config_dir=done.snap)
    (done.snap / "config.json").write_text(json.dumps(CONFIG))
    assert any("configuration" in p for p in gates(done)[0])


def test_a_run_in_the_wrong_dtype_is_refused(done):
    s1, *_ = dirs(done)
    shutil.rmtree(s1 / "reference" / "bfloat16")
    write_run(s1 / "reference" / "bfloat16", perturb(base_arrays(), 0.01, seed=2), dtype="float32", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT,
              config_dir=done.snap)
    assert any("dtype" in p for p in gates(done)[0])


def test_a_run_for_another_prompt_is_refused(done):
    s1, *_ = dirs(done)
    shutil.rmtree(s1 / "reference" / "native")
    write_run(s1 / "reference" / "native", perturb(base_arrays(), 0.01, seed=2), dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt="another prompt",
              config_dir=done.snap)
    assert any("prompt" in p for p in gates(done)[0])


def test_a_greedy_path_with_one_distinct_token_is_refused(done):
    s1, *_ = dirs(done)
    for sub, dt in (("float32", "float32"), ("bfloat16", "bfloat16"), ("native", "bfloat16")):
        arrays = base_arrays()
        arrays[("other", tr.TOKENS)] = arrays[("other", tr.TOKENS)] * 0 + 3
        shutil.rmtree(s1 / "reference" / sub)
        write_run(s1 / "reference" / sub, arrays, dtype=dt, model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    assert any("distinct token" in p for p in gates(done)[0])


def test_a_run_with_another_decode_length_is_refused(done):
    s1, *_ = dirs(done)
    shutil.rmtree(s1 / "reference" / "native")
    write_run(s1 / "reference" / "native", perturb(base_arrays(), 0.01, seed=2), dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT,
              config_dir=done.snap, decode_tokens=7)
    assert any("decode length" in p for p in gates(done)[0])


def test_an_array_with_a_value_that_is_not_finite_is_refused(done):
    s1, *_ = dirs(done)
    arrays = perturb(base_arrays(), 0.01, seed=2)
    arrays[("layer", "layer_02")] = arrays[("layer", "layer_02")] * float("nan")
    shutil.rmtree(s1 / "reference" / "native")
    write_run(s1 / "reference" / "native", arrays, dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    assert any("not finite" in p for p in gates(done)[0])


def test_a_perplexity_reference_over_the_wrong_window_is_refused(done):
    s1, *_ = dirs(done)
    edit(s1 / st.PPL_REFERENCE_FILE, lambda d: d.update(tokens=10))
    assert any("perplexity reference covers" in p for p in gates(done)[0])


def test_a_perplexity_reference_that_is_not_a_number_is_refused(done):
    s1, *_ = dirs(done)
    edit(s1 / st.PPL_REFERENCE_FILE, lambda d: d.update(perplexity=-3))
    assert any("positive finite" in p for p in gates(done)[0])


def test_a_record_for_another_runtime_is_refused(done):
    s1, *_ = dirs(done)
    edit(s1 / rt.REFERENCE_FILE, lambda d: d.update(runtime="granite"))
    assert any("runtime" in p for p in gates(done)[0])


# ---- stage 2 --------------------------------------------------------------------------------------------------------------

def test_a_recorded_agreement_that_differs_from_the_run_is_refused(done):
    _, s2, *_ = dirs(done)
    edit(s2 / rt.RESULT_FILE, lambda d: d["variants"]["bf16"]["numeric"]["agreement"].update(matches=0))
    assert any("not what the runs give now" in p for p in gates(done)[1])


def test_a_recorded_perplexity_that_differs_from_the_file_is_refused(done):
    _, s2, *_ = dirs(done)
    edit(s2 / "ppl_chip_bf16.json", lambda d: d.update(perplexity=7.99))
    assert any("bf16" in p for p in gates(done)[1])


def test_a_default_variant_outside_the_bar_fails_the_gate_and_says_why(done):
    _, s2, *_ = dirs(done)
    shutil.rmtree(s2 / "chip" / "bf16")
    bad = perturb(base_arrays(), 0.03, seed=3)
    bad[("layer", "layer_01")] = bad[("layer", "layer_01")] * 3
    write_run(s2 / "chip" / "bf16", bad, dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    problems = gates(done)[1]
    assert any("layer_01" in p for p in problems)


def test_a_variant_outside_the_bar_does_not_fail_the_stage(done):
    _, s2, *_ = dirs(done)
    shutil.rmtree(s2 / "chip" / "mlp8")
    bad = perturb(base_arrays(), 0.03, seed=3)
    bad[("layer", "layer_01")] = bad[("layer", "layer_01")] * 3
    write_run(s2 / "chip" / "mlp8", bad, dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    # the recorded entry still says ok; the gate reports the mismatch, not a failed bar
    problems = gates(done)[1]
    assert problems and all(p.startswith("mlp8") for p in problems) and any("not what the runs give" in p for p in problems)


def test_a_chip_run_of_another_token_path_is_refused(done):
    _, s2, *_ = dirs(done)
    shutil.rmtree(s2 / "chip" / "bf16")
    arrays = perturb(base_arrays(), 0.03, seed=3)
    arrays[("other", tr.TOKENS)] = arrays[("other", tr.TOKENS)][::-1].copy()
    write_run(s2 / "chip" / "bf16", arrays, dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    assert any("token path" in p for p in gates(done)[1])


def test_a_result_with_a_variant_missing_is_refused(done):
    _, s2, *_ = dirs(done)
    edit(s2 / rt.RESULT_FILE, lambda d: d["variants"].pop("mlp8"))
    assert any("one entry for each variant" in p for p in gates(done)[1])


def test_a_per_array_report_that_was_edited_is_refused(done):
    _, s2, *_ = dirs(done)
    edit(s2 / "evidence" / "numeric_report_bf16.json", lambda d: d.update(ok=False))
    assert any("per-array report" in p for p in gates(done)[1])


def test_a_result_that_names_other_reference_runs_is_refused(done):
    _, s2, *_ = dirs(done)
    edit(s2 / rt.RESULT_FILE, lambda d: d["reference_runs"]["reference"].update(manifest_sha256="0" * 64))
    assert any("not stage 1's" in p for p in gates(done)[1])


# ---- stage 6 --------------------------------------------------------------------------------------------------------------

def test_a_benchmark_with_two_measured_runs_is_refused(done):
    *_, s6, _ = (dirs(done)[0], dirs(done)[1], dirs(done)[2], dirs(done)[3])
    edit(s6 / "bench_bf16.json", lambda d: d["runs"].pop())
    problems = gates(done)[2]
    assert any("measured run(s); the rule is 3" in p for p in problems)


def test_a_benchmark_with_no_warmup_run_is_refused(done):
    _, _, s6, _ = dirs(done)
    edit(s6 / "bench_bf16.json", lambda d: d["runs"][0].update(warmup=False))
    assert any("warm-up" in p for p in gates(done)[2])


def test_a_decode_speed_that_is_not_steps_over_time_is_refused(done):
    _, _, s6, _ = dirs(done)
    edit(s6 / "bench_bf16.json", lambda d: d["runs"][2].update(decode_tok_s=99.0))
    assert any("steps over its decode time" in p for p in gates(done)[2])


def test_a_recorded_share_that_differs_from_the_floor_is_refused(done):
    _, _, s6, _ = dirs(done)
    edit(s6 / rt.BENCH_FILE, lambda d: d["variants"]["bf16"].update(share_of_floor_median=0.99))
    assert any("not what its file gives now" in p for p in gates(done)[2])


def test_a_speed_above_the_floor_is_not_a_share(done):
    _, _, s6, _ = dirs(done)
    edit(s6 / "bench_bf16.json", lambda d: [r.update(decode_tok_s=1000.0, decode_s=64 / 1000.0) for r in d["runs"]])
    problems = gates(done)[2]
    assert any("not a share" in p for p in problems)


def test_a_benchmark_result_that_lacks_a_variant_is_refused(done):
    _, _, s6, _ = dirs(done)
    edit(s6 / rt.BENCH_FILE, lambda d: d["variants"].pop("mlp8"))
    assert any("one entry for each variant" in p for p in gates(done)[2])


# ---- stage 0 --------------------------------------------------------------------------------------------------------------

def triage_gate(r):
    from orchard.stages import gate_text_triage
    return gate_text_triage(r.run_dir / "stages" / "0", r.run_dir)


def test_stage_0_passes_on_the_files_the_supervisor_wrote(done):
    assert triage_gate(done).ok


def test_stage_0_refuses_a_delta_that_names_no_registered_runtime(done):
    edit(done.run_dir / "stages" / "0" / "delta.json", lambda d: d["text_runtime"].update(name="nonesuch"))
    assert not triage_gate(done).ok and "registered text runtime" in triage_gate(done).reasons[0]


def test_stage_0_refuses_a_delta_with_no_runtime_record(done):
    edit(done.run_dir / "stages" / "0" / "delta.json", lambda d: d.pop("text_runtime"))
    assert not triage_gate(done).ok


def test_stage_0_refuses_a_model_that_is_not_text(done):
    edit(done.run_dir / "stages" / "0" / "delta.json", lambda d: d["facts"].update(modality="audio"))
    assert any("modality" in r for r in triage_gate(done).reasons)


def test_stage_0_refuses_a_configuration_the_runtime_refused(done):
    edit(done.run_dir / "stages" / "0" / "delta.json", lambda d: d["text_runtime"].update(config_problems=["no kv_lora_rank"]))
    assert any("kv_lora_rank" in r for r in triage_gate(done).reasons)


def test_a_token_path_shorter_than_the_claimed_decode_steps_is_refused_even_when_every_run_shares_it(done):
    s1, _, _, d = dirs(done)
    short = [3, 7]                                            # two tokens for four claimed decode steps
    for sub, dtype in (("float32", "float32"), ("bfloat16", "bfloat16"), ("native", "bfloat16")):
        arrays = base_arrays() if sub == "float32" else perturb(base_arrays(), 0.01, seed=2)
        arrays[("other", tr.TOKENS)] = __import__("numpy").array(short, dtype="float32")
        shutil.rmtree(s1 / "reference" / sub)
        write_run(s1 / "reference" / sub, arrays, dtype=dtype, model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=done.snap)
    assert any("token path has 2 entries" in p for p in gates(done)[0])
