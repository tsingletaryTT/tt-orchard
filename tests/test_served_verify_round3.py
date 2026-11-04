# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 370, review of PR 443, round 3.

1. A progress bar belongs to the component whose announcement it goes with, in both announcement forms: a bar that follows a `Loading N layers`
   line is that line's, and a completion statement for another count must not take it.
2. A block-count override belongs to the cache report it precedes, and so does the concurrency line that follows that report.
3. The structured facts of the two results reach the boot evidence (`evidence["engine_facts"]`), not only the verdict and the sentence."""
from orchard import served_verify as sv

CFG64 = {"num_hidden_layers": 64}


def run(log, **kw):
    kw.setdefault("config", CFG64)
    kw.setdefault("max_model_len", 1024)
    kw.setdefault("max_num_seqs", 1)
    return {r["name"]: r for r in sv.check_engine_log(log, **kw)}


def _loading(n):
    return f"(EngineCore pid=101) 2026-11-03 12:39:15.043 | INFO     | models.x.tt.model:__init__:174 - Loading {n} transformer layers (indices=[0])\n"


def _bar(done, total):
    return f"Loading layers:   0%|          | 0/{total} [00:00<?, ?it/s]\rLoading layers: 100%|██████████| {done}/{total} [04:23<00:00,  4.12s/it]\n"


def _override(n):
    return f"(EngineCore pid=101) INFO 11-03 12:43:46 [kv_cache_utils.py:2115] Overriding num_gpu_blocks={n} with num_gpu_blocks_override={n}\n"


def _cache(tokens):
    return f"(EngineCore pid=101) INFO 11-03 12:43:46 [kv_cache_utils.py:2177] GPU KV cache size: {tokens} tokens\n"


def _conc(per_request, factor):
    return f"(EngineCore pid=101) INFO 11-03 12:43:46 [kv_cache_utils.py:2178] Maximum concurrency for {per_request} tokens per request: {factor}x\n"


# ---- 1. bars across the two announcement forms --------------------------------------------------------------------------------------

def test_a_completion_statement_for_another_count_does_not_take_an_earlier_loading_lines_completed_bar():
    log = _loading(32) + _bar(32, 32) + "loaded 64 layers\n"          # the 32/32 bar is the 32-layer component's
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers"] == 64 and r["layers_completed"] is None and "no progress bar" in r["detail"]


def test_a_completion_statement_for_the_same_count_takes_the_bar_of_the_loading_line_it_completes():
    log = _loading(64) + _bar(64, 64) + "loaded 64 layers\n"
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] == 64 and "64/64" in r["detail"]


def test_a_short_bar_of_the_same_component_still_fails_through_its_completion_statement():
    log = _loading(64) + _bar(40, 64) + "loaded 64 layers\n"
    r = run(log)["layers_loaded"]
    assert r["state"] == "fail" and r["layers_completed"] == 40 and "40/64" in r["detail"]


def test_a_completion_statement_takes_its_own_bar_when_an_earlier_loading_lines_bar_lies_before_it():
    log = _loading(32) + _bar(32, 32) + _bar(64, 64) + "loaded 64 layers\n"       # the 32/32 bar is the earlier component's, the 64/64 bar is ours
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] == 64
    short = _loading(32) + _bar(32, 32) + _bar(40, 64) + "loaded 64 layers\n"
    assert run(short)["layers_loaded"]["state"] == "fail"


def test_a_completion_statement_after_another_completion_statement_takes_only_the_bar_between_them():
    log = _loading(32) + _bar(32, 32) + "loaded 32 layers\n" + "loaded 64 layers\n"
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] is None


# ---- 2. an override belongs to the cache report it precedes --------------------------------------------------------------------------

def test_an_override_after_the_selected_cache_line_is_not_that_reports_override():
    log = _cache("64") + _override(4100)
    r = run(log, max_model_len=16)["kv_cache"]
    assert r["capacity_basis"] == "engine" and r["override_blocks"] is None and "plugin set" not in r["detail"]


def test_an_override_before_an_earlier_report_does_not_carry_to_a_later_report_without_one():
    log = _override(4100) + _cache("262,400") + _conc(262144, "1.00") + _cache("2,048") + _conc(1024, "2.00")
    r = run(log, max_model_len=1024)["kv_cache"]
    assert r["capacity_tokens"] == 2048 and r["capacity_basis"] == "engine" and r["override_blocks"] is None


def test_each_report_takes_its_own_override_and_the_selected_one_decides():
    log = _override(100) + _cache("1,600") + _override(200) + _cache("3,200")
    r = run(log, max_model_len=1024)["kv_cache"]
    assert r["capacity_tokens"] == 3200 and r["capacity_basis"] == "plugin override" and r["override_blocks"] == 200


def test_an_override_before_the_selected_report_marks_it_as_the_requested_size():
    r = run(_override(4100) + _cache("262,400") + _conc(262144, "1.00"), max_model_len=262144)["kv_cache"]
    assert r["capacity_basis"] == "plugin override" and r["override_blocks"] == 4100 and r["concurrency"] == 1.0


def test_the_concurrency_line_of_an_earlier_report_is_not_read_as_the_selected_reports():
    log = _cache("4,096") + _conc(1024, "4.00") + _cache("2,048")             # the last report states no concurrency of its own
    r = run(log, max_model_len=1024)["kv_cache"]
    assert r["capacity_tokens"] == 2048 and r["concurrency"] is None and r["tokens_per_request"] is None


def test_a_concurrency_line_after_the_selected_report_is_that_reports():
    log = _cache("4,096") + _conc(1024, "4.00") + _cache("2,048") + _conc(1024, "2.00")
    r = run(log, max_model_len=1024)["kv_cache"]
    assert r["capacity_tokens"] == 2048 and r["concurrency"] == 2.0


def test_the_block_count_form_takes_the_override_that_precedes_it_and_not_one_after_it():
    before = run(_override(40) + "# GPU blocks: 40\n", max_model_len=16, block_size=16)["kv_cache"]
    assert before["capacity_basis"] == "plugin override" and before["override_blocks"] == 40
    after = run("# GPU blocks: 40\n" + _override(40), max_model_len=16, block_size=16)["kv_cache"]
    assert after["capacity_basis"] == "engine" and after["override_blocks"] is None


def test_a_concurrency_line_alone_still_reports_its_concurrency():
    r = run(_conc(1024, "1.50"), max_model_len=1024)["kv_cache"]
    assert r["capacity_tokens"] is None and r["concurrency"] == 1.5


# ---- 3. the structured facts reach the boot evidence ---------------------------------------------------------------------------------

def _facts_of(log, **kw):
    evidence = {}
    checks = {}

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
    for result in sv.check_engine_log(log, config=CFG64, max_model_len=kw.get("max_model_len", 1024), max_num_seqs=kw.get("max_num_seqs", 4),
                                      block_size=kw.get("block_size")):
        sv.record(result, check, evidence)
    return evidence, checks


def test_record_keeps_the_structured_facts_of_the_engine_log_results():
    ev, checks = _facts_of(_loading(64) + _bar(64, 64) + _override(4100) + _cache("262,400") + _conc(262144, "1.00"), max_model_len=262144)
    kv, layers = ev["engine_facts"]["kv_cache"], ev["engine_facts"]["layers_loaded"]
    assert kv == {"capacity_tokens": 262400, "capacity_basis": "plugin override", "override_blocks": 4100, "concurrency": 1.0,
                  "tokens_per_request": 262144, "full_length_sequences": 1, "fewer_than_max_num_seqs": True, "blocks_per_sequence": None}
    assert layers == {"layers": 64, "layer_counts_seen": [64], "layers_completed": 64}
    assert set(checks) == {"layers_loaded", "kv_cache"}


def test_record_keeps_the_facts_of_a_failing_result_too():
    ev, checks = _facts_of(_loading(64) + _bar(40, 64) + _cache("64"), max_model_len=1024)
    assert checks["kv_cache"]["ok"] is False and ev["engine_facts"]["kv_cache"]["capacity_tokens"] == 64
    assert checks["layers_loaded"]["ok"] is False and ev["engine_facts"]["layers_loaded"]["layers_completed"] == 40


def test_record_adds_no_engine_facts_for_results_that_are_not_engine_log_results():
    ev = {}
    sv.record({"name": "mesh_devices", "state": "pass", "detail": "ok", "devices": 2}, lambda *a, **k: None, ev)
    assert "engine_facts" not in ev


def test_a_not_checked_result_is_listed_and_adds_no_facts():
    ev, _ = _facts_of("nothing relevant\n")
    assert "engine_facts" not in ev and {n["name"] for n in ev["not_checked"]} == {"layers_loaded", "kv_cache"}


def test_the_recorded_facts_are_plain_json_values_of_bounded_size():
    import json
    log = "".join(f"loaded {n} layers\n" for n in range(1, 40)) + _cache("1,024")
    ev, _ = _facts_of(log, max_model_len=1024)
    text = json.dumps(ev["engine_facts"], allow_nan=False)
    assert len(text) < 1500 and len(ev["engine_facts"]["layers_loaded"]["layer_counts_seen"]) <= sv.MAX_LAYER_COUNTS_LISTED
