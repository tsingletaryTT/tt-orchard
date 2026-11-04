# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 370: the layers-loaded and KV-cache checks read the lines a real tt-metal model server prints.

The fixture below is modelled on the log of a Qwen3.8-27B-class package served on two Blackhole chips by vLLM 0.26.0 (V1 engine) with the
tt-metal `qwen36` model code, read on 2026-11-03: the layer line, the tqdm bar that follows it, the block-count override and the two KV-cache
lines. Paths, hosts and process ids are replaced. The real log held each of these once (23 of 23 server logs on the box that carry a layer
line show the same shape), so the formats are pinned here by release."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import served_verify as sv

ROOT = Path(__file__).resolve().parent.parent
INDICES = ", ".join(str(i) for i in range(64))

REAL_TT_V1 = (
    "(EngineCore pid=101) INFO 11-03 12:38:38 [core.py:116] Initializing a V1 LLM engine (v0.26.0) with config: model='<MODEL_DIR>', speculative_config=None\n"
    "(EngineCore pid=101) INFO 11-03 12:39:01 [tt/worker.py:629] Getting max_tokens_all_users=262144 for number of blocks in KV cache from generator "
    "'<class 'models.demos.blackhole.qwen36.tt.qwen36_vllm_dflash.Qwen36DFlashForCausalLM'>'.\n"
    "Loading weights: 100%|██████████| 851/851 [00:00<00:00, 2462.41it/s]\n"
    f"(EngineCore pid=101) 2026-11-03 12:39:15.043 | INFO     | models.demos.blackhole.qwen36.tt.model:__init__:174 - Loading 64 transformer layers (indices=[{INDICES}])\n"
    "Loading layers:   0%|          | 0/64 [00:00<?, ?it/s]2026-11-03 12:39:15.047 | DEBUG    | core:from_torch_and_dump:702 - Generating cache for <CACHE>/layers.0.input_layernorm.weight\n"
    "\rLoading layers:  50%|█████     | 32/64 [02:10<02:05,  3.90s/it]\rLoading layers: 100%|██████████| 64/64 [04:23<00:00,  4.12s/it]\n"
    "(EngineCore pid=101) INFO 11-03 12:43:46 [kv_cache_utils.py:2115] Overriding num_gpu_blocks=4100 with num_gpu_blocks_override=4100\n"
    "(EngineCore pid=101) INFO 11-03 12:43:46 [kv_cache_utils.py:2177] GPU KV cache size: 262,400 tokens\n"
    "(EngineCore pid=101) INFO 11-03 12:43:46 [kv_cache_utils.py:2178] Maximum concurrency for 262,144 tokens per request: 1.00x\n"
)
CFG = {"num_hidden_layers": 64}


def by_name(results):
    return {r["name"]: r for r in results}


def run(log, **kw):
    kw.setdefault("config", CFG)
    kw.setdefault("max_model_len", 262144)
    kw.setdefault("max_num_seqs", 1)
    return by_name(sv.check_engine_log(log, **kw))


# ---- the real formats, pinned -----------------------------------------------------------------------------------------------------

def test_the_real_tt_metal_layer_line_and_the_vllm_0_26_cache_lines_are_read():
    r = run(REAL_TT_V1)
    assert r["layers_loaded"]["state"] == "pass" and r["layers_loaded"]["layers"] == 64
    assert r["kv_cache"]["state"] == "pass"
    assert r["kv_cache"]["capacity_tokens"] == 262400 and r["kv_cache"]["concurrency"] == 1.0 and r["kv_cache"]["tokens_per_request"] == 262144


def test_the_raw_capture_with_carriage_returns_reads_the_same_as_the_clean_one():
    assert run(REAL_TT_V1)["layers_loaded"]["state"] == run(REAL_TT_V1.replace("\r", "\n"))["layers_loaded"]["state"] == "pass"


def test_a_model_that_loaded_fewer_layers_than_the_config_has_fails_with_both_numbers():
    log = REAL_TT_V1.replace("Loading 64 transformer", "Loading 4 transformer").replace("64/64", "4/4").replace("0/64", "0/4").replace("32/64", "2/4")
    r = run(log)["layers_loaded"]
    assert r["state"] == "fail" and "4 layers" in r["detail"] and "64" in r["detail"]


# ---- the layer bar: the announcement is not the completion --------------------------------------------------------------------------

def test_a_layer_bar_that_ends_short_of_its_total_fails_even_when_the_announced_count_is_right():
    log = REAL_TT_V1.replace("\rLoading layers: 100%|██████████| 64/64 [04:23<00:00,  4.12s/it]", "")
    r = run(log)["layers_loaded"]
    assert r["state"] == "fail" and r["layers_completed"] == 32 and "32/64" in r["detail"]


def test_a_bar_that_counts_a_different_total_than_the_announced_layers_fails():
    log = REAL_TT_V1.replace("64/64", "63/63")
    r = run(log)["layers_loaded"]
    assert r["state"] == "fail" and "counts 63" in r["detail"]


def test_a_completed_bar_is_named_in_the_result_and_a_missing_bar_is_said_not_assumed():
    done = run(REAL_TT_V1)["layers_loaded"]
    assert done["layers_completed"] == 64 and "64/64" in done["detail"]
    no_bar = run("\n".join(ln for ln in REAL_TT_V1.splitlines() if "Loading layers" not in ln) + "\n")["layers_loaded"]
    assert no_bar["state"] == "pass" and no_bar["layers_completed"] is None and "no progress bar" in no_bar["detail"]


def test_a_bar_that_belongs_to_something_else_is_not_read_as_the_layer_bar():
    log = REAL_TT_V1.replace("Loading layers", "Loading shards")
    assert run(log)["layers_loaded"]["layers_completed"] is None


# ---- the KV capacity: say where the number comes from -------------------------------------------------------------------------------

def test_a_capacity_that_follows_a_block_count_override_is_marked_as_the_requested_size():
    r = run(REAL_TT_V1)["kv_cache"]
    assert r["capacity_basis"] == "plugin override" and r["override_blocks"] == 4100
    assert "not a measured allocation" in r["detail"]


def test_a_capacity_with_no_override_line_is_marked_as_the_engines_own():
    log = "\n".join(ln for ln in REAL_TT_V1.splitlines() if "Overriding" not in ln) + "\n"
    r = run(log)["kv_cache"]
    assert r["capacity_basis"] == "engine" and r["override_blocks"] is None and "not a measured allocation" not in r["detail"]


def test_the_override_does_not_turn_a_too_small_cache_into_a_pass():
    log = REAL_TT_V1.replace("262,400 tokens", "100,000 tokens").replace("1.00x", "0.38x")
    assert run(log)["kv_cache"]["state"] == "fail"


# ---- how many full-length sequences fit against the can's max_num_seqs --------------------------------------------------------------

def test_one_full_length_sequence_against_four_declared_is_a_stated_fact_not_a_buried_phrase():
    r = run(REAL_TT_V1, max_num_seqs=4)["kv_cache"]
    assert r["state"] == "pass" and r["full_length_sequences"] == 1 and r["fewer_than_max_num_seqs"] is True
    assert "only 1 full-length sequence fits" in r["detail"] and "max_num_seqs is 4" in r["detail"]


def test_a_cache_that_holds_every_declared_sequence_makes_no_such_statement():
    log = REAL_TT_V1.replace("262,400 tokens", "1,048,576 tokens").replace("1.00x", "4.00x")
    r = run(log, max_num_seqs=4)["kv_cache"]
    assert r["full_length_sequences"] == 4 and r["fewer_than_max_num_seqs"] is False and "only" not in r["detail"]


def test_without_a_declared_length_or_count_there_is_nothing_to_compare():
    r = run(REAL_TT_V1, max_model_len=None, max_num_seqs=None)["kv_cache"]
    assert r["full_length_sequences"] is None and r["fewer_than_max_num_seqs"] is None


# ---- types and sizes of the declared numbers ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, True, False, "262144", 262144.0, [262144], {}, 0, -1])
def test_a_declared_length_or_count_of_the_wrong_type_or_size_compares_nothing_and_raises_nothing(bad):
    r = run(REAL_TT_V1, max_model_len=bad, max_num_seqs=bad)["kv_cache"]
    assert r["state"] == "pass" and r["full_length_sequences"] is None and r["fewer_than_max_num_seqs"] is None


def test_a_declared_length_far_above_the_cache_is_a_fail_not_a_crash():
    r = run(REAL_TT_V1, max_model_len=10**20, max_num_seqs=4)["kv_cache"]
    assert r["state"] == "fail" and r["full_length_sequences"] == 0


# ---- more than one layer line ---------------------------------------------------------------------------------------------------------

def test_two_different_layer_counts_are_both_reported_and_the_last_still_decides():
    log = REAL_TT_V1 + "2026-11-03 12:44:00 | INFO | draft.model:__init__:10 - Loading 5 transformer layers (indices=[0, 1, 2, 3, 4])\n"
    r = run(log)["layers_loaded"]
    assert r["state"] == "fail" and r["layer_counts_seen"] == [5, 64]
    assert "another line states 64" in r["detail"]


def test_one_repeated_count_is_not_called_two_counts():
    r = run(REAL_TT_V1 + REAL_TT_V1)["layers_loaded"]
    assert r["state"] == "pass" and r["layer_counts_seen"] == [64]


# ---- bounded reading ------------------------------------------------------------------------------------------------------------------

def test_a_line_beyond_the_cap_is_not_read(monkeypatch):
    monkeypatch.setattr(sv, "MAX_ENGINE_LOG_CHARS", 400)
    late = "x" * 500 + "\nLoading 64 transformer layers\n"
    assert run(late)["layers_loaded"]["state"] == "not_checked"
    early = "Loading 64 transformer layers\n" + "x" * 500
    assert run(early)["layers_loaded"]["state"] == "pass"


HOSTILE = textwrap.dedent("""
    import resource, sys, time
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    sys.path.insert(0, sys.argv[1])
    from orchard import served_verify as sv
    n = 50 * 1024 * 1024
    cases = {
        "layers": lambda: "Loading " * (n // 8),
        "layers-number": lambda: "Loading 1" * (n // 9),
        "bar": lambda: "Loading layers: 1%|" * (n // 19),
        "bar-open": lambda: "Loading layers:" + " 1%|" * (n // 4),
        "kv": lambda: "GPU KV cache size:" + " " * n,
        "kv-commas": lambda: "GPU KV cache size: " + "1," * (n // 2),
        "concurrency": lambda: "Maximum concurrency for " + "1," * (n // 2),
        "override": lambda: "Overriding num_gpu_blocks=" + "9" * n,
        "kv-digits": lambda: "GPU KV cache size: " + "1" * n,
        "concurrency-digits": lambda: "Maximum concurrency for " + "1" * n,
        "layers-digits": lambda: "Loading " + "1" * n,
        "bar-digits": lambda: "Loading layers: 1%|" + " " * 8 + "|" + "1" * n,
        "one-line": lambda: "x" * n,
    }
    for name, make in cases.items():
        text = make()
        t = time.time()
        sv.check_engine_log(text, config={"num_hidden_layers": 64}, max_model_len=8192, max_num_seqs=4, block_size=64)
        took = time.time() - t
        print(name, round(took, 2), flush=True)
        del text
""")


def test_a_hostile_fifty_megabyte_line_is_read_in_bounded_memory_and_the_child_ends():
    """The child has a 2 GiB address-space limit, so a copy that grows fails the run. No clock decides a pass: the 150 s limit on the child
    is far above the 15 to 30 s the thirteen cases take, and a backtracking pattern would not end at all."""
    done = subprocess.run([sys.executable, "-I", "-c", HOSTILE, str(ROOT)], capture_output=True, text=True, timeout=150,
                          env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
    assert done.returncode == 0, done.stderr[-800:] + done.stdout[-400:]
    assert len(done.stdout.split()) == 26          # thirteen cases, each printed with its time


# ---- review of PR 443, round 2: a progress bar belongs to the announcement it goes with ------------------------------------------------

def _loading(n):
    return f"(EngineCore pid=101) 2026-11-03 12:39:15.043 | INFO     | models.x.tt.model:__init__:174 - Loading {n} transformer layers (indices=[0])\n"


def _bar(done, total):
    return f"Loading layers:   0%|          | 0/{total} [00:00<?, ?it/s]\rLoading layers: 100%|██████████| {done}/{total} [04:23<00:00,  4.12s/it]\n"


def test_an_earlier_components_completed_bar_is_not_the_bar_of_the_final_announcement():
    # a 32-layer component loads and finishes, then a 64-layer component is announced and its bar has not appeared (or never will)
    log = _loading(32) + _bar(32, 32) + _loading(64)
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers"] == 64 and r["layers_completed"] is None and "no progress bar" in r["detail"]


def test_an_earlier_components_short_bar_does_not_fail_a_final_announcement_whose_own_bar_completed():
    log = _loading(32) + _bar(10, 32) + _loading(64) + _bar(64, 64)
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] == 64 and "64/64" in r["detail"]


def test_the_final_announcements_own_short_bar_still_fails_after_an_earlier_complete_one():
    log = _loading(32) + _bar(32, 32) + _loading(64) + _bar(40, 64)
    r = run(log)["layers_loaded"]
    assert r["state"] == "fail" and r["layers_completed"] == 40 and "40/64" in r["detail"]


def test_a_bar_printed_before_the_announcement_is_not_the_announcements_bar_when_the_line_says_loading():
    log = _bar(64, 64) + _loading(64)
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers_completed"] is None


def test_a_completion_statement_takes_the_bar_that_came_before_it_and_not_an_earlier_components():
    # past-tense statements are printed after the layers finish, so their bar lies between the previous statement and themselves
    log = _bar(32, 32) + "loaded 32 layers\n" + "loaded 64 layers\n"                  # the only bar goes with the 32-layer statement
    r = run(log)["layers_loaded"]
    assert r["state"] == "pass" and r["layers"] == 64 and r["layers_completed"] is None
    log2 = _bar(32, 32) + "loaded 32 layers\n" + _bar(64, 64) + "loaded 64 layers\n"
    r2 = run(log2)["layers_loaded"]
    assert r2["state"] == "pass" and r2["layers_completed"] == 64
    log3 = _bar(32, 32) + "loaded 32 layers\n" + _bar(30, 64) + "loaded 64 layers\n"
    r3 = run(log3)["layers_loaded"]
    assert r3["state"] == "fail" and r3["layers_completed"] == 30


def test_a_bar_that_follows_a_completion_statement_belongs_to_the_next_component():
    log = _bar(64, 64) + "loaded 64 layers\n" + _loading(32) + _bar(5, 32)
    r = run(log)["layers_loaded"]               # the last statement is the 32-layer one: against a 64-layer config it fails on the count alone
    assert r["state"] == "fail" and r["layers"] == 32


# ---- review of PR 443, round 2: sequences are counted in whole blocks ---------------------------------------------------------------

def _cache(tokens):
    return f"(EngineCore pid=101) INFO 11-03 12:43:46 [kv_cache_utils.py:2177] GPU KV cache size: {tokens} tokens\n"


def test_a_length_that_is_not_a_multiple_of_the_block_size_is_counted_in_whole_blocks():
    # 48 tokens of cache, blocks of 16, requests of 17 tokens: each needs 2 blocks, so one fits, not two
    r = run(_cache("48"), max_model_len=17, max_num_seqs=2, block_size=16)["kv_cache"]
    assert r["state"] == "pass" and r["full_length_sequences"] == 1 and r["fewer_than_max_num_seqs"] is True
    assert r["blocks_per_sequence"] == 2 and "only 1 full-length sequence fits" in r["detail"]


def test_a_length_that_is_a_multiple_of_the_block_size_counts_as_before():
    r = run(_cache("64"), max_model_len=16, max_num_seqs=4, block_size=16)["kv_cache"]
    assert r["full_length_sequences"] == 4 and r["fewer_than_max_num_seqs"] is False and r["blocks_per_sequence"] == 1


def test_a_cache_that_cannot_hold_one_sequence_in_whole_blocks_is_a_fail():
    # 20 tokens is 1 whole block of 16; a sequence of 18 tokens needs 2 blocks, although 20 >= 18 tokens
    r = run(_cache("20"), max_model_len=18, max_num_seqs=1, block_size=16)["kv_cache"]
    assert r["state"] == "fail" and r["full_length_sequences"] == 0 and r["blocks_per_sequence"] == 2 and "whole blocks" in r["detail"]


def test_a_cache_that_is_not_a_whole_number_of_blocks_counts_only_its_whole_blocks():
    # 50 tokens is 3 whole blocks of 16; sequences of 16 tokens need 1 block each: 3 fit, not 3.1
    r = run(_cache("50"), max_model_len=16, max_num_seqs=4, block_size=16)["kv_cache"]
    assert r["full_length_sequences"] == 3 and r["fewer_than_max_num_seqs"] is True


@pytest.mark.parametrize("bad", [None, 0, -16, True, 1.5, "16"])
def test_without_a_usable_block_size_the_unrounded_count_is_used_and_no_block_figure_is_stated(bad):
    r = run(_cache("48"), max_model_len=17, max_num_seqs=2, block_size=bad)["kv_cache"]
    assert r["full_length_sequences"] == 2 and r["blocks_per_sequence"] is None


def test_the_block_count_line_gives_the_same_rounding_through_the_declared_block_size():
    log = "INFO # GPU blocks: 3\n"                       # 3 blocks of 16 = 48 tokens
    r = run(log, max_model_len=17, max_num_seqs=2, block_size=16)["kv_cache"]
    assert r["capacity_tokens"] == 48 and r["full_length_sequences"] == 1 and r["blocks_per_sequence"] == 2
