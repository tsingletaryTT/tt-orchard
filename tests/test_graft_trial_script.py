# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/skills/weights-swap-templates/graft_trial.py: the script the supervisor runs under a lease to time the safe configuration and the
grafted variant. The three shipped scripts it calls (measure_serving.py, prepare_swap.py, serve_and_compare.py) are replaced by stubs in a
fake checkout, so the tests cover the script's own work: the order, the arguments, the files it builds, and what it writes when a step fails.
No server starts and no chip is used."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orchard.graft_trial import TRIAL_RESULT, decide

SCRIPT = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates" / "graft_trial.py"

STUB_MEASURE = '''import json, sys, pathlib
argv = sys.argv[1:]
arg = lambda k: argv[argv.index(k) + 1]
here = pathlib.Path(__file__).resolve().parent
behave = json.loads((here.parent / "behavior.json").read_text())
with open(here.parent / "calls.log", "a") as f:
    f.write(json.dumps({"tool": "measure", "argv": argv}) + "\\n")
b = behave["measure"][arg("--label")]
if b.get("exit"):
    print(b.get("say", "boom")); sys.exit(b["exit"])
def prompt(d, t):
    return {"summary": {"decode_tok_s": {"median": d}, "ttft_s": {"median": t}, "completion_tokens": {"median": 100}}, "runs": []}
rec = {"label": arg("--label"), "server_ready_s": b.get("ready", 100.0), "prompt_set_sha256": "abc", "reps": 3,
       "prompts": {"a": prompt(b["decode"][0], b.get("ttft", [0.3, 0.2, 0.4])[0]), "b": prompt(b["decode"][1], b.get("ttft", [0.3, 0.2, 0.4])[1]),
                   "c": prompt(b["decode"][2], b.get("ttft", [0.3, 0.2, 0.4])[2])}}
pathlib.Path(arg("--out")).write_text(json.dumps(rec))
'''

STUB_PREPARE = '''import json, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
log = pathlib.Path(json.loads((here / "swap_config.json").read_text())["calls_log"])
behave = json.loads(pathlib.Path(json.loads((here / "swap_config.json").read_text())["behavior"]).read_text())
log.open("a").write(json.dumps({"tool": "prepare", "cwd": str(here), "config": json.loads((here / "swap_config.json").read_text())}) + "\\n")
if behave.get("prepare_exit"):
    print("prepare_swap: graft_mtp is set, but the new model already has mtp.* tensors"); sys.exit(behave["prepare_exit"])
(here / "model-dir").mkdir()
(here / "run.sh").write_text("#!/bin/bash\\n")
(here / "model-dir-sources.json").write_text("{}")
'''

STUB_SERVE = '''import json, pathlib, sys
here = pathlib.Path(__file__).resolve().parent
cfg = json.loads((here / "swap_config.json").read_text())
behave = json.loads(pathlib.Path(cfg["behavior"]).read_text())
pathlib.Path(cfg["calls_log"]).open("a").write(json.dumps({"tool": "serve", "cwd": str(here)}) + "\\n")
if behave.get("serve_exit"):
    print("serve_and_compare: the server exited with code 1 before it was healthy"); sys.exit(behave["serve_exit"])
(here / "evidence").mkdir()
draft = {"serves": True, "server_ready_s": 244.1, "coherent": behave.get("coherent", True), "top1_agreement": behave.get("top1", 0.96875),
         "n_tokens": 32}
(here / "evidence" / "swap-check.json").write_text(json.dumps({"result_draft": draft}))
(here / "evidence" / "server.log").write_text("log")
if behave.get("serve_exit_after_report"):
    sys.exit(behave["serve_exit_after_report"])
'''


@pytest.fixture
def rig(tmp_path):
    orchard = tmp_path / "checkout"
    (orchard / "orchard" / "skills" / "weights-swap-templates").mkdir(parents=True)
    (orchard / "orchard" / "measure_serving.py").write_text(STUB_MEASURE)
    t = orchard / "orchard" / "skills" / "weights-swap-templates"
    (t / "prepare_swap.py").write_text(STUB_PREPARE)
    (t / "serve_and_compare.py").write_text(STUB_SERVE)
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "model-dir").mkdir(parents=True)
    (stage / "run.sh").write_text("#!/bin/bash\n")
    log, behavior = orchard / "calls.log", orchard / "behavior.json"
    cfg = {"run_dir": str(run), "new_model_id": "Org/Model@" + "b" * 40, "hf_home": "/hf", "port": 8100,
           "tt_cache": str(tmp_path / "cache" / "tt_cache"), "nearest_model_id": "Qwen/Qwen3.8-27B",
           "calls_log": str(log), "behavior": str(behavior)}
    (stage / "swap_config.json").write_text(json.dumps(cfg))
    trial = stage / "graft-trial"
    trial.mkdir()
    (trial / "graft_trial.py").write_text(SCRIPT.read_text())
    (trial / "trial_config.json").write_text(json.dumps({"stage_dir": str(stage), "orchard_dir": str(orchard)}))
    r = type("Rig", (), {})()
    r.tmp, r.stage, r.trial, r.orchard, r.cfg, r.log, r.behavior = tmp_path, stage, trial, orchard, cfg, log, behavior
    r.set = lambda **b: behavior.write_text(json.dumps(b))
    r.set(measure={"safe": {"decode": [19, 20, 21]}, "variant": {"decode": [60, 62, 64]}})
    r.run = lambda: subprocess.run([sys.executable, str(trial / "graft_trial.py")], capture_output=True, text=True, timeout=60)
    r.calls = lambda: [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    r.result = lambda: json.loads((trial / TRIAL_RESULT).read_text())
    return r


def test_the_trial_times_the_safe_configuration_builds_the_variant_checks_it_and_times_it(rig):
    p = rig.run()
    assert p.returncode == 0, p.stdout + p.stderr
    tools = [c["tool"] for c in rig.calls()]
    assert tools == ["measure", "prepare", "serve", "measure"]
    res = rig.result()
    assert [s["name"] for s in res["steps"]] == ["safe-speed", "prepare", "serve", "variant-speed"]
    assert res["safe"]["decode_tok_s"] == 20 and res["safe"]["ttft_s"] == pytest.approx(0.3)
    assert res["variant"]["served"] is True and res["variant"]["coherent"] is True
    assert res["variant"]["top1_agreement"] == 0.96875 and res["variant"]["n_tokens"] == 32
    assert res["variant"]["decode_tok_s"] == 62 and res["variant"]["server_ready_s"] == 244.1
    assert res["variant"]["failure"] is None


def test_the_decision_reads_what_the_script_wrote(rig):
    """The script and the rule share a schema: a run of the real script feeds decide() with no adapter between them."""
    assert rig.run().returncode == 0
    d = decide(rig.result(), safe_top1=0.96875)
    assert d["choice"] == "graft" and d["numbers"]["speedup"] == pytest.approx(62 / 20)


def test_the_safe_configuration_is_timed_on_the_stage_own_run_script_model_and_cache(rig):
    rig.run()
    first = rig.calls()[0]["argv"]
    arg = lambda k: first[first.index(k) + 1]
    assert arg("--label") == "safe" and arg("--run-sh") == str(rig.stage / "run.sh")
    assert arg("--model-dir") == str(rig.stage / "model-dir") and arg("--tt-cache") == rig.cfg["tt_cache"]
    assert arg("--model-id") == rig.cfg["new_model_id"] and arg("--port") == "8100" and arg("--reps") == "3"
    assert arg("--hf-home") == "/hf" and arg("--out") == str(rig.trial / "speed-safe.json")


def test_the_variant_is_built_with_the_graft_on_and_its_own_tensor_cache(rig):
    rig.run()
    prep = next(c for c in rig.calls() if c["tool"] == "prepare")
    assert prep["cwd"] == str(rig.trial / "variant")
    assert prep["config"]["graft_mtp"] is True
    assert prep["config"]["tt_cache"] == rig.cfg["tt_cache"] + "-graft"
    assert prep["config"]["tt_cache"] != rig.cfg["tt_cache"]
    for key in ("new_model_id", "nearest_model_id", "hf_home", "port", "run_dir"):
        assert prep["config"][key] == rig.cfg[key], key


def test_the_stage_own_swap_config_is_left_alone(rig):
    before = (rig.stage / "swap_config.json").read_text()
    rig.run()
    assert (rig.stage / "swap_config.json").read_text() == before


def test_the_variant_is_timed_on_its_own_run_script_model_and_cache(rig):
    rig.run()
    second = [c for c in rig.calls() if c["tool"] == "measure"][1]["argv"]
    arg = lambda k: second[second.index(k) + 1]
    v = rig.trial / "variant"
    assert arg("--label") == "variant" and arg("--run-sh") == str(v / "run.sh") and arg("--model-dir") == str(v / "model-dir")
    assert arg("--tt-cache") == rig.cfg["tt_cache"] + "-graft" and arg("--out") == str(rig.trial / "speed-variant.json")
    assert arg("--model-id") == rig.cfg["new_model_id"]          # the cache marker serve_and_compare wrote names the same id


def test_the_shipped_scripts_are_used_and_not_a_copy_an_agent_could_have_edited(rig):
    (rig.stage / "prepare_swap.py").write_text("raise SystemExit('an agent edited this')")
    (rig.stage / "serve_and_compare.py").write_text("raise SystemExit('an agent edited this')")
    assert rig.run().returncode == 0
    assert [c["tool"] for c in rig.calls()] == ["measure", "prepare", "serve", "measure"]


def test_a_safe_speed_that_fails_stops_the_trial_and_says_so(rig):
    rig.set(measure={"safe": {"exit": 4, "say": "the server did not answer"}, "variant": {"decode": [60, 60, 60]}})
    assert rig.run().returncode == 0
    assert [c["tool"] for c in rig.calls()] == ["measure"]          # the variant is not tried
    res = rig.result()
    assert "code 4" in res["safe"]["failure"] and "did not answer" in res["safe"]["failure"]
    assert res["variant"]["served"] is False and "not tried" in res["variant"]["failure"]
    assert decide(res, safe_top1=0.9)["choice"] == "safe"


def test_a_graft_that_prepare_swap_refuses_is_a_result_not_a_crash(rig):
    rig.set(measure={"safe": {"decode": [20, 20, 20]}, "variant": {"decode": [60, 60, 60]}}, prepare_exit=2)
    assert rig.run().returncode == 0
    assert [c["tool"] for c in rig.calls()] == ["measure", "prepare"]
    res = rig.result()
    assert res["variant"]["served"] is False
    assert "refused the graft" in res["variant"]["failure"] and "already has mtp" in res["variant"]["failure"]
    assert decide(res, safe_top1=0.9)["choice"] == "safe"


def test_a_variant_that_does_not_serve_is_not_timed(rig):
    rig.set(measure={"safe": {"decode": [20, 20, 20]}, "variant": {"decode": [60, 60, 60]}}, serve_exit=4)
    assert rig.run().returncode == 0
    assert [c["tool"] for c in rig.calls()] == ["measure", "prepare", "serve"]
    res = rig.result()
    assert res["variant"]["served"] is False and "code 4" in res["variant"]["failure"] and "healthy" in res["variant"]["failure"]


def test_a_variant_timing_that_fails_keeps_the_serving_result_and_records_the_failure(rig):
    rig.set(measure={"safe": {"decode": [20, 20, 20]}, "variant": {"exit": 3, "say": "no tokens"}})
    assert rig.run().returncode == 0
    res = rig.result()
    assert res["variant"]["served"] is True and res["variant"]["top1_agreement"] == 0.96875
    assert res["variant"]["decode_tok_s"] is None and "no tokens" in res["variant"]["failure"]
    assert decide(res, safe_top1=0.9)["choice"] == "safe"


def test_an_incoherent_variant_is_reported_as_it_was_measured(rig):
    rig.set(measure={"safe": {"decode": [20, 20, 20]}, "variant": {"decode": [60, 60, 60]}}, coherent=False)
    rig.run()
    assert rig.result()["variant"]["coherent"] is False


def test_the_result_lists_its_evidence_files_run_relative_and_only_those_that_exist(rig):
    rig.run()
    ev = rig.result()["evidence"]
    base = "stages/2/graft-trial/"
    assert ev == sorted([base + "speed-safe.json", base + "speed-variant.json", base + "variant/evidence/swap-check.json",
                         base + "variant/evidence/server.log", base + "variant/model-dir-sources.json"])
    rig.set(measure={"safe": {"exit": 1}, "variant": {"decode": [1, 1, 1]}})
    rig.run()
    assert rig.result()["evidence"] == []


def test_a_rerun_starts_the_variant_again_from_nothing(rig):
    rig.run()
    (rig.trial / "variant" / "stale.txt").write_text("left from a killed run")
    rig.run()
    assert not (rig.trial / "variant" / "stale.txt").exists()


def test_the_result_is_written_whole_and_leaves_no_temporary_file(rig):
    rig.run()
    assert not list(rig.trial.glob("*.tmp"))


def test_a_missing_input_exits_2_and_writes_no_result(rig):
    (rig.trial / "trial_config.json").unlink()
    p = rig.run()
    assert p.returncode == 2 and "trial_config.json" in p.stderr and not (rig.trial / TRIAL_RESULT).exists()


def test_a_missing_shipped_script_exits_2_and_names_it(rig):
    (rig.orchard / "orchard" / "skills" / "weights-swap-templates" / "prepare_swap.py").unlink()
    p = rig.run()
    assert p.returncode == 2 and "prepare_swap.py" in p.stderr and not (rig.trial / TRIAL_RESULT).exists()


def test_a_swap_config_without_a_port_exits_2(rig):
    cfg = dict(rig.cfg)
    del cfg["port"]
    (rig.stage / "swap_config.json").write_text(json.dumps(cfg))
    p = rig.run()
    assert p.returncode == 2 and "port" in p.stderr


def test_the_speeds_are_the_median_of_the_prompts_medians(rig):
    rig.set(measure={"safe": {"decode": [10, 100, 30]}, "variant": {"decode": [60, 62, 64]}})
    rig.run()
    assert rig.result()["safe"]["decode_tok_s"] == 30
    assert rig.result()["safe"]["per_prompt"]["b"]["decode_tok_s"] == 100


def test_a_variant_whose_check_exits_non_zero_did_not_serve_even_if_it_left_a_report(rig):
    rig.set(measure={"safe": {"decode": [20, 20, 20]}, "variant": {"decode": [60, 60, 60]}}, serve_exit_after_report=3)
    rig.run()
    res = rig.result()
    assert res["variant"]["served"] is False and "code 3" in res["variant"]["failure"]
    assert [c["tool"] for c in rig.calls()] == ["measure", "prepare", "serve"]


def test_a_prompt_speed_that_is_a_boolean_is_not_counted(rig):
    rig.set(measure={"safe": {"decode": [True, 100, 50]}, "variant": {"decode": [60, 60, 60]}})
    rig.run()
    assert rig.result()["safe"]["decode_tok_s"] == 75


def test_the_speeds_are_rounded_so_the_card_can_show_exactly_what_the_file_holds(rig):
    rig.set(measure={"safe": {"decode": [19.876543, 19.876543, 19.876543], "ttft": [0.123456, 0.123456, 0.123456]},
                     "variant": {"decode": [61.234567, 61.234567, 61.234567], "ttft": [0.0123456] * 3}})
    rig.run()
    res = rig.result()
    assert res["safe"]["decode_tok_s"] == 19.88 and res["variant"]["decode_tok_s"] == 61.23
    assert res["safe"]["ttft_s"] == 0.123 and res["variant"]["ttft_s"] == 0.012
    assert res["safe"]["per_prompt"]["a"]["decode_tok_s"] == 19.88 and res["safe"]["per_prompt"]["a"]["ttft_s"] == 0.123
