# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A text-runtime run (orchard/text_runtime.py, orchard/text_stages.py, stages.py `TEXT_TABLE`) goes from stage 0 to the bundle without booting the coder.
The CPU commands of stage 1 and the chip command of stages 2 and 6 are replaced by fakes that write recorded runs and result files; everything else is the
supervisor as it runs."""
import json
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from orchard import blocked, supervisor, text_cli, text_remote, text_reference as tr, text_runtime as rt, text_stages as st
from orchard.stages import GateResult, TEXT_TABLE, evidence_record, spec_for
from orchard.supervisor import EXIT_BLOCKED, EXIT_READY
from fakes import Crash
from run_fakes import argv
from test_supervisor import Rig, Stop
from text_fixtures import N_DEC, N_LAYERS, base_arrays, perturb, write_run

pytestmark = pytest.mark.usefixtures("stub_tools")

CONFIG = {"architectures": ["MiniCPM3ForCausalLM"], "model_type": "minicpm3", "hidden_size": 2560, "num_hidden_layers": N_LAYERS, "intermediate_size": 6400,
          "max_position_embeddings": 32768, "num_attention_heads": 40, "num_key_value_heads": 40, "qk_nope_head_dim": 64, "qk_rope_head_dim": 32,
          "q_lora_rank": 768, "kv_lora_rank": 256, "rms_norm_eps": 1e-05, "vocab_size": 73448, "scale_emb": 12, "dim_model_base": 256, "scale_depth": 1.4,
          "rope_scaling": {"type": "longrope", "long_factor": [1.0] * 16, "short_factor": [1.0] * 16, "original_max_position_embeddings": 32768}}
MODEL = "openbmb/MiniCPM3-4B@" + "a" * 40


def bench_json(tok_s):
    runs = [{"repeat": i, "warmup": i == 0, "steps": 64, "decode_s": 64 / tok_s, "decode_tok_s": tok_s, "prefill_tokens": 40, "prefill_s": 0.09,
             "host_loadavg_1m": 1.0} for i in range(4)]
    return {"runs": runs, "config": {"chips": 1}, "text": "a lighthouse"}


@pytest.fixture
def trig(tmp_path, monkeypatch, request):
    r = Rig(tmp_path)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop("the run waited for the operator"))
    snap = tmp_path / "hub" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(CONFIG))
    (snap / "README.md").write_text("---\nlicense: apache-2.0\n---\n# card\n")
    (snap / "pytorch_model.bin").write_bytes(b"x")
    parquet = tmp_path / "wikitext.parquet"
    parquet.write_bytes(b"x")
    a = argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=2)
    a[a.index("--model") + 1] = "openbmb/MiniCPM3-4B"
    from orchard.supervisor import parse
    r.args = parse(a + ["--input", f"model={snap}", "--input", "text_runtime=minicpm3", "--input", f"text_ppl_parquet={parquet}",
                          "--input", f"text_decode_tokens={N_DEC}"])
    r.calls = {"cpu": [], "chip": []}
    r.chip_noise = {}
    r.snap = snap
    monkeypatch.setattr(supervisor.Supervisor, "model_id", MODEL, raising=False)
    ref_arrays = base_arrays()

    def cpu(argv_, env, timeout_s, log):
        r.calls["cpu"].append(argv_)
        script = Path(argv_[1]).name
        arg = lambda flag: argv_[argv_.index(flag) + 1]          # noqa: E731
        if script == "convert_weights.py":
            Path(arg("--out")).parent.mkdir(parents=True, exist_ok=True)
            Path(arg("--out")).write_bytes(b"w")
        elif script == "make_reference.py":
            out = Path(arg("--out-dir"))
            write_run(out / "float32", ref_arrays, dtype="float32", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=snap)
            write_run(out / "bfloat16", perturb(ref_arrays, 0.01, seed=2), dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=snap)
            write_run(out / "native", perturb(ref_arrays, 0.012, seed=4), dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=snap)
        elif script == "ppl_check.py":
            Path(arg("--out")).write_text(json.dumps({"mode": "ref", "tokens": rt.PPL_TOKENS - 1, "perplexity": 8.0, "mean_nll": float(np.log(8.0)),
                                                      "logprobs": [-float(np.log(8.0))] * (rt.PPL_TOKENS - 1)}))
        else:
            raise AssertionError(f"unexpected CPU command {argv_}")
        return 0

    real_spawn = supervisor.Supervisor._spawn

    def spawn(self, n, command, chips, ids, deadline, out_path, env_extra=None, **record):
        if "text_chip_test.py" not in command:
            return real_spawn(self, n, command, chips, ids, deadline, out_path, env_extra, **record)
        r.calls["chip"].append((command, list(chips)))
        cfg_path = Path(command.split("--config ")[1])
        stage = cfg_path.parent
        for cmd in json.loads(cfg_path.read_text())["commands"]:
            av = cmd["argv"]
            arg = lambda flag: av[av.index(flag) + 1]          # noqa: E731
            script, mode = Path(av[1]).name, av[2]
            if mode == "agree":
                write_run(Path(arg("--out")), perturb(ref_arrays, r.chip_noise.get(Path(arg("--out")).name, 0.03), seed=3), dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=snap)
            elif script == "ppl_check.py":
                Path(arg("--out")).write_text(json.dumps({"mode": "chip", "tokens": rt.PPL_TOKENS - 1, "perplexity": 7.95, "mean_nll": float(np.log(7.95)),
                                                          "logprobs": [-float(np.log(7.95))] * (rt.PPL_TOKENS - 1)}))
            elif mode == "bench":
                Path(arg("--out")).write_text(json.dumps(bench_json(40.0 if "bfp8" in av else 33.0)))
            else:
                raise AssertionError(f"unexpected chip command {av}")
        out_path.write_text("fake chip output\n")
        return {"command": command, "returncode": 0, "timed_out": False, "seconds": 1.0, "chips": list(chips), "output": evidence_record(self.run_dir, out_path)}

    monkeypatch.setattr(st, "subprocess_runner", lambda env: cpu)
    monkeypatch.setattr(supervisor.Supervisor, "_spawn", spawn)

    def in_process(python, verb, args, *, timeout_s, env=None):
        out = text_cli.dispatch(verb, args)
        if "error" in out:
            raise rt.TextStageError(out["error"])
        return out["result"]
    monkeypatch.setattr(text_remote, "call", in_process)

    def bundle(self, spec, stage_dir):
        (stage_dir / "bundle").mkdir(parents=True, exist_ok=True)
        (stage_dir / "bundle" / "RESULTS.md").write_text("fake bundle\n")
        return "pass", [], GateResult(True, (), ())
    if not request.node.get_closest_marker("real_bundle"):
        monkeypatch.setattr(supervisor.Supervisor, "_bundle_body", bundle)
    # the stage commands name the weights file under the run's cache root and the chip scripts under this checkout
    yield r
    r.close()


def make_environment(rig):
    paths = rt.text_paths(rig.home)
    Path(paths.python).parent.mkdir(parents=True, exist_ok=True)
    Path(paths.python).write_text("#!/bin/sh\n")
    os.chmod(paths.python, os.stat(paths.python).st_mode | stat.S_IXUSR)
    return paths


def started(rig, n):
    return [e for e in rig.entries() if e["event"] == "stage_start" and e["stage"] == n]


def decisions(rig, name):
    return [d for d in rig.decisions() if d.get("decision") == name]


def test_a_text_run_reaches_the_bundle_and_never_boots_the_coder(trig):
    make_environment(trig)
    try:
        code = trig.run()
    except Stop:
        print([ (e["event"], e["stage"], {k: v for k, v in e["data"].items() if k in ("decision","reason","reasons","result")}) for e in trig.entries()][-12:])
        raise
    assert code == EXIT_READY
    assert trig.m.coder_starts == 0, "no text stage asks the coder anything"
    assert not [d for d in trig.decisions() if d.get("decision") in ("coder lease taken", "coder started")]
    ends = {n: trig.ends(n)[0] for n in range(9)}
    assert [ends[n]["result"] for n in (0, 1, 2, 6, 8)] == ["pass"] * 5
    assert [ends[n]["result"] for n in (3, 4, 5, 7)] == ["skipped"] * 4
    assert "text runtime path" in ends[3]["reason"]
    assert ends[0]["path"] == "full-port" and not decisions(trig, "pause")
    sel = decisions(trig, rt.TEXT_RUNTIME_SELECTED)
    assert len(sel) == 1 and sel[0]["runtime"] == "minicpm3" and "run_chip.py" in sel[0]["scripts"]


def test_the_result_files_hold_what_the_stages_measured(trig):
    make_environment(trig)
    trig.run()
    res = json.loads((trig.run_dir / "stages" / "2" / rt.RESULT_FILE).read_text())
    default = res["variants"]["bf16"]
    assert default["ok"] and default["numeric"]["agreement"]["matches"] == N_DEC and default["perplexity"]["rel_diff"] == pytest.approx((7.95 - 8) / 8)
    assert set(res["variants"]) == {"bf16", "mlp8"}
    bench = json.loads((trig.run_dir / "stages" / "6" / rt.BENCH_FILE).read_text())
    assert bench["variants"]["bf16"]["label"] == "benchmark" and bench["variants"]["bf16"]["decode_tok_s"]["median"] == pytest.approx(33.0)
    assert 0 < bench["variants"]["bf16"]["share_of_floor_median"] < 1
    ref = json.loads((trig.run_dir / "stages" / "1" / rt.REFERENCE_FILE).read_text())
    assert len(ref["tokens"]) == N_DEC and set(ref["runs"]) == {"reference", "floor_bfloat16", "floor_native"}


def test_the_stage_commands_are_the_adapters(trig):
    paths = make_environment(trig)
    trig.run()
    names = [Path(a[1]).name for a in trig.calls["cpu"]]
    assert names == ["convert_weights.py", "make_reference.py", "ppl_check.py"]
    assert all(a[0] == paths.python for a in trig.calls["cpu"])
    assert len(trig.calls["chip"]) == 2                           # stage 2 and stage 6
    for command, chips in trig.calls["chip"]:
        assert "text_chip_test.py" in command and len(chips) == 2


def test_a_resume_after_stage_1_keeps_the_selection_and_boots_no_coder(trig):
    make_environment(trig)
    with pytest.raises(Crash):
        trig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    assert not started(trig, 2)
    assert trig.run() == EXIT_READY
    assert trig.m.coder_starts == 0 and len(decisions(trig, rt.TEXT_RUNTIME_SELECTED)) == 1
    assert len(trig.calls["cpu"]) == 3, "stage 1's commands ran once"


def test_a_missing_interpreter_fails_stage_0_with_the_fix(trig):
    with pytest.raises(Stop):
        trig.run()
    end = trig.ends(0)[0]
    assert end["result"] == "fail" and "ORCHARD_TEXT_PYTHON" in end["reasons"][0]
    assert trig.m.coder_starts == 0


def test_a_configuration_the_runtime_refuses_fails_stage_0(trig):
    make_environment(trig)
    bad = dict(CONFIG, model_type="llama")
    (trig.snap / "config.json").write_text(json.dumps(bad))
    with pytest.raises(Stop):
        trig.run()
    end = trig.ends(0)[0]
    assert end["result"] == "fail" and "minicpm3" in " ".join(end["reasons"])
    assert trig.m.coder_starts == 0


def test_a_chip_run_that_leaves_the_bar_fails_stage_2_and_names_the_block(trig, monkeypatch):
    make_environment(trig)
    real = supervisor.Supervisor._spawn

    def damaged(self, n, command, chips, ids, deadline, out_path, env_extra=None, **record):
        out = real(self, n, command, chips, ids, deadline, out_path, env_extra, **record)
        if n == 2:
            run = self.run_dir / "stages" / "2" / "chip" / "bf16"
            import shutil
            shutil.rmtree(run)
            bad = perturb(base_arrays(), 0.03, seed=3)
            bad[("layer", "layer_02")] = bad[("layer", "layer_02")] * 2.0
            write_run(run, bad, dtype="bfloat16", model="openbmb/MiniCPM3-4B", prompt=rt.DEFAULT_PROMPT, config_dir=trig.snap)
        return out
    monkeypatch.setattr(supervisor.Supervisor, "_spawn", damaged)
    with pytest.raises(Stop):
        trig.run()
    end = trig.ends(2)[-1]
    assert end["result"] == "fail" and any("layer_02" in r for r in end["reasons"])
    assert not started(trig, 6)


def test_the_text_table_is_selected_only_for_a_text_run():
    assert spec_for(2, "full-port", None, "full-port", text_runtime=True) is TEXT_TABLE[2]
    assert spec_for(2, "full-port", None, "full-port") is not TEXT_TABLE[2]
    assert all(not TEXT_TABLE[n].needs_coder for n in TEXT_TABLE)
    assert {n for n, s in TEXT_TABLE.items() if s.skip} == {3, 4, 5, 7}


def test_selected_reads_only_a_registered_name_from_run_start():
    entries = [{"event": "run_start", "data": {"inputs": {"text_runtime": "minicpm3"}}}]
    assert rt.selected(entries) == "minicpm3"
    assert rt.selected([{"event": "run_start", "data": {"inputs": {"text_runtime": "nope"}}}]) is None
    assert rt.selected([{"event": "run_start", "data": {"inputs": {}}}]) is None


@pytest.mark.real_bundle
def test_the_real_bundle_builder_prints_the_text_section_and_the_limits(trig):
    make_environment(trig)
    assert trig.run() == EXIT_READY
    results = (trig.run_dir / "stages" / "8" / "bundle" / "RESULTS.md").read_text()
    assert "## Text model on a chip" in results and "| bf16 |" in results and "| mlp8 |" in results
    assert "33" in results and "benchmark" in results and "What these numbers do not show" in results
    assert "no chip arithmetic" in results.replace("holds no chip arithmetic", "no chip arithmetic")
    risks = (trig.run_dir / "stages" / "8" / "bundle" / "RISKS.md").read_text()
    assert "## Text run" in risks and "provisional" in risks
    assert "stages/0/delta.json" in results or "Stage 0" in results


@pytest.mark.real_bundle
def test_a_variant_outside_the_bar_is_shown_as_not_met_and_listed_in_the_risks(trig):
    make_environment(trig)
    trig.chip_noise = {"mlp8": 0.6}
    assert trig.run() == EXIT_READY                     # the default variant decides the stage
    results = (trig.run_dir / "stages" / "8" / "bundle" / "RESULTS.md").read_text()
    row = next(line for line in results.splitlines() if line.startswith("| mlp8 |"))
    assert row.rstrip().endswith("not met |") and next(line for line in results.splitlines() if line.startswith("| bf16 |")).rstrip().endswith("| met |")
    risks = (trig.run_dir / "stages" / "8" / "bundle" / "RISKS.md").read_text()
    assert "Variant mlp8 did not meet the stage 2 bar" in risks and "Variant bf16" not in risks
    assert "The chip comparison is in the section Text model on a chip" in results
