# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One test for each finding of the Copilot review of PR 245 (the text route): operation failures, perplexity validation, snapshot links, concurrent weight
conversion, lab mode, the coder-only preflight rows, malformed evidence in the bundle, and the interpreter of a resumed run."""
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from orchard import preflight as pf
from orchard import text_bars as tb, text_reference as tr, text_runtime as rt, text_stages as st
from test_preflight import CLEF_PAYLOAD, cfg, signals
from test_supervisor_text_runtime import CONFIG, MODEL, bench_json, make_environment, trig  # noqa: F401
from test_text_bars import runs
from test_text_stages import dirs, done, edit  # noqa: F401
from text_fixtures import base_arrays, load, perturb, write_run

M = rt.get("minicpm3")


# ---- an operation that leaves fails the stage ---------------------------------------------------------------------------------------

def test_an_operation_that_leaves_fails_the_check_when_every_block_output_is_within_its_tolerance(tmp_path):
    ref_arrays, ref, floors, _ = runs(tmp_path)
    bad = perturb(ref_arrays, 0.03, seed=3)
    bad[("layer", "layer_01.attn")] = bad[("layer", "layer_01.attn")] * 3.0
    chip = load(write_run(tmp_path / "bad", bad, dtype="bfloat16"))
    rep = tb.chip_report(ref, floors, chip)
    assert rep["first_block_leaving"] is None and rep["first_op_leaving"] == "layer_01.attn"
    assert not rep["ok"] and any("layer_01.attn" in r for r in rep["reasons"])


# ---- perplexity is recomputed from the log probabilities --------------------------------------------------------------------------------

def ppl_file(value, n=6):
    return {"tokens": n, "perplexity": value, "mean_nll": float(np.log(value)), "logprobs": [-float(np.log(value))] * n}


def test_a_perplexity_the_log_probabilities_do_not_support_is_refused():
    good = ppl_file(8.0)
    assert tb.perplexity_report(good, ppl_file(8.0))["ok"]
    lying = ppl_file(8.0)
    lying["logprobs"] = [-4.0] * 6                                     # implies a perplexity of about 55
    r = tb.perplexity_report(good, lying)
    assert not r["ok"] and any("log probabilities" in x for x in r["reasons"])


def test_a_log_probability_above_zero_is_refused():
    pos = ppl_file(8.0)
    pos["logprobs"][0] = 0.5
    r = tb.perplexity_report(ppl_file(8.0), pos)
    assert not r["ok"] and any("above zero" in x for x in r["reasons"])


def test_a_log_probability_count_that_is_not_the_token_count_is_refused():
    short = ppl_file(8.0)
    short["logprobs"] = short["logprobs"][:3]
    assert not tb.perplexity_report(ppl_file(8.0), short)["ok"]
    nan = ppl_file(8.0)
    nan["logprobs"][0] = float("nan")
    assert not tb.perplexity_report(ppl_file(8.0), nan)["ok"]


def test_the_reference_gate_applies_the_same_validation(done):
    s1, *_ = dirs(done)
    edit(s1 / st.PPL_REFERENCE_FILE, lambda d: d.update(logprobs=[-9.0] * len(d["logprobs"])))
    assert any("log probabilities" in p for p in st.reference_problems(s1, done.run_dir))


# ---- Hugging Face snapshots hold their files as links into blobs/ ---------------------------------------------------------------------------

def linked_snapshot(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    snap = tmp_path / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    for name, text in (("config.json", json.dumps(CONFIG)), ("README.md", "---\nlicense: apache-2.0\n---\n")):
        (blobs / name.replace(".", "_")).write_text(text)
        (snap / name).symlink_to(blobs / name.replace(".", "_"))
    return snap


def test_a_snapshot_whose_files_are_links_is_read_by_stage_0_helpers(tmp_path):
    snap = linked_snapshot(tmp_path)
    assert M.config_problems(snap) == []
    assert rt.card_license(snap) == "apache-2.0"
    facts = rt.model_facts(snap, M)
    assert facts["n_layers"] == CONFIG["num_hidden_layers"] and len(facts["config_sha256"]) == 64 and facts["card_license"] == "apache-2.0"


def test_finishing_the_benchmark_reads_a_linked_configuration(done, tmp_path):
    s1, _, s6, d = dirs(done)
    edit(s1 / rt.STAGE_CONFIG_FILE, lambda c: c.update(model_dir=str(linked_snapshot(tmp_path))))
    assert st.finish_bench(s6, d, s1, "minicpm3")["variants"]["bf16"]["label"] == "benchmark"


def test_the_benchmark_gate_reads_a_linked_configuration(done, tmp_path):
    s1, _, s6, d = dirs(done)
    snap = linked_snapshot(tmp_path)
    cfg_path = s1 / rt.STAGE_CONFIG_FILE
    edit(cfg_path, lambda c: c.update(model_dir=str(snap)))
    assert st.bench_problems(s6, d, s1) == []


# ---- weights conversion publishes atomically and never overwrites -----------------------------------------------------------------------------------

def test_convert_weights_uses_its_own_temporary_file_and_does_not_replace_a_file_that_appeared(tmp_path, monkeypatch):
    pytest.importorskip("safetensors")
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location("_t_convert", M.scripts_dir / "convert_weights.py")
    convert = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(convert)
    d = tmp_path / "m"
    d.mkdir()
    torch.save({"model.w": torch.ones(2)}, d / "pytorch_model.bin")
    out = tmp_path / "cache" / "w.safetensors"
    seen = []
    import safetensors.torch as stt
    real = stt.save_file

    def racing_save(tensors, filename, *a, **k):
        seen.append(filename)
        real(tensors, filename, *a, **k)
        out.write_bytes(b"another process's finished file")          # the other run publishes while this one writes
    monkeypatch.setattr(stt, "save_file", racing_save)
    monkeypatch.setattr(sys, "argv", ["convert_weights.py", "--model-dir", str(d), "--out", str(out)])
    convert.main()                                                    # the file that is there stays, and this run does not fail
    assert out.read_bytes() == b"another process's finished file"
    assert seen and seen[0] != str(out.with_name(out.name + ".tmp")) and not list(out.parent.glob("*.tmp*"))


# ---- preflight --------------------------------------------------------------------------------------------------------------------------------

MINICPM = dict(CLEF_PAYLOAD, id="openbmb/MiniCPM3-4B", cardData={"license": "apache-2.0"},
               siblings=[{"rfilename": "config.json", "size": 2000}, {"rfilename": "pytorch_model.bin", "size": 8_000_000_000}])


def test_a_lab_run_does_not_take_the_text_route(tmp_path):
    hub = pf.parse_hub(MINICPM)
    local = pf.check_base(model_id="openbmb/MiniCPM3-4B", hub=hub, base_override=None, local_snapshot=lambda m: None, installed=[], search=None)
    lab = pf.check_base(model_id="openbmb/MiniCPM3-4B", hub=hub, base_override=None, local_snapshot=lambda m: None, installed=[], search=None, text_runtime_ok=False)
    assert local.data["text_runtime"] == "minicpm3" and not (lab.data or {}).get("text_runtime")


def test_the_coder_port_and_the_cpu_stand_in_do_not_block_a_text_run(tmp_path):
    cpu_calls = []
    sig = signals(hub_info=lambda m: (pf.parse_hub(MINICPM), None), port_in_use=lambda port: True, installed_bundles=lambda: [],
                  cpu_standin=lambda e, m: cpu_calls.append(e) or "not answering")
    out = pf.run_preflight(cfg(tmp_path), "openbmb/MiniCPM3-4B", accept_credentials=False, signals=sig)
    names = [c.name for c in out]
    assert "port" not in names and "cpu" not in names and not cpu_calls and not pf.blocked(out)
    other = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals(port_in_use=lambda p: True))
    assert "port" in [c.name for c in other]


# ---- the bundle survives malformed text evidence ------------------------------------------------------------------------------------------------

@pytest.mark.real_bundle
def test_the_bundle_builds_when_the_text_result_files_are_malformed(trig):
    pytest.importorskip("numpy")
    make_environment(trig)
    assert trig.run() == 0
    s2, s6 = trig.run_dir / "stages" / "2" / rt.RESULT_FILE, trig.run_dir / "stages" / "6" / rt.BENCH_FILE
    edit(s2, lambda d: d.update(variants={"bf16": None, "mlp8": 5}, limits=7))
    edit(s6, lambda d: d.update(variants={"mlp8": None}, limits="x"))        # no entry at all for bf16
    import subprocess
    r = subprocess.run([sys.executable, str(trig.run_dir / "stages" / "8" / "build_bundle.py")], cwd=trig.run_dir, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    text = (trig.run_dir / "stages" / "8" / "bundle" / "RESULTS.md").read_text()
    assert "Text model on a chip" in text and "not recorded" in text


# ---- a resumed run keeps the interpreter its stage 0 recorded -------------------------------------------------------------------------------------------

def test_the_text_context_uses_the_interpreter_in_the_ledger_not_the_environment(trig, monkeypatch):
    make_environment(trig)
    assert trig.run() == 0
    monkeypatch.setenv("ORCHARD_TEXT_PYTHON", "/some/other/python")
    from orchard import supervisor
    seen = {}

    def spy(self, rtm, stage_dir):
        ctx = real(self, rtm, stage_dir)
        seen["python"] = ctx.python
        return ctx
    real = supervisor.Supervisor._text_context
    monkeypatch.setattr(supervisor.Supervisor, "_text_context", spy)
    recorded = rt.selection(trig.entries())["paths"]["python"]
    # a second start of the finished run builds no stage context; ask for one directly
    sup_holder = {}
    orig_init = supervisor.Supervisor.__init__

    def grab(self, *a, **k):
        orig_init(self, *a, **k)
        sup_holder["s"] = self
    monkeypatch.setattr(supervisor.Supervisor, "__init__", grab)
    trig.run()
    sup_holder["s"]._text_context(M, trig.run_dir / "stages" / "2")
    assert seen["python"] == recorded and recorded != "/some/other/python"


# ---- the benchmark must have run the whole model ------------------------------------------------------------------------------------------------

def test_a_benchmark_that_ran_fewer_layers_than_the_model_has_is_refused(done):
    _, _, s6, d = dirs(done)
    s1 = d / "stages" / "1"
    edit(s6 / "bench_bf16.json", lambda x: x.update(config={"chips": 1, "layers": 3}))
    assert any("layers" in p for p in st.bench_problems(s6, d, s1))


def test_a_lab_configuration_does_not_select_the_text_route_in_run_preflight(tmp_path):
    import dataclasses
    lab_cfg = dataclasses.replace(cfg(tmp_path), mode="lab")
    sig = signals(hub_info=lambda m: (pf.parse_hub(MINICPM), None), installed_bundles=lambda: [])
    out = pf.run_preflight(lab_cfg, "openbmb/MiniCPM3-4B", accept_credentials=False, signals=sig)
    assert not (next(c for c in out if c.name == "base").data or {}).get("text_runtime")


def test_stage_0_records_the_snapshot_directory_name(done):
    ev = json.loads((done.run_dir / "stages" / "0" / "evidence" / "triage.json").read_text())
    assert ev["snapshot"] == "a" * 40


# ---- the second review of PR 245 ------------------------------------------------------------------------------------------------------------------

def test_two_model_directories_with_one_basename_do_not_share_a_converted_weights_file(tmp_path):
    a, b = tmp_path / "rev-a" / "model", tmp_path / "rev-b" / "model"
    for d in (a, b):
        d.mkdir(parents=True)
    cache = tmp_path / "cache"
    assert M.weights_path(a, cache) != M.weights_path(b, cache)
    assert M.weights_path(a, cache) == M.weights_path(a, cache) and M.weights_path(a, cache).parent == cache


def test_a_saved_per_array_report_is_compared_with_the_recomputed_one_not_only_hashed(done):
    s1, s2, _, d = dirs(done)
    rp = s2 / "evidence" / "numeric_report_bf16.json"
    edit(rp, lambda r: r["blocks"][0].update(rel_l2=1e-9))
    import hashlib
    digest = hashlib.sha256(rp.read_bytes()).hexdigest()
    edit(s2 / rt.RESULT_FILE, lambda r: r["variants"]["bf16"].update(report_sha256=digest))
    assert any("per-array report" in p for p in st.chip_problems(s2, d, s1))


def test_a_benchmark_that_reports_two_chips_is_not_accepted_as_one_chip_timings(done):
    s1, _, s6, d = dirs(done)
    edit(s6 / "bench_bf16.json", lambda x: x.update(config={"chips": 2, "layers": 62}))
    assert any("chips" in p for p in st.bench_problems(s6, d, s1))
    with pytest.raises(rt.TextStageError, match="chip"):
        st.finish_bench(s6, d, s1, "minicpm3")


def test_the_chip_cache_is_sized_for_the_prompt_and_the_requested_decode_length():
    assert rt.chip_max_len(16) == 256 and rt.chip_max_len(256) >= 256 + rt.PROMPT_ALLOWANCE
    assert rt.chip_max_len(256) % 128 == 0
    with pytest.raises(rt.TextStageError, match="decode"):
        rt.chip_max_len(0)
    with pytest.raises(rt.TextStageError, match="decode"):
        rt.chip_max_len(10_000)


def test_a_reference_whose_prompt_and_decode_do_not_fit_the_chip_cache_is_refused(done):
    s1, _, _, d = dirs(done)
    edit(s1 / rt.STAGE_CONFIG_FILE, lambda c: c.update(max_len=8))
    assert any("cache" in p for p in st.reference_problems(s1, d))


def test_later_stages_use_the_token_counts_stage_1_saved_not_the_current_inputs(trig, monkeypatch):
    make_environment(trig)
    assert trig.run() == 0
    from orchard import supervisor
    holder = {}
    orig = supervisor.Supervisor.__init__

    def grab(self, *a, **k):
        orig(self, *a, **k)
        holder["s"] = self
    monkeypatch.setattr(supervisor.Supervisor, "__init__", grab)
    trig.run()
    sup = holder["s"]
    sup.inputs["text_ppl_tokens"] = "512"
    sup.inputs["text_decode_tokens"] = "99"
    ctx = sup._text_context(M, trig.run_dir / "stages" / "2")
    saved = json.loads((trig.run_dir / "stages" / "1" / rt.STAGE_CONFIG_FILE).read_text())
    assert (ctx.ppl_tokens, ctx.decode_tokens, ctx.max_len) == (saved["ppl_tokens"], saved["decode_tokens"], saved["max_len"])


def test_a_configuration_with_no_layer_count_fails_stage_0_with_a_reason(trig):
    make_environment(trig)
    bad = {k: v for k, v in CONFIG.items() if k != "num_hidden_layers"}
    (trig.snap / "config.json").write_text(json.dumps(bad))
    from test_supervisor_text_runtime import Stop
    with pytest.raises(Stop):
        trig.run()
    end = trig.ends(0)[0]
    assert end["result"] == "fail" and "num_hidden_layers" in " ".join(end["reasons"])
    facts = rt.model_facts(trig.snap, M)
    assert facts["n_layers"] is None and any("num_hidden_layers" in p for p in facts["config_problems"])
