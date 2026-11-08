# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`layer-diff` (orchard/skills/full-port-templates/stage_tools.py, issue 505): the port's per-layer hidden states against the CPU reference's, on
the same prompt, and the first layer whose relative error leaves a threshold. The port side is the file the hardware test wrote (see
tests/test_port_check_layers.py). The reference side runs in the reference interpreter; these tests give it a cached reference file, so no
torch is needed, and one test runs the real reference code when ORCHARD_REFERENCE_PYTHON names an interpreter that has torch and transformers."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from test_stage_tools_last_check import TOOL, load_tool

PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]
N, SEQ, HID = 6, len(PROMPT), 16


def reference_layers(rng=None):
    rng = rng or np.random.default_rng(0)
    return [rng.standard_normal((SEQ, HID)).astype(np.float32) + 1.0 for _ in range(N)]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(tmp_path, *, port_layers=None, ref_layers=None, schedule=None, seal=True, port_prompt=None, snapshot=True):
    """A stage directory with the port's layer file, the supervisor's sealed output naming it, the stage 1 prompt, a config, and a cached reference."""
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "evidence").mkdir(parents=True)
    ref_dir = run / "stages" / "1" / "evidence" / "reference"
    ref_dir.mkdir(parents=True)
    (ref_dir / "prompt-ids.json").write_text(json.dumps({"prompt_ids": PROMPT}))
    snap = tmp_path / "snap"
    snap.mkdir()
    (stage / "port_check_config.json").write_text(json.dumps({"model_snapshot": str(snap)}))
    ref = ref_layers if ref_layers is not None else reference_layers()
    port = port_layers if port_layers is not None else [a.copy() for a in ref]
    arrays = {f"layer_{k:03d}": a for k, a in enumerate(port)}
    arrays["prompt_ids"] = np.asarray(port_prompt or PROMPT, dtype=np.int64)
    np.savez(stage / "evidence" / "layer-port.npz", **arrays)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text("top-1 agreement 0.5625 over 32 positions\n" + (f"PORT_CHECK_LAYERS {sha(stage / 'evidence' / 'layer-port.npz')} {len(port)}\n" if seal else "")
                   + "PORT_CHECK_SHA256 " + "a" * 64 + "\n")
    (stage / "test-result.json").write_text(json.dumps({"command": "python3 -P -s stages/2/port_check.py", "returncode": 0, "timed_out": False,
                                                        "output": {"path": "stages/2/evidence/hw-test-output.txt", "sha256": sha(out)}}))
    write_reference_cache(stage, ref, snap)
    if schedule is not None:
        (stage / "port_facts.json").write_text(json.dumps({"layer_schedule": schedule}))
    return stage


def write_reference_cache(stage, ref, snap, dtype="float32"):
    """The reference the tool would compute, as it would cache it: an npz and a sidecar that binds it to the prompt, the snapshot and the dtype."""
    npz = stage / "evidence" / "layer-reference.npz"
    np.savez(npz, **{f"layer_{k:03d}": a for k, a in enumerate(ref)})
    (stage / "evidence" / "layer-reference.json").write_text(json.dumps(
        {"prompt_sha256": hashlib.sha256(json.dumps(PROMPT).encode()).hexdigest(), "snapshot": str(snap), "dtype": dtype, "npz_sha256": sha(npz), "layers": len(ref)}))


def run_tool(stage, *args, python=sys.executable, cwd=None):
    """The command line as the coder runs it: the tool file copied into the stage directory, a reference interpreter named."""
    import shutil
    shutil.copy2(TOOL, stage / "stage_tools.py")
    env = {"PATH": os.environ.get("PATH", "")}
    return subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), "layer-diff", "--python", python, *args], cwd=cwd or stage.parent.parent,
                          env=env, capture_output=True, text=True, timeout=120)


def test_identical_layers_all_pass_and_the_report_says_the_error_is_outside_the_layers(tmp_path):
    done = run_tool(build(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "all 6 compared layers are within" in done.stdout and "outside the layers" in done.stdout


def test_the_first_layer_over_the_threshold_is_named_with_its_numbers_and_the_one_before(tmp_path):
    ref = reference_layers()
    port = [a.copy() for a in ref]
    for k in (3, 4, 5):                                 # the error enters at layer 3 and stays
        port[k] = port[k] + 0.9
    done = run_tool(build(tmp_path, port_layers=port, ref_layers=ref))
    assert done.returncode == 1, done.stdout + done.stderr
    assert "first layer over the threshold: layer 3" in done.stdout
    assert "layer 2 was" in done.stdout
    assert len(done.stdout.splitlines()) < 40


def test_a_layer_that_is_over_and_comes_back_is_still_reported_as_the_first(tmp_path):
    ref = reference_layers()
    port = [a.copy() for a in ref]
    port[1] = port[1] + 5.0
    done = run_tool(build(tmp_path, port_layers=port, ref_layers=ref))
    assert "first layer over the threshold: layer 1" in done.stdout


def test_the_layer_is_labelled_with_its_kind_from_the_facts_schedule(tmp_path):
    ref = reference_layers()
    port = [a.copy() for a in ref]
    port[3] = port[3] * 2
    done = run_tool(build(tmp_path, port_layers=port, ref_layers=ref, schedule={"linear_attention": "0-2", "full_attention": "3,5"}))
    assert "layer 3 is full_attention" in done.stdout


def test_a_nan_in_the_port_is_a_divergence_even_though_a_nan_compares_false(tmp_path):
    ref = reference_layers()
    port = [a.copy() for a in ref]
    port[2][0, 0] = np.nan
    done = run_tool(build(tmp_path, port_layers=port, ref_layers=ref))
    assert done.returncode == 1 and "first layer over the threshold: layer 2" in done.stdout and "non-finite" in done.stdout


def test_a_layer_with_another_size_is_reported_not_compared(tmp_path):
    ref = reference_layers()
    port = [a.copy() for a in ref]
    port[1] = np.zeros((SEQ, HID * 4), dtype=np.float32)       # for example a hyper-connection model that keeps four streams
    done = run_tool(build(tmp_path, port_layers=port, ref_layers=ref))
    assert done.returncode == 1 and "layer 1" in done.stdout and "size" in done.stdout


def test_the_same_numbers_in_another_shape_are_compared_and_the_shapes_are_shown(tmp_path):
    ref = reference_layers()
    port = [a.reshape(1, SEQ, HID).copy() for a in ref]
    done = run_tool(build(tmp_path, port_layers=port, ref_layers=ref))
    assert done.returncode == 0 and "(1, 8, 16)" in done.stdout


def test_a_port_with_fewer_layers_is_compared_as_far_as_it_goes_and_says_so(tmp_path):
    ref = reference_layers()
    done = run_tool(build(tmp_path, port_layers=ref[:4], ref_layers=ref))
    assert "the port returned 4 layers and the reference has 6" in done.stdout


def test_a_layer_file_with_a_gap_in_its_layer_names_is_refused(tmp_path):
    stage = build(tmp_path)
    ref = reference_layers()
    np.savez(stage / "evidence" / "layer-port.npz", layer_000=ref[0], layer_002=ref[2], prompt_ids=np.asarray(PROMPT))
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(f"PORT_CHECK_LAYERS {sha(stage / 'evidence' / 'layer-port.npz')} 2\n")
    marker = json.loads((stage / "test-result.json").read_text())
    marker["output"]["sha256"] = sha(out)
    (stage / "test-result.json").write_text(json.dumps(marker))
    done = run_tool(stage)
    assert done.returncode == 2 and "in order" in done.stdout


def test_a_zero_threshold_and_a_threshold_that_is_not_a_number_are_refused(tmp_path):
    stage = build(tmp_path)
    for bad in ("nan", "-1", "x", "inf", "0"):
        done = run_tool(stage, "--threshold", bad)
        assert done.returncode == 2 and "threshold" in done.stdout, bad


def test_a_layer_file_that_is_not_the_one_the_test_wrote_is_refused(tmp_path):
    stage = build(tmp_path)
    port = [a + 0.5 for a in reference_layers()]
    np.savez(stage / "evidence" / "layer-port.npz", **{f"layer_{k:03d}": a for k, a in enumerate(port)}, prompt_ids=np.asarray(PROMPT))
    done = run_tool(stage)
    assert done.returncode == 2 and "not the file the test wrote" in done.stdout and "first layer" not in done.stdout


def test_a_layer_file_with_no_line_in_the_sealed_output_is_refused(tmp_path):
    done = run_tool(build(tmp_path, seal=False))
    assert done.returncode == 2 and "no PORT_CHECK_LAYERS line" in done.stdout


def test_no_layer_file_says_how_to_get_one(tmp_path):
    stage = build(tmp_path, seal=False)
    (stage / "evidence" / "layer-port.npz").unlink()
    done = run_tool(stage)
    assert done.returncode == 2 and "layer_outputs(model, ids)" in done.stdout and "0.85" in done.stdout


def test_a_line_that_names_a_file_that_is_gone_is_refused_with_the_file_name(tmp_path):
    stage = build(tmp_path)
    (stage / "evidence" / "layer-port.npz").unlink()
    done = run_tool(stage)
    assert done.returncode == 2 and "layer-port.npz missing" in done.stdout


def test_two_layer_lines_in_the_output_are_refused(tmp_path):
    stage = build(tmp_path)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(out.read_text() + "PORT_CHECK_LAYERS " + "0" * 64 + " 6\n")
    marker = json.loads((stage / "test-result.json").read_text())
    marker["output"]["sha256"] = sha(out)
    (stage / "test-result.json").write_text(json.dumps(marker))
    done = run_tool(stage)
    assert done.returncode == 2 and "2 PORT_CHECK_LAYERS lines" in done.stdout


def test_an_output_file_that_no_longer_matches_the_supervisors_record_is_refused(tmp_path):
    stage = build(tmp_path)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(out.read_text() + "edited\n")
    done = run_tool(stage)
    assert done.returncode == 2 and "does not match" in done.stdout


def test_a_port_prompt_that_is_not_the_stage_1_prompt_is_refused(tmp_path):
    done = run_tool(build(tmp_path, port_prompt=[9, 9, 9]))
    assert done.returncode == 2 and "prompt" in done.stdout


def test_a_cached_reference_for_another_prompt_or_snapshot_is_not_used(tmp_path):
    stage = build(tmp_path)
    side = stage / "evidence" / "layer-reference.json"
    meta = json.loads(side.read_text())
    meta["snapshot"] = "/somewhere/else"
    side.write_text(json.dumps(meta))
    done = run_tool(stage)                      # this interpreter has numpy and no torch: a cache that is used needs neither, one that is not needs torch
    assert done.returncode == 2 and "cannot compute the reference" in done.stdout


def test_an_edited_cached_reference_is_not_used(tmp_path):
    stage = build(tmp_path)
    np.savez(stage / "evidence" / "layer-reference.npz", **{f"layer_{k:03d}": a + 1 for k, a in enumerate(reference_layers())})
    done = run_tool(stage)
    assert done.returncode == 2 and "cannot compute the reference" in done.stdout


def test_a_missing_interpreter_is_a_reason_that_says_how_to_name_one(tmp_path):
    stage = build(tmp_path)
    done = run_tool(stage, python=str(tmp_path / "no-such-python"))
    assert done.returncode == 2 and "--python" in done.stdout and "ORCHARD_REFERENCE_PYTHON" in done.stdout


def test_the_tool_reads_stage_tools_json_for_the_interpreter(tmp_path):
    stage = build(tmp_path)
    (stage / "stage_tools.json").write_text(json.dumps({"reference_python": sys.executable}))
    import shutil
    shutil.copy2(TOOL, stage / "stage_tools.py")
    done = subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), "layer-diff"], cwd=tmp_path, env={"PATH": os.environ.get("PATH", "")},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "all 6 compared layers" in done.stdout


def pickled_npz() -> bytes:
    import io
    buf = io.BytesIO()
    np.savez(buf, layer_000=np.array([{"x": 1}], dtype=object))      # numpy pickles an object array on save; loading it unpickles unless refused
    return buf.getvalue()


@pytest.mark.parametrize("bad", [b"", b"not an npz", b"PK\x03\x04" + b"\0" * 50, pickled_npz()])
def test_a_layer_file_that_cannot_be_read_gives_a_reason_and_no_traceback(tmp_path, bad):
    stage = build(tmp_path)
    (stage / "evidence" / "layer-port.npz").write_bytes(bad)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(f"PORT_CHECK_LAYERS {sha(stage / 'evidence' / 'layer-port.npz')} 6\n")
    marker = json.loads((stage / "test-result.json").read_text())
    marker["output"]["sha256"] = sha(out)
    (stage / "test-result.json").write_text(json.dumps(marker))
    done = run_tool(stage)
    assert done.returncode == 2 and "Traceback" not in done.stderr and "cannot be read" in done.stdout


def test_a_link_in_place_of_the_layer_file_is_refused(tmp_path):
    stage = build(tmp_path)
    real = tmp_path / "real.npz"
    (stage / "evidence" / "layer-port.npz").rename(real)
    (stage / "evidence" / "layer-port.npz").symlink_to(real)
    done = run_tool(stage)
    assert done.returncode == 2 and "layer-port.npz" in done.stdout


def test_the_compare_function_is_a_pure_function_of_two_lists(tmp_path):
    tool = load_tool()
    ref = reference_layers()
    out = tool.compare_layers(ref, [a * 1.0 for a in ref], 0.05)
    assert out["first"] is None and len(out["rows"]) == N
    out = tool.compare_layers(ref, [a * (1.0 if k < 2 else 1.5) for k, a in enumerate(ref)], 0.05)
    assert out["first"] == 2 and out["rows"][2]["rel"] == pytest.approx(0.5)


@pytest.mark.skipif(not os.environ.get("ORCHARD_REFERENCE_PYTHON"), reason="needs ORCHARD_REFERENCE_PYTHON: an interpreter with torch and transformers")
def test_the_reference_code_on_a_real_tiny_model_matches_the_decoder_layer_outputs(tmp_path):
    """The real reference side, run under the reference interpreter on a tiny random Llama: its per-layer states equal the model's own hidden
    states for every layer but the last (which transformers returns after the final norm), and the last layer's differs from that, so the tool
    holds the raw layer output and not the normalised one."""
    py = os.environ["ORCHARD_REFERENCE_PYTHON"]
    probe = tmp_path / "probe.py"
    probe.write_text(f'''
import importlib.util, json, sys, torch
spec = importlib.util.spec_from_file_location("st", {str(TOOL)!r}); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from transformers import LlamaConfig, LlamaForCausalLM
torch.manual_seed(0)
cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=4, vocab_size=50)
LlamaForCausalLM(cfg).save_pretrained({str(tmp_path / "tiny")!r})
ids = [1, 2, 3, 4, 5]
layers = m.reference_layers({str(tmp_path / "tiny")!r}, ids, "float32")
model = LlamaForCausalLM.from_pretrained({str(tmp_path / "tiny")!r}); model.eval()
with torch.no_grad():
    hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
ok = [bool(torch.allclose(torch.tensor(layers[k]).reshape(hs[k + 1].shape), hs[k + 1], atol=1e-5)) for k in range(3)]
print(json.dumps({{"n": len(layers), "equal": ok}}))
''')
    done = subprocess.run([py, "-I", str(probe)], capture_output=True, text=True, timeout=300, env={"PATH": os.environ.get("PATH", ""), "HF_HUB_OFFLINE": "1"})
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["n"] == 3 and got["equal"] == [True, True, False]


def test_what_the_check_writes_is_what_layer_diff_reads(tmp_path):
    """The two halves joined: the real check template runs a failing fake port with the hook (CPU, fake ttnn), then `layer-diff` compares the file it
    left, through the line the check printed, with a cached reference that differs from the port in layer 2 only."""
    from port_check_fakes import PROMPT_IDS, WRONG, make_run, run_template
    hook = ("\nimport numpy as np\ndef layer_outputs(model, ids):\n"
            "    return [np.full((len(ids), 4), float(k + 1), dtype=np.float32) for k in range(3)]\n")
    run, stage, snap = make_run(tmp_path, WRONG + hook)
    assert run_template(tmp_path, run, stage).returncode == 0
    ref = [np.full((len(PROMPT_IDS), 4), float(k + 1), dtype=np.float32) for k in range(3)]
    ref[2] = ref[2] * 2
    key = {"prompt_sha256": hashlib.sha256(json.dumps(PROMPT_IDS).encode()).hexdigest(), "snapshot": str(snap), "dtype": "float32"}
    npz = stage / "evidence" / "layer-reference.npz"
    np.savez(npz, **{f"layer_{k:03d}": a for k, a in enumerate(ref)})
    (stage / "evidence" / "layer-reference.json").write_text(json.dumps({**key, "npz_sha256": sha(npz), "layers": 3}))
    done = run_tool(stage)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "first layer over the threshold: layer 2" in done.stdout
