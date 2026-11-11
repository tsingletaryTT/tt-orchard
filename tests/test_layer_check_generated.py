"""scripts/layer_check_generated.py: the layer-by-layer check of a generated port against the transformers reference (issue 696).

The comparison runs on the CPU with the torch-backed fake ttnn (tests/fake_ttnn_compute.py); the board run is the parent's, under a gozer lease.
The reference phase needs an interpreter whose torch and transformers can run Qwen4Exp end to end (the device interpreter's torch 2.7 cannot:
the model's indexer scatters with an index dtype it refuses), so these tests skip without ORCHARD_REFERENCE_PYTHON.

Mutation checks: each guard of the script is removed once in a copy and the copy's own selftest (or a unit check of `compare`) must turn red.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "layer_check_generated.py"
sys.path.insert(0, str(REPO / "scripts"))

import layer_check_generated as lc  # noqa: E402


def qwen4_reference_python() -> str | None:
    cand = os.environ.get("ORCHARD_REFERENCE_PYTHON")
    if not cand or not os.path.isfile(cand):
        return None
    probe = subprocess.run([cand, "-I", "-c", "import torch; from transformers import Qwen4ExpForCausalLM"], capture_output=True, timeout=180)
    return cand if probe.returncode == 0 else None


needs_reference = pytest.mark.skipif(qwen4_reference_python() is None,
                                     reason="needs ORCHARD_REFERENCE_PYTHON: an interpreter with torch and a transformers that has Qwen4ExpForCausalLM")


# ---- compare: no reference interpreter needed ----------------------------------------------------------------------------------------------

def test_compare_identical_tensors_is_ok():
    t = torch.randn(24, 8, generator=torch.Generator().manual_seed(4))
    r = lc.compare(t, t.clone())
    assert r["ok"] and r["pcc"] == pytest.approx(1.0) and r["max_abs"] == 0 and r["nonfinite"] == 0


def test_compare_uncorrelated_tensors_is_not_ok():
    g = torch.Generator().manual_seed(1)
    r = lc.compare(torch.randn(24, 8, generator=g), torch.randn(24, 8, generator=g))
    assert not r["ok"] and abs(r["pcc"]) < 0.5


def test_compare_counts_non_finite_values_and_is_not_ok_for_them():
    t = torch.randn(24, 8, generator=torch.Generator().manual_seed(3))
    got = t.clone()
    got[3, 3] = float("nan")
    got[4, 4] = float("inf")
    r = lc.compare(got, t)
    assert r["nonfinite"] == 2 and not r["ok"]
    assert r["pcc"] > 0.9                                           # the rest still correlates, so only the count can make this fail


def test_compare_a_constant_result_has_no_correlation_and_is_not_ok():
    r = lc.compare(torch.zeros(24, 8), torch.randn(24, 8))
    assert not r["ok"]


def test_compare_names_a_shape_mismatch_and_is_not_ok():
    r = lc.compare(torch.zeros(32, 8), torch.zeros(24, 8))
    assert not r["ok"] and "shape mismatch" in r["error"]


def test_compare_relative_error_catches_a_scaled_result_that_pcc_cannot():
    t = torch.randn(64, 8, generator=torch.Generator().manual_seed(5))
    r = lc.compare(1.5 * t, t, 0.95, 0.25)
    assert r["pcc"] == pytest.approx(1.0) and r["rel_l2"] == pytest.approx(0.5, abs=1e-6) and not r["ok"]
    assert lc.compare(1.5 * t, t, 0.95)["ok"]                          # without a relative bound the scaled result passes
    assert lc.compare(1.05 * t, t, 0.95, 0.25)["ok"]


def test_compare_threshold_is_a_parameter():
    t = torch.randn(64, 8, generator=torch.Generator().manual_seed(6))
    got = t + 0.5 * torch.randn(64, 8, generator=torch.Generator().manual_seed(2))
    loose, strict = lc.compare(got, t, 0.5), lc.compare(got, t, 0.999)
    assert loose["ok"] and not strict["ok"] and loose["pcc"] == strict["pcc"]


def test_the_script_does_not_import_ttnn_when_it_is_imported():
    code = "import sys; sys.path.insert(0, %r); import layer_check_generated; assert 'ttnn' not in sys.modules, 'ttnn was imported'" % str(SCRIPT.parent)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-400:]


def test_the_device_phase_exits_2_with_a_message_when_ttnn_cannot_be_imported(capsys):
    def no_ttnn():
        raise ImportError("no ttnn here")
    args = lc.argparse.Namespace(snapshot="s", construction="c", inventory=None, excerpts=None, reference="r", reference_out=None, out=None)
    assert lc.device_main(args, importer=no_ttnn) == lc.EXIT_NO_TTNN == 2
    err = capsys.readouterr().err
    assert "ttnn cannot be imported" in err and "gozer run" in err


# ---- the port's structure --------------------------------------------------------------------------------------------------------------------

def _tiny_port_text() -> str:
    sys.path.insert(0, str(REPO / "tests"))
    from test_port_template import tiny_qwen4
    return tiny_qwen4()[0]


def test_a_plan_with_every_block_family_has_no_stub():
    text = _tiny_port_text()
    assert lc.stub_functions(text) == set()


def test_the_stub_and_the_layers_that_reach_it_are_found_when_the_attention_family_is_missing(monkeypatch):
    sys.path.insert(0, str(REPO / "orchard"))
    from orchard import port_template as pt
    monkeypatch.setattr(pt, "BLOCK_FAMILIES", {k: v for k, v in pt.BLOCK_FAMILIES.items() if k != "qsa_attention"})
    text = _tiny_port_text()
    stubs = lc.stub_functions(text)
    graph = lc.call_graph(text)
    assert stubs == {"_C_self_attn"}
    assert lc.reaches(graph, "_layer_C", stubs) == ["_C_self_attn"]
    assert lc.reaches(graph, "_layer_A", stubs) == [] and lc.reaches(graph, "_layer_B", stubs) == []


def test_reaches_follows_calls_through_other_port_functions():
    graph = {"a": {"b"}, "b": {"c"}, "c": set(), "d": set()}
    assert lc.reaches(graph, "a", {"c"}) == ["c"] and lc.reaches(graph, "d", {"c"}) == [] and lc.reaches(graph, "c", {"c"}) == ["c"]


# ---- the whole run: tiny checkpoint, reference interpreter, fake ttnn -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    py = qwen4_reference_python()
    if py is None:
        pytest.skip("needs ORCHARD_REFERENCE_PYTHON: an interpreter with torch and a transformers that has Qwen4ExpForCausalLM")
    sys.path.insert(0, str(REPO / "scripts" / "tiny_qwen4"))
    import make_checkpoint
    tmp = tmp_path_factory.mktemp("layercheck")
    snap = tmp / "snapshot"
    assert make_checkpoint.main(["--out", str(snap), "--variant", "dense", "--no-tokenizer", "--no-reference"]) == 0
    ref = tmp / "reference.pt"
    lc.run_reference(py, snap, 24, 0, ref)
    excerpts = tmp / "excerpts.md"
    excerpts.write_text(lc.SELFTEST_EXCERPTS, encoding="utf-8")
    fixtures = REPO / "tests" / "fixtures" / "port-generator"
    saved = {k: sys.modules.get(k) for k in ("ttnn", "ttnn.experimental", "ttnn.transformer")}
    fake = lc._install_fake_ttnn()

    def run(**over):
        args = dict(snapshot=snap, construction=fixtures / "tiny-qwen4-construction.json", inventory=fixtures / "tiny-qwen4-ops-inventory.json",
                    excerpts=excerpts, reference_path=ref, device=fake.MeshDevice(1))
        args.update(over)
        return lc.device_phase(**args)
    yield type("Env", (), {"run": staticmethod(run), "ref": ref, "tmp": tmp})
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


@needs_reference
def test_the_reference_file_holds_the_ids_the_input_stream_each_layer_and_the_logits(env):
    ref = torch.load(str(env.ref), weights_only=True)
    assert len(ref["ids"]) == 24 and len(ref["layers"]) == 4
    assert ref["stream_in"].shape == (24, 256) and all(t.shape == (24, 256) for t in ref["layers"])
    assert ref["logits"].shape[0] == 24 and ref["hc_count"] == 4


@needs_reference
def test_layers_0_to_2_are_compared_and_ok_and_layer_3_is_skipped_for_its_stub(env):
    r = env.run()
    rows = {row["layer"]: row for row in r["layers"] if row["layer"] is not None}
    assert r["covered"] == [0, 1, 2] and r["skipped"] == [3] and r["raised"] == [] and r["ok"] and r["first_bad_layer"] is None
    for i in (0, 1, 2):
        assert rows[i]["status"] == "compared" and rows[i]["stream"]["pcc"] > 0.999 and rows[i]["update"]["pcc"] > 0.99
        assert rows[i]["update"]["ref_abs_max"] > 1e-3                      # the update is not trivially zero
    assert rows[3]["status"] == "skipped" and rows[3]["why"] == "stub: _C_self_attn"
    assert "_C_self_attn" in r["note"] and "not run" in r["note"] and r["stubs"] == ["_C_self_attn"] and r["warnings"] == []


@needs_reference
def test_a_port_generated_without_the_excerpts_is_flagged_by_the_update_and_the_warning(env):
    """Without the excerpts the generator uses the norm weights as stored, and the Qwen4 norms add 1: the stream still correlates (the residual
    input dominates it) but the layer's update does not. This is why the update is the number that decides ok."""
    r = env.run(excerpts=None)
    assert not r["ok"] and r["warnings"] and r["excerpts_given"] is False
    row0 = next(row for row in r["layers"] if row["layer"] == 0)
    assert row0["stream"]["pcc"] > 0.9 and not row0["update"]["pcc"] >= 0.5 and not row0["ok"]      # the update correlates at under 0.5, or has none (nan)
    assert r["first_bad_layer"] == 0


@needs_reference
def test_a_layer_that_raises_is_reported_with_the_port_function_and_line_and_the_layers_before_it_keep_their_numbers(env):
    def boom(mod):
        def _B_linear_attn(model, i, x, ids):
            raise RuntimeError("boom in the delta net")
        mod._B_linear_attn = _B_linear_attn
    r = env.run(patch=boom)
    rows = {row["layer"]: row for row in r["layers"] if row["layer"] is not None}
    assert rows[0]["status"] == "compared" and rows[0]["ok"]
    assert rows[1]["status"] == "raised" and "boom in the delta net" in rows[1]["error"] and rows[1]["error"].startswith("RuntimeError")
    assert rows[1]["port_function"] == "_layer_B" and isinstance(rows[1]["port_line"], int) and "_B_linear_attn(" in rows[1]["code"]
    assert rows[2]["status"] == "skipped" and rows[2]["why"] == "an earlier layer was not run"
    assert r["raised"] == [1] and not r["ok"] and "layer 1 raised" in r["note"]


@needs_reference
def test_the_selftest_function_passes_and_includes_its_negative_control(env):
    res = lc.selftest(24, qwen4_reference_python())
    assert res["selftest_ok"] and res["control_detected"] and res["covered"] == [0, 1, 2]
    assert res["controls"] == {"wrong_layer": True, "zero_update": True, "scaled_update": True}


# ---- mutations: remove each guard once, in a copy, and watch the check go red ---------------------------------------------------------------
# Two guards are not mutation-checked through the selftest: on the tiny model the stream's pcc alone already fails all three negative controls
# (a layer's update is a quarter of the stream there), so removing the update's pcc check, or making the update always ok, leaves the selftest
# green. They matter on a model whose updates are small against its stream; the unit checks of `compare` cover the pcc and relative bounds.

def _mutated(tmp_path: Path, old: str, new: str):
    text = SCRIPT.read_text(encoding="utf-8")
    assert old in text, f"the mutation target is not in the script: {old!r}"
    mutated = text.replace(old, new).replace("REPO = Path(__file__).resolve().parents[1]", f"REPO = Path({str(REPO)!r})")
    path = tmp_path / "lc_mutated.py"
    path.write_text(mutated, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("lc_mutated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SELFTEST_MUTATIONS = {
    "layer index": ('ref["layers"][i]', 'ref["layers"][(i + 1) % n_layers]'),
    "the update is the stream": ("got - prev_dev", "got"),
    "row slice": ("return got.reshape(-1, got.shape[-1])[:n]", "return got.reshape(-1, got.shape[-1])"),
    "stub detection": ("blocked = reaches(graph, fn.__name__, stubs)", "blocked = []"),
    "update relative error": ("UPDATE_REL_OK = 0.25 ", "UPDATE_REL_OK = 1e9 "),
    "update relative error is not applied": ("update = compare(got - prev_dev, ref[\"layers\"][i] - prev_ref, UPDATE_PCC_OK, UPDATE_REL_OK)",
                                             "update = compare(got - prev_dev, ref[\"layers\"][i] - prev_ref, UPDATE_PCC_OK)"),
    "control references": ("\"scaled_update\": {**ref, \"layers\": scaled}}", "\"scaled_update\": {**ref, \"layers\": layers}}"),
}


@needs_reference
@pytest.mark.parametrize("name", sorted(SELFTEST_MUTATIONS))
def test_a_mutated_script_fails_its_own_selftest(tmp_path, name):
    old, new = SELFTEST_MUTATIONS[name]
    mod = _mutated(tmp_path, old, new)
    try:
        res = mod.selftest(24, qwen4_reference_python())
    except Exception:  # noqa: BLE001 - a mutation that makes the run raise is caught as well
        return
    assert not res.get("selftest_ok"), f"the mutation {name!r} left the selftest green"


COMPARE_MUTATIONS = {
    "non-finite values": ("nonfinite == 0 and not math.isnan(pcc)", "not math.isnan(pcc)"),
    "the pcc threshold": ("pcc >= threshold", "pcc >= -1.0"),
    "nan pcc": ("not math.isnan(pcc) and pcc >= threshold", "(math.isnan(pcc) or pcc >= threshold)"),
    "pcc": ("pcc = float((gc * wc).sum() / denom) if denom > 0 else float(\"nan\")", "pcc = 1.0"),
    "relative bound": ("(rel_max is None or (not math.isnan(out[\"rel_l2\"]) and out[\"rel_l2\"] <= rel_max))", "True"),
}


@pytest.mark.parametrize("name", sorted(COMPARE_MUTATIONS))
def test_a_mutated_compare_is_caught_by_the_unit_checks(tmp_path, name):
    old, new = COMPARE_MUTATIONS[name]
    mod = _mutated(tmp_path, old, new)
    g = torch.Generator().manual_seed(1)
    t = torch.randn(24, 8, generator=g)
    nan_got = t.clone()
    nan_got[0, 0] = float("nan")
    checks = [mod.compare(nan_got, t)["ok"] is False,                                           # a non-finite value
              mod.compare(torch.randn(24, 8, generator=g), torch.randn(24, 8, generator=g))["ok"] is False,    # uncorrelated
              mod.compare(torch.zeros(24, 8), t)["ok"] is False,                                 # constant: pcc is nan
              mod.compare(1.5 * t, t, 0.95, 0.25)["ok"] is False,                               # perfectly correlated but 50% too large
              mod.compare(t, t.clone())["ok"] is True]
    assert not all(checks), f"the mutation {name!r} passed every unit check"
