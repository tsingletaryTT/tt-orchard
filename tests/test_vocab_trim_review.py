# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The four findings of Copilot's review of PR 57 (the vocabulary trim): each was real and each is reproduced here before its fix."""
import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_container_template import drafter_env, edit, paths, pairs, printed, without_drafter  # noqa: E402,F401
from test_delta_triage_vocab_trim import BASE_VOCAB, MODEL_VOCAB, World, analyse, cfg, template, write_st  # noqa: E402,F401
from test_container_template import crig  # noqa: E402,F401
from test_weights_swap_templates import prep, run_prepare  # noqa: E402,F401


# ---- 1. a header whose bodies are missing is not a verified subset -------------------------------------------------------------

def cut_after_header(path: Path, keep_body: int = 0):
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    path.write_bytes(raw[:8 + n + keep_body])


@pytest.mark.parametrize("which", ["model", "base", "both"])
def test_tensor_bodies_that_are_missing_leave_the_comparison_unread_and_the_class_full_port(tmp_path, which):
    w = World(tmp_path)
    for side in (("model", "base") if which == "both" else (which,)):
        cut_after_header(getattr(w, side) / "model.safetensors")
    t, c, tens, sc, vt = analyse(w)
    assert vt["trimmed"] is True and vt["kept_subset"] is None and vt["tensors"] == []
    assert "shorter" in vt["note"] or "read" in vt["note"], vt["note"]
    assert t.decide_class(c, tens, t.decide_path(c, tens), sc, vt) == "full-port"


def test_a_partly_present_body_is_not_a_verified_subset_either(tmp_path):
    w = World(tmp_path)
    cut_after_header(w.model / "model.safetensors", keep_body=40)
    vt = analyse(w)[4]
    assert vt["kept_subset"] is None


def test_a_header_whose_byte_range_does_not_hold_the_rows_is_unread(tmp_path):
    """A file long enough for the read, but a header that says a smaller byte range than the shape needs."""
    w = World(tmp_path)
    path = w.model / "model.safetensors"
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    off = header["model.embed_tokens.weight"]["data_offsets"]
    header["model.embed_tokens.weight"]["data_offsets"] = [off[0], off[1] - 8]
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + n:])
    vt = analyse(w)[4]
    assert vt["trimmed"] is True and vt["kept_subset"] is None and "byte range" in vt["note"]
    assert vt["tensors"] == []        # lm_head is read first and was fine; its row count must not survive the failure on the embedding


def test_a_short_read_is_an_error_not_a_row():
    t = template()
    with pytest.raises(ValueError):
        t._row_digests({"file": Path(__file__), "start": Path(__file__).stat().st_size - 3}, 0, 2, 8)


# ---- 2. the compared matrices must have the configured numbers of rows ------------------------------------------------------------

def test_matrices_whose_rows_are_not_the_configured_vocabulary_are_not_a_trim(tmp_path):
    t, c, tens, sc, vt = analyse(World(tmp_path), c=cfg(model_vocab=MODEL_VOCAB + 1, base_vocab=BASE_VOCAB + 1))
    assert vt["trimmed"] is False and "rows" in vt["note"]
    assert t.decide_class(c, tens, t.decide_path(c, tens), sc, vt) == "full-port"


def test_one_side_off_by_one_is_not_a_trim(tmp_path):
    for i, (mv, bv) in enumerate(((MODEL_VOCAB + 1, BASE_VOCAB), (MODEL_VOCAB, BASE_VOCAB + 1))):
        d = tmp_path / str(i)
        d.mkdir()
        assert analyse(World(d), c=cfg(model_vocab=mv, base_vocab=bv))[4]["trimmed"] is False, (mv, bv)


def test_matrices_with_the_configured_rows_are_still_a_trim(tmp_path):
    assert analyse(World(tmp_path))[4]["trimmed"] is True


# ---- 3. stage 7's graft planner refuses a different vocabulary too ----------------------------------------------------------------

def snapshot(root: Path, name: str, config: dict, tensors: dict):
    snap = root / name
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(config))
    write_st(snap / "model-00001-of-00001.safetensors", tensors)
    (snap / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}))
    return snap


def test_the_stage_7_graft_plan_refuses_a_model_with_another_vocabulary(tmp_path):
    from orchard import package
    base = snapshot(tmp_path, "a" * 40, {"text_config": {"vocab_size": 248320, "hidden_size": 4}},
                    {"model.embed_tokens.weight": np.ones((4, 4)), "mtp.fc.weight": np.ones((4, 8)), "mtp.norm.weight": np.ones((4,))})
    new = snapshot(tmp_path, "b" * 40, {"text_config": {"vocab_size": 181760, "hidden_size": 4}},
                   {"model.embed_tokens.weight": np.ones((4, 4))})
    with pytest.raises(package.PackageError) as e:
        package.plan_graft("Qwen/Qwen3.8-27B", base, new)
    assert "vocabulary" in str(e.value) and "181760" in str(e.value) and "248320" in str(e.value)


def test_the_stage_7_graft_plan_still_works_for_the_same_vocabulary(tmp_path):
    from orchard import package
    cfg_ = {"text_config": {"vocab_size": 248320, "hidden_size": 4}}
    base = snapshot(tmp_path, "a" * 40, cfg_, {"model.embed_tokens.weight": np.ones((4, 4)), "mtp.fc.weight": np.ones((4, 8)),
                                               "mtp.norm.weight": np.ones((4,))})
    new = snapshot(tmp_path, "b" * 40, cfg_, {"model.embed_tokens.weight": np.ones((4, 4))})
    plan = package.plan_graft("Qwen/Qwen3.8-27B", base, new)
    assert plan["repo"] == "Qwen/Qwen3.8-27B" and plan["tensors"] == ["mtp.fc.weight", "mtp.norm.weight"]


# ---- 4. a trim records the drafter as off even when the original setting is absent ------------------------------------------------

BASE_CFG = {"text_config": {"vocab_size": 248320}}
MODEL_CFG = {"text_config": {"vocab_size": 181760}}


def test_a_bundle_run_script_with_no_drafter_line_gets_an_explicit_off_line_for_a_trim(prep):
    from test_prepare_swap_vocab import configs
    configs(prep)
    assert "QWEN36_DRAFTER" not in (prep["bundle"] / "run.sh").read_text()
    assert run_prepare(prep["stage"]).returncode == 0
    lines = (prep["stage"] / "run.sh").read_text().splitlines()
    marks = [i for i, l in enumerate(lines) if "QWEN36_DRAFTER" in l]
    assert [lines[i] for i in marks] == ['export QWEN36_DRAFTER=""']
    assert marks[0] < next(i for i, l in enumerate(lines) if l.lstrip().startswith("CMD=("))
    sc = _serving(prep["stage"] / "run.sh")
    assert sc["drafter"] == "off"


def _serving(run_sh: Path) -> dict:
    import importlib.util
    spec = importlib.util.spec_from_file_location("sac_under_test", Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates" / "serve_and_compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.serving_state(run_sh.read_text())


def test_no_line_is_added_when_the_vocabulary_is_the_same(prep):
    from test_prepare_swap_vocab import configs
    configs(prep, model=BASE_CFG)
    assert run_prepare(prep["stage"]).returncode == 0
    assert "QWEN36_DRAFTER" not in (prep["stage"] / "run.sh").read_text()


def test_a_model_without_mtp_tensors_and_no_drafter_line_is_unchanged(prep):
    """Only a trim is made explicit: the earlier rule for a model with no mtp.* tensors keeps its behaviour."""
    from test_weights_swap_templates import new_index
    new_index(prep, ["model.embed_tokens.weight", "lm_head.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    assert "QWEN36_DRAFTER" not in (prep["stage"] / "run.sh").read_text()


def test_a_run_script_with_no_cmd_line_and_a_trim_is_refused(prep):
    from test_prepare_swap_vocab import configs
    configs(prep)
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace("CMD=(", "COMMAND=("))
    r = run_prepare(prep["stage"])
    assert r.returncode == 2 and "QWEN36_DRAFTER" in (r.stdout + r.stderr)


def test_a_container_command_with_no_drafter_option_gets_an_explicit_off_option_for_a_trim(paths):
    argv = edit(paths, printed=without_drafter(paths), drafter_off=True, explicit_off=True)
    assert drafter_env(argv) == ["QWEN36_DRAFTER="]
    from test_container_template import sac
    assert sac.serving_from_argv(argv)["drafter"] == "off"


def test_the_explicit_option_is_added_only_when_asked_and_only_when_absent(paths):
    assert drafter_env(edit(paths, printed=without_drafter(paths))) == []
    assert drafter_env(edit(paths, printed=without_drafter(paths), drafter_off=True)) == []
    assert drafter_env(edit(paths, drafter_off=True, explicit_off=True)) == ["QWEN36_DRAFTER="]       # present: cleared, not repeated
    assert drafter_env(edit(paths, explicit_off=True)) == ["QWEN36_DRAFTER=mtp"]                    # asked to clear nothing: left alone


def test_the_explicit_option_leaves_every_other_argument_as_it_was(paths):
    plain, explicit = edit(paths, printed=without_drafter(paths)), edit(paths, printed=without_drafter(paths), drafter_off=True, explicit_off=True)
    rest = list(explicit)
    k = rest.index("QWEN36_DRAFTER=")
    del rest[k - 1:k + 1]
    assert rest == plain


def _swap_check(crig):
    return json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())


def test_a_container_package_that_sets_no_drafter_is_served_with_an_explicit_off_for_a_trim(crig):
    from test_container_template import run_env_values, set_vocab_record
    set_vocab_record(crig, {"model": 181760, "base": 248320, "trimmed": True})
    r = crig["start"](printed={"drafter": False})
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == [""]
    assert _swap_check(crig)["serving"]["drafter"] == "off"


def test_the_same_package_without_a_trim_record_is_left_with_no_drafter_option(crig):
    from test_container_template import run_env_values
    r = crig["start"](printed={"drafter": False})
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == []
    assert _swap_check(crig)["serving"]["drafter"] == "not set by the package"
