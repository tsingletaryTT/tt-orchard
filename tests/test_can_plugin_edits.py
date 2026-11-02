# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 362 (tenstorrent/skills PR 40): a bundle that carries a change to the vLLM plugin or to vLLM needs a stated reason, and the gate
lists those files and refuses one without a reason. The reason lives in the spec's [plugin_edits] table."""
import shutil
from pathlib import Path

import pytest

from orchard import can
from test_can_vllm import SPEC, base, edit, needs_tt_model, spec  # noqa: F401  (fixtures)

EDIT_PATH = "extra_code/vllm_tt_plugin/patch.py"
REASON = ('[plugin_edits."%s"]\nreason = "the plugin has no option for this model\'s rotary layout, and the model cannot start without it"\n'
          'checked = "read vllm_tt_plugin/platform.py and the capability keys in the plugin README; none covers rotary layout"\n' % EDIT_PATH)
EXTRA = '[[extra_code]]\nsrc = "patch.py"\ndst = "vllm_tt_plugin/patch.py"\n'


def with_patch(base, reason=True, text=""):
    (base / "patch.py").write_text("X = 1\n")
    p = base / "can.toml"
    p.write_text(p.read_text() + "\n" + EXTRA + (REASON if reason else "") + text)
    return can.load_spec(p, base=base)


# ---- the spec table

def test_a_reason_table_loads(base):
    s = with_patch(base)
    assert set(s.plugin_edits) == {EDIT_PATH}
    assert s.plugin_edits[EDIT_PATH]["reason"].startswith("the plugin has no option")


@pytest.mark.parametrize("table,words", [
    ('[plugin_edits."a/vllm/x.py"]\nreason = "short"\nchecked = "read the plugin interface and found nothing"\n', "reason"),
    ('[plugin_edits."a/vllm/x.py"]\nreason = "a full sentence that says why the edit is needed"\n', "checked"),
    ('[plugin_edits."a/vllm/x.py"]\nreason = "a full sentence that says why the edit is needed"\nchecked = "ok"\n', "checked"),
    ('[plugin_edits."a/vllm/x.py"]\nreason = 5\nchecked = "read the plugin interface and found nothing"\n', "reason"),
    ('[plugin_edits."a/vllm/x.py"]\nreason = "a full sentence that says why the edit is needed"\nchecked = "read the plugin interface and found nothing"\nextra = 1\n', "unknown"),
    ('[plugin_edits]\n"a/vllm/x.py" = "just a string"\n', "table"),
    ('[plugin_edits."../vllm/x.py"]\nreason = "a full sentence that says why the edit is needed"\nchecked = "read the plugin interface and found nothing"\n', ".."),
])
def test_a_bad_reason_entry_is_refused(base, table, words):
    p = base / "can.toml"
    p.write_text(p.read_text() + "\n" + table)
    with pytest.raises(can.CanError) as exc:
        can.load_spec(p, base=base)
    assert words in str(exc.value)


# ---- the files the gate lists

def make_tree(tmp_path, names):
    for n in names:
        f = tmp_path / n
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x\n")
    return tmp_path


def test_plugin_and_vllm_files_are_found_by_path_component(tmp_path):
    root = make_tree(tmp_path, ["extra_code/vllm_tt_plugin/a.py", "vllm/entrypoints/b.py", "extra_code/vllm.py", "model.py",
                                "vllm_models/toy/vllm_metadata.json", "wheels/vllm_tt_plugin-0.1.0-py3-none-any.whl", "extra_code/notvllm/c.py"])
    assert can.plugin_edit_files(root) == ["extra_code/vllm.py", "extra_code/vllm_tt_plugin/a.py", "vllm/entrypoints/b.py"]


def test_problems_say_which_file_has_no_reason_and_which_reason_has_no_file(tmp_path):
    root = make_tree(tmp_path, ["extra_code/vllm_tt_plugin/a.py"])
    problems = can.plugin_edit_problems(root, {})
    assert len(problems) == 1 and "extra_code/vllm_tt_plugin/a.py" in problems[0] and "no stated reason" in problems[0]
    reasons = {"extra_code/vllm_tt_plugin/a.py": {"reason": "r" * 30, "checked": "c" * 30}, "extra_code/vllm/gone.py": {"reason": "r" * 30, "checked": "c" * 30}}
    problems = can.plugin_edit_problems(root, reasons)
    assert len(problems) == 1 and "gone.py" in problems[0] and "not in the bundle" in problems[0]
    del reasons["extra_code/vllm/gone.py"]
    assert can.plugin_edit_problems(root, reasons) == []


# ---- the gate

@needs_tt_model
def test_the_gate_passes_a_bundle_with_no_plugin_file(spec, tmp_path):
    out = tmp_path / "o" / "toy-moe-p300"
    can.stage(spec, out)
    assert can.plugin_edit_files(out) == []
    assert can.gate(out, spec) == []


@needs_tt_model
def test_the_gate_refuses_a_plugin_edit_with_no_reason(base, tmp_path):
    s = with_patch(base, reason=False)
    with pytest.raises(can.CanError) as exc:             # stage runs the gate and raises on a problem
        can.stage(s, tmp_path / "o1" / "toy-moe-p300")
    assert EDIT_PATH in str(exc.value) and "no stated reason" in str(exc.value)


@needs_tt_model
def test_the_gate_passes_a_plugin_edit_with_a_reason_and_lists_the_file(base, tmp_path):
    s = with_patch(base)
    out = tmp_path / "o2" / "toy-moe-p300"
    can.stage(s, out)
    assert can.plugin_edit_files(out) == [EDIT_PATH]
    assert can.gate(out, s) == []
    # the same bundle checked against a spec that has lost the reason is refused
    bare = can.load_spec(base / "can.toml", base=base)
    object.__setattr__(bare, "plugin_edits", {})
    assert any(EDIT_PATH in p and "no stated reason" in p for p in can.gate(out, bare))
