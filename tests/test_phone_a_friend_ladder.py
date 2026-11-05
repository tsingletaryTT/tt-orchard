# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Phone a friend, round 2 (run9 of the tiny Qwen4 rehearsal).

Run9's first helper call went to the cheapest model and came back as a stub whose `decode_step` raised NotImplementedError. The coder then
read the excerpts file eight times and the watchdog stopped it. Three changes, tested here:

1. a draft that is a stub (a required function missing, empty, or raising NotImplementedError) counts as "not helped" at once, so the next
   ask in the stage goes one step up the model ladder, and the ledger says why;
2. (tests/test_port_excerpts_follow.py) the excerpts follow the classes a kept class builds;
3. a draft that parses and is not a stub is put into `stages/2/port/port_model.py` by the supervisor when that file is still the unchanged
   skeleton, so the check runs against real code. A file the coder has written is never replaced."""
import ast

import pytest

from orchard import phone_a_friend as paf
from orchard import port_facts
from orchard import supervisor as sv
from orchard import telegraph as tg
from orchard.supervisor import EXIT_ABORTED, Control
from port_check_fakes import HONEST
from run_fakes import where
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, make, with_snapshot
from test_supervisor_phone_a_friend import coder, helper_reply, telegraph_decisions
from test_supervisor_port_repair import BROKEN
from test_telegraph import KEY, Wire, openai_reply, table

STUB = '''
def load(model_dir, mesh_device):
    tensors = read_tensors(model_dir)
    return {"t": tensors}

def prefill(model, ids):
    return None

def decode_step(model, token_id, position):
    raise NotImplementedError("hyper-connection mixer and PLE: source not in excerpts")

def close(model):
    model.clear()
'''

REAL = '''
import math

_SCALE = 2.0

def _scaled(x):
    return [v * _SCALE for v in x]

def load(model_dir, mesh_device):
    t = read_tensors(model_dir)
    return {"t": t, "device": mesh_device}

def prefill(model, ids):
    return _scaled([float(i) for i in ids])

def decode_step(model, token_id, position):
    return _scaled([float(token_id), float(position)])

def close(model):
    model.clear()
'''

SKELETON = port_facts.SKELETON.read_text()


# ---- what counts as a stub ------------------------------------------------------------------------------------------------------------------

def test_a_draft_that_raises_not_implemented_is_a_stub_and_the_reason_names_the_function():
    problems = paf.draft_problems(STUB)
    assert any("decode_step" in p and "NotImplementedError" in p for p in problems)


def test_a_body_that_is_only_pass_or_a_docstring_or_an_ellipsis_is_empty():
    src = 'def load(a, b):\n    pass\n\ndef prefill(m, i):\n    """x"""\n\ndef decode_step(m, t, p):\n    ...\n\ndef close(m):\n    return None\n'
    problems = paf.draft_problems(src)
    assert sum("empty" in p for p in problems) == 3 and not any("close" in p for p in problems)


def test_a_missing_function_is_named():
    problems = paf.draft_problems("def load(a, b):\n    return 1\n")
    assert {"prefill", "decode_step", "close"} <= {p.split()[1] for p in problems if p.startswith("no ")}


def test_a_complete_draft_has_no_problems_and_close_may_be_trivial():
    assert paf.draft_problems(REAL) == []
    assert paf.draft_problems(HONEST) == []                      # its close() is `pass`: the only function allowed to be empty


def test_a_draft_that_does_not_parse_has_no_problem_list_from_here():
    assert paf.draft_problems("def load(:\n") == ["the draft does not parse"]


# ---- putting a draft into the skeleton ---------------------------------------------------------------------------------------------------

def test_the_merge_keeps_the_skeletons_imports_and_read_tensors_and_takes_the_drafts_functions_and_helpers():
    merged, why = paf.merge_into_skeleton(SKELETON, REAL)
    assert merged is not None, why
    tree = ast.parse(merged)
    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"read_tensors", "load", "prefill", "decode_step", "close", "_scaled"} <= set(top)
    assert "import ttnn" in merged and "from safetensors import safe_open" in merged and "import math" in merged
    assert "_SCALE = 2.0" in merged
    assert "raise NotImplementedError" not in merged            # every skeleton body was replaced
    assert merged.count("def load(") == 1 and merged.count("def read_tensors(") == 1


def test_an_import_the_skeleton_already_has_is_not_repeated():
    merged, _ = paf.merge_into_skeleton(SKELETON, "import ttnn\nimport torch\nimport math\n" + REAL)
    assert merged.count("import ttnn") == 1 and merged.count("import torch") == 1 and merged.count("import math") == 1


def test_the_merge_refuses_a_draft_without_all_four_functions_or_with_a_stub():
    assert paf.merge_into_skeleton(SKELETON, "def load(a, b):\n    return 1\n")[0] is None
    assert paf.merge_into_skeleton(SKELETON, STUB)[0] is None
    assert paf.merge_into_skeleton(SKELETON, "def load(:\n")[0] is None


def test_a_draft_that_defines_read_tensors_replaces_the_skeletons():
    src = "def read_tensors(model_dir):\n    return {'mine': 1}\n" + REAL
    merged, _ = paf.merge_into_skeleton(SKELETON, src)
    assert merged.count("def read_tensors(") == 1 and "'mine'" in merged


def stage_with(tmp_path, port_text, draft=REAL):
    stage = tmp_path / "stage"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(port_text)
    draft_path = paf.write_draft(stage, 1, draft, "abc123", "helper-model")
    return stage, draft_path


def test_the_draft_is_placed_over_the_unchanged_skeleton_and_the_result_says_what_was_done(tmp_path):
    stage, draft_path = stage_with(tmp_path, SKELETON)
    placed, why = paf.place_draft(stage, draft_path, SKELETON.encode())
    assert placed, why
    text = (stage / "port" / "port_model.py").read_text()
    assert "def decode_step" in text and "_scaled" in text and text.startswith("# Placed by the supervisor")
    assert "draft-1.py" in text.splitlines()[0] or "draft-1.py" in text.splitlines()[1]


@pytest.mark.parametrize("edit", ["# coder edit\n", "def extra():\n    pass\n"])
def test_a_file_the_coder_changed_is_never_replaced(tmp_path, edit):
    stage, draft_path = stage_with(tmp_path, SKELETON + "\n" + edit)
    before = (stage / "port" / "port_model.py").read_bytes()
    placed, why = paf.place_draft(stage, draft_path, SKELETON.encode())
    assert not placed and "not the unchanged skeleton" in why
    assert (stage / "port" / "port_model.py").read_bytes() == before


def test_a_stub_draft_is_not_placed(tmp_path):
    stage, draft_path = stage_with(tmp_path, SKELETON, draft=STUB)
    placed, why = paf.place_draft(stage, draft_path, SKELETON.encode())
    assert not placed and "NotImplementedError" in why
    assert (stage / "port" / "port_model.py").read_text() == SKELETON


def test_a_missing_port_file_or_a_link_in_its_place_is_not_written_through(tmp_path):
    stage, draft_path = stage_with(tmp_path, SKELETON)
    target = tmp_path / "outside.py"
    target.write_text(SKELETON)
    port = stage / "port" / "port_model.py"
    port.unlink()
    port.symlink_to(target)
    placed, why = paf.place_draft(stage, draft_path, SKELETON.encode())
    assert not placed and target.read_text() == SKELETON
    port.unlink()
    placed, why = paf.place_draft(stage, draft_path, SKELETON.encode())
    assert not placed and not port.exists()


# ---- in the supervisor ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    yield r
    r.close()


def wire_two_models(rig, monkeypatch, tmp_path, reply):
    wire = Wire(reply)
    cfg = tg.load(table(models=["small-model", "big-model"], prices={"small-model": [1.0, 5.0], "big-model": [3.0, 15.0]}), base=tmp_path)
    tel = tg.Telegraph(cfg, tmp_path / "tg", transport=wire, environ={"TG_KEY": KEY}, clock=lambda: 1000.0, hostname="node9", home="/home/someone")
    monkeypatch.setattr(sv, "_telegraph_from", lambda *a, **k: tel)
    return tel, wire


def stub_reply(body):
    return openai_reply(f"Here.\n```python\n{STUB}\n```\n", 900, 300)


def test_a_stub_draft_is_reported_as_not_helping_at_once_and_the_ledger_says_why(rig, monkeypatch, tmp_path):
    tel, wire = wire_two_models(rig, monkeypatch, tmp_path, stub_reply)
    coder(rig, uses_the_draft=False)
    rig.run()
    names = [d["decision"] for d in telegraph_decisions(rig) + [d for d in rig.decisions() if d.get("decision") == paf.DRAFT_STUB]]
    assert paf.DRAFT_STUB in names
    stub = next(d for d in rig.decisions() if d.get("decision") == paf.DRAFT_STUB)
    assert any("decode_step" in p for p in stub["problems"]) and "next model" in stub["effect"]
    reports = [d for d in rig.decisions() if d.get("decision") == paf.REPORTED and d.get("call_id") == stub["call_id"]]
    assert reports and reports[0]["helped"] is False and "stub" in reports[0]["reason"]
    assert wire.requests[0]["body"]["model"] == "small-model"


def test_after_a_stub_the_next_ask_in_the_same_problem_goes_to_the_next_model(rig, monkeypatch, tmp_path):
    tel, wire = wire_two_models(rig, monkeypatch, tmp_path, stub_reply)
    coder(rig, uses_the_draft=False)
    rig.run()
    tel.ask(paf.PROBLEM, "task", "failure")
    assert wire.requests[-1]["body"]["model"] == "big-model"


def test_a_real_draft_that_the_stage_then_passes_with_keeps_the_first_model_for_the_next_ask(rig, monkeypatch, tmp_path):
    tel, wire = wire_two_models(rig, monkeypatch, tmp_path, helper_reply(REAL))
    coder(rig, uses_the_draft=True)                       # the coder follows the note and writes the working port: the attempt passes
    rig.run()
    assert not [d for d in rig.decisions() if d.get("decision") == paf.DRAFT_STUB]
    assert [d["helped"] for d in rig.decisions() if d.get("decision") == paf.REPORTED] == [True]
    tel.ask(paf.PROBLEM, "task", "failure")
    assert wire.requests[-1]["body"]["model"] == "small-model"


def test_a_real_draft_is_placed_over_the_fresh_skeleton_and_the_ledger_holds_both_hashes(rig, monkeypatch, tmp_path):
    wire_two_models(rig, monkeypatch, tmp_path, helper_reply(REAL))
    prepares = coder(rig, uses_the_draft=False)
    rig.run()
    placed = [d for d in rig.decisions() if d.get("decision") == paf.PLACED]
    assert placed, [d.get("decision") for d in rig.decisions()]
    assert len(placed[0]["draft_sha256"]) == 64 and len(placed[0]["port_sha256"]) == 64 and placed[0]["path"] == "stages/2/port/port_model.py"
    assert "has put that draft into `stages/2/port/port_model.py`" in prepares[-1]       # the coder is told the file already holds it


def test_a_draft_that_was_not_placed_is_described_to_the_coder_as_a_file_to_copy(rig, monkeypatch, tmp_path):
    wire_two_models(rig, monkeypatch, tmp_path, helper_reply(REAL))
    monkeypatch.setattr(paf, "place_draft", lambda *a, **k: (False, "not the unchanged skeleton"))
    prepares = coder(rig, uses_the_draft=False)
    rig.run()
    assert "has put that draft" not in prepares[-1] and "write it to `stages/2/port/port_model.py`" in prepares[-1]


def test_nothing_is_placed_when_the_coder_already_wrote_the_port(rig, monkeypatch, tmp_path):
    # A prepare that has already replaced the skeleton before the helper is asked cannot be overwritten: here the ask happens at the start of
    # the attempt, so the test hands `_phone_a_friend` a port the coder changed.
    wire_two_models(rig, monkeypatch, tmp_path, helper_reply(REAL))
    original = paf.place_draft

    def changed_first(stage_dir, draft_path, skeleton_bytes):
        port = stage_dir / "port" / "port_model.py"
        port.write_text(port.read_text() + "\n# the coder was here\n")
        return original(stage_dir, draft_path, skeleton_bytes)
    monkeypatch.setattr(paf, "place_draft", changed_first)
    coder(rig, uses_the_draft=False)
    rig.run()
    refused = [d for d in rig.decisions() if d.get("decision") == paf.NOT_PLACED]
    assert refused and "not the unchanged skeleton" in refused[0]["reason"]
    assert not [d for d in rig.decisions() if d.get("decision") == paf.PLACED]


# ---- a later attempt that is not due for the helper starts from the last real draft ----------------------------------------------------------

def test_standing_draft_is_the_highest_numbered_regular_file_in_a_real_directory(tmp_path):
    stage = tmp_path / "stage"
    assert paf.standing_draft(stage) is None
    (stage / "telegraph").mkdir(parents=True)
    for name in ("draft-1.py", "draft-2.py", "draft-10.py", "draft-x.py", "notes.txt"):
        (stage / "telegraph" / name).write_text("x = 1\n")
    (stage / "telegraph" / "draft-11.py").symlink_to(stage / "telegraph" / "draft-1.py")
    assert paf.standing_draft(stage).name == "draft-10.py"


def test_a_link_in_place_of_the_telegraph_directory_has_no_standing_draft(tmp_path):
    stage = tmp_path / "stage"
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "draft-1.py").write_text("x = 1\n")
    stage.mkdir()
    (stage / "telegraph").symlink_to(tmp_path / "elsewhere")
    assert paf.standing_draft(stage) is None


def test_an_attempt_that_is_not_due_for_the_helper_starts_from_a_standing_real_draft(rig, monkeypatch, tmp_path):
    wire_two_models(rig, monkeypatch, tmp_path, helper_reply(REAL))
    stage = rig.run_dir / "stages" / "2.partial-1"                       # the supervisor moves a failed attempt's stage directory aside
    stage.mkdir(parents=True, exist_ok=True)
    paf.write_draft(stage, 1, REAL, "earlier", "helper-model")
    coder(rig, uses_the_draft=False)
    rig.run()
    placed = [d for d in rig.decisions() if d.get("decision") == paf.PLACED and d.get("call_id") == ""]
    assert placed and placed[0]["path"] == "stages/2/port/port_model.py"


def test_a_standing_stub_is_not_placed_and_says_why(rig, monkeypatch, tmp_path):
    wire_two_models(rig, monkeypatch, tmp_path, helper_reply(REAL))
    stage = rig.run_dir / "stages" / "2.partial-1"
    stage.mkdir(parents=True, exist_ok=True)
    paf.write_draft(stage, 1, STUB, "earlier", "helper-model")
    coder(rig, uses_the_draft=False)
    rig.run()
    refused = [d for d in rig.decisions() if d.get("decision") == paf.NOT_PLACED and d.get("call_id") == ""]
    assert refused and "NotImplementedError" in refused[0]["reason"]


def test_a_draft_in_a_moved_aside_stage_directory_is_found_and_the_newest_partial_wins(tmp_path):
    base = tmp_path / "stages"
    for n, k in ((1, 1), (2, 1)):
        d = base / f"2.partial-{n}"
        d.mkdir(parents=True)
        paf.write_draft(d, k, f"x = {n}\n", f"c{n}", "m")
    (base / "2").mkdir()
    assert paf.standing_draft(base / "2").parent.parent.name == "2.partial-2"
    paf.write_draft(base / "2", 3, "x = 9\n", "c9", "m")                     # a draft in the live directory comes first
    assert paf.standing_draft(base / "2").parent.parent.name == "2"


def test_a_due_attempt_asks_and_places_the_new_draft_and_the_standing_draft_is_left_alone(rig, monkeypatch, tmp_path):
    wire_two_models(rig, monkeypatch, tmp_path, helper_reply(REAL))
    stage = rig.run_dir / "stages" / "2.partial-1"
    stage.mkdir(parents=True, exist_ok=True)
    paf.write_draft(stage, 1, REAL, "earlier", "helper-model")
    coder(rig, uses_the_draft=False)
    rig.run()
    placed = [d for d in rig.decisions() if d.get("decision") == paf.PLACED]
    assert [d["call_id"] == "" for d in placed] == [True, False]              # attempt 1 starts from the standing draft; attempt 2 from the new one
    assert not [d for d in rig.decisions() if d.get("decision") == paf.NOT_PLACED]
