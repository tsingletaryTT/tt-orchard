# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The base control for a benchmark (orchard/bench_control.py): its run.sh comes from the model's own run.sh, and the board comes from `gozer status`.

Nothing here touches a chip. `gozer status --json` is a canned payload in the shape the real command prints."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import bench_control as bc
from orchard import bench_session as bs

MODEL_DIR = "/runs/m/stages/4/configs/2/model-dir"
BASE = "/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f"

RUN_SH = f"""#!/bin/bash
HERE="/pkg"
export HF_HOME="${{HF_HOME:-$HERE/.hf}}"
export HF_MODEL="{MODEL_DIR}"
export TT_MODEL_WEIGHTS_REVISION="${{TT_MODEL_WEIGHTS_REVISION:-1d4bf0f}}"
export QWEN36_DRAFTER=""
export DFLASH_WEIGHTS="incoai/Qwen3.8-27B-DFlash2@dedf8df"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model {MODEL_DIR} --max_num_seqs 4 --additional-config '{{"tt": {{"l1_small_size": 24576}}}}' "$@")
"$CMD" "$@"
"""


def build():
    return bc.control_run_sh(RUN_SH, MODEL_DIR, BASE)


# ---- the control run.sh ---------------------------------------------------------------------------------------------------------------

def test_the_control_swaps_the_model_path_in_the_environment_line_and_the_model_flag():
    text = build()
    assert f'export HF_MODEL="{BASE}"' in text
    assert f"--model {BASE} --max_num_seqs 4" in text
    assert MODEL_DIR not in text


def test_the_control_differs_from_the_models_script_in_those_two_lines_and_no_other():
    diff = bc.differing_lines(RUN_SH, build())
    assert [n for n, _, _ in diff] == [4, 8]


def test_the_control_keeps_the_flags_the_drafter_setting_and_the_sampling_mode_of_the_model():
    text = build()
    assert 'export QWEN36_DRAFTER=""' in text and "--max_num_seqs 4" in text and "l1_small_size" in text


def test_a_model_path_the_script_does_not_contain_is_refused():
    with pytest.raises(bc.ControlError, match="does not contain"):
        bc.control_run_sh(RUN_SH, "/elsewhere/model-dir", BASE)


def test_a_script_that_names_the_model_only_in_one_place_is_refused():
    only_env = RUN_SH.replace(f"--model {MODEL_DIR}", "--model /other")
    with pytest.raises(bc.ControlError, match="--model"):
        bc.control_run_sh(only_env, MODEL_DIR, BASE)
    only_flag = RUN_SH.replace(f'export HF_MODEL="{MODEL_DIR}"', 'export HF_MODEL="/other"')
    with pytest.raises(bc.ControlError, match="HF_MODEL"):
        bc.control_run_sh(only_flag, MODEL_DIR, BASE)


def test_a_base_snapshot_equal_to_the_model_directory_is_refused():
    with pytest.raises(bc.ControlError, match="same"):
        bc.control_run_sh(RUN_SH, MODEL_DIR, MODEL_DIR)


def test_a_longer_path_that_starts_with_the_model_directory_is_not_swapped():
    longer = RUN_SH + f"# note {MODEL_DIR}-old/keep\n"
    text = bc.control_run_sh(longer, MODEL_DIR, BASE)
    assert f"{MODEL_DIR}-old/keep" in text


def test_a_sibling_directory_sharing_the_prefix_does_not_count_as_the_model_path():
    only_sibling = RUN_SH.replace(MODEL_DIR, MODEL_DIR + "2")
    with pytest.raises(bc.ControlError, match="does not contain"):
        bc.control_run_sh(only_sibling, MODEL_DIR, BASE)


# ---- proving the control is comparable -----------------------------------------------------------------------------------------------

def test_a_control_built_here_has_no_problems():
    assert bc.verify_control(RUN_SH, build(), MODEL_DIR, BASE) == []


def test_a_control_with_another_flag_changed_is_reported_with_its_line():
    bad = build().replace("--max_num_seqs 4", "--max_num_seqs 8")
    problems = bc.verify_control(RUN_SH, bad, MODEL_DIR, BASE)
    assert len(problems) == 1 and "line 8" in problems[0]


def test_a_control_with_the_drafter_switched_on_is_reported():
    bad = build().replace('QWEN36_DRAFTER=""', 'QWEN36_DRAFTER="dflash"')
    problems = bc.verify_control(RUN_SH, bad, MODEL_DIR, BASE)
    assert len(problems) == 1 and "line 6" in problems[0]


def test_a_control_with_a_different_number_of_lines_is_reported():
    problems = bc.verify_control(RUN_SH, build() + "echo more\n", MODEL_DIR, BASE)
    assert problems and "lines" in problems[0]


def test_a_control_identical_to_the_model_script_is_reported_because_it_serves_the_model_again():
    problems = bc.verify_control(RUN_SH, RUN_SH, MODEL_DIR, BASE)
    assert problems and "no line differs" in problems[0]


def test_a_control_that_swaps_the_path_in_only_one_line_is_reported():
    half = RUN_SH.replace(f'export HF_MODEL="{MODEL_DIR}"', f'export HF_MODEL="{BASE}"')
    problems = bc.verify_control(RUN_SH, half, MODEL_DIR, BASE)
    assert problems and any("still names" in p for p in problems)


def test_building_checks_the_control_it_made_and_refuses_one_that_fails(tmp_path, monkeypatch):
    model_sh = tmp_path / "run.sh"
    model_sh.write_text(RUN_SH)
    monkeypatch.setattr(bc, "control_run_sh", lambda text, md, bs: text.replace("--max_num_seqs 4", "--max_num_seqs 9"))
    out = tmp_path / "base-run.sh"
    with pytest.raises(bc.ControlError, match="more than the model path"):
        bc.build_control(model_run_sh=model_sh, model_dir=MODEL_DIR, base_snapshot=BASE, base_id="x", out=out)
    assert not out.exists()


def test_the_record_says_the_serving_modes_differ_when_they_do(tmp_path, monkeypatch):
    model_sh = tmp_path / "run.sh"
    model_sh.write_text(RUN_SH)
    modes = iter([{"drafter": "off", "sampling": "host"}, {"drafter": "on", "sampling": "host"}])
    monkeypatch.setattr(bc, "serving_facts", lambda text: next(modes))
    rec = bc.build_control(model_run_sh=model_sh, model_dir=MODEL_DIR, base_snapshot=BASE, base_id="x", out=tmp_path / "o.sh")
    assert rec["same_serving"] is False and rec["serving_control"]["drafter"] == "on"


# ---- serving facts ----------------------------------------------------------------------------------------------------------------------

def test_the_serving_facts_read_the_drafter_and_the_sampling_mode_from_the_script():
    assert bc.serving_facts(RUN_SH) == {"drafter": "off", "sampling": "host"}
    on = RUN_SH.replace('QWEN36_DRAFTER=""', 'QWEN36_DRAFTER="dflash"').replace("--max_num_seqs 4", '--additional-config {"sample_on_device_mode": "all"}')
    assert bc.serving_facts(on) == {"drafter": "on", "sampling": "device"}


def test_a_script_with_no_drafter_line_says_it_is_not_set():
    assert bc.serving_facts("#!/bin/bash\necho hi\n")["drafter"] == "not set"


# ---- the record ---------------------------------------------------------------------------------------------------------------------------

def test_the_record_names_both_scripts_their_hashes_the_swapped_lines_and_the_serving_mode(tmp_path):
    model_sh = tmp_path / "run.sh"
    model_sh.write_text(RUN_SH)
    out = tmp_path / "base-run.sh"
    rec = bc.build_control(model_run_sh=model_sh, model_dir=MODEL_DIR, base_snapshot=BASE, base_id="Qwen/Qwen3.8-27B@1d4bf0f", out=out)
    assert out.read_text() == build()
    assert rec["built_from"] == str(model_sh) and rec["control"] == str(out)
    assert rec["built_from_sha256"] != rec["control_sha256"]
    import hashlib
    assert rec["control_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert rec["swapped_lines"] == [4, 8]
    assert rec["model_dir"] == MODEL_DIR and rec["base_snapshot"] == BASE and rec["base_id"] == "Qwen/Qwen3.8-27B@1d4bf0f"
    assert rec["serving_model"] == rec["serving_control"] == {"drafter": "off", "sampling": "host"}
    assert rec["same_serving"] is True
    assert json.loads(Path(str(out) + ".control.json").read_text()) == rec


def test_building_over_an_unverifiable_script_writes_nothing(tmp_path):
    model_sh = tmp_path / "run.sh"
    model_sh.write_text(RUN_SH.replace(f"--model {MODEL_DIR}", "--model /other"))
    out = tmp_path / "base-run.sh"
    with pytest.raises(bc.ControlError):
        bc.build_control(model_run_sh=model_sh, model_dir=MODEL_DIR, base_snapshot=BASE, base_id="x", out=out)
    assert not out.exists() and not Path(str(out) + ".control.json").exists()


# ---- choosing a board ---------------------------------------------------------------------------------------------------------------

def chip(dev, board, state="FREE"):
    return {"bdf": f"0000:0{dev + 1}:00.0", "board": board, "card": "p300c", "dev_index": dev, "state": state}


def status(*states):
    """Two boards of two chips; `states` are the four chips' states in order."""
    boards = ["B0", "B0", "B1", "B1"]
    return {"grain": "board", "chips": [chip(i, boards[i], s) for i, s in enumerate(states)]}


def test_a_free_board_is_chosen_and_named_by_the_device_index_of_its_first_chip():
    plan = bc.plan_boards(status("HELD", "HELD", "FREE", "FREE"), [2])
    assert plan == [{"board": "B1", "chips": [2, 3], "exact": 2}]


def test_the_lowest_free_board_comes_first_when_both_are_free():
    assert bc.plan_boards(status("FREE", "FREE", "FREE", "FREE"), [2])[0]["board"] == "B0"


def test_two_sessions_get_two_different_boards():
    plan = bc.plan_boards(status("FREE", "FREE", "FREE", "FREE"), [2, 2])
    assert [p["board"] for p in plan] == ["B0", "B1"] and [p["exact"] for p in plan] == [0, 2]


def test_a_board_with_one_chip_not_free_is_not_free():
    with pytest.raises(bc.NoFreeBoard) as e:
        bc.plan_boards(status("FREE", "BUSY-UNTRACKED", "HELD", "HELD"), [2])
    assert "B0" in str(e.value) and "BUSY-UNTRACKED" in str(e.value)


def test_asking_for_more_boards_than_are_free_names_how_many_are():
    with pytest.raises(bc.NoFreeBoard, match="2 sessions.*1 free"):
        bc.plan_boards(status("HELD", "HELD", "FREE", "FREE"), [2, 2])


def test_a_session_that_needs_more_chips_than_one_board_has_is_refused():
    with pytest.raises(bc.NoFreeBoard, match="more chips"):
        bc.plan_boards(status("FREE", "FREE", "FREE", "FREE"), [4])


def test_an_excluded_board_is_skipped():
    plan = bc.plan_boards(status("FREE", "FREE", "FREE", "FREE"), [2], exclude=("B0",))
    assert plan[0]["board"] == "B1"


def sized(*boards):
    """Boards of differing size: each entry is (board name, [chip states])."""
    chips, dev = [], 0
    for name, states in boards:
        for s in states:
            chips.append(chip(dev, name, s))
            dev += 1
    return {"grain": "board", "chips": chips}


def test_a_two_chip_session_is_not_given_a_free_one_chip_board_when_a_two_chip_board_is_free():
    plan = bc.plan_boards(sized(("S", ["FREE"]), ("D", ["FREE", "FREE"])), [2])
    assert plan == [{"board": "D", "chips": [1, 2], "exact": 1}]


def test_the_larger_request_gets_the_larger_board_and_the_plans_keep_the_order_of_the_requests():
    st = sized(("S", ["FREE"]), ("D", ["FREE", "FREE"]))
    assert [p["board"] for p in bc.plan_boards(st, [1, 2])] == ["S", "D"]
    assert [p["board"] for p in bc.plan_boards(st, [2, 1])] == ["D", "S"]


def test_a_one_chip_session_takes_the_smallest_board_that_holds_it():
    plan = bc.plan_boards(sized(("D", ["FREE", "FREE"]), ("S", ["FREE"])), [1])
    assert plan[0]["board"] == "S"


def test_no_free_board_large_enough_is_refused_with_the_sizes_that_are_free():
    with pytest.raises(bc.NoFreeBoard, match="at least 2 chips.*S"):
        bc.plan_boards(sized(("S", ["FREE"]), ("D", ["HELD", "HELD"])), [2])


def test_two_requests_that_need_the_same_large_board_are_refused_even_when_the_count_of_free_boards_is_enough():
    with pytest.raises(bc.NoFreeBoard, match="at least 2 chips"):
        bc.plan_boards(sized(("S", ["FREE"]), ("D", ["FREE", "FREE"])), [2, 2])


def test_the_gozer_command_carries_the_chip_count_the_exact_unit_and_the_label():
    argv = bc.gozer_run_args({"board": "B1", "chips": [2, 3], "exact": 2}, who="claude:bench", reason="control run")
    assert argv == ["gozer", "run", "--chips", "2", "--exact", "2", "--who", "claude:bench", "--reason", "control run", "--"]


def test_the_chip_count_asked_for_is_the_one_in_the_command_even_when_the_board_has_more():
    argv = bc.gozer_run_args({"board": "B1", "chips": [2, 3], "exact": 2}, who="w", reason="r", chips=1)
    assert argv[2:6] == ["--chips", "1", "--exact", "2"]


def test_a_status_with_no_chips_is_an_error():
    with pytest.raises(bc.NoFreeBoard, match="no chips"):
        bc.plan_boards({"grain": "board", "chips": []}, [2])


class FakeGozer:
    def __init__(self, payloads):
        self.payloads, self.calls = list(payloads), 0

    def __call__(self, argv, **kw):
        self.calls += 1
        assert argv[:3] == ["gozer", "status", "--json"]
        p = self.payloads.pop(0) if len(self.payloads) > 1 else self.payloads[0]
        return type("R", (), {"returncode": 0, "stdout": json.dumps(p), "stderr": ""})()


def test_waiting_for_a_board_polls_until_one_is_free():
    busy, free = status("BUSY-UNTRACKED", "BUSY-UNTRACKED", "HELD", "HELD"), status("FREE", "FREE", "HELD", "HELD")
    g, slept = FakeGozer([busy, busy, free]), []
    clock = iter(range(0, 1000, 10))
    plan = bc.choose_boards([2], gozer="gozer", run=g, wait_s=120, sleep=slept.append, now=lambda: next(clock))
    assert plan[0]["board"] == "B0" and g.calls == 3 and len(slept) == 2


def test_waiting_gives_up_after_the_time_given_and_says_what_it_saw():
    busy = status("BUSY-UNTRACKED", "BUSY-UNTRACKED", "HELD", "HELD")
    clock = iter(range(0, 1000, 50))
    with pytest.raises(bc.NoFreeBoard, match="BUSY-UNTRACKED"):
        bc.choose_boards([2], gozer="gozer", run=FakeGozer([busy]), wait_s=100, sleep=lambda s: None, now=lambda: next(clock))


def test_no_waiting_asks_gozer_once():
    g = FakeGozer([status("HELD", "HELD", "HELD", "HELD")])
    with pytest.raises(bc.NoFreeBoard):
        bc.choose_boards([2], gozer="gozer", run=g, wait_s=0, sleep=lambda s: None, now=lambda: 0)
    assert g.calls == 1


def test_a_gozer_that_fails_or_prints_no_json_is_reported_as_that():
    def fail(argv, **kw):
        return type("R", (), {"returncode": 3, "stdout": "", "stderr": "boom"})()

    def junk(argv, **kw):
        return type("R", (), {"returncode": 0, "stdout": "not json", "stderr": ""})()
    with pytest.raises(bc.NoFreeBoard, match="exit 3"):
        bc.choose_boards([2], gozer="gozer", run=fail, wait_s=0)
    with pytest.raises(bc.NoFreeBoard, match="no JSON"):
        bc.choose_boards([2], gozer="gozer", run=junk, wait_s=0)


# ---- the command ---------------------------------------------------------------------------------------------------------------------------

def test_the_build_command_writes_the_control_and_prints_the_record(tmp_path, capsys):
    model_sh = tmp_path / "run.sh"
    model_sh.write_text(RUN_SH)
    out = tmp_path / "base-run.sh"
    rc = bc.main(["build", "--model-run-sh", str(model_sh), "--model-dir", MODEL_DIR, "--base-snapshot", BASE, "--base-id", "Qwen/Qwen3.8-27B@1d4bf0f",
                  "--out", str(out)])
    assert rc == 0 and out.exists()
    assert json.loads(capsys.readouterr().out)["swapped_lines"] == [4, 8]


def test_the_build_command_exits_two_and_says_why_on_a_refusal(tmp_path, capsys):
    model_sh = tmp_path / "run.sh"
    model_sh.write_text(RUN_SH)
    rc = bc.main(["build", "--model-run-sh", str(model_sh), "--model-dir", "/nowhere", "--base-snapshot", BASE, "--base-id", "x", "--out", str(tmp_path / "o.sh")])
    assert rc == 2 and "does not contain" in capsys.readouterr().err


def test_the_board_command_prints_one_gozer_command_per_session(capsys):
    g = FakeGozer([status("FREE", "FREE", "FREE", "FREE")])
    rc = bc.main(["board", "--chips", "2", "--chips", "2", "--who", "claude:b", "--reason", "r"], run=g)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and [p["exact"] for p in out["plans"]] == [0, 2]
    assert out["gozer"][0][:6] == ["gozer", "run", "--chips", "2", "--exact", "0"]


def test_the_board_command_prints_the_second_sessions_board_and_chip_count_too(capsys):
    g = FakeGozer([status("FREE", "FREE", "FREE", "FREE")])
    bc.main(["board", "--chips", "2", "--chips", "1"], run=g)
    out = json.loads(capsys.readouterr().out)
    assert len(out["gozer"]) == 2 and out["gozer"][1][2:6] == ["--chips", "1", "--exact", "2"]


def test_the_board_command_exits_two_when_no_board_is_free(capsys):
    rc = bc.main(["board", "--chips", "2"], run=FakeGozer([status("HELD", "HELD", "HELD", "HELD")]))
    assert rc == 2 and "no free board" in capsys.readouterr().err


# ---- the session summary ------------------------------------------------------------------------------------------------------------------

def test_the_session_parser_takes_a_control_record():
    a = bs.build_parser().parse_args(["--label", "x", "--out-dir", "/o", "--control-record", "/r/base-run.sh.control.json"])
    assert a.control_record == "/r/base-run.sh.control.json"


def test_a_session_summary_carries_the_serving_mode_and_the_control_record_when_given(tmp_path):
    run_sh = tmp_path / "base-run.sh"
    run_sh.write_text(build())
    rec = {"control_sha256": __import__("hashlib").sha256(run_sh.read_bytes()).hexdigest(), "built_from": "/m/run.sh", "swapped_lines": [4, 8]}
    block = bs.control_summary(run_sh, rec)
    assert block["serving"] == {"drafter": "off", "sampling": "host"}
    assert block["control"] == rec


def test_a_control_record_for_another_script_is_refused():
    with pytest.raises(ValueError, match="another script"):
        run_sh = Path(__file__)
        bs.control_summary(run_sh, {"control_sha256": "0" * 64})
