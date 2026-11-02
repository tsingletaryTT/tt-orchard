# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 269, the CPU command of a profile item and the per-step stage 2 commands: `audio_plan` builds them from `AudioPaths` fields for the ACE-Step
bundle, without opening a device. The supervisor does not run them yet (see the issue); these tests fix the command lines the supervisor will run."""
from pathlib import Path

import pytest

from orchard import audio_plan as ap
from orchard import audio_profile as prof
from orchard import audio_runtime as rt

ACE = "ACE-Step/Ace-Step1.5"


def paths(**kw):
    return rt.AudioPaths("/r/python", "/c/python", "/src", **kw)


FULL = dict(acestep_bundle_code="/b/code", acestep_snapshot="/s/snap", acestep_python="/a/python")


def test_paths_without_acestep_fields_keep_their_old_record():
    p = paths()
    assert p.record() == {"reference_python": "/r/python", "chip_python": "/c/python", "src_dir": "/src"}
    assert rt.AudioPaths.from_record(p.record()) == p


def test_paths_with_acestep_fields_round_trip_through_the_record():
    p = paths(**FULL)
    assert p.record()["acestep_bundle_code"] == "/b/code"
    assert rt.AudioPaths.from_record(p.record()) == p


def test_audio_paths_reads_the_acestep_variables(tmp_path):
    env = {"ORCHARD_ACESTEP_BUNDLE_CODE": "/b/code", "ORCHARD_ACESTEP_SNAPSHOT": "/s/snap", "ORCHARD_ACESTEP_PYTHON": "/a/python"}
    p = rt.audio_paths(tmp_path, env)
    assert (p.acestep_bundle_code, p.acestep_snapshot, p.acestep_python) == ("/b/code", "/s/snap", "/a/python")
    assert rt.audio_paths(tmp_path, {}).acestep_bundle_code == ""


def test_a_profile_item_gets_the_acestep_cpu_command():
    item = ap.reference_plan(ACE, None)[0]
    argv, env = ap.reference_command(item, paths(**FULL), Path("stages/1"), Path("make_reference.py"))
    assert argv[0] == "/a/python" and Path(argv[1]).name == "acestep_run.py" and Path(argv[1]).is_file()
    flag = lambda f: argv[argv.index(f) + 1]
    assert flag("--device") == "cpu" and flag("--bundle-code") == "/b/code" and flag("--snapshot") == "/s/snap"
    assert flag("--out") == str(Path("stages/1") / item["dir"])
    assert flag("--root") == str(Path("stages/1") / "acestep-root") and flag("--root") != flag("--out")
    s = ap.settings_for(item["name"])
    assert flag("--prompt") == s["prompt"] and flag("--duration") == str(s["duration"]) and flag("--seed") == str(s["seed"])
    assert "--cpu-dtype" not in argv or flag("--cpu-dtype") == "float32"      # the reference is the float32 run
    assert env == {}


def test_the_acestep_python_falls_back_to_the_reference_python():
    item = ap.reference_plan(ACE, None)[0]
    argv, _ = ap.reference_command(item, paths(acestep_bundle_code="/b", acestep_snapshot="/s"), Path("s"), Path("x.py"))
    assert argv[0] == "/r/python"


@pytest.mark.parametrize("missing", ["acestep_bundle_code", "acestep_snapshot"])
def test_a_profile_item_without_its_bundle_or_snapshot_is_refused_with_the_variable_to_set(missing):
    kw = {**FULL, missing: ""}
    item = ap.reference_plan(ACE, None)[0]
    with pytest.raises(ap.AudioStageError, match="ORCHARD_ACESTEP_"):
        ap.reference_command(item, paths(**kw), Path("s"), Path("x.py"))


def test_a_stable_audio_item_is_unchanged():
    item = ap.reference_plan("stabilityai/stable-audio-3-small-music", None)[0]
    argv, _ = ap.reference_command(item, paths(), Path("stages/1"), Path("make_reference.py"))
    assert argv[:2] == ["/r/python", "make_reference.py"] and "--device" not in argv


# ---- per-step stage 2 ----
def test_per_step_commands_are_the_floor_the_chip_check_and_the_bars_in_that_order(tmp_path):
    item = ap.chip_plan(ACE, None)[0]
    cmds = ap.per_step_commands(item, paths(**FULL), tmp_path / "stage", tmp_path / "ref-run")
    assert [c["name"] for c in cmds] == ["floor", "chip", "bars"]
    floor, chip, bars = (c["argv"] for c in cmds)
    assert "--teacher-forced" in floor and floor[floor.index("--teacher-forced") + 1] == str(tmp_path / "ref-run")
    assert floor[floor.index("--cpu-dtype") + 1] == "bfloat16" and floor[floor.index("--device") + 1] == "cpu"
    assert "--stepcheck" in chip and chip[chip.index("--device") + 1] == "chip" and chip[chip.index("--stepcheck") + 1] == str(tmp_path / "ref-run")
    assert bars[:3] == ["/a/python", "-m", "orchard.step_bars"] or bars[1:3] == ["-m", "orchard.step_bars"]
    assert bars[bars.index("--run") + 1] == str(tmp_path / "ref-run")
    assert bars[bars.index("--floor") + 1].endswith(".teacher.json") and bars[bars.index("--chip") + 1].endswith(".stepcheck.json")
    assert [c["device"] for c in cmds] == ["cpu", "chip", "cpu"]            # only the middle command needs a board


def test_per_step_outputs_do_not_collide_and_stay_inside_the_stage(tmp_path):
    item = ap.chip_plan(ACE, None)[0]
    stage = tmp_path / "stage"
    cmds = ap.per_step_commands(item, paths(**FULL), stage, tmp_path / "ref")
    outs = [c["argv"][c["argv"].index("--out") + 1] for c in cmds if "--out" in c["argv"]]
    assert len(set(outs)) == len(outs) and all(Path(o).is_relative_to(stage) for o in outs)


def test_a_model_whose_stage2_is_end_to_end_has_no_per_step_commands(tmp_path):
    with pytest.raises(ap.AudioStageError, match="per_step"):
        ap.per_step_commands({"role": ap.ROLE_CHIP, "name": "small-music"}, paths(), tmp_path, tmp_path)


def test_stage2_kind_follows_the_profile():
    assert ap.stage2_kind({"name": "small-music"}) == prof.STAGE2_END_TO_END
    assert ap.stage2_kind(ap.chip_plan(ACE, None)[0]) == prof.STAGE2_PER_STEP


def test_run_references_gives_the_cpu_command_of_a_profile_item_absolute_out_and_root_paths(tmp_path):
    """The runner is not started in the run directory, so --out and --root must be absolute and inside the stage. The fake runner stops the stage after
    it has seen the command (a model run is not needed here)."""
    from orchard import audio_stages as st
    run_dir = tmp_path / "run"
    stage = run_dir / "stages" / "1"
    stage.mkdir(parents=True)
    seen = {}

    def runner(argv, env, timeout_s, log):
        seen["argv"] = argv
        return 1                                   # a failed run: the stage raises after the command was captured

    plan = ap.reference_plan(ACE, None)
    with pytest.raises(ap.AudioStageError):
        st.run_references(stage, run_dir, paths(**FULL), plan, runner, expected={}, library_commit=None)
    argv = seen["argv"]
    out, root = argv[argv.index("--out") + 1], argv[argv.index("--root") + 1]
    assert Path(out).is_absolute() and Path(root).is_absolute()
    assert Path(out).is_relative_to(stage) and Path(root).is_relative_to(stage)
