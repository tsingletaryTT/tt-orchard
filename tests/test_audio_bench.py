# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The sweep script (orchard/skills/audio-chip-templates/audio_bench.py): one loaded runtime, the stated sizes, the warm-up rule, a JSON file."""
import importlib.util
import json
from pathlib import Path

import pytest

from orchard import resident

spec = importlib.util.spec_from_file_location("audio_bench_under_test", Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates" / "audio_bench.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)

ARGS = ["--device", "chip", "--model", "small-music", "--prompt", "p", "--steps", "8", "--cfg-scale", "1.0", "--seed", "42"]


def run(tmp_path, extra=(), rt=None):
    rt = rt or resident.FakeRuntime()
    code = ab.main(ARGS + ["--out", str(tmp_path / "bench.json")] + list(extra), runtime_factory=lambda args: rt)
    return code, rt


def test_the_default_sweep_runs_ten_thirty_sixty_and_one_twenty_seconds_with_the_rule(tmp_path):
    code, rt = run(tmp_path)
    assert code == 0 and rt.loads == 1 and rt.closed is True
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert [row["seconds"] for row in rec["sizes"]] == [10.0, 30.0, 60.0, 120.0]
    assert rec["warmup_runs"] == 1 and rec["repeats"] == 3 and all(row["label"] == "benchmark" for row in rec["sizes"])
    assert rt.generate_calls == 4 * (1 + 3) and rec["load"]["model_load_s"] == rt.load_timings["model_load_s"]
    assert all(len(row["runs"]) == 3 and "host_loadavg_1m" in row["runs"][0] for row in rec["sizes"])
    assert "host_loadavg_1m_at_start" in rec and rec["device"] == "chip"


def test_sizes_warmup_and_repeats_can_be_stated(tmp_path):
    code, rt = run(tmp_path, ["--sizes", "10,120", "--warmup", "0", "--repeats", "1"])
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert [row["seconds"] for row in rec["sizes"]] == [10.0, 120.0] and rt.generate_calls == 2
    assert {row["label"] for row in rec["sizes"]} == {"single sample"} and rec["warmup_runs"] == 0 and rec["repeats"] == 1


def test_three_repeats_with_no_warmup_are_labelled_unwarmed(tmp_path):
    run(tmp_path, ["--sizes", "10", "--warmup", "0", "--repeats", "3"])
    assert json.loads((tmp_path / "bench.json").read_text())["sizes"][0]["label"] == "unwarmed"


@pytest.mark.parametrize("sizes", ["10,121", "0", "-3", "x", "nan"])
def test_a_bad_size_is_refused_before_anything_loads(tmp_path, sizes):
    rt = resident.FakeRuntime()
    with pytest.raises(SystemExit):
        run(tmp_path, ["--sizes", sizes], rt)
    assert rt.loads == 0 and not (tmp_path / "bench.json").exists()


def test_an_existing_result_file_is_never_overwritten(tmp_path):
    (tmp_path / "bench.json").write_text("{}")
    rt = resident.FakeRuntime()
    with pytest.raises(SystemExit, match="exists"):
        run(tmp_path, [], rt)
    assert rt.loads == 0 and (tmp_path / "bench.json").read_text() == "{}"


def test_the_runtime_is_closed_when_the_sweep_fails(tmp_path):
    rt = resident.FakeRuntime(times=[1.0, 2.0, 3.0, 4.0], drop_part_on_call=3)
    with pytest.raises(resident.ResidentError):
        run(tmp_path, ["--sizes", "10"], rt)
    assert rt.closed is True and not (tmp_path / "bench.json").exists()


def test_the_record_keeps_the_settings_but_not_the_duration_which_each_size_states(tmp_path):
    run(tmp_path)
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert rec["settings"]["seed"] == 42 and rec["settings"]["steps"] == 8 and "duration" not in rec["settings"]
