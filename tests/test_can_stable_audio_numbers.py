"""The Stable Audio 3 can's numbers (scripts/can_stable_audio_numbers.py): the raw session files become evidence files, and the card's table is computed from them."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec_ = importlib.util.spec_from_file_location("can_stable_audio_numbers_under_test", ROOT / "scripts" / "can_stable_audio_numbers.py")
nm = importlib.util.module_from_spec(spec_)
sys.modules["can_stable_audio_numbers_under_test"] = nm
spec_.loader.exec_module(nm)

SETTINGS = {"prompt": "p", "duration": 10, "steps": 8, "cfg_scale": 1.0, "seed": 42}


def job(total, seconds, ok=True):
    if not ok:
        return {"ok": False, "error": "boom"}
    return {"ok": True, "settings": SETTINGS, "server_total_s": total, "audio_s": float(seconds), "rtf": round(total / seconds, 4), "wall_s": total + 0.2,
            "stages": {"conditioning_s": 0.1, "chip_transformer_call_s": 0.3, "chip_decoder_call_s": 0.5, "model_load_s": 7.0}}


def write(raw, name, value):
    (raw / name).write_text(json.dumps(value))


def make_raw(tmp_path, *, bad_job=False, drop=None):
    raw = tmp_path / "raw"
    raw.mkdir()
    plans = {"s1": {10: [91.0, 12.0, 7.0, 17.0], 30: [24.0, 8.0, 7.0, 8.5], 60: [20.0, 4.3, 4.7, 4.7], 120: [34.0, 13.6, 6.8, 6.4]},
             "s3": {10: [38.0, 1.4, 1.1, 1.1], 30: [11.0, 2.0, 2.7], 60: [14.0, 3.2, 4.6], 120: [24.0, 20.0, 18.0]},
             "s2": {10: [2.6, 2.9, 2.4, 1.6, 1.6, 2.9], 120: [7.6, 6.9, 6.8]}}
    for label, lengths in plans.items():
        runs = []
        for sec, totals in lengths.items():
            jobs = [job(t, sec) for t in totals]
            if bad_job and label == "s1" and sec == 30:
                jobs[2] = job(0, sec, ok=False)
            write(raw, f"{label}-{sec}s.json", {"settings": SETTINGS, "jobs": jobs, "health": {"status": "ok"}})
            runs.append({"duration": sec, "repeat": len(totals), "rc": 0, "file": f"{label}-{sec}s.json"})
        write(raw, f"{label}-summary.json", {"label": label, "ready_s": 18.0, "host_load_at_start": 17.55, "host_load_at_end": 30.5, "runs": runs})
    write(raw, "ttmodel-10s.json", {"settings": SETTINGS, "jobs": [job(56.0, 10), job(11.0, 10), job(9.0, 10)]})
    write(raw, "ttmodel-120s.json", {"settings": SETTINGS, "jobs": [job(54.0, 120), job(19.0, 120)]})
    write(raw, "ttmodel-summary.json", {"install_s": 29.4, "ready_s": 17.0, "host_load_at_ready": 25.5, "jobs": ["ttmodel-10s.json", "ttmodel-120s.json"], "openapi_status": 200})
    write(raw, "recipe-10s.json", {"jobs": [job(79.0, 10), job(21.2, 10), job(22.7, 10)]})
    write(raw, "recipe-120s.json", {"jobs": [job(40.0, 120), job(28.7, 120)]})
    write(raw, "boot-check.json", {"ready_s": 21.0, "server_pid": 123, "audio": {"first": {"server_total_s": 41.4}, "second": {"server_total_s": 1.2}}, "passed": True})
    cpu_runs = lambda vals: {"runs": [{"generate_and_decode_s": v, "wall_s": v, "host_loadavg_1m": 30.0} for v in vals], "median_s": vals[0]}  # noqa: E731
    write(raw, "cpu-floor.json", {"threads": 8, "runs": {"10.0": cpu_runs([42.0, 16.0, 54.0]), "30.0": cpu_runs([30.0, 22.0, 33.0]), "60.0": cpu_runs([37.0, 29.0, 39.0]),
                                                           "120.0": cpu_runs([40.0, 39.0, 42.0])}})
    same = {"identical_samples": True, "max_abs_diff": 0.0, "pcc": 1.0, "rel_l2": 0.0}
    write(raw, "clip-comparison.json", {"10 s, bundle (warm, resident) against recipe package": same, "120 s, bundle against recipe package": same})
    write(raw, "clip-comparison-ttmodel.json", {"10 s, tt-model installer path against recipe package": same, "120 s, tt-model installer path against recipe package": same})
    write(raw, "clip-vs-cpu.json", {"10 s, chip (bundle) against the CPU float32 clip of the same bundle environment": {"pcc": 0.9676, "rel_l2": 0.2526, "identical_samples": False,
                                                                                                                           "max_abs_diff": 0.42}})
    if drop:
        (raw / drop).unlink()
    return raw


@pytest.fixture
def evidence(tmp_path):
    raw = make_raw(tmp_path)
    ev = tmp_path / "ev"
    nm.merge_raw(raw, ev)
    return ev


def section(ev):
    return nm.numbers_section(ev, date="2026-10-30", versions={"ttnn": "0.80.0", "torch": "2.7.1", "numpy": "1.26.4", "python": "3.12"})


def row(text, start):
    return next(line for line in text.splitlines() if line.startswith("| " + start))


def test_merge_writes_the_evidence_files_the_bundle_ships_and_drops_the_server_pid(tmp_path):
    made = nm.merge_raw(make_raw(tmp_path), tmp_path / "ev")
    spec = importlib.util.spec_from_file_location("base_for_names", ROOT / "scripts" / "can_stable_audio_base.py")
    base = importlib.util.module_from_spec(spec)
    sys.modules["base_for_names"] = base
    spec.loader.exec_module(base)
    assert sorted(made) == sorted(base.EVIDENCE_FILES)
    assert "server_pid" not in json.loads((tmp_path / "ev" / "boot-check.json").read_text())


def test_the_first_job_of_each_length_is_discarded_from_the_warm_statistics(evidence):
    text = section(evidence)
    line = row(text, "Generation time for a 10 s clip, session 1")
    assert "12.0 (7.00 to 17.0)" in line                    # 91.0 is the first job and is not in the median
    assert "benchmark (3 measured runs after 1 discarded run)" in line


def test_fewer_than_three_measured_runs_is_a_single_sample(evidence):
    line = row(section(evidence), "Generation time for a 30 s clip, session 3")
    assert "single sample (2 measured runs)" in line


def test_the_first_request_numbers_come_from_every_process(evidence):
    text = section(evidence)
    first = row(text, "First 10 s clip after start, empty kernel cache")
    assert "single samples from 4 processes" in first and "(38.0 to 91.0)" in first        # session 1, session 3, the boot check, tt-model
    assert "2.60" in row(text, "First 10 s clip after start, kernel cache on disk")


def test_one_value_is_written_once_and_a_spread_has_median_min_max():
    assert nm.spread([28.7]) == "28.7"
    assert nm.spread([3.0, 1.0, 2.0]) == "2.00 (1.00 to 3.00)"


def test_the_recipe_package_row_reports_the_model_load_inside_each_clip(evidence):
    line = row(section(evidence), "Recipe package (a new process for each clip): a 10 s clip")
    assert "21.9 (21.2 to 22.7)" in line and "the model load inside it takes 7.00 s" in line and "the first clip took 79.0 s" in line


def test_the_cpu_rows_state_the_threads_and_the_host_load(evidence):
    line = row(section(evidence), "CPU, float32, 8 threads: a 10 s clip")
    assert "42.0 (16.0 to 54.0)" in line and "host load 30.0 to 30.0" in line


def test_agreement_and_identity_rows(evidence):
    text = section(evidence)
    assert "| 0.9676 | 1 (the CPU clip) | 1 |" in row(text, "Waveform correlation with the float32 CPU clip")
    assert "| 0.2526 | 0 (the CPU clip) | 1 |" in row(text, "Waveform relative L2 error against the float32 CPU clip")
    identical = [line for line in text.splitlines() if line.startswith("| Clip samples identical")]
    assert len(identical) == 2 and all("| yes |" in line for line in identical) and "10 s and 120 s" in identical[0]


def test_a_clip_that_differs_from_the_recipe_packages_is_reported_as_no_with_the_size(tmp_path):
    raw = make_raw(tmp_path)
    cmp = json.loads((raw / "clip-comparison.json").read_text())
    cmp["10 s, bundle (warm, resident) against recipe package"] = {"identical_samples": False, "max_abs_diff": 0.0123, "pcc": 0.99, "rel_l2": 0.1}
    write(raw, "clip-comparison.json", cmp)
    nm.merge_raw(raw, tmp_path / "ev")
    line = row(section(tmp_path / "ev"), "Clip samples identical to the recipe package's (bundle")
    assert "| no (largest sample difference 0.0123) |" in line


def test_the_card_viewer_reads_at_most_30_rows_of_a_section_so_the_table_has_no_more(evidence):
    text = section(evidence)
    table = [line for line in text.split("Speed and identity:", 1)[1].splitlines() if line.startswith("| ") and not line.startswith(("| Number", "| ---"))]
    assert len(table) <= 30 - 2, len(table)        # the two accuracy rows are in the same section


def test_the_section_states_its_conditions_and_what_was_not_measured(evidence):
    text = section(evidence)
    assert "Measured on 2026-10-30" in text and "ttnn 0.80.0" in text and "numpy 1.26.4" in text and "17.6 to 30.5" in text
    assert "Not measured:" in text and "per sampler step" in text


def test_every_evidence_file_the_table_cites_is_one_the_bundle_ships(evidence):
    import re
    spec = importlib.util.spec_from_file_location("base_for_cites", ROOT / "scripts" / "can_stable_audio_base.py")
    base = importlib.util.module_from_spec(spec)
    sys.modules["base_for_cites"] = base
    spec.loader.exec_module(base)
    cited = set(re.findall(r"evidence/([A-Za-z0-9_.-]+\.json)", section(evidence)))
    assert cited and cited <= set(base.EVIDENCE_FILES)


def test_a_failed_job_in_the_evidence_is_an_error_and_not_a_number(tmp_path):
    raw = make_raw(tmp_path, bad_job=True)
    nm.merge_raw(raw, tmp_path / "ev")
    with pytest.raises(nm.EvidenceError, match="boom"):
        section(tmp_path / "ev")


def test_a_missing_raw_file_is_an_error_that_names_it(tmp_path):
    raw = make_raw(tmp_path, drop="cpu-floor.json")
    with pytest.raises(nm.EvidenceError, match="cpu-floor.json"):
        nm.merge_raw(raw, tmp_path / "ev")


def test_the_table_has_no_machine_path(evidence):
    text = section(evidence)
    for bad in ("/home/", "/mnt/", "ttuser"):
        assert bad not in text
