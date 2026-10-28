# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 8 bundle of an audio run: the "Audio clip on a chip" section, the clips and the audio risks.

The builder (orchard/skills/operator-bundle-templates/build_bundle.py) reads `stages/2/audio_result.json` only when it exists.
These tests build the fake run of test_operator_bundle_template.py, add an audio_result.json written the way stage 2 writes it
(orchard/audio_stages.py finish_chip) and stand-in FLAC files (the builder copies bytes and never decodes them), and run the
builder as a subprocess. The real stage 8 gate and the real bundle scrub check the result. A run with no audio_result.json
must give the same RESULTS.md, RISKS.md and file list as before.
"""
import json
import pytest
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_template import (GOLDEN, build, bundle_of, make_run, section, write, write_json)  # noqa: E402

from orchard.scrub import scrub_bundle  # noqa: E402
from orchard.stages import gate_bundle  # noqa: E402
from orchard.supervisor import Supervisor  # noqa: E402

HEADING = "## Audio clip on a chip"
RESULT = Path("stages") / "2" / "audio_result.json"
SWEEP = [", not ", "rather than", "instead of", "reads as", "honest", "the one", "which is why", "This is what"]


def summary(ok=True, *, pcc=0.9991, rel=0.0412, failed=(), problems=(), comparable=True):
    return {"ok": ok, "comparable": comparable, "problems": list(problems), "arrays": 31, "failed": list(failed),
            "latent_final": {"pcc": pcc, "rel_l2": rel, "slope": 0.99},
            "waveform": {"pcc": 0.9876, "rel_l2": 0.1555, "slope": 0.98, "log_spectrum_distance": 0.2718},
            "worst_layer": {"name": "layer_17", "rel_l2": 0.0733, "pcc": 0.9973}, "bars": "latent pcc >= 0.99"}


def run_entry(name, *, clip, timings=None):
    return {"dir": f"stages/2/chip/{name}", "model": name, "revision": "abc123", "library_commit": "def4567",
            "settings": {}, "timings": timings, "manifest_sha256": "0" * 64, "clip": clip, "device": "p150a", "total_seconds": 12.5}


FULL_TIMINGS = {"generate_and_decode_s": 41.25, "conditioning_s": 2.5, "model_load_s": 8.125, "chip_weights_upload_s": 3.75,
                "chip_transformer_call_s": 19.5, "chip_passes": 200}


def audio_result(**over):
    res = {
        "schema": 1, "runtime": "stable-audio-3-dit-on-chip",
        "settings": {"prompt": "a calm piano melody", "duration": 11, "steps": 8, "cfg_scale": 1.0, "sampler": "euler", "seed": 42,
                     "sample_size": 524288},
        "runs": {"chip": run_entry("small-music", clip="stages/2/chip/small-music/clips/audio.flac", timings=dict(FULL_TIMINGS)),
                 "control_chip": run_entry("small-base", clip="stages/2/chip/small-base/clips/audio.flac")},
        "reference_runs": {
            "reference": {**run_entry("small-music", clip="stages/1/runs/reference/clips/audio.flac"), "from_cache": True,
                          "wall_seconds": 0.01, "interpreter": "venv/bin/python",
                          "timings": {"generate_and_decode_s": 24.8}},
            "host_same_env": {**run_entry("small-music", clip="stages/1/runs/host/clips/audio.flac"), "wall_seconds": 280.0},
            "control_reference": {**run_entry("small-base", clip="stages/1/runs/control/clips/audio.flac"), "wall_seconds": 270.0}},
        "agreement": {"chip_vs_reference": summary(), "control_chip_vs_control_reference": summary(rel=0.05),
                      "chip_vs_host_same_env": summary(rel=0.0411)},
        "environment_floor": summary(pcc=0.9999, rel=0.0098),
        "control_difference": {"ok": True, "rel_l2": 0.83, "pcc": 0.31, "problems": []},
        "control_discrimination": {"ok": True, "own_rel_l2": 0.3, "other_rel_l2": 1.9},
        "limits": ["Nobody has listened to the clips; no check here says the music is good.",
                   "The agreement bars are provisional choices, not yet set from a measured noise floor."]}
    res.update(over)
    if "repeats" in over and "timing" not in over:           # a result with repeats also states the warm-up the stage used
        res["timing"] = {"warmup_runs": 1, "min_repeats": 3, "roles": {}}
    if res.get("timing", 0) is None:
        del res["timing"]
    return res


CLIPS = {"stages/2/chip/small-music/clips/audio.flac": b"fLaC-chip",
         "stages/2/chip/small-base/clips/audio.flac": b"fLaC-control-chip",
         "stages/1/runs/reference/clips/audio.flac": b"fLaC-reference",
         "stages/1/runs/control/clips/audio.flac": b"fLaC-control-reference"}


def audio_run(tmp_path, result=None, *, clips=CLIPS):
    """The fake run plus audio_result.json and the clip files."""
    run = make_run(tmp_path)
    for rel, data in clips.items():
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    write_json(run / RESULT, audio_result() if result is None else result)
    return run


def built(run):
    r = build(run)
    assert r.returncode == 0, r.stdout + r.stderr
    b = bundle_of(run)
    return (b / "RESULTS.md").read_text(), (b / "RISKS.md").read_text(), r.stdout


def block(text):
    return section(text, HEADING)


def files_of(run):
    return sorted(p.relative_to(bundle_of(run)).as_posix() for p in bundle_of(run).rglob("*") if p.is_file())


# ---- a run with no audio_result.json is unchanged ------------------------------------------------------

def test_a_run_without_audio_result_gets_the_same_results_risks_and_files(tmp_path):
    run = make_run(tmp_path)
    results, risks, out = built(run)
    assert results == GOLDEN.read_text()
    assert HEADING not in results and "Audio run" not in risks and "audio" not in " ".join(files_of(run))
    assert "audio_clips" not in json.loads((run / "stages/8/evidence/bundle-build.json").read_text())
    # and the audio run differs from it only by the audio parts
    other = tmp_path / "other"
    other.mkdir()
    arun = audio_run(other)
    aresults, arisks, _ = built(arun)
    cut = aresults[aresults.index("\n" + HEADING):aresults.index("\n## Stage 0")]
    assert "Audio clip on a chip" in cut and "stages/1/audio_reference.json" in aresults    # the audio run points at its own stage 1 record
    assert "stages/1/reference.json" not in aresults        # and cites no text-model file it does not have
    assert "stages/4/result.json" not in arisks            # an audio run skips stage 4, so it owes no stage 4 numbers


# ---- the section ----------------------------------------------------------------------------------------

def test_the_section_shows_the_model_settings_and_what_ran_where(tmp_path):
    results, _, _ = built(audio_run(tmp_path))
    body = block(results)
    for text in ("| Model | small-music |", "| Revision | abc123 |", "a calm piano melody", "| Seed | 42 |", "| Steps | 8 |",
                 "| Guidance scale | 1 |", "| Sampler | euler |", "| Duration (seconds) | 11 |"):
        assert text in body, text
    flat = " ".join(body.split())
    assert "diffusion transformer ran on one chip" in flat
    assert "T5Gemma text encoder, the sampler and the autoencoder decoder ran on the host" in flat


def test_every_timing_is_printed_with_its_plain_sentence(tmp_path):
    body = block(built(audio_run(tmp_path))[0])
    for key, value in (("generate_and_decode_s", "41.25 seconds"), ("conditioning_s", "2.5 seconds"),
                       ("model_load_s", "8.125 seconds"), ("chip_weights_upload_s", "3.75 seconds"),
                       ("chip_transformer_call_s", "19.5 seconds"), ("chip_passes", "200 passes"), ("total_seconds", "12.5 seconds")):
        row = next(line for line in body.splitlines() if line.startswith(f"| `{key}` |"))
        assert value in row, row
        assert row.rstrip().endswith("|") and len(row.split("|")[3].strip()) > 10          # the explanation is there
    assert "not recorded" not in "\n".join(line for line in body.splitlines() if line.startswith("| `"))


def test_a_missing_timing_key_prints_not_recorded(tmp_path):
    res = audio_result()
    del res["runs"]["chip"]["timings"]["conditioning_s"]
    del res["runs"]["chip"]["timings"]["chip_passes"]
    body = block(built(audio_run(tmp_path, res))[0])
    assert "| `conditioning_s` | not recorded |" in body and "| `chip_passes` | not recorded |" in body
    assert "| `model_load_s` | 8.125 seconds |" in body


def test_no_timings_dict_at_all_prints_not_recorded_for_every_key(tmp_path):
    res = audio_result()
    res["runs"]["chip"]["timings"] = None
    body = block(built(audio_run(tmp_path, res))[0])
    for key in ("generate_and_decode_s", "conditioning_s", "model_load_s", "chip_weights_upload_s", "chip_transformer_call_s",
                "chip_passes"):
        assert f"| `{key}` | not recorded |" in body


def test_the_agreement_table_has_the_three_comparisons_with_their_numbers(tmp_path):
    body = block(built(audio_run(tmp_path))[0])
    table = section(body, "### Agreement with the CPU reference")
    rows = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]] for line in table.splitlines()
            if line.startswith("| ") and not line.startswith("| Comparison")}
    assert len(rows) == 3
    chip = rows["Chip against the CPU reference"]
    assert chip[0] == "within the bars" and chip[1] == "0.9991" and chip[2] == "0.0412"
    assert chip[3] == "0.9876" and chip[4] == "0.1555" and chip[5] == "0.2718"
    assert "layer_17" in chip[6] and "0.0733" in chip[6] and chip[7] == "none"
    assert rows["Chip against the same-environment CPU run"][2] == "0.0411"
    floor = rows["Environment floor: the same-environment CPU run against the reference"]
    assert floor[2] == "0.0098"
    flat = " ".join(table.split())
    assert "provisional" in flat and "amount two CPU environments already differ" in flat


def test_a_failed_array_list_and_problems_are_shown(tmp_path):
    res = audio_result()
    res["agreement"]["chip_vs_reference"] = summary(False, failed=["latent/final", "layer/layer_03"], problems=["layer_03 is off"])
    body = block(built(audio_run(tmp_path, res))[0])
    assert "outside the bars" in body and "latent/final, layer/layer_03" in body and "layer_03 is off" in body


def test_the_control_section_says_what_the_control_shows(tmp_path):
    body = section(block(built(audio_run(tmp_path))[0]), "### Control: the base model")
    flat = " ".join(body.split())
    assert "catches a chip run that served the wrong weights" in flat
    assert "differs from the fine-tune's clip (relative L2 0.83, PCC 0.31)" in flat
    assert "| 0.05 |" in body                                    # the control's own agreement row


def test_a_control_that_matches_the_fine_tune_is_said_plainly(tmp_path):
    res = audio_result(control_difference={"ok": False, "rel_l2": 0.0, "pcc": 1.0, "problems": []})
    results, risks, _ = built(audio_run(tmp_path, res))
    assert "does not differ from the fine-tune's clip" in block(results)
    assert "does not differ from the fine-tune's" in section(risks, "## Audio run")


def test_a_run_without_a_control_says_so_in_results_and_risks(tmp_path):
    res = audio_result()
    del res["runs"]["control_chip"]
    del res["reference_runs"]["control_reference"]
    results, risks, _ = built(audio_run(tmp_path, res))
    assert "The control did not run" in block(results)
    assert "The control did not run" in risks
    assert "control-chip.flac" not in results


def test_the_limits_are_copied_verbatim(tmp_path):
    body = block(built(audio_run(tmp_path))[0])
    for line in audio_result()["limits"]:
        assert f"- {line}" in body


def test_the_clips_are_linked_by_relative_path(tmp_path):
    body = block(built(audio_run(tmp_path))[0])
    for name in ("chip.flac", "reference.flac", "control-chip.flac", "control-reference.flac"):
        assert f"(audio/{name})" in body, name


# ---- the clips and the gate ----------------------------------------------------------------------------

def test_the_clips_are_copied_recorded_and_accepted_by_the_gate_and_the_scrub(tmp_path):
    run = audio_run(tmp_path)
    _, _, out = built(run)
    b = bundle_of(run)
    for name, data in (("chip.flac", b"fLaC-chip"), ("reference.flac", b"fLaC-reference"),
                       ("control-chip.flac", b"fLaC-control-chip"), ("control-reference.flac", b"fLaC-control-reference")):
        assert (b / "audio" / name).read_bytes() == data
    record = json.loads((run / "stages/8/evidence/bundle-build.json").read_text())
    recorded = {f["path"] for f in record["files"]}
    assert {f"stages/8/bundle/audio/{n}" for n in ("chip.flac", "reference.flac", "control-chip.flac",
                                                   "control-reference.flac")} <= recorded
    shutil.copyfile(run / "ledger.jsonl", b / "ledger.jsonl")
    Supervisor._copy_package(SimpleNamespace(run_dir=run), b)
    gate = gate_bundle(run / "stages" / "8", run)
    assert gate.ok, gate.reasons
    assert scrub_bundle(b) == [] and "scrub: no findings" in out


def test_a_clip_over_the_limit_is_not_copied_and_the_results_say_so(tmp_path):
    run = audio_run(tmp_path)
    (run / "stages/2/chip/small-music/clips/audio.flac").write_bytes(b"x" * 20_000_001)
    results, risks, out = built(run)
    b = bundle_of(run)
    assert not (b / "audio" / "chip.flac").exists() and (b / "audio" / "reference.flac").exists()
    body = block(results)
    assert "not copied into this bundle, because it is 20000001 bytes, over the 20 MB limit" in body
    assert "(audio/chip.flac)" not in body
    assert "is not in this bundle" in risks and "left out audio/chip.flac" in out


def test_a_clip_of_exactly_the_limit_is_copied(tmp_path):
    run = audio_run(tmp_path)
    (run / "stages/2/chip/small-music/clips/audio.flac").write_bytes(b"x" * 20_000_000)
    built(run)
    assert (bundle_of(run) / "audio" / "chip.flac").stat().st_size == 20_000_000


def test_a_clip_path_outside_the_run_or_not_flac_is_not_copied(tmp_path):
    res = audio_result()
    res["runs"]["chip"]["clip"] = "../outside.flac"
    res["reference_runs"]["reference"]["clip"] = "/etc/hostname"
    res["runs"]["control_chip"]["clip"] = "stages/2/chip/small-base/clips/notes.txt"
    run = audio_run(tmp_path, res)
    (tmp_path / "outside.flac").write_bytes(b"fLaC-secret")
    (run / "stages/2/chip/small-base/clips/notes.txt").write_bytes(b"hello")
    results, _, _ = built(run)
    assert files_of(run).count("audio/chip.flac") == 0 and "audio/reference.flac" not in files_of(run)
    assert "audio/control-chip.flac" not in files_of(run)
    assert "/etc/hostname" not in results and "outside.flac" not in results


def test_a_symlinked_clip_is_not_followed(tmp_path):
    run = audio_run(tmp_path)
    clip = run / "stages/2/chip/small-music/clips/audio.flac"
    clip.unlink()
    clip.symlink_to(tmp_path / "elsewhere.flac")
    (tmp_path / "elsewhere.flac").write_bytes(b"fLaC-other")
    built(run)
    assert "audio/chip.flac" not in files_of(run)


# ---- an untrusted file never crashes the builder --------------------------------------------------------

def test_a_list_instead_of_an_object_says_so_and_does_not_crash(tmp_path):
    run = audio_run(tmp_path, [1, 2, 3])
    results, risks, _ = built(run)
    assert "is present but it is not a readable JSON object" in block(results)
    assert "is present but it is not a readable JSON object" in section(risks, "## Audio run")
    assert not (bundle_of(run) / "audio").exists()


def test_a_file_that_is_not_json_says_so(tmp_path):
    run = audio_run(tmp_path)
    write(run / RESULT, "{not json")
    assert "not a readable JSON object" in block(built(run)[0])


def test_strings_where_numbers_belong_print_not_recorded(tmp_path):
    res = audio_result()
    res["runs"]["chip"]["timings"] = {k: "fast" for k in FULL_TIMINGS}
    res["runs"]["chip"]["total_seconds"] = "12"
    res["agreement"]["chip_vs_reference"] = summary(pcc="high", rel=[1])
    res["agreement"]["chip_vs_reference"]["failed"] = "all"
    res["settings"]["seed"] = {"a": 1}
    results, _, _ = built(audio_run(tmp_path, res))
    body = block(results)
    for key in FULL_TIMINGS:
        assert f"| `{key}` | not recorded |" in body
    assert "| `total_seconds` | not recorded |" in body and "| Seed | not recorded |" in body
    row = next(line for line in body.splitlines() if line.startswith("| Chip against the CPU reference"))
    assert row.count("not recorded") == 3 and "high" not in row and row.rstrip().endswith("| not recorded |")


def test_a_huge_or_infinite_number_is_not_recorded(tmp_path):
    text = json.dumps(audio_result()).replace('"conditioning_s": 2.5', '"conditioning_s": 1e999') \
        .replace('"model_load_s": 8.125', '"model_load_s": -1e999').replace('"chip_passes": 200', '"chip_passes": 1e300')
    run = audio_run(tmp_path)
    write(run / RESULT, text)
    body = block(built(run)[0])
    for key in ("conditioning_s", "model_load_s", "chip_passes"):
        assert f"| `{key}` | not recorded |" in body, key
    assert "inf" not in body.lower().replace("information", "") and "e+300" not in body


def test_an_integer_too_long_for_python_makes_the_file_unreadable_and_the_builder_says_so(tmp_path):
    text = json.dumps(audio_result()).replace('"chip_passes": 200', '"chip_passes": ' + "9" * 5000)
    run = audio_run(tmp_path)
    write(run / RESULT, text)
    results, _, _ = built(run)
    assert "is present but it is not a readable JSON object" in block(results)


def test_nested_fields_of_the_wrong_type_do_not_crash(tmp_path):
    res = audio_result(runs="x", reference_runs=[], agreement=5, environment_floor="x", control_difference=[], limits="text",
                       settings=None)
    results, risks, _ = built(audio_run(tmp_path, res))
    body = block(results)
    assert "| Model | not recorded |" in body and "| Prompt | not recorded |" in body
    assert "- not recorded" in body                                    # the limits list is not a list
    assert "The control did not run" in body
    assert "agreement with the CPU reference is not recorded" in section(risks, "## Audio run")


def test_text_with_paths_and_tokens_is_redacted_and_cannot_break_the_table(tmp_path):
    res = audio_result()
    res["settings"]["prompt"] = "piano | in /home/someone/x.wav\n# Injected hf_abcdefghijklmnopqrstuvwxyz0123456789"
    run = audio_run(tmp_path, res)
    res["runs"]["chip"]["device"] = "p150a on " + str(run)
    write_json(run / RESULT, res)
    results, _, _ = built(run)
    assert "/home/someone" not in results and "hf_abcdefghijklmnopqrstuvwxyz" not in results
    assert "\n# Injected" not in results and str(run) not in results and "<RUN_DIR>" in results
    assert scrub_bundle(bundle_of(run)) == []


# ---- RISKS.md ------------------------------------------------------------------------------------------

def test_the_risks_carry_the_four_fixed_lines(tmp_path):
    _, risks, _ = built(audio_run(tmp_path))
    body = section(risks, "## Audio run")
    for line in ("Audio quality is not judged: nobody has listened to the clips", "Bars are provisional",
                 "Only the diffusion transformer ran on the chip", "One prompt, one duration, one seed"):
        assert line in body, line
    assert "failed" not in body and "The control did not run" not in body


def test_a_failed_chip_agreement_is_the_first_risk(tmp_path):
    res = audio_result()
    res["agreement"]["chip_vs_reference"] = summary(False, failed=["latent/final"])
    _, risks, _ = built(audio_run(tmp_path, res))
    items = [x for x in section(risks, "## Audio run").splitlines() if x.startswith("- ")]
    assert items[0].startswith("- The chip run's agreement with the CPU reference failed") and "latent/final" in items[0]
    assert items[1].startswith("- Audio quality is not judged")
    assert risks.index("## Audio run") < risks.index("## TODO numbers")


def test_an_agreement_that_cannot_be_compared_counts_as_failed(tmp_path):
    res = audio_result()
    res["agreement"]["chip_vs_reference"] = summary(True, comparable=False)
    _, risks, _ = built(audio_run(tmp_path, res))
    assert "agreement with the CPU reference failed" in risks


# ---- wording -------------------------------------------------------------------------------------------

def test_the_new_text_has_none_of_the_phrases_the_style_rules_forbid(tmp_path):
    run = audio_run(tmp_path)
    res = audio_result()
    del res["runs"]["control_chip"]
    run2 = audio_run(tmp_path / "second", res)
    for r in (run, run2):
        results, risks, _ = built(r)
        # the limits are stage 2's own words, copied verbatim, so they are left out of the sweep
        text = block(results).split("### Limits recorded by stage 2")[0] + section(risks, "## Audio run")
        for phrase in SWEEP:
            assert phrase not in text, phrase
        assert not re.search(r"(/home/|/mnt/|/tmp/)", text)


def test_an_audio_run_needs_no_text_reference_file_and_a_text_run_still_does(tmp_path):
    arun = audio_run(tmp_path)
    ref = arun / "stages" / "1" / "reference.json"
    if ref.exists():
        ref.unlink()
    (arun / "stages" / "2" / "result.json").unlink(missing_ok=True)
    results, _, _ = built(arun)
    assert "Audio clip on a chip" in results
    text = tmp_path / "text"
    text.mkdir()
    trun = make_run(text)
    (trun / "stages" / "1" / "reference.json").unlink()
    with pytest.raises(Exception):
        built(trun)


def test_the_reference_time_is_the_references_own_and_not_the_time_to_copy_a_cached_one(tmp_path):
    flat = " ".join(block(built(audio_run(tmp_path))[0]).split())
    assert "took 24.8 seconds when it was made (this run reused a saved reference)" in flat and "0.01" not in flat


def test_the_control_discrimination_numbers_are_printed_and_a_failure_is_said_in_results_and_risks(tmp_path):
    flat = " ".join(section(block(built(audio_run(tmp_path))[0]), "### Control: the base model").split())
    assert "0.3 against the base model's own reference and 1.9 against the fine-tune's reference. It is nearer its own." in flat
    res = audio_result(control_discrimination={"ok": False, "own_rel_l2": 1.5, "other_rel_l2": 0.2})
    results, risks, _ = built(audio_run(tmp_path / "bad", res)) if (tmp_path / "bad").mkdir() is None else None
    assert "not nearer its own" in " ".join(block(results).split())
    assert "wrong weights" in section(risks, "## Audio run")


def test_a_missing_discrimination_record_is_not_recorded_and_raises_the_risk(tmp_path):
    res = audio_result()
    del res["control_discrimination"]
    (tmp_path / "m").mkdir()
    results, risks, _ = built(audio_run(tmp_path / "m", res))
    assert "nearer its own reference than the fine-tune's: not recorded" in " ".join(block(results).split())
    assert "wrong weights" in section(risks, "## Audio run")


def test_the_stage_2_entry_of_an_audio_run_points_at_the_audio_section(tmp_path):
    results = built(audio_run(tmp_path))[0]
    stage2 = results.split("## Stage 2:", 1)[1].split("## Stage 3", 1)[0]
    assert "Audio clip on a chip" in stage2 and "No result file" not in stage2


# ---- the timing repeats ---------------------------------------------------------------------------------------------------------------

BENCH = "### Timing repeats: benchmark"
SINGLE = "### Timing repeats: single sample"
UNWARMED = "### Timing repeats: unwarmed"


def repeats(chip=(5.0, 1.6, 1.5), control=(60.0, 30.0, 31.0), loads=(6.0, 7.0, 5.0)):
    def lst(vals, call):
        out = [{"timings": {"generate_and_decode_s": v, "chip_transformer_call_s": call * v, "conditioning_s": 0.04,
                            "host_loadavg_1m": loads[i % len(loads)]}} for i, v in enumerate(vals)]
        out[0]["timings"].update({"model_load_s": 6.125, "chip_weights_upload_s": 1.25})     # the single loaded process reports its load in repeat 0
        return out
    return {"chip": lst(chip, 0.3), "control_chip": lst(control, 0.9)}


def test_three_repeats_make_a_benchmark_table_with_min_median_max_and_the_one_cpu_sample(tmp_path):
    results = built(audio_run(tmp_path, audio_result(repeats=repeats())))[0]
    assert BENCH in results and SINGLE not in results
    body = section(block(results), BENCH)
    row = next(l for l in body.splitlines() if l.startswith("| Fine-tune, 8 steps | the whole `generate` call"))
    assert "5, 1.6, 1.5" in row and "| 1.5 | 1.6 | 5 |" in row and "24.8" in row
    base = next(l for l in body.splitlines() if l.startswith("| Base control, 50 steps, guidance 7 | the whole `generate` call"))
    assert "60, 30, 31" in base and "| 30 | 31 | 60 |" in base
    assert "one sample" in body and "Repeats per model: 3, 3" in body


def test_the_benchmark_section_states_the_warmup_the_load_once_and_the_host_load(tmp_path):
    res = audio_result(repeats=repeats(), timing={"warmup_runs": 1, "min_repeats": 3, "roles": {}})
    body = " ".join(section(block(built(audio_run(tmp_path, res))[0]), BENCH).split())
    assert "1 discarded warm-up run" in body
    assert "Load, once for the process" in body and "6.125" in body and "1.25" in body
    assert "Host load average" in body and "6, 7, 5" in body


def test_a_missing_warmup_record_makes_three_repeats_unwarmed_and_says_not_recorded(tmp_path):
    results = built(audio_run(tmp_path, audio_result(repeats=repeats(), timing=None)))[0]
    assert UNWARMED in results and BENCH not in results
    body = " ".join(section(block(results), UNWARMED).split())
    assert "warm-up runs before the repeats: not recorded" in body and "not a benchmark" in body


def test_a_recorded_warmup_of_zero_makes_three_repeats_unwarmed(tmp_path):
    results = built(audio_run(tmp_path, audio_result(repeats=repeats(), timing={"warmup_runs": 0, "roles": {}})))[0]
    assert UNWARMED in results and BENCH not in results
    assert "0 discarded warm-up run(s)" in " ".join(section(block(results), UNWARMED).split())


def test_two_repeats_are_a_single_sample_and_the_heading_does_not_say_benchmark(tmp_path):
    results = built(audio_run(tmp_path, audio_result(repeats=repeats(chip=(5.0, 1.6), control=(60.0, 30.0)))))[0]
    assert SINGLE in results and BENCH not in results
    body = " ".join(section(block(results), SINGLE).split())
    assert "single sample" in body and "fewer than 3" in body


def test_one_role_with_two_repeats_makes_the_whole_section_a_single_sample(tmp_path):
    results = built(audio_run(tmp_path, audio_result(repeats=repeats(control=(60.0, 30.0)))))[0]
    assert SINGLE in results and BENCH not in results


def test_without_repeats_the_section_says_so_and_does_not_crash(tmp_path):
    res = audio_result()
    assert "repeats" not in res
    results = built(audio_run(tmp_path, res))[0]
    assert "was not repeated" in section(block(results), SINGLE) and BENCH not in results


def test_malformed_repeats_print_not_recorded_and_do_not_crash(tmp_path):
    bad = {"chip": [{"timings": {"generate_and_decode_s": "fast"}}, 5, {"timings": {"generate_and_decode_s": 1e30}}, {"timings": {"generate_and_decode_s": 2.0}}],
           "control_chip": "x"}
    results = built(audio_run(tmp_path, audio_result(repeats=bad)))[0]
    assert BENCH not in results                  # one good value is not three repeats
    row = next(l for l in section(block(results), SINGLE).splitlines() if "the whole `generate` call" in l)
    assert row.count("not recorded") >= 6 and "| 2 |" not in row          # three missing values, then no min, median or max over the one that is left
    assert "was not repeated" in section(block(built(audio_run(tmp_path / "b", audio_result(repeats=[1, 2])) if (tmp_path / "b").mkdir() is None else None)[0]),
                                         SINGLE)


def test_a_repeat_count_in_the_record_that_the_lists_do_not_support_does_not_make_a_benchmark(tmp_path):
    """The builder counts the measured values itself; a `timing` record that says benchmark changes nothing."""
    res = audio_result(repeats=repeats(chip=(5.0, 1.6), control=(60.0, 30.0)),
                       timing={"warmup_runs": 1, "roles": {"chip": {"label": "benchmark", "repeats": 3}}})
    assert BENCH not in built(audio_run(tmp_path, res))[0]


def finish_bundle(run):
    b = bundle_of(run)
    shutil.copyfile(run / "ledger.jsonl", b / "ledger.jsonl")
    Supervisor._copy_package(SimpleNamespace(run_dir=run), b)
    return gate_bundle(run / "stages" / "8", run)


def test_the_bundle_gate_passes_a_benchmark_and_a_single_sample_that_each_say_what_they_are(tmp_path):
    for sub, reps in (("a", repeats()), ("b", repeats(chip=(5.0, 1.6), control=(60.0, 30.0)))):
        run = audio_run(tmp_path / sub, audio_result(repeats=reps))
        built(run)
        gate = finish_bundle(run)
        assert gate.ok, gate.reasons


def test_the_bundle_gate_refuses_a_benchmark_heading_over_fewer_than_three_repeats(tmp_path):
    run = audio_run(tmp_path, audio_result(repeats=repeats()))
    built(run)
    res = json.loads((run / RESULT).read_text())
    res["repeats"]["chip"] = res["repeats"]["chip"][:2]               # the repeats the bundle was built from are no longer three
    (run / RESULT).write_text(json.dumps(res))
    gate = finish_bundle(run)
    assert not gate.ok and any("benchmark" in r and "chip" in r and "2" in r for r in gate.reasons), gate.reasons


def test_a_part_that_one_repeat_lacks_shows_no_spread_and_says_not_recorded(tmp_path):
    reps = repeats()
    del reps["chip"][1]["timings"]["conditioning_s"]
    body = section(block(built(audio_run(tmp_path, audio_result(repeats=reps)))[0]), BENCH)
    row = next(l for l in body.splitlines() if l.startswith("| Fine-tune, 8 steps | the T5Gemma encoder"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3:6] == ["not recorded"] * 3 and "not recorded" in cells[2]
    full = next(l for l in body.splitlines() if l.startswith("| Fine-tune, 8 steps | the whole `generate` call"))
    assert "| 1.5 | 1.6 | 5 |" in full                                       # a complete part keeps its spread


def test_a_control_with_no_repeats_makes_the_section_a_single_sample(tmp_path):
    for sub, reps in (("empty", {"control_chip": []}), ("missing", {}), ("string", {"control_chip": "x"}), ("none", {"control_chip": None})):
        full = repeats()
        full.pop("control_chip")
        full.update(reps)
        results = built(audio_run(tmp_path / sub, audio_result(repeats=full)))[0]
        assert BENCH not in results and SINGLE in results, sub


def test_the_bundle_gate_counts_a_control_with_no_repeats_as_zero(tmp_path):
    for sub, change in (("empty", lambda r: r["repeats"].__setitem__("control_chip", [])), ("missing", lambda r: r["repeats"].pop("control_chip")),
                        ("string", lambda r: r["repeats"].__setitem__("control_chip", "x")), ("none", lambda r: r["repeats"].__setitem__("control_chip", None))):
        run = audio_run(tmp_path / sub, audio_result(repeats=repeats()))
        built(run)
        res = json.loads((run / RESULT).read_text())
        change(res)
        (run / RESULT).write_text(json.dumps(res))
        gate = finish_bundle(run)
        assert not gate.ok and any("control_chip has 0" in r for r in gate.reasons), (sub, gate.reasons)


def test_repeats_listed_for_a_role_the_result_has_no_run_for_still_count(tmp_path):
    res = audio_result(repeats=repeats(control=(60.0, 30.0)))
    del res["runs"]["control_chip"]
    run = audio_run(tmp_path / "b", res)
    assert BENCH not in built(run)[0]
    run2 = audio_run(tmp_path / "g", audio_result(repeats=repeats()))
    built(run2)
    data = json.loads((run2 / RESULT).read_text())
    del data["runs"]["control_chip"]
    data["repeats"]["control_chip"] = data["repeats"]["control_chip"][:2]
    (run2 / RESULT).write_text(json.dumps(data))
    assert any("control_chip has 2" in r for r in finish_bundle(run2).reasons)


def test_a_run_with_only_the_fine_tune_needs_only_the_fine_tunes_repeats(tmp_path):
    res = audio_result(repeats={"chip": repeats()["chip"]})
    del res["runs"]["control_chip"]
    run = audio_run(tmp_path, res)
    assert BENCH in built(run)[0] and finish_bundle(run).ok


def test_the_bundle_gate_does_not_count_a_true_as_a_measured_time(tmp_path):
    run = audio_run(tmp_path, audio_result(repeats=repeats()))
    built(run)
    res = json.loads((run / RESULT).read_text())
    res["repeats"]["chip"][2]["timings"]["generate_and_decode_s"] = True
    (run / RESULT).write_text(json.dumps(res))
    gate = finish_bundle(run)
    assert not gate.ok and any("chip has 2" in r for r in gate.reasons), gate.reasons


def test_a_warmup_that_is_true_is_not_recorded(tmp_path):
    results = built(audio_run(tmp_path, audio_result(repeats=repeats(), timing={"warmup_runs": True})))[0]
    body = " ".join(section(block(results), UNWARMED).split())
    assert "warm-up runs before the repeats: not recorded" in body and "True" not in body and BENCH not in results


def test_the_load_comes_from_the_first_repeat_only(tmp_path):
    reps = repeats()
    reps["chip"][2]["timings"]["model_load_s"] = 99.0           # a later repeat's figure is not the load
    body = " ".join(section(block(built(audio_run(tmp_path, audio_result(repeats=reps)))[0]), BENCH).split())
    assert "model load 6.125 seconds" in body and "99" not in body


def test_the_bundle_gate_refuses_a_benchmark_heading_when_the_warmup_is_not_what_the_rule_asks(tmp_path):
    run = audio_run(tmp_path, audio_result(repeats=repeats()))
    built(run)
    res = json.loads((run / RESULT).read_text())
    res["timing"]["warmup_runs"] = 0
    (run / RESULT).write_text(json.dumps(res))
    gate = finish_bundle(run)
    assert not gate.ok and any("benchmark" in r and "warm-up" in r for r in gate.reasons), gate.reasons
    del res["timing"]
    (run / RESULT).write_text(json.dumps(res))
    assert any("warm-up" in r for r in finish_bundle(run).reasons)
    res["timing"] = {"warmup_runs": True}                        # true is not one discarded run
    (run / RESULT).write_text(json.dumps(res))
    assert any("warm-up" in r for r in finish_bundle(run).reasons)


def test_the_bundle_gate_passes_an_unwarmed_section_that_says_what_it_is(tmp_path):
    run = audio_run(tmp_path, audio_result(repeats=repeats(), timing=None))
    built(run)
    assert finish_bundle(run).ok


def test_the_bundle_gate_refuses_a_benchmark_heading_when_the_result_has_no_repeats(tmp_path):
    run = audio_run(tmp_path, audio_result(repeats=repeats()))
    built(run)
    res = json.loads((run / RESULT).read_text())
    del res["repeats"]
    (run / RESULT).write_text(json.dumps(res))
    gate = finish_bundle(run)
    assert not gate.ok and any("benchmark" in r for r in gate.reasons), gate.reasons


def test_the_bundle_gate_refuses_a_benchmark_heading_when_the_result_file_is_unreadable(tmp_path):
    run = audio_run(tmp_path, audio_result(repeats=repeats()))
    built(run)
    (run / RESULT).write_text("[1]")
    gate = finish_bundle(run)
    assert not gate.ok and any("benchmark" in r for r in gate.reasons), gate.reasons


def test_the_section_text_has_none_of_the_phrases_the_style_sweep_looks_for(tmp_path):
    results = built(audio_run(tmp_path, audio_result(repeats=repeats(), timing={"warmup_runs": 1, "roles": {}})))[0]
    single = built(audio_run(tmp_path / "s", audio_result(repeats=repeats(chip=(5.0, 1.6), control=(60.0, 30.0)))))[0]
    for text in (section(block(results), BENCH), section(block(single), SINGLE)):
        for phrase in SWEEP:
            assert phrase not in text, phrase


def test_the_headings_the_builder_writes_are_the_ones_the_gate_looks_for():
    from orchard import resident
    src = (Path(__file__).resolve().parents[1] / "orchard" / "skills" / "operator-bundle-templates" / "build_bundle.py").read_text()
    assert f'AUDIO_BENCH_HEADING = "{resident.BENCH_HEADING}"' in src and f'AUDIO_SINGLE_HEADING = "{resident.SINGLE_HEADING}"' in src
    assert (BENCH, SINGLE, UNWARMED) == (resident.BENCH_HEADING, resident.SINGLE_HEADING, resident.UNWARMED_HEADING)
    assert f'AUDIO_UNWARMED_HEADING = "{resident.UNWARMED_HEADING}"' in src and f"AUDIO_MIN_WARMUP = {resident.WARMUP_RUNS} " in src
    assert f"AUDIO_MIN_REPEATS = {resident.MIN_REPEATS} " in src
