# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The audio package rebuilt for the decoder-on-chip path (issue 161): what the builder copies, the packaged generate command's default, the card's
numbers for the decoder and the CPU, the operator's listening note, the explicit result flag, the extra evidence files and the staging directory.

Everything is a fake run in a temporary directory, as in test_audio_package.py (no chip, no network, no model).
"""
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import audio_package as ap
from test_audio_package import (LIB, REV, SETTINGS, args, arr, audio_result, bench_file, card_of, gate, run_pinned, size, timings,  # noqa: F401
                                world, staged)

HERE = Path(ap.__file__).resolve().parent
DEC_NOTE = "2026-10-29: the operator listened to the clips of the decoder-on-chip run and judged them good."


def timings_dec(g, dec=0.44):
    t = timings(g)
    t["chip_decoder_call_s"] = dec
    return t


def size_dec(seconds, gens, warmup=1):
    """A size whose runs carry the chip decoder's call time, with its spread recomputed as the sweep file records it."""
    s = size(seconds, gens, warmup)
    vals = []
    for g, run in zip(gens, s["runs"]):
        run["timings"] = timings_dec(g, round(g / 2, 3))
        vals.append(run["timings"]["chip_decoder_call_s"])
    vals.sort()
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    s["parts"]["chip_decoder_call_s"] = {"n": n, "min": vals[0], "median": round(med, 3), "max": vals[-1]}
    return s


def decoder_result(**kw):
    res = audio_result(**kw)
    res["runs"]["chip"]["chip_decoder"] = True
    return res


def decoder_world(world, **kw):
    """The world with a benchmark whose sizes include the chip decoder part, and a result that says the chip run had the decoder on the chip."""
    world.write_bench(bench_file([size_dec(10.0, [0.8, 0.82, 0.83]), size_dec(30.0, [1.4, 1.5, 1.6])]))
    world.write_result(decoder_result(**kw))
    return world


def cpu_runs():
    return {"reference": {"role": "reference", "settings": dict(SETTINGS, device="cpu", dtype="float32"), "timings": {"generate_and_decode_s": 24.795}},
            "host_same_env": {"role": "host_same_env", "settings": dict(SETTINGS, device="cpu", dtype="float32"), "timings": {"generate_and_decode_s": 4.087}}}


def command(world, **kw):
    base = dict(result=None, probe=None, agreement_report=None, listening_note=None, out=None)
    base.update(kw)
    buf = io.StringIO()
    code = ap.main(args(world, **base), out=buf, hostname="some-host")
    return code, buf.getvalue()


# ---- 1. the runtime the card's command needs is in the package ----------------------------------------------------------------------

def runtime_needs(runtime: Path) -> set:
    """Every file under runtime/ that a shipped runtime file imports or loads by name, found in the shipped sources."""
    need = set()
    for f in runtime.rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r"from orchard import (\w+)", src):
            need.add(runtime / "orchard" / f"{m.group(1)}.py")
        for m in re.finditer(r"from orchard\.(\w+) import", src):
            need.add(runtime / "orchard" / f"{m.group(1)}.py")
        for m in re.finditer(r'HERE / "(\w+\.py)"', src):
            need.add(f.parent / m.group(1))
        for m in re.finditer(r'"skills" / "([\w-]+)" / "(\w+\.py)"', src):
            need.add(runtime / "orchard" / "skills" / m.group(1) / m.group(2))
    return need


def test_every_file_a_shipped_runtime_file_imports_or_loads_is_shipped(staged):
    needs = runtime_needs(staged / "runtime")
    assert needs                                                                         # the scan found something to check
    missing = sorted(p.relative_to(staged).as_posix() for p in needs if not p.is_file())
    assert missing == []


def test_the_chip_decoder_and_its_loaders_are_shipped_unmodified(staged):
    for rel in ("skills/audio-chip-templates/chip_decoder.py", "skills/audio-chip-templates/audio_runtimes.py", "resident.py"):
        assert (staged / "runtime" / "orchard" / rel).read_bytes() == (HERE / rel).read_bytes(), rel


def test_the_builder_refuses_to_stage_when_a_runtime_source_is_missing(world, tmp_path):
    root = tmp_path / "fake-root"
    (root / "skills" / "audio-chip-templates").mkdir(parents=True)
    with pytest.raises(ap.AudioPackageError, match="chip_generate.py"):
        world.stage(repo_root=root)


# ---- 2. the packaged generate command defaults to the decoder on the chip -------------------------------------------------------------

def test_the_packaged_generate_command_adds_the_chip_decoder_by_default(tmp_path):
    r = run_pinned(tmp_path, REV, REV)
    assert r.returncode == 0 and "GENERATED ['--model', 'small-music', '--out', 'x', '--chip-decoder']" in r.stdout


def test_the_host_decoder_flag_is_the_documented_fallback_and_chip_generate_never_sees_it(tmp_path):
    r = run_pinned(tmp_path, REV, REV, args=("--model", "small-music", "--host-decoder", "--out", "x"))
    assert r.returncode == 0 and "GENERATED ['--model', 'small-music', '--out', 'x']" in r.stdout
    assert "--host-decoder" not in r.stdout and "--chip-decoder" not in r.stdout


def test_the_environment_variable_selects_the_host_decoder_because_the_server_cannot_pass_the_flag(tmp_path):
    r = run_pinned(tmp_path, REV, REV)
    assert "--chip-decoder" in r.stdout                                                  # the default, with the same fake library
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path / "lib"), "AUDIO_PIN_REVISION": REV, "AUDIO_HOST_DECODER": "1"}
    r = subprocess.run([sys.executable, str(tmp_path / "gen" / "pinned_generate.py"), "--model", "small-music", "--out", "x"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "GENERATED ['--model', 'small-music', '--out', 'x']" in r.stdout


def test_an_explicit_chip_decoder_flag_is_not_repeated(tmp_path):
    r = run_pinned(tmp_path, REV, REV, args=("--model", "small-music", "--chip-decoder", "--out", "x"))
    assert r.stdout.count("--chip-decoder") == 1


def test_the_two_decoder_flags_together_are_refused(tmp_path):
    r = run_pinned(tmp_path, REV, REV, args=("--model", "small-music", "--chip-decoder", "--host-decoder", "--out", "x"))
    assert r.returncode == 2 and "GENERATED" not in r.stdout and "--host-decoder" in r.stderr


def test_run_sh_and_running_md_document_the_host_decoder_fallback(staged):
    assert "AUDIO_HOST_DECODER" in (staged / "run.sh").read_text()
    assert "AUDIO_HOST_DECODER" in (staged / "docs" / "running.md").read_text()
    assert "--host-decoder" in (staged / "docs" / "running.md").read_text()


def test_the_gate_refuses_documents_that_do_not_name_the_fallback(staged):
    f = staged / "docs" / "running.md"
    f.write_text(f.read_text().replace("AUDIO_HOST_DECODER", "SOMETHING"))
    assert any("AUDIO_HOST_DECODER" in p for p in gate(staged))


# ---- 3. the benchmark's decoder part and the CPU timings are card rows ----------------------------------------------------------------

def test_a_sweep_with_the_decoder_part_adds_a_decoder_row_per_size_and_the_card_says_the_decoder_runs_on_the_chip(world):
    decoder_world(world)
    world.stage()
    card = card_of(world.pkg)
    assert "| 10 s clip, decoder call on the chip, median of 3 | 0.41 s | benchmark |" in card
    assert "| 30 s clip, decoder call on the chip, median of 3 | 0.75 s | benchmark |" in card
    assert "the autoencoder decoder's six transformer layers" in card
    assert "Only the diffusion transformer runs on the chip" not in card
    assert gate(world.pkg) == []


def test_a_sweep_without_the_decoder_part_keeps_the_old_card_wording(staged):
    card = card_of(staged)
    assert "decoder call on the chip" not in card and "Only the diffusion transformer runs on the chip" in card


def test_the_decoder_part_must_be_in_every_run_of_a_size_to_count(world):
    b = bench_file([size_dec(10.0, [0.8, 0.82, 0.83])])
    del b["sizes"][0]["runs"][1]["timings"]["chip_decoder_call_s"]
    world.write_bench(b)
    world.write_result(decoder_result())
    with pytest.raises(ap.AudioPackageError, match="chip_decoder_call_s"):
        world.stage()


def test_a_spread_that_the_decoder_runs_do_not_give_is_refused(world):
    b = bench_file([size_dec(10.0, [0.8, 0.82, 0.83])])
    b["sizes"][0]["parts"]["chip_decoder_call_s"]["median"] = 0.1
    world.write_bench(b)
    world.write_result(decoder_result())
    with pytest.raises(ap.AudioPackageError, match="median of chip_decoder_call_s"):
        world.stage()


def test_a_benchmark_with_the_decoder_on_the_chip_needs_agreement_measured_with_it_too(world):
    world.write_bench(bench_file([size_dec(10.0, [0.8, 0.82, 0.83])]))
    world.write_result(audio_result())                                                   # the agreement run did not have the decoder on the chip
    with pytest.raises(ap.AudioPackageError, match="decoder"):
        world.stage()


def test_agreement_with_the_decoder_on_the_chip_needs_a_benchmark_with_it_too(world):
    world.write_result(decoder_result())                                                 # the default benchmark has no decoder part
    with pytest.raises(ap.AudioPackageError, match="decoder"):
        world.stage()


def test_the_card_gives_the_cpu_float32_timings_as_single_samples(world):
    decoder_world(world, reference_runs=cpu_runs())
    world.stage()
    card = card_of(world.pkg)
    assert "| 10 s clip on the CPU in float32 (reference environment), generate and decode, one sample | 24.795 s | single sample |" in card
    assert "| 10 s clip on the CPU in float32 (same environment as the chip run), generate and decode, one sample | 4.087 s | single sample |" in card
    assert gate(world.pkg) == []


def test_a_cpu_timing_that_is_not_the_evidence_is_refused_by_the_gate(world):
    decoder_world(world, reference_runs=cpu_runs())
    world.stage()
    f = world.pkg / "README.md"
    f.write_text(f.read_text().replace("| 4.087 s |", "| 1.087 s |"))
    assert any("must be 4.087 s" in p for p in gate(world.pkg))


def test_a_cpu_run_that_was_not_float32_on_the_cpu_is_not_a_card_row(world):
    runs = cpu_runs()
    runs["reference"]["settings"]["dtype"] = "bfloat16"
    decoder_world(world, reference_runs=runs)
    world.stage()
    assert "reference environment" not in card_of(world.pkg)


def test_a_decoder_row_edited_in_the_card_is_refused_by_the_gate(world):
    decoder_world(world)
    world.stage()
    f = world.pkg / "README.md"
    f.write_text(f.read_text().replace("| 0.41 s |", "| 0.1 s |"))
    assert any("decoder call on the chip" in p for p in gate(world.pkg))


def test_a_card_that_claims_the_decoder_on_the_chip_without_the_evidence_is_refused(staged):
    f = staged / "README.md"
    f.write_text(f.read_text().replace("Only the diffusion transformer runs on the chip.",
                                       "The diffusion transformer and the autoencoder decoder's six transformer layers run on the chip."))
    assert any("decoder" in p and "evidence" in p for p in gate(staged))


def test_a_card_that_says_only_the_transformer_runs_on_the_chip_beside_decoder_evidence_is_refused(world):
    decoder_world(world)
    world.stage()
    f = world.pkg / "README.md"
    f.write_text(f.read_text().replace("## Limitations\n", "## Limitations\n\n- Only the diffusion transformer runs on the chip.\n"))
    assert any("Only the diffusion transformer runs on the chip" in p for p in gate(world.pkg))


# ---- 4. the operator's listening note --------------------------------------------------------------------------------------------

def test_the_listening_note_is_stated_with_its_date_as_the_operators_note(world):
    decoder_world(world)
    world.stage(listening_note=DEC_NOTE)
    card = card_of(world.pkg)
    assert "## Operator's listening note" in card and DEC_NOTE in card
    assert "No quality evaluation was run" in card                                       # the gate's sentence stays
    assert gate(world.pkg) == []
    assert json.loads((world.pkg / ap.RECORD_FILE).read_text())["listening_note"] == DEC_NOTE


def test_a_listening_note_without_a_date_is_refused(world):
    with pytest.raises(ap.AudioPackageError, match="date"):
        world.stage(listening_note="the operator listened and judged them good")


def test_a_listening_note_that_claims_sound_quality_in_the_gates_words_is_refused_by_the_gate(world):
    world.stage(listening_note="2026-10-29: the operator says the clips sound great.")
    assert any("sound quality" in p for p in gate(world.pkg))


def test_a_listening_note_with_a_timing_is_refused_by_the_gate(world):
    world.stage(listening_note="2026-10-29: the operator listened to a clip made in 2 seconds.")
    assert any("timing outside the Numbers table" in p for p in gate(world.pkg))


def test_a_card_without_a_note_has_no_note_section(staged):
    assert "listening note" not in card_of(staged).lower()


def test_a_note_section_without_a_date_is_refused_by_the_gate(world):
    world.stage(listening_note=DEC_NOTE)
    f = world.pkg / "README.md"
    f.write_text(f.read_text().replace("2026-10-29", "today"))
    assert any("listening note" in p and "date" in p for p in gate(world.pkg))


# ---- 5. an explicit result file, extra evidence, and a staging directory ------------------------------------------------------------

def test_an_explicit_result_file_is_used_instead_of_stage_two(world, tmp_path):
    old = audio_result()
    old["agreement"]["chip_vs_reference"] = arr(0.5, 0.9)                                # a different value, so the card shows which file it read
    world.write_result(old)
    new = tmp_path / "audio_result.new.json"
    new.write_text(json.dumps(audio_result()))
    world.stage(result_path=new)
    assert (world.pkg / "evidence" / "audio_result.json").read_bytes() == new.read_bytes()
    assert "| chip against the CPU reference, final latent PCC | 0.9905 |" in card_of(world.pkg)


def test_without_the_flag_the_stage_two_file_is_read(world):
    world.stage()
    assert (world.pkg / "evidence" / "audio_result.json").read_bytes() == (world.run / "stages" / "2" / "audio_result.json").read_bytes()


def test_the_run_files_are_not_edited_by_staging_with_a_result_file(world, tmp_path):
    before = (world.run / "stages" / "2" / "audio_result.json").read_bytes()
    new = tmp_path / "r.json"
    new.write_text(json.dumps(audio_result(extra=1)))
    world.stage(result_path=new)
    assert (world.run / "stages" / "2" / "audio_result.json").read_bytes() == before


def test_a_result_file_that_cannot_back_a_package_is_still_refused(world, tmp_path):
    bad = audio_result()
    bad["agreement"]["chip_vs_reference"]["ok"] = False
    f = tmp_path / "bad.json"
    f.write_text(json.dumps(bad))
    with pytest.raises(ap.AudioPackageError, match="did not agree"):
        world.stage(result_path=f)


def test_extra_evidence_files_are_copied_with_home_paths_replaced_and_listed_in_the_record(world, tmp_path):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"model": "small-music", "where": "/home/someone/orchard-runs/x/chip2"}))
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps({"ok": True}))
    world.stage(probe_path=probe, agreement_report_path=rep, home="/home/someone")
    text = (world.pkg / "evidence" / "decode-probe.json").read_text()
    assert "/home/someone" not in text and "<home>/orchard-runs/x/chip2" in text
    assert json.loads((world.pkg / "evidence" / "audio-agreement-report.json").read_text()) == {"ok": True}
    paths = {r["path"] for r in json.loads((world.pkg / ap.RECORD_FILE).read_text())["files"]}
    assert {"evidence/decode-probe.json", "evidence/audio-agreement-report.json"} <= paths
    assert gate(world.pkg) == []


def test_an_extra_evidence_file_that_is_not_json_is_refused(world, tmp_path):
    bad = tmp_path / "probe.json"
    bad.write_text("not json")
    with pytest.raises(ap.AudioPackageError, match="probe"):
        world.stage(probe_path=bad)


def test_an_extra_evidence_file_with_a_token_is_refused_by_the_gate(world, tmp_path):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"note": "hf_" + "a" * 34}))
    world.stage(probe_path=probe)
    assert any("evidence/decode-probe.json" in p for p in gate(world.pkg))


def test_extra_evidence_edited_after_staging_is_refused_by_the_gate(world, tmp_path):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"layers": 6}))
    world.stage(probe_path=probe)
    (world.pkg / "evidence" / "decode-probe.json").write_text(json.dumps({"layers": 7}))
    assert any("decode-probe.json changed after it was staged" in p for p in gate(world.pkg))


def test_staging_into_a_new_directory_leaves_the_runs_own_package_directory_alone(world, tmp_path):
    out = tmp_path / "new-stage" / ap.PACKAGE_NAME
    world.stage(out=out)
    assert (out / "README.md").is_file() and not world.pkg.exists()


def test_the_command_with_out_writes_the_commands_text_beside_the_package_and_not_in_the_run(world, tmp_path):
    out = tmp_path / "new-stage" / ap.PACKAGE_NAME
    code, text = command(world, out=str(out))
    assert code == 0, text
    assert (out.parent / "AUDIO_PUBLISH_COMMANDS.txt").is_file()
    assert not (world.run / "stages" / "7").exists()


def test_the_command_with_check_and_out_gates_that_directory(world, tmp_path):
    out = tmp_path / "new-stage" / ap.PACKAGE_NAME
    world.stage(out=out)
    (out / "README.md").write_text("no card")
    code, text = command(world, out=str(out), check=True)
    assert code == 1 and "does not pass the gate" in text


def test_the_command_passes_the_listening_note_the_result_file_and_the_extras_through(world, tmp_path):
    decoder_world(world)
    new = tmp_path / "res.json"
    new.write_text(json.dumps(decoder_result(extra=2)))
    probe = tmp_path / "p.json"
    probe.write_text("{}")
    out = tmp_path / "s" / ap.PACKAGE_NAME
    code, text = command(world, out=str(out), result=str(new), probe=str(probe), listening_note=DEC_NOTE)
    assert code == 0, text
    assert DEC_NOTE in (out / "README.md").read_text() and (out / "evidence" / "decode-probe.json").is_file()
    assert (out / "evidence" / "audio_result.json").read_bytes() == new.read_bytes()


# ---- 6. deriving a result file from a newer chip run ---------------------------------------------------------------------------------

def chip_run(**over):
    run = {"dir": "overnight/e2e1", "model": "small-music", "revision": REV, "library_commit": LIB, "settings": SETTINGS,
           "timings": {"generate_and_decode_s": 0.921, "chip_decoder_call_s": 0.438}}
    run.update(over)
    return run


def test_a_derived_result_replaces_the_chip_run_and_its_agreement_and_says_what_it_did():
    from orchard import audio_result_derive as rd
    base = audio_result(runtime="stable-audio-3-dit-on-chip")
    base["agreement"]["chip_vs_host_same_env"] = arr(0.97, 0.2)
    base["runs"]["control_chip"] = {"role": "control_chip"}
    base["repeats"] = {"chip": [{"timings": {"generate_and_decode_s": 1.7}}], "control_chip": [{"role": "control_chip"}]}
    new = arr(0.99, 0.1)
    out = rd.derive(base, chip_run=chip_run(), agreement=new, base_sha256="abc", chip_label="e2e1")
    assert out["runs"]["chip"]["timings"]["chip_decoder_call_s"] == 0.438 and out["runs"]["chip"]["chip_decoder"] is True
    assert out["agreement"]["chip_vs_reference"] == new
    assert "chip_vs_host_same_env" not in out["agreement"]                                # it was measured on the old run, so it is dropped
    assert "chip" not in out["repeats"] and out["repeats"]["control_chip"] == [{"role": "control_chip"}]      # the earlier chip run's repeats are dropped
    assert "repeats.chip" in out["derived_from"]["dropped"]
    assert out["runs"]["control_chip"] == {"role": "control_chip"}                         # the control is kept, and the note says whose it is
    d = out["derived_from"]
    assert d["base_sha256"] == "abc" and "runs.chip" in d["replaced"] and "agreement.chip_vs_reference" in d["replaced"]
    assert "control_discrimination" in d["kept_from_base"] and "environment_floor" in d["kept_from_base"]
    assert "decoder" in out["runtime"] and any("decoder" in x for x in out["limits"])
    assert base["runs"]["chip"].get("chip_decoder") is None                                # the base dictionary is not changed


def test_a_derived_result_refuses_a_chip_run_that_is_not_the_runs_revision():
    from orchard import audio_result_derive as rd
    with pytest.raises(rd.DeriveError, match="revision"):
        rd.derive(audio_result(), chip_run=chip_run(revision="1" * 40), agreement=arr(0.99, 0.1), base_sha256="a", chip_label="x")


def test_a_derived_result_refuses_a_chip_run_with_other_settings():
    from orchard import audio_result_derive as rd
    with pytest.raises(rd.DeriveError, match="seed"):
        rd.derive(audio_result(), chip_run=chip_run(settings=dict(SETTINGS, seed=1)), agreement=arr(0.99, 0.1), base_sha256="a", chip_label="x")


def test_a_derived_result_refuses_a_chip_run_without_the_decoder_timing():
    from orchard import audio_result_derive as rd
    with pytest.raises(rd.DeriveError, match="decoder"):
        rd.derive(audio_result(), chip_run=chip_run(timings={"generate_and_decode_s": 1.0}), agreement=arr(0.99, 0.1), base_sha256="a", chip_label="x")


def test_a_derived_result_refuses_an_agreement_that_did_not_pass():
    from orchard import audio_result_derive as rd
    bad = arr(0.5, 0.9)
    bad["ok"] = False
    with pytest.raises(rd.DeriveError, match="agree"):
        rd.derive(audio_result(), chip_run=chip_run(), agreement=bad, base_sha256="a", chip_label="x")


def test_a_derived_result_backs_a_package_end_to_end(world, tmp_path):
    from orchard import audio_result_derive as rd
    out = rd.derive(audio_result(), chip_run=chip_run(), agreement=arr(0.99, 0.1), base_sha256="a", chip_label="x")
    f = tmp_path / "derived.json"
    f.write_text(json.dumps(out))
    world.write_bench(bench_file([size_dec(10.0, [0.8, 0.82, 0.83])]))
    world.stage(result_path=f)
    assert gate(world.pkg) == []


def test_a_card_for_a_derived_result_says_so_and_the_gate_refuses_a_card_that_does_not(world, tmp_path):
    from orchard import audio_result_derive as rd
    out = rd.derive(audio_result(), chip_run=chip_run(), agreement=arr(0.99, 0.1), base_sha256="a", chip_label="x")
    f = tmp_path / "derived.json"
    f.write_text(json.dumps(out))
    world.write_bench(bench_file([size_dec(10.0, [0.8, 0.82, 0.83])]))
    world.stage(result_path=f)
    card = card_of(world.pkg)
    assert "`derived_from`" in card and "measured on the earlier run, with the decoder on the host" in card
    (world.pkg / "README.md").write_text(card.replace("`derived_from`", "the file"))
    assert any("derived from a newer chip run" in p for p in gate(world.pkg))


def test_a_card_for_an_underived_result_does_not_mention_derivation(staged):
    assert "derived_from" not in card_of(staged)


def test_the_derive_command_refuses_to_overwrite_a_file_and_a_run_with_no_result(tmp_path, capsys):
    from orchard import audio_result_derive as rd
    out = tmp_path / "out.json"
    out.write_text("{}")
    assert rd.main([str(tmp_path), str(tmp_path), str(out)]) == 2 and "never overwritten" in capsys.readouterr().err
    assert rd.main([str(tmp_path), str(tmp_path), str(tmp_path / "new.json")]) == 2 and "refused" in capsys.readouterr().err
    assert not (tmp_path / "new.json").exists()
