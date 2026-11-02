# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Every reader of a run's evidence, marker and record files reads through orchard/safeio.py, and a file it refuses gives a named reason
that reaches the gate or the stage failure (issues 335 and 287). Before this, a null byte in an evidence path raised out of the gate, the
code that records evidence hashed a file with a plain `open()` (a FIFO blocked it), and the readers of the sampling and graft decisions
turned a refused file into None, which their callers read as "no file".

Each hostile file runs in a child process with an address-space limit and a timeout (tests/hostile_inputs.py `run_limited`), because a
reader that follows a link to /dev/zero reads until memory is gone and one that opens a FIFO blocks. The child prints one JSON line.
The shapes are a FIFO, a link to /dev/zero, a file over the size cap (the cap is lowered in the child), a link that leaves the run
directory, a path with a null byte and, where the reader parses JSON, a file holding NaN.
"""
import json
import os
import textwrap
from pathlib import Path

import pytest

import hostile_inputs as hi
from test_stages import GOOD_DELTA, mesh_stage, swap_config, write

CHILD_PRELUDE = textwrap.dedent('''
    import json, os, tempfile
    from pathlib import Path
    from orchard import safeio, stages, sampling_choice, graft_trial
    for mod, name in ((stages, "JSON_MAX_BYTES"), (stages, "TEXT_MAX_BYTES"),
                      (sampling_choice, "MAX_JSON_BYTES"), (graft_trial, "MAX_JSON_BYTES")):
        setattr(mod, name, 64)                          # "oversize" is anything over 64 bytes here
    base = Path(tempfile.mkdtemp(prefix="run-files-"))
    run = base / "run"
    run.mkdir()
    outside = base / "outside.json"
    outside.write_text("{}")

    def build(kind, dest):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        if kind == "fifo":
            os.mkfifo(dest)
        elif kind == "dev_zero":
            dest.symlink_to("/dev/zero")
        elif kind == "oversize":
            dest.write_text('{"a": "' + "x" * 200 + '"}')
        elif kind == "symlink_out":
            dest.symlink_to(outside)
        elif kind == "nan":
            dest.write_text('{"a": NaN}')
        else:
            raise AssertionError(kind)

    def report(**kw):
        print(json.dumps(kw, default=str))
''')

SHAPES = ["fifo", "dev_zero", "oversize", "symlink_out"]


def run_child(body: str) -> dict:
    """Run `body` in a memory-limited child after CHILD_PRELUDE; the body calls report(...). A timeout or a crash fails the test."""
    done = hi.run_limited(CHILD_PRELUDE + textwrap.dedent(body), timeout=60)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    assert done.returncode == 0 and lines, f"the child failed (exit {done.returncode}): {done.stderr[-600:]}"
    return json.loads(lines[-1])


# ------------------------------------------------------------------------------------------------ evidence paths (issue 335, point 1)

@pytest.mark.parametrize("shape", SHAPES)
def test_locate_names_why_a_hostile_evidence_file_is_not_usable(shape):
    out = run_child(f'''
        stages.EVIDENCE_HASH_MAX = 64
        build({shape!r}, run / "ev.json")
        found, why = stages.locate(run, "ev.json")
        report(found=found, why=why)
    ''')
    want = {"fifo": "regular file", "dev_zero": "leaves the run directory", "oversize": "larger than", "symlink_out": "leaves the run directory"}[shape]
    assert out["found"] is None and want in out["why"], out


@pytest.mark.parametrize("rel,expect", [("a\0b", "null byte"), ("/etc/passwd", "absolute"), ("../outside", "leaves the run directory"),
                                         ("missing.json", "does not exist"), ("", "not a non-empty string"), (None, "not a non-empty string"),
                                         (7, "not a non-empty string"), ("a\udcffb", None)])
def test_locate_never_raises_and_gives_a_reason(tmp_path, rel, expect):
    from orchard.stages import locate
    found, why = locate(tmp_path, rel)
    assert found is None and why
    if expect:
        assert expect in why


def test_a_null_byte_in_an_evidence_path_is_a_gate_reason_not_an_exception(tmp_path):
    """Issue 335: stages.inside() called os.path.realpath on the path and a ValueError left the gate. Two gates that list evidence."""
    from orchard.stages import gate_delta, gate_mesh_swap
    sd, run = mesh_stage(tmp_path / "a", [swap_config(2, evidence=["stages/4/configs/2/evidence/swap-check.json", "stages/4/x\0y.txt"])])
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("null byte" in r for r in g.reasons)
    run2 = tmp_path / "b"
    write(run2, "stages/0/evidence/e.txt", "e")
    delta = json.loads(json.dumps(GOOD_DELTA))
    delta["differences"][0]["evidence"] = ["stages/0/e\0.txt"]
    write(run2, "stages/0/delta.json", delta)
    g2 = gate_delta(run2 / "stages" / "0", run2)
    assert not g2.ok and any("null byte" in r for r in g2.reasons)


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "symlink_out"])
def test_a_hostile_evidence_file_in_a_gate_list_is_a_named_reason(shape):
    out = run_child(f'''
        from orchard.stages import _evidence
        build({shape!r}, run / "stages" / "2" / "evidence" / "x.json")
        reasons, seen = [], []
        _evidence(run, ["stages/2/evidence/x.json"], "differences[0]", reasons, seen)
        report(reasons=reasons, seen=seen)
    ''')
    assert out["seen"] == [] and len(out["reasons"]) == 1
    assert "is not a file inside the run directory (" in out["reasons"][0]


# --------------------------------------------------------------------------------------- recording evidence (issue 335, point 2)

@pytest.mark.parametrize("shape", SHAPES)
def test_evidence_record_refuses_a_hostile_file_with_a_named_reason(shape):
    out = run_child(f'''
        stages.EVIDENCE_HASH_MAX = 64
        build({shape!r}, run / "stages" / "2" / "x.json")
        try:
            stages.evidence_record(run, run / "stages" / "2" / "x.json")
            report(raised=None)
        except safeio.SafeReadError as exc:
            report(raised=type(exc).__name__, reason=exc.reason)
    ''')
    assert out["raised"] and out["reason"], out


def test_evidence_record_refuses_a_null_byte_path(tmp_path):
    from orchard import safeio
    from orchard.stages import evidence_record
    with pytest.raises(safeio.SafeReadError):
        evidence_record(tmp_path, str(tmp_path / "a\0b"))


def test_evidence_record_follows_a_link_that_stays_inside_the_run_directory(tmp_path):
    """The gates follow such a link once; an evidence list can name a link to the marker, so the record must follow it too."""
    from orchard.stages import evidence_record
    (tmp_path / "real.json").write_text("{}")
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "link.json").symlink_to("../real.json")
    rec = evidence_record(tmp_path, tmp_path / "evidence" / "link.json")
    assert rec["path"] == "real.json" and len(rec["sha256"]) == 64


# ------------------------------------------------------------------------------------------ the graft check (issue 287)

@pytest.mark.parametrize("shape", SHAPES + ["nan"])
def test_a_refused_swap_config_is_a_reason_even_with_no_graft_decision(shape):
    out = run_child(f'''
        build({shape!r}, run / "stages" / "4" / "configs" / "2" / "swap_config.json")
        report(reasons=stages._graft_reasons(run / "stages" / "4", run, 2, "the 2-chip configuration"))
    ''')
    assert len(out["reasons"]) == 1 and "swap_config.json is not readable" in out["reasons"][0], out


def test_a_missing_swap_config_with_no_decision_still_passes(tmp_path):
    from orchard.stages import _graft_reasons
    assert _graft_reasons(tmp_path / "stages" / "4", tmp_path, 2, "x") == []


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "nan"])
def test_a_refused_sources_record_is_named(shape):
    out = run_child(f'''
        write = lambda rel, text: (run / rel).parent.mkdir(parents=True, exist_ok=True) or (run / rel).write_text(text)
        write("stages/4/configs/2/swap_config.json", json.dumps({{"bundle_dir": "/b", "tt_cache": "/c"}}))
        build({shape!r}, run / "stages" / "4" / "configs" / "2" / "model-dir-sources.json")
        report(reasons=stages._graft_reasons(run / "stages" / "4", run, 2, "w"))
    ''')
    assert len(out["reasons"]) == 1 and "model-dir-sources.json is not readable JSON" in out["reasons"][0], out


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "symlink_out"])
def test_a_swap_check_the_gate_cannot_read_is_a_named_reason_in_the_stage_4_gate(shape):
    out = run_child(f'''
        import sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from test_stages import mesh_stage, swap_config
        import tempfile
        sd, run2 = mesh_stage(Path(tempfile.mkdtemp(prefix="mesh-")), [swap_config(2)])
        build({shape!r}, run2 / "stages/4/configs/2/evidence/swap-check.json")
        g = stages.gate_mesh_swap(sd, run2, required=(2,))
        report(ok=g.ok, reasons=list(g.reasons))
    ''')
    assert not out["ok"] and any("swap-check.json cannot be used (" in r for r in out["reasons"]), out


def _decision(sd: Path, name: str, record: dict):
    sd.mkdir(parents=True, exist_ok=True)
    (sd / name).write_text(json.dumps(record))


SAMPLING_FILES = ["sampling-decision.json", "graft-decision.json", "evidence/swap-check.json", "sampling-trial/test-result.json",
                  "sampling-trial/result.json", "sampling-trial/variant/evidence/swap-check.json",
                  "graft-trial/variant/evidence/swap-check.json"]


@pytest.mark.parametrize("shape", SHAPES + ["nan"])
@pytest.mark.parametrize("rel", ["sampling-decision.json", "graft-decision.json", "evidence/swap-check.json", "sampling-trial/test-result.json",
                                 "sampling-trial/variant/evidence/swap-check.json", "graft-trial/variant/evidence/swap-check.json"])
def test_the_sampling_gate_names_each_refused_file_it_reads(shape, rel):
    """sampling_choice read these through a helper that returned None for a refusal; a refused decision read as no decision."""
    out = run_child(f'''
        sd = run / "stages" / "2"
        sd.mkdir(parents=True)
        (sd / "sampling-decision.json").write_text(json.dumps({{"choice": "keep", "reason": "r", "override": "auto"}}))
        build({shape!r}, sd / {rel!r})
        report(reasons=sampling_choice.decision_problems(sd, run), refusals=sampling_choice.refusals(sd, run))
    ''')
    assert len(out["reasons"]) >= 1 and any(rel.split("/", 1)[-1] in r and "is not readable" in r for r in out["reasons"]), out
    assert out["refusals"]


@pytest.mark.parametrize("shape", SHAPES + ["nan"])
def test_a_refused_genconfig_file_is_named_and_the_need_says_why(shape):
    out = run_child(f'''
        build({shape!r}, run / "stages/0/evidence/genconfig-license.json")
        need = sampling_choice.need(run, "auto")
        report(evidence=need["evidence"], refusals=sampling_choice.refusals(run / "stages" / "2", run))
    ''')
    assert "is not readable" in out["evidence"][0] and out["refusals"], out


@pytest.mark.parametrize("shape", SHAPES + ["nan"])
def test_a_refused_graft_decision_is_named_and_does_not_read_as_no_decision(shape):
    out = run_child(f'''
        sd = run / "stages" / "2"
        build({shape!r}, sd / "graft-decision.json")
        report(problems=graft_trial.decision_problems(sd, run), decision=graft_trial.read_decision(sd), chosen=graft_trial.chosen(run))
    ''')
    assert out["decision"] is None and out["chosen"] is False
    assert len(out["problems"]) >= 1 and "graft-decision.json is not readable" in out["problems"][0], out


@pytest.mark.parametrize("rel", ["graft-trial/trial-result.json", "result.json", "graft-trial/variant/evidence/swap-check.json"])
@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "nan"])
def test_a_refused_file_a_graft_decision_was_made_from_is_named(shape, rel):
    out = run_child(f'''
        sd = run / "stages" / "2"
        sd.mkdir(parents=True)
        (sd / "graft-decision.json").write_text(json.dumps({{"choice": "graft", "reason": "r"}}))
        build({shape!r}, sd / {rel!r})
        report(problems=graft_trial.decision_problems(sd, run))
    ''')
    assert out["problems"] and "is not readable" in out["problems"][0], out


def test_a_missing_decision_and_missing_inputs_are_still_no_problem(tmp_path):
    from orchard import graft_trial, sampling_choice
    assert graft_trial.decision_problems(tmp_path, tmp_path) == [] and sampling_choice.decision_problems(tmp_path, tmp_path) == []
    assert graft_trial.refusals(tmp_path, tmp_path) == [] and sampling_choice.refusals(tmp_path, tmp_path) == []


# --------------------------------------------------------------------------------------------- the supervisor's own reads

@pytest.mark.parametrize("shape", SHAPES + ["nan"])
def test_the_supervisor_writes_a_refusal_to_the_ledger_and_does_not_use_the_file(shape):
    out = run_child(f'''
        from types import SimpleNamespace
        from orchard import supervisor
        supervisor.RUN_JSON_MAX = 64
        entries = []
        sup = SimpleNamespace(ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
        build({shape!r}, run / "m.json")
        got = supervisor.Supervisor._run_json(sup, run / "m.json", "m.json", 3)
        report(value=got.value, refused=got.refused, entries=entries)
    ''')
    assert out["value"] is None and out["refused"]
    (args, kw), = out["entries"]
    assert args[0] == "decision" and kw["file"] == "m.json" and kw["reason"], out


def test_the_supervisor_does_not_record_a_missing_file_as_a_refusal(tmp_path):
    from types import SimpleNamespace
    from orchard.supervisor import Supervisor
    entries = []
    sup = SimpleNamespace(ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    got = Supervisor._run_json(sup, tmp_path / "none.json", "none.json")
    assert got.missing and got.value is None and entries == []


# ----------------------------------------------------------------------------------------------- the template copy and bundle text

@pytest.mark.parametrize("shape", ["dev_zero", "symlink_out"])
def test_a_template_copy_that_is_a_link_is_a_named_reason(shape):
    out = run_child(f'''
        sd = run / "stages" / "1"
        build({shape!r}, sd / "reference_gate.py")
        report(reasons=stages._template_reasons(sd, "reference_gate.py"))
    ''')
    assert len(out["reasons"]) == 1 and "cannot be compared with the shipped template" in out["reasons"][0], out


def test_a_missing_template_copy_keeps_its_reason(tmp_path):
    from orchard.stages import _template_reasons
    (r,) = _template_reasons(tmp_path, "reference_gate.py")
    assert "is not in this stage directory" in r


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "symlink_out", "oversize"])
def test_a_refused_bundle_results_file_is_named(shape):
    out = run_child(f'''
        bundle = run / "stages" / "8" / "bundle"
        build({shape!r}, bundle / "RESULTS.md")
        report(reasons=stages._bundle_benchmark_reasons(bundle, run))
    ''')
    assert len(out["reasons"]) == 1 and "bundle/RESULTS.md is not readable" in out["reasons"][0], out


# --------------------------------------------------------------------------------------------------------- the safeio helpers

@pytest.mark.parametrize("shape", SHAPES + ["nan"])
def test_checked_reads_separate_a_refused_file_from_a_missing_one(shape):
    out = run_child(f'''
        build({shape!r}, run / "f.json")
        j = safeio.read_json_checked(run / "f.json", max_bytes=64)
        t = safeio.read_text_checked(run / "f.json", max_bytes=64)
        m = safeio.read_json_checked(run / "none.json", max_bytes=64)
        report(j=[j.refused, j.missing, j.reason], t=[t.refused, t.missing], m=[m.refused, m.missing, m.reason], pj=j.problem("f.json"))
    ''')
    if shape == "nan":
        assert out["j"][0] and out["t"] == [False, False]           # the text is readable, only the JSON is refused
    else:
        assert out["j"][0] and out["t"][0] and out["j"][2]
    assert out["m"] == [False, True, None] and "f.json is not readable" in out["pj"]


def test_a_path_with_a_null_byte_is_a_refusal_not_an_exception(tmp_path):
    from orchard import safeio
    assert safeio.read_json_checked(str(tmp_path / "a\0b"), max_bytes=64).refused
    assert safeio.read_text_checked(str(tmp_path / "a\0b"), max_bytes=64).refused
    with pytest.raises(safeio.PathRefused):
        safeio.sha256_file(str(tmp_path / "a\0b"), max_bytes=64)


def test_sha256_file_matches_hashlib_and_stops_at_the_cap(tmp_path):
    import hashlib
    from orchard import safeio
    p = tmp_path / "f"
    p.write_bytes(b"abc" * 100000)
    assert safeio.sha256_file(p, max_bytes=1 << 20) == hashlib.sha256(b"abc" * 100000).hexdigest()
    with pytest.raises(safeio.TooLarge):
        safeio.sha256_file(p, max_bytes=1000)


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "oversize", "nan"])
def test_the_gate_reader_gives_none_and_a_named_reason_for_a_refused_file_and_none_without_one_for_a_missing_file(shape):
    out = run_child(f'''
        build({shape!r}, run / "r.json")
        reasons, none = [], []
        value = stages._read_checked(run / "r.json", "r.json", reasons)
        missing = stages._read_checked(run / "gone.json", "gone.json", none)
        report(value=value, reasons=reasons, missing=missing, none=none)
    ''')
    assert out["value"] is None and len(out["reasons"]) == 1 and "r.json is not readable" in out["reasons"][0]
    assert out["missing"] is None and out["none"] == []


# ------------------------------------------------------------------------------------- PR 345 review: scrub and stage end

@pytest.mark.parametrize("shape", ["oversize", "symlink_out"])
def test_the_bundle_scrub_names_a_file_it_cannot_search_and_reads_no_more_than_the_cap(shape):
    out = run_child(f'''
        from orchard import scrub
        scrub.SCRUB_MAX_BYTES = 64
        bundle = run / "bundle"
        build({shape!r}, bundle / "notes.txt")
        report(hits=scrub.scrub_bundle(bundle, hostname="nohost", home="/nohome"))
    ''')
    assert out["hits"] == [out["hits"][0]] and out["hits"][0].startswith("notes.txt: cannot be searched (")


def test_the_package_scrub_does_not_block_on_a_fifo():
    out = run_child('''
        from orchard import scrub
        build("fifo", run / "pkg" / "run.sh")
        report(hits=scrub.scrub_package(run / "pkg", hostname="nohost", home="/nohome"))
    ''')
    assert out["hits"] and "run.sh: cannot be searched (" in out["hits"][0]


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "oversize"])
def test_the_whole_bundle_gate_survives_a_hostile_bundle_file(shape):
    """The complete gate, not one helper: gate_bundle reads each bundle file, the build record and then runs the scrub."""
    out = run_child(f'''
        from orchard import scrub
        scrub.SCRUB_MAX_BYTES = 64
        sd = run / "stages" / "8"
        build({shape!r}, sd / "bundle" / "RESULTS.md")
        g = stages.gate_bundle(sd, run)
        report(ok=g.ok, reasons=list(g.reasons))
    ''')
    assert not out["ok"] and any("RESULTS.md" in r for r in out["reasons"]), out


def test_an_evidence_refusal_goes_through_the_failure_path_so_escalation_and_pause_apply():
    """PR 345 review: the refusal must reach _end_stage_body with status "fail" and the reason, not write a bare stage_end."""
    from types import SimpleNamespace
    from orchard import safeio
    from orchard.supervisor import Supervisor
    calls = []

    def body(spec, stage_dir, status, reasons, gate, escalated):
        calls.append((status, reasons, escalated))
        if status == "pass":
            raise safeio.TooLarge("the file is larger than 5 bytes")
        return "escalate"

    sup = SimpleNamespace(_end_stage_body=body)
    spec = SimpleNamespace(number=3)
    assert Supervisor._end_stage(sup, spec, Path("."), "pass", [], None, False) == "escalate"
    assert [c[0] for c in calls] == ["pass", "fail"] and "larger than 5 bytes" in calls[1][1][0] and calls[1][2] is False


@pytest.mark.parametrize("shape", ["oversize", "symlink_out"])
def test_a_failed_marker_that_is_refused_is_recorded_with_the_reason_before_it_is_moved_aside(shape):
    """Supervisor._run_stage hashes the failed marker for its ledger entry; a refusal must not stop the resume."""
    out = run_child(f'''
        from types import SimpleNamespace
        from orchard import supervisor
        stages.EVIDENCE_HASH_MAX = 64
        build({shape!r}, run / "stages" / "2" / "test-result.json")
        sup = SimpleNamespace(run_dir=run)
        report(rec=supervisor.Supervisor._moved_aside_record(sup, run / "stages" / "2" / "test-result.json"))
    ''')
    assert out["rec"]["path"] == "stages/2/test-result.json" and out["rec"]["unreadable"], out


def test_a_failed_marker_that_can_be_read_is_recorded_with_its_hash(tmp_path):
    from types import SimpleNamespace
    from orchard.supervisor import Supervisor
    (tmp_path / "stages" / "2").mkdir(parents=True)
    (tmp_path / "stages" / "2" / "test-result.json").write_text("{}")
    rec = Supervisor._moved_aside_record(SimpleNamespace(run_dir=tmp_path), tmp_path / "stages" / "2" / "test-result.json")
    assert rec["path"] == "stages/2/test-result.json" and len(rec["sha256"]) == 64


# ------------------------------------------------ second Copilot review of PR 345: a refusal followed by a write to the same path

@pytest.mark.parametrize("shape", ["fifo", "symlink_out", "dev_zero"])
def test_a_write_of_a_run_file_replaces_a_fifo_or_link_and_never_opens_it(shape):
    out = run_child(f'''
        from orchard import supervisor
        target = run / "stages" / "2" / "hw_tests.json"
        build({shape!r}, target)
        supervisor._put(target, '{{"ok": true}}')
        report(regular=target.is_file() and not target.is_symlink(), text=target.read_text(), outside=outside.read_text())
    ''')
    assert out["regular"] and out["text"] == '{"ok": true}' and out["outside"] == "{}", out


@pytest.mark.parametrize("shape", ["fifo", "symlink_out", "dev_zero"])
def test_the_hardware_output_file_is_opened_fresh_not_through_what_stands_there(shape):
    out = run_child(f'''
        from orchard import supervisor
        target = run / "stages" / "2" / "out.txt"
        build({shape!r}, target)
        with supervisor._open_fresh(target) as f:
            f.write(b"output")
        report(regular=target.is_file() and not target.is_symlink(), text=target.read_text(), outside=outside.read_text())
    ''')
    assert out["regular"] and out["text"] == "output" and out["outside"] == "{}", out


def test_a_fifo_at_hw_tests_json_does_not_block_the_drafted_list_going_back():
    out = run_child('''
        from types import SimpleNamespace
        from orchard import supervisor
        S = supervisor.Supervisor
        entries = []
        ledger = SimpleNamespace(read=lambda: [{"stage": 4, "data": {"decision": "stage 4 configs drafted", "tests": {"tests": [{"chips": 1}]}}}],
                                 append=lambda *a, **k: entries.append((a, k)))
        stub = SimpleNamespace(ledger=ledger, lab=None, _chip_cap=lambda spec: 2)
        stub._run_json = lambda *a: S._run_json(stub, *a)
        sd = run / "stages" / "4"
        build("fifo", sd / "hw_tests.json")
        S._keep_drafted_tests(stub, SimpleNamespace(number=4), sd)
        report(regular=(sd / "hw_tests.json").is_file(), text=(sd / "hw_tests.json").read_text(), refused=[a[1].get("decision") for a in entries])
    ''')
    assert out["regular"] and '"chips": 1' in out["text"] and "a run file was refused and not used" in out["refused"], out


@pytest.mark.parametrize("shape", ["fifo", "symlink_out", "dev_zero"])
def test_the_lab_cache_audit_does_not_block_on_a_fifo_so_the_lease_is_released(shape):
    """Both lab hardware-test paths call _audit_lab_caches before they release their leases; a block here would hold the hardware."""
    out = run_child(f'''
        from types import SimpleNamespace
        from orchard import supervisor
        S = supervisor.Supervisor
        stub = SimpleNamespace(lab=SimpleNamespace(cache_audit=lambda caches: {{"c": {{"files": 1}}}}),
                               ledger=SimpleNamespace(append=lambda *a, **k: None))
        stub._run_json = lambda *a: S._run_json(stub, *a)
        sd = run / "stages" / "4"
        build({shape!r}, sd / "evidence" / "lab-caches.json")
        S._audit_lab_caches(stub, 4, sd, ["/c"])
        p = sd / "evidence" / "lab-caches.json"
        report(regular=p.is_file() and not p.is_symlink(), text=p.read_text(), outside=outside.read_text())
    ''')
    assert out["regular"] and '"files": 1' in out["text"] and out["outside"] == "{}", out


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "oversize", "nan"])
def test_the_hardware_test_plan_readers_are_bounded(shape):
    out = run_child(f'''
        from orchard import hwtests
        hwtests.JSON_MAX = 64
        sd = run / "stages" / "4"
        build({shape!r}, sd / "hw_tests.json")
        build({shape!r}, sd / "handoff.json")
        tests, problems = hwtests.read_plan(sd, required=None, max_chips=4, budget_s=100, home="/h")
        build({shape!r}, sd / "tests" / "plan.json")
        report(tests=tests, problems=problems, plan=hwtests.load_plan(sd))
    ''')
    assert out["tests"] == [] and out["plan"] is None
    assert len(out["problems"]) == 1 and "hw_tests.json is not readable" in out["problems"][0] and "handoff.json is not readable" in out["problems"][0], out


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "oversize"])
def test_a_refused_test_record_is_a_reason_in_unrecorded_and_a_marked_entry_in_the_summary(shape):
    out = run_child(f'''
        from orchard import hwtests
        hwtests.JSON_MAX = 64
        stages.EVIDENCE_HASH_MAX = 64
        sd = run / "stages" / "4"
        build({shape!r}, sd / "tests" / "2" / "test-result.json")
        reasons = hwtests.unrecorded([], sd, run, 4) if {shape!r} == "oversize" else []
        summary = hwtests.write_summary(sd, [hwtests.HwTest(2, "serve_and_compare.py", 10.0, "/c")])
        report(reasons=reasons, summary=json.loads(summary.read_text()))
    ''')
    entry = out["summary"]["tests"][0]
    assert entry["missing"] is True and entry["unreadable"], out
    if shape == "oversize":
        assert len(out["reasons"]) == 1 and "cannot be read" in out["reasons"][0], out


@pytest.mark.parametrize("shape", ["fifo", "symlink_out"])
def test_the_nearest_model_check_names_a_refused_swap_config(shape):
    out = run_child(f'''
        from orchard import swap_draft
        cfg = run / "stages" / "2" / "swap_config.json"
        build({shape!r}, cfg)
        report(problem=swap_draft.nearest_problem(run, cfg), missing=swap_draft.nearest_problem(run, run / "none.json"))
    ''')
    assert out["problem"] and "swap_config.json is not readable" in out["problem"] and out["missing"] is None, out


@pytest.mark.parametrize("shape", ["fifo", "dev_zero", "oversize"])
def test_a_refused_delta_does_not_block_the_nearest_model_check_or_the_draft(shape):
    out = run_child(f'''
        from orchard import swap_draft
        swap_draft.JSON_MAX = 64
        (run / "stages" / "2").mkdir(parents=True)
        cfg = run / "stages" / "2" / "swap_config.json"
        cfg.write_text(json.dumps({{"nearest_model_id": "a/b"}}))
        build({shape!r}, run / "stages" / "0" / "delta.json")
        facts = swap_draft._facts(run, {{}})
        report(problem=swap_draft.nearest_problem(run, cfg), facts_problems=facts[3])
    ''')
    assert out["problem"] is None and out["facts_problems"], out


def test_an_evidence_file_the_agent_step_cannot_hash_is_named_in_the_ledger_and_the_step_goes_on():
    out = run_child('''
        from types import SimpleNamespace
        from orchard.agent import AgentStep
        stages.EVIDENCE_HASH_MAX = 64
        big = run / "evidence" / "big.txt"
        big.parent.mkdir(parents=True)
        big.write_text("x" * 200)
        ok = run / "evidence" / "ok.txt"
        ok.write_text("ok")
        entries = []
        stub = SimpleNamespace(_snapshot=lambda: {str(big): (200, 1), str(ok): (2, 1)}, _seen={}, run_dir=run, stage=2, phase="work",
                               ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))), _event=lambda *a, **k: None)
        AgentStep._record_new_evidence(stub)
        report(kinds=[k.get("decision") or k.get("what") for a, k in entries])
    ''')
    assert "an evidence file was refused and not recorded" in out["kinds"] and "evidence file" in out["kinds"], out


def test_a_refusal_inside_a_stage_body_is_a_failed_attempt_with_the_reason():
    from types import SimpleNamespace
    from orchard import safeio
    from orchard.supervisor import Supervisor

    def body(*a):
        raise safeio.TooLarge("the file is larger than 5 bytes")

    stub = SimpleNamespace(_stage_body=body)
    status, reasons, gate = Supervisor._body_or_failure(stub, SimpleNamespace(number=4), Path("."), False, False)
    assert status == "fail" and gate is None and "larger than 5 bytes" in reasons[0] and "stage 4" in reasons[0]


@pytest.mark.parametrize("shape", ["fifo", "symlink_out"])
def test_the_test_record_writer_replaces_a_fifo_or_link_at_the_record_and_at_its_temporary_name(shape):
    out = run_child(f'''
        from orchard import hwtests
        sd = run / "stages" / "4"
        rec = hwtests.hw_record_path(sd, 2)
        build({shape!r}, rec)
        build({shape!r}, rec.with_name(rec.name + ".tmp"))
        hwtests.write_record(sd, 2, {{"returncode": 0}})
        report(regular=rec.is_file() and not rec.is_symlink(), text=rec.read_text(), outside=outside.read_text())
    ''')
    assert out["regular"] and '"returncode": 0' in out["text"] and out["outside"] == "{}", out


@pytest.mark.parametrize("shape", ["fifo", "symlink_out", "dev_zero"])
def test_the_port_check_files_are_written_over_a_fifo_or_link_and_a_refused_hw_test_does_not_block(shape):
    """PR 347 added the full-port check: the supervisor copies the template, writes its config and rewrites hw_test.json in the stage directory."""
    out = run_child(f'''
        from types import SimpleNamespace
        from orchard import full_port, supervisor
        S = supervisor.Supervisor
        sd = run / "stages" / "2"
        sd.mkdir(parents=True)
        for name in (full_port.TEMPLATE_NAME, full_port.CONFIG_NAME, "hw_test.json"):
            build({shape!r}, sd / name)
        entries = []
        stub = SimpleNamespace(inputs={{"model": str(run)}}, paths=SimpleNamespace(orchard_dir=str(Path(supervisor.__file__).resolve().parent.parent)),
                               ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
        stub._run_json = lambda *a: S._run_json(stub, *a)
        problems = S._pin_full_port(stub, SimpleNamespace(number=2), sd, command=True)
        regular = [(sd / n).is_file() and not (sd / n).is_symlink() for n in (full_port.TEMPLATE_NAME, full_port.CONFIG_NAME, "hw_test.json")]
        report(problems=problems, regular=regular, outside=outside.read_text())
    ''')
    assert out["problems"] == [] and all(out["regular"]) and out["outside"] == "{}", out


# ------------------------------------------------ third Copilot review of PR 345: a refusal while the test's chips are held

LISTED_TEST_SETUP = '''
    from types import SimpleNamespace
    from orchard import supervisor
    S = supervisor.Supervisor
    stages.EVIDENCE_HASH_MAX = 64 if MODE == "ok" else 4096
    entries = []
    sd = run / "stages" / "4"
    spawns = []

    def spawn(*a, **k):
        spawns.append(1)
        out = a[5]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("x" * 200)
        if MODE == "raise":
            raise safeio.TooLarge("the file is larger than 64 bytes")
        return {"command": "c", "returncode": 1, "timed_out": False, "seconds": 1.0, "chips": ["a"], "output": {}}

    stub = SimpleNamespace(run_dir=run, _spawn=spawn, _reset_test_chips=lambda *a: None,
                           ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    stub._moved_aside_record = lambda p: S._moved_aside_record(stub, p)
    stub._run_listed_test_body = lambda *a: S._run_listed_test_body(stub, *a)
    test = SimpleNamespace(chips=4, boards=2, command=lambda n: "cmd", deadline_s=10.0, cache="/c")
    lease = SimpleNamespace(chips=["a", "b", "c", "d"], dev_indices=[0, 1, 2, 3], lease_id="L")
'''


def test_an_oversize_first_try_output_does_not_stop_the_retry_or_the_cleanup_that_follows():
    out = run_child('MODE = "ok"\n' + textwrap.dedent(LISTED_TEST_SETUP) + textwrap.dedent('''
    S._run_listed_test(stub, SimpleNamespace(number=4, budget_s=100.0), sd, test, [lease])
    first = [k["first_output"] for a, k in entries if k.get("first_output")]
    report(spawns=len(spawns), first=first, record=(sd / "tests" / "4" / "test-result.json").is_file())
    '''))
    assert out["spawns"] == 2 and out["record"] and out["first"][0].get("unreadable"), out


def test_a_refusal_in_a_listed_test_is_a_record_and_returns_so_the_lease_is_released_and_the_coder_restored():
    """_run_listed_test returns normally; its callers then run the sweep, _give_back_test_lease and _restore_coder."""
    out = run_child('MODE = "raise"\n' + textwrap.dedent(LISTED_TEST_SETUP) + textwrap.dedent('''
    S._run_listed_test(stub, SimpleNamespace(number=4, budget_s=100.0), sd, test, [lease])
    rec = json.loads((sd / "tests" / "4" / "test-result.json").read_text())
    ev = [k for a, k in entries if a[0] == "evidence"]
    report(rec=rec, ev=ev)
    '''))
    assert out["rec"]["returncode"] is None and "run file was refused" in out["rec"]["not_run"]
    assert len(out["ev"]) == 1 and out["ev"][0]["timed_out"] is False and out["ev"][0]["sha256"], out


@pytest.mark.parametrize("shape", ["symlink_out", "fifo", "oversize", "nan"])
def test_a_refused_bundle_manifest_is_a_named_problem_not_an_accepted_answer(shape):
    out = run_child(f'''
        from orchard import swap_draft
        swap_draft.JSON_MAX = 64
        (run / "stages" / "0").mkdir(parents=True)
        (run / "stages" / "0" / "delta.json").write_text(json.dumps({{"nearest_model": "right/model"}}))
        bundle = run / "bundle"
        bundle.mkdir()
        if {shape!r} == "symlink_out":
            outside.write_text(json.dumps({{"weights": {{"repo_id": "wrong/model"}}}}))
        build({shape!r}, bundle / "tt_kernel_manifest.json")
        cfg = run / "swap_config.json"
        cfg.write_text(json.dumps({{"nearest_model_id": "right/model", "bundle_dir": "bundle"}}))
        report(problem=swap_draft.nearest_problem(run, cfg))
    ''')
    assert out["problem"] and "tt_kernel_manifest.json is not readable" in out["problem"], out


def test_a_manifest_without_a_repo_is_a_problem_and_a_missing_manifest_is_still_no_problem(tmp_path):
    import json as j
    from orchard import swap_draft
    (tmp_path / "stages" / "0").mkdir(parents=True)
    (tmp_path / "stages" / "0" / "delta.json").write_text(j.dumps({"nearest_model": "right/model"}))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    cfg = tmp_path / "swap_config.json"
    cfg.write_text(j.dumps({"nearest_model_id": "right/model", "bundle_dir": str(bundle)}))
    assert swap_draft.nearest_problem(tmp_path, cfg) is None                       # no manifest: the template refuses the missing bundle itself
    (bundle / "tt_kernel_manifest.json").write_text("{}")
    assert "has no weights.repo_id" in swap_draft.nearest_problem(tmp_path, cfg)
    (bundle / "tt_kernel_manifest.json").write_text(j.dumps({"weights": {"repo_id": "wrong/model"}}))
    assert "serves wrong/model" in swap_draft.nearest_problem(tmp_path, cfg)


# ------------------------------------------------ fourth Copilot review of PR 345

@pytest.mark.parametrize("fname", ["graft-decision.json", "sampling-decision.json"])
def test_stage_4_refuses_a_stage_2_decision_that_is_a_link_instead_of_reading_it_as_no_decision(tmp_path, fname):
    """A link to the same valid graft decision made graft_chosen() return False, so an ungrafted bundle passed."""
    from orchard.stages import gate_mesh_swap
    from test_stage4_graft_gate import bundle_config, gate
    sd, run = gate(tmp_path)
    bundle_config(run, 2, grafted=False)
    real = run / "stages" / "2" / "real-decision.json"
    write(run, "stages/2/real-decision.json", {"choice": "graft" if fname.startswith("graft") else "drafter-off", "reason": "r"})
    (run / "stages" / "2" / fname).symlink_to(real.name)
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any(f"stages/2/{fname} is not readable" in r for r in g.reasons), g.reasons


def test_stage_4_still_passes_with_no_stage_2_decision_files(tmp_path):
    from orchard.stages import gate_mesh_swap
    from test_stage4_graft_gate import bundle_config, gate
    sd, run = gate(tmp_path)
    bundle_config(run, 2, grafted=False)
    assert gate_mesh_swap(sd, run, required=(2,)).ok


def test_a_decision_recorded_with_the_old_missing_genconfig_wording_still_matches(tmp_path):
    """A resumed stage 2 whose genconfig-license.json is missing recorded this exact sentence; the recomputation must give the same one."""
    from orchard import sampling_choice
    need = sampling_choice.need(tmp_path, "auto")
    assert need["evidence"] == ["no evidence: stage 0's evidence/genconfig-license.json is missing or unreadable"]
