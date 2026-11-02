# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The optional recorded-state file of the canning step (orchard/can.py `write_state`, `check_state`, `--state-dir`).

`can gate`, `can boot` and `can stage` write `<state dir>/<name>.can-state.json` only when `--state-dir` is given. The Cans view
(orchard/cans_view.py) reads that file. Stand-ins replace the gate and the boot check, so no chip is used."""
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from orchard import can


def bundle(tmp_path, text="abc  run.sh\n"):
    b = tmp_path / "bundle"
    b.mkdir(exist_ok=True)
    (b / can.SUMS_FILE).write_text(text)
    return b


def digest(b):
    return hashlib.sha256((b / can.SUMS_FILE).read_bytes()).hexdigest()


def read(state_dir, name="toy"):
    return json.loads((state_dir / f"{name}.can-state.json").read_text())


def test_a_gate_record_holds_the_bundle_digest_and_at_most_ten_problems(tmp_path):
    b = bundle(tmp_path)
    can.write_state(tmp_path / "st", "toy", b, gate=can.gate_record([f"p{i}" for i in range(25)]))
    g = read(tmp_path / "st")["gate"]
    assert g["passed"] is False and g["problem_count"] == 25 and len(g["problems"]) == 10
    assert g["bundle_sums_sha256"] == digest(b)


def test_a_gate_with_no_problem_is_recorded_as_passed(tmp_path):
    can.write_state(tmp_path / "st", "toy", bundle(tmp_path), gate=can.gate_record([]))
    g = read(tmp_path / "st")["gate"]
    assert g["passed"] is True and g["problem_count"] == 0 and g["problems"] == []


def test_a_later_gate_keeps_the_boot_record_and_replaces_the_gate_record(tmp_path):
    b = bundle(tmp_path)
    st = tmp_path / "st"
    can.write_state(st, "toy", b, boot=can.boot_record({"passed": True, "checks": {}, "rate_timings": {"decode_tok_s": 31.5}, "ready_s": 12.0}, "empty"))
    can.write_state(st, "toy", b, gate=can.gate_record(["x"]))
    can.write_state(st, "toy", b, gate=can.gate_record([]))
    s = read(st)
    assert s["gate"]["passed"] is True and s["boot"]["decode_tok_s"] == 31.5 and s["boot"]["cache_state"] == "empty"


def test_a_boot_record_takes_the_decode_rate_from_the_rate_timings_then_the_timings():
    r = can.boot_record({"passed": True, "checks": {}, "rate_timings": {"decode_tok_s": 20.0}, "timings": {"decode_tok_s": 9.0}, "ready_s": 5}, None)
    assert r["decode_tok_s"] == 20.0 and r["ready_s"] == 5 and r["cache_state"] is None
    r = can.boot_record({"passed": True, "checks": {}, "timings": {"decode_tok_s": 9.0}}, "warm")
    assert r["decode_tok_s"] == 9.0 and r["ready_s"] is None and r["cache_state"] == "warm"


def test_a_boot_record_lists_the_checks_that_failed_and_drops_a_rate_that_is_not_a_finite_number():
    ev = {"passed": False, "checks": {"health": {"ok": True}, "chat": {"ok": False}, "stream": {"ok": False}},
          "rate_timings": {"decode_tok_s": float("nan")}}
    r = can.boot_record(ev, None)
    assert r["passed"] is False and r["failed"] == ["chat", "stream"] and r["decode_tok_s"] is None


def test_a_cache_state_other_than_empty_or_warm_is_refused():
    with pytest.raises(can.CanError):
        can.boot_record({"passed": True, "checks": {}}, "lukewarm")


def test_a_bundle_with_no_sums_file_records_a_null_digest(tmp_path):
    b = tmp_path / "b"
    b.mkdir()
    can.write_state(tmp_path / "st", "toy", b, gate=can.gate_record([]))
    assert read(tmp_path / "st")["gate"]["bundle_sums_sha256"] is None


def test_the_state_name_must_be_a_plain_name(tmp_path):
    with pytest.raises(can.CanError):
        can.write_state(tmp_path / "st", "../evil", bundle(tmp_path), gate=can.gate_record([]))
    assert not (tmp_path / "evil.can-state.json").exists()


def test_check_state_accepts_what_write_state_wrote_and_refuses_each_bad_field(tmp_path):
    can.write_state(tmp_path / "st", "toy", bundle(tmp_path), gate=can.gate_record(["x"]),
                    boot=can.boot_record({"passed": True, "checks": {}}, "warm"))
    good = read(tmp_path / "st")
    assert can.check_state(good) == good

    def bad(mutate):
        raw = json.loads(json.dumps(good))
        mutate(raw)
        with pytest.raises(can.CanError):
            can.check_state(raw)
    bad(lambda r: r.update(schema=2))
    bad(lambda r: r.update(extra=1))
    bad(lambda r: r["gate"].update(passed="yes"))
    bad(lambda r: r["gate"].update(problem_count=-1))
    bad(lambda r: r["gate"].update(problems=["x"] * 11))
    bad(lambda r: r["gate"].update(problems=["x" * 301]))
    bad(lambda r: r["gate"].update(bundle_sums_sha256="nothex"))
    bad(lambda r: r["gate"].update(at="yesterday"))
    bad(lambda r: r["boot"].update(decode_tok_s=float("inf")))
    bad(lambda r: r["boot"].update(decode_tok_s=True))
    bad(lambda r: r["boot"].update(cache_state="lukewarm"))
    bad(lambda r: r["boot"].update(failed="chat"))
    bad(lambda r: r.update(name="../x"))
    bad(lambda r: r.update(gate=[]))


# ---- the command line writes the file only when --state-dir is given --------------------------------------------------------

def args(tmp_path, action, **kw):
    base = dict(action=action, spec="spec.toml", bundle=str(tmp_path / "bundle"), out=None, work=str(tmp_path / "work"), base=None,
                namespace=None, run_dir=None, hf_home=None, health_timeout=1.0, port=None, state_dir=None, cache_state=None)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    bundle(tmp_path)
    monkeypatch.setattr(can, "load_spec", lambda path, base=None: SimpleNamespace(name="toy", kind="vllm", repo="o/r", revision="0" * 40))
    monkeypatch.setattr(can, "gate", lambda b, spec, **kw: ["scrub: a thing"])
    monkeypatch.setattr(can, "boot_check", lambda b, w, spec, **kw: {"passed": True, "checks": {"health": {"ok": True}},
                                                                    "rate_timings": {"decode_tok_s": 40.0}, "ready_s": 3.0})


def test_gate_with_a_state_dir_records_the_result_and_without_one_writes_nothing(tmp_path, fake):
    assert can.main(args(tmp_path, "gate"), out=io.StringIO()) == 1
    assert not list(tmp_path.glob("**/*.can-state.json"))
    assert can.main(args(tmp_path, "gate", state_dir=str(tmp_path / "st")), out=io.StringIO()) == 1
    g = read(tmp_path / "st")["gate"]
    assert g["passed"] is False and g["problems"] == ["scrub: a thing"]


def test_boot_with_a_state_dir_records_the_decode_rate_and_the_cache_state(tmp_path, fake):
    assert can.main(args(tmp_path, "boot", state_dir=str(tmp_path / "st"), cache_state="empty"), out=io.StringIO()) == 0
    b = read(tmp_path / "st")["boot"]
    assert b["passed"] is True and b["decode_tok_s"] == 40.0 and b["cache_state"] == "empty" and b["ready_s"] == 3.0


def test_boot_without_a_state_dir_writes_no_state_file(tmp_path, fake):
    assert can.main(args(tmp_path, "boot"), out=io.StringIO()) == 0
    assert not list(tmp_path.glob("**/*.can-state.json"))


def test_a_bad_cache_state_stops_the_boot_before_it_starts(tmp_path, fake, monkeypatch):
    called = []
    monkeypatch.setattr(can, "boot_check", lambda *a, **k: called.append(1))
    out = io.StringIO()
    assert can.main(args(tmp_path, "boot", state_dir=str(tmp_path / "st"), cache_state="lukewarm"), out=out) == 2
    assert called == [] and "cache" in out.getvalue()


def test_stage_with_a_state_dir_records_a_passed_gate(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(can, "stage", lambda spec, out, **kw: {"name": "toy", "kind": "vllm", "weights": {}, "files": []})
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / can.SUMS_FILE).write_text("")
    assert can.main(args(tmp_path, "stage", out=str(tmp_path / "out"), state_dir=str(tmp_path / "st")), out=io.StringIO()) == 0
    g = read(tmp_path / "st")["gate"]
    assert g["passed"] is True and g["problem_count"] == 0
