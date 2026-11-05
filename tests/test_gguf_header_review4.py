# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 441, independent review after round 3 (issue 440, a slice of 439): four findings, each reproduced by a test here before the fix.

1. MEDIUM: `quantization.by_kind` had one entry per distinct tensor kind and nothing capped the evidence file: a header of 100,000 tensors
   with 600-character names made 136 MB of evidence for one file, stage 0 reached about 600 MB, and the operator bundle's 4 MiB reader then
   dropped the whole GGUF section (every file's table, the harmless ones too) without saying why.
2. LOW: an unreadable chat_template.jinja and a list-form `chat_template` were both described as "the repository has none".
3. LOW: the string-array loop took about 6 s for the worst legal header (8 million string elements), about 50 s for 8 files.
4. LOW: a link in the snapshot was followed wherever it led.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_gguf_header import (F32, Q4_K, T_ARR, T_STR, T_U32, TEMPLATE, TEMPLATE_TEXT, gguf, hybrid_config, hybrid_kvs, hybrid_tensors, kv, module,
                              s, write)


def names_header(count: int, width: int, *, types=(Q4_K, F32)) -> bytes:
    """A header with `count` tensors, each its own kind (`x<i>.` plus `width` letters) and alternate types."""
    pad = "a" * width
    return gguf(hybrid_kvs(), [(f"x{i}.{pad}", [4], types[i % len(types)], 0) for i in range(count)], data=False)


# ---- 1. the evidence stays small and readable ------------------------------------------------------

def test_by_kind_is_capped_and_every_tensor_is_still_counted(tmp_path):
    m = module()
    p = write(tmp_path, names_header(3000, 300), "k.gguf")
    e = m.gguf_file_evidence(p, None, None)
    q = e["quantization"]
    assert e["status"] == "ok" and len(q["by_kind"]) <= m.GGUF_MAX_BY_KIND
    assert all(len(k) <= m.GGUF_MAX_KIND_CHARS for k in q["by_kind"])
    kept = sum(n for c in q["by_kind"].values() for n in c.values())
    other = sum(q["by_kind_other"].values())
    assert kept + other == e["tensor_count"] == 3000
    cut = q["by_kind_cut"]
    assert cut["kinds_seen"] == 3000 and cut["kinds_shown"] == len(q["by_kind"]) and cut["tracking_stopped"] is False
    assert cut["tensors_in_other"] == other
    assert len(json.dumps(e)) < 300_000


def test_the_kinds_kept_are_the_most_numerous_ones(tmp_path):
    m = module()
    tensors = [(f"blk.{i}.big.weight", [4], Q4_K, 0) for i in range(10)] + [(f"x{i}.once", [4], F32, 0) for i in range(200)]
    p = write(tmp_path, gguf(hybrid_kvs(), tensors, data=False), "k.gguf")
    q = m.gguf_file_evidence(p, None, None)["quantization"]
    assert "blk.N.big.weight" in q["by_kind"] and q["by_kind"]["blk.N.big.weight"] == {"Q4_K": 10}


def test_tracking_stops_after_a_bound_so_memory_stays_bounded(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "GGUF_MAX_KINDS_TRACKED", 100)
    p = write(tmp_path, names_header(500, 10), "k.gguf")
    q = m.gguf_file_evidence(p, None, None)["quantization"]
    assert q["by_kind_cut"]["tracking_stopped"] is True and q["by_kind_cut"]["kinds_seen"] <= 100
    assert sum(n for c in q["by_kind"].values() for n in c.values()) + sum(q["by_kind_other"].values()) == 500


def test_full_precision_kinds_are_capped_and_the_cut_is_recorded(tmp_path):
    m = module()
    p = write(tmp_path, names_header(600, 5, types=(F32,)), "k.gguf")
    q = m.gguf_file_evidence(p, None, None)["quantization"]
    assert len(q["full_precision"]) == m.GGUF_MAX_FULL_PRECISION
    assert q["full_precision_cut"] == {"kinds": 600, "shown": m.GGUF_MAX_FULL_PRECISION}


def evidence_size_in_child(path: Path) -> dict:
    """gguf_evidence over one file in a child with RLIMIT_AS 2 GiB: the serialized size, the peak size and the time taken."""
    code = r"""
import importlib.util, json, resource, sys, time
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
spec = importlib.util.spec_from_file_location("t", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.monotonic()
out = m.gguf_evidence(sys.argv[2], [{"file": sys.argv[3]}], None, None)
peak = None
for line in open("/proc/self/status"):
    if line.startswith("VmHWM:"):
        peak = int(line.split()[1])
print(json.dumps({"bytes": len(m.json_bytes(out)), "peak_kb": peak, "seconds": time.monotonic() - t0, "status": out["files"][0]["status"],
                  "reason": out["files"][0].get("reason"), "cut": out["files"][0].get("quantization", {}).get("by_kind_cut")}))
"""
    r = subprocess.run([sys.executable, "-c", code, str(TEMPLATE), str(path.parent), path.name], capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def test_the_reviewers_header_of_100000_tensors_with_600_letter_names_makes_small_evidence(tmp_path):
    m = module()
    p = tmp_path / "hostile.gguf"
    with open(p, "wb") as f:
        pad = "a" * 600
        head_kvs = hybrid_kvs()
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 100_000, len(head_kvs)) + b"".join(head_kvs))
        for i in range(100_000):
            f.write(s(f"x{i}.{pad}") + struct.pack("<I", 1) + struct.pack("<Q", 4) + struct.pack("<IQ", F32, 0))
    out = evidence_size_in_child(p)
    assert out["status"] == "ok" and out["bytes"] < m.GGUF_EVIDENCE_MAX_BYTES // 8, out
    assert out["cut"]["kinds_shown"] <= m.GGUF_MAX_BY_KIND
    if out["peak_kb"] is not None:
        assert out["peak_kb"] < 450_000, out["peak_kb"]               # it was about 600 MB before the cap


def test_the_whole_file_never_exceeds_its_cap_and_drops_detail_in_a_defined_order(tmp_path, monkeypatch):
    m = module()
    write(tmp_path, names_header(300, 60), "a.gguf")
    write(tmp_path, names_header(300, 60), "b.gguf")
    ignored = [{"file": "a.gguf"}, {"file": "b.gguf"}]
    full = m.gguf_evidence(tmp_path, ignored, None, None)
    size = len(m.json_bytes(full))
    monkeypatch.setattr(m, "GGUF_EVIDENCE_MAX_BYTES", size - 1500)
    cut = m.gguf_evidence(tmp_path, ignored, None, None)
    assert len(m.json_bytes(cut)) <= size - 1500
    first = cut["files"][0]
    assert first["status"] == "ok" and first["agreement"] == full["files"][0]["agreement"]          # the table survives
    assert cut["detail_dropped"] and cut["detail_dropped"][0]["what"] == "by_kind"                  # the first thing to go
    assert "by_kind" not in first["quantization"] and first["quantization"]["by_type"] == full["files"][0]["quantization"]["by_type"]


def test_a_cap_too_small_for_any_detail_still_writes_a_readable_file_that_says_so(tmp_path, monkeypatch):
    m = module()
    write(tmp_path, names_header(50, 20), "a.gguf")
    monkeypatch.setattr(m, "GGUF_EVIDENCE_MAX_BYTES", 1200)
    out = m.gguf_evidence(tmp_path, [{"file": "a.gguf"}], None, None)
    assert len(m.json_bytes(out)) <= 1200 and out["files"][0]["file"] == "a.gguf" and out["detail_dropped"]
    monkeypatch.setattr(m, "GGUF_EVIDENCE_MAX_BYTES", 50)           # a cap nothing fits under: still a valid, empty-handed file that says so
    tiny = m.gguf_evidence(tmp_path, [{"file": "a.gguf"}], None, None)
    assert tiny["files"] == [] and tiny["files_cut"] == 1 and tiny["detail_dropped"]


# ---- 1b. the bundle says why it could not read the evidence, per file ------------------------------

def bundle_with(tmp_path, content: bytes):
    from test_operator_bundle_template import build, bundle_of, make_run, section
    run = make_run(tmp_path)
    p = run / "stages/0/evidence/gguf-header.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    r = build(run)
    assert r.returncode == 0, r.stdout + r.stderr
    risks = (bundle_of(run) / "RISKS.md").read_text()
    return section(risks, "## What the GGUF files show") if "## What the GGUF files show\n" in risks else risks


def test_an_evidence_file_over_the_read_cap_is_named_with_its_reason_and_its_size(tmp_path):
    sec = bundle_with(tmp_path, b'{"schema": 1, "files": [], "pad": "' + b"x" * (5 << 20) + b'"}')
    assert "could not be read" in sec and "stages/0/evidence/gguf-header.json" in sec
    assert "larger than" in sec or "too large" in sec or "cap" in sec


def test_an_evidence_file_that_is_not_json_is_named_with_its_reason(tmp_path):
    sec = bundle_with(tmp_path, b"not json at all")
    assert "could not be read" in sec and "not valid JSON" in sec


def test_one_bad_record_does_not_hide_the_others(tmp_path):
    good = {"file": "g.gguf", "size": 5, "status": "ok", "role": "language model", "architecture": "x", "tensor_count": 3,
            "agreement": [{"item": "block count", "gguf": 4, "config": 5, "verdict": "differ"}], "quantization": {}}
    junk = {"file": 7, "status": "ok"}
    sec = bundle_with(tmp_path, json.dumps({"schema": 1, "files": [junk, good, "text", None]}).encode())
    assert "### g.gguf" in sec and "block count: GGUF 4; config 5." in sec


# ---- 2. the chat-template note says which of the three situations it is ----------------------------

def test_the_template_state_names_absent_list_form_and_unreadable(tmp_path):
    m = module()
    assert m.repo_chat_template_state(tmp_path) == (None, "absent")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": [{"name": "default", "template": "x"}]}))
    assert m.repo_chat_template_state(tmp_path) == (None, "list-form")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "abc"}))
    assert m.repo_chat_template_state(tmp_path) == ("abc", None)
    jinja = tmp_path / "chat_template.jinja"
    jinja.write_bytes(b"\xff\xfe" * 10)
    text, reason = m.repo_chat_template_state(tmp_path)
    assert (text, reason) == (None, "unreadable") or text is not None            # undecodable bytes may be read with replacement
    jinja.write_text("")
    assert m.repo_chat_template_state(tmp_path) == ("", None)
    assert m.repo_chat_template(tmp_path) == ""                                  # the old function still gives the text


def test_an_oversize_template_file_is_unreadable_and_says_so(tmp_path):
    m = module()
    (tmp_path / "chat_template.jinja").write_text("x" * (m.REPORT_MAX_BYTES + 1))
    assert m.repo_chat_template_state(tmp_path) == (None, "unreadable")
    (tmp_path / "chat_template.jinja").unlink()
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "y" * (m.REPORT_MAX_BYTES + 1)}))
    assert m.repo_chat_template_state(tmp_path) == (None, "unreadable")


@pytest.mark.parametrize("reason,words", [("unreadable", "cannot be read"), ("list-form", "named chat templates"), ("absent", "has no chat_template.jinja")])
def test_the_row_note_follows_the_reason(tmp_path, reason, words):
    m = module()
    p = write(tmp_path, gguf(hybrid_kvs(**{"tokenizer.chat_template": (T_STR, TEMPLATE_TEXT)}), hybrid_tensors()), "t.gguf")
    e = m.gguf_file_evidence(p, hybrid_config(), None, template_reason=reason)
    row = next(r for r in e["agreement"] if r["item"] == "chat template")
    assert row["verdict"] == "not comparable" and words in row["note"], row


def test_stage_0_passes_the_reason_to_the_evidence():
    src = (Path(TEMPLATE)).read_text()
    assert "repo_chat_template_state(model_dir)" in src and "template_reason=" in src


# ---- 3. the reader has a time budget ---------------------------------------------------------------

class FakeClock:
    def __init__(self, step=1.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        self.t += self.step
        return self.t


def test_a_header_read_past_its_deadline_is_refused_as_too_slow(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "_now", FakeClock(1.0))
    monkeypatch.setattr(m, "GGUF_CLOCK_EVERY", 1)
    p = write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors()), "t.gguf")
    with pytest.raises(m.GgufError, match="too slow"):
        m.read_gguf_header(p, deadline=3.0)
    assert m.read_gguf_header(p, deadline=None)["version"] == 3                    # no deadline: as before


def test_a_file_that_runs_out_of_time_is_recorded_not_fatal(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "_now", FakeClock(1.0))
    monkeypatch.setattr(m, "GGUF_CLOCK_EVERY", 1)
    monkeypatch.setattr(m, "GGUF_FILE_TIME_S", 2.0)
    write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors()), "t.gguf")
    out = m.gguf_evidence(tmp_path, [{"file": "t.gguf"}], hybrid_config(), None)
    f = out["files"][0]
    assert f["status"] == "unreadable" and "too slow" in f["reason"]


def test_the_run_budget_stops_opening_files_once_it_is_spent(tmp_path, monkeypatch):
    m = module()
    clock = FakeClock(0.0)
    monkeypatch.setattr(m, "_now", clock)
    monkeypatch.setattr(m, "GGUF_TIME_BUDGET_S", 15.0)
    monkeypatch.setattr(m, "GGUF_FILE_TIME_S", 100.0)
    real, calls = m.read_gguf_header, []

    def slow(path, **kw):
        calls.append(Path(path).name)
        out = real(path, **kw)
        clock.t += 10.0
        return out

    monkeypatch.setattr(m, "read_gguf_header", slow)
    for n in "abc":
        write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors()), f"{n}.gguf")
    out = m.gguf_evidence(tmp_path, [{"file": f"{n}.gguf"} for n in "abc"], hybrid_config(), None)
    assert calls == ["a.gguf", "b.gguf"]
    assert [f["status"] for f in out["files"]] == ["ok", "ok", "unreadable"]
    assert "time budget" in out["files"][2]["reason"]


def test_the_worst_legal_header_is_cut_off_by_the_deadline_in_a_limited_child(tmp_path):
    p = tmp_path / "worst.gguf"
    count = 3_900_000                                              # two arrays of 3.9 million empty strings: 62.4 MB, under the 64 MiB cap
    with open(p, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 2))
        for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
            f.write(s(key) + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_STR, count))
            for _ in range(count // 100_000):
                f.write(b"\0" * (8 * 100_000))
    code = r"""
import importlib.util, json, resource, sys, time
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
spec = importlib.util.spec_from_file_location("t", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t0 = time.monotonic()
try:
    m.read_gguf_header(sys.argv[2], deadline=time.monotonic() + 1.0)
    out = "read"
except m.GgufError as e:
    out = str(e)
print(json.dumps({"out": out, "seconds": time.monotonic() - t0}))
"""
    r = subprocess.run([sys.executable, "-c", code, str(TEMPLATE), str(p)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout)
    assert "too slow" in res["out"] and res["seconds"] < 20, res


# ---- 4. a link is followed only inside the snapshot or its cache's blobs directory -----------------

def hf_layout(tmp_path):
    cache = tmp_path / "models--org--name"
    blobs, snap = cache / "blobs", cache / "snapshots" / "rev"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    return blobs, snap


GOOD = gguf(hybrid_kvs(), hybrid_tensors())


def test_a_link_into_the_caches_blobs_directory_is_read(tmp_path):
    m = module()
    blobs, snap = hf_layout(tmp_path)
    (blobs / "abc123").write_bytes(GOOD)
    (snap / "m.gguf").symlink_to("../../blobs/abc123")
    out = m.gguf_evidence(snap, [{"file": "m.gguf"}], hybrid_config(), None)
    assert out["files"][0]["status"] == "ok"


def test_a_plain_file_and_a_link_inside_the_snapshot_are_read(tmp_path):
    m = module()
    _, snap = hf_layout(tmp_path)
    (snap / "plain.gguf").write_bytes(GOOD)
    (snap / "alias.gguf").symlink_to("plain.gguf")
    out = m.gguf_evidence(snap, [{"file": "plain.gguf"}, {"file": "alias.gguf"}], hybrid_config(), None)
    assert [f["status"] for f in out["files"]] == ["ok", "ok"]


def test_a_link_to_a_file_outside_the_snapshot_and_its_blobs_is_refused_with_a_reason(tmp_path):
    m = module()
    _, snap = hf_layout(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "real.gguf").write_bytes(GOOD)
    (outside / "notes.txt").write_text("secret")
    (snap / "m.gguf").symlink_to(outside / "real.gguf")
    (snap / "n.gguf").symlink_to(outside / "notes.txt")
    out = m.gguf_evidence(snap, [{"file": "m.gguf"}, {"file": "n.gguf"}], hybrid_config(), None)
    for f in out["files"]:
        assert f["status"] == "unreadable" and "outside" in f["reason"]
        assert f["size"] is None                                   # not even the size of the target
    assert "secret" not in json.dumps(out)


def test_the_check_is_made_on_the_opened_file_not_on_a_path_that_could_change(tmp_path):
    """The reader opens first and asks the kernel where the descriptor really points, so a link swapped between a path check and the open
    cannot lead outside."""
    src = Path(TEMPLATE).read_text()
    assert "/proc/self/fd/" in src and "allowed_roots" in src
    m = module()
    _, snap = hf_layout(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "real.gguf").write_bytes(GOOD)
    (snap / "m.gguf").symlink_to(outside / "real.gguf")
    with pytest.raises(m.GgufError, match="outside"):
        m.read_gguf_header(snap / "m.gguf", allowed_roots=(snap.resolve(), (snap.parent.parent / "blobs").resolve()))
    assert m.read_gguf_header(snap / "m.gguf")["version"] == 3                     # the reader alone still follows links


def test_when_the_kernel_cannot_say_where_the_descriptor_points_the_path_check_is_used(tmp_path, monkeypatch):
    m = module()
    _, snap = hf_layout(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "real.gguf").write_bytes(GOOD)
    (snap / "m.gguf").symlink_to(outside / "real.gguf")
    (snap / "plain.gguf").write_bytes(GOOD)
    real = os.readlink

    def broken(p, *a, **k):
        if str(p).startswith("/proc/self/fd/"):
            raise OSError("no /proc")
        return real(p, *a, **k)

    monkeypatch.setattr(os, "readlink", broken)
    roots = (snap.resolve(), (snap.parent.parent / "blobs").resolve())
    with pytest.raises(m.GgufError, match="outside"):
        m.read_gguf_header(snap / "m.gguf", allowed_roots=roots)
    assert m.read_gguf_header(snap / "plain.gguf", allowed_roots=roots)["version"] == 3


def test_the_deadline_is_also_checked_in_the_key_value_loop_and_in_the_tensor_loop(tmp_path, monkeypatch):
    """A header with no arrays can still be slow: many key-value pairs, or many tensors. Each loop reads the clock."""
    m = module()
    monkeypatch.setattr(m, "GGUF_CLOCK_EVERY", 1)
    many_kvs = write(tmp_path, gguf([kv(f"k{i}", T_U32, i) for i in range(20)], []), "kvs.gguf")
    many_tensors = write(tmp_path, gguf([kv("general.architecture", T_STR, "x")], [(f"t{i}", [4], F32, 0) for i in range(20)], data=False),
                         "tensors.gguf")
    for p in (many_kvs, many_tensors):
        monkeypatch.setattr(m, "_now", FakeClock(1.0))
        with pytest.raises(m.GgufError, match="too slow"):
            m.read_gguf_header(p, deadline=5.0)
        monkeypatch.setattr(m, "_now", FakeClock(0.0))                    # a clock that does not move never runs out
        assert m.read_gguf_header(p, deadline=5.0)["version"] == 3
