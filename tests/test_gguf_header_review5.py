# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 441, Copilot round 4 (issue 440): four findings, each reproduced by a test here before the fix.

1. `_fit_evidence` can reduce a file to its summary (no `agreement` rows), and `gguf_finding` then raised KeyError, which stopped stage 0
   before delta.json was written.
2. The same reduced file made RISKS.md say "0 agree, 0 differ, 0 not comparable" while the summary recorded differences.
3. Cutting a long tensor name before using it as a key merged `...weight` and `...bias` into one histogram entry.
4. A file name that is not valid UTF-8 left a lone surrogate in the evidence, and the bundle's strict reader dropped the whole file.
"""
from __future__ import annotations

import json

from test_gguf_header import F32, Q4_K, gguf, hybrid_config, hybrid_kvs, module, write
from test_gguf_header_review4 import bundle_with, names_header

SUMMARY_ONLY = {"file": "a.gguf", "size": 10, "status": "ok", "role": "language model", "architecture": "qwen35", "tensor_count": 12,
                "agreement_summary": {"agree": 9, "differ": 2, "not comparable": 1},
                "detail": "reduced to its summary because the evidence file reached its size cap"}


# ---- 1. gguf_finding on a file reduced to its summary ----------------------------------------------

def test_a_file_reduced_to_its_summary_gives_a_finding_with_the_counts_and_no_error():
    m = module()
    text = m.gguf_finding({"files": [dict(SUMMARY_ONLY)]})
    assert "9 rows agree" in text and "2 differ" in text and "1 not comparable" in text
    assert "differing items are not listed" in text


def test_a_summary_with_no_differences_says_nothing_about_dropped_rows():
    m = module()
    f = dict(SUMMARY_ONLY, agreement_summary={"agree": 3, "differ": 0, "not comparable": 0})
    assert "not listed" not in m.gguf_finding({"files": [f]})


def test_evidence_trimmed_by_the_size_cap_goes_through_the_finding(tmp_path, monkeypatch):
    m = module()
    write(tmp_path, gguf(hybrid_kvs(), [("token_embd.weight", [4], F32, 0)], data=False), "a.gguf")
    full = m.gguf_evidence(tmp_path, [{"file": "a.gguf"}], hybrid_config(), None)
    assert full["files"][0]["agreement"]
    reduced = None
    for cap in range(len(m.json_bytes(full)), 400, -200):         # find a cap where the rows are gone and the file is still listed
        monkeypatch.setattr(m, "GGUF_EVIDENCE_MAX_BYTES", cap)
        out = m.gguf_evidence(tmp_path, [{"file": "a.gguf"}], hybrid_config(), None)
        if out["files"] and out["files"][0].get("status") == "ok" and "agreement" not in out["files"][0]:
            reduced = out
            break
    assert reduced is not None, "no cap produced a summary-only file"
    assert m.gguf_finding(reduced).startswith("The repository ships GGUF files")


# ---- 2. the bundle keeps the recorded counts -------------------------------------------------------

def test_the_bundle_shows_the_summary_counts_when_the_rows_are_gone(tmp_path):
    sec = bundle_with(tmp_path, json.dumps({"schema": 1, "files": [SUMMARY_ONLY]}).encode())
    assert "9 agree, 2 differ, 1 not comparable" in sec
    assert "individual rows are not available" in sec
    assert "0 agree, 0 differ" not in sec


def test_the_bundle_counts_the_rows_when_they_are_there(tmp_path):
    f = dict(SUMMARY_ONLY, agreement=[{"item": "block count", "gguf": 4, "config": 5, "verdict": "differ"}])
    f.pop("detail")
    sec = bundle_with(tmp_path, json.dumps({"schema": 1, "files": [f]}).encode())
    assert "0 agree, 1 differ, 0 not comparable." in sec and "not available" not in sec


def test_a_summary_with_a_bad_count_is_not_shown_as_a_number(tmp_path):
    f = dict(SUMMARY_ONLY, agreement_summary={"agree": True, "differ": -1, "not comparable": "x"})
    sec = bundle_with(tmp_path, json.dumps({"schema": 1, "files": [f]}).encode())
    assert "True" not in sec and "-1 differ" not in sec


# ---- 3. two long names that share a start stay two kinds -------------------------------------------

def test_long_names_that_share_their_first_characters_are_two_kinds(tmp_path):
    m = module()
    stem = "x" * (m.GGUF_MAX_KIND_CHARS + 20)
    tensors = [(stem + ".weight", [4], Q4_K, 0), (stem + ".bias", [4], F32, 0)]
    e = m.gguf_file_evidence(write(tmp_path, gguf(hybrid_kvs(), tensors, data=False), "k.gguf"), None, None)
    q = e["quantization"]
    assert len(q["by_kind"]) == 2 and all(len(k) <= m.GGUF_MAX_KIND_CHARS for k in q["by_kind"])
    assert sorted(tuple(c) for c in q["by_kind"].values()) == [("F32",), ("Q4_K",)]
    assert len({x["kind"] for x in q["full_precision"]}) == len(q["full_precision"]) and [x["type"] for x in q["full_precision"]] == ["F32"]


def test_a_short_name_keeps_its_plain_key():
    m = module()
    assert m._kind_key("blk.3.attn.weight") == "blk.N.attn.weight"


# ---- 4. a file name that is not valid UTF-8 ----------------------------------------------------------

def _surrogates(v):
    if isinstance(v, str):
        return any(0xD800 <= ord(c) <= 0xDFFF for c in v)
    if isinstance(v, dict):
        return any(_surrogates(k) or _surrogates(x) for k, x in v.items())
    if isinstance(v, list):
        return any(_surrogates(x) for x in v)
    return False


def test_a_file_name_with_invalid_utf8_leaves_no_lone_surrogate_in_the_evidence(tmp_path):
    m = module()
    odd = "bad\udcff.gguf"
    (tmp_path / odd).write_bytes(names_header(3, 5))
    out = m.gguf_evidence(tmp_path, [{"file": odd}, {"file": "no\udcfe\x00.gguf"}], None, None)
    assert out["files"][0]["status"] == "ok" and out["files"][0]["file"] == "bad�.gguf"
    assert not _surrogates(out)
    assert not _surrogates(json.loads(m.json_bytes(out).decode("utf-8")))
