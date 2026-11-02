# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 381, Copilot's seventh review: the two small, certain findings (token-shaped dictionary keys, and a bound on the probe's output).
Everything else in that review is in the one public issue that lists the remaining findings."""
from __future__ import annotations

import json
import sys

import pytest

from orchard import skillsfresh, stack_pins as sp
from test_stack_pins import GOOD, build, config, full, names, probe_says, py, repo  # noqa: F401  (py is a fixture)
from test_stack_pins_round2 import real_runner


# ---- dictionary keys are redacted like values --------------------------------------------------------------------------

def test_a_token_shaped_architecture_name_does_not_survive_as_a_key_of_the_record(tmp_path, py):
    token = "hf_abcdefgh12345678"
    model = config(tmp_path, model_type="m", architectures=[token, "Good"], transformers_version="4.0.0")
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("5.0.0", {token: True, "Good": True}))
    text = json.dumps(rec)
    assert token not in text and "abcdefgh12345678" not in text
    assert set(rec["parts"]["transformers"]["classes"]) == {"[token removed]", "Good"}


def test_names_that_redact_alike_keep_separate_results():
    out = sp._tilde({"hf_aaaaaaaaaaaa1": True, "hf_bbbbbbbbbbbb2": False, "plain": 1})
    assert sorted(out) == ["[token removed]", "[token removed]#2", "plain"] and sorted(map(str, out.values())) == ["1", "False", "True"]


def test_keys_are_redacted_at_every_depth_and_values_still_are():
    out = sp._tilde({"a": [{"https://u:pw12345678@h.example/x": "ghp_zzzzzzzzzzzz9"}]})
    text = json.dumps(out)
    assert "pw12345678" not in text and "zzzzzzzzzzzz9" not in text and "h.example/x" in text


# ---- the probe's output is bounded while the child runs ----------------------------------------------------------------------

NOISY = "import sys\nsys.stdout.write('x' * 50_000_000)\nsys.stdout.flush()\nsys.stderr.write('e' * 50_000_000)\n"


def test_a_noisy_child_cannot_fill_memory_with_its_output(tmp_path):
    raw = real_runner()
    rc, out, err = raw([sys.executable, "-c", NOISY], 60, None, cwd=str(tmp_path), max_bytes=4096)
    assert len(out) <= 4096 and len(err) <= 4096                           # at most the bound is read back, and the child was stopped at it


def test_the_child_is_stopped_at_the_bound_not_only_read_back_to_it(tmp_path):
    """The bound is a file-size limit on the child, so the write past it fails inside the child; a read-back cut alone would let it fill the disk."""
    marker = tmp_path / "write-failed"
    code = ("import sys\ntry:\n    sys.stdout.write('x' * 5_000_000); sys.stdout.flush()\nexcept OSError:\n"
            f"    open({str(marker)!r}, 'w').write('stopped')\n")
    real_runner()([sys.executable, "-c", code], 60, None, cwd=str(tmp_path), max_bytes=4096)
    assert marker.exists()


def test_the_bound_leaves_a_quiet_child_alone(tmp_path):
    raw = real_runner()
    rc, out, err = raw([sys.executable, "-c", "print('hello')"], 60, None, cwd=str(tmp_path), max_bytes=4096)
    assert rc == 0 and out.strip() == "hello"


def test_the_probe_reports_a_noisy_interpreter_as_unknown_and_reads_only_the_bound(tmp_path, monkeypatch):
    script = tmp_path / "noisy-python"
    script.write_text(f"#!{sys.executable}\n" + NOISY)
    script.chmod(0o755)
    monkeypatch.setattr(skillsfresh, "_raw_run", real_runner())
    out = sp.probe_interpreter(str(script), [])
    assert out["state"] == "unknown"


def test_the_probe_runner_passes_the_bound(monkeypatch):
    seen = {}

    def fake_raw(argv, timeout, env=None, cwd=None, max_bytes=None):
        seen["max_bytes"] = max_bytes
        return 1, "", ""
    monkeypatch.setattr(skillsfresh, "_raw_run", fake_raw)
    sp.real_probe_run([sys.executable, "-c", "pass"], 5, {})
    assert seen["max_bytes"] == sp.PROBE_MAX_BYTES
