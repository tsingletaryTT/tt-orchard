# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/qwen35_package.py: the staged recipe package for Qwen3.5-35B-A3B and its gate."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orchard import qwen35_package as qp

SRC = Path(__file__).resolve().parents[1] / "package_src"

ROW = {"prompt": "The history", "tokens": 32, "leading_agree": 20, "same_position_share": 0.8,
       "first_diff": {"index": 20, "ref_token": 1, "device_token": 2, "device_token_in_ref_top5": True, "ref_gap_nats": 0.2}}
FACTS = {"bench": {"median": 126.78, "min": 126.68, "max": 126.91, "runs": 5, "warmup": 1, "tokens": 128},
         "floor": {"floor_tok_s": 428.0, "floor_ms": 2.336, "mb_per_token": 2392.4, "policy": "bfp4 routed experts"},
         "ttft_ms": 107, "warm_start_s": [267, 272], "cold_start_s": None, "load": "5.2 to 6.5",
         "quality": {"rows": [ROW, dict(ROW, prompt="def fib", leading_agree=32, first_diff=None)]},
         "chat": {"default": "A card.", "finish": "stop", "thinking_off": "Red, blue, yellow."},
         "long_prompt": {"tokens": 3100, "text": " PELICAN"}, "ablations": {"grid_median": 126.84, "rk_median": 126.78}}
for r in FACTS["quality"]["rows"]:
    if r.get("first_diff") is None:
        r.pop("first_diff", None)


@pytest.fixture
def pkg(tmp_path):
    ev = tmp_path / "ev"
    ev.mkdir()
    (ev / "facts.json").write_text(json.dumps(FACTS))
    out = tmp_path / "pkg"
    qp.stage_package(SRC, FACTS, ev, out)
    return out


def gate(p):
    return qp.package_problems(p, hostname="testhost", home="/home/tester")


def test_a_staged_package_passes_the_gate(pkg):
    assert gate(pkg) == []


def test_staging_refuses_an_existing_directory(pkg, tmp_path):
    with pytest.raises(qp.PackageError):
        qp.stage_package(SRC, FACTS, tmp_path / "ev", pkg)


def test_weights_file_fails(pkg):
    (pkg / "model.safetensors").write_bytes(b"x")
    assert any("weights" in p for p in gate(pkg))


def test_home_path_fails(pkg):
    (pkg / "docs" / "running.md").write_text("see /home/tester/x\n")
    assert any("home path" in p for p in gate(pkg))


def test_credential_file_and_litellm_fail(pkg):
    (pkg / ".tt_ai_key").write_text("k")
    (pkg / "notes.txt").write_text("uses litellm\n")
    p = gate(pkg)
    assert any(".tt_ai_key" in x and "credential" in x for x in p) and any("litellm" in x for x in p)


def test_token_fails(pkg):
    (pkg / "notes.txt").write_text("hf_" + "a" * 34)
    assert any("token" in p for p in gate(pkg))


def test_edited_file_after_staging_fails_the_record(pkg):
    (pkg / "run.sh").write_text((pkg / "run.sh").read_text() + "# x\n")
    assert any("run.sh was changed" in p for p in gate(pkg))


def test_unrecorded_file_fails(pkg):
    (pkg / "extra.txt").write_text("x")
    assert any("not in the record" in p for p in gate(pkg))


def test_license_must_be_apache(pkg):
    (pkg / "LICENSE").write_text("Some other license\n")
    assert any("not the Apache" in p for p in gate(pkg))


def test_a_different_revision_in_the_manifest_fails(pkg):
    t = (pkg / "tt-model.yaml").read_text().replace(qp.REVISION, "0" * 40)
    (pkg / "tt-model.yaml").write_text(t)
    assert any("tt-model.yaml does not pin" in p for p in gate(pkg))


def test_card_must_show_the_measured_median(pkg):
    c = (pkg / "README.md").read_text().replace("126.78", "130.00")
    (pkg / "README.md").write_text(c)
    assert any("126.78" in p for p in gate(pkg))


def test_card_without_the_vision_limit_fails(pkg):
    c = (pkg / "README.md").read_text().replace("vision", "sight")
    (pkg / "README.md").write_text(c)
    assert any("'vision'" in p for p in gate(pkg))


def test_cold_start_claim_must_match_the_facts(tmp_path):
    f = dict(FACTS, cold_start_s=None)
    ev = tmp_path / "ev"; ev.mkdir()
    (ev / "facts.json").write_text(json.dumps(dict(FACTS, cold_start_s=900)))     # the evidence has a cold start, the card says not measured
    out = tmp_path / "p"
    qp.stage_package(SRC, f, ev, out)
    assert any("cold start" in p for p in gate(out))


def test_executable_bit_is_required_on_run_sh(pkg):
    (pkg / "run.sh").chmod(0o644)
    assert any("run.sh is not executable" in p for p in gate(pkg))


def test_symlink_fails(pkg, tmp_path):
    (pkg / "link").symlink_to(tmp_path)
    assert any("is a link" in p for p in gate(pkg))


def test_publish_text_is_text_with_the_operator_caveat():
    t = qp.publish_text("ns")
    assert "has not approved" in t and "hf upload" in t and t.startswith("#")


def test_run_sh_pins_image_and_loader_hashes():
    s = (SRC / "run.sh").read_text()
    assert qp.IMAGE_ID.split(":")[1] in s and "ORIG_SHA=" in s and "PATCHED_SHA=" in s


def test_fetch_script_pins_the_revision():
    assert qp.REVISION in (SRC / "fetch_weights.py").read_text()


def test_a_file_over_the_size_limit_fails_even_when_its_name_is_innocent(pkg):
    (pkg / "data.txt").write_text("a" * (qp.MAX_FILE_BYTES + 1))
    assert any("larger than" in p for p in gate(pkg))
