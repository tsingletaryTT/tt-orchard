# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runtime hazards the Lottolabs cards report (study of 2026-11-24, learnings 22 to 24 and 31). They are not read from a config. Each row says it is a
card claim that this repository has not reproduced, names the card and the section it comes from, and says what a stage should run to look for it."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orchard import hazards as hz

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {"trace_inputs_overwritten_by_later_capture", "lm_head_noc_barrier_hang", "host_read_during_trace_capture",
            "tensor_cache_identity_is_name_and_shape"}


def test_the_table_holds_the_four_hazards_the_cards_report():
    assert {r.name for r in hz.RUNTIME_HAZARD_TABLE} == EXPECTED


def test_every_row_is_complete_unique_and_snake_case():
    names = [r.name for r in hz.RUNTIME_HAZARD_TABLE]
    assert len(set(names)) == len(names)
    for r in hz.RUNTIME_HAZARD_TABLE:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", r.name)
        for field in ("applies_to", "what", "source", "check", "known_here"):
            assert isinstance(getattr(r, field), str) and getattr(r, field).strip(), (r.name, field)


def test_every_row_says_it_is_a_card_claim_and_names_the_card_and_the_section():
    for r in hz.RUNTIME_HAZARD_TABLE:
        assert "card claim" in r.source and "not reproduced here" in r.source, r.name
        assert "Lottolabs" in r.source and "section" in r.source, r.name


def test_the_hang_row_gives_the_rate_the_card_gives_and_says_the_cause_is_inferred():
    r = hz.runtime_hazard("lm_head_noc_barrier_hang")
    assert "20,000" in r.what and "inferred" in r.what
    assert "20,000" in r.check and "tt-triage" in r.check


def test_the_trace_rows_name_what_gets_overwritten_and_the_read_that_fails():
    assert "page table" in hz.runtime_hazard("trace_inputs_overwritten_by_later_capture").what
    assert "Reads are not supported during trace capture" in hz.runtime_hazard("host_read_during_trace_capture").what


def test_the_cache_row_says_what_orchard_checks_today_and_what_the_card_adds():
    r = hz.runtime_hazard("tensor_cache_identity_is_name_and_shape")
    assert ".orchard-model" in r.known_here and "checkpoint revision" in r.known_here
    assert "manifest" in r.what and "image" in r.what


def test_an_unknown_row_is_a_key_error():
    with pytest.raises(KeyError):
        hz.runtime_hazard("nope")


def test_the_record_is_plain_json():
    rec = [r.as_dict() for r in hz.RUNTIME_HAZARD_TABLE]
    json.dumps(rec, allow_nan=False)
    assert set(rec[0]) == {"name", "applies_to", "what", "source", "check", "known_here"}


def test_the_sampler_row_describes_the_stock_op_and_names_the_custom_scans_that_do_not_wrap():
    row = next(e for e in hz.HAZARD_TABLE if e.name == "vocab_per_die_above_65536")
    assert "stock sampler op" in row.why and "custom" in row.why and "201,024" in row.why and "248,320" in row.why
    assert "card claim" in row.why                                       # said once, so the row is not read as our measurement


def test_the_command_prints_the_table_with_runtime(capsys):
    assert hz.main(["--runtime"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {r["name"] for r in out["runtime_hazards"]} == EXPECTED


def test_the_command_still_needs_a_config_without_runtime(capsys):
    with pytest.raises(SystemExit) as exc:
        hz.main([])
    assert exc.value.code == 2


def test_runtime_and_a_config_together_are_refused_with_one_line(capsys, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
                               "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 32000}))
    assert hz.main([str(cfg), "--runtime"]) == 2
    assert "Traceback" not in capsys.readouterr().out


def test_runtime_with_a_nearest_config_alone_is_refused(capsys):
    assert hz.main(["--runtime", "--nearest", "nearest.json"]) == 2
    assert "takes no config" in capsys.readouterr().out


def test_the_scan_of_a_config_is_unchanged_by_the_runtime_table(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
                               "num_key_value_heads": 2, "head_dim": 32, "vocab_size": 32000}))
    assert hz.main([str(cfg)]) == 0
    assert "runtime_hazards" not in json.loads(capsys.readouterr().out)


def test_the_doc_names_every_row_and_says_where_they_come_from():
    text = (ROOT / "docs" / "runtime-hazards.md").read_text()
    for r in hz.RUNTIME_HAZARD_TABLE:
        assert r.name in text, r.name
    assert "card claim" in text and "Lottolabs" in text


def test_the_readme_points_to_the_doc():
    assert "docs/runtime-hazards.md" in (ROOT / "README.md").read_text()
