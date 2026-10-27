# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 reads the scores a model's authors report from the model card (issue 19, second half).

The extractor reads text it is given and nothing else: no network, no model, no prose. It reads two structured
forms, in this order: the YAML front matter's `model-index` block and tables (markdown pipe tables and HTML
tables). A row counts only when it is about the model under test: in a comparison table the row of another model
is somebody else's score. Every row quotes the card's own text and line. Anything it will not read is listed with
the reason, so a person can see what was left out.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"


def tpl():
    spec = importlib.util.spec_from_file_location("delta_triage_reported_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODEL = "Acme/Wren-7B"


def extract(text, model=MODEL):
    return tpl().extract_reported(text, model)


def by_name(result):
    return {r["name"]: r for r in result["rows"]}


def reasons(result):
    return [s["reason"] for s in result["skipped"]]


# ---- model-index ------------------------------------------------------------------------------------------

CARD_INDEX = """---
license: apache-2.0
model-index:
  - name: Acme/Wren-7B
    results:
      - task:
          type: text-generation
          name: Grade school math
        dataset:
          type: gsm8k
          name: GSM8K
          split: test
        metrics:
          - type: accuracy
            name: exact match
            value: 84.2
        source:
          name: Open LLM Leaderboard
          url: https://example.org/leaderboard
      - task:
          type: text-generation
        dataset:
          type: humaneval
          name: HumanEval
        metrics:
          - type: pass@1
            value: 0.61
---
# Wren
"""


def test_model_index_gives_one_row_per_metric_with_its_line_and_text():
    r = extract(CARD_INDEX)
    rows = by_name(r)
    assert set(rows) == {"GSM8K / exact match", "HumanEval / pass@1"}
    g = rows["GSM8K / exact match"]
    assert g["value"] == "84.2"
    assert g["extracted_from"]["kind"] == "model-index"
    assert g["extracted_from"]["line"] == 16                      # the `value:` line, counted from line 1 of the file
    assert g["extracted_from"]["text"] == "value: 84.2"
    assert "https://example.org/leaderboard" in g["note"]         # the card's own source link goes in the note
    assert "model-index" in g["note"] and "line 16" in g["note"]
    assert rows["HumanEval / pass@1"]["value"] == "0.61"
    assert r["skipped"] == []


def test_model_index_flow_style_is_not_read_and_says_so():
    card = "---\nmodel-index:\n  - name: x\n    results: [{task: {type: t}, metrics: [{type: a, value: 1}]}]\n---\n"
    r = extract(card)
    assert r["rows"] == []
    assert any("model-index" in x and "not read" in x for x in reasons(r))


def test_model_index_metric_without_a_plain_number_is_skipped_with_its_line():
    card = CARD_INDEX.replace("value: 84.2", "value: high")
    r = extract(card)
    assert "GSM8K / exact match" not in by_name(r)
    bad = [s for s in r["skipped"] if s["line"] == 16]
    assert bad and "plain number" in bad[0]["reason"]


@pytest.mark.parametrize("value", ["nan", ".nan", "inf", "-inf", "1e999", "true", "null", "~", "''"])
def test_model_index_refuses_non_finite_and_non_numeric_values(value):
    r = extract(CARD_INDEX.replace("value: 84.2", f"value: {value}"))
    assert "GSM8K / exact match" not in by_name(r)
    assert any(s["line"] == 16 for s in r["skipped"])


def test_model_index_cannot_run_away_on_a_hostile_card():
    card = "---\nmodel-index:\n" + "  - name: x\n    results:\n" * 5000 + "---\n"
    r = extract(card)                                              # must return, with no rows
    assert r["rows"] == []


# ---- markdown tables ---------------------------------------------------------------------------------------

def test_row_labelled_with_the_model_gives_one_row_per_benchmark_column():
    card = ("| Model | MMLU | GSM8K |\n|---|---|---|\n| Other-9B | 70.1 | 80.0 |\n| Wren-7B | 71.5 | 82.25 |\n")
    r = extract(card)
    rows = by_name(r)
    assert set(rows) == {"MMLU", "GSM8K"}
    assert rows["MMLU"]["value"] == "71.5" and rows["GSM8K"]["value"] == "82.25"
    ef = rows["GSM8K"]["extracted_from"]
    assert ef["kind"] == "table" and ef["line"] == 4 and ef["text"] == "| Wren-7B | 71.5 | 82.25 |"
    assert "table" in rows["GSM8K"]["note"] and "line 4" in rows["GSM8K"]["note"]
    assert r["other_model_rows"] == 1                              # another model's score is counted, never kept


def test_a_row_of_another_model_is_never_a_row_of_this_one():
    card = "| Model | MMLU |\n|---|---|\n| Wren-7B-Instruct | 99 |\n| Wren-70B | 98 |\n"
    r = extract(card)
    assert r["rows"] == [] and r["other_model_rows"] == 2


def test_the_model_label_must_match_exactly_after_normalising():
    card = "| Model | MMLU |\n|---|---|\n| **wren 7b** | 71 |\n"
    assert set(by_name(extract(card))) == {"MMLU"}                 # case, spaces and hyphens do not matter
    assert set(by_name(extract("| Model | MMLU |\n|---|---|\n| **Wren-7B** | 71 |\n"))) == {"MMLU"}
    assert set(by_name(extract("| Model | MMLU |\n|---|---|\n| Acme/Wren-7B | 71 |\n"))) == {"MMLU"}
    assert set(by_name(extract("| Model | MMLU |\n|---|---|\n| `wren-7b` | 71 |\n"))) == {"MMLU"}
    assert extract("| Model | MMLU |\n|---|---|\n| Wren-7B-Base | 71 |\n")["rows"] == []


def test_column_headed_with_the_model_gives_one_row_per_benchmark_row():
    card = ("| Benchmark | Base | Wren-7B |\n|---|---|---|\n| MMLU | 60 | 71 |\n| GSM8K | 70 | 82 |\n")
    rows = by_name(extract(card))
    assert {k: v["value"] for k, v in rows.items()} == {"MMLU": "71", "GSM8K": "82"}
    assert rows["MMLU"]["extracted_from"]["line"] == 3


def test_two_column_score_table_is_the_models_own_scores():
    card = "| Benchmark | Score |\n|---|---|\n| MMLU | 71.2 |\n| ARC | 55 |\n"
    assert {k: v["value"] for k, v in by_name(extract(card)).items()} == {"MMLU": "71.2", "ARC": "55"}


def test_three_column_table_with_no_model_column_is_not_guessed():
    card = "| Benchmark | Score | Base |\n|---|---|---|\n| MMLU | 71.2 | 60 |\n"
    r = extract(card)
    assert r["rows"] == []                                         # which column is the model's is not stated


def test_row_label_column_names_get_the_header_as_a_prefix_unless_generic():
    card = "| State | Wren-7B |\n|---|---|\n| `FULL` | **.930** |\n"
    assert set(by_name(extract(card))) == {"State: FULL"}
    assert by_name(extract(card))["State: FULL"]["value"] == ".930"   # bold markers are stripped, the number is kept
    card = "| Metric | Wren-7B |\n|---|---|\n| Action accuracy | .938 |\n"
    assert set(by_name(extract(card))) == {"Action accuracy"}


@pytest.mark.parametrize("cell", ["**.938 [.932, .944]**", "58.0*", "84 (est.)", "~70", "1,234", "70 / 80", "1e999",
                                  "nan", "inf", "0x10", "٧٠"])
def test_a_cell_that_is_not_a_plain_number_is_skipped_with_the_reason(cell):
    r = extract(f"| Benchmark | Wren-7B |\n|---|---|\n| MMLU | {cell} |\n")
    assert r["rows"] == []
    assert any("plain number" in why for why in reasons(r)), r["skipped"]
    assert r["skipped"][0]["line"] == 3 and cell in r["skipped"][0]["text"]


@pytest.mark.parametrize("cell", ["-", "--", "–", "—", "", "N/A", "n/a"])
def test_an_empty_cell_or_dash_means_no_result_and_is_not_a_row(cell):
    r = extract(f"| Model | MMLU | GSM8K |\n|---|---|---|\n| Wren-7B | {cell} | 5 |\n")
    assert set(by_name(r)) == {"GSM8K"}
    assert r["skipped"] == []                                       # no result is not a refusal


@pytest.mark.parametrize("cell,shown", [("84%", "84%"), ("+5", "+5"), ("-0.5", "-0.5"), (".5", ".5"), ("7.", "7."), ("12", "12")])
def test_plain_numbers_with_an_optional_percent_are_kept_verbatim(cell, shown):
    r = extract(f"| Model | MMLU |\n|---|---|\n| Wren-7B | {cell} |\n")
    assert by_name(r)["MMLU"]["value"] == shown


def test_percent_and_fraction_in_one_column_are_both_skipped():
    card = "| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71% |\n| ARC | 0.55 |\n| GSM8K | 82 |\n"
    r = extract(card)
    assert r["rows"] == []
    assert sum("percent" in why and "mixed" in why for why in reasons(r)) == 3


def test_percent_in_every_row_of_a_column_is_fine():
    card = "| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71% |\n| ARC | 55% |\n"
    assert len(extract(card)["rows"]) == 2


def test_a_table_under_a_measured_heading_is_not_the_authors_report():
    for heading in ("## Measured on Tenstorrent Blackhole", "### Results reproduced with tt-orchard", "## Speed on p300"):
        r = extract(f"{heading}\n\n| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n")
        assert r["rows"] == [], heading
        assert any("measured on Tenstorrent" in why for why in reasons(r))
    r = extract("## Evaluation\n\n| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n")
    assert len(r["rows"]) == 1


def test_a_speed_or_hardware_row_name_is_skipped():
    card = "| Benchmark | Wren-7B |\n|---|---|\n| decode tok/s on p300 | 41 |\n| MMLU | 71 |\n"
    r = extract(card)
    assert set(by_name(r)) == {"MMLU"}
    assert any("measured on Tenstorrent" in why for why in reasons(r))


def test_the_second_row_with_a_name_is_skipped_as_a_duplicate():
    card = "| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n| MMLU | 72 |\n"
    r = extract(card)
    assert by_name(r)["MMLU"]["value"] == "71"
    assert any("twice" in why for why in reasons(r))


def test_names_over_the_limit_are_skipped():
    long = "x" * 101
    r = extract(f"| Benchmark | Wren-7B |\n|---|---|\n| {long} | 71 |\n")
    assert r["rows"] == [] and any("too long" in why for why in reasons(r))


def test_rows_over_the_cap_are_listed_as_skipped():
    t = tpl()
    body = "".join(f"| bench{i} | {i} |\n" for i in range(t.MAX_REPORTED_ROWS + 5))
    r = extract("| Benchmark | Wren-7B |\n|---|---|\n" + body)
    assert len(r["rows"]) == t.MAX_REPORTED_ROWS == 60
    assert sum("over the 60-row cap" in why for why in reasons(r)) == 5


# ---- HTML tables (AREX-2's card is made of them) -----------------------------------------------------------

HTML = """## Evaluation

<table>
<thead>
<tr><th>Model</th><th>Params</th><th>Frontier-CS</th><th>HLE</th></tr>
</thead>
<tbody>
<tr><td colspan="4">Closed-weight models</td></tr>
<tr><td>GPT-9</td><td>-</td><td>76.4</td><td>58.0*</td></tr>
<tr style="x"><td>Wren-7B</td><td style="text-align:center">7B</td><td style="text-align:center">70.7</td><td>58.0*</td></tr>
</tbody>
</table>
"""


def test_html_table_row_of_the_model_is_read_and_another_models_is_not():
    r = extract(HTML)
    rows = by_name(r)
    assert rows["Frontier-CS"]["value"] == "70.7"
    assert rows["Frontier-CS"]["extracted_from"]["kind"] == "table"
    assert rows["Frontier-CS"]["extracted_from"]["line"] == 10         # the line of the row's <tr>
    assert "Params" not in rows                                         # "7B" is not a plain number: it is skipped, not a row
    assert r["other_model_rows"] == 1                                   # GPT-9
    assert any("plain number" in s["reason"] and "58.0*" in s["text"] for s in r["skipped"])


def test_html_entities_and_nested_markup_are_decoded_not_trusted():
    card = "<table><tr><th>Model</th><th>MMLU &amp; more</th></tr><tr><td><b>Wren-7B</b></td><td><i>71.5</i></td></tr></table>\n"
    rows = by_name(extract(card))
    assert rows["MMLU & more"]["value"] == "71.5"


def test_html_script_content_is_not_read():
    card = "<table><tr><th>Model</th><th>MMLU</th></tr><tr><td>Wren-7B</td><td><script>99</script>71</td></tr></table>\n"
    r = extract(card)
    assert r["rows"] == [] or by_name(r)["MMLU"]["value"] == "71"


def test_malformed_html_does_not_raise():
    r = extract("<table><tr><td>Wren-7B</td><td>5</td></table></tr></td><<<>>> <th")
    assert isinstance(r["rows"], list)


# ---- bounds ------------------------------------------------------------------------------------------------

def test_a_huge_card_is_read_only_up_to_the_byte_limit_and_says_so():
    t = tpl()
    head = "| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n"
    r = extract(head + "x" * (t.MAX_README_BYTES + 10) + "\n| ARC | 5 |\n")
    assert r["truncated"] is True
    assert set(by_name(r)) == {"MMLU"}


def test_only_so_many_tables_are_read():
    t = tpl()
    one = "| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n\n"
    r = extract(one * (t.MAX_TABLES + 3))
    assert len(r["rows"]) == 1                                         # the repeated name is a duplicate: one kept
    assert any("tables" in why and "limit" in why for why in reasons(r))


def test_the_source_lists_nothing_when_the_card_has_nothing():
    r = extract("# Just prose\n\nIt scores 84 on MMLU.\n")
    assert r["rows"] == [] and r["skipped"] == []
    assert r["other_model_rows"] == 0 and r["truncated"] is False


def test_prose_is_never_parsed():
    r = extract("Wren-7B reaches 84.2 on GSM8K and 61 on HumanEval.\n")
    assert r["rows"] == []


def test_model_index_comes_first_then_tables_and_a_name_in_both_is_kept_once():
    card = CARD_INDEX + "\n| Benchmark | Wren-7B |\n|---|---|\n| GSM8K / exact match | 1 |\n| MMLU | 70 |\n"
    r = extract(card)
    rows = by_name(r)
    assert rows["GSM8K / exact match"]["extracted_from"]["kind"] == "model-index"
    assert rows["MMLU"]["extracted_from"]["kind"] == "table"
    assert any("twice" in why for why in reasons(r))


def test_the_result_is_plain_json():
    r = extract(CARD_INDEX)
    assert json.loads(json.dumps(r)) == r
