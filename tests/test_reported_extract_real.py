# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Cases found by running the extractor on the real model cards in the Hugging Face cache (2026-10-26).

Each test is the smallest card that shows what the first version got wrong. The cards are named in the docstrings.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
MODEL = "Acme/Wren-7B"


def extract(text, model=MODEL):
    spec = importlib.util.spec_from_file_location("delta_triage_reported_real_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_reported(text, model)


def by_name(result):
    return {r["name"]: r for r in result["rows"]}


def reasons(result):
    return [s["reason"] for s in result["skipped"]]


def test_a_property_table_is_not_a_score_table():
    """episod/tt-tnt: `| Property | Value |` rows (Hidden size 384, Layers 6) came out as scores in the first version."""
    r = extract("| Property | Value |\n|---|---|\n| Hidden size | 384 |\n| Layers | 6 |\n")
    assert r["rows"] == []


def test_label_columns_before_the_model_column_are_part_of_the_name():
    """Edge0/Audio8-ASR-Infinite: a CER row and a WER row of the same test set must not read as one benchmark."""
    card = ("| test set | metric | Wren-7B | Other-4B |\n|---|---|---|---|\n"
            "| aishell1/test | CER | 1.75 | 16.8 |\n| librispeech test.clean | WER | 3.04 | 2.21 |\n")
    assert set(by_name(extract(card))) == {"aishell1/test / CER", "librispeech test.clean / WER"}


def test_label_columns_include_the_first_column_and_skip_empty_cells():
    card = "| Category | Benchmark | Shots | Wren-7B |\n|---|---|---|---|\n| General | MMLU | 5 | 69.4 |\n| Math |  | 0 | 51 |\n"
    assert set(by_name(extract(card))) == {"General / MMLU / 5", "Math / 0"}


def test_a_row_with_a_different_number_of_cells_than_the_header_is_skipped():
    """unsloth/Llama-3.1-8B-Instruct: rowspan cells shift the columns, so a value would come from the wrong column."""
    card = ("<table><tr><th>Category</th><th>Benchmark</th><th>Wren-7B</th><th>Other</th></tr>"
            "\n<tr><td rowspan=2>General</td><td>MMLU</td><td>69.4</td><td>82</td></tr>"
            "\n<tr><td>MMLU CoT</td><td>80.9</td><td>83</td></tr></table>\n")
    r = extract(card)
    assert {k: v["value"] for k, v in by_name(r).items()} == {"General / MMLU": "69.4"}
    assert any("cells" in why and "header" in why for why in reasons(r))


def test_an_html_cell_made_of_several_elements_keeps_a_space_between_them():
    """Qwen/Qwen3.8-27B: `<span>Agentic coding</span><span>SWE-bench Pro</span>` was read as 'Agentic codingSWE-bench Pro'."""
    card = ("<table><tr><th>Benchmark</th><th>Wren-7B</th></tr>"
            "<tr><td><span>Agentic coding</span><span>SWE-bench Pro</span></td><td>61.7</td></tr></table>\n")
    assert set(by_name(extract(card))) == {"Agentic coding SWE-bench Pro"}


def test_a_model_row_with_the_wrong_number_of_cells_is_skipped_not_guessed():
    card = "| Model | A | B |\n|---|---|---|\n| Wren-7B | 1 |\n"
    r = extract(card)
    assert r["rows"] == [] and any("cells" in why and "header" in why for why in reasons(r))


# ---- guards that need their own card ---------------------------------------------------------------------------------

def test_a_table_inside_a_fenced_block_is_an_example_not_the_cards_table():
    card = "```\n| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n```\n"
    assert extract(card)["rows"] == []
    assert len(extract("~~~\n~~~\n| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n")["rows"]) == 1


def test_only_an_https_source_link_goes_into_the_note():
    card = ("---\nmodel-index:\n  - name: x\n    results:\n      - dataset:\n          name: D\n        metrics:\n"
            "          - type: a\n            value: 1\n        source:\n          url: http://example.org/x\n---\n")
    assert "http://" not in extract(card)["rows"][0]["note"]
    assert "https://example.org/x" in extract(card.replace("http://", "https://"))["rows"][0]["note"]


def test_a_cell_that_holds_only_script_text_gives_no_value():
    card = "<table><tr><th>Model</th><th>MMLU</th></tr><tr><td>Wren-7B</td><td><script>99</script></td></tr></table>\n"
    assert extract(card)["rows"] == []


def test_two_splits_of_one_dataset_keep_both_rows_with_the_split_in_the_second_name():
    card = ("---\nmodel-index:\n  - name: x\n    results:\n"
            "      - dataset:\n          name: D\n          split: test\n        metrics:\n          - type: acc\n            value: 1\n"
            "      - dataset:\n          name: D\n          split: validation\n        metrics:\n          - type: acc\n            value: 2\n---\n")
    names = [r["name"] for r in extract(card)["rows"]]
    assert names == ["D / acc", "D / acc (validation)"]


def test_a_number_too_large_for_a_float_is_not_a_plain_number():
    big = "1" + "0" * 320
    r = extract(f"| Benchmark | Wren-7B |\n|---|---|\n| MMLU | {big} |\n")
    assert r["rows"] == [] and any("plain number" in s["reason"] for s in r["skipped"])


def test_the_skipped_list_is_capped_and_the_rest_counted():
    t = importlib.util.spec_from_file_location("x", TEMPLATE)
    mod = importlib.util.module_from_spec(t)
    t.loader.exec_module(mod)
    body = "".join(f"| b{i} | bad |\n" for i in range(mod.MAX_SKIPPED + 7))
    r = mod.extract_reported("| Benchmark | Wren-7B |\n|---|---|\n" + body, MODEL)
    assert len(r["skipped"]) == mod.MAX_SKIPPED and r["skipped_dropped"] == 7


def test_a_heading_that_says_measured_here_skips_even_a_table_with_an_innocent_name():
    card = "## Results measured on QuietBox\n\n| Benchmark | Wren-7B |\n|---|---|\n| MMLU | 71 |\n"
    assert extract(card)["rows"] == []


def module():
    spec = importlib.util.spec_from_file_location("delta_triage_reported_real_mod", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_number_with_more_digits_than_a_value_may_hold_is_skipped():
    r = extract("| Benchmark | Wren-7B |\n|---|---|\n| MMLU | " + "1" * 41 + " |\n")
    assert r["rows"] == [] and any("value is too long" in s["reason"] for s in r["skipped"])


def test_a_row_with_a_blank_name_is_never_added():
    rows = module()._Rows(MODEL)
    rows.add("   ", "1", "table", 3, "| |")
    assert rows.rows == [] and "no name" in rows.skipped[0]["reason"]


def test_a_source_link_over_120_characters_stays_out_of_the_note():
    url = "https://example.org/" + "a" * 120
    card = ("---\nmodel-index:\n  - name: x\n    results:\n      - dataset:\n          name: D\n        metrics:\n"
            f"          - type: a\n            value: 1\n        source:\n          url: {url}\n---\n")
    assert "example.org" not in extract(card)["rows"][0]["note"]


def test_a_model_index_indented_with_tabs_is_not_read_and_says_so():
    card = "---\nmodel-index:\n\t- name: x\n---\n"
    assert any("tab" in s["reason"] for s in extract(card)["skipped"])


def test_a_model_index_nested_deeper_than_the_limit_is_not_read_and_says_so():
    lines = ["---", "model-index:"] + [" " * (2 * k + 2) + f"k{k}:" for k in range(30)] + ["---"]
    r = extract("\n".join(lines) + "\n")
    assert r["rows"] == [] and any("deeper" in s["reason"] for s in r["skipped"])


def test_a_table_after_the_byte_limit_is_not_read():
    limit = module().MAX_README_BYTES
    tail = "\n\n| Benchmark | Wren-7B |\n|---|---|\n| ARC | 5 |\n"
    assert extract("x" * limit + tail)["rows"] == []
    assert len(extract("x" * (limit - len(tail)) + tail)["rows"]) == 1


def test_a_label_column_in_a_model_row_table_is_not_a_benchmark():
    r = extract("| Model | Metric | MMLU |\n|---|---|---|\n| Wren-7B | acc | 71 |\n")
    assert set(by_name(r)) == {"MMLU"} and r["skipped"] == []


def test_a_linked_model_name_and_an_escaped_pipe_in_a_cell_are_read():
    card = "| Model | MMLU |\n|---|---|\n| [Wren-7B](https://example.org/wren) | 71 |\n"
    assert by_name(extract(card))["MMLU"]["value"] == "71"
    card = "| Benchmark | Wren-7B |\n|---|---|\n| a \\| b | 5 |\n"
    assert set(by_name(extract(card))) == {"a | b"}
