# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 366: the catalog card gate asks an Expected performance table that has a reference column to say what the reference is.

A table with no reference column is not asked for one (a throughput-only card stays valid). When a column is headed `reference`,
`HF reference` or `baseline`, each body row that has a number in a result column needs text or a number in that column, or the
section needs a sentence under the table that names the reference.
"""
import pytest

from orchard import can


def card(table: str, after: str = "") -> str:
    return f"# toy\n## Expected performance\n{table}\n{after}## Limitations\nOne limit.\n"


FILLED = "| eval | this package | HF reference |\n| --- | --- | --- |\n| score corr | 0.998 | 1.000 |\n| top-1 | 0.97 | 1.00 |\n"
EMPTY_REF = "| eval | this package | HF reference |\n| --- | --- | --- |\n| score corr | 0.998 |  |\n| top-1 | 0.97 | 1.00 |\n"


def test_a_table_with_no_reference_column_is_not_asked_for_one():
    assert can.card_problems(card("| metric | value |\n| --- | --- |\n| decode | 12.5 |\n"), "catalog") == []


def test_a_reference_column_filled_in_every_numeric_row_passes():
    assert can.card_problems(card(FILLED), "catalog") == []


def test_a_numeric_row_with_an_empty_reference_cell_is_named():
    problems = can.card_problems(card(EMPTY_REF), "catalog")
    assert any("reference" in p and "score corr" in p for p in problems), problems
    assert not any("top-1" in p for p in problems), problems          # a row that fills the cell is not named


@pytest.mark.parametrize("cell", ["-", "--", "n/a", "N/A", "tbd", "TBD", "—", "none", "?"])
def test_a_placeholder_is_not_a_reference(cell):
    table = f"| eval | this package | baseline |\n| --- | --- | --- |\n| score corr | 0.998 | {cell} |\n"
    assert can.card_problems(card(table), "catalog"), cell


def test_a_sentence_under_the_table_that_names_the_reference_stands_in_for_empty_cells():
    after = "The reference is the transformers implementation on CPU, run on the same token ids.\n"
    assert can.card_problems(card(EMPTY_REF, after), "catalog") == []


def test_a_sentence_that_does_not_name_a_reference_does_not_stand_in():
    assert can.card_problems(card(EMPTY_REF, "Measured on one chip.\n"), "catalog")


def test_a_sentence_above_the_table_does_not_stand_in():
    text = f"# toy\n## Expected performance\nThe reference is the CPU run.\n{EMPTY_REF}## Limitations\nOne limit.\n"
    assert can.card_problems(text, "catalog")


@pytest.mark.parametrize("head", ["reference", "Reference", "HF reference", "`HF reference`", "**baseline**", "CPU reference (fp32)"])
def test_the_reference_column_is_found_by_its_heading(head):
    table = f"| eval | this package | {head} |\n| --- | --- | --- |\n| score corr | 0.998 |  |\n"
    assert can.card_problems(card(table), "catalog"), head


@pytest.mark.parametrize("head", ["references", "reference count", "preference", "evidence"])
def test_a_heading_that_only_contains_the_word_is_not_a_reference_column(head):
    table = f"| eval | this package | {head} |\n| --- | --- | --- |\n| score corr | 0.998 |  |\n"
    assert can.card_problems(card(table), "catalog") == [], head


def test_a_row_with_no_number_in_a_result_column_is_not_asked_for_a_reference():
    table = "| eval | this package | HF reference |\n| --- | --- | --- |\n| device | blackhole |  |\n| score corr | 0.998 | 1.0 |\n"
    assert can.card_problems(card(table), "catalog") == []


def test_a_second_table_is_judged_on_its_own_header():
    second = "| profile | req/s |\n| --- | --- |\n| 1 client | 4.2 |\n"
    assert can.card_problems(card(FILLED + "\n" + second), "catalog") == []
    assert can.card_problems(card(EMPTY_REF + "\n" + second), "catalog")


def test_a_reference_table_inside_a_fence_is_not_read():
    text = card(FILLED + "\n```\n" + EMPTY_REF + "```\n")
    assert can.card_problems(text, "catalog") == []


def test_a_table_without_outer_pipes_is_read():
    table = "eval | this package | baseline\n--- | --- | ---\nscore corr | 0.998 |\n"
    assert can.card_problems(card(table), "catalog")


def test_a_cell_past_the_header_width_is_not_the_reference():
    table = "| eval | this package |\n| --- | --- |\n| score corr | 0.998 | 1.0 |\n"
    assert can.card_problems(card(table), "catalog") == []


def test_no_profile_means_no_reference_check():
    assert can.card_problems(card(EMPTY_REF), None) == []


def test_the_problem_text_says_how_to_fix_it():
    (msg,) = [p for p in can.card_problems(card(EMPTY_REF), "catalog") if "reference" in p]
    assert "sentence" in msg and "reference" in msg
