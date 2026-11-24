# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A measured scaling note for the decode floor (Inkling-2-Mini study, 2026-11-24, learning 1).

The floor divides the bytes read per token by the sum of the dies' bandwidth, so it assumes a speed that grows in step with the die count.
Raahem's published Inkling-2-Mini bundle gives one-user plain decode on 1, 2 and 4 dies (53.2, 74.8 and 92.3 tok/s at temperature 1), which is
53, 37 and 23 percent of the floor of the same bytes. `scaling_table` states those shares and the speed-up over the smallest die count next to
the ideal speed-up. The figures are the caller's (a card's), the table says so, and the floor itself never changes."""
import json
from pathlib import Path

import pytest

from orchard import decode_floor as df

FIX = Path(__file__).parent / "fixtures" / "model-configs"
INKLING = json.loads((FIX / "inkling-2-mini.config.json").read_text())
BFP8 = {"default": "bfp8"}
CARD = {1: 53.2, 2: 74.8, 4: 92.3}


def table(measured=CARD, policy=BFP8, bw=512.0):
    return df.scaling_table(INKLING, policy, bw, measured)


def test_the_shares_of_the_floor_are_the_card_speeds_over_the_floors_of_the_same_bytes():
    rows = table()
    # 5,079,462,912 bytes per token over 1, 2 and 4 dies at 512 GB/s
    for row, dies in zip(rows, (1, 2, 4)):
        floor = df.decode_floor(INKLING, BFP8, dies, 512.0)
        assert row["dies"] == dies
        assert row["floor_tok_s"] == pytest.approx(floor.floor_tok_s)
        assert row["share_of_floor"] == pytest.approx(CARD[dies] / floor.floor_tok_s)
    assert [round(r["share_of_floor"], 2) for r in rows] == [0.53, 0.37, 0.23]


def test_the_speed_up_over_the_smallest_die_count_is_set_beside_the_ideal_one():
    rows = table()
    assert [r["speedup_vs_smallest"] for r in rows] == pytest.approx([1.0, 74.8 / 53.2, 92.3 / 53.2])
    assert [r["ideal_speedup"] for r in rows] == [1.0, 2.0, 4.0]
    assert [round(r["scaling_efficiency"], 2) for r in rows] == [1.0, 0.70, 0.43]


def test_rows_come_out_in_die_order_whatever_the_input_order():
    assert [r["dies"] for r in table({4: 92.3, 1: 53.2, 2: 74.8})] == [1, 2, 4]


def test_the_table_says_the_figures_are_the_callers_and_were_not_measured_here():
    for row in table():
        assert "supplied by the caller" in row["label"] and "not measured by this module" in row["label"]


def test_the_floor_is_the_same_with_or_without_the_table():
    before = df.decode_floor(INKLING, BFP8, 4, 512.0)
    table()
    after = df.decode_floor(INKLING, BFP8, 4, 512.0)
    assert before == after


def test_a_single_point_is_a_valid_table_with_a_speed_up_of_one():
    rows = table({2: 74.8})
    assert len(rows) == 1 and rows[0]["speedup_vs_smallest"] == 1.0 and rows[0]["scaling_efficiency"] == 1.0


@pytest.mark.parametrize("bad", [{}, None, [], {0: 50.0}, {True: 50.0}, {"2": 50.0}, {2: 0.0}, {2: -1.0}, {2: float("nan")}, {2: float("inf")},
                                 {2: True}, {2: "74.8"}, {2: 10 ** 400}])
def test_a_measurement_that_is_not_a_positive_finite_speed_on_a_whole_die_count_is_refused(bad):
    with pytest.raises(df.FloorInputError):
        table(bad)


def test_the_assumptions_text_names_the_gap_between_the_floor_and_the_measured_scaling():
    assert "scaling_table" in df.ASSUMPTIONS


def test_the_command_prints_the_table_when_asked(capsys):
    cfg = str(FIX / "inkling-2-mini.config.json")
    rc = df.main([cfg, "--dies", "4", "--per-die-gb-s", "512", "--policy", "default=bfp8", "--measured-by-dies", "1=53.2,2=74.8,4=92.3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "supplied by the caller" in out and "not measured by this module" in out
    assert "1.00" in out and "0.70" in out and "0.43" in out


def test_the_command_refuses_a_malformed_list(capsys):
    cfg = str(FIX / "inkling-2-mini.config.json")
    assert df.main([cfg, "--dies", "4", "--per-die-gb-s", "512", "--policy", "default=bfp8", "--measured-by-dies", "four=92"]) == 2
    assert "decode_floor:" in capsys.readouterr().out
