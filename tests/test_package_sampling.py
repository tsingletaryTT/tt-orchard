# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package card and the sampling probe (issue 107). The card of the required profile says what stage 2's probe found about a sampled request, from the
raw record in `stages/2/evidence/swap-check.json` and not from the model's own card. The Numbers table has the speeds and the drafter's accepted share;
the evidence file the rows cite, `stages/2/sampling-summary.json`, is derived from the raw record, and the gate derives it again before it believes the
card. A run with no probe record gets the card it got before."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_graft_auto import finished_auto, rows  # noqa: E402
from test_sampling_probe import LINE, OTHER, STORY, entry, record, refused  # noqa: E402
from package_fakes import write  # noqa: E402

from orchard import package, sampling_probe  # noqa: E402
from orchard.stages import gate_package  # noqa: E402

SPEED_ROWS = ("decode speed, greedy request (stage 2 sampling probe)", "decode speed, sampled request (stage 2 sampling probe)")
RATE_ROWS = ("drafter acceptance, greedy request (stage 2 sampling probe)", "drafter acceptance, sampled request (stage 2 sampling probe)")
SWAP = "stages/2/evidence/swap-check.json"
SUMMARY = "stages/2/sampling-summary.json"


def put_probe(w, rec, serving=None):
    p = w["run"] / SWAP
    old = json.loads(p.read_text()) if p.is_file() else {}
    write(p, {**old, "serving": serving or {"drafter": "off", "sampling": "host"}, "sampling_probe": rec})


def card_of(w):
    stage_dir, _, pkg = finished_auto(w)
    return stage_dir, pkg, (pkg / "README.md").read_text()


def section(card, head="## Sampling"):
    return card.split(head, 1)[1].split("\n## ", 1)[0]


def test_a_run_with_no_probe_record_has_no_sampling_section_rows_or_summary_file(world):
    stage_dir, pkg, card = card_of(world)
    assert "## Sampling" not in card
    assert not any(name in rows(card) for name in SPEED_ROWS + RATE_ROWS)
    assert not (world["run"] / SUMMARY).exists()
    assert gate_package(stage_dir, world["run"]).ok


def test_a_healthy_probe_gives_a_section_and_rows_that_cite_the_derived_file_first(world):
    put_probe(world, record(greedy=entry(seconds=2.0, log=LINE), a=entry(text=STORY, seconds=2.0, log=LINE), b=entry(text=OTHER, seconds=3.0)))
    stage_dir, pkg, card = card_of(world)
    r = rows(card)
    assert r[SPEED_ROWS[0]] == ("48.0 tokens/s", f"`{SUMMARY}`, `{SWAP}`")
    assert r[SPEED_ROWS[1]][0] == "40.0 tokens/s"
    assert r[RATE_ROWS[0]][0] == "0.536 fraction" and r[RATE_ROWS[1]][0] == "0.536 fraction"
    body = " ".join(section(card).split())
    assert sampling_probe.CARD_SENTENCES["sampled-full-speed"] in body
    assert "drafter: off, sampling: host" in body
    assert "top-k, penalties and logprobs were not tested" in body
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons
    assert json.loads((world["run"] / SUMMARY).read_text())["greedy_tokens_per_s"] == 48.0
    assert (pkg / "evidence" / SUMMARY).is_file() and (pkg / "evidence" / SWAP).is_file()        # the cited files ship with the package


def test_ignored_sampling_is_on_the_card_and_the_sampled_speed_row_is_still_there(world):
    put_probe(world, record(a=entry(text=STORY), b=entry(text=STORY)))
    stage_dir, pkg, card = card_of(world)
    assert sampling_probe.CARD_SENTENCES["sampled-ignored"] in " ".join(section(card).split())
    assert gate_package(stage_dir, world["run"]).ok


def test_a_refused_sampled_request_has_no_sampled_rows_and_the_card_says_refused(world):
    put_probe(world, record(a=refused("HTTP 400: nope"), b=refused("HTTP 400: nope")))
    stage_dir, pkg, card = card_of(world)
    r = rows(card)
    assert SPEED_ROWS[0] in r and SPEED_ROWS[1] not in r
    assert sampling_probe.CARD_SENTENCES["sampled-refused"] in " ".join(section(card).split())
    assert "nope" not in card
    assert gate_package(stage_dir, world["run"]).ok


def test_a_record_that_did_not_run_gives_no_section(world):
    put_probe(world, {"ran": False})
    stage_dir, pkg, card = card_of(world)
    assert "## Sampling" not in card and not (world["run"] / SUMMARY).exists()


def test_a_malformed_record_stops_packaging_because_stage_2_s_gate_refuses_it(world):
    rec = record()
    rec["requests"]["greedy"]["seconds"] = "fast"
    put_probe(world, rec)
    with pytest.raises(package.PackageError, match="sampling_probe"):
        finished_auto(world)


def test_the_gate_refuses_a_card_that_lost_the_section(world):
    put_probe(world, record())
    stage_dir, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    head, tail = card.split("## Sampling", 1)
    (pkg / "README.md").write_text(head + "## Limitations" + tail.split("## Limitations", 1)[1])
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("Sampling" in r for r in g.reasons), g.reasons


def test_the_gate_refuses_a_card_that_changed_the_verdict_sentence(world):
    put_probe(world, record(a=entry(text=STORY), b=entry(text=STORY)))
    stage_dir, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    (pkg / "README.md").write_text(card.replace(sampling_probe.CARD_SENTENCES["sampled-ignored"],
                                                sampling_probe.CARD_SENTENCES["sampled-full-speed"]))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("sampling probe" in r and "verdict" in r for r in g.reasons), g.reasons


def test_the_gate_refuses_a_card_that_lost_a_speed_row(world):
    put_probe(world, record())
    stage_dir, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    (pkg / "README.md").write_text("\n".join(ln for ln in card.splitlines() if SPEED_ROWS[1] not in ln) + "\n")
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any(SPEED_ROWS[1] in r for r in g.reasons), g.reasons


def test_the_gate_derives_the_summary_again_so_an_edited_summary_file_is_refused(world):
    put_probe(world, record())
    stage_dir, _, pkg = finished_auto(world)
    p = world["run"] / SUMMARY
    d = json.loads(p.read_text())
    d["greedy_tokens_per_s"] = 999.0
    p.write_text(json.dumps(d, indent=2) + "\n")
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any(SUMMARY in r and "derived" in r for r in g.reasons), g.reasons


def test_the_gate_refuses_a_run_whose_summary_file_is_missing(world):
    put_probe(world, record())
    stage_dir, _, pkg = finished_auto(world)
    (world["run"] / SUMMARY).unlink()
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any(SUMMARY in r for r in g.reasons), g.reasons


def test_the_gate_refuses_a_card_with_a_sampling_section_and_no_record_behind_it(world):
    stage_dir, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    (pkg / "README.md").write_text(card.replace("## Limitations", "## Sampling\n\nSampled requests were served.\n\n## Limitations", 1))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("no sampling probe" in r for r in g.reasons), g.reasons


def test_the_numbers_come_from_the_raw_requests_not_from_a_summary_in_the_record(world):
    rec = record(greedy=entry(seconds=2.0))
    rec["verdict"] = "sampled-ignored"
    rec["requests"]["greedy"]["tokens_per_s"] = 9999
    put_probe(world, rec)
    stage_dir, pkg, card = card_of(world)
    assert rows(card)[SPEED_ROWS[0]][0] == "48.0 tokens/s" and "sampled-ignored" not in card


@pytest.mark.parametrize("serving, phrase", [
    ({"drafter": "on (dflash2)", "sampling": "on device (decode_only)"}, "drafter: on (dflash2), sampling: on device (decode_only)"),
    (None, "drafter: off, sampling: host")])
def test_the_card_names_the_serving_state_stage_2_recorded(world, serving, phrase):
    put_probe(world, record(), serving=serving)
    _, _, card = card_of(world)
    assert phrase in " ".join(section(card).split())


ZERO_LINE = LINE.replace("accept 3.75/7", "accept 0.00/7").replace("histogram [0, 2, 0, 0, 0, 0, 1, 1]", "histogram [4, 0, 0, 0, 0, 0, 0, 0]")


def test_a_drafter_that_accepted_nothing_is_a_measurement_and_gets_its_rows(world):
    put_probe(world, record(greedy=entry(log=ZERO_LINE), a=entry(text=STORY, log=ZERO_LINE), b=entry(text=OTHER)))
    _, _, card = card_of(world)
    assert rows(card)[RATE_ROWS[0]][0] == "0.0 fraction" and rows(card)[RATE_ROWS[1]][0] == "0.0 fraction"


def test_a_probe_with_no_usable_greedy_count_gives_the_inconclusive_sentence_and_the_gate_passes(world):
    rec = record()
    rec["requests"]["greedy"]["completion_tokens"] = None
    put_probe(world, rec)
    stage_dir, _, card = card_of(world)
    assert sampling_probe.CARD_SENTENCES["inconclusive"] in " ".join(section(card).split())
    assert SPEED_ROWS[0] not in rows(card)
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_speeds_are_rounded_to_one_decimal_place_in_the_rows(world):
    put_probe(world, record(greedy=entry(seconds=1.995), a=entry(text=STORY, seconds=2.0), b=entry(text=OTHER, seconds=2.0)))
    _, _, card = card_of(world)
    assert rows(card)[SPEED_ROWS[0]][0] == "48.1 tokens/s"
