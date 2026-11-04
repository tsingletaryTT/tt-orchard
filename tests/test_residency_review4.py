# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 435, the independent review of its final head (after Copilot's round limit).

1. MEDIUM. The conversion's in-flight host RAM grouped tensors by `layers.<n>.` only. A checkpoint that names its blocks `h.N`, `blocks.N`,
   `decoder.layers.N` and so on fell back to one group per category, so the figure was about twice the whole category: a 67.5 GiB model gave
   120 GiB of host RAM, and `machine_check` told the operator that a valid model did not fit the box. A tensor now sits in the numbered block of its
   name (`layers.N` as before, or the first numbered path component that repeats across the checkpoint's tensors), and a tensor with no such block is
   a unit on its own. The block text says which quantity exceeded which, that it is an estimate, and the grouping rule used.
2. LOW. The Hub readers had a per-read socket timeout and no total deadline; a server that drips bytes could hold a 16 MiB read for a long time.
   They read in chunks under a total deadline and a byte cap enforced while reading.
3. LOW. A truncated chunked answer (`http.client.IncompleteRead`, an HTTPException and not an OSError) escaped the readers and `hub_info`.
4. LOW. `_check_residency` reads MemAvailable before the coder is parked (known limit, see the PR body and its issue)."""
from __future__ import annotations

import http.client
import json
import time

import pytest

from orchard import preflight as pf, residency as rs
from residency_headers import write_snapshot
from test_residency_options import FakeResp, _cfg, hub_payload

MIB = 1 << 20
BLOCK_FLIGHT = 128 * MIB          # one block: two 32 MiB bf16 tensors, each counted as its source plus the larger of source and converted size


def estimate_of(tmp_path, tensors, name="snap"):
    return rs.estimate(rs.read_headers(write_snapshot(tmp_path / name, tensors, shards=1)), box_chips=4)


def numbered_blocks(fmt: str, n: int = 6):
    t = [("embed_tokens.weight", "BF16", [1000, 64])]
    for i in range(n):
        p = fmt.format(i=i)
        t += [(f"{p}.mlp.up_proj.weight", "BF16", [4096, 4096]), (f"{p}.mlp.down_proj.weight", "BF16", [4096, 4096])]
    return t


SCHEMES = ["model.layers.{i}", "transformer.h.{i}", "transformer.blocks.{i}", "model.decoder.layers.{i}",
           "language_model.model.layers.{i}", "h.{i}", "blocks.{i}"]


# ---- 1. the grouping of the conversion's in-flight host RAM -----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", SCHEMES)
def test_the_in_flight_conversion_is_one_block_whatever_the_blocks_are_named(tmp_path, fmt):
    fb = estimate_of(tmp_path, numbered_blocks(fmt))["estimated"]["first_boot"]
    assert fb["conversion_in_flight_bytes"] == BLOCK_FLIGHT, fmt


def test_a_model_with_unusual_block_names_is_not_priced_at_twice_its_whole_category(tmp_path):
    odd = estimate_of(tmp_path, numbered_blocks("transformer.h.{i}", 60), "odd")["estimated"]["first_boot"]
    usual = estimate_of(tmp_path, numbered_blocks("model.layers.{i}", 60), "usual")["estimated"]["first_boot"]
    assert odd["host_ram_bytes"] == usual["host_ram_bytes"] and odd["conversion_in_flight_bytes"] == BLOCK_FLIGHT


def test_tensors_with_no_numbered_block_are_units_on_their_own_and_not_one_group(tmp_path):
    t = [("alpha.mlp.weight", "BF16", [4096, 4096]), ("beta.mlp.weight", "BF16", [4096, 4096]), ("gamma.mlp.weight", "BF16", [4096, 4096])]
    fb = estimate_of(tmp_path, t)["estimated"]["first_boot"]
    assert fb["conversion_in_flight_bytes"] == 64 * MIB                    # the largest single tensor (32 MiB source + 32 MiB converted), not 192 MiB


def test_a_numbered_component_that_does_not_repeat_is_not_a_block(tmp_path):
    t = [("alpha.7.mlp.a.weight", "BF16", [4096, 4096]), ("alpha.7.mlp.b.weight", "BF16", [4096, 4096])]      # two tensors under one number that appears once
    fb = estimate_of(tmp_path, t)["estimated"]["first_boot"]
    assert fb["conversion_in_flight_bytes"] == 64 * MIB                    # one tensor at a time, not the 128 MiB of a block
    assert fb["conversion_grouping"]["blocks"] == 0 and fb["conversion_grouping"]["tensors_alone"] == 2


def test_vision_blocks_and_text_blocks_are_each_grouped_and_the_largest_group_counts(tmp_path):
    t = numbered_blocks("model.layers.{i}")
    t += [(f"model.visual.blocks.{i}.attn.qkv.weight", "BF16", [4096, 4096]) for i in range(40)]            # 40 blocks of one 32 MiB tensor each
    fb = estimate_of(tmp_path, t)["estimated"]["first_boot"]
    assert fb["conversion_in_flight_bytes"] == BLOCK_FLIGHT                  # not the 2.5 GiB of the vision category taken as one group


def test_experts_stay_inside_their_block_and_do_not_become_blocks_of_their_own(tmp_path):
    def experts(fmt):
        return [(f"{fmt.format(i=i)}.mlp.experts.{e}.w1.weight", "BF16", [1024, 1024]) for i in range(4) for e in range(8)]
    a = estimate_of(tmp_path, experts("model.layers.{i}"), "a")["estimated"]["first_boot"]["conversion_in_flight_bytes"]
    b = estimate_of(tmp_path, experts("transformer.h.{i}"), "b")["estimated"]["first_boot"]["conversion_in_flight_bytes"]
    assert a == b and a > 8 * MIB                                           # eight experts in one layer, not one expert at a time


def test_a_single_numbered_layer_still_groups_by_the_layers_rule_as_before(tmp_path):
    t = [("model.layers.0.mlp.up_proj.weight", "BF16", [4096, 4096]), ("model.layers.0.mlp.down_proj.weight", "BF16", [4096, 4096])]
    assert estimate_of(tmp_path, t)["estimated"]["first_boot"]["conversion_in_flight_bytes"] == BLOCK_FLIGHT


def test_the_record_says_which_rule_grouped_the_conversion(tmp_path):
    g = estimate_of(tmp_path, numbered_blocks("transformer.h.{i}"))["estimated"]["first_boot"]["conversion_grouping"]
    assert g["blocks"] == 6 and g["tensors_alone"] == 1 and "numbered block" in g["rule"]
    g2 = estimate_of(tmp_path, [("alpha.mlp.weight", "BF16", [64, 64])], "n")["estimated"]["first_boot"]["conversion_grouping"]
    assert g2["blocks"] == 0 and g2["tensors_alone"] == 1


def test_the_block_text_names_the_quantities_the_estimate_and_the_grouping_rule(tmp_path):
    rec = estimate_of(tmp_path, numbered_blocks("transformer.h.{i}", 60))
    rec["estimated"]["first_boot"]["host_ram_bytes"] = 120 * (1 << 30)
    block, _ = rs.machine_check(rec, free_disk_bytes=1 << 50, mem_available_bytes=100 * (1 << 30))
    assert block and "120.0 GiB" in block and "100.0 GiB" in block and "host RAM" in block
    assert "estimate" in block and "numbered block" in block and "does not fit this box" in block


def test_an_older_record_without_the_grouping_still_gives_its_block():
    rec = {"scan": "ran", "label": "x", "estimated": {"chips_needed": 1, "device_bytes": 1, "first_boot": {"disk_bytes": 1, "host_ram_bytes": 10 << 30,
           "checkpoint_bytes_to_download": 0}}}
    block, _ = rs.machine_check(rec, free_disk_bytes=1 << 50, mem_available_bytes=1 << 30)
    assert block and "host RAM" in block


def test_grouping_names_is_fast_on_hostile_names():
    names = [f"a{i}.{j}.{k}.w" for i in range(60) for j in range(60) for k in range(60)] + ["0." * 5000 + "w", "x" * 100_000, ""]
    t0 = time.monotonic()
    repeated = rs._repeated_parents(names)
    keys = [rs._block_key(n, repeated) for n in names]
    assert time.monotonic() - t0 < 10 and len(keys) == len(names)
    assert keys[-1] is None and keys[-2] is None


def test_the_prefixes_remembered_for_grouping_are_bounded(monkeypatch):
    monkeypatch.setattr(rs, "_MAX_PARENTS", 2)
    repeated = rs._repeated_parents(["a.0.w", "b.0.w", "c.0.w", "c.1.w"])              # `c` would repeat, but only two prefixes are remembered
    assert "c" not in repeated
    monkeypatch.setattr(rs, "_MAX_PARENTS", 10)
    assert "c" in rs._repeated_parents(["a.0.w", "b.0.w", "c.0.w", "c.1.w"])


def test_a_name_is_read_only_to_a_fixed_depth():
    deep = ["x." * rs._MAX_PATH_PARTS + f"{i}.w" for i in (3, 4)]                        # the number sits past the depth that is read
    assert rs._repeated_parents(deep) == frozenset()                                     # the search for repeating prefixes stops at the depth
    parent = ".".join(["x"] * rs._MAX_PATH_PARTS)
    assert rs._block_key(deep[0], frozenset({parent})) is None                           # and so does the block key, whatever it is given
    assert rs._block_key(deep[0], rs._repeated_parents(deep)) is None
    shallow = ["x." * (rs._MAX_PATH_PARTS - 2) + f"{i}.w" for i in (3, 4)]
    assert rs._block_key(shallow[0], rs._repeated_parents(shallow)) is not None


def test_a_numbered_part_has_at_most_six_digits():
    long_numbers = ["m.1234567.w", "m.1234568.w"]
    assert rs._repeated_parents(long_numbers) == frozenset()                             # the search ignores a number of seven digits
    assert rs._block_key(long_numbers[0], frozenset({"m"})) is None                      # and so does the block key, whatever it is given
    assert rs._block_key(long_numbers[0], rs._repeated_parents(long_numbers)) is None
    short_numbers = ["m.123456.w", "m.123457.w"]
    assert rs._block_key(short_numbers[0], rs._repeated_parents(short_numbers)) == "m.123456"


# ---- 2 and 3. the Hub readers: a total deadline, a byte cap while reading, and a truncated answer ------------------------------------


class Drip:
    """A response that gives `step` bytes per read, and a clock that moves `tick` seconds per read."""
    def __init__(self, step=10, tick=1.0, total=None):
        self.step, self.tick, self.reads, self.served, self.now, self.total = step, tick, 0, 0, 0.0, total

    def clock(self):
        return self.now

    def read(self, n=-1):
        self.reads += 1
        self.now += self.tick
        n = self.step if n is None or n < 0 else min(n, self.step)
        if self.total is not None:
            n = min(n, self.total - self.served)
        self.served += n
        return b" " * n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_dripping_server_is_cut_off_by_the_total_deadline_for_the_config():
    d = Drip()
    assert pf.read_hub_config("owner/name", "main", opener=lambda *a, **k: d, clock=d.clock) is None
    assert d.reads <= int(pf.HUB_TOTAL_S) + 3 and d.served < 1000


def test_a_dripping_server_is_cut_off_by_the_total_deadline_for_the_index():
    d = Drip()
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: d, clock=d.clock) is None
    assert d.reads <= int(pf.HUB_TOTAL_S) + 3


def test_the_byte_cap_is_enforced_while_reading_and_not_after_the_whole_answer_is_held():
    d = Drip(step=1 << 20, tick=0.0)                                     # an endless fast answer
    assert pf.read_hub_config("owner/name", "main", opener=lambda *a, **k: d, clock=d.clock) is None
    assert d.served <= pf.CONFIG_FETCH_BYTES + (1 << 16) + 1
    d2 = Drip(step=1 << 20, tick=0.0)
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: d2, clock=d2.clock) is None
    assert d2.served <= pf.INDEX_FETCH_BYTES + (1 << 16) + 1


def test_a_normal_answer_in_many_small_reads_is_read_whole():
    body = json.dumps({"weight_map": {"model.layers.0.q.weight": "a"}}).encode()
    class Small(FakeResp):
        def read(self, n=-1):
            return super().read(min(5, n) if n and n > 0 else 5)
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: Small(body)) == ["model.layers.0.q.weight"]
    assert pf.read_hub_config("owner/name", "main", opener=lambda *a, **k: Small(b'{"a": 1}')) == {"a": 1}


class Truncated:
    def read(self, n=-1):
        raise http.client.IncompleteRead(b"{\"hid", 40)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("exc", [http.client.IncompleteRead(b"x", 3), http.client.BadStatusLine("x"), http.client.RemoteDisconnected("gone")])
def test_a_truncated_or_broken_answer_is_no_figure_and_not_an_exception(exc):
    def opener(*a, **k):
        raise exc
    assert pf.read_hub_config("owner/name", "main", opener=opener) is None
    assert pf.read_hub_weight_index("owner/name", "main", opener=opener) is None
    assert pf.read_hub_config("owner/name", "main", opener=lambda *a, **k: Truncated()) is None
    assert pf.read_hub_weight_index("owner/name", "main", opener=lambda *a, **k: Truncated()) is None


def test_hub_info_keeps_the_counts_and_the_fit_row_when_the_config_answer_is_cut_short(monkeypatch, tmp_path):
    def fake(req, timeout=None):
        if "/api/models/" in req.full_url:
            return FakeResp(json.dumps(hub_payload()).encode())
        return Truncated()
    monkeypatch.setattr(pf.urllib.request, "urlopen", fake)
    info, err = pf.default_signals(_cfg(tmp_path)).hub_info("deep-principle/Hydrogen-1.0")
    assert err is None and info is not None and info.parameters and info.host_table is None
    assert pf.check_fit(info, 4).status == "warn"                           # the numbers are still there, and nothing is ruled out or in


def test_a_truncated_main_hub_answer_is_an_error_text_and_not_an_exception(monkeypatch, tmp_path):
    monkeypatch.setattr(pf.urllib.request, "urlopen", lambda req, timeout=None: Truncated())
    info, err = pf.default_signals(_cfg(tmp_path)).hub_info("deep-principle/Hydrogen-1.0")
    assert info is None and "IncompleteRead" in err
