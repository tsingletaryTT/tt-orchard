# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 327, the parts after the first estimator: the precision options and the sentence that names the highest precision that fits, the
mixture-of-experts mark, the first boot's disk and host RAM against the machine, and the bound from the Hub's parameter counts (a model
whose tensor names and shapes were not read). The first acceptance case is deep-principle/Hydrogen-1.0: 753,329,940,480 parameters,
1,506.7 GB in bf16, on a box of four chips. Every figure here is a test fixture: nothing touches a network and no weight is read."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import blocked, preflight as pf, residency as rs
from residency_headers import dense_tensors, flash_next_tensors, write_snapshot

FIX = Path(__file__).parent / "fixtures" / "model-configs"
GIB = 1 << 30

REAL_MACHINE_FACTS = rs.machine_facts          # the conftest hook replaces rs.machine_facts for every test; this test of the real one keeps a reference

# What the Hub reports for deep-principle/Hydrogen-1.0 (`safetensors.parameters`): the totals only, no tensor name or shape.
HYDROGEN = {"BF16": 753_329_921_024, "F32": 19_456}
HYDROGEN_TOTAL = 753_329_940_480


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("a test reached the network")
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata", boom)
    monkeypatch.setattr(huggingface_hub, "model_info", boom)


def chips_for(nbytes, dram=32.0, reserve=3.0, kv=4.0):
    """The chip count by the arithmetic in the module's docstring, written out here on its own."""
    return math.ceil((nbytes + kv * GIB) / ((dram - reserve) * GIB))


def option(rec, name):
    return next(r for r in rec["estimated"]["precision_options"]["rows"] if r["precision"] == name)


# ---- Hydrogen: refused at every listed precision ------------------------------------------------------------------------------

def test_hydrogen_does_not_fit_four_chips_at_any_listed_precision():
    rec = rs.totals_bound(HYDROGEN, box_chips=4)
    assert rec["label"] == "bound" and rec["scan"] == "ran" and rs.usable(rec)
    assert rec["measured"]["total_parameters"] == HYDROGEN_TOTAL
    expect = {"bf16": HYDROGEN_TOTAL * 2, "bfp8": HYDROGEN_TOTAL * 17 // 16, "bfp4": HYDROGEN_TOTAL * 9 // 16}
    for name, nbytes in expect.items():
        assert option(rec, name)["device_bytes"] == nbytes
        assert option(rec, name)["chips_needed"] == chips_for(nbytes)
    assert [option(rec, n)["chips_needed"] for n in ("bf16", "bfp8", "bfp4")] == [49, 26, 14]
    assert rec["verdict"] == {"box_chips": 4, "fits_box": False, "chips_needed": 14, "best_fit": None}
    assert [h["name"] for h in rec["hazards"]] == ["model_does_not_fit_box"] and "at any listed precision" in rec["hazards"][0]["reason"]
    assert rs.totals_bound(HYDROGEN)["hazards"] == []                  # with no box named there is nothing to be a hazard against


def test_the_refusal_says_the_numbers_the_lowest_precision_and_that_nothing_listed_fits():
    rec = rs.totals_bound(HYDROGEN, box_chips=4)
    why = rs.shortfall(rec, 4)
    assert why.startswith("the model does not fit this box")
    assert "at least 14 chips" in why and "this box has 4 chips" in why
    assert "bf16 needs 49" in why and "bfp8 needs 26" in why and "bfp4 needs 14" in why
    assert "does not fit this box at any listed precision" in why
    assert "choice" in why and "no allocator reading" in why
    assert "bound" in why.lower()                      # the figure is a lower bound, and the sentence says so
    assert blocked.block_code({"reason": "blocked: " + why}) == "model-does-not-fit"


def test_the_bound_says_what_it_did_not_read():
    rec = rs.totals_bound(HYDROGEN, box_chips=4)
    text = " ".join(rec["not_measured"]).lower()
    assert "tensor names" in text and "one-dimensional" in text and "allocator" in text


def test_eight_chips_do_not_hold_hydrogen_either():
    rec = rs.totals_bound(HYDROGEN, box_chips=8)
    assert rec["verdict"]["fits_box"] is False and rs.shortfall(rec, 8)


def test_a_model_that_fits_at_bfp8_but_not_at_bf16_names_bfp8():
    rec = rs.totals_bound({"BF16": 70_000_000_000}, box_chips=4)
    assert [option(rec, n)["chips_needed"] for n in ("bf16", "bfp8", "bfp4")] == [5, 3, 2]
    assert rec["verdict"]["fits_box"] is True and rec["verdict"]["best_fit"] == "bfp8"
    assert rs.shortfall(rec, 4) is None                     # the bound says nothing against it
    # on a box of three chips bf16 (5) fails and bfp8 (3) fits; on two chips only bfp4 fits
    assert rs.totals_bound({"BF16": 70_000_000_000}, box_chips=2)["verdict"]["best_fit"] == "bfp4"
    assert rs.totals_bound({"BF16": 70_000_000_000}, box_chips=5)["verdict"]["best_fit"] == "bf16"


def test_the_sentence_for_a_smaller_lease_names_the_highest_precision_that_fits_it():
    """Seventy billion values: bf16 needs 5 chips, bfp8 needs 3 and bfp4 needs 2. A stage lease of 3 chips holds it at bfp8."""
    rec = rs.totals_bound({"BF16": 70_000_000_000}, box_chips=8)
    rec["estimated"]["chips_needed"] = 4                       # the policy the run uses needs 4; the lease has 3
    why = rs.shortfall(rec, 3, what="stage 2's lease")
    assert why.startswith("the model does not fit stage 2's lease")
    assert "the highest listed precision that fits is bfp8, which needs 3 chips" in why
    rec["estimated"]["chips_needed"] = 3
    assert "the highest listed precision that fits is bfp4, which needs 2 chips" in rs.shortfall(rec, 2, what="stage 2's lease")


def test_a_packed_sibling_is_counted_at_the_bytes_it_stores():
    """A packed repository (an MXFP4 or NVFP4 sibling) reports its stored bytes as U8 counts. A block format cannot shrink them."""
    sibling = {"U8": 400_000_000_000, "F32": 1_000_000}
    rec = rs.totals_bound(sibling, box_chips=4)
    assert rec["verdict"]["fits_box"] is False
    assert option(rec, "bf16")["device_bytes"] == option(rec, "bfp4")["device_bytes"] - (1_000_000 * 9 // 16) + 1_000_000 * 2
    assert "stored" in " ".join(rec["measured"]["notes"]).lower()
    assert option(rec, "bfp4")["chips_needed"] == chips_for(400_000_000_000 + 1_000_000 * 9 // 16)


@pytest.mark.parametrize("bad", [None, [], "BF16", {}, {"BF16": -1}, {"BF16": 1.5}, {"BF16": True}, {"BF16": "7"}, {"NOPE": 5}, {5: 5},
                                 {"BF16": 10 ** 16}, {f"X{i}": 1 for i in range(100)}])
def test_totals_that_are_not_counts_by_a_known_dtype_are_refused(bad):
    with pytest.raises(rs.ResidencyError):
        rs.totals_bound(bad, box_chips=4)


def test_totals_with_no_parameters_are_refused_not_counted_as_a_model_that_fits():
    with pytest.raises(rs.ResidencyError, match="no parameters"):
        rs.totals_bound({"BF16": 0}, box_chips=4)


# ---- the Hub call ---------------------------------------------------------------------------------------------------------------

def info(parameters=None, total=None):
    st = None if parameters is None else SimpleNamespace(parameters=parameters, total=total)
    return SimpleNamespace(safetensors=st)


def test_totals_come_from_one_model_info_call_and_the_repo_id_is_checked_first():
    seen = []

    def fetch(repo, revision):
        seen.append((repo, revision))
        return info(dict(HYDROGEN), HYDROGEN_TOTAL)
    assert rs.totals_from_hub("deep-principle/Hydrogen-1.0", fetch=fetch) == HYDROGEN
    assert seen == [("deep-principle/Hydrogen-1.0", None)]
    for bad in ("../x", "no-slash", "a/b/c", 5, "a/..", ""):
        with pytest.raises(rs.ResidencyError):
            rs.totals_from_hub(bad, fetch=fetch)
    with pytest.raises(rs.ResidencyError, match="revision"):
        rs.totals_from_hub("a/b", revision="x y", fetch=fetch)


def test_a_repository_the_hub_gives_no_counts_for_is_reported_as_such():
    with pytest.raises(rs.ResidencyError, match="no safetensors parameter counts"):
        rs.totals_from_hub("a/b", fetch=lambda r, v: info(None))
    with pytest.raises(rs.ResidencyError, match="no safetensors parameter counts"):
        rs.totals_from_hub("a/b", fetch=lambda r, v: SimpleNamespace())


def test_a_failed_hub_call_names_its_type_and_never_its_text():
    def boom(r, v):
        raise RuntimeError("401 with token hf_SECRET123")
    with pytest.raises(rs.ResidencyError) as exc:
        rs.totals_from_hub("a/b", fetch=boom)
    assert "RuntimeError" in str(exc.value) and "hf_SECRET123" not in str(exc.value)


def test_the_command_with_totals_only_prints_the_refusal_and_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(rs, "_hub_model_info", lambda repo, revision: info(dict(HYDROGEN), HYDROGEN_TOTAL))
    code = rs.main(["deep-principle/Hydrogen-1.0", "--totals-only", "--chips", "4"])
    out = capsys.readouterr().out
    assert code == 1
    assert "bound" in out and "753" in out and "does not fit" in out and "bfp4" in out
    assert "tensor names" in out                           # the not-measured list is printed
    assert rs.main(["deep-principle/Hydrogen-1.0", "--totals-only", "--chips", "64"]) == 0


def test_the_command_with_totals_only_can_print_json(monkeypatch, capsys):
    monkeypatch.setattr(rs, "_hub_model_info", lambda repo, revision: info(dict(HYDROGEN), HYDROGEN_TOTAL))
    assert rs.main(["deep-principle/Hydrogen-1.0", "--totals-only", "--json"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["label"] == "bound" and rec["estimated"]["chips_needed"] == 14


def test_a_policy_with_totals_only_is_refused_because_the_bound_lists_every_precision(monkeypatch, capsys):
    monkeypatch.setattr(rs, "_hub_model_info", lambda repo, revision: info(dict(HYDROGEN), HYDROGEN_TOTAL))
    assert rs.main(["a/b", "--totals-only", "--policy", "attention=bf16"]) == 2
    assert "--policy" in capsys.readouterr().out


def test_a_local_directory_with_totals_only_is_refused(tmp_path, capsys):
    assert rs.main([str(tmp_path), "--totals-only"]) == 2
    assert "repo id" in capsys.readouterr().out


# ---- precision options on a header estimate ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flash(tmp_path_factory):
    cfg = json.loads((FIX / "qwen3.8-flash-next.json").read_text())
    d = write_snapshot(tmp_path_factory.mktemp("flash"), flash_next_tensors(cfg), shards=7, config=cfg)
    return rs.estimate(rs.read_headers(d), box_chips=4)


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    cfg = json.loads((FIX / "qwen3.8-27b.json").read_text())
    d = write_snapshot(tmp_path_factory.mktemp("dense"), dense_tensors(cfg), shards=4, config=cfg)
    return rs.estimate(rs.read_headers(d), box_chips=4)


def test_a_header_estimate_lists_bf16_bfp8_and_bfp4_for_the_whole_model(flash):
    rows = flash["estimated"]["precision_options"]["rows"]
    assert [r["precision"] for r in rows] == ["bf16", "bfp8", "bfp4"]
    b = [r["device_bytes"] for r in rows]
    assert b[0] > b[1] > b[2] > 0
    for r in rows:
        assert r["chips_needed"] == chips_for(r["device_bytes"])
    # the host-resident n-gram table stays on the host in every row, as in the policy
    assert flash["estimated"]["host_bytes"] > 0


def test_the_options_follow_the_dtype_widths_of_the_policy_table(dense):
    by = dense["measured"]["by_category"]
    floats = sum(c["bytes"] for c in by.values())            # the dense fixture is all BF16, norms included
    bf16 = option(dense, "bf16")["device_bytes"]
    assert 0 < bf16 <= floats
    assert option(dense, "bfp8")["device_bytes"] < bf16


def test_a_header_estimate_names_the_highest_precision_that_fits_a_smaller_box(flash):
    rows = {r["precision"]: r["chips_needed"] for r in flash["estimated"]["precision_options"]["rows"]}
    smaller = rows["bfp4"]                                    # a box exactly as large as bfp4 needs
    assert rs.shortfall(flash, smaller) is None or "highest listed precision" in rs.shortfall(flash, smaller)
    if flash["estimated"]["chips_needed"] > 1:
        why = rs.shortfall(flash, flash["estimated"]["chips_needed"] - 1)
        assert why and ("the highest listed precision that fits" in why or "at any listed precision" in why)
        best = next((p for p in ("bf16", "bfp8", "bfp4") if rows[p] <= flash["estimated"]["chips_needed"] - 1), None)
        if best:
            assert f"is {best}, which needs {rows[best]} chips" in why
        else:
            assert "does not fit this box at any listed precision" in why


# ---- the mixture-of-experts mark -----------------------------------------------------------------------------------------------

def test_a_model_with_routed_experts_is_marked_and_the_note_says_every_expert_stays_resident(flash):
    moe = flash["estimated"]["mixture_of_experts"]
    assert moe["present"] is True and moe["routed_expert_tensors"] > 0
    assert moe["routed_expert_stored_bytes"] == flash["estimated"]["by_category"]["routed_experts"]["stored_bytes"]
    assert "every expert" in moe["note"] and "resident" in moe["note"]


def test_a_dense_model_is_not_marked(dense):
    assert dense["estimated"]["mixture_of_experts"] == {"present": False}


def test_the_text_table_shows_the_options_and_the_mark(flash, capsys):
    lines = "\n".join(rs._text(flash, []))
    assert "bf16" in lines and "bfp8" in lines and "bfp4" in lines and "mixture of experts" in lines


# ---- first boot against the machine ---------------------------------------------------------------------------------------------

def record(disk=10 * GIB, ram=20 * GIB, download=0):
    return {"label": "estimate", "scan": "ran", "assumptions": [],
            "estimated": {"chips_needed": 1, "device_bytes": 5 * GIB,
                          "first_boot": {"snapshot_local": download == 0, "checkpoint_bytes_to_download": download, "cache_bytes": 5 * GIB,
                                         "disk_bytes": disk, "conversion_in_flight_bytes": 1 * GIB, "host_ram_bytes": ram}}}


def test_the_machine_check_is_quiet_when_disk_and_ram_hold_the_first_boot():
    assert rs.machine_shortfall(record(), free_disk_bytes=11 * GIB, mem_available_bytes=21 * GIB) is None
    assert rs.machine_shortfall(record(), free_disk_bytes=10 * GIB, mem_available_bytes=20 * GIB) is None      # exactly enough


def test_too_little_disk_names_both_numbers_and_is_a_disk_block():
    why = rs.machine_shortfall(record(disk=500 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=None)
    assert "500.0 GiB" in why and "120.0 GiB" in why and "free disk" in why
    assert blocked.block_code({"reason": "blocked: " + why}) == "disk-full"


def test_too_little_ram_names_both_numbers_and_says_the_model_does_not_fit_this_box():
    why = rs.machine_shortfall(record(ram=300 * GIB), free_disk_bytes=None, mem_available_bytes=100 * GIB)
    assert "300.0 GiB" in why and "100.0 GiB" in why and "host RAM" in why
    assert why.startswith("the model does not fit this box")
    assert blocked.block_code({"reason": "blocked: " + why}) == "model-does-not-fit"


def test_a_download_is_counted_in_the_disk_sentence():
    why = rs.machine_shortfall(record(disk=500 * GIB, download=400 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=None)
    assert "download" in why


def test_an_unknown_reading_is_no_claim_and_a_damaged_record_is_no_claim():
    assert rs.machine_shortfall(record(disk=10 ** 15, ram=10 ** 15), free_disk_bytes=None, mem_available_bytes=None) is None
    assert rs.machine_shortfall({"scan": "skipped"}, free_disk_bytes=1, mem_available_bytes=1) is None
    bad = record()
    bad["estimated"]["first_boot"]["disk_bytes"] = "lots"
    assert rs.machine_shortfall(bad, free_disk_bytes=1, mem_available_bytes=1) is None
    bad["estimated"]["first_boot"] = None
    assert rs.machine_shortfall(bad, free_disk_bytes=1, mem_available_bytes=1) is None


def test_machine_facts_reads_the_filesystem_of_the_path_and_never_raises(tmp_path):
    free, mem = REAL_MACHINE_FACTS(tmp_path)
    assert isinstance(free, int) and free > 0
    assert mem is None or (isinstance(mem, int) and mem > 0)
    assert REAL_MACHINE_FACTS(tmp_path / "does" / "not" / "exist")[0] is None
    assert REAL_MACHINE_FACTS(None)[0] is None


# ---- preflight --------------------------------------------------------------------------------------------------------------------

def hub_payload(parameters=HYDROGEN):
    return {"id": "deep-principle/Hydrogen-1.0", "sha": "abc", "siblings": [{"rfilename": "LICENSE", "size": 10}], "tags": ["license:mit"],
            "safetensors": {"parameters": parameters, "total": sum(parameters.values()) if isinstance(parameters, dict) else 0}}


def test_the_hub_parse_keeps_the_parameter_counts_and_drops_anything_else():
    assert pf.parse_hub(hub_payload()).parameters == HYDROGEN
    assert pf.parse_hub({"id": "a/b"}).parameters is None
    for bad in ("x", [], {"BF16": "7"}, {"BF16": True}, {"BF16": -3}, {3: 3}, {}):
        assert pf.parse_hub({"id": "a/b", "safetensors": {"parameters": bad}}).parameters is None
    assert pf.parse_hub({"id": "a/b", "safetensors": "x"}).parameters is None
    assert pf.parse_hub({"id": "a/b", "safetensors": {"parameters": {f"X{i}": 1 for i in range(65)}}}).parameters is None
    assert pf.parse_hub({"id": "a/b", "safetensors": {"parameters": {f"X{i}": 1 for i in range(64)}}}).parameters is not None


FOUR_CHIPS = """grain: board   (2 boards, 4 chips)
board 0000046131924062  (p300c)
  chip 0  0000:01:00.0  FREE
  chip 1  0000:02:00.0  FREE
board 0000046131924055  (p300c)
  chip 2  0000:03:00.0  FREE
  chip 3  0000:04:00.0  FREE
"""


def test_the_fit_row_blocks_hydrogen_before_any_download_with_the_numbers():
    hub = pf.parse_hub(hub_payload())
    c = pf.check_fit(hub, 4)
    assert (c.name, c.status, c.reason) == ("fit", "block", "model-does-not-fit")
    assert "at least 14 chips" in c.detail and "this box has 4 chips" in c.detail and "at any listed precision" in c.detail


def test_the_fit_row_passes_a_model_the_bound_does_not_rule_out_and_says_it_is_a_bound():
    hub = pf.parse_hub(hub_payload({"BF16": 8_000_000_000}))
    c = pf.check_fit(hub, 4)
    assert c.status == "ok" and "bound" in c.detail.lower() and "no tensor name" in c.detail


@pytest.mark.parametrize("hub,chips", [(None, 4), ("nocounts", 4), ("params", None), ("params", 0)])
def test_the_fit_row_makes_no_claim_without_counts_or_a_chip_count(hub, chips):
    h = {None: None, "nocounts": pf.parse_hub({"id": "a/b"}), "params": pf.parse_hub(hub_payload())}[hub]
    assert pf.check_fit(h, chips) is None


def _cfg(tmp_path):
    from orchard.bringup_config import BringupConfig, Coder
    return BringupConfig(runs_root=tmp_path / "runs", coder=Coder(target="t", port=8000, chips=4),
                         tiers=tmp_path / "tiers.toml", hf_home=tmp_path / "hf", cache_root=tmp_path / "cache")


def test_run_preflight_adds_the_fit_row_after_gozer_only_when_the_hub_gave_counts(tmp_path):
    import test_preflight as tp
    s = tp.signals(hub_info=lambda m: (pf.parse_hub(hub_payload()), None), gozer_status=lambda: FOUR_CHIPS, free_gb=lambda p: 5000.0)
    out = pf.run_preflight(_cfg(tmp_path), "deep-principle/Hydrogen-1.0", accept_credentials=False, signals=s)
    names = [c.name for c in out]
    assert names.index("fit") == names.index("gozer") + 1
    fit = next(c for c in out if c.name == "fit")
    assert fit.status == "block" and fit.reason == "model-does-not-fit"
    assert "fit" in [c.name for c in pf.blocked(out)]
    plain = pf.run_preflight(_cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=tp.signals())
    assert "fit" not in [c.name for c in plain]


def test_a_lab_run_does_not_judge_the_fit_by_this_box(tmp_path):
    import test_preflight as tp
    cfg = _cfg(tmp_path)
    cfg.mode = "lab"
    s = tp.signals(hub_info=lambda m: (pf.parse_hub(hub_payload()), None), gozer_status=lambda: FOUR_CHIPS, free_gb=lambda p: 5000.0)
    out = pf.run_preflight(cfg, "deep-principle/Hydrogen-1.0", accept_credentials=False, signals=s)
    assert "fit" not in [c.name for c in out]
