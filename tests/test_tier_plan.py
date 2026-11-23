# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard tier-plan` (issue 739, step 1): what a mixture-of-experts model needs on N chips, and when some experts have to live in host
memory, worked out from the model's config.json before any weight is downloaded. Every expected number below is worked out by hand from the
tiny config at the top of the file."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import tier_plan

GIB = float(1 << 30)
ROOT = Path(__file__).resolve().parent.parent

# hidden 64, 2 layers, 4 heads of 16, 4 KV heads, vocab 100, 8 experts of width 32 (top 2), untied head
CFG = {"model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
       "head_dim": 16, "vocab_size": 100, "intermediate_size": 128, "moe_intermediate_size": 32, "num_experts": 8,
       "num_experts_per_tok": 2, "tie_word_embeddings": False}
# by hand: attention per layer 64*64 (q) + 2*64*64 (k, v) + 64*64 (o) = 16384; one expert 3*64*32 = 6144; router 64*8 = 512
ATTN, EXPERT, ROUTER, EMBED, HEAD = 2 * 16384, 6144, 2 * 512, 100 * 64, 100 * 64
ALWAYS = ATTN + ROUTER + EMBED + HEAD                    # 46,592 parameters
INDEX = 2 * EXPERT                                       # one expert index in both layers: 12,288 parameters
ALL_EXPERTS = 8 * INDEX                                  # 98,304 parameters


def test_the_shape_counts_each_group_of_parameters_from_the_config():
    s = tier_plan.shape(CFG)
    assert (s.attention_params, s.router_params, s.embedding_params, s.head_params) == (ATTN, ROUTER, EMBED, HEAD)
    assert s.dense_mlp_params == 0 and s.shared_params == 0
    assert s.always_params == ALWAYS
    assert (s.experts, s.top_k, s.moe_layers) == (8, 2, 2)
    assert s.expert_index_params == INDEX and s.expert_params_total == ALL_EXPERTS


def test_a_shared_expert_and_a_dense_layer_count_as_always_resident():
    cfg = dict(CFG, shared_expert_intermediate_size=16, mlp_only_layers=[0])
    s = tier_plan.shape(cfg)
    assert s.shared_params == 3 * 64 * 16                 # one MoE layer has the shared expert
    assert s.dense_mlp_params == 3 * 64 * 128             # layer 0 is a plain MLP
    assert s.moe_layers == 1 and s.expert_index_params == EXPERT and s.router_params == 512


def test_tied_embeddings_store_the_table_once():
    s = tier_plan.shape(dict(CFG, tie_word_embeddings=True))
    assert s.head_params == 0 and s.embedding_params == EMBED


def test_a_dense_model_is_refused_with_the_command_that_handles_it():
    cfg = {k: v for k, v in CFG.items() if k not in ("num_experts", "num_experts_per_tok", "moe_intermediate_size")}
    with pytest.raises(tier_plan.TierPlanError, match="no routed experts.*tt-orchard plan"):
        tier_plan.shape(cfg)


def test_a_family_whose_layers_cannot_be_counted_says_unknown_family():
    cfg = dict(CFG, layer_types=["full_attention", "hyena"])
    with pytest.raises(tier_plan.TierPlanError, match="unknown family.*hyena"):
        tier_plan.shape(cfg)


def test_a_config_missing_a_number_names_the_key():
    cfg = {k: v for k, v in CFG.items() if k != "hidden_size"}
    with pytest.raises(tier_plan.TierPlanError, match="hidden_size"):
        tier_plan.shape(cfg)


def test_the_weight_budget_per_chip_takes_the_reserve_and_a_share_of_the_kv_cache_off_the_usable_dram():
    # 32 GiB chip, 93.4% usable, 3 GiB reserve, 4 GiB of KV cache spread over 2 chips
    got = tier_plan.budget_bytes(32.0, chips=2, reserve_gib=3.0, kv_gib=4.0)
    assert got == pytest.approx((32.0 * 0.934 - 3.0 - 4.0 / 2) * GIB)


def test_a_reserve_larger_than_the_chip_gives_a_zero_budget_not_a_negative_one():
    assert tier_plan.budget_bytes(4.0, chips=1, reserve_gib=3.0, kv_gib=4.0) == 0.0


BF16 = ("bf16", "bf16")


def place(chips, budget, fmts=BF16, **kw):
    return tier_plan.place(tier_plan.shape(CFG), fmts[0], fmts[1], chips, budget, **kw)


def test_everything_that_fits_one_chip_is_replicated():
    r = place(1, 289792)                                   # (46,592 + 98,304) x 2 bytes
    assert r["verdict"] == "fits on chips" and r["placement"] == "replicated"
    assert r["resident_experts_per_layer"] == 8 and r["host_ram_gib"] == 0


def test_experts_split_over_the_chips_when_replication_does_not_fit():
    r = place(2, 200000)                                   # 93,184 + 196,608 / 2 = 191,488 per chip
    assert r["verdict"] == "fits on chips" and r["placement"] == "experts split over the chips"
    assert r["per_chip_bytes"] == 191488 and r["resident_experts_per_layer"] == 8


def test_experts_that_do_not_fit_the_chips_stay_in_host_memory_and_the_resident_count_is_whole_experts():
    r = place(2, 150000)                                   # 150,000 - 93,184 = 56,816 per chip; two chips hold 4 of the 24,576-byte expert indexes
    assert r["verdict"] == "fits with host experts" and r["placement"] == "hot experts on chips, the rest in host memory"
    assert r["resident_experts_per_layer"] == 4
    assert r["host_ram_gib"] == pytest.approx((196608 + 10 * GIB) / GIB)       # every expert is pinned in host memory, plus the margin


def test_the_host_margin_is_a_flag():
    r = place(2, 150000, host_margin_gib=2.0)
    assert r["host_ram_gib"] == pytest.approx((196608 + 2 * GIB) / GIB)


def test_weights_that_always_stay_on_a_chip_but_do_not_fit_one_are_a_no():
    r = place(4, 90000)                                    # the 93,184 bytes of attention, routers and tables alone exceed a chip
    assert r["verdict"] == "does not fit" and r["why"].startswith("the always-resident weights") and "need" in r["why"]


def test_no_room_for_a_single_expert_beyond_the_always_resident_weights_is_a_no():
    r = place(1, 100000)                                   # 6,816 spare bytes against 24,576 for one expert index
    assert r["verdict"] == "does not fit" and "one expert" in r["why"]


def test_a_smaller_format_for_the_experts_brings_the_model_onto_the_chips():
    # experts 98,304 x 0.5625 = 55,296; replicated that is 93,184 + 55,296 = 148,480 (too much for 140,000); split over 2 it is 93,184 + 27,648 = 120,832
    r = place(2, 140000, ("bf16", "bfp4"))
    assert r["verdict"] == "fits on chips" and r["placement"] == "experts split over the chips" and r["per_chip_bytes"] == pytest.approx(120832)
    assert place(2, 150000, ("bf16", "bfp4"))["placement"] == "replicated"


def test_host_ram_is_compared_with_the_ram_the_caller_says_is_free():
    ok = place(2, 150000, host_free_gib=64.0)
    no = place(2, 150000, host_free_gib=5.0)
    assert ok["verdict"] == "fits with host experts"
    assert no["verdict"] == "does not fit" and "host memory" in no["why"]


# ---- the share of expert picks that land on a resident expert ----------------------------------------------------------

def test_the_hit_share_goes_through_the_measured_points_and_through_zero_and_one():
    pts = tier_plan.OLMOE_SKEW
    assert tier_plan.hit_share(0.0, pts) == 0.0 and tier_plan.hit_share(1.0, pts) == 1.0
    assert tier_plan.hit_share(0.25, pts) == pytest.approx(pts[1][1])
    mid = tier_plan.hit_share(0.375, pts)
    assert pts[1][1] < mid < pts[2][1]


def test_a_uniform_router_gives_a_hit_share_equal_to_the_resident_fraction():
    assert tier_plan.hit_share(0.3, ()) == pytest.approx(0.3)


def test_the_default_skew_points_are_the_ones_in_the_committed_olmoe_measurement():
    data = json.loads((ROOT / "docs/run-logs/2026-11-11-expert-skew-olmoe-1b-7b.json").read_text())["mean"]
    for frac, share in tier_plan.OLMOE_SKEW:
        assert share == pytest.approx(data[f"top_{round(frac * 100)}pct_share"], abs=0.001)


@pytest.mark.parametrize("text", ["0.25:0.10", "0.5:0.6,0.25:0.7", "0.25:0.5,0.5:0.4", "1.5:0.9", "a:b", "0.25", ""])
def test_a_skew_that_cannot_be_true_is_refused(text):
    with pytest.raises(tier_plan.TierPlanError):
        tier_plan.parse_skew(text)


def test_a_skew_string_reads_into_sorted_points():
    assert tier_plan.parse_skew("0.10:0.41,0.25:0.61") == ((0.10, 0.41), (0.25, 0.61))


def test_the_row_reports_the_hit_share_as_an_estimate_with_its_source():
    r = place(2, 150000)                                   # half the experts resident
    assert r["hit_share"] == pytest.approx(tier_plan.OLMOE_SKEW[2][1])
    assert "estimate" in r["hit_share_label"] and "OLMoE" in r["hit_share_label"] and "not measured for this model" in r["hit_share_label"]


def test_a_skew_the_caller_gives_is_labelled_as_theirs():
    r = place(2, 150000, skew=((0.5, 0.9),), skew_source="from a router count on this model's own text")
    assert r["hit_share"] == pytest.approx(0.9) and "own text" in r["hit_share_label"] and "OLMoE" not in r["hit_share_label"]


def test_misses_per_token_follow_from_the_picks_the_layers_and_the_hit_share():
    r = place(2, 150000)
    assert r["host_expert_picks_per_token"] == pytest.approx(2 * 2 * (1 - r["hit_share"]))      # top_k x moe layers x miss share


# ---- reading the source: a local config, a directory, or a repository's config.json and weight index -------------------

class FakeHub:
    """Stands in for drop_check.HubClient and records every file asked for."""
    def __init__(self, files):
        self.files, self.asked = files, []

    def get(self, repo, revision, name, *, max_bytes):
        self.asked.append((repo, revision, name))
        return self.files.get(name)


def test_a_local_config_file_is_read(tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps(CFG))
    cfg, total, where = tier_plan.load(str(f))
    assert cfg == CFG and total is None and "config.json" in where


def test_a_directory_gives_its_config_and_its_weight_index_total(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CFG))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 123456}, "weight_map": {"a": "x.safetensors"}}))
    cfg, total, _ = tier_plan.load(str(tmp_path))
    assert cfg == CFG and total == 123456


def test_a_repository_id_reads_only_the_config_and_the_index_and_never_a_weight_file():
    hub = FakeHub({"config.json": json.dumps(CFG).encode(),
                   "model.safetensors.index.json": json.dumps({"metadata": {"total_size": 99}, "weight_map": {"w": "model-00001.safetensors"}}).encode()})
    cfg, total, where = tier_plan.load("owner/some-model", client=hub, revision="main")
    assert cfg == CFG and total == 99 and "owner/some-model" in where
    assert sorted(n for _, _, n in hub.asked) == ["config.json", "model.safetensors.index.json"]
    assert not any(n.endswith((".safetensors", ".bin", ".gguf")) for _, _, n in hub.asked)


def test_a_repository_without_an_index_still_plans():
    hub = FakeHub({"config.json": json.dumps(CFG).encode()})
    cfg, total, _ = tier_plan.load("owner/some-model", client=hub)
    assert cfg == CFG and total is None


def test_a_repository_with_no_config_says_so_plainly():
    with pytest.raises(tier_plan.TierPlanError, match="no config.json"):
        tier_plan.load("owner/some-model", client=FakeHub({}))


def test_something_that_is_neither_a_path_nor_a_repository_id_is_refused(tmp_path):
    with pytest.raises(tier_plan.TierPlanError, match="not a file, a directory or a Hugging Face repository id"):
        tier_plan.load(str(tmp_path / "missing" / "deeper" / "x"))


def test_a_config_that_is_a_link_is_refused(tmp_path):
    real = tmp_path / "real.json"
    real.write_text(json.dumps(CFG))
    link = tmp_path / "config.json"
    link.symlink_to(real)
    with pytest.raises(tier_plan.TierPlanError, match="not usable"):
        tier_plan.load(str(link))


def test_a_config_over_the_size_limit_is_refused(tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"pad": "x" * (2 << 20)}))
    with pytest.raises(tier_plan.TierPlanError, match="not usable"):
        tier_plan.load(str(f))


def test_a_hub_that_cannot_be_reached_is_one_plain_sentence():
    from orchard import drop_check

    class Down:
        def get(self, *a, **k):
            raise drop_check.HubUnreachable("the Hub did not answer (URLError)")
    with pytest.raises(tier_plan.TierPlanError, match="did not answer"):
        tier_plan.load("owner/some-model", client=Down())


def test_an_index_total_near_the_bf16_size_adds_no_note_and_a_far_one_does():
    s = tier_plan.shape(CFG)
    bf16 = (s.always_params + s.expert_params_total) * 2
    assert tier_plan.checkpoint_note(s, bf16) is None
    assert tier_plan.checkpoint_note(s, None) is None
    far = tier_plan.checkpoint_note(s, bf16 // 2)
    assert far and "50%" in far and "different format" in far


# ---- the whole plan, its labels and its text ----------------------------------------------------------------------------

def test_the_plan_has_a_row_for_every_precision_and_chip_count_and_labels_each_source():
    res = tier_plan.plan(CFG, chips_list=(1, 2, 4), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    assert {r["chips"] for r in res["rows"]} == {1, 2, 4}
    assert {r["name"] for r in res["rows"]} == {"all bf16", "all bfp8", "bfp8 + bfp4 experts", "all bfp4"}
    assert len(res["rows"]) == 12
    kinds = {v["source"] for v in res["inputs"].values()}
    assert kinds <= {"config-derived", "assumed constant", "caller's flag", "estimate"} and {"config-derived", "assumed constant", "estimate"} <= kinds
    assert res["inputs"]["expert_pick_skew"]["source"] == "estimate"
    assert all(r["verdict"] in ("fits on chips", "fits with host experts", "does not fit") for r in res["rows"])
    assert res["model"]["routed_experts_per_layer"] == 8 and res["model"]["moe_layers"] == 2


def test_a_tiny_model_fits_every_row_on_a_real_chip():
    res = tier_plan.plan(CFG, chips_list=(1,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    assert {r["verdict"] for r in res["rows"]} == {"fits on chips"}


def test_a_flag_that_changes_a_constant_is_marked_as_the_callers():
    res = tier_plan.plan(CFG, chips_list=(1,), chip_gib=24.0, reserve_gib=3.0, kv_gib=4.0, flags={"chip_gib"})
    assert res["inputs"]["chip_gib"]["source"] == "caller's flag" and res["inputs"]["reserve_gib"]["source"] == "assumed constant"


def test_the_fidelity_rule_from_the_weights_plan_is_carried_on_each_row():
    res = tier_plan.plan(CFG, chips_list=(1,), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    gates = {r["name"]: r["format_gate"] for r in res["rows"]}
    assert gates["all bf16"] == "ok" and gates["all bfp8"] == "ok" and gates["all bfp4"] == "unscored"


def test_the_text_names_each_verdict_and_says_which_numbers_are_estimates():
    cfg = dict(CFG, hidden_size=4096, moe_intermediate_size=2048, intermediate_size=2048, num_experts=256, num_hidden_layers=40, vocab_size=150000,
               num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    res = tier_plan.plan(cfg, chips_list=(1, 2, 4), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0, host_free_gib=200.0)
    text = tier_plan.render(res)
    assert "fits with host experts" in text and "does not fit" in text
    assert "estimate" in text and "config-derived" in text and "assumed constant" in text
    assert "not measured for this model" in text


def test_the_plan_never_imports_ttnn_or_touches_the_network_for_a_local_config(tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps(CFG))
    code = ("import sys, socket\n"
            "def boom(*a, **k): raise SystemExit('network used')\n"
            "socket.socket.connect = boom\n"
            "from orchard import tier_plan\n"
            f"tier_plan.main([{str(f)!r}, '--json'])\n"
            "assert 'ttnn' not in sys.modules and 'ttml' not in sys.modules\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_the_command_prints_json_and_returns_zero(tmp_path, capsys):
    f = tmp_path / "config.json"
    f.write_text(json.dumps(CFG))
    assert tier_plan.main([str(f), "--chips", "1,2", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {r["chips"] for r in out["rows"]} == {1, 2} and out["model"]["routed_experts_per_layer"] == 8


@pytest.mark.parametrize("argv", [["--chips", "0"], ["--chips", "1,x"], ["--chips", "65"], ["--chip-gib", "0"], ["--reserve-gib", "-1"],
                                  ["--kv-gib", "-1"], ["--host-margin-gib", "-1"], ["--host-free-gib", "0"], ["--skew", "0.5:0.1"]])
def test_a_bad_flag_is_refused_with_exit_2_and_one_line(tmp_path, capsys, argv):
    f = tmp_path / "config.json"
    f.write_text(json.dumps(CFG))
    assert tier_plan.main([str(f), *argv]) == 2
    err = capsys.readouterr().err.strip()
    assert err and "\n" not in err and "Traceback" not in err


def test_a_dense_model_exits_2_with_the_hint(tmp_path, capsys):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({k: v for k, v in CFG.items() if k not in ("num_experts", "num_experts_per_tok", "moe_intermediate_size")}))
    assert tier_plan.main([str(f)]) == 2
    assert "tt-orchard plan" in capsys.readouterr().err


def test_the_subcommand_is_registered_in_the_cli():
    r = subprocess.run([sys.executable, "-m", "orchard.cli", "tier-plan", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "FRACTION:SHARE" in r.stdout


# ---- the Hub reader: redirects stay on the repository asked for ---------------------------------------------------------

class Resp:
    def __init__(self, body: bytes, final: str):
        self.body, self.final = body, final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def geturl(self):
        return self.final

    def read(self, n=-1):
        return self.body[:n] if n >= 0 else self.body


def reader_with(final_for, body=b"{}"):
    asked = []

    def opener(req, timeout=None):
        asked.append(req.full_url)
        return Resp(body, final_for(req.full_url))
    return tier_plan.HubReader(opener=opener), asked


def test_the_hub_answers_a_config_request_with_a_redirect_inside_the_same_repository_and_that_is_read():
    # what the Hub really does: .../resolve/main/config.json -> /api/resolve-cache/models/<repo>/<commit>/config.json
    final = "https://huggingface.co/api/resolve-cache/models/some-org/some-model/" + "a" * 40 + "/config.json"
    rd, asked = reader_with(lambda url: final, body=b'{"a": 1}')
    assert rd.get("some-org/some-model", "main", "config.json", max_bytes=1000) == b'{"a": 1}'
    assert asked == ["https://huggingface.co/some-org/some-model/resolve/main/config.json"]


def test_a_redirect_to_a_content_host_of_the_hub_is_read():
    rd, _ = reader_with(lambda url: "https://cas-bridge.xethub.hf.co/some/path/config.json", body=b"{}")
    assert rd.get("some-org/some-model", "main", "config.json", max_bytes=1000) == b"{}"


@pytest.mark.parametrize("final", ["https://huggingface.co/api/resolve-cache/models/other-org/other-model/" + "a" * 40 + "/config.json",
                                   "https://huggingface.co/other-org/other-model/resolve/main/config.json",
                                   "https://evil.example/some-org/some-model/config.json",
                                   "http://huggingface.co/some-org/some-model/resolve/main/config.json",
                                   "https://user@huggingface.co/some-org/some-model/resolve/main/config.json"])
def test_a_redirect_to_another_repository_or_host_is_refused_before_anything_is_read(final):
    rd, _ = reader_with(lambda url: final)
    from orchard import drop_check
    with pytest.raises(drop_check.RedirectRefused):
        rd.get("some-org/some-model", "main", "config.json", max_bytes=1000)


@pytest.mark.parametrize("name", ["model-00001-of-00100.safetensors", "pytorch_model.bin", "../config.json", "tokenizer.json"])
def test_the_reader_will_not_ask_for_any_other_file(name):
    rd, asked = reader_with(lambda url: url)
    with pytest.raises(ValueError, match="only config.json"):
        rd.get("some-org/some-model", "main", name, max_bytes=1000)
    assert asked == []


def test_a_response_over_the_size_cap_is_refused():
    from orchard import drop_check
    rd, _ = reader_with(lambda url: url, body=b"x" * 100)
    with pytest.raises(drop_check.TooLarge):
        rd.get("some-org/some-model", "main", "config.json", max_bytes=10)


# ---- the second layout: whole layers on the chips (what the published hybrid bundles do) -------------------------------

def test_the_whole_layer_layout_keeps_complete_layers_on_the_chips_and_leaves_the_rest_to_the_host():
    r = place(2, 150000)             # the chips hold 113,632 bytes beside the always-resident weights; one layer's 8 experts take 98,304 (8 x 6,144 x 2)
    wl = r["whole_layers"]
    assert wl["layers_on_chips"] == 1 and wl["moe_layers"] == 2
    assert wl["host_ram_gib"] == pytest.approx((98304 + 10 * GIB) / GIB)         # the experts of the one layer left to the host, plus the margin
    assert wl["hit_share"] == pytest.approx(0.5)                                   # every layer takes the same number of picks: one layer of two
    assert "every layer takes the same number of picks" in wl["hit_share_label"]


def test_a_row_that_fits_on_chips_has_no_whole_layer_figures():
    assert place(1, 289792)["whole_layers"] is None


def test_the_uniform_layout_keeps_the_all_experts_pinned_host_figure_and_the_whole_layer_layout_asks_for_less():
    r = place(2, 150000)
    assert r["host_ram_gib"] > r["whole_layers"]["host_ram_gib"] - 1e-9
    assert r["host_ram_gib"] == pytest.approx((196608 + 10 * GIB) / GIB)


def test_the_whole_layer_figure_is_checked_against_free_host_memory_too():
    r = place(2, 150000, host_free_gib=64.0)
    assert r["whole_layers"]["host_ram_gib"] <= 64.0
    # a plan that fits on neither layout is a no, so the free-memory check applies to the smaller of the two needs
    no = place(2, 150000, host_free_gib=5.0)
    assert no["verdict"] == "does not fit"


def test_the_text_prints_the_whole_layer_layout_for_rows_that_need_the_host():
    cfg = dict(CFG, hidden_size=4096, moe_intermediate_size=2048, intermediate_size=2048, num_experts=256, num_hidden_layers=40, vocab_size=150000,
               num_attention_heads=32, num_key_value_heads=8, head_dim=128)
    text = tier_plan.render(tier_plan.plan(cfg, chips_list=(1, 2), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0))
    assert "whole layers on chips" in text and "every layer takes the same number of picks" in text


def test_a_row_that_only_the_whole_layer_layout_can_fit_in_free_host_memory_says_so():
    r = place(2, 150000, host_margin_gib=0.0, host_free_gib=1.2e-4)      # one layer's experts 9.2e-5 GiB, all experts 1.8e-4 GiB
    assert r["verdict"] == "fits with host experts"
    assert r["uniform_fits_host_ram"] is False and r["whole_layers"]["fits_host_ram"] is True
    res = {"source": "x", "model": {"layers": 2, "moe_layers": 2, "routed_experts_per_layer": 8, "experts_per_token": 2, "always_resident_params": 1,
                                    "routed_expert_params_total": 1, "params_per_expert_index": 1},
           "inputs": {"chip_gib": {"value": 1}, "usable_fraction": {"value": 0.9}, "reserve_gib": {"value": 0}, "kv_gib": {"value": 0}, "host_margin_gib": {"value": 0}},
           "rows": [dict(r, name="all bf16", format_gate="ok")], "notes": []}
    text = tier_plan.render(res)
    assert "more than the free host memory" in text
    assert re.search(r"\d!\s", text)                      # the row's host figure carries the mark


def test_the_whole_layer_layout_never_puts_every_layer_on_the_chips_in_a_row_that_needs_the_host():
    s = tier_plan.shape(CFG)
    for chips in (1, 2, 3, 4, 8):
        for budget in range(90000, 400000, 3001):
            for fmts in (BF16, ("bfp8", "bfp4"), ("bfp4", "bfp4")):
                r = tier_plan.place(s, fmts[0], fmts[1], chips, budget)
                if r["verdict"] == "fits with host experts":
                    assert 0 <= r["whole_layers"]["layers_on_chips"] < r["whole_layers"]["moe_layers"]


def test_a_weight_index_whose_total_is_a_boolean_or_negative_is_not_a_size():
    assert tier_plan._index_total(b'{"metadata": {"total_size": true}}') is None
    assert tier_plan._index_total(b'{"metadata": {"total_size": -5}}') is None
    assert tier_plan._index_total(b'{"metadata": {"total_size": 7}}') == 7
    assert tier_plan._index_total(b"not json") is None


@pytest.mark.parametrize("repo,rev", [("../etc/passwd", "main"), ("owner/../x", "main"), ("owner/model", "../../x"), ("owner/model", "refs/pr/1"), ("owner/model", 5)])
def test_the_reader_refuses_a_repository_or_revision_that_could_change_the_url(repo, rev):
    rd, asked = reader_with(lambda url: url)
    with pytest.raises(ValueError):
        rd.get(repo, rev, "config.json", max_bytes=1000)
    assert asked == []


# ---- hostile numbers in a config (hardening checklist items 1 and 9) --------------------------------------------------

KEYS = ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
        "moe_intermediate_size", "num_experts", "num_experts_per_tok", "shared_expert_intermediate_size"]
BAD = [None, 0, -1, 1.5, "x", [], {}, True, 10 ** 20, 10 ** 400, float("1e308")]


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("bad", BAD, ids=[type(b).__name__ + (str(len(str(b))) if isinstance(b, int) and not isinstance(b, bool) else "") for b in BAD])
def test_a_hostile_number_in_the_config_is_a_refusal_or_a_plan_and_never_a_raw_exception(key, bad):
    cfg = dict(CFG, shared_expert_intermediate_size=16)
    cfg[key] = bad
    try:
        res = tier_plan.plan(json.loads(json.dumps(cfg, allow_nan=False)) if bad != float("1e308") else cfg,
                             chips_list=(1, 2, 4), chip_gib=32.0, reserve_gib=3.0, kv_gib=4.0)
    except tier_plan.TierPlanError as exc:
        assert "\n" not in str(exc)
        return
    assert all(r["verdict"] in ("fits on chips", "fits with host experts", "does not fit") for r in res["rows"])
    json.dumps(res, allow_nan=False)          # a plan can always be printed as strict JSON


@pytest.mark.parametrize("bad", ["nan", "inf", "1e400", "-1", "0"])
def test_a_hostile_number_on_the_command_line_is_exit_2(tmp_path, capsys, bad):
    f = tmp_path / "config.json"
    f.write_text(json.dumps(CFG))
    for flag in ("--chip-gib", "--host-free-gib"):
        assert tier_plan.main([str(f), flag, bad]) == 2
    capsys.readouterr()
