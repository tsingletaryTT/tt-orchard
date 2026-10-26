# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `base` preflight row when no installed bundle serves the base but tt-transformers supports the model (issue 61).

With an exact `model_params` match in the research report and that directory on this box, the row is a warning and the run
goes on as stock-runtime. No flag is needed. Without the match, or without the directory, the earlier block stands."""
from pathlib import Path

import pytest

from orchard import preflight as pf
from orchard.nearest import Installed
from test_cli import conf  # noqa: F401  (the fixture)
from test_research_preflight import check, hub, report

ROW_PARAMS = {"name": "Llama-3.1-8B-Instruct", "via": "model_params", "matched": "x/humanizer", "boards": [], "blackhole": None}
ROW_README = {"name": "Llama 3.1 8B", "via": "readme", "matched": "x/humanizer", "boards": ["N150", "P100", "P150"], "blackhole": True}
BASE = "meta-llama/Llama-3.1-8B-Instruct"


def stock_check(rep, *, found="/host/models/tt_transformers/model_params/Llama-3.1-8B-Instruct", **kw):
    seen = []

    def stock(name):
        seen.append(name)
        return Path(found) if found else None
    c = pf.check_base(model_id="x/humanizer", hub=kw.pop("h", hub(base_model=BASE)), base_override=kw.pop("override", None),
                      local_snapshot=lambda m: None, installed=list(kw.pop("installed", ())), search=lambda q: [], report=rep,
                      accept_inferred=False, stock=stock, **kw)
    return c, seen


def test_an_exact_params_match_on_this_box_is_a_warning_that_names_the_support():
    c, seen = stock_check(report(tt_transformers=[ROW_PARAMS, ROW_README]))
    assert c.status == pf.WARN and c.reason is None
    assert "tt-transformers supports Llama-3.1-8B-Instruct" in c.detail
    assert "N150, P100, P150" in c.detail and "names Blackhole boards" in c.detail
    assert seen == ["Llama-3.1-8B-Instruct"]
    assert c.data["stock_runtime"] == {"name": "Llama-3.1-8B-Instruct", "params_dir": "/host/models/tt_transformers/model_params/Llama-3.1-8B-Instruct",
                                       "matched": "x/humanizer", "blackhole": True}
    assert c.data["bundle"] is None and c.data["snapshot"] is None and c.data["needs_fetch"] is False


def test_a_row_with_no_blackhole_board_still_warns_and_says_stage_two_decides():
    row = {**ROW_README, "boards": ["N150"], "blackhole": False}
    c, _ = stock_check(report(tt_transformers=[ROW_PARAMS, row]))
    assert c.status == pf.WARN
    assert "names no Blackhole board" in c.detail and "stage 2" in c.detail and c.data["stock_runtime"]["blackhole"] is False


def test_a_match_with_no_readme_row_has_unknown_boards():
    c, _ = stock_check(report(tt_transformers=[ROW_PARAMS]))
    assert c.status == pf.WARN and "no README row" in c.detail and c.data["stock_runtime"]["blackhole"] is None


def test_without_the_params_directory_on_this_box_the_block_stands():
    c, _ = stock_check(report(tt_transformers=[ROW_PARAMS]), found=None)
    assert c.status == pf.BLOCK and c.reason == "nearest-model-missing"


def test_a_readme_row_alone_is_not_enough():
    c, seen = stock_check(report(tt_transformers=[ROW_README]))
    assert c.status == pf.BLOCK and seen == []


def test_a_readme_row_matched_for_another_model_is_not_attached_to_the_selected_row():
    other = {**ROW_README, "name": "Llama 3.1 8B", "matched": "meta-llama/Llama-3.1-8B-Instruct", "boards": ["P150"], "blackhole": True}
    row = {**ROW_PARAMS, "name": "Llama-3.2-1B-Instruct"}
    c, _ = stock_check(report(tt_transformers=[row, other]))
    assert c.status == pf.WARN and "no README row" in c.detail and "P150" not in c.detail
    assert c.data["stock_runtime"]["blackhole"] is None
    own = {**ROW_README, "matched": "x/humanizer", "boards": ["N150"], "blackhole": False}
    c, _ = stock_check(report(tt_transformers=[row, other, own]))
    assert "N150" in c.detail and "names no Blackhole board" in c.detail and c.data["stock_runtime"]["blackhole"] is False


def test_a_readme_row_with_a_plain_name_is_still_not_enough():
    c, seen = stock_check(report(tt_transformers=[{**ROW_README, "name": "Llama-3.1-8B"}]))
    assert c.status == pf.BLOCK and seen == []


def test_no_support_record_keeps_the_block():
    assert stock_check(report(tt_transformers=[]))[0].status == pf.BLOCK
    assert stock_check(report())[0].status == pf.BLOCK
    assert stock_check(None)[0].status == pf.BLOCK


def test_without_the_signal_the_block_stands():
    c = check(report(tt_transformers=[ROW_PARAMS]), h=hub(base_model=BASE))
    assert c.status == pf.BLOCK


def test_an_installed_bundle_that_serves_the_base_wins_over_the_stock_path():
    inst = [Installed("tt/llama-p150", BASE, "6", 1)]
    c, seen = stock_check(report(tt_transformers=[ROW_PARAMS]), installed=inst)
    assert c.status == pf.OK and c.data["bundle"] == "tt/llama-p150" and "stock_runtime" not in c.data
    assert seen == []


def test_a_card_with_no_base_still_takes_the_stock_path():
    c, _ = stock_check(report(tt_transformers=[ROW_PARAMS]), h=hub())
    assert c.status == pf.WARN and c.data["stock_runtime"]["name"] == "Llama-3.1-8B-Instruct"


def test_a_name_that_is_not_one_path_part_is_never_looked_up():
    bad = {**ROW_PARAMS, "name": "../x"}
    c, seen = stock_check(report(tt_transformers=[bad]))
    assert c.status == pf.BLOCK and seen == []


def test_a_params_row_that_is_not_a_dict_or_has_no_name_is_skipped():
    c, seen = stock_check(report(tt_transformers=["x", {"via": "model_params"}, ROW_PARAMS]))
    assert c.status == pf.WARN and seen == ["Llama-3.1-8B-Instruct"]


def test_the_warning_text_has_no_home_path():
    c, _ = stock_check(report(tt_transformers=[ROW_PARAMS]))
    assert "/host/" not in c.detail


# ---- the wiring through run_preflight and default_signals ---------------------------------------------------

def test_run_preflight_passes_the_stock_signal_to_the_base_check(tmp_path):
    from test_preflight import cfg, signals
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    s = signals(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name)
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s)
    base = next(c for c in out if c.name == "base")
    assert base.status == pf.WARN and base.data["stock_runtime"]["params_dir"] == "/host/params/Llama-3.1-8B-Instruct"
    assert not pf.blocked(out)


def test_run_preflight_without_the_signal_keeps_the_block(tmp_path):
    from test_preflight import cfg, signals
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals(research=lambda *a: rep))
    assert next(c for c in out if c.name == "base").status in (pf.WARN, pf.BLOCK)
    assert "stock_runtime" not in (next(c for c in out if c.name == "base").data or {})


def test_default_signals_look_for_the_params_directory_in_the_tt_metal_tree(tmp_path, monkeypatch):
    from test_preflight import cfg
    tree = tmp_path / "tt-metal"
    d = tree / "models" / "tt_transformers" / "model_params" / "Llama-3.1-8B-Instruct"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    monkeypatch.setenv("ORCHARD_TT_METAL_DIR", str(tree))
    sig = pf.default_signals(cfg(tmp_path))
    assert sig.stock_params("Llama-3.1-8B-Instruct") == d
    assert sig.stock_params("Nope") is None


def test_default_signals_use_tt_metal_under_the_operator_home_when_nothing_is_set(tmp_path, monkeypatch):
    from test_preflight import cfg
    monkeypatch.delenv("ORCHARD_TT_METAL_DIR", raising=False)
    home = tmp_path / "op"
    d = home / "tt-metal" / "models" / "tt_transformers" / "model_params" / "X-1B"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    c = cfg(tmp_path)
    c.operator_home = home
    assert pf.default_signals(c).stock_params("X-1B") == d


# ---- bringup passes the parameters to the supervisor, with no flag ---------------------------------------------

def _cli_run(conf, extra=(), **over):
    from test_cli import Recorder, run, signals
    snap = conf.parent / "hf" / "hub" / "models--Cloudflare--clef" / "snapshots" / "abc"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text("{}")
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    sig = signals(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name,
                  local_snapshot=lambda m: snap, **over)
    rec = Recorder()
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", *extra], rec=rec, sig=sig)
    return code, out, rec, snap


def test_bringup_passes_stock_params_as_an_input_when_the_base_row_found_support(conf):
    code, out, rec, snap = _cli_run(conf)
    assert code == 0 and rec.argv is not None
    inputs = [rec.argv[i + 1] for i, a in enumerate(rec.argv) if a == "--input"]
    assert "stock_params=/host/params/Llama-3.1-8B-Instruct" in inputs
    assert f"model={snap}" in inputs and not any(i.startswith("base=") for i in inputs)


def test_a_dry_run_shows_the_input_in_its_command_and_downloads_nothing(conf):
    code, out, rec, _ = _cli_run(conf, ["--dry-run"])
    assert code == 0 and rec.argv is None and rec.fetched == []
    assert "stock_params=/host/params/Llama-3.1-8B-Instruct" in out


def test_a_run_with_no_stock_support_passes_no_stock_input(conf):
    from test_cli import Recorder, run, signals
    rec = Recorder()
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], rec=rec, sig=signals())
    assert "stock_params" not in out


# ---- a resume keeps the run's recorded inputs (review of PR 78) ----------------------------------------------------

def _started(conf, inputs):
    from orchard.ledger import Ledger
    run_dir = conf.parent / "runs" / "cloudflare--clef"
    with Ledger(run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Cloudflare/clef", inputs=inputs)
    return run_dir


def _bundle_signals(snap, **over):
    from orchard.nearest import Installed
    from test_cli import signals
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    base = dict(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name, local_snapshot=lambda m: snap,
                installed_bundles=lambda: [Installed("tt/base-p150", "g/base", "6", 1)],
                hub_info=lambda m: (hub(base_model="g/base"), None))
    base.update(over)
    return signals(**base)


def _inputs(rec):
    return [rec.argv[i + 1] for i, a in enumerate(rec.argv) if a == "--input"]


def test_a_resume_keeps_the_recorded_stock_params_when_preflight_now_finds_a_bundle(conf, tmp_path):
    from test_cli import Recorder, run
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    run_dir = _started(conf, {"model": str(snap), "stock_params": "/host/params/Llama-3.1-8B-Instruct"})
    rec = Recorder()
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=rec, sig=_bundle_signals(snap))
    assert code == 0, out
    assert "stock_params=/host/params/Llama-3.1-8B-Instruct" in _inputs(rec)


def test_a_resume_of_a_run_that_started_without_stock_params_does_not_add_them(conf, tmp_path):
    from test_cli import Recorder, run, signals
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    run_dir = _started(conf, {"model": str(snap)})
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    sig = signals(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name, local_snapshot=lambda m: snap)
    rec = Recorder()
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=rec, sig=sig)
    assert code == 0, out
    assert not [i for i in _inputs(rec) if i.startswith("stock_params=")]


def test_a_resume_with_an_unreadable_ledger_adds_no_stock_params(conf, tmp_path):
    from test_cli import Recorder, run, signals
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    run_dir = conf.parent / "runs" / "cloudflare--clef"
    run_dir.mkdir(parents=True)
    (run_dir / "ledger.jsonl").write_text("not a ledger line\n")
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    sig = signals(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name, local_snapshot=lambda m: snap)
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=Recorder(), sig=sig)
    assert not [i for i in _inputs(rec) if i.startswith("stock_params=")]


def _selected_stock_signals(snap):
    from test_cli import signals
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    return signals(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name, local_snapshot=lambda m: snap)


def test_a_ledger_with_no_run_start_keeps_the_route_the_preflight_selected(conf, tmp_path):
    # a first ledger append that failed can leave the file without a run_start; the run never started, so nothing is recorded to keep
    from orchard.ledger import Ledger
    from test_cli import Recorder, run
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    run_dir = conf.parent / "runs" / "cloudflare--clef"
    with Ledger(run_dir / "ledger.jsonl") as led:
        led.append("decision", None, decision="something before run_start")
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=Recorder(), sig=_selected_stock_signals(snap))
    assert code == 0, out
    assert "stock_params=/host/params/Llama-3.1-8B-Instruct" in _inputs(rec)


def test_an_empty_ledger_file_keeps_the_selected_route_too(conf, tmp_path):
    from test_cli import Recorder, run
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    run_dir = conf.parent / "runs" / "cloudflare--clef"
    run_dir.mkdir(parents=True)
    (run_dir / "ledger.jsonl").write_text("")
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=Recorder(), sig=_selected_stock_signals(snap))
    assert code == 0 and "stock_params=/host/params/Llama-3.1-8B-Instruct" in _inputs(rec)


def test_a_started_run_without_stock_params_still_gets_none(conf, tmp_path):
    # the guard above must not weaken the resume rule: a run_start with no stock_params stays without
    from test_cli import Recorder, run
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    run_dir = _started(conf, {"model": str(snap)})
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=Recorder(), sig=_selected_stock_signals(snap))
    assert not [i for i in _inputs(rec) if i.startswith("stock_params=")]


def test_a_run_start_with_no_inputs_counts_as_started(conf, tmp_path):
    from orchard.ledger import Ledger
    from test_cli import Recorder, run
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    run_dir = conf.parent / "runs" / "cloudflare--clef"
    with Ledger(run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, model="Cloudflare/clef")
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=Recorder(), sig=_selected_stock_signals(snap))
    assert not [i for i in _inputs(rec) if i.startswith("stock_params=")]


def test_a_fresh_run_is_not_affected_by_the_resume_rule(conf):
    code, out, rec, snap = _cli_run(conf)
    assert "stock_params=/host/params/Llama-3.1-8B-Instruct" in _inputs(rec)


# ---- the requested model's row first, and a compatible one (second review of PR 78) ---------------------------------

def _dirs(tmp_path, specs):
    """{name: Path} of model_params directories with a config.json each."""
    import json as _j
    out = {}
    for name, cfg in specs.items():
        d = tmp_path / "params" / name
        d.mkdir(parents=True)
        (d / "config.json").write_text(_j.dumps(cfg))
        out[name] = d
    return out


LLAMA_CFG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100}
QWEN_CFG = {**LLAMA_CFG, "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}
ANCESTOR = {"name": "Qwen2.5-7B-Instruct", "via": "model_params", "matched": "ancestor/qwen", "boards": [], "blackhole": None}
OWN = {"name": "Llama-3.1-8B-Instruct", "via": "model_params", "matched": "x/humanizer", "boards": [], "blackhole": None}


def _pick(tmp_path, rows, *, hub_arch=None, snap_cfg=None, card=None):
    import json as _j
    dirs = _dirs(tmp_path, {"Qwen2.5-7B-Instruct": QWEN_CFG, "Llama-3.1-8B-Instruct": LLAMA_CFG})
    snap = None
    if snap_cfg is not None:
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "config.json").write_text(_j.dumps(snap_cfg))
    h = pf.parse_hub({"id": "x/humanizer", "sha": "c0", "cardData": card if card is not None else {"license": "mit", "base_model": BASE},
                      "config": {"architectures": [hub_arch]} if hub_arch else {}, "siblings": []})
    c = pf.check_base(model_id="x/humanizer", hub=h, base_override=None, local_snapshot=lambda m: snap, installed=[], search=lambda q: [],
                      report=report(tt_transformers=rows), stock=lambda name: dirs.get(name))
    return c


def test_the_requested_models_own_row_comes_before_a_row_for_an_ancestor(tmp_path):
    c = _pick(tmp_path, [ANCESTOR, OWN])
    assert c.status == pf.WARN and c.data["stock_runtime"]["name"] == "Llama-3.1-8B-Instruct"


def test_a_row_whose_architecture_differs_from_the_models_is_skipped_for_a_compatible_one(tmp_path):
    # the model is a Llama; the only own-model row is not offered, the ancestor's is Qwen, and the card's base row is Llama
    c = _pick(tmp_path, [ANCESTOR, {**OWN, "matched": "meta-llama/Llama-3.1-8B-Instruct"}], hub_arch="LlamaForCausalLM")
    assert c.status == pf.WARN and c.data["stock_runtime"]["name"] == "Llama-3.1-8B-Instruct"


def test_a_card_with_no_base_also_takes_the_models_own_compatible_row(tmp_path):
    c = _pick(tmp_path, [ANCESTOR, OWN], card={"license": "mit"})
    assert c.status == pf.WARN and c.data["stock_runtime"]["name"] == "Llama-3.1-8B-Instruct"


def test_when_no_row_is_compatible_the_block_stands(tmp_path):
    c = _pick(tmp_path, [ANCESTOR], hub_arch="LlamaForCausalLM")
    assert c.status == pf.BLOCK and c.reason == "nearest-model-missing"


def test_the_local_snapshots_config_decides_when_it_is_there(tmp_path):
    ok = _pick(tmp_path, [ANCESTOR, OWN], snap_cfg=LLAMA_CFG)
    assert ok.data["stock_runtime"]["name"] == "Llama-3.1-8B-Instruct"
    bad = _pick(tmp_path / "b", [OWN], snap_cfg={**LLAMA_CFG, "hidden_size": 128}) if (tmp_path / "b").mkdir() is None else None
    assert bad.status == pf.BLOCK


def test_with_no_config_to_compare_the_first_row_is_taken_and_stage_zero_measures(tmp_path):
    c = _pick(tmp_path, [ANCESTOR, OWN])
    assert c.status == pf.WARN                      # no hub architecture and no snapshot: unknown, not a reason to skip


# ---- the recorded route is restored before the preflight (second review of PR 78) -----------------------------------

def _params_dir(tmp_path):
    d = tmp_path / "tt-metal" / "models" / "tt_transformers" / "model_params" / "Llama-3.1-8B-Instruct"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    return d


def test_a_recorded_route_makes_the_base_row_a_warning_without_research_or_a_bundle_search(tmp_path):
    from test_preflight import cfg, signals
    d = _params_dir(tmp_path)

    def boom(*a):
        raise AssertionError("research must not run")
    s = signals(research=boom, installed_bundles=boom)
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s, resuming=True, stock_recorded=str(d))
    base = next(c for c in out if c.name == "base")
    assert base.status == pf.WARN and "recorded" in base.detail and base.data["stock_runtime"]["params_dir"] == str(d)
    assert base.data["stock_runtime"]["name"] == "Llama-3.1-8B-Instruct" and base.data["needs_fetch"] is False and base.data["base"] is None


def test_a_recorded_directory_that_is_gone_falls_back_to_the_normal_check(tmp_path):
    from test_preflight import cfg, signals
    d = _params_dir(tmp_path)
    (d / "config.json").unlink()
    seen = []
    s = signals(research=lambda *a: seen.append(1) or report())
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s, resuming=True, stock_recorded=str(d))
    assert seen == [1] and "stock_runtime" not in (next(c for c in out if c.name == "base").data or {})


def test_a_recorded_route_is_ignored_in_lab_mode(tmp_path):
    from test_preflight import lab_cfg, signals
    d = _params_dir(tmp_path)
    s = signals(lab_status=lambda: (True, "", ""))
    out = pf.run_preflight(lab_cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s, resuming=True, stock_recorded=str(d))
    assert "stock_runtime" not in (next(c for c in out if c.name == "base").data or {})


def test_resuming_a_stock_run_survives_research_failing_and_downloads_no_base(conf, tmp_path):
    from test_cli import Recorder, run
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    d = _params_dir(tmp_path)
    run_dir = _started(conf, {"model": str(snap), "stock_params": str(d)})

    def boom(*a):
        raise RuntimeError("offline")
    sig = _bundle_signals(snap, research=boom, local_snapshot=lambda m: snap if m == "Cloudflare/clef" else None)
    rec = Recorder()
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(run_dir)], rec=rec, sig=sig)
    assert code == 0, out
    assert f"stock_params={d}" in _inputs(rec) and rec.fetched == []


def test_a_fresh_run_ignores_the_recorded_route_argument_path(conf):
    code, out, rec, snap = _cli_run(conf)
    assert code == 0 and "stock_params=/host/params/Llama-3.1-8B-Instruct" in _inputs(rec)


# ---- lab mode (review of PR 78): the parameters directory is on this box, and a lab runs its tests elsewhere -----------

def test_in_lab_mode_the_stock_path_is_not_offered(tmp_path):
    from test_preflight import lab_cfg, signals
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    seen = []
    s = signals(research=lambda *a: rep, stock_params=lambda name: seen.append(name) or Path("/host/params") / name,
                lab_status=lambda: (True, "", ""))
    out = pf.run_preflight(lab_cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s)
    base = next(c for c in out if c.name == "base")
    assert "stock_runtime" not in (base.data or {}) and "tt-transformers supports" not in base.detail
    assert seen == []


def test_in_local_mode_the_same_signals_take_the_stock_path(tmp_path):
    from test_preflight import cfg, signals
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    s = signals(research=lambda *a: rep, stock_params=lambda name: Path("/host/params") / name)
    base = next(c for c in pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s) if c.name == "base")
    assert base.status == pf.WARN
