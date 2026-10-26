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
