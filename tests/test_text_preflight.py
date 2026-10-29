# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The preflight base row and the bringup input for a text model the harness's own text runtime serves (orchard/text_runtime.py)."""
from orchard import preflight as pf
from test_preflight import CLEF_PAYLOAD, cfg, signals

MINICPM = dict(CLEF_PAYLOAD, id="openbmb/MiniCPM3-4B", cardData={"license": "apache-2.0"}, siblings=[
    {"rfilename": "config.json", "size": 2000}, {"rfilename": "pytorch_model.bin", "size": 8_000_000_000}])


def row(model="openbmb/MiniCPM3-4B", **kw):
    hub = pf.parse_hub(dict(MINICPM, id=model))
    return pf.check_base(model_id=model, hub=hub, base_override=kw.get("base_override"), local_snapshot=lambda m: None, installed=[], search=None)


def test_a_model_the_text_runtime_serves_needs_no_bundle_and_says_what_follows():
    c = row()
    assert c.status == "warn" and c.reason is None and c.data["text_runtime"] == "minicpm3" and c.data["needs_fetch"] is False
    assert "does not boot the local coder" in c.detail and "stages 0 to 2, 6 and 8" in c.detail


def test_a_revision_in_the_model_id_is_ignored():
    assert row("openbmb/MiniCPM3-4B@" + "a" * 40).data["text_runtime"] == "minicpm3"


def test_a_base_the_operator_named_is_a_swap_and_not_a_text_runtime_run():
    c = row(base_override="Qwen/Qwen3.8-27B")
    assert not (c.data or {}).get("text_runtime")


def test_another_model_keeps_the_old_row_and_names_no_base():
    c = row("someone/unknown-model")
    assert not (c.data or {}).get("text_runtime") and "names no base_model" in c.detail


def test_run_preflight_takes_the_text_route_before_any_research(tmp_path):
    calls = []

    def research(*a):
        calls.append(a)
        raise AssertionError("research must not run for a text-runtime model")
    sig = signals(hub_info=lambda m: (pf.parse_hub(MINICPM), None), research=research, installed_bundles=lambda: [])
    out = pf.run_preflight(cfg(tmp_path), "openbmb/MiniCPM3-4B", accept_credentials=False, signals=sig)
    base = next(c for c in out if c.name == "base")
    assert base.data["text_runtime"] == "minicpm3" and not calls and base.status == "warn"


# ---- the command ------------------------------------------------------------------------------------------------------

def test_bringup_passes_the_text_runtime_to_the_supervisor_as_an_input(tmp_path):
    from test_cli import Recorder, conf as _conf, run  # noqa: F401
    p = tmp_path / "bringup.toml"
    p.write_text(f'runs_root = "{tmp_path}/runs"\nhf_home = "{tmp_path}/hf"\ncache_root = "{tmp_path}/cache"\n[coder]\ntarget = "pkg/coder"\nport = 8000\nchips = 4\n')
    sig = signals(hub_info=lambda m: (pf.parse_hub(MINICPM), None))
    code, out, rec = run(p, ["bringup", "openbmb/MiniCPM3-4B", "--dry-run"], sig=sig)
    assert code == 0 and "--input text_runtime=minicpm3" in out
