# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard bringup` says what it is doing while the preflight and the research run.

Measured on 2026-10-21: research took 81 s for a 27B model (three sequential reads of weight headers) and the page
printed only after every check finished, so the first line after "mode:" came 84 s later."""
import threading

import pytest

from orchard import bringup_config, cli
from orchard import preflight as pf
from orchard.research import Options, report
from orchard.research import types as rtypes
from test_cli import Pipe, Tty, conf, run, signals  # noqa: F401
from test_preflight import cfg
from test_research_report import api, build, cfg_url, hub_model, new_hub, world


# ---- the reporter ---------------------------------------------------------------------------------------

def test_the_reporter_passes_cleaned_text_to_the_callback():
    seen = []
    rtypes.reporter(seen.append)("reading \x1b[31mfoo\x07 bar\n baz")
    assert seen == ["reading [31mfoo bar baz"]                         # controls become spaces, one line


def test_the_reporter_hides_a_home_path_the_way_every_research_line_does():
    seen = []
    rtypes.reporter(seen.append)("reading /home/someone/secret")
    assert seen == ["[removed: an absolute home path]"]


def test_a_failing_callback_never_reaches_the_caller():
    def boom(text):
        raise RuntimeError("terminal gone")
    rtypes.reporter(boom)("anything")                                  # no exception


def test_no_callback_is_a_no_op():
    rtypes.reporter(None)("anything")


# ---- research --------------------------------------------------------------------------------------------

def collected(**over):
    lines = []
    r = build(world(), progress=lines.append, **over)
    return lines, r


def test_research_says_what_it_reads_in_the_order_it_reads_it():
    lines, _ = collected()
    joined = "\n".join(lines)
    order = [joined.index(p) for p in ("lineage of h/new", "bundles", "weight headers of h/new", "weight headers of g/base",
                                       "existing ports")]
    assert order == sorted(order), lines


def test_a_header_line_says_how_many_shards_it_will_read():
    names = [f"model-{i:05d}-of-00011.safetensors" for i in range(1, 12)] + ["config.json"]
    hub = pf.parse_hub({"id": "h/new", "cardData": {"base_model": ["g/base"]},
                        "siblings": [{"rfilename": n, "size": 1} for n in names]})
    lines = []
    build(world(), hub=hub, progress=lines.append)
    assert "reading the weight headers of h/new (11 shards)" in lines
    assert "reading the weight headers of g/base (1 shard)" in lines          # singular for one


def test_offline_research_says_nothing():
    lines = []
    w = world()
    w.offline = True
    build(w, progress=lines.append)
    assert lines == []


def test_research_without_a_callback_gives_the_same_report():
    plain = build(world())
    with_cb = build(world(), progress=lambda t: None)
    assert plain == with_cb


def test_a_callback_that_raises_does_not_change_the_report():
    def boom(text):
        raise RuntimeError("terminal gone")
    assert build(world(), progress=boom) == build(world())


def test_bundles_and_ports_are_announced_only_when_there_is_a_family_to_search():
    w = world({cfg_url("h/new"): {"hidden_size": 8}, api("h/new"): hub_model("h/new")})   # no model_type, no base
    lines = []
    build(w, hub=new_hub(), installed=[], progress=lines.append)
    assert lines and not any("bundles" in l or "existing ports" in l for l in lines), lines
    assert lines[0] == "reading the lineage of h/new on the hub"


# ---- the preflight ---------------------------------------------------------------------------------------

def traced(tmp_path, **over):
    """run_preflight with every slow signal recording when it is called, in one list with the progress lines."""
    events = []

    def rec(name, fn):
        def inner(*a, **k):
            events.append(("call", name))
            return fn(*a, **k)
        return inner

    base = signals()
    sig = signals(hub_info=rec("hub_info", base.hub_info), free_gb=rec("free_gb", base.free_gb),
                  research=rec("research", lambda m, h, o, i: None), **over)
    return events, sig


def test_the_preflight_names_each_phase_before_it_starts(tmp_path):
    events, sig = traced(tmp_path)
    pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=sig,
                     progress=lambda t: events.append(("say", t)))
    at = {kind_text: i for i, kind_text in enumerate(events)}
    say = [(i, t) for i, (k, t) in enumerate(events) if k == "say"]
    assert [t for _, t in say] == [
        "asking the hub about Cloudflare/clef",
        "researching the base model: lineage, bundles, weight headers and existing ports",
        "checking this machine: disk, tiers, ports and chips"]
    first = {name: i for i, (k, name) in enumerate(events) if k == "call"}
    assert say[0][0] < first["hub_info"] < say[1][0] < first["research"] < say[2][0] < first["free_gb"]


def test_without_research_there_is_no_research_line(tmp_path):
    lines = []
    pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=signals(),
                     progress=lines.append)
    assert lines == ["asking the hub about Cloudflare/clef", "checking this machine: disk, tiers, ports and chips"]


def test_a_preflight_without_a_callback_is_unchanged(tmp_path):
    a = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=signals())
    b = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=signals(),
                         progress=lambda t: None)
    assert a == b


def test_a_callback_that_raises_does_not_stop_the_preflight(tmp_path):
    def boom(text):
        raise RuntimeError("terminal gone")
    a = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=signals())
    assert pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=signals(),
                            progress=boom) == a


def test_the_default_research_signal_hands_its_options_callback_to_the_report(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(report, "build", lambda **kw: seen.update(kw) or {})
    lines = []
    sig = pf.default_signals(cfg(tmp_path), research_options=Options(progress=lines.append))
    sig.research("h/new", None, None, [])
    assert seen["progress"] is not None
    seen["progress"]("hello")
    assert lines == ["hello"]


# ---- the command -----------------------------------------------------------------------------------------

def test_the_command_prints_the_phases_before_the_preflight_page(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"])
    lines = out.splitlines()
    page = next(i for i, l in enumerate(lines) if l.startswith("╔══ tt-orchard bringup"))
    asked = next(i for i, l in enumerate(lines) if l.startswith("asking the hub about Cloudflare/clef"))
    machine = next(i for i, l in enumerate(lines) if l.startswith("checking this machine"))
    assert asked < machine < page
    assert lines[asked].endswith("...")


def test_quiet_leaves_the_phase_lines_out(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--quiet"])
    assert "asking the hub" not in out and "checking this machine" not in out and "tt-orchard bringup" in out


def test_the_command_hands_research_the_same_printer(conf, monkeypatch):
    got = {}

    def capture(cfg_, research_options=None):
        got["options"] = research_options
        return signals()

    monkeypatch.setattr(pf, "default_signals", capture)
    out = Pipe()
    cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", "--dry-run"], env={}, stdout=out,
             supervisor_main=lambda argv: 0, fetcher=lambda *a, **k: None)
    got["options"].progress("reading something")
    assert "reading something..." in out.getvalue().splitlines()


def test_the_command_hands_research_no_printer_when_quiet(conf, monkeypatch):
    got = {}

    def capture(cfg_, research_options=None):
        got["options"] = research_options
        return signals()

    monkeypatch.setattr(pf, "default_signals", capture)
    cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", "--dry-run", "--quiet"], env={}, stdout=Pipe(),
             supervisor_main=lambda argv: 0, fetcher=lambda *a, **k: None)
    assert got["options"].progress is None


def test_on_a_terminal_the_phase_lines_carry_an_icon_and_stay_on_their_own_line(conf):
    out = Tty()
    cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", "--dry-run", "--style", "pretty"], env={}, stdout=out,
             signals=signals(), supervisor_main=lambda argv: 0, fetcher=lambda *a, **k: None)
    line = next(l for l in out.getvalue().splitlines() if "asking the hub" in l)
    assert line.startswith("🔎 ") and line.endswith("...")
