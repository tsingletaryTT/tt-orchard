# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""check_base with a research report, and the preflight wiring (orchard/preflight.py)."""
import pytest

from orchard import preflight as pf
from orchard.nearest import Installed

GEMMA = "google/gemma-4-12B"


def hub(**card):
    return pf.parse_hub({"id": "x/humanizer", "private": False, "gated": False, "sha": "c0",
                         "cardData": {"license": "apache-2.0", **card}, "config": {},
                         "siblings": [{"rfilename": "LICENSE", "size": 1},
                                      {"rfilename": "model.safetensors", "size": 24_000_000_000}]})


def report(**over):
    base = {"model": "x/humanizer", "selected": None, "candidates": [], "bundle_choices": [], "ports": [],
            "evidence": [], "unavailable": [], "offline": False, "stated_bases": [], "generated_at": "t"}
    base.update(over)
    return base


def choice(name, repo, installed, cls="full-port", conf="inferred", group=2, schema="5.1"):
    return {"name": name, "installed": installed, "repo": repo, "schema": schema, "class": cls,
            "confidence": conf, "group": group}


def port(claim, url="https://huggingface.co/x/y"):
    return {"claim": claim, "url": url, "confidence": "inferred", "source": "bundle", "fetched_at": "t"}


def check(rep, *, h=None, override=None, installed=(), snapshot=None, accept_inferred=False):
    return pf.check_base(model_id="x/humanizer", hub=h if h is not None else hub(), base_override=override,
                         local_snapshot=lambda m: snapshot, installed=list(installed), search=lambda q: [], report=rep,
                         accept_inferred=accept_inferred)


def test_without_a_report_the_messages_are_the_earlier_ones():
    block = check(None, h=hub(base_model=GEMMA))
    assert block.status == pf.BLOCK and block.data["report"] is None
    assert block.detail.endswith("found no bundle. Pick one with --base <bundle>, or install a bundle yourself "
                                  "and run bringup again")
    assert check(None).detail.endswith("Pass --base <bundle or model id> to name the model to start from")


def test_a_card_with_no_base_and_an_inferred_installed_bundle_warns_and_names_the_inference():
    sel = {"repo": "g/base", "bundle": "tt/base-p150", "confidence": "inferred", "class": "weights-only"}
    rep = report(selected=sel, candidates=[{"repo": "g/base", "class_estimate": {"class": "weights-only"}}])
    c = check(rep, installed=[Installed("tt/base-p150", "g/base", "6", 1)], accept_inferred=True)
    assert c.status == pf.WARN
    assert "inferred from its architecture" in c.detail and "g/base" in c.detail
    assert "research expects class weights-only" in c.detail
    assert c.data["base"] == "g/base" and c.data["bundle"] == "tt/base-p150" and c.data["needs_fetch"] is True


def test_the_inferred_text_names_the_row_why_and_a_stated_selection_says_stated():
    sel = {"repo": "g/base", "bundle": "tt/base-p150", "confidence": "inferred", "class": None}
    cand = {"repo": "g/base", "class_estimate": None, "why": "another fine-tune of g/root"}
    inst = [Installed("tt/base-p150", "g/base", "6", 1)]
    c = check(report(selected=sel, candidates=[cand]), installed=inst, accept_inferred=True)
    assert "inferred (another fine-tune of g/root)" in c.detail and "inferred from its architecture" not in c.detail
    stated = check(report(selected={**sel, "confidence": "stated"},
                          candidates=[{**cand, "why": "named as the base model"}]), installed=inst)
    assert "stated (named as the base model)" in stated.detail and "inferred" not in stated.detail
    assert "the card names none" in stated.detail


def test_a_card_base_wins_over_the_selection():
    sel = {"repo": "g/other", "bundle": "tt/other", "confidence": "inferred", "class": None}
    c = check(report(selected=sel), h=hub(base_model=GEMMA),
              installed=[Installed("tt/g", GEMMA, "6", 1), Installed("tt/other", "g/other", "6", 1)])
    assert c.status == pf.OK and c.data["base"] == GEMMA


def test_a_card_with_no_base_and_no_selection_lists_the_closest_bundles():
    rep = report(bundle_choices=[choice("o/a", "g/a", False, "weights-only", "weak"),
                                 choice("o/b", "g/b", True, None, "inferred")])
    c = check(rep)
    assert c.status == pf.WARN and c.data["base"] is None
    assert "Closest bundles found: o/a (weights-only, weak), o/b (no estimate, inferred)." in c.detail


def test_the_block_lists_the_ranked_bundles_with_class_and_the_first_two_ports():
    rep = report(bundle_choices=[choice("stisiTT/gemma-4-12b-it-p150", "google/gemma-4-12B-it", False,
                                        "weights-only", "inferred"),
                                 choice("tt-hous/gemma-4-26B", "google/gemma-4-26B", False, "full-port", "weak", 3)],
                 ports=[port("bundle stisiTT/x ships code for gemma4"), port("tt-metal has models/demos/gemma3"),
                        port("third")])
    c = check(rep, h=hub(base_model=GEMMA))
    assert c.status == pf.BLOCK and c.reason == "nearest-model-missing"
    assert [x["name"] for x in c.data["candidates"]] == ["stisiTT/gemma-4-12b-it-p150", "tt-hous/gemma-4-26B"]
    assert ("ranked bundles: stisiTT/gemma-4-12b-it-p150 (weights-only, inferred), "
            "tt-hous/gemma-4-26B (full-port, weak).") in c.detail
    assert "Existing work: bundle stisiTT/x ships code for gemma4 (https://huggingface.co/x/y); " \
           "tt-metal has models/demos/gemma3 (https://huggingface.co/x/y)." in c.detail
    assert "third" not in c.detail


def test_a_report_with_no_choices_keeps_the_old_listing():
    c = check(report(), h=hub(base_model=GEMMA))
    assert c.status == pf.BLOCK and c.data["candidates"] == [] and "found no bundle" in c.detail


def test_an_installed_bundle_that_serves_the_base_gets_the_class_note():
    rep = report(candidates=[{"repo": GEMMA, "class_estimate": {"class": "weights-only"}}])
    c = check(rep, h=hub(base_model=GEMMA), installed=[Installed("tt/g", GEMMA, "6", 1)])
    assert c.status == pf.OK and c.detail.endswith("its snapshot will be downloaded; research expects class weights-only")


def test_a_container_bundle_that_serves_the_base_still_warns_with_the_class_note():
    rep = report(candidates=[{"repo": GEMMA, "class_estimate": {"class": "full-port"}}])
    c = check(rep, h=hub(base_model=GEMMA), installed=[Installed("tt/g", GEMMA, "5.1", 1)])
    assert c.status == pf.WARN and "5.1 container package" in c.detail and "research expects class full-port" in c.detail


def test_the_report_is_carried_in_every_result():
    rep = report()
    assert check(rep).data["report"] is rep
    assert check(rep, h=hub(base_model=GEMMA)).data["report"] is rep
    assert check(rep, h=hub(base_model=GEMMA), installed=[Installed("tt/g", GEMMA, "6", 1)]).data["report"] is rep


# ---- run_preflight wiring ------------------------------------------------------------------------------

def test_run_preflight_gives_research_the_installed_bundles_and_uses_its_report(tmp_path):
    from test_preflight import cfg, signals
    seen = []
    rep = report(bundle_choices=[choice("o/a", "g/a", False, "weights-only", "weak")])

    def research(model_id, h, override, installed):
        seen.append((model_id, h.id, override, [b.name for b in installed]))
        return rep
    s = signals(installed_bundles=lambda: [Installed("tt/q", "Qwen/Qwen3.8-27B", "6", 1)], research=research)
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s, base_override=None)
    base = next(c for c in out if c.name == "base")
    assert seen == [("Cloudflare/clef", "Cloudflare/clef", None, ["tt/q"])]
    assert base.data["report"] is rep and "o/a (weights-only, weak)" in base.detail


def test_a_failing_research_leaves_the_earlier_behaviour_and_no_crash(tmp_path):
    from test_preflight import cfg, signals

    def research(*a):
        raise RuntimeError("hub down")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(research=research))
    base = next(c for c in out if c.name == "base")
    assert base.status == pf.WARN and base.data["report"] is None
    assert base.detail.endswith("(research failed: RuntimeError)")
    assert "hub down" not in base.detail                  # the exception text is not copied, only its type


def test_a_blocking_base_check_also_names_a_failed_research(tmp_path):
    from test_preflight import cfg, signals

    def research(*a):
        raise KeyError("x")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, base_override="g/none",
                           signals=signals(research=research))
    base = next(c for c in out if c.name == "base")
    assert base.status == pf.BLOCK and "research failed: KeyError" in base.detail


def test_a_research_that_works_adds_no_failure_note(tmp_path):
    from test_preflight import cfg, signals
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(research=lambda *a: report()))
    assert "research failed" not in next(c for c in out if c.name == "base").detail


def test_without_research_run_preflight_is_unchanged(tmp_path):
    from test_preflight import cfg, signals
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert next(c for c in out if c.name == "base").data["report"] is None


def test_default_signals_builds_research_only_when_options_are_given(tmp_path):
    from test_preflight import cfg
    from orchard.research import Options
    assert pf.default_signals(cfg(tmp_path)).research is None
    assert callable(pf.default_signals(cfg(tmp_path), research_options=Options(offline=True)).research)


def test_default_signals_research_with_offline_makes_no_request(tmp_path):
    from test_preflight import cfg
    from orchard.research import Options
    sig = pf.default_signals(cfg(tmp_path), research_options=Options(offline=True))
    rep = sig.research("a/b", None, None, [])
    assert rep["offline"] is True and rep["unavailable"] == [{"source": "research", "reason": "--offline"}]


class RecordingFetcher:
    """Stands in for orchard.research.http.Fetcher: keeps the constructor arguments and every URL asked for."""
    instances = []

    def __init__(self, cache_dir, **kw):
        self.cache_dir, self.kw, self.urls, self.offline = cache_dir, kw, [], kw.get("offline", False)
        self.refresh = kw.get("refresh", False)
        RecordingFetcher.instances.append(self)

    def now(self):
        return "t"

    def json(self, url, **kw):
        from orchard.research.http import FetchError
        self.urls.append(url)
        if url.endswith("/resolve/main/config.json"):
            return {"model_type": "gemma4", "architectures": ["X"]}
        if url.startswith("https://huggingface.co/api/models/"):
            return {"id": "a/b", "tags": [], "cardData": {}}
        raise FetchError("HTTP 404")

    def get(self, url, **kw):
        from orchard.research.http import FetchError
        self.urls.append(url)
        raise FetchError("HTTP 404")


def run_real_research(tmp_path, monkeypatch, options):
    """The real default_signals and the real report.build, with the network, `tt model search` and `gh` faked."""
    from test_preflight import cfg
    from orchard import nearest
    from orchard.research import http, ports
    from orchard.research.types import Result
    RecordingFetcher.instances = []
    github_code_calls = []
    monkeypatch.setattr(http, "Fetcher", RecordingFetcher)
    monkeypatch.setattr(nearest, "search", lambda q: [])
    monkeypatch.setattr(ports, "catalog", lambda *a, **kw: Result(data={"bundles": []}))
    monkeypatch.setattr(ports, "github_code", lambda *a, **kw: github_code_calls.append(1) or Result())
    sig = pf.default_signals(cfg(tmp_path), research_options=options)
    sig.research("a/b", None, None, [])
    (fetcher,) = RecordingFetcher.instances
    return fetcher, github_code_calls


def test_the_real_default_signals_hand_the_options_to_the_fetcher_and_the_report(tmp_path, monkeypatch):
    from orchard.research import Options
    cache = tmp_path / "cache"
    fetcher, github_code_calls = run_real_research(
        tmp_path, monkeypatch, Options(cache_dir=cache, no_github=True, refresh=True))
    assert fetcher.cache_dir == cache
    assert fetcher.kw.get("refresh") is True and fetcher.kw.get("offline") is False
    assert not any("search/issues" in u for u in fetcher.urls)
    assert github_code_calls == []
    assert any("models/demos" in u or "tt-metal" in u for u in fetcher.urls)     # the run did reach the ports step


def test_without_no_github_the_issue_and_code_searches_are_made(tmp_path, monkeypatch):
    from orchard.research import Options
    fetcher, github_code_calls = run_real_research(tmp_path, monkeypatch, Options(cache_dir=tmp_path / "c"))
    assert any("search/issues" in u for u in fetcher.urls)
    assert github_code_calls == [1]
    assert fetcher.kw.get("refresh") is False


# ---- an inferred base is used only when the operator says so ---------------------------------------

INFERRED_SEL = {"repo": "g/base", "bundle": "tt/base-p150", "confidence": "inferred", "class": "weights-only"}
INST = [Installed("tt/base-p150", "g/base", "6", 1)]


def test_an_inferred_base_without_the_opt_in_blocks_downloads_nothing_and_offers_its_bundle_first():
    rep = report(selected=INFERRED_SEL, candidates=[{"repo": "g/base", "class_estimate": {"class": "weights-only"}}],
                 bundle_choices=[choice("o/other", "g/other", False, None, "weak", 3),
                                 choice("tt/base-p150", "g/base", True, "weights-only", "inferred", 0, "6")])
    c = check(rep, installed=INST)
    assert c.status == pf.BLOCK and c.reason == "inferred-base-unconfirmed"
    assert c.data["needs_fetch"] is False and c.data["snapshot"] is None and c.data["bundle"] is None
    assert c.data["base"] == "g/base" and c.data["inferred"] is True
    assert [x["name"] for x in c.data["candidates"]] == ["tt/base-p150", "o/other"]
    assert "--accept-inferred-base" in c.detail and "--base tt/base-p150" in c.detail
    assert "card names no base" in c.detail


def test_with_the_opt_in_an_inferred_base_is_the_earlier_warning():
    c = check(report(selected=INFERRED_SEL), installed=INST, accept_inferred=True)
    assert c.status == pf.WARN and c.data["needs_fetch"] is True and c.data["base"] == "g/base"


def test_a_stated_selection_needs_no_opt_in():
    c = check(report(selected={**INFERRED_SEL, "confidence": "stated"}), installed=INST)
    assert c.status == pf.WARN and c.data["base"] == "g/base" and c.data["needs_fetch"] is True


def test_a_card_base_needs_no_opt_in():
    c = check(report(selected=INFERRED_SEL), h=hub(base_model="g/base"), installed=INST)
    assert c.status == pf.OK and c.data["base"] == "g/base"


def test_run_preflight_passes_the_opt_in_through():
    seen = {}
    real = pf.check_base

    def spy(**kw):
        seen.update(kw)
        return real(**kw)
    import pytest
    mp = pytest.MonkeyPatch()
    mp.setattr(pf, "check_base", spy)
    try:
        sig = pf.Signals(hub_info=lambda m: (hub(), None), local_snapshot=lambda m: None, free_gb=lambda p: 1000.0,
                         same_device=lambda a, b: False, credentials=lambda: [], port_in_use=lambda port: False,
                         gozer_status=lambda: "", load_tiers=lambda p: None)
        try:
            pf.run_preflight(None, "x/humanizer", accept_credentials=True, signals=sig, accept_inferred_base=True)
        except Exception:
            pass                                    # only the base check's arguments matter here
    finally:
        mp.undo()
    assert seen.get("accept_inferred") is True


# ---- code exists, no bundle --------------------------------------------------------------------------------
# A tt-metal or tt-transformers directory for the base is code, not a bundle. The block says that first, and says
# whether the code names a Blackhole board, before the long list of unrelated bundles.

def code_port(claim, source="tt-metal", conf="inferred", url="https://github.com/tenstorrent/tt-metal/tree/main/models/demos/x"):
    return {"claim": claim, "url": url, "confidence": conf, "source": source, "fetched_at": "t"}


def code_block(ports, **kw):
    return check(report(ports=ports, **kw), h=hub(base_model=GEMMA))


def test_code_for_the_base_is_named_before_the_bundles_and_the_reason_is_unchanged():
    c = code_block([code_port("tt-metal has models/demos/gemma3 (vLLM generator present)")],
                   bundle_choices=[choice("o/a", "g/a", False, "weights-only", "inferred")])
    assert c.status == pf.BLOCK and c.reason == "nearest-model-missing"
    assert ("Code for it exists, but code is not a bundle: tt-metal has models/demos/gemma3 (vLLM generator present) "
            "(https://github.com/tenstorrent/tt-metal/tree/main/models/demos/x).") in c.detail
    assert "has to be built or installed outside tt-orchard first" in c.detail
    assert c.detail.index("Code for it exists") < c.detail.index("ranked bundles:")
    assert c.detail.endswith("Pick one with --base <bundle>, or install a bundle yourself and run bringup again")
    assert "Existing work:" not in c.detail


@pytest.mark.parametrize("p", [code_port("tt-metal has models/demos/x", conf="weak"),
                               code_port("PR #1 (open): x", source="github"),
                               code_port("bundle o/b ships code for x", source="bundle"),
                               code_port("the community catalog lists o/b", source="catalog")])
def test_a_weak_port_or_one_that_is_not_code_keeps_the_earlier_message(p):
    c = code_block([p])
    assert "Code for it exists" not in c.detail and "Existing work: " + p["claim"] in c.detail


def test_tt_transformers_rows_count_as_code_too():
    c = code_block([code_port("tt-metal's tt_transformers README lists Llama 3.1 8B on n150 / p100 / p150", source="tt-transformers")])
    assert "Code for it exists, but code is not a bundle: tt-metal's tt_transformers README lists Llama 3.1 8B" in c.detail


def test_a_readme_that_names_a_blackhole_board_is_said_so():
    c = code_block([code_port("tt-metal has models/demos/x (README names Blackhole boards (P150))")])
    assert "The code names a Blackhole board." in c.detail and "check that it runs here" not in c.detail


def test_code_that_names_no_blackhole_board_is_a_warning_because_this_box_is_blackhole():
    c = code_block([code_port("tt-metal has models/demos/x (no Blackhole board named in its README, which names N300, T3K)")])
    assert "names no Blackhole board" in c.detail and "this box has Blackhole boards" in c.detail
    assert "check that it runs here before building a bundle" in c.detail


def test_code_with_no_board_facts_says_nothing_about_boards():
    c = code_block([code_port("tt-metal has models/demos/x")])
    assert "Blackhole" not in c.detail


def test_a_blackhole_row_wins_over_a_wormhole_only_row():
    c = code_block([code_port("tt-metal has models/demos/a (no Blackhole board named in its README, which names N300)"),
                    code_port("tt-metal's tt_transformers README lists M on n150 / p150", source="tt-transformers")])
    assert "The code names a Blackhole board." in c.detail and "this box has Blackhole boards" not in c.detail


def test_at_most_three_code_rows_are_listed_in_the_order_given():
    c = code_block([code_port(f"tt-metal has models/demos/d{i}") for i in range(5)])
    assert "d0" in c.detail and "d1" in c.detail and "d2" in c.detail and "d3" not in c.detail


def test_the_searched_listing_is_capped_so_it_does_not_bury_the_code():
    names = [{"name": f"o/qwen3-{i:02d}", "installed": False} for i in range(30)]
    c = pf.check_base(model_id="x/m", hub=hub(base_model="Qwen/Qwen2.5-VL-7B-Instruct"), base_override=None,
                      local_snapshot=lambda m: None, installed=[], search=lambda q: names,
                      report=report(ports=[code_port("tt-metal has models/demos/qwen25_vl")]))
    assert "o/qwen3-07" in c.detail and "o/qwen3-08" not in c.detail and "and 22 more" in c.detail
    assert len(c.data["candidates"]) == 30                         # the data keeps every candidate


def test_a_base_that_a_bundle_serves_gets_no_code_note():
    b = Installed(name="o/b", weights_repo=GEMMA, schema="6", chips=2)
    c = check(report(ports=[code_port("tt-metal has models/demos/gemma3")]), h=hub(base_model=GEMMA), installed=[b])
    assert c.status == pf.OK and "Code for it exists" not in c.detail


def test_a_real_demo_claim_from_the_research_step_reaches_the_block_message_with_its_board_warning():
    # The writer (ports.demo_facts) and the reader (preflight._code_note) share one phrase; nothing here spells it.
    from orchard.research import ports
    from orchard.research.types import Evidence
    qv = "https://api.github.com/repos/tenstorrent/tt-metal/contents/models/demos/qwen25_vl/tt"
    readme = "https://raw.githubusercontent.com/tenstorrent/tt-metal/main/models/demos/qwen25_vl/README.md"

    class F:
        def json(self, url, **kw):
            return [{"name": "generator_vllm.py"}]

        def get(self, url, **kw):
            return b"N300 T3K TG"

    ev = Evidence("tt-metal", "tt-metal has models/demos/qwen25_vl", "https://github.com/tenstorrent/tt-metal/tree/main/models/demos/qwen25_vl",
                  "inferred", "t")
    row = ports.demo_facts(F(), ports.Result(evidence=[ev])).evidence[0].as_dict()
    c = code_block([row])
    assert "vLLM generator present" in c.detail and "this box has Blackhole boards" in c.detail


def test_a_real_tt_transformers_row_for_wormhole_boards_reaches_the_block_message_with_the_warning():
    # The collector (ports.tt_transformers) writes the claim and preflight._code_note reads it; no test string here
    # spells the phrase they share.
    from orchard.research import ports
    readme = ("| Model | Hardware | HF_MODEL |\n|---|---|---|\n"
              "| [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B) | n300 | ```Qwen/Qwen2.5-7B``` |\n").encode()

    class F:
        def json(self, url, **kw):
            return []

        def get(self, url, **kw):
            return readme

        def now(self):
            return "t"

    row = ports.tt_transformers(F(), ports.terms({}), ["Qwen/Qwen2.5-7B"]).evidence[0].as_dict()
    c = code_block([row])
    assert "n300" in c.detail and "this box has Blackhole boards" in c.detail
