# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tt-orchard updates: rows, rules, the read-only runner, hostile answers, the cache and the preflight hint."""
import json
import time

import pytest

from orchard import skillsfresh, ttlang_pin, updates, verifications
from orchard.updates import BEHIND, CANDIDATE, CURRENT, UNKNOWN, NotReadOnly, Tools

TOKEN = "hf_SECRETSECRETSECRET"
PIN_TAG = ttlang_pin.VERSION_TAG
NOW = time.mktime((2026, 10, 29, 12, 0, 0, 0, 0, 0))


def j(x):
    return (200, json.dumps(x).encode(), "")


class Net:
    """Fake `get`: the first prefix that matches the URL answers. Records every URL and header set."""

    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def __call__(self, url, headers, timeout, max_bytes):
        self.seen.append((url, dict(headers), timeout))
        for prefix, ans in self.routes:
            if url.startswith(prefix):
                return ans(url) if callable(ans) else ans
        return 404, b"", "HTTP 404"


class Runs:
    def __init__(self, table):
        self.table, self.seen = table, []

    def __call__(self, argv, timeout, env=None):
        self.seen.append(list(argv))
        for k, v in self.table.items():
            if " ".join(argv).startswith(k):
                return v
        return 1, "", "no such"


def installed_json(tmp_path, rev="a" * 40):
    p = tmp_path / "installed.json"
    p.write_text(json.dumps({"raahemnabeel/qwen3-coder-next": {"revision": rev}}))
    return p


def routes(**over):
    r = {
        "hf_coder": ("https://huggingface.co/api/models/raahemnabeel/qwen3-coder-next",
                     j({"sha": "b" * 40, "lastModified": "2026-10-28T10:00:00Z"})),
        "hf_list": ("https://huggingface.co/api/models?", j([])),
        "gh_tags": ("https://api.github.com/repos/tenstorrent/tt-lang/tags", j([{"name": "v" + PIN_TAG, "commit": {"sha": "c" * 40}}])),
        "gh_commit": ("https://api.github.com/repos/tenstorrent/tt-lang/commits/", j({"commit": {"committer": {"date": "2026-10-24T00:00:00Z"}}})),
        "dev": (updates.DEV_INDEX, (200, b'<a href="tt_lang-1.1.10.dev20261026+light-cp312.whl">x</a>', "")),
        "pypi": ("https://pypi.org/pypi/tt-lang/json", j({"info": {"version": "1.1.6"}})),
        "rel": ("https://api.github.com/repos/", j({"tag_name": "v0.1.0", "published_at": "2026-10-01T00:00:00Z"})),
    }
    for k, v in over.items():
        r[k] = v
    return list(r.values())


def run(tmp_path, net_routes=None, runs=None, offline=False, skills=None, ver=None, environ=None, tree=None):
    net = Net(net_routes if net_routes is not None else routes())
    rn = Runs(runs or {"git -C": (0, "v0.80.0\n", ""), "tt-model --version": (0, "0.1.0\n", ""),
                       "tt --version": (0, "tt 1.2.0\n", ""), "gozer --version": (0, "gozer 0.3.4\n", "")})
    tools = Tools(net, rn, offline=offline, environ=environ or {"HF_TOKEN": TOKEN}, which=lambda n: True)
    tree = tree or tmp_path
    rows = updates.run_all(tools, updates.load_config(None), installed_path=installed_json(tmp_path), tt_metal_dir=tree,
                           skills=skills, ver=ver or verifications.Verifications())
    return rows, net, rn


def row(rows, key, startswith=""):
    return next(r for r in rows if r.key == key and r.item.startswith(startswith))


# ---- statuses -------------------------------------------------------------------------------------------------

def test_coder_behind_when_hf_sha_differs(tmp_path):
    rows, _, _ = run(tmp_path)
    r = row(rows, "coder", "coder qwen3-coder-next")
    assert r.status == BEHIND and r.installed == "a" * 10 and r.upstream == "b" * 10 and r.upstream_date == "2026-10-28"
    assert any(c.startswith("tt-model pull ") for c in r.proposal)


def test_coder_current_when_sha_matches(tmp_path):
    r2 = routes(hf_coder=("https://huggingface.co/api/models/raahemnabeel/qwen3-coder-next", j({"sha": "a" * 40, "lastModified": "2026-10-01T00:00:00Z"})))
    rows, _, _ = run(tmp_path, r2)
    assert row(rows, "coder", "coder qwen3-coder-next").status == CURRENT


def test_uninstalled_fallback_is_unknown_with_reason(tmp_path):
    rows, _, _ = run(tmp_path)
    r = row(rows, "coder", "coder qwen3-coder-next-blackhole")
    assert r.status == UNKNOWN and "not installed" in r.why and r.label().startswith("unknown (")


def test_candidate_found_by_family_and_not_configured(tmp_path):
    lst = [{"id": "someone/Qwen3.5-Coder-Next-80B", "lastModified": "2026-10-27T00:00:00Z"},
           {"id": "raahemnabeel/qwen3-coder-next", "lastModified": "2026-10-20T00:00:00Z"},
           {"id": "someone/llama-chat", "lastModified": "2026-10-28T00:00:00Z"}]
    rows, _, _ = run(tmp_path, routes(hf_list=("https://huggingface.co/api/models?", j(lst))))
    c = row(rows, "coder", "coder candidates")
    assert c.status == CANDIDATE and "someone/Qwen3.5-Coder-Next-80B (2026-10-27)" in c.upstream
    assert "llama-chat" not in c.upstream and "raahemnabeel/qwen3-coder-next (" not in c.upstream


def test_family_list_is_a_config_key(tmp_path):
    cfgp = tmp_path / "u.toml"
    cfgp.write_text('family = ["glm-coder"]\n')
    assert updates.load_config(cfgp)["family"] == ["glm-coder"]
    cfgp.write_text('famly = ["x"]\n')
    with pytest.raises(updates.ConfigError):
        updates.load_config(cfgp)


def test_no_candidates_is_current(tmp_path):
    rows, _, _ = run(tmp_path)
    assert row(rows, "coder", "coder candidates").status == CURRENT


def test_ttlang_current_behind_candidate(tmp_path):
    rows, _, _ = run(tmp_path)
    assert row(rows, "tt-lang").status == CURRENT
    newer = routes(gh_tags=("https://api.github.com/repos/tenstorrent/tt-lang/tags", j([{"name": "v9.9.9", "commit": {"sha": "d" * 40}}])))
    assert row(run(tmp_path, newer)[0], "tt-lang").status == BEHIND
    dev = routes(dev=(updates.DEV_INDEX, (200, b"tt_lang-9.9.9.dev20270101", "")))
    assert row(run(tmp_path, dev)[0], "tt-lang").status == CANDIDATE


def test_ttlang_dev_wheel_newer_than_verified_wheel_is_candidate(tmp_path):
    ver = verifications.Verifications([verifications.Record(
        {"tt-lang": ttlang_pin.SHA, "tt-lang-wheel": "1.1.10.dev20261020"}, "2026-10-21", "pass", "x.md")])
    assert row(run(tmp_path, ver=ver)[0], "tt-lang").status == CANDIDATE


def test_ttmetal_behind_and_current(tmp_path):
    rel = routes(rel=("https://api.github.com/repos/", j({"tag_name": "v0.81.0", "published_at": "2026-10-20T00:00:00Z"})))
    r = row(run(tmp_path, rel)[0], "tt-metal")
    assert r.status == BEHIND and r.installed == "v0.80.0" and r.upstream == "v0.81.0"
    same = routes(rel=("https://api.github.com/repos/", j({"tag_name": "v0.80.0"})))
    assert row(run(tmp_path, same)[0], "tt-metal").status == CURRENT


def test_tool_versions_compared(tmp_path):
    rel = routes(rel=("https://api.github.com/repos/", j({"tag_name": "v1.3.0"})))
    rows, _, _ = run(tmp_path, rel)
    assert row(rows, "tt-cli").status == BEHIND and row(rows, "tt-gozer").status == BEHIND
    assert row(rows, "tt-model-manager").status == BEHIND          # 0.1.0 < 1.3.0
    rows2, _, _ = run(tmp_path)                                    # upstream v0.1.0
    assert row(rows2, "tt-model-manager").status == CURRENT


def test_missing_tool_is_unknown_not_crash(tmp_path):
    tools = Tools(Net(routes()), Runs({}), environ={}, which=lambda n: False)
    r = updates.check_tool(tools, updates.load_config(None), "tt-cli", "tt", "tt-cli", "x")
    assert r.status == UNKNOWN and "not installed" in r.why


def test_skills_row_from_freshness(tmp_path):
    f = skillsfresh.Freshness(skillsfresh.SOURCE_SKILLS, "/x", skillsfresh.BEHIND, "d", update="git pull")
    assert updates.check_skills([f]).status == BEHIND
    g = skillsfresh.Freshness(skillsfresh.SOURCE_SKILLS, "/x", skillsfresh.CURRENT, "d")
    assert updates.check_skills([g]).status == CURRENT
    u = skillsfresh.Freshness(skillsfresh.SOURCE_SKILLS, "/x", skillsfresh.UNKNOWN, "no network")
    assert updates.check_skills([u]).status == UNKNOWN


# ---- failures ---------------------------------------------------------------------------------------------------

def test_every_failure_is_an_unknown_row(tmp_path):
    rows, _, _ = run(tmp_path, net_routes=[("https://", (0, b"", "URLError"))], runs={"nothing": (1, "", "")})
    assert rows and all(r.status in (UNKNOWN, CURRENT) for r in rows)
    assert row(rows, "tt-lang").status == UNKNOWN and "URLError" in row(rows, "tt-lang").why


def test_offline_makes_no_call(tmp_path):
    rows, net, _ = run(tmp_path, offline=True)
    assert net.seen == []
    assert row(rows, "tt-lang").why.startswith("upstream: offline")


def test_budget_stops_calls(tmp_path):
    t = iter(range(0, 1000, 100))
    net = Net(routes())
    tools = Tools(net, Runs({}), budget_s=150, clock=lambda: next(t), environ={}, which=lambda n: True)
    for _ in range(5):
        tools.json("https://pypi.org/pypi/tt-lang/json")
    assert 1 <= len(net.seen) < 5 and all(s[2] <= 150 for s in net.seen)


def test_get_that_raises_is_unknown(tmp_path):
    def boom(*a):
        raise RuntimeError("x")
    tools = Tools(boom, Runs({}), environ={}, which=lambda n: True)
    assert tools.json("https://pypi.org/pypi/tt-lang/json")[0] is None


# ---- read-only guard --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["tt-model", "pull", "x/y"], ["tt", "run", "x"], ["gozer", "acquire", "--chips", "1"], ["docker", "restart", "c"],
    ["pip", "install", "x"], ["git", "-C", "/x", "pull"], ["git", "-C", "/x", "fetch"], ["git", "checkout", "x"],
    ["tt-model", "--version", "--also"], ["uv", "pip", "install", "x"], ["gozer", "release", "1"], [],
    ["git", "show", "--output=/tmp/x", "HEAD"], ["sh", "-c", "tt --version"]])
def test_runner_refuses_mutating_commands(argv):
    ran = []
    tools = Tools(None, lambda *a: ran.append(a) or (0, "", ""), environ={})
    with pytest.raises(NotReadOnly):
        tools.cmd(argv)
    assert ran == []


@pytest.mark.parametrize("url", ["http://huggingface.co/api/models/x", "https://evil.example/x",
                                 "https://huggingface.co.evil.example/x", "file:///etc/passwd"])
def test_runner_refuses_other_hosts(url):
    with pytest.raises(NotReadOnly):
        Tools(lambda *a: (200, b"{}", ""), None, environ={}).json(url)


def test_full_run_only_sends_allowed_commands(tmp_path):
    _, net, rn = run(tmp_path)
    allowed_subs = {"describe", "show", "--version"}
    for argv in rn.seen:
        assert argv[0] in ("git", "tt-model", "tt", "gozer")
        assert argv[-1] == "--version" or any(a in allowed_subs for a in argv)
    assert all(u.startswith("https://") for u, _, _ in net.seen)


# ---- token and hostile input ----------------------------------------------------------------------------------

def test_token_goes_only_to_huggingface_and_is_never_in_output(tmp_path, capsys, monkeypatch):
    rows, net, _ = run(tmp_path)
    for url, headers, _ in net.seen:
        assert ("Authorization" in headers) == url.startswith("https://huggingface.co/")
    blob = json.dumps([r.__dict__ for r in rows]) + "\n".join(updates.render(rows, NOW))
    assert TOKEN not in blob


def test_hostile_answers_do_not_crash_or_print_control_chars(tmp_path):
    hostile = [{"id": "x/\x1b[31mcoder-next\x07", "lastModified": "\x1b[0m"}, 5, None, {"id": ["a"]}, {"id": "a/" + "coder-next" * 5000},
               {"id": "ok/coder-next-x", "lastModified": 12345}]
    r = routes(hf_coder=("https://huggingface.co/api/models/raahemnabeel/qwen3-coder-next", j({"sha": "\x1b[2Jzzz", "lastModified": {"a": 1}})),
               hf_list=("https://huggingface.co/api/models?", j(hostile)),
               gh_tags=("https://api.github.com/repos/tenstorrent/tt-lang/tags", j([{"name": "v\x1b1.2.3"}, "x", None, {"name": 7}])),
               dev=(updates.DEV_INDEX, (200, b"\x00\xff" * 1000, "")),
               pypi=("https://pypi.org/pypi/tt-lang/json", (200, b"{" * 100000, "")),
               rel=("https://api.github.com/repos/", (200, b"[" * 100000, "")))
    rows, _, _ = run(tmp_path, r, runs={"git -C": (0, "v0.80.0\x1b[31m\n", ""), "tt --version": (0, "\x1b[2J" * 100, "")})
    text = "\n".join(updates.render(rows, NOW)) + json.dumps([r.__dict__ for r in rows])
    assert not any(ord(c) < 32 and c not in "\n" for c in "\n".join(updates.render(rows, NOW)))
    assert "\x1b" not in "\n".join(updates.render(rows, NOW))
    assert text


def test_oversize_answer_is_refused(tmp_path):
    big = (200, b"[" + b"1," * (updates.BODY_MAX // 2 + 10) + b"1]", "")
    v, why = Tools(lambda *a: big, None, environ={}).json("https://pypi.org/pypi/tt-lang/json")
    assert v is None and why


# ---- rules and verification -----------------------------------------------------------------------------------

def test_behind_row_names_rule_partners(tmp_path):
    rel = routes(rel=("https://api.github.com/repos/", j({"tag_name": "v0.81.0"})))
    r = row(run(tmp_path, rel)[0], "tt-metal")
    text = " ".join(r.moves_with)
    assert "R1" in text and "tt-lang" in text and "find-compatible-ttlang.sh" in text and "R2" in text


def test_rules_are_data_and_cover_the_chain():
    ids = {r["id"] for r in updates.RULES}
    members = {m for r in updates.RULES for m in r["members"]}
    assert {"tt-lang", "tt-metal", "tt-model-manager", "tt-cli", "coder"} <= members and len(ids) == len(updates.RULES)


def test_ready_is_unverified_without_a_record_and_verified_with_one(tmp_path):
    rel = routes(rel=("https://api.github.com/repos/", j({"tag_name": "v0.81.0"})))
    assert row(run(tmp_path, rel)[0], "tt-metal").ready.startswith("unverified: run the smoke and probe kernels")
    ver = verifications.Verifications([verifications.Record({"tt-lang": ttlang_pin.SHA, "tt-metal": "v0.81.0"}, "2026-10-30", "pass", "ev.md")])
    assert row(run(tmp_path, rel, ver=ver)[0], "tt-metal").ready.startswith("verified (2026-10-30, ev.md)")
    other = verifications.Verifications([verifications.Record({"tt-lang": ttlang_pin.SHA, "tt-metal": "v0.82.0"}, "2026-10-30", "pass", "ev.md")])
    assert row(run(tmp_path, rel, ver=other)[0], "tt-metal").ready.startswith("unverified")
    fail = verifications.Verifications([verifications.Record({"tt-lang": ttlang_pin.SHA, "tt-metal": "v0.81.0"}, "2026-10-30", "fail", "ev.md")])
    assert row(run(tmp_path, rel, ver=fail)[0], "tt-metal").ready.startswith("unverified")


def test_verifications_reader(tmp_path):
    p = tmp_path / "v.json"
    assert verifications.load(p).records == []
    good = {"versions": {"a": "1"}, "date": "2026-10-29", "result": "pass", "evidence": "e.md"}
    p.write_text(json.dumps({"schema": 1, "records": [good, {"versions": {}, "date": "x"}, 5, dict(good, result="maybe")]}))
    v = verifications.load(p)
    assert len(v.records) == 1 and len(v.problems) == 3
    p.write_text("{not json")
    assert verifications.load(p).problems
    p.write_text(json.dumps({"schema": 2, "records": []}))
    assert verifications.load(p).problems
    assert verifications.load(p).match({"a": "1"}) is None


def test_shipped_verifications_file_is_valid():
    v = verifications.load(verifications.default_path())
    assert v.problems == []


# ---- command, cache, hint ---------------------------------------------------------------------------------------

class Args:
    json = False
    offline = False
    refresh = False
    tt_metal_dir = None
    updates_config = None
    skills_dirs = None


def test_main_prints_table_and_caches_then_reads_cache(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ORCHARD_VERIFICATIONS", str(tmp_path / "none.json"))
    monkeypatch.setenv("ORCHARD_TT_METAL_DIR", str(tmp_path))
    tools = Tools(Net(routes()), Runs({"git -C": (0, "v0.80.0\n", "")}), environ={}, which=lambda n: False)
    assert updates.main(Args(), tools=tools, now=NOW) == 0
    out = capsys.readouterr().out
    assert "tt-lang" in out and "This command changes nothing" in out
    n = tools.calls
    assert n > 0
    assert updates.main(Args(), tools=tools, now=time.time()) == 0
    assert tools.calls == n and "from the cache" in capsys.readouterr().out


def test_refresh_ignores_cache_and_json_is_valid(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ORCHARD_VERIFICATIONS", str(tmp_path / "none.json"))
    tools = Tools(Net(routes()), Runs({}), environ={}, which=lambda n: False)
    updates.main(Args(), tools=tools)
    n = tools.calls
    capsys.readouterr()
    a = Args()
    a.refresh = a.json = True
    updates.main(a, tools=tools)
    assert tools.calls > n
    assert json.loads(capsys.readouterr().out)["rows"]


def test_offline_main_uses_cache_without_network(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ORCHARD_VERIFICATIONS", str(tmp_path / "none.json"))
    updates.main(Args(), tools=Tools(Net(routes()), Runs({}), environ={}, which=lambda n: False))
    capsys.readouterr()
    net = Net(routes())
    a = Args()
    a.offline = True
    updates.main(a, tools=Tools(net, Runs({}), offline=True, environ={}))
    assert net.seen == [] and "from the cache" in capsys.readouterr().out


def test_hint_only_when_behind_or_candidate(tmp_path):
    from orchard import skillsfresh as sf
    p = tmp_path / "u.json"
    assert updates.hint_from_cache(p) is None
    c = sf.Cache(p)
    c.put("report", {"rows": [{"item": "tt-lang", "status": CURRENT}, {"item": "x", "status": UNKNOWN}]})
    assert updates.hint_from_cache(p) is None
    c.put("report", {"rows": [{"item": "tt-metal (host tree)", "status": BEHIND}, {"item": "coder candidates", "status": CANDIDATE}]})
    h = updates.hint_from_cache(p)
    assert h.count("\n") == 0 and "tt-orchard updates" in h and "tt-metal" in h
    c.put("report", {"rows": "garbage"})
    assert updates.hint_from_cache(p) is None


def test_hint_never_touches_network(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, "_real_get", lambda *a: pytest.fail("network"))
    updates.hint_from_cache(tmp_path / "nope.json")


# ---- preflight and cli wiring ---------------------------------------------------------------------------------

def test_preflight_prints_one_informational_line_and_never_blocks(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from orchard import preflight as pf
    from test_preflight import cfg, signals
    line = "updates (behind: tt-metal). Run `tt-orchard updates` for details."
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals(updates_hint=lambda: line))
    rows = [c for c in out if c.name == "updates"]
    assert len(rows) == 1 and rows[0].status == pf.OK and rows[0].detail == line and not pf.blocked(out)
    for hint in (lambda: None, lambda: (_ for _ in ()).throw(OSError("x"))):
        out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals(updates_hint=hint))
        assert "updates" not in [c.name for c in out]


def test_cli_registers_updates(tmp_path, capsys, monkeypatch):
    from orchard import cli
    monkeypatch.setenv("ORCHARD_VERIFICATIONS", str(tmp_path / "none.json"))
    seen = {}
    monkeypatch.setattr(updates, "main", lambda args, out=None: seen.update(vars(args)) or 0)
    assert cli.main(["updates", "--json", "--offline", "--refresh"], env={"HOME": str(tmp_path)}, stdout=None) == 0
    assert seen["json"] and seen["offline"] and seen["refresh"]


def test_dev_wheel_is_read_from_the_newest_month_directory(tmp_path):
    root = b'<a href="2026-09/">a</a><a href="2026-10/">b</a><a href="ttmetal/">c</a>'
    dev = routes(dev=(updates.DEV_INDEX, lambda u: (200, root, "") if u == updates.DEV_INDEX else
                      (200, b"tt_lang_light-9.9.9.dev20270101-py3-none-any.whl", "") if u.endswith("2026-10/") else (404, b"", "x")))
    r = row(run(tmp_path, dev)[0], "tt-lang")
    assert r.status == CANDIDATE and "9.9.9.dev20270101" in r.upstream
