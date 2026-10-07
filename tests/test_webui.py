# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The browser UI (orchard/webui.py, `tt-orchard ui`).

Every outside signal is injected, so no test touches a chip, gozer, the hub or a real supervisor.
The tests that matter most prove what must never happen: a request from another site changing a run,
a file outside the allow-list (or a token inside an allowed one) reaching the browser, a control word
sent to a run that cannot act on it, a launch while the preflight blocks, and the server listening
beyond this machine.
"""
import http.client
import json
import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import cli, status, webui
from orchard.ledger import Ledger
from orchard.preflight import BLOCK, OK, WARN, Check

GOZER = """grain: board   (2 boards, 4 chips)
board 000004613193402F  (p300c)
  chip 0  0000:01:00.0  HELD             orchard:supervisor pid 4242
  chip 1  0000:02:00.0  HELD             orchard:supervisor pid 4242
board 0000046131934060  (p300c)
  chip 2  0000:03:00.0  FREE
  chip 3  0000:04:00.0  STALE            orchard:supervisor pid 99  — stale; clear it with: gozer reconcile
"""
TOKEN = "test-token-123"
HOME = "/home/operator"


def write_ledger(run: Path, *events, model="org/some-model"):
    run.mkdir(parents=True, exist_ok=True)
    fresh = not (run / "ledger.jsonl").exists()
    with Ledger(run / "ledger.jsonl") as led:
        if fresh:
            led.append("run_start", None, model=model, versions={}, inputs={})
        for ev, stage, data in events:
            led.append(ev, stage, **data)
    return run


class Spawns:
    """Records detached launches instead of starting anything."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, *, log_path, cwd):
        self.calls.append({"argv": list(argv), "log_path": Path(log_path), "cwd": Path(cwd)})
        return 31337


class Gozer:
    def __init__(self, text=GOZER):
        self.text, self.calls = text, 0

    def __call__(self):
        self.calls += 1
        return self.text


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


def make_app(root, *, alive=(), preflight=None, gozer=None, clock=None, ports=None, launched_alive=None):
    """A WebApp whose supervisor liveness, gozer, ports, preflight and launches are all fakes.
    `alive` names the runs whose supervisor counts as alive (via their supervisor.pid)."""
    def pid_alive(pid):
        return pid in alive

    return webui.WebApp(
        runs_root=root,
        config_path=root / "bringup.toml",
        cfg=SimpleNamespace(runs_root=root, coder=SimpleNamespace(port=8001, target="org/coder", chips=2)),
        token=TOKEN,
        gozer_status=gozer or Gozer(),
        clock=clock or time.time,
        spawn=Spawns(),
        preflight=preflight or (lambda model, base: [Check("hub", OK, "found")]),
        collect_kwargs={"pid_alive": pid_alive, "lock_holder": lambda p: None,
                        "disk_free_gb": lambda p: 321.0},
        port_open=ports or (lambda port: port == 11434),
        pid_running=launched_alive or (lambda pid: False),
        hostname="quietbox", home=HOME, sse_poll_s=0.05,
    )


def with_supervisor(run: Path, pid: int):
    (run / "supervisor.pid").write_text(f"{pid}\n")


# ---- reading ----------------------------------------------------------------------------------

def test_runs_are_the_children_of_runs_root_that_have_a_ledger(root):
    write_ledger(root / "org--a")
    (root / "cache").mkdir()                       # the default cache root is not a run
    (root / "notes").mkdir()
    (root / "org--b.ui-launch.log").write_text("x")
    names = [r["name"] for r in make_app(root).list_runs()]
    assert names == ["org--a"]


def test_a_run_summary_comes_from_status_collect(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    with_supervisor(run, 4242)
    app = make_app(root, alive={4242})
    row = app.list_runs()[0]
    facts = status.collect(run, **app.collect_kwargs, gozer_status=lambda: GOZER)
    assert row["state"] == facts["state"] == "running"
    assert row["model"] == "org/some-model"
    assert row["stage"] == facts["stage"]
    assert row["hint"] == facts["hint"]


def test_runs_that_need_attention_sort_first(root):
    write_ledger(root / "org--ready", ("decision", None, {"decision": "ready for operator review",
                                                           "bundle": "stages/8/bundle"}))
    write_ledger(root / "org--blocked", ("stage_start", 1, {}),
                 ("decision", 1, {"decision": "blocked", "code": "stage-failed", "reason": "stage 1 failed"}))
    run = write_ledger(root / "org--running", ("stage_start", 0, {}))
    with_supervisor(run, 7)
    names = [r["name"] for r in make_app(root, alive={7}).list_runs()]
    assert names.index("org--blocked") < names.index("org--running") < names.index("org--ready")


def test_a_corrupt_ledger_is_listed_as_unreadable_not_an_error(root):
    run = write_ledger(root / "org--bad", ("stage_start", 0, {}))
    text = (run / "ledger.jsonl").read_text().replace('"org/some-model"', '"org/edited"')   # breaks line 2's prev
    (run / "ledger.jsonl").write_text(text)
    row = make_app(root).list_runs()[0]
    assert row["state"] == "unreadable" and row["error"]


def test_run_detail_adds_the_block_digest(root):
    write_ledger(root / "org--b", ("stage_start", 1, {}),
                 ("decision", 1, {"decision": "blocked", "code": "agent-stuck", "reason": "watchdog"}))
    d = make_app(root).run_detail("org--b")
    assert d["state"] == "blocked"
    assert d["block"]["code"] == "agent-stuck"
    assert d["block"]["unblock"] and all(isinstance(x, str) for x in d["block"]["unblock"])
    assert isinstance(d["block"]["tried"], list)


def test_gozer_is_asked_at_most_once_per_interval(root):
    write_ledger(root / "org--a")
    write_ledger(root / "org--b")
    t = [1000.0]
    gz = Gozer()
    app = make_app(root, gozer=gz, clock=lambda: t[0])
    app.list_runs()
    app.machine()
    app.run_detail("org--a")
    assert gz.calls == 1
    t[0] += webui.GOZER_TTL_S + 1
    app.machine()
    assert gz.calls == 2


def test_the_machine_panel_parses_boards_and_chips(root):
    m = make_app(root).machine()
    assert [b["id"] for b in m["boards"]] == ["000004613193402F", "0000046131934060"]
    assert [b["kind"] for b in m["boards"]] == ["p300c", "p300c"]
    chips = [c for b in m["boards"] for c in b["chips"]]
    assert [c["state"] for c in chips] == ["HELD", "HELD", "FREE", "STALE"]
    assert chips[0]["owner"] == "orchard:supervisor pid 4242"
    assert chips[2]["owner"] == ""
    assert m["coder"] == {"port": 8001, "up": False}
    assert m["cpu_tier"]["up"] is True
    assert m["disk_free_gb"] == 321.0


def test_an_unknown_run_name_is_not_found(root):
    with pytest.raises(webui.NotFound):
        make_app(root).run_detail("../etc")
    with pytest.raises(webui.NotFound):
        make_app(root).run_detail("org--missing")


# ---- the live feed ------------------------------------------------------------------------------

def test_the_feed_turns_ledger_entries_into_structured_lines(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}),
                       ("decision", None, {"decision": "pause", "reason": "operator"}))
    feed = webui.Feed(run, replay=True)
    events = feed.poll()
    assert events, "replay=True shows the history"
    for e in events:
        assert set(e) >= {"ts", "actor", "icon", "role", "text", "bad"}
    assert any("pause" in e["text"] for e in events)


def test_the_feed_sends_only_new_lines_after_the_first_poll(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    feed = webui.Feed(run, replay=False)
    assert feed.poll() == []
    write_ledger(run, ("stage_start", 1, {}))
    new = feed.poll()
    assert len(new) >= 1 and all("1" in e["text"] or e["actor"] for e in new)
    assert feed.poll() == []


def test_the_feed_is_redacted_like_the_files(root):
    # An agent that cats a token file, or a path in the operator's home, must not put either in the page.
    run = write_ledger(root / "org--a", ("decision", None, {"decision": "pause",
                                                           "reason": f"saw hf_{'d' * 34} in {HOME}/x"}))
    app = make_app(root)
    events = webui.Feed(run, replay=True, redact=app.redact).poll()
    text = " ".join(e["text"] for e in events)
    assert "pause" in text and "hf_d" not in text and HOME not in text


# ---- file access ----------------------------------------------------------------------------------

def test_allowed_files_are_served_with_tokens_and_home_paths_redacted(root):
    run = write_ledger(root / "org--a")
    (run / "BLOCKED.md").write_text(
        f"# Blocked\nsee {HOME}/orchard-runs/org--a on quietbox with hf_{'a' * 34}\n")
    text = make_app(root).read_file("org--a", "BLOCKED.md")
    assert "hf_" not in text and HOME not in text and "quietbox" not in text
    assert "<token>" in text and "~" in text and "<host>" in text


@pytest.mark.parametrize("path", ["ledger.jsonl", "control", "../org--b/BLOCKED.md", "/etc/passwd",
                                  "stages/0/log/run-00001.jsonl", "supervisor.pid", "stages/0/../../x",
                                  "home/.netrc"])
def test_files_outside_the_allow_list_are_refused(root, path):
    run = write_ledger(root / "org--a")
    (run / "control").write_text("pause\n")
    with pytest.raises(webui.Forbidden):
        make_app(root).read_file("org--a", path)


def test_a_symlink_that_leaves_the_run_is_refused(root, tmp_path):
    run = write_ledger(root / "org--a")
    secret = tmp_path / "secret.txt"
    secret.write_text("hf_" + "b" * 34)
    (run / "BLOCKED.md").symlink_to(secret)
    with pytest.raises(webui.Forbidden):
        make_app(root).read_file("org--a", "BLOCKED.md")


def test_stage_and_bundle_files_are_allowed_and_listed(root):
    run = write_ledger(root / "org--a")
    for rel in ("stages/0/delta.json", "stages/2/evidence/swap-check.json", "stages/8/bundle/RESULTS.md",
                "stages/4/tests/2/test-result.json", "coder.log"):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text("{}" if rel.endswith(".json") else "ok\n")
    (run / "stages/0/log").mkdir(parents=True)
    (run / "stages/0/log/run-00001.jsonl").write_text("{}\n")
    app = make_app(root)
    listed = app.run_detail("org--a")["files"]
    assert "stages/8/bundle/RESULTS.md" in listed and "coder.log" in listed
    assert not any(f.endswith(".jsonl") for f in listed)
    for rel in listed:
        app.read_file("org--a", rel)


def test_a_long_log_is_served_as_its_tail(root):
    run = write_ledger(root / "org--a")
    (run / "coder.log").write_text("early\n" + "x" * (webui.FILE_TAIL_BYTES + 10) + "\nlast line\n")
    text = make_app(root).read_file("org--a", "coder.log")
    assert "last line" in text and "early" not in text


# ---- control --------------------------------------------------------------------------------------

def test_pause_and_abort_write_the_control_word_for_a_running_run(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    with_supervisor(run, 5)
    app = make_app(root, alive={5})
    app.control("org--a", "pause")
    assert (run / "control").read_text().strip() == "pause"
    app.control("org--a", "abort")
    assert (run / "control").read_text().strip() == "abort"


@pytest.mark.parametrize("word,events,alive,ok", [
    ("resume", [("stage_start", 0, {})], True, False),                       # running: nothing to resume
    ("pause", [("stage_start", 0, {})], False, False),                       # no supervisor to hear it
    ("abort", [("decision", None, {"decision": "abort", "by": "operator"})], False, False),
    ("resume", [("stage_start", 0, {}), ("decision", 0, {"decision": "pause", "reason": "operator"})], True, True),
    ("publish", [("stage_start", 0, {})], True, False),                      # not a control word
])
def test_a_control_word_is_refused_when_the_run_cannot_act_on_it(root, word, events, alive, ok):
    run = write_ledger(root / "org--a", *events)
    with_supervisor(run, 5)
    app = make_app(root, alive={5} if alive else set())
    if ok:
        app.control("org--a", word)
        assert (run / "control").read_text().strip() == word
    else:
        with pytest.raises(webui.Conflict):
            app.control("org--a", word)
        assert not (run / "control").exists()


# ---- launching ------------------------------------------------------------------------------------

def test_a_retry_relaunches_the_same_bringup_detached(root):
    write_ledger(root / "org--a", ("stage_start", 1, {}),
                 ("decision", 1, {"decision": "blocked", "code": "stage-failed", "reason": "x"}),
                 model="org/a")
    app = make_app(root)
    app.retry("org--a")
    call = app.spawn.calls[0]
    argv = call["argv"]
    assert argv[1].endswith("bin/tt-orchard") and argv[2:4] == ["bringup", "org/a"]
    assert argv[argv.index("--run-dir") + 1] == str(root / "org--a")
    assert argv[argv.index("--config") + 1] == str(root / "bringup.toml")
    assert "--quiet" in argv
    assert call["log_path"] == root / "org--a.ui-launch.log"


def test_a_running_run_cannot_be_retried(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    with_supervisor(run, 5)
    app = make_app(root, alive={5})
    with pytest.raises(webui.Conflict):
        app.retry("org--a")
    assert app.spawn.calls == []


def test_a_second_launch_is_refused_while_the_first_is_still_starting(root):
    write_ledger(root / "org--a", ("decision", 1, {"decision": "blocked", "code": "x", "reason": "y"}), model="org/a")
    app = make_app(root, launched_alive=lambda pid: pid == 31337)
    app.retry("org--a")
    with pytest.raises(webui.Conflict):
        app.retry("org--a")
    assert len(app.spawn.calls) == 1


def test_a_bringup_is_refused_while_the_preflight_blocks(root):
    app = make_app(root, preflight=lambda m, b: [Check("hub", OK, "found"),
                                                   Check("gozer", BLOCK, "chips busy", "hardware-unhealthy")])
    with pytest.raises(webui.Conflict) as exc:
        app.bringup("org/new", None)
    assert "hardware-unhealthy" in str(exc.value)
    assert app.spawn.calls == []


def test_a_bringup_with_warnings_is_launched_with_its_base(root):
    seen = []
    app = make_app(root, preflight=lambda m, b: seen.append((m, b)) or [Check("credentials", WARN, "visible")])
    out = app.bringup("org/new", "episod/some-bundle")
    argv = app.spawn.calls[0]["argv"]
    assert seen == [("org/new", "episod/some-bundle")]
    assert argv[2:4] == ["bringup", "org/new"]
    assert argv[argv.index("--base") + 1] == "episod/some-bundle"
    assert out["run"] == "org--new" and out["pid"] == 31337


@pytest.mark.parametrize("model", ["", "noslash", "a/b/c", "org/name; rm -rf ~", "-x/y", "org/ name"])
def test_a_malformed_model_id_is_refused_before_the_preflight(root, model):
    called = []
    app = make_app(root, preflight=lambda m, b: called.append(m) or [])
    with pytest.raises(webui.BadRequest):
        app.preflight(model, None)
    with pytest.raises(webui.BadRequest):
        app.bringup(model, None)
    assert called == [] and app.spawn.calls == []


def test_the_preflight_reports_checks_and_base_candidates(root):
    cands = [{"name": "episod/x", "installed": True}]
    app = make_app(root, preflight=lambda m, b: [
        Check("hub", OK, "found"),
        Check("base", BLOCK, "no bundle", "nearest-model-missing", {"base": "google/g", "candidates": cands})])
    out = app.preflight("org/new", None)
    assert out["ok"] is False
    assert out["run"] == "org--new" and out["resuming"] is False
    base = [c for c in out["checks"] if c["name"] == "base"][0]
    assert base["status"] == "block" and base["reason"] == "nearest-model-missing"
    assert base["candidates"] == cands


def test_the_real_spawn_starts_a_new_session(tmp_path, monkeypatch):
    seen = {}

    class P:
        pid = 4321

    def popen(argv, **kw):
        seen.update(kw)
        seen["argv"] = argv
        return P()

    monkeypatch.setattr(subprocess, "Popen", popen)
    pid = webui.spawn_detached(["echo", "hi"], log_path=tmp_path / "x.log", cwd=tmp_path)
    assert pid == 4321
    assert seen["start_new_session"] is True
    assert seen["stdin"] == subprocess.DEVNULL


def test_launches_show_the_output_of_a_bringup_started_here(root):
    app = make_app(root, launched_alive=lambda pid: True)
    app.bringup("org/new", None)
    (root / "org--new.ui-launch.log").write_text(f"preflight ok\nfetching into {HOME}/hf with hf_{'c' * 34}\n")
    rows = app.launches()
    assert [r["run"] for r in rows] == ["org--new"]
    assert rows[0]["alive"] is True and rows[0]["model"] == "org/new"
    assert "fetching into ~/hf" in rows[0]["log_tail"] and "hf_c" not in rows[0]["log_tail"]


# ---- the HTTP server --------------------------------------------------------------------------------

class Served:
    def __init__(self, app):
        self.server = webui.make_server(app, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        h.update(headers or {})
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            h["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        return resp.status, dict(resp.getheaders()), raw

    def post(self, path, body, *, token=TOKEN, origin=None):
        headers = {"X-Orchard-Token": token} if token else {}
        headers["Origin"] = origin or f"http://127.0.0.1:{self.port}"
        return self.request("POST", path, body, headers)


def test_the_page_and_its_assets_are_served_with_a_strict_policy(root):
    with Served(make_app(root)) as s:
        code, headers, body = s.request("GET", "/")
        assert code == 200 and b"<title>" in body
        assert "default-src 'self'" in headers["Content-Security-Policy"]
        for asset in ("/static/app.js", "/static/app.css"):
            code, _, _ = s.request("GET", asset)
            assert code == 200
        assert s.request("GET", "/static/../webui.py")[0] == 404


def test_meta_gives_the_token_and_the_vocabulary(root):
    with Served(make_app(root)) as s:
        code, _, body = s.request("GET", "/api/meta")
        meta = json.loads(body)
    assert code == 200 and meta["token"] == TOKEN
    assert set(meta["states"]) == set(status.STATES)
    assert set(meta["palette"]) >= {"good", "warn", "bad", "alarm", "dim"}
    assert meta["stages"][0]["real"] and meta["stages"][0]["name"]


def test_a_post_without_the_token_is_forbidden_and_changes_nothing(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    with_supervisor(run, 5)
    with Served(make_app(root, alive={5})) as s:
        assert s.post("/api/runs/org--a/control", {"word": "abort"}, token=None)[0] == 403
        assert s.post("/api/runs/org--a/control", {"word": "abort"}, token="wrong")[0] == 403
    assert not (run / "control").exists()


def test_a_post_from_another_site_is_forbidden_even_with_the_token(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    with_supervisor(run, 5)
    with Served(make_app(root, alive={5})) as s:
        assert s.post("/api/runs/org--a/control", {"word": "abort"}, origin="http://evil.example")[0] == 403
    assert not (run / "control").exists()


def test_a_request_for_another_host_name_is_refused(root):
    # A DNS-rebinding page reaches 127.0.0.1 under its own name. The Host header gives it away.
    with Served(make_app(root)) as s:
        assert s.request("GET", "/api/meta", host="evil.example:8780")[0] == 403
        assert s.request("GET", "/api/meta", host=f"localhost:{s.port}")[0] == 200


def test_control_over_http_writes_the_word_and_answers_with_the_new_state(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    with_supervisor(run, 5)
    with Served(make_app(root, alive={5})) as s:
        code, _, body = s.post("/api/runs/org--a/control", {"word": "pause"})
    assert code == 200 and json.loads(body)["control_pending"] == "pause"
    assert (run / "control").read_text().strip() == "pause"


def test_errors_become_status_codes(root):
    write_ledger(root / "org--a", ("stage_start", 0, {}))
    with Served(make_app(root)) as s:
        assert s.request("GET", "/api/runs/org--nope")[0] == 404
        assert s.request("GET", "/api/runs/org--a/file?path=ledger.jsonl")[0] == 403
        assert s.post("/api/runs/org--a/control", {"word": "pause"})[0] == 409     # no supervisor
        assert s.post("/api/bringup", {"model": "bad"})[0] == 400
        assert s.request("GET", "/api/nothing")[0] == 404


def test_the_event_stream_sends_new_lines_as_they_are_written(root):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    with Served(make_app(root)) as s:
        conn = http.client.HTTPConnection("127.0.0.1", s.port, timeout=5)
        conn.request("GET", "/api/runs/org--a/events?replay=0", headers={"Host": f"127.0.0.1:{s.port}"})
        resp = conn.getresponse()
        assert resp.status == 200 and resp.getheader("Content-Type").startswith("text/event-stream")
        assert resp.fp.readline().startswith(b":")          # the opening comment
        write_ledger(run, ("decision", None, {"decision": "pause", "reason": "operator"}))
        deadline = time.time() + 5
        got = None
        while time.time() < deadline:
            line = resp.fp.readline()
            if line.startswith(b"data:"):
                got = json.loads(line[5:])
                break
        conn.close()
    assert got is not None and "pause" in got["text"]


def test_the_ledger_endpoint_pages_by_sequence(root):
    write_ledger(root / "org--a", ("stage_start", 0, {}), ("stage_end", 0, {"result": "pass"}))
    with Served(make_app(root)) as s:
        _, _, body = s.request("GET", "/api/runs/org--a/ledger?after=1")
    rows = json.loads(body)["entries"]
    assert [r["seq"] for r in rows] == [2, 3]
    assert rows[-1]["summary"] == "stage 0 pass"


# ---- the command ------------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.50.51", "example.com", "::"])
def test_the_command_refuses_to_listen_beyond_this_machine(host, capsys):
    code = cli.main(["ui", "--host", host], env={"HOME": "/nonexistent"}, stdout=None)
    assert code == cli.EXIT_REFUSED
    assert "loopback" in capsys.readouterr().err


def test_no_asset_loads_anything_from_another_host():
    web = Path(webui.__file__).with_name("web")
    for f in web.iterdir():
        text = f.read_text(encoding="utf-8")
        assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", text), f.name
        assert "fetch('http" not in text and 'fetch("http' not in text, f.name


def test_the_page_never_uses_blocking_browser_dialogs():
    js = Path(webui.__file__).with_name("web").joinpath("app.js").read_text(encoding="utf-8")
    assert not re.search(r"\b(alert|confirm|prompt)\(", js)
