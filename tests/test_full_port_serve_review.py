# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 5 of a full port: the findings of the Copilot review of PR 490. Each test names the finding. The files are real directories and real
links under a temporary directory; no test touches a chip."""
import json
import os
from types import SimpleNamespace

import pytest

from orchard import full_port, full_port_serve, full_port_serve_stage, safeio
from orchard.adapters import ChipState
from orchard.supervisor import Supervisor
import port_serve_fakes as fk
from test_full_port_serve import good, record_of, refused, rewrite, run, sha  # noqa: F401  (good and run are fixtures)


# ---- the port copy anchors both ends beneath the run directory --------------------------------------------------------------------

def outside_port(tmp_path):
    out = tmp_path / "outside"
    (out / "port").mkdir(parents=True)
    (out / "port" / "port_model.py").write_text("SECRET = 1\n")
    return out


def test_a_linked_stage_directory_as_the_source_is_refused_and_nothing_is_read_from_outside(run, tmp_path):
    """`stages/2` itself replaced by a link to a directory outside the run (finding: an ancestor link let the copy read outside)."""
    out = outside_port(tmp_path)
    stage2 = run["run"] / "stages" / "2"
    os.rename(stage2, run["run"] / "stages" / "2.orig")
    stage2.symlink_to(out, target_is_directory=True)
    files, why = full_port_serve_stage.copy_port(run["run"], 2, run["stage"])
    assert files == [] and "stages/2/port/" in why
    assert not (run["stage"] / "port" / "port_model.py").exists() or "SECRET" not in (run["stage"] / "port" / "port_model.py").read_text()


def test_a_linked_stage_5_directory_is_refused_and_no_file_outside_the_run_is_removed(run, tmp_path):
    """`stages/5` replaced by a link: the removal of an old port and the writes must not go through it (finding: shutil.rmtree followed it)."""
    out = tmp_path / "outside5"
    (out / "port").mkdir(parents=True)
    (out / "port" / "victim.py").write_text("precious\n")
    stage5 = run["stage"]
    os.rename(stage5, run["run"] / "stages" / "5.orig")
    stage5.symlink_to(out, target_is_directory=True)
    files, why = full_port_serve_stage.copy_port(run["run"], 2, stage5)
    assert files == [] and why
    assert (out / "port" / "victim.py").read_text() == "precious\n"
    assert [p.name for p in (out / "port").iterdir()] == ["victim.py"], "something was written through the link"


def test_a_linked_port_directory_in_stage_5_is_removed_itself_and_its_target_is_kept(run, tmp_path):
    out = tmp_path / "outside_port5"
    out.mkdir()
    (out / "victim.py").write_text("precious\n")
    shutil_target = run["stage"] / "port"
    if shutil_target.is_dir():
        import shutil
        shutil.rmtree(shutil_target)
    shutil_target.symlink_to(out, target_is_directory=True)
    files, why = full_port_serve_stage.copy_port(run["run"], 2, run["stage"])
    assert why is None and files
    assert (out / "victim.py").read_text() == "precious\n" and [p.name for p in out.iterdir()] == ["victim.py"]
    assert not (run["stage"] / "port").is_symlink()


def test_the_copy_reads_no_more_of_a_huge_directory_than_the_entry_cap_allows(run, monkeypatch):
    """Finding: `sorted(os.scandir(directory))` read the whole directory before MAX_ENTRIES was checked."""
    src = run["run"] / "stages" / "2" / "port"
    for i in range(300):
        (src / f"f{i:04d}.py").write_text("pass\n")
    monkeypatch.setattr(full_port_serve_stage, "MAX_ENTRIES", 5)
    taken = []
    real = os.scandir

    class Counting:
        def __init__(self, it):
            self.it = it

        def __iter__(self):
            return self

        def __next__(self):
            item = next(self.it)
            taken.append(item.name)
            return item

        def close(self):
            self.it.close()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.it.close()

    monkeypatch.setattr(os, "scandir", lambda p: Counting(real(p)))
    files, why = full_port_serve_stage.copy_port(run["run"], 2, run["stage"])
    assert files == [] and "more than 5 entries" in why
    assert len(taken) <= 7, f"the scan took {len(taken)} entries of a directory that holds 301"


def test_a_port_nested_deeper_than_the_depth_cap_is_refused_and_not_half_copied(run, monkeypatch):
    monkeypatch.setattr(full_port_serve_stage, "MAX_DEPTH", 2)
    deep = run["run"] / "stages" / "2" / "port" / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "x.py").write_text("pass\n")
    files, why = full_port_serve_stage.copy_port(run["run"], 2, run["stage"])
    assert files == [] and "more than 2 directory levels" in why


# ---- the corpus builder compares the record's hashes with the files it reads ------------------------------------------------------

def edit_json(path, fn):
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def test_a_stage_1_prompt_changed_after_the_stage_passed_is_not_paired_with_the_records_old_outputs(run):
    """Finding: the builder never compared the record's prompt hash with the file it read."""
    prompt = run["run"] / full_port.PROMPT_FILE
    edit_json(prompt, lambda d: d.update(prompt_ids=[7, 8, 9, 10]))
    corpus, problems = full_port_serve.build_corpus(run["run"], 2)
    assert corpus is None and any("prompt" in p and "record" in p for p in problems), problems


def test_a_stage_1_reference_changed_after_the_stage_passed_is_refused(run):
    ref = run["run"] / full_port.REFERENCE_FILE
    edit_json(ref, lambda d: d.update(generated_ids=list(d["generated_ids"])[:-1] + [199]))
    corpus, problems = full_port_serve.build_corpus(run["run"], 2)
    assert corpus is None and any("reference" in p and "record" in p for p in problems), problems


def test_a_stage_1_reference_extended_after_the_stage_passed_is_refused(run):
    """Extra ids at the end keep the record's reference_ids a prefix of the file, so only the hash tells the file changed."""
    ref = run["run"] / full_port.REFERENCE_FILE
    edit_json(ref, lambda d: d.update(generated_ids=list(d["generated_ids"]) + [5, 5, 5]))
    corpus, problems = full_port_serve.build_corpus(run["run"], 2)
    assert corpus is None and any("reference" in p and "record" in p for p in problems), problems


def test_a_record_that_names_other_stage_1_files_is_refused(run):
    rec = run["run"] / full_port_serve.record_path(2)
    edit_json(rec, lambda d: d["prompt"].update(path="stages/1/evidence/reference/other.json"))
    corpus, problems = full_port_serve.build_corpus(run["run"], 2)
    assert corpus is None and any("prompt" in p and "record" in p for p in problems), problems


# ---- max_len_of reads a Hugging Face cache snapshot ---------------------------------------------------------------------------------

def test_max_len_of_follows_the_config_link_of_a_hugging_face_snapshot(tmp_path):
    """Finding: a snapshot's config.json is a link into blobs/, which `safeio.read_json` refuses, so the limit fell back to 2048."""
    repo = tmp_path / "models--org--m"
    (repo / "blobs").mkdir(parents=True)
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "blobs" / "deadbeef").write_text(json.dumps({"max_position_embeddings": 1024}))
    (repo / "snapshots" / "abc" / "config.json").symlink_to("../../blobs/deadbeef")
    assert full_port_serve.max_len_of(repo / "snapshots" / "abc") == 1024
    assert 1024 != full_port_serve.DEFAULT_MAX_LEN


def test_max_len_of_keeps_the_default_for_a_config_without_a_limit_or_a_missing_one(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert full_port_serve.max_len_of(tmp_path) == full_port_serve.DEFAULT_MAX_LEN
    assert full_port_serve.max_len_of(tmp_path / "nothing") == full_port_serve.DEFAULT_MAX_LEN


# ---- a malformed record is a reason, not a crash --------------------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    lambda r: r["boot"].update(kernel_cache=[]),
    lambda r: r["boot"].update(kernel_cache="warm"),
    lambda r: r["passkey"]["runs"][0].pop("length"),
    lambda r: r["passkey"].update(runs=[1, 2]),
    lambda r: r["passkey"].update(runs=[{"key": "1", "answer_text": "1", "ids_first": [1], "ids_second": [1]}] * 2),
])
def test_derive_returns_a_reason_for_a_record_with_a_malformed_nested_field(good, tmp_path, edit):
    b = fk.clone(good, tmp_path)
    rewrite(b, edit)
    result, why = full_port_serve.derive(b["stage"], b["run"])
    assert result is None and why, "derive must return (None, why), never raise"


# ---- two passkey lengths that measure the same are not two lengths --------------------------------------------------------------------

def passkey_run(length, tokens, key):
    return {"length": length, "key": key, "answer_text": key, "ids_first": [1], "ids_second": [1], "prompt_tokens": tokens}


def test_two_planned_lengths_that_the_server_measured_as_the_same_prompt_are_refused(tmp_path):
    """Finding: with a 128-token context the plan asks for 32 and 64 tokens, the filler comes out the same size for both, and the gate accepted
    both counts. The record has to show two different measured prompt lengths."""
    reasons: list = []
    record = {"passkey": {"runs": [passkey_run(32, 40, "11111"), passkey_run(64, 40, "22222")]}}
    full_port_serve._passkey_reasons(record, None, tmp_path, reasons)
    assert any("same prompt length" in r for r in reasons), reasons


def test_two_planned_lengths_with_different_measured_prompts_are_accepted(tmp_path):
    reasons: list = []
    record = {"passkey": {"runs": [passkey_run(32, 30, "11111"), passkey_run(64, 60, "22222")]}}
    full_port_serve._passkey_reasons(record, None, tmp_path, reasons)
    assert not any("same prompt length" in r for r in reasons), reasons


# ---- --resume-aborted: stage 5 with no coder lease -------------------------------------------------------------------------------------

class Calls:
    def __init__(self):
        self.taken, self.ran, self.given_back = [], [], []


def stub_supervisor(chips, calls):
    s = object.__new__(Supervisor)
    s.coder_lease = None                                   # after an abort the coder's lease was released; stage 5 has needs_coder False
    s.adapter = SimpleNamespace(status=lambda: chips)
    s.ledger = SimpleNamespace(append=lambda *a, **k: {"seq": 1})
    s._release_test_lease = lambda: True
    s._take_test_lease = lambda spec, shape, boards, exact: (calls.taken.append((shape.boards, boards, exact)) or "lease")
    s._run_test = lambda spec, stage_dir, test, lease, **kw: calls.ran.append((lease, kw)) or {"returncode": 0, "timed_out": False, "seconds": 1.0}
    s._cleanup_after_test = lambda after_test: None
    s._give_back_test_lease = lambda n, lease: calls.given_back.append(lease)
    return s


def four_chips(busy=False):
    out = []
    for board, serial in enumerate(("b0", "b1")):
        for k in range(2):
            out.append(ChipState(f"0000:0{board * 2 + k + 1}:00.0", "BUSY-UNTRACKED" if busy else "FREE", None, board=serial, dev_index=board * 2 + k))
    return out


@pytest.mark.parametrize("busy", [False, True])
def test_the_wide_phase_without_a_coder_lease_takes_the_whole_test_lease_and_does_not_crash(busy, tmp_path):
    calls = Calls()
    s = stub_supervisor(four_chips(busy), calls)
    spec = SimpleNamespace(number=5, skill=full_port_serve.SKILL, boards=1)
    s._wide_hardware_phase(spec, tmp_path, {}, 2)
    assert [t[:2] for t in calls.taken] == [(2, 2)], "all the boards come from one lease of its own (it waits when they are busy)"
    assert calls.ran == [("lease", {"boards": 2})] and calls.given_back == ["lease"]


# ---- the second round of the review of PR 490 -------------------------------------------------------------------------------------

def test_a_config_with_max_len_as_a_string_is_a_reason_and_not_a_skipped_plan_check(tmp_path):
    reasons: list = []
    (tmp_path / full_port_serve.CONFIG_NAME).write_text(json.dumps({"max_len": "2048", "passkeys": []}))
    record = {"passkey": {"runs": [passkey_run(32, 30, "11111"), passkey_run(64, 60, "22222")]}}
    full_port_serve._passkey_reasons(record, "0" * 64, tmp_path, reasons)
    assert any("max_len" in r and "whole number" in r for r in reasons), reasons


def test_an_unreadable_config_is_a_reason_and_not_a_skipped_plan_check(tmp_path):
    reasons: list = []
    record = {"passkey": {"runs": [passkey_run(32, 30, "11111"), passkey_run(64, 60, "22222")]}}
    full_port_serve._passkey_reasons(record, "0" * 64, tmp_path, reasons)
    assert any("max_len" in r for r in reasons), reasons


@pytest.mark.parametrize("name", ["boots", "canary"])
def test_a_result_whose_boot_or_canary_flag_is_false_over_a_passing_record_is_refused(run, name):
    from test_full_port_serve import refused
    r = json.loads((run["stage"] / "result.json").read_text())
    r["checks"][name]["pass"] = False
    (run["stage"] / "result.json").write_text(json.dumps(r))
    refused(run, f"checks.{name}.pass is not true")


def test_the_template_closes_the_socket_and_does_not_shut_down_a_loop_that_never_started():
    # The forbidden-port branch raises before serve_forever starts; the finally block calls shutdown() only when `server` is not None.
    from pathlib import Path
    text = (Path(full_port_serve.__file__).parent / "skills" / "full-port-templates" / "port_serve.py").read_text()
    branch = text[text.index("if port in FORBIDDEN_PORTS"):]
    branch = branch[:branch.index("raise ServeFailed")]
    assert "server.server_close()" in branch and "server = None" in branch


# ---- round 3: the gate compares each check summary with the canonical one -------------------------------------------------------------

def _edit_check(b, name, **over):
    p = b["stage"] / "result.json"
    r = json.loads(p.read_text())
    r["checks"][name].update(over)
    p.write_text(json.dumps(r))


@pytest.mark.parametrize("name,over,word", [
    ("corpus", {"matched": 1000000}, "checks.corpus.matched"),
    ("corpus", {"positions": 5}, "checks.corpus.positions"),
    ("boots", {"seconds": 0.001}, "checks.boots.seconds"),
    ("passkey", {"answered": [True, True, True]}, "checks.passkey.answered"),
    ("passkey", {"lengths": [1, 2]}, "checks.passkey.lengths"),
    ("canary", {"evidence": [None]}, "checks.canary.evidence"),
    ("corpus", {"evidence": [None]}, "checks.corpus.evidence"),
])
def test_an_altered_check_summary_is_refused(run, name, over, word):
    """Finding: `checks.corpus.matched = 1000000` and an evidence list of `[null]` passed because only the flags and the list's presence were checked."""
    _edit_check(run, name, **over)
    refused(run, word)


def test_the_unaltered_check_summaries_still_pass(run):
    reasons, _, _ = fk.gate(run)
    assert not [r for r in reasons if "checks." in r], reasons


# ---- round 3: the record is written without following a link the port planted --------------------------------------------------------

PLANTS = '''
_close_orig = close
def close(model):
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(here) == "5":
        PLANT(here)
    return _close_orig(model)
'''


def _plant_and_run(tmp_path_factory, plant_body):
    b = fk.build(tmp_path_factory.mktemp("plant"))
    port = b["stage"] / "port" / "port_model.py"
    port.write_text(port.read_text() + PLANTS.replace("PLANT(here)", plant_body))
    return b, fk.run_serve(b)


def test_a_link_at_the_temporary_name_does_not_truncate_the_file_it_points_to(tmp_path_factory, tmp_path):
    """Finding: a port could plant `evidence/port-serve.json.tmp` as a link and the write truncated its target, outside the run."""
    victim = tmp_path / "victim.txt"
    victim.write_text("precious\n")
    b, done = _plant_and_run(tmp_path_factory, f"os.symlink({str(victim)!r}, os.path.join(here, 'evidence', 'port-serve.json.tmp'))")
    assert victim.read_text() == "precious\n", done.stdout + done.stderr
    assert (b["stage"] / "evidence" / "port-serve.json").is_file() and not (b["stage"] / "evidence" / "port-serve.json").is_symlink()


def test_an_evidence_directory_replaced_by_a_link_is_refused_and_nothing_is_written_through_it(tmp_path_factory, tmp_path):
    out = tmp_path / "elsewhere"
    out.mkdir()
    b, done = _plant_and_run(
        tmp_path_factory,
        f"(os.rename(os.path.join(here, 'evidence'), os.path.join(here, 'evidence.moved')), os.symlink({str(out)!r}, os.path.join(here, 'evidence')))")
    assert done.returncode == 2 and "link" in done.stdout, done.stdout + done.stderr
    assert [p.name for p in out.iterdir() if p.name.startswith("port-serve")] == []  # (hw-test-output.txt is the test runner's own)


# ---- round 4 (second Copilot round of PR 490) -----------------------------------------------------------------------------------------

def _template(tmp_path):
    """The shipped template as a module, with its run directory moved under `tmp_path` (it opens no device on import)."""
    import importlib.util
    from pathlib import Path
    src = Path(full_port_serve.__file__).parent / "skills" / "full-port-templates" / "port_serve.py"
    spec = importlib.util.spec_from_file_location("port_serve_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run_dir = tmp_path / "run"
    (run_dir / "stages" / "5" / "evidence").mkdir(parents=True)
    mod.HERE, mod.RUN_DIR = run_dir / "stages" / "5", run_dir
    mod.RECORD = mod.HERE / "evidence" / "port-serve.json"
    return mod


def test_a_stage_directory_swapped_for_a_link_between_the_check_and_the_write_is_not_written_through(tmp_path, monkeypatch):
    """Finding: the realpath comparison and the open were two steps; swapping `stages/5` for a link in between sent the record outside the run.
    The swap happens at each point the check or the descent passes: the old code's realpath and islink calls, and the new code's descent."""
    mod = _template(tmp_path)
    outside = tmp_path / "outside"
    (outside / "evidence").mkdir(parents=True)
    swapped = []

    def swap():
        if not swapped:
            swapped.append(1)
            os.rename(mod.HERE, mod.HERE.with_name("5.moved"))
            os.symlink(outside, mod.HERE)
    real_realpath, real_islink, real_open = os.path.realpath, os.path.islink, os.open
    monkeypatch.setattr(os.path, "realpath", lambda p, *a, **k: (lambda r: (swap(), r)[1])(real_realpath(p, *a, **k)))
    monkeypatch.setattr(os.path, "islink", lambda p: (lambda r: (swap(), r)[1])(real_islink(p)))
    monkeypatch.setattr(os, "open", lambda p, *a, **k: (swap() if p == "5" else None, real_open(p, *a, **k))[1])
    with pytest.raises(mod.ServeFailed):
        mod.write_record(b"{}")
    assert list((outside / "evidence").iterdir()) == []


class _Worker:
    def __init__(self):
        self.calls = []

    def ask(self, **kw):
        self.calls.append(kw)
        return 5


class _Tok:
    def encode(self, text, add_special_tokens=False):
        return type("E", (), {"ids": [1, 2, 3]})()

    def decode(self, ids, skip_special_tokens=True):
        return "x"


def _served(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    mod = _template(tmp_path)
    worker = _Worker()
    engine = mod.Engine(worker, 100, _Tok(), [], lambda m: "t")
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return mod, engine, worker, server


def _post(mod, server, body, headers=None):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.request("POST", "/v1/completions", json.dumps(body), {"Content-Type": "application/json", **(headers or {})})
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_an_ordinary_request_is_refused_until_the_replay_has_matched_and_the_port_is_not_run(tmp_path):
    """Finding: the listener took ordinary requests during the corpus replay."""
    mod, engine, worker, server = _served(tmp_path)
    try:
        body = {"prompt": [1, 2], "max_tokens": 1}
        status, _ = _post(mod, server, body)
        assert status == 503 and worker.calls == []
        status, _ = _post(mod, server, body, {"X-Orchard-Replay": engine.replay_token})
        assert status == 200 and worker.calls
        engine.open = True
        status, _ = _post(mod, server, body)
        assert status == 200
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("body", [{"prompt": [-1], "max_tokens": 1}, {"prompt": [1], "max_tokens": 2, "forced_tokens": [-1]}])
def test_a_negative_token_id_is_a_400_and_the_port_is_not_run(tmp_path, body):
    """Finding: `prompt: [-1]` and `forced_tokens: [-1]` reached the worker."""
    mod, engine, worker, server = _served(tmp_path)
    engine.open = True
    try:
        status, _ = _post(mod, server, body)
        assert status == 400 and worker.calls == []
    finally:
        server.shutdown()
        server.server_close()


def test_a_result_that_lists_other_files_with_the_right_names_is_refused(run):
    """Finding: the gate checked only the base names of result.evidence, so copies elsewhere in the run passed as the stage's evidence."""
    import shutil
    other = run["run"] / "elsewhere"
    other.mkdir()
    r = json.loads((run["stage"] / "result.json").read_text())
    listed = []
    for rel in r["evidence"]:
        name = os.path.basename(rel)
        shutil.copy(run["run"] / rel, other / name)
        listed.append(f"elsewhere/{name}")
    edit = {"evidence": listed, "checks": {k: {**v, "evidence": listed} for k, v in r["checks"].items()}}
    (run["stage"] / "result.json").write_text(json.dumps({**r, **edit}))
    refused(run, "evidence must be exactly")


@pytest.mark.parametrize("value", [[1, 2], None])
def test_a_changed_top_level_passkey_lengths_is_refused(run, value):
    """Finding: result.json passkey_lengths was never compared with the record."""
    p = run["stage"] / "result.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "passkey_lengths": value}))
    refused(run, "passkey_lengths")


# ---- review of PR 490, round 5 -------------------------------------------------------------------------------------------------

def _served_with_vocab(tmp_path, vocab_size):
    import threading
    from http.server import ThreadingHTTPServer
    mod = _template(tmp_path)
    worker = _Worker()
    engine = mod.Engine(worker, 100, _Tok(), [], lambda m: "t", vocab_size=vocab_size)
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(engine))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    engine.open = True
    return mod, engine, worker, server


@pytest.mark.parametrize("body", [{"prompt": [999], "max_tokens": 1}, {"prompt": [1], "max_tokens": 2, "forced_tokens": [999]},
                                  {"prompt": [10], "max_tokens": 1}])
def test_an_id_outside_the_vocabulary_is_a_400_and_the_port_is_not_run(tmp_path, body):
    """Finding: a positive id beyond the embedding table reached the worker and gave an HTTP 500."""
    mod, engine, worker, server = _served_with_vocab(tmp_path, 10)
    try:
        status, _ = _post(mod, server, body)
        assert status == 400 and worker.calls == []
        status, _ = _post(mod, server, {"prompt": [9], "max_tokens": 1})          # the last id inside the range still runs
        assert status == 200 and worker.calls
    finally:
        server.shutdown()
        server.server_close()


def test_a_request_that_is_waiting_when_the_server_closes_never_reaches_the_worker(tmp_path):
    """Finding: `close` was sent while a completion could still be running under the engine lock, and a request that got the lock afterwards
    reached a closed worker. `quiesce` takes the lock, so a running completion finishes first, and refuses every later one."""
    import threading
    mod, engine, worker, server = _served_with_vocab(tmp_path, None)
    try:
        started, release = threading.Event(), threading.Event()
        real_ask = worker.ask

        def slow_ask(**kw):
            started.set()
            release.wait(10)
            return real_ask(**kw)
        worker.ask = slow_ask
        results = []
        t = threading.Thread(target=lambda: results.append(_post(mod, server, {"prompt": [1], "max_tokens": 1})[0]))
        t.start()
        assert started.wait(10)
        done = []
        q = threading.Thread(target=lambda: done.append(engine.quiesce(10)))
        q.start()
        q.join(0.3)
        assert q.is_alive() and not done                  # quiesce waits for the running completion
        release.set()
        t.join(10)
        q.join(10)
        assert done == [True] and results == [200]
        calls_before = len(worker.calls)
        status, _ = _post(mod, server, {"prompt": [1], "max_tokens": 1})
        assert status == 503 and len(worker.calls) == calls_before      # after it, a request is refused and the worker is not asked
    finally:
        server.shutdown()
        server.server_close()


# ---- an empty teacher-forced reference is a reason, not a ZeroDivisionError (PR 490, round 6) -------------------------------------------

def test_an_empty_teacher_forced_reference_with_an_empty_replay_is_a_reason_and_not_a_crash(good, tmp_path):
    b = fk.clone(good, tmp_path)
    corpus = b["stage"] / full_port_serve.CORPUS_NAME

    def empty_reference(doc):
        for e in doc["entries"]:
            if e["kind"] == "teacher-forced":
                e["reference_ids"] = []

    edit_json(corpus, empty_reference)

    def empty_replay(rec):
        for e in rec["replay"]["entries"]:
            if e["kind"] == "teacher-forced":
                e["got_ids"] = []

    rewrite(b, empty_replay)
    result, why = full_port_serve.derive(b["stage"], b["run"])
    assert result is None and why


# ---- a padded embedding table has valid ids above the tokenizer's count (PR 490, round 6) -----------------------------------------------

def _vocab_template():
    import importlib.util
    spec = importlib.util.spec_from_file_location("port_serve_vocab_bound", full_port_serve.TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("config, expected", [
    ({"vocab_size": 160}, 160),                                  # padded table: the config is larger than the tokenizer
    ({"text_config": {"vocab_size": 170}}, 170),
    ({"vocab_size": 50}, 100),                                   # a smaller config never lowers the bound
    ({"vocab_size": True}, 100), ({"vocab_size": "9"}, 100), ({"vocab_size": 10**12}, 100), ({}, 100),
])
def test_the_serving_bound_is_the_larger_of_the_tokenizer_count_and_the_configs_vocab_size(tmp_path, config, expected):
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert _vocab_template().vocab_bound(tmp_path, 100) == expected
