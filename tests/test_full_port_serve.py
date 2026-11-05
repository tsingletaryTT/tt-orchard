# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 5 of a full port (issue 329): the acceptance corpus, the serving template and the gate. The template runs for real as a subprocess on the
CPU, against a fake `ttnn` and ports that are plain Python; nothing touches a chip. Each guard of the gate has a test that edits one fact of a good run
and watches the gate name it. The supervisor's own flow is in tests/test_supervisor_full_port_serve.py."""
import hashlib
import importlib.util
import json
import re
import shutil
import threading
from http.server import ThreadingHTTPServer

import pytest

from orchard import full_port, full_port_serve, full_port_serve_stage
import port_serve_fakes as fk
from port_check_fakes import CHIPS, HONEST, REF_IDS, make_run, run_template


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    """A passing run: stage 2 passed with a port that can read its context, stage 5 ran once and wrote its record and result."""
    b = fk.build(tmp_path_factory.mktemp("good"))
    done = fk.run_serve(b)
    assert done.returncode == 0, done.stdout + done.stderr
    assert fk.derive(b)[0] is not None
    return b


@pytest.fixture
def run(good, tmp_path):
    return fk.clone(good, tmp_path)


def record_of(b):
    return json.loads((b["stage"] / "evidence" / "port-serve.json").read_text())


def rewrite(b, fn):
    """Edit the record and keep everything else consistent (the printed hash, the supervisor's output hash, the record's own file hashes), so the
    gate's reason for the edit is the only one it can give."""
    path = b["stage"] / "evidence" / "port-serve.json"
    rec = json.loads(path.read_text())
    fn(rec)
    data = (json.dumps(rec, indent=2) + "\n").encode()
    path.write_bytes(data)
    out = b["stage"] / "evidence" / "hw-test-output.txt"
    out.write_text(re.sub(r"PORT_CHECK_SHA256 [0-9a-f]{64}", f"PORT_CHECK_SHA256 {sha(data)}", out.read_text()))
    marker = json.loads((b["stage"] / "test-result.json").read_text())
    marker["output"] = full_port.file_record(b["run"], out)
    (b["stage"] / "test-result.json").write_text(json.dumps(marker))


def edit_result(b, **over):
    p = b["stage"] / "result.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), **over}))


def refused(b, word, **kw):
    reasons, _, _ = fk.gate(b, **kw)
    hit = [r for r in reasons if word in r]
    assert hit, f"the gate did not name {word!r}; it said {reasons}"
    return reasons


# ---- the corpus -------------------------------------------------------------------------------------------------------------------

def test_the_corpus_is_built_from_the_stage_1_files_and_the_accepted_record(good):
    corpus, problems = full_port_serve.build_corpus(good["run"], 2)
    assert not problems and corpus["schema"] == full_port_serve.CORPUS_SCHEMA and corpus["source_stage"] == 2
    tf, free = corpus["entries"]
    assert tf["kind"] == "teacher-forced" and tf["prompt_ids"] == [1, 2, 3] and tf["reference_ids"] == REF_IDS
    assert tf["forced_ids"] == REF_IDS[:-1] and tf["expected_ids"] == REF_IDS      # the honest port repeated the reference at every position
    assert free["kind"] == "free-run" and len(free["expected_ids"]) == 48 and free["expected_ids"][0] == 100
    assert corpus["reference"]["sha256"] == sha((good["run"] / full_port.REFERENCE_FILE).read_bytes())
    assert full_port_serve.positions(corpus) == 32 + 48


def test_the_same_inputs_give_the_same_corpus(good):
    a, _ = full_port_serve.build_corpus(good["run"], 2)
    b, _ = full_port_serve.build_corpus(good["run"], 2)
    assert full_port_serve.corpus_bytes(a) == full_port_serve.corpus_bytes(b)


def test_a_record_that_differs_from_the_ledgers_hash_is_not_a_corpus_source(run):
    corpus, problems = full_port_serve.build_corpus(run["run"], 2, accepted="0" * 64)
    assert corpus is None and "changed after the stage passed" in problems[0]


def test_a_record_whose_reference_ids_are_not_the_stage_1_ids_is_refused(run):
    path = run["run"] / "stages" / "2" / "evidence" / "port-check.json"
    rec = json.loads(path.read_text())
    rec["reference_ids"] = [i + 1 for i in rec["reference_ids"]]
    path.write_text(json.dumps(rec))
    corpus, problems = full_port_serve.build_corpus(run["run"], 2)
    assert corpus is None and "first ids of the stage 1 reference" in problems[0]


def test_too_few_accepted_positions_are_refused(run):
    path = run["run"] / "stages" / "2" / "evidence" / "port-check.json"
    rec = json.loads(path.read_text())
    rec["chip_ids"], rec["reference_ids"] = rec["chip_ids"][:8], rec["reference_ids"][:8]
    path.write_text(json.dumps(rec))
    corpus, problems = full_port_serve.build_corpus(run["run"], 2)
    assert corpus is None and "at least" in problems[0]


@pytest.mark.parametrize("field", ["chip_ids", "free_run_ids", "reference_ids"])
@pytest.mark.parametrize("bad", [None, "abc", [1, "x"], [True], []])
def test_a_malformed_list_in_the_accepted_record_is_a_reason_and_not_an_error(run, field, bad):
    path = run["run"] / "stages" / "2" / "evidence" / "port-check.json"
    rec = json.loads(path.read_text())
    rec[field] = bad
    path.write_text(json.dumps(rec))
    corpus, problems = full_port_serve.build_corpus(run["run"], 2)
    assert corpus is None and problems


def test_which_stage_is_served_follows_the_latest_end_of_each_stage():
    end = lambda n, r: {"event": "stage_end", "stage": n, "data": {"result": r}}  # noqa: E731
    assert full_port_serve.source_stage([end(2, "pass")]) == 2
    assert full_port_serve.source_stage([end(2, "pass"), end(3, "pass")]) == 3
    assert full_port_serve.source_stage([end(2, "pass"), end(3, "skipped")]) == 2
    assert full_port_serve.source_stage([end(2, "pass"), end(3, "pass"), end(3, "fail")]) == 2      # the latest end of stage 3 failed
    assert full_port_serve.source_stage([end(2, "fail")]) is None and full_port_serve.source_stage([]) is None


def test_the_passkey_plan_is_reproducible_has_two_lengths_and_needs_a_context_for_them():
    a = full_port_serve.plan_passkeys("ab" * 32, 2048)
    assert a == full_port_serve.plan_passkeys("ab" * 32, 2048) != full_port_serve.plan_passkeys("cd" * 32, 2048)
    assert [p["length"] for p in a] == [256, 1984] and all(len(p["key"]) == 5 and p["key"].isdigit() for p in a)
    assert full_port_serve.plan_passkeys("ab" * 32, 10 ** 6)[-1]["length"] == full_port_serve.PASSKEY_CAP
    assert full_port_serve.plan_passkeys("ab" * 32, 64) == []


# ---- the port copy ----------------------------------------------------------------------------------------------------------------

def test_the_port_is_copied_with_the_hash_of_each_file_and_a_cache_is_left_behind(run):
    src = run["run"] / "stages" / "2" / "port"
    (src / "__pycache__").mkdir(exist_ok=True)
    (src / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"junk")
    (src / "sub").mkdir()
    (src / "sub" / "helper.py").write_text("X = 1\n")
    files, why = full_port_serve_stage.copy_port(run["run"], 2, run["stage"])
    assert why is None and sorted(f["path"] for f in files) == ["stages/5/port/port_model.py", "stages/5/port/sub/helper.py"]
    assert all(sha((run["run"] / f["path"]).read_bytes()) == f["sha256"] for f in files)
    assert not (run["stage"] / "port" / "__pycache__").exists()


def test_a_link_in_the_port_is_refused_and_leaves_the_old_copy_alone(run):
    (run["run"] / "stages" / "2" / "port" / "evil.py").symlink_to("/etc/passwd")
    files, why = full_port_serve_stage.copy_port(run["run"], 2, run["stage"])
    assert files == [] and "evil.py is a link" in why


def test_too_many_files_and_a_file_over_the_cap_are_refused(run, monkeypatch):
    monkeypatch.setattr(full_port_serve_stage, "MAX_FILES", 1)
    (run["run"] / "stages" / "2" / "port" / "second.py").write_text("pass\n")
    assert "more than 1 files" in full_port_serve_stage.copy_port(run["run"], 2, run["stage"])[1]
    monkeypatch.setattr(full_port_serve_stage, "MAX_FILES", 200)
    monkeypatch.setattr(full_port_serve_stage, "MAX_FILE_BYTES", 10)
    assert "cannot be copied" in full_port_serve_stage.copy_port(run["run"], 2, run["stage"])[1]


def test_a_port_left_by_an_earlier_attempt_is_replaced(run):
    (run["stage"] / "port" / "stale.py").write_text("pass\n")
    files, why = full_port_serve_stage.copy_port(run["run"], 2, run["stage"])
    assert why is None and not (run["stage"] / "port" / "stale.py").exists()


def test_prepare_names_what_is_missing(run, good):
    entries = good["entries"]
    kw = {"snapshot": good["snap"], "orchard_dir": fk.REPO}
    assert "no model snapshot" in full_port_serve_stage.prepare(run["run"], run["stage"], entries, snapshot=None, orchard_dir=fk.REPO).problems[0]
    assert "no port to serve" in full_port_serve_stage.prepare(run["run"], run["stage"], [], **kw).problems[0]
    no_evidence = [{"event": "stage_end", "stage": 2, "data": {"result": "pass", "evidence": []}}]
    assert "no sha256" in full_port_serve_stage.prepare(run["run"], run["stage"], no_evidence, **kw).problems[0]
    swapped = [{"event": "stage_end", "stage": 2, "data": {"result": "pass", "evidence": [{"path": "stages/2/evidence/port-check.json", "sha256": "0" * 64}]}}]
    assert "changed after the stage passed" in full_port_serve_stage.prepare(run["run"], run["stage"], swapped, **kw).problems[0]


def test_a_model_with_too_short_a_context_is_refused_at_prepare(run, good):
    (good["snap"] / "config.json").write_text(json.dumps({"max_position_embeddings": 64}))
    try:
        out = full_port_serve_stage.prepare(run["run"], run["stage"], good["entries"], snapshot=good["snap"], orchard_dir=fk.REPO)
    finally:
        (good["snap"] / "config.json").unlink()
    assert out.problems and "too short for two passkey lengths" in out.problems[0]


# ---- a good run -------------------------------------------------------------------------------------------------------------------

def test_a_good_run_passes_the_gate_and_the_result_says_what_was_measured(good):
    reasons, seen, waived = fk.gate(good)
    assert reasons == [] and waived == []
    assert sorted(s.rsplit("/", 1)[1] for s in seen) == ["corpus.json", "hw-test-output.txt", "port-serve.json"]
    result = json.loads((good["stage"] / "result.json").read_text())
    assert result["serves"] is True and result["coherent"] is True and result["corpus_positions"] == result["corpus_matched"] == 80
    assert result["corpus_reference_agreement"] == 1.0 and result["derived_by"] == "supervisor"
    assert set(result["checks"]) == {"boots", "corpus", "passkey", "canary"} and all(c["pass"] is True for c in result["checks"].values())
    assert "random weights" in result["limits"] and result["passkey_answered"] == [True, True]


def test_the_template_ran_the_port_in_its_worker_and_the_server_answered_over_http(good):
    out = (good["stage"] / "evidence" / "hw-test-output.txt").read_text()
    assert out.count("READY orchard-port on 127.0.0.1:") == 1 and "corpus 80/80 positions" in out
    assert out.index("listening on") < out.index("READY") < out.index("passkey length")            # the replay came before any client question
    assert "POST /v1/chat/completions" in out and "POST /v1/completions" in out and "GET /v1/models" in out
    assert out.count("PORT_CHECK_SHA256 ") == 1


def test_the_record_holds_the_boot_line_the_load_time_and_the_cache_state(good):
    boot = record_of(good)["boot"]
    assert boot["ready_line"].startswith("READY orchard-port on 127.0.0.1:") and boot["load_s"] > 0 and boot["ready_s"] >= boot["load_s"]
    assert boot["models"] == ["orchard-port"] and boot["chat_template"] == "plain"
    assert set(boot["kernel_cache"]) >= {"dir", "entries_before", "entries_after", "warm", "basis"}


def test_the_cache_is_warm_when_the_directory_held_files_before_the_boot(tmp_path):
    b = fk.build(tmp_path)
    cache = tmp_path / "kcache"
    (cache / "a").mkdir(parents=True)
    (cache / "a" / "kernel.o").write_bytes(b"x")
    assert fk.run_serve(b, cache=cache).returncode == 0
    fk.derive(b)
    kc = record_of(b)["boot"]["kernel_cache"]
    assert kc["warm"] is True and kc["entries_before"] == 1
    assert json.loads((b["stage"] / "result.json").read_text())["kernel_cache_warm"] is True
    assert fk.gate(b)[0] == []


def test_the_cache_is_cold_when_the_directory_is_empty_or_missing(tmp_path):
    b = fk.build(tmp_path)
    assert fk.run_serve(b, cache=tmp_path / "no-such-dir").returncode == 0
    fk.derive(b)
    kc = record_of(b)["boot"]["kernel_cache"]
    assert kc["warm"] is False and kc["entries_before"] == 0
    assert fk.gate(b)[0] == []


def test_stage_3_is_the_source_when_it_passed_and_the_gate_expects_its_chips(tmp_path):
    b = fk.build(tmp_path, stage3=True)
    assert b["prep"].source == 3 and all(f["path"].startswith("stages/5/port/") for f in b["prep"].port_files)
    assert fk.run_serve(b).returncode == 0
    assert fk.derive(b)[0] is not None
    assert fk.gate(b)[0] == []
    refused(b, "differs", source=2)                       # stage 2's record is not the one this corpus was built from, and the config names stage 3


# ---- the model with random weights ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def random_weights(tmp_path_factory):
    b = fk.build(tmp_path_factory.mktemp("random"), fk.RANDOM_WEIGHTS)
    assert fk.run_serve(b).returncode == 0
    return b


def test_a_port_that_cannot_read_its_context_is_refused_without_the_rehearsal_option(random_weights, tmp_path):
    b = fk.clone(random_weights, tmp_path)
    result, _ = fk.derive(b)
    assert result["coherent"] is False and result["passkey_answered"] == [False, False]
    assert result["checks"]["passkey"]["pass"] is False
    refused(b, "coherent must be true")


def test_the_rehearsal_option_waives_coherent_and_the_gate_says_so(random_weights, tmp_path):
    b = fk.clone(random_weights, tmp_path)
    result, _ = fk.derive(b, rehearsal=True)
    assert result["coherent"] is False and result["checks"]["passkey"]["pass"] is True
    reasons, _, waived = fk.gate(b, rehearsal=True)
    assert reasons == [] and waived == ["coherent"]


def test_the_rehearsal_option_waives_nothing_else(random_weights, tmp_path):
    b = fk.clone(random_weights, tmp_path)
    fk.derive(b, rehearsal=True)
    rewrite(b, lambda r: r["canary"].update(second_ids=[9] * 16))
    refused(b, "state leaked", rehearsal=True)
    b2 = fk.clone(random_weights, tmp_path / "two")
    fk.derive(b2, rehearsal=True)
    rewrite(b2, lambda r: r["replay"]["entries"][0].update(got_ids=[0] * 32))
    refused(b2, "differs from the corpus", rehearsal=True)
    b3 = fk.clone(random_weights, tmp_path / "three")
    fk.derive(b3, rehearsal=True)
    rewrite(b3, lambda r: r["passkey"]["runs"][0].update(ids_second=[5]))
    refused(b3, "different ids", rehearsal=True)


def test_a_field_in_result_json_waives_nothing(random_weights, tmp_path):
    b = fk.clone(random_weights, tmp_path)
    fk.derive(b)
    edit_result(b, rehearsal=True, waive_coherent=True)
    refused(b, "coherent must be true")


# ---- the refusal at boot -------------------------------------------------------------------------------------------------------------

def test_a_server_whose_replay_differs_is_refused_before_any_ready_line(tmp_path):
    b = fk.build(tmp_path)
    (b["stage"] / "port" / "port_model.py").write_text(fk.port_source(fk.leaky(1), b["snap"]))      # the second prefill answers differently
    done = fk.run_serve(b)
    assert done.returncode == 2
    assert "FAILED: the corpus replay matched" in done.stdout and "free-run" in done.stdout and "refused" in done.stdout
    assert "READY" not in done.stdout and "passkey length" not in done.stdout
    assert not (b["stage"] / "evidence" / "port-serve.json").exists()
    assert fk.derive(b)[0] is None
    reasons, _, _ = fk.gate(b)
    assert any("exited with code 2" in r for r in reasons)


def test_a_server_that_leaks_state_after_the_replay_fails_the_canary(tmp_path):
    b = fk.build(tmp_path)
    (b["stage"] / "port" / "port_model.py").write_text(fk.port_source(fk.CARRIES_STATE, b["snap"]))
    done = fk.run_serve(b)
    assert done.returncode == 0                                       # the replay passed (two prefills); the gate judges what came after
    assert "canary DIFFERENT" in done.stdout
    fk.derive(b)
    refused(b, "state leaked")


def test_a_server_that_changes_after_the_replay_fails_the_canary_against_the_accepted_free_run(tmp_path):
    b = fk.build(tmp_path)
    (b["stage"] / "port" / "port_model.py").write_text(fk.port_source(fk.leaky(2), b["snap"]))      # prefills 1 and 2 are the replay; the rest differ
    assert fk.run_serve(b).returncode == 0
    fk.derive(b)
    refused(b, "not the first ids of the accepted port's free run")


def test_a_port_that_prints_the_mark_line_fails_the_run(tmp_path):
    b = fk.build(tmp_path)
    src = (b["stage"] / "port" / "port_model.py").read_text()
    (b["stage"] / "port" / "port_model.py").write_text(src + "\nprint('PORT_CHECK_SHA256 ' + '0' * 64)\n")
    done = fk.run_serve(b)
    assert done.returncode == 2 and "FAILED" in done.stdout


def test_a_template_copy_that_is_not_the_shipped_file_fails_at_the_worker(tmp_path):
    b = fk.build(tmp_path)
    (b["stage"] / "port_check.py").write_text("raise SystemExit(0)\n")
    done = fk.run_serve(b)
    assert done.returncode == 2 and "not the shipped file" in done.stdout


def test_a_corpus_edited_after_prepare_fails_before_the_worker_starts(tmp_path):
    b = fk.build(tmp_path)
    p = b["stage"] / "corpus.json"
    p.write_text(p.read_text().replace('"free-run"', '"free-run-x"'))
    done = fk.run_serve(b)
    assert done.returncode == 2 and "not the corpus the config names" in done.stdout


def test_no_leased_chip_is_a_failure_that_names_it(tmp_path):
    b = fk.build(tmp_path)
    done = fk.run_serve(b, chips=[])
    assert done.returncode == 2 and "TT_VISIBLE_DEVICES" in done.stdout


# ---- each guard of the gate, one at a time ---------------------------------------------------------------------------------------------

def test_a_corpus_file_that_is_not_the_one_the_supervisor_recorded_is_refused(run):
    refused(run, "not the corpus the supervisor recorded", corpus_sha="0" * 64)


def test_a_missing_ledger_hash_for_the_corpus_is_a_refusal_and_not_a_skip(run):
    refused(run, "ledger holds no sha256", corpus_sha=None)


def test_a_corpus_edited_with_its_hashes_rewritten_still_differs_from_the_rebuilt_one(run):
    p = run["stage"] / "corpus.json"
    doc = json.loads(p.read_text())
    doc["entries"][0]["expected_ids"] = [0] * 32
    p.write_text(json.dumps(doc))
    new = sha(p.read_bytes())
    rewrite(run, lambda r: r["corpus"].update(sha256=new))
    reasons, _, _ = fk.gate(run, corpus_sha=new)
    assert any("differs from the corpus the stage 1 reference" in r for r in reasons)


def test_a_replay_position_that_differs_from_the_corpus_is_refused(run):
    def alter(r):
        r["replay"]["entries"][1]["got_ids"][10] = 3
    rewrite(run, alter)
    reasons = refused(run, "differs from the corpus at position 10")
    assert "refused" in " ".join(reasons)


def test_a_replay_with_a_missing_or_doubled_entry_is_refused(run):
    rewrite(run, lambda r: r["replay"]["entries"].pop())
    refused(run, "needs exactly one")
    run2 = fk.clone(run, run["tmp"] / "dup")
    rewrite(run2, lambda r: r["replay"]["entries"].append(r["replay"]["entries"][0]))
    refused(run2, "needs exactly one")


def test_a_result_that_inflates_the_matched_count_is_refused(run):
    edit_result(run, corpus_matched=80, corpus_positions=100)
    refused(run, "corpus_positions")


def test_a_result_whose_reference_agreement_is_not_the_recomputed_one_is_refused(run):
    edit_result(run, corpus_reference_agreement=0.5)
    refused(run, "corpus_reference_agreement")


def test_a_replay_below_the_reference_bar_is_refused_even_when_it_matches_the_corpus(run):
    """A corpus can only hold ids the accepted port gave. Built from a poor port (stage 2's own gate would have refused it), the replay matches the
    corpus at every position and the agreement with the reference is the only thing that stops it."""
    p = run["run"] / "stages" / "2" / "evidence" / "port-check.json"
    rec = json.loads(p.read_text())
    rec["chip_ids"] = [0] * 32
    p.write_text(json.dumps(rec))
    run["entries"] = [fk.stage_end(2, run["run"])]
    run["prep"] = full_port_serve_stage.prepare(run["run"], run["stage"], run["entries"], snapshot=run["snap"], orchard_dir=fk.REPO)
    assert run["prep"].problems == []                            # the builder trusts the accepted record; stage 2's gate decides whether it passes
    rewrite(run, lambda r: r["replay"]["entries"][0].update(got_ids=[0] * 32))
    reasons, _, _ = fk.gate(run)
    assert any("below the minimum" in r for r in reasons), reasons
    assert not any("differs from the corpus at position" in r for r in reasons), reasons


def test_a_canary_that_changed_after_another_request_is_refused(run):
    rewrite(run, lambda r: r["canary"].update(second_ids=[1] * 16))
    refused(run, "state leaked")


def test_a_canary_that_is_not_the_accepted_free_run_is_refused(run):
    rewrite(run, lambda r: r["canary"].update(first_ids=[2] * 16, second_ids=[2] * 16))
    refused(run, "not the first ids of the accepted port's free run")


def test_a_canary_whose_middle_request_was_the_same_prompt_is_refused(run):
    rewrite(run, lambda r: r["canary"].update(between_prompt_ids=r["canary"]["prompt_ids"]))
    refused(run, "must be a different prompt")


def test_a_passkey_prompt_that_gave_other_ids_the_second_time_is_refused(run):
    rewrite(run, lambda r: r["passkey"]["runs"][1].update(ids_second=[1, 2]))
    refused(run, "different ids the second time")


def test_a_passkey_with_one_length_or_a_key_outside_the_plan_is_refused(run):
    run2 = fk.clone(run, run["tmp"] / "key")
    rewrite(run, lambda r: r["passkey"].update(runs=r["passkey"]["runs"][:1]))
    refused(run, "at least 2 prompt lengths")
    rewrite(run2, lambda r: r["passkey"]["runs"][0].update(key="00000"))
    refused(run2, "not in the plan")


def test_a_prompt_the_server_counted_as_short_is_refused(run):
    rewrite(run, lambda r: r["passkey"]["runs"][1].update(prompt_tokens=10))
    refused(run, "prompt tokens for a requested")


def test_an_answer_that_lost_its_key_makes_the_stage_incoherent_whatever_the_result_says(run):
    rewrite(run, lambda r: r["passkey"]["runs"][0].update(answer_text="no idea"))
    reasons, _, _ = fk.gate(run)
    assert any("coherent True differs from the False" in r for r in reasons) and any("passkey_answered" in r for r in reasons)


def test_a_planted_passkey_config_is_refused(run):
    p = run["stage"] / "port_serve_config.json"
    cfg = json.loads(p.read_text())
    cfg["passkeys"][0]["key"] = "11111"
    p.write_text(json.dumps(cfg))
    new = sha(p.read_bytes())
    rewrite(run, lambda r: r["config"].update(sha256=new))
    refused(run, "passkeys are not the ones")


def test_a_ready_line_that_is_not_in_the_output_is_refused(run):
    rewrite(run, lambda r: r["boot"].update(ready_line="READY orchard-port on 127.0.0.1:1 pid 1 corpus 80/80 positions"))
    refused(run, "does not appear exactly once")


def test_a_ready_line_that_admits_a_partial_replay_is_refused(run):
    line = "READY orchard-port on 127.0.0.1:1 pid 1 corpus 79/80 positions"
    out = run["stage"] / "evidence" / "hw-test-output.txt"
    out.write_text(out.read_text() + line + "\n")
    rewrite(run, lambda r: r["boot"].update(ready_line=line))
    refused(run, "every corpus position matched")


@pytest.mark.parametrize("edit,word", [
    (lambda c: c.update(warm=False), "kernel_cache"),
    (lambda c: c.update(entries_before=-1), "kernel_cache"),
    (lambda c: c.update(entries_before="3"), "kernel_cache"),
])
def test_a_cache_figure_that_contradicts_itself_is_refused(run, edit, word):
    rewrite(run, lambda r: (r["boot"]["kernel_cache"].update(entries_before=5, warm=True), edit(r["boot"]["kernel_cache"])))
    refused(run, word)


def test_a_ready_time_before_the_load_time_is_refused(run):
    rewrite(run, lambda r: r["boot"].update(ready_s=r["boot"]["load_s"] / 2))
    refused(run, "readiness line cannot come before")


def test_a_result_that_differs_from_the_record_is_refused(run):
    edit_result(run, load_s=999.0)
    refused(run, "load_s")
    edit_result(run, load_s=record_of(run)["boot"]["load_s"], kernel_cache_warm=not record_of(run)["boot"]["kernel_cache"]["warm"])
    refused(run, "kernel_cache_warm")


def test_a_result_that_is_not_derived_by_the_supervisor_is_refused(run):
    edit_result(run, derived_by="agent")
    refused(run, "derived_by")


def test_a_result_with_a_check_that_lacks_evidence_is_refused(run):
    r = json.loads((run["stage"] / "result.json").read_text())
    r["checks"]["canary"] = {"pass": True}
    (run["stage"] / "result.json").write_text(json.dumps(r))
    refused(run, "checks.canary needs")


def test_a_corpus_check_that_says_pass_over_a_partial_match_is_refused(run):
    r = json.loads((run["stage"] / "result.json").read_text())
    r["checks"]["corpus"]["pass"] = False
    (run["stage"] / "result.json").write_text(json.dumps(r))
    refused(run, "checks.corpus.pass")


def test_a_failed_hardware_test_is_named_first(run):
    m = json.loads((run["stage"] / "test-result.json").read_text())
    m["returncode"] = 3
    (run["stage"] / "test-result.json").write_text(json.dumps(m))
    refused(run, "exited with code 3")


def test_a_command_other_than_the_fixed_one_is_refused(run):
    m = json.loads((run["stage"] / "test-result.json").read_text())
    m["command"] = "python3 evil.py"
    (run["stage"] / "test-result.json").write_text(json.dumps(m))
    refused(run, "supervisor's command for this stage")


def test_chips_that_are_not_the_chips_of_the_accepted_stage_are_refused(run):
    refused(run, "stage 2 showed the port on 4", expected_chips=4)
    refused(run, "expected_chips must be", expected_chips=0)


def test_chips_that_are_not_the_leased_chips_are_refused(run):
    rewrite(run, lambda r: r.update(chips=["0000:09:00.0", "0000:09:00.1"]))
    refused(run, "are not the chips the supervisor leased")


def test_a_gate_called_without_a_source_stage_refuses(run):
    reasons, _, _ = full_port_serve.check(run["stage"], run["run"], source=None)[0], 0, 0
    assert reasons and "needs the stage whose port it serves" in reasons[0]
    assert full_port_serve.check(run["stage"], run["run"], source=4)[0]


def test_a_port_file_that_differs_from_the_one_stage_2_accepted_is_refused(run):
    """The served port is edited after the stage 2 record and the record of this stage follows it, as the template's would."""
    p = run["stage"] / "port" / "port_model.py"
    p.write_text(p.read_text() + "\n# changed\n")
    rewrite(run, lambda r: r["port_sources"][0].update(sha256=sha(p.read_bytes())))
    refused(run, "differ from the files stage 2 accepted")


def test_a_port_file_added_after_the_record_is_refused(run):
    (run["stage"] / "port" / "extra.py").write_text("import ttnn\n")
    refused(run, "not in port_sources")


def test_a_template_copy_that_differs_from_the_shipped_one_is_refused(run):
    p = run["stage"] / "port_serve.py"
    p.write_text(p.read_text() + "\n# edited\n")
    refused(run, "port_serve.py in this stage differs from the shipped file")


def test_a_worker_copy_that_differs_from_the_shipped_one_is_refused(run):
    p = run["stage"] / "port_check.py"
    p.write_text(p.read_text() + "\n# edited\n")
    refused(run, "port_check.py in this stage differs from the shipped file")


def test_a_record_signed_by_another_template_is_refused(run):
    rewrite(run, lambda r: r.update(template_sha256="0" * 64))
    refused(run, "template_sha256")


def test_a_printed_hash_that_does_not_match_the_record_is_refused(run):
    out = run["stage"] / "evidence" / "hw-test-output.txt"
    out.write_text(re.sub(r"PORT_CHECK_SHA256 [0-9a-f]{64}", "PORT_CHECK_SHA256 " + "0" * 64, out.read_text()))
    m = json.loads((run["stage"] / "test-result.json").read_text())
    m["output"] = full_port.file_record(run["run"], out)
    (run["stage"] / "test-result.json").write_text(json.dumps(m))
    refused(run, "does not carry a PORT_CHECK_SHA256 line")


def test_a_second_mark_line_in_the_output_is_refused(run):
    out = run["stage"] / "evidence" / "hw-test-output.txt"
    out.write_text(out.read_text() + "PORT_CHECK_SHA256 " + "1" * 64 + "\n")
    m = json.loads((run["stage"] / "test-result.json").read_text())
    m["output"] = full_port.file_record(run["run"], out)
    (run["stage"] / "test-result.json").write_text(json.dumps(m))
    refused(run, "2 lines that begin with")


def test_a_missing_record_or_result_is_a_named_reason(run):
    (run["stage"] / "evidence" / "port-serve.json").unlink()
    refused(run, "port-serve.json is missing")
    (run["stage"] / "result.json").unlink()
    refused(run, "result.json is missing")


@pytest.mark.parametrize("section", ["boot", "replay", "passkey", "canary", "corpus", "config", "port_sources"])
@pytest.mark.parametrize("bad", [None, [], "x", 7, {}])
def test_a_malformed_section_is_a_reason_and_never_an_exception(run, section, bad):
    rewrite(run, lambda r: r.update({section: bad}))
    reasons, _, _ = fk.gate(run)
    assert reasons


@pytest.mark.parametrize("field", ["serves", "coherent", "checks", "evidence", "load_s", "ready_line", "corpus_positions", "passkey_answered"])
@pytest.mark.parametrize("bad", [None, [], "x", {}, -1])
def test_a_malformed_result_field_is_a_reason_and_never_an_exception(run, field, bad):
    edit_result(run, **{field: bad})
    reasons, _, _ = fk.gate(run)
    assert reasons


def test_derive_gives_a_reason_for_every_missing_piece(run):
    (run["stage"] / "result.json").unlink()
    (run["stage"] / "evidence" / "port-serve.json").unlink()
    assert full_port_serve.derive(run["stage"], run["run"])[1].startswith("evidence/port-serve.json is missing")
    (run["stage"] / "test-result.json").unlink()
    assert "test-result.json" in full_port_serve.derive(run["stage"], run["run"])[1]


def test_derive_leaves_a_record_that_lacks_a_section_to_the_gate(run):
    rewrite(run, lambda r: r.pop("canary"))
    assert full_port_serve.derive(run["stage"], run["run"])[0] is None


# ---- the server on its own ------------------------------------------------------------------------------------------------------------

@pytest.fixture
def server():
    """The template's handler over a fake worker, so the endpoints can be asked directly."""
    spec = importlib.util.spec_from_file_location("port_serve_under_test", full_port_serve.TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Worker:
        def ask(self, **req):
            return 100 if req["op"] == "prefill" else req["token"] + 1

    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({f"w{i}": i for i in range(200)}, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    engine = mod.Engine(Worker(), 64, tok, [105], lambda m: " ".join(x["content"] for x in m))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(engine))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield mod, srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def test_the_server_stops_at_an_end_token_unless_told_to_ignore_it(server):
    mod, port = server
    assert mod.post(port, "/v1/completions", {"prompt": [1, 2], "max_tokens": 20})["choices"][0]["token_ids"] == [100, 101, 102, 103, 104]
    got = mod.post(port, "/v1/completions", {"prompt": [1, 2], "max_tokens": 8, "ignore_eos": True})["choices"][0]
    assert got["token_ids"] == list(range(100, 108)) and got["finish_reason"] == "length"


def test_teacher_forcing_feeds_the_given_ids_and_returns_the_ports_choice_at_each_position(server):
    mod, port = server
    got = mod.post(port, "/v1/completions", {"prompt": [1], "max_tokens": 4, "ignore_eos": True, "forced_tokens": [50, 60, 70]})
    assert got["choices"][0]["token_ids"] == [100, 51, 61, 71] and got["usage"]["prompt_tokens"] == 1


@pytest.mark.parametrize("body", [
    {"prompt": [1], "max_tokens": 4, "forced_tokens": [1]},          # forced_tokens must be max_tokens - 1 ids
    {"prompt": [], "max_tokens": 4}, {"prompt": "", "max_tokens": 4}, {"prompt": [True], "max_tokens": 4}, {"prompt": 5, "max_tokens": 4},
    {"prompt": [1], "max_tokens": 0}, {"prompt": [1], "max_tokens": "4"}, {"prompt": [1], "max_tokens": 10 ** 9},
    {"prompt": [1] * 70, "max_tokens": 4},                            # longer than the context of 64
    {"prompt": [1], "max_tokens": 2, "forced_tokens": "a"}, [1, 2], "x"])
def test_a_bad_completion_request_is_a_400_with_a_message(server, body):
    mod, port = server
    with pytest.raises(mod.ServeFailed, match="answered 400"):
        mod.post(port, "/v1/completions", body)


@pytest.mark.parametrize("body", [{}, {"messages": []}, {"messages": [1]}, {"messages": [{"role": "user", "content": 5}]}, {"messages": "x"}])
def test_a_bad_chat_request_is_a_400(server, body):
    mod, port = server
    with pytest.raises(mod.ServeFailed, match="answered 400"):
        mod.post(port, "/v1/chat/completions", body)


def test_chat_returns_a_message_the_ids_and_the_usage(server):
    mod, port = server
    got = mod.post(port, "/v1/chat/completions", {"messages": [{"role": "user", "content": "w5 w6"}], "max_tokens": 3})
    assert got["choices"][0]["message"] == {"role": "assistant", "content": "w100 w101 w102"} and got["choices"][0]["token_ids"] == [100, 101, 102]
    assert got["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert mod.get(port, "/v1/models")["data"][0]["id"] == "orchard-port"
    with pytest.raises(mod.ServeFailed):
        mod.post(port, "/v1/other", {})


def test_the_chat_template_of_the_snapshot_is_used_when_jinja2_can_run_it_and_the_plain_form_when_not(tmp_path):
    spec = importlib.util.spec_from_file_location("port_serve_under_test2", full_port_serve.TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    msgs = [{"role": "user", "content": "hi"}]
    assert mod.chat_renderer(tmp_path)[1] == "plain" and mod.chat_renderer(tmp_path)[0](msgs) == "user: hi\nassistant:"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}<assistant>"}))
    render, name = mod.chat_renderer(tmp_path)
    assert name == "tokenizer_config" and render(msgs) == "<user>hi<assistant>"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{% for m in messages %}{{ m.nope.x.y }}{% endfor %}{{ raise_exception('no') }}"}))
    assert mod.chat_renderer(tmp_path)[1] == "plain"


def test_the_passkey_prompt_has_the_key_once_in_the_middle_and_the_requested_size(tmp_path):
    spec = importlib.util.spec_from_file_location("port_serve_under_test3", full_port_serve.TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fk.tokenizer_with(tmp_path / "t.json", ["48213"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tmp_path / "t.json"))
    for length in (128, 256, 1000):
        text = mod.passkey_prompt(tok, length, "48213")
        n = len(tok.encode(text, add_special_tokens=False).ids)
        assert text.count("48213") == 1 and 0.8 * length <= n <= length, (length, n)
        assert 0.3 < text.index("48213") / len(text) < 0.7 and text.endswith("The pass key is")


def test_the_template_and_the_gate_agree_on_the_mark_and_the_endpoint_names():
    text = full_port_serve.TEMPLATE.read_text()
    assert "PORT_CHECK_SHA256" in text and full_port.OUTPUT_MARK == "PORT_CHECK_SHA256" and full_port_serve.RECORD_SCHEMA in text
    assert full_port_serve.COMMAND == "python3 -P -s stages/5/port_serve.py" and "8100 <= port <= 8103" in text


def test_the_template_has_no_orchard_import_and_binds_to_loopback_only():
    text = full_port_serve.TEMPLATE.read_text()
    assert "import orchard" not in text and "from orchard" not in text
    assert '("127.0.0.1", 0)' in text and "0.0.0.0" not in text
