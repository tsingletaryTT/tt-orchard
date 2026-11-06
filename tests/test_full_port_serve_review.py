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
from test_full_port_serve import good, record_of, rewrite, run, sha  # noqa: F401  (good and run are fixtures)


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
    s._run_test = lambda spec, stage_dir, test, lease, **kw: calls.ran.append((lease, kw))
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
