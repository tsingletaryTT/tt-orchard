# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Findings of the Copilot review of PR 398: stage 1 checks the token ids that stage 2's template reads, media is hashed through a pinned root
under a size cap, and a scoring model of class full-port has one pause after stage 2 and not two."""
import hashlib
import os

import pytest

from orchard import scoring_check as sc
from orchard import supervisor
from orchard.supervisor import EXIT_BLOCKED, Control
from scoring_check_fakes import HONEST, make_run
from scoring_embed_fakes import MEDIA_EMBED, embed_rows, make_embed_run, media_entry
from test_scoring_stage1_gate import PNG, edit_json, gate, reference_dir
from test_supervisor_scoring import FULL, Stop, make, never_wait, rig, started  # noqa: F401  (rig is a fixture)

BAD_IDS = [None, [], "abc", [1, "2"], [1, True], [-1], [1.5], [None], [10 ** 400], 7]


def retarget(run, ev):
    """The reference names the pair file by its hash, so it is rewritten to name the edited pair file."""
    edit_json(ev / "scoring-reference.json", lambda d: d["pairs"].update(sha256=hashlib.sha256((ev / "scoring-pairs.json").read_bytes()).hexdigest()))


@pytest.mark.parametrize("ids", BAD_IDS + [list(range(sc.MAX_IDS + 1))])
@pytest.mark.parametrize("kind", ["embedding", "reranker"])
def test_the_gate_refuses_a_pair_row_whose_token_ids_stage_2_would_refuse(tmp_path, kind, ids):
    run, stage, _ = make_embed_run(tmp_path) if kind == "embedding" else make_run(tmp_path, HONEST)
    ev = reference_dir(run)
    edit_json(ev / "scoring-pairs.json", lambda d: d["pairs"][3].update(input_ids=ids))
    retarget(run, ev)
    got = gate(run)
    word = "an integer has too many digits" if ids == [10 ** 400] else "token ids"        # the safe reader refuses a 401-digit integer first
    assert not got.ok and any(word in r for r in got.reasons), got.reasons


def test_the_gate_refuses_a_relevant_label_that_is_not_true_or_false(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST)
    ev = reference_dir(run)
    edit_json(ev / "scoring-pairs.json", lambda d: d["pairs"][0].update(relevant="yes"))
    retarget(run, ev)
    got = gate(run)
    assert not got.ok and any("relevant is not true or false" in r for r in got.reasons)


def media_run(tmp_path, **files):
    return make_embed_run(tmp_path, MEDIA_EMBED, rows=embed_rows({0: [media_entry("a.png", PNG)]}), media_files={"a.png": PNG, **files})


def test_a_media_file_replaced_by_a_link_to_identical_bytes_is_refused(tmp_path):
    run, stage, _ = media_run(tmp_path, **{"real.png": PNG})
    assert gate(run).ok
    media = reference_dir(run) / "media"
    (media / "a.png").unlink()
    os.symlink(media / "real.png", media / "a.png")                  # the same bytes, so the same sha256, but not a regular file
    got = gate(run)
    assert not got.ok and any("can be read within" in r for r in got.reasons), got.reasons


def test_a_media_file_over_the_size_cap_is_refused_whatever_digest_the_pair_file_gives(tmp_path):
    run, stage, _ = media_run(tmp_path)
    with open(reference_dir(run) / "media" / "a.png", "wb") as f:
        f.truncate(sc.MEDIA_MAX + 1)                                 # a sparse file just over the cap: refused before any read
    got = gate(run)
    assert not got.ok and any("can be read within" in r and "larger than" in r for r in got.reasons), got.reasons


def test_a_media_directory_that_is_a_link_inside_the_run_directory_is_refused(tmp_path):
    run, stage, _ = media_run(tmp_path)
    ev = reference_dir(run)
    (ev / "media").rename(ev / "real-media")
    os.symlink(ev / "real-media", ev / "media")
    got = gate(run)
    assert not got.ok and any("can be read within" in r for r in got.reasons), got.reasons


@pytest.mark.parametrize("path", ["stages/1/evidence/reference/media/", "stages/1/evidence/reference/media/sub/a.png"])
def test_a_media_path_that_is_not_a_file_directly_under_the_media_directory_is_refused(tmp_path, path):
    run, stage, _ = media_run(tmp_path)
    edit_json(reference_dir(run) / "scoring-pairs.json", lambda d: d["pairs"][0]["media"][0].update(path=path))
    got = gate(run)
    assert not got.ok and any("not a file directly under" in r for r in got.reasons), got.reasons


# ---- the pause of a full-port scoring run -------------------------------------------------------------------------------------

def test_an_unattended_full_port_scoring_run_ends_blocked_with_scoring_route_and_not_full_port_stages_unbuilt(rig):
    rig.args.full_port = True
    rig.args.unattended = True
    make(rig, delta=FULL)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["scoring-route"]
    assert [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"] == [supervisor.SCORING_STAGES_UNBUILT]
    assert not started(rig, 3)


def test_an_attended_full_port_scoring_run_pauses_once_before_stage_3_and_one_resume_is_enough(rig):
    rig.args.full_port = True
    make(rig, delta=FULL)
    waits = []

    def on_sleep():
        waits.append(1)
        if len(waits) > 1:
            raise Stop("the run paused a second time")
        Control(rig.run_dir).write("resume")
    rig.on_sleep = on_sleep
    try:
        rig.run()
    except Stop:
        pass
    pauses = [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]
    assert supervisor.FULL_PORT_STAGES_UNBUILT not in pauses and pauses.count(supervisor.SCORING_STAGES_UNBUILT) == 1
    assert started(rig, 3)


# ---- review round 2 ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reason", [supervisor.SCORING_STAGES_UNBUILT, supervisor.SCORING_NOT_ROUTED])
def test_the_status_hint_for_a_scoring_pause_says_a_person_decides(reason):
    from orchard import status
    facts = {"state": "paused", "pause": {"kind": status.classify_pause({"reason": reason})[0]}, "control_pending": False,
             "disk": {"run_dir_free_gb": 500}, "pauses_at_stage": 1, "quiet_s": 0}
    hint = next(h for name, test, h in status.HINT_RULES if test(facts))
    assert facts["pause"]["kind"] == "scoring" and "needs a decision" in hint


def test_a_scoring_model_that_a_text_runtime_serves_pauses_before_stage_1_instead_of_comparing_tokens(rig, monkeypatch):
    rig.args.unattended = True
    make(rig)
    # The text runtime is selected after stage 0 (its selection reads the ledger); spec_for gives its stages precedence over the scoring ones.
    monkeypatch.setattr(supervisor.text_runtime, "selected", lambda entries: object() if any(e["event"] == "stage_end" and e["stage"] == 0 for e in entries) else None)
    never_wait(rig)
    rig.run()
    blocks = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert blocks and blocks[0]["code"] == "scoring-route" and "no scoring route" in blocks[0]["reason"]
    assert not started(rig, 1)


def test_a_run_with_no_model_input_cannot_take_the_scoring_route_and_says_so(tmp_path):
    from run_fakes import FILES, argv
    from test_supervisor import Rig
    from test_supervisor_scoring import snapshot
    snap = snapshot(tmp_path)
    r = Rig(tmp_path)
    try:
        r.args = supervisor.parse(argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=2))        # no --input model=
        r.args.unattended = True
        # stage 0 names its snapshot in its triage config, the way the template's input does; the run itself was given none
        make(r)
        inner = r.script

        def script(request):
            from run_fakes import where
            if where(request) == (0, "run"):
                (r.run_dir / "stages" / "0" / "triage_config.json").write_text('{"model_snapshot": "%s"}' % snap)
            return inner(request)
        r.script = script
        never_wait(r)
        r.run()
        assert r.ends(0)[0]["model_kind"]["kind"] == "reranker"             # the kind came from the snapshot stage 0 named
        blocks = [d for d in r.decisions() if d["decision"] == "blocked"]
        assert blocks and blocks[0]["code"] == "scoring-route" and "model= input" in blocks[0]["reason"]
        assert not started(r, 1)
    finally:
        r.close()


# ---- review round 3 ----------------------------------------------------------------------------------------------------------

def test_a_vector_of_tiny_numbers_does_not_pass_by_underflow_and_the_template_and_gate_agree():
    import importlib.util
    spec = importlib.util.spec_from_file_location("scoring_check_template_round3", sc.TEMPLATE)
    tpl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpl)
    ref = [1 / 64] * 4096
    chip = [1e-162] * 4000 + [3e-162] * 96            # the squares underflow to 0.0 in the plain formula, which then returned 1.0
    expected = (4000 * 1 + 96 * 3) / (4096 ** 0.5 * (4000 + 96 * 9) ** 0.5)
    assert 0.9606 < expected < 0.9607
    for cosine in (sc.cosine, tpl.cosine):
        assert cosine(ref, chip) == pytest.approx(expected, abs=1e-9)
        assert cosine(chip, ref) == pytest.approx(expected, abs=1e-9)
    assert sc.embed_numbers([ref], [chip], ["text"]) == tpl.embed_numbers([ref], [chip], ["text"])
    assert sc.embed_bound_reasons({"count": 24, "cosine_mean": expected, "cosine_min": expected, "by_kind": {"a": {"mean": expected, "min": expected}} | {
        "b": {"mean": 1, "min": 1}, "c": {"mean": 1, "min": 1}}}, sc.EMBEDDING_BOUNDS)
    for cosine in (sc.cosine, tpl.cosine):
        assert cosine([1e-300, 2e-300], [2e-300, 4e-300]) == pytest.approx(1.0)       # the same direction at any scale
        assert cosine([1e-320, 0.0], [0.0, 1e-320]) == 0.0 and cosine([0.0, 0.0], [1e-320, 1e-320]) == 0.0
        assert cosine([1e6, -1e6], [-1e6, 1e6]) == pytest.approx(-1.0)


@pytest.mark.parametrize("readme", [
    "---\nbooks:\n  pipeline_tag: text-ranking\npipeline_tag: text-generation\n---\n",
    "---\nnote: |\n  pipeline_tag: feature-extraction\npipeline_tag: text-generation\n---\n",
])
def test_an_indented_pipeline_tag_inside_a_nested_mapping_does_not_set_the_kind(tmp_path, readme):
    from orchard import model_kind
    (tmp_path / "README.md").write_text(readme)
    assert model_kind.pipeline_tag(tmp_path) == "text-generation"
    (tmp_path / "README.md").write_text(readme.replace("pipeline_tag: text-generation\n", ""))
    assert model_kind.pipeline_tag(tmp_path) is None


def test_a_resume_without_the_model_input_after_stage_2_still_pauses_before_stage_3(rig):
    from run_fakes import argv
    rig.args.unattended = True
    make(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["scoring-route"] and not started(rig, 3)
    rig.args = supervisor.parse(argv(rig.run_dir.parent, rig.tiers, rig.chip_server.endpoint, chips=2))      # the retry names no model= input
    rig.args.unattended = True
    assert rig.run() == EXIT_BLOCKED
    assert [d["code"] for d in rig.decisions() if d["decision"] == "blocked"] == ["scoring-route"] * 2
    assert not started(rig, 3)


# A port that replaces the media directory with a link to a copy while it embeds the first input: the next request reads the file by path.
SWAP_DIR = '''
import ttnn, random, os, shutil
def load(model_dir, mesh_device):
    return {}
def embed(model, ids, media=None):
    n = ids[0] - 1
    if n == 0:
        base = "stages/1/evidence/reference"
        shutil.copytree(base + "/media", base + "/media-copy")
        shutil.rmtree(base + "/media")
        os.symlink("media-copy", base + "/media")
    r = random.Random(n)
    return [r.uniform(-1, 1) for _ in range(8)]
def close(model):
    pass
'''


def test_a_media_directory_swapped_for_a_link_while_the_port_runs_is_refused_even_with_identical_bytes(tmp_path):
    from scoring_check_fakes import run_template
    rows = embed_rows({0: [media_entry("a.png", PNG)], 9: [media_entry("a.png", PNG)]})
    run, stage, _ = make_embed_run(tmp_path, SWAP_DIR, rows=rows, media_files={"a.png": PNG})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "failed at 'embed'" in done.stdout and "a.png" not in done.stderr, done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


def test_a_link_in_any_component_of_the_media_path_is_refused_by_the_template_reader(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("scoring_check_template_review", sc.TEMPLATE)
    tpl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tpl)
    run, stage, _ = media_run(tmp_path)
    monkeypatch.setattr(tpl, "RUN_DIR", run)
    path = sc.MEDIA_DIR + "a.png"
    assert tpl.read_media(path) == PNG
    ev = run / "stages" / "1" / "evidence"
    (ev).rename(run / "stages" / "1" / "evidence-real")
    os.symlink(run / "stages" / "1" / "evidence-real", ev)             # a link in the middle of the path, to a directory inside the run
    with pytest.raises(OSError):
        tpl.read_media(path)
