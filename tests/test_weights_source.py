# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 73: a package that hard-codes its model repository serves the base model whatever weights the bring-up swaps in.

PR 72 made a test that fetched from the Hub fail. These tests pin the early answer: a record of what each installed package does with
MODEL_WEIGHTS_DIR (orchard/data/weights_source.json), a dry run of the swap's own run.sh edit on every installed bundle, and a `base`
preflight row that blocks, before any coder boot, when every package that serves the base is refused. Nothing here needs a chip."""
import json
from pathlib import Path

import pytest

from orchard import nearest, safeio
from orchard import preflight as pf
from orchard import weights_source as ws
from orchard.nearest import Installed

BASE = "Qwen/Base-7B"
HARD = "org/hard-coded-pkg"
OPEN = "org/probed-open-pkg"
NEW = "org/not-in-the-record-pkg"

RECORDS = {
    HARD: ws.Record(HARD, ws.HARD_CODED, "the server log shows 'Fetching 28 files' while 32 of 32 tokens matched", 73),
    OPEN: ws.Record(OPEN, ws.HONOURS, "an isolated, empty /hf stayed empty over a full test", 73),
}

OLDER = f'''#!/usr/bin/env bash
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PYBIN="$HERE/venv/bin/python"
export HF_MODEL="${{HF_MODEL:-{BASE}}}"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{BASE}" --max_num_seqs 4 "$@")
exec "${{CMD[@]}}"
'''

CURRENT = '''#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="$HERE/venv/bin/python"
export HF_MODEL="${HF_MODEL:-Qwen/Base-7B}"
"$PYBIN" "$HERE/prepare_model_dir.py"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "$HERE/model-dir" --max_num_seqs 4 "$@")
exec "${CMD[@]}"
'''


def container(name, repo=BASE, chips=2):
    return Installed(name, repo, "5.1", chips)


def bundle(tmp_path, name, text, repo=BASE):
    d = tmp_path / name.replace("/", "__")
    d.mkdir()
    if text is not None:
        (d / "run.sh").write_text(text)
    return Installed(name, repo, "6", 2, str(d))


# ---- the shipped record ---------------------------------------------------------------------------------------------------------

def test_the_shipped_record_loads_and_says_the_coder_30b_package_hard_codes_its_repository():
    recs = ws.load_records()
    r = recs["raahemnabeel/qwen3-coder-30b-a3b"]
    assert r.status == ws.HARD_CODED and "Fetching 28 files" in r.evidence and r.issue == 73


def test_the_shipped_record_says_the_coder_next_package_reads_the_model_directory_and_how_that_was_seen():
    r = ws.load_records()["raahemnabeel/qwen3-coder-next"]
    assert r.status == ws.HONOURS and "FileNotFoundError" in r.evidence and "stayed empty" in r.evidence and "2026-11-03" in r.evidence


def test_every_shipped_row_has_a_known_status_and_the_evidence_its_status_needs():
    for name, r in ws.load_records().items():
        assert r.status in (ws.HONOURS, ws.HARD_CODED, ws.UNPROBED), name
        assert r.evidence.strip(), f"{name} has no evidence or note"
    assert any(r.status == ws.UNPROBED for r in ws.load_records().values()), "the unprobed packages are listed, so the gap is visible"


def test_no_shipped_row_names_a_machine_path_or_a_host():
    text = (Path(ws.__file__).parent / "data" / "weights_source.json").read_text()
    assert "/home/" not in text and "/mnt/" not in text


def write_records(tmp_path, body):
    p = tmp_path / "weights_source.json"
    p.write_text(body if isinstance(body, str) else json.dumps(body))
    return p


@pytest.mark.parametrize("body, why", [
    ("[]", "not an object"),
    ('{"schema": 2, "packages": {}}', "schema"),
    ('{"schema": 1, "packages": []}', "packages"),
    ('{"schema": 1, "packages": {"a/b": {"status": "maybe", "evidence": "x"}}}', "status"),
    ('{"schema": 1, "packages": {"a/b": {"status": "hard-coded", "evidence": ""}}}', "evidence"),
    ('{"schema": 1, "packages": {"a/b": {"status": "honours"}}}', "evidence"),
    ('{"schema": 1, "packages": {"a/b": {"status": "unprobed", "evidence": "x", "extra": 1}}}', "unknown key"),
    ('{"schema": 1, "packages": {"ab": {"status": "unprobed", "evidence": "x"}}}', "org/name"),
    ('{"schema": 1, "packages": {"a/b": {"status": "unprobed", "evidence": "x", "issue": "73"}}}', "issue"),
    ("not json", "JSON"),
])
def test_a_malformed_record_is_refused_with_the_reason(tmp_path, body, why):
    with pytest.raises(ValueError, match=why):
        ws.load_records(write_records(tmp_path, body))


def test_a_missing_or_linked_record_file_is_refused(tmp_path):
    with pytest.raises(ValueError):
        ws.load_records(tmp_path / "absent.json")
    real = write_records(tmp_path, {"schema": 1, "packages": {}})
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError):
        ws.load_records(link)


def test_a_package_that_is_not_in_the_record_is_unprobed():
    r = ws.record_for("someone/new-package", RECORDS)
    assert r.status == ws.UNPROBED and r.name == "someone/new-package"


# ---- the dry run of the swap's run.sh edit ----------------------------------------------------------------------------------------

def test_both_known_run_sh_forms_pass_the_dry_run():
    assert ws.run_sh_problem(OLDER, BASE) is None
    assert ws.run_sh_problem(CURRENT, BASE) is None


def test_a_run_sh_with_no_model_argument_is_refused_with_the_swap_scripts_own_words():
    text = OLDER.replace(f'--model "{BASE}" ', "")
    why = ws.run_sh_problem(text, BASE)
    assert why and "expected exactly one --model argument" in why


def test_a_second_model_option_is_refused_because_it_could_override_the_model_directory():
    text = OLDER.replace("--max_num_seqs 4", "--max_num_seqs 4 --model other/model")
    why = ws.run_sh_problem(text, BASE)
    assert why and "--model" in why


def test_a_current_form_run_sh_without_the_prepare_line_is_refused():
    why = ws.run_sh_problem(CURRENT.replace('"$PYBIN" "$HERE/prepare_model_dir.py"\n', ""), BASE)
    assert why and "prepare_model_dir.py" in why


def test_the_dry_run_names_the_wrong_base_when_the_run_sh_serves_a_different_model():
    why = ws.run_sh_problem(OLDER, "Qwen/Another-Base")
    assert why and "expected exactly one --model argument" in why


def test_the_dry_run_touches_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws.run_sh_problem(OLDER, BASE)
    ws.run_sh_problem(OLDER.replace(f'--model "{BASE}" ', ""), BASE)
    assert list(tmp_path.iterdir()) == []


# ---- judging the installed packages -----------------------------------------------------------------------------------------------

def test_a_hard_coded_container_is_refused_and_an_unprobed_one_is_kept(tmp_path):
    usable, refused = ws.judge([container(HARD), container(NEW)], BASE, RECORDS)
    assert [i.name for i in usable] == [NEW]
    assert [(i.name, why) for i, why in refused] == [(HARD, refused[0][1])]
    assert "hard-codes its model repository" in refused[0][1] and "Fetching 28 files" in refused[0][1]


def test_an_honouring_container_is_kept():
    usable, refused = ws.judge([container(OPEN)], BASE, RECORDS)
    assert [i.name for i in usable] == [OPEN] and refused == []


def test_a_bundle_whose_run_sh_cannot_take_swapped_weights_is_refused(tmp_path):
    bad = bundle(tmp_path, "org/bad-bundle", OLDER.replace(f'--model "{BASE}" ', ""))
    good = bundle(tmp_path, "org/good-bundle", CURRENT)
    usable, refused = ws.judge([bad, good], BASE, RECORDS)
    assert [i.name for i in usable] == ["org/good-bundle"]
    assert refused[0][0].name == "org/bad-bundle" and "cannot take swapped weights" in refused[0][1]


def test_a_bundle_with_no_directory_or_no_run_sh_is_not_refused_for_what_cannot_be_read(tmp_path):
    no_dir = Installed("org/a", BASE, "6", 2)
    no_file = bundle(tmp_path, "org/b", None)
    usable, refused = ws.judge([no_dir, no_file], BASE, RECORDS)
    assert [i.name for i in usable] == ["org/a", "org/b"] and refused == []


def test_the_usable_packages_keep_their_order_and_the_refused_ones_leave(tmp_path):
    order = [container(HARD), container(NEW), bundle(tmp_path, "org/z", CURRENT), container(OPEN)]
    usable, _ = ws.judge(order, BASE, RECORDS)
    assert [i.name for i in usable] == [NEW, "org/z", OPEN]


def test_an_oversized_run_sh_is_not_read(tmp_path):
    big = bundle(tmp_path, "org/big", "x" * (ws.MAX_RUN_SH_BYTES + 1))
    usable, refused = ws.judge([big], BASE, RECORDS)
    assert refused == [] and [i.name for i in usable] == ["org/big"]


# ---- the base row -----------------------------------------------------------------------------------------------------------------

def check(inst, *, records=RECORDS, base=BASE, search=None):
    hub = pf.parse_hub({"id": "someone/fine-tune", "private": False, "gated": False, "sha": "c07c06c8",
                        "cardData": {"license": "apache-2.0", "base_model": base},
                        "config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
                        "siblings": [{"rfilename": "config.json", "size": 1000}, {"rfilename": "model.safetensors", "size": 14_000_000_000}]})
    return pf.check_base(model_id="someone/fine-tune", hub=hub, base_override=None, local_snapshot=lambda m: Path("/s"),
                         installed=list(inst), search=search, weights_records=records)


def test_a_base_served_only_by_a_hard_coded_package_blocks_before_any_coder_boot():
    c = check([container(HARD)])
    assert c.status == "block" and c.reason == "nearest-model-missing"
    assert HARD in c.detail and "hard-codes its model repository" in c.detail
    assert "patched generator" in c.detail and "a new package, or a bundle" in c.detail
    assert [r["name"] for r in c.data["refused"]] == [HARD] and "hard-codes" in c.data["refused"][0]["reason"]


def test_a_base_served_by_a_hard_coded_package_and_a_probed_open_one_passes_through_the_open_one():
    c = check([container(HARD), container(OPEN)])
    assert c.status != "block" and c.data["bundle"] == OPEN


def test_a_base_served_by_a_hard_coded_package_and_a_good_bundle_picks_the_bundle(tmp_path):
    c = check([container(HARD), bundle(tmp_path, "org/good", CURRENT)])
    assert c.status == "ok" and c.data["bundle"] == "org/good" and "org/good" in c.detail


def test_a_base_served_only_by_a_bundle_that_cannot_take_swapped_weights_blocks(tmp_path):
    c = check([bundle(tmp_path, "org/bad", OLDER.replace(f'--model "{BASE}" ', ""))])
    assert c.status == "block" and c.reason == "nearest-model-missing"
    assert "org/bad" in c.detail and "expected exactly one --model argument" in c.detail


def test_an_unprobed_container_still_passes_with_a_warning_that_says_it_was_not_probed():
    c = check([container(NEW)])
    assert c.status == "warn" and "not been probed" in c.detail


def test_a_probed_open_container_says_so_in_its_warning():
    c = check([container(OPEN)])
    assert c.status == "warn" and "honours MODEL_WEIGHTS_DIR" in c.detail


def test_a_base_served_by_no_one_keeps_the_old_message():
    c = check([])
    assert c.status == "block" and c.reason == "nearest-model-missing" and "no installed bundle serves" in c.detail
    assert c.data["refused"] == []


def test_the_refusal_text_is_one_sentence_per_package_and_ends_with_the_way_out():
    c = check([container(HARD)])
    assert c.detail.count("hard-codes its model repository") == 1
    assert "Pick one with --base" in c.detail


def test_the_wiring_from_the_installed_list_to_the_blocked_result(tmp_path):
    """The layer that fails: run_preflight reads the installed packages from its signals, so a refused package blocks the run there."""
    from test_preflight import cfg, signals
    hub_payload = {"id": "someone/fine-tune", "private": False, "gated": False, "sha": "c07c06c8",
                   "cardData": {"license": "apache-2.0", "base_model": BASE},
                   "config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
                   "siblings": [{"rfilename": "config.json", "size": 1000}, {"rfilename": "model.safetensors", "size": 14_000_000_000}]}
    sig = signals(hub_info=lambda m: (pf.parse_hub(hub_payload), None), installed_bundles=lambda: [container("raahemnabeel/qwen3-coder-30b-a3b")])
    out = pf.run_preflight(cfg(tmp_path), "someone/fine-tune", accept_credentials=False, signals=sig)
    base = [c for c in out if c.name == "base"][0]
    assert base.status == "block" and "raahemnabeel/qwen3-coder-30b-a3b" in base.detail and pf.blocked(out)


# ---- the installed list carries the directory --------------------------------------------------------------------------------------

def test_read_installed_records_each_bundles_directory(tmp_path):
    d = tmp_path / "org" / "pkg"
    d.mkdir(parents=True)
    (d / "tt_kernel_manifest.json").write_text(json.dumps({"schema_version": "6", "device_count": 2, "weights": {"repo_id": BASE}}))
    [got] = nearest.read_installed(tmp_path)
    assert got.directory == str(d) and got.name == "org/pkg"


def test_a_container_package_has_no_directory(tmp_path):
    assert container(HARD).directory == ""


# ---- a refused package is no choice for --base ----------------------------------------------------------------------------------------

def test_a_refused_package_is_not_listed_among_the_candidates_for_base():
    c = check([container(HARD), container("org/other-open-pkg", repo="Qwen/Base-7B-Instruct")], base=BASE)
    assert c.status == "block" and HARD not in [x["name"] for x in c.data["candidates"]]


def test_a_refused_package_is_dropped_from_the_researched_bundle_choices():
    hub = pf.parse_hub({"id": "someone/fine-tune", "private": False, "gated": False, "sha": "c07c06c8",
                        "cardData": {"license": "apache-2.0", "base_model": BASE},
                        "config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
                        "siblings": [{"rfilename": "config.json", "size": 1000}, {"rfilename": "model.safetensors", "size": 14_000_000_000}]})
    report = {"bundle_choices": [{"name": HARD, "class": "weights-only", "confidence": "stated"},
                                 {"name": "org/some-bundle", "class": "weights-only", "confidence": "stated"}]}
    c = pf.check_base(model_id="someone/fine-tune", hub=hub, base_override=None, local_snapshot=lambda m: Path("/s"),
                      installed=[container(HARD)], search=None, report=report, weights_records=RECORDS)
    assert [x["name"] for x in c.data["candidates"]] == ["org/some-bundle"] and f"{HARD} (" not in c.detail.split("ranked bundles:")[-1]


def test_a_bundle_with_no_directory_never_reads_a_run_sh_from_the_working_directory(tmp_path, monkeypatch):
    """An empty directory would make Path('') / 'run.sh' mean the process's own working directory, which may hold some other bundle."""
    (tmp_path / "run.sh").write_text(OLDER.replace(f'--model "{BASE}" ', ""))      # a run.sh that would be refused, if it were read
    monkeypatch.chdir(tmp_path)
    usable, refused = ws.judge([Installed("org/no-dir", BASE, "6", 2)], BASE, RECORDS)
    assert [i.name for i in usable] == ["org/no-dir"] and refused == []


def test_a_refused_package_is_dropped_from_the_search_results_too():
    c = check([container(HARD)], search=lambda family: [{"name": HARD, "installed": True}, {"name": "org/searched", "installed": False}])
    assert [x["name"] for x in c.data["candidates"]] == ["org/searched"] and f"found: {HARD}" not in c.detail
