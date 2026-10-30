# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 8 bundle of a stock-runtime run: the "Stock runtime on a chip" section and the stock risks (issue 93).

The builder (orchard/skills/operator-bundle-templates/build_bundle.py) reads `stages/2/evidence/stock-check.json` only when it exists, the way it
reads audio_result.json. These tests build the fake run of test_operator_bundle_template.py, add a stock-check.json written the way the
stock_check.py template writes it, and run the builder as a subprocess. A run without the file must give the same RESULTS.md and RISKS.md as before.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_template import build, bundle_of, make_run, section, write_json  # noqa: E402

HEADING = "## Stock runtime on a chip"
RESULT = Path("stages") / "2" / "evidence" / "stock-check.json"
SWEEP = [", not ", "rather than", "instead of", "reads as", "honest", "the one", "which is why", "This is what"]


def stock_check(**over):
    res = {"template_sha256": "a" * 64, "model_id": "meta-llama/Llama-3.1-8B-Instruct@abc123", "chips": 1, "device_name": "P150",
           "optimizations": "accuracy", "tt_metal": {"tag": "v0.80.0", "dirty": False},
           "teacher_forced": {"tokens": 511, "top1": 0.9139, "top5": 0.9902},
           "free_run": {"prompt": "Say hello.", "text": "Hello there, friend.", "new_tokens": 128, "finish": "length"},
           "speed": {"decode_tokens_per_s": 28.4, "ttft_s": 0.21}, "hub_fetch": [],
           "reference": {"source": "generated", "model_id": "meta-llama/Llama-3.1-8B-Instruct", "snapshot": "abc123", "sha256": "b" * 64},
           "scratch_patch": "evidence/scratch.patch", "exit_codes": {"token_accuracy": 0, "free_run": 0}, "seconds": 812.5}
    res.update(over)
    return res


def stock_run(tmp_path, result=None, patch="--- a/models/x\n+++ b/models/x\n@@\n-a\n+b\n"):
    run = make_run(tmp_path)
    write_json(run / RESULT, stock_check() if result is None else result)
    (run / "stages" / "2" / "evidence" / "scratch.patch").write_text(patch)
    return run


def results(run):
    r = build(run)
    assert r.returncode == 0, r.stderr + r.stdout
    return (bundle_of(run) / "RESULTS.md").read_text(), (bundle_of(run) / "RISKS.md").read_text()


def test_a_run_without_stock_check_has_no_stock_section(tmp_path):
    run = make_run(tmp_path)
    text, risks = results(run)
    assert HEADING not in text and "## Stock runtime run" not in risks


def test_the_section_prints_the_token_accuracy_reference_tree_and_patch(tmp_path):
    text, _ = results(stock_run(tmp_path))
    sec = section(text, HEADING)
    for want in ("0.9139", "0.9902", "511", "P150", "accuracy", "v0.80.0", "28.4", "abc123", "b" * 64, "evidence/scratch.patch"):
        assert want in sec, want
    assert "Evidence: `stages/2/evidence/stock-check.json`" in sec


def test_the_patch_is_summarised_as_a_line_count(tmp_path):
    text, _ = results(stock_run(tmp_path, patch="--- a\n+++ b\n@@\n-a\n+b\n+c\n"))
    assert "2 lines added, 1 removed" in section(text, HEADING)
    text, _ = results(stock_run(tmp_path / "two", patch=""))
    assert "no change to the copied host tree paths" in section(text, HEADING)


def test_a_dirty_tree_and_a_patch_are_risks(tmp_path):
    _, risks = results(stock_run(tmp_path, stock_check(tt_metal={"tag": "v0.80.0", "dirty": True})))
    block = section(risks, "## Stock runtime run")
    assert "uncommitted changes" in block and "scratch patch" in block


def test_malformed_stock_evidence_prints_not_recorded_and_still_builds(tmp_path):
    text, _ = results(stock_run(tmp_path, {"teacher_forced": "x", "tt_metal": [], "speed": None, "reference": 3}))
    assert "not recorded" in section(text, HEADING)


def test_the_section_text_passes_the_plain_english_sweep(tmp_path):
    text, risks = results(stock_run(tmp_path))
    mine = section(text, HEADING) + section(risks, "## Stock runtime run")
    assert [w for w in SWEEP if w in mine] == []


def test_an_evidence_path_that_leaves_the_stage_directory_is_not_read(tmp_path):
    """Copilot, PR 275: the containment check accepted every string, so a path such as /dev/zero or a link out of the stage could hang the build."""
    secret = tmp_path / "secret.txt"
    secret.write_text("+++ not a patch\n+x\n+y\n")
    for i, name in enumerate(("/dev/zero", "../../../secret.txt", "evidence/../../../secret.txt", str(secret), "evidence/link.patch")):
        run = stock_run(tmp_path / f"case{i}", stock_check(scratch_patch=name))
        if name == "evidence/link.patch":
            (run / "stages" / "2" / "evidence" / "link.patch").symlink_to(secret)
        text, _ = results(run)
        assert "lines added" not in section(text, HEADING), name
        assert "Scratch patch | " in text and "not recorded" in section(text, HEADING)


def test_a_patch_file_over_the_size_limit_is_not_read(tmp_path):
    big = "+x\n" * 800_000
    text, _ = results(stock_run(tmp_path, patch=big))
    assert "lines added" not in section(text, HEADING)
