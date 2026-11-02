# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor-owned scoring check (orchard/skills/scoring-check-templates/scoring_check.py and the gate orchard/scoring_check.py, issue
358): it runs a fake port module on the CPU with a fake chip, computes the parity numbers itself, and the gate refuses a port that misses a
bound and each way a port module (or anything else the coder wrote) tries to forge the record."""
import importlib.util
import json
import os
import random
import re
import signal
import subprocess
import sys
import time

import pytest

from orchard import scoring_check as sc
from port_check_fakes import CHIPS, python_path
from scoring_check_fakes import (CONSTANT, HONEST, NOISY, REFERENCE_SCORES, REVERSED, make_run, pair_rows, run_template, write_marker)

MARK = re.compile(r"^SCORING_CHECK_SHA256 ([0-9a-f]{64})$", re.M)
spec = importlib.util.spec_from_file_location("scoring_check_template", sc.TEMPLATE)
tpl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpl)


def gate(run, stage):
    return sc.check(stage, run)[0]


def record(stage):
    return json.loads((stage / "evidence" / "scoring-check.json").read_text())


def mentions(reasons, word):
    assert reasons, "the gate passed a result it must refuse"
    return [r for r in reasons if word in r]


def honest_run(tmp_path, **kw):
    run, stage, _ = make_run(tmp_path, **kw)
    done = run_template(tmp_path, run, stage)
    return run, stage, done


# ---- the honest run -----------------------------------------------------------------------------------

def test_an_honest_port_gets_a_record_the_template_wrote_and_the_gate_passes_it(tmp_path):
    run, stage, done = honest_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    assert rec["chip_scores"] == REFERENCE_SCORES == rec["reference_scores"] and rec["kind"] == "reranker"
    m = rec["metrics"]
    assert m["pearson"] == pytest.approx(1.0) and m["mean_abs_diff"] == 0 and m["top1_groups_matching"] == m["top1_groups"] == 8
    assert m["order_pairs"] > 10 and m["order_pairs_kept"] == m["order_pairs"] and m["order_kept"] == 1.0
    assert rec["chips"] == CHIPS and rec["load_s"] > 0 and rec["score_s"] > 0 and rec["template_sha256"] == sc.template_sha256()
    assert rec["groups"] == [f"g{g}" for g in range(8) for _ in range(4)] and rec["pair_kinds"][0] == "short" and rec["pair_kinds"][-1] == "long"
    assert [s["path"] for s in rec["port_sources"]] == ["stages/2/port/port_model.py"]
    assert MARK.fullmatch(done.stdout.strip().splitlines()[-1])
    assert gate(run, stage) == []


def test_the_port_is_given_only_token_ids_and_the_chips_the_supervisor_leased(tmp_path):
    spy = HONEST.replace("def score_pairs(model, pairs):", "def score_pairs(model, pairs):\n    assert all(type(i) is int for ids in pairs for i in ids)\n    print('device', model['device'], 'n', len(pairs))")
    run, stage, done = honest_run(tmp_path, port_src=spy)
    assert done.returncode == 0 and "port| device {'mesh': (1, 2)} n 4" in done.stdout


def test_a_record_left_by_an_earlier_run_is_deleted_before_the_port_runs(tmp_path):
    run, stage, _ = make_run(tmp_path, port_src=HONEST.replace("return [(ids", "raise RuntimeError('boom')\n    return [(ids"))
    (stage / "evidence").mkdir()
    (stage / "evidence" / "scoring-check.json").write_text("{}")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and not (stage / "evidence" / "scoring-check.json").exists()


def test_a_port_file_edited_after_the_record_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path)
    f = stage / "port" / "port_model.py"
    f.write_text(f.read_text() + "\n# edited later\n")
    assert mentions(gate(run, stage), "changed after the record was written")


def test_a_port_that_raises_fails_the_run_and_leaves_no_record(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=HONEST.replace("return [(ids", "raise RuntimeError('no kernel for this layer')\n    return [(ids"))
    assert done.returncode == 2 and "no kernel for this layer" in done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


@pytest.mark.parametrize("bad, word", [
    ("[1.0]", "scores for the 4 prompts"),
    ("[float('nan')] * len(pairs)", "not a finite number"),
    ("[float('inf')] * len(pairs)", "not a finite number"),
    ("['x'] * len(pairs)", "not a finite number"),
    ("[None] * len(pairs)", "not a finite number"),
    ("[1e300] * len(pairs)", "not a finite number"),
    ("[[1.0]] * len(pairs)", "not a finite number"),
    ("[True] * len(pairs)", "not a finite number"),
    ("None", "TypeError"),
])
def test_a_port_that_returns_the_wrong_shape_or_a_non_finite_score_fails_the_run(tmp_path, bad, word):
    run, stage, done = honest_run(tmp_path, port_src=HONEST.replace("return [(ids[-1] - 100) / 10 for ids in pairs]", f"return {bad}"))
    assert done.returncode == 2 and word in done.stdout, done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


# ---- bounds ---------------------------------------------------------------------------------------------

def test_a_port_that_reverses_every_group_is_refused_on_correlation_top_one_and_order(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=REVERSED)
    assert done.returncode == 0
    r = gate(run, stage)
    assert mentions(r, "pearson") and mentions(r, "top document matches") and mentions(r, "keep the reference's order")


def test_a_port_with_a_constant_score_has_correlation_zero_and_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=CONSTANT)
    assert record(stage)["metrics"]["pearson"] == 0.0
    assert mentions(gate(run, stage), "pearson 0.00000 is below")


def test_a_port_whose_scores_are_off_by_more_than_the_mean_bound_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=NOISY)
    assert mentions(gate(run, stage), "mean absolute score difference")


def test_a_max_difference_above_its_bound_alone_is_refused(tmp_path):
    one = HONEST.replace("return [(ids[-1] - 100) / 10 for ids in pairs]", "return [(ids[-1] - 100) / 10 + (1.5 if ids[0] == 1 and ids[1] == 4 else 0) for ids in pairs]")
    run, stage, done = honest_run(tmp_path, port_src=one)
    r = gate(run, stage)
    assert mentions(r, "max absolute score difference 1.5") and not mentions(r, "mean absolute")


def test_the_order_share_is_judged_on_pairs_the_reference_separates_by_the_gap(tmp_path):
    # Group g0's two lowest documents (reference -1.0 and -2.5 shifted) swap. That pair is separated by 1.5, so it counts; the share falls
    # below the floor only when enough pairs flip, so one swap in 8 groups passes the order bound and the same swap in every group fails it.
    swap = HONEST.replace("return [(ids[-1] - 100) / 10 for ids in pairs]",
                          "return [(ids[-1] - 100) / 10 if ids[1] not in (3, 4) else (ids[-1] - 100) / 10 + (-1.5 if ids[1] == 3 else 1.5) for ids in pairs]")
    run, stage, done = honest_run(tmp_path, port_src=swap)
    assert mentions(gate(run, stage), "keep the reference's order")


@pytest.mark.parametrize("n_groups", [3, 5])
def test_a_set_with_too_few_groups_or_pairs_is_refused(tmp_path, n_groups):
    rows = pair_rows()[:4 * n_groups]
    run, stage, done = honest_run(tmp_path, rows=rows, scores=REFERENCE_SCORES[:4 * n_groups])
    assert done.returncode == 0
    r = gate(run, stage)
    assert mentions(r, "pairs; at least 24") and mentions(r, "groups; at least 6")


def test_a_config_may_tighten_a_bound(tmp_path):
    cfg = sc.draft_config("/x")
    cfg["bounds"]["mean_abs_diff_max"] = 0.01
    run, stage, done = honest_run(tmp_path, port_src=HONEST.replace("/ 10 for", "/ 10 + 0.05 for"), config=cfg)
    assert mentions(gate(run, stage), "mean absolute score difference 0.05")


@pytest.mark.parametrize("key, val", [("pearson_min", 0.5), ("mean_abs_diff_max", 5.0), ("max_abs_diff_max", 100), ("top1_match_min", 0.0),
                                      ("order_kept_min", 0.1), ("min_pairs", 1), ("min_groups", 1), ("min_order_pairs", 0)])
def test_a_config_that_loosens_a_bound_is_refused_whatever_the_numbers(tmp_path, key, val):
    cfg = sc.draft_config("/x")
    cfg["bounds"][key] = val
    run, stage, done = honest_run(tmp_path, config=cfg)
    assert mentions(gate(run, stage), f"bound {key} {val} is looser")


@pytest.mark.parametrize("bounds", [{"pearson_min": None}, {"pearson_min": "0"}, {"pearson_min": [0.9]}, {"pearson_min": True}, {"nope": 1}, [1], "x", 5])
def test_a_config_with_a_bound_of_the_wrong_type_or_an_unknown_bound_is_refused(tmp_path, bounds):
    cfg = sc.draft_config("/x")
    cfg["bounds"] = bounds
    run, stage, done = honest_run(tmp_path, config=cfg)
    assert mentions(gate(run, stage), "bound" if isinstance(bounds, dict) else "bounds must be an object")


def test_a_looser_config_bound_does_not_loosen_the_bound_that_is_applied(tmp_path):
    cfg = sc.draft_config("/x")
    cfg["bounds"]["mean_abs_diff_max"] = 5.0
    run, stage, done = honest_run(tmp_path, port_src=NOISY, config=cfg)
    assert mentions(gate(run, stage), "mean absolute score difference 0.4")        # judged against the default 0.25


def test_a_config_of_the_other_kind_than_the_stage_1_files_is_refused_by_the_template(tmp_path):
    cfg = sc.draft_config("/x")
    cfg["kind"] = "embedding"
    run, stage, done = honest_run(tmp_path, config=cfg)
    assert done.returncode == 2 and "reference files are for kind 'reranker', and this check runs kind 'embedding'" in done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


def test_a_config_with_a_kind_the_check_does_not_know_is_refused_by_the_template(tmp_path):
    cfg = sc.draft_config("/x")
    cfg["kind"] = "generative"
    run, stage, done = honest_run(tmp_path, config=cfg)
    assert done.returncode == 2 and "kind is 'generative'" in done.stdout


# ---- the numbers: the template and the gate compute the same ----------------------------------------------

def _cases():
    rnd = random.Random(5)
    groups = [f"g{i // 4}" for i in range(24)]
    yield [rnd.uniform(-5, 5) for _ in range(24)], [rnd.uniform(-5, 5) for _ in range(24)], groups
    base = [rnd.uniform(-5, 5) for _ in range(24)]
    yield base, [b + rnd.uniform(-0.3, 0.3) for b in base], groups
    yield base, list(base), groups
    yield [1.0] * 24, [1.0] * 24, groups                       # no separated pair, constant
    yield base, [2.0] * 24, groups                             # a constant chip
    yield [0.0, 0.5, 1.0, 1.0], [1.0, 1.0, 0.0, 0.5], ["a", "a", "b", "b"]
    yield [3.0], [3.0], ["a"]
    tiny = [0.0, 1e-160, 2e-160, 3e-160] * 8                    # the variance sums are positive and their product underflows to 0
    yield tiny, list(tiny), [f"g{i // 4}" for i in range(32)]
    yield tiny, [3e-160 - x for x in tiny], [f"g{i // 4}" for i in range(32)]


@pytest.mark.parametrize("case", list(_cases()))
def test_the_templates_numbers_and_the_gates_numbers_are_the_same(case):
    ref, chip, groups = case
    assert tpl.parity(ref, chip, groups) == sc.parity(ref, chip, groups)
    assert tpl.ORDER_GAP == sc.ORDER_GAP


def test_scores_so_close_together_that_the_variance_product_underflows_do_not_crash_the_numbers():
    tiny = [0.0, 1e-160, 2e-160, 3e-160] * 8
    groups = [f"g{i // 4}" for i in range(32)]
    for parity in (tpl.parity, sc.parity):
        assert parity(tiny, list(tiny), groups)["pearson"] == pytest.approx(1.0)
        assert parity(tiny, [-x for x in tiny], groups)["pearson"] == pytest.approx(-1.0)
        big = [1e6, -1e6, 1e6, -1e6] * 8                      # the largest size a score may have
        m = parity(big, [-x for x in big], groups)
        assert m["pearson"] == pytest.approx(-1.0) and m["max_abs_diff"] == 2e6
        assert parity(big, [0.0] * 32, groups)["pearson"] == 0.0


def test_the_numbers_on_a_small_case_worked_by_hand():
    ref, chip, groups = [3.0, 0.0, -3.0, 1.0, 0.8], [2.0, 2.5, -2.0, 0.0, 1.0], ["a", "a", "a", "b", "b"]
    m = sc.parity(ref, chip, groups)
    assert m["top1_groups"] == 2 and m["top1_groups_matching"] == 0           # group a: the reference picks document 0, the chip 1; group b: 0 against 1
    # separated pairs (gap >= 0.5), group a only: (0,1) gap 3 chip -0.5 -> broken; (0,2) gap 6 chip 4 -> kept; (1,2) gap 3 chip 4.5 -> kept; group b gap 0.2 is not counted
    assert (m["order_pairs"], m["order_pairs_kept"]) == (3, 2)
    assert m["mean_abs_diff"] == pytest.approx((1 + 2.5 + 1 + 1 + 0.2) / 5) and m["max_abs_diff"] == 2.5


# ---- the forgeries ----------------------------------------------------------------------------------------

def test_a_port_that_prints_its_own_check_line_fails_the_run(tmp_path):
    forged = HONEST.replace("def load(model_dir, mesh_device):", "def load(model_dir, mesh_device):\n    print('SCORING_CHECK_SHA256 ' + '0' * 64)")
    run, stage, done = honest_run(tmp_path, port_src=forged)
    assert done.returncode == 2 and "mentions SCORING_CHECK" in done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists() and not MARK.search(done.stdout)


def test_a_check_line_added_to_the_output_afterwards_is_refused_by_the_gate(tmp_path):
    run, stage, done = honest_run(tmp_path)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(out.read_text() + "SCORING_CHECK_SHA256 " + "0" * 64 + "\n")
    write_marker(run, stage, subprocess.CompletedProcess([], 0, out.read_text(), ""), CHIPS)
    assert mentions(gate(run, stage), "begin with SCORING_CHECK_SHA256")


def test_a_port_that_writes_the_record_itself_has_it_replaced_by_the_templates(tmp_path):
    forged = REVERSED.replace("def load(model_dir, mesh_device):", '''def load(model_dir, mesh_device):
    import json, pathlib
    p = pathlib.Path("stages/2/evidence/scoring-check.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": "orchard-scoring-check/1", "chip_scores": [0.0]}))''')
    run, stage, done = honest_run(tmp_path, port_src=forged)
    assert done.returncode == 0
    assert record(stage)["metrics"]["pearson"] < 0 and len(record(stage)["chip_scores"]) == 32          # the real scores, not the forged ones
    assert mentions(gate(run, stage), "pearson")


def test_a_record_rewritten_after_the_run_does_not_match_the_output_and_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=REVERSED)
    rec = record(stage)
    rec["chip_scores"] = list(rec["reference_scores"])
    rec["metrics"] = tpl.parity(rec["reference_scores"], rec["chip_scores"], rec["groups"])
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))
    r = gate(run, stage)
    assert mentions(r, "SCORING_CHECK_SHA256 line that matches")


def test_a_record_whose_scores_were_changed_but_not_its_metrics_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=REVERSED)
    rec = record(stage)
    rec["chip_scores"] = list(rec["reference_scores"])
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))
    r = gate(run, stage)
    assert mentions(r, "metrics.pearson") and mentions(r, "metrics.mean_abs_diff") and mentions(r, "metrics.top1_groups_matching")


def test_a_reported_metric_that_is_off_by_a_little_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=NOISY)
    rec = record(stage)
    rec["metrics"]["mean_abs_diff"] -= 1e-6
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))
    assert mentions(gate(run, stage), "metrics.mean_abs_diff")


def test_a_set_where_the_reference_separates_too_few_pairs_cannot_judge_the_order_and_is_refused(tmp_path):
    # Every reference score is scaled by 0.1: only the top and bottom document of a group stay 0.5 apart, so 8 pairs are separated
    # (the floor is 10). The port matches the reference exactly, so no other bound is in play.
    scaled = HONEST.replace("(ids[-1] - 100) / 10 for", "(ids[-1] - 100) / 100 for")
    run, stage, done = honest_run(tmp_path, port_src=scaled, scores=[s / 10 for s in REFERENCE_SCORES])
    r = gate(run, stage)
    assert mentions(r, "separates only 8 pairs") and len(r) == 1


def test_a_record_that_swaps_in_other_reference_scores_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=REVERSED)
    rec = record(stage)
    rec["reference_scores"] = [-s for s in rec["reference_scores"]]
    rec["metrics"] = tpl.parity(rec["reference_scores"], rec["chip_scores"], rec["groups"])
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))
    assert mentions(gate(run, stage), "reference_scores are not the scores in")


def test_a_record_that_regroups_the_pairs_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=REVERSED)
    rec = record(stage)
    rec["groups"] = [f"x{i}" for i in range(32)]
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))
    assert mentions(gate(run, stage), "another grouping")


@pytest.mark.parametrize("which", ["scoring-reference.json", "scoring-pairs.json"])
def test_a_stage_1_file_changed_after_the_run_is_refused(tmp_path, which):
    run, stage, done = honest_run(tmp_path)
    f = run / "stages" / "1" / "evidence" / "reference" / which
    f.write_text(f.read_text().replace('"reranker"', '"reranker" ', 1))
    assert mentions(gate(run, stage), "sha256 does not match")


def test_a_reference_whose_pair_file_changed_before_the_run_fails_the_run(tmp_path):
    run, stage, _ = make_run(tmp_path)
    p = run / "stages" / "1" / "evidence" / "reference" / "scoring-pairs.json"
    p.write_text(p.read_text() + " ")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "pair file changed" in done.stdout


def test_a_port_that_replaces_the_template_copy_is_refused(tmp_path):
    swap = HONEST.replace("def load(model_dir, mesh_device):", '''def load(model_dir, mesh_device):
    import pathlib
    pathlib.Path("stages/2/scoring_check.py").write_text("print('forged')\\n")''')
    run, stage, done = honest_run(tmp_path, port_src=swap)
    assert mentions(gate(run, stage), "differs from the shipped template")


def test_a_record_written_by_an_edited_template_is_refused(tmp_path):
    edited = sc.TEMPLATE.read_text().replace("ORDER_GAP = 0.5", "ORDER_GAP = 50.0")
    assert edited != sc.TEMPLATE.read_text()
    run, stage, done = honest_run(tmp_path, template_text=edited)
    r = gate(run, stage)
    assert mentions(r, "differs from the shipped template") and mentions(r, "template_sha256")


def test_a_missing_template_copy_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path)
    (stage / sc.TEMPLATE_NAME).unlink()
    assert mentions(gate(run, stage), "is not in this stage directory")


def test_a_different_command_than_the_templates_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path)
    marker = json.loads((stage / "test-result.json").read_text())
    marker["command"] = "python3 stages/2/port/check.py"
    (stage / "test-result.json").write_text(json.dumps(marker))
    assert mentions(gate(run, stage), "command for this stage")


@pytest.mark.parametrize("change, word", [({"returncode": 1}, "exited with code 1"), ({"returncode": None}, "no exit code"),
                                          ({"timed_out": True}, "past its deadline"), ({"chips": ["0000:09:00.0"]}, "not the chips")])
def test_a_test_that_failed_timed_out_or_ran_on_other_chips_is_refused(tmp_path, change, word):
    run, stage, done = honest_run(tmp_path)
    marker = json.loads((stage / "test-result.json").read_text())
    marker.update(change)
    (stage / "test-result.json").write_text(json.dumps(marker))
    assert mentions(gate(run, stage), word)


def test_a_config_changed_after_the_test_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path)
    cfg = sc.draft_config("/elsewhere")
    (stage / sc.CONFIG_NAME).write_text(json.dumps(cfg))
    assert mentions(gate(run, stage), sc.CONFIG_NAME)


def test_a_port_file_added_after_the_record_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path)
    (stage / "port" / "late.py").write_text("import ttnn\n")
    assert mentions(gate(run, stage), "late.py")


def test_a_port_that_does_not_mention_ttnn_or_reads_the_reference_is_refused(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=HONEST.replace("import ttnn", "import os"))
    assert mentions(gate(run, stage), "mentions ttnn")
    run, stage, done = honest_run(tmp_path / "b", port_src=HONEST + "\nREF = 'stages/1/evidence/reference/scoring-reference.json'\n")
    assert mentions(gate(run, stage), "refers to the stage 1 reference files")


@pytest.mark.parametrize("where", ["sitecustomize.py", "port/sitecustomize.py", "json.py", "hashlib.py"])
def test_a_file_the_coder_leaves_next_to_the_template_or_in_the_port_directory_is_not_imported_at_startup(tmp_path, where):
    run, stage, _ = make_run(tmp_path, extra_files={where: "import pathlib\npathlib.Path('SITECUSTOMIZE-RAN').write_text('yes')\n"})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (run / "SITECUSTOMIZE-RAN").exists()


def test_a_port_that_patches_the_templates_functions_and_hooks_imports_cannot_change_the_record(tmp_path):
    patch = '''
import sys
main = sys.modules["__main__"]
main.sha_bytes = lambda data: "0" * 64
main.sha_file = lambda path: "0" * 64
main.parity = lambda r, c, g: {"pearson": 1.0}
main.pearson = lambda a, b: 1.0
main.PREFIX = ""
class Hook:
    def find_spec(self, name, path=None, target=None):
        return None
sys.meta_path.insert(0, Hook())
main.main = lambda: 0
'''
    run, stage, done = honest_run(tmp_path, port_src=patch + REVERSED)
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    assert rec["template_sha256"] == sc.template_sha256() and rec["metrics"]["pearson"] < 0
    assert MARK.fullmatch(done.stdout.strip().splitlines()[-1])
    r = gate(run, stage)
    assert mentions(r, "pearson") and not mentions(r, "template")


LATE_WRITER = '''
def close(model):
    import threading, time, json, pathlib
    def later():
        rec = pathlib.Path("stages/2/evidence/scoring-check.json")
        for _ in range(300):
            if rec.exists():
                rec.write_text(json.dumps({"chip_scores": [0] * 32}))
                return
            time.sleep(0.01)
    threading.Thread(target=later).start()
'''


def test_a_port_thread_that_writes_the_record_after_close_is_waited_for_and_overwritten(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=HONEST + LATE_WRITER)
    assert done.returncode == 0, done.stdout + done.stderr
    time.sleep(0.5)
    assert record(stage)["chip_scores"] == REFERENCE_SCORES
    assert gate(run, stage) == []


def test_a_link_under_the_port_directory_fails_the_run_and_the_gate(tmp_path):
    run, stage, _ = make_run(tmp_path)
    (tmp_path / "elsewhere.py").write_text("import ttnn\n")
    os.symlink(tmp_path / "elsewhere.py", stage / "port" / "layers.py")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "is a link" in done.stdout
    run2, stage2, done2 = honest_run(tmp_path / "b")
    os.symlink(tmp_path / "elsewhere.py", stage2 / "port" / "layers.py")
    assert mentions(gate(run2, stage2), "is a link")


def test_a_process_the_port_starts_is_killed_before_the_run_ends(tmp_path):
    spawn = HONEST.replace("def load(model_dir, mesh_device):", '''def load(model_dir, mesh_device):
    import subprocess
    p = subprocess.Popen(["sleep", "60"])
    open("CHILD-PID", "w").write(str(p.pid))''')
    run, stage, done = honest_run(tmp_path, port_src=spawn)
    assert done.returncode == 0
    pid = int((run / "CHILD-PID").read_text())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("a process the port started was still running after the check ended")


def test_the_worker_does_not_survive_when_the_outer_command_is_killed(tmp_path):
    hang = HONEST.replace("def load(model_dir, mesh_device):\n    return {\"dir\": model_dir, \"device\": mesh_device}\n", "") + '''
def load(model_dir, mesh_device):
    import os, time
    open("WORKER-PID", "w").write(str(os.getpid()))
    time.sleep(120)
'''
    run, stage, _ = make_run(tmp_path, hang)
    env = {"PATH": os.environ.get("PATH", ""), "TT_VISIBLE_DEVICES": ",".join(CHIPS), "PYTHONPATH": python_path(tmp_path)}
    outer = subprocess.Popen([sys.executable, *sc.COMMAND.split()[1:]], cwd=run, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(200):
        if (run / "WORKER-PID").exists() and (run / "WORKER-PID").read_text():
            break
        time.sleep(0.05)
    pid = int((run / "WORKER-PID").read_text())
    os.killpg(outer.pid, signal.SIGKILL)
    outer.wait()
    alive = True
    for _ in range(100):
        try:
            os.kill(pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            alive = False
            break
    if alive:
        os.kill(pid, 9)
    assert not alive, "the worker outlived the killed outer command"


# ---- hostile inputs to the template ------------------------------------------------------------------------

def _stage1(run, name):
    return run / "stages" / "1" / "evidence" / "reference" / name


@pytest.mark.parametrize("name", ["scoring-pairs.json", "scoring-reference.json"])
def test_a_stage_1_file_that_is_a_link_or_a_fifo_fails_the_run_without_hanging(tmp_path, name):
    run, stage, _ = make_run(tmp_path)
    f = _stage1(run, name)
    real = f.read_bytes()
    f.unlink()
    (tmp_path / "real.json").write_bytes(real)
    os.symlink(tmp_path / "real.json", f)
    assert run_template(tmp_path, run, stage).returncode == 2
    f.unlink()
    os.mkfifo(f)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "is not a regular file" in done.stdout


def _patch_pairs(run, mutate):
    f = _stage1(run, "scoring-pairs.json")
    doc = json.loads(f.read_text())
    mutate(doc)
    f.write_text(json.dumps(doc))
    ref = _stage1(run, "scoring-reference.json")
    r = json.loads(ref.read_text())
    import hashlib
    r["pairs"]["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    ref.write_text(json.dumps(r))


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(pairs=[]), lambda d: d.update(pairs=None), lambda d: d.update(pairs={}), lambda d: d["pairs"].__setitem__(0, None),
    lambda d: d["pairs"][0].update(input_ids=[]), lambda d: d["pairs"][0].update(input_ids=[None]), lambda d: d["pairs"][0].update(input_ids=[True]),
    lambda d: d["pairs"][0].update(input_ids=[-1]), lambda d: d["pairs"][0].update(input_ids=[1.5]), lambda d: d["pairs"][0].update(input_ids="abc"),
    lambda d: d["pairs"][0].update(group=None), lambda d: d["pairs"][0].update(group=["a"]), lambda d: d["pairs"][0].update(input_ids=[10 ** 30]),
])
def test_a_pair_file_of_the_wrong_shape_fails_the_run_with_a_message(tmp_path, mutate):
    run, stage, _ = make_run(tmp_path)
    _patch_pairs(run, mutate)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "FAILED:" in done.stdout and "Traceback" not in done.stdout + done.stderr


@pytest.mark.parametrize("scores", [[], None, "x", [1.0] * 31, [None] * 32, [True] * 32, ["1"] * 32, [1e30] * 32, {"a": 1}])
def test_a_reference_of_the_wrong_shape_fails_the_run_with_a_message(tmp_path, scores):
    run, stage, _ = make_run(tmp_path)
    ref = _stage1(run, "scoring-reference.json")
    r = json.loads(ref.read_text())
    r["scores"] = scores
    ref.write_text(json.dumps(r))
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "finite scores" in done.stdout and "Traceback" not in done.stdout + done.stderr


def test_a_reference_score_of_nan_is_refused(tmp_path):
    run, stage, _ = make_run(tmp_path)
    ref = _stage1(run, "scoring-reference.json")
    ref.write_text(ref.read_text().replace("[", "[NaN, ", 1))
    assert run_template(tmp_path, run, stage).returncode == 2


@pytest.mark.parametrize("cfg", ["[]", "null", '{"kind": "reranker"}', '{"model_snapshot": 5, "kind": "reranker"}', '{"model_snapshot": "", "kind": "reranker"}', "{"])
def test_a_config_of_the_wrong_shape_fails_the_run_with_a_message(tmp_path, cfg):
    run, stage, _ = make_run(tmp_path)
    (stage / sc.CONFIG_NAME).write_text(cfg)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "FAILED:" in done.stdout and "Traceback" not in done.stdout + done.stderr


def test_no_chips_in_the_environment_fails_the_run(tmp_path):
    run, stage, _ = make_run(tmp_path)
    done = run_template(tmp_path, run, stage, chips=[])
    assert done.returncode == 2 and "TT_VISIBLE_DEVICES names no chip" in done.stdout


# ---- the gate on a missing or hostile record ---------------------------------------------------------------

def test_a_missing_record_is_a_refusal_and_not_a_skip(tmp_path):
    run, stage, done = honest_run(tmp_path)
    (stage / "evidence" / "scoring-check.json").unlink()
    assert mentions(gate(run, stage), "scoring-check.json is missing")


@pytest.mark.parametrize("field", ["chip_scores", "metrics", "groups", "reference_scores", "chips", "port_sources", "config", "load_s", "score_s", "reference", "pairs"])
@pytest.mark.parametrize("value", [None, [], {}, "x", 7, [None], {"a": 1}, -1])
def test_a_record_field_of_the_wrong_type_is_a_reason_and_never_an_exception(tmp_path, field, value):
    run, stage, done = honest_run(tmp_path)
    rec = record(stage)
    rec[field] = value
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))
    assert gate(run, stage)


# ---- review findings (PR 383) -------------------------------------------------------------------------------

def _write_record(stage, rec):
    (stage / "evidence" / "scoring-check.json").write_text(json.dumps(rec))


def test_a_pair_file_over_the_pair_limit_is_refused_without_the_quadratic_work(tmp_path):
    # 100,000 pairs in one group would be about five billion comparisons in `parity`; the limit is checked first.
    run, stage, done = honest_run(tmp_path)
    n = 100_000
    f = run / "stages" / "1" / "evidence" / "reference" / "scoring-pairs.json"
    f.write_text(json.dumps({"model_kind": "reranker", "pairs": [{"group": "g", "input_ids": [1]}] * n}))
    ref = run / "stages" / "1" / "evidence" / "reference" / "scoring-reference.json"
    ref.write_text(json.dumps({"model_kind": "reranker", "pairs": {"sha256": "0" * 64}, "scores": [0.0] * n}))
    rec = record(stage)
    rec.update(chip_scores=[0.0] * n, reference_scores=[0.0] * n, groups=["g"] * n)
    _write_record(stage, rec)
    t0 = time.monotonic()
    r = gate(run, stage)
    assert time.monotonic() - t0 < 5 and mentions(r, "1 to 4096")


@pytest.mark.parametrize("where", ["chip", "reference"])
def test_a_score_of_huge_size_is_a_reason_and_not_an_overflow(tmp_path, where):
    run, stage, done = honest_run(tmp_path)
    if where == "chip":
        rec = record(stage)
        rec["chip_scores"][0] = 1e200
        _write_record(stage, rec)
    else:
        ref = run / "stages" / "1" / "evidence" / "reference" / "scoring-reference.json"
        doc = json.loads(ref.read_text())
        doc["scores"][0] = 1e200
        ref.write_text(json.dumps(doc))
    assert mentions(gate(run, stage), "of size at most")


@pytest.mark.parametrize("value", [[1], "x", 5, None, True])
def test_a_reference_whose_pairs_field_is_not_an_object_is_a_reason_and_not_an_exception(tmp_path, value):
    run, stage, done = honest_run(tmp_path)
    ref = run / "stages" / "1" / "evidence" / "reference" / "scoring-reference.json"
    doc = json.loads(ref.read_text())
    doc["pairs"] = value
    ref.write_text(json.dumps(doc))
    assert mentions(gate(run, stage), "pairs must be an object")


REWRITE_SELF = '''
def close(model):
    import pathlib
    p = pathlib.Path("stages/2/port/port_model.py")
    p.write_text(p.read_text() + "\\n# swapped in by close\\n")
'''
ADD_FILE = '''
def close(model):
    import pathlib
    pathlib.Path("stages/2/port/extra.py").write_text("import ttnn\\n")
'''


@pytest.mark.parametrize("src", [REWRITE_SELF, ADD_FILE])
def test_a_port_that_changes_its_own_source_while_it_runs_fails_the_run_and_leaves_no_record(tmp_path, src):
    run, stage, done = honest_run(tmp_path, port_src=HONEST + src)
    assert done.returncode == 2 and "while the port ran" in done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()


@pytest.mark.parametrize("name", ["__pycache__/port_model.cpython-312.pyc", "stale.pyc", "native.so", "x.pth", "pkg/__pycache__/a.txt"])
def test_importable_code_the_manifest_does_not_hash_fails_the_run_and_is_refused_by_the_gate(tmp_path, name):
    # A pre-existing .pyc with a matching source stamp would be loaded in place of the .py file the gate reads (PR 383 review).
    run, stage, _ = make_run(tmp_path, extra_files={"port/" + name: "x"})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "importable code the check does not hash" in done.stdout
    assert not (stage / "evidence" / "scoring-check.json").exists()
    run2, stage2, done2 = honest_run(tmp_path / "b")
    p = stage2 / "port" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    assert mentions(gate(run2, stage2), "importable code that port_sources does not hash")


ATEXIT_SPAWN = '''
import atexit
def _late():
    import subprocess
    p = subprocess.Popen(["sleep", "60"])
    open("ATEXIT-PID", "w").write(str(p.pid))
    open("ATEXIT-RAN", "w").write("yes")
atexit.register(_late)
'''
THREAD_SPAWN = '''
def close(model):
    import subprocess, threading, time
    def loop():
        while True:
            p = subprocess.Popen(["sleep", "60"])
            with open("SPAWNED-PIDS", "a") as f:
                f.write(str(p.pid) + "\\n")
            time.sleep(0.005)
    threading.Thread(target=loop).start()
    time.sleep(0.2)              # the thread has started some processes by the time close returns
'''


FORKING_CHILD = '''
def close(model):
    import subprocess, sys, time
    code = ("import subprocess, time\\n"
            "while True:\\n"
            "    p = subprocess.Popen(['sleep', '60'])\\n"
            "    open('GRANDCHILD-PIDS', 'a').write(str(p.pid) + '\\\\n')\\n"
            "    time.sleep(0.005)\\n")
    subprocess.Popen([sys.executable, "-c", code])
    time.sleep(0.3)              # the child has started some grandchildren by the time close returns
'''


ORPHAN = '''
def close(model):
    import subprocess, time
    # the shell starts a sleeper and exits at once: the sleeper is re-parented, and its parent is gone before close returns
    subprocess.run(["sh", "-c", "sleep 60 & echo $! > ORPHAN-PID"])
    time.sleep(0.1)
'''


def _alive(pid):
    try:
        os.kill(pid, 0)
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def _gone(pids):
    for _ in range(100):
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.05)
    for p in pids:
        if _alive(p):
            os.kill(p, 9)
    return False


def test_a_child_that_keeps_forking_is_stopped_with_everything_it_started(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=HONEST + FORKING_CHILD)
    assert done.returncode == 0, done.stdout + done.stderr
    pids = [int(x) for x in (run / "GRANDCHILD-PIDS").read_text().split()]
    assert pids and _gone(pids), "a grandchild was still running after the check ended"
    time.sleep(0.2)
    assert len((run / "GRANDCHILD-PIDS").read_text().split()) == len(pids)


def test_a_process_still_alive_after_the_kill_fails_the_run_instead_of_writing_a_record(monkeypatch):
    class Stub:
        descendants = staticmethod(lambda: [1])
    monkeypatch.setattr(tpl.time, "sleep", lambda s: None)
    with pytest.raises(tpl.CheckFailed, match="still running after the check killed"):
        tpl.Worker.confirm_none_left(Stub())
    Stub.descendants = staticmethod(lambda: [])
    tpl.Worker.confirm_none_left(Stub())


def test_a_port_score_of_four_hundred_digits_is_a_controlled_refusal_and_not_a_traceback(tmp_path):
    big = HONEST.replace("return [(ids[-1] - 100) / 10 for ids in pairs]", "return [10 ** 400] * len(pairs)")
    run, stage, done = honest_run(tmp_path, port_src=big)
    assert done.returncode == 2 and "not a finite number" in done.stdout and "Traceback" not in done.stdout + done.stderr


def test_a_reference_score_of_four_hundred_digits_is_a_controlled_refusal_in_the_template(tmp_path):
    run, stage, _ = make_run(tmp_path)
    ref = _stage1_ref(run)
    doc = json.loads(ref.read_text())
    doc["scores"][0] = 10 ** 400
    ref.write_text(json.dumps(doc))
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "finite scores" in done.stdout and "Traceback" not in done.stdout + done.stderr


def _stage1_ref(run):
    return run / "stages" / "1" / "evidence" / "reference" / "scoring-reference.json"


def test_the_gate_helpers_do_not_raise_on_an_integer_of_four_hundred_digits():
    assert sc._num(10 ** 400) is False and sc._score(10 ** 400) is False and sc._score(-10 ** 400) is False
    assert sc._score(5) and sc._score(1e6) and not sc._score(1e6 + 1)


def test_a_record_with_a_four_hundred_digit_score_is_a_reason_in_the_gate(tmp_path):
    run, stage, done = honest_run(tmp_path)
    rec = record(stage)
    rec["chip_scores"][0] = 10 ** 400
    _write_record(stage, rec)
    assert mentions(gate(run, stage), "not readable JSON")          # safeio refuses an integer of that many digits before the gate sees it


def test_stop_calls_the_confirmation_after_it_has_killed_the_worker():
    calls = []
    sleeper = subprocess.Popen(["sleep", "60"])

    class Stub:
        proc = sleeper
        pumps = []
        wr = open(os.devnull, "w")
        descendants = staticmethod(lambda: [])
        confirm_none_left = lambda self: calls.append(sleeper.poll())      # the worker must already be dead when the check runs
    try:
        tpl.Worker.stop(Stub())
    finally:
        sleeper.kill()
        sleeper.wait()
    assert calls and calls[0] is not None


def test_a_grandchild_whose_parent_exited_before_close_is_killed_and_a_record_is_written(tmp_path):
    # worker -> child -> grandchild; the child exits before close returns, so the grandchild is re-parented to the template.
    grand = '''
def close(model):
    import subprocess, time
    subprocess.run(["sh", "-c", "(sleep 60 & echo $! > GRANDCHILD-PID) ; exit 0"])
    time.sleep(0.1)
'''
    run, stage, done = honest_run(tmp_path, port_src=HONEST + grand)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _gone([int((run / "GRANDCHILD-PID").read_text())]) and gate(run, stage) == []


def test_a_process_whose_parent_exited_before_the_check_stopped_is_killed_too(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=HONEST + ORPHAN)
    assert done.returncode == 0, done.stdout + done.stderr
    pid = int((run / "ORPHAN-PID").read_text())
    assert _gone([pid]), "an orphaned process the port started was still running after the check ended"


def test_an_atexit_callback_in_the_worker_never_runs_and_starts_nothing(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=HONEST + ATEXIT_SPAWN)
    assert done.returncode == 0, done.stdout + done.stderr
    time.sleep(0.3)
    assert not (run / "ATEXIT-RAN").exists() and not (run / "ATEXIT-PID").exists()
    assert gate(run, stage) == []


def test_a_thread_that_keeps_starting_processes_after_close_leaves_none_running(tmp_path):
    run, stage, done = honest_run(tmp_path, port_src=HONEST + THREAD_SPAWN)
    assert done.returncode == 0, done.stdout + done.stderr
    pids = [int(x) for x in (run / "SPAWNED-PIDS").read_text().split()]
    assert pids and _gone(pids), "a process the port's thread started was still running after the check ended"
    time.sleep(0.2)
    assert len((run / "SPAWNED-PIDS").read_text().split()) == len(pids)         # nothing was started after the check stopped the worker


def test_the_record_lists_the_port_files_as_they_were_before_the_worker_started(tmp_path):
    import hashlib
    run, stage, done = honest_run(tmp_path)
    listed = {s["path"]: s["sha256"] for s in record(stage)["port_sources"]}
    assert listed == {"stages/2/port/port_model.py": hashlib.sha256((stage / "port" / "port_model.py").read_bytes()).hexdigest()}
