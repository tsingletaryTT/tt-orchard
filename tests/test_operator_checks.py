# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The operator's post-run checks (orchard/operator_checks.py): read-only, no hardware.

The hub lookup is injected, so no test touches the network."""
import json

from orchard import operator_checks as oc

PUBLISH = (
    "# comment hf upload --repo-type model --private x/commented .\n"
    "hf upload --repo-type model --private episod/model-p300 stages/7/package/model-p300 .\n"
    "hf upload --repo-type model --private episod/model-p150 stages/7/package/model-p150 .\n")


def make_run(tmp_path, *, calls=(), bundle_text="clean", written=()):
    run = tmp_path / "run"
    (run / "stages" / "8" / "bundle").mkdir(parents=True)
    (run / "stages" / "8" / "log").mkdir(parents=True)
    (run / "stages" / "8" / "bundle" / "PUBLISH_COMMANDS.txt").write_text(PUBLISH)
    (run / "stages" / "8" / "bundle" / "RESULTS.md").write_text(bundle_text)
    with open(run / "stages" / "8" / "log" / "run-1.jsonl", "w") as f:
        # The "sent" side holds the skill text, which names these commands as forbidden. Only the
        # model's own tool calls count, so this line must not be reported.
        f.write(json.dumps({"turn": 1, "sent": [{"content": "never run hf upload or git push"}],
                            "received": {"tool_calls": [{"function": {"name": "shell",
                                                                      "arguments": json.dumps({"command": c})}}
                                                        for c in calls]
                                                       + [{"function": {"name": "write_file",
                                                                        "arguments": json.dumps({"content": w})}}
                                                          for w in written]}}) + "\n")
    return run


def test_repo_ids_come_from_uncommented_upload_lines(tmp_path):
    assert oc.repos_in_publish_file(make_run(tmp_path)) == ["episod/model-p300", "episod/model-p150"]


def test_a_clean_run_passes(tmp_path):
    run = make_run(tmp_path, calls=["ls", "cat RESULTS.md"])
    result = oc.check(run, fetch=lambda repo: 404, hostname="h", home="/home/nobody")
    assert result["ok"] is True and result["problems"] == []


def test_a_publish_like_tool_call_is_a_problem_but_the_skill_text_is_not(tmp_path):
    run = make_run(tmp_path, calls=["hf upload --private a/b ."])
    problems = oc.check(run, fetch=lambda repo: 404, hostname="h", home="/home/nobody")["problems"]
    assert len(problems) == 1 and "hf upload" in problems[0]


def test_text_written_with_write_file_is_not_a_command(tmp_path):
    # Stage 8 writes PUBLISH_COMMANDS.txt, which is full of `hf upload` lines, with write_file.
    run = make_run(tmp_path, written=[PUBLISH])
    assert oc.check(run, fetch=lambda repo: 404, hostname="h", home="/home/nobody")["ok"] is True


def test_a_repo_that_exists_is_a_problem_and_an_unreachable_hub_is_reported(tmp_path):
    run = make_run(tmp_path)
    r = oc.check(run, fetch=lambda repo: 200 if repo.endswith("p150") else 404, hostname="h", home="/h")
    assert any("episod/model-p150" in p and "exists" in p for p in r["problems"])
    r = oc.check(run, fetch=lambda repo: None, hostname="h", home="/h")
    assert r["ok"] is True and any("hub lookup skipped" in n for n in r["notes"])


def test_a_secret_or_home_path_in_the_bundle_is_a_problem(tmp_path):
    run = make_run(tmp_path, bundle_text="see /home/someone/x and hf_" + "a" * 34)
    problems = oc.check(run, fetch=lambda repo: 404, hostname="h", home="/home/nobody")["problems"]
    assert any("RESULTS.md" in p and "token" in p for p in problems)
    assert any("RESULTS.md" in p and "home path" in p for p in problems)


def test_main_exit_codes(tmp_path, capsys):
    run = make_run(tmp_path, bundle_text="/home/someone/x")
    assert oc.main(["--run-dir", str(run), "--no-network"]) == 1
    assert "PROBLEM" in capsys.readouterr().out
    assert oc.main(["--run-dir", str(tmp_path / "missing")]) == 2
