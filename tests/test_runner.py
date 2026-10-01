"""Runner tests. Every rule has denied forms in DENY. A structural test fails when a rule is added
without one, so a rule cannot exist untested."""
import subprocess

import pytest

from orchard.runner import RULES, Denied, check_string, run_argv, run_shell

ALLOW = [
    "ls -la",
    "git status",
    "git -C {run} log --oneline",
    "git commit -m 'push it to the shelf'",
    "tt-smi -s",
    "gozer status --json",
    "tt-model serve some/model",
    "python3 -c 'print(1)'",
    "rm -rf {run}/scratch/old",
    "rm {run}/out.txt",
    "echo hi > {run}/out.txt",
    "cd {run}/sub; rm -rf cache",
]

DENY = {
    "tt-model-push": [
        "tt-model push episod/x",
        "/usr/local/bin/tt-model push .",
        "sudo tt-model push x",
        "sudo -u bob tt-model push x",
        "env A=1 tt-model push x",
        "FOO=1 tt-model push x",
        "timeout 10 tt-model push x",
        "bash -c 'tt-model push x'",
        "ls; tt-model push x",
        "true && tt-model push x",
        "false || tt-model push x",
        "echo a | tt-model push x",
    ],
    "tt-model-publish": ["tt-model publish episod/x"],
    "git-push": [
        "git push",
        "git -C /tmp/x push origin main",
        "git -c user.name=a push",
        "nice -n 5 git push",
        "/usr/bin/git push --force",
    ],
    "gh-repo-create": ["gh repo create foo --public", "gh --hostname x repo create foo"],
    "hf-upload": ["hf upload a b", "huggingface-cli upload a b"],
    "tt-smi-reset": ["tt-smi -r", "tt-smi -r 0,1", "tt-smi --reset", "sudo tt-smi -r", "tt-smi -r0"],
    "rm-outside-run-dir": [
        "rm -rf /home/someone/.cache/x",
        "rm {run}/../escaped",
        "rm -rf $HOME/cache",
        "rm -rf ~/code",
        "rm -rf {run}",
        "rm -rf {run}/*/../..",
        "cd /; rm -rf *",
        "cd $SOMEWHERE; rm file",
        "rmdir /tmp/x",
        "unlink /tmp/x",
    ],
    "rm-ledger": ["rm {run}/ledger.jsonl", "rm -f {run}/ledger*"],
    "rm-xargs": ["find . -name x | xargs rm", "ls | xargs -n 1 rm -rf"],
    "substitution": ["echo $(date)", "echo `date`", "diff <(ls) <(ls)"],
}


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    (d / "sub").mkdir(parents=True)
    return d


@pytest.mark.parametrize("cmd", ALLOW)
def test_allowed_commands_pass(cmd, run_dir):
    check_string(cmd.format(run=run_dir), run_dir)


@pytest.mark.parametrize(
    "rule,cmd", [(r, c) for r, cmds in DENY.items() for c in cmds])
def test_denied_commands_name_their_rule(rule, cmd, run_dir):
    with pytest.raises(Denied) as exc:
        check_string(cmd.format(run=run_dir), run_dir)
    assert exc.value.rule == rule


def test_every_rule_has_denied_examples():
    assert {name for name, _ in RULES} | {"substitution"} == set(DENY)


def test_run_argv_executes_allowed_and_refuses_denied(run_dir):
    done = run_argv(["echo", "hello"], run_dir, capture_output=True, text=True)
    assert done.stdout.strip() == "hello"
    with pytest.raises(Denied):
        run_argv(["git", "push"], run_dir)


def test_run_shell_checks_before_running(run_dir):
    marker = run_dir / "marker"
    with pytest.raises(Denied):
        run_shell(f"touch {marker}; git push", run_dir)
    assert not marker.exists()  # nothing ran, including the harmless first command
    run_shell(f"touch {marker}", run_dir)
    assert marker.exists()
