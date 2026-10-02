"""Runner tests. Every rule has denied forms in DENY. A structural test fails when a rule is added
without one, so a rule cannot exist untested."""
import subprocess

import pytest

from orchard.runner import RULES, Denied, check_string, run_argv, run_shell

ALLOW = [
    "cat < {run}/ledger.jsonl",
    "wc -l < {run}/ledger.jsonl 2>&1",
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
    "git commit -m 'x'",
    "git log --oneline -n 5",
    "git diff HEAD~1",
    "pytest -q tests/",
    "python3 -m pytest tests/test_x.py -q",
    "ls | grep x",
    "cd {run}/sub && make",
    "echo \"a b\" > {run}/f",
    "grep '#' file",
    "cmd 2>&1 | tail -5",
    "ls # a comment",
    "ls &>/dev/null",
    "echo ${HOME}",
    "false || echo no",
    "sleep 1 &",
    "printf 'a\\nb\\n' | wc -l",
    "tt-smi -s > {run}/snap.json 2>&1",
    "tt-smi -h",
    "timeout 10 pytest -q",
    "nice -n 5 make -j4",
    "env FOO=1 pytest -q",
    "sudo -n ls",
    "gozer run --chips 1 --who x --reason y -- pytest -q",
    "gozer status",
    "bash -c 'echo hi'",
    "rm -rf {run}/build",
    "cd /; ls",
    "cd / | cat; rm foo",
    "echo a#; ls",
    "ls # tt-smi -r and git push, inside a comment",
    "ls # a comment\nls",
    "ls # it's a comment with (parens), {braces} and more text",
    "git \\\n  status",
    "bash -c 'echo hi' arg0",
    "bash --norc -c 'echo hi'",
    "bash -lc 'echo hi'",
    "bash -c -- 'echo hi'",
    "bash -o pipefail -c 'echo hi'",
    "bash -c 'echo hi'  ; ls",
    "sh -ec 'echo hi'",
    "bash -O extglob -c 'echo hi'",
    "flock {run}/lock -c 'echo hi'",
    "flock -n {run}/lock echo hi",
    "flock --command 'echo hi' {run}/lock",
    "export FOO=1",
    "hf download org/m --include '*.safetensors'",
    "hf download org/m --include \"*.safetensors\"",
    "gh pr create --body 'a ? b'",
    "gh pr create --body \"a [b] c\"",
    "tt-model serve some/model --note 'x*y'",
    "hf download org/m --local-dir uploads",
    "huggingface-cli download org/m --local-dir uploads",
    "git commit -m 'costs $5' ",
    "git log --format='%h $x'",
    "rm {run}/sub/*.log",
    "rm -rf {run}/sub/old*",
    "echo x > /tmp/not-judged.txt",
    "cd {run}/sub && make",
    "mkdir {run}/n && cd {run}/n",
    "echo a; cd {run}/sub; rm x",
]

DENY = {
    "tt-model-push": [
        "setsid tt-model push x",
        "env -i tt-model push x",
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
    "tt-model-publish": [
        "tt-model publish episod/x",
    ],
    "git-push": [
        "ls # note\ngit push",
        "echo a#; git push",
        "bash -lc 'git push'",
        "sh -ec 'git push'",
        "bash --norc -c 'git push'",
        "env -i A=1 git push",
        "sudo -E git push",
        "sudo --user bob git push",
        "timeout --signal KILL 5 git push",
        "nice --adjustment 5 git push",
        "exec -a x git push",
        "command -p git push",
        "nohup git push &",
        "setsid git push",
        "g\\it push",
        "\"git\" push",
        "git \"push\"",
        "git \\\npush",
        "git -c alias.p=push p",
        "git subtree push --prefix x origin main",
        "/usr/lib/git-core/git-push origin",
        "ls | xargs git push",
        "git push # trailing comment",
        "cd /; bash -c 'git push'",
        "git push",
        "git -C /tmp/x push origin main",
        "git -c user.name=a push",
        "nice -n 5 git push",
        "/usr/bin/git push --force",
        "bash -c --norc 'git push'",
        "bash -c -x 'git push'",
        "bash -c -o pipefail 'git push'",
        "bash -c --posix 'git push'",
        "bash -ec -O extglob 'git push'",
        "bash -c -- 'git push'",
        "bash -c +x 'git push'",
        "bash -c --rcfile /dev/null 'git push'",
        "bash -c 'git push' arg0",
        "flock {run}/lock -c 'git push'",
        "flock -c 'git push' {run}/lock",
        "flock --command='git push' {run}/lock",
        "flock -n {run}/lock git push",
        "git --attr-source HEAD push",
        "git --git-dir x push",
        "git --paginate push",
        "git --exec-path push",
        "git --list-cmds=main push",
    ],
    "gh-repo-create": [
        "gh repo create foo --public",
        "gh --hostname x repo create foo",
    ],
    "hf-upload": [
        "hf upload-large-folder r d",
        "huggingface-cli upload-large-folder r d",
        "hf upload a b",
        "huggingface-cli upload a b",
        "hf --help upload a b",
    ],
    "tt-smi-reset": [
        "gozer run --chips 1 -- tt-smi -r",
        "tt-smi --res",
        "tt-smi --r",
        "tt-smi -sr",
        "gozer run --chips 1 --who a -- sudo tt-smi --reset",
        "tt-smi -r",
        "tt-smi -r 0,1",
        "tt-smi --reset",
        "sudo tt-smi -r",
        "tt-smi -r0",
        "bash -c --norc 'tt-smi -r'",
    ],
    "rm-outside-run-dir": [
        "false && cd {run}/sub; rm ../x",
        "bash -c 'rmdir {run}/sub'; cd {run}/sub; rm ../x",
        "cd /; bash -c 'rm -rf home'",
        "env -C / rm -rf home",
        "sudo -D / rm -rf home",
        "sudo --chdir=/ rm -rf home",
        "cd /; pushd {run}; popd; rm -rf home",
        "rm -rf -- /home",
        "rm -r /x --no-preserve-root",
        "cd {run}/missing; rm ../x",
        "cd / || cd {run}; rm -rf home",
        "cd {run}/sub && cd /; rm foo",
        "cd {run}/sub; cd ..; cd ..; rm x",
        "pushd {run}/sub; rm x",
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
        "flock {run}/lock -c 'cd /; rm -rf home'",
        "true && cd / && rm foo",
        "true || cd / ; rm foo",
        "cd {run}/sub && cd .. ; rm ../x",
        "rmdir {run}/sub; cd {run}/sub; rm ../x",
        "mkdir {run}/n; cd {run}/n; rm ../../x",
        "cd {run}/sub & rm ../x",
    ],
    "rm-ledger": [
        "rm {run}/ledger.jsonl",
        "rm -f {run}/ledger*",
    ],
    "rm-xargs": [
        "ls | xargs --max-args 1 rm -rf",
        "find x | xargs sh -c 'rm -rf x'",
        "find . -name x | xargs rm",
        "ls | xargs -n 1 rm -rf",
    ],
    "symlink-with-delete": [
        "ln -s / {run}/l; rm -rf {run}/l/home",
        "ln -sf / {run}/l && rm {run}/l/x",
        "ln --symbolic / {run}/l; rm {run}/l/x",
    ],
    "unsupported-syntax": [
        "bash -zc 'ls'",
        "bash -c -z 'ls'",
        "git ${X} push",
        "tt-model ${X}",
        "eval 'git push'",
        "echo 'git push' | bash",
        "echo a | sh",
        "bash <<< 'git push'",
        "cat <<EOF\nx\nEOF",
        "bash script.sh",
        "bash script.sh -c 'ls'",
        "sh",
        "{ git push; }",
        "function f { ls; }",
        "if true; then git push; fi",
        "for i in 1; do git push; done",
        "while true; do :; done",
        "case x in y) ls;; esac",
        "[[ -f x ]]",
        "! git push",
        "env -S 'git push'",
        "G=git; $G push",
        "git $X push",
        "tt-model p*",
        "rm -rf {/home,x}",
        "cd /tmp; (cd {run}); rm -rf foo",
        "(git push)",
        "find / -name x -exec rm -rf {} +",
        "find . -delete",
        "source x.sh",
        ". x.sh",
        "trap 'git push' EXIT",
        "exec >log",
        "coproc ls",
        "echo $'\\x67it'",
        "echo 'unterminated",
        "echo \"unterminated",
        "sudo --wat git push",
        "timeout --wat 5 ls",
        "gozer run --chips 1 git push",
        "GIT_CONFIG_COUNT=1 git status",
        "flock {run}/lock -x ls -c",
        "-c ls",
        "alias g=git",
        "shopt -s expand_aliases",
        "hash -p /bin/true git",
        "enable -n cd",
        "export GIT_CONFIG_COUNT=1",
        "declare -x BASH_ENV=/tmp/x",
        "typeset ENV=x",
        "BASH_ENV=/tmp/x bash -c 'ls'",
        "export BASH_ENV=/tmp/x",
        "readonly GIT_CONFIG_KEY_0=alias.p",
        "bash -c",
        "bash -c --wat 'ls'",
        "bash --wat -c 'ls'",
        "bash -z 'ls'",
        "bash -c --rcfile",
        "csh -c 'git push'",
        "fish -c 'git push'",
        "tcsh -c 'ls'",
        "ksh -c 'ls'",
        "git --wat push",
        "git --paginate --wat push",
        "git \"$X\" push",
        "tt-model 'a' p[u]sh",
        "[ -f x ]",
    ],
    "substitution": [
        "echo $(date)",
        "echo `date`",
        "diff <(ls) <(ls)",
    ],
    "rm-glob-in-run-root": [
        "rm {run}/*.log",
        "cd {run}; rm *.tmp",
        "rm -rf {run}/[a-z]*",
    ],
    "redirect-ledger": [
        "echo x > {run}/ledger.jsonl",
        "> {run}/ledger.jsonl",
        "echo x >> {run}/ledger.jsonl",
        "echo x &> {run}/ledger.jsonl",
        "cd {run}; echo x > ledger.jsonl",
        "echo x >| {run}/ledger.jsonl",
    ],
}


def _fill(cmd, run_dir):
    """Put the run directory in place of {run}. Not str.format, because several commands under test
    contain literal braces (brace expansion, `find -exec ... {}`)."""
    return cmd.replace("{run}", str(run_dir))


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "run"
    (d / "sub").mkdir(parents=True)
    return d


@pytest.mark.parametrize("cmd", ALLOW)
def test_allowed_commands_pass(cmd, run_dir):
    check_string(_fill(cmd, run_dir), run_dir)


@pytest.mark.parametrize(
    "rule,cmd", [(r, c) for r, cmds in DENY.items() for c in cmds])
def test_denied_commands_name_their_rule(rule, cmd, run_dir):
    with pytest.raises(Denied) as exc:
        check_string(_fill(cmd, run_dir), run_dir)
    assert exc.value.rule == rule


def test_every_rule_has_denied_examples():
    assert {name for name, _ in RULES} | {
        "substitution", "unsupported-syntax", "symlink-with-delete", "redirect-ledger"} == set(DENY)


def test_cd_follows_dotdot_textually_like_bash(run_dir):
    """`cd link/..` leaves the link's parent (bash's logical rule), not the link target's parent.
    With a link to a directory two levels down, the shell ends in run/, so `rm ../x` is outside."""
    (run_dir / "sub" / "deep").mkdir()
    (run_dir / "l").symlink_to(run_dir / "sub" / "deep")
    check_string(f"cd {run_dir}/l/..; rm x", run_dir)
    with pytest.raises(Denied) as exc:
        check_string(f"cd {run_dir}/l/..; rm ../x", run_dir)
    assert exc.value.rule == "rm-outside-run-dir"


@pytest.mark.parametrize("cmd,needle", [
    ("cat <<EOF\nx\nEOF", "heredoc"),
    ("echo {a,b}", "brace"),
    ("(ls)", "parenthes"),
    ("source venv/bin/activate", "shell keyword: source"),
    ("sudo --wat git push", "in: sudo --wat git push"),
    ("timeout --wat 5 ls", "in: timeout --wat 5 ls"),
    ("echo 'unterminated", "unbalanced quote"),
    ("echo $'x'", "$'"),
    ("bash script.sh", "script"),
    ("echo a | bash", "-c"),
    ("csh -c ls", "bash, sh, dash and zsh"),
    ("echo $(date)", "command substitution"),
    ("$X push", "variable"),
])
def test_unsupported_refusals_name_the_construct(cmd, needle, run_dir):
    with pytest.raises(Denied) as exc:
        check_string(cmd, run_dir)
    message = str(exc.value)
    assert needle in message, message
    assert "To proceed:" in message, message  # every refusal carries a rewrite hint


@pytest.mark.parametrize("cmd", [
    "rm {run}/a\x00b", "cd {run}/a\x00b", "rm \udc80x", "cd \udcff", "ls \x00",
])
def test_nul_and_lone_surrogates_are_refused_through_denied(cmd, run_dir):
    with pytest.raises(Denied) as exc:
        check_string(_fill(cmd, run_dir), run_dir)
    assert exc.value.rule == "unsupported-syntax"


def test_check_argv_refuses_nul_through_denied(run_dir):
    from orchard.runner import check_argv
    with pytest.raises(Denied):
        check_argv(["rm", str(run_dir) + "/a\x00b"], run_dir, str(run_dir))


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
