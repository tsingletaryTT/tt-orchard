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
    "tt-model --version",
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
    "gozer status",
    "gozer env",
    "gozer queue",
    "gozer history",
    "gozer --help",
    "gozer --version",
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
    "tt-model --help",
    "hf download org/m --local-dir uploads",
    "huggingface-cli download org/m --local-dir uploads",
    "git commit -m 'costs $5' ",
    "git log --format='%h $x'",
    "rm {run}/sub/*.log",
    "rm -rf {run}/sub/old*",
    "echo x > /tmp/not-judged.txt",
    "echo x > 'a$b.txt'",
    "cd {run}/sub && make",
    "mkdir {run}/n && cd {run}/n",
    "echo a; cd {run}/sub; rm x",
    "tt device list",
    "tt status",
    "docker ps",
    "docker logs coder",
    "docker images",
    "systemctl status docker",
    "ps aux",
    "rsync -a {run}/a {run}/b",
    "python3 -m huggingface_hub.commands.huggingface_cli download org/m",
    "curl -sL https://huggingface.co/api/models/x",
    "curl -s http://127.0.0.1:11434/v1/models",
    "curl -X GET http://127.0.0.1:8000/v1/models",
    "curl -sS -o {run}/f http://127.0.0.1:8000/v1/models",
    "wget -q -O {run}/f https://example.com/x",
    "wget --method=HEAD https://example.com/x",
    "printenv TT_VISIBLE_DEVICES",
    "pip list",
    "pip --version",
    "pip show torch",
    "pip freeze",
    "pip check",
    "python3 -m pip list",
    "pip download transformers",
    "python3 -m venv {run}/stages/1/venv",
    "uv venv {run}/stages/1/venv",
    "{run}/stages/1/venv/bin/pip install transformers",
    "{run}/stages/1/venv/bin/python -m pip install transformers",
    "{run}/stages/1/venv/bin/pip3 install -U transformers",
    "uv pip install --python {run}/stages/1/venv/bin/python transformers",
    "pip install --target {run}/stages/1/libs transformers",
    "pip install --target={run}/stages/1/libs transformers",
    "pip install -t {run}/stages/1/libs transformers",
    "conda list",
    "stages/1/venv/bin/pip install transformers",
    "python3 -m mymodule install",
    "true && venv/bin/pip install x",
    "python3 -m build --wheel install",
]

DENY = {
    "package-install": [
        "pip install transformers",
        "pip3 install --upgrade transformers",
        "pip3.12 install x",
        "pip uninstall -y transformers",
        "pip install --force-reinstall transformers==4.52.4 safetensors",
        "python3 -m pip install x",
        "python -m pip uninstall x",
        "python3.12 -m pip install --upgrade pip",
        "/home/someone/.tenstorrent-venv/bin/pip install x",
        "/home/someone/.tenstorrent-venv/bin/python3 -m pip install x",
        "uv pip install transformers",
        "uv pip install --python /home/someone/.tenstorrent-venv/bin/python x",
        "uv pip uninstall x",
        "pipx install x",
        "conda install numpy",
        "mamba install numpy",
        "micromamba remove x",
        "pip install --target /home/someone/lib x",
        "pip install --target {run}/../elsewhere x",
        "pip install -r requirements.txt",
        "cd {run}/sub && pip install x",
        "bash -c 'pip install x'",
        "ls; pip install x",
        "env A=1 pip install x",
        "sudo pip install x",
        "pip install --target {run}/a --target /tmp/b x",
        "cd $FOO; venv/bin/pip install x",
        "cd /; venv/bin/pip install x",
    ],
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
    "tt-model-package": [
        "tt-model package-thin episod/hemmingway-1-p300 --model-py model.py --out x",
        "tt-model package-thin --model-py model.py --out {run}/x",
        "tt-model package episod/x --wheels-dir w",
        "tt-model package --container tt-model.yaml",
        "env A=1 tt-model package-thin --out x",
        "bash -c 'tt-model package-thin --out x'",
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
        "huggingface-cli --token x upload a b",
        "hf --token x upload a b",
        "python3 -m huggingface_hub.commands.huggingface_cli upload x",
        "python3 -m huggingface_hub.cli.hf upload r d",
        "python -mhuggingface_hub.cli.hf upload-large-folder r d",
        "/usr/bin/python3.12 -m huggingface_hub upload x",
        "venv/bin/python -u -m huggingface_hub.commands.huggingface_cli upload x",
        "bash -c 'hf upload a b'",
        "ls | xargs hf upload",
        "env A=1 hf upload a b",
        "HF_HUB_OFFLINE=0 huggingface-cli upload a b",
        "hf upload-large-folder r d",
        "huggingface-cli upload-large-folder r d",
        "hf upload a b",
        "huggingface-cli upload a b",
        "hf --help upload a b",
    ],
    "tt-smi-reset": [
        "tt-smi --res",
        "tt-smi --r",
        "tt-smi -sr",
        "tt-smi -r",
        "tt-smi -r 0,1",
        "tt-smi --reset",
        "sudo tt-smi -r",
        "tt-smi -r0",
        "bash -c --norc 'tt-smi -r'",
    ],
    "gozer-write": [
        "gozer release L123 --force",
        "gozer reset L123 --force",
        "gozer acquire --chips 4",
        "gozer reconcile",
        "gozer wait t1",
        "gozer cancel t1",
        "gozer status --force",
        "gozer --force status",
        "gozer",
        "gozer frobnicate",
        "gozer release --help",
        "gozer run --chips 1 --who x --reason y -- pytest -q",
        "gozer run --chips 1 -- tt-smi -r",
        "gozer run --chips 1 --who a -- sudo tt-smi --reset",
        "gozer run --chips 1 git push",
        "/home/someone/.local/bin/gozer release L1",
        "GOZER_ROOT=/x gozer release L1",
        "env A=1 gozer acquire --chips 1",
        "sudo gozer release L1",
        "timeout 10 gozer reset L1",
        "ls | xargs gozer release",
        "bash -c 'gozer reset L1'",
        "gozer --json release L1",
    ],
    "tt-device-control": [
        "tt device reset",
        "tt device reset 0",
        "tt-cli device reset",
        "tt device --reset",
        "tt reset",
        "tt device reboot",
        "tt firmware flash fw.bundle",
        "tt stop",
        "tt serve some/model",
        "tt run some/model",
        "/usr/local/bin/tt device reset",
        "env A=1 tt device reset",
        "sudo tt device reset",
        "ls | xargs tt device reset",
        "bash -c 'tt device reset'",
    ],
    "docker-control": [
        "docker stop abc",
        "docker rm -f coder",
        "docker kill x",
        "docker run --rm img",
        "docker start x",
        "docker restart x",
        "docker push x",
        "docker exec -it x bash",
        "docker cp x:/a b",
        "docker container stop x",
        "docker -H unix:///var/run/docker.sock stop y",
        "/usr/bin/docker stop x",
        "DOCKER_HOST=x docker stop y",
        "sudo docker stop x",
        "ls | xargs docker rm",
        "bash -c 'docker kill x'",
    ],
    "tt-model-control": [
        "tt-model stop",
        "tt-model stop some/model --profile p",
        "tt-model serve some/model",
        "tt-model serve some/model --note 'x*y'",
        "tt-model run some/model",
        "tt-model rm some/model",
        "tt-model unpublish x",
        "tt-model login",
        "env A=1 tt-model serve x",
        "ls | xargs tt-model stop",
        "bash -c 'tt-model stop'",
    ],
    "process-kill": [
        "kill 1234",
        "kill -9 1234",
        "kill -s TERM 1",
        "pkill -f vllm",
        "killall python3",
        "systemctl stop docker",
        "systemctl restart docker",
        "systemctl --user stop x",
        "sudo systemctl stop docker",
        "/bin/kill 1",
        "env A=1 kill 1",
        "timeout 5 kill 1",
        "ls | xargs kill",
        "bash -c 'pkill -f vllm'",
        "reboot",
        "shutdown -h now",
    ],
    "remote-access": [
        "ssh host",
        "ssh -i k user@host ls",
        "scp a host:b",
        "sftp host",
        "rsync -a d host:/x",
        "rsync -a host:/x d",
        "rsync -a d user@host:x",
        "rsync rsync://host/m d",
        "/usr/bin/ssh host",
        "env A=1 scp a h:b",
        "ls | xargs ssh",
        "bash -c 'ssh host'",
    ],
    "http-write": [
        "curl -X POST https://x",
        "curl -XPOST https://x",
        "curl --request PUT https://x",
        "curl --request=DELETE https://x",
        "curl -sS -X PATCH https://x",
        "curl -d a=b https://x",
        "curl -dfoo https://x",
        "curl --data @f https://x",
        "curl --data-binary @f https://x",
        "curl --data-urlencode a=b https://x",
        "curl --json '{}' https://x",
        "curl -F f=@x https://x",
        "curl --form f=@x https://x",
        "curl -T f https://x",
        "curl -sT f https://x",
        "curl --upload-file f https://x",
        "curl -X",
        "wget --post-data=a https://x",
        "wget --post-file f https://x",
        "wget --method=PUT https://x",
        "wget --body-file=f --method PUT https://x",
        "env A=1 curl -X POST https://x",
        "ls | xargs curl -T",
        "bash -c 'curl -d a https://x'",
    ],
    "device-mask": [
        "TT_VISIBLE_DEVICES=0 python3 x.py",
        "TT_METAL_VISIBLE_DEVICES=0,1 pytest",
        "env TT_VISIBLE_DEVICES=0 python3 x.py",
        "env -u TT_VISIBLE_DEVICES python3 x.py",
        "env --unset=TT_METAL_VISIBLE_DEVICES python3 x.py",
        "export TT_VISIBLE_DEVICES=0",
        "declare -x TT_METAL_VISIBLE_DEVICES=0",
        "unset TT_VISIBLE_DEVICES",
        "bash -c 'TT_VISIBLE_DEVICES= python3 x.py'",
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
        "zsh -c 'ls'",
        "zsh script.sh",
        "git status\rgit push",
        "echo x >",
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
        # A target built from a variable or glob could be the ledger; the runner cannot tell.
        "echo x > $F",
        'echo x >> "$RUN/$NAME"',
        "echo x > {run}/l*",
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


@pytest.mark.parametrize("cmd", ["echo x > $F", 'echo x >> "$RUN/$NAME"', "echo x > {run}/l*"])
def test_dynamic_redirect_target_says_why(cmd, run_dir):
    with pytest.raises(Denied, match="built from a variable or glob, so the runner cannot tell "
                                     "whether it is the ledger") as exc:
        check_string(_fill(cmd, run_dir), run_dir)
    assert exc.value.rule == "redirect-ledger"


def test_carriage_return_is_refused_by_name(run_dir):
    # bash treats \r as part of a word; the lexer would treat it as a space, so the two disagree.
    with pytest.raises(Denied, match="a carriage return") as exc:
        check_string("git status\rgit push", run_dir)
    assert exc.value.rule == "unsupported-syntax"


@pytest.mark.parametrize("cmd", ["echo x >", "echo x >>", "echo x &>", "cat <"])
def test_redirect_at_end_of_text_is_refused_for_its_own_reason(cmd, run_dir):
    with pytest.raises(Denied, match="redirect with no target") as exc:
        check_string(cmd, run_dir)
    assert exc.value.rule == "unsupported-syntax"


def test_rm_after_losing_the_directory_says_so_and_how_to_proceed(run_dir):
    # `false && cd` may not have run, so the runner no longer knows where `rm ../x` points.
    with pytest.raises(Denied, match="lost track of the current directory") as exc:
        check_string(_fill("false && cd {run}/sub; rm ../x", run_dir), run_dir)
    assert exc.value.rule == "rm-outside-run-dir"
    assert "To proceed: use an absolute path inside the run directory" in str(exc.value)


def test_rm_outside_with_a_known_directory_does_not_claim_a_lost_one(run_dir):
    with pytest.raises(Denied) as exc:
        check_string(_fill("cd {run}/sub; rm ../../x", run_dir), run_dir)
    assert exc.value.rule == "rm-outside-run-dir"
    assert "lost track" not in str(exc.value)


def test_every_rule_has_denied_examples():
    assert {name for name, _ in RULES} | {
        "substitution", "unsupported-syntax", "symlink-with-delete", "redirect-ledger",
        "device-mask"} == set(DENY)


NEW_RULES = ("gozer-write", "tt-device-control", "docker-control", "tt-model-control",
             "process-kill", "remote-access", "http-write", "device-mask", "package-install")


@pytest.mark.parametrize("rule", NEW_RULES)
def test_hardware_and_remote_refusals_say_how_to_rewrite(rule, run_dir):
    # A model that is refused needs to know what to do instead, or it will try other spellings.
    with pytest.raises(Denied) as exc:
        check_string(_fill(DENY[rule][0], run_dir), run_dir)
    assert exc.value.rule == rule
    assert "To proceed:" in str(exc.value), str(exc.value)


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
    ("csh -c ls", "only bash, sh and dash command strings"),
    ("zsh -c ls", "shell zsh"),
    ("zsh script.sh", "shell zsh"),
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


def test_a_package_refusal_says_stage_7_packages_and_how_to_proceed(run_dir):
    # `tt-model package` and `package-thin` push when given a repo id, so agents never run them.
    with pytest.raises(Denied) as exc:
        check_string("tt-model package-thin --model-py model.py --out x", run_dir)
    assert exc.value.rule == "tt-model-package"
    assert "stage 7" in str(exc.value) and "To proceed:" in str(exc.value)


def test_a_venv_tool_reached_through_a_link_that_leaves_the_run_directory_is_refused(run_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere")
    (outside / "bin").mkdir()
    (run_dir / "link").symlink_to(outside)
    with pytest.raises(Denied) as exc:
        check_string(f"{run_dir}/link/bin/pip install x", run_dir)
    assert exc.value.rule == "package-install"


def test_a_directory_whose_name_starts_with_the_run_directorys_name_is_not_inside_it(run_dir):
    sibling = f"{run_dir}-evil"
    with pytest.raises(Denied) as exc:
        check_string(f"{sibling}/venv/bin/pip install x", run_dir)
    assert exc.value.rule == "package-install"


def test_the_run_directory_itself_is_not_a_target_inside_it(run_dir):
    with pytest.raises(Denied):
        check_string(f"pip install --target {run_dir} x", run_dir)
