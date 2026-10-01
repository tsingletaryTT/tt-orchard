"""The command runner: the only way the supervisor runs commands for a stage agent.

It refuses a fixed list of commands (see RULES): publishing, pushing, hand-run chip resets, and
deletion outside the run directory. The block sits here, where commands execute, so an ignored
prompt cannot bypass it.

WHAT THIS GUARDS, AND WHAT IT DOES NOT. The rules judge the commands they name, after stripping
wrappers (sudo, env, timeout, nice, nohup, time, xargs, ...), splitting compound commands at
`;`, `&&`, `||`, `|` and `&`, and looking inside `bash -c '...'`. They do not stop arbitrary
code (for example `python3 -c 'shutil.rmtree(...)'`) or deletion by other tools (`find -delete`).
Command substitution and process substitution are refused outright because the inner command
cannot be judged. Treat this as a guard against the named mistakes. The run's user account and the
lease tool remain the real limits.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass


class Denied(Exception):
    def __init__(self, rule: str, command: str):
        super().__init__(f"denied by rule {rule!r}: {command}")
        self.rule = rule
        self.command = command


SHELLS = {"bash", "sh", "zsh", "dash"}

# Commands that run another command. Options that take a value are listed so the value is not
# mistaken for the command (for example `sudo -u bob git push`).
WRAPPERS = {"sudo", "nohup", "time", "command", "exec", "env", "nice", "timeout", "stdbuf", "xargs"}
VALUE_OPTIONS = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-T", "-U"},
    "env": {"-u", "-C", "-S"},
    "nice": {"-n"},
    "timeout": {"-k", "-s"},
    "stdbuf": {"-i", "-o", "-e"},
    "xargs": {"-n", "-I", "-L", "-P", "-d", "-s", "-E", "-a"},
}
SEPARATOR_CHARS = set(";&|()")
DELETERS = {"rm", "rmdir", "unlink"}


@dataclass
class Ctx:
    run_dir: str          # real path of the run directory
    cwd: str | None       # directory the command runs in; None when a `cd` target was unknowable
    via_xargs: bool       # the command's operands come from a pipe we cannot see


# ---- parsing ------------------------------------------------------------------------------

def _simple_commands(text: str) -> list[list[str]]:
    """Split a shell string into simple commands at ; && || | & and newlines."""
    lexer = shlex.shlex(text.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    commands, current = [], []
    for tok in lexer:
        if tok and set(tok) <= SEPARATOR_CHARS:
            if current:
                commands.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        commands.append(current)
    return commands


def _is_assignment(tok: str) -> bool:
    name, eq, _ = tok.partition("=")
    return bool(eq) and name.isidentifier()


def _unwrap(argv: list[str]) -> tuple[list[str], bool]:
    """Strip leading assignments and wrapper commands. Returns (argv, saw_xargs)."""
    argv, saw_xargs = list(argv), False
    while argv:
        if _is_assignment(argv[0]):
            argv.pop(0)
            continue
        name = os.path.basename(argv[0])
        if name not in WRAPPERS:
            break
        argv.pop(0)
        saw_xargs = saw_xargs or name == "xargs"
        while argv and argv[0].startswith("-"):
            opt = argv.pop(0)
            if opt in VALUE_OPTIONS.get(name, ()) and argv:
                argv.pop(0)
        if name == "timeout" and argv:
            argv.pop(0)  # the duration operand
    return argv, saw_xargs


def _operands(argv: list[str]) -> list[str]:
    """Arguments that are not options. Everything after `--` is an operand."""
    out, after_dd = [], False
    for a in argv[1:]:
        if after_dd or not a.startswith("-"):
            out.append(a)
        elif a == "--":
            after_dd = True
    return out


def _git_subcommand(argv: list[str]) -> str | None:
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i] in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"} else 1
    return argv[i] if i < len(argv) else None


# ---- rules --------------------------------------------------------------------------------
# Each rule is (name, fn(name, argv, ctx) -> bool); True means "deny".

def _tt_model_push(name, argv, ctx):
    return name == "tt-model" and "push" in argv[1:]


def _tt_model_publish(name, argv, ctx):
    return name == "tt-model" and "publish" in argv[1:]


def _git_push(name, argv, ctx):
    return name == "git" and _git_subcommand(argv) == "push"


def _gh_repo_create(name, argv, ctx):
    rest = argv[1:]
    return name == "gh" and "repo" in rest and "create" in rest[rest.index("repo"):]


def _hf_upload(name, argv, ctx):
    return name in {"hf", "huggingface-cli"} and "upload" in argv[1:]


def _tt_smi_reset(name, argv, ctx):
    if name != "tt-smi":
        return False
    for a in argv[1:]:
        if a in {"-r", "--reset"} or a.startswith("--reset="):
            return True
        if a.startswith("--") and "reset" in a:
            return True
        if a.startswith("-r") and not a.startswith("--"):  # -r0, -r0,1
            return True
    return False


def _rm_outside_run_dir(name, argv, ctx):
    if name not in DELETERS:
        return False
    for a in _operands(argv):
        if "$" in a:
            return True  # unknown path
        path = os.path.expanduser(a)
        if not os.path.isabs(path):
            if ctx.cwd is None:
                return True  # relative path in an unknown directory
            path = os.path.join(ctx.cwd, path)
        if any(c in os.path.dirname(path) for c in "*?["):
            return True  # a glob in a directory part cannot be judged
        if any(c in os.path.basename(path) for c in "*?["):
            path = os.path.dirname(path)  # a glob is judged by the directory it expands in
        target = os.path.realpath(path)
        if target == ctx.run_dir or not target.startswith(ctx.run_dir + os.sep):
            return True
    return False


def _rm_ledger(name, argv, ctx):
    return name in DELETERS and any(
        os.path.basename(a).startswith("ledger") for a in _operands(argv))


def _rm_xargs(name, argv, ctx):
    return name in DELETERS and ctx.via_xargs


RULES = [
    ("tt-model-push", _tt_model_push),
    ("tt-model-publish", _tt_model_publish),
    ("git-push", _git_push),
    ("gh-repo-create", _gh_repo_create),
    ("hf-upload", _hf_upload),
    ("tt-smi-reset", _tt_smi_reset),
    # rm-ledger comes before rm-outside-run-dir on purpose: a glob such as `{run}/ledger*` expands in
    # the run directory itself, which the outside rule also refuses. The more specific name wins.
    ("rm-ledger", _rm_ledger),
    ("rm-outside-run-dir", _rm_outside_run_dir),
    ("rm-xargs", _rm_xargs),
]


# ---- checking and running -----------------------------------------------------------------

def check_argv(argv: list[str], run_dir, cwd: str | None) -> str | None:
    """Raise Denied if the command breaks a rule. Returns the directory after the command.

    `cwd` is the directory the command runs in, as a real path. None means the directory is
    unknowable (an earlier `cd $VAR`), and relative paths are then refused by the delete rules.
    """
    run_real = os.path.realpath(run_dir)
    unwrapped, via_xargs = _unwrap(argv)
    if not unwrapped:
        return cwd
    name = os.path.basename(unwrapped[0])
    if name in SHELLS and "-c" in unwrapped:
        i = unwrapped.index("-c")
        if i + 1 < len(unwrapped):
            check_string(unwrapped[i + 1], run_dir)
    ctx = Ctx(run_dir=run_real, cwd=cwd, via_xargs=via_xargs)
    for rule_name, deny in RULES:
        if deny(name, unwrapped, ctx):
            raise Denied(rule_name, " ".join(argv))
    if name in {"cd", "pushd"}:
        target = next((a for a in unwrapped[1:] if not a.startswith("-")), "~")
        if "$" in target or target == "-" or cwd is None:
            return None
        return os.path.realpath(os.path.join(cwd, os.path.expanduser(target)))
    return cwd


def check_string(text: str, run_dir) -> None:
    if any(marker in text for marker in ("$(", "`", "<(", ">(")):
        raise Denied("substitution", text)
    cwd: str | None = os.path.realpath(run_dir)  # every command starts in the run directory
    for argv in _simple_commands(text):
        cwd = check_argv(argv, run_dir, cwd)


def run_argv(argv: list[str], run_dir, **kw) -> subprocess.CompletedProcess:
    """Run one command, no shell. Refused commands never start."""
    check_argv(argv, run_dir, os.path.realpath(run_dir))
    return subprocess.run(argv, shell=False, cwd=run_dir, **kw)


def run_shell(text: str, run_dir, **kw) -> subprocess.CompletedProcess:
    """Run a shell string after judging every command in it. Nothing runs if any part is refused."""
    check_string(text, run_dir)
    return subprocess.run(["bash", "-c", text], cwd=run_dir, **kw)
