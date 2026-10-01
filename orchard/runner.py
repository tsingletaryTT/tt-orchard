"""The command runner: the only way the supervisor runs commands for a stage agent.

It refuses a fixed list of commands (see RULES): publishing, pushing, hand-run chip resets, and
deletion outside the run directory. The block sits here, where commands execute, so an ignored
prompt cannot bypass it.

HOW IT DECIDES. The runner understands a small subset of shell and refuses everything outside it
(rule name "unsupported-syntax"). A command it cannot read is refused. It does not guess.

  Understood: simple commands joined by `;`, `&&`, `||`, `|`, `&` and newlines; quoted and
  escaped words; `#` comments (only where a word starts, and only to the end of that line);
  backslash-newline continuations; simple redirects (`>`, `>>`, `<`, `2>&1`, `&>`); `cd`;
  leading `NAME=value` assignments; `${NAME}` inside a word; `bash -c '...'` (the string is
  checked in turn, in the current directory); and these wrapper commands, which are stripped
  before the rules look at the real command: sudo, env, nohup, time, command, builtin, exec,
  nice, timeout, stdbuf, xargs, setsid, ionice, taskset, chrt, flock and `gozer run ... --`.

  Refused as unsupported: shell keywords and builtins that run commands (if, for, while, case,
  function, eval, source, `.`, trap, `!`, `[[`, coproc, exec with no command), subshell
  parentheses, braces, heredocs and here-strings, `$'...'` quoting, unbalanced quotes, a command
  name built from a variable or a glob, a shell with no `-c` string (so piping into a shell is
  refused), a wrapper given an option the runner does not know, `env -S`, `find -exec` and
  `find -delete`, and `GIT_CONFIG*` assignments. Command substitution and process substitution
  (`$(...)`, backticks, `<(...)`) are refused as "substitution" because the inner command
  cannot be judged.

  The rules then judge the real command: tt-model push/publish, git push (including `-c alias.*`,
  `subtree push`, and `git-push`), gh repo create, hf/huggingface-cli upload*, any tt-smi
  reset spelling (long-option prefixes and clustered short flags included), and rm/rmdir/unlink
  outside the run directory, of a ledger file, or fed by xargs. A string that makes a symbolic
  link and also deletes something is refused ("symlink-with-delete"), because the delete could
  go through the link.

WHAT THIS DOES NOT COVER. Splitting a shell string by tokenizing is best effort. It is not a shell
parser, and a way around it may exist. It does not stop arbitrary code (for example
`python3 -c 'shutil.rmtree(...)'` or `perl -e`), deletion by tools it does not name (`mv`, `rsync
--delete`, `truncate`), wrappers it does not list (`watch`, `su`, `doas`, `ssh host cmd`), scripts
that run a denied command inside them, git aliases or hooks defined in config files, or a `cd`
that depends on CDPATH. Tools named in the rules are judged only by the words the rules look at.
The real limits are the user account the run uses, the lease tool (gozer), and not giving the
agent credentials to publish or push in the first place. Treat this as a guard against the named
mistakes, not as a sandbox.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


class Denied(Exception):
    def __init__(self, rule: str, command: str):
        super().__init__(f"denied by rule {rule!r}: {command}")
        self.rule = rule
        self.command = command


def _unsupported(text) -> Denied:
    """The one way every 'cannot read this safely' refusal is raised."""
    return Denied("unsupported-syntax", text if isinstance(text, str) else " ".join(text))


SHELLS = {"bash", "sh", "zsh", "dash", "ksh", "csh", "tcsh", "fish"}

# Words that, in command position, run other commands or change how the line is parsed. Wrappers
# that are understood (time, command, exec, ...) are not here; they are in WRAPPERS.
KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "select", "function", "{", "}", "!", "[[", "eval", "source", ".", "trap", "coproc",
}

DELETERS = {"rm", "rmdir", "unlink"}


@dataclass
class WrapperSpec:
    """Options a wrapper accepts. Anything else makes the runner refuse (fail closed): guessing
    which token is the wrapped command is how `sudo --user bob git push` slipped past."""
    flags: str = ""                  # short flags with no value
    values: str = ""                 # short options that take a value
    long_flags: tuple = ()           # long options with no value (a `=value` tail is tolerated)
    long_values: tuple = ()          # long options that take a value
    positional: int = 0              # operands before the command (timeout's duration, flock's file)
    numeric: bool = False            # accepts `-5` style (nice)
    chdir: tuple = ()                # options whose value is the directory the command runs in


WRAPPERS = {
    "sudo": WrapperSpec(
        flags="EHnSbkKP", values="ughpCDrtTU",
        long_flags=("--preserve-env", "--set-home", "--non-interactive", "--stdin", "--background"),
        long_values=("--user", "--group", "--host", "--prompt", "--close-from", "--chdir", "--role",
                     "--type", "--command-timeout", "--other-user"),
        chdir=("-D", "--chdir")),
    "env": WrapperSpec(
        flags="i0v", values="uC",
        long_flags=("--ignore-environment", "--null", "--debug"),
        long_values=("--unset", "--chdir"), chdir=("-C", "--chdir")),
    "nohup": WrapperSpec(),
    "time": WrapperSpec(flags="pav", values="fo", long_flags=("--portability", "--append", "--verbose"),
                        long_values=("--format", "--output")),
    "command": WrapperSpec(flags="p"),
    "builtin": WrapperSpec(),
    "exec": WrapperSpec(flags="cl", values="a"),
    "nice": WrapperSpec(values="n", long_values=("--adjustment",), numeric=True),
    "timeout": WrapperSpec(flags="v", values="ks",
                           long_flags=("--foreground", "--preserve-status", "--verbose"),
                           long_values=("--kill-after", "--signal"), positional=1),
    "stdbuf": WrapperSpec(values="ioe", long_values=("--input", "--output", "--error")),
    "xargs": WrapperSpec(
        flags="0rtpxo", values="nILPdsEa",
        long_flags=("--null", "--no-run-if-empty", "--verbose", "--interactive", "--exit",
                    "--open-tty", "--replace", "--eof"),
        long_values=("--max-args", "--max-lines", "--max-procs", "--delimiter", "--max-chars",
                     "--arg-file", "--process-slot-var")),
    "setsid": WrapperSpec(flags="cfw", long_flags=("--ctty", "--fork", "--wait")),
    "ionice": WrapperSpec(flags="t", values="cn", long_flags=("--ignore",),
                          long_values=("--class", "--classdata")),
    "taskset": WrapperSpec(flags="ac", positional=1),
    "chrt": WrapperSpec(flags="bfiorRv", positional=1),
    "flock": WrapperSpec(flags="nsxuo", values="wE",
                         long_flags=("--nonblock", "--shared", "--exclusive", "--unlock"),
                         long_values=("--timeout", "--conflict-exit-code"), positional=1),
}

# Tools the rules read by position. A `$` or glob in their words could turn into a different
# subcommand at run time (`git $X push`, or `tt-model p*` with a file named push), so such words
# are refused rather than judged.
NAMED_TOOLS = {"git", "gh", "hf", "huggingface-cli", "tt-model", "tt-smi"}
GIT_VALUE_OPTIONS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env"}


@dataclass
class Ctx:
    run_dir: str          # real path of the run directory
    cwd: str | None       # real directory the command runs in; None when it is unknowable
    via_xargs: bool       # the command's operands come from a pipe we cannot see


@dataclass
class _State:
    """What one whole checked string has done so far, for rules that span commands."""
    made_symlink: bool = False
    deletes: bool = False


# ---- lexing -------------------------------------------------------------------------------
# A small hand-written lexer, not shlex: shlex cannot report whether a character was quoted, treats
# `#` as a comment anywhere in the line, and reads `;` glued to a word differently from bash.

def _lex(text: str) -> list[tuple[list[str], str | None, str | None]]:
    """Split a shell string into simple commands.

    Returns (words, operator_before, operator_after) per command, operators being one of
    ';' '&&' '||' '|' '&' or None at the ends. Redirections are dropped from the words.
    Raises Denied('unsupported-syntax') for anything outside the supported subset.
    """
    n = len(text)
    i = 0
    cur: list[str] = []
    in_word = False
    quoted = False          # the word so far contained a quote or escape (so `2` is not an fd)
    expect_target = False   # the previous token was a redirect, so the next word is its target
    words: list[str] = []
    raw: list[tuple[list[str], str | None]] = []

    def end_word():
        nonlocal cur, in_word, quoted, expect_target
        if in_word:
            word = "".join(cur)
            if expect_target:
                expect_target = False   # a redirect target is a file name, not part of the command
            else:
                words.append(word)
        cur, in_word, quoted = [], False, False

    def end_command(op):
        nonlocal words
        end_word()
        if expect_target:
            raise _unsupported(text)    # a redirect with nothing after it
        if words:
            raw.append((words, op))
        words = []

    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "\\":
            if nxt == "\n":             # line continuation: joins the lines, even mid-word
                i += 2
                continue
            if nxt:
                cur.append(nxt)
                in_word = quoted = True
                i += 2
            else:
                cur.append("\\")
                in_word = True
                i += 1
        elif c in " \t\r":
            end_word()
            i += 1
        elif c == "\n":
            end_command(";")
            i += 1
        elif c == "#" and not in_word:  # a comment runs to the end of THIS line only
            while i < n and text[i] != "\n":
                i += 1
        elif c == "'":
            j = text.find("'", i + 1)
            if j < 0:
                raise _unsupported(text)
            cur.append(text[i + 1:j])
            in_word = quoted = True
            i = j + 1
        elif c == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    if text[j + 1] == "\n":
                        j += 2
                        continue
                    if text[j + 1] in '"\\$`':
                        buf.append(text[j + 1])
                        j += 2
                        continue
                buf.append(text[j])
                j += 1
            if j >= n:
                raise _unsupported(text)
            cur.append("".join(buf))
            in_word = quoted = True
            i = j + 1
        elif c == "$" and nxt in ("'", '"'):
            raise _unsupported(text)    # $'\x67it' spells words the lexer cannot see
        elif c == "$" and nxt == "{":   # ${NAME}: keep it whole so its braces are not "brace expansion"
            j = text.find("}", i)
            if j < 0:
                raise _unsupported(text)
            cur.append(text[i:j + 1])
            in_word = True
            i = j + 1
        elif c in "(){}":
            raise _unsupported(text)    # subshells, groups, brace expansion
        elif c == ";":
            end_command(";")
            i += 1
        elif c == "&":
            if nxt == ">":              # &> and &>> redirect both streams
                end_word()
                i += 3 if text[i + 2:i + 3] == ">" else 2
                expect_target = True
            elif nxt == "&":
                end_command("&&")
                i += 2
            else:
                end_command("&")
                i += 1
        elif c == "|":
            if nxt == "|":
                end_command("||")
                i += 2
            elif nxt == "&":
                end_command("|")
                i += 2
            else:
                end_command("|")
                i += 1
        elif c in "<>":
            if in_word and not quoted and "".join(cur).isdigit():
                cur, in_word = [], False    # `2>&1`: the digit names a file descriptor
            else:
                end_word()
            if c == "<" and nxt == "<":
                raise _unsupported(text)    # heredoc or here-string
            i += 2 if (nxt in "&|" or (c == ">" and nxt == ">") or (c == "<" and nxt == ">")) else 1
            expect_target = True
        else:
            cur.append(c)
            in_word = True
            i += 1
    end_command(None)

    # Attach the operator before each command (it is the previous command's trailing operator).
    out = []
    before = None
    for words_, after in raw:
        out.append((words_, before, after))
        before = after
    return out


def _is_assignment(tok: str) -> bool:
    name, eq, _ = tok.partition("=")
    return bool(eq) and name.isidentifier()


# ---- wrappers -----------------------------------------------------------------------------

def _chdir_value(cwd, value):
    """The directory a wrapper's chdir option leads to, as a real path, or None when unknowable."""
    if cwd is None or any(ch in value for ch in "$*?[`"):
        return None
    return os.path.realpath(os.path.join(os.path.realpath(cwd), os.path.expanduser(value)))


def _consume_option(argv, spec, opt):
    """Consume one wrapper option (already popped from argv). Returns (key, value) for options
    that carry a value, else (None, None). Raises on an option the spec does not list."""
    if opt.startswith("--"):
        key, eq, val = opt.partition("=")
        if key in spec.long_values:
            if not eq:
                if not argv:
                    raise _unsupported(opt)
                val = argv.pop(0)
            return key, val
        if key in spec.long_flags:
            return None, None
        raise _unsupported(opt)
    body = opt[1:]
    if spec.numeric and body.isdigit():
        return None, None
    k = 0
    while k < len(body):
        if body[k] in spec.values:
            val = body[k + 1:]
            if not val:
                if not argv:
                    raise _unsupported(opt)
                val = argv.pop(0)
            return "-" + body[k], val
        if body[k] in spec.flags:
            k += 1
        else:
            raise _unsupported(opt)
    return None, None


def _unwrap(argv, cwd):
    """Strip leading assignments and wrapper commands.

    Returns (argv, saw_xargs, cwd, wrappers_used). `cwd` follows chdir options (`env -C DIR`),
    and becomes None when the directory cannot be worked out.
    """
    argv = list(argv)
    saw_xargs = False
    used: list[str] = []
    while argv:
        if _is_assignment(argv[0]):
            if argv[0].startswith("GIT_CONFIG"):
                raise _unsupported(argv)   # can define a git alias that hides a push
            argv.pop(0)
            continue
        name = os.path.basename(argv[0])
        if name == "gozer" and len(argv) > 1 and argv[1] == "run":
            # `gozer run <lease options> -- COMMAND`: everything before `--` is the lease's own.
            if "--" not in argv:
                raise _unsupported(argv)
            argv = argv[argv.index("--") + 1:]
            used.append(name)
            continue
        spec = WRAPPERS.get(name)
        if spec is None:
            break
        argv.pop(0)
        used.append(name)
        saw_xargs = saw_xargs or name == "xargs"
        while argv and argv[0].startswith("-") and argv[0] != "-":
            opt = argv.pop(0)
            if opt == "--":
                break
            key, val = _consume_option(argv, spec, opt)
            if key in spec.chdir:
                cwd = _chdir_value(cwd, val)
        for _ in range(spec.positional):
            if argv:
                argv.pop(0)
    return argv, saw_xargs, cwd, used


def _operands(argv: list[str]) -> list[str]:
    """Arguments that are not options. Everything after `--` is an operand."""
    out, after_dd = [], False
    for a in argv[1:]:
        if after_dd or not a.startswith("-"):
            out.append(a)
        elif a == "--":
            after_dd = True
    return out


def _git_subcommand_index(argv: list[str]) -> int:
    """Index of git's real subcommand, after any global options (and their values)."""
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        i += 2 if argv[i] in GIT_VALUE_OPTIONS else 1
    return i


# ---- rules --------------------------------------------------------------------------------
# Each rule is (name, fn(name, argv, ctx) -> bool); True means "deny".

def _tt_model_push(name, argv, ctx):
    return name == "tt-model" and "push" in argv[1:]


def _tt_model_publish(name, argv, ctx):
    return name == "tt-model" and "publish" in argv[1:]


def _git_push(name, argv, ctx):
    if name in {"git-push", "git-send-pack"}:      # the helper binary run directly
        return True
    if name != "git":
        return False
    i = _git_subcommand_index(argv)
    # An alias defined on the command line can be `push` under another name, so refuse them all.
    if any(a.lower().startswith(("alias.", "--config-env=alias.")) for a in argv[1:i]):
        return True
    sub = argv[i] if i < len(argv) else None
    if sub in {"push", "send-pack"}:
        return True
    return sub in {"subtree", "lfs"} and "push" in argv[i + 1:]


def _gh_repo_create(name, argv, ctx):
    rest = argv[1:]
    return name == "gh" and "repo" in rest and "create" in rest[rest.index("repo"):]


def _hf_upload(name, argv, ctx):
    # upload, upload-large-folder, and any later upload-* subcommand
    return name in {"hf", "huggingface-cli"} and any(a.startswith("upload") for a in argv[1:])


def _tt_smi_reset(name, argv, ctx):
    if name != "tt-smi":
        return False
    for a in argv[1:]:
        if a.startswith("--"):
            key = a[2:].split("=", 1)[0]
            # argparse accepts any prefix of a long option, so --r, --re and --res all mean --reset
            if key and ("reset".startswith(key) or "reset" in key):
                return True
        elif a.startswith("-") and len(a) > 1 and "r" in a[1:]:
            return True   # -r, -r0, and clusters such as -sr
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

def _shell_command_string(unwrapped: list[str]) -> str:
    """The `-c` string of a shell invocation. Raises Denied if the shell is not given one.

    `-c` may stand alone or sit inside a cluster (`-lc`, `-ec`), after other options such as
    `--norc`. A shell with no command string reads stdin or a script, neither of which can be judged.
    """
    args = unwrapped[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a in {"-o", "+o", "-O", "+O", "--rcfile", "--init-file"}:
            i += 2                       # these take a value; it is not a script name
        elif a.startswith("--") or (a[:1] in "-+" and len(a) > 1):
            if a[:1] == "-" and not a.startswith("--") and "c" in a[1:]:
                if i + 1 < len(args):
                    return args[i + 1]
                raise _unsupported(unwrapped)
            i += 1
        else:
            raise _unsupported(unwrapped)   # a script path
    raise _unsupported(unwrapped)           # no -c: reads stdin


def _cd_target(args, cwd, after_or):
    """Where `cd ARGS` leaves the shell, as a path, or None when it cannot be known."""
    physical = "-P" in args
    operands = [a for a in args if not a.startswith("-") or a == "-"]
    target = operands[0] if operands else "~"
    if len(operands) > 1 or target == "-" or any(ch in target for ch in "$*?[`"):
        return None
    # `cd A || cd B`: B runs only if A failed, so which directory we are in is not known.
    if after_or:
        return None
    path = os.path.expanduser(target)
    if not os.path.isabs(path):
        if cwd is None:
            return None
        path = os.path.join(cwd, path)
    # bash follows `..` textually unless -P is given, so `cd link/..` leaves the link's parent.
    cand = os.path.realpath(path) if physical else os.path.normpath(path)
    # A directory that does not exist yet means the cd would fail and leave the shell where it was.
    return cand if os.path.isdir(cand) else None


def _check_argv(argv, run_dir, cwd, st, via_xargs_in=False, after_or=False):
    run_real = os.path.realpath(run_dir)
    unwrapped, via_xargs, eff_cwd, used = _unwrap(argv, cwd)
    via_xargs = via_xargs or via_xargs_in
    if not unwrapped:
        if "exec" in used:
            raise _unsupported(argv)    # `exec >file` rewires the shell itself
        return cwd
    first = unwrapped[0]
    name = os.path.basename(first)
    if first in KEYWORDS:
        raise _unsupported(argv)
    if "$" in first or any(ch in first for ch in "*?"):
        raise _unsupported(argv)        # the command's name is not known until run time

    if name in SHELLS:
        string = _shell_command_string(unwrapped)
        # Recurse in the CURRENT directory; a cd inside it does not leak out to the caller.
        _check_text(string, run_dir, eff_cwd, st, via_xargs)
        return cwd

    if name == "find" and any(a in {"-exec", "-execdir", "-ok", "-okdir", "-delete"} for a in unwrapped):
        raise _unsupported(argv)        # runs or deletes things the rules never see

    if name in NAMED_TOOLS:
        words = unwrapped[:_git_subcommand_index(unwrapped) + 1] if name == "git" else unwrapped
        if any(ch in w for w in words for ch in "$*?["):
            raise _unsupported(argv)

    if name == "ln" and any(a == "--symbolic" or (a.startswith("-") and not a.startswith("--") and "s" in a)
                            for a in unwrapped[1:]):
        st.made_symlink = True
    if name in DELETERS:
        st.deletes = True

    ctx = Ctx(run_dir=run_real,
              cwd=os.path.realpath(eff_cwd) if eff_cwd is not None else None,
              via_xargs=via_xargs)
    for rule_name, deny in RULES:
        if deny(name, unwrapped, ctx):
            raise Denied(rule_name, " ".join(argv))

    if name in {"pushd", "popd", "dirs"} or (name == "cd" and used):
        return None                     # directory stack, or cd run through a wrapper: unknowable
    if name == "cd":
        return _cd_target(unwrapped[1:], cwd, after_or)
    return cwd


def _check_text(text, run_dir, cwd, st, via_xargs=False):
    """Check every command in a shell string; returns the directory after it."""
    if any(marker in text for marker in ("$(", "`", "<(", ">(")):
        raise Denied("substitution", text)
    for words, before, after in _lex(text):
        new_cwd = _check_argv(words, run_dir, cwd, st, via_xargs, after_or=(before == "||"))
        # In a pipeline or in the background the command runs in a subshell, so a `cd` there does
        # not change the directory the following commands run in.
        if before != "|" and after not in {"|", "&"}:
            cwd = new_cwd
    return cwd


def _finish(st: _State, text: str) -> None:
    if st.made_symlink and st.deletes:
        raise Denied("symlink-with-delete", text)


def check_argv(argv: list[str], run_dir, cwd: str | None) -> str | None:
    """Raise Denied if the command breaks a rule. Returns the directory after the command.

    `cwd` is the directory the command runs in. None means it is unknowable (an earlier
    `cd $VAR`), and relative paths are then refused by the delete rules.
    """
    st = _State()
    result = _check_argv(argv, run_dir, cwd, st)
    _finish(st, " ".join(argv))
    return result


def check_string(text: str, run_dir) -> None:
    st = _State()
    # every command starts in the run directory
    _check_text(text, run_dir, os.path.realpath(run_dir), st)
    _finish(st, text)


def run_argv(argv: list[str], run_dir, **kw) -> subprocess.CompletedProcess:
    """Run one command, no shell. Refused commands never start."""
    check_argv(argv, run_dir, os.path.realpath(run_dir))
    return subprocess.run(argv, shell=False, cwd=run_dir, **kw)


def run_shell(text: str, run_dir, **kw) -> subprocess.CompletedProcess:
    """Run a shell string after judging every command in it. Nothing runs if any part is refused."""
    check_string(text, run_dir)
    return subprocess.run(["bash", "-c", text], cwd=run_dir, **kw)
