"""The command runner: the only way the supervisor runs commands for a stage agent.

It refuses a fixed list of commands (see RULES): publishing, pushing, hand-run chip resets, and
deletion outside the run directory. The block sits here, where commands execute, so an ignored
prompt cannot bypass it.

HOW IT DECIDES. The runner understands a small subset of shell and refuses everything outside it
(rule name "unsupported-syntax"). A command it cannot read is refused. It does not guess. Each
refusal names the construct and says how to rewrite the command.

  Understood: simple commands joined by `;`, `&&`, `||`, `|`, `&` and newlines; quoted and
  escaped words; `#` comments (only where a word starts, and only to the end of that line);
  backslash-newline continuations; simple redirects (`>`, `>>`, `<`, `2>&1`, `&>`); `cd`;
  leading `NAME=value` assignments; `${NAME}` inside a word; and the wrapper commands sudo, env,
  nohup, time, command, builtin, exec, nice, timeout, stdbuf, xargs, setsid, ionice, taskset,
  chrt, flock and `gozer run ... --`, which are stripped before the rules look at the real
  command. A wrapper given an option it does not know is refused.

  Shells: bash, sh and dash are accepted only with a command string (`-c`, alone or in a
  cluster such as `-lc`, with bash's option grammar parsed so the string the runner checks is
  the string bash runs). The string, and flock's `-c`/`--command` string, are checked in turn in
  the current directory. A shell with no command string (stdin, a script path) is refused, so
  piping into a shell is refused. csh, tcsh, fish, ksh and zsh are refused: they are not lexed here, and zsh
  has expansions that bash lacks.

  Refused as unsupported: shell keywords and builtins that run commands or change how later
  commands resolve (if, for, while, case, function, eval, source, `.`, trap, `!`, `[[`, coproc,
  alias, unalias, shopt, hash, enable, and export/declare/typeset/readonly/local of GIT_CONFIG*,
  BASH_ENV or ENV), `exec` with no command, subshell parentheses, braces, heredocs and
  here-strings, `$'...'` quoting, unbalanced quotes, NUL bytes, carriage returns and invalid characters, a command
  name built from a variable or a glob or starting with `-`, `env -S`, `find -exec` and
  `find -delete`, and `GIT_CONFIG*` or `BASH_ENV` assignments. Command substitution and process
  substitution (`$(...)`, backticks, `<(...)`) are refused as "substitution".

  The rules then judge the real command: tt-model push/publish; git push (including `-c alias.*`,
  `subtree push`, and `git-push`); gh repo create; hf/huggingface-cli upload* as the subcommand;
  any tt-smi reset spelling (long-option prefixes and clustered short flags included); and
  rm/rmdir/unlink outside the run directory, of a ledger file, with a glob directly in the run
  directory root, or fed by xargs. An output redirect whose target name starts with "ledger"
  is refused (it would truncate the ledger), and so is one whose target has an unquoted glob or a
  `$` outside single quotes (the runner cannot tell whether that name is the ledger). A string that makes a symbolic link and also deletes something
  is refused ("symlink-with-delete"), because the delete could go through the link. In the
  words of git, gh, hf, huggingface-cli, tt-model and tt-smi, an unquoted glob character or a `$`
  outside single quotes is refused, because it could change which subcommand runs.

  Directory tracking: `cd` is believed only when it is the first thing that moves the directory
  in its position: not after `&&`, `||`, `|` or `&`, not when its target is not a plain literal,
  not when the directory does not exist at check time, and not when an earlier command in the
  string is anything but a short list of read-only ones (echo, ls, ...), since an earlier command
  could create or remove the directory. Otherwise the directory is unknown and relative
  deletes are refused. A `cd` inside a pipeline or the background does not count.

WHAT THIS DOES NOT COVER. Splitting a shell string by tokenizing is best effort. It is not a shell
parser, and a way around it may exist. Redirects are not judged except for ledger targets: a
redirect can write or truncate any other file, and stage agents need to write outside the run
directory. The runner does not stop arbitrary code (for example `python3 -c 'shutil.rmtree(...)'`
or `perl -e`), deletion by tools it does not name (`mv`, `rsync --delete`, `truncate`), wrappers
it does not list (`watch`, `su`, `doas`, `ssh host cmd`), scripts that run a denied command inside
them, git aliases or hooks defined in config files, writes to the ledger by tools other than
redirects and rm (`tee`, `cp`), or a `cd` that depends on CDPATH. hf and huggingface-cli are
judged only at the subcommand position, so a value-taking option placed before the subcommand can
hide it. The real limits are the user account the run uses, the lease tool (gozer), and not giving
the agent credentials to publish or push in the first place. Treat this as a guard against the
named mistakes, not as a sandbox.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


class Denied(Exception):
    def __init__(self, rule: str, command: str, detail: str = ""):
        text = f"denied by rule {rule!r}: {detail} (in: {command})" if detail \
            else f"denied by rule {rule!r}: {command}"
        super().__init__(text)
        self.rule = rule
        self.command = command
        self.detail = detail


DEFAULT_HINT = "run the pieces as separate commands"


def _unsupported(command, construct: str, hint: str = DEFAULT_HINT) -> Denied:
    """The one way every 'cannot read this safely' refusal is raised. `construct` names what was
    found and `hint` says how to say it in a form the runner understands."""
    if not isinstance(command, str):
        command = " ".join(command)
    return Denied("unsupported-syntax", command, f"{construct}. To proceed: {hint}")


# Shells whose `-c` strings are lexed with the rules below. Other shells have different quoting and
# grammar, so they are refused rather than misread. zsh is refused too: it has expansions bash lacks
# (for example `=cmd`, `**/` and `(#q)` globs), so the words the lexer sees can differ from the
# words zsh runs.
SHELLS = {"bash", "sh", "dash"}
REFUSED_SHELLS = {"csh", "tcsh", "fish", "ksh", "zsh"}

# Words that, in command position, run other commands or change how the line is parsed or how later
# commands resolve. Wrappers that are understood (time, command, exec, ...) are in WRAPPERS.
KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
    "select", "function", "{", "}", "!", "[[", "eval", "source", ".", "trap", "coproc",
    "alias", "unalias", "shopt", "hash", "enable",
}
KEYWORD_HINT = ("run the pieces as separate commands; to use a virtualenv call venv/bin/python "
                "directly instead of sourcing an activate script; write the text to a file "
                "and pass the path instead of eval or a heredoc")
# Builtins that set variables; refused only when they set one that changes how commands resolve.
ENV_SETTERS = {"export", "declare", "typeset", "readonly", "local"}
DANGEROUS_ENV = {"BASH_ENV", "ENV"}

DELETERS = {"rm", "rmdir", "unlink"}

# Commands that cannot create, remove or relink a directory, so a `cd` that follows them can still
# trust the filesystem as it was when the string was checked.
BENIGN = {"cd", "echo", "ls", "cat", "grep", "pwd", "true", "false", "printf", "sleep", "head", "tail",
          "wc", "sort", "uniq", "diff", "test"}


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
    shell_opts: tuple = ()           # options whose value is a command string handed to a shell


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
    # `flock FILE -c COMMAND` and `flock -c COMMAND FILE` hand COMMAND to a shell.
    "flock": WrapperSpec(flags="nsxuo", values="wEc",
                         long_flags=("--nonblock", "--shared", "--exclusive", "--unlock"),
                         long_values=("--timeout", "--conflict-exit-code", "--command"),
                         positional=1, shell_opts=("-c", "--command")),
}

# Tools the rules read by position. An unquoted glob or a `$` in their words could turn into a
# different subcommand at run time (`git $X push`, or `tt-model p*` with a file named push), so such
# words are refused rather than judged. Quoted globs are literal and are fine.
NAMED_TOOLS = {"git", "gh", "hf", "huggingface-cli", "tt-model", "tt-smi"}

# git's global options (checked against `man git`, git 2.43). Options that take a separate value:
GIT_VALUE_OPTIONS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env",
                     "--attr-source"}
# Global options with no value. A `--long` option that is in neither set and has no `=` is refused:
# the runner cannot tell whether the next word is its value or the subcommand.
GIT_FLAG_OPTIONS = {"--version", "--help", "--exec-path", "--html-path", "--man-path", "--info-path",
                    "--paginate", "--no-pager", "--bare", "--no-replace-objects",
                    "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
                    "--icase-pathspecs", "--no-optional-locks"}

# bash's option grammar, enough to find the command string it will run.
SHELL_SHORT_FLAGS = set("abefhkmnptuvxBCEHPTlriDc")   # -c is among them
SHELL_SHORT_VALUES = set("oO")                        # `-o NAME`, `-O NAME`, `+o`, `+O` take a value
SHELL_LONG_FLAGS = {"--norc", "--noprofile", "--posix", "--login", "--restricted", "--verbose",
                    "--noediting", "--debugger", "--dump-strings", "--dump-po-strings"}
SHELL_LONG_VALUES = {"--rcfile", "--init-file"}


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
    fs_stable: bool = True    # only read-only commands have run, so the filesystem is as first seen


class _Word(str):
    """A word after quote removal that remembers whether the shell could expand it.

    `dynamic` is True when the word has an unquoted glob character or a `$` outside single quotes:
    the text the rules see may then differ from what runs."""
    dynamic: bool = False


def _dynamic(word, plain_default: bool) -> bool:
    """Could this word change at run time? Words from the lexer know; a plain str (from a caller
    that gave argv directly) is judged by `plain_default`."""
    if isinstance(word, _Word):
        return word.dynamic
    return plain_default


# ---- lexing -------------------------------------------------------------------------------
# A small hand-written lexer, not shlex: shlex cannot report whether a character was quoted, treats
# `#` as a comment anywhere in the line, and reads `;` glued to a word differently from bash.

def _lex(text: str):
    """Split a shell string into simple commands.

    Returns (words, operator_before, operator_after, output_redirect_targets) per command,
    operators being one of ';' '&&' '||' '|' '&' or None at the ends. Redirections are dropped
    from the words. Raises Denied('unsupported-syntax') for anything outside the supported subset.
    """
    n = len(text)
    i = 0
    cur: list[str] = []
    in_word = False
    quoted = False          # the word so far contained a quote or escape (so `2` is not an fd)
    dyn = False             # the word so far has an unquoted glob or a `$` outside single quotes
    expect_target = False   # the previous token was a redirect, so the next word is its target
    target_is_output = False
    words: list[str] = []
    redirs: list[_Word] = []
    raw: list[tuple[list[str], str | None, list[_Word]]] = []

    def end_word():
        nonlocal cur, in_word, quoted, dyn, expect_target
        if in_word:
            w = _Word("".join(cur))
            w.dynamic = dyn
            if expect_target:
                expect_target = False   # a redirect target is a file name, not part of the command
                if target_is_output:
                    # Keep the _Word: the ledger check needs to know whether `$VAR` or a glob
                    # could make this name something other than the text the runner sees.
                    redirs.append(w)
            else:
                words.append(w)
        cur, in_word, quoted, dyn = [], False, False, False

    def end_command(op):
        nonlocal words, redirs
        end_word()
        if expect_target:
            raise _unsupported(text, "redirect with no target", "give the redirect a file name")
        if words or redirs:
            raw.append((words, op, redirs))
        words, redirs = [], []

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
        elif c in " \t":               # \r never gets here: _check_text refuses it
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
                raise _unsupported(text, "unbalanced quote (') ", "close the quote")
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
                if text[j] == "$":
                    dyn = True          # expands inside double quotes (a glob does not)
                buf.append(text[j])
                j += 1
            if j >= n:
                raise _unsupported(text, 'unbalanced quote (")', "close the quote")
            cur.append("".join(buf))
            in_word = quoted = True
            i = j + 1
        elif c == "$" and nxt in ("'", '"'):
            raise _unsupported(text, "$'...' or $\"...\" quoting",
                               "use plain quotes and write the characters out")
        elif c == "$" and nxt == "{":   # ${NAME}: keep it whole so its braces are not "brace expansion"
            j = text.find("}", i)
            if j < 0:
                raise _unsupported(text, "unterminated ${...}", "close the brace or quote the word")
            cur.append(text[i:j + 1])
            in_word = dyn = True
            i = j + 1
        elif c in "()":
            raise _unsupported(text, "parenthesis (subshell, group or function)",
                               "run the pieces as separate commands")
        elif c in "{}":
            raise _unsupported(text, "unquoted brace (group or brace expansion)",
                               "quote the word, or list the paths explicitly")
        elif c == ";":
            end_command(";")
            i += 1
        elif c == "&":
            if nxt == ">":              # &> and &>> redirect both streams
                end_word()
                i += 3 if text[i + 2:i + 3] == ">" else 2
                expect_target, target_is_output = True, True
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
                cur, in_word, dyn = [], False, False    # `2>&1`: the digit names a file descriptor
            else:
                end_word()
            if c == "<" and nxt == "<":
                raise _unsupported(text, "heredoc or here-string",
                                   "write the text to a file and pass the path")
            i += 2 if (nxt and nxt in "&|" or (c == ">" and nxt == ">") or (c == "<" and nxt == ">")) else 1
            expect_target = True
            target_is_output = c == ">" or nxt == ">"
        else:
            cur.append(c)
            in_word = True
            if c == "$" or c in "*?[":  # unquoted: the shell may expand it
                dyn = True
            i += 1
    end_command(None)

    # Attach the operator before each command (it is the previous command's trailing operator).
    out = []
    before = None
    for words_, after, redirs_ in raw:
        out.append((words_, before, after, redirs_))
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


def _consume_option(argv, spec, opt, wname, whole):
    """Consume one wrapper option (already popped from argv). Returns (key, value) for options
    that carry a value, else (None, None). Raises on an option the spec does not list."""
    def unknown():
        return _unsupported(
            whole, f"option {opt} of {wname} is not known to the runner",
            f"drop the option, or call the command without the {wname} wrapper")

    def need_value():
        return _unsupported(whole, f"option {opt} of {wname} is missing its value",
                            "give the option a value")

    if opt.startswith("--"):
        key, eq, val = opt.partition("=")
        if key in spec.long_values:
            if not eq:
                if not argv:
                    raise need_value()
                val = argv.pop(0)
            return key, val
        if key in spec.long_flags:
            return None, None
        raise unknown()
    body = opt[1:]
    if spec.numeric and body.isdigit():
        return None, None
    k = 0
    while k < len(body):
        if body[k] in spec.values:
            val = body[k + 1:]
            if not val:
                if not argv:
                    raise need_value()
                val = argv.pop(0)
            return "-" + body[k], val
        if body[k] in spec.flags:
            k += 1
        else:
            raise unknown()
    return None, None


def _unwrap(argv, cwd):
    """Strip leading assignments and wrapper commands.

    Returns (argv, saw_xargs, cwd, wrappers_used). `cwd` follows chdir options (`env -C DIR`),
    and becomes None when the directory cannot be worked out. A wrapper that hands a string to a
    shell (`flock -c`) is rewritten as `sh -c STRING` so the shell path checks it.
    """
    original = list(argv)     # kept whole so a refusal can quote the command it was part of
    argv = list(argv)
    saw_xargs = False
    used: list[str] = []
    while argv:
        if _is_assignment(argv[0]):
            var = argv[0].partition("=")[0]
            if var.startswith("GIT_CONFIG") or var == "BASH_ENV":
                raise _unsupported(argv, f"assignment of {var}",
                                   "set it in the tool's own config instead; it can hide a push "
                                   "alias or run a startup file")
            argv.pop(0)
            continue
        name = os.path.basename(argv[0])
        if name == "gozer" and len(argv) > 1 and argv[1] == "run":
            # `gozer run <lease options> -- COMMAND`: everything before `--` is the lease's own.
            if "--" not in argv:
                raise _unsupported(argv, "gozer run without `--` before the command",
                                   "write `gozer run <options> -- <command>`")
            argv = argv[argv.index("--") + 1:]
            used.append(name)
            continue
        spec = WRAPPERS.get(name)
        if spec is None:
            break
        argv.pop(0)
        used.append(name)
        saw_xargs = saw_xargs or name == "xargs"
        shell_string = None
        while argv and argv[0].startswith("-") and argv[0] != "-":
            opt = argv.pop(0)
            if opt == "--":
                break
            key, val = _consume_option(argv, spec, opt, name, original)
            if key in spec.chdir:
                cwd = _chdir_value(cwd, val)
            if key in spec.shell_opts:
                shell_string = val
        for _ in range(spec.positional):
            if argv:
                argv.pop(0)
        if shell_string is None and spec.shell_opts and argv and argv[0] in spec.shell_opts:
            if len(argv) < 2:
                raise _unsupported(argv, f"{name} {argv[0]} is missing its command string",
                                   "give the option a command string")
            shell_string = argv[1]
        if shell_string is not None:
            argv = ["sh", "-c", shell_string]
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


def _first_operand(argv: list[str]) -> str | None:
    """The first word after the command that is not an option: a tool's subcommand."""
    for a in argv[1:]:
        if not a.startswith("-"):
            return a
    return None


def _git_subcommand_index(argv) -> int:
    """Index of git's real subcommand, after any global options (and their values). Raises on a
    `--long` global option the runner does not know, because it cannot tell whether the next word
    is that option's value or the subcommand."""
    i = 1
    while i < len(argv) and argv[i].startswith("-"):
        a = argv[i]
        if a in GIT_VALUE_OPTIONS:
            i += 2
        elif a.startswith("--") and "=" not in a and a not in GIT_FLAG_OPTIONS:
            raise _unsupported(argv, f"git option {a} is not known to the runner",
                               "write the option as --name=value, or drop it")
        else:
            i += 1
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
    # upload, upload-large-folder, and any later upload-* subcommand; only at the subcommand
    # position, so `hf download --local-dir uploads` is fine.
    if name not in {"hf", "huggingface-cli"}:
        return False
    sub = _first_operand(argv)
    return sub is not None and sub.startswith("upload")


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


def _rm_operand_path(a, ctx):
    """The absolute path an rm operand names, or None when it cannot be worked out."""
    path = os.path.expanduser(a)
    if not os.path.isabs(path):
        if ctx.cwd is None:
            return None
        path = os.path.join(ctx.cwd, path)
    return path


def _rm_glob_in_run_root(name, argv, ctx):
    """`rm *.log` run from the run directory root: the glob could match the ledger. Globs in a
    deeper directory are judged by rm-outside-run-dir like any other path."""
    if name not in DELETERS:
        return False
    for a in _operands(argv):
        if "$" in a:
            continue                      # the outside rule refuses it
        path = _rm_operand_path(a, ctx)
        if path is None or any(c in os.path.dirname(path) for c in "*?["):
            continue
        if any(c in os.path.basename(path) for c in "*?[") \
                and os.path.realpath(os.path.dirname(path)) == ctx.run_dir:
            return True
    return False


def _rm_outside_run_dir(name, argv, ctx):
    if name not in DELETERS:
        return False
    for a in _operands(argv):
        if "$" in a:
            return True  # unknown path
        path = _rm_operand_path(a, ctx)
        if path is None:
            return True  # relative path in an unknown directory
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
    # rm-ledger comes before the glob and outside rules on purpose: a glob such as `{run}/ledger*`
    # expands in the run directory itself, which they also refuse. The more specific name wins.
    ("rm-ledger", _rm_ledger),
    ("rm-glob-in-run-root", _rm_glob_in_run_root),
    ("rm-outside-run-dir", _rm_outside_run_dir),
    ("rm-xargs", _rm_xargs),
]

# Extra text for rules whose name alone does not say why.
RULE_DETAIL = {
    "rm-glob-in-run-root": "a glob in the run directory root could match the ledger; "
                           "name the files, or glob inside a subdirectory",
}


# ---- checking and running -----------------------------------------------------------------

def _shell_command_string(unwrapped) -> str:
    """The command string a shell invocation will run. Raises Denied if there is none or the
    options cannot be parsed.

    bash takes the FIRST NON-OPTION word after its options as the command string, so options are
    parsed with bash's grammar: short clusters (`-lc`), `-o NAME`/`-O NAME` (which take a value),
    `--rcfile FILE`/`--init-file FILE`, and a fixed list of long flags. An option not in these lists
    is refused rather than guessed at. A shell with no `-c` reads stdin or a script, neither of
    which can be judged.
    """
    args = unwrapped[1:]
    shell = os.path.basename(unwrapped[0])
    has_c = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            break
        if a.startswith("--"):
            if a in SHELL_LONG_FLAGS:
                i += 1
            elif a in SHELL_LONG_VALUES:
                if i + 1 >= len(args):
                    raise _unsupported(unwrapped, f"{shell} option {a} is missing its value",
                                       "give the option a value")
                i += 2
            else:
                raise _unsupported(unwrapped, f"option {a} of {shell} is not known to the runner",
                                   "drop the option")
        elif a[:1] in "-+" and len(a) > 1:
            pending = 0
            for ch in a[1:]:
                if ch in SHELL_SHORT_VALUES:
                    pending += 1
                elif ch in SHELL_SHORT_FLAGS:
                    has_c = has_c or (ch == "c" and a[0] == "-")
                else:
                    raise _unsupported(unwrapped, f"option {a} of {shell} is not known to the runner",
                                       "drop the option")
            if i + pending >= len(args):
                raise _unsupported(unwrapped, f"{shell} option {a} is missing its value",
                                   "give the option a value")
            i += 1 + pending
        else:
            break
    if not has_c:
        raise _unsupported(unwrapped, f"{shell} without a -c command string (it would read stdin "
                                      "or a script)",
                           "pass the command with -c, or run the script's commands directly")
    if i >= len(args):
        raise _unsupported(unwrapped, f"{shell} -c with no command string", "give -c a string")
    return args[i]


def _cd_target(args, cwd, before):
    """Where `cd ARGS` leaves the shell, as a path, or None when it cannot be known."""
    physical = "-P" in args
    operands = [a for a in args if not a.startswith("-") or a == "-"]
    target = operands[0] if operands else "~"
    if len(operands) > 1 or target == "-" or any(ch in target for ch in "$*?[`"):
        return None
    # After `&&`, `||`, `|` or `&` the cd may not have run (or ran in a subshell), so which
    # directory the later commands see is not known.
    if before in {"&&", "||", "|", "&"}:
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


def _check_argv(argv, run_dir, cwd, st, via_xargs_in=False, before=None):
    try:
        return _check_argv_inner(argv, run_dir, cwd, st, via_xargs_in, before)
    except (ValueError, UnicodeError):
        # os.path raises these for a NUL byte or a lone surrogate in a path; refuse, never crash.
        raise _unsupported(argv, "a NUL byte or invalid character in a word",
                           "remove it; paths must be plain text")


def _check_argv_inner(argv, run_dir, cwd, st, via_xargs_in, before):
    run_real = os.path.realpath(run_dir)
    unwrapped, via_xargs, eff_cwd, used = _unwrap(argv, cwd)
    via_xargs = via_xargs or via_xargs_in
    if not unwrapped:
        if "exec" in used:
            raise _unsupported(argv, "exec with no command (it rewires the shell itself)",
                               DEFAULT_HINT)
        return cwd
    first = unwrapped[0]
    name = os.path.basename(first)
    if first in KEYWORDS:
        raise _unsupported(argv, f"shell keyword: {first}", KEYWORD_HINT)
    if first.startswith("-"):
        raise _unsupported(argv, f"command word {first!r} starts with '-'",
                           "start the command with a program name")
    if _dynamic(first, plain_default=any(ch in first for ch in "$*?[")):
        raise _unsupported(argv, "a command name built from a variable or glob",
                           "write the command name out, or quote it")
    if name in ENV_SETTERS:
        for a in unwrapped[1:]:
            var = a.partition("=")[0]
            if var.startswith("GIT_CONFIG") or var in DANGEROUS_ENV:
                raise _unsupported(argv, f"{name} of {var}",
                                   "do not set it; it can hide a push alias or run a startup file")

    if name in REFUSED_SHELLS:
        raise _unsupported(argv, f"shell {name}",
                           "only bash, sh and dash command strings are understood; use bash -c")
    if name in SHELLS:
        string = _shell_command_string(unwrapped)
        # Recurse in the CURRENT directory; a cd inside it does not leak out to the caller.
        # (the nested check updates `st`, so a non-benign command inside it ends directory trust)
        _check_text(string, run_dir, eff_cwd, st, via_xargs)
        return cwd

    if name == "find" and any(a in {"-exec", "-execdir", "-ok", "-okdir", "-delete"} for a in unwrapped):
        raise _unsupported(argv, "find -exec/-delete (runs or deletes things the rules never see)",
                           "pipe find's output to a command you can name, or delete by explicit path")

    if name in NAMED_TOOLS:
        words = unwrapped[:_git_subcommand_index(unwrapped) + 1] if name == "git" else unwrapped
        if any(_dynamic(w, plain_default=False) for w in words):
            raise _unsupported(argv, f"an unquoted glob or a `$` in the words of {name} "
                                     "(it could change the subcommand)",
                               "quote the word, or write it out")

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
            raise Denied(rule_name, " ".join(argv), RULE_DETAIL.get(rule_name, ""))

    stable_before = st.fs_stable
    if name not in BENIGN:
        st.fs_stable = False            # it could create, remove or relink a directory
    if name in {"pushd", "popd", "dirs"} or (name == "cd" and used):
        return None                     # directory stack, or cd run through a wrapper: unknowable
    if name == "cd":
        if not stable_before:
            return None                 # an earlier command could have changed what exists
        return _cd_target(unwrapped[1:], cwd, before)
    return cwd


def _check_text(text, run_dir, cwd, st, via_xargs=False):
    """Check every command in a shell string; returns the directory after it."""
    if "\0" in text:
        raise _unsupported(text, "a NUL byte", "remove it; commands must be plain text")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise _unsupported(text, "an invalid character (lone surrogate)",
                           "remove it; commands must be plain text") from None
    if "\r" in text:
        # bash keeps \r inside a word, so `git status\rgit push` is one odd word to bash; the lexer
        # would split it. Refuse it so the two cannot disagree.
        raise _unsupported(text, "a carriage return (\\r)", "remove it; use plain newlines")
    if any(marker in text for marker in ("$(", "`", "<(", ">(")):
        raise Denied("substitution", text,
                     "command substitution or process substitution cannot be judged. To proceed: "
                     "run the inner command first, write its output to a file, and pass the path")
    commands = _lex(text)
    for _, _, _, redirs in commands:
        for target in redirs:
            if target.dynamic:
                # The rm rules refuse any `$` as an unknown path; a redirect needs the same care.
                raise Denied("redirect-ledger", text,
                             "the target is built from a variable or glob, so the runner cannot "
                             "tell whether it is the ledger; write the file name out")
            if os.path.basename(target).startswith("ledger"):
                raise Denied("redirect-ledger", text,
                             "a redirect to a ledger file would truncate it; "
                             "write somewhere else")
    for words, before, after, _ in commands:
        if not words:
            continue                    # a bare redirect such as `> file`
        new_cwd = _check_argv(words, run_dir, cwd, st, via_xargs, before=before)
        # In a pipeline or in the background the command runs in a subshell, so a `cd` there does
        # not change the directory the following commands run in.
        if before != "|" and after not in {"|", "&"}:
            cwd = new_cwd
    return cwd


def _finish(st: _State, text: str) -> None:
    if st.made_symlink and st.deletes:
        raise Denied("symlink-with-delete", text,
                     "the string makes a symbolic link and deletes, and the delete could go "
                     "through the link; run them as separate commands")


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
