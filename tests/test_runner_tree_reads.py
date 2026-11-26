# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 864: a recursive search from a stage agent step must not reach outside the run directory.

In a stage 2 trial an agent ran `grep -R "<error text>" -n <the operator's home directory>`. The command runner let it through. The
search ran until the tool timeout and read the whole home directory, where credential files live. The runner now refuses a command that
reads a whole tree (a recursive grep, rg, ag, ack, fd, find, `ls -R`, tree, du) when any starting point, or the directory the search starts
in when none is named, lies outside the run directory and the read-only places the run names. A single named file is not a tree read and
is not judged here (the module docstring of orchard/runner.py says what remains).

What this file checks, in order:
  1. the refusal, for every spelling the issue names and for the ways round it (links, `..`, a sibling directory whose name starts with the
     run directory's name, globs, `~`, `$HOME`, wrappers, xargs);
  2. what stays allowed;
  3. that nothing outside the allowed places is opened or listed while the runner decides (an audit hook records opens and listings);
  4. hostile trees in a child process with a memory limit and an alarm (deep trees, wide directories, link loops, a FIFO);
  5. the agent's side: read roots come from the run's named inputs and exclude the Hugging Face token directory, and a search command has a
     shorter wall-time limit that the step's ledger record carries.
Everything runs against temporary directories. No chip, no model, no network."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fake_model import FakeModel, call, final, turn
from orchard import runner
from orchard.agent import Tools, agent_env, agent_read_roots, spawn_checked
from orchard.ledger import Ledger
from orchard.runner import Denied, check_string, check_string_info, run_shell
from test_tool_dedupe import put, run, shell, step, tools   # noqa: F401  (run is a fixture of that module)

RULE = "tree-read-outside-run-dir"
REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def w(tmp_path):
    """A run directory, a fake operator home with credential-looking files, a read-only input, and a sibling whose name starts like the run's."""
    home = tmp_path / "opshome"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("SENTINEL-CREDENTIAL\n")
    (home / ".cache" / "huggingface").mkdir(parents=True)
    (home / ".cache" / "huggingface" / "token").write_text("hf_SENTINEL\n")
    (home / "notes.txt").write_text("a note\n")
    run = tmp_path / "run"
    (run / "sub").mkdir(parents=True)
    (run / "stages" / "2" / "port").mkdir(parents=True)
    (run / "stages" / "2" / "port" / "port_model.py").write_text("print('embedding')\n")
    (run / "sub" / "a.txt").write_text("x\n")
    (run / "home").mkdir()
    snap = tmp_path / "inputs" / "snapshot"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}\n")
    evil = tmp_path / "runevil"
    evil.mkdir()
    (evil / "secret.txt").write_text("evil\n")
    return SimpleNamespace(tmp=tmp_path, run=run, home=home, snap=snap, evil=evil)


def fill(cmd, w):
    """Put the paths in. A plain replace, because a command may hold shell braces such as ${HOME}."""
    for key, value in (("run", w.run), ("home", w.home), ("snap", w.snap), ("evil", w.evil), ("tmp", w.tmp)):
        cmd = cmd.replace("{" + key + "}", str(value))
    return cmd


# ---- 1. the refusal --------------------------------------------------------------------------------------------------------------

DENY = [
    # grep: every way to ask for recursion, with the home directory as the start
    "grep -R x {home}", "grep -r x {home}", "grep -rn x {home}", "grep -Rin x {home}", "grep -iR x {home}", "grep -A3 -R x {home}",
    "grep --recursive x {home}", "grep --dereference-recursive x {home}", "grep --rec x {home}", "grep -d recurse x {home}",
    "grep --directories=recurse x {home}", "grep -R -e x {home}", "grep -R -f {run}/sub/a.txt {home}", "grep -R -- x {home}",
    "grep -R x {home}/.ssh", "grep -R x {home}/.cache/huggingface", "grep -rl 'embedding wants ROW_MAJOR ids' {home}",
    "rgrep x {home}", "egrep -R 'a|b' {home}", "fgrep -r x {home}",
    # no path given: the search starts in the current directory
    "cd {home}; grep -R x", "cd {home}; grep -r x .", "cd {home}; rg x", "cd {home}; find", "cd {home}; ls -R", "cd {home}; tree",
    "cd {home}; du -a", "cd {tmp}; grep -R x",
    # other readers
    "rg x {home}", "rg -uuu x {home}", "rg -n --hidden x {home}", "rg --files {home}", "ag x {home}", "ack x {home}", "fd x {home}",
    "fdfind x {home}", "find {home}", "find {home} -name id_ed25519", "find / -name id_ed25519", "find -L {home} -type f",
    "find -H {home} -name x", "find -P {home}", "find {run} {home} -name x", "ls -R {home}", "ls -laR {home}", "ls --recursive {home}",
    "ls -R {run} {home}", "tree {home}", "tree -a {home}", "du {home}", "du -a {home}", "du -sh {home}/..",
    # round the run directory: dotdot, a sibling with the same name start, a parent
    "grep -R x {run}/..", "grep -R x {run}/sub/../../opshome", "grep -R x {evil}", "grep -R x {run}evil", "find {run}/.. -name x",
    "grep -R x ../opshome", "cd sub; grep -R x ../../opshome", "grep -R x {tmp}", "grep -R x /", "grep -R x /home", "grep -R x /etc",
    # words the runner cannot place
    "grep -R x ~", "grep -R x ~/.ssh", "grep -R x $HOME", "grep -R x \"$HOME\"", "grep -R x ${HOME}", "grep -R x {home}/$SUB",
    "find ~ -name x", "ls -R ~", "grep -R x {home}/*", "grep -R x {run}/../*", "find {home}/.ssh/* -name x",
    # wrappers and shells that the runner already reads through
    "bash -c 'grep -R x {home}'", "sh -c \"find {home}\"", "env grep -R x {home}", "env -C {home} grep -R x", "timeout 5 grep -R x {home}",
    "nice -n 5 find {home}", "sudo -n grep -R x {home}", "nohup grep -R x {home}", "time grep -R x {home}", "stdbuf -o0 grep -R x {home}",
    "echo hi; grep -R x {home}", "echo hi && grep -R x {home}", "false || grep -R x {home}", "grep -R x {home} | head",
    "cd {run}/sub && cd {home} && grep -R x .", "bash -c 'cd {home}; grep -R x .'",
    # the start points come from a pipe the runner cannot see
    "find {run} -name x | xargs grep -R y", "echo {home} | xargs find", "echo {home} | xargs ls -R", "echo {home} | xargs du",
    "echo {home} | xargs rg x", "cat list | xargs -n1 tree", "echo {home} | xargs fd x",
    # the directory is lost, so a search without a path cannot be placed
    "cd $DIR; grep -R x", "true && cd {home}; grep -R x .",
]


@pytest.mark.parametrize("cmd", DENY)
def test_a_tree_read_outside_the_run_directory_is_refused_by_name(cmd, w):
    with pytest.raises(Denied) as exc:
        check_string(fill(cmd, w), w.run)
    assert exc.value.rule == RULE, f"{cmd!r} was refused by {exc.value.rule!r}: {exc.value}"


def test_the_refusal_names_the_path_and_says_what_to_search_instead(w):
    with pytest.raises(Denied) as exc:
        check_string(f"grep -R 'embedding wants ROW_MAJOR ids' -n {w.home}", w.run)
    text = str(exc.value)
    assert str(w.home) in text                       # which path
    assert "To proceed" in text and str(w.run) in text  # what to do, and where the search may go
    assert exc.value.detail.startswith("grep would read a whole tree")     # which command: the reason alone, without the echoed command line
    with pytest.raises(Denied) as exc2:
        check_string(f"find {w.home} -name x", w.run)
    assert exc2.value.detail.startswith("find would read a whole tree")


def test_the_refusal_for_a_glob_and_for_a_variable_say_how_to_write_the_path(w):
    for cmd, needle in ((f"grep -R x {w.home}/$SUB", "variable"), ("grep -R x ~", "~"), (f"grep -R x {w.run}/../*", "..")):
        with pytest.raises(Denied) as exc:
            check_string(cmd, w.run)
        assert needle in str(exc.value) and "To proceed" in str(exc.value), cmd


def test_a_link_inside_the_run_directory_that_leads_out_is_refused_as_a_start_point(w):
    os.symlink(w.home, w.run / "linkhome")
    for cmd in ("grep -R x {run}/linkhome", "grep -r x {run}/linkhome/.ssh", "find {run}/linkhome -name x", "ls -R {run}/linkhome"):
        with pytest.raises(Denied) as exc:
            check_string(fill(cmd, w), w.run)
        assert exc.value.rule == RULE, cmd
        assert str(w.home) in str(exc.value), cmd       # the resolved place is named as well as the link


@pytest.mark.parametrize("cmd", ["grep -R x {run}", "grep -R x .", "grep -R x", "find -L {run} -name x", "find -L . -name x",
                                 "find {run} -follow -name x", "rg -L x {run}", "rg --follow x", "ls -RL {run}", "tree -l {run}", "du -L {run}",
                                 "fd -L x {run}"])
def test_a_tool_that_follows_links_refuses_a_tree_holding_a_link_that_leads_out(cmd, w):
    os.symlink(w.home, w.run / "sub" / "linkhome")
    with pytest.raises(Denied) as exc:
        check_string(fill(cmd, w), w.run)
    assert exc.value.rule == RULE
    assert "linkhome" in str(exc.value) and str(w.home) in str(exc.value)


@pytest.mark.parametrize("cmd", ["grep -r x {run}", "grep -r x", "find {run} -name x", "rg x {run}", "ls -R {run}", "tree {run}", "du {run}"])
def test_a_tool_that_does_not_follow_links_leaves_a_link_in_the_tree_alone(cmd, w):
    # grep -r, rg, find, ls -R, tree and du do not follow a link met while walking, so the link is not a way out.
    os.symlink(w.home, w.run / "sub" / "linkhome")
    check_string(fill(cmd, w), w.run)


# ---- 2. what stays allowed -------------------------------------------------------------------------------------------------------

ALLOW = [
    "grep -R x {run}", "grep -r x {run}/stages", "grep -R x .", "grep -R x", "grep -r pattern sub", "grep -R -e x -e y {run}/sub",
    "grep -R --include='*.py' x {run}/stages/2", "grep -R -i -n 'embedding wants' {run}/stages/2/port", "grep -d recurse x {run}",
    "grep -R x {run}/sub {run}/stages", "grep -R 'a/b' {run}", "grep -R x {run}/stages/*/port", "grep -R x {run}/sub/../stages",
    "rg x {run}", "rg -n x", "rg --files {run}/stages", "rg 'embedding wants' {run}/stages", "fd x {run}",
    "find {run} -name x", "find . -name '*.py'", "find", "find {run}/stages -type f | head", "find {run}/sub {run}/stages -name x",
    "find -L {run} -name x", "ls -R {run}/sub", "ls -laR", "ls -R", "tree {run}", "tree -L 2", "du -sh {run}", "du -a .",
    "bash -c 'grep -R x {run}/sub'", "cd {run}/sub; grep -R x .", "cd {run}/sub && grep -R x .", "timeout 60 grep -R x {run}/sub",
    "env FOO=1 grep -R x {run}", "env -C {run}/sub grep -R x", "sudo -n find {run}", "find {run} -name x | xargs grep -l y",
    "cat {run}/ledger.jsonl | xargs echo",
    # not a tree read: a named file, a plain listing, a pipe with no path
    "grep x {home}/notes.txt", "grep -n x {run}/sub/a.txt", "grep -c x /etc/hostname", "ls {home}", "ls -la /", "cat {home}/notes.txt",
    "head -n 3 {home}/notes.txt", "wc -l {home}/notes.txt", "ls | grep x", "git grep x", "git -C {run} log --oneline",
    "echo x | grep x", "grep x < {run}/sub/a.txt", "cd /; ls", "cd / | cat; rm foo", "du -sh", "stat {home}",
]


@pytest.mark.parametrize("cmd", ALLOW)
def test_searches_inside_the_run_directory_and_single_file_reads_pass(cmd, w):
    check_string(fill(cmd, w), w.run)


def test_a_search_of_a_named_read_only_input_passes_only_when_the_run_names_it(w):
    cmd = f"grep -R x {w.snap}"
    with pytest.raises(Denied):
        check_string(cmd, w.run)
    check_string(cmd, w.run, read_roots=(str(w.snap),))
    check_string(f"find {w.snap} -name config.json", w.run, read_roots=[w.snap])
    check_string(f"grep -R x {w.snap}/config.json", w.run, read_roots=(str(w.snap),))


def test_a_read_root_is_a_directory_tree_and_not_its_name_prefix_or_its_parent(w):
    roots = (str(w.snap),)
    for cmd in (f"grep -R x {w.snap}evil", f"grep -R x {w.snap}/..", f"grep -R x {w.snap.parent}", f"grep -R x {w.home}"):
        with pytest.raises(Denied) as exc:
            check_string(cmd, w.run, read_roots=roots)
        assert exc.value.rule == RULE, cmd


def test_a_read_root_that_is_a_link_is_judged_by_where_it_leads(w):
    os.symlink(w.home, w.tmp / "inputs" / "linked")
    # the operator named a link: the root is where it leads, so the home directory becomes readable only by the operator's choice, and a
    # sibling of the link's target is not.
    check_string(f"grep -R x {w.tmp}/inputs/linked", w.run, read_roots=(str(w.tmp / "inputs" / "linked"),))
    with pytest.raises(Denied):
        check_string(f"grep -R x {w.tmp}/inputs/linked", w.run, read_roots=(str(w.snap),))


def test_check_string_keeps_its_old_two_argument_form_and_returns_nothing(w):
    assert check_string("ls -la", w.run) is None
    assert check_string(f"grep -R x {w.run}", w.run) is None


def test_check_string_info_counts_the_tree_readers_it_judged(w):
    assert check_string_info("ls -la", w.run).tree_reads == 0
    assert check_string_info(f"grep -R x {w.run}/sub; find . -name x; ls", w.run).tree_reads == 2
    assert check_string_info("bash -c 'rg x .' | head", w.run).tree_reads == 1
    assert check_string_info("grep x file", w.run).tree_reads == 0        # not recursive
    with pytest.raises(Denied):
        check_string_info(f"grep -R x {w.home}", w.run)


def test_run_shell_refuses_before_anything_starts_and_still_runs_a_search_inside(w):
    with pytest.raises(Denied):
        run_shell(f"grep -R SENTINEL {w.home}", w.run, capture_output=True, universal_newlines=True)
    done = run_shell(f"grep -R embedding {w.run}/stages", w.run, capture_output=True, universal_newlines=True)
    assert done.returncode == 0 and "embedding" in done.stdout


# ---- 3. nothing outside the allowed places is opened or listed while the runner decides ---------------------------------------------

_SEEN = []
_WATCH = {"on": False, "needle": ""}


def _hook(event, args):
    if _WATCH["on"] and event in {"open", "os.scandir", "os.listdir", "glob.glob", "os.chdir"}:
        target = args[0] if args else None
        if isinstance(target, (str, bytes, os.PathLike)) and _WATCH["needle"] in os.fsdecode(target):
            _SEEN.append((event, os.fsdecode(target)))


sys.addaudithook(_hook)


@pytest.mark.parametrize("cmd", [c for c in DENY if "{home}" in c and "$" not in c and "cd {home}" not in c])
def test_a_refused_search_opens_and_lists_nothing_of_the_home_directory(cmd, w):
    # The needle is the home directory's own name, which no other path in the test shares (the run directory and its siblings have other
    # names). The hook sees opens, directory listings, globs and directory changes.
    del _SEEN[:]
    _WATCH.update(on=True, needle="opshome")
    try:
        with pytest.raises(Denied):
            check_string(fill(cmd, w), w.run)
    finally:
        _WATCH["on"] = False
    touched = [(e, p) for e, p in _SEEN if "opshome" in p]
    assert not touched, f"{cmd!r} touched {touched}"


def test_the_sentinel_is_unreadable_and_the_refusal_still_comes_with_no_permission_error(w):
    os.chmod(w.home / ".ssh" / "id_ed25519", 0)
    os.chmod(w.home / ".ssh", 0)
    try:
        with pytest.raises(Denied) as exc:
            check_string(f"grep -R x {w.home}", w.run)
        assert exc.value.rule == RULE and "Permission" not in str(exc.value)
    finally:
        os.chmod(w.home / ".ssh", 0o755)


# ---- 4. hostile trees, in a child with a memory limit and an alarm -------------------------------------------------------------

ALARM_S = 10
ADDRESS_SPACE = 2 << 30


def in_child(code: str) -> dict:
    prog = ("import json, resource, signal, sys\n"
            f"resource.setrlimit(resource.RLIMIT_AS, ({ADDRESS_SPACE}, {ADDRESS_SPACE}))\n"
            f"signal.alarm({ALARM_S})\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "from orchard import runner\n"
            "def verdict(cmd, run, **kw):\n"
            "    try:\n"
            "        runner.check_string(cmd, run, **kw)\n"
            "        return {'v': 'allowed'}\n"
            "    except runner.Denied as e:\n"
            "        return {'v': 'refused', 'rule': e.rule, 'text': str(e)}\n"
            + code)
    done = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, timeout=ALARM_S + 10, start_new_session=True)
    assert done.returncode == 0, f"child ended {done.returncode}: {done.stderr[-400:]}"
    return json.loads(done.stdout.strip().splitlines()[-1])


def _deep(root: Path, depth: int) -> Path:
    here = root
    for _ in range(depth):
        here = here / "d"
    here.mkdir(parents=True)
    return here


def test_a_tree_deeper_than_the_walk_limit_is_refused_when_the_tool_follows_links(w):
    _deep(w.run / "deep", 100)
    out = in_child(f"print(json.dumps(verdict('grep -R x {w.run}', {str(w.run)!r})))")
    assert out["v"] == "refused" and out["rule"] == RULE and "too" in out["text"]


def test_a_link_to_the_outside_far_down_a_deep_tree_is_found_before_the_limit(w):
    bottom = _deep(w.run / "deep", 30)
    os.symlink(w.home, bottom / "out")
    out = in_child(f"print(json.dumps(verdict('grep -R x {w.run}', {str(w.run)!r})))")
    assert out["v"] == "refused" and "out" in out["text"] and str(w.home) in out["text"]


def test_a_directory_with_more_entries_than_the_walk_limit_is_refused(w):
    wide = w.run / "wide"
    wide.mkdir()
    for i in range(700):
        (wide / f"f{i}").write_text("x")
    out = in_child("runner.TREE_SCAN_MAX_ENTRIES = 500\n"
                   f"print(json.dumps(verdict('grep -R x {w.run}', {str(w.run)!r})))")
    assert out["v"] == "refused" and out["rule"] == RULE and "too large" in out["text"]
    # a tool that does not follow links never walks the tree, so the same tree is fine for it
    out = in_child("runner.TREE_SCAN_MAX_ENTRIES = 500\n"
                   f"print(json.dumps(verdict('grep -r x {w.run}', {str(w.run)!r})))")
    assert out["v"] == "allowed"


def test_loops_of_links_and_a_fifo_inside_the_run_directory_neither_hang_nor_crash_the_decision(w):
    os.symlink(w.run / "b", w.run / "a")
    os.symlink(w.run / "a", w.run / "b")
    os.symlink(".", w.run / "self")
    os.symlink("..", w.run / "sub" / "up")          # from run/sub, `..` is the run directory itself
    os.mkfifo(w.run / "sub" / "pipe.jsonl")
    for cmd in (f"grep -R x {w.run}", f"find -L {w.run} -name x", f"ls -RL {w.run}", f"grep -R x {w.run}/a", f"grep -R x {w.run}/self"):
        out = in_child(f"print(json.dumps(verdict({cmd!r}, {str(w.run)!r})))")
        assert out["v"] == "allowed", (cmd, out)         # every link here stays in the run directory
    os.symlink("../..", w.run / "sub" / "out")           # from run/sub, `../..` is the run directory's parent: it leaves
    out = in_child(f"print(json.dumps(verdict('grep -R x {w.run}', {str(w.run)!r})))")
    assert out["v"] == "refused" and "out" in out["text"] and str(w.tmp) in out["text"]


def test_a_start_point_that_is_a_fifo_or_a_missing_path_does_not_block(w):
    os.mkfifo(w.run / "pipe.jsonl")
    for cmd in (f"grep -R x {w.run}/pipe.jsonl", f"grep -R x {w.run}/missing", f"find {w.run}/missing -name x"):
        out = in_child(f"print(json.dumps(verdict({cmd!r}, {str(w.run)!r})))")
        assert out["v"] == "allowed", (cmd, out)


def test_a_command_with_a_huge_glob_is_refused_without_listing_it_all(w):
    many = w.run / "many"
    many.mkdir()
    for i in range(300):
        (many / f"f{i}").write_text("x")
    out = in_child("runner.TREE_GLOB_MAX = 100\n"
                   f"print(json.dumps(verdict('grep -R x {many}/f*', {str(w.run)!r})))")
    assert out["v"] == "refused" and "glob" in out["text"]


# ---- 5. the agent's side ---------------------------------------------------------------------------------------------------------

def test_read_roots_come_from_the_runs_named_inputs_and_leave_out_the_token_directory(tmp_path):
    hf = tmp_path / "hf"
    (hf / "hub").mkdir(parents=True)
    (hf / "token").write_text("hf_SENTINEL")
    model = tmp_path / "model"
    model.mkdir()
    ref = tmp_path / "ref.py"
    ref.write_text("x")
    cache = tmp_path / "cache"
    cache.mkdir()
    ttm = tmp_path / "ttm"
    ttm.mkdir()
    orch = tmp_path / "orch"
    orch.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()
    paths = SimpleNamespace(orchard_dir=str(orch), hf_home=str(hf), tt_model_root=str(ttm), cache_root=str(cache),
                            operator_home=str(tmp_path / "opshome"))
    roots = agent_read_roots(paths, {"model": str(model), "reference_python": str(ref), "gone": str(tmp_path / "gone")},
                             {"ORCHARD_AGENT_READ_ROOTS": f"{extra}{os.pathsep}relative/dir{os.pathsep}{ref}{os.pathsep}"})
    real = {os.path.realpath(r) for r in roots}
    assert real >= {os.path.realpath(p) for p in (orch, hf / "hub", ttm, cache, model, extra)}
    assert os.path.realpath(hf) not in real                     # the token lives in the Hugging Face home itself
    assert os.path.realpath(tmp_path / "opshome") not in real
    assert os.path.realpath(ref) not in real and os.path.realpath(tmp_path / "gone") not in real     # a file and a missing path are no root
    assert all(os.path.isabs(r) for r in roots)                 # a relative entry of the environment variable is dropped


def test_read_roots_work_with_no_inputs_and_no_environment(tmp_path):
    paths = SimpleNamespace(orchard_dir=str(tmp_path), hf_home=str(tmp_path / "nohf"), tt_model_root=str(tmp_path / "nottm"),
                            cache_root=str(tmp_path / "nocache"), operator_home=str(tmp_path))
    assert list(agent_read_roots(paths, {}, {})) == [str(tmp_path)]


def test_the_shell_tool_refuses_a_search_of_the_home_directory_and_the_sentinel_never_appears(run, w):
    t = tools(run)
    out = shell(t, 1, f"grep -R SENTINEL {w.home}")
    assert out.startswith("refused:") and str(w.home) in out and "SENTINEL-CREDENTIAL" not in out and "hf_SENTINEL" not in out


def test_the_shell_tool_searches_a_named_input_when_it_was_given_the_root(run, w):
    t = tools(run, read_roots=(str(w.snap),))
    (w.snap / "config.json").write_text('{"needle": 1}\n')
    out = shell(t, 1, f"grep -R needle {w.snap}")
    assert out.startswith("exit 0") and "needle" in out
    assert shell(tools(run), 1, f"grep -R needle {w.snap}").startswith("refused:")


def test_spawn_checked_takes_read_roots_and_keeps_its_old_arguments(w):
    with open(w.tmp / "out", "wb") as out:
        with pytest.raises(Denied):
            spawn_checked(f"grep -R x {w.snap}", w.run, {"PATH": os.environ["PATH"]}, 5.0, out)
        code, timed_out = spawn_checked(f"grep -R x {w.snap}", w.run, {"PATH": os.environ["PATH"]}, 5.0, out, read_roots=(str(w.snap),))
    assert (code, timed_out) == (1, False)       # grep finds nothing: exit 1, and it ran


def test_a_search_has_a_shorter_wall_time_limit_than_other_commands(run):
    t = tools(run, search_timeout=0.4, timeout=30.0)
    started = time.monotonic()
    out = shell(t, 1, "grep -R zzz . ; sleep 20")
    elapsed = time.monotonic() - started
    assert out.startswith("killed after 0.4 s") and "search" in out and "narrower" in out
    assert elapsed < 8
    [rec] = t.search_timeouts
    assert rec["limit"] == 0.4 and rec["seconds"] >= 0.35 and rec["turn"] == 1 and rec["command"].startswith("grep -R zzz")


def test_a_command_without_a_tree_reader_keeps_the_ordinary_limit(run):
    t = tools(run, search_timeout=0.2, timeout=30.0)
    out = shell(t, 1, "sleep 1; echo done")
    assert out.startswith("exit 0") and "done" in out and t.search_timeouts == []


def test_the_search_limit_has_a_default_and_a_setting(run):
    from orchard import defaults
    assert defaults.SEARCH_TIMEOUT_S == tools(run).search_timeout
    assert 5 <= defaults.SEARCH_TIMEOUT_S < defaults.TOOL_TIMEOUT_S
    assert tools(run, search_timeout=7.0).search_timeout == 7.0


def transcript(led):
    return [e["data"] for e in led.read() if e["event"] == "evidence" and e["data"].get("what") == "transcript"][-1]


def test_the_steps_ledger_record_carries_the_search_timeouts_with_the_elapsed_time(run):
    script = [call("shell", command="grep -R zzz . ; sleep 20"), final("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: script[turn(r)]) as fm:
        s = step(run, led, fm.endpoint)[0]
        s.tools.search_timeout = 0.3
        out = s.run("system", "user")
        rec = transcript(led)
    assert out.status == "done"
    [one] = rec["search_timeouts"]
    assert one["limit"] == 0.3 and one["seconds"] >= 0.25 and one["command"].startswith("grep -R zzz")


def test_the_steps_ledger_record_has_no_such_key_when_no_search_ran_long(run):
    put(run, "evidence/a.txt", "text")
    script = [call("shell", command="grep -R text ."), final("done")]
    with Ledger(run / "ledger.jsonl") as led, FakeModel(lambda r: script[turn(r)]) as fm:
        step(run, led, fm.endpoint)[0].run("system", "user")
        rec = transcript(led)
    assert "search_timeouts" not in rec


# ---- options that hide the start points or run a program on each file ------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["rg --pre cat x {run}", "rg --pre=cat x", "fd -x cat . {run}", "fd --exec cat x", "fd -X cat x", "fd --exec-batch cat x",
                                 "find {run} -files0-from list", "du --files0-from=list", "tree --fromfile list", "grep -R --no-such-option x {run}"])
def test_options_that_hide_the_start_points_or_run_programs_are_refused_by_name(cmd, w):
    with pytest.raises(Denied) as exc:
        check_string(fill(cmd, w), w.run)
    assert exc.value.rule == "unsupported-syntax", f"{cmd!r}: {exc.value}"
    assert "To proceed" in str(exc.value)


def test_an_unknown_long_option_of_a_grep_that_is_not_recursive_is_left_alone(w):
    check_string(f"grep --no-such-option x {w.run}/sub/a.txt", w.run)        # grep itself reports it; the rule is about tree reads


def test_a_prefix_of_a_long_option_is_read_the_way_grep_reads_it(w):
    for cmd in (f"grep --recurs x {w.home}", f"grep --dereference-rec x {w.home}", f"grep --direct=recurse x {w.home}"):
        with pytest.raises(Denied) as exc:
            check_string(cmd, w.run)
        assert exc.value.rule == RULE, cmd


# ---- the search limit never lengthens the ordinary one ----------------------------------------------------------------------------

def test_a_search_limit_longer_than_the_ordinary_limit_changes_nothing(run):
    t = tools(run, timeout=0.5, search_timeout=60.0)
    started = time.monotonic()
    out = shell(t, 1, "grep -R zzz . ; sleep 20")
    assert time.monotonic() - started < 8
    assert out.startswith("killed after 0.5 s") and "searches a whole tree" not in out and t.search_timeouts == []


# ---- the run's read roots reach the shell tool through the supervisor ------------------------------------------------------------

def _supervisor_with(tmp_path, paths, inputs):
    from orchard.supervisor import Supervisor
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir = tmp_path / "run"
    sup.run_dir.mkdir(exist_ok=True)
    sup.paths, sup.inputs, sup.extra_env, sup.tool_dedupe = paths, inputs, {}, True
    return sup


def test_the_supervisors_agent_tools_carry_the_runs_read_roots_and_the_search_limit(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    hf = tmp_path / "hf"
    (hf / "hub").mkdir(parents=True)
    paths = SimpleNamespace(orchard_dir=str(tmp_path / "orch"), hf_home=str(hf), tt_model_root=str(tmp_path / "ttm"),
                            cache_root=str(tmp_path / "cache"), operator_home=str(tmp_path / "opshome"))
    monkeypatch.delenv("ORCHARD_SEARCH_TIMEOUT_S", raising=False)
    sup = _supervisor_with(tmp_path, paths, {"model": str(model)})
    t = sup._agent_tools(2, sup.run_dir / "stages" / "2", [])
    real = {os.path.realpath(r) for r in t.read_roots}
    assert os.path.realpath(model) in real and os.path.realpath(hf / "hub") in real and os.path.realpath(hf) not in real
    from orchard import defaults
    assert t.search_timeout == defaults.SEARCH_TIMEOUT_S
    monkeypatch.setenv("ORCHARD_SEARCH_TIMEOUT_S", "45")
    assert sup._agent_tools(2, sup.run_dir / "stages" / "2", []).search_timeout == 45.0


@pytest.mark.parametrize("text,expected", [("45", 45.0), ("0.5", 0.5), (" 30 ", 30.0), ("0", None), ("-5", None), ("nan", None), ("inf", None),
                                           ("1e3", None), ("", None), ("abc", None), ("10" * 20, None), ("5 seconds", None)])
def test_the_search_limit_from_the_environment_is_a_plain_positive_number_or_the_default(text, expected):
    from orchard import defaults
    from orchard.agent import search_timeout_from_env
    assert search_timeout_from_env({"ORCHARD_SEARCH_TIMEOUT_S": text}) == (defaults.SEARCH_TIMEOUT_S if expected is None else expected)
    assert search_timeout_from_env({}) == defaults.SEARCH_TIMEOUT_S


def test_a_relative_entry_of_the_read_roots_variable_is_dropped_even_when_it_names_a_directory_here(tmp_path, monkeypatch):
    (tmp_path / "reldir").mkdir()
    monkeypatch.chdir(tmp_path)
    paths = SimpleNamespace(orchard_dir=str(tmp_path / "x"), hf_home=str(tmp_path / "y"), tt_model_root=str(tmp_path / "z"),
                            cache_root=str(tmp_path / "w"), operator_home=str(tmp_path))
    assert agent_read_roots(paths, {}, {"ORCHARD_AGENT_READ_ROOTS": "reldir"}) == ()


# ---- gaps the mutation table found ------------------------------------------------------------------------------------------------

def test_a_glob_that_starts_inside_is_judged_by_each_match(w):
    os.symlink(w.home, w.run / "sub" / "linkhome")
    with pytest.raises(Denied) as exc:
        check_string(f"grep -r x {w.run}/sub/*", w.run)
    assert exc.value.rule == RULE and "linkhome" in exc.value.detail and str(w.home) in exc.value.detail
    check_string(f"grep -r x {w.run}/sub/a*", w.run)           # a glob whose matches all stay inside is fine


@pytest.mark.parametrize("cmd", ["fd x --search-path {home}", "fd x --search-path={home}", "fd x --base-directory {home}", "fd x . {home}"])
def test_fd_start_points_given_as_options_are_judged_too(cmd, w):
    with pytest.raises(Denied) as exc:
        check_string(fill(cmd, w), w.run)
    assert exc.value.rule == RULE


def test_a_link_to_another_allowed_place_is_followed_by_the_walk_and_that_places_links_are_judged_too(w):
    (w.snap / "inner").mkdir()
    os.symlink(w.home, w.snap / "inner" / "out")              # a link out of the read-only input
    os.symlink(w.snap, w.run / "sub" / "tosnap")              # a link from the run directory into it (the input is an allowed place)
    with pytest.raises(Denied) as exc:
        check_string(f"grep -R x {w.run}/sub", w.run, read_roots=(str(w.snap),))
    assert "out" in exc.value.detail and str(w.home) in exc.value.detail
    (w.snap / "inner" / "out").unlink()
    check_string(f"grep -R x {w.run}/sub", w.run, read_roots=(str(w.snap),))      # without the link out it is fine


def test_check_argv_takes_read_roots_like_check_string(w):
    from orchard.runner import check_argv
    argv = ["grep", "-R", "x", str(w.snap)]
    with pytest.raises(Denied):
        check_argv(argv, w.run, str(w.run))
    check_argv(argv, w.run, str(w.run), read_roots=(str(w.snap),))
    check_argv(argv, w.run, str(w.run), str(w.snap))            # a lone path is read as one root


def test_the_agent_step_builds_its_tools_through_the_method_that_carries_the_read_roots():
    # A source-level check of the call site: driving a whole step needs the supervisor's full test rig, and the method itself is tested above.
    import inspect
    from orchard.supervisor import Supervisor
    src = inspect.getsource(Supervisor._step)
    assert "tools=self._agent_tools(" in src
    assert "Tools(" not in src.replace("self._agent_tools(", "")
