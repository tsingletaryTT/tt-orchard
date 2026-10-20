# Kernel attempt: op inventory and tt-lang kernel probe, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tt-orchard kernel-env`, `tt-orchard inventory` and `tt-orchard kernel-probe` to this repository, on the
branch `kernel-attempt`, so that the op inventory and the tt-lang kernel measurement work end to end.

**Architecture:** Three independent pieces. (1) An environment command that builds a tt-lang simulator venv and verifies
it by named diagnostic text. (2) An op inventory: a torch-side template traces a model's forward pass on the `meta`
device with `torch.export`, and a standard-library module maps the ops to a recorded dump of ttnn's op names. (3) A kernel
probe: a harness and check template run an agent-written kernel on the simulator, `orchard/kernelprobe.py` drives the
existing `AgentStep` against the local coder and re-checks every result itself, and an optional one-chip phase runs under
a gozer lease. A stall watcher (progress signals, with a deadline only as a backstop) guards every long child process.

**Tech Stack:** Python 3.12 standard library for `orchard/` modules; torch and transformers only inside template scripts
that run in the reference venv; pytest; tt-lang-sim 1.1.10+g9b7ad24 in `~/orchard-venvs/ttlang`.

**Spec:** `docs/superpowers/specs/2026-10-15-kernel-attempt-design.md` (read it first; this plan implements it and amends
two details, listed under "Amendments to the spec" at the end of Task 12).

## Global Constraints

- Work only in the worktree `/home/ttuser/code/tt-orchard/.claude/worktrees/kernel-attempt` on branch `kernel-attempt`.
  Never `cd` to `/home/ttuser/code/tt-orchard`; a `tt-orchard ui` server runs from that checkout.
- Never use `git stash`. Commit work in progress instead.
- `orchard/opinventory.py`, `orchard/stallwatch.py`, `orchard/ttlang_env.py`, `orchard/kernelprobe.py` import only the
  standard library and `orchard.*`.
- Template scripts under `orchard/skills/*-templates/` are standalone files (they cannot import `orchard`). They are
  copied into a working directory and run by an interpreter named in a config file.
- Never `pip install` into `~/.tenstorrent-venv`. Never `import ttnn` outside a gozer lease (the import opens the device).
  Tests run with no network and no device: the `stub_tools` fixture from `tests/conftest.py` puts stub `gozer`, `docker`,
  `tt-model`, `git`, `hf`, `curl` and others first on PATH; a test that must call a real tool says so.
- Tests that need torch and transformers skip with a message that starts `SKIPPED:` and says what was not run (pattern:
  `tests/test_reference_gate_template.py`). The reference venv `~/orchard-venvs/reference/bin/python` has both; run those
  tests with `ORCHARD_REFERENCE_PYTHON=~/orchard-venvs/reference/bin/python` set (Task 4 defines the variable).
- Thresholds not yet measured are labelled `provisional` in the code, the JSON and every report: PCC bars 0.999 (simulator)
  and 0.99 (hardware, bf16); 40 turns, 1800 s wall time and 300 s per simulator run per attempt; hardware run backstop 600 s.
- Test first, watch it fail, then implement. After a task's tests pass, run each listed mutation: change the code, confirm a
  named test goes red, restore. Clear `__pycache__` before every mutation run (a stale `.pyc` once hid a red result).
  Check the baseline is green before reading a mutation table.
- Plain English in docs, docstrings and commit messages (global CLAUDE.md "Writing for people"). Before each commit, grep
  the text you wrote for `, not `, `rather than`, `instead of`, `reads as`, `honest`, `the one`, `which is why`,
  `This is what`, and rewrite each hit.
- Comments explain why. Keep them true as the code changes.
- Commit after each task: `git add <files>; git commit -m "<message>"`. No `--no-verify`, no `core.hooksPath` overrides.
- The default branch is `main`. Do not push, publish, upload, open PRs, or touch tt-model-manager or tt-lang sources.

## Execution protocol (applies to the controller and to every subagent)

- **No loops.** A subagent that cannot get a test to pass after three focused attempts stops and reports `BLOCKED` with the
  failing output. It does not keep retrying. It never repeats the same command more than twice without changing something.
- **Watchers, not timers.** Nobody sleeps and polls. To wait for something, use the Monitor tool with an `until` loop on a
  file or process state, or a background command whose exit notifies. Long hardware or model runs use
  `orchard.stallwatch` (Task 3) once it exists.
- **Bad behaviour stops work.** The controller stops and reassesses a subagent that repeats a tool call, writes outside its
  task's files, edits a test to make it pass without saying why, skips a listed mutation, or reports success without
  command output.
- **Hardware.** Only the controller runs anything that touches `/dev/tenstorrent/*`, always under `gozer run` with
  `--who claude:orchard-kernel`. Subagents never lease, reset or open a device.
- Report format for every task: files changed, test command and its exact pass/fail counts, the mutation table (mutation,
  test that went red, restored yes/no), and anything unverified.

## Review Focus

Inputs and conditions the spec implies and no task's happy path exercises, most likely first. Each has a test in the owning
task.

1. `config.json` has `auto_map` (remote code): the template writes `skipped`, exits 0, and never imports the repo's code;
   a missing `config.json` exits 2. (Task 4)
2. A ttnn dump taken for another ttnn version than the installed one: every op is `unchecked`, never `mapped`. (Task 5)
3. An agent that edits `kernel_check.py`, `kernel_harness.py` or `check.json` to fake a pass: the probe re-runs the shipped
   templates itself, and the attempt records `agent_check_untrusted` when the two disagree. (Task 9)
4. A child process that hangs or spins: the stall watcher kills its process group (grandchildren too), the category is
   `hang` or `timeout`, and nothing is left running. (Tasks 3 and 9)
5. A model whose config has no recognisable layer count or nests it: `layers_in_config` is null, the scaled count equals the
   traced count, and the report says so. (Tasks 4 and 5)
6. `kernel-env` finds `--src-dir` already holding another tt-lang commit or a non-git directory: it fails with a message and
   changes nothing. (Task 1)

---

## File structure

| Path | Responsibility |
| --- | --- |
| `orchard/defaults.py` (modify) | tt-lang pins and the negative-example table |
| `orchard/ttlang_env.py` (new) | build and verify the simulator environment |
| `orchard/stallwatch.py` (new) | `StallWatcher`, `tree_cpu_ticks`, `run_watched` |
| `orchard/skills/ops-inventory-templates/ops_inventory.py` (new) | torch side: trace on meta, write `raw-ops.json` |
| `orchard/opinventory.py` (new) | statuses, scaling, rendering, `inventory` command logic |
| `orchard/ttnn_dump.py` (new) | `refresh-ttnn`: dump ttnn op names under a lease |
| `orchard/data/aten_to_ttnn.json`, `decompositions.json`, `ttnn_ops-<version>.json` (new) | tables and the recorded dump |
| `orchard/skills/kernel-loop-templates/` (new) | `references.py`, `kernel_harness.py`, `kernel_check.py`, `ops/*.json`, `reference/eltwise_add.py` |
| `orchard/kernelprobe.py` (new) | attempts, authoritative re-check, hardware phase, report, provenance |
| `orchard/skills/kernel-probe.md` (new) | the agent's skill text |
| `orchard/cli.py` (modify) | subcommands `kernel-env`, `inventory`, `kernel-probe` |
| `README.md` (modify) | a row for every new flag (tests/test_readme.py requires it) |
| `orchard/skills/operator-bundle-templates/build_bundle.py` (modify) | show provenance in `RESULTS.md` |
| `tests/test_*.py`, `tests/fixtures/ops-inventory/`, `tests/fixtures/kernel-loop/` (new) | tests and recorded real outputs |

---

### Task 1: tt-lang pins and `orchard/ttlang_env.py`

**Files:**
- Modify: `orchard/defaults.py` (append the constants below)
- Create: `orchard/ttlang_env.py`
- Test: `tests/test_ttlang_env.py`

**Interfaces:**
- Produces: `defaults.TTLANG_URL`, `TTLANG_PIN`, `TTLANG_VERSION_TAG`, `TTLANG_DIAGNOSTICS_SHA256`,
  `TTLANG_NEGATIVE_EXAMPLES`; in `ttlang_env`: `Result(name, status, detail)`, `KEnv(src_dir, venv_dir, scratch_dir, run,
  environ)`, `ensure(env, *, check_only=False, say=print) -> list[Result]`, `verify(env) -> Result`, `expected_version()`,
  `main(args, *, env=None, out=None, run=None) -> int`, status constants `OK`, `TODO`, `FAIL`.

- [ ] **Step 1: Add the constants** to the end of `orchard/defaults.py`, in that file's style (each constant has a comment
  that says why):

```python
# ---- tt-lang, for `tt-orchard kernel-env` and `kernel-probe` (spec 2026-10-15-kernel-attempt-design.md) ----
TTLANG_URL = "https://github.com/tenstorrent/tt-lang.git"
TTLANG_PIN = "9b7ad24d1bd6a89dc1f8f885686bc4db86ac7244"   # choice: v1.1.10 plus 23 commits, read 2026-10-15
TTLANG_VERSION_TAG = "1.1.10"                             # the tag TTLANG_PIN is counted from
# sha256 of python/ttl/diagnostics.py at TTLANG_PIN. The tt-lang-sim wheel looks for this file at
# site-packages/ttl/ttl/diagnostics.py and does not ship it, so a diagnosed kernel error ends as
# FileNotFoundError. kernel-env copies the file there, and refuses to if the file at the pin has changed.
TTLANG_DIAGNOSTICS_SHA256 = "e3964f222cc5dc253157f97645a914f8751ab613f0fa82eb334e97046da8f84f"
# (example path under tt-lang's examples/, text that must appear in the simulator's output). A bare
# non-zero exit proves nothing here: the wheel's bug also ends these examples with a non-zero exit.
TTLANG_NEGATIVE_EXAMPLES = (
    ("errors/eltwise_add_error.py", ("does not match Block shape",)),
    ("compiler_only_external_call.py", ("ttl.call_extern_func is not supported by the Python simulator",
                                        "use tt-lang-sim --backend=emule")),
)
```

  If `tests/test_defaults.py` checks that every constant is documented or labelled `choice:`, follow its rule.

- [ ] **Step 2: Write the failing tests** in `tests/test_ttlang_env.py`. Start with the fake runner and the first four tests;
  add the rest from the list below the code.

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tt-orchard kernel-env: the tt-lang simulator environment. A fake runner stands in for git, uv and the venv's
python, and the files live in tmp_path, so nothing here touches the network, the shared venv or a device."""
from __future__ import annotations

from pathlib import Path

import pytest

from orchard import defaults, ttlang_env as ke

DIAG = b"# fake diagnostics\n"        # the sha256 of this stands in for the pinned file


class FakeRun:
    """Records every call. A rule is (argv prefix, (rc, out, err), effect); the first matching rule answers.
    An argv with no rule answers 127, so a command the test did not expect is visible."""

    def __init__(self):
        self.calls, self.rules = [], []

    def on(self, prefix, result=(0, "", ""), effect=None):
        self.rules.append((tuple(prefix), result, effect))

    def __call__(self, argv, timeout, env=None):
        self.calls.append((list(argv), dict(env or {})))
        for prefix, result, effect in self.rules:
            if tuple(argv[:len(prefix)]) == prefix:
                if effect:
                    effect(argv)
                return result
        return 127, "", f"no rule for {argv}"

    def argvs(self):
        return [a for a, _ in self.calls]

    def changed_something(self):
        """True when any call could change the machine: git init/remote/fetch/checkout, uv venv/build/pip."""
        for a in self.argvs():
            if a[0] == "git" and any(w in a for w in ("init", "remote", "fetch", "checkout")):
                return True
            if a[0] == "uv":
                return True
        return False


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", ke.hashlib.sha256(DIAG).hexdigest())
    return ke.KEnv(src_dir=tmp_path / "src" / "tt-lang", venv_dir=tmp_path / "venv", scratch_dir=tmp_path / "scratch",
                   run=FakeRun(), environ={"PATH": "/usr/bin"})


def test_check_only_never_changes_the_machine(env):
    results = ke.ensure(env, check_only=True, say=lambda s: None)
    assert not env.run.changed_something()
    assert results[0].name == "clone" and results[0].status == ke.TODO


def test_a_clone_at_another_commit_fails_and_is_left_alone(env):
    (env.src_dir / ".git").mkdir(parents=True)
    env.run.on(["git", "-C", str(env.src_dir), "rev-parse", "HEAD"], (0, "deadbeef" * 5 + "\n", ""))
    results = ke.ensure(env, say=lambda s: None)
    assert results[-1].status == ke.FAIL and "is not reset" in results[-1].detail
    assert not env.run.changed_something()


def test_a_directory_that_is_not_a_checkout_fails_and_is_left_alone(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / "notes.txt").write_text("mine")
    results = ke.ensure(env, say=lambda s: None)
    assert results[-1].status == ke.FAIL and "not a git checkout" in results[-1].detail
    assert not env.run.changed_something()
    assert (env.src_dir / "notes.txt").read_text() == "mine"


def test_verify_needs_the_named_text_and_not_just_a_failing_exit(env):
    """The wheel bug ends upstream's negative examples with a non-zero exit and FileNotFoundError. That must not pass."""
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / "bin").mkdir()
    env.sim.write_text("")
    env.run.on([str(env.sim)], (1, "", "FileNotFoundError: [Errno 2] ... ttl/ttl/diagnostics.py"))
    got = ke.verify(env)
    assert got.status == ke.FAIL
```

  Further tests, each with the assertion written out in code before its implementation exists:
  - `test_verify_passes_when_every_named_text_is_present_and_the_exit_is_non_zero` (rc 1, output holds every needle for
    both examples, no `FileNotFoundError`).
  - `test_verify_fails_when_the_text_is_present_but_filenotfounderror_is_too` (the fallback text appearing alongside the
    needle still fails: a partial fix is not a fix).
  - `test_the_diagnostics_copy_is_refused_when_the_pinned_file_changed` (the source file's sha256 differs from
    `defaults.TTLANG_DIAGNOSTICS_SHA256`: status FAIL, target not written).
  - `test_the_diagnostics_copy_lands_at_site_packages_ttl_ttl` (fake `purelib` answered by a rule on
    `[str(env.py), "-c", ...]` returning `tmp_path/"site"`; afterwards `site/ttl/ttl/diagnostics.py` holds `DIAG`).
  - `test_the_wheel_is_built_with_the_version_override_in_its_environment` (the `uv build` call's env has
    `TTLANG_VERSION_OVERRIDE == "1.1.10+g9b7ad24"`, i.e. `ke.expected_version()`).
  - `test_torch_comes_from_the_cpu_index_before_the_wheel` (the `uv pip install ... torch` argv contains
    `--index-url` and `ke.TORCH_CPU_INDEX`, and appears before the wheel install).
  - `test_ensure_stops_at_the_first_failure` (a failing clone means no `uv` call).
  - `test_a_second_run_does_nothing_when_everything_is_in_place` (all probes answer done: only read-only calls).

- [ ] **Step 3: Run the tests; confirm they fail** because `orchard.ttlang_env` does not exist:
  `python3 -m pytest tests/test_ttlang_env.py -q` → collection error / ImportError.

- [ ] **Step 4: Implement `orchard/ttlang_env.py`.** Full module:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard kernel-env`: build and check the tt-lang simulator environment that `kernel-probe` uses.

What it makes, outside the repository:

    <src-dir>    a shallow checkout of tt-lang at defaults.TTLANG_PIN, no submodules (docs, examples, packaging)
    <venv-dir>   a venv with CPU torch and the tt-lang-sim wheel built from <src-dir> (the PyPI release lags the
                 source). It has no ttnn, so it cannot open a device.

The wheel has a packaging bug: ttl/sim/diagnostics.py loads sim_dir.parent / "ttl" / "diagnostics.py", a source-tree
path, and the wheel does not ship that file. A diagnosed kernel error then ends as FileNotFoundError and the
diagnostic text is lost. Step "diagnostics" copies the checkout's file to site-packages/ttl/ttl/diagnostics.py. This is
a local patch of a vendor package. It is not reported upstream (operator decision, 2026-10-15) and it is labelled as
a patch wherever the probe reports tt-lang versions.

Step "verify" is the instrument check for the whole probe. It runs two upstream examples that must fail and requires
the diagnostic text upstream's own tests look for. A non-zero exit alone passes with the bug present, so it counts for
nothing.

Safety: it never touches the shared venv (~/.tenstorrent-venv); it never resets or deletes a checkout that is at another
commit or is not a git checkout; `--check` runs nothing that changes the machine.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orchard import defaults

OK, TODO, FAIL = "ok", "todo", "fail"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
FALLBACK_TEXT = "FileNotFoundError"      # what the missing diagnostics file turns a diagnosed error into


@dataclass
class Result:
    name: str
    status: str
    detail: str


def real_run(argv, timeout, env=None):
    """(returncode, stdout, stderr). A timeout answers 124 and a missing program 127."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout} s"
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


@dataclass
class KEnv:
    src_dir: Path
    venv_dir: Path
    scratch_dir: Path
    run: Callable = real_run          # (argv, timeout, env=None) -> (returncode, stdout, stderr)
    environ: dict | None = None

    @property
    def py(self) -> Path:
        return self.venv_dir / "bin" / "python"

    @property
    def sim(self) -> Path:
        return self.venv_dir / "bin" / "tt-lang-sim"


def expected_version() -> str:
    """The version label the built wheel carries: the tag, plus the commit it was built from."""
    return f"{defaults.TTLANG_VERSION_TAG}+g{defaults.TTLANG_PIN[:7]}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def head_sha(env: KEnv) -> str | None:
    if not (env.src_dir / ".git").exists():
        return None
    rc, out, _ = env.run(["git", "-C", str(env.src_dir), "rev-parse", "HEAD"], 30)
    return out.strip() if rc == 0 and out.strip() else None


def purelib(env: KEnv) -> Path | None:
    rc, out, _ = env.run([str(env.py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], 30)
    return Path(out.strip()) if rc == 0 and out.strip() else None


def diagnostics_target(env: KEnv) -> Path | None:
    """Where the tt-lang-sim wheel looks for diagnostics.py. The wheel does not ship the file."""
    lib = purelib(env)
    return lib / "ttl" / "ttl" / "diagnostics.py" if lib else None


def sim_version(env: KEnv) -> str | None:
    if not env.sim.exists():
        return None
    rc, out, _ = env.run([str(env.sim), "--version"], 60)
    return out.split()[-1] if rc == 0 and out.strip() else None


def _fail(name: str, argv, err: str) -> Result:
    return Result(name, FAIL, f"`{' '.join(str(a) for a in argv[:5])}` failed: {err.strip()[-300:]}")


def step_clone(env: KEnv, check_only: bool) -> Result:
    have, pin, src = head_sha(env), defaults.TTLANG_PIN, str(env.src_dir)
    if have == pin:
        return Result("clone", OK, f"{env.src_dir} at {pin[:9]}")
    if have is not None:
        return Result("clone", FAIL, f"{env.src_dir} is at {have[:9]} and the pin is {pin[:9]}; it is not reset. "
                                     "Move it aside or pass another --src-dir")
    if env.src_dir.exists() and any(env.src_dir.iterdir()):
        return Result("clone", FAIL, f"{env.src_dir} exists and is not a git checkout; it is left alone")
    if check_only:
        return Result("clone", TODO, f"would fetch tt-lang {pin[:9]} into {env.src_dir}")
    env.src_dir.mkdir(parents=True, exist_ok=True)
    # A fetch of one commit, because main moves and the pin must not.
    for argv in (["git", "-C", src, "init", "--quiet"],
                 ["git", "-C", src, "remote", "add", "origin", defaults.TTLANG_URL],
                 ["git", "-C", src, "fetch", "--quiet", "--depth", "1", "origin", pin],
                 ["git", "-C", src, "checkout", "--quiet", "FETCH_HEAD"]):
        rc, _, err = env.run(argv, 900)
        if rc != 0:
            return _fail("clone", argv, err)
    return Result("clone", OK, f"fetched {pin[:9]} into {env.src_dir}")


def step_venv(env: KEnv, check_only: bool) -> Result:
    if env.py.exists():
        return Result("venv", OK, str(env.venv_dir))
    if check_only:
        return Result("venv", TODO, f"would create {env.venv_dir}")
    argv = ["uv", "venv", "--quiet", "--python", "3.12", str(env.venv_dir)]
    rc, _, err = env.run(argv, 300)
    return Result("venv", OK, f"created {env.venv_dir}") if rc == 0 else _fail("venv", argv, err)


def step_torch(env: KEnv, check_only: bool) -> Result:
    rc, _, _ = env.run([str(env.py), "-c", "import torch"], 120)
    if rc == 0:
        return Result("torch", OK, "importable")
    if check_only:
        return Result("torch", TODO, "would install CPU torch")
    # CPU torch first and alone: the CPU index carries old copies of certifi and requests, and uv ranks an
    # --extra-index-url above PyPI, so nothing else may resolve through it (README, "setup").
    argv = ["uv", "pip", "install", "--quiet", "--python", str(env.py), "--index-url", TORCH_CPU_INDEX, "torch"]
    rc, _, err = env.run(argv, 1800)
    return Result("torch", OK, "installed CPU torch") if rc == 0 else _fail("torch", argv, err)


def step_wheel(env: KEnv, check_only: bool) -> Result:
    want = expected_version()
    have = sim_version(env)
    if have == want:
        return Result("tt-lang-sim", OK, f"{have} (built from {defaults.TTLANG_PIN[:9]})")
    if check_only:
        return Result("tt-lang-sim", TODO, f"have {have or 'none'}, want {want}")
    wheels = env.scratch_dir / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    build_env = dict(env.environ if env.environ is not None else os.environ, TTLANG_VERSION_OVERRIDE=want)
    argv = ["uv", "build", "--wheel", "--out-dir", str(wheels), str(env.src_dir / "packaging" / "sim")]
    rc, _, err = env.run(argv, 900, build_env)
    if rc != 0:
        return _fail("tt-lang-sim", argv, err)
    built = sorted(wheels.glob(f"tt_lang_sim-{want}-*.whl"))
    if not built:
        return Result("tt-lang-sim", FAIL, f"the build ran and left no tt_lang_sim-{want} wheel in {wheels}")
    argv = ["uv", "pip", "install", "--quiet", "--python", str(env.py), str(built[-1])]
    rc, _, err = env.run(argv, 600)
    return Result("tt-lang-sim", OK, f"installed {built[-1].name}") if rc == 0 else _fail("tt-lang-sim", argv, err)


def step_diagnostics(env: KEnv, check_only: bool) -> Result:
    target = diagnostics_target(env)
    if target is None:
        return Result("diagnostics", FAIL, f"{env.py} did not say where its site-packages is")
    source = env.src_dir / "python" / "ttl" / "diagnostics.py"
    if not source.is_file():
        return Result("diagnostics", FAIL, f"{source} does not exist")
    if sha256_file(source) != defaults.TTLANG_DIAGNOSTICS_SHA256:
        return Result("diagnostics", FAIL, "diagnostics.py at the pin is not the file the patch was checked against; "
                                           "review it before changing defaults.TTLANG_DIAGNOSTICS_SHA256")
    if target.is_file() and sha256_file(target) == defaults.TTLANG_DIAGNOSTICS_SHA256:
        return Result("diagnostics", OK, f"{target} (local patch of the tt-lang-sim wheel)")
    if check_only:
        return Result("diagnostics", TODO, f"would copy diagnostics.py to {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return Result("diagnostics", OK, f"copied to {target} (local patch of the tt-lang-sim wheel)")


def verify(env: KEnv) -> Result:
    """Run the negative examples and require upstream's own diagnostic text. See the module docstring."""
    for rel, needles in defaults.TTLANG_NEGATIVE_EXAMPLES:
        rc, out, err = env.run([str(env.sim), str(env.src_dir / "examples" / rel)], 300)
        text = f"{out}\n{err}"
        if rc == 0:
            return Result("verify", FAIL, f"{rel} exited 0 but it must fail")
        if FALLBACK_TEXT in text:
            return Result("verify", FAIL, f"{rel} ended in {FALLBACK_TEXT}: the diagnostics file is still missing")
        missing = [n for n in needles if n not in text]
        if missing:
            return Result("verify", FAIL, f"{rel} did not print {missing[0]!r}")
    return Result("verify", OK, f"{len(defaults.TTLANG_NEGATIVE_EXAMPLES)} negative examples print their diagnostics")


def ensure(env: KEnv, *, check_only: bool = False, say: Callable = print) -> list[Result]:
    """Run the steps in order and stop at the first failure. With check_only nothing changes the machine."""
    results: list[Result] = []
    for step in (step_clone, step_venv, step_torch, step_wheel, step_diagnostics):
        r = step(env, check_only)
        results.append(r)
        say(f"{r.status:5s} {r.name}: {r.detail}")
        if r.status == FAIL:
            return results
    if any(r.status == TODO for r in results):
        r = Result("verify", TODO, "skipped until the steps above are done")
    else:
        r = verify(env)
    results.append(r)
    say(f"{r.status:5s} {r.name}: {r.detail}")
    return results


def main(args, *, env: KEnv | None = None, out=None, run=None) -> int:
    """`tt-orchard kernel-env [--check] [--src-dir D] [--venv-dir D]`. Exit 0 when every step is ok, 2 when a step
    failed or (with --check) is still to do."""
    say = (lambda line: print(line, file=out)) if out is not None else print
    home = Path.home()
    env = env or KEnv(
        src_dir=Path(args.src_dir or os.environ.get("ORCHARD_TTLANG_SRC") or home / "orchard-src" / "tt-lang"),
        venv_dir=Path(args.venv_dir or os.environ.get("ORCHARD_TTLANG_VENV") or home / "orchard-venvs" / "ttlang"),
        scratch_dir=Path(os.environ.get("ORCHARD_SCRATCH") or home / "orchard-src" / "scratch"),
        run=run or real_run, environ=dict(os.environ))
    results = ensure(env, check_only=args.check, say=say)
    return 0 if all(r.status == OK for r in results) else 2
```

- [ ] **Step 5: Run the tests; confirm they pass:** `python3 -m pytest tests/test_ttlang_env.py -q`. Also run
  `python3 -m pytest tests/test_defaults.py -q`.

- [ ] **Step 6: Mutations** (one at a time, clear `__pycache__`, expect the named test red, restore):
  1. `step_diagnostics` returns OK without copying → `test_the_diagnostics_copy_lands_at_site_packages_ttl_ttl`.
  2. `verify` drops the `FALLBACK_TEXT` check → `test_verify_fails_when_the_text_is_present_but_filenotfounderror_is_too`.
  3. `verify` checks only `rc != 0` → `test_verify_needs_the_named_text_and_not_just_a_failing_exit`.
  4. `step_clone` runs `git checkout` even when `have` is another SHA → `test_a_clone_at_another_commit_fails_and_is_left_alone`.
  5. `ensure(check_only=True)` still calls `step_clone` with `check_only=False` → `test_check_only_never_changes_the_machine`.
  6. Drop `TTLANG_VERSION_OVERRIDE` from the build env → `test_the_wheel_is_built_with_the_version_override_in_its_environment`.
  7. Torch installed without `--index-url` → `test_torch_comes_from_the_cpu_index_before_the_wheel`.
  8. Skip the sha256 check on the pinned source → `test_the_diagnostics_copy_is_refused_when_the_pinned_file_changed`.

- [ ] **Step 7: Commit.** `git add orchard/defaults.py orchard/ttlang_env.py tests/test_ttlang_env.py` then
  `git commit -m "kernel-env: build and verify the tt-lang simulator environment"`.

---

### Task 2: the `kernel-env` command and its README rows

**Files:**
- Modify: `orchard/cli.py` (a subparser in `_parser()`, a dispatch branch in `main`, the module docstring's command list)
- Modify: `README.md` (rows for `kernel-env`, `--check`, `--src-dir`, `--venv-dir`)
- Test: `tests/test_cli.py` (add tests), `tests/test_readme.py` must pass unchanged

**Interfaces:**
- Consumes: `ttlang_env.main(args, *, env=None, out=None, run=None) -> int`.
- Produces: `tt-orchard kernel-env [--check] [--src-dir DIR] [--venv-dir DIR]`.

- [ ] **Step 1: Read** `tests/test_readme.py` lines 110-230 to see how the README must list subcommand flags, and
  `tests/test_cli.py` for how `main` is called with an injected `stdout`.

- [ ] **Step 2: Write the failing tests** in `tests/test_cli.py`:

```python
def test_kernel_env_check_runs_the_check_and_returns_its_exit_code(tmp_path, monkeypatch):
    from orchard import cli, ttlang_env
    seen = {}

    def fake_main(args, *, env=None, out=None, run=None):
        seen["check"], seen["src"], seen["venv"] = args.check, args.src_dir, args.venv_dir
        return 2

    monkeypatch.setattr(ttlang_env, "main", fake_main)
    out = io.StringIO()
    code = cli.main(["kernel-env", "--check", "--src-dir", str(tmp_path / "s"), "--venv-dir", str(tmp_path / "v")],
                    stdout=out)
    assert code == 2 and seen == {"check": True, "src": str(tmp_path / "s"), "venv": str(tmp_path / "v")}


def test_kernel_env_needs_no_bringup_config(tmp_path, monkeypatch):
    """The command must work on a machine with no bringup.toml: it is how a machine gets ready."""
    from orchard import cli, ttlang_env
    monkeypatch.setattr(ttlang_env, "main", lambda args, **kw: 0)
    monkeypatch.setenv("ORCHARD_BRINGUP_CONFIG", str(tmp_path / "does-not-exist.toml"))
    assert cli.main(["kernel-env", "--check"], stdout=io.StringIO()) == 0
```

  (`io` is already imported in that test file; if not, add the import.) Run: both fail with `invalid choice: 'kernel-env'`.

- [ ] **Step 3: Implement.** In `_parser()`, after the `coder` subparser:

```python
    k = sub.add_parser("kernel-env", parents=[common],
                       help="build or check the tt-lang simulator environment that kernel-probe uses")
    k.add_argument("--check", action="store_true",
                   help="change nothing; report each step and exit 2 if one is missing or failed")
    k.add_argument("--src-dir", help="where the tt-lang checkout lives (default ~/orchard-src/tt-lang, or "
                                     "$ORCHARD_TTLANG_SRC)")
    k.add_argument("--venv-dir", help="where the simulator venv lives (default ~/orchard-venvs/ttlang, or "
                                      "$ORCHARD_TTLANG_VENV)")
```

  In `main`, before `if args.cmd == "ui":`:

```python
    if args.cmd == "kernel-env":
        from orchard import ttlang_env
        return ttlang_env.main(args, out=out)
```

  Add `tt-orchard kernel-env [--check] [--src-dir DIR] [--venv-dir DIR]` to the docstring's command list.

- [ ] **Step 4: README.** Add a short subsection "Kernel environment" next to the other subcommand sections, with the
  command, what it builds, the diagnostics patch in two sentences, and a flag table row for each of `--check`,
  `--src-dir`, `--venv-dir` in the style of the existing tables. Run `python3 -m pytest tests/test_readme.py tests/test_cli.py -q`.

- [ ] **Step 5: Mutations:** dispatch branch removed → first test red; `--check` not passed through → first test red;
  the command made to load `bringup.toml` first → second test red.

- [ ] **Step 6: Commit** `git add orchard/cli.py README.md tests/test_cli.py` then
  `git commit -m "kernel-env: the tt-orchard subcommand and README rows"`.

---

### Task 3: `orchard/stallwatch.py`, watchers instead of timers

**Files:**
- Create: `orchard/stallwatch.py`
- Test: `tests/test_stallwatch.py`

**Interfaces:**
- Produces: `StallWatcher(quiet_s=60.0, min_cpu_ticks=2, repeat_lines=20)` with `observe(now, out_bytes, cpu_ticks) -> str | None`
  and `feed_lines(lines) -> str | None`; `tree_cpu_ticks(pid, proc_root="/proc") -> int`;
  `Watched(returncode, reason, killed, elapsed_s)`;
  `run_watched(argv, *, cwd, env, log_path, watcher, backstop_s, poll_s=1.0, proc_root="/proc") -> Watched`.

- [ ] **Step 1: Write the failing tests** in `tests/test_stallwatch.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stall watcher: it kills on signs that nothing is happening, and a deadline is only its backstop."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from orchard.stallwatch import StallWatcher, run_watched, tree_cpu_ticks


def test_no_output_and_no_cpu_for_the_quiet_window_is_a_stall():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    assert w.observe(0.0, 100, 50) is None
    assert w.observe(9.0, 100, 50) is None
    assert "stalled" in w.observe(10.5, 100, 51)          # one tick of CPU is below min_cpu_ticks


def test_output_growth_counts_as_progress():
    w = StallWatcher(quiet_s=10)
    w.observe(0.0, 100, 50)
    assert w.observe(9.0, 101, 50) is None                # a byte of output resets the window
    assert w.observe(18.0, 101, 50) is None
    assert "stalled" in w.observe(19.5, 101, 50)


def test_cpu_use_counts_as_progress_and_slow_cpu_adds_up():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    w.observe(0.0, 100, 50)
    assert w.observe(9.0, 100, 52) is None                # two ticks since the mark: progress
    w2 = StallWatcher(quiet_s=10, min_cpu_ticks=3)
    w2.observe(0.0, 100, 50)
    w2.observe(4.0, 100, 51)
    w2.observe(8.0, 100, 52)
    assert w2.observe(9.0, 100, 53) is None               # one tick a sample still reaches 3 over the window


def test_a_repeating_line_is_reported():
    w = StallWatcher(repeat_lines=5)
    assert w.feed_lines(["compiling"] * 4) is None
    assert "repeating" in w.feed_lines(["compiling"])


def test_a_changed_line_resets_the_repeat_count():
    w = StallWatcher(repeat_lines=5)
    w.feed_lines(["a"] * 4)
    assert w.feed_lines(["b"]) is None
    assert w.feed_lines(["a"] * 4) is None


def test_tree_cpu_ticks_sums_a_process_and_its_descendants(tmp_path):
    def proc(pid, ppid, ut, st, comm="x"):
        d = tmp_path / str(pid)
        d.mkdir()
        # fields after the comm: state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 1 0 0 0 0 0 0 0 {ut} {st} 0 0 20 0 1 0 0 0 0\n")
    proc(100, 1, 10, 5)
    proc(101, 100, 3, 2, comm="a (b) c")                  # a command name with spaces and parentheses
    proc(102, 101, 1, 1)
    proc(200, 1, 999, 999)                                # not a descendant
    assert tree_cpu_ticks(100, proc_root=str(tmp_path)) == 10 + 5 + 3 + 2 + 1 + 1


def test_a_silent_idle_child_is_killed_as_stalled(tmp_path):
    w = StallWatcher(quiet_s=1.0, min_cpu_ticks=2)
    got = run_watched(["sleep", "60"], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "stalled" in got.reason and got.elapsed_s < 10


def test_a_busy_silent_child_is_not_a_stall_and_the_backstop_ends_it(tmp_path):
    w = StallWatcher(quiet_s=1.0, min_cpu_ticks=2)
    spin = [sys.executable, "-c", "while True: pass"]
    got = run_watched(spin, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=3, poll_s=0.2)
    assert got.killed and got.reason.startswith("backstop")


def test_a_child_that_keeps_printing_is_left_alone_until_it_exits(tmp_path):
    w = StallWatcher(quiet_s=1.0, repeat_lines=1000)
    cmd = [sys.executable, "-c", "import time\nfor i in range(8):\n print(i, flush=True); time.sleep(0.3)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert not got.killed and got.returncode == 0


def test_a_child_that_repeats_one_line_is_killed(tmp_path):
    w = StallWatcher(quiet_s=30, repeat_lines=10)
    cmd = [sys.executable, "-c", "import time\nwhile True:\n print('same', flush=True); time.sleep(0.05)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "repeating" in got.reason


def test_the_whole_process_group_is_killed_so_a_grandchild_does_not_survive(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    w = StallWatcher(quiet_s=1.0)
    cmd = ["sh", "-c", f"sleep 120 & echo $! > {pidfile}; wait"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed
    gone = False
    pid = int(pidfile.read_text())
    for _ in range(20):                      # the kernel needs a moment to reap; this waits on the pid, not a timer
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.1)
    assert gone
```

  Run: `python3 -m pytest tests/test_stallwatch.py -q` → ImportError.

- [ ] **Step 2: Implement `orchard/stallwatch.py`:**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Watch a child process for the signs that it is stuck, so a long run does not wait on a timer alone.

A timer says "this has taken long". These signs say "nothing is happening": the output has stopped growing and the
process tree has used no CPU, or one output line keeps repeating. A hard deadline stays as the last backstop, because
a process can be busy and still wrong (a spin that prints nothing).

`run_watched` samples about once a second. The sampling interval is not the decision. The decision comes from what the
child did: bytes written, CPU ticks used by it and everything it started, and the lines it printed.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


class StallWatcher:
    """Decides from samples whether a process has stopped making progress. No clock of its own: the caller passes `now`."""

    def __init__(self, *, quiet_s: float = 60.0, min_cpu_ticks: int = 2, repeat_lines: int = 20):
        self.quiet_s, self.min_cpu_ticks, self.repeat_lines = quiet_s, min_cpu_ticks, repeat_lines
        self._mark_at = self._mark_bytes = self._mark_ticks = None
        self._line, self._run = None, 0

    def observe(self, now: float, out_bytes: int, cpu_ticks: int) -> str | None:
        if self._mark_at is None:
            self._mark_at, self._mark_bytes, self._mark_ticks = now, out_bytes, cpu_ticks
            return None
        # Progress is measured against the last time progress was seen, so slow CPU use still adds up.
        if out_bytes > self._mark_bytes or cpu_ticks - self._mark_ticks >= self.min_cpu_ticks:
            self._mark_at, self._mark_bytes, self._mark_ticks = now, out_bytes, cpu_ticks
            return None
        quiet = now - self._mark_at
        if quiet >= self.quiet_s:
            return f"stalled: no output and no CPU use for {quiet:.0f} s"
        return None

    def feed_lines(self, lines) -> str | None:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            self._run = self._run + 1 if line == self._line else 1
            self._line = line
            if self._run >= self.repeat_lines:
                return f"repeating: {line[:80]!r} x {self._run}"
        return None


def tree_cpu_ticks(pid: int, proc_root: str = "/proc") -> int:
    """User plus system CPU ticks of `pid` and every descendant, read from /proc. 0 when the pid is gone."""
    stats: dict[int, tuple[int, int]] = {}            # pid -> (ppid, ticks)
    try:
        names = os.listdir(proc_root)
    except OSError:
        return 0
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, name, "stat"), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        # The command name sits in parentheses and may hold spaces and parentheses, so split after the last ")".
        rest = text[text.rfind(")") + 2:].split()
        try:
            stats[int(name)] = (int(rest[1]), int(rest[11]) + int(rest[12]))   # ppid; utime + stime
        except (IndexError, ValueError):
            continue
    children: dict[int, list[int]] = {}
    for p, (pp, _) in stats.items():
        children.setdefault(pp, []).append(p)
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        if p in stats:
            total += stats[p][1]
        todo.extend(children.get(p, []))
    return total


@dataclass
class Watched:
    returncode: int | None
    reason: str | None        # why it was killed, or None when it ended by itself
    killed: bool
    elapsed_s: float


def _kill_group(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)       # the child leads its own session, so this reaches grandchildren
    except (ProcessLookupError, PermissionError):
        pass


def run_watched(argv, *, cwd, env, log_path, watcher: StallWatcher, backstop_s: float, poll_s: float = 1.0,
                proc_root: str = "/proc", popen=subprocess.Popen, clock=time.monotonic) -> Watched:
    """Run `argv` with stdout and stderr appended to `log_path`, and stop it when the watcher says it is stuck or
    when `backstop_s` has passed. The child gets its own session so the whole group can be killed."""
    start, read_pos = clock(), 0
    with open(log_path, "ab") as log:
        proc = popen(list(argv), cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                return Watched(proc.wait(timeout=poll_s), None, False, clock() - start)
            except subprocess.TimeoutExpired:
                pass
            now = clock()
            size = os.path.getsize(log_path)
            with open(log_path, "rb") as f:
                f.seek(read_pos)
                fresh = f.read(size - read_pos).decode("utf-8", errors="replace").splitlines()
            read_pos = size
            reason = watcher.feed_lines(fresh) or watcher.observe(now, size, tree_cpu_ticks(proc.pid, proc_root))
            if reason is None and now - start >= backstop_s:
                reason = f"backstop: still running after {backstop_s:.0f} s"
            if reason is not None:
                _kill_group(proc)
                return Watched(proc.wait(), reason, True, clock() - start)
```

- [ ] **Step 3: Run:** `python3 -m pytest tests/test_stallwatch.py -q` → all pass (the process tests take a few seconds).
  If a test is flaky on CPU ticks, raise its `quiet_s`, not the assertion.

- [ ] **Step 4: Mutations:** `observe` ignoring `out_bytes` → `test_output_growth_counts_as_progress`; ignoring CPU →
  `test_cpu_use_counts_as_progress_and_slow_cpu_adds_up`; updating the mark every call → the stall tests; `feed_lines` never
  resetting on a changed line → `test_a_changed_line_resets_the_repeat_count`; `tree_cpu_ticks` counting only `pid` →
  `test_tree_cpu_ticks_sums_a_process_and_its_descendants`; `os.kill(proc.pid, ...)` instead of `killpg` →
  `test_the_whole_process_group_is_killed_so_a_grandchild_does_not_survive`; backstop removed → the busy-child test hangs
  until pytest's own timeout (run it with `timeout 60`).

- [ ] **Step 5: Commit** `git add orchard/stallwatch.py tests/test_stallwatch.py` then
  `git commit -m "stallwatch: kill on signs of a stall, with a deadline only as a backstop"`.


---

### Task 4: the torch-side template `ops_inventory.py` and real fixtures

**Files:**
- Create: `orchard/skills/ops-inventory-templates/ops_inventory.py`
- Create: `tests/fixtures/ops-inventory/make_fixtures.py` and its output `raw-qwen3-tiny.json`, `raw-rwkv.json`,
  `raw-mamba2.json`, `raw-moshi-lm.json`
- Test: `tests/test_ops_inventory_template.py`

**Interfaces:**
- Produces: the template's command line `python ops_inventory.py --model-dir DIR --out FILE [--layers 2] [--seq 8] [--class NAME]`
  and the `raw-ops.json` schema (below), which Task 5 consumes.

`raw-ops.json` (schema 1):

```json
{"schema": 1, "model_dir": "...", "architectures": ["Qwen3ForCausalLM"], "traced_class": "Qwen3ForCausalLM",
 "layers_in_config": 4, "layers_traced": 2, "seq_len": 8, "torch": "2.14.1+cpu", "transformers": "5.19.0",
 "note": "prefill graph from torch.export(strict=False, use_cache=False) on the meta device; cache and state updates of decoding are not traced",
 "ops": [{"op": "aten.linear.default", "count": 15, "count_in_layers": 14,
          "sample": {"output_shapes": [[1, 8, 32]], "dtype": "torch.float32"}, "modules": ["model.layers.0.self_attn.q_proj"]}],
 "failures": []}
```

or `{"schema": 1, ..., "skipped": "<reason>"}` with no `ops`, or `{..., "failures": [{"stage": "export", "error": "..."}]}`.
`count_in_layers` counts graph nodes whose module path has a `layers.N`, `blocks.N` or `h.N` segment.

Measured before this plan was written (2026-10-15, reference venv, transformers 5.19.0, default configs, 2 layers,
`input_ids` of shape (1, 8) on the meta device): `torch.export.export(model, (), {"input_ids": ids, "use_cache": False},
strict=False)` succeeded for Qwen3, RWKV, Mamba2 and `MoshiForCausalLM` (21 to 39 distinct ops). Without `use_cache=False` it failed
on three of the four with "Found DynamicCache in output". A dispatch-mode recorder was tried and failed on Moshi (`.item()` on a
meta tensor), so it is not used.

- [ ] **Step 1: Write the failing tests** in `tests/test_ops_inventory_template.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ops_inventory.py, the template `tt-orchard inventory` runs in the reference interpreter. Run end to end on a tiny
config; nothing is downloaded and no device is opened (the model lives on the meta device)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "ops-inventory-templates" / "ops_inventory.py"
QWEN3_TINY = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "vocab_size": 128, "hidden_size": 32,
              "intermediate_size": 64, "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2,
              "head_dim": 16, "max_position_embeddings": 64}


def stack_python():
    """An interpreter with torch and transformers: $ORCHARD_REFERENCE_PYTHON, else this one, else None."""
    for cand in (os.environ.get("ORCHARD_REFERENCE_PYTHON"), sys.executable):
        if cand and subprocess.run([cand, "-c", "import torch, transformers"], capture_output=True).returncode == 0:
            return cand
    return None


PY = stack_python()
needs_stack = pytest.mark.skipif(PY is None, reason=(
    "SKIPPED: no interpreter with torch and transformers (set ORCHARD_REFERENCE_PYTHON), so the ops-inventory "
    "template was not run end to end"))


def run_template(tmp_path, config: dict, *extra):
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(config))
    out = tmp_path / "raw.json"
    done = subprocess.run([PY or sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(out), *extra],
                          capture_output=True, text=True)
    return done, (json.loads(out.read_text()) if out.exists() else None)


def test_a_config_with_auto_map_is_skipped_and_no_code_runs(tmp_path):
    """Review focus 1. A repo's own modeling code is never imported. This test needs no torch."""
    done, raw = run_template(tmp_path, {**QWEN3_TINY, "auto_map": {"AutoModelForCausalLM": "modeling_x.X"}})
    assert done.returncode == 0 and "remote code" in raw["skipped"] and "ops" not in raw


def test_a_missing_config_exits_2(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    done = subprocess.run([sys.executable, str(TEMPLATE), "--model-dir", str(d), "--out", str(tmp_path / "o.json")],
                          capture_output=True, text=True)
    assert done.returncode == 2 and "config.json" in done.stderr


@needs_stack
def test_a_class_the_installed_transformers_lacks_is_skipped(tmp_path):
    done, raw = run_template(tmp_path, {**QWEN3_TINY, "architectures": ["NoSuchForCausalLM"]})
    assert done.returncode == 0 and "NoSuchForCausalLM" in raw["skipped"]


@needs_stack
def test_a_tiny_qwen3_traces_and_reports_layers_and_ops(tmp_path):
    done, raw = run_template(tmp_path, QWEN3_TINY, "--layers", "2")
    assert done.returncode == 0, done.stderr
    assert raw["layers_in_config"] == 4 and raw["layers_traced"] == 2 and raw["traced_class"] == "Qwen3ForCausalLM"
    ops = {o["op"]: o for o in raw["ops"]}
    assert len(ops) >= 10 and not raw["failures"]
    assert any(o["count_in_layers"] > 0 for o in ops.values())
    assert any(o["count_in_layers"] == 0 for o in ops.values()), "embedding and the final norm are outside the layers"
    assert all(0 <= o["count_in_layers"] <= o["count"] for o in ops.values())


def test_reduce_layers_reports_none_when_the_config_has_no_layer_count():
    """Review focus 5. The template must not invent a layer count. No torch needed."""
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("ops_inventory_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.reduce_layers(types.SimpleNamespace(hidden_size=8), 2) == (None, None)
    cfg = types.SimpleNamespace(num_hidden_layers=24)
    assert mod.reduce_layers(cfg, 2) == (24, 2) and cfg.num_hidden_layers == 2
    assert mod.reduce_layers(types.SimpleNamespace(num_hidden_layers=1), 2) == (1, 1)


@needs_stack
def test_an_export_failure_is_recorded_and_exits_3(tmp_path):
    """--seq 0 makes the export fail on a zero-length input. The template must record it and exit 3, not crash."""
    done, raw = run_template(tmp_path, QWEN3_TINY, "--seq", "0")
    if done.returncode == 0:
        pytest.skip("this transformers traced a zero-length input; pick another way to force a failure")
    assert done.returncode == 3 and raw["failures"] and raw["failures"][0]["stage"] in ("export", "build")
```

- [ ] **Step 2: Run, confirm failures:** `ORCHARD_REFERENCE_PYTHON=~/orchard-venvs/reference/bin/python python3 -m pytest tests/test_ops_inventory_template.py -q`
  → the template file is missing.

- [ ] **Step 3: Implement the template.** Full file:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Trace a model's forward pass on the meta device and write the ATen ops it uses.

`tt-orchard inventory` runs this with the reference interpreter (torch and transformers must import):

    python ops_inventory.py --model-dir DIR --out raw-ops.json [--layers 2] [--seq 8] [--class NAME]

DIR needs only config.json. No weights are read and nothing is downloaded: the model is built on the meta device, so a
27B model traces in seconds. The graph comes from torch.export(strict=False) on one prefill-sized input
(input_ids of shape (1, SEQ)) with use_cache=False. Without use_cache=False the export fails on models that return a
DynamicCache. Decoding-time cache and state updates are not traced. A recurrent model's slow reference path unrolls the
recurrence over the SEQ positions, so its scan ops do appear.

What it will not do: import a repo's own modeling code. A config.json with `auto_map` is written as `skipped`
(exit 0), and so is an architecture the installed transformers does not have. The traced class is the one config.json
names unless --class says otherwise (for example MoshiForCausalLM, the text model inside MoshiForConditionalGeneration,
which takes audio inputs this script does not build). The output says which class was traced.

Exit codes: 0 written (including a `skipped` file), 2 config.json missing, 3 the build or the export failed (the file
is written with the failure), 4 torch or transformers does not import.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = 1
LAYER_KEYS = ("num_hidden_layers", "n_layer", "num_layers")
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks|h)\.\d+(?:\.|$)")
MAX_MODULES = 5
NOTE = ("prefill graph from torch.export(strict=False, use_cache=False) on the meta device; "
        "cache and state updates of decoding are not traced")


def write(path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reduce_layers(config, wanted: int):
    """(layers in the config, layers traced). Sets the first recognised layer-count attribute to at most `wanted`.
    (None, None) when the config has none; the model is then traced as built."""
    for key in LAYER_KEYS:
        value = getattr(config, key, None)
        if isinstance(value, int) and not isinstance(value, bool):
            traced = min(wanted, value)
            try:
                setattr(config, key, traced)
            except Exception:                 # a read-only property: trace at full depth and say so
                return value, value
            return value, traced
    return None, None


def sample_of(val):
    """Output shapes and dtype of one graph node, or None."""
    items = val if isinstance(val, (list, tuple)) else [val]
    shapes, dtype = [], None
    for v in items:
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            shapes.append(list(v.shape))
            dtype = dtype or str(v.dtype)
    return {"output_shapes": shapes, "dtype": dtype} if shapes else None


def module_path(node) -> str:
    stack = node.meta.get("nn_module_stack") or {}
    if not stack:
        return ""
    last = list(stack.values())[-1]
    return last[0] if isinstance(last, (tuple, list)) and last else str(last)


def aggregate(nodes) -> list[dict]:
    ops: dict[str, dict] = {}
    for node in nodes:
        if node.op != "call_function":
            continue
        rec = ops.setdefault(str(node.target), {"op": str(node.target), "count": 0, "count_in_layers": 0,
                                                "sample": None, "modules": []})
        rec["count"] += 1
        path = module_path(node)
        if LAYER_RE.search(path):
            rec["count_in_layers"] += 1
        if rec["sample"] is None:
            rec["sample"] = sample_of(node.meta.get("val"))
        if path and path not in rec["modules"] and len(rec["modules"]) < MAX_MODULES:
            rec["modules"].append(path)
    return sorted(ops.values(), key=lambda r: (-r["count"], r["op"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--class", dest="cls")
    a = ap.parse_args(argv)
    mdir = Path(a.model_dir)
    cfg_path = mdir / "config.json"
    if not cfg_path.is_file():
        print(f"error: {cfg_path} does not exist", file=sys.stderr)
        return 2
    raw_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = {"schema": SCHEMA, "model_dir": str(mdir), "architectures": raw_cfg.get("architectures") or [],
            "seq_len": a.seq, "note": NOTE, "failures": []}
    if "auto_map" in raw_cfg:                                   # remote code: never imported
        write(a.out, {**base, "skipped": "needs remote code: config.json has auto_map"})
        return 0
    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    base.update(torch=torch.__version__, transformers=transformers.__version__)
    name = a.cls or (base["architectures"][0] if base["architectures"] else None)
    cls = getattr(transformers, name, None) if name else None
    if cls is None:
        write(a.out, {**base, "skipped": f"class {name!r} is not in the installed transformers {transformers.__version__}"})
        return 0
    base["traced_class"] = name
    try:
        config = transformers.AutoConfig.from_pretrained(str(mdir), trust_remote_code=False)
        base["layers_in_config"], base["layers_traced"] = reduce_layers(config, a.layers)
        with torch.device("meta"):
            model = cls(config).eval()
        ids = torch.zeros(1, a.seq, dtype=torch.long, device="meta")
    except Exception as exc:
        write(a.out, {**base, "ops": [], "failures": [{"stage": "build", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}]})
        return 3
    try:
        ep = torch.export.export(model, (), {"input_ids": ids, "use_cache": False}, strict=False)
    except Exception as exc:
        write(a.out, {**base, "ops": [], "failures": [{"stage": "export", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}]})
        return 3
    write(a.out, {**base, "ops": aggregate(ep.graph.nodes)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

  `nn_module_stack` values are `(path, class)` pairs in current torch; confirm against the real export when the tiny-Qwen3
  test runs, and fix `module_path` if the shape differs.

- [ ] **Step 4: Run the tests; they pass.** Then write `tests/fixtures/ops-inventory/make_fixtures.py`, which builds a
  `config.json` for each architecture (default config class, `save_pretrained`, then add `"architectures"` by hand because a
  saved config has none), runs the template with the reference interpreter, and writes `raw-<name>.json`:

```python
"""Regenerate the raw-ops fixtures with the reference interpreter. Usage:
    ~/orchard-venvs/reference/bin/python tests/fixtures/ops-inventory/make_fixtures.py
The fixtures are real traces of the architectures in the installed transformers, so the suite needs no torch."""
import json, subprocess, sys, tempfile
from pathlib import Path

import transformers

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent.parent.parent / "orchard" / "skills" / "ops-inventory-templates" / "ops_inventory.py"
CASES = {   # fixture name -> (config class, architectures entry, --class override or None, config overrides)
    "qwen3-tiny": ("Qwen3Config", "Qwen3ForCausalLM", None,
                   dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                        num_attention_heads=2, num_key_value_heads=2, head_dim=16)),
    "rwkv": ("RwkvConfig", "RwkvForCausalLM", None, {}),
    "mamba2": ("Mamba2Config", "Mamba2ForCausalLM", None, {}),
    "moshi-lm": ("MoshiConfig", "MoshiForConditionalGeneration", "MoshiForCausalLM", {}),
}
for name, (cfg_cls, arch, override, kw) in CASES.items():
    with tempfile.TemporaryDirectory() as d:
        getattr(transformers, cfg_cls)(**kw).save_pretrained(d)
        p = Path(d) / "config.json"
        cfg = json.loads(p.read_text())
        cfg["architectures"] = [arch]
        p.write_text(json.dumps(cfg))
        out = HERE / f"raw-{name}.json"
        cmd = [sys.executable, str(TEMPLATE), "--model-dir", d, "--out", str(out)] + (["--class", override] if override else [])
        print(name, subprocess.run(cmd).returncode)
        raw = json.loads(out.read_text())
        raw["model_dir"] = f"<fixture {name}>"          # the temp directory is gone and must not leak into a fixture
        out.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
```

  Run it; each of the four must print `0`. If one prints `3`, keep that file as a fixture too (a recorded failure is a
  useful test input) and say so in the report. Add a test that loads each fixture and asserts its schema keys
  (`schema`, `traced_class`, `ops` or `skipped`, `failures`).

- [ ] **Step 5: Mutations:** `auto_map` check removed → the auto_map test (the template would try to import and build);
  `use_cache=False` removed from the export call → the tiny-Qwen3 test goes red or the fixtures regenerate with a failure;
  `count_in_layers` always 0 → the tiny-Qwen3 test; `reduce_layers` returning `(value, value)` → the layers test;
  `trust_remote_code=True` → add a test that greps the template source for `trust_remote_code=True` and fails on it (a
  source-level guard, because this one cannot be seen at run time).

- [ ] **Step 6: Commit** `git add orchard/skills/ops-inventory-templates tests/fixtures/ops-inventory tests/test_ops_inventory_template.py`
  then `git commit -m "inventory: a torch-side template that traces a model on the meta device"`.

---

### Task 5: `orchard/opinventory.py` and the mapping tables

**Files:**
- Create: `orchard/opinventory.py`, `orchard/data/aten_to_ttnn.json`, `orchard/data/decompositions.json`
- Test: `tests/test_opinventory.py`

**Interfaces:**
- Consumes: `raw-ops.json` (Task 4 schema), fixtures in `tests/fixtures/ops-inventory/`.
- Produces: `load_table(path) -> dict`, `load_dump(version, data_dir=DATA_DIR) -> dict | None`,
  `status_of(op, mapping, decomps, names, seen=()) -> (status, ttnn_name | None, leaves)`,
  `build(raw, *, mapping, decomps, dump, installed_ttnn=None, against=None) -> dict`,
  `render_md(inv) -> str`, `write_outputs(inv, out_dir) -> (json_path, md_path)`,
  `run_inventory(model_dir, *, python, ttnn_python, out_dir, layers=2, seq=8, cls=None, against=None, run=..., data_dir=DATA_DIR, say=print) -> int`,
  `installed_ttnn_version(run, python) -> str | None`. Statuses: `mapped`, `composable`, `unmapped`, `unchecked`.

- [ ] **Step 1: Starter tables.** `orchard/data/aten_to_ttnn.json` maps an aten op name to `{"ttnn": "<name>"}`. Start from this list
  and add an entry only where ttnn has an op with the same meaning. A wrong entry makes `mapped` claim too much, and the recorded
  dump (Task 6) is the only check on the name, so keep it conservative:

```json
{
  "aten.add.Tensor": {"ttnn": "add"}, "aten.sub.Tensor": {"ttnn": "subtract"}, "aten.mul.Tensor": {"ttnn": "multiply"},
  "aten.div.Tensor": {"ttnn": "divide"}, "aten.neg.default": {"ttnn": "neg"}, "aten.mm.default": {"ttnn": "matmul"},
  "aten.bmm.default": {"ttnn": "matmul"}, "aten.linear.default": {"ttnn": "linear"},
  "aten.embedding.default": {"ttnn": "embedding"}, "aten.softmax.int": {"ttnn": "softmax"},
  "aten._softmax.default": {"ttnn": "softmax"}, "aten.silu.default": {"ttnn": "silu"},
  "aten.sigmoid.default": {"ttnn": "sigmoid"}, "aten.tanh.default": {"ttnn": "tanh"}, "aten.relu.default": {"ttnn": "relu"},
  "aten.gelu.default": {"ttnn": "gelu"}, "aten.rsqrt.default": {"ttnn": "rsqrt"}, "aten.exp.default": {"ttnn": "exp"},
  "aten.cos.default": {"ttnn": "cos"}, "aten.sin.default": {"ttnn": "sin"}, "aten.pow.Tensor_Scalar": {"ttnn": "pow"},
  "aten.mean.dim": {"ttnn": "mean"}, "aten.sum.dim_IntList": {"ttnn": "sum"}, "aten.cat.default": {"ttnn": "concat"},
  "aten.permute.default": {"ttnn": "permute"}, "aten.transpose.int": {"ttnn": "transpose"},
  "aten.view.default": {"ttnn": "reshape"}, "aten.reshape.default": {"ttnn": "reshape"},
  "aten._unsafe_view.default": {"ttnn": "reshape"}, "aten.unsqueeze.default": {"ttnn": "unsqueeze"},
  "aten.slice.Tensor": {"ttnn": "slice"}
}
```

  `orchard/data/decompositions.json` maps an aten op to the list of aten ops it is built from:

```json
{"aten.addmm.default": ["aten.mm.default", "aten.add.Tensor"], "aten.silu_.default": ["aten.silu.default"]}
```

  Then run `python3 - <<EOF` over the four fixtures and print every distinct op name that is in no table. For each, either add
  a mapping or decomposition you are sure of, or leave it out (it will report `unmapped`, which is the truthful answer).
  Put the list of ops left unmapped in your report.

- [ ] **Step 2: Write the failing tests** in `tests/test_opinventory.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import opinventory as oi

FIXTURES = Path(__file__).parent / "fixtures" / "ops-inventory"
MAPPING = {"aten.mm.default": {"ttnn": "matmul"}, "aten.embedding.default": {"ttnn": "embedding"},
           "aten.add.Tensor": {"ttnn": "add"}}
DECOMPS = {"aten.addmm.default": ["aten.mm.default", "aten.add.Tensor"],
           "aten.bad.default": ["aten.mm.default", "aten.weird_scan.default"]}
DUMP = {"ttnn_version": "0.65.1", "names": ["matmul", "embedding", "add"]}


def op(name, count, in_layers, modules=("model.layers.0.x",)):
    return {"op": name, "count": count, "count_in_layers": in_layers, "sample": None, "modules": list(modules)}


def raw(**over):
    base = {"schema": 1, "model_dir": "/m", "architectures": ["XForCausalLM"], "traced_class": "XForCausalLM",
            "layers_in_config": 24, "layers_traced": 2, "seq_len": 8, "torch": "2", "transformers": "5", "failures": [],
            "ops": [op("aten.mm.default", 10, 8), op("aten.embedding.default", 1, 0, ("model.embed",)),
                    op("aten.weird_scan.default", 3, 3), op("aten.addmm.default", 2, 2), op("aten.bad.default", 1, 1)]}
    base.update(over)
    return base


def rows(inv):
    return {r["op"]: r for r in inv["ops"]}


def test_a_name_in_the_matching_dump_is_mapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["status"] == "mapped" and rows(inv)["aten.mm.default"]["ttnn"] == "matmul"


@pytest.mark.parametrize("dump,installed", [(None, "0.65.1"), (DUMP, "0.70.0"), (DUMP, None)],
                         ids=["no dump", "dump for another version", "installed version unknown"])
def test_nothing_is_claimed_without_a_dump_for_the_installed_version(dump, installed):
    """Review focus 2."""
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=dump, installed_ttnn=installed)
    assert {r["status"] for r in inv["ops"]} == {"unchecked"}
    assert inv["ttnn_dump"] is None and inv["ttnn_dump_note"]


def test_an_op_with_no_mapping_and_no_decomposition_is_unmapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.weird_scan.default"]["status"] == "unmapped"


def test_a_decomposition_with_every_leaf_mapped_is_composable():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    r = rows(inv)["aten.addmm.default"]
    assert r["status"] == "composable" and r["leaves"] == ["aten.mm.default", "aten.add.Tensor"]


def test_a_decomposition_with_an_unmapped_leaf_is_unmapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.bad.default"]["status"] == "unmapped"


def test_a_decomposition_cycle_does_not_loop():
    cyc = {"aten.a.default": ["aten.b.default"], "aten.b.default": ["aten.a.default"]}
    assert oi.status_of("aten.a.default", {}, cyc, {"matmul"})[0] == "unmapped"


def test_counts_in_layers_are_scaled_to_the_real_depth():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 8 * 12 + 2      # 8 in layers x (24 / 2) + 2 outside
    assert rows(inv)["aten.embedding.default"]["scaled_count"] == 1


def test_without_a_layer_count_nothing_is_scaled_and_the_report_says_so():
    """Review focus 5."""
    inv = oi.build(raw(layers_in_config=None, layers_traced=None), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                   installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 10 and "not scaled" in inv["scaling"]


def test_a_skipped_trace_gives_an_inventory_with_the_reason_and_no_ops():
    inv = oi.build({"schema": 1, "model_dir": "/m", "architectures": [], "skipped": "needs remote code"},
                   mapping=MAPPING, decomps=DECOMPS, dump=DUMP)
    assert inv["skipped"] == "needs remote code" and inv["ops"] == []


def test_against_marks_ops_the_base_model_also_uses():
    base = oi.build(raw(ops=[op("aten.mm.default", 1, 1)]), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                    installed_ttnn="0.65.1")
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1", against=base)
    assert rows(inv)["aten.mm.default"]["in_base"] is True and rows(inv)["aten.weird_scan.default"]["in_base"] is False


def test_the_summary_counts_ops_by_status():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert inv["summary"]["distinct"] == {"mapped": 2, "composable": 1, "unmapped": 2, "unchecked": 0}


def test_a_dump_version_that_is_a_path_is_refused(tmp_path):
    assert oi.load_dump("../x", data_dir=tmp_path) is None and oi.load_dump("", data_dir=tmp_path) is None


def test_load_dump_needs_the_version_inside_the_file_to_match(tmp_path):
    (tmp_path / "ttnn_ops-0.65.1.json").write_text(json.dumps({"ttnn_version": "0.9", "names": ["add"]}))
    assert oi.load_dump("0.65.1", data_dir=tmp_path) is None


def test_a_table_entry_without_a_ttnn_name_is_refused(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"aten.x.default": {"nope": 1}}))
    with pytest.raises(ValueError):
        oi.load_table(p)


def test_every_table_entry_names_an_op_that_a_real_trace_contains():
    """A typo in a table would never match and would quietly leave an op unmapped."""
    seen = set()
    for f in FIXTURES.glob("raw-*.json"):
        seen |= {o["op"] for o in json.loads(f.read_text()).get("ops", [])}
    table = oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json")
    unknown = sorted(k for k in table if k not in seen)
    assert not unknown, f"not in any fixture trace: {unknown}"


def test_the_markdown_says_what_mapped_means_and_names_the_dump():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    md = oi.render_md(inv)
    assert "name exists in the dump of ttnn 0.65.1" in md and "does not say" in md
```

  If `test_every_table_entry_names_an_op_that_a_real_trace_contains` fails for a starter entry that no fixture happens to
  contain, remove the entry (do not weaken the test); it can be added back with the model that needs it.

- [ ] **Step 3: Run; confirm failures** (`ImportError: cannot import name 'opinventory'`).

- [ ] **Step 4: Implement `orchard/opinventory.py`:**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Turn a raw op trace into an inventory: which of a model's ops ttnn already has, which it can build, which are missing.

Statuses, and what each claims:

    mapped      the mapping table names a ttnn op and the recorded dump for the installed ttnn contains that name.
                This is a name match. It does not say the op supports the shape, dtype or layout the model needs.
    composable  a decomposition exists and every leaf is `mapped`.
    unmapped    neither.
    unchecked   there is no dump for the installed ttnn version, so nothing is claimed.

The dump comes from `tt-orchard inventory refresh-ttnn` (orchard/ttnn_dump.py), which imports ttnn under a lease,
because ttnn's op names cannot be read without importing it. A dump for another ttnn version than the installed one is
not used.

Standard library only. The torch side is orchard/skills/ops-inventory-templates/ops_inventory.py.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
TEMPLATE = Path(__file__).resolve().parent / "skills" / "ops-inventory-templates" / "ops_inventory.py"
SCHEMA = 1
STATUSES = ("mapped", "composable", "unmapped", "unchecked")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
LABEL = {
    "traced": "measured: the op names, counts and one sample shape per op come from torch.export on the meta device",
    "status": "table plus dump: `mapped` is a name match against the recorded ttnn dump and does not say the op "
              "supports the model's shape, dtype or layout",
    "scaled_count": "estimate: ops inside layers are multiplied by layers in the config / layers traced",
}


def load_table(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    for key, entry in data.items():
        if isinstance(entry, dict):
            if not isinstance(entry.get("ttnn"), str):
                raise ValueError(f"{path}: {key!r} has no string `ttnn`")
        elif not (isinstance(entry, list) and all(isinstance(x, str) for x in entry)):
            raise ValueError(f"{path}: {key!r} must be {{\"ttnn\": name}} or a list of aten op names")
    return data


def load_dump(version, data_dir=DATA_DIR):
    """The recorded dump for exactly this ttnn version, or None. A version that is not a plain token is refused,
    so it cannot name a file outside data_dir."""
    if not isinstance(version, str) or not VERSION_RE.match(version):
        return None
    p = Path(data_dir) / f"ttnn_ops-{version}.json"
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return d if d.get("ttnn_version") == version and isinstance(d.get("names"), list) else None


def status_of(op, mapping, decomps, names, seen=()):
    """(status, ttnn name, leaves). `names` is the set of names in the dump, or None for no usable dump."""
    entry = mapping.get(op)
    if names is None:
        return "unchecked", (entry or {}).get("ttnn"), []
    if entry and entry["ttnn"] in names:
        return "mapped", entry["ttnn"], []
    parts = decomps.get(op)
    if parts and op not in seen:
        leaves, ok = [], True
        for p in parts:
            s, _, sub = status_of(p, mapping, decomps, names, seen + (op,))
            if s == "mapped":
                leaves.append(p)
            elif s == "composable":
                leaves.extend(sub)
            else:
                ok = False
        if ok:
            return "composable", None, leaves
    return "unmapped", (entry or {}).get("ttnn"), []


def build(raw, *, mapping, decomps, dump, installed_ttnn=None, against=None) -> dict:
    inv = {"schema": SCHEMA, "model_dir": raw.get("model_dir"), "architectures": raw.get("architectures", []),
           "traced_class": raw.get("traced_class"), "torch": raw.get("torch"), "transformers": raw.get("transformers"),
           "traced": {"graph": "prefill", "seq_len": raw.get("seq_len"), "layers_in_config": raw.get("layers_in_config"),
                      "layers_traced": raw.get("layers_traced"), "note": raw.get("note")},
           "failures": raw.get("failures", []), "label": LABEL,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if "skipped" in raw:
        inv.update(skipped=raw["skipped"], ops=[], summary={}, ttnn_dump=None, ttnn_dump_note=None, scaling=None)
        return inv
    names, note = None, None
    if dump is None:
        note = "no recorded dump for the installed ttnn; nothing is claimed as mapped"
    elif installed_ttnn is None:
        note = "the installed ttnn version is unknown, so the recorded dump is not used"
    elif dump.get("ttnn_version") != installed_ttnn:
        note = f"the recorded dump is for ttnn {dump.get('ttnn_version')} and ttnn {installed_ttnn} is installed; not used"
    else:
        names = set(dump["names"])
    inv["ttnn_dump"] = {"ttnn_version": dump["ttnn_version"], "sha256": dump.get("sha256")} if names is not None else None
    inv["ttnn_dump_note"] = note
    layers, traced = raw.get("layers_in_config"), raw.get("layers_traced")
    factor = layers / traced if layers and traced else None
    inv["scaling"] = (f"in-layer ops scaled by {layers}/{traced}" if factor
                      else "the config has no recognisable layer count: counts are not scaled")
    base_ops = {o["op"] for o in against["ops"]} if against else None
    rows, distinct, scaled_total = [], {s: 0 for s in STATUSES}, {s: 0 for s in STATUSES}
    for o in raw["ops"]:
        status, ttnn_name, leaves = status_of(o["op"], mapping, decomps, names)
        inl = o.get("count_in_layers", 0)
        scaled = round(inl * factor + (o["count"] - inl)) if factor else o["count"]
        row = {"op": o["op"], "count": o["count"], "scaled_count": scaled, "status": status, "ttnn": ttnn_name,
               "leaves": leaves, "sample": o.get("sample"), "modules": o.get("modules", [])}
        if base_ops is not None:
            row["in_base"] = o["op"] in base_ops
        rows.append(row)
        distinct[status] += 1
        scaled_total[status] += scaled
    inv["ops"] = rows
    inv["summary"] = {"distinct": distinct, "scaled_total": scaled_total}
    return inv


def render_md(inv) -> str:
    lines = [f"# Op inventory: {inv.get('traced_class') or inv.get('architectures')}", ""]
    if inv.get("skipped"):
        return "\n".join(lines + [f"Skipped: {inv['skipped']}", ""])
    d = inv.get("ttnn_dump")
    lines += [f"Model directory: `{inv['model_dir']}`. transformers {inv.get('transformers')}, torch {inv.get('torch')}.",
              f"Traced: {inv['traced']['note']}. Layers in config {inv['traced']['layers_in_config']}, "
              f"traced {inv['traced']['layers_traced']}. {inv['scaling']}.", ""]
    if d:
        lines.append(f"`mapped` means the name exists in the dump of ttnn {d['ttnn_version']}. It does not say the op "
                     "supports the shape, dtype or layout this model needs.")
    else:
        lines.append(f"Nothing is claimed as mapped: {inv['ttnn_dump_note']}.")
    lines += ["", "| status | distinct ops |", "| --- | --- |"]
    lines += [f"| {s} | {n} |" for s, n in inv["summary"]["distinct"].items()]
    if inv["failures"]:
        lines += ["", "Trace failures:"] + [f"- {f['stage']}: {f['error']}" for f in inv["failures"]]
    lines += ["", "| op | count | scaled | status | ttnn |", "| --- | --- | --- | --- | --- |"]
    order = {s: i for i, s in enumerate(("unmapped", "unchecked", "composable", "mapped"))}
    for r in sorted(inv["ops"], key=lambda r: (order[r["status"]], -r["scaled_count"], r["op"])):
        lines.append(f"| `{r['op']}` | {r['count']} | {r['scaled_count']} | {r['status']} | {r['ttnn'] or ''} |")
    return "\n".join(lines) + "\n"


def write_outputs(inv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "ops-inventory.json", out / "ops-inventory.md"
    j.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    m.write_text(render_md(inv), encoding="utf-8")
    return j, m


def _real_run(argv, timeout):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


def installed_ttnn_version(run, python):
    """The installed ttnn's version, from package metadata. ttnn itself is not imported: that would open the device."""
    if not python:
        return None
    rc, out, _ = run([str(python), "-c", "import importlib.metadata as m; print(m.version('ttnn'))"], 60)
    return out.strip() or None if rc == 0 else None


def run_inventory(model_dir, *, python, ttnn_python, out_dir, layers=2, seq=8, cls=None, against=None,
                  run=_real_run, data_dir=DATA_DIR, say=print) -> int:
    """Trace, map, write. 0 on success (a skipped model is a success), 2 when the template could not run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw-ops.json"
    argv = [str(python), str(TEMPLATE), "--model-dir", str(model_dir), "--out", str(raw_path),
            "--layers", str(layers), "--seq", str(seq)] + (["--class", cls] if cls else [])
    rc, stdout, stderr = run(argv, 1800)
    if rc not in (0, 3) or not raw_path.is_file():
        say(f"the trace did not run (exit {rc}): {(stderr or stdout).strip()[-300:]}")
        return 2
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    version = installed_ttnn_version(run, ttnn_python)
    base = json.loads(Path(against).read_text(encoding="utf-8")) if against else None
    inv = build(raw, mapping=load_table(Path(data_dir) / "aten_to_ttnn.json"),
                decomps=load_table(Path(data_dir) / "decompositions.json"), dump=load_dump(version, data_dir),
                installed_ttnn=version, against=base)
    j, m = write_outputs(inv, out)
    say(f"wrote {j} and {m}")
    if inv.get("skipped"):
        say(f"skipped: {inv['skipped']}")
    else:
        say("distinct ops by status: " + ", ".join(f"{s} {n}" for s, n in inv["summary"]["distinct"].items()))
    return 0
```

- [ ] **Step 5: Run the tests; they pass.** Also add a test of `run_inventory` with a fake `run` that writes a fixture to
  `raw-ops.json` and answers the metadata call, asserting both output files exist and that exit code 4 from the template
  returns 2.

- [ ] **Step 6: Mutations:** dump used when its version differs (`installed_ttnn` check removed) →
  `test_nothing_is_claimed_without_a_dump_for_the_installed_version[dump for another version]`; `unknown installed version`
  accepted → that case; decomposition leaves not checked → `test_a_decomposition_with_an_unmapped_leaf_is_unmapped`; `seen`
  removed → the cycle test (RecursionError); scaling uses `count` not `count_in_layers` → the scaling test; `VERSION_RE` check
  removed → the path test; `LABEL` text dropped from the markdown → the markdown test.

- [ ] **Step 7: Commit** `git add orchard/opinventory.py orchard/data tests/test_opinventory.py` then
  `git commit -m "inventory: statuses, scaling and rendering, with the tables"`.

---

### Task 6: the ttnn dump and the `inventory` command

**Files:**
- Create: `orchard/ttnn_dump.py`
- Modify: `orchard/cli.py`, `README.md`
- Test: `tests/test_ttnn_dump.py`, additions to `tests/test_cli.py`

**Interfaces:**
- Consumes: `opinventory.run_inventory`, `opinventory.DATA_DIR`.
- Produces: `ttnn_dump.dump_command(python, who="claude:orchard-inventory") -> list[str]`, `parse(stdout) -> dict`,
  `write_dump(dump, data_dir) -> Path`, `main(python, *, run, data_dir, say) -> int`;
  `tt-orchard inventory TARGET [--python P] [--ttnn-python P] [--against F] [--layers N] [--seq N] [--class C] [--out DIR]`
  where TARGET is a model directory or the word `refresh-ttnn`.

- [ ] **Step 1: Write the failing tests** in `tests/test_ttnn_dump.py`:

```python
from __future__ import annotations

import hashlib
import json

import pytest

from orchard import ttnn_dump as td


def test_the_dump_runs_under_a_one_chip_lease_and_never_directly():
    argv = td.dump_command("/py/bin/python")
    assert argv[:2] == ["gozer", "run"] and "--chips" in argv and argv[argv.index("--chips") + 1] == "1"
    assert "--who" in argv and argv[argv.index("--who") + 1].startswith("claude:")
    assert argv[argv.index("--") + 1] == "/py/bin/python"


def test_parse_reads_the_marked_line_and_ignores_the_rest():
    out = "noise\nORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "0.65.1", "names": ["add", "matmul"]}) + "\nmore\n"
    assert td.parse(out) == {"ttnn_version": "0.65.1", "names": ["add", "matmul"]}


def test_parse_without_the_marker_raises():
    with pytest.raises(ValueError):
        td.parse("Traceback ...")


def test_write_dump_names_the_file_by_version_and_records_a_sha256_of_the_names(tmp_path):
    p = td.write_dump({"ttnn_version": "0.65.1", "names": ["matmul", "add"]}, tmp_path)
    data = json.loads(p.read_text())
    assert p.name == "ttnn_ops-0.65.1.json" and data["names"] == ["add", "matmul"]      # sorted, no duplicates
    assert data["sha256"] == hashlib.sha256(json.dumps(["add", "matmul"]).encode()).hexdigest()


def test_write_dump_refuses_a_version_that_is_a_path(tmp_path):
    with pytest.raises(ValueError):
        td.write_dump({"ttnn_version": "../evil", "names": ["x"]}, tmp_path)


def test_main_prints_the_command_before_running_it_and_writes_the_file(tmp_path):
    said, calls = [], []

    def run(argv, timeout):
        calls.append(argv)
        return 0, "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": "0.65.1", "names": ["add"]}) + "\n", ""

    code = td.main("/py/bin/python", run=run, data_dir=tmp_path, say=said.append)
    assert code == 0 and (tmp_path / "ttnn_ops-0.65.1.json").is_file()
    assert any("gozer run" in line for line in said[:1]), "the command is shown before it runs"


def test_main_reports_a_failed_dump_and_writes_nothing(tmp_path):
    code = td.main("/py/bin/python", run=lambda a, t: (1, "", "RuntimeError: no device"), data_dir=tmp_path, say=lambda s: None)
    assert code == 2 and not list(tmp_path.glob("ttnn_ops-*"))
```

- [ ] **Step 2: Implement `orchard/ttnn_dump.py`:**

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard inventory refresh-ttnn`: record the names of the ops the installed ttnn offers.

ttnn's ops are bound from compiled code, so their names are in no Python file, and importing ttnn opens the device.
The import therefore runs under a one-chip gozer lease, once per ttnn version, and the names are written to
orchard/data/ttnn_ops-<version>.json. The operator runs this command. The supervisor never does.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from orchard import opinventory

MARK = "ORCHARD_TTNN_DUMP "
# Public callables at the top of ttnn and one level into the two namespaces that hold model ops.
SCRIPT = r'''
import importlib.metadata as m, json, types
import ttnn
names = []
for n in sorted(dir(ttnn)):
    if n.startswith("_"):
        continue
    obj = getattr(ttnn, n, None)
    if callable(obj):
        names.append(n)
    elif isinstance(obj, types.ModuleType) and n in ("experimental", "transformer"):
        names += [f"{n}.{k}" for k in sorted(dir(obj)) if not k.startswith("_") and callable(getattr(obj, k, None))]
print("ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": m.version("ttnn"), "names": names}))
'''


def dump_command(python, who: str = "claude:orchard-inventory") -> list[str]:
    return ["gozer", "run", "--chips", "1", "--exact", "1", "--who", who,
            "--reason", "dump ttnn op names for tt-orchard inventory", "--", str(python), "-c", SCRIPT]


def parse(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    raise ValueError("the dump printed no ORCHARD_TTNN_DUMP line")


def write_dump(dump: dict, data_dir) -> Path:
    version = dump["ttnn_version"]
    if not opinventory.VERSION_RE.match(version):
        raise ValueError(f"ttnn version {version!r} is not a plain version token")
    names = sorted(set(dump["names"]))
    out = Path(data_dir) / f"ttnn_ops-{version}.json"
    out.write_text(json.dumps({"ttnn_version": version, "names": names,
                               "sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest()}, indent=2) + "\n",
                   encoding="utf-8")
    return out


def _real_run(argv, timeout):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


def main(python, *, run=_real_run, data_dir=opinventory.DATA_DIR, say=print) -> int:
    argv = dump_command(python)
    say("running (it takes a one-chip lease): " + " ".join(argv[:11]) + " -c <script>")
    rc, out, err = run(argv, 1800)
    if rc != 0:
        say(f"the dump failed (exit {rc}): {(err or out).strip()[-300:]}")
        return 2
    try:
        path = write_dump(parse(out), data_dir)
    except (ValueError, KeyError) as exc:
        say(f"the dump could not be read: {exc}")
        return 2
    say(f"wrote {path}")
    return 0
```

- [ ] **Step 3: CLI.** In `_parser()` add:

```python
    inv = sub.add_parser("inventory", parents=[common],
                         help="list the ops a model's reference code uses and which ones ttnn has; "
                              "`inventory refresh-ttnn` records ttnn's op names (takes a one-chip lease)")
    inv.add_argument("target", help="a model directory with a config.json, or the word refresh-ttnn")
    inv.add_argument("--python", help="interpreter with torch and transformers (default $ORCHARD_REFERENCE_PYTHON, "
                                      "then reference_python in bringup.toml)")
    inv.add_argument("--ttnn-python", help="interpreter whose installed ttnn version is read, and that refresh-ttnn "
                                           "runs under a lease (default ~/.tenstorrent-venv/bin/python)")
    inv.add_argument("--against", metavar="FILE", help="an ops-inventory.json of a nearest model, to mark shared ops")
    inv.add_argument("--layers", type=int, default=2, help="layers to trace (default 2)")
    inv.add_argument("--seq", type=int, default=8, help="sequence length to trace (default 8)")
    inv.add_argument("--class", dest="cls", help="the transformers class to trace if not config.json's first architecture")
    inv.add_argument("--out", help="output directory (default ./ops-inventory)")
```

  and in `main` after the `kernel-env` branch:

```python
    if args.cmd == "inventory":
        return _inventory(args, env, home, out)
```

  with a module-level helper in `cli.py`:

```python
def _inventory(args, env, home, out) -> int:
    from orchard import opinventory, ttnn_dump
    say = lambda line: print(line, file=out)
    ttnn_python = args.ttnn_python or str(home / ".tenstorrent-venv" / "bin" / "python")
    if args.target == "refresh-ttnn":
        return ttnn_dump.main(ttnn_python, say=say)
    python = args.python or env.get("ORCHARD_REFERENCE_PYTHON")
    if not python:
        try:
            python = bringup_config.load(find_config(getattr(args, "config", None), env, CHECKOUT, home)).reference_python
        except (NoConfig, bringup_config.BringupConfigError):
            python = None
    if not python:
        return _refuse("no interpreter with torch and transformers: pass --python, set ORCHARD_REFERENCE_PYTHON, "
                       "or set reference_python in bringup.toml")
    return opinventory.run_inventory(args.target, python=python, ttnn_python=ttnn_python,
                                     out_dir=args.out or "ops-inventory", layers=args.layers, seq=args.seq,
                                     cls=args.cls, against=args.against, say=say)
```

  Tests in `tests/test_cli.py`: `refresh-ttnn` calls `ttnn_dump.main` with the ttnn python and never reaches
  `run_inventory`; a model directory calls `run_inventory` with the `--python` given; with no interpreter anywhere it
  refuses (exit 2) and calls neither. Add README rows for every new flag and a short "Op inventory" section that says what
  `mapped` means and that `refresh-ttnn` takes a lease. Run `tests/test_readme.py`.

- [ ] **Step 4: Mutations:** lease dropped from the dump command (`gozer run` removed) → the first dump test; the version path
  check removed → the path test; a failed dump still writing a file → `test_main_reports_a_failed_dump_and_writes_nothing`;
  `refresh-ttnn` falling through to `run_inventory` → the CLI test.

- [ ] **Step 5: Commit** `git add orchard/ttnn_dump.py orchard/cli.py README.md tests/test_ttnn_dump.py tests/test_cli.py` then
  `git commit -m "inventory: the ttnn dump and the tt-orchard inventory command"`.

---

### Task 7: stage 0 records the inventory when it can (additive, last of Part A)

**Files:**
- Modify: `orchard/skills/delta-triage.md` (one added step), `orchard/opinventory.py` (a `stage0` helper), possibly the stage 0
  gate in `orchard/stages.py` if it rejects unknown keys in `delta.json`
- Test: `tests/test_opinventory_stage0.py`

**Interfaces:**
- Consumes: `opinventory.run_inventory`, the `delta.json` that `delta_triage.py` writes in the stage directory.
- Produces: `opinventory.record_in_delta(stage_dir) -> bool`, which reads `ops-inventory.json` in `stage_dir`, and merges
  `{"inventory": {"traced_class", "skipped", "distinct", "unmapped_ops": [up to 10 names], "ttnn_dump"}}` into `delta.json`
  by writing a temporary file and renaming it. Returns False when either file is missing. The stage 0 gate is unchanged
  except that it must accept the optional `inventory` key.

- [ ] **Step 1: Read** `orchard/skills/delta-triage.md`, the stage 0 spec in `orchard/stages.py` (`gate_delta` and the stage
  table), and `tests/test_delta_triage_template.py`. Note how the skill names inputs and which placeholders
  (`{{ORCHARD_DIR}}` and the others in `orchard/paths.py`) it may use. A repo test fails on any `/home/` or `/mnt/` path in
  `orchard/skills`.

- [ ] **Step 2: Write the failing tests** in `tests/test_opinventory_stage0.py`:
  - `record_in_delta` merges the summary and keeps every other key of `delta.json` byte-for-byte equal in value
    (compare the parsed dicts apart from `inventory`).
  - It returns False and changes nothing when `ops-inventory.json` is missing.
  - A skipped inventory is recorded as `{"skipped": "<reason>"}` with no `distinct`.
  - `unmapped_ops` holds at most 10 names, the most frequent first.
  - The stage 0 gate (`gate_delta`, or whatever the stage table names) accepts a `delta.json` that carries `inventory` and
    still rejects one that lacks a field it required before (copy a passing `delta.json` from the existing gate tests).
  - The skill text mentions `ops_inventory.py` and `reference_python`, names no `/home/` path, and says that the measured
    class decides when the inventory disagrees.

- [ ] **Step 3: Implement** `record_in_delta` (standard library only) and the skill step: after the triage draft, when the
  run's inputs include `reference_python`, run the template through `opinventory` (`python3 -m orchard.opinventory --model-dir
  <the new model's snapshot> --out <stage dir>` needs a `__main__` block: add `python3 -m orchard.opinventory` as a thin
  wrapper over `run_inventory` plus `record_in_delta`, with `--python`, `--model-dir`, `--out`, `--ttnn-python`), then read
  `ops-inventory.md` once, and add one finding to `delta.json` only if the inventory has `unmapped` ops while the measured class
  is `weights-only`. The skill says the measured class still decides. If `reference_python` is absent the step is skipped and
  the skill says so.

- [ ] **Step 4: Mutations:** merge overwrites other keys → the merge test; missing inventory file raises → the False test;
  gate rejects `inventory` → the gate test; the skill step removed → the skill test.

- [ ] **Step 5: Run the whole suite** `python3 -m pytest -q` and confirm the count is the earlier count plus the new tests,
  with the same single skip. Commit `git add -A orchard tests` (check `git status` first: nothing outside these directories)
  then `git commit -m "inventory: stage 0 records the op inventory when a reference interpreter is set"`.


---

### Task 8: the kernel loop templates: references, harness, check

**Files:**
- Create: `orchard/skills/kernel-loop-templates/references.py`, `kernel_harness.py`, `kernel_check.py`,
  `ops/add.json`, `ops/silu_mul.json`, `ops/rmsnorm.json`, `ops/linear_scan.json`
- Create: `tests/fixtures/kernel-loop/kernels/*.py` (kernels, below) and `tests/fixtures/kernel-loop/outputs/*.txt` (recorded
  simulator output per category)
- Test: `tests/test_kernel_loop_templates.py`

**Interfaces:**
- Produces: the kernel contract. `kernel.py` defines `OPERATION`, an `@ttl.operation(grid="full")` function that takes the input
  tensors in the order `ops/<op>.json` lists them under `inputs`, then the output tensor last (the shape of upstream's
  `examples/eltwise_add.py`). `kernel_check.py` prints `PASS` or `FAIL: <category>: <first line>` and writes `check.json`.
  `kernel_check.classify(...)` (stdlib only; Task 9 imports it by path).
- The categories: `no_kernel`, `import_error`, `tt_lang_error`, `shape_mismatch`, `numeric_mismatch`, `timeout`,
  `compile_error` (hardware), `hang` (hardware, set by the probe), `hw_numeric_mismatch`, `harness_error` (a fault in these
  templates or the environment, never the agent's kernel).

- [ ] **Step 1: The op specs and references.**

`references.py` (the spec may name only a function in this table; there is no way for a spec to name other code):

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Torch references for the probe's ops. Inputs arrive as float32 tensors, rounded to bf16 first, in the spec's order."""
import torch


def add(x):
    return x["a"] + x["b"]


def silu_mul(x):
    return torch.nn.functional.silu(x["gate"]) * x["up"]


def rmsnorm(x):
    # w is the weight already expanded to the full (rows, cols) shape: a deliberate simplification so the first
    # measurement does not depend on a broadcast. eps is 1e-6.
    h = x["x"]
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * x["w"]


def linear_scan(x):
    # h[t] = a[t] * h[t-1] + b[t], scanned down axis 0, independently per column.
    a, b = x["a"], x["b"]
    h, out = torch.zeros_like(b[0]), torch.empty_like(b)
    for t in range(b.shape[0]):
        h = a[t] * h + b[t]
        out[t] = h
    return out


REFERENCES = {"add": add, "silu_mul": silu_mul, "rmsnorm": rmsnorm, "linear_scan": linear_scan}
```

`ops/add.json` (the calibration op: the working kernel is upstream's example, so a failure here is a fault in the
harness or the environment):

```json
{"op": "add", "description": "out = a + b, elementwise", "inputs": {"a": [128, 128], "b": [128, 128]}, "output": [128, 128],
 "dtype": "bfloat16", "reference": "add", "pcc_bar_sim": 0.999, "pcc_bar_hw": 0.99, "seeds": [0, 1, 2], "heavy_tail": true,
 "provisional": true}
```

`ops/silu_mul.json`: same fields with `"op": "silu_mul"`, inputs `{"gate": [64, 128], "up": [64, 128]}`, output `[64, 128]`,
reference `silu_mul`, description `out = silu(gate) * up, elementwise`.
`ops/rmsnorm.json`: inputs `{"x": [64, 256], "w": [64, 256]}`, output `[64, 256]`, reference `rmsnorm`, description
`out = x * rsqrt(mean(x^2, last axis) + 1e-6) * w, with w given at full size`.
`ops/linear_scan.json`: inputs `{"a": [64, 128], "b": [64, 128]}`, output `[64, 128]`, reference `linear_scan`, description
`out[t] = a[t] * out[t-1] + b[t] down axis 0, per column, out[-1] = 0`, and `"gen": {"a": "sigmoid"}` (a in (0, 1) so the scan
stays bounded; unlisted inputs are normal).

- [ ] **Step 2: The harness.** `kernel_harness.py` runs under `tt-lang-sim` (phase `sim`) or the real stack (phase `hw`). Under
  the simulator `import ttl` and `import ttnn` resolve to the simulator, as upstream's examples rely on.

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run an agent-written tt-lang kernel on its op's inputs and write result.json.

    tt-lang-sim kernel_harness.py -- --spec FILE --kernel FILE --out FILE [--phase sim|hw]
    python kernel_harness.py --spec FILE --kernel FILE --out FILE --phase hw      (real stack, under a lease)

The harness, not the kernel, opens the device, builds the inputs and reads the output back, so the agent writes only the
kernel. It never evaluates anything the spec names: the reference is a key into references.REFERENCES.
Prints `PROGRESS <what>` lines so a watcher can see it is alive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # references.py sits next to this file; tt-lang-sim does not add the script's directory


def progress(msg: str) -> None:
    print(f"PROGRESS {msg}", flush=True)


def make_inputs(spec: dict, seed: int, heavy_tail: bool, torch) -> dict:
    g = torch.Generator().manual_seed(seed)
    ins = {}
    for name, shape in spec["inputs"].items():
        kind = spec.get("gen", {}).get(name, "normal")
        x = torch.randn(shape, generator=g)
        if kind == "sigmoid":
            x = torch.sigmoid(x)
        elif heavy_tail:                                  # a norm on unit-normal inputs is easy: add outliers
            x = torch.where(torch.rand(shape, generator=g) < 0.01, x * 20.0, x)
        ins[name] = x.to(torch.bfloat16)
    return ins


def pcc(got, exp, torch):
    g, e = got.flatten().float(), exp.flatten().float()
    if g.numel() < 2 or float(g.std()) == 0.0 or float(e.std()) == 0.0:
        return None                                       # a constant output has no correlation: counted as a failure
    value = float(torch.corrcoef(torch.stack([g, e]))[0, 1])
    return None if value != value else value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("sim", "hw"), default="sim")
    a = ap.parse_args(argv)
    result = {"phase": a.phase, "sets": [], "error": None}
    stage = "import"
    try:
        spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
        import torch
        import ttnn
        from references import REFERENCES
        if spec["reference"] not in REFERENCES:
            raise KeyError(f"reference {spec['reference']!r} is not in references.REFERENCES")
        loader = importlib.util.spec_from_file_location("kernel_under_test", a.kernel)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        operation = getattr(module, "OPERATION", None)
        if operation is None:
            result["error"] = {"phase": "import", "type": "NoOperation", "text": "kernel.py defines no OPERATION"}
        else:
            stage = "run"
            progress("open_device")
            device = ttnn.open_device(device_id=0)
            try:
                for seed in spec["seeds"]:
                    for tail in ([False, True] if spec.get("heavy_tail") else [False]):
                        name = f"seed{seed}{'-tail' if tail else ''}"
                        ins = make_inputs(spec, seed, tail, torch)
                        expected = REFERENCES[spec["reference"]]({k: v.float() for k, v in ins.items()})
                        progress(f"set {name}")
                        tt_ins = [ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                                  for v in ins.values()]
                        out = ttnn.from_torch(torch.zeros(spec["output"], dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                                              layout=ttnn.TILE_LAYOUT, device=device)
                        operation(*tt_ins, out)
                        got = ttnn.to_torch(out).float()
                        err = (got - expected).abs().max()
                        result["sets"].append({"name": name, "seed": seed, "heavy_tail": tail, "pcc": pcc(got, expected, torch),
                                               "max_abs_err": float(err), "finite": bool(torch.isfinite(got).all())})
            finally:
                ttnn.close_device(device)
    except BaseException as exc:                          # the diagnostic text must reach the check, so record it and print it
        traceback.print_exc()
        result["error"] = {"phase": stage, "type": type(exc).__name__, "text": str(exc)[:2000]}
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

  Confirm with the real simulator that a kernel's own exception reaches `result["error"]["text"]` and the console; if the
  simulator prints its diagnostic to the console and raises a bare exception, `classify` reads both (it already does).
  The harness exits 0 even after an error, because the error is in the result file; the check reads that.

- [ ] **Step 3: The check.** `kernel_check.py`:

```python
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check an agent-written kernel.py against its op's reference on the tt-lang simulator.

    python3 kernel_check.py [--attempt-dir DIR] [--phase sim] [--spec FILE] [--sim PATH] [--timeout-s N]
                            [--tt-lang-json JSON] [--out FILE]

Defaults come from DIR/check_config.json: {"sim": "<tt-lang-sim>", "timeout_s": 300, "tt_lang": {...}}. It runs
kernel_harness.py from the directory THIS file is in, so the harness that judges a kernel is the one next to the check
that was run. Exit 0 and `PASS` when the kernel passed; exit 1 and `FAIL: <category>: <first line>` when it did not;
exit 2 on a usage error. It writes check.json with sha256 values of the files that took part.

The category comes from the diagnostic text the simulator prints, never from an exit code alone. (The tt-lang-sim
wheel without its diagnostics file ends every diagnosed error as FileNotFoundError with a non-zero exit; that case is
reported as harness_error, because it is a fault in the environment.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPE_RE = re.compile(r"Tensor shape .*does not match Block shape")


def sha256_file(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def first_line(text: str) -> str:
    """The simulator's first diagnostic line: `error: ...`, a deadlock notice, or the `Error in kernel` header."""
    lines = [l.strip() for l in text.splitlines()]
    for l in lines:
        if l.startswith("error:") or l.startswith("Deadlock detected"):
            return l[:200]
    for l in lines:
        if "Error in kernel" in l:
            return l[:200]
    return ""


def classify(*, kernel_present, timed_out, returncode, text, result, bar, phase="sim"):
    """(category or None, first line). None means the kernel passed."""
    numeric = "hw_numeric_mismatch" if phase == "hw" else "numeric_mismatch"
    runtime = "compile_error" if phase == "hw" else "tt_lang_error"
    if not kernel_present:
        return "no_kernel", "kernel.py does not exist"
    if timed_out:
        return "timeout", "no result within the time allowed"
    err = (result or {}).get("error")
    blob = text + "\n" + ((err or {}).get("text") or "")
    if "FileNotFoundError" in blob and "diagnostics.py" in blob:
        return "harness_error", "tt-lang-sim's diagnostics.py is missing; run `tt-orchard kernel-env`"
    if err and err.get("type") == "NoOperation":
        return "no_kernel", "kernel.py defines no OPERATION"
    if err and err.get("phase") == "import":
        first = ((err.get("text") or "").splitlines() or [""])[0]
        return "import_error", f"{err.get('type')}: {first}"[:200]
    m = SHAPE_RE.search(blob)
    if m:
        return "shape_mismatch", first_line(blob) or m.group(0)[:200]
    if err or "Deadlock detected" in blob:
        return runtime, first_line(blob) or ((err or {}).get("text") or "")[:200]
    if result is None:
        return "harness_error", f"the harness ended with exit {returncode} and wrote no result"
    sets = result.get("sets", [])
    if not sets:
        return "harness_error", "the harness ran no input sets"
    bad = [s for s in sets if s["pcc"] is None or not s["finite"] or s["pcc"] < bar]
    if bad:
        s = bad[0]
        return numeric, f"set {s['name']}: pcc {s['pcc']}, finite {s['finite']}, bar {bar}"
    return None, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt-dir", default=".")
    ap.add_argument("--phase", choices=("sim",), default="sim")
    ap.add_argument("--spec")
    ap.add_argument("--sim")
    ap.add_argument("--timeout-s", type=float)
    ap.add_argument("--tt-lang-json")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    adir = Path(a.attempt_dir).resolve()
    cfg_path = adir / "check_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    sim = a.sim or cfg.get("sim")
    timeout_s = a.timeout_s or cfg.get("timeout_s") or 300
    spec_path = Path(a.spec) if a.spec else adir / "kernel_spec.json"
    if not sim or not spec_path.is_file():
        print("usage error: need --sim (or check_config.json) and a spec", file=sys.stderr)
        return 2
    tt_lang = json.loads(a.tt_lang_json) if a.tt_lang_json else cfg.get("tt_lang", {})
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    kernel, result_path, log_path = adir / "kernel.py", adir / "harness-result.json", adir / "harness-output.txt"
    result_path.unlink(missing_ok=True)
    timed_out, rc, text = False, None, ""
    start = time.monotonic()
    if kernel.is_file():
        try:
            done = subprocess.run([sim, str(HERE / "kernel_harness.py"), "--", "--spec", str(spec_path), "--kernel",
                                   str(kernel), "--out", str(result_path), "--phase", "sim"],
                                  capture_output=True, text=True, timeout=timeout_s, cwd=adir)
            rc, text = done.returncode, (done.stdout or "") + "\n" + (done.stderr or "")
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            text = exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    log_path.write_text(text, encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    category, line = classify(kernel_present=kernel.is_file(), timed_out=timed_out, returncode=rc, text=text,
                              result=result, bar=spec["pcc_bar_sim"], phase="sim")
    check = {"schema": 1, "op": spec["op"], "phase": "sim", "passed": category is None, "category": category,
             "first_line": line, "sets": (result or {}).get("sets", []), "pcc_bar": spec["pcc_bar_sim"],
             "provisional": True, "elapsed_s": round(time.monotonic() - start, 2), "tt_lang": tt_lang,
             "sha256": {"kernel.py": sha256_file(kernel), "kernel_harness.py": sha256_file(HERE / "kernel_harness.py"),
                        "references.py": sha256_file(HERE / "references.py"), "kernel_check.py": sha256_file(Path(__file__)),
                        "spec": sha256_file(spec_path)}}
    Path(a.out or adir / "check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print("PASS" if category is None else f"FAIL: {category}: {line}")
    return 0 if category is None else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Kernels and recorded outputs.** Using `~/orchard-venvs/ttlang/bin/tt-lang-sim` and the files under
  `~/orchard-src/tt-lang` (docs in `docs/sphinx/specs/TTLangSpecification.md`, examples in `examples/`, the negative examples in
  `examples/errors/`), write these kernels under `tests/fixtures/kernel-loop/kernels/`, each defining `OPERATION`:
  - `add_good.py`: upstream's `examples/eltwise_add.py` body with the function renamed `OPERATION` and the host code removed.
  - `add_wrong_op.py`: the same with `a_blk - b_blk` (must end `numeric_mismatch`).
  - `add_shape_error.py`: derived from `examples/errors/eltwise_add_error.py` (must end `shape_mismatch`).
  - `add_deadlock.py`: derived from `examples/errors/eltwise_add_deadlock.py` (must end `tt_lang_error`).
  - `add_syntax_error.py` (`import_error`), `add_no_operation.py` (a valid file with no `OPERATION`; `no_kernel`),
    `add_spin.py` (`OPERATION` that loops forever in Python; `timeout` with `--timeout-s 5`).
  - `silu_mul_good.py` and `rmsnorm_good.py`: working kernels, verified with the simulator at PCC >= 0.999. Read the
    specification for the unary and reduce operations. If after three focused attempts per op you cannot write one, stop and
    report `BLOCKED` with what you tried; do not weaken the op or the bar. A hand-written kernel for `linear_scan` is wanted
    if you can write one, and not required.
  For each of `numeric_mismatch`, `shape_mismatch`, `tt_lang_error` and `import_error`, run `kernel_check.py` on the matching
  kernel and save `harness-output.txt` as `tests/fixtures/kernel-loop/outputs/<category>.txt` and `harness-result.json` as
  `tests/fixtures/kernel-loop/outputs/<category>.result.json`, so `classify` has real text and a real result to test against
  without the venv.

- [ ] **Step 5: Tests** in `tests/test_kernel_loop_templates.py`. Import `kernel_check` by path (stdlib only, no torch). Tests:

```python
VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))
SIM = VENV / "bin" / "tt-lang-sim"
needs_sim = pytest.mark.skipif(not SIM.exists(), reason="SKIPPED: no tt-lang-sim in the ttlang venv (run `tt-orchard kernel-env`); "
                                                        "the kernel loop was not run on the simulator")


OUT = Path(__file__).parent / "fixtures" / "kernel-loop" / "outputs"


def _load_check():
    spec = importlib.util.spec_from_file_location(
        "kernel_check_under_test", REPO / "orchard" / "skills" / "kernel-loop-templates" / "kernel_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHECK = _load_check()


@pytest.mark.parametrize("name", ["numeric_mismatch", "shape_mismatch", "tt_lang_error", "import_error"])
def test_classify_reads_real_recorded_simulator_output(name):
    """Offline: the text and the result file are what the simulator and the harness really wrote (step 4)."""
    text = (OUT / f"{name}.txt").read_text()
    rp = OUT / f"{name}.result.json"
    result = json.loads(rp.read_text()) if rp.exists() else None
    got, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result, bar=0.999)
    assert got == name and line
```

  Then the simulator tests (marked `needs_sim`), each copying
  the templates and a fixture kernel into `tmp_path` and running `kernel_check.py` as a subprocess with `--sim`, asserting
  exit code, `check.json["category"]` and `check.json["passed"]`:
  `add_good` passes; `add_wrong_op` is `numeric_mismatch`; `add_shape_error` is `shape_mismatch`; `add_deadlock` is
  `tt_lang_error`; `add_syntax_error` is `import_error`; `add_no_operation` is `no_kernel`; `add_spin` with `--timeout-s 5` is
  `timeout`. Plus:
  - `test_a_spec_cannot_name_other_code`: a spec copy with `"reference": "os.system"` gives `harness_error`.
  - `test_without_the_diagnostics_file_the_check_reports_the_environment` (classify given
    `FileNotFoundError ... diagnostics.py` text and a non-zero exit returns `harness_error`, not `tt_lang_error`).
  - `test_check_json_hashes_the_files_that_took_part` (all five hashes present; `kernel.py` hash matches the file).
  - `test_references_match_independent_formulas`: run in the venv's python as a subprocess (the suite's own interpreter may have
    no torch): for `silu_mul`, `rmsnorm` and `linear_scan` compare against a hand-written loop or formula on a 4x8 input.
  - `test_the_good_kernels_pass_on_the_simulator` for `silu_mul` and `rmsnorm` (skip with a `BLOCKED` note in the report if
    step 4 could not produce them).

- [ ] **Step 6: Mutations** (each must turn a named test red): bar check removed from `classify` →
  `add_wrong_op`; `SHAPE_RE` branch removed → `add_shape_error`; the `diagnostics.py` branch removed → the environment
  test; `timed_out` ignored → `add_spin`; spec reference looked up with `eval` or `getattr` of an arbitrary module →
  `test_a_spec_cannot_name_other_code`; harness `finite` always True → add a kernel fixture that writes NaN (`add_nan.py`,
  `numeric_mismatch`) and use it here; heavy-tail set removed → a test that `check.json["sets"]` has both `seed0` and `seed0-tail`.

- [ ] **Step 7: Commit** `git add orchard/skills/kernel-loop-templates tests/fixtures/kernel-loop tests/test_kernel_loop_templates.py`
  then `git commit -m "kernel probe: references, harness and check for the simulator"`.

---

### Task 9: `orchard/kernelprobe.py` and the agent skill

**Files:**
- Create: `orchard/kernelprobe.py`, `orchard/skills/kernel-probe.md`
- Test: `tests/test_kernelprobe.py`

**Interfaces:**
- Consumes: `agent.AgentStep`, `agent.Tools`, `agent.agent_env`, `ledger.Ledger`, `watchdog.RetryGuard`, `stallwatch.run_watched`,
  `stallwatch.StallWatcher`, `ttlang_env.KEnv`, the Task 8 templates, `tests/fake_model.py` (`FakeModel`, `call`, `final`).
- Produces: `OPS`, `BUDGET`, `load_spec(op)`, `prepare_attempt(adir, *, op, env)`, `authoritative_check(adir, *, op, env, run_watched=run_watched) -> dict`,
  `ProbeControl`, `run_attempt(...) -> dict`, `smoke_test(...)`, `hardware_phase(...)`, `write_report(probe_dir, attempts, header)`,
  `provenance(...)`, `ProbeBlocked`, and `main(...)` (Task 10 wires the command).

Design points that the tests pin:
- The agent works in `DIR/<op>/attempt-<n>/` with `kernel_spec.json`, copies of the three templates, `check_config.json`, and
  `reference/eltwise_add.py` copied from `<src_dir>/examples/eltwise_add.py` (not vendored into this repository). It writes only
  `kernel.py`.
- **The verdict never comes from a file the agent can write.** After the agent ends, `authoritative_check` runs the SHIPPED
  `kernel_check.py` (the one under `orchard/skills/kernel-loop-templates/`, not the copy in the attempt directory) with
  `--spec` pointing at the SHIPPED `ops/<op>.json`, `--sim`, `--timeout-s` and `--tt-lang-json` passed explicitly, and
  `--out` set to `check-authoritative.json`. It runs under `stallwatch.run_watched`. If the agent's own `check.json` says
  `passed` and the authoritative one does not, or the agent's `check.json` hashes differ from the shipped templates', the
  attempt records `agent_check_untrusted: true`.
- Budget (provisional, in `BUDGET`): 40 turns, 1800 s wall time for the agent (via `ProbeControl.stop_reason`), 300 s per
  simulator run, 600 s backstop for a hardware run.
- The probe starts no server. If the endpoint does not answer `agent.probe_model(endpoint, model)`, `main` returns 2 with a
  message.
- `attempt.json`: `{op, attempt, status (the Outcome's), turns, agent_wall_s, passed_sim, category, first_line, pcc_min, sets,
  hw: null | {...}, agent_check_untrusted, provenance}`.
- `provenance`: `{origin: "kernel", kernel_sha256, tt_lang: {sim_version, source_sha, diagnostics_patched, patch_sha256},
  coder_model, probe_id, simulator_pcc_min, hardware_pcc_min}`.

- [ ] **Step 1: Write the failing tests.** The core ones in code (add the rest from the list after):

```python
# tests/test_kernelprobe.py
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from orchard import kernelprobe as kp, ttlang_env
from fake_model import FakeModel, call, final

REPO = Path(__file__).resolve().parent.parent
KERNELS = Path(__file__).parent / "fixtures" / "kernel-loop" / "kernels"
VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))
SRC = Path(os.environ.get("ORCHARD_TTLANG_SRC", Path.home() / "orchard-src" / "tt-lang"))
needs_sim = pytest.mark.skipif(not (VENV / "bin" / "tt-lang-sim").exists() or not (SRC / "examples").is_dir(),
                               reason="SKIPPED: no tt-lang-sim venv or tt-lang checkout (run `tt-orchard kernel-env`)")


@pytest.fixture
def kenv(tmp_path):
    return ttlang_env.KEnv(src_dir=SRC, venv_dir=VENV, scratch_dir=tmp_path / "scratch")


def scripted(*messages):
    it = iter(messages)
    return lambda request: next(it, final("done"))


@needs_sim
def test_an_agent_that_writes_a_good_kernel_passes_and_the_report_records_it(tmp_path, kenv):
    good = (KERNELS / "add_good.py").read_text()
    with FakeModel(scripted(call("write_file", path="kernel.py", content=good),
                            call("shell", command="python3 kernel_check.py"), final("done"))) as server:
        attempt = kp.run_attempt(op="add", n=1, probe_dir=tmp_path, env=kenv, endpoint=server.endpoint, model="fake",
                                 coder_model="fake", probe_id="t", max_turns=10, wall_s=120)
    assert attempt["passed_sim"] is True and attempt["category"] is None and attempt["agent_check_untrusted"] is False
    assert (tmp_path / "add" / "attempt-1" / "attempt.json").is_file()


@needs_sim
def test_an_agent_that_fakes_its_own_check_is_not_believed(tmp_path, kenv):
    """Review focus 3. The agent replaces kernel_check.py with a script that prints PASS and writes a passing check.json."""
    liar = ("import json,sys\n"
            "json.dump({'passed': True, 'category': None, 'sha256': {}}, open('check.json','w'))\n"
            "print('PASS')\n")
    bad = (KERNELS / "add_wrong_op.py").read_text()
    with FakeModel(scripted(call("write_file", path="kernel.py", content=bad),
                            call("write_file", path="kernel_check.py", content=liar),
                            call("shell", command="python3 kernel_check.py"), final("done"))) as server:
        attempt = kp.run_attempt(op="add", n=1, probe_dir=tmp_path, env=kenv, endpoint=server.endpoint, model="fake",
                                 coder_model="fake", probe_id="t", max_turns=10, wall_s=120)
    assert attempt["passed_sim"] is False and attempt["category"] == "numeric_mismatch"
    assert attempt["agent_check_untrusted"] is True


def test_the_authoritative_check_uses_the_shipped_spec_and_not_the_attempts_copy(tmp_path, kenv):
    """The agent can edit kernel_spec.json in its directory (lower the bar, change the reference). The verdict must not read it."""
    seen = {}

    def fake_run_watched(argv, **kw):
        seen["argv"] = list(argv)
        (Path(argv[argv.index("--out") + 1])).write_text(json.dumps({"passed": False, "category": "numeric_mismatch", "first_line": "x", "sets": []}))
        return kp.Watched(1, None, False, 0.1)

    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=kenv)
    (adir / "kernel_spec.json").write_text("{}")                       # tampered
    kp.authoritative_check(adir, op="add", env=kenv, run_watched=fake_run_watched)
    argv = seen["argv"]
    assert Path(argv[1]) == kp.TEMPLATES / "kernel_check.py"
    assert Path(argv[argv.index("--spec") + 1]) == kp.TEMPLATES / "ops" / "add.json"


def test_a_check_killed_for_a_stall_is_a_timeout_or_hang_never_a_pass(tmp_path, kenv):
    def stalled(argv, **kw):
        return kp.Watched(-9, "stalled: no output and no CPU use for 120 s", True, 130.0)
    adir = tmp_path / "add" / "attempt-1"
    kp.prepare_attempt(adir, op="add", env=kenv)
    got = kp.authoritative_check(adir, op="add", env=kenv, run_watched=stalled)
    assert got["passed"] is False and got["category"] == "timeout" and "stalled" in got["first_line"]


def test_the_budget_ends_the_agent_through_the_control_object():
    now = [0.0]
    ctl = kp.ProbeControl(wall_s=100, clock=lambda: now[0])
    assert ctl.stop_reason() is None and ctl.take_nudges("a") == []
    now[0] = 101.0
    assert "wall" in ctl.stop_reason()


def test_hardware_commands_run_only_through_gozer_and_only_after_a_passing_smoke_test(tmp_path, kenv):
    calls = []

    def run_watched_fake(argv, **kw):
        calls.append(list(argv))
        return kp.Watched(0, None, False, 1.0)

    ok, _ = kp.smoke_test(env=kenv, shared_python="/shared/python", run_watched=run_watched_fake)
    assert calls[0][:2] == ["gozer", "run"] and "--chips" in calls[0] and calls[0][calls[0].index("--chips") + 1] == "1"
    assert calls[0][calls[0].index("--") + 1] == "/shared/python"
```

  More tests, each written out before the code: `test_a_hardware_hang_blocks_the_probe_and_names_the_attempt` (a fake
  `run_watched` returning `killed=True, reason="stalled..."` makes `hardware_phase` raise `ProbeBlocked` whose message names the
  op and attempt, and no second hardware call follows); `test_a_failed_smoke_test_makes_the_report_say_hw_unavailable_and_no_gozer_call_for_kernels_follows`;
  `test_the_report_header_lists_versions_patch_hashes_coder_and_provisional_bars`;
  `test_write_report_counts_attempts_to_first_pass_per_op`; `test_prepare_attempt_copies_the_reference_example_from_the_checkout_and_vendors_nothing`
  (no file under `orchard/` or `tests/` is a copy of `eltwise_add.py`); `test_the_agent_environment_has_no_device_and_no_network_tools`
  (`agent_env(adir)` carries the no-chip device mask; a shell call `pip install x` through the real `Tools` is `refused:`);
  `test_a_model_that_repeats_the_same_call_is_stopped_by_the_watchdog` (see step 3, only if the watchdog can be wired).

- [ ] **Step 2: Write the skill** `orchard/skills/kernel-probe.md`:

```markdown
# Write a tt-lang kernel

You are in a directory that holds `kernel_spec.json` (the op), `kernel_check.py` (the check), `reference/eltwise_add.py` (a
working tt-lang example) and a few support files. You write one file: `kernel.py`.

1. Read `kernel_spec.json`. It lists the inputs and their shapes, the output shape, and what the op computes.
2. Read `reference/eltwise_add.py` once. It shows how a tt-lang operation is declared (`@ttl.operation(grid="full")`), how
   dataflow buffers carry tiles between the reader, compute and writer functions, and how a result is stored.
3. Write `kernel.py`. It must define `OPERATION`, a function decorated with `@ttl.operation` that takes the input tensors in the
   order `kernel_spec.json` lists them, then the output tensor last. At the top level of `kernel.py` put only imports,
   constants and that function.
4. Run `python3 kernel_check.py`. It prints `PASS`, or `FAIL: <category>: <first line>`. The first line is the simulator's own
   message.
5. Read that line. Change one thing. Run it again.

Rules:
- Write only `kernel.py`. Do not edit any other file.
- Do not install packages, use the network, or touch any device. The check runs on a simulator.
- If the same failure comes back three times and you have no new idea, stop and say what you tried.
- Finish with at most five lines: what the kernel does, and what the last check said.
```

- [ ] **Step 3: Implement `orchard/kernelprobe.py`.** Build it in this order, running the tests after each part: constants,
  `load_spec`, `prepare_attempt`, `ProbeControl`, `authoritative_check`, `run_attempt`, `smoke_test`, `hardware_phase`,
  `write_report`, `provenance`. The agent step is wired exactly as the supervisor wires one (`orchard/supervisor.py`, `_step`,
  around line 1388); the code to write is:

```python
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from orchard import defaults, ttlang_env
from orchard.agent import AgentStep, Tools, agent_env
from orchard.canary import post_json
from orchard.ledger import Ledger
from orchard.stallwatch import StallWatcher, Watched, run_watched
from orchard.watchdog import (IdenticalResponses, Ladder, NoFileWritten, RepeatedToolCall, RetryGuard,
                              ThinkingWithoutAction, TurnRepeat, Watchdog)

TEMPLATES = Path(__file__).resolve().parent / "skills" / "kernel-loop-templates"
SKILL = Path(__file__).resolve().parent / "skills" / "kernel-probe.md"
OPS = ("add", "silu_mul", "rmsnorm", "linear_scan")
# provisional: none of these has been measured (spec section 5)
BUDGET = {"max_turns": 40, "wall_s": 1800.0, "sim_timeout_s": 300.0, "hw_backstop_s": 600.0}
GOZER_WHO = "claude:orchard-kernel-probe"


class ProbeBlocked(Exception):
    """A hardware run hung. The probe stops and names the attempt; nothing is reset by hand."""


def load_spec(op: str) -> dict:
    return json.loads((TEMPLATES / "ops" / f"{op}.json").read_text(encoding="utf-8"))


def _template(name: str):
    """A template script loaded by path, for its classify() function."""
    spec = importlib.util.spec_from_file_location(f"kernel_loop_{name}", TEMPLATES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tt_lang_facts(env) -> dict:
    """What a report says about the tt-lang the kernels ran on, including that the wheel carries a local patch."""
    target = ttlang_env.diagnostics_target(env)
    patched = bool(target and target.is_file() and ttlang_env.sha256_file(target) == defaults.TTLANG_DIAGNOSTICS_SHA256)
    return {"sim_version": ttlang_env.sim_version(env), "source_sha": ttlang_env.head_sha(env),
            "diagnostics_patched": patched, "patch_sha256": defaults.TTLANG_DIAGNOSTICS_SHA256}


def prepare_attempt(adir: Path, *, op: str, env) -> None:
    adir.mkdir(parents=True, exist_ok=True)
    for name in ("kernel_check.py", "kernel_harness.py", "references.py"):
        shutil.copy2(TEMPLATES / name, adir / name)
    shutil.copy2(TEMPLATES / "ops" / f"{op}.json", adir / "kernel_spec.json")
    (adir / "check_config.json").write_text(json.dumps({"sim": str(env.sim), "timeout_s": BUDGET["sim_timeout_s"],
                                                        "tt_lang": tt_lang_facts(env)}, indent=2) + "\n", encoding="utf-8")
    ref = Path(env.src_dir) / "examples" / "eltwise_add.py"
    if ref.is_file():
        (adir / "reference").mkdir(exist_ok=True)
        shutil.copy2(ref, adir / "reference" / "eltwise_add.py")


class ProbeControl:
    """The supervisor's actuator reduced to what a probe attempt needs: the three things the watchdog's Ladder calls
    (nudge, escalate, pause) and the two things AgentStep asks (take_nudges, stop_reason). A wall-clock stop is the
    backstop; the watchdog's findings are the primary stop."""

    def __init__(self, *, wall_s: float, clock=time.monotonic):
        self.deadline, self.clock = clock() + wall_s, clock
        self.nudges, self.stop = [], None

    def nudge(self, agent, message):
        self.nudges.append(message)

    def escalate(self, agent, stage):
        self.stop = "the watchdog saw the agent loop again after a nudge"

    def pause(self, reason, evidence):
        self.stop = reason

    def take_nudges(self, agent):
        out, self.nudges = self.nudges, []
        return out

    def stop_reason(self):
        if self.stop:
            return self.stop
        return "wall-clock budget spent" if self.clock() >= self.deadline else None


def authoritative_check(adir: Path, *, op: str, env, run_watched=run_watched) -> dict:
    """Run the SHIPPED check on the agent's kernel.py. The agent's directory holds copies of the templates and the
    spec, and the agent can edit them, so nothing the verdict depends on is read from there except kernel.py."""
    out = adir / "check-authoritative.json"
    out.unlink(missing_ok=True)
    argv = [sys.executable, str(TEMPLATES / "kernel_check.py"), "--attempt-dir", str(adir),
            "--spec", str(TEMPLATES / "ops" / f"{op}.json"), "--sim", str(env.sim),
            "--timeout-s", str(BUDGET["sim_timeout_s"]), "--tt-lang-json", json.dumps(tt_lang_facts(env)),
            "--out", str(out)]
    got = run_watched(argv, cwd=adir, env=None, log_path=adir / "authoritative-output.txt",
                      watcher=StallWatcher(quiet_s=BUDGET["sim_timeout_s"] / 2), backstop_s=BUDGET["sim_timeout_s"] + 120)
    if got.killed:
        return {"passed": False, "category": "timeout", "first_line": got.reason, "sets": []}
    if not out.is_file():
        return {"passed": False, "category": "harness_error", "first_line": f"kernel_check.py wrote no result (exit {got.returncode})", "sets": []}
    return json.loads(out.read_text(encoding="utf-8"))
```

  `run_watched`'s `env=None` means the child inherits the probe's environment; that is correct here because the shipped check
  needs `PATH`. It is not an agent shell.

  `run_attempt(*, op, n, probe_dir, env, endpoint, model, coder_model, probe_id, max_turns=..., wall_s=..., http=post_json,
  clock=time.time, watchdog=None)`:
  1. `adir = probe_dir / op / f"attempt-{n}"`; `prepare_attempt`.
  2. `ledger = Ledger(adir / "ledger.jsonl")`; `control = ProbeControl(wall_s=wall_s)`; `watchdog = Watchdog([IdenticalResponses(),
     RepeatedToolCall(), TurnRepeat(), NoFileWritten(), ThinkingWithoutAction()], Ladder(control, ledger, {"kernel-probe"}))`
     (read each detector's constructor in `orchard/watchdog.py` and use its defaults unless a test shows a default is wrong for a
     40-turn attempt); build `AgentStep(agent="kernel-probe", endpoint=endpoint, model=model,
     tools=Tools(adir, adir, agent_env(adir)), ledger=ledger, stage=0, phase="probe", feed=watchdog.feed,
     control=control, run_dir=adir, evidence_dir=adir / "evidence",
     log_path=adir / "log" / "probe-00001.jsonl", http=http, clock=clock, guard=RetryGuard(), max_turns=max_turns)`.
  3. `system = SKILL.read_text()`; `user = f"Write kernel.py for the op `{op}` in kernel_spec.json. Start by reading kernel_spec.json and reference/eltwise_add.py."`
     `outcome = step.run(system, user)`; close the ledger (`ledger.close()` in a `finally`).
  4. `check = authoritative_check(...)`; read the agent's own `check.json` if present; set
     `agent_check_untrusted = bool(agent_check and (agent_check.get("passed") and not check["passed"] or
     agent_check.get("sha256", {}).get("kernel_check.py") not in (None, sha256 of the shipped template)))`. Only
     an agent `check.json` that exists can be untrusted; a missing one is `False`.
  5. Write and return `attempt.json` as specified above (`pcc_min` is the smallest non-null `pcc` in `check["sets"]`, else null).

  **Watchdog (the operator asked for watchers).** `tests/test_agent.py` already runs a real `Watchdog` with an `AgentStep`
  outside the supervisor (`Watchdog([IdenticalResponses()], Ladder(control, ledger, {"stage-agent"}))`, `feed=wd.feed`), so
  the wiring above is required. Add `test_a_model_that_repeats_the_same_call_is_stopped_by_the_watchdog`: a `FakeModel` that
  answers one identical `shell` call with identical visible text every turn (see `repeating()` in `tests/test_agent.py`) must
  end the attempt well before `max_turns`, with the watchdog's `retry` and `escalate` entries in the attempt's ledger. Also
  add `test_a_model_that_never_writes_kernel_py_is_nudged` using `NoFileWritten`.

  `smoke_test(*, env, shared_python, run_watched=run_watched)`: runs
  `["gozer", "run", "--chips", "1", "--exact", "1", "--who", GOZER_WHO, "--reason", "kernel-probe smoke test", "--",
  shared_python, str(Path(env.src_dir) / "examples" / "eltwise_add.py")]` under `run_watched` with
  `StallWatcher(quiet_s=60)` and backstop `BUDGET["hw_backstop_s"]`, logging to `env.scratch_dir / "smoke.txt"`. Returns
  `(ok, last_lines)` where `ok` needs return code 0, not killed, and `PASSED!` in the log.

  `hardware_phase(adir, attempt, *, env, shared_python, run_watched=run_watched)`: only called for an attempt whose
  authoritative simulator check passed. Runs `["gozer", "run", "--chips", "1", "--exact", "1", "--who", GOZER_WHO, "--reason",
  f"kernel-probe {op} attempt {n}", "--", shared_python, str(TEMPLATES / "kernel_harness.py"), "--spec", <shipped spec>,
  "--kernel", str(adir / "kernel.py"), "--out", str(adir / "hw-result.json"), "--phase", "hw"]` under `run_watched` with
  `StallWatcher(quiet_s=60)`. If `killed`, record category `hang`, then raise `ProbeBlocked(f"{op} attempt {n}: {reason}")`. Otherwise
  classify with `_template("kernel_check").classify(..., phase="hw", bar=spec["pcc_bar_hw"])` and return
  `{"passed", "category", "first_line", "pcc_min", "sets"}`.

  `write_report(probe_dir, attempts, header)`: writes `kernel-probe-report.json` (header, attempts) and
  `kernel-probe-report.md`: a header with the tt-lang versions and source SHA, the patch hashes, the coder's model id and the
  provisional bars; then per op a table of attempts (simulator pass, hardware pass or `-`, PCC min, turns, wall time,
  category, first line, `untrusted`) and "attempts to first pass". The text says: attempts are independent draws and the
  inputs are synthetic.

  `provenance(...)` returns the block from the design points above; `run_attempt` puts it in `attempt.json`.

- [ ] **Step 4: Run the new tests, then the whole suite** (`python3 -m pytest -q`).

- [ ] **Step 5: Mutations:** verdict read from the attempt's copy of `kernel_check.py` (use `adir / "kernel_check.py"` in the
  argv) → the liar test; spec read from `adir / "kernel_spec.json"` → the shipped-spec test; `agent_check_untrusted` always
  False → the liar test; a killed check treated as `passed` → the stall test; hardware phase run without `gozer run` → the
  smoke and hardware tests; `ProbeBlocked` not raised on a kill → the hang test; `ProbeControl` never stopping → the budget
  test; `check_config.json` omitting the tt-lang facts → the report-header test.

- [ ] **Step 6: Commit** `git add orchard/kernelprobe.py orchard/skills/kernel-probe.md orchard/ttlang_env.py tests/test_kernelprobe.py`
  then `git commit -m "kernel probe: attempts, an authoritative re-check, the hardware phase and the report"`.

---

### Task 10: the `kernel-probe` command

**Files:**
- Modify: `orchard/cli.py`, `README.md`, `orchard/kernelprobe.py` (the `main` function)
- Test: additions to `tests/test_cli.py`, `tests/test_kernelprobe.py`

**Interfaces:**
- Produces: `tt-orchard kernel-probe --op NAME [--attempts N] [--phase sim|hw] [--out DIR] [--endpoint URL] [--model NAME]
  [--src-dir D] [--venv-dir D] [--shared-python P]`; `kernelprobe.main(args, *, env=None, out=None, http=post_json,
  probe=agent.probe_model) -> int` with exit 0 (a report was written, whatever the results), 2 (refused: unknown op, endpoint
  down, environment not ready), 5 (blocked by a hardware hang, report written for what ran).

- [ ] **Step 1: Read** how the supervisor resolves the coder endpoint (`orchard/stages.py` `tier_for`, `resolve_endpoint`;
  `orchard/tiers.py` `load`; `orchard/bringup_config.py` for where the tiers file path lives). If the bringup config exposes the
  tier config, the defaults for `--endpoint` and `--model` come from the tier stage 2 uses (`tier_for(cfg, 2, phase="main",
  escalated=False)`); otherwise both flags are required. `--endpoint` and `--model` always win.

- [ ] **Step 2: Failing tests.** In `tests/test_cli.py`: `kernel-probe` with an unknown `--op` is refused (exit 2) and calls nothing;
  with a valid op it calls `kernelprobe.main` with the parsed flags. In `tests/test_kernelprobe.py`:
  - `test_main_refuses_when_the_endpoint_does_not_answer` (a `probe` that returns False: exit 2, no attempt directory created,
    and no `gozer` call).
  - `test_main_refuses_when_the_environment_check_fails` (`ttlang_env.ensure(check_only=True)` returns a FAIL or TODO: exit 2
    with the message to run `tt-orchard kernel-env`; nothing else runs).
  - `test_main_runs_n_attempts_writes_the_report_and_exits_0_even_when_every_attempt_fails` (a `FakeModel` that never writes a
    kernel; three attempts; the report exists; exit 0; every category is `no_kernel`).
  - `test_phase_sim_never_calls_gozer` (a tripwire `run_watched` that fails the test if argv[0] is `gozer`).
  - `test_phase_hw_runs_the_smoke_test_once_and_hardware_only_for_a_sim_pass` (fake `run_watched`; two attempts, one passing in
    the simulator; exactly one smoke call and one hardware call).
  - `test_a_hardware_hang_exits_5_and_the_report_lists_what_ran`.

- [ ] **Step 3: Implement** `main` (calls `ttlang_env.ensure(check_only=True)`, then `probe(endpoint, model)`, then the
  attempts, then the hardware phase when requested, then `write_report`, printing one line per attempt as it ends) and the
  subparser in `cli.py` with `choices=kernelprobe.OPS` for `--op`. README rows for every flag and a short "Kernel probe"
  section: what it measures, that it starts no server, that the verdict is the probe's own re-check, that numbers are
  provisional, and that the hardware phase leases one chip and runs a smoke test first.

- [ ] **Step 4: Mutations:** endpoint check removed → the endpoint test; `ensure` not consulted → the environment test; `gozer`
  reached in the sim phase → `test_phase_sim_never_calls_gozer`; smoke run per attempt → the smoke-count test; exit code 0 on
  a hang → the hang test.

- [ ] **Step 5: Commit** `git add orchard/cli.py orchard/kernelprobe.py README.md tests/test_cli.py tests/test_kernelprobe.py` then
  `git commit -m "kernel probe: the tt-orchard kernel-probe command"`.

---

### Task 11: the operator bundle shows the inventory

**Files:**
- Modify: `orchard/skills/operator-bundle-templates/build_bundle.py`
- Test: `tests/test_operator_bundle_template.py` (additions)

- [ ] **Step 1: Read** `build_bundle.py` and how it writes `RESULTS.md`; find an existing test that builds a bundle from a
  fake run directory.
- [ ] **Step 2: Failing tests:** (a) a run directory whose `stages/0/ops-inventory.json` exists gets a "Ops and kernels" section in
  `RESULTS.md` that states the model's traced class, the distinct counts by status, the dump version or the reason there is
  none, up to 10 unmapped op names, and the sentence that `mapped` is a name match; (b) a run directory with no such file gets
  no such section and the rest of `RESULTS.md` is byte-identical to before; (c) a skipped inventory prints its reason; (d)
  text copied from the inventory goes through the same redaction the rest of the bundle uses (a home path in a module name is
  replaced).
- [ ] **Step 3: Implement** the section, reusing the script's redaction helper. Standard library only (this is a template).
- [ ] **Step 4: Mutations:** section always written → (b); redaction skipped → (d); skipped reason dropped → (c).
- [ ] **Step 5: Commit** `git add orchard/skills/operator-bundle-templates tests/test_operator_bundle_template.py` then
  `git commit -m "bundle: show the op inventory in RESULTS.md when stage 0 recorded one"`.

---

### Task 12: version, docs, spec amendments, log, final checks

**Files:** `pyproject.toml` and `orchard/__init__.py` (version `0.6.0`; `tests` require them equal), `README.md`, `CLAUDE.md`,
`docs/superpowers/specs/2026-10-15-kernel-attempt-design.md`

- [ ] **Step 1: Bump the version** to 0.6.0 in both places.
- [ ] **Step 2: Amendments to the spec**, made in the spec file itself and listed in a new final section "Amendments (2026-10-15,
  during implementation)":
  1. The probe has a fourth op, `add`, a calibration op whose working kernel is upstream's example, so a fault in the harness or
     the environment shows up as a failure of a kernel that must pass.
  2. Spec 5.2's hash check on the agent's `check.json` is replaced by a stronger rule: the probe re-runs the shipped check
     itself and records `agent_check_untrusted` when the agent's own result disagrees or its hashes differ.
  3. A tenth failure category, `harness_error`, for faults in the templates or the environment.
  4. The export call is `torch.export.export(model, (), {"input_ids": ids, "use_cache": False}, strict=False)`, measured to
     trace Qwen3, RWKV, Mamba2 and Moshi's text model on 2026-10-15 (without `use_cache=False` three of four failed). A
     dispatch-level recorder was tried and fails on Moshi.
  5. `orchard/stallwatch.py` (operator instruction, 2026-10-15: "Instead of pure timers, we need watchers"): every long child
     process runs under a watcher that kills on a stall (no output and no CPU) or a repeating output line; deadlines are a
     backstop. Spec 5.3 and 5.4's budgets are backstops.
  6. `scaled_count` is `count_in_layers x (layers in config / layers traced) + count outside layers`, and the report says when
     it is not scaled.
- [ ] **Step 3: CLAUDE.md** log entry for the implementation: the prompt, what was built per task, measured results from the
  live steps below, mistakes and notable moments (as in earlier entries), and "Not done".
- [ ] **Step 4: Full verification.** `python3 -m pytest -q` (record passed/skipped counts; the only expected skips are the
  opt-in replay and tests that need the simulator venv or torch if those are absent from the suite's interpreter, each with its
  `SKIPPED:` message). Then run the sweep over every new or changed doc, docstring and comment text:
  `git diff main...HEAD -- '*.md' '*.py' | grep -n -i -E ', not |rather than|instead of|reads as|honest|the one |which is why|This is what'`
  and rewrite each hit or justify it in the report.
- [ ] **Step 5: Commit** `git add -A pyproject.toml orchard/__init__.py README.md CLAUDE.md docs` (check `git status` first) then
  `git commit -m "kernel attempt: version 0.6.0, spec amendments and log"`.

---

## Live verification (the controller does these, not a subagent)

These need real hardware, a real model server, or network, and run after Task 12 on the branch's worktree.

- **L1. Environment.** `tt-orchard kernel-env --check` against the existing `~/orchard-src/tt-lang` and `~/orchard-venvs/ttlang`
  must report every step ok (the clone is at the pin, the wheel version is `1.1.10+g9b7ad24`, the diagnostics patch is in place, and
  verification passes by diagnostic text).
- **L2. Inventory.** `tt-orchard inventory` on a config directory for Qwen3 (a tiny config), RWKV, Mamba2 and Moshi's text model
  (`--class MoshiForCausalLM`), with the reference interpreter. Each must write both files. With no ttnn dump yet, every op
  must be `unchecked`.
- **L3. ttnn dump.** `tt-orchard inventory refresh-ttnn` under a lease, then L2 again: statuses must now split into
  mapped, composable and unmapped, with the dump's version named. Read the unmapped list for each model and report which ops
  are missing, which is the first answer to the original question.
- **L4. Probe, simulator phase.** With a coder server running (Coder-Next; started by the controller under a lease, the
  existing `tt-orchard bringup` machinery or `tt-model serve`, never by a subagent), `tt-orchard kernel-probe --op add
  --attempts 3`, then `silu_mul`, `rmsnorm`, `linear_scan`, five attempts each. Report the table, including how many
  `harness_error` results appeared (each is a fault in the instrument, to be fixed first).
- **L5. Probe, hardware phase.** Only if L4's `add` passes: the smoke test, then one `add` attempt with `--phase hw`.

Stop conditions for the live steps: a `harness_error`, a hung process, a lease held longer than its run, or a model server that
stops answering ends the step and is reported, not retried in a loop.

---

## Task 13: UI parity (added 2026-10-16, operator request: "make sure this all has parity in the UI")

**Goal:** `tt-orchard ui` shows everything the new commands produce: the op inventory of a run, the kernel environment's
status, the list of kernel probes with live progress, and a probe's attempts with each kernel. The page stays read-only for
kernel work: it launches no probe, no inventory and no hardware action. (Ruling: the page has no login with `--lan`; a button
that starts hardware-capable runs is a security decision for the operator. The existing `bringup` launch and the pause, resume
and abort controls are unchanged.)

**Prerequisites:** the final fix wave (branch `ka-final-fixes`) is merged into `kernel-attempt`, and the live probe batch has
ended (the batch uses the shipped templates from that checkout). Work in a new worktree from the merged head.

**Precedent to copy:** the base research view. `orchard/webui.py` `research_view` (a bounded, redacted summary; text from a
downloaded repo is never trusted), `run_research`, the route `/api/runs/<name>/research`, the file allow-list regex `ALLOWED`
(line ~103), `REPORT_MAX_BYTES`, and `tests/test_webui_research.py`. Read `orchard/webui.py` (1273 lines) and
`orchard/web/{index.html,app.js,app.css}` before writing anything; read the module docstring's list of what the page may write
and read.

### 13a. Make probes discoverable (CLI side; `orchard/kernelprobe.py`, `orchard/cli.py`, `README.md`)
- `kernelprobe.probe_root(runs_root=None)` returns `<runs root>/kernel-probes`, where the runs root is the bringup config's
  runs directory when a config is found (the way `cli._inventory` and `bringup_config.run_dir` find it) and `~/orchard-runs`
  otherwise. `--out` defaults to `<probe root>/<UTC stamp>-<op>` (was `./kernel-probe`); an explicit `--out` is unchanged.
  Update the README flag text and the parser help.
- `main` writes `probe.json` into the probe directory at the start and rewrites it at the end (atomic: temp file in the same
  directory plus `os.replace`; never through a link): `{"probe_id", "op", "attempts", "phase", "pid", "started_at", "state",
  "endpoint" (credentials stripped), "coder_model"}` with `state` `running` at the start and, at the end, one of `finished`,
  `blocked`, `crashed`, `interrupted`, plus `exit_code` and `finished_at`. Writing it must never stop a probe (an OSError is
  ignored).
- Tests first, a mutation each: default out under the probe root; explicit `--out` unchanged; `probe.json` is `running` before
  the first attempt and carries the final state and exit code for each exit path (0, 3, 4, 5); atomic write (a crash between the
  write and the replace leaves no truncated file); credentials stripped from the endpoint.

### 13b. Server (`orchard/webui.py`)
All new text read from disk is bounded, redacted with the existing redaction (home paths, tokens, host name) and set in the
page with `textContent`. All reads of files an agent could have written use a capped regular-file reader (no symlink follow, no
FIFO, size cap); the probe-owned `.verdict/<op>-<n>/` files are preferred over anything under the attempt directories. IDs and
names that name directories are matched by strict ASCII regexes before use.
- `GET /api/runs/<name>/inventory`: the view of `stages/0/ops-inventory.json` (2 MiB cap): traced class, `skipped`, `failures`,
  the dump version or note, counts by status (`mapped`, `composable`, `unmapped`, `unchecked`, `ignored`), up to 20 `unmapped`
  ops (most frequent by scaled count first, op name only), `python_nodes` and `higher_order_nodes` counts, `scaling`, and the
  label sentence that `mapped` is a name match. Add `stages/0/ops-inventory.(json|md)` to the allow-list.
  Malformed input never raises: the view says the file is unreadable.
- `GET /api/kernel/env`: the status of the simulator environment WITHOUT running the verification examples: clone at the pin,
  venv present, simulator version, diagnostics file present with the pinned sha256 (a read-only version of
  `ttlang_env` steps; add a `ttlang_env.status(env) -> list[Result]` that runs no `uv`, no `git` write and no example; it may
  call `git rev-parse` and `tt-lang-sim --version`). Cached for 60 s like the gozer cache. States `ok`, `missing`, `stale`,
  with the command to run (`tt-orchard kernel-env`).
- `GET /api/probes`: up to 50 probe directories under the probe root (newest first) that hold a `probe.json` or a
  `kernel-probe-report.json`: id, op, state, attempts done of total (from the records under `.verdict`), simulator passes,
  hardware state (`not run`, `unavailable`, `blocked`, or passes of runs), coder model, started and finished times. A
  `running` state whose pid is no longer alive (use the same liveness helper `orchard/status.py` uses for supervisors, which
  checks the process start time) is reported as `died`.
- `GET /api/probes/<id>`: the header (tt-lang versions and source SHA, patch applied, bars with the word provisional, coder,
  budgets, the `hw` line) and up to 200 attempt rows: attempt number, status and first line of the status detail when not
  `done`, simulator pass, category, first line, turns, wall seconds, smallest PCC, largest relative L2 error, slope range,
  `agent_check_untrusted`, `violations` count, swept stray processes count, hardware outcome. Built from `kernel-probe-report.json`
  when present, and from the `.verdict` records while the probe runs.
- `GET /api/probes/<id>/kernel?op=<op>&n=<int>`: the text of the probe-owned clean copy
  `.verdict/<op>-<n>/authoritative/kernel.py` (256 KiB cap, redacted); `GET /api/probes/<id>/report`: the report Markdown
  (1 MiB cap). No other file under a probe directory is served.
- Tests first in a new `tests/test_webui_kernels.py` in the style of `tests/test_webui_research.py`, a mutation per guard:
  redaction of every text field (plant a home path and an `hf_` token in an op name, a first line and a kernel file);
  strict id/op/n validation (path traversal, `..`, unicode digits, leading zeros); a symlinked `probe.json`, `.verdict` entry or
  `kernel.py` is refused; a FIFO is refused without blocking (hard timeout); size caps; a malformed inventory or probe record
  never raises; a dead pid shows `died`; `kernel/env` runs no verification example and no subprocess other than the two read-only
  ones (assert on a fake runner); cache honoured; the page's CSP and the existing request checks (Host header, Origin) apply to
  the new GET routes unchanged.

### 13c. Page (`orchard/web/index.html`, `app.js`, `app.css`)
- A run's Details tab gets an "Ops and kernels" section from `/api/runs/<name>/inventory`: counts by status as small labelled
  bars (text first, bar second), the unmapped list, the dump line, and the caption that `mapped` is a name match. States for
  `skipped`, unreadable and no inventory.
- A new left-rail section "Kernels": the kernel environment row (status and the command to run) and the probes list (op, state
  badge `running`/`finished`/`blocked`/`crashed`/`interrupted`/`died`, `n of m attempts`, passes, newest first). Selecting a
  probe opens a pane with the header (provisional bars and the `hw: not run` line shown plainly, the calibration note for
  `add`, the simulator float32 note), the attempt table (the columns in 13b; `escalate`/`error` status and `untrusted`
  flags visibly marked), and a kernel viewer (read-only, monospace, `textContent`).
- Same conventions as the rest of the page: keyboard-accessible tabs (`role="tab"`), `aria-live` for the running probe's
  progress, light and dark themes from the existing tokens, the farm look, the polling rules that avoid rebuilding focused
  controls (only on change; 30 s when the tab is hidden), no external hosts, no inline script.
- Do NOT add any control that starts or stops a probe, an inventory or hardware.
- Checks: `node --check` if node exists on the box (it did not on 2026-10-12: say so in the report); a headless-Chrome browser
  check as the 2026-10-08 UI entry did (the official `playwright` package in a scratch venv with `executable_path` set to
  `/usr/bin/google-chrome`, no browser download, run against a temporary server with fixture probe and run directories):
  load the page at 1440, 1024, 768 and 400 px, assert no horizontal page scroll, take screenshots of the Kernels section and the
  Ops section in light and dark, and save them in the scratch directory (not in the repository); report what the screenshots show
  and anything that looked wrong.

### 13d. Docs and version
README "Web page" section: the new panels and that they are read-only; `CLAUDE.md` log entry (what was added, the decision that
the page launches nothing for kernel work and why, what the browser check found); version 0.6.1.

**Review focus for this task:** (1) a hostile probe directory (symlinks, FIFOs, huge files, a planted `probe.json` with
script text in every field) must not break the page or leak a path; (2) nothing in the new routes can write or start a
process; (3) the page works with no probes, no inventory, no kernel environment; (4) every number shown is labelled with what it
is (simulator, provisional, name match).
