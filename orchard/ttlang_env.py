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
import stat
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
    # A fetch of one commit, because main moves and the pin must not.
    fetch = [["git", "-C", src, "fetch", "--quiet", "--depth", "1", "origin", pin],
             ["git", "-C", src, "checkout", "--quiet", "FETCH_HEAD"]]
    start = [["git", "-C", src, "init", "--quiet"],
             ["git", "-C", src, "remote", "add", "origin", defaults.TTLANG_URL]]
    if (env.src_dir / ".git").exists():
        # HEAD could not be read. An earlier run of this command may have stopped after `git init` (for example
        # with the network down). That case has our origin and no commit, and it is finished by fetching again.
        # Anything else is left alone: nothing is deleted or reset.
        rc, url, _ = env.run(["git", "-C", src, "remote", "get-url", "origin"], 30)
        if rc != 0 or url.strip() != defaults.TTLANG_URL:
            return Result("clone", FAIL, f"HEAD could not be read in {env.src_dir} (git is missing or the repository "
                                         "is not one this command started); it is left alone")
        if check_only:
            return Result("clone", TODO, f"would finish the interrupted clone in {env.src_dir}")
        steps = fetch
    elif env.src_dir.exists() and any(env.src_dir.iterdir()):
        return Result("clone", FAIL, f"{env.src_dir} exists and is not a git checkout; it is left alone")
    elif check_only:
        return Result("clone", TODO, f"would fetch tt-lang {pin[:9]} into {env.src_dir}")
    else:
        env.src_dir.mkdir(parents=True, exist_ok=True)
        steps = start + fetch
    for argv in steps:
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
    source = env.src_dir / "python" / "ttl" / "diagnostics.py"
    if check_only and not (env.py.exists() and source.is_file()):
        # The venv and the checkout are made by earlier steps, which --check reports as to do.
        return Result("diagnostics", TODO, "waits for the checkout and the venv")
    target = diagnostics_target(env)
    if target is None:
        return Result("diagnostics", FAIL, f"{env.py} did not say where its site-packages is")
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


# ---- the coder's reference files --------------------------------------------------------------------------------
#
# The kernel probe gives the coder these files in its attempt directory, under reference/ (spec section 11, item 28).
# Two come from the installed tt-lang skill (vendored by tt-warp from tenstorrent/tt-metal, branch
# agentic-research/fast-models-fast); three come from the tt-lang checkout this module builds the simulator from. The
# skill's guide and examples document tt-lang specification 0.12; the checkout's specification and examples are the
# version the simulator runs (0.20 at the pin), so they go in too and the probe's skill says which wins.

TTLANG_SKILL_ENV = "ORCHARD_TTLANG_SKILL"
SKILL_SOURCE, CHECKOUT_SOURCE = "tt-lang skill", "tt-lang checkout"
REFERENCE_MAX_BYTES = 1024 * 1024       # choice: the largest file is the checkout's specification, 92 KB
# (name under reference/, source, path under that source's directory)
REFERENCES = (
    ("guide.md", SKILL_SOURCE, "guide.md"),
    ("examples.md", SKILL_SOURCE, "examples.md"),
    ("TTLangSpecification.md", CHECKOUT_SOURCE, "docs/sphinx/specs/TTLangSpecification.md"),
    ("eltwise_add.py", CHECKOUT_SOURCE, "examples/eltwise_add.py"),
    ("eltwise_broadcast_reduce.py", CHECKOUT_SOURCE, "examples/eltwise_broadcast_reduce.py"),
)


def skill_dir_from(args, environ: dict | None = None, home: Path | None = None) -> Path:
    """The installed tt-lang skill: --tt-lang-skill-dir, else $ORCHARD_TTLANG_SKILL, else ~/.claude/skills/ttm-tt-lang."""
    environ = os.environ if environ is None else environ
    flag = getattr(args, "tt_lang_skill_dir", None)
    if flag:
        return Path(flag)
    if environ.get(TTLANG_SKILL_ENV):
        return Path(environ[TTLANG_SKILL_ENV])
    return Path(home or Path.home()) / ".claude" / "skills" / "ttm-tt-lang"


def _read_reference(path: Path, root: Path | None = None) -> tuple[bytes | None, str | None]:
    """(bytes, None) for a regular file of at most REFERENCE_MAX_BYTES, else (None, why).

    Every directory between `root` and the file must be a real directory: O_NOFOLLOW covers only the last component,
    so a linked `examples/` would otherwise be followed. (`root` itself may be a link.) The file's type is checked with
    lstat before the open, so a device node or a FIFO is never opened; the open does not follow a link and does not
    block, and the type is checked again on the open file."""
    if root is not None:
        cur = Path(root)
        for part in Path(path).relative_to(root).parts[:-1]:
            cur = cur / part
            if os.path.islink(cur):
                return None, f"{cur} is a link; the reference files must not be reached through a linked directory"
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, f"{path} does not exist"
    except OSError as exc:
        return None, f"{path} cannot be read: {exc.strerror or exc}"
    if stat.S_ISLNK(st.st_mode):
        return None, f"{path} is a link; put the file itself there"
    if not stat.S_ISREG(st.st_mode):
        return None, f"{path} is not a regular file"
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        return None, f"{path} cannot be opened: {exc.strerror or exc}"
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            return None, f"{path} is not a regular file"
        data = f.read(REFERENCE_MAX_BYTES + 1)
    if len(data) > REFERENCE_MAX_BYTES:
        return None, f"{path} is larger than {REFERENCE_MAX_BYTES // 1024} KiB"
    return data, None


def read_references(env: KEnv, skill_dir: Path) -> tuple[list[dict], list[str]]:
    """Read every reference file once. (files, problems): each file is {"name", "source", "path", "sha256", "data"};
    each problem is a sentence naming a file that is missing, a link, not a regular file or too large. The probe
    refuses to start while any problem remains, and writes these same bytes into every attempt."""
    roots = {SKILL_SOURCE: Path(skill_dir), CHECKOUT_SOURCE: Path(env.src_dir)}
    files, problems = [], []
    for name, source, rel in REFERENCES:
        path = roots[source] / rel
        data, why = _read_reference(path, roots[source])
        if why:
            problems.append(why)
            continue
        files.append({"name": name, "source": source, "path": str(path),
                      "sha256": hashlib.sha256(data).hexdigest(), "data": data})
    return files, problems


def skills_status(env: KEnv, skill_dir: Path, *, write_cache: bool = True, preread: dict | None = None) -> list:
    """The tt-lang checkout and the installed tt-lang skill compared with upstream (orchard/skillsfresh.py): a list of
    Freshness results. Read only; `write_cache` False keeps the result cache in memory. `preread` maps skill files
    (relative paths) to the bytes read_references already read, so those are hashed and not read again."""
    from orchard import skillsfresh
    return skillsfresh.check_kernel_sources(env.src_dir, skill_dir, cache=skillsfresh.Cache(write=write_cache),
                                            preread=preread)


def skill_bytes(references: list[dict]) -> dict[str, bytes]:
    """{path under the skill directory: bytes} of the reference files that came from the tt-lang skill."""
    roots = {name: rel for name, source, rel in REFERENCES if source == SKILL_SOURCE}
    return {roots[r["name"]]: r["data"] for r in references if r["source"] == SKILL_SOURCE and r["name"] in roots}


def skills_lines(results) -> list[str]:
    """One line per source for the terminal: ok, or fail with what to run to update."""
    lines = []
    for r in results:
        status = FAIL if r.stale else OK
        text = f"{status:5s} skills: {r.line()}"
        if r.stale:
            text += " " + r.update_text()
        lines.append(text)
    return lines


def main(args, *, env: KEnv | None = None, out=None, run=None) -> int:
    """`tt-orchard kernel-env [--check] [--src-dir D] [--venv-dir D] [--tt-lang-skill-dir D] [--allow-stale-skills]`.
    Exit 0 when every step is ok, 2 when a step failed or (with --check) is still to do.

    With --check it also checks what the kernel probe gives the coder: the reference files must be readable, and the
    tt-lang checkout and the installed tt-lang skill must be the latest upstream has (skills_status, with the result
    cache kept in memory, so --check still changes nothing). A stale source fails the check unless
    --allow-stale-skills is given. Without --check nothing is compared with upstream: the build never updates a
    source to the latest, and the probe makes its own check before it starts."""
    say = (lambda line: print(line, file=out)) if out is not None else print
    home = Path.home()
    env = env or KEnv(
        src_dir=Path(args.src_dir or os.environ.get("ORCHARD_TTLANG_SRC") or home / "orchard-src" / "tt-lang"),
        venv_dir=Path(args.venv_dir or os.environ.get("ORCHARD_TTLANG_VENV") or home / "orchard-venvs" / "ttlang"),
        scratch_dir=Path(os.environ.get("ORCHARD_SCRATCH") or home / "orchard-src" / "scratch"),
        run=run or real_run, environ=dict(os.environ))
    results = ensure(env, check_only=args.check, say=say)
    ok = all(r.status == OK for r in results)
    if args.check:
        skill_dir = skill_dir_from(args)
        refs, problems = read_references(env, skill_dir)
        for p in problems:
            say(f"{FAIL:5s} references: {p}")
        fresh = skills_status(env, skill_dir, write_cache=False, preread=skill_bytes(refs))
        for line in skills_lines(fresh):
            say(line)
        stale = [r for r in fresh if r.stale]
        if stale and getattr(args, "allow_stale_skills", False):
            say(f"{OK:5s} skills: stale skills allowed (--allow-stale-skills): "
                + ", ".join(r.source for r in stale))
            stale = []
        ok = ok and not problems and not stale
    return 0 if ok else 2
