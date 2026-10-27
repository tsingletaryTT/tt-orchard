# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard audio-reference-env`: the interpreter and source checkout that run the Stable Audio 3 CPU reference
(orchard/skills/audio-reference-templates/make_reference.py; design docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md, step 0).

It makes two things outside the repository:

    <src-dir>    a checkout of Stability AI's `stable-audio-3` library (MIT licence) at one pinned commit, fetched by hash. A checkout that
                 already exists is only read: one at another commit is refused, never reset.
    <venv-dir>   Python 3.12 with CPU-only torch 2.7.1 (installed alone from the PyTorch CPU index), the library's other dependencies from PyPI,
                 and the checkout installed editable without its dependencies. `orchard-audio-env.json` records the commit and every version.

It never touches the shared Tenstorrent virtual environment (a directory that is, or is inside, that venv or the tt-orchard checkout is refused,
links resolved) and passes no Hugging Face token to any command. Importing the library does not open a device: it needs no Tenstorrent software.
`--check` and `--dry-run` change nothing.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orchard.demo_env import FAIL, OK, TODO, Result, real_run     # the result type and status words are shared

LIBRARY_URL = "https://github.com/Stability-AI/stable-audio-3.git"
# The commit read on 2026-10-28 (main at that time). A checkout is trusted only at exactly this commit.
LIBRARY_COMMIT = "77e0398c26f7d22d801ec1bb036307e8b231f741"
PYTHON_VERSION = "3.12"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
TORCH_PINS = ("torch==2.7.1", "torchaudio==2.7.1")
TORCH = " ".join(TORCH_PINS)
# The library's own dependency floors (its pyproject.toml), without torch. The versions that resolve are recorded, not pinned further.
REQUIREMENTS = ("einops>=0.8.2", "einops-exts>=0.0.4", "numpy>=2.2.6", "packaging>=26.0", "safetensors>=0.7.0", "tqdm>=4.67.3",
                "huggingface_hub>=1.7.1", "transformers>=5.8.0", "soundfile>=0.13.1")
RECORD = "orchard-audio-env.json"
RECORDED = ("torch", "torchaudio", "numpy", "transformers", "huggingface_hub", "safetensors", "einops", "soundfile")
# Variables that authenticate or redirect git, uv and pip: tokens, GIT_DIR and its relatives (they point git at another repository), UV_PYTHON and
# UV_PROJECT_ENVIRONMENT (they point uv at another interpreter or environment), extra indexes, and Python's own variables.
TOKEN_PREFIXES = ("HF_", "HUGGING", "UV_", "PIP_", "GIT_", "GITHUB_TOKEN", "GH_TOKEN", "PYTHON")


@dataclass
class AEnv:
    venv_dir: Path
    src_dir: Path
    shared_venv: Path
    repo_dir: Path
    run: Callable = real_run
    environ: dict | None = None

    def __post_init__(self):
        self.venv_dir = Path(self.venv_dir).expanduser().resolve()
        self.src_dir = Path(self.src_dir).expanduser().resolve()

    @property
    def py(self) -> Path:
        return self.venv_dir / "bin" / "python"


def env_from_args(args, *, environ=None, home: Path | None = None, repo_dir: Path | None = None, run=None) -> AEnv:
    environ = dict(os.environ if environ is None else environ)
    home = Path(home) if home else Path.home()
    return AEnv(
        venv_dir=Path(getattr(args, "venv_dir", None) or environ.get("ORCHARD_AUDIO_VENV") or home / "orchard-venvs" / "stable-audio"),
        src_dir=Path(getattr(args, "src_dir", None) or environ.get("ORCHARD_AUDIO_SRC") or home / "orchard-src" / "stable-audio-3"),
        shared_venv=home / ".tenstorrent-venv", repo_dir=Path(repo_dir) if repo_dir else Path(__file__).resolve().parent.parent,
        run=run or real_run, environ=environ)


def requirements_sha256() -> str:
    return hashlib.sha256("\n".join([LIBRARY_COMMIT, TORCH_CPU_INDEX, *TORCH_PINS, PYPI_INDEX, *REQUIREMENTS]).encode()).hexdigest()


def _clean(env: AEnv) -> dict:
    """The environment for git and uv: nothing from the operator's shell that authenticates them or points them at another repository,
    interpreter, environment or package index (TOKEN_PREFIXES)."""
    source = env.environ if env.environ is not None else os.environ
    out = {k: v for k, v in source.items() if not k.startswith(TOKEN_PREFIXES)}
    out["GIT_TERMINAL_PROMPT"] = "0"
    return out


def _inside(path: Path, parent: Path) -> bool:
    p, q = path.resolve(), parent.resolve()
    return p == q or q in p.parents


def _fail(name: str, argv, err: str) -> Result:
    return Result(name, FAIL, f"`{' '.join(str(a) for a in argv[:6])}` failed: {err.strip()[-300:]}")


def refusal(env: AEnv) -> Result | None:
    for label, other in (("the shared TT venv", env.shared_venv), ("the tt-orchard checkout", env.repo_dir)):
        if _inside(env.venv_dir, other) or _inside(env.src_dir, other):
            return Result("venv-dir", FAIL, f"{env.venv_dir} or {env.src_dir} is {label} or inside it. This command never touches {label}; "
                                            "pass another --venv-dir or --src-dir")
    if _inside(env.src_dir, env.venv_dir) or _inside(env.venv_dir, env.src_dir):
        return Result("venv-dir", FAIL, f"the venv directory {env.venv_dir} and the source directory {env.src_dir} must not contain each other")
    return None


def lock_path(env: AEnv) -> Path:
    return env.venv_dir.parent / f"{env.venv_dir.name}.lock"


def step_source(env: AEnv, check_only: bool) -> Result:
    git = ["git", "-C", str(env.src_dir)]
    dot_git = env.src_dir / ".git"
    if dot_git.is_symlink() or (dot_git.exists() and not dot_git.is_dir()):
        # A `.git` file or link names another git directory (`gitdir: ...`), so the commit and the status would describe some other place.
        return Result("source", FAIL, f"{dot_git} is a link or a file, not a directory; this command trusts only a plain checkout. "
                                      "Pass another --src-dir")
    if dot_git.exists():
        rc, out, err = env.run([*git, "rev-parse", "HEAD"], 60, _clean(env))
        head = out.strip()
        if rc != 0:
            return _fail("source", [*git, "rev-parse", "HEAD"], err)
        if head != LIBRARY_COMMIT:
            return Result("source", FAIL, f"{env.src_dir} is at {head[:12]}, not the pinned {LIBRARY_COMMIT[:12]}. It is left alone; "
                                          "pass another --src-dir or move it away")
        # The commit alone does not say the files are the pinned ones: a tracked edit or an untracked file leaves HEAD where it is, and the
        # checkout is installed editable, so changed code would run under the pin's name.
        status = [*git, "status", "--porcelain", "--untracked-files=all"]
        rc, out, err = env.run(status, 120, _clean(env))
        if rc != 0:
            return _fail("source", status, err)
        if out.strip():
            changed = [line for line in out.splitlines() if line.strip()]
            return Result("source", FAIL, f"{env.src_dir} is at the pinned commit but has local changes ({len(changed)} paths, for example "
                                          f"{changed[0].strip()[:80]!r}). It is left alone; pass another --src-dir or clean it")
        return Result("source", OK, f"{env.src_dir} at {LIBRARY_COMMIT[:12]}, clean")
    if env.src_dir.exists() and any(env.src_dir.iterdir()):
        return Result("source", FAIL, f"{env.src_dir} holds files and is not a git checkout; it is left alone. Pass another --src-dir")
    if check_only:
        return Result("source", TODO, f"would fetch {LIBRARY_URL} at {LIBRARY_COMMIT[:12]} into {env.src_dir}")
    for argv in (["git", "init", "--quiet", str(env.src_dir)], [*git, "remote", "add", "origin", LIBRARY_URL],
                 [*git, "fetch", "--quiet", "--depth", "1", "origin", LIBRARY_COMMIT], [*git, "checkout", "--quiet", LIBRARY_COMMIT]):
        rc, _, err = env.run(argv, 900, _clean(env))
        if rc != 0:
            return _fail("source", argv, err)
    return Result("source", OK, f"fetched {LIBRARY_COMMIT[:12]} into {env.src_dir}")


def step_venv(env: AEnv, check_only: bool) -> Result:
    if env.py.exists():
        return Result("venv", OK, str(env.venv_dir))
    if env.venv_dir.exists() and any(env.venv_dir.iterdir()):
        return Result("venv", FAIL, f"{env.venv_dir} holds files and has no bin/python; it is left alone. Pass another --venv-dir")
    argv = ["uv", "venv", "--quiet", "--python", PYTHON_VERSION, str(env.venv_dir)]
    if check_only:
        return Result("venv", TODO, f"would run: {' '.join(argv)}")
    rc, _, err = env.run(argv, 300, _clean(env))
    return Result("venv", OK, f"created {env.venv_dir}") if rc == 0 else _fail("venv", argv, err)


def _record_matches(env: AEnv) -> bool:
    rec = read_record(env)
    return bool(env.py.exists() and rec and rec.get("requirements_sha256") == requirements_sha256() and rec.get("library_commit") == LIBRARY_COMMIT)


def _install(env: AEnv, name: str, argv: list[str], check_only: bool, done: str) -> Result:
    """A build runs the install every time (uv does nothing for a package that is already right). `--check` and `--dry-run` change nothing: they
    take the record's word for the install and the rules judge the result."""
    if check_only:
        if _record_matches(env):
            return Result(name, OK, f"{done} (record {requirements_sha256()[:12]})")
        return Result(name, TODO, f"would run: {' '.join(argv[:8])} ...")
    rc, _, err = env.run(argv, 1800, _clean(env))
    return Result(name, OK, done) if rc == 0 else _fail(name, argv, err)


def read_record(env: AEnv) -> dict | None:
    try:
        rec = json.loads((env.venv_dir / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _ask(env: AEnv, code: str, timeout: int = 300):
    return env.run([str(env.py), "-I", "-c", code], timeout, _clean(env))


def rules(env: AEnv) -> list[Result]:
    out = []
    rc, text, err = _ask(env, "import stable_audio_3, stable_audio_3.model; print(stable_audio_3.__file__)")
    where = text.strip().splitlines()[-1].strip() if rc == 0 and text.strip() else ""
    if rc != 0:
        out.append(Result("import", FAIL, f"import stable_audio_3 failed: {err.strip()[-300:]}"))
    elif not where or not os.path.isabs(where):
        out.append(Result("import", FAIL, f"stable_audio_3 did not say where it was imported from (got {where!r})"))
    elif not _inside(Path(where), env.src_dir):
        # The install is editable, so it follows whichever checkout it was built from. Another checkout that is clean and at the pin does not
        # vouch for the code that actually runs.
        out.append(Result("import", FAIL, f"stable_audio_3 is imported from {Path(where).resolve().parent.parent}, not from the checked source "
                                          f"tree {env.src_dir}; run `tt-orchard audio-reference-env --src-dir {env.src_dir}` to rebuild the install"))
    else:
        out.append(Result("import", OK, f"stable_audio_3 imports from {env.src_dir}"))
    rc, text, err = _ask(env, "import torch; print(torch.__version__, torch.cuda.is_available())")
    parts = text.split()
    cpu = rc == 0 and len(parts) == 2 and parts[0] == "2.7.1+cpu" and parts[1] == "False"
    out.append(Result("cpu-only", OK if cpu else FAIL,
                      "torch 2.7.1+cpu, no CUDA" if cpu else f"torch is not the CPU-only 2.7.1 build (got {text.strip() or err.strip()[-200:]!r})"))
    return out


def installed_versions(env: AEnv) -> dict | None:
    code = ("import importlib.metadata as m, json; out = {}\n"
            f"for n in {list(RECORDED)!r}:\n    try:\n        out[n] = m.version(n)\n    except m.PackageNotFoundError:\n        out[n] = None\n"
            "print(json.dumps(out))")
    rc, text, _ = _ask(env, code)
    try:
        data = json.loads(text) if rc == 0 else None
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def step_record(env: AEnv, check_only: bool, upstream_ok: bool) -> Result:
    good = _record_matches(env)
    if check_only:
        return Result("record", OK if good else TODO, f"{env.venv_dir / RECORD} {'matches the pins' if good else 'is missing or stale'}")
    if not upstream_ok:
        return Result("record", FAIL, "not written: an earlier step failed")
    versions = installed_versions(env)
    if versions is None:
        return Result("record", FAIL, "could not read the installed versions")
    (env.venv_dir / RECORD).write_text(json.dumps({"library_commit": LIBRARY_COMMIT, "library_url": LIBRARY_URL, "torch_pin": TORCH,
                                                   "requirements": list(REQUIREMENTS), "requirements_sha256": requirements_sha256(),
                                                   "installed": versions}, indent=1) + "\n", encoding="utf-8")
    return Result("record", OK, f"wrote {env.venv_dir / RECORD}")


def src_lock_path(env: AEnv) -> Path:
    return env.src_dir.parent / f"{env.src_dir.name}.lock"


def lock_paths(env: AEnv) -> list[Path]:
    """The lock files a build needs, each once and always in the same (sorted) order, so two builds that share only one directory cannot hold
    one lock each and wait for the other."""
    return sorted({lock_path(env), src_lock_path(env)})


def ensure(env: AEnv, *, check_only: bool, say=lambda s: None) -> list[Result]:
    """The steps in order, a failure ending the build, then the rules. A refusal ends it before anything runs. A build takes an exclusive lock on
    the venv directory and on the source directory (lock files beside each, taken in sorted order), so two builds that differ in only one of them
    cannot clone, fetch, create the venv or install at once; `--check` and `--dry-run` take none."""
    bad = refusal(env)
    if bad:
        say(f"{bad.status:5s} {bad.name}: {bad.detail}")
        return [bad]
    if check_only:
        return _ensure(env, check_only, say)
    with contextlib.ExitStack() as stack:
        for path in lock_paths(env):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context(open(path, "w"))
            except OSError as exc:
                r = Result("lock", FAIL, f"cannot open {path}: {exc}")
                say(f"{r.status:5s} {r.name}: {r.detail}")
                return [r]
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                r = Result("lock", FAIL, f"another audio-reference-env is already running ({path} is locked)")
                say(f"{r.status:5s} {r.name}: {r.detail}")
                return [r]
        return _ensure(env, check_only, say)


def _ensure(env: AEnv, check_only: bool, say) -> list[Result]:
    results: list[Result] = []

    def add(r: Result) -> Result:
        results.append(r)
        say(f"{r.status:5s} {r.name}: {r.detail}")
        return r

    if add(step_source(env, check_only)).status == FAIL or add(step_venv(env, check_only)).status == FAIL:
        return results
    py = str(env.py)
    torch = add(_install(env, "torch", ["uv", "pip", "install", "--quiet", "--python", py, "--index-url", TORCH_CPU_INDEX, *TORCH_PINS],
                         check_only, f"{TORCH} from the CPU index"))
    pk = add(_install(env, "packages", ["uv", "pip", "install", "--quiet", "--python", py, "--index-url", PYPI_INDEX, *REQUIREMENTS],
                      check_only, f"{len(REQUIREMENTS)} packages from PyPI")) if torch.status != FAIL else None
    lib = add(_install(env, "library", ["uv", "pip", "install", "--quiet", "--python", py, "--no-deps", "-e", str(env.src_dir)],
                       check_only, f"{env.src_dir} installed editable")) if pk and pk.status != FAIL else None
    ok = bool(lib and lib.status == OK and not check_only)
    if env.py.exists():
        rule_results = [add(r) for r in rules(env)]
        ok = ok and all(r.status == OK for r in rule_results)
    add(step_record(env, check_only, upstream_ok=ok))
    return results


def main(args, *, env: AEnv | None = None, out=None) -> int:
    """`tt-orchard audio-reference-env [--check] [--dry-run] [--venv-dir D] [--src-dir D]`. Exit 0 when every step is ok (a dry run: when
    nothing failed), 2 otherwise."""
    say = (lambda line: print(line, file=out)) if out is not None else print
    env = env or env_from_args(args)
    check_only = bool(getattr(args, "check", False) or getattr(args, "dry_run", False))
    results = ensure(env, check_only=check_only, say=say)
    if getattr(args, "dry_run", False) and not getattr(args, "check", False):
        return 2 if any(r.status == FAIL for r in results) else 0
    return 0 if all(r.status == OK for r in results) else 2
