# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard demo-env`: build and check the interpreter that runs tt-metal demos and tt-train (issue 92).

Why it exists. The shared TT virtual environment (~/.tenstorrent-venv) holds numpy 2.5.3 from an upgrade nobody recorded,
while ttnn 0.80.0 declares `numpy<2`. With numpy 2, `cv2` fails to import inside tt-metal's root conftest.py, and
`import datasets` fails because `HfFolder` is gone from huggingface_hub 1.x, which also breaks tt-train's lora_llama
example. Two agents worked around it with a numpy 1.26.4 copy on PYTHONPATH, which lasts for one run. The stock-runtime
stage (orchard/stock_runtime.py) runs tt-transformers' own demo from the host tt-metal tree and needs an interpreter that
imports everything the demo imports.

What it makes, outside the repository:

    <venv-dir>   Python 3.12, CPU torch, numpy below 2, a huggingface_hub that `datasets` accepts, transformers,
                 opencv-python-headless and the packages ttnn declares, every one pinned (REQUIREMENTS below), and the host
                 tree's `ttnn` and `ttml` on sys.path. `orchard-demo-env.json` records the host tree's tag and sha, the
                 pins and a hash of them.

How the host tree gets in. The shared venv got `ttnn` and `ttml` from the host tree as editable installs plus two hand-made
.pth files. `pip install -e` on tt-metal runs its setup.py, which can start a CMake build and writes build and egg-info
directories into the tree. This command must not change the host tree, so it writes the same thing the shared venv ended up
with: a .pth file listing the tree's source and build directories, and a small dist-info for each of `ttnn` and `ttml` at the
tree's release tag, so `importlib.metadata` (and pip) see them as installed. Nothing is built.

Safety: it never touches the shared venv (a venv directory that is, or is inside, the shared venv is refused, links
resolved), never writes into the repository checkout or the host tree, and `--check` and `--dry-run` change nothing.
`--check` does not import ttnn or ttml: importing them opens a device. It reads distribution metadata, and imports only
cv2, datasets and torch with the venv's own python in isolated mode (`-I`), none of which opens a device.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

OK, TODO, FAIL = "ok", "todo", "fail"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
PYTHON_VERSION = "3.12"
RECORD = "orchard-demo-env.json"
PTH_NAME = "tt-host-tree.pth"
INSTALLER = "orchard-demo-env"

# CPU torch comes first and alone: the CPU index carries old copies of certifi and requests (README, "setup").
TORCH = "torch==2.7.0"
# The root conftest.py of the host tree imports torchvision (found by the first real stock-runtime run, 2026-10-29). It comes from the same CPU index as torch.
TORCHVISION = "torchvision==0.22.0"
# Resolved together by uv on 2026-10-26 (Python 3.12, linux x86_64) and chosen because they import together:
# numpy 1.26.4 (ttnn declares numpy<2, and cv2 and torch 2.7 accept it), huggingface_hub 1.33.0 with datasets 4.8.5
# (datasets 2.x imports HfFolder, which hub 1.x dropped) and transformers 5.19.0 (the version stage 1 was run with).
# The ttnn packages come from the host tree's pyproject.toml dependency list; pytest, requests, loguru, pillow, pydantic,
# termcolor, tqdm and pyyaml are what tt-transformers' demo and the root conftest.py import.
REQUIREMENTS = (
    "numpy==1.26.4",
    "huggingface_hub==1.33.0",
    "datasets==4.8.5",
    # datasets needs pyarrow 21 or newer. pyarrow 26.0.0, the newest, refuses to import with numpy 1 ("pyarrow requires NumPy 2.0 or
    # newer"), found by the first real `--check` on 2026-10-26. 21.0.0 imports with numpy 1.26.4 (checked with `import datasets`).
    "pyarrow==21.0.0",
    "transformers==5.19.0",
    # tt-transformers' generate_reference_hf.py loads the model with device_map="auto", which transformers refuses without accelerate
    # (found by the first real stock-runtime prepare on 2026-10-29, Llama-3.1-8B-Instruct).
    "accelerate==1.15.0",
    # benchmarking_utils in the host tree imports pytz (found by the same run).
    "pytz==2026.6",
    "tokenizers==0.23.3",
    "safetensors==0.8.0",
    "pytest==9.1.1",
    "requests==2.34.2",
    "loguru==0.7.3",
    "networkx==3.7",
    "graphviz==0.21",
    "pyyaml==6.0.3",
    "click==8.5.0",
    "pandas==3.0.6",
    "seaborn==0.13.2",
    "ml_dtypes==0.5.4",
    "pillow==13.0.0",
    "termcolor==3.3.0",
    "tqdm==4.70.1",
    "pydantic==2.14.1",
    "opencv-python-headless==4.11.0.86",
)

# What the venv's python is asked, always as `python -I -c <code>`. None imports ttnn or ttml.
PURELIB_CODE = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
CV2_CODE = "import cv2"
DATASETS_CODE = "import datasets"
TORCH_CODE = "import torch, torchvision, transformers, tokenizers, safetensors"
VERSIONS_CODE = ("import importlib.metadata as m, json; "
                 "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))")
TAG = re.compile(r"v\d+(\.\d+){1,3}([-.]?(rc|dev|post)\d+)?")


def dist_code(name: str) -> str:
    """Ask for a distribution's version from its metadata. This never imports the package."""
    return f"import importlib.metadata as m; print(m.version('{name}'))"


def requirements_sha256() -> str:
    text = "\n".join([TORCH_CPU_INDEX, TORCH, TORCHVISION, PYPI_INDEX, *REQUIREMENTS])
    return hashlib.sha256(text.encode()).hexdigest()


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
class DEnv:
    venv_dir: Path
    tt_metal_dir: Path
    shared_venv: Path
    repo_dir: Path
    run: Callable = real_run          # (argv, timeout, env=None) -> (returncode, stdout, stderr)
    environ: dict | None = None

    def __post_init__(self):
        # Absolute and link-free before anything uses them. A relative tree would go into tt-host-tree.pth unchanged, and Python
        # resolves a relative .pth line against site-packages, so the check could pass while the import fails.
        self.venv_dir = Path(self.venv_dir).expanduser().resolve()
        self.tt_metal_dir = Path(self.tt_metal_dir).expanduser().resolve()

    @property
    def py(self) -> Path:
        return self.venv_dir / "bin" / "python"


def env_from_args(args, *, environ=None, home: Path | None = None, repo_dir: Path | None = None, run=None) -> DEnv:
    environ = dict(os.environ if environ is None else environ)
    home = Path(home) if home else Path.home()
    return DEnv(
        venv_dir=Path(getattr(args, "venv_dir", None) or environ.get("ORCHARD_DEMO_VENV") or home / "orchard-venvs" / "tt-demo"),
        tt_metal_dir=Path(getattr(args, "tt_metal_dir", None) or environ.get("ORCHARD_TT_METAL_DIR") or home / "tt-metal"),
        shared_venv=home / ".tenstorrent-venv",
        repo_dir=Path(repo_dir) if repo_dir else Path(__file__).resolve().parent.parent,
        run=run or real_run, environ=environ)


def fix_command(env: DEnv) -> str:
    """The command that builds (or repairs) this environment, spelled out for an operator."""
    return f"tt-orchard demo-env --venv-dir {env.venv_dir} --tt-metal-dir {env.tt_metal_dir}"


def _fail(name: str, argv, err: str) -> Result:
    return Result(name, FAIL, f"`{' '.join(str(a) for a in argv[:6])}` failed: {err.strip()[-300:]}")


def _inside(path: Path, parent: Path) -> bool:
    p, q = path.resolve(), parent.resolve()
    return p == q or q in p.parents


def refusal(env: DEnv) -> Result | None:
    """The directory must not be the shared venv, the repository checkout, or hold the host tree."""
    venv = env.venv_dir
    for label, other in (("the shared TT venv", env.shared_venv), ("the tt-orchard checkout", env.repo_dir)):
        if _inside(venv, other):
            return Result("venv-dir", FAIL, f"{venv} is {label} or inside it. This command never touches {label}; "
                                            "pass another --venv-dir")
    if _inside(env.tt_metal_dir, venv):
        return Result("venv-dir", FAIL, f"the host tt-metal tree {env.tt_metal_dir} is inside the venv directory {venv}")
    if _inside(venv, env.tt_metal_dir):
        return Result("venv-dir", FAIL, f"the venv directory {venv} is inside the host tt-metal tree {env.tt_metal_dir}; "
                                        "this command never writes into the host tree. Pass another --venv-dir")
    return None


# ---- the host tree ----------------------------------------------------------------------------------------------

def tree_facts(env: DEnv) -> tuple[dict | None, str | None]:
    """({"path", "tag", "sha"}, None), or (None, why the tree cannot be used). Read only."""
    tree = env.tt_metal_dir
    if not (tree / "ttnn" / "ttnn" / "_ttnn.so").is_file():
        return None, f"{tree} has no ttnn/ttnn/_ttnn.so, so it is not a built tt-metal tree"
    rc, out, err = env.run(["git", "-C", str(tree), "describe", "--tags", "--exact-match"], 60)
    tag = out.strip()
    if rc != 0 or not tag:
        return None, (f"{tree} is not at a release tag ({err.strip()[-120:] or 'git describe gave no tag'}); "
                      "check out the tag the demos were tested on")
    if not TAG.fullmatch(tag):
        return None, f"{tree} is at {tag!r}, which is not a release tag such as v0.80.0"
    rc, sha, _ = env.run(["git", "-C", str(tree), "rev-parse", "HEAD"], 60)
    return {"path": str(tree), "tag": tag, "sha": sha.strip() if rc == 0 else ""}, None


def tree_version(facts: dict) -> str:
    return facts["tag"][1:]


def pth_lines(env: DEnv) -> list[str]:
    """The directories the shared venv's two hand-made .pth files list, when the tree has them."""
    t = env.tt_metal_dir
    want = [t, t / "ttnn", t / "tools", t / "tt-train" / "sources" / "ttml", t / "build" / "tt-train" / "sources" / "ttml"]
    return [str(p) for p in want if p.is_dir()]


def purelib(env: DEnv) -> Path | None:
    if not env.py.exists():
        return None
    rc, out, _ = env.run([str(env.py), "-I", "-c", PURELIB_CODE], 60)
    return Path(out.strip()) if rc == 0 and out.strip() else None


def _dist_info(site: Path, name: str, version: str, tree: Path) -> Path:
    d = site / f"{name}-{version}.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        f"Summary: The host tt-metal tree at v{version}, put on sys.path by {PTH_NAME} (tt-orchard demo-env)\n"
        "Requires-Python: >=3.10\n", encoding="utf-8")
    (d / "INSTALLER").write_text(INSTALLER + "\n", encoding="utf-8")
    (d / "WHEEL").write_text(f"Wheel-Version: 1.0\nGenerator: {INSTALLER}\nRoot-Is-Purelib: true\nTag: py3-none-any\n", encoding="utf-8")
    (d / "top_level.txt").write_text(name + "\n", encoding="utf-8")
    (d / "direct_url.json").write_text(json.dumps({"url": f"file://{tree}", "dir_info": {"editable": True}}) + "\n", encoding="utf-8")
    (d / "RECORD").write_text("".join(f"{d.name}/{f},,\n" for f in ("METADATA", "WHEEL", "INSTALLER", "top_level.txt", "direct_url.json", "RECORD")),
                              encoding="utf-8")
    return d


def _other_version_installed(site: Path, version: str) -> bool:
    """True when this command's complete ttnn and ttml dist-infos exist at ONE other version (the same for both), and none at `version`."""
    versions = []
    for name in ("ttnn", "ttml"):
        if (site / f"{name}-{version}.dist-info").exists():
            return False
        found = {d.name[len(name) + 1:-len(".dist-info")] for d in site.glob(f"{name}-*.dist-info")
                 if all((d / f).is_file() for f in ("INSTALLER", "WHEEL", "METADATA"))
                 and (d / "INSTALLER").read_text(encoding="utf-8").strip() == INSTALLER}
        versions.append(found)
    return len(versions[0]) == 1 and versions[0] == versions[1]


def step_host_tree(env: DEnv, check_only: bool, facts: dict | None, problem: str | None) -> Result:
    if facts is None:
        return Result("host-tree", FAIL, problem or "the host tree cannot be used")
    site = purelib(env)
    if site is None:
        return Result("host-tree", TODO, f"would write {PTH_NAME} and the ttnn and ttml distributions for {facts['tag']} into the venv")
    want, pth = pth_lines(env), site / PTH_NAME
    version = tree_version(facts)
    have = pth.read_text(encoding="utf-8").splitlines() if pth.is_file() else None
    dists_ok = all((site / f"{n}-{version}.dist-info" / f).is_file() for n in ("ttnn", "ttml") for f in ("INSTALLER", "WHEEL", "METADATA"))
    if have == want and dists_ok:
        return Result("host-tree", OK, f"{len(want)} directories of {env.tt_metal_dir} at {facts['tag']}")
    if check_only:
        if have is not None and have != want:
            return Result("host-tree", FAIL, f"{pth} lists {have}; the host tree needs {want}. Run `{fix_command(env)}`")
        if have == want and _other_version_installed(site, version):
            # Everything is in place and only the version differs from the tree's tag. `ttnn-version` and `record-tag` report that
            # drift (the preflight turns them into a warning); it is not a missing distribution.
            return Result("host-tree", OK, f"{len(want)} directories of {env.tt_metal_dir}; the distributions are for another tag than "
                                           f"{facts['tag']}")
        return Result("host-tree", TODO, f"would write {PTH_NAME} and the ttnn and ttml distributions for {facts['tag']}")
    # A ttnn or ttml distribution this command did not write belongs to someone else: it is left alone.
    for name in ("ttnn", "ttml"):
        for old in site.glob(f"{name}-*.dist-info"):
            ours = (old / "INSTALLER").is_file() and (old / "INSTALLER").read_text(encoding="utf-8").strip() == INSTALLER
            if not ours:
                return Result("host-tree", FAIL, f"{old} was not installed by this command; it is left alone. "
                                                 "Remove it or use another --venv-dir")
    for name in ("ttnn", "ttml"):
        for old in site.glob(f"{name}-*.dist-info"):
            shutil.rmtree(old)
        _dist_info(site, name, version, env.tt_metal_dir)
    pth.write_text("\n".join(want) + "\n", encoding="utf-8")
    return Result("host-tree", OK, f"wrote {pth} and the ttnn and ttml distributions for {facts['tag']}")


# ---- the steps ---------------------------------------------------------------------------------------------------

def step_venv(env: DEnv, check_only: bool) -> Result:
    if env.py.exists():
        return Result("venv", OK, str(env.venv_dir))
    if env.venv_dir.exists() and any(env.venv_dir.iterdir()):
        return Result("venv", FAIL, f"{env.venv_dir} holds files and has no bin/python; it is left alone. Pass another --venv-dir")
    argv = ["uv", "venv", "--quiet", "--python", PYTHON_VERSION, str(env.venv_dir)]
    if check_only:
        return Result("venv", TODO, f"would run: {' '.join(argv)}")
    rc, _, err = env.run(argv, 300)
    return Result("venv", OK, f"created {env.venv_dir}") if rc == 0 else _fail("venv", argv, err)


def read_record(env: DEnv) -> dict | None:
    try:
        rec = json.loads((env.venv_dir / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _clean_environ(env: DEnv) -> dict:
    """The environment for uv: no extra package indexes from the operator's shell."""
    return {k: v for k, v in (env.environ if env.environ is not None else os.environ).items()
            if not k.startswith(("UV_INDEX", "UV_EXTRA_INDEX", "PIP_INDEX", "PIP_EXTRA_INDEX"))}


def step_packages(env: DEnv, check_only: bool) -> Result:
    """A build installs the pins every time: uv does nothing for a package that is already right, and puts back one that was upgraded or
    removed after the last build (numpy 2 after a stray `pip install`, for example), which the record's hash of the pins cannot see.
    `--check` and `--dry-run` change nothing: they take the record's word for the install and the rules judge the result."""
    py = str(env.py)
    torch = ["uv", "pip", "install", "--quiet", "--python", py, "--index-url", TORCH_CPU_INDEX, TORCH, TORCHVISION]
    pins = ["uv", "pip", "install", "--quiet", "--python", py, "--index-url", PYPI_INDEX, *REQUIREMENTS]
    if check_only:
        rec = read_record(env)
        if env.py.exists() and rec and rec.get("requirements_sha256") == requirements_sha256():
            return Result("packages", OK, f"{TORCH} and {len(REQUIREMENTS)} pinned packages (record {requirements_sha256()[:12]})")
        return Result("packages", TODO, f"would run: {' '.join(torch)}, then uv pip install of {len(REQUIREMENTS)} pins from {PYPI_INDEX}")
    for argv in (torch, pins):
        rc, _, err = env.run(argv, 1800, _clean_environ(env))
        if rc != 0:
            return _fail("packages", argv, err)
    return Result("packages", OK, f"{TORCH} and {len(REQUIREMENTS)} pinned packages are in place")


def step_record(env: DEnv, check_only: bool, facts: dict | None, upstream_ok: bool) -> Result:
    rec = read_record(env)
    want_tree = facts and {"path": facts["path"], "tag": facts["tag"], "sha": facts["sha"]}
    if rec and rec.get("requirements_sha256") == requirements_sha256() and rec.get("torch") == TORCH and rec.get("host_tree") == want_tree:
        return Result("record", OK, f"{env.venv_dir / RECORD} (tag {facts['tag']}, requirements {requirements_sha256()[:12]})")
    if rec is not None and rec.get("requirements_sha256") != requirements_sha256():
        why = "was written for other requirements"
    elif rec is None:
        why = "is missing"
    else:
        why = "is out of date"
    if check_only or not upstream_ok or facts is None:
        return Result("record", TODO, f"{env.venv_dir / RECORD} {why}. Run `{fix_command(env)}`")
    rc, out, _ = env.run([str(env.py), "-I", "-c", VERSIONS_CODE], 120)
    try:
        installed = json.loads(out) if rc == 0 else {}
    except ValueError:
        installed = {}
    record = {"host_tree": want_tree, "python": PYTHON_VERSION, "torch": TORCH, "requirements": list(REQUIREMENTS),
              "requirements_sha256": requirements_sha256(), "installed": installed}
    (env.venv_dir / RECORD).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Result("record", OK, f"wrote {env.venv_dir / RECORD}")


# ---- the rules ---------------------------------------------------------------------------------------------------

def _ask(env: DEnv, code: str, timeout: int = 180):
    return env.run([str(env.py), "-I", "-c", code], timeout)


def _import_rule(env: DEnv, name: str, code: str) -> Result:
    rc, _, err = _ask(env, code)
    if rc == 0:
        return Result(name, OK, "imports")
    last = (err.strip().splitlines() or ["no message"])[-1]
    return Result(name, FAIL, f"cannot import: {last}")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pin_rule(env: DEnv) -> Result:
    """Every pinned package, not only the ones with a rule of their own: a package removed or changed after the build must show here, and
    the command in the detail repairs it because a build installs the pins again. Reads metadata only."""
    rc, out, _ = _ask(env, VERSIONS_CODE, 120)
    try:
        installed = {_norm(k): str(v) for k, v in json.loads(out).items()} if rc == 0 else None
    except (ValueError, AttributeError):
        installed = None
    if installed is None:
        return Result("pins", FAIL, f"the installed versions could not be listed. Run `{fix_command(env)}`")
    bad = []
    for pin in (TORCH, TORCHVISION, *REQUIREMENTS):
        name, want = pin.split("==")
        have = installed.get(_norm(name))
        if have is None:
            bad.append(f"{name} missing (want {want})")
        elif have != want and not have.startswith(want + "+"):          # torch reports 2.7.0+cpu
            bad.append(f"{name} {have} (want {want})")
    if bad:
        return Result("pins", FAIL, "; ".join(bad) + f". Run `{fix_command(env)}`")
    return Result("pins", OK, f"{TORCH} and {len(REQUIREMENTS)} pinned packages are at their pinned versions")


def rules(env: DEnv, facts: dict | None) -> list[Result]:
    """What the demo needs. Needs the venv's python. Never imports ttnn or ttml."""
    out: list[Result] = []
    rc, ver, err = _ask(env, dist_code("numpy"), 60)
    ver = ver.strip()
    try:
        major = int(ver.split(".")[0])
    except ValueError:
        major = None
    if rc != 0 or major is None:
        out.append(Result("numpy", FAIL, f"the numpy version could not be read ({(err.strip() or ver)[-120:]})"))
    elif major >= 2:
        out.append(Result("numpy", FAIL, f"numpy {ver}: ttnn declares numpy below 2 and cv2 needs it. Run `{fix_command(env)}`"))
    else:
        out.append(Result("numpy", OK, f"numpy {ver}, below 2"))
    out.append(pin_rule(env))
    out.append(_import_rule(env, "cv2", CV2_CODE))
    out.append(_import_rule(env, "datasets", DATASETS_CODE))
    out.append(_import_rule(env, "torch", TORCH_CODE))
    versions = {}
    for dist in ("ttnn", "ttml"):
        rc, v, _ = _ask(env, dist_code(dist), 60)
        versions[dist] = v.strip() if rc == 0 and v.strip() else None
        out.append(Result(dist, OK, f"distribution {dist} {versions[dist]}") if versions[dist]
                   else Result(dist, FAIL, f"{dist} is not installed in {env.venv_dir}. Run `{fix_command(env)}`"))
    if facts is not None and versions["ttnn"]:
        want = tree_version(facts)
        out.append(Result("ttnn-version", OK, f"ttnn {versions['ttnn']} is the host tree's tag {facts['tag']}")
                   if versions["ttnn"] == want else
                   Result("ttnn-version", FAIL, f"ttnn {versions['ttnn']} in the venv, but the host tree is at {facts['tag']} ({want}). "
                                                f"Run `{fix_command(env)}`"))
    rec = read_record(env)
    if facts is not None and rec and isinstance(rec.get("host_tree"), dict):
        have = rec["host_tree"].get("tag")
        out.append(Result("record-tag", OK, f"built for {have}") if have == facts["tag"] else
                   Result("record-tag", FAIL, f"the venv was built for host tree {have}; the tree is at {facts['tag']}. Run `{fix_command(env)}`"))
    return out


def ensure(env: DEnv, *, check_only: bool, say=lambda s: None) -> list[Result]:
    """Run the steps (or only report them), then the rules. A refusal ends it before anything runs."""
    bad = refusal(env)
    if bad:
        say(f"{bad.status:5s} {bad.name}: {bad.detail}")
        return [bad]
    results: list[Result] = []

    def add(r: Result) -> Result:
        results.append(r)
        say(f"{r.status:5s} {r.name}: {r.detail}")
        return r

    facts, problem = tree_facts(env)
    venv = add(step_venv(env, check_only))
    pk = add(step_packages(env, check_only)) if venv.status != FAIL else None
    tree = add(step_host_tree(env, check_only, facts, problem)) if venv.status != FAIL else None
    if venv.status != FAIL:
        add(step_record(env, check_only, facts, upstream_ok=bool(pk and pk.status == OK and tree and tree.status == OK)))
    if env.py.exists():
        for r in rules(env, facts):
            add(r)
    return results


def main(args, *, env: DEnv | None = None, out=None) -> int:
    """`tt-orchard demo-env [--check] [--dry-run] [--venv-dir D] [--tt-metal-dir D]`.
    Exit 0 when every step is ok (a dry run: when nothing failed), 2 otherwise."""
    say = (lambda line: print(line, file=out)) if out is not None else print
    env = env or env_from_args(args)
    check_only = bool(getattr(args, "check", False) or getattr(args, "dry_run", False))
    results = ensure(env, check_only=check_only, say=say)
    if getattr(args, "dry_run", False) and not getattr(args, "check", False):
        return 2 if any(r.status == FAIL for r in results) else 0
    return 0 if all(r.status == OK for r in results) else 2
