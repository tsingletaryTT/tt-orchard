# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prepare a machine for tt-orchard: `scripts/setup.sh`, or `python3 -m orchard.setup_machine`.

It checks every prerequisite, says what is missing, and with your yes does what it can do safely:

    tt-gozer          cloned and installed with its own installer when `gozer` is not on PATH. tt-orchard
                      holds every board through gozer leases, so a gozer that lacks `reset` or
                      `acquire --owner-pid` is a failure, not a warning. Stale leases are reported and
                      never cleared here (`gozer reconcile` is yours to run).
    tt-model, docker  checked only. They are installed from their own projects.
    hf                installed with uv or pip when missing (downloads happen before a run, not during).
    ollama            checked; the CPU-tier model is pulled with your yes; `--start-ollama` starts the server.
    reference venv    a venv with CPU torch and a transformers new enough for the Qwen3.5 family, which
                      stage 1 runs the CPU reference with. Agents are not allowed to install packages.
    coder package     `tt-model pull` of the recommended agent model for the machine.
    config            config/bringup.toml and config/tiers.toml, written from the QuietBox 2 templates and
                      NEVER overwritten.
    tt-orchard        linked into ~/.local/bin as `tt-orchard` (a link, so it follows the checkout).

Nothing here publishes, pushes, resets a chip or clears a lease. Every command it runs is printed first.
Without `--yes` it asks before each one; `--check` runs nothing. Every step is safe to run again.

The recommended agent model for a QuietBox 2 (two p300c boards) is Qwen3-Coder-Next on one board. That is
a recommendation from one machine and one bring-up (README, "Recommended setup"), so `--coder 27b` selects the
earlier Qwen3.8-27B arrangement, and any other machine gets a warning and no configuration written.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from orchard import ui

OK, TODO, WARN, FAIL = "ok", "todo", "warn", "fail"
MIN_PYTHON = (3, 12)
MIN_GOZER = (0, 3, 2)                       # the first gozer with `reset <lease>` (README 2)
GOZER_URL = "https://github.com/tsingletaryTT/tt-gozer.git"
TT_MODEL_URL = "https://github.com/tenstorrent/tt-model-manager"
OLLAMA_MODEL = "qwen3-coder:30b"
REFERENCE_TRANSFORMERS = "transformers==5.19.0"       # the version stage 1 was run with on Clef
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
COMMANDS_NEVER_RUN = ("reset", "publish", "push", "reconcile", "acquire", "release")   # asserted in the tests

CODERS = {
    "coder-next": {"package": "raahemnabeel/qwen3-coder-next-blackhole",
                   "note": "Qwen3-Coder-Next on one board (the p300 profile); the other board stays free",
                   "weights_gb": 160},
    "27b": {"package": "mando2222/qwen3.8-27b-dflash2-p300x2-q4kv",
            "note": "Qwen3.8-27B on all four chips, the earlier arrangement", "weights_gb": 60},
}


@dataclass
class Action:
    """One thing a step does: ("run", argv), ("write", path, text), ("link", link, target) or
    ("spawn", argv, log_path)."""
    kind: str
    args: tuple
    label: str | None = None      # what to print instead of a long argv (an inline script)

    def describe(self) -> str:
        if self.label is not None:
            return self.label
        if self.kind == "run":
            return " ".join(self.args[0])
        if self.kind == "write":
            return f"write {self.args[0]}"
        if self.kind == "link":
            return f"link {self.args[0]} -> {self.args[1]}"
        return f"start in the background: {' '.join(self.args[0])} (log {self.args[1]})"


@dataclass
class Step:
    name: str
    status: str
    detail: str
    actions: list[Action] = field(default_factory=list)


@dataclass
class Env:
    """Everything the checks read from the machine, so a test supplies its own."""
    home: Path
    checkout: Path
    which: Callable = shutil.which
    run: Callable = None            # (argv, timeout) -> (returncode, stdout, stderr)
    spawn: Callable = None          # (argv, log_path) -> pid
    exists: Callable = os.path.exists
    free_gb: Callable = None        # (path) -> free decimal GB
    python_version: tuple = sys.version_info[:3]
    environ: dict = field(default_factory=lambda: dict(os.environ))


def _run(argv, timeout=60):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


def _spawn(argv, log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True).pid


def _free_gb(path):
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def default_env(checkout: Path | None = None) -> Env:
    return Env(home=Path.home(), checkout=checkout or Path(__file__).resolve().parent.parent,
               run=_run, spawn=_spawn, free_gb=_free_gb)


# ---- options ---------------------------------------------------------------------------------------

@dataclass
class Options:
    coder: str = "coder-next"
    start_ollama: bool = False
    runs_root: Path | None = None
    venv_dir: Path | None = None
    gozer_dir: Path | None = None


def runs_root_for(env: Env, opts: Options) -> Path:
    return opts.runs_root or env.home / "orchard-runs"


def venv_dir_for(env: Env, opts: Options) -> Path:
    return opts.venv_dir or env.home / ".local" / "share" / "tt-orchard" / "venvs" / "reference"


def gozer_dir_for(env: Env, opts: Options) -> Path:
    return opts.gozer_dir or env.home / "code" / "tt-gozer"


def parse_version(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(x) for x in m.groups()) if m else None


# ---- the checks -------------------------------------------------------------------------------------

def check_python(env: Env) -> Step:
    have = ".".join(str(x) for x in env.python_version[:3])
    if tuple(env.python_version[:2]) >= MIN_PYTHON:
        return Step("python", OK, f"Python {have}")
    return Step("python", FAIL, f"Python {have} is too old; tt-orchard needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer")


GOZER_TEXT = {}      # filled per call: the last `gozer status` text, read by the machine checks


def check_gozer(env: Env, opts: Options) -> Step:
    """tt-gozer: installed, new enough, able to reset and to lease under an owner pid, and reading the boards."""
    if not env.which("gozer"):
        target = gozer_dir_for(env, opts)
        acts = [] if env.exists(target / "install.sh") else [Action("run", (["git", "clone", GOZER_URL, str(target)],))]
        acts.append(Action("run", ([str(target / "install.sh")],)))
        return Step("tt-gozer", TODO, f"gozer is not on PATH. It leases the chips, and tt-orchard needs it. Installing "
                    f"clones {GOZER_URL} to {target} and runs its install.sh (links `gozer` into ~/.local/bin and its "
                    "skills into ~/.claude/skills)", acts)
    rc, out, err = env.run(["gozer", "--version"], 20)
    ver = parse_version(out + err)
    if ver is None or ver < MIN_GOZER:
        return Step("tt-gozer", FAIL, f"gozer {'.'.join(map(str, ver)) if ver else 'of unknown version'} is older than "
                    f"{'.'.join(map(str, MIN_GOZER))}; update it: git -C {gozer_dir_for(env, opts)} pull, then run its install.sh")
    missing = []
    rc, out, err = env.run(["gozer", "acquire", "--help"], 20)
    if "--owner-pid" not in out + err:
        missing.append("`acquire --owner-pid`")
    rc, out, err = env.run(["gozer", "reset", "--help"], 20)
    if rc != 0:
        missing.append("`reset`")
    if missing:
        return Step("tt-gozer", FAIL, f"gozer {'.'.join(map(str, ver))} lacks {' and '.join(missing)}, which tt-orchard "
                    "needs to hold a board through a model swap. Update tt-gozer")
    rc, out, err = env.run(["gozer", "status"], 30)
    GOZER_TEXT["status"] = out
    if rc != 0:
        return Step("tt-gozer", FAIL, f"`gozer status` failed: {(err or out).strip()[:200]}")
    chips = re.findall(r"^\s*chip\s+\d+\s+\S+\s+(\S+)", out, re.M)
    if not chips:
        return Step("tt-gozer", FAIL, "`gozer status` lists no chips; check the driver with tt-smi")
    stale = sum(1 for s in chips if s == "STALE")
    busy = sum(1 for s in chips if s not in ("FREE", "STALE"))
    detail = f"gozer {'.'.join(map(str, ver))}: {len(chips)} chips"
    if stale:
        return Step("tt-gozer", WARN, detail + f", {stale} with a stale lease. Run `gozer reconcile` yourself when "
                    "nothing of yours is running; setup does not clear leases")
    if busy:
        return Step("tt-gozer", WARN, detail + f", {busy} in use right now")
    return Step("tt-gozer", OK, detail + ", all free")


def machine_kind(text: str) -> str | None:
    """`quietbox2` for two boards of p300 with four chips in `gozer status`; otherwise None."""
    m = re.search(r"\((\d+) boards?, (\d+) chips?\)", text or "")
    if m and (int(m.group(1)), int(m.group(2))) == (2, 4) and "p300" in text:
        return "quietbox2"
    return None


def check_tt_model(env: Env) -> Step:
    if not env.which("tt-model"):
        return Step("tt-model", FAIL, f"tt-model is not installed. It serves the agents' model and the model under "
                    f"test. Install tt-model-manager from {TT_MODEL_URL} (it is not on PyPI), then run this again")
    rc, out, err = env.run(["tt-model", "--version"], 30)
    return Step("tt-model", OK if rc == 0 else FAIL,
                f"tt-model {out.strip()}" if rc == 0 else f"`tt-model --version` failed: {(err or out).strip()[:200]}")


def check_docker(env: Env) -> Step:
    if not env.which("docker"):
        return Step("docker", FAIL, "docker is not installed; container packages need it")
    rc, out, err = env.run(["docker", "info"], 30)
    if rc != 0:
        return Step("docker", FAIL, "docker is installed but this user cannot talk to it (is the user in the docker "
                    f"group?): {(err or out).strip().splitlines()[-1:] or ['']}"[:240])
    return Step("docker", OK, "docker answers")


def check_hf(env: Env) -> Step:
    if env.which("hf"):
        return Step("hf", OK, "the `hf` command is installed")
    cmd = (["uv", "tool", "install", "huggingface_hub"] if env.which("uv")
           else [sys.executable, "-m", "pip", "install", "--user", "huggingface_hub"])
    return Step("hf", TODO, "the `hf` command downloads the model's weights before a run. Installing huggingface_hub "
                "gives it", [Action("run", (cmd,))])


def check_ollama(env: Env, opts: Options) -> Step:
    if not env.which("ollama"):
        return Step("ollama", FAIL, "ollama is not installed. It serves the CPU tier (a stand-in while the coder is "
                    "parked). Install it from https://ollama.com/download, then run this again")
    rc, out, err = env.run(["ollama", "list"], 30)
    if rc != 0:
        log = env.home / ".cache" / "tt-orchard" / "ollama.log"
        acts = [Action("spawn", (["ollama", "serve"], str(log)))] if opts.start_ollama else []
        return Step("ollama", WARN if not acts else TODO, "the ollama server is not answering. Start it with "
                    "`ollama serve`, or run this with --start-ollama" + (" (it will start it)" if acts else ""), acts)
    if OLLAMA_MODEL not in out:
        return Step("ollama", TODO, f"{OLLAMA_MODEL} is not pulled (about 18 GB)",
                    [Action("run", (["ollama", "pull", OLLAMA_MODEL],))])
    return Step("ollama", OK, f"ollama serves {OLLAMA_MODEL}")


def check_hugepages(env: Env) -> Step:
    if env.exists("/dev/hugepages-1G"):
        return Step("hugepages", OK, "/dev/hugepages-1G exists")
    return Step("hugepages", WARN, "/dev/hugepages-1G is missing; tt-model's containers mount it. See the "
                "Tenstorrent driver instructions to enable 1 GB hugepages")


def check_reference_venv(env: Env, opts: Options) -> Step:
    venv = venv_dir_for(env, opts)
    py = venv / "bin" / "python"
    imports = "import torch, transformers, tokenizers, safetensors"
    if env.exists(py):
        rc, out, err = env.run([str(py), "-c", imports], 180)
        if rc == 0:
            return Step("reference venv", OK, f"{py} imports torch, transformers, tokenizers and safetensors")
    if env.which("uv"):
        acts = [] if env.exists(py) else [Action("run", (["uv", "venv", str(venv), "--python", "3.12"],))]
        acts += [Action("run", (["uv", "pip", "install", "--python", str(py), "--index-url", TORCH_CPU_INDEX, "torch"],)),
                 Action("run", (["uv", "pip", "install", "--python", str(py), REFERENCE_TRANSFORMERS, "tokenizers",
                                 "safetensors", "pillow", "numpy"],))]
    else:
        acts = [] if env.exists(py) else [Action("run", ([sys.executable, "-m", "venv", str(venv)],))]
        acts += [Action("run", ([str(py), "-m", "pip", "install", "--index-url", TORCH_CPU_INDEX, "torch"],)),
                 Action("run", ([str(py), "-m", "pip", "install", REFERENCE_TRANSFORMERS, "tokenizers", "safetensors",
                                 "pillow", "numpy"],))]
    return Step("reference venv", TODO, f"stage 1 runs the CPU reference with its own interpreter ({py}) with CPU torch "
                f"and {REFERENCE_TRANSFORMERS}; agents may not install packages", acts)


def check_coder_package(env: Env, opts: Options) -> Step:
    c = CODERS[opts.coder]
    pkg = c["package"]
    if not env.which("tt-model"):
        return Step("coder package", WARN, f"cannot check {pkg} without tt-model")
    rc, out, err = env.run(["tt-model", "list"], 60)
    line = next((l for l in out.splitlines() if pkg in l), None)
    free = env.free_gb(env.home)
    disk = (f" The weights are about {c['weights_gb']} GB and only {free:.0f} GB is free on the home disk."
            if free < c["weights_gb"] + 40 else "")
    if line is None:
        return Step("coder package", TODO, f"{pkg} is not installed ({c['note']}). The package is an 8-9 GB image; the "
                    f"first serve also needs the model weights (about {c['weights_gb']} GB).{disk}",
                    [Action("run", (["tt-model", "pull", pkg],))])
    if "✗" in line:
        return Step("coder package", WARN, f"{pkg} is installed but `tt-model list` says it cannot be served here: "
                    f"{line.strip()[:160]}")
    return Step("coder package", WARN if disk else OK, f"{pkg} is installed and servable.{disk}")


def render(template: str, **values) -> str:
    for k, v in values.items():
        template = template.replace(f"@{k}@", v)
    return template


def check_config(env: Env, opts: Options, kind: str | None) -> Step:
    cfg = env.checkout / "config"
    bring, tiers = cfg / "bringup.toml", cfg / "tiers.toml"
    present = [p.name for p in (bring, tiers) if env.exists(p)]
    if len(present) == 2:
        return Step("config", OK, "config/bringup.toml and config/tiers.toml exist; setup never overwrites them")
    if kind != "quietbox2":
        return Step("config", WARN, "this does not look like a QuietBox 2 (two p300c boards), and the recommended "
                    "arrangement was measured only on one. Copy config/tiers.example.toml and "
                    "config/bringup.example.toml, fill them in (README 3.6) and run the role-fit test for your agent "
                    "model before you rely on it")
    ref = venv_dir_for(env, opts) / "bin" / "python"
    skills = env.home / "code" / "skills" / "plugins" / "tt-model-bringup" / "skills"
    skills_line = f'skills_dirs = ["{skills}"]\n' if env.exists(skills) else ""
    acts = []
    for dest, src, extra in ((bring, f"bringup.qb2-{opts.coder}.toml",
                              dict(RUNS_ROOT=str(runs_root_for(env, opts)), REFERENCE_PYTHON=str(ref),
                                   SKILLS_LINE=skills_line)),
                             (tiers, f"tiers.qb2-{opts.coder}.toml", {})):
        if not env.exists(dest):
            acts.append(Action("write", (str(dest), render((cfg / src).read_text(encoding="utf-8"), **extra))))
    return Step("config", TODO, f"writes {', '.join(Path(a.args[0]).name for a in acts)} for a QuietBox 2 with "
                f"{CODERS[opts.coder]['note']}", acts)


def check_link(env: Env) -> Step:
    link = env.home / ".local" / "bin" / "tt-orchard"
    target = env.checkout / "bin" / "tt-orchard"
    if os.path.islink(link):
        if os.path.realpath(link) == os.path.realpath(target):
            return Step("tt-orchard", OK, f"{link} points at this checkout")
        return Step("tt-orchard", WARN, f"{link} points somewhere else ({os.path.realpath(link)}); remove it yourself "
                    "to link this checkout")
    if env.exists(link):
        return Step("tt-orchard", WARN, f"{link} exists and is not a link; setup will not replace it")
    return Step("tt-orchard", TODO, f"puts `tt-orchard` on PATH ({link}, a link to this checkout)",
                [Action("link", (str(link), str(target)))])


def plan(env: Env, opts: Options) -> list[Step]:
    GOZER_TEXT.clear()
    steps = [check_python(env), check_gozer(env, opts), check_tt_model(env), check_docker(env), check_hf(env),
             check_ollama(env, opts), check_hugepages(env), check_reference_venv(env, opts),
             check_coder_package(env, opts)]
    steps.append(check_config(env, opts, machine_kind(GOZER_TEXT.get("status", ""))))
    steps.append(check_link(env))
    return steps


# ---- applying ---------------------------------------------------------------------------------------

def perform(action: Action, env: Env) -> tuple[bool, str]:
    if action.kind == "run":
        rc, out, err = env.run(action.args[0], 3600)
        return rc == 0, (err or out).strip()[-300:]
    if action.kind == "write":
        path = Path(action.args[0])
        if path.exists():
            return False, f"{path} appeared since the plan was made; not overwriting it"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(action.args[1], encoding="utf-8")
        return True, ""
    if action.kind == "link":
        link = Path(action.args[0])
        link.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(link):
            return False, f"{link} appeared since the plan was made; not replacing it"
        os.symlink(action.args[1], link)
        return True, ""
    pid = env.spawn(action.args[0], action.args[1])
    return True, f"started, pid {pid}"


def apply(steps: list[Step], env: Env, *, yes: bool, check: bool, ask: Callable[[str], bool],
          say: Callable[[str], None]) -> None:
    for step in steps:
        if step.status != TODO or not step.actions or check:
            continue
        say(f"{step.name}: " + "; ".join(a.describe() for a in step.actions))
        if not (yes or ask(f"run this for {step.name}?")):
            say("  skipped")
            continue
        for action in step.actions:
            ok, note = perform(action, env)
            if not ok:
                say(f"  failed: {action.describe()}" + (f" ({note})" if note else ""))
                break
            if note:
                say(f"  {note}")


MARKS = {OK: ("✅", "good"), TODO: ("📋", "warn"), WARN: ("⚠️", "warn"), FAIL: ("⛔", "bad")}


def render_steps(steps: list[Step], style: ui.Style) -> list[str]:
    out = []
    for s in steps:
        mark, role = MARKS[s.status]
        out.append(f"{style.icon(mark)}{style.paint(f'{s.name:<15}', role)} {s.status:<5} {s.detail}")
    return out


def main(argv=None, *, env: Env | None = None, ask=None, say=print, stdout=None) -> int:
    p = argparse.ArgumentParser(prog="tt-orchard setup", description=__doc__.split("\n\n")[0])
    p.add_argument("--coder", choices=sorted(CODERS), default="coder-next",
                   help="the agent model to set up for a QuietBox 2 (default: coder-next, the recommendation)")
    p.add_argument("--yes", action="store_true", help="do every step that can be done, without asking")
    p.add_argument("--check", action="store_true", help="only check and report; run nothing")
    p.add_argument("--start-ollama", action="store_true", help="start `ollama serve` in the background if it is down")
    p.add_argument("--runs-root", type=Path, help="where runs live (default ~/orchard-runs)")
    p.add_argument("--venv-dir", type=Path, help="the reference venv (default ~/.local/share/tt-orchard/venvs/reference)")
    p.add_argument("--gozer-dir", type=Path, help="where to clone tt-gozer (default ~/code/tt-gozer)")
    args = p.parse_args(argv)
    env = env or default_env()
    opts = Options(args.coder, args.start_ollama, args.runs_root, args.venv_dir, args.gozer_dir)
    ask = ask or (lambda q: sys.stdin.isatty() and input(f"{q} [y/N] ").strip().lower() in ("y", "yes"))
    style = ui.detect(stdout or sys.stdout, dict(env.environ), "auto")
    say(f"{style.icon('🍎')}tt-orchard setup ({args.coder}: {CODERS[args.coder]['note']})")
    steps = plan(env, opts)
    for line in render_steps(steps, style):
        say(line)
    apply(steps, env, yes=args.yes, check=args.check, ask=ask, say=say)
    if not args.check and any(s.status == TODO and s.actions for s in steps):
        say("checking again after the steps above:")
        steps = plan(env, opts)
        for line in render_steps(steps, style):
            say(line)
    failed = [s for s in steps if s.status == FAIL]
    todo = [s for s in steps if s.status == TODO]
    if failed:
        say(f"{len(failed)} step(s) need you: " + ", ".join(s.name for s in failed))
        return 1
    if todo:
        say(f"{len(todo)} step(s) not done: " + ", ".join(s.name for s in todo)
            + ("" if args.check else ". Run again with --yes to do them without asking"))
        return 1
    say("ready. Try: tt-orchard bringup <org/name> --dry-run   (then, with the coder package's weights downloaded, "
        "python3 -m orchard.rolefit to check the agent model on this machine)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
