# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard lab setup`: get a lab box ready for `--lab`, from the brain.

It reads the `[lab]` table of config/bringup.toml and checks, in order: that ssh reaches the lab; that
the lab root exists and is writable on both boxes (the same absolute path on each); the run layout under
it; gozer, hugepages and the test interpreter on the lab; the brain's reference interpreter; that every
bundle installed under the lab root on the brain is also on the lab; the models asked for with --model;
that the two boxes run the same firmware; and that the lab's chips are free.

Like `tt-orchard setup` it prints every command before it runs it, asks first unless --yes, and runs
nothing with --check. It never runs sudo (a missing lab root is the operator's to create, and it says
how), never takes, resets, releases or reconciles a lease, and never copies a tensor cache to the lab.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

from orchard import bundle_relink
from orchard.labclient import SSH
from orchard.setup_machine import (FAIL, OK, TODO, WARN, Action, Step, _run, apply, render_steps)

LAYOUT = ("runs", "cache", "hf/hub", "tt-model/models", "venvs")
BUNDLE_SKIP = (".tt_cache", ".hf", ".cache")         # caches: each box keeps its own
GOZER_URL = "https://github.com/tsingletaryTT/tt-gozer.git"
CHIP_STATE = re.compile(r"^\s*chip\s+\d+\s+\S+\s+(\S+)", re.M)


def remote(lab, command: str, ssh=SSH) -> list[str]:
    """argv that runs `command` on the lab with its extra PATH directories first."""
    path = ":".join(lab.path)
    prefix = f'export PATH="{path}:$PATH"; ' if path else ""
    return [*ssh, lab.host, prefix + command]


def plan(lab, *, run, reference_python, local_fw, models=(), model_sources=(), tt_model_root=None,
         hf_home=None, ssh=SSH) -> list[Step]:
    root = Path(lab.root)
    tt_model_root = Path(tt_model_root or root / "tt-model" / "models")
    hf_home = Path(hf_home or root / "hf")
    q = shlex.quote
    steps: list[Step] = []

    rc, out, err = run([*ssh, lab.host, "true"], 30)
    if rc != 0:
        return [Step("ssh", FAIL, f"ssh {lab.host} failed: {(err or out).strip()[-200:]}. Set up key login "
                                  "(BatchMode: no password prompt) first")]
    steps.append(Step("ssh", OK, f"ssh {lab.host} answers"))

    make = f"sudo mkdir -p {root} && sudo chown \"$USER\" {root}"
    if root.is_dir() and os.access(root, os.W_OK):
        steps.append(Step("brain root", OK, f"{root} is writable here"))
    else:
        steps.append(Step("brain root", FAIL, f"{root} is missing or not writable here. Create it once: {make}"))
    rc, _, _ = run(remote(lab, f"test -d {q(str(root))} -a -w {q(str(root))}", ssh), 30)
    if rc == 0:
        steps.append(Step("lab root", OK, f"{root} is writable on {lab.host}"))
    else:
        steps.append(Step("lab root", FAIL, f"{root} is missing or not writable on {lab.host}. Create it once "
                                            f"there: {make}"))
        return steps

    missing_here = [d for d in LAYOUT if not (root / d).is_dir()]
    rc, out, _ = run(remote(lab, "for d in " + " ".join(LAYOUT) + f"; do test -d {q(str(root))}/$d || echo $d; done",
                            ssh), 30)
    missing_there = out.split() if rc == 0 else list(LAYOUT)
    actions = []
    if missing_here:
        actions.append(Action("run", (["mkdir", "-p", *[str(root / d) for d in missing_here]],)))
    if missing_there:
        actions.append(Action("run", (remote(lab, "mkdir -p " + " ".join(q(f"{root}/{d}") for d in missing_there), ssh),)))
    steps.append(Step("layout", TODO if actions else OK,
                      "makes " + ", ".join(sorted(set(missing_here + missing_there))) if actions
                      else "runs, cache, hf, tt-model and venvs exist on both boxes", actions))

    rc, _, _ = run(remote(lab, f"command -v {q(lab.gozer)}", ssh), 30)
    if rc == 0:
        steps.append(Step("gozer (lab)", OK, f"{lab.gozer} is on the lab's PATH"))
    else:
        install = (f"mkdir -p ~/code && (test -d ~/code/tt-gozer || git clone {GOZER_URL} ~/code/tt-gozer) && "
                   "~/code/tt-gozer/install.sh")
        steps.append(Step("gozer (lab)", TODO, "gozer is not on the lab's PATH; installs tt-gozer with its own "
                                               "installer (links gozer into ~/.local/bin)",
                          [Action("run", (remote(lab, install, ssh),))]))

    rc, _, _ = run(remote(lab, "test -e /dev/hugepages-1G", ssh), 30)
    steps.append(Step("hugepages (lab)", OK if rc == 0 else FAIL,
                      "/dev/hugepages-1G exists" if rc == 0 else "the lab has no /dev/hugepages-1G; the TT runtime needs "
                                                                 "1 GB hugepages (see the tt-metal install guide)"))

    test_py = lab.test_python or str(root / "venvs" / "reference" / "bin" / "python")
    rc, _, _ = run(remote(lab, f"{q(test_py)} -c 'import tokenizers'", ssh), 120)
    if rc == 0:
        steps.append(Step("test python (lab)", OK, f"{test_py} imports tokenizers"))
    else:
        venv = str(Path(test_py).parent.parent)
        make_venv = (f"uv venv -p 3.12 {q(venv)} && uv pip install -p {q(venv)} torch --index-url "
                     f"https://download.pytorch.org/whl/cpu && uv pip install -p {q(venv)} transformers tokenizers safetensors")
        steps.append(Step("test python (lab)", TODO, f"makes {venv} on the lab (the hardware tests run with it)",
                          [Action("run", (remote(lab, make_venv, ssh),))]))

    rc, _, _ = run([str(reference_python), "-c", "import torch, transformers, tokenizers, safetensors"], 120)
    steps.append(Step("reference python", OK if rc == 0 else FAIL,
                      f"{reference_python} imports torch and transformers" if rc == 0 else
                      f"{reference_python} cannot import torch and transformers; stage 1 runs on this box with it. "
                      "Run `tt-orchard setup` with --venv-dir under the lab root"))

    bundles = sorted(p.parent for p in tt_model_root.glob("*/*/tt_kernel_manifest.json"))
    missing = []
    for b in bundles:
        rc, _, _ = run(remote(lab, f"test -f {q(str(b / 'tt_kernel_manifest.json'))}", ssh), 30)
        if rc != 0:
            missing.append(b)
    steps.append(_interpreters_step(lab, bundles, missing, run=run, ssh=ssh))
    actions = []
    for b in missing:
        argv = ["rsync", "-a", "--mkpath"]
        for skip in BUNDLE_SKIP:
            argv += ["--exclude", skip]
        argv += ["-e", " ".join(ssh), f"{b}/", f"{lab.host}:{b}/"]
        actions.append(Action("run", (argv,)))
    names = [str(b.relative_to(tt_model_root)) for b in missing]
    steps.append(Step("bundles (lab)", TODO if missing else OK,
                      ("copies " + ", ".join(names) + " to the lab (relocatable venvs; no caches)") if missing else
                      (f"{len(bundles)} bundle(s) under {tt_model_root} are on the lab" if bundles else
                       f"no bundles under {tt_model_root} yet: install the nearest model's bundles there with "
                       f"TT_MODEL_MODELS_DIR={tt_model_root} tt-model pull <bundle>"), actions))

    actions, notes = [], []
    for model in models:
        repo = "models--" + model.replace("/", "--")
        dst = hf_home / "hub" / repo
        if dst.exists():
            continue
        src = next((Path(s) / "hub" / repo for s in model_sources if (Path(s) / "hub" / repo).is_dir()), None)
        if src is None:
            notes.append(f"{model} is not in {hf_home} or the given caches; `hf download {model} --cache-dir {hf_home}/hub`")
            continue
        same = os.stat(src).st_dev == os.stat(hf_home if hf_home.exists() else root).st_dev
        actions.append(Action("run", (["cp", "-al", str(src), str(dst)] if same else
                                      ["rsync", "-a", "--mkpath", f"{src}/", f"{dst}/"],)))
    if notes:
        steps.append(Step("models", FAIL, "; ".join(notes), actions))
    else:
        steps.append(Step("models", TODO if actions else OK,
                          (f"places {len(actions)} model(s) in {hf_home} (hard links when on the same disk)") if actions
                          else "the run's models are copied to the lab before each test (rsync, incremental)", actions))

    rc, out, _ = run(remote(lab, "cat /sys/class/tenstorrent/*/tt_fw_bundle_ver 2>/dev/null | sort -u", ssh), 30)
    lab_fw = out.split()
    if rc != 0 or not lab_fw:
        steps.append(Step("firmware", WARN, "the lab's firmware version could not be read"))
    elif lab_fw == [local_fw]:
        steps.append(Step("firmware", OK, f"both boxes run firmware {local_fw}"))
    else:
        steps.append(Step("firmware", WARN, f"the lab runs firmware {', '.join(lab_fw)} and this box {local_fw}; a "
                                            "bundle measured on one may not serve on the other"))

    rc, out, _ = run(remote(lab, f"{q(lab.gozer)} status", ssh), 60)
    states = CHIP_STATE.findall(out) if rc == 0 else []
    if not states:
        steps.append(Step("lab chips", WARN, "`gozer status` on the lab listed no chips"))
    else:
        busy = [s for s in states if s != "FREE"]
        steps.append(Step("lab chips", OK if not busy else WARN,
                          f"{len(states)} chips, all free" if not busy else
                          f"{len(busy)} of {len(states)} chips are not free ({', '.join(sorted(set(busy)))}); setup "
                          "never clears a lease"))
    return steps


def _interpreters_step(lab, bundles, missing, *, run, ssh=SSH) -> Step:
    """A bundle copied under the lab root still runs the interpreter of the directory tt-model installed it
    in (orchard/bundle_relink.py). Fixed on the brain first, so a bundle copied to the lab afterwards is
    already right; bundles already on the lab are checked and fixed there."""
    q = shlex.quote
    src = Path(bundle_relink.__file__).read_text()
    here = [b for b in bundles if bundle_relink.problems(b)]
    present = [b for b in bundles if b not in missing]
    there: list = []
    unknown = False
    if present:
        rc, out, _ = run(remote(lab, f"python3 -c {q(src)} --check " + " ".join(q(str(b)) for b in present), ssh), 60)
        if rc == 1:
            there = [b for b in present if f"{b}: " in out]
        elif rc != 0:
            unknown = True
    actions = []
    if here:
        actions.append(Action("run", ([sys.executable, "-c", src, *map(str, here)],)))
    if there:
        actions.append(Action("run", (remote(lab, f"python3 -c {q(src)} " + " ".join(q(str(b)) for b in there), ssh),)))
    if unknown:
        return Step("bundle interpreters", WARN, "could not check the bundles' interpreters on the lab", actions)
    if not actions:
        return Step("bundle interpreters", OK, "every bundle's venv runs the interpreter inside the bundle")
    where = [f"{len(here)} here"] * bool(here) + [f"{len(there)} on {lab.host}"] * bool(there)
    return Step("bundle interpreters", TODO, "relinks the venv interpreter inside the bundle for " + " and ".join(where)
                + " (it still names the directory tt-model installed it in)", actions)


def local_firmware() -> str:
    for p in sorted(Path("/sys/class/tenstorrent").glob("*/tt_fw_bundle_ver")):
        try:
            return p.read_text().strip()
        except OSError:
            continue
    return "unknown"


def main(argv=None, *, cfg=None, run=_run, say=print, ask=None) -> int:
    p = argparse.ArgumentParser(prog="tt-orchard lab setup", description=__doc__.split("\n\n")[0])
    p.add_argument("--model", action="append", default=[], metavar="ORG/NAME",
                   help="a model the lab run uses (the new model, its base, a drafter); placed in the lab root's "
                        "Hugging Face cache, by hard links from ~/.cache/huggingface when it is on the same disk")
    p.add_argument("--yes", action="store_true", help="do every step that can be done, without asking")
    p.add_argument("--check", action="store_true", help="only check and report; run nothing")
    args = p.parse_args(argv)
    if cfg is None or cfg.lab is None:
        say("refused: config/bringup.toml has no [lab] table (host, root); see README 5.9")
        return 2
    from orchard import ui
    style = ui.detect(sys.stdout, dict(os.environ), "auto")
    ref = cfg.reference_python or (Path(cfg.lab.root) / "venvs" / "reference" / "bin" / "python")
    sources = [Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")]
    say(f"{style.icon('🧪')}tt-orchard lab setup: {cfg.lab.host}, lab root {cfg.lab.root}")
    kw = dict(run=run, reference_python=ref, local_fw=local_firmware(), models=args.model, model_sources=sources,
              tt_model_root=cfg.tt_model_root, hf_home=cfg.hf_home)
    steps = plan(cfg.lab, **kw)
    for line in render_steps(steps, style):
        say(line)
    ask = ask or (lambda q: sys.stdin.isatty() and input(f"{q} [y/N] ").strip().lower() in ("y", "yes"))
    env = type("E", (), {"run": staticmethod(run)})()
    apply(steps, env, yes=args.yes, check=args.check, ask=ask, say=say)
    if not args.check and any(s.status == TODO and s.actions for s in steps):
        say("checking again after the steps above:")
        steps = plan(cfg.lab, **kw)
        for line in render_steps(steps, style):
            say(line)
    failed = [s.name for s in steps if s.status == FAIL]
    todo = [s.name for s in steps if s.status == TODO]
    if failed or todo:
        say(f"not ready: {', '.join(failed + todo)}")
        return 1
    say(f"the lab is ready. Try: tt-orchard bringup <org/name> --dry-run   (runs use {cfg.lab.host} for hardware tests)")
    return 0
