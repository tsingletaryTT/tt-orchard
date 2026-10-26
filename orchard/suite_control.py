# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard suite-control`: produce the prompt suite's control from the nearest model (issue 15).

A control is the same prompt suite served by the nearest model on the same bundle and recorded the way stage 2
records the new model (`evidence/swap-check.json` with a `prompt_suite` record). The gate reads the new model's
findings against it (orchard/prompt_suite.py `evaluate`). Making one costs a boot of the nearest model, so nothing
runs it by itself.

Two actions:

  prepare  Build `<stage dir>/suite-control/` from the stage's own `swap_config.json`: the shipped swap scripts, a
           config that names the nearest model as the model to serve (`new_model_id` = `nearest_model_id`,
           `new_snapshot` = `base_snapshot`), its own tensor cache (`<tt_cache>-control`), no graft and no control of
           its own. When the stage's own run.sh serves with the drafter off, the control gets `"drafter": "off"` so it
           serves the same way (a control with the drafter on could differ from the run for that reason alone). Runs the shipped `prepare_swap.py` there (it opens no chip) and prints the `gozer run` line that
           serves it. Refuses a stage whose config already serves the nearest model.
  run      `prepare`, then that `gozer run` command. A queued request (exit 10) is reported with its ticket and is
           not cancelled; wait with `gozer wait <ticket>` and run the printed serve command yourself.

The result is `<stage dir>/suite-control/evidence/swap-check.json`. To use it, add
`"suite_control": "<that path, relative to the run directory>"` to the stage's `swap_config.json` before the stage's
test runs; serve_and_compare.py then embeds it and checks that it came from the nearest model.

The command never releases a lease, resets a chip or publishes anything. Only `run` starts a process that opens a
chip, and only through `gozer run`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parent.parent
TEMPLATES = Path("orchard") / "skills" / "weights-swap-templates"
CONTROL_DIR = "suite-control"
CACHE_SUFFIX = "-control"
SUITE_FILE = "prompt_suite.json"
QUEUED_EXIT = 10            # gozer's exit status for a request that was queued
REQUIRED = ("run_dir", "nearest_model_id", "base_snapshot", "new_snapshot", "new_model_id", "tt_cache", "hf_home", "port")


class ControlError(Exception):
    """The control cannot be built from this stage; the message says why."""


def scripts_for(cfg: dict) -> tuple[str, str]:
    """(the prepare script, the serve script) for the stage's serving type: a container package or a bundle's run.sh."""
    if cfg.get("package"):
        return "prepare_swap.py", "serve_and_compare_container.py"
    return "prepare_swap.py", "serve_and_compare.py"


def files_for(cfg: dict) -> list[str]:
    """Every shipped file the control directory needs. The container script imports serve_and_compare.py, so both serving types copy it."""
    prepare_name, serve_name = scripts_for(cfg)
    return list(dict.fromkeys([prepare_name, serve_name, "serve_and_compare.py", SUITE_FILE]))


def _repo(model_id: str) -> str:
    return str(model_id).split("@", 1)[0]


def stage_drafter_off(stage: Path) -> bool:
    """True when the stage was served with the drafter off. A bundle stage says so in its own run.sh (an empty QWEN36_DRAFTER export; the
    last such line wins, as in bash; a comment does not count). A container stage has no run.sh, so the `serving.drafter` of its
    evidence/swap-check.json is read ("off" is what the container template records from the docker command it ran). Anything else is False."""
    stage = Path(stage)
    try:
        text = (stage / "run.sh").read_text(encoding="utf-8")
    except OSError:
        text = None
    if text is not None:
        values = [m.group(1).strip().strip("\"'") for ln in text.splitlines()
                  if (m := re.match(r"\s*export\s+QWEN36_DRAFTER=(.*)$", ln))]
        return bool(values) and values[-1] == ""
    try:
        return json.loads((stage / "evidence" / "swap-check.json").read_text(encoding="utf-8"))["serving"]["drafter"] == "off"
    except (OSError, ValueError, KeyError, TypeError):
        return False


PATH_KEYS = ("bundle_dir", "base_snapshot", "new_snapshot", "tt_cache", "hf_home", "operator_home")


def config_chips(cfg: dict) -> int:
    """How many chips the control's server needs: the container package's `chips`, or the bundle's own `device_count`
    (tt_kernel_manifest.json). Never a guess, so a 4-chip bundle is not leased 2 chips."""
    if cfg.get("package"):
        n = cfg.get("chips")
        where = "the stage's config"
    else:
        where = f"{cfg.get('bundle_dir')}/tt_kernel_manifest.json"
        try:
            n = json.loads((Path(str(cfg["bundle_dir"])) / "tt_kernel_manifest.json").read_text(encoding="utf-8")).get("device_count")
        except (OSError, ValueError, AttributeError):
            n = None
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ControlError(f"cannot tell the chip count the control needs: {where} has no whole number of chips")
    return n


def run_dir_of(stage: Path) -> Path:
    """The run directory of a stage directory: the parent of the last `stages` component (stages/2, stages/4/configs/N)."""
    parts = Path(stage).resolve().parts
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "stages":
            return Path(*parts[:i])
    raise ControlError(f"{stage} is not under a `stages` directory, so the run directory cannot be found")


def _absolute(value, run_dir: Path) -> str:
    """A path the stage's config holds, made absolute against the run directory (the control runs from suite-control/, so a relative path would
    point somewhere else there)."""
    p = Path(str(value)).expanduser()
    return str(p if p.is_absolute() else (Path(run_dir) / p).resolve())


def control_config(cfg: dict, *, run_dir, port=None, drafter_off=False) -> dict:
    """The stage's config with the nearest model as the model served. Raises ControlError for a config that cannot make one."""
    missing = [k for k in REQUIRED if cfg.get(k) in (None, "")]
    if missing:
        raise ControlError(f"the stage's swap_config.json has no {', '.join(missing)}")
    if not cfg.get("bundle_dir") and not cfg.get("package"):
        raise ControlError("the stage's swap_config.json names neither a bundle_dir nor a package")
    cfg = {**cfg, **{k: _absolute(cfg[k], run_dir) for k in PATH_KEYS if cfg.get(k)}}
    if _repo(cfg["new_model_id"]) == _repo(cfg["nearest_model_id"]) or cfg["new_snapshot"] == cfg["base_snapshot"]:
        raise ControlError("this stage already serves the nearest model, so it is a control and has none of its own")
    out = {k: v for k, v in cfg.items() if k not in ("graft_mtp", "suite_control")}
    out.update({"new_model_id": cfg["nearest_model_id"], "new_snapshot": cfg["base_snapshot"],
                "tt_cache": str(cfg["tt_cache"]).rstrip("/") + CACHE_SUFFIX, "prompt_suite": True})
    if port is not None:
        out["port"] = int(port)
    if drafter_off:
        out["drafter"] = "off"       # a run served without the drafter needs a control served the same way
    return out


def _tail(text: str, n: int = 12) -> str:
    return " | ".join(x.strip() for x in text.splitlines()[-n:] if x.strip())[:600]


def prepare(stage_dir, *, orchard_dir=CHECKOUT, port=None, chips=None, run=subprocess.run, python=None) -> dict:
    """Build the control directory and run its prepare script. Returns {control_dir, serve, chips, evidence, wire, config}."""
    stage = Path(stage_dir).resolve()
    try:
        cfg = json.loads((stage / "swap_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ControlError(f"cannot read {stage / 'swap_config.json'}: {exc}") from None
    if not isinstance(cfg, dict):
        raise ControlError(f"{stage / 'swap_config.json'} is not a JSON object")
    off = stage_drafter_off(stage)
    run_dir = Path(cfg["run_dir"]) if isinstance(cfg.get("run_dir"), str) and os.path.isabs(cfg["run_dir"]) else run_dir_of(stage)
    ccfg = control_config(cfg, run_dir=run_dir, port=port, drafter_off=off)
    n_chips = config_chips(ccfg)
    if chips is not None and int(chips) != n_chips:
        raise ControlError(f"the server needs {n_chips} chips (from {'the package' if cfg.get('package') else 'the bundle'}'s own metadata); "
                           f"--chips {chips} would make the lease and the config disagree")
    ccfg["chips"] = n_chips
    ccfg["run_dir"] = str(run_dir)
    prepare_name, serve_name = scripts_for(cfg)
    templates = Path(orchard_dir) / TEMPLATES
    names = files_for(cfg)
    absent = [n for n in names if not (templates / n).is_file()]
    if absent:
        raise ControlError(f"the checkout's templates directory {templates} lacks {', '.join(absent)}")
    control = stage / CONTROL_DIR
    shutil.rmtree(control, ignore_errors=True)          # a control from an earlier attempt must not be read as this one
    control.mkdir(parents=True)
    for n in names:
        shutil.copy2(templates / n, control / n)
    (control / "swap_config.json").write_text(json.dumps(ccfg, indent=2) + "\n", encoding="utf-8")
    py = python or sys.executable
    res = run([py, str(control / prepare_name)], cwd=str(control), capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        raise ControlError(f"{prepare_name} exited with code {res.returncode}: {_tail((res.stdout or '') + (res.stderr or ''))}")
    run_dir = Path(ccfg["run_dir"]).resolve()
    evidence = control / "evidence" / "swap-check.json"
    try:
        wire = str(evidence.relative_to(run_dir))
    except ValueError:
        wire = str(evidence)
    return {"control_dir": str(control), "serve": [py, str(control / serve_name)], "chips": n_chips,
            "evidence": str(evidence), "wire": wire, "config": ccfg, "mirrors_drafter_off": off}


def gozer_argv(prepared: dict, *, who: str, reason: str, gozer: str = "gozer") -> list[str]:
    """The one lease command that serves the control. `--exact` is left out: gozer picks a free board of that size."""
    return [gozer, "run", "--chips", str(prepared["chips"]), "--who", who, "--reason", reason, "--", *prepared["serve"]]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard suite-control", description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=("prepare", "run"))
    p.add_argument("--stage-dir", required=True, help="the stage directory (stages/2, or stages/4/configs/N) whose nearest model is served")
    p.add_argument("--chips", type=int, help="must equal the chips the bundle or package needs (read from its own metadata); a different number is refused")
    p.add_argument("--port", type=int, help="port for the control's server (default: the stage's port)")
    p.add_argument("--who", default="claude:suite-control")
    p.add_argument("--reason", default="serve the nearest model with the prompt suite (control)")
    p.add_argument("--gozer", default="gozer")
    p.add_argument("--orchard-dir", default=str(CHECKOUT), help="the tt-orchard checkout whose templates are copied")
    return p


def main(argv=None, *, out=None, run=subprocess.run, call=subprocess.call) -> int:
    """Exit 0 done, 2 refused or failed to prepare, 3 gozer queued the request, otherwise the serve command's exit status."""
    out = out or sys.stdout
    a = _parser().parse_args(argv)
    try:
        prepared = prepare(a.stage_dir, orchard_dir=a.orchard_dir, port=a.port, chips=a.chips, run=run)
    except ControlError as exc:
        print(f"suite-control: {exc}", file=sys.stderr)
        return 2
    argv_g = gozer_argv(prepared, who=a.who, reason=a.reason, gozer=a.gozer)
    print(json.dumps({"control_dir": prepared["control_dir"], "evidence": prepared["evidence"],
                      "suite_control": prepared["wire"], "gozer": argv_g}, indent=1), file=out)
    if a.action == "prepare":
        return 0
    code = call(argv_g, cwd=prepared["control_dir"])
    if code == QUEUED_EXIT:
        print("suite-control: gozer queued the request. Wait for the ticket (gozer wait <ticket>), then run the command above.",
              file=sys.stderr)
        return 3
    return code


if __name__ == "__main__":
    sys.exit(main())
