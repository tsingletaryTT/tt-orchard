# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard suite-control`: produce the prompt suite's control from the nearest model (issue 15).

A control is the same prompt suite served by the nearest model on the same bundle and recorded the way stage 2
records the new model (`evidence/swap-check.json` with a `prompt_suite` record). The gate reads the new model's
findings against it (orchard/prompt_suite.py `evaluate`). Making one costs a boot of the nearest model, so nothing
runs it by itself.

Three actions:

  prepare  Build `<stage dir>/suite-control/` from the stage's own `swap_config.json`: the shipped swap scripts, a
           config that names the nearest model as the model to serve (`new_model_id` = `nearest_model_id`,
           `new_snapshot` = `base_snapshot`), its own tensor cache (`<tt_cache>-control`), no graft and no control of
           its own. When the stage was served with the drafter off (its run.sh, or for a container its swap-check.json), the control
           gets `"drafter": "off"` so it serves the same way (a control with the drafter on could differ from the run for that reason
           alone). Paths in the stage's config are made absolute against the run directory. The chip count is read from the bundle's
           or package's own metadata. Runs the shipped `prepare_swap.py` there (it opens no chip) and prints the launcher command,
           both as a `gozer run` line and as the `run` command. Refuses a stage whose config already serves the nearest model.
  run      `prepare`, then `gozer acquire` (a queued request exits 3 and names its ticket; nothing is cancelled), the launcher under
           that lease, and `gozer release`. `run --ticket T` claims a queued ticket with `gozer wait` and then does the same. It never
           starts a `gozer run`, which would queue behind a lease already held.
  launch   What runs inside the lease (`gozer run ... -- python -m orchard.suite_control launch --control-dir D`). It reads the chips from
           TT_VISIBLE_DEVICES, sets ORCHARD_DEVICE_IDS and a control-specific ORCHARD_TEST_LABEL for the container template, runs the
           serve script, and on every exit (SIGTERM included) stops the servers it started and removes its labelled containers, and
           checks that they are gone (exit 4 when it cannot).

The result is `<stage dir>/suite-control/evidence/swap-check.json`. To use it, add
`"suite_control": "<that path, relative to the run directory>"` to the stage's `swap_config.json` before the stage's
test runs; serve_and_compare.py then embeds it and checks that it came from the nearest model.

The command never resets a chip or publishes anything. Only `run` and `launch` start a process that opens a chip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
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
    launcher = ["env", f"PYTHONPATH={Path(orchard_dir).resolve()}", py, "-m", "orchard.suite_control", "launch", "--control-dir", str(control)]
    return {"control_dir": str(control), "serve": launcher, "serve_script": str(control / serve_name), "chips": n_chips,
            "evidence": str(evidence), "wire": wire, "config": ccfg, "mirrors_drafter_off": off}


def gozer_argv(prepared: dict, *, who: str, reason: str, gozer: str = "gozer") -> list[str]:
    """The lease command that runs the launcher. `--exact` is left out: gozer picks a free board of that size."""
    return [gozer, "run", "--chips", str(prepared["chips"]), "--who", who, "--reason", reason, "--", *prepared["serve"]]


# ---- the launcher: what runs inside the lease ----------------------------------------------------------------------------

CONTROL_VAR = "ORCHARD_CONTROL_DIR"   # marks every process the control starts, so the cleanup can find a server that outlived its script
EXIT_UNCLEAN = 4                      # the launcher could not confirm that every server and container is gone
EXIT_TERMINATED = 143                 # 128 + SIGTERM
STOP_GRACE_S = 30.0                   # how long the serve script gets after SIGTERM before SIGKILL


class Terminated(BaseException):
    """SIGTERM arrived. A BaseException so that no `except Exception` in between swallows it."""


def test_label(control_dir) -> str:
    """The docker label the control's container carries. It is not the supervisor's `orchard.test=` label, so the supervisor's own sweep
    never takes a control's container."""
    return "orchard.suite-control=" + hashlib.sha256(str(Path(control_dir).resolve()).encode()).hexdigest()[:12]


def device_ids(visible: str, status: dict) -> list[int]:
    """The /dev/tenstorrent indices of the chips in TT_VISIBLE_DEVICES (PCI addresses), in that order, from `gozer status --json`."""
    index = {c.get("bdf"): c.get("dev_index") for c in status.get("chips", []) if isinstance(c, dict)}
    bdfs = [b.strip() for b in str(visible).split(",") if b.strip()]
    if not bdfs:
        raise ControlError("TT_VISIBLE_DEVICES names no chip")
    out = []
    for b in bdfs:
        i = index.get(b)
        if isinstance(i, bool) or not isinstance(i, int):
            raise ControlError(f"gozer does not list a device index for chip {b}")
        out.append(i)
    return out


def child_env(env: dict, control_dir, status: dict) -> dict:
    """The environment of the serve script: the variables the container template reads (Supervisor._spawn sets the same two) and the marker."""
    out = dict(env)
    out["ORCHARD_DEVICE_IDS"] = ",".join(str(i) for i in device_ids(env.get("TT_VISIBLE_DEVICES", ""), status))
    out["ORCHARD_TEST_LABEL"] = test_label(control_dir)
    out[CONTROL_VAR] = str(Path(control_dir).resolve())
    return out


def _docker(argv: list[str], env=None) -> tuple[int, str]:
    try:
        r = subprocess.run(["docker", *argv], capture_output=True, text=True, timeout=120, env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _sweep(run_dir, **kw):
    from orchard import kernelprobe
    return kernelprobe.sweep_strays(run_dir, identity=True, **kw)


def cleanup(control_dir, label: str, *, sweep=_sweep, docker=_docker, uses_docker: bool) -> list[str]:
    """Stop what the serve script left running and say what could not be stopped. Servers: every process of this user whose environment holds
    CONTROL_VAR for this directory (SIGTERM, then SIGKILL). Containers (container configs only): every container with the control's label,
    removed and then listed again. Returns the problems; an empty list means everything is confirmed gone."""
    problems = []
    for rec in sweep(control_dir, var=CONTROL_VAR):
        if not rec.get("stopped"):
            problems.append(f"process {rec.get('pid')} ({str(rec.get('cmdline', ''))[:80]}) did not stop")
    if uses_docker:
        code, listed = docker(["ps", "-aq", "--filter", f"label={label}"])
        ids = listed.split() if code == 0 else []
        if code != 0:
            problems.append(f"docker could not list the control's containers (exit {code}: {listed.strip()[:120]})")
        elif ids:
            docker(["rm", "-f", *ids])
            code, again = docker(["ps", "-aq", "--filter", f"label={label}"])
            left = again.split() if code == 0 else []
            if code != 0:
                problems.append(f"docker could not confirm that the containers {' '.join(ids)} are gone (exit {code})")
            elif left:
                problems.append(f"container {' '.join(left)} is still there after docker rm -f")
    return problems


def launch(control_dir, *, env=None, gozer: str = "gozer", docker=None, sweep=_sweep) -> int:
    """Run the serve script of a control directory inside a lease and clean up after it. Needs TT_VISIBLE_DEVICES from the lease. Exit codes: the
    serve script's own; 2 refused before starting; 4 something it started could not be confirmed stopped; 143 terminated by SIGTERM."""
    env = dict(os.environ if env is None else env)
    docker = docker or (lambda argv: _docker(argv, env))
    control_dir = Path(control_dir).resolve()
    try:
        cfg = json.loads((control_dir / "swap_config.json").read_text(encoding="utf-8"))
        r = subprocess.run([gozer, "status", "--json"], capture_output=True, text=True, timeout=120, env=env)
        if r.returncode != 0:
            raise ControlError(f"`{gozer} status --json` exited {r.returncode}")
        ids = device_ids(env.get("TT_VISIBLE_DEVICES", ""), json.loads(r.stdout))
        if len(ids) != int(cfg["chips"]):
            raise ControlError(f"the control needs {cfg['chips']} chips and the lease has {len(ids)} ({env.get('TT_VISIBLE_DEVICES')})")
        child = child_env(env, control_dir, json.loads(r.stdout))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError, ControlError) as exc:
        print(f"suite-control launch: {exc}", file=sys.stderr)
        return 2
    serve = control_dir / scripts_for(cfg)[1]
    uses_docker = bool(cfg.get("package"))
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(Terminated()))
    code = None
    proc = None
    try:
        proc = subprocess.Popen([sys.executable, str(serve)], cwd=str(control_dir), env=child, stdin=subprocess.DEVNULL)
        code = proc.wait()
    except (Terminated, KeyboardInterrupt):
        code = EXIT_TERMINATED
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    finally:
        problems = cleanup(control_dir, test_label(control_dir), sweep=sweep, docker=docker, uses_docker=uses_docker)
        signal.signal(signal.SIGTERM, previous)
    if problems:
        print("suite-control launch: not everything the control started could be confirmed stopped: " + "; ".join(problems), file=sys.stderr)
        return EXIT_UNCLEAN
    return code


# ---- run: acquire (or wait for a ticket), launch under the lease, release --------------------------------------------------

def _gozer(argv: list[str]) -> tuple[int, str]:
    r = subprocess.run(argv, capture_output=True, text=True, timeout=7200)
    return r.returncode, r.stdout or ""


def _spawn(argv: list[str], env: dict, cwd: str) -> int:
    """Run the launcher in its own session and wait. SIGTERM to this process goes to the launcher (which cleans up) and then raises Terminated here."""
    proc = subprocess.Popen(argv, env=env, cwd=cwd, start_new_session=True)

    def forward(*_):
        raise Terminated()
    previous = signal.signal(signal.SIGTERM, forward)
    try:
        return proc.wait()
    except (Terminated, KeyboardInterrupt):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=STOP_GRACE_S * 3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        raise Terminated() from None
    finally:
        signal.signal(signal.SIGTERM, previous)


def _json(text: str) -> dict:
    try:
        v = json.loads(text)
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def _run(prepared: dict, a, *, gozer_call, spawn) -> int:
    """Take a lease (`gozer acquire`, or claim the ticket with `gozer wait`), run the launcher under it, always release it. No `gozer run`:
    a fresh `gozer run` behind a lease already held would queue behind that lease."""
    if a.ticket:
        code, text = gozer_call([a.gozer, "wait", a.ticket, "--json", "--timeout", a.wait_timeout])
    else:
        code, text = gozer_call([a.gozer, "acquire", "--chips", str(prepared["chips"]), "--who", a.who, "--reason", a.reason,
                                 "--expect", a.expect, "--json"])
    grant = _json(text)
    if not grant.get("granted"):
        if grant.get("reason") == "gone" or grant.get("error"):
            print(f"suite-control: {grant.get('error') or 'ticket ' + str(a.ticket) + ' is no longer queued'}", file=sys.stderr)
            return 2
        ticket = grant.get("ticket") or a.ticket
        pos = grant.get("position")
        print(f"suite-control: not granted yet: ticket {ticket}" + (f", position {pos}" if pos else "") + ". Nothing was started and nothing "
              f"is cancelled. Wait and take the lease with: tt-orchard suite-control run --stage-dir {a.stage_dir} --ticket {ticket}", file=sys.stderr)
        return 3
    lease, chips = grant.get("lease_id"), grant.get("chips")
    try:
        if not lease or not isinstance(chips, list) or not chips:
            print("suite-control: gozer granted a lease with no lease id or chips; releasing it", file=sys.stderr)
            return 2
        env = dict(os.environ, TT_VISIBLE_DEVICES=",".join(str(c) for c in chips))
        env.pop("TT_METAL_VISIBLE_DEVICES", None)
        try:
            return spawn(prepared["serve"], env, prepared["control_dir"])
        except Terminated:
            return EXIT_TERMINATED
    finally:
        if lease:
            rcode, rtext = gozer_call([a.gozer, "release", str(lease)])
            if rcode != 0:
                print(f"suite-control: gozer release {lease} exited {rcode}: {rtext.strip()[:200]}. The lease may still be held.", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard suite-control", description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=("prepare", "run", "launch"))
    p.add_argument("--stage-dir", help="the stage directory (stages/2, or stages/4/configs/N) whose nearest model is served")
    p.add_argument("--control-dir", help="launch only: the prepared suite-control directory")
    p.add_argument("--chips", type=int, help="must equal the chips the bundle or package needs (read from its own metadata); a different number is refused")
    p.add_argument("--port", type=int, help="port for the control's server (default: the stage's port)")
    p.add_argument("--who", default="claude:suite-control")
    p.add_argument("--reason", default="serve the nearest model with the prompt suite (control)")
    p.add_argument("--expect", default="60m", help="advisory duration for the lease")
    p.add_argument("--ticket", help="run only: a queue ticket from an earlier queued run; waits for it and takes the lease it grants")
    p.add_argument("--wait-timeout", default="9m", help="run only: how long --ticket waits per call (gozer wait's own limit applies)")
    p.add_argument("--gozer", default="gozer")
    p.add_argument("--orchard-dir", default=str(CHECKOUT), help="the tt-orchard checkout whose templates are copied")
    return p


def main(argv=None, *, out=None, run=subprocess.run, gozer_call=None, spawn=None) -> int:
    """Exit 0 done, 2 refused or failed to prepare, 3 not granted yet (a ticket is named), 4 something started could not be confirmed stopped,
    143 terminated, otherwise the serve script's exit status."""
    out = out or sys.stdout
    a = _parser().parse_args(argv)
    if a.action == "launch":
        if not a.control_dir:
            print("suite-control launch: --control-dir is required", file=sys.stderr)
            return 2
        return launch(a.control_dir, gozer=a.gozer)
    if not a.stage_dir:
        print("suite-control: --stage-dir is required", file=sys.stderr)
        return 2
    try:
        prepared = prepare(a.stage_dir, orchard_dir=a.orchard_dir, port=a.port, chips=a.chips, run=run)
    except ControlError as exc:
        print(f"suite-control: {exc}", file=sys.stderr)
        return 2
    argv_g = gozer_argv(prepared, who=a.who, reason=a.reason, gozer=a.gozer)
    print(json.dumps({"control_dir": prepared["control_dir"], "evidence": prepared["evidence"], "suite_control": prepared["wire"],
                      "gozer": argv_g, "run": f"tt-orchard suite-control run --stage-dir {a.stage_dir}"}, indent=1), file=out)
    if a.action == "prepare":
        return 0
    return _run(prepared, a, gozer_call=gozer_call or _gozer, spawn=spawn or _spawn)


if __name__ == "__main__":
    sys.exit(main())
