# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard package RUN` and `tt-orchard package-card RUN`: stage 7 on a finished run (issue 22).

A run started without `--package-format` skips stage 7. Before this command, a person ran the same steps
by hand from a script kept with a run log. The command does what that script did and writes each step in
the run's ledger:

    package RUN          stage and scrub every profile, boot-check the required one under a gozer lease,
                         write the cards and the publish text, run the gate, print the publish text
    package RUN --step stage|boot|finish   one part of that (boot is what runs under the lease)
    package-card RUN     put benchmark results, and optionally the authors' reported scores, in the card

Nothing here uploads, publishes or pushes. The publish commands are printed as text. The only program the
command starts is `gozer run`, with one fixed command line (`check_lease_argv`), around the boot check.
A run is changed only when no supervisor holds its ledger, so the ledger chain stays single-writer.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from orchard import package
from orchard.agent import agent_env, spawn_checked
from orchard.kernelprobe import _TICKET, cancel_ticket
from orchard.ledger import Ledger, LedgerLocked, LedgerCorrupt, read_entries
from orchard.package import (BENCHMARK_NUMBERS_FILE, PUBLISH_FILE, REPORTED_FILE, PackageError,
                             read_reported_file)
from orchard.paths import recorded_paths
from orchard.runner import Denied, check_string
from orchard.stages import evidence_record, gate_package

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_QUEUED = 0, 1, 2, 3
GOZER_QUEUED = 10                      # gozer's exit code for a request that was queued
LEASE_WHO = "claude:package"
BOOT_SLACK_S = 600                     # the lease's timeout is the test's own deadline plus this
OUTPUT_KEEP = 65536                    # the tail of `gozer run`'s own output the command keeps, to find a queue ticket
SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|CREDENTIAL|API_?KEY", re.I)
BENCHMARK_COPY = "stages/7/benchmark/benchmark.json"
TASK_LABELS = {"gsm8k": "GSM8K", "ifeval": "IFEval", "humaneval": "HumanEval", "mbpp": "MBPP",
               "bfcl_simple": "BFCL simple", "bfcl_multiple": "BFCL multiple"}


class PackageCommandError(Exception):
    """The command refuses; the message says why."""


def _say(out, text: str = "") -> None:
    print(text, file=out, flush=True)


# ---- the ledger -------------------------------------------------------------------------------------

def _entries(run: Path) -> list[dict]:
    path = run / "ledger.jsonl"
    if not path.is_file():
        raise PackageCommandError(f"{run} has no ledger.jsonl: it is not a run directory")
    try:
        return read_entries(path)
    except LedgerCorrupt as exc:
        raise PackageCommandError(f"the ledger of {run} is damaged: {exc}") from None


def _finished(entries: list[dict]) -> bool:
    ready = any(e["event"] == "decision" and e["data"].get("decision") == "ready for operator review"
                for e in entries)
    aborted = any(e["event"] == "decision" and e["data"].get("decision") == "abort" for e in entries)
    return ready and not aborted


def _open(run: Path) -> Ledger:
    """The ledger writer. A supervisor that is still running holds it, and then nothing is touched."""
    try:
        return Ledger(run / "ledger.jsonl")
    except LedgerLocked:
        raise PackageCommandError(f"a supervisor holds {run / 'ledger.jsonl'}: the run is still going. "
                                  "Stop it first, or wait for it to finish") from None


def _root(entries: list[dict], args, attr: str, key: str, flag: str) -> Path:
    value = getattr(args, attr, None) or (recorded_paths(entries) or {}).get(key)
    if not value:
        raise PackageCommandError(f"the run did not record its paths, so give {flag}")
    return Path(value)


def _models_root(entries: list[dict], args) -> Path:
    """The models root: the flag, else what the run recorded. Staging needs it."""
    return _root(entries, args, "models_root", "tt_model_root", "--models-root")


def _cache_root(entries: list[dict], args) -> Path:
    """The cache root: the flag, else what the run recorded. The boot step needs only this one, and `lease_argv`
    passes only this one to the child."""
    return _root(entries, args, "cache_root", "cache_root", "--cache-root")


def _paths(entries: list[dict], args) -> tuple[Path, Path]:
    """(models root, cache root), for the steps that need both. One message names every flag that is missing."""
    missing = []
    for fn in (_models_root, _cache_root):
        try:
            fn(entries, args)
        except PackageCommandError as exc:
            missing.append(str(exc).rsplit("give ", 1)[1])
    if missing:
        raise PackageCommandError("the run did not record its paths, so give " + " and ".join(missing))
    return _models_root(entries, args), _cache_root(entries, args)


# ---- the lease ---------------------------------------------------------------------------------------

def lease_argv(run: Path, *, chips: int, deadline_s: float, python: str, cache_root: Path, gozer: str = "gozer",
               exact: int | None = None) -> list[str]:
    argv = [gozer, "run", "--chips", str(chips)]
    if exact is not None:
        argv += ["--exact", str(exact)]
    argv += ["--who", LEASE_WHO, "--reason", f"stage 7 boot check of the package for {run.name}", "--",
             "timeout", str(int(deadline_s + BOOT_SLACK_S)), python, "-m", "orchard.cli", "package", str(run),
             "--step", "boot", "--cache-root", str(cache_root)]
    return argv


def check_lease_argv(argv: list[str], *, run_dir, python: str, gozer: str = "gozer") -> None:
    """Refuse any command line that is not `gozer run [--chips N] [--exact N] --who W --reason R --
    timeout S python -m orchard.cli package RUN --step boot [--cache-root P]`. The command starts nothing
    else, so a changed argument list (another gozer verb, an upload) never runs."""
    argv = [str(a) for a in argv]
    if "--" not in argv:
        raise PackageCommandError("the lease command has no `--`")
    i = argv.index("--")
    head, tail = argv[:i], argv[i + 1:]
    if head[:2] != [gozer, "run"]:
        raise PackageCommandError(f"the command runs only `{gozer} run`; this one starts with {head[:2]}")
    j = 2
    seen = set()
    while j < len(head):
        if head[j] in ("--chips", "--exact", "--who", "--reason") and j + 1 < len(head) and head[j] not in seen:
            seen.add(head[j])
            j += 2
        else:
            raise PackageCommandError(f"the lease command has an argument it does not pass: {head[j]!r}")
    if len(tail) < 2 or tail[0] != "timeout" or not re.fullmatch(r"\d+", tail[1]):
        raise PackageCommandError("the lease command must run `timeout SECONDS python -m orchard.cli package ...`")
    rest = tail[2:]
    base = [python, "-m", "orchard.cli", "package", str(run_dir), "--step", "boot"]
    extra = rest[len(base):]
    if rest[:len(base)] != base or (extra and not (len(extra) == 2 and extra[0] == "--cache-root")):
        raise PackageCommandError("the lease may run only the package boot step; this one runs " + " ".join(rest))


def _scrubbed(env) -> dict:
    """The operator's environment for `gozer run`, without anything that looks like a credential. The
    boot step builds its own, even smaller, environment."""
    return {k: v for k, v in dict(env).items() if not SECRET.search(k)}


CHECKOUT = str(Path(__file__).resolve().parent.parent)     # the directory that holds the `orchard` package


def _with_checkout(env: dict) -> dict:
    """The lease runs `python -m orchard.cli` from wherever the operator is, so the checkout goes first on the
    import path."""
    out = dict(env)
    out["PYTHONPATH"] = os.pathsep.join([CHECKOUT] + ([out["PYTHONPATH"]] if out.get("PYTHONPATH") else []))
    return out


def _default_runner(argv, env) -> tuple[int, str]:
    """Run the lease command. Returns (exit code, the tail of what it printed). The operator sees the output as it
    comes; the command keeps the tail so a queue ticket in it can be cancelled."""
    proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")
    kept = ""
    for line in proc.stdout:
        print(line, end="", flush=True)
        kept = (kept + line)[-OUTPUT_KEEP:]
    return proc.wait(), kept


def _queued_text(text: str, cancel) -> tuple[dict, str]:
    """gozer queued the request (exit 10): cancel its ticket, or an abandoned ticket is granted later to a process
    that is not there and holds the chips until it expires. Returns the ledger fields and the sentence for the
    operator."""
    found = _TICKET.search(text or "")
    if found is None:
        return ({"ticket": None, "cancel_error": "no ticket in gozer's output, so nothing was cancelled"},
                "There was no ticket in gozer's output, so nothing was cancelled: look with `gozer status`")
    ticket = found.group(1)
    error = cancel(ticket)
    if error is None:
        return {"ticket": ticket, "cancel_error": None}, f"The ticket {ticket} was cancelled."
    return ({"ticket": ticket, "cancel_error": error},
            f"The ticket {ticket} could not be cancelled: {error}. Look with `gozer status`")


# ---- steps ---------------------------------------------------------------------------------------------

def _stage(run: Path, led: Ledger, args, entries, out, *, extra_env, hostname) -> Path:
    stage_dir = run / "stages" / "7"
    if not args.namespace:
        raise PackageCommandError("--namespace is needed to stage: the Hugging Face organisation the package "
                                  "will be published under")
    models, _ = _paths(entries, args)
    if stage_dir.exists():
        if not args.restage:
            raise PackageCommandError(f"{stage_dir} already exists. Use --restage to move it aside and stage "
                                      "again, or run the remaining steps with --step boot or --step finish")
        aside = run / "stages" / f"7.old-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        stage_dir.rename(aside)
        _say(out, f"moved the earlier stage 7 to {aside}")
    led.append("decision", 7, decision="package command", step=args.step, namespace=args.namespace,
               models_root=str(models), restage=bool(args.restage))
    stage_dir.mkdir(parents=True)
    try:
        staged = package.stage_all(run, stage_dir, namespace=args.namespace, models_root=models,
                                   env=agent_env(run, extra=extra_env), hostname=hostname)
    except PackageError as exc:
        raise PackageCommandError(f"packaging: {exc}") from None
    names = [p["name"] for p in staged["profiles"]]
    led.append("evidence", 7, what="package staged", profiles=names,
               **evidence_record(run, stage_dir / "package.json"))
    _say(out, f"staged: {', '.join(names)}"
              + (f" (skipped: {len(staged['skipped_profiles'])})" if staged["skipped_profiles"] else ""))
    return stage_dir


def _test(stage_dir: Path) -> dict:
    try:
        test = json.loads((stage_dir / "hw_test.json").read_text(encoding="utf-8"))
        command, deadline = test["command"], float(test["deadline_s"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PackageCommandError(f"{stage_dir}/hw_test.json is missing or wrong: {exc}") from None
    if not isinstance(command, str) or not command.strip() or deadline <= 0:
        raise PackageCommandError(f"{stage_dir}/hw_test.json needs a command and a positive deadline_s")
    return {"command": command, "deadline_s": deadline}


def _refuse_command(command: str, run: Path) -> None:
    try:
        check_string(command, run)
    except Denied as exc:
        raise PackageCommandError(f"the boot check's command is refused: {exc}") from None


def boot_step(run: Path, args, out, *, env, spawn, extra_env) -> int:
    """The boot check itself. It runs under a lease: gozer sets TT_VISIBLE_DEVICES for its child."""
    stage_dir = run / "stages" / "7"
    test = _test(stage_dir)
    _refuse_command(test["command"], run)
    chips = (env.get("TT_VISIBLE_DEVICES") or "").strip()
    if not chips:
        raise PackageCommandError("the boot step opens the chips, so it runs under a lease: use "
                                  "`tt-orchard package RUN` (it takes one), or `gozer run --chips N -- ...`")
    cache = _cache_root(_entries(run), args)
    benv = agent_env(run, extra=extra_env)
    benv["TT_VISIBLE_DEVICES"] = chips
    benv.pop("TT_METAL_VISIBLE_DEVICES", None)
    benv["TT_METAL_CACHE"] = str(cache / "kernels")
    (cache / "kernels").mkdir(parents=True, exist_ok=True)
    out_path = stage_dir / "evidence" / "hw-test-output.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "wb") as log:
        code, timed_out = spawn(test["command"], run, benv, test["deadline_s"], log)
    result = {"command": test["command"], "returncode": code, "timed_out": timed_out,
              "seconds": round(time.time() - t0, 3), "chips": chips.split(","),
              "output": evidence_record(run, out_path)}
    tmp = stage_dir / "test-result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp, stage_dir / "test-result.json")
    _say(out, f"boot check: exit {code}, timed out {timed_out}, {result['seconds']} s")
    return EXIT_OK if code == 0 and not timed_out else EXIT_FAILED


def _finish(run: Path, led: Ledger, stage_dir: Path, out, *, hostname) -> int:
    result = stage_dir / "test-result.json"
    if result.is_file():
        led.append("evidence", 7, what="package boot check", **evidence_record(run, result))
    try:
        package.finish(run, stage_dir, hostname=hostname)
    except PackageError as exc:
        raise PackageCommandError(f"packaging: {exc}") from None
    gate = gate_package(stage_dir, run)
    led.append("decision", 7, decision="package gate", ok=gate.ok, reasons=list(gate.reasons))
    if not gate.ok:
        _say(out, "gate: not ok")
        for reason in gate.reasons:
            _say(out, f"  - {reason}")
        return EXIT_FAILED
    _say(out, "gate: ok")
    publish = stage_dir / PUBLISH_FILE
    _say(out)
    _say(out, f"Publish commands (text only; nothing was run, uploaded or published). From {publish}:")
    _say(out, publish.read_text(encoding="utf-8").rstrip())
    _say(out)
    _say(out, "The operator bundle in stages/8 was built before this package existed. Build it again "
              "(stages/8/build_bundle.py) if the package should appear in it.")
    return EXIT_OK


def main(args, *, out=None, env=None, runner=None, spawn=None, extra_env=None, hostname=None,
         cancel=None) -> int:
    """`tt-orchard package`. Returns 0 (done and the gate is ok), 1 (a step failed or the gate is not ok),
    2 (refused before anything changed) or 3 (the boards were busy and gozer queued the request)."""
    out = out or sys.stdout
    env = os.environ if env is None else env
    runner = runner or _default_runner
    spawn = spawn or spawn_checked
    hostname = socket.gethostname() if hostname is None else hostname
    run = Path(args.run_dir).resolve()
    try:
        entries = _entries(run)
        if args.step == "boot":
            return boot_step(run, args, out, env=env, spawn=spawn, extra_env=extra_env)
        if not _finished(entries):
            raise PackageCommandError(f"the run in {run} has not finished (its ledger has no 'ready for operator "
                                      "review'). Packaging a run that is still going could change files under it")
        with _open(run) as led:
            entries = led.read()
            if args.step in ("all", "stage"):
                stage_dir = _stage(run, led, args, entries, out, extra_env=extra_env, hostname=hostname)
            else:
                stage_dir = run / "stages" / "7"
                if not (stage_dir / "package.json").is_file():
                    raise PackageCommandError(f"nothing is staged in {stage_dir}: run `package RUN --step stage` "
                                              "first")
                led.append("decision", 7, decision="package command", step=args.step, namespace=args.namespace)
            if args.step == "stage":
                return EXIT_OK
            if args.step == "all":
                test = _test(stage_dir)
                _refuse_command(test["command"], run)
                package_json = json.loads((stage_dir / "package.json").read_text(encoding="utf-8"))
                chips = next(p["chips"] for p in package_json["profiles"] if p["required"])
                _, cache = _paths(entries, args)
                argv = lease_argv(run, chips=chips, deadline_s=test["deadline_s"], python=args.python,
                                  cache_root=cache, gozer=args.gozer, exact=args.exact)
                check_lease_argv(argv, run_dir=run, python=args.python, gozer=args.gozer)
                _say(out, "boot check: taking a lease on " + f"{chips} chips (this can take 20 to 30 minutes)")
                got = runner(argv, _with_checkout(_scrubbed(env)))
                code, printed = got if isinstance(got, tuple) else (got, "")
                if code == GOZER_QUEUED:
                    fields, sentence = _queued_text(printed, cancel or cancel_ticket)
                    led.append("decision", 7, decision="package command", step="boot", outcome="queued", **fields)
                    _say(out, "gozer queued the lease request: the boards are busy. Nothing was finished. "
                              + sentence + " When they are free, run the same command again with --restage")
                    return EXIT_QUEUED
                if not (stage_dir / "test-result.json").is_file():
                    _say(out, f"the boot check did not run to a result (gozer exited {code}). Nothing was "
                              "finished")
                    return EXIT_FAILED
            return _finish(run, led, stage_dir, out, hostname=hostname)
    except PackageCommandError as exc:
        _say(out, f"package: {exc}")
        return EXIT_REFUSED


# ---- the card step -----------------------------------------------------------------------------------------

def _score(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def benchmark_rows(bench: dict, model_key: str, base_key: str | None) -> list[dict]:
    """The card's measured rows from a benchmark file: `{"models": {key: id}, "tasks": {task: {key: {n, score,
    cut_off?}, "paired": {sign_test_p?}}}}`, the shape the bytkim benchmark compile scripts wrote. Raises
    PackageCommandError naming what is wrong."""
    tasks = bench.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise PackageCommandError("the benchmark file needs `tasks`: an object of tasks")
    keys = sorted({k for t in tasks.values() if isinstance(t, dict) for k in t if k != "paired"}
                  | set((bench.get("models") or {}) if isinstance(bench.get("models"), dict) else ()))
    if not any(isinstance(t, dict) and model_key in t for t in tasks.values()):
        raise PackageCommandError(f"the benchmark file has no results for {model_key!r}; its keys are "
                                  f"{', '.join(keys)}")
    rows = []
    for task, entry in tasks.items():
        label = TASK_LABELS.get(task, task)
        mine = entry.get(model_key) if isinstance(entry, dict) else None
        if not isinstance(mine, dict):
            raise PackageCommandError(f"task {task!r} has no results for {model_key!r}")
        if base_key and not isinstance(entry.get(base_key), dict):
            # A requested control that is not here is a typo or a different file. Skipping it would still leave the
            # paired p-value in the card, which claims a comparison no row shows.
            raise PackageCommandError(f"task {task!r} has no results for the control {base_key!r}; the keys in the "
                                      f"file are {', '.join(keys)}. Leave out --base-key for a card without a comparison")
        for who, key, suffix in ((mine, model_key, "this model"),
                                 (entry.get(base_key) if base_key else None, base_key,
                                  "nearest model, same conditions")):
            if who is None:
                continue
            if not isinstance(who, dict) or not _score(who.get("score")):
                raise PackageCommandError(f"task {task!r}, {key!r}: `score` must be a number")
            rows.append({"name": f"{label} score, {suffix}", "value": who["score"], "unit": "fraction"})
            if _score(who.get("cut_off")):
                rows.append({"name": f"{label} replies cut off at the token limit, {suffix}",
                             "value": who["cut_off"], "unit": "items"})
        paired = entry.get("paired")
        if isinstance(paired, dict) and _score(paired.get("sign_test_p")) and base_key:
            rows.append({"name": f"{label} paired sign test p, this model against the nearest model",
                         "value": paired["sign_test_p"], "unit": "p-value"})
    for r in rows:
        r["evidence"] = [BENCHMARK_COPY]
    return rows


def _replace_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def card_main(args, *, out=None, extra_env=None, hostname=None) -> int:
    """`tt-orchard package-card`: benchmark results (and the authors' reported scores) into the cards, then
    the cards are written again, scrubbed again and gated. Returns 0, 1 (the gate is not ok) or 2 (refused)."""
    out = out or sys.stdout
    hostname = socket.gethostname() if hostname is None else hostname
    run = Path(args.run_dir).resolve()
    stage_dir = run / "stages" / "7"
    try:
        entries = _entries(run)
        if not ((stage_dir / "package.json").is_file() and (stage_dir / "test-result.json").is_file()):
            raise PackageCommandError(f"{stage_dir} has no staged and boot-checked package: run "
                                      "`tt-orchard package RUN` first")
        try:
            bench_text = Path(args.benchmark).read_text(encoding="utf-8")
            bench = json.loads(bench_text)
        except (OSError, ValueError) as exc:
            raise PackageCommandError(f"cannot read the benchmark file {args.benchmark}: {exc}") from None
        if not isinstance(bench, dict):
            raise PackageCommandError("the benchmark file must be a JSON object")
        rows = benchmark_rows(bench, args.model_key, args.base_key)
        reported_text = None
        installed = run / REPORTED_FILE
        if args.reported:
            try:
                read_reported_file(args.reported)
                reported_text = Path(args.reported).read_text(encoding="utf-8")
            except (PackageError, OSError) as exc:
                raise PackageCommandError(f"the reported scores file is refused: {exc}") from None
            if installed.exists() and installed.read_text(encoding="utf-8") != reported_text \
                    and not args.replace_reported:
                raise PackageCommandError(f"{installed} already holds other reported scores. Pass "
                                          "--replace-reported to replace them")
        with _open(run) as led:
            _replace_file(run / BENCHMARK_COPY, bench_text)
            _replace_file(run / BENCHMARK_NUMBERS_FILE, json.dumps({"rows": rows}, indent=2) + "\n")
            if reported_text is not None:
                _replace_file(installed, reported_text)
            led.append("decision", 7, decision="package card", benchmark=BENCHMARK_COPY,
                       model_key=args.model_key, base_key=args.base_key, rows=len(rows),
                       reported=bool(args.reported), sha256=evidence_record(run, run / BENCHMARK_COPY)["sha256"])
            _say(out, f"{len(rows)} measured rows from {Path(args.benchmark).name}"
                      + ("; reported scores installed" if reported_text is not None else ""))
            return _finish(run, led, stage_dir, out, hostname=hostname)
    except PackageCommandError as exc:
        _say(out, f"package-card: {exc}")
        return EXIT_REFUSED
