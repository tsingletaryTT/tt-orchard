# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard telegraph check | ask | report | usage` (orchard/telegraph.py).

`check` shows what is configured and whether the key is there (never the key). It makes no request unless `--ping`
is given, which asks the endpoint for its model list: that request spends no tokens. `ask` sends one question by hand.
`report` records whether an answer helped, and `usage` shows what the help has cost.

The `[telegraph]` table is read from `--telegraph-config FILE`, else from the bringup config (`--config`,
$ORCHARD_BRINGUP_CONFIG, config/bringup.toml). Exit codes: 0 done, 2 a problem with the setup or the arguments, 3 the
ask got no answer (the reason is printed).

The state directory (it holds the ledger database, the reply files and the cache) is chosen in this order: `--dir`, the
`[telegraph] dir` of the config, `$ORCHARD_TELEGRAPH_DIR` (an absolute path; the supervisor sets it for agent shells), and only
then `~/.cache/tt-orchard/telegraph` of the caller's home. An agent shell has a HOME inside the run directory, so it never takes
the last choice: with no configured directory it stops and says so, and no second ledger is opened whose caps nobody shares.
"""
from __future__ import annotations

import json
import os
import shlex
import stat
import tomllib
from pathlib import Path

from orchard import telegraph as tg
from orchard import telegraph_notice as tn

EXIT_OK, EXIT_PROBLEM, EXIT_NO_ANSWER = 0, 2, 3
ENV_DIR = "ORCHARD_TELEGRAPH_DIR"           # set by the supervisor for agent shells (orchard/agent.py)
AGENT_SHELL_MARK = "ORCHARD_RUN_DIR"        # present in an agent shell's environment (orchard/agent.py)


def default_dir(home) -> Path:
    return Path(home) / ".cache" / "tt-orchard" / "telegraph"


def resolve_directory(explicit, cfg: tg.Config, environ, home) -> Path:
    """The ledger directory: the explicit one, the config's, `$ORCHARD_TELEGRAPH_DIR`, then the caller's home. Never the home of an
    agent shell (issue 133): its HOME is inside the run directory, and a ledger there would not share the operator's caps."""
    if explicit:
        return Path(explicit)
    if cfg.dir:
        return Path(cfg.dir)
    from_env = environ.get(ENV_DIR)
    if from_env:
        path = Path(from_env)
        if not path.is_absolute():
            raise tg.ConfigError(f"{ENV_DIR} must be an absolute path")
        return path
    if environ.get(AGENT_SHELL_MARK):
        raise tg.ConfigError(f"this is an agent shell (HOME is inside the run directory) and no ledger directory is configured: "
                             f"set [telegraph] dir, ${ENV_DIR}, or pass --dir, so the caps are shared with the operator's ledger")
    return default_dir(home)


def load_config(path, _unused=None) -> tg.Config:
    """The `[telegraph]` table of a TOML file as a Config (off when the file has none)."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise tg.ConfigError(f"cannot read {str(path)!r}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise tg.ConfigError(f"cannot parse {str(path)!r}: {exc}") from exc
    return tg.load(raw.get("telegraph"), base=path.resolve().parent)


def _config_path(args, environ, home) -> Path:
    from orchard import cli
    explicit = getattr(args, "telegraph_config", None)
    if explicit:
        return Path(explicit)
    return cli.find_config(getattr(args, "config", None), environ, cli.CHECKOUT, Path(home))


def _read(path: str, what: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise OSError(f"cannot read the {what} file {path!r}: {exc.__class__.__name__}") from exc


def main(args, *, out, environ=None, transport=None, getter=None, home=None) -> int:
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)

    def say(text=""):
        print(text, file=out)

    try:
        cfg = load_config(_config_path(args, environ, home))
    except (tg.ConfigError, Exception) as exc:                      # a missing config file is the same kind of problem
        say(f"telegraph: {exc}")
        return EXIT_PROBLEM
    try:
        directory = resolve_directory(getattr(args, "dir", None), cfg, environ, home)
    except tg.ConfigError as exc:
        say(f"telegraph: {exc}")
        return EXIT_PROBLEM
    tel = tg.Telegraph(cfg, directory, transport=transport, environ=environ, home=str(home))
    action = args.action
    flags = tn.flags_from_args(args)                       # the commands of a notice act on the directory this command used
    if action == "check":
        return _check(cfg, tel, args, say, getter)
    if action == "usage":
        return _usage(directory, args, say, flags)
    if action == "report":
        waiting = tn.state_of(directory, flags=flags)
        if waiting is not None and waiting.blocks_opening:        # before the ledger is opened: opening creates files here
            say("telegraph: " + waiting.text())
            say("The verdict was not written.")
            return EXIT_PROBLEM
        try:
            tel.report(args.call_id, helped=bool(args.helped))
        except KeyError:
            say(f"telegraph: no call {args.call_id} in {directory}")
            return EXIT_PROBLEM
        except tg.LEDGER_ERRORS as exc:
            say(f"telegraph: the verdict could not be written to {directory} ({exc})")
            return EXIT_PROBLEM
        say(f"recorded: call {args.call_id} {'helped' if args.helped else 'did not help'}")
        return EXIT_OK
    return _ask(tel, args, say, flags)


def _check(cfg: tg.Config, tel: tg.Telegraph, args, say, getter) -> int:
    say(f"enabled: {'yes' if cfg.enabled else 'no'}")
    if not cfg.enabled:
        say("telegraph is off; the harness runs without it. Add a [telegraph] table with enabled = true to use it.")
        return EXIT_OK
    problems = 0
    say(f"endpoint: {cfg.endpoint} ({cfg.api} style)")
    say(f"models: {', '.join(cfg.models)} (cheapest first)")
    say(f"asks after {cfg.after_failures} repeated failures; caps: {cfg.max_calls} calls, {cfg.max_total_tokens} tokens"
        + (f", ${cfg.max_cost_usd}" if cfg.max_cost_usd is not None else ", no cost cap")
        + f"; {cfg.max_output_tokens} output tokens and {cfg.max_prompt_chars} prompt characters per call")
    source = (f"variable {cfg.key_env}" if cfg.key_env else f"file {cfg.key_file}" if cfg.key_file else "none (keyless)")
    key, why = tel._key()
    say(f"key source: {source}")
    if why:
        say(f"key: MISSING ({why})")
        problems += 1
    else:
        say("key: present" if key else "key: not needed")
    if cfg.key_file and why is None:
        try:
            if stat.S_IMODE(cfg.key_file.stat().st_mode) & 0o077:
                say("warning: the key file is readable by other users; run chmod 600 on it")
        except OSError:
            pass
    if args.ping:
        problems += _ping(cfg, key, say, getter)
    return EXIT_PROBLEM if problems else EXIT_OK


def _ping(cfg: tg.Config, key, say, getter) -> int:
    getter = getter or tg.http_get
    base = cfg.endpoint[:-3] if cfg.api == "anthropic" and cfg.endpoint.endswith("/v1") else cfg.endpoint
    url = (base + "/v1/models") if cfg.api == "anthropic" and not cfg.endpoint.endswith("/v1") else cfg.endpoint + "/models"
    headers = {}
    if cfg.api == "anthropic":
        headers["anthropic-version"] = tg.ANTHROPIC_VERSION
        if key:
            headers["x-api-key"] = key
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        status, raw = getter(url, headers, cfg.timeout_s)
        if status != 200:
            raise ValueError(f"HTTP {status}")
        listed = {m["id"] for m in json.loads(raw)["data"] if isinstance(m, dict)}
    except Exception as exc:                                        # noqa: BLE001
        reason = tg.redact(f"{type(exc).__name__}: {exc}", secrets=(key or "",))
        say(f"ping failed: {reason}")
        return 1
    missing = 0
    for model in cfg.models:
        if model in listed:
            say(f"  {model}: listed")
        else:
            say(f"  {model}: NOT listed")
            missing += 1
    return 1 if missing else 0


def _report_command(tel: tg.Telegraph, args, call_id: str) -> str:
    """The command that reports this answer: it carries the directory and any explicit config the ask used."""
    parts = ["tt-orchard", "telegraph", "report", call_id, "--helped|--not-helped", "--dir", shlex.quote(str(tel.dir))]
    for flag, value in (("--telegraph-config", getattr(args, "telegraph_config", None)),
                        ("--config", getattr(args, "config", None))):
        if value:
            parts += [flag, shlex.quote(str(value))]
    return " ".join(parts)


def _ask(tel: tg.Telegraph, args, say, flags: str = "") -> int:
    try:
        task, failure = _read(args.task_file, "task"), _read(args.failure_file, "failure")
    except OSError as exc:
        say(f"telegraph: {exc}")
        return EXIT_PROBLEM
    waiting = tn.state_of(tel.dir, flags=flags) if tel.cfg.enabled else None     # a disabled telegraph says so below, and nothing else
    if waiting is not None and waiting.blocks_opening:
        say("telegraph: " + waiting.text())
        say("No outside call was made and nothing was spent.")
        return EXIT_NO_ANSWER
    got = tel.ask(args.problem, task, failure)
    if not got.ok:
        say(f"no answer: {got.reason}")
        return EXIT_NO_ANSWER
    say(got.as_hint())
    where = "cache" if got.cached else got.model
    line = (f"call {got.call_id}: {where}, {got.tokens_in} in, {got.tokens_out} out, {got.latency_s:.1f}s"
            + (f", ${got.cost_usd:.6f}" if got.cost_usd is not None else ", cost unknown"))
    if got.reason:                                       # the answer arrived but its record could not be written
        say(f"warning: {got.reason}")
        say(line + ".")
    else:
        say(line + f". Tell the ledger whether it helped: {_report_command(tel, args, got.call_id)}")
    return EXIT_OK


def _usage(directory: Path, args, say, flags: str = "") -> int:
    try:
        s = tg.summarise_records(tg._read_ledger_strict(directory))
    except tg.LEDGER_ERRORS as exc:                                  # an unreadable ledger is not an empty one
        waiting = tn.state_of(directory, flags=flags)
        if waiting is not None:                                      # the first line says what to do
            say("telegraph: " + waiting.text())
            say("The spend is unknown until then.")
            return EXIT_PROBLEM
        say(f"telegraph: the ledger in {directory} cannot be read ({exc}); the spend is unknown")
        return EXIT_PROBLEM
    if args.json:
        say(json.dumps(s, indent=2, sort_keys=True))
        return EXIT_OK
    say(f"calls: {s['calls']} ({s['ok']} answered), cache hits: {s['cache_hits']}, refused: {s['refused']}")
    say(f"helped: {s['helped']}, did not help: {s['not_helped']}, not yet reported: {s['unreported']}")
    if s["unsettled"] or s["unknown_usage_calls"]:
        say(f"not settled: {s['unsettled']} reservations with no result (a crash or a kill), "
            f"{s['unknown_usage_calls']} answers with no usable token count; their worst case counts against the caps")
    say(f"tokens: {s['tokens_in']} in, {s['tokens_out']} out; time: {s['latency_s']:.1f}s")
    say(f"cost: ${s['cost_usd']:.6f}" + (f" (+ {s['cost_unknown_calls']} calls with no known cost: no price, or the provider sent no usage)" if s["cost_unknown_calls"] else ""))
    for model, v in sorted(s["by_model"].items()):
        say(f"  model {model}: {v['calls']} calls")
    for problem, v in sorted(s["by_problem"].items()):
        say(f"  problem {problem}: {v['calls']} calls, {v['helped']} helped")
    return EXIT_OK
