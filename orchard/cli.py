"""`tt-orchard`: the front door to a model bring-up.

    tt-orchard bringup MODEL [--run-dir DIR] [--dry-run] [--no-fetch] [--accept-credentials-visible]
    tt-orchard status   [MODEL | --run-dir DIR] [--json] [--style S]
    tt-orchard pause | resume | abort   [MODEL | --run-dir DIR]

It is a thin layer over `python3 -m orchard.supervisor`. `bringup` reads config/bringup.toml, checks what
can be checked without a lease (orchard/preflight.py), fetches the model snapshot when it is not local
(orchard/fetch.py), and then calls the supervisor's `run` with the flags the config implies. Nothing
about the run itself changes: a run started this way is an ordinary supervisor run, and the same
directory can be resumed by running the same command again.

What must not happen, and the tests that say so: a blocked preflight never reaches the supervisor or the
download; a dry run starts nothing; a snapshot that is already local is not downloaded; credentials are
accepted only when the operator passes the flag. The name is `tt-orchard`, not `tt`, because `tt` is the
official Tenstorrent CLI.

The supervisor, the download and the outside signals are parameters of `main`, so tests need neither.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from orchard import __version__, bringup_config, fetch, lexicon, orchard_view, preflight, ui

EXIT_OK, EXIT_REFUSED, EXIT_ERROR, EXIT_ABORTED, EXIT_BLOCKED = 0, 2, 3, 4, 5   # the supervisor's own codes
CONFIG_ENV = "ORCHARD_BRINGUP_CONFIG"
CHECKOUT = Path(__file__).resolve().parent.parent


class NoConfig(Exception):
    """No bringup.toml was found, or the one named does not exist."""


def find_config(explicit, env, checkout: Path, home: Path) -> Path:
    """The config file: --config, then $ORCHARD_BRINGUP_CONFIG, then <checkout>/config/bringup.toml, then
    ~/.config/tt-orchard/bringup.toml. A file that was named explicitly and is missing is an error; it is
    never replaced by a default the operator did not ask for."""
    for named, source in ((explicit, "--config"), (env.get(CONFIG_ENV), CONFIG_ENV)):
        if named:
            p = Path(named)
            if not p.is_file():
                raise NoConfig(f"{source} names {p}, which does not exist")
            return p
    candidates = [Path(checkout) / "config" / "bringup.toml", Path(home) / ".config" / "tt-orchard" / "bringup.toml"]
    for p in candidates:
        if p.is_file():
            return p
    raise NoConfig("no bringup.toml found. Looked at: " + ", ".join(str(p) for p in candidates)
                   + f". Copy config/bringup.example.toml to one of them, edit it, or pass --config or set {CONFIG_ENV}")


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help="path to bringup.toml")
    common.add_argument("--style", choices=ui.STYLES, default=argparse.SUPPRESS,
                        help="auto (default): colour and emoji on a capable terminal; pretty; plain")
    p = argparse.ArgumentParser(prog="tt-orchard", parents=[common],
                                description="Bring a new model up on a Tenstorrent machine, unattended.")
    p.add_argument("--version", action="version", version=f"tt-orchard {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bringup", parents=[common], help="check, fetch and start (or resume) a run")
    b.add_argument("model", help="Hugging Face model id, org/name")
    b.add_argument("--run-dir", help="default: <runs_root>/<org>--<name>")
    b.add_argument("--dry-run", action="store_true", help="print the checks and the command; start nothing")
    b.add_argument("--no-fetch", action="store_true", help="never download; the snapshot must be local")
    b.add_argument("--accept-credentials-visible", action="store_true",
                   help="start even though credential files are visible to agent shells (the ledger records it)")
    s = sub.add_parser("status", parents=[common], help="print the state of a run (read-only)")
    s.add_argument("model", nargs="?")
    s.add_argument("--run-dir")
    s.add_argument("--json", action="store_true")
    for word in ("pause", "resume", "abort"):
        c = sub.add_parser(word, parents=[common], help=f"send {word} to a running supervisor")
        c.add_argument("model", nargs="?")
        c.add_argument("--run-dir")
    return p


def _refuse(message: str) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def main(argv=None, *, env=None, stdout=None, signals=None, supervisor_main=None, fetcher=None) -> int:
    env = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    if supervisor_main is None:
        from orchard import supervisor
        supervisor_main = supervisor.main
    fetcher = fetcher or fetch.fetch_snapshot
    args = _parser().parse_args(argv)
    style = ui.detect(out, env, getattr(args, "style", "auto"))
    home = Path(env.get("HOME") or Path.home())

    def config():
        return bringup_config.load(find_config(getattr(args, "config", None), env, CHECKOUT, home)), \
            find_config(getattr(args, "config", None), env, CHECKOUT, home)

    # status and the control words only need a run directory.
    if args.cmd != "bringup":
        run_dir = args.run_dir
        if not run_dir:
            if not args.model:
                return _refuse("give a model id or --run-dir")
            try:
                cfg, _ = config()
                run_dir = str(bringup_config.run_dir(cfg, args.model))
            except (NoConfig, bringup_config.BringupConfigError) as exc:
                return _refuse(str(exc))
        if args.cmd == "status":
            fwd = ["status", "--run-dir", run_dir] + (["--json"] if args.json else [])
            if hasattr(args, "style"):
                fwd += ["--style", args.style]
            return supervisor_main(fwd)
        return supervisor_main(["control", "--run-dir", run_dir, args.cmd])

    try:
        cfg, cfg_path = config()
        run_dir = Path(args.run_dir) if args.run_dir else bringup_config.run_dir(cfg, args.model)
    except (NoConfig, bringup_config.BringupConfigError) as exc:
        return _refuse(str(exc))
    resuming = (run_dir / "ledger.jsonl").exists()
    sig = signals or preflight.default_signals(cfg)
    checks = preflight.run_preflight(cfg, args.model, accept_credentials=args.accept_credentials_visible,
                                     signals=sig, resuming=resuming)
    print(orchard_view.render_preflight(args.model, run_dir, resuming, cfg_path, checks, style), file=out)
    stops = preflight.blocked(checks)
    if stops:
        print(lexicon.refused_line([c.reason for c in stops], style), file=out)
        return EXIT_REFUSED

    hub = checks[0].data
    hf_home = preflight.hf_home_for(cfg)
    local = sig.local_snapshot(args.model)
    if local is not None:
        snapshot = Path(local)
    elif args.no_fetch:
        return _refuse("--no-fetch was given and the snapshot is not in " + str(hf_home))
    else:
        snapshot = fetch.snapshot_dir(hf_home, args.model, hub.sha)      # where the download will put it
    argv_run = bringup_config.supervisor_argv(cfg, args.model, run_dir, inputs={"model": str(snapshot)},
                                              accept_credentials=args.accept_credentials_visible)
    command = "python3 -m orchard.supervisor " + shlex.join(argv_run)
    if args.dry_run:
        print(f"║  command  {command}\n╚══", file=out)
        return EXIT_OK

    if local is None:
        print(f"{style.icon('🌱')}fetching {args.model} at {hub.sha[:8]} ({hub.total_bytes / 1e9:.1f} GB) into "
              f"{hf_home}", file=out)
        try:
            snapshot = Path(fetcher(args.model, hub.sha, hf_home, files=hub.files))
        except fetch.FetchError as exc:
            return _refuse(str(exc))
    code = supervisor_main(argv_run)
    where = run_dir / "stages" / "8" / "bundle"
    if code == EXIT_OK:
        print(lexicon.closing_line("ready-for-operator-review", style, where=str(where)), file=out)
    elif code == EXIT_ABORTED:
        print(lexicon.closing_line("aborted", style, where=str(run_dir)), file=out)
    elif code == EXIT_BLOCKED:
        try:
            reason = json.loads((run_dir / "blocked.json").read_text(encoding="utf-8"))["code"]
        except (OSError, ValueError, KeyError):
            reason = "blocked"
        print(lexicon.frost_line(reason, style) + f" See {run_dir / 'BLOCKED.md'}. Run the same command "
              "again to retry.", file=out)
    elif code == EXIT_ERROR:
        print(f"{style.icon('⛈')}the run stopped on an error and the hardware was released. Run the same "
              "command again to resume.", file=out)
    return code


if __name__ == "__main__":
    sys.exit(main())
