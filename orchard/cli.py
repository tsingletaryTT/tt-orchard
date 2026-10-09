# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard`: the front door to a model bring-up.

    tt-orchard bringup MODEL [--run-dir DIR] [--dry-run] [--no-fetch] [--refuse-credentials-visible]
    tt-orchard status   [MODEL | --run-dir DIR] [--json] [--style S]
    tt-orchard watch    [MODEL | --run-dir DIR] [--all] [--once] [--style S]
    tt-orchard pause | resume | abort   [MODEL | --run-dir DIR]
    tt-orchard lab setup [--model ORG/NAME]... [--yes] [--check]

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
import time
from pathlib import Path

from orchard import __version__, bringup_config, fetch, lexicon, narrate, nearest, orchard_view, preflight, ui

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
                   help="accept that credential files are visible to agent shells (this is the default; the "
                        "flag is kept so older commands still work)")
    b.add_argument("--refuse-credentials-visible", action="store_true",
                   help="block the run while credential files are visible to agent shells")
    b.add_argument("--base", help="the model to base the run on: a tt-model bundle (as `tt model search` lists "
                                  "it) or a Hugging Face model id. Default: the model card's base_model")
    b.add_argument("--quiet", action="store_true", help="do not print what the run is doing as it goes")
    w = sub.add_parser("watch", parents=[common], help="follow a run: what each role is doing, live (read-only)")
    w.add_argument("model", nargs="?")
    w.add_argument("--run-dir")
    w.add_argument("--all", action="store_true", help="start from the first entry, not the last 15")
    w.add_argument("--once", action="store_true", help="print the recent history and stop")
    s = sub.add_parser("status", parents=[common], help="print the state of a run (read-only)")
    s.add_argument("model", nargs="?")
    s.add_argument("--run-dir")
    s.add_argument("--json", action="store_true")
    for word in ("pause", "resume", "abort"):
        c = sub.add_parser(word, parents=[common], help=f"send {word} to a running supervisor")
        c.add_argument("model", nargs="?")
        c.add_argument("--run-dir")
    lab = sub.add_parser("lab", parents=[common], help="get the [lab] box ready for runs whose hardware tests "
                                                       "run there (`lab setup --help`)")
    lab.add_argument("action", choices=["setup"])
    lab.add_argument("rest", nargs=argparse.REMAINDER, help="options for the action")
    return p


def _refuse(message: str) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _installed(sig) -> list:
    return list(sig.installed_bundles()) if sig.installed_bundles else []


def resolve_base(arg: str, sig, puller, out, style, *, dry_run: bool = False) -> tuple[str | None, str | None]:
    """(the weights repo to base the run on, an error text). `arg` is an installed bundle, a model id that
    an installed bundle serves, a published bundle (installed here with `tt-model pull`), or any other
    model id, which the base check then judges."""
    inst = _installed(sig)
    for b in inst:
        if b.name == arg:
            return b.weights_repo, None
    if any(b.weights_repo == arg for b in inst) or dry_run:
        return arg, None
    found = sig.search_bundles(nearest.family_query(arg)) if sig.search_bundles else None
    if arg not in {c["name"] for c in found or []}:
        return arg, None                                  # a model id
    print(f"{style.icon('📦')}installing the bundle {arg} with `tt-model pull` (this can take a while)", file=out)
    ok, why = puller(arg)
    if not ok:
        return None, f"`tt-model pull {arg}` failed: {why}"
    for b in _installed(sig):
        if b.name == arg:
            return b.weights_repo, None
    return None, f"{arg} was pulled but is not among the installed bundles tt-orchard looks at"


def pick_base(candidates: list[dict], model: str, base: str, *, out, ask=input) -> str | None:
    """Ask the operator which bundle to base the run on. Returns its name, or None for none."""
    print(f"║  {model} is based on {base}, and no installed bundle serves it. Bundles found:", file=out)
    for i, c in enumerate(candidates, 1):
        tag = "installed" if c.get("installed") else "not installed (will be installed with tt-model pull)"
        print(f"║    {i}. {c['name']}  [{tag}]", file=out)
    print("║    0. none of these; stop", file=out)
    while True:
        try:
            answer = ask("║  choose a number: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer in ("", "0"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            return candidates[int(answer) - 1]["name"]
        print("║  not a choice", file=out)


def _base_help(check, model: str, out) -> None:
    cands = (check.data or {}).get("candidates") or []
    print("╔══ How to choose a base", file=out)
    if cands:
        for c in cands:
            print(f"║  tt-orchard bringup {model} --base {c['name']}"
                  + ("" if c.get("installed") else "    (installs it first)"), file=out)
    print("║  or install a bundle for the base yourself (`tt-model pull <bundle>`) and run bringup again", file=out)
    print("╚══", file=out)


def _print_block_digest(run_dir: Path, code: str, model: str, out) -> None:
    """What was tried and what to do, under a left bar, after a run ends blocked."""
    try:
        stage = json.loads((run_dir / "blocked.json").read_text(encoding="utf-8")).get("stage")
    except (OSError, ValueError):
        stage = None
    print("╔══ What was tried" + (f" in stage {stage}" if stage is not None else ""), file=out)
    for line in (narrate.what_was_tried(run_dir, stage=stage) if stage is not None else ["Nothing was recorded."]):
        print(f"║  {line}", file=out)
    print("╠══ How to unblock", file=out)
    for line in narrate.how_to_unblock(code, model):
        print(f"║  - {line}", file=out)
    print("╚══", file=out)


def _watch(run_dir: Path, args, style, out) -> int:
    """Follow a run from another terminal. Read-only: it never opens the ledger writer."""
    if not (run_dir / "ledger.jsonl").exists():
        return _refuse(f"no run at {run_dir} (there is no ledger there yet)")
    n = narrate.Narrator(run_dir, style, out, replay=True if args.all else 15)
    n.poll()
    try:
        while not args.once and not n.finished:
            time.sleep(narrate.POLL_S)
            n.poll()
    except KeyboardInterrupt:
        pass
    return EXIT_OK


def main(argv=None, *, env=None, stdout=None, signals=None, supervisor_main=None, fetcher=None,
         chooser=None, puller=None) -> int:
    env = os.environ if env is None else env
    out = sys.stdout if stdout is None else stdout
    if supervisor_main is None:
        from orchard import supervisor
        supervisor_main = supervisor.main
    fetcher = fetcher or fetch.fetch_snapshot
    puller = puller or nearest.pull
    args = _parser().parse_args(argv)
    style = ui.detect(out, env, getattr(args, "style", "auto"))
    home = Path(env.get("HOME") or Path.home())

    def config():
        return bringup_config.load(find_config(getattr(args, "config", None), env, CHECKOUT, home)), \
            find_config(getattr(args, "config", None), env, CHECKOUT, home)

    if args.cmd == "lab":
        from orchard import lab_setup
        try:
            cfg, _ = config()
        except (NoConfig, bringup_config.BringupConfigError) as exc:
            return _refuse(str(exc))
        return lab_setup.main(args.rest, cfg=cfg)

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
        if args.cmd == "watch":
            return _watch(Path(run_dir), args, style, out)
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
    override = None
    if args.base:
        override, why = resolve_base(args.base, sig, puller, out, style, dry_run=args.dry_run)
        if why:
            return _refuse(why)

    def preflight_page():
        got = preflight.run_preflight(cfg, args.model, accept_credentials=not args.refuse_credentials_visible,
                                      signals=sig, resuming=resuming, base_override=override)
        print(orchard_view.render_preflight(args.model, run_dir, resuming, cfg_path, got, style), file=out)
        return got

    checks = preflight_page()
    base_check = next(c for c in checks if c.name == "base")
    if (base_check.status == preflight.BLOCK and (base_check.data or {}).get("candidates") and not args.base
            and not args.dry_run):
        if chooser is None and sys.stdin.isatty() and ui.stream_is_tty(out):
            chooser = lambda c, m, b: pick_base(c, m, b, out=out)   # noqa: E731
        pick = chooser(base_check.data["candidates"], args.model, base_check.data["base"]) if chooser else None
        if pick:
            override, why = resolve_base(pick, sig, puller, out, style)
            if why:
                return _refuse(why)
            checks = preflight_page()
    stops = preflight.blocked(checks)
    if stops:
        print(lexicon.refused_line([c.reason for c in stops], style), file=out)
        for c in stops:
            if c.name == "base":
                _base_help(c, args.model, out)
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
    base_data = next(c for c in checks if c.name == "base").data or {}
    base_hub, base_snapshot = None, base_data.get("snapshot")
    if base_data.get("needs_fetch"):
        base_hub, why = sig.hub_info(base_data["base"])
        if base_hub is None:
            return _refuse(f"cannot look up the base model {base_data['base']} on the hub: {why}")
        base_snapshot = fetch.snapshot_dir(hf_home, base_data["base"], base_hub.sha)
    inputs = {"model": str(snapshot), **({"base": str(base_snapshot)} if base_snapshot else {})}
    argv_run = bringup_config.supervisor_argv(cfg, args.model, run_dir, inputs=inputs,
                                              accept_credentials=not args.refuse_credentials_visible)
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
    if base_hub is not None:
        print(f"{style.icon('🌱')}fetching the base model {base_data['base']} at {base_hub.sha[:8]} "
              f"({base_hub.total_bytes / 1e9:.1f} GB) into {hf_home}", file=out)
        try:
            fetcher(base_data["base"], base_hub.sha, hf_home, files=base_hub.files)
        except fetch.FetchError as exc:
            return _refuse(str(exc))
    narrator = None
    if not args.quiet:
        # An existing run shows its last few entries first, so a resume says where it left off.
        narrator = narrate.Narrator(run_dir, style, out, replay=6 if resuming else False)
        narrator.start()
    try:
        code = supervisor_main(argv_run)
    finally:
        if narrator is not None:
            narrator.stop()
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
        _print_block_digest(run_dir, reason, args.model, out)
    elif code == EXIT_ERROR:
        print(f"{style.icon('⛈')}the run stopped on an error and the hardware was released. Run the same "
              "command again to resume.", file=out)
    return code


if __name__ == "__main__":
    sys.exit(main())
