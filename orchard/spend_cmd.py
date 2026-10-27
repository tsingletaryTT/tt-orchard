# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard spend [DIR ...] [--json]`: the real helper spend and the virtual local spend, side by side.

REAL is what the outside helper (telegraph) charged, from its ledger. LOCAL is the token use of the local models in the
given directories (kernel-probe output, bring-up runs, benchmark runs), priced at the reference models of the
`[virtual_spend]` table (orchard/virtualspend.py). The local figure is virtual: nobody was charged. The two totals are
printed on separate lines and are never added together.

With no DIR it reads every run directory under `runs_root` of the bringup config. Reads only; it sends no request and
writes no file. Exit codes: 0 done, 2 a problem with the setup or the arguments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from orchard import telegraph as tg
from orchard import telegraph_cmd
from orchard import virtualspend as vs

EXIT_OK, EXIT_PROBLEM = 0, 2
NOTE = ("Local spend is virtual. The agent logs re-send the whole conversation on every turn, so the prompt count is large; "
        "a hosted provider with prompt caching would bill most of those tokens for much less. Use it as a comparison figure only.")


def _real(directory: Path) -> dict:
    try:
        s = tg.summarise(directory)
    except Exception:                                               # noqa: BLE001 - a bad record must not stop the command
        return {"error": "the telegraph ledger has a record that cannot be read", "dir": str(directory)}
    return {"dir": str(directory), "calls": s["calls"], "ok": s["ok"], "cache_hits": s["cache_hits"],
            "refused": s["refused"], "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
            "cost_usd": s["cost_usd"], "cost_unknown_calls": s["cost_unknown_calls"]}


def _config(args, environ, home):
    """(virtual Config, telegraph ledger dir, runs_root or None, problem text or None).

    Two files can be involved. The bringup config is found the usual way (`--config`, the environment, the checkout, the home
    folder) and gives `runs_root`. The tables file is `--telegraph-config` when it is given, otherwise the bringup config; it gives
    the `[telegraph]` and `[virtual_spend]` tables. A tables file that holds only those tables therefore never decides `runs_root`."""
    from orchard import bringup_config, cli
    named = getattr(args, "config", None) or environ.get(cli.CONFIG_ENV)
    try:
        bringup_path = cli.find_config(getattr(args, "config", None), environ, cli.CHECKOUT, Path(home))
        problem = None
    except cli.NoConfig as exc:
        if named:                                                    # a config the person named and that is missing is a setup error
            return None, None, None, str(exc)
        bringup_path, problem = None, str(exc)
    explicit = getattr(args, "telegraph_config", None)
    tables_path = Path(explicit) if explicit else bringup_path
    if tables_path is None:
        return vs.Config(), telegraph_cmd.default_dir(home), None, problem
    try:
        vcfg = vs.load_file(tables_path)
    except vs.ConfigError as exc:
        return None, None, None, str(exc)
    try:
        tcfg = telegraph_cmd.load_config(tables_path)
    except tg.ConfigError as exc:                                    # a bad table must not send the real spend to another ledger
        return None, None, None, str(exc)
    runs_root = None
    for candidate in (bringup_path, tables_path if bringup_path is None else None):    # the tables file is a bringup config only as a last resort
        if candidate is None:
            continue
        try:
            runs_root = bringup_config.load(candidate).runs_root
        except Exception:                                            # noqa: BLE001 - a config that does not load has no runs root
            continue
        break
    return vcfg, (tcfg.dir or telegraph_cmd.default_dir(home)), runs_root, problem


def main(args, *, out, environ=None, home=None) -> int:
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)

    def say(text=""):
        print(text, file=out)

    vcfg, ledger_dir, runs_root, problem = _config(args, environ, home)
    if vcfg is None:
        say(f"spend: {problem}")
        return EXIT_PROBLEM
    if getattr(args, "dir", None):
        ledger_dir = Path(args.dir)
    dirs = [Path(d) for d in (args.dirs or [])]
    if not dirs:
        if runs_root is None or not Path(runs_root).is_dir():
            say("spend: name at least one directory, or set runs_root in the bringup config" + (f" ({problem})" if problem else ""))
            return EXIT_PROBLEM
        try:
            dirs = sorted(p for p in Path(runs_root).iterdir() if p.is_dir() and not p.is_symlink())
        except OSError:
            say("spend: the runs folder cannot be listed")
            return EXIT_PROBLEM
    rows, total = [], {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for d in dirs:
        s = vs.priced(vs.scan(d), vcfg)
        if not s["turns"] and not args.dirs:
            continue                                                 # a folder under runs_root with no local turns is not listed
        rows.append({"dir": str(d), **s})
        for k in total:
            total[k] += s[k]
    all_local = vs.priced({**total, "agent_wall_s": None, "request_s": None, "skipped": {}, "files": 0, "sources": {}, "complete": True}, vcfg)
    real = _real(Path(ledger_dir))
    if args.json:
        say(json.dumps({"real_helper_spend": real,
                        "local_virtual_spend": {"label": vs.LABEL, "note": NOTE, "directories": rows,
                                                "all_directories": {k: all_local[k] for k in ("turns", "prompt_tokens", "completion_tokens", "virtual")}}},
                       indent=2, sort_keys=True))
        return EXIT_OK
    say("REAL helper spend (what the outside helper charged; telegraph ledger)")
    if "error" in real:
        say(f"  {real['error']}")
    else:
        unknown = f" (+ {real['cost_unknown_calls']} calls with no price)" if real["cost_unknown_calls"] else ""
        say(f"  ${real['cost_usd']:.4f}{unknown} · {real['calls']} calls · {real['tokens_in']} in, {real['tokens_out']} out")
    say()
    say("LOCAL virtual spend (nobody was charged; local tokens at a hosted model's list price)")
    if not rows:
        say("  no local agent logs with token counts were found")
    for r in rows:
        wall = f" · {r['agent_wall_s'] / 3600:.2f} h of agent time" if r["agent_wall_s"] else ""
        say(f"  {r['dir']}: {r['turns']} turns · {r['prompt_tokens']} prompt, {r['completion_tokens']} completion tokens{wall}")
        for ref in r["virtual"]:
            say(f"      at {ref['name']} (${ref['price_in']:g} in, ${ref['price_out']:g} out per million): ${ref['cost_usd']:.4f}")
        text = vs.skipped_text(r["skipped"])
        if text:
            say(f"      {text}")
    if len(rows) > 1:
        say(f"  all {len(rows)} directories: {all_local['turns']} turns · {all_local['prompt_tokens']} prompt, "
            f"{all_local['completion_tokens']} completion tokens")
        for ref in all_local["virtual"]:
            say(f"      at {ref['name']}: ${ref['cost_usd']:.4f}")
    say()
    say(NOTE)
    say("Chip lease time is not recorded in these logs, so it is not shown.")
    return EXIT_OK
