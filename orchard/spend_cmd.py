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
from orchard import virtualspend_view as vsview

EXIT_OK, EXIT_PROBLEM = 0, 2
NOTE = ("Local spend is virtual. The agent logs re-send the whole conversation on every turn, so the prompt count is large; "
        "a hosted provider with prompt caching would bill most of those tokens for much less. Use it as a comparison figure only.")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _real(directory: Path) -> dict:
    """The helper's spend from one snapshot of its ledger. Confirmed charges (`cost_usd` and the counts beside it) are kept apart from
    what is not confirmed: reservations with no recorded result, torn records that were quarantined, and calls whose provider reported
    no usage. A ledger that cannot be read is unknown spend and is never shown as an empty history (a missing file is empty)."""
    try:
        recs = tg._read_ledger_strict(Path(directory))
    except tg.LedgerUnavailable as exc:
        return {"dir": str(directory), "error": f"the telegraph ledger cannot be read ({exc}); the helper spend is unknown"}
    try:
        s = tg.summarise_records(recs)
        acct = tg.accounting(recs)
    except Exception:                                               # noqa: BLE001 - a bad record must not stop the command
        return {"dir": str(directory), "error": "the telegraph ledger has a record that cannot be read; the helper spend is unknown"}
    return {"dir": str(directory), "calls": s["calls"], "ok": s["ok"], "cache_hits": s["cache_hits"],
            "refused": s["refused"], "tokens_in": s["tokens_in"], "tokens_out": s["tokens_out"],
            "cost_usd": s["cost_usd"], "cost_unknown_calls": s["cost_unknown_calls"],
            "unsettled_reservations": s["unsettled"], "unknown_usage_calls": s["unknown_usage_calls"],
            "torn_records": sum(1 for r in recs if "torn" in r),
            "worst_case_cost_usd": acct["cost"], "worst_case_unknown_cost_calls": acct["unknown_cost"]}


def _not_confirmed_line(real: dict) -> str:
    """One line about spend that is not confirmed, or an empty string when the ledger has none."""
    parts = []
    if real["unsettled_reservations"]:
        parts.append(f"{_plural(real['unsettled_reservations'], 'unsettled reservation')}")
    if real["torn_records"]:
        parts.append(_plural(real["torn_records"], "torn record"))
    if real["unknown_usage_calls"]:
        parts.append(f"{_plural(real['unknown_usage_calls'], 'call')} with unknown usage")
    if not parts:
        return ""
    line = (f"  not confirmed (may still be billed): {', '.join(parts)}; worst case, up to "
            f"${real['worst_case_cost_usd']:.4f} in all")
    if real["worst_case_unknown_cost_calls"]:
        line += f", plus {_plural(real['worst_case_unknown_cost_calls'], 'call')} of unknown cost"
    return line


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
    pin, listing_cut = None, False
    if not dirs:
        if runs_root is None or not Path(runs_root).is_dir():
            say("spend: name at least one directory, or set runs_root in the bringup config" + (f" ({problem})" if problem else ""))
            return EXIT_PROBLEM
        pin = vs.open_pin(runs_root)                                 # the runs folder is pinned: no link below it is followed
        if pin is None:
            say("spend: the runs folder cannot be opened")
            return EXIT_PROBLEM
        listed, listing_cut = vsview.discover(Path(runs_root), pin)
        dirs = [path for _, path in listed]
    rows, total = [], {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0}
    try:
        for d in dirs:
            s = vs.priced(vs.scan(d, beneath=pin), vcfg)
            if not s["turns"] and not args.dirs and s["complete"]:
                continue                                             # a folder under runs_root with no local turns is not listed
            rows.append({"dir": str(d), **s})
            for k in total:
                total[k] += s[k]
    finally:
        if pin is not None:
            pin.close()
    all_local = vs.priced({**total, "agent_wall_s": None, "request_s": None, "skipped": {}, "files": 0, "sources": {}, "complete": True}, vcfg)
    real = _real(Path(ledger_dir))
    if args.json:
        say(json.dumps({"real_helper_spend": real,
                        "local_virtual_spend": {"label": vs.LABEL, "note": NOTE, "directories": rows, "listing_cut_short": listing_cut,
                                                "all_directories": {k: all_local[k] for k in ("turns", "prompt_tokens", "completion_tokens", "virtual")}}},
                       indent=2, sort_keys=True))
        return EXIT_OK
    say("REAL helper spend (what the outside helper charged; telegraph ledger)")
    if "error" in real:
        say(f"  {real['error']}")
    else:
        unknown = f" (+ {real['cost_unknown_calls']} calls with no price)" if real["cost_unknown_calls"] else ""
        say(f"  confirmed: ${real['cost_usd']:.4f}{unknown} · {real['calls']} calls · {real['tokens_in']} in, {real['tokens_out']} out")
        pending = _not_confirmed_line(real)
        if pending:
            say(pending)
    say()
    say("LOCAL virtual spend (nobody was charged; local tokens at a hosted model's list price)")
    if not rows:
        say("  no local agent logs with token counts were found")
    for r in rows:
        wall = f" · {r['agent_wall_s'] / 3600:.2f} h of agent time" if r["agent_wall_s"] else ""
        say(f"  {r['dir']}: {r['turns']} turns · {r['prompt_tokens']} prompt, {r['completion_tokens']} completion tokens{wall}")
        for ref in r["virtual"]:
            say(f"      at {ref['name']} (${ref['price_in']:g} in, ${ref['price_out']:g} out per million): ${ref['cost_usd']:.4f}")
        if not r["complete"]:
            say("      INCOMPLETE: some logs were not read, so these figures are lower than the real use.")
        text = vs.skipped_text(r["skipped"])
        if text:
            say(f"      {text}")
    if len(rows) > 1:
        say(f"  all {len(rows)} directories: {all_local['turns']} turns · {all_local['prompt_tokens']} prompt, "
            f"{all_local['completion_tokens']} completion tokens")
        for ref in all_local["virtual"]:
            say(f"      at {ref['name']}: ${ref['cost_usd']:.4f}")
    if listing_cut:
        say("  The folder listing was cut short, so some folders are not listed at all.")
    say()
    say(NOTE)
    say("Chip lease time is not recorded in these logs, so it is not shown.")
    return EXIT_OK
