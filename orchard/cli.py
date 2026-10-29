# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard`: the front door to a model bring-up.

    tt-orchard bringup MODEL [--run-dir DIR] [--dry-run] [--no-fetch] [--refuse-credentials-visible] [--no-graft-trial] [--sampling auto|needed|not-needed] [--resume-aborted]
    tt-orchard status   [MODEL | --run-dir DIR] [--json] [--style S]
    tt-orchard watch    [MODEL | --run-dir DIR] [--all] [--once] [--style S]
    tt-orchard pause | resume | abort   [MODEL | --run-dir DIR]
    tt-orchard ui       [--lan] [--host ADDR] [--port 8780] [--toplike PATH]
    tt-orchard lab setup [--model ORG/NAME]... [--yes] [--check]
    tt-orchard caches   [--lab] [--prune --older-than DAYS [--yes]]
    tt-orchard coder    [status | stop [--yes]]
    tt-orchard package  RUN_DIR --namespace ORG [--step stage|boot|finish|all] [--restage] [--graft-mtp | --no-graft-mtp] [--exact CHIP]
    tt-orchard package-card RUN_DIR --benchmark FILE --model-key KEY [--base-key KEY] [--reported FILE] [--accept-unverified-identity]
    tt-orchard reported show RUN_DIR
    tt-orchard reported accept RUN_DIR (--all | --rows NAME,NAME | --row NAME) [--replace]
    tt-orchard ttlang-consistency [--src-dir DIR] [--venv-dir DIR] [--tt-lang-skill-dir DIR] [--tt-metal-dir DIR]
    tt-orchard kernel-env [--check] [--src-dir DIR] [--venv-dir DIR] [--tt-lang-skill-dir DIR] [--allow-stale-skills]
    tt-orchard demo-env [--check] [--dry-run] [--venv-dir DIR] [--tt-metal-dir DIR]
    tt-orchard audio-reference-env [--check] [--dry-run] [--venv-dir DIR] [--src-dir DIR]
    tt-orchard audio-package RUN_DIR --namespace ORG --bench FILE --demo-repo DIR [--snapshot DIR] [--check]
    tt-orchard inventory TARGET [--python P] [--ttnn-python P] [--against F] [--layers N] [--seq N] [--class C] [--out DIR]
    tt-orchard inventory verify-candidates --inventory FILE|DIR [--out DIR] [--image IMG] [--max-ops N] [--timeout S] [--record]
    tt-orchard coder-speed [--endpoint URL --model NAME | --tiers] [--samples N] [--watch SECONDS] [--json]
    tt-orchard kernel-probe (--op NAME | --ladder [--rungs A,B] [--rung-wall OP=S] [--max-hours H]) [--wall-s S] [--attempts N] [--phase sim|hw] [--out DIR] [--endpoint URL] [--model NAME]
                            [--temperature T | --temperatures T1,T2] [--top-p P] [--seed N]
                            [--src-dir D] [--venv-dir D] [--shared-python P] [--tt-lang-skill-dir D]
                            [--allow-stale-skills] [--allow-slow-coder]
                            [--hw-checks N | --no-agent-hw] [--hw-lease-s S] [--hw-call-timeout-s S] [--hw-wait-s S]

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
import contextlib
import dataclasses
import io
import fcntl
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path

from orchard import __version__, blocked, bringup_config, dlprogress, fetch, lexicon, narrate, nearest, orchard_view, preflight, ui
from orchard import caches as tensor_caches

EXIT_OK, EXIT_REFUSED, EXIT_ERROR, EXIT_ABORTED, EXIT_BLOCKED = 0, 2, 3, 4, 5   # the supervisor's own codes
AUTO_RETRIES = 2            # a choice: a transient block gets two more tries, then a person decides
AUTO_RETRY_WAIT_S = 60.0    # `gozer status` shows a released board BUSY-UNTRACKED for about a minute
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
    b.add_argument("--no-graft-trial", action="store_true",
                   help="skip stage 2's graft trial. When the model has no mtp.* tensors and the nearest model has a head that "
                        "fits, stage 2 serves the grafted drafter once beside the safe configuration and keeps it when it is "
                        "faster; that is one more hardware test of 10 to 35 minutes. The run records the choice")
    b.add_argument("--sampling", choices=("auto", "needed", "not-needed"), default=None,
                   help="whether the model's purpose needs sampling (issue 107). A speculative drafter can ignore sampled requests. With 'auto' "
                        "(the default) stage 2 reads the model's generation_config.json and card, and when sampling is needed and the served "
                        "configuration does not show it applied, serves a drafter-off variant once and chooses by measurement. 'needed' and "
                        "'not-needed' are your word. The run records it and a resume keeps it")
    b.add_argument("--resume-aborted", action="store_true",
                   help="go on with a run that an operator or a signal aborted (the web page's Resume run does this). The ledger records the resume and the run continues from there. Without it an aborted run stays aborted. A finished run, and an abort of any other kind, are left as they are")
    b.add_argument("--prune-caches", action="store_true",
                   help="when the only block is a full disk, remove the oldest tensor caches that no unfinished run "
                        "uses, just enough to fit (never the kernel cache), then check again. With --dry-run it "
                        "only says what it would remove")
    b.add_argument("--auto-retries", type=int, default=AUTO_RETRIES, metavar="N",
                   help=f"retry a run that ends blocked for a transient reason (coder-unusable, hardware-unhealthy, "
                        f"agent-stuck) up to N times, waiting {AUTO_RETRY_WAIT_S:.0f} s before each. 0 turns it off. "
                        f"Default {AUTO_RETRIES}")
    b.add_argument("--offline", action="store_true",
                   help="skip the base-model research and make no request for it. The preflight's hub lookup, "
                        "`tt model search` and any download still happen")
    b.add_argument("--no-github", action="store_true",
                   help="skip the GitHub issue and pull request search and the GitHub code search in the "
                        "base-model research. tt-metal's model listings and README are still read")
    b.add_argument("--refresh-research", action="store_true",
                   help="ignore the cached answers from an earlier attempt of this run")
    b.add_argument("--accept-inferred-base", action="store_true",
                   help="when the card names no base model, use the one the research inferred from the "
                        "architecture (and download its snapshot). Without it an inferred base blocks the run")
    b.add_argument("--mode", choices=bringup_config.MODES,
                   help="where the hardware tests run for this run: local (this box) or lab (the [lab] box). "
                        "Default: the config's mode")
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
    u = sub.add_parser("ui", parents=[common], help="watch and control the runs on this machine in a browser")
    u.add_argument("--host", default=None,
                   help="the address to listen on. Default 127.0.0.1; with --lan, 0.0.0.0 (every interface). Without "
                        "--lan only a loopback address is accepted; from another computer use "
                        "`ssh -L 8780:localhost:8780 <box>`")
    u.add_argument("--toplike", metavar="PATH",
                   help="the tt-toplike binary the Hardware view runs (view-only). Default: tt-toplike or "
                        "tt-toplike-tui on PATH; without one the view says how to install it")
    u.add_argument("--lan", action="store_true",
                   help="listen on the local network with no login: anyone who can reach the port can pause, abort "
                        "and start runs. Open the port in the firewall yourself")
    u.add_argument("--port", type=int, default=8780, help="the port to listen on (default 8780)")
    lab = sub.add_parser("lab", parents=[common], help="get the [lab] box ready for runs whose hardware tests "
                                                       "run there (`lab setup --help`)")
    lab.add_argument("action", choices=["setup"])
    lab.add_argument("rest", nargs=argparse.REMAINDER, help="options for the action")
    sc = sub.add_parser("suite-control", parents=[common],
                        help="build the prompt suite's control from the nearest model (`suite-control prepare --help`)")
    sc.add_argument("action", choices=["prepare", "run"])
    sc.add_argument("rest", nargs=argparse.REMAINDER, help="--stage-dir DIR and the other options")
    from orchard import caches as _caches
    sub.add_parser("caches", parents=[common, _caches.parser(add_help=False)],
                   help="list the tensor caches (here and on the lab) and prune the ones no run uses")
    from orchard import kept_coder as _kept
    sub.add_parser("coder", parents=[common, _kept.parser(add_help=False)],
                   help="show or stop the coder a lab run kept up ([coder] keep_up)")
    pk = sub.add_parser("package", parents=[common],
                        help="package a finished run (stage 7): stage, boot-check under a gozer lease, write the "
                             "cards, run the gate, and print the publish commands as text. Publishes nothing")
    pk.add_argument("run_dir", help="the run directory (it must have ended ready-for-operator-review)")
    pk.add_argument("--namespace", help="the Hugging Face organisation the package will be published under "
                                        "(needed to stage)")
    pk.add_argument("--step", choices=("all", "stage", "boot", "finish"), default="all",
                    help="all (default); or one part of it. boot opens the chips, so it runs only under a lease")
    pk.add_argument("--restage", action="store_true",
                    help="move an earlier stages/7 aside and stage again (without it an existing stage 7 is refused)")
    pk.add_argument("--graft-mtp", action="store_true",
                    help="stage a package that adds the nearest model's mtp.* tensors at each start and keeps the "
                         "speculative drafter and on-device sampling on (for a model that ships none). The boot check "
                         "serves it that way. Stage 2's graft trial (a model with a graft candidate) decides this by itself; the "
                         "flag forces a graft the trial did not choose")
    pk.add_argument("--no-graft-mtp", action="store_true",
                    help="stage a package without the graft. Refused when stage 2's graft trial chose the graft (its stage 4 tests were "
                         "built with it) or stage 2 itself served with the graft")
    pk.add_argument("--exact", type=int, metavar="CHIP",
                    help="give gozer this chip number: it leases the board that holds it")
    pk.add_argument("--models-root", help="where installed tt-model bundles live (default: the run's recorded path)")
    pk.add_argument("--cache-root", help="the cache root; the kernel cache is under it (default: the run's recorded path)")
    pk.add_argument("--gozer", default="gozer", help="the gozer command (default gozer)")
    pk.add_argument("--handoff", metavar="TOKEN", help="set by `package` for the boot step its lease command runs: "
                                                       "proves the parent holds the run. Leave it out; a bare "
                                                       "`--step boot` takes the run itself")
    pk.set_defaults(python=sys.executable)
    pc = sub.add_parser("package-card", parents=[common],
                        help="put benchmark results, and optionally the authors' reported scores, in the package "
                             "cards of a packaged run, then gate it again")
    pc.add_argument("run_dir", help="the run directory that `tt-orchard package` finished")
    pc.add_argument("--benchmark", required=True, metavar="FILE",
                    help="a benchmark.json: {\"tasks\": {task: {KEY: {score, n, cut_off?}, \"paired\": {sign_test_p?}}}}")
    pc.add_argument("--model-key", required=True, metavar="KEY", help="the key of the model under test in that file")
    pc.add_argument("--base-key", metavar="KEY", help="the key of the nearest model, run under the same conditions")
    pc.add_argument("--reported", metavar="FILE", help="a reported-results.json of the scores the authors report")
    pc.add_argument("--replace-reported", action="store_true",
                    help="replace reported scores the run already holds")
    pc.add_argument("--accept-unverified-identity", action="store_true",
                    help="go on when the benchmark file does not say which model (and revision) each key is, so the "
                         "scores cannot be shown to be this package's. A file that names another model is always "
                         "refused. The ledger records what was not verified")
    rp = sub.add_parser("reported", parents=[common],
                        help="show or accept the scores the model card reports for the model (drafted by stage 0)")
    rsub = rp.add_subparsers(dest="action", required=True)
    rsub.add_parser("show", parents=[common], help="list the drafted rows and what was left out (read-only)") \
        .add_argument("run_dir", help="the run directory")
    ra = rsub.add_parser("accept", parents=[common],
                         help="copy chosen draft rows into the run's reported-results.json and record who did it")
    ra.add_argument("run_dir", help="the run directory")
    pick = ra.add_mutually_exclusive_group(required=True)
    pick.add_argument("--all", action="store_true", help="accept every drafted row")
    pick.add_argument("--rows", metavar="NAME,NAME", help="accept the rows with these names (comma separated)")
    pick.add_argument("--row", action="append", metavar="NAME",
                      help="accept one row by its exact name (a name with a comma); repeat for more")
    ra.add_argument("--replace", action="store_true", help="replace a reported-results.json the run already holds")
    tp = sub.add_parser("telegraph", parents=[common],
                        help="the opt-in outside helper model: check it, ask it by hand, record whether it helped, "
                             "show what it cost (off unless a [telegraph] table enables it)")
    tsub = tp.add_subparsers(dest="action", required=True)

    def _tele(name, help_text):
        sp = tsub.add_parser(name, parents=[common], help=help_text)
        sp.add_argument("--telegraph-config", metavar="FILE",
                        help="a TOML file with the [telegraph] table (default: the bringup config's)")
        sp.add_argument("--dir", metavar="DIR", help="the ledger, cache and reply directory "
                                                      "(default ~/.cache/tt-orchard/telegraph)")
        return sp

    _tele("check", "show the setting and whether the key is present (never the key)") \
        .add_argument("--ping", action="store_true", help="also ask the endpoint for its model list (no tokens spent)")
    ta = _tele("ask", "send one question by hand")
    ta.add_argument("--problem", required=True, help="a short id for the problem (the ladder and the summary use it)")
    ta.add_argument("--task-file", required=True, metavar="FILE", help="what was being attempted")
    ta.add_argument("--failure-file", required=True, metavar="FILE", help="what went wrong")
    tr = _tele("report", "record whether the answer to a call helped")
    tr.add_argument("call_id")
    tpick = tr.add_mutually_exclusive_group(required=True)
    tpick.add_argument("--helped", action="store_true", help="the answer helped")
    tpick.add_argument("--not-helped", dest="not_helped", action="store_true", help="the answer did not help")
    _tele("usage", "show calls, tokens, cost and how often the help worked") \
        .add_argument("--json", action="store_true", help="print the summary as JSON")
    dp = sub.add_parser("db", parents=[common],
                        help="the project database (the telegraph ledger lives in it): status, migrate telegraph, export "
                             "telegraph, backup, restore, confirm-restore")
    dsub = dp.add_subparsers(dest="action", required=True)

    def _db(name, help_text):
        sp = dsub.add_parser(name, parents=[common], help=help_text)
        sp.add_argument("--dir", metavar="DIR", help="the telegraph state directory (default: the config's [telegraph] dir, "
                                                     "$ORCHARD_TELEGRAPH_DIR, then ~/.cache/tt-orchard/telegraph)")
        sp.add_argument("--telegraph-config", metavar="FILE", help="a TOML file with the [telegraph] table (default: the bringup config's)")
        sp.add_argument("--scope", default=None, help="the ledger scope inside the database (default: default)")
        return sp

    _db("status", "what is in the state directory: database, backups, markers, migration phase")
    dm = _db("migrate", "import telegraph.jsonl and the cache into the database and cut the old files over")
    dm.add_argument("what", choices=["telegraph"])
    dm.add_argument("--dry-run", action="store_true", help="check that an import gives equal totals; change nothing")
    dm.add_argument("--dry-run-first", action="store_true",
                    help="run the dry run, show the checks and, when every check is equal and you confirm once, run the real "
                         "migration (never without a yes, never while another process holds telegraph.lock)")
    dm.add_argument("--accept-malformed", type=int, default=None, metavar="N",
                    help="go on although the ledger has exactly N lines that are not records (they were never counted)")
    de = _db("export", "write a ledger as jsonl on standard output (the lines the old file held)")
    de.add_argument("what", choices=["telegraph"])
    de.add_argument("--from-archive", action="store_true", help="read the archived ledger telegraph.jsonl.archived (the database is not read)")
    _db("backup", "take an online backup into backups/")
    dr = _db("restore", "put a backup in place of the database (the ledger is then unconfirmed)")
    dr.add_argument("backup", metavar="BACKUP", help="a file from the backups/ directory")
    dc = _db("confirm-restore", "say the history is accounted for, so calls are allowed again")
    dc.add_argument("--note", required=True, help="a sentence on how you checked (for example, the provider's usage page)")
    dc.add_argument("--no-replies", action="store_true",
                    help="do not charge reply files that have no ledger record at the worst case")
    sp_ = sub.add_parser("spend", parents=[common],
                         help="the real helper spend and the virtual spend of the local models, side by side "
                              "(the virtual figure is never added to the real one)")
    sp_.add_argument("dirs", nargs="*", metavar="DIR",
                     help="a kernel-probe output, run or benchmark directory (default: every run under runs_root)")
    sp_.add_argument("--json", action="store_true", help="print the figures as JSON")
    sp_.add_argument("--dir", metavar="DIR", help="the telegraph ledger directory (default: the config's, else ~/.cache/tt-orchard/telegraph)")
    sp_.add_argument("--telegraph-config", metavar="FILE", help="a TOML file with the [telegraph] and [virtual_spend] tables (default: the bringup config's)")
    k = sub.add_parser("kernel-env", parents=[common],
                       help="build or check the tt-lang simulator environment that kernel-probe uses")
    k.add_argument("--check", action="store_true",
                   help="change nothing; report each step and exit 2 if one is missing or failed")
    k.add_argument("--src-dir", help="where the tt-lang checkout lives (default ~/orchard-src/tt-lang, or "
                                     "$ORCHARD_TTLANG_SRC)")
    k.add_argument("--venv-dir", help="where the simulator venv lives (default ~/orchard-venvs/ttlang, or "
                                      "$ORCHARD_TTLANG_VENV)")
    k.add_argument("--tt-lang-skill-dir", help="the installed tt-lang skill whose guide and examples the probe gives "
                                               "the coder (default ~/.claude/skills/ttm-tt-lang, or "
                                               "$ORCHARD_TTLANG_SKILL)")
    k.add_argument("--allow-stale-skills", action="store_true",
                   help="with --check: a tt-lang checkout or tt-lang skill that is not the latest upstream has does "
                        "not fail the check")
    tc_ = sub.add_parser("ttlang-consistency", parents=[common],
                         help="check the tt-lang pin against the checkout, the simulator, the skill, the reference files "
                              "and the tt-metal tree; read only")
    tc_.add_argument("--src-dir", help="the tt-lang checkout (default ~/orchard-src/tt-lang, or $ORCHARD_TTLANG_SRC)")
    tc_.add_argument("--venv-dir", help="the simulator venv (default ~/orchard-venvs/ttlang, or $ORCHARD_TTLANG_VENV)")
    tc_.add_argument("--tt-lang-skill-dir", help="the installed tt-lang skill (default ~/.claude/skills/ttm-tt-lang, "
                                                 "or $ORCHARD_TTLANG_SKILL)")
    tc_.add_argument("--tt-metal-dir", help="the host tt-metal tree, read only (default ~/tt-metal, or "
                                            "$ORCHARD_TT_METAL_DIR)")
    d = sub.add_parser("demo-env", parents=[common],
                       help="build or check the interpreter that runs tt-metal demos and tt-train (numpy below 2, cv2, datasets, "
                            "the host tree's ttnn and ttml)")
    d.add_argument("--check", action="store_true",
                   help="change nothing; report each rule (numpy, cv2, datasets, torch, ttnn and ttml as distributions, the ttnn "
                        "version against the host tree's tag) and exit 2 if one is missing or failed. Imports no ttnn")
    d.add_argument("--dry-run", action="store_true", help="say what the build would do and change nothing")
    d.add_argument("--venv-dir", help="where the venv lives (default ~/orchard-venvs/tt-demo, or $ORCHARD_DEMO_VENV)")
    d.add_argument("--tt-metal-dir", help="the host tt-metal tree, built and at a release tag (default ~/tt-metal, or "
                                          "$ORCHARD_TT_METAL_DIR)")
    ar = sub.add_parser("audio-reference-env", parents=[common],
                        help="build or check the CPU-only environment and pinned checkout of the stable-audio-3 library that run the "
                             "audio reference (no chip, no Hugging Face token)")
    ar.add_argument("--check", action="store_true", help="change nothing; report each rule and exit 2 if one failed")
    ar.add_argument("--dry-run", action="store_true", help="say what the build would do and change nothing")
    ar.add_argument("--venv-dir", help="where the venv lives (default ~/orchard-venvs/stable-audio, or $ORCHARD_AUDIO_VENV)")
    ar.add_argument("--src-dir", help="where the pinned stable-audio-3 checkout lives (default ~/orchard-src/stable-audio-3, or $ORCHARD_AUDIO_SRC)")
    ap = sub.add_parser("audio-package", parents=[common],
                        help="stage a local, weights-free package from a finished Stable Audio 3 small-music run, render its card from the run's "
                             "numbers and check the card against the model's license. Uploads and publishes nothing")
    ap.add_argument("run_dir", help="the run directory (holds ledger.jsonl and stages/2/audio_result.json)")
    ap.add_argument("--namespace", required=True, help="the Hugging Face user or organisation the package would be published under (used in text only)")
    ap.add_argument("--bench", help="the audio_bench.py sweep file the card's timings come from (one warm-up, 3 or more measured runs)")
    ap.add_argument("--demo-repo", help="the separate demo repository checkout whose server, client and page are copied into the package (read only)")
    ap.add_argument("--snapshot", help="the pinned model snapshot that holds LICENSE.md and NOTICE (default: the path the run recorded)")
    ap.add_argument("--check", action="store_true", help="stage nothing; run the gate on the package already staged in the run")
    inv = sub.add_parser("inventory", parents=[common],
                         help="list the ops a model's reference code uses and which ones ttnn has; "
                              "`inventory refresh-ttnn` records ttnn's op names (takes a one-chip lease); "
                              "`inventory verify-candidates` tests the proposed ttnn names on the ttsim simulator (no chip)")
    inv.add_argument("target", help="a model directory with a config.json, or the word refresh-ttnn or verify-candidates")
    inv.add_argument("--simulator", metavar="LIB",
                     help="refresh-ttnn: list the names against this ttsim library (libttsim_bh.so, with its "
                          "soc_descriptor.yaml beside it) and take no lease")
    inv.add_argument("--inventory", metavar="FILE", help="verify-candidates: an ops-inventory.json, or the directory that holds it")
    inv.add_argument("--image", default=None, help="verify-candidates: the ttsim simulator image (default ttsim-blackhole-space:local)")
    inv.add_argument("--max-ops", type=int, default=None, help="verify-candidates: most candidate pairs to run (default 40)")
    inv.add_argument("--timeout", type=int, default=None, help="verify-candidates: seconds before the simulator run is stopped (default 900)")
    inv.add_argument("--record", action="store_true",
                     help="verify-candidates: also merge the verified entries into the package's verified_mappings.json")
    inv.add_argument("--python", help="interpreter with torch and transformers (default $ORCHARD_REFERENCE_PYTHON, "
                                      "then reference_python in bringup.toml)")
    inv.add_argument("--ttnn-python", help="interpreter whose installed ttnn version is read, and that refresh-ttnn "
                                           "runs under a lease (default ~/.tenstorrent-venv/bin/python)")
    inv.add_argument("--against", metavar="FILE", help="an ops-inventory.json of a nearest model, to mark shared ops")
    inv.add_argument("--layers", type=int, default=2, help="layers to trace (default 2)")
    inv.add_argument("--seq", type=int, default=8, help="sequence length to trace (default 8)")
    inv.add_argument("--class", dest="cls", help="the transformers class to trace if not config.json's first architecture")
    inv.add_argument("--out", help="output directory (default ./ops-inventory)")
    cs = sub.add_parser("coder-speed", parents=[common],
                        help="measure how fast a model server decodes and compare it with the best it has done "
                             "(read-only; it reports a slow server and never restarts one)")
    cs.add_argument("--endpoint", help="the server's OpenAI-compatible URL (default: the coder tier of the tier config)")
    cs.add_argument("--model", help="the model id the server serves; needed with --endpoint")
    cs.add_argument("--tiers", action="store_true", help="check every distinct server the tier config names")
    cs.add_argument("--samples", type=int, default=1,
                    help="samples per server in one run (default 1); three low samples in a row make a server slow")
    cs.add_argument("--interval", type=float, default=sw_interval(),
                    help="seconds between the samples of one run (default %(default)s)")
    cs.add_argument("--watch", type=float, metavar="SECONDS",
                    help="keep sampling every SECONDS until a server is slow (exit 3), or three errors in a row (exit 2)")
    cs.add_argument("--baseline", type=float, metavar="TOK_S",
                    help="compare with this speed instead of the stored best (the stored best is not changed)")
    cs.add_argument("--reset-baseline", action="store_true",
                    help="forget the stored best for these servers first (after changing a server on purpose)")
    cs.add_argument("--baseline-file", help="where the best speeds are kept (default ~/.cache/tt-orchard/speed-baseline.json)")
    cs.add_argument("--json", action="store_true", help="print one JSON list instead of lines")
    from orchard.defaults import KERNEL_PROBE_OPS
    kp = sub.add_parser("kernel-probe", parents=[common],
                        help="ask the running coder to write a tt-lang kernel for one op, N times, and check each "
                             "on the simulator (and with --phase hw, on one leased chip). Starts no server")
    kp.add_argument("--op", choices=KERNEL_PROBE_OPS, help="the op to write a kernel for (required without --ladder)")
    kp.add_argument("--attempts", type=int, default=5,
                    help="attempts per temperature, each a fresh conversation (default 5)")
    kp.add_argument("--temperature", metavar="T",
                    help="sampling temperature, 0 to 2 (default 0: asks the server for greedy decoding; attempts then "
                         "often repeat each other but are not guaranteed to be identical. Raise it to explore wilder "
                         "or more ingenious kernels)")
    kp.add_argument("--temperatures", metavar="T1,T2,...",
                    help="several temperatures, comma-separated (for example 0,0.5,1.0); --attempts runs at each. "
                         "Not with --temperature")
    kp.add_argument("--top-p", dest="top_p", metavar="P", help="nucleus sampling, 0 < P <= 1 (default: not sent)")
    kp.add_argument("--seed", metavar="N",
                    help="a base seed: above temperature 0, attempt k sends N + k (default: derived from the probe id "
                         "and recorded in the report). A seed makes the request the same on a rerun; whether the server "
                         "honours it is not verified")
    kp.add_argument("--phase", choices=("sim", "hw"), default="sim",
                    help="sim (default): the simulator only. hw: then a smoke test and each simulator pass on one "
                         "chip, under `gozer run --chips 1`")
    kp.add_argument("--out", default="kernel-probe", help="output directory (default ./kernel-probe)")
    kp.add_argument("--endpoint", help="the coder's OpenAI-compatible URL, ending in /v1 (default: the tier stage 2 "
                                       "uses, from the tier config bringup.toml names)")
    kp.add_argument("--model", help="the model id the coder serves (default: that tier's model)")
    kp.add_argument("--src-dir", help="the tt-lang checkout (default ~/orchard-src/tt-lang, or $ORCHARD_TTLANG_SRC)")
    kp.add_argument("--venv-dir", help="the simulator venv (default ~/orchard-venvs/ttlang, or $ORCHARD_TTLANG_VENV)")
    kp.add_argument("--shared-python", help="the interpreter with ttnn and tt-lang that the hardware phase runs "
                                            "(default ~/.tenstorrent-venv/bin/python)")
    kp.add_argument("--tt-lang-skill-dir", help="the installed tt-lang skill whose guide.md and examples.md go into "
                                                "each attempt's reference/ (default ~/.claude/skills/ttm-tt-lang, or "
                                                "$ORCHARD_TTLANG_SKILL)")
    kp.add_argument("--allow-slow-coder", action="store_true",
                    help="run even when the coder is decoding far below its best speed (the probe measures it before "
                         "each attempt and otherwise stops, exit 6); the report records each measurement")
    kp.add_argument("--library", choices=("all", "others", "none"), default="all",
                    help="the worked kernels put into each attempt's reference/: all (default), others (leave out the "
                         "kernel for the op under test, so a pass is not a copy) or none; the report records the mode")
    kp.add_argument("--staged", action="store_true",
                    help="build the kernel in stages, each ending at a simulator check against a reference for that step "
                         "(copy, one arithmetic step, the reduction, the broadcast back, the whole op); the verified "
                         "prefix is the next stage's starting file. For ops with a stage table (center, rmsnorm); needs "
                         "--no-agent-hw and --phase sim")
    kp.add_argument("--stage-turns", dest="stage_turns", type=int, metavar="N",
                    help="with --staged: the turns each stage may use (default: 40 divided by the op's stages)")
    kp.add_argument("--stage-wall-s", dest="stage_wall_s", type=float, metavar="S",
                    help="with --staged: the seconds each stage may use (default: 1800 divided by the op's stages)")
    kp.add_argument("--hw-checks", dest="hw_checks", type=int, metavar="N",
                    help="how many times each attempt may call the `hardware_check` tool, which runs its current kernel.py "
                         "on one leased chip (default 6; 0 turns the tool off). A call takes about 45 s with the lease "
                         "and the chip reset")
    kp.add_argument("--no-agent-hw", dest="no_agent_hw", action="store_true",
                    help="offer the agent no hardware tool: the same as --hw-checks 0, for measuring the coder alone")
    kp.add_argument("--hw-lease-s", dest="hw_lease_s", type=float, metavar="S",
                    help="chip seconds one attempt's hardware checks may use in all (default 600; 120 s of it are reserved for the stop and the reset)")
    kp.add_argument("--hw-call-timeout-s", dest="hw_call_timeout_s", type=float, metavar="S",
                    help="the longest one hardware check may run before the probe stops it (default 240; the stall "
                         "watcher usually stops a hang sooner)")
    kp.add_argument("--hw-wait-s", dest="hw_wait_s", type=float, metavar="S",
                    help="how long a hardware check waits for a busy board before it tells the agent to use the "
                         "simulator (default 120)")
    kp.add_argument("--allow-stale-skills", action="store_true",
                    help="run even when the tt-lang checkout or the tt-lang skill is not the latest upstream has, or "
                         "could not be compared; the report records it with the shas")
    kp.add_argument("--wall-s", dest="wall_s", type=float, metavar="S",
                    help="the wall budget of one attempt in seconds (default 1800; checked between turns). With --ladder "
                         "it is the default for every rung; --rung-wall sets a rung's own")
    kp.add_argument("--ladder", action="store_true",
                    help="run the capability ladder in place of a single op: the ops from easy to hard (add, mul, scale_shift, "
                         "add_row, rsqrt, row_sum, center, rmsnorm, linear_scan), each as its own run in <out>/<op>/ "
                         "with the same sampling, then <out>/ladder-report.md with the passes and failure classes per "
                         "rung. Do not give --op")
    kp.add_argument("--rungs", metavar="OP1,OP2,...", help="with --ladder: only these rungs (default: all of them)")
    kp.add_argument("--rung-wall", dest="rung_wall", metavar="OP=S,...",
                    help="with --ladder: the wall budget of named rungs in seconds (see size_wall_budget in "
                         "orchard/kernelladder.py: 4 times the slowest passing pilot, from 300 to 1800)")
    kp.add_argument("--rung-expect-s", dest="rung_expect_s", metavar="OP=S,...",
                    help="with --ladder: the seconds one attempt of a rung is expected to take, for the projection "
                         "that --max-hours checks (default: the rung's wall budget)")
    kp.add_argument("--max-hours", dest="max_hours", type=float, metavar="H",
                    help="with --ladder: stop after a rung when the elapsed time plus the projection for the rungs left "
                         "is above H hours; the report says so")
    kp.add_argument("--report-only", dest="report_only", action="store_true",
                    help="with --ladder: write <out>/ladder-report.md from the rung directories already there and run "
                         "nothing")
    return p


def _inventory(args, env, home, out) -> int:
    """`inventory refresh-ttnn` records ttnn's op names (takes a lease); `inventory DIR` traces a model."""
    from orchard import opinventory, ttnn_dump
    say = lambda line: print(line, file=out)
    ttnn_python = args.ttnn_python or str(home / ".tenstorrent-venv" / "bin" / "python")
    if args.target == "refresh-ttnn":
        return ttnn_dump.main(ttnn_python, say=say, simulator=args.simulator)
    if args.target == "verify-candidates":
        from orchard import opverify
        if not args.inventory:
            return _refuse("verify-candidates needs --inventory (an ops-inventory.json or the directory that holds it)")
        return opverify.main(args.inventory, out_dir=args.out or "verify-candidates",
                             image=args.image or opverify.DEFAULT_IMAGE,
                             max_ops=args.max_ops if args.max_ops is not None else opverify.DEFAULT_MAX_OPS,
                             timeout=args.timeout if args.timeout is not None else opverify.DEFAULT_TIMEOUT_S,
                             record=args.record, say=say)
    python = args.python or env.get("ORCHARD_REFERENCE_PYTHON")
    if not python:
        try:
            python = bringup_config.load(find_config(getattr(args, "config", None), env, CHECKOUT, home)).reference_python
        except (NoConfig, bringup_config.BringupConfigError):
            python = None
    if not python:
        return _refuse("no interpreter with torch and transformers: pass --python, set ORCHARD_REFERENCE_PYTHON, "
                       "or set reference_python in bringup.toml")
    return opinventory.run_inventory(args.target, python=python, ttnn_python=ttnn_python,
                                     out_dir=args.out or "ops-inventory", layers=args.layers, seq=args.seq,
                                     cls=args.cls, against=args.against, say=say)


def sw_interval() -> float:
    from orchard.speedwatch import DEFAULT_INTERVAL_S
    return DEFAULT_INTERVAL_S


def _coder_speed(args, env, home, out) -> int:
    """`coder-speed`: the server comes from --endpoint and --model, or from the tier config bringup.toml names."""
    from orchard import speedwatch, tiers
    tier_config = None
    if args.tiers or not (args.endpoint and args.model):
        try:
            cfg = bringup_config.load(find_config(getattr(args, "config", None), env, CHECKOUT, home))
            tier_config = tiers.load(cfg.tiers)
        except (NoConfig, bringup_config.BringupConfigError, tiers.TierConfigError) as exc:
            print(f"note: no tier config read ({exc})", file=out)
    return speedwatch.run(args, out=out, home=home, tier_config=tier_config)


def _kernel_probe(args, env, home, out) -> int:
    """`kernel-probe`: the coder comes from --endpoint and --model, or else from the tier config that bringup.toml
    names. No config is needed when both flags are given; with neither flags nor a config the probe refuses."""
    from orchard import kernelprobe, tiers
    if not args.shared_python:
        args.shared_python = str(home / ".tenstorrent-venv" / "bin" / "python")
    tier_config = None
    if not (args.endpoint and args.model):
        try:
            cfg = bringup_config.load(find_config(getattr(args, "config", None), env, CHECKOUT, home))
            tier_config = tiers.load(cfg.tiers)
        except (NoConfig, bringup_config.BringupConfigError, tiers.TierConfigError) as exc:
            print(f"note: no tier config read ({exc})", file=out)
    from orchard import speedwatch
    return kernelprobe.main(args, out=out, tier_config=tier_config, speed_check=speedwatch.probe_check(home))


def _download_meter(model_id: str, hub, hf_home, out, style, quiet: bool):
    """The progress line around one download (orchard/dlprogress.py); nothing when `--quiet` is given."""
    if quiet:
        return contextlib.nullcontext()
    return dlprogress.DownloadMeter(
        model_id, blob_dir=fetch.blob_dir(hf_home, model_id), snapshot=fetch.snapshot_dir(hf_home, model_id, hub.sha),
        files=hub.files, total_bytes=hub.total_bytes, out=out, style=style)


def _refuse(message: str) -> int:
    print(f"refused: {message}", file=sys.stderr)
    return EXIT_REFUSED


def _installed(sig) -> list:
    return list(sig.installed_bundles()) if sig.installed_bundles else []


def _downloaded_audio_problem(snapshot: Path) -> str | None:
    """After a download for an audio run: why the snapshot is not one stage 0's audio branch accepts (its config, its text encoder's
    config), or None. The preflight judged the model audio from the Hub's lists; this checks the files that arrived."""
    why = preflight.snapshot_audio_problem(snapshot)
    if why:
        return why
    missing = preflight.missing_audio_components(Path(snapshot))
    return f"missing {', '.join(missing[:3])}" if missing else None


def audio_base_decision(arg: str, model: str, sig) -> tuple[str, str | None]:
    """(state, a reason to refuse `--base <arg>` or None). `state` is "audio", "text" or "unknown" (the Hub could not say, so nothing
    shows whether the model is audio). An audio model (orchard/audio_facts.py) is compared with its base repository directly and no bundle
    serves audio, so `--base` has to name a model id. For an audio or unknown model, a name that an installed bundle has, or that
    `tt model search` lists as a bundle, is refused before anything is installed, and so is a search that fails: a failed search is an
    unknown, never "no bundle", and the caller must not install anything on the strength of it. The caller passes `pull=False` to
    resolve_base for any state but "text", so a second search cannot lead to an install either."""
    try:
        hub, _ = sig.hub_info(model)
        local = sig.local_snapshot(model)
    except Exception:
        hub, local = None, None
    state = "unknown" if hub is None else ("audio" if preflight.audio_evidence_for(hub, local) is not None else "text")
    if state == "text":
        return state, None
    inst = _installed(sig)
    if any(b.name == arg for b in inst):
        named = True
    elif any(b.weights_repo == arg for b in inst) or not sig.search_bundles:
        named = False                    # a model id an installed bundle serves, or no search to ask: nothing here can be pulled
    else:
        try:
            found = sig.search_bundles(nearest.family_query(arg))
        except Exception:
            found = None
        if not isinstance(found, list):
            return state, (f"the bundle search failed, so it is not known whether --base {arg} names a bundle, and nothing is installed on a "
                           "guess. Run the same command again, or name the base as a model id")
        named = arg in {c.get("name") for c in found if isinstance(c, dict)}
    if not named:
        return state, None
    what = "an audio model" if state == "audio" else "a model the Hub could not be asked about just now"
    return state, (f"{model} is {what}, and --base {arg} names a bundle. No bundle serves an audio model; stage 0 compares the two "
                   "repositories directly. Name the base as a model id instead (the card's base_model, for example "
                   "stabilityai/stable-audio-3-small-music-base)" + ("" if state == "audio" else ". If the model is not audio, run again once the Hub answers"))


def resolve_base(arg: str, sig, puller, out, style, *, dry_run: bool = False, pull: bool = True) -> tuple[str | None, str | None]:
    """(the weights repo to base the run on, an error text). `arg` is an installed bundle, a model id that
    an installed bundle serves, a published bundle (installed here with `tt-model pull`), or any other
    model id, which the base check then judges. With `pull=False` nothing is searched or installed: an installed bundle's name gives its
    weights repo and anything else is taken as a model id."""
    inst = _installed(sig)
    for b in inst:
        if b.name == arg:
            return b.weights_repo, None
    if any(b.weights_repo == arg for b in inst) or dry_run or not pull:
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


def _run_start_inputs(run_dir: Path) -> dict | None:
    """The inputs in the run's run_start entry, or None when there is none: no ledger, an empty one, or one whose first
    append failed so it holds no run_start (a ledger that cannot be read counts as started, with nothing recorded). Such a run never started, so there is nothing recorded to keep."""
    from orchard import ledger as ledger_mod
    try:
        entries = ledger_mod.read_entries(run_dir / "ledger.jsonl")
    except (OSError, ledger_mod.LedgerCorrupt):
        return {}               # a ledger that cannot be read belongs to a run that may have started: nothing is added to it
    for e in entries:
        if e.get("event") == "run_start":
            inputs = e.get("data", {}).get("inputs")
            return inputs if isinstance(inputs, dict) else {}
    return None


def _recorded_stock_params(run_dir: Path) -> str | None:
    """The `stock_params` input in the run's run_start entry, or None."""
    value = (_run_start_inputs(run_dir) or {}).get("stock_params")
    return value if isinstance(value, str) and value else None


def pick_base(candidates: list[dict], model: str, base: str, *, out, ask=input, inferred: bool = False) -> str | None:
    """Ask the operator which bundle to base the run on. Returns its name, or None for none. For a base the
    research inferred, the pick is the operator's confirmation."""
    if inferred:
        print(f"║  {model}'s card names no base; research inferred {base}. Confirm a bundle to base the run on:",
              file=out)
    else:
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
    if (check.data or {}).get("audio"):
        return              # the bundle help does not apply: an audio model needs no bundle, and its row says what it needs
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


UI_RESEARCH_TTL_S = 300
UI_RESEARCH_MEMO_MAX = 32       # research answers the UI keeps (one per model, base and research setting)


class _ResearchMemo:
    """The UI's research answers, each kept for UI_RESEARCH_TTL_S. The server answers requests on threads, so
    every read and write holds the lock; storing drops expired answers and then the oldest beyond
    UI_RESEARCH_MEMO_MAX. Two requests that miss at the same moment may both ask; that costs a repeat, not an
    error."""

    def __init__(self):
        self._items: dict = {}          # key -> (stored at, report), oldest first
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def get(self, key):
        with self._lock:
            hit = self._items.get(key)
            return hit[1] if hit and time.monotonic() - hit[0] < UI_RESEARCH_TTL_S else None

    def put(self, key, report) -> None:
        now = time.monotonic()
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = (now, report)
            for old in [k for k, (at, _) in self._items.items() if now - at >= UI_RESEARCH_TTL_S]:
                del self._items[old]
            while len(self._items) > UI_RESEARCH_MEMO_MAX:
                del self._items[next(iter(self._items))]


def _ui_preflight_for():
    """The `preflight_for` the UI server uses. The server builds the preflight callable again for every request
    (and for every mode), so the research memo is made once here and handed to each callable."""
    memo = _ResearchMemo()
    return lambda cfg: _ui_preflight(cfg, memo)


def _ui_preflight(cfg, memo=None):
    """The preflight the UI runs: the same checks as `bringup --dry-run`, with `--base` resolved without
    installing anything (an install happens only when the run itself starts). The base-model research runs
    as in a dry run: read-only requests, no cache and no file. The page asks again after every edit of the
    form, so one answer is kept for UI_RESEARCH_TTL_S per model, base and option; GitHub allows 60
    unauthenticated requests an hour and one research run makes about 6."""
    from orchard.research import Options
    memo = _ResearchMemo() if memo is None else memo

    def run(model, base, research=None):
        options = Options(cache_dir=None, offline=research == "offline",
                          no_github=research in ("offline", "no-github"))
        sig = preflight.default_signals(cfg, research_options=options)
        if sig.research is not None:
            fresh = sig.research

            def remembered(model_id, hub, override, inst):
                key = (model_id, override, research)
                hit = memo.get(key)
                if hit is not None:
                    return hit
                report = fresh(model_id, hub, override, inst)
                memo.put(key, report)
                return report
            sig = dataclasses.replace(sig, research=remembered)
        override = None
        if base:
            state, why = audio_base_decision(base, model, sig)
            if why:
                return [preflight.Check("base", preflight.BLOCK, why, "nearest-model-missing", data={"audio": True, "candidates": []})]
            override, why = resolve_base(base, sig, None, io.StringIO(), ui.Style("none", False), dry_run=True)
            if why:
                return [preflight.Check("base", preflight.BLOCK, why, "nearest-model-missing")]
        resuming = (bringup_config.run_dir(cfg, model) / "ledger.jsonl").exists()
        return preflight.run_preflight(cfg, model, accept_credentials=True, signals=sig, resuming=resuming,
                                       base_override=override)
    return run


def _toplike_argv(path) -> list[str] | None:
    import shutil
    if path:
        return [path]
    for name in ("tt-toplike", "tt-toplike-tui"):
        found = shutil.which(name)
        if found:
            return [found]
    return None


def _ui_host(args) -> str:
    return args.host or ("0.0.0.0" if args.lan else "127.0.0.1")


def _lan_addresses() -> list[str]:
    """This machine's address on the local network, found by asking the kernel which address a packet to a
    LAN host would leave from (nothing is sent)."""
    import socket
    found = []
    for probe in ("192.168.0.1", "10.0.0.1", "172.16.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                ip = s.getsockname()[0]
        except OSError:
            continue
        if ip not in found and not ip.startswith("127."):
            found.append(ip)
    return found


def _print_telegraph_notice(cfg, env, home, out, style=None, args=None) -> None:
    """Print the Telegraph state notice (orchard/telegraph_notice.py) when the state directory needs a person to act, and nothing
    when it does not. It only looks at file names: nothing is opened or changed."""
    from orchard import telegraph_notice
    notice = telegraph_notice.telegraph_state_notice(getattr(cfg, "telegraph", None), environ=env, home=home,
                                                     flags=telegraph_notice.flags_from_args(args) if args is not None else "")
    if notice is None:
        return
    icon = style.icon("📋") if style is not None else ""
    print(f"{icon}telegraph: {notice.summary}", file=out)
    for line in notice.lines:
        print(f"  {line}", file=out)


def _ui(args, env, home, out) -> int:
    from orchard import tiers, webui
    host = _ui_host(args)
    if host not in tiers.LOCAL_HOSTS and not args.lan:
        return _refuse(f"--host {host} is not a loopback address; the UI listens only on "
                       f"{', '.join(sorted(tiers.LOCAL_HOSTS))} unless you pass --lan. From another computer, "
                       f"forward the port instead: ssh -L {args.port}:localhost:{args.port} <this machine>")
    try:
        path = find_config(getattr(args, "config", None), env, CHECKOUT, home)
        cfg = bringup_config.load(path)
    except (NoConfig, bringup_config.BringupConfigError) as exc:
        return _refuse(str(exc))
    app = webui.WebApp(runs_root=cfg.runs_root, config_path=path, cfg=cfg, preflight_for=_ui_preflight_for(),
                       lan=args.lan, toplike=_toplike_argv(args.toplike))
    try:
        server = webui.make_server(app, host, args.port)
    except (OSError, ValueError) as exc:
        return _refuse(f"cannot listen on {host}:{args.port}: {exc}")
    shown = f"[{host}]" if ":" in host else host
    print(f"tt-orchard ui {__version__}: http://{shown}:{args.port}/  (runs in {cfg.runs_root})", file=out)
    _print_telegraph_notice(cfg, env, home, out, None, args)
    if args.lan:
        for addr in [app.hostname, *_lan_addresses()]:
            print(f"on the network: http://{addr}:{args.port}/", file=out)
        print("NO LOGIN: anyone who can reach this port can pause, abort and start runs. The firewall must allow "
              f"the port (for ufw: sudo ufw allow from <your LAN>/24 to any port {args.port} proto tcp).", file=out)
    else:
        print(f"from another computer: ssh -L {args.port}:localhost:{args.port} {app.hostname}  "
              f"then open http://localhost:{args.port}/  (or start it with --lan)", file=out)
    print("Ctrl-C stops the page. It never stops a run.", file=out)
    out.flush()
    import signal

    def stop(signum, frame):
        raise KeyboardInterrupt
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)                    # tmux kill-session sends SIGHUP: clean up as for Ctrl-C
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stopping.set()                     # open live feeds end; serve_forever has already returned
        app.toplike_close_all()
        server.server_close()
    return EXIT_OK


def _served_the_model(run_dir: Path) -> bool:
    """True when the run's ledger records a passing stage 2. A run that served the model read real weights. The lines are
    read one by one, because a torn or unreadable line must not stop a bring-up from starting."""
    try:
        lines = (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if (isinstance(e, dict) and e.get("event") == "stage_end" and e.get("stage") == 2
                and isinstance(e.get("data"), dict) and e["data"].get("result") == "pass"):
            return True
    return False


def _supervisor_holds_lock(run_dir: Path) -> bool:
    """True when a live supervisor holds the run's ledger lock. Only an existing lock file is opened, so the check
    creates nothing; the lock is dropped again at once."""
    lock = run_dir / "ledger.jsonl.lock"
    try:
        fh = open(lock, "r")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False
    finally:
        fh.close()


def _block_reason(run_dir: Path) -> str:
    """The reason text in the run's blocked.json, or "" when it is missing or unreadable."""
    try:
        return str(json.loads((run_dir / "blocked.json").read_text(encoding="utf-8")).get("reason") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def _block_code(run_dir: Path) -> str:
    """The reason code in the run's blocked.json, or "blocked" when it is missing or unreadable."""
    try:
        return str(json.loads((run_dir / "blocked.json").read_text(encoding="utf-8"))["code"])
    except (OSError, ValueError, KeyError):
        return "blocked"


def main(argv=None, *, env=None, stdout=None, signals=None, supervisor_main=None, fetcher=None,
         chooser=None, puller=None, sleep=None) -> int:
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

    if args.cmd == "package":
        from orchard import package_cmd
        return package_cmd.main(args, out=out, env=env)
    if args.cmd == "package-card":
        from orchard import package_cmd
        return package_cmd.card_main(args, out=out)
    if args.cmd == "telegraph":
        from orchard import telegraph_cmd
        return telegraph_cmd.main(args, out=out, environ=env, home=home)
    if args.cmd == "db":
        from orchard import db_cmd
        return db_cmd.main(args, out=out, environ=env, home=home)
    if args.cmd == "spend":
        from orchard import spend_cmd
        return spend_cmd.main(args, out=out, environ=env, home=home)
    if args.cmd == "reported":
        from orchard import reported_cmd
        return reported_cmd.main(args, out=out)
    if args.cmd == "suite-control":
        from orchard import suite_control
        return suite_control.main([args.action, *args.rest], out=out)
    if args.cmd == "ttlang-consistency":
        from orchard import ttlang_consistency
        return ttlang_consistency.main(args, out=out)
    if args.cmd == "kernel-env":
        from orchard import ttlang_env
        return ttlang_env.main(args, out=out)
    if args.cmd == "audio-reference-env":
        from orchard import audio_reference_env
        return audio_reference_env.main(args, env=audio_reference_env.env_from_args(args, environ=env, home=home), out=out)
    if args.cmd == "audio-package":
        from orchard import audio_package
        return audio_package.main(args, out=out)
    if args.cmd == "demo-env":
        from orchard import demo_env
        return demo_env.main(args, env=demo_env.env_from_args(args, environ=env, home=home), out=out)
    if args.cmd == "inventory":
        return _inventory(args, env, home, out)
    if args.cmd == "kernel-probe":
        return _kernel_probe(args, env, home, out)
    if args.cmd == "coder-speed":
        return _coder_speed(args, env, home, out)
    if args.cmd == "ui":
        return _ui(args, env, home, out)
    if args.cmd == "caches":
        from orchard import caches
        try:
            cfg, _ = config()
        except (NoConfig, bringup_config.BringupConfigError) as exc:
            return _refuse(str(exc))
        return caches.main(cfg=cfg, args=args, say=lambda line: print(line, file=out))
    if args.cmd == "coder":
        from orchard import kept_coder
        try:
            cfg, _ = config()
        except (NoConfig, bringup_config.BringupConfigError) as exc:
            return _refuse(str(exc))
        return kept_coder.main(cfg=cfg, args=args, say=lambda line: print(line, file=out))
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
        cfg = bringup_config.for_mode(cfg, args.mode or cfg.mode)
        run_dir = Path(args.run_dir) if args.run_dir else bringup_config.run_dir(cfg, args.model)
    except (NoConfig, bringup_config.BringupConfigError) as exc:
        return _refuse(str(exc))
    resuming = (run_dir / "ledger.jsonl").exists()
    print(f"{style.icon('🧭')}mode: " + (f"lab ({cfg.lab.host}): hardware tests run there; the coder stays here"
                                         if cfg.mode == "lab" else "local: everything runs on this box"), file=out)
    _print_telegraph_notice(cfg, env, home, out, style, args)         # once, before the checks; it warns and never blocks the run
    from orchard.research import Options
    from orchard.research import types as research_types
    def status(text: str) -> None:
        print(f"{style.icon('🔎')}{text}...", file=out, flush=True)

    progress = None if args.quiet else status
    research_options = Options(cache_dir=None if args.dry_run else run_dir / "research" / "cache",
                               offline=args.offline, no_github=args.no_github, refresh=args.refresh_research,
                               progress=progress)
    sig = signals or preflight.default_signals(cfg, research_options=research_options)
    override = None
    if args.base:
        state, why = audio_base_decision(args.base, args.model, sig)
        if why:
            return _refuse(why)
        override, why = resolve_base(args.base, sig, puller, out, style, dry_run=args.dry_run, pull=state == "text")
        if why:
            return _refuse(why)

    run_started = resuming and _run_start_inputs(run_dir) is not None       # a ledger with no run_start has nothing to keep
    recorded_stock = _recorded_stock_params(run_dir) if run_started else None

    def preflight_page():
        got = preflight.run_preflight(cfg, args.model, accept_credentials=not args.refuse_credentials_visible,
                                      signals=sig, resuming=resuming, base_override=override,
                                      accept_inferred_base=args.accept_inferred_base, progress=progress,
                                      stock_recorded=recorded_stock)
        print(orchard_view.render_preflight(args.model, run_dir, resuming, cfg_path, got, style), file=out)
        return got

    checks = preflight_page()
    base_check = next(c for c in checks if c.name == "base")
    if (base_check.status == preflight.BLOCK and (base_check.data or {}).get("candidates") and not args.base
            and not args.dry_run):
        if chooser is None and sys.stdin.isatty() and ui.stream_is_tty(out):
            chooser = lambda c, m, b: pick_base(c, m, b, out=out,                    # noqa: E731
                                                inferred=bool(base_check.data.get("inferred")))
        pick = chooser(base_check.data["candidates"], args.model, base_check.data["base"]) if chooser else None
        if pick:
            override, why = resolve_base(pick, sig, puller, out, style)
            if why:
                return _refuse(why)
            checks = preflight_page()
    stops = preflight.blocked(checks)
    if stops and args.prune_caches and all(c.reason == "disk-full" for c in stops):
        short = preflight.prunable_shortfall(next(c for c in stops if c.name == "disk"))
        if short:
            ok, _gone = tensor_caches.prune_to_fit(cfg, short, say=lambda m: print(f"{style.icon('🧹')}{m}", file=out),
                                            dry_run=args.dry_run)
            if ok and not args.dry_run:
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
    if local is None and resuming and not args.dry_run and not _served_the_model(run_dir):
        # The weights are about to be downloaded, so an earlier run read a snapshot with none (a run that passed stage 2 did not). Its stage 0 verdict (a full port,
        # for bytkim/Qwen3.8-27B-pi) and stage 1 reference describe that empty snapshot, and a retry would keep both.
        # The directory is moved, not deleted, and never while a supervisor still holds its ledger lock.
        if _supervisor_holds_lock(run_dir):
            return _refuse(f"the run in {run_dir} has a live supervisor (it holds ledger.jsonl.lock) and its weights are "
                           "not in the cache. Stop that run first, or wait for it to end")
        aside = run_dir.with_name(f"{run_dir.name}.stale-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
        run_dir.rename(aside)
        resuming = False
        print(f"{style.icon('🍂')}the earlier run read a snapshot with no weights; set aside as {aside}", file=out)
    base_data = next(c for c in checks if c.name == "base").data or {}
    base_hub, base_snapshot = None, base_data.get("snapshot")
    if base_data.get("needs_fetch"):
        base_hub, why = sig.hub_info(base_data["base"])
        if base_hub is None:
            return _refuse(f"cannot look up the base model {base_data['base']} on the hub: {why}")
        base_snapshot = fetch.snapshot_dir(hf_home, base_data["base"], base_hub.sha)
    candidates_path = None
    # The report is written only after the preflight passed (a blocked run returned above) and never on a
    # dry run. The research cache (<run dir>/research/cache) is different: the fetcher fills it during the
    # preflight, before anyone knows the preflight passes, so a blocked run can leave that directory behind. The model id is checked again here because --run-dir skips the slug check that guards the path.
    if base_data.get("report") and not args.dry_run and research_types.valid_id(args.model):
        from orchard.research import report as research_report
        research_report.write(run_dir, base_data["report"])
        # Stage 0 reads the candidates-only view. The full report (base-candidates.json) holds text from the
        # internet and stays for the operator.
        candidates_path = run_dir / research_report.RESEARCH_DIR / research_report.STAGE0_FILE
    # A model tt-transformers supports takes the stock-runtime path: the supervisor reads this input and runs stage 0 as code
    # (orchard/stock_runtime.py). No flag is involved; the base row found the support.
    stock = base_data.get("stock_runtime")
    stock_dir = stock["params_dir"] if isinstance(stock, dict) and stock.get("params_dir") else None
    if run_started and resuming:       # `resuming` is false again when the earlier run was set aside above
        # A resume keeps the run's recorded choice. The supervisor picks the stock-runtime stages from the run_start entry, so a
        # preflight that now selects a bundle (or finds the parameters) must not change the input stage 2's draft reads.
        stock_dir = recorded_stock
    inputs = {"model": str(snapshot), **({"base": str(base_snapshot)} if base_snapshot else {}),
              **({"stock_params": stock_dir} if stock_dir else {}),
              **({"base_candidates": str(candidates_path)} if candidates_path else {})}
    argv_run = bringup_config.supervisor_argv(cfg, args.model, run_dir, inputs=inputs,
                                              accept_credentials=not args.refuse_credentials_visible,
                                              graft_trial=not args.no_graft_trial,
                                              resume_aborted=args.resume_aborted, sampling=args.sampling)
    command = "python3 -m orchard.supervisor " + shlex.join(argv_run)
    if args.dry_run:
        print(f"║  command  {command}\n╚══", file=out)
        return EXIT_OK

    if local is None:
        print(f"{style.icon('🌱')}fetching {args.model} at {hub.sha[:8]} ({hub.total_bytes / 1e9:.1f} GB) into "
              f"{hf_home}", file=out)
        try:
            with _download_meter(args.model, hub, hf_home, out, style, args.quiet):
                snapshot = Path(fetcher(args.model, hub.sha, hf_home, files=hub.files))
        except fetch.FetchError as exc:
            return _refuse(str(exc))
        if base_data.get("audio") and (why := _downloaded_audio_problem(snapshot)):
            return _refuse(f"the downloaded {args.model} is not an audio snapshot that stage 0 accepts: {why}")
    if base_hub is not None:
        print(f"{style.icon('🌱')}fetching the base model {base_data['base']} at {base_hub.sha[:8]} "
              f"({base_hub.total_bytes / 1e9:.1f} GB) into {hf_home}", file=out)
        try:
            with _download_meter(base_data["base"], base_hub, hf_home, out, style, args.quiet):
                got = fetcher(base_data["base"], base_hub.sha, hf_home, files=base_hub.files)
        except fetch.FetchError as exc:
            return _refuse(str(exc))
        if base_data.get("audio") and (why := _downloaded_audio_problem(Path(got))):
            return _refuse(f"the downloaded base {base_data['base']} is not an audio snapshot that stage 0 accepts: {why}")
    narrator = None
    if not args.quiet:
        # An existing run shows its last few entries first, so a resume says where it left off.
        narrator = narrate.Narrator(run_dir, style, out, replay=6 if resuming else False)
        narrator.start()
    retried = 0
    try:
        code = supervisor_main(argv_run)
        while code == EXIT_BLOCKED and retried < args.auto_retries:
            reason = _block_code(run_dir)
            if not blocked.retryable(reason, _block_reason(run_dir)):
                break
            retried += 1
            print(f"{style.icon('🔁')}blocked: {reason}. Retrying automatically ({retried} of {args.auto_retries}) "
                  f"after {AUTO_RETRY_WAIT_S:.0f} s.", file=out)
            (sleep or time.sleep)(AUTO_RETRY_WAIT_S)
            code = supervisor_main(argv_run)
    finally:
        if narrator is not None:
            narrator.stop()
    if retried and code == EXIT_BLOCKED:
        print(f"Still blocked after {retried} automatic retries.", file=out)
    where = run_dir / "stages" / "8" / "bundle"
    if code == EXIT_OK:
        print(lexicon.closing_line("ready-for-operator-review", style, where=str(where)), file=out)
    elif code == EXIT_ABORTED:
        print(lexicon.closing_line("aborted", style, where=str(run_dir)), file=out)
    elif code == EXIT_BLOCKED:
        reason = _block_code(run_dir)
        print(lexicon.frost_line(reason, style) + f" See {run_dir / 'BLOCKED.md'}. Run the same command "
              "again to retry.", file=out)
        _print_block_digest(run_dir, reason, args.model, out)
    elif code == EXIT_ERROR:
        print(f"{style.icon('⛈')}the run stopped on an error and the hardware was released. Run the same "
              "command again to resume.", file=out)
    return code


if __name__ == "__main__":
    sys.exit(main())
