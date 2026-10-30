# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The page's pure functions (orchard/web/format.js), run under Node with no browser (issue 31).

format.js holds the formatters, the chip and ledger tone rules, the file grouping and the URL-hash router. app.js uses them
through the shared page scope; this test loads the file with `require`. The tests skip when `node` is not installed.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "orchard" / "web"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def js(expr: str):
    """Evaluate a JavaScript expression with the module loaded as F; return its JSON value."""
    script = f"const F = require({json.dumps(str(WEB / 'format.js'))}); process.stdout.write(JSON.stringify({expr}));"
    out = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout) if out.stdout else None


def test_usd_never_shows_a_real_cost_as_zero():
    assert js('[F.usd(0), F.usd(0.00001), F.usd(0.0042), F.usd(1.239), F.usd(null)]') == [
        "$0.00", "<$0.0001", "$0.0042", "$1.24", "$0.00"]


def test_spend_text_counts_an_unknown_cost_apart():
    assert js('[F.spendText({cost_usd: 0, unknown_cost: 2}), F.spendText({cost_usd: 1.5, unknown_cost: 2}),'
              ' F.spendText({cost_usd: 1.5})]') == ["unknown cost", "$1.50 + 2 unknown", "$1.50"]


def test_plural_dur_and_pct():
    assert js('[F.plural(1, "run"), F.plural(2, "run"), F.plural(0, "run")]') == ["1 run", "2 runs", "0 runs"]
    assert js('[F.dur(null), F.dur(5), F.dur(65), F.dur(3725), F.dur(0)]') == ["", "5s", "1m 05s", "1h 02m", "0s"]
    assert js('[F.pct(null), F.pct(undefined), F.pct(0.456), F.pct(0)]') == ["–", "–", "46%", "0%"]


def test_ago_takes_the_clock_as_an_argument():
    now = "Date.parse('2026-10-29T12:00:00Z')"
    assert js(f'[F.ago("2026-10-29T11:59:30Z", {now}), F.ago("2026-10-29T11:30:00Z", {now}),'
              f' F.ago("2026-10-29T09:15:00Z", {now}), F.ago("2026-10-27T12:00:00Z", {now}),'
              f' F.ago("2026-10-29T12:05:00Z", {now}), F.ago("garbage", {now}), F.ago(null, {now})]') == [
        "30s ago", "30 min ago", "2 h 45 min ago", "2 d ago", "0s ago", "", ""]


def test_chip_tone_treats_an_orchard_lease_as_held():
    assert js('[F.chipTone({state:"FREE"}), F.chipTone({state:"STALE"}), F.chipTone({state:"BUSY-UNTRACKED"}),'
              ' F.chipTone({state:"HELD-FOREIGN", owner:"orchard:run1"}), F.chipTone({state:"HELD-FOREIGN", owner:"claude:x"}),'
              ' F.chipTone({state:"HELD-FOREIGN"})]') == ["good", "warn", "alarm", "accent", "alarm", "alarm"]


def test_ledger_tone_and_file_group():
    assert js('[F.ledgerTone({event:"stage_end",kind:"pass"}), F.ledgerTone({event:"stage_end",kind:"fail"}),'
              ' F.ledgerTone({event:"x",kind:"blocked"}), F.ledgerTone({event:"x",kind:"abort"}), F.ledgerTone({event:"note"})]'
              ) == ["good", "bad", "bad", "bad", ""]
    assert js('["stages/8/bundle/a.md","stages/2/evidence/x","stages/3/tests/t","stages/3/result.json","ledger.jsonl"]'
              '.map(F.fileGroup)') == ["Operator bundle", "Evidence", "Evidence", "Stage results", "Run"]


def test_route_of_hash_accepts_the_old_and_new_names():
    assert js('["#machine","#hardware","#models","#settings","#telegraph","#run/a%20b","#run/","","#other"].map(F.routeOf)') == [
        {"view": "hardware"}, {"view": "hardware"}, {"view": "settings"}, {"view": "settings"}, {"view": "telegraph"},
        {"run": "a b"}, None, None, None]


def test_cost_cap_sentence_follows_the_caps_that_are_set():
    t_cost = '{caps:[{id:"cost"}], config:{per_problem:{cost_usd:null}}}'
    t_pp = '{caps:[], config:{per_problem:{cost_usd:1}}}'
    t_none = '{caps:[], config:{per_problem:{cost_usd:null}}}'
    assert js(f'[F.costCapSentence({t_cost}, true), F.costCapSentence({t_pp}, false), F.costCapSentence({t_none}, true)]') == [
        "A cost cap refuses new calls while any exist.",
        "a per-problem cost cap refuses new calls for a problem while it has any",
        "No cost cap is set, so new calls still go out."]


def test_tg_cap_text_shows_a_blocked_cap_as_blocked():
    assert js('[F.tgCapText({unit:"usd",used:1,limit:5,percent:20,blocked:false}),'
              ' F.tgCapText({unit:"calls",used:3,limit:10,percent:30,blocked:true})]') == [
        "$1.00 of $5.00 (20%)", "3 known of 10: blocked, a call has no known cost"]


def test_app_js_no_longer_defines_what_format_js_owns():
    app = (WEB / "app.js").read_text(encoding="utf-8")
    for name in ("usd", "plural", "spendText", "tgCapText", "costCapSentence", "ledgerTone", "fileGroup", "chipTone", "dur", "ago"):
        assert f"function {name}(" not in app, name
    assert "/static/format.js" in (WEB / "index.html").read_text(encoding="utf-8")
