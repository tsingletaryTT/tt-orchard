// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// The page's pure functions: formatters, the chip and ledger tone rules, file grouping and the URL-hash router.
// Nothing here touches the document, the network or storage, so tests load this file under Node (tests/test_web_js_pure.py).
// In the page it is a plain script loaded before app.js; its top-level names are shared with app.js. `ago` takes the clock as
// an optional argument so a test can fix the time.
"use strict";

const CHIP_TONE = { FREE: "good", CLAIMED: "accent", HELD: "accent", STALE: "warn" };
function parseTs(ts) {
  if (!ts) return null;
  const t = Date.parse(ts);
  return Number.isNaN(t) ? null : t;
}
function ago(ts, now = Date.now()) {
  const t = typeof ts === "number" ? ts : parseTs(ts);
  if (t === null) return "";
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min ago`;
  return `${Math.floor(s / 86400)} d ago`;
}
function dur(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const s = Math.round(seconds);
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  if (hh) return `${hh}h ${String(mm).padStart(2, "0")}m`;
  if (mm) return `${mm}m ${String(ss).padStart(2, "0")}s`;
  return `${ss}s`;
}
// Dollars with enough digits that a small cost never shows as zero (the same rule as telegraph_view.usd).
function usd(value) {
  const v = Number(value) || 0;
  if (v === 0) return "$0.00";
  if (v < 0.0001) return "<$0.0001";                // four decimals would show $0.0000 for a real cost
  return v < 0.01 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`;
}
function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }
function spendText(s) {
  // The same rule as telegraph_view._spend_text: an unknown cost is counted apart and never shown as zero dollars.
  const unknown = s.unknown_cost || 0;
  if (unknown && !s.cost_usd) return "unknown cost";
  return unknown ? `${usd(s.cost_usd)} + ${unknown} unknown` : usd(s.cost_usd);
}
// What an unknown cost does, which depends on the cost caps that are set: a global cost cap refuses every new call, a per-problem
// cost cap refuses new calls for a problem that has one, and with neither the calls still go out.
function costCapSentence(t, long) {
  if (t.caps.some((c) => c.id === "cost")) return long ? "A cost cap refuses new calls while any exist." : "a cost cap refuses new calls while any exist";
  if (t.config.per_problem.cost_usd !== null) {
    return long ? "A per-problem cost cap refuses new calls for a problem while it has any." : "a per-problem cost cap refuses new calls for a problem while it has any";
  }
  return long ? "No cost cap is set, so new calls still go out." : "no cost cap is set, so calls still go out";
}
function tgCapText(c) {
  const used = c.unit === "usd" ? usd(c.used) : String(c.used), limit = c.unit === "usd" ? usd(c.limit) : String(c.limit);
  return c.blocked ? `${used} known of ${limit}: blocked, a call has no known cost` : `${used} of ${limit} (${c.percent}%)`;
}
// gozer calls a chip HELD-FOREIGN when a process outside the lease owner's tree has it open; under an orchard lease
// that is the coder's container serving, which is expected, so it is drawn as held, not as an alarm.
function chipTone(c) {
  if (c.state === "HELD-FOREIGN" && (c.owner || "").startsWith("orchard:")) return CHIP_TONE.HELD;
  return CHIP_TONE[c.state] || "alarm";
}
function ledgerTone(r) {
  if (r.event === "stage_end" && r.kind === "pass") return "good";
  if ((r.event === "stage_end" && r.kind === "fail") || r.kind === "blocked" || r.kind === "abort") return "bad";
  return "";
}
function fileGroup(path) {
  if (path.startsWith("stages/8/bundle/")) return "Operator bundle";
  if (path.includes("/evidence/") || path.includes("/tests/")) return "Evidence";
  if (path.startsWith("stages/")) return "Stage results";
  return "Run";
}
const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const where = (host) => (host ? `on ${host}` : "on this box");

// Where a URL hash points: {view} for a view, {run} for a run page, null for anything else. #machine and #models are the names
// now; #hardware and #settings are what earlier links and bookmarks carry.
function routeOf(hash) {
  if (hash === "#machine" || hash === "#hardware") return { view: "hardware" };
  if (hash === "#models" || hash === "#settings") return { view: "settings" };
  if (hash === "#telegraph") return { view: "telegraph" };
  const m = /^#run\/(.+)$/.exec(hash || "");
  if (!m) return null;
  try { return { run: decodeURIComponent(m[1]) }; } catch { return null; }      // a malformed escape in a pasted link
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { CHIP_TONE, parseTs, ago, dur, usd, plural, spendText, costCapSentence, tgCapText, chipTone, ledgerTone,
    fileGroup, pct, where, routeOf };
}
