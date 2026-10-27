// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// The tt-orchard page. Plain JavaScript, no build step, nothing loaded from another host.
// Every piece of text from the server goes into the page with textContent, never as HTML.
"use strict";

const POLL_RUNS_MS = 5000;
const POLL_RUN_MS = 3000;
const FEED_MAX = 2000;
const STALE_MS = 15000;
const HIDDEN_POLL_MS = 30000;

const S = {
  meta: null, runs: [], launches: [], machine: null, selected: null, detail: null,
  tab: "feed", source: null, feedSeen: new Set(), feedFilter: "", follow: true,
  ledgerAfter: 0, ledgerRows: [], ledgerFilter: "", ledgerFirst: {}, file: null, lastOk: 0, startingRun: null,
  scene: null, view: "orchard", hw: null, health: null, leases: null, tgStart: Date.now(), tgOk: 0, tgPending: false,
};

// ---- small helpers ------------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

// Rebuild `el` only when what it shows has changed. Polling every few seconds must not take keyboard focus
// or a half-finished click away from a control that is still the same.
function changed(el, value) {
  const key = JSON.stringify(value);
  if (el.dataset.key === key) return false;
  el.dataset.key = key;
  return true;
}

function parseTs(ts) {
  if (!ts) return null;
  const t = Date.parse(ts);
  return Number.isNaN(t) ? null : t;
}

function ago(ts) {
  const t = typeof ts === "number" ? ts : parseTs(ts);
  if (t === null) return "";
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min ago`;
  return `${Math.floor(s / 86400)} d ago`;
}

function clock(ts) {
  const t = parseTs(ts);
  if (t === null) return "";
  return new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

function dur(seconds) {
  if (seconds === null || seconds === undefined) return "";
  const s = Math.round(seconds);
  const hh = Math.floor(s / 3600), mm = Math.floor((s % 3600) / 60), ss = s % 60;
  if (hh) return `${hh}h ${String(mm).padStart(2, "0")}m`;
  if (mm) return `${mm}m ${String(ss).padStart(2, "0")}s`;
  return `${ss}s`;
}

const TONE = { good: "good", warn: "warn", bad: "bad", alarm: "alarm", dim: "dim", accent: "good", title: "good" };

function stateBadge(state) {
  const st = S.meta && S.meta.states[state];
  if (!st) return h("span", { class: "badge", dataset: { tone: state === "unreadable" ? "alarm" : "dim" } },
    ...(state === "unreadable" ? [h("span", { class: "emoji", "aria-hidden": "true" }, "⚠ "), "unreadable"] : [state]));
  return h("span", { class: "badge", dataset: { tone: TONE[st.role] || "dim" }, title: `${st.word} (${state})` },
    h("span", { "aria-hidden": "true", class: "emoji" }, st.emoji),
    h("span", { class: "op-pic", "aria-hidden": "true" }, S.meta.op.states[state] || ""),      // the Operator style's picture; the real name stays beside it
    h("span", { class: "word" }, st.word), h("span", { class: "name" }, state.replace(/-/g, " ")),
    h("span", { class: "real" }, `· ${state}`));      // "name" is the real state, shown instead of the orchard word in operator mode
}

// "stage 1 🧪 soil test · environment and CPU reference". The orchard part sits in its own span so operator mode can drop it.
function stageLine(stage) {
  if (!stage || stage.current === null || stage.current === undefined) return "";
  const s = S.meta && S.meta.stages[stage.current];
  return h("span", {}, `stage ${stage.current} `, s ? h("span", { class: "orchard" }, `${s.emoji} ${s.name} `) : "",
    `· ${stage.name || (s && s.real) || ""}`);
}

function toast(text, tone) {
  const el = h("div", { class: "toast", dataset: { tone: tone || "good" } }, text);
  $("toasts").append(el);
  setTimeout(() => el.remove(), tone === "bad" ? 9000 : 5000);
}

// ---- the API --------------------------------------------------------------------------------------

async function api(path, opts) {
  const init = { headers: {}, cache: "no-store" };
  if (opts && opts.body !== undefined) {
    init.method = "POST";
    init.headers["Content-Type"] = "application/json";
    init.headers["X-Orchard-Token"] = S.meta ? S.meta.token : "";
    init.body = JSON.stringify(opts.body);
  }
  let timer = null;
  if (opts && opts.timeout) {            // an optional limit: the request is aborted and counts as a failure
    const ctl = new AbortController();
    init.signal = ctl.signal;
    timer = setTimeout(() => ctl.abort(), opts.timeout);
  }
  let resp;
  try {
    resp = await fetch(path, init);
  } catch (err) {
    if (timer) clearTimeout(timer);
    setConn("offline");
    throw new Error("The UI server cannot be reached. Is `tt-orchard ui` still running?");
  }
  S.lastOk = Date.now();
  setConn("live");
  const isJson = (resp.headers.get("Content-Type") || "").includes("application/json");
  const data = isJson ? await resp.json() : await resp.text();      // the limit covers the body too
  if (timer) clearTimeout(timer);
  if (!resp.ok) throw new Error((data && data.error) || `${resp.status} ${resp.statusText}`);
  return data;
}

function setConn(state) {
  const el = $("conn");
  el.dataset.state = state;
  el.querySelector(".conn-text").textContent =
    state === "live" ? "Live" : state === "stale" ? "Reconnecting…" : state === "offline" ? "Offline" : "Connecting…";
}

// ---- machine and runs -------------------------------------------------------------------------------

// gozer's chip states: FREE; CLAIMED (leased, no device open yet) and HELD (leased and in use) are normal;
// STALE needs `gozer reconcile`; anything else (HELD-FOREIGN, BUSY-UNTRACKED, ...) is a chip used outside a lease.
const CHIP_TONE = { FREE: "good", CLAIMED: "accent", HELD: "accent", STALE: "warn" };

// gozer calls a chip HELD-FOREIGN when a process outside the lease owner's tree has it open; under an orchard lease
// that is the coder's container serving, which is expected, so it is drawn as held, not as an alarm.
function chipTone(c) {
  if (c.state === "HELD-FOREIGN" && (c.owner || "").startsWith("orchard:")) return CHIP_TONE.HELD;
  return CHIP_TONE[c.state] || "alarm";
}

function boardEl(b, readings) {
  return h("div", { class: "board" },
    h("div", { class: "board-head" }, h("span", { class: "mono", title: b.id }, `board …${b.id.slice(-6)}`), h("span", {}, b.kind)),
    h("div", { class: "chips" }, b.chips.map((c) => h("div", {
      class: "chip", dataset: { tone: chipTone(c) },
      title: [c.bdf, c.owner, c.note].filter(Boolean).join(" · "),
    }, h("strong", {}, `chip ${c.index} · ${c.state}`), h("span", { class: "owner" }, c.owner || c.bdf),
      readings ? h("span", { class: "chip-reading", dataset: { reading: c.bdf } }) : null))));
}

function renderMachine() {
  const m = S.machine;
  if (!m) return;
  // The top bar's chip is drawn on every poll, before the panel's change guard. An outage marks the chip unknown and leaves the
  // panel's cache as it was, so when the machine answers again with the same facts the guard returns early and nothing else
  // would put the chip right.
  const chips = m.boards.flatMap((b) => b.chips);
  const freeNow = chips.filter((c) => c.state === "FREE").length;
  renderMachineChip(m, chips, freeNow);
  if (!changed($("machine"), [m.boards, m.coder, m.cpu_tier, m.disk_free_gb, m.gozer_ok, m.lab])) return;
  const box = clear($("machine"));
  if (!m.gozer_ok) {
    box.append(h("p", { class: "muted" }, "gozer did not answer. Chip leases are unknown; check `gozer status`."));
  }
  for (const b of m.boards) box.append(boardEl(b, true));
  // gozer marks a chip HELD-FOREIGN when a process outside the lease owner's tree has it open. A tt-model
  // container serving under the supervisor's lease is such a process (docs/project-log.md, 2026-10-02), so say so.
  const all = m.boards.flatMap((b) => b.chips);
  const foreign = all.filter((c) => c.state === "HELD-FOREIGN");
  if (foreign.length && foreign.every((c) => c.owner.startsWith("orchard:"))) {
    box.append(h("p", { class: "muted small" }, "HELD-FOREIGN under an orchard lease is expected while a coder " +
      "container serves: gozer counts the container as outside the supervisor's process tree."));
  }
  box.append(h("div", { class: "services" },
    h("span", { class: "svc", dataset: { up: String(m.coder.up) } }, `coder :${m.coder.port ?? "?"} ${m.coder.up ? "up" : "down"}`),
    h("span", { class: "svc", dataset: { up: String(m.cpu_tier.up) } }, `CPU tier :${m.cpu_tier.port} ${m.cpu_tier.up ? "up" : "down"}`),
    h("span", {}, `${m.disk_free_gb} GB free`)));
  if (m.lab) {
    box.append(h("h3", { class: "lab-head" }, `Lab · ${m.lab.host}`));
    if (!m.lab.ok) box.append(h("p", { class: "muted small" }, `The lab did not answer: ${m.lab.error || "no chips listed"}`));
    for (const b of m.lab.boards) box.append(boardEl(b, false));
  }
  const free = all.filter((c) => c.state === "FREE").length;
  $("machine-note").textContent = all.length ? `${free} of ${all.length} chips free` : "";
  renderReadings();
}

// The top bar's one-line answer to "can I start something?": the same chips and coder the rail panel lists, so it never
// disagrees with it. The tone is a coloured dot and the words say how many chips are free and whether the coder is up.
function renderMachineChip(m, all, free) {
  const chip = $("machine-chip");
  if (!m || !m.gozer_ok || !all.length) {
    chip.textContent = "Chips: unknown";
    chip.dataset.tone = "bad";
    return;
  }
  const stale = all.some((c) => c.state === "STALE" || c.state === "BUSY-UNTRACKED");
  chip.textContent = free === all.length ? `${free} of ${all.length} chips free`
    : `${free} of ${all.length} chips free · coder ${m.coder.up ? "up" : "down"}`;
  chip.dataset.tone = free === all.length ? "good" : stale ? "bad" : "warn";
}

// ---- telegraph: the spend meter in the top bar (always on screen) and the Telegraph view ----------------------------
// The server builds the meter's words (orchard/telegraph_view.py). The page only shows them. The meter is redrawn on every
// poll, outside any change guard, so an outage cannot leave a stale amount on screen: after STALE_MS without an answer it says the
// spend is unknown. The view only reads; nothing on it sends a request to the helper.

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

// While the spend cannot be read, the Telegraph view keeps its last numbers but says they are old, so a screen of figures is never
// taken for the current spend.
function markTelegraphViewStale(stale, age) {
  const note = $("tg-stale"), view = $("telegraph-view");
  note.hidden = !stale;
  if (stale) {
    view.dataset.stale = "true";
    note.textContent = S.tgOk
      ? `The page cannot read the helper's spend right now. The numbers below were read ${age} s ago and may be out of date.`
      : "The page cannot read the helper's spend right now, and no amount has been read yet.";
  } else {
    delete view.dataset.stale;
    note.textContent = "";
  }
}

function renderSpendMeter(t) {
  const el = $("spend-meter");
  if (!t) {
    // The current spend cannot be read. The last amount that was read stays visible, marked as old, so an outage never hides
    // a spend that was already high; before any amount was read, nothing is invented.
    const last = S.telegraph && S.telegraph.meter ? S.telegraph.meter.text.replace(/^Helper: /, "") : null;
    const age = S.tgOk ? Math.round((Date.now() - S.tgOk) / 1000) : null;
    $("spend-text").textContent = last ? `Helper: spend unknown · last known ${last} (${age} s ago)` : "Helper: spend unknown";
    el.dataset.state = "unknown";
    el.dataset.tone = "bad";
    el.title = "The page cannot read the helper's spend right now. Click for the Telegraph view.";
    markTelegraphViewStale(true, age);
    return;
  }
  markTelegraphViewStale(false, null);
  $("spend-text").textContent = t.meter.text;
  el.dataset.state = t.state;
  el.dataset.tone = t.tone;
  el.title = t.meter.title;
}

const POLL_TELEGRAPH_MS = 5000;

// The amount on screen is unknown once it is older than STALE_MS, however that came about.
function markTelegraphStale() {
  if (Date.now() - (S.tgOk || S.tgStart) > STALE_MS) renderSpendMeter(null);
}

// Telegraph has its own timers, so no other request (the run list, the machine, the launches) can stop it: one poll every five
// seconds (every 30 s in a hidden tab) and a separate check every two seconds that the amount is not stale.
function startTelegraphPolling() {
  pollTelegraph();
  let last = Date.now();
  setInterval(() => {
    if (document.hidden && Date.now() - last < HIDDEN_POLL_MS) { markTelegraphStale(); return; }
    last = Date.now();
    pollTelegraph();
  }, POLL_TELEGRAPH_MS);
  setInterval(markTelegraphStale, 2000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) pollTelegraph(); });
}

const TG_TIMEOUT_MS = 25000;           // a poll that has not answered by now is dropped, so a hung server cannot block later polls

async function pollTelegraph() {
  // Staleness is judged first, before any waiting: a request that never answers would never reach a catch block, and the last
  // dollar amount would stay on screen. Every poll tick therefore checks how old the amount is.
  markTelegraphStale();
  if (S.tgPending) return;               // one request at a time
  S.tgPending = true;
  try {
    S.telegraph = await api("/api/telegraph", { timeout: TG_TIMEOUT_MS });
    S.tgOk = Date.now();
    renderSpendMeter(S.telegraph);
    if (S.view === "telegraph") renderTelegraph();
  } catch (err) {
    // the next tick's check above marks the amount unknown once it is old enough
  } finally {
    S.tgPending = false;
  }
}

const TG_BANNER = {
  off: "Telegraph is off. The harness runs without an outside helper. How to turn it on is at the bottom of this page.",
  ok: "Telegraph is on. No cap is near its limit.",
  near: "Telegraph is on, and a cap is at 80 percent or more.",
  capped: "Telegraph is on, and a cap has stopped it.",
  problem: "Telegraph is on, and something needs attention.",
};
const TG_OUTCOME_TONE = { ok: "good", error: "bad", budget: "bad", "no-key": "warn", "cache hit": "dim", unsettled: "warn" };

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

function tgRow(cells, tone) {
  const tr = h("tr", tone ? { dataset: { tone } } : {});
  for (const [text, cls, tip] of cells) tr.append(h("td", { class: cls || null, title: tip || null }, text));
  return tr;
}

function renderTelegraph() {
  const t = S.telegraph;
  if (!t) return;
  const { now, ...rest } = t;                         // the clock changes on every answer; the rest decides whether to redraw
  if (!changed($("telegraph-view"), rest)) return;
  const banner = clear($("tg-banner"));
  banner.dataset.tone = t.tone;
  banner.append(h("strong", {}, TG_BANNER[t.state] || ""));
  if (t.capped) banner.append(h("span", {}, ` Reason: ${t.capped.reason}.`));
  if (t.spend.unknown_cost) {
    banner.append(h("span", {}, ` ${plural(t.spend.unknown_cost, "call")} ${t.spend.unknown_cost === 1 ? "has" : "have"} no known cost. ` +
      costCapSentence(t, true)));
  }
  if (t.spend.unsettled) banner.append(h("span", {}, ` ${plural(t.spend.unsettled, "reservation")} ${t.spend.unsettled === 1 ? "has" : "have"} no recorded result and counts at its worst case.`));
  if (t.error) banner.append(h("span", {}, ` ${t.error}.`));
  if (t.key.state === "missing") banner.append(h("span", {}, ` The key is missing: ${t.key.reason}.`));

  const s = t.summary;
  const num = (k, v, note) => h("div", { class: "tg-num" }, h("span", { class: "k" }, k), h("span", { class: "v" }, v), note ? h("span", { class: "muted small" }, note) : null);
  if (t.error) {
    // Nothing could be read, so no number is shown: a row of zeros would look like a spend of nothing.
    clear($("tg-numbers")).append(num("Spend", "unknown", t.error), num("Calls counted", "unknown", "the ledger was not read"));
  } else clear($("tg-numbers")).append(
    num("Spend", spendText(t.spend), t.spend.unknown_cost ? `${plural(t.spend.unknown_cost, "call")} with no known cost` : "known costs; reservations at their worst case"),
    num("Calls counted", String(t.spend.calls), `${s.calls} sent and recorded · ${s.ok} answered`),
    num("Unsettled", String(t.spend.unsettled), "no result recorded; counted at the worst case"),
    num("Unknown cost", String(t.spend.unknown_cost),
      t.spend.unknown_cost ? costCapSentence(t, false)
        : "every counted call has a known cost"),
    num("Helped", String(s.helped), `${s.not_helped} did not help · ${s.unreported} not reported`),
    num("Cache hits", String(s.cache_hits), "cost nothing"),
    num("Refused", String(s.refused), "a cap or a missing key"),
    num("Tokens", `${s.tokens_in + s.tokens_out}`, `${s.tokens_in} in · ${s.tokens_out} out`),
    num("Time", `${Math.round(s.latency_s)} s`, "waiting for the helper"),
    num("Damaged or unreported", String(t.spend.torn + t.spend.unknown_usage_calls),
      `${plural(t.spend.torn, "torn record")} · ${plural(t.spend.unknown_usage_calls, "call")} with no usage reported`));

  const caps = clear($("tg-caps"));
  for (const c of t.caps) {
    caps.append(h("div", { class: "tg-cap", dataset: { tone: c.tone } },
      h("span", { class: "k" }, `${c.label} cap`), h("span", { class: "v" }, tgCapText(c)),
      h("div", { class: "tg-bar", role: "progressbar", "aria-label": `${c.label} cap used`, "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.min(100, c.percent)) },
        h("span", { style: `width:${Math.min(100, c.percent)}%` }))));
  }
  const pp = t.config.per_problem;
  if (t.enabled) {
    caps.append(h("p", { class: "muted small tg-pp" }, `Each problem may use up to ${plural(pp.calls, "call")}` +
      (pp.tokens !== null ? `, ${pp.tokens} tokens` : "") + (pp.cost_usd !== null ? `, ${usd(pp.cost_usd)}` : "") + ". See By problem."));
  }

  const rec = clear($("tg-recent"));
  for (const r of t.recent) {
    rec.append(tgRow([
      [r.ts === null ? "" : new Date(r.ts * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }), "when"],
      [r.problem, "wrap"], [r.model, "wrap"], [r.held ? "–" : String(r.tokens_in), "num"], [r.held ? "–" : String(r.tokens_out), "num"],
      [r.held ? "–" : r.latency_s.toFixed(1), "num"],
      [r.cost_usd === null ? "unknown" : r.held ? `up to ${usd(r.cost_usd)}` : r.usage_known === false ? `${usd(r.cost_usd)} (${r.cost_basis})` : usd(r.cost_usd), "num"],
      [r.outcome, "outcome", r.reason || null],
      [r.helped === true ? "yes" : r.helped === false ? "no" : r.outcome === "ok" ? "not reported" : "", null]], TG_OUTCOME_TONE[r.outcome] || null));
    if (r.reason) rec.lastChild.lastChild.previousSibling.append(h("div", { class: "tg-reason" }, r.reason));
  }
  $("tg-recent-empty").hidden = t.recent.length > 0;
  $("tg-recent-note").textContent = t.recent.length ? `newest first, up to ${t.recent.length}` : "";

  const bm = clear($("tg-by-model"));
  for (const m of t.by_model) bm.append(tgRow([[m.model, "wrap"], [String(m.calls), "num"], [String(m.tokens), "num"], [spendText({ cost_usd: m.cost_usd, unknown_cost: m.cost_unknown_calls }), "num"]]));
  if (!t.by_model.length) bm.append(tgRow([["No calls yet", "muted"], ["", ""], ["", ""], ["", ""]]));
  const bp = clear($("tg-by-problem"));
  for (const p of t.problems) {
    bp.append(tgRow([[p.name, "wrap"], [String(p.calls), "num"], [String(p.tokens), "num"],
      [spendText({ cost_usd: p.cost_usd, unknown_cost: p.cost_unknown_calls }), "num"],
      [`${p.helped} yes · ${p.not_helped} no`, null], [`${p.percent}%${p.at_cap ? " (at its cap)" : ""}`, "num cap"]], p.at_cap ? "bad" : p.near ? "warn" : null));
  }
  if (t.problems_hidden) bp.append(tgRow([[`${plural(t.problems_hidden, "more problem")} not shown. The state and the closest cap count every problem.`, "muted"], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""]]));
  if (!t.problems.length) bp.append(tgRow([["No calls yet", "muted"], ["", ""], ["", ""], ["", ""], ["", ""], ["", ""]]));

  const c = t.config;
  const keyWords = { present: `present (from ${t.key.source === "file" ? "a file" : "an environment variable"})`,
    missing: `missing: ${t.key.reason || ""}`, "not-needed": "not needed (the endpoint takes no key)", "n/a": "not used while telegraph is off" };
  const dl = clear($("tg-config"));
  for (const [k, v] of [
    ["Enabled", t.enabled ? "yes" : "no"], ["Endpoint host", c.endpoint_host || "none"], ["API style", c.api || "none"],
    ["Asks after", t.enabled ? `${plural(c.after_failures, "repeated failure")}` : "none"],
    ["Limits per call", t.enabled ? `${c.max_output_tokens} output tokens, ${c.max_prompt_chars} prompt characters, ${c.timeout_s} s` : "none"],
    ["Key", keyWords[t.key.state] || t.key.state], ["Ledger folder", c.ledger_dir]]) {
    dl.append(h("div", { class: "tg-kv" }, h("dt", {}, k), h("dd", { class: k === "Ledger folder" ? "mono wrap" : null }, v)));
  }
  const ms = clear($("tg-models"));
  c.models.forEach((m, i) => ms.append(tgRow([[String(i + 1), "num"], [m.name, "wrap"],
    [m.input_per_mtok === null ? "no price" : `$${m.input_per_mtok}`, "num"], [m.output_per_mtok === null ? "no price" : `$${m.output_per_mtok}`, "num"]])));
  if (!c.models.length) ms.append(tgRow([["", ""], ["No models: telegraph is off", "muted"], ["", ""], ["", ""]]));
}

// ---- machine health: the scene's weather and how hard the farmer works --------------------------------

const WEATHER_WORD = { clear: "☀ clear skies", heatwave: "🔥 heatwave", overcast: "☁ overcast", storm: "⛈ storm", unknown: "· weather unknown" };

function renderReadings() {
  const hl = S.health;
  if (!hl) return;
  for (const el of document.querySelectorAll("[data-reading]")) {
    const c = hl.chips[el.dataset.reading];
    el.textContent = c ? [c.temp_c !== null ? `${Math.round(c.temp_c)} °C` : null, c.power_w !== null ? `${Math.round(c.power_w)} W` : null,
      `${Math.round(c.utilization * 100)}% busy`].filter(Boolean).join(" · ") : "";
  }
}

function runUtilization() {
  // While a run is going, the farmer works as hard as the chips its supervisor holds; otherwise the busiest chip.
  const hl = S.health;
  if (!hl) return 0;
  const chips = Object.entries(hl.chips);
  const held = chips.filter(([bdf]) => (hl.leases[bdf] || "").includes("orchard:"));
  const pool = S.detail && ["running", "paused"].includes(S.detail.state) && held.length ? held : chips;
  return pool.length ? Math.max(...pool.map(([, c]) => c.utilization)) : 0;
}

async function pollHealth() {
  try {
    const hl = await api("/api/health");
    const orchard = (ls) => JSON.stringify(Object.entries(ls || {}).filter(([, o]) => (o || "").includes("orchard:")));
    const changed = S.leases && orchard(S.leases) !== orchard(hl.leases);
    S.health = hl;
    S.leases = hl.leases;
    if (S.scene) {
      S.scene.setMachine({ weather: hl.weather.kind, util: runUtilization() });
      if (changed) S.scene.event("lease");          // chips changing hands: off to the shed for tools
    }
    renderReadings();
    if (S.detail) renderSceneCaption(S.detail);
  } catch { /* the runs poll reports a lost connection */ }
}

function dismissedLaunches() {
  try { return JSON.parse(localStorage.getItem("orchard.dismissed") || "[]"); } catch { return []; }
}

function dismissLaunch(started) {
  try { localStorage.setItem("orchard.dismissed", JSON.stringify([...dismissedLaunches(), String(started)].slice(-50))); } catch { /* private window */ }
}

function renderLaunches(force) {
  if (!changed($("launches"), [S.launches.map((l) => [l.run, l.alive, l.has_ledger, l.log_tail]), dismissedLaunches()]) && !force) return;
  const box = clear($("launches"));
  for (const l of S.launches.filter((x) => x.alive && !x.has_ledger)) {
    box.append(h("div", { class: "launch" },
      h("strong", {}, `Starting ${l.model}`), " ", h("span", { class: "muted" }, ago(l.started * 1000)),
      h("pre", {}, l.log_tail || "waiting for output…")));
  }
  const gone = dismissedLaunches();
  for (const l of S.launches.filter((x) => !x.alive && !x.has_ledger && !gone.includes(String(x.started)))) {
    box.append(h("div", { class: "launch" },
      h("strong", {}, `${l.model} did not start`),
      h("button", { type: "button", class: "btn btn-small", onclick: () => { dismissLaunch(l.started); renderLaunches(true); } }, "Dismiss"),
      h("pre", {}, l.log_tail || "(no output)")));
  }
}

// Which run holds the table's single tab stop: the selected one, or the first row when nothing (or a vanished run) is selected.
function rovingStop(r, i) {
  return r.name === S.selected || (i === 0 && !S.runs.some((x) => x.name === S.selected));
}

function renderRuns() {
  const list = $("runs");
  if (!changed(list, [S.selected, S.runs.map((r) => [r.name, r.state, r.model, r.mode, r.stage && r.stage.current, r.stage && r.stage.name, r.last_ts,
    r.launching, Math.floor((Date.now() - (parseTs(r.last_ts) || 0)) / 60000)])])) return;
  clear(list);
  // a poll that redraws the rows must not take the keyboard focus away from the row the person is on
  const focusName = $("runs-body").contains(document.activeElement) ? (document.activeElement.closest("tr") || {}).dataset?.runName : null;
  const body = clear($("runs-body"));
  $("runs-note").textContent = S.runs.length ? `${S.runs.length}` : "";
  if (!S.runs.length) {
    list.append(h("li", { class: "empty-small" },
      h("span", { class: "muted" }, "No runs here yet."),
      h("button", { type: "button", class: "btn btn-small", onclick: openNewRun }, "Start a bring-up")));
    body.append(h("tr", { role: "row" }, h("td", { role: "gridcell", colspan: "4", class: "c-empty" },
      h("span", { class: "muted" }, "No runs here yet. "),
      h("button", { type: "button", class: "btn btn-small", onclick: openNewRun }, "Start a bring-up"))));
    return;
  }
  const total = (S.meta && S.meta.stages ? S.meta.stages.length : 9) || 9;
  S.runs.forEach((r, i) => {
    // each view gets its own nodes: one DOM node can sit in only one place, so sharing them would move the tags out of the list
    const tag = () => (r.mode || "").startsWith("lab") ? h("span", { class: "mode-tag", title: `hardware tests ran on ${r.mode.slice(5, -1)}` }, "lab") : null;
    const first = () => r === S.runs[0] ? h("span", { class: "mode-tag", title: "the page opens this run first: the one in progress, or the most recent" },
      r.state === "running" || r.state === "paused" || r.launching ? "now" : "latest") : null;
    list.append(h("li", { "data-run-row": "1", "data-run-name": r.name }, h("button", {
      type: "button", class: "run-item", "aria-current": String(r.name === S.selected),
      onclick: () => select(r.name),
    }, stateBadge(r.state),
      h("span", { class: "model" }, r.model || r.name, tag(), first()),
      h("span", { class: "meta mono" }, r.name),
      h("span", { class: "meta" },
        h("span", {}, r.stage && r.stage.current !== null ? `stage ${r.stage.current}` : ""),
        h("span", {}, r.last_ts ? ago(r.last_ts) : ""),
        r.launching ? h("span", {}, "starting…") : null))));

    // the Operator style's table row: state, model, stage, age. The full id, the stage name and the exact time are tooltips.
    const full = r.model || r.name;
    const tip = full === r.name ? full : `${full}\n${r.name}`;
    const cur = r.stage && r.stage.current !== null && r.stage.current !== undefined ? r.stage.current : null;
    const when = parseTs(r.last_ts);
    const selected = r.name === S.selected;
    const row = h("tr", {
      role: "row", "data-run-row": "1", "data-run-name": r.name, "data-state": r.state, "aria-selected": String(selected),
      tabindex: String(rovingStop(r, i) ? 0 : -1),
      onclick: () => {
        const sel = window.getSelection && window.getSelection().toString();
        if (sel && body.contains(window.getSelection().anchorNode)) return;      // a drag to copy text is not a click on the row
        select(r.name);
      },
    },
      h("td", { role: "gridcell", class: "c-state" }, stateBadge(r.state)),
      h("td", { role: "gridcell", class: "c-model", title: tip },
        h("span", { class: "c-model-text" }, full), tag(), first(), r.launching ? h("span", { class: "mode-tag" }, "starting…") : null,
        // the run directory tells apart runs of one model; it is plain selectable text and part of the tooltip
        h("span", { class: "c-dir mono" }, r.name)),
      h("td", { role: "gridcell", class: "c-stage", title: cur === null ? "no stage yet" : `stage ${cur}${r.stage.name ? ` · ${r.stage.name}` : ""}` },
        cur === null ? "" : [h("span", { class: "c-stage-n" }, String(cur)),
          h("span", { class: "c-bar", "aria-hidden": "true" }, h("span", { style: `width:${Math.round(((cur + 1) / total) * 100)}%` }))]),
      h("td", { role: "gridcell", class: "c-age", title: when === null ? "" : new Date(when).toLocaleString() }, r.last_ts ? ago(r.last_ts) : ""));
    body.append(row);
  });
  if (focusName) { const again = body.querySelector(`tr[data-run-name="${CSS.escape(focusName)}"]`); if (again) again.focus(); }
}

async function refreshRuns() {
  try {
    const [runs, machine, launches] = await Promise.all([api("/api/runs"), api("/api/machine"), api("/api/launches")]);
    S.runs = runs.runs;
    S.machine = machine;
    S.launches = launches.launches;
    renderRuns();
    renderMachine();
    renderLaunches();
    if (S.startingRun && S.runs.some((r) => r.name === S.startingRun)) {
      const name = S.startingRun;
      S.startingRun = null;
      select(name);
    }
  } catch (err) {
    if (Date.now() - S.lastOk > STALE_MS) { setConn("offline"); renderMachineChip(null, [], 0); }
  }
}

// ---- the selected run -----------------------------------------------------------------------------

function select(name, { push = true } = {}) {
  if (S.selected === name) return;
  S.selected = name;
  S.detail = null;
  S.ledgerAfter = 0; S.ledgerRows = []; S.file = null;
  if (push) history.replaceState(null, "", `#run/${encodeURIComponent(name)}`);
  renderRuns();
  if (S.view !== "orchard") setView("orchard", { push: false });
  $("run-empty").hidden = true;
  $("run").hidden = false;
  clear($("feed")); clear($("ledger-rows")); clear($("file-list"));
  for (const id of ["file-list", "run-actions"]) delete $(id).dataset.key;
  $("file-name").textContent = "Pick a file"; $("file-body").textContent = "";
  openFeed();
  refreshRun().then(() => { if (S.tab === "ledger") loadLedger(); if (S.tab === "files") renderFiles(); });
}

async function refreshRun() {
  if (!S.selected) return;
  const name = S.selected;
  if (!(name in S.ledgerFirst)) {
    S.ledgerFirst[name] = null;
    api(`/api/runs/${encodeURIComponent(name)}/ledger?after=0`).then((out) => {
      S.ledgerFirst[name] = out.entries.length ? parseTs(out.entries[0].ts) : null;
    }).catch(() => {});
  }
  try {
    const d = await api(`/api/runs/${encodeURIComponent(name)}`);
    if (S.selected !== name) return;
    S.detail = d;
    renderRun();
  } catch (err) {
    if (S.selected === name && /no run named/.test(err.message)) {
      $("run-model").textContent = name;
      $("run-hint-text").textContent = "This run has no ledger yet. A bring-up started from this page shows its first output in the run list.";
    }
  }
}

// The verdict: one sentence per finding with its yardstick, worst first. Operator mode shows it (the CSS hides it otherwise).
// Strings come from the server and are set as text, never as markup.
function renderVerdict(items) {
  const box = $("verdict");
  const list = Array.isArray(items) ? items : [];
  box.hidden = list.length === 0;
  if (!changed($("verdict-list"), list)) return;
  const ul = clear($("verdict-list"));
  for (const i of list) {
    ul.append(h("li", { class: "verdict-item", dataset: { tone: i.tone, id: i.id } },
      h("span", { class: "op-pic", "aria-hidden": "true" }, S.meta.op.tones[i.tone] || ""),
      h("strong", { class: "verdict-headline" }, i.headline), i.detail ? h("span", { class: "verdict-detail" }, i.detail) : null));
  }
}

function renderRun() {
  const d = S.detail;
  if (!d) return;
  const badge = $("run-badge");
  const fresh = Object.assign(stateBadge(d.state), { id: "run-badge" });
  badge.replaceWith(fresh);
  $("run").dataset.state = d.state;                 // operator mode colours its state band from these two
  $("run").dataset.tone = fresh.dataset.tone;
  $("run-model").textContent = d.model || d.name;
  const sub = clear($("run-sub"));
  sub.append(
    h("span", { class: "mono" }, d.name),
    h("span", {}, stageLine(d.stage), d.stage && d.stage.attempt > 1 ? ` · attempt ${d.stage.attempt}` : ""),
    h("span", {}, d.supervisor.alive ? `supervisor pid ${d.supervisor.pid}` : "no supervisor running"),
    h("span", {}, d.last_events.length ? `last activity ${ago(d.last_events[d.last_events.length - 1].ts)}` : ""));
  $("run-hint-text").textContent = d.next_step || d.hint || "";
  const spend = $("run-spend");                       // this run's own helper spend, when it kept a telegraph ledger
  spend.hidden = !d.telegraph;
  spend.textContent = d.telegraph ? `Helper spend for this run: ${spendText(d.telegraph)} · ${plural(d.telegraph.calls, "call")}` : "";
  const mode = $("run-mode");
  mode.hidden = !d.mode;
  // the pictures sit in .emoji spans: the operator style hides them
  if (!d.mode) mode.replaceChildren();
  else if (d.mode.startsWith("lab")) mode.replaceChildren(h("span", { class: "emoji", "aria-hidden": "true" }, "🧪 "), `tests on ${d.mode.slice(5, -1)}`);
  else mode.replaceChildren(h("span", { class: "emoji", "aria-hidden": "true" }, "🏡 "), "this box");
  mode.title = !d.mode ? "" : d.mode.startsWith("lab") ? "Lab mode: hardware tests run on the lab box; the coder stays here and is never parked"
    : "Local mode: everything runs on this machine";
  S.activity = d.activity ? { ...d.activity, seenAt: Date.now() } : null;
  renderNow();
  renderVerdict(d.verdict);
  renderResults(d);
  if (S.tab === "timeline") renderTimeline(d);
  renderActions(d);
  renderBlock(d);
  renderPause(d);
  renderStepper(d);
  renderScene(d);
  renderDetails(d);
  if (S.tab === "files") renderFiles();
  document.title = `${(S.meta.states[d.state] || {}).emoji || ""} ${d.model || d.name} · tt-orchard`;
}

// ---- what the run is doing now ----------------------------------------------------------------------------

function renderNow() {
  const a = S.activity, box = $("run-now");
  const live = a && a.since;
  box.hidden = !a;
  if (!a) return;
  box.dataset.kind = a.kind;
  $("run-now-text").textContent = a.text;
  const elapsed = live ? a.elapsed_s + (Date.now() - a.seenAt) / 1000 : null;
  $("run-now-time").textContent = live ? (a.deadline_s ? `${dur(elapsed)} of ${dur(a.deadline_s)}` : dur(elapsed)) : "";
  const bar = $("run-now-bar");
  bar.hidden = !(live && a.deadline_s);
  if (!bar.hidden) $("run-now-fill").style.width = `${Math.min(100, (100 * elapsed) / a.deadline_s).toFixed(1)}%`;
}

// ---- results, the timeline, other runs of the model --------------------------------------------------

const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const where = (host) => (host ? `on ${host}` : "on this box");

function renderResults(d) {
  const r = d.results || {}, box = clear($("results-body"));
  const any = r.swap || (r.configs && r.configs.length) || r.triage;
  $("results").hidden = !any;
  if (!any) return;
  if (r.triage) {
    box.append(h("p", { class: "result-line" }, h("strong", {}, "Survey: "),
      `${r.triage.class || "?"} (path ${r.triage.path || "?"}), based on ${(r.triage.nearest_model || "?").split("@")[0]}`));
  }
  if (r.swap) {
    const s = r.swap, n = s.n_tokens;
    const hits = s.top1_agreement !== null && n ? Math.round(s.top1_agreement * n) : null;
    box.append(h("div", { class: "result-card", dataset: { ok: String(Boolean(s.coherent && s.top1_agreement >= 0.85)) } },
      h("div", { class: "result-head" }, h("strong", {}, "Graft: one-board swap check "), h("span", { class: "muted" }, where(s.host))),
      h("div", { class: "result-nums" },
        h("span", {}, h("b", {}, hits !== null ? `${hits} of ${n}` : pct(s.top1_agreement)), " tokens match the CPU reference"),
        h("span", {}, "server ready in ", h("b", {}, s.server_ready_s ? dur(s.server_ready_s) : "–")),
        h("span", {}, s.coherent ? "✓ coherent text" : "✕ not coherent")),
      s.free_run_text ? h("blockquote", { class: "free-text" }, `“${s.free_run_text}…”`) : null));
  }
  if (r.configs && r.configs.length) {
    const rows = r.configs.map((c) => h("tr", { dataset: { tone: c.pass === true ? "good" : c.pass === false ? "bad" : "" } },
      h("td", {}, `${c.chips} chip${c.chips > 1 ? "s" : ""}`),
      h("td", {}, c.pass === true ? "✓ pass" : c.pass === false ? "✕ no" : "…"),
      h("td", {}, pct(c.top1_agreement)),
      h("td", {}, c.server_ready_s ? dur(c.server_ready_s) : "–"),
      h("td", {}, c.host ? c.host : c.pass === undefined || c.pass === null ? "" : "this box"),
      h("td", { class: "muted" }, c.reason || c.package || "")));
    box.append(h("div", { class: "result-card" },
      h("div", { class: "result-head" }, h("strong", {}, "Rows: each chip configuration")),
      h("div", { class: "table-wrap" }, h("table", { class: "ledger configs" },
        h("thead", {}, h("tr", {}, ["Chips", "Result", "Match", "Ready in", "Where", "Note"].map((x) => h("th", { scope: "col" }, x)))),
        h("tbody", {}, rows)))));
  }
  if (r.bundle) {
    box.append(h("p", {}, h("button", { type: "button", class: "btn btn-small", onclick: openResults },
      h("span", { class: "emoji", "aria-hidden": "true" }, "📄 "), "Open RESULTS.md")));
  }
  renderCompare(d, box);
}

function renderCompare(d, box) {
  const others = (d.compare || []);
  if (others.length < 2) return;
  const cfgs = [...new Set(others.flatMap((o) => Object.keys(o.configs || {})))].sort((a, b) => a - b);
  box.append(h("details", { class: "compare" },
    h("summary", {}, `${others.length} runs of ${d.model}`),
    h("div", { class: "table-wrap" }, h("table", { class: "ledger" },
      h("thead", {}, h("tr", {}, ["Run", "Mode", "Started", "Took", "Graft ready", ...cfgs.map((c) => `${c}-chip ready`)].map((x) => h("th", { scope: "col" }, x)))),
      h("tbody", {}, others.map((o) => h("tr", { dataset: { tone: o.this ? "good" : "" } },
        h("td", { class: "mono" }, o.this ? `${o.name} (this)` : o.name),
        h("td", {}, o.mode || "–"),
        h("td", {}, o.started ? ago(o.started * 1000) : "–"),
        h("td", {}, o.wall_s ? dur(o.wall_s) : "–"),
        h("td", {}, o.swap_ready_s ? dur(o.swap_ready_s) : "–"),
        ...cfgs.map((c) => { const x = (o.configs || {})[c];
          return h("td", {}, !x ? "–" : x.pass === null || x.pass === undefined ? "…" : `${x.pass ? "" : "✕ "}${x.ready_s ? dur(x.ready_s) : "–"}`); }))))))));
}

const LANE_NAME = { stage: "Stages", agent: "Agent steps", coder: "Coder boot", copy: "Copy to lab", queue: "Wait for chips", test: "Hardware tests", park: "Coder parked" };

function renderTimeline(d) {
  const t = d.timeline, box = clear($("timeline"));
  if (!t || !t.spans || !t.spans.length) { box.append(h("p", { class: "muted" }, "Nothing on the timeline yet.")); return; }
  const span = Math.max(1, t.end - t.start);
  const at = (x) => `${(100 * (x - t.start) / span).toFixed(2)}%`;
  const width = (s) => `${Math.max(0.4, 100 * (s.end - s.start) / span).toFixed(2)}%`;
  const lanes = t.lanes.filter((l) => t.spans.some((s) => s.kind === l));
  box.append(h("div", { class: "tl-axis" }, h("span", {}, clock(new Date(t.start * 1000).toISOString())),
    h("span", {}, `${dur(span)} in all`), h("span", {}, clock(new Date(t.end * 1000).toISOString()))));
  for (const lane of lanes) {
    const bars = t.spans.filter((s) => s.kind === lane).map((s) => h("span", {
      class: "tl-bar", dataset: { kind: s.kind, ok: s.ok === false ? "false" : "true", open: String(Boolean(s.open)) },
      style: `left:${at(s.start)};width:${width(s)}`,
      title: `${s.label || s.kind}${s.host ? ` on ${s.host}` : ""}: ${dur(s.end - s.start)}${s.open ? " (still going)" : ""}`,
    }, lane === "stage" ? `${s.stage}` : ""));
    box.append(h("div", { class: "tl-lane" }, h("span", { class: "tl-name" }, LANE_NAME[lane] || lane),
      h("span", { class: "tl-track" }, bars)));
  }
}

function renderActions(d) {
  const box = $("run-actions");
  if (!changed(box, [d.name, d.state, d.control_pending, d.launch && d.launch.pid])) return;
  clear(box);
  // `pic` names an entry of the Operator style's button pictures; the label always stays as the button's text
  const add = (label, cls, fn, title, pic) => box.append(h("button", { type: "button", class: `btn ${cls}`, onclick: fn, title },
    pic ? h("span", { class: "op-pic", "aria-hidden": "true" }, `${S.meta.op.buttons[pic]} `) : null, label));
  if (d.control_pending) {
    box.append(h("span", { class: "badge", dataset: { tone: "warn" } }, `“${d.control_pending}” sent, waiting for the supervisor`));
  }
  if (d.state === "running") {
    add("Pause", "btn-warn", () => doControl("pause"), "Pause at the next check; the chips stay leased", "pause");
    add("Abort", "btn-danger btn-apart", () => doControl("abort"), "Stop the run and release the chips", "abort");
  } else if (d.state === "paused") {
    add("Resume", "btn-primary", () => doControl("resume"), undefined, "resume");
    add("Abort", "btn-danger btn-apart", () => doControl("abort"), undefined, "abort");
  } else if (RETRY_STATES.includes(d.state)) {
    add(d.state === "aborted" ? "Resume run" : "Retry run", "btn-primary", doRetry,
      "Run the same bring-up again; it resumes from the ledger", "retry");
  } else if (d.state === "ready-for-operator-review") {
    add("Open RESULTS.md", "btn-primary", openResults);
  }
  if (d.launch) box.append(h("span", { class: "badge", dataset: { tone: "good" } }, `started from this page · pid ${d.launch.pid}`));
}

function renderBlock(d) {
  const panel = $("block-panel");
  if (!d.block) { panel.hidden = true; return; }
  panel.hidden = false;
  $("block-retry").hidden = !RETRY_STATES.includes(d.state);          // a block record can outlive the block: only a run that can start again gets the button
  $("block-h").textContent = `Blocked: ${d.block.code}`;
  $("block-reason").textContent = d.block.reason || "";
  const tried = clear($("block-tried"));
  (d.block.tried.length ? d.block.tried : ["Nothing was recorded."]).forEach((t) => tried.append(h("li", {}, t)));
  const un = clear($("block-unblock"));
  d.block.unblock.forEach((t) => un.append(h("li", {}, t)));
}

function renderPause(d) {
  const panel = $("pause-panel");
  if (d.state !== "paused" || !d.pause) { panel.hidden = true; return; }
  panel.hidden = false;
  const p = d.pause;
  const why = p.kind === "operator" ? "Paused by an operator (from this page or `tt-orchard pause`)"
    : `Paused: ${p.reason || p.kind}${p.detector ? ` (${p.detector})` : ""}`;
  $("pause-reason").textContent = `${why}${p.stage !== null && p.stage !== undefined ? `, in stage ${p.stage}` : ""}. ` +
    "The chips stay leased while paused. Resume carries on; abort releases them.";
}

function renderStepper(d) {
  const ol = clear($("stepper"));
  const rows = new Map(d.stages.map((r) => [r.stage, r]));
  const live = d.state === "running";
  const going = d.state === "running" || d.state === "paused";
  // what each stage was at the last draw of this same run: only a change to "pass" counts as fresh, so opening a finished run pops nothing
  const prev = S.stepSeen && S.stepSeen.run === d.name ? S.stepSeen.status : null;
  const seen = {};
  for (const s of S.meta.stages) {
    const r = rows.get(s.number);
    // A stage the ledger left open counts as running only while the run itself is going. In an aborted,
    // blocked or stopped run it is where the run stopped, and its clock is not still running.
    const stopped = r && r.status === "running" && !going;
    const status = r ? (stopped ? "stopped" : r.status) : "pending";
    const mark = h("span", { class: "emoji", "aria-hidden": "true" }, `${stopped ? "⏹" : S.meta.marks[status] || ""} `);
    const opMark = h("span", { class: "op-pic", "aria-hidden": "true" }, `${S.meta.op.marks[status] || ""}`);
    // a stage that passed since the last draw of this run pops once (Operator style, motion allowed only)
    const fresh = Boolean(prev && prev[s.number] !== "pass" && status === "pass");
    seen[s.number] = status;
    const isCur = d.stage && d.stage.current === s.number && !["ready-for-operator-review", "aborted"].includes(d.state);
    ol.append(h("li", {
      class: "step", dataset: { status, current: String(Boolean(isCur)), live: String(live), fresh: String(fresh) },
      "aria-current": isCur ? "step" : null,
    },
      h("span", { class: "num" }, `${s.number}`),
      h("span", { class: "orchard" }, h("span", { "aria-hidden": "true" }, s.emoji + " "), s.name),
      h("span", { class: "real" }, s.real),
      h("span", { class: "stat" }, ...(r ? [mark, opMark,
        `${status}${r.wall_s && !stopped ? " · " + dur(r.wall_s) : ""}${r.attempts > 1 ? ` · ×${r.attempts}` : ""}`] : [opMark, "not yet"]))));
  }
  S.stepSeen = { run: d.name, status: seen };
}

const WEATHER = {
  "running": "☀ sunny", "paused": "🌧 rain", "blocked": "❄ frost", "stopped-or-crashed": "⛈ storm",
  "aborted": "🍂 leaves falling", "ready-for-operator-review": "🧺 harvest day", "not-started": "🌰 seed",
};

function renderScene(d) {
  // Before a stage has started (the coder is still booting) there is no plot to work yet.
  const started = d.stage && d.stage.attempt > 0;
  if (S.scene) {
    S.scene.setRun({ state: d.state, stages: d.stages, current: started ? d.stage.current : null });
    S.scene.setLab((d.mode || "").startsWith("lab") ? d.mode.slice(5, -1) : null);
    S.scene.setActivity(d.activity || null);
  }
  renderSceneCaption(d);
}

function renderSceneCaption(d) {
  const passed = d.stages.filter((r) => r.status === "pass").length;
  const planned = 9 - d.stages.filter((r) => r.status === "skipped").length;
  const a = d.activity || {};
  const started = S.ledgerFirst[d.name];
  const day = started ? Math.max(1, Math.floor((Date.now() - started) / 86400000) + 1) : 1;
  const cap = clear($("scene-caption"));
  const hl = S.health, u = runUtilization();
  const sky = hl ? WEATHER_WORD[hl.weather.kind] || hl.weather.kind : WEATHER_WORD.unknown;
  cap.append(h("span", {}, `Day ${day}`),
    h("span", { title: hl ? hl.weather.why : "" }, sky),
    h("span", {}, a.kind === "test" && a.host ? `🧪 testing on ${a.host}` : a.kind === "copy" ? `🛒 carting files to ${a.host || "the lab"}`
      : u > 0.08 ? `⚒ chips ${Math.round(u * 100)}% busy` : "💤 chips resting"),
    h("span", {}, `${(WEATHER[d.state] || d.state).replace(/^\S+ /, "")}`),
    h("span", {}, `🍎 ${passed} of ${planned} planned plots fruiting`));
}

function renderDetails(d) {
  const dl = clear($("details"));
  const row = (k, v) => dl.append(h("dt", {}, k), h("dd", {}, v));
  row("Run directory", h("span", { class: "mono" }, d.run_dir));
  row("State", `${d.state}${d.supervisor.alive ? ` (supervisor pid ${d.supervisor.pid}, from ${d.supervisor.source})` : ""}`);
  row("Ledger", `${d.ledger.entries} entries, hash chain verified`);
  row("Counts", Object.entries(d.counts).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · "));
  row("Disk free", `run dir ${d.disk.run_dir_free_gb} GB · home ${d.disk.home_free_gb} GB`);
  row("Mode", d.mode ? (d.mode.startsWith("lab") ? `lab: hardware tests on ${d.mode.slice(5, -1)}` : "local: everything on this box") : "–");
  row("Leases on this machine now", d.leases.length ? h("span", {}, d.leases.map((l) => h("span", { class: "mono" }, l, h("br")))) : "none listed");
  if (d.hint) row("Runbook hint", h("span", { class: "muted" }, d.hint));
  row("Control word", d.control_pending || "none waiting");
  const cached = S.research && S.research.name === d.name ? S.research.data : null;
  showResearch($("research"), cached && cached.report, cached && cached.stage0);
  loadResearch(d.name);
}

// ---- actions ------------------------------------------------------------------------------------------

function confirmDialog({ title, text, ok, danger, twoStep }) {
  return new Promise((resolve) => {
    const dlg = $("confirm"), okBtn = $("confirm-ok"), cancel = $("confirm-cancel");
    $("confirm-title").textContent = title;
    $("confirm-text").textContent = text;
    okBtn.textContent = ok;
    okBtn.className = `btn ${danger ? "btn-danger" : "btn-primary"}`;
    let armed = !twoStep, timer = null;
    const done = (v) => { clearTimeout(timer); okBtn.onclick = null; cancel.onclick = null; dlg.onclose = null; if (dlg.open) dlg.close(); resolve(v); };
    okBtn.onclick = () => {
      if (!armed) {
        armed = true;
        okBtn.textContent = `Click again to ${ok.toLowerCase()}`;
        timer = setTimeout(() => { armed = false; okBtn.textContent = ok; }, 4000);
        return;
      }
      done(true);
    };
    cancel.onclick = () => done(false);
    dlg.onclose = () => done(false);
    dlg.showModal();
    cancel.focus();
  });
}

const CONTROL_COPY = {
  pause: { title: "Pause this run?", text: "The supervisor stops at its next check and waits. The chips stay leased and the coder keeps running, so nothing else can use them until you resume or abort.", ok: "Pause" },
  resume: { title: "Resume this run?", text: "The supervisor carries on from where it paused.", ok: "Resume" },
  abort: { title: "Abort this run?", text: "The supervisor stops the coder, releases every chip lease and ends the run. The run stays aborted until a person chooses Resume run, which carries on from the ledger.", ok: "Abort run", danger: true, twoStep: true },
};

async function doControl(word) {
  const name = S.selected;
  if (!(await confirmDialog(CONTROL_COPY[word]))) return;
  try {
    S.detail = await api(`/api/runs/${encodeURIComponent(name)}/control`, { body: { word } });
    renderRun();
    toast(`Sent “${word}”. The supervisor acts on it within about 10 seconds.`);
  } catch (err) { toast(err.message, "bad"); }
}

// The states a run can be started again from. The server keeps the same list (orchard/webui.py RETRYABLE); a test holds them equal.
const RETRY_STATES = ["blocked", "stopped-or-crashed", "aborted"];

async function doRetry() {
  const name = S.selected;
  const aborted = S.detail && S.detail.state === "aborted";
  const ok = await confirmDialog({
    title: aborted ? "Resume this run?" : "Retry this run?",
    text: aborted
      ? "This starts `tt-orchard bringup` for the same model in the background. It runs the preflight, takes the chip lease, boots the coder and resumes from the ledger, where the abort left it. The ledger records a resume after the abort."
      : "This starts the same `tt-orchard bringup` again in the background. It runs the preflight, takes the chip lease, boots the coder and resumes from the ledger. The ledger records it as a retry.",
    ok: aborted ? "Resume run" : "Retry run",
  });
  if (!ok) return;
  try {
    const out = await api(`/api/runs/${encodeURIComponent(name)}/retry`, { body: {} });
    toast(`Retry started (pid ${out.pid}). The feed shows it as it comes up.`);
    refreshRuns(); refreshRun();
  } catch (err) { toast(err.message, "bad"); }
}

// ---- the live feed --------------------------------------------------------------------------------

function openFeed() {
  if (S.source) S.source.close();
  S.feedSeen = new Set();
  S.follow = true;
  $("feed-jump").hidden = true;
  const feed = clear($("feed"));
  feed.append(h("li", { class: "placeholder" }, "Waiting for the run to say something…"));
  const src = new EventSource(`/api/runs/${encodeURIComponent(S.selected)}/events?replay=60`);
  S.source = src;
  $("feed-state").textContent = "connecting…";
  src.onopen = () => { $("feed-state").textContent = "live"; };
  src.onerror = () => { $("feed-state").textContent = "reconnecting…"; };
  src.onmessage = (msg) => {
    let e;
    try { e = JSON.parse(msg.data); } catch { return; }
    addFeedLine(e);
  };
}

function addFeedLine(e) {
  const key = `${e.ts}|${e.actor}|${e.text}`;
  if (S.feedSeen.has(key)) return;            // a reconnect replays recent lines
  S.feedSeen.add(key);
  const feed = $("feed");
  const ph = feed.querySelector(".placeholder");
  if (ph) ph.remove();
  ensureActorOption(e.actor);
  // A reconnect or a newly opened run replays recent lines; only lines from the last half minute animate the scene.
  if (S.scene && Date.now() - (parseTs(e.ts) || 0) < 30000) {
    S.scene.event(e.actor);
    if (/\blease\b|\bpark|\brestore|hardware released/i.test(e.text)) S.scene.event("lease");
  }
  // "still hardware test running ... (12m)" comes every minute: it updates the line before it, not a new line.
  const last = feed.lastElementChild;
  if (/^still /.test(e.text) && last && last.dataset.heartbeat === "true" && last.dataset.actor === e.actor) {
    last.querySelector(".t").textContent = clock(e.ts);
    last.querySelector(".what").textContent = e.text;
    return;
  }
  const li = h("li", { dataset: { actor: e.actor, bad: String(e.bad), heartbeat: String(/^still /.test(e.text)) } },
    h("span", { class: "t" }, clock(e.ts)),
    h("span", { class: "who", dataset: { role: e.role } }, h("span", { "aria-hidden": "true", class: "icon" }, (e.icon || "") + " "), e.actor),
    h("span", { class: "what" }, e.text));
  if (S.feedFilter && e.actor !== S.feedFilter) li.hidden = true;
  feed.append(li);
  while (feed.children.length > FEED_MAX) feed.firstElementChild.remove();
  if (S.follow) feed.scrollTop = feed.scrollHeight;
  else $("feed-jump").hidden = false;
}

function ensureActorOption(actor) {
  const sel = $("feed-filter");
  if ([...sel.options].some((o) => o.value === actor)) return;
  sel.append(h("option", { value: actor }, actor));
}

// ---- ledger ---------------------------------------------------------------------------------------

async function loadLedger() {
  if (!S.selected) return;
  const name = S.selected;
  try {
    const out = await api(`/api/runs/${encodeURIComponent(name)}/ledger?after=${S.ledgerAfter}`);
    if (S.selected !== name) return;
    for (const r of out.entries) {
      S.ledgerRows.push(r);
      S.ledgerAfter = r.seq;
    }
    renderLedger();
  } catch (err) { toast(err.message, "bad"); }
}

function ledgerTone(r) {
  if (r.event === "stage_end" && r.kind === "pass") return "good";
  if ((r.event === "stage_end" && r.kind === "fail") || r.kind === "blocked" || r.kind === "abort") return "bad";
  return "";
}

function renderLedger() {
  const sel = $("ledger-filter");
  for (const ev of new Set(S.ledgerRows.map((r) => r.event))) {
    if (![...sel.options].some((o) => o.value === ev)) sel.append(h("option", { value: ev }, ev));
  }
  const body = clear($("ledger-rows"));
  const rows = S.ledgerRows.filter((r) => !S.ledgerFilter || r.event === S.ledgerFilter);
  for (const r of rows.slice().reverse()) {
    body.append(h("tr", { dataset: { tone: ledgerTone(r) } },
      h("td", {}, r.seq), h("td", { title: r.ts }, clock(r.ts)), h("td", {}, r.stage ?? "–"),
      h("td", {}, r.event), h("td", {}, r.summary)));
  }
  $("ledger-count").textContent = `${rows.length} of ${S.ledgerRows.length} entries, newest first`;
}

// ---- files -----------------------------------------------------------------------------------------

function fileGroup(path) {
  if (path.startsWith("stages/8/bundle/")) return "Operator bundle";
  if (path.includes("/evidence/") || path.includes("/tests/")) return "Evidence";
  if (path.startsWith("stages/")) return "Stage results";
  return "Run";
}

function renderFiles() {
  const d = S.detail, nav = $("file-list");
  if (!d || !changed(nav, [d.name, d.files])) return;
  clear(nav);
  if (!d.files.length) { nav.append(h("p", { class: "muted" }, "No files yet.")); return; }
  const groups = new Map();
  for (const f of d.files) {
    const g = fileGroup(f);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(f);
  }
  for (const g of ["Operator bundle", "Run", "Stage results", "Evidence"]) {
    if (!groups.has(g)) continue;
    nav.append(h("div", { class: "file-group" }, h("h3", {}, g),
      h("ul", {}, groups.get(g).map((f) => h("li", {}, h("button", {
        type: "button", class: "file-link", "aria-current": String(f === S.file), onclick: () => openFile(f),
      }, f))))));
  }
}

// Where a run's bundle report is opened: the header button of a finished run and the button under Results both call it.
function openResults() { setTab("files"); openFirstBundleFile(); }

function openFirstBundleFile() {
  const d = S.detail;
  const f = d && (d.files.find((x) => x.endsWith("bundle/RESULTS.md")) || d.files.find((x) => x.startsWith("stages/8/bundle/")));
  if (f) openFile(f);
}

async function openFile(path) {
  S.file = path;
  renderFiles();
  for (const b of $("file-list").querySelectorAll(".file-link")) b.setAttribute("aria-current", String(b.textContent === path));
  $("file-name").textContent = path + (path.startsWith("stages/8/bundle/") ? "  ·  read-only: the harness never publishes" : "");
  $("file-body").textContent = "Loading…";
  try {
    let text = await api(`/api/runs/${encodeURIComponent(S.selected)}/file?path=${encodeURIComponent(path)}`);
    if (path.endsWith(".json")) {
      try { text = JSON.stringify(JSON.parse(text), null, 2); } catch { /* show it as it is */ }
    }
    $("file-body").textContent = text || "(empty)";
  } catch (err) {
    $("file-body").textContent = err.message;
  }
}

// ---- tabs ------------------------------------------------------------------------------------------

const TABS = ["feed", "ledger", "files", "timeline", "details"];

function setTab(tab, focus) {
  S.tab = tab;
  for (const t of TABS) {
    const btn = $(`tab-${t}`), pane = $(`pane-${t}`);
    const on = t === tab;
    btn.setAttribute("aria-selected", String(on));
    btn.tabIndex = on ? 0 : -1;
    pane.hidden = !on;
    if (on && focus) btn.focus();
  }
  if (tab === "ledger") loadLedger();
  if (tab === "files") renderFiles();
  if (tab === "timeline" && S.detail) renderTimeline(S.detail);
  if (tab === "feed" && S.follow) $("feed").scrollTop = $("feed").scrollHeight;
}

// ---- new bring-up -------------------------------------------------------------------------------------

const MODEL_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$/;
let lastPreflight = null;

function openNewRun() {
  lastPreflight = null;
  const cfg = (S.meta && S.meta.config) || {};
  $("nr-mode").value = cfg.mode === "lab" && cfg.lab ? "lab" : "local";
  $("nr-mode").querySelector('option[value="lab"]').disabled = !cfg.lab;
  $("nr-mode").querySelector('option[value="lab"]').textContent = cfg.lab ? `The lab box (${cfg.lab})` : "The lab box (none in the config)";
  $("nr-mode-help").textContent = cfg.lab ? `The config says ${cfg.mode === "lab" ? "the lab box" : "this box"}; this run can choose either.`
    : "No [lab] table in the config: the run stays on this box.";
  $("nr-result").hidden = true;
  $("nr-error").hidden = true;
  setStartReady(false);
  $("nr-check").disabled = false;
  $("nr-check").textContent = "Run checks";
  updateOptionsSummary();
  $("new-run").showModal();
  $("nr-model").focus();
}

// Run checks is the main button until the checks pass; then Start is. `ready` means the last check passed and nothing has changed since.
function setStartReady(ready) {
  $("nr-start").disabled = !ready;
  $("nr-start").classList.toggle("btn-primary", ready);
  $("nr-check").classList.toggle("btn-primary", !ready);
}

// The options are folded away because they rarely change, so the closed line says what they are now.
function updateOptionsSummary() {
  const where = $("nr-mode").value === "lab" ? "the lab box" : "this box";
  const research = { full: "full research", "no-github": "research without GitHub", offline: "no research" }[$("nr-research").value] || "full research";
  const base = $("nr-base").value.trim();
  $("nr-options-summary").textContent = `Options: ${where}, ${research}${base ? `, base ${base}` : ""}`;
}

function formError(text) {
  $("nr-error").textContent = text;
  $("nr-error").hidden = !text;
}

async function runChecks() {
  const model = $("nr-model").value.trim(), base = $("nr-base").value.trim();
  formError("");
  if (!MODEL_RE.test(model)) { formError("Enter the model as org/name, for example Altworld/Hemmingway-1."); $("nr-model").focus(); return; }
  if (base && !MODEL_RE.test(base)) { formError("The base must be org/name, or left empty."); $("nr-options").open = true; $("nr-base").focus(); return; }
  $("nr-check").disabled = true;
  $("nr-check").textContent = "Checking…";
  setStartReady(false);
  try {
    const out = await api("/api/preflight", { body: { model, base: base || null, mode: $("nr-mode").value, research: $("nr-research").value } });
    out.mode = $("nr-mode").value;
    out.researchChoice = $("nr-research").value;
    lastPreflight = out;
    renderPreflight(out);
  } catch (err) {
    formError(err.message);
  } finally {
    $("nr-check").disabled = false;
    $("nr-check").textContent = "Run checks again";
  }
}

const CHECK_MARK = { ok: "✓", warn: "!", block: "✕" };

// The base-model research (orchard/research). `view` is the server's bounded, redacted summary; every string in
// it is set as text, and a link is shown only when the server kept it (https).
function researchBlock(view, stage0) {
  const box = h("div", { class: "research-body" });
  if (!view && !stage0) return null;
  if (stage0) {
    box.append(h("p", { class: "research-line" },
      h("strong", {}, "Stage 0 measured: "), `${stage0.measured_class || "unknown"} against the estimate ${stage0.estimated_class || "none"}`,
      stage0.candidate && stage0.candidate.repo
        ? ` (compared with ${stage0.candidate.repo}${stage0.candidate.confidence ? `, ${stage0.candidate.confidence}` : ""})` : "",
      stage0.agreed === true ? " · agreed" : stage0.agreed === false ? " · did not agree; the measurement decides" : ""));
  }
  if (!view) return box;
  if (view.offline) box.append(h("p", { class: "muted" }, "Research was off for this run (no requests were made)."));
  if (view.new_model_gated) box.append(h("p", { class: "muted" }, "The new model is gated or private: research sends no token, so it could not compare its config or estimate its class. Few or no candidates does not mean there is no base."));
  if (view.selected) {
    box.append(h("p", { class: "research-line" }, h("strong", {}, "Chosen: "),
      h("span", { class: "mono" }, view.selected.bundle || view.selected.repo || "–"),
      ` · ${view.selected.confidence || ""}${view.selected.class ? ` · ${view.selected.class}` : ""}`));
  }
  if (view.choices.length) {
    box.append(h("h4", {}, "Bundles that could serve it, best first"),
      h("ul", { class: "research-list" }, view.choices.map((c) => h("li", {},
        h("span", { class: "mono" }, c.name || "?"),
        h("span", { class: "muted" }, ` ${c.installed ? "installed" : "published"} · ${c.class || "no estimate"} · ${c.confidence || ""}${c.schema ? ` · v${c.schema}` : ""}`)))));
  }
  if (view.candidates.length) {
    box.append(h("h4", {}, "Candidate base models"),
      h("ul", { class: "research-list" }, view.candidates.map((c) => h("li", {},
        h("span", { class: "mono" }, c.repo || "?"),
        h("span", { class: "muted" }, ` ${c.confidence || ""}${c.download_gb ? ` · ${c.download_gb} GB` : ""}${c.class ? ` · ${c.class}` : ""}${c.why ? ` · ${c.why}` : ""}`)))));
  }
  if (view.ports.length) {
    box.append(h("h4", {}, "Existing work"),
      h("ul", { class: "research-list" }, view.ports.map((p) => h("li", {}, p.claim || "",
        p.url ? h("span", {}, " ", h("a", { href: p.url, target: "_blank", rel: "noopener noreferrer" }, p.url)) : ""))));
  }
  if (view.unavailable.length) {
    box.append(h("h4", {}, "Could not be read"),
      h("ul", { class: "research-list muted" }, view.unavailable.map((u) => h("li", {}, `${u.source}: ${u.reason}`))));
  }
  return box;
}

function showResearch(el, view, stage0) {
  const body = researchBlock(view, stage0);
  clear(el);
  el.hidden = !body;
  if (body) el.append(h("h3", {}, "Base-model research"), body);
}

// A run's report and stage 0's comparison. Asked for when a run is opened and again every 30 s, because the
// report appears before stage 0 and the comparison after it.
async function loadResearch(name) {
  const r = S.research;
  if (r && r.name === name && Date.now() - r.at < 30000) return;
  S.research = { name, at: Date.now(), data: r && r.name === name ? r.data : null };
  try {
    const data = await api(`/api/runs/${encodeURIComponent(name)}/research`);
    S.research = { name, at: Date.now(), data };
    if (S.detail && S.detail.name === name) showResearch($("research"), data.report, data.stage0);
  } catch (err) { /* the details still show; the research card is optional */ }
}

function renderPreflight(out) {
  $("nr-result").hidden = false;
  $("nr-run").textContent = `${out.resuming ? "Resumes" : "Creates"} the run ${out.run} in ${out.run_dir}.`;
  const ul = clear($("nr-checks"));
  for (const c of out.checks) {
    ul.append(h("li", { class: "check", dataset: { status: c.status } },
      h("span", { class: "mark", "aria-label": c.status }, CHECK_MARK[c.status] || "?"),
      h("span", { class: "name" }, c.name),
      h("span", { class: "detail" }, c.detail + (c.reason ? ` [${c.reason}]` : ""))));
  }
  showResearch($("nr-research-view"), out.research, null);
  const base = out.checks.find((c) => c.name === "base" && c.status === "block" && c.candidates && c.candidates.length);
  const fs = $("nr-candidates"), list = clear($("nr-candidate-list"));
  fs.hidden = !base;
  if (base) {
    base.candidates.forEach((c, i) => list.append(h("label", {},
      h("input", { type: "radio", name: "nr-cand", value: c.name, id: `nr-cand-${i}`,
        onchange: () => { $("nr-base").value = c.name; updateOptionsSummary(); runChecks(); } }),   // an assignment fires no input event
      h("span", { class: "mono" }, c.name), h("span", { class: "muted" }, c.installed ? "installed" : "installs with tt-model pull"))));
  }
  setStartReady(out.ok);
  if (!out.ok) formError("A check blocks the run. Fix it, or choose a base below, and run the checks again.");
}

async function startBringup() {
  if (!lastPreflight || !lastPreflight.ok) return;
  const model = $("nr-model").value.trim(), base = $("nr-base").value.trim();
  if (model !== lastPreflight.model || (base || null) !== lastPreflight.base || $("nr-mode").value !== lastPreflight.mode
      || $("nr-research").value !== lastPreflight.researchChoice) {
    formError("The model, base, mode or research setting changed since the checks ran. Run the checks again."); return;
  }
  $("nr-start").disabled = true;
  try {
    const out = await api("/api/bringup", { body: { model, base: base || null, mode: $("nr-mode").value, research: $("nr-research").value } });
    $("new-run").close();
    toast(`Started ${model} (pid ${out.pid}). It shows in the run list once its ledger exists; the download can take a while.`);
    S.startingRun = out.run;
    refreshRuns();
  } catch (err) {
    formError(err.message);
    $("nr-start").disabled = false;
  }
}

// ---- start ------------------------------------------------------------------------------------------

function wire() {
  $("new-run-btn").addEventListener("click", openNewRun);
  $("block-retry").addEventListener("click", doRetry);
  $("view-orchard-btn").addEventListener("click", () => setView("orchard"));
  $("view-hardware-btn").addEventListener("click", () => setView("hardware"));
  $("view-settings-btn").addEventListener("click", () => setView("settings"));
  $("view-telegraph-btn").addEventListener("click", () => setView("telegraph"));
  $("spend-meter").addEventListener("click", () => setView("telegraph"));
  $("settings-form").addEventListener("submit", saveSettings);
  $("checks-run").addEventListener("click", startChecks);
  $("caches-run").addEventListener("click", loadCaches);
  setInterval(renderNow, 1000);
  $("machine-chip").addEventListener("click", () => setView("hardware"));
  $("machine-details-link").addEventListener("click", (ev) => { ev.preventDefault(); setView("hardware"); });
  $("nr-mode").addEventListener("change", () => { setStartReady(false); updateOptionsSummary(); });
  $("nr-research").addEventListener("change", () => { setStartReady(false); updateOptionsSummary(); });
  $("hw-zoom-in").addEventListener("click", () => zoomToplike(+1));
  $("hw-zoom-out").addEventListener("click", () => zoomToplike(-1));
  $("hw-wide").addEventListener("click", () => {
    const on = !document.body.classList.contains("wide");
    document.body.classList.toggle("wide", on);
    $("hw-wide").setAttribute("aria-pressed", String(on));
  });
  $("hw-full").addEventListener("click", () => {
    const tv = document.querySelector(".tv");
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (tv.requestFullscreen) tv.requestFullscreen().catch(() => toast("Full screen is not available here.", "bad"));
  });
  document.addEventListener("fullscreenchange", () => setTimeout(refitToplike, 100));
  $("nr-cancel").addEventListener("click", () => $("new-run").close());
  $("new-run-form").addEventListener("submit", (ev) => { ev.preventDefault(); runChecks(); });
  $("nr-start").addEventListener("click", startBringup);
  for (const id of ["nr-model", "nr-base"]) $(id).addEventListener("input", () => { setStartReady(false); updateOptionsSummary(); });

  for (const t of TABS) $(`tab-${t}`).addEventListener("click", () => setTab(t));
  $("runs-body").addEventListener("keydown", (ev) => {
    // Up and Down move the selection through the rows (and the focus with it); Enter opens the selected run's page.
    const rows = [...$("runs-body").querySelectorAll("tr[data-run-row]")];
    const at = rows.indexOf(document.activeElement.closest ? document.activeElement.closest("tr") : null);
    if (at < 0) return;
    let to = null;
    if (ev.key === "ArrowDown") to = Math.min(rows.length - 1, at + 1);
    else if (ev.key === "ArrowUp") to = Math.max(0, at - 1);
    else if (ev.key === "Enter") { select(rows[at].dataset.runName); const run = $("run"); if (!run.hidden) { run.scrollIntoView({ block: "start" }); } return; }
    else return;
    ev.preventDefault();
    const name = rows[to].dataset.runName;
    select(name);
    const next = $("runs-body").querySelector(`tr[data-run-name="${CSS.escape(name)}"]`);
    if (next) next.focus();
  });
  $("tabs").addEventListener("keydown", (ev) => {
    const i = TABS.indexOf(S.tab);
    if (ev.key === "ArrowRight") { ev.preventDefault(); setTab(TABS[(i + 1) % TABS.length], true); }
    if (ev.key === "ArrowLeft") { ev.preventDefault(); setTab(TABS[(i + TABS.length - 1) % TABS.length], true); }
    if (ev.key === "Home") { ev.preventDefault(); setTab(TABS[0], true); }
    if (ev.key === "End") { ev.preventDefault(); setTab(TABS[TABS.length - 1], true); }
  });

  const feed = $("feed");
  feed.addEventListener("scroll", () => {
    const atEnd = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 24;
    S.follow = atEnd;
    if (atEnd) $("feed-jump").hidden = true;
  });
  $("feed-jump").addEventListener("click", () => { S.follow = true; feed.scrollTop = feed.scrollHeight; $("feed-jump").hidden = true; });
  $("feed-filter").addEventListener("change", (ev) => {
    S.feedFilter = ev.target.value;
    for (const li of feed.children) if (li.dataset.actor) li.hidden = Boolean(S.feedFilter) && li.dataset.actor !== S.feedFilter;
  });
  $("ledger-filter").addEventListener("change", (ev) => { S.ledgerFilter = ev.target.value; renderLedger(); });
}

function tickClock() {
  const now = new Date();
  const wd = now.toLocaleDateString([], { weekday: "short" });
  $("clock-day").textContent = `${wd}. ${now.getDate()}`;
  $("clock-time").textContent = now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }).toLowerCase();
}

// ---- views: the orchard and the hardware TV -------------------------------------------------------------

function setView(view, { push = true } = {}) {
  S.view = view;
  $("view-orchard-btn").setAttribute("aria-pressed", String(view === "orchard"));
  $("view-hardware-btn").setAttribute("aria-pressed", String(view === "hardware"));
  $("view-settings-btn").setAttribute("aria-pressed", String(view === "settings"));
  $("view-telegraph-btn").setAttribute("aria-pressed", String(view === "telegraph"));
  $("hardware-view").hidden = view !== "hardware";
  $("settings-view").hidden = view !== "settings";
  $("telegraph-view").hidden = view !== "telegraph";
  const runShown = view === "orchard" && S.selected;
  $("run").hidden = !runShown;
  $("run-empty").hidden = view !== "orchard" || Boolean(S.selected);
  if (view === "hardware") {
    if (push) history.replaceState(null, "", "#machine");
    startToplike(S.hwMode || "normal");
    pollChecks();                       // the last setup check, if there was one, is shown where the checks now live
    return;
  }
  stopToplike();
  document.body.classList.remove("wide");
  $("hw-wide").setAttribute("aria-pressed", "false");
  if (view === "telegraph") {
    if (push) history.replaceState(null, "", "#telegraph");
    renderTelegraph();
    pollTelegraph();
  } else if (view === "settings") {
    if (push) history.replaceState(null, "", "#models");
    loadSettings();
  } else if (push) {
    history.replaceState(null, "", S.selected ? `#run/${encodeURIComponent(S.selected)}` : "#");
  }
}

// ---- settings: which chip layout and which CPU stand-in -------------------------------------------------
// The server offers the QuietBox 2 layouts setup ships (orchard/settings.py) and saves a choice only after the
// supervisor's own loaders accept it. A change applies to the next run or retry.

// ---- Check machines: setup's checks, read-only, for this box and the lab ------------------------------

const CHECK_TONE = { ok: "ok", warn: "warn", todo: "warn", fail: "block" };
let checksTimer = null;

async function startChecks() {
  $("checks-run").disabled = true;
  try {
    renderChecks(await api("/api/checks", { body: {} }));
  } catch (err) {
    $("checks-run").disabled = false;
    $("checks-status").textContent = err.message;
    return;
  }
  pollChecks();
}

async function pollChecks() {
  clearTimeout(checksTimer);
  try {
    const st = await api("/api/checks");
    renderChecks(st);
    if (st.running) checksTimer = setTimeout(pollChecks, 1500);
  } catch (err) {
    $("checks-status").textContent = err.message;
  }
}

function copyButton(text) {
  return h("button", { type: "button", class: "btn btn-small", onclick: (ev) => {
    const done = () => { ev.target.textContent = "Copied"; setTimeout(() => { ev.target.textContent = "Copy"; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
  } }, "Copy");
}

function renderChecks(st) {
  $("checks-run").disabled = Boolean(st.running);
  const when = (t) => new Date(t * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  $("checks-status").textContent = st.running ? "Checking… (an import test and the lab's ssh calls take a minute)"
    : st.error ? `The check failed: ${st.error}`
    : st.finished ? `Checked at ${when(st.finished)}. Nothing was changed.`
    : "Runs the setup checks on this box, and on the lab when there is one. Nothing is changed: each problem shows the command that fixes it.";
  const box = clear($("checks-boxes"));
  for (const b of st.boxes || []) {
    const bad = b.steps.filter((x) => x.status !== "ok").length;
    box.append(h("div", { class: "checkbox-panel" },
      h("h3", {}, `${b.name} `, h("span", { class: "muted small" }, bad ? `${bad} to look at` : "ready")),
      h("ul", { class: "checks" }, b.steps.map((x) => h("li", { class: "check", dataset: { status: CHECK_TONE[x.status] || "warn" } },
        h("span", { class: "mark", "aria-label": x.status }, CHECK_MARK[CHECK_TONE[x.status]] || "!"),
        h("span", { class: "name" }, x.name),
        h("span", { class: "detail" }, x.detail,
          x.fix.map((f) => h("span", { class: "fix" }, h("code", {}, f), copyButton(f)))))))));
  }
}

async function loadCaches() {
  const lab = $("caches-lab").checked;
  $("caches-run").disabled = true;
  $("caches-status").textContent = lab ? "Reading the caches here and on the lab…" : "Reading the caches…";
  try {
    const out = await api(`/api/caches${lab ? "?lab=1" : ""}`);
    const rows = clear($("caches-rows"));
    const now = Date.now() / 1000;
    for (const c of out.caches.slice().sort((a, b) => a.box.localeCompare(b.box) || b.gb - a.gb)) {
      rows.append(h("tr", { dataset: { tone: c.used_by.length ? "good" : "" } },
        h("td", {}, c.box), h("td", { class: "num" }, c.gb.toFixed(1)),
        h("td", { class: "num" }, c.last_write ? Math.round((now - c.last_write) / 86400) : "–"),
        h("td", {}, c.used_by.join(", ") || "–"),
        h("td", {}, h("span", {}, c.model || "?"), h("br"), h("span", { class: "mono muted small" }, c.path))));
    }
    $("caches-table").hidden = !out.caches.length;
    const total = out.caches.reduce((a, c) => a + c.gb, 0);
    $("caches-status").textContent = (out.error ? `${out.error}. ` : "") +
      `${out.caches.length} cache(s), ${total.toFixed(1)} GB, under ${out.root}. Green rows are used by a run that is not finished.`;
  } catch (err) {
    $("caches-status").textContent = err.message;
  } finally {
    $("caches-run").disabled = false;
  }
}

async function loadSettings() {
  $("settings-error").hidden = true;
  try {
    S.settings = await api("/api/settings");
  } catch (err) {
    clear($("settings-layouts")).append(h("p", { class: "form-error" }, err.message));
    return;
  }
  renderSettings();
}

function renderSettings() {
  const st = S.settings;
  const box = clear($("settings-layouts"));
  if (!st.layout) {
    box.append(h("p", { class: "muted small" },
      `The coder in the config now (${st.coder.target}) is not one of these layouts. Saving replaces it.`));
  }
  for (const p of st.layouts) {
    const id = `layout-${p.name}`;
    const input = h("input", { type: "radio", name: "layout", id, value: p.name, disabled: !p.installed });
    input.checked = p.name === st.layout;
    input.addEventListener("change", previewRoles);
    box.append(h("label", { class: "layout-card", for: id, dataset: { installed: String(p.installed) } },
      input,
      h("span", { class: "layout-text" },
        h("span", { class: "layout-name" }, p.chip_model),
        h("span", { class: "layout-meta" }, `${p.coder.chips} chips · port ${p.coder.port} · ${p.coder.target}`),
        h("span", { class: "muted small" }, p.note),
        p.installed ? null : h("span", { class: "layout-missing small" }, `Not installed here: tt-model pull ${p.coder.target}`))));
  }
  const sel = clear($("settings-cpu"));
  const cur = (st.roles.find((r) => r.tier === "cpu") || {}).model;
  const names = cur && !st.cpu_models.includes(cur) ? [cur, ...st.cpu_models] : st.cpu_models;
  for (const n of names) {
    const o = h("option", { value: n }, n + (st.cpu_models.includes(n) ? "" : " (not in ollama now)"));
    o.selected = n === cur;
    sel.append(o);
  }
  sel.onchange = previewRoles;
  $("settings-cpu-help").textContent = st.cpu_models.length
    ? "Any model ollama has on this machine. It serves steps while the chips are busy."
    : "ollama did not answer, so no CPU model can be picked. Start it (ollama serve) and reload.";
  $("mode-local").checked = st.mode !== "lab";
  $("mode-lab").checked = st.mode === "lab";
  $("mode-lab").disabled = !st.lab;
  $("mode-lab-host").textContent = st.lab ? `(${st.lab})` : "";
  $("mode-help").textContent = st.lab ? "A bring-up can still choose the other mode for itself."
    : "There is no [lab] table in the config, so every run stays on this box. README 5.9 says how to add a lab.";
  $("settings-files").textContent = `Saved to ${st.config} and ${st.tiers_path}, with a dated copy of each.`;
  const running = $("settings-running");
  running.hidden = !st.running.length;
  running.textContent = st.running.length
    ? `Running now: ${st.running.join(", ")}. It keeps the models it started with; a change applies to the next run or retry.`
    : "";
  previewRoles();
}

function previewRoles() {
  const st = S.settings;
  const picked = document.querySelector('input[name="layout"]:checked');
  const layout = picked && st.layouts.find((p) => p.name === picked.value);
  const cpu = $("settings-cpu").value;
  const body = clear($("settings-roles"));
  for (const r of st.roles) {
    let model = r.model;
    let endpoint = r.endpoint;
    if (layout && layout.tiers[r.tier]) ({ model, endpoint } = layout.tiers[r.tier]);
    if (r.tier === "cpu" && cpu) model = cpu;
    const changed = model !== r.model || endpoint !== r.endpoint;
    body.append(h("tr", { dataset: { changed: String(changed) } },
      h("td", {}, r.role), h("td", {}, r.does), h("td", {}, r.tier),
      h("td", {}, model, changed ? h("span", { class: "muted small" }, " after saving") : null),
      h("td", {}, endpoint)));
  }
}

async function saveSettings(ev) {
  ev.preventDefault();
  const picked = document.querySelector('input[name="layout"]:checked');
  const err = $("settings-error");
  err.hidden = true;
  if (!picked) { err.textContent = "Pick a chip layout first."; err.hidden = false; return; }
  const btn = $("settings-save");
  btn.disabled = true;
  try {
    const mode = $("mode-lab").checked ? "lab" : "local";
    const out = await api("/api/settings", { body: { layout: picked.value, cpu_model: $("settings-cpu").value, mode } });
    toast(out.note);
    S.meta = await api("/api/meta");
    await loadSettings();
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  } finally {
    btn.disabled = false;
  }
}

// ---- tt-toplike, view-only ------------------------------------------------------------------------
// The page draws tt-toplike's own terminal UI with xterm.js. It never sends keystrokes: stdin is disabled in the
// terminal and the server has no input route. A view button restarts tt-toplike with that --mode.

const MODE_LABEL = { normal: "Table", starfield: "Starfield", castle: "Memory Castle", flow: "Memory Flow",
  arcade: "Arcade", training: "Training", rotate: "Rotate all" };

function buildChannels() {
  const box = clear($("hw-channels"));
  const tp = S.meta.toplike;
  if (!tp.available) { $("hw-missing").hidden = false; return; }
  for (const mode of tp.modes) {
    box.append(h("button", { type: "button", class: "btn btn-small", "aria-pressed": "false", dataset: { mode },
      onclick: () => startToplike(mode) }, MODE_LABEL[mode] || mode));
  }
}

// The skin. tt-toplike paints its own near-black backgrounds (ESC[48;2;r;g;b with r+g+b small). Only those are
// retinted to the TV's night indigo, so the picture sits in the page; every colour tt-toplike chooses for what it
// draws stays exactly as it sends it. A chunk can end inside an escape, so an unfinished CSI is carried over.
const SKIN_BG = [29, 24, 48];          // --tv, #1d1830
const SKIN_THEME = {
  background: "#1d1830", foreground: "#f3e6c8", cursor: "#1d1830", cursorAccent: "#1d1830",
  black: "#1d1830", red: "#e0533d", green: "#7cc35a", yellow: "#f2c84b", blue: "#6aa8e0", magenta: "#c47ad6",
  cyan: "#5cc8c0", white: "#f3e6c8", brightBlack: "#5a4a6e", brightRed: "#ff8a6e", brightGreen: "#a8e07c",
  brightYellow: "#ffe08a", brightBlue: "#9ccfff", brightMagenta: "#e8a8f5", brightCyan: "#9ef0e6", brightWhite: "#fffbe6",
};

function skinSgr(params) {
  const p = params.split(";");
  for (let i = 0; i < p.length; i++) {
    if ((p[i] === "38" || p[i] === "48") && p[i + 1] === "5") { i += 2; continue; }
    if ((p[i] === "38" || p[i] === "48") && p[i + 1] === "2" && i + 4 < p.length) {
      if (p[i] === "48" && (+p[i + 2]) + (+p[i + 3]) + (+p[i + 4]) < 80) {
        p[i + 2] = String(SKIN_BG[0]); p[i + 3] = String(SKIN_BG[1]); p[i + 4] = String(SKIN_BG[2]);
      }
      i += 4;
    }
  }
  return p.join(";");
}

function makeSkinner() {
  const dec = new TextDecoder("utf-8");
  let carry = "";
  return (bytes) => {
    let text = carry + dec.decode(bytes, { stream: true });
    carry = "";
    const cut = text.lastIndexOf("\x1b");
    if (cut !== -1 && /^\x1b(\[[0-9;?]*)?$/.test(text.slice(cut))) { carry = text.slice(cut); text = text.slice(0, cut); }
    return text.replace(/\x1b\[([0-9;]*)m/g, (m, ps) => "\x1b[" + skinSgr(ps) + "m");
  };
}

function loadFont() { try { return Math.min(18, Math.max(8, +localStorage.getItem("orchard.hwFont") || 11)); } catch { return 11; } }
function saveFont(n) { try { localStorage.setItem("orchard.hwFont", String(n)); } catch { /* fine */ } }

function refitToplike() {
  const hw = S.hw;
  if (!hw) return;
  hw.fit.fit();
  if (hw.id) api(`/api/toplike/${encodeURIComponent(hw.id)}/resize`, { body: { cols: hw.term.cols, rows: hw.term.rows } }).catch(() => {});
}

function zoomToplike(delta) {
  const hw = S.hw;
  const size = Math.min(18, Math.max(8, (hw ? hw.term.options.fontSize : loadFont()) + delta));
  saveFont(size);
  if (hw) { hw.term.options.fontSize = size; refitToplike(); }
}

function stopToplike() {
  const hw = S.hw;
  if (!hw) return;
  hw.source.close();
  hw.ro.disconnect();
  hw.term.dispose();
  S.hw = null;
  $("hw-status").textContent = "Off";
}

function startToplike(mode) {
  stopToplike();
  if (!S.meta.toplike.available || typeof Terminal === "undefined") return;
  S.hwMode = mode;
  for (const b of $("hw-channels").querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  const el = clear($("term"));
  const term = new Terminal({
    disableStdin: true, cursorBlink: false, cursorStyle: "bar", cursorWidth: 1, convertEol: false, scrollback: 0,
    fontSize: loadFont(), lineHeight: 1, letterSpacing: 0, customGlyphs: true, drawBoldTextInBrightColors: false,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace",
    theme: SKIN_THEME,
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(el);
  // The WebGL renderer draws block and box characters as exact pixel glyphs, where tt-toplike's portraits live;
  // without WebGL the default renderer is used.
  try {
    if (typeof WebglAddon !== "undefined") {
      const gl = new WebglAddon.WebglAddon();
      gl.onContextLoss(() => gl.dispose());
      term.loadAddon(gl);
    }
  } catch { /* the DOM renderer still works */ }
  fit.fit();
  const skin = makeSkinner();
  const source = new EventSource(`/api/toplike?mode=${encodeURIComponent(mode)}&cols=${term.cols}&rows=${term.rows}`);
  const hw = { term, fit, source, id: null, ro: null, mode };
  S.hw = hw;
  $("hw-status").textContent = `Tuning in: ${MODE_LABEL[mode] || mode}…`;
  source.addEventListener("session", (ev) => {
    hw.id = JSON.parse(ev.data).id;
    $("hw-status").textContent = `tt-toplike · ${MODE_LABEL[mode] || mode} · on ${S.meta.hostname}`;
  });
  source.onmessage = (ev) => {
    const bin = atob(ev.data), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    term.write(skin(bytes));
  };
  source.addEventListener("exit", () => {
    source.close();
    $("hw-status").textContent = "tt-toplike stopped. Pick a view to start it again.";
  });
  source.onerror = () => {
    // The server refuses (busy or not installed) with an error status, and a dropped stream ends the session.
    // Either way this view is over; do not let EventSource start another tt-toplike by reconnecting.
    source.close();
    if (S.hw === hw && !hw.id) $("hw-status").textContent = "Could not start tt-toplike (too many open, or not installed).";
    else if (S.hw === hw) $("hw-status").textContent = "The connection dropped. Pick a view to reconnect.";
  };
  let timer = null;
  hw.ro = new ResizeObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(() => { if (S.hw === hw) refitToplike(); }, 200);
  });
  hw.ro.observe(el);
}

function routeFromHash() {
  // #machine and #models are the names now; #hardware and #settings are what earlier links and bookmarks carry
  if (location.hash === "#machine" || location.hash === "#hardware") { setView("hardware", { push: false }); return; }
  if (location.hash === "#models" || location.hash === "#settings") { setView("settings", { push: false }); return; }
  if (location.hash === "#telegraph") { setView("telegraph", { push: false }); return; }
  const m = location.hash.match(/^#run\/(.+)$/);
  if (m) select(decodeURIComponent(m[1]), { push: false });
}

async function main() {
  wire();
  startTelegraphPolling();               // first: neither the metadata request nor the run list may stop the spend meter being polled
  try {
    S.meta = await api("/api/meta");
  } catch (err) {
    setConn("offline");
    clear($("machine")).append(h("p", { class: "form-error" }, err.message));
    return;
  }
  $("brand-host").textContent = `${S.meta.hostname} · ${S.meta.version}${S.meta.lan ? " · open on the network" : ""}`;
  $("brand-host").title = $("brand-host").textContent;        // the bar truncates it on narrower screens
  S.scene = window.OrchardScene ? new window.OrchardScene($("scene")) : null;
  if (S.scene) S.scene.start();
  tickClock();
  setInterval(tickClock, 10000);
  pollHealth();
  setInterval(() => { if (!document.hidden) pollHealth(); }, 2000);
  buildChannels();
  await refreshRuns();
  routeFromHash();
  if (!S.selected && S.runs.length && S.view === "orchard") select(S.runs[0].name);
  // A hidden tab keeps its title (which shows the run's state) current, but polls only every 30 s.
  let lastHidden = 0;
  const due = () => {
    if (!document.hidden) return true;
    if (Date.now() - lastHidden < HIDDEN_POLL_MS) return false;
    lastHidden = Date.now();
    return true;
  };
  setInterval(() => { if (due()) refreshRuns(); }, POLL_RUNS_MS);
  setInterval(() => {
    if (!due()) return;
    refreshRun();
    if (S.tab === "ledger") loadLedger();
  }, POLL_RUN_MS);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { refreshRuns(); refreshRun(); }
  });
  setInterval(() => { if (!document.hidden && S.lastOk && Date.now() - S.lastOk > STALE_MS) setConn("stale"); }, 2000);
  window.addEventListener("hashchange", routeFromHash);
}

document.addEventListener("DOMContentLoaded", main);
