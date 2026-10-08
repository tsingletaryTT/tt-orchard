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
  scene: null, view: "orchard", hw: null,
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
    state === "unreadable" ? "⚠ unreadable" : state);
  return h("span", { class: "badge", dataset: { tone: TONE[st.role] || "dim" }, title: `${st.word} (${state})` },
    h("span", { "aria-hidden": "true" }, st.emoji), st.word, h("span", { class: "real" }, `· ${state}`));
}

function stageText(stage) {
  if (!stage || stage.current === null || stage.current === undefined) return "";
  const s = S.meta && S.meta.stages[stage.current];
  const orchard = s ? `${s.emoji} ${s.name}` : "";
  return `stage ${stage.current} ${orchard} · ${stage.name || (s && s.real) || ""}`.replace(/\s+/g, " ").trim();
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
  let resp;
  try {
    resp = await fetch(path, init);
  } catch (err) {
    setConn("offline");
    throw new Error("The UI server cannot be reached. Is `tt-orchard ui` still running?");
  }
  S.lastOk = Date.now();
  setConn("live");
  const isJson = (resp.headers.get("Content-Type") || "").includes("application/json");
  const data = isJson ? await resp.json() : await resp.text();
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

function renderMachine() {
  const m = S.machine;
  if (!m || !changed($("machine"), [m.boards, m.coder, m.cpu_tier, m.disk_free_gb, m.gozer_ok])) return;
  const box = clear($("machine"));
  if (!m.gozer_ok) {
    box.append(h("p", { class: "muted" }, "gozer did not answer. Chip leases are unknown; check `gozer status`."));
  }
  for (const b of m.boards) {
    box.append(h("div", { class: "board" },
      h("div", { class: "board-head" }, h("span", { class: "mono", title: b.id }, `board …${b.id.slice(-6)}`), h("span", {}, b.kind)),
      h("div", { class: "chips" }, b.chips.map((c) => h("div", {
        class: "chip", dataset: { tone: CHIP_TONE[c.state] || "alarm" },
        title: [c.bdf, c.owner, c.note].filter(Boolean).join(" · "),
      }, h("strong", {}, `chip ${c.index} · ${c.state}`), h("span", { class: "owner" }, c.owner || c.bdf))))));
  }
  // gozer marks a chip HELD-FOREIGN when a process outside the lease owner's tree has it open. A tt-model
  // container serving under the supervisor's lease is such a process (CLAUDE.md, 2026-10-02), so say so.
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
  const free = all.filter((c) => c.state === "FREE").length;
  $("machine-note").textContent = all.length ? `${free} of ${all.length} chips free` : "";
}

function renderLaunches() {
  if (!changed($("launches"), S.launches.map((l) => [l.run, l.alive, l.has_ledger, l.log_tail]))) return;
  const box = clear($("launches"));
  for (const l of S.launches.filter((x) => x.alive && !x.has_ledger)) {
    box.append(h("div", { class: "launch" },
      h("strong", {}, `Starting ${l.model}`), " ", h("span", { class: "muted" }, ago(l.started * 1000)),
      h("pre", {}, l.log_tail || "waiting for output…")));
  }
  for (const l of S.launches.filter((x) => !x.alive && !x.has_ledger)) {
    box.append(h("div", { class: "launch" },
      h("strong", {}, `${l.model} did not start`), h("pre", {}, l.log_tail || "(no output)")));
  }
}

function renderRuns() {
  const list = $("runs");
  if (!changed(list, [S.selected, S.runs.map((r) => [r.name, r.state, r.model, r.stage && r.stage.current, r.last_ts,
    r.launching, Math.floor((Date.now() - (parseTs(r.last_ts) || 0)) / 60000)])])) return;
  clear(list);
  $("runs-note").textContent = S.runs.length ? `${S.runs.length}` : "";
  if (!S.runs.length) {
    list.append(h("li", { class: "empty-small" },
      h("span", { class: "muted" }, "No runs here yet."),
      h("button", { type: "button", class: "btn btn-small", onclick: openNewRun }, "Start a bring-up")));
    return;
  }
  for (const r of S.runs) {
    list.append(h("li", {}, h("button", {
      type: "button", class: "run-item", "aria-current": String(r.name === S.selected),
      onclick: () => select(r.name),
    }, stateBadge(r.state),
      h("span", { class: "model" }, r.model || r.name),
      h("span", { class: "meta mono" }, r.name),
      h("span", { class: "meta" },
        h("span", {}, r.stage && r.stage.current !== null ? `stage ${r.stage.current}` : ""),
        h("span", {}, r.last_ts ? ago(r.last_ts) : ""),
        r.launching ? h("span", {}, "starting…") : null))));
  }
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
    if (Date.now() - S.lastOk > STALE_MS) setConn("offline");
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

function renderRun() {
  const d = S.detail;
  if (!d) return;
  const badge = $("run-badge");
  badge.replaceWith(Object.assign(stateBadge(d.state), { id: "run-badge" }));
  $("run-model").textContent = d.model || d.name;
  const sub = clear($("run-sub"));
  sub.append(
    h("span", { class: "mono" }, d.name),
    h("span", {}, stageText(d.stage) + (d.stage && d.stage.attempt > 1 ? ` · attempt ${d.stage.attempt}` : "")),
    h("span", {}, d.supervisor.alive ? `supervisor pid ${d.supervisor.pid}` : "no supervisor running"),
    h("span", {}, d.last_events.length ? `last activity ${ago(d.last_events[d.last_events.length - 1].ts)}` : ""));
  $("run-hint-text").textContent = d.hint || "";
  renderActions(d);
  renderBlock(d);
  renderPause(d);
  renderStepper(d);
  renderScene(d);
  renderDetails(d);
  if (S.tab === "files") renderFiles();
  document.title = `${(S.meta.states[d.state] || {}).emoji || ""} ${d.model || d.name} · tt-orchard`;
}

function renderActions(d) {
  const box = $("run-actions");
  if (!changed(box, [d.name, d.state, d.control_pending, d.launch && d.launch.pid])) return;
  clear(box);
  const add = (label, cls, fn, title) => box.append(h("button", { type: "button", class: `btn ${cls}`, onclick: fn, title }, label));
  if (d.control_pending) {
    box.append(h("span", { class: "badge", dataset: { tone: "warn" } }, `“${d.control_pending}” sent, waiting for the supervisor`));
  }
  if (d.state === "running") {
    add("Pause", "btn-warn", () => doControl("pause"), "Pause at the next check; the chips stay leased");
    add("Abort", "btn-danger", () => doControl("abort"), "Stop the run and release the chips");
  } else if (d.state === "paused") {
    add("Resume", "btn-primary", () => doControl("resume"));
    add("Abort", "btn-danger", () => doControl("abort"));
  } else if (d.state === "blocked" || d.state === "stopped-or-crashed") {
    add("Retry run", "btn-primary", doRetry, "Run the same bring-up again; it resumes from the ledger");
  } else if (d.state === "ready-for-operator-review") {
    add("Open the bundle", "btn-primary", () => { setTab("files"); openFirstBundleFile(); });
  }
  if (d.launch) box.append(h("span", { class: "badge", dataset: { tone: "good" } }, `started from this page · pid ${d.launch.pid}`));
}

function renderBlock(d) {
  const panel = $("block-panel");
  if (!d.block) { panel.hidden = true; return; }
  panel.hidden = false;
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
  for (const s of S.meta.stages) {
    const r = rows.get(s.number);
    // A stage the ledger left open counts as running only while the run itself is going. In an aborted,
    // blocked or stopped run it is where the run stopped, and its clock is not still running.
    const stopped = r && r.status === "running" && !going;
    const status = r ? (stopped ? "stopped" : r.status) : "pending";
    const mark = stopped ? "⏹" : S.meta.marks[status] || "";
    const isCur = d.stage && d.stage.current === s.number && !["ready-for-operator-review", "aborted"].includes(d.state);
    ol.append(h("li", {
      class: "step", dataset: { status, current: String(Boolean(isCur)), live: String(live) },
      "aria-current": isCur ? "step" : null,
    },
      h("span", { class: "num" }, `${s.number}`),
      h("span", { class: "orchard" }, h("span", { "aria-hidden": "true" }, s.emoji + " "), s.name),
      h("span", { class: "real" }, s.real),
      h("span", { class: "stat" }, r ? `${mark} ${status}${r.wall_s && !stopped ? " · " + dur(r.wall_s) : ""}${r.attempts > 1 ? ` · ×${r.attempts}` : ""}` : "not yet")));
  }
}

const WEATHER = {
  "running": "☀ sunny", "paused": "🌧 rain", "blocked": "❄ frost", "stopped-or-crashed": "⛈ storm",
  "aborted": "🍂 leaves falling", "ready-for-operator-review": "🧺 harvest day", "not-started": "🌰 seed",
};

function renderScene(d) {
  if (S.scene) S.scene.setRun({ state: d.state, stages: d.stages, current: d.stage ? d.stage.current : null });
  const passed = d.stages.filter((r) => r.status === "pass").length;
  const started = S.ledgerFirst[d.name];
  const day = started ? Math.max(1, Math.floor((Date.now() - started) / 86400000) + 1) : 1;
  const cap = clear($("scene-caption"));
  cap.append(h("span", {}, `Day ${day} of this run`), h("span", {}, WEATHER[d.state] || d.state),
    h("span", {}, `🍎 ${passed} of 9 plots fruiting`));
}

function renderDetails(d) {
  const dl = clear($("details"));
  const row = (k, v) => dl.append(h("dt", {}, k), h("dd", {}, v));
  row("Run directory", h("span", { class: "mono" }, d.run_dir));
  row("State", `${d.state}${d.supervisor.alive ? ` (supervisor pid ${d.supervisor.pid}, from ${d.supervisor.source})` : ""}`);
  row("Ledger", `${d.ledger.entries} entries, hash chain verified`);
  row("Counts", Object.entries(d.counts).map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`).join(" · "));
  row("Disk free", `run dir ${d.disk.run_dir_free_gb} GB · home ${d.disk.home_free_gb} GB`);
  row("Leases", d.leases.length ? h("span", {}, d.leases.map((l) => h("span", { class: "mono" }, l, h("br")))) : "none listed");
  row("Control word", d.control_pending || "none waiting");
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
  abort: { title: "Abort this run?", text: "The supervisor stops the coder, releases every chip lease and ends the run. An aborted run cannot be resumed; a new bring-up of the same model starts over in a new run directory only if you move this one aside.", ok: "Abort run", danger: true, twoStep: true },
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

async function doRetry() {
  const name = S.selected;
  const ok = await confirmDialog({
    title: "Retry this run?",
    text: "This starts the same `tt-orchard bringup` again in the background. It runs the preflight, takes the chip lease, boots the coder and resumes from the ledger. The ledger records it as a retry.",
    ok: "Retry run",
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
  if (S.scene) S.scene.event(e.actor);
  const li = h("li", { dataset: { actor: e.actor, bad: String(e.bad) } },
    h("span", { class: "t" }, clock(e.ts)),
    h("span", { class: "who", dataset: { role: e.role } }, h("span", { "aria-hidden": "true" }, (e.icon || "") + " "), e.actor),
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

const TABS = ["feed", "ledger", "files", "details"];

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
  if (tab === "feed" && S.follow) $("feed").scrollTop = $("feed").scrollHeight;
}

// ---- new bring-up -------------------------------------------------------------------------------------

const MODEL_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9._-]+$/;
let lastPreflight = null;

function openNewRun() {
  lastPreflight = null;
  $("nr-result").hidden = true;
  $("nr-error").hidden = true;
  $("nr-start").disabled = true;
  $("nr-check").disabled = false;
  $("nr-check").textContent = "Run checks";
  $("new-run").showModal();
  $("nr-model").focus();
}

function formError(text) {
  $("nr-error").textContent = text;
  $("nr-error").hidden = !text;
}

async function runChecks() {
  const model = $("nr-model").value.trim(), base = $("nr-base").value.trim();
  formError("");
  if (!MODEL_RE.test(model)) { formError("Enter the model as org/name, for example Altworld/Hemmingway-1."); $("nr-model").focus(); return; }
  if (base && !MODEL_RE.test(base)) { formError("The base must be org/name, or left empty."); $("nr-base").focus(); return; }
  $("nr-check").disabled = true;
  $("nr-check").textContent = "Checking…";
  $("nr-start").disabled = true;
  try {
    const out = await api("/api/preflight", { body: { model, base: base || null } });
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
  const base = out.checks.find((c) => c.name === "base" && c.status === "block" && c.candidates && c.candidates.length);
  const fs = $("nr-candidates"), list = clear($("nr-candidate-list"));
  fs.hidden = !base;
  if (base) {
    base.candidates.forEach((c, i) => list.append(h("label", {},
      h("input", { type: "radio", name: "nr-cand", value: c.name, id: `nr-cand-${i}`,
        onchange: () => { $("nr-base").value = c.name; runChecks(); } }),
      h("span", { class: "mono" }, c.name), h("span", { class: "muted" }, c.installed ? "installed" : "installs with tt-model pull"))));
  }
  $("nr-start").disabled = !out.ok;
  if (!out.ok) formError("A check blocks the run. Fix it, or choose a base below, and run the checks again.");
}

async function startBringup() {
  if (!lastPreflight || !lastPreflight.ok) return;
  const model = $("nr-model").value.trim(), base = $("nr-base").value.trim();
  if (model !== lastPreflight.model || (base || null) !== lastPreflight.base) {
    formError("The model or base changed since the checks ran. Run the checks again."); return;
  }
  $("nr-start").disabled = true;
  try {
    const out = await api("/api/bringup", { body: { model, base: base || null } });
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
  $("view-orchard-btn").addEventListener("click", () => setView("orchard"));
  $("view-hardware-btn").addEventListener("click", () => setView("hardware"));
  $("nr-cancel").addEventListener("click", () => $("new-run").close());
  $("new-run-form").addEventListener("submit", (ev) => { ev.preventDefault(); runChecks(); });
  $("nr-start").addEventListener("click", startBringup);
  for (const id of ["nr-model", "nr-base"]) $(id).addEventListener("input", () => { $("nr-start").disabled = true; });

  for (const t of TABS) $(`tab-${t}`).addEventListener("click", () => setTab(t));
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
  $("hardware-view").hidden = view !== "hardware";
  const runShown = view === "orchard" && S.selected;
  $("run").hidden = !runShown;
  $("run-empty").hidden = view !== "orchard" || Boolean(S.selected);
  if (view === "hardware") {
    if (push) history.replaceState(null, "", "#hardware");
    startToplike(S.hwMode || "normal");
  } else {
    stopToplike();
    if (push) history.replaceState(null, "", S.selected ? `#run/${encodeURIComponent(S.selected)}` : "#");
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
    disableStdin: true, cursorBlink: false, convertEol: false, scrollback: 0, fontSize: 13,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    theme: { background: "#1b1a2e", foreground: "#f3e6c8", cursor: "#1b1a2e" },
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(el);
  fit.fit();
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
    term.write(bytes);
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
    timer = setTimeout(() => {
      if (S.hw !== hw) return;
      fit.fit();
      if (hw.id) api(`/api/toplike/${encodeURIComponent(hw.id)}/resize`, { body: { cols: term.cols, rows: term.rows } }).catch(() => {});
    }, 200);
  });
  hw.ro.observe(el);
}

function routeFromHash() {
  if (location.hash === "#hardware") { setView("hardware", { push: false }); return; }
  const m = location.hash.match(/^#run\/(.+)$/);
  if (m) select(decodeURIComponent(m[1]), { push: false });
}

async function main() {
  wire();
  try {
    S.meta = await api("/api/meta");
  } catch (err) {
    setConn("offline");
    clear($("machine")).append(h("p", { class: "form-error" }, err.message));
    return;
  }
  $("brand-host").textContent = `${S.meta.hostname} · ${S.meta.version}${S.meta.lan ? " · open on the network" : ""}`;
  S.scene = window.OrchardScene ? new window.OrchardScene($("scene")) : null;
  if (S.scene) S.scene.start();
  tickClock();
  setInterval(tickClock, 10000);
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
