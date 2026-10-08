// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// The orchard: a pixel-art farm that shows the machine and the selected run.
//
// The weather is the chips' health (from /api/health): clear sky when every chip is well, a heatwave when one
// runs hot, overcast when a lease is stale or a chip is used outside a lease, and a storm when a chip's ARC
// heartbeat stops or it nears its temperature limit. The sky follows the local time of day.
//
// The farmer is the orchardist (the supervisor). He works the plot of the running stage as hard as the chips
// are working, and when the chips are idle he sits on the bench by the shed. When chips change leases he walks
// to the shed, goes in for tools and comes back. The grafter (the coder) works beside him while the agent talks;
// the sheepdog (the watchdog) runs across when it steps in.
//
// Nine plots, one per stage, grow from tilled soil to fruit trees as stages pass. The run's state is on the ground:
// frost on the plots when it is blocked, autumn trees when it was aborted, a full basket when it is ripe.
//
// Everything is drawn in code on a 320 x 128 canvas scaled up with crisp pixels; there are no image files. The
// scene is decorative: every fact in it is also text on the page. Under reduced motion it is drawn still.
"use strict";

(function () {
  const W = 320, H = 128, GROUND = 78, PLOT_W = 24, PLOT_GAP = 5, PLOTS_X0 = 50, FPS = 12;
  const SHED_X = 4, SHED_DOOR = SHED_X + 14, BENCH_X = 34;

  const C = {
    sky: ["#7cc7f2", "#bfe8ff"], dusk: ["#f2a65a", "#ffd9a0"], night: ["#141a3a", "#2b3270"],
    heat: ["#f2a65a", "#ffe3a8"], heatNight: ["#3a1f3a", "#6b3a4a"], storm: ["#3f4658", "#6f7a90"], grey: ["#9aa8b8", "#d4dce4"],
    hillFar: "#7fb85a", hillNear: "#5e9c3d", grass: "#6aa83e", grassDark: "#57902f", grassLight: "#86c24f", dry: "#a9a84a",
    soil: "#7a4e2d", soilDark: "#5c3a20", soilWet: "#5a3a22", fence: "#a8743f", fenceDark: "#6e4522",
    trunk: "#6b4423", leaf: "#3e8e3e", leafLight: "#5fb04a", leafDark: "#2c6b2c", apple: "#d9443a", appleHi: "#ff8a7a",
    wilt: "#8a6b3a", frost: "#e8f2fa", sign: "#c98b4b", signDark: "#7a4e25", ink: "#3b2414",
    sun: "#ffd84a", sunHot: "#ff9a3a", moon: "#f3f0d8", cloud: "#ffffff", cloudDark: "#8a93a5", drop: "#cfe6ff",
    autumn: ["#e07b2a", "#c9542a", "#f2b544", "#a8432a"], gold: "#f2c84b",
    shed: "#a0522d", shedDark: "#6b3218", roof: "#7a3b2a", roofDark: "#4e2418", door: "#4a2e16", bench: "#8a5a2b",
  };

  const SPRITES = {
    farmer: { pal: { H: "#e8c35a", h: "#b5651d", S: "#f2c18d", E: "#3b2414", B: "#4a7fc1", P: "#3e5c8a", b: "#5a3a1e" },
      frames: [
        ["...HHHH...", "..HHHHHH..", ".hhhhhhhh.", "...SSSS...", "...SESE...", "...SSSS...", "..BBBBBB..",
         ".SBBBBBBS.", ".SBBBBBBS.", "..PPPPPP..", "..PPPPPP..", "..PP..PP..", "..PP..PP..", "..bb..bb.."],
        ["...HHHH...", "..HHHHHH..", ".hhhhhhhh.", "...SSSS...", "...SESE...", "...SSSS...", "..BBBBBB..",
         ".SBBBBBBS.", ".SBBBBBBS.", "..PPPPPP..", "..PPPPPP..", "...PP.PP..", "..PP...PP.", "..bb...bb."]] },
    sitting: { pal: { H: "#e8c35a", h: "#b5651d", S: "#f2c18d", E: "#3b2414", B: "#4a7fc1", P: "#3e5c8a", b: "#5a3a1e" },
      frames: [["...HHHH...", "..HHHHHH..", ".hhhhhhhh.", "...SSSS...", "...S-S-...", "...SSSS...", "..BBBBBB..",
                ".SBBBBBBS.", ".SBBBBBBS.", "..PPPPPPPP", "..PPPPPPPP", "........bb"]] },
    grafter: { pal: { R: "#7a3b2a", S: "#f2c18d", E: "#3b2414", G: "#3f8f4f", A: "#e9d8a6", P: "#5a4a3a", b: "#4a2e18", X: "#c9ced6" },
      frames: [
        ["..RRRR..", ".RRRRRR.", "..SSSS..", "..SESE..", "..SSSS..", ".GGGGGG.", "SGAAAAGX", "SGAAAAGX",
         ".PPPPPP.", ".PP..PP.", ".PP..PP.", ".bb..bb."],
        ["..RRRR..", ".RRRRRR.", "..SSSS..", "..SESE..", "..SSSS..", ".GGGGGG.", "SGAAAAG.", "SGAAAAGX",
         ".PPPPPP.", ".PP..PP.", ".PP..PP.", ".bb..bb."]] },
    dog: { pal: { D: "#9a6233", W: "#f3ebdd", K: "#2b1b10" },
      frames: [
        ["........DD..", "D......DDKD.", ".DDDDDDDDDDK", ".DWWWWWDD...", ".DDDDDDDD...", ".D.D..D.D...", ".K.K..K.K..."],
        ["........DD..", ".D.....DDKD.", ".DDDDDDDDDDK", ".DWWWWWDD...", ".DDDDDDDD...", "..D.DD.D....", "..K.KK.K...."]] },
    basket: { pal: { B: "#b5793a", b: "#7a4e25", A: "#d9443a", a: "#ff8a7a", G: "#3e8e3e" },
      frames: [["...G..G...", "..AaAAaA..", ".AAAAAAAA.", "bBbBbBbBbB", "BbBbBbBbBb", "bBbBbBbBbB", ".bBbBbBbB."]] },
  };
  SPRITES.sitting.pal["-"] = "#3b2414";

  const DIGITS = {
    0: ["111", "101", "101", "101", "111"], 1: ["010", "110", "010", "010", "111"], 2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "011", "001", "111"], 4: ["101", "101", "111", "001", "001"], 5: ["111", "100", "111", "001", "111"],
    6: ["111", "100", "111", "101", "111"], 7: ["111", "001", "010", "010", "010"], 8: ["111", "101", "111", "101", "111"],
  };

  function rng(seed) {
    let s = seed >>> 0;
    return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  }

  class OrchardScene {
    constructor(canvas) {
      this.cv = canvas;
      this.cx = canvas.getContext("2d");
      canvas.width = W; canvas.height = H;
      this.state = "not-started";
      this.stages = Array(9).fill("pending");
      this.current = null;
      this.weather = "unknown";
      this.util = 0;                 // 0..1, how hard the chips are working
      this.t = 0;
      this.x = BENCH_X;              // the farmer
      this.errand = null;            // "to-shed" | "inside" | "back"
      this.errandUntil = 0;
      this.tool = false;
      this.dog = null;
      this.bursts = [];
      this.busyUntil = 0;
      this.flash = 0;
      this.dirt = [];
      this.reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
      this._last = 0;
      this._seed = rng(7);
      this.drops = Array.from({ length: 80 }, () => [this._seed() * W, this._seed() * H, 0.6 + this._seed()]);
      this.stars = Array.from({ length: 40 }, () => [Math.floor(this._seed() * W), Math.floor(this._seed() * (GROUND - 30))]);
      this.reduce.addEventListener?.("change", () => this.draw());
    }

    // ---- what the page tells the scene ----
    setRun({ state, stages, current }) {
      const prev = this.stages.slice();
      this.state = state || "not-started";
      const next = Array(9).fill("pending");
      for (const r of stages || []) if (r.stage >= 0 && r.stage < 9) next[r.stage] = r.status;
      const going = this.state === "running" || this.state === "paused";
      next.forEach((st, i) => {
        if (st === "running" && !going) next[i] = "stopped";
        if (this._known && prev[i] !== "pass" && st === "pass") this.bursts.push({ plot: i, until: this.t + 3 });
      });
      this._known = true;
      this.stages = next;
      this.current = going ? current : null;
      if (this.isStill()) this.draw();
    }

    setMachine({ weather, util }) {
      this.weather = weather || "unknown";
      this.util = Math.max(0, Math.min(1, util || 0));
      if (this.isStill()) this.draw();
    }

    event(kind) {
      if (kind === "sheepdog") this.dog = { x: -14, until: this.t + 6 };
      else if (kind === "grafter") this.busyUntil = this.t + 20;
      else if (kind === "lease" && !this.errand) this.errand = "to-shed";
      if (this.isStill()) this.draw();
    }

    isStill() { return this.reduce.matches || document.hidden; }

    start() {
      const tick = (now) => {
        requestAnimationFrame(tick);
        if (this.isStill() || now - this._last < 1000 / FPS) return;
        const dt = this._last ? Math.min(0.5, (now - this._last) / 1000) : 0;
        this._last = now;
        this.t += dt;
        this.step(dt);
        this.draw();
      };
      requestAnimationFrame(tick);
      this.draw();
    }

    // ---- who does what ----
    working() { return this.state === "running" && this.current !== null && this.current !== undefined && this.util > 0.08; }

    target() {
      if (this.errand === "to-shed" || this.errand === "inside") return SHED_DOOR;
      if (this.working() || (this.state === "running" && this.current !== null && this.current !== undefined))
        return this.working() ? this.plotX(this.current) - 9 : BENCH_X;
      if (this.state === "ready-for-operator-review") return this.plotX(8) + PLOT_W + 2;
      return BENCH_X;
    }

    step(dt) {
      const tgt = this.target(), d = tgt - this.x;
      const speed = 22 + 30 * this.util;
      this.x += Math.sign(d) * Math.min(Math.abs(d), speed * dt);
      if (this.errand === "to-shed" && Math.abs(this.x - SHED_DOOR) < 1) { this.errand = "inside"; this.errandUntil = this.t + 1.6; }
      else if (this.errand === "inside" && this.t > this.errandUntil) { this.errand = "back"; this.tool = true; }
      else if (this.errand === "back" && Math.abs(this.x - this.target()) < 1) { this.errand = null; }
      if (this.dog) {
        this.dog.x += 70 * dt;
        if (this.dog.x > W + 14 || this.t > this.dog.until) this.dog = null;
      }
      this.bursts = this.bursts.filter((b) => b.until > this.t);
      // dirt flicks up from the hoe, more of it the harder the chips work
      if (this.working() && Math.abs(this.x - this.target()) < 1 && Math.random() < dt * (2 + 14 * this.util)) {
        const px = this.plotX(this.current) + 4;
        this.dirt.push({ x: px, y: GROUND + 14, vx: (Math.random() - 0.3) * 20, vy: -18 - 20 * this.util, life: 0.8 });
      }
      for (const p of this.dirt) { p.x += p.vx * dt; p.y += p.vy * dt; p.vy += 60 * dt; p.life -= dt; }
      this.dirt = this.dirt.filter((p) => p.life > 0);
      const fall = this.weather === "storm" ? 140 : 0;
      for (const p of this.drops) {
        p[1] += (fall || 20) * p[2] * dt;
        p[0] += (this.weather === "storm" ? -18 : this.state === "aborted" ? 8 : 0) * dt;
        if (p[1] > GROUND + 40) { p[1] = -4; p[0] = this._seed() * W; }
        if (p[0] < -4) p[0] = W;
        if (p[0] > W + 4) p[0] = 0;
      }
      if (this.weather === "storm" && !this.reduce.matches && this._seed() < dt / 9) this.flash = 0.6;   // soft, rare
      this.flash = Math.max(0, this.flash - dt * 1.5);
    }

    plotX(i) { return PLOTS_X0 + i * (PLOT_W + PLOT_GAP); }

    // ---- drawing ----
    px(x, y, w, h, color) { this.cx.fillStyle = color; this.cx.fillRect(Math.round(x), Math.round(y), w, h); }

    sprite(name, x, y, frame = 0, flip = false) {
      const sp = SPRITES[name], rows = sp.frames[frame % sp.frames.length];
      for (let r = 0; r < rows.length; r++) {
        const row = rows[r];
        for (let c = 0; c < row.length; c++) {
          const k = row[flip ? row.length - 1 - c : c];
          if (k !== ".") this.px(x + c, y + r, 1, 1, sp.pal[k]);
        }
      }
    }

    digit(n, x, y, color) {
      DIGITS[n].forEach((row, r) => [...row].forEach((b, c) => { if (b === "1") this.px(x + c, y + r, 1, 1, color); }));
    }

    night() { const h = new Date().getHours(); return h >= 20 || h < 6; }

    skyColors() {
      if (this.weather === "storm") return C.storm;
      if (this.weather === "overcast") return C.grey;
      if (this.weather === "heatwave") return this.night() ? C.heatNight : C.heat;
      const h = new Date().getHours();
      if (this.night()) return C.night;
      if (h >= 18 || h < 7) return C.dusk;
      return C.sky;
    }

    draw() {
      const cx = this.cx, still = this.reduce.matches;
      const [top, bottom] = this.skyColors();
      const g = cx.createLinearGradient(0, 0, 0, GROUND);
      g.addColorStop(0, top); g.addColorStop(1, bottom);
      cx.fillStyle = g; cx.fillRect(0, 0, W, GROUND);
      const dark = top === C.night[0] || top === C.heatNight[0];
      if (dark) for (const [x, y] of this.stars) this.px(x, y, 1, 1, (x + Math.floor(this.t * 2)) % 7 ? "#fffbe0" : "#9aa0d0");
      if (this.weather === "clear" || this.weather === "heatwave" || this.weather === "unknown") {
        if (top === C.night[0]) { this.circle(290, 20, 8, C.moon); this.circle(294, 17, 7, top); }
        else this.circle(290, 20, this.weather === "heatwave" ? 12 : 9, this.weather === "heatwave" ? C.sunHot : C.sun);
      }
      this.clouds();
      this.hills(GROUND - 4, 22, C.hillFar, 0.021, 3);
      this.hills(GROUND, 14, C.hillNear, 0.034, 11);
      const frost = this.state === "blocked";
      const dry = this.weather === "heatwave";
      this.px(0, GROUND, W, H - GROUND, frost ? "#cfe0d6" : dry ? C.dry : this.state === "aborted" ? "#9a9a48" : C.grass);
      const r = rng(3);
      for (let i = 0; i < 220; i++) {
        const x = Math.floor(r() * W), y = GROUND + 2 + Math.floor(r() * (H - GROUND - 2));
        this.px(x, y, 1, 2, frost ? C.frost : r() < 0.5 ? C.grassDark : C.grassLight);
      }
      if (this.weather === "heatwave" && !still) this.shimmer();
      this.shed();
      for (let i = 0; i < 9; i++) this.plot(i);
      this.fence();
      this.people();
      for (const p of this.dirt) this.px(p.x, p.y, 1, 1, C.soilDark);
      if (this.dog) this.sprite("dog", this.dog.x, GROUND + 28, Math.floor(this.t * 10));
      if (this.state === "ready-for-operator-review") this.sprite("basket", this.plotX(8) + PLOT_W - 2, GROUND + 24);
      this.precipitation();
      if (this.flash > 0) { cx.fillStyle = `rgba(255,255,240,${this.flash * 0.35})`; cx.fillRect(0, 0, W, H); }
    }

    people() {
      const still = this.reduce.matches, at = Math.abs(this.x - this.target()) < 1;
      const y = GROUND + 6;
      if (this.errand === "inside") {
        // the door stands open while he is in the shed
        this.px(SHED_DOOR - 1, GROUND + 8, 7, 12, "#2a1a0c");
      } else if (this.working() && at && !this.errand) {
        // hoeing: the swing gets quicker as the chips work harder; sweat when they run hot
        const swing = still ? 0 : Math.floor(this.t * (3 + 9 * this.util)) % 2;
        this.sprite("farmer", this.x, y, 0);
        const hx = this.x + 10, hy = y + 6 + (swing ? 4 : 0);
        this.px(hx, hy, 1, 8 - (swing ? 4 : 0), "#8a5a2b"); this.px(hx - 1, hy + (swing ? 4 : 8), 4, 2, "#9aa3b5");
        if (this.util > 0.7 && !still && Math.floor(this.t * 3) % 2) this.px(this.x + 1, y + 2, 1, 2, C.drop);
      } else if (!this.errand && at && this.x <= BENCH_X + 1) {
        this.sprite("sitting", BENCH_X, GROUND + 9);                    // resting on the bench
        if (this.util <= 0.08 && !still) {
          const k = Math.floor(this.t) % 3;
          this.zzz(BENCH_X + 10 + k * 3, GROUND + 4 - k * 4);
        }
      } else {
        const frame = still ? 0 : Math.floor(this.t * 6);
        this.sprite("farmer", this.x, y, frame, this.target() < this.x);
        if (this.tool && this.errand === "back") this.px(this.x + 9, y + 3, 1, 9, "#8a5a2b");
      }
      if (this.state === "paused" && !this.errand) this.bubble(this.x + 2, y - 7, "…");
      if (this.state === "blocked" && !this.errand) this.bubble(this.x + 2, y - 7, "?");
      const cur = this.current;
      if (cur !== null && cur !== undefined && this.t < this.busyUntil && this.state === "running") {
        const gx = this.plotX(cur) + PLOT_W - 6, bob = still ? 0 : Math.round(Math.sin(this.t * (6 + 8 * this.util)));
        this.sprite("grafter", gx, GROUND + 6 + bob, Math.floor(this.t * (3 + 6 * this.util)));
        if (!still && Math.floor(this.t * 4) % 2) this.px(gx + 9, GROUND + 12, 2, 2, "#fff6a8");
      }
    }

    zzz(x, y) { this.px(x, y, 3, 1, "#ffffff"); this.px(x + 2, y + 1, 1, 1, "#ffffff"); this.px(x + 1, y + 2, 1, 1, "#ffffff"); this.px(x, y + 3, 3, 1, "#ffffff"); }

    bubble(x, y, ch) {
      this.px(x - 1, y - 1, 8, 7, "#ffffff"); this.px(x + 2, y + 6, 2, 1, "#ffffff");
      if (ch === "?") { this.px(x + 2, y, 3, 1, C.ink); this.px(x + 4, y + 1, 1, 1, C.ink); this.px(x + 3, y + 2, 1, 1, C.ink); this.px(x + 3, y + 4, 1, 1, C.ink); }
      else { this.px(x + 1, y + 3, 1, 1, C.ink); this.px(x + 3, y + 3, 1, 1, C.ink); this.px(x + 5, y + 3, 1, 1, C.ink); }
    }

    shed() {
      const x = SHED_X, y = GROUND + 2;
      this.px(x, y + 6, 28, 18, C.shed);
      for (let i = 0; i < 28; i += 4) this.px(x + i, y + 6, 1, 18, C.shedDark);
      this.px(x - 2, y + 2, 32, 5, C.roof); this.px(x, y - 1, 28, 3, C.roof); this.px(x + 3, y - 3, 22, 2, C.roofDark);
      this.px(SHED_DOOR - 1, y + 10, 7, 14, C.door);
      this.px(SHED_DOOR + 4, y + 16, 1, 1, C.gold);
      this.px(x + 3, y + 10, 6, 5, "#cfe6ff"); this.px(x + 5, y + 10, 1, 5, C.shedDark);
      this.px(BENCH_X - 1, GROUND + 18, 14, 2, C.bench); this.px(BENCH_X, GROUND + 20, 1, 4, C.bench); this.px(BENCH_X + 11, GROUND + 20, 1, 4, C.bench);
    }

    circle(cx0, cy0, rad, color) {
      for (let y = -rad; y <= rad; y++) {
        const w = Math.round(Math.sqrt(rad * rad - y * y));
        this.px(cx0 - w, cy0 + y, 2 * w + 1, 1, color);
      }
    }

    hills(base, height, color, freq, phase) {
      for (let x = 0; x < W; x++) {
        const h = Math.round(height * (0.55 + 0.45 * Math.sin(x * freq + phase)) * (0.8 + 0.2 * Math.sin(x * freq * 2.7)));
        this.px(x, base - h, 1, h, color);
      }
    }

    shimmer() {
      for (let k = 0; k < 6; k++) {
        const y = GROUND - 10 + k * 3, off = Math.sin(this.t * 3 + k) * 3;
        for (let x = (k * 37) % 40; x < W; x += 40) this.px(x + off, y, 10, 1, "rgba(255,240,200,0.35)");
      }
    }

    clouds() {
      const grey = this.weather === "storm" || this.weather === "overcast";
      if (this.weather === "heatwave") return;
      const n = grey ? 7 : 3, still = this.reduce.matches;
      for (let i = 0; i < n; i++) {
        const x = ((i * 97 + (still ? 0 : this.t * (4 + i))) % (W + 60)) - 30, y = 6 + (i % 3) * 9;
        const col = grey ? C.cloudDark : C.cloud;
        this.px(x, y + 4, 28, 6, col); this.px(x + 5, y, 16, 6, col); this.px(x + 14, y + 2, 12, 6, col);
      }
    }

    plot(i) {
      const x = this.plotX(i), y = GROUND + 10, st = this.stages[i];
      const wet = this.weather === "storm" && st !== "pending";
      this.px(x, y, PLOT_W, 14, wet ? C.soilWet : C.soil);
      for (let f = 0; f < 3; f++) this.px(x + 2, y + 3 + f * 4, PLOT_W - 4, 1, C.soilDark);
      this.px(x + PLOT_W - 7, y + 16, 7, 7, C.sign); this.px(x + PLOT_W - 7, y + 16, 7, 1, C.signDark);
      this.px(x + PLOT_W - 4, y + 23, 1, 2, C.signDark);
      this.digit(i, x + PLOT_W - 5, y + 17, C.ink);
      const mid = x + Math.floor(PLOT_W / 2) - 1, base = y + 6, still = this.reduce.matches;
      const sway = still ? 0 : Math.round(Math.sin(this.t * (this.weather === "storm" ? 6 : 2) + i));
      const frost = this.state === "blocked";
      if (st === "skipped") { this.px(x + 3, y + 2, PLOT_W - 6, 10, C.grassDark); return; }
      if (st === "pending") { for (let s = 0; s < 4; s++) this.px(x + 4 + s * 5, y + 6, 1, 1, "#d8b07a"); return; }
      if (st === "fail") {
        this.px(mid, base - 8, 1, 8, C.wilt); this.px(mid - 3, base - 6, 3, 1, C.wilt); this.px(mid + 1, base - 4, 3, 1, C.wilt);
        return;
      }
      if (st === "running" || st === "escalated" || st === "stopped") {
        // the sapling grows with the work: taller the busier the chips are
        const grow = st === "running" ? 5 + Math.round(6 * this.util) + (still ? 0 : Math.floor((this.t % 4) / 2)) : 9;
        const leaf = st === "stopped" ? "#8f9a86" : frost ? C.frost : this.weather === "heatwave" ? "#9fb04a" : C.leafLight;
        this.px(mid, base - grow, 1, grow, st === "stopped" ? "#7c7466" : C.leafDark);
        this.px(mid - 3 + sway, base - grow, 3, 2, leaf); this.px(mid + 1 + sway, base - grow + 2, 3, 2, leaf);
        if (st === "escalated") { this.px(mid + 6, base - 14, 2, 6, C.gold); this.px(mid + 6, base - 6, 2, 2, C.gold); }
        return;
      }
      const top = y - 26;
      this.px(mid, top + 14, 3, 18, C.trunk);
      const leaf = frost ? C.frost : this.state === "aborted" ? C.autumn[i % 4] : C.leaf;
      this.blob(mid + 1 + sway * 0.5, top + 8, 9, 8, leaf);
      this.blob(mid - 2 + sway * 0.5, top + 5, 5, 5, frost ? "#ffffff" : C.leafLight);
      const rr = rng(i + 11), apples = 3 + (i === 8 ? 4 : 0);
      for (let a = 0; a < apples; a++) {
        const ax = mid - 6 + Math.floor(rr() * 14), ay = top + 3 + Math.floor(rr() * 10);
        this.px(ax, ay, 2, 2, C.apple); this.px(ax, ay, 1, 1, C.appleHi);
      }
      const burst = this.bursts.find((b) => b.plot === i);
      if (burst && !still) {
        const k = 3 - (burst.until - this.t);
        for (let s = 0; s < 8; s++) {
          const ang = s * Math.PI / 4, d = 6 + k * 10;
          this.px(mid + Math.cos(ang) * d, top + 8 + Math.sin(ang) * d, 2, 2, s % 2 ? C.gold : "#ffffff");
        }
      }
    }

    blob(cx0, cy0, rx, ry, color) {
      for (let y = -ry; y <= ry; y++) {
        const w = Math.round(rx * Math.sqrt(1 - (y * y) / (ry * ry)));
        this.px(cx0 - w, cy0 + y, 2 * w + 1, 1, color);
      }
    }

    fence() {
      const y = H - 12;
      for (let x = 2; x < W; x += 12) { this.px(x, y, 3, 10, C.fence); this.px(x, y, 3, 1, C.fenceDark); }
      this.px(0, y + 3, W, 2, C.fence); this.px(0, y + 7, W, 2, C.fence);
    }

    precipitation() {
      const still = this.reduce.matches;
      if (this.weather === "storm") {
        for (const [x, y, s] of this.drops) this.px(x, y, 1, still ? 3 : 2 + Math.round(s * 2), C.drop);
      } else if (this.state === "blocked") {
        for (const [x, y, s] of this.drops.slice(0, 40)) this.px(x + (still ? 0 : Math.sin(this.t + s * 6) * 2), y, 1, 1, "#ffffff");
      } else if (this.state === "aborted") {
        this.drops.slice(0, 26).forEach(([x, y], k) => this.px(x, y, 2, 1 + (k % 2), C.autumn[k % 4]));
      } else if (this.weather === "clear" && !still) {
        for (let b = 0; b < 3; b++) {
          const bx = (b * 101 + this.t * 12) % W, by = GROUND - 6 + Math.sin(this.t * 2 + b) * 8;
          const open = Math.floor(this.t * 8 + b) % 2;
          this.px(bx, by, 1, 1, "#3b2414"); this.px(bx - 1 - open, by - 1, 1 + open, 1 + open, b % 2 ? "#ffd84a" : "#ff9ad5");
          this.px(bx + 1, by - 1, 1 + open, 1 + open, b % 2 ? "#ffd84a" : "#ff9ad5");
        }
      }
    }
  }

  window.OrchardScene = OrchardScene;
})();
