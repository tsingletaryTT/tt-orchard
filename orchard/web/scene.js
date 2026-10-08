// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// The orchard: a small pixel-art farm that shows what a run is doing.
//
// Nine plots, one per stage, grow from tilled soil to a fruit tree as the stages pass. The orchardist (the
// supervisor) walks to the stage that is running; the grafter (the coder) works its plot while the agent is
// busy; the sheepdog (the watchdog) runs across when it steps in. The weather is the run's state: sun while
// it grows, rain while it is paused, snow when it is blocked (frost), a storm when it crashed, falling leaves
// when it was aborted, and a full harvest basket when it is ripe. The sky follows the local time of day.
//
// Everything is drawn in code on a 320 x 128 canvas scaled up with crisp pixels; there are no image files.
// The scene is decorative: every fact it shows is also on the page as text. Under reduced motion it is
// drawn once per change, with no movement and no lightning.
"use strict";

(function () {
  const W = 320, H = 128, GROUND = 78, PLOT_W = 26, PLOT_GAP = 6;
  const PLOTS_X0 = Math.round((W - (9 * PLOT_W + 8 * PLOT_GAP)) / 2);
  const FPS = 12;

  const C = {
    sky: ["#7cc7f2", "#bfe8ff"], dusk: ["#f2a65a", "#ffd9a0"], night: ["#141a3a", "#2b3270"],
    storm: ["#4a5266", "#78839a"], rain: ["#8aa6bf", "#b9cddd"], snow: ["#c9d6e3", "#eef3f8"],
    hillFar: "#7fb85a", hillNear: "#5e9c3d", grass: "#6aa83e", grassDark: "#57902f", grassLight: "#86c24f",
    soil: "#7a4e2d", soilDark: "#5c3a20", soilWet: "#5a3a22", fence: "#a8743f", fenceDark: "#6e4522",
    trunk: "#6b4423", leaf: "#3e8e3e", leafLight: "#5fb04a", leafDark: "#2c6b2c", apple: "#d9443a", appleHi: "#ff8a7a",
    wilt: "#8a6b3a", frost: "#e8f2fa", sign: "#c98b4b", signDark: "#7a4e25", ink: "#3b2414",
    sun: "#ffd84a", moon: "#f3f0d8", cloud: "#ffffff", cloudDark: "#9aa3b5", drop: "#cfe6ff", flake: "#ffffff",
    autumn: ["#e07b2a", "#c9542a", "#f2b544", "#a8432a"], gold: "#f2c84b", basket: "#b5793a", basketDark: "#7a4e25",
  };

  // Sprites: one character per pixel, "." is transparent.
  const SPRITES = {
    farmer: { pal: { H: "#e8c35a", h: "#b5651d", S: "#f2c18d", E: "#3b2414", B: "#4a7fc1", P: "#3e5c8a", b: "#5a3a1e" },
      frames: [[
        "...HHHH...", "..HHHHHH..", ".hhhhhhhh.", "...SSSS...", "...SESE...", "...SSSS...", "..BBBBBB..",
        ".SBBBBBBS.", ".SBBBBBBS.", "..PPPPPP..", "..PPPPPP..", "..PP..PP..", "..PP..PP..", "..bb..bb.."],
      [
        "...HHHH...", "..HHHHHH..", ".hhhhhhhh.", "...SSSS...", "...SESE...", "...SSSS...", "..BBBBBB..",
        ".SBBBBBBS.", ".SBBBBBBS.", "..PPPPPP..", "..PPPPPP..", "...PP.PP..", "..PP...PP.", "..bb...bb."]] },
    grafter: { pal: { R: "#7a3b2a", S: "#f2c18d", E: "#3b2414", G: "#3f8f4f", A: "#e9d8a6", P: "#5a4a3a", b: "#4a2e18", X: "#c9ced6" },
      frames: [[
        "..RRRR..", ".RRRRRR.", "..SSSS..", "..SESE..", "..SSSS..", ".GGGGGG.", "SGAAAAGX", "SGAAAAGX",
        ".PPPPPP.", ".PP..PP.", ".PP..PP.", ".bb..bb."],
      [
        "..RRRR..", ".RRRRRR.", "..SSSS..", "..SESE..", "..SSSS..", ".GGGGGG.", "SGAAAAG.", "SGAAAAGX",
        ".PPPPPP.", ".PP..PP.", ".PP..PP.", ".bb..bb."]] },
    dog: { pal: { D: "#9a6233", W: "#f3ebdd", K: "#2b1b10" },
      frames: [[
        "........DD..", "D......DDKD.", ".DDDDDDDDDDK", ".DWWWWWDD...", ".DDDDDDDD...", ".D.D..D.D...", ".K.K..K.K..."],
      [
        "........DD..", ".D.....DDKD.", ".DDDDDDDDDDK", ".DWWWWWDD...", ".DDDDDDDD...", "..D.DD.D....", "..K.KK.K...."]] },
    basket: { pal: { B: "#b5793a", b: "#7a4e25", A: "#d9443a", a: "#ff8a7a", G: "#3e8e3e" },
      frames: [[
        "...G..G...", "..AaAAaA..", ".AAAAAAAA.", "bBbBbBbBbB", "BbBbBbBbBb", "bBbBbBbBbB", ".bBbBbBbB."]] },
  };

  const DIGITS = {
    0: ["111", "101", "101", "101", "111"], 1: ["010", "110", "010", "010", "111"], 2: ["111", "001", "111", "100", "111"],
    3: ["111", "001", "011", "001", "111"], 4: ["101", "101", "111", "001", "001"], 5: ["111", "100", "111", "001", "111"],
    6: ["111", "100", "111", "101", "111"], 7: ["111", "001", "010", "010", "010"], 8: ["111", "101", "111", "101", "111"],
  };

  function rng(seed) {          // small deterministic generator, so a scene looks the same each frame
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
      this.t = 0;
      this.farmerX = PLOTS_X0 - 16;
      this.dog = null;          // {x, until}
      this.bursts = [];         // [{plot, until}] fruit bursts when a stage passes
      this.busyUntil = 0;       // the grafter works while lines arrive
      this.grower = 0;
      this.flash = 0;
      this.reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
      this._last = 0;
      this._raf = null;
      this._seed = rng(7);
      this.drops = Array.from({ length: 70 }, () => [this._seed() * W, this._seed() * H, 0.6 + this._seed()]);
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
        if (prev[i] !== "pass" && st === "pass" && this._known) this.bursts.push({ plot: i, until: this.t + 3 });
      });
      this._known = true;
      this.stages = next;
      this.current = going ? current : null;
      if (this.isStill()) this.draw();
    }

    event(actor) {
      if (actor === "sheepdog") this.dog = { x: -14, until: this.t + 6 };
      else if (actor === "grafter") this.busyUntil = this.t + 20;
      else if (actor === "head grower") this.grower = this.t + 20;
      if (this.isStill()) this.draw();
    }

    isStill() { return this.reduce.matches || document.hidden; }

    start() {
      const tick = (now) => {
        this._raf = requestAnimationFrame(tick);
        if (this.isStill() || now - this._last < 1000 / FPS) return;
        const dt = this._last ? Math.min(0.5, (now - this._last) / 1000) : 0;
        this._last = now;
        this.t += dt;
        this.step(dt);
        this.draw();
      };
      this._raf = requestAnimationFrame(tick);
      this.draw();
    }

    // ---- motion ----
    step(dt) {
      const target = this.current === null || this.current === undefined
        ? (this.state === "ready-for-operator-review" ? this.plotX(8) + PLOT_W + 6 : PLOTS_X0 - 16)
        : this.plotX(this.current) - 11;
      const d = target - this.farmerX;
      this.farmerX += Math.sign(d) * Math.min(Math.abs(d), 28 * dt);
      if (this.dog) {
        this.dog.x += 70 * dt;
        if (this.dog.x > W + 14 || this.t > this.dog.until) this.dog = null;
      }
      this.bursts = this.bursts.filter((b) => b.until > this.t);
      const fall = this.state === "blocked" ? 18 : this.state === "aborted" ? 14 : 110;
      for (const p of this.drops) {
        p[1] += fall * p[2] * dt;
        p[0] += (this.state === "aborted" ? 8 : this.state === "stopped-or-crashed" ? -14 : 0) * dt;
        if (p[1] > GROUND + 40) { p[1] = -4; p[0] = this._seed() * W; }
        if (p[0] < -4) p[0] = W;
        if (p[0] > W + 4) p[0] = 0;
      }
      // A storm lightens the sky now and then, softly and rarely (never a strobe).
      if (this.state === "stopped-or-crashed" && !this.reduce.matches && this._seed() < dt / 9) this.flash = 0.6;
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

    skyColors() {
      if (this.state === "stopped-or-crashed") return C.storm;
      if (this.state === "paused") return C.rain;
      if (this.state === "blocked") return C.snow;
      const h = new Date().getHours();
      if (h >= 20 || h < 6) return C.night;
      if (h >= 18 || h < 7) return C.dusk;
      return C.sky;
    }

    draw() {
      const cx = this.cx, still = this.reduce.matches;
      const [top, bottom] = this.skyColors();
      const g = cx.createLinearGradient(0, 0, 0, GROUND);
      g.addColorStop(0, top); g.addColorStop(1, bottom);
      cx.fillStyle = g; cx.fillRect(0, 0, W, GROUND);
      const night = top === C.night[0];
      if (night) for (const [x, y] of this.stars) this.px(x, y, 1, 1, (x + Math.floor(this.t * 2)) % 7 ? "#fffbe0" : "#9aa0d0");
      // sun or moon
      if (["running", "ready-for-operator-review", "not-started", "aborted"].includes(this.state)) {
        if (night) { this.circle(282, 20, 8, C.moon); this.circle(286, 17, 7, top); }
        else this.circle(282, 20, 9, C.sun);
      }
      this.clouds();
      // hills
      this.hills(GROUND - 4, 22, C.hillFar, 0.021, 3);
      this.hills(GROUND, 14, C.hillNear, 0.034, 11);
      // ground and grass
      const frost = this.state === "blocked";
      this.px(0, GROUND, W, H - GROUND, frost ? "#cfe0d6" : this.state === "aborted" ? "#9a9a48" : C.grass);
      const r = rng(3);
      for (let i = 0; i < 220; i++) {
        const x = Math.floor(r() * W), y = GROUND + 2 + Math.floor(r() * (H - GROUND - 2));
        this.px(x, y, 1, 2, frost ? C.frost : r() < 0.5 ? C.grassDark : C.grassLight);
      }
      // the plots, back to front
      for (let i = 0; i < 9; i++) this.plot(i);
      this.fence();
      // characters
      const walking = !still && Math.abs(this.farmerX - this.targetX()) > 1;
      const frame = walking ? Math.floor(this.t * 6) : 0;
      const workingPlot = this.current;
      if (workingPlot !== null && workingPlot !== undefined && this.t < this.busyUntil && this.state === "running") {
        const gx = this.plotX(workingPlot) + PLOT_W - 6;
        this.sprite("grafter", gx, GROUND + 18 - 12 + (still ? 0 : Math.round(Math.sin(this.t * 8))), Math.floor(this.t * 4));
        if (!still && Math.floor(this.t * 4) % 2) this.px(gx + 9, GROUND + 12, 2, 2, "#fff6a8");   // a snip of the shears
      }
      if (this.state !== "not-started") this.sprite("farmer", this.farmerX, GROUND + 6, frame, this.targetX() < this.farmerX);
      if (this.t < this.grower && this.state === "running") this.sprite("farmer", 6, GROUND + 6, 0);
      if (this.dog) this.sprite("dog", this.dog.x, GROUND + 26, Math.floor(this.t * 10));
      if (this.state === "ready-for-operator-review") this.sprite("basket", this.plotX(8) + PLOT_W + 18, GROUND + 22);
      // weather in front
      this.weather();
      if (this.flash > 0) { cx.fillStyle = `rgba(255,255,240,${this.flash * 0.35})`; cx.fillRect(0, 0, W, H); }
    }

    targetX() {
      if (this.current === null || this.current === undefined)
        return this.state === "ready-for-operator-review" ? this.plotX(8) + PLOT_W + 6 : PLOTS_X0 - 16;
      return this.plotX(this.current) - 11;
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

    clouds() {
      const dark = ["paused", "stopped-or-crashed", "blocked"].includes(this.state);
      const n = dark ? 6 : 3, still = this.reduce.matches;
      for (let i = 0; i < n; i++) {
        const x = ((i * 97 + (still ? 0 : this.t * (4 + i))) % (W + 60)) - 30, y = 8 + (i % 3) * 9;
        const col = dark ? C.cloudDark : C.cloud;
        this.px(x, y + 4, 28, 6, col); this.px(x + 5, y, 16, 6, col); this.px(x + 14, y + 2, 12, 6, col);
      }
    }

    plot(i) {
      const x = this.plotX(i), y = GROUND + 10, st = this.stages[i];
      const wet = this.state === "paused" && st !== "pending";
      // tilled soil with furrows
      this.px(x, y, PLOT_W, 14, wet ? C.soilWet : C.soil);
      for (let f = 0; f < 3; f++) this.px(x + 2, y + 3 + f * 4, PLOT_W - 4, 1, C.soilDark);
      // a little wooden sign with the stage number
      this.px(x + PLOT_W - 7, y + 16, 7, 7, C.sign); this.px(x + PLOT_W - 7, y + 16, 7, 1, C.signDark);
      this.px(x + PLOT_W - 4, y + 23, 1, 2, C.signDark);
      this.digit(i, x + PLOT_W - 5, y + 17, C.ink);
      const mid = x + Math.floor(PLOT_W / 2) - 1, base = y + 6, still = this.reduce.matches;
      const sway = still ? 0 : Math.round(Math.sin(this.t * 2 + i));
      const frost = this.state === "blocked";
      if (st === "skipped") {
        this.px(x + 3, y + 2, PLOT_W - 6, 10, C.grassDark);                      // left fallow, grassed over
        return;
      }
      if (st === "pending") { for (let s = 0; s < 4; s++) this.px(x + 5 + s * 5, y + 6, 1, 1, "#d8b07a"); return; }
      if (st === "fail") {                                                          // a wilted stalk
        this.px(mid, base - 8, 1, 8, C.wilt); this.px(mid - 3, base - 6, 3, 1, C.wilt); this.px(mid + 1, base - 4, 3, 1, C.wilt);
        return;
      }
      if (st === "running" || st === "escalated" || st === "stopped") {           // a growing sapling
        const grow = st === "running" && !still ? 6 + Math.floor((this.t % 8)) : 9;
        const leaf = st === "stopped" ? "#8f9a86" : frost ? C.frost : C.leafLight;
        this.px(mid, base - grow, 1, grow, st === "stopped" ? "#7c7466" : C.leafDark);
        this.px(mid - 3 + sway, base - grow, 3, 2, leaf); this.px(mid + 1 + sway, base - grow + 2, 3, 2, leaf);
        if (st === "escalated") { this.px(mid + 6, base - 14, 2, 6, C.gold); this.px(mid + 6, base - 6, 2, 2, C.gold); }
        if (st === "running" && this.state === "running" && !still && Math.floor(this.t * 3) % 3 === 0)
          this.px(mid - 1, base - grow - 3, 1, 1, "#fff6a8");                    // a glint: it's growing
        return;
      }
      // pass: a fruit tree; stage 8 bears the most
      const top = y - 26;
      this.px(mid, top + 14, 3, 18, C.trunk);
      const leaf = frost ? C.frost : this.state === "aborted" ? C.autumn[i % 4] : C.leaf;
      this.blob(mid + 1 + sway * 0.5, top + 8, 10, 8, leaf);
      this.blob(mid - 2 + sway * 0.5, top + 5, 6, 5, frost ? "#ffffff" : C.leafLight);
      const r = rng(i + 11), apples = 3 + (i === 8 ? 4 : 0);
      for (let a = 0; a < apples; a++) {
        const ax = mid - 7 + Math.floor(r() * 16), ay = top + 3 + Math.floor(r() * 10);
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
      const y = H - 14;
      for (let x = 2; x < W; x += 12) { this.px(x, y, 3, 12, C.fence); this.px(x, y, 3, 1, C.fenceDark); }
      this.px(0, y + 3, W, 2, C.fence); this.px(0, y + 8, W, 2, C.fence);
      this.px(0, y + 5, W, 1, C.fenceDark); this.px(0, y + 10, W, 1, C.fenceDark);
    }

    weather() {
      const st = this.state, still = this.reduce.matches;
      if (st === "paused" || st === "stopped-or-crashed") {
        for (const [x, y, s] of this.drops) this.px(x, y, 1, still ? 3 : 2 + Math.round(s * 2), C.drop);
      } else if (st === "blocked") {
        for (const [x, y, s] of this.drops) this.px(x + (still ? 0 : Math.sin(this.t + s * 6) * 2), y, s > 1.2 ? 2 : 1, s > 1.2 ? 2 : 1, C.flake);
      } else if (st === "aborted") {
        this.drops.slice(0, 26).forEach(([x, y, s], k) => this.px(x, y, 2, 1 + (k % 2), C.autumn[k % 4]));
      } else if ((st === "running" || st === "ready-for-operator-review") && !still) {
        for (let b = 0; b < 3; b++) {                                              // butterflies
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
