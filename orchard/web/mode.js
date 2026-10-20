// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// The page style: "playful" (the farm look) or "operator" (plain type, state first, no scene).
// This file is a plain script in <head>, with no defer, so the style is on <html> before the first paint and the page never
// flashes the wrong look. The choice comes from `?mode=` (a link someone sends you), else from what this browser remembered,
// else "playful". Anything that is not one of the two names is ignored, so a link or a stored value cannot inject a style.
(function () {
  const MODES = ["playful", "operator"];
  const KEY = "orchard.mode";

  function stored() {
    try { return window.localStorage.getItem(KEY); } catch { return null; }      // a private window can throw
  }
  function remember(mode) {
    try { window.localStorage.setItem(KEY, mode); } catch { /* the choice just lasts until the page closes */ }
  }
  function valid(value) { return MODES.includes(value) ? value : null; }

  const fromLink = valid(new URLSearchParams(window.location.search).get("mode"));
  if (fromLink) remember(fromLink);
  let current = fromLink || valid(stored()) || "playful";
  document.documentElement.dataset.mode = current;

  function apply(mode) {
    current = mode;
    document.documentElement.dataset.mode = mode;
    remember(mode);
    for (const b of document.querySelectorAll("#mode-switch button[data-mode]")) {
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    for (const b of document.querySelectorAll("#mode-switch button[data-mode]")) {
      b.setAttribute("aria-pressed", String(b.dataset.mode === current));
      b.addEventListener("click", () => { if (valid(b.dataset.mode)) apply(b.dataset.mode); });
    }
  });
})();
