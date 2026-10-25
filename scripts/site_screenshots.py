#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Take the site's screenshots of the real `tt-orchard ui` page, running against real runs.

    python3 -m orchard.cli ui --port 8782 &
    python scripts/site_screenshots.py http://127.0.0.1:8782/ site/assets     # any Python with the `playwright` package; uses system Chrome

Before each shot the home directory, the host name and the user name are replaced in the page text, and the script stops if one is still visible.
A shot is a picture of one real run in one style; nothing is mocked.
"""
import socket
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/usr/bin/google-chrome"
SHOTS = [  # file, run-name fragment, style
    ("ui-running-playful.png", "huihui-ai--huihui-qwen3", "playful"),
    ("ui-running-operator.png", "huihui-ai--huihui-qwen3", "operator"),
    ("ui-finished-playful.png", "proprietarylegal--qwen3.8-27b-cjk-trim", "playful"),
    ("ui-finished-operator.png", "proprietarylegal--qwen3.8-27b-cjk-trim", "operator"),
    ("ui-blocked-operator.png", "perplexity-ai--pplx", "operator"),
]
REDACT = """([host, user]) => {
  const re = [[/\\/home\\/[^\\s·"']*/g, "~"], [new RegExp(host, "gi"), "this-box"], [new RegExp("\\\\b" + user + "\\\\b", "g"), "user"]];
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
  while ((n = w.nextNode())) { for (const [r, s] of re) n.nodeValue = n.nodeValue.replace(r, s); }
  for (const e of document.querySelectorAll("input, textarea")) { for (const [r, s] of re) e.value = e.value.replace(r, s); }
}"""


def main(url: str, out: str) -> int:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    host, user = socket.gethostname(), Path.home().name
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        for name, fragment, style in SHOTS:
            ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, color_scheme="dark")
            page = ctx.new_page()
            page.set_default_timeout(20000)
            page.goto(f"{url}?mode={style}")
            page.wait_for_selector("#runs li")
            # pick the run by the name the list row carries (the row shows the model id; the run directory is in its title)
            rows = page.locator("#runs li")
            for i in range(rows.count()):
                if fragment.split("--")[-1].split("-")[0] in (rows.nth(i).inner_text() + (rows.nth(i).get_attribute("title") or "")).lower():
                    rows.nth(i).locator("button, a").first.click()
                    break
            page.wait_for_timeout(2500)
            page.evaluate(REDACT, [host, user])
            text = page.inner_text("body")
            assert "/home/" not in text, f"a home path is visible in {name}"
            assert host.lower() not in text.lower() or host.lower() in ("this-box", ""), f"the host name is visible in {name}"
            page.screenshot(path=str(out_dir / name), full_page=False)
            print("wrote", name, flush=True)
            ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
