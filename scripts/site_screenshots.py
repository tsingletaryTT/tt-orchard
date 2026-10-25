#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Take the site's screenshots of the real `tt-orchard ui` page, running against real runs.

    python3 -m orchard.cli ui --port 8782 &
    python scripts/site_screenshots.py http://127.0.0.1:8782/ site/assets     # any Python with the `playwright` package; uses system Chrome

Before each shot the home directory, the host name and the user name are replaced in the page text, and the script stops if one is still visible.
A shot is a picture of one real run in one style; nothing is mocked.
"""
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from orchard import scrub  # noqa: E402

from playwright.sync_api import expect, sync_playwright

CHROME = "/usr/bin/google-chrome"
SHOTS = [  # file, run directory name (exact), style
    ("ui-running-playful.png", "huihui-ai--huihui-qwen3-coder-30b-a3b-instruct-abliterated", "playful"),
    ("ui-running-operator.png", "huihui-ai--huihui-qwen3-coder-30b-a3b-instruct-abliterated", "operator"),
    ("ui-finished-playful.png", "proprietarylegal--qwen3.8-27b-cjk-trim", "playful"),
    ("ui-finished-operator.png", "proprietarylegal--qwen3.8-27b-cjk-trim", "operator"),
    ("ui-blocked-operator.png", "perplexity-ai--pplx-decider-v1.2", "operator"),
]
SECRET_SHAPES = (re.compile(r"(?i)\b(secret|password|passwd|api[_-]?key|authorization|bearer)\b"), re.compile(r"(?i)\btokens?\b[\"']?\s*[:=]"),
                 re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), re.compile(r"\bAKIA[0-9A-Z]{16}\b"))


def credential_hits(text: str) -> list[str]:
    """What in the visible page text looks like a credential: the scrub module's token and key patterns, plus secret words and shapes. A hit stops the script."""
    hits = [what for what, pattern in scrub.TOKEN_PATTERNS if pattern.search(text)]
    hits += [p.pattern for p in SECRET_SHAPES if p.search(text)]
    return hits


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
            # the list row shows the run directory name on a line of its own: pick the one row where a line equals the name, and fail otherwise
            rows = page.locator("#runs li")
            hits = [i for i in range(rows.count()) if fragment in [ln.strip() for ln in rows.nth(i).inner_text().splitlines()]]
            assert len(hits) == 1, f"{name}: {len(hits)} rows show the run {fragment!r}; need exactly one"
            rows.nth(hits[0]).locator("button, a").first.click()
            expect(page.locator("#run")).to_contain_text(fragment)
            page.wait_for_timeout(2500)
            page.evaluate(REDACT, [host, user])

            def check() -> None:
                text = page.inner_text("body")
                assert "/home/" not in text, f"a home path is visible in {name}"
                assert host.lower() not in text.lower() or host.lower() in ("this-box", ""), f"the host name is visible in {name}"
                assert not credential_hits(text), f"{name}: the page text looks like it holds a credential: {credential_hits(text)}"
            check()
            page.screenshot(path=str(out_dir / name), full_page=False)
            try:
                check()                       # the live page may have changed while the picture was taken: if it now fails, the picture is not kept
            except AssertionError:
                (out_dir / name).unlink(missing_ok=True)
                raise
            print("wrote", name, flush=True)
            ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
