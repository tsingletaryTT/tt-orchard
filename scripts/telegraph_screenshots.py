#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Take screenshots of the Telegraph view of the real `tt-orchard ui` page, reading a real telegraph ledger.

    python3 -m orchard.cli ui --config <bringup.toml with a [telegraph] table> --port 8782 &
    python scripts/telegraph_screenshots.py http://127.0.0.1:8782/ docs/ui   # then copy to site/assets/ui-telegraph-<style>.png; any Python with `playwright`; uses system Chrome

The page only reads the ledger, so no key is needed and no call is made. Before each shot the home directory, the host name and the
user name are replaced in the page text, and the script stops if one is still visible or the text looks like a credential.
"""
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

USAGE = "usage: telegraph_screenshots.py URL OUTPUT_DIR [OUTPUT_DIR ...]   (at least one output directory is required)"

SHOTS = [("operator-telegraph.png", "operator"), ("playful-telegraph.png", "playful")]


def main(*argv: str) -> int:
    # Check the arguments first, before playwright is imported or a browser starts: a call with no output directory used to run the whole
    # session and write no picture (issue 131).
    if len(argv) < 2:
        print(USAGE, file=sys.stderr)
        return 2
    url, out_dirs = argv[0], argv[1:]
    from site_screenshots import CHROME, REDACT, credential_hits
    from playwright.sync_api import expect, sync_playwright
    dirs = [Path(d) for d in out_dirs]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    host, user = socket.gethostname(), Path.home().name
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        for name, style in SHOTS:
            ctx = browser.new_context(viewport={"width": 1280, "height": 1480}, color_scheme="dark")
            page = ctx.new_page()
            page.set_default_timeout(20000)
            page.goto(f"{url}?mode={style}")
            page.click("#view-telegraph-btn")
            view = page.locator("#telegraph-view")
            expect(view).to_be_visible()
            expect(page.locator("#tg-recent tr").first).to_be_visible()        # the ledger has been read and drawn
            page.wait_for_timeout(1500)
            page.evaluate(REDACT, [host, user])

            def check() -> None:
                text = page.inner_text("body")
                assert "/home/" not in text, f"a home path is visible in {name}"
                assert host.lower() not in text.lower(), f"the host name is visible in {name}"
                assert not credential_hits(text), f"{name}: the page text looks like it holds a credential: {credential_hits(text)}"
            check()
            written = []
            for d in dirs:
                page.evaluate("window.scrollTo(0, 0)")
                page.screenshot(path=str(d / name))               # the top of the page: the bar, the banner, the numbers, the tables
                written.append(d / name)
            try:
                check()
            except AssertionError:
                for f in written:
                    f.unlink(missing_ok=True)
                raise
            print("wrote", name, flush=True)
            ctx.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
