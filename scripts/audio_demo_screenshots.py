#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Take screenshots of the Stable Audio demo page running against its job server.

    python orchard/audio_demo/audio_server.py --generator fake --port 20111 --ui --ui-port 7881 --fake-delay 8 &
    python scripts/audio_demo_screenshots.py http://127.0.0.1:7881/ http://127.0.0.1:20111 docs/audio-demo     # any Python with `playwright`; uses system Chrome

Pick ports that are free. The page shows what the server is: with `--generator fake` the status line says "fake generator ... no chip" and the clip is a
plain tone, so a picture taken this way is a picture of the page and says so. Pictures of the real model need the server started with `--generator chip`
under a gozer lease. Before each shot the home directory, the host name and the user name are replaced in the page text, and the script stops if one is
still visible or the text looks like a credential.
"""
import json
import socket
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from site_screenshots import CHROME, REDACT, credential_hits  # noqa: E402

from playwright.sync_api import expect, sync_playwright  # noqa: E402


def post_job(server: str, seed: int) -> None:
    """Put a job straight on the server, to fill its queue."""
    req = urllib.request.Request(server + "/v1/audio/jobs", data=json.dumps({"prompt": "queue filler", "duration": 5, "seed": seed}).encode(),
                                 method="POST", headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()


def main(url: str, server: str, out: str) -> int:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    host, user = socket.gethostname(), Path.home().name
    health = json.loads(urllib.request.urlopen(server + "/v1/health", timeout=10).read())
    assert health["status"] == "ok", health
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1280, "height": 900}, color_scheme="dark")
        page = ctx.new_page()
        page.set_default_timeout(60000)

        def shot(name: str) -> None:
            page.evaluate(REDACT, [host, user])
            text = page.inner_text("body")
            assert "/home/" not in text, f"a home path is visible in {name}"
            assert host.lower() not in text.lower(), f"the host name is visible in {name}"
            assert not credential_hits(text), f"{name}: the page text looks like it holds a credential: {credential_hits(text)}"
            page.screenshot(path=str(out_dir / name))
            print("wrote", name, flush=True)

        page.goto(url)
        expect(page.get_by_text("Server ready")).to_be_visible()
        shot("audio-demo-ready.png")

        page.get_by_role("button", name="Generate").click()
        expect(page.get_by_text("Making the clip").first).to_be_visible()
        page.wait_for_timeout(1500)
        shot("audio-demo-generating.png")

        expect(page.get_by_text("Done:")).to_be_visible()
        page.wait_for_timeout(1500)
        shot("audio-demo-done.png")

        # fill the queue (one running, the rest waiting), then ask for one more through the page
        for i in range(health["queue_max"] + 1):
            post_job(server, 900 + i)
        page.get_by_role("button", name="Generate").click()
        expect(page.get_by_text("queue is full")).to_be_visible()
        shot("audio-demo-busy.png")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
