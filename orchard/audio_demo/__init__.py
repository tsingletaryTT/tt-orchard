# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The demo an audio package carries: a local job server, a client and a Gradio page for Stable Audio 3 small music on a Tenstorrent chip.

The three modules next to this file are written to be copied as they are into a package's `demo/` directory and to run there with nothing else of this
repository beside them (`audio_server.py` and `audio_client.py` use only the standard library; `audio_app.py` also needs Gradio). This module holds what
a packaging step needs:

  * `stage_demo(package_dir)` copies the files and a requirements file into `<package_dir>/demo/` and returns each file's path and sha256;
  * `card_section()` is the README text for the package card that says how to start the page. It carries no performance number and no claim about
    commercial use, so the card gates (`orchard.package_card.card_problems`) accept it next to any licence; tests/test_audio_demo_package.py checks that.

Nothing here imports Gradio. Wiring a packaging step to call these two functions is the audio package work's to do (see docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md,
section 6): no audio package is built on main yet.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ("audio_client.py", "audio_server.py", "audio_app.py")
DEMO_DIR = "demo"
GRADIO_REQUIREMENT = "gradio>=6.20,<7"
SERVER_PORT = 20100
UI_PORT = 7872


class DemoError(Exception):
    """The demo could not be staged."""


def stage_demo(package_dir) -> list[dict]:
    """Copy the demo into `<package_dir>/demo/`. Returns `[{"path": "demo/<name>", "sha256": ...}]` for every file written, requirements included.

    A `demo` entry that is a link is refused, so a package directory cannot be used to write somewhere else.
    """
    pkg = Path(package_dir)
    if not pkg.is_dir():
        raise DemoError(f"{pkg} is not a directory")
    dest = pkg / DEMO_DIR
    if dest.is_symlink():
        raise DemoError(f"{dest} is a link; the demo is not written through links")
    dest.mkdir(exist_ok=True)
    out = []
    contents = {name: (HERE / name).read_bytes() for name in FILES}
    contents["requirements.txt"] = (GRADIO_REQUIREMENT + "\n").encode()
    for name, data in contents.items():
        target = dest / name
        if target.is_symlink():
            raise DemoError(f"{target} is a link; the demo is not written through links")
        target.write_bytes(data)
        os.chmod(target, 0o644)
        out.append({"path": f"{DEMO_DIR}/{name}", "sha256": hashlib.sha256(data).hexdigest()})
    return out


def card_section() -> str:
    """The README text that tells a reader how to start the page. Plain steps, with no number that would belong in the card's Numbers table."""
    return f"""## Try it in a browser

The package carries a small local server and a Gradio page in `demo/`. The page takes a prompt, a length, the number of steps, a guidance scale and a
seed (with a button for a random one), plays the clip and shows the settings that made it. It says so when the server is busy, not running or still
loading, and when a clip ends before the length asked for.

```
pip install -r demo/requirements.txt
gozer run --chips 1 -- python demo/audio_server.py --generator chip --src-dir <stable-audio-3 checkout> --script <chip_generate.py> --ui
```

Open http://127.0.0.1:{UI_PORT}. The job server listens on http://127.0.0.1:{SERVER_PORT}. Both bind the local machine only; use `--host` and `--ui-host`
to change that on a network you trust, because nothing in the page signs anyone in. Without a chip, `--generator fake` runs the same server and page
with a plain tone in place of music, so the page can be tried anywhere.

The server makes one clip at a time and keeps a short queue; when the queue is full a new request is refused and the page says the server is busy.
This generator loads the model again for each clip, and the page shows the stage times the run reports. Only 10 second clips have been checked on the
chip. Longer lengths are allowed up to the model's own limit and are not checked. Listening to the clips has not been part of any check."""
