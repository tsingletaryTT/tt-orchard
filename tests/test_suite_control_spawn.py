# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`suite-control run` passes SIGTERM on to the launcher before it releases the lease (review of PR 80, item 4)."""
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchard import suite_control as sc  # noqa: E402


def child(tmp_path, body):
    f = tmp_path / "child.py"
    f.write_text(body)
    return [sys.executable, str(f)]


def test_the_launcher_gets_sigterm_and_the_run_raises_terminated(tmp_path):
    ready, marker = tmp_path / "ready", tmp_path / "marker"
    argv = child(tmp_path, f"import signal, sys, time\n"
                           f"signal.signal(signal.SIGTERM, lambda *a: (open({str(marker)!r}, 'w').write('t'), sys.exit(0)))\n"
                           f"open({str(ready)!r}, 'w').write('1')\ntime.sleep(60)\n")

    def term():
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGTERM)
    t = threading.Thread(target=term)
    t.start()
    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(sc.Terminated):
            sc._spawn(argv, dict(os.environ), str(tmp_path))
    finally:
        t.join()
    assert marker.exists()
    assert signal.getsignal(signal.SIGTERM) == previous          # the handler is put back


def test_a_normal_exit_returns_the_launchers_status(tmp_path):
    assert sc._spawn(child(tmp_path, "import sys\nsys.exit(6)\n"), dict(os.environ), str(tmp_path)) == 6
