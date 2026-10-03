"""ServerControl's stop checks with the real ps, pgrep, ss and curl against a fake server.

The FakeRun tests check how ServerControl reads command output written by hand. This file checks
that the real tools print what ServerControl expects. It opens no device. A skip is not evidence.
"""
import shutil
import socket
import sys
import time
from pathlib import Path

import pytest

from orchard.server import ServerControl, ServerSpec

MISSING = [t for t in ("ps", "pgrep", "ss", "curl") if shutil.which(t) is None]
pytestmark = pytest.mark.skipif(bool(MISSING), reason=f"missing {MISSING}; a skip is not evidence")
FAKE = Path(__file__).resolve().parent.parent / "orchard" / "fake_server.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_tools_see_a_live_server_and_then_see_it_gone(tmp_path):
    port = free_port()
    spec = ServerSpec("test/fake", "process", port, "fake",
                      argv=(sys.executable, str(FAKE), "--port", str(port), "--answer", "forty-two"))
    srv = ServerControl(spec, log_path=str(tmp_path / "server.log"), ready_poll_s=0.1,
                        sleep=time.sleep)
    srv.start(None)
    try:
        srv.wait_ready(20)
        live = srv.confirm_stopped()
        assert live.checks == {"process": False, "process_group": False, "port_closed": False,
                               "health_refused": False}, live.evidence
        assert srv.ask("anything") == "forty-two"
    finally:
        srv.stop()
    gone = srv.confirm_stopped()
    assert gone.stopped, gone.evidence
