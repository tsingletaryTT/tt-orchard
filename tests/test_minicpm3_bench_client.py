# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The MiniCPM3 benchmark client (docs/run-logs/evidence/2026-11-08-minicpm3-can/bench.py) records whether the server it measured is on this host.

Review of PR 628: `url.startswith("http://127.0.0.1")` is true for `http://127.0.0.1.example:20130` and `http://127.0.0.1@remote.example`, both
of which point at another host. The check now reads the parsed host name, and the client refuses a URL that is not a loopback address."""
import importlib.util
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "docs" / "run-logs" / "evidence" / "2026-11-08-minicpm3-can" / "bench.py"
spec = importlib.util.spec_from_file_location("minicpm3_bench_under_test", BENCH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("url", ["http://127.0.0.1:20130", "http://127.0.0.1", "http://localhost:20130", "http://[::1]:20130"])
def test_loopback_addresses_are_loopback(url):
    assert bench.is_loopback(url) is True


@pytest.mark.parametrize("url", [
    "http://127.0.0.1.example:20130",          # a host name that only starts like the address
    "http://127.0.0.1@remote.example:20130",   # the part before @ is a user name
    "http://remote.example/127.0.0.1",
    "https://127.0.0.1:20130",                 # the client speaks plain http to a local server only
    "ftp://127.0.0.1/",
    "http://",
    "",
])
def test_other_hosts_are_not_loopback(url):
    assert bench.is_loopback(url) is False


def test_the_client_refuses_a_url_that_is_not_loopback(monkeypatch, tmp_path):
    out = tmp_path / "out.json"
    monkeypatch.setattr("sys.argv", ["bench.py", "--url", "http://127.0.0.1.example:20130", "--tokenizer-dir", str(tmp_path), "--out", str(out), "--skip-decode"])
    with pytest.raises(SystemExit) as exc:
        bench.main()
    assert exc.value.code == 2 and not out.exists()
