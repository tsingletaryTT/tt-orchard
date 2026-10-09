# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/kept_coder.py: the record a lab run leaves for the next, and `tt-orchard coder status|stop`."""
from types import SimpleNamespace

from orchard import kept_coder
from orchard.adapters import AdapterError
from orchard.server import StopCheck

IDENT = {"target": "org/coder", "kind": "container", "port": 8001, "profile": "p300", "image_id": None,
         "model": "Qwen/Qwen3-Coder-Next", "chips": 2}
LEASE = {"lease_id": "f78464", "chips": ["0000:03:00.0", "0000:04:00.0"], "dev_indices": [2, 3],
         "env": {"TT_VISIBLE_DEVICES": "0000:03:00.0,0000:04:00.0"}, "units": ["B1"]}


class Server:
    def __init__(self, stops=True):
        self.running, self.stops, self.calls = True, stops, []

    def adopt(self, rec):
        self.calls.append("adopt")

    def stop(self):
        self.calls.append("stop")
        self.running = not self.stops

    def confirm_stopped(self):
        return StopCheck(not self.running, {}, {})


class Adapter:
    def __init__(self, fail=False):
        self.fail, self.released = fail, []

    def release(self, lease):
        if self.fail:
            raise AdapterError("exit 15")
        self.released.append(lease.lease_id)


def setup(tmp_path):
    cfg = SimpleNamespace(cache_root=tmp_path / "cache", runs_root=tmp_path / "runs", gozer="gozer")
    kept_coder.write(cfg.cache_root, {"identity": IDENT, "lease": LEASE, "holder_pids": [9],
                                      "server": {"kind": "container"}, "canary": "x", "run_dir": "/r/one",
                                      "left_at": 0})
    return cfg


def main(cfg, action, out, **kw):
    args = SimpleNamespace(action=action, yes=kw.pop("yes", True))
    clock = iter(range(0, 10000, 10))
    return kept_coder.main(cfg=cfg, args=args, say=out.append, probe=lambda e, m: True,
                           clock=lambda: next(clock), sleep=lambda s: None, **kw)


def test_mismatch_names_each_difference():
    assert kept_coder.mismatch({"identity": IDENT}, IDENT) is None
    why = kept_coder.mismatch({"identity": IDENT}, {**IDENT, "port": 8000, "image_id": "sha256:b"})
    assert "port (8001, this run 8000)" in why and "image_id" in why
    assert kept_coder.mismatch({}, IDENT).startswith("the kept coder differs")


def test_forget_removes_only_the_named_lease(tmp_path):
    cfg = setup(tmp_path)
    kept_coder.forget(cfg.cache_root, "other")
    assert kept_coder.read(cfg.cache_root) is not None
    kept_coder.forget(cfg.cache_root, "f78464")
    assert kept_coder.read(cfg.cache_root) is None


def test_status_shows_the_kept_coder_and_changes_nothing(tmp_path):
    cfg, out, server = setup(tmp_path), [], Server()
    assert main(cfg, "status", out, server=server) == 0
    text = "\n".join(out)
    assert "org/coder on port 8001" in text and "f78464" in text and "serving Qwen/Qwen3-Coder-Next: yes" in text
    assert server.calls == [] and kept_coder.read(cfg.cache_root) is not None


def test_stop_stops_the_coder_releases_its_lease_and_forgets_it(tmp_path):
    cfg, out, server, adapter = setup(tmp_path), [], Server(), Adapter()
    assert main(cfg, "stop", out, server=server, adapter=adapter) == 0
    assert server.calls == ["adopt", "stop"] and adapter.released == ["f78464"]
    assert kept_coder.read(cfg.cache_root) is None


def test_stop_asks_first_and_a_no_leaves_it_running(tmp_path):
    cfg, out, server = setup(tmp_path), [], Server()
    args = SimpleNamespace(action="stop", yes=False)
    assert kept_coder.main(cfg=cfg, args=args, say=out.append, ask=lambda q: "n", probe=lambda e, m: True,
                           server=server, adapter=Adapter()) == 1
    assert server.calls == [] and kept_coder.read(cfg.cache_root) is not None


def test_a_coder_that_will_not_stop_keeps_its_lease(tmp_path):
    cfg, out, adapter = setup(tmp_path), [], Adapter()
    assert main(cfg, "stop", out, server=Server(stops=False), adapter=adapter) == 1
    assert adapter.released == [] and kept_coder.read(cfg.cache_root) is not None
    assert "not confirmed stopped" in out[-1]


def test_a_failed_release_is_reported_and_the_record_kept(tmp_path):
    cfg, out = setup(tmp_path), []
    assert main(cfg, "stop", out, server=Server(), adapter=Adapter(fail=True)) == 1
    assert "gozer release f78464" in out[-1] and kept_coder.read(cfg.cache_root) is not None


def test_with_nothing_kept_status_says_so(tmp_path):
    cfg = SimpleNamespace(cache_root=tmp_path / "cache", runs_root=tmp_path / "runs", gozer="gozer")
    out = []
    assert main(cfg, "status", out) == 0 and "no coder is kept up" in out[0]
