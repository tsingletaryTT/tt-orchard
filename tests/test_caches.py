# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/caches.py, `tt-orchard caches`: list the tensor caches and prune the ones no run uses.

The tests that matter most prove what must never happen: a cache an unfinished run uses being removed, the
shared kernel cache being removed, anything outside the cache root being removed, and removal without the
operator's yes.
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import caches
from orchard.bringup_config import Lab
from orchard.ledger import Ledger

DAY = 86400.0


def cache(root: Path, name: str, model: str, gb_bytes=1000, age_days=30, now=None):
    d = root / name
    d.mkdir(parents=True)
    (d / caches.MARKER).write_text(model)
    f = d / "weights.bin"
    f.write_bytes(b"x" * gb_bytes)
    t = (now or time.time()) - age_days * DAY
    for p in (f, d / caches.MARKER):
        os.utime(p, (t, t))
    return d


def run_using(runs: Path, name: str, cache_dir: Path, *, finished=False):
    run = runs / name
    (run / "stages" / "4" / "configs" / "1").mkdir(parents=True)
    (run / "stages" / "4" / "configs" / "1" / "swap_config.json").write_text(json.dumps({"tt_cache": str(cache_dir)}))
    with Ledger(run / "ledger.jsonl") as led:
        led.append("run_start", None, model="org/m", versions={}, inputs={})
        if finished:
            led.append("decision", None, decision="ready for operator review")
    return run


@pytest.fixture
def box(tmp_path):
    root = tmp_path / "cache"
    runs = tmp_path / "runs"
    runs.mkdir()
    cfg = SimpleNamespace(cache_root=root, runs_root=runs, lab=None)
    return root, runs, cfg


def main(cfg, *argv, ask=lambda q: False, run=None):
    said = []
    code = caches.main(list(argv), cfg=cfg, say=said.append, ask=ask, **({"run": run} if run else {}))
    return code, "\n".join(said)


def test_the_list_shows_each_cache_with_its_model_and_who_uses_it(box):
    root, runs, cfg = box
    used = cache(root, "m/1chip-a/tt_cache", "org/m@abc")
    cache(root, "m/2chip-b/tt_cache", "org/m@abc")
    (root / "kernels").mkdir()
    run_using(runs, "org--m", used)
    code, out = main(cfg)
    assert code == 0 and "org/m@abc" in out and "org--m" in out and caches.KERNEL_MODEL in out
    assert "3 cache(s)" in out


def test_prune_removes_only_old_unused_caches_after_a_yes(box):
    root, runs, cfg = box
    used = cache(root, "m/1chip-a/tt_cache", "org/m")
    old = cache(root, "m/2chip-b/tt_cache", "org/m")
    new = cache(root, "m/4chip-c/tt_cache", "org/m", age_days=1)
    done = cache(root, "m/1chip-old-run/tt_cache", "org/m")
    (root / "kernels").mkdir()
    os.utime(root / "kernels", (0, 0))
    run_using(runs, "org--m", used)
    run_using(runs, "org--m.finished", done, finished=True)          # a finished run no longer holds its caches
    code, out = main(cfg, "--prune", "--older-than", "7", "--yes")
    assert code == 0
    assert used.is_dir() and new.is_dir() and (root / "kernels").is_dir()
    assert not old.exists() and not done.exists()


def test_prune_asks_and_a_no_removes_nothing(box):
    root, runs, cfg = box
    old = cache(root, "m/2chip-b/tt_cache", "org/m")
    code, out = main(cfg, "--prune", "--older-than", "7", ask=lambda q: False)
    assert code == 0 and old.is_dir() and "nothing removed" in out


def test_prune_needs_an_age(box):
    root, runs, cfg = box
    assert main(cfg, "--prune")[0] == 2


def test_the_lab_list_comes_from_one_remote_command_and_prune_removes_there_by_exact_path(box):
    root, runs, cfg = box
    cfg.lab = Lab(host="node4", root=root.parent)
    now = time.time()
    listing = (f"./m/1chip-a/tt_cache\torg/m\t2048\t{now - 40 * DAY}\n"
               f"./kernels\t{caches.KERNEL_MODEL}\t4096\t{now - 90 * DAY}\n")
    calls = []

    def run(argv, timeout=60):
        calls.append(argv)
        return (0, listing, "") if "find" in argv[-1] else (0, "", "")
    code, out = main(cfg, "--lab", "--prune", "--older-than", "7", "--yes", run=run)
    assert code == 0 and "node4" in out
    removes = [c[-1] for c in calls if c[-1].startswith("rm -rf")]
    assert removes == [f"rm -rf -- {root}/m/1chip-a/tt_cache"]              # never the kernel cache


def test_nothing_outside_the_cache_root_is_ever_removed(box):
    root, runs, cfg = box
    cfg.lab = Lab(host="node4", root=root.parent)
    listing = f"../../etc\torg/m\t1\t0\n"
    calls = []

    def run(argv, timeout=60):
        calls.append(argv)
        return (0, listing, "") if "find" in argv[-1] else (0, "", "")
    main(cfg, "--lab", "--prune", "--older-than", "0", "--yes", run=run)
    assert not [c for c in calls if c[-1].startswith("rm -rf")]
