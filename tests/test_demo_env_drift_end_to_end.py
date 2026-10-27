# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A host-tree release-tag change, end to end through the real code (review of PR 101).

The venv is built while the tree is at v0.80.0, so its ttnn and ttml dist-info carry 0.80.0 and the .pth points at the tree. Then the
tree's tag becomes v0.81.0. `demo_env.ensure(check_only=True)` (what `demo-env --check` and the preflight signal run) and
`preflight.run_preflight` with a signal that calls it must report version drift as a warning, never `host-tree: todo` and never a block.
A wrong .pth, a missing distribution and a mixed-version set must still block."""
import shutil
from pathlib import Path

import pytest

from orchard import demo_env as de
from orchard import preflight as pf
from test_demo_env import World, names
from test_preflight import cfg, signals
from test_research_preflight import report
from test_stock_preflight import ROW_PARAMS, ROW_README


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def move_tag(world, tag="v0.81.0"):
    world.run.rules.insert(0, (lambda a: "--exact-match" in a, (0, tag + "\n", ""), None))


def built_at_080(world):
    de.ensure(world.env(), check_only=False, say=lambda s: None)
    assert world.installed_dist("ttnn") == "0.80.0" and world.installed_dist("ttml") == "0.80.0"
    assert (world.site / de.PTH_NAME).read_text().splitlines()[0] == str(world.tree)
    world.run.calls.clear()


def preflight_row(world, c=None):
    """The preflight through run_preflight, with the signal running the real check_only path against the world's venv."""
    c = c or cfg(world.tmp)
    c.demo_python = world.venv / "bin" / "python"
    rep = report(tt_transformers=[ROW_PARAMS, ROW_README])
    params = world.tree / "models" / "tt_transformers" / "model_params"

    def demo_env(python, tree=None):
        env = world.env(tt_metal_dir=tree or world.tree)
        return de.ensure(env, check_only=True), de.fix_command(env)
    sig = signals(research=lambda *a: rep, stock_params=lambda name: params / name, demo_env=demo_env)
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=sig)
    return next(x for x in out if x.name == "demo-env"), out


def test_a_tag_change_is_a_warning_through_demo_env_check_and_the_preflight(world):
    built_at_080(world)
    move_tag(world)
    results = de.ensure(world.env(), check_only=True, say=lambda s: None)
    got = names(results)
    assert got["host-tree"].status == de.OK and "todo" not in [r.status for r in results if r.name == "host-tree"]
    assert got["ttnn-version"].status == de.FAIL and "0.81.0" in got["ttnn-version"].detail and "0.80.0" in got["ttnn-version"].detail
    assert got["record-tag"].status == de.FAIL
    row, out = preflight_row(world)
    assert row.status == pf.WARN and row.reason is None
    assert "0.81.0" in row.detail and "host-tree" not in row.detail
    assert not pf.blocked(out)


def test_the_cli_check_still_exits_2_on_drift_because_the_venv_is_for_another_tag(world):
    built_at_080(world)
    move_tag(world)
    out = []
    code = de.main(type("A", (), {"check": True, "dry_run": False, "venv_dir": None, "tt_metal_dir": None})(), env=world.env(),
                   out=type("S", (), {"write": lambda self, s: out.append(s), "flush": lambda self: None})())
    assert code == 2 and "0.81.0" in "".join(out)


def test_a_wrong_pth_still_blocks_when_the_tag_also_moved(world):
    built_at_080(world)
    move_tag(world)
    (world.site / de.PTH_NAME).write_text("/somewhere/else\n")
    row, _ = preflight_row(world)
    assert row.status == pf.BLOCK and "host-tree" in row.detail


def test_a_missing_distribution_still_blocks_when_the_tag_also_moved(world):
    built_at_080(world)
    move_tag(world)
    shutil.rmtree(world.site / "ttml-0.80.0.dist-info")
    row, _ = preflight_row(world)
    assert row.status == pf.BLOCK


def test_a_mixed_version_set_still_blocks_when_the_tag_also_moved(world):
    built_at_080(world)
    shutil.copytree(world.site / "ttml-0.80.0.dist-info", world.site / "ttml-0.79.0.dist-info")
    shutil.rmtree(world.site / "ttml-0.80.0.dist-info")
    move_tag(world)
    row, _ = preflight_row(world)
    assert row.status == pf.BLOCK


def test_a_half_written_distribution_at_the_new_tag_still_blocks(world):
    built_at_080(world)
    (world.site / "ttnn-0.81.0.dist-info").mkdir()
    move_tag(world)
    row, _ = preflight_row(world)
    assert row.status == pf.BLOCK


def test_a_missing_pth_still_blocks_when_the_tag_also_moved(world):
    built_at_080(world)
    (world.site / de.PTH_NAME).unlink()
    move_tag(world)
    row, _ = preflight_row(world)
    assert row.status == pf.BLOCK


def test_a_mixed_version_set_blocks_at_the_tree_tag_too(world):
    built_at_080(world)
    shutil.copytree(world.site / "ttml-0.80.0.dist-info", world.site / "ttml-0.79.0.dist-info")
    shutil.rmtree(world.site / "ttml-0.80.0.dist-info")
    row, _ = preflight_row(world)
    assert row.status == pf.BLOCK


def test_with_no_tag_change_the_same_venv_is_ok(world):
    built_at_080(world)
    row, out = preflight_row(world)
    assert row.status == pf.OK and not pf.blocked(out)


def test_a_tag_change_with_the_venv_python_missing_is_not_drift(world):
    built_at_080(world)
    move_tag(world)
    (world.venv / "bin" / "python").unlink()
    row, _ = preflight_row(world)
    assert row.status == pf.BLOCK
