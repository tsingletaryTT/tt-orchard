# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 321: the stage 7 card row said "fresh install, empty tensor cache" as fixed text. The boot check now records the state of the
cache its server used (`tensor_cache_at_start` in verify.json) and the card says what was recorded."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_fakes import make_run, make_source  # noqa: E402
from test_package_verify import boot, load_verifier  # noqa: E402,F401  (boot is a fixture)

from orchard import package  # noqa: E402

PROFILE = {"required": True, "chips": 2, "source": "bundle-x"}


def stage7_row(tmp_path, verify_extra):
    r = make_run(tmp_path, make_source(tmp_path / "models"))
    verify = {"top1_agreement": 1.0, "server_ready_s": 300.0, "evidence": ["stages/7/verify/evidence/verify.json"], **verify_extra}
    nums = package.card_numbers(SimpleNamespace(run_dir=r["run"]), PROFILE, verify)
    return next(n for n in nums if n.name.startswith("server ready after start, this package"))


def test_the_stage_7_row_says_empty_only_when_the_boot_check_recorded_zero_files(tmp_path):
    assert "empty tensor cache" in stage7_row(tmp_path, {"tensor_cache_at_start": {"state": "empty", "files": 0}}).name


def test_the_stage_7_row_with_a_warm_cache_gives_the_file_count(tmp_path):
    name = stage7_row(tmp_path, {"tensor_cache_at_start": {"state": "warm", "files": 7}}).name
    assert "warm tensor cache, 7 files at start" in name and "empty tensor cache" not in name


def test_the_stage_7_row_without_a_record_says_not_recorded(tmp_path):
    name = stage7_row(tmp_path, {}).name
    assert "empty tensor cache" not in name and "not recorded" in name


def test_the_boot_check_counts_only_files_older_than_the_launch(tmp_path):
    vb = load_verifier()
    cache = tmp_path / "c"
    (cache / "sub").mkdir(parents=True)
    old, new = cache / "sub" / "old.bin", cache / "new.bin"
    old.write_text("x")
    new.write_text("y")
    (cache / vb.sac.MARKER).write_text("m")
    os.utime(old, (100, 100))
    assert vb.cache_at_start(cache, launched_at=1000.0) == {"state": "warm", "files": 1}
    assert vb.cache_at_start(cache / "missing", launched_at=1000.0) == {"state": "empty", "files": 0}
    os.utime(old, (2000, 2000))
    assert vb.cache_at_start(cache, launched_at=1000.0) == {"state": "empty", "files": 0}


def test_verify_json_records_an_empty_cache_for_a_fresh_install(boot):
    assert boot["check"]("perfect").returncode == 0
    v = json.loads((boot["verify"] / "evidence" / "verify.json").read_text())
    assert v["tensor_cache_at_start"] == {"state": "empty", "files": 0}


def test_verify_json_records_a_warm_cache_the_bundle_already_had(boot):
    cache = boot["verify"] / "bundle" / ".tt_cache"       # where the bundle's run.sh points TT_CACHE_PATH
    cache.mkdir()
    for i in range(3):
        f = cache / f"t{i}.bin"
        f.write_text("x")
        os.utime(f, (100, 100))
    assert boot["check"]("perfect").returncode == 0
    v = json.loads((boot["verify"] / "evidence" / "verify.json").read_text())
    assert v["tensor_cache_at_start"] == {"state": "warm", "files": 3}


def test_verify_json_records_nothing_when_the_servers_environment_names_no_cache(boot):
    run_sh = boot["verify"] / "bundle" / "run.sh"
    run_sh.write_text("".join(ln for ln in run_sh.read_text().splitlines(True) if "TT_CACHE_PATH" not in ln))
    assert boot["check"]("perfect").returncode == 0
    v = json.loads((boot["verify"] / "evidence" / "verify.json").read_text())
    assert "tensor_cache_at_start" not in v
