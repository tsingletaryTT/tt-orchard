# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 319: the card named the tensor cache state from text. 'empty tensor cache' sat on a stage 2 start time that was taken with a warm
cache. The hardware test now records the cache's state when it starts (`tensor_cache_at_start` in swap-check.json) and the card says that."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from package_fakes import make_run, make_source  # noqa: E402
from test_weights_swap_templates import load_swap_template  # noqa: E402

from orchard import package  # noqa: E402

PROFILE = {"required": True, "chips": 2, "source": "bundle-x"}


def stage2_row(tmp_path, record):
    """The card's stage 2 start-time row, for a run whose swap-check.json holds `record` (absent when None)."""
    r = make_run(tmp_path, make_source(tmp_path / "models"))
    path = r["run"] / "stages/2/evidence/swap-check.json"
    data = json.loads(path.read_text())
    if record is not None:
        data["tensor_cache_at_start"] = record
    path.write_text(json.dumps(data))
    nums = package.card_numbers(SimpleNamespace(run_dir=r["run"]), PROFILE, None)
    return next(n for n in nums if n.name.startswith("server ready after start, stage 2"))


def test_a_warm_cache_start_is_not_called_empty(tmp_path):
    row = stage2_row(tmp_path, {"state": "warm", "files": 412})
    assert "warm tensor cache" in row.name and "412 files" in row.name
    assert "empty tensor cache" not in row.name


def test_an_empty_cache_start_says_empty(tmp_path):
    assert "(empty tensor cache)" in stage2_row(tmp_path, {"state": "empty", "files": 0}).name


def test_a_run_with_no_cache_record_does_not_claim_a_state(tmp_path):
    row = stage2_row(tmp_path, None)
    assert "empty tensor cache" not in row.name and "warm tensor cache" not in row.name
    assert "not recorded" in row.name


def test_an_inconsistent_record_does_not_claim_empty(tmp_path):
    """state empty with files counted, or an unknown state, is not a measurement of an empty cache."""
    for rec in ({"state": "empty", "files": 5}, {"state": "cold", "files": 0}, {"state": "warm", "files": 0}, "empty"):
        name = stage2_row(tmp_path / str(abs(hash(json.dumps(rec)))), rec).name
        assert "empty tensor cache" not in name and "not recorded" in name


# The layer that writes the record: the hardware test template.
def test_cache_state_counts_files_and_ignores_the_marker(tmp_path):
    sac = load_swap_template()
    cache = tmp_path / "c"
    assert sac.cache_state(cache) == {"state": "empty", "files": 0}
    cache.mkdir()
    (cache / sac.MARKER).write_text("m")
    assert sac.cache_state(cache) == {"state": "empty", "files": 0}
    (cache / "sub").mkdir()
    (cache / "sub" / "a.tensorbin").write_text("x")
    (cache / "b.tensorbin").write_text("x")
    assert sac.cache_state(cache) == {"state": "warm", "files": 2}


def test_write_report_records_the_cache_state_it_was_given(tmp_path):
    sac = load_swap_template()
    tok = SimpleNamespace(encode=lambda text, add_special_tokens=False: SimpleNamespace(ids=[1, 2]))
    m = {"matches": 30, "free_run_text": "The capital of France is Paris, a city of art.", "server_ready_s": 138.1,
         "forced": [], "mismatches": []}
    ev = tmp_path / "evidence"
    ev.mkdir()
    state = {"state": "warm", "files": 7}
    sac.write_report({"new_model_id": "a/b", "port": 8100}, tmp_path, tmp_path / "c", {"MODEL_WEIGHTS_DIR": "d", "HF_MODEL": "d"}, m, tok,
                     ([1], [2] * 32, "t"), ev / "server.log", cache_at_start=state)
    assert json.loads((ev / "swap-check.json").read_text())["tensor_cache_at_start"] == state
