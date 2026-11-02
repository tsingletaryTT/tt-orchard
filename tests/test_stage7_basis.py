# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 346: the boot check infers the tensor cache state from file times, and the stage 7 card says so."""
from test_stage7_cache_state import stage7_row


def test_the_stage_7_row_says_the_state_was_judged_by_file_age(tmp_path):
    cases = (("a", {"state": "empty", "files": 0, "basis": "file age"}, "empty tensor cache, judged by file age"),
             ("b", {"state": "warm", "files": 4, "basis": "file age"}, "4 files at start, judged by file age"))
    for sub, rec, text in cases:
        assert text in stage7_row(tmp_path / sub, {"tensor_cache_at_start": rec}).name


def test_a_record_without_a_basis_gets_no_suffix(tmp_path):
    assert "judged" not in stage7_row(tmp_path, {"tensor_cache_at_start": {"state": "empty", "files": 0}}).name
