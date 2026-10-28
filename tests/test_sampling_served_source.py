# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which swap check describes the served configuration (issue 107). After merging PR 109's acceptance-source tests: when the graft was chosen but only stage 2's
own swap check holds a sampling probe, the probe is read from stage 2's; a graft variant that has a probe is read from the variant."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sampling_choice_files import make_run, report  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402


def test_a_chosen_graft_whose_variant_swap_check_has_no_probe_falls_back_to_stage_2s_own(tmp_path):
    run, s2 = make_run(tmp_path, graft={"choice": "graft", "report": report(ignored=False, seconds=1.6, with_probe=False)},
                       stage2=report(drafter="off", ignored=False, seconds=6.0))
    assert sc.served_source(s2) == "evidence/swap-check.json"
    assert sc.served_side(s2)["verdict"] == "sampled-full-speed" and sc.probe_source(s2) == "evidence/swap-check.json"


def test_a_chosen_graft_whose_variant_has_a_probe_is_read_from_the_variant(tmp_path):
    run, s2 = make_run(tmp_path, graft={"choice": "graft", "report": report(ignored=False, seconds=1.6)})
    assert sc.served_source(s2) == "graft-trial/variant/evidence/swap-check.json"


def test_a_chosen_graft_with_no_variant_swap_check_at_all_falls_back_too(tmp_path):
    run, s2 = make_run(tmp_path, graft={"choice": "graft"})
    assert sc.served_source(s2) == "evidence/swap-check.json"


def test_a_variant_probe_that_did_not_run_counts_as_no_probe(tmp_path):
    rep = report(ignored=False, seconds=1.6)
    rep["sampling_probe"] = {"ran": False}
    run, s2 = make_run(tmp_path, graft={"choice": "graft", "report": rep})
    assert sc.served_source(s2) == "evidence/swap-check.json"
