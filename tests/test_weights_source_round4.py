# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 442, Copilot round 3 review (issue 73): the picker must not offer a bundle for ANOTHER base that the swap could not take.

`refused_names` came from the packages that serve the base asked for. A bundle for `Qwen/Base-7B-Instruct` whose run.sh has no `--model`
was not in that set, so it was listed as an alternative and refused by the next preflight after the operator chose it. Every installed
package is now judged against its own `weights_repo`. Nothing here needs a chip, a container or the network."""
from test_weights_source_round2 import BAD, BASE, GOOD, HARD, RECORDS, check, v6
from orchard import weights_source as ws

ALT = "Qwen/Base-7B-Instruct"
ALT_GOOD = GOOD.replace(BASE, ALT)
ALT_BAD = BAD.replace(BASE, ALT)


def names(c):
    return [x["name"] for x in c.data["candidates"]]


def test_a_bundle_for_another_base_that_cannot_take_swapped_weights_is_not_an_installed_candidate(tmp_path):
    good, bad = v6(tmp_path, "org/alt-good", ALT_GOOD, repo=ALT), v6(tmp_path, "org/alt-bad", ALT_BAD, repo=ALT)
    c = check([good, bad])
    assert c.status == "block" and names(c) == ["org/alt-good"] and "org/alt-bad" not in c.detail


def test_a_bundle_for_another_base_that_cannot_take_swapped_weights_is_not_a_ranked_choice(tmp_path):
    bad = v6(tmp_path, "org/alt-bad", ALT_BAD, repo=ALT)
    report = {"bundle_choices": [{"name": "org/alt-bad", "class": "weights-only", "confidence": "stated"},
                                 {"name": "org/other", "class": "weights-only", "confidence": "stated"}]}
    c = check([bad], report=report)
    assert names(c) == ["org/other"] and "org/alt-bad" not in c.detail


def test_a_searched_name_that_is_an_installed_refused_alternative_is_not_listed(tmp_path):
    bad = v6(tmp_path, "org/alt-bad", ALT_BAD, repo=ALT)
    c = check([bad], search=lambda family: [{"name": "org/alt-bad", "installed": True}, {"name": "org/fresh", "installed": False}])
    assert names(c) == ["org/fresh"]


def test_unoffered_names_judges_each_package_against_its_own_repository(tmp_path):
    good, bad = v6(tmp_path, "org/alt-good", ALT_GOOD, repo=ALT), v6(tmp_path, "org/alt-bad", ALT_BAD, repo=ALT)
    assert ws.unoffered_names([good, bad], RECORDS) == {"org/alt-bad", HARD} - {"org/alt-good"}
