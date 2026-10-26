# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""prepare_swap.py's `"drafter": "off"` key (issue 15, suite control).

A model without mtp.* tensors is served with the speculative drafter off. Its control (the nearest model, which has the head)
would be served with the drafter on, and the two serving modes can give different replies, so the control would differ from
the run for a reason that has nothing to do with the weights. `"drafter": "off"` in swap_config.json makes the script clear the
drafter whatever the model's tensors say, with the same edit it makes for a model that lacks the head."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import drafter_lines, new_index, prep, run_prepare, with_drafter  # noqa: E402,F401

WITH_MTP = ["model.embed_tokens.weight", "lm_head.weight", "mtp.fc.weight", "model.layers.0.mlp.up_proj.weight"]


def set_cfg(prep, **kw):
    p = prep["stage"] / "swap_config.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), **kw}))


def test_the_key_clears_the_drafter_for_a_model_that_has_the_head(prep):
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    set_cfg(prep, drafter="off")
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert drafter_lines(prep) == ['export QWEN36_DRAFTER=""']
    assert "swap_config.json says drafter off" in r.stdout


def test_without_the_key_a_model_with_the_head_keeps_the_drafter(prep):
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    assert run_prepare(prep["stage"]).returncode == 0
    assert drafter_lines(prep)[0] == 'export QWEN36_DRAFTER="dflash2"'


def test_the_key_also_drops_on_device_sampling(prep):
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace(
        "--max_model_len", "--additional-config '{\"tt\": {\"sample_on_device_mode\": \"decode_only\"}}' --max_model_len", 1))
    with_drafter(prep)
    new_index(prep, WITH_MTP)
    set_cfg(prep, drafter="off")
    assert run_prepare(prep["stage"]).returncode == 0
    assert "sample_on_device_mode" not in (prep["stage"] / "run.sh").read_text()


def test_the_key_places_an_empty_drafter_line_when_the_bundle_has_none(prep):
    new_index(prep, WITH_MTP)
    set_cfg(prep, drafter="off")
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert drafter_lines(prep) == ['export QWEN36_DRAFTER=""']


@pytest.mark.parametrize("value", ["on", True, "", None, 1])
def test_any_other_value_is_refused(prep, value):
    set_cfg(prep, drafter=value)
    r = run_prepare(prep["stage"])
    assert r.returncode == 2 and "drafter" in r.stderr


def test_the_key_with_a_graft_is_refused(prep):
    set_cfg(prep, drafter="off", graft_mtp=True)
    r = run_prepare(prep["stage"])
    assert r.returncode == 2 and "a graft exists to keep the drafter on" in r.stderr
