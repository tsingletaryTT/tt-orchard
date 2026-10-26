# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stock-runtime class in the stage table (issue 61, docs/superpowers/specs/2026-10-26-full-port-pilot-design.md).

A model that tt-transformers already supports takes its own path: stage 0 is supervisor code (orchard/stock_runtime.py),
stage 2 serves the model through tt-transformers on one board, and the stages that a weights swap or a port would need
are skipped. The class is chosen without any flag: the run's inputs name the tt-transformers parameters."""
import json

import pytest

from orchard import classes, defaults, stages
from orchard.stages import STAGES, PATHS, spec_for


def test_the_class_is_in_the_table_and_takes_its_own_path():
    assert "stock-runtime" in classes.CLASSES
    assert classes.path_of("stock-runtime") == "stock-runtime"
    assert "stock-runtime" in PATHS
    assert "stock-runtime" not in classes.SIDECAR_CLASSES


def test_a_path_alone_never_implies_the_stock_runtime_class():
    # only stage 0's own answer can say it; an older ledger with a path must not be read as one
    assert classes.class_of_path("stock-runtime") is None


def test_stage_zero_is_supervisor_code_when_the_run_has_stock_parameters():
    s = spec_for(0, None, None, None, stock=True)
    assert s.harness is True and s.gate is stages.gate_delta_stock and s.gate_file == "delta.json"
    assert spec_for(0, None, None, None) is STAGES[0]                 # every other run keeps the agent stage


def test_stage_two_of_the_stock_path_has_its_own_skill_and_gate():
    s = spec_for(2, "stock-runtime")
    assert s.skill == "stock-runtime-check" and s.gate is stages.gate_stock_runtime
    assert s.boards == 1 and s.marker == "test-result.json" and s.skip is None


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_the_other_stages_of_the_stock_path_are_skipped_with_a_reason(n):
    s = spec_for(n, "stock-runtime")
    assert s.skip and len(s.skip) > 30


def test_stage_four_says_it_is_not_built_yet():
    assert "not built" in spec_for(4, "stock-runtime").skip


def test_the_weights_only_and_full_port_tables_are_unchanged():
    assert spec_for(2, "weights-only").skill == "weights-swap-check"
    assert spec_for(2, "full-port") is STAGES[2]
    assert spec_for(2, None) is STAGES[2]


def test_the_gate_bars_are_choices_with_the_approved_values():
    assert defaults.STOCK_TOP1_MIN == 0.85 and defaults.STOCK_TOP5_MIN == 0.95
    assert defaults.STOCK_SMALL_TOP1_MIN == 0.75 and defaults.STOCK_SMALL_TOP5_MIN == 0.95
    assert defaults.STOCK_SMALL_PARAMS_MAX_B == 3


def test_the_small_model_bar_applies_at_exactly_the_size_limit(tmp_path):
    import test_stock_runtime_gates as g
    small = {**g.DELTA, "stock_runtime": {**g.STOCK, "params_b": float(defaults.STOCK_SMALL_PARAMS_MAX_B)}}
    ok = stages.gate_stock_runtime(*g.stage2(tmp_path, g.tf(top1=0.78), delta=small))
    assert ok.ok, ok.reasons


def test_every_run_the_ledger_started_with_stock_parameters_reads_them_back():
    entries = [{"event": "run_start", "stage": None, "data": {"inputs": {"model": "/m", "stock_params": "/p/Llama-3.1-8B-Instruct"}}}]
    assert stages.stock_params(entries) == "/p/Llama-3.1-8B-Instruct"
    assert stages.stock_params([{"event": "run_start", "stage": None, "data": {"inputs": {"model": "/m"}}}]) is None
    assert stages.stock_params([]) is None
    bad = [{"event": "run_start", "stage": None, "data": {"inputs": {"stock_params": 5}}}]
    assert stages.stock_params(bad) is None
