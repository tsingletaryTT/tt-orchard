# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The local draft stage skills, and the skill names the stage table uses."""
import re
from pathlib import Path

import pytest

from orchard.stages import STAGES, resolve_skill, spec_for

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
LOCAL = ("delta-triage", "reference-gate", "weights-swap-check", "weights-swap-configs", "serving-check")
# Spec section 11: existing skills the stages use, referenced by name only.
SPEC_EXISTING = {"model-bringup", "functional-decoder", "full-model", "multichip", "mesh-shrink",
                 "vllm-integration", "qualitative-check", "benchmark-model", "tt-device-usage",
                 "stage-review", "tti-release"}
GATE_FILES = {"delta-triage": "delta.json", "reference-gate": "reference.json",
              "weights-swap-check": "result.json", "weights-swap-configs": "result.json",
              "serving-check": "result.json"}


@pytest.mark.parametrize("name", LOCAL)
def test_each_local_skill_is_a_marked_draft_with_its_gate_file(name):
    path = resolve_skill(name, [SKILLS])
    assert path == SKILLS / f"{name}.md"
    text = path.read_text()
    front = text.split("---")[1]
    assert f"\nname: {name}\n" in front and "\nstatus: draft." in front
    assert GATE_FILES[name] in text


@pytest.mark.parametrize("name", LOCAL)
def test_no_stage_skill_names_a_lease_tool(name):
    # Spec section 11: stage skills say "under whatever lease the machine provides".
    assert "gozer" not in (SKILLS / f"{name}.md").read_text().lower()


def test_every_skill_the_table_names_is_local_or_named_in_the_spec():
    # Every path's table: stage 2 on the weights-only path names a different skill.
    specs = {spec_for(s.number, path) for s in STAGES for path in ("weights-only", "full-port", None)}
    for s in sorted(specs, key=lambda s: (s.number, s.skill)):
        if s.skip or s.harness:
            continue
        assert s.skill in LOCAL or s.skill in SPEC_EXISTING, s.skill
        assert set(s.refs) <= SPEC_EXISTING, s.refs


def test_the_swap_skill_copies_the_templates_that_exist_in_this_repo():
    # The skill names the checkout through {{ORCHARD_DIR}}, which the context fills in. The same
    # relative path must exist here, so the skill and the templates merge together.
    text = (SKILLS / "weights-swap-check.md").read_text()
    main = "{{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/"
    for name in ("prepare_swap.py", "serve_and_compare.py"):
        assert main + name in text
        assert (SKILLS / "weights-swap-templates" / name).is_file()
    assert '"command": "python3 stages/2/serve_and_compare.py", "deadline_s": 3600' in text
    flat = " ".join(text.split())
    assert "If `stages/2/swap_config.json` exists, read it once and use it as it is" in flat
    assert "does not exist, write it FIRST" in flat
    assert "`serves` false" in text and "Do not investigate firmware or cache directories" in text


def test_the_swap_skill_finish_section_records_a_failed_test_and_stops():
    text = " ".join((SKILLS / "weights-swap-check.md").read_text().split())
    finish = text.split("## Finish phase", 1)[1].split("## Do not", 1)[0]
    assert "`serves` false" in finish and "`failure` field" in finish
    assert "`coherent`, `n_tokens`, `top1_agreement` and `server_ready_s` null" in finish
    assert "Then stop." in finish and "Do not investigate" in finish


def test_the_weights_only_stage_2_skill_is_the_local_flat_file():
    assert spec_for(2, "weights-only").skill == "weights-swap-check"
    assert resolve_skill("weights-swap-check", [SKILLS]) == SKILLS / "weights-swap-check.md"


def test_the_swap_skill_quotes_the_gate_bar_from_defaults():
    from orchard.defaults import SWAP_TOP1_MIN
    text = (SKILLS / "weights-swap-check.md").read_text()
    assert f"`top1_agreement` of at\nleast {SWAP_TOP1_MIN}" in text or \
        f"`top1_agreement` of at least {SWAP_TOP1_MIN}" in text
    assert "0.6" not in text


def test_the_swap_skill_explains_the_weights_directory_fact():
    text = " ".join((SKILLS / "weights-swap-check.md").read_text().split())   # line breaks as spaces
    assert "Four facts decide whether a swap works." in text
    assert "4. The server must be told where the new weights are." in text
    for needed in ("MODEL_WEIGHTS_DIR", "HF_MODEL", "30 of 32", "25 of 32", "The template sets"):
        assert needed in text, needed


def test_the_configs_skill_copies_the_three_templates_that_exist_in_this_repo():
    text = (SKILLS / "weights-swap-configs.md").read_text()
    main = "{{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/"
    for name in ("prepare_swap.py", "serve_and_compare.py", "serve_and_compare_container.py"):
        assert main + name in text
        assert (SKILLS / "weights-swap-templates" / name).is_file()


def test_the_configs_skill_writes_the_list_the_supervisor_reads():
    import json
    import re

    from orchard.hwtests import SCRIPTS
    from orchard.stages import WEIGHTS_ONLY_STAGE_4
    text = (SKILLS / "weights-swap-configs.md").read_text()
    block = re.search(r'(\{"tests": \[.*?\]\})', text, re.S).group(1)
    tests = json.loads(" ".join(block.split()))["tests"]
    assert sorted(t["chips"] for t in tests) == [1, 2, 4]
    assert all(t["script"] in SCRIPTS for t in tests)
    assert sum(t["deadline_s"] for t in tests) <= WEIGHTS_ONLY_STAGE_4.budget_s


def test_the_configs_skill_quotes_the_gate_bar_and_the_cache_rule():
    from orchard.defaults import SWAP_TOP1_MIN
    text = " ".join((SKILLS / "weights-swap-configs.md").read_text().split())
    assert f"`top1_agreement` of at least {SWAP_TOP1_MIN}" in text
    assert "Every configuration gets its own new `tt_cache`" in text
    assert "A configuration whose test never ran cannot pass." in text


# ---- machine paths ------------------------------------------------------------------------------
# The agent reads every file under orchard/skills (the skills and the templates it copies) and
# under orchard/package_templates (copied into the bundle). None of them may name a path on the
# development machine. A path on the machine is written as a placeholder (orchard/paths.py).

AGENT_FACING = [p for root in (SKILLS, SKILLS.parent / "package_templates")
                for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
MACHINE_PATH = re.compile(r"/home/|/mnt/|/Users/|/root/")


def test_the_agent_facing_file_list_is_not_empty():
    names = {p.name for p in AGENT_FACING}
    assert {"weights-swap-check.md", "prepare_swap.py", "verify_bundle.py"} <= names


@pytest.mark.parametrize("path", AGENT_FACING, ids=lambda p: str(p.relative_to(SKILLS.parent)))
def test_no_agent_facing_file_names_a_machine_path(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [f"{i}: {line.strip()}" for i, line in enumerate(lines, 1) if MACHINE_PATH.search(line)]
    assert not hits, f"{path.name} holds absolute machine paths; use a placeholder: {hits}"


@pytest.mark.parametrize("name", LOCAL)
def test_every_placeholder_in_a_local_skill_renders(name):
    from orchard.paths import RunPaths, render
    p = RunPaths(orchard_dir="/o", hf_home="/hf", operator_home="/op", tt_model_root="/ttm",
                 cache_root="/cache")
    out = render((SKILLS / f"{name}.md").read_text(), p, source=name)
    assert "{{" not in out


def test_the_swap_skills_put_each_tensor_cache_under_the_cache_root():
    check = (SKILLS / "weights-swap-check.md").read_text()
    configs = (SKILLS / "weights-swap-configs.md").read_text()
    assert "{{CACHE_ROOT}}/<slug>/tt_cache" in check
    assert "{{CACHE_ROOT}}/<slug>/<N>chip-<package name>/tt_cache" in configs
    assert "{{CACHE_ROOT}}/<slug>/4chip-qwen3.8-27b-p300x2/tt_cache" in configs
    for text in (check, configs):
        assert '"hf_home": "{{HF_HOME}}"' in text
        assert "{{TT_MODEL_ROOT}}/episod/qwen3.8-27b-dflash2-p300" in text
    assert '"operator_home": "{{OPERATOR_HOME}}"' in configs


@pytest.mark.parametrize("name", ("weights-swap-check", "weights-swap-configs"))
def test_the_swap_skills_copy_stage_0s_model_id_verbatim(name):
    # Run 3's stage 7 stopped because stage 2's label left off stage 0's @revision. Stage 7 now
    # reads the revision from the served weights, and the label must still agree with stage 0.
    text = " ".join((SKILLS / f"{name}.md").read_text().split())
    assert '"new_model_id": "<the model value from stages/0/delta.json, for example ' \
           'Altworld/Hemmingway-1@<revision>>"' in text
    assert "Copy the `model` value from `stages/0/delta.json` verbatim, including the `@revision`" in text
    assert "Stage 7 compares this label with stage 0's model and the revision of the weights" in text


# ---- stage 0: the delta-triage template ---------------------------------------------------------

def _config_block(text: str, first_key: str) -> dict:
    """The JSON config block a skill shows, starting at `{"<first_key>"`."""
    import json
    block = re.search(r'(\{"' + first_key + r'".*?\})', text, re.S).group(1)
    return json.loads(" ".join(block.split()))


def test_the_delta_triage_skill_runs_the_template_that_exists_in_this_repo():
    import importlib.util
    text = (SKILLS / "delta-triage.md").read_text()
    flat = " ".join(text.split())
    script = SKILLS / "delta-triage-templates" / "delta_triage.py"
    assert script.is_file()
    assert "cp {{ORCHARD_DIR}}/orchard/skills/delta-triage-templates/delta_triage.py stages/0/" in flat
    assert "python3 stages/0/delta_triage.py" in flat
    # The config block names exactly the keys the script reads.
    spec = importlib.util.spec_from_file_location("delta_triage_for_skill_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(_config_block(text, "run_dir")) == set(mod.CONFIG_KEYS)
    assert "{{HF_HOME}}/hub/models--<org>--<name>/snapshots/<sha>/" in flat


def test_the_delta_triage_skill_has_the_write_first_rules():
    flat = " ".join((SKILLS / "delta-triage.md").read_text().split())
    assert "Write `triage_config.json` FIRST" in flat
    assert "Never read the run's ledger" in flat and "transcripts" in flat
    assert "Do not investigate anything outside the paths this skill lists" in flat
    assert "Edit a `finding` only where a measured fact needs explaining" in flat
    assert "Do not change `path`" in flat


# ---- stage 1: the reference-gate template -------------------------------------------------------

def test_the_reference_gate_skill_runs_the_template_that_exists_in_this_repo():
    import importlib.util
    text = (SKILLS / "reference-gate.md").read_text()
    flat = " ".join(text.split())
    script = SKILLS / "reference-gate-templates" / "reference_gate.py"
    assert script.is_file()
    assert "cp {{ORCHARD_DIR}}/orchard/skills/reference-gate-templates/reference_gate.py stages/1/" in flat
    assert "<python> stages/1/reference_gate.py" in flat
    spec = importlib.util.spec_from_file_location("reference_gate_for_skill_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(_config_block(text, "run_dir")) == set(mod.CONFIG_KEYS)
    assert mod.PROMPT in flat


def test_the_reference_gate_skill_has_the_write_first_rules_and_the_card_caveat():
    flat = " ".join((SKILLS / "reference-gate.md").read_text().split())
    assert "Write `reference_config.json` FIRST" in flat
    assert "Never read the run's ledger" in flat and "transcripts" in flat
    assert "Do not investigate anything outside the paths this skill lists" in flat
    assert "form check only" in flat and "Keep that sentence." in flat
    assert "Do not change a check's `pass` from false to true." in flat


def test_the_reference_gate_skill_tells_the_agent_to_use_the_reference_python_input_and_never_to_install():
    text = " ".join((Path(__file__).resolve().parent.parent / "orchard" / "skills" / "reference-gate.md")
                    .read_text().split())
    assert "input named `reference_python`" in text and "use that path exactly" in text
    assert "Never install or upgrade a package" in text


def test_both_swap_skills_start_from_the_drafted_config_and_expect_the_architecture_difference():
    # orchard/swap_draft.py writes the config; on the lab run the agent kept investigating the
    # ConditionalGeneration / CausalLM difference instead of using it.
    for name in ("weights-swap-check.md", "weights-sidecar-check.md"):
        flat = " ".join((SKILLS / name).read_text().split())
        assert "The supervisor has usually written `swap_config.json` already" in flat, name
        assert "That difference is expected on this path" in flat and "Do not investigate it." in flat, name


def test_the_prepare_steps_start_from_the_drafted_config_and_fact_finding_is_only_a_fallback():
    # With the draft only mentioned in a note, Coder-Next followed step 1 ("Find four facts") and
    # never opened the config the supervisor had written.
    for name in ("weights-swap-check.md", "weights-sidecar-check.md"):
        text = (SKILLS / name).read_text()
        steps = text.split("## Prepare phase: the steps", 1)[1].split("\n## ", 1)[0]
        assert steps.lstrip().startswith("1. Read `stages/2/swap_config.json`"), name
        assert "tt-model list`, then read" not in steps and "Find four facts" not in steps, name
        fallback = text.split("## When there is no swap_config.json", 1)[1].split("\n## ", 1)[0]
        assert "Find four facts" in fallback and "weights-swap-templates/prepare_swap.py" in fallback, name



def test_the_stage_4_steps_start_from_the_drafted_configs_and_the_table_is_only_a_fallback():
    text = (SKILLS / "weights-swap-configs.md").read_text()
    steps = text.split("## Prepare phase: the steps", 1)[1].split("\n## ", 1)[0]
    assert steps.lstrip().startswith("1. Read `stages/4/hw_tests.json`"), steps[:80]
    assert "tt-model list`, then" not in steps and "mkdir -p stages/4/configs/N" not in steps
    fallback = text.split("## When a configuration has no drafted config", 1)[1].split("\n## ", 1)[0]
    assert "mkdir -p stages/4/configs/N" in fallback and "four_chip_package" in fallback
    assert "changh95/qwen3.8-27b-p300x2" in fallback


def test_the_stage_4_skill_keeps_to_the_drafted_counts():
    # Lab run 2: the skill listed "2 chips, 4 chips, then 1 chip" and called a count with no drafted
    # config "not drafted", so the agent built a 4-chip config by hand on a 2-chip lab.
    flat = " ".join((SKILLS / "weights-swap-configs.md").read_text().split())
    assert "When `hw_tests.json` exists, its counts are the whole list. Do not add a count to it" in flat
    assert "2 chips, 4 chips, then 1 chip" not in flat
    assert "Only when `hw_tests.json` does not exist" in flat
