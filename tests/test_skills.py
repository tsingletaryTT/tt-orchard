"""The local draft stage skills, and the skill names the stage table uses."""
import re
from pathlib import Path

import pytest

from orchard.stages import STAGES, resolve_skill, spec_for

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
LOCAL = ("delta-triage", "reference-gate", "weights-swap-check", "weights-swap-configs", "serving-check",
         "operator-bundle")
# Spec section 11: existing skills the stages use, referenced by name only.
SPEC_EXISTING = {"model-bringup", "functional-decoder", "full-model", "multichip", "mesh-shrink",
                 "vllm-integration", "qualitative-check", "benchmark-model", "tt-device-usage",
                 "stage-review", "tti-release"}
GATE_FILES = {"delta-triage": "delta.json", "reference-gate": "reference.json",
              "weights-swap-check": "result.json", "weights-swap-configs": "result.json",
              "serving-check": "result.json",
              "operator-bundle": "PUBLISH_COMMANDS.txt"}


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
        if s.skip:
            continue
        assert s.skill in LOCAL or s.skill in SPEC_EXISTING, s.skill
        assert set(s.refs) <= SPEC_EXISTING, s.refs


def test_the_bundle_skill_keeps_publishing_with_the_operator():
    text = (SKILLS / "operator-bundle.md").read_text()
    assert "You never run them." in text and "ready for operator review" in text


def test_the_swap_skill_copies_the_templates_that_exist_in_this_repo():
    # The skill names the checkout through {{ORCHARD_DIR}}, which the context fills in. The same
    # relative path must exist here, so the skill and the templates merge together.
    text = (SKILLS / "weights-swap-check.md").read_text()
    main = "{{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/"
    for name in ("prepare_swap.py", "serve_and_compare.py"):
        assert main + name in text
        assert (SKILLS / "weights-swap-templates" / name).is_file()
    assert '"command": "python3 stages/2/serve_and_compare.py", "deadline_s": 3600' in text
    assert "swap_config.json` FIRST" in text
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


def test_the_bundle_skill_carries_stage_7s_publish_commands_word_for_word():
    text = (SKILLS / "operator-bundle.md").read_text()
    assert "bundle/package/PUBLISH_COMMANDS.txt" in text and "word for word" in text
    assert "non-commercial" in text
