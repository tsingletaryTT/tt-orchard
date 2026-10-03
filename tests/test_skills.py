"""The local draft stage skills, and the skill names the stage table uses."""
from pathlib import Path

import pytest

from orchard.stages import STAGES, resolve_skill, spec_for

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
LOCAL = ("delta-triage", "reference-gate", "weights-swap-check", "serving-check", "operator-bundle")
# Spec section 11: existing skills the stages use, referenced by name only.
SPEC_EXISTING = {"model-bringup", "functional-decoder", "full-model", "multichip", "mesh-shrink",
                 "vllm-integration", "qualitative-check", "benchmark-model", "tt-device-usage",
                 "stage-review", "tti-release"}
GATE_FILES = {"delta-triage": "delta.json", "reference-gate": "reference.json",
              "weights-swap-check": "result.json", "serving-check": "result.json",
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
    # The skill names the main checkout's absolute path. The same relative path must exist here,
    # so the skill and the templates merge together.
    text = (SKILLS / "weights-swap-check.md").read_text()
    main = "/home/ttuser/code/tt-orchard/orchard/skills/weights-swap-templates/"
    for name in ("prepare_swap.py", "serve_and_compare.py"):
        assert main + name in text
        assert (SKILLS / "weights-swap-templates" / name).is_file()
    assert '"command": "python3 stages/2/serve_and_compare.py", "deadline_s": 2400' in text
    assert "swap_config.json` FIRST" in text
    assert "`serves` false" in text and "Do not investigate firmware or cache directories" in text


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
