# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The weights-sidecar-check skill and the templates it tells the agent to copy must agree.

The agent writes parity_config.json from the skill's example, so the example has to hold every key the
scripts require and no key they never read. The skill's `cp` lines and its hw_test command must name files
that exist. A skill that drifts from its scripts fails a stage on a real run and not before, so this reads
both."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / "orchard" / "skills"
SKILL = (SKILLS / "weights-sidecar-check.md").read_text()
PARITY = SKILLS / "sidecar-parity-templates"
SWAP = SKILLS / "weights-swap-templates"


def example_after(heading_text: str) -> dict:
    """The first JSON object in the skill after a line that contains `heading_text`."""
    start = SKILL.index(heading_text)
    m = re.search(r"^[ ]{4,}(\{.*?\})\s*$", SKILL[start:], re.S | re.M)
    block = "\n".join(line.strip() for line in m.group(1).splitlines())
    return json.loads(block)


def module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_parity_config_example_has_every_key_the_scripts_require():
    keys = set(example_after("Write `parity_config.json`"))
    hidden = set(module(PARITY / "hidden_parity.py").REQUIRED_KEYS)
    prepare = {"bundle_dir", "model_snapshot", "tt_cache"}                  # prepare_parity.load_config
    assert (hidden | prepare) <= keys, sorted((hidden | prepare) - keys)


def test_the_parity_config_example_has_no_key_the_scripts_never_read():
    keys = set(example_after("Write `parity_config.json`"))
    sources = "".join(p.read_text() for p in PARITY.glob("*.py"))
    unread = sorted(k for k in keys if f'"{k}"' not in sources and f"'{k}'" not in sources)
    assert not unread, f"the skill tells the agent to write keys no script reads: {unread}"


def test_the_swap_config_example_still_matches_prepare_swap():
    keys = set(example_after("Write `swap_config.json`"))
    src = "".join(p.read_text() for p in SWAP.glob("*.py"))
    assert all(f'"{k}"' in src for k in keys), sorted(k for k in keys if f'"{k}"' not in src)


def test_every_file_the_skill_copies_exists():
    for src in re.findall(r"\{\{ORCHARD_DIR\}\}/(orchard/skills/[\w./-]+\.py)", SKILL):
        assert (REPO / src).is_file(), src


def test_the_hardware_test_command_names_a_template_the_skill_copies():
    cmd = re.search(r'"command": "python3 (stages/2/[\w.]+)"', SKILL).group(1)
    name = Path(cmd).name
    assert (PARITY / name).is_file()
    assert f"sidecar-parity-templates/{name}" in SKILL            # copied into the stage directory


def test_the_deadline_is_inside_the_stage_2_budget_and_longer_than_the_cold_compile():
    from orchard.defaults import STAGE_BUDGET_S
    deadline = int(re.search(r'"deadline_s": (\d+)', SKILL).group(1))
    assert 3300 < deadline < STAGE_BUDGET_S[2]


def test_the_skill_never_asks_the_agent_to_run_the_third_party_code():
    assert "Do not open, read or run the sidecar's code file" in SKILL


def test_the_skill_names_only_placeholders_the_context_fills_in():
    from orchard.paths import PLACEHOLDERS
    assert set(re.findall(r"\{\{(.*?)\}\}", SKILL)) <= set(PLACEHOLDERS)
