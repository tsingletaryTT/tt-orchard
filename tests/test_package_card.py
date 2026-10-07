"""The package card and its license (plan 5, stage 7)."""
import json

import pytest

from orchard.package_card import (CardFacts, Number, card_problems, non_commercial, read_license,
                                  render_card)

REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf"


def run_with_evidence(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    (run / "stages" / "2" / "result.json").write_text(json.dumps({"top1_agreement": 0.94,
                                                                  "server_ready_s": 280.5}))
    (run / "stages" / "2" / "evidence").mkdir()
    (run / "stages" / "2" / "evidence" / "swap-check.json").write_text("{}")
    return run


def facts(**over):
    base = dict(
        name="hemmingway-1-p300", namespace="episod", model_id="Altworld/Hemmingway-1",
        revision=REV, nearest_model="Qwen/Qwen3.8-27B", source_name="qwen3.8-27b-dflash2-p300",
        license_id="cc-by-nc-4.0", chips=2, mesh="P150x2", arch="blackhole", max_model_len=262144,
        max_num_seqs=4, drafter="incoai/Qwen3.8-27B-DFlash2", verified=True,
        numbers=(Number("top1 agreement with the CPU reference (2 chips, stage 2)", 0.94, "fraction",
                        "measured", ("stages/2/result.json", "stages/2/evidence/swap-check.json")),
                 Number("time to first token", None, "ms", "TODO", ())),
        not_measured=("the drafter's acceptance rate on this model",))
    base.update(over)
    return CardFacts(**base)


def test_a_rendered_card_passes_its_own_check(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts())
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=run) == []
    assert card.startswith("---\nlicense: cc-by-nc-4.0\nbase_model: Altworld/Hemmingway-1\n")
    for tag in ("tt-model-cache", "blackhole", "vllm", "thin", "p150x2"):
        assert f"\n- {tag}\n" in card
    assert "tt-model serve episod/hemmingway-1-p300" in card
    assert "| time to first token | TODO | TODO | - |" in card


def test_a_non_commercial_license_is_shown_and_says_non_commercial(tmp_path):
    card = render_card(facts())
    assert "CC BY-NC 4.0" in card and "Non-commercial use only." in card
    assert "Commercial use is not permitted by that license." in card


def test_a_permissive_license_card_has_no_non_commercial_line(tmp_path):
    card = render_card(facts(license_id="apache-2.0"))
    assert "Apache 2.0" in card and "Non-commercial" not in card
    assert card_problems(card, license_id="apache-2.0", run_dir=run_with_evidence(tmp_path)) == []


@pytest.mark.parametrize("claim", [
    "Commercial use is permitted.", "Ready for commercial deployments.", "Use it commercially.",
    "Production-ready for your customers.", "Licensed for enterprise use."])
def test_a_card_that_implies_commercial_use_of_a_non_commercial_model_is_refused(tmp_path, claim):
    card = render_card(facts()).replace("## What it runs", claim + "\n\n## What it runs")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any("implies commercial use" in p for p in problems), problems


def test_a_card_that_lost_its_license_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts())
    assert any("license" in p for p in card_problems(card.replace("license: cc-by-nc-4.0\n", ""),
                                                     license_id="cc-by-nc-4.0", run_dir=run))
    no_line = card.replace("Non-commercial use only. ", "")
    assert any("Non-commercial use only." in p
               for p in card_problems(no_line, license_id="cc-by-nc-4.0", run_dir=run))
    other = render_card(facts(license_id="apache-2.0"))
    assert any("apache-2.0" in p and "cc-by-nc-4.0" in p
               for p in card_problems(other, license_id="cc-by-nc-4.0", run_dir=run))


def test_a_measured_number_needs_its_evidence_and_its_value_in_the_first_file(tmp_path):
    run = run_with_evidence(tmp_path)
    missing = facts(numbers=(Number("ready", 280.5, "s", "measured", ("stages/2/nope.json",)),))
    assert any("stages/2/nope.json" in p for p in
               card_problems(render_card(missing), license_id="cc-by-nc-4.0", run_dir=run))
    wrong = facts(numbers=(Number("ready", 99.0, "s", "measured", ("stages/2/result.json",)),))
    assert any("99.0" in p for p in
               card_problems(render_card(wrong), license_id="cc-by-nc-4.0", run_dir=run))


def test_a_number_in_the_prose_is_refused(tmp_path):
    card = render_card(facts()).replace("## How to serve", "It decodes at 80 tok/s.\n\n## How to serve")
    assert any("80 tok/s" in p for p in
               card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path)))


def test_a_row_labelled_measured_by_hand_without_a_value_is_refused(tmp_path):
    card = render_card(facts()).replace("| time to first token | TODO | TODO | - |",
                                        "| time to first token | fast | measured | - |")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any("time to first token" in p for p in problems), problems


def test_the_license_is_read_from_the_snapshot_card(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: cc-by-nc-4.0\nbase_model:\n- Qwen/Qwen3.8-27B\n"
                                        "---\n# Hemmingway-1\n")
    assert read_license(tmp_path) == "cc-by-nc-4.0"
    (tmp_path / "README.md").write_text("---\nlicense: 'Apache-2.0'\n---\n")
    assert read_license(tmp_path) == "apache-2.0"
    (tmp_path / "README.md").write_text("# no front matter\nlicense: mit\n")
    assert read_license(tmp_path) is None
    (tmp_path / "README.md").unlink()
    assert read_license(tmp_path) is None


@pytest.mark.parametrize("license_id, nc", [("cc-by-nc-4.0", True), ("cc-by-nc-sa-4.0", True),
                                            ("cc-by-nc-nd-4.0", True), ("apache-2.0", False),
                                            ("mit", False), ("cc-by-4.0", False), ("other", True)])
def test_non_commercial_licenses_are_recognised(license_id, nc):
    assert non_commercial(license_id) is nc


def test_the_card_has_the_sections_the_catalog_standard_names(tmp_path):
    card = render_card(facts(sidecars=("joint_head.safetensors",), drafter_off=True))
    for head in ("## Intended use", "## Expected performance", "## Limitations",
                 "## Risks and safety considerations"):
        assert head in card, head
    limits = card.split("## Limitations")[1].split("\n## ")[0]
    assert "joint_head.safetensors" in limits and "speculative decoding is off" in limits.lower()
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path)) == []


@pytest.mark.parametrize("head", ["## Intended use", "## Limitations", "## Risks and safety considerations"])
def test_a_card_without_a_required_section_is_a_problem(tmp_path, head):
    card = render_card(facts()).replace(head, "## Other")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run_with_evidence(tmp_path))
    assert any(head in p for p in problems), problems
