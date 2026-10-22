# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The package card and its license (plan 5, stage 7)."""
import json

import pytest

from orchard.package_card import (CardFacts, Number, Reported, card_problems, non_commercial, read_license,
                                  render_card, sampling_disclosures)

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


# The card says which files the package takes from the nearest model. A model that ships no
# generation_config.json is served with the nearest model's stop tokens and sampling defaults (BAAI/AREX-2).

BASE_FILES = ("config.json", "preprocessor_config.json")


def test_the_card_lists_the_files_it_takes_from_the_nearest_model():
    card = render_card(facts(base_files=BASE_FILES))
    assert "(`config.json`, `preprocessor_config.json`, shipped in `base_config/`)" in card


def test_a_card_for_a_model_with_no_generation_config_says_whose_sampling_defaults_it_serves():
    card = render_card(facts(base_files=(*BASE_FILES, "generation_config.json"), sampling_from_base=True))
    assert "ships no generation_config.json" in card
    assert "stop tokens and the sampling defaults" in card
    assert "those of Qwen/Qwen3.8-27B" in card
    assert "Default sampling settings are those of Qwen/Qwen3.8-27B" in card.split("## Limitations")[1].split("##")[0]


def test_a_card_for_a_model_that_ships_its_own_generation_config_does_not_say_so():
    card = render_card(facts(base_files=BASE_FILES))
    assert "ships no generation_config.json" not in card
    assert "Default sampling settings" not in card


def test_the_check_accepts_the_rendered_card_for_both_cases(tmp_path):
    run = run_with_evidence(tmp_path)
    own = render_card(facts(base_files=BASE_FILES))
    assert card_problems(own, license_id="cc-by-nc-4.0", run_dir=run, base_files=BASE_FILES) == []
    names = (*BASE_FILES, "generation_config.json")
    borrowed = render_card(facts(base_files=names, sampling_from_base=True))
    f = facts(base_files=names, sampling_from_base=True)
    assert card_problems(borrowed, license_id="cc-by-nc-4.0", run_dir=run, base_files=names,
                         model_id=f.model_id, nearest_model=f.nearest_model) == []


def test_a_borrowed_generation_config_needs_both_sentences_naming_the_nearest_model(tmp_path):
    run = run_with_evidence(tmp_path)
    names = (*BASE_FILES, "generation_config.json")
    f = facts(base_files=names, sampling_from_base=True)
    card = render_card(f)
    def problems(c, **kw):
        return card_problems(c, license_id="cc-by-nc-4.0", run_dir=run, base_files=names,
                             **{"model_id": f.model_id, "nearest_model": f.nearest_model, **kw})
    first, second = sampling_disclosures(f.model_id, f.nearest_model)
    assert problems(card.replace(first, f"{f.model_id} ships no generation_config.json."))
    assert problems(card.replace(second, ""))
    assert problems(card.replace(f"those of {f.nearest_model}", "those of other/model"))
    assert problems(card, model_id="", nearest_model="")        # the gate must say whose defaults it expects
    assert "those of Qwen/Qwen3.8-27B." in card                  # the rendered text itself names the model
    assert problems(card.replace(", so the stop tokens", ",\nso the   stop tokens")) == []     # a re-wrapped card


def test_a_card_that_leaves_out_the_borrowed_sampling_defaults_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    names = (*BASE_FILES, "generation_config.json")
    card = render_card(facts(base_files=BASE_FILES))                   # rendered as if the model shipped its own
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, base_files=names)
    assert any("generation_config.json" in p and "sampling defaults" in p for p in problems)


def test_a_card_that_does_not_name_a_base_file_the_package_ships_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts(base_files=("config.json",)))
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, base_files=BASE_FILES)
    assert any("preprocessor_config.json" in p for p in problems)


def test_a_file_name_inside_other_text_does_not_count_as_named(tmp_path):
    run = run_with_evidence(tmp_path)
    card = render_card(facts(base_files=("config.json",))) + "\nsee my_preprocessor_config.json.bak\n"
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, base_files=BASE_FILES)
    assert any("preprocessor_config.json" in p for p in problems)


# ---- numbers the authors reported, kept apart from the ones measured here (#19) ---------------------------------

REPORTED = (Reported("Frontier-CS (agent track)", "70.7", "188 tasks"),
            Reported("MLE-Lite", "81.8", "Any Medal, mean of 3 seeds"),
            Reported("DeepSearchQA", "93.8%", ""))
SOURCE = "the model's card on Hugging Face"


def reported_card(**over):
    return render_card(facts(reported=REPORTED, reported_source=SOURCE, **over))


def test_a_card_with_reported_scores_has_a_section_that_says_they_were_not_measured_here():
    card = reported_card()
    sec = card.split("## Reported by the authors")[1].split("\n## ")[0]
    assert f"These scores come from {SOURCE}." in sec and "were not measured here" in sec
    assert "| Frontier-CS (agent track) | 70.7 | 188 tasks |" in sec
    assert "| DeepSearchQA | 93.8% |  |" in sec


def test_the_reported_section_comes_after_the_measured_numbers_and_before_the_limitations():
    card = reported_card()
    assert card.index("## Expected performance") < card.index("## Reported by the authors") < card.index("## Limitations")


def test_a_card_without_reported_scores_has_no_such_section():
    assert "Reported by the authors" not in render_card(facts())


def test_a_card_with_reported_scores_passes_its_own_check(tmp_path):
    run = run_with_evidence(tmp_path)
    assert card_problems(reported_card(), license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED) == []


def test_a_percent_in_the_reported_section_is_not_a_number_outside_the_numbers_table(tmp_path):
    run = run_with_evidence(tmp_path)
    problems = card_problems(reported_card(), license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED)
    assert not any("outside the Numbers table" in p for p in problems)


def test_a_percent_in_prose_outside_the_reported_section_is_still_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = reported_card().replace("## Limitations", "## Limitations\n\nIt is 93.8% good.\n")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED)
    assert any("outside the Numbers table" in p and "93.8%" in p for p in problems)


@pytest.mark.parametrize("cut, word", [
    ("## Reported by the authors", "section"),
    ("| MLE-Lite | 81.8 | Any Medal, mean of 3 seeds |", "MLE-Lite"),
    ("were not measured here", "not measured here"),
    ("| DeepSearchQA | 93.8% |  |", "DeepSearchQA"),
])
def test_a_card_that_leaves_reported_scores_or_the_disclaimer_out_is_refused(tmp_path, cut, word):
    run = run_with_evidence(tmp_path)
    card = reported_card()
    assert cut in card
    problems = card_problems(card.replace(cut, "x"), license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED)
    assert any(word in p for p in problems), problems


def test_a_card_that_changes_a_reported_value_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    card = reported_card().replace("| MLE-Lite | 81.8 |", "| MLE-Lite | 91.8 |")
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED)
    assert any("MLE-Lite" in p and "81.8" in p for p in problems)


def test_a_reported_score_listed_as_measured_is_refused(tmp_path):
    run = run_with_evidence(tmp_path)
    measured = Number("MLE-Lite", 0.5, "fraction", "measured", ("stages/2/result.json",))
    card = render_card(facts(reported=REPORTED, reported_source=SOURCE,
                             numbers=(*facts().numbers, measured)))
    problems = card_problems(card, license_id="cc-by-nc-4.0", run_dir=run, reported=REPORTED)
    assert any("MLE-Lite" in p and "measured" in p for p in problems)


def test_a_cell_cannot_break_the_table_or_the_card():
    rows = (Reported("a | b", "1\n2", "x\r\n## Heading in a note"),)
    card = render_card(facts(reported=rows, reported_source=SOURCE))
    sec = card.split("## Reported by the authors")[1].split("\n## ")[0]
    table = [ln for ln in sec.splitlines() if ln.startswith("| a")]
    assert len(table) == 1 and table[0].count("|") == 4 and "## Heading" in table[0]
    assert "\n## Heading" not in card
