# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which tests are guards, which are slow, which need a chip (docs/test-tiers.md).

This module holds data and two small functions. `tests/conftest.py` applies it as pytest markers, `scripts/tests_for_diff.py` reads
`WHOLE_REPO` to know which tests every selection must keep, and `tests/test_tier_lists.py` checks the lists against the files.

A test file that is in no list is in the default tier: it runs under `-m "not slow"` and under the full suite.
"""
from __future__ import annotations

# Guards: tests that scan the repository, so a new module, command flag, skill file or change fragment can fail them while no test of the
# new code does. Each runs on every change. An empty tuple means every test of the file; a tuple of names picks those functions only.
# Found with a fault run on 2026-11-23: a scratch copy of main with one fault of each kind (a module with raw file access, an unchecked
# float, direct sqlite use and an unregistered reader; a new flag the README does not name; a home path in a skill file and in a package
# template; a malformed change fragment), then one full suite. The ten failures that run produced named the first eight entries below.
# tests/test_claude_md_size.py is listed because it checks a repository-wide file size that every change to CLAUDE.md can break. The last
# two come from the scanner check in tests/test_tier_lists.py, which found test files that walk the source tree.
WHOLE_REPO: dict[str, tuple[str, ...]] = {
    "tests/test_finite_ratchet.py": (),
    "tests/test_hardening_ratchet.py": (),
    "tests/test_no_sql_outside_the_database_layer.py": (),
    "tests/test_readme.py": (),
    "tests/test_skills.py": (),
    "tests/test_version.py": (),
    "tests/test_version_cli.py": ("test_the_script_check_passes_on_this_repository",),
    "tests/test_hostile_readers.py": ("test_the_registry_covers_these_modules_and_the_rest_are_listed",),
    "tests/test_claude_md_size.py": (),
    "tests/test_runaway.py": ("test_no_source_test_or_doc_file_of_this_repository_is_called_a_loop",),
    "tests/test_block_family_authoring_skill.py": (),
}

# Test files the scanner check flags that are not guards, with the reason.
NOT_GUARDS: dict[str, str] = {
    "tests/test_expert_skew.py": "the script reads docs/*.md only in read_text_chunks, which the tests never call",
    "tests/test_fidelity_score.py": "the script reads docs/*.md only in read_prompts, which the tests never call",
    "tests/test_text_package.py": "it walks a package the test builds in a temporary directory, not the repository",
}

# Files above 30 seconds of test time in the full run of 2026-11-23 (six workers, other suites running on the same machine, so each time
# is a little high). The 51 files hold 8,547 of 31,910 tests and 3,987 of the 5,796 seconds of test time (69 percent). A guard that
# lives in one of these files is named in WHOLE_REPO and keeps the whole_repo marker; the rest of the file is slow.
SLOW_FILES: frozenset[str] = frozenset({
    "tests/test_agreement_set_template.py",
    "tests/test_audio_chip_decoder.py",
    "tests/test_audio_reference_script.py",
    "tests/test_audio_stages.py",
    "tests/test_bench_session.py",
    "tests/test_can.py",
    "tests/test_can_audio.py",
    "tests/test_can_audio_job.py",
    "tests/test_can_music.py",
    "tests/test_can_served_probes_boot.py",
    "tests/test_can_served_verify.py",
    "tests/test_can_vllm.py",
    "tests/test_code_video_bench.py",
    "tests/test_cpu_run_before_board.py",
    "tests/test_delta_triage_audio.py",
    "tests/test_delta_triage_template.py",
    "tests/test_dense_load_reuse.py",
    "tests/test_full_port_serve.py",
    "tests/test_hardware_check.py",
    "tests/test_helper_draft_tested.py",
    "tests/test_helper_drives_port.py",
    "tests/test_helper_numerics_rounds.py",
    "tests/test_hostile_readers.py",
    "tests/test_kernel_loop_templates.py",
    "tests/test_kernelprobe.py",
    "tests/test_kernelstaged.py",
    "tests/test_make_franken.py",
    "tests/test_ops_inventory_stub.py",
    "tests/test_ops_inventory_template.py",
    "tests/test_package_command.py",
    "tests/test_phone_a_friend_ladder.py",
    "tests/test_port_template_moe_parallel.py",
    "tests/test_prompt_suite_template.py",
    "tests/test_queue_probe.py",
    "tests/test_qwen4_dense.py",
    "tests/test_qwen4_ops_probe_moe.py",
    "tests/test_reference_gate_template.py",
    "tests/test_run_e2e.py",
    "tests/test_sampling_probe_template.py",
    "tests/test_scoring_check_template.py",
    "tests/test_spawn_checked_sweep.py",
    "tests/test_staged_run_log_scripts.py",
    "tests/test_supervisor_audio_runtime.py",
    "tests/test_supervisor_full_port.py",
    "tests/test_supervisor_full_port_stage3.py",
    "tests/test_supervisor_full_port_stage4.py",
    "tests/test_supervisor_lab.py",
    "tests/test_supervisor_phone_a_friend.py",
    "tests/test_supervisor_port_repair.py",
    "tests/test_supervisor_telegraph_in_repair.py",
    "tests/test_unrecoverable_positions.py",
})

# Files whose tests need a Tenstorrent chip. The suite never opens a device (CLAUDE.md: importing ttnn or ttml opens it), so none exist.
# A file listed here is skipped unless ORCHARD_RUN_HARDWARE=1, and its test has to take a gozer lease itself.
HARDWARE_FILES: frozenset[str] = frozenset()


def markers_for(rel: str, test_name: str, hardware_files: frozenset[str] | None = None) -> tuple[str, ...]:
    """The tier markers for one test: `rel` is the repository-relative path of its file, `test_name` its function name without parameters."""
    hardware = HARDWARE_FILES if hardware_files is None else hardware_files
    out: list[str] = []
    names = WHOLE_REPO.get(rel)
    guard = names is not None and (not names or test_name in names)
    if guard:
        out.append("whole_repo")
    elif rel in SLOW_FILES:
        out.append("slow")
    if rel in hardware:
        out.append("hardware")
    return tuple(out)


def hardware_skip_reason(env) -> str | None:
    """Why a hardware test is skipped, or None when the operator has opted in."""
    if env.get("ORCHARD_RUN_HARDWARE") == "1":
        return None
    return "needs a Tenstorrent chip: set ORCHARD_RUN_HARDWARE=1 and take a gozer lease"
