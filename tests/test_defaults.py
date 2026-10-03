"""The defaults keep the relations the measurements require."""
from orchard import defaults as d


def test_queue_poll_stays_well_inside_gozers_claim_window():
    # The head ticket gets a 90 s claim window. A poll that sleeps longer can lose its place.
    assert d.QUEUE_POLL_S * 3 <= d.GOZER_CLAIM_WINDOW_S


def test_quiet_wait_outlasts_a_neighbours_reset():
    # During any reset the other board shows BUSY-UNTRACKED for about 42 s.
    assert d.QUIET_WAIT_S > d.NEIGHBOUR_BUSY_S
    assert d.QUIET_WAIT_S + d.MESH_RESET_EXTRA_S > 2 * d.NEIGHBOUR_BUSY_S


def test_a_container_start_gets_longer_than_a_command():
    assert d.START_TIMEOUT_S > d.CMD_TIMEOUT_S > 10 * d.TT_MODEL_STOP_S


def test_cold_boot_budget_exceeds_the_measured_cold_boot():
    assert d.COLD_BOOT_BUDGET_S > d.COLD_BOOT_S > d.WARM_RESTART_2CHIP_S


def test_reset_timeout_is_far_past_the_measured_reset():
    assert d.RESET_TIMEOUT_S >= 10 * d.GOZER_RESET_S


def test_watchdog_thresholds_sit_between_the_quiet_chats_and_the_loop():
    # Quiet chats: thoughts_token_count p99 16861. The loop: 27939 on each call.
    assert 16861 < d.THINKING_CAP < 27939
    assert d.IDENTICAL_N == 3 and d.REPEAT_TOOL_N == 3


def test_every_rung_has_a_cap_of_at_least_one():
    assert set(d.RUNG_CAPS) == {"nudge", "escalate", "pause"}
    assert all(v >= 1 for v in d.RUNG_CAPS.values())


def test_every_stage_has_a_budget_and_a_disk_need():
    assert set(d.STAGE_BUDGET_S) == set(range(9)) == set(d.STAGE_DISK_GB)


def test_a_hardware_stage_budget_holds_a_cold_boot():
    # Stages 2 to 6 may boot the model cold (about 30 min) at least once.
    assert all(d.STAGE_BUDGET_S[n] > d.COLD_BOOT_BUDGET_S for n in (2, 3, 4, 5, 6))


def test_a_hardware_stage_disk_need_covers_the_measured_tensor_cache():
    # The base model's 2-chip tensor cache measured 34 GB.
    assert all(d.STAGE_DISK_GB[n] >= 34 for n in (2, 3, 4, 5, 6))


def test_a_model_request_may_outlast_the_measured_prefill():
    assert d.AGENT_REQUEST_TIMEOUT_S > 78


def test_run_caps_allow_one_escalation_and_one_relaunch():
    assert d.RUN_ESCALATION_CAP >= 1 and d.RUN_COLD_BOOT_CAP >= 2


def test_a_cold_boot_is_told_apart_from_a_warm_restart():
    # Warm 2-chip restart 2-3 min, cold first boot about 30 min (spec section 3).
    assert d.WARM_RESTART_2CHIP_S < d.COLD_START_S < d.COLD_BOOT_S


def test_an_agent_reply_has_room_for_reasoning_and_a_tool_call():
    # The live Qwen3.8 run used up 8,192 max_tokens on reasoning in 3 of its failed replies.
    assert d.AGENT_MAX_TOKENS == 16384


def test_the_weights_only_stage_2_budget_holds_the_skills_hardware_test():
    # The weights-swap-check skill's template script allows HEALTH_TIMEOUT_S (3300 s) for the
    # server to be ready (a cold weight conversion, a cold kernel compile and the start) plus about
    # 120 s of requests, and the skill asks for a deadline_s of 3600. The supervisor runs the test for min(deadline_s, the
    # stage budget), so the budget must not cut it.
    import re
    from pathlib import Path

    from orchard.stages import spec_for
    skills = Path(d.__file__).with_name("skills")
    skill = (skills / "weights-swap-check.md").read_text()
    deadline = int(re.search(r'"deadline_s": (\d+)', skill).group(1))
    script = (skills / "weights-swap-templates" / "serve_and_compare.py").read_text()
    health = float(re.search(r"^HEALTH_TIMEOUT_S = ([\d.]+)", script, re.MULTILINE).group(1))
    assert deadline == 3600 and health == 3300 and deadline >= health + 120
    assert min(deadline, spec_for(2, "weights-only").budget_s) == deadline
    assert spec_for(2, "weights-only").disk_gb >= 34       # one converted 2-chip tensor cache


def test_the_weights_only_stage_4_disk_holds_three_measured_caches():
    # 34 GB (2-chip bundle) and 31 GB (4-chip container) measured; the 1-chip cache is assumed 34 GB.
    assert d.STAGE4_SWAP_DISK_GB >= 34 + 34 + 31
    assert d.TEST_DISK_GB >= 34


def test_the_stage_4_budget_holds_three_tests_and_a_park():
    # Three tests at the skill's 3600 s each (a first boot with cold caches took more than 26 min
    # on 2026-10-03, and the 4-chip conversion is not measured), one park and restore with a coder
    # boot within the cold-boot budget, and one agent step of TOOL_TIMEOUT_S each for prepare and
    # finish.
    need = (3 * 3600 + d.TT_MODEL_STOP_S + 2 * d.GOZER_RESET_S + d.COLD_BOOT_BUDGET_S
            + 2 * d.TOOL_TIMEOUT_S)
    assert d.STAGE_BUDGET_S[4] >= need


def test_the_package_stage_budget_holds_the_install_and_the_boot_check():
    assert (d.PACKAGE_INSTALL_TIMEOUT_S + d.PACKAGE_VERIFY_DEADLINE_S + d.PACKAGE_THIN_TIMEOUT_S
            < d.STAGE_BUDGET_S[7])
    assert d.PACKAGE_HEALTH_TIMEOUT_S + 120 <= d.PACKAGE_VERIFY_DEADLINE_S
    assert d.PACKAGE_VERIFY_DEADLINE_S > d.COLD_BOOT_S


def test_the_boot_check_waits_as_long_as_stage_2s_first_boot():
    # The installed copy starts with cold caches, like stage 2's first boot, so its health wait is
    # at least the swap template's (3300 s: weight conversion plus a cold kernel compile).
    import re
    from pathlib import Path
    script = (Path(d.__file__).with_name("skills") / "weights-swap-templates" / "serve_and_compare.py").read_text()
    health = float(re.search(r"^HEALTH_TIMEOUT_S = ([\d.]+)", script, re.MULTILINE).group(1))
    assert d.PACKAGE_HEALTH_TIMEOUT_S >= health


def test_the_package_stage_disk_need_covers_a_fresh_tensor_cache():
    assert d.STAGE_DISK_GB[7] >= 34 + 40


def test_every_deferred_package_format_is_a_known_format_and_v6_is_not_deferred():
    assert set(d.PACKAGE_DEFERRED) <= set(d.PACKAGE_FORMATS)
    assert "v6" in d.PACKAGE_FORMATS and "v6" not in d.PACKAGE_DEFERRED
