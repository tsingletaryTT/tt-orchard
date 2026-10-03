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
