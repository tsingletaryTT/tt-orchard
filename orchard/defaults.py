"""Every timing, budget and threshold the supervisor uses, one named constant each.

Each constant says where its number comes from. "Measured" means measured on this Quietbox 2 and
recorded in the spec (section 3 or 14) or the gozer-park skill. "Choice" means nobody measured
it; it is a starting value to revisit when a measurement exists.
"""
from __future__ import annotations

# ---- hardware facts -------------------------------------------------------------------------
# A QB2 p300c board has two chips. gozer leases whole boards: lease fb9995 (2026-10-02, H5) asked
# for 1 chip and was granted one unit with chips 0000:01:00.0 and 0000:02:00.0.
CHIPS_PER_BOARD = 2

# ---- measured timings (2026-10-02, this machine) ---------------------------------------------
GOZER_RESET_S = 41.7            # `gozer reset`, 7 runs, 41.6 to 41.7 s (spec section 3, gozer-park)
GOZER_RELEASE_S = 41.7          # `gozer release` including its reset, 4 runs, 41.6 to 41.7 s
TT_MODEL_STOP_S = 1.9           # `tt-model stop`, clean shutdown, 1.6 to 1.9 s (H5 container)
CONTAINER_WARM_BOOT_S = 20.3    # Audio8 container warm boot to ready, 19.6 to 20.3 s (H5)
WARM_RESTART_2CHIP_S = 180.0    # 2-chip Qwen3.8 warm restart, 2 to 3 min (spec section 3)
COLD_BOOT_S = 1800.0            # 2-chip cold first boot, about 30 min (spec section 3)
NEIGHBOUR_BUSY_S = 42.0         # the other board shows BUSY-UNTRACKED during any reset (spec section 8)

# ---- gozer protocol constants (read from tt-gozer gozer/queue.py) ------------------------------
GOZER_CLAIM_WINDOW_S = 90.0     # CLAIM_WINDOW_SECONDS: the head ticket's window to claim
GOZER_TICKET_MAX_AGE_S = 3600.0 # TICKET_MAX_AGE_SECONDS: a ticket expires after one hour

# ---- budgets derived from the measurements (the multipliers are choices) -----------------------
CMD_TIMEOUT_S = 120.0           # choice: gozer status/acquire, docker, ps, ss, curl; each takes < 2 s
START_TIMEOUT_S = 900.0         # choice, not measured: `tt-model serve --detach` for a container. The
                                # 20 s warm boot includes the watch; a start that reloads an image is
                                # unmeasured. On a timeout the start is left running and checked.
RESET_TIMEOUT_S = 600.0         # choice: about 14 times the measured reset; same as hardware_check
STOP_TIMEOUT_S = 120.0          # choice: `tt-model stop` measured under 4 s; docker's SIGKILL path is longer
QUIET_WAIT_S = 60.0             # choice: longer than NEIGHBOUR_BUSY_S, so a neighbour's reset can end
MESH_RESET_EXTRA_S = 60.0       # choice: added to the quiet wait when `tt-model stop` says it reset
                                # the mesh itself (its SIGKILL path), a reset gozer does not run
POLL_S = 2.0                    # choice: re-read interval while waiting for chips to go quiet
READY_POLL_S = 5.0              # choice: health poll interval while a server boots
COLD_BOOT_BUDGET_S = 2700.0     # choice: 1.5 times COLD_BOOT_S; past this the stage blocks (spec section 6)
STANDIN_READY_S = 600.0         # choice, not measured: CPU stand-in load time is open (spec section 12)
CANARY_TIMEOUT_S = 300.0        # choice, not measured: one short greedy answer
CANARY_MAX_TOKENS = 64          # choice: the canary needs a short answer only
QUEUE_POLL_S = 10.0             # choice: well inside GOZER_CLAIM_WINDOW_S
IDLE_RELEASE_S = 900.0          # choice: hold a lease through a phase with no hardware use up to
                                # 15 min. A release costs one reset (42 s) plus a queue wait; an
                                # image build (1.5 to 2.5 h) is far past this.

# ---- watchdog thresholds (replay of ~/.qwen transcripts on 2026-10-02) -------------------------
# The recorded loop: qwen-code 0.24.7, chat 197354ac, five identical calls of about 653 s each
# (input 145299, output 33348, thoughts 27939 tokens). 582 main-agent responses in 31 chats:
# thoughts p50 348, p95 5935, p99 16861, max 27939 (the loop). With these values the detectors
# fire only in the loop chat (tests/test_transcripts.py pins the exact findings).
# The basis is in-sample: the thresholds were chosen on the same 31 chats the tests replay. They
# separate this corpus; nothing yet shows they generalise. Plan 4's proxy is a different
# instrument from qwen-code telemetry, so the thresholds must be checked again on proxy events.
# Plan 4 passes them in from the run config; these constants are the defaults.
IDENTICAL_N = 3                 # measured: fires on the third call (about 33 min) of each 5-call repeat
THINKING_CAP = 20000            # measured: above every quiet response, below the loop's 27939
REPEAT_TOOL_N = 3               # measured: no transcript repeats a tool call; outputs repeat at most twice
NO_EVIDENCE_S = 3600.0          # choice, not measured: transcripts carry no evidence events
LEASE_IDLE_S = 1800.0           # choice, not measured
LEASE_POLL_S = 60.0             # choice: `gozer status` is read at most once a minute by the watchdog
RUNG_CAPS = {"nudge": 1, "escalate": 1, "pause": 1}   # choice: each rung once per agent and stage
