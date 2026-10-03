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
START_TIMEOUT_S = 900.0         # choice; not measured: `tt-model serve --detach` for a container. The
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
STANDIN_READY_S = 600.0         # choice; not measured: CPU stand-in load time is open (spec section 12)
CANARY_TIMEOUT_S = 300.0        # choice; not measured: one short greedy answer
CANARY_MAX_TOKENS = 64          # choice: with thinking switched off the answer is a few tokens; 64 leaves room
                                # for a short sentence. Not enough if a server ignores the switch.
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
NO_EVIDENCE_S = 3600.0          # choice; not measured: transcripts carry no evidence events
LEASE_IDLE_S = 1800.0           # choice; not measured
LEASE_POLL_S = 60.0             # choice: `gozer status` is read at most once a minute by the watchdog
RUNG_CAPS = {"nudge": 1, "escalate": 1, "pause": 1}   # choice: each rung once per agent and stage
TURN_REPEAT_N = 3               # choice: model turns in a row with the same set of tool calls. Same
                                # value as REPEAT_TOOL_N. The live stage 2 run ran the same two grep
                                # commands in each of 5 turns; REPEAT_TOOL_N saw them alternate and
                                # never fired. The committed qwen transcript signatures stay quiet.
WRITELESS_TURNS = 20            # choice: model turns in a row in which no file was written (no
                                # successful write_file, no new evidence file) before the step is
                                # nudged. A third of AGENT_MAX_TURNS. In the committed qwen transcript
                                # signatures the longest such run is 18 turns (counting qwen-code's
                                # write tools), so 20 stays quiet there. The live stage 2 run that
                                # grepped vLLM source for 60 turns would have been nudged at turn 20.

# ---- plan 4: stages, agent steps and the run ---------------------------------------------------
# Per-stage wall-clock budgets (spec section 10). Choices; none is measured. Stage 5 holds one 2-chip
# cold boot (COLD_BOOT_S, about 30 min) plus its checks; stage 4 boots two configurations. Stage 7
# is skipped in plan 4, so its budget is 0.
STAGE_BUDGET_S = {0: 7200.0, 1: 14400.0, 2: 14400.0, 3: 21600.0, 4: 28800.0, 5: 10800.0,
                  6: 14400.0, 7: 0.0, 8: 3600.0}
# Free disk each stage needs on the run directory's filesystem before it starts (spec section 10).
# 40 GB covers one converted 2-chip tensor cache: the base Qwen3.8-27B TP=2 cache measured 34 GB
# (2026-09-30). Stage 4 converts a second (1-chip) cache. The other values are choices.
STAGE_DISK_GB = {0: 1.0, 1: 5.0, 2: 40.0, 3: 40.0, 4: 80.0, 5: 40.0, 6: 40.0, 7: 0.0, 8: 1.0}
LONG_STAGE_S = 3600.0           # spec section 10: a stage with a longer budget must declare a resume marker
STAGE2_PCC_MIN = 0.995          # the functional-decoder skill's default acceptance bar (prefill and decode)
SWAP_TOP1_MIN = 0.85            # choice: the weights-only stage 2 bar for top1_agreement (teacher-forced
                                # next-token agreement between the chip and the stage 1 CPU reference).
                                # Two measurements on 2026-10-03, Hemmingway-1 CPU reference, the 2-chip
                                # Qwen3.8-27B bundle, one prompt of 32 tokens each: 0.78 (25 of 32) when
                                # the base Qwen3.8-27B weights stood in by mistake (the bundle's HF_MODEL),
                                # and 0.94 (30 of 32) with the correct weights (MODEL_WEIGHTS_DIR set).
                                # The old bar of 0.6 passed the base weights. 0.85 sits between the two;
                                # the margin is thin and rests on one prompt. Revisit when more
                                # weights-only runs exist.
SWAP_MIN_TOKENS = 16            # choice: the fewest compared tokens the weights-only gate accepts; the
                                # skill's script compares 32
AGENT_MAX_TURNS = 60            # choice: model turns in one agent step before the step counts as failed
AGENT_CONTINUATION_TURNS = 20   # choice: model turns for a step's gate-feedback continuation (at most one).
                                # It only has to fix what the gate named, so it gets a third of a step.
AGENT_THINKING = False          # choice: agent turns run with the model's thinking mode off. The 2-chip
                                # DFlash2 server decodes greedily only. In thinking mode a greedy model
                                # circles in its reasoning (6,000 tokens on one point, no command) and
                                # the step fails. Replaying one failed turn with thinking off gave a
                                # correct command in 437 tokens. Checked on 2026-10-03 on one turn only.
AGENT_MAX_TOKENS = 16384        # choice: max_tokens for one model response. A reasoning model's
                                # thinking tokens count toward it. On the live Qwen3.8 run, 8,192 was
                                # used up by reasoning in 3 of the failed replies. At about 80 tokens/s
                                # a reply that uses all 16,384 takes about 200 s.
AGENT_REQUEST_TIMEOUT_S = 900.0 # choice: one non-streaming response; prefill alone measured 42 to 78 s
                                # at 130K to 204K tokens (spec section 3), and contexts here stay short
TOOL_TIMEOUT_S = 1800.0         # choice: one shell command run for an agent
TOOL_OUTPUT_CHARS = 12000       # choice: the head and tail of a command's output that go back to the model
CONTEXT_FILE_CHARS = 4000       # choice: how much of each earlier stage's result file a new context holds
SKILL_CHARS = 40000             # choice: a skill longer than this is cut, and the context says so
RUN_ESCALATION_CAP = 3          # choice: escalations since the last operator resume before the run pauses
RUN_COLD_BOOT_CAP = 3           # choice: cold coder boots since the last resume before the run pauses
COLD_START_S = 600.0            # choice: a coder start slower than this counts as a cold boot; it sits
                                # between the 2-3 min warm restart and the ~30 min cold boot (spec section 3)
RUN_WALL_CLOCK_S = 259200.0     # choice: 72 h from run start or the last resume, then the run pauses
CONTROL_POLL_S = 10.0           # choice: how often a paused supervisor reads the control file
RUN_CANARY_PROMPT = "What is 17 + 25? Answer with one number."   # choice: short, one greedy answer
FIRST_BOOT_PROMPT = "What is 7 times 6? Reply with only the number."   # choice: a known answer, asked once
FIRST_BOOT_EXPECTED = "42"      # a first start must answer with text that contains this, or the run blocks

# ---- stage 4 on the weights-only path: one hardware test per chip configuration -----------------
# Each configuration converts the new weights into its own tensor cache. Measured sizes for a
# Qwen3.8-27B-sized model: 34 GB for the 2-chip bundle (2026-09-30, and the Hemmingway-1 prototype
# on 2026-10-03) and 31 GB for the 4-chip plain container (rebuilt 2026-10-03). The 1-chip cache was
# not measured and is assumed to be the same size.
STAGE4_SWAP_DISK_GB = 110.0     # choice: three caches of about 34 GB (1, 2 and 4 chips) plus margin,
                                # checked on the run directory's disk before stage 4 starts
TEST_DISK_GB = 40.0             # choice: one cache plus margin, checked again before each
                                # configuration's test, so a resumed stage stops before a test that
                                # cannot finish its conversion
