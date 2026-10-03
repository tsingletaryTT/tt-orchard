# Hardware-check driver: build report

Date: 2026-10-02. Nothing here touched a real device, a real gozer lease, real tt-smi, ttnn or docker.

## What was built

- `orchard/hardware_check.py`: the driver. Run as `python3 -m orchard.hardware_check --board <BDF> [options]`.
- `tests/test_hardware_check.py`: 30 tests against fake gozer roots.

The driver owns one lease under its own pid and runs P, A, H2, H3, H4, H1, H6 in that order (H1's idle
wait comes after H2-H4 so the lease is older than 900 s when the idle board is checked). Every result is
a ledger entry in `<out-dir>/ledger.jsonl`. `summary.md` is written next to it and printed.

Ledger shape: `run_start` first (options, gozer path, `gozer --version` result, driver pid). Then `notice`
entries with `data={"check", "ok", "evidence"}` (evidence has the command, exit code, stdout and stderr cut
to 2000 characters, and a one-line `summary`), `measurement` entries with `label="measured"`, and
`decision` entries. Check ids: `P`, `P.docker`, `P.queue`, `A`, `H2`, `H3.refuse`, `H3.stop`, `H3.gone`,
`H3.reset`, `H3.smi`, `H4`, `H4.stop`, `H4.gone`, `H1`, `H1.reconcile`, `H6.release`, `H6.free`,
`H6.docker`, plus `ABORT` and `cleanup.*` when they apply. Measurements: `H2_open_seconds`,
`H4_open_seconds`, `reset_seconds`, `lease_age_at_idle_check_seconds`, `release_seconds`.

Exit codes: 0 all pass, 1 a check failed or a stop condition hit, 2 preflight refused, 3 acquire failed
(or the grant did not match), 4 internal error.

## Options

| Option | Default |
|---|---|
| `--board BDF` (required) | first chip of the board |
| `--out-dir` | `runs/hardware-check/<UTC timestamp>-<BDF with : as ->/` (a numeric suffix is added if the name exists) |
| `--gozer` | `gozer` on PATH |
| `--tt-smi` | `tt-smi`; `none` skips the snapshot |
| `--docker` | `docker`; `none` skips `docker ps` |
| `--child-cmd` | JSON argv list; `{bdf}` and `{dev}` are filled in. Default: `bash -c 'source /home/ttuser/code/audio8-asr/dev_env.sh; export TT_VISIBLE_DEVICES=<BDF>; exec /home/ttuser/venvs/qwen36-p150x2/bin/python -c <snippet>'` |
| `--env-script`, `--python` | the two paths above, used by the default child |
| `--idle-seconds` | 960 |
| `--opened-timeout` | 300 |
| `--stop-timeout` | 30 (after `quit`, and again after SIGTERM) |
| `--cmd-timeout` | 120 (gozer status, queue, acquire, docker, ps, tt-smi) |
| `--reset-timeout` | 600 (reset and release) |

`main(argv, *, clock=time.monotonic, sleep=time.sleep, hook=None)`. `clock` and `sleep` are used only for
the idle wait. `hook(name, **info)` is called at `before_acquire`, `after_acquire`, `after_h2_open`,
`after_refuse`, `after_reset`, `before_idle_status`, `after_stop`; tests use it to inject events.

## How to run it on a real board

One instance per board. Two instances may run at the same time as separate processes. Each takes its own
lease with its own pid, writes its own directory, and ignores leases on the other board.

```bash
cd ~/code/tt-orchard
python3 -m orchard.hardware_check --board 0000:03:00.0     # board 1 (chips 2,3)
python3 -m orchard.hardware_check --board 0000:01:00.0     # board 0 (chips 0,1), a second terminal
```

Each writes `runs/hardware-check/<timestamp>-<BDF>/ledger.jsonl` and `summary.md`. Run it from a
long-lived shell or `tmux`. Do not wrap it in `gozer run`. The default child uses
`dev_env.sh`, which is written for board 1; the driver exports `TT_VISIBLE_DEVICES=<BDF>` after sourcing
it, but the file's `TT_MESH_GRAPH_DESC_PATH` (p150 descriptor) is the same for both boards of this
machine. For board 0, check the env script's other paths before relying on the default child.
The default total is about 17 minutes of wall time, dominated by `--idle-seconds 960`.

Reconcile rule: after the idle check, `gozer reconcile` runs only if no other lease is on the box. With
two instances running at once, each normally sees the other's lease and skips it. The skip decision names
the other lease's `who` (both instances use `orchard:hardware-check`, so the name is the same; the pid is
in the decision's `leases` list).

## Test results

- `python3 -m pytest -q -p no:cacheprovider`: 552 passed (522 existing plus 30 new). The new module takes
  about 47 s, mostly from the SIGTERM subprocess test, a no-OPENED timeout and a HELD-FOREIGN retry.
- The module is skipped with a reason if `/home/ttuser/code/tt-gozer-orchard/bin/gozer` is missing.
- Tests call `main()` in-process (the driver pid is the pytest pid). The SIGTERM test runs the driver in a
  subprocess and sends a real SIGTERM. Every test runs with fake `GOZER_ROOT`, sysfs, proc root,
  history root and reset command. `docker` and `tt-smi` are stub scripts. Real `ps` and `pgrep` are used
  (read only); they look at real pids of stub children and a `sleep`.

## Deviations from the brief

1. The brief said the marker file should show "2 resets by `gozer reset` plus the release (3 lines)". The
   specified sequence has one successful `gozer reset` (H3, after the child is gone). The refused call in
   H3 runs no reset, as the brief also requires. The happy-path test therefore expects 2 marker lines
   (one reset, one release) and 2 `gozer reset` calls (one refused, one that ran). If a second reset
   after H4 is wanted (the gozer-park skill does one after a workload ends), add it to `check_h4`.
2. A check that fails without being a stop condition (for example H2 showing HELD-FOREIGN) lets the run
   continue, and the exit code is 1. Stop conditions end the run at once.
3. `H3.stop` counts as a failure if the child needed SIGTERM or SIGKILL, because the runbook expects a
   clean stop.
4. The grant is checked after acquire. If the lease is granted but its chips are not CLAIMED, or the
   board is not in the grant, the driver exits 3 after releasing.
5. SIGINT and SIGTERM are blocked while `gozer acquire` runs, so a signal cannot arrive after the grant
   and before the driver knows the lease id. Signals during cleanup are ignored so release is not cut short.

## Mutation table

Each row removes or inverts one guard in `orchard/hardware_check.py`, runs the module with `python -B`
after clearing `__pycache__`, and records the first failing test. The script restores the file afterwards
(final source confirmed free of mutations). First pass: 8 survivors. Each got a test (rows marked "new").

| Mutation | Result | First failing test |
|---|---|---|
| finally: lease release removed | killed | child_never_opens_aborts_and_cleans_up |
| finally: child stop removed | killed | child_never_opens_aborts_and_cleans_up |
| cleanup release uses `--force` | killed (new) | child_never_opens_aborts_and_cleans_up |
| H2 accepts HELD-FOREIGN | killed | held_foreign_child_fails_h2 |
| H3 refusal check always true | killed (new) | h3_refusal_check_fails_when_the_reset_is_not_refused |
| H3 refusal check inverted | killed | happy_path_end_to_end |
| H3 refuse call skipped | killed | happy_path_end_to_end |
| H1 CLAIMED check always true | killed (new) | idle_check_fails_when_a_foreign_process_holds_the_device |
| reconcile gating removed | killed | reconcile_skipped_when_another_lease_is_present |
| reconcile gating inverted | killed | happy_path_end_to_end |
| child-gone: `ps -p` ignored | killed (new) | confirm_gone_needs_ps_to_show_no_row |
| child-gone: `pgrep -g` ignored | killed | stub_that_leaves_a_process_in_the_group_is_not_gone |
| child-gone: CLAIMED ignored | killed (new) | child_gone_check_fails_when_a_foreign_holder_remains |
| stop: unexpected lease on our board | killed | unexpected_lease_on_our_board_is_a_stop_condition |
| stop: lease disappeared | killed | lease_vanishing_is_a_stop_condition |
| stop: reset exit 17/18 | killed | reset_that_succeeds_under_a_holder_is_a_failure |
| stop: OPENED never arrives | killed | child_never_opens_aborts_and_cleans_up |
| stop: child exits before OPENED | killed (new) | child_that_exits_before_opening_is_reported_as_such |
| stop: command timeout is not an abort | killed (new) | a_command_that_times_out_is_a_stop_condition |
| docker timeout hard-coded to 300 s (found while writing the test above) | killed | a_command_that_times_out_is_a_stop_condition |
| idle wait skipped | killed | idle_check_waits_until_the_lease_is_old_enough |
| acquire without `--no-queue` | killed | happy_path_end_to_end |
| acquire without `--owner-pid` | killed | happy_path_end_to_end |
| signal handler does nothing | killed | sigterm_mid_run_stops_the_child_and_releases |
| signals not blocked during acquire | killed (new) | signals_are_blocked_while_acquire_runs_and_not_afterwards |
| acquire's CLAIMED check skipped | killed (new) | acquire_failing_its_claimed_check_exits_3_and_releases |
| H6 does not mark the lease released | killed | happy_path_end_to_end |
| `is_ours` ignores the pid | killed | another_drivers_lease_on_the_other_board_is_not_unexpected |
| board view covers every chip | killed | reconcile_skipped_when_another_lease_is_present |
| preflight ignores a busy board | killed | preflight_refuses_a_board_that_is_leased |
| child not in its own session | killed | happy_path_end_to_end |
| `quit` not written | killed | happy_path_end_to_end |
| SIGKILL escalation removed | killed | child_that_ignores_quit_and_sigterm_is_killed |
| TT_VISIBLE_DEVICES not set for the child | killed | happy_path_end_to_end |

Survivors: none on the final run. Not mutated: the retry-on-HELD-FOREIGN delay (a timing detail) and the
default-argv builder beyond the assertions in `test_default_child_argv`.

## Not covered by the dry run

- The real `ttnn.open_mesh_device` open and close, including its timing and whether `OPENED` appears at all.
- The real `tt-smi -s` output shape. The driver counts `device_info` if it is a list, else records `null`.
  The pass criterion is exit 0 only.
- Real `/proc` visibility: whether gozer sees the child's fds as HELD on this kernel, how long after exit
  the fds disappear, and whether `HELD-FOREIGN` shows for a moment while the child starts or stops. The
  driver reads status up to three times, one second apart, before it counts a mismatch.
- Real `gozer reset` (tt-smi reset) duration and the real `gozer release` reset.
- Real `docker ps` output, and whether containers from other agents map our devices.
- Real `ps` and `pgrep` on the actual process tree, including workers that leave the group.
- A real 960 s idle wait (the fake clock replaces it in tests), and the real owner-pid liveness rule over
  that time.
- Two instances running at the same time on the two real boards. The tests run one driver after another
  with the other board's lease held. Two live drivers at once were not run.
- The default `dev_env.sh` for board 0.

## Fix round 1 (safety review, 2026-10-02)

The sections above describe the first build. Where they disagree with this section, this section wins
(for example `--tt-smi` now defaults to `none`, the default `--gozer` is the branch binary, and the
sequence now has three `gozer reset` calls). Source: `orchard/hardware_check.py`. Tests:
`tests/test_hardware_check.py` (73 tests). Full suite: 595 passed (522 existing plus 73).

### What changed, per item

1. Before every `start_child` (H2 and H4) the driver reads `gozer status --json` through
   `guarded_board` and requires every granted chip to be ours and CLAIMED (`require_ready`, check ids
   `H2.ready`, `H4.ready`). The reset after H3 aborts on any exit other than 0, and on a JSON status
   other than `reset`.
2. `confirm_gone` is a gate at both call sites. It needs `ps -p` empty, `pgrep -g` empty and every
   granted chip CLAIMED.
3. Every gozer subprocess runs with `start_new_session=True`. `gozer reset` and `gozer release` run
   with signals held back (`protected()`): the handler records the first signal, the call runs to the
   end, and `deliver_pending()` raises it after the result is recorded. This differs from the brief's
   `pthread_sigmask`: a mask is inherited by gozer and by `tt-smi -r` under it, which would leave them
   unable to take SIGTERM. Acquire still uses the mask, now for SIGHUP too. Cleanup blocks the signals
   first, sets `in_cleanup` so the handler ignores them, then unblocks. `main` also calls the idempotent
   `cleanup` if a signal lands in the gap before `run()`'s own finally. No call is killed on an
   exception. A protected call that exceeds its timeout gets one more full timeout, then is left
   running (`abandoned`), the run aborts, and cleanup does not start a second release or reset.
4. The H3 refusal probe runs with `GOZER_RESET_CMD` set, for that call only, to
   `<out-dir>/refusal-probe-reset.sh`, which appends to `<out-dir>/refusal-probe-marker` and exits 1.
   It passes only with exit 15, the text `still open`, and no marker. Anything else aborts with
   "the refusal failed". The later real resets use the normal environment.
5. Default `--gozer` is `/home/ttuser/code/tt-gozer-orchard/bin/gozer`. Preflight runs
   `<gozer> reset --help` first (check `P.gozer`) and exits 2 if it fails.
6. Cleanup: kills the child's whole group when `pgrep -g` still finds processes (`cleanup.group`);
   retries a release that exits 15, 3 times, 5 s apart through the injected sleep, never with
   `--force`; prints a recovery block (lease id, holder pids, steps) to stderr and into `summary.md` when
   the lease cannot be released; wraps every ledger write in cleanup in `safe_notice` and releases
   before recording; handles SIGHUP like SIGTERM; marks `cleanup.stop` failed with "board needs a
   reset" when SIGTERM or SIGKILL was needed.
7. H6 passes only if release exited 0, the history log's last `released` event for the lease shows
   `reset_ran` and `reset_ok` true, and the message has neither `NOT marked clean` nor `not resetting`.
   Source used: the history log (`$GOZER_HISTORY_ROOT`, else `$GOZER_ROOT`, else `/tmp/tt-gozer`,
   read only), because `Keymaster.release` returns only the reset command's output plus the failure text,
   and a successful reset has no fixed message. The message check is a second guard. `H6.files` checks
   that `leases/<id>.json` and `gate/<unit>.lock` are gone.
8. The grant must be exactly the board: same chip set and dev indices, one unit, no neighbours,
   `owner_pid` equal to the driver pid, every chip CLAIMED. Otherwise the driver releases and exits 3.
9. `guarded_board` aborts if the child's pid is in `pids_holding` of a chip on another board.
   `--tt-smi` defaults to `none`. The help text and module docstring say it opens every device and must
   not run while another driver is active.
10. `summary.md` lists the gozer path and version, checks not run because of a stop, whether reconcile
    ran or was skipped and why, the H5 line, the `detached` gap, gozer status and `docker ps` at start
    and end (`END.status`, `END.docker` notices), the SIGKILL behaviour and any recovery block.
11. After `H4.gone` passes, `gozer reset` runs again (`H4.reset`, measurement `reset2_seconds`), with
    the same exit, status and all-CLAIMED requirements. The happy path has 3 `gozer reset` calls (the
    probe, then two real) and 3 marker lines (two resets and the release).

Minor items: the old 17-exit test is renamed `test_failed_chip_reset_exit_17_is_a_stop_condition`; no
test uses `pkill -f` (cleanup now kills the child's group, and the test checks it); the default child
exports `TT_METAL_CACHE=<out-dir>/cache` and `TT_METAL_LOGS_PATH=<out-dir>/logs` after sourcing the env
script; `detached` is not in the acquire JSON, so the summary says so; unreadable acquire output prints
a stderr warning and records `A.malformed`; the module docstring and summary describe a SIGKILL of the
driver. The two phrase-sweep hits (this report and `hardware-validation.md` line 29) are rewritten.

### Review probes turned into tests

| Probe | Test |
|---|---|
| reset runs after failed gone check | `test_failed_gone_check_stops_before_any_reset` |
| un-refused reset does not stop the run | `test_unrefused_probe_is_a_stop_and_never_runs_the_real_reset` |
| second reset refused, H4 still opens | `test_a_reset_exit_other_than_0_after_the_child_is_gone_stops_the_run[13, 15]` |
| signal during release kills gozer | `test_signal_during_release_lets_gozer_finish` |

### RED evidence

Before the change, the new and updated tests ran against the first build: 35 failed, 27 passed. The
failures included all four probe tests above (for the reasons the reviewer found: the second reset
still ran with 2 reset calls and H4 present; no ABORT after an un-refused probe; H4 opened after
exit 13 and 15; a signal during release gave 2 release calls and a killed gozer), every grant-shape
case except the `owner_pid` one (the first build already checked that), the summary, history, stray
file, retry, recovery, group-kill, malformed-JSON and preflight tests. A few failed first on a changed
function signature (`default_child_argv`) or a missing hook, which is a weaker reason than the
behaviour under test. Tests added after that run were written to kill specific survivors (below).

### Mutation table (final source restored; `python -B`, `__pycache__` cleared before each run)

60 mutations. First pass: 8 survivors. Each got a test or an explanation. The rows below show the final
state.

| Mutation | Result | First failing test |
|---|---|---|
| cleanup: release removed | killed | child_never_opens_aborts_and_cleans_up |
| cleanup: child stop removed | killed | child_never_opens_aborts_and_cleans_up |
| cleanup: group kill removed | killed | stub_that_leaves_a_process_in_the_group_is_not_gone |
| cleanup: release retry removed | killed | cleanup_retries_a_release_refused_with_15_and_never_forces |
| cleanup: release uses --force | killed | child_never_opens_aborts_and_cleans_up |
| cleanup: recovery block not printed | killed | cleanup_that_cannot_release_prints_recovery_steps |
| cleanup: recovery text not in summary | killed | cleanup_that_cannot_release_prints_recovery_steps |
| cleanup: ledger write can skip release | killed | a_failing_ledger_write_in_cleanup_cannot_skip_the_release |
| cleanup: forced stop not marked failed | killed | signal_mid_run_stops_the_child_and_releases[15] |
| cleanup: abandoned reset ignored | killed | a_reset_that_outlives_its_timeout_is_not_killed_or_repeated |
| abandon flag not set | killed | a_reset_that_outlives_its_timeout_is_not_killed_or_repeated |
| timeout second wait removed | killed | a_reset_slower_than_one_timeout_but_not_two_is_waited_for |
| H2 accepts HELD-FOREIGN | killed | held_foreign_child_fails_h2 |
| probe: exit 15 not required | killed (new test) | probe_needs_exit_15_even_with_the_right_text |
| probe: `still open` not required | killed | probe_needs_still_open_text_with_exit_15 |
| probe: marker not required | killed (new test) | probe_fails_if_its_stand_in_reset_ran_even_with_exit_15 |
| probe failure not a stop | killed | unrefused_probe_is_a_stop_and_never_runs_the_real_reset |
| probe: reset-command override removed | killed | unrefused_probe_is_a_stop_and_never_runs_the_real_reset |
| probe: not protected from signals | killed (new test) | signal_during_the_refusal_probe_lets_it_finish |
| reset: only 17 and 18 stop | killed | a_reset_exit_other_than_0_after_the_child_is_gone_stops_the_run[13] |
| reset: status text not required | killed | reset_exit_0_with_a_status_other_than_reset_stops_the_run |
| reset: not protected from signals | killed | a_reset_that_outlives_its_timeout_is_not_killed_or_repeated |
| reset: CLAIMED-after check not a stop | killed (test tightened) | chips_must_be_claimed_after_the_reset |
| gate: H3 gone not enforced | killed | stub_that_leaves_a_process_in_the_group_is_not_gone |
| gate: H4 gone not enforced | killed | failed_gone_check_after_h4_stops_before_the_second_reset |
| gone: `ps -p` ignored | killed (test fixed) | confirm_gone_needs_ps_to_show_no_row |
| gone: `pgrep -g` ignored | killed | stub_that_leaves_a_process_in_the_group_is_not_gone |
| gone: CLAIMED ignored | killed | child_gone_check_fails_when_a_foreign_holder_remains |
| claimed: first chip only | killed | every_granted_chip_must_be_claimed_after_the_stop |
| ready: check removed before open | killed | the_lease_is_checked_before_every_device_open |
| ready: failure not a stop | killed | the_lease_is_checked_before_every_device_open |
| grant: whole shape check removed | killed | grant_that_is_not_exactly_our_board[owner_pid] |
| grant: chip set equality removed | equivalent | none (see note) |
| grant: single unit removed | killed | grant_that_is_not_exactly_our_board[units] |
| grant: dev indices removed | killed (new case `[2, 9]`) | grant_that_is_not_exactly_our_board[dev_indices] |
| grant: neighbours removed | killed | grant_that_is_not_exactly_our_board[neighbours] |
| grant: owner pid removed | killed | grant_that_is_not_exactly_our_board[owner_pid] |
| other board: check removed | killed | child_touching_the_other_board_is_a_stop_condition |
| preflight: `reset --help` not required | killed | preflight_refuses_a_gozer_without_reset |
| default gozer is the PATH gozer | killed | default_gozer_is_the_branch_binary_and_tt_smi_is_off |
| tt-smi on by default | killed | default_gozer_is_the_branch_binary_and_tt_smi_is_off |
| H6: history confirmation removed | killed | h6_fails_on_the_history_even_if_the_message_looks_clean |
| H6: message check removed | killed | h6_fails_on_the_message_even_if_history_looks_fine |
| H6: stray-files check always passes | killed | h6_checks_for_stray_files_under_gozer_root |
| H6: not protected from signals | killed | signal_during_release_lets_gozer_finish |
| H4.reset removed | killed | happy_path_end_to_end |
| signals: handler does not defer | killed | signal_during_release_lets_gozer_finish |
| signals: pending never delivered | killed | signal_during_release_lets_gozer_finish |
| signals: handler ignores everything | killed | signal_mid_run_stops_the_child_and_releases[15] |
| signals: SIGHUP not handled | killed | signal_mid_run_stops_the_child_and_releases[1] |
| signals: cleanup does not ignore signals | killed | second_signal_during_cleanup_does_not_start_another_release |
| gozer calls share the driver's session | killed | a_command_that_times_out_is_a_stop_condition (and every_gozer_call_runs_in_its_own_session) |
| acquire: signals not blocked | killed | signals_are_blocked_while_acquire_runs_and_not_afterwards |
| acquire: unreadable-output warning removed | killed | acquire_with_unreadable_json_says_a_lease_may_exist |
| child cache not per run | killed | default_child_argv |
| SIGKILL escalation removed | killed | child_that_ignores_quit_and_sigterm_is_killed |

Note on the equivalent mutant (retired in round 2): the chip-set equality in the grant check overlaps with the length
test in `all_claimed`, so removing only one of them leaves the behaviour unchanged. Removing both (run
by hand, then restored) fails `grant_that_is_not_exactly_our_board[chips]`.

Not mutation-checked: the `main()` fallback `cleanup` for a signal that lands in the gap before
`run()`'s finally (a race that no test can force), and the `Child` reader-thread details.

### Still not covered by the dry run

- Everything in the earlier list: the real ttnn open, the real `tt-smi` output, real `/proc`
  visibility and timing, real reset durations, real `docker ps`, a real 960 s wait, two live drivers.
- Whether the real `gozer release` writes the `released` event to the history path the driver
  resolves. On this machine `GOZER_HISTORY_ROOT` is normally unset, so the driver reads
  `$GOZER_ROOT/history.jsonl` or `/tmp/tt-gozer/history.jsonl`. If the file cannot be read, H6 fails
  with "reset confirmed in history: None". That fails closed.
- The real `gozer reset` text for a refusal: the driver needs `still open`, which the code and the tests
  in `tt-gozer-orchard` produce today.
- A real reset command under the probe: the stand-in proves the call path and says nothing about what a real
  `tt-smi -r` would do if gozer failed to refuse.
- `signal.pthread_sigmask` and process groups on a different kernel than this one.
- First real session advice from the review: one board alone, a short `--idle-seconds`, a
  human-watched second terminal. The full 960 s run and two drivers at once come after one clean run per board.

## Fix round 2

Tests: 85 in `tests/test_hardware_check.py` (12 new, plus the existing grant test gained a case). All
existing tests now pass `--allow-gozer-env`, because they run with fake `GOZER_*` roots.

### What changed

- I1. For a protected call (`gozer reset`, `gozer release`) the signal deferral now starts before
  `Popen`, and ends after the result is recorded. A signal during fork and exec waits. Test:
  `test_a_signal_during_popen_of_a_protected_call_is_deferred` sends SIGTERM from inside a patched
  `Popen`; the probe completes and is logged, only the cleanup release resets the chips, and the run ends
  with the signal's abort.
- I2. `run_start` records every `GOZER_*` variable (`gozer_env`). Preflight check `P.env` refuses with
  exit 2, before any gozer call that changes state, when one is set. `--allow-gozer-env` (test only)
  skips the refusal. Tests: refusal names `GOZER_RESET_CMD` and no acquire happens; with the flag the run
  passes; `gozer_env()` is empty in a clean environment.
- M1. `release_reset_confirmed(..., since=)` ignores `released` events older than the acquire time
  (recorded in gozer's `%Y-%m-%dT%H:%M:%SZ` form). Tests: a unit test, and a run where an older event with
  our lease id and `reset_ok` true is appended after the real release (whose reset failed): H6 fails.
- M2. The probe stand-in lives under the absolute out-dir and quotes the marker path with `shlex.quote`.
  Tests: an out-dir named `out dir $x`y`"q`, and `--out-dir .`, both with an un-refused probe so the
  stand-in runs and its marker is found.
- M3. New grant case `d['chips'] = d['chips'][:1]`. The driver aborts, exits 3 and releases.
- M4. For a protected call, an exception while waiting (for example KeyboardInterrupt) sets `abandoned`
  and leaves gozer running. Unprotected calls are still killed. Test calls `run_cmd` with a patched
  `Popen` whose `communicate` raises.
- M5. `--pause-after-open SECONDS` (default 0). After H2 and before the probe, the driver prints the
  child pid and `ls -l /proc/<pid>/fd` to stderr, then sleeps in 1 s slices through the injected sleep.
  Hook `after_pause` marks the end. The runbook was not edited.

### RED evidence

Before the code change (with only the `--allow-gozer-env` and `--pause-after-open` options added so the
tests could parse), 10 of the 19 selected tests failed: the Popen signal test (a signal during Popen left
no logged probe), the env refusal and flag tests, the `since` unit and run tests, both odd out-dir cases
(the stand-in marker was not found), the BaseException test, and the pause test. The new M3 case already
passed, because `all_claimed`'s length check catches it too. Its mutation check is below.

### Targeted mutations (each ran only the matching tests, `python -B`, source restored and compared)

| Mutation | Result | Test |
|---|---|---|
| I1: deferral moved after Popen | killed | a_signal_during_popen_of_a_protected_call_is_deferred |
| I2: env check removed | killed | gozer_env_overrides_refuse_the_run |
| M1: `since` dropped | killed | h6_ignores_an_old_released_event_for_the_same_lease |
| M2: abspath removed | killed | probe_stand_in_survives_odd_out_dirs[dot] |
| M2: quoting removed | killed | probe_stand_in_survives_odd_out_dirs[odd name] |
| M3: grant chip-set equality removed | killed | grant_that_is_not_exactly_our_board[chips[:1]] |
| M4: protected BaseException kills gozer | killed | a_baseexception_while_waiting_does_not_kill_a_protected_call |
| M5: pause removed | killed | pause_after_open_waits_before_the_probe_and_says_what_to_check |

The M3 case retires the "equivalent mutant" note from round 1.

### Still not covered

- The earlier lists (real ttnn open, real /proc timing, real reset durations, a real 960 s wait, two
  live drivers).
- A signal landing between the `with guard:` exit and `deliver_pending` is held until the next
  `deliver_pending`, which every protected step calls after recording. A step that never reaches one
  (for example a crash) leaves the signal pending.
- `--pause-after-open` was tested with a fake sleep. Its real timing against a person is untested.
- `P.env` checks only variables that start with `GOZER_`. A `PATH` that resolves a different `tt-smi`
  for the real reset command is out of its reach.
