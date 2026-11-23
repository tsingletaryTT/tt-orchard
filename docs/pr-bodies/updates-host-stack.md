Closes #809.

## What

`tt-orchard updates` also reports SFPI, tt-kmd, tt-flash, tt-smi, the firmware bundle on the chips and the reference venv's torch, transformers, tokenizers, safetensors and huggingface-hub. Read only, in the existing row shape (table, JSON, cache, preflight hint and the web view need no change). New module `orchard/updates_host.py`; small edits in `orchard/updates.py` and `orchard/cli.py`.

| Row | Installed from | Compared with |
| --- | --- | --- |
| sfpi | `dpkg-query -W -f=${Version} sfpi` | apt candidate; the host tt-metal tree's `tt_metal/sfpi-version`; the floor of each installed bundle's ttnn (`<bundle>/venv/.../ttnn/tt_metal/sfpi-version`) |
| tt-kmd | `modinfo -F version tenstorrent` | apt candidate of `tenstorrent-dkms` |
| tt-flash, tt-smi | the `.dist-info` folder of the venv named on the script's first line (no process started) | public PyPI |
| firmware | the driver's sysfs file, read in a daemon thread with a 3 s wait | newest release in the public Atom feed of tt-system-firmware |
| reference venv (5 packages) | `.dist-info` folders of the `reference_python` venv | public PyPI; a major jump is labelled `major: do not auto-upgrade` |

New rules in the table: R4 (SFPI moves with tt-metal and tt-model-manager), R5 (tt-kmd, firmware, tt-flash, tt-smi: no vendor pairing table exists for newer releases, so the report invents none), R6 (the reference packages; a torch or transformers change can move reference numerics).

New flag `--reference-python`. The bringup config's `reference_python` and `tt_model_root` supply the defaults.

## Why it never runs tt-smi

On 2026-11-22 several `tt-smi -s` calls queued behind the kernel driver's mutex while another process was starting a mesh. They stayed in uninterruptible wait and could not be killed. This command opens no chip. The only processes it starts are `dpkg-query -W -f=${Version}` and `apt-cache policy` for `sfpi` and `tenstorrent-dkms`, and `modinfo -F version tenstorrent`. Any other argument raises `NotReadOnly` before the command runs.

## Evidence

- 41 tests in `tests/test_updates_host.py` with fake command output and files under `tmp_path`. No real `/sys`, no real command, no chip.
- 26 mutations, one per guard, all turn a test red (the list is below). Two survived at first: a shebang that points outside a venv, and a firmware file that holds no version. Each got a test.
- A run against the real machine and the live services (every row except the firmware read) found two defects the fakes had missed: tt-smi is a link on PATH into the venv, which `safeio` refuses to read, and PyPI's JSON for tokenizers is 2.5 MB, above the 2 MB answer limit. Each got a test and a mutation; PyPI answers now have an 8 MB limit.
- The two file readers are in `tests/hostile_inputs.py` `READERS`: 84 hostile cases pass. Replacing the bounded read with a plain `read_bytes()` turns the FIFO, `/dev/zero` and 50 MB cases red.
- Existing updates, refresh, view, README, CLI, hardening ratchet, finite ratchet, hazards and version tests pass (5,365 passed, 2 skipped in the broad targeted run; the full suite was not run).
- Not run: the sysfs read against the real driver (the chips were in use and this work was not to touch them). Whether that file can block on the driver mutex is unknown; the thread and the timeout are the guard.

Mutations: M1 to M3 (the three fixed commands accept any argument), M4 and M5 (host list, suffix match), M6 (release candidates count), M7 (first feed entry instead of the highest), M8 (no zero padding), M9 (firmware read waits for the driver), M10 (symlinked dist-info followed), M11 and M12 (major label, numerics note), M13 (kmd note for any candidate), M14 (bundle floors ignored), M15 (host rows keep the kernel-probe ready text), M16 (R5 invents a pairing), M17 (shebang outside a venv), M18 (PyPI version not cleaned), M19 (bundle scan unbounded), M20 (highest chip reported), M21 (tree pin ignored), M22 (a failed read counted as a value), M23 (host rows without a host), M25 (console script link not resolved), M26 and M27 (PyPI size limit), and M24 (plain `read_bytes()`, red in the hostile registry).

## Hardening checklist

1. Wrong type or shape in input: yes. Every remote string goes through `updates.clean` (size, type, control characters); a PyPI version without digits is "no version in the answer"; a pin file that is not `sfpi_version='x.y.z'` gives None (`test_the_highest_bundle_floor_wins_and_a_garbage_pin_file_is_ignored`, `test_hostile_answers_are_cut_and_cleaned_before_they_reach_a_row`).
2. Files something else controls: yes. `sfpi-version` files, shebang lines and the sysfs file are read with `safeio.read_bytes` or `read_prefix` (4 KB bound); `.dist-info` folders are listed with `os.scandir`, symlinks refused, 5,000 entries at most; bundle floors at most 40 files. Both readers are in the hostile-input registry.
3. Check, then use: n/a. Nothing is acted on; the report prints text.
4. Evidence an agent can write: n/a. No agent writes these inputs.
5. Verdicts and labels: yes. `ready to farm` for host rows is always `unverified`, with the reason. A row says `not read: ...` when the driver did not answer. A major jump is labelled, and it gets no upgrade command.
6. The check inspects the same object the action uses: n/a. The report runs no action. It says the proposed commands are text.
7. Parsers and matchers: yes. Version compare pads with zeros (19.17.1 equals 19.17.1.0); release candidates are not releases (the tag pattern accepts digits and dots only); feed order does not matter.
8. Failure paths: yes. A missing tool, package, module, venv, feed or driver answer gives one `unknown (why)` row; one failing check does not stop the others (`check_host`).
9. Numbers and sizes: yes. Bounded reads and counts as listed under item 2; the feed text is cut at the existing 2 MB body limit and 60 entries.
10. Secrets, paths and host names: yes. Rows carry bundle folder names and versions, no paths. The `HF_TOKEN` rule is unchanged; the new host `github.com` receives no token.
11. Tests: yes. No skips. Each guard has a mutation that turns a test red.
12. Housekeeping: yes. `changes/updates-host-stack.toml` (0.9.192; other open PRs also claim 0.9.192), a README section and rows, a project-log entry.
