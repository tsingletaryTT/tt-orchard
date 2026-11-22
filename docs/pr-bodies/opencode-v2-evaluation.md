## What

An evaluation, with no change under `orchard/`: is OpenCode v2 (2.3.2) a better coding harness than orchard's own agent loop for tt-lang kernel work and for
tenstorrent/skills, and where it runs slower, are its results better? The run log is `docs/run-logs/2026-11-21-opencode-v2-vs-orchard-harness.md`; its data is in
`docs/run-logs/2026-11-21-opencode-v2-vs-orchard-harness-data/`.

## Result (counts; the log has every table)

Same model (Qwen3-Coder-Next, four chips, one server), same task text, skill text, attempt directory and unedited simulator check; only the agent loop differs. 28 attempts, 14 per harness.

- Simulator passes: OpenCode 5 of 14, orchard 4 of 14 (`add` 2 of 2 each; `center` 0 of 6 against 1 of 6; `row_sum` 3 of 6 against 1 of 6). No comparison has a Fisher exact p below 0.48.
- Passes the chip compilers accept (compile step on the CPU, no device; the script is from #802): the pair's compiler accepts 4 of OpenCode's 5 and 2 of orchard's 4; tt-lang 1.1.6 accepts 2 and 2.
- The nine passing kernels are equal on the simulator's measures and all pass three held-out seeds. No failure is a near miss.
- OpenCode sends 1.8 to 6.4 times the prompt tokens and decodes more slowly at its long contexts; decode is 78% to 97% of wall time. On attempts with no load flag it was slower only on `add`.
- OpenCode finds `SKILL.md` skills by itself in four layouts and loaded the skill in 9 of 12 fact questions (the other 3 answered from the description alone and missed facts). No bring-up stage was run through it.

Conclusion: not shown to be better or worse; a larger, quieter run is the next step. The log lists what bounds this: the old plain cheat sheet's wrong chip claims (issue 801, PR 803), simulator passes that are not chip passes, load windows and another client on the coder
during 3 attempts, small N, the retraction of the recipes A/B (1 of 7 against 0 of 8 on chip-acceptable passes).

## Added

- `scripts/opencode_probe.py`: runs the real kernel probe with OpenCode as the agent loop (it replaces only `kernelprobe.AgentStep`).
- `scripts/landlock_exec.py`, `scripts/landlock_selfcheck.py`: the sandbox. Bubblewrap cannot start on the test host (AppArmor refuses unprivileged user namespaces), so the runner uses Landlock (filesystem and TCP rules, no root).
- `scripts/opencode_ab.py` (pair driver with load and server counters), `scripts/oc_quality.py` and `scripts/oc_report.py` (the quality readout and the tables), `scripts/oc_skill_discovery.py`, `scripts/oc_request_proxy.py` (logs what OpenCode sends).
- The run log, its data directory and `changes/opencode-v2-evaluation.toml`.

## Evidence

The log records every setup trap with its evidence: OpenCode sent no temperature until the sampling went into the model options; with no model set it picked a hosted model of its own; the tool list held 15 tools where six were registered; `tools.skill: false` has no effect;
the step limit is soft; a reply cut at 4096 tokens twice ends an attempt in both harnesses. Not run here: the full suite (the change touches no `orchard/` file; the ratchet and path tests, 69, pass), any chip or lease, any change to the coder.

## Hardening checklist

1. Wrong type or shape in input: the scripts read files this evaluation wrote (timeline, quality, compile-probe output) and the probe's own records; nothing under `orchard/` reads them. A wrong shape raises `KeyError`, it does not pass.
2. Files something else controls: the runner starts OpenCode on a directory a model writes, under Landlock (read-only system paths, write only the attempt directory and a private home, connect to the model port only). `oc_quality.py` parses `kernel.py` with `ast` (no execution) and reads the probe's own judged copy, checked against the sha256 in provenance; the held-out run goes through the probe's `authoritative_check` (copy, then check).
3. Check, then use: the held-out run and the verdicts use the copy the probe made and judged; the agent's path is never read.
4. Evidence an agent can write: verdicts come from the unedited check on the probe-owned copy. The OpenCode event streams are written by the host from the process's stdout, outside the paths the sandbox can write. The attempt record is read from the probe-owned verdict directory (it sorts first).
5. A verdict stronger than its evidence: the log gives counts, exact p-values and what each cannot show; the compile probe is described as the first step only; no chip result is claimed.
6. The check looks at a different object than the action uses: no gate is added.
7. A parser or matcher that is too broad or narrow: the static call matcher is approximate and says so; the load-window parsers read the coordinator's files once (the first version misread the mutation lines and was replaced; the counts are checked against the originals).
8. Failure paths that leave state behind: each attempt starts a server and a client in their own sessions and kills both groups in a `finally`; pids are recorded; the logging proxy is stopped by pid. State is under `~/orchard-runs/ocprobe` outside the repository.
9. Numbers and sizes: wall, step and token limits are the registered ones; event files read by the quality tool are 5 to 30 MB (these are evaluation scripts and no harness code).
10. Secrets, paths and host names: the committed data has home paths and the host name replaced (checked with grep); no token is in any file; the per-attempt OpenCode server password stays in host-side files that are not committed. The log names the operator's OpenCode session by role only.
11. Tests that do not test the claim: the sandbox has a self-check with real calls (`landlock_selfcheck.py`, 21 of 21 as expected). No orchard test was added because no orchard code changed.
12. Housekeeping: change fragment added (target 0.9.192, `scripts/version.py check` passes), the ratchet and path tests pass; the PR is not merged and no review is requested.
