# Run log: bringing up Altworld/Hemmingway-1 with the tt-orchard harness

Kept by Claude for the operator. Times are local (PDT) unless marked Z (UTC). Each entry says what
happened and how it was checked. Numbers are measured unless marked "estimate" or "not verified".
The harness has not yet driven a model bring-up. This log records the work up to the first run and
then the run itself.

## 2026-10-02: choosing the model and preparing

- **Model.** The operator asked for `Altworld/Hemingway-1`. That repo does not exist. The model is
  `Altworld/Hemmingway-1` (two m's), a creative-writing fine-tune of `Qwen/Qwen3.8-27B`,
  license CC BY-NC 4.0. Found by searching the Hub for the author.
- **Stage 0 by hand, as a reference answer.** Compared config, tokenizer and every tensor against the
  base. The text config and all 866 text tensors match in name, shape and dtype. The base has 333
  more (vision) tensors. The tokenizer pre-tokenizer pattern differs for combining marks. The result
  is a weights-only delta. This was done by hand only to have an answer to check the harness against.
  The operator's goal is that the harness does this work. The reference is a test fixture.
- **Weights.** 51 GB downloaded to `/mnt/bonus/models/hemmingway-1/hf`, all 13 safetensors files match
  the Hub sizes. A disk swap by another session (SSD became `/mnt/bonus`) overlapped the first
  download attempt. The first attempt was stopped by its own pid and restarted on the new disk.
- **CPU tier measured.** `qwen3-coder:30b` (30B mixture-of-experts, 3B active, 4-bit) on ollama, chips
  idle: decode 14.1 tokens/s, prefill 93 tokens/s on a 1,605-token prompt. A dense 27B in bf16 on
  CPU: decode 0.70 tokens/s. The operator chose `qwen3-coder:30b`. Speed while the large model
  serves, and tool-call reliability, are not yet measured.
- **Tiers set.** Large = Qwen3.8-27B on all 4 chips. Small = Qwen3.8-27B on 2 chips. CPU =
  `qwen3-coder:30b`. (`config/tiers.toml`, not committed.)
- **Plan 3 built and merged (local main).** Adapters (gozer, single-tenant), park and restore with
  crash recovery, watchdog detectors and ladder. Two reviews before and after the build found one
  Critical and ten Important issues in the plan and one Critical and four Important in the code;
  all were fixed test-first. Suite: 854 passed, 1 skipped.
- **Real-board check of park and restore.** `orchard.park_check` on board 1 with fake servers: exit 0,
  both resets 41.7 s, canary compare passed, board free afterwards.
- **Watchdog replay against the real recorded loop.** The loop chat from the earlier failed
  attempt produces exactly the expected findings. The other 30 chats on disk produce none.
  This is in-sample: the thresholds were chosen from the same chats.
- **Plan 4 built and merged (local main).** Stage table 0 to 8, agent step loop, supervisor, end-to-end
  test that kills the supervisor after every ledger event (162 kill points with a 4-chip coder, 107
  with a 2-chip coder; every resume reached the same final state, in fakes). The final review found
  2 Critical and 5 Important problems, mainly that agent shells could reach lease, docker and process
  commands the runner did not refuse. All fixed test-first. Suite: 1141 passed, 1 skipped.
- **Device mask checked on hardware.** Agent shells get `TT_VISIBLE_DEVICES=0000:ff:00.0`. Under a
  lease on board 0, a device open with that mask fails with `RuntimeError: BDF pattern ... did not
  match any devices`. Code that clears the variable still gets every chip.
- **Credentials.** The operator's HF token, gh login, docker config and an ssh key without a
  passphrase are readable on this machine. The operator chose to run with them visible
  (`--accept-credentials-visible`). The command runner refuses push, upload, ssh and curl writes,
  but it is best effort and does not stop `python3 -c`. No adversarial search of the runner has been
  done.
- **Small tier smoke test.** Qwen3.8-27B served on board 0 (2 chips) through the full 262,144-token
  context setting, one chat request answered correctly. It ran under a gozer lease and was stopped
  when its background task hit a time limit; the lease went stale and was reaped.

## 2026-10-03: the first run

**Goal as restated by the operator (2026-10-03):** the harness should be able to produce a model that
can be packaged as a v5.1 or v6 tt-model package for the same hardware profiles we already ship
(1 chip, 2 chips, 4 chips). Plan 4 skips stage 7 (package and container build) on the weights-only
path, so packaging is the next piece to build after stages 0 to 6 are shown to work.

- **12:55:32Z (05:55 PDT) run 1 started.** `orchard.supervisor run`, pid 405138, run directory
  `/mnt/bonus/models/orchard-runs/hemmingway-1-run1`, ledger at `ledger.jsonl` there. Preconditions
  checked first: suite 1141 passed and 1 skipped, all four chips FREE, ollama up, ports 8000/8001
  free. The ledger records the operator accepting visible credentials. The supervisor took a lease
  on all four chips in 0.05 s and started the 4-chip coder container
  (`mando2222/qwen3.8-27b-dflash2-p300x2-q4kv`, profile batch8-dflash2, port 8000). The boot is
  running. This package was chosen from notes and `tt-model` output; its boot time on 4 chips has not
  been measured before.
