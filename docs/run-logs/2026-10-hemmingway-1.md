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
- **12:55:33Z to 12:58:46Z run 1 blocked at the coder.** The 4-chip container booted in about 2 minutes
  (warm cache). The supervisor's canary request then got `content: null` and the run paused
  ("the coder did not start and answer"). Two causes, found by sending the same style of request by hand:
  1. *Harness bug.* The model reasons before it answers. With a small token budget the reply is cut off
     while still reasoning, so `content` is null. The canary code reported that as "not text" with no
     explanation, and it does not switch thinking off.
  2. *The package is broken on this box.* `mando2222/qwen3.8-27b-dflash2-p300x2-q4kv` produced noise in
     every request (reasoning text such as `ignyikkoignyikko…`). Its own log shows
     `accept 0.00/7`: the speculative decoder accepted none of its drafts. It decodes at about
     17 tokens/s. The 2-chip small tier served earlier from the `episod` package answered correctly, so
     the box and the base weights are fine. Cause of the noise is not known; it was not debugged.
  The harness stopped on the right thing, but by accident: a server returning garbage inside
  `content` would have passed this canary. A first-boot check with a known answer is needed.
- **12:58:46Z operator abort, first use of the abort path on hardware.** The supervisor stopped the
  coder with `tt-model stop`, released the lease, and exited in 47 s. All four chips FREE, no
  containers left. The ledger holds the pause, the abort, `stopping the coder`, and `hardware released`.
- **Decision (operator, 2026-10-03):** required chip configurations for this model are 2 and 4.
  1 chip is optional. Stage 4's gate is being changed to take `--required-chips 2,4`.
- **Fixes after run 1 (merged to main, suite 1173 passed, 1 skipped).**
  (a) The canary now switches thinking off and, if a reply is still cut off during reasoning, says so.
  (b) On a coder's first start in a run, the supervisor asks a known-answer question (7 times 6) and
  blocks the run if the answer does not contain 42. Four new tests cover the noise case, and each
  guard was mutated and seen to fail. (c) Stage 4's gate takes `--required-chips 2,4`; a required
  configuration that is missing or fails still fails the stage, and a failed 1-chip attempt does not.
- **13:18:40Z run 2 started** (pid 662891, run directory `hemmingway-1-run2`). The coder is the plain
  4-chip package `changh95/qwen3.8-27b-p300x2` (profile batch32, 32 sequences, 262,144 context, no
  speculative decoder), chosen because the Mando package returned noise. Same credentials acceptance
  as run 1.
- **13:18:40Z to 13:22:27Z run 2 blocked at the coder, again.** The plain 4-chip package
  `changh95/qwen3.8-27b-p300x2` booted in 2 minutes and returned noise too (the same
  `ignyikko` tokens as the Mando package). This time the first-boot known-answer check caught it:
  asked "What is 7 times 6?", the coder returned noise, and the run paused with the answer saved as
  evidence. This is the first fault the harness found by design. Abort worked again (coder stopped,
  lease released, chips FREE, 4 of 4 verified).
  Two different 4-chip packages, with different images and caches, give the same wrong output. The
  2-chip `episod` package answered correctly on board 0 earlier on 2026-10-02. Firmware is the same on
  all four chips (bundle 19.15.0.0, `tt-smi -s`). The cause is not known. It looks specific to the
  4-chip configuration on this machine and not to either package. Not debugged yet; it is an open
  hardware question.
- **13:23:52Z run 3 started** (pid 669563, run directory `hemmingway-1-run3`) with the 2-chip bundle
  `episod/qwen3.8-27b-dflash2-p300` as the coder, so the harness can be exercised while the 4-chip
  question is open. One board stays free for hardware stages, so no parking is needed. Its first-boot
  check will also show whether the chips still run correctly.
- **13:25:49Z run 3: the coder is healthy and stage 0 has started.** The 2-chip bundle coder was ready in
  120.1 s (warm cache). Its canary and first-boot answer was `42`, so the first-boot check passed. This
  also shows the chips and base weights are fine, which supports the reading that the 4-chip noise is
  specific to the 4-chip configuration. At 13:25:49Z the harness began stage 0 (delta triage): the
  first agent step on the large tier, model Qwen/Qwen3.8-27B, skill `orchard/skills/delta-triage.md`.
  This is the first stage the harness has started on a real model.
- **13:25:49Z to 13:53:15Z run 3, stage 0 (delta triage): pass, 27 minutes, 59 agent turns.** The agent
  ran its own comparison scripts and wrote 9 kinds of evidence plus `stages/0/delta.json` (path:
  weights-only). The gate passed. I (not the harness) compared `delta.json` with the hand-written
  reference. Every reference item is in the harness's list: identical text config; 866 matching
  text tensors with 333 extra vision tensors in the base; the `model.*` versus
  `model.language_model.*` prefix; identical vocabulary, merges and added tokens; the pre-tokenizer
  dropping `\p{M}` (the agent measured 6 of 176 test strings differing, all with Indic or Thai
  combining marks); identical chat template and generation config; 13 shards with the 15 `mtp` keys in
  a separate file; the three hazards (the old tensor cache is keyed by layer name only, the drafter
  was trained on the base model, root disk is 99% full). It also found a difference the reference did
  not list: the license changes from Apache 2.0 to CC BY-NC 4.0 (the reference had noted the
  license but not the change from the base's). Its numbers differ slightly from mine (176 test
  strings against my 212), which is expected from different test sets.
- **Contamination check on the reference answer.** The agent was not given the reference. In turn 4 it
  ran `ls` on the sibling `work/` directory, which showed that `stage0-reference.md` exists.
  No command opened it. In its notes and final message it wrote "my findings match it" without
  having read it, so that sentence is unsupported and is not evidence. My own comparison above is the
  check. The runbook's grep is how this was found. A lesson for the harness: an agent's claim of
  checking something it did not open is a failure to catch in the review step.
- **13:53:16Z stage 1 started** (reference gate). The ledger notes "tier substituted": the small
  tier is not serving, so the large server runs the step.
- **13:53:16Z to 13:56:22Z run 3, stage 1 (reference gate): first attempt failed in 3 minutes.** The
  agent spent 7 turns surveying Python environments and the model README. On turn 8 the model used
  its whole 8,192-token budget reasoning and returned `finish_reason: length` with no text and no
  tool call. The agent loop reads "no tool call" as "finished" and never checks `finish_reason`, so
  the stage ended without `reference.json`. The gate failed and the stage machine escalated, moved the
  partial directory to `stages/1.partial-1` (kept, not deleted), and restarted stage 1 at 13:56:22Z with
  the escalation tier (the same large server, since it is the only chip tier serving). Fix in
  progress: a truncated reply gets one retry with a nudge; two in a row end the step as an error.
- **13:56:22Z to 14:01:43Z run 3, stage 1 failed again after escalation** (second attempt: a model reply
  of 8,192 tokens with no tool call and no text on turn 12). The run paused with
  "stage 1 failed after escalation: reference.json is missing". The supervisor kept its lease and the
  coder, as designed.
- **14:02Z truncation fix merged** (`agent-truncation`; suite 1180 passed, 1 skipped): a reply cut off at
  `max_tokens` with no tool call gets a nudge and one retry, and a second cut-off in a row ends the
  step as an error. Tests were seen to fail before the fix and each guard was mutated. The commit
  message and report came from an implementer whose hand-back text was a placeholder; the commit,
  report file and suite result were checked directly.
- **14:03Z crash recovery tried on real hardware.** To load the fixed code into a paused run, I sent
  SIGKILL to the supervisor (pid 669576, which I started). The coder survived, still holding board 0
  under a lease whose owner was dead (gozer showed HELD-FOREIGN). I restarted the supervisor on the same
  run directory (pid 748699) and sent `resume` at 14:08:52Z. The supervisor wrote "relaunch the coder
  under this supervisor's lease", stopped the old coder with `tt-model stop` (clean, 1.0 s), took a new
  lease (it landed on board 1, chips 2 and 3, because board 0's old lease had not been reaped yet),
  started the 2-chip bundle there, and had it ready in about 110 s. The canary answer after the
  restart has the same sha256 as before it (answer `42`), so the greedy canary was identical across a
  restart for this model. This is the first run of the crash-recovery path on hardware; the
  tests had covered it only with fakes. The board-1 `IndexError` seen earlier with another build did
  not occur with this bundle.
- **14:10:47Z stage 1 started again** with the fixed agent loop (escalated attempt, partial directory
  `stages/1.partial-2` kept).
- **14:10:47Z to 14:41:42Z run 3, stage 1 failed a third time (fixed agent loop).** The agent worked
  49 turns over 30 minutes and wrote real evidence (CPU reference logits, a tokenizer round-trip, a
  model-card check, a summary). The truncation fix fired twice as designed (turns 9 and 48, with a
  watchdog notice each time). At turn 48 the agent had just found a bug in its own reference script
  (an off-by-one in how it recorded the sequence). Its reply on turn 48 was cut off. After the nudge,
  turn 49 returned an empty reply: 159 tokens of reasoning, no text, no tool call,
  `finish_reason: stop`. The loop read that as "finished", the gate found no `reference.json`, and the
  stage failed after escalation. Two more gaps found:
  (1) an empty reply is not a final answer; (2) a failed gate restarts the stage with a fresh
  context, which throws away 48 turns of work, and the escalation tier here is the same model.
  Fix in progress: empty replies get a nudge like truncated ones; the reply budget goes from 8,192 to
  16,384 tokens; and the agent gets one continuation in the same conversation, with the gate's reasons,
  before the stage is escalated. The run is paused; the coder is idle on board 1.
  Stage 1 is the point where the harness has spent the most effort without a pass.
- **15:12Z to 15:20Z run 3, stage 1 failed a fourth time, within 6 turns.** With the empty-reply handling,
  the 16,384-token budget and the same-conversation gate feedback all in place (merged, suite 1199
  passed, 1 skipped), the agent's turn 5 used all 16,384 tokens and turn 6 was an empty reply.
  A second recovery worked: the supervisor was restarted, stopped the idle coder, took board 0, and the
  coder was ready in 130 s with the canary answer again identical to the first (two restarts, same
  answer). The reply budget was not the limit.
- **Why the agent fails: greedy decoding in thinking mode.** The operator's reading was that this
  needs a stronger, 4-chip model. I replayed the failing turn-5 request against the idle coder.
  With thinking on: 6,000 tokens of reasoning (20,175 characters), no command, circling the same
  point. With thinking off: one correct tool call in 437 tokens, plus a correct diagnosis of the
  earlier attempt's script bug. The 2-chip server decodes greedily only (temperature 0), and a
  reasoning model without sampling loops in its thinking. This is one replayed turn, not a
  measured rate. A model that can sample, such as the plain 4-chip package, would avoid the loop,
  but both 4-chip packages return noise on this machine, so the 4-chip route is blocked until that is
  understood.
- **Change:** agent turns now send `enable_thinking: false` (`AGENT_THINKING = False`; a step can turn it
  back on). Two tests, each mutated and seen to fail. Suite 1201 passed, 1 skipped.
- **15:29:59Z to 15:38:13Z run 3, stage 1 (reference gate): pass, 25 turns, 8 minutes, with agent thinking
  off.** A third recovery also worked (supervisor restarted; coder back on board 0 in 120 s; canary again
  identical, so three restarts with the same answer `42`). The agent wrote `reference.json` with four
  checks. Three are solid: the weights load on CPU with 0 missing, 0 unexpected and 0 mismatched
  keys (the 15 `mtp.*` keys are ignored by the model class, which it noticed and explained); 14 of 14
  tokenizer round trips, including Devanagari and Thai; and a greedy decode of the card's landlord
  prompt whose first token equals the argmax of the logits, with reference ids and top-5 logits saved
  as evidence for later comparison with the chip. The fourth, "matches the card", is weak: the generated
  text begins "Let me work through this carefully. The user wants me to write a text message...",
  which is the model reasoning aloud, while the card says it "just gives you the text". The check
  passed because the output was non-empty and had no preamble pattern; the agent wrote that caveat
  into its own note, so it was honest about it. To follow up: why the model reasons first under the
  default chat template, and whether the chip's output should be compared in that mode.
- **15:38:13Z stage 2 (functional decoder on one chip) started,** prepare step on the large tier. This is
  the first stage that will need a board.
- **15:38Z to 16:08Z run 3, stage 2 prepare step: 60 turns, no hardware test written.** The agent spent the
  whole step reading tt-metal source (94 `cd`/`ls`/`sed` calls, one real command) and never wrote
  `hw_test.json`. The stage machine escalated and restarted it. This is a skill problem: the stage uses the
  generic `functional-decoder` skill, written for porting a new architecture. For a weights-only delta no
  decoder needs writing. The existing Qwen3.8 TT implementation only needs the new weights loaded.
  I paused the run at 16:09:19Z and prototyped the weights swap by hand on board 1 to learn the real
  recipe, so the skill I give the harness is true. **This was done by hand to write the skill; it is
  not harness output.**
- **Prototype findings (16:10Z to 16:20Z, board 1, 2 chips, existing episod bundle stack).**
  1. The bundle registers the TT model class only for the architecture name
     `Qwen3_5ForConditionalGeneration`; Hemmingway-1 says `Qwen3_5ForCausalLM`. Registering the second name
     got past the lookup (the log shows the TT class selected) and then failed in vLLM's multimodal
     setup: it expects the vision-language config type and got the text-only one.
  2. What works: a model directory shaped like the base, with the base's `config.json`,
     `preprocessor_config.json` and `video_preprocessor_config.json` copied in and Hemmingway-1's tokenizer
     files, chat template, generation config, weight index and shards linked in. The runtime skips the vision
     tower and accepts the flat `model.*` tensor names, so no vision weights are needed. The bundle's
     `run.sh` is copied with `--model` pointed at that directory and the pinned revisions removed.
  3. Weights are converted into a fresh, empty tensor cache (34 GB). The first server start, including that
     conversion, took about 5 minutes (09:10:59 to about 09:16 local), not 30. The page cache was warm.
  4. The server rejects `logprobs` ("owns its Gumbel sampling"), so comparisons use generated ids only.
  5. Chip output is coherent Hemmingway-style text with no noise. Greedy free-run text differs from the CPU
     reference from the first token ("We need to respond..." against "Let me work through this..."), as
     expected for bfp4/bfp8 chip weights against bf16 CPU weights. Teacher-forced next-token agreement with
     the stage 1 CPU reference, over the 32 reference tokens: **25 of 32 (78%)**. All seven differences are
     near-synonyms ("Let"/"We", "work"/"think", "wants"/"is", "practical"/"straightforward",
     "real"/"everyday"). One prompt and 32 tokens, so an indication and not a benchmark.
  The lease was released with a reset afterwards. Board 1 is free.
- **16:20Z to 16:35Z the 4-chip noise: cause found, a stale tensor cache.** The operator suggested it could be
  sharing weights or caches with other Qwen3.8-based models, which they had seen before. I checked: each
  4-chip package has its own tensor cache under `~/.cache/tt-model/<package>/tensors`, and all read the
  same single Hugging Face snapshot (`1d4bf0f`, the one the working 2-chip package uses). The September
  caches (built 2026-09-15 and 2026-09-17) are older than that snapshot (rewritten 2026-09-23). The test:
  I stopped the idle coder (supervisor killed with SIGKILL, coder stopped with `tt-model stop`, stale lease
  released with a reset), moved `qwen3.8-27b-p300x2/tensors` aside (renamed to
  `tensors.aside-20261003`, 31 GB, not deleted), and booted the plain 4-chip package with an empty cache
  under a 4-chip lease. It answered `42` to "What is 7 times 6?" and "The capital of France is **Paris**.",
  with no noise. The cache rebuilt to 31 GB. **So the two 4-chip packages' noise was a stale cache and not
  the hardware or the fabric.** The operator's hypothesis was right. The Mando package's cache
  (34 GB, 2026-09-17) was not rebuilt and is still suspect. My earlier statement that the fault "looks
  specific to the 4-chip configuration on this machine" was wrong. The first-boot known-answer check is
  what exposed it. Server stopped and lease released afterwards; all four chips FREE.
- **16:50Z to 17:14Z run 3, stage 2 with the new `weights-swap-check` skill (merged: weights-only path,
  suite 1256 passed, 1 skipped).** A fourth recovery worked (coder ready in 121 s, canary identical again).
  The prepare step took 12.5 minutes and produced `serve_and_compare.py`, a model directory, a run-script
  copy and `hw_test.json` (deadline 2400 s), all following the skill: its own model-dir, a fresh tensor
  cache under `orchard-runs/cache/hemmingway-1`, port 8100. **First hardware test run by the harness
  itself, 17:05:03Z.** The supervisor found a free board (board 1) and did not park the coder, took a
  test lease on chips 2 and 3, and started the test. The test crashed after 0.36 s:
  `AttributeError: 'Popen' object has no attribute 'pgid'`. **That bug was in my skill text**, which said
  `os.killpg(proc.pgid, ...)`. The agent copied it. The crash happened after the script had started the
  server in its own session, so a vLLM server was left running on board 1 and holding the chips.
  I stopped that process group by hand (the harness's own stray child), after which the supervisor released
  the test lease (17:05:45Z). The skill is corrected (`proc.pid`, and start the `try` right after `Popen`).
- **17:14:44Z the watchdog fired on a real agent for the first time.** The stage 2 finish step (36 turns)
  had wandered into tt-metal firmware cache directories after the crash and then repeated the same
  `ls` three times with identical output. The `repeated_tool_call` detector fired, the ladder sent a nudge
  (`retry` entry) and then escalated, and the stage machine started a fresh prepare attempt at 17:14:45Z
  with the corrected skill. This is the plan 3 watchdog working on real model behaviour, not a
  replay. The earlier failed script's lesson for the harness: a test script that starts a server must
  stop it on every exit path, which the skill now says first.
- **17:14Z to 17:27Z run 3, stage 2 prepare failed after escalation: 60 turns, no file written.** The new
  attempt spent 116 tool calls, 91 of them `grep`, on a side question (whether `VLLM_RPC_TIMEOUT` is set
  in vLLM) and never called `write_file`. It alternated two commands each turn for the last five turns.
  The watchdog did not fire, because its detector needs the same single call three times in a row and
  these alternated. The run paused ("stage 2 failed after escalation").
  Two causes. (1) The skill asked the model to write a 400-line script from a description, which is too
  much for a local 27B model that was exploring instead of writing. (2) Two watchdog gaps: it does not see
  a step that never writes a file, and it does not see a repeating pattern of calls. Two fixes are in
  progress: tested template scripts the agent copies and configures with a small JSON file (with tests that
  run the template against a fake server and check the server is gone after every failure, including the
  `proc.pgid` bug), and two detectors (turns without any file written; the same per-turn call pattern three
  turns in a row). The first attempt's script was reviewed and was nearly right: only my `pgid` error.
  So far, in this run, the model has shown a pattern: given a task with a clear path it works; given an
  open search it explores without writing. The skills need to be closer to scripts than to guidance.
- **17:45Z CORRECTION: the hand prototype measured the base model, not Hemmingway-1.** The implementer of
  the swap templates noted that the bundle's `run.sh` exports `HF_MODEL=Qwen/Qwen3.8-27B`. I read the TT
  runtime (`tt/qwen36_vllm.py:227-235`): it resolves the weights directory as `MODEL_WEIGHTS_DIR`, then
  `HF_MODEL`, then the config path, and a hub id is resolved to the local Hugging Face snapshot. So with
  `HF_MODEL` left at the base model, my prototype (and the 34 GB tensor cache it built) used the **base**
  Qwen3.8-27B weights. `--model` only gave vLLM the config and tokenizer. The earlier log entries that say
  the chip served Hemmingway-1 weights, and the 25 of 32 (78%) teacher-forced agreement "with Hemmingway's
  CPU reference", are **wrong as stated**: they show the base model on the chip agreeing 78% with
  Hemmingway's CPU decode, which says only that the two models are close. The recipe's other findings
  (architecture registration, the base-shaped model directory, the empty cache, no logprobs) stand.
  The same mistake is in the templates the implementer just built. Next: re-run the prototype with
  `MODEL_WEIGHTS_DIR` set to the model directory and a fresh cache, check the server log for the weights
  path, and compare the chip's output with the earlier base run. Then fix the templates.
- **17:55Z to 18:20Z corrected prototype: the chip now serves Hemmingway-1 weights, measured.** I re-ran the
  hand prototype on board 1 with `MODEL_WEIGHTS_DIR` set to the model directory and a fresh cache (the
  base-weights cache was renamed aside, not deleted). Same prompt and the same 32 reference tokens.
  Free-run text: "Let me work through this carefully. The user wants a text message to their landlord
  about a broken boiler. This is a short, practical piece of writing". The CPU reference text is
  "Let me work through this carefully. The user wants me to write a text message to their landlord about
  a broken boiler. This is a practical, real-world". The earlier (base-weights) chip text was "We need
  to respond to user: ...". **Teacher-forced top-1 agreement with the Hemmingway CPU reference: 30 of 32
  (0.94), against 25 of 32 (0.78) with base weights.** The two misses are near-synonyms (" me"/" a",
  " practical"/" short"). One prompt, 32 tokens: a thin margin, but both the text and the agreement moved the
  way correct weights should. This is the first measurement that actually concerns Hemmingway-1 on the chip.
  It also shows the gate's 0.6 floor was too low: the base model passed it. The floor is being raised to
  0.85 (a choice from two measurements).
  The lease was released with a reset afterwards. While my stale lease waited, a `tt-smi -r` started by the
  systemd user manager reset the board (probably the gozer reconcile timer); `gozer release` refused to
  run while it held the chip and I did not force it.
- **18:20Z to 19:00Z merged: swap templates, two watchdog detectors, weights path, gate 0.85 (suite 1308 passed,
  1 skipped).** Details in the commits: `serve_and_compare.py` and `prepare_swap.py` are tested against a fake
  server; the server's environment carries `MODEL_WEIGHTS_DIR` and `HF_MODEL` set to the model directory;
  a cache directory that is not empty must hold a marker naming the model or the script refuses; the server
  is stopped on every exit path (a test injects an error after the server is up). `NoFileWritten` (20 turns
  without a written file) and `TurnRepeat` (the same set of calls in 3 turns in a row) are wired into the
  supervisor. The gate floor for the chip-versus-CPU agreement is 0.85, from two single-prompt
  measurements (0.78 base weights, 0.94 correct weights): a thin margin, flagged as a choice.
- **18:45Z to 19:00Z agent model test on the 4-chip server.** The 4-chip plain package (now working after the
  cache rebuild) was asked to continue the failing stage 1 turn in six modes (same 14 messages, 8,000-token
  cap). Thinking on with sampling (temperature 0.6): 3 of 3 trials hit the cap with no tool call. Thinking on
  greedy: hit the cap. **Thinking off, greedy: a correct tool call in 120 tokens (6 s).** Thinking off with
  sampling: a tool call but 8,000 tokens. So thinking mode fails on this prompt with sampling too: my earlier
  explanation (greedy decoding is the cause) was too narrow. Thinking off with greedy decoding is the best
  mode on both servers, so the agents stay on thinking off and the 2-chip coder. One prompt and one replayed
  turn: an indication, not a rate. Server stopped; the lease ended with a reset (gozer status was unreadable
  for about a minute while all four chips reset); all four chips FREE.
- **19:09Z to 19:41Z run 3, stage 2 with the tested templates: the prepare step took 3.5 minutes** (19:11:36Z to
  19:15:04Z, against 12 to 30 minutes before), and the supervisor again picked the free board and started the
  harness-written hardware test at 19:15:04Z on board 1 (chips 2 and 3). **The test ran 26 minutes and exited
  with code 4: the server did not become healthy within the template's 1500 s.** The script stopped the
  server cleanly (the new `finally` worked), the supervisor released the test lease at 19:41:47Z, and the
  finish step started. Cause: the log shows the server still compiling kernels (SDPA and GDN warmups).
  The harness test runs with the run's own home directory, so its kernel compile cache starts cold; my hand
  prototypes booted in 5 minutes because their compile cache was already warm. The spec's "cold first boot
  about 30 minutes" was right and my template's timeout was not. The weight conversion (34 GB tensor
  cache, with the model marker) and part of the compile cache are now on disk, so the next attempt should
  boot much faster. Timeouts raised to 3300 s (health) and 3600 s (deadline), pinned by tests.
- **19:44Z to 19:48Z the same-conversation gate feedback misfired after a failed hardware test.** The finish step
  wrote `result.json` with `serves: false`, which is the honest record of the 26-minute timeout. The gate
  failed on that, so the new continuation (added at 08:10 to help with malformed files) told the agent to make
  `serves` true. It cannot do that honestly. It explored for 20 turns without writing a file and the new
  `no_file_written` watchdog paused the run at 19:48:37Z, which is the watchdog doing its job. The
  continuation should not fire when the hardware test itself failed: the stage should end and the next
  attempt should run the test again. Fix in progress. The run is paused; the supervisor and the 2-chip coder
  are idle.
- **20:16Z to 20:35Z run 3, stage 2: the harness's own hardware test succeeded.** After merging the no-feedback
  fix (suite 1326 passed, 1 skipped) I restarted the supervisor and resumed. It wrote "not resuming from a
  failed hardware test", moved the failed attempt aside, and ran a fresh prepare (about 3 minutes) and a fresh
  hardware test. The coder came back on board 1 this time; the test ran on board 0 (chips 0 and 1) from
  20:22:01Z with a 3600 s deadline. **Result, from the test's own evidence:** exit code 0 in 739 s;
  `serves: true`; server ready in 728 s; `coherent: true`; teacher-forced top-1 agreement with the stage 1
  CPU reference **30 of 32 = 0.9375**; free-run text "Let me work through this carefully. The user wants a
  text message to their landlord about a broken boiler. This is a short, practical piece of writing. The".
  The two mismatches are at the same positions and tokens as my hand prototype (position 10 `me`/`a`,
  position 27 `practical`/`short`), so the harness's independent run reproduced the hand measurement. The
  evidence records `MODEL_WEIGHTS_DIR` and `HF_MODEL` both set to the stage's model directory, so these are
  Hemmingway-1's weights. **This is the first time the harness itself has run Hemmingway-1 on TT hardware.**
  It still needs the finish step to write `result.json` and the gate to pass.
- **20:35:27Z STAGE 2 PASSED on the harness's own result.** The finish step wrote `result.json` from the
  test's evidence in 25 seconds and the gate passed: `serves` true, `coherent` true, `n_tokens` 32,
  `top1_agreement` 0.9375 (bar 0.85), `server_ready_s` 728.1. **Stages passed by the harness so far: 0, 1,
  2.** Stage 3 was skipped as designed (weights-only path: stage 2 covers the full model).
- **20:35:27Z to 20:43Z stage 4 started with the old code and flailed, so I paused the run.** The supervisor
  process still had the generic `mesh-shrink` skill for stage 4 (the multi-chip plan was not yet built). The
  agent made 20 turns without writing a file by 20:39, and the `no_file_written` detector fired. I sent
  `pause` at 20:43:20Z. The multi-chip plan (12 tasks, 90 new tests, verified by its writer in a scratch
  clone) is ready; executing it next. The run stays paused, with the coder idle on board 1.
