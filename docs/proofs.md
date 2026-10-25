# What has been shown, and what has not

This page lists each claim the README makes about tt-orchard, the evidence for it, and where that evidence is. Every measurement came from one
QuietBox 2 (two p300c boards, four Blackhole chips). Numbers are single runs unless a row says otherwise. `CLAUDE.md` holds the dated log entry
for each item; the run logs under `docs/run-logs/` hold the longer write-ups.

## Shown on real chips

| Claim | Evidence | Where |
|---|---|---|
| A weights-only fine-tune of Qwen3.8-27B reaches `ready for operator review` from one command | Hemmingway-1, openthai, Clef, RAG-Gate-27B, bytkim, AREX-2 | `CLAUDE.md` (2026-10-03 to 2026-10-22); `docs/run-logs/2026-10-hemmingway-1.md`, `2026-10-openthai-run1.md` |
| A model with a separate task head (`weights+sidecar`) is brought up and its head is checked | Clef: backbone swap 31 of 32 tokens; head agrees with the CPU on 15 of 16 questions, hidden-state correlation at least 0.954 | `docs/run-logs/2026-10-06-sidecar-parity-prototype.md`; `CLAUDE.md` 2026-10-07 |
| A vocabulary-trimmed model is recognised by comparing rows and served | CJK-Trim: all 181,760 rows of two matrices found in the nearest model; 30 of 32 tokens on 2 and on 4 chips; supervised unattended run in 2,804 s of ledger time | `CLAUDE.md` 2026-10-24 (issue 56, PR 57) |
| A small dtype difference does not need new code | `sabi-noema`: 96 small tensors stored as F32; served, 31 of 32 tokens, eight suite prompts without a finding | `CLAUDE.md` 2026-10-24 (issue 29, PR 58) |
| A model without `mtp.*` tensors can keep the drafter by grafting the nearest model's | AREX-2 by hand: booted in 244 s, 31 of 32 tokens, decode 44 to 68 tokens/s against about 20, 0.943 against 0.947 on 300 GSM8K items | `CLAUDE.md` 2026-10-23 (issue 14); the supervisor's own trial has not run on hardware |
| Stage 2 serves a swapped model through a container package | `huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated` through a 4-chip container: 32 of 32 tokens, top-1 1.0, ready in 1,356 s | `CLAUDE.md` 2026-10-25 (issue 60, PR 66) |
| A package the harness produced boots from the Hub on a clean install | Hemmingway-1 and openthai (2026-10-04, about 35 minutes each), RAG-Gate-27B (ready in 2,841 s, reply 42), AREX-2 | `docs/run-logs/2026-10-publishing.md`; issue 24 |
| The fine-tune was compared with its base on the same server | bytkim against the base: HumanEval, MBPP, GSM8K, IFEval, BFCL; 9.8 hours of chip time, cut-off items asked again at 32,768 tokens. The fine-tune's GSM8K lead is mostly an artefact: GSM8K's stop string cuts thinking models off (log entry 2026-10-22), and the published card has not been corrected | `docs/run-logs/2026-10-20-bytkim-pi-benchmark-full.md`, `2026-10-21-bytkim-pi-benchmark-rerun.md` |
| Stage 0's class decision matches careful human labels | 42 models read from the Hub headers without downloading weights: 39 of 40 agree with three blind labelers | `docs/classify-bench/` |
| The coder can run the agent role | Qwen3-Coder-Next on one board: 50 of 50 replayed turns well formed, no repeat of 35 replays of the 27B's failures, 41.8 tokens/s at 8K | `docs/run-logs/2026-10-06-rolefit-qwen3-coder-next.md` |
| The park and restore of the coder and the pre-test reset work | The 4-chip tests of runs with the coder on one board park it and restore it (first: Clef); the 4-chip fabric timeout is not fully understood (issue 26) | `CLAUDE.md` 2026-10-07, 2026-10-13, 2026-10-18 |
| A model whose head reads rows under another attention mask can be checked | pplx-decider: 14 of 14 top-1 agree, largest probability difference 0.048, against 0.47 for a causal runtime | `docs/run-logs/2026-10-19-pplx-decider-parity.md` |

## Shown only with fakes, or by one hand-run

- The graft trial inside the supervisor (the hand experiment is above).
- The `bus` preflight row on a failed board (only a healthy box was available).
- The 4-chip stage 2's retry path (the first attempt blocked and was resumed by hand; the retry passed).

## Not shown

- Any model that needs new model code (`full-port`, stages 3, 5 and 6): issue 61.
- A run from start to end with nobody touching the machine. The latest ones needed a person for disk space or for a restart.
- A fidelity bar measured from controls: the prompt suite and the 32-token check use choices (issue 15).
- Long-context behaviour (issue 18), agent-style benchmarks (issue 21), 1-chip and 4-chip packages from the harness (issue 23).
- Any machine other than the development QuietBox 2.
- A container-only package through stage 4 (issue 68), and a quantized copy of a supported model (issue 63).

The open issues are the list of what is next: <https://github.com/tsingletaryTT/tt-orchard/issues>.
