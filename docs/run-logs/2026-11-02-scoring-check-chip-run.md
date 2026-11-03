# Scoring check, first chip run: Qwen3-Reranker-4B (issue 375)

Date: 2026-11-02. Issue 375, a follow-up to issue 358 (PR 383). Design note: `docs/superpowers/specs/2026-11-02-scoring-model-check-design.md`
(section 9 repeats the table below). Evidence files: `docs/run-logs/evidence/2026-11-02-scoring-check-chip-run/`.

## Result

The gate (`orchard.scoring_check.check`) passed on a real chip. A person ran the check with `scripts/scoring_check/run_gate.py`. The
supervisor did not. The scores came from the public bundle's server. No coder wrote the port.

| number | this run | the 4B bundle's table | bound in force |
|---|---|---|---|
| pairs, groups | 32, 8 | 188, 33 | at least 24, 6 |
| Pearson r | 0.99940 | 0.9997 | at least 0.998 |
| mean absolute difference (logits) | 0.1889 | 0.140 | at most 0.25 |
| max absolute difference (logits) | 0.5430 | 0.64 | at most 1.0 |
| top-1 document the same per group | 8 of 8 | 33 of 33 | at least 0.95 |
| same order for pairs the reference separates by 0.5 | 43 of 43 | 100% of 449 | at least 0.98 |

The sets differ in size and content (the bundle's set has documents of 2K to 8K tokens; ours has 8 documents of about 1,400 tokens and 24
short ones), so the two columns are not a like-for-like comparison. They are on the same scale and the same recipe: the model card's prompt,
logit(yes) minus logit(no), a Hugging Face run on the same token ids.

Other numbers from the run:

- Run 2 (a new run directory, the same server): the 32 chip scores are bit-identical to run 1. The numbers above are the same.
- Against the bf16 CPU reference instead of fp32: Pearson 0.99938, mean 0.2023, max 0.4741, top-1 8 of 8, order 43 of 43. The gate passed.
- The two CPU references differ from each other (bf16 against fp32): Pearson 0.99991, mean 0.0847, max 0.3647. That is the part of the
  chip difference that rounding in bf16 alone explains. The rest comes from the chip's bf16 activations and bfp8 attention weights.
- By kind of pair (chip against fp32, mean and max): short 0.148 and 0.387; multilingual 0.219 and 0.329; code 0.130 and 0.265; long 0.258
  and 0.543. The long documents carry the largest differences.
- Smallest top-1 margin in the reference: 7.34 logits. Smallest separated gap: 0.66 logits (three of the 43 pairs are under 1.1).

## Do the bounds hold?

They held for the policy the bundle ships (bf16 activations, bfp8 attention, bf16 MLP weights). Findings:

1. No bound was too tight for that build. The nearest is the mean difference: 0.189 against 0.25.
2. Whether a bound is too loose cannot be shown from one passing build. No other weight policy was run, and no failing run was made on the
   chip. (The CPU tests in `tests/test_scoring_check_template.py` show the gate refusing a noisy port, a reversed port and a constant one.)
3. On this 32-pair set the top-1 floor (0.95 of 8 groups) and the order floor (0.98 of 43 pairs) allow no flip, because 7 of 8 is 0.875 and
   42 of 43 is 0.977. The top-1 margins are at least 7 logits, so a top-1 flip is not close. The order floor is closer: three separated
   pairs have a gap under 1.1 logits, so a build whose mean difference still passes could flip one of them and fail the gate. The bundle's
   own table has 449 separated pairs, where 0.98 allows 8 flips. Changing the floor to 0.97 (one flip of 43) is an operator choice. The
   defaults are unchanged in this PR.
4. The mean difference here is 1.35 times the bundle's 0.140. At that ratio the bundle's unselected bfp8 policy (0.217 on its own set)
   would give about 0.29 on this set and fail the 0.25 ceiling. This is a prediction. It was not run.

## What ran, in order

Everything is under `~/orchard-runs/scoring-chip-run/` (outside the repo). Weights: Qwen/Qwen3-Reranker-4B at revision
22e683669bc0f0bd69640a1354a6d0aebcfeede5 (the revision the bundle pins), downloaded with `hf download`, about 8 GB.

1. Stage 1 reference on the CPU, in the reference interpreter (torch 2.14.1+cpu, transformers 5.19.0, tokenizers 0.23.2):

       python3 -m orchard.scoring_reference rerank --model-dir <snapshot> --out reference-fp32 --dtype fp32
       python3 -m orchard.scoring_reference rerank --model-dir <snapshot> --out reference-bf16 --dtype bf16

   Time: 1,765 s for fp32 and 914 s for bf16, on a 16-core host whose load average was about 90 from other jobs, so the times are an upper
   bound. The pair file has sha256 `94fc9237ee46ae68140c5c3ee431dfb1fb12f33c0774be85fab74c9df1f3218a` (32 pairs, pair set
   `scoring-pairs/1`) and is identical for both precisions. The fp32 reference file has sha256
   `af10f402cdb8c9aab53a535c29e233881123d2a37a703e69a7e56b98e0b830e4`. The reference scores are in `reference-fp32.json`.
2. Lease: `gozer acquire --chips 1 --who claude:hw2-scoring --owner-pid <a long-lived pid> --expect 3h`. gozer granted a whole board
   (chips 0000:03:00.0 and 0000:04:00.0, devices 2 and 3). Released with `gozer release`, which reset both chips. One board was used and
   the other board belonged to another session.
3. Server: the container `tt-model serve raahemnabeel/qwen3-reranker-4b --profile p150 --print` prints (bundle revision `e5c8e958`, image
   `tt-model/qwen3-reranker-4b:aab813e84ef3`). I ran that `docker run` command with four changes: a container name of my own, only
   `/dev/tenstorrent/2` passed in, the port published on `127.0.0.1:20000` only, and `HF_TOKEN` not passed (the weights were already in
   the cache and `HF_HUB_OFFLINE=1`). First boot: build 27 s, warm-up 283 s (kernel compile and trace capture). The server reported
   `ready`, one die, `max_tokens` 8192.
4. Port module: `scripts/scoring_check/port_model.py`. It posts to `/v1/score` with `raw_scores: true` (the bundle's `/v1/score` returns
   `raw_score`, the logit difference) and refuses a request whose rebuilt prompt has other ids, and a reply whose token count differs
   from the ids sent. It was piloted on 8 pairs first: the token-count check held against the real server.
5. The check, with the template's device on the other chip of the board (the server holds chip 0000:03:00.0, the template opens
   0000:04:00.0, so the two do not collide):

       python3 scripts/scoring_check/run_gate.py setup --run-dir run-fp32 --reference-dir reference-fp32 --snapshot <snapshot>
       TT_MESH_GRAPH_DESC_PATH=<tt-metal>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
         python3 scripts/scoring_check/run_gate.py run --run-dir run-fp32 --chips 0000:04:00.0 --server-url http://127.0.0.1:20000

   The template ran `python3 -P -s stages/2/scoring_check.py`. The mesh descriptor variable is needed because a single chip of a p300
   board is a custom cluster to tt-metal. Template sha256 `ab6f491724f103afb06eedfd84e963feded9b7923fe568b46c5fedbe36e44499`, port
   module sha256 `4c696b1abf2e16cfbd28056cf2482d74373feb03f039f6289314f37057add76c` (the file in the repo is the same), config bounds: the defaults. Test time 11.6 s. Record sha256 (run 1)
   `202788783541df39f216a65fd94be2be90e1b1fdda63fe185e8e30ed10a618b5`. Gate: no reasons.
6. Runs 2 and 3: the same command in new run directories (run 3 with the bf16 reference). Server totals at the end: 26 requests, 104 pairs
   (three runs of 32 and the pilot's 8), 0 failed, 0 rejected.
7. Stopped the container, released the lease, killed the lease-holder process by its pid.

## What the run does not show

- The model did not run in the check's process. The port forwards to the bundle's server. The template's source check that a port mentions
  `ttnn` passed on a comment in `port_model.py` that says so. That check cannot tell a port that computes from one that forwards. In this
  run forwarding is the design, and this log and the module's docstring say which process computed.
- `record.chips` names the chip the template opened (0000:04:00.0). The scores were computed on the server's chip (0000:03:00.0) of the
  same leased board. The gate compares `record.chips` with the chips in the marker, and both name 0000:04:00.0. `load_s` (0.85 s) is the
  time to read the tokenizer and check the server. It is not the model load (the server's was 32.6 s plus 250 s of trace capture).
- The supervisor did not run the check: no stage machine wiring exists (issue 378). `run_gate.py` wrote `test-result.json` the way
  `tests/scoring_check_fakes.py` does. A port written by the coder was not tested.
- Only the weight policy the bundle ships was run. Only one chip of one board. No batch other than a group at a time.
- The model card's probability example (item 1 of the issue) was not run. The comparison is with the bundle's table.
- No embedding model (issue 376) and no vision pairs (issue 377). The bundle's retrieval benchmark rows (issue 379) were not run.
- The default interpreter (system python3, with `ttnn` and `tokenizers`) ran the template and the port. The CPU reference needs the
  reference interpreter.
- After review, `run_gate.py` gained two fixes (it stops the template's whole process group on a timeout, and it writes absolute paths into the config). They touch the helper only. The numbers above are from before them, and no new chip run was made.
