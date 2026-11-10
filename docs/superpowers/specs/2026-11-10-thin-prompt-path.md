# A thinner prompt path for the stage 2 coder (issue 677, Part 2)

Date: 2026-11-10. Status: design note with data from the four tiny Qwen4 audition arms, plus the two pieces that could be built and
tested on the CPU (`orchard/stub_prompts.py`). Nothing here ran on a chip; the on-chip experiment is at the end.

## What drives the coder today

`orchard/agent.py` runs a tool loop against a local OpenAI-compatible server: a system message (the run rules and the whole stage skill,
about 19,200 characters) and a task message (the run's inputs, the ledger so far and the results of earlier stages, 12,000 to 14,000
characters), then as many turns as the model takes, each with the tools `read_file` (12,000-character pages), `write_file` and `shell`.
The whole conversation is sent again every turn. A watchdog nudges at 20 turns without a write and a write-deadline hint arrives at 12.
The `qwen-code` harness is no longer used to drive the coder: `grep` finds it only in `scripts/code_video_bench` (a separate benchmark
that runs under Qwen Code), `scripts/extract_qwen_signature.py` and the transcript replay (`tests/test_replay_local.py`), which read the
old transcripts the watchdog thresholds were set from. The agent loop is the only prompt path.

## Where the tokens and the turns went (stage 2 prepare, all four arms)

Arms A and B ran the Qwen3.8-27B bundle (cp4); arms C and D ran Qwen3-Coder-Next. Read from `stages/2/log/prepare-*.jsonl` (one record
per turn with the messages sent, the reply and the server's usage counts).

| arm | turns | prompt tokens, summed over turns | completion tokens | reasoning tokens | largest prompt | repeated context share | first write at turn | files read before the first write |
|---|---|---|---|---|---|---|---|---|
| A (27B) | 33 | 1,503,207 | 14,379 | 0 | 61,878 | 95% | 15 | 12 reads (excerpts in 4 pages, facts, construction, skeleton, sheet in 2 pages, idioms, port_check.py) and 6 shells |
| B (27B) | 38 | 2,135,855 | 111,300 | 80,854 (73% of completion) | 84,052 | 91% | 16 | 10 reads (the same pages, no sheet: it used `stage_tools.py cheat`), then 10 shells paging the installed transformers source |
| C (Coder-Next) | 14 | 375,828 | 15,149 | 0 | 39,743 | 85% | 10 | 9 reads (excerpts in 4 pages, facts, construction, skeleton, sheet, idioms), no shell |
| D (Coder-Next) | 46 | 1,942,617 | 12,421 | 0 | 61,600 | 96% | 11 | 6 reads in 2 turns, then 8 shells (`head`, `cat`, `stage_tools.py` on the same files) |

"Repeated context share" is `1 - (largest prompt + all completions) / summed prompt tokens`: the fraction of the prompt tokens the server
received that were a copy of something it had already received. Without a prefix cache that fraction is work; with one it is at least
transfer and attention over the prefix.

What the context was made of (characters sent, prepare step): the system and task messages were 32,000 to 33,400 in every arm; tool
results were 84,000 (C) to 157,000 (B); the model's own text and tool arguments 42,000 (A, D) to 89,000 (B). Tool results are the largest
part, and they are the stage files the supervisor wrote: `reference_excerpts.md` (46,400 characters, 4 pages, read first by every arm),
`cheat_sheet.md` (23,900, 2 pages), `port_facts.json` (10,000), `effective_construction.txt` (7,600), `decoder_idioms.md` (5,000). Every
arm read the excerpts whole before anything else, because the skill and the task say to; arm A read `port_check.py` (the supervisor's own
check, 1,057 lines) as well. Repeated reads were paging of the same file (4 pages of the excerpts), plus one re-read of the port after
writing it (A, B) and two re-reads of the facts (D).

Reasoning: only arm B's server returned reasoning tokens, 80,854 of them, 73 percent of everything it generated; turn 6 alone spent 5,674
reasoning tokens to decide to run `stage_tools.py cheat`. The three other arms reported 0.

## What the first-write stalls and the host-only ports have in common

1. Every arm's first 9 to 16 turns were the same reading sequence, in the order the task names the files: excerpts (torch reference
   code, 46 KB) first, the sheet last or never. By the time the sheet arrived, 25,000 to 36,000 tokens of torch code were in the context.
   Arm C then wrote a 747-line port in one `write_file` whose `prefill` begins with a torch forward, and never made a matmul-family ttnn
   call (the device-ops gate failed it before the device was touched). Arm D wrote three ports of 90 lines with no ttnn call at all and
   spent its remaining 30 turns in `shell` reading the same files again; it never wrote `hw_test.json`. The write-deadline hint did not
   fire for D: it fires on "no file written", and D had written one.
2. The two 27B arms followed `decoder_idioms.md`: one `lin` helper with one `ttnn.linear` call and host glue around it (A: 1 matmul-family
   call in 356 lines, 47 `torch.` mentions; B: 1 in 448 lines, 60 `torch.` mentions). Both reached the chip and failed on plumbing, which
   the sheet's mistakes list already describes: A passed a weight as stored (`width=256 height=64`: the transpose), B reshaped a mesh
   read-back without dropping the second chip's copy (`shape [1, 58, 64] is invalid for input of size 7424`, which is 2 x 58 x 64). Each
   repair step then started a fresh context (9,500 tokens of system and task) and spent 10 to 13 turns to make one `edit`.
3. The device-ops gate message puts the list of calls found first and "what to do" (upload with `from_torch`, run each projection with
   `ttnn.linear`) at the end of a 900-character line; the rendered sheet's first section is "Tensors, layout and dtype" (13 entries) and
   "Linear and matmul" is third. The skill's device-ops paragraph is at line 84 of 200.

In short: the prompts put the reference's torch code first and the TTNN call forms last, the loop paid for the whole context on every
turn, and nothing in the loop checked a port for a reachable `ttnn.linear` before the board was taken.

## The thin path (what Part 1 and `orchard/stub_prompts.py` give)

With a generated first port (`orchard/port_template.py`) the coder's work is a list of named stubs. The thin path sends one request per
stub and splices the reply into the port:

1. `stub_request(port_text, stub, sheet=..., excerpts=..., construction_text=..., inventory=...)` builds one chat request: the rules
   (system), then the port's helper signatures, the sheet's Linear and matmul entries first and then the entries stage 0 mapped for the
   block's aten ops (with the names the sheet has not verified listed as such), the sheet's mistakes, the construction lines for the
   block's class, the reference excerpt of that one class (cut at 12,000 characters) and, last, the stub function as it stands. For
   `_A_linear_attn` of the tiny Qwen4 port the request is under 20,000 characters; the loop's first prompt was 32,000 before any file was
   read. The fixed parts come first so a prefix cache can reuse them across the stubs of one port.
2. The reply format is one code block that defines the function with the same name and arguments. `splice` accepts it only when it is
   imports and function definitions, defines the stub's name with the stub's argument names, redefines no port function, keeps no
   `# AGENT:` marker, imports none of transformers, subprocess, os, sys or a network module, and leaves a port that parses. Anything else
   is a SpliceError with a reason and the port is unchanged.
3. Between requests the supervisor runs `port_template.check_port` (reachable matmul call, every file parses, stubs left), on the CPU, in
   milliseconds. The board is leased only when no stub is left. `tests/test_stub_prompts.py` runs this loop over all 15 stubs of the tiny
   Qwen4 port with trivial replies and ends with a port the check accepts.
4. After a board run, `repair_request(port_text, function, last_check=..., layer_diff=...)` builds a request from the function the
   check's `port_frame` line named, the `last-check` report cut to 4,000 characters, the `layer-diff` report cut to 3,000, and the sheet's
   mistakes. Same reply format, same splice.

What stays out of the request: the stage skill (17 KB), the ledger and earlier results (12 KB), the full excerpts, the full sheet, the
facts file and the idioms. What the model loses: the shell. A block whose author needs a fact the request does not hold cannot look it
up; the request has to hold it, and the first version of the rules asks the model to say what is missing when it is.

The existing local coders need no change: the request is one chat completion with a system and a user message, no tools, and the
reply is a code block. Thinking models (arm B) spend their reasoning tokens on one block at a time.

## The alternative: improve the loop

The cheap fixes to the existing loop, each one line or one file:

- Put the device-ops instruction ("upload with from_torch, run every projection with ttnn.linear") first in the gate message and first in
  the skill's "Start here" (today it is at line 84).
- Reorder `ttnn_cheat_sheet.json` so "Linear and matmul" is the first section and "Tensors, layout and dtype" the second; the facts and
  excerpts already say what the shapes are.
- Make the prepare task say "write a small first version within 5 turns and run `stage_tools.py` on it". Today it says "read the
  excerpts first", and the loop then sends 46 KB of torch code before the first write when the model follows the skill.
- Give the loop the CPU check (`port_template.check_port`) as a tool or as a gate before `hw_test.json` is honoured, so a host-only port
  is refused in seconds, in the same step. Today it is refused after a park and a board reset.

These help the first version and cost nothing on the chip side. They do not change the 85 to 96 percent repeated context, the 10 to 16
turns before the first write, or the 10-turn repair steps, because those come from the loop's shape: the whole conversation every turn,
and the reading sequence the task prescribes.

## Recommendation

Use the generated port as the start of stage 2 and the thin path for the stubs, with the loop kept for one job: the finish phase
(`hw_test.json`, `handoff.json`), which is a fixed form the supervisor could also write itself. Apply the two reorderings (gate message,
sheet sections) now, because they cost nothing and help the repair prompts too. Do not spend more on the loop's reading sequence.

The decision issue 677 asks for ("how the coder is prompted to fill the stubs"): one request per stub, the stub last, the sheet before
the reference excerpt, splice guarded, CPU check between requests, board only when no stub is left. The supervisor wiring (a stage 2
mode that runs the generator, the request loop and the check) is not in this change; the pieces are tested on the CPU with fakes and
the loop over them is the test `test_every_stub_of_the_tiny_port_can_be_requested_and_spliced_in_turn`.

## The on-chip experiment

Not run here (no board in this session). Two arms on the same run directory inputs as the cp4 audition (tiny Qwen4, one board, the 27B
bundle as the coder, because its logs carry reasoning counts and it reached the chip in the audition):

- Arm E (thin path): run `python3 -m orchard.port_template generate` on the stage directory's inputs, then send one `stub_request` per
  stub to the coder endpoint (a 40-line driver: a chat completion per stub, `splice`, `check_port`), then the supervisor's check on the
  board, then one `repair_request` per failed check, at most three rounds.
- Arm F (improved loop): the existing loop with the gate message reordered, the sheet's Linear section first and the CPU check run before
  `hw_test.json` is honoured.

Measure, per arm, from the logs and the ledger: summed prompt tokens, completion and reasoning tokens, turns (requests) before the first
port that passes `check_port`, wall time to the first board run, the device-ops counts of the first board run (`device_ops` in
`evidence/port-check.json`), whether the run reached a shape error, a numerics failure or the agreement bar, and the number of board
runs. The audition numbers above are the baseline for arms A to D.
