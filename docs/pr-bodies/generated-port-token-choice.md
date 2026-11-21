Follow-up to PR 787. The vocabulary-per-die check was recorded as `skipped` for the generated Aion port, although the hazard scan had flagged its vocabulary (200,029 columns over two dies). This PR decides from the code whether the hazard can apply to a generated port, and records the answer.

## Answer: the token is chosen on the host, over the whole row
A generated port picks nothing on a die. The evidence, in the final tree:

- The generated `_forward` ends in `out.reshape(-1, out.shape[-1])[n - 1, :VOCAB]` (`orchard/port_template.py` line 2273). The cached-decode path returns `[0, :VOCAB]` (line 2325). Both return the whole vocabulary row as a host tensor.
- Weights are uploaded with `ReplicateTensorToMesh` (lines 2050 and 2057), so each die holds the whole vocabulary. `_to_host` keeps the first device's copy (`[:1]`, line 2062).
- The check's worker answers `prefill` and `decode` with `argmax(out)` over that row on the host (`port_check.py` lines 384, 1002, 1010).
- The stage 5 server (`Engine.run`, `port_serve.py` lines 129 and 133) asks the worker for `prefill` and `decode` and takes the worker's id. The file has no `sample` op, no temperature and no top-k (0 matches).

No per-die column index exists on that path, so a sampler cannot wrap one. The outcome chosen is therefore `not-applicable` with the reason in the record. A generated `sample` hook would be one that nothing serves.

## Change
- The generator writes one line, `TOKEN_CHOICE = "host-row"`. The generated text without it is byte for byte the text of before (pinned by hash, for the standard small decoder and the tiny Qwen4 model).
- For a port with no `sample`, the worker reports the declaration (only when it is the exact string) and the width of the last row the port returned.
- The check records `not-applicable` when the port declares `host-row` and the row had exactly the vocabulary's width. The block carries `token_choice`, `row_length` and a fixed sentence. A port with no declaration (every hand-written port), another value, or a narrower row stays `skipped`; when a declaration was made the reason gives the width. A port that defines `sample` is asked, as before.
- The gate accepts `not-applicable` only when the declaration, the width (against the snapshot's own vocabulary), the sentence and the empty case list all match.
- Shared constant and sentence: `orchard/hazards.py`. The template keeps literal copies (it runs where `orchard` cannot be imported) and a test keeps them equal. Skill text (`full-port-check.md`) and the template's interface text describe the declaration.

## Evidence
- 38 new tests: 32 in `tests/test_port_check_vocab_not_applicable.py`, 6 in `tests/test_port_template_token_choice.py`. They were written first. In the first run against the unchanged code, 20 of the 34 tests then in the two files failed; the 14 that already passed are the cases that must not change (hand-written ports stay `skipped`, the server wiring, the host argmax). Four tests were added after that run (the narrow-row, declaration and cached-path cases).
- 22 mutations, one per guard, each turns a test red: template (any width, no declaration needed, width never recorded, decode does not record it, any object passed to the pipe, array width, sentence, constant, no width in the skipped reason), gate (declaration, width, sentence, cases, branch, status message), generator (constant absent, another value, a row one entry short on the uncached and on the cached path), shared constants, and a server that asks for a sample.
- A real defect found by the pin: my first edit of the generator dropped a space on the `LAYER_VARIANT` line. The pin that holds the text apart from the constant line caught it.
- A guard removed: the worker once dropped a declaration over 40 characters. No test could turn it red (a 5-million-character declaration is handled the same without it), so it was removed. The pass-through is now only an exact string, which a non-JSON constant (`object()`, bytes) does turn red.
- Targeted runs: 1,715 passed (generator, port-check, full-port, thin-port, hazard and skill tests); 296 passed in the ratchets, paths, README and the four supervisor stand-in files that a missed attribute broke once; `version.py check` passes.
- No chip was used. The full suite was not run (a timing run was in progress on this host).

## Not shown
- Whether the lm_head output above column 65,535 is right for a generated port. The fidelity block covers the reference's tokens only; nothing plants a maximum there. A comparison of the port's row with the reference's row at those columns would, and would need the reference logits.
- A hand-written port can declare `host-row` and return a whole row; then the claim is true for that port too, since the server never calls a hook.

## Observation outside this PR
The served path never calls a `sample` hook for any port. For a hand-written port that defines one, PR 787's check exercises a function the server does not run. If a port should sample on a die, the worker's `prefill` and `decode` would have to use its hook for the token, and the server with it.

## Checklist (docs/pr-hardening-checklist.md)
1. Wrong type or shape in input: the worker passes the declaration only as an exact `str` equal to the constant; the gate needs `type(width) is int` and the exact declaration; tests cover bool, string, `None`, `object()` and bytes.
2. Files that something else controls: no new file read. The gate's snapshot vocabulary comes through the existing `_snapshot_vocab`.
3. Check, then use: no new check-then-use.
4. Evidence an agent can write: the port writes the declaration, so it is not enough alone. The width comes from the supervisor's own worker, and the gate compares it with the snapshot's vocabulary and with the fixed sentence.
5. A verdict stronger than its evidence: `not-applicable` needs the declaration and an observed full-width row, and the wiring test fails if the server ever asks for a sample. Limits are under Not shown.
6. The check looks at a different object than the action uses: this is the point of the PR. The hook the check exercised was not what the server runs; the new test pins what the server asks the worker.
7. A parser or matcher too broad or narrow: equality on an exact string and an exact integer only.
8. Failure paths that leave state behind: none added.
9. Numbers and sizes: `row_length` is a whole number compared for equality with the snapshot's vocabulary; no float is read.
10. Secrets, paths and host names: none added (the paths test passes).
11. Tests that do not test the claim: the mutation table above; the hash pins; no clock decides a result; a test that could not fail was removed with the guard it covered.
12. Housekeeping: change fragment `changes/generated-port-token-choice.toml`, dated entry at the end of `docs/project-log.md`, skill text updated, no new flag, `scripts/version.py check` passes.
