# Full port: a stage 0 to 8 chain test with fakes

Branch `stack-4-chain`, link 4 of the stack, on `stack-3-stage6`. Tests only: no change to `orchard/`.

## What it does

`tests/test_full_port_chain.py` runs a rehearsal run through every stage, 0 to 8, with the chip, the coder and the model as fakes. Each stage's own supervisor tests keep the neighbouring stages generic, so none of them can show that the stages fit together: a ledger decision one stage writes and the next reads, a seed one stage changes and another relies on, a pause left behind by a stage that has since been built. In this run stage 5 serves the port and replays the corpus and stage 6 measures quality and the benchmark, both for real on the CPU against the fake `ttnn`. The run ends ready, with no pause for an unbuilt stage. Three tests: the run goes through every stage and ends ready; each stage reads what the stage before it wrote; and with chips configured, stage 3 runs on four chips and stage 4 covers both required counts (2 and 4), so it runs no test.

## The twelve items of docs/pr-hardening-checklist.md

1. **Wrong type or shape in input.** No input is read by new code; the tests build their inputs from fixtures.
2. **Files that something else controls.** No new reader or writer in `orchard/`. The test rig writes into `tmp_path` only.
3. **Check, then use.** Not applicable: there is no new check. The run uses the checks of stages 2 to 6 as they are.
4. **Evidence that an agent can write.** The scripted coder writes only what the real coder may write; the gates judge it as they do in a real run.
5. **A verdict stronger than its evidence.** The test says what a fake run shows (the stages fit together) and nothing about speed or about a real chip.
6. **The check looks at a different object than the action uses.** The run uses the real supervisor, the real templates and the real gates; only the device, the model and the coder are fakes.
7. **A parser or matcher that is too broad or too narrow.** No new parser.
8. **Failure paths that leave state behind.** A pause anywhere fails the test, with the ledger's reason in the message.
9. **Numbers and sizes.** None added.
10. **Secrets, paths and host names.** None in the test or in this text.
11. **Tests that do not test the claim.** The chain test fails when a stage's pause is put back (the `FULL_PORT_UNBUILT` switch of the previous link) and when a stage reads a ledger decision under a name the stage before it does not write. Both were seen red before the code that fixes them was in place (they are the combination fixes of the previous link).
12. **Housekeeping.** No version fragment (no code change). A dated project-log entry is not needed for tests only; the previous link carries the log entry for the combination.

## What could not be tested without hardware

Everything that needs a chip: a real run, the park and restore of the real coder, the real chat server. The chain test is a fake run.
