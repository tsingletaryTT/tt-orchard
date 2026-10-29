# Kernel probe report

Probe id: 20261029T153853Z-22a6bd00. Coder model: Qwen/Qwen3-Coder-Next.

Temperature 0 asks the server for greedy decoding. Attempts then often repeat each other (the report counts distinct kernels) but are not guaranteed to be identical. Attempts at temperature above 0 with different seeds sample independently, if the server honours the temperature. The report gives counts and no rates. The inputs are synthetic: seeded random values plus a heavy-tailed set with a few scaled outliers. A kernel that passes them can still fail on real activations.

The simulator promotes bf16 to f32 (tt-lang-sim's default) and the references compute in f32, so on a simple op a correct kernel shows a max_abs_err of 0.0 and a PCC of 1.0. Rounding on hardware is coarser (bf16) than the simulator's f32.

The wall budget (900 s) is checked between turns, so an attempt can overrun it by up to one tool timeout (420 s at most, for the kernel check).

Sampling: temperature 0.7 (5 attempts each), top_p the server's default. No seed is sent at temperature 0. Above 0, attempt k sends seed 2026120100 + k (base from --seed; send the same requests again with --seed 2026120100).

## Coder decode speed

Coder decode speed, measured just before each attempt (against the best seen for this server):

- rmsnorm attempt 1: 107.4 tok/s (ok)
- rmsnorm attempt 2: 104.9 tok/s (ok)
- rmsnorm attempt 3: 100.9 tok/s (ok)
- rmsnorm attempt 4: 105.9 tok/s (ok)
- rmsnorm attempt 5: 105.3 tok/s (ok)

## tt-lang

- simulator version: 1.1.10+gfec38c6
- source SHA: fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0
- diagnostics patch applied: yes (a local patch of the tt-lang-sim wheel; patch sha256 e3964f222cc5dc253157f97645a914f8751ab613f0fa82eb334e97046da8f84f)

## Skill sources

- tt-lang source (<home>/orchard-src/tt-lang): current; local fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0, upstream fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0. HEAD fec38c6de is upstream main (tenstorrent/tt-lang).
- tt-lang skill (<home>/.claude/skills/ttm-tt-lang): current; local 30dea84814123d303f7ea0a6710c266ead52e0c0cc04fbdaffa64a4f09a14f56, upstream 30dea84814123d303f7ea0a6710c266ead52e0c0cc04fbdaffa64a4f09a14f56. Every file matches tenstorrent/tt-metal branch agentic-research/fast-models-fast, .agents/skills/tt-lang.
  - warning: the skill documents tt-lang specification 0.12; the tt-lang checkout documents 0.20. Where they differ (for example `ttl.kernel` became `ttl.operation` in 0.13), the checkout's specification and examples are the ones the simulator runs.

## Reference files

Each attempt's reference/ directory held these files (tt-lang skill directory: <home>/.claude/skills/ttm-tt-lang):

- reference/guide.md: from the tt-lang checkout, generated from docs/sphinx/overview.md, docs/sphinx/programming-guide.md, docs/sphinx/simulator-getting-started.md, docs/sphinx/simulator.md in <home>/orchard-src/tt-lang, sha256 894c2765e51b78615c252b104bb6f4947c4afa2c664e80a0016500b63cacbbfe
- reference/examples.md: from the tt-lang checkout, generated from docs/EXAMPLES.md, examples/README.md in <home>/orchard-src/tt-lang, sha256 4f1232b649f98a1107e59aa352bbeeef3988929b42831bd3d23c7f7da75ceb2a
- reference/TTLangSpecification.md: from the tt-lang checkout, <home>/orchard-src/tt-lang/docs/sphinx/specs/TTLangSpecification.md, sha256 257b158b7689bc6f911c0133c4f1aad9f21c996def36bf61327b666e487028c8
- reference/eltwise_add.py: from the tt-lang checkout, <home>/orchard-src/tt-lang/examples/eltwise_add.py, sha256 f32bfbc7bfa86fb1d70e55536ba19ccf31098e1edfc7d8aa571c59b11b070b26
- reference/eltwise_broadcast_reduce.py: from the tt-lang checkout, <home>/orchard-src/tt-lang/examples/eltwise_broadcast_reduce.py, sha256 bace53c837bae3ca484513b0c5fca29828cf70ad188a9ba866d0a7047ee3d743
- reference/broadcast.py: from the tt-lang checkout, <home>/orchard-src/tt-lang/examples/broadcast.py, sha256 25aab370067ac764ec5eec61bd4a741b66f2ef91b8a08a4ccf4f54fa1ca0f9d9
- reference/eltwise_add_3d.py: from the tt-lang checkout, <home>/orchard-src/tt-lang/examples/eltwise_add_3d.py, sha256 660ee7ad799fbb4c88cb8445c093d4ec21f53c7443d897c1d37a5cc39bc9b975
- reference/eltwise_pipe.py: from the tt-lang checkout, <home>/orchard-src/tt-lang/examples/eltwise_pipe.py, sha256 93faa714ca70dcdff771541e7eea9b2528df6ac7bc11f1d1fb6f72ae325329d0
- reference/eltwise_pipe_node3.py: from the tt-lang checkout, <home>/orchard-src/tt-lang/examples/eltwise_pipe_node3.py, sha256 91c7f3ed588fee328470ac4ec1404438d20683002f3054ef0ac8e2af21f65417
- reference/general_broadcast.py: from the tt-lang checkout, <home>/orchard-src/tt-lang/examples/general_broadcast.py, sha256 41eb95a48bca9d6208d09ce218c811ef576de35253ccf8caf100b36118e4ab71

Worked-kernel library mode: none.

## Reference versions

- guide.md: generated from docs/sphinx/overview.md, docs/sphinx/programming-guide.md, docs/sphinx/simulator-getting-started.md, docs/sphinx/simulator.md; from the tt-lang checkout, spec version 0.20, sha256 894c2765e51b78615c252b104bb6f4947c4afa2c664e80a0016500b63cacbbfe
- examples.md: generated from docs/EXAMPLES.md, examples/README.md; from the tt-lang checkout, spec version 0.20, sha256 4f1232b649f98a1107e59aa352bbeeef3988929b42831bd3d23c7f7da75ceb2a
- TTLangSpecification.md: from the tt-lang checkout, spec version 0.20, sha256 257b158b7689bc6f911c0133c4f1aad9f21c996def36bf61327b666e487028c8
- eltwise_add.py: from the tt-lang checkout, spec version 0.20, sha256 f32bfbc7bfa86fb1d70e55536ba19ccf31098e1edfc7d8aa571c59b11b070b26
- eltwise_broadcast_reduce.py: from the tt-lang checkout, spec version 0.20, sha256 bace53c837bae3ca484513b0c5fca29828cf70ad188a9ba866d0a7047ee3d743
- broadcast.py: from the tt-lang checkout, spec version 0.20, sha256 25aab370067ac764ec5eec61bd4a741b66f2ef91b8a08a4ccf4f54fa1ca0f9d9
- eltwise_add_3d.py: from the tt-lang checkout, spec version 0.20, sha256 660ee7ad799fbb4c88cb8445c093d4ec21f53c7443d897c1d37a5cc39bc9b975
- eltwise_pipe.py: from the tt-lang checkout, spec version 0.20, sha256 93faa714ca70dcdff771541e7eea9b2528df6ac7bc11f1d1fb6f72ae325329d0
- eltwise_pipe_node3.py: from the tt-lang checkout, spec version 0.20, sha256 91c7f3ed588fee328470ac4ec1404438d20683002f3054ef0ac8e2af21f65417
- general_broadcast.py: from the tt-lang checkout, spec version 0.20, sha256 41eb95a48bca9d6208d09ce218c811ef576de35253ccf8caf100b36118e4ab71

The guide and the specification both document 0.20.

Consistency rows the probe evaluated before it started:
- checkout: ok. <home>/orchard-src/tt-lang is at the pinned commit fec38c6de and documents specification 0.20
- simulator: ok. tt-lang-sim 1.1.10+gfec38c6, built from fec38c6, matches the pin
- checkout files: ok. every file under examples/ and docs/ that was compared is as the pinned commit has it (1 larger file(s) were not compared)
- skill: mismatch. the skill documents specification 0.12 and the checkout documents 0.20
- references: ok. every reference file documents specification 0.20
- hardware tree: todo. todo: needs a leased run. `ttl.build_info()` in the hardware venv is compared with tt-lang fec38c6de and tt-metal 0e9d200; the host tree is v0.80.0

tt-lang pin: fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0 (simulator 1.1.10+gfec38c6, specification 0.20, tt-metal ref 0e9d200db976120c129ab0deb13aa3f6d972b723).

## Bars (provisional: none has been measured)

- rmsnorm, simulator: pcc >= 0.999, rel_l2 <= 0.01, slope in [0.98, 1.02], max_abs_err <= 0.01 + 0.01 * max|expected|
- rmsnorm, hardware: pcc >= 0.99, rel_l2 <= 0.05, slope in [0.95, 1.05], max_abs_err <= 0.1 + 0.05 * max|expected|

## Hardware

smoke test: longest quiet stretch 0.0 s, longest repeat 0 lines; watcher limits 60 s and 20 lines

hw: not run. gozer queued the request because the chips were busy, so the command was queued, not run; the ticket 4b68 was cancelled. Run the probe again when `gozer status` shows free chips. The smoke test's last lines:

```
queued: ticket 4b68
```

## Hardware checks called by the coder

Hardware checks called by the coder: off (the tool was not offered, so these attempts used the simulator alone).


## rmsnorm

### temperature 0.7

| attempt | simulator | hardware | status | PCC min (sim) | PCC min (hw) | turns | wall s | category | first line | agent check | simulator detail | agent hw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | no | - | pause: watchdog: the same tool output (write_file) 3 times in a row | 0.856709 | - | 31 | 889.4 | numeric_mismatch | set seed0: pcc 0.9927718639373779, finite True, bar 0.999 | - | - | - |
| 2 | no | - | wall-clock: the attempt's wall-clock budget is spent | - | - | 32 | 907.42 | tt_lang_error | error: ValueError: Shape mismatch in binary operation: left shape (1, 1) does not match right shape (1, 8). Use broadcast() to expand operands first. | - | - | - |
| 3 | no | - | error: the reply was cut off at max_tokens twice in a row (4096 tokens each); no action was taken | 0.856709 | - | 17 | 919.05 | numeric_mismatch | set seed0: pcc 0.9927718639373779, finite True, bar 0.999 | - | - | - |
| 4 | no | - | wall-clock: the attempt's wall-clock budget is spent | - | - | 23 | 1012.52 | tt_lang_error | error: ValueError: Block-to-block copy requires byte_count | - | - | - |
| 5 | no | - | wall-clock: the attempt's wall-clock budget is spent | - | - | 33 | 972.89 | shape_mismatch | error: ValueError: Tensor shape (64, 32) does not match Block shape (2, 8) (tile counts: 2 vs 16) | - | - | - |

distinct kernels: 5 of 5 attempts

none of 5 attempts passed in the simulator
