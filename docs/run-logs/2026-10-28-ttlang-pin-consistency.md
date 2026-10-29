# One tt-lang for the simulator, the coder's reading and the hardware build (2026-10-28)

Operator question: "would we succeed better if our simulator and our kernel were shared envs? I'm willing to compromise if it
means consistency." This run records what disagreed before the change, what the change does, and the same probe preflight
before and after. No coder, board or lease was used. The probe preflight was run against a port with no server
(`--endpoint http://127.0.0.1:9/v1`), so it stops at the coder check, after every environment check has printed.

## What mismatched before

| Place | What it held | Against the pin |
| --- | --- | --- |
| tt-lang checkout | commit `fec38c6de`, documents specification 0.20 | matches the pin in `defaults.TTLANG_PIN` |
| Simulator wheel | `tt-lang-sim 1.1.10+gfec38c6`, built from that commit | matches |
| `reference/TTLangSpecification.md`, `eltwise_add.py`, `eltwise_broadcast_reduce.py` | from the checkout, specification 0.20 | matches |
| `reference/guide.md`, `reference/examples.md` | from the installed skill, which documents specification 0.12 | **differs: the coder read a language the simulator no longer runs** (`ttl.kernel` became `ttl.operation` in 0.13) |
| tt-metal tree for the hardware build | v0.80.0, no pinned ref anywhere | nothing to compare with (issue 95) |

The only sign of the guide mismatch was one warning sentence inside the skill-freshness line of the preflight. Nothing refused
on it, nothing recorded the specification version of each reference file in the probe report, and the pin lived in
`orchard/defaults.py` beside unrelated timeouts, with no record of the simulator's source or the hardware ref.

## What changed

- `orchard/ttlang_pin.py` holds the single record: the tt-lang commit, the simulator's source commit and version, the diagnostics
  patch hash, and the tt-metal ref for the hardware build. The ref is `None` ("to be pinned with issue 95"), because no tt-lang
  commit is known to build on a tt-metal newer than the host's v0.80.0. `defaults.TTLANG_*` are aliases of it. The module holds
  constants and reads no file, so it needs no hostile-input reader. The specification version is not stored: it is read from
  the checkout's own `TTLangSpecification.md` (`ttlang_consistency.checkout_spec_version`).
- `orchard/ttlang_reference.py` assembles the coder's reference files. The skill's `guide.md` and `examples.md` ship only when the
  skill documents the same specification version as the checkout. When the versions differ, both files are generated, in memory,
  from the checkout's docs (`GUIDE_DOCS`: overview, programming guide, simulator getting started, simulator; `EXAMPLES_DOCS`:
  `docs/EXAMPLES.md` and `examples/README.md`) and written into each attempt's own `reference/` directory. Each file gets a
  first-line comment that names the commit, the version and the sources. When either version cannot be read the skill's files are
  kept and the record says the version is unknown. The checkout, the skill directory, the venv and the tt-metal tree are only read.
- The probe report records, for each reference file, the source, sha256, specification version, whether it was generated and from
  which docs. A new "Reference versions" section says plainly whether the guide and the specification document the same version.
  The header also carries the pin and the specification version read from the checkout.
- `orchard/ttlang_consistency.py` and `tt-orchard ttlang-consistency` print one row per place that must agree with the pin:
  checkout, simulator, skill, references, hardware tree. A mismatch has a one-line fix. The probe runs the same check after it
  assembles the references and refuses when a blocking row mismatches (checkout or simulator off the pin, a shipped reference
  file that documents another version, a pinned tt-metal ref the tree does not match). `--allow-stale-skills` overrides it and the
  report records the override. An old skill is a mismatch row that does not block, because the probe then builds the guide from the
  checkout.
- The `kernel-probe` skill text now says the guide is built from the checkout when the versions differ.

Guards, each removed once and seen red (tests in `tests/test_ttlang_pin.py` and `tests/test_kernelprobe.py`):

| Removed | Tests that went red |
| --- | --- |
| the version comparison that chooses a generated guide | 7 (a generated guide, the report, the probe report, the command) |
| the probe's refusal on a blocking row | 2 (checkout off the pin, simulator off the pin) |
| the checkout-versus-pin comparison | 3 |
| the reference-version comparison | 1 |
| the simulator-version comparison | 2 |
| the report sentence on guide and specification versions | 2 |

## The reference files the coder would get, before and after (real checkout and skill, read only)

| File | Before | After |
| --- | --- | --- |
| `guide.md` | tt-lang skill, 34958 bytes, sha256 `c9cbabcc28515b0d...`, version not recorded (0.12) | tt-lang checkout, generated, 40934 bytes, sha256 `894c2765e51b7861...`, specification 0.20 |
| `examples.md` | tt-lang skill, 37187 bytes, sha256 `ece7b4825264f060...` (0.12) | tt-lang checkout, generated, 10999 bytes, sha256 `4f1232b649f98a11...`, specification 0.20 |
| `TTLangSpecification.md` | checkout, 92238 bytes, `257b158b7689bc6f...` | same file, specification 0.20 recorded |
| `eltwise_add.py`, `eltwise_broadcast_reduce.py` | checkout | same files, specification 0.20 recorded |
| `broadcast.py`, `eltwise_add_3d.py`, `eltwise_pipe.py`, `eltwise_pipe_node3.py`, `general_broadcast.py` | not given | checkout, full files, 6474 + 4993 + 7802 + 7694 + 7922 bytes, specification 0.20 |

Sizes of what the coder reads, in bytes:

| | Before | After |
| --- | --- | --- |
| `guide.md` | 34958 | 40934 (generated) |
| `examples.md` | 37187 | 10999 (generated) |
| `TTLangSpecification.md` | 92238 | 92238 |
| the two fixed example kernels | 12909 | 12909 |
| extra example kernels | 0 | 34885 (five files) |
| total | 177292 | 191965 |

The prose examples changed most. The skill's `examples.md` held 37187 bytes of worked examples, including porting examples from
CUDA and Triton. The generated `examples.md` is an index of 10999 bytes taken from the checkout's `docs/EXAMPLES.md` and
`examples/README.md`, and it has no porting examples. To make up for that, the generator now adds the checkout's own example programs
whose file names hold `eltwise`, `broadcast` or `reduce` (other than the two fixed ones) as full files in `reference/`, because these
are the patterns the failing kernels need. Files with `multidevice` in the name are left out, since a cross-device all-reduce is not a
single-chip kernel pattern. At most eight are added; five match at the pin.

The amount and kind of example material is a variable the pass rates could depend on. The coder now gets less prose and more
runnable kernels. Whether that helps or hurts is not measured here. The running ladder used the old guide and no extra kernels, so its
results and a later probe's are not on the same reference set. A comparison needs two probes that differ only in this.

## Probe preflight before (origin/main)

```
refused: the coder at http://127.0.0.1:9/v1 does not answer with the model 'none'. ...
ok    clone: <src> at fec38c6de
ok    venv: <venv>
ok    torch: importable
ok    tt-lang-sim: 1.1.10+gfec38c6 (built from fec38c6de)
ok    diagnostics: <venv>/lib/python3.12/site-packages/ttl/ttl/diagnostics.py (local patch of the tt-lang-sim wheel)
ok    verify: 2 negative examples print their diagnostics
ok    skills: tt-lang source (<src>): current. HEAD fec38c6de is upstream main (tenstorrent/tt-lang).
ok    skills: tt-lang skill (<skill>): current. Every file matches tenstorrent/tt-metal branch agentic-research/fast-models-fast, .agents/skills/tt-lang. Warning: the skill documents tt-lang specification 0.12; the tt-lang checkout documents 0.20. Where they differ (for example `ttl.kernel` became `ttl.operation` in 0.13), the checkout's specification and examples are the ones the simulator runs.
```

Exit 2 (no coder). Every line is "ok". The mismatch is a trailing warning on one of them, and the coder is given the 0.12 guide anyway.

## Probe preflight after (this branch)

```
refused: the coder at http://127.0.0.1:9/v1 does not answer with the model 'none'. ...
ok    clone: <src> at fec38c6de
ok    venv: <venv>
ok    torch: importable
ok    tt-lang-sim: 1.1.10+gfec38c6 (built from fec38c6de)
ok    diagnostics: <venv>/lib/python3.12/site-packages/ttl/ttl/diagnostics.py (local patch of the tt-lang-sim wheel)
ok    verify: 2 negative examples print their diagnostics
ok       checkout: <src> is at the pinned commit fec38c6de and documents specification 0.20
ok       simulator: tt-lang-sim 1.1.10+gfec38c6, built from fec38c6, matches the pin
mismatch skill: the skill documents specification 0.12 and the checkout documents 0.20
         fix: nothing to run: the probe builds guide.md and examples.md from the checkout; update the skill with `tt-warp fastmodels sync` when upstream catches up
ok       references: every reference file documents specification 0.20
todo     hardware tree: the tt-metal ref is to be pinned with issue 95; the host tree is v0.80.0
ok    skills: tt-lang source (<src>): current. ...
ok    skills: tt-lang skill (<skill>): current. ... Warning: the skill documents tt-lang specification 0.12; ...
```

`tt-orchard ttlang-consistency` prints the same five rows on its own and exits 0, because only the skill row mismatches and it
does not block. Paths are shown as placeholders here; the command prints the real ones.

## Left for the hardware build (issue 95)

- `TT_METAL_REF` is open. When a tt-lang commit builds on the tree the hardware work will use, set it in `orchard/ttlang_pin.py`
  beside the new `SHA`. The hardware-tree row then compares the tree (`git describe`, `git rev-parse`) with it and blocks the
  probe on a mismatch.
- The check that compiles the smoke kernel after a tree change (issue 95, "Needs") is not built here, because it needs a leased chip.
- Moving the pin is one file to edit, followed by `tt-orchard kernel-env` (clones and builds at the new commit) and
  `tt-orchard ttlang-consistency`.

## Review round 1 (Copilot, PR 198, three medium findings, fixed by class)

The class: a pin, a version or an override is read in one place and decided in another, and the report loses what was decided.

- The environment gate (`ttlang_env.ensure(check_only=True)`) refused an off-pin checkout or simulator before the consistency check
  ran, so `--allow-stale-skills` could not reach it. A result that only says "not the pinned tt-lang" now carries `pin_mismatch`. A check
  reports the steps after it and does not stop, and the probe accepts only those results under the override. A missing venv, torch,
  wheel or diagnostics patch still refuses, and a real build still stops at a checkout at another commit.
- The report now stores every consistency row and whether the override let a mismatch through (`header.ttlang_consistency`), and
  `skills.stale_allowed` is true whenever the override accepted a stale source, an off-pin environment result or a blocking row. The
  `skills.stale` list names each one with its detail. The "Reference versions" section prints the rows.
- A reference file whose specification version cannot be read (a skill with no versions table) gave an "ok" references row. It now
  gives "unknown", and orchard's own worked-kernel library files are left out of that comparison.

Each guard was removed once and a test went red: the override in the gate (2 tests), the check going on past a mismatch (2), the
`pin_mismatch` flag on the clone step (3), the consistency rows in the allowed list (1), the unknown branch (1), the rows in the
header (3), the library exclusion (2).

## Review round 2 (Copilot, PR 198, three findings, fixed by class)

The class: anything executed or read from the tt-lang checkout, the environment it runs in, and whether the pinned commit has it.

- Running code from an off-pin checkout. The first round let the environment check go on past a checkout at another commit, and
  that reached `verify`, which runs the checkout's negative-example programs. Nothing from a checkout that is not at the pin runs
  now unless `--allow-stale-skills` was given: without it the check stops at the mismatch, and `verify` is skipped while any
  result is off the pin (a simulator built from another commit included). With the override, and at the pin too, the examples run
  in an allow-listed environment: the same variables an agent shell gets, a `HOME` that is not created, no token or credential
  path, and a device mask that matches no chip. A test holds the allow-list equal to the agent shells'. Real `kernel-env --check`
  passes `verify` in that environment on this machine.
- A directory that cannot be listed. `examples/` that raises a permission error is a refusal that names the directory, never a
  traceback.
- Extra examples and files taken from the checkout. The checkout's `git` is asked (read only, `GIT_OPTIONAL_LOCKS=0`,
  `core.fsmonitor=false` so the checkout's own config cannot name a program for git to run) which files the pinned commit has under
  `examples/` and `docs/`, and which are modified or untracked. An extra example that the pinned commit does not have is refused
  and never given to the coder. A modified file among the files taken (the specification, the fixed examples, the docs a generated guide
  is made of, the extras) is refused. When git cannot answer, no extra example is taken.
- A new consistency row, `checkout files`, reports modified or untracked files under `examples/` and `docs/` as a blocking
  mismatch, and "unknown" when git cannot answer. I put it in the consistency check (which the probe runs and `ttlang-consistency`
  prints) and not in `skillsfresh`, which compares sources with upstream and has no view of the working tree.

Each guard was removed once and a test went red: the verify skip off the pin (1), the stop at a mismatch without the override (1),
the minimal environment (2), the override passed to the gate (1), the permission handling (1), the tracked-at-the-pin filter (1), the
rule for an unanswered git (1), the dirty-file refusal (1), the no-lock setting (1), the `checkout files` row (1).
