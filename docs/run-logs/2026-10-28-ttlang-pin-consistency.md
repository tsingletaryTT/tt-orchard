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

- `orchard/ttlang_pin.py` is the one record: the tt-lang commit, the simulator's source commit and version, the diagnostics
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
| the version comparison that chooses a generated guide | 7 (a generated guide, the report, the probe's report, the one-line command) |
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

The generated guide is shorter than the skill's practical guide and has no CUDA or Triton porting material. The checkout's docs
are the version the simulator runs, so the coder reads the specification and the two .py kernels for names and signatures, as the
`kernel-probe` skill already told it to. Whether the generated guide helps or hurts pass rates is not measured here; the running
ladder used the old guide, so its results and a later probe's are not on the same reference set.

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
