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

## Review round 3 (Copilot, PR 198, head 1e21162: a modified negative example could run before checkout integrity was checked)

Round 2 let `verify` run the checkout's negative-example programs in a minimal environment, but it ran the working tree's version from
wherever the process stood. A modified program could write files before the `checkout files` check ever ran, in `kernel-probe` and in
`kernel-env`. The class: code from the checkout runs only after its integrity is settled, and it runs somewhere that cannot hurt.

- `verify` now checks first, with read-only git, that the programs and the file they need beside them (`compiler_only_add.hpp`) are
  unmodified and tracked. One that is modified or untracked makes `verify` fail with its name, with or without the operator's
  override, and nothing runs. When git cannot answer, nothing runs either. `ensure` calls `verify` last, so in `kernel-env --check`
  and in the probe (which calls `ensure` before anything else) this check is the first thing that touches the checkout's programs.
- The programs are not run from the working tree. `git show <pinned commit>:examples/...` copies them into a directory made for the
  call, and the copies run there, in the allow-listed environment, with that directory as the working directory. The directory is
  removed afterwards. With the override and no pinned commit in the checkout, the checkout's own HEAD is used. Without the override
  there is no fallback.
- Permission errors in discovery: `examples/` is listed inside one guarded block, so a parent that cannot be searched is a refusal and
  not a traceback; an entry that cannot be examined is not offered; the head-commit read and the tt-metal tree check treat an
  unsearchable directory as "no readable checkout" and "no readable tree". A test locks a checkout directory and a tree directory
  (mode 0) and calls each.

Tests include a real git repository with a stand-in simulator that records its working directory, so they show where a program ran,
that a program modified to write a file never ran, and that the checkout is clean afterwards. Each guard was removed once and a test went
red: the modified-file block (4 tests), running the working tree's version (1), the working directory (1), removing the directory (1),
the no-fallback rule (1), the unanswered-git rule (1), the `examples/` guard (1), the head-commit guard (1), the tree guard (1).

## Review round 4 (Copilot, PR 198, head b12c7f9)

- The comment on the verification fake in `tests/test_kernelprobe.py` was stale. At b12c7f9 the fake answers `git show`, `ls-tree` and `status`, and the three tests it named (two override-success tests and the clean-run report test) pass when run alone and in the full suite on that head.
- A relative `--venv-dir` (or `ORCHARD_TTLANG_VENV`) gave a relative simulator path, and the programs now run in a temporary working directory, so the executable was looked up under that directory and `verify` failed although the version check had passed. The class: every path that crosses into the temporary-directory run. The simulator and the program are named by absolute paths, a relative `PATH` entry is made absolute, and `HOME`, the config and cache directories and the token path are absolute. A test runs `verify` from a relative checkout, venv, scratch and `PATH`. Each guard was removed once and a test went red (the simulator path, the `PATH` entries, `HOME`, the program path).

## Review round 5 (Copilot, PR 198, head d02f80e: `git status` can run the checkout's own filters)

`git status` hashes modified files through the checkout's own attributes and config, so a clean filter, a textconv driver, an
fsmonitor program or a hook named in the checkout could write to the operator's tree before any check of ours rejected the file.
The class: every way we inspect the checkout must use only operations that cannot run what the checkout names.

- `git status` is gone, and so is every `git show` and `git diff`. The only git calls on the checkout read objects: `git ls-tree -r -z`
  at the pinned commit (the files and blob ids the commit has) and `git cat-file blob <pin>:path` (a file's bytes). Neither applies a
  clean or smudge filter, a textconv or diff driver, or the file-system monitor, and neither runs a hook.
- Modified, untracked and deleted files are found in Python. Each tracked file's working-tree bytes are hashed as a git blob
  (`sha1("blob <size>\0" + bytes)`, a link by its target) and compared with the commit's id. Untracked files come from a walk of
  `examples/` and `docs/` with `lstat`, no links entered, `__pycache__` directories skipped. A directory that cannot be listed is
  reported as unreadable. File modes are not compared.
- Every git call also carries `-c core.fsmonitor=false -c core.hooksPath=/dev/null -c core.attributesFile=/dev/null` and the
  environment `GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_ATTR_NOSYSTEM=1 GIT_OPTIONAL_LOCKS=0`, on top of the allow-listed
  environment. `.git/info/attributes` cannot be switched off, so it is not relied on: with only object-reading calls it has nothing to
  act on. A test fails on any call that lacks the settings or names `status`, `diff` or `show`.
- The regression test builds a repository whose own config names a clean filter, a smudge filter, a textconv driver, an fsmonitor
  program and a hooks directory, and whose `.gitattributes` and `info/attributes` apply them, each writing a marker file. A control test
  shows that `git status` in that repository does write the marker. The marker never appears when the code inspects the checkout
  (`git_state`, `assemble`, the consistency check and `verify`, with and without the override).
- `verify` also takes the commit's file list from `ls-tree`, so a program that the commit does not have is refused, and the
  consistency check's `checkout files` row reports deleted files as well. On this machine the check is clean and takes under a second.

Each guard was removed once and a test went red: reintroducing a `git status` call (3 tests, including the marker), the hooks setting,
the global-config setting, `cat-file` replaced by `show`, the content comparison, the untracked rule, the deleted rule, and the
`__pycache__` skip.

## Threat model (written in round 6, after six rounds on one class)

The tt-lang checkout is owned by the operator and installed at the pin by the harness (`tt-orchard kernel-env`). The checks in this
change cover three cases: an accidental edit, an untracked file left in the checkout, and a checkout that has configured git (a
filter, a textconv driver, an fsmonitor program, hooks, a replace ref) or that contains links or very large files. In those cases
nothing from the checkout is run or read unless the pinned commit has it unchanged, and `verify()` runs pinned copies from `git cat-file
blob` in a directory made for the call, with an allow-listed environment. The change does not claim to defend against someone who can
already write to the operator's files: that person has easier ways to the same end. The same text is in the docstring of
`orchard/ttlang_reference.py`.

## Review round 6 (Copilot, PR 198, head 4d3ed73)

- The High finding (clean filters) is the previous round's, listed again. At this head the regression test with the hostile
  repository passes, and its control shows `git status` writes the marker there. The marker never appears.
- Replace refs. A local `refs/replace/<pin>` made `ls-tree` and `cat-file` read another commit as the pinned one. Every git call now sets
  `GIT_NO_REPLACE_OBJECTS=1` and `-c core.useReplaceRefs=false`. The test builds a repository with a replace ref for the pin,
  shows that plain git reads the other commit, and shows that verification still sees the pinned one. Each setting is enough alone, and
  with both removed the test goes red.
- Links at the start. `os.walk(followlinks=False)` still follows the directory it starts in. A checkout root that is a link,
  or `examples/` or `docs/` that is a link, is now refused before any walk or hash (state `link`), and `verify` and the reference
  assembly refuse on it.
- Sizes. A file above 4 MiB is not opened and is reported `unchecked` (not a difference, unless it is a program `verify` would run); the
  files hashed together stop at 64 MiB; a walk stops at 20000 entries and says so. Files are hashed 1 MiB at a time after the
  blob header, and the size is read again from the open descriptor, so a file that grew after the size check is not read.

### Sweep for the class "things git or the filesystem can redirect"

| What can redirect | Where it was checked | Result |
| --- | --- | --- |
| Environment (`GIT_DIR`, `GIT_WORK_TREE`, `GIT_CONFIG_*`, hook and filter variables) | every git call in `ttlang_reference`, `ttlang_env.verify`, `head_sha`, the tt-metal tree reads in `ttlang_consistency`; `skillsfresh.ReadOnlyRunner` | fixed: the first group uses an allow-listed environment; the runner now removes the variables that point git elsewhere. Found in this sweep: `head_sha` and the tree reads passed the operator's whole environment; the runner did too |
| Repository config and hooks | same calls | fixed earlier and kept: `core.fsmonitor`, `core.hooksPath`, `core.attributesFile`, `core.useReplaceRefs`, and no system or user config |
| Attributes, filters, textconv | `git status` in `skillsfresh.check_clone` | found in this sweep and fixed: the freshness check's dirty test used `git status`; it now reads the tree with `ls-tree` and compares bytes in Python, and `status` and `diff` are refused by the read-only runner. A real-repository test with filter, textconv, fsmonitor and hooks shows the marker is never written |
| Replace refs, grafts, alternates | all git calls | replace refs off; the pin is a full 40-character id, so a branch or tag cannot stand in for it; object ids are content hashes, so an alternates file cannot substitute content |
| Links | checkout root, `examples/`, `docs/`, directories on the way to a reference file, the file itself, entries in a walk, the skill directory | fixed: root and top directories refused; `_read_reference` refuses a linked intermediate directory and opens the file with `O_NOFOLLOW`; walks never enter a linked directory; the skill directory may itself be a link because the operator names it |
| Sizes | reference files, the scan, generated guides | caps everywhere a file is read: 1 MiB per reference file and per generated file, 4 MiB per scanned file, 64 MiB total, 20000 entries, chunked hashing |
| Relative paths | the run in the temporary directory | fixed in round 4 and kept: the simulator, program, `PATH` entries and `HOME` are absolute |
| Build-time git calls (`init`, `remote add`, `fetch`, `checkout` in `step_clone`) | `ttlang_env` | not changed. They run in a directory the harness creates, before any check, need the operator's network credentials, and are outside the cases in the threat model |
| Permissions | listing and stat calls on the checkout, the tt-metal tree | handled in round 4 and kept: an unsearchable directory is "no readable checkout" or a refusal, never a traceback |

Each guard was removed once and a test went red: the replace-ref settings (the test needs both removed to see it), the start-directory
and root link checks, the size cap before and after the open, chunked reading, the total cap, the entry cap, the needed-file
`unchecked` block, the safe environment on the head and tree reads, the replace and attribute settings in the skills runner, the
cut-walk rule in the freshness check, and the redirecting-variable removal.
