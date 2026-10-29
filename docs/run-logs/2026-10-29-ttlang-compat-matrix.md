# tt-lang: do the simulator, the compiler and the chip agree anywhere? (2026-10-29)

Operator question: "is there any matrix where the simulator and compiler and hardware all line up in agreement for these kinds of
operations?" This run is read-only research. No device was opened, no lease taken, nothing installed, upgraded or built. Two published
wheels were downloaded into an empty scratch directory and unpacked as zip files only, so their source could be read. Nothing from them
was imported or run. `<src>` is the tt-lang clone, `<venv>` the shared hardware venv, `<metal>` the host tt-metal tree.

## Short answer

1. Upstream publishes a matrix, and it does not promise agreement. Appendix D of the tt-lang specification lists, per feature, the
   release in which the simulator and the compiler first supported it. `docs/sphinx/simulator.md` says: "The simulator typically
   supports more language features than the compiler at any given point."
2. No published release agrees on `reduce_sum`. The public 1.1.6 package contains both a simulator and a compiler, and they disagree
   inside that one wheel. The simulator requires `shape`; the compiler rejects it.
3. A pair that agrees for every operation the probe uses exists, but not on public PyPI. The compiler accepts `shape=` from the 1.1.10
   line on. The development wheel `1.1.10.dev20261026` on Tenstorrent's S3 index is built from the pinned commit `fec38c6de`, so its
   simulator and compiler are the same source as the pinned checkout. Its light wheel names tt-metal commit `0e9d200`, which is in the
   history of the host tt-metal v0.80.0. Whether it compiles and runs on this chip has not been tested, because that needs a lease.
4. `ttl.grid_size` inside a thread: the source does not show a reason for a split. See the limits section.

## What is installed here

| Place | Package | Version | Built from | tt-metal it targets |
| --- | --- | --- | --- | --- |
| `<venv>` (shared hardware venv) | `tt-lang` (compiler, runtime and a simulator) | 1.1.6 | tag v1.1.6 (checked: `ttl/operators.py`, `ttl/ttl_api.py` and the three simulator files have the same sha1 as the tag) | v0.74.0, commit `c49bb7625e6a`, and `Requires-Dist: ttnn==0.74.0` (`ttl/config.py` lines 17 and 18, `METADATA` line 37) |
| `<venv>` | `ttnn` | 0.80.0, editable install of `<metal>` | host tt-metal v0.80.0 (`978f93b56e3`, 2026-10-13) | |
| simulator venv | `tt-lang-sim` | 1.1.10+gfec38c6, local build | the pinned checkout `fec38c6de` (2026-10-22) | none (no ttnn, no compiler) |
| `<src>` | tt-lang clone, shallow, detached | `fec38c6de` | one commit after v1.1.10 (`88cb759d4`) | `third-party/tt-metal-version`: `TT_METAL_TAG=v0.79.0-dev20260917` |

The installed 1.1.6 compiler targets tt-metal v0.74.0 and runs on a v0.80.0 host. The issue 205 smoke kernel (an elementwise kernel)
compiled and ran there, so that mismatch did not stop elementwise kernels. Issue 95 records that every tt-lang kernel fails to compile on
v0.81.

## What exists to install

| Source | Versions | Dates | Contents |
| --- | --- | --- | --- |
| Public PyPI `tt-lang` | 1.0.1, 1.0.2, 1.0.3, 1.0.4, 1.0.6, 1.1.0, 1.1.1, 1.1.3, 1.1.5, 1.1.6 | 2026-05-06 to 2026-07-18 | compiler and simulator, needs `ttnn==0.74.0` (1.1.6) |
| Public PyPI `tt-lang-sim` | 1.0.6, 1.1.0, 1.1.1, 1.1.3, 1.1.5, 1.1.6 | 2026-05-13 to 2026-07-18 | simulator only |
| Git tags | v1.1.7 (2026-07-17), v1.1.8 (2026-08-11), v1.1.9 (2026-08-12), v1.1.10 (2026-09-18) | | never published to public PyPI (`pip index versions` and the PyPI JSON list nothing after 1.1.6) |
| S3 index, `tt-lang/2026-10/` | `1.1.10.dev20261012`, `.dev20261019`, `.dev20261026` | weekly, Monday | bundled wheel (ships ttnn), `+light` wheel (no ttnn), `tt-lang-light`, `tt-lang-sim` |
| S3 index, `tt-lang/ttmetal/<sha7>/` | light wheels built and device-tested against one tt-metal commit | `0e9d200`: tt-lang 1.1.10.dev20260921; `e908c31`: 1.1.6.dev20260717; `c04ae27`: 1.1.5.dev20260708; `13adda8`: 1.1.5.dev20260704 | |

Evidence: `curl` of `https://pypi.eng.aws.tenstorrent.com/tt-lang/`, `.../2026-10/` and `.../ttmetal/` on 2026-10-29; the PyPI JSON
for `tt-lang` and `tt-lang-sim`; `<src>` `docs/sphinx/build.md`, section "Publishing to S3 PyPI" (scheduled weekly at 08:00 UTC Monday,
version `<X.Y.Z>.dev<YYYYMMDD>`).

I unpacked two wheels to read them:

| Wheel | `TTLANG_COMMIT` | `TT_METAL_TAG`, `TT_METAL_COMMIT` | `reduce_sum` in `ttl/operators.py` | simulator `reduce_sum` |
| --- | --- | --- | --- | --- |
| `tt_lang-1.1.10.dev20261026+light` (cp312) | `fec38c6de4b6f24c` (the pin) | `v0.79.0-dev20260917`, `0e9d200db976` | line 1556: `(input, *, dims, shape=None)` | line 660: `shape` required |
| `tt_lang-1.1.10.dev20260921+light` (cp312, from `ttmetal/0e9d200/`) | `cb44e80b7fa1` | `v0.79.0-dev20260917`, `0e9d200db976` | line 1382: `(input, *, dims, shape=None)` | not read |

For the `dev20261026` wheel, `ttl/operators.py` and `ttl/sim/{math,block,nodecontext}.py` have the same sha1 as the pinned checkout, and
the separate `tt_lang_sim-1.1.10.dev20261026` wheel has the same three simulator files. The `+light` wheel has no `Requires-Dist: ttnn`; it
takes `ttnn` from an external tt-metal (`ttl/_setup/external_tt_metal.py`, `tt-lang-setup-external-tt-metal`).

Host tree check: `<metal>` `git merge-base --is-ancestor` says `0e9d200`, `e908c31`, `c04ae27` and `13adda8` are all in the history of v0.80.0.
The commit that `config/tt-lang-emule-stack.json` pins, `b6c508c4`, is not an ancestor of v0.80.0 (both are on the tree, on different
lines).

## Operation by operation, by version

"Sim" is `python/sim/` (also `ttl/sim/` in the wheels). "Compiler" is `python/ttl/operators.py` and the files it re-exports. Signatures
were read with `ast` from `git show <ref>:<path>` in `<src>`.

| Operation (probe use) | v1.1.6 sim vs compiler (public PyPI, installed) | v1.1.10 | `fec38c6de` (pin; = `1.1.10.dev20261026`) |
| --- | --- | --- | --- |
| `reduce_sum(x, dims=[-1], shape=(1, 1))` | **disagree.** sim `(block, dims, shape)` with `shape` required (`python/sim/math.py` line 635 to 640); compiler `(input, *, dims)` (`python/ttl/operators.py` line 738). The kernel the sim accepts is rejected with `unexpected keyword argument 'shape'`. A kernel that omits `shape` compiles and fails in the sim. No call form works in both. | agree for the keyword form: compiler `shape=None` (line 1382), sim `shape` required | agree: compiler `shape=None` with a deprecation warning when omitted (line 1556 and `_warn_if_reduce_shape_omitted`, line 1486), sim `shape` required (line 660) |
| `reduce_max` | same as `reduce_sum` (compiler line 747, sim line 608) | agree (line 1397) | agree (line 1571) |
| `block.broadcast(x, dims=[-1], shape=...)` | agree: compiler `(input, *, dims, shape)` (line 625), sim `(block, dims, shape)` (`block.py` line 89) | agree | agree |
| `block.fill(v, shape=...)` | agree for keyword `shape` (compiler `(value, *, shape, dtype=None)`, sim `(value, shape)`) | agree, both add `tile` | agree |
| `math.rsqrt` | agree: compiler generated op `rsqrt(input)` (`ttl/_generated_elementwise.py` line 93, installed), sim table entry `"rsqrt": torch.rsqrt` (`ttl/sim/math.py` line 95) | agree | agree by the same table; the spec marks it supported in both since 0.1.7 |
| `math.exp` | agree: compiler `exp(input)` (installed `_generated_elementwise.py` line 73), sim `"exp": torch.exp` (line 88) | agree; both gain `approx`, `scale`, `skip_clamp_check`, `iterations` (compiler line 1638, sim `math.py` line 199) | agree, same signature on both sides (compiler line 1812) |
| `+`, `*`, `-`, `/` (add, mul, scale by a constant) | agree: the spec's Appendix D lists them as supported in both since 0.1.7 | agree | agree |
| `scale_shift` as `x * s + b` | same operators as the row above; there is no `scale_shift` function in either package | same | same |
| matmul, `@` | the spec lists simulator 0.1.7, compiler 0.1.8; compiler `matmul(lhs, rhs, *, transpose_rhs=False)` (line 808) | agree on the operator | agree (compiler line 1673) |
| `ttl.grid_size(dims=2)`, in the operation body | agree: compiler `(*, dims)`, supports `dims=2` only (docstring line 581); sim `(dims=2)` | agree | agree; the compiler now also accepts `dims=1` (line 1177 onward) |
| `ttl.grid_size` inside `@ttl.compute` or `@ttl.datamovement` | not shown by the source; see limits | the same | the pin's `atom_split.py` lists `grid_size` as a control-placement call, which concerns thread-unified operations and is not in the installed 1.1.6 (`ttl/_src/` has no `atom_split.py`) |

Spec Appendix D (`docs/sphinx/specs/TTLangSpecification.md` lines 1559 to 1605 at the pin) holds the release numbers. The rows that matter
here: `ttl.math.reduce_sum` and `reduce_max` simulator 0.1.7, compiler 0.1.8; `ttl.block.broadcast` 0.1.7 and 0.1.7; `ttl.math` unary
`exp`, `log`, `sqrt`, `rsqrt` 0.1.7 and 0.1.7; `grid_size` with `dims=2` 0.1.7 and 0.1.7, with any `dims` simulator 0.1.7, compiler "N/S".
The table lists features and gives no call signatures. It says `reduce_sum` is supported in both, and it gives no hint that the 1.1.6 compiler
refuses the `shape` argument that the specification itself requires.

The `reduce_sum` history in one place: the specification (0.20) and the simulator require `shape`. The compiler at 1.1.6 and 1.1.9 has
no `shape` parameter. The compiler gained `shape=None` between v1.1.9 (`364091457`) and v1.1.10 (`88cb759d4`). The clone is shallow, so I
could not name the commit. Even at 1.1.6, upstream's own `examples/eltwise_broadcast_reduce.py` calls `reduce_sum(..., shape=...)`
(lines 153 to 160), and its compiler tests such as `test/python/simple_reduce.py` call it without `shape`. At that release the examples
and the compiler tests used different forms.

## Version pairs

| Simulator | Compiler and runtime | Hardware path | `reduce_sum` with `shape=` | `broadcast`, `fill`, `rsqrt`, `exp`, elementwise | `grid_size` | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1.6 (public `tt-lang-sim`, or the simulator inside `tt-lang` 1.1.6) | `tt-lang` 1.1.6, built for tt-metal v0.74.0 | host v0.80.0 (installed now) | sim accepts, compiler rejects | agree | body use agrees | **disagree** on reductions. This is the failure of issue 205. |
| 1.1.10+gfec38c6 (local build, pin) | `tt-lang` 1.1.6 (installed now) | v0.80.0 | sim accepts, compiler rejects | agree | see limits | **disagree**. This is the probe's present setup. |
| 1.1.6 sim, using no `shape` | 1.1.6 compiler | | the sim raises on a missing `shape`; the compiler accepts | agree | | disagree in the other direction. No single reduction source passes both. |
| `1.1.10.dev20261026` sim (pin) | `1.1.10.dev20261026+light` (pin), tt-metal `0e9d200` (v0.79.0-dev20260917), light wheel uses the host `ttnn` | v0.80.0 host, not run | both accept; compiler warns if `shape` is omitted | agree by source | agree by source | **agrees by source.** Not run on a chip. |
| `1.1.10.dev20261026` sim | bundled `1.1.10.dev20261026` wheel (ships its own ttnn for `0e9d200`) | uses its own bundled tt-metal runtime and the board; it does not use the host v0.80.0 `ttnn` | same | same | same | same source; it replaces the host `ttnn` in the venv, so it is a separate venv with the driver as the shared part. Not run. |
| `1.1.10.dev20260921` (`cb44e80b7`), from `ttmetal/0e9d200/` | same light wheel, "built and device-tested against" `0e9d200` | v0.80.0 host | compiler accepts `shape` | agree | | Device-tested by upstream on `0e9d200`. Its commit predates the pin, and I did not read its simulator. |
| any tt-lang commit | tt-metal v0.81 and later | | | | | **no pair works.** Every tt-lang kernel fails to compile (issue 95): `kernel_profiler.hpp` moved, `binary_op_init_common` and `pack_tile_block` removed. Upstream main still emits the old forms. |

## Compiler-backed emulation

`docs/sphinx/simulator.md` describes a second simulator backend, `./bin/tt-lang-sim --backend=emule`, for a source checkout. It
runs the real compiler and tt-metal and sends the generated kernels to tt-emule, so a kernel that passes it has compiled with the
hardware path's compiler. The supported stack in `config/tt-lang-emule-stack.json`: tt-lang base `88cb759d4` (v1.1.10), tt-metal
`b6c508c4`, target `p150`. `docs/sphinx/compiler-emule-test-status.md` records a suite run of 5,742 passed and 165 failed (132 dynamic-buffer
tests, 24 RMSNorm tests that lack `ckernel_sfpu_rsqrt.h`, and others). The RMSNorm failure is in the emulator's JIT, so `rsqrt` kernels
are not covered there. This backend needs a Docker image and a source build. I did not run it.

There is also an upstream open pull request, tenstorrent/tt-lang #957, "Add optional compiler validation before simulation". Its
description says: "The functional simulator can accept and execute kernels that the compiler later rejects." With validation required,
90 of 100 kernels in its test set were rejected by compiler diagnostics before simulation. The change is open, is absent from the pin, and is absent from
every wheel I read.

## Statements found upstream

| Where | Statement |
| --- | --- |
| `docs/sphinx/simulator.md` (pin) | "The simulator typically supports more language features than the compiler at any given point", with a link to Appendix D |
| `docs/sphinx/simulator.md`, "Per-node body execution" | the simulator runs an operation body once per node and the compiler runs it once; a body that changes state outside itself behaves differently. Issue 804 tracks two more differences (pipe guards, nodes outside the active rectangle). |
| `docs/sphinx/specs/TTLangSpecification.md` Appendix D | per-feature first release for the simulator and the compiler; "N/S" for not supported |
| `third-party/tt-metal-version` (each tag) | the tt-metal tag each tt-lang commit builds against: v1.1.6 v0.74.0; v1.1.7 v0.75.0-dev20260715; v1.1.8 v0.75.0; v1.1.9 v0.77.0-dev20260812; v1.1.10 and the pin v0.79.0-dev20260917 |
| `.github/scripts/find-compatible-ttlang.sh` | the project's own tool for the tt-lang side of the question: given a tt-metal install and its date, it walks tt-lang history and reports the newest commit that builds and passes a device gate |
| `docs/sphinx/build.md` and the S3 index README | a light wheel in `ttmetal/<sha7>/` is "built and device-tested against" that tt-metal commit |
| `CHANGELOG.md` at the pin | stops at Version 1.1.1; it gives no release notes for 1.1.3 onward, so it cannot answer the question |
| Glean (searched 2026-10-29) | the public docs (`docs.tenstorrent.com/tt-lang/simulator.html`, `getting-started.html`, `build.html`) repeat the repository text. No page publishes a matrix of simulator, compiler and hardware agreement by release. The pull request above is the only item on the gap. |

I found no release cadence statement for public PyPI. The dates show 1.1.0 to 1.1.6 in 9 weeks (2026-05-15 to 2026-07-18) and nothing
public since. The S3 development index publishes weekly.

## Limits of this research

- Nothing was compiled or run on a chip, in the simulator, or in the emulator. Every "agree" above is "the source accepts the same call".
  A passing signature comparison does not show that the generated kernel is correct on the chip.
- `grid_size` inside a thread. The operator's finding is that it compiles only in the simulator. In the sources I read, `ttl.grid_size`
  is a compile-time call that reads the grid set "before compiling threads" (`ttl_api.py` lines 1423 to 1426 at v1.1.6, identical in the
  installed wheel), and the specification says it "can be used anywhere in an operation". I found no source line that rejects it inside a
  thread, and the repository holds no recorded compiler message for it. The failing message should be added to issue 205 or a new issue
  before anyone decides this row. At the pin the thread-unified code in `ttl/_src/atom_split.py` treats `grid_size` as a control call,
  which could change behaviour, and the installed 1.1.6 does not have that file.
- The 1.1.6 compiler targets tt-metal v0.74.0 and the host is v0.80.0. The dev wheels target v0.79.0-dev20260917. Compatibility
  of the light wheel with the host v0.80.0 tree is the open question, and so is the reverse (the host `ttnn` import against a wheel built
  for `0e9d200`). Upstream's `ttmetal/<sha7>/` device test covers only that exact commit.
- The shallow clone has one commit of history. The commit that added `shape=None` to the compiler is in 1.1.10 (between `364091457` and
  `88cb759d4`), and I did not find its pull request.
- The scratch downloads were two wheels from an internal-looking index that answered without credentials. They were not installed.

## Recommendation

No public release pair makes a simulator-passing reduction compile on the chip. With the installed 1.1.6 compiler there is no call form
for `reduce_sum` that both the simulator and the compiler accept, so a simulator pin cannot fix this on its own.

The pair to try first is the one in the pin itself, taken as published wheels in place of a local build:

1. Simulator: keep `1.1.10+gfec38c6` (or install `tt-lang-sim 1.1.10.dev20261026` from the S3 index; its three simulator files match the
   pin byte for byte).
2. Compiler: `tt-lang 1.1.10.dev20261026+light` from the S3 index (`https://pypi.eng.aws.tenstorrent.com/tt-lang/2026-10/`, as
   `tt-lang-light==1.1.10.dev20261026`), in a new venv beside the hardware venv, pointed at the host tt-metal v0.80.0 tree with
   `tt-lang-setup-external-tt-metal`. It is built from `fec38c6de`, so its compiler accepts `reduce_sum(..., shape=)`.
3. The fallback if the light wheel does not import against the host v0.80.0 `ttnn`: the light wheel from `ttmetal/0e9d200/`
   (`1.1.10.dev20260921`), which upstream device-tested against a commit that is inside v0.80.0, although it is built from the earlier
   commit `cb44e80b7` and its simulator was not read.
4. Before the coder sees any of this, take one lease and compile the issue 205 reduction kernel and the smoke kernel in that venv. That
   run answers the question that this research cannot. It also gives the preflight row that issue 95 asks for ("a preflight row that
   compiles the smoke kernel after a tree change").

What would have to be built if step 4 fails: a hardware tt-lang built from `fec38c6de` against the host tree (issue 95), with
`TT_METAL_REF` set in `orchard/ttlang_pin.py`. `.github/scripts/find-compatible-ttlang.sh` in `<src>` is the upstream tool for picking
the tt-lang commit for a given tt-metal install and date.

Two changes on the orchard side do not need a chip:

- The consistency check (`orchard/ttlang_consistency.py`) can compare the hardware venv's `ttl.build_info()` (the tt-lang commit and the
  tt-metal commit the wheel was built from) with the pin. At present the "hardware tree" row says "todo" and checks only the host tree.
  The installed 1.1.6 reports tt-lang v1.1.6 and tt-metal v0.74.0, so the row would show the mismatch that issue 205 found, before the
  coder spends an attempt.
- The probe can run the signature comparison that issue 205 suggests: for each function a passing kernel calls, compare the simulator's
  parameters with the installed compiler's, using `ast` on the installed `ttl/operators.py`. This run did that by hand for the
  functions in the table above, and `reduce_sum`, `reduce_max` are the only ones that differ at 1.1.6.
