# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The one record of which tt-lang the whole environment uses.

One tt-lang commit supplies three things, so the coder is tested against the language it was taught:

  * the simulator (the tt-lang-sim wheel is built from this commit),
  * the specification, the guide and the examples the coder reads in reference/ (from the checkout at this commit),
  * the hardware build, once tt-lang kernels build on the host's tt-metal tree (issue 95).

Everything else reads these constants: `defaults.TTLANG_PIN` and friends are aliases of them, `ttlang_env` clones and
builds from them, and `ttlang_consistency` compares the machine with them. To move the pin, edit this file only, then
run `tt-orchard kernel-env` and `tt-orchard ttlang-consistency`.

The specification version is not stored here. It is read from the checkout's own docs at the pinned commit
(`ttlang_consistency.checkout_spec_version`), so it cannot disagree with the commit.

The module holds constants only and reads no file.
"""
from __future__ import annotations

URL = "https://github.com/tenstorrent/tt-lang.git"

# The tt-lang commit. choice: v1.1.10 plus 31 commits (simulator emulator tooling and compiler changes; diagnostics.py
# unchanged), read 2026-10-26.
SHA = "fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0"
VERSION_TAG = "1.1.10"                  # the release tag SHA is counted from

# The simulator wheel is built from SHA, so its source sha is SHA and its version label is the tag plus the short sha.
SIM_SOURCE_SHA = SHA
SIM_VERSION = f"{VERSION_TAG}+g{SHA[:7]}"

# sha256 of python/ttl/diagnostics.py at SHA. The wheel looks for the file at site-packages/ttl/ttl/diagnostics.py and
# does not ship it, so a diagnosed kernel error ends as FileNotFoundError. kernel-env copies the file there and refuses
# to if the file at the pin has changed.
DIAGNOSTICS_SHA256 = "e3964f222cc5dc253157f97645a914f8751ab613f0fa82eb334e97046da8f84f"

# The tt-metal ref the hardware build should use. None until it is known: tt-lang kernels do not build on tt-metal
# v0.81 and the host tree is v0.80.0 (issue 95). Nobody has found a tt-lang commit that builds on a newer tree, so the
# value is left open. The consistency check reports the hardware row as "to be pinned" while this is None.
TT_METAL_REF: str | None = None
TT_METAL_NOTE = "to be pinned with issue 95"
