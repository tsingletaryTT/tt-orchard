# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The one record of which tt-lang the whole environment uses.

One tt-lang commit supplies three things, so the coder is tested against the language it was taught:

  * the simulator (the tt-lang-sim wheel is built from this commit),
  * the specification, the guide and the examples the coder reads in reference/ (from the checkout at this commit),
  * the hardware build: the compiler wheel and tt-metal commit recorded at the bottom (issue 95).

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

# The hardware pair. PR 223 ran it on the chip and every kernel that passed the pinned simulator and compiled ran correctly
# (docs/run-logs/2026-10-29-ttlang-pair-chip-check.md; the matrix that chose it is 2026-10-29-ttlang-compat-matrix.md).
# `ttl.build_info()` in a venv made from the wheel below printed these two commits. TT_METAL_REF is the tt-metal commit
# the compiler was built against (tag v0.79.0-dev20260917). It runs on the host tt-metal v0.80.0 runtime, so it is not the
# host tree's commit: the "hardware tree" consistency row compares the venv's build_info with it and only names the host tree.
TT_METAL_REF: str | None = "0e9d200db976120c129ab0deb13aa3f6d972b723"
TT_METAL_TAG = "v0.79.0-dev20260917"
TT_METAL_NOTE = f"tt-metal {TT_METAL_TAG} ({TT_METAL_REF[:7]}), the compiler's build tree; the host runtime is newer"

# The compiler wheel of the pair, from https://pypi.eng.aws.tenstorrent.com/tt-lang/2026-10/ (public PyPI stops at 1.1.6).
# Its tt-lang commit is SHA, so the simulator, the references and the compiler all come from one commit.
HARDWARE_TTLANG_SHA = SHA
WHEEL_INDEX = "https://pypi.eng.aws.tenstorrent.com/tt-lang/2026-10/"
WHEEL_VERSION = "1.1.10.dev20261026"
WHEEL_SHA256 = {
    "tt_lang-1.1.10.dev20261026+light-cp312-cp312-manylinux_2_34_x86_64.whl":
        "56d8f3e33b41fb41dcd013b05760f7c0deb6bcec61a4a7e8f58f417135bfdafc",
    "tt_lang_light-1.1.10.dev20261026-py3-none-any.whl":
        "5f28a7180c6d6866a2c57e42a586f25023a132d5470ce7c12153029220b99888",
}
