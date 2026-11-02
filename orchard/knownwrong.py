# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ttnn calls that returned a wrong result with no error on a probed build, so a port does not use them silently.

An entry here is a measurement on one build. It says nothing about other builds. Each entry names the GitHub issue that holds the
numbers and the build that was probed. The inventory (orchard/opinventory.py) flags a traced op whose arguments match an entry,
and the Qwen4 ops probe (scripts/qwen4_ops_probe/p1_conv1d.py) refuses the call unless the caller says it wants the wrong result
(to reproduce it).

Standard library only.
"""
from __future__ import annotations

# ttnn.conv1d with dilation above 1 and groups above 1 (depthwise, in the probe). PCC 0.18 to 0.28 against torch at C = 64, 128 and
# 2048, and no error. Dilation 1 and groups 1 are both fine on their own. The workaround is a sum of shifted multiplies.
CONV1D_DILATED_GROUPED = {
    "id": "conv1d-dilated-grouped",
    "ttnn": "conv1d",
    "issue": 338,
    "build": "tt-metal build of 2026-10-30, one Blackhole chip",
    "what": "ttnn.conv1d with dilation above 1 and groups above 1 returns a wrong result and raises no error",
    "workaround": "sum of shifted multiplies with ttnn elementwise ops, then silu",
}
CONV_OPS = ("convolution", "_convolution", "conv1d")


class KnownWrong(Exception):
    """A call that matches a known-wrong entry."""

    def __init__(self, entry: dict, detail: str):
        self.entry = entry
        super().__init__(f"{entry['what']} ({detail}; issue {entry['issue']}, {entry['build']}). Workaround: {entry['workaround']}")


def conv1d_known_wrong(dilation, groups) -> dict | None:
    """The entry when `dilation` and `groups` are both above 1, else None. Values that are not ints match nothing."""
    ok = all(isinstance(v, int) and not isinstance(v, bool) for v in (dilation, groups))
    return CONV1D_DILATED_GROUPED if ok and dilation > 1 and groups > 1 else None


def check_conv1d(dilation, groups) -> None:
    """Raise KnownWrong for a call ttnn.conv1d gets wrong on the probed build."""
    entry = conv1d_known_wrong(dilation, groups)
    if entry is not None:
        raise KnownWrong(entry, f"dilation {dilation}, groups {groups}")


def conv_args(op: str, consts) -> tuple[int, int] | None:
    """(dilation, groups) of a traced convolution from the constants the trace records, or None when they cannot be read.

    The trace template records the numbers and short integer lists a node was called with, in argument order. For
    aten.convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups) that is three lists
    (stride, padding, dilation), the bool, a list (output_padding) and the groups int. aten.conv1d(input, weight, bias, stride,
    padding, dilation, groups) gives three lists and the groups int. The dilation is the first value of the third list."""
    name = op[5:] if op.startswith("aten.") else op
    if name.split(".", 1)[0] not in CONV_OPS or not isinstance(consts, list):
        return None
    lists = [c for c in consts if isinstance(c, list)]
    ints = [c for c in consts if isinstance(c, int) and not isinstance(c, bool)]
    if len(lists) < 3 or not ints or not lists[2] or not isinstance(lists[2][0], int):
        return None
    # aten.convolution also stands for 2D and 3D and for transposed convolutions, which issue 338 does not cover. A 1D convolution
    # has one-value stride, padding and dilation lists, and the first bool (aten.convolution's `transposed`) is False.
    if any(len(v) != 1 for v in lists[:3]):
        return None
    bools = [c for c in consts if isinstance(c, bool)]
    if name.split(".", 1)[0] != "conv1d" and (not bools or bools[0] is not False):
        return None
    return lists[2][0], ints[-1]


def flag_sequence(sequence) -> list[dict]:
    """The traced convolutions in `sequence` (the trace's ordered op records) that match a known-wrong entry, one finding per
    distinct (op, dilation, groups, module). Records that are not dicts or have no readable constants are skipped."""
    out, seen = [], set()
    for n in sequence if isinstance(sequence, list) else []:
        if not isinstance(n, dict) or not isinstance(n.get("op"), str):
            continue
        dg = conv_args(n["op"], n.get("const"))
        entry = conv1d_known_wrong(*dg) if dg else None
        if entry is None:
            continue
        key = (n["op"], dg, n.get("module"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"op": n["op"], "ttnn": entry["ttnn"], "dilation": dg[0], "groups": dg[1], "module": n.get("module"),
                    "issue": entry["issue"], "build": entry["build"], "what": entry["what"], "workaround": entry["workaround"]})
    return out
