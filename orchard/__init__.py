# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tt-orchard: supervisor for autonomous model bring-up on a Tenstorrent Quietbox."""

# The last released version. This literal is the only place the number is written: setuptools reads it
# (pyproject.toml, `attr = "orchard.RELEASE"`) without importing the package, and scripts/version.py
# release rewrites it. Pull requests never edit it; they add changes/<slug>.toml instead.
RELEASE = "0.9.36"


def __getattr__(name):
    """`orchard.__version__` is the computed current version (orchard/version.py): the highest of RELEASE
    and every fragment target in changes/ of a source checkout, or RELEASE when no changes/ is found.
    It is computed on first use, so `python3 -m orchard.version` does not import its own module twice."""
    if name == "__version__":
        from orchard import version
        value = version.current()
        globals()["__version__"] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
