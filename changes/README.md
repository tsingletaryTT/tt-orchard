# changes/

Note for agents: do not edit the version in `pyproject.toml` or `orchard/__init__.py`. Add one file here for
your branch instead: `python3 scripts/version.py new <branch-slug> --summary "One plain sentence."`. Run
`python3 scripts/version.py check` before you push. The process is in `docs/releases.md`.

Each file is `changes/<slug>.toml` (lower-case letters, digits and hyphens, at most 60 characters):

```toml
target = "0.9.37"        # the version this feature claims; two features may claim the same number
summary = "One sentence in plain English."
pr = 135                 # optional
kind = "feature"         # optional: feature, fix, docs or chore
```

The version main reports is the highest target in this directory (or `RELEASE` when there is none).
`scripts/version.py release` turns the fragments into release notes and deletes them.
