# Versions and releases

The version number is computed. Nobody edits a version field in a pull request.

## The rule

* `RELEASE` is the last released version. It is one line in `orchard/__init__.py`: `RELEASE = "X.Y.Z"`.
  Only `scripts/version.py release` changes it.
* Each feature adds one file, `changes/<slug>.toml`, named for its branch. The file says which version the
  feature claims. Two features can claim the same number.
* The version that `tt-orchard --version`, the web page header and `orchard.__version__` report is the highest of
  `RELEASE` and every target in `changes/`. This is also the version `main` reports after a merge.
* `pyproject.toml` has no version of its own. It reads `orchard.RELEASE` (`dynamic = ["version"]`), so
  `pip show tt-orchard` reports the last release. An installed copy has no `changes/` directory, so it
  reports `RELEASE` too.

Versions are `MAJOR.MINOR.PATCH`: three whole numbers, no leading zeros (except `0`), each below 1,000,000.
They are compared as numbers, so `0.9.10` is higher than `0.9.9`.

## How a feature picks a target

1. Look at the current version: `python3 scripts/version.py show`.
2. Create the fragment: `python3 scripts/version.py new <slug> --summary "One plain sentence." --kind feature`.
   The default target is the next patch above the current version on your branch. Add `--target X.Y.Z` to claim a
   different number, and `--pr N` once the pull request exists.
3. Run `python3 scripts/version.py check` before you push.

The slug is lower-case letters, digits and hyphens, at most 60 characters. Use the branch name. Because each
branch owns its own file, two pull requests never edit the same file, and a merge has no version conflict.

A fragment file looks like this:

```toml
target = "0.9.37"
summary = "One sentence in plain English."
pr = 135              # optional
kind = "feature"      # optional: feature, fix, docs or chore
```

Unknown keys, a summary over 300 characters, a file over 4 KiB, a link, or a target that is not a valid version
all make `check` fail. At run time a bad fragment is skipped and never stops the program.

## How the merge picks the highest

Nothing is decided at merge time. After each merge, `changes/` holds the fragments of every merged feature, and
the computed version is the highest target among them. If feature A claims `0.9.38` and feature B claims `0.9.37`
and both are merged, `main` reports `0.9.38`. If both claim `0.9.38`, `main` reports `0.9.38`.

## Cutting a release (steps for a person)

A release is cut on `main`, with everything you want in it already merged.

1. `git checkout main && git pull`, and make sure the tree is clean (`git status`).
2. `python3 scripts/version.py check`. Fix anything it reports.
3. `python3 scripts/version.py release --dry-run`. It prints the new version, the release notes it would write and
   the fragments it would delete. It changes nothing.
4. `python3 scripts/version.py release`. It refuses to run on a dirty tree. It sets `RELEASE` in
   `orchard/__init__.py` to the current version, writes `docs/releases/<version>.md` (the date and every fragment's
   summary grouped by kind, with the pull request number) and deletes the fragments it used. It does not tag,
   commit or push.
5. Read `docs/releases/<version>.md` and edit the wording if you want to.
6. Make a branch, commit the changes (message `Release X.Y.Z`), push it, open a pull request, and merge it. Do not tag the
   branch commit. A merge commit or a squash gives `main` a different commit, and the tag must name the commit that `main`
   points to.
7. After the merge: `git fetch origin`, then `git checkout main && git merge --ff-only origin/main`.
8. Run `git log -1`. It must be the merge commit of the release pull request.
9. Run `python3 scripts/version.py tag-check X.Y.Z`. It changes nothing and exits non-zero unless HEAD is `main` (and is
   the commit `origin/main` points to), the tree is clean, `RELEASE` equals `X.Y.Z`, no fragment remains in `changes/`,
   and the tag `vX.Y.Z` does not exist yet. On success it prints the commit and the exact tag command.
10. Tag that commit: `git tag -a vX.Y.Z -m "Release X.Y.Z" <commit>`, then `git push origin vX.Y.Z`.
11. `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file docs/releases/X.Y.Z.md --verify-tag`.

`release` prints this sequence with the version filled in.

## A feature that merges after a release was cut

The release deletes the fragments it used and moves `RELEASE` up. A branch that was started earlier may still carry a
fragment whose target is now at or below `RELEASE`. After the merge, `check` fails for that fragment and says to
retarget it. Edit the `target` line in `changes/<slug>.toml` to a version above `RELEASE` (the next patch is
usual) and push. Until then the fragment does not raise the version, because only targets above `RELEASE` count.

Do the same on the branch before merging: `python3 scripts/version.py check` after `git merge origin/main` shows it.

## Moving an open PR to fragments

For a branch that still bumps `pyproject.toml` and `orchard/__init__.py` the old way:

1. Merge `origin/main` into the branch. In the version conflicts, take main's `pyproject.toml` and
   `orchard/__init__.py` unchanged (`git checkout origin/main -- pyproject.toml orchard/__init__.py`).
2. Run `python3 scripts/version.py new <slug> --target <the version the PR had> --summary "..." --pr <number>`.
   If the PR's version is not above main's `RELEASE`, pick the next patch instead.
3. Keep your `docs/project-log.md` entry, and add it at the end of the file.

To see the fragment before you write it, run `python3 scripts/version.py migrate-pr --from-ref <branch>`. It reads
`orchard/__init__.py` at that ref, prints the fragment it would create, and writes nothing. The summary it prints is
the last commit subject on the branch with a trailing `(version X.Y.Z)` removed; edit it.

## What a release is for

A release gives a version number to a state of `main` that someone checked and wrote down. Until a release exists,
the version that `main` reports changes with every merge, and no tag or release note names it. A release is the
point to which a bug report, a machine's installed copy or a run log can refer: "this ran on `v0.10.0`". It
also gives the project a short, dated list of what changed (`docs/releases/`), which the per-pull-request log does
not.
