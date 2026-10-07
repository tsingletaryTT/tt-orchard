# Plan: the `tt-orchard` command (Phase 2 of the bringup spec)

Spec: `docs/superpowers/specs/2026-10-06-bringup-command-design.md`, sections 4 and 9. Branch
`bringup-command`. Test first, one commit per task, and a mutation for each guard (each mutation must turn
a test red; clear `__pycache__` before each run).

One correction to the spec: the tier loader refuses unknown tables, so the defaults live in their own file,
`config/bringup.toml`, not in a `[bringup]` table of `tiers.toml`.

## Scope

`tt-orchard bringup MODEL` reads `config/bringup.toml`, checks everything it can check without a lease,
then starts `orchard.supervisor run` with the flags the config implies. `status`, `pause`, `resume` and
`abort` forward to the existing commands. Nothing in the supervisor changes in this phase. Not in this
phase: outcome classes, the `blocked` end state, the run budget, the sidecar gate (Phase 3).

## Tasks

1. **Config** (`orchard/bringup_config.py`, `config/bringup.example.toml`). Strict TOML like the tier
   loader: unknown keys refused with a did-you-mean, a missing required key named, `CHANGE-ME` refused.
   Required: `runs_root`, and `[coder]` `target`, `port`, `chips`. Optional: `tiers`, `cache_root`,
   `hf_home`, `operator_home`, `gozer`, `required_chips`, `skills_dirs`, `package_format`,
   `package_namespace`, `package_models_root`, `min_free_gb`, `[env]`, and `[coder]` `kind`, `profile`,
   `image_id`. Relative paths resolve against the config file's directory.
2. **Run directory** (same module). The model id is checked (`org/name`, letters, digits, `.`, `_`, `-`;
   no `..`, no leading `-`). The default run directory is `<runs_root>/<org>--<name>` in lower case.
3. **Argv builder** (same module). Config + model id -> the exact `supervisor run` argument list. A golden
   test compares it with the example in README section 4.3. `--accept-credentials-visible` is never added by
   the builder; only the `--accept-credentials-visible` flag of `bringup` passes it.
4. **Preflight** (`orchard/preflight.py`). Each check is a pure function of injected signals and returns a
   result with a name, a status (`ok`, `warn`, `block`), a detail line and, for a block, a reason from the
   spec's enumerated set. Checks: model id and hub lookup (exists, public, has a license, file sizes);
   disk (model bytes plus `TEST_DISK_GB`, on the filesystems of `hf_home` and `cache_root`); credentials
   (`supervisor.visible_credentials`); tiers load and the coder port names exactly one chips tier; coder
   port free; gozer reachable, with `STALE` and foreign leases reported as warnings and never cleared. A hub
   that cannot be reached is a warning when the snapshot is already local and a block otherwise.
5. **Command** (`orchard/cli.py`, console script `tt-orchard`, `python -m orchard.cli`). `bringup` prints the
   preflight with the orchard styling, refuses (exit 2) on any block, supports `--dry-run` (prints the
   checks and the command, starts nothing), and otherwise calls `supervisor.main`. `status`, `pause`,
   `resume`, `abort` forward. Exit codes of the supervisor pass through.
6. **Docs.** README section 4 gets `tt-orchard bringup`, the config table and an updated status line;
   CLAUDE.md log; package version.

## Order and checks

Tasks 1 to 3 first (pure functions, no I/O beyond reading the config). Task 4 uses injected signals, so no
test touches the network, gozer or a socket. Task 5's tests call `cli.main` with a fake supervisor and prove
that a block never reaches it (the supervisor's main is replaced by a function that fails the test if
called). A dry run on this machine at the end of Phase 2 shows the real preflight for `Cloudflare/clef`.
