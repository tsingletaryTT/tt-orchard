# Driver safety review: findings and rulings (2026-10-02)

Reviewer: opus, read-only, fake-root probes. Verdict: with fixes; do not run the driver as it stands.

Critical (must fix before any real run):
1. H4 starts a device open without re-checking the lease after a reset that did not exit 0 (exits 13 and 15 fall through to `start_child`).
2. `confirm_gone`'s return value is ignored, so the reset runs after a failed "child is gone" check; only the first chip is checked for CLAIMED.
3. A signal during `gozer reset` or `gozer release` kills gozer mid-reset (subprocess.run SIGKILLs its child on an exception) while its `tt-smi -r` keeps running, and cleanup then starts a second reset on the same chips.
4. The H3 probe (`gozer reset` while the child holds the device) does not stop the run if gozer fails to refuse, and accepts any exit 15. Worst case on real hardware: a real `tt-smi -r` under a live ttnn open. Safer probe: run that one call with `GOZER_RESET_CMD` pointing at a marker script that writes a file and exits 1; a refusal gives exit 15 and no marker; no refusal gives exit 17 and a marker, and the real chips are never reset.
5. The default `--gozer` resolves to the live gozer, which has no `reset`; the driver would take a lease and then fail on an untested path.

Important: cleanup gaps; H6 passes when the reset inside the release failed; the grant check is a membership test; the "touches the other board" stop condition is unchecked; the summary does not say what was not checked; the second reset after H4 is missing.

Rulings (controller): accept all of them. The H3 probe stays, with the marker-guarded reset command. `tt-smi -s` becomes opt-in. First real session: one board alone, short idle, a human-watched second terminal; the full 960 s run and two drivers at once come only after one clean run per board.
