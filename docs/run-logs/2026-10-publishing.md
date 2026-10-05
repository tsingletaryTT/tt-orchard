# Publishing the first two bring-ups (2026-10-04)

At the operator's request both models were measured, given real cards and published as public repos under the
`episod` Hugging Face user. Nothing was added to the tt-model community catalog (no `--publish`).

| Repo | Chips | License | Checked |
|---|---|---|---|
| `episod/hemmingway-1-p300` | 2 | CC BY-NC 4.0 | benchmarked; pulled from the Hub and booted cold |
| `episod/hemmingway-1-p150` | 1 | CC BY-NC 4.0 | booted once from a local copy |
| `episod/openthai2.0-qwen3.8-27b-p300` | 2 | Apache-2.0 | benchmarked; pulled from the Hub and booted cold |
| `episod/openthai2.0-qwen3.8-27b-p150` | 1 | Apache-2.0 | booted once from a local copy |

Order of work: two benchmark agents (one per board) measured the staged packages; a card agent rewrote the four
cards from the results; the four packages were uploaded private; the two 2-chip packages were pulled with
`tt-model pull` into a scratch directory and served with `tt-model serve` on empty tensor caches (both ready in
about 35 minutes, 7 x 6 = 42, a correct Fibonacci function); the cards were updated with that result; the repos
were made public. All four chips were free afterward.

Results and gradings: `docs/run-logs/publishing/` (copies of each run's `RESULTS-bench.md` and quality notes).

Findings that changed the cards:
- Warm boots took 28 to 34 minutes in these measurements, not the 2 to 5 minutes seen earlier. About 1260 s of the
  Hemmingway boot was drafter and verify warm-up. The cause was not investigated.
- The DFlash2 drafter was trained on base Qwen3.8-27B. Its acceptance falls on prose and on Thai: Hemmingway 4.38 of 7
  on coding and 1.72 of 7 on writing; the Thai model 5.57 of 7 on coding and 1.03 of 7 on Thai (decode about 25 to
  29 tok/s on Thai against about 80 on code).
- Hemmingway-1 needs thinking turned off for a visible answer.
- The Thai model garbles long fixed Thai text (the full ceremonial name of Bangkok).

Not done: no accuracy benchmarks, no base-model comparison, no 4-chip package, no fresh download of weights from the
upstream repos (the Hub test used local copies), no v5.1 package. The 1-chip checks reused the 2-chip venv.
Operational notes: `gozer run` killed with SIGTERM left the server running and its lease STALE; the server's process
group had to be killed and `gozer reconcile` run. Releasing a lease while another agent resets its board can print
"reset failed" and still end with the chips FREE.
