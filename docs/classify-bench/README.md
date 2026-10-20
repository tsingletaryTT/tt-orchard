# Classification bench

Runs stage 0's class decision on a list of models from their Hugging Face headers (no weights are downloaded; sidecar files over 8 MiB are not read)
and sets the result beside independent labels.

1. `python3 -m orchard.classify_bench collect --corpus docs/classify-bench/corpus-40.json --base-snapshot <nearest model snapshot> --out <dir>`
   writes `<dir>/tool.json` (the tool's decisions) and `<dir>/packets/*.md` (one evidence packet per model, with no verdict in it).
2. Give each labeler the packets and a labeling guide only, and have each write `<dir>/labels/labeler-<name>.json`. Use fresh contexts that have not seen the tool's output.
3. `python3 -m orchard.classify_bench compare --out <dir>` writes the agreement table and every disagreement.

`--offline` reads only the HTTP cache; `--refresh` ignores it. `tests/test_classify_bench.py` compares each fake-Hub case with the real stage 0 functions on the same files.
