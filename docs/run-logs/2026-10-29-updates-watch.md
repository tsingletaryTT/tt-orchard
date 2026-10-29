# tt-orchard updates: a read-only check of tools and the coder (2026-10-29)

Operator request: orchard should notice old tools, a new coder model, and synced upgrades of tt-lang, tt-metal, tt-model-manager and
tt-cli, and say when the new combination is ready to farm with. The operator was away, so the defaults below were chosen without asking.

## Defaults taken

- Config is a separate optional file, `config/updates.toml` (keys `family`, `coder_packages`, `authors`, `repos`, `cache_hours`). The
  bringup config rejects unknown keys, and this check should not make a bringup config invalid.
- Coder packages: both packages of `setup_machine.CODERS` whose key contains "coder". Candidate authors: the authors of those packages.
- Candidate scan: Hugging Face model lists by author and by each family word, filtered by id substring. It lists up to 5, newest first.
  tt-model-manager's catalog is the Hugging Face Hub, so the same API covers it.
- Newer tt-lang: a tag above the pin's tag is `behind`; a dev wheel on the S3 index newer than every verified wheel for the pin is
  `newer-candidate`. Main moving past the pin without a new tag is not a status.
- Upstream for tools: GitHub latest release, else the newest tag with a version number. `tt-model-manager` is `tenstorrent/tt-model-manager`,
  `tt-cli` is `tenstorrent/tt-cli`, tt-gozer is `tsingletaryTT/tt-gozer`; each can be changed under `repos`.
- The one record in `config/verifications.json` is the pair check of `2026-10-29-ttlang-pair-chip-check.md` (pin `fec38c6`, wheel
  `1.1.10.dev20261026`, tt-metal v0.80.0, pass). No other record was written.
- Cache: `~/.cache/tt-orchard/updates.json`, 6 hours.
- Preflight shows one `updates` line with status OK when the cache has a behind row or a candidate.

## What a real run printed here

`tt-orchard updates --refresh`, network only, no lease (home paths shortened):

```
item                                 installed                  age     newest upstream                              age     status
coder qwen3-coder-next               bf490f59f0                 1 d     bf490f59f0                                   1 d     current
coder qwen3-coder-next-blackhole     4468d348c4                 ?       -                                            ?       unknown (Hugging Face: HTTP 404)
coder candidates                     qwen3-coder-next, qwen3-co ?       krush420/qwen3-coder-30b-sre-lora (2026-10-2 today   newer-candidate
tt-lang                              1.1.10 fec38c6de (the pin) 7 d     tag v1.1.10, dev wheel 1.1.10.dev20261026, P ?       current
tt-metal (host tree)                 v0.80.0                    16 d    v0.80.0                                      15 d    current
tt-model-manager (tt-model)          0.1.0                      ?       -                                            ?       unknown (upstream: tenstorrent/tt-model-manager has no release or tag with a version)
tt-cli (tt, the official CLI)        1.2.0                      ?       v1.2.1                                       3 d     behind
tt-gozer (gozer)                     0.3.4                      ?       -                                            ?       unknown (upstream: tsingletaryTT/tt-gozer has no release or tag with a version)
tenstorrent/skills checkout          <skills clone>: current    ?       main                                         ?       current
```

Findings:

- The coder bundle is the latest Hugging Face commit. The fallback repo answers 404 on the API (renamed or private), so that row is unknown.
- The candidate scan matched 40 repos that are not configured. Newest five: `krush420/qwen3-coder-30b-sre-lora`,
  `loom-fun/unsloth-Qwen3-Coder-30B-A3B-Instruct-loom`, `tokiers/Qwen3-Coder-30B-A3B-Instruct`,
  `hakunokina/Huihui-Qwen3-Coder-Next-abliterated-oQ6e`, `furiosa-ai/Qwen3-Coder-30B-A3B-Instruct-FP8`. None is a tt-model bundle by an
  author already in use. The match is by name only, so fine-tunes and quantizations of other hardware appear. Narrowing to repos that
  carry a tt-model manifest is deferred.
- tt-cli is one patch release behind (1.2.0, upstream v1.2.1). Its `ready to farm` line says `unverified`.
- tt-model-manager and tt-gozer publish no release or tag, so their rows are unknown. Neither is reported as current.
- The tt-lang pin and the S3 dev wheel agree (1.1.10.dev20261026), and the host tt-metal tree is the latest release.

## Limits

- No test proves behavior against a live Hugging Face or GitHub answer. Tests use fakes; the real run above is the only live check.
- The tt-lang row does not read the newest tag's tt-metal ref, so it cannot say which tt-metal tag a new tt-lang commit needs. Rule R1
  sends the person to `find-compatible-ttlang.sh`.
