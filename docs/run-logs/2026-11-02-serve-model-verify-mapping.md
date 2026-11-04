# The tt-serve-model verification table against what tt-orchard checks (2026-11-02)

Source: the `tt-serve-model` plugin in tenstorrent/skills (merged as PR 29), read at upstream main: the `verify` skill (checks V1 to V25),
the `discover` skill (host questions D1 to D16) and its `knowledge/` folder. The descriptions below are in this repository's words. Nothing
was copied from the skill text, and nothing here ran against hardware.

The plugin targets a server started from the tt-inference-server container (`run.py` or a direct `docker run`). tt-orchard serves a model
with `tt-model` and a bundle's own `run.sh`, under a gozer lease. Several checks carry over unchanged because they ask about any
OpenAI-shaped server. Others ask about the container and do not apply.

Files named in the last column: `orchard/can.py` (`boot_check`, the `can boot` command), `orchard/served_verify.py` (new in this change),
`orchard/served_chat.py` (the stage 2 and stage 4 served check, offline), `orchard/skills/weights-swap-templates/serve_and_compare.py` (the
stage 2 server run), `orchard/package_templates/verify_bundle.py` (stage 7 boot of the installed copy), `orchard/hardware_check.py`,
`orchard/preflight.py`, `orchard/verify_package.py` and `orchard/cans_view.py`.

Verdicts: covered, partly, missing, or not applicable with the reason. "Built" means this change adds the check, for `can boot`.

## Verification table V1 to V25

| Id | What it asks | Where tt-orchard does it | Verdict |
|---|---|---|---|
| V1 | Does `GET /health` answer 200? | `can.boot_check` waits on `[check] health_path` and again after the requests (`health` and `health_after`). `serve_and_compare.wait_healthy` (stage 2), `verify_bundle.py` (exit 4) and `server.py` (the coder's health curl) do the same | covered |
| V2 | What model and context length does `/v1/models` report, and do they match what was declared? | Before: `can.boot_check` sent the first listed name and checked only status 200. Stage 7 `verify_bundle.py` exits 7 when the listing lacks the copy's model directory. Built: `served_verify.check_listing` compares the listed name and `max_model_len` with the spec's `max_model_len`, fails an empty listing, and labels the context as below, equal to or above the model's own length when that is known | covered for `can boot` (built); stage 2 and stage 4 partly (issue 368) |
| V3 | Does a raw completion with no chat template start sensibly? | Stage 2 asks `/v1/completions` with token ids and compares 32 greedy tokens with the CPU reference (`serve_and_compare.complete`, `measure`). That compares more than the single word the skill expects. `can boot` asks chat only | covered at stage 2 and stage 7; `can boot` partly (V4 and V5 probes built) |
| V4 | Does `17*24=` complete to 408 at temperature 0? | Not checked anywhere before. Built: `served_verify.probe_completions` (name `arithmetic`). The answer must begin with the whole number 408, so 4080 does not pass | covered by the build (`can boot`) |
| V5 | Does a counting prompt continue 9, 10, 11 with no repeated token (sequential state)? | Not checked before. Built: `probe_completions` (name `sequence`) | covered by the build (`can boot`) |
| V6 | Is generated code consistent when parsed (never run)? | Not checked. The prompt suite (`prompt_suite.py`) sends a code prompt and `served_chat.check_replies` looks at reply shape only | missing (issue 369) |
| V7 | Did the engine open a mesh with every ASIC? | Not checked from the server log. Built: `served_verify.check_mesh` reads the last `multidevice with N devices and grid` line of the bundle's server log and compares N with the spec's `device_count`; no such line is `not_checked`. `hardware_check.py` H2 checks that a child holds the device, which tests lease ownership and says nothing about the mesh | covered by the build when the server logs the line, otherwise `not_checked` |
| V8 | Is the engine doing real work (read from its own counters; telemetry is context only)? | `speedwatch.running_requests` reads `vllm:num_requests_running` for the speed monitor. Built: `served_verify.scrape` and `check_counter` require `vllm:generation_tokens_total` to rise by at least the tokens one chat request reports. No check uses `tt-smi` power as a verdict, which the skill warns against | covered by the build when `/metrics` exists, otherwise `not_checked` |
| V9 | Do the loaded layer count and KV-cache blocks match `config.json`? | Not checked. `can.py` has `ready_contains` for fixed log strings but no layer or block parse | built (issue 370; the layer line was read from a real log on 2026-11-03, see the section for V9 below) |
| V10 | What is the single-stream decode rate? | `can.boot_check` check `decode_rate` (`[check] min_decode_tok_s`, timings or a stream, `stream_rate`). `measure_serving.py` times fixed prompts for the card | covered |
| V11 | Does the server batch concurrent requests, or serialise them? | Not checked by a probe. `bench_session.py` and `toolcall_bench.py` send concurrent requests for benchmarks and give no batching verdict | missing (issue 371) |
| V12 | Does a prompt of about 2,000 tokens prefill and reach its answer? | `prefill_bench.py` guards a hardware timing run for one model family and `measure_serving.py` times fixed prompts. No generic long-prompt probe in `can boot` | partly (issue 371) |
| V13 | Is `reasoning_content` populated for a reasoning model? | `serve_and_compare.ask_suite_prompt` records the reasoning text and `served_chat.check_replies` looks for role headers and end tokens in answer and reasoning. No verdict on whether the field is populated | partly (issue 372) |
| V14 | Is the reasoning parser wired in the engine config? | Not checked | missing (issue 372) |
| V15 | Does this model need a client flag the server does not document? | `[check] request_extra` carries such flags per can (for example `enable_thinking = false`), recorded by hand | partly (a hand-kept list; no probe) |
| V16 | Does turning thinking off cut the completion tokens? | Not measured. `request_extra` turns it off for the boot request | missing (issue 372) |
| V17 | Does a small token budget produce empty content with finish reason `length`? | Built: `served_verify.check_reply` names that pair as a spent token budget when the boot chat reply is empty. The budget sweep is not built | partly (the diagnosis is built; the two-budget sweep is not) |
| V18 | Is tool-call routing worth testing yet (gated on V13 to V17)? | `toolcall_bench.py` and the suite's `needs_tools` prompts test tool calls. Nothing gates them on generation checks | partly |
| V19 | Does the failure repeat at temperature 0 (is it sampling)? | Every boot, suite and probe request is greedy. `sampling_probe.py` (stage 2 step i) records what a sampled request does. No automatic repeat of a failing request | partly |
| V20 | Does the failure repeat through `/v1/completions` (is it the chat template)? | `served_chat.check_files` compares the served template with the new model's, and the suite runs in `chat` or `completion` mode by whether a template exists. No paired rerun of a failing prompt | partly |
| V21 | Is the injected spec entry the same as upstream's? | There is no upstream spec entry. The nearest control: `can.check_state` binds a boot record to the digest of the spec file and SHA256SUMS | not applicable: tt-orchard builds bundles from its own can spec and does not inject a tt-inference-server spec override |
| V22 | Did the server log a quiet warning (fallback, mismatch, ignored option)? | Not checked from a log filter. `stallwatch.py` watches for silence and does not read warnings | missing (issue 373) |
| V23 | Does the implementation's own README validate this hardware? | Cards carry measured rows (`package_card.py`, `can.gate` with `card_profile`). There is no lookup of upstream `docs/model_support` | not applicable: the models served here are not tt-inference-server release entries |
| V24 | Is the symptom a known upstream bug? | `knownwrong.py` lists ttnn calls that returned wrong results, each with its issue. `updates.py` reads upstream release state. No symptom search | not applicable for a boot check (it is a research step); the known-wrong list covers part of it |
| V25 | Does a published image contain a given fix (ancestry by compare, never by version string)? | Not checked | not applicable: tt-orchard pins wheels and revisions in the can spec and does not run tt-inference-server images |

## Discover host facts (D1 to D16)

| Id | Fact | Where tt-orchard does it | Verdict |
|---|---|---|---|
| D1 | Cores, RAM, disk free, distro, CPU | `preflight.check_disk` (free GB per root against what the model needs); the cores, RAM and CPU are not read | partly (issue 374) |
| D2 | Kernel driver version | `setup_machine` step `hardware` (reads the driver version, warns below `MIN_KMD`) and `machine_checks.py` | covered |
| D3 | Boards and ASICs on the machine | `boards_seen.py` (gozer topology, gozer status and an optional `tt-smi -s` snapshot against the chips the config needs); `bus.py` reads the PCI directory | covered |
| D4 | 1 GB hugepages free, docker present and working | `setup_machine.check_hugepages` tests that `/dev/hugepages-1G` exists. The free count and the mount page size are not read. Docker is not needed for bundles | partly (issue 374) |
| D5 | Hugging Face cache present and its size | `preflight.disk_hints` and `hf_home_for` size the cache per root; the token's presence is never reported as a value | covered |
| D6 | Models the host can deploy | `preflight.check_base` (is a bundle for the base installed) and `check_hub` | partly (different question: orchard asks about one requested model) |
| D7 | The release spec entry for one model on this device | none; there is no tt-inference-server spec | not applicable: bundles carry their own can spec |
| D8 | Is port 8000 free? | `preflight.check_port` for the coder port, and `can.check_port` refuses the ports other work owns | covered for the ports orchard uses |
| D9 | The `--tt-device` string from board serials | `boards_seen.py` counts boards from serials; no device string is derived because bundles take `--device-count` | not applicable (the count is covered) |
| D10 | Physical boards against ASICs | `boards_seen.py`, `hardware_check.py` (board BDF), `handoff.decide_park` (boards the coder holds) | covered |
| D11 | Firmware and driver against the model's minimums | `setup_machine` records firmware and driver; no per-model minimum is compared | partly (issue 374) |
| D12 | Idle telemetry baseline | Not recorded. `hardware_check.py` leaves `tt-smi -s` off by default because it opens every device | missing (issue 374) |
| D13 | Who holds the device | `preflight.check_gozer` (leases and untracked holders), `server.py` `_check_docker` (our container holds the device), `hardware_check.py` H2 | covered |
| D14 | Is a reset needed before launch | `gozer` and the `gozer-gatekeeper` skill decide; `hardware_check.py` H3 and H4 test a reset. `preflight` reports and never resets | covered |
| D15 | Gating and license of a model | `preflight.check_hub` blocks a gated or private model and a model with no license; no credentials are sent | covered |
| D16 | Disk budget and clean-box verdict | `preflight.check_disk` (with `prunable_shortfall`) | partly (no single clean-box verdict) |

## Knowledge folder

| Topic | tt-orchard | Verdict |
|---|---|---|
| Mandatory docker flags (`--device`, `--ipc host`, the 1 GB hugepage bind) | `server.py` starts servers through `tt-model`, which owns the container flags. `server.py` `_check_docker` inspects the devices a container holds | not applicable: orchard does not build `docker run` lines |
| `run.py` against direct docker, decided by the HF token | Bundles run with `bash run.sh` and a pinned revision. The token is never used by the harness (`preflight`) | not applicable |
| Boards, ASICs and the device string | see D3, D9, D10 | covered |
| Hugging Face Hub gating | `preflight.check_hub`, `check_base` | covered |

## Where each check runs

`can boot` is the only place the new probes run. The stage 2 and stage 4 served check (`served_chat.py`) reads files on disk and the replies
that `serve_and_compare.py` recorded. It sends no request, and the swap check does not record the `/v1/models` listing, a raw-completion
probe or the engine counter. Those checks need the live server, so they belong in `serve_and_compare.py` (a template copied into the stage
directory and unable to import `orchard`). That is the largest remaining gap and has its own issue (368). Stage 7's `verify_bundle.py` already
checks the listing against the copy's model directory.

## Update after issues 360, 368 to 373 and 385

Built since the table above. Probe code is in `orchard/served_probes.py` (requests) and `orchard/served_verify.py` (judges and log filters). Verdicts that changed:

| Id | Now |
|---|---|
| V2 | covered at stage 2 and stage 4 too (issue 368): `serve_and_compare.py` records `GET /v1/models` and `served_chat.check_live_probe` judges it against the name the swap test asked for and the `--max-model-len` that `run.sh` passes. `can boot` also reads a `--served-model-name` alias from `extra_args` (issue 385) |
| V4, V5 | covered at stage 2 and stage 4 (the two raw completions are recorded and judged by `served_verify.judge_completion`) |
| V6 | built: `served_probes.check_generated_code` (parse, function present, a returned value, no undefined name, no endless loop; static, never run) |
| V8 | covered at stage 2 and stage 4 when `/metrics` exists |
| V9 | built for the KV cache (`GPU KV cache size`, `Maximum concurrency`, `# GPU blocks`) and for a layer line (`loaded N layers`). The layer line of a tt-metal model was read from a real server log on 2026-11-03 (`Loading 64 transformer layers (indices=[...])` and the bar `Loading layers: 100%\|...\| 64/64`, vLLM 0.26.0, V1 engine) and is pinned in the tests; a log without it is `not_checked`. The result also says whether the capacity follows a block count the plugin set (`Overriding num_gpu_blocks=...`), which makes it the size the plugin asked for and not a measurement, and how many full-length sequences the cache holds against the can's `max_num_seqs` |
| V11 | built: `served_probes.probe_batching` (aggregate tokens per second against the single-stream rate, and `vllm:num_requests_running` when the server reports it) |
| V12 | built: `served_probes.probe_long_prefill` (a prompt sized from `usage.prompt_tokens`) |
| V13, V14, V16, V17 | built for a can that declares `[check] reasoning = true`: `reasoning_field`, `reasoning_parser`, `thinking_switch`, `budget_sweep` |
| V22 | built: `served_verify.log_warnings`, recorded under `log_warnings`, never a failure |

Not built: the resume-equals-full-prefill check and the ten continuation invariants of tenstorrent/skills PR 42 (issue 360), which is still open upstream, and a card row for reasoning tokens (issue 361).

Where the stage 2 and stage 4 probes run: the swap template cannot import `orchard`, so it records the raw answers under `live_probe` in `swap-check.json` and `served_chat.py` judges them. A `swap-check.json` from before this change has no record, and the check reports it as not checked.
