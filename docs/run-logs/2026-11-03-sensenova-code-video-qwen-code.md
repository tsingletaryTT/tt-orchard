# SenseNova 6.8 Code2Video with the `code-video` skill, driven by Qwen Code (2026-11-03)

## Results in brief

- **Pilot (10 s title card):** finished in 63 minutes. The film is correct and clean, and `VERIFY.md` has measured evidence. Without vision, the model used pixel measurements.
- **Explainer (20 s math film):** did not finish in the 120-minute limit. It left a complete render with correct numbers at rest, but some numbers on screen are mid-transition values, and it never wrote `final.mp4` or `VERIFY.md`. Another client was on the server for part of this run.
- **Harness:** one image, one script, Docker limits that held. Found problems are listed below.

## Why

The model card of SenseNova 6.8 Code2Video says the model is meant to run inside an agent harness with a `code-video` skill. The skill is public (`skills/code-video` in the OpenSenseNova/SenseNova-6.8-Code2Video repository). The authors use Claude Code. This run uses Qwen Code instead, against the tt-model package of the model (`episod/sensenova-6.8-code2video-p300`, 2 Blackhole chips), to see whether the package can carry the skill's whole loop: read the skill, write a spec, write the page, render, check, write the verification notes.

Everything below was measured on one box on 2026-11-03. Each number names the file it came from. The runs are single runs, so they show what is possible and say nothing about variance.

## Setup

**Model server.** The package, served with `tt-model serve` on 2 chips, OpenAI-compatible, context 262,144 tokens, thinking on (the server default), host-side sampling, speculative decoding off. The server listened on loopback only. Its counters were at zero before the pilot because it had been restarted.

**Harness.** `scripts/code_video_bench/` in this repository.

- `Dockerfile` builds the sandbox image (3.4 GB): Node 24, Python 3.11 with numpy, scipy, soundfile and librosa, ffmpeg 5.1 (libx264, aac), Playwright 1.64 with Chromium, `@napi-rs/canvas`, and Qwen Code 0.26.0. `edge-tts` and `faster-whisper` are left out on purpose: `edge-tts` sends text to an outside speech service and `faster-whisper` downloads models. Every brief forbids audio.
- `run_bench.sh` makes a task directory, installs the skill unchanged (and records the source path and file hashes outside the container), writes Qwen Code's settings, samples the server's metrics every 15 s, and runs Qwen Code in the container. It removes only the container it created, by the id docker wrote to a cidfile.
- `collect.py` turns a finished run into a summary. It reads the logs and checks the video with ffprobe, ffmpeg and the skill's own `check_video.sh`, in a read-only container with no network. After Copilot's review of this PR, the checker comes from a trusted read-only copy of the skill and not from the task directory, and a figure that was not measured is `null`. The summaries behind the tables below were made before that change: they ran the task's own copy of `check_video.sh`, which the model could have edited, and they did not record which skill files the run changed. The pilot's check result therefore stands as the harness's reading at the time, and a rerun with the current `collect.py` is the way to confirm it.
- `briefs/` holds the briefs.

**Container limits.** Only the task directory is mounted (read-write, same path inside). HOME is inside it. No credentials are passed (no Hub, GitHub or ssh tokens). All capabilities are dropped, new privileges are off, the root file system is read-only, 16 GiB memory with no swap, 8 CPUs, 2,048 processes. The container uses the host network namespace because the model endpoint listens on loopback, so loopback services on the host are reachable from inside. The whole `docker run` also sits in a capped systemd scope. These limits matter on a box with about 120 GiB of usable memory.

**Qwen Code settings** (`logs/qwen-settings.json` of each run, no secrets): the model provider is `openai` with the endpoint's URL and the model name, context window 262,144, `max_tokens` 32,768, request timeout 30 min, 2 retries; `tools.approvalMode` is `yolo`; automatic updates, usage statistics and telemetry are off; managed auto-memory and auto-dream are off. The memory switches matter here. Qwen Code otherwise sends a second background request per turn, and on an endpoint that decodes about 18 tokens/s those would compete with the main loop. Session limits were 1,500 turns and 120 minutes. Thinking was left at the server default (on). Qwen Code's own warning about running headless in yolo mode appears in `stderr.log`; the container is the sandbox.

**Skill discovery.** Qwen Code finds project skills under `.qwen/skills/<name>/`. The skill was copied there unchanged. A probe run asked the model to list its skills; it listed `code-video` and quoted the first sentence of its description correctly.

**Request check.** The logged API request carried the model name, `stream: true`, `max_tokens: 32768` and 10 tool definitions. No sampling parameters were sent, so the server's `generation_config` defaults applied.

**The briefs.** The upstream workflow has a "strong general model" turn a short request into a dense production prompt (`prompts/enhance_prompt.md`). Claude wrote the briefs here, following that template. The model under test did not write them. They are in `scripts/code_video_bench/briefs/`.

**First message to the agent** (the upstream text with the Qwen Code skill path): "Use the project code-video skill. Read .qwen/skills/code-video/SKILL.md and dense_prompt.md, then the relevant craft guides. This directory is the project root. Produce the complete video, run the rendering tools, and deliver out/final.mp4, SPEC.md and VERIFY.md."

## Pilot: a 10-second silent title card

Brief: kinetic type reading `TENSTORRENT`, 1280×720, 24 fps, background `#092221`, accents `#1B8EB1` and `#74C5DF`, deliverable `out/final.mp4`, `SPEC.md`, `VERIFY.md`.

| Number | Value | Label | Evidence |
|---|---|---|---|
| Result | finished, exit 0 | measured | `logs/exit_code`, `logs/summary.json` |
| Wall time | 3,785 s (63 min) | measured | `logs/started_at`, `logs/ended_at` |
| Model requests | 31, all ended with `stop`, none hit the output cap | measured | `logs/metrics-start.json`, `logs/metrics-end.json`, 31 files in Qwen Code's request log |
| Prompt tokens | 1,879,126 (Qwen Code); 1,879,978 (server counter) | measured | `logs/stream.jsonl`, metrics files |
| Completion tokens | 49,501 (Qwen Code); 50,573 (server counter) | measured | same |
| Largest prompt | 96,062 tokens of 262,144; no compaction, no cached prompt tokens reported | measured | `logs/stream.jsonl` |
| Tool calls | 40: 22 shell, 10 file reads, 4 edits, 3 file writes, 1 glob; 3 returned errors; no web fetches | measured | `logs/stream.jsonl` |
| Decode share of wall time | about 72% (2,730 s at the 18.5 tokens/s measured separately on this package) | derived | server token counter, package benchmark |
| Prefill share of wall time | about 14 to 21% (537 to 783 s at 2,400 to 3,500 tokens/s) | derived | same |
| Video | h264, 1280×720, yuv420p, 24 fps, 240 frames, 10.000 s, no audio stream, 78 KB | measured | `logs/summary.json` (ffprobe) |
| Skill's own check (`check_video.sh`, audio forbidden) | OK; one warning: a still stretch from 5.33 to 9.29 s | measured | `logs/summary.json` |
| Frames | 240, of which 199 distinct and 4 identical to the previous one | measured | `logs/summary.json` (per-frame hashes) |
| `SPEC.md`, `VERIFY.md` | written, 3,751 and 4,260 bytes | measured | task directory |

**Server load.** The server had one request running for almost the whole run. Four metric samples, between minute 61 and minute 63, show two running. Another client sent a short request then: the server's counters are 852 prompt tokens and 1,072 generated tokens above Qwen Code's own totals. At most the last two minutes of the run were slowed, and that is under 4% of the wall time.

**What the model did.** It read the skill and craft guides (the first seven minutes), wrote `SPEC.md` (minute 11), wrote an HTML page with `window.seek(t)` and `window.READY` (minute 20), ran the skill's capture script, rendered a first full cut to a versioned path, ran `check_video.sh`, made a contact sheet, measured frames, copied the cut to `out/final.mp4` (minute 57) and wrote `VERIFY.md`. It followed most of the skill's rules without being reminded of them: a pure function of time, stills rendered at the brief's check times before the full render, a versioned render path, checks before delivery. It skipped one: it ran the capture script once for the whole film and did not test 2 to 3 frames first, as the skill advises.

**What it did without vision.** This package has the vision tower off. The skill says to look at stills and the contact sheet. The model called `read_file` on three PNG stills. Qwen Code answered each time "Skipped image file ... (model doesn't support image input)", and the model said it could not view the images and used another route: it decoded frames with ffmpeg and measured pixels, with bounding boxes of the text, sampled colours at chosen coordinates, rule width over time, and a pixel-identity check between two renders of the same time. `VERIFY.md` cites those measurements per requirement. It also made a contact sheet but did not try to read it.

**Judgement.** I read the contact sheet and sampled frames. The film is correct and clean. The word is spelled right (eleven letters), centred, in the specified colours, large, with the accent rule drawn from the centre outward and a smooth fade at the end. The measurements in `VERIFY.md` that I spot-checked match the frames: the rule is about 185 px wide at 5.2 s and centred, the fade is half done at 9.5 s, and the last frame is the background colour. It is a plain design that does what the brief says and nothing more. The letters arrive left to right from a fixed left edge, so the word is only centred when complete, which the brief allowed.

**A mistake in my brief.** The brief asked for a cap height of at least 120 px and also for a word about 900 px wide. Eleven wide capitals at 120 px cap height need roughly 1,500 px, which does not fit a 1,280 px frame. The model noticed, said so in `SPEC.md`, kept a cap height of 88 px, and made the word 1,100 px wide (86% of the frame), where the brief said 900. It listed the cap height under known deviations and the width in the same note. It did not ask. A better brief would have avoided the conflict. This is a pilot, and the conflict resolution is still a useful observation: the model kept the hard requirements (spelling, colours, timing, format) and gave up the soft ones.

## Problems with the harness and the skill

- **Pillow is not in the image.** One of the three tool errors was `from PIL import Image` failing. The skill's text mentions a NumPy/PIL compositor but its `requirements.txt` does not list Pillow. The model recovered by using ffmpeg and raw bytes. The next image should include Pillow.
- **The contact sheet's time labels run ahead.** The skill's `contact_sheet.sh` uses the `fps` filter, which picks the next frame at or after each second. The label for "1.000" shows a frame from about 1.4 s. Frame strips and exact-time stills are more reliable than the sheet for timing claims.
- **One tool error was a typo in a long path.** The model dropped the end of the directory path in a `cd`. Long absolute task paths invite this. A short task path such as `/work` would avoid it.
- **No prompt caching on the server.** Every request re-sends the whole conversation and the server reports no cached prompt tokens, so 1.88 million prompt tokens were prefilled for a 96,000-token final context. That is 14 to 21% of the wall time.
- **My first duplicate-frame metric was wrong.** `mpdecimate` reported 236 unique frames for a film with a 4-second hold. Exact per-frame hashes are used now.

## Explainer: a 20-second silent math explainer

Brief: how attention weights are computed, with one query and three keys and computed numbers (scores 1, 4, 3; scaled 0.707, 2.828, 2.121; weights 0.074, 0.620, 0.306; output [0.380, 0.926]). The numbers were computed with Python before the brief was written.

| Number | Value | Label | Evidence |
|---|---|---|---|
| Result | stopped at the 120-minute limit (exit 55). No `out/final.mp4`, no `VERIFY.md`. `SPEC.md` was written (6,348 bytes). A complete render exists as `out/render_v1.mp4`. | measured | `logs/exit_code`, `logs/stderr.log`, task directory |
| Wall time | 7,205 s | measured | `logs/started_at`, `logs/ended_at` |
| Model requests | 59 (Qwen Code's request log). The server counted 67 finished, so 8 came from another client. | measured | request log, `logs/metrics-*.json` |
| Prompt tokens | 5,188,370 (Qwen Code); 5,190,902 (server counter) | measured | `logs/stream.jsonl`, metrics files |
| Completion tokens | 67,775 (Qwen Code); 84,932 (server counter), so the other client generated about 17,000 | measured | same |
| Largest prompt | 140,773 tokens of 262,144; no compaction | measured | `logs/stream.jsonl` |
| Largest single completion | 11,928 tokens (median 444); no request of this run hit the 32,768-token cap | measured | `logs/stream.jsonl` |
| Tool calls | 74: 36 edits, 18 shell commands, 13 file reads, 6 file writes, 1 glob; 7 errors (3 edits that matched nothing or changed nothing, 4 failing runs of its own scripts); no web fetches | measured | `logs/stream.jsonl` |
| First full render | minute 92 (file modification time of `out/render_v1.mp4`); `SPEC.md` at minute 17 | measured | task directory |
| Time share | prefill about 25 to 36 min of 120 (5.19 million prompt tokens at 2,400 to 3,500 tokens/s); decode at least 61 min (67,775 tokens at 18.5 tokens/s, slower while another client was on the server) | derived | server token counter, package benchmark |
| Video (`render_v1.mp4`) | h264, 1280×720, yuv420p, 24 fps, 480 frames, 20.000 s, no audio stream, 247 KB | measured | `logs/summary.json` (ffprobe) |
| Skill's own check | OK; six still stretches of 1.6 to 2.5 s flagged (the holds) | measured | `logs/summary.json` |
| Frames | 480, of which 423 distinct and 9 identical to the previous one | measured | `logs/summary.json` |
| Server load | two requests running in 96 of 481 samples (20%) | measured | `logs/metrics.jsonl` |

**Server load.** Another client was on the endpoint for about a fifth of this run. Decode is slower while two requests run, so the wall time here is longer than a run alone would take. How much longer I cannot say.

**What the model did.** It read the skill and wrote `SPEC.md` (minute 17). It edited a page that draws the whole film (until about minute 55) and then built four helper scripts for itself: a geometry checker (overlaps, margins, page errors), a pixel checker that compares measured bar heights with the expected ones, a still renderer and a video check. It rendered the six check stills from the brief and a dozen more, and rendered the full film for the first time at minute 92. It then extracted frames from that render and checked them until the limit stopped it. The skill says to get a rough full-length render early and to budget the time. The model did the opposite here: 92 of 120 minutes went before the first full render. The pilot, a much simpler film, had its first render at about minute 45.

**Judgement.** I read the contact sheet and the frames at 12, 15 and 19.5 s. The render has the structure the brief asked for: title, three tokens with their vectors, the bars for the scores, the scaling step, softmax, the weights, the sum and the output. The values at rest match my computed ones: 0.707, 2.828 and 2.121; weights 0.074, 0.620 and 0.306; sum 1.000; output [0.380, 0.926]. At 12 s the three bars are 42, 170 and 127 px, exactly 60 px per unit. The palette and the type are right and the text is legible.

It fails the brief's most important quality in two places. During the softmax step the numbers are interpolated: at 15.0 s the bars read 1.583, 13.209 and 6.513, which are not the result of any step, and the exponentials the brief asked for (2.028, 16.919, 8.342) are never held long enough to read; the contact sheet at 14.0 s shows 1.686, 14.062 and 6.933. In the last second the weight bars hang from their numbers and no longer stand on the baseline, which the brief did not ask for and which looks unfinished. Without a `VERIFY.md` there is also no record of which of these the model knew about. I would not publish this as a finished film.

**What this run shows.** The package can carry a longer skill run: 59 requests, prompts up to 141,000 tokens, no compaction, no cap hits. The model's process is the limit here. Its checking is thorough and expensive, and on a slow endpoint a 2-hour budget is not enough for a film with an animation this size unless the model renders early.

## Questions a performance number must answer (docs/can-cards.md)

- **Warm or cold.** Warm. The server was restarted from a tensor cache built earlier, and no kernel compilation happened during either run. The first start of this package on this box, with an empty cache, took 47 to 52 minutes.
- **How many runs.** One per brief. No difference between two conditions is claimed.
- **What else ran on the host.** Not measured as a load average. Other sessions on the box ran test suites and builds during both runs, and another client used the model server (4 samples of 253 in the pilot, 96 of 481 in the explainer). Tool time inside the container (Chromium rendering with software GL, ffmpeg) may include contention from those jobs. The container was limited to 8 CPUs.
- **Device time or wall time.** Wall time throughout. The split into decode and prefill is derived from token counts and separately measured rates and is labelled as derived.
- **Same harness on both sides.** Not applicable: nothing is compared with another setup.
- **Absolute figures.** Seconds, tokens and tokens per second. Shares of wall time appear only next to the totals they come from.
- **Agreement with the decode bound.** The decode rate used (18.5 tokens/s) is the one measured on this package through its endpoint. This run did not compare it with the board's bandwidth floor.
- **Precision policy.** The package's own, unchanged. Its tensor cache is bfp8.

## Limits

- The explainer's timing is affected by another client that used the server during the run, and its 2-hour limit may have been too short. It does not show how long the film would take alone.
- One run per brief. The runs show that the loop works end to end on this package. They do not show how often it works, and they say nothing about how the model does with a brief written by someone else.
- The model cannot see images here. The skill's frame-review loop therefore ran as pixel measurement. Serving the vision tower on these chips was not tried.
- Only silent films were tried. Narration and music need `edge-tts` (an outside service), model downloads, or other tools that were left out.
- The judgement is mine, from stills and a contact sheet. No one watched the films in real time.
- The briefs were written by Claude. The upstream stage-1 model is Claude Opus 5.5.
- The host network namespace is shared with the container. The container holds no credentials and cannot see the host file system, but it can reach loopback services on the host.

## How to repeat

```
docker build -t cvbench:0.1 -f scripts/code_video_bench/Dockerfile scripts/code_video_bench
git clone --depth 1 https://github.com/OpenSenseNova/SenseNova-6.8-Code2Video.git upstream
WORK_ROOT=/path/for/runs SKILL_SRC=upstream/skills/code-video \
  scripts/code_video_bench/run_bench.sh pilot1 scripts/code_video_bench/briefs/pilot-kinetic-type.md 120 1500
python3 scripts/code_video_bench/collect.py /path/for/runs/pilot1 1280 720 24 9.9 10.1
```

Start the run as a detached unit (`systemd-run --user --unit=...`) so that closing the terminal does not stop it, and keep the model server's endpoint to yourself while a run is going, or the timing numbers include the other client.
