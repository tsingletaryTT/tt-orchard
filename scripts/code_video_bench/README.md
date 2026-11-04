# code-video benchmark harness

Runs the public `code-video` skill of SenseNova-6.8-Code2Video under Qwen Code, against any OpenAI-compatible endpoint, inside a locked-down container, and measures what happened. The first runs and their results are in `docs/run-logs/2026-11-03-sensenova-code-video-qwen-code.md`.

| File | What it does |
|---|---|
| `Dockerfile` | The sandbox image: Node 24, Python with numpy, scipy, soundfile and librosa, ffmpeg, Playwright with Chromium, `@napi-rs/canvas`, Qwen Code 0.26.0. No `edge-tts` (it sends text to an outside service) and no `faster-whisper` (it downloads models), so films are silent. It has no Pillow; add it if a brief needs it (one pilot tool error came from that). |
| `run_bench.sh` | One run: task directory, skill installed unchanged, Qwen Code settings, metric sampling, the container. Needs `WORK_ROOT` and `SKILL_SRC`. |
| `poll_metrics.py` | Samples the server's `/metrics` (running and waiting requests, token counters) so a run can say whether anyone else used the endpoint. |
| `collect.py` | Summarizes a finished run: stream statistics, server load, and ffprobe, the skill's own `check_video.sh` and per-frame hashes of `out/final.mp4`, all checked in a read-only container with no network. |
| `briefs/` | The dense production prompts used (written by Claude following the upstream `prompts/enhance_prompt.md`, not by the model under test). |

The container gets only the task directory, no credentials, no capabilities, a read-only root file system and memory and process limits. It shares the host network namespace because the endpoint listens on loopback. Start runs as detached units (`systemd-run --user --unit=...`) so closing the terminal does not stop them. Keep the endpoint to yourself during a run, or the timing numbers include the other client; `poll_metrics.py` records when that happened.
