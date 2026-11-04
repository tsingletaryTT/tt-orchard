<!-- Written by Claude (not by the model under test) as the stand-in for the upstream "strong general model" prompt-enhancement stage, following prompts/enhance_prompt.md of the SenseNova-6.8-Code2Video repository. Every number below was computed with Python before the brief was written. -->
<!-- Original request: "A 20-second silent math explainer of how attention weights are computed. 1280x720, 24 fps, same dark teal palette as the title card. Deliverable out/final.mp4 plus SPEC.md and VERIFY.md." -->

Make a 20-second silent explainer titled "Attention weights". The one idea: a token turns three dot products into three weights that add up to 1, and those weights decide how much of each value it takes. The audience is a software engineer who has heard of attention and wants to see the arithmetic once, with real numbers. The tone is calm and exact, like a lecture slide that moves. The most important quality the film must have is that every number on screen is the correct result of the step it labels, and that each step visibly produces the next.

## Format

- 1280×720, 24 fps, H.264 (libx264), yuv420p, MP4.
- Target duration 20.0 s, allowed range 19.9 to 20.1 s. Frame count at the target: 480.
- No loop. No audio stream at all (encode with `-an`).
- Language of all on-screen text: English.

## Exact content

The example has three tokens in a row: `the`, `cat`, `sat`. The query token is `sat`. Dimension d = 2. All values are exact and must be shown as written, rounded as given.

| Item | `the` | `cat` | `sat` |
|---|---|---|---|
| key vector k | [1, 0] | [0, 2] | [1, 1] |
| value vector v | [1, 0] | [0, 1] | [1, 1] |
| score q·k, with q = [1, 2] | 1 | 4 | 3 |
| scaled score q·k / √2 (√2 = 1.414) | 0.707 | 2.828 | 2.121 |
| e raised to the scaled score | 2.028 | 16.919 | 8.342 |
| attention weight (each exp ÷ 27.289) | 0.074 | 0.620 | 0.306 |

- The sum of the three exponentials is `27.289`. The weights sum to `1.000`.
- The output is the weighted sum of the value vectors: `[0.380, 0.926]`.
- Strings that may appear on screen, verbatim and nothing else: `Attention weights`, `one query, three keys`, `the`, `cat`, `sat`, `query q = [1, 2]`, `k`, `v`, `q · k`, `÷ √2`, `softmax`, `weights`, `sum = 1.000`, `output = Σ weight × v`, and the numbers in the table above and the sum and output. Use no other words, no formulas beyond these, and no claims about real models.

## Look

- Background solid `#092221`. Primary text and bars `#74C5DF`; the query token, the highlight of the step being computed, and thin rules `#1B8EB1`; secondary labels `#4F8A99`. Nothing else is coloured.
- Typography: a clean sans-serif installed in the environment (check with `fc-list`; Liberation Sans or DejaVu Sans). Token names 40 px bold; numbers 34 px; small labels at least 24 px; the title 56 px bold. Monospace for vectors is allowed if installed.
- No glow, no shadows, no 3D, no particles.

## Layout and composition

- Three columns, one per token, with their centres at x = 240, 640 and 1040. The token names sit at y = 150. Vectors sit under them at y = 215 (k) and y = 265 (v).
- The step area is the band y = 320 to 640. Bars grow upward from a common baseline at y = 600 with a maximum bar height of 240 px. Numbers sit directly above each bar.
- The query block `query q = [1, 2]` sits at the left of the top band under the title area at x = 60, y = 60 once it has been introduced; the title is centred at the top only during 0–2 s and then leaves.
- Keep a 5% safe margin (64 px) on all sides. No text overlaps another element at any frame.

## Camera

- Fixed. No zoom, pan or shake.

## Structure and timing

| Time (s) | On screen | What happens |
|---|---|---|
| 0.0 – 2.0 | Title | `Attention weights` fades in over 0.5 s, `one query, three keys` appears under it at 0.8 s, both fade out together from 1.6 s to 2.0 s. |
| 2.0 – 6.0 | Tokens and vectors | The three token names slide in from above at 2.0, 2.3 and 2.6 s. Under each, its `k` vector appears (3.2, 3.4, 3.6 s) and then its `v` vector (4.0, 4.2, 4.4 s). `query q = [1, 2]` appears at 5.0 s and the token `sat` gets a `#1B8EB1` underline. Hold to 6.0 s. |
| 6.0 – 10.0 | Scores | Label `q · k` appears at 6.0 s. Three bars grow from the baseline, one per token, to heights proportional to 1, 4 and 3 (4 reaches 240 px), starting at 6.5, 7.0 and 7.5 s with a cubic ease-out over 0.8 s. The numbers `1`, `4`, `3` count up from 0 above the bars. Hold to 10.0 s. |
| 10.0 – 13.0 | Scaling | Label `÷ √2` appears. Each bar shrinks smoothly to its scaled height (0.707, 2.828, 2.121, on the same pixel scale as before: 1 unit = 60 px) over 1.2 s, and the numbers change to `0.707`, `2.828`, `2.121`. Hold to 13.0 s. |
| 13.0 – 17.0 | Softmax | Label `softmax` appears. The bars morph to the exponentials (2.028, 16.919, 8.342) for 1.0 s, shown as numbers only (the bars are clipped at 240 px and the `cat` bar is drawn at 240 px with its number), then to the weights: bar heights 0.074, 0.620, 0.306 times 240 px, with numbers `0.074`, `0.620`, `0.306` and the label `weights`. `sum = 1.000` appears at 16.0 s. Hold to 17.0 s. |
| 17.0 – 20.0 | Output | The three weights slide to the `v` vectors they scale; the label `output = Σ weight × v` appears and the result `[0.380, 0.926]` appears at 18.2 s in `#74C5DF`. Everything else dims to `#4F8A99`. Hold the result completely still from 18.8 s to 20.0 s. The last frame is not blank. |

## Motion

- Every animated quantity is a closed-form function of time with easing (cubic in/out). No bounce or overshoot. Numbers that count up show the interpolated value to the same decimals as their final value.
- Objects keep their identity: the `cat` bar stays the `cat` bar from 6.5 s to the end.

## Sound

- None. The film is silent and must have no audio stream.

## Production

- The picture is a pure function of time: the same frame number always renders the same image; no `Date.now()`, no `requestAnimationFrame` loops that advance state, no running CSS animations. A browser page with `window.seek(t)` and `window.READY`, captured with the skill's capture script, is a good fit. Compute the numbers in code and compare them with the table before drawing them. No image or video generation models, no downloaded artwork.
- Use only fonts already installed in the environment.

## Check and revise

- First render stills at t = 1.0, 5.5, 9.0, 12.0, 15.0 and 19.5 s. At each, list the text that is on screen and compare it with the table; measure each bar's height in pixels against its value; check that nothing overlaps.
- After the full render make a contact sheet at one frame per second and read it end to end.
- Run the skill's video check with: 1280 720 24 19.9 20.1 and audio forbidden (`0`).
- Improvement pass: name the weakest moment, fix it, re-render to a new versioned path, keep the better one.

## Delivery

A finished, playable video at `out/final.mp4`: 1280×720, 24 fps, H.264, yuv420p, 19.9–20.1 s, no audio stream. Code, stills or a plan alone do not count. Also write `SPEC.md` (before coding) and `VERIFY.md` (after the checks, with evidence of what the pixels show).
