<!-- Written by Claude (not by the model under test) as the stand-in for the upstream "strong general model" prompt-enhancement stage, following prompts/enhance_prompt.md of the SenseNova-6.8-Code2Video repository. -->
<!-- Original request: "A 10-second silent kinetic-type title card reading TENSTORRENT, 1280x720, 24 fps, dark teal background #092221 with accent #1B8EB1 and #74C5DF. Deliverable out/final.mp4 plus SPEC.md and VERIFY.md." -->

Make a 10-second silent kinetic-typography title card titled "TENSTORRENT". The one idea: eleven letters arrive one at a time, lock into a single confident word, and hold still long enough to read. The audience is someone watching the first seconds of a talk or a product video. The tone is calm and precise, like engineering drawings, not flashy. The most important quality the film must have is that the word TENSTORRENT is perfectly legible, perfectly spelled and centred at the end, and that the motion into that final state is smooth and deliberate.

## Format

- 1280×720, 24 fps, H.264 (libx264), yuv420p, MP4.
- Target duration 10.0 s, allowed range 9.9 to 10.1 s. Frame count at the target: 240.
- No loop. No audio stream at all (encode with `-an`; do not add a silent track).
- Language of all on-screen text: English.

## Exact content

- The only on-screen text is the word `TENSTORRENT`, in capital letters, spelled exactly like this, once, on one line. No subtitle, no tagline, no logo, no date, no other characters, no numbers.

## Look

- Background: solid dark teal `#092221` for the whole film. A very faint, static vignette is allowed (not darker than `#061817` at the corners); no grain, no gradients that move.
- Palette and roles: letters at rest `#74C5DF`; the active letter while it moves `#1B8EB1`; one thin horizontal rule under the word `#1B8EB1`. Nothing else may be coloured.
- Typography: a bold geometric sans-serif that is installed in the environment (check with `fc-list`; DejaVu Sans Bold or Liberation Sans Bold are acceptable). Cap height at least 120 px, so the final word is large. Letter spacing wide and even (tracking about 0.12 em). All ten letters use the same font, weight and size.
- What the style is not: no glow, no drop shadows, no 3D, no particles, no emoji, no rainbow colours.

## Layout and composition

- The final word is centred horizontally and sits with its vertical centre at about y = 330 px. The word fills about 70% of the frame width (about 900 px) and never comes closer than 5% (64 px) to the left or right edge.
- The rule is 4 px thick, as wide as the word, 36 px below the baseline, centred.
- No other element is on screen.

## Camera

- Fixed. No zoom, no pan, no shake.

## Structure and timing

| Time (s) | What is on screen | What happens |
|---|---|---|
| 0.0 – 0.5 | Empty background | Hold. Nothing moves. |
| 0.5 – 4.5 | Letters arriving | One letter every 0.4 s, left to right: T, E, N, S, T, O, R, R, E, N, T. Each letter slides up 40 px from below its final place while fading from 0 to 100% opacity over 0.45 s with a cubic ease-out, in `#1B8EB1`, then eases to `#74C5DF` over 0.3 s after it lands. The word has eleven letters, so the 11th letter starts at 4.5 s and lands by 4.95 s. |
| 4.95 – 6.0 | Complete word | All letters at rest in `#74C5DF`. The rule draws from the centre outward to full width over 0.6 s with a cubic ease-in-out, starting at 5.0 s. |
| 6.0 – 9.0 | Complete word and rule | Hold completely still so the text can be read (about 3 s). |
| 9.0 – 10.0 | Complete word and rule | Everything fades together to the empty background colour over 1.0 s with a smooth ease-in-out, ending at exactly 10.0 s. The last frame shows only the background. |

Key moments: first letter starts at 0.5 s; last letter lands at about 4.95 s; rule complete at 5.6 s; fade-out starts at 9.0 s.

## Motion

- Every animated quantity is a closed-form function of time with easing (smoothstep or cubic in/out). No springs, no bounce, no overshoot. Letters never jump or pop; they travel and fade in.
- Each letter keeps its identity: the same letter, same size, from the first frame it appears until the fade-out.

## Sound

- None. The film is silent and must have no audio stream.

## Production

- The picture is a pure function of time: the same frame number always renders the same image; no `Date.now()`, no `requestAnimationFrame` loops that advance state, no running CSS animations. A browser page with `window.seek(t)` and `window.READY`, captured with the skill's capture script, is a good fit. No image or video generation models, no downloaded artwork, no outside network use for the film itself.
- Use only fonts already installed in the environment.

## Check and revise

- First render stills at t = 0.2 s (empty), 2.0 s (mid-arrival), 5.2 s (word complete, rule drawing), 7.0 s (final hold) and 9.9 s (fade). Check the spelling letter by letter (eleven letters), the final word's width (about 900 px) and centring (left and right margins equal within 4 px), and that the colours match the palette.
- Render frame strips 0.1 s apart around 0.5–1.0 s (first letter arriving) and 9.0–10.0 s (fade).
- After the full render make a contact sheet at one frame per second and read it end to end.
- Run the skill's video check with: 1280 720 24 9.9 10.1 and audio forbidden (`0`).
- Improvement pass: name the weakest moment, fix it, re-render to a new versioned path and keep the better one.

## Delivery

A finished, playable video at `out/final.mp4`: 1280×720, 24 fps, H.264, yuv420p, 9.9–10.1 s, no audio stream. Code, stills or a plan alone do not count. Also write `SPEC.md` (before coding) and `VERIFY.md` (after the checks, with evidence of what the pixels show).
