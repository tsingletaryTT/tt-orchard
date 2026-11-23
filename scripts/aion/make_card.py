#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Write pkgsrc/can/aion-README.md, the model card of the Aion bundle, from the measurement records.

    python3 scripts/aion/make_card.py --warm WARM.json --cold COLD.json --cold-start-s N --warm-start-s N --quality QUALITY.json \
        --served SERVED.json --diag DIAG.json --harness-result RESULT.json --out CARD.md

The section order follows docs/can-cards.md (the tt-model card conventions). Every number is copied from a record this script is given; the script
writes no machine path, host name or account name. A free-run explanation (a bf16 near tie) is written only when the diagnosis record backs every part
of it, as for the Olmo card. The card carries no bring-up history and no opinion on quality.
"""
import argparse
import json
import unicodedata

NAME = "aion-1.0-instruct-2.3b-blackhole"
BASE = "microsoft/Aion-1.0-Instruct-2.3B"
REV = "28902cbfd9f4e0347d6c853aa1301f5d559b2fc0"


def spread(s, unit="", digits=1):
    return f"{s['median']:.{digits}f}{unit} (min {s['min']:.{digits}f}, max {s['max']:.{digits}f}, {s['n']} runs)"


def load_text(rec):
    b, a = rec["load_before"], rec["load_after"]
    return f"{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f} before and {a[0]:.1f}, {a[1]:.1f}, {a[2]:.1f} after"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", required=True)
    ap.add_argument("--cold", required=True)
    ap.add_argument("--cold-start-s", type=float, required=True)
    ap.add_argument("--warm-start-s", type=float, required=True)
    ap.add_argument("--quality", required=True)
    ap.add_argument("--served", required=True, help="SERVED.json of served_checks.py (the one the card's passkey and chat tables come from)")
    ap.add_argument("--chat", help="bench_chat.py record taken on a quiet host: the decode rate through the chat route")
    ap.add_argument("--boot-rate", type=float, help="decode tok/s of the 64-token chat answer in the can boot check (cold kernel cache, one run)")
    ap.add_argument("--long", required=True, help="bench_long.py record: the long-prompt prefill and the decode rate with a long context")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--harness-result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w, c, q, sv, dg = (json.load(open(x)) for x in (a.warm, a.cold, a.quality, a.served, a.diag))
    h = json.load(open(a.harness_result))
    hb = h["benchmark"]["decode_tok_s"]
    t = q["total"]
    dec = w["decode"]["tok_s"]
    lg = json.load(open(a.long))
    prefill = [p for p in lg["prefill"] if "seconds" in p]          # prefill and long-context rows are from one session of bench_long.py, taken quiet
    prefill_err = [p for p in lg["prefill"] if "error" in p]
    at = [p for p in lg.get("decode_at_length", []) if "tok_s" in p and p["tok_s"]["n"]]
    at_missing = [p for p in lg.get("decode_at_length", []) if "error" in p or not p.get("tok_s", {}).get("n")]
    longest = max([p["prompt_tokens"] for p in prefill] + [r["prompt_tokens"] or 0 for r in sv["passkey"]])

    chat_text = ""
    if a.chat:
        ch = json.load(open(a.chat))
        chat_text = (f"Through the chat route every prompt starts with a system header of about 40 tokens, so the context is longer at the same answer length. Five prompts that ask for a paragraph, "
                     f"{ch['tok_s']['n'] // 5} runs each, 64 new tokens, prompts of {ch['prompt_tokens']['min']} to {ch['prompt_tokens']['max']} tokens, warm kernel cache, host load {load_text(ch)}: "
                     f"decode median {ch['tok_s']['median']:.2f} tok/s (min {ch['tok_s']['min']:.2f}, max {ch['tok_s']['max']:.2f}, {ch['tok_s']['n']} runs). ")
        if a.boot_rate:
            chat_text += f"The boot check's single 64-token chat answer, on an empty kernel cache, ran at {a.boot_rate:.2f} tok/s. "
        chat_text += "\n\n"
    # --- tables -----------------------------------------------------------------------------------------------------------------
    speed = ["| measure | context (prompt tokens) | new tokens | N | value |", "| --- | ---: | ---: | ---: | --- |",
             f"| decode, batch 1 | 5 to 14 (5 prompts) | {w['decode']['new_tokens_each']} | {dec['n']} | median {dec['median']:.2f} tok/s (min {dec['min']:.2f}, max {dec['max']:.2f}) |"]
    for p in at:
        s = p["tok_s"]
        speed.append(f"| decode, batch 1 | {p['prompt_tokens']} | {p['new_tokens']} | {s['n']} | median {s['median']:.3f} tok/s (min {s['min']:.3f}, max {s['max']:.3f}) |")
    for p in prefill:
        s = p["seconds"]
        speed.append(f"| prefill (time to the end of the prompt) | {p['prompt_tokens']} | 1 | {s['n']} | median {s['median']:.2f} s (min {s['min']:.2f}, max {s['max']:.2f}), {p['tok_s_median']:.0f} tok/s |")
    quality = ["| eval | metric | this package | reference | size |", "| --- | --- | ---: | --- | --- |",
               f"| 6 prompts, teacher-forced on the reference's own continuation | top-1 agreement with the reference's token | {t['top1_vs_reference_text_fraction']:.4f} | transformers, CPU, bf16 | {t['positions']} positions |",
               f"| same | top-1 equals the reference's own top-1 | {t['top1_vs_reference_top1_fraction']:.4f} | same | {t['positions']} positions |",
               f"| same | positions where the port's top-1 differs | {t['misses']} of {t['positions']} | same | {t['near_tie_misses']} of them with a reference gap of 0.25 logits or less |"]
    per = ["| prompt | positions | top-1 agreement |", "| --- | ---: | ---: |"]
    for r in q["prompts"]:
        per.append(f"| `{r['prompt']}` | {r['positions']} | {r['top1_vs_reference_text']:.4f} |")
    pk = ["| passkey in filler text, chat route | prompt tokens | found |", "| --- | ---: | --- |"]
    for r in sv["passkey"]:
        pk.append(f"| {r['length']} tokens | {r['prompt_tokens']} | {'yes' if r['found'] else 'no'} |")
    chat_found = sum(1 for r in sv["chat"] if r["found"])
    chat_found_nfkc = sum(1 for r in sv["chat"] if r["expected"] in unicodedata.normalize("NFKC", r["answer"]).lower())          # H2O written with a subscript two
    free = ["| prompt | tokens compared | free run |", "| --- | ---: | --- |"]
    ex = ""
    for r in dg["prompts"]:
        res = "all match" if r["matched_fully"] else f"first difference at token {r['first_difference_at_token_index']} ({r['matching_positions']} of {r['compared']} positions equal)"
        free.append(f"| `{' '.join(r['prompt'].split())[:60]}` | {r['compared']} | {res} |")
        if r["matched_fully"]:
            continue
        r1, r2 = r["reference_top2"]
        c1, c2 = r["chip_top2"]
        label = " ".join(r["prompt"].split())[:40]
        # Written only when the records back every part of it: the reference's gap between its top two is within a bf16 tie, the port ranks the same
        # two tokens first and second, and the port's own gap between them is within the same tie (it may be zero: equal logits).
        explained = bool(r["near_tie"] and r.get("chip_top2_same_tokens_as_reference_top2") and r["chip_gap"] <= dg["near_tie_logits"])
        if explained:
            ex += (f"\nFirst difference of `{label}` (token {r['first_difference_at_token_index']}): the reference's logits for {r1['text']!r} and {r2['text']!r} are {r1['logit']} and {r2['logit']} "
                   f"(a gap of {r['reference_gap']}), and the port's top two are the same two tokens, {c1['text']!r} at {c1['logit']} and {c2['text']!r} at {c2['logit']} (a gap of {r['chip_gap']}"
                   + ("; equal logits, so which of them greedy decoding takes depends on how the tie is broken" if r.get("chip_exact_tie_in_top2") else "")
                   + f"). A gap that small is within bf16 rounding (neighbouring bf16 values near 20 are 0.125 apart), so this difference is a near tie.\n")
        else:
            ex += (f"\nFirst difference of `{label}` (token {r['first_difference_at_token_index']}): the records do not support an explanation, so it is not explained.\n")

    card = f"""---
tags:
- blackhole
- p300x2
- QB2
- tt-dit-server
- thin
- tt-model-cache
- text-generation
license: mit
pipeline_tag: text-generation
base_model:
- {BASE}
---

# {NAME} on Tenstorrent p300x2 (QuietBox 2)

> Derived from [{BASE}](https://huggingface.co/{BASE}) by Microsoft (revision `{REV}`), weights unmodified and not part of this repository. The ttnn port and the server in this bundle are Tenstorrent's, under the Apache License 2.0.

Aion-1.0-Instruct-2.3B is a 2.3 billion parameter instruction-tuned language model for text tasks such as question answering, summarization and rewriting, in eight languages (English, German, Spanish, Portuguese, Italian, French, Chinese, Japanese). It has 28 decoder layers (21 sliding-window attention layers with a 2,048 token window and 7 full attention layers), grouped-query attention with 24 query heads and 8 key-value heads, a vocabulary of 200,029 tokens and a context of 32,768 tokens. This bundle serves it through an OpenAI-style chat and completions API on four Blackhole chips.

Runs on Tenstorrent **p300x2** (QuietBox 2: two p300 boards, four chips), mesh 1x4.

Packaged with tt-model-manager 0.1.0 (manifest schema 6).

## At a glance

| | |
| --- | --- |
| Architecture | decoder-only transformer, 28 layers, 21 sliding-window and 7 full attention, YaRN rotary scaling on the full-attention layers |
| Parameters | 2.3 billion |
| Context | 32,768 tokens in the model; this server accepts up to 8,192 (prompt plus answer) by default; the longest prompt measured was {longest:,} tokens |
| Hardware | p300x2 (four Blackhole chips, mesh 1x4) |
| Precision on the chips | bfloat16, one full copy of the model on each chip |
| Decoding | greedy by default; `temperature`, `top_p` and `seed` are honoured |
| License | MIT (model), Apache-2.0 (port and server) |

## Intended use

Question answering, summarization, rewriting and chat in the model's eight languages, on a QuietBox 2 or another machine with four Blackhole chips, behind one OpenAI-style endpoint for one user at a time. It is out of scope for anything the model's own card rules out: generating harmful or misleading content, and automated decisions in critical contexts such as medical or legal judgments. This package is a bring-up port with a modest decode rate (see Expected performance), so it fits evaluation and development more than serving several users.

## Prerequisites

- The [Tenstorrent CLI](https://github.com/tenstorrent/tt-cli/blob/main/docs/prerequisites.md), `tt`: `uv tool install tenstorrent` (or use tt-model alone, see the Quickstart)
- Tenstorrent hardware: **p300x2**, or any machine where four Blackhole chips are free
- 1 GB hugepages mounted at `/dev/hugepages-1G`
- SFPI 7.85.0 or newer on the host (the version `ttnn` 0.80.0 declares). `install.sh` stops with a message when the host's SFPI is older. The figures below were taken with SFPI 7.85.0, tt-kmd 2.11.0 and Blackhole firmware bundle 19.17.1
- About 8 GB of free disk: the checkpoint (4.6 GB), the Python environment `install.sh` builds (1.9 GB) and the compiled-kernel cache (1.4 GB after the runs below)

## Quickstart

```bash
tt serve <owner>/{NAME}
```

`tt serve` installs the bundle into its own Python 3.12 environment from pinned requirements (`ttnn==0.80.0` from PyPI) and downloads the [`{BASE}`](https://huggingface.co/{BASE}) weights at `{REV}` into your Hugging Face cache (about 4.6 GB; they are not in this repository). The first start with an empty kernel cache took {a.cold_start_s:.0f} s to become ready, and later starts took {a.warm_start_s:.0f} s. The server is ready when `/health` answers and its log shows `"event": "ready"`.

Without tt-cli: `tt-model serve <owner>/{NAME}`. Without tt-model: run `bash install.sh`, then `bash run.sh` (it serves on 127.0.0.1, port 20180; `PORT` and `HOST` change that). The server needs four free chips: on a shared machine, lease both boards first.

## Capabilities

`GET /health`, `GET /v1/models`, `POST /v1/chat/completions` and `POST /v1/completions`.

- The chat route builds the model's own turn format (system, user and assistant turns, each closed by `<|end|>`, with the model's identity line, knowledge cutoff and today's date first), the same text the model's chat template produces. Message roles are `system`, `user` and `assistant`. Message text that contains one of the model's control tokens (`<|user|>`, `<|end|>` and similar) is refused with a `400`, as the model's template refuses it. `/v1/completions` sends the prompt as written.
- Fields: `prompt` or `messages`, `max_tokens` (default 128) or `max_completion_tokens`, `stream`, `stop` (up to four strings of at most 256 characters each), `temperature`, `top_p`, `seed`, and `return_token_ids` (adds the generated token ids). A response carries `usage` and a `timings` object (prefill seconds, decode seconds, decode tokens per second).
- One request runs at a time. A second one gets `503` with `Retry-After: 5`. A body over 1 MiB, malformed JSON, an unsupported field (`n` other than 1, `tools`, `logprobs`, `response_format` and the like) or a prompt longer than the context gets a `400`.

```bash
curl -s http://127.0.0.1:20180/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{{"messages":[{{"role":"user","content":"What is the capital of France? Answer in one word."}}],"max_tokens":16}}'
```

| variable | default | meaning |
| --- | --- | --- |
| `AION_MODEL_DIR` | the pinned revision from the Hugging Face cache | a directory with the model files |
| `AION_MAX_LEN` | 8192 | context limit in tokens, prompt plus answer (the model supports 32,768) |
| `AION_CHIPS` | 4 | chips in the mesh |

## Expected performance

Measured through this bundle's own `run.sh` on four Blackhole chips (two p300 boards) with `ttnn` 0.80.0, one request at a time, greedy decoding, wall time through the server, warm kernel cache. Each figure is the median of the runs stated in the N column. The host load average (1, 5 and 15 minutes) was {load_text(w)} the short-prompt decode runs. The port runs part of its work on the host CPU, so the decode rate falls when other jobs use the host. The load average also rises because of the server itself: during the long-prompt runs below it was {load_text(lg)}, because the host work uses many cores, so a load average above 1 does not by itself mean another job was running.

{chr(10).join(speed)}
{('Prefill sizes that did not complete: ' + '; '.join(str(e) for e in prefill_err)) if prefill_err else ''}
{chat_text}The port has no key-value cache: each new token runs the whole sequence again, so the time per token grows with the length of the prompt and the answer together, as the rows with a longer context show. The in-process decode benchmark of the bring-up harness, which ran on one prompt with 32 steps and no server, gave a median of {hb['median']:.1f} tok/s (min {hb['min']:.1f}, max {hb['max']:.1f}, {hb['n']} runs). A second measurement run, started with an empty kernel cache, gave a served decode median of {c['decode']['tok_s']['median']:.2f} tok/s (min {c['decode']['tok_s']['min']:.2f}, max {c['decode']['tok_s']['max']:.2f}, {c['decode']['tok_s']['n']} runs; host load {load_text(c)}); its first request took {c['first_request_s']} s. The two runs ran at different times, so the difference between their medians says nothing about the kernel cache.

Accuracy against the model's own implementation, run on the CPU in bf16 (transformers, custom code of the pinned revision), on exactly the same token ids. For each prompt the reference generated 128 tokens greedily, and both implementations then read the same text one position at a time:

{chr(10).join(quality)}

{chr(10).join(per)}

These rows show agreement with the reference's own continuation on six prompts (four plain, two in the chat format). They do not measure accuracy on held-out text, and they are not benchmark scores: the model's published benchmark results are on its own model card.

Free-running generation (greedy, through the server) against the reference's 128 tokens:

{chr(10).join(free)}
{ex}
Served checks (greedy, chat route): a passkey hidden in filler text and five short questions, each checked for the expected fact as a substring.

{chr(10).join(pk)}

{chr(10).join(f"- `{r['question']}` expected `{r['expected']}`: {'found' if r['found'] else 'not found'}" for r in sv['chat'])}

Questions answered with the expected fact by the plain substring check: {chat_found} of {len(sv['chat'])}. {('The miss is the formula for water, which the model wrote with a subscript two (H₂O) where the check looked for H2O; after Unicode compatibility normalisation of the answer the count is ' + str(chat_found_nfkc) + ' of ' + str(len(sv['chat'])) + '.') if chat_found_nfkc != chat_found else ''} The substring check cannot tell a good answer from a lucky one.

## Limitations

- Four chips are needed. Each chip holds a full copy of the model and runs the same work; this package was measured on four chips only, and it makes no claim that the extra chips make it faster. One request runs at a time; a request that arrives while one is running gets `503`.
- Decode speed is modest. The measured rate is far below the memory-bandwidth limit of the boards (the harness's in-process median is {hb['median'] / h['benchmark']['floor']['floor_tok_s'] * 100:.1f} percent of an estimated floor of {h['benchmark']['floor']['floor_tok_s']:.0f} tok/s for decode with one token in flight, assuming bf16 weights), and it falls as the context grows (see the rows above), because the port has no key-value cache and runs the whole sequence again for every token.
- The decode rate depends on host load, because part of the work runs on the host CPU.
- Prompt plus answer is limited to 8,192 tokens by default (`AION_MAX_LEN`). The longest prompt measured was {longest:,} tokens. The model supports 32,768 tokens; lengths above {longest:,} tokens were not checked on the chips.
- Only the sliding-window mask, rotary scaling and attention paths exercised by the checks above were verified. Quality on other prompts, other languages, sampled decoding, long generations and tool use was not evaluated. Tool calls and the model's tool format are not supported.
- Greedy decoding was used for every number here.
- Message text with control tokens is refused (see Capabilities). The server has no sign-in, no TLS and no rate limit. It binds 127.0.0.1 by default; `HOST=0.0.0.0` listens on every interface, which is only for a network you trust.
- The model is small. Its own card says it can miss fine details, give misstatements and fail at complex multi-step reasoning, and that its safety alignment may not hold against adversarial input. Read the upstream card before using it.
- Tested on one machine with two Blackhole p300 boards, ttnn 0.80.0 (the PyPI wheel), Python 3.12. Other board types and other chip counts were not tested.

## Licensing

The model is released by Microsoft under the MIT License ([`LICENSE`](https://huggingface.co/{BASE}/blob/main/LICENSE) in the upstream repository). The port, the server and the tools in this bundle are Copyright 2026 Tenstorrent USA, Inc., under the Apache License 2.0 (`LICENSE`, `NOTICE` in this bundle).

## Related packages

- Upstream model: [{BASE}](https://huggingface.co/{BASE}).
- Other ttnn ports packaged the same way for four chips: `episod/olmo-hybrid-7b-blackhole`.

## Feedback

Questions and reports: the Discussions tab of this repository, or support@tenstorrent.com.

## Provenance

| component | built from |
| --- | --- |
| weights | `{BASE}` at `{REV}`, downloaded at first start, not in this bundle |
| port | `extra_code/aion_port.py`, generated by the tt-orchard bring-up harness and checked against the CPU reference. One change was made for this bundle: the tensor reader copies each tensor out of the memory-mapped checkpoint (`.clone()`), because with the PyPI `ttnn` 0.80.0 wheel, `ttnn.from_torch` of a memory-mapped tensor (the embedding table) did not return in 120 s, while a copy uploads in 0.2 s |
| server | `extra_code/aion_asgi.py` (ASGI app served by uvicorn) |
| runner | `aion_model.py` at the bundle root, which re-exports the app for tt-model's packager |
| bring-up and measurement records | `docs/run-logs/2026-11-22-aion-can.md` and `docs/run-logs/evidence/aion/` in the tt-orchard repository |
| integrity | `SHA256SUMS` lists the hash of every file |
"""
    open(a.out, "w").write(card)


if __name__ == "__main__":
    main()
