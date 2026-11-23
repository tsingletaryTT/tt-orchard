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
    ap.add_argument("--served", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--harness-result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w, c, q, sv, dg = (json.load(open(x)) for x in (a.warm, a.cold, a.quality, a.served, a.diag))
    h = json.load(open(a.harness_result))
    hb = h["benchmark"]["decode_tok_s"]
    t = q["total"]
    dec = w["decode"]["tok_s"]
    prefill = [p for p in w["prefill"] if "seconds" in p]
    prefill_err = [p for p in w["prefill"] if "error" in p]
    at = [p for p in w.get("decode_at_length", []) if "tok_s" in p]

    # --- tables -----------------------------------------------------------------------------------------------------------------
    speed = ["| measure | context (prompt tokens) | new tokens | N | value |", "| --- | ---: | ---: | ---: | --- |",
             f"| decode, batch 1 | about 10 to 20 (5 prompts) | {w['decode']['new_tokens_each']} | {dec['n']} | median {dec['median']:.2f} tok/s (min {dec['min']:.2f}, max {dec['max']:.2f}) |"]
    for p in at:
        s = p["tok_s"]
        speed.append(f"| decode, batch 1 | {p['prompt_tokens']} | {p['new_tokens']} | {s['n']} | median {s['median']:.2f} tok/s (min {s['min']:.2f}, max {s['max']:.2f}) |")
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
    free = ["| prompt | tokens compared | free run |", "| --- | ---: | --- |"]
    ex = ""
    for r in dg["prompts"]:
        res = "all match" if r["matched_fully"] else f"first difference at token {r['first_difference_at_token_index']} ({r['matching_positions']} of {r['compared']} positions equal)"
        free.append(f"| `{' '.join(r['prompt'].split())[:60]}` | {r['compared']} | {res} |")
        if r["matched_fully"]:
            continue
        r1, r2 = r["reference_top2"]
        c1, c2 = r["chip_top2"]
        gen = r.get("reference_generate_choice") or {}
        same_order = c1["id"] == r1["id"] and c2["id"] == r2["id"]
        reversal = gen.get("id") == r2["id"] and bool(r.get("reference_generate_choice_differs_from_its_full_forward_top1"))
        if r["near_tie"] and same_order and reversal:
            ex += (f"\nFirst difference of `{' '.join(r['prompt'].split())[:40]}` (token {r['first_difference_at_token_index']}): the reference's full forward ranks {r1['text']!r} above {r2['text']!r} by "
                   f"{r['reference_gap']} logits, the port ranks the same two tokens in the same order, and the reference's own generation took {r2['text']!r}. A gap that small is within "
                   f"bf16 rounding, so this difference is a near tie.\n")
        else:
            ex += (f"\nFirst difference of `{' '.join(r['prompt'].split())[:40]}` (token {r['first_difference_at_token_index']}): the records do not support an explanation, so it is not explained.\n")

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
| Context | 32,768 tokens in the model; this server accepts up to 8,192 (prompt plus answer) by default, the longest length checked |
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
- About 6 GB of free disk for the checkpoint, plus the Python environment `install.sh` builds

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

Measured through this bundle's own `run.sh` on four Blackhole chips (two p300 boards) with `ttnn` 0.80.0, one request at a time, greedy decoding, wall time through the server, warm kernel cache. Each figure is the median of the runs stated in the N column. The host load average (1, 5 and 15 minutes) was {load_text(w)} the decode and prefill runs. The port runs part of its work on the host CPU, so the decode rate falls when other jobs use the host.

{chr(10).join(speed)}
{('Prefill sizes that did not complete: ' + '; '.join(str(e) for e in prefill_err)) if prefill_err else ''}
The port has no key-value cache: each new token runs the whole sequence again, so the time per token grows with the length of the prompt and the answer together, as the rows with a longer context show. The in-process decode benchmark of the bring-up harness, which ran on one prompt with 32 steps and no server, gave a median of {hb['median']:.1f} tok/s (min {hb['min']:.1f}, max {hb['max']:.1f}, {hb['n']} runs). A second measurement run, started with an empty kernel cache, gave a served decode median of {c['decode']['tok_s']['median']:.2f} tok/s (min {c['decode']['tok_s']['min']:.2f}, max {c['decode']['tok_s']['max']:.2f}, {c['decode']['tok_s']['n']} runs; host load {load_text(c)}); its first request took {c['first_request_s']} s. The two runs ran at different times with other jobs on the host, so the difference between their medians says nothing about the kernel cache.

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

Questions answered with the expected fact: {chat_found} of {len(sv['chat'])}. The substring check cannot tell a good answer from a lucky one.

## Limitations

- Four chips are needed, and each chip holds a full copy of the model: the other three chips add no speed. One request runs at a time; a request that arrives while one is running gets `503`.
- Decode speed is modest. The measured rate is far below the memory-bandwidth limit of the boards (the harness's in-process median is {hb['median'] / h['benchmark']['floor']['floor_tok_s'] * 100:.1f} percent of an estimated floor of {h['benchmark']['floor']['floor_tok_s']:.0f} tok/s for decode with one token in flight, assuming bf16 weights), and it falls as the context grows (see the rows above), because the port has no key-value cache and runs the whole sequence again for every token.
- The decode rate depends on host load, because part of the work runs on the host CPU.
- Prompt plus answer is limited to {8192:,} tokens by default (`AION_MAX_LEN`). The model supports 32,768 tokens; lengths above 8,192 were not checked on the chips.
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
| port | `extra_code/aion_port.py`, generated by the tt-orchard bring-up harness and checked against the CPU reference |
| server | `extra_code/aion_asgi.py` (ASGI app served by uvicorn) |
| runner | `aion_model.py` at the bundle root, which re-exports the app for tt-model's packager |
| bring-up and measurement records | `docs/run-logs/2026-11-22-aion-can.md` and `docs/run-logs/evidence/aion/` in the tt-orchard repository |
| integrity | `SHA256SUMS` lists the hash of every file |
"""
    open(a.out, "w").write(card)


if __name__ == "__main__":
    main()
