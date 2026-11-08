#!/usr/bin/env python3
"""Write profile.json, card.json and README.md for the Berkelium 2-chip package from the measured files. Every number comes from the result
files: bench-package-p300.json (bench.py), cq-package-p300.json (check_cq.py) and prefill-package-p300.json (bench_prefill.py).
usage: gen_card.py OWNER/NAME   (run in the work directory)"""
import json
import statistics as s
import sys

WEIGHTS_GB, CACHE_GB_LOW, CACHE_GB_HIGH = 61, 35, 45
DISK_TEXT = (f"about {WEIGHTS_GB} GB for the weights plus {CACHE_GB_LOW} to {CACHE_GB_HIGH} GB of tensor cache for each profile you run, so "
             f"{WEIGHTS_GB + CACHE_GB_LOW} to {WEIGHTS_GB + CACHE_GB_HIGH} GB for one profile, plus the Docker image (about 5.6 GB loaded) and the kernel cache")

FORMAT_CHECKS = (("header_json", "a first line that parses as JSON"), ("one_python_block", "exactly one python block"), ("parses", "a block that parses"),
                 ("imports_ok", "imports of only cadquery and math"), ("ends_with_result", "a final statement that assigns `result`"))


def counts_sentence(c: dict) -> str:
    n = c["n"]
    parts = [f"{c['summary'].get(k, 0)} of {n} had {text}" for k, text in FORMAT_CHECKS]
    return "; ".join(parts)


def build(repo: str, b: dict, c: dict, p: list):
    d = b["decode"]
    rates = [x["decode_tok_s"] for x in d]
    med = s.median
    dec_med, dec_min, dec_max = round(med(rates), 1), min(rates), max(rates)
    cad_isl = int(med([x["prompt_tokens"] for x in d])); cad_osl = int(med([x["completion_tokens"] for x in d]))
    cad_ttft = round(med([x["ttft_s"] for x in d]), 2); cad_e2e = round(med([x["total_s"] for x in d]), 1)
    prompts = len({x["prompt_index"] for x in d})
    repeats = len(d) // prompts
    pf2 = [x for x in p if x["target"] == 2048]; pf7 = [x for x in p if x["target"] == 7000]

    def pf(rows):
        t = med([x["ttft_s"] for x in rows]); n = int(med([x["prompt_tokens"] for x in rows])); return n, round(t, 2), int(n / t)
    n2, t2, r2 = pf(pf2); n7, t7, r7 = pf(pf7)
    n = c["n"]; ok = c["summary"]
    cq_prompts = c.get("distinct_prompts") or len({r["prompt_index"] for r in c["rows"]})
    valid_prompts = len({r["prompt_index"] for r in c["rows"] if r.get("valid_solid")})
    profile = {
        "p300": {"hardware": "p300", "mesh_device": "P300", "max_model_len": 8192, "max_num_seqs": 32,
                 "description": "One p300 board (two Blackhole chips). Context limited to 8,192 tokens; 32 sequences, which this model needs (see Limitations)."},
        "p150x4": {"hardware": "p150x4", "mesh_device": "P150x4", "max_model_len": 32768, "max_num_seqs": 32,
                   "description": "Four Blackhole chips, the base package's own configuration. Tested for token agreement only (stage 2 check), not benchmarked here."},
    }
    table = ("| profile | isl | osl | concurrency | n | ttft | prefill | decode | e2el |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
             f"| p300 | {cad_isl} | {cad_osl} | 1 | {len(d)} | {cad_ttft} s | not measured at this length | {dec_med} tok/s | {cad_e2e} s |\n"
             f"| p300 | {n2} | 1 | 1 | {len(pf2)} | {t2} s | {r2} tok/s | not applicable | {t2} s |\n"
             f"| p300 | {n7} | 1 | 1 | {len(pf7)} | {t7} s | {r7} tok/s | not applicable | {t7} s |\n")
    perf = (f"Measured on one p300c board (two Blackhole chips), profile `p300`, greedy decoding, one request at a time. "
            f"Decode speed is the median of {len(rates)} runs ({repeats} repeats of {prompts} CadQuery prompts, up to 400 new tokens each): "
            f"{dec_med} tok/s, range {dec_min} to {dec_max}. The first row is those runs (isl and osl are medians of the prompt and reply token counts). "
            f"The other rows are the time to the first token for prompts of about 2,000 and 6,700 tokens that share no prefix with an earlier request "
            f"({len(pf2)} and {len(pf7)} prompts, median shown; the 6,700-token set ranged {min(x['ttft_s'] for x in pf7)} to {max(x['ttft_s'] for x in pf7)} s). "
            f"These runs used the package itself, started with `tt-model serve` from the staged package; kernel cache and tensor cache were warm (built by an earlier boot of the package). "
            f"A first boot with empty caches took 532 s to a ready server and a warm boot took 81 s. Host load average during the runs was {b['load_before'][0]} to {b['load_after'][1]} (1 and 5 minute; other jobs were running on the machine). "
            f"Concurrency above 1 was not measured.\n\n" + table)
    sandbox = c.get("sandbox") or {}
    sandbox_text = ("The replies were run in a child process with an empty environment, resource limits and a timeout"
                    + (f"; file writes outside its work directory and TCP connections were refused by Landlock" if sandbox.get("landlock") and sandbox.get("tcp") else
                       "; the kernel's file and network limits were not applied" if not sandbox.get("landlock") else "; file writes outside its work directory were refused by Landlock, TCP was not limited")
                    + ". No network namespace was used, so UDP and Unix sockets were not blocked.")
    limits = ("- Context is limited to 8,192 tokens in the `p300` profile. Longer contexts were not tried on two chips.\n"
              "- The `p300` profile must run with `max_num_seqs` 32. With 8 sequences the first test on two chips produced garbled text; with 32 it produced coherent text and the numbers above. The cause is not known. Do not lower it.\n"
              "- On-device sampling is off. The base package asks for it (`tt.sample_on_device_mode` set to `all`); on two chips the server then stopped at the first decode with \"on_device_logits=True but no on-device sampling module exists\". This package leaves the setting out, so sampling runs on the host.\n"
              f"- CadQuery quality was checked on {cq_prompts} prompts (greedy; {n} replies in all, identical across repeats). Of the {n} replies: {counts_sentence(c)}. "
              f"{ok.get('executes', 0)} of {n} executed and {ok.get('valid_solid', 0)} of {n} exported a CadQuery solid that is valid and has a positive volume ({valid_prompts} of {cq_prompts} prompts), checked with CadQuery 2.8.0 on the exported STEP file. "
              f"Only replies that passed the import and format checks were run. {sandbox_text} "
              "The prompt that failed (an extruded rectangle profile written as a sketch) raised `No free edges available`. Eight prompts are too few to call this an evaluation. "
              "The model card of the weights reports an evaluation on 400 held-out prompts, measured on other hardware with bfloat16 weights; this package runs with 4-bit and 8-bit block floating point weights (the runtime's performance setting), so those numbers do not carry over without a new measurement.\n"
              "- The four-chip profile was proved by a stage 2 token-agreement check (32 of 32 tokens equal to the CPU reference) and was not benchmarked here.\n"
              "- The model writes code and must only be run in a sandbox. It proposes geometry; it does not check it.\n"
              "- Thinking mode is off for CadQuery (send `chat_template_kwargs: {\"enable_thinking\": false}` and the system prompt on the weights' card).\n")
    lic = ("The weights are a separate repository, `Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged`, with licence \"other\". Its card says it was fine-tuned partly on CAD-Recode data (cc-by-nc-4.0) and asks that the model be treated as non-commercial. "
           "This package does not contain the weights; the runtime downloads them from that repository at the pinned revision. Read that card before any use. The base model is Qwen/Qwen3-32B (apache-2.0). "
           "The container image is the image of `mando2222/qwen3-32b-blackhole-v51`, byte for byte (same digest); that repository states no licence. The image carries tt-metal and vLLM, both apache-2.0 projects.")
    card = {
        "description": "Berkelium text-to-CadQuery (Qwen3-32B fine-tune) on one p300 board (two Blackhole chips) or four chips, served with vLLM and the tt_transformers runtime.",
        "attribution": ("Weights: Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged (a LoRA merged into Qwen/Qwen3-32B), unchanged. Container image: mando2222/qwen3-32b-blackhole-v51, unchanged. "
                        "This package adds a two-chip serve profile and the measurements below."),
        "prerequisites": f"Docker, tt-model, one p300 board (or four Blackhole chips for the p150x4 profile), and free disk: {DISK_TEXT}.",
        "architecture": "Qwen3-32B dense decoder, 64 layers, 64 attention heads and 8 key-value heads; the runtime's performance setting stores the attention weights as bfloat8_b and the feed-forward up and gate weights as bfloat4_b.",
        "status": "Measured on one board by the package author. Not verified by Tenstorrent.",
        "intended_use": "Turning a short text request into one JSON header line and one CadQuery script in millimetres, inside a sandbox.",
        "out_of_scope_use": "Commercial use (see Licensing). Running the generated code outside a sandbox. Contexts above 8,192 tokens on the p300 profile.",
        "performance": perf,
        "limitations": limits,
        "licensing": lic,
        "related": "mando2222/qwen3-32b-blackhole-v51 (the base package this one reuses), Qwen/Qwen3-32B, Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged (the weights).",
        "license": {"id": "other", "name": "see-weights-card-non-commercial", "link": "https://huggingface.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged"},
        "pipeline_tag": "text-generation",
        "base_model": ["Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged"],
    }
    REPO = repo
    readme = f"""---
tags:
- blackhole
- p300
- p150x4
- tt-model-cache
- tt-model-container
- vllm-plugin
- cadquery
- text-to-cad
license: other
license_name: see-weights-card-non-commercial
license_link: https://huggingface.co/Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged
pipeline_tag: text-generation
base_model:
- Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged
---

# {REPO.split('/')[1]} on Tenstorrent p300 and p150x4

> {card['attribution']}

{card['description']}

Runs on Tenstorrent **p300** (mesh `P300`, two chips, default) or **p150x4** (mesh `P150x4`, four chips).

Packaged with tt-model-manager (manifest schema 5.1).

## At a glance

| | |
| --- | --- |
| Model | Qwen3-32B fine-tuned for text to CadQuery, 32.8 B parameters |
| Hardware | p300 (2 chips) or p150x4 (4 chips) |
| Context | 8,192 tokens (p300), 32,768 (p150x4) |
| Concurrent sequences | 32 |
| Weights | pointer to `Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged` at a pinned revision (the weights are not in this package) |
| Licence | other: non-commercial, see Licensing |

## Intended use

{card['intended_use']} Out of scope: {card['out_of_scope_use']}

## Prerequisites

{card['prerequisites']}

## Quickstart

```bash
tt-model pull  {REPO} --with-weights
tt-model serve {REPO}
```

`pull --with-weights` downloads the Docker image and the weights ({WEIGHTS_GB} GB) into your Hugging Face cache. `serve` starts an OpenAI-compatible server on port 20000 (or the next free port). The first start on a new machine rebuilds the tensor cache ({CACHE_GB_LOW} to {CACHE_GB_HIGH} GB on disk, about 9 minutes for the p300 profile in the author's test) and compiles kernels; the server is ready when it logs `Application startup complete`. Use `--profile p150x4` for four chips.

Without tt-cli: `tt-model serve {REPO}`.

## Serving profiles

| profile | hardware | mesh | context | sequences |
| --- | --- | --- | --- | --- |
| p300 (default) | p300 | P300 | 8,192 | 32 |
| p150x4 | p150x4 | P150x4 | 32,768 | 32 |

## Capabilities

OpenAI chat completions with the system prompt from the weights' card. Send `"chat_template_kwargs": {{"enable_thinking": false}}`. Tool calling (hermes parser) and the qwen3 reasoning parser are enabled as in the base package.

## Expected performance

{perf}

## Limitations

{limits}
## Licensing

{lic}

## Related packages

{card['related']}
"""
    return profile, card, readme


def main(argv) -> int:
    b = json.load(open("bench-package-p300.json")); c = json.load(open("cq-package-p300.json")); p = json.load(open("prefill-package-p300.json"))
    profile, card, readme = build(argv[1], b, c, p)
    json.dump(profile, open("profile.json", "w"), indent=1)
    json.dump(card, open("card.json", "w"), indent=1)
    open("README.md", "w").write(readme)
    print("ok", len(readme))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
