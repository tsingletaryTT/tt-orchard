"""CPU reference for the quality check: the Hugging Face model in bfloat16, greedy, 32 tokens per prompt, with the top-5 log-probabilities at each
step. Run it with the model directory made by fetch_weights.py (`architectures` is reset to the multimodal class here, so the class that
matches the weights loads). Usage: ref_cpu.py MODELDIR PROMPTS_JSON OUT_JSON [N_TOKENS]"""
import json, sys, torch
from transformers import AutoTokenizer, Qwen3_5MoeForConditionalGeneration
d, prompts_path, out_path = sys.argv[1:4]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 32
tok = AutoTokenizer.from_pretrained(d)
m = Qwen3_5MoeForConditionalGeneration.from_pretrained(d, dtype=torch.bfloat16).eval()
res = []
for p in json.load(open(prompts_path)):
    ids = tok(p, return_tensors="pt").input_ids
    out, top, cache, cur = [], [], None, ids
    with torch.no_grad():
        for _ in range(n):
            o = m(input_ids=cur, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            lg = o.logits[0, -1].float()
            t = int(lg.argmax())
            tp = torch.topk(torch.log_softmax(lg, -1), 5)
            out.append(t); top.append({"ids": tp.indices.tolist(), "logprobs": [round(x, 4) for x in tp.values.tolist()]})
            cur = torch.tensor([[t]])
    res.append({"prompt": p, "prompt_tokens": ids.shape[1], "tokens": out, "top5": top, "text": tok.decode(out)})
    print("done", len(res), flush=True)
json.dump(res, open(out_path, "w"))
