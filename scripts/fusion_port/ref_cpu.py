import json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
d = sys.argv[1]; prompt = open(sys.argv[2]).read()
tok = AutoTokenizer.from_pretrained(d)
try:
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.bfloat16)
except Exception as e:
    from transformers import Qwen3_5MoeForConditionalGeneration as C
    print("fallback", type(e).__name__, file=sys.stderr); m = C.from_pretrained(d, dtype=torch.bfloat16)
m.eval()
ids = tok(prompt, return_tensors="pt").input_ids
out = []; lp = []; cur = ids
with torch.no_grad():
    for _ in range(int(sys.argv[3])):
        lg = m(input_ids=cur).logits[0, -1].float()
        t = int(lg.argmax()); out.append(t)
        top = torch.topk(torch.log_softmax(lg, -1), 5)
        lp.append({"top": top.indices.tolist(), "lp": [round(x, 4) for x in top.values.tolist()]})
        cur = torch.cat([cur, torch.tensor([[t]])], 1)
print("REF " + json.dumps({"prompt_ids": ids[0].tolist(), "tokens": out, "top5": lp, "text": tok.decode(out)}))
