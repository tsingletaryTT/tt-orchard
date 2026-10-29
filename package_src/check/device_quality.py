"""Send the fixed prompts to the running server (greedy, 32 tokens, top-5 log-probabilities), then a short chat request, a chat request with
thinking off, and one long prompt. Usage: device_quality.py PORT PROMPTS_JSON OUT_JSON"""
import json, sys, time, urllib.request
port, prompts_path, out_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)
res = {"completions": [], "chat": {}, "long_prompt": {}}
for p in json.load(open(prompts_path)):
    o = post("/v1/completions", {"model": "m", "prompt": p, "max_tokens": 32, "temperature": 0, "logprobs": 5, "ignore_eos": True,
                                 "return_tokens_as_token_ids": True})
    ch = o["choices"][0]
    res["completions"].append({"prompt": p, "tokens": [int(t.split(":")[1]) for t in ch["logprobs"]["tokens"]],
                               "top5": [{k.split(":")[1]: v for k, v in d.items()} for d in ch["logprobs"]["top_logprobs"]], "text": ch["text"]})
t0 = time.time()
o = post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "In one sentence, what is a Tenstorrent Blackhole card?"}], "max_tokens": 96, "temperature": 0})
res["chat"]["default"] = {"seconds": round(time.time() - t0, 2), "text": o["choices"][0]["message"].get("content"), "finish": o["choices"][0]["finish_reason"], "usage": o["usage"]}
o = post("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "Name the three primary colors, in one short line."}], "max_tokens": 64, "temperature": 0,
                                   "chat_template_kwargs": {"enable_thinking": False}})
res["chat"]["thinking_off"] = {"text": o["choices"][0]["message"].get("content"), "finish": o["choices"][0]["finish_reason"], "usage": o["usage"]}
# long prompt: a fact at the start, filler, a question at the end (about 3,000 tokens)
filler = " ".join(f"Item {i} of the inventory list is a plain steel bolt." for i in range(260))
long = "The secret code word is PELICAN.\n" + filler + "\nQuestion: what is the secret code word? Answer: The secret code word is"
t0 = time.time()
o = post("/v1/completions", {"model": "m", "prompt": long, "max_tokens": 8, "temperature": 0})
res["long_prompt"] = {"prompt_tokens": o["usage"]["prompt_tokens"], "text": o["choices"][0]["text"], "seconds": round(time.time() - t0, 2)}
json.dump(res, open(out_path, "w"), indent=1)
print("done")
