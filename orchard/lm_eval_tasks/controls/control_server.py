# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A chat server that answers HumanEval and MBPP prompts with the dataset's own solution (mode full or body) or a stub (mode wrong).
Used as a positive and a negative control for the humaneval_chat and mbpp_chat tasks."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datasets import load_dataset
port, mode = int(sys.argv[1]), sys.argv[2]
he = list(load_dataset("openai/openai_humaneval")["test"])
mb = list(load_dataset("google-research-datasets/mbpp", "full")["test"])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        reply = "I could not find the task."
        for d in he:
            if d["prompt"].strip() in user:
                if mode == "full": reply = "Here it is.\n```python\n" + d["prompt"] + d["canonical_solution"] + "```\nHope that helps."
                elif mode == "body": reply = "```python\n" + d["canonical_solution"] + "```"
                else: reply = "```python\n" + d["prompt"] + "    return None\n```"
                break
        else:
            for d in mb:
                if d["text"] in user:
                    reply = "```python\n" + (d["code"] if mode != "wrong" else "def nothing():\n    return 0") + "\n```\n```python\nprint('example')\n```"
                    break
        raw = json.dumps({"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
