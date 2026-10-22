# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The benchmark session driver (orchard/bench_session.py): one model, one server, several phases, one summary.

Nothing here touches a chip or the network. The server is a stdlib fake, `lm_eval` is a small script that posts one request through the
proxy and writes a results file in the layout the real harness wrote (tests/fixtures/lm_eval/results_sample.json is a trimmed real one).
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from orchard import bench_session as bs

FIX = Path(__file__).parent / "fixtures"


# ---- the lm_eval command and its environment ---------------------------------------------------------------------------------------

def cmd(task="gsm8k", **kw):
    args = dict(lm_eval="/venv/bin/lm_eval", model="served", base_url="http://127.0.0.1:9/v1/chat/completions", task=task, limit=20,
                concurrency=4, max_gen_toks=8192, output_path=Path("/out/gsm8k"), include_path=Path("/repo/orchard/lm_eval_tasks"))
    args.update(kw)
    return bs.lm_eval_command(**args)


def flag(argv, name):
    return argv[argv.index(name) + 1]


def test_the_command_names_the_chat_api_model_the_proxy_url_and_the_task():
    argv = cmd()
    assert argv[0] == "/venv/bin/lm_eval" and flag(argv, "--model") == "local-chat-completions" and flag(argv, "--tasks") == "gsm8k"
    margs = dict(kv.split("=", 1) for kv in flag(argv, "--model_args").split(","))
    assert margs["model"] == "served" and margs["base_url"] == "http://127.0.0.1:9/v1/chat/completions"
    assert margs["num_concurrent"] == "4" and margs["tokenized_requests"] == "False"
    assert int(margs["timeout"]) >= 1800 and margs["max_retries"] == "1"


def test_the_command_applies_the_chat_template_logs_samples_and_sets_greedy_generation():
    argv = cmd()
    assert "--apply_chat_template" in argv and "--log_samples" in argv
    assert flag(argv, "--gen_kwargs") == "max_gen_toks=8192,temperature=0"
    assert flag(argv, "--output_path") == "/out/gsm8k" and flag(argv, "--limit") == "20"


def test_the_command_points_the_harness_at_the_chat_code_tasks_in_this_repository():
    assert flag(cmd(), "--include_path") == "/repo/orchard/lm_eval_tasks"


# ---- the stop-string override ---------------------------------------------------------------------------------------------------------------
# gsm8k's own stop strings ("Question:", "</s>", "<|im_end|>") also match the reasoning text, so a thinking model that restates "Question:" while
# thinking is stopped with no answer (271 of 1,319 base replies, 46 of AREX-2's). `--stop` replaces the task's list with one string. The harness
# reads --gen_kwargs as comma-separated key=value pairs, so one string is all it can take.

def test_without_a_stop_the_generation_arguments_are_unchanged():
    assert flag(cmd(stop=None), "--gen_kwargs") == "max_gen_toks=8192,temperature=0"
    assert flag(cmd(), "--gen_kwargs") == "max_gen_toks=8192,temperature=0"


def test_a_stop_becomes_the_harnesss_until_argument():
    assert flag(cmd(stop="</s>"), "--gen_kwargs") == "max_gen_toks=8192,temperature=0,until=</s>"


def test_a_stop_keeps_the_other_generation_arguments_in_place():
    kv = dict(p.split("=", 1) for p in flag(cmd(stop="###", max_gen_toks=77), "--gen_kwargs").split(","))
    assert kv == {"max_gen_toks": "77", "temperature": "0", "until": "###"}


def test_a_stop_with_an_equals_sign_survives_the_harnesss_key_value_split():
    kv = dict(p.split("=", 1) for p in flag(cmd(stop="a=b"), "--gen_kwargs").split(","))
    assert kv["until"] == "a=b"


@pytest.mark.parametrize("bad", ["", "a,b", ",", "  "])
def test_a_stop_the_harness_would_split_or_lose_is_refused(bad):
    with pytest.raises(ValueError):
        bs.validate_stop(bad)


@pytest.mark.parametrize("good", ["</s>", "###", "a=b", "<|im_end|>", " </s> "])
def test_a_plain_stop_string_is_accepted_as_it_is(good):
    assert bs.validate_stop(good) == good


def phase(name, suspects):
    return {"name": name, "proxy": {"stop_truncated_suspects": suspects}}


def test_the_total_adds_up_the_phases_and_the_by_phase_map_names_the_ones_with_suspects():
    got = bs.stop_suspects([phase("gsm8k", 3), phase("ifeval", 0), phase("mbpp", 4)])
    assert got["total"] == 7 and got["by_phase"] == {"gsm8k": 3, "mbpp": 4}


def test_a_phase_with_no_proxy_record_is_left_out_and_does_not_stop_the_summary():
    assert bs.stop_suspects([{"name": "bfcl"}, phase("gsm8k", 2)]) == {"total": 2, "by_phase": {"gsm8k": 2}, "note": bs.STOP_SUSPECT_NOTE}
    assert bs.stop_suspects([{"name": "bfcl", "proxy": None}]) == {"total": 0, "by_phase": {}}


def test_the_note_gives_the_option_and_an_example_string():
    assert "--stop '</s>'" in bs.STOP_SUSPECT_NOTE


def test_the_code_phases_use_the_chat_tasks_and_not_the_harnesss_prefill_tasks():
    assert bs.LM_EVAL_TASKS["humaneval"] == "humaneval_chat" and bs.LM_EVAL_TASKS["mbpp"] == "mbpp_chat"
    assert bs.CODE_TASKS == {"humaneval_chat", "mbpp_chat"}


def test_the_tasks_directory_holds_the_task_definitions_the_phases_name():
    for name in ("humaneval_chat", "mbpp_chat"):
        assert (bs.TASKS_DIR / f"{name}.yaml").is_file()
    assert (bs.TASKS_DIR / "code_chat_utils.py").is_file()


def test_no_limit_means_the_whole_task():
    assert "--limit" not in cmd(limit=None)


def test_gsm8k_uses_five_shot_examples_as_turns_and_other_tasks_do_not():
    argv = cmd("gsm8k")
    assert flag(argv, "--num_fewshot") == "5" and "--fewshot_as_multiturn" in argv
    for task in ("humaneval_chat", "ifeval", "mbpp_chat"):
        argv = cmd(task)
        assert "--num_fewshot" not in argv and "--fewshot_as_multiturn" not in argv


def test_code_tasks_confirm_the_unsafe_code_gate_and_other_tasks_do_not():
    assert "--confirm_run_unsafe_code" in cmd("humaneval_chat") and "--confirm_run_unsafe_code" in cmd("mbpp_chat")
    assert "--confirm_run_unsafe_code" not in cmd("gsm8k") and "--confirm_run_unsafe_code" not in cmd("ifeval")


def test_the_environment_has_no_token_and_a_home_inside_the_output_directory(tmp_path, monkeypatch):
    for k, v in {"HF_TOKEN": "x", "HUGGING_FACE_HUB_TOKEN": "x", "GITHUB_TOKEN": "x", "ANTHROPIC_API_KEY": "x", "AWS_SECRET_ACCESS_KEY": "x",
                 "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin", "LANG": "C.UTF-8", "HOME": "/home/someone"}.items():
        monkeypatch.setenv(k, v)
    env = bs.lm_eval_env("humaneval_chat", tmp_path)
    assert not {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "AWS_SECRET_ACCESS_KEY"} & set(env)
    assert "HF_HUB_OFFLINE" not in env                                         # the datasets must be downloadable
    assert env["HOME"].startswith(str(tmp_path)) and env["HF_HOME"].startswith(str(tmp_path))
    assert env["PATH"] == "/usr/bin" and env["LANG"] == "C.UTF-8"
    assert "HF_ALLOW_CODE_EVAL" not in env


def test_no_task_gets_the_hugging_face_code_execution_gate(tmp_path):
    """The chat code tasks score through orchard/lm_eval_tasks/code_sandbox.py. The gate would let Hugging Face's `code_eval` run a model's code in
    the harness's own process, with the operator's files and network (its reliability_guard is not a sandbox), so it is never set."""
    for task in ("gsm8k", "ifeval", "humaneval_chat", "mbpp_chat"):
        assert "HF_ALLOW_CODE_EVAL" not in bs.lm_eval_env(task, tmp_path)
        assert "HF_ALLOW_CODE_EVAL" not in bs.lm_eval_env(task, tmp_path, unsandboxed=True)


def test_the_unsandboxed_variable_is_set_only_when_the_operator_allowed_it_and_only_for_code_tasks(tmp_path):
    assert "ORCHARD_CODE_UNSANDBOXED" not in bs.lm_eval_env("humaneval_chat", tmp_path)
    assert bs.lm_eval_env("humaneval_chat", tmp_path, unsandboxed=True)["ORCHARD_CODE_UNSANDBOXED"] == "1"
    assert bs.lm_eval_env("mbpp_chat", tmp_path, unsandboxed=True)["ORCHARD_CODE_UNSANDBOXED"] == "1"
    assert "ORCHARD_CODE_UNSANDBOXED" not in bs.lm_eval_env("gsm8k", tmp_path, unsandboxed=True)


def test_the_docstring_does_not_claim_more_than_the_environment_does():
    doc = bs.__doc__
    assert "scrubbed environment (no tokens" not in doc
    assert "code_sandbox" in doc and "Landlock" in doc


# ---- re-running a chosen list of items -----------------------------------------------------------------------------------------------

def test_a_list_of_items_becomes_the_harness_samples_argument_and_no_limit():
    argv = cmd("mbpp_chat", limit=None, samples=[7, 3, 9])
    assert json.loads(flag(argv, "--samples")) == {"mbpp_chat": [3, 7, 9]}
    assert "--limit" not in argv


def test_samples_win_over_a_limit_when_both_reach_the_command():
    assert "--limit" not in cmd("mbpp_chat", limit=20, samples=[1])


def test_no_samples_means_no_samples_argument():
    assert "--samples" not in cmd("mbpp_chat", limit=None)


def test_the_session_passes_each_phases_items_and_records_them(harness, tmp_path):
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"gsm8k": [4, 2]}))
    out = tmp_path / "out"
    argv = ["--label", "t", "--url", f"http://127.0.0.1:{harness['server'].port}", "--out-dir", str(out), "--lm-eval", str(harness["lm_eval"]),
            "--phases", "gsm8k", "--items", str(items), "--concurrency", "2", "--quiet-s", "30"]
    assert bs.main(argv) == 0
    run = json.loads((harness["tmp"] / "argv.jsonl").read_text().splitlines()[0])["argv"]
    assert json.loads(run[run.index("--samples") + 1]) == {"gsm8k": [2, 4]}
    s = json.loads((out / "summary.json").read_text())
    assert s["settings"]["items"] == {"gsm8k": [2, 4]} and s["settings"]["limit"] is None


def test_items_and_limit_together_are_refused(harness, tmp_path):
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"gsm8k": [1]}))
    with pytest.raises(SystemExit):
        bs.main(["--label", "t", "--url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "o"), "--phases", "gsm8k", "--items", str(items), "--limit", "2"])


def test_a_phase_with_no_entry_in_the_items_file_is_refused(harness, tmp_path):
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"gsm8k": [1]}))
    with pytest.raises(SystemExit):
        bs.main(["--label", "t", "--url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "o"), "--phases", "gsm8k,ifeval", "--items", str(items)])


def test_the_tool_call_phase_cannot_take_items(harness, tmp_path):
    items = tmp_path / "items.json"
    items.write_text(json.dumps({"bfcl": [1]}))
    with pytest.raises(SystemExit):
        bs.main(["--label", "t", "--url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "o"), "--phases", "bfcl", "--items", str(items), "--bfcl-dir", str(FIX / "bfcl")])


@pytest.mark.parametrize("body", ["[1, 2]", '{"gsm8k": "1,2"}', '{"gsm8k": [1, "2"]}', '{"gsm8k": [true]}', '{"gsm8k": [-1]}', '{"gsm8k": []}', "not json"])
def test_a_malformed_items_file_is_refused(harness, tmp_path, body):
    items = tmp_path / "items.json"
    items.write_text(body)
    with pytest.raises(SystemExit):
        bs.main(["--label", "t", "--url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "o"), "--phases", "gsm8k", "--items", str(items)])


# ---- reading the harness's results --------------------------------------------------------------------------------------------------

def sample_dir(tmp_path):
    d = tmp_path / "res" / "served"
    d.mkdir(parents=True)
    (d / "results_2026-10-19T16-19-54.808021.json").write_text((FIX / "lm_eval" / "results_sample.json").read_text())
    return tmp_path / "res"


def test_results_give_each_metric_with_its_filter_value_stderr_and_the_sample_counts(tmp_path):
    got = bs.parse_lm_eval_results(sample_dir(tmp_path))
    gsm = got["tasks"]["gsm8k"]
    assert gsm["version"] == 3.0 and gsm["n_samples"] == {"original": 1319, "effective": 5}
    assert {"metric": "exact_match", "filter": "flexible-extract", "value": 0.0, "stderr": 0.0} in gsm["metrics"]
    ife = got["tasks"]["ifeval"]
    strict = next(m for m in ife["metrics"] if m["metric"] == "prompt_level_strict_acc")
    assert strict["filter"] == "none" and strict["value"] == 0.2 and strict["stderr"] == 0.2
    assert got["lm_eval_version"] == "0.4.13" and got["total_evaluation_time_seconds"] is not None


def test_a_stderr_of_n_a_is_none(tmp_path):
    ife = bs.parse_lm_eval_results(sample_dir(tmp_path))["tasks"]["ifeval"]
    inst = next(m for m in ife["metrics"] if m["metric"] == "inst_level_strict_acc")
    assert inst["stderr"] is None and inst["value"] == 0.25


def test_name_and_alias_and_sample_len_are_not_metrics(tmp_path):
    for task in bs.parse_lm_eval_results(sample_dir(tmp_path))["tasks"].values():
        assert not {"name", "alias", "sample_len"} & {m["metric"] for m in task["metrics"]}


def test_no_metric_is_a_standard_error_entry_and_text_values_are_not_metrics(tmp_path):
    root = sample_dir(tmp_path)
    path = next((root / "served").glob("results_*.json"))
    data = json.loads(path.read_text())
    data["results"]["gsm8k"]["note,none"] = "a text value"
    path.write_text(json.dumps(data))
    for task in bs.parse_lm_eval_results(root)["tasks"].values():
        assert not [m for m in task["metrics"] if m["metric"].endswith("_stderr")]
        assert all(isinstance(m["value"], (int, float)) for m in task["metrics"])
    assert "note" not in {m["metric"] for m in bs.parse_lm_eval_results(root)["tasks"]["gsm8k"]["metrics"]}


def test_the_newest_results_file_is_read_when_there_are_several(tmp_path):
    root = sample_dir(tmp_path)
    old = root / "served" / "results_2026-10-19T10-00-00.000000.json"
    data = json.loads((FIX / "lm_eval" / "results_sample.json").read_text())
    data["lm_eval_version"] = "old"
    old.write_text(json.dumps(data))
    assert bs.parse_lm_eval_results(root)["lm_eval_version"] == "0.4.13"


def test_no_results_file_is_an_error_that_names_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        bs.parse_lm_eval_results(tmp_path)


# ---- a whole session against a fake server and a fake harness ------------------------------------------------------------------------

class FakeServer:
    def __init__(self):
        self.seen = []
        self.fail = False
        self.reasoning_only = False         # answer with reasoning text and no content, the way a stop string inside the thinking leaves a reply
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, payload):
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send({"data": [{"id": "fake-served"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append(body)
                if outer.fail:
                    raw = b'{"error": {"message": "down"}}'
                    self.send_response(500)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                if body.get("tools"):
                    fn = body["tools"][0]["function"]
                    args = {"base": 10, "height": 5} if "triangle" in fn["name"] else {"x": 4, "y": 5} if "hypot" in fn["name"] else {"number": 5}
                    msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": fn["name"], "arguments": json.dumps(args)}}]}
                elif outer.reasoning_only:
                    msg = {"role": "assistant", "content": None, "reasoning_content": "the user pattern is Question: ..."}
                else:
                    msg = {"role": "assistant", "content": "ok"}
                self._send({"choices": [{"finish_reason": "stop", "message": msg}], "usage": {"prompt_tokens": 5, "completion_tokens": 4}})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


FAKE_LM_EVAL = r'''#!%(python)s
import json, sys, os, http.client, urllib.parse
a = sys.argv[1:]
def flag(n): return a[a.index(n) + 1]
margs = dict(kv.split("=", 1) for kv in flag("--model_args").split(","))
u = urllib.parse.urlsplit(margs["base_url"])
c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
c.request("POST", u.path, json.dumps({"model": margs["model"], "messages": [{"role": "user", "content": "q"}]}), {"Content-Type": "application/json"})
c.getresponse().read()
here = os.path.dirname(os.path.abspath(__file__))
open(here + "/argv.jsonl", "a").write(json.dumps({"argv": a, "home": os.environ.get("HOME"), "token": os.environ.get("HF_TOKEN"),
                                                  "unsandboxed": os.environ.get("ORCHARD_CODE_UNSANDBOXED"), "gate": os.environ.get("HF_ALLOW_CODE_EVAL")}) + "\n")
out = flag("--output_path")
task = flag("--tasks")
if os.path.exists(here + "/fail"):
    print("boom"); sys.exit(1)
os.makedirs(out + "/" + margs["model"], exist_ok=True)
json.dump({"results": {task: {"alias": task, "acc,none": 0.5, "acc_stderr,none": 0.1}}, "versions": {task: 1.0}, "n-samples": {task: {"original": 100, "effective": 2}},
           "n-shot": {task: 0}, "config": {"model": "local-chat-completions", "limit": 2, "gen_kwargs": {}}, "lm_eval_version": "9.9.9",
           "model_name_sanitized": margs["model"], "total_evaluation_time_seconds": "1.5"}, open(out + "/" + margs["model"] + "/results_2026-01-01T00-00-00.000000.json", "w"))
'''


@pytest.fixture
def harness(tmp_path, monkeypatch):
    fake = tmp_path / "lm_eval"
    fake.write_text(FAKE_LM_EVAL % {"python": sys.executable})
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_reach_the_child")
    server = FakeServer()
    yield {"lm_eval": fake, "server": server, "tmp": tmp_path}
    server.close()


def run_session(h, phases, **kw):
    out = h["tmp"] / "out"
    argv = ["--label", "t", "--url", f"http://127.0.0.1:{h['server'].port}", "--out-dir", str(out), "--lm-eval", str(h["lm_eval"]), "--phases", phases,
            "--bfcl-dir", str(FIX / "bfcl"), "--limit", "2", "--concurrency", "2", "--quiet-s", "30"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    code = bs.main(argv)
    return code, out


def test_a_session_runs_each_phase_and_writes_one_summary(harness):
    code, out = run_session(harness, "gsm8k,bfcl")
    assert code == 0
    s = json.loads((out / "summary.json").read_text())
    assert [p["name"] for p in s["phases"]] == ["gsm8k", "bfcl"]
    g, b = s["phases"]
    assert g["ok"] is True and g["scores"]["tasks"]["gsm8k"]["metrics"][0]["value"] == 0.5
    assert g["proxy"]["requests"] == 1 and g["proxy"]["completion_tokens_total"] == 4
    assert b["ok"] is True and b["scores"]["simple"]["n"] == 2 and b["scores"]["multiple"]["n"] == 2 and b["scores"]["simple"]["score"] == 1.0
    assert s["served_name"] == "fake-served" and s["label"] == "t"


def test_the_summary_records_what_makes_two_runs_comparable(harness):
    code, out = run_session(harness, "gsm8k", thinking="off", max_gen_toks=1234)
    s = json.loads((out / "summary.json").read_text())
    assert s["settings"]["thinking"] == "off" and s["settings"]["limit"] == 2 and s["settings"]["concurrency"] == 2
    assert s["settings"]["max_gen_toks"] == 1234 and s["settings"]["seed"] == 0
    assert s["lm_eval_version"] == "9.9.9"


def harness_argv(h):
    return json.loads((h["tmp"] / "argv.jsonl").read_text().splitlines()[0])["argv"]


def test_a_stop_reaches_the_harness_and_is_recorded_in_the_settings(harness):
    code, out = run_session(harness, "gsm8k", stop="</s>")
    assert code == 0
    assert flag(harness_argv(harness), "--gen_kwargs") == "max_gen_toks=8192,temperature=0,until=</s>"
    assert json.loads((out / "summary.json").read_text())["settings"]["stop"] == "</s>"


def test_a_run_with_no_stop_records_none_and_leaves_the_tasks_own_stop_strings(harness):
    code, out = run_session(harness, "gsm8k")
    assert "until" not in flag(harness_argv(harness), "--gen_kwargs")
    assert json.loads((out / "summary.json").read_text())["settings"]["stop"] is None


@pytest.mark.parametrize("bad", ["", "a,b"])
def test_a_stop_the_harness_would_split_is_refused_before_anything_starts(tmp_path, bad):
    with pytest.raises(SystemExit):
        bs.main(["--label", "t", "--url", "http://127.0.0.1:9", "--out-dir", str(tmp_path / "o"), "--phases", "gsm8k", "--stop", bad])
    assert not (tmp_path / "o" / "summary.json").exists()


def test_the_summary_counts_replies_that_look_cut_off_by_a_stop_string(harness):
    harness["server"].reasoning_only = True
    code, out = run_session(harness, "gsm8k")
    s = json.loads((out / "summary.json").read_text())
    assert s["phases"][0]["proxy"]["stop_truncated_suspects"] == 1
    assert s["stop_truncated_suspects"]["total"] == 1 and s["stop_truncated_suspects"]["by_phase"] == {"gsm8k": 1}
    note = s["stop_truncated_suspects"]["note"]
    assert "stop string" in note and "--stop" in note


def test_a_run_with_no_such_replies_says_none_and_has_no_note(harness):
    code, out = run_session(harness, "gsm8k")
    s = json.loads((out / "summary.json").read_text())
    assert s["stop_truncated_suspects"] == {"total": 0, "by_phase": {}}


def test_thinking_off_reaches_the_server_for_the_harness_and_for_the_tool_call_phase(harness):
    run_session(harness, "gsm8k,bfcl", thinking="off")
    assert harness["server"].seen and all(b["chat_template_kwargs"] == {"enable_thinking": False} for b in harness["server"].seen)


def test_the_default_thinking_setting_adds_nothing_to_requests(harness):
    run_session(harness, "gsm8k,bfcl")
    assert all("chat_template_kwargs" not in b for b in harness["server"].seen)


def test_the_harness_child_gets_the_server_through_the_proxy_not_directly(harness):
    run_session(harness, "gsm8k")
    logged = json.loads((harness["tmp"] / "argv.jsonl").read_text().splitlines()[0])
    margs = dict(kv.split("=", 1) for kv in logged["argv"][logged["argv"].index("--model_args") + 1].split(","))
    assert f":{harness['server'].port}/" not in margs["base_url"] and margs["model"] == "fake-served"
    assert logged["token"] is None and logged["home"].startswith(str(harness["tmp"] / "out"))


def test_a_failing_harness_marks_the_phase_failed_and_the_session_goes_on(harness):
    (harness["tmp"] / "fail").write_text("")
    code, out = run_session(harness, "gsm8k,bfcl")
    s = json.loads((out / "summary.json").read_text())
    g, b = s["phases"]
    assert g["ok"] is False and "exit" in g["problem"] and b["ok"] is True
    assert code == 1


def test_a_tool_call_phase_whose_server_keeps_failing_is_marked_failed_with_the_reason(harness):
    harness["server"].fail = True
    code, out = run_session(harness, "bfcl", limit=3)
    b = json.loads((out / "summary.json").read_text())["phases"][0]
    assert b["ok"] is False and "errors in a row" in b["problem"] and code == 1


def test_an_unknown_phase_is_refused_before_anything_runs(harness):
    with pytest.raises(SystemExit):
        run_session(harness, "gsm8k,nonsense")
    assert harness["server"].seen == []


def test_the_tool_call_phase_writes_its_own_result_files(harness):
    code, out = run_session(harness, "bfcl")
    for name in ("bfcl-simple.json", "bfcl-multiple.json"):
        data = json.loads((out / "bfcl" / name).read_text())
        assert data["n"] == 2 and data["settings"]["dataset"]


def test_a_stalled_harness_is_stopped_and_reported(harness, tmp_path):
    slow = tmp_path / "slow_lm_eval"
    slow.write_text("#!/bin/sh\nsleep 600\n")
    slow.chmod(slow.stat().st_mode | stat.S_IEXEC)
    harness["lm_eval"] = slow
    t0 = time.monotonic()
    code, out = run_session(harness, "gsm8k", quiet_s=3, phase_backstop_s=60)
    s = json.loads((out / "summary.json").read_text())
    assert time.monotonic() - t0 < 40 and s["phases"][0]["ok"] is False and "stalled" in s["phases"][0]["problem"]


# ---- starting and stopping the server --------------------------------------------------------------------------------------------------

FAKE_RUN_SH = r'''#!/bin/bash
# a stand-in for a bundle's run.sh: serves the fake API on --port and records its pid
port="$2"
echo $$ > "$PIDFILE"
exec %(python)s - "$port" <<'PY'
import json, sys
LINE = "(EngineCore pid=1) 2026-10-19 18:13:51.463 | INFO     | m:end:907 - [dflash2-serve] slot 0 session: 8 iters, 20 committed tokens, accept 1.50/7 -> 2.50 tok/iter, 40.0 tok/s in-model (incl. seed); histogram [2, 3, 1, 1, 1, 0, 0, 0]"
print(LINE, flush=True)                                   # before any phase: it must not be counted for a phase
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, p, status=200):
        raw = json.dumps(p).encode(); self.send_response(status); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def do_GET(self):
        self._send({"data": [{"id": "from-run-sh"}]} if self.path.startswith("/v1/models") else {"ok": True})
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"])); print(LINE, flush=True); self._send({"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
'''


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    run_sh = tmp_path / "run.sh"
    run_sh.write_text(FAKE_RUN_SH % {"python": sys.executable})
    monkeypatch.setenv("PIDFILE", str(tmp_path / "server.pid"))
    cache = tmp_path / "cache"
    fake = tmp_path / "lm_eval"
    fake.write_text(FAKE_LM_EVAL % {"python": sys.executable})
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return {"run_sh": run_sh, "cache": cache, "tmp": tmp_path, "lm_eval": fake}


def session_argv(b, **extra):
    argv = ["--label", "t", "--model-id", "org/model@abc", "--run-sh", str(b["run_sh"]), "--model-dir", str(b["tmp"] / "model"), "--tt-cache", str(b["cache"]),
            "--hf-home", str(b["tmp"] / "hf"), "--port", str(free_port()), "--out-dir", str(b["tmp"] / "out"), "--lm-eval", str(b["lm_eval"]),
            "--phases", "bfcl", "--bfcl-dir", str(FIX / "bfcl"), "--limit", "1", "--health-timeout", "30"]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


def pid_gone(pid):
    try:
        os.killpg(pid, 0)
        return False
    except ProcessLookupError:
        return True


def test_the_server_is_started_used_and_stopped_with_its_whole_process_group(bundle):
    code = bs.main(session_argv(bundle))
    pid = int((bundle["tmp"] / "server.pid").read_text())
    assert code == 0 and pid_gone(pid)
    s = json.loads((bundle["tmp"] / "out" / "summary.json").read_text())
    assert s["served_name"] == "from-run-sh" and s["server_ready_s"] is not None and s["model_id"] == "org/model@abc"
    assert s["run_sh_sha256"] and (bundle["tmp"] / "out" / "server.log").exists()


def test_the_server_is_stopped_even_when_a_phase_raises(bundle, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("phase blew up")
    monkeypatch.setattr(bs, "run_bfcl_phase", boom)
    with pytest.raises(RuntimeError, match="blew up"):
        bs.main(session_argv(bundle))
    assert pid_gone(int((bundle["tmp"] / "server.pid").read_text()))


def test_a_tensor_cache_with_no_marker_is_refused_before_the_server_starts(bundle):
    bundle["cache"].mkdir()
    (bundle["cache"] / "something").write_text("x")
    with pytest.raises(SystemExit):
        bs.main(session_argv(bundle))
    assert not (bundle["tmp"] / "server.pid").exists()


def test_an_empty_cache_gets_a_marker_naming_the_model(bundle):
    bs.main(session_argv(bundle))
    assert (bundle["cache"] / ".orchard-model").read_text() == "org/model@abc"


def test_the_server_gets_the_cache_hf_home_and_the_weights_directory_in_its_environment(bundle, monkeypatch):
    envlog = bundle["tmp"] / "env.txt"
    bundle["run_sh"].write_text(bundle["run_sh"].read_text().replace('echo $$ > "$PIDFILE"', f'echo $$ > "$PIDFILE"; env > {envlog}'))
    bs.main(session_argv(bundle))
    env = dict(line.split("=", 1) for line in envlog.read_text().splitlines() if "=" in line)
    assert env["TT_CACHE_PATH"] == str(bundle["cache"]) and env["TT_CACHE_HOME"] == str(bundle["cache"])
    assert env["HF_HOME"] == str(bundle["tmp"] / "hf") and env["MODEL_WEIGHTS_DIR"] == str(bundle["tmp"] / "model") and env["HF_MODEL"] == str(bundle["tmp"] / "model")
    assert env["HF_HUB_OFFLINE"] == "1"


def test_the_operators_hugging_face_token_does_not_reach_the_server(bundle, monkeypatch):
    envlog = bundle["tmp"] / "env2.txt"
    bundle["run_sh"].write_text(bundle["run_sh"].read_text().replace('echo $$ > "$PIDFILE"', f'echo $$ > "$PIDFILE"; env > {envlog}'))
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secret_value")
    bs.main(session_argv(bundle))
    text = envlog.read_text()
    assert "hf_secret_value" not in text and "HF_TOKEN" not in text


def test_a_server_that_exits_before_it_is_healthy_is_reported_and_stops_the_session(bundle):
    bundle["run_sh"].write_text("#!/bin/bash\necho 'engine failed to start' >&2\nexit 3\n")
    with pytest.raises(SystemExit):
        bs.main(session_argv(bundle, health_timeout=10))


def test_the_seed_is_recorded_and_picks_which_tool_call_items_are_used(harness):
    from orchard import toolcall_bench as tb
    pool = tb.load_items(FIX / "bfcl" / "simple_questions.jsonl", FIX / "bfcl" / "simple_answers.jsonl")
    base = tb.choose(pool, limit=1, seed=0)[0]["id"]
    seed = next(sd for sd in range(1, 200) if tb.choose(pool, limit=1, seed=sd)[0]["id"] != base)       # a seed that picks a different item
    code, out = run_session(harness, "bfcl", seed=seed, limit=1)
    assert json.loads((out / "summary.json").read_text())["settings"]["seed"] == seed
    got = [r["id"] for r in json.loads((out / "bfcl" / "bfcl-simple.json").read_text())["items"]]
    assert got == [tb.choose(pool, limit=1, seed=seed)[0]["id"]] and got != [base]


def test_a_phase_that_outlives_its_backstop_is_stopped_and_reported(harness, tmp_path):
    slow = tmp_path / "slow_lm_eval"
    slow.write_text("#!/bin/sh\nsleep 600\n")
    slow.chmod(slow.stat().st_mode | stat.S_IEXEC)
    harness["lm_eval"] = slow
    t0 = time.monotonic()
    code, out = run_session(harness, "gsm8k", quiet_s=600, phase_backstop_s=3)
    s = json.loads((out / "summary.json").read_text())
    assert time.monotonic() - t0 < 40 and s["phases"][0]["ok"] is False and "backstop" in s["phases"][0]["problem"]


def test_the_tool_call_phase_without_a_data_directory_is_refused_before_anything_runs(harness):
    out = harness["tmp"] / "out"
    with pytest.raises(SystemExit):
        bs.main(["--label", "t", "--url", f"http://127.0.0.1:{harness['server'].port}", "--out-dir", str(out), "--phases", "bfcl"])
    assert harness["server"].seen == [] and not (out / "summary.json").exists()


def test_a_session_gives_the_harness_the_task_directory_of_this_repository(harness):
    run_session(harness, "gsm8k")
    logged = json.loads((harness["tmp"] / "argv.jsonl").read_text().splitlines()[0])
    argv = logged["argv"]
    assert argv[argv.index("--include_path") + 1] == str(bs.TASKS_DIR) and bs.TASKS_DIR.is_dir()


# ---- the drafter's acceptance, read from the server's own log ---------------------------------------------------------------------------

SAMPLE = (FIX / "dflash2_serve_sample.log").read_text()


def test_each_session_line_is_read_and_the_totals_are_weighted_by_iterations():
    got = bs.drafter_stats(SAMPLE)
    assert got["sessions"] == 5 and got["iters"] == 4 + 1 + 16 + 65 + 28 and got["committed_tokens"] == 3 + 0 + 103 + 237 + 178
    assert got["tokens_per_iter"] == pytest.approx(521 / 114)
    assert got["histogram"] == [sum(h) for h in zip(*[[4, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 2, 1, 1, 0, 0, 11],
                                                       [19, 12, 7, 6, 2, 5, 2, 12], [1, 2, 1, 2, 2, 3, 1, 16]])]
    accepted = sum(i * c for i, c in enumerate(got["histogram"]))
    assert got["accepted_per_iter"] == pytest.approx(accepted / 114) and got["draft_length"] == 7
    assert got["accept_rate"] == pytest.approx(accepted / (114 * 7))


def test_the_median_in_model_speed_is_over_sessions_that_committed_tokens():
    got = bs.drafter_stats(SAMPLE)
    assert got["tok_s_in_model_median"] == pytest.approx(40.1)                     # the middle two of 3.5, 36.3, 43.9, 44.4; the 0.0 session is left out


def test_text_with_no_session_lines_gives_none():
    assert bs.drafter_stats("") is None and bs.drafter_stats("INFO nothing here\nwarm sweep: 4 seeds") is None


def test_other_lines_around_the_session_lines_are_ignored():
    got = bs.drafter_stats("noise\n" + SAMPLE.splitlines()[2] + "\nmore noise\n")
    assert got["sessions"] == 1 and got["iters"] == 16 and got["committed_tokens"] == 103


def test_a_line_that_is_cut_off_is_skipped_not_guessed():
    line = SAMPLE.splitlines()[2]
    assert bs.drafter_stats(line[: line.index("histogram")]) is None


def test_a_phase_counts_only_the_drafter_lines_written_while_it_ran(bundle):
    bs.main(session_argv(bundle, limit=1))
    s = json.loads((bundle["tmp"] / "out" / "summary.json").read_text())
    d = s["phases"][0]["drafter"]
    assert d["sessions"] == 2 and d["iters"] == 16 and d["committed_tokens"] == 40       # two requests, not the line written at startup


def test_a_session_against_a_running_server_has_no_drafter_section(harness):
    code, out = run_session(harness, "bfcl")
    assert json.loads((out / "summary.json").read_text())["phases"][0]["drafter"] is None


def _line(iters, committed, k, hist, speed=40.0):
    return (f"[dflash2-serve] slot 0 session: {iters} iters, {committed} committed tokens, accept 1.00/{k} -> 2.00 tok/iter, {speed} tok/s in-model (incl. seed); "
            f"histogram [{', '.join(str(h) for h in hist)}]\n")


def test_a_line_whose_histogram_has_the_wrong_number_of_counts_is_skipped():
    text = _line(8, 20, 7, [8, 0, 0, 0, 0, 0, 0]) + _line(8, 20, 7, [2, 3, 1, 1, 1, 0, 0, 0])
    got = bs.drafter_stats(text)
    assert got["sessions"] == 1 and got["iters"] == 8


def test_a_line_whose_histogram_does_not_add_up_to_its_iterations_is_skipped():
    text = _line(9, 20, 7, [2, 3, 1, 1, 1, 0, 0, 0]) + _line(8, 20, 7, [2, 3, 1, 1, 1, 0, 0, 0])
    got = bs.drafter_stats(text)
    assert got["sessions"] == 1 and got["iters"] == 8


def test_a_line_with_a_different_draft_length_from_the_first_is_skipped_so_totals_stay_comparable():
    text = _line(8, 20, 7, [2, 3, 1, 1, 1, 0, 0, 0]) + _line(4, 9, 3, [1, 1, 1, 1])
    got = bs.drafter_stats(text)
    assert got["sessions"] == 1 and got["draft_length"] == 7 and len(got["histogram"]) == 8


# ---- the code phases need the sandbox (review finding 6) -----------------------------------------------------------------------------

SANDBOX_OK = {"ok": True, "detail": "every probe action was refused", "seen": {}}
SANDBOX_BAD = {"ok": False, "detail": "this kernel has no Landlock", "seen": None}


@pytest.fixture
def sandbox(monkeypatch):
    state = {"result": SANDBOX_OK, "calls": 0}

    def check():
        state["calls"] += 1
        return state["result"]
    monkeypatch.setattr(bs, "code_sandbox_check", check)
    return state


def logged(harness):
    return [json.loads(l) for l in (harness["tmp"] / "argv.jsonl").read_text().splitlines()]


def test_a_code_phase_is_refused_before_anything_runs_when_the_sandbox_is_unavailable(harness, sandbox):
    sandbox["result"] = SANDBOX_BAD
    with pytest.raises(SystemExit) as exc:
        run_session(harness, "gsm8k,humaneval")
    assert exc.value.code == 2 and harness["server"].seen == []
    assert not (harness["tmp"] / "argv.jsonl").exists()


def test_the_refusal_names_the_reason_and_the_flag(harness, sandbox, capsys):
    sandbox["result"] = SANDBOX_BAD
    with pytest.raises(SystemExit):
        run_session(harness, "mbpp")
    err = capsys.readouterr().err
    assert "no Landlock" in err and "--allow-unsandboxed-code" in err


def test_the_operators_flag_lets_the_code_phase_run_without_the_sandbox_and_the_summary_says_so(harness, sandbox):
    sandbox["result"] = SANDBOX_BAD
    code, out = run_session(harness, "humaneval", allow_unsandboxed_code=True)
    s = json.loads((out / "summary.json").read_text())
    assert code == 0 and s["code_sandbox"]["mode"] == "none" and "no Landlock" in s["code_sandbox"]["reason"]
    assert logged(harness)[0]["unsandboxed"] == "1" and logged(harness)[0]["gate"] is None


def test_a_working_sandbox_is_recorded_and_the_child_is_not_told_to_skip_it(harness, sandbox):
    code, out = run_session(harness, "humaneval,mbpp")
    s = json.loads((out / "summary.json").read_text())
    assert code == 0 and s["code_sandbox"]["mode"] == "landlock+seccomp"
    assert all(e["unsandboxed"] is None and e["gate"] is None for e in logged(harness))


def test_a_session_with_no_code_phase_does_not_check_the_sandbox(harness, sandbox):
    sandbox["result"] = SANDBOX_BAD
    code, out = run_session(harness, "gsm8k,bfcl")
    assert code == 0 and sandbox["calls"] == 0
    assert "code_sandbox" not in json.loads((out / "summary.json").read_text())


def test_the_real_check_is_the_one_in_the_task_directory():
    """The name the session calls must reach code_sandbox.self_check, not a stub of its own."""
    import inspect
    assert "self_check" in inspect.getsource(bs.code_sandbox_check)


# ---- a phase whose requests failed is not ok --------------------------------------------------------------------------------------

def test_a_harness_phase_with_failed_server_requests_is_flagged_even_though_the_harness_exited_zero(harness):
    harness["server"].fail = True
    code, out = run_session(harness, "gsm8k")
    g = json.loads((out / "summary.json").read_text())["phases"][0]
    assert g["proxy"]["errors"] == 1
    assert g["ok"] is False and "1 of 1 requests" in g["problem"] and code == 1
    assert g["scores"]["tasks"]["gsm8k"], "the scores are kept, so the reader can see what the failures did to them"


def test_a_phase_with_no_failed_requests_stays_ok(harness):
    code, out = run_session(harness, "gsm8k")
    assert json.loads((out / "summary.json").read_text())["phases"][0]["ok"] is True and code == 0
