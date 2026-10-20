# Chat code tasks for lm-evaluation-harness

`humaneval_chat` and `mbpp_chat` score a chat model behind the chat completions API. The harness's own `humaneval_instruct` and `mbpp_instruct` cannot:
they expect the assistant's reply to continue a prefilled turn (`gen_prefix`), which the API does not support, and `humaneval_instruct` stops generation at
`\nclass`, `\ndef`, `\n#`, `\nif` and `\nprint`, so a reply that writes the whole function ends at once. On 2026-10-19 that gave `bytkim/Qwen3.8-27B-pi` 0.0
pass@1 on 20 HumanEval items with 12 of the 20 stored replies empty.

These tasks ask for the whole function in one code block, stop at no string, and build the program from the reply (`code_chat_utils.py`). Use them with
`--include_path orchard/lm_eval_tasks`; `orchard/bench_session.py` does.

## Where the replies run

The score is computed by `code_chat_utils.py`, which runs each reply in `code_sandbox.py`: a child process that limits itself with Landlock and seccomp to a
scratch directory, no sockets and no `/proc`, so a reply cannot read the operator's token file or reach the network. Hugging Face's `code_eval` is not used
(its `reliability_guard` is not a sandbox, and the harness needs `HF_ALLOW_CODE_EVAL` for it). `python3 code_sandbox.py` runs the self-check and prints
whether the kernel enforced every restriction. When the sandbox cannot start, scoring raises an error. `ORCHARD_CODE_UNSANDBOXED=1` runs replies without it;
`bench_session --allow-unsandboxed-code` sets that.

## Check the instrument before reading a score

`controls/control_server.py` answers HumanEval and MBPP prompts from the datasets' own solutions (`full`: the whole function, `body`: only the body) or with a
stub (`wrong`). Run lm-eval against it (datasets download; use a scratch `HF_HOME` and no token):

    python3 controls/control_server.py PORT full &     # also body, wrong
    lm_eval --model local-chat-completions \
      --model_args model=ctl,base_url=http://127.0.0.1:PORT/v1/chat/completions,num_concurrent=4,max_retries=1,tokenized_requests=False \
      --tasks humaneval_chat,mbpp_chat --limit 40 --apply_chat_template --gen_kwargs max_gen_toks=512,temperature=0 \
      --confirm_run_unsafe_code --include_path orchard/lm_eval_tasks

Result on 2026-10-19 (lm-eval 0.4.13, 40 items each): `full` 1.0 and 1.0, `body` 1.0 and 1.0, `wrong` 0.0 and 0.0.

The server's replies must carry `"index": 0` in each choice. Without it lm-eval turns the reply into an empty string without an error and every score is 0.
