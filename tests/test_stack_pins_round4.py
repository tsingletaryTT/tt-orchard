# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 381, Copilot's fourth review: every token prefix is removed, and a name that ends in a line break is not a class name."""
from __future__ import annotations

import json

import pytest

from orchard import skillsfresh, stack_pins as sp
from test_stack_pins import GOOD, build, config, full, names, probe_says, py, repo  # noqa: F401  (py is a fixture)

PREFIXES = ("hf_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_")


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_token_prefix_is_removed_outside_and_inside_url_user_information(prefix):
    token = prefix + "AbCdEf0123456789xyz"
    outside = skillsfresh.scrub_secrets(f"fatal: bad credentials {token} for the request")
    with_user = skillsfresh.scrub_secrets(f"fatal: unable to access 'https://user:{token}@github.com/x/y.git/'")
    token_only = skillsfresh.scrub_secrets(f"fatal: unable to access 'https://{token}@github.com/x/y.git/'")
    for text in (outside, with_user, token_only):
        assert token not in text and "AbCdEf0123456789xyz" not in text
    assert "github.com/x/y.git" in with_user and "bad credentials" in outside               # the rest of the message stays


@pytest.mark.parametrize("prefix", PREFIXES)
def test_every_token_prefix_stays_out_of_the_record_when_a_git_error_carries_it(tmp_path, py, prefix):
    token = prefix + "ZyXwVu9876543210abc"
    m = repo(tmp_path, "tt-metal", down=True)
    src, fake = full(tmp_path, py, metal=m)

    def leaky(argv, t, e=None):
        if "ls-remote" in argv and argv[2] == m[0]:
            return 128, "", f"fatal: {token} and https://u:{token}@h.example/x"
        return fake(argv, t, e)
    assert "9876543210abc" not in json.dumps(build(src, leaky, GOOD()))


@pytest.mark.parametrize("ending", ["\n", "\r", "\r\n", "\0", " ", "\t"])
def test_a_name_with_a_line_break_or_another_trailing_character_is_not_a_class_name(tmp_path, py, ending):
    bad = "LlamaForCausalLM" + ending
    assert sp.CLASS_NAME.fullmatch(bad) is None and sp.CLASS_NAME.fullmatch("LlamaForCausalLM")
    assert sp.MODEL_TYPE.fullmatch("llama" + ending) is None and sp.MODEL_TYPE.fullmatch("llama")
    seen = []
    answer = (0, json.dumps({"importable": True, "version": "5.0.0", "classes": {"Good": True}}), "")
    sp.probe_interpreter(py, ["Good", bad], run=lambda argv, t, e=None: (seen.append(argv), answer)[1], model_type="llama" + ending)
    assert seen[0][3] == "-" and seen[0][4:] == ["Good"]
    # through the config: the entry is rejected (an unchecked finding) and is never reported as a missing class
    model = config(tmp_path, model_type="m", architectures=["Good", bad], transformers_version="4.0.0")
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("5.0.0", {"Good": True}))
    assert sp.H_CLASS not in names(rec["hazards"]) and "transformers_architectures_unchecked" in names(rec["advisories"]) and rec["severity"] == "advisory"
