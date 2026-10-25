# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The screenshot script's credential check (review of PR 71): visible page text is read for credentials before a picture is published."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("site_screenshots", Path(__file__).resolve().parent.parent / "scripts" / "site_screenshots.py")
sh = importlib.util.module_from_spec(SPEC)
try:
    SPEC.loader.exec_module(sh)
except ImportError:                  # playwright is not installed here: the script imports it at the top
    import sys, types
    stub = types.ModuleType("playwright.sync_api")
    stub.expect = stub.sync_playwright = None
    sys.modules.setdefault("playwright", types.ModuleType("playwright"))
    sys.modules["playwright.sync_api"] = stub
    SPEC.loader.exec_module(sh)


def test_tokens_and_keys_in_the_page_text_are_hits():
    for text in ("hf_" + "A" * 36, "ghp_" + "b" * 40, "sk-" + "x" * 24, "AKIA" + "A" * 16, "-----BEGIN RSA PRIVATE KEY-----",
                 "Authorization: Basic abc", "api_key: x", "the password is x", "token=abc", 'token": "abc'):
        assert sh.credential_hits(text), text


def test_ordinary_page_text_is_clean():
    page = "RUNNING huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated stage 2 functional decoder the tokenizer has 248320 tokens operator accepted visible credentials"
    assert sh.credential_hits(page) == []
