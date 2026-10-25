# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The screenshot script's credential check (review of PR 71): visible page text is read for credentials before a picture is published."""
import importlib.util
import sys
import types
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("site_screenshots", Path(__file__).resolve().parent.parent / "scripts" / "site_screenshots.py")
sh = importlib.util.module_from_spec(SPEC)


def _load():
    """The script imports playwright at the top. Where it is missing, load the script with a stand-in module for the length of the import only:
    leaving a stand-in in sys.modules would make the browser tests (which skip when playwright is missing) fail instead."""
    try:
        SPEC.loader.exec_module(sh)
        return
    except ImportError:
        pass
    saved = {k: sys.modules.get(k) for k in ("playwright", "playwright.sync_api")}
    stub = types.ModuleType("playwright.sync_api")
    stub.expect = stub.sync_playwright = None
    sys.modules["playwright"] = types.ModuleType("playwright")
    sys.modules["playwright.sync_api"] = stub
    try:
        SPEC.loader.exec_module(sh)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


_load()


def test_tokens_and_keys_in_the_page_text_are_hits():
    for text in ("hf_" + "A" * 36, "ghp_" + "b" * 40, "sk-" + "x" * 24, "AKIA" + "A" * 16, "-----BEGIN RSA PRIVATE KEY-----",
                 "Authorization: Basic abc", "api_key: x", "the password is x", "token=abc", 'token": "abc'):
        assert sh.credential_hits(text), text


def test_ordinary_page_text_is_clean():
    page = "RUNNING huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated stage 2 functional decoder the tokenizer has 248320 tokens operator accepted visible credentials"
    assert sh.credential_hits(page) == []


def test_loading_the_script_leaves_no_playwright_stand_in_behind():
    """The browser tests use importorskip('playwright.sync_api'); a leftover stand-in would let them run and fail."""
    mod = sys.modules.get("playwright.sync_api")
    assert mod is None or getattr(mod, "sync_playwright", None) is not None
