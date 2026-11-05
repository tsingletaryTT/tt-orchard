# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 458: a tier may set the coder's reply cap and sampling, and the retry profile used after a capped reply."""
import pytest

from orchard.tiers import TierConfigError, load
from test_tiers import GOOD


def tiers_with(tmp_path, line: str):
    text = GOOD.replace('model = "big-model"\n', f'model = "big-model"\n{line}\n', 1)
    p = tmp_path / "tiers.toml"
    p.write_text(text)
    return load(p)


ACCEPTED = ["max_tokens = 8192", "temperature = 0.2", "frequency_penalty = 0.1", "repetition_penalty = 1.05",
            "unstick_max_tokens = 4096", "unstick_temperature = 0.7", "unstick_frequency_penalty = 0.4",
            "unstick_repetition_penalty = 1.1", "temperature = 0", "frequency_penalty = -0.5"]


@pytest.mark.parametrize("line", ACCEPTED)
def test_each_sampling_key_is_accepted_and_kept(tmp_path, line):
    cfg = tiers_with(tmp_path, line)
    key, value = [s.strip() for s in line.split("=")]
    expected = int(value) if key.endswith("max_tokens") else float(value)
    assert cfg.tiers["large"][key] == expected


REFUSED = [("max_tokens = true", "max_tokens"), ("max_tokens = 100", "max_tokens"), ("max_tokens = 999999999", "max_tokens"),
           ("max_tokens = 8192.5", "max_tokens"), ('max_tokens = "8192"', "max_tokens"),
           ("temperature = 3", "temperature"), ("temperature = -0.1", "temperature"), ("temperature = true", "temperature"),
           ('temperature = "hot"', "temperature"), ("frequency_penalty = 5", "frequency_penalty"),
           ("repetition_penalty = 0.1", "repetition_penalty"), ("repetition_penalty = 9", "repetition_penalty"),
           ("unstick_max_tokens = 5", "unstick_max_tokens"), ("unstick_temperature = 2.5", "unstick_temperature"),
           ("unstick_frequency_penalty = true", "unstick_frequency_penalty"), ("unstick_repetition_penalty = 0", "unstick_repetition_penalty")]


@pytest.mark.parametrize("line,key", REFUSED)
def test_a_bad_sampling_value_is_refused_and_the_message_names_the_key(tmp_path, line, key):
    with pytest.raises(TierConfigError) as exc:
        tiers_with(tmp_path, line)
    message = str(exc.value)
    assert key in message and "'large'" in message and "must be" in message and "unknown" not in message.lower()


def test_an_unknown_sampling_key_is_still_refused_with_a_suggestion(tmp_path):
    with pytest.raises(TierConfigError) as exc:
        tiers_with(tmp_path, "max_token = 8192")
    assert "max_token" in str(exc.value)


def test_a_tier_without_the_keys_loads_as_before(tmp_path):
    p = tmp_path / "tiers.toml"
    p.write_text(GOOD)
    cfg = load(p)
    assert not any(k in cfg.tiers["large"] for k in ("max_tokens", "temperature", "unstick_temperature"))


# ---- the coder's sampling by default: top_p and top_k are tier keys too (a paired test showed greedy decoding wrote no port) ----

@pytest.mark.parametrize("line,key,expected", [("top_p = 0.8", "top_p", 0.8), ("top_p = 1", "top_p", 1.0), ("top_p = 0", "top_p", 0.0),
                                               ("top_k = 20", "top_k", 20), ("top_k = 0", "top_k", 0), ("top_k = 200", "top_k", 200)])
def test_top_p_and_top_k_are_accepted_and_kept(tmp_path, line, key, expected):
    value = tiers_with(tmp_path, line).tiers["large"][key]
    assert value == expected and (isinstance(value, int) if key == "top_k" else isinstance(value, (int, float)))


REFUSED_TOP = [("top_p = 1.5", "top_p"), ("top_p = -0.1", "top_p"), ("top_p = true", "top_p"), ('top_p = "high"', "top_p"),
               ("top_k = 201", "top_k"), ("top_k = -1", "top_k"), ("top_k = 20.5", "top_k"), ("top_k = true", "top_k")]


@pytest.mark.parametrize("line,key", REFUSED_TOP)
def test_a_bad_top_p_or_top_k_is_refused_and_the_message_names_the_key(tmp_path, line, key):
    with pytest.raises(TierConfigError) as exc:
        tiers_with(tmp_path, line)
    message = str(exc.value)
    assert key in message and "'large'" in message and "must be" in message
