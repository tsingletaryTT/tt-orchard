# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
import pytest
from pathlib import Path

from orchard.tiers import TierConfigError, load

GOOD = """
[tiers.large]
role = "plan and diagnose"
endpoint = "http://127.0.0.1:8000/v1"
model = "big-model"
placement = "chips"
context_tokens = 262144

[tiers.small]
role = "routine steps"
endpoint = "http://localhost:8001/v1"
model = "small-model"
placement = "cpu"

[stages.0]
run = "large"
[stages.1]
run = "small"
[stages.2]
run = "small"
diagnose = "large"
[stages.3]
run = "small"
diagnose = "large"
[stages.4]
run = "small"
plan = "large"
diagnose = "large"
[stages.5]
run = "small"
[stages.6]
run = "small"
[stages.7]
run = "none"
[stages.8]
run = "small"

[escalation]
default = "large"
"""

EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "config" / "tiers.example.toml"


def write(tmp_path, text):
    p = tmp_path / "tiers.toml"
    p.write_text(text)
    return p


def test_good_config_loads(tmp_path):
    cfg = load(write(tmp_path, GOOD))
    assert set(cfg.tiers) == {"large", "small"}
    assert cfg.stages[2] == {"run": "small", "diagnose": "large"}
    assert cfg.stages[7] == {"run": "none"}


@pytest.mark.parametrize("old,new,why", [
    ('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "https://api.example.com/v1"', "remote"),
    ('model = "big-model"', 'model = "CHANGE-ME"', "sentinel"),
    ('placement = "chips"', 'placement = "gpu"', "placement"),
    ('role = "plan and diagnose"\n', '', "missing key"),
    ('[stages.8]\nrun = "small"\n', '', "stage missing"),
    ('[stages.7]\nrun = "none"', '[stages.7]\nrun = "small"', "stage 7 must be none"),
    ('[stages.5]\nrun = "small"', '[stages.5]\nrun = "none"', "only stage 7 may be none"),
    ('[stages.1]\nrun = "small"', '[stages.1]\nrun = "huge"', "unknown tier"),
    ('diagnose = "large"\n[stages.3]', 'diagnose = "nobody"\n[stages.3]', "unknown diagnose tier"),
    ('context_tokens = 262144', 'context_tokens = 0', "context must be positive"),
])
def test_bad_configs_are_refused(tmp_path, old, new, why):
    assert old in GOOD, why  # the edit must hit something, or the case proves nothing
    match_patterns = {
        "remote": "endpoint.*not on this machine",
        "sentinel": "CHANGE-ME",
        "placement": "placement",
        "missing key": "missing",
        "stage missing": "stage 8",
        "stage 7 must be none": "stage 7.*none",
        "only stage 7 may be none": "stage 5",
        "unknown tier": "run tier.*not defined",
        "unknown diagnose tier": "diagnose tier.*not defined",
        "context must be positive": "context_tokens",
    }
    with pytest.raises(TierConfigError, match=match_patterns.get(why, why)):
        load(write(tmp_path, GOOD.replace(old, new, 1)))


@pytest.mark.parametrize("line,key", [
    ('endpoint = "http://127.0.0.1:8000/v1"\n', "endpoint"),
    ('model = "big-model"\n', "model"),
    ('placement = "chips"\n', "placement"),
])
def test_missing_required_tier_key(tmp_path, line, key):
    """Each required tier key is checked on its own."""
    assert line in GOOD
    with pytest.raises(TierConfigError, match=f"is missing '{key}'"):
        load(write(tmp_path, GOOD.replace(line, '', 1)))


def test_stage_without_run_key_rejected(tmp_path):
    """A stage table with no run key gets the clear 'missing run' message."""
    text = GOOD.replace('[stages.1]\nrun = "small"', '[stages.1]\ndiagnose = "large"')
    with pytest.raises(TierConfigError, match="stage 1 is missing run"):
        load(write(tmp_path, text))


def test_a_cpu_tier_is_required(tmp_path):
    text = GOOD.replace('placement = "cpu"', 'placement = "chips"')
    with pytest.raises(TierConfigError, match="placement 'cpu'"):
        load(write(tmp_path, text))


def test_example_config_is_refused_until_edited():
    with pytest.raises(TierConfigError, match="CHANGE-ME"):
        load(EXAMPLE_PATH)


def test_example_config_loads_when_edited(tmp_path):
    """Example config loads once CHANGE-ME values are replaced."""
    text = EXAMPLE_PATH.read_text()
    text = text.replace("CHANGE-ME", "actual-model")
    cfg = load(write(tmp_path, text))
    assert "large" in cfg.tiers
    assert "small" in cfg.tiers
    assert cfg.stages[4]["plan"] == "large"
    assert cfg.escalation == "large"


# File access and parsing errors wrapped in TierConfigError
def test_missing_file_raises_tierconfigerror():
    path = "/nonexistent/tiers.toml"
    with pytest.raises(TierConfigError, match="cannot read") as excinfo:
        load(path)
    # Verify the path appears in the error message
    assert path in str(excinfo.value)


def test_invalid_toml_raises_tierconfigerror(tmp_path):
    """Invalid TOML syntax raises TierConfigError."""
    p = write(tmp_path, "[tiers.bad\n")  # unclosed bracket
    with pytest.raises(TierConfigError, match="parse"):
        load(p)


def test_non_utf8_file_raises_tierconfigerror(tmp_path):
    """Non-UTF-8 file raises TierConfigError."""
    p = tmp_path / "tiers.toml"
    p.write_bytes(b"\xff\xfe")  # Invalid UTF-8
    with pytest.raises(TierConfigError, match="cannot read"):
        load(p)


def test_read_error_message_names_the_path(tmp_path):
    """The rule, not the OS error, must name the file: a UnicodeDecodeError has no path."""
    p = tmp_path / "x.toml"
    p.write_bytes(b"\xff\xfe")
    with pytest.raises(TierConfigError) as excinfo:
        load(p)
    assert str(p) in str(excinfo.value)


def test_directory_path_raises_tierconfigerror(tmp_path):
    """Directory path raises TierConfigError (IsADirectoryError)."""
    with pytest.raises(TierConfigError, match="cannot read"):
        load(tmp_path)


def test_invalid_ipv6_url_raises_tierconfigerror(tmp_path):
    """Invalid IPv6 URL raises TierConfigError."""
    bad_endpoint = "http://[::1/v1"
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', f'endpoint = "{bad_endpoint}"')
    with pytest.raises(TierConfigError, match="endpoint.*not a valid URL") as excinfo:
        load(write(tmp_path, text))
    # Verify the endpoint string appears in the error message
    assert bad_endpoint in str(excinfo.value)
    assert "tier 'large'" in str(excinfo.value)


def test_stage_key_non_ascii_digit_rejected(tmp_path):
    """Non-ASCII digit in quoted stage key should be rejected."""
    # Arabic-Indic digit eight (٨)
    text = """
[tiers.large]
role = "r"
endpoint = "http://127.0.0.1:8000"
model = "m"
placement = "cpu"

[stages]
"٨" = { run = "large" }
"""
    with pytest.raises(TierConfigError, match="decimal integer"):
        load(write(tmp_path, text))


# Type validation: tiers and stages must be tables
def test_tiers_not_table_raises_error(tmp_path):
    """tiers must be a table, not a string."""
    text = 'tiers = "x"\n[stages.0]\nrun = "none"\n'
    with pytest.raises(TierConfigError, match="tiers"):
        load(write(tmp_path, text))


def test_tiers_not_table_empty_list_raises_error(tmp_path):
    """tiers as empty list raises TierConfigError."""
    text = 'tiers = []\n[stages.0]\nrun = "none"\n'
    with pytest.raises(TierConfigError, match="tiers"):
        load(write(tmp_path, text))


def test_stages_not_table_raises_error(tmp_path):
    """stages must be a table, not a string."""
    text = 'stages = "x"\n[tiers.t]\nrole = "r"\nendpoint = "http://127.0.0.1:8000"\nmodel = "m"\nplacement = "cpu"\n'
    with pytest.raises(TierConfigError, match="stages"):
        load(write(tmp_path, text))


def test_stages_not_table_empty_list_raises_error(tmp_path):
    """stages as empty list raises TierConfigError."""
    text = 'stages = []\n[tiers.t]\nrole = "r"\nendpoint = "http://127.0.0.1:8000"\nmodel = "m"\nplacement = "cpu"\n'
    with pytest.raises(TierConfigError, match="stages"):
        load(write(tmp_path, text))


def test_stage_value_not_table_raises_error(tmp_path):
    """Stage value must be a table (not a string or other type)."""
    # Create config where stages.0 is a string value instead of a table
    text = """
[tiers.large]
role = "r"
endpoint = "http://127.0.0.1:8000"
model = "m"
placement = "cpu"

[stages]
"0" = "small"
"1" = "small"
"2" = 1
"3" = "small"
"4" = "small"
"5" = "small"
"6" = "small"
"7" = "small"
"8" = "small"
"""
    with pytest.raises(TierConfigError, match="stage"):
        load(write(tmp_path, text))


def test_tier_value_not_table_raises_error(tmp_path):
    """Tier value must be a table."""
    text = """
tiers = { "small" = 123 }
[stages.0]
run = "small"
[stages.1]
run = "small"
[stages.2]
run = "small"
diagnose = "small"
[stages.3]
run = "small"
diagnose = "small"
[stages.4]
run = "small"
diagnose = "small"
[stages.5]
run = "small"
[stages.6]
run = "small"
[stages.7]
run = "none"
[stages.8]
run = "small"
"""
    with pytest.raises(TierConfigError, match="tier"):
        load(write(tmp_path, text))


# Stage key validation: must be decimal digits
def test_stage_key_with_leading_zeros_rejected(tmp_path):
    """Stage key like 01 (leading zeros) should be rejected."""
    text = GOOD.replace('[stages.0]', '[stages.01]')
    with pytest.raises(TierConfigError, match="stage"):
        load(write(tmp_path, text))


@pytest.mark.parametrize("old,new", [
    ('[stages.1]', '[stages.01]'),
    ('[stages.0]', '[stages.00]'),
])
def test_stage_key_leading_zeros_message(tmp_path, old, new):
    """The leading-zero rule itself must fire, not a later 'stage has no entry' error."""
    assert old in GOOD
    with pytest.raises(TierConfigError, match="no leading zeros"):
        load(write(tmp_path, GOOD.replace(old, new, 1)))


def test_stage_key_with_plus_sign_rejected(tmp_path):
    """Stage key like "+1" (quoted) should be rejected."""
    text = GOOD.replace('[stages.0]', '[stages]\n"+1" = { run = "large" }')
    with pytest.raises(TierConfigError, match="decimal integer"):
        load(write(tmp_path, text))


def test_stage_key_with_space_rejected(tmp_path):
    """Stage key like " 1" (quoted with leading space) should be rejected."""
    text = GOOD.replace('[stages.0]', '[stages]\n" 1" = { run = "large" }')
    with pytest.raises(TierConfigError, match="decimal integer"):
        load(write(tmp_path, text))


def test_stage_key_non_numeric_rejected(tmp_path):
    """Stage key like 'eight' should be rejected."""
    text = GOOD.replace('[stages.0]', '[stages.eight]')
    with pytest.raises(TierConfigError, match="stage"):
        load(write(tmp_path, text))


def test_stage_number_out_of_range_high(tmp_path):
    """Stage numbers outside 0-8 are out of range."""
    # Add stages 9 and 99 in addition to GOOD (do not replace stage 8)
    text = GOOD + '\n[stages.9]\nrun = "small"\n'
    with pytest.raises(TierConfigError, match="stage 9.*0-8"):
        load(write(tmp_path, text))


def test_stage_number_99_out_of_range(tmp_path):
    """Stage 99 is out of range (0-8 only)."""
    text = GOOD + '\n[stages.99]\nrun = "small"\n'
    with pytest.raises(TierConfigError, match="stage 99.*0-8"):
        load(write(tmp_path, text))


def test_stage_number_negative_quoted_rejected(tmp_path):
    """Negative stage number "-1" (quoted) should be rejected."""
    text = GOOD + '\n[stages]\n"-1" = { run = "small" }\n'
    with pytest.raises(TierConfigError, match="decimal integer"):
        load(write(tmp_path, text))


# Sentinel validation: case-insensitive and substring
def test_sentinel_lowercase_rejected(tmp_path):
    """Sentinel 'change-me' (lowercase) should be rejected."""
    text = GOOD.replace('model = "big-model"', 'model = "change-me"')
    with pytest.raises(TierConfigError, match="CHANGE-ME"):
        load(write(tmp_path, text))


def test_sentinel_uppercase_rejected(tmp_path):
    """Sentinel 'CHANGE-ME' should be rejected."""
    text = GOOD.replace('model = "big-model"', 'model = "CHANGE-ME"')
    with pytest.raises(TierConfigError, match="CHANGE-ME"):
        load(write(tmp_path, text))


def test_sentinel_mixed_case_rejected(tmp_path):
    """Sentinel 'Change-Me' (mixed case) should be rejected."""
    text = GOOD.replace('model = "big-model"', 'model = "Change-Me"')
    with pytest.raises(TierConfigError, match="CHANGE-ME"):
        load(write(tmp_path, text))


def test_sentinel_substring_rejected(tmp_path):
    """Value containing 'CHANGE-ME' should be rejected."""
    text = GOOD.replace('model = "big-model"', 'model = "prefix-CHANGE-ME-suffix"')
    with pytest.raises(TierConfigError, match="CHANGE-ME"):
        load(write(tmp_path, text))


def test_sentinel_in_role_rejected(tmp_path):
    """Sentinel in role field should be rejected."""
    text = GOOD.replace('role = "plan and diagnose"', 'role = "CHANGE-ME"')
    with pytest.raises(TierConfigError, match="CHANGE-ME"):
        load(write(tmp_path, text))


def test_sentinel_in_endpoint_rejected(tmp_path):
    """Sentinel in endpoint field should be rejected."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "CHANGE-ME"')
    with pytest.raises(TierConfigError, match="CHANGE-ME"):
        load(write(tmp_path, text))


# Value type and format validation
def test_empty_string_role_rejected(tmp_path):
    """Empty role string should be rejected."""
    text = GOOD.replace('role = "plan and diagnose"', 'role = ""')
    with pytest.raises(TierConfigError, match="role"):
        load(write(tmp_path, text))


def test_empty_string_model_rejected(tmp_path):
    """Empty model string should be rejected."""
    text = GOOD.replace('model = "big-model"', 'model = ""')
    with pytest.raises(TierConfigError, match="model"):
        load(write(tmp_path, text))


def test_empty_string_endpoint_rejected(tmp_path):
    """Empty endpoint string should be rejected."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = ""')
    with pytest.raises(TierConfigError, match="non-empty string"):
        load(write(tmp_path, text))


def test_whitespace_only_role_rejected(tmp_path):
    """Whitespace-only role should be rejected."""
    text = GOOD.replace('role = "plan and diagnose"', 'role = "   "')
    with pytest.raises(TierConfigError, match="role"):
        load(write(tmp_path, text))


def test_whitespace_only_model_rejected(tmp_path):
    """Whitespace-only model should be rejected."""
    text = GOOD.replace('model = "big-model"', 'model = "   "')
    with pytest.raises(TierConfigError, match="model"):
        load(write(tmp_path, text))


def test_whitespace_only_endpoint_rejected(tmp_path):
    """Whitespace-only endpoint should be rejected."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "   "')
    with pytest.raises(TierConfigError, match="non-empty string"):
        load(write(tmp_path, text))


def test_non_string_model_rejected(tmp_path):
    """Non-string model should be rejected."""
    text = GOOD.replace('model = "big-model"', 'model = 123')
    with pytest.raises(TierConfigError, match="model"):
        load(write(tmp_path, text))


def test_non_string_role_rejected(tmp_path):
    """Non-string role should be rejected."""
    text = GOOD.replace('role = "plan and diagnose"', 'role = 123')
    with pytest.raises(TierConfigError, match="role"):
        load(write(tmp_path, text))


def test_non_string_endpoint_rejected(tmp_path):
    """Non-string endpoint should be rejected."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = 123')
    with pytest.raises(TierConfigError, match="endpoint"):
        load(write(tmp_path, text))


def test_context_tokens_bool_rejected(tmp_path):
    """context_tokens = true (bool) should be rejected."""
    text = GOOD.replace('context_tokens = 262144', 'context_tokens = true')
    with pytest.raises(TierConfigError, match="context_tokens"):
        load(write(tmp_path, text))


def test_context_tokens_negative_rejected(tmp_path):
    """Negative context_tokens should be rejected."""
    text = GOOD.replace('context_tokens = 262144', 'context_tokens = -1')
    with pytest.raises(TierConfigError, match="context_tokens"):
        load(write(tmp_path, text))


def test_context_tokens_string_rejected(tmp_path):
    """String context_tokens should be rejected."""
    text = GOOD.replace('context_tokens = 262144', 'context_tokens = "262144"')
    with pytest.raises(TierConfigError, match="context_tokens"):
        load(write(tmp_path, text))


def test_placement_list_rejected(tmp_path):
    """placement as a list should be rejected."""
    text = GOOD.replace('placement = "chips"', 'placement = ["chips"]')
    with pytest.raises(TierConfigError, match="placement"):
        load(write(tmp_path, text))


def test_run_list_rejected(tmp_path):
    """run as a list should be rejected."""
    text = GOOD.replace('[stages.0]\nrun = "large"', '[stages.0]\nrun = ["large"]')
    with pytest.raises(TierConfigError, match="run"):
        load(write(tmp_path, text))


def test_diagnose_list_rejected(tmp_path):
    """diagnose as a list should be rejected."""
    text = GOOD.replace('[stages.2]\nrun = "small"\ndiagnose = "large"', '[stages.2]\nrun = "small"\ndiagnose = ["large"]')
    with pytest.raises(TierConfigError, match="diagnose"):
        load(write(tmp_path, text))


# Endpoint validation
def test_endpoint_ftp_scheme_rejected(tmp_path):
    """FTP endpoints should be rejected."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "ftp://127.0.0.1:8000/v1"')
    with pytest.raises(TierConfigError, match="scheme must be http or https") as excinfo:
        load(write(tmp_path, text))
    # Ensure this is NOT a re-wrapped error
    assert "not a valid URL" not in str(excinfo.value)


def test_endpoint_https_accepted(tmp_path):
    """HTTPS endpoints are accepted."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "https://127.0.0.1:8000/v1"')
    cfg = load(write(tmp_path, text))
    assert "large" in cfg.tiers


def test_endpoint_userinfo_trick_rejected(tmp_path):
    """Userinfo trick like http://127.0.0.1@evil.example/ should be rejected."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "http://127.0.0.1@evil.example/v1"')
    with pytest.raises(TierConfigError, match="not on this machine") as excinfo:
        load(write(tmp_path, text))
    # Ensure this is NOT a re-wrapped error
    assert "not a valid URL" not in str(excinfo.value)


def test_endpoint_subdomain_trick_rejected(tmp_path):
    """Subdomain trick like localhost.evil.example should be rejected."""
    text = GOOD.replace('endpoint = "http://localhost:8001/v1"', 'endpoint = "http://localhost.evil.example/v1"')
    with pytest.raises(TierConfigError, match="not on this machine"):
        load(write(tmp_path, text))


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.5"])
def test_endpoint_non_loopback_address_rejected(tmp_path, host):
    """0.0.0.0 and LAN addresses are not loopback, so a loosened host list must be caught."""
    text = GOOD.replace('http://127.0.0.1:8000/v1', f'http://{host}:8000/v1')
    with pytest.raises(TierConfigError, match="not on this machine"):
        load(write(tmp_path, text))


def test_endpoint_ipv6_loopback_accepted(tmp_path):
    """IPv6 loopback [::1] is accepted."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "http://[::1]:8000/v1"')
    cfg = load(write(tmp_path, text))
    assert "large" in cfg.tiers


def test_endpoint_no_scheme_rejected(tmp_path):
    """Endpoint without scheme should be rejected."""
    text = GOOD.replace('endpoint = "http://127.0.0.1:8000/v1"', 'endpoint = "127.0.0.1:8000/v1"')
    with pytest.raises(TierConfigError, match="must have a scheme") as excinfo:
        load(write(tmp_path, text))
    # Ensure this is NOT a re-wrapped error
    assert "not a valid URL" not in str(excinfo.value)


# Stage-specific rules
def test_stage_2_requires_diagnose(tmp_path):
    """Stage 2 must have diagnose (small runs, large diagnoses)."""
    text = GOOD.replace('[stages.2]\nrun = "small"\ndiagnose = "large"', '[stages.2]\nrun = "small"')
    with pytest.raises(TierConfigError, match="stage 2"):
        load(write(tmp_path, text))


def test_stage_3_requires_diagnose(tmp_path):
    """Stage 3 must have diagnose (small runs, large diagnoses)."""
    text = GOOD.replace('[stages.3]\nrun = "small"\ndiagnose = "large"', '[stages.3]\nrun = "small"')
    with pytest.raises(TierConfigError, match="stage 3"):
        load(write(tmp_path, text))


def test_stage_4_requires_diagnose(tmp_path):
    """Stage 4 must have diagnose (small runs, large diagnoses)."""
    text = GOOD.replace('[stages.4]\nrun = "small"\nplan = "large"\ndiagnose = "large"', '[stages.4]\nrun = "small"\nplan = "large"')
    with pytest.raises(TierConfigError, match="stage 4"):
        load(write(tmp_path, text))


def test_stage_7_forbids_diagnose(tmp_path):
    """Stage 7 must not have diagnose (no model is loaded)."""
    text = GOOD.replace('[stages.7]\nrun = "none"', '[stages.7]\nrun = "none"\ndiagnose = "large"')
    with pytest.raises(TierConfigError, match="stage 7"):
        load(write(tmp_path, text))


# Tier naming
def test_tier_cannot_be_named_none(tmp_path):
    """A tier cannot be named 'none' (reserved word)."""
    text = GOOD.replace('[tiers.large]', '[tiers.none]')
    with pytest.raises(TierConfigError, match="none"):
        load(write(tmp_path, text))


# ---- final review: unknown keys, stage 4 `plan`, and the [escalation] table ----

def test_stage_4_requires_plan(tmp_path):
    """Spec section 5: stage 4 is 'large plans, small runs', so it must name its planning tier."""
    text = GOOD.replace('plan = "large"\n', '', 1)
    with pytest.raises(TierConfigError, match="stage 4 must have a plan tier"):
        load(write(tmp_path, text))


def test_plan_is_allowed_only_on_stage_4(tmp_path):
    text = GOOD.replace('[stages.3]\nrun = "small"', '[stages.3]\nrun = "small"\nplan = "large"')
    with pytest.raises(TierConfigError, match="stage 3 must not have a plan tier"):
        load(write(tmp_path, text))


def test_stage_7_forbids_plan(tmp_path):
    text = GOOD.replace('[stages.7]\nrun = "none"', '[stages.7]\nrun = "none"\nplan = "large"')
    with pytest.raises(TierConfigError, match="stage 7 must not have a plan tier"):
        load(write(tmp_path, text))


def test_plan_must_name_a_defined_tier(tmp_path):
    text = GOOD.replace('plan = "large"', 'plan = "nobody"')
    with pytest.raises(TierConfigError, match="stage 4: plan tier 'nobody' is not defined"):
        load(write(tmp_path, text))


def test_plan_must_be_a_string(tmp_path):
    text = GOOD.replace('plan = "large"', 'plan = 3')
    with pytest.raises(TierConfigError, match="stage 4 plan must be a string"):
        load(write(tmp_path, text))


def test_escalation_table_is_required(tmp_path):
    text = GOOD.replace('[escalation]\ndefault = "large"\n', '')
    with pytest.raises(TierConfigError, match=r"\[escalation\] table is required"):
        load(write(tmp_path, text))


def test_escalation_default_must_name_a_defined_tier(tmp_path):
    text = GOOD.replace('default = "large"', 'default = "nobody"')
    with pytest.raises(TierConfigError, match=r"\[escalation\] default tier 'nobody' is not defined"):
        load(write(tmp_path, text))


def test_escalation_needs_a_default(tmp_path):
    text = GOOD.replace('default = "large"\n', '')
    with pytest.raises(TierConfigError, match=r"\[escalation\] is missing 'default'"):
        load(write(tmp_path, text))


def test_escalation_default_must_be_a_string(tmp_path):
    text = GOOD.replace('default = "large"', 'default = 1')
    with pytest.raises(TierConfigError, match=r"\[escalation\] default must be a string"):
        load(write(tmp_path, text))


def test_escalation_must_be_a_table(tmp_path):
    text = GOOD.replace('[escalation]\ndefault = "large"\n', '')
    with pytest.raises(TierConfigError, match=r"\[escalation\] must be a table"):
        load(write(tmp_path, 'escalation = "large"\n' + text))


def test_escalation_refuses_unknown_keys(tmp_path):
    text = GOOD.replace('default = "large"', 'default = "large"\ndefualt = "small"')
    with pytest.raises(TierConfigError, match=r"\[escalation\] has unknown key 'defualt'"):
        load(write(tmp_path, text))


@pytest.mark.parametrize("old,new,needle", [
    ('role = "plan and diagnose"', 'role = "plan and diagnose"\ncontext_token = 1',
     "tier 'large' has unknown key 'context_token'; did you mean 'context_tokens'"),
    ('role = "plan and diagnose"', 'role = "plan and diagnose"\nzzz = 1',
     "tier 'large' has unknown key 'zzz'; allowed keys are"),
    ('[stages.2]\nrun = "small"\ndiagnose = "large"', '[stages.2]\nrun = "small"\ndiagnos = "large"',
     "stage 2 has unknown key 'diagnos'; did you mean 'diagnose'"),
    ('[stages.1]\nrun = "small"', '[stages.1]\nrun = "small"\nzzz = 1',
     "stage 1 has unknown key 'zzz'; allowed keys are"),
])
def test_unknown_tier_and_stage_keys_are_refused(tmp_path, old, new, needle):
    assert old in GOOD
    with pytest.raises(TierConfigError, match=needle):
        load(write(tmp_path, GOOD.replace(old, new, 1)))


def test_unknown_top_level_table_is_refused(tmp_path):
    with pytest.raises(TierConfigError, match="the config has unknown top-level key 'escalations'; did you mean 'escalation'"):
        load(write(tmp_path, GOOD + '\n[escalations]\ndefault = "large"\n'))
    with pytest.raises(TierConfigError, match="the config has unknown top-level key 'extra'; allowed keys are"):
        load(write(tmp_path, GOOD + '\n[extra]\nx = 1\n'))
