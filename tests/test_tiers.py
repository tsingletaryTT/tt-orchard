import pytest

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
diagnose = "large"
[stages.5]
run = "small"
[stages.6]
run = "small"
[stages.7]
run = "none"
[stages.8]
run = "small"
"""


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
    with pytest.raises(TierConfigError):
        load(write(tmp_path, GOOD.replace(old, new, 1)))


def test_a_cpu_tier_is_required(tmp_path):
    text = GOOD.replace('placement = "cpu"', 'placement = "chips"')
    with pytest.raises(TierConfigError):
        load(write(tmp_path, text))


def test_example_config_is_refused_until_edited():
    with pytest.raises(TierConfigError):
        load("config/tiers.example.toml")
