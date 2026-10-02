"""Tier config: which local model fills each role, and which tier runs each stage.

The file is local to the machine and never committed. It names models, and the choice of models
depends on measurements (see sizing.py). The validator refuses anything that would send a run
somewhere unintended: a remote endpoint, an unedited example, or a stage nobody owns.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from urllib.parse import urlparse

STAGES = range(9)                 # stages 0..8 in the spec
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
PLACEMENTS = {"chips", "cpu"}
REQUIRED_TIER_KEYS = ("role", "endpoint", "model", "placement")
SENTINEL = "CHANGE-ME"            # example config values; a run must not start on them
NO_MODEL_STAGE = 7                # image build: the supervisor waits and no model is loaded


class TierConfigError(ValueError):
    pass


@dataclass
class TierConfig:
    tiers: dict
    stages: dict


def load(path) -> TierConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    tiers = raw.get("tiers", {})
    stages = {int(k): v for k, v in raw.get("stages", {}).items()}

    for name, tier in tiers.items():
        for key in REQUIRED_TIER_KEYS:
            if key not in tier:
                raise TierConfigError(f"tier {name!r} is missing {key!r}")
            if tier[key] == SENTINEL:
                raise TierConfigError(f"tier {name!r} still has {SENTINEL} in {key!r}")
        if tier["placement"] not in PLACEMENTS:
            raise TierConfigError(f"tier {name!r}: placement must be one of {sorted(PLACEMENTS)}")
        host = urlparse(tier["endpoint"]).hostname
        if host not in LOCAL_HOSTS:
            raise TierConfigError(
                f"tier {name!r}: endpoint host {host!r} is not on this machine; runs use local models only")
        if "context_tokens" in tier and not (isinstance(tier["context_tokens"], int) and tier["context_tokens"] > 0):
            raise TierConfigError(f"tier {name!r}: context_tokens must be a positive integer")

    if not any(t["placement"] == "cpu" for t in tiers.values()):
        raise TierConfigError("a tier with placement 'cpu' is required (the stand-in during parking)")

    for n in STAGES:
        if n not in stages:
            raise TierConfigError(f"stage {n} has no entry")
        stage = stages[n]
        run = stage.get("run")
        if n == NO_MODEL_STAGE:
            if run != "none":
                raise TierConfigError(f"stage {n} must have run = 'none'")
        elif run not in tiers:
            raise TierConfigError(f"stage {n}: run tier {run!r} is not defined")
        if "diagnose" in stage and stage["diagnose"] not in tiers:
            raise TierConfigError(f"stage {n}: diagnose tier {stage['diagnose']!r} is not defined")

    return TierConfig(tiers=tiers, stages=stages)
